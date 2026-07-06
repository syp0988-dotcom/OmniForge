"""API routes for chat and knowledge base management."""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from agentflow.agents.registry import get_all as get_all_agents
from agentflow.database.sqlite import SQLiteStore
from agentflow.graph.workflow import build_workflow, run_workflow
from agentflow.knowledge.store import KnowledgeStore
from agentflow.models.chat import ChatRequest, ChatResponse, FileProposal
from agentflow.models.model_config import LLMModelCreate, LLMModelUpdate
from agentflow.services.file_proposer import propose_files
from agentflow.utils.logging import build_logger

router = APIRouter()
logger = build_logger("api")
store = SQLiteStore()
knowledge_store = KnowledgeStore(db=store)

# Directory for uploaded document files
UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# -- Agent introspection ---------------------------------------------------


@router.get("/agents")
def list_agents() -> list[dict[str, object]]:
    """List all registered agents with metadata (name, key, status, capabilities)."""
    return get_all_agents()


# -- Chat -------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Handle chat requests through the workflow."""
    try:
        # Ensure session exists
        session_id = request.session_id
        if session_id is None:
            # Create a new session for anonymous chats
            sess = store.create_session()
            session_id = sess["id"]
        else:
            existing = store.get_session(session_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="Session not found")

        workflow = build_workflow()
        history_dicts = [
            {"role": m.role, "content": m.content} for m in request.history
        ]

        # Load session_state from DB (Conversation Runtime)
        saved_state_str = store.get_session_state(session_id)
        session_state = json.loads(saved_state_str) if saved_state_str else None

        result = run_workflow(
            workflow,
            request.message,
            history=history_dicts,
            session_state=session_state,
        )

        # Persist session_state after the turn
        new_state = result.get("session_state")
        if new_state and isinstance(new_state, dict):
            store.update_session_state(session_id, json.dumps(new_state, ensure_ascii=False))

        store.add_message("user", request.message, session_id=session_id)
        store.add_message("assistant", result["answer"], session_id=session_id)

        # Auto-title: use first user message as session title
        sess = store.get_session(session_id)
        if sess and sess["title"] == "新对话":
            title = request.message[:50]
            if len(request.message) > 50:
                title += "…"
            store.update_session_title(session_id, title)

        debug_data = {
            "category": result.get("category"),
            "workflow": result.get("workflow"),
            "search_results": result.get("search_results", []),
            "router": result.get("router", {}),
        }
        return ChatResponse(
            reply=result["answer"],
            metadata={"status": "ok", "session_id": session_id},
            debug=debug_data,
            proposed_files=propose_files(result["answer"]),
        )
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming endpoint — yields events as workflow nodes complete.

    Events::

        event: thinking\\n data: {{"phase": "分析问题", "category": "search"}}
        event: searching\\n data: {{"phase": "搜索网络信息"}}
        event: generating\\n data: {{"phase": "生成回答"}}
        event: done\\n data: {{"answer": "..."}}
    """
    from agentflow.conversation.session_state import SessionState
    from agentflow.graph.context import WorkflowContext

    async def _event_generator():
        # -- Session setup --
        session_id = request.session_id
        if session_id is None:
            sess = store.create_session()
            session_id = sess["id"]
        else:
            existing = store.get_session(session_id)
            if existing is None:
                yield f"event: error\\ndata: {json.dumps({'error': 'Session not found'})}\\n\\n"
                return

        history_dicts = [
            {"role": m.role, "content": m.content} for m in request.history
        ]

        # Load session_state from DB
        saved_state_str = store.get_session_state(session_id)
        session_state_dict = json.loads(saved_state_str) if saved_state_str else None

        workflow = build_workflow()

        initial_state: dict = {
            "question": request.message,
            "workflow": [],
            "history": history_dicts,
        }
        if history_dicts:
            initial_state["memory"] = {"history": list(history_dicts)}
        if session_state_dict:
            initial_state["session_state"] = SessionState.from_dict(session_state_dict)

        # -- Stream workflow execution --
        final_state: dict | None = None
        try:
            async for event in workflow.astream(initial_state):
                for node_name, state_update in event.items():
                    if node_name == "router":
                        category = state_update.get("category", "")
                        yield _sse_event("thinking", {"phase": "分析问题", "category": category})
                    elif node_name == "planner":
                        yield _sse_event("planning", {"phase": "制定执行计划"})
                    elif node_name == "knowledge":
                        yield _sse_event("searching", {"phase": "检索知识库"})
                    elif node_name == "search":
                        yield _sse_event("searching", {"phase": "搜索网络信息"})
                    elif node_name == "python":
                        yield _sse_event("executing", {"phase": "执行代码"})
                    elif node_name == "tool_executor":
                        yield _sse_event("executing", {"phase": "执行文件系统/工具操作"})
                    elif node_name == "answer":
                        yield _sse_event("generating", {"phase": "生成回答"})
                    elif node_name == "memory":
                        final_state = dict(state_update)
        except Exception as exc:
            logger.exception("Streaming workflow failed")
            yield _sse_event("error", {"error": str(exc)})
            return

        # -- Extract final answer --
        answer = final_state.get("answer", "") if final_state else ""

        # -- Persist messages --
        store.add_message("user", request.message, session_id=session_id)
        if answer:
            store.add_message("assistant", answer, session_id=session_id)

        # -- Persist session_state --
        if final_state:
            ctx = WorkflowContext(final_state)
            result_dict = ctx.to_dict()
            new_state = result_dict.get("session_state")
            if new_state and isinstance(new_state, dict):
                store.update_session_state(session_id, json.dumps(new_state, ensure_ascii=False))

        # -- Auto-title --
        sess = store.get_session(session_id)
        if sess and sess["title"] == "新对话":
            title = request.message[:50]
            if len(request.message) > 50:
                title += "…"
            store.update_session_title(session_id, title)

        yield _sse_event("done", {"answer": answer})

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


def _sse_event(event: str, data: dict) -> str:
    """Format an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# -- Knowledge base: document management ------------------------------------


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a document, parse it, and index it into the knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file type
    allowed_types = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_types)}",
        )

    # Save uploaded file temporarily
    temp_path = UPLOAD_DIR / file.filename
    try:
        content = await file.read()
        temp_path.write_bytes(content)
        logger.info("Saved uploaded file: %s (%d bytes)", file.filename, len(content))

        # Ingest into knowledge base
        doc_id = knowledge_store.add_document(temp_path, file.filename)
        return JSONResponse(
            content={
                "status": "ok",
                "document_id": doc_id,
                "filename": file.filename,
                "size": len(content),
            }
        )
    except Exception as exc:
        logger.exception("Upload ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # Clean up temp file after indexing
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.get("/knowledge/documents")
def list_documents() -> list[dict[str, object]]:
    """List all indexed documents."""
    return knowledge_store.list_documents()


@router.delete("/knowledge/documents/{doc_id}")
def delete_document(doc_id: int) -> JSONResponse:
    """Delete a document and its chunks/embeddings from the knowledge base."""
    knowledge_store.delete_document(doc_id)
    return JSONResponse(content={"status": "deleted", "document_id": doc_id})


@router.post("/knowledge/search")
def search_knowledge(query: str, top_k: int = 5) -> list[dict[str, object]]:
    """Search the knowledge base for relevant chunks."""
    return knowledge_store.search(query, top_k=top_k)


# -- Chat history ----------------------------------------------------------


@router.get("/history")
def history(limit: int = 20) -> list[dict[str, str]]:
    """Fetch recent chat history."""
    return store.list_messages(limit=limit)


# -- File operations (agent-generated files) --------------------------------


class CreateFileRequest(BaseModel):
    filename: str
    content: str
    workspace_path: str | None = None


@router.post("/files/create")
def create_file(req: CreateFileRequest) -> JSONResponse:
    """Write a proposed file to the outputs/ or workspace directory."""
    safe_name = Path(req.filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if req.workspace_path:
        target = Path(req.workspace_path).resolve() / safe_name
        # Prevent escaping outside the workspace
        if not str(target).startswith(str(Path(req.workspace_path).resolve())):
            raise HTTPException(status_code=400, detail="Invalid path")
    else:
        target = OUTPUT_DIR / safe_name

    if target.exists():
        raise HTTPException(status_code=409, detail="File already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content, encoding="utf-8")
    return JSONResponse(
        content={
            "status": "created",
            "filename": safe_name,
            "path": str(target),
        }
    )


@router.get("/files")
def list_output_files(workspace_path: str | None = None) -> list[dict[str, object]]:
    """List files in the outputs/ directory or a workspace path."""
    if workspace_path:
        base = Path(workspace_path).resolve()
        if not base.exists() or not base.is_dir():
            raise HTTPException(status_code=400, detail="Invalid workspace path")
    else:
        base = OUTPUT_DIR

    if not base.exists():
        return []
    files: list[dict[str, object]] = []
    for p in sorted(base.iterdir()):
        if p.is_file():
            files.append(
                {
                    "filename": p.name,
                    "size": p.stat().st_size,
                    "created_at": datetime.fromtimestamp(p.stat().st_ctime).isoformat(),
                    "path": str(p),
                }
            )
    return files


# -- Workspace operations ----------------------------------------------------


@router.get("/workspace")
def get_workspace() -> JSONResponse:
    """Check if a path is a valid workspace directory."""
    return JSONResponse(content={"status": "ok", "message": "Provide a path via POST /workspace/set"})


class SetWorkspaceRequest(BaseModel):
    path: str


@router.post("/workspace/set")
def set_workspace(req: SetWorkspaceRequest) -> JSONResponse:
    """Validate and set workspace folder path."""
    p = Path(req.path).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {req.path}")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    # Test write permission
    test_file = p / ".omni_forge_write_test"
    try:
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
    except OSError as exc:
        raise HTTPException(status_code=403, detail=f"No write permission: {exc}")
    return JSONResponse(content={"status": "ok", "path": str(p)})


class CreateFolderRequest(BaseModel):
    parent_path: str
    folder_name: str


@router.post("/workspace/create-folder")
def create_workspace_folder(req: CreateFolderRequest) -> JSONResponse:
    """Create a new folder under the given parent path."""
    parent = Path(req.parent_path).resolve()
    if not parent.exists() or not parent.is_dir():
        raise HTTPException(status_code=404, detail=f"Parent path does not exist: {req.parent_path}")
    safe_name = Path(req.folder_name).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid folder name")
    target = parent / safe_name
    if target.exists():
        raise HTTPException(status_code=409, detail=f"Folder already exists: {safe_name}")
    try:
        target.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create folder: {exc}")
    return JSONResponse(content={"status": "created", "path": str(target)})


@router.get("/workspace/browse")
def browse_directory(path: str = ".") -> JSONResponse:
    """List directories and files at the given path for folder browsing."""
    base = Path(path).resolve()
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {path}")
    entries: list[dict[str, object]] = []
    for p in sorted(base.iterdir()):
        entries.append({
            "name": p.name,
            "is_dir": p.is_dir(),
            "path": str(p),
        })
    return JSONResponse(content={"current_path": str(base), "entries": entries})


# -- Sessions ----------------------------------------------------------------


@router.post("/sessions/create")
def create_session() -> JSONResponse:
    """Create a new chat session."""
    sess = store.create_session()
    return JSONResponse(content=sess)


@router.get("/sessions")
def list_sessions(limit: int = 50) -> list[dict[str, object]]:
    """List all chat sessions, most recent first."""
    return store.list_sessions(limit=limit)


@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: int) -> list[dict[str, object]]:
    """Get all messages for a session."""
    sess = store.get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return store.get_session_messages(session_id)


@router.put("/sessions/{session_id}/rename")
def rename_session(session_id: int, body: dict[str, str]) -> JSONResponse:
    """Rename a session."""
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    ok = store.update_session_title(session_id, title)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(content={"status": "ok"})


@router.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: int) -> JSONResponse:
    """Delete a session and all its messages."""
    ok = store.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(content={"status": "deleted"})


# -- Model configuration ---------------------------------------------------


@router.get("/models")
def list_models() -> list[dict[str, object]]:
    """List all configured LLM models (API key excluded in response)."""
    from agentflow.models.model_config import LLMModelResponse
    rows = store.get_all_models()
    return [LLMModelResponse.from_db_row(r).model_dump() for r in rows]


@router.post("/models", status_code=201)
def create_model(config: LLMModelCreate) -> dict[str, object]:
    """Create a new LLM model configuration."""
    model_id = store.add_model(
        name=config.name,
        provider=config.provider,
        base_url=config.base_url,
        api_key=config.api_key,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return {"id": model_id, "status": "created"}


@router.put("/models/{model_id}")
def update_model(model_id: int, config: LLMModelUpdate) -> dict[str, object]:
    """Update an existing LLM model configuration."""
    ok = store.update_model(model_id, **config.model_dump(exclude_unset=True))
    if not ok:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "updated"}


@router.delete("/models/{model_id}")
def delete_model(model_id: int) -> dict[str, object]:
    """Delete an LLM model configuration."""
    ok = store.delete_model(model_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted"}


@router.post("/models/{model_id}/activate")
def activate_model(model_id: int) -> dict[str, object]:
    """Set a model as the active LLM configuration."""
    model = store.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    store.set_active_model(model_id)
    return {"status": "activated", "model_name": model["model_name"]}


# -- Long-term memory ------------------------------------------------------


@router.get("/memory")
def list_memories(category: str = "", limit: int = 50) -> list[dict[str, object]]:
    """List all long-term memories, optionally filtered by category."""
    from agentflow.services.long_term_memory import LongTermMemory
    return LongTermMemory(db=store).get_all(category=category)


@router.get("/memory/search")
def search_memories(query: str, limit: int = 10) -> list[dict[str, object]]:
    """Search long-term memories by keyword."""
    from agentflow.services.long_term_memory import LongTermMemory
    return LongTermMemory(db=store).recall(query, limit=limit)


@router.delete("/memory/{key}")
def delete_memory(key: str) -> JSONResponse:
    """Delete a specific long-term memory."""
    from agentflow.services.long_term_memory import LongTermMemory
    ok = LongTermMemory(db=store).forget(key)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return JSONResponse(content={"status": "deleted"})


@router.delete("/memory")
def clear_memories(category: str = "") -> JSONResponse:
    """Clear all long-term memories, optionally filtered by category."""
    from agentflow.services.long_term_memory import LongTermMemory
    LongTermMemory(db=store).clear(category=category)
    return JSONResponse(content={"status": "cleared"})


# -- Tool introspection -------------------------------------------------------


@router.get("/tools")
def list_tools() -> list[dict[str, object]]:
    """List all registered tools with metadata."""
    from agentflow.graph.workflow import get_executor
    ex = get_executor()
    if ex is None:
        return []
    return ex.tool_metadata()


@router.get("/tools/capabilities")
def list_tool_capabilities() -> list[str]:
    """List all aggregated capabilities from registered tools."""
    from agentflow.graph.workflow import get_executor
    ex = get_executor()
    if ex is None:
        return []
    return ex.get_capabilities()


@router.get("/tools/executor")
def executor_status() -> dict[str, object]:
    """Return the Executor's status summary."""
    from agentflow.graph.workflow import get_executor
    ex = get_executor()
    if ex is None:
        return {"status": "not_initialised"}
    return {
        "status": "ready",
        "tools": ex.list_tools(),
        "capabilities": ex.get_capabilities(),
        "summary": ex.summary,
    }
