"""API routes for chat and knowledge base management."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

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
        workflow = build_workflow()
        # Convert ChatMessage pydantic models to plain dicts for the workflow
        history_dicts = [
            {"role": m.role, "content": m.content} for m in request.history
        ]
        result = run_workflow(workflow, request.message, history=history_dicts)
        store.add_message("user", request.message)
        store.add_message("assistant", result["answer"])
        debug_data = {
            "category": result.get("category"),
            "workflow": result.get("workflow"),
            "search_results": result.get("search_results", []),
            "router": result.get("router", {}),
        }
        return ChatResponse(
            reply=result["answer"],
            metadata={"status": "ok"},
            debug=debug_data,
            proposed_files=propose_files(result["answer"]),
        )
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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


@router.post("/files/create")
def create_file(req: CreateFileRequest) -> JSONResponse:
    """Write a proposed file to the outputs/ directory."""
    safe_name = Path(req.filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    target = OUTPUT_DIR / safe_name
    if target.exists():
        raise HTTPException(status_code=409, detail="File already exists")
    target.write_text(req.content, encoding="utf-8")
    return JSONResponse(
        content={
            "status": "created",
            "filename": safe_name,
            "path": str(target.relative_to(Path(__file__).resolve().parents[2])),
        }
    )


@router.get("/files")
def list_output_files() -> list[dict[str, object]]:
    """List all files in the outputs/ directory."""
    if not OUTPUT_DIR.exists():
        return []
    files: list[dict[str, object]] = []
    for p in sorted(OUTPUT_DIR.iterdir()):
        if p.is_file():
            files.append(
                {
                    "filename": p.name,
                    "size": p.stat().st_size,
                    "created_at": datetime.fromtimestamp(p.stat().st_ctime).isoformat(),
                    "path": str(p.relative_to(Path(__file__).resolve().parents[2])),
                }
            )
    return files


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
