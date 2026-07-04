from fastapi import APIRouter, HTTPException

from agentflow.database.sqlite import SQLiteStore
from agentflow.graph.workflow import build_workflow, run_workflow
from agentflow.models.chat import ChatRequest, ChatResponse

router = APIRouter()
store = SQLiteStore()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Handle chat requests through the workflow."""
    try:
        workflow = build_workflow()
        result = run_workflow(workflow, request.message)
        store.add_message("user", request.message)
        store.add_message("assistant", result["answer"])
        debug_data = {
            "category": result.get("category"),
            "workflow": result.get("workflow"),
            "search_results": result.get("search_results", []),
            "router": result.get("router", {}),
        }
        return ChatResponse(reply=result["answer"], metadata={"status": "ok"}, debug=debug_data)
    except Exception as exc:  # pragma: no cover - defensive path
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/upload")
def upload_file() -> dict[str, str]:
    """Upload endpoint placeholder."""
    return {"status": "accepted", "message": "Upload endpoint is ready for future document ingestion."}


@router.get("/history")
def history(limit: int = 20) -> list[dict[str, str]]:
    """Fetch recent chat history."""
    return store.list_messages(limit=limit)
