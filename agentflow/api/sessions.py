"""API router: sessions endpoints (split from the original monolith routes.py)."""

from __future__ import annotations

from fastapi import APIRouter
from agentflow.utils.logging import build_logger

from agentflow.config.settings import settings
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from agentflow.api.routes import (
    get_store,
)

router = APIRouter()
logger = build_logger("api.sessions")

@router.get("/history")
def history(limit: int = 20) -> list[dict[str, str]]:
    """Fetch recent chat history."""
    return get_store().list_messages(limit=limit)


@router.post("/sessions/create")
def create_session() -> JSONResponse:
    """Create a new chat session."""
    sess = get_store().create_session()
    return JSONResponse(content=sess)


@router.get("/sessions")
def list_sessions(limit: int = 50) -> list[dict[str, object]]:
    """List all chat sessions, most recent first."""
    return get_store().list_sessions(limit=limit)


@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: int) -> list[dict[str, object]]:
    """Get all messages for a session."""
    sess = get_store().get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return get_store().get_session_messages(session_id)


@router.put("/sessions/{session_id}/rename")
def rename_session(session_id: int, body: dict[str, str]) -> JSONResponse:
    """Rename a session."""
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    ok = get_store().update_session_title(session_id, title)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(content={"status": "ok"})


@router.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: int) -> JSONResponse:
    """Delete a session and all its messages."""
    ok = get_store().delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(content={"status": "deleted"})


@router.post("/sessions/cleanup")
def cleanup_sessions() -> JSONResponse:
    """Manually trigger cleanup of expired sessions and memories.

    Uses settings ``session_ttl_hours`` and ``memory_ttl_days``.
    """
    deleted_sessions = get_store().delete_sessions_older_than(settings.session_ttl_hours)
    deleted_memories = get_store().delete_old_memories(settings.memory_ttl_days)
    logger.info(
        "Manual cleanup: removed %d sessions, %d memory entries",
        deleted_sessions, deleted_memories,
    )
    return JSONResponse(content={
        "status": "ok",
        "deleted_sessions": deleted_sessions,
        "deleted_memories": deleted_memories,
    })
