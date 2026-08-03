"""API router: memory endpoints (split from the original monolith routes.py)."""

from __future__ import annotations

from fastapi import APIRouter
from agentflow.utils.logging import build_logger

from agentflow.services.long_term_memory import LongTermMemory
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from agentflow.api.routes import (
    get_store,
)

router = APIRouter()
logger = build_logger("api.memory")

@router.get("/memory")
def list_memories(category: str = "", limit: int = 50) -> list[dict[str, object]]:
    """List all long-term memories, optionally filtered by category."""
    return LongTermMemory(db=get_store()).get_all(category=category)


@router.get("/memory/search")
def search_memories(query: str, limit: int = 10) -> list[dict[str, object]]:
    """Search long-term memories by keyword."""
    return LongTermMemory(db=get_store()).recall(query, limit=limit)


@router.delete("/memory/{key}")
def delete_memory(key: str) -> JSONResponse:
    """Delete a specific long-term memory."""
    ok = LongTermMemory(db=get_store()).forget(key)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return JSONResponse(content={"status": "deleted"})


@router.delete("/memory")
def clear_memories(category: str = "") -> JSONResponse:
    """Clear all long-term memories, optionally filtered by category."""
    LongTermMemory(db=get_store()).clear(category=category)
    return JSONResponse(content={"status": "cleared"})
