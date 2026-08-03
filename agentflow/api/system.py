"""API router: system endpoints (split from the original monolith routes.py)."""

from __future__ import annotations

from fastapi import APIRouter
from agentflow.utils.logging import build_logger

from agentflow.agents.registry import get_all as get_all_agents

router = APIRouter()
logger = build_logger("api.system")

@router.get("/agents")
def list_agents() -> list[dict[str, object]]:
    """List all registered agents with metadata (name, key, status, capabilities)."""
    return get_all_agents()


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
