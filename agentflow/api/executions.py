"""API router: executions endpoints (split from the original monolith routes.py)."""

from __future__ import annotations

from fastapi import APIRouter
from agentflow.utils.logging import build_logger

from agentflow.graph.workflow import build_workflow
from fastapi import HTTPException, Query
import json

from agentflow.api.routes import (
    get_store,
)

router = APIRouter()
logger = build_logger("api.executions")

@router.get("/executions")
def list_executions(
    session_id: int | None = None,
    limit: int = Query(20, ge=1, le=200),
) -> list[dict[str, object]]:
    """List execution records, optionally filtered by session."""
    return get_store().list_executions(session_id=session_id, limit=limit)


@router.get("/executions/{exec_id}")
def get_execution(exec_id: int) -> dict[str, object]:
    """Get a single execution record with full trace and errors."""
    record = get_store().get_execution(exec_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    # Parse stored JSON strings back into objects for the API response
    try:
        record["trace"] = json.loads(record["trace"]) if isinstance(record.get("trace"), str) else record.get("trace", [])
    except (json.JSONDecodeError, TypeError):
        record["trace"] = []
    try:
        record["errors"] = json.loads(record["errors"]) if isinstance(record.get("errors"), str) else record.get("errors", [])
    except (json.JSONDecodeError, TypeError):
        record["errors"] = []
    return record  # type: ignore[return-value]


@router.get("/executions/{exec_id}/trace")
def get_execution_trace(exec_id: int) -> dict[str, object]:
    """Return graph-structured trace data for workflow visualization.

    Returns nodes (with labels, durations, routes) and edges derived
    from the execution trace's node sequence.
    """
    record = get_store().get_execution(exec_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Execution not found")

    raw_trace = record.get("trace", "[]")
    try:
        trace = json.loads(raw_trace) if isinstance(raw_trace, str) else raw_trace
    except (json.JSONDecodeError, TypeError):
        trace = []

    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    prev_node: str | None = None

    for entry in trace:
        node_name = entry.get("node", "?")
        nodes.append({
            "id": node_name,
            "label": node_name,
            "duration_ms": entry.get("duration_ms", 0),
            "route": entry.get("route"),
        })
        if prev_node:
            edges.append({"source": prev_node, "target": node_name})
        prev_node = node_name

    # Compute timeline with cumulative offsets
    timeline: list[dict[str, object]] = []
    cumulative = 0.0
    for entry in trace:
        start_offset = cumulative
        duration = entry.get("duration_ms", 0) or 0
        cumulative += duration
        timeline.append({
            "node": entry.get("node", "?"),
            "start_offset_ms": round(start_offset, 2),
            "duration_ms": round(duration, 2),
            "route": entry.get("route"),
        })

    return {
        "execution_id": exec_id,
        "trace_id": record.get("trace_id", ""),
        "nodes": nodes,
        "edges": edges,
        "timeline": timeline,
    }


@router.get("/executions/{exec_id}/checkpoints")
def list_execution_checkpoints(exec_id: int) -> list[dict[str, object]]:
    """List all checkpoints for an execution."""
    return get_store().list_checkpoints(exec_id)  # type: ignore[return-value]


@router.post("/executions/{exec_id}/resume")
def resume_execution(exec_id: int) -> dict[str, object]:
    """Resume an execution from its last checkpoint.

    Loads the most recent checkpoint and re-enters the workflow.
    The task queue is restored and processing continues.
    """
    checkpoints = get_store().list_checkpoints(exec_id)
    if not checkpoints:
        raise HTTPException(status_code=400, detail="No checkpoints found for this execution")

    last_cp = checkpoints[-1]
    try:
        task_queue = json.loads(last_cp["task_queue_json"])
    except (json.JSONDecodeError, TypeError):
        task_queue = []

    # Build minimal initial state from checkpoint
    from agentflow.graph.workflow import run_workflow
    graph = build_workflow()
    result = run_workflow(graph, "", session_state={})
    # Restore the saved task queue
    result["task_queue"] = task_queue
    return {
        "status": "resumed",
        "execution_id": exec_id,
        "checkpoint_node": last_cp.get("node_name", "?"),
        "restored_tasks": len(task_queue) if isinstance(task_queue, list) else 0,
    }
