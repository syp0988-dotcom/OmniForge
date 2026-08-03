"""Shared accessors for workflow state.

The workflow state carries two generations of fields: the original
``plan``/``router``/``category`` fields and the goal-driven
``goal_analysis``/``task_queue`` fields.  These helpers centralize the
"dict-or-object" and "old-or-new" fallbacks so agents and routers do not
repeat ``isinstance(plan, dict)`` checks inline.

Legacy fields are kept for backward compatibility (saved sessions and
downstream consumers), but new code should read through these accessors.
"""

from __future__ import annotations

from typing import Any


def get_goal_analysis(state: dict[str, Any]) -> dict[str, Any]:
    """Return ``state["goal_analysis"]`` as a plain dict (never None)."""
    ga = state.get("goal_analysis", {})
    return ga if isinstance(ga, dict) else {}


def get_goal(state: dict[str, Any]) -> str:
    """Return the resolved user goal, falling back to the raw question."""
    ga = get_goal_analysis(state)
    goal = ga.get("goal") or state.get("question", "")
    return str(goal or "")


def get_goal_type(state: dict[str, Any]) -> str:
    """Return the goal type, falling back to the legacy ``category`` field."""
    ga = get_goal_analysis(state)
    goal_type = ga.get("goal_type") or state.get("category", "other")
    return str(goal_type or "other")


def get_knowledge_source(state: dict[str, Any]) -> str:
    """Return the knowledge source from goal analysis (default ``"hybrid"``)."""
    ga = get_goal_analysis(state)
    return str(ga.get("knowledge_source", "hybrid") or "hybrid")


def get_source_mode(state: dict[str, Any]) -> str:
    """Return the source mode from state or goal analysis."""
    mode = state.get("source_mode", "") or ""
    if not mode:
        mode = get_goal_analysis(state).get("source_mode", "") or ""
    return str(mode or "auto")


def get_plan(state: dict[str, Any]) -> Any:
    """Return ``state["plan"]`` (dict or Plan object), or ``None``."""
    return state.get("plan")


def plan_flag(plan: Any, *flags: str) -> bool:
    """Return True when *plan* has any of the given flags truthy.

    Works for both dict plans and object plans (backward compat).
    """
    if plan is None:
        return False
    if isinstance(plan, dict):
        return any(bool(plan.get(flag)) for flag in flags)
    return any(bool(getattr(plan, flag, False)) for flag in flags)


def is_plan_completed(state: dict[str, Any]) -> bool:
    """True when the plan says the goal is completed or needs a direct answer."""
    return plan_flag(get_plan(state), "goal_completed", "direct_answer")


def get_plan_field(state: dict[str, Any], field: str, default: Any = None) -> Any:
    """Read a field from the plan whether it is a dict or an object."""
    plan = get_plan(state)
    if plan is None:
        return default
    if isinstance(plan, dict):
        return plan.get(field, default)
    return getattr(plan, field, default)


def get_plan_tasks(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return plan tasks as a list of plain dicts."""
    tasks = get_plan_field(state, "tasks", []) or []
    return [
        t if isinstance(t, dict)
        else t.to_dict() if hasattr(t, "to_dict")
        else {}
        for t in tasks
    ]
