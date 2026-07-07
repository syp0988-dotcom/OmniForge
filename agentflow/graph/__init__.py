"""Graph — workflow orchestration core for AgentFlow."""

from __future__ import annotations

from typing import Any

from agentflow.conversation.session_state import SessionState
from agentflow.graph.context import WorkflowContext
from agentflow.graph.event import Event, EventBus, EventType
from agentflow.graph.executor import Executor
from agentflow.graph.plan import Plan
from agentflow.graph.task import Task, TaskStatus

__all__ = [
    "WorkflowContext",
    "Event",
    "EventBus",
    "EventType",
    "Executor",
    "Plan",
    "SessionState",
    "Task",
    "TaskStatus",
    "TaskQueue",
    "ContextBuilder",
    "build_workflow",
    "get_executor",
    "run_workflow",
]


def __getattr__(name: str) -> Any:
    """Lazy-import optional modules to avoid circular imports."""
    if name == "TaskQueue":
        from agentflow.graph.task_queue import TaskQueue as _tq
        return _tq
    if name == "ContextBuilder":
        from agentflow.graph.context_builder import ContextBuilder as _cb
        return _cb
    if name in ("build_workflow", "get_executor", "run_workflow"):
        import agentflow.graph.workflow as wf
        return getattr(wf, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __getattr__(name: str) -> Any:
    """Lazy-import ``build_workflow`` / ``get_executor`` / ``run_workflow``
    to avoid a circular import chain::

        graph.__init__ → workflow → PlannerAgent → graph.plan
    """
    if name in ("build_workflow", "get_executor", "run_workflow"):
        import agentflow.graph.workflow as wf

        return getattr(wf, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
