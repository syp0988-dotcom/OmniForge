"""Event — runtime observability events for the Agent Framework.

The Event Bus provides a minimal, typed event system that records what
happens during a workflow invocation.  Events are stored on the
WorkflowContext and can be consumed by:

  - The frontend (real-time streaming via SSE / WebSocket)
  - Logging / audit pipelines
  - Timeline or replay views
  - Debug tooling

Design principles:
  - Events are append-only — never mutated after creation
  - Events carry enough context to be useful standalone
  - The EventBus class provides convenience factories for common events
  - New event types can be added without changing existing code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

if __name__ == "__main__":
    import sys

    sys.path.insert(0, "..")

from agentflow.graph.task import Task


class EventType(str, Enum):
    """Canonical event types.

    Convention: ``<domain>.<action>`` for hierarchical filtering.
    """

    # -- Task lifecycle ---------------------------------------------------------
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_FINISHED = "task.finished"
    TASK_FAILED = "task.failed"

    # -- Tool lifecycle ---------------------------------------------------------
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"


@dataclass
class Event:
    """A single immutable event emitted during workflow execution.

    Attributes:
        type: Event type (from ``EventType``).
        timestamp: ISO-8601 timestamp of when the event occurred.
        task_id: The task this event relates to (empty string if global).
        agent: The agent that triggered this event.
        tool: The tool involved (for tool events).
        data: Arbitrary structured payload.
    """

    type: EventType
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    task_id: str = ""
    agent: str = ""
    tool: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "agent": self.agent,
            "tool": self.tool,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# EventBus — convenience factories
# ---------------------------------------------------------------------------

# Avoid circular import at module level — import WorkflowContext lazily.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentflow.graph.context import WorkflowContext


class EventBus:
    """Static convenience methods for emitting common workflow events.

    Usage::

        from agentflow.graph.event import EventBus

        EventBus.task_created(ctx, my_task)
        EventBus.task_started(ctx, my_task)
    """

    # -- Task events ---------------------------------------------------------

    @staticmethod
    def task_created(ctx: WorkflowContext, task: Task) -> None:
        """Emit when a new Task is created and added to the context."""
        ctx.add_event(
            Event(
                type=EventType.TASK_CREATED,
                task_id=task.id,
                agent=task.agent,
                tool=task.tool,
                data={"goal": task.goal, "parent_id": task.parent_id},
            )
        )

    @staticmethod
    def task_started(ctx: WorkflowContext, task: Task) -> None:
        """Emit when a Task begins execution."""
        ctx.add_event(
            Event(
                type=EventType.TASK_STARTED,
                task_id=task.id,
                agent=task.agent,
                tool=task.tool,
                data={"goal": task.goal},
            )
        )

    @staticmethod
    def task_finished(ctx: WorkflowContext, task: Task) -> None:
        """Emit when a Task completes successfully."""
        ctx.add_event(
            Event(
                type=EventType.TASK_FINISHED,
                task_id=task.id,
                agent=task.agent,
                tool=task.tool,
                data={"goal": task.goal},
            )
        )

    @staticmethod
    def task_failed(ctx: WorkflowContext, task: Task) -> None:
        """Emit when a Task fails with an unrecoverable error."""
        ctx.add_event(
            Event(
                type=EventType.TASK_FAILED,
                task_id=task.id,
                agent=task.agent,
                tool=task.tool,
                data={"goal": task.goal, "error": task.error},
            )
        )

    # -- Tool events ---------------------------------------------------------

    @staticmethod
    def tool_started(
        ctx: WorkflowContext,
        task: Task,
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        """Emit when a tool begins executing a task's work."""
        ctx.add_event(
            Event(
                type=EventType.TOOL_STARTED,
                task_id=task.id,
                agent=task.agent,
                tool=task.tool,
                data={"goal": task.goal, "input": tool_input or {}},
            )
        )

    @staticmethod
    def tool_finished(
        ctx: WorkflowContext,
        task: Task,
        result: Any = None,
    ) -> None:
        """Emit when a tool finishes executing a task's work."""
        ctx.add_event(
            Event(
                type=EventType.TOOL_FINISHED,
                task_id=task.id,
                agent=task.agent,
                tool=task.tool,
                data={"goal": task.goal, "result": result},
            )
        )
