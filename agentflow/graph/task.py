"""Task — core work unit for the Agent Framework.

Each Task represents one decision → execution cycle:
  - An Agent decides an action (tool call, sub-agent dispatch, etc.)
  - The Executor creates a Task
  - The Tool / sub-agent executes
  - The Task collects the result or error

Tasks form a tree via ``parent_id`` for multi-agent / multi-tool collaboration,
and are owned by WorkflowContext which tracks all tasks in a workflow.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Full lifecycle states for a single Task.

    Primary states (Task Queue model)::

        TODO → RUNNING → DONE
                   ├── → FAILED
                   └── → BLOCKED → TODO
        TODO → SKIPPED

    Legacy states (kept for backward compat): PENDING, READY, WAITING,
    RETRYING, COMPLETED, CANCELLED.
    """

    # ── Primary states (Task Queue) ──
    TODO = "todo"             # Waiting to be picked up
    RUNNING = "running"       # Currently executing
    DONE = "done"             # Finished successfully
    FAILED = "failed"         # Finished with an unrecoverable error
    BLOCKED = "blocked"       # Blocked on dependency or external signal
    SKIPPED = "skipped"       # Deliberately skipped

    # ── Legacy (kept for backward compat) ──
    PENDING = "pending"
    READY = "ready"
    WAITING = "waiting"
    RETRYING = "retrying"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A single unit of work within a workflow.

    Attributes:
        goal: What this task aims to accomplish (human-readable).
        agent: Which agent decided this task (e.g. "planner", "router").
        tool: Which tool executes this task (e.g. "search", "python").
        status: Current lifecycle status.
        parent_id: ID of the parent task (for sub-tasks / multi-agent).
        input: Arguments passed to the tool/executor.
        steps: Sub-steps (for multi-step tools like browser).
        result: Output from the tool/executor.
        error: Error message if status == FAILED.
        metadata: Arbitrary metadata for extensibility.
        id: Unique identifier (12-char hex).
        created_at: ISO-8601 timestamp of creation.
        updated_at: ISO-8601 timestamp of last status change.
    """

    task_id: str = ""
    title: str = ""
    priority: int = 50
    dependencies: list[str] = field(default_factory=list)
    goal: str = ""
    capability: str = ""
    agent: str = ""
    tool: str = ""
    status: TaskStatus = TaskStatus.TODO
    parent_id: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # -- Status helpers ---------------------------------------------------------

    def mark_ready(self) -> None:
        """Mark the task as ready to execute."""
        self.status = TaskStatus.READY
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_running(self) -> None:
        """Mark the task as currently executing."""
        self.status = TaskStatus.RUNNING
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def complete(self, result: Any = None) -> None:
        """Mark the task as completed, optionally storing a result."""
        self.status = TaskStatus.DONE
        if result is not None:
            self.result = result
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def fail(self, error: str) -> None:
        """Mark the task as failed with an error message."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def skip(self, reason: str | None = None) -> None:
        """Mark the task as skipped, optionally with a reason."""
        self.status = TaskStatus.SKIPPED
        if reason:
            self.metadata["skip_reason"] = reason
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def cancel(self, reason: str | None = None) -> None:
        """Mark the task as cancelled, optionally with a reason."""
        self.status = TaskStatus.CANCELLED
        if reason:
            self.metadata["cancel_reason"] = reason
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # -- Convenience predicates -------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Whether the task has reached a terminal state."""
        return self.status in (
            TaskStatus.DONE,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.SKIPPED,
        )

    @property
    def is_finished(self) -> bool:
        """Whether the task finished successfully."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.DONE)

    # -- Serialization ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "title": self.title,
            "priority": self.priority,
            "dependencies": list(self.dependencies),
            "goal": self.goal,
            "capability": self.capability,
            "agent": self.agent,
            "tool": self.tool,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "input": self.input,
            "steps": self.steps,
            "result": _safe_value(self.result),
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Restore a Task from a dict produced by ``to_dict``."""
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            task_id=data.get("task_id", ""),
            title=data.get("title", ""),
            priority=data.get("priority", 50),
            dependencies=data.get("dependencies", []),
            goal=data.get("goal", ""),
            capability=data.get("capability", ""),
            agent=data.get("agent", ""),
            tool=data.get("tool", ""),
            status=TaskStatus(data.get("status", TaskStatus.TODO.value)),
            parent_id=data.get("parent_id"),
            input=data.get("input", {}),
            steps=data.get("steps", []),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


def _safe_value(value: Any) -> Any:
    """Convert non-serializable values to strings for JSON safety."""
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)
