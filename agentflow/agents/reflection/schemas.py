"""Structured schemas for ReflectionAgent LLM output.

The Reflector asks the LLM for a JSON payload describing task-queue updates.
Before those updates are applied to the queue, the payload is validated with
these Pydantic models so that malformed or hallucinated fields never mutate
the task queue silently.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TaskUpdate(BaseModel):
    """A status/priority change for an existing task."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    status: str | None = None
    priority: int | None = None


class NewTask(BaseModel):
    """A new task the reflector wants added to the queue."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    title: str = ""
    priority: int = Field(default=50, ge=0, le=100)
    tool: str = "filesystem"
    goal: str = ""
    input: dict = Field(default_factory=dict)


class ReflectionOutput(BaseModel):
    """Validated reflector decision payload."""

    model_config = ConfigDict(extra="ignore")

    goal_completed: bool = False
    task_updates: list[TaskUpdate] = Field(default_factory=list)
    new_tasks: list[NewTask] = Field(default_factory=list)
    remove_tasks: list[str] = Field(default_factory=list)
    need_replan: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        """Return a plain dict safe for downstream consumers."""
        return {
            "goal_completed": self.goal_completed,
            "task_updates": [u.model_dump() for u in self.task_updates],
            "new_tasks": [t.model_dump() for t in self.new_tasks],
            "remove_tasks": list(self.remove_tasks),
            "need_replan": self.need_replan,
            "reason": self.reason,
        }
