"""TaskQueue — prioritized task queue for Dynamic Task Queue Planning.

The TaskQueue is the **single source of truth** for work state in the
Dynamic Task Queue Planner runtime.  All agents (Planner, Executor,
Reflector) read from and write to the same queue.

Lifecycle states::

    TODO -> RUNNING -> DONE
              |-> FAILED
              |-> BLOCKED -> TODO
    TODO -> SKIPPED

Usage::

    queue = TaskQueue()
    queue.add(Task(task_id="create_backend", priority=80))
    queue.add(Task(task_id="create_db", priority=70))

    next_task = queue.get_next()          # highest-priority TODO
    queue.update("create_backend", status="DONE")
    queue.remove("create_db")
"""

from __future__ import annotations

import logging
from typing import Any

from agentflow.graph.task import Task, TaskStatus

logger = logging.getLogger("planner.task_queue")


class TaskQueue:
    """A prioritized queue of tasks with status tracking."""

    def __init__(self, tasks: list[Task] | None = None) -> None:
        self._tasks: list[Task] = list(tasks) if tasks else []

    # -- Mutation -----------------------------------------------------------

    def add(self, *tasks: Task) -> None:
        """Add one or more tasks.

        Existing tasks with the same task_id are replaced, unless the
        existing task is already DONE — completed tasks are never
        overwritten by new task definitions.  This prevents infinite
        re-execution loops when the planner repeatedly generates the
        same task id.
        """
        for t in tasks:
            idx = self._find_index(t.task_id)
            if idx is not None:
                if self._tasks[idx].status == TaskStatus.DONE:
                    continue  # Never replace completed tasks
                self._tasks[idx] = t
            else:
                self._tasks.append(t)

    def remove(self, task_id: str) -> bool:
        """Remove a task by task_id.  Returns True if found."""
        idx = self._find_index(task_id)
        if idx is not None:
            self._tasks.pop(idx)
            return True
        return False

    def update(self, task_id: str, **updates: Any) -> bool:
        """Update fields on an existing task.

        Accepts any Task field (status, priority, title, etc.).
        Status strings are normalized to lowercase.
        """
        idx = self._find_index(task_id)
        if idx is None:
            return False
        task = self._tasks[idx]
        for key, value in updates.items():
            if key == "status":
                if isinstance(value, TaskStatus):
                    task.status = value
                elif isinstance(value, str):
                    try:
                        task.status = TaskStatus(value.lower())
                    except ValueError:
                        # Unknown statuses used to raise out of update(), which
                        # aborted the whole planner cycle.  Degrade instead:
                        # keep the previous status and log the value.
                        logger.warning(
                            "Ignoring unknown status %r for task %s", value, task_id,
                        )
                        continue
                else:
                    logger.warning(
                        "Ignoring non-string status %r for task %s", value, task_id,
                    )
                    continue
            else:
                setattr(task, key, value)
        return True

    # -- Query ---------------------------------------------------------------

    def get(self, task_id: str) -> Task | None:
        """Look up a task by task_id."""
        idx = self._find_index(task_id)
        return self._tasks[idx] if idx is not None else None

    def get_next(self) -> Task | None:
        """Return the highest-priority TODO task, or None."""
        todo = self.filter(status="todo")
        if not todo:
            return None
        return max(todo, key=lambda t: t.priority)

    def filter(
        self,
        *,
        status: str | None = None,
        tool: str | None = None,
    ) -> list[Task]:
        """Return tasks matching the given criteria."""
        result = self._tasks[:]
        if status:
            status_val = status.lower()
            result = [t for t in result if t.status.value == status_val]
        if tool:
            result = [t for t in result if t.tool == tool]
        return result

    # -- Properties ---------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return len(self._tasks) == 0

    @property
    def has_todo(self) -> bool:
        return any(t.status == TaskStatus.TODO for t in self._tasks)

    @property
    def todo_count(self) -> int:
        return sum(1 for t in self._tasks if t.status == TaskStatus.TODO)

    @property
    def all(self) -> list[Task]:
        """All tasks sorted by priority (highest first)."""
        return sorted(self._tasks, key=lambda t: t.priority, reverse=True)

    @property
    def summary(self) -> str:
        """Format the queue as a readable string for LLM prompts."""
        if not self._tasks:
            return "(empty task queue)"

        lines = [f"Task Queue: {len(self._tasks)} tasks"]
        for t in self.all:
            icon = {
                TaskStatus.TODO: "[TODO]",
                TaskStatus.RUNNING: "[RUN]",
                TaskStatus.DONE: "[DONE]",
                TaskStatus.FAILED: "[FAIL]",
                TaskStatus.BLOCKED: "[BLKD]",
                TaskStatus.SKIPPED: "[SKIP]",
            }.get(t.status, "[?]")
            title = t.title or t.goal or t.task_id
            lines.append(f"  {icon} P={t.priority} [{t.task_id}] {title}")
        return "\n".join(lines)

    # -- Serialization -------------------------------------------------------

    def to_dict_list(self) -> list[dict]:
        """Serialize all tasks to a list of plain dicts."""
        return [t.to_dict() for t in self._tasks]

    @classmethod
    def from_dict_list(cls, items: list[dict]) -> TaskQueue:
        """Restore from a list of dicts."""
        return cls(tasks=[Task.from_dict(d) for d in items])

    # -- Internal ------------------------------------------------------------

    def _find_index(self, task_id: str) -> int | None:
        for i, t in enumerate(self._tasks):
            if t.task_id == task_id:
                return i
        return None
