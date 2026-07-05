"""Executor — Task lifecycle manager for the Agent Framework.

The Executor routes Tasks to the correct Tool via a typed registry and
manages the full Task lifecycle:

  - Status transitions (READY → RUNNING → COMPLETED / FAILED)
  - Tool dispatch via ``tool.execute(**task.input)`` (uniform BaseTool protocol)
  - Lifecycle events via EventBus (task.created, tool.started, …)
  - Error capture and task marking

Every Tool registered with the Executor must implement ``BaseTool`` so that
dispatch requires zero per-tool adapter code.
"""

from __future__ import annotations

from typing import Any

from agentflow.graph.context import WorkflowContext
from agentflow.graph.event import EventBus
from agentflow.graph.task import Task, TaskStatus
from agentflow.tools.base import BaseTool
from agentflow.utils.logging import build_logger

logger = build_logger("executor")


class Executor:
    """Routes Tasks to Tools and manages their lifecycle.

    Usage::

        executor = Executor()
        executor.register_tool("search", SearchTool())

        task = Task(goal="搜索", tool="search", input={"query": "hello"})
        executor.execute(ctx, task)
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # -- Tool registry ----------------------------------------------------------

    def register_tool(self, name: str, tool: BaseTool) -> None:
        """Register a ``BaseTool`` instance under a logical name.

        ``name`` is matched against ``task.tool`` at execution time.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Expected a BaseTool instance for '{name}', got {type(tool).__name__}"
            )
        self._tools[name] = tool
        logger.debug("Registered tool '%s': %s", name, type(tool).__name__)

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    def get_tool(self, name: str) -> BaseTool | None:
        """Look up a registered tool by name."""
        return self._tools.get(name)

    # -- Execution --------------------------------------------------------------

    def execute(self, ctx: WorkflowContext, task: Task) -> Task:
        """Run a single task through its full lifecycle.

        The task is mutated in place.  Returns the task for chaining.

        Lifecycle::

            PENDING → READY → RUNNING → COMPLETED
                                    └── → FAILED
        """
        if task.status not in (TaskStatus.PENDING, TaskStatus.READY):
            logger.warning(
                "Task %s is not PENDING/READY (status=%s); skipping.",
                task.id,
                task.status.value,
            )
            return task

        EventBus.task_created(ctx, task)
        task.mark_ready()
        task.mark_running()
        EventBus.task_started(ctx, task)

        # Look up the tool
        tool = self._tools.get(task.tool)
        if tool is None:
            task.fail(
                f"No tool registered for '{task.tool}'.  "
                f"Available: {list(self._tools.keys())}"
            )
            EventBus.task_failed(ctx, task)
            return task

        # Execute via uniform BaseTool protocol
        try:
            EventBus.tool_started(ctx, task, task.input)
            result = tool.execute(**task.input)
            task.complete(result)
            EventBus.tool_finished(ctx, task, result)
            EventBus.task_finished(ctx, task)
            logger.info(
                "Task %s (%s) completed",
                task.id,
                task.tool,
            )
        except Exception as exc:
            logger.exception("Task %s (%s) failed", task.id, task.tool)
            task.fail(str(exc))
            EventBus.task_failed(ctx, task)

        return task
