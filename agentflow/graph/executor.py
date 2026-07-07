"""Executor — task lifecycle manager backed by the ToolRegistry.

The Executor is the single dispatch point between the Planner's tasks and
the concrete Tool implementations.  It:

  - Receives ``Task`` objects (or task dicts) from the Planner
  - Validates and dispatches each task to the correct tool via ``ToolRegistry``
  - Manages the full Task lifecycle (READY → RUNNING → COMPLETED / FAILED)
  - Emits structured events via ``EventBus``
  - Records structured logs for every dispatch (tool, action, duration, result)

Architecture::

    Planner ──→ Task ──→ Executor ──→ ToolRegistry ──→ BaseTool.execute()
                                                           │
                                                           ▼
                                                      ToolResult
"""

from __future__ import annotations

import time
from typing import Any

from agentflow.graph.context import WorkflowContext
from agentflow.graph.event import EventBus
from agentflow.graph.task import Task, TaskStatus
from agentflow.tools.base import BaseTool
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.result import ToolResult
from agentflow.utils.logging import build_logger

logger = build_logger("executor")


class Executor:
    """Routes Tasks to Tools via the ToolRegistry and manages lifecycle.

    Usage::

        executor = Executor()
        executor.register_tool("search", SearchTool())  # legacy API
        # or
        executor.registry.register(SearchTool())         # new plugin API

        task = Task(goal="搜索", tool="search", input={"query": "hello"})
        executor.execute(ctx, task)                      # single task
        executor.execute_plan(ctx, plan)                 # all tasks in a Plan
    """

    def __init__(self) -> None:
        self.registry = ToolRegistry()

    # ------------------------------------------------------------------
    # Legacy tool registration (delegates to ToolRegistry)
    # ------------------------------------------------------------------

    def register_tool(self, name: str, tool: BaseTool) -> None:
        """Register a tool by name (legacy — delegates to ``registry``).

        For new code, prefer ``registry.register(tool)`` which uses the
        tool's own ``.name`` attribute.
        """
        tool.name = name
        self.registry.register(tool)

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return self.registry.list_tools()

    def get_tool(self, name: str) -> BaseTool | None:
        """Look up a registered tool by name."""
        return self.registry.get(name)

    # ------------------------------------------------------------------
    # Single task execution
    # ------------------------------------------------------------------

    def execute(self, ctx: WorkflowContext, task: Task) -> Task:
        """Run a single task through its full lifecycle.

        The task is mutated in place and returned for chaining.

        Lifecycle::

            PENDING → READY → RUNNING → COMPLETED
                                    └── → FAILED
        """
        if task.status not in (TaskStatus.TODO, TaskStatus.PENDING, TaskStatus.READY):
            logger.warning(
                "Task %s is not TODO/PENDING/READY (status=%s); skipping.",
                task.id, task.status.value,
            )
            return task

        start = time.time()
        EventBus.task_created(ctx, task)
        task.mark_ready()
        task.mark_running()
        EventBus.task_started(ctx, task)

        # Look up tool via registry
        tool = self.registry.get(task.tool)
        if tool is None:
            err = (
                f"No tool registered for '{task.tool}'.  "
                f"Available: {self.registry.list_tools()}"
            )
            task.fail(err)
            EventBus.task_failed(ctx, task)
            logger.error("Task %s failed: %s", task.id, err)
            return task

        # Execute via ToolRegistry (which handles validate + dispatch + logging)
        # Use action from task.input if available, fall back to task.goal
        inputs = dict(task.input)
        tool_action = inputs.pop("action", None) or task.goal
        tool_result = self.registry.execute_task(
            task.tool,
            action=tool_action,
            **inputs,
        )

        duration = round(time.time() - start, 4)

        if tool_result.success:
            task.complete(tool_result.to_dict())
            EventBus.tool_finished(ctx, task, tool_result.to_dict())
            EventBus.task_finished(ctx, task)
            logger.info(
                "Task %s (%s.%s) completed in %.2fs — %s",
                task.id, task.tool, task.goal, duration, tool_result.message,
            )
        else:
            task.fail(tool_result.error or "Unknown error")
            EventBus.task_failed(ctx, task)
            logger.warning(
                "Task %s (%s.%s) FAILED in %.2fs — %s",
                task.id, task.tool, task.goal, duration, tool_result.error,
            )

        return task

    # ------------------------------------------------------------------
    # Dict-based execution (convenience for tool_executor node)
    # ------------------------------------------------------------------

    def execute_task_dict(
        self,
        task_dict: dict[str, Any],
        ctx: WorkflowContext | None = None,
    ) -> ToolResult:
        """Execute a task described as a dict.

        Expected keys: ``{"tool", "action", ...}``.
        Automatically unwraps the nested ``input`` field if present so that
        ``{"tool": "filesystem", "input": {"path": "app"}}`` becomes
        ``tool.execute(path="app")``.
        When a *ctx* is provided, events are emitted.
        """
        tool_name = str(task_dict.get("tool", ""))
        action = str(task_dict.get("action", ""))
        goal = str(task_dict.get("goal", ""))
        # Unwrap nested "input" dict, or collect top-level keys as kwargs
        if "input" in task_dict:
            raw_input = task_dict["input"]
            if isinstance(raw_input, dict):
                kwargs = dict(raw_input)
            else:
                kwargs = {"input": raw_input}
        else:
            # No "input" key: collect all non-meta keys as kwargs
            kwargs = {k: v for k, v in task_dict.items() if k not in ("tool", "action", "goal")}

        # Preserve the action for tool dispatch (FileSystemTool needs
        # action="mkdir" or "write_file" to find the right handler).
        if action:
            kwargs["action"] = action

        if ctx:
            # Create a lightweight Task for event tracking
            from agentflow.graph.task import Task as _Task
            t = _Task(goal=goal or action, tool=tool_name, input=kwargs, agent="executor")
            executed = self.execute(ctx, t)
            if executed.is_finished and isinstance(executed.result, dict):
                rd = executed.result
                return ToolResult(
                    success=rd.get("success", True),
                    tool=rd.get("tool", tool_name),
                    action=rd.get("action", action or goal),
                    result=rd.get("result"),
                    message=rd.get("message", ""),
                    duration=rd.get("duration", 0.0),
                    error=rd.get("error"),
                )
            return ToolResult(
                success=False, tool=tool_name, action=action or goal,
                error=executed.error or "Unknown error",
            )
        # When no ctx, pass kwargs directly (action is already in kwargs)
        return self.registry.execute_task(tool_name, **kwargs)

    def execute_batch(
        self,
        task_dicts: list[dict[str, Any]],
        ctx: WorkflowContext | None = None,
        stop_on_failure: bool = False,
    ) -> list[ToolResult]:
        """Execute a sequence of task dicts in order.

        When *stop_on_failure* is True, the batch short-circuits on error.
        """
        results: list[ToolResult] = []
        for td in task_dicts:
            r = self.execute_task_dict(td, ctx=ctx)
            results.append(r)
            if not r.success and stop_on_failure:
                logger.warning("Batch stopped at task %d due to failure", len(results))
                break
        return results

    # ------------------------------------------------------------------
    # Plan execution (batch from a Plan object)
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        ctx: WorkflowContext,
        plan: Any,  # Plan dataclass — avoid circular import at type-check level
        stop_on_failure: bool = False,
    ) -> list[ToolResult]:
        """Execute all tasks in a Plan and return their results.

        Each task is converted to a dict and dispatched through the registry.
        """
        from agentflow.graph.plan import Plan as _Plan

        if not isinstance(plan, _Plan):
            logger.warning("execute_plan called with non-Plan object: %s", type(plan).__name__)
            return []

        results: list[ToolResult] = []
        for task in plan.tasks:
            if task.is_terminal:
                continue
            task_dict = {
                "tool": task.tool,
                "action": task.goal,
                "goal": task.goal,
                **task.input,
            }
            r = self.execute_task_dict(task_dict, ctx=ctx)
            results.append(r)
            if not r.success and stop_on_failure:
                logger.warning("Plan execution stopped at task %s due to failure", task.id)
                break
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def tool_metadata(self) -> list[dict[str, Any]]:
        """Return metadata for every registered tool."""
        return self.registry.list_with_metadata()

    def get_capabilities(self) -> list[str]:
        """Aggregate capabilities from all registered tools."""
        caps: list[str] = []
        for tool in self.registry._tools.values():
            caps.extend(tool.capabilities())
        return sorted(set(caps))

    @property
    def summary(self) -> str:
        """Human-readable status summary."""
        tools = self.registry.list_tools()
        caps = self.get_capabilities()
        return (
            f"Executor: {len(tools)} tool(s), {len(caps)} capabilit(ies)\n"
            f"  Tools: {', '.join(tools)}\n"
            f"  Capabilities: {', '.join(caps)}"
        )
