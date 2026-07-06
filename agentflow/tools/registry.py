"""ToolRegistry — central plugin registry for all tools.

Responsibilities:

  - Register / unregister / query tools (plugin model)
  - Validate tasks before dispatching
  - Execute individual tasks or batches
  - Emit structured logs for every invocation

Usage::

    registry = ToolRegistry()
    registry.register(FileSystemTool(workspace="/safe/path"))
    registry.register(SearchTool())

    # Single task
    result = registry.execute_task({"tool": "filesystem", "action": "mkdir",
                                     "path": "app/models"})

    # Batch (returns results in order)
    results = registry.execute_batch([task1, task2, task3])
"""

from __future__ import annotations

import time
from typing import Any

from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult
from agentflow.utils.logging import build_logger

logger = build_logger("tool_registry")


class ToolRegistry:
    """Plugin-style registry that manages tool lifecycle and dispatch.

    Thread-safe for reads; writes are expected at startup / config time.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Plugin registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.

        The tool's ``.name`` attribute is used as the lookup key.

        Raises:
            TypeError: If *tool* is not a ``BaseTool`` instance.
            ValueError: If a tool with the same name is already registered.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Expected a BaseTool instance, got {type(tool).__name__}"
            )
        name = tool.name
        if not name:
            raise ValueError("Tool must have a non-empty '.name'")
        if name in self._tools:
            logger.warning("Overwriting already-registered tool '%s'", name)
        self._tools[name] = tool
        logger.info(
            "Registered tool '%s' (%s) — capabilities: %s",
            name,
            type(tool).__name__,
            tool.capabilities(),
        )

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Silently succeeds when *name* is not registered.
        """
        removed = self._tools.pop(name, None)
        if removed:
            logger.info("Unregistered tool '%s'", name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseTool | None:
        """Look up a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return names of all registered tools (sorted)."""
        return sorted(self._tools.keys())

    def list_with_metadata(self) -> list[dict[str, Any]]:
        """Return metadata dict for every registered tool."""
        return [
            {"name": name, **tool.metadata()}
            for name, tool in sorted(self._tools.items())
        ]

    def has_tool(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    # ------------------------------------------------------------------
    # Execution (single task)
    # ------------------------------------------------------------------

    def execute_task(
        self,
        tool_name: str,
        action: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a single action on a registered tool.

        Arguments:
            tool_name: Registered tool name (e.g. ``"filesystem"``).
            action:    Logical action name (e.g. ``"read_file"``).  Passed
                       through to the ``ToolResult`` for tracing.
            **kwargs:  Parameters forwarded to ``tool.execute()``.

        Returns:
            ``ToolResult`` — **never** raises.
        """
        start = time.time()
        tool = self._tools.get(tool_name)

        # --- Unknown tool -------------------------------------------------------
        if tool is None:
            logger.error("Unknown tool '%s' — available: %s", tool_name, self.list_tools())
            return ToolResult.fail(
                tool=tool_name,
                action=action,
                error=f"Unknown tool '{tool_name}'. Available: {self.list_tools()}",
            )

        # --- Validate -----------------------------------------------------------
        valid, error_msg = tool.validate(**kwargs)
        if not valid:
            logger.warning("Validation failed for %s.%s: %s", tool_name, action, error_msg)
            return ToolResult.fail(
                tool=tool_name,
                action=action,
                error=error_msg,
            )

        # --- Execute ------------------------------------------------------------
        try:
            logger.debug("Executing %s.%s (kwargs=%s)", tool_name, action, _summarise(kwargs))
            # Pass action as a keyword arg so tools that dispatch on it can use it
            execute_kwargs = dict(kwargs)
            if action and "action" not in execute_kwargs:
                execute_kwargs["action"] = action
            result: ToolResult = tool.execute(**execute_kwargs)
            dur = round(time.time() - start, 4)
            result.tool = tool_name
            result.action = action or result.action
            result.duration = dur
            if result.success:
                logger.info(
                    "OK  %s.%s  (%.2fs) — %s",
                    tool_name, action, dur, result.message,
                )
            else:
                logger.warning(
                    "FAIL %s.%s  (%.2fs) — %s",
                    tool_name, action, dur, result.error,
                )
            return result
        except Exception as exc:
            dur = round(time.time() - start, 4)
            logger.exception("EXCEPTION %s.%s  (%.2fs)", tool_name, action, dur)
            return ToolResult.fail(
                tool=tool_name,
                action=action,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Execution (task dict with "tool" / "action" keys)
    # ------------------------------------------------------------------

    def execute_task_dict(self, task_dict: dict[str, Any]) -> ToolResult:
        """Execute a task described as a dict.

        Expected keys (subset of the ``Task`` dataclass fields)::

            {"tool": "filesystem", "action": "mkdir", "path": "app"}
            {"tool": "search",  "action": "web.search", "query": "..."}

        ``action`` is optional; when absent the tool's default action is used.
        All remaining keys are passed as ``**kwargs`` to ``tool.execute()``.
        """
        tool_name = str(task_dict.get("tool", ""))
        action = str(task_dict.get("action", ""))
        # Everything except "tool" and "action" is a parameter
        kwargs = {k: v for k, v in task_dict.items() if k not in ("tool", "action")}
        return self.execute_task(tool_name, action=action, **kwargs)

    # ------------------------------------------------------------------
    # Execution (batch)
    # ------------------------------------------------------------------

    def execute_batch(
        self,
        task_dicts: list[dict[str, Any]],
        stop_on_failure: bool = False,
    ) -> list[ToolResult]:
        """Execute a sequence of task dicts in order.

        When *stop_on_failure* is ``True``, the batch short-circuits on the
        first unsuccessful result.
        """
        results: list[ToolResult] = []
        for td in task_dicts:
            r = self.execute_task_dict(td)
            results.append(r)
            if not r.success and stop_on_failure:
                logger.warning("Batch stopped at task %d due to failure", len(results))
                break
        return results


# -- Helpers -------------------------------------------------------------------

def _summarise(d: dict[str, Any], max_len: int = 120) -> str:
    """Short repr of a dict for log messages."""
    parts: list[str] = []
    for k, v in d.items():
        sv = str(v)
        if len(sv) > 60:
            sv = sv[:57] + "..."
        parts.append(f"{k}={sv}")
    s = ", ".join(parts)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s
