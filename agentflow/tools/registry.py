"""ToolRegistry — central plugin registry for all tools.

Responsibilities:

  - Register / unregister / query tools (plugin model)
  - Validate tasks before dispatching
  - Execute individual tasks or batches
  - Emit structured logs for every invocation
  - Auto-discover tool classes by scanning the tools directory
  - Dynamically aggregate LLM schemas, capabilities, and routing info

Usage::

    registry = ToolRegistry()
    registry.register(FileSystemTool(workspace="/safe/path"))
    registry.register(SearchTool())

    # Single task
    result = registry.execute_task({"tool": "filesystem", "action": "mkdir",
                                     "path": "app/models"})

    # Batch (returns results in order)
    results = registry.execute_batch([task1, task2, task3])

    # Dynamic introspection (single source of truth for Planner)
    schemas = registry.get_all_tool_schemas()
    actions_text = registry.get_tool_actions_text()
    node = registry.get_node_for_tool("search")  # -> "query_rewriter"
"""

from __future__ import annotations

import importlib
import inspect
import time
from pathlib import Path
from typing import Any

from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult
from agentflow.utils.logging import build_logger

logger = build_logger("tool_registry")


class ToolRegistry:
    """Plugin-style registry that manages tool lifecycle and dispatch.

    Thread-safe for reads; writes are expected at startup / config time.
    The **single source of truth** for all tool metadata in the system.
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
        actions = list(tool.actions().keys()) if tool.actions() else []
        logger.info(
            "Registered tool '%s' (%s) — %d action(s): %s",
            name,
            type(tool).__name__,
            len(actions),
            actions,
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
    # Dynamic schema aggregation (single source of truth)
    # ------------------------------------------------------------------

    def get_all_tool_schemas(self) -> list[dict]:
        """Aggregate OpenAI function schemas from all registered tools.

        This replaces the hardcoded SEARCH_FUNCTIONS, FILESYSTEM_FUNCTIONS,
        etc. lists in schemas.py.  Always current with registered tools.
        """
        all_schemas: list[dict] = []
        for tool in self._tools.values():
            try:
                all_schemas.extend(tool.tool_schemas())
            except Exception as exc:
                logger.warning("Failed to get schemas from '%s': %s", tool.name, exc)
        return all_schemas

    def get_all_capabilities(self) -> list[str]:
        """Aggregate capabilities from all registered tools (sorted)."""
        caps: list[str] = []
        for tool in self._tools.values():
            caps.extend(tool.capabilities())
        return sorted(set(caps))

    def get_tool_actions_text(self) -> str:
        """Return a formatted summary of tools and actions for Planner prompts.

        Format::

            - filesystem: mkdir, write_file, create_file, read_file, ...
            - search: search
            - python: execute
            - git: status, diff, add, commit, ...
        """
        lines: list[str] = []
        for name in sorted(self._tools.keys()):
            tool = self._tools[name]
            actions_list = list(tool.actions().keys())
            if actions_list:
                lines.append(f"  - {name}: {', '.join(actions_list)}")
        return "\n".join(lines)

    def get_capability_descriptions(self) -> str:
        """Return formatted capability descriptions for Planner prompts.

        Format::

            - {tool}.{action}  —  {description}
        """
        lines: list[str] = []
        for name in sorted(self._tools.keys()):
            tool = self._tools[name]
            for action_name, action_def in tool.actions().items():
                desc = action_def.get("description", "")
                lines.append(f"  - {name}.{action_name}  —  {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Routing (dynamic — no more hardcoded _TOOL_TO_NODE dicts)
    # ------------------------------------------------------------------

    def get_node_for_tool(self, tool_name: str) -> str | None:
        """Return the LangGraph node name for a given tool.

        Queries the tool instance's ``routing_node()`` at runtime.
        Returns ``None`` for unknown tools.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return None
        return tool.routing_node()

    def get_executor_tools(self) -> frozenset:
        """Return tool names routed to ``tool_executor`` node."""
        return frozenset(
            name for name, tool in self._tools.items()
            if tool.routing_node() == "tool_executor"
        )

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

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    @staticmethod
    def auto_discover() -> list[type[BaseTool]]:
        """Scan ``agentflow/tools/*.py`` for ``BaseTool`` subclasses.

        Returns a list of **classes** (not instances).  The caller is
        responsible for instantiating and registering them — this allows
        per-tool configuration (workspace path, timeouts, etc.).

        Excludes private files (prefix ``_``) and the base module itself.
        """
        tools_dir = Path(__file__).resolve().parent
        discovered: list[type[BaseTool]] = []

        for py_file in sorted(tools_dir.glob("*.py")):
            if py_file.name.startswith("_") or py_file.name.startswith("."):
                continue
            if py_file.name in ("base.py", "registry.py", "result.py"):
                continue

            module_name = f"agentflow.tools.{py_file.stem}"
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                logger.debug("Skipping %s (import error: %s)", module_name, exc)
                continue

            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, BaseTool) or obj is BaseTool:
                    continue
                if obj.__module__ != module.__name__:
                    continue  # imported, not defined here
                discovered.append(obj)

        logger.info("Auto-discovered %d tool class(es): %s",
                     len(discovered),
                     [cls.name for cls in discovered])
        return discovered

    def register_all_discovered(
        self,
        overrides: dict[str, BaseTool] | None = None,
    ) -> list[str]:
        """Auto-discover and register all tools.

        Use *overrides* to supply pre-configured instances for tools that
        need special setup (e.g. ``FileSystemTool(workspace="/custom")``).
        Auto-discovered classes that have no override are instantiated with
        their default constructor.

        Returns names of all registered tools.
        """
        overrides = overrides or {}
        registered: list[str] = []

        for cls in self.auto_discover():
            name = cls.name
            if name in overrides:
                self.register(overrides[name])
            elif name not in self._tools:
                try:
                    self.register(cls())
                except Exception as exc:
                    logger.warning("Failed to instantiate %s: %s", cls.__name__, exc)
                    continue
            registered.append(name)

        return registered


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
