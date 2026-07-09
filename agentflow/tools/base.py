"""BaseTool — abstract interface for all tools in the AgentFlow framework.

Every tool **must** implement ``execute(**kwargs)`` and return a ``ToolResult``.

The uniform protocol means:

  - The Executor routes tasks to tools with zero per-tool adapter logic
  - Adding a new tool = implementing one class + calling ``registry.register(...)``
  - The Planner describes *what* needs doing (capability) and the Executor
    finds *who* can do it (tool registry)

Extending ``BaseTool``
----------------------
Subclasses should set ``name``, ``description`` and implement ``actions()``.

Example::

    class MyTool(BaseTool):
        name = "my_tool"
        description = "Does something useful"

        def actions(self) -> dict[str, dict]:
            return {
                "do_thing": {
                    "description": "Perform the thing",
                    "parameters": {
                        "input_file": {"type": "string", "description": "File to process"},
                    },
                    "required": ["input_file"],
                },
            }

        def execute(self, action="", **kwargs: Any) -> ToolResult:
            ...

The ToolRegistry derives all Planner schemas, capabilities, and prompts
dynamically from ``actions()`` — no more hardcoded lists in 7 files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentflow.tools.result import ToolResult


class BaseTool(ABC):
    """Abstract base for every tool in the framework.

    Required overrides:
        ``name`` (class-level string)
        ``actions()``
        ``execute()``

    Optional overrides:
        ``validate()``
        ``tool_schemas()``
        ``routing_node()``
        ``metadata()``
    """

    #: Short unique identifier — used as ``task.tool`` in the Planner.
    name: str = ""

    #: Human-readable summary shown in /tools introspection.
    description: str = ""

    #: Schema version of this tool's contract.
    version: str = "1.0"

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool's core function.

        Args:
            **kwargs: Tool-specific keyword arguments forwarded from
                      ``task.input`` by the Executor.

        Returns:
            A ``ToolResult`` instance.  **Every** code path must return a
            ``ToolResult`` — never a raw dict or ``None``.
        """
        ...

    # ------------------------------------------------------------------
    # Actions (single source of truth for schemas & capabilities)
    # ------------------------------------------------------------------

    def actions(self) -> dict[str, dict]:
        """Return action_name → {description, parameters, required}.

        This is the **single source of truth** for:
          - LLM function schemas (via ``tool_schemas()``)
          - Capabilities (``{tool}.{action}``)
          - Planner prompts (action allowlists)

        Each action dict has:
          ``description``  — human-readable (Chinese OK) description
          ``parameters``   — {param: {type, description}} dict
          ``required``     — list of required parameter names

        Subclasses **must** override this.  The old ``capabilities()``
        override is no longer necessary — it derives from ``actions()``
        automatically.
        """
        return {}

    # ------------------------------------------------------------------
    # LLM function schemas (derived from actions)
    # ------------------------------------------------------------------

    def tool_schemas(self) -> list[dict]:
        """Return OpenAI-compatible function definitions for all actions.

        Derived from ``actions()`` automatically.  Override only when a
        tool needs custom schema generation (e.g. additionalProperties).
        """
        schemas: list[dict] = []
        for action_name, action_def in self.actions().items():
            function_name = f"{self.name}__{action_name}"
            desc = action_def.get("description", "")
            params_raw = action_def.get("parameters", {})
            required = action_def.get("required", [])

            properties: dict[str, dict] = {}
            for param_name, param_def in params_raw.items():
                prop: dict = {"type": param_def.get("type", "string")}
                if "description" in param_def:
                    prop["description"] = param_def["description"]
                if "default" in param_def:
                    prop["default"] = param_def["default"]
                if "enum" in param_def:
                    prop["enum"] = param_def["enum"]
                properties[param_name] = prop

            schema: dict = {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                    },
                },
            }
            if required:
                schema["function"]["parameters"]["required"] = required

            # Allow tools to pass extra parameter fields (e.g. additionalProperties)
            extra_params = action_def.get("_extra_params")
            if extra_params:
                schema["function"]["parameters"].update(extra_params)

            schemas.append(schema)

        return schemas

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def routing_node(self) -> str:
        """LangGraph node that executes this tool.

        Default ``"tool_executor"`` covers filesystem, git, browser, etc.
        Override for:
          - ``"query_rewriter"`` for search tools
          - ``"python"`` for Python execution
        """
        return "tool_executor"

    # ------------------------------------------------------------------
    # Capabilities (derived from actions)
    # ------------------------------------------------------------------

    def capabilities(self) -> list[str]:
        """Return semantic capabilities: ``{tool}.{action}`` for each action.

        Derived automatically from ``actions()``.  No need to override.
        """
        return [f"{self.name}.{a}" for a in self.actions()]

    # ------------------------------------------------------------------
    # Optional — safety & introspection
    # ------------------------------------------------------------------

    def validate(self, **kwargs: Any) -> tuple[bool, str]:
        """Pre-execution parameter validation.

        Override to check parameter types, path safety, allowed values etc.

        Returns:
            ``(True, "")`` when parameters are valid.
            ``(False, "reason")`` when they are not.
        """
        _ = kwargs
        return True, ""

    def metadata(self) -> dict[str, Any]:
        """Rich metadata for introspection, documentation, and UI."""
        return {
            "name": self.name,
            "description": self.description or self.__doc__ or "",
            "version": self.version,
            "capabilities": self.capabilities(),
            "actions": list(self.actions().keys()),
            "routing_node": self.routing_node(),
        }
