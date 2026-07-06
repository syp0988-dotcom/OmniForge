"""BaseTool — abstract interface for all tools in the AgentFlow framework.

Every tool **must** implement ``execute(**kwargs)`` and return a ``ToolResult``.

The uniform protocol means:

  - The Executor routes tasks to tools with zero per-tool adapter logic
  - Adding a new tool = implementing one class + calling ``registry.register(...)``
  - The Planner describes *what* needs doing (capability) and the Executor
    finds *who* can do it (tool registry)

Extending ``BaseTool``
----------------------
Subclasses should set ``name``, ``description`` and may optionally override
``validate()``, ``capabilities()`` and ``metadata()``.

Example::

    class MyTool(BaseTool):
        name = "my_tool"
        description = "Does something useful"

        def execute(self, **kwargs: Any) -> ToolResult:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentflow.tools.result import ToolResult


class BaseTool(ABC):
    """Abstract base for every tool in the framework.

    Required overrides:
        ``name`` (class-level string)
        ``execute()``

    Optional overrides:
        ``validate()``
        ``capabilities()``
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

    def capabilities(self) -> list[str]:
        """Return the semantic capabilities this tool provides.

        These strings are matched against ``task.capability`` in the Planner.
        Example: ``["web.search", "web.news"]``
        """
        return []

    def metadata(self) -> dict[str, Any]:
        """Rich metadata for introspection, documentation, and UI.

        The default build includes ``name``, ``description``, ``version``,
        ``capabilities``, and ``actions`` (derived from public methods that
        start with ``cmd_`` or from the class docstring).
        """
        return {
            "name": self.name,
            "description": self.description or self.__doc__ or "",
            "version": self.version,
            "capabilities": self.capabilities(),
        }
