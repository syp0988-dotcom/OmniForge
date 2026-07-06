"""ToolResult — unified return type for all tools in the framework.

All tools return a ``ToolResult`` instance so that consumers (Executor,
Planner, AnswerAgent) can handle success/failure uniformly without
inspecting per-tool output shapes.

Schema::

    {
        "success": true,
        "tool": "filesystem",
        "action": "read_file",
        "result": {"content": "..."},
        "message": "File read successfully",
        "duration": 0.023,
        "error": null
    }
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Unified return envelope for every tool invocation.

    Attributes:
        success: Whether the operation completed without error.
        tool:    Name of the tool that executed (e.g. ``"filesystem"``).
        action:  The specific action performed (e.g. ``"read_file"``).
        result:  The primary payload produced by the tool.
        message: Human-readable summary of what happened.
        duration: Wall-clock seconds the tool took to execute.
        error:   Error message when ``success`` is ``False`` (``None`` otherwise).
    """

    success: bool
    tool: str
    action: str
    result: Any = None
    message: str = ""
    duration: float = 0.0
    error: str | None = None

    # -- Internal bookkeeping ----------------------------------------------------

    _start: float = field(default_factory=time.time, repr=False)

    def __post_init__(self) -> None:
        """Auto-capture duration if not explicitly set."""
        if self.duration == 0.0 and self._start:
            self.duration = round(time.time() - self._start, 4)

    # -- Factory helpers ---------------------------------------------------------

    @classmethod
    def ok(
        cls,
        tool: str,
        action: str,
        result: Any = None,
        message: str = "",
    ) -> ToolResult:
        """Shortcut for a successful result."""
        return cls(
            success=True,
            tool=tool,
            action=action,
            result=result,
            message=message or f"{tool}.{action} succeeded",
        )

    @classmethod
    def fail(
        cls,
        tool: str,
        action: str,
        error: str,
        result: Any = None,
    ) -> ToolResult:
        """Shortcut for a failed result."""
        return cls(
            success=False,
            tool=tool,
            action=action,
            result=result,
            error=error,
            message=f"{tool}.{action} failed: {error}",
        )

    # -- Serialization -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialisation."""
        return {
            "success": self.success,
            "tool": self.tool,
            "action": self.action,
            "result": _safe_value(self.result),
            "message": self.message,
            "duration": self.duration,
            "error": self.error,
        }


def _safe_value(value: Any) -> Any:
    """Convert non-serializable values to strings."""
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)
