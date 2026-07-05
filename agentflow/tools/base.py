"""BaseTool — abstract interface for all tools in AgentFlow.

All tools must implement ``execute(**kwargs)``.  The Executor routes tasks
to tools based on ``task.tool`` and calls ``tool.execute(**task.input)``.

This uniform protocol means:

  - The Executor has zero per-tool adapter logic
  - Adding a new tool = implementing one class + registering it
  - The Planner describes *what* needs doing (capability) and the Executor
    finds *who* can do it (tool registry)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base for all tools in the AgentFlow framework.

    Subclasses must set ``name`` and implement ``execute()``.

    Example::

        class SearchTool(BaseTool):
            name = "search"

            def execute(self, query: str = "", **kwargs: Any) -> Any:
                ...  # perform the search
    """

    name: str = ""

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool's core function.

        Args:
            **kwargs: Tool-specific keyword arguments forwarded from
                      ``task.input`` by the Executor.

        Returns:
            Any JSON-serialisable result.
        """
        ...
