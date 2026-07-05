"""SearchTool — web search capability via pluggable providers."""

from __future__ import annotations

from typing import Any

from agentflow.services.search_provider import (
    BaseSearchProvider,
    DuckDuckGoProvider,
)
from agentflow.tools.base import BaseTool
from agentflow.utils.logging import build_logger

logger = build_logger("search_tool")


class SearchTool(BaseTool):
    """Pluggable-provider web search tool.

    Executor usage::

        executor.execute(ctx, Task(
            goal="搜索网络信息",
            tool="search",
            input={"query": "..."},
        ))

    Provider injection::

        tool = SearchTool(provider=BraveProvider(api_key="..."))
        tool.execute(query="hello")
    """

    name = "search"

    def __init__(self, provider: BaseSearchProvider | None = None) -> None:
        self._provider = provider or DuckDuckGoProvider()

    def execute(self, query: str = "", **kwargs: Any) -> list[dict[str, Any]]:
        """Perform a web search (primary interface for Executor)."""
        return self.search(query or kwargs.get("q", ""))

    # -- Legacy interface (kept for backward compat) ------------------------

    def search(self, query: str) -> list[dict[str, Any]]:
        """Perform a web search via the configured provider."""
        return self._provider.search(query)

    @staticmethod
    def clean_url(url: str) -> str:
        """Decode DuckDuckGo redirect URLs (delegates to provider utility)."""
        return DuckDuckGoProvider._clean_url(url)
