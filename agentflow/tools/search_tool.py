"""SearchTool — web search capability via pluggable providers.

Supports DuckDuckGo (default) and Tavily (when ``TAVILY_API_KEY`` is set).
All results are returned in the unified ``ToolResult`` envelope.
"""

from __future__ import annotations

from typing import Any

from agentflow.services.search_provider import (
    BaseSearchProvider,
    DuckDuckGoProvider,
)
from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult
from agentflow.utils.logging import build_logger

logger = build_logger("search_tool")


class SearchTool(BaseTool):
    """Pluggable-provider web search tool.

    Usage via Executor::

        registry.execute_task("search", action="web.search", query="...")
    """

    name = "search"
    description = "Web search via pluggable providers (DuckDuckGo / Tavily)"

    def __init__(self, provider: BaseSearchProvider | None = None) -> None:
        if provider is not None:
            self._provider = provider
        else:
            from agentflow.services.search_provider import TavilyProvider

            tavily = TavilyProvider()
            if tavily._api_key:
                self._provider = tavily
                logger.info("SearchTool using Tavily provider")
            else:
                self._provider = DuckDuckGoProvider()
                logger.info("SearchTool using DuckDuckGo provider")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def actions(self) -> dict[str, dict]:
        return {
            "search": {
                "description": "从互联网搜索实时信息（新闻、天气、价格等）",
                "parameters": {
                    "query": {"type": "string", "description": "简洁具体的搜索关键词"},
                },
                "required": ["query"],
            },
        }

    def routing_node(self) -> str:
        return "query_rewriter"

    def capabilities(self) -> list[str]:
        return ["web.search", "search.search"]

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base["provider"] = type(self._provider).__name__
        return base

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, query: str = "", **kwargs: Any) -> ToolResult:
        """Perform a web search.

        Accepts the query via ``query`` keyword or ``q`` / ``question``
        fallback keys for backward compatibility with different task formats.
        """
        q = query or kwargs.get("q") or kwargs.get("question") or ""
        if not q:
            return ToolResult.fail(self.name, "search", "No search query provided")

        try:
            results = self._provider.search(q)
            return ToolResult.ok(
                self.name,
                "search",
                result={"items": list(results), "count": len(results), "query": q},
                message=f"Found {len(results)} results for '{q[:60]}'",
            )
        except Exception as exc:
            logger.exception("Search failed: %s", exc)
            return ToolResult.fail(self.name, "search", f"Search failed: {exc}")

    # ------------------------------------------------------------------
    # Legacy helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict[str, Any]]:
        """Legacy interface — returns raw results list.

        Deprecated: prefer ``execute(query=...)`` which returns ``ToolResult``.
        """
        return self._provider.search(query)

    @staticmethod
    def clean_url(url: str) -> str:
        """Decode DuckDuckGo redirect URLs."""
        return DuckDuckGoProvider._clean_url(url)
