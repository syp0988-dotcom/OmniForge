from __future__ import annotations

from typing import Any


class SearchTool:
    """Abstract search tool interface for future providers such as DuckDuckGo, Tavily, or Firecrawl."""

    def search(self, query: str) -> list[dict[str, Any]]:
        """Perform a search for the given query."""
        return [{"source": "placeholder", "query": query, "summary": "Search provider not configured"}]
