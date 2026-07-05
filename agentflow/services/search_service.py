"""SearchService — business logic layer for web search.

Layering::

    SearchAgent (decision)
         │
         ▼
    SearchService (business logic, result normalization)
         │
         ▼
    SearchTool (execution, implements BaseTool for Executor)
         │
         ▼
    DuckDuckGoProvider (concrete search implementation)

Usage::

    service = SearchService()
    result: SearchResult = service.search("your query")
    for item in result.items:
        print(item["title"], item["url"], item["snippet"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentflow.tools.search_tool import SearchTool
from agentflow.utils.logging import build_logger

logger = build_logger("search_service")


@dataclass
class SearchResult:
    """Unified, structured search result."""

    query: str
    """The original search query."""

    items: list[dict[str, Any]] = field(default_factory=list)
    """Normalized result list. Each item has ``title``, ``url``, ``snippet``."""

    count: int = 0
    """Number of results in ``items`` (derived, kept for serialization)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Provider metadata (e.g. ``{"source": "duckduckgo"}``)."""

    def __post_init__(self) -> None:
        self.count = len(self.items)


class SearchService:
    """Orchestrates web search: parameter validation, execution, result normalization.

    The service is the **only** component that agents interact with directly.
    It encapsulates all search-specific business logic so that agents remain
    purely decision-making.
    """

    def __init__(self, search_tool: SearchTool | None = None) -> None:
        self._tool = search_tool or SearchTool()

    def search(self, query: str) -> SearchResult:
        """Execute a search and return a normalized ``SearchResult``."""
        # --- Parameter validation ---
        if not query or not query.strip():
            logger.warning("Empty search query received")
            return SearchResult(query=query or "", metadata={"source": "none"})

        cleaned = query.strip()
        logger.info("Executing search: %.80s", cleaned)

        # --- Execute via SearchTool (BaseTool protocol) ---
        try:
            items = self._tool.execute(query=cleaned)
        except Exception as exc:
            logger.exception("Search failed for query: %.80s", cleaned)
            return SearchResult(
                query=cleaned,
                metadata={"source": "duckduckgo", "error": str(exc)},
            )

        # --- Normalization ---
        normalized = self._normalize(items)
        result = SearchResult(
            query=cleaned,
            items=normalized,
            metadata={"source": "duckduckgo"},
        )
        logger.info("Search returned %d results", result.count)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(raw: list[Any]) -> list[dict[str, Any]]:
        """Ensure every result dict contains ``title``, ``url``, ``snippet``.

        Silently drops non-dict entries and fills missing fields with defaults.
        """
        normalized: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            normalized.append({
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("snippet", item.get("content", ""))),
            })
        return normalized
