from __future__ import annotations

from agentflow.agents.base import AgentProtocol
from agentflow.services.search_service import SearchService
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("search")


class SearchAgent(AgentProtocol):
    """Decide whether to search and delegate execution to SearchService.

    This agent does NOT hold or import SearchTool — all search execution
    goes through ``SearchService``, which encapsulates business logic,
    parameter validation, and result normalization.
    """

    def __init__(self, search_service: SearchService | None = None) -> None:
        self._service = search_service or SearchService()

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        category = str(state.get("category", "reasoning"))
        # Use rewritten_query from QueryRewriter if available, fallback to raw question
        query = str(state.get("rewritten_query", "") or state.get("question", ""))

        if not query:
            logger.info("Empty query, skipping search")
            state["search_results"] = []
            return state

        logger.info("Searching: %s (category=%s)", query[:80], category)
        result = self._service.search(query)
        state["search_results"] = result.items
        return state
