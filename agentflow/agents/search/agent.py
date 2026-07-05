from __future__ import annotations

from agentflow.services.search_service import SearchService
from agentflow.utils.logging import build_logger

logger = build_logger("search")


class SearchAgent:
    """Decide whether to search and delegate execution to SearchService.

    This agent does NOT hold or import SearchTool — all search execution
    goes through ``SearchService``, which encapsulates business logic,
    parameter validation, and result normalization.
    """

    def __init__(self, search_service: SearchService | None = None) -> None:
        self._service = search_service or SearchService()

    def run(self, state: dict[str, object]) -> dict[str, object]:
        category = str(state.get("category", "reasoning"))
        question = str(state.get("question", ""))
        if category != "search":
            logger.info("Skipping search for category: %s", category)
            state["search_results"] = []
            return state

        logger.info("Searching for: %s", question)
        result = self._service.search(question)
        state["search_results"] = result.items
        return state
