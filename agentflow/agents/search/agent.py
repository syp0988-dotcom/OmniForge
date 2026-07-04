from __future__ import annotations

from agentflow.tools.search_tool import SearchTool
from agentflow.utils.logging import build_logger

logger = build_logger("search")


class SearchAgent:
    """Perform a real web search and expose structured results."""

    def __init__(self) -> None:
        self.tool = SearchTool()

    def run(self, state: dict[str, object]) -> dict[str, object]:
        category = str(state.get("category", "reasoning"))
        question = str(state.get("question", ""))
        if category != "search":
            logger.info("Skipping search for category: %s", category)
            state["search_results"] = []
            return state

        logger.info("Searching for: %s", question)
        results = self.tool.search(question)
        state["search_results"] = results
        return state
