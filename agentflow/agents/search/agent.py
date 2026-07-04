from __future__ import annotations

from agentflow.utils.logging import build_logger

logger = build_logger("search")


class SearchAgent:
    """Placeholder search agent with a DuckDuckGo-ready abstraction."""

    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        logger.info("Searching for: %s", question)
        state["search_results"] = [{"source": "duckduckgo", "query": question, "summary": "Search results placeholder"}]
        return state
