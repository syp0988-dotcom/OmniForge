from __future__ import annotations

from agentflow.services.llm_service import get_llm_service
from agentflow.utils.logging import build_logger

logger = build_logger("planner")


class PlannerAgent:
    """Plan a workflow for an incoming user question."""

    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        category = str(state.get("category", "reasoning"))
        logger.info("Planning workflow for: %s (category=%s)", question, category)

        if category == "search":
            workflow = ["router", "search", "answer", "memory"]
        elif category == "identity":
            workflow = ["router", "answer", "memory"]
        else:
            workflow = ["router", "answer", "memory"]

        state["workflow"] = workflow
        state["plan"] = {"question": question, "category": category, "steps": workflow}
        return state
