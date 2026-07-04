from __future__ import annotations

from agentflow.utils.logging import build_logger

logger = build_logger("planner")


class PlannerAgent:
    """Plan a workflow for an incoming user question."""

    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        workflow = ["search", "knowledge", "python", "report"]
        logger.info("Planning workflow for: %s", question)
        state["workflow"] = workflow
        state["plan"] = {"question": question, "steps": workflow}
        return state
