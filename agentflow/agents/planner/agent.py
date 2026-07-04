from __future__ import annotations

from agentflow.services.llm_service import get_llm_service
from agentflow.utils.logging import build_logger

logger = build_logger("planner")


class PlannerAgent:
    """Plan a workflow for an incoming user question."""

    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        workflow = ["search", "summary"]
        llm_service = get_llm_service()
        plan_prompt = (
            "Create a short workflow plan for the user request. "
            f"User request: {question}. "
            "Return only a short comma-separated list of steps."
        )
        llm_output = llm_service.complete(plan_prompt)
        logger.info("Planning workflow for: %s", question)
        state["workflow"] = workflow
        state["plan"] = {"question": question, "steps": workflow, "llm_plan": llm_output}
        return state
