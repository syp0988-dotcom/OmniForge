from __future__ import annotations

from agentflow.utils.logging import build_logger

logger = build_logger("report")


class ReportAgent:
    """Generate a concise report from workflow results."""

    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        workflow = state.get("workflow", [])
        logger.info("Creating report for: %s", question)
        state["answer"] = (
            f"Processed request: {question}. "
            f"Workflow steps: {', '.join(str(step) for step in workflow)}."
        )
        return state
