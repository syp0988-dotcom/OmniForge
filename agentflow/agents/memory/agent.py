from __future__ import annotations

from agentflow.utils.logging import build_logger

logger = build_logger("memory")


class MemoryAgent:
    """Maintain a lightweight memory summary for the session."""

    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        logger.info("Persisting memory for: %s", question)
        state["memory"] = {"last_question": question}
        return state
