from __future__ import annotations

import json
from typing import Any

from agentflow.services.llm_service import get_llm_service
from agentflow.utils.logging import build_logger

logger = build_logger("report")


class ReportAgent:
    """Generate a concise report from workflow results."""

    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        workflow = state.get("workflow", [])
        search_results = state.get("search_results", [])
        llm_service = get_llm_service()

        logger.info("Creating report for: %s", question)
        search_context = json.dumps(search_results, ensure_ascii=False, indent=2)
        prompt = (
            "You are summarizing a multi-agent workflow result. "
            f"User question: {question}. "
            f"Workflow steps: {', '.join(str(step) for step in workflow)}. "
            f"Search results: {search_context}. "
            "Write a concise answer that cites the search sources by title and URL."
        )
        answer = llm_service.complete(prompt)
        state["answer"] = (
            f"Processed request: {question}. "
            f"Workflow steps: {', '.join(str(step) for step in workflow)}.\n\n{answer}"
        )
        return state
