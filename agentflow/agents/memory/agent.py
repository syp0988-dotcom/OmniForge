"""Memory Agent — maintains conversation history across turns."""

from __future__ import annotations

from typing import Any

from agentflow.utils.logging import build_logger

logger = build_logger("memory")


class MemoryAgent:
    """Maintain conversation history across turns.

    State contract:
      state["memory"] = {
        "history": list[{"role": str, "content": str}],
        "context_str": str   # formatted last N turns for prompt injection
      }

    The agent appends the current question and answer to history.
    ``state["history"]`` seeds the initial history (from ChatRequest).
    """

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns

    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        answer = str(state.get("answer", ""))
        existing_memory = state.get("memory")

        history: list[dict[str, str]] = []

        # Recover previous history if this is not the first turn
        if isinstance(existing_memory, dict):
            existing = existing_memory.get("history")
            if isinstance(existing, list):
                history = existing

        # Append current user turn
        history.append({"role": "user", "content": question})

        # Append assistant answer (set by AnswerAgent before memory runs)
        if answer:
            history.append({"role": "assistant", "content": answer})

        # Keep only last N turns
        if len(history) > self.max_turns * 2:
            history = history[-(self.max_turns * 2) :]

        # Build formatted string for LLM prompt injection
        context_lines = []
        for msg in history:
            role_label = "用户" if msg["role"] == "user" else "助手"
            context_lines.append(f"{role_label}: {msg['content']}")
        context_str = "\n".join(context_lines)

        state["memory"] = {
            "history": history,
            "context_str": context_str,
        }

        logger.info("Memory now holds %d messages", len(history))
        return state
