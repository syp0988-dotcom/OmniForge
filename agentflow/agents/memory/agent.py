"""Memory Agent — maintains conversation history across turns."""

from __future__ import annotations

from typing import Any

from agentflow.agents.base import AgentProtocol
from agentflow.conversation.manager import ConversationManager
from agentflow.conversation.session_state import SessionState
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("memory")


class MemoryAgent(AgentProtocol):
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

    @safe_run
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

        # -- Update session state heuristics based on the answer -----------
        if answer:
            ss = state.get("session_state")
            if not isinstance(ss, SessionState):
                ss = SessionState()
            ConversationManager.finalize_turn(state, ss, answer)
            state["session_state"] = ss  # SessionState object (to_dict handled by WorkflowContext)

        # -- Enhanced memory: summary, goals, topic tracking ---------------
        self._update_memory_meta(state["memory"], state, question, answer, history)

        logger.info("Memory now holds %d messages", len(history))
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_memory_meta(
        memory: dict[str, Any],
        state: dict[str, object],
        question: str,
        answer: str,
        history: list[dict[str, str]],
    ) -> None:
        """Update memory with conversation metadata.

        Tracks:
          - current_goal from the session state or first user question
          - last_topic from the current user question
          - summary from recent messages (rule-based, no LLM)
          - conversation_type (single-turn vs. multi-turn)
        """
        memory["last_topic"] = question or ""

        # Capture current_goal from session state, or from the first user message
        ss = state.get("session_state")
        if isinstance(ss, SessionState) and ss.current_goal:
            memory["current_goal"] = ss.current_goal
        elif not memory.get("current_goal"):
            # Use the first substantive question as the inferred goal
            if question and len(question) > 4:
                memory["current_goal"] = question

        # Build a lightweight summary from the last 2 turns
        summary_parts = []
        recent = history[-6:] if len(history) > 6 else history[:]
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"]
            if len(content) > 60:
                content = content[:60] + "…"
            summary_parts.append(f"{role}: {content}")
        memory["summary"] = "\n".join(summary_parts) if summary_parts else ""

        # Determine conversation type
        if len(history) >= 4:
            memory["conversation_type"] = "multi_turn"
        elif len(history) >= 2:
            memory["conversation_type"] = "follow_up"
        else:
            memory["conversation_type"] = "single_turn"
