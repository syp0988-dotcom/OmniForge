"""Answer Agent — finalizes agent outputs into polished user-facing responses."""

from __future__ import annotations

from agentflow.services.llm_service import get_llm_service
from agentflow.utils.logging import build_logger

logger = build_logger("answer")


class AnswerAgent:
    """Produce a polished, user-facing answer from workflow context.

    Responsibilities:
      - Collect prepared context (question, history, knowledge, search)
      - Build a minimal prompt from that context
      - Call the LLM to generate a natural final answer
      - Clean the raw LLM output

    This agent does NOT:
      - Route queries, decide tool usage, or produce workflow logs
      - Impose rigid output templates -- format is left to the LLM
    """

    MAX_HISTORY_TURNS = 8

    # ------------------------------------------------------------------
    # Public API (preserved interface)
    # ------------------------------------------------------------------

    def run(self, state: dict[str, object]) -> dict[str, object]:
        """Produce a final answer from the workflow state."""
        question = str(state.get("question", ""))
        category = state.get("category", "reasoning")
        search_results = state.get("search_results", [])
        knowledge_context = state.get("knowledge_context", "")
        memory = state.get("memory", {})

        llm_service = get_llm_service()
        logger.info("Formatting answer for category: %s", category)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
        ]

        # Limited conversation history window
        messages.extend(self._build_history(memory))

        # Current turn: user prompt with full context
        messages.append({
            "role": "user",
            "content": self.build_prompt(
                category, question, search_results, knowledge_context
            ),
        })

        answer = llm_service.complete(messages=messages)
        state["answer"] = self.clean_answer(answer)
        return state

    def build_prompt(
        self,
        category: str,
        question: str,
        search_results: object,
        knowledge_context: str = "",
    ) -> str:
        """Build the user prompt for the current turn.

        Preserved interface -- delegates to internal builder.
        """
        return self._build_user_prompt(
            question, category, search_results, knowledge_context
        )

    def format_search_results(self, results: object) -> str:
        """Format search results for inclusion in the prompt.

        Preserved interface -- delegates to internal formatter.
        """
        return self._format_search_context(results)

    @staticmethod
    def clean_answer(text: str) -> str:
        """Remove residual noise from the raw LLM output."""
        return text.strip()

    # ------------------------------------------------------------------
    # Internal: prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt() -> str:
        """Minimal system prompt -- identity only, no rule lists."""
        return (
            "你是一个专业、准确的 AI 助手。"
            "请根据提供的上下文回答用户问题"
        )

    @staticmethod
    def _build_history(memory: object) -> list[dict[str, str]]:
        """Extract a limited window of recent conversation history."""
        if not isinstance(memory, dict):
            return []
        history = memory.get("history", [])
        if not isinstance(history, list):
            return []
        # Each turn = user + assistant, so keep last (turn_limit * 2) messages
        recent = history[-(AnswerAgent.MAX_HISTORY_TURNS * 2):]
        return [
            {"role": m["role"], "content": m["content"]}
            for m in recent
            if isinstance(m, dict) and "role" in m and "content" in m
        ]

    def _build_user_prompt(
        self,
        question: str,
        category: str,
        search_results: object,
        knowledge_context: str = "",
    ) -> str:
        """Stack available context blocks into a clean, minimal user prompt."""
        parts: list[str] = [f"用户问题：{question}"]

        # Identity questions: direct answer, no extra context needed
        if category == "identity":
            parts.append(
                "请直接回答身份问题。如果无法确认具体模型部署信息，"
                "请回答：我是当前系统配置的大语言模型助手。"
            )
            return "\n\n".join(parts)

        # Knowledge context (when available and meaningful)
        if knowledge_context and len(knowledge_context) > 20:
            parts.append(f"知识库资料：\n{knowledge_context}")

        # Search context (when available)
        if category == "search" and search_results:
            parts.append(
                f"搜索结果：\n{self._format_search_context(search_results)}"
            )

        # Single, short instruction -- no rule lists
        parts.append("请根据以上内容回答用户问题。")
        return "\n\n".join(parts)

    @staticmethod
    def _format_search_context(results: object) -> str:
        """Format search results as structured blocks the LLM can easily parse.

        Each result gets its own block with explicit labels making it much
        easier for the model to consume than flat concatenation.
        """
        if not isinstance(results, list):
            return ""
        blocks: list[str] = []
        for i, item in enumerate(results, 1):
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            snippet = item.get("snippet", item.get("content", "")).strip()
            url = item.get("url", "").strip()
            block = f"搜索结果 {i}"
            if title:
                block += f"\n标题：{title}"
            if snippet:
                block += f"\n摘要：{snippet}"
            if url:
                block += f"\n链接：{url}"
            blocks.append(block)
        return "\n\n".join(blocks)
