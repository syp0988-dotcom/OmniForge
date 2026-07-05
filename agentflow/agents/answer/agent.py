"""Answer Agent — finalizes agent outputs into polished user-facing responses.

Phase 7 enhancement:
  - ContextBuilder organizes all context sources into a clear prompt
  - Enhanced system prompt tells the model this is a continuing conversation
  - conversation_context provides structured turn type information
  - session_context provides the active task state
  - Memory summary + history are formatted for LLM comprehension
"""

from __future__ import annotations

from agentflow.services.llm_service import get_llm_service
from agentflow.utils.logging import build_logger

logger = build_logger("answer")

# -- Context block separator -------------------------------------------------

_HISTORY_SEPARATOR = "-" * 40


class ContextBuilder:
    """Builds structured prompt context from workflow state.

    Organizes all available context sources (session, memory, conversation,
    knowledge, search) into clear sections so the LLM can easily understand
    the conversation history and current request — even when the user's
    question is short or anaphoric.
    """

    def __init__(self, state: dict[str, object]) -> None:
        self.question = str(state.get("question", ""))
        self.category = str(state.get("category", "reasoning"))
        self.memory = state.get("memory", {})
        self.session_context = str(state.get("session_context", ""))
        self.conversation_context = state.get("conversation_context")
        self.is_continue = bool(state.get("_continue_mode", False))
        self.search_results = state.get("search_results", [])
        self.knowledge_context = str(state.get("knowledge_context", ""))
        self.rewritten_question = str(state.get("rewritten_question", ""))

    def build_system_prompt(self) -> str:
        """Build the system prompt with conversation awareness."""
        parts = [
            "你是一个专业、准确的 AI 助手 OmniForge。",
        ]

        if self.is_continue:
            parts.append(
                "当前是连续对话，用户的消息可能无法独立理解（例如「继续」「第二个」"
                "「优化一下」「展开」等）。请结合会话上下文自动理解用户意图，"
                "不要要求用户重新描述。如果历史信息足够确定用户意图，请直接回答。"
            )

        parts.append("请根据提供的上下文回答用户问题。")
        return " ".join(parts)

    def build_user_prompt(self) -> str:
        """Assemble all context into a single structured user prompt."""
        blocks: list[str] = []

        # --- Conversation summary (from MemoryAgent) ---
        if isinstance(self.memory, dict):
            summary = self.memory.get("summary", "")
            if summary:
                blocks.append(f"对话摘要：\n{summary}")

        # --- Session context (active task state) ---
        if self.session_context:
            blocks.append(f"当前会话：\n{self.session_context}")

        # --- Conversation context (turn type + rewritten) ---
        ctx_str = self._format_conversation_context()
        if ctx_str:
            blocks.append(ctx_str)

        # --- Conversation history (formatted as text) ---
        history_text = self._format_history()
        if history_text:
            blocks.append(history_text)

        # --- Knowledge context ---
        if self.knowledge_context and len(self.knowledge_context) > 20:
            blocks.append(f"知识库资料：\n{self.knowledge_context}")

        # --- Search results ---
        if self.category == "search" and self.search_results:
            blocks.append(
                f"搜索结果：\n{AnswerAgent._format_search_context(self.search_results)}"
            )

        # --- The actual question ---
        if self.is_continue and self.rewritten_question:
            # In continue mode, use the rewritten question to make intent clear
            blocks.append(f"用户问题：{self.rewritten_question}")
        else:
            blocks.append(f"用户问题：{self.question}")

        # --- Identity override ---
        if self.category == "identity":
            blocks.append(
                "请直接回答身份问题。如果无法确认具体模型部署信息，"
                "请回答：我是当前系统配置的大语言模型助手。"
            )
        else:
            blocks.append("请根据以上内容回答用户问题。")

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Internal formatting
    # ------------------------------------------------------------------

    def _format_conversation_context(self) -> str:
        """Format conversation_context into a readable string."""
        cc = self.conversation_context
        if not cc:
            return ""

        # cc might be a dict or ConversationContext
        if isinstance(cc, dict):
            ctx_type = cc.get("type", "")
            rewritten = cc.get("rewritten_question", "")
            entities = cc.get("entities", [])
        else:
            ctx_type = getattr(cc, "type", "")
            rewritten = getattr(cc, "rewritten_question", "")
            entities = getattr(cc, "entities", [])

        parts = []
        if ctx_type:
            parts.append(f"对话类型：{ctx_type}")
        if rewritten:
            parts.append(f"完整意图：{rewritten}")
        if entities:
            parts.append(f"关键实体：{'、'.join(entities)}")
        return "对话上下文：\n" + "\n".join(parts) if parts else ""

    def _format_history(self) -> str:
        """Format conversation history as readable text blocks."""
        if not isinstance(self.memory, dict):
            return ""
        history = self.memory.get("history", [])
        if not isinstance(history, list) or len(history) < 2:
            return ""

        # Show last N turns as structured text
        max_turns = 6
        recent = history[-(max_turns * 2):]

        lines = ["历史对话："]
        for msg in recent:
            if not isinstance(msg, dict):
                continue
            role_label = "用户" if msg.get("role") == "user" else "助手"
            content = str(msg.get("content", ""))
            lines.append(f"{role_label}：{content}")
            lines.append(_HISTORY_SEPARATOR)

        return "\n".join(lines)


class AnswerAgent:
    """Produce a polished, user-facing answer from workflow context.

    Responsibilities:
      - Collect prepared context (question, history, knowledge, search)
      - Build a minimal prompt from that context
      - Call the LLM to generate a natural final answer
      - Clean the raw LLM output

    Phase 7: Uses ``ContextBuilder`` for structured prompt assembly
    and provides rich context awareness for continuing conversations.
    """

    MAX_HISTORY_TURNS = 8

    # ------------------------------------------------------------------
    # Public API (preserved interface)
    # ------------------------------------------------------------------

    def run(self, state: dict[str, object]) -> dict[str, object]:
        """Produce a final answer from the workflow state."""
        is_continue = bool(state.get("_continue_mode", False))

        llm_service = get_llm_service()
        logger.info(
            "Formatting answer for category: %s (continue_mode=%s)",
            state.get("category", "reasoning"), is_continue,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt(is_continue)},
        ]

        # Build full context using ContextBuilder
        builder = ContextBuilder(state)
        user_prompt = builder.build_user_prompt()

        # Limited conversation history window (backward compat)
        messages.extend(self._build_history(state.get("memory", {})))

        # Current turn: full context prompt
        messages.append({"role": "user", "content": user_prompt})

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

        Preserved interface for backward compatibility.
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
    def _system_prompt(continue_mode: bool = False) -> str:
        """System prompt — identity with conversation awareness."""
        if continue_mode:
            return (
                "你是一个专业、准确的 AI 助手。这是一次连续对话，"
                "用户的消息可能很短或依赖上下文（例如「继续」「第二个」「优化一下」"
                "「展开」「为什么」）。请结合会话上下文（当前目标、任务、步骤）"
                "自动理解用户意图并继续完成。不要要求用户重新描述。"
            )
        return (
            "你是一个专业、准确的 AI 助手。"
            "请根据提供的上下文回答用户问题。"
        )

    @staticmethod
    def _build_history(memory: object) -> list[dict[str, str]]:
        """Extract a limited window of recent conversation history."""
        if not isinstance(memory, dict):
            return []
        history = memory.get("history", [])
        if not isinstance(history, list):
            return []
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

        if category == "identity":
            parts.append(
                "请直接回答身份问题。如果无法确认具体模型部署信息，"
                "请回答：我是当前系统配置的大语言模型助手。"
            )
            return "\n\n".join(parts)

        if knowledge_context and len(knowledge_context) > 20:
            parts.append(f"知识库资料：\n{knowledge_context}")

        if category == "search" and search_results:
            parts.append(
                f"搜索结果：\n{self._format_search_context(search_results)}"
            )

        parts.append("请根据以上内容回答用户问题。")
        return "\n\n".join(parts)

    @staticmethod
    def _format_search_context(results: object) -> str:
        """Format search results as structured blocks the LLM can easily parse."""
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
