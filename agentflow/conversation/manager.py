"""ConversationManager — orchestrates conversation state across turns.

The ConversationManager is the **first node** in the workflow.  It decides
whether the current turn is a *continuation* of an existing task (and thus
bypasses Router/Planner/Tools) or a *new task* that needs full planning.

Key capabilities:
  - **Pending-option resolution**: "选项一" → resolved real value
  - **Slot filling**: "北京" → ``slots.city = "北京"``
  - **Anaphora detection**: "改成 Java" → understands "it" = current goal
  - **Continue-bypass**: When waiting_for is set, skips Router/Planner
  - **Question rewriting**: Short inputs enriched with conversation context
  - **Conversation context**: Structured context for downstream agents
"""

from __future__ import annotations

import re
from typing import Any

from agentflow.conversation.context import (
    CLARIFICATION,
    FOLLOW_UP,
    NEW_TASK,
    OPTION_SELECTION,
    QUESTION_REWRITE,
    WAITING_REPLY,
    ConversationContext,
)
from agentflow.conversation.rewrite import RewriteEngine
from agentflow.conversation.session_state import SessionState
from agentflow.utils.logging import build_logger

logger = build_logger("conversation_manager")

# Chinese ordinal characters → digit mapping
_ORDINAL_MAP: dict[str, str] = {
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
}

# Short commands that signal "continue the current task"
_CONTINUE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*继续\s*$"),
    re.compile(r"^\s*好的\s*$"),
    re.compile(r"^\s*好\s*$"),
    re.compile(r"^\s*嗯\s*$"),
    re.compile(r"^\s*是的\s*$"),
    re.compile(r"^\s*对\s*$"),
    re.compile(r"^\s*ok\s*$", re.IGNORECASE),
]

# Anaphora patterns — user refers to the current task indirectly
_ANAPHORA_PATTERNS: list[re.Pattern] = [
    re.compile(r"改(一?下|成|为)"),   # 改成 Java, 改一下
    re.compile(r"换(一?下|成|为)"),   # 换成 Python
    re.compile(r"继续.*"),            # 继续之前的话题
    re.compile(r"按照刚[才刚].*"),    # 按照刚才那个
    re.compile(r"(那个|这个|它|他)"),  # 那个方案, 这个
    re.compile(r"^\s*(第一|第二|第三|第四|第五|第六|第七|第八|第九|第十)\s*$"),
]


class ConversationManager:
    """Manages conversation runtime state — decides *how* to process input.

    This class is stateless (all state lives in ``SessionState``).
    It provides pure functions for:
      - Checking if a continuation is in progress
      - Resolving user input against the current session state
      - Generating the enriched question for downstream nodes
      - Rewriting short/anaphoric questions with context
      - Building structured conversation context
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def should_continue(session_state: SessionState) -> bool:
        """Check whether the current session has an active task.

        Returns ``True`` when the system is waiting for user input on
        an existing task — the new question should continue that task
        rather than starting a fresh planning cycle.
        """
        return (
            session_state.is_waiting
            or bool(session_state.current_goal)
        ) and session_state.status != "idle"

    @staticmethod
    def resolve_question(
        question: str,
        session_state: SessionState,
    ) -> str:
        """Resolve the user's raw input against the current session state.

        This handles:
          1. Pending option resolution (选项一 → real value)
          2. Slot filling (北京 → slots.city = "北京")
          3. Short anaphora (继续, 好的)
          4. Modification intent (改成 Java)

        Returns the *resolved question* string for downstream nodes.
        The session_state is mutated in place.
        """
        original = question
        question = question.strip()

        # --- 1. Pending option resolution ---
        if session_state.has_pending_options:
            resolved = session_state.resolve_option(question)
            if resolved:
                logger.info(
                    "Resolved option '%s' → '%s'",
                    original, resolved,
                )
                question = resolved
                # After resolving an option, clear the pending state
                session_state.pending_options.clear()
                session_state.resume()
                return question

        # --- 2. Continue/short confirmation while waiting ---
        if session_state.is_waiting:
            matched_continue = any(
                p.match(question) for p in _CONTINUE_PATTERNS
            )
            if matched_continue:
                logger.info(
                    "Continue signal detected while waiting for '%s'",
                    session_state.waiting_for,
                )
                # Use the waiting_for context as the resolved question
                question = session_state.waiting_for
                session_state.resume()
                return question

            # --- 3. Slot filling ---
            if session_state.has_unfilled_slots:
                # Find the first empty slot and try to fill it
                for slot_name, slot_val in session_state.slots.items():
                    if slot_val == "":
                        session_state.fill_slot(slot_name, question)
                        logger.info("Filled slot '%s' = '%s'", slot_name, question)
                        break

                # Check if all slots are now filled
                if not session_state.has_unfilled_slots:
                    session_state.resume()
                # Build enriched question from slot context
                enriched = ConversationManager._build_slot_context(session_state)
                if enriched:
                    return enriched
                return question

        # --- 4. Anaphora / modification detection ---
        if session_state.current_goal and not ConversationManager._is_self_contained(question):
            enriched = ConversationManager._enrich_with_context(
                question, session_state
            )
            if enriched != question:
                logger.info(
                    "Anaphora resolved: '%s' → '%s'",
                    original, enriched,
                )
                return enriched

        return question

    @staticmethod
    def build_continue_context(session_state: SessionState) -> dict[str, Any]:
        """Build additional context dict for the continue-flow.

        Returns a dict with keys that downstream nodes (AnswerAgent) can use:
          - ``session_context``: human-readable string
          - ``_continue_mode``: ``True`` (signals skip Router/Planner)
        """
        return {
            "_continue_mode": True,
            "session_context": str(session_state),
        }

    @staticmethod
    def finalize_turn(
        state: dict[str, Any],
        session_state: SessionState,
        answer: str,
    ) -> None:
        """Update session_state after a turn completes.

        Called by the MemoryAgent (or workflow end) to persist state.
        Heuristics:
          - If the answer contains numbered options (1. ... 2. ...),
            capture them as pending_options.
          - If the answer asks for input, set waiting_for.
          - Otherwise, mark as idle after a new task completes.
        """
        # Detect pending options from the answer text
        options = ConversationManager._extract_options(answer)
        if options:
            session_state.pending_options = options
            session_state.start_waiting("选择一个选项")
            logger.info("Detected %d pending options in answer", len(options))
            return

        # Detect question-like answers that need user input
        if ConversationManager._is_asking_for_input(answer):
            session_state.start_waiting("提供更多信息")
            logger.info("Answer is asking for user input")
            return

        # If there's a current_goal set from before, keep it alive
        # Only reset when a brand-new task completes with a final answer
        if session_state.status == "processing":
            session_state.resume()
            session_state.status = "idle"

    @staticmethod
    def rewrite_question(
        question: str,
        session_state: SessionState | None = None,
        memory: dict[str, Any] | None = None,
    ) -> str:
        """Rewrite a short/anaphoric question with conversation context.

        Delegates to ``RewriteEngine`` for rule-based rewriting.

        Args:
            question: The user's raw (possibly short) input.
            session_state: Current session state for context.
            memory: Memory dict with ``current_goal``, ``last_topic``.

        Returns:
            A self-contained, context-enriched question.
        """
        if not question or not RewriteEngine.needs_rewrite(question):
            return question
        return RewriteEngine.rewrite(question, session_state, memory)

    @staticmethod
    def build_conversation_context(
        original_question: str,
        rewritten_question: str,
        session_state: SessionState | None = None,
        memory: dict[str, Any] | None = None,
    ) -> ConversationContext:
        """Build a structured ``ConversationContext`` for the current turn.

        Determines the turn type based on session state and the relationship
        between original and rewritten questions.

        Args:
            original_question: Raw user input.
            rewritten_question: After resolution + rewriting.
            session_state: Current ``SessionState``.
            memory: Memory dict from MemoryAgent.

        Returns:
            A ``ConversationContext`` with type, enriched question, entities.
        """
        ss = session_state or SessionState()
        is_continue = ConversationManager.should_continue(ss)

        # Determine turn type
        # Option selection: original matches ordinal pattern + session has ongoing goal
        if ConversationManager._is_option_selection(original_question, ss):
            ctx_type = OPTION_SELECTION
        elif ss.has_pending_options and is_continue:
            ctx_type = OPTION_SELECTION
        elif ss.is_waiting:
            ctx_type = WAITING_REPLY
        elif rewritten_question != original_question:
            ctx_type = QUESTION_REWRITE
        elif is_continue and ss.current_goal:
            ctx_type = FOLLOW_UP
        else:
            ctx_type = NEW_TASK

        # Extract simple entities (Chinese words that look meaningful)
        entities = ConversationManager._extract_entities(original_question)

        # Build summary from memory
        summary = ""
        if isinstance(memory, dict):
            summary = memory.get("summary", "") or ""

        last_topic = ""
        if isinstance(memory, dict):
            last_topic = memory.get("last_topic", "") or ""

        return ConversationContext(
            type=ctx_type,
            original_question=original_question,
            rewritten_question=rewritten_question,
            current_goal=ss.current_goal or "",
            last_topic=last_topic,
            waiting_for=ss.waiting_for or "",
            entities=entities,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        """Basic entity extraction from Chinese text.

        Returns words that look like key terms (2-6 CJK characters).
        """
        if not text:
            return []
        # Match sequences of CJK characters (common nouns / topics)
        terms = re.findall(r"[一-鿿]{2,6}", text)
        # Filter out common stop words
        stop_words = {"什么", "怎么", "为什么", "如何", "哪个", "这个", "那个",
                      "一下", "一个", "可以", "能够", "需要", "是否", "怎样"}
        return [t for t in terms if t not in stop_words]

    @staticmethod
    def _is_self_contained(question: str) -> bool:
        """Check if the question is self-contained (doesn't need context).

        A self-contained question mentions a specific topic or action
        that doesn't obviously refer to the current task.
        """
        # Questions with specific entities are self-contained
        if len(question) > 15:
            return True
        # Questions that look like new topics
        self_contained_patterns = [
            re.compile(r"^(介绍|什么是|解释|说明|帮我|写一个|生成)"),
            re.compile(r"^(搜索|查找|查询|找一下)"),
            re.compile(r"^(翻译|润色|改写|总结)"),
            re.compile(r"^(你好|你是谁|你能)"),
            re.compile(r"今天.*(天气|新闻)"),
        ]
        return any(p.match(question) for p in self_contained_patterns)

    @staticmethod
    def _enrich_with_context(question: str, ss: SessionState) -> str:
        """Enrich a short/anaphoric question with session context."""
        context = str(ss)
        if not context or context == "(无活跃任务)":
            return question
        return f"{question}\n\n[会话上下文]\n{context}"

    @staticmethod
    def _build_slot_context(ss: SessionState) -> str:
        """Build an enriched question from slot state (slot filling flow)."""
        parts = [f"任务：{ss.current_goal}"]
        if ss.current_task:
            parts.append(f"当前操作：{ss.current_task}")
        filled = {k: v for k, v in ss.slots.items() if v}
        if filled:
            parts.append(f"已获取参数：{filled}")
        if ss.has_unfilled_slots:
            empty = [k for k, v in ss.slots.items() if v == ""]
            parts.append(f"仍需提供：{'、'.join(empty)}")
        return "\n".join(parts)

    @staticmethod
    def _extract_options(text: str) -> dict[str, str]:
        """Extract numbered options from answer text.

        Detects patterns like:
           1. 儿童教育
           2. 公共卫生
        """
        options: dict[str, str] = {}
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            m = re.match(r"^(\d+)[.、\)]\s*(.+)", line)
            if m:
                key = m.group(1)
                value = m.group(2).strip()
                # Filter out non-option lines (too long, code, etc.)
                if 1 <= len(value) <= 60 and not value.startswith(("```", "http")):
                    options[key] = value
        # Only return if we found at least 2 options in a contiguous block
        if len(options) >= 2:
            keys = sorted(options.keys(), key=int)
            expected = set(str(i) for i in range(int(keys[0]), int(keys[-1]) + 1))
            if set(keys) == expected:
                return options
        return {}

    @staticmethod
    def _is_asking_for_input(text: str) -> bool:
        """Detect if the answer is asking the user for input."""
        question_patterns = [
            r"请选择",
            r"请告诉我",
            r"请输入",
            r"请提供",
            r"请问.*[？?]",
            r"选择.*[？?]",
            r"你.*想.*[？?]",
            r"哪[个一].*[？?]",
            r"[？?]\s*$",
        ]
        return any(re.search(p, text) for p in question_patterns)

    @staticmethod
    def _is_option_selection(question: str, ss: SessionState) -> bool:
        """Detect if the user's input is selecting from pending options.

        Checks the original question against ordinal/selection patterns.
        This is needed because pending_options may have been cleared by
        ``resolve_question`` before ``build_conversation_context`` runs.
        """
        if not question or not ss.current_goal:
            return False
        option_patterns = [
            re.compile(r"选项[一二三四五六七八九十]"),
            re.compile(r"第[一二三四五六七八九十]个"),
            re.compile(r"方案[一二三四五六七八九十]"),
            re.compile(r"步骤[一二三四五六七八九十]"),
            re.compile(r"^[一二三四五六七八九十]$"),
            re.compile(r"^\d+$"),
        ]
        return any(p.match(question.strip()) for p in option_patterns)
