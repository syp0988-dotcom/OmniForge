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
    FOLLOW_UP,
    NEW_TASK,
    OPTION_SELECTION,
    ORDINAL_OPTION_PATTERNS,
    QUESTION_REWRITE,
    WAITING_REPLY,
    ConversationContext,
)
from agentflow.conversation.rewrite import RewriteEngine
from agentflow.conversation.session_state import SessionState
from agentflow.conversation.state import ConversationState
from agentflow.utils.logging import build_logger

logger = build_logger("conversation_manager")

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

        # --- 0. Initialize conversation tracking ---
        if question and session_state.tracking is None:
            session_state.tracking = ConversationState()

        # --- 1. Pending option resolution ---
        if session_state.has_pending_options:
            resolved = session_state.resolve_option(question)
            if resolved:
                logger.info(
                    "Resolved option '%s' → '%s'",
                    original, resolved,
                )
                question = resolved
                # Update tracking BEFORE clearing pending state
                if session_state.tracking is not None:
                    ConversationManager._update_tracking_from_question(
                        original, session_state.tracking, session_state,
                    )
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
                if session_state.tracking is not None:
                    ConversationManager._update_tracking_from_question(
                        original, session_state.tracking, session_state,
                    )
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
                if session_state.tracking is not None:
                    ConversationManager._update_tracking_from_question(
                        original, session_state.tracking, session_state,
                    )
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
                if session_state.tracking is not None:
                    ConversationManager._update_tracking_from_question(
                        original, session_state.tracking, session_state,
                    )
                # If the session was waiting and the user provided substantive
                # new input (not a continue signal), resume so the full
                # Router → Planner → QueryRewriter pipeline can run with
                # context enrichment.
                if session_state.is_waiting:
                    session_state.resume()
                    session_state.status = "idle"
                    logger.info(
                        "Resumed from waiting with new input: '%s'", enriched
                    )
                return enriched

        # --- 5. Self-contained question while waiting: resume session ---
        if session_state.is_waiting and ConversationManager._is_self_contained(question):
            logger.info(
                "Self-contained question while waiting: '%s' → resuming", original
            )
            session_state.resume()
            session_state.status = "idle"

        # --- 6. Update conversation tracking (default path) ---
        if session_state.tracking is not None:
            ConversationManager._update_tracking_from_question(
                original, session_state.tracking, session_state,
            )
        return question

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
        # --- Always update tracking first (capture entities/summary) ---
        ConversationManager._update_tracking(session_state, answer)

        # Preserve current_goal before any early returns so follow-up
        # questions (e.g. "我在杭州" after "请告诉我城市") can be enriched
        # with conversation context by resolve_question and QueryRewriter.
        if not session_state.current_goal:
            question = state.get("question", "")
            if isinstance(question, str) and len(question) > 4:
                session_state.current_goal = question[:200]

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
        elif ss.is_waiting:
            ctx_type = WAITING_REPLY
        elif rewritten_question != original_question:
            ctx_type = QUESTION_REWRITE
        elif is_continue and ss.current_goal:
            ctx_type = FOLLOW_UP
        else:
            ctx_type = NEW_TASK

        # Extract simple entities, merging with tracking entities when available
        current_entities = ConversationManager._extract_entities(original_question)
        if ss.tracking is not None and ss.tracking.entities:
            seen = set(current_entities)
            merged = list(current_entities)
            for e in ss.tracking.entities:
                if e not in seen:
                    merged.append(e)
                    seen.add(e)
            entities = merged[:10]  # limit to avoid bloat
        else:
            entities = current_entities

        # Build summary from memory (fallback to tracking summary)
        summary = ""
        if isinstance(memory, dict):
            summary = memory.get("summary", "") or ""
        if not summary and ss.tracking is not None:
            summary = ss.tracking.summary

        last_topic = ""
        if isinstance(memory, dict):
            last_topic = memory.get("last_topic", "") or ""

        # Append tracking focus to current_goal when available
        current_goal = ss.current_goal or ""
        if ss.tracking is not None and ss.tracking.current_focus:
            if current_goal and ss.tracking.current_focus not in current_goal:
                current_goal = f"{current_goal}（焦点：{ss.tracking.current_focus}）"

        return ConversationContext(
            type=ctx_type,
            original_question=original_question,
            rewritten_question=rewritten_question,
            current_goal=current_goal,
            last_topic=last_topic,
            waiting_for=ss.waiting_for or "",
            entities=entities,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_tracking(session_state: SessionState, answer: str) -> None:
        """Update conversation tracking fields from the assistant's answer."""
        if session_state.tracking is None:
            return
        session_state.tracking.last_answer = answer
        answer_entities = ConversationManager._extract_entities(answer)
        for e in answer_entities:
            session_state.tracking.add_entity(e)
        if session_state.pending_options:
            for key, value in session_state.pending_options.items():
                session_state.tracking.add_entity(value)
        if answer:
            truncated = answer[:80] + "…" if len(answer) > 80 else answer
            session_state.tracking.summary = truncated

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        """Basic entity extraction from Chinese text.

        Returns words that look like key terms (2-6 CJK characters).
        """
        if not text:
            return []
        # Match sequences of CJK characters (common nouns / topics)
        terms = re.findall(r"[一-鿿]{3,6}", text)
        # Filter out common stop words
        stop_words = {"什么", "怎么", "为什么", "如何", "哪个", "这个", "那个",
                      "一下", "一个", "可以", "能够", "需要", "是否", "怎样"}
        return [t for t in terms if t not in stop_words]

    @staticmethod
    def _extract_topic(question: str, entities: set[str]) -> str:
        """Extract the main topic from a question given known entities.

        Priority:
          1. A known entity that appears in the question.
          2. The longest fresh entity extracted from the question.
          3. Empty string if nothing meaningful found.
        """
        if not question:
            return ""
        # 1. Check known entities mentioned in the question
        for e in sorted(entities, key=len, reverse=True):
            if e and e in question:
                return e
        # 2. Extract fresh entities, take the longest
        fresh = ConversationManager._extract_entities(question)
        if fresh:
            return max(fresh, key=len)
        # 3. Fallback: first 12 chars if question is long enough
        return question[:12].strip() if len(question) > 4 else ""

    @staticmethod
    def _update_focus(
        question: str,
        session_state: SessionState,
        tracking: ConversationState,
    ) -> str:
        """Detect ordinal/digit selection and update tracking focus.

        Returns the new focus value (or current focus if unchanged).
        """
        if not question or not tracking:
            return tracking.current_focus if tracking else ""

        # Ordinal: "第二个", "选项一", "方案三", "步骤四", "三"
        ordinal_match = any(
            p.match(question.strip()) for p in ORDINAL_OPTION_PATTERNS
        )
        if ordinal_match or re.match(r"^\d+$", question.strip()):
            if session_state.has_pending_options:
                resolved = session_state.resolve_option(question)
                if resolved:
                    tracking.set_focus(resolved)
                    return resolved
            # No pending options: build combined focus
            if tracking.current_focus:
                base = re.sub(r"[\d一二三四五六七八九十]+$", "", tracking.current_focus)
                if base:
                    tracking.set_focus(base + question.strip())
                    return tracking.current_focus

        # Check if question directly references the current focus
        if tracking.current_focus and tracking.current_focus in question:
            return tracking.current_focus

        return tracking.current_focus

    @staticmethod
    def _update_tracking_from_question(
        question: str,
        tracking: ConversationState,
        session_state: SessionState,
    ) -> None:
        """Update tracking state based on the current user question.

        Called during ``resolve_question`` to accumulate:
          - Entities from the question
          - Topic extraction
          - Focus updates from ordinal/digit selection
        """
        if not question or not tracking:
            return

        # 1. Merge fresh entities
        fresh = ConversationManager._extract_entities(question)
        for e in fresh:
            tracking.add_entity(e)

        # 2. Update topic if not yet set, or if question mentions a new one
        topic = ConversationManager._extract_topic(question, tracking.entities)
        if topic:
            tracking.topic = topic

        # 3. Update focus based on ordinal/digit selection
        ConversationManager._update_focus(question, session_state, tracking)

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
        """Enrich a short/anaphoric question with session context.

        When tracking is available, uses structured fields
        (topic, focus) for more precise enrichment.
        """
        if ss.tracking is not None and ss.tracking.topic:
            parts = [f"当前目标：{ss.current_goal}"] if ss.current_goal else []
            if ss.tracking.topic:
                parts.append(f"话题：{ss.tracking.topic}")
            if ss.tracking.current_focus:
                parts.append(f"当前焦点：{ss.tracking.current_focus}")
            if parts:
                return f"{question}\n\n[会话上下文]\n{'、'.join(parts)}"

        # Fallback: plain string context
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
        option_patterns: list[re.Pattern] = list(ORDINAL_OPTION_PATTERNS) + [
            re.compile(r"^\d+$"),
        ]
        return any(p.match(question.strip()) for p in option_patterns)
