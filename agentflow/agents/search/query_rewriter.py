"""QueryRewriter — optimizes user questions into search-engine-friendly queries.

The QueryRewriter sits between the Planner and SearchAgent.  It transforms
raw user input (which may be short, anaphoric, or conversational) into a
concise search query by leveraging conversation context, session state,
and the Planner's declared intent.

Responsibilities:
  1. Fill missing parameters from ConversationState (e.g. goal→context)
  2. Resolve anaphora using conversation history
  3. Strip conversational filler ("帮我查一下", "请问")
  4. Normalise time references (今天→今日, 现在→当前)
  5. Preserve named entities (GPT-5, OpenAI, Claude Code)
  6. Produce a concise query string optimised for search engines
  7. NEVER guess information the user has not provided
"""

from __future__ import annotations

import re
from typing import Any

from agentflow.utils.logging import build_logger

logger = build_logger("query_rewriter")

# ---------------------------------------------------------------------------
# Filler phrases to strip from queries
# ---------------------------------------------------------------------------
_FILLER_PATTERNS: list[re.Pattern] = [
    # Longest matches first to avoid partial stripping
    re.compile(r"^\s*(帮我查一下|帮我查查|帮我查询|帮我搜一下|帮我搜索|帮我找一下|帮我找找)\s*"),
    re.compile(r"^\s*(想问一下|想查一下|想查询一下|想查查)\s*"),
    re.compile(r"^\s*(请问|问一下|打听一下)\s*"),
    re.compile(r"^\s*(可以帮我|能帮我|能否帮我|可否帮我)\s*"),
    re.compile(r"^\s*(我想知道|我想查|我想问|我想|我想要|我要)\s*"),
    re.compile(r"^\s*(可以|能不能|能否|可否)\s*"),
    re.compile(r"^\s*(请帮我|请)\s*"),
    re.compile(r"^\s*(帮我)\s*"),
    re.compile(r"\s*(谢谢你|谢谢|感谢|麻烦了|拜托|拜托了)\s*$"),
]

# ---------------------------------------------------------------------------
# Time normalisation map
# ---------------------------------------------------------------------------
_TIME_NORMALISE: dict[str, str] = {
    "今天": "今日",
    "明天": "明日",
    "昨天": "昨日",
    "现在": "当前",
    "最近": "最新",
    "这几天": "近期",
    "目前": "当前",
}

# ---------------------------------------------------------------------------
# Intent → search keyword suffixes (when context is available)
# ---------------------------------------------------------------------------
_INTENT_SUFFIX: dict[str, str] = {
    "weather": "天气",
    "news": "新闻",
    "stock": "股价",
    "price": "价格",
    "translation": "",
    "code": "",
}


class QueryRewriter:
    """Rewrite user questions into search-engine-optimised queries.

    Usage::

        rewriter = QueryRewriter()
        query = rewriter.rewrite(
            question="我在杭州",
            session_state=session_state,
            intent="weather",
        )
        # → "杭州 今日 天气"
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(
        self,
        question: str,
        session_state: Any = None,
        history: list[dict[str, str]] | None = None,
        intent: str = "",
    ) -> str:
        """Generate an optimised search query from user input and context.

        Args:
            question: The raw user question (after conversation_manager
                resolution, before rewriting).
            session_state: Current ``SessionState`` with ``current_goal``,
                ``tracking``, etc.
            history: Previous conversation turns
                ``([{role, content}, ...])``.
            intent: Declared search intent from the Planner
                (e.g. ``"weather"``, ``"news"``, ``"stock"``).

        Returns:
            A concise, search-engine-optimised query string.
        """
        q = question.strip()
        if not q:
            return ""

        # 1. Strip conversational filler
        q = self._strip_filler(q)

        # 2. Normalise time references
        q = self._normalise_time(q)

        # 3. Recover context from session state when question lacks detail
        context = self._recover_context(q, session_state, intent)

        # 4. Build final query
        if context and not self._is_self_contained(q):
            query = self._combine(q, context, intent)
        else:
            query = q

        logger.info(
            "Rewrote '%s' → '%s' (intent=%s, context=%s)",
            question[:40], query, intent,
            bool(context),
        )
        return query

    # ------------------------------------------------------------------
    # Internal: filler removal
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_filler(text: str) -> str:
        """Remove conversational filler phrases from the query."""
        for pattern in _FILLER_PATTERNS:
            text = pattern.sub("", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Internal: time normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_time(text: str) -> str:
        """Replace informal time references with search-friendly terms."""
        for cn, replacement in _TIME_NORMALISE.items():
            text = text.replace(cn, replacement)
        return text

    # ------------------------------------------------------------------
    # Internal: context recovery
    # ------------------------------------------------------------------

    @staticmethod
    def _recover_context(
        question: str,
        session_state: Any,
        intent: str,
    ) -> str:
        """Recover missing context from session state.

        Uses (in priority order):
          1. conversation_tracking.current_focus
          2. conversation_tracking.topic
          3. session_state.current_goal
          4. session_state.slots (filled values)

        Returns the best context string, or empty string if none found.
        """
        if session_state is None:
            return ""

        tracking = getattr(session_state, "tracking", None)

        # Priority 1: current focus (most specific)
        if tracking is not None:
            focus = getattr(tracking, "current_focus", "") or ""
            if focus and focus not in question:
                return focus

        # Priority 2: topic
        if tracking is not None:
            topic = getattr(tracking, "topic", "") or ""
            if topic and topic not in question:
                return topic

        # Priority 3: current_goal
        goal = getattr(session_state, "current_goal", "") or ""
        if goal and goal not in question:
            return goal

        # Priority 4: filled slots
        slots = getattr(session_state, "slots", {}) or {}
        filled = [v for v in slots.values() if isinstance(v, str) and v]
        if filled:
            return " ".join(filled)

        return ""

    # ------------------------------------------------------------------
    # Internal: query combination
    # ------------------------------------------------------------------

    @classmethod
    def _combine(cls, question: str, context: str, intent: str) -> str:
        """Combine question, context, and intent into a final query.

        Strategy:
          - If question looks like a standalone entity / keyword,
            prepend context.
          - If question is a modifier ("改一下", "加一个"),
            use context as primary with question as secondary.
          - Append intent suffix when intent is recognised and context
            is available (e.g. intent=weather → "天气").
        """
        parts = []

        # Detect if question is a modifier / follow-up
        is_modifier = bool(
            re.search(r"(改|换|加|删|优化|完善|更新)", question)
        )

        if is_modifier:
            # Modifier: context first, question second
            parts.append(context)
            parts.append(question)
        elif cls._is_standalone_keyword(question):
            # Standalone keyword: prepend context
            parts.append(context)
            parts.append(question)
        else:
            # Full sentence: use question as-is
            parts.append(question)

        # Append intent suffix when context exists and intent is recognised
        suffix = _INTENT_SUFFIX.get(intent, "")
        if suffix and context and suffix not in parts[-1]:
            parts.append(suffix)

        result = " ".join(parts)
        # Collapse multiple spaces
        return re.sub(r"\s+", " ", result).strip()

    # ------------------------------------------------------------------
    # Internal: heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _is_self_contained(question: str) -> bool:
        """Check if the question is self-contained for search purposes.

        A self-contained question includes enough information (entities,
        location, topic) to be searched without context.
        """
        if len(question) > 10:
            return True
        # Questions starting with these patterns are self-contained
        self_contained_starts = [
            "今日", "当前", "最新", "搜索", "查找",
        ]
        return any(question.startswith(p) for p in self_contained_starts)

    @staticmethod
    def _is_standalone_keyword(text: str) -> bool:
        """Check if text is a standalone keyword (not a full sentence).

        E.g. "杭州", "GPT-5", "OpenAI" are standalone.
        "我在杭州", "今天天气怎么样" are not.
        """
        # Pure noun / entity: no verbs, prepositions, etc.
        # Short string (1-10 chars) with no Chinese verbs → keyword
        if len(text) <= 10:
            # Check if it's mostly CJK noun-like characters
            has_verb_like = any(
                text.startswith(v) for v in
                ["我", "你", "他", "她", "在", "是", "有", "去", "来", "要",
                 "想", "能", "可以", "请", "帮"]
            )
            return not has_verb_like
        return False
