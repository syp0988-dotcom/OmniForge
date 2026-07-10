"""RewriteEngine — rule-based question rewriting for context understanding.

The RewriteEngine detects when a user's question is too short, anaphoric,
or ambiguous to stand alone — and rewrites it into a self-contained,
context-rich question that downstream agents (Router, Planner, Answer)
can process correctly.

This is the core of "Context Understanding": instead of passing raw
short inputs like "第二个" or "优化一下" through the pipeline, we enrich
them with the current conversation context.

Design:
  - Pure functions, no state.
  - Rule-based (no LLM calls) for speed and determinism.
  - Works with any SessionState and memory dict.
"""

from __future__ import annotations

import re
from typing import Any

from agentflow.conversation.context import ORDINAL_OPTION_PATTERNS
from agentflow.utils.logging import build_logger

logger = build_logger("rewrite_engine")

# -- Pattern sets -------------------------------------------------------------

# Patterns that indicate ordinal / option selection references
_ORDINAL_PATTERNS: list[re.Pattern] = list(ORDINAL_OPTION_PATTERNS) + [
    re.compile(r"^[1-9]$"),
    re.compile(r"^[1-9][0-9]?$"),
]

# Patterns that indicate modification intent
_MODIFIER_PATTERNS: list[re.Pattern] = [
    re.compile(r"改(一?下|成|为|进|善|良)"),
    re.compile(r"换(一?下|成|为)"),
    re.compile(r"优化(一?下)"),
    re.compile(r"完善(一?下)"),
    re.compile(r"更新(一?下)"),
    re.compile(r"重构(一?下)"),
    re.compile(r"重写"),
    re.compile(r"润色"),
    re.compile(r"精简"),
    re.compile(r"扩展"),
    re.compile(r"补充"),
]

# Patterns that indicate follow-up / elaboration requests
_FOLLOW_UP_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(继续|然后|然后呢|还有呢|接着说|继续说|展开|详细|详细一点|详细说说|具体一点|具体说说)\s*$"),
    re.compile(r"^\s*(为什么|为何|原因|理由|原理|怎么做到的)\s*"),
    re.compile(r"^\s*(解释|说明)一下\s*$"),
    re.compile(r"^\s*(后面|接下来|下一步|然后)"),
]

# Deictic / anaphoric references that require context
_DEICTIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(这个|那个|这|那|它|他|她)\s*$"),
    re.compile(r"^\s*(这个|那个|这)\s*(方案|方法|方式|代码|数据|结果|问题|功能|步骤|流程)\s*$"),
    re.compile(r"^\s*(数据|代码|结果)\s*$"),
    re.compile(r"^\s*那.*呢\s*$"),
    re.compile(r"^\s*(所以|那|那么)\s*"),
]

# Confirmation / agreement (don't rewrite, these are continue signals)
# Note: "继续" is NOT a confirmation — it's a command that needs context.
_CONFIRM_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(好的|好|嗯|是的|对|ok|可以|行|确认)\s*$", re.IGNORECASE),
]


class RewriteEngine:
    """Determine if and how a question should be rewritten with context."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def needs_rewrite(question: str) -> bool:
        """Check whether *question* needs context-aware rewriting.

        Returns ``True`` when the question is too short to stand alone or
        matches known anaphoric / deictic patterns.
        """
        q = question.strip()

        # Confirmations don't need rewriting (they are continue signals)
        if any(p.match(q) for p in _CONFIRM_PATTERNS):
            return False

        if _looks_like_new_task_request(q):
            return False

        # Ordinal references
        if any(p.match(q) for p in _ORDINAL_PATTERNS):
            return True

        # Modification intent
        if any(p.search(q) for p in _MODIFIER_PATTERNS):
            return True

        # Follow-up requests
        if any(p.match(q) for p in _FOLLOW_UP_PATTERNS):
            return True

        # Deictic references
        if any(p.match(q) for p in _DEICTIC_PATTERNS):
            return True

        if _looks_like_standalone_short_topic(q):
            return False

        # Very short inputs are ambiguous unless they look like a standalone
        # entity/topic and were handled above.
        if len(q) < 4:
            return True

        # General short input (< 15 chars without context markers)
        if len(q) < 15:
            # Named entities with mixed CJK + Latin chars are self-contained
            # e.g. "Python贪吃蛇", "GPT-4模型", "IDA反编译器"
            if re.search(r"[一-鿿].*[a-zA-Z0-9]|[a-zA-Z0-9].*[一-鿿]", q):
                return False
            # Pure CJK noun phrases (4+ chars, no modifier verbs) are self-contained
            # e.g. "机器学习入门", "数据结构", "数据分析方法"
            # But interrogative words signal a follow-up question, not a topic
            if re.fullmatch(r"[一-鿿]{4,}", q) and not re.search(r"[哪吗何怎]", q):
                return False
            # Check if it starts with a verb phrase that might be complete
            # Short complete questions like "你好" are OK
            if not any(p.search(q) for p in _MODIFIER_PATTERNS):
                # It's short but not anaphoric — check if self-contained
                if q.startswith(("你好", "你是谁", "你能", "今天", "帮我", "搜索",
                                 "查找", "查询", "翻译", "润色", "总结", "写一个",
                                 "生成", "介绍", "什么是", "解释", "说明")):
                    return False
                return True

        return False

    @staticmethod
    def rewrite(
        question: str,
        session_state: Any = None,
        memory: dict[str, Any] | None = None,
    ) -> str:
        """Rewrite *question* using context from *session_state* and *memory*.

        Args:
            question: The (possibly short / anaphoric) user input.
            session_state: Current ``SessionState`` (or any object with
                ``current_goal``, ``pending_options``, ``waiting_for``).
            memory: Memory dict (with ``current_goal``, ``last_topic``).

        Returns:
            A self-contained, context-enriched question string.
        """
        q = question.strip()

        if not q:
            return question

        # Gather context sources
        goal = ""
        waiting_for = ""
        pending_options: dict[str, str] = {}
        last_topic = ""
        tracking = None

        if session_state is not None:
            goal = getattr(session_state, "current_goal", "") or ""
            waiting_for = getattr(session_state, "waiting_for", "") or ""
            pending_options = getattr(session_state, "pending_options", {}) or {}
            tracking = getattr(session_state, "tracking", None)

        if isinstance(memory, dict):
            last_topic = memory.get("last_topic", "") or ""

        # Use what's available: session goal > memory goal > last_topic
        if not goal and isinstance(memory, dict):
            goal = memory.get("current_goal", "") or ""
        if not goal and last_topic:
            goal = last_topic

        # Priority chain: tracking.focus > tracking.topic > goal > waiting_for > last_topic
        context = goal or waiting_for or last_topic
        if tracking is not None:
            if tracking.current_focus:
                context = tracking.current_focus
            elif tracking.topic:
                context = tracking.topic

        # --- Ordinal / option selection ---
        if any(p.match(q) for p in _ORDINAL_PATTERNS):
            # If tracking has a focus, use it directly
            if tracking is not None and tracking.current_focus:
                logger.info("Rewrote ordinal '%s' with focus: %s", q, tracking.current_focus)
                return f"关于「{tracking.current_focus}」的选择：{q}"

            # Try to resolve against pending_options
            if pending_options and session_state is not None:
                resolved = getattr(session_state, "resolve_option", None)
                if resolved:
                    val = resolved(q)
                    if val:
                        logger.info("Rewrote ordinal '%s' → selection of '%s'", q, val)
                        if goal:
                            return f"请从当前任务「{goal}」中选择：{val}"
                        return f"请详细介绍之前提到的：{val}"

            # Ordinal but no options available — use context
            if context:
                logger.info("Rewrote ordinal '%s' with context: %s", q, context)
                return f"关于{context}，请介绍{q}相关内容"
            return question

        # --- Modification intent ---
        if any(p.search(q) for p in _MODIFIER_PATTERNS):
            if tracking is not None and tracking.current_focus:
                logger.info("Rewrote modifier '%s' with focus: %s", q, tracking.current_focus)
                return f"{q} 当前焦点：{tracking.current_focus}"
            if context:
                logger.info("Rewrote modifier '%s' → apply to: %s", q, context)
                return f"{q}当前任务：{context}"
            return question

        # --- Follow-up / elaboration ---
        if any(p.match(q) for p in _FOLLOW_UP_PATTERNS):
            if tracking is not None and tracking.current_focus:
                return f"关于「{tracking.current_focus}」，{q}"
            if context:
                logger.info("Rewrote follow-up '%s' with context: %s", q, context)
                return f"关于{context}，{q}"
            return question

        # --- Deictic reference ---
        if any(p.match(q) for p in _DEICTIC_PATTERNS):
            if tracking is not None and tracking.current_focus:
                return f"关于「{tracking.current_focus}」，用户说：{q}"
            if context:
                logger.info("Rewrote deictic '%s' → %s: %s", q, context, q)
                return f"关于{context}，用户说：{q}"
            return question

        # --- General short input ---
        if len(q) < 15 and context:
            if _looks_like_standalone_short_topic(q):
                return question
            logger.info("Rewrote short input '%s' with context: %s", q, context)
            return f"关于{context}，用户问：{q}"

        return question


def _looks_like_standalone_short_topic(text: str) -> bool:
    """Detect short standalone entities/topics that should not inherit context."""
    q = text.strip()
    if not q:
        return False
    if re.search(r"[？?！!，,。；;：:\s]", q):
        return False
    # Interrogative particles → this is a question, not a standalone topic
    if re.search(r"[哪吗何怎]", q):
        return False
    if re.search(r"[一-\u9fff].*[a-zA-Z0-9]|[a-zA-Z0-9].*[一-\u9fff]", q):
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,14}", q):
        return True
    if re.fullmatch(r"[一-\u9fff]{3,8}", q):
        return True
    return False


# Question suffixes that turn a task keyword into a status/context inquiry
_TASK_INQUIRY_PATTERNS: list[re.Pattern] = [
    re.compile(r"在哪里"),
    re.compile(r"在哪"),
    re.compile(r"了吗"),
    re.compile(r"好了吗"),
    re.compile(r"完了吗"),
    re.compile(r"有没有"),
    re.compile(r"怎么.*没有"),
    re.compile(r"没.*看到"),
    re.compile(r"找不到"),
]


def _looks_like_new_task_request(text: str) -> bool:
    """Detect explicit new tasks that should stay independent."""
    q = text.strip()
    if not q:
        return False

    lower = q.lower()
    if re.match(r"^(继续|接着|然后|再|把它|将它|这个|那个|上面|刚才|之前)", q):
        return False
    if re.match(r"^(continue|then|also|again|modify|change|update)\b", lower):
        return False

    starts = (
        "创建", "新建", "生成", "写一个", "写个", "写一份", "做一个", "做个",
        "开发", "实现", "制作", "搭建", "帮我创建", "帮我生成", "帮我写",
        "请创建", "请生成", "请写", "搜索", "查询", "介绍", "解释",
    )
    if q.startswith(starts):
        if any(p.search(q) for p in _TASK_INQUIRY_PATTERNS):
            return False
        return True

    return bool(
        re.match(r"^(create|build|generate|write|make|implement|develop|search|explain)\b", lower)
    )
