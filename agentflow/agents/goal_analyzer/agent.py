"""GoalAnalyzer — hybrid intent understanding (embedding + LLM fallback).

Architecture::

    User question
        │
        ├→ IntentIndex (embedding, ~10 ms, zero cost)
        │     ├→ ratio ≥ 2.0x → matched goal (bypass LLM)
        │     └→ ratio < 2.0x → LLM fallback (semantic understanding)
        │
        └→ Output: goal_analysis dict (same schema as before)

The embedding path handles ~30-50 % of queries instantly and for free.
Only ambiguous or mixed-intent queries incur an LLM call.
"""

from __future__ import annotations

import json
from typing import Any

from agentflow.agents.base import AgentProtocol
from agentflow.agents.goal_analyzer.intent_index import (
    INTENT_LABEL_TO_GOAL_TYPE,
    IntentIndex,
)
from agentflow.config.prompts import GOAL_ANALYZER_SYSTEM_PROMPT
from agentflow.services.llm_service import get_llm_service
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("goal_analyzer")

# Lazily-initialised singleton — the SentenceTransformer model is loaded on
# first use, so creating the index at import time is cheap.
_intent_index: IntentIndex | None = None


def _get_intent_index() -> IntentIndex:
    global _intent_index
    if _intent_index is None:
        _intent_index = IntentIndex()
    return _intent_index


# -- Mapping from intent label to default knowledge_source --------------------
# Embedding-matched intents use these rule-based defaults instead of calling
# the LLM to guess.  When the frontend toggle (future) passes an explicit
# knowledge_mode, that takes precedence.
_LABEL_KNOWLEDGE_SOURCE: dict[str, str] = {
    "coding": "general",
    "project": "general",
    "question": "general",
    "search": "general",
    "tool": "general",
    "chat": "general",
}

# -- Mapping from intent label to expected_outputs ---------------------------
_LABEL_EXPECTED_OUTPUTS: dict[str, list[str]] = {
    "coding": ["source_code"],
    "project": ["project", "source_code", "readme"],
    "question": ["answer"],
    "search": ["answer"],
    "tool": ["answer"],
    "chat": ["answer"],
}

# -- Mapping from intent label to priority -----------------------------------
_LABEL_PRIORITY: dict[str, str] = {
    "coding": "normal",
    "project": "high",
    "question": "normal",
    "search": "normal",
    "tool": "normal",
    "chat": "low",
}

# -- Compact LLM fallback prompt (only used for low-confidence queries) ------
SYSTEM_PROMPT = GOAL_ANALYZER_SYSTEM_PROMPT


class GoalAnalyzer(AgentProtocol):
    """Hybrid goal understanding: embedding first, LLM as safety net."""

    def __init__(self) -> None:
        self._llm = get_llm_service()

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        conversation_context = state.get("conversation_context")
        session_state = state.get("session_state")
        continue_mode = bool(state.get("_continue_mode", False))

        # Use the rewritten question when available — it carries conversation
        # context (e.g. "什么意思" → "关于帮我写文章，用户问：什么意思"),
        # which makes embedding matching far more accurate for follow-ups.
        rewritten = str(state.get("rewritten_question", "") or "")
        query_for_intent = rewritten if rewritten else question

        existing_goal = None
        if continue_mode and session_state:
            existing_goal = getattr(session_state, "current_goal", None) or (
                session_state.get("current_goal") if isinstance(session_state, dict) else None
            )

        goal = self._analyze_goal(
            question=question,
            query_for_intent=query_for_intent,
            conversation_context=conversation_context,
            continue_mode=continue_mode,
            existing_goal=existing_goal,
        )
        goal = _apply_source_mode(goal, str(state.get("source_mode", "auto") or "auto"))

        state["goal_analysis"] = goal
        state["category"] = goal.get("goal_type", "other")
        state["router"] = {
            "goal_type": goal.get("goal_type", "other"),
            "goal": goal.get("goal", ""),
            "confidence": goal.get("confidence", 0.0),
        }

        if goal.get("fallback"):
            state["_degraded"] = True
            state["_llm_error"] = "GoalAnalyzer: LLM 不可用，使用默认目标分析"

        logger.info(
            "Goal: type=%s knowledge=%s confidence=%.2f goal='%s'%s%s",
            goal.get("goal_type", "?"),
            goal.get("knowledge_source", "?"),
            goal.get("confidence", 0.0),
            goal.get("goal", "")[:60],
            " (fallback)" if goal.get("fallback") else "",
            " (embedding)" if goal.get("_embedding_match") else "",
        )
        return state

    def _analyze_goal(
        self,
        question: str,
        query_for_intent: str,
        conversation_context: Any = None,
        continue_mode: bool = False,
        existing_goal: str | None = None,
    ) -> dict[str, Any]:
        """Analyze user goal — embedding first, LLM as fallback."""
        if not question:
            return self._default_goal()

        # ── Path 1: Embedding match (fast, free, handles ~80 % of queries) ─
        if not continue_mode:
            result = _get_intent_index().match(query_for_intent)
            if result is not None:
                label, goal_type, confidence = result
                return self._build_goal_from_match(
                    question=question,
                    label=label,
                    goal_type=goal_type,
                    confidence=confidence,
                )

        # ── Path 2: Continue mode with existing goal ──
        if continue_mode and existing_goal:
            # Re-use the existing goal type — the user is still on the same task.
            # But we still need to understand the new sub-intent, so let the LLM
            # decide (it has the full conversation context).
            pass

        # ── Path 3: LLM fallback (ambiguous / mixed / continue-mode queries) ─
        return self._llm_analyze(question, conversation_context, continue_mode, existing_goal)

    # ------------------------------------------------------------------
    # Embedding match → goal dict
    # ------------------------------------------------------------------

    @staticmethod
    def _build_goal_from_match(
        question: str,
        label: str,
        goal_type: str,
        confidence: float,
    ) -> dict[str, Any]:
        """Build a ``goal_analysis`` dict from an embedding match."""
        return {
            "goal": question,
            "goal_type": goal_type,
            "knowledge_source": _LABEL_KNOWLEDGE_SOURCE.get(label, "general"),
            "expected_outputs": _LABEL_EXPECTED_OUTPUTS.get(label, ["answer"]),
            "priority": _LABEL_PRIORITY.get(label, "normal"),
            "confidence": round(confidence, 2),
            "_embedding_match": True,
        }

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _llm_analyze(
        self,
        question: str,
        conversation_context: Any = None,
        continue_mode: bool = False,
        existing_goal: str | None = None,
    ) -> dict[str, Any]:
        """Call LLM to analyze the user's goal (original behavior)."""
        ctx_parts = [f"用户输入：{question}"]

        if continue_mode and existing_goal:
            ctx_parts.append(f"\n这是继续对话。当前已有目标：{existing_goal}")
            ctx_parts.append("请根据新的用户输入，判断是继续原有目标还是转向新目标。")

        if conversation_context:
            if isinstance(conversation_context, dict):
                ctx_type = conversation_context.get("type", "")
                summary = conversation_context.get("summary", "")
                if ctx_type:
                    ctx_parts.append(f"\n对话类型：{ctx_type}")
                if summary:
                    ctx_parts.append(f"对话摘要：{summary}")
            else:
                cc_type = getattr(conversation_context, "type", "")
                cc_summary = getattr(conversation_context, "summary", "")
                if cc_type:
                    ctx_parts.append(f"\n对话类型：{cc_type}")
                if cc_summary:
                    ctx_parts.append(f"对话摘要：{cc_summary}")

        ctx_parts.append(
            "\n\n请分析用户的真实目标，输出 JSON 格式的分析结果。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(ctx_parts)},
        ]

        try:
            raw = self._llm.complete(messages=messages)
            parsed = self._parse_goal_json(raw)
            if parsed:
                return parsed
        except Exception as exc:
            logger.warning("GoalAnalyzer LLM call failed: %s", exc)

        return self._default_goal(question)

    def _parse_goal_json(self, raw: str) -> dict[str, Any] | None:
        """Extract JSON from LLM output."""
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for marker in ("```json", "```JSON", "```"):
            start = text.find(marker)
            if start == -1:
                continue
            content = text[start + len(marker):]
            end = content.rfind("```")
            if end != -1:
                content = content[:end]
            content = content.strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _default_goal(question: str = "") -> dict[str, Any]:
        """Fallback goal when LLM is unavailable."""
        return {
            "goal": question or "处理用户请求",
            "goal_type": "other",
            "knowledge_source": "hybrid",
            "expected_outputs": ["answer"],
            "priority": "normal",
            "confidence": 0.1,
            "fallback": True,
        }


def _apply_source_mode(goal: dict[str, Any], source_mode: str) -> dict[str, Any]:
    """Apply the user's explicit answer-source preference conservatively."""
    mode = source_mode if source_mode in {"auto", "web", "knowledge"} else "auto"
    if mode == "auto":
        goal["source_mode"] = "auto"
        return goal

    current_type = str(goal.get("goal_type", "other") or "other")
    informational_types = {"other", "question", "analysis", "document", "search"}
    if mode == "knowledge":
        if current_type in informational_types:
            goal["goal_type"] = "question" if current_type == "other" else current_type
            goal["knowledge_source"] = "local"
        goal["source_mode"] = "knowledge"
        return goal

    if mode == "web":
        if current_type in informational_types:
            goal["goal_type"] = "search"
            goal["knowledge_source"] = "general"
            expected = goal.get("expected_outputs")
            if not isinstance(expected, list) or "answer" not in expected:
                goal["expected_outputs"] = ["answer"]
        goal["source_mode"] = "web"
        return goal

    goal["source_mode"] = mode
    return goal
