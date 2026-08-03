"""Layered-memory history compression.

Instead of dropping old turns when a conversation grows, the system keeps:

  - **Working memory**: the most recent ``min_keep_messages`` turns, verbatim.
  - **Rolling summary**: older turns compressed into one summary string
    (LLM-based when available, deterministic fallback otherwise).

The compressed summary is injected into prompts via ``memory["summary"]``
while the recent window stays available for exact context.
"""

from __future__ import annotations

import logging
from typing import Any

from agentflow.config.settings import settings
from agentflow.services.llm_service import estimate_tokens, get_llm_service

logger = logging.getLogger("conversation.compression")

_COMPRESSION_SYSTEM_PROMPT = (
    "你是对话记忆压缩器。请把用户提供的早期对话压缩成一段中文摘要，"
    "保留：用户的目标、关键事实与实体、未完成事项。"
    "只输出摘要本身，不超过 200 字。"
)


def count_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate the total token count of a message list."""
    if not messages:
        return 0
    text = "\n".join(
        str(m.get("content", ""))
        for m in messages
        if isinstance(m, dict) and m.get("content")
    )
    return estimate_tokens(text)


def _transcript_text(messages: list[dict[str, Any]], max_chars: int = 12000) -> str:
    """Format dropped messages as a bounded transcript for summarization."""
    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = "用户" if msg.get("role") == "user" else "助手"
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if len(content) > 300:
            content = content[:300] + "…"
        lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    return text[-max_chars:]


def _llm_summarize(dropped: list[dict[str, Any]], llm: Any | None) -> str | None:
    """Try to summarize *dropped* with the LLM. Returns None on any failure."""
    if llm is None:
        return None
    transcript = _transcript_text(dropped)
    if not transcript:
        return None
    try:
        raw = llm.complete(
            messages=[
                {"role": "system", "content": _COMPRESSION_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            node_name="history_compress",
            max_tokens=400,
        )
        summary = (raw or "").strip()
        if len(summary) < 5:
            return None
        return summary[:500]
    except Exception as exc:  # LLM unavailable -> deterministic fallback
        logger.debug("LLM history compression failed: %s", exc)
        return None


def _deterministic_summary(dropped: list[dict[str, Any]]) -> str:
    """Rule-based fallback: first message anchor + compact older-turn bullets."""
    lines: list[str] = []
    shown = 0
    for msg in dropped:
        if not isinstance(msg, dict):
            continue
        role = "用户" if msg.get("role") == "user" else "助手"
        content = str(msg.get("content", "")).strip().replace("\n", " ")
        if not content:
            continue
        if len(content) > 60:
            content = content[:60] + "…"
        lines.append(f"{role}: {content}")
        shown += 1
        if shown >= 12:
            break
    if not lines:
        return ""
    remainder = len(dropped) - shown
    suffix = f"（另有 {remainder} 条更早消息已省略）" if remainder > 0 else ""
    return "早期对话摘要：" + "；".join(lines) + suffix


def _get_llm_for_compression() -> Any | None:
    """Return a usable LLM service or None (never raises)."""
    try:
        llm = get_llm_service()
        if getattr(llm, "is_mock", False):
            return None
        return llm
    except Exception:
        return None


def compress_history(
    history: list[dict[str, Any]],
    budget_tokens: int | None = None,
    min_keep_messages: int | None = None,
    llm: Any | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Split *history* into (recent_window, rolling_summary).

    The most recent ``min_keep_messages`` messages are always kept verbatim.
    Older messages are compressed into a summary string.  When the history is
    already within budget, it is returned unchanged with an empty summary.
    """
    budget = budget_tokens if budget_tokens is not None else settings.history_token_budget
    min_keep = (
        min_keep_messages
        if min_keep_messages is not None
        else settings.history_min_keep_messages
    )
    history = [m for m in history if isinstance(m, dict)]
    if not history:
        return [], ""

    if count_tokens(history) <= budget:
        return history, ""

    # Always keep the most recent turns for continuity.
    recent = list(history[-min_keep:])
    older = list(history[:-min_keep])
    if not older:
        return history, ""

    # If the recent window alone exceeds the budget, trim its oldest messages
    # (but never below 2 turns so immediate continuity is preserved).
    while len(recent) > 2 and count_tokens(recent) > budget:
        older.append(recent.pop(0))

    if llm is None:
        llm = _get_llm_for_compression()
    summary = _llm_summarize(older, llm)
    if summary is None:
        summary = _deterministic_summary(older)

    logger.info(
        "History compressed: %d -> %d messages kept, summary %d chars",
        len(history), len(recent), len(summary),
    )
    return recent, summary
