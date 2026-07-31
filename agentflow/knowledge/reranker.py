"""Pluggable reranking layer for retrieval results.

The hybrid retriever returns good *recall* candidates; a reranker improves
*precision* by re-ordering them before they reach the answer generator.

Modes (``KNOWLEDGE_RERANKER``):

- ``"none"`` (default): passthrough, zero extra latency/cost.
- ``"llm"``: the LLM picks the most relevant candidates from the top-N
  retrieved chunks (``KNOWLEDGE_RERANK_CANDIDATES``) and returns the top-K
  (``KNOWLEDGE_RERANK_TOP_K``).

Any reranker failure degrades gracefully to the original order.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from agentflow.config.settings import settings
from agentflow.utils.logging import build_logger

logger = build_logger("reranker")


class Reranker(Protocol):
    """Interface: reorder *results* for *query* and return the new order."""

    def rerank(
        self, query: str, results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ...


class NoopReranker:
    """Default: keep the retriever's order (no extra cost)."""

    def rerank(
        self, query: str, results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return results


class LLMReranker:
    """LLM-based reranker: ask the model to order the candidates by relevance."""

    def __init__(
        self,
        top_k: int | None = None,
        candidates: int | None = None,
        llm: object | None = None,
    ) -> None:
        self._top_k = top_k or settings.knowledge_rerank_top_k
        self._candidates = candidates or settings.knowledge_rerank_candidates
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from agentflow.services.llm_service import get_llm_service
            self._llm = get_llm_service()
        return self._llm

    def rerank(
        self, query: str, results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(results) <= 1:
            return results

        candidates = results[: self._candidates]
        lines = []
        for idx, item in enumerate(candidates):
            content = str(item.get("content", ""))[:400].replace("\n", " ")
            lines.append(f"[{idx}] {content}")

        prompt = (
            "以下是知识库检索到的候选片段。请根据与查询的相关程度，"
            "从高到低重新排序，只保留最相关的几条。\n\n"
            f"查询：{query}\n\n候选片段：\n"
            + "\n".join(lines)
            + "\n\n只输出 JSON：{\"order\": [片段编号, ...]}，编号必须在上面给出的范围内，"
              "按相关度从高到低排列。"
        )

        try:
            raw = self._get_llm().complete(
                messages=[
                    {
                        "role": "system",
                        "content": "你是检索结果重排器，输出 JSON，不输出其他内容。",
                    },
                    {"role": "user", "content": prompt},
                ],
                node_name="reranker",
            )
            order = _parse_order(raw)
            if not order:
                logger.info("Reranker: no valid order in LLM output, keeping original")
                return results
        except Exception as exc:
            logger.warning("Reranker failed (%s); keeping original order", exc)
            return results

        ranked: list[dict[str, Any]] = []
        seen: set[int] = set()
        for idx in order:
            if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
                continue
            if idx in seen:
                continue
            seen.add(idx)
            ranked.append(candidates[idx])
        for idx, item in enumerate(candidates):
            if idx not in seen:
                ranked.append(item)

        trimmed = ranked[: self._top_k]
        if trimmed != results[: self._top_k]:
            logger.info("Reranker: reordered %d candidate(s) → top-%d", len(candidates), len(trimmed))
        return trimmed


def _parse_order(raw: str) -> list[int] | None:
    """Extract ``{"order": [...]}`` (or a bare list) from LLM output."""
    if not raw:
        return None
    text = raw.strip()
    # Bare JSON list, e.g. [2, 0, 1]
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
        if isinstance(parsed, dict):
            for key in ("order", "ranking", "indices"):
                if key in parsed and isinstance(parsed[key], list):
                    return [int(x) for x in parsed[key]]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fenced JSON block
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return _parse_order(m.group(1))

    # Bare list inside braces, e.g. {"order": [2, 0, 1]}
    m = re.search(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", text)
    if m:
        try:
            return [int(x) for x in json.loads(m.group(0))]
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def get_reranker() -> Reranker:
    """Return the configured reranker (never raises)."""
    mode = settings.knowledge_reranker.strip().lower()
    if mode == "llm":
        return LLMReranker()
    if mode not in ("none", ""):
        logger.warning("Unknown reranker mode '%s'; using noop", mode)
    return NoopReranker()
