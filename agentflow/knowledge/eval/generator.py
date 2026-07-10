"""LLM-driven question generation from knowledge-base chunks.

Produces evaluation datasets by prompting an LLM to generate natural-language
questions answerable from each chunk.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path

from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.eval.dataset import EvalDataset
from agentflow.services.llm_service import LLMService

logger = logging.getLogger("knowledge.eval.generator")

_GENERATE_PROMPT = """\
根据以下文档片段，生成 {questions_per_chunk} 个用户可能会问的问题。

要求：
1. 问题的答案必须能从此片段中直接得出
2. 问题应该自然、口语化，像真实用户会问的样子
3. 问题之间尽量多样化，避免相似表述
4. 返回纯 JSON 数组，格式：["问题1", "问题2", ...]

文档片段：
---
{chunk_text}
---

JSON:"""

_GENERATE_PROMPT_EN = """\
Based on the following document snippet, generate {questions_per_chunk} questions
that a user might ask.

Requirements:
1. The answer to each question MUST be derivable from this snippet
2. Questions should be natural, conversational, like real users would ask
3. Questions should be diverse, avoid similar phrasing
4. Return a pure JSON array: ["question 1", "question 2", ...]

Document snippet:
---
{chunk_text}
---

JSON:"""


def _extract_json_array(text: str) -> list[str]:
    """Robustly extract a JSON string array from LLM output."""
    # Try direct parse first
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(item) for item in result]
    except json.JSONDecodeError:
        pass

    # Try to find [...] in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(item) for item in result]
        except json.JSONDecodeError:
            pass

    # Last resort: extract quoted strings
    matches = re.findall(r'"([^"]+)"', text)
    if matches:
        return matches

    logger.warning("Could not parse questions from LLM output: %s", text[:200])
    return []


def generate_from_chunks(
    chunk_texts: list[tuple[int, str]],
    questions_per_chunk: int = 3,
    llm: LLMService | None = None,
    language: str = "zh",
    batch_size: int = 10,
    sleep_interval: float = 0.5,
) -> EvalDataset:
    """Generate evaluation questions from a list of (chunk_id, chunk_text) pairs.

    Args:
        chunk_texts: List of (chunk_id, text) tuples from the knowledge base.
        questions_per_chunk: How many questions to generate per chunk.
        llm: LLM service instance (creates a default if None).
        language: Prompt language, ``"zh"`` or ``"en"``.
        batch_size: Not used for batching, but for logging progress every N chunks.
        sleep_interval: Seconds to wait between LLM calls to avoid rate limits.

    Returns:
        ``EvalDataset`` with generated samples.
    """
    if llm is None:
        llm = LLMService()

    if not llm.client:
        raise RuntimeError(
            "LLM client is not configured. Set DEEPSEEK_API_KEY or "
            "configure an active model in the database."
        )

    prompt_template = _GENERATE_PROMPT if language == "zh" else _GENERATE_PROMPT_EN
    dataset = EvalDataset()
    total = len(chunk_texts)

    for i, (chunk_id, text) in enumerate(chunk_texts):
        if not text.strip():
            continue

        # Truncate overly long chunks to avoid blowing context
        snippet = text[:2000]

        prompt = prompt_template.format(
            questions_per_chunk=questions_per_chunk,
            chunk_text=snippet,
        )

        try:
            response = llm.complete(prompt)
            questions = _extract_json_array(response)
        except Exception as exc:
            logger.warning(
                "LLM call failed for chunk %d: %s. Skipping.", chunk_id, exc
            )
            continue

        for question in questions:
            if question.strip():
                dataset.add(
                    question=question.strip(),
                    relevant_chunk_ids=[chunk_id],
                )

        if (i + 1) % batch_size == 0:
            logger.info(
                "Progress: %d/%d chunks processed (%d questions generated)",
                i + 1, total, len(dataset),
            )

        if i < total - 1:
            time.sleep(sleep_interval)

    logger.info(
        "Generation complete: %d questions from %d chunks",
        len(dataset), total,
    )
    return dataset


def generate_from_knowledge_store(
    store,
    questions_per_chunk: int = 3,
    llm: LLMService | None = None,
    language: str = "zh",
    sample_chunks: int | None = None,
    seed: int | None = 42,
    min_chunk_length: int = 50,
    skip_file_types: tuple[str, ...] = ("xlsx", "csv"),
    max_pipe_ratio: float = 0.15,
) -> EvalDataset:
    """Convenience: generate from all (or a sample of) chunks in a KnowledgeStore.

    Args:
        store: A ``KnowledgeStore`` instance with indexed documents.
        questions_per_chunk: Questions to generate per chunk.
        llm: LLM service instance.
        language: Prompt language.
        sample_chunks: If set, randomly sample this many chunks instead of using all.
        seed: Random seed for sampling.
        min_chunk_length: Skip chunks shorter than this (not informative enough).
        skip_file_types: Document file types to skip (default: xlsx, csv).
        max_pipe_ratio: Skip chunks where ``|`` characters exceed this fraction
            of total characters (filters out table data).

    Returns:
        ``EvalDataset``.
    """
    chunk_texts: list[tuple[int, str]] = []
    for doc in store.db.get_all_documents():
        if doc["file_type"] in skip_file_types:
            continue
        for chunk in store.db.get_chunks_by_document(doc["id"]):
            content = chunk["content"].strip()
            if len(content) < min_chunk_length:
                continue
            pipe_ratio = content.count("|") / len(content)
            if pipe_ratio > max_pipe_ratio:
                continue
            chunk_texts.append((chunk["id"], content))

    if not chunk_texts:
        raise ValueError("No chunks found in the knowledge store.")

    if sample_chunks is not None and sample_chunks < len(chunk_texts):
        import random
        rng = random.Random(seed)
        chunk_texts = rng.sample(chunk_texts, sample_chunks)
        logger.info(
            "Sampled %d chunks from %d total", sample_chunks, len(chunk_texts)
        )

    return generate_from_chunks(
        chunk_texts=chunk_texts,
        questions_per_chunk=questions_per_chunk,
        llm=llm,
        language=language,
    )
