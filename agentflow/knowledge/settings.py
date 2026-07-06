"""Knowledge-base configuration — typed accessors for settings.knowledge_*."""

from __future__ import annotations

from pathlib import Path

from agentflow.config.settings import settings


def embedder_type() -> str:
    """Return the active embedder type: ``"semantic"`` or ``"tfidf"``."""
    return settings.knowledge_embedder


def embedding_model() -> str:
    """Return the sentence-transformers model name."""
    return settings.knowledge_embedding_model


def hybrid_weights() -> tuple[float, float]:
    """Return (alpha, beta) for hybrid search fusion."""
    return settings.knowledge_alpha, settings.knowledge_beta


def chunk_params() -> tuple[int, int]:
    """Return (chunk_size, chunk_overlap)."""
    return settings.knowledge_chunk_size, settings.knowledge_chunk_overlap


def faiss_index_path() -> Path:
    """Return the FAISS index file path.

    If ``knowledge_faiss_index_path`` is set, use it directly;
    otherwise default to ``<project_root>/data/faiss.index``.
    """
    custom = settings.knowledge_faiss_index_path
    if custom:
        return Path(custom)
    return settings.project_root / "data" / "faiss.index"


def faiss_index_type() -> str:
    """Return the FAISS index type (``"HNSW"`` | ``"Flat"``)."""
    return settings.knowledge_faiss_index_type


def search_defaults() -> tuple[int, float]:
    """Return (top_k, min_score) defaults for search."""
    return settings.knowledge_top_k, settings.knowledge_min_score
