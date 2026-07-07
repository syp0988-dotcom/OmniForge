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


def chroma_path() -> Path:
    """Return the ChromaDB persistent storage path.

    If ``knowledge_chroma_path`` is set, use it directly;
    otherwise default to ``<project_root>/data/chromadb``.
    """
    custom = settings.knowledge_chroma_path
    if custom:
        return Path(custom)
    return settings.project_root / "data" / "chromadb"


def chroma_collection() -> str:
    """Return the ChromaDB collection name."""
    return settings.knowledge_chroma_collection


def search_defaults() -> tuple[int, float]:
    """Return (top_k, min_score) defaults for search."""
    return settings.knowledge_top_k, settings.knowledge_min_score
