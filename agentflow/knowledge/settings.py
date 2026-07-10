"""Knowledge-base configuration — typed accessors for settings.knowledge_*."""

from __future__ import annotations

from pathlib import Path

from agentflow.config.settings import settings


def hybrid_weights() -> tuple[float, float]:
    """Return (alpha, beta) for hybrid search fusion."""
    return settings.knowledge_alpha, settings.knowledge_beta


def chunk_params() -> tuple[int, int]:
    """Return (chunk_size, chunk_overlap)."""
    return settings.knowledge_chunk_size, settings.knowledge_chunk_overlap


def search_defaults() -> tuple[int, float]:
    """Return (top_k, min_score) defaults for search."""
    return settings.knowledge_top_k, settings.knowledge_min_score


def embedding_api_key() -> str:
    """Return the embedding API key (DashScope or compatible)."""
    return settings.embedding_api_key


def embedding_base_url() -> str:
    """Return the embedding API base URL."""
    return settings.embedding_base_url


def embedding_model_name() -> str:
    """Return the embedding model name."""
    return settings.embedding_model_name


def qdrant_url() -> str:
    """Return the Qdrant server URL."""
    return settings.qdrant_url


def qdrant_api_key() -> str:
    """Return the Qdrant API key (empty for local mode)."""
    return settings.qdrant_api_key


def qdrant_collection() -> str:
    """Return the Qdrant collection name."""
    return settings.qdrant_collection


def qdrant_storage_path() -> Path:
    """Return the Qdrant local storage path (used when no server URL)."""
    return settings.project_root / "data" / "qdrant"
