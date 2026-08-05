"""Tests for production-readiness hardening:

- required-env fast-fail validation
- log rotation
- RAG retrieval degradation (vector -> lexical when embedder/Qdrant is down)
"""

from logging.handlers import TimedRotatingFileHandler

import pytest

from agentflow.knowledge.retrieval import HybridRetriever
from agentflow.utils.logging import build_logger


class _RaisingEmbedder:
    def embed_query(self, text):
        raise ConnectionError("embedding API down")


class _FakeDb:
    def search_chunks_fts(self, query, limit):
        return [{"chunk_id": 1, "rank": -5.0}]

    def get_chunks_with_documents_batch(self, chunk_ids):
        return {
            1: {"document_id": 10, "filename": "guide.md", "content": "hello world"},
        }

    def get_chunk_with_document(self, chunk_id):
        return self.get_chunks_with_documents_batch([chunk_id]).get(chunk_id)


def test_retrieval_degrades_to_lexical_when_embedding_down():
    retriever = HybridRetriever(
        embedder=_RaisingEmbedder(), db=_FakeDb(), qdrant_index=None,
    )
    results = retriever.search("hello", top_k=5, min_score=0.0)
    assert len(results) == 1
    assert results[0]["method"] == "lexical"
    assert results[0]["filename"] == "guide.md"


def test_retrieval_returns_empty_when_all_sources_down():
    class _EmptyDb(_FakeDb):
        def search_chunks_fts(self, query, limit):
            return []

    retriever = HybridRetriever(
        embedder=_RaisingEmbedder(), db=_EmptyDb(), qdrant_index=None,
    )
    assert retriever.search("hello", top_k=5, min_score=0.0) == []


def test_required_env_validation_raises_in_production(monkeypatch):
    from agentflow.app.main import _validate_required_env
    from agentflow.config.settings import settings

    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "enforce_required_env", False)
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    monkeypatch.setattr(settings, "embedding_api_key", "")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        _validate_required_env()


def test_required_env_validation_passes_when_keys_present(monkeypatch):
    from agentflow.app.main import _validate_required_env
    from agentflow.config.settings import settings

    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "enforce_required_env", False)
    monkeypatch.setattr(settings, "deepseek_api_key", "x")
    monkeypatch.setattr(settings, "embedding_api_key", "y")
    _validate_required_env()  # must not raise


def test_required_env_validation_skipped_in_development(monkeypatch):
    from agentflow.app.main import _validate_required_env
    from agentflow.config.settings import settings

    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "enforce_required_env", False)
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    _validate_required_env()  # must not raise


def test_logger_uses_rotating_file_handler():
    logger = build_logger("test_rotation_check")
    assert any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers)
