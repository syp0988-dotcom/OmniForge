"""Tests for the pluggable reranking layer."""

from __future__ import annotations

from pathlib import Path

from agentflow.config.settings import settings
from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.index import QdrantIndex
from agentflow.knowledge.reranker import LLMReranker, NoopReranker, _parse_order
from agentflow.knowledge.store import KnowledgeStore
from tests.mock_llm import MockLLMService
from tests.test_knowledge_retrieval import FakeEmbedder


def _candidate(score: float) -> dict:
    return {"chunk_id": int(score * 1000), "content": f"chunk {score}", "score": score}


def test_noop_reranker_keeps_order():
    results = [_candidate(0.9), _candidate(0.5)]
    assert NoopReranker().rerank("q", results) == results


def test_llm_reranker_reorders():
    llm = MockLLMService(responses={"default": '{"order": [1, 0, 2]}'})
    reranker = LLMReranker(top_k=2, candidates=10, llm=llm)
    results = [_candidate(0.9), _candidate(0.8), _candidate(0.7)]
    ordered = reranker.rerank("q", results)
    assert [r["chunk_id"] for r in ordered] == [800, 900]


def test_llm_reranker_falls_back_on_garbage():
    llm = MockLLMService(responses={"default": "我不明白你在说什么"})
    reranker = LLMReranker(top_k=2, candidates=10, llm=llm)
    results = [_candidate(0.9), _candidate(0.8)]
    assert reranker.rerank("q", results) == results


def test_parse_order_variants():
    assert _parse_order('{"order": [2, 0, 1]}') == [2, 0, 1]
    assert _parse_order("[1, 2, 0]") == [1, 2, 0]
    assert _parse_order('```json\n{"order": [0, 1]}\n```') == [0, 1]
    assert _parse_order("") is None


class ReverseReranker:
    """Deterministic fake used to verify store integration."""

    def rerank(self, query: str, results: list[dict]):
        return list(reversed(results))


def test_store_reranks_and_trims(tmp_path, monkeypatch):
    db_path: Path = tmp_path / "rerank.db"
    db = SQLiteStore(db_path)
    embedder = FakeEmbedder()
    store = KnowledgeStore(
        db=db,
        qdrant_index=QdrantIndex.in_memory(collection_name="rerank_store_test"),
        embedder=embedder,
    )
    monkeypatch.setattr(settings, "knowledge_rerank_candidates", 4)
    store.rerank_candidates = max(settings.knowledge_rerank_candidates, settings.knowledge_top_k)

    try:
        file_a = tmp_path / "doc_a.txt"
        file_b = tmp_path / "doc_b.txt"
        file_a.write_text("alpha content here.", encoding="utf-8")
        file_b.write_text("beta content there.", encoding="utf-8")
        store.add_document(file_a, "doc_a.txt")
        store.add_document(file_b, "doc_b.txt")

        plain = store.search("content", top_k=2, min_score=0.0)
        store.reranker = ReverseReranker()
        reranked = store.search("content", top_k=2, min_score=0.0)
        assert len(plain) == 2 and len(reranked) == 2
        # A reversing reranker must change the top-1 result.
        assert reranked[0]["chunk_id"] != plain[0]["chunk_id"]
    finally:
        try:
            db.close()
        except Exception:
            pass
