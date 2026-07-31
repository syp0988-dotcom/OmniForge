"""Tests for hybrid retrieval filtering and scoring behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.embedder import BaseEmbedder
from agentflow.knowledge.index import QdrantIndex
from agentflow.knowledge.store import KnowledgeStore


class FakeEmbedder(BaseEmbedder):
    """Deterministic random embedder — never calls an external API."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(7)

    def embed(self, texts: list[str], batch_size: int = 20) -> list[np.ndarray]:
        return [
            self._rng.standard_normal(8).astype(np.float32)
            for _ in texts
        ]

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return 8

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"


def _make_store(tmp_dir: Path, db_path: Path) -> KnowledgeStore:
    db = SQLiteStore(db_path)
    return KnowledgeStore(
        db=db,
        qdrant_index=QdrantIndex.in_memory(collection_name="retrieval_filter_test"),
        embedder=FakeEmbedder(),
    )


def test_search_filters_by_document(tmp_path):
    db_path = tmp_path / "k.db"
    store = _make_store(tmp_path, db_path)
    try:
        # Two documents with distinct contents.
        file_a = tmp_path / "alpha.txt"
        file_b = tmp_path / "beta.txt"
        file_a.write_text("Alpha document: first content block.", encoding="utf-8")
        file_b.write_text("Beta document: second content block.", encoding="utf-8")
        doc_a = store.add_document(file_a, "alpha.txt")
        doc_b = store.add_document(file_b, "beta.txt")

        # Unfiltered search sees chunks from both documents.
        all_results = store.search("content", top_k=10, min_score=0.0)
        assert {r["document_id"] for r in all_results} == {doc_a, doc_b}

        # Filtered search only returns chunks from the requested document.
        only_a = store.search(
            "content", top_k=10, min_score=0.0, document_ids=[doc_a],
        )
        assert only_a, "expected at least one result"
        assert {r["document_id"] for r in only_a} == {doc_a}

        only_b = store.search(
            "content", top_k=10, min_score=0.0, document_ids=[doc_b],
        )
        assert only_b
        assert {r["document_id"] for r in only_b} == {doc_b}
    finally:
        try:
            store.db.close()
        except Exception:
            pass
        try:
            db_path.unlink(missing_ok=True)
        except PermissionError:
            pass
