"""Tests for embedding-model detection and knowledge index rebuild."""

from __future__ import annotations

import pytest

from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.index import QdrantIndex
from agentflow.knowledge.store import KnowledgeStore
from tests.test_knowledge_retrieval import FakeEmbedder


def _store(tmp_path, name="model_migration_test") -> tuple[KnowledgeStore, SQLiteStore]:
    db_path = tmp_path / "migration.db"
    db = SQLiteStore(db_path)
    store = KnowledgeStore(
        db=db,
        qdrant_index=QdrantIndex.in_memory(collection_name=name),
        embedder=FakeEmbedder(),
    )
    return store, db


def test_model_mismatch_raises_clear_error(tmp_path):
    store, db = _store(tmp_path)
    try:
        db.set_knowledge_meta("embedding_model", "text-embedding-v2")
        source = tmp_path / "doc.txt"
        source.write_text("some content", encoding="utf-8")
        with pytest.raises(ValueError, match="rebuild"):
            store.add_document(source, "doc.txt")
    finally:
        db.close()


def test_rebuild_reingests_documents(tmp_path):
    store, db = _store(tmp_path)
    try:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("alpha doc", encoding="utf-8")
        b.write_text("beta doc", encoding="utf-8")
        store.add_document(a, "a.txt")
        store.add_document(b, "b.txt")
        assert db.get_knowledge_meta("embedding_model") == "fake-model"

        # Simulate the upload route's bookkeeping: record permanent paths.
        for doc in db.get_all_documents():
            db.update_document_metadata(
                doc["id"], {"permanent_path": str(tmp_path / doc["filename"])},
            )

        # A rebuild clears everything and re-ingests from the saved paths.
        result = store.rebuild()
        assert result["reingested"] == 2
        assert result["failed"] == []
        assert len(store.list_documents()) == 2
        assert db.get_knowledge_meta("embedding_model") == "fake-model"
    finally:
        db.close()
