from __future__ import annotations

import os
import tempfile
import zipfile
import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from agentflow.api import routes
from agentflow.app.main import app
from agentflow.config.settings import settings
from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.embedder import BaseEmbedder
from agentflow.knowledge.index import QdrantIndex
from agentflow.knowledge.store import KnowledgeStore


class _DeterministicEmbedder(BaseEmbedder):
    """Local stub so upload tests never depend on EMBEDDING_API_KEY or the network."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(11)

    def embed(self, texts: list[str], batch_size: int = 20) -> list[np.ndarray]:
        return [self._rng.standard_normal(8).astype(np.float32) for _ in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return 8

    @property
    def name(self) -> str:
        return "deterministic-test-embedder"

    @property
    def model_name(self) -> str:
        return "deterministic-test-model"


def test_upload_txt_indexes_document():
    old_store = routes._store
    old_knowledge_store = routes._knowledge_store
    tmp_db_path = tempfile.mktemp(suffix=".db")

    try:
        db = SQLiteStore(Path(tmp_db_path))
        knowledge_store = KnowledgeStore(
            db=db,
            qdrant_index=QdrantIndex.in_memory(collection_name="upload_route_test"),
            embedder=_DeterministicEmbedder(),
        )
        routes.set_store(db)
        routes.set_knowledge_store(knowledge_store)

        client = TestClient(app)
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", b"Knowledge upload smoke test content.", "text/plain")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["filename"] == "notes.txt"
        assert body["document_id"] > 0
        assert any(doc["filename"] == "notes.txt" for doc in knowledge_store.list_documents())
    finally:
        routes.set_store(old_store)
        routes.set_knowledge_store(old_knowledge_store)
        try:
            os.unlink(tmp_db_path)
        except OSError:
            pass


def test_upload_duplicate_content_is_deduped():
    """Uploading the same bytes twice must not create a second document."""
    old_store = routes._store
    old_knowledge_store = routes._knowledge_store
    tmp_db_path = tempfile.mktemp(suffix=".db")

    try:
        db = SQLiteStore(Path(tmp_db_path))
        knowledge_store = KnowledgeStore(
            db=db,
            qdrant_index=QdrantIndex.in_memory(collection_name="upload_dedup_test"),
            embedder=_DeterministicEmbedder(),
        )
        routes.set_store(db)
        routes.set_knowledge_store(knowledge_store)

        client = TestClient(app)
        payload = b"Dedup me please: the exact same bytes."

        first = client.post(
            "/upload",
            files={"file": ("a.txt", payload, "text/plain")},
        )
        second = client.post(
            "/upload",
            files={"file": ("copy-of-a.txt", payload, "text/plain")},
        )

        assert first.status_code == 200
        first_body = first.json()
        assert first_body["status"] == "ok"

        assert second.status_code == 200
        second_body = second.json()
        assert second_body["status"] == "duplicate"
        assert second_body["document_id"] == first_body["document_id"]

        docs = knowledge_store.list_documents()
        assert len(docs) == 1, f"Expected exactly one document, got {len(docs)}"
    finally:
        routes.set_store(old_store)
        routes.set_knowledge_store(old_knowledge_store)
        try:
            os.unlink(tmp_db_path)
        except OSError:
            pass


def test_upload_exceeding_size_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 32)
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("big.txt", b"x" * 4096, "text/plain")},
    )
    assert response.status_code == 413


def test_zip_entry_count_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(settings, "max_zip_entries", 1)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("a.txt", "aaaa")
        zf.writestr("b.txt", "bbbb")
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("archive.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 413


def test_zip_uncompressed_size_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(settings, "max_zip_uncompressed_bytes", 64)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("a.txt", "x" * 4096)
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("archive.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 413
