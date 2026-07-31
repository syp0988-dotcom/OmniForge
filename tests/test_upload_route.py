from __future__ import annotations

import os
import tempfile
import zipfile
import io
from pathlib import Path

from fastapi.testclient import TestClient

from agentflow.api import routes
from agentflow.app.main import app
from agentflow.config.settings import settings
from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.index import QdrantIndex
from agentflow.knowledge.store import KnowledgeStore


def test_upload_txt_indexes_document():
    old_store = routes._store
    old_knowledge_store = routes._knowledge_store
    tmp_db_path = tempfile.mktemp(suffix=".db")

    try:
        db = SQLiteStore(Path(tmp_db_path))
        knowledge_store = KnowledgeStore(
            db=db,
            qdrant_index=QdrantIndex.in_memory(collection_name="upload_route_test"),
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
