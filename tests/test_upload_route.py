from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from agentflow.api import routes
from agentflow.app.main import app
from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.index import ChromaIndex
from agentflow.knowledge.store import KnowledgeStore


def test_upload_txt_indexes_document():
    old_store = routes._store
    old_knowledge_store = routes._knowledge_store
    tmp_db_path = tempfile.mktemp(suffix=".db")

    try:
        db = SQLiteStore(Path(tmp_db_path))
        knowledge_store = KnowledgeStore(
            db=db,
            chroma_index=ChromaIndex.in_memory(collection_name="upload_route_test"),
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
