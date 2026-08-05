"""Comprehensive tests for the SQLiteStore (sessions, docs, models, memory,
executions, cleanup)."""


import pytest

from agentflow.database.sqlite import SQLiteStore


_OPEN_STORES: list[SQLiteStore] = []


@pytest.fixture(autouse=True)
def _close_stores():
    yield
    for store in _OPEN_STORES:
        store.close()
    _OPEN_STORES.clear()


def _store(tmp_path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "test.db")
    _OPEN_STORES.append(store)
    return store


# -- sessions & messages ------------------------------------------------------


def test_session_lifecycle(tmp_path):
    store = _store(tmp_path)
    sess = store.create_session("测试会话")
    assert sess["title"] == "测试会话"
    assert store.get_session(sess["id"]) is not None
    assert any(s["id"] == sess["id"] for s in store.list_sessions())

    assert store.update_session_title(sess["id"], "改名")
    assert store.get_session(sess["id"])["title"] == "改名"

    assert store.update_session_state(sess["id"], '{"k": 1}')
    assert "k" in store.get_session_state(sess["id"])

    assert store.delete_session(sess["id"]) is True
    assert store.get_session(sess["id"]) is None


def test_messages(tmp_path):
    store = _store(tmp_path)
    sess = store.create_session()
    msg_id = store.add_message("user", "你好", session_id=sess["id"])
    store.add_message("assistant", "你好！", session_id=sess["id"])
    assert msg_id > 0
    msgs = store.get_session_messages(sess["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    recent = store.list_messages(limit=10)
    assert recent[0]["role"] == "assistant"  # newest first
    assert any(m["content"] == "你好" for m in recent)


# -- documents & chunks -------------------------------------------------------


def test_documents_and_chunks(tmp_path):
    store = _store(tmp_path)
    doc_id = store.add_document("a.md", "md", 100, content_hash="abc123")
    assert doc_id > 0
    assert store.get_document_by_hash("abc123")["id"] == doc_id
    assert store.get_document_by_hash("") is None

    store.add_chunk(doc_id, "hello world", 0)
    store.add_chunk(doc_id, "second chunk", 1)
    chunks = store.get_chunks_by_document(doc_id)
    assert len(chunks) == 2
    assert chunks[0]["content"] == "hello world"

    first_chunk_id = chunks[0]["id"]
    info = store.get_chunk_with_document(first_chunk_id)
    assert info["filename"] == "a.md"
    batch = store.get_chunks_with_documents_batch([first_chunk_id])
    assert batch[first_chunk_id]["content"] == "hello world"

    fts = store.search_chunks_fts("hello", limit=5)
    assert any(c["chunk_id"] == first_chunk_id for c in fts)

    store.update_document_metadata(doc_id, {"permanent_path": "/x/a.md"})
    assert store.get_all_documents()[0]["doc_metadata"] == '{"permanent_path": "/x/a.md"}'

    store.delete_document_cascade(doc_id)
    assert store.get_chunks_by_document(doc_id) == []


def test_knowledge_meta(tmp_path):
    store = _store(tmp_path)
    store.set_knowledge_meta("version", "2")
    assert store.get_knowledge_meta("version") == "2"
    assert store.get_knowledge_meta("missing") is None


# -- models -------------------------------------------------------------------


def test_models_lifecycle(tmp_path):
    store = _store(tmp_path)
    m1 = store.add_model("DeepSeek", "deepseek", "https://api.deepseek.com", "k", "deepseek-v4", is_active=True)
    m2 = store.add_model("OpenAI", "openai", "https://api.openai.com", "k2", "gpt-4o")
    assert store.get_model(m1)["model_name"] == "deepseek-v4"
    assert store.get_active_model()["id"] == m1

    store.set_active_model(m2)
    assert store.get_active_model()["id"] == m2

    assert store.update_model(m1, temperature=0.1)
    assert store.get_model(m1)["temperature"] == 0.1

    assert store.delete_model(m1) is True
    assert store.get_model(m1) is None
    assert len(store.get_all_models()) == 1


# -- long-term memory ---------------------------------------------------------


def test_long_term_memory(tmp_path):
    store = _store(tmp_path)
    store.set_long_term_memory("user_interest_python", "用户喜欢 Python", "topic")
    assert store.get_long_term_memory("user_interest_python") == "用户喜欢 Python"
    results = store.search_long_term_memory("Python", limit=5)
    assert any(r["key"] == "user_interest_python" for r in results)
    batch = store.search_long_term_memory_batch(["Python"], limit_per_term=3)
    assert any(r["key"] == "user_interest_python" for r in batch)
    assert len(store.list_long_term_memories(category="topic")) == 1
    assert store.delete_long_term_memory("user_interest_python") is True
    store.set_long_term_memory("a", "1", "topic")
    store.set_long_term_memory("b", "2", "pref")
    store.clear_long_term_memories(category="topic")
    remaining = store.list_long_term_memories()
    assert [r["key"] for r in remaining] == ["b"]
    store.clear_long_term_memories()
    assert store.list_long_term_memories() == []


# -- executions & checkpoints ------------------------------------------------


def test_executions_and_checkpoints(tmp_path):
    store = _store(tmp_path)
    exec_id = store.save_execution(
        session_id=1, trace_id="trace-1", question="q", answer="a",
        goal_type="project", trace_json="[]", errors_json="[]",
        degraded=False, duration_ms=12.5,
    )
    assert store.get_execution(exec_id)["trace_id"] == "trace-1"
    assert store.list_executions(session_id=1)[0]["id"] == exec_id

    store.save_checkpoint(exec_id, "planner", '[{"task_id": "a"}]')
    cps = store.list_checkpoints(exec_id)
    assert len(cps) == 1
    assert cps[0]["node_name"] == "planner"
    assert "task_id" in cps[0]["task_queue_json"]


# -- cleanup ------------------------------------------------------------------


def test_cleanup_expired_sessions_and_memories(tmp_path):
    store = _store(tmp_path)
    sess = store.create_session()
    store.add_message("user", "old", session_id=sess["id"])
    store.set_long_term_memory("old_memory", "x", "topic")

    # Force expiry by updating timestamps backwards.
    import sqlite3
    with sqlite3.connect(str(tmp_path / "test.db")) as conn:
        conn.execute("UPDATE sessions SET created_at = datetime('now', '-10 days')")
        conn.execute("UPDATE long_term_memory SET updated_at = datetime('now', '-60 days')")
        conn.commit()

    deleted_sessions = store.delete_sessions_older_than(hours=72)
    deleted_memories = store.delete_old_memories(days=30)
    assert deleted_sessions >= 1
    assert deleted_memories >= 1
    assert store.list_sessions() == []
    assert store.list_long_term_memories() == []
