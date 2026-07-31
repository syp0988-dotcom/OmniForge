"""Tests for SQLite store lifecycle (WAL, migrations, schema version)."""

from __future__ import annotations

import sqlite3

from agentflow.database.sqlite import SQLiteStore


def test_wal_mode_and_schema_version(tmp_path):
    db_path = tmp_path / "lifecycle.db"
    store = SQLiteStore(db_path)
    try:
        with sqlite3.connect(str(db_path)) as conn:
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert journal_mode.lower() == "wal"
        assert version >= 1
        # content_hash migration must be applied on fresh databases.
        with sqlite3.connect(str(db_path)) as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(documents)")]
        assert "content_hash" in cols
    finally:
        store.close()


def test_reopen_is_idempotent(tmp_path):
    db_path = tmp_path / "reopen.db"
    SQLiteStore(db_path).close()
    # Reopening an existing database must not fail or duplicate migrations.
    store = SQLiteStore(db_path)
    try:
        assert store.get_all_documents() == []
    finally:
        store.close()
