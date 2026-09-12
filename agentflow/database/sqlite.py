"""SQLite-backed persistence for chat history and knowledge base."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from agentflow.config.settings import settings
from agentflow.utils.logging import build_logger as _build_logger

_log = _build_logger("sqlite")

# Schema version.  Bump this when adding a migration in ``_initialize``;
# the migration chain runs in order from ``PRAGMA user_version``.
_SCHEMA_VERSION = 1

# Hard ceiling for caller-supplied row limits.  SQLite treats ``LIMIT -1`` as
# "no limit", so a negative value used to dump an entire table.
_MAX_ROWS = 500


def clamp_limit(limit: int, default: int = 50, maximum: int = _MAX_ROWS) -> int:
    """Return a safe row limit: 1..maximum, falling back to *default*."""
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(value, maximum))


class SQLiteStore:
    """Simple SQLite-backed persistence for chat/history and knowledge base data.

    Uses a thread-local connection cache to avoid per-query connection overhead.
    WAL mode is enabled for concurrent read/write safety.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialize()

    @contextmanager
    def _connect(self):
        """Get a cached thread-local connection (context manager).

        Reuses the same connection across all queries in a single request,
        avoiding per-query ``sqlite3.connect()`` overhead.  A connection is
        only created on first use in a thread; :meth:`close` releases it.
        """
        conn = getattr(self._local, "connection", None)
        new = conn is None
        if new:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.connection = conn
        try:
            yield conn
        except Exception:
            if new:
                conn.close()
                self._local.connection = None
            raise

    def close(self) -> None:
        """Close the cached connection (if any). Call at shutdown."""
        conn = getattr(self._local, "connection", None)
        if conn:
            conn.close()
            self._local.connection = None

    def _initialize(self) -> None:
        # ``with sqlite3.connect(...)`` only commits/rolls back — it does NOT
        # close the connection, so every SQLiteStore() leaked a file handle
        # (ResourceWarning, and on Windows a locked database file).
        with closing(sqlite3.connect(str(self.db_path))) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")

            connection.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL DEFAULT 0,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            # Performance indexes (after their respective tables exist)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
                ON sessions(updated_at)
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chats_session_id
                ON chats(session_id)
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    doc_metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # --- FTS5 full-text search index for chunks (optional, try) ---
            try:
                connection.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        content, content=chunks, content_rowid=id
                    )
                """)
            except sqlite3.OperationalError:
                pass  # FTS5 may not be available in all SQLite builds

            # FTS5 sync triggers (same try/except)
            try:
                connection.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks
                    BEGIN
                        INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
                    END
                """)
            except sqlite3.OperationalError:
                pass
            try:
                connection.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks
                    BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, content)
                        VALUES('delete', old.id, old.content);
                    END
                """)
            except sqlite3.OperationalError:
                pass
            try:
                connection.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks
                    BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, content)
                        VALUES('delete', old.id, old.content);
                        INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
                    END
                """)
            except sqlite3.OperationalError:
                pass

            # -- Lightweight column migration: content_hash for upload dedup --
            columns = [
                row[1]
                for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            ]
            if "content_hash" not in columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_hash "
                "ON documents(content_hash)"
            )

            connection.execute("""
                CREATE TABLE IF NOT EXISTS llm_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'custom',
                    base_url TEXT NOT NULL,
                    api_key TEXT NOT NULL DEFAULT '',
                    model_name TEXT NOT NULL,
                    temperature REAL NOT NULL DEFAULT 0.7,
                    max_tokens INTEGER NOT NULL DEFAULT 4096,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_long_term_memory_category
                ON long_term_memory(category)
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    trace_id TEXT NOT NULL DEFAULT '',
                    question TEXT NOT NULL DEFAULT '',
                    answer TEXT NOT NULL DEFAULT '',
                    goal_type TEXT NOT NULL DEFAULT '',
                    trace_json TEXT NOT NULL DEFAULT '[]',
                    errors_json TEXT NOT NULL DEFAULT '[]',
                    degraded INTEGER NOT NULL DEFAULT 0,
                    duration_ms REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_session_id
                ON executions(session_id)
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_created_at
                ON executions(created_at)
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS execution_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id INTEGER NOT NULL,
                    node_name TEXT NOT NULL DEFAULT '',
                    task_queue_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (execution_id) REFERENCES executions(id) ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_execution_id
                ON execution_checkpoints(execution_id)
            """)

            # --- Migration: add session_id column to existing chats table ---
            try:
                connection.execute("ALTER TABLE chats ADD COLUMN session_id INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # column already exists

            # --- Migration: add session_state column to sessions ---
            try:
                connection.execute("ALTER TABLE sessions ADD COLUMN session_state TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists

            # --- Migration: create a default session for orphaned messages ---
            cursor = connection.execute("SELECT COUNT(*) FROM chats")
            total_chats = cursor.fetchone()[0]
            if total_chats > 0:
                cursor = connection.execute("SELECT COUNT(*) FROM sessions")
                if cursor.fetchone()[0] == 0:
                    connection.execute(
                        "INSERT INTO sessions(id, title, created_at, updated_at) "
                        "VALUES (0, '历史记录', datetime('now'), datetime('now'))"
                    )
                    connection.execute(
                        "UPDATE chats SET session_id = 0 WHERE session_id = 0"
                    )

            # -- Versioned migration bookkeeping -------------------------------
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version < _SCHEMA_VERSION:
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                _log.info(
                    "Database schema migrated %d -> %d",
                    version, _SCHEMA_VERSION,
                )

            connection.commit()

    # -- Chat history / Sessions -----------------------------------------------

    def create_session(self, title: str = "新对话") -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO sessions(title, created_at, updated_at) "
                "VALUES (?, datetime('now'), datetime('now'))",
                (title,),
            )
            connection.commit()
            return self.get_session(cursor.lastrowid)  # type: ignore[arg-type]

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
        }

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = clamp_limit(limit, default=50)
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, title, created_at, updated_at FROM sessions "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [
            {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}
            for row in rows
        ]

    def update_session_title(self, session_id: int, title: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET title = ?, updated_at = datetime('now') WHERE id = ?",
                (title, session_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def update_session_state(self, session_id: int, state_json: str) -> bool:
        """Persist serialized session_state JSON for a session."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET session_state = ?, updated_at = datetime('now') WHERE id = ?",
                (state_json, session_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def get_session_state(self, session_id: int) -> str:
        """Load serialized session_state JSON for a session."""
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT session_state FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
        return row[0] if row and row[0] else ""

    def delete_session(self, session_id: int) -> bool:
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            cursor = connection.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            connection.execute(
                "DELETE FROM chats WHERE session_id = ?", (session_id,)
            )
            connection.commit()
            return cursor.rowcount > 0

    def add_message(self, role: str, content: str, session_id: int = 0) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chats(session_id, role, content, created_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (session_id, role, content),
            )
            connection.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def list_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = clamp_limit(limit, default=20)
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT role, content, created_at FROM chats ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [
            {"role": role, "content": content, "created_at": created_at}
            for role, content, created_at in rows
        ]

    def get_session_messages(self, session_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, role, content, created_at FROM chats "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            rows = cursor.fetchall()
        return [
            {"id": row[0], "role": row[1], "content": row[2], "created_at": row[3]}
            for row in rows
        ]

    # -- Documents -------------------------------------------------------------

    def add_document(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        doc_metadata: str = "{}",
        content_hash: str = "",
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO documents(filename, file_type, file_size, doc_metadata, content_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (filename, file_type, file_size, doc_metadata, content_hash),
            )
            connection.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        """Return the first document that matches a content hash ('' never matches)."""
        if not content_hash:
            return None
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, filename, file_type, file_size, doc_metadata, created_at "
                "FROM documents WHERE content_hash = ? ORDER BY id LIMIT 1",
                (content_hash,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "filename": row[1],
            "file_type": row[2],
            "file_size": row[3],
            "doc_metadata": row[4],
            "created_at": row[5],
        }

    def update_document_metadata(self, doc_id: int, updates: dict[str, object]) -> None:
        """Merge *updates* into the JSON metadata of an existing document."""
        import json
        with self._connect() as connection:
            row = connection.execute(
                "SELECT doc_metadata FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                return
            meta = json.loads(row[0]) if row[0] else {}
            meta.update(updates)
            connection.execute(
                "UPDATE documents SET doc_metadata = ? WHERE id = ?",
                (json.dumps(meta, ensure_ascii=False), doc_id),
            )
            connection.commit()

    def get_all_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, filename, file_type, file_size, doc_metadata, created_at "
                "FROM documents ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "filename": row[1],
                "file_type": row[2],
                "file_size": row[3],
                "doc_metadata": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def delete_document_cascade(self, doc_id: int) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            connection.commit()

    def clear_all_knowledge(self) -> None:
        """Delete every chunk and document, and reset embedding model meta."""
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            connection.execute(
                "DELETE FROM knowledge_meta "
                "WHERE key IN ('embedding_model', 'embedding_dimension')"
            )
            connection.commit()

    # -- Chunks ----------------------------------------------------------------

    def add_chunk(self, document_id: int, content: str, chunk_index: int) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chunks(document_id, content, chunk_index) VALUES (?, ?, ?)",
                (document_id, content, chunk_index),
            )
            connection.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_chunks_by_document(self, doc_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, content, chunk_index FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc_id,),
            )
            rows = cursor.fetchall()
        return [
            {"id": row[0], "content": row[1], "chunk_index": row[2]} for row in rows
        ]

    def get_chunk_with_document(self, chunk_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT c.id, c.document_id, c.content, d.filename "
                "FROM chunks c JOIN documents d ON c.document_id = d.id "
                "WHERE c.id = ?",
                (chunk_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "document_id": row[1],
            "content": row[2],
            "filename": row[3],
        }

    def get_chunks_with_documents_batch(
        self, chunk_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        """Batch-fetch multiple chunks with their document metadata.

        Returns a dict mapping chunk_id → chunk_info, avoiding N+1 queries
        when formatting search results.
        """
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            cursor = connection.execute(
                f"SELECT c.id, c.document_id, c.content, d.filename "
                f"FROM chunks c JOIN documents d ON c.document_id = d.id "
                f"WHERE c.id IN ({placeholders})",
                chunk_ids,
            )
            rows = cursor.fetchall()
        return {
            row[0]: {
                "id": row[0],
                "document_id": row[1],
                "content": row[2],
                "filename": row[3],
            }
            for row in rows
        }

    # -- FTS5 full-text search ------------------------------------------------

    def search_chunks_fts(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search chunks using SQLite FTS5 (if available).

        Returns a list of dicts with keys: chunk_id, document_id, content,
        filename, rank.  Empty list if FTS5 is not available.
        """
        limit = clamp_limit(limit, default=10)
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "SELECT c.id, c.document_id, c.content, d.filename, "
                    "       rank as fts_rank "
                    "FROM chunks_fts "
                    "JOIN chunks c ON chunks_fts.rowid = c.id "
                    "JOIN documents d ON c.document_id = d.id "
                    "WHERE chunks_fts MATCH ? "
                    "ORDER BY rank "
                    "LIMIT ?",
                    (query, limit),
                )
                rows = cursor.fetchall()
            return [
                {
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "content": row[2],
                    "filename": row[3],
                    "rank": row[4],
                }
                for row in rows
            ]
        except sqlite3.OperationalError:
            return []

    # -- Knowledge metadata ----------------------------------------------------

    def get_knowledge_meta(self, key: str) -> str | None:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT value FROM knowledge_meta WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def set_knowledge_meta(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO knowledge_meta(key, value) VALUES (?, ?)",
                (key, value),
            )
            connection.commit()

    # -- LLM Model Configs ----------------------------------------------------

    def add_model(
        self,
        name: str,
        provider: str,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        is_active: bool = False,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO llm_models(name, provider, base_url, api_key, model_name, "
                "temperature, max_tokens, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, provider, base_url, api_key, model_name, temperature, max_tokens, int(is_active)),
            )
            connection.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_all_models(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, name, provider, base_url, api_key, model_name, "
                "temperature, max_tokens, is_active, created_at, updated_at "
                "FROM llm_models ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "provider": row[2],
                "base_url": row[3],
                "api_key": row[4],
                "model_name": row[5],
                "temperature": row[6],
                "max_tokens": row[7],
                "is_active": bool(row[8]),
                "created_at": row[9],
                "updated_at": row[10],
            }
            for row in rows
        ]

    def get_model(self, model_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, name, provider, base_url, api_key, model_name, "
                "temperature, max_tokens, is_active, created_at, updated_at "
                "FROM llm_models WHERE id = ?",
                (model_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "provider": row[2],
            "base_url": row[3],
            "api_key": row[4],
            "model_name": row[5],
            "temperature": row[6],
            "max_tokens": row[7],
            "is_active": bool(row[8]),
            "created_at": row[9],
            "updated_at": row[10],
        }

    def update_model(self, model_id: int, **kwargs: Any) -> bool:
        allowed = {
            "name", "provider", "base_url", "api_key", "model_name",
            "temperature", "max_tokens", "is_active",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values())
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE llm_models SET {set_clause}, updated_at = datetime('now') WHERE id = ?",
                (*values, model_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def delete_model(self, model_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM llm_models WHERE id = ?", (model_id,)
            )
            connection.commit()
            return cursor.rowcount > 0

    def get_active_model(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, name, provider, base_url, api_key, model_name, "
                "temperature, max_tokens, is_active, created_at, updated_at "
                "FROM llm_models WHERE is_active = 1 LIMIT 1"
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "provider": row[2],
            "base_url": row[3],
            "api_key": row[4],
            "model_name": row[5],
            "temperature": row[6],
            "max_tokens": row[7],
            "is_active": bool(row[8]),
            "created_at": row[9],
            "updated_at": row[10],
        }

    def set_active_model(self, model_id: int) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE llm_models SET is_active = 0, updated_at = datetime('now')")
            connection.execute(
                "UPDATE llm_models SET is_active = 1, updated_at = datetime('now') WHERE id = ?",
                (model_id,),
            )
            connection.commit()

    # -- Long-term memory ----------------------------------------------------

    def set_long_term_memory(self, key: str, value: str, category: str = "general") -> None:
        """Store a long-term memory fact."""
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO long_term_memory(key, value, category, updated_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (key, value, category),
            )
            connection.commit()

    def get_long_term_memory(self, key: str) -> str | None:
        """Retrieve a single memory by key."""
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT value FROM long_term_memory WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def search_long_term_memory(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories by key or value (simple LIKE match)."""
        limit = clamp_limit(limit, default=10)
        pattern = f"%{query}%"
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT key, value, category, updated_at FROM long_term_memory "
                "WHERE key LIKE ? OR value LIKE ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (pattern, pattern, limit),
            )
            rows = cursor.fetchall()
        return [
            {"key": r[0], "value": r[1], "category": r[2], "updated_at": r[3]}
            for r in rows
        ]

    def search_long_term_memory_batch(
        self, terms: list[str], limit_per_term: int = 3,
    ) -> list[dict[str, Any]]:
        """Search memories for multiple terms in a single query.

        Merges results with deduplication by key.  Avoids N separate queries
        when recalling memories for a multi-term question.
        """
        if not terms:
            return []
        clauses = " OR ".join("key LIKE ? OR value LIKE ?" for _ in terms)
        params: list[Any] = []
        for term in terms:
            pattern = f"%{term}%"
            params.extend([pattern, pattern])
        # Use a generous total limit; dedup happens in Python
        total_limit = len(terms) * limit_per_term
        with self._connect() as connection:
            cursor = connection.execute(
                f"SELECT key, value, category, updated_at FROM long_term_memory "
                f"WHERE {clauses} "
                f"ORDER BY updated_at DESC LIMIT ?",
                (*params, total_limit),
            )
            rows = cursor.fetchall()
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                results.append({"key": r[0], "value": r[1], "category": r[2], "updated_at": r[3]})
        return results

    def list_long_term_memories(self, category: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """List all memories, optionally filtered by category."""
        limit = clamp_limit(limit, default=50)
        if category:
            with self._connect() as connection:
                cursor = connection.execute(
                    "SELECT key, value, category, updated_at FROM long_term_memory "
                    "WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                    (category, limit),
                )
                rows = cursor.fetchall()
        else:
            with self._connect() as connection:
                cursor = connection.execute(
                    "SELECT key, value, category, updated_at FROM long_term_memory "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
        return [
            {"key": r[0], "value": r[1], "category": r[2], "updated_at": r[3]}
            for r in rows
        ]

    def delete_long_term_memory(self, key: str) -> bool:
        """Delete a single memory by key."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM long_term_memory WHERE key = ?", (key,)
            )
            connection.commit()
            return cursor.rowcount > 0

    def clear_long_term_memories(self, category: str = "") -> None:
        """Clear all memories, optionally filtered by category."""
        with self._connect() as connection:
            if category:
                connection.execute(
                    "DELETE FROM long_term_memory WHERE category = ?", (category,)
                )
            else:
                connection.execute("DELETE FROM long_term_memory")
            connection.commit()

    # ------------------------------------------------------------------
    # Execution records
    # ------------------------------------------------------------------

    def save_execution(
        self,
        session_id: int | None,
        trace_id: str,
        question: str,
        answer: str,
        goal_type: str,
        trace_json: str,
        errors_json: str,
        degraded: bool,
        duration_ms: float,
    ) -> int:
        """Persist a completed workflow execution record."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO executions(session_id, trace_id, question, answer, "
                "goal_type, trace_json, errors_json, degraded, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, trace_id, question, answer, goal_type,
                 trace_json, errors_json, int(degraded), duration_ms),
            )
            connection.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_execution(self, exec_id: int) -> dict[str, Any] | None:
        """Retrieve a single execution record by ID."""
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, session_id, trace_id, question, answer, goal_type, "
                "trace_json, errors_json, degraded, duration_ms, created_at "
                "FROM executions WHERE id = ?",
                (exec_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "session_id": row[1],
            "trace_id": row[2],
            "question": row[3],
            "answer": row[4],
            "goal_type": row[5],
            "trace": row[6],
            "errors": row[7],
            "degraded": bool(row[8]),
            "duration_ms": row[9],
            "created_at": row[10],
        }

    def list_executions(
        self, session_id: int | None = None, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List execution records, optionally filtered by session."""
        limit = clamp_limit(limit, default=20)
        if session_id is not None:
            with self._connect() as connection:
                cursor = connection.execute(
                    "SELECT id, session_id, trace_id, question, answer, goal_type, "
                    "trace_json, errors_json, degraded, duration_ms, created_at "
                    "FROM executions WHERE session_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                )
                rows = cursor.fetchall()
        else:
            with self._connect() as connection:
                cursor = connection.execute(
                    "SELECT id, session_id, trace_id, question, answer, goal_type, "
                    "trace_json, errors_json, degraded, duration_ms, created_at "
                    "FROM executions ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
        return [
            {
                "id": r[0], "session_id": r[1], "trace_id": r[2],
                "question": r[3], "answer": r[4], "goal_type": r[5],
                "trace": r[6], "errors": r[7],
                "degraded": bool(r[8]), "duration_ms": r[9],
                "created_at": r[10],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Execution checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, execution_id: int, node_name: str, task_queue_json: str,
    ) -> int:
        """Persist a task queue checkpoint after a node execution."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO execution_checkpoints(execution_id, node_name, task_queue_json) "
                "VALUES (?, ?, ?)",
                (execution_id, node_name, task_queue_json),
            )
            connection.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def list_checkpoints(
        self, execution_id: int,
    ) -> list[dict[str, Any]]:
        """List all checkpoints for an execution, in chronological order."""
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, node_name, task_queue_json, created_at "
                "FROM execution_checkpoints "
                "WHERE execution_id = ? ORDER BY id ASC",
                (execution_id,),
            )
            rows = cursor.fetchall()
        return [
            {
                "id": r[0], "node_name": r[1],
                "task_queue_json": r[2], "created_at": r[3],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Cleanup methods (session timeout, TTL)
    # ------------------------------------------------------------------

    def delete_sessions_older_than(self, hours: int) -> int:
        """Delete sessions created more than ``hours`` ago.

        Chats are cascaded via FK ON DELETE CASCADE.
        Returns the number of deleted sessions.
        """
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            cursor = connection.execute(
                "DELETE FROM sessions WHERE created_at < datetime('now', ?)",
                (f"-{hours} hours",),
            )
            connection.commit()
            count = cursor.rowcount
        if count:
            _log.info("Cleaned up %d sessions older than %d hours", count, hours)
        return count

    def delete_old_memories(self, days: int) -> int:
        """Delete long-term memory entries not updated in ``days``."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM long_term_memory WHERE updated_at < datetime('now', ?)",
                (f"-{days} days",),
            )
            connection.commit()
            count = cursor.rowcount
        if count:
            _log.info("Cleaned up %d memory entries older than %d days", count, days)
        return count
