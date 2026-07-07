"""SQLite-backed persistence for chat history and knowledge base."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from agentflow.config.settings import settings


class SQLiteStore:
    """Simple SQLite-backed persistence for chat/history and knowledge base data."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
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

            connection.commit()

    # -- Chat history / Sessions -----------------------------------------------

    def create_session(self, title: str = "新对话") -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO sessions(title, created_at, updated_at) "
                "VALUES (?, datetime('now'), datetime('now'))",
                (title,),
            )
            connection.commit()
            return self.get_session(cursor.lastrowid)  # type: ignore[arg-type]

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as connection:
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
        with sqlite3.connect(self.db_path) as connection:
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
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "UPDATE sessions SET title = ?, updated_at = datetime('now') WHERE id = ?",
                (title, session_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def update_session_state(self, session_id: int, state_json: str) -> bool:
        """Persist serialized session_state JSON for a session."""
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "UPDATE sessions SET session_state = ?, updated_at = datetime('now') WHERE id = ?",
                (state_json, session_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def get_session_state(self, session_id: int) -> str:
        """Load serialized session_state JSON for a session."""
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT session_state FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
        return row[0] if row and row[0] else ""

    def delete_session(self, session_id: int) -> bool:
        with sqlite3.connect(self.db_path) as connection:
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
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO chats(session_id, role, content, created_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (session_id, role, content),
            )
            connection.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def list_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as connection:
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
        with sqlite3.connect(self.db_path) as connection:
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
        self, filename: str, file_type: str, file_size: int, doc_metadata: str = "{}"
    ) -> int:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO documents(filename, file_type, file_size, doc_metadata) VALUES (?, ?, ?, ?)",
                (filename, file_type, file_size, doc_metadata),
            )
            connection.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_all_documents(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as connection:
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
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            connection.commit()

    # -- Chunks ----------------------------------------------------------------

    def add_chunk(self, document_id: int, content: str, chunk_index: int) -> int:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO chunks(document_id, content, chunk_index) VALUES (?, ?, ?)",
                (document_id, content, chunk_index),
            )
            connection.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_chunks_by_document(self, doc_id: int) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT id, content, chunk_index FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc_id,),
            )
            rows = cursor.fetchall()
        return [
            {"id": row[0], "content": row[1], "chunk_index": row[2]} for row in rows
        ]

    def get_chunk_with_document(self, chunk_id: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as connection:
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

    # -- FTS5 full-text search ------------------------------------------------

    def search_chunks_fts(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search chunks using SQLite FTS5 (if available).

        Returns a list of dicts with keys: chunk_id, document_id, content,
        filename, rank.  Empty list if FTS5 is not available.
        """
        try:
            with sqlite3.connect(self.db_path) as connection:
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
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT value FROM knowledge_meta WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def set_knowledge_meta(self, key: str, value: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
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
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO llm_models(name, provider, base_url, api_key, model_name, "
                "temperature, max_tokens, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, provider, base_url, api_key, model_name, temperature, max_tokens, int(is_active)),
            )
            connection.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_all_models(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as connection:
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
        with sqlite3.connect(self.db_path) as connection:
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
        fields = {k: v for k, v in kwargs.items() if v is not None}
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values())
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                f"UPDATE llm_models SET {set_clause}, updated_at = datetime('now') WHERE id = ?",
                (*values, model_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def delete_model(self, model_id: int) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "DELETE FROM llm_models WHERE id = ?", (model_id,)
            )
            connection.commit()
            return cursor.rowcount > 0

    def get_active_model(self) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as connection:
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
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("UPDATE llm_models SET is_active = 0, updated_at = datetime('now')")
            connection.execute(
                "UPDATE llm_models SET is_active = 1, updated_at = datetime('now') WHERE id = ?",
                (model_id,),
            )
            connection.commit()

    # -- Long-term memory ----------------------------------------------------

    def set_long_term_memory(self, key: str, value: str, category: str = "general") -> None:
        """Store a long-term memory fact."""
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO long_term_memory(key, value, category, updated_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (key, value, category),
            )
            connection.commit()

    def get_long_term_memory(self, key: str) -> str | None:
        """Retrieve a single memory by key."""
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT value FROM long_term_memory WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def search_long_term_memory(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories by key or value (simple LIKE match)."""
        pattern = f"%{query}%"
        with sqlite3.connect(self.db_path) as connection:
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

    def list_long_term_memories(self, category: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """List all memories, optionally filtered by category."""
        if category:
            with sqlite3.connect(self.db_path) as connection:
                cursor = connection.execute(
                    "SELECT key, value, category, updated_at FROM long_term_memory "
                    "WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                    (category, limit),
                )
                rows = cursor.fetchall()
        else:
            with sqlite3.connect(self.db_path) as connection:
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
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "DELETE FROM long_term_memory WHERE key = ?", (key,)
            )
            connection.commit()
            return cursor.rowcount > 0

    def clear_long_term_memories(self, category: str = "") -> None:
        """Clear all memories, optionally filtered by category."""
        with sqlite3.connect(self.db_path) as connection:
            if category:
                connection.execute(
                    "DELETE FROM long_term_memory WHERE category = ?", (category,)
                )
            else:
                connection.execute("DELETE FROM long_term_memory")
            connection.commit()
