from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from agentflow.config.settings import settings


class SQLiteStore:
    """Simple SQLite-backed persistence for chat/history data."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def add_message(self, role: str, content: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO chats(role, content, created_at) VALUES (?, ?, datetime('now'))",
                (role, content),
            )
            connection.commit()

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
