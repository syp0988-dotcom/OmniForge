"""Persistent SQLite-backed cache for embedding vectors.

Embedding API calls are the dominant latency/cost factor in knowledge-base
searches.  Caching vectors by (model, text-hash) means repeated or similar
queries hit the local cache instead of the remote API.

The cache is a standalone SQLite database (``data/embedding_cache.db`` by
default) so it survives restarts without touching the main application DB.
Vectors are stored as JSON arrays; the values are opaque to this module.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from agentflow.config.settings import settings
from agentflow.utils.logging import build_logger

logger = build_logger("embedding_cache")


class EmbeddingCache:
    """Thread-safe persistent cache keyed by ``(model, sha256(text))``."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else Path(
            settings.embedding_cache_path or settings.project_root / "data" / "embedding_cache.db"
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (text_hash, model)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def lookup(
        self, model: str, texts: list[str],
    ) -> tuple[list[np.ndarray | None], list[int], list[str]]:
        """Look up vectors for *texts* in order.

        Returns ``(vectors, missing_indices, missing_texts)`` where *vectors*
        is aligned with *texts* (``None`` for cache misses) and
        *missing_indices* / *missing_texts* describe the misses.
        """
        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        if not texts:
            return vectors, missing_indices, missing_texts

        hashes = [self._text_hash(t) for t in texts]
        with self._lock:
            for idx, (text, text_hash) in enumerate(zip(texts, hashes)):
                row = self._conn.execute(
                    "SELECT vector_json FROM embedding_cache "
                    "WHERE text_hash = ? AND model = ?",
                    (text_hash, model),
                ).fetchone()
                if row is not None:
                    vectors[idx] = np.array(
                        json.loads(row[0]), dtype=np.float32,
                    )
                else:
                    missing_indices.append(idx)
                    missing_texts.append(text)
        return vectors, missing_indices, missing_texts

    def store(self, model: str, texts: list[str], vectors: list[np.ndarray]) -> None:
        """Persist vectors for *texts* (bulk upsert)."""
        if not texts:
            return
        rows = [
            (self._text_hash(t), model, json.dumps(vec.tolist()))
            for t, vec in zip(texts, vectors)
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache(text_hash, model, vector_json) "
                "VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def merge(
        self,
        cached: list[np.ndarray | None],
        missing_indices: list[int],
        new_vectors: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Fill *cached* (None placeholders) with *new_vectors* and return it."""
        for idx, vec in zip(missing_indices, new_vectors):
            cached[idx] = vec
        return [v for v in cached if v is not None]

    def size(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM embedding_cache"
            ).fetchone()
        return int(row[0]) if row else 0


_shared_cache: EmbeddingCache | None = None


def get_shared_cache() -> EmbeddingCache | None:
    """Return the process-wide cache (``None`` when disabled by settings)."""
    global _shared_cache
    if not settings.embedding_cache_enabled:
        return None
    if _shared_cache is None:
        try:
            _shared_cache = EmbeddingCache()
        except Exception as exc:
            logger.warning("Embedding cache disabled: %s", exc)
            _shared_cache = None
    return _shared_cache
