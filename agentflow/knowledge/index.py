"""Qdrant vector index for ANN search.

``QdrantIndex`` wraps a Qdrant collection for vector storage and search.
Supports both local (on-disk / in-memory) and remote (server) modes.

Usage::

    from agentflow.knowledge.index import QdrantIndex

    index = QdrantIndex()
    index.add([1, 2, 3], vectors, metadatas)
    results = index.search(query_vec, 10)  # [(chunk_id, score), ...]
    index.remove([2])                      # remove by chunk_id
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from agentflow.knowledge.settings import (
    qdrant_api_key,
    qdrant_collection,
    qdrant_storage_path,
    qdrant_url,
)

logger = logging.getLogger("knowledge.index")


class QdrantIndex:
    """Qdrant-backed vector index with local persistent storage.

    Parameters
    ----------
    path : str | Path | None
        Local storage path.  ``:memory:`` for in-memory (testing).
        ``None`` = default path under ``data/qdrant``.
    url : str | None
        Remote Qdrant server URL.  Takes precedence over *path*.
    api_key : str | None
        API key for remote Qdrant Cloud.
    collection_name : str | None
        Collection name.  ``None`` = default from settings.
    dimension : int | None
        Vector dimension.  Set on first ``add()`` if unknown.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        dimension: int | None = None,
    ) -> None:
        from qdrant_client import QdrantClient

        self._collection_name = collection_name or qdrant_collection()
        self._url = url or qdrant_url()
        self._api_key = api_key or qdrant_api_key()
        self._dimension = dimension
        self._client: Any = None

        if self._api_key and self._url:
            self._client = QdrantClient(
                url=self._url, api_key=self._api_key, timeout=30.0,
            )
        elif path == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            _path = Path(path) if path else qdrant_storage_path()
            _path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(_path))

    @classmethod
    def in_memory(
        cls, collection_name: str = "test", dimension: int | None = None,
    ) -> QdrantIndex:
        """Create an in-memory QdrantIndex for testing."""
        return cls(path=":memory:", collection_name=collection_name, dimension=dimension)

    # -- Public API ---------------------------------------------------------

    def add(
        self,
        chunk_ids: list[int],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors indexed by *chunk_ids*."""
        if len(chunk_ids) == 0:
            return

        from qdrant_client.models import Distance, PointStruct, VectorParams

        # Lazy-init collection on first add
        if self._dimension is None:
            self._dimension = vectors.shape[1]
            try:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._dimension, distance=Distance.COSINE,
                    ),
                )
            except ValueError:
                pass  # collection already exists from previous session

        points = [
            PointStruct(
                id=cid,
                vector=vec.tolist(),
                payload=meta or {},
            )
            for cid, vec, meta in zip(
                chunk_ids, vectors,
                metadatas or [{} for _ in chunk_ids],
            )
        ]
        self._client.upsert(
            collection_name=self._collection_name, points=points,
        )
        logger.debug(
            "Added %d vectors to Qdrant collection '%s'",
            len(chunk_ids), self._collection_name,
        )

    def remove(self, chunk_ids: list[int]) -> None:
        """Remove vectors by their chunk IDs."""
        if not chunk_ids:
            return
        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=chunk_ids,
            )
        except ValueError:
            return  # collection not created yet (nothing to remove)
        logger.debug(
            "Removed %d vectors from Qdrant collection '%s'",
            len(chunk_ids), self._collection_name,
        )

    def search(
        self, query_vec: np.ndarray, top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search the vector index.

        Returns
        -------
        list[(chunk_id, score), ...]
            Sorted descending by similarity score.
        """
        if self.size == 0:
            return []
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vec.tolist(),
            limit=min(top_k, self.size),
        )
        return [(hit.id, hit.score) for hit in results.points]

    @property
    def size(self) -> int:
        try:
            return self._client.count(
                collection_name=self._collection_name,
            ).count
        except Exception:
            return 0

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def reset(self) -> None:
        """Drop and recreate the backing collection."""
        try:
            self._client.delete_collection(
                collection_name=self._collection_name,
            )
        except Exception as exc:
            logger.debug(
                "Qdrant collection reset ignored delete error: %s", exc,
            )
        self._dimension = None
