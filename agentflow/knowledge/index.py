"""ChromaDB vector index — drop-in replacement for the old FAISS ANNIndex.

``ChromaIndex`` wraps a persistent ChromaDB collection for vector storage
and ANN search.  ChromaDB handles persistence, indexing, and CRUD
internally — no manual load/save/rebuild needed.

Usage::

    from agentflow.knowledge.index import ChromaIndex

    index = ChromaIndex()
    index.add([1, 2, 3], vectors, metadatas)
    results = index.search(query_vec, 10)  # [(chunk_id, score), ...]
    index.remove([2])                      # remove by chunk_id
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from agentflow.knowledge.settings import chroma_collection, chroma_path

logger = logging.getLogger("knowledge.index")


class ChromaIndex:
    """Flag used internally when ``in_memory`` is set."""
    _client_override: Any = None
    """ChromaDB-backed vector index with persistent storage.

    Parameters
    ----------
    collection_name : str | None
        ChromaDB collection name.  ``None`` = default from settings.
    persist_path : str | Path | None
        On-disk path for ChromaDB data.  ``None`` = default from settings.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self._collection_name = collection_name or chroma_collection()
        self._persist_path = Path(persist_path) if persist_path else chroma_path()
        self._collection = None  # lazy-init

    @classmethod
    def in_memory(cls, collection_name: str = "test") -> ChromaIndex:
        """Create an in-memory ChromaIndex for testing."""
        import chromadb
        idx = cls(collection_name=collection_name)
        idx._client_override = chromadb.EphemeralClient()
        return idx

    # -- Public API ---------------------------------------------------------

    def add(
        self,
        chunk_ids: list[int],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors indexed by *chunk_ids*.

        Parameters
        ----------
        chunk_ids : list[int]
            Unique chunk IDs (must be convertible to strings).
        vectors : np.ndarray
            2-D array of shape ``(n, dimension)``.
        metadatas : list[dict] | None
            Optional per-vector metadata (e.g. ``{"document_id": ...}``).
        """
        if len(chunk_ids) == 0:
            return
        collection = self._get_collection()
        collection.add(
            ids=[str(cid) for cid in chunk_ids],
            embeddings=vectors.tolist(),
            metadatas=metadatas,
        )
        logger.debug("Added %d vectors to ChromaDB collection '%s'", len(chunk_ids), self._collection_name)

    def remove(self, chunk_ids: list[int]) -> None:
        """Remove vectors by their chunk IDs."""
        if not chunk_ids:
            return
        collection = self._get_collection()
        collection.delete(ids=[str(cid) for cid in chunk_ids])
        logger.debug("Removed %d vectors from ChromaDB collection '%s'", len(chunk_ids), self._collection_name)

    def search(
        self, query_vec: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Search the vector index.

        Returns
        -------
        list[(chunk_id, score), ...]
            Sorted descending by similarity score.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=query_vec.reshape(1, -1).tolist(),
            n_results=min(top_k, collection.count()),
        )
        # results: {"ids": [["id1", "id2", ...]],
        #            "distances": [[d1, d2, ...]],
        #            "metadatas": [[{}, ...]],
        #            "documents": ...}

        chunk_ids = [int(id_) for id_ in results["ids"][0]]
        # ChromaDB cosine distance → similarity score
        # cosine distance = 1 - cos(a,b), so score = 1 - distance
        scores = [1.0 - float(d) for d in results["distances"][0]]
        return list(zip(chunk_ids, scores))

    @property
    def size(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    # -- Internal -----------------------------------------------------------

    def _get_collection(self):
        """Lazy-init the ChromaDB collection."""
        if self._collection is not None:
            return self._collection
        import chromadb
        client = self._client_override or chromadb.PersistentClient(
            path=str(self._persist_path)
        )
        if self._client_override is None:
            self._persist_path.mkdir(parents=True, exist_ok=True)
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d vectors)",
            self._collection_name,
            self._collection.count(),
        )
        return self._collection
