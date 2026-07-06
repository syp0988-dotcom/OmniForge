"""FAISS ANN (Approximate Nearest Neighbour) index management.

Provides ``ANNIndex`` — a lazy-loading, incrementally-updatable HNSW/Flat
index backed by a persistent on-disk file.

Usage::

    from agentflow.knowledge.index import ANNIndex

    index = ANNIndex(dimension=384)
    index.load()                           # load from disk or create new
    index.add([1, 2, 3], vectors)          # chunk_ids + vectors
    results = index.search(query_vec, 10)  # [(chunk_id, score), ...]
    index.remove([2])                      # remove by chunk_id
    index.save()                           # persist
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from agentflow.knowledge.settings import faiss_index_path, faiss_index_type

logger = logging.getLogger("knowledge.index")


class ANNIndex:
    """FAISS ANN index with lazy loading and incremental updates.

    Parameters
    ----------
    dimension : int
        Vector dimension (must match embedder output).
    index_path : str | Path | None
        On-disk path for the FAISS index file.  ``None`` = default
        (``<project_root>/data/faiss.index``).
    index_type : str
        ``"HNSW"`` (default, fast) or ``"Flat"`` (exact, slower).
    """

    def __init__(
        self,
        dimension: int,
        index_path: str | Path | None = None,
        index_type: str | None = None,
    ) -> None:
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else faiss_index_path()
        self.index_type = (index_type or faiss_index_type()).upper()
        self._index = None
        # faiss internal_id → chunk_id
        self._id_map: dict[int, int] = {}
        # chunk_id → faiss internal_id (reverse lookup for removal)
        self._reverse_id_map: dict[int, int] = {}
        self._loaded = False
        self._needs_rebuild = False

    # -- Public API ---------------------------------------------------------

    def load(self) -> None:
        """Load the index from disk, or create a new empty one.

        Safe to call multiple times — subsequent calls are no-ops if
        already loaded.
        """
        if self._loaded:
            return
        self._ensure_parent()
        self._index = self._build_index()
        self._index_path().parent.mkdir(parents=True, exist_ok=True)

        if self._index_path().exists():
            try:
                import faiss
                self._index = faiss.read_index(str(self._index_path()))
                logger.info(
                    "Loaded FAISS index from %s (%s entries, type=%s)",
                    self._index_path(), self._index.ntotal, self.index_type,
                )
            except Exception as exc:
                logger.warning("Failed to load FAISS index, creating new: %s", exc)
                self._index = self._build_index()
        else:
            logger.info("Creating new FAISS %s index (dim=%d)", self.index_type, self.dimension)

        self._loaded = True

    def save(self) -> None:
        """Persist the index to disk.

        If the index needs rebuilding (e.g. after unsupported removal),
        all vectors are re-read from the database and the index is
        reconstructed from scratch.
        """
        if self._index is None or not self._loaded:
            logger.debug("No index to save — skipping.")
            return
        import faiss

        if self._needs_rebuild and self._index.ntotal > 0:
            logger.info("Rebuilding FAISS index from existing vectors…")
            # Rebuild: extract all vectors and re-add
            embeddings = self._index.reconstruct_n(0, self._index.ntotal)
            ids = np.array([self._index.id_map.at(i) for i in range(self._index.ntotal)], dtype=np.int64)
            self._index = self._build_index()
            self._index.add_with_ids(embeddings, ids)
            self._needs_rebuild = False

        self._ensure_parent()
        faiss.write_index(self._index, str(self._index_path()))
        logger.debug("Saved FAISS index to %s (%d entries)", self._index_path(), self._index.ntotal)

    def add(self, chunk_ids: list[int], vectors: np.ndarray) -> None:
        """Add vectors indexed by *chunk_ids*.

        Both lists must have the same length.  Vectors should be a 2-D
        array of shape ``(n, dimension)``.
        """
        if len(chunk_ids) == 0:
            return
        self.load()  # ensure loaded
        import faiss

        n = len(chunk_ids)
        assert vectors.shape == (n, self.dimension), (
            f"Expected ({n}, {self.dimension}) got {vectors.shape}"
        )

        # Use IndexIDMap so we can reference vectors by chunk_id
        faiss_ids = np.array(chunk_ids, dtype=np.int64)
        self._index.add_with_ids(vectors, faiss_ids)

        for cid, fid in zip(chunk_ids, faiss_ids):
            self._id_map[int(fid)] = int(cid)
            self._reverse_id_map[int(cid)] = int(fid)

        logger.debug("Added %d vectors to FAISS index (total=%d)", n, self._index.ntotal)

    def remove(self, chunk_ids: list[int]) -> None:
        """Remove vectors by their chunk IDs.

        Note: HNSW indices wrapped in ``IndexIDMap`` do not support
        ``remove_ids``.  In that case the removal is logged and the
        caller should rebuild the index periodically.
        """
        if not chunk_ids or self._index is None or not self._loaded:
            return

        faiss_ids: list[int] = []
        for cid in chunk_ids:
            fid = self._reverse_id_map.pop(int(cid), None)
            if fid is not None:
                faiss_ids.append(fid)
                self._id_map.pop(fid, None)

        if not faiss_ids:
            return

        try:
            import faiss
            ids_to_remove = np.array(faiss_ids, dtype=np.int64)
            self._index.remove_ids(ids_to_remove)
            logger.debug("Removed %d vectors from FAISS index", len(faiss_ids))
        except RuntimeError as exc:
            # HNSW does not support removal — mark dirty for later rebuild
            logger.warning(
                "FAISS removal not supported for this index type (%s). "
                "The index will be rebuilt on next save. Error: %s",
                self.index_type, exc,
            )
            self._needs_rebuild = True

    def search(
        self, query_vec: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Search the ANN index.

        Returns
        -------
        list[(chunk_id, score), ...]
            Sorted descending by similarity score.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query_2d = query_vec.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query_2d, min(top_k, self._index.ntotal))

        results: list[tuple[int, float]] = []
        for dist, fid in zip(distances[0], indices[0]):
            if fid == -1:
                continue
            chunk_id = self._id_map.get(int(fid), int(fid))
            # FAISS returns L2 distances; convert to similarity score
            # score = 1 / (1 + distance)
            score = 1.0 / (1.0 + float(dist))
            results.append((chunk_id, score))

        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def size(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    # -- State serialization (for id_map persistence) -----------------------

    def save_metadata(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for storing alongside DB data."""
        return {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "reverse_id_map": {str(k): v for k, v in self._reverse_id_map.items()},
            "dimension": self.dimension,
            "index_type": self.index_type,
        }

    def load_metadata(self, data: dict[str, Any]) -> None:
        """Restore metadata from a previously saved dict."""
        self._id_map = {int(k): v for k, v in data.get("id_map", {}).items()}
        self._reverse_id_map = {int(k): v for k, v in data.get("reverse_id_map", {}).items()}
        self.dimension = data.get("dimension", self.dimension)
        self.index_type = data.get("index_type", self.index_type)

    # -- Internal -----------------------------------------------------------

    def _index_path(self) -> Path:
        return self.index_path

    def _ensure_parent(self) -> None:
        self._index_path().parent.mkdir(parents=True, exist_ok=True)

    def _build_index(self):
        import faiss
        if self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # M=32 neighbours
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
        else:
            index = faiss.IndexFlatL2(self.dimension)
        # Wrap for ID-mapped operations
        return faiss.IndexIDMap(index)
