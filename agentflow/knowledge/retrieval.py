"""Hybrid retrieval pipeline: vector similarity + lexical (FTS5) search.

The ``HybridRetriever`` runs both retrievers in parallel and fuses results
via Reciprocal Rank Fusion (RRF).  If no ANN index is available it falls
back to brute-force cosine similarity over all stored embeddings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.embedder import (
    BaseEmbedder,
    batch_cosine_similarity,
    deserialize_vector,
)
from agentflow.knowledge.index import ANNIndex
from agentflow.knowledge.settings import hybrid_weights, search_defaults

logger = logging.getLogger("knowledge.retrieval")


class HybridRetriever:
    """Combines vector (dense) and lexical (FTS5) search with configurable fusion.

    Parameters
    ----------
    embedder : BaseEmbedder
        The embedding model used for vector search.
    db : SQLiteStore
        Database handle for FTS5 and embedding lookups.
    ann_index : ANNIndex | None
        Optional ANN index for fast vector search (O(log N) instead of O(N)).
    alpha : float
        Weight for vector similarity score (default 0.7).
    beta : float
        Weight for lexical score (default 0.3).
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        db: SQLiteStore,
        ann_index: ANNIndex | None = None,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> None:
        self.embedder = embedder
        self.db = db
        self.ann_index = ann_index
        if alpha is not None and beta is not None:
            self.alpha = alpha
            self.beta = beta
        else:
            self.alpha, self.beta = hybrid_weights()

    # -- Public API ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid vector + lexical search.

        Returns
        -------
        list[dict]
            Each dict has keys:
                chunk_id, document_id, filename, content,
                score (final hybrid), vector_score, lexical_score,
                method ("hybrid" | "vector" | "lexical")
        """
        top_k, min_score = search_defaults() if top_k is None else (top_k, min_score or 0.0)

        # 1. Run both searches
        query_vec = self.embedder.embed_query(query)
        vector_results = self._vector_search(query_vec, top_k * 2)
        lexical_results = self._lexical_search(query, top_k * 2)

        # 2. If only one side has results, return that
        if not vector_results and not lexical_results:
            return []
        if not vector_results:
            return self._format_lexical_only(lexical_results, top_k, min_score)
        if not lexical_results:
            return self._format_vector_only(vector_results, query_vec, top_k, min_score)

        # 3. Hybrid fusion via RRF
        merged = self._rrf_fusion(vector_results, lexical_results, top_k)

        # 4. Augment with metadata and filter by score
        results: list[dict[str, Any]] = []
        for chunk_id, score, v_score, l_score in merged:
            if score < min_score:
                continue
            chunk_info = self.db.get_chunk_with_document(chunk_id)
            if not chunk_info:
                continue
            results.append({
                "chunk_id": chunk_id,
                "document_id": chunk_info["document_id"],
                "filename": chunk_info["filename"],
                "content": chunk_info["content"],
                "score": round(float(score), 4),
                "vector_score": round(float(v_score), 4),
                "lexical_score": round(float(l_score), 4),
                "method": "hybrid",
            })

        return results[:top_k]

    # -- Vector search ------------------------------------------------------

    def _vector_search(
        self, query_vec: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """Vector (dense) retrieval — ANN preferred, brute-force fallback."""
        if self.ann_index and self.ann_index.is_loaded and self.ann_index.size > 0:
            try:
                return self.ann_index.search(query_vec, top_k)
            except Exception as exc:
                logger.warning("ANN search failed, falling back to brute-force: %s", exc)

        return self._brute_force_search(query_vec, top_k)

    def _brute_force_search(
        self, query_vec: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """Fallback: load all embeddings from DB and compute cosine similarity."""
        all_emb = self.db.get_all_embeddings_with_chunk()
        if not all_emb:
            return []

        candidates: list[tuple[int, np.ndarray]] = []
        for emb in all_emb:
            chunk_id = emb["chunk_id"]
            vec = deserialize_vector(emb["embedding"])
            candidates.append((chunk_id, vec))

        scored = batch_cosine_similarity(query_vec, candidates)
        return scored[:top_k]

    # -- Lexical search (FTS5) ---------------------------------------------

    def _lexical_search(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        """Lexical (keyword) search via SQLite FTS5.

        Returns [(chunk_id, score), ...] where score is derived from
        the FTS5 rank (BM25-style).
        """
        # Prepare FTS5 query: escape special characters and tokenise
        fts_query = self._to_fts_query(query)
        if not fts_query:
            return []

        rows = self.db.search_chunks_fts(fts_query, limit=top_k)
        if not rows:
            return []

        results: list[tuple[int, float]] = []
        for row in rows:
            chunk_id = row["chunk_id"]
            rank = row["rank"]
            # Convert FTS5 rank (negative = better) to a normalised [0, 1] score
            # BM25 rank is typically negative; more negative = better match
            score = 1.0 / (1.0 + abs(rank))
            results.append((chunk_id, score))

        return results

    @staticmethod
    def _to_fts_query(query: str) -> str:
        """Convert a user query to an FTS5 query string.

        Splits on whitespace and joins with OR. Each term is quoted to
        prevent FTS5 reserved keywords (OR, NOT, AND) from being
        interpreted as operators.
        """
        terms = []
        for part in query.split():
            part = part.strip().strip('"\'(),.!?:;')
            if part:
                terms.append(part)
        if not terms:
            return ""
        return " OR ".join(f'"{t}"' for t in terms)

    # -- RRF fusion ---------------------------------------------------------

    @staticmethod
    def _rrf_fusion(
        vector_results: list[tuple[int, float]],
        lexical_results: list[tuple[int, float]],
        top_k: int,
        k: int = 60,
    ) -> list[tuple[int, float, float, float]]:
        """Reciprocal Rank Fusion with score carry-over.

        Returns
        -------
        list[(chunk_id, hybrid_score, vector_score, lexical_score)]
        """
        # Build rank maps
        v_ranks: dict[int, int] = {
            cid: idx for idx, (cid, _) in enumerate(vector_results)
        }
        l_ranks: dict[int, int] = {
            cid: idx for idx, (cid, _) in enumerate(lexical_results)
        }

        # Score maps (raw similarity)
        v_scores: dict[int, float] = dict(vector_results)
        l_scores: dict[int, float] = dict(lexical_results)

        # Collect all unique chunk_ids
        all_ids = set(v_ranks) | set(l_ranks)

        scored: list[tuple[int, float, float, float]] = []
        for cid in all_ids:
            vr = v_ranks.get(cid, top_k * 2)
            lr = l_ranks.get(cid, top_k * 2)
            vs = v_scores.get(cid, 0.0)
            ls_ = l_scores.get(cid, 0.0)
            # RRF score
            hybrid_score = (1.0 / (k + vr)) + (1.0 / (k + lr))
            scored.append((cid, hybrid_score, vs, ls_))

        # Sort by hybrid score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # -- Format helpers for single-strategy results -------------------------

    def _format_vector_only(
        self,
        vector_results: list[tuple[int, float]],
        query_vec: np.ndarray,
        top_k: int,
        min_score: float,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for chunk_id, score in vector_results:
            if score < min_score:
                continue
            if len(results) >= top_k:
                break
            chunk_info = self.db.get_chunk_with_document(chunk_id)
            if chunk_info:
                results.append({
                    "chunk_id": chunk_id,
                    "document_id": chunk_info["document_id"],
                    "filename": chunk_info["filename"],
                    "content": chunk_info["content"],
                    "score": round(score, 4),
                    "vector_score": round(score, 4),
                    "lexical_score": 0.0,
                    "method": "vector",
                })
        return results

    def _format_lexical_only(
        self,
        lexical_results: list[tuple[int, float]],
        top_k: int,
        min_score: float,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for chunk_id, score in lexical_results:
            if score < min_score:
                continue
            if len(results) >= top_k:
                break
            chunk_info = self.db.get_chunk_with_document(chunk_id)
            if chunk_info:
                results.append({
                    "chunk_id": chunk_id,
                    "document_id": chunk_info["document_id"],
                    "filename": chunk_info["filename"],
                    "content": chunk_info["content"],
                    "score": round(score, 4),
                    "vector_score": 0.0,
                    "lexical_score": round(score, 4),
                    "method": "lexical",
                })
        return results
