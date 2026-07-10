"""Hybrid retrieval pipeline: vector similarity (Qdrant) + lexical (FTS5) search.

The ``HybridRetriever`` runs both retrievers in parallel and fuses results
via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.embedder import BaseEmbedder
from agentflow.knowledge.index import QdrantIndex
from agentflow.knowledge.settings import hybrid_weights, search_defaults

logger = logging.getLogger("knowledge.retrieval")


def _is_punct(s: str) -> bool:
    """Return True if *s* is a pure-punctuation / whitespace token."""
    import unicodedata
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] not in ("P", "Z", "C"):
            return False
    return True


class HybridRetriever:
    """Combines vector (dense) and lexical (FTS5) search with configurable fusion.

    Parameters
    ----------
    embedder : BaseEmbedder
        The embedding model used for vector search.
    db : SQLiteStore
        Database handle for FTS5 and chunk-metadata lookups.
    qdrant_index : QdrantIndex | None
        Qdrant index for fast vector search.
    alpha : float
        Weight for vector similarity score (default 0.7).
    beta : float
        Weight for lexical score (default 0.3).
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        db: SQLiteStore,
        qdrant_index: QdrantIndex | None = None,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> None:
        self.embedder = embedder
        self.db = db
        self.qdrant_index = qdrant_index
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
            return self._format_vector_only(vector_results, top_k, min_score)

        # 3. Hybrid fusion via RRF
        merged = self._rrf_fusion(vector_results, lexical_results, top_k, alpha=self.alpha, beta=self.beta)

        # 4. Augment with metadata and filter by score (batch fetch, no N+1)
        chunk_ids = [cid for cid, score, _, _ in merged if score >= min_score]
        chunk_map = self.db.get_chunks_with_documents_batch(chunk_ids)
        results: list[dict[str, Any]] = []
        for chunk_id, score, v_score, l_score in merged:
            if score < min_score:
                continue
            chunk_info = chunk_map.get(chunk_id)
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

    # -- Vector search (Qdrant) ----------------------------------------------

    def _vector_search(
        self, query_vec: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """Vector (dense) retrieval via Qdrant."""
        if self.qdrant_index is None or self.qdrant_index.size == 0:
            return []
        return self.qdrant_index.search(query_vec, top_k)

    # -- Lexical search (FTS5) ---------------------------------------------

    def _lexical_search(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        """Lexical (keyword) search via SQLite FTS5.

        Returns [(chunk_id, score), ...] where score is derived from
        the FTS5 rank (BM25-style).
        """
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
            score = 1.0 / (1.0 + abs(rank))
            results.append((chunk_id, score))

        return results

    @staticmethod
    def _to_fts_query(query: str) -> str:
        """Convert a user query to an FTS5 query string.

        Whitespace-split for Latin text; jieba cut for CJK text.
        Each term is quoted to prevent FTS5 reserved keywords from being
        interpreted as operators.
        """
        import re

        terms: list[str] = []
        _cjk_pattern = re.compile(r"[一-鿿㐀-䶿]")

        for part in query.split():
            part = part.strip().strip('"\'(),.!?:;')
            if not part:
                continue
            if _cjk_pattern.search(part):
                try:
                    import jieba
                except ImportError:
                    terms.append(part)
                    continue
                for word in jieba.cut(part):
                    word = word.strip()
                    if word and not _is_punct(word):
                        terms.append(word)
            else:
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
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> list[tuple[int, float, float, float]]:
        """Reciprocal Rank Fusion with weighted raw-score carry-over.

        RRF ranking determines result order, but the returned ``hybrid_score``
        is the weighted combination of raw similarity scores so that the value
        stays in a meaningful range (not compressed by ``1/(k+rank)``).

        Returns
        -------
        list[(chunk_id, hybrid_score, vector_score, lexical_score)]
        """
        v_ranks: dict[int, int] = {
            cid: idx for idx, (cid, _) in enumerate(vector_results)
        }
        l_ranks: dict[int, int] = {
            cid: idx for idx, (cid, _) in enumerate(lexical_results)
        }

        v_scores: dict[int, float] = dict(vector_results)
        l_scores: dict[int, float] = dict(lexical_results)

        all_ids = set(v_ranks) | set(l_ranks)

        scored: list[tuple[int, float, float, float]] = []
        for cid in all_ids:
            vr = v_ranks.get(cid, top_k * 2)
            lr = l_ranks.get(cid, top_k * 2)
            vs = v_scores.get(cid, 0.0)
            ls_ = l_scores.get(cid, 0.0)
            # RRF score for ranking only
            rrf_score = (1.0 / (k + vr)) + (1.0 / (k + lr))
            # Weighted raw scores for the displayed score
            hybrid_score = alpha * vs + beta * ls_
            scored.append((cid, rrf_score, vs, ls_, hybrid_score))

        # Sort by RRF score for ranking
        scored.sort(key=lambda x: x[1], reverse=True)
        # Return (chunk_id, hybrid_score, vector_score, lexical_score)
        return [(cid, hs, vs, ls) for cid, _, vs, ls, hs in scored[:top_k]]

    # -- Format helpers for single-strategy results -------------------------

    def _format_vector_only(
        self,
        vector_results: list[tuple[int, float]],
        top_k: int,
        min_score: float,
    ) -> list[dict[str, Any]]:
        # Filter and collect IDs first, then batch-fetch metadata
        qualified = [
            (cid, score) for cid, score in vector_results
            if score >= min_score
        ][:top_k]
        if not qualified:
            return []
        chunk_ids = [cid for cid, _ in qualified]
        chunk_map = self.db.get_chunks_with_documents_batch(chunk_ids)
        results: list[dict[str, Any]] = []
        for chunk_id, score in qualified:
            chunk_info = chunk_map.get(chunk_id)
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
        qualified = [
            (cid, score) for cid, score in lexical_results
            if score >= min_score
        ][:top_k]
        if not qualified:
            return []
        chunk_ids = [cid for cid, _ in qualified]
        chunk_map = self.db.get_chunks_with_documents_batch(chunk_ids)
        results: list[dict[str, Any]] = []
        for chunk_id, score in qualified:
            chunk_info = chunk_map.get(chunk_id)
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
