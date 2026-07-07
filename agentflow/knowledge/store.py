"""High-level KnowledgeStore: parser → chunk → embed → ChromaDB → hybrid search.

The ``KnowledgeStore`` ties together all knowledge-base components:
  - Document parsing and chunking (``parser`` / ``chunking``)
  - Embedding (``embedder.BaseEmbedder`` implementations)
  - ANN indexing (``index.ChromaIndex``)
  - Hybrid retrieval (``retrieval.HybridRetriever``)

Unlike the old FAISS-based architecture, ChromaDB handles vector persistence
and ANN search internally — no manual load/save/rebuild needed.

Usage::

    store = KnowledgeStore()
    doc_id = store.add_document("/path/to/doc.pdf", "doc.pdf")
    results = store.search("user query", top_k=5)
    store.delete_document(doc_id)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.embedder import (
    SemanticEmbedder,
    TfidfEmbedder,
)
from agentflow.knowledge.index import ChromaIndex
from agentflow.knowledge.parser import parse_document
from agentflow.knowledge.retrieval import HybridRetriever
from agentflow.knowledge.settings import (
    chunk_params,
    embedder_type,
    embedding_model,
    search_defaults,
)

logger = logging.getLogger("knowledge.store")


class KnowledgeStore:
    """Manages document ingestion, embedding, indexing, and hybrid search.

    Parameters
    ----------
    db : SQLiteStore | None
        Database backend.  Creates a default instance if ``None``.
    embedder : str | None
        Embedder type: ``"semantic"`` (default) or ``"tfidf"``.
        Falls back to ``settings.knowledge_embedder`` if ``None``.
    """

    def __init__(
        self,
        db: SQLiteStore | None = None,
        embedder: str | None = None,
        chroma_index: ChromaIndex | None = None,
    ) -> None:
        self.db = db or SQLiteStore()
        embedder_type_name = embedder or embedder_type()
        self.embedder = self._create_embedder(embedder_type_name)
        self.chroma_index = chroma_index  # None → lazy-init on first use
        self.retriever: HybridRetriever | None = None

        # Load existing TF-IDF state if using tfidf embedder
        self._maybe_load_cache()

    # -- Document management ---------------------------------------------------

    def add_document(self, file_path: str | Path, filename: str) -> int:
        """Parse a file, chunk, embed, and persist everything.

        Returns:
            The document ID.
        """
        path = Path(file_path)
        file_type = _detect_file_type(path, filename)
        file_size = path.stat().st_size
        chunk_size, chunk_overlap = chunk_params()

        logger.info("Ingesting document: %s (%d bytes)", filename, file_size)

        # 1. Parse and chunk
        chunks = parse_document(path, file_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            logger.warning("  → No chunks extracted from %s", filename)
        logger.info("  → %d chunks extracted", len(chunks))

        # 2. Store document metadata
        doc_id = self.db.add_document(
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            doc_metadata=json.dumps(
                {"chunk_count": len(chunks), "original_path": str(path)}
            ),
        )

        if not chunks:
            return doc_id

        # 3. Ensure embedder is ready (TF-IDF needs fitting)
        self._ensure_embedder_ready(chunks)

        # 4. Batch embed all chunks
        try:
            vectors = self.embedder.embed(chunks)
        except Exception as exc:
            logger.error("Embedding failed for %s: %s", filename, exc)
            self.db.delete_document_cascade(doc_id)
            raise

        # 5. Persist chunks to SQLite
        chunk_ids: list[int] = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = self.db.add_chunk(document_id=doc_id, content=chunk_text, chunk_index=i)
            chunk_ids.append(chunk_id)

        # 6. Store vectors in ChromaDB
        if chunk_ids:
            self._ensure_index()
            metadatas = [
                {"document_id": doc_id, "chunk_index": i}
                for i in range(len(chunk_ids))
            ]
            vectors_array = np.array([v for v in vectors], dtype=np.float32)
            self.chroma_index.add(chunk_ids, vectors_array, metadatas)

        logger.info("  → Document #%d indexed successfully (%d chunks)", doc_id, len(chunks))
        return doc_id

    def delete_document(self, doc_id: int) -> None:
        """Remove a document and all its chunks/vectors."""
        chunks = self.db.get_chunks_by_document(doc_id)
        chunk_ids = [c["id"] for c in chunks]

        # Remove from ChromaDB (ensure index is loaded first)
        if chunk_ids:
            self._ensure_index()
            if self.chroma_index is not None:
                self.chroma_index.remove(chunk_ids)

        # Cascade delete from SQLite
        self.db.delete_document_cascade(doc_id)
        logger.info("Document #%d deleted (removed %d chunks)", doc_id, len(chunk_ids))

    def list_documents(self) -> list[dict[str, Any]]:
        """List all indexed documents with metadata."""
        return self.db.get_all_documents()

    # -- Search ----------------------------------------------------------------

    def search(
        self, query: str, top_k: int | None = None, min_score: float | None = None
    ) -> list[dict[str, Any]]:
        """Hybrid search: vector similarity (ChromaDB) + lexical (FTS5).

        Args:
            query: Natural language query string.
            top_k: Maximum number of results (default from settings).
            min_score: Minimum score threshold (default from settings).

        Returns:
            List of dicts with keys:
                chunk_id, document_id, filename, content, score,
                vector_score, lexical_score, method.
        """
        if top_k is None:
            top_k, _ = search_defaults()
        if min_score is None:
            _, min_score = search_defaults()

        # Early exit if no documents indexed
        all_docs = self.db.get_all_documents()
        if not all_docs:
            logger.info("No documents indexed; returning empty search results.")
            return []

        # Ensure retriever is initialised
        self._ensure_retriever()

        return self.retriever.search(query, top_k=top_k, min_score=min_score)

    # -- Embedder factory ------------------------------------------------------

    @staticmethod
    def _create_embedder(embedder_type_name: str):
        if embedder_type_name == "semantic":
            try:
                embedder = SemanticEmbedder(model_name=embedding_model())
                # Probe: embed a dummy string to verify model loads
                embedder.embed(["probe"])
                logger.info("Using SemanticEmbedder (model=%s)", embedding_model())
                return embedder
            except ImportError:
                logger.warning(
                    "SemanticEmbedder requested but sentence-transformers not installed. "
                    "Falling back to TfidfEmbedder. "
                    "Install: pip install sentence-transformers"
                )
                return TfidfEmbedder()
        logger.info("Using TfidfEmbedder (fallback mode)")
        return TfidfEmbedder()

    # -- Lazy init helpers -----------------------------------------------------

    def _ensure_embedder_ready(self, new_chunks: list[str]) -> None:
        """Ensure the embedder is fitted (TF-IDF) or loaded (semantic).

        For TfidfEmbedder: collect all existing chunk texts plus the new
        chunks and fit the vectorizer.
        """
        if not isinstance(self.embedder, TfidfEmbedder):
            return  # semantic embedder is always ready
        if getattr(self.embedder, "_fitted", False):
            return

        # Collect all existing chunk texts from the DB
        all_docs = self.db.get_all_documents()
        existing_texts: list[str] = []
        for doc in all_docs:
            chunks = self.db.get_chunks_by_document(doc["id"])
            existing_texts.extend(c["content"] for c in chunks)

        all_texts = existing_texts + new_chunks
        if not all_texts:
            return

        logger.info("Fitting TfidfEmbedder on %d texts…", len(all_texts))
        self.embedder.fit(all_texts)
        self._save_tfidf_cache()

    def _save_tfidf_cache(self) -> None:
        """Persist TF-IDF vocabulary cache to DB."""
        if not isinstance(self.embedder, TfidfEmbedder):
            return
        try:
            cache = json.dumps(self.embedder.to_dict())
            self.db.set_knowledge_meta("tfidf_cache", cache)
        except Exception as exc:
            logger.debug("Failed to save TF-IDF cache (non-critical): %s", exc)

    def _ensure_index(self) -> None:
        if self.chroma_index is None:
            self.chroma_index = ChromaIndex()

    def _ensure_retriever(self) -> None:
        if self.retriever is not None:
            return
        self._ensure_index()
        from agentflow.knowledge.settings import hybrid_weights
        alpha, beta = hybrid_weights()
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            db=self.db,
            chroma_index=self.chroma_index,
            alpha=alpha,
            beta=beta,
        )

    def _maybe_load_cache(self) -> None:
        """For TF-IDF embedder: restore cached vocab if available.

        This is a best-effort cache that avoids re-fitting when the
        embedder type is ``"tfidf"`` and previous state exists.
        """
        if not isinstance(self.embedder, TfidfEmbedder):
            return
        raw = self.db.get_knowledge_meta("tfidf_cache")
        if raw:
            try:
                self.embedder.from_dict(json.loads(raw))
                logger.info(
                    "Loaded TF-IDF cache: %d terms, %d docs",
                    len(self.embedder.vocab),
                    self.embedder.num_docs,
                )
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to load TF-IDF cache: %s", exc)


# -- File-type detection helper -----------------------------------------------


def _detect_file_type(path: Path, filename: str) -> str:
    """Detect file type from path or filename."""
    ext = path.suffix.lstrip(".").lower()
    if ext:
        return ext
    if "." in filename:
        return filename.rsplit(".", 1)[-1].lower()
    return "txt"
