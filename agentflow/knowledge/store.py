"""High-level KnowledgeStore: parser → chunk → embed → Qdrant → hybrid search.

The ``KnowledgeStore`` ties together all knowledge-base components:
  - Document parsing and chunking (``parser`` / ``chunking``)
  - Embedding (``embedder.QwenEmbedder``)
  - ANN indexing (``index.QdrantIndex``)
  - Hybrid retrieval (``retrieval.HybridRetriever``)

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
from agentflow.knowledge.embedder import QwenEmbedder
from agentflow.knowledge.index import QdrantIndex
from agentflow.knowledge.parser import parse_document
from agentflow.knowledge.retrieval import HybridRetriever
from agentflow.knowledge.settings import chunk_params, search_defaults

logger = logging.getLogger("knowledge.store")


class KnowledgeStore:
    """Manages document ingestion, embedding, indexing, and hybrid search.

    Parameters
    ----------
    db : SQLiteStore | None
        Database backend.  Creates a default instance if ``None``.
    qdrant_index : QdrantIndex | None
        Qdrant vector index.  Creates a default local instance if ``None``.
    """

    def __init__(
        self,
        db: SQLiteStore | None = None,
        qdrant_index: QdrantIndex | None = None,
    ) -> None:
        self.db = db or SQLiteStore()
        self.embedder = QwenEmbedder()
        self.qdrant_index = qdrant_index  # None → lazy-init on first use
        self.retriever: HybridRetriever | None = None

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
        chunks = parse_document(
            path, file_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
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

        # 3. Persist chunks to SQLite
        chunk_ids: list[int] = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = self.db.add_chunk(
                document_id=doc_id, content=chunk_text, chunk_index=i,
            )
            chunk_ids.append(chunk_id)

        try:
            # 4. Embed and add to Qdrant
            vectors = self.embedder.embed(chunks, batch_size=20)
            vectors_array = np.array([v for v in vectors], dtype=np.float32)
            metadatas = [
                {"document_id": doc_id, "chunk_index": i}
                for i in range(len(chunk_ids))
            ]
            self._ensure_index()
            self.qdrant_index.add(chunk_ids, vectors_array, metadatas)
            self.retriever = None
        except Exception:
            self.db.delete_document_cascade(doc_id)
            raise

        logger.info(
            "  → Document #%d indexed successfully (%d chunks)", doc_id, len(chunks),
        )
        return doc_id

    def delete_document(self, doc_id: int) -> None:
        """Remove a document and all its chunks/vectors."""
        chunks = self.db.get_chunks_by_document(doc_id)
        chunk_ids = [c["id"] for c in chunks]

        if chunk_ids:
            self._ensure_index()
            if self.qdrant_index is not None:
                self.qdrant_index.remove(chunk_ids)

        self.db.delete_document_cascade(doc_id)
        logger.info(
            "Document #%d deleted (removed %d chunks)", doc_id, len(chunk_ids),
        )

    def list_documents(self) -> list[dict[str, Any]]:
        """List all indexed documents with metadata."""
        return self.db.get_all_documents()

    # -- Search ----------------------------------------------------------------

    def search(
        self, query: str, top_k: int | None = None, min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search: vector similarity (Qdrant) + lexical (FTS5).

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

        all_docs = self.db.get_all_documents()
        if not all_docs:
            logger.info("No documents indexed; returning empty search results.")
            return []

        self._ensure_retriever()
        return self.retriever.search(query, top_k=top_k, min_score=min_score)

    # -- Lazy init helpers -----------------------------------------------------

    def _ensure_index(self) -> None:
        if self.qdrant_index is None:
            self.qdrant_index = QdrantIndex()

    def _ensure_retriever(self) -> None:
        if self.retriever is not None:
            return
        self._ensure_index()
        from agentflow.knowledge.settings import hybrid_weights
        alpha, beta = hybrid_weights()
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            db=self.db,
            qdrant_index=self.qdrant_index,
            alpha=alpha,
            beta=beta,
        )


# -- File-type detection helper -----------------------------------------------


def _detect_file_type(path: Path, filename: str) -> str:
    """Detect file type from path or filename."""
    ext = path.suffix.lstrip(".").lower()
    if ext:
        return ext
    if "." in filename:
        return filename.rsplit(".", 1)[-1].lower()
    return "txt"
