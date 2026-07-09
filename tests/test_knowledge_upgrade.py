"""Tests for the upgraded knowledge base module.

Covers:
  - chunking strategies (markdown, code, paragraph)
  - embedder interface (TfidfEmbedder)
  - parser integration
  - KnowledgeStore CRUD + search
  - backward compatibility (legacy chunk_text, API response format)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from agentflow.knowledge.chunking import (
    chunk_by_code,
    chunk_by_markdown,
    chunk_by_paragraph,
    chunk_document,
)
from agentflow.knowledge.embedder import (
    TfidfEmbedder,
    SemanticEmbedder,
    cosine_similarity,
    batch_cosine_similarity,
    serialize_vector,
    deserialize_vector,
)
from agentflow.knowledge.index import ChromaIndex
from agentflow.knowledge.parser import parse_document, chunk_text
from agentflow.knowledge.store import KnowledgeStore


# =========================================================================
# Chunking tests
# =========================================================================


class TestChunking:
    def test_paragraph_chunking_basic(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_by_paragraph(text, chunk_size=100, overlap=0)
        # All fit in one chunk
        assert len(chunks) >= 1

    def test_paragraph_chunking_overflow(self):
        """Paragraphs that overflow chunk_size should be split."""
        text = "\n\n".join([f"Paragraph number {i} with some padding text." for i in range(20)])
        chunks = chunk_by_paragraph(text, chunk_size=100, overlap=10)
        assert len(chunks) >= 2

    def test_paragraph_chunking_empty(self):
        assert chunk_by_paragraph("", chunk_size=500, overlap=50) == []
        assert chunk_by_paragraph("   \n\n  ", chunk_size=500, overlap=50) == []

    def test_markdown_chunking_by_headings(self):
        text = (
            "# Title\n\nIntro.\n\n"
            "## Section A\n\nContent A.\n\n"
            "### Sub A.1\n\nSub content.\n\n"
            "## Section B\n\nContent B."
        )
        # Use a small chunk_size so sections don't all merge
        chunks = chunk_by_markdown(text, chunk_size=50, overlap=0)
        assert len(chunks) >= 2

    def test_markdown_chunking_heading_context(self):
        """Each chunk should preserve the heading line."""
        text = "## Section A\n\nBody A.\n\n## Section B\n\nBody B."
        chunks = chunk_by_markdown(text, chunk_size=500, overlap=0)
        for c in chunks:
            assert c.startswith("## ")

    def test_code_chunking(self):
        text = (
            "def foo():\n    return 1\n\n"
            "def bar():\n    return 2\n\n"
            "class MyClass:\n    def method(self):\n        return 3\n"
        )
        chunks = chunk_by_code(text, chunk_size=30, overlap=0)
        assert len(chunks) >= 3  # foo, bar, MyClass

    def test_code_chunking_no_boundaries(self):
        """Plain text with no code boundaries should fall back to paragraph."""
        text = "Just some text.\n\nNothing to see here."
        chunks = chunk_by_code(text, chunk_size=500, overlap=0)
        assert len(chunks) >= 1

    def test_chunk_document_auto_detect(self):
        md_text = "# H1\n\nBody.\n\n## H2\n\nMore."
        md_chunks = chunk_document(md_text, "md", chunk_size=500)
        txt_chunks = chunk_document(md_text, "txt", chunk_size=500)
        # Markdown should use heading-based; txt uses paragraph
        assert len(md_chunks) >= 1
        assert len(txt_chunks) >= 1

    def test_legacy_chunk_text_backward_compat(self):
        """The deprecated chunk_text should still work."""
        text = "A\n\nB\n\nC"
        chunks = chunk_text(text, chunk_size=10, overlap=0)
        assert len(chunks) >= 1


# =========================================================================
# Embedder tests
# =========================================================================


class TestTfidfEmbedder:
    def test_fit_and_embed(self):
        embedder = TfidfEmbedder()
        embedder.fit(["hello world", "goodbye world"])
        vecs = embedder.embed(["hello"])
        assert len(vecs) == 1
        assert vecs[0].shape[0] == embedder.dimension
        assert embedder.name == "tfidf"

    def test_embed_query(self):
        embedder = TfidfEmbedder()
        embedder.fit(["hello world"])
        qvec = embedder.embed_query("hello")
        assert qvec.shape[0] == embedder.dimension

    def test_cosine_similarity(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

        c = np.array([1.0, 1.0], dtype=np.float32)
        d = np.array([1.0, 1.0], dtype=np.float32)
        assert cosine_similarity(c, d) == pytest.approx(1.0, abs=1e-6)

    def test_batch_cosine_similarity(self):
        embedder = TfidfEmbedder()
        embedder.fit(["hello world", "foo bar"])
        qvec = embedder.embed_query("hello")
        candidates = [(1, embedder.embed(["world"])[0]), (2, embedder.embed(["foo"])[0])]
        results = batch_cosine_similarity(qvec, candidates)
        assert len(results) == 2
        # "world" should be more similar to "hello" than "foo"
        assert results[0][0] == 1

    def test_serialization_roundtrip(self):
        embedder = TfidfEmbedder()
        embedder.fit(["hello world", "test document"])
        state = embedder.to_dict()

        restored = TfidfEmbedder().from_dict(state)
        assert restored._fitted
        assert restored.vocab == embedder.vocab
        assert restored.num_docs == embedder.num_docs
        # Embeddings should be consistent
        original = embedder.embed(["hello"])[0]
        restored_v = restored.embed(["hello"])[0]
        assert np.allclose(original, restored_v)

    def test_vector_serialization(self):
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        blob = serialize_vector(vec)
        restored = deserialize_vector(blob)
        assert np.allclose(vec, restored)

    def test_min_df_filtering(self):
        embedder = TfidfEmbedder(min_df=2)
        embedder.fit(["a b c", "a b d", "e f g"])
        # "c", "d", "e", "f", "g" appear only once → filtered out by min_df=2
        assert "c" not in embedder.vocab
        assert "a" in embedder.vocab
        assert "b" in embedder.vocab

    def test_not_fitted_error(self):
        embedder = TfidfEmbedder()
        with pytest.raises(RuntimeError, match="not been fitted"):
            embedder.embed(["test"])

    def test_semantic_embedder_import_error(self):
        """SemanticEmbedder should raise ImportError without sentence-transformers."""
        with pytest.raises(ImportError):
            se = SemanticEmbedder()
            se.embed(["test"])


# =========================================================================
# KnowledgeStore integration tests
# =========================================================================


class TestKnowledgeStore:
    @pytest.fixture(autouse=True)
    def _setup(self):
        """Use a temporary DB and in-memory ChromaDB for each test."""
        import time
        self._tmp_db_path = tempfile.mktemp(suffix=".db")
        import uuid
        chroma_idx = ChromaIndex.in_memory(collection_name=f"test_{uuid.uuid4().hex}")
        from agentflow.database.sqlite import SQLiteStore
        self.db = SQLiteStore(Path(self._tmp_db_path))
        self.store = KnowledgeStore(db=self.db, chroma_index=chroma_idx)
        yield
        # Cleanup: close all connections by deleting the reference
        self.store = None
        self.db = None
        # Retry deletion a few times to handle Windows locks
        for _ in range(3):
            try:
                os.unlink(self._tmp_db_path)
                break
            except PermissionError:
                time.sleep(0.1)

    def _create_tmp_file(self, content: str, suffix: str = ".txt") -> str:
        path = tempfile.mktemp(suffix=suffix)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_add_and_search(self):
        path = self._create_tmp_file(
            "量子计算是一种新技术。\n\n它利用量子力学原理。\n\n"
            "量子比特是基本单位。\n\n在密码学中有重要应用。"
        )
        try:
            doc_id = self.store.add_document(path, "quantum.txt")
            assert doc_id > 0

            results = self.store.search("量子技术", top_k=2)
            assert len(results) >= 1
            for r in results:
                assert "chunk_id" in r
                assert "document_id" in r
                assert "filename" in r
                assert "content" in r
                assert "score" in r
                # New fields
                assert "method" in r
                assert r["method"] in ("vector", "lexical", "hybrid")
                assert "vector_score" in r
                assert "lexical_score" in r
        finally:
            os.unlink(path)

    def test_add_and_list(self):
        path = self._create_tmp_file("Test content.\n\nMore content.")
        try:
            doc_id = self.store.add_document(path, "test.txt")
            docs = self.store.list_documents()
            assert len(docs) >= 1
            assert any(d["id"] == doc_id for d in docs)
        finally:
            os.unlink(path)

    def test_tfidf_dimension_change_rebuilds_index(self):
        first = self._create_tmp_file(
            "alpha beta gamma shared knowledge baseline.\n\n"
            "delta epsilon zeta internal notes."
        )
        second = self._create_tmp_file(
            "kiwi mango papaya dragonfruit deployment guide.\n\n"
            "lychee rambutan guava release checklist."
        )
        try:
            first_id = self.store.add_document(first, "first.txt")
            first_dim = self.store.embedder.dimension

            second_id = self.store.add_document(second, "second.txt")
            second_dim = self.store.embedder.dimension

            assert first_id > 0
            assert second_id > 0
            assert second_dim > first_dim

            docs = self.store.list_documents()
            assert any(d["filename"] == "first.txt" for d in docs)
            assert any(d["filename"] == "second.txt" for d in docs)

            results = self.store.search("dragonfruit deployment", top_k=3)
            assert any(r["filename"] == "second.txt" for r in results)
        finally:
            os.unlink(first)
            os.unlink(second)

    def test_add_and_delete(self):
        path = self._create_tmp_file("Delete me.\n\nPlease delete.")
        try:
            doc_id = self.store.add_document(path, "delete.txt")
            self.store.delete_document(doc_id)
            docs = self.store.list_documents()
            assert not any(d["id"] == doc_id for d in docs)
        finally:
            os.unlink(path)

    def test_search_empty(self):
        results = self.store.search("nothing")
        assert results == []

    def test_search_result_format(self):
        """Verify the search result dict format is backward-compatible."""
        path = self._create_tmp_file("AI and machine learning.\n\nDeep learning is a subset.")
        try:
            doc_id = self.store.add_document(path, "ai.txt")
            results = self.store.search("machine learning", top_k=1)
            if results:
                r = results[0]
                # Backward-compatible keys
                assert "chunk_id" in r
                assert "document_id" in r
                assert "filename" in r
                assert "content" in r
                assert "score" in r
                # New keys
                assert "method" in r
                assert "vector_score" in r
                assert "lexical_score" in r
        finally:
            os.unlink(path)


# =========================================================================
# Parser integration tests
# =========================================================================


class TestParser:
    def test_parse_txt(self):
        path = tempfile.mktemp(suffix=".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Line 1\n\nLine 2\n\nLine 3")
        try:
            chunks = parse_document(path)
            assert len(chunks) >= 1
        finally:
            os.unlink(path)

    def test_parse_markdown(self):
        path = tempfile.mktemp(suffix=".md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Title\n\nBody.\n\n## Section\n\nContent.")
        try:
            chunks = parse_document(path)
            assert len(chunks) >= 1
        finally:
            os.unlink(path)

    def test_parse_with_chunk_params(self):
        path = tempfile.mktemp(suffix=".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join([f"Paragraph {i} with some extra text." for i in range(20)]))
        try:
            chunks = parse_document(path, chunk_size=50, chunk_overlap=10)
            assert len(chunks) >= 2
        finally:
            os.unlink(path)

#
# Preamble preservation tests (text before first heading/definition)
#


class TestPreamblePreservation:
    """Text before the first heading or definition must not be lost."""

    def test_markdown_preamble_preserved(self):
        text = "This is an intro paragraph.\n\n# Section 1\n\nBody text.\n\n## Subsection\n\nMore body."
        chunks = chunk_by_markdown(text, chunk_size=500, overlap=0)
        assert len(chunks) >= 1
        all_text = "\n\n".join(chunks)
        assert "This is an intro paragraph" in all_text

    def test_markdown_preamble_only(self):
        """Document with only intro text and no actual heading."""
        text = "Just some introduction text without any headings."
        chunks = chunk_by_markdown(text, chunk_size=500, overlap=0)
        assert len(chunks) >= 1
        assert "Just some introduction" in chunks[0]

    def test_code_preamble_preserved(self):
        text = (
            "# License header\n# Copyright 2024\n\n"
            "import os\nimport sys\n\n"
            "def foo():\n    return 1\n\n"
            "def bar():\n    return 2\n"
        )
        chunks = chunk_by_code(text, chunk_size=500, overlap=0)
        assert len(chunks) >= 1
        all_text = "\n\n".join(chunks)
        assert "License header" in all_text
        assert "import os" in all_text

    def test_code_preamble_only(self):
        """Plain text without any code definitions -- should fall back to paragraph."""
        text = "Just some text that has no function or class definitions."
        chunks = chunk_by_code(text, chunk_size=500, overlap=0)
        assert len(chunks) >= 1

    def test_chunk_by_code_unindented_preamble(self):
        """Preamble at the very start, no leading whitespace."""
        text = "#!/usr/bin/env python\n# encoding: utf-8\n\ndef main():\n    pass"
        chunks = chunk_by_code(text, chunk_size=500, overlap=0)
        all_text = "\n\n".join(chunks)
        assert "#!/usr/bin/env python" in all_text


# =========================================================================
# HybridRetriever constructor tests
# =========================================================================


class TestHybridRetrieverConfig:
    """HybridRetriever alpha/beta parameter passing."""

    def test_custom_alpha_beta(self):
        from unittest.mock import MagicMock
        from agentflow.knowledge.retrieval import HybridRetriever

        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        mock_embedder.name = "mock"
        mock_db = MagicMock()

        retriever = HybridRetriever(
            embedder=mock_embedder,
            db=mock_db,
            alpha=0.3,
            beta=0.7,
        )
        assert retriever.alpha == 0.3
        assert retriever.beta == 0.7

    def test_default_alpha_beta(self, monkeypatch):
        from unittest.mock import MagicMock
        from agentflow.knowledge.retrieval import HybridRetriever

        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        mock_embedder.name = "mock"
        mock_db = MagicMock()

        from agentflow.config.settings import settings
        monkeypatch.setattr(settings, "knowledge_alpha", 0.9)
        monkeypatch.setattr(settings, "knowledge_beta", 0.1)

        retriever = HybridRetriever(
            embedder=mock_embedder,
            db=mock_db,
        )
        assert retriever.alpha == 0.9
        assert retriever.beta == 0.1


# =========================================================================
# FTS5 query escaping tests
# =========================================================================


class TestFtsQuery:
    """FTS5 query construction from user query strings."""

    def _make_query(self, query: str) -> str:
        from agentflow.knowledge.retrieval import HybridRetriever
        return HybridRetriever._to_fts_query(query)

    def test_simple_query(self):
        result = self._make_query("hello world")
        assert result == '"hello" OR "world"'

    def test_reserved_keyword_or(self):
        result = self._make_query("to be or not")
        assert '"or"' in result
        assert '"not"' in result

    def test_reserved_keyword_and(self):
        result = self._make_query("this and that")
        assert '"and"' in result

    def test_chinese_query(self):
        result = self._make_query("quantum computing")
        assert '"quantum"' in result
        assert '"computing"' in result

    def test_empty_query(self):
        result = self._make_query("")
        assert result == ""

    def test_query_with_punctuation(self):
        result = self._make_query("hello, world! test.")
        assert '"hello"' in result
        assert '"world"' in result
        assert '"test"' in result
