"""Embedding interface and implementations for the knowledge base.

Architecture
------------
``BaseEmbedder`` (ABC) defines the stateless embedding contract.
Two concrete implementations:

* **TfidfEmbedder**  — lightweight TF-IDF + cosine similarity (fallback).
* **SemanticEmbedder** — sentence-transformers based semantic embeddings (primary).

Usage::

    embedder: BaseEmbedder
    if settings.knowledge_embedder == "semantic":
        embedder = SemanticEmbedder()
    else:
        embedder = TfidfEmbedder().fit(all_texts)

    vectors = embedder.embed(["hello world", "你好世界"])
    query_vec = embedder.embed_query("some question")
"""

from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Tokenizer (shared)
# ---------------------------------------------------------------------------

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-]+|[^\s]")


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text.

    Chinese characters → unigrams.
    English tokens → lowercased, split on whitespace/punctuation.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        tok = match.group()
        if _CHINESE_RE.match(tok):
            tokens.extend(list(tok))
        else:
            tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    """Stateless embedding contract.

    Implementations **must not** depend on in-memory corpus state — the
    same ``embed()`` call with the same input must always return the same
    vector (up to model weights).
    """

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 128) -> list[np.ndarray]:
        """Embed a batch of texts into float32 vectors.

        Returns a list of 1-D ``np.ndarray``, one per input text, each of
        shape ``(dimension,)``.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text into a float32 vector.

        Some models use a different instruction/prefix for queries vs.
        documents — this method allows implementations to differentiate.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a short identifier e.g. ``"tfidf"``, ``"semantic"``."""
        ...


# ---------------------------------------------------------------------------
# TF-IDF embedder (stateless fallback)
# ---------------------------------------------------------------------------


class TfidfEmbedder(BaseEmbedder):
    """Lightweight TF-IDF vectorizer.

    Unlike the legacy implementation, this version is **stateless after fit**:
    ``fit(texts)`` builds a vocabulary from the provided corpus; subsequent
    ``embed()`` calls ignore OOV tokens.  State can be serialised via
    ``to_dict()`` / ``from_dict()`` for cache warm-up.

    Parameters
    ----------
    min_df : int
        Minimum document frequency for a term to be kept in the vocabulary.
    """

    def __init__(self, min_df: int = 1) -> None:
        # term → column index
        self.vocab: dict[str, int] = {}
        # term_index → document frequency
        self.doc_freq: dict[int, int] = {}
        self.num_docs: int = 0
        self._fitted: bool = False
        self._dim: int = 0
        self._min_df = min_df

    # -- Fitting -----------------------------------------------------------

    def fit(self, texts: list[str]) -> TfidfEmbedder:
        """Build vocabulary and document frequencies from a corpus.

        Calling ``fit`` replaces any previously learned vocabulary.
        """
        self.vocab.clear()
        self.doc_freq.clear()
        self.num_docs = len(texts)
        self._fitted = True

        # Collect all terms
        term_doc_sets: list[set[str]] = []
        all_terms: set[str] = set()
        for text in texts:
            toks = tokenize(text)
            unique = set(toks)
            term_doc_sets.append(unique)
            all_terms.update(unique)

        # Filter by min_df
        doc_freq_raw: dict[str, int] = {}
        for uniq in term_doc_sets:
            for term in uniq:
                doc_freq_raw[term] = doc_freq_raw.get(term, 0) + 1

        # Build vocab (sorted for determinism)
        kept = sorted(
            t for t, df in doc_freq_raw.items() if df >= self._min_df
        )
        self.vocab = {t: i for i, t in enumerate(kept)}
        self.doc_freq = {
            self.vocab[t]: df for t, df in doc_freq_raw.items() if t in self.vocab
        }
        self._dim = len(self.vocab)
        return self

    def update(self, new_texts: list[str]) -> TfidfEmbedder:
        """Incrementally add new documents without rebuilding the entire vocab.

        Updates document frequencies and total document count, then adds any
        new terms that were not in the original vocabulary.  Much faster than
        ``fit()`` for incremental ingestion (O(|new_texts|) instead of
        O(|corpus|)).

        If the embedder has not been fitted yet, delegates to ``fit()``.
        """
        if not self._fitted:
            return self.fit(new_texts)

        if not new_texts:
            return self

        new_term_counts: dict[str, int] = {}
        for text in new_texts:
            for term in set(tokenize(text)):
                new_term_counts[term] = new_term_counts.get(term, 0) + 1

        # Update document frequencies for existing terms
        for term, df_add in new_term_counts.items():
            idx = self.vocab.get(term)
            if idx is not None:
                self.doc_freq[idx] = self.doc_freq.get(idx, 0) + df_add

        # Add genuinely new terms to vocab (rare, but handle them)
        new_terms = sorted(t for t, df in new_term_counts.items()
                          if t not in self.vocab and df >= self._min_df)
        if new_terms:
            offset = len(self.vocab)
            for i, term in enumerate(new_terms):
                self.vocab[term] = offset + i
                self.doc_freq[offset + i] = new_term_counts[term]
            self._dim = len(self.vocab)

        self.num_docs += len(new_texts)
        return self

    # -- BaseEmbedder interface -------------------------------------------

    def embed(self, texts: list[str], batch_size: int = 128) -> list[np.ndarray]:
        """Transform texts to TF-IDF vectors using the fitted vocabulary.

        OOV tokens are silently ignored.
        The *batch_size* parameter is ignored (TF-IDF runs per-text).
        """
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder has not been fitted yet — call .fit()")
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        """Alias for ``embed([text])[0]``."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        if not self._fitted:
            return 0
        return self._dim

    @property
    def name(self) -> str:
        return "tfidf"

    # -- Internal ----------------------------------------------------------

    def _vectorize(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        if not tokens or not self.vocab:
            return np.zeros(self._dim, dtype=np.float32)

        vec = np.zeros(self._dim, dtype=np.float32)
        tf = Counter(tokens)
        max_tf = max(tf.values())

        for tok, count in tf.items():
            idx = self.vocab.get(tok)
            if idx is None:
                continue
            tf_val = count / max_tf
            df = self.doc_freq.get(idx, 1)
            idf_val = math.log((self.num_docs + 1) / (df + 1)) + 1.0
            vec[idx] = tf_val * idf_val

        # L2 normalize
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    # -- Serialization (optional cache) ------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted state for persistence."""
        return {
            "vocab": self.vocab.copy(),
            "doc_freq": {str(k): v for k, v in self.doc_freq.items()},
            "num_docs": self.num_docs,
            "min_df": self._min_df,
            "_dim": self._dim,
            "_fitted": self._fitted,
        }

    def from_dict(self, data: dict[str, Any]) -> TfidfEmbedder:
        """Restore fitted state from a dictionary."""
        self.vocab = data.get("vocab", {})
        self.doc_freq = {int(k): v for k, v in data.get("doc_freq", {}).items()}
        self.num_docs = data.get("num_docs", 0)
        self._min_df = data.get("min_df", 1)
        self._dim = data.get("_dim", 0)
        self._fitted = data.get("_fitted", False)
        return self


# ---------------------------------------------------------------------------
# Semantic embedder (sentence-transformers)
# ---------------------------------------------------------------------------


class SemanticEmbedder(BaseEmbedder):
    """Sentence-transformers based semantic embedder.

    The model is loaded lazily on first ``embed()`` call, so creating an
    instance is cheap until the first actual embedding operation.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or path.  Defaults to a multilingual model
        that handles both Chinese and English well.
    query_prefix : str
        Prefix prepended to queries (not documents) for asymmetric embedding
        models.  Improves retrieval quality by telling the model this is a
        search query rather than a passage.  Set to ``""`` to disable.
    """

    _DEFAULT_QUERY_PREFIX = "search_query: "

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        query_prefix: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._model = None  # lazy-loaded
        self._dim: int | None = None
        self._query_prefix = query_prefix if query_prefix is not None else self._DEFAULT_QUERY_PREFIX

    # -- BaseEmbedder interface -------------------------------------------

    def embed(self, texts: list[str], batch_size: int = 128) -> list[np.ndarray]:
        """Embed texts in configurable batches to limit memory usage."""
        model = self._load_model()
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            for j in range(embeddings.shape[0]):
                all_embeddings.append(embeddings[j])
        return all_embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with the search prefix for better retrieval."""
        if self._query_prefix:
            text = self._query_prefix + text
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        if self._dim is not None:
            return self._dim
        # Probe: embed a dummy string to learn dimension
        dummy = self.embed(["dummy"])
        self._dim = dummy[0].shape[0]
        return self._dim

    @property
    def name(self) -> str:
        return "semantic"

    # -- Internal ----------------------------------------------------------

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return self._model
        if os.getenv("AGENTFLOW_ENABLE_SEMANTIC_EMBEDDER", "").lower() not in ("1", "true", "yes"):
            raise ImportError(
                "SemanticEmbedder is disabled by default. "
                "Set AGENTFLOW_ENABLE_SEMANTIC_EMBEDDER=1 to enable it."
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SemanticEmbedder requires sentence-transformers.\n"
                "Install: pip install sentence-transformers"
            ) from exc

        # Fix conda SSL_CERT_FILE pointing to non-existent path
        import os as _os
        for _env_key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            _cert_path = _os.environ.get(_env_key, "")
            if _cert_path and not _os.path.exists(_cert_path):
                _os.environ.pop(_env_key, None)

        kwargs = {"device": "cpu"}
        if os.getenv("AGENTFLOW_ALLOW_MODEL_DOWNLOAD", "").lower() not in ("1", "true", "yes"):
            kwargs["local_files_only"] = True
        try:
            self._model = SentenceTransformer(self._model_name, **kwargs)
        except Exception as exc:
            raise ImportError(
                "SemanticEmbedder model is not available locally. "
                "Set AGENTFLOW_ALLOW_MODEL_DOWNLOAD=1 to allow downloading it."
            ) from exc
        return self._model


# ---------------------------------------------------------------------------
# Cosine similarity helpers (preserved for backward compatibility)
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def batch_cosine_similarity(
    query_vec: np.ndarray, candidates: list[tuple[int, np.ndarray]]
) -> list[tuple[int, float]]:
    """Compute cosine similarity between query and many candidates.

    Args:
        query_vec: Query vector (1-D).
        candidates: List of (chunk_id, vector) pairs.

    Returns:
        List of (chunk_id, score) sorted descending by score.
    """
    results: list[tuple[int, float]] = []
    for cid, vec in candidates:
        score = cosine_similarity(query_vec, vec)
        results.append((cid, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Vector serialization (format-agnostic)
# ---------------------------------------------------------------------------


def serialize_vector(vec: np.ndarray) -> bytes:
    """Serialize a numpy vector to bytes for SQLite BLOB storage."""
    return vec.tobytes()


def deserialize_vector(data: bytes) -> np.ndarray:
    """Deserialize a numpy vector from SQLite BLOB."""
    return np.frombuffer(data, dtype=np.float32)
