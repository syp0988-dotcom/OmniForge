"""Embedding interface and implementation for the knowledge base.

Architecture
------------
``BaseEmbedder`` (ABC) defines the stateless embedding contract.
``QwenEmbedder`` is the concrete implementation using DashScope's
OpenAI-compatible embedding API.

Usage::

    embedder = QwenEmbedder()
    vectors = embedder.embed(["hello world", "你好世界"])
    query_vec = embedder.embed_query("some question")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from agentflow.config.settings import settings

logger = logging.getLogger("knowledge.embedder")


class BaseEmbedder(ABC):
    """Abstract embedding interface."""

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 20) -> list[np.ndarray]:
        """Embed a list of texts into vectors."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text into a vector."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this embedder."""


class QwenEmbedder(BaseEmbedder):
    """DashScope / OpenAI-compatible embedding API.

    Parameters
    ----------
    api_key : str | None
        API key.  Defaults to ``EMBEDDING_API_KEY`` env var.
    base_url : str | None
        API base URL.  Defaults to DashScope compatible-mode endpoint.
    model_name : str | None
        Model name.  Defaults to ``text-embedding-v3`` (1024-d).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.embedding_api_key
        self._base_url = base_url or settings.embedding_base_url
        self._model_name = model_name or settings.embedding_model_name
        self._client = None
        self._dim: int | None = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            import os as _os
            for _env_key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
                _cert_path = _os.environ.get(_env_key, "")
                if _cert_path and not _os.path.exists(_cert_path):
                    _os.environ.pop(_env_key, None)
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=60.0,
            )
        return self._client

    def embed(self, texts: list[str], batch_size: int = 20) -> list[np.ndarray]:
        """Embed texts in batches via the API."""
        client = self._get_client()
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(
                model=self._model_name,
                input=batch,
            )
            for item in resp.data:
                all_embeddings.append(
                    np.array(item.embedding, dtype=np.float32)
                )
        return all_embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        if self._dim is None:
            dummy = self.embed(["probe"])
            self._dim = dummy[0].shape[0]
        return self._dim

    @property
    def name(self) -> str:
        return "qwen"
