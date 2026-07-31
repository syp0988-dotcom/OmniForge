"""Tests for the persistent embedding cache."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from agentflow.knowledge.embedding_cache import EmbeddingCache
from agentflow.knowledge.embedder import QwenEmbedder


class FakeEmbedClient:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def embeddings(self):
        return self

    def create(self, model: str, input: list[str]):
        self.calls += 1
        data = [
            SimpleNamespace(embedding=[0.1, 0.2, 0.3] * len(t))
            for t in input
        ]
        return SimpleNamespace(data=data)


def test_cache_avoids_duplicate_api_calls(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb_cache.db")
    embedder = QwenEmbedder(
        api_key="test-key", model_name="test-model", cache=cache,
    )
    client = FakeEmbedClient()
    embedder._client = client

    first = embedder.embed(["hello world", "another text"])
    assert client.calls == 1

    # Same texts again → served entirely from cache.
    second = embedder.embed(["hello world", "another text"])
    assert client.calls == 1
    for a, b in zip(first, second):
        assert np.array_equal(a, b)

    # A new text triggers exactly one more API call.
    embedder.embed(["brand new text"])
    assert client.calls == 2
    assert cache.size() == 3


def test_cache_survives_reopen(tmp_path):
    path = tmp_path / "emb_cache.db"
    cache = EmbeddingCache(path)
    cache.store("m", ["persisted"], [np.array([1.0, 2.0], dtype=np.float32)])
    cache.close()

    reopened = EmbeddingCache(path)
    cached, missing_idx, missing_texts = reopened.lookup("m", ["persisted"])
    assert missing_texts == []
    assert cached[0] is not None and np.allclose(cached[0], [1.0, 2.0])
    reopened.close()
