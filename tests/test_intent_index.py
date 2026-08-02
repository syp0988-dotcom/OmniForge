"""Tests for the multi-anchor intent matcher."""

from __future__ import annotations

import numpy as np
import pytest

from agentflow.agents.goal_analyzer.intent_index import (
    IntentIndex,
    _score_labels,
)
from agentflow.config.settings import settings


class MapEmbedder:
    """Deterministic embedder backed by an explicit vector map."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = {
            text: np.array(vec, dtype=np.float32)
            for text, vec in mapping.items()
        }

    def embed(self, texts: list[str], batch_size: int = 20) -> list[np.ndarray]:
        return [self._mapping[text] for text in texts]

    @property
    def name(self) -> str:
        return "map"

    @property
    def model_name(self) -> str:
        return "map-embedder"


def test_score_labels_uses_max_pooling_per_label():
    anchors = {
        "coding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        "chat": [np.array([0.3, 0.9])],
    }
    query = np.array([0.0, 1.0])
    scores = _score_labels(query, anchors)
    # coding wins because its second anchor is a perfect match, even though
    # its first anchor is orthogonal.
    assert scores["coding"] == pytest.approx(1.0, abs=1e-6)
    assert scores["chat"] < scores["coding"]


def test_match_returns_best_label_when_ratio_satisfied(monkeypatch):
    monkeypatch.setattr(settings, "intent_confidence_ratio", 1.2)
    monkeypatch.setattr(settings, "intent_min_score_floor", 0.2)
    embedder = MapEmbedder({
        "write code": [1.0, 0.0, 0.0],
        "fix bug": [0.9, 0.1, 0.0],
        "hello": [0.0, 1.0, 0.0],
        "please write code for me": [0.95, 0.05, 0.0],
    })
    index = IntentIndex(
        anchors={
            "coding": ["write code", "fix bug"],
            "chat": ["hello"],
        },
        embedder=embedder,
    )
    index._ensure_ready()
    label, goal_type, score = index.match("please write code for me")
    assert label == "coding"
    assert goal_type == "coding"
    assert score > 0.9


def test_match_falls_back_on_low_ratio(monkeypatch):
    monkeypatch.setattr(settings, "intent_confidence_ratio", 2.0)
    monkeypatch.setattr(settings, "intent_min_score_floor", 0.2)
    embedder = MapEmbedder({
        "write code": [1.0, 0.0],
        "hello": [0.0, 1.0],
        "ambig": [0.6, 0.4],
    })
    index = IntentIndex(
        anchors={"coding": ["write code"], "chat": ["hello"]},
        embedder=embedder,
    )
    index._ensure_ready()
    assert index.match("ambig") is None


def test_match_falls_back_below_score_floor(monkeypatch):
    monkeypatch.setattr(settings, "intent_confidence_ratio", 1.2)
    monkeypatch.setattr(settings, "intent_min_score_floor", 0.5)
    embedder = MapEmbedder({
        "write code": [1.0, 0.0],
        "garbage": [0.0, 0.1],  # orthogonal → cosine ≈ 0, below the floor
    })
    index = IntentIndex(
        anchors={"coding": ["write code"]},
        embedder=embedder,
    )
    index._ensure_ready()
    assert index.match("garbage") is None
