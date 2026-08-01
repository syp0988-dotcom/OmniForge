"""Tests for intent-analysis improvements (labels, source defaults, degraded
signalling)."""

from __future__ import annotations

from agentflow.agents.goal_analyzer.agent import GoalAnalyzer
from agentflow.agents.goal_analyzer.intent_index import INTENT_LABEL_TO_GOAL_TYPE
from agentflow.config.settings import settings


def test_embedding_labels_cover_routing_goal_types():
    # These goal types are used by the router but were previously unreachable
    # through the embedding fast path (only via LLM fallback).
    for label, goal_type in (
        ("analysis", "analysis"),
        ("document", "document"),
        ("translation", "translation"),
        ("editing", "editing"),
    ):
        assert INTENT_LABEL_TO_GOAL_TYPE[label] == goal_type


def test_question_intent_defaults_to_hybrid_source(monkeypatch):
    monkeypatch.setattr(settings, "intent_question_source", "hybrid")
    goal = GoalAnalyzer._build_goal_from_match(
        question="什么是微服务", label="question", goal_type="question",
        confidence=0.9,
    )
    assert goal["knowledge_source"] == "hybrid"

    monkeypatch.setattr(settings, "intent_question_source", "local")
    goal = GoalAnalyzer._build_goal_from_match(
        question="什么是微服务", label="question", goal_type="question",
        confidence=0.9,
    )
    assert goal["knowledge_source"] == "local"


def test_other_labels_keep_rule_based_sources():
    goal = GoalAnalyzer._build_goal_from_match(
        question="分析两个方案", label="analysis", goal_type="analysis",
        confidence=0.9,
    )
    assert goal["knowledge_source"] == "hybrid"
    goal = GoalAnalyzer._build_goal_from_match(
        question="翻译一下", label="translation", goal_type="translation",
        confidence=0.9,
    )
    assert goal["knowledge_source"] == "general"


class _UnavailableIndex:
    available = False

    def match(self, query: str):
        return None


def test_embedding_unavailable_is_recorded(monkeypatch):
    from agentflow.agents.goal_analyzer import agent as goal_module

    monkeypatch.setattr(goal_module, "_get_intent_index", lambda: _UnavailableIndex())
    state: dict = {"question": "你好"}
    GoalAnalyzer().run(state)

    errors = state.get("_errors", [])
    assert any(e["type"] == "embedding_unavailable" for e in errors)
