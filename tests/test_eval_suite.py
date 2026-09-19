"""Tests for the offline evaluation framework (metrics + dataset base)."""

import json

import pytest

from agentflow.eval.common import BaseEvalDataset
from agentflow.eval.completion_eval import metrics as completion_metrics
from agentflow.eval.intent_eval import metrics as intent_metrics
from agentflow.eval.planner_eval import metrics as planner_metrics
from agentflow.eval.tool_eval import metrics as tool_metrics
from agentflow.knowledge.eval import metrics as rag_metrics


# -- run_all.py entry points (must stay runnable) -----------------------------


def test_knowledge_store_exposes_index_size():
    """``run_all.py`` used a non-existent ``store.index.ntotal`` attribute."""
    from agentflow.knowledge.index import QdrantIndex
    from agentflow.knowledge.store import KnowledgeStore

    store = KnowledgeStore(qdrant_index=QdrantIndex.in_memory("eval_size_test"))
    assert isinstance(store.index_size, int)
    assert store.index_size == 0
    store.qdrant_index.close()


def test_rag_eval_skips_cleanly_when_index_is_empty(monkeypatch):
    """An empty index must skip the suite, not crash the whole run."""
    from agentflow.eval import run_all

    class _EmptyStore:
        qdrant_index = None

        @property
        def index_size(self) -> int:
            return 0

    # ``run_rag_eval`` imports KnowledgeStore lazily, so patch it at the source.
    monkeypatch.setattr(
        "agentflow.knowledge.store.KnowledgeStore", lambda *a, **kw: _EmptyStore(),
    )

    assert run_all.run_rag_eval() is None


class _DummyDataset(BaseEvalDataset):
    @staticmethod
    def _validate_sample(sample, line_num):
        return isinstance(sample, dict) and "x" in sample

    @staticmethod
    def stats(samples):
        return {"count": len(samples)}


# -- BaseEvalDataset I/O ------------------------------------------------------


def test_dataset_load_save_roundtrip(tmp_path):
    p = tmp_path / "samples.jsonl"
    p.write_text(
        "\n".join([
            json.dumps({"x": 1}),
            "# comment line should be skipped",
            json.dumps({"x": 2}),
        ]),
        encoding="utf-8",
    )
    ds = _DummyDataset.load(p)
    assert len(ds.samples) == 2

    out = tmp_path / "out.jsonl"
    ds.save(out)
    assert _DummyDataset.load(out).samples == ds.samples


def test_dataset_load_rejects_invalid_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        _DummyDataset.load(p)


def test_dataset_add_sample_assigns_id():
    ds = _DummyDataset()
    sample_id = ds.add_sample({"x": 1})
    assert sample_id.startswith("ev_")
    assert ds.samples[0]["id"] == sample_id


# -- RAG retrieval metrics ----------------------------------------------------


def test_rag_metrics():
    relevant = {1, 3}
    retrieved = [3, 2, 1, 4, 5]
    assert rag_metrics.recall_at_k(relevant, retrieved, 3) == 1.0
    assert rag_metrics.precision_at_k(relevant, retrieved, 3) == 2 / 3
    assert rag_metrics.mrr(relevant, retrieved) == 1.0  # rank 1
    assert rag_metrics.hit_rate(relevant, retrieved, 1) is True
    assert rag_metrics.ndcg_at_k(relevant, retrieved, 5) > 0
    assert rag_metrics.recall_at_k(set(), retrieved, 3) == 1.0

    all_m = rag_metrics.compute_all(relevant, retrieved, k_values=[1, 3])
    assert all_m["recall@3"] == 1.0
    assert all_m["hit@1"] == 1.0
    agg = rag_metrics.aggregate([all_m, all_m])
    assert agg["mrr"] == 1.0
    assert rag_metrics.aggregate([]) == {}


# -- Intent metrics -----------------------------------------------------------


def test_intent_metrics():
    actual = ["project", "question", "project"]
    expected = ["project", "question", "project"]
    flags = [True, False, True]
    conf = [0.9, 0.5, 0.8]
    m = intent_metrics.compute_all(actual, expected, flags, conf)
    assert m["goal_type_accuracy"] == 1.0
    assert m["embedding_hit_rate"] == 2 / 3
    assert m["embedding_accuracy"] == 1.0
    assert m["llm_accuracy"] == 1.0
    assert m["confidence_mean"] == pytest.approx((0.9 + 0.5 + 0.8) / 3)
    assert m["per_label_accuracy"]["question"] == 1.0
    assert m["confusion"]["project"]["project"] == 2

    assert intent_metrics.goal_type_accuracy([], []) == 0.0
    assert intent_metrics.embedding_hit_rate([]) == 0.0
    assert intent_metrics.embedding_accuracy([], [], []) == 1.0
    assert intent_metrics.llm_accuracy([], [], []) == 1.0


# -- Planner metrics ----------------------------------------------------------


def test_planner_metrics():
    sample = {
        "expected_tools": ["filesystem", "python"],
        "expected_actions": ["write_file", "execute"],
        "expected_task_count_range": [2, 4],
        "expected_goal_type": "project",
        "expected_goal_completed": False,
    }
    m = planner_metrics.compute_all(
        actual_tools=["filesystem", "python"],
        actual_actions=["write_file", "execute", "read_file"],
        actual_task_count=3,
        actual_goal_type="project",
        actual_goal_completed=False,
        sample=sample,
    )
    assert m["tool_acc"] == 1.0
    assert m["action_recall"] == 1.0
    assert m["task_count_acc"] == 1.0
    assert m["goal_type_acc"] == 1.0
    assert m["goal_completed_acc"] == 1.0

    assert planner_metrics.jaccard_similarity(set(), set()) == 1.0
    assert planner_metrics.task_count_accuracy(5, (2, 4)) == 0.0
    assert planner_metrics.aggregate([m, m])["tool_acc"] == 1.0


# -- Tool metrics -------------------------------------------------------------


def test_tool_metrics():
    results = [
        {"success": True, "tool": "filesystem", "action": "write_file", "message": "ok"},
        {"success": False, "tool": "filesystem", "action": "delete_file", "error": "EACCES: denied"},
    ]
    samples = [
        {"expected_result_checks": {"message_contains": "ok"}},
        {"expected_result_checks": {"result.path": "/x"}},
    ]
    m = tool_metrics.compute_all(samples, results)
    assert m["success_rate"] == 0.5
    assert m["action_rates"]["filesystem.write_file"] == 1.0
    assert m["tool_rates"]["filesystem"] == 0.5
    assert m["error_distribution"].get("EACCES") == 1
    assert tool_metrics.success_rate([]) == 0.0


# -- Completion metrics -------------------------------------------------------


def test_completion_metrics():
    results = [
        {"goal_completed": True, "tasks_done": 5, "turns_taken": 2},
        {"goal_completed": False, "tasks_done": 1, "turns_taken": 1},
    ]
    samples = [
        {"min_expected_tasks_done": 4, "expected_completed": True},
        {"min_expected_tasks_done": 2, "expected_completed": False},
    ]
    m = completion_metrics.compute_all(results, samples)
    assert m["completion_rate"] == 0.5
    assert m["avg_tasks_done"] == 3.0
    assert m["avg_turns_taken"] == 1.5
    assert m["min_tasks_done_rate"] == 0.5
    assert m["completion_match_rate"] == 1.0
    assert completion_metrics.completion_rate([]) == 0.0
    assert completion_metrics.aggregate([m, m])["completion_rate"] == 0.5
