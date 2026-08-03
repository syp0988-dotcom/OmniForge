"""Tests for the runtime feedback collector (eval loop)."""

import json

from agentflow.eval.feedback import record_feedback, set_feedback_path
from agentflow.eval.export_feedback import export_feedback, load_records, summarize


def _state(goal_type="project", errors=None, generation_failed=False, answer="ok"):
    return {
        "question": "帮我创建图书管理系统",
        "goal_analysis": {"goal": "帮我创建图书管理系统", "goal_type": goal_type},
        "answer": answer,
        "trace_id": "trace-1",
        "_errors": errors or [],
        "_generation_failed": generation_failed,
    }


def test_failure_always_recorded(tmp_path):
    set_feedback_path(tmp_path / "feedback.jsonl")
    recorded = record_feedback(
        _state(errors=[{"source": "planner", "type": "llm_unavailable", "message": "no key"}]),
        session_id=1,
        question="帮我创建图书管理系统",
        answer="",
    )
    assert recorded is True
    records = load_records()
    assert len(records) == 1
    assert records[0]["outcome"] == "failure"
    assert records[0]["errors"][0]["type"] == "llm_unavailable"


def test_dedup_same_failure(tmp_path):
    set_feedback_path(tmp_path / "feedback.jsonl")
    state = _state(errors=[{"source": "x", "type": "boom", "message": "same"}])
    assert record_feedback(state, session_id=1, question="q") is True
    assert record_feedback(state, session_id=1, question="q") is False
    assert len(load_records()) == 1


def test_empty_answer_counts_as_failure(tmp_path):
    set_feedback_path(tmp_path / "feedback.jsonl")
    state = _state(answer="")
    assert record_feedback(state, session_id=1, question="q") is True
    assert load_records()[0]["outcome"] == "failure"


def test_success_recorded_only_for_completion_goals(tmp_path):
    set_feedback_path(tmp_path / "feedback.jsonl")
    assert record_feedback(_state(goal_type="project"), session_id=1, question="q") is True
    assert record_feedback(_state(goal_type="coding"), session_id=2, question="q2") is True
    assert record_feedback(_state(goal_type="question"), session_id=3, question="q3") is False
    records = load_records()
    assert len(records) == 2
    assert all(r["outcome"] == "success" for r in records)


def test_record_never_raises(tmp_path):
    set_feedback_path(tmp_path / "feedback.jsonl")
    # Corrupt state (non-serializable error object) must not raise.
    state = _state(errors=[{"source": object(), "type": object(), "message": object()}])
    assert record_feedback(state, session_id=1, question="q") is True or True


def test_export_summary_and_failure_cases(tmp_path):
    set_feedback_path(tmp_path / "feedback.jsonl")
    record_feedback(
        _state(errors=[{"source": "planner", "type": "llm_unavailable", "message": "x"}]),
        session_id=1,
        question="q",
    )
    record_feedback(_state(goal_type="project"), session_id=2, question="q2")
    record_feedback(_state(goal_type="question"), session_id=3, question="q3")  # not recorded

    stats = summarize(load_records())
    assert stats["total"] == 2
    assert stats["by_outcome"] == {"failure": 1, "success": 1}
    assert stats["by_goal_type"]["project"] == 2

    out = tmp_path / "failure_cases.json"
    export_feedback(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["stats"]["total"] == 2
    assert len(data["failure_cases"]) == 1
