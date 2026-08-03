"""Tests for layered-memory history compression."""

from agentflow.conversation.compression import (
    compress_history,
    count_tokens,
    _deterministic_summary,
)


def _messages(n: int, content_len: int = 80) -> list[dict]:
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"消息{i}: " + "字" * content_len})
    return msgs


def test_within_budget_returns_unchanged():
    history = _messages(4)
    kept, summary = compress_history(history, budget_tokens=10**9, llm=None)
    assert kept == history
    assert summary == ""


def test_compression_keeps_recent_window():
    history = _messages(20)
    kept, summary = compress_history(
        history,
        budget_tokens=100,  # tiny budget forces compression
        min_keep_messages=4,
        llm=None,
    )
    assert len(kept) >= 2
    assert len(kept) < len(history)
    # Most recent messages must be preserved verbatim.
    assert kept[-1] == history[-1]
    assert kept[-2] == history[-2]
    assert summary != ""


def test_min_keep_respected():
    history = _messages(10)
    kept, summary = compress_history(
        history,
        budget_tokens=1,
        min_keep_messages=6,
        llm=None,
    )
    assert len(kept) >= 2  # never below the absolute floor
    assert len(kept) <= 6
    assert summary != ""


def test_deterministic_summary_bounded():
    history = _messages(30, content_len=50)
    summary = _deterministic_summary(history)
    assert "早期对话摘要" in summary
    assert len(summary) < 2000


def test_llm_summary_used_when_available():
    class _FakeLLM:
        def complete(self, messages, node_name="", max_tokens=100):
            assert node_name == "history_compress"
            return "用户目标是创建项目，已提到 Python 与部署。"

    history = _messages(12)
    kept, summary = compress_history(
        history,
        budget_tokens=50,
        min_keep_messages=4,
        llm=_FakeLLM(),
    )
    assert "Python" in summary
    assert len(kept) < len(history)


def test_count_tokens_empty():
    assert count_tokens([]) == 0


def test_empty_history():
    kept, summary = compress_history([], llm=None)
    assert kept == []
    assert summary == ""
