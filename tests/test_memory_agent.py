"""Dedicated tests for the MemoryAgent."""

from agentflow.agents.memory.agent import MemoryAgent
from agentflow.config.settings import settings
from agentflow.conversation.session_state import SessionState


class _FakeLTM:
    def __init__(self) -> None:
        self.extractions = []

    def extract_and_store(self, question, answer, entities, goal):
        self.extractions.append((question, answer, entities, goal))
        return 0


def _agent(ltm=None, max_turns=10) -> tuple[MemoryAgent, _FakeLTM]:
    fake = ltm or _FakeLTM()
    agent = MemoryAgent.__new__(MemoryAgent)
    agent.max_turns = max_turns
    agent._long_term = fake
    return agent, fake


def test_run_appends_history():
    agent, _ = _agent()
    state = {"question": "你好", "answer": "你好！", "memory": {"history": []}}
    result = agent.run(state)
    history = result["memory"]["history"]
    assert history[-2] == {"role": "user", "content": "你好"}
    assert history[-1] == {"role": "assistant", "content": "你好！"}
    assert "context_str" in result["memory"]


def test_run_recovers_existing_history():
    agent, _ = _agent()
    existing = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    result = agent.run({"question": "q2", "answer": "a2", "memory": {"history": existing}})
    assert len(result["memory"]["history"]) == 4


def test_run_truncates_to_max_turns():
    agent, _ = _agent(max_turns=2)
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(10)
    ]
    result = agent.run({"question": "q", "answer": "a", "memory": {"history": history}})
    assert len(result["memory"]["history"]) <= 5


def test_run_compresses_long_history(monkeypatch):
    # Force deterministic compression (no LLM).
    monkeypatch.setattr(
        "agentflow.conversation.compression._get_llm_for_compression",
        lambda: None,
    )
    monkeypatch.setattr(settings, "enable_history_compression", True)
    monkeypatch.setattr(settings, "history_token_budget", 100)
    monkeypatch.setattr(settings, "history_min_keep_messages", 4)
    agent, _ = _agent()
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "内容" * 20}
        for i in range(12)
    ]
    result = agent.run({"question": "q", "answer": "a", "memory": {"history": history}})
    memory = result["memory"]
    assert len(memory["history"]) < len(history)
    assert memory.get("rolled_summary")


def test_run_extracts_long_term_memory():
    agent, fake = _agent()
    ss = SessionState()
    agent.run({
        "question": "我喜欢 Python",
        "answer": "好的，记住了。",
        "memory": {"history": []},
        "conversation_context": {"entities": ["Python"]},
        "session_state": ss,
    })
    assert len(fake.extractions) == 1
    q, a, entities, goal = fake.extractions[0]
    assert q == "我喜欢 Python"
    assert "Python" in entities


def test_run_no_answer_skips_extraction():
    agent, fake = _agent()
    agent.run({
        "question": "q",
        "answer": "",
        "memory": {"history": []},
        "conversation_context": {"entities": ["x"]},
    })
    assert fake.extractions == []
def test_rolling_summary_survives_to_the_next_turn(monkeypatch):
    """The rolling summary must not be lost when memory is rebuilt (P1-MEM-1)."""
    from agentflow.agents.memory.agent import MemoryAgent

    agent = MemoryAgent()
    previous = {
        "history": [{"role": "user", "content": "早前的话题"}],
        "rolled_summary": "早期对话摘要：用户在做图书管理系统",
    }
    state = {
        "question": "继续",
        "answer": "好的",
        "memory": previous,
        "session_state": None,
    }

    result = agent.run(state)

    assert "图书管理系统" in result["memory"]["rolled_summary"]
    assert "图书管理系统" in result["memory"]["summary"]


def test_history_is_compressed_before_truncation(monkeypatch):
    """Older turns must reach the compressor before the window cap trims them."""
    from agentflow.agents.memory import agent as memory_module

    seen: dict[str, int] = {}

    def fake_compress(history, budget_tokens=None, min_keep_messages=None, llm=None):
        seen["received"] = len(history)
        return history[-2:], "摘要：更早的对话"

    monkeypatch.setattr(memory_module, "compress_history", fake_compress)
    agent = memory_module.MemoryAgent(max_turns=2)
    history = [
        {"role": "user", "content": f"问题 {i}"} if i % 2 == 0
        else {"role": "assistant", "content": f"回答 {i}"}
        for i in range(10)
    ]
    state = {"question": "新问题", "answer": "新回答", "memory": {"history": history}}

    result = agent.run(state)

    assert seen["received"] == 12, "compressor saw the untrimmed history"
    assert "更早的对话" in result["memory"]["rolled_summary"]
