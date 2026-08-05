"""Dedicated tests for the AnswerAgent."""

from agentflow.agents.answer.agent import AnswerAgent


class _FakeLLM:
    def __init__(self, answer="这是回答"):
        self.answer = answer
        self.calls = []

    def complete(self, messages, node_name="", max_tokens=None, **kwargs):
        self.calls.append(messages)
        return self.answer


def _state(goal_type="question", source_mode="auto", degraded=False) -> dict:
    state = {
        "question": "你好",
        "goal_analysis": {
            "goal": "你好",
            "goal_type": goal_type,
            "knowledge_source": "general",
            "source_mode": source_mode,
        },
        "task_queue": [],
        "_degraded": {"_answer"} if degraded else set(),
    }
    return state


def test_answer_mode_uses_llm(monkeypatch):
    fake = _FakeLLM("回答内容")
    monkeypatch.setattr("agentflow.agents.answer.agent.get_llm_service", lambda: fake)
    result = AnswerAgent().run(_state())
    assert result["answer"] == "回答内容"
    assert len(fake.calls) == 1
    assert fake.calls[0][0]["role"] == "system"


def test_answer_continue_mode_marks_continue(monkeypatch):
    fake = _FakeLLM("继续回答")
    monkeypatch.setattr("agentflow.agents.answer.agent.get_llm_service", lambda: fake)
    state = _state()
    state["_continue_mode"] = True
    state["conversation_context"] = {"type": "FOLLOW_UP", "summary": "上轮聊代码"}
    AnswerAgent().run(state)
    system_prompt = fake.calls[0][0]["content"]
    assert "继续" in system_prompt or "conversation" in system_prompt.lower()


def test_answer_degraded_skips_llm(monkeypatch):
    fake = _FakeLLM("不应被调用")
    monkeypatch.setattr("agentflow.agents.answer.agent.get_llm_service", lambda: fake)
    result = AnswerAgent().run(_state(degraded=True))
    assert fake.calls == []
    assert "受限" in result["answer"] or "不可用" in result["answer"]


def test_summary_mode_skips_llm(monkeypatch):
    fake = _FakeLLM("不应被调用")
    monkeypatch.setattr("agentflow.agents.answer.agent.get_llm_service", lambda: fake)
    state = _state(goal_type="project")
    state["task_queue"] = [
        {"task_id": "a", "status": "done", "tool": "filesystem", "goal": "创建 a",
         "input": {"action": "write_file", "path": "a.py"}},
    ]
    result = AnswerAgent().run(state)
    assert fake.calls == []
    assert "完成" in result["answer"]


def test_knowledge_mode_appends_sources(monkeypatch):
    fake = _FakeLLM("基于知识库的回答")
    monkeypatch.setattr("agentflow.agents.answer.agent.get_llm_service", lambda: fake)
    state = _state(source_mode="knowledge")
    state["goal_analysis"]["knowledge_source"] = "local"
    state["knowledge_results"] = [
        {"filename": "guide.md", "score": 0.9, "method": "hybrid"},
    ]
    AnswerAgent().run(state)
    user_prompt = fake.calls[0][1]["content"]
    assert "guide.md" in user_prompt


def test_generation_failure_reports_reason(monkeypatch):
    fake = _FakeLLM("不应被调用")
    monkeypatch.setattr("agentflow.agents.answer.agent.get_llm_service", lambda: fake)
    state = _state(goal_type="project")
    state["_generation_failed"] = True
    state["_generation_failure_reason"] = "无法自动生成代码内容"
    result = AnswerAgent().run(state)
    assert "无法自动生成代码内容" in result["answer"]
    assert fake.calls == []


def test_clean_answer_strips_artifacts():
    agent = AnswerAgent()
    assert agent.clean_answer("```python\nprint(1)\n```") is not None
    assert isinstance(agent.clean_answer("纯文本回答"), str)
