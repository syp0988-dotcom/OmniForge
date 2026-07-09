from __future__ import annotations

from types import SimpleNamespace

from agentflow.agents.answer.agent import AnswerAgent
from agentflow.services.llm_service import LLMService


def test_answer_agent_prepares_stream_messages_without_blocking_llm(monkeypatch):
    agent = AnswerAgent()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("blocking complete() should not be called in stream mode")

    llm = SimpleNamespace(complete=fail_if_called)
    monkeypatch.setattr("agentflow.agents.answer.agent.get_llm_service", lambda: llm)

    state = agent.run({
        "question": "什么是 SSE？",
        "goal_analysis": {
            "goal": "什么是 SSE？",
            "goal_type": "question",
            "knowledge_source": "general",
        },
        "_stream_answer": True,
    })

    assert state["answer"] == ""
    assert state["_answer_stream_mode"] is True
    assert state["_answer_stream_messages"]
    assert state["_answer_stream_messages"][0]["role"] == "system"
    assert state["_answer_stream_messages"][1]["role"] == "user"


def test_knowledge_source_mode_forces_local_prompt_and_sources(monkeypatch):
    agent = AnswerAgent()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("blocking complete() should not be called in stream mode")

    llm = SimpleNamespace(complete=fail_if_called)
    monkeypatch.setattr("agentflow.agents.answer.agent.get_llm_service", lambda: llm)

    state = agent.run({
        "question": "What does the KB say?",
        "source_mode": "knowledge",
        "goal_analysis": {
            "goal": "What does the KB say?",
            "goal_type": "question",
            "knowledge_source": "hybrid",
        },
        "knowledge_context": "[来源: OmniForge.docx | 相似度 0.91]\nOmniForge supports tool execution.",
        "knowledge_results": [
            {
                "filename": "OmniForge.docx",
                "score": 0.91,
                "method": "hybrid",
                "content": "OmniForge supports tool execution.",
            },
        ],
        "_stream_answer": True,
    })

    system_prompt = state["_answer_stream_messages"][0]["content"]
    user_prompt = state["_answer_stream_messages"][1]["content"]

    assert "Strict local knowledge mode" in system_prompt
    assert "OmniForge.docx" in user_prompt
    assert "知识库来源文件" in user_prompt
    assert "来源：" in user_prompt


def test_llm_service_complete_stream_yields_delta_content():
    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            chunks = ["你", "好"]
            for text in chunks:
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=text),
                        ),
                    ],
                )

    service = LLMService()
    service._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(),
        ),
    )
    service._model_name = "fake-model"
    service._temperature = 0
    service._max_tokens = 100

    pieces = list(service.complete_stream(messages=[{"role": "user", "content": "hi"}]))

    assert pieces == ["你", "好"]
