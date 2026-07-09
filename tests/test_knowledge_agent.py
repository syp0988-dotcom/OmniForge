from __future__ import annotations

from agentflow.agents.knowledge.agent import KnowledgeAgent


class FakeKnowledgeStore:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int | None, float | None]] = []

    def search(self, query: str, top_k: int | None = None, min_score: float | None = None):
        self.queries.append((query, top_k, min_score))
        return [
            {
                "filename": "OmniForge.docx",
                "content": "OmniForge has a strict knowledge-base answer mode.",
                "score": 0.91,
                "method": "hybrid",
            }
        ]


def test_knowledge_agent_uses_shared_store_and_rewritten_question(monkeypatch):
    fake_store = FakeKnowledgeStore()
    monkeypatch.setattr(
        "agentflow.agents.knowledge.agent._shared_knowledge_store",
        lambda: fake_store,
    )

    state = KnowledgeAgent().run({
        "question": "它有什么亮点？",
        "rewritten_question": "OmniForge 有什么亮点？",
    })

    assert fake_store.queries == [("OmniForge 有什么亮点？", 5, 0.05)]
    assert state["knowledge_results"][0]["filename"] == "OmniForge.docx"
    assert "OmniForge.docx" in state["knowledge_context"]


def test_knowledge_agent_falls_back_to_original_question(monkeypatch):
    fake_store = FakeKnowledgeStore()
    monkeypatch.setattr(
        "agentflow.agents.knowledge.agent._shared_knowledge_store",
        lambda: fake_store,
    )

    KnowledgeAgent().run({"question": "OmniForge 亮点"})

    assert fake_store.queries == [("OmniForge 亮点", 5, 0.05)]
