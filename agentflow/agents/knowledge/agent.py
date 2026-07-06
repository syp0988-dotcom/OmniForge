"""Knowledge Agent — retrieves relevant document chunks from the local knowledge base."""

from __future__ import annotations

from agentflow.agents.base import AgentProtocol
from agentflow.knowledge.store import KnowledgeStore
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("knowledge")


class KnowledgeAgent(AgentProtocol):
    """Retrieves relevant document chunks from the local vector store.

    Uses TF-IDF vectorization and cosine similarity to find the most relevant
    content for the user's question.
    """

    def __init__(self) -> None:
        self.store = KnowledgeStore()

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        logger.info("KnowledgeAgent retrieving for: %s", question)

        results = self.store.search(question, top_k=5, min_score=0.05)

        if results:
            context_parts = []
            for r in results:
                method = r.get("method", "vector")
                score = r.get("score", 0.0)
                context_parts.append(
                    f"[来源: {r['filename']} | 相似度: {score:.2f} | "
                    f"检索方式: {method}]\n{r['content']}"
                )
            knowledge_text = "\n\n---\n\n".join(context_parts)
            logger.info("  → Found %d relevant chunks", len(results))
        else:
            knowledge_text = "（知识库中未找到相关结果）"
            logger.info("  → No relevant knowledge found")

        state["knowledge_results"] = results
        state["knowledge_context"] = knowledge_text
        return state
