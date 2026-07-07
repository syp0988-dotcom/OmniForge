"""Knowledge Agent — retrieves relevant document references from the local knowledge base.

Knowledge does NOT generate answers. It only provides reference material:
- Company standards and norms
- Coding conventions
- Templates
- API documentation
- Database designs
- Project examples

The references are used by the Planner and Executor for context.
"""

from __future__ import annotations

from agentflow.agents.base import AgentProtocol
from agentflow.knowledge.store import KnowledgeStore
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("knowledge")


class KnowledgeAgent(AgentProtocol):
    """Retrieves relevant document references from the local vector store.

    Provides reference material only — does not generate answers.
    The retrieved references inform the Planner's task tree generation.
    """

    def __init__(self) -> None:
        self.store = KnowledgeStore()

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        """Retrieve knowledge references for the current question."""
        question = str(state.get("question", ""))
        logger.info("KnowledgeAgent retrieving references for: %s", question[:80])

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
            logger.info("  → Found %d relevant references", len(results))
        else:
            knowledge_text = "（知识库中未找到相关参考）"
            logger.info("  → No relevant references found")

        state["knowledge_results"] = results
        state["knowledge_context"] = knowledge_text
        return state
