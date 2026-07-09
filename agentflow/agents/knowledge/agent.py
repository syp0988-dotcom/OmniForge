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
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("knowledge")


class KnowledgeAgent(AgentProtocol):
    """Retrieves relevant document references from the local vector store.

    Provides reference material only — does not generate answers.
    The retrieved references inform the Planner's task tree generation.

    The KnowledgeStore (and its embedding model) is lazily initialised
    on first use to avoid loading the 470 MB sentence-transformers model
    on every request.
    """

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        """Retrieve knowledge references for the current question."""
        query = _knowledge_query(state)
        logger.info("KnowledgeAgent retrieving references for: %s", query[:80])

        results = _shared_knowledge_store().search(query, top_k=5, min_score=0.05)

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


def _knowledge_query(state: dict[str, object]) -> str:
    """Prefer the context-enriched rewritten question for retrieval."""
    rewritten = str(state.get("rewritten_question", "") or "").strip()
    if rewritten:
        return rewritten
    return str(state.get("question", "") or "")


def _shared_knowledge_store():
    """Use the same KnowledgeStore as the upload/search API.

    Imported lazily to avoid an import cycle during app startup.
    """
    from agentflow.api.routes import get_knowledge_store

    return get_knowledge_store()
