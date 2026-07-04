"""LangGraph workflow definition for the multi-agent system."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agentflow.agents.answer.agent import AnswerAgent
from agentflow.agents.knowledge.agent import KnowledgeAgent
from agentflow.agents.memory.agent import MemoryAgent
from agentflow.agents.planner.agent import PlannerAgent
from agentflow.agents.python.agent import PythonAgent
from agentflow.agents.router.agent import QueryRouterAgent
from agentflow.agents.search.agent import SearchAgent
from agentflow.utils.logging import build_logger

logger = build_logger("workflow")


class WorkflowState(TypedDict, total=False):
    """Typed state container for workflow nodes."""

    question: str
    workflow: list[str]
    category: str
    plan: dict[str, Any]
    search_results: list[dict[str, Any]]
    knowledge_results: list[dict[str, Any]]
    knowledge_context: str
    python_result: dict[str, Any]
    answer: str
    memory: dict[str, Any]
    history: list[dict[str, str]]
    router: dict[str, Any]


def build_workflow() -> Any:
    """Build the LangGraph workflow for the system.

    Flow:
      router -> (conditional) knowledge / planner
      knowledge -> planner
      planner -> (conditional) search / python / answer
      search -> answer
      python -> answer
      answer -> memory
      memory -> END
    """
    router = QueryRouterAgent()
    planner = PlannerAgent()
    search = SearchAgent()
    answer = AnswerAgent()
    memory = MemoryAgent()
    knowledge = KnowledgeAgent()
    python_executor = PythonAgent()

    workflow = StateGraph(WorkflowState)

    workflow.add_node("router", router.run)
    workflow.add_node("planner", planner.run)
    workflow.add_node("search", search.run)
    workflow.add_node("answer", answer.run)
    workflow.add_node("memory", memory.run)
    workflow.add_node("knowledge", knowledge.run)
    workflow.add_node("python", python_executor.run)

    workflow.set_entry_point("router")

    # After router: identity/search skip knowledge, everything else hits KB
    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "knowledge": "knowledge",
            "planner": "planner",
        },
    )

    workflow.add_edge("knowledge", "planner")

    # After planner: route based on category
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "search": "search",
            "python": "python",
            "answer": "answer",
        },
    )

    # Execution nodes converge to answer
    workflow.add_edge("search", "answer")
    workflow.add_edge("python", "answer")

    # Finalize
    workflow.add_edge("answer", "memory")
    workflow.add_edge("memory", END)

    return workflow.compile()


def _route_after_router(state: WorkflowState) -> str:
    """Determine the next node after routing based on the category.

    Identity and search queries skip knowledge retrieval.
    Everything else (reasoning, coding, writing, python, knowledge) checks
    the knowledge base for relevant context.
    """
    category = state.get("category", "reasoning")
    if category in ("identity", "search"):
        return "planner"
    return "knowledge"


def _route_after_planner(state: WorkflowState) -> str:
    """Route based on category to the correct execution node."""
    category = state.get("category", "reasoning")
    if category == "search":
        return "search"
    if category == "python":
        return "python"
    return "answer"


def run_workflow(
    graph: Any,
    message: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run the workflow for a user message, optionally seeding with history."""
    initial_state: WorkflowState = {
        "question": message,
        "workflow": [],
        "history": history or [],
    }
    result = graph.invoke(initial_state)
    return dict(result)
