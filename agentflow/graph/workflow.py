from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agentflow.agents.answer.agent import AnswerAgent
from agentflow.agents.memory.agent import MemoryAgent
from agentflow.agents.planner.agent import PlannerAgent
from agentflow.agents.router.agent import QueryRouterAgent
from agentflow.agents.search.agent import SearchAgent


class WorkflowState(TypedDict, total=False):
    """Typed state container for workflow nodes."""

    question: str
    workflow: list[str]
    plan: dict[str, Any]
    search_results: list[dict[str, Any]]
    knowledge_results: list[dict[str, Any]]
    python_result: dict[str, Any]
    answer: str
    memory: dict[str, Any]


def build_workflow() -> Any:
    """Build the LangGraph workflow for the system."""
    router = QueryRouterAgent()
    planner = PlannerAgent()
    search = SearchAgent()
    answer = AnswerAgent()
    memory = MemoryAgent()

    workflow = StateGraph(WorkflowState)

    workflow.add_node("router", router.run)
    workflow.add_node("planner", planner.run)
    workflow.add_node("search", search.run)
    workflow.add_node("answer", answer.run)
    workflow.add_node("memory", memory.run)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "planner")
    workflow.add_edge("planner", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", "memory")
    workflow.add_edge("memory", END)

    return workflow.compile()


def run_workflow(graph: Any, message: str) -> dict[str, Any]:
    """Run the workflow for a user message."""
    initial_state: WorkflowState = {"question": message, "workflow": []}
    result = graph.invoke(initial_state)
    return dict(result)
