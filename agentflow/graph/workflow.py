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
from agentflow.conversation.context import ConversationContext
from agentflow.conversation.manager import ConversationManager
from agentflow.conversation.session_state import SessionState
from agentflow.graph.context import WorkflowContext
from agentflow.graph.executor import Executor
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

    # Conversation Runtime fields
    session_state: dict[str, Any]   # serialized SessionState dict
    _continue_mode: bool            # True = skip Router/Planner/Tools
    session_context: str            # human-readable context for AnswerAgent
    rewritten_question: str         # context-enriched question (Phase 7)
    conversation_context: Any       # ConversationContext object (Phase 7)


def build_workflow() -> Any:
    """Build the LangGraph workflow for the system.

    Flow:

      conversation_manager → (conditional)
        ├── continue mode → answer → memory → END
        └── new task → router → (conditional) knowledge / planner
                                           planner → (conditional) search / python / answer
                                           search → answer
                                           python → answer
                                           answer → memory → END
    """
    cm = ConversationManager()
    router = QueryRouterAgent()
    planner = PlannerAgent()
    search = SearchAgent()
    answer = AnswerAgent()
    memory = MemoryAgent()
    knowledge = KnowledgeAgent()
    python_executor = PythonAgent()
    executor = _build_executor()

    workflow = StateGraph(WorkflowState)

    # -- Conversation Manager (new entry point) --
    workflow.add_node("conversation_manager", _make_conversation_manager_node(cm))
    workflow.add_node("router", router.run)
    workflow.add_node("planner", planner.run)
    workflow.add_node("search", search.run)
    workflow.add_node("answer", answer.run)
    workflow.add_node("memory", memory.run)
    workflow.add_node("knowledge", knowledge.run)
    workflow.add_node("python", python_executor.run)

    # -- Executor is available via get_executor() for agents that opt in --

    workflow.set_entry_point("conversation_manager")

    # After conversation_manager: continue mode → skip to answer
    #                           new task → normal router flow
    workflow.add_conditional_edges(
        "conversation_manager",
        _route_after_conversation_manager,
        {
            "answer": "answer",
            "router": "router",
        },
    )

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


def _route_after_conversation_manager(state: WorkflowState) -> str:
    """After conversation manager: continue mode → answer, else → router."""
    if state.get("_continue_mode", False) or _session_is_waiting(state):
        return "answer"
    return "router"


def _session_is_waiting(state: WorkflowState) -> bool:
    """Check if the session_state has an active wait."""
    raw = state.get("session_state")
    if isinstance(raw, dict):
        return raw.get("status") == "waiting_user"
    elif hasattr(raw, "is_waiting"):
        return raw.is_waiting
    return False


def _make_conversation_manager_node(
    cm: ConversationManager,
) -> object:
    """Factory: creates a conversation_manager node for the LangGraph."""

    def _node(state: WorkflowState) -> dict[str, object]:
        """Entry point — resolves, rewrites, enriches question, decides flow."""
        question = state.get("question", "")
        memory = state.get("memory", {})

        # Deserialize session_state
        raw = state.get("session_state")
        if isinstance(raw, SessionState):
            session_state = raw
        else:
            session_state = SessionState.from_dict(raw) if isinstance(raw, dict) else SessionState()

        # 1. Resolve user input against session state (options, slots, anaphora)
        resolved = cm.resolve_question(question, session_state)

        # 2. Rewrite short/anaphoric questions with conversation context
        rewritten = cm.rewrite_question(resolved, session_state, memory)

        # 3. Build structured conversation context
        conv_ctx = cm.build_conversation_context(
            question, rewritten, session_state, memory,
        )

        # 4. Decide: continue or new task?
        should = cm.should_continue(session_state)
        waiting = session_state.is_waiting

        # Use the rewritten question for downstream nodes
        result: dict[str, object] = {
            "question": rewritten,
            "session_state": session_state,
            "_continue_mode": should or waiting,
            "rewritten_question": rewritten,
            "conversation_context": conv_ctx,
        }

        if should or waiting:
            result["session_context"] = str(session_state)
            logger.info(
                "Continue mode: goal='%s' waiting_for='%s' resolved='%s' rewritten='%s'",
                session_state.current_goal,
                session_state.waiting_for,
                resolved,
                rewritten,
            )
        else:
            logger.info(
                "New task: resolved='%s' rewritten='%s'",
                resolved, rewritten,
            )

        return result

    return _node


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
    """Route based on Plan (direct_answer) or category to the correct node."""

    # ---- direct_answer: skip tool nodes, go straight to answer ----
    plan = state.get("plan", {})
    if isinstance(plan, dict):
        direct_answer = plan.get("direct_answer", False)
    else:
        direct_answer = getattr(plan, "direct_answer", False)
    if direct_answer:
        return "answer"

    # ---- fallback: category-based routing (backward compat) ----
    category = state.get("category", "reasoning")
    if category == "search":
        return "search"
    if category == "python":
        return "python"
    return "answer"


# -- Executor global (lazy-initialised by build_workflow) -------------------

_executor_instance: Executor | None = None


def _build_executor() -> Executor:
    """Create and configure the shared Executor instance."""
    global _executor_instance
    if _executor_instance is not None:
        return _executor_instance

    ex = Executor()
    from agentflow.tools.python_tool import PythonTool
    from agentflow.tools.search_tool import SearchTool

    ex.register_tool("search", SearchTool())
    ex.register_tool("python", PythonTool())

    _executor_instance = ex
    logger.info("Executor initialised with tools: %s", ex.list_tools())
    return ex


def get_executor() -> Executor | None:
    """Return the shared Executor instance (``None`` if not yet built)."""
    return _executor_instance


def run_workflow(
    graph: Any,
    message: str,
    history: list[dict[str, str]] | None = None,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the workflow for a user message, optionally seeding with history
    and session state.

    Args:
        graph: Compiled LangGraph workflow.
        message: The user's message / question.
        history: Previous conversation history (``[{role, content}, ...]``).
        session_state: Serialized ``SessionState`` dict from previous turn.
            When provided, the ``conversation_manager`` node will use it for
            continuation planning.

    Returns:
        Dict with workflow results including ``answer``, ``memory``,
        ``session_state``, etc.
    """
    initial_state: WorkflowState = {
        "question": message,
        "workflow": [],
        "history": history or [],
    }

    if session_state:
        initial_state["session_state"] = session_state

    result = graph.invoke(initial_state)
    ctx = WorkflowContext(dict(result))
    return ctx.to_dict()
