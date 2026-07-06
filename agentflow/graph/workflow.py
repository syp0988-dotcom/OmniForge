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
from agentflow.agents.search.query_rewriter import QueryRewriter
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
    session_state: SessionState     # SessionState object (unified type)
    _continue_mode: bool            # True = skip Router/Planner/Tools
    session_context: str            # human-readable context for AnswerAgent
    _original_question: str         # original user input (for Router classification)
    rewritten_question: str         # context-enriched question (Phase 7)
    conversation_context: Any       # ConversationContext object (Phase 7)
    rewritten_query: str            # search-optimised query from QueryRewriter


# Compiled workflow cache (built once, reused across requests)
_compiled_workflow: Any | None = None


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

    The compiled graph is cached after the first call and reused thereafter.
    """
    global _compiled_workflow
    if _compiled_workflow is not None:
        return _compiled_workflow

    cm = ConversationManager()
    router = QueryRouterAgent()
    planner = PlannerAgent()
    query_rewriter = QueryRewriter()
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
    workflow.add_node("query_rewriter", _make_query_rewriter_node(query_rewriter))
    workflow.add_node("search", search.run)
    workflow.add_node("answer", answer.run)
    workflow.add_node("memory", memory.run)
    workflow.add_node("knowledge", knowledge.run)
    workflow.add_node("python", python_executor.run)

    # -- Executor is available via get_executor() for agents that opt in --

    workflow.set_entry_point("conversation_manager")

    # After conversation_manager: continue mode → knowledge → answer
    #                           new task → normal router flow
    workflow.add_conditional_edges(
        "conversation_manager",
        _route_after_conversation_manager,
        {
            "knowledge": "knowledge",
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

    # After knowledge: continue mode → answer; normal flow → planner
    workflow.add_conditional_edges(
        "knowledge",
        _route_after_knowledge,
        {
            "answer": "answer",
            "planner": "planner",
        },
    )

    # After planner: route based on category
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "query_rewriter": "query_rewriter",
            "python": "python",
            "answer": "answer",
        },
    )

    # After query_rewriter: always go to search
    workflow.add_edge("query_rewriter", "search")

    # Execution nodes converge to answer
    workflow.add_edge("search", "answer")
    workflow.add_edge("python", "answer")

    # Finalize
    workflow.add_edge("answer", "memory")
    workflow.add_edge("memory", END)

    _compiled_workflow = workflow.compile()
    return _compiled_workflow


def _route_after_conversation_manager(state: WorkflowState) -> str:
    """After conversation manager: continue mode → knowledge, else → router."""
    if state.get("_continue_mode", False) or _session_is_waiting(state):
        return "knowledge"
    return "router"


def _route_after_knowledge(state: WorkflowState) -> str:
    """After knowledge: continue mode → answer, else → planner."""
    if state.get("_continue_mode", False) or _session_is_waiting(state):
        return "answer"
    return "planner"


def _session_is_waiting(state: WorkflowState) -> bool:
    """Check if the session_state has an active wait."""
    ss = state.get("session_state")
    return ss is not None and ss.is_waiting


def _make_query_rewriter_node(qr: QueryRewriter) -> object:
    """Factory: creates a query_rewriter node for the LangGraph."""

    def _node(state: WorkflowState) -> dict[str, object]:
        """Rewrite user question into an optimised search query."""
        # Use raw original question (not LLM-enriched) for search query building.
        # The QueryRewriter does its own context recovery from session_state.
        question = str(
            state.get("_original_question", "")
            or state.get("question", "")
        )
        session_state = state.get("session_state")
        history = state.get("history", None)

        # Extract intent from plan
        plan = state.get("plan", {})
        if isinstance(plan, dict):
            intent = plan.get("intent", "")
        else:
            intent = getattr(plan, "intent", "")

        rewritten = qr.rewrite(
            question=question,
            session_state=session_state,
            history=history,
            intent=intent,
        )

        logger.info(
            "QueryRewriter: '%s' → '%s' (intent=%s)",
            question[:50], rewritten[:80], intent,
        )

        return {"rewritten_query": rewritten}

    return _node


def _make_conversation_manager_node(
    cm: ConversationManager,
) -> object:
    """Factory: creates a conversation_manager node for the LangGraph."""

    def _node(state: WorkflowState) -> dict[str, object]:
        """Entry point — resolves, rewrites, enriches question, decides flow."""
        question = state.get("question", "")
        memory = state.get("memory", {})

        # session_state is a SessionState object (set by run_workflow or previous turn)
        session_state = state.get("session_state", SessionState())

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
            "session_state": session_state,          # SessionState object, not dict
            "_continue_mode": should or waiting,
            "_original_question": question,        # raw input for Router classification
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
    """Route based on Plan to the correct execution node."""

    # ---- direct_answer: skip tool nodes, go straight to answer ----
    plan = state.get("plan", {})
    if isinstance(plan, dict):
        direct_answer = plan.get("direct_answer", False)
    else:
        direct_answer = getattr(plan, "direct_answer", False)
    if direct_answer:
        return "answer"

    # ---- Check plan tasks for required tools ----
    if isinstance(plan, dict):
        tasks = plan.get("tasks", [])
    else:
        tasks = getattr(plan, "tasks", [])
    for task in tasks:
        if isinstance(task, dict):
            tool = task.get("tool", "") or ""
        else:
            tool = getattr(task, "tool", "") or ""
        if tool == "search":
            return "query_rewriter"
        if tool == "python":
            return "python"

    # ---- fallback: category-based routing (backward compat) ----
    category = state.get("category", "reasoning")
    if category == "search":
        return "query_rewriter"
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

    # Seed memory with conversation history so the LLM sees past turns.
    # Without this, MemoryAgent starts fresh each invocation and the
    # AnswerAgent's prompt has no history beyond the current turn.
    if history:
        initial_state["memory"] = {"history": list(history)}

    # API boundary: convert serialized dict → SessionState object
    if session_state:
        initial_state["session_state"] = SessionState.from_dict(session_state)

    result = graph.invoke(initial_state)
    # API boundary: WorkflowContext.to_dict() handles SessionState → dict
    ctx = WorkflowContext(dict(result))
    return ctx.to_dict()
