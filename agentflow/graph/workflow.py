"""LangGraph workflow definition for the multi-agent system.

The updated architecture::

    ConversationManager → Router → Planner → Tool Executor → AnswerAgent → MemoryAgent

The ``tool_executor`` node is the central dispatch for all tool tasks
(filesystem, git, browser, database, mcp).  Existing search and python
nodes are preserved for backward compatibility.
"""

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
from agentflow.services.long_term_memory import LongTermMemory
from agentflow.tools.result import ToolResult
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
    session_state: SessionState
    _continue_mode: bool
    session_context: str
    _original_question: str
    rewritten_question: str
    conversation_context: Any
    rewritten_query: str

    # Tool execution results
    tool_results: list[dict[str, Any]]


def build_workflow() -> Any:
    """Build and compile the LangGraph workflow for the system.

    Returns a **fresh compiled instance** on every call.
    LangGraph's ``StateGraph`` is designed to be compiled per-invocation;
    caching the compiled graph as a module-level singleton would risk
    shared-state corruption under concurrent ``astream`` calls.

    Flow::

      conversation_manager → (conditional)
        ├── continue mode → knowledge → planner → (conditional)
        └── new task → router → (conditional) knowledge / planner
                                           planner → (conditional)
                                               search → answer
                                               python → answer
                                               tool_executor → answer
                                               direct_answer → answer
                                               answer → memory → END
    """
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
    # -- New: tool_executor dispatches all tool tasks through the Executor --
    workflow.add_node("tool_executor", _make_tool_executor_node(executor))

    workflow.set_entry_point("conversation_manager")

    # After conversation_manager
    workflow.add_conditional_edges(
        "conversation_manager",
        _route_after_conversation_manager,
        {"knowledge": "knowledge", "router": "router"},
    )

    # After router
    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {"knowledge": "knowledge", "planner": "planner"},
    )

    # After knowledge — always route to planner (even in continue mode)
    # so the planner can decide whether new tools are needed or just answer.
    workflow.add_conditional_edges(
        "knowledge",
        _route_after_knowledge,
        {"planner": "planner"},
    )

    # After planner: route based on required tools
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "query_rewriter": "query_rewriter",
            "python": "python",
            "tool_executor": "tool_executor",
            "answer": "answer",
        },
    )

    # After query_rewriter: always go to search
    workflow.add_edge("query_rewriter", "search")

    # Execution nodes converge to answer
    workflow.add_edge("search", "answer")
    workflow.add_edge("python", "answer")
    workflow.add_edge("tool_executor", "answer")

    # Finalize
    workflow.add_edge("answer", "memory")
    workflow.add_edge("memory", END)

    return workflow.compile()


# -- Routing functions ---------------------------------------------------------


def _route_after_conversation_manager(state: WorkflowState) -> str:
    """Continue mode → knowledge, else → router."""
    if state.get("_continue_mode", False) or _session_is_waiting(state):
        return "knowledge"
    return "router"


def _route_after_knowledge(state: WorkflowState) -> str:
    """Always route to planner — it decides whether tools or a direct answer are needed.

    Previously this function routed directly to ``answer`` in continue mode,
    which meant the ``AnswerAgent`` had to produce a response without any
    plan at all.  Routing through the planner lets it emit ``direct_answer``
    (which goes straight to ``answer``) while keeping the option to create
    new tool tasks if the turn warrants them.
    """
    return "planner"


def _session_is_waiting(state: WorkflowState) -> bool:
    ss = state.get("session_state")
    return ss is not None and ss.is_waiting


def _route_after_router(state: WorkflowState) -> str:
    """Route after router. Identity/search skip knowledge retrieval."""
    category = state.get("category", "reasoning")
    if category in ("identity", "search"):
        return "planner"
    return "knowledge"


def _route_after_planner(state: WorkflowState) -> str:
    """Route based on Plan tasks to the correct execution node.

    Priority:
      1. direct_answer → answer
      2. search tasks → query_rewriter
      3. python tasks → python
      4. filesystem/git/browser/database/mcp tasks → tool_executor
      5. fallback by category
    """
    plan = state.get("plan", {})

    # ---- direct_answer: skip tool nodes entirely ----
    if isinstance(plan, dict):
        direct_answer = plan.get("direct_answer", False)
    else:
        direct_answer = getattr(plan, "direct_answer", False)
    if direct_answer:
        return "answer"

    # ---- Check plan tasks for required tools ----
    tasks = _get_plan_tasks(plan)
    for task in tasks:
        tool = task.get("tool", "") or ""
        if tool == "search":
            return "query_rewriter"
        if tool == "python":
            return "python"
        if tool in ("filesystem", "git", "browser", "database", "mcp", "composio"):
            return "tool_executor"

    # ---- Fallback: category-based routing (backward compat) ----
    category = state.get("category", "reasoning")
    if category == "search":
        return "query_rewriter"
    if category == "python":
        return "python"
    return "answer"


def _get_plan_tasks(plan: Any) -> list[dict[str, Any]]:
    """Extract task dicts from a Plan object or dict."""
    if isinstance(plan, dict):
        tasks = plan.get("tasks", [])
    else:
        tasks = getattr(plan, "tasks", [])
    return [
        t if isinstance(t, dict) else t.to_dict() if hasattr(t, "to_dict") else {}
        for t in tasks
    ]


# -- Node factories ------------------------------------------------------------


def _make_query_rewriter_node(qr: QueryRewriter) -> object:
    """Factory: creates a query_rewriter node for the LangGraph."""

    def _node(state: WorkflowState) -> dict[str, object]:
        question = str(
            state.get("_original_question", "")
            or state.get("question", "")
        )
        session_state = state.get("session_state")
        history = state.get("history", None)

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


def _make_conversation_manager_node(cm: ConversationManager) -> object:
    """Factory: creates a conversation_manager node."""

    def _node(state: WorkflowState) -> dict[str, object]:
        question = state.get("question", "")
        memory = state.get("memory", {})
        session_state = state.get("session_state", SessionState())

        resolved = cm.resolve_question(question, session_state)
        rewritten = cm.rewrite_question(resolved, session_state, memory)
        conv_ctx = cm.build_conversation_context(question, rewritten, session_state, memory)
        should = cm.should_continue(session_state)
        waiting = session_state.is_waiting

        result: dict[str, object] = {
            "question": rewritten,
            "session_state": session_state,
            "_continue_mode": should or waiting,
            "_original_question": question,
            "rewritten_question": rewritten,
            "conversation_context": conv_ctx,
        }

        ctx_str = str(session_state)
        lt_memories = _recall_long_term_memories(rewritten, ctx_str)
        result["session_context"] = lt_memories

        if should or waiting:
            logger.info(
                "Continue mode: goal='%s' waiting_for='%s'",
                session_state.current_goal, session_state.waiting_for,
            )
        else:
            logger.info("New task: resolved='%s' rewritten='%s'", resolved, rewritten)

        return result

    return _node


def _make_tool_executor_node(executor: Executor) -> object:
    """Factory: creates a tool_executor node that runs all Plan tasks.

    The node extracts tasks from the Plan and dispatches them through
    the Executor (which delegates to ToolRegistry → BaseTool).
    """

    def _node(state: WorkflowState) -> dict[str, object]:
        plan = state.get("plan", {})
        tasks = _get_plan_tasks(plan)

        if not tasks:
            logger.info("Tool executor: no tasks to execute")
            return {"tool_results": []}

        ctx = WorkflowContext(dict(state))
        results: list[ToolResult] = []
        for task_dict in tasks:
            tool = task_dict.get("tool", "")
            if tool in ("search", "python"):
                # These are handled by their dedicated nodes — skip here
                continue
            result = executor.execute_task_dict(task_dict, ctx=ctx)
            results.append(result)

        logger.info(
            "Tool executor: %d/%d tasks completed",
            sum(1 for r in results if r.success), len(results),
        )

        return {"tool_results": [r.to_dict() for r in results]}

    return _node


# -- Long-term memory recall -------------------------------------------------


def _recall_long_term_memories(question: str, session_context: str) -> str:
    """Append relevant long-term memories to the session context."""
    try:
        ltm = LongTermMemory()
        memory_text = ltm.recall_for_question(question)
        if memory_text:
            return f"{session_context}\n\n{memory_text}" if session_context else memory_text
    except Exception:
        logger.debug("Long-term memory recall failed (non-critical)")
    return session_context


# -- Executor global (lazy-initialised by build_workflow) -------------------


_executor_instance: Executor | None = None


def _build_executor() -> Executor:
    """Create and configure the shared Executor instance.

    Registers all available tools.  New tools are added here.
    """
    global _executor_instance
    if _executor_instance is not None:
        return _executor_instance

    ex = Executor()

    # -- Register tools (plugin model) --
    from agentflow.tools.search_tool import SearchTool
    from agentflow.tools.python_tool import PythonTool
    from agentflow.tools.filesystem_tool import FileSystemTool
    from agentflow.tools.git_tool import GitTool
    from agentflow.tools.browser_tool import BrowserTool
    from agentflow.tools.database_tool import DatabaseTool
    from agentflow.tools.mcp_tool import MCPTool
    from agentflow.tools.composio_tool import ComposioTool

    ex.registry.register(SearchTool())
    ex.registry.register(PythonTool())
    ex.registry.register(FileSystemTool())
    ex.registry.register(GitTool())
    ex.registry.register(BrowserTool())
    ex.registry.register(DatabaseTool())
    ex.registry.register(MCPTool())
    ex.registry.register(ComposioTool())

    # Warn about stub tools that have no real implementation yet
    _stub_tools = {"browser": BrowserTool, "database": DatabaseTool, "mcp": MCPTool}
    for name, cls in _stub_tools.items():
        meta = cls().metadata()
        if meta.get("status") == "interface_only":
            logger.warning(
                "Tool '%s' is an interface placeholder (status=interface_only). "
                "Its actions will fail until a concrete implementation is provided.",
                name,
            )

    _executor_instance = ex
    logger.info("Executor initialised with tools: %s", ex.list_tools())
    return ex


def get_executor() -> Executor | None:
    """Return the shared Executor instance (``None`` if not yet built)."""
    return _executor_instance


# -- Run workflow -------------------------------------------------------------


def run_workflow(
    graph: Any,
    message: str,
    history: list[dict[str, str]] | None = None,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the workflow for a user message."""
    initial_state: WorkflowState = {
        "question": message,
        "workflow": [],
        "history": history or [],
    }

    if history:
        initial_state["memory"] = {"history": list(history)}

    if session_state:
        initial_state["session_state"] = SessionState.from_dict(session_state)

    result = graph.invoke(initial_state)
    ctx = WorkflowContext(dict(result))
    return ctx.to_dict()
