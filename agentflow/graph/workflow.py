"""LangGraph workflow definition for the goal-driven multi-agent system.

The goal-driven architecture::

    ConversationManager
         │
         ▼
    GoalAnalyzer ──→ CapabilityAnalyzer ──→ Knowledge ──→ Planner
                                                             │
                                              ┌──────────────┼──────────────┐
                                              ▼              ▼              ▼
                                        tool_executor    python      query_rewriter
                                              │              │              │
                                              ▼              ▼              ▼
                                          Reflector ──────→│         search → answer
                                              │              │
                                  ┌───────────┼───────────┐  │
                                  ▼           ▼           ▼  │
                              planner   tool_executor  answer │
                                  (replan)   (retry)    (done)│
                                                             │
                                                             ▼
                                                          memory → END

Core principles:
  1. Goal-Driven: Router is replaced by GoalAnalyzer (LLM-based goal understanding)
  2. No regex: Intent is determined by LLM, not pattern matching
  3. Autonomous: Planner generates Task Trees, Executor runs them, Reflector decides next step
  4. Loop until done: Execute → Reflect → Continue/Replan until Goal Completed
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agentflow.agents.answer.agent import AnswerAgent
from agentflow.agents.capability_analyzer.agent import CapabilityAnalyzer
from agentflow.agents.goal_analyzer.agent import GoalAnalyzer
from agentflow.agents.knowledge.agent import KnowledgeAgent
from agentflow.agents.memory.agent import MemoryAgent
from agentflow.agents.planner.agent import PlannerAgent
from agentflow.agents.python.agent import PythonAgent
from agentflow.agents.reflection.agent import ReflectionAgent
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
    """Typed state container for workflow nodes (goal-driven architecture)."""

    question: str
    workflow: list[str]
    category: str  # Backward compat — set from goal_type
    plan: dict[str, Any]
    search_results: list[dict[str, Any]]
    knowledge_results: list[dict[str, Any]]
    knowledge_context: str
    python_result: dict[str, Any]
    answer: str
    memory: dict[str, Any]
    history: list[dict[str, str]]
    router: dict[str, Any]  # Backward compat — now carries goal info

    # ── Goal-Driven fields (replaces RouterAgent) ──
    goal_analysis: dict[str, Any]       # GoalAnalyzer output
    capability_analysis: dict[str, Any]  # CapabilityAnalyzer output

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

    # Task Queue (Dynamic Task Queue Planning)
    task_queue: list[dict]    # serialized Task objects

    # Reflection loop state
    _reflection_result: str   # "done" | "next" | "replan" | "retry"
    _reflection_message: str  # context for re-plan on failure
    _replan_count: int        # number of re-plan iterations
    _reflection_output: dict[str, Any]  # ReflectionAgent structured output


def build_workflow() -> Any:
    """Build and compile the goal-driven LangGraph workflow.

    Returns a **fresh compiled instance** on every call.
    """
    cm = ConversationManager()
    goal_analyzer = GoalAnalyzer()
    capability_analyzer = CapabilityAnalyzer()
    planner = PlannerAgent()
    query_rewriter = QueryRewriter()
    search = SearchAgent()
    answer = AnswerAgent()
    memory = MemoryAgent()
    knowledge = KnowledgeAgent()
    python_executor = PythonAgent()
    reflection = ReflectionAgent()
    executor = _build_executor()

    workflow = StateGraph(WorkflowState)

    # ── Nodes ──
    workflow.add_node("conversation_manager", _make_conversation_manager_node(cm))
    workflow.add_node("goal_analyzer", goal_analyzer.run)
    workflow.add_node("capability_analyzer", capability_analyzer.run)
    workflow.add_node("knowledge", knowledge.run)
    workflow.add_node("planner", planner.run)
    workflow.add_node("query_rewriter", _make_query_rewriter_node(query_rewriter))
    workflow.add_node("search", search.run)
    workflow.add_node("tool_executor", _make_tool_executor_node(executor))
    workflow.add_node("reflector", reflection.run)
    workflow.add_node("python", python_executor.run)
    workflow.add_node("answer", answer.run)
    workflow.add_node("memory", memory.run)

    workflow.set_entry_point("conversation_manager")

    # ── Linear flow through goal analysis stack ──
    workflow.add_edge("conversation_manager", "goal_analyzer")
    workflow.add_edge("goal_analyzer", "capability_analyzer")
    workflow.add_edge("capability_analyzer", "knowledge")
    workflow.add_edge("knowledge", "planner")

    # ── Planner → execution nodes (conditional) ──
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

    # ── Search chain ──
    workflow.add_edge("query_rewriter", "search")
    workflow.add_edge("search", "answer")

    # ── Execution → Reflection ──
    workflow.add_edge("python", "reflector")
    workflow.add_edge("tool_executor", "reflector")

    # ── Reflection loop ──
    workflow.add_conditional_edges(
        "reflector",
        _route_after_reflector,
        {
            "planner": "planner",            # Re-plan on failure
            "tool_executor": "tool_executor", # Retry / continue
            "query_rewriter": "query_rewriter",  # Search task
            "python": "python",               # Python task
            "answer": "answer",               # All done → summarise
        },
    )

    # ── Finalise ──
    workflow.add_edge("answer", "memory")
    workflow.add_edge("memory", END)

    return workflow.compile()


# =========================================================================
# Routing functions
# =========================================================================


def _route_after_planner(state: WorkflowState) -> str:
    """Route based on Plan and Task Queue to execution node.

    Priority: goal_completed, direct_answer, then highest-priority TODO.
    """
    plan = state.get("plan", {})
    if isinstance(plan, dict):
        if plan.get("goal_completed") or plan.get("direct_answer"):
            return "answer"
    else:
        if plan.goal_completed or plan.direct_answer:
            return "answer"

    queue = state.get("task_queue", [])
    next_task = _find_highest_priority_todo(queue)
    if next_task:
        tool = next_task.get("tool", "") or ""
        if tool in ("filesystem", "git", "browser", "database", "mcp", "composio"):
            return "tool_executor"
        if tool == "search":
            return "query_rewriter"
        if tool == "python":
            return "python"

    return "answer"


def _route_after_reflector(state: WorkflowState) -> str:
    """Reflection router for Dynamic Task Queue.

    Reads ``state["_reflection_result"]`` set by the ReflectionAgent.

    Returns:
        "planner"       — re-plan or need more tasks
        "tool_executor" — retry failed tasks or continue with next TODO
        "answer"        — all tasks completed, goal achieved
    """
    result = state.get("_reflection_result", "done")
    replan_count = int(state.get("_replan_count", 0))

    if result == "done":
        logger.info("Reflector -> answer (goal completed)")
        return "answer"

    if result == "replan":
        if replan_count >= 3:
            logger.warning("Reflector: max re-plans (%d) reached, forcing answer", replan_count)
            return "answer"
        logger.info("Reflector -> re-plan (attempt %d/3)", replan_count)
        return "planner"

    if result == "retry":
        logger.info("Reflector -> retry (re-execute failed tasks)")
        return "tool_executor"

    # "next" — normal continuation: route to executor or planner
    queue = state.get("task_queue", [])
    next_task = _find_highest_priority_todo(queue)
    if next_task:
        tool = next_task.get("tool", "") or ""
        if tool in ("filesystem", "git", "browser", "database", "mcp", "composio"):
            logger.info("Reflector -> tool_executor (next TODO: %s)", next_task.get("task_id", "?"))
            return "tool_executor"
        if tool == "search":
            return "query_rewriter"
        if tool == "python":
            return "python"

    # No TODO tasks but goal not completed -> need more from planner
    logger.info("Reflector -> planner (need more tasks)")
    return "planner"


def _find_highest_priority_todo(queue: list[dict]) -> dict | None:
    """Find the highest-priority TODO task in the task queue."""
    todo = [t for t in queue if t.get("status", "") == "todo"]
    if not todo:
        return None
    return max(todo, key=lambda t: t.get("priority", 0))


def _get_plan_tasks(plan: Any) -> list[dict[str, Any]]:
    """Extract task dicts from a Plan object or dict."""
    if isinstance(plan, dict):
        tasks = plan.get("tasks", [])
    else:
        tasks = getattr(plan, "tasks", [])
    return [
        t if isinstance(t, dict)
        else t.to_dict() if hasattr(t, "to_dict")
        else {}
        for t in tasks
    ]


# =========================================================================
# Node factories
# =========================================================================


def _make_conversation_manager_node(cm: ConversationManager) -> object:
    """Factory: creates a conversation_manager node."""

    def _node(state: WorkflowState) -> dict[str, object]:
        question = state.get("question", "")
        memory = state.get("memory", {})
        session_state = state.get("session_state", SessionState())

        resolved = cm.resolve_question(question, session_state)
        rewritten = cm.rewrite_question(resolved, session_state, memory)
        conv_ctx = cm.build_conversation_context(
            question, rewritten, session_state, memory,
        )
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


def _make_query_rewriter_node(qr: QueryRewriter) -> object:
    """Factory: creates a query_rewriter node."""

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


def _make_tool_executor_node(executor: Executor) -> object:
    """Factory: creates a tool_executor node that runs one task at a time."""

    def _node(state: WorkflowState) -> dict[str, object]:
        queue = list(state.get("task_queue", []) or [])

        # Find highest priority TODO
        next_task = _find_highest_priority_todo(queue)
        if not next_task:
            logger.info("Tool executor: no TODO tasks in queue")
            return {"tool_results": []}

        # Mark running
        next_task["status"] = "running"

        ctx = WorkflowContext(dict(state))
        result = executor.execute_task_dict(next_task, ctx=ctx)

        # Update status based on result
        if result and result.success:
            next_task["status"] = "done"
        elif result:
            next_task["status"] = "failed"
            next_task["error"] = result.error or "Unknown error"
        else:
            next_task["status"] = "failed"
            next_task["error"] = "No result from executor"

        logger.info(
            "Tool executor: %s -> %s",
            next_task.get("task_id", "?"), next_task["status"],
        )
        return {"task_queue": queue, "tool_results": [result.to_dict()] if result else []}

    return _node


# =========================================================================
# Long-term memory recall
# =========================================================================


def _recall_long_term_memories(question: str, session_context: str) -> str:
    """Append relevant long-term memories to the session context."""
    try:
        ltm = LongTermMemory()
        memory_text = ltm.recall_for_question(question)
        if memory_text:
            return (
                f"{session_context}\n\n{memory_text}"
                if session_context else memory_text
            )
    except Exception:
        logger.debug("Long-term memory recall failed (non-critical)")
    return session_context


# =========================================================================
# Executor (lazy-initialised singleton)
# =========================================================================


_executor_instance: Executor | None = None


def _build_executor() -> Executor:
    """Create and configure the shared Executor instance."""
    global _executor_instance
    if _executor_instance is not None:
        return _executor_instance

    ex = Executor()

    from agentflow.tools.search_tool import SearchTool
    from agentflow.tools.python_tool import PythonTool
    from agentflow.tools.filesystem_tool import FileSystemTool
    from agentflow.tools.git_tool import GitTool
    from agentflow.tools.browser_tool import BrowserTool
    from agentflow.tools.database_tool import DatabaseTool
    from agentflow.tools.mcp_tool import MCPTool
    from agentflow.tools.composio_tool import ComposioTool

    from pathlib import Path

    # FileSystemTool writes to outputs/ so the frontend API (which scans
    # that directory by default) can discover generated files.
    _outputs_dir = Path(__file__).resolve().parents[2] / "outputs"
    _outputs_dir.mkdir(exist_ok=True)

    ex.registry.register(SearchTool())
    ex.registry.register(PythonTool())
    ex.registry.register(FileSystemTool(workspace=str(_outputs_dir)))
    ex.registry.register(GitTool())
    ex.registry.register(BrowserTool())
    ex.registry.register(DatabaseTool())
    ex.registry.register(MCPTool())
    ex.registry.register(ComposioTool())

    _stub_tools = {"browser": BrowserTool, "database": DatabaseTool, "mcp": MCPTool}
    for name, cls in _stub_tools.items():
        meta = cls().metadata()
        if meta.get("status") == "interface_only":
            logger.warning(
                "Tool '%s' is an interface placeholder (status=interface_only).",
                name,
            )

    _executor_instance = ex
    logger.info("Executor initialised with tools: %s", ex.list_tools())
    return ex


def get_executor() -> Executor | None:
    """Return the shared Executor instance (``None`` if not yet built)."""
    return _executor_instance


# =========================================================================
# Run workflow
# =========================================================================


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
