"""LangGraph workflow definition for the goal-driven multi-agent system.

The goal-driven architecture::

    ConversationManager
         │
         ▼
    GoalAnalyzer ──→ Knowledge ──→ Planner
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

    # ── Goal-Driven fields ──
    goal_analysis: dict[str, Any]       # GoalAnalyzer output

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

    # Degraded / fallback mode
    _degraded: bool           # LLM unavailable — skip LLM-dependent agents
    _llm_error: str           # Last LLM error detail for fallback messaging


def build_workflow() -> Any:
    """Build and compile the goal-driven LangGraph workflow.

    Returns a **cached compiled instance**.  Agents and the compiled
    graph are built once and reused across requests for performance.
    """
    global _workflow_cache
    if _workflow_cache is not None:
        return _workflow_cache

    cm = ConversationManager()
    goal_analyzer = GoalAnalyzer()
    executor = _build_executor()
    planner = PlannerAgent(registry=executor.registry)
    query_rewriter = QueryRewriter()
    search = SearchAgent()
    answer = AnswerAgent()
    memory = MemoryAgent()
    knowledge = KnowledgeAgent()
    python_executor = PythonAgent()
    reflection = ReflectionAgent()

    workflow = StateGraph(WorkflowState)

    # ── Nodes ──
    workflow.add_node("conversation_manager", _make_conversation_manager_node(cm))
    workflow.add_node("goal_analyzer", goal_analyzer.run)
    workflow.add_node("knowledge", knowledge.run)
    workflow.add_node("planner", _make_planner_node(planner))
    workflow.add_node("query_rewriter", _make_query_rewriter_node(query_rewriter))
    workflow.add_node("search", _make_search_node(search))
    workflow.add_node("tool_executor", _make_tool_executor_node(executor))
    workflow.add_node("reflector", reflection.run)
    workflow.add_node("python", python_executor.run)
    workflow.add_node("answer", answer.run)
    workflow.add_node("memory", memory.run)

    workflow.set_entry_point("conversation_manager")

    # ── Linear flow through goal analysis stack ──
    workflow.add_edge("conversation_manager", "goal_analyzer")

    # Conditional: only query knowledge base for goal types that need it
    workflow.add_conditional_edges(
        "goal_analyzer",
        _route_after_goal_analyzer,
        {
            "knowledge": "knowledge",
            "planner": "planner",
            "answer": "answer",
        },
    )

    workflow.add_edge("knowledge", "planner")

    # ── Planner → execution nodes (conditional) ──
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "query_rewriter": "query_rewriter",
            "python": "python",
            "tool_executor": "tool_executor",
            "reflector": "reflector",
            "answer": "answer",
        },
    )

    # ── Search chain ──
    workflow.add_edge("query_rewriter", "search")
    workflow.add_edge("search", "answer")

    # ── Execution → conditional routing ──
    # On success with more TODO tasks: skip the LLM-based reflector and
    # route directly to the next executor.  Only reflect on failures,
    # periodic checkpoints, or when no more tasks remain.
    workflow.add_conditional_edges(
        "tool_executor",
        _route_after_executor,
        {
            "reflector": "reflector",
            "tool_executor": "tool_executor",
            "query_rewriter": "query_rewriter",
            "python": "python",
        },
    )
    workflow.add_conditional_edges(
        "python",
        _route_after_executor,
        {
            "reflector": "reflector",
            "tool_executor": "tool_executor",
            "query_rewriter": "query_rewriter",
            "python": "python",
        },
    )

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

    _workflow_cache = workflow.compile()
    logger.info("Workflow compiled and cached")
    return _workflow_cache


# =========================================================================
# Routing functions
# =========================================================================


def _route_after_goal_analyzer(state: WorkflowState) -> str:
    """Route based on goal_type AND knowledge_source (dual-framework).

    Three knowledge paths:
      1. ``"general"`` → answer directly (skip Knowledge + Planner).
         LLM answers from its own parametric knowledge — fastest path.
      2. ``"local"``   → Knowledge → Planner → ... (current flow).
         RAG with local document retrieval.
      3. ``"hybrid"``  → Knowledge → Planner → ... then Answer fuses both.
         RAG context + LLM own knowledge combined naturally.

    Also considers goal_type (6-class simplified system):
      - ``other`` → direct answer (chat, editing, translation, no planning)
      - ``question`` → use knowledge_source (general → answer, local/hybrid → RAG)
      - ``coding / project / search / tool_use`` → planner (skip knowledge)
    """
    goal = state.get("goal_analysis", {})
    goal_type = goal.get("goal_type", "") if isinstance(goal, dict) else ""
    knowledge_source = goal.get("knowledge_source", "") if isinstance(goal, dict) else ""
    source_mode = str(state.get("source_mode", "") or "")
    if not source_mode and isinstance(goal, dict):
        source_mode = str(goal.get("source_mode", "") or "")

    # Manual source selection is an explicit user command. When the user
    # chooses Knowledge Base, always retrieve local references first; "auto"
    # is the only mode where intent routing may skip retrieval.
    if source_mode == "knowledge":
        logger.info("Source mode 'knowledge': forcing KnowledgeAgent retrieval")
        return "knowledge"

    # Goal types that never need planning — conversational, informational
    _DIRECT_ANSWER_TYPES = frozenset({"other", "translation", "editing"})

    # Goal types that may benefit from knowledge base retrieval
    _KNOWLEDGE_GOAL_TYPES = frozenset({
        "question", "analysis", "document",
    })

    # 0) Non-actionable types → skip planner entirely, go straight to answer.
    #    Conversational statements like "我叫张三" don't need task planning.
    if goal_type in _DIRECT_ANSWER_TYPES:
        logger.info("Goal type '%s': non-actionable, routing directly to AnswerAgent", goal_type)
        return "answer"

    # 1) General knowledge (fast path) — only for question/analysis types.
    #    Project/coding types always need Planner regardless of knowledge_source.
    if knowledge_source == "general" and goal_type in _KNOWLEDGE_GOAL_TYPES:
        logger.info("Knowledge source 'general': routing directly to AnswerAgent")
        return "answer"

    # 2) Non-project types that benefit from RAG retrieval
    if goal_type in _KNOWLEDGE_GOAL_TYPES:
        logger.info(
            "Goal type '%s' knowledge='%s': routing to KnowledgeAgent",
            goal_type, knowledge_source,
        )
        return "knowledge"

    logger.info("Goal type '%s': skipping KnowledgeAgent", goal_type)
    return "planner"


def _route_after_executor(state: WorkflowState) -> str:
    """Route after a tool/python execution.

    Skips the LLM-based reflector for routine successful continuations.
    Only reflects when:
      - The last task failed (need retry/replan decision)
      - No more TODO tasks remain (need goal_completed check)
      - Every N successful tasks (configurable periodic checkpoint)

    Routing is derived dynamically from the ToolRegistry — no hardcoded
    tool-name lists.
    """
    # Check if the last task failed
    tool_results = state.get("tool_results", [])
    if tool_results:
        last = tool_results[-1]
        if isinstance(last, dict) and not last.get("success", True):
            logger.info("Executor -> reflector (task failed)")
            return "reflector"

    # Success: check for more TODO tasks
    queue = state.get("task_queue", [])
    next_task = _find_highest_priority_todo(queue)
    if next_task:
        tool = next_task.get("tool", "") or ""
        if tool == "knowledge":
            logger.info("Executor: knowledge tool task, routing to reflector")
            return "reflector"
        node = _resolve_tool_node(tool)
        if node:
            return node
        logger.warning("Executor: unknown tool '%s', trying tool_executor", tool)
        return "tool_executor"

    # No more TODO tasks → reflector for completion check
    logger.info("Executor -> reflector (no more TODO tasks)")
    return "reflector"


def _route_after_planner(state: WorkflowState) -> str:
    """Route based on Plan and Task Queue to execution node.

    Priority: goal_completed, direct_answer, then highest-priority TODO.
    When there are no TODO tasks and the plan says incomplete, routes to
    ``reflector`` for LLM evaluation instead of going to ``answer``.

    All tool→node mappings are resolved dynamically from the ToolRegistry
    instead of hardcoded per-tool lists.
    """
    # Planner may force completion after too many cycles without progress
    if state.get("_reflection_result") == "done":
        logger.info("Router: planner forced completion -> answer")
        return "answer"

    plan = state.get("plan", {})
    if isinstance(plan, dict):
        plan_completed = plan.get("goal_completed") or plan.get("direct_answer")
    else:
        plan_completed = plan.goal_completed or plan.direct_answer

    if plan_completed:
        logger.info("Router: plan completed -> answer")
        return "answer"

    queue = state.get("task_queue", [])
    next_task = _find_highest_priority_todo(queue)
    if next_task:
        tool = next_task.get("tool", "") or ""
        logger.info("Router: next TODO task tool=%s id=%s -> dispatcher", tool, next_task.get("task_id", "?"))
        if tool == "knowledge":
            logger.info("Router: knowledge tool task (pre-planner node), routing to reflector")
            return "reflector"
        node = _resolve_tool_node(tool)
        if node:
            return node
        # Fallback: try tool_executor for unknown tools
        logger.warning("Router: unknown tool '%s', trying tool_executor", tool)
        return "tool_executor"

    # No TODO tasks but plan says incomplete → let the LLM-based reflector
    # evaluate whether the goal is truly done or needs more tasks.
    logger.info("Router: no TODO tasks, plan incomplete -> reflector")
    return "reflector"


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
        if tool == "knowledge":
            logger.info("Reflector: knowledge tool task, skipping -> planner")
            return "planner"
        node = _resolve_tool_node(tool)
        if node:
            logger.info("Reflector -> %s (next TODO: %s)", node, next_task.get("task_id", "?"))
            return node
        logger.warning("Reflector: unknown tool '%s', trying tool_executor", tool)
        return "tool_executor"

    # No TODO tasks but goal not completed -> need more from planner.
    # Guard against infinite planner↔reflector loops: check cycle count.
    cycle_count = int(state.get("_planner_cycle_count", 0))
    stuck_rounds = int(state.get("_stuck_rounds", 0))
    if stuck_rounds >= 3:
        logger.warning("Reflector: %d stuck rounds without TODO, forcing answer", stuck_rounds)
        return "answer"
    if cycle_count >= 4:
        logger.warning("Reflector: %d planner cycles without progress, forcing answer", cycle_count)
        return "answer"
    logger.info("Reflector -> planner (need more tasks, cycle %d)", cycle_count)
    return "planner"


def _resolve_tool_node(tool: str) -> str | None:
    """Resolve a tool name to a LangGraph node name via the ToolRegistry.

    Returns ``None`` for unknown tools (caller should fall back).
    """
    if not tool:
        return None
    reg = _get_registry()
    if reg is not None:
        return reg.get_node_for_tool(tool)
    # Fallback when registry not available (should not happen in practice)
    _KNOWN = {
        "search": "query_rewriter", "python": "python",
        "filesystem": "tool_executor", "git": "tool_executor",
        "browser": "tool_executor", "database": "tool_executor",
        "mcp": "tool_executor", "composio": "tool_executor",
    }
    return _KNOWN.get(tool)


def _get_registry():
    """Return the ToolRegistry from the global executor instance."""
    if _executor_instance is not None:
        return _executor_instance.registry
    return None


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


def _make_planner_node(planner: PlannerAgent) -> object:
    """Factory: creates a planner node with cycle-count tracking to prevent infinite loops."""

    MAX_PLANNER_CYCLES = 5

    def _node(state: WorkflowState) -> dict[str, object]:
        count = int(state.get("_planner_cycle_count", 0)) + 1
        result = planner.run(state)
        # When the planner has been invoked many times without producing TODO
        # tasks, force goal_completed to break the planner↔reflector loop.
        task_queue = result.get("task_queue", []) or []
        has_todo = any(t.get("status", "") == "todo" for t in task_queue if isinstance(t, dict))
        plan = result.get("plan", {})
        if isinstance(plan, dict):
            plan_completed = plan.get("goal_completed") or plan.get("direct_answer")
        else:
            plan_completed = getattr(plan, "goal_completed", False) or getattr(plan, "direct_answer", False)
        if not plan_completed and not has_todo and count >= MAX_PLANNER_CYCLES:
            logger.warning("Planner: %d invocations without progress, forcing goal_completed", count)
            if isinstance(plan, dict):
                plan["goal_completed"] = True
                plan["direct_answer"] = True
                plan["reasoning"] = str(plan.get("reasoning", "")) + " (达到最大规划次数，强制结束)"
                result["plan"] = plan
            elif plan is not None:
                try:
                    plan.goal_completed = True
                    plan.direct_answer = True
                    plan.reasoning = f"{getattr(plan, 'reasoning', '')} (达到最大规划次数，强制结束)"
                    result["plan"] = plan
                except AttributeError:
                    pass
            result["_reflection_result"] = "done"
            result["_reflection_message"] = "达到最大规划次数，强制结束"
        result["_planner_cycle_count"] = count
        return result

    return _node


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
    """Factory: creates a query_rewriter node.

    Also marks the first TODO search/pending task as ``running`` so the
    frontend task tree updates correctly (the search node will mark it as
    ``done`` afterwards).
    """

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

        # Mark the first TODO search task as running
        queue = list(state.get("task_queue", []) or [])
        for t in queue:
            if isinstance(t, dict) and t.get("status", "todo") in ("todo", "queued"):
                t["status"] = "running"
                break

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
        return {"rewritten_query": rewritten, "task_queue": queue}

    return _node


def _make_search_node(search_agent: object) -> object:
    """Factory: wraps the search agent so it updates task queue status.

    Marks the first RUNNING search task as DONE after the search completes.
    """

    def _node(state: WorkflowState) -> dict[str, object]:
        result = search_agent.run(state)
        # Mark any running search tasks as done
        queue = list(state.get("task_queue", []) or [])
        for t in queue:
            if isinstance(t, dict) and t.get("status") == "running":
                t["status"] = "done"
        result["task_queue"] = queue
        return result

    return _node


def _make_tool_executor_node(executor: Executor) -> object:
    """Factory: creates a tool_executor node."""

    def _node(state: WorkflowState) -> dict[str, object]:
        queue = list(state.get("task_queue", []) or [])

        parallel_tasks = _select_parallel_filesystem_tasks(queue)
        if len(parallel_tasks) > 1:
            for task in parallel_tasks:
                task["status"] = "running"
            results = executor.execute_batch_parallel(parallel_tasks)
            result_by_index = {
                index: result for index, result in enumerate(results)
            }
            for index, task in enumerate(parallel_tasks):
                result = result_by_index.get(index)
                if result and result.success:
                    task["status"] = "done"
                elif result:
                    task["status"] = "failed"
                    task["error"] = result.error or "Unknown error"
                else:
                    task["status"] = "failed"
                    task["error"] = "No result from executor"
            logger.info(
                "Tool executor: parallel filesystem batch completed (%d task(s))",
                len(parallel_tasks),
            )
            return {
                "task_queue": queue,
                "tool_results": [r.to_dict() for r in results],
            }

        # Find highest priority TODO
        next_task = _find_highest_priority_todo(queue)
        if not next_task:
            logger.info("Tool executor: no TODO tasks in queue")
            return {"tool_results": []}

        # Mark running
        next_task["status"] = "running"

        result = None
        try:
            ctx = WorkflowContext(dict(state))
            result = executor.execute_task_dict(next_task, ctx=ctx)
        except Exception as exc:
            logger.error("Tool executor crashed: %s", exc)
            next_task["status"] = "failed"
            next_task["error"] = str(exc)
            return {
                "task_queue": queue,
                "tool_results": [{
                    "success": False,
                    "tool": next_task.get("tool", "?"),
                    "action": next_task.get("action", next_task.get("goal", "")),
                    "error": str(exc),
                    "message": f"Executor crashed: {exc}",
                    "result": None,
                }],
            }

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


def _select_parallel_filesystem_tasks(queue: list[dict]) -> list[dict]:
    """Select independent filesystem write tasks that can safely run together."""
    candidates = []
    seen_paths: set[str] = set()
    for task in queue:
        if not isinstance(task, dict) or task.get("status", "") != "todo":
            continue
        if task.get("tool") != "filesystem":
            continue
        inp = task.get("input", {}) or {}
        if not isinstance(inp, dict):
            continue
        action = str(inp.get("action", task.get("action", "")))
        if action not in ("write_file", "create_file", "append_file"):
            continue
        path = str(inp.get("path", "")).strip()
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        candidates.append(task)

    if len(candidates) < 2:
        return []
    max_priority = max(int(t.get("priority", 0) or 0) for t in candidates)
    selected = [t for t in candidates if int(t.get("priority", 0) or 0) == max_priority]
    return selected[:8]


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
_workflow_cache: Any = None


def _build_executor() -> Executor:
    """Create and configure the shared Executor instance.

    Uses ``ToolRegistry.auto_discover()`` to find all tool classes
    automatically.  Tools needing special configuration (e.g. workspace
    path) are passed as overrides.
    """
    global _executor_instance
    if _executor_instance is not None:
        return _executor_instance

    from pathlib import Path

    from agentflow.tools.filesystem_tool import FileSystemTool
    from agentflow.tools.docx_tool import DocxTool

    # FileSystemTool writes to outputs/ so the frontend API (which scans
    # that directory by default) can discover generated files.
    _outputs_dir = Path(__file__).resolve().parents[2] / "outputs"
    _outputs_dir.mkdir(exist_ok=True)

    ex = Executor()

    # Auto-discover and register all tools, with overrides for special config
    ex.registry.register_all_discovered(overrides={
        "filesystem": FileSystemTool(workspace=str(_outputs_dir)),
        "docx": DocxTool(workspace=str(_outputs_dir)),
    })

    # Warn about interface-only placeholders
    for name, tool in sorted(ex.registry._tools.items()):
        meta = tool.metadata()
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
