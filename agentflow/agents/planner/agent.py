"""Planner Agent — Dynamic Task Queue Planner.

The Planner is the **task generator** in the Dynamic Task Queue system:

  - Receives a **Goal** and **Task Queue** + **Workspace State**
  - Generates only **3-5 tasks** per invocation, not the full project
  - On first invocation (empty queue), initializes from a Project Template
  - Subsequent invocations add/update tasks based on current state
  - Falls back gracefully (templates -> LLM -> minimal)

Architecture::

    GoalAnalyzer -> CapabilityAnalyzer -> Knowledge -> Planner
                                                          |
    Reflector <- Executor <- Task Queue <- ContextBuilder |
       |                                                   |
       +-- goal_completed -> answer -> memory -> END       |
       +-- more tasks -> Executor (one at a time)          |
       +-- need_replan -> Planner                          |
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentflow.agents.base import AgentProtocol
from agentflow.agents.planner.capability import resolve as resolve_capability
from agentflow.agents.planner.prompt import build_fc_planner_prompt, build_planner_prompt
from agentflow.agents.planner.schemas import get_tool_schemas, parse_function_name
from agentflow.agents.planner.task_queue import TaskQueue
from agentflow.agents.planner.templates import (
    extract_project_name,
    get_existing_files,
    get_initial_tasks,
    match_template,
)
from agentflow.graph.context_builder import ContextBuilder
from agentflow.graph.plan import Plan
from agentflow.graph.task import Task, TaskStatus
from agentflow.services.llm_service import LLMResponse, ToolCall, get_llm_service
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("planner")


class PlannerAgent(AgentProtocol):
    """Dynamic Task Queue planner: generates 3-5 tasks per invocation."""

    def __init__(self) -> None:
        self._llm = get_llm_service()

    @safe_run
    def run(self, state: dict) -> dict:
        """Generate the next batch of tasks from goal + workspace + task queue.

        Sets::
          state["plan"]        -> Plan object (with goal_completed, tasks)
          state["task_queue"]  -> merged task queue (existing + new tasks)
          state["workflow"]    -> list[str] (backward compat)
        """
        # -- Extract goal -------------------------------------------------
        goal_analysis = state.get("goal_analysis", {})
        if isinstance(goal_analysis, dict):
            goal = goal_analysis.get("goal", state.get("question", ""))
            goal_type = goal_analysis.get("goal_type", "other")
        else:
            goal = state.get("question", "")
            goal_type = "other"

        # -- Non-project: use direct_answer flow (backward compat) --------
        if goal_type != "project":
            return self._handle_non_project(state, goal, goal_type)

        # -- Handle project task queue ------------------------------------
        return self._handle_project(state, goal, goal_type)

    # ------------------------------------------------------------------
    # Project flow (Dynamic Task Queue)
    # ------------------------------------------------------------------

    def _handle_project(self, state: dict, goal: str, goal_type: str) -> dict:
        """Handle project-type goals with the Dynamic Task Queue."""
        current_queue = TaskQueue.from_dict_list(
            state.get("task_queue", []) or []
        )

        # If task queue is empty, initialize from template
        if current_queue.is_empty:
            plan = self._initialize_from_template(goal)
            if plan is None:
                plan = self._llm_generate_tasks(
                    goal, goal_type, state, replan_context=""
                )
        else:
            # Generate 3-5 additional tasks based on current state
            plan = self._generate_more_tasks(
                goal, goal_type, state, current_queue
            )

        # Merge plan tasks into the queue
        merged = self._merge_into_queue(current_queue, plan)

        state["plan"] = plan
        state["task_queue"] = merged.to_dict_list()
        state["category"] = goal_type
        state["workflow"] = _plan_to_workflow(plan, goal_type)

        if plan.goal_completed:
            logger.info("Plan: goal_completed")
        else:
            logger.info(
                "Plan: added %d tasks, queue has %d TODO",
                len(plan.tasks), merged.todo_count,
            )

        return state

    def _initialize_from_template(self, goal: str) -> Plan | None:
        """Initialize the task queue from a matching project template."""
        template = match_template(goal, "project")
        if not template:
            logger.info("No template matched for goal, using LLM")
            return None

        project_name = extract_project_name(goal)
        project_path = Path(project_name) if project_name else None
        if project_path and project_path.exists() and project_path.is_dir():
            existing = get_existing_files(str(project_path))
        else:
            existing = set()

        tasks = get_initial_tasks(template, goal, existing)
        if not tasks:
            return Plan(
                goal=goal, category="project",
                tasks=[], goal_completed=True,
                reasoning="没有需要初始化的任务",
            )

        logger.info(
            "Template '%s': initialized %d tasks (%d TODO, %d DONE)",
            template.get("name", "?"), len(tasks),
            sum(1 for t in tasks if t.status == TaskStatus.TODO),
            sum(1 for t in tasks if t.status == TaskStatus.DONE),
        )

        return Plan(
            goal=goal, category="project",
            tasks=tasks, goal_completed=False,
            reasoning=f"Template {template['id']}: initialized {len(tasks)} tasks",
        )

    def _generate_more_tasks(
        self,
        goal: str,
        goal_type: str,
        state: dict,
        current_queue: TaskQueue,
    ) -> Plan:
        """Generate 3-5 more tasks based on current workspace and queue."""
        # Try FC planner first
        builder = ContextBuilder(state)
        context_str = builder.format_planner_prompt()
        replan_msg = str(state.get("_reflection_message", ""))
        replan_count = int(state.get("_replan_count", 0))
        replan_context = replan_msg if replan_count > 0 else ""

        plan = self._fc_plan(goal, goal_type, context_str, replan_context)

        if plan is None or (not plan.tasks and not plan.goal_completed):
            logger.info("FC planner failed, trying JSON-based planner")
            plan = self._llm_plan(goal, goal_type, context_str, replan_context)

        if plan is None or (not plan.tasks and not plan.goal_completed):
            logger.info("LLM planner failed, returning empty plan")
            plan = Plan(
                goal=goal, category=goal_type,
                tasks=[], goal_completed=False,
                reasoning="无法生成新任务",
            )

        return plan

    def _merge_into_queue(self, queue: TaskQueue, plan: Plan) -> TaskQueue:
        """Merge Plan tasks into the TaskQueue."""
        for task in plan.tasks:
            queue.add(task)
        return queue

    # ------------------------------------------------------------------
    # Non-project flow (backward compat)
    # ------------------------------------------------------------------

    def _handle_non_project(self, state: dict, goal: str, goal_type: str) -> dict:
        """Handle non-project goals using the existing flow."""
        builder = ContextBuilder(state)
        context_str = builder.format_planner_prompt()
        replan_msg = str(state.get("_reflection_message", ""))
        replan_count = int(state.get("_replan_count", 0))
        replan_context = replan_msg if replan_count > 0 else ""

        plan = self._fc_plan(goal, goal_type, context_str, replan_context)

        if plan is None or (not plan.tasks and not plan.goal_completed):
            logger.info("FC planner failed, trying JSON-based planner")
            plan = self._llm_plan(goal, goal_type, context_str, replan_context)

        if plan is None or (not plan.tasks and not plan.goal_completed):
            logger.info("LLM planner failed, using direct answer")
            plan = Plan(
                goal=goal, category=goal_type,
                tasks=[], direct_answer=True, goal_completed=True,
                reasoning=f"无法生成计划（goal_type={goal_type}），直接回答",
            )

        self._resolve_capabilities(plan)

        if plan.goal_completed:
            logger.info("Plan: goal_completed (goal_type=%s)", goal_type)
        else:
            logger.info(
                "Plan: %d tasks (goal_type=%s)",
                plan.step_count, goal_type,
            )

        state["plan"] = plan
        state["category"] = goal_type
        state["workflow"] = _plan_to_workflow(plan, goal_type)
        return state

    # ------------------------------------------------------------------
    # LLM-based task generation (shared by both flows)
    # ------------------------------------------------------------------

    def _llm_generate_tasks(
        self,
        goal: str,
        goal_type: str,
        state: dict,
        replan_context: str = "",
    ) -> Plan:
        """Generate tasks via LLM when template initialization is not possible."""
        builder = ContextBuilder(state)
        context_str = builder.format_planner_prompt()

        plan = self._llm_plan(goal, goal_type, context_str, replan_context)
        if plan is None or (not plan.tasks and not plan.goal_completed):
            return Plan(
                goal=goal, category=goal_type,
                tasks=[], direct_answer=True, goal_completed=True,
                reasoning=f"无可用模板（goal_type={goal_type}），直接回答",
            )
        return plan

    # ------------------------------------------------------------------
    # Function-calling planning
    # ------------------------------------------------------------------

    def _fc_plan(
        self,
        goal: str,
        goal_type: str,
        context_str: str = "",
        replan_context: str = "",
    ) -> Plan | None:
        """Try to generate tasks via function calling."""
        if not goal:
            return None

        try:
            messages = build_fc_planner_prompt(
                goal=goal, goal_type=goal_type,
                context_str=context_str, replan_context=replan_context,
            )
        except Exception as exc:
            logger.warning("FC planner prompt build failed: %s", exc)
            return None

        tools = get_tool_schemas()

        try:
            resp: LLMResponse = self._llm.complete_with_tools(
                messages=messages, tools=tools, tool_choice="auto",
            )
        except Exception as exc:
            logger.warning("FC planner LLM call failed: %s", exc)
            return None

        if resp.tool_calls:
            return self._build_plan_from_tool_calls(resp.tool_calls, resp.content)

        content = resp.content.strip()
        if content:
            parsed = self._parse_json(content)
            if parsed:
                return self._build_plan_from_json(parsed, goal, goal_type)
            return Plan(
                goal=goal, category=goal_type,
                tasks=[], direct_answer=True,
                goal_completed=True,
                reasoning=content[:500],
            )
        return None

    # ------------------------------------------------------------------
    # JSON-based LLM planning (fallback)
    # ------------------------------------------------------------------

    def _llm_plan(
        self,
        goal: str,
        goal_type: str,
        context_str: str = "",
        replan_context: str = "",
    ) -> Plan | None:
        """Try to generate tasks via LLM call with JSON output."""
        if not goal:
            return None

        try:
            messages = build_planner_prompt(
                goal=goal, goal_type=goal_type,
                context_str=context_str, replan_context=replan_context,
            )
            raw = self._llm.complete(messages=messages)
        except Exception as exc:
            logger.warning("LLM planner call failed: %s", exc)
            return None

        if not raw or not raw.strip():
            return None

        parsed = self._parse_json(raw)
        if parsed is None:
            return None

        try:
            return self._build_plan_from_json(parsed, goal, goal_type)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("LLM planner JSON validation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Build Plan from tool calls
    # ------------------------------------------------------------------

    @staticmethod
    def _build_plan_from_tool_calls(
        tool_calls: list[ToolCall], reasoning: str,
    ) -> Plan:
        """Convert ToolCall objects into a Plan with Task objects."""
        tasks: list[Task] = []
        for tc in tool_calls:
            tool, action = parse_function_name(tc.name)
            inp: dict = {}
            if tc.arguments and tc.arguments.strip():
                try:
                    inp = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse arguments for %s", tc.name)

            # Embed action for executor dispatch
            if action and "action" not in inp:
                inp["action"] = action

            capability = f"{tool}.{action}" if action else tool
            tasks.append(Task(
                task_id=tc.name.replace("__", "_"),
                title=f"执行 {tc.name}",
                priority=80,
                goal=action or f"执行 {tc.name}",
                capability=capability,
                tool=tool,
                input=inp,
                agent="planner",
            ))

        return Plan(
            goal="", category="",
            tasks=tasks, direct_answer=False,
            goal_completed=False,
            reasoning=reasoning[:1000] if reasoning else f"执行 {len(tasks)} 个任务",
        )

    # ------------------------------------------------------------------
    # Build Plan from JSON
    # ------------------------------------------------------------------

    @staticmethod
    def _build_plan_from_json(
        data: dict[str, Any], goal: str, goal_type: str,
    ) -> Plan:
        """Convert a validated JSON dict into a Plan."""
        goal_completed = bool(data.get("goal_completed", False))
        tasks_raw = data.get("tasks", [])
        reasoning = str(data.get("reasoning", ""))

        # Accept legacy "current_stage" field silently (ignore it)
        _ = data.get("current_stage", "")

        if goal_completed or not tasks_raw:
            direct = bool(data.get("direct_answer", not tasks_raw))
            return Plan(
                goal=goal, category=goal_type,
                tasks=[], direct_answer=direct,
                goal_completed=goal_completed or direct,
                reasoning=reasoning or "目标已完成",
            )

        if not isinstance(tasks_raw, list):
            raise ValueError("'tasks' must be a list")

        tasks: list[Task] = []
        for i, item in enumerate(tasks_raw):
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict task at index %d", i)
                continue

            task_id = str(item.get("task_id", f"task_{i}") or f"task_{i}")
            title = str(item.get("title", item.get("goal", task_id)))
            priority = int(item.get("priority", 50))
            tool = str(item.get("tool", "")).strip()
            action = str(item.get("action", "")).strip()
            tgoal = str(item.get("goal", "")).strip()
            inp = item.get("input", {})
            if not isinstance(inp, dict):
                inp = {}

            # Legacy format: capability field instead of tool
            capability_raw = str(item.get("capability", "")).strip()
            if not tool and capability_raw:
                tool = resolve_capability(capability_raw) or ""
                capability = capability_raw
            elif tool:
                capability = f"{tool}.{action}" if action else tool
            else:
                logger.warning("Task %d has no 'tool' or 'capability'; skipping", i)
                continue

            # Embed action into input for Executor
            if action and "action" not in inp:
                inp["action"] = action

            tasks.append(Task(
                task_id=task_id,
                title=title,
                priority=priority,
                goal=tgoal or f"执行 {tool}.{action}" if action else f"执行 {tool}",
                capability=capability,
                tool=tool,
                agent="planner",
                input=inp,
            ))

        return Plan(
            goal=goal, category=goal_type,
            tasks=tasks, direct_answer=False,
            goal_completed=goal_completed,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Capability resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_capabilities(plan: Plan) -> None:
        """Resolve each task's capability to a concrete tool name."""
        for task in plan.tasks:
            if not task.capability:
                continue
            tool = resolve_capability(task.capability)
            if tool:
                task.tool = tool

    # ------------------------------------------------------------------
    # JSON parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        """Extract a JSON object from LLM output."""
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for marker in ("```json", "```JSON", "```"):
            start = text.find(marker)
            if start == -1:
                continue
            content = text[start + len(marker):]
            end = content.rfind("```")
            if end != -1:
                content = content[:end]
            content = content.strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                continue
        return None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _plan_to_workflow(plan: Plan, goal_type: str) -> list[str]:
    """Derive a LangGraph node-name list from a Plan (backward compat)."""
    nodes = ["goal_analyzer"]

    if goal_type not in ("identity", "search"):
        nodes.append("knowledge")

    if plan.goal_completed or plan.direct_answer:
        pass
    else:
        for t in plan.tasks:
            node = _TOOL_TO_NODE.get(t.tool or "")
            if node and node not in nodes:
                nodes.append(node)

    if "answer" not in nodes:
        nodes.append("answer")
    if "memory" not in nodes:
        nodes.append("memory")
    return nodes


# Map tool names -> LangGraph node names
_TOOL_TO_NODE: dict[str, str] = {
    "search": "search",
    "python": "python",
    "filesystem": "tool_executor",
    "git": "tool_executor",
    "browser": "tool_executor",
    "database": "tool_executor",
    "mcp": "tool_executor",
    "composio": "tool_executor",
}
