"""Planner Agent — Dynamic Task Queue Planner.

The Planner is the **task generator** in the Dynamic Task Queue system:

  - Receives a **Goal** and **Task Queue** + **Workspace State**
  - Generates only **3-5 tasks** per invocation, not the full project
  - On first invocation (empty queue), initializes from a Project Template
  - Subsequent invocations add/update tasks based on current state
  - Falls back gracefully (templates -> LLM -> minimal)

Architecture::

    GoalAnalyzer -> Knowledge -> Planner
                                                          |
    Reflector <- Executor <- Task Queue <- ContextBuilder |
       |                                                   |
       +-- goal_completed -> answer -> memory -> END       |
       +-- more tasks -> Executor (one at a time)          |
       +-- need_replan -> Planner                          |
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from agentflow.agents.base import AgentProtocol
from agentflow.agents.planner.capability import resolve as resolve_capability
from agentflow.config.settings import settings
from agentflow.utils.errors import record_error as _record_error
from agentflow.agents.planner.prompt import (
    build_codegen_prompt,
    build_fc_planner_prompt,
    build_planner_prompt,
    build_project_design_prompt,
)
from agentflow.agents.planner.schemas import get_tool_schemas, parse_function_name
from agentflow.agents.planner.task_queue import TaskQueue
from agentflow.agents.planner.templates import (
    extract_project_name,
    get_existing_files,
    get_initial_tasks,
    match_template,
)
from agentflow.blueprints import BlueprintLoader, FileSpec, ProjectConfig, ProjectConfigurator
from agentflow.graph.context_builder import ContextBuilder
from agentflow.graph.plan import Plan
from agentflow.graph.state_utils import get_goal, get_goal_type
from agentflow.graph.task import Task, TaskStatus
from agentflow.services.llm_service import LLMResponse, ToolCall, get_llm_service
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("planner")

# Tool actions that only read/inspect — never useful as queued tasks.
# The LLM sometimes generates these despite prompt instructions.
_NON_CREATIVE_ACTIONS = frozenset({
    "read_file", "list_directory", "tree", "exists",
    "search_file", "search_content", "read", "list", "search",
    "get", "find", "stat", "glob", "ls", "cat", "head", "tail",
})


class PlannerAgent(AgentProtocol):
    """Dynamic Task Queue planner: generates 3-5 tasks per invocation."""

    def __init__(self, registry=None) -> None:
        self._llm = get_llm_service()
        self.registry = registry  # ToolRegistry — dynamic source of truth

    @safe_run
    def run(self, state: dict) -> dict:
        """Generate the next batch of tasks from goal + workspace + task queue.

        Sets::
          state["plan"]        -> Plan object (with goal_completed, tasks)
          state["task_queue"]  -> merged task queue (existing + new tasks)
          state["workflow"]    -> list[str] (backward compat)
        """
        # -- Extract goal -------------------------------------------------
        goal = get_goal(state)
        goal_type = get_goal_type(state)

        _DIRECT_ANSWER_TYPES = frozenset({"other", "translation", "editing", "question"})
        if goal_type in _DIRECT_ANSWER_TYPES:
            logger.info("Goal type '%s': non-actionable, using direct answer", goal_type)
            plan = Plan(
                goal=goal, category=goal_type,
                tasks=[], direct_answer=True, goal_completed=True,
                reasoning=f"直接回答模式（goal_type={goal_type}）",
            )
            state["plan"] = plan
            state["category"] = goal_type
            state["task_queue"] = []
            state["workflow"] = _plan_to_workflow(plan, goal_type, self.registry)
            return state

        # -- Degraded mode: skip LLM-dependent planning -------------------
        _degraded: set = state.get("_degraded", set()) or set()
        if "_planner" in _degraded or state.get("_llm_error"):
            logger.warning("Degraded mode: skipping LLM planning, using direct answer")
            plan = Plan(
                goal=goal, category=goal_type,
                tasks=[], direct_answer=True, goal_completed=False,
                reasoning="系统运行在受限模式，无法进行完整规划",
            )
            state["plan"] = plan
            state["category"] = goal_type
            state["workflow"] = _plan_to_workflow(plan, goal_type, self.registry)
            return state

        # -- Non-project: use direct_answer flow (backward compat) --------
        if goal_type != "project":
            return self._handle_non_project(state, goal, goal_type)

        # -- Handle project task queue ------------------------------------
        if getattr(self._llm, "is_mock", False):
            return self._handle_mock_project(state, goal, goal_type)
        return self._handle_project(state, goal, goal_type)

    # ------------------------------------------------------------------
    # Project flow (Dynamic Task Queue)
    # ------------------------------------------------------------------

    def _handle_project(self, state: dict, goal: str, goal_type: str) -> dict:
        """Handle project-type goals with the Dynamic Task Queue."""
        current_queue = TaskQueue.from_dict_list(
            state.get("task_queue", []) or []
        )

        # If task queue is empty, initialize from blueprint or template
        if current_queue.is_empty:
            # 1) Try Blueprint (best-practice skeleton)
            plan = self._initialize_from_blueprint(goal, state)
            if plan is None:
                # 2) Fallback to legacy template
                plan = self._initialize_from_template(goal)
            if plan is None:
                # 3) LLM-generated tasks
                plan = self._llm_generate_tasks(
                    goal, goal_type, state, replan_context=""
                )
        else:
            # Generate 3-5 additional tasks based on current state
            plan = self._generate_more_tasks(
                goal, goal_type, state, current_queue
            )

        # Design-first: one shared project design for all codegen calls,
        # so every generated file follows the same data model / API list.
        project_brief = self._generate_project_design(goal, plan)
        if not project_brief:
            project_brief = self._build_project_brief(plan, goal)

        # Fill code content for tasks that need it (CodeGenerator)
        codegen_errors = self._fill_code_content(plan, project_brief)
        for err in codegen_errors:
            _record_error(state, "planner", "codegen_invalid", err)

        # Merge plan tasks into the queue
        merged = self._merge_into_queue(current_queue, plan)

        state["plan"] = plan
        state["task_queue"] = merged.to_dict_list()
        state["category"] = goal_type
        state["workflow"] = _plan_to_workflow(plan, goal_type, self.registry)

        # Detect degraded fallback — LLM was unavailable
        if plan.direct_answer and not plan.goal_completed and not plan.tasks:
            state.setdefault("_degraded", set()).add("planner")
            _record_error(state, "planner", "llm_unavailable",
                          "LLM planner unavailable — both function-calling and JSON "
                          "planning failed (timeout or network error)")
            logger.warning("Plan: degraded fallback (LLM unavailable)")

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

    # ------------------------------------------------------------------
    # Blueprint-based initialisation (replaces legacy templates)
    # ------------------------------------------------------------------

    _blueprint_loader: BlueprintLoader | None = None

    @classmethod
    def _get_blueprint_loader(cls) -> BlueprintLoader:
        if cls._blueprint_loader is None:
            cls._blueprint_loader = BlueprintLoader()
        return cls._blueprint_loader

    def _initialize_from_blueprint(
        self, goal: str, state: dict,
    ) -> Plan | None:
        """Try to initialise the task queue from a best-practice Blueprint.

        Flow::

            BlueprintLoader.match(goal)  → Blueprint | None
            ProjectConfigurator          → ProjectConfig
            BlueprintLoader.render()     → list[FileSpec]
            _blueprint_specs_to_tasks()  → list[Task]  ← you are here
        """
        goal_type = "project"
        loader = self._get_blueprint_loader()
        blueprint = loader.match(goal, goal_type)
        if blueprint is None:
            return None

        logger.info("Blueprint matched: '%s' (%s)", blueprint.id, blueprint.name)

        # ── Derive project config ────────────────────────────────────
        try:
            config = ProjectConfigurator.from_goal(goal)
        except Exception as exc:
            logger.warning("Blueprint: config derivation failed (%s), falling back", exc)
            return None

        # Try LLM-enhanced variable filling (best-effort)
        try:
            config = ProjectConfigurator.with_llm(goal, blueprint, config)
        except Exception as exc:
            logger.warning("Blueprint: LLM config filling failed (%s), using rule-based", exc)

        # ── Scan existing files ──────────────────────────────────────
        existing: set[str] = set()
        project_path = Path(config.project_name) if config.project_name else None
        if project_path and project_path.exists() and project_path.is_dir():
            existing = get_existing_files(str(project_path))

        # ── Render ───────────────────────────────────────────────────
        specs = loader.render(blueprint, config, existing_files=existing)

        # Convert specs → tasks
        tasks = self._blueprint_specs_to_tasks(specs, config)
        if not tasks:
            return Plan(
                goal=goal, category="project",
                tasks=[], goal_completed=True,
                reasoning=f"Blueprint '{blueprint.id}': no files to create (all exist)",
            )

        logger.info(
            "Blueprint '%s': %d file(s) (%d create, %d modify, %d skip)",
            blueprint.id, len(specs),
            sum(1 for s in specs if s.type == "create"),
            sum(1 for s in specs if s.type == "modify"),
            sum(1 for s in specs if s.type == "skip"),
        )

        return Plan(
            goal=goal, category="project",
            tasks=tasks, goal_completed=False,
            reasoning=f"Blueprint '{blueprint.id}': {len(tasks)} file(s) to create",
        )

    @staticmethod
    def _blueprint_specs_to_tasks(
        specs: list[FileSpec],
        config: ProjectConfig,
    ) -> list[Task]:
        """Convert rendered FileSpecs into Executor Task objects.

        - ``create`` → ``filesystem.create_file`` with rendered content
        - ``modify`` → ``filesystem.edit_file`` with rendered content
        - ``skip``   → omitted
        - ``reference`` → omitted
        """
        tasks: list[Task] = []
        created_dirs: set[str] = set()
        _UNUSED = object()

        for spec in specs:
            if spec.type in ("skip", "reference"):
                continue

            path = spec.path
            parent = str(Path(path).parent) if "/" in path else None

            # Ensure parent directory exists (mkdir -p)
            if parent and parent not in created_dirs and parent != ".":
                tasks.append(Task(
                    task_id=f"mkdir_{parent.replace('/', '_').replace('.', '')}",
                    title=f"创建目录 {parent}",
                    priority=100,
                    tool="filesystem",
                    goal=f"创建 {parent}/",
                    input={"action": "mkdir", "path": parent},
                    status=TaskStatus.TODO,
                ))
                created_dirs.add(parent)

            task_id = path.replace("/", "_").replace(".", "_").replace("-", "_")
            action = "edit_file" if spec.type == "modify" else "write_file"

            tasks.append(Task(
                task_id=task_id,
                title=spec.description or f"创建 {path}",
                priority=80,
                tool="filesystem",
                goal=f"{action}: {path}",
                input={
                    "action": action,
                    "path": path,
                    "content": spec.content_template,
                },
                status=TaskStatus.TODO,
            ))

        # Sort: mkdir tasks first, then write tasks
        tasks.sort(key=lambda t: (0 if "mkdir" in t.task_id else 1, -t.priority))
        return tasks

    def _generate_more_tasks(
        self,
        goal: str,
        goal_type: str,
        state: dict,
        current_queue: TaskQueue,
    ) -> Plan:
        """Generate 3-5 more tasks based on current workspace and queue.

        Uses JSON-based planning first (not FC) because JSON can output
        multiple file-creation tasks in a single response, while FC
        planners often return only 1 tool call per invocation.
        """
        builder = ContextBuilder(state)
        context_str = builder.format_planner_prompt()
        replan_msg = str(state.get("_reflection_message", ""))
        replan_count = int(state.get("_replan_count", 0))
        replan_context = replan_msg if replan_count > 0 else ""

        # JSON-based planning first — outputs multiple tasks per response
        plan = self._llm_plan(goal, goal_type, context_str, replan_context)

        if plan is None or (not plan.tasks and not plan.goal_completed):
            logger.info("JSON planner failed, trying FC planner")
            plan = self._fc_plan(goal, goal_type, context_str, replan_context)

        if plan is None or (not plan.tasks and not plan.goal_completed):
            logger.info("All planners failed, returning empty plan")
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

    def _handle_mock_project(self, state: dict, goal: str, goal_type: str) -> dict:
        """Handle project-type goals under Mock LLM — skip blueprint/template."""
        current_queue = TaskQueue.from_dict_list(
            state.get("task_queue", []) or []
        )

        if current_queue.is_empty:
            plan = self._llm_generate_tasks(goal, goal_type, state, replan_context="")
        else:
            plan = self._generate_more_tasks(goal, goal_type, state, current_queue)

        # Fill code content for tasks that need it (CodeGenerator); include
        # the deterministic project brief so non-project code tasks still get
        # project context.
        project_brief = self._build_project_brief(plan, goal)
        self._fill_code_content(plan, project_brief)

        merged = self._merge_into_queue(current_queue, plan)
        state["plan"] = plan
        state["task_queue"] = merged.to_dict_list()
        state["category"] = goal_type
        state["workflow"] = _plan_to_workflow(plan, goal_type, self.registry)

        if plan.direct_answer and not plan.goal_completed and not plan.tasks:
            state.setdefault("_degraded", set()).add("planner")
            _record_error(state, "planner", "llm_unavailable",
                          "LLM planner unavailable (mock degraded)")

        if plan.goal_completed:
            logger.info("Plan (mock): goal_completed")
        else:
            logger.info("Plan (mock): added %d tasks, queue has %d TODO", len(plan.tasks), merged.todo_count)

        return state

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

        # Fill code content for tasks that need it (CodeGenerator)
        self._fill_code_content(plan)

        if plan.goal_completed:
            logger.info("Plan: goal_completed (goal_type=%s)", goal_type)
        else:
            logger.info(
                "Plan: %d tasks (goal_type=%s)",
                plan.step_count, goal_type,
            )

        state["plan"] = plan
        state["category"] = goal_type
        state["workflow"] = _plan_to_workflow(plan, goal_type, self.registry)

        # Serialize plan tasks into the task queue so the executor can run them
        state["task_queue"] = [t.to_dict() for t in plan.tasks] if plan.tasks else []

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
        """Generate tasks via LLM when template initialization is not possible.

        Tries function-calling first (more reliable with tool definitions),
        then falls back to JSON-based planning.
        """
        builder = ContextBuilder(state)
        context_str = builder.format_planner_prompt()

        # 1) Try function-calling planning first (more reliable)
        plan = self._fc_plan(goal, goal_type, context_str, replan_context)
        if plan is not None:
            if plan.tasks:
                return plan
            if plan.goal_completed:
                logger.info("FC planner returned goal_completed, but no tasks — trying JSON fallback")

        # 2) Fallback to JSON-based planning
        plan = self._llm_plan(goal, goal_type, context_str, replan_context)
        if plan is not None and (plan.tasks or plan.goal_completed):
            return plan

        # 3) Final fallback: direct answer, but mark as NOT goal_completed
        # so the answer agent knows this is a degraded response.
        logger.warning(
            "Planner: both FC and JSON planning failed for goal_type=%s — "
            "falling back to degraded answer path", goal_type,
        )
        return Plan(
            goal=goal, category=goal_type,
            tasks=[], direct_answer=True, goal_completed=False,
            reasoning=f"LLM 规划不可用（goal_type={goal_type}），使用降级回答模式",
        )

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
                registry=self.registry,
            )
        except Exception as exc:
            logger.warning("FC planner prompt build failed: %s", exc)
            return None

        tools = get_tool_schemas(self.registry) if self.registry else []

        try:
            resp: LLMResponse = self._llm.complete_with_tools(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=settings.planner_max_tokens,
            )
        except Exception as exc:
            logger.warning("FC planner LLM call failed: %s", exc)
            return None

        if resp.tool_calls:
            logger.info("FC planner: got %d tool calls", len(resp.tool_calls))
            for tc in resp.tool_calls:
                logger.info("  Tool: %s args: %s", tc.name, tc.arguments[:200])
            return self._build_plan_from_tool_calls(resp.tool_calls, resp.content, goal, goal_type)

        # LLM returned a degraded fallback (e.g. timeout, network error)
        if resp.degraded:
            logger.warning("FC planner returned degraded response — falling back to JSON planner")
            return None

        content = resp.content.strip()
        if content:
            parsed = self._parse_json(content)
            if parsed:
                return self._build_plan_from_json(parsed, goal, goal_type)
            logger.warning("FC planner returned non-JSON content, not goal_completed: %s", content[:200])
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
                registry=self.registry,
            )
            raw = self._llm.complete(
                messages=messages,
                node_name="planner",
                max_tokens=settings.planner_max_tokens,
            )
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
        goal: str = "", category: str = "",
    ) -> Plan:
        """Convert ToolCall objects into a Plan with Task objects."""
        tasks: list[Task] = []
        for i, tc in enumerate(tool_calls):
            tool, action = parse_function_name(tc.name)

            # Safety filter: skip read-only / inspection tool calls.
            # The LLM sometimes generates these despite prompt instructions;
            # they are never useful as queued tasks.
            if action in _NON_CREATIVE_ACTIONS:
                logger.info(
                    "Skipping non-creative tool call %s (action=%s)", tc.name, action
                )
                continue

            inp = _parse_tool_arguments(tc.arguments, tc.name)

            # Write_file with no path = args parsing failed (common with DeepSeek FC
            # when large code content corrupts the JSON).  Try to recover path,
            # content, and code_prompt from the raw args.
            if action in ("write_file", "create_file") and not inp.get("path"):
                path = _extract_path_from_args(tc.arguments)
                if not path:
                    logger.warning(
                        "Skipping %s task — no valid path after parsing args", tc.name
                    )
                    continue
                content = _extract_content_from_args(tc.arguments)
                code_prompt = _extract_code_prompt_from_args(tc.arguments)
                if content and len(content) > 10:
                    tasks.append(Task(
                        task_id=f"{tc.name.replace('__', '_')}_{i}",
                        title=f"创建 {path}",
                        priority=80,
                        tool="filesystem",
                        goal=f"write_file: {path}",
                        input={"action": "write_file", "path": path, "content": content},
                        agent="planner",
                    ))
                    logger.info(
                        "Recovered path='%s' + content (%d chars) from malformed FC args",
                        path, len(content),
                    )
                elif code_prompt:
                    # code_prompt present but content empty → CodeGenerator fills it later
                    tasks.append(Task(
                        task_id=f"{tc.name.replace('__', '_')}_{i}",
                        title=f"创建 {path}",
                        priority=80,
                        tool="filesystem",
                        goal=f"write_file: {path}",
                        input={"action": "write_file", "path": path, "code_prompt": code_prompt},
                        agent="planner",
                    ))
                    logger.info(
                        "Recovered path='%s' + code_prompt (%d chars) from malformed FC args",
                        path, len(code_prompt),
                    )
                else:
                    parent = str(Path(path).parent)
                    if parent and parent != ".":
                        mkdir_path = parent
                    else:
                        mkdir_path = Path(path).stem
                    mkdir_id = f"mkdir_{mkdir_path.replace('/', '_').replace('.', '')}"
                    tasks.append(Task(
                        task_id=mkdir_id,
                        title=f"创建目录 {mkdir_path}",
                        priority=100,
                        tool="filesystem",
                        goal=f"创建 {mkdir_path}/",
                        input={"action": "mkdir", "path": mkdir_path},
                        agent="planner",
                    ))
                    logger.info(
                        "Recovered path='%s' (no content or code_prompt) → mkdir '%s'",
                        path, mkdir_path,
                    )
                continue

            # Embed action for executor dispatch
            if action and "action" not in inp:
                inp["action"] = action

            capability = f"{tool}.{action}" if action else tool
            tasks.append(Task(
                task_id=f"{tc.name.replace('__', '_')}_{i}",
                title=f"执行 {tc.name}",
                priority=80,
                goal=action or f"执行 {tc.name}",
                capability=capability,
                tool=tool,
                input=inp,
                agent="planner",
            ))

        if len(tasks) > 8:
            original = len(tasks)
            tasks.sort(key=lambda t: -t.priority)
            tasks = tasks[:8]
            reasoning = f"{reasoning} (trimmed from {original} to 8 tasks)"

        return Plan(
            goal=goal, category=category,
            tasks=tasks, direct_answer=False,
            goal_completed=False,
            reasoning=reasoning[:1000] if reasoning else f"执行 {len(tasks)} 个任务",
        )

    # ------------------------------------------------------------------
    # Build Plan from JSON
    # ------------------------------------------------------------------

    def _build_plan_from_json(
        self, data: dict[str, Any], goal: str, goal_type: str,
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
            priority = int(item.get("priority") or 50)
            tool = str(item.get("tool", "")).strip()
            action = str(item.get("action", "")).strip()
            tgoal = str(item.get("goal", "")).strip()
            inp = item.get("input", {})
            if not isinstance(inp, dict):
                inp = {}

            # Legacy format: capability field instead of tool
            capability_raw = str(item.get("capability", "")).strip()
            if not tool and capability_raw:
                tool = resolve_capability(capability_raw, self.registry) or ""
                capability = capability_raw
            elif tool:
                capability = f"{tool}.{action}" if action else tool
            else:
                logger.warning("Task %d has no 'tool' or 'capability'; skipping", i)
                continue

            # Embed action into input for Executor
            if action and "action" not in inp:
                inp["action"] = action

            # Unwrap code from markdown code blocks in content field.
            # LLMs are instructed to wrap code in ``` fences so that
            # special characters don't corrupt the JSON.  We strip the
            # fences here so the file on disk contains just the code.
            if "content" in inp and isinstance(inp["content"], str):
                inp["content"] = _extract_code_from_markdown(inp["content"])

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

        # Cap tasks per invocation to prevent blueprint over-expansion
        if len(tasks) > 8:
            original = len(tasks)
            tasks.sort(key=lambda t: -t.priority)
            tasks = tasks[:8]
            reasoning += f" (trimmed from {original} to 8 tasks)"

        return Plan(
            goal=goal, category=goal_type,
            tasks=tasks, direct_answer=False,
            goal_completed=goal_completed,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # CodeGenerator — plain-text LLM call, zero JSON
    # ------------------------------------------------------------------

    def _call_codegen(
        self,
        code_prompt: str,
        language: str = "",
        project_brief: str = "",
    ) -> str:
        """Generate code via a plain-text LLM completion (no JSON, no tools).

        *project_brief* carries the shared project design so every generated
        file stays consistent with the rest of the project.
        """
        if not code_prompt.strip():
            return ""
        messages = build_codegen_prompt(code_prompt, language, project_brief)
        try:
            raw = self._llm.complete(
                messages=messages,
                node_name="planner",
                max_tokens=settings.codegen_max_tokens,
                model=settings.codegen_model or None,
            )
            if raw and raw.strip():
                logger.info(
                    "CodeGen: generated %d chars for '%s' (%s)",
                    len(raw), code_prompt[:60], language,
                )
                return raw.strip()
        except Exception as exc:
            logger.warning("CodeGen LLM call failed: %s", exc)
        return ""

    def _build_project_brief(self, plan: Plan, goal: str) -> str:
        """Deterministic project brief (used when the design call is unavailable)."""
        files = sorted({
            str(t.input.get("path", ""))
            for t in plan.tasks
            if str(t.input.get("path", ""))
        })
        role_lines = []
        for f in files:
            role_lines.append(f"  - {f}：{_infer_file_role(f)}")
        lines = [f"用户目标：{goal}"]
        if files:
            lines.append("计划创建的文件及职责：")
            lines.extend(role_lines)
        return "\n".join(lines)

    def _generate_project_design(self, goal: str, plan: Plan) -> str:
        """One-shot LLM design pass shared by every codegen call in this plan.

        Asks the LLM to design the tech stack, data model, API list and module
        responsibilities up-front, so later per-file codegen calls all follow
        the same design instead of inventing isolated, incoherent files.
        Returns a compact design text, or "" on any failure (caller falls back
        to the deterministic brief).
        """
        files = sorted({
            str(t.input.get("path", ""))
            for t in plan.tasks
            if str(t.input.get("path", ""))
        })
        try:
            messages = build_project_design_prompt(goal, files)
            raw = self._llm.complete(
                messages=messages,
                node_name="planner",
                max_tokens=2000,
                model=settings.codegen_model or None,
            )
            parsed = self._parse_json(raw or "")
            if not isinstance(parsed, dict):
                return ""
            lines: list[str] = []
            ts = parsed.get("tech_stack")
            if ts:
                lines.append(f"技术栈：{ts}")
            for req in (parsed.get("key_requirements") or []):
                if req:
                    lines.append(f"核心需求：{req}")
            dm = parsed.get("data_model") or []
            if dm:
                lines.append("数据模型：")
                for t in dm:
                    if isinstance(t, dict):
                        fields = "；".join(str(f) for f in (t.get("fields") or []))
                        lines.append(f"  - {t.get('table', '')}: {fields}")
            eps = parsed.get("api_endpoints") or []
            if eps:
                lines.append("接口列表：")
                for e in eps:
                    lines.append(f"  - {e}")
            mods = parsed.get("modules") or []
            if mods:
                lines.append("模块职责：")
                for m in mods:
                    if isinstance(m, dict):
                        lines.append(f"  - {m.get('file', '')}: {m.get('responsibility', '')}")
            brief = "\n".join(lines)[:1500]
            if brief.strip():
                logger.info("Project design generated (%d chars)", len(brief))
                return brief
        except Exception as exc:
            logger.warning("Project design call failed: %s", exc)
        return ""

    def _fill_code_content(self, plan: Plan, project_brief: str = "") -> list[str]:
        """Post-process: fill code content for tasks that need it.

        For each write_file/create_file task targeting a code file:
        - If content is already set (template or inline), skip.
        - Otherwise generate code from the task prompt + the shared project
          brief, validate syntax, and retry once when the first attempt is
          invalid (the retry prompt includes the syntax error for feedback).

        Returns a list of validation error messages for the caller to record.
        """
        errors: list[str] = []
        for task in plan.tasks:
            inp = task.input
            action = str(inp.get("action", "") or task.goal or "")
            path = str(inp.get("path", ""))

            # Only handle write operations on code files
            if action not in ("write_file", "create_file"):
                continue
            if not _is_code_file(path) and not _is_fillable_text_file(path):
                continue

            # Skip if content is already present and substantial
            existing = str(inp.get("content", ""))
            if existing and _looks_substantial(existing, path):
                continue

            # Determine code prompt
            code_prompt = str(inp.get("code_prompt", "")).strip()
            if not code_prompt:
                code_prompt = str(task.goal or "")
            # If the goal is just an action name, derive from path + role
            _ACTION_NAMES = frozenset({
                "write_file", "create_file", "mkdir", "append_file", "edit_file",
            })
            if not code_prompt or code_prompt in _ACTION_NAMES:
                role = _infer_file_role(path)
                code_prompt = f"编写 {path}。它在项目中的职责：{role}。"
            if not code_prompt or len(code_prompt) < 3:
                continue

            language = _guess_language(path)
            logger.info(
                "CodeGen: filling content for '%s' (lang=%s, prompt='%s')",
                path, language, code_prompt[:80],
            )
            code = self._call_codegen(code_prompt, language, project_brief)
            if code:
                ok, detail = _validate_generated_code(path, code)
                if not ok:
                    retry_prompt = (
                        f"{code_prompt}\n\n"
                        f"上次生成的代码存在语法错误：{detail}\n"
                        "请修复后输出完整代码。"
                    )
                    code2 = self._call_codegen(retry_prompt, language, project_brief)
                    if code2:
                        ok2, detail2 = _validate_generated_code(path, code2)
                        if ok2:
                            code = code2
                        else:
                            errors.append(f"{path}: {detail2}")
                            logger.warning("CodeGen retry still invalid: %s", detail2)
                    else:
                        errors.append(f"{path}: retry produced no code")
                inp["content"] = code
                # Remove code_prompt now that it's been fulfilled
                inp.pop("code_prompt", None)
            else:
                logger.warning("CodeGen: failed to generate code for '%s'", path)
                errors.append(f"{path}: no code generated")
        return errors

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
# Tool argument parsing
# ------------------------------------------------------------------


def _parse_tool_arguments(arguments_str: str | None, tool_name: str) -> dict[str, Any]:
    """Parse tool call arguments with fallback for common LLM JSON issues.

    LLM outputs (especially DeepSeek) often embed unescaped newlines
    or quotes inside string values.  This tries multiple approaches.
    """
    if not arguments_str or not arguments_str.strip():
        return {}

    raw = arguments_str.strip()

    # 1) Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2) Try replacing literal newlines with \\n within string values.
    #    This is the most common LLM JSON issue.
    fixed = _fix_json_newlines(raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 3) Last-resort: find outermost { … } and try json.loads on content
    #    after stripping leading/trailing non-JSON text.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    logger.warning(
        "Failed to parse arguments for %s (length=%d, preview=%s)",
        tool_name, len(raw), raw[:120],
    )
    return {}


def _extract_path_from_args(raw: str | None) -> str | None:
    """Extract the ``path`` field from malformed JSON arguments via regex.

    When LLM function calling returns corrupted JSON (common with large code
    content), full JSON parsing fails but the ``path`` field is often at the
    start of the JSON and still recoverable.  This simple regex extraction
    serves as a fallback so the planner can at least create the directory
    structure for the reflector to fill in later.
    """
    if not raw:
        return None
    m = re.search(r'"path"\s*:\s*"([^"]+)"', raw)
    return m.group(1) if m else None


def _extract_content_from_args(raw: str | None) -> str | None:
    """Extract the ``content`` field from malformed JSON arguments via regex.

    When LLM function calling returns corrupted JSON (common with large code
    content containing unescaped newlines/quotes), the ``content`` field is
    typically the last field in the JSON object.  This extracts it so that
    write_file tasks don't need to be silently dropped.

    After extraction, if the content contains markdown code blocks, the code
    inside the blocks is returned as the final file content.
    """
    if not raw:
        return None
    raw_text = None
    # Strategy 1: content is the last field — greedy match to trailing "}
    m = re.search(r'"content"\s*:\s*"(.*)"\s*\}?\s*$', raw, re.DOTALL)
    if m:
        raw_text = _unescape_json_content(m.group(1))
    # Strategy 2: content is followed by other fields — lazy match to next "field":
    if raw_text is None:
        m = re.search(r'"content"\s*:\s*"(.+?)"\s*,\s*"[a-z_]+"\s*:', raw, re.DOTALL)
        if m:
            raw_text = _unescape_json_content(m.group(1))
    if raw_text is None:
        return None
    return _extract_code_from_markdown(raw_text)


def _extract_code_prompt_from_args(raw: str | None) -> str | None:
    """Extract the ``code_prompt`` field from malformed JSON arguments.

    When the FC planner passes ``code_prompt`` (not ``content``) for code files,
    this recovers the code prompt so the CodeGenerator can fill content later.
    """
    if not raw:
        return None
    m = re.search(r'"code_prompt"\s*:\s*"(.*?)"\s*[,}]', raw, re.DOTALL)
    if m:
        return _unescape_json_content(m.group(1))
    return None


def _extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown code blocks.

    If *text* contains fenced code blocks (```), returns the content inside
    the first code block, stripping the language tag and surrounding fences.
    Multiple blocks are concatenated with double newlines.

    If *text* has no code blocks, returns *text* as-is (it's plain content).
    """
    if not text or "```" not in text:
        return text

    blocks: list[str] = []
    in_block = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_block:
                in_block = False
            else:
                in_block = True
            continue
        if in_block:
            blocks.append(line)

    if not blocks:
        # Had ``` markers but no content between them — return original
        return text

    return "\n".join(blocks).strip()


def _unescape_json_content(raw: str) -> str:
    """Undo basic JSON string escaping in recovered content."""
    result = raw
    result = result.replace("\\n", "\n")
    result = result.replace("\\t", "\t")
    result = result.replace("\\r", "\r")
    result = result.replace('\\"', '"')
    result = result.replace("\\\\", "\\")
    return result


def _fix_json_newlines(raw: str) -> str:
    """Replace literal newlines inside JSON strings with \\n escapes.

    A simple heuristic: inside a JSON string (between unescaped quotes)
    we find actual newlines (\\n) and escape them.  This is **not** a
    full JSON repair — it handles the most common LLM output defect.
    """
    result = []
    in_string = False
    escape = False
    for ch in raw:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == "\\":
            result.append(ch)
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch in "\n\r":
            result.append("\\n")
            continue
        result.append(ch)
    return "".join(result)


# CodeGenerator helpers
# ------------------------------------------------------------------

# File extensions that contain code (need CodeGenerator).
_CODE_EXTENSIONS = frozenset({
    ".py", ".java", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".cpp",
    ".c", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala", ".rb", ".php",
    ".vue", ".svelte", ".html", ".css", ".scss", ".less", ".sql", ".sh",
    ".bash", ".ps1", ".yaml", ".yml", ".toml", ".xml",
})

# Language name hints for CodeGenerator prompts.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "Python", ".java": "Java", ".js": "JavaScript", ".ts": "TypeScript",
    ".go": "Go", ".rs": "Rust", ".cpp": "C++", ".c": "C", ".html": "HTML",
    ".css": "CSS", ".vue": "Vue", ".jsx": "React JSX", ".tsx": "React TSX",
    ".sql": "SQL", ".sh": "Bash", ".yaml": "YAML", ".yml": "YAML",
}


def _is_code_file(path: str) -> bool:
    """Check if a file path corresponds to a code file that needs generation."""
    dot = path.rfind(".")
    if dot == -1:
        return False
    ext = path[dot:].lower()
    return ext in _CODE_EXTENSIONS


def _guess_language(path: str) -> str:
    """Guess programming language from file extension."""
    dot = path.rfind(".")
    if dot == -1:
        return ""
    return _EXT_TO_LANG.get(path[dot:].lower(), "")


# Helpers
# ------------------------------------------------------------------


def _infer_file_role(path: str) -> str:
    """Infer a file's role in the project from its name/extension."""
    name = Path(path).name.lower()
    stem = Path(path).stem.lower()
    if name in ("app.py", "main.py"):
        return "应用入口（启动、注册路由）"
    if "route" in stem or name == "urls.py":
        return "API 路由/控制器"
    if "model" in stem:
        return "数据模型定义"
    if name.endswith(".sql"):
        return "数据库表结构（schema）"
    if "schema" in stem:
        return "数据库结构"
    if "test" in stem:
        return "单元测试"
    if "config" in stem or name == "settings.py":
        return "配置"
    if name.endswith(".html"):
        return "前端页面"
    if name.endswith((".css", ".js", ".ts", ".vue")):
        return "前端资源/交互"
    if name == "dockerfile" or "docker" in stem or name.endswith((".yml", ".yaml")):
        return "部署/容器配置"
    if name == "readme.md":
        return "项目说明文档"
    if name in ("requirements.txt", "pyproject.toml"):
        return "依赖声明"
    return "项目组件"


def _looks_substantial(content: str, path: str) -> bool:
    """Heuristic: is this real code, or just a template stub?

    Templates seed code files with a short header comment (e.g. ``# 项目名``)
    which is long enough to pass a naive "len > 10" check but contains no
    logic.  Such stubs must NOT suppress code generation, otherwise files like
    ``models.py`` stay empty while ``routes.py`` imports their functions.
    """
    if len(content) >= 500:
        return True
    if path.lower().endswith(".py"):
        code_lines = [
            line for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        has_import = any(
            line.startswith(("import ", "from "))
            for line in code_lines
        )
        return has_import and len(code_lines) >= 4
    return len(content) >= 200


_FILLABLE_TEXT_FILES = {"requirements.txt", "pyproject.toml"}


def _is_fillable_text_file(path: str) -> bool:
    """Dependency-declaration files are also generated (not left as stubs)."""
    return Path(path).name.lower() in _FILLABLE_TEXT_FILES


def _validate_generated_code(path: str, code: str) -> tuple[bool, str]:
    """Lightweight syntax validation for generated code files."""
    if not path.lower().endswith(".py"):
        return True, ""
    source = _extract_code_from_markdown(code)
    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno})"


def _plan_to_workflow(plan: Plan, goal_type: str, registry=None) -> list[str]:
    """Derive a LangGraph node-name list from a Plan (backward compat).

    Tool→node mappings are resolved dynamically from the ToolRegistry.
    """
    nodes = ["goal_analyzer"]

    if goal_type not in ("identity", "search"):
        nodes.append("knowledge")

    if not plan.goal_completed and not plan.direct_answer:
        for t in plan.tasks:
            tool_name = t.tool or ""
            if not tool_name:
                continue
            # Resolve dynamically from registry, with fallback
            if registry is not None:
                node = registry.get_node_for_tool(tool_name)
            else:
                node = None
            if not node:
                node = "tool_executor"  # safe default
            if node not in nodes:
                nodes.append(node)

    if "answer" not in nodes:
        nodes.append("answer")
    if "memory" not in nodes:
        nodes.append("memory")
    return nodes
