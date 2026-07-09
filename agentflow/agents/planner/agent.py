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

import json
import re
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
from agentflow.blueprints import BlueprintLoader, FileSpec, ProjectConfigurator
from agentflow.graph.context_builder import ContextBuilder
from agentflow.graph.plan import Plan
from agentflow.graph.task import Task, TaskStatus
from agentflow.services.llm_service import LLMResponse, ToolCall, get_llm_service
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("planner")


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
        goal_analysis = state.get("goal_analysis", {})
        if isinstance(goal_analysis, dict):
            goal = goal_analysis.get("goal", state.get("question", ""))
            goal_type = goal_analysis.get("goal_type", "other")
        else:
            goal = state.get("question", "")
            goal_type = "other"

        # Goal types that never need task planning — conversational or simple
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
        if state.get("_degraded") or state.get("_llm_error"):
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

        # Merge plan tasks into the queue
        merged = self._merge_into_queue(current_queue, plan)

        state["plan"] = plan
        state["task_queue"] = merged.to_dict_list()
        state["category"] = goal_type
        state["workflow"] = _plan_to_workflow(plan, goal_type, self.registry)

        # Detect degraded fallback — LLM was unavailable
        if plan.direct_answer and not plan.goal_completed and not plan.tasks:
            state["_degraded"] = True
            state["_llm_error"] = (
                "LLM planner unavailable — both function-calling and JSON "
                "planning failed (timeout or network error)"
            )
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
                messages=messages, tools=tools, tool_choice="auto",
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
        goal: str = "", category: str = "",
    ) -> Plan:
        """Convert ToolCall objects into a Plan with Task objects."""
        tasks: list[Task] = []
        for i, tc in enumerate(tool_calls):
            tool, action = parse_function_name(tc.name)
            inp = _parse_tool_arguments(tc.arguments, tc.name)

            # Write_file with no path = args parsing failed (common with DeepSeek FC
            # when large code content corrupts the JSON).  Instead of silently
            # skipping, try to recover the path from raw args and create a mkdir
            # task so the reflector can detect the empty directory and generate
            # the actual file content via _generate_stuck_tasks.
            if action in ("write_file", "create_file") and not inp.get("path"):
                path = _extract_path_from_args(tc.arguments)
                if path:
                    parent = str(Path(path).parent)
                    if parent and parent != ".":
                        mkdir_path = parent
                    else:
                        # No parent dir in path, use the filename stem as project dir
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
                        "Extracted path='%s' from malformed FC args → mkdir '%s'",
                        path, mkdir_path,
                    )
                else:
                    logger.warning(
                        "Skipping %s task — no valid path after parsing args", tc.name
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

        return Plan(
            goal=goal, category=category,
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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


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
