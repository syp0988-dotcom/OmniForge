"""Planner Agent — LLM-driven task planner with rule-based fallback.

The Planner is the **single entry point** for task planning in the system:

  - Analyses the user question and decides **whether** tools are needed
  - Outputs a structured ``Plan`` with concrete ``Task`` objects
  - Supports two task formats:
      * **New**: ``{"tool": "filesystem", "action": "mkdir", "input": {...}}``
      * **Legacy**: ``{"capability": "web.search"}`` (resolved via CapabilityRegistry)
  - Falls back to rule-based planning when the LLM is unavailable

Architecture::

    RouterAgent  (category classification)
         │
         ▼
    PlannerAgent  (LLM-first, rule-fallback)
         │
         ▼
    Plan (tasks[] with tool, action, goal, input)
         │
         ▼
    Executor → ToolRegistry → BaseTool.execute()
"""

from __future__ import annotations

import json
from typing import Any

from agentflow.agents.base import AgentProtocol
from agentflow.agents.planner.capability import (
    list_capabilities,
    resolve as resolve_capability,
)
from agentflow.agents.planner.prompt import build_planner_prompt
from agentflow.graph.plan import Plan
from agentflow.graph.task import Task
from agentflow.services.llm_service import get_llm_service
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("planner")

# Map tool names → LangGraph node names for backward-compatible workflow lists.
_TOOL_TO_NODE: dict[str, str] = {
    "search": "search",
    "python": "python",
    "filesystem": "tool_executor",
    "git": "tool_executor",
    "browser": "tool_executor",
    "database": "tool_executor",
    "mcp": "tool_executor",
}


class PlannerAgent(AgentProtocol):
    """Analyse the user question and produce a Plan with explicit task objects.

    Supports two task formats:

    **New format** (preferred)::

        {
            "tool": "filesystem",
            "action": "mkdir",
            "goal": "创建项目目录",
            "input": {"path": "app"}
        }

    **Legacy format** (backward compat)::

        {
            "capability": "web.search",
            "goal": "搜索网络信息"
        }

    Legacy tasks are resolved via CapabilityRegistry → tool name.
    New format tasks are used directly as-is.
    """

    def __init__(self) -> None:
        self._llm = get_llm_service()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @safe_run
    def run(self, state: dict) -> dict:
        """Generate a Plan and attach it to the workflow state.

        Sets::
          state["plan"]      → Plan object (primary)
          state["workflow"]  → list[str] (derived, backward compat)
        """
        question = str(state.get("question", ""))
        category = str(state.get("category", "reasoning"))

        # --- 1. Try LLM-based planning ---
        plan = self._llm_plan(question, category)

        # --- 2. Fallback to rule-based planning ---
        if plan is None:
            logger.info("LLM plan failed, falling back to rule-based planner")
            plan = self._build_plan(question, category)

        # --- 3. Resolve capabilities → concrete tool names ---
        self._resolve_capabilities(plan)

        if plan.direct_answer:
            logger.info(
                "Plan: direct_answer (capabilities=%s)",
                plan.required_tools or "none",
            )
        else:
            logger.info(
                "Plan: %d steps, tools=%s",
                plan.step_count,
                plan.required_tools,
            )

        state["plan"] = plan

        # Backward compat: derive workflow node list from plan
        state["workflow"] = _plan_to_workflow(plan, category)

        # Register tasks on WorkflowContext (if available)
        if hasattr(state, "add_task"):
            for task in plan.tasks:
                state.add_task(task)
                logger.debug(
                    "  → Task: capability=%-20s tool=%-8s goal=%s",
                    task.capability or "(none)",
                    task.tool or "(none)",
                    task.goal,
                )

        return state

    # ------------------------------------------------------------------
    # LLM-based planning  (primary path)
    # ------------------------------------------------------------------

    def _llm_plan(self, question: str, category: str) -> Plan | None:
        """Try to generate a Plan via LLM call.

        Returns ``None`` on any failure (unparseable JSON, missing fields,
        LLM not configured, etc.) so the caller can fall back.
        """
        if not question:
            return None

        try:
            messages = build_planner_prompt(question, category)
            raw = self._llm.complete(messages=messages)
        except Exception as exc:
            logger.warning("LLM planner call failed: %s", exc)
            return None

        if not raw or not raw.strip():
            logger.warning("LLM planner returned empty response")
            return None

        # Parse JSON from the LLM output
        parsed = self._parse_json(raw)
        if parsed is None:
            logger.warning("LLM planner response was not valid JSON")
            return None

        # Validate and build Plan
        try:
            return self._build_plan_from_json(parsed)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("LLM planner JSON validation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # JSON parser (robust — handles code blocks and extra text)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        """Extract a JSON object from LLM output.

        Tries (in order):
          1. Direct parse of the full string.
          2. Extract from a `````json ... ``` code block.
          3. Extract from any ````` ... ``` code block.
        """
        # Strategy 1: direct parse
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: ```json ... ``` block
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
    # JSON → Plan builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_plan_from_json(data: dict[str, Any]) -> Plan:
        """Convert a validated JSON dict into a Plan object.

        Supports three JSON schemas (auto-detected):

        **1. Explicit task format** (preferred)::

            {
                "direct_answer": false,
                "reasoning": "...",
                "tasks": [
                    {"tool": "filesystem", "action": "mkdir", "goal": "...", "input": {...}},
                    {"tool": "search", "action": "web.search", "goal": "...", "input": {...}}
                ]
            }

        **2. need_web format** (legacy)::

            {"need_web": bool, "tool": "web.search", "intent": "...", "reasoning": "..."}

        **3. Capability format** (legacy)::

            {"direct_answer": bool, "reasoning": str, "tasks": [{"goal": str, "capability": str}]}

        Raises ``ValueError`` when required fields are missing or invalid.
        """
        # --- Detect format ---
        if "need_web" in data:
            return PlannerAgent._build_plan_from_new_format(data)

        if "direct_answer" in data:
            tasks_raw = data.get("tasks", [])
            # Heuristic: if any task has a "tool" key, it's the explicit format
            if tasks_raw and isinstance(tasks_raw, list) and any(
                isinstance(t, dict) and "tool" in t for t in tasks_raw
            ):
                return PlannerAgent._build_plan_from_explicit_tasks(data)
            return PlannerAgent._build_plan_from_legacy_format(data)

        # Fallback: treat as explicit format
        return PlannerAgent._build_plan_from_explicit_tasks(data)

    @staticmethod
    def _build_plan_from_new_format(data: dict[str, Any]) -> Plan:
        """Build Plan from the new ``need_web`` / ``intent`` format."""
        need_web = bool(data.get("need_web", False))
        tool = str(data.get("tool", "")).strip()
        intent = str(data.get("intent", "")).strip()
        reasoning = str(data.get("reasoning", ""))

        tasks: list[Task] = []

        if need_web and tool == "web.search":
            tasks.append(Task(
                goal="从互联网搜索相关信息",
                capability="web.search",
                input={},
            ))
        elif tool == "python.execute":
            tasks.append(Task(
                goal="执行 Python 代码并获取结果",
                capability="python.execute",
                input={"code": data.get("code", "")},
            ))

        return Plan(
            goal="",
            category="",
            tasks=tasks,
            direct_answer=not need_web and tool != "python.execute",
            reasoning=reasoning,
            intent=intent,
        )

    @staticmethod
    def _build_plan_from_legacy_format(data: dict[str, Any]) -> Plan:
        """Build Plan from the legacy ``direct_answer`` / ``tasks`` format."""
        # Validate required fields
        if "direct_answer" not in data:
            raise ValueError("Missing required field: direct_answer")

        direct_answer = bool(data["direct_answer"])
        tasks_raw = data.get("tasks", [])

        if not isinstance(tasks_raw, list):
            raise ValueError("'tasks' must be a list")

        tasks: list[Task] = []
        known_capabilities = list_capabilities()

        for i, item in enumerate(tasks_raw):
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict task entry at index %d", i)
                continue

            capability = str(item.get("capability", "")).strip()
            goal = str(item.get("goal", "")).strip()

            if capability and capability not in known_capabilities:
                logger.warning(
                    "Unknown capability '%s' at index %d; removing from plan",
                    capability,
                    i,
                )
                continue

            tasks.append(
                Task(
                    goal=goal or f"执行 {capability}" if capability else "处理用户请求",
                    capability=capability,
                    agent="planner",
                    input=item.get("input", {}),
                )
            )

        # Consistency check
        if direct_answer and tasks:
            logger.warning(
                "Plan has direct_answer=true but %d tasks; clearing tasks",
                len(tasks),
            )
            tasks = []

        return Plan(
            goal="",
            category="",
            tasks=tasks,
            direct_answer=direct_answer,
            reasoning=str(data.get("reasoning", "")),
            intent=str(data.get("intent", "")),
        )

    @staticmethod
    def _build_plan_from_explicit_tasks(data: dict[str, Any]) -> Plan:
        """Build Plan from the new explicit task format.

        Each task has ``tool``, ``action``, ``goal``, and ``input`` fields::

            {"tool": "filesystem", "action": "mkdir", "goal": "创建目录",
             "input": {"path": "app"}}
        """
        direct_answer = bool(data.get("direct_answer", False))
        tasks_raw = data.get("tasks", [])
        reasoning = str(data.get("reasoning", ""))

        if not isinstance(tasks_raw, list):
            raise ValueError("'tasks' must be a list")

        tasks: list[Task] = []
        for i, item in enumerate(tasks_raw):
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict task entry at index %d", i)
                continue

            tool = str(item.get("tool", "")).strip()
            action = str(item.get("action", "")).strip()
            goal = str(item.get("goal", "")).strip()
            inp = item.get("input", {})

            if not isinstance(inp, dict):
                logger.warning("Task %d 'input' is not a dict; resetting", i)
                inp = {}

            if not tool:
                logger.warning("Task %d has no 'tool'; skipping", i)
                continue

            # Build capability from tool.action for backward compat
            capability = f"{tool}.{action}" if action else tool

            tasks.append(Task(
                goal=goal or f"执行 {tool}.{action}" if action else f"执行 {tool}",
                capability=capability,
                tool=tool,
                agent="planner",
                input=inp,
            ))

        # Consistency check
        if direct_answer and tasks:
            logger.warning(
                "Plan has direct_answer=true but %d tasks; clearing tasks",
                len(tasks),
            )
            tasks = []

        return Plan(
            goal="",
            category="",
            tasks=tasks,
            direct_answer=direct_answer,
            reasoning=reasoning,
        )

    @staticmethod
    def _resolve_capabilities(plan: Plan) -> None:
        """Resolve each task's ``capability`` to a concrete ``tool`` name.

        After this call, every task with a recognised capability will have
        its ``tool`` field set so the Executor can dispatch it.
        Planner-only side-effect: no Tool or Executor code is touched.
        """
        for task in plan.tasks:
            if not task.capability:
                continue
            tool = resolve_capability(task.capability)
            if tool:
                task.tool = tool
                logger.debug("Resolved %s → %s", task.capability, task.tool)

    # ------------------------------------------------------------------
    # Rule-based planning (fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_plan(question: str, category: str) -> Plan:
        """Generate a Plan using capability-oriented rules.

        This is the **fallback** path, used when the LLM is unavailable or
        returns an unparseable response.  All tasks use ``capability``
        (not ``tool``), consistent with the LLM path.
        """

        if category == "search":
            return Plan(
                goal=question,
                category=category,
                tasks=[
                    Task(
                        goal="从网络搜索获取最新信息",
                        capability="web.search",
                    ),
                    Task(
                        goal="综合搜索结果生成自然语言回答",
                    ),
                    Task(
                        goal="将此次对话写入历史记录",
                    ),
                ],
                reasoning="需要实时网络搜索获取最新信息（规则回退）",
                intent="general",
            )

        if category == "identity":
            return Plan(
                goal=question,
                category=category,
                tasks=[
                    Task(goal="生成身份介绍信息回答用户"),
                    Task(goal="将此次对话写入历史记录"),
                ],
                direct_answer=True,
                reasoning="身份问题无需搜索或知识库（规则回退）",
            )

        if category == "python":
            return Plan(
                goal=question,
                category=category,
                tasks=[
                    Task(
                        goal="执行 Python 代码并获取结果",
                        capability="python.execute",
                        input={"code": question},
                    ),
                    Task(
                        goal="综合代码执行结果生成回答",
                    ),
                    Task(
                        goal="将此次对话写入历史记录",
                    ),
                ],
                reasoning="需要 Python 代码执行（规则回退）",
            )

        # Default: reasoning / coding / writing / knowledge
        return Plan(
            goal=question,
            category=category,
            tasks=[
                Task(
                    goal="综合上下文信息回答用户问题",
                ),
                Task(
                    goal="将此次对话写入历史记录",
                ),
            ],
            direct_answer=True,
            reasoning=f"通用问答无需工具调用（category={category}，规则回退）",
        )


# -- Backward compat helpers -------------------------------------------------


def _plan_to_workflow(plan: Plan, category: str) -> list[str]:
    """Derive a LangGraph node-name list from a Plan (backward compat)."""
    nodes = ["router"]

    # Knowledge retrieval runs before planner for most categories
    if category not in ("identity", "search"):
        nodes.append("knowledge")

    # direct_answer: skip tool nodes entirely
    if plan.direct_answer:
        pass  # no tool nodes to add
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
