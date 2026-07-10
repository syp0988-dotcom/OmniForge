"""ReflectionAgent — LLM-based task queue validation for Dynamic Task Queue Planning.

In the Dynamic Task Queue system, the ReflectionAgent examines:
  1. Did the executed tasks complete successfully?
  2. Does the workspace match expectations?
  3. What updates are needed to the Task Queue?
  4. Is the overall goal completed?

Output (stored in state["_reflection_output"]):

    {
        "goal_completed": false,
        "task_updates": [
            {"task_id": "create_app", "status": "DONE"},
            {"task_id": "create_config", "status": "FAILED", "priority": 60}
        ],
        "new_tasks": [
            {"task_id": "create_requirements", "title": "创建 requirements.txt",
             "priority": 95, "tool": "filesystem"}
        ],
        "remove_tasks": ["create_docker"],
        "reason": "后端 app.py 已创建，但缺少 requirements.txt"
    }

Derived fields (backward compat):
  - state["_reflection_result"] = "done" | "next" | "replan" | "retry"
  - state["_reflection_message"] = reason text
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentflow.agents.base import AgentProtocol
from agentflow.agents.planner.task_queue import TaskQueue
from agentflow.agents.planner.templates import (
    extract_project_name,
    get_existing_files,
    get_initial_tasks,
    is_goal_completed,
    match_template,
)
from agentflow.graph.task import Task, TaskStatus
from agentflow.services.llm_service import get_llm_service
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("reflection")

SYSTEM_PROMPT = """你是一个任务队列反射评估器（Task Queue Reflector）。
你的职责是检查已执行任务的结果，并更新任务队列。

输出 JSON 格式（不要包含其他文字）：

{
    "goal_completed": false,
    "task_updates": [
        {"task_id": "create_app", "status": "DONE"},
        {"task_id": "create_config", "status": "FAILED"}
    ],
    "new_tasks": [
        {"task_id": "create_requirements", "title": "创建 requirements.txt",
         "priority": 95, "tool": "filesystem",
         "input": {"action": "write_file", "path": "图书管理/requirements.txt", "content": "flask\\n"}}
    ],
    "remove_tasks": [],
    "reason": "判断理由说明"
}

字段说明：
- goal_completed: 整个目标是否已完成
- task_updates: 需要更新的任务（status 变化或 priority 调整）
- new_tasks: 需要新增的任务（当发现工作区缺少必要内容时）
- remove_tasks: 需要删除的任务（当任务已不再需要时）
- reason: 判断理由

判断逻辑：
1. 根据 tool_results 判断哪些任务执行成功/失败
2. 成功 → task_updates 中标记为 DONE
3. 失败且可重试（网络超时等）→ 标记 FAILED 但不触发 replan
4. 失败且不可恢复 → 标记 FAILED 需触发 replan
5. 查看工作区：如果发现缺少必要文件 → 通过 new_tasks 新增任务
6. 如果某个任务的产出文件已存在但任务未完成 → 直接标记 DONE
7. 如果任务已过时（如 Docker 已存在）→ 加入 remove_tasks
8. 所有高优先级任务都 DONE 且工作区满足预期 → goal_completed=true

重要：如果你发现工作区缺少文件，通过 new_tasks 创建 write_file 任务时，**必须提供完整的文件内容**（input.content 字段）。请根据用户目标和已存在的文件，生成完整的、可直接运行的代码。

注意：JSON 中的字符串值如果包含换行，请使用 \\n 转义，不要使用实际换行。
"""


class ReflectionAgent(AgentProtocol):
    """LLM-based task queue validation for Dynamic Task Queue Planning."""

    def __init__(self) -> None:
        self._llm = get_llm_service()

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        """Evaluate task results and update the task queue.

        Uses rule-based evaluation by default. Only calls LLM for complex
        decisions: failures (replan/retry) or goal-completion confirmation.
        This eliminates ~1 LLM call per task for routine success flows.
        """
        goal_analysis = state.get("goal_analysis", {})
        if isinstance(goal_analysis, dict):
            goal = goal_analysis.get("goal", state.get("question", ""))
            goal_type = goal_analysis.get("goal_type", "other")
        else:
            goal = state.get("question", "")
            goal_type = "other"

        tool_results = state.get("tool_results", [])
        current_queue = TaskQueue.from_dict_list(
            state.get("task_queue", []) or []
        )
        project_name = extract_project_name(goal)

        # Non-project: just return done (backward compat)
        if current_queue.is_empty:
            if goal_type == "project":
                generated = _generate_stuck_tasks(goal, project_name, tool_results)
                if generated:
                    for cfg in generated:
                        task_id = cfg.get("task_id", "")
                        if not task_id:
                            continue
                        current_queue.add(Task(
                            task_id=task_id,
                            title=cfg.get("title", task_id),
                            priority=cfg.get("priority", 50),
                            tool=cfg.get("tool", "filesystem"),
                            goal=cfg.get("title", task_id),
                            input=cfg.get("input", {}),
                            status=TaskStatus.TODO,
                        ))
                    state["task_queue"] = current_queue.to_dict_list()
                    state["_reflection_result"] = "next"
                    state["_reflection_message"] = "已生成文件创建兜底任务"
                    state["_reflection_output"] = {
                        "goal_completed": False,
                        "task_updates": [],
                        "new_tasks": generated,
                        "remove_tasks": [],
                        "reason": "项目队列为空，已补充文件创建任务",
                    }
                    logger.info(
                        "Reflection: empty project queue -> added %d fallback task(s)",
                        len(generated),
                    )
                    return state
            logger.info("Reflection: empty queue -> done")
            return self._done("任务队列为空，无需继续")

        # Evaluate: rule-based first, LLM only for complex decisions
        reflection = self._evaluate(
            goal, goal_type, current_queue, tool_results,
        )

        # Apply task updates
        self._apply_reflection(current_queue, reflection)

        # ── Fallback: generate content tasks when the queue is stuck ──
        # If the reflection didn't produce any TODO tasks (e.g. the LLM
        # generated write_file tasks without content, which got skipped
        # by the guard above), generate tasks WITH content to break the
        # infinite loop.  This is a safety net.
        #
        # Also triggers when the LLM incorrectly reports goal_completed=True
        # but the queue only has infrastructure tasks (mkdir) without any
        # file-creation work — the goal cannot be "complete" with no files.
        _WRITE_ACTIONS = frozenset({"write_file", "create_file", "append_file", "edit_file"})
        _queue_has_file_creation = any(
            t.input.get("action", "") in _WRITE_ACTIONS
            for t in current_queue.all
        )
        _stuck = (
            not reflection.get("goal_completed") and not current_queue.has_todo
        ) or (
            reflection.get("goal_completed") and not _queue_has_file_creation
        )
        if _stuck:
            # If the LLM incorrectly said "done" but we're stuck, reset it
            if reflection.get("goal_completed"):
                reflection["goal_completed"] = False
                logger.info("Reflection: overriding false goal_completed (queue has no file-creation tasks)")
            generated = _generate_stuck_tasks(goal, project_name, tool_results)
            if generated:
                for cfg in generated:
                    task_id = cfg.get("task_id", "")
                    if task_id and not current_queue.get(task_id):
                        task = Task(
                            task_id=task_id,
                            title=cfg.get("title", task_id),
                            priority=cfg.get("priority", 50),
                            tool=cfg.get("tool", "filesystem"),
                            goal=cfg.get("title", task_id),
                            input=cfg.get("input", {}),
                            status=TaskStatus.TODO,
                        )
                        current_queue.add(task)
                        reflection.setdefault("new_tasks", []).append(cfg)
                logger.info(
                    "Reflection fallback: added %d content task(s) to break stuck queue",
                    len(generated),
                )
            else:
                # No deterministic template matched and no LLM fallback is used.
                # Report the failure reason instead of trying endless recovery.
                reflection["goal_completed"] = True
                reflection["reason"] = "无法自动生成文件内容"
                state["_generation_failed"] = True
                state["_generation_failure_reason"] = (
                    f"已创建项目目录，但无法自动生成「{goal}」的代码文件。\n\n"
                    "可能的原因：\n"
                    "1. 大模型输出的代码内容在传输过程中损坏，导致文件写入任务丢失\n"
                    "2. 当前内置模板不支持该编程语言或项目类型\n\n"
                    "请重新描述你的需求，或直接告诉我需要创建哪些文件及其具体内容。"
                )
                # Persist to session_state so follow-up questions (e.g. "为什么失败")
                # can access the concrete failure reason instead of hallucinating.
                ss = state.get("session_state")
                if ss is not None:
                    ss.metadata["last_failure_reason"] = state["_generation_failure_reason"]
                    ss.metadata["last_failure_goal"] = goal
                logger.warning(
                    "Reflection fallback: no template for goal '%s', reporting failure",
                    goal[:80],
                )

        # Check template-based completion
        template = match_template(goal, goal_type)
        if template and is_goal_completed(template, current_queue.all):
            reflection["goal_completed"] = True
            reflection["reason"] = reflection.get("reason", "") + "（所有必需任务已完成）"

        # Store output
        state["_reflection_output"] = reflection
        state["task_queue"] = current_queue.to_dict_list()

        goal_completed = reflection.get("goal_completed", False)
        need_replan = reflection.get("need_replan", False)

        if goal_completed:
            state["_reflection_result"] = "done"
            state["_reflection_message"] = reflection.get("reason", "目标已完成")
            logger.info("Reflection -> done")
        elif need_replan:
            replan_count = int(state.get("_replan_count", 0)) + 1
            state["_replan_count"] = replan_count
            state["_reflection_result"] = "replan"
            state["_reflection_message"] = reflection.get("reason", "需要重新规划")
            logger.info("Reflection -> replan (attempt %d/3)", replan_count)
        elif current_queue.has_todo:
            state["_reflection_result"] = "next"
            state["_reflection_message"] = reflection.get("reason", "继续执行")
            logger.info("Reflection -> next (%d TODO tasks remain)", current_queue.todo_count)
        else:
            # No TODO tasks but goal not completed -> need more tasks from planner
            state["_reflection_result"] = "next"
            state["_reflection_message"] = reflection.get("reason", "需要新任务")
            logger.info("Reflection -> next (no TODO, need more tasks)")

        return state

    # ------------------------------------------------------------------
    # Apply reflection updates to the task queue
    # ------------------------------------------------------------------

    def _apply_reflection(
        self,
        queue: TaskQueue,
        reflection: dict[str, Any],
    ) -> None:
        """Apply task_updates, new_tasks, and remove_tasks to the queue."""
        # Update existing tasks
        for upd in reflection.get("task_updates", []):
            task_id = upd.get("task_id", "")
            if not task_id:
                continue
            status = upd.get("status", "")
            priority = upd.get("priority")
            kwargs: dict[str, Any] = {}
            if status:
                kwargs["status"] = status
            if priority is not None:
                kwargs["priority"] = priority
            if kwargs:
                queue.update(task_id, **kwargs)

        # Add new tasks
        for cfg in reflection.get("new_tasks", []):
            task_id = cfg.get("task_id", "")
            if not task_id or queue.get(task_id):
                continue  # Skip if already exists

            inp = cfg.get("input", {}) or {}
            tool = cfg.get("tool", "filesystem")
            action = inp.get("action", "")

            # Skip write_file without content.  The LLM was told to provide
            # content but may have failed; the _generate_stuck_tasks fallback
            # (called below) will try to fill in the missing content via
            # a focused LLM call.
            if tool == "filesystem" and action in ("write_file", "create_file") and not inp.get("content"):
                logger.info(
                    "Reflection: skipping write_file task '%s' (no content, fallback will fill)",
                    task_id,
                )
                continue

            task = Task(
                task_id=task_id,
                title=cfg.get("title", task_id),
                priority=cfg.get("priority", 50),
                tool=tool,
                goal=cfg.get("title", task_id),
                input=inp,
                status=TaskStatus.TODO,
            )
            queue.add(task)

        # Remove tasks
        for task_id in reflection.get("remove_tasks", []):
            queue.remove(task_id)

    # ------------------------------------------------------------------
    # LLM-based evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        goal: str,
        goal_type: str,
        queue: TaskQueue,
        tool_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate task results — rule-based by default, LLM only for complex cases.

        Rule evaluation handles simple success flows without LLM cost.
        LLM is only called when:
          - Tasks have failed (need replan/retry decision)
          - All tasks appear done (need goal_completed confirmation)
          - Queue is empty but goal not completed (need new task ideas)
        """
        # Always start with rule evaluation (fast, no LLM cost)
        rule_result = self._rule_evaluation(queue, tool_results, goal, goal_type)

        # Determine if LLM is needed for deeper analysis
        has_failure = rule_result.get("need_replan", False)
        all_done = rule_result.get("goal_completed", False)
        stuck = not queue.has_todo and not all_done

        logger.info(
            "Reflection eval: has_failure=%s all_done=%s stuck=%s rule_new_tasks=%d",
            has_failure, all_done, stuck, len(rule_result.get("new_tasks", [])),
        )

        if all_done and not has_failure:
            logger.info("Reflection eval -> rule-complete, skipping LLM")
            return rule_result

        if has_failure or stuck:
            logger.info("Reflection eval -> calling LLM")
            llm_result = self._llm_evaluate(goal, goal_type, queue, tool_results)
            if llm_result:
                logger.info("Reflection eval -> LLM returned new_tasks=%d",
                    len(llm_result.get("new_tasks", [])),
                )
                return llm_result
            else:
                logger.info("Reflection eval -> LLM returned None, using rule result")

        return rule_result

    def _llm_evaluate(
        self,
        goal: str,
        goal_type: str,
        queue: TaskQueue,
        tool_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Call LLM for complex evaluation decisions (failures, completion check)."""
        # Build task summary
        queue_summary = queue.summary

        # Build result summary
        result_lines = []
        for r in tool_results:
            if not isinstance(r, dict):
                continue
            success = r.get("success", False)
            err = r.get("error", "")
            path = ""
            res = r.get("result", {}) or {}
            if isinstance(res, dict):
                path = res.get("path", "")
            status = "OK" if success else "FAIL"
            detail = path or err or "ok"
            result_lines.append(f"  {status} {detail}")

        # Build workspace context
        ws_lines = []
        project_name = extract_project_name(goal)
        scan_dir = Path(project_name) if project_name else Path.cwd()
        if scan_dir.exists():
            ws_lines.append(f"工作区文件（{scan_dir}）：")
            for entry in sorted(scan_dir.iterdir())[:20]:
                marker = "/" if entry.is_dir() else ""
                ws_lines.append(f"  {entry.name}{marker}")

        user_content = (
            f"用户目标：{goal}\n"
            f"目标类型：{goal_type}\n\n"
            f"当前任务队列：\n{queue_summary}\n\n"
            f"执行结果：\n" + "\n".join(result_lines) + "\n"
        )
        if ws_lines:
            user_content += "\n" + "\n".join(ws_lines) + "\n"

        user_content += (
            "\n请分析任务执行结果，更新任务队列状态。"
            "注意检查工作区是否缺少必要文件。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            logger.info("Reflection LLM: calling with queue=%s", queue.summary[:100])
            raw = self._llm.complete(messages=messages)
            logger.info("Reflection LLM: raw output (first 300) = %s", raw[:300])
            parsed = self._parse_json(raw)
            if parsed:
                logger.info("Reflection LLM: parsed OK — goal_completed=%s, new_tasks=%d",
                    parsed.get("goal_completed"),
                    len(parsed.get("new_tasks", [])),
                )
                return parsed
            else:
                logger.warning("Reflection LLM: parse FAILED, raw=%s", raw[:300])
        except Exception as exc:
            logger.warning("ReflectionAgent LLM call failed: %s", exc)

        # Fallback: rule-based evaluation
        return self._rule_evaluation(queue, tool_results, goal, goal_type)

    def _rule_evaluation(
        self,
        queue: TaskQueue,
        results: list[dict[str, Any]],
        goal: str,
        goal_type: str,
    ) -> dict[str, Any]:
        """Fallback evaluation when LLM is unavailable."""
        task_updates: list[dict[str, Any]] = []
        new_tasks: list[dict[str, Any]] = []
        remove_tasks: list[str] = []
        # Only consider all_ok if there were actual results to evaluate
        all_ok = bool(results)
        has_failure = False

        # Process tool results and update matching tasks
        for r in results:
            if not isinstance(r, dict):
                continue
            success = r.get("success", False)
            error = r.get("error", "")
            task_name = r.get("action", r.get("goal", ""))

            # Try to find the corresponding task by matching goal/path.
            # Executor already sets status to done/failed before reflector runs,
            # so search all non-TODO tasks instead of filtering by "running".
            matched = None
            for t in queue.all:
                if t.status in (TaskStatus.TODO, TaskStatus.RUNNING):
                    continue
                if task_name and (task_name in t.goal or task_name in t.title):
                    matched = t
                    break

            if matched:
                if success:
                    task_updates.append({"task_id": matched.task_id, "status": "DONE"})
                else:
                    task_updates.append({"task_id": matched.task_id, "status": "FAILED"})
                    has_failure = True

            if success:
                # Check path from result
                res = r.get("result", {}) or {}
                if isinstance(res, dict) and res.get("path"):
                    pass  # Task completed and wrote to path
            else:
                all_ok = False

        # Check if any TODO tasks need to be DONE because files already exist
        project_name = extract_project_name(goal)
        if project_name:
            project_path = Path(project_name)
            if project_path.exists() and project_path.is_dir():
                existing = get_existing_files(str(project_path))
                template = match_template(goal, goal_type)
                if template:
                    template_tasks = get_initial_tasks(template, goal, existing)
                    for tt in template_tasks:
                        if tt.status == TaskStatus.DONE:
                            existing_task = queue.get(tt.task_id)
                            if existing_task and existing_task.status == TaskStatus.TODO:
                                task_updates.append({
                                    "task_id": tt.task_id,
                                    "status": "DONE",
                                })

        # Determine completion
        # A project queue with only mkdir/infrastructure tasks but no
        # content-creating tasks (write_file, etc.) is NOT complete.
        _WRITE_ACTIONS = frozenset({"write_file", "create_file", "append_file", "edit_file"})
        _has_file_creation = any(
            t.input.get("action", "") in _WRITE_ACTIONS
            for t in queue.all
        )
        if all_ok and not _has_file_creation:
            all_ok = False
            logger.info(
                "Rule eval: queue has no file-creation tasks (only mkdir/infra), "
                "forcing all_ok=False to trigger stuck fallback"
            )

        if all_ok:
            template = match_template(goal, goal_type)
            if template:
                # Re-read queue after updates
                temp_queue = TaskQueue()
                temp_queue._tasks = list(queue.all)  # Access internal for efficiency
                for upd in task_updates:
                    tid = upd.get("task_id", "")
                    s = upd.get("status", "")
                    if tid and s:
                        temp_queue.update(tid, status=s)
                goal_completed = is_goal_completed(template, temp_queue.all)
            elif goal_type in ("project", "coding"):
                goal_completed = _all_project_tasks_done(queue, results)
            else:
                # Non-project goals: simple all-done check
                goal_completed = all_ok and not queue.has_todo
        else:
            goal_completed = False

        # If no results were returned but tasks exist in the queue,
        # check for failed tasks to avoid silent false completion
        if not results and queue.all:
            has_failure = has_failure or any(
                t.status == TaskStatus.FAILED for t in queue.all
            )

        return {
            "goal_completed": goal_completed,
            "task_updates": task_updates,
            "new_tasks": new_tasks,
            "remove_tasks": remove_tasks,
            "need_replan": has_failure,
            "reason": (
                f"{len(task_updates)} 个任务已更新"
                + (f"，{len(new_tasks)} 个新任务" if new_tasks else "")
                + (f"，{len(remove_tasks)} 个已删除" if remove_tasks else "")
            ),
        }

    # ------------------------------------------------------------------
    # JSON parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        """Extract JSON from LLM output."""
        text = raw.strip()
        candidates = [text]

        # Extract text from ```json ... ``` blocks
        for marker in ("```json", "```JSON", "```"):
            start = text.find(marker)
            if start == -1:
                continue
            content = text[start + len(marker):]
            end = content.rfind("```")
            if end != -1:
                content = content[:end]
            content = content.strip()
            if content:
                candidates.append(content)

        for c in candidates:
            # Direct parse
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                pass

            # Repair: escape literal newlines inside string values
            fixed = _fix_json_newlines(c)
            if fixed != c:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        return None

    @staticmethod
    def _done(reason: str) -> dict[str, object]:
        return {
            "_reflection_output": {
                "goal_completed": True,
                "task_updates": [],
                "new_tasks": [],
                "remove_tasks": [],
                "reason": reason,
            },
            "_reflection_result": "done",
            "_reflection_message": reason,
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _fix_json_newlines(raw: str) -> str:
    """Replace literal newlines inside JSON strings with \\n escapes.

    LLMs commonly output JSON with unescaped newlines inside string
    values (e.g. the ``content`` field of a write_file task).  This
    simple state-machine fix escapes them without a full JSON parser.
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


def _fallback_project_dir(goal: str, project_name: str) -> str:
    """Return a stable directory for generated file fallbacks."""
    text = (goal or "").lower()
    if any(word in text for word in ("snake", "贪吃蛇")):
        return "generated_files/snake_games"
    if project_name:
        cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", project_name).strip("._")
        if cleaned and len(cleaned) <= 48:
            return f"generated_files/{cleaned}"
    return "generated_files"


def _existing_file_names(results: list[dict[str, Any]], dir_name: str) -> set[str]:
    """Collect known existing file names from previous filesystem results."""
    existing: set[str] = set()
    base = Path(dir_name)
    if base.exists() and base.is_dir():
        existing.update(p.name for p in base.iterdir() if p.is_file())

    for r in results:
        if not isinstance(r, dict):
            continue
        res = r.get("result", {}) or {}
        path = ""
        if isinstance(res, dict):
            path = str(res.get("path", "") or res.get("file", "") or "")
        elif isinstance(res, str):
            path = res
        if path:
            existing.add(Path(path).name)
    return existing


def _deterministic_file_fallback_tasks(
    goal: str,
    project_name: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create concrete file tasks for common explicit file-generation goals."""
    text = (goal or "").lower()
    wants_python = "python" in text or re.search(r"\bpy\b", text) is not None
    wants_java = "java" in text
    wants_go = "go" in text or "golang" in text or "go语言" in goal
    wants_snake = "snake" in text or "贪吃蛇" in goal
    wants_files = any(token in goal for token in ("文件", "创建", "新建", "生成", "写", "完成"))

    if not wants_files:
        return []

    dir_name = _fallback_project_dir(goal, project_name)
    existing = _existing_file_names(results, dir_name)
    specs: list[tuple[str, str]] = []

    if wants_snake and wants_python:
        specs.append(("python_snake.py", _PYTHON_SNAKE_TEMPLATE))
    if wants_snake and wants_java:
        specs.append(("SnakeGame.java", _JAVA_SNAKE_TEMPLATE))
    if wants_snake and wants_go:
        specs.append(("snake.go", _GO_SNAKE_TEMPLATE))

    if not specs:
        if wants_python:
            specs.append(("main.py", _BASIC_MAIN_PY.format(project_name=project_name or "project")))
        if wants_java:
            specs.append(("Main.java", _BASIC_MAIN_JAVA))
        if wants_go:
            specs.append(("main.go", _BASIC_MAIN_GO.format(project_name=project_name or "project")))

    tasks: list[dict[str, Any]] = []
    for filename, content in specs:
        if filename in existing:
            continue
        task_id = f"create_{filename.replace('.', '_').replace('-', '_')}"
        tasks.append({
            "task_id": task_id,
            "title": f"创建 {filename}",
            "priority": 95,
            "tool": "filesystem",
            "input": {
                "action": "write_file",
                "path": f"{dir_name}/{filename}",
                "content": content,
            },
        })

    if tasks:
        logger.info(
            "Deterministic fallback: created %d file task(s): %s",
            len(tasks),
            [t.get("input", {}).get("path", "?") for t in tasks],
        )
    return tasks


def _all_project_tasks_done(queue: TaskQueue, results: list[dict[str, Any]]) -> bool:
    """Return True when a project/coding queue can be completed without LLM."""
    if not queue.all or queue.has_todo:
        return False
    if any(t.status == TaskStatus.FAILED for t in queue.all):
        return False
    if not all(t.status == TaskStatus.DONE for t in queue.all):
        return False
    if not results:
        return False
    # Don't consider done if the queue only has infrastructure tasks (mkdir)
    # and no actual file-creation tasks.
    _WRITE_ACTIONS = frozenset({"write_file", "create_file", "append_file", "edit_file"})
    if not any(t.input.get("action", "") in _WRITE_ACTIONS for t in queue.all):
        return False
    return all(not isinstance(r, dict) or r.get("success", False) for r in results)


def _generate_stuck_tasks(
    goal: str,
    project_name: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate file creation tasks from deterministic templates only.

    No LLM fallback — if no template matches, returns empty so the
    reflector can report the failure reason to the user.
    """
    return _deterministic_file_fallback_tasks(goal, project_name, results)


# ── Minimal content templates (used by deterministic fallback) ─────────

_BASIC_MAIN_PY = '''"""
{project_name} - 主程序入口
"""

def main():
    print("Hello from {project_name}!")


if __name__ == "__main__":
    main()
'''

_BASIC_MAIN_JAVA = '''public class Main {
    public static void main(String[] args) {
        System.out.println("Hello from generated Java project!");
    }
}
'''

_BASIC_MAIN_GO = '''package main

import "fmt"

func main() {{
    fmt.Println("Hello from {project_name}!")
}}
'''

_GO_SNAKE_TEMPLATE = r'''package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/gdamore/tcell/v2"
)

const (
	cell  = 20
	width = 30
	height = 20
	tickMs = 120
)

type Point struct{ X, Y int }

var (
	screen   tcell.Screen
	snake    []Point
	food     Point
	dir      = Point{1, 0}
	nextDir  = Point{1, 0}
	score    int
	gameOver bool
)

func main() {
	var err error
	screen, err = tcell.NewScreen()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	if err := screen.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	defer screen.Fini()

	screen.SetStyle(tcell.StyleDefault.Background(tcell.ColorBlack).Foreground(tcell.ColorWhite))
	screen.EnableMouse()
	rand.Seed(time.Now().UnixNano())

	reset()
	go inputLoop()
	gameLoop()
}

func reset() {
	snake = []Point{{width / 2, height / 2}, {width/2 - 1, height / 2}}
	food = newFood()
	dir = Point{1, 0}
	nextDir = Point{1, 0}
	score = 0
	gameOver = false
}

func newFood() Point {
	for {
		p := Point{rand.Intn(width), rand.Intn(height)}
		hit := false
		for _, s := range snake {
			if s == p {
				hit = true
				break
			}
		}
		if !hit {
			return p
		}
	}
}

func inputLoop() {
	for {
		ev := screen.PollEvent()
		switch ev := ev.(type) {
		case *tcell.EventKey:
			if gameOver && ev.Key() == tcell.KeyRune && ev.Rune() == ' ' {
				reset()
				continue
			}
			switch ev.Key() {
			case tcell.KeyUp, tcell.KeyRune:
				if ev.Rune() == 'w' || ev.Rune() == 'W' {
					if dir.Y != 1 { nextDir = Point{0, -1} }
				}
			case tcell.KeyDown:
				if dir.Y != -1 { nextDir = Point{0, 1} }
			case tcell.KeyLeft:
				if dir.X != 1 { nextDir = Point{-1, 0} }
			case tcell.KeyRight:
				if dir.X != -1 { nextDir = Point{1, 0} }
			case tcell.KeyRune:
				switch ev.Rune() {
				case 'w', 'W':
					if dir.Y != 1 { nextDir = Point{0, -1} }
				case 's', 'S':
					if dir.Y != -1 { nextDir = Point{0, 1} }
				case 'a', 'A':
					if dir.X != 1 { nextDir = Point{-1, 0} }
				case 'd', 'D':
					if dir.X != -1 { nextDir = Point{1, 0} }
				}
			}
		}
	}
}

func gameLoop() {
	ticker := time.NewTicker(time.Duration(tickMs) * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		if !gameOver {
			dir = nextDir
			head := snake[0]
			newHead := Point{head.X + dir.X, head.Y + dir.Y}
			if newHead.X < 0 || newHead.X >= width || newHead.Y < 0 || newHead.Y >= height {
				gameOver = true
			} else {
				for _, s := range snake {
					if s == newHead {
						gameOver = true
						break
					}
				}
			}
			if !gameOver {
				snake = append([]Point{newHead}, snake...)
				if newHead == food {
					score++
					food = newFood()
				} else {
					snake = snake[:len(snake)-1]
				}
			}
		}
		draw()
		if gameOver {
			time.Sleep(3 * time.Second)
			return
		}
	}
}

func draw() {
	screen.Clear()
	// Draw food
	foodStyle := tcell.StyleDefault.Foreground(tcell.ColorRed)
	screen.SetContent(food.X*2, food.Y, '█', nil, foodStyle)

	// Draw snake
	for i, s := range snake {
		var style tcell.Style
		if i == 0 {
			style = tcell.StyleDefault.Foreground(tcell.ColorGreen)
		} else {
			style = tcell.StyleDefault.Foreground(tcell.ColorLightGreen)
		}
		screen.SetContent(s.X*2, s.Y, '█', nil, style)
	}

	// Score
	scoreStr := fmt.Sprintf("Score: %d", score)
	for i, r := range scoreStr {
		screen.SetContent(i, height, r, nil, tcell.StyleDefault.Foreground(tcell.ColorWhite))
	}
	if gameOver {
		msg := "Game Over - press Space"
		for i, r := range msg {
			screen.SetContent(width - len(msg)/2 + i, height/2, r, nil, tcell.StyleDefault.Foreground(tcell.ColorYellow))
		}
	}
	screen.Show()
}
'''

_PYTHON_SNAKE_TEMPLATE = '''import random
import tkinter as tk

CELL = 20
WIDTH = 30
HEIGHT = 20
TICK_MS = 120


class SnakeGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Python Snake")
        self.canvas = tk.Canvas(
            self.root,
            width=WIDTH * CELL,
            height=HEIGHT * CELL,
            bg="#111827",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.root.bind("<KeyPress>", self.on_key)
        self.reset()

    def reset(self):
        self.snake = [(WIDTH // 2, HEIGHT // 2), (WIDTH // 2 - 1, HEIGHT // 2)]
        self.direction = (1, 0)
        self.pending_direction = self.direction
        self.food = self.new_food()
        self.score = 0
        self.game_over = False
        self.tick()

    def new_food(self):
        while True:
            food = (random.randrange(WIDTH), random.randrange(HEIGHT))
            if food not in self.snake:
                return food

    def on_key(self, event):
        keys = {
            "Up": (0, -1),
            "Down": (0, 1),
            "Left": (-1, 0),
            "Right": (1, 0),
            "w": (0, -1),
            "s": (0, 1),
            "a": (-1, 0),
            "d": (1, 0),
        }
        if event.keysym == "space" and self.game_over:
            self.reset()
            return
        next_dir = keys.get(event.keysym)
        if next_dir and (next_dir[0] != -self.direction[0] or next_dir[1] != -self.direction[1]):
            self.pending_direction = next_dir

    def tick(self):
        if not self.game_over:
            self.direction = self.pending_direction
            head_x, head_y = self.snake[0]
            dx, dy = self.direction
            new_head = (head_x + dx, head_y + dy)

            hit_wall = not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT)
            hit_self = new_head in self.snake
            if hit_wall or hit_self:
                self.game_over = True
            else:
                self.snake.insert(0, new_head)
                if new_head == self.food:
                    self.score += 1
                    self.food = self.new_food()
                else:
                    self.snake.pop()

        self.draw()
        self.root.after(TICK_MS, self.tick)

    def draw(self):
        self.canvas.delete("all")
        fx, fy = self.food
        self.canvas.create_oval(
            fx * CELL + 3,
            fy * CELL + 3,
            (fx + 1) * CELL - 3,
            (fy + 1) * CELL - 3,
            fill="#ef4444",
            outline="",
        )
        for index, (x, y) in enumerate(self.snake):
            color = "#22c55e" if index else "#84cc16"
            self.canvas.create_rectangle(
                x * CELL + 1,
                y * CELL + 1,
                (x + 1) * CELL - 1,
                (y + 1) * CELL - 1,
                fill=color,
                outline="",
            )
        self.canvas.create_text(10, 10, anchor="nw", fill="white", text=f"Score: {self.score}")
        if self.game_over:
            self.canvas.create_text(
                WIDTH * CELL // 2,
                HEIGHT * CELL // 2,
                fill="white",
                font=("Arial", 22, "bold"),
                text="Game Over - press Space",
            )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    SnakeGame().run()
'''

_JAVA_SNAKE_TEMPLATE = '''import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.Timer;

public class SnakeGame extends JPanel {
    private static final int CELL = 20;
    private static final int WIDTH = 30;
    private static final int HEIGHT = 20;

    private final Random random = new Random();
    private final List<Point> snake = new ArrayList<>();
    private Point food;
    private int dx = 1;
    private int dy = 0;
    private int nextDx = 1;
    private int nextDy = 0;
    private int score = 0;
    private boolean gameOver = false;

    public SnakeGame() {
        setPreferredSize(new Dimension(WIDTH * CELL, HEIGHT * CELL));
        setBackground(new Color(17, 24, 39));
        setFocusable(true);
        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent event) {
                handleKey(event.getKeyCode());
            }
        });
        reset();
        new Timer(120, event -> tick()).start();
    }

    private void reset() {
        snake.clear();
        snake.add(new Point(WIDTH / 2, HEIGHT / 2));
        snake.add(new Point(WIDTH / 2 - 1, HEIGHT / 2));
        dx = nextDx = 1;
        dy = nextDy = 0;
        score = 0;
        gameOver = false;
        food = newFood();
    }

    private Point newFood() {
        Point point;
        do {
            point = new Point(random.nextInt(WIDTH), random.nextInt(HEIGHT));
        } while (snake.contains(point));
        return point;
    }

    private void handleKey(int key) {
        if (key == KeyEvent.VK_SPACE && gameOver) {
            reset();
            repaint();
            return;
        }
        int candidateDx = nextDx;
        int candidateDy = nextDy;
        if (key == KeyEvent.VK_UP || key == KeyEvent.VK_W) {
            candidateDx = 0;
            candidateDy = -1;
        } else if (key == KeyEvent.VK_DOWN || key == KeyEvent.VK_S) {
            candidateDx = 0;
            candidateDy = 1;
        } else if (key == KeyEvent.VK_LEFT || key == KeyEvent.VK_A) {
            candidateDx = -1;
            candidateDy = 0;
        } else if (key == KeyEvent.VK_RIGHT || key == KeyEvent.VK_D) {
            candidateDx = 1;
            candidateDy = 0;
        }
        if (candidateDx != -dx || candidateDy != -dy) {
            nextDx = candidateDx;
            nextDy = candidateDy;
        }
    }

    private void tick() {
        if (!gameOver) {
            dx = nextDx;
            dy = nextDy;
            Point head = snake.get(0);
            Point next = new Point(head.x + dx, head.y + dy);
            boolean hitWall = next.x < 0 || next.x >= WIDTH || next.y < 0 || next.y >= HEIGHT;
            boolean hitSelf = snake.contains(next);
            if (hitWall || hitSelf) {
                gameOver = true;
            } else {
                snake.add(0, next);
                if (next.equals(food)) {
                    score++;
                    food = newFood();
                } else {
                    snake.remove(snake.size() - 1);
                }
            }
        }
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.setColor(new Color(239, 68, 68));
        g.fillOval(food.x * CELL + 3, food.y * CELL + 3, CELL - 6, CELL - 6);
        for (int i = 0; i < snake.size(); i++) {
            Point part = snake.get(i);
            g.setColor(i == 0 ? new Color(132, 204, 22) : new Color(34, 197, 94));
            g.fillRect(part.x * CELL + 1, part.y * CELL + 1, CELL - 2, CELL - 2);
        }
        g.setColor(Color.WHITE);
        g.drawString("Score: " + score, 10, 18);
        if (gameOver) {
            g.setFont(new Font("Arial", Font.BOLD, 22));
            g.drawString("Game Over - press Space", WIDTH * CELL / 2 - 130, HEIGHT * CELL / 2);
        }
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Java Snake");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(false);
        frame.add(new SnakeGame());
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
'''


def _workspace_has_content(
    tool_results: list[dict[str, Any]],
) -> bool:
    """Check whether the workspace directory has any files in it."""
    for r in tool_results:
        if not isinstance(r, dict):
            continue
        if r.get("success") and r.get("action") in ("mkdir", "create_directory"):
            res = r.get("result", {}) or {}
            if isinstance(res, dict):
                dir_path = res.get("path", "") or res.get("directory", "") or ""
                if dir_path:
                    p = Path(dir_path)
                    if p.exists() and p.is_dir():
                        return any(p.iterdir())
    return False
