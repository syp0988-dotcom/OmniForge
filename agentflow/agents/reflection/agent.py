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
        if not reflection.get("goal_completed") and not current_queue.has_todo:
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
                # Nothing to generate -- workspace might be complete or LLM failed.
                # Use a stuck counter to prevent infinite loops.
                stuck_rounds = int(state.get("_stuck_rounds", 0)) + 1
                state["_stuck_rounds"] = stuck_rounds
                ws_has_content = _workspace_has_content(tool_results)
                if ws_has_content and stuck_rounds >= 3:
                    reflection["goal_completed"] = True
                    reflection["reason"] = "工作区已满足目标要求"
                    logger.info(
                        "Reflection fallback: no more content after %d rounds, "
                        "workspace has content -> completed",
                        stuck_rounds,
                    )
                elif not ws_has_content and stuck_rounds >= 5:
                    reflection["goal_completed"] = True
                    reflection["reason"] = "无法生成文件内容，终止循环"
                    logger.warning(
                        "Reflection fallback: empty workspace after %d rounds, giving up",
                        stuck_rounds,
                    )
                else:
                    logger.info(
                        "Reflection fallback: no content (round %d, ws_has_content=%s)",
                        stuck_rounds, ws_has_content,
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
    wants_snake = "snake" in text or "贪吃蛇" in goal
    wants_files = any(token in goal for token in ("文件", "创建", "新建", "生成", "写"))

    if not wants_files:
        return []

    dir_name = _fallback_project_dir(goal, project_name)
    existing = _existing_file_names(results, dir_name)
    specs: list[tuple[str, str]] = []

    if wants_snake and wants_python:
        specs.append(("python_snake.py", _PYTHON_SNAKE_TEMPLATE))
    if wants_snake and wants_java:
        specs.append(("SnakeGame.java", _JAVA_SNAKE_TEMPLATE))

    if not specs:
        if wants_python:
            specs.append(("main.py", _BASIC_MAIN_PY.format(project_name=project_name or "project")))
        if wants_java:
            specs.append(("Main.java", _BASIC_MAIN_JAVA))

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
    return all(not isinstance(r, dict) or r.get("success", False) for r in results)


def _generate_stuck_tasks(
    goal: str,
    project_name: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate fallback tasks when the queue is stuck.

    Two-phase approach to avoid embedding large code content in JSON
    (which DeepSeek's output frequently corrupts):

      Phase 1: Ask LLM for a small JSON array with just file paths.
      Phase 2: For each file, ask LLM for raw code (no JSON wrapping).

    This is slower (N+1 LLM calls) but avoids the content-in-JSON
    corruption that plagued the single-call approach.
    """
    deterministic = _deterministic_file_fallback_tasks(goal, project_name, results)
    if deterministic:
        return deterministic

    # ── Find the directory from the last successful mkdir result ──────
    dir_path = None
    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("success") and r.get("action") in ("mkdir", "create_directory"):
            path = ""
            res = r.get("result", {}) or {}
            if isinstance(res, dict):
                path = res.get("path", "") or res.get("directory", "") or ""
            elif isinstance(res, str):
                path = res
            if not path:
                msg = r.get("message", "")
                for kw in ("Directory created: ", "目录已创建: "):
                    if kw in msg:
                        path = msg.split(kw, 1)[-1].strip()
                        break
            if path:
                dir_path = path

    if not dir_path:
        dir_path = _fallback_project_dir(goal, project_name)
        logger.info("Rule fallback: no mkdir result found, using %s", dir_path)

    project_path = Path(dir_path)
    if project_path.exists() and not project_path.is_dir():
        logger.warning("Rule fallback: fallback path %r is not a directory", dir_path)
        return []

    dir_name = project_path.name

    existing_files = set()
    if project_path.exists():
        for entry in project_path.iterdir():
            if entry.is_file():
                existing_files.add(entry.name)

    file_list = "\n".join(f"  - {f}" for f in sorted(existing_files)) if existing_files else "  (空)"

    llm = get_llm_service()

    batch_configs = _batch_code_fallback(goal, dir_name, project_path, existing_files, llm)
    if batch_configs:
        return batch_configs

    # ═══════════════════════════════════════════════════════════════════
    # Fallback: ask LLM for file list, then generate each file separately.
    # ═══════════════════════════════════════════════════════════════════
    structure_prompt = (
        f"用户目标：{goal}\n\n"
        f"工作目录名：{dir_name}\n"
        f"已有文件：\n{file_list}\n\n"
        f"请分析还需要创建哪些文件。\n"
        f"以 JSON 数组格式输出（不要其他文字）：\n"
        f'[\n'
        f'  {{\n'
        f'    "path": "{dir_name}/example.py",\n'
        f'    "language": "python",\n'
        f'    "description": "文件功能描述"\n'
        f'  }}\n'
        f']\n\n'
        f"要求：\n"
        f"1. 只输出 JSON，不要任何其他文字\n"
        f"2. 跳过已有文件，只创建缺失的文件\n"
        f"3. path 必须以 '{dir_name}/' 开头\n"
        f"4. 每个文件给出合理的路径、语言和描述"
    )

    # Phase 2: per-file content prompts
    content_prompt_tpl = (
        '你是一个代码生成器。请生成以下文件的完整、可直接运行的代码。\n'
        '只输出代码本身，不要任何解释、注释说明或 markdown 格式。\n'
        '\n'
        '项目描述：{goal}\n'
        '文件路径：{path}\n'
        '语言：{language}\n'
        '描述：{description}\n'
    )

    try:
        raw = llm.complete(messages=[{"role": "user", "content": structure_prompt}])
        logger.info("Rule fallback (phase 1): response (first 300)=%s", raw[:300])

        # Try to parse as JSON array, with common repairs
        text = raw.strip()
        candidates = [text]
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

        entries = None
        for c in candidates:
            try:
                entries = json.loads(c)
                break
            except json.JSONDecodeError:
                fixed = _fix_json_newlines(c)
                if fixed != c:
                    try:
                        entries = json.loads(fixed)
                        break
                    except json.JSONDecodeError:
                        pass

        if not entries or not isinstance(entries, list):
            logger.warning("Rule fallback (phase 1): LLM response not parseable, trying batch code gen")
            # ── Batch fallback: ask for ALL code at once with markers ──
            return _batch_code_fallback(goal, dir_name, project_path, existing_files, llm)

        # ═══════════════════════════════════════════════════════════════
        # Phase 2:  Generate content for each file (raw code, no JSON)
        # ═══════════════════════════════════════════════════════════════
        configs = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            language = str(item.get("language", "")).strip()
            description = str(item.get("description", "")).strip()

            if not path:
                continue
            fname = Path(path).name
            if fname in existing_files:
                continue

            content_prompt = content_prompt_tpl.format(
                goal=goal, path=path,
                language=language or "python",
                description=description or path,
            )

            logger.info("Rule fallback (phase 2): generating %s ...", path)
            try:
                content = llm.complete(messages=[{"role": "user", "content": content_prompt}])
            except Exception as exc:
                logger.warning("Rule fallback (phase 2): LLM failed for %s: %s", path, exc)
                continue

            content = content.strip()

            # Strip markdown code fences if the LLM added them anyway
            if content.startswith("```"):
                first_nl = content.find("\n")
                if first_nl != -1:
                    content = content[first_nl + 1:]
            if content.endswith("```"):
                content = content[:-3].strip()
            elif content.endswith("```\n"):
                content = content[:-4].strip()

            if len(content) < 50:
                logger.warning(
                    "Rule fallback (phase 2): content for %s too short (%d chars), skipping",
                    path, len(content),
                )
                continue

            task_id = f"create_{fname.replace('.', '_')}"
            configs.append({
                "task_id": task_id,
                "title": f"创建 {fname}",
                "priority": 95,
                "tool": "filesystem",
                "input": {
                    "action": "write_file",
                    "path": path,
                    "content": content,
                },
            })

        if configs:
            logger.info(
                "Rule fallback: created %d task(s) via phase-2 LLM: %s",
                len(configs), [c.get("input", {}).get("path", "?") for c in configs],
            )
            return configs

        # ── Phase 2 produced no valid content → try batch fallback ──
        logger.info("Rule fallback (phase 2): no valid configs, trying batch code gen")
        return _batch_code_fallback(goal, dir_name, project_path, existing_files, llm)

    except Exception as exc:
        logger.warning("Rule fallback: LLM call failed (%s), using minimal fallback", exc)
        return _minimal_fallback_tasks(dir_name, existing_files)


def _batch_code_fallback(
    goal: str,
    dir_name: str,
    project_path: Path,
    existing_files: set[str],
    llm: Any,
) -> list[dict[str, Any]]:
    """Generate ALL file code in a single LLM call using marker delimiters.

    Falls back to per-file generation if the batch output is truncated
    or unparseable.
    """
    batch_prompt = (
        f'请为以下项目生成所有缺失文件的完整代码。\n\n'
        f'项目描述：{goal}\n'
        f'工作目录：{dir_name}\n\n'
        f'使用以下格式（等号和文件名作为分隔符）：\n'
        f'===== path/to/file.ext =====\n'
        f'完整代码...\n'
        f'===== next/file.py =====\n'
        f'完整代码...\n\n'
        f'要求：\n'
        f'1. 每个文件以 ===== 相对路径 ===== 开头\n'
        f'2. 代码必须完整可直接运行\n'
        f'3. 不要使用 markdown 代码块标记\n'
        f'4. 跳过已有文件（{", ".join(sorted(existing_files)) or "无"}）'
    )

    try:
        raw = llm.complete(messages=[{"role": "user", "content": batch_prompt}])
        logger.info("Rule fallback (batch): response (first 300)=%s", raw[:300])

        # Parse markers: ===== path ===== ...code...
        marker_re = re.compile(r'^=====\s+(.+?)\s+=====\s*$', re.MULTILINE)
        parts = marker_re.split(raw.strip())
        # parts: [before, path1, code1, path2, code2, ...]
        # Skip parts[0] which is text before the first marker

        configs = []
        for i in range(1, len(parts) - 1, 2):
            path = parts[i].strip()
            code = parts[i + 1].strip()

            if not path or not code:
                continue
            fname = Path(path).name
            if fname in existing_files:
                continue
            if len(code) < 50:
                continue

            configs.append({
                "task_id": f"create_{fname.replace('.', '_')}",
                "title": f"创建 {fname}",
                "priority": 95,
                "tool": "filesystem",
                "input": {
                    "action": "write_file",
                    "path": path,
                    "content": code,
                },
            })

        if configs:
            logger.info(
                "Rule fallback (batch): created %d task(s): %s",
                len(configs), [c.get("input", {}).get("path", "?") for c in configs],
            )
            return configs
        logger.info("Rule fallback (batch): no valid configs, using per-file code gen")

        # ── Per-file fallback: one LLM call per file ──
        # If the batch didn't produce results, let the LLM decide what
        # files to create by asking for individual Python and Java files.
        return _per_file_code_fallback(goal, dir_name, existing_files, llm)

    except Exception as exc:
        logger.warning("Rule fallback (batch): failed (%s), trying per-file gen", exc)
        return _per_file_code_fallback(goal, dir_name, existing_files, llm)


def _per_file_code_fallback(
    goal: str,
    dir_name: str,
    existing_files: set[str],
    llm: Any,
) -> list[dict[str, Any]]:
    """Generate one LLM call per file as the final resort."""

    # Ask LLM what files to create (single-call, no content)
    list_prompt = (
        f'项目：{goal}\n'
        f'工作目录：{dir_name}\n'
        f'已有文件：{", ".join(sorted(existing_files)) or "无"}\n\n'
        f'还需要创建哪些文件？以 JSON 数组格式输出，每项包含 path 和 language：\n'
        f'[{{"path": "{dir_name}/file.py", "language": "python"}}]\n'
        f'只输出 JSON。'
    )

    try:
        raw = llm.complete(messages=[{"role": "user", "content": list_prompt}])
        text = raw.strip()
        # Extract JSON array from response
        for marker in ("```json", "```JSON", "```"):
            start = text.find(marker)
            if start != -1:
                end = text.find("```", start + len(marker))
                content = text[start + len(marker):end].strip() if end != -1 else text[start + len(marker):].strip()
                if content:
                    try:
                        text = json.loads(content) if isinstance(json.loads(content), list) else text
                    except json.JSONDecodeError:
                        pass
                    break

        try:
            entries = json.loads(text) if isinstance(text, str) else text
            if not isinstance(entries, list):
                entries = []
        except json.JSONDecodeError:
            entries = []

        if not entries:
            # Last resort: hardcoded snake game prompt
            entries = [
                {"path": f"{dir_name}/snake.py", "language": "python"},
                {"path": f"{dir_name}/SnakeGame.java", "language": "java"},
            ]

        configs = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            language = str(item.get("language", "python")).strip()
            if not path:
                continue
            fname = Path(path).name
            if fname in existing_files:
                continue

            gen_prompt = (
                f'生成以下文件的完整可直接运行的代码。只输出代码，不要任何其他文字。\n'
                f'文件：{path}\n'
                f'语言：{language}\n'
                f'项目：{goal}\n'
            )

            content = llm.complete(messages=[{"role": "user", "content": gen_prompt}])
            content = content.strip()

            # Strip code fences
            if content.startswith("```"):
                idx = content.find("\n")
                if idx != -1:
                    content = content[idx:]
            content = content.strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            if len(content) < 50:
                continue

            configs.append({
                "task_id": f"create_{fname.replace('.', '_')}",
                "title": f"创建 {fname}",
                "priority": 95,
                "tool": "filesystem",
                "input": {
                    "action": "write_file",
                    "path": path,
                    "content": content,
                },
            })

        if configs:
            logger.info(
                "Rule fallback (per-file): created %d task(s): %s",
                len(configs), [c.get("input", {}).get("path", "?") for c in configs],
            )
        return configs

    except Exception as exc:
        logger.warning("Rule fallback (per-file): failed (%s)", exc)
        return []


def _minimal_fallback_tasks(
    dir_name: str,
    existing_files: set[str],
) -> list[dict[str, Any]]:
    """Absolute last-resort: create a simple main.py if it doesn't exist."""
    if "main.py" in existing_files:
        return []
    logger.info("Rule (minimal) fallback: creating basic main.py")
    return [
        {
            "task_id": "create_main_py",
            "title": "创建 main.py",
            "priority": 50,
            "tool": "filesystem",
            "input": {
                "action": "write_file",
                "path": f"{dir_name}/main.py",
                "content": _BASIC_MAIN_PY,
            },
        },
    ]


# ── Last-resort minimal content template ──────────────────────────────

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
