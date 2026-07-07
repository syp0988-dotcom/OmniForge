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
"""


class ReflectionAgent(AgentProtocol):
    """LLM-based task queue validation for Dynamic Task Queue Planning."""

    def __init__(self) -> None:
        self._llm = get_llm_service()

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        """Evaluate task results and update the task queue."""
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

        # Non-project: just return done (backward compat)
        if goal_type != "project" or current_queue.is_empty:
            logger.info("Reflection: non-project or empty queue -> done")
            return self._done("非项目目标，无需继续")

        # Evaluate via LLM
        reflection = self._evaluate(
            goal, goal_type, current_queue, tool_results,
        )

        # Apply task updates
        self._apply_reflection(current_queue, reflection)

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
            task = Task(
                task_id=task_id,
                title=cfg.get("title", task_id),
                priority=cfg.get("priority", 50),
                tool=cfg.get("tool", "filesystem"),
                goal=cfg.get("title", task_id),
                input=cfg.get("input", {}),
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
        """Call LLM to evaluate task results and decide next actions."""
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
            raw = self._llm.complete(messages=messages)
            parsed = self._parse_json(raw)
            if parsed:
                return parsed
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
        all_ok = True
        has_failure = False

        # Process tool results and update matching tasks
        for r in results:
            if not isinstance(r, dict):
                continue
            success = r.get("success", False)
            error = r.get("error", "")
            task_name = r.get("action", r.get("goal", ""))

            # Try to find the corresponding task by matching goal/path
            matched = None
            for t in queue.filter(status="running"):
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
            else:
                goal_completed = all_ok and not queue.has_todo
        else:
            goal_completed = False

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
