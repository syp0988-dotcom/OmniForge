"""Completion Eval Runner — 端到端任务完成率评估。

模拟完整的顺序循环：
    planner → executor → reflector → (loop) → answer

使用 TurnAwareMockLLMService 控制多轮 LLM 响应，
使用 MockWriteTool 模拟文件系统操作。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agentflow.eval.common import (
    MockWriteTool,
    TurnAwareMockLLMService,
    save_results,
)
from agentflow.eval.completion_eval.dataset import CompletionEvalDataset
from agentflow.eval.completion_eval.metrics import compute_all
from agentflow.tools.registry import ToolRegistry

logger = logging.getLogger("completion_eval")


class CompletionEvalRunner:
    """端到端任务完成率评估运行器。

    每条样本运行简化的多轮 Agent 循环。
    """

    def __init__(self, dataset: CompletionEvalDataset) -> None:
        from agentflow.agents.planner.agent import PlannerAgent
        from agentflow.agents.reflection.agent import ReflectionAgent
        from agentflow.graph.executor import Executor

        self.dataset = dataset
        self.registry = ToolRegistry()
        self.registry.register(MockWriteTool())
        self.executor = Executor()
        self.executor_registry = self.registry  # 覆盖 executor 的 registry

        # 预构建 agent 实例（LLM 在每条样本中动态替换）
        self.planner = PlannerAgent(registry=self.registry)
        self.reflector = ReflectionAgent()
        self.per_sample: list[dict[str, Any]] = []
        self.summary: dict = {}

    def run(self, max_turns: int = 5, verbose: bool = True) -> dict[str, Any]:
        """遍历数据集，逐条运行完整多轮循环。"""
        import agentflow.services.llm_service as llm_mod

        original_llm = llm_mod._llm_service
        results: list[dict] = []

        for i, sample in enumerate(self.dataset, 1):
            sid = sample.get("id", f"comp_{i}")
            if verbose:
                print(f"\n[{i}/{len(self.dataset)}] {sample['question'][:80]}")

            try:
                sample_result = self._run_single_sample(sample, max_turns, verbose)
                results.append(sample_result)
                self.per_sample.append(sample_result)
            except Exception as exc:
                logger.exception("Sample %s failed: %s", sid, exc)
                results.append({
                    "sample_id": sid,
                    "question": sample["question"],
                    "goal_completed": False,
                    "tasks_done": 0,
                    "tasks_total": 0,
                    "turns_taken": 0,
                    "error": str(exc),
                })
                self.per_sample.append(results[-1])
            finally:
                llm_mod._llm_service = original_llm

        self.summary = compute_all(results, [s for s in self.dataset])
        return {
            "summary": self.summary,
            "per_sample": self.per_sample,
            "config": {
                "total_samples": len(self.dataset),
                "max_turns": max_turns,
            },
        }

    def _run_single_sample(self, sample: dict, max_turns: int, verbose: bool) -> dict:
        """执行单条样本的完整评估。"""
        import agentflow.services.llm_service as llm_mod

        # 构建 Mock LLM
        mock_llm = TurnAwareMockLLMService(
            planner_responses=sample.get("planner_responses", []),
            reflection_responses=sample.get("reflection_responses", []),
        )

        # 替换全局 LLM 单例和 agent 实例内部的 LLM
        llm_mod._llm_service = mock_llm
        self.planner._llm = mock_llm
        self.reflector._llm = mock_llm

        # 初始 state
        state = {
            "question": sample["question"],
            "goal_analysis": sample["goal_analysis"],
            "task_queue": [],
            "tool_results": [],
            "history": [],
            "_replan_count": 0,
        }

        goal_completed = False
        turn = 0
        tasks_done = 0
        tasks_total = 0

        while turn < max_turns:
            # Planner
            planner_state = self.planner.run(state)
            state.update(planner_state)

            plan = state.get("plan")
            if plan and hasattr(plan, "goal_completed") and plan.goal_completed:
                goal_completed = True
                if verbose:
                    print(f"  Turn {turn}: planner set goal_completed=True")
                break

            task_queue = state.get("task_queue", [])
            if not task_queue:
                if verbose:
                    print(f"  Turn {turn}: no tasks in queue, breaking")
                break

            # Executor — 执行所有 TODO 任务
            for task_dict in task_queue:
                if task_dict.get("status") in ("done", "completed", "failed"):
                    continue
                task_dict["status"] = "running"

                tool_name = task_dict.get("tool", "filesystem")
                action = task_dict.get("input", {}).get("action", task_dict.get("goal", ""))
                kwargs = dict(task_dict.get("input", {}))
                kwargs.pop("action", None)
                kwargs["action"] = action

                result = self.registry.execute_task(tool_name, **kwargs)
                if result.success:
                    task_dict["status"] = "done"
                    tasks_done += 1
                else:
                    task_dict["status"] = "failed"
                state.setdefault("tool_results", []).append(result.to_dict())

            tasks_total = len(task_queue)
            if verbose:
                done = sum(1 for t in task_queue if t.get("status") == "done")
                print(f"  Turn {turn}: {done}/{len(task_queue)} tasks done")

            # Reflector
            reflector_state = self.reflector.run(state)
            state.update(reflector_state)

            reflection_result = state.get("_reflection_result", "")
            if verbose:
                print(f"  Turn {turn}: reflection -> {reflection_result}")

            if reflection_result == "done":
                goal_completed = True
                break
            elif reflection_result == "answer":
                goal_completed = True
                break

            turn += 1
            state["_replan_count"] = state.get("_replan_count", 0) + 1
            mock_llm.reset()  # 为下一轮重置计数器

        return {
            "sample_id": sample.get("id", ""),
            "question": sample["question"],
            "turns_taken": turn + 1,
            "tasks_done": tasks_done,
            "tasks_total": tasks_total,
            "goal_completed": goal_completed,
            "expected_completed": sample["expected_completed"],
            "min_expected_tasks_done": sample.get("min_expected_tasks_done", 0),
            "expected_match": goal_completed == sample["expected_completed"],
        }

    def save_results(self, path: str | Path) -> None:
        save_results(path, self.summary, self.per_sample, {
            "total_samples": len(self.dataset),
        })
