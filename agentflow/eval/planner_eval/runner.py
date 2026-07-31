"""Planner Eval Runner — 用 Mock LLM 测试 PlannerAgent 的规划准确性。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agentflow.eval.common import MockLLMService, save_results
from agentflow.eval.planner_eval.dataset import PlannerEvalDataset
from agentflow.eval.planner_eval.metrics import compute_all, aggregate
from agentflow.tools.registry import ToolRegistry

logger = logging.getLogger("planner_eval")


class PlannerEvalRunner:
    """规划器评估运行器。

    对确定性路径（snake、docx），不注入 Mock LLM，直接走 planner 内建逻辑。
    对 LLM 驱动路径，替换 ``_llm_service`` 单例为 MockLLMService。
    """

    def __init__(self, dataset: PlannerEvalDataset, registry: ToolRegistry | None = None) -> None:
        self.dataset = dataset
        self.registry = registry or ToolRegistry()
        self.per_sample: list[dict[str, Any]] = []
        self.summary: dict[str, float] = {}

    def run(self, verbose: bool = True) -> dict[str, Any]:
        """遍历数据集，逐条运行 PlannerAgent 并计算指标。"""
        from agentflow.agents.planner.agent import PlannerAgent

        all_metrics: list[dict[str, float]] = []

        for i, sample in enumerate(self.dataset, 1):
            sid = sample.get("id", f"plan_{i}")
            if verbose:
                print(f"[{i}/{len(self.dataset)}] {sample['question'][:60]} ...", end=" ")

            # 构造 state
            state = {
                "question": sample["question"],
                "goal_analysis": sample["goal_analysis"],
                "task_queue": [],
            }

            bypass = sample.get("bypass_llm", False)
            mock_resp = sample.get("mock_llm_response")

            if bypass:
                # 确定性路径或降级模式
                if mock_resp and mock_resp.get("type") == "degraded":
                    state["_degraded"] = {"planner"}
                result_state = PlannerAgent(registry=self.registry).run(state)
            elif mock_resp:
                # 注入 Mock LLM
                result_state = self._run_with_mock_llm(state, mock_resp)
            else:
                # 尝试确定性路径触发（snake/docx 关键字）
                result_state = PlannerAgent(registry=self.registry).run(state)

            # 提取实际输出
            plan = result_state.get("plan")
            task_queue = result_state.get("task_queue", [])
            category = result_state.get("category", "")

            actual_tools = []
            actual_actions = []
            if plan and hasattr(plan, "tasks"):
                for t in plan.tasks:
                    actual_tools.append(t.tool if hasattr(t, "tool") else t.get("tool", ""))
                    action = t.input.get("action", "") if hasattr(t, "input") else t.get("input", {}).get("action", "")
                    if action:
                        actual_actions.append(action)
            else:
                for t in task_queue:
                    actual_tools.append(t.get("tool", ""))
                    action = t.get("input", {}).get("action", "") if isinstance(t.get("input"), dict) else ""
                    if action:
                        actual_actions.append(action)

            actual_count = len(task_queue)
            actual_completed = plan.goal_completed if plan and hasattr(plan, "goal_completed") else False

            metrics = compute_all(
                actual_tools, actual_actions, actual_count,
                category, actual_completed, sample,
            )
            all_metrics.append(metrics)

            self.per_sample.append({
                "sample_id": sid,
                "question": sample["question"],
                "actual_tools": actual_tools,
                "expected_tools": sample["expected_tools"],
                "actual_actions": actual_actions,
                "expected_actions": sample["expected_actions"],
                "actual_task_count": actual_count,
                "expected_range": sample["expected_task_count_range"],
                "actual_goal_type": category,
                "expected_goal_type": sample["expected_goal_type"],
                "metrics": metrics,
            })

            if verbose:
                acc = (
                    metrics["tool_acc"],
                    metrics["action_acc"],
                    metrics["task_count_acc"],
                )
                print(f"tool={acc[0]:.2f} action={acc[1]:.2f} count={acc[2]:.2f}")

        self.summary = aggregate(all_metrics)
        return {
            "summary": self.summary,
            "per_sample": self.per_sample,
            "config": {
                "total_samples": len(self.dataset),
            },
        }

    def _run_with_mock_llm(self, state: dict, mock_resp: dict) -> dict:
        """注入 MockLLMService 后运行 PlannerAgent。"""
        import agentflow.services.llm_service as llm_mod
        from agentflow.agents.planner.agent import PlannerAgent

        original_llm = llm_mod._llm_service

        # 按类型构建 mock
        resp_type = mock_resp.get("type", "json")
        if resp_type == "tool_calls":
            mock_llm = MockLLMService(tool_call_responses=[mock_resp])
        elif resp_type == "degraded":
            mock_llm = MockLLMService(degraded=True)
        else:
            # json 类型
            content = mock_resp.get("content", {})
            if isinstance(content, dict):
                import json
                content_str = json.dumps(content, ensure_ascii=False)
            else:
                content_str = str(content)
            mock_llm = MockLLMService(responses={"default": content_str})

        try:
            llm_mod._llm_service = mock_llm
            planner = PlannerAgent(registry=self.registry)
            planner._llm = mock_llm
            result = planner.run(state)
            return result
        finally:
            llm_mod._llm_service = original_llm

    def save_results(self, path: str | Path) -> None:
        save_results(path, self.summary, self.per_sample, {
            "total_samples": len(self.dataset),
        })
