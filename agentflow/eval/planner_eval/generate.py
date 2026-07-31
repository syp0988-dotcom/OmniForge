"""LLM 驱动的规划器测试用例生成器。

从工具注册表的 action 定义出发，用 LLM 生成
(用户问题, 预期工具, 预期动作, Mock LLM 响应) 四元组。
"""

from __future__ import annotations

import json
import logging
import re

from agentflow.eval.planner_eval.dataset import PlannerEvalDataset
from agentflow.services.llm_service import LLMService

logger = logging.getLogger("planner_eval.generator")

_GENERATE_PROMPT = """\
你是一个代码助手的测试用例生成器。根据以下可用的工具列表，生成 {count} 个用户问题。

可用的工具及其操作：
{tool_actions}

每条测试用例需要包含：
1. question: 用户可能提出的开发任务问题（中文，自然口语化）
2. expected_tools: 完成此任务需要的工具名列表
3. expected_actions: 完成此任务需要的操作名列表
4. expected_task_count_range: [最少任务数, 最多任务数]
5. mock_llm_response: 一个模拟的 LLM 规划响应（JSON 格式，包含 tasks 数组）

每个 task 对象格式：
  {{"task_id": "t1", "title": "任务标题", "tool": "工具名",
    "goal": "操作描述", "input": {{"action": "操作名", ...参数}}}}

要求：
1. 问题多样化，覆盖单文件操作、多文件项目、重构等场景
2. 返回纯 JSON 数组格式：[{{"question": "...", "expected_tools": [...], "expected_actions": [...], "expected_task_count_range": [min, max], "mock_llm_response": {{...}}}}, ...]

JSON:"""


def _extract_json_array(text: str) -> list[dict]:
    """从 LLM 回复中提取 JSON 数组。"""
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning("Could not extract JSON array from LLM response")
    return []


def generate_from_tools(
    registry,
    count: int = 20,
    llm: LLMService | None = None,
) -> PlannerEvalDataset:
    """从工具注册表生成规划器测试用例。

    Parameters
    ----------
    registry : ToolRegistry
        已注册工具的注册表。
    count : int
        期望生成的测试用例数。
    llm : LLMService | None
        LLM 服务实例。

    Returns
    -------
    PlannerEvalDataset
    """
    if llm is None:
        llm = LLMService()

    if not llm.client:
        raise RuntimeError("LLM client not configured.")

    # 构建工具列表文本
    tool_lines = []
    for tool_name in registry.list_tools():
        tool = registry.get(tool_name)
        if tool is None:
            continue
        for action_name, action_def in tool.actions().items():
            tool_lines.append(f"  - {tool_name}.{action_name}: {action_def.get('description', '')}")

    tool_actions_text = "\n".join(tool_lines)

    prompt = _GENERATE_PROMPT.format(
        count=count,
        tool_actions=tool_actions_text,
    )

    logger.info("Generating %d planner test cases...", count)

    try:
        response = llm.complete(prompt=prompt)
        cases = _extract_json_array(response)
    except Exception as exc:
        logger.error("Failed to generate planner cases: %s", exc)
        return PlannerEvalDataset()

    dataset = PlannerEvalDataset()
    for case in cases:
        if not isinstance(case, dict):
            continue
        question = case.get("question", "")
        if not question:
            continue

        mock_resp = case.get("mock_llm_response", {})
        if isinstance(mock_resp, dict):
            mock_resp["type"] = "json"

        dataset.add(
            question=question,
            expected_tools=case.get("expected_tools", []),
            expected_actions=case.get("expected_actions", []),
            expected_task_count_range=case.get("expected_task_count_range", [1, 5]),
            expected_goal_type=case.get("expected_goal_type", "project"),
            mock_llm_response=mock_resp,
            bypass_llm=False,
        )

    logger.info("Generated %d planner test cases", len(dataset))
    return dataset
