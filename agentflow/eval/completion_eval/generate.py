"""LLM 驱动的端到端任务完成测试用例生成器。

生成多轮任务场景，包含 Planner 和 Reflector 的 mock 响应序列。
"""

from __future__ import annotations

import json
import logging
import re

from agentflow.eval.completion_eval.dataset import CompletionEvalDataset
from agentflow.services.llm_service import LLMService

logger = logging.getLogger("completion_eval.generator")

_GENERATE_PROMPT = """\
你是一个代码助手测试用例生成器。根据可用的工具列表，生成 {count} 个端到端任务完成测试场景。

可用的工具：
{tool_actions}

每个测试场景需要包含：
1. question: 用户问题（中文，自然口语化）
2. expected_completed: true（任务应该能完成）或 false（任务不能完成）
3. min_expected_tasks_done: 最少应完成的任务数
4. goal_analysis: {{"goal": "目标摘要", "goal_type": "coding|project|question", "confidence": 0.9}}
5. planner_responses: 数组，每个元素是 mock 的 Planner LLM 响应，格式为：
   {{"type": "json", "content": {{"goal_completed": false, "tasks": [...], "reasoning": "..."}}}}
   其中每个 task 格式：{{"task_id": "t1", "title": "...", "tool": "filesystem", "goal": "...", "input": {{"action": "...", ...}}}}
6. reflection_responses: 数组，每个元素是 mock 的 Reflector LLM 响应，格式为：
   {{"type": "json", "content": {{"goal_completed": true, "reason": "..."}}}}

要求：
1. 场景多样化：单文件创建、多文件项目、重构、CI配置等
2. 多轮任务：有些任务需要多次规划
3. 降级任务：expected_completed=false 的任务（LLM不可用等）
4. 返回纯 JSON 数组格式

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
    count: int = 15,
    llm: LLMService | None = None,
) -> CompletionEvalDataset:
    """从工具注册表生成任务完成测试用例。

    Parameters
    ----------
    registry : ToolRegistry
        已注册工具的注册表。
    count : int
        期望生成的测试场景数。
    llm : LLMService | None
        LLM 服务实例。

    Returns
    -------
    CompletionEvalDataset
    """
    if llm is None:
        llm = LLMService()

    if not llm.client:
        raise RuntimeError("LLM client not configured.")

    # 构建工具列表
    tool_lines = []
    for tool_name in registry.list_tools():
        tool = registry.get(tool_name)
        if tool is None:
            continue
        for action_name, action_def in tool.actions().items():
            tool_lines.append(f"  - {tool_name}.{action_name}: {action_def.get('description', '')}")

    prompt = _GENERATE_PROMPT.format(
        count=count,
        tool_actions="\n".join(tool_lines),
    )

    logger.info("Generating %d completion test cases...", count)

    try:
        response = llm.complete(prompt=prompt)
        cases = _extract_json_array(response)
    except Exception as exc:
        logger.error("Failed to generate completion cases: %s", exc)
        return CompletionEvalDataset()

    dataset = CompletionEvalDataset()
    for case in cases:
        if not isinstance(case, dict):
            continue
        question = case.get("question", "")
        if not question:
            continue

        dataset.add(
            question=question,
            expected_completed=case.get("expected_completed", True),
            goal_analysis=case.get("goal_analysis", {
                "goal": question,
                "goal_type": "project",
                "confidence": 0.9,
            }),
            planner_responses=case.get("planner_responses", []),
            reflection_responses=case.get("reflection_responses", []),
            min_expected_tasks_done=case.get("min_expected_tasks_done", 1),
            expected_actions=case.get("expected_actions", []),
        )

    logger.info("Generated %d completion test cases", len(dataset))
    return dataset
