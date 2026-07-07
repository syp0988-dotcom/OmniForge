"""CapabilityAnalyzer — LLM-based capability detection.

Determines which capabilities/tools are needed to accomplish the user's goal.
No regex or rule-based matching — purely LLM-driven.

Output (stored in state["capability_analysis"])::

    {
        "planning": true,
        "filesystem": true,
        "workspace": true,
        "knowledge": true,
        "coding": true,
        "python": false,
        "git": true,
        "browser": false,
        "mcp": false,
        "skills": [],
        "reasoning": "创建图书管理系统需要文件操作、编码和版本控制"
    }
"""

from __future__ import annotations

import json
from typing import Any

from agentflow.agents.base import AgentProtocol
from agentflow.services.llm_service import get_llm_service
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("capability_analyzer")

SYSTEM_PROMPT = """你是一个能力分析器（Capability Analyzer）。你的职责是根据用户的目标，决定需要启用哪些能力和工具。

不要使用规则或关键词匹配。完全基于对目标的理解来推理。

输出 JSON 格式（不要包含其他文字）：

{
    "planning": true/false,
    "filesystem": true/false,
    "workspace": true/false,
    "knowledge": true/false,
    "coding": true/false,
    "python": true/false,
    "git": true/false,
    "browser": true/false,
    "mcp": true/false,
    "skills": [],
    "reasoning": "简短解释为什么需要这些能力"
}

能力说明：
- planning:     需要制定多步骤执行计划（几乎所有任务都需要）
- filesystem:   需要创建、读取、修改文件或目录
- workspace:    需要管理工作区
- knowledge:    需要从知识库检索参考信息
- coding:       需要编写代码（任何语言）
- python:       需要执行 Python 代码
- git:          需要版本控制操作
- browser:      需要浏览网页
- mcp:          需要 MCP 工具
- skills:       需要加载的特定技能列表（通常为空）

判断逻辑：
- 创建项目、构建应用 → planning, filesystem, coding, git (if project)
- 知识问答、概念解释 → knowledge (if relevant to KB), possibly search
- 调试代码 → coding, python (if Python)
- 搜索信息 → browser or search-related tools
- 翻译、润色 → no special capabilities needed (planning only)
"""


class CapabilityAnalyzer(AgentProtocol):
    """LLM-based capability detection. No regex, no rules."""

    def __init__(self) -> None:
        self._llm = get_llm_service()

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        """Analyze the goal and determine needed capabilities."""
        goal_analysis = state.get("goal_analysis", {})
        if isinstance(goal_analysis, dict):
            goal = goal_analysis.get("goal", state.get("question", ""))
            goal_type = goal_analysis.get("goal_type", "other")
        else:
            goal = state.get("question", "")
            goal_type = "other"

        capabilities = self._analyze_capabilities(goal, goal_type)

        state["capability_analysis"] = capabilities

        logger.info(
            "Capabilities: planning=%s filesystem=%s coding=%s git=%s knowledge=%s — %s",
            capabilities.get("planning", False),
            capabilities.get("filesystem", False),
            capabilities.get("coding", False),
            capabilities.get("git", False),
            capabilities.get("knowledge", False),
            capabilities.get("reasoning", ""),
        )
        return state

    def _analyze_capabilities(self, goal: str, goal_type: str) -> dict[str, Any]:
        """Call LLM to determine needed capabilities."""
        if not goal:
            return self._default_capabilities()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户目标：{goal}\n"
                    f"目标类型：{goal_type}\n\n"
                    "请分析完成这个目标需要哪些能力，输出 JSON。"
                ),
            },
        ]

        try:
            raw = self._llm.complete(messages=messages)
            parsed = self._parse_json(raw)
            if parsed:
                # Ensure all fields exist
                caps = self._default_capabilities()
                caps.update(parsed)
                return caps
        except Exception as exc:
            logger.warning("CapabilityAnalyzer LLM call failed: %s", exc)

        return self._default_capabilities(goal_type)

    def _parse_json(self, raw: str) -> dict[str, Any] | None:
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
    def _default_capabilities(goal_type: str = "other") -> dict[str, Any]:
        """Fallback capabilities based on goal_type (minimal heuristic fallback)."""
        base = {
            "planning": True,
            "filesystem": False,
            "workspace": False,
            "knowledge": False,
            "coding": False,
            "python": False,
            "git": False,
            "browser": False,
            "mcp": False,
            "skills": [],
            "reasoning": f"目标类型：{goal_type} 的默认能力（LLM不可用时）",
        }

        # Minimal type-based heuristics as LAST RESORT fallback
        if goal_type in ("project",):
            base.update({
                "filesystem": True,
                "coding": True,
                "git": True,
                "reasoning": "项目创建需要文件操作、编码和版本控制",
            })
        elif goal_type in ("coding", "refactor", "debug"):
            base.update({
                "coding": True,
                "reasoning": f"{goal_type} 需要编码能力",
            })
        elif goal_type == "search":
            base.update({
                "browser": True,
                "reasoning": "搜索需要浏览器能力",
            })
        elif goal_type == "question":
            base.update({
                "knowledge": True,
                "reasoning": "知识问答可能需要知识库检索",
            })

        return base
