"""GoalAnalyzer — LLM-based goal understanding agent.

Replaces the regex-based QueryRouterAgent. This agent uses an LLM to
understand the user's true goal without relying on keyword patterns.

Output (stored in state["goal_analysis"])::

    {
        "goal": "创建一个完整可运行的图书管理系统",
        "goal_type": "project",
        "expected_outputs": ["project", "source_code", "database", "readme"],
        "priority": "high",
        "confidence": 0.99
    }
"""

from __future__ import annotations

import json
from typing import Any

from agentflow.agents.base import AgentProtocol
from agentflow.services.llm_service import get_llm_service
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("goal_analyzer")

SYSTEM_PROMPT = """你是一个目标分析器（Goal Analyzer）。你的职责是理解用户的真实目标，而不是分类。

请分析用户的输入，理解其背后的真实意图和目标。完全不要使用关键词匹配或规则判断。
仅依靠对语义的深层理解。

输出 JSON 格式（不要包含其他文字）：

{
    "goal": "对用户真实目标的详细描述（清晰、具体）",
    "goal_type": "目标类型",
    "expected_outputs": ["期望得到的输出类型列表"],
    "priority": "优先级",
    "confidence": 置信度 0-1
}

goal_type 必须是以下之一：
- question:    知识问答、寻求解释、概念理解
- project:     创建项目、搭建系统、构建应用（涉及多文件、多步骤）
- coding:      编写代码片段、实现单个功能
- debug:       调试代码、修复Bug、排查问题
- refactor:    重构代码、优化性能、改善结构
- analysis:    分析数据、比较方案、评估结果
- workflow:    多步骤工作流、自动化流程
- search:      搜索实时信息（新闻、天气、价格等）
- document:    生成文档、编写说明
- tool_use:    使用特定工具（git, shell等）
- planning:    制定计划、设计方案
- editing:     编辑修改现有内容
- translation: 翻译
- other:       其他（上述都不匹配时）

expected_outputs 可选值：answer, project, source_code, database, readme, test, docker, api, frontend, backend, config, script, document, plan

判断依据：
- 如果用户要求创建完整的项目/系统/应用（涉及多文件、多目录）→ goal_type="project"
- 如果用户只要求写一段代码、一个函数、一个类 → goal_type="coding"
- 如果用户询问概念、知识、解释 → goal_type="question"
- 完全基于语义理解，不要用关键词匹配"""


class GoalAnalyzer(AgentProtocol):
    """LLM-based goal understanding. No regex, no rule-based classification."""

    def __init__(self) -> None:
        self._llm = get_llm_service()

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        """Analyze the user's question and produce a structured goal."""
        question = str(state.get("question", ""))
        conversation_context = state.get("conversation_context")
        session_state = state.get("session_state")
        continue_mode = bool(state.get("_continue_mode", False))

        # In continue mode, use existing goal from session if available
        existing_goal = None
        if continue_mode and session_state:
            existing_goal = getattr(session_state, "current_goal", None) or (
                session_state.get("current_goal") if isinstance(session_state, dict) else None
            )

        goal = self._analyze_goal(question, conversation_context, continue_mode, existing_goal)

        state["goal_analysis"] = goal
        # Backward compat: set category from goal_type
        state["category"] = goal.get("goal_type", "other")
        state["router"] = {
            "goal_type": goal.get("goal_type", "other"),
            "goal": goal.get("goal", ""),
            "confidence": goal.get("confidence", 0.0),
        }

        logger.info(
            "Goal: type=%s priority=%s confidence=%.2f goal='%s'",
            goal.get("goal_type", "?"),
            goal.get("priority", "normal"),
            goal.get("confidence", 0.0),
            goal.get("goal", "")[:60],
        )
        return state

    def _analyze_goal(
        self,
        question: str,
        conversation_context: Any = None,
        continue_mode: bool = False,
        existing_goal: str | None = None,
    ) -> dict[str, Any]:
        """Call LLM to analyze the user's goal."""
        if not question:
            return self._default_goal()

        # Build context
        ctx_parts = [f"用户输入：{question}"]

        if continue_mode and existing_goal:
            ctx_parts.append(f"\n这是继续对话。当前已有目标：{existing_goal}")
            ctx_parts.append("请根据新的用户输入，判断是继续原有目标还是转向新目标。")

        if conversation_context:
            if isinstance(conversation_context, dict):
                ctx_type = conversation_context.get("type", "")
                summary = conversation_context.get("summary", "")
                if ctx_type:
                    ctx_parts.append(f"\n对话类型：{ctx_type}")
                if summary:
                    ctx_parts.append(f"对话摘要：{summary}")
            else:
                cc_type = getattr(conversation_context, "type", "")
                cc_summary = getattr(conversation_context, "summary", "")
                if cc_type:
                    ctx_parts.append(f"\n对话类型：{cc_type}")
                if cc_summary:
                    ctx_parts.append(f"对话摘要：{cc_summary}")

        ctx_parts.append(
            "\n\n请分析用户的真实目标，输出 JSON 格式的分析结果。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(ctx_parts)},
        ]

        try:
            raw = self._llm.complete(messages=messages)
            parsed = self._parse_goal_json(raw)
            if parsed:
                return parsed
        except Exception as exc:
            logger.warning("GoalAnalyzer LLM call failed: %s", exc)

        return self._default_goal(question)

    def _parse_goal_json(self, raw: str) -> dict[str, Any] | None:
        """Extract JSON from LLM output."""
        text = raw.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try code blocks
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
    def _default_goal(question: str = "") -> dict[str, Any]:
        """Fallback goal when LLM is unavailable."""
        return {
            "goal": question or "处理用户请求",
            "goal_type": "other",
            "expected_outputs": ["answer"],
            "priority": "normal",
            "confidence": 0.1,
        }
