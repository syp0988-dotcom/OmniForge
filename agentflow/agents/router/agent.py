from __future__ import annotations

import re
from agentflow.agents.base import AgentProtocol
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("router")


class QueryRouterAgent(AgentProtocol):
    """Route user queries to the correct AI agent pipeline."""

    IDENTITY_PATTERNS = [
        r"你是谁",
        r"你是.*模型",
        r"是什么模型",
        r"是否调用.*大模型",
        r"你调用.*大模型",
        r"你的能力",
        r"你.*身份",
        r"你叫什么",
        r"你好",
    ]

    SEARCH_PATTERNS = [
        r"今[天|日].*(新闻|天气|气温|情况)",
        r"最新.*(新闻|AI|模型|版本|GitHub|release)",
        r"GitHub.*(最新|版本|repo|仓库)",
        r"今天.*(新闻|天气|股市|行情)",
        r"(新闻|天气|实时|最新).*",
        r"(价格|价格走势|实时).*",
    ]

    CODING_PATTERNS = [
        r"写.*代码",
        r"生成.*代码",
        r"调试.*",
        r"修复.*bug",
        r"解释.*Python",
        r"Python.*(解释|怎么|如何)",
        r"函数|类|语法|遍历|循环|异常",
    ]

    WRITING_PATTERNS = [
        r"翻译",
        r"润色",
        r"改写",
        r"重写",
        r"总结",
        r"归纳",
        r"写一段",
        r"写一篇",
        r"文案",
        r"报告",
    ]

    REASONING_PATTERNS = [
        r"为什么",
        r"如何",
        r"怎样",
        r"推理",
        r"分析",
        r"计算",
        r"数学",
        r"逻辑",
    ]

    KNOWLEDGE_PATTERNS = [
        r"知识库",
        r"文档",
        r"资料",
        r"介绍",
        r"说明.*",
        r"解释.*",
        r"描述.*",
        r"包含哪些",
        r"包括哪些",
        r"有哪些",
        r"是什么",
        r"什么是",
        r"概念",
        r"定义",
        r"功能",
        r"用途",
        r"方法",
        r"步骤",
        r"流程",
        r"规范",
        r"标准",
        r"指南",
        r"手册",
        r"帮助",
    ]

    PYTHON_PATTERNS = [
        r"Python",
        r"脚本",
        r"变量",
        r"模块",
        r"库",
    ]

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        # Use the original (un-enriched) question for routing classification.
        # The rewritten question may contain context from RewriteEngine that
        # would pollute category detection (e.g. "Python" in conversation
        # context would trigger PYTHON_PATTERNS incorrectly).
        question = str(state.get("_original_question", "") or state.get("question", "")).strip()

        category = self.classify(question)
        logger.info(
            "Routing '%s' → '%s'",
            question, category,
        )
        state["category"] = category
        state["router"] = {"category": category, "raw_input": question}
        return state

    def classify(self, question: str) -> str:
        normalized = question.lower()
        if self.match_any(normalized, self.IDENTITY_PATTERNS):
            return "identity"
        if self.match_any(normalized, self.SEARCH_PATTERNS):
            return "search"
        if self.match_any(normalized, self.CODING_PATTERNS):
            return "coding"
        if self.match_any(normalized, self.WRITING_PATTERNS):
            return "writing"
        if self.match_any(normalized, self.PYTHON_PATTERNS):
            return "python"
        if self.match_any(normalized, self.REASONING_PATTERNS):
            return "reasoning"
        if self.match_any(normalized, self.KNOWLEDGE_PATTERNS):
            return "knowledge"
        return "reasoning"

    @staticmethod
    def match_any(text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)
