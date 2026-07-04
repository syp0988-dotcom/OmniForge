from __future__ import annotations

import json
from typing import Any

from agentflow.services.llm_service import get_llm_service
from agentflow.utils.logging import build_logger

logger = build_logger("report")


class ReportAgent:
    """Generate a concise report from workflow results."""

    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        workflow = state.get("workflow", [])
        search_results = state.get("search_results", [])
        llm_service = get_llm_service()

        logger.info("Creating report for: %s", question)
        search_context = json.dumps(search_results, ensure_ascii=False, indent=2)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个简明、专业的中文 AI 助手。"
                    "请根据用户问题和现有工作流结果，用简洁中文回答。"
                    "回答应清晰、专业，不要添加无关说明。"
                    "如果有来源，请在回答末尾以“参考来源”列出标题和 URL。"
                    "不要编造信息；如果无法确定答案，请说明不确定性并建议后续步骤。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"工作流阶段：{', '.join(str(step) for step in workflow)}\n\n"
                    f"搜索结果：{search_context}\n\n"
                    "请直接给出最终答案，优先使用中文，保持格式简洁。"
                ),
            },
        ]
        answer = llm_service.complete(messages=messages)
        state["answer"] = answer.strip()
        return state
