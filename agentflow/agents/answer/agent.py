from __future__ import annotations

import re
from typing import Any

from agentflow.services.llm_service import get_llm_service
from agentflow.utils.logging import build_logger

logger = build_logger("answer")


class AnswerAgent:
    """Finalize agent outputs into polished user-facing responses."""

    def run(self, state: dict[str, object]) -> dict[str, object]:
        category = str(state.get("category", "reasoning"))
        question = str(state.get("question", ""))
        search_results = state.get("search_results", [])
        llm_service = get_llm_service()

        logger.info("Formatting answer for category: %s", category)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名专业、简洁、符合 ChatGPT 风格的中文 AI 助手。"
                    "当需要回答问题时，直接给出最终结论，不要暴露内部工作流或工具日志。"
                    "禁止引用搜索引擎页面作为答案主体，不要输出 duckduckgo 跳转链接。"
                    "如果是联网搜索问题，最后以“参考资料”列出来源名称，不要显示中间搜索日志。"
                    "如果是身份问题，直接根据系统配置回答，不要猜测。"
                ),
            },
            {
                "role": "user",
                "content": self.build_prompt(category, question, search_results),
            },
        ]

        answer = llm_service.complete(messages=messages)
        state["answer"] = self.clean_answer(answer)
        return state

    def build_prompt(self, category: str, question: str, search_results: object) -> str:
        if category == "identity":
            return (
                f"用户问题：{question}\n\n"
                "请直接回答以下身份问题：你是谁，你是什么模型，是否调用大模型。"
                "禁止联网搜索。"
                "如果无法准确确认部署环境，请回答："
                "我是当前系统配置的大语言模型助手，具体模型名称取决于部署配置。"
                "不要引用知乎或 Wikipedia。"
            )

        prompt = f"用户问题：{question}\n\n"

        if category == "search":
            prompt += (
                "请基于以下搜索结果，生成简洁中文回答，保持 ChatGPT 风格。"
                "不要输出搜索日志、工作流或中间结果。"
                "只在答案末尾以“参考资料”列出来源名称。\n\n"
                f"搜索结果：{self.format_search_results(search_results)}"
            )
        else:
            prompt += (
                "请直接回答该问题，保持专业、清晰、简洁。"
                "不要输出搜索日志或工具调用内容。"
            )

        prompt += "\n\n输出格式要求：# 标题\n正文\n## 要点\n- ...\n## 参考资料（如果联网）\n- 来源名称"
        return prompt

    def format_search_results(self, results: object) -> str:
        if not isinstance(results, list):
            return ""
        formatted = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            url = item.get("url", "").strip()
            if url.startswith("https://duckduckgo.com/l/?uddg="):
                url = self.extract_redirect_url(url)
            formatted.append(f"标题：{title}，链接：{url}")
        return "；".join(formatted)

    def extract_redirect_url(self, url: str) -> str:
        match = re.search(r"uddg=(.+)$", url)
        if match:
            return match.group(1)
        return url

    def clean_answer(self, text: str) -> str:
        text = re.sub(r"Processed request:.*", "", text)
        text = re.sub(r"Workflow steps:.*", "", text)
        text = re.sub(r"Search Result.*", "", text)
        text = re.sub(r"Summary.*", "", text)
        return text.strip()
