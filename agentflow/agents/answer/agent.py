"""Answer Agent — finalizes agent outputs into polished user-facing responses."""

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
        knowledge_context = state.get("knowledge_context", "")
        memory = state.get("memory", {})
        llm_service = get_llm_service()

        logger.info("Formatting answer for category: %s", category)

        system_content = (
            "你是一名专业、简洁、符合 ChatGPT 风格的中文 AI 助手。"
            "当需要回答问题时，直接给出最终结论，不要暴露内部工作流或工具日志。"
            "禁止引用搜索引擎页面作为答案主体，不要输出 duckduckgo 跳转链接。"
            "如果是联网搜索问题，最后以“参考资料”列出来源名称，不要显示中间搜索日志。"
            "如果是身份问题，直接根据系统配置回答，不要猜测。"
            "当提供了知识库参考资料时，请优先基于资料内容回答。"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        # Inject conversation history from memory (previous turns)
        # MemoryAgent runs AFTER AnswerAgent in the current turn, so
        # state["memory"] contains the history from ALL previous turns.
        if isinstance(memory, dict):
            prev_history = memory.get("history", [])
            for msg in prev_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})

        # Append the current user prompt
        messages.append({
            "role": "user",
            "content": self.build_prompt(category, question, search_results, knowledge_context),
        })

        answer = llm_service.complete(messages=messages)
        state["answer"] = self.clean_answer(answer)
        return state

    def build_prompt(self, category: str, question: str, search_results: object, knowledge_context: str = "") -> str:
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

        # Knowledge context takes priority when available
        if knowledge_context and len(knowledge_context) > 20:
            prompt += (
                "以下是知识库中相关的参考资料，请基于这些资料回答用户问题。"
                "如果资料足够回答，直接给出结论并标注信息来源（文件名）。"
                "如果资料不足以回答，可以补充你自己的知识。\n\n"
                f"知识库资料：\n{knowledge_context}\n\n"
            )

        if category == "search":
            prompt += (
                "请基于以下搜索结果，生成简洁中文回答，保持 ChatGPT 风格。"
                "不要输出搜索日志、工作流或中间结果。"
                "只在答案末尾以“参考资料”列出来源名称。\n\n"
                f"搜索结果：{self.format_search_results(search_results)}"
            )
        elif not (knowledge_context and len(knowledge_context) > 20):
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
