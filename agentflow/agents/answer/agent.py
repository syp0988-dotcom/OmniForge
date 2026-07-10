"""Answer Agent — finalises agent outputs into user-facing responses.

In the goal-driven architecture, AnswerAgent has two modes:

1. **Summary mode** (for project/coding/refactor/workflow goals):
   Reports what was created, the project structure, and next steps.
   Does NOT generate code or content — only summarises.

2. **Answer mode** (for question/translation/search goals):
   Generates a natural language answer using available context
   (knowledge references, search results, memory).
"""

from __future__ import annotations

from agentflow.agents.base import AgentProtocol
from agentflow.config.prompts import answer_system_prompt
from agentflow.graph.context_builder import ContextBuilder
from agentflow.services.llm_service import get_llm_service
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("answer")

# Goal types that produce summary output (not LLM-generated answers)
_SUMMARY_GOAL_TYPES = frozenset({
    "project", "coding", "refactor", "workflow", "debug", "tool_use",
})


class AnswerAgent(AgentProtocol):
    """Produce a user-facing response from workflow state.

    - For project/completion goals: outputs a structured summary of what was done.
    - For question/search goals: generates an LLM-powered answer with context.
    """

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        """Produce a user-facing response from workflow state.

        Three knowledge-source modes (dual-framework RAG + LLM Wiki):
          - ``"general"`` → direct LLM answer, no RAG context (fast path)
          - ``"local"``   → RAG-only, use knowledge_context exclusively
          - ``"hybrid"``  → fuse RAG context + LLM own knowledge
        """
        is_continue = bool(state.get("_continue_mode", False))
        goal_analysis = state.get("goal_analysis", {})
        degraded = bool(state.get("_degraded", False))
        llm_error = str(state.get("_llm_error", ""))

        if isinstance(goal_analysis, dict):
            goal_type = goal_analysis.get("goal_type", "other")
            goal = goal_analysis.get("goal", state.get("question", ""))
            knowledge_source = goal_analysis.get("knowledge_source", "hybrid")
            source_mode = goal_analysis.get("source_mode", state.get("source_mode", "auto"))
        else:
            goal_type = "other"
            goal = state.get("question", "")
            knowledge_source = "hybrid"
            source_mode = state.get("source_mode", "auto")

        source_mode = str(source_mode or "auto")
        if source_mode == "knowledge":
            knowledge_source = "local"

        # ── Degraded mode: LLM unavailable, produce fallback message ──
        if degraded:
            from agentflow.services.llm_service import classify_error, get_fallback_message
            error_type = classify_error(Exception(llm_error)) if llm_error else "unknown"
            fallback = get_fallback_message(error_type, goal_type)
            state["answer"] = self._degraded_answer(goal, error_type, fallback, goal_type)
            logger.warning("Answer: degraded mode (error_type=%s)", error_type)
            return state

        if source_mode == "knowledge" and state.get("knowledge_results") == []:
            state["answer"] = (
                "我已按“知识库”模式检索，但没有找到与这个问题相关的知识库内容。"
                "请先确认相关文档已经上传并完成索引，或者换一个更接近文档原文的问法。"
            )
            logger.info("Answer: knowledge mode had no matching references")
            return state

        # ── Summary mode: project / coding / refactor / workflow / debug ──
        if goal_type in _SUMMARY_GOAL_TYPES:
            state["answer"] = self._build_completion_summary(state, goal, goal_type)
            logger.info("Answer: summary mode (goal_type=%s)", goal_type)
            return state

        # ── Answer mode: question / search / translation / editing ──────
        llm_service = get_llm_service()
        logger.info(
            "Answer: LLM answer mode (goal_type=%s, knowledge=%s, continue=%s)",
            goal_type, knowledge_source, is_continue,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt(
                continue_mode=is_continue,
                knowledge_source=knowledge_source,
            )},
        ]

        builder = ContextBuilder(state)
        user_prompt = builder.format_answer_prompt()
        if source_mode == "knowledge":
            user_prompt = self._append_knowledge_sources(user_prompt, state)

        # If the previous turn had a generation failure, include the concrete
        # failure reason so the LLM doesn't hallucinate an explanation.
        failure_context = self._get_failure_context(state)
        if failure_context:
            user_prompt = failure_context + "\n\n" + user_prompt

        messages.append({"role": "user", "content": user_prompt})

        if state.get("_stream_answer"):
            state["_answer_stream_messages"] = messages
            state["_answer_stream_mode"] = True
            state["answer"] = ""
            logger.info("Answer: prepared stream messages, skipping blocking LLM call")
            return state

        answer = llm_service.complete(messages=messages)
        logger.info("Answer: LLM returned %d chars: %s", len(answer), answer[:100])
        state["answer"] = self.clean_answer(answer)
        return state

    @staticmethod
    def _append_knowledge_sources(prompt: str, state: dict[str, object]) -> str:
        sources = AnswerAgent._format_knowledge_sources(state.get("knowledge_results"))
        if not sources:
            return prompt
        return (
            f"{prompt}\n\n"
            "## 知识库来源文件\n"
            f"{sources}\n\n"
            "请在回答末尾添加“来源：”，列出上面的文件名；不要列出未出现在上方的来源。"
        )

    @staticmethod
    def _get_failure_context(state: dict[str, object]) -> str:
        ss = state.get("session_state")
        if ss is None:
            return ""
        reason = ss.metadata.pop("last_failure_reason", "")
        goal = ss.metadata.pop("last_failure_goal", "")
        if not reason:
            return ""
        return (
            "## 上一轮任务失败信息\n"
            f"上一轮任务「{goal}」执行失败，具体原因如下：\n\n"
            f"{reason}\n\n"
            "请基于以上失败信息回答用户的问题。"
            "如果用户询问失败原因，请直接引用上述信息回答，不要自行推测。"
        )

    @staticmethod
    def _format_knowledge_sources(results: object) -> str:
        if not isinstance(results, list):
            return ""

        lines: list[str] = []
        seen: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            filename = str(item.get("filename", "") or "").strip()
            if not filename or filename in seen:
                continue
            seen.add(filename)

            score = item.get("score")
            method = str(item.get("method", "") or "").strip()
            details: list[str] = []
            if isinstance(score, (int, float)):
                details.append(f"相似度 {float(score):.2f}")
            if method:
                details.append(f"检索方式 {method}")
            suffix = f"（{'，'.join(details)}）" if details else ""
            lines.append(f"- {filename}{suffix}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Summary mode
    # ------------------------------------------------------------------

    def _build_completion_summary(
        self,
        state: dict[str, object],
        goal: str,
        goal_type: str,
    ) -> str:
        """Build a completion summary for project/coding/refactor goals."""
        plan = state.get("plan", {})
        tool_results = state.get("tool_results", [])
        reflection_msg = str(state.get("_reflection_message", ""))

        # Generation failed -> show the specific failure reason
        if state.get("_generation_failed"):
            reason = str(state.get("_generation_failure_reason", ""))
            lines = [f"❌ 目标未完成：**{goal}**", ""]
            if reason:
                lines.append(reason)
            if reflection_msg:
                lines.extend(["", reflection_msg])
            return "\n".join(lines)

        # Count from task_queue (has actual execution status) rather than
        # plan.tasks (which retains original TODO status).
        task_queue = state.get("task_queue", []) or []
        if task_queue:
            done_count = sum(1 for t in task_queue if t.get("status") in ("done", "completed"))
            total_count = len(task_queue)
        else:
            # Fallback to plan tasks when task_queue is not set
            if isinstance(plan, dict):
                plan_tasks = plan.get("tasks", [])
            else:
                plan_tasks = getattr(plan, "tasks", [])
            done_count = sum(1 for t in plan_tasks if isinstance(t, dict) and t.get("status") == "completed"
                            or not isinstance(t, dict) and hasattr(t, "is_finished") and t.is_finished)
            total_count = len(plan_tasks)

        # Collect created paths from completed tasks in the queue
        # (tool_results only holds the last result, so we read from task_queue instead)
        created = []
        for t in (task_queue or []):
            if t.get("status") in ("done", "completed"):
                inp = t.get("input", {}) or {}
                action = str(inp.get("action") or t.get("goal") or "")
                tool = str(t.get("tool") or "")
                if tool == "filesystem" and action not in {"write_file", "create_file", "append_file"}:
                    continue
                if tool == "docx" and action != "create":
                    continue
                path = inp.get("path", "")
                if path and path not in created:
                    created.append(path)

        if done_count > 0 and done_count >= total_count:
            status_icon = "✅"
            status_text = "目标完成"
        elif done_count > 0:
            status_icon = "⏳"
            status_text = "部分完成"
        else:
            status_icon = "❌"
            status_text = "目标未完成"
        lines = [f"{status_icon} {status_text}：**{goal}**", ""]

        if created:
            tree = self._build_path_tree(created)
            lines.append("**已创建的文件结构：**\n")
            lines.append("```")
            lines.extend(tree)
            lines.append("```")
            lines.append("")

        # Task summary
        if total_count > 0:
            lines.append(f"**执行统计：** {done_count}/{total_count} 个任务完成")

        if reflection_msg:
            lines.append("")
            lines.append(reflection_msg)

        # Next steps (for project goals)
        if goal_type == "project":
            lines.extend([
                "",
                "---",
                "**下一步：**",
                "1. 进入项目目录检查生成的文件",
                "2. 查看 README 获取项目说明",
                "3. 如有需要，告诉我进行修改或扩展",
            ])

        return "\n".join(lines)

    @staticmethod
    def _build_path_tree(paths: list[str]) -> list[str]:
        """Build a simple tree view from a list of file paths."""
        if not paths:
            return []

        parts_list = [p.replace("\\", "/").strip("/").split("/") for p in paths]
        tree: dict[str, set[str]] = {}
        for parts in parts_list:
            for i, part in enumerate(parts):
                parent = "/".join(parts[:i]) if i > 0 else ""
                tree.setdefault(parent, set()).add(part)

        result: list[str] = []

        def _emit(parent: str, prefix: str = "") -> None:
            children = sorted(tree.get(parent, set()))
            for i, child in enumerate(children):
                child_path = f"{parent}/{child}" if parent else child
                is_last = i == len(children) - 1
                is_dir = child_path in tree
                connector = "└── " if is_last else "├── "
                suffix = "/" if is_dir else ""
                result.append(f"{prefix}{connector}{child}{suffix}")
                if is_dir:
                    ext = "    " if is_last else "│   "
                    _emit(child_path, prefix + ext)

        _emit("")
        return result if result else paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def clean_answer(text: str) -> str:
        """Remove residual noise from the raw LLM output."""
        return text.strip()

    @staticmethod
    def _system_prompt(continue_mode: bool = False, knowledge_source: str = "hybrid") -> str:
        """System prompt for answer mode, adapted to knowledge source.

        Three prompts for the dual-framework (RAG + LLM Wiki):
          - ``"general"``: LLM answers from its own parametric knowledge.
            No RAG context needed — faster and cheaper.
          - ``"local"``:   RAG-only. Strictly answer from provided context.
            Prevents hallucination on project-specific queries.
          - ``"hybrid"``:  Fuse RAG context with LLM's own knowledge.
            Best for questions that need both local docs and general knowledge.
        """
        return answer_system_prompt(
            continue_mode=continue_mode,
            knowledge_source=knowledge_source,
        )

    def _degraded_answer(
        self, goal: str, error_type: str, fallback: str, goal_type: str,
    ) -> str:
        """Produce a fallback answer when the LLM is unavailable."""
        lines = [f"**{fallback}**", ""]

        if error_type == "budget_exceeded":
            lines.append("请开启一个新会话继续提问。")
        elif error_type == "auth_error":
            lines.append("请到设置页面检查 API 密钥配置。")
        elif error_type == "rate_limit":
            lines.append("请稍等片刻后重试。")
        elif error_type == "network":
            lines.append("请检查后端服务和网络连接状态。")
        elif goal_type in _SUMMARY_GOAL_TYPES:
            lines.append(
                "系统处于受限模式，此前已完成的部分任务结果可能仍可用。"
            )
        else:
            lines.append("系统处于受限模式，部分功能暂时不可用。")

        # Always show the user what we understood their request to be
        if goal:
            lines.extend(["", f"你的请求：{goal}"])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Preserved backward compat interfaces
    # ------------------------------------------------------------------

    def build_prompt(
        self, category: str, question: str,
        search_results: object, knowledge_context: str = "",
    ) -> str:
        """Preserved interface for backward compatibility."""
        return ""

    def format_search_results(self, results: object) -> str:
        """Preserved interface for backward compatibility."""
        return ""
