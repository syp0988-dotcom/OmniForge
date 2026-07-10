"""ContextBuilder — aggregates all context sources for the Planner.

Planner no longer receives raw Conversation. Instead, ContextBuilder
assembles context from:

- Goal analysis (goal + goal_type + expected_outputs)
- Conversation (turn type, rewritten question, entities, summary)
- Workspace (current project files, structure)
- Knowledge (retrieved references from KB)
- Memory (cross-session long-term memory)
- Git Status (branch, uncommitted changes)
- Project Structure (directory tree at workspace root)
- Task History (previous tasks from the current session)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentflow.config.settings import settings
from agentflow.utils.logging import build_logger as _build_logger

_log = _build_logger("context_builder")


def _tier_label(tiers: set[int]) -> str:
    """Human-readable label for dropped tiers."""
    labels = {4: "核心", 3: "上下文", 2: "搜索结果", 1: "辅助"}
    names = [labels.get(t, str(t)) for t in sorted(tiers, reverse=True)]
    return "、".join(names)


class ContextBuilder:
    """Aggregates all context sources into a structured prompt context.

    Usage::

        builder = ContextBuilder(state)
        ctx = builder.build()  # returns dict with all context sources
        planner_prompt = builder.format_planner_prompt()
    """

    def __init__(self, state: dict[str, object]) -> None:
        self.state = state

        # Goal analysis
        self.goal_analysis = state.get("goal_analysis", {})
        if isinstance(self.goal_analysis, dict):
            self.goal = self.goal_analysis.get("goal", state.get("question", ""))
            self.goal_type = self.goal_analysis.get("goal_type", "other")
            self.expected_outputs = self.goal_analysis.get("expected_outputs", [])
        else:
            self.goal = state.get("question", "")
            self.goal_type = "other"
            self.expected_outputs = []

        # Capability hints derived from goal_type (replaces old CapabilityAnalyzer)
        self.needs_filesystem, self.needs_coding, self.needs_git, self.needs_knowledge = \
            self._derive_capabilities(self.goal_type)

        # Conversation
        self.conversation_context = state.get("conversation_context")
        self.question = str(state.get("question", ""))
        self.original_question = str(state.get("_original_question", ""))
        self.rewritten_question = str(state.get("rewritten_question", ""))
        self.is_continue = bool(state.get("_continue_mode", False))
        self.memory = state.get("memory", {})
        self.session_context = str(state.get("session_context", ""))

        # Knowledge
        self.knowledge_context = str(state.get("knowledge_context", ""))

        # Search
        self.search_results = state.get("search_results", [])

        # Reflection & replan context
        self._reflection_message = str(state.get("_reflection_message", ""))
        self._replan_count = int(state.get("_replan_count", 0))

        # Task Queue (Dynamic Task Queue Planning)
        self.task_queue: list[dict] = list(state.get("task_queue", []) or [])

        # Previous tool execution results
        self.tool_results = state.get("tool_results", [])

    @staticmethod
    def _derive_capabilities(goal_type: str) -> tuple[bool, bool, bool, bool]:
        """Derive capability hints from goal type (6-class simplified system).

        This replaces the old CapabilityAnalyzer LLM call with a simple
        heuristic, since the Planner's function calling already handles
        tool selection directly. These hints are only used for prompt
        decoration in ``format_planner_prompt()``.
        """
        if goal_type == "project":
            return True, True, True, False
        if goal_type in ("coding", "debug", "refactor"):
            return False, True, False, False
        if goal_type == "search":
            return False, False, False, False
        if goal_type == "question":
            return False, False, False, True
        if goal_type == "tool_use":
            return False, False, True, False
        return False, False, False, True

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """Build the full context dict with all available sources."""
        ctx: dict[str, Any] = {
            "goal": self.goal,
            "goal_type": self.goal_type,
            "question": self.question,
            "is_continue": self.is_continue,
        }

        # Goal & expected outputs
        if self.expected_outputs:
            ctx["expected_outputs"] = self.expected_outputs

        # Capabilities needed
        caps = {}
        if self.needs_filesystem:
            caps["filesystem"] = True
        if self.needs_coding:
            caps["coding"] = True
        if self.needs_git:
            caps["git"] = True
        if self.needs_knowledge:
            caps["knowledge"] = True
        if caps:
            ctx["capabilities"] = caps

        # Conversation summary
        conv_summary = self._get_conversation_summary()
        if conv_summary:
            ctx["conversation_summary"] = conv_summary

        # Knowledge refs
        if self.knowledge_context and len(self.knowledge_context) > 20:
            ctx["knowledge_references"] = self.knowledge_context
        else:
            ctx["knowledge_references"] = ""

        # Search results
        if self.search_results:
            ctx["search_results"] = self._format_search_context(self.search_results)

        # Replan context
        if self._replan_count > 0 and self._reflection_message:
            ctx["replan_context"] = self._reflection_message
            ctx["replan_count"] = self._replan_count

        # Memory/session
        if self.session_context:
            ctx["session_context"] = self.session_context

        # Project structure
        ctx["project_structure"] = self._get_project_structure()

        # ── Task Queue context ──
        ctx["workspace_state"] = self.get_workspace_state()
        ctx["task_queue"] = self.task_queue
        ctx["tool_results"] = self.tool_results

        return ctx

    def format_planner_prompt(self) -> str:
        """Format the context into a structured prompt string for the Planner.

        Applies priority-based truncation when total exceeds ``max_context_chars``.
        Lower-tier sections are truncated first; the goal and type are never truncated.
        """
        ctx = self.build()
        max_chars = settings.max_context_chars

        # -- Build labelled sections. Each is a (label, content, tier) tuple.
        # Tier 4 = never truncated, Tier 1 = truncated first.
        sections: list[tuple[str, str, int]] = []

        # Tier 4 — core information, never truncated
        sections.append(("## 用户目标", f"## 用户目标\n{ctx['goal']}", 4))
        sections.append(("## 目标类型", f"## 目标类型\n{ctx.get('goal_type', 'other')}", 4))

        if ctx.get("expected_outputs"):
            sections.append(("## 期望输出", f"## 期望输出\n{', '.join(ctx['expected_outputs'])}", 4))

        caps = ctx.get("capabilities", {})
        if caps:
            sections.append(("## 需要的能力", "## 需要的能力\n" + "\n".join(
                f"- {k}: {'需要' if v else '不需要'}" for k, v in caps.items()
            ), 4))

        # Tier 3 — conversation & knowledge
        if ctx.get("conversation_summary"):
            sections.append(("## 对话上下文", f"## 对话上下文\n{ctx['conversation_summary']}", 3))

        if ctx.get("knowledge_references"):
            kb = ctx["knowledge_references"]
            if len(kb) > 2000:
                kb = kb[:2000] + "\n...（更多知识库内容已截断）"
            sections.append(("## 知识库参考", f"## 知识库参考\n{kb}", 3))

        # Tier 2 — search results & replan context
        if ctx.get("search_results"):
            sections.append(("## 搜索结果", f"## 搜索结果\n{ctx['search_results']}", 2))

        if ctx.get("replan_context"):
            sections.append(("## 重新规划上下文",
                f"## 重新规划上下文（第 {ctx.get('replan_count', 1)} 次重试）\n"
                f"{ctx['replan_context']}\n\n请根据上述失败信息调整计划。", 2))

        # Tier 1 — project structure, task queue, workspace, tool results
        if ctx.get("project_structure"):
            sections.append(("## 项目结构", f"## 项目结构\n{ctx['project_structure']}", 1))

        tq_summary = self.format_task_queue_summary()
        if tq_summary:
            sections.append(("## 当前任务队列", f"## 当前任务队列\n{tq_summary}", 1))

        ws_summary = self.format_workspace_summary()
        if ws_summary:
            sections.append(("## 当前工作区状态", f"## 当前工作区状态\n{ws_summary}", 1))

        tr = ctx.get("tool_results", [])
        if tr:
            success_count = sum(1 for r in tr if isinstance(r, dict) and r.get("success"))
            sections.append(("## 工具执行结果总结",
                f"## 工具执行结果总结\n"
                f"成功：{success_count}/{len(tr)} 个任务\n"
                f"失败：{len(tr) - success_count}/{len(tr)} 个任务", 1))

        # -- Assemble and check total size --
        all_text = "\n\n".join(content for _, content, _ in sections)

        if len(all_text) <= max_chars:
            return all_text

        # -- Priority-based truncation: drop Tier 1 first, then Tier 2, then Tier 3 --
        _log.warning("Context exceeds %d chars (%d), truncating...", max_chars, len(all_text))

        truncated: list[str] = []
        dropped_tiers: set[int] = set()

        for name, content, tier in sections:
            candidate = "\n\n".join(truncated + [content])
            if len(candidate) <= int(max_chars * 1.05):
                truncated.append(content)
            elif tier == 4:
                # Always include tier 4 (core) even if slightly over budget
                truncated.append(content)
            else:
                dropped_tiers.add(tier)
                _log.debug("Dropped section '%s' (tier %d) to fit context limit", name, tier)

        result = "\n\n".join(truncated)
        if dropped_tiers:
            result += f"\n\n（注：因上下文长度限制，已省略部分{_tier_label(dropped_tiers)}信息）"

        return result

    def format_answer_prompt(self) -> str:
        """Format context for the AnswerAgent — only relevant answer sections.

        Unlike ``format_planner_prompt()`` which includes task queue, workspace
        state, tool results and other executor noise, this method produces a
        clean prompt with only the context the AnswerAgent needs:

          - Goal and goal_type (core)
          - Conversation summary (for follow-up continuity)
          - Knowledge references (from vector DB)
          - Search results (from web search)
          - Session context (long-term memory)
        """
        ctx = self.build()
        sections: list[str] = []

        # Goal
        sections.append(f"## 用户目标\n{ctx['goal']}")

        # Goal type
        sections.append(f"## 目标类型\n{ctx.get('goal_type', 'other')}")

        # Expected outputs
        if ctx.get("expected_outputs"):
            sections.append(f"## 期望输出\n{', '.join(ctx['expected_outputs'])}")

        # Conversation summary (for follow-up / continue mode)
        if ctx.get("conversation_summary"):
            sections.append(f"## 对话上下文\n{ctx['conversation_summary']}")

        # Knowledge references (truncated at 2000 chars)
        if ctx.get("knowledge_references"):
            kb = ctx["knowledge_references"]
            if len(kb) > 2000:
                kb = kb[:2000] + "\n...（更多知识库内容已截断）"
            sections.append(f"## 知识库参考\n{kb}")

        # Search results
        if ctx.get("search_results"):
            sections.append(f"## 搜索结果\n{ctx['search_results']}")

        # Session context (long-term memory)
        if ctx.get("session_context"):
            sections.append(f"## 长期记忆\n{ctx['session_context']}")

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conversation_summary(self) -> str:
        """Extract a concise summary from conversation context."""
        cc = self.conversation_context
        if not cc:
            return ""

        if isinstance(cc, dict):
            ctx_type = cc.get("type", "")
            summary = cc.get("summary", "")
            entities = cc.get("entities", [])
            current_goal = cc.get("current_goal", "")
        else:
            ctx_type = getattr(cc, "type", "")
            summary = getattr(cc, "summary", "")
            entities = getattr(cc, "entities", [])
            current_goal = getattr(cc, "current_goal", "")

        parts = []
        if ctx_type:
            parts.append(f"对话类型：{ctx_type}")
        if current_goal:
            parts.append(f"当前目标：{current_goal}")
        if summary:
            parts.append(f"摘要：{summary[:200]}")
        if entities:
            parts.append(f"实体：{'、'.join(entities)}")

        return "\n".join(parts) if parts else ""

    @property
    def workspace_path(self) -> Path:
        """The workspace directory for file operations (matches FileSystemTool)."""
        return settings.project_root / "outputs"

    def _get_project_structure(self) -> str:
        """Get a lightweight project directory listing."""
        root = self.workspace_path
        if not root.exists():
            return ""
        try:
            entries = list(root.iterdir())[:30]  # max 30 entries
            if entries:
                lines = []
                for e in entries:
                    marker = "/" if e.is_dir() else ""
                    lines.append(f"  {e.name}{marker}")
                return "\n".join(lines)
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Workspace awareness
    # ------------------------------------------------------------------

    def get_workspace_state(self) -> dict[str, Any]:
        """Scan the workspace and return current file/directory state."""
        ws_path = self.workspace_path
        files: set[str] = set()
        dirs: set[str] = set()

        # If we have a project directory from a previous plan, scan it
        project_dir = self._find_project_dir()
        if project_dir:
            scan_root = ws_path / project_dir
        else:
            scan_root = ws_path

        if scan_root.exists() and scan_root.is_dir():
            for entry in scan_root.rglob("*"):
                rel = entry.relative_to(scan_root).as_posix()
                if entry.is_file():
                    files.add(rel)
                elif entry.is_dir():
                    dirs.add(rel)

        return {
            "workspace_path": str(ws_path),
            "project_dir": project_dir or "",
            "files": sorted(files),
            "directories": sorted(dirs),
            "file_count": len(files),
            "dir_count": len(dirs),
        }

    def _find_project_dir(self) -> str | None:
        """Try to find the project directory from completed tasks."""
        for r in (self.tool_results or []):
            if isinstance(r, dict):
                inp = r.get("input", {}) or {}
                res = r.get("result", {}) or {}
                path = inp.get("path", "") or res.get("path", "")
                if path:
                    # If it's a directory from mkdir, use it
                    parts = path.replace("\\", "/").split("/")
                    if parts:
                        return parts[0]
        return None

    def format_task_queue_summary(self) -> str:
        """Format the current task queue as a readable string."""
        if not self.task_queue:
            return ""

        status_counts: dict[str, int] = {}
        for t in self.task_queue:
            s = t.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1

        lines = [
            f"任务队列共 {len(self.task_queue)} 个："
            + " ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
        ]

        # Show top TODO tasks
        todo = sorted(
            [t for t in self.task_queue if t.get("status") == "todo"],
            key=lambda x: x.get("priority", 0), reverse=True,
        )
        if todo:
            lines.append("待执行（按优先级）：")
            for t in todo[:8]:
                lines.append(
                    f"  P={t.get('priority', 0)} [{t.get('task_id', '?')}] "
                    f"{t.get('title', t.get('goal', ''))}"
                )

        return "\n".join(lines)

    def format_workspace_summary(self) -> str:
        """Format workspace state as a readable string for the prompt."""
        ws = self.get_workspace_state()
        if not ws["files"] and not ws["directories"]:
            return "（工作区为空，尚无任何文件）"

        lines = [f"当前工作目录：{ws['workspace_path']}"]
        if ws["project_dir"]:
            lines.append(f"项目目录：{ws['project_dir']}")

        if ws["directories"]:
            lines.append("目录：")
            for d in ws["directories"][:20]:
                lines.append(f"  {d}/")

        if ws["files"]:
            lines.append("文件：")
            for f in ws["files"][:30]:
                lines.append(f"  {f}")

        return "\n".join(lines)

    @staticmethod
    def _format_search_context(results: object) -> str:
        """Format search results as structured blocks."""
        if not isinstance(results, list):
            return ""
        blocks: list[str] = []
        for i, item in enumerate(results, 1):
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            snippet = item.get("snippet", item.get("content", "")).strip()
            block = f"结果 {i}"
            if title:
                block += f"\n标题：{title}"
            if snippet:
                block += f"\n摘要：{snippet[:300]}"
            blocks.append(block)
        return "\n\n".join(blocks)
