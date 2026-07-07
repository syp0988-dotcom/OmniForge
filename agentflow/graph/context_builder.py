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

import subprocess
from pathlib import Path
from typing import Any


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

        # Capability analysis
        self.capability_analysis = state.get("capability_analysis", {})
        if isinstance(self.capability_analysis, dict):
            self.needs_filesystem = self.capability_analysis.get("filesystem", False)
            self.needs_coding = self.capability_analysis.get("coding", False)
            self.needs_git = self.capability_analysis.get("git", False)
            self.needs_knowledge = self.capability_analysis.get("knowledge", False)
        else:
            self.needs_filesystem = False
            self.needs_coding = False
            self.needs_git = False
            self.needs_knowledge = False

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

        # Git status (lightweight)
        ctx["git_status"] = self._get_git_status()

        # Project structure
        ctx["project_structure"] = self._get_project_structure()

        # ── Task Queue context ──
        ctx["workspace_state"] = self.get_workspace_state()
        ctx["task_queue"] = self.task_queue
        ctx["tool_results"] = self.tool_results

        return ctx

    def format_planner_prompt(self) -> str:
        """Format the context into a structured prompt string for the Planner."""
        ctx = self.build()
        parts: list[str] = []

        # Goal
        parts.append(f"## 用户目标\n{ctx['goal']}")
        parts.append(f"## 目标类型\n{ctx.get('goal_type', 'other')}")

        if ctx.get("expected_outputs"):
            parts.append(f"## 期望输出\n{', '.join(ctx['expected_outputs'])}")

        # Capabilities
        caps = ctx.get("capabilities", {})
        if caps:
            parts.append(
                "## 需要的能力\n"
                + "\n".join(f"- {k}: {'需要' if v else '不需要'}" for k, v in caps.items())
            )

        # Conversation
        if ctx.get("conversation_summary"):
            parts.append(f"## 对话上下文\n{ctx['conversation_summary']}")

        # Knowledge
        if ctx.get("knowledge_references"):
            kb = ctx["knowledge_references"]
            if len(kb) > 2000:
                kb = kb[:2000] + "\n...（更多知识库内容已截断）"
            parts.append(f"## 知识库参考\n{kb}")

        # Search
        if ctx.get("search_results"):
            parts.append(f"## 搜索结果\n{ctx['search_results']}")

        # Replan
        if ctx.get("replan_context"):
            parts.append(
                f"## 重新规划上下文（第 {ctx.get('replan_count', 1)} 次重试）\n"
                f"{ctx['replan_context']}\n\n"
                "请根据上述失败信息调整计划。"
            )

        # Git status
        if ctx.get("git_status"):
            parts.append(f"## Git 状态\n{ctx['git_status']}")

        # Project structure
        if ctx.get("project_structure"):
            parts.append(f"## 项目结构\n{ctx['project_structure']}")

        # ── Task Queue state ──
        tq_summary = self.format_task_queue_summary()
        if tq_summary:
            parts.append(f"## 当前任务队列\n{tq_summary}")

        # ── Workspace state ──
        ws_summary = self.format_workspace_summary()
        if ws_summary:
            parts.append(f"## 当前工作区状态\n{ws_summary}")

        # Tool results summary
        tr = ctx.get("tool_results", [])
        if tr:
            success_count = sum(1 for r in tr if isinstance(r, dict) and r.get("success"))
            parts.append(
                f"## 工具执行结果总结\n"
                f"成功：{success_count}/{len(tr)} 个任务\n"
                f"失败：{len(tr) - success_count}/{len(tr)} 个任务"
            )

        return "\n\n".join(parts)

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

    def _get_git_status(self) -> str:
        """Get a lightweight git status summary."""
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()

            status = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()

            if branch:
                lines = [f"分支：{branch}"]
                if status:
                    changed = len(status.split("\n"))
                    lines.append(f"未提交更改：{changed} 个文件")
                else:
                    lines.append("工作区干净")
                return "\n".join(lines)
        except Exception:
            pass
        return ""

    def _get_project_structure(self) -> str:
        """Get a lightweight project directory listing."""
        try:
            root = Path.cwd()
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
        ws_path = Path.cwd()
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
