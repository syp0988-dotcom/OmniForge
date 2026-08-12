"""Special goal templates - deterministic demos and report generation.

DOCX report generation is a hard-coded special case that sits outside the
generic planner / reflector logic, so the document pipeline stays
high-quality and deterministic.  Keeping it in one module gives the template
a single source of truth and keeps the generic task-generation paths clean.
"""

from __future__ import annotations

from agentflow.graph.plan import Plan
from agentflow.graph.task import Task

# ------------------------------------------------------------------
# Deterministic file-generation templates
# ------------------------------------------------------------------


# Docx report template helpers
# ------------------------------------------------------------------


def is_docx_report_goal(goal: str) -> bool:
    text = goal.lower()
    wants_docx = any(token in text for token in ("docx", ".docx", "word"))
    wants_report = any(token in goal for token in ("报告", "文档", "整理"))
    create_intent = any(token in goal for token in ("整理", "生成", "创建", "输出", "做成", "写成"))
    return wants_docx and wants_report and create_intent


def build_docx_report_plan(goal: str, state: dict) -> Plan:
    content = _build_docx_report_content(goal, state)
    task = Task(
        task_id="create_docx_report",
        title="创建 DOCX 报告",
        priority=100,
        goal="create",
        capability="docx.create",
        tool="docx",
        input={
            "action": "create",
            "path": _docx_report_path(goal, state),
            "content": content,
        },
        agent="planner",
    )
    return Plan(
        goal=goal,
        category="project",
        tasks=[task],
        goal_completed=False,
        reasoning="Matched deterministic DOCX report template",
    )


def _build_docx_report_content(goal: str, state: dict) -> str:
    source = _latest_assistant_content(state)
    if not source:
        knowledge_context = str(state.get("knowledge_context", "") or "").strip()
        search_results = state.get("search_results")
        source = knowledge_context or _stringify_search_results(search_results)
    if not source:
        source = "暂无可整理的上一轮内容，请补充报告材料。"

    return (
        "# DOCX 报告\n\n"
        "## 用户需求\n\n"
        f"{goal}\n\n"
        "## 整理内容\n\n"
        f"{source.strip()}\n"
    )


def _latest_assistant_content(state: dict) -> str:
    history = state.get("history") or []
    if not isinstance(history, list):
        memory = state.get("memory")
        if isinstance(memory, dict):
            history = memory.get("history") or []
    if not isinstance(history, list):
        return ""

    for message in reversed(history):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = str(message.get("content", "") or "").strip()
            if content:
                return content
    return ""


def _stringify_search_results(search_results: object) -> str:
    if not isinstance(search_results, list):
        return ""
    lines: list[str] = []
    for item in search_results[:5]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "") or "").strip()
        summary = str(item.get("summary") or item.get("snippet") or item.get("content") or "").strip()
        if title or summary:
            lines.append(f"### {title or '搜索结果'}\n\n{summary}")
    return "\n\n".join(lines)


def _docx_report_path(goal: str, state: dict) -> str:
    history_text = _latest_assistant_content(state)
    combined = f"{goal}\n{history_text}".lower()
    if "omniforge" in combined:
        return "OmniForge报告.docx"
    return "report.docx"
