"""Compare planner_prompt vs answer_prompt — quantify noise reduction.

Usage: python tests/test_answer_prompt.py

What it measures:
  - Total character count of each prompt
  - Section count (how many context blocks)
  - Noise sections: task queue, workspace, tool results, git, project structure
  - Signal sections: goal, conversation, knowledge, search, memory
  - Signal-to-noise ratio
  - Estimated token count

The expected result is that ``format_answer_prompt()`` has:
  - 60-80% fewer total chars
  - Zero noise sections
  - Higher signal-to-noise ratio
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.graph.context_builder import ContextBuilder
from agentflow.services.llm_service import estimate_tokens


def build_mock_state() -> dict:
    """Build a realistic workflow state with noise (task queue, workspace, etc.)."""
    return {
        "question": "帮我写一个图书管理系统",
        "goal_analysis": {
            "goal": "创建一个完整可运行的图书管理系统",
            "goal_type": "project",
            "expected_outputs": ["project", "source_code", "database", "readme"],
            "priority": "high",
            "confidence": 0.95,
        },
        # ── Conversation context (signal) ──
        "conversation_context": {
            "type": "NEW_TASK",
            "current_goal": "创建一个完整可运行的图书管理系统",
            "summary": "用户想要创建一个图书管理系统，包含前端和后端",
            "entities": ["图书管理", "系统"],
        },
        # ── Knowledge results (signal) ──
        "knowledge_context": (
            "项目规范：使用 Flask + SQLAlchemy, 前端使用 Vue 3\n"
            "数据库设计：books(id, title, author, isbn, created_at)\n"
            "API 规范：RESTful, /api/books CRUD\n"
        ),
        # ── Search results (signal) ──
        "search_results": [
            {
                "title": "Flask 项目结构最佳实践",
                "snippet": "推荐的项目结构包括 app/, config.py, requirements.txt 等",
            },
            {
                "title": "图书管理系统设计模式",
                "snippet": "MVC 架构，Model 层负责数据，View 层负责展示",
            },
        ],
        # ── Session/memory (signal) ──
        "session_context": (
            "长期记忆：用户之前创建过 Flask 项目，熟悉 Python"
        ),
        # ── Task queue (NOISE for AnswerAgent) ──
        "task_queue": [
            {"task_id": "create_project_dir", "title": "创建项目目录",
             "priority": 100, "tool": "filesystem", "status": "done",
             "input": {"action": "mkdir", "path": "图书管理/"}},
            {"task_id": "create_app", "title": "创建 app.py",
             "priority": 80, "tool": "filesystem", "status": "done",
             "input": {"action": "write_file", "path": "图书管理/app.py"}},
            {"task_id": "create_models", "title": "创建 models.py",
             "priority": 70, "tool": "filesystem", "status": "done",
             "input": {"action": "write_file", "path": "图书管理/models.py"}},
            {"task_id": "create_routes", "title": "创建路由",
             "priority": 60, "tool": "filesystem", "status": "todo",
             "input": {"action": "write_file", "path": "图书管理/routes.py"}},
            {"task_id": "create_tests", "title": "创建测试",
             "priority": 20, "tool": "filesystem", "status": "todo",
             "input": {"action": "write_file", "path": "图书管理/tests/"}},
        ],
        # ── Tool results (NOISE for AnswerAgent) ──
        "tool_results": [
            {"success": True, "tool": "filesystem", "action": "mkdir",
             "result": {"path": "g:/outputs/图书管理/"},
             "input": {"path": "图书管理/"},
             "message": "Directory created"},
            {"success": True, "tool": "filesystem", "action": "write_file",
             "result": {"path": "g:/outputs/图书管理/app.py"},
             "input": {"path": "图书管理/app.py"},
             "message": "File written: app.py"},
            {"success": False, "tool": "filesystem", "action": "write_file",
             "error": "File already exists",
             "input": {"path": "图书管理/models.py"},
             "message": "File already exists"},
        ],
        # ── Replan context (NOISE for AnswerAgent) ──
        "_replan_count": 1,
        "_reflection_message": "models.py 写入失败，已存在同名文件",
        # ── Git status (NOISE for AnswerAgent) ──
        # Will be live-fetched by ContextBuilder._get_git_status()
        # ── Workspace (NOISE for AnswerAgent) ──
        # Will be live-scanned by ContextBuilder.get_workspace_state()
    }


# Sections that are useful for answering questions
_SIGNAL_SECTIONS = {"用户目标", "目标类型", "期望输出", "对话上下文",
                    "知识库参考", "搜索结果", "长期记忆"}

# Sections that are noise for answer generation
_NOISE_SECTIONS = {"需要的能力", "Git 状态", "项目结构",
                   "当前任务队列", "当前工作区状态", "工具执行结果总结",
                   "重新规划上下文"}


def classify_sections(text: str) -> tuple[list[str], list[str]]:
    """Split sections into signal and noise based on section headers."""
    signals, noises = [], []
    for line in text.split("\n"):
        if line.startswith("## "):
            name = line.removeprefix("## ").strip()
            if any(s in name for s in _SIGNAL_SECTIONS):
                signals.append(name)
            else:
                noises.append(name)
    return signals, noises


def measure(text: str, label: str) -> dict:
    """Measure metrics for a prompt text."""
    char_count = len(text)
    token_est = estimate_tokens(text)
    section_count = sum(1 for line in text.split("\n") if line.startswith("## "))
    signals, noises = classify_sections(text)
    signal_count = len(signals)
    noise_count = len(noises)
    snr = signal_count / max(noise_count, 1)

    return {
        "label": label,
        "chars": char_count,
        "tokens_est": token_est,
        "sections": section_count,
        "signal_sections": signal_count,
        "noise_sections": noise_count,
        "signal_to_noise": round(snr, 2),
        "signals": signals,
        "noises": noises,
    }


def print_report(a: dict, b: dict) -> None:
    """Print side-by-side comparison."""
    print("=" * 72)
    print(f"{'指标':<25} {'Planner Prompt':>18} {'Answer Prompt':>18} {'改进':>10}")
    print("-" * 72)

    rows = [
        ("字符数", f"{a['chars']:,}", f"{b['chars']:,}",
         f"{_pct(a['chars'], b['chars'])}"),
        ("估算 Token", f"{a['tokens_est']:,}", f"{b['tokens_est']:,}",
         f"{_pct(a['tokens_est'], b['tokens_est'])}"),
        ("总 Section 数", str(a['sections']), str(b['sections']),
         f"{_pct(a['sections'], b['sections'])}"),
        ("信号 Section", str(a['signal_sections']), str(b['signal_sections']),
         f"{_pct(a['signal_sections'], b['signal_sections'])}"),
        ("噪音 Section", str(a['noise_sections']), str(b['noise_sections']),
         _inv_pct(a['noise_sections'], b['noise_sections'])),
        ("信噪比", str(a['signal_to_noise']), str(b['signal_to_noise']),
         _inv_pct(a['signal_to_noise'], b['signal_to_noise'])),
    ]

    for name, old_v, new_v, change in rows:
        print(f"{name:<25} {old_v:>18} {new_v:>18} {change:>10}")

    print("=" * 72)
    print()

    print("--- Planner Prompt Sections ---")
    for s in a['noises']:
        print(f"  [噪音] {s}")
    for s in a['signals']:
        print(f"  [信号] {s}")

    print()
    print("--- Answer Prompt Sections ---")
    for s in b['noises']:
        print(f"  [噪音] {s}")
    for s in b['signals']:
        print(f"  [信号] {s}")

    print()
    print("--- 首 300 字符对比 ---")
    print(f"[Planner] {a['label'][:300]}")
    print()
    print(f"[Answer ] {b['label'][:300]}")


def _pct(old: int | float, new: int | float) -> str:
    if old == 0:
        return "N/A"
    p = (new - old) / old * 100
    return f"{p:+.0f}%"


def _inv_pct(old: int | float, new: int | float) -> str:
    """For metrics where LOWER is better (noise, inverse SNR)."""
    if old == 0:
        return "N/A"
    p = (new - old) / old * 100
    return f"{p:+.0f}%" if p >= 0 else f"{p:+.0f}%"


def main():
    state = build_mock_state()
    builder = ContextBuilder(state)

    planner_text = builder.format_planner_prompt()
    answer_text = builder.format_answer_prompt()

    planner_metrics = measure(planner_text, planner_text)
    answer_metrics = measure(answer_text, answer_text)

    print_report(planner_metrics, answer_metrics)

    # Summary
    char_save = planner_metrics['chars'] - answer_metrics['chars']
    token_save = planner_metrics['tokens_est'] - answer_metrics['tokens_est']
    noise_removed = planner_metrics['noise_sections'] - answer_metrics['noise_sections']
    print()
    print(">>> 结论")
    print(f"  - 每次 LLM 调用减少 {char_save:,} 字符 ({token_save} 估算 token)")
    print(f"  - 消除了 {noise_removed} 个噪音 section")
    print(f"  - 信噪比从 {planner_metrics['signal_to_noise']} 提升到 {answer_metrics['signal_to_noise']}")
    print("  - LLM 收到的是干净的目标+上下文，不再被工作区/任务队列干扰")


if __name__ == "__main__":
    main()
