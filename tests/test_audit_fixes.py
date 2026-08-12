"""Regression tests for the module-audit fixes (intent / planner / tools)."""

import numpy as np
import pytest

from agentflow.agents.goal_analyzer.agent import GoalAnalyzer
from agentflow.agents.goal_analyzer.intent_index import IntentIndex
from agentflow.agents.planner.schemas import parse_function_name
from agentflow.agents.planner.templates import extract_project_name
from agentflow.config.settings import settings
from agentflow.graph.executor import _translate_action
from agentflow.graph.workflow import _select_parallel_tasks


# -- intent: empty question must not be flagged as LLM failure ----------------


def test_default_goal_fallback_flag_requires_question():
    assert GoalAnalyzer._default_goal("")["fallback"] is False
    assert GoalAnalyzer._default_goal("你好")["fallback"] is True


# -- intent: embedding fast path retries after transient failure -------------


class _FlakyEmbedder:
    def __init__(self):
        self.calls = 0
        self.fail_first = True

    def embed(self, texts, batch_size=20):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise ConnectionError("embedding API down")
        return [np.zeros(4) for _ in texts]


def test_intent_index_respects_retry_cooldown(monkeypatch):
    flaky = _FlakyEmbedder()
    idx = IntentIndex(anchors={"chat": ["你好"]}, embedder=flaky)
    monkeypatch.setattr(settings, "intent_index_retry_seconds", 3600)
    assert idx.match("你好") is None
    assert flaky.calls == 1
    assert idx.match("你好") is None  # within cooldown: no retry
    assert flaky.calls == 1


def test_intent_index_recovers_after_cooldown(monkeypatch):
    flaky = _FlakyEmbedder()
    idx = IntentIndex(anchors={"chat": ["你好"]}, embedder=flaky)
    monkeypatch.setattr(settings, "intent_index_retry_seconds", 0)
    assert idx.match("你好") is None  # first attempt fails
    idx.match("你好")  # cooldown expired -> re-init succeeds
    assert idx.available is True
    assert flaky.calls >= 2


# -- planner: project name extraction ----------------------------------------


@pytest.mark.parametrize(
    "goal, expected",
    [
        ("帮我写个图书管理系统", "图书管理"),
        ("帮我创建一个员工考勤系统", "员工考勤"),
        ("开发一个博客系统", "博客"),
        ("给我做一个爬虫项目", "爬虫"),
        ("创建一个贪吃蛇游戏", "贪吃蛇游戏"),
        ("创建FastAPI图书管理系统", "fastapi图书管理"),
    ],
)
def test_extract_project_name_clean(goal, expected):
    assert extract_project_name(goal) == expected


def test_extract_project_name_never_returns_full_sentence():
    name = extract_project_name("帮我写个图书管理系统")
    assert "帮我写个" not in name


# -- planner: defensive function-name parsing --------------------------------


def test_parse_function_name_dot_fallback():
    assert parse_function_name("filesystem.mkdir") == ("filesystem", "mkdir")
    assert parse_function_name("filesystem__mkdir") == ("filesystem", "mkdir")
    assert parse_function_name("search__web.search") == ("search", "web.search")


# -- tools: Chinese action translation ---------------------------------------


def test_translate_chinese_action():
    assert _translate_action("创建文件") == "write_file"
    assert _translate_action("查看文件") == "read_file"
    assert _translate_action("列出目录") == "list_directory"
    assert _translate_action("write_file") == "write_file"
    assert _translate_action("未知动作") == "未知动作"


# -- tools: parallel selection ------------------------------------------------


def test_select_parallel_mixes_reads_with_top_priority_writes():
    queue = [
        {"task_id": "w1", "tool": "filesystem", "status": "todo", "priority": 90,
         "input": {"action": "write_file", "path": "a.py"}},
        {"task_id": "r1", "tool": "filesystem", "status": "todo", "priority": 10,
         "input": {"action": "read_file", "path": "b.py"}},
        {"task_id": "s1", "tool": "search", "status": "todo", "priority": 5,
         "input": {"action": "search", "query": "x"}},
    ]
    selected = _select_parallel_tasks(queue)
    assert {t["task_id"] for t in selected} == {"w1", "r1", "s1"}


def test_select_parallel_writes_require_same_priority():
    queue = [
        {"task_id": "w1", "tool": "filesystem", "status": "todo", "priority": 90,
         "input": {"action": "write_file", "path": "a.py"}},
        {"task_id": "w2", "tool": "filesystem", "status": "todo", "priority": 10,
         "input": {"action": "write_file", "path": "b.py"}},
    ]
    assert _select_parallel_tasks(queue) == []
