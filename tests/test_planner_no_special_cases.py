"""The planner must not short-circuit through hard-coded goal special cases.

Benchmark-era and report special cases (snake game, DOCX report) were removed
so every request flows through the generic blueprint/template/LLM planner.
"""

from __future__ import annotations

import importlib.util


def test_no_special_goals_module():
    # The special_goals module that hosted snake-game / DOCX-report shortcuts
    # no longer exists.
    assert importlib.util.find_spec("agentflow.agents.planner.special_goals") is None


def test_planner_has_no_special_case_imports():
    import ast
    from pathlib import Path

    source = Path("agentflow/agents/planner/agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "agentflow.agents.planner.special_goals":
            raise AssertionError("planner still imports special_goals")
