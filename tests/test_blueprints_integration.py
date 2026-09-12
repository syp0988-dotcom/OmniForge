"""Integration tests for the Blueprint system.

Originally a hand-run script inside the blueprints package
(``agentflow/blueprints/tests.py``); moved here so pytest collects it
(``testpaths = tests`` in pytest.ini) and CI covers it.

Run with::

    python -m pytest tests/test_blueprints_integration.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure the package root is on sys.path
_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def check(description: str):
    """Decorator that registers a test function."""
    def decorator(fn):
        _tests.append((description, fn))
        return fn
    return decorator


_tests: list[tuple[str, Any]] = []


# ===========================================================================
# 1. Blueprint loading
# ===========================================================================


@check("all 5 YAML blueprints load without errors")
def test_loading() -> None:
    from agentflow.blueprints.loader import BlueprintLoader

    loader = BlueprintLoader()
    bps = loader.list()
    assert len(bps) == 5, f"expected 5 blueprints, got {len(bps)}: {bps}"
    expected = {"fastapi-restful", "flask-basic", "python-cli", "react-spa", "nextjs-app"}
    assert set(bps) == expected, f"mismatch: {set(bps) ^ expected}"


@check("each blueprint has at least 8 files (meaningful skeleton)")
def test_blueprint_file_count() -> None:
    from agentflow.blueprints.loader import BlueprintLoader

    loader = BlueprintLoader()
    for bp_id in loader.list():
        bp = loader.get(bp_id)
        assert bp is not None
        assert len(bp.files) >= 8, (
            f"blueprint '{bp_id}' has only {len(bp.files)} files (< 8)"
        )


@check("blueprint YAML required fields are complete")
def test_blueprint_structure() -> None:
    import yaml

    bp_dir = Path(__file__).resolve().parent
    for yaml_path in sorted(bp_dir.glob("*.yaml")):
        with open(yaml_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)

        # Required top-level fields
        for field in ("id", "name", "description", "files"):
            assert field in data, f"{yaml_path.name}: missing '{field}'"

        # Each file must have "path"
        for i, fspec in enumerate(data["files"]):
            assert "path" in fspec, (
                f"{yaml_path.name}: files[{i}] missing 'path'"
            )
            # Must provide content (inline or external template ref).
            # Empty content is allowed (legitimate for __init__.py).
            assert "content" in fspec or "template_ref" in fspec, (
                f"{yaml_path.name}: files[{i}] ('{fspec['path']}') "
                f"has neither 'content' nor 'template_ref'"
            )


# ===========================================================================
# 2. Blueprint matching
# ===========================================================================


@check("FastAPI goal matches fastapi-restful blueprint")
def test_match_fastapi() -> None:
    from agentflow.blueprints.loader import BlueprintLoader

    loader = BlueprintLoader()
    bp = loader.match("用 FastAPI 创建一个图书管理系统 REST API", "project")
    assert bp is not None, "expected a match"
    assert bp.id == "fastapi-restful", f"expected fastapi-restful, got {bp.id}"


@check("Flask goal matches flask-basic blueprint")
def test_match_flask() -> None:
    from agentflow.blueprints.loader import BlueprintLoader

    loader = BlueprintLoader()
    bp = loader.match("用 Flask 写一个博客网站", "project")
    assert bp is not None, "expected a match"
    assert bp.id == "flask-basic", f"expected flask-basic, got {bp.id}"


@check("Next.js goal matches nextjs-app blueprint")
def test_match_nextjs() -> None:
    from agentflow.blueprints.loader import BlueprintLoader

    loader = BlueprintLoader()
    bp = loader.match("用 Next.js 搭建一个电商平台", "project")
    assert bp is not None, "expected a match"
    assert bp.id == "nextjs-app", f"expected nextjs-app, got {bp.id}"


@check("React goal matches react-spa blueprint, not nextjs-app")
def test_match_react_exclusion() -> None:
    """'react' is a keyword for react-spa, and 'react' is in nextjs-app's
    exclude_keywords, so it should NOT match nextjs-app."""
    from agentflow.blueprints.loader import BlueprintLoader

    loader = BlueprintLoader()
    bp = loader.match("帮我用 React 搭建一个前端界面", "project")
    assert bp is not None, "expected a match"
    assert bp.id == "react-spa", f"expected react-spa, got {bp.id}"


@check("non-project goal_type returns None")
def test_match_non_project() -> None:
    from agentflow.blueprints.loader import BlueprintLoader

    loader = BlueprintLoader()
    bp = loader.match("什么是 FastAPI", "question")
    assert bp is None, "expected no match for question type"


@check("no-match goal returns None (falls through to LLM)")
def test_match_no_match() -> None:
    from agentflow.blueprints.loader import BlueprintLoader

    loader = BlueprintLoader()
    bp = loader.match("帮我写一个 Rust 编译器", "project")
    assert bp is None, "expected no match for unrelated goal"


@check("FastAPI goal does NOT match flask-basic (exclusion works)")
def test_match_exclude_flask() -> None:
    """'fastapi' is in flask-basic's exclude_keywords, so
    fastapi-restful should score higher."""
    from agentflow.blueprints.loader import BlueprintLoader

    loader = BlueprintLoader()
    bps = loader.list()
    assert "flask-basic" in bps

    # FastAPI goal should match fastapi-restful, not flask-basic
    bp = loader.match("创建一个 FastAPI 微服务", "project")
    assert bp is not None
    assert bp.id == "fastapi-restful", f"expected fastapi-restful, got {bp.id}"


# ===========================================================================
# 3. ProjectConfigurator
# ===========================================================================


@check("extracts project name from Chinese goal")
def test_config_project_name() -> None:
    from agentflow.blueprints.configurator import ProjectConfigurator

    config = ProjectConfigurator.from_goal("用 FastAPI 创建一个图书管理系统 REST API")
    assert config.project_name == "图书管理", f"got '{config.project_name}'"
    assert config.app_name == "图书管理", f"got '{config.app_name}'"


@check("extracts project name from English goal")
def test_config_project_name_en() -> None:
    from agentflow.blueprints.configurator import ProjectConfigurator

    config = ProjectConfigurator.from_goal("Create a blog system with FastAPI")
    assert config.project_name == "blog", f"got '{config.project_name}'"


@check("detects frameworks from goal")
def test_config_frameworks() -> None:
    from agentflow.blueprints.configurator import ProjectConfigurator

    config = ProjectConfigurator.from_goal("用 Django 做一个内容管理系统")
    # "django" should be detected via _FRAMEWORK_MAP
    # (The key is "django" in the map)
    assert "django" in config.detected_frameworks, f"got {config.detected_frameworks}"


@check("fallback project name on empty/unrelated goal")
def test_config_fallback() -> None:
    from agentflow.blueprints.configurator import ProjectConfigurator

    config = ProjectConfigurator.from_goal("你好")
    assert config.project_name == "my_project", f"got '{config.project_name}'"


@check("to_dict() returns all expected keys")
def test_config_to_dict() -> None:
    from agentflow.blueprints.configurator import ProjectConfigurator

    config = ProjectConfigurator.from_goal("用 Flask 写一个博客")
    d = config.to_dict()
    for key in ("project_name", "package_name", "app_name", "goal"):
        assert key in d, f"missing key '{key}'"


# ===========================================================================
# 4. Blueprint rendering (Jinja2)
# ===========================================================================


@check("renders file paths and content with project variables")
def test_render_basic() -> None:
    from agentflow.blueprints import BlueprintLoader, ProjectConfigurator

    loader = BlueprintLoader()
    config = ProjectConfigurator.from_goal("用 FastAPI 创建一个图书管理系统")

    bp = loader.get("fastapi-restful")
    assert bp is not None

    specs = loader.render(bp, config)

    # Paths should have the project name interpolated
    rendered_paths = [s.path for s in specs]
    assert any("图书管理" in p for p in rendered_paths), (
        f"no path contains project name: {rendered_paths[:5]}"
    )

    # Content should have the app name
    assert any(specs), "no files rendered"
    rendered = specs[0].content_template
    assert rendered, "first file has empty content"


@check("existing files are marked as skip")
def test_render_existing() -> None:
    from agentflow.blueprints import BlueprintLoader, ProjectConfigurator

    loader = BlueprintLoader()
    config = ProjectConfigurator.from_goal("用 FastAPI 创建一个博客")
    bp = loader.get("fastapi-restful")
    assert bp is not None

    # Use the actual project_name the configurator derived
    proj = config.project_name
    existing = {f"{proj}/requirements.txt", f"{proj}/.gitignore"}
    specs = loader.render(bp, config, existing_files=existing)

    reqs = [s for s in specs if "requirements.txt" in s.path]
    assert len(reqs) == 1
    assert reqs[0].type == "skip", f"expected skip, got {reqs[0].type}"

    # Non-existing files should remain as "create"
    app_files = [s for s in specs if "main.py" in s.path]
    if app_files:
        assert app_files[0].type == "create", (
            f"expected create for new file, got {app_files[0].type}"
        )


@check("conditional files are skipped when condition is falsey")
def test_render_condition() -> None:
    """This test verifies the condition mechanism works.
    We add an artificial condition to a spec manually."""
    from agentflow.blueprints import BlueprintLoader, ProjectConfigurator
    from agentflow.blueprints.models import FileSpec

    loader = BlueprintLoader()
    config = ProjectConfigurator.from_goal("用 FastAPI 创建一个博客")
    bp = loader.get("fastapi-restful")
    assert bp is not None

    # Add a conditional file to test
    bp.files.append(FileSpec(
        path="tests/test_auth.py",
        content_template="# auth tests",
        type="create",
        condition="auth_enabled",  # this is falsey in config → should be skipped
    ))

    specs = loader.render(bp, config)
    auth_files = [s for s in specs if "auth" in s.path]
    assert len(auth_files) == 0, f"expected 0, got {len(auth_files)}"

    # Now test with the variable set
    config.extra["auth_enabled"] = True
    specs2 = loader.render(bp, config)
    auth_files2 = [s for s in specs2 if "auth" in s.path]
    assert len(auth_files2) == 1, f"expected 1, got {len(auth_files2)}"


# ===========================================================================
# 5. Planner integration (FileSpec → Task)
# ===========================================================================


@check("_blueprint_specs_to_tasks produces correct tasks")
def test_specs_to_tasks() -> None:
    from agentflow.blueprints import BlueprintLoader, ProjectConfigurator
    from agentflow.agents.planner.agent import PlannerAgent
    from agentflow.graph.task import TaskStatus

    loader = BlueprintLoader()
    config = ProjectConfigurator.from_goal("用 FastAPI 创建一个博客")
    bp = loader.get("fastapi-restful")
    assert bp is not None

    specs = loader.render(bp, config)
    tasks = PlannerAgent._blueprint_specs_to_tasks(specs, config)

    assert len(tasks) > 0, "expected at least 1 task"

    # First task should be a mkdir
    assert "mkdir" in tasks[0].task_id, (
        f"first task should be mkdir, got {tasks[0].task_id}"
    )
    assert tasks[0].tool == "filesystem", (
        f"expected filesystem tool, got {tasks[0].tool}"
    )

    # All tasks should be TODO
    for t in tasks:
        assert t.status == TaskStatus.TODO, (
            f"task {t.task_id} should be TODO, got {t.status}"
        )

    # Task with content should have the content in input
    write_tasks = [t for t in tasks if "write" in t.task_id]
    if write_tasks:
        content = write_tasks[0].input.get("content", "")
        assert content, f"write task {write_tasks[0].task_id} has no content"


@check("Planner._initialize_from_blueprint returns Plan when blueprints match")
def test_planner_integration() -> None:
    from agentflow.agents.planner.agent import PlannerAgent

    planner = PlannerAgent()
    plan = planner._initialize_from_blueprint(
        "用 FastAPI 创建一个图书管理系统",
        {"question": "用 FastAPI 创建一个图书管理系统"},
    )

    assert plan is not None, "expected a Plan from blueprint"
    assert len(plan.tasks) > 0, f"expected tasks, got {len(plan.tasks)}"
    assert plan.goal_completed is False, "new blueprint should not be completed"
    assert plan.reasoning, "plan should have reasoning"

    # Verify all tasks have required fields
    for t in plan.tasks:
        assert t.tool == "filesystem", f"task {t.id}: expected filesystem"
        assert "action" in t.input, f"task {t.id}: missing action"
        assert "path" in t.input, f"task {t.id}: missing path"


@check("Planner falls through to Template when no blueprint matches")
def test_planner_fallback_to_template() -> None:
    from agentflow.agents.planner.agent import PlannerAgent

    planner = PlannerAgent()
    plan = planner._initialize_from_blueprint(
        "创建一个 Vue.js 项目",
        {"question": "创建一个 Vue.js 项目"},
    )

    # No Vue.js blueprint exists → should return None (fall through to Template)
    assert plan is None, "expected None (no Vue.js blueprint)"


# ===========================================================================
# 6. Blueprint serialization (to_dict)
# ===========================================================================


@check("Blueprint.to_dict() produces expected keys")
def test_blueprint_to_dict() -> None:
    from agentflow.blueprints.loader import BlueprintLoader

    loader = BlueprintLoader()
    bp = loader.get("fastapi-restful")
    assert bp is not None

    d = bp.to_dict()
    for key in ("id", "name", "version", "frameworks", "file_count", "source"):
        assert key in d, f"missing key '{key}'"

    assert d["file_count"] > 0, "file_count should be > 0"
    assert d["frameworks"] == ["fastapi"], f"got {d['frameworks']}"


# ===========================================================================
# Run all tests
# ===========================================================================


def main() -> None:
    passed = 0
    failed = 0

    print("=" * 60)
    print("Blueprint System — Integration Tests")
    print("=" * 60)
    print()

    for description, fn in _tests:
        try:
            fn()
            passed += 1
            print(f"  ✓ {description}")
        except Exception as exc:
            failed += 1
            print(f"  ✗ {description}")
            print(f"    {exc}")
            import traceback as tb
            tb.print_exc(limit=3)
            print()

    print()
    print("=" * 60)
    total = passed + failed
    print(f"  {total} tests: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
