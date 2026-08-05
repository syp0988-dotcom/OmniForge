"""Tests for the project blueprint loader / renderer / configurator."""

from agentflow.blueprints.configurator import ProjectConfigurator
from agentflow.blueprints.loader import BlueprintLoader
from agentflow.blueprints.models import FileSpec


def test_loader_lists_blueprints():
    loader = BlueprintLoader()
    ids = loader.list()
    assert "fastapi-restful" in ids
    assert "flask-basic" in ids
    assert "react-spa" in ids


def test_loader_get_and_match():
    loader = BlueprintLoader()
    bp = loader.get("fastapi-restful")
    assert bp is not None
    assert bp.name
    assert bp.keywords

    match = loader.match("帮我创建一个 FastAPI 项目", "project")
    assert match is not None


def test_loader_match_returns_none_for_chat():
    loader = BlueprintLoader()
    assert loader.match("你好", "other") is None


def test_render_produces_file_specs():
    loader = BlueprintLoader()
    bp = loader.get("fastapi-restful")
    config = ProjectConfigurator.from_goal("帮我创建一个 FastAPI 图书管理系统")
    specs = loader.render(bp, config)
    assert isinstance(specs, list)
    assert len(specs) > 0
    for spec in specs:
        assert isinstance(spec, FileSpec)
        assert spec.path
        # Rendered paths must not contain unresolved Jinja braces.
        assert "{{" not in spec.path


def test_render_marks_existing_files_skipped():
    loader = BlueprintLoader()
    bp = loader.get("fastapi-restful")
    config = ProjectConfigurator.from_goal("创建 FastAPI 项目")
    first = loader.render(bp, config)[0]
    specs = loader.render(bp, config, existing_files={first.path})
    skipped = [s for s in specs if s.path == first.path]
    assert skipped and skipped[0].type == "skip"


def test_configurator_detects_framework():
    config = ProjectConfigurator.from_goal("帮我创建一个 FastAPI 项目")
    assert config is not None
    assert "fastapi" in str(config.to_dict()).lower() or "fastapi" in str(config)
