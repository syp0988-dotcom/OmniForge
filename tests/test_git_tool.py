"""Tests for the Git tool using a real temporary repository."""

from pathlib import Path

from agentflow.tools.git_tool import GitTool


def _repo(tmp_path) -> tuple[GitTool, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    tool = GitTool(repo_path=str(repo))
    tool._git("init")
    tool._git("config", "user.email", "test@example.com")
    tool._git("config", "user.name", "Test")
    return tool, repo


def test_status_empty_repo(tmp_path):
    tool, _ = _repo(tmp_path)
    result = tool.cmd_status()
    assert result.success


def test_add_and_commit(tmp_path):
    tool, repo = _repo(tmp_path)
    (repo / "file.txt").write_text("hello", encoding="utf-8")

    status = tool.cmd_status()
    assert "file.txt" in status.result["output"]

    added = tool.cmd_add("file.txt")
    assert added.success

    committed = tool.cmd_commit("init commit")
    assert committed.success
    assert "init commit" in committed.result["output"]

    log = tool.cmd_log(count=5)
    assert log.success
    assert "init commit" in log.result["output"]


def test_branch_operations(tmp_path):
    tool, repo = _repo(tmp_path)
    (repo / "a.txt").write_text("a", encoding="utf-8")
    tool.cmd_add("a.txt")
    tool.cmd_commit("first")

    created = tool.cmd_branch("feature")
    assert created.success

    branches = tool.cmd_branch()
    assert branches.success
    assert "feature" in branches.result["output"]

    checked = tool.cmd_checkout("feature")
    assert checked.success


def test_diff(tmp_path):
    tool, repo = _repo(tmp_path)
    (repo / "a.txt").write_text("v1", encoding="utf-8")
    tool.cmd_add("a.txt")
    tool.cmd_commit("first")
    (repo / "a.txt").write_text("v2", encoding="utf-8")

    diff = tool.cmd_diff()
    assert diff.success
    assert "v2" in diff.result["output"]


def test_show(tmp_path):
    tool, repo = _repo(tmp_path)
    (repo / "a.txt").write_text("hello", encoding="utf-8")
    tool.cmd_add("a.txt")
    tool.cmd_commit("first")
    result = tool.cmd_show("HEAD")
    assert result.success


def test_actions_metadata(tmp_path):
    tool, _ = _repo(tmp_path)
    actions = tool.actions()
    for name in ("status", "diff", "add", "commit", "branch", "log", "show"):
        assert name in actions
