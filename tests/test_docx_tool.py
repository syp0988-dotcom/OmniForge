"""Tests for the DOCX report tool."""


from agentflow.tools.docx_tool import DocxTool


def _tool(tmp_path) -> DocxTool:
    return DocxTool(workspace=str(tmp_path))


def test_create_docx(tmp_path):
    tool = _tool(tmp_path)
    result = tool.execute(action="create", path="report.docx", content="# 标题\n\n正文内容")
    assert result.success, result.error
    assert (tmp_path / "report.docx").exists()
    assert result.result.get("path", "").endswith("report.docx")


def test_read_docx_roundtrip(tmp_path):
    tool = _tool(tmp_path)
    tool.execute(action="create", path="r.docx", content="# 标题\n\n正文内容")
    result = tool.execute(action="read", path="r.docx")
    assert result.success
    text = str(result.result)
    assert "标题" in text and "正文内容" in text


def test_replace_text(tmp_path):
    tool = _tool(tmp_path)
    tool.execute(action="create", path="r.docx", content="hello world")
    result = tool.execute(action="replace_text", path="r.docx", old_text="world", new_text="OmniForge")
    assert result.success
    read = tool.execute(action="read", path="r.docx")
    assert "OmniForge" in str(read.result)


def test_validate_docx(tmp_path):
    tool = _tool(tmp_path)
    tool.execute(action="create", path="r.docx", content="ok")
    result = tool.execute(action="validate", path="r.docx")
    # Validation depends on an optional external dependency (defusedxml);
    # the tool must never raise and always return a ToolResult envelope.
    assert isinstance(result.success, bool)
    if not result.success:
        assert "defusedxml" in (result.error or "")


def test_read_tables(tmp_path):
    tool = _tool(tmp_path)
    tool.execute(action="create", path="t.docx", content="| a | b |\n|---|---|\n| 1 | 2 |")
    result = tool.execute(action="read_tables", path="t.docx")
    assert result.success
    assert "1" in str(result.result) or "2" in str(result.result)


def test_unknown_action_fails(tmp_path):
    tool = _tool(tmp_path)
    result = tool.execute(action="does_not_exist")
    assert not result.success
    assert "Unknown action" in (result.error or "")


def test_missing_path_fails(tmp_path):
    tool = _tool(tmp_path)
    result = tool.execute(action="create")
    assert not result.success


def test_actions_metadata(tmp_path):
    tool = _tool(tmp_path)
    actions = tool.actions()
    assert "create" in actions
    assert "read" in actions
    assert "replace_text" in actions
