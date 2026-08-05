"""Tests for the Composio tool (graceful degradation without the SDK)."""


from agentflow.tools.composio_tool import ComposioTool


def test_sdk_missing_fails_gracefully(monkeypatch):
    tool = ComposioTool()
    monkeypatch.setattr("agentflow.tools.composio_tool._check_composio", lambda: False)
    result = tool.execute(slug="GMAIL_SEND_EMAIL", to="a@b.com")
    assert not result.success
    assert "Composio not configured" in (result.error or "")


def test_missing_slug_fails(monkeypatch):
    tool = ComposioTool()
    monkeypatch.setenv("COMPOSIO_API_KEY", "test-key")
    monkeypatch.setattr(type(tool), "available", property(lambda self: True))
    result = tool.execute()
    assert not result.success
    assert "slug" in (result.error or "").lower()


def test_actions_metadata():
    tool = ComposioTool()
    actions = tool.actions()
    assert "execute" in actions
    assert "slug" in actions["execute"]["parameters"]


def test_ensure_env_loaded_never_raises():
    from agentflow.tools.composio_tool import _ensure_env_loaded

    _ensure_env_loaded()  # must not raise
