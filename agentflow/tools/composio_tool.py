"""ComposioTool — adapter for the Composio platform (500+ tool integrations).

Composio provides unified access to hundreds of third-party services
(Gmail, Slack, GitHub, Notion, Jira, Linear, etc.) through a single SDK.

This tool dynamically discovers and executes Composio tools by slug.
Authentication is managed through Composio's OAuth flow.

Usage via Executor::

    registry.execute_task("composio", slug="GMAIL_SEND_EMAIL",
                          to="user@example.com", subject="Hello", body="World")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult
from agentflow.utils.logging import build_logger

logger = build_logger("composio_tool")

# On Windows, httpx (used by Composio) reads SSL_CERT_FILE at import time.
# If the var points to a missing file, unset it before importing Composio.
_cert = os.environ.get("SSL_CERT_FILE", "")
if _cert and not os.path.exists(_cert):
    os.environ.pop("SSL_CERT_FILE", None)

# Lazy import — SDK import is deferred until the tool is actually needed.
_COMPOSIO_AVAILABLE: bool | None = None


def _ensure_env_loaded() -> None:
    """Ensure .env file is loaded into os.environ.

    Safe to call multiple times; uses ``override=False``.
    """
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        pass


def _check_composio() -> bool:
    """Check whether the Composio SDK can be imported."""
    global _COMPOSIO_AVAILABLE
    if _COMPOSIO_AVAILABLE is not None:
        return _COMPOSIO_AVAILABLE
    try:
        import composio  # noqa: F401
        _COMPOSIO_AVAILABLE = True
    except ImportError:
        _COMPOSIO_AVAILABLE = False
        logger.warning("Composio SDK not installed. Run: pip install composio-core")
    return _COMPOSIO_AVAILABLE


class ComposioTool(BaseTool):
    """Execute tools from the 500+ Composio integrations.

    Accepts a ``slug`` identifying the tool (e.g. ``GMAIL_SEND_EMAIL``,
    ``SLACK_POST_MESSAGE``) plus any parameters required by that tool as
    additional keyword arguments.
    """

    name = "composio"
    description = "Composio platform — 500+ integrations (Gmail, Slack, GitHub, Notion, Jira, Linear, etc.)"

    def __init__(self) -> None:
        self._client: Any = None
        self._init_client()

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Initialise the Composio client if the SDK and API key are available."""
        if not _check_composio():
            return

        _ensure_env_loaded()

        api_key = os.environ.get("COMPOSIO_API_KEY", "")
        if not api_key:
            logger.warning(
                "COMPOSIO_API_KEY not found in environment or .env file. "
                "Run `composio login` or add it to .env"
            )
            return

        from composio import Composio

        try:
            self._client = Composio(api_key=api_key)
            logger.info("Composio client initialised")
        except Exception as exc:
            logger.warning("Composio client init failed: %s", exc)

    @property
    def available(self) -> bool:
        """Whether the Composio client is correctly configured."""
        return self._client is not None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def actions(self) -> dict[str, dict]:
        return {
            "execute": {
                "description": (
                    "通过 Composio 平台调用 500+ 第三方工具"
                    "（Gmail、Slack、GitHub、Notion、Jira、Linear、Google Sheets 等）。"
                    "传入工具 slug 和所需参数（以额外 keyword arguments 形式）"
                ),
                "parameters": {
                    "slug": {
                        "type": "string",
                        "description": (
                            "Composio 工具 slug，例如 "
                            "GMAIL_SEND_EMAIL, SLACK_POST_MESSAGE, "
                            "GITHUB_CREATE_ISSUE, NOTION_CREATE_PAGE, "
                            "LINEAR_CREATE_ISSUE, GOOGLESHEETS_CREATE_SHEET"
                        ),
                    },
                },
                "required": ["slug"],
                "_extra_params": {"additionalProperties": True},
            },
        }

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base["status"] = "available" if self.available else "unconfigured"
        base["integrations"] = "500+"
        return base

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, slug: str = "", **kwargs: Any) -> tuple[bool, str]:
        """Check that a slug is provided and the client is available."""
        if not self.available:
            return False, (
                "Composio is not configured. Set COMPOSIO_API_KEY in your "
                "environment or .env file."
            )
        slug = slug or kwargs.get("slug", "")
        if not slug:
            return False, "No Composio tool slug provided (e.g. GMAIL_SEND_EMAIL)"
        return True, ""

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, slug: str = "", **kwargs: Any) -> ToolResult:
        """Execute a Composio tool by slug.

        Args:
            slug: The Composio tool identifier (e.g. ``GMAIL_SEND_EMAIL``).
            **kwargs: Tool-specific parameters passed directly as arguments.

        Returns:
            ``ToolResult`` with the raw API response in ``result``.
        """
        slug = slug or kwargs.pop("slug", "")

        if not self.available:
            return ToolResult.fail(
                self.name, slug or "unknown",
                "Composio not configured. Set COMPOSIO_API_KEY.",
            )

        if not slug:
            return ToolResult.fail(
                self.name, "execute",
                "No Composio tool slug provided (e.g. GMAIL_SEND_EMAIL)",
            )

        # Collect arguments: everything remaining in kwargs is tool-specific
        # Also support an explicit ``arguments`` dict / JSON string
        arguments = kwargs.pop("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        # Merge extra kwargs on top (they take precedence)
        arguments.update(kwargs)

        try:
            result = self._client.tools.execute(
                slug=slug,
                arguments=arguments,
                dangerously_skip_version_check=True,
            )
            return ToolResult.ok(
                self.name,
                slug,
                result=_safe_result(result),
                message=f"Composio tool '{slug}' executed successfully",
            )
        except Exception as exc:
            logger.exception("Composio tool '%s' failed: %s", slug, exc)
            return ToolResult.fail(self.name, slug, f"Composio error: {exc}")

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def list_connected_accounts(self) -> list[dict[str, Any]]:
        """Return all connected (authorised) accounts."""
        if not self.available:
            return []
        try:
            accounts = self._client.connected_accounts.list()
            return [
                {
                    "id": getattr(a, "id", ""),
                    "toolkit": getattr(a, "toolkit", ""),
                    "status": getattr(a, "status", ""),
                    "alias": getattr(a, "alias", ""),
                }
                for a in (accounts or [])
            ]
        except Exception as exc:
            logger.warning("Failed to list Composio accounts: %s", exc)
            return []

    def initiate_auth(self, toolkit: str, redirect_uri: str | None = None) -> str | None:
        """Start OAuth for a toolkit and return the redirect URL.

        The user must visit the returned URL to authorise the connection.
        """
        if not self.available:
            return None
        try:
            req = self._client.connected_accounts.initiate(
                user_id="default",
                toolkit=toolkit,
                redirect_url=redirect_uri,
            )
            return getattr(req, "redirect_url", None)
        except Exception as exc:
            logger.warning("Composio auth initiation failed: %s", exc)
            return None


# -- Helpers -------------------------------------------------------------------


def _safe_result(result: Any) -> dict[str, Any]:
    """Normalise a Composio response into a JSON-safe dict."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "__dict__"):
        return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
    return {"data": str(result)}


# -- Action map (single "execute" action) ---------------------------------------

_ACTION_MAP: dict[str, Any] = {
    "execute": ComposioTool.execute,
}
