"""BrowserTool — web browser automation (interface / placeholder).

This tool provides a uniform interface for browser-based operations:

  - ``open_url`` — navigate to a URL
  - ``extract_text`` — extract visible text from the current page
  - ``screenshot`` — capture a screenshot of the viewport
  - ``click`` — click an element by selector
  - ``input`` — type text into an input field
  - ``scroll`` — scroll the page

The current implementation is a **no-op interface** that documents the
contract.  Real browser automation requires a Playwright / Selenium
integration.

When implementing, replace the ``execute()`` method with concrete calls
to the browser automation library of your choice.
"""

from __future__ import annotations

from typing import Any

from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult


class BrowserTool(BaseTool):
    """Web browser automation (interface — requires concrete driver)."""

    name = "browser"
    description = "Web browser automation — open URL, extract text, screenshot, click, input, scroll"

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless

    def actions(self) -> dict[str, dict]:
        return {
            "open_url": {
                "description": "[接口预留] 在浏览器中打开 URL",
                "parameters": {"url": {"type": "string", "description": "要打开的 URL"}},
                "required": ["url"],
            },
            "extract_text": {
                "description": "[接口预留] 提取当前页面的可见文本",
                "parameters": {},
                "required": [],
            },
            "screenshot": {
                "description": "[接口预留] 截取页面视口截图",
                "parameters": {},
                "required": [],
            },
            "click": {
                "description": "[接口预留] 点击指定 CSS 选择器的元素",
                "parameters": {"selector": {"type": "string", "description": "CSS 选择器"}},
                "required": ["selector"],
            },
            "input_text": {
                "description": "[接口预留] 在输入框中输入文本",
                "parameters": {
                    "selector": {"type": "string", "description": "CSS 选择器"},
                    "text": {"type": "string", "description": "要输入的文本"},
                },
                "required": ["selector"],
            },
            "scroll": {
                "description": "[接口预留] 滚动页面",
                "parameters": {
                    "direction": {"type": "string", "description": "滚动方向（up/down）", "default": "down"},
                    "amount": {"type": "integer", "description": "滚动像素数", "default": 300},
                },
                "required": [],
            },
        }

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base["headless"] = self.headless
        base["status"] = "interface_only"
        base["message"] = (
            "This is an interface placeholder. "
            "Implement with Playwright / Selenium to enable browser automation."
        )
        return base

    def capabilities(self) -> list[str]:
        return ["browser.open", *super().capabilities()]

    def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        handler = _ACTION_MAP.get(action)
        if handler is None:
            return ToolResult.fail(
                self.name, action or "execute",
                f"Unknown browser action '{action}'. "
                f"Available: {', '.join(sorted(_ACTION_MAP))}",
            )
        return handler(self, **kwargs)

    # ==================================================================
    # Interface stubs — replace with concrete implementations
    # ==================================================================

    def cmd_open_url(self, url: str = "", **kwargs: Any) -> ToolResult:
        if not url:
            return ToolResult.fail(self.name, "open_url", "URL is required")
        return ToolResult.fail(
            self.name, "open_url",
            "BrowserTool not yet implemented — integrate Playwright or Selenium",
        )

    def cmd_extract_text(self, **kwargs: Any) -> ToolResult:
        return ToolResult.fail(
            self.name, "extract_text",
            "BrowserTool not yet implemented",
        )

    def cmd_screenshot(self, **kwargs: Any) -> ToolResult:
        return ToolResult.fail(
            self.name, "screenshot",
            "BrowserTool not yet implemented",
        )

    def cmd_click(self, selector: str = "", **kwargs: Any) -> ToolResult:
        if not selector:
            return ToolResult.fail(self.name, "click", "CSS selector is required")
        return ToolResult.fail(
            self.name, "click",
            "BrowserTool not yet implemented",
        )

    def cmd_input_text(self, selector: str = "", text: str = "", **kwargs: Any) -> ToolResult:
        if not selector:
            return ToolResult.fail(self.name, "input_text", "CSS selector is required")
        return ToolResult.fail(
            self.name, "input_text",
            "BrowserTool not yet implemented",
        )

    def cmd_scroll(self, direction: str = "down", amount: int = 300, **kwargs: Any) -> ToolResult:
        return ToolResult.fail(
            self.name, "scroll",
            "BrowserTool not yet implemented",
        )


# -- Action dispatch map --------------------------------------------------------

_ACTION_MAP: dict[str, Any] = {
    "open_url": BrowserTool.cmd_open_url,
    "extract_text": BrowserTool.cmd_extract_text,
    "screenshot": BrowserTool.cmd_screenshot,
    "click": BrowserTool.cmd_click,
    "input_text": BrowserTool.cmd_input_text,
    "scroll": BrowserTool.cmd_scroll,
}
