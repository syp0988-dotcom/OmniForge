"""Standalone MockLLMService for use in tests.

Usage::

    from tests.mock_llm import MockLLMService

    # Replace the global LLM service with a mock
    with MockLLMService.as_default(answer="Mock response"):
        result = my_function_that_calls_llm()
        assert result == "Mock response"

    # Simulate an LLM timeout
    with MockLLMService.as_default(raise_on_call=TimeoutError("LLM timeout")):
        ...

    # Simulate specific responses per prompt keyword
    with MockLLMService.as_default(responses={"planner": '{"tasks": []}', "default": "ok"}):
        ...
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from agentflow.services.llm_service import LLMResponse

# Modules that import get_llm_service at the module level — these need
# their function reference patched so agents constructed after the mock
# is installed use the mock instead of the real service.
_AGENT_MODULES = [
    "agentflow.agents.goal_analyzer.agent",
    "agentflow.agents.planner.agent",
    "agentflow.agents.reflection.agent",
    "agentflow.agents.answer.agent",
]


def _patch_agent_get_llm_service(
    new_get: object,
) -> None:
    """Replace ``get_llm_service`` in all agent modules that import it.

    This ensures agents constructed *after* the mock is installed
    pick up the mock.  Pass the *original* function to restore.
    """
    for mod_name in _AGENT_MODULES:
        try:
            mod = __import__(mod_name, fromlist=["get_llm_service"])
            mod.get_llm_service = new_get  # type: ignore[assignment]
        except (ImportError, AttributeError):
            pass  # Module not yet loaded — will use real on first import


class MockLLMService:
    """Test double for ``LLMService`` that returns canned responses.

    Attributes:
        is_mock: Always ``True`` — detected by ``PlannerAgent._handle_mock_project``.
        responses: Dict of ``(prompt_keyword: str) → (response: str)``.
            ``"default"`` key is used when no keyword matches.
        raise_on_call: If set, all calls raise this exception (simulates LLM failure).
        last_prompt: The last prompt string passed to ``complete()``.
        last_messages: The last messages list passed to ``complete()``.
        call_count: How many times ``complete()`` was called.
    """

    def __init__(
        self,
        answer: str = "",
        responses: dict[str, str] | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self.is_mock = True
        self._answer = answer
        self._responses = responses or {}
        self._raise_on_call = raise_on_call

        # Call tracking
        self.last_prompt = ""
        self.last_messages: list[dict[str, str]] = []
        self.call_count = 0

    # ── Public API matching LLMService ──

    @property
    def client(self) -> None:
        """Return None — no real client needed for mocks."""
        return None

    def complete(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        session_state: object | None = None,
        **kwargs: Any,  # Accept tool_choice etc. for compat
    ) -> str:
        """Return a canned response based on prompt content."""
        self._track_call(prompt, messages)
        self._maybe_raise()

        if self._answer:
            return self._answer

        # Match against response keywords in prompt or messages
        search_text = prompt or ""
        if not search_text and messages:
            search_text = " ".join(m.get("content", "") for m in messages)
        for keyword, response in self._responses.items():
            if keyword in search_text:
                return response

        # Fall back to "default" keyword, then empty
        return self._responses.get("default", "")

    def complete_stream(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        session_state: object | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield canned response character by character."""
        self._track_call(prompt, messages)
        self._maybe_raise()

        text = self.complete(prompt=prompt, messages=messages, session_state=session_state)
        for ch in text:
            yield ch

    def complete_with_tools(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        session_state: object | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return a canned ``LLMResponse``."""
        self._track_call(prompt, messages)
        self._maybe_raise()

        content = self.complete(prompt=prompt, messages=messages, session_state=session_state)
        return LLMResponse(content=content, tool_calls=None, degraded=False)

    # ── Tracking ──

    def _track_call(
        self, prompt: str | None, messages: list[dict[str, str]] | None
    ) -> None:
        self.call_count += 1
        if prompt:
            self.last_prompt = prompt
        if messages:
            self.last_messages = messages

    def _maybe_raise(self) -> None:
        if self._raise_on_call:
            raise self._raise_on_call

    # ── Convenience: patch the global singleton ──

    @classmethod
    @contextlib.contextmanager
    def as_default(
        cls,
        answer: str = "",
        responses: dict[str, str] | None = None,
        raise_on_call: Exception | None = None,
    ):
        """Context manager that replaces ``_llm_service`` with a mock.

        Usage::

            with MockLLMService.as_default(answer="Hello"):
                result = llm_service.complete(prompt="hi")
                assert result == "Hello"

        The original service is restored on exit.
        """
        import agentflow.services.llm_service as _llm_mod
        original_svc = _llm_mod._llm_service
        original_get = _llm_mod.get_llm_service
        mock = cls(answer=answer, responses=responses, raise_on_call=raise_on_call)

        def _mock_get() -> MockLLMService:  # type: ignore[misc]
            return mock

        # Patch both the singleton and the factory function
        _llm_mod._llm_service = mock  # type: ignore[assignment]
        _llm_mod.get_llm_service = _mock_get  # type: ignore[assignment]

        # Also patch agent-level imports so agents constructed AFTER the
        # patch pick up the mock (needed for build_workflow cache reset).
        _patch_agent_get_llm_service(_mock_get)

        try:
            yield mock
        finally:
            _llm_mod._llm_service = original_svc
            _llm_mod.get_llm_service = original_get  # type: ignore[assignment]
            _patch_agent_get_llm_service(original_get)
