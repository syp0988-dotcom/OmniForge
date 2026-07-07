from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from agentflow.config.settings import settings
from agentflow.database.sqlite import SQLiteStore
from agentflow.utils.logging import build_logger

logger = build_logger("llm")

# Retry configuration for LLM calls
_MAX_RETRIES = 2
_BASE_DELAY = 1.0
_MAX_DELAY = 10.0


@dataclass
class ToolCall:
    """Represents a single function-call request from the LLM."""

    id: str
    name: str  # e.g. "filesystem.mkdir"
    arguments: str  # JSON string of parameters


@dataclass
class LLMResponse:
    """Rich response from ``complete_with_tools``.

    Attributes:
        content: Textual reply (when no tool is invoked).
        tool_calls: List of tool-call requests from the LLM.
    """

    content: str = ""
    tool_calls: list[ToolCall] | None = None


class LLMService:
    """Thin wrapper around an OpenAI-compatible client.

    Supports two modes:
    1. Database-driven: an active model config is loaded from the llm_models table.
    2. Env-driven: falls back to settings.py / .env configuration.

    Accepts an optional ``db`` for dependency injection (e.g. a mock in tests).
    When omitted, a default ``SQLiteStore`` is created automatically.
    """

    def __init__(self, db: SQLiteStore | None = None) -> None:
        self._client: Any | None = None
        self._model_name: str = settings.model_name
        self._temperature: float = settings.temperature
        self._max_tokens: int = settings.max_tokens
        self._db = db or SQLiteStore()
        self._try_load_active_model()

    def _try_load_active_model(self) -> None:
        """If a model is marked active in the database, override env settings."""
        try:
            active = self._db.get_active_model()
            if active and active.get("api_key"):
                self._init_client(
                    api_key=active["api_key"],
                    base_url=active["base_url"],
                )
                self._model_name = active["model_name"]
                self._temperature = active.get("temperature", settings.temperature)
                self._max_tokens = active.get("max_tokens", settings.max_tokens)
                logger.info("Using active model: %s (%s)", active["name"], active["model_name"])
                return
        except Exception:
            logger.debug("No active model in database, falling back to env config")

        # Fallback to env-based client
        if settings.deepseek_api_key:
            self._init_client(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )

    def _init_client(self, api_key: str, base_url: str) -> None:
        """Initialize the OpenAI client with the given credentials."""
        import os as _os
        cert = _os.environ.get("SSL_CERT_FILE", "")
        if cert and not _os.path.exists(cert):
            _os.environ.pop("SSL_CERT_FILE", None)
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def use_model(self, model_config: dict) -> None:
        """Switch to a specific model configuration at runtime."""
        if model_config.get("api_key"):
            self._init_client(
                api_key=model_config["api_key"],
                base_url=model_config.get("base_url", settings.deepseek_base_url),
            )
        self._model_name = model_config.get("model_name", settings.model_name)
        self._temperature = model_config.get("temperature", settings.temperature)
        self._max_tokens = model_config.get("max_tokens", settings.max_tokens)
        logger.info("Switched to model: %s", self._model_name)

    @property
    def client(self) -> Any | None:
        return self._client

    def _call_with_retry(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM with exponential backoff retry.

        Retries up to ``_MAX_RETRIES`` times with jitter between attempts.
        All exceptions are re-raised after exhausting retries so the caller
        can apply its own fallback logic.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
                    delay += random.uniform(0, 0.5)  # jitter
                    logger.warning(
                        "LLM request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, _MAX_RETRIES + 1, exc, delay,
                    )
                    time.sleep(delay)
        # All retries exhausted
        raise last_exc  # type: ignore[misc]

    def complete(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a completion using the configured model or a deterministic fallback.

        Retries transient failures with exponential backoff before falling back.
        """
        if not self.client:
            logger.warning("No API key configured; using fallback response")
            if prompt is None:
                return ""
            return f"[fallback] {prompt[:160]}"

        if messages is None:
            if prompt is None:
                logger.warning("No prompt or messages provided to LLMService.complete")
                return ""
            messages = [{"role": "user", "content": prompt}]

        try:
            return self._call_with_retry(messages)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("LLM request failed after %d retries: %s", _MAX_RETRIES + 1, exc)
            if prompt is None:
                fallback_content = messages[-1].get("content", "")[:160]
                return f"[fallback] {fallback_content}" if fallback_content else ""
            return f"[fallback] {prompt[:160]}"

    # ------------------------------------------------------------------
    # Function calling support
    # ------------------------------------------------------------------

    def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> LLMResponse:
        """Call the LLM with OpenAI-compatible function definitions.

        Args:
            messages: Chat messages (system + user).
            tools: List of OpenAI-compatible function definitions.
            tool_choice: ``"auto"`` (default), ``"required"``, ``"none"``,
                or ``{"type": "function", "function": {"name": "..."}}``.

        Returns:
            ``LLMResponse`` with ``content`` and/or ``tool_calls``.
        """
        if not self.client:
            logger.warning("No API key configured; returning empty response")
            return LLMResponse(content="")

        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message

                content = msg.content or ""
                raw_calls = msg.tool_calls

                if not raw_calls:
                    return LLMResponse(content=content)

                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in raw_calls
                ]
                return LLMResponse(content=content, tool_calls=tool_calls)

            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
                    delay += random.uniform(0, 0.5)
                    logger.warning(
                        "LLM request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, _MAX_RETRIES + 1, exc, delay,
                    )
                    time.sleep(delay)

        logger.exception("LLM request failed after %d retries: %s", _MAX_RETRIES + 1, last_exc)
        return LLMResponse(content="")


_llm_service = LLMService()


def get_llm_service() -> LLMService:
    """Return the shared LLM service instance for agents."""
    return _llm_service
