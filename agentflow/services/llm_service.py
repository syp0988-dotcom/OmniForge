from __future__ import annotations

from typing import Any

from openai import OpenAI

from agentflow.config.settings import settings
from agentflow.utils.logging import build_logger

logger = build_logger("llm")


class LLMService:
    """Thin wrapper around the OpenAI-compatible DeepSeek client."""

    def __init__(self) -> None:
        self.client: Any | None = None
        if settings.deepseek_api_key:
            self.client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)

    def complete(self, prompt: str) -> str:
        """Generate a completion using the configured model or a deterministic fallback."""
        if not self.client:
            logger.warning("DeepSeek API key is not configured; using fallback response")
            return f"[fallback] {prompt[:160]}"

        try:
            response = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("LLM request failed: %s", exc)
            return f"[fallback] {prompt[:160]}"


_llm_service = LLMService()


def get_llm_service() -> LLMService:
    """Return the shared LLM service instance for agents."""
    return _llm_service
