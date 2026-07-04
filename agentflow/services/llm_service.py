from __future__ import annotations

from openai import OpenAI

from agentflow.config.settings import settings


class LLMService:
    """Thin wrapper around the OpenAI-compatible DeepSeek client."""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)

    def complete(self, prompt: str) -> str:
        """Generate a completion using the configured model."""
        response = self.client.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        return response.choices[0].message.content or ""
