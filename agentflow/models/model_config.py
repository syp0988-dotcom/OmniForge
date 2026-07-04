"""Pydantic models for LLM model configuration CRUD."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMModelCreate(BaseModel):
    """Request model for creating a new LLM model configuration."""

    name: str = Field(..., description="Human-readable name")
    provider: str = Field(default="custom", description="Provider slug (deepseek, openai, custom, etc.)")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(default="", description="API key")
    model_name: str = Field(..., description="Model identifier (e.g. deepseek-chat, gpt-4o)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128000)


class LLMModelUpdate(BaseModel):
    """Request model for updating an existing LLM model configuration."""

    name: str | None = None
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class LLMModelResponse(BaseModel):
    """Response model for LLM model configuration (API key excluded)."""

    id: int
    name: str
    provider: str
    base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    is_active: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_db_row(cls, row: dict[str, Any]) -> LLMModelResponse:
        """Construct a response from a database row dict (api_key is stripped)."""
        return cls(
            id=row["id"],
            name=row["name"],
            provider=row["provider"],
            base_url=row["base_url"],
            model_name=row["model_name"],
            temperature=row["temperature"],
            max_tokens=row["max_tokens"],
            is_active=row["is_active"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
