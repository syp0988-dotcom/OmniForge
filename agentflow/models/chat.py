from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message payload."""

    role: str = Field(..., description="Message role, such as user or assistant")
    content: str = Field(..., description="Message content")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request payload for chat endpoints."""

    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class FileProposal(BaseModel):
    """A code block detected in an agent response, proposed for file creation."""

    suggestion_id: str
    filename: str
    language: str
    content: str
    preview: str


class ChatResponse(BaseModel):
    """Response payload for chat endpoints."""

    reply: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] | None = None
    proposed_files: list[FileProposal] = Field(default_factory=list)
