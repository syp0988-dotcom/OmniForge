"""ConversationContext — structured context for a single conversation turn.

Captures *why* the user said what they said, what it means in context,
and what type of turn this is — so downstream agents (Answer, Planner,
Router) can make better decisions without re-deriving context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Turn type constants — each describes the conversational function of this turn
NEW_TASK = "NEW_TASK"
FOLLOW_UP = "FOLLOW_UP"
OPTION_SELECTION = "OPTION_SELECTION"
WAITING_REPLY = "WAITING_REPLY"
CLARIFICATION = "CLARIFICATION"
QUESTION_REWRITE = "QUESTION_REWRITE"


@dataclass
class ConversationContext:
    """Rich context for a single conversation turn.

    Fields:
        type: Conversational function of this turn
            (``NEW_TASK`` | ``FOLLOW_UP`` | ``OPTION_SELECTION`` |
             ``WAITING_REPLY`` | ``CLARIFICATION`` | ``QUESTION_REWRITE``).
        original_question: Raw user input before any resolution.
        rewritten_question: Context-enriched version passed to downstream nodes.
        current_goal: The high-level goal from session state.
        last_topic: The topic of the previous assistant turn.
        waiting_for: What the system is waiting for from the user.
        entities: Key entities mentioned in the question.
        summary: Brief summary of recent conversation.
    """

    type: str = NEW_TASK
    original_question: str = ""
    rewritten_question: str = ""
    current_goal: str = ""
    last_topic: str = ""
    waiting_for: str = ""
    entities: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "type": self.type,
            "original_question": self.original_question,
            "rewritten_question": self.rewritten_question,
            "current_goal": self.current_goal,
            "last_topic": self.last_topic,
            "waiting_for": self.waiting_for,
            "entities": list(self.entities),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ConversationContext:
        """Restore from a dict produced by ``to_dict``."""
        if not data:
            return cls()
        return cls(
            type=str(data.get("type", NEW_TASK)),
            original_question=str(data.get("original_question", "")),
            rewritten_question=str(data.get("rewritten_question", "")),
            current_goal=str(data.get("current_goal", "")),
            last_topic=str(data.get("last_topic", "")),
            waiting_for=str(data.get("waiting_for", "")),
            entities=list(data.get("entities", [])),
            summary=str(data.get("summary", "")),
        )

    def __str__(self) -> str:
        """Human-readable summary for prompt injection."""
        parts = [f"对话类型：{self.type}"]
        if self.rewritten_question and self.rewritten_question != self.original_question:
            parts.append(f"重写后问题：{self.rewritten_question}")
        if self.current_goal:
            parts.append(f"当前目标：{self.current_goal}")
        if self.last_topic:
            parts.append(f"上一个话题：{self.last_topic}")
        if self.waiting_for:
            parts.append(f"等待用户：{self.waiting_for}")
        if self.summary:
            parts.append(f"对话摘要：{self.summary}")
        if self.entities:
            parts.append(f"实体：{'、'.join(self.entities)}")
        return "\n".join(parts)
