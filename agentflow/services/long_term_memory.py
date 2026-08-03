"""LongTermMemory — cross-session persistent memory for user facts and preferences.

Stores key-value facts extracted from conversations and recalls them
across sessions, enabling the system to remember user preferences,
recurring topics, and important entities over time.
"""

from __future__ import annotations

import re
from typing import Any

from agentflow.database.sqlite import SQLiteStore
from agentflow.utils.logging import build_logger

logger = build_logger("long_term_memory")

# Stop words to filter out from entity extraction
_STOP_WORDS: set[str] = {
    "什么", "怎么", "为什么", "如何", "哪个", "这个", "那个",
    "一下", "一个", "可以", "能够", "需要", "是否", "怎样",
    "这里", "那里", "这些", "那些", "他们", "它们", "我们",
}


class LongTermMemory:
    """Cross-session persistent memory.

    Usage::

        memory = LongTermMemory()
        memory.remember("user_interest", "Python", "topic")
        facts = memory.recall("Python")
    """

    def __init__(self, db: SQLiteStore | None = None) -> None:
        self.db = db or SQLiteStore()

    # -- Public API ----------------------------------------------------------

    def remember(self, key: str, value: str, category: str = "general") -> None:
        """Store a fact in long-term memory.

        Args:
            key: Unique identifier (e.g. ``"user_interest_python"``).
            value: The fact content (e.g. ``"用户对Python感兴趣"``).
            category: Fact category (``"topic"``, ``"preference"``, ``"entity"``).
        """
        self.db.set_long_term_memory(key, value, category)
        logger.debug("Stored memory: [%s] %s = %s", category, key, value[:60])

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Recall facts relevant to *query*.

        Args:
            query: Search text (entity name, topic keyword).
            limit: Maximum number of results.

        Returns:
            List of dicts with keys: key, value, category, updated_at.
        """
        if not query or len(query) < 2:
            return []
        return self.db.search_long_term_memory(query, limit=limit)

    def extract_and_store(
        self,
        question: str,
        answer: str,
        entities: list[str],
        goal: str = "",
    ) -> int:
        """Extract facts from a conversation turn and persist them.

        Returns the number of facts stored.
        """
        stored = 0

        # 1. Store entities as notable terms
        for entity in entities:
            if len(entity) >= 2 and entity not in _STOP_WORDS:
                key = f"entity_{entity.lower()}"
                existing = self.db.get_long_term_memory(key)
                if not existing:
                    self.remember(key, entity, "entity")
                    stored += 1

        # 2. Store user's goal/interest if substantive
        if goal and len(goal) > 4:
            # Extract key topic from goal
            terms = re.findall(r"[一-鿿a-zA-Z0-9一-鿿]{2,}", goal)
            for term in terms[:3]:
                if term.lower() not in _STOP_WORDS and len(term) >= 2:
                    key = f"topic_{term.lower()}"
                    existing = self.db.get_long_term_memory(key)
                    if not existing:
                        self.remember(key, goal[:120], "topic")
                        stored += 1

        # 3. Store answer summary as a recollection
        if answer and len(answer) > 20:
            summary = answer[:200]
            for entity in entities[:2]:
                if len(entity) >= 2:
                    key = f"fact_{entity.lower()}"
                    existing = self.db.get_long_term_memory(key)
                    if not existing:
                        self.remember(key, summary, "fact")
                        stored += 1

        if stored:
            logger.info("Stored %d new long-term memories", stored)
        return stored

    def recall_for_question(self, question: str) -> str:
        """Build a context string of relevant memories for a question.

        Uses a single batch query for all search terms instead of one query
        per term (reduces 5-15 DB round-trips to 1).

        Returns a formatted string suitable for injection into prompts.
        """
        if not question or len(question) < 2:
            return ""

        # Extract search terms from question
        terms = re.findall(r"[一-鿿a-zA-Z0-9一-鿿]{2,}", question)[:5]
        if not terms:
            return ""

        results = self.db.search_long_term_memory_batch(terms, limit_per_term=3)
        if not results:
            return ""

        memories = [f"  - {r['value']}" for r in results]
        return "长期记忆：\n" + "\n".join(memories)

    def get_all(self, category: str = "") -> list[dict[str, Any]]:
        """List all stored memories (for API/management)."""
        return self.db.list_long_term_memories(category=category)

    def forget(self, key: str) -> bool:
        """Delete a specific memory."""
        return self.db.delete_long_term_memory(key)

    def clear(self, category: str = "") -> None:
        """Clear all memories (optionally filtered by category)."""
        self.db.clear_long_term_memories(category=category)
        logger.info("Cleared long-term memories (category='%s')", category)


# -- Shared instance ---------------------------------------------------------

_ltm_instance: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    """Return a shared ``LongTermMemory`` instance.

    ``LongTermMemory`` wraps a ``SQLiteStore`` connection; creating one per
    request wastes a connection handle.  Use this factory for long-lived
    components (workflow nodes, agents) instead of ``LongTermMemory()``.
    """
    global _ltm_instance
    if _ltm_instance is None:
        _ltm_instance = LongTermMemory()
    return _ltm_instance
