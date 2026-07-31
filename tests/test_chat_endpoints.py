"""Tests for the /chat and /chat/stream API endpoints with mocked LLM.

Uses ``MockLLMService`` to replace the LLM service so tests run
without consuming real API calls.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from agentflow.app.main import app
from agentflow.graph.workflow import reset_workflow_cache

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset():
    """Reset workflow cache and mock LLM for each test."""
    reset_workflow_cache()
    yield


# ── /chat endpoint ──


class TestChatEndpoint:
    """Tests for POST /chat."""

    def test_chat_returns_reply(self):
        """POST /chat should return a reply and session_id."""
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(answer="模拟回答。"):
            response = client.post(
                "/chat",
                json={"message": "你好", "history": []},
            )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data, f"Expected reply in response: {data}"
        assert data["reply"], "Reply should not be empty"
        assert "metadata" in data
        assert "session_id" in data["metadata"]

    def test_chat_with_source_mode(self):
        """source_mode parameter should be passed through to the workflow."""
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(answer="搜索结果回答。"):
            response = client.post(
                "/chat",
                json={"message": "搜索一下", "history": [], "source_mode": "web"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data

    def test_chat_with_history(self):
        """Conversation history should be included in the request."""
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(answer="后续回答。"):
            response = client.post(
                "/chat",
                json={
                    "message": "继续说",
                    "history": [
                        {"role": "user", "content": "第一句话"},
                        {"role": "assistant", "content": "第一句回复"},
                    ],
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data

    def test_chat_returns_debug_data(self):
        """The response should include debug metadata."""
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(answer="调试回答。"):
            response = client.post(
                "/chat",
                json={"message": "调试模式", "history": []},
            )
        assert response.status_code == 200
        data = response.json()
        # debug field should contain goal and category info
        debug = data.get("debug", {})
        assert "goal" in debug or "category" in debug

    def test_chat_creates_session_when_missing(self):
        """When no session_id is provided, a new session is created."""
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(answer="新会话回答。"):
            response = client.post(
                "/chat",
                json={"message": "新建会话", "history": []},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["session_id"] is not None

    def test_chat_handles_empty_message(self):
        """An empty message should not crash the server."""
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(answer=""):
            response = client.post(
                "/chat",
                json={"message": "", "history": []},
            )
        # May return 200 or 422 depending on validation
        assert response.status_code in (200, 422)


# ── /chat/stream endpoint ──


class TestChatStreamEndpoint:
    """Tests for POST /chat/stream (SSE streaming)."""

    def test_stream_returns_sse_events(self):
        """Should return a stream of SSE events ending with 'done'."""
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(answer="流式回答。"):
            response = client.post(
                "/chat/stream",
                json={"message": "流式测试", "history": []},
            )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE events
        events = _parse_sse(response.text)
        event_types = [e["event"] for e in events]

        assert "start" in event_types, f"Expected 'start' event, got {event_types}"
        assert "done" in event_types, f"Expected 'done' event, got {event_types}"

    def test_stream_answer_text_delivered(self):
        """The stream should deliver answer text via 'text' events."""
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(answer="这是一段测试正文回答。"):
            response = client.post(
                "/chat/stream",
                json={"message": "正文测试", "history": []},
            )
        assert response.status_code == 200

        events = _parse_sse(response.text)
        text_parts = []
        for e in events:
            if e["event"] == "text":
                text_parts.append(e["data"].get("text", ""))
        full_text = "".join(text_parts)
        # Should have text content (either the mock answer or degraded fallback)
        assert len(full_text) > 0, "Should have text content in stream"
        # The done event should have the final answer
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1
        assert "answer" in done_events[0]["data"]

    def test_stream_event_sequence(self):
        """The stream event sequence should start with 'start' and end with 'done'."""
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(answer="测试序列。"):
            response = client.post(
                "/chat/stream",
                json={"message": "序列测试", "history": []},
            )
        assert response.status_code == 200
        events = _parse_sse(response.text)
        assert events[0]["event"] == "start"
        # One of the last events should be 'done'
        last_events = [e["event"] for e in events[-3:]]
        assert "done" in last_events, f"Expected 'done' among last events: {events[-3:]}"


# ── Helper ──


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into a list of event dicts."""
    events: list[dict] = []
    for line in text.strip().split("\n\n"):
        if not line.strip():
            continue
        event = "message"
        data: dict = {}
        for part in line.split("\n"):
            if part.startswith("event: "):
                event = part[7:]
            elif part.startswith("data: "):
                try:
                    data = json.loads(part[6:])
                except json.JSONDecodeError:
                    data = {"raw": part[6:]}
        events.append({"event": event, "data": data})
    return events
