"""Tests for QueryRewriter — search query optimisation.

Covers:
  - Filler stripping ("帮我查一下", "请问")
  - Time normalisation (今天→今日, 现在→当前)
  - Context recovery from SessionState (current_goal, tracking)
  - Intent-aware query building (weather→add "天气")
  - Self-contained vs. context-dependent queries
  - Edge cases: empty, short, standalone keywords
"""

from __future__ import annotations

import pytest

from agentflow.agents.search.query_rewriter import QueryRewriter
from agentflow.conversation.session_state import SessionState
from agentflow.conversation.state import ConversationState

qr = QueryRewriter()


# ---------------------------------------------------------------------------
# Filler stripping
# ---------------------------------------------------------------------------


class TestFillerStripping:
    def test_remove_qingwen(self):
        result = qr._strip_filler("请问今天天气怎么样")
        assert "今天天气怎么样" in result

    def test_remove_bangwo(self):
        result = qr._strip_filler("帮我查一下杭州天气")
        assert "杭州天气" in result

    def test_remove_woxiang(self):
        result = qr._strip_filler("我想知道北京天气")
        assert "北京天气" in result

    def test_remove_ket_combined(self):
        result = qr._strip_filler("可以帮我查一下上海天气吗")
        assert "上海天气" in result

    def test_remove_trailing_xiexie(self):
        result = qr._strip_filler("搜索杭州天气 谢谢")
        assert "谢谢" not in result
        assert "杭州天气" in result

    def test_no_filler_left(self):
        assert qr._strip_filler("杭州 今日 天气") == "杭州 今日 天气"

    def test_empty_after_strip(self):
        assert qr._strip_filler("谢谢") == ""


# ---------------------------------------------------------------------------
# Time normalisation
# ---------------------------------------------------------------------------


class TestTimeNormalisation:
    def test_jintian(self):
        assert qr._normalise_time("今天天气") == "今日天气"

    def test_mingtian(self):
        assert qr._normalise_time("明天天气") == "明日天气"

    def test_zuotian(self):
        assert qr._normalise_time("昨天新闻") == "昨日新闻"

    def test_xianzai(self):
        assert qr._normalise_time("现在价格") == "当前价格"

    def test_zuijin(self):
        assert qr._normalise_time("最近AI新闻") == "最新AI新闻"

    def test_no_time_ref(self):
        assert qr._normalise_time("杭州气温") == "杭州气温"


# ---------------------------------------------------------------------------
# Context recovery from SessionState
# ---------------------------------------------------------------------------


class TestContextRecovery:
    def test_recover_from_current_goal(self):
        ss = SessionState(current_goal="查询杭州天气")
        context = qr._recover_context("今日", ss, intent="weather")
        # The goal contains "杭州" and "天气", question is just "今日"
        # so context should return the goal
        assert context == "查询杭州天气"

    def test_recover_from_tracking_focus(self):
        tracker = ConversationState(current_focus="杭州")
        ss = SessionState(current_goal="查询天气", tracking=tracker)
        context = qr._recover_context("今天天气", ss, intent="weather")
        assert context == "杭州"

    def test_recover_from_tracking_topic(self):
        tracker = ConversationState(topic="黄金价格")
        ss = SessionState(tracking=tracker)
        context = qr._recover_context("当前价格", ss, intent="")
        assert context == "黄金价格"

    def test_recover_from_slots(self):
        ss = SessionState(slots={"city": "成都", "date": "2026-07-06"})
        context = qr._recover_context("天气", ss, intent="weather")
        assert "成都" in context

    def test_no_context(self):
        ss = SessionState()
        context = qr._recover_context("今天天气怎么样", ss, intent="")
        assert context == ""

    def test_none_session_state(self):
        context = qr._recover_context("今天天气", None, intent="")
        assert context == ""


# ---------------------------------------------------------------------------
# Full rewrite (integration)
# ---------------------------------------------------------------------------


class TestFullRewrite:
    def test_weather_first_turn(self):
        """User asks '今天天气怎么样' with no context → stays as-is."""
        result = qr.rewrite("今天天气怎么样", session_state=SessionState(), intent="weather")
        # Should keep self-contained query
        assert "天气" in result

    def test_weather_with_city_second_turn(self):
        """User says '杭州' after agent asked for city."""
        ss = SessionState(
            current_goal="查询天气",
            slots={"city": ""},
        )
        result = qr.rewrite("杭州", session_state=ss, intent="weather")
        assert "杭州" in result
        assert "天气" in result or "天气" in result

    def test_weather_self_contained(self):
        """User asks a self-contained weather question."""
        result = qr.rewrite("杭州 今日 天气", session_state=SessionState(), intent="weather")
        assert "杭州" in result
        assert "今日" in result or "天气" in result

    def test_strip_and_normalise(self):
        """Filler stripping + time normalisation together."""
        result = qr.rewrite("请问今天上海天气怎么样", session_state=SessionState(), intent="weather")
        assert "请问" not in result
        assert "今日" in result
        assert "上海" in result
        assert "天气" in result

    def test_cross_turn_context_recovery(self):
        """Second turn: '我在杭州' → recovers goal from session state."""
        tracker = ConversationState()
        tracker.topic = "天气"
        ss = SessionState(
            current_goal="查询天气",
            tracking=tracker,
        )
        result = qr.rewrite("我在杭州", session_state=ss, intent="weather")
        # Should keep "杭州" and append intent suffix "天气"
        assert "杭州" in result
        assert "天气" in result

    def test_news_query(self):
        """News search with time normalisation."""
        result = qr.rewrite("最近AI新闻", session_state=SessionState(), intent="news")
        assert "最新" in result or "AI" in result

    def test_empty_input(self):
        assert qr.rewrite("", session_state=SessionState(), intent="") == ""

    def test_preserve_entities(self):
        """Named entities should be preserved."""
        result = qr.rewrite("GPT-5 最新进展", session_state=SessionState(), intent="news")
        assert "GPT-5" in result

    def test_no_guessing(self):
        """Without context, don't add info the user didn't provide."""
        ss = SessionState(current_goal="查询天气")
        result = qr.rewrite("今天天气怎么样", session_state=ss, intent="weather")
        # Should NOT contain a city name since user didn't provide one
        known_cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
        assert not any(c in result for c in known_cities), (
            f"Should not guess city, got: {result}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_standalone_keyword(self):
        assert qr._is_standalone_keyword("杭州")
        assert qr._is_standalone_keyword("GPT-5")
        assert qr._is_standalone_keyword("OpenAI")
        assert not qr._is_standalone_keyword("我在杭州")
        assert not qr._is_standalone_keyword("帮我查天气")

    def test_is_self_contained_short(self):
        assert not qr._is_self_contained("杭州")
        assert qr._is_self_contained("杭州 今日 天气 怎么样")
        assert qr._is_self_contained("今日天气")

    def test_strip_filler_multiple(self):
        text = "可以帮我查一下今天杭州的天气吗"
        result = qr._strip_filler(text)
        assert "可以" not in result
        assert "帮我" not in result
        assert "今天" in result
        assert "杭州" in result

    def test_normalise_time_only(self):
        assert qr._normalise_time("现在") == "当前"
        assert qr._normalise_time("这几天") == "近期"
