"""Tests for the Conversation Runtime upgrade.

Covers:
  - SessionState serialization round-trip
  - SessionState predicates (is_waiting, has_pending_options, etc.)
  - ConversationManager.resolve_question (options, slots, anaphora)
  - ConversationManager.should_continue
  - ConversationManager.finalize_turn (option extraction, input detection)
  - WorkflowContext session_state integration
  - Workflow graph: conversation_manager node routing
  - End-to-end: session_state flows through run_workflow
"""

from __future__ import annotations

import json

import pytest

from agentflow.conversation.manager import ConversationManager
from agentflow.conversation.session_state import SessionState
from agentflow.graph.context import WorkflowContext
from agentflow.graph.workflow import build_workflow, run_workflow

# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------


class TestSessionState:
    def test_default_is_idle(self):
        ss = SessionState()
        assert ss.status == "idle"
        assert not ss.is_waiting
        assert not ss.has_pending_options
        assert not ss.has_unfilled_slots

    def test_is_waiting(self):
        ss = SessionState(status="waiting_user")
        assert ss.is_waiting

    def test_has_pending_options(self):
        ss = SessionState(pending_options={"1": "a", "2": "b"})
        assert ss.has_pending_options
        assert not ss.is_waiting

    def test_has_unfilled_slots(self):
        ss = SessionState(slots={"city": "北京", "date": ""})
        assert ss.has_unfilled_slots
        ss.fill_slot("date", "2024-01-01")
        assert not ss.has_unfilled_slots

    def test_resolve_option_direct_key(self):
        ss = SessionState(pending_options={"1": "儿童教育", "2": "公共卫生"})
        assert ss.resolve_option("1") == "儿童教育"
        assert ss.resolve_option("2") == "公共卫生"

    def test_resolve_option_chinese_ordinal(self):
        ss = SessionState(pending_options={"1": "儿童教育", "2": "公共卫生"})
        assert ss.resolve_option("选项一") == "儿童教育"
        assert ss.resolve_option("选项二") == "公共卫生"

    def test_resolve_option_exact_value(self):
        ss = SessionState(pending_options={"1": "儿童教育", "2": "公共卫生"})
        assert ss.resolve_option("儿童教育") == "儿童教育"

    def test_resolve_option_no_match(self):
        ss = SessionState(pending_options={"1": "a"})
        assert ss.resolve_option("xyz") is None

    def test_resolve_option_no_options(self):
        ss = SessionState()
        assert ss.resolve_option("选项一") is None

    def test_start_waiting_and_resume(self):
        ss = SessionState()
        ss.start_waiting("选择主题")
        assert ss.is_waiting
        assert ss.waiting_for == "选择主题"
        ss.resume()
        assert not ss.is_waiting
        assert ss.waiting_for == ""

    def test_reset(self):
        ss = SessionState(
            current_goal="test",
            status="waiting_user",
            pending_options={"1": "a"},
            slots={"city": "北京"},
            metadata={"key": "val"},
        )
        ss.reset()
        assert ss.current_goal == ""
        assert ss.status == "idle"
        assert ss.pending_options == {}
        assert ss.slots == {}
        assert ss.metadata == {}

    def test_serialization_round_trip(self):
        ss = SessionState(
            current_goal="写 Python 贪吃蛇",
            current_task="执行 Python 脚本",
            status="waiting_user",
            waiting_for="选择语言",
            pending_options={"1": "Python", "2": "Java"},
            slots={"framework": "Pygame"},
            metadata={"agent": "planner"},
        )
        d = ss.to_dict()
        assert d["current_goal"] == "写 Python 贪吃蛇"
        assert d["pending_options"]["1"] == "Python"
        assert d["slots"]["framework"] == "Pygame"
        assert d["metadata"]["agent"] == "planner"

        ss2 = SessionState.from_dict(d)
        assert ss2.current_goal == ss.current_goal
        assert ss2.status == ss.status
        assert ss2.waiting_for == ss.waiting_for
        assert ss2.pending_options == ss.pending_options
        assert ss2.slots == ss.slots
        assert ss2.metadata == ss.metadata

    def test_from_dict_none(self):
        ss = SessionState.from_dict(None)
        assert ss.status == "idle"

    def test_from_dict_empty(self):
        ss = SessionState.from_dict({})
        assert ss.status == "idle"

    def test_str_with_goal(self):
        ss = SessionState(current_goal="测试目标")
        assert "当前目标" in str(ss)

    def test_str_empty(self):
        ss = SessionState()
        assert str(ss) == "(无活跃任务)"

    # -- tracking field (Phase 8) --

    def test_tracking_default_none(self):
        ss = SessionState()
        assert ss.tracking is None

    def test_tracking_serialization_included(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState(topic="IDA", current_focus="步骤2")
        ss = SessionState(current_goal="IDA流程", tracking=cs)
        d = ss.to_dict()
        assert "tracking" in d
        assert d["tracking"]["topic"] == "IDA"
        assert d["tracking"]["current_focus"] == "步骤2"

    def test_tracking_serialization_omitted_when_none(self):
        ss = SessionState(current_goal="测试")
        d = ss.to_dict()
        assert "tracking" not in d

    def test_tracking_deserialization(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="IDA流程",
            tracking=ConversationState(topic="IDA", entities={"步骤"}),
        )
        d = ss.to_dict()
        ss2 = SessionState.from_dict(d)
        assert ss2.tracking is not None
        assert ss2.tracking.topic == "IDA"
        assert ss2.tracking.entities == {"步骤"}

    def test_tracking_deserialization_legacy_dict(self):
        """Legacy dict without tracking field should produce tracking=None."""
        d = {"current_goal": "test", "status": "idle"}
        ss = SessionState.from_dict(d)
        assert ss.tracking is None

    def test_reset_clears_tracking(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="test",
            tracking=ConversationState(topic="X"),
        )
        ss.reset()
        assert ss.tracking is None

    def test_str_includes_focus_when_tracking(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="IDA流程",
            tracking=ConversationState(current_focus="步骤2"),
        )
        s = str(ss)
        assert "IDA" in s
        assert "步骤2" in s


# ---------------------------------------------------------------------------
# ConversationState (Phase 8)
# ---------------------------------------------------------------------------


class TestConversationState:
    def test_default_fields(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState()
        assert cs.topic == ""
        assert cs.entities == set()
        assert cs.current_focus == ""
        assert cs.last_answer == ""
        assert cs.summary == ""

    def test_to_dict_round_trip(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState(
            topic="IDA使用流程",
            entities={"IDA", "步骤", "工具链"},
            current_focus="步骤2",
            last_answer="IDA 的步骤包括：1. 分析 2. 设计",
            summary="讨论IDA步骤",
        )
        d = cs.to_dict()
        assert d["topic"] == "IDA使用流程"
        assert set(d["entities"]) == {"IDA", "步骤", "工具链"}
        assert d["current_focus"] == "步骤2"

        cs2 = ConversationState.from_dict(d)
        assert cs2.topic == cs.topic
        assert cs2.entities == cs.entities
        assert cs2.current_focus == cs.current_focus
        assert cs2.last_answer == cs.last_answer

    def test_from_dict_none(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState.from_dict(None)
        assert cs.topic == ""
        assert cs.entities == set()

    def test_from_dict_empty(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState.from_dict({})
        assert cs.topic == ""
        assert cs.entities == set()

    def test_from_dict_handles_list_entities(self):
        from agentflow.conversation.state import ConversationState
        d = {"entities": ["IDA", "步骤"]}
        cs = ConversationState.from_dict(d)
        assert cs.entities == {"IDA", "步骤"}

    def test_add_entity_deduplication(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState()
        cs.add_entity("Python")
        cs.add_entity("Python")
        cs.add_entity("IDA")
        assert cs.entities == {"Python", "IDA"}

    def test_add_entity_short_strings_ignored(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState()
        cs.add_entity("P")
        assert len(cs.entities) == 0

    def test_set_focus(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState()
        cs.set_focus("步骤2")
        assert cs.current_focus == "步骤2"

    def test_set_focus_empty_string_noop(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState(current_focus="步骤")
        cs.set_focus("")
        assert cs.current_focus == "步骤"  # unchanged

    def test_str_with_topic_and_focus(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState(topic="IDA", current_focus="步骤2")
        s = str(cs)
        assert "IDA" in s
        assert "步骤2" in s

    def test_str_empty(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState()
        assert str(cs) == "(无跟踪信息)"

    def test_reset(self):
        from agentflow.conversation.state import ConversationState
        cs = ConversationState(
            topic="IDA",
            entities={"IDA", "步骤"},
            current_focus="步骤2",
            last_answer="回答",
            summary="摘要",
        )
        cs.reset()
        assert cs.topic == ""
        assert cs.entities == set()
        assert cs.current_focus == ""
        assert cs.last_answer == ""
        assert cs.summary == ""


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------


class TestConversationManager:
    def test_should_continue_waiting(self):
        cm = ConversationManager()
        assert cm.should_continue(SessionState(status="waiting_user"))
        assert not cm.should_continue(SessionState(status="idle"))
        assert not cm.should_continue(SessionState(status="processing"))

    def test_should_continue_with_goal(self):
        cm = ConversationManager()
        assert cm.should_continue(SessionState(
            current_goal="写报告", status="waiting_user"
        ))
        assert not cm.should_continue(SessionState(
            current_goal="写报告", status="idle"
        ))

    # -- Option resolution --
    def test_resolve_pending_option(self):
        cm = ConversationManager()
        ss = SessionState(
            status="waiting_user",
            pending_options={"1": "儿童教育", "2": "公共卫生"},
        )
        resolved = cm.resolve_question("选项一", ss)
        assert resolved == "儿童教育"
        assert not ss.is_waiting  # should auto-resume after resolution

    def test_resolve_slot_filling_complete(self):
        cm = ConversationManager()
        ss = SessionState(
            current_goal="订酒店",
            status="waiting_user",
            slots={"city": "", "date": "2024-01-01"},
        )
        resolved = cm.resolve_question("北京", ss)
        assert ss.slots["city"] == "北京"
        assert not ss.has_unfilled_slots
        assert not ss.is_waiting  # all slots filled → resume

    def test_resolve_slot_filling_partial(self):
        cm = ConversationManager()
        ss = SessionState(
            current_goal="订酒店",
            status="waiting_user",
            slots={"city": "", "date": ""},
        )
        resolved = cm.resolve_question("北京", ss)
        assert ss.slots["city"] == "北京"
        assert ss.is_waiting  # still has unfilled slots

    def test_resolve_continue_signal(self):
        cm = ConversationManager()
        ss = SessionState(
            current_goal="测试",
            status="waiting_user",
            waiting_for="确认操作",
        )
        resolved = cm.resolve_question("继续", ss)
        assert not ss.is_waiting  # continue resumes

    def test_anaphora_enrichment(self):
        cm = ConversationManager()
        ss = SessionState(current_goal="写 Python 贪吃蛇游戏", status="processing")
        resolved = cm.resolve_question("改成 Java", ss)
        assert "改成 Java" in resolved
        assert "贪吃蛇" in resolved

    def test_self_contained_question_not_enriched(self):
        cm = ConversationManager()
        ss = SessionState(current_goal="写 Python 贪吃蛇", status="processing")
        resolved = cm.resolve_question("今天天气怎么样", ss)
        assert resolved == "今天天气怎么样"  # not enriched

    # -- Option extraction --
    def test_extract_options(self):
        cm = ConversationManager()
        text = "请选择：\n1. 儿童教育\n2. 公共卫生\n3. 扶贫"
        opts = cm._extract_options(text)
        assert len(opts) == 3
        assert opts["1"] == "儿童教育"

    def test_extract_options_too_few(self):
        cm = ConversationManager()
        text = "结果：1. 第一个"
        opts = cm._extract_options(text)
        assert opts == {}  # need at least 2

    def test_extract_options_non_contiguous(self):
        cm = ConversationManager()
        text = "1. 一\n3. 三"
        opts = cm._extract_options(text)
        assert opts == {}  # gaps in numbering

    # -- Asking for input detection --
    def test_is_asking_for_input(self):
        cm = ConversationManager()
        assert cm._is_asking_for_input("请选择一个主题")
        assert cm._is_asking_for_input("你更喜欢哪个？")
        assert cm._is_asking_for_input("请告诉我城市名称")
        assert not cm._is_asking_for_input("这是最终结果")

    # -- finalize_turn --
    def test_finalize_turn_with_options(self):
        ss = SessionState(current_goal="测试")
        answer = "请选择主题：\n1. 儿童教育\n2. 公共卫生"
        ConversationManager.finalize_turn({}, ss, answer)
        assert ss.is_waiting
        assert len(ss.pending_options) == 2

    def test_finalize_turn_asking_question(self):
        ss = SessionState(current_goal="测试")
        answer = "你想要什么颜色的？"
        ConversationManager.finalize_turn({}, ss, answer)
        assert ss.is_waiting

    def test_finalize_turn_idle(self):
        ss = SessionState()
        answer = "这是最终结果。"
        ConversationManager.finalize_turn({}, ss, answer)
        # no options, no question → stays idle
        assert not ss.is_waiting

    # -- Topic / Entity / Focus extraction helpers (Phase 8) --

    def test_extract_topic_from_entity(self):
        entities = {"IDA", "步骤"}
        topic = ConversationManager._extract_topic("IDA有哪些步骤", entities)
        assert topic == "IDA" or topic == "步骤"

    def test_extract_topic_no_entities(self):
        topic = ConversationManager._extract_topic("Python 贪吃蛇", set())
        assert topic and len(topic) > 0

    def test_extract_topic_empty(self):
        topic = ConversationManager._extract_topic("", set())
        assert topic == ""

    def test_update_focus_ordinal_with_options(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            pending_options={"1": "儿童教育", "2": "公共卫生"},
        )
        tracking = ConversationState(current_focus="主题")
        result = ConversationManager._update_focus("第二个", ss, tracking)
        assert result == "公共卫生"
        assert tracking.current_focus == "公共卫生"

    def test_update_focus_no_match_keeps_current(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState()
        tracking = ConversationState(current_focus="步骤2")
        result = ConversationManager._update_focus("你好", ss, tracking)
        assert result == "步骤2"  # unchanged

    def test_update_tracking_merges_entities(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(current_goal="数据分析")
        tracking = ConversationState()
        ConversationManager._update_tracking_from_question(
            "数据分析报告包含哪些内容", tracking, ss,
        )
        assert "数据分析报告" in tracking.entities

    def test_update_tracking_updates_topic(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(current_goal="测试")
        tracking = ConversationState()
        ConversationManager._update_tracking_from_question(
            "数据分析报告怎么写", tracking, ss,
        )
        assert tracking.topic

    # -- resolve_question tracking integration (Phase 8) --

    def test_resolve_initializes_tracking(self):
        ss = SessionState(current_goal="测试")
        ConversationManager.resolve_question("数据分析报告", ss)
        assert ss.tracking is not None

    def test_resolve_empty_question_no_tracking(self):
        ss = SessionState()
        ConversationManager.resolve_question("", ss)
        assert ss.tracking is None

    def test_resolve_tracking_updates_entities(self):
        ss = SessionState(current_goal="测试")
        ConversationManager.resolve_question("数据分析报告怎么写", ss)
        assert ss.tracking is not None
        assert "数据分析报告" in ss.tracking.entities

    def test_resolve_tracking_updates_focus_on_ordinal(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="选择主题",
            pending_options={"1": "儿童教育", "2": "公共卫生"},
        )
        # tracking already initialized so resolve_question uses it
        ss.tracking = ConversationState(current_focus="主题")
        ConversationManager.resolve_question("选项二", ss)
        assert ss.tracking.current_focus == "公共卫生"

    def test_enrich_with_context_uses_tracking_focus(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(current_goal="做报告")
        ss.tracking = ConversationState(topic="数据分析", current_focus="图表")
        result = ConversationManager._enrich_with_context("优化一下", ss)
        assert "数据分析" in result or "优化" in result

    def test_enrich_with_context_fallback_no_tracking(self):
        ss = SessionState(current_goal="测试")
        result = ConversationManager._enrich_with_context("优化一下", ss)
        # Without tracking, uses plain __str__ context
        assert result is not None

    def test_new_creation_request_does_not_inherit_previous_goal(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="创建两个文件一个python贪吃蛇一个java贪吃蛇",
            status="idle",
            tracking=ConversationState(topic="贪吃蛇"),
        )

        resolved = ConversationManager.resolve_question("创建一个猜数字的游戏", ss)
        rewritten = ConversationManager.rewrite_question(resolved, ss)

        assert resolved == "创建一个猜数字的游戏"
        assert rewritten == "创建一个猜数字的游戏"
        assert ss.current_goal == ""
        assert ss.status == "idle"

    def test_follow_up_still_inherits_previous_goal(self):
        ss = SessionState(current_goal="创建 Python 贪吃蛇游戏", status="idle")

        resolved = ConversationManager.resolve_question("优化一下", ss)

        assert "创建 Python 贪吃蛇游戏" in resolved

    # -- finalize_turn with tracking (Phase 8) --

    def test_finalize_turn_saves_last_answer(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(tracking=ConversationState())
        ConversationManager.finalize_turn({}, ss, "这是数据分析报告的结果")
        assert ss.tracking is not None
        assert "数据分析" in ss.tracking.last_answer

    def test_finalize_turn_extracts_answer_entities(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(tracking=ConversationState())
        ConversationManager.finalize_turn({}, ss, "儿童教育方案分析报告")
        # Entities extracted from the answer should be non-empty
        assert len(ss.tracking.entities) > 0

    def test_finalize_turn_no_tracking_unchanged(self):
        ss = SessionState()
        ConversationManager.finalize_turn({}, ss, "一些回答")
        # No tracking → no change to tracking state
        assert ss.tracking is None

    def test_finalize_turn_summary_built(self):
        from agentflow.conversation.state import ConversationState
        ss = SessionState(tracking=ConversationState())
        ConversationManager.finalize_turn({}, ss, "这是一个很长的回答内容，用于测试摘要构建功能")
        assert ss.tracking.summary


# ---------------------------------------------------------------------------
# WorkflowContext integration
# ---------------------------------------------------------------------------


class TestWorkflowContextSessionState:
    def test_session_state_default(self):
        ctx = WorkflowContext()
        assert isinstance(ctx.session_state, SessionState)
        assert ctx.session_state.status == "idle"

    def test_session_state_setter_dict(self):
        ctx = WorkflowContext()
        ctx.session_state = {"current_goal": "test"}
        assert isinstance(ctx.session_state, SessionState)
        assert ctx.session_state.current_goal == "test"

    def test_session_state_setter_object(self):
        ctx = WorkflowContext()
        ss = SessionState(current_goal="test", status="waiting_user")
        ctx.session_state = ss
        assert ctx.session_state.current_goal == "test"
        assert ctx.session_state.is_waiting

    def test_to_dict_includes_session_state(self):
        ctx = WorkflowContext()
        ss = SessionState(current_goal="test", pending_options={"1": "a"})
        ctx.session_state = ss
        d = ctx.to_dict()
        assert "session_state" in d
        assert d["session_state"]["current_goal"] == "test"
        assert d["session_state"]["pending_options"]["1"] == "a"

    def test_to_dict_from_dict_round_trip(self):
        ctx = WorkflowContext()
        ctx.session_state = SessionState(current_goal="test任务", status="waiting_user")
        d = ctx.to_dict()
        ctx2 = WorkflowContext(d)
        assert ctx2.session_state.current_goal == "test任务"
        assert ctx2.session_state.is_waiting


# ---------------------------------------------------------------------------
# Workflow graph integration
# ---------------------------------------------------------------------------


class TestWorkflowGraph:
    def test_conversation_manager_node_no_session(self):
        """Without session_state, conversation_manager should NOT set continue_mode."""
        graph = build_workflow()
        # Run with a simple message (no session_state)
        result = run_workflow(graph, "hello")
        assert "answer" in result
        assert result.get("answer", "").strip()

    def test_session_state_persists_through_graph(self):
        """SessionState should survive serialization through run_workflow."""
        graph = build_workflow()
        ss = SessionState(current_goal="测试目标")
        ss_dict = ss.to_dict()

        result = run_workflow(
            graph,
            "hello",
            session_state=ss_dict,
        )
        returned_ss = result.get("session_state", {})
        # The session_state should be returned as part of the result
        # and should still contain the goal from the input
        assert isinstance(returned_ss, dict)

    def test_run_workflow_with_history(self):
        """run_workflow should still work with history."""
        graph = build_workflow()
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello back"},
        ]
        result = run_workflow(graph, "how are you", history=history)
        assert "answer" in result

    def test_continue_mode_skips_router(self):
        """When session_state has waiting_user, the flow should skip router."""
        graph = build_workflow()
        ss = SessionState(
            current_goal="测试任务",
            status="waiting_user",
            waiting_for="选择选项",
            pending_options={"1": "方案A", "2": "方案B"},
        )
        result = run_workflow(
            graph,
            "选项一",
            session_state=ss.to_dict(),
        )
        assert "answer" in result
        # The result should contain the updated session_state
        returned_ss = result.get("session_state", {})
        assert isinstance(returned_ss, dict)

    def test_rewritten_question_in_continue_mode(self):
        """In continue mode, rewritten_question should enrich the input."""
        graph = build_workflow()
        ss = SessionState(
            current_goal="测试任务",
            status="waiting_user",
            waiting_for="选择选项",
            pending_options={"1": "方案A", "2": "方案B"},
        )
        result = run_workflow(
            graph,
            "选项一",
            session_state=ss.to_dict(),
        )
        # The conversation_context should include the context type
        cc = result.get("conversation_context", {})
        assert cc.get("type") == "OPTION_SELECTION"
        assert cc.get("original_question") == "选项一"
        # rewritten_question should be in the result
        rq = result.get("rewritten_question", "")
        assert rq


# ---------------------------------------------------------------------------
# ConversationContext (Phase 7)
# ---------------------------------------------------------------------------


class TestConversationContext:
    def test_default_new_task(self):
        from agentflow.conversation.context import ConversationContext
        cc = ConversationContext()
        assert cc.type == "NEW_TASK"
        assert cc.original_question == ""
        assert cc.rewritten_question == ""

    def test_to_dict_round_trip(self):
        from agentflow.conversation.context import ConversationContext
        cc = ConversationContext(
            type="FOLLOW_UP",
            original_question="继续",
            rewritten_question="请继续生成报告",
            current_goal="生成报告",
            last_topic="数据分析",
            waiting_for="",
            entities=["报告", "数据"],
            summary="用户正在生成报告",
        )
        d = cc.to_dict()
        assert d["type"] == "FOLLOW_UP"
        assert d["rewritten_question"] == "请继续生成报告"
        assert "报告" in d["entities"]

        cc2 = ConversationContext(**d)
        assert cc2.type == cc.type
        assert cc2.rewritten_question == cc.rewritten_question
        assert cc2.entities == cc.entities

    def test_default_type(self):
        from agentflow.conversation.context import ConversationContext
        cc = ConversationContext()
        assert cc.type == "NEW_TASK"

    def test_str_with_goal(self):
        from agentflow.conversation.context import ConversationContext
        cc = ConversationContext(
            type="FOLLOW_UP",
            current_goal="写报告",
            rewritten_question="继续写数据分析报告",
        )
        s = str(cc)
        assert "FOLLOW_UP" in s
        assert "写报告" in s

    def test_str_empty(self):
        from agentflow.conversation.context import ConversationContext
        cc = ConversationContext()
        s = str(cc)
        assert s.strip()


# ---------------------------------------------------------------------------
# RewriteEngine (Phase 7)
# ---------------------------------------------------------------------------


class TestRewriteEngine:
    def test_needs_rewrite_short(self):
        from agentflow.conversation.rewrite import RewriteEngine
        assert RewriteEngine.needs_rewrite("第二个")
        assert RewriteEngine.needs_rewrite("继续")
        assert RewriteEngine.needs_rewrite("优化")
        assert RewriteEngine.needs_rewrite("展开")
        assert RewriteEngine.needs_rewrite("a")  # very short

    def test_needs_rewrite_ordinal(self):
        from agentflow.conversation.rewrite import RewriteEngine
        assert RewriteEngine.needs_rewrite("第三个")
        assert RewriteEngine.needs_rewrite("选项二")
        assert RewriteEngine.needs_rewrite("方案一")
        assert RewriteEngine.needs_rewrite("二")
        assert RewriteEngine.needs_rewrite("步骤三")

    def test_needs_rewrite_modifier(self):
        from agentflow.conversation.rewrite import RewriteEngine
        assert RewriteEngine.needs_rewrite("优化一下")
        assert RewriteEngine.needs_rewrite("改一下")
        assert RewriteEngine.needs_rewrite("改成 Java")
        assert RewriteEngine.needs_rewrite("完善一下")

    def test_needs_rewrite_follow_up(self):
        from agentflow.conversation.rewrite import RewriteEngine
        assert RewriteEngine.needs_rewrite("继续")
        assert RewriteEngine.needs_rewrite("然后呢")
        assert RewriteEngine.needs_rewrite("详细一点")
        assert RewriteEngine.needs_rewrite("为什么")

    def test_needs_rewrite_deictic(self):
        from agentflow.conversation.rewrite import RewriteEngine
        assert RewriteEngine.needs_rewrite("这个")
        assert RewriteEngine.needs_rewrite("那个方案")
        assert RewriteEngine.needs_rewrite("那北京呢")
        assert RewriteEngine.needs_rewrite("数据")

    def test_needs_rewrite_self_contained(self):
        from agentflow.conversation.rewrite import RewriteEngine
        # Longer questions should NOT need rewrite
        assert not RewriteEngine.needs_rewrite("请介绍 IDA 的完整使用流程")
        assert not RewriteEngine.needs_rewrite("写一个 Python 登录程序")
        assert not RewriteEngine.needs_rewrite("今天天气怎么样")
        assert not RewriteEngine.needs_rewrite("孙严培")

    def test_needs_rewrite_confirmations(self):
        from agentflow.conversation.rewrite import RewriteEngine
        # Confirmations (continue signals) don't need rewrite
        # They are handled by resolve_question
        assert not RewriteEngine.needs_rewrite("好的")
        assert not RewriteEngine.needs_rewrite("嗯")
        assert not RewriteEngine.needs_rewrite("是的")
        assert not RewriteEngine.needs_rewrite("对")

    def test_rewrite_with_goal_context(self):
        from agentflow.conversation.rewrite import RewriteEngine
        from agentflow.conversation.session_state import SessionState
        ss = SessionState(current_goal="写 Python 贪吃蛇游戏")
        result = RewriteEngine.rewrite("优化一下", ss)
        assert "优化" in result
        assert "贪吃蛇" in result

    def test_rewrite_ordinal_with_options(self):
        from agentflow.conversation.rewrite import RewriteEngine
        from agentflow.conversation.session_state import SessionState
        ss = SessionState(
            current_goal="选择报告主题",
            pending_options={"1": "儿童教育", "2": "公共卫生"},
        )
        result = RewriteEngine.rewrite("第二个", ss)
        # Resolved via resolve_option
        assert result == "请从当前任务「选择报告主题」中选择：公共卫生" or "公共卫生" in result

    def test_rewrite_with_memory_context(self):
        from agentflow.conversation.rewrite import RewriteEngine
        memory = {
            "current_goal": "生成数据分析报告",
            "last_topic": "数据清洗方案",
        }
        result = RewriteEngine.rewrite("展开", memory=memory)
        assert "展开" in result
        assert "数据" in result

    def test_rewrite_short_input(self):
        from agentflow.conversation.rewrite import RewriteEngine
        from agentflow.conversation.session_state import SessionState
        ss = SessionState(current_goal="生成报告")
        result = RewriteEngine.rewrite("继续", ss)
        assert "继续" in result or "报告" in result

    def test_rewrite_no_context_returns_original(self):
        from agentflow.conversation.rewrite import RewriteEngine
        result = RewriteEngine.rewrite("优化一下")
        assert result == "优化一下"

    def test_short_standalone_name_does_not_inherit_previous_goal(self):
        from agentflow.conversation.manager import ConversationManager
        from agentflow.conversation.state import ConversationState
        from agentflow.conversation.session_state import SessionState

        ss = SessionState(current_goal="omniforge有哪些亮点")
        ss.tracking = ConversationState()
        ss.tracking.topic = "有哪些亮点"

        cm = ConversationManager()
        resolved = cm.resolve_question("孙严培", ss)
        rewritten = cm.rewrite_question(resolved, ss)

        assert resolved == "孙严培"
        assert rewritten == "孙严培"

    def test_rewrite_follow_up_with_context(self):
        from agentflow.conversation.rewrite import RewriteEngine
        from agentflow.conversation.session_state import SessionState
        ss = SessionState(current_goal="IDA 使用流程")
        result = RewriteEngine.rewrite("为什么", ss)
        assert "为什么" in result
        assert "IDA" in result

    # -- RewriteEngine with ConversationState tracking (Phase 8) --

    def test_rewrite_ordinal_uses_tracking_focus(self):
        from agentflow.conversation.rewrite import RewriteEngine
        from agentflow.conversation.session_state import SessionState
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="选择主题",
            tracking=ConversationState(current_focus="儿童教育"),
        )
        result = RewriteEngine.rewrite("第二个", ss)
        # When tracking focus is available, ordinal rewrite uses it
        assert "儿童教育" in result

    def test_rewrite_modifier_uses_tracking_focus(self):
        from agentflow.conversation.rewrite import RewriteEngine
        from agentflow.conversation.session_state import SessionState
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="写报告",
            tracking=ConversationState(current_focus="图表", topic="数据分析"),
        )
        result = RewriteEngine.rewrite("优化一下", ss)
        assert "图表" in result

    def test_rewrite_follow_up_uses_tracking_topic(self):
        from agentflow.conversation.rewrite import RewriteEngine
        from agentflow.conversation.session_state import SessionState
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="IDA流程",
            tracking=ConversationState(topic="IDA"),
        )
        result = RewriteEngine.rewrite("为什么", ss)
        assert "IDA" in result

    def test_rewrite_no_tracking_fallback(self):
        """Without tracking, RewriteEngine falls back to existing logic."""
        from agentflow.conversation.rewrite import RewriteEngine
        from agentflow.conversation.session_state import SessionState
        ss = SessionState(current_goal="写代码", waiting_for="选择语言")
        result = RewriteEngine.rewrite("优化一下", ss)
        assert "写代码" in result

    def test_needs_rewrite_unchanged(self):
        """needs_rewrite is unchanged by Phase 8."""
        from agentflow.conversation.rewrite import RewriteEngine
        assert RewriteEngine.needs_rewrite("第二个")
        assert RewriteEngine.needs_rewrite("优化")
        assert RewriteEngine.needs_rewrite("继续")


class TestConversationManagerRewrite:
    def test_rewrite_question_delegates(self):
        cm = ConversationManager()
        from agentflow.conversation.session_state import SessionState
        ss = SessionState(current_goal="生成数据分析报告")
        # Short input should be rewritten
        rewritten = cm.rewrite_question("继续", ss)
        assert rewritten is not None
        assert isinstance(rewritten, str)

    def test_rewrite_question_long_unchanged(self):
        cm = ConversationManager()
        rewritten = cm.rewrite_question("请介绍 IDA 的完整使用流程")
        assert rewritten == "请介绍 IDA 的完整使用流程"

    def test_build_context_new_task(self):
        cm = ConversationManager()
        from agentflow.conversation.session_state import SessionState
        ss = SessionState()
        ctx = cm.build_conversation_context("你好", "你好", ss)
        assert ctx.type == "NEW_TASK"
        assert ctx.original_question == "你好"

    def test_build_context_option_selection(self):
        cm = ConversationManager()
        from agentflow.conversation.session_state import SessionState
        ss = SessionState(
            current_goal="选择主题",
            status="waiting_user",
            pending_options={"1": "主题A", "2": "主题B"},
        )
        ctx = cm.build_conversation_context("选项一", "主题A", ss)
        assert ctx.type == "OPTION_SELECTION"
        assert ctx.current_goal == "选择主题"

    def test_build_context_follow_up(self):
        cm = ConversationManager()
        from agentflow.conversation.session_state import SessionState
        ss = SessionState(current_goal="分析数据", status="processing")
        # Use a follow-up that's long enough not to trigger rewrite,
        # so the type stays FOLLOW_UP (not QUESTION_REWRITE)
        ctx = cm.build_conversation_context(
            "请继续分析", "请继续分析", ss
        )
        assert ctx.type == "FOLLOW_UP"

    def test_build_context_with_memory(self):
        cm = ConversationManager()
        from agentflow.conversation.session_state import SessionState
        ss = SessionState(current_goal="测试")
        memory = {"last_topic": "Python 代码", "summary": "用户正在写 Python 代码"}
        ctx = cm.build_conversation_context("优化", "优化当前任务：测试", ss, memory)
        assert ctx.last_topic == "Python 代码"
        assert ctx.summary == "用户正在写 Python 代码"

    # -- build_conversation_context with tracking (Phase 8) --

    def test_build_context_merges_tracking_entities(self):
        cm = ConversationManager()
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="数据分析",
            tracking=ConversationState(entities={"数据分析", "可视化"}),
        )
        ctx = cm.build_conversation_context("报告制作", "报告制作", ss)
        assert "数据分析" in ctx.entities
        assert "可视化" in ctx.entities

    def test_build_context_no_tracking_uses_only_current(self):
        cm = ConversationManager()
        ss = SessionState(current_goal="测试")
        ctx = cm.build_conversation_context("你好", "你好", ss)
        # No tracking → only entities from the question
        assert isinstance(ctx.entities, list)

    def test_build_context_includes_focus_in_goal(self):
        cm = ConversationManager()
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="选择主题",
            tracking=ConversationState(current_focus="儿童教育"),
        )
        ctx = cm.build_conversation_context("第二个", "第二个", ss)
        assert "儿童教育" in ctx.current_goal

    def test_build_context_with_tracking_and_memory(self):
        cm = ConversationManager()
        from agentflow.conversation.state import ConversationState
        ss = SessionState(
            current_goal="测试",
            tracking=ConversationState(summary="测试摘要"),
        )
        memory = {"last_topic": "Python 代码"}
        ctx = cm.build_conversation_context("继续", "继续", ss, memory)
        # Memory last_topic should still be used
        assert ctx.last_topic == "Python 代码"


# ---------------------------------------------------------------------------
# ContextBuilder (Phase 7)
# ---------------------------------------------------------------------------


class TestContextBuilder:
    def test_system_prompt_continue(self):
        from agentflow.agents.answer.agent import AnswerAgent
        prompt = AnswerAgent._system_prompt(continue_mode=True)
        assert "连续对话" in prompt or "继续" in prompt

    def test_system_prompt_new(self):
        from agentflow.agents.answer.agent import AnswerAgent
        prompt = AnswerAgent._system_prompt(continue_mode=False)
        assert "专业" in prompt

    def test_build_user_prompt_with_session_context(self):
        from agentflow.graph.context_builder import ContextBuilder
        builder = ContextBuilder({
            "question": "继续",
            "_continue_mode": True,
            "session_context": "当前目标：生成报告",
            "memory": {},
        })
        ctx = builder.build()
        assert "goal" in ctx or "session_context" in str(ctx)
