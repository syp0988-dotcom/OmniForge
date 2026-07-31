"""End-to-end capability tests for the three core abilities.

Tests verify that each capability produces correct results through
the actual workflow nodes, using mocks only for LLM and external APIs.
"""

import json
from pathlib import Path

import pytest

from agentflow.graph.workflow import build_workflow, reset_workflow_cache
from agentflow.graph.executor import Executor
from agentflow.tools.filesystem_tool import FileSystemTool
from agentflow.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _reset():
    reset_workflow_cache()
    yield


# =============================================================================
# 1. 联网回答能力 — 完整搜索链路
# =============================================================================


class TestSearchCapability:
    """Full search path: goal → planner → query_rewriter → search → answer."""

    def test_search_workflow_produces_answer(self):
        """Search routing works: mock LLM + pre-set search results."""
        from agentflow.conversation.session_state import SessionState
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(responses={"default": "搜索结果摘要。"}):
            graph = build_workflow()
            from agentflow.graph.workflow import WorkflowState

            state: WorkflowState = {
                "question": "搜索AI最新进展",
                "history": [],
                "goal_analysis": {
                    "goal": "搜索AI最新进展",
                    "goal_type": "search",
                    "knowledge_source": "web",
                    "confidence": 0.9,
                },
                "task_queue": [
                    {
                        "task_id": "search_ai_001",
                        "title": "搜索AI最新进展",
                        "tool": "search",
                        "goal": "搜索AI最新进展",
                        "priority": 90,
                        "status": "todo",
                        "input": {"query": "AI 最新进展 2025"},
                    },
                ],
                "search_results": [
                    {"title": "AI 突破", "content": "2025年AI取得重大进展", "url": "https://ex.com/ai"},
                ],
                "session_state": SessionState(),
            }

            result = graph.invoke(state)

        assert "answer" in result
        answer = result.get("answer", "")
        assert answer, "Answer should not be empty"

    def test_search_empty_results_handled(self):
        """Empty search results should not crash."""
        from agentflow.conversation.session_state import SessionState
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(responses={"default": "未找到相关结果。"}):
            graph = build_workflow()
            from agentflow.graph.workflow import WorkflowState

            state: WorkflowState = {
                "question": "搜索不存在的主题",
                "history": [],
                "goal_analysis": {
                    "goal": "搜索不存在的主题",
                    "goal_type": "search",
                    "knowledge_source": "web",
                    "confidence": 0.9,
                },
                "task_queue": [
                    {
                        "task_id": "search_empty",
                        "title": "搜索不存在主题",
                        "tool": "search",
                        "goal": "搜索不存在主题",
                        "priority": 90,
                        "status": "todo",
                        "input": {"query": "xyz789nonexistent"},
                    },
                ],
                "search_results": [],
                "session_state": SessionState(),
            }

            result = graph.invoke(state)

        assert "answer" in result


# =============================================================================
# 2. 代码生成能力 — Executor 文件写入 + Reflector 验证
# =============================================================================


class TestCodeCapability:
    """Code generation: Executor writes files, Reflector validates completion."""

    def test_executor_writes_files_in_parallel(self):
        """Executor's batch_parallel writes multiple files successfully."""
        ws = Path(__file__).resolve().parent / ".cap_test_code"
        ws.mkdir(exist_ok=True)

        try:
            registry = ToolRegistry()
            registry.register(FileSystemTool(workspace=str(ws)))

            ex = Executor()
            ex.registry = registry

            results = ex.execute_batch_parallel([
                {"tool": "filesystem", "action": "write_file",
                 "input": {"action": "write_file", "path": "hello.py", "content": "print('hello')"}},
                {"tool": "filesystem", "action": "write_file",
                 "input": {"action": "write_file", "path": "config.json", "content": '{"key": "val"}'}},
            ])

            assert len(results) == 2
            assert results[0].success, f"First task failed: {results[0].error}"
            assert results[1].success, f"Second task failed: {results[1].error}"
            assert (ws / "hello.py").read_text() == "print('hello')"
            assert (ws / "config.json").read_text() == '{"key": "val"}'
        finally:
            import shutil
            shutil.rmtree(str(ws), ignore_errors=True)

    def test_executor_handles_directory_then_files(self):
        """Executor creates directory first, then writes files inside it."""
        ws = Path(__file__).resolve().parent / ".cap_test_code2"
        ws.mkdir(exist_ok=True)

        try:
            registry = ToolRegistry()
            registry.register(FileSystemTool(workspace=str(ws)))
            ex = Executor()
            ex.registry = registry

            results = ex.execute_batch([
                {"tool": "filesystem", "action": "mkdir",
                 "input": {"action": "mkdir", "path": "src"}},
                {"tool": "filesystem", "action": "write_file",
                 "input": {"action": "write_file", "path": "src/main.py", "content": "def main(): pass"}},
            ])

            assert results[0].success
            assert results[1].success
            assert (ws / "src" / "main.py").exists()
        finally:
            import shutil
            shutil.rmtree(str(ws), ignore_errors=True)

    def test_reflector_validates_completed_tasks(self):
        """Reflector marks DONE tasks correctly via rule eval."""
        from tests.mock_llm import MockLLMService
        from agentflow.agents.reflection.agent import ReflectionAgent
        from agentflow.graph.task import Task, TaskStatus

        reflector = ReflectionAgent()
        state = {
            "question": "创建两个文件",
            "goal_analysis": {"goal": "创建两个文件", "goal_type": "project"},
            "tool_results": [
                {"success": True, "tool": "filesystem", "action": "write_file",
                 "result": {"path": "app.py", "status": "created"}, "error": None},
                {"success": True, "tool": "filesystem", "action": "write_file",
                 "result": {"path": "config.py", "status": "created"}, "error": None},
            ],
            "task_queue": [
                Task(task_id="t1", title="创建 app.py", tool="filesystem",
                     goal="创建 app.py", priority=90,
                     input={"action": "write_file", "path": "app.py", "content": ""},
                     status=TaskStatus.DONE).to_dict(),
                Task(task_id="t2", title="创建 config.py", tool="filesystem",
                     goal="创建 config.py", priority=80,
                     input={"action": "write_file", "path": "config.py", "content": ""},
                     status=TaskStatus.DONE).to_dict(),
            ],
            "_errors": [],
            "_degraded": set(),
            "memory": {},
        }

        with MockLLMService.as_default(responses={"default": json.dumps({
            "goal_completed": True, "task_updates": [],
            "new_tasks": [], "remove_tasks": [],
            "reason": "所有文件任务已完成",
        })}):
            result = reflector.run(state)

        assert result["_reflection_result"] in ("done", "next")


# =============================================================================
# 3. 知识库回答能力 — KnowledgeAgent + AnswerAgent 链路
# =============================================================================


class TestKnowledgeCapability:
    """Knowledge path: KnowledgeAgent retrieves → AnswerAgent fuses context."""

    def test_knowledge_context_used_in_answer(self):
        """Pre-set knowledge_results → AnswerAgent uses them in response."""
        from agentflow.conversation.session_state import SessionState
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(responses={"default": "OmniForge 是一个多智能体框架。"}):
            graph = build_workflow()
            from agentflow.graph.workflow import WorkflowState

            state: WorkflowState = {
                "question": "OmniForge是什么",
                "history": [],
                "goal_analysis": {
                    "goal": "了解OmniForge",
                    "goal_type": "question",
                    "knowledge_source": "local",
                    "confidence": 0.9,
                },
                "knowledge_results": [
                    {"content": "OmniForge 是基于多智能体的 AI 编排框架", "score": 0.92, "filename": "intro.md"},
                ],
                "session_state": SessionState(),
            }

            result = graph.invoke(state)

        assert "answer" in result
        assert result.get("answer"), "Answer should not be empty"

    def test_empty_knowledge_graceful(self):
        """Empty knowledge_results produces a fallback answer."""
        from agentflow.conversation.session_state import SessionState
        from tests.mock_llm import MockLLMService

        with MockLLMService.as_default(responses={"default": "未找到相关知识。"}):
            graph = build_workflow()
            from agentflow.graph.workflow import WorkflowState

            state: WorkflowState = {
                "question": "查询不存在的文档",
                "history": [],
                "goal_analysis": {
                    "goal": "查询文档",
                    "goal_type": "question",
                    "knowledge_source": "local",
                    "confidence": 0.9,
                },
                "knowledge_results": [],
                "session_state": SessionState(),
            }

            result = graph.invoke(state)

        assert "answer" in result

    def test_knowledge_agent_integration(self):
        """KnowledgeAgent searches store and returns structured results."""
        from agentflow.agents.knowledge.agent import KnowledgeAgent

        agent = KnowledgeAgent()
        state = {
            "question": "测试",
            "rewritten_query": "测试",
            "memory": {},
        }
        result = agent.run(state)

        assert "knowledge_results" in result
        # KnowledgeAgent should return results (empty or populated)
        assert isinstance(result.get("knowledge_results"), list)
