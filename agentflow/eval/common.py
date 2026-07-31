"""共享基础类：BaseEvalDataset、MockLLMService、MockWriteTool、工具函数。"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agentflow.services.llm_service import LLMResponse, ToolCall
from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult


# ============================================================================
# BaseEvalDataset — 通用 JSONL 评测数据集基类
# ============================================================================

class BaseEvalDataset(ABC):
    """通用 JSONL 评测数据集基类。子类只需实现 ``_validate_sample`` 和 ``stats``。"""

    def __init__(self, samples: list[dict[str, Any]] | None = None) -> None:
        self.samples: list[dict[str, Any]] = samples or []

    # -- I/O ----------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> BaseEvalDataset:
        """从 JSONL 文件加载数据集。"""
        path = Path(path)
        samples: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Line {line_num}: invalid JSON: {exc}") from exc
                if cls._validate_sample(sample, line_num):
                    samples.append(sample)
        return cls(samples)

    def save(self, path: str | Path) -> None:
        """保存数据集到 JSONL 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # -- Mutation -----------------------------------------------------------

    def add_sample(self, sample: dict[str, Any]) -> str:
        """追加一条样本，自动补充 id。返回样本 id。"""
        if "id" not in sample:
            sample["id"] = f"ev_{uuid.uuid4().hex[:8]}"
        self.samples.append(sample)
        return sample["id"]

    def extend(self, other: BaseEvalDataset) -> None:
        """合并另一个数据集。"""
        self.samples.extend(other.samples)

    # -- Inspection ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    # -- Subclass contract --------------------------------------------------

    @staticmethod
    @abstractmethod
    def _validate_sample(sample: dict[str, Any], line_num: int) -> bool:
        """验证单条样本的必填字段。子类必须实现。"""
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """返回数据集的统计信息。子类必须实现。"""
        ...


# ============================================================================
# MockLLMService — 单轮 Mock LLM
# ============================================================================

class MockLLMService:
    """Mock LLM 服务，返回预置响应，不发起真实 API 调用。

    Parameters
    ----------
    responses : dict
        文本响应字典，key 为 prompt 指纹或 "default"。
    tool_call_responses : list[dict]
        工具调用响应列表，按调用顺序返回。
    degraded : bool
        是否模拟 LLM 不可用（返回 degraded=True）。
    """

    is_mock = True

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        tool_call_responses: list[dict] | None = None,
        degraded: bool = False,
    ) -> None:
        self._responses = responses or {}
        self._tool_call_responses = tool_call_responses or []
        self._tc_index = 0
        self._degraded = bool(degraded)

    @property
    def client(self):
        """返回 truthy dummy 值，使生产代码的 ``if not self.client`` 检查通过。"""
        return True

    def complete(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        session_state: object | None = None,
    ) -> str:
        """返回预置文本响应或 fallback 字符串。"""
        # 按 prompt 中关键字匹配
        if prompt:
            for key, value in self._responses.items():
                if key in prompt:
                    return value
        if messages:
            last_content = messages[-1].get("content", "") if messages else ""
            for key, value in self._responses.items():
                if key in last_content:
                    return value
        # fallback
        default = self._responses.get("default", "")
        if default:
            return default
        if prompt:
            return f"[mock] {prompt[:160]}"
        if messages:
            last = messages[-1].get("content", "") if messages else ""
            return f"[mock] {last[:160]}"
        return "[mock]"

    def complete_stream(self, *args, **kwargs):
        """流式接口兼容（不用于评测）。"""
        yield self.complete(*args, **kwargs)

    def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> LLMResponse:
        """返回预置工具调用响应或降级响应。"""
        if self._degraded:
            return LLMResponse(content="[LLM_UNAVAILABLE]", degraded=True)

        if self._tc_index < len(self._tool_call_responses):
            resp = self._tool_call_responses[self._tc_index]
            self._tc_index += 1
            return self._dict_to_llm_response(resp)

        return LLMResponse(content="[mock] no more tool call responses")

    def _dict_to_llm_response(self, resp: dict) -> LLMResponse:
        """将 dict 转为 LLMResponse。"""
        resp_type = resp.get("type", "json")

        if resp_type == "degraded":
            return LLMResponse(content=str(resp.get("content", "")), degraded=True)

        if resp_type == "tool_calls":
            tool_calls = []
            for tc in resp.get("tool_calls", []):
                args = tc.get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                tool_calls.append(ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    name=tc.get("name", ""),
                    arguments=args,
                ))
            return LLMResponse(
                content=resp.get("content", ""),
                tool_calls=tool_calls,
            )

        # json 类型：将 content 序列化
        content = resp.get("content", "")
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        return LLMResponse(content=content)

    def reset(self) -> None:
        """重置工具调用计数器。"""
        self._tc_index = 0


# ============================================================================
# TurnAwareMockLLMService — 多轮 Mock LLM
# ============================================================================

class TurnAwareMockLLMService(MockLLMService):
    """按 turn 索引返回不同响应的 Mock LLM，用于多轮任务完成评测。

    Parameters
    ----------
    planner_responses : list[dict]
        每轮 Planner 调用对应的响应。
    reflection_responses : list[dict]
        每轮 Reflector 调用对应的响应。
    goal_analyzer_response : dict | None
        GoalAnalyzer 的响应（通常不需要，因为用预置 goal_analysis 跳过）。
    """

    def __init__(
        self,
        planner_responses: list[dict] | None = None,
        reflection_responses: list[dict] | None = None,
        goal_analyzer_response: dict | None = None,
    ) -> None:
        super().__init__()
        self._planner_responses = planner_responses or []
        self._reflection_responses = reflection_responses or []
        self._goal_analyzer_response = goal_analyzer_response
        self._planner_calls = 0
        self._reflection_calls = 0
        self._last_call_was_planner = False

    def complete(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        session_state: object | None = None,
    ) -> str:
        """按调用来源返回对应的 mock 响应。"""
        # 判断调用来源
        source = self._detect_source(prompt, messages)

        if source == "planner":
            resp = self._get_planner_response()
        elif source == "reflector":
            resp = self._get_reflection_response()
        elif source == "goal_analyzer" and self._goal_analyzer_response:
            return json.dumps(self._goal_analyzer_response, ensure_ascii=False)
        else:
            return super().complete(prompt=prompt, messages=messages)

        if resp is None:
            return "[mock] no response configured"

        if isinstance(resp, dict):
            content = resp.get("content", "")
            if isinstance(content, dict):
                return json.dumps(content, ensure_ascii=False)
            return str(content)
        return str(resp)

    def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> LLMResponse:
        """Planner 的函数调用模式返回预置 tool_calls。"""
        resp = self._get_planner_response()
        if resp is None:
            return LLMResponse(content="[mock] no planner response", degraded=True)
        return self._dict_to_llm_response(resp)

    def _detect_source(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        """通过 prompt/messages 内容判断是 Planner 还是 Reflector 调用。"""
        content = ""
        if prompt:
            content = prompt
        elif messages:
            # 检查 system message
            for m in messages:
                if m.get("role") == "system":
                    content = m.get("content", "")
                    break
            if not content:
                content = messages[-1].get("content", "") if messages else ""

        # Reflector 特征词
        reflector_keywords = ["evaluate", "reflection", "task queue reflector",
                              "evaluate the task", "reflect on", "评估", "反思"]
        for kw in reflector_keywords:
            if kw.lower() in content.lower():
                return "reflector"

        # Planner 特征词
        planner_keywords = ["plan", "task planning", "generate tasks",
                            "规划", "任务规划", "goal_completed"]
        for kw in planner_keywords:
            if kw.lower() in content.lower():
                return "planner"

        # GoalAnalyzer 特征词
        ga_keywords = ["goal analyzer", "analyze the goal", "目标分析", "意图分析"]
        for kw in ga_keywords:
            if kw.lower() in content.lower():
                return "goal_analyzer"

        return "unknown"

    def _get_planner_response(self) -> dict | None:
        """获取当前轮次的 Planner mock 响应。"""
        idx = min(self._planner_calls, len(self._planner_responses) - 1)
        if idx < 0 or not self._planner_responses:
            return None
        resp = self._planner_responses[idx]
        self._planner_calls += 1
        return resp

    def _get_reflection_response(self) -> dict | None:
        """获取当前轮次的 Reflector mock 响应。"""
        idx = min(self._reflection_calls, len(self._reflection_responses) - 1)
        if idx < 0 or not self._reflection_responses:
            return None
        resp = self._reflection_responses[idx]
        self._reflection_calls += 1
        return resp

    def reset(self) -> None:
        """重置所有计数器。"""
        super().reset()
        self._planner_calls = 0
        self._reflection_calls = 0


# ============================================================================
# MockWriteTool — 永远返回成功的 filesystem 模拟工具
# ============================================================================

class MockWriteTool(BaseTool):
    """Mock 文件系统工具，所有操作永远返回成功。

    用于 Planner 和 Completion 评测，不产生真实文件 I/O。
    """

    name = "filesystem"
    description = "Mock file system tool for eval testing"

    def actions(self) -> dict[str, dict]:
        return {
            "mkdir": {
                "description": "创建目录",
                "parameters": {"path": {"type": "string", "description": "目录路径"}},
                "required": ["path"],
            },
            "create_file": {
                "description": "创建新文件",
                "parameters": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["path"],
            },
            "write_file": {
                "description": "写入文件",
                "parameters": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["path", "content"],
            },
            "append_file": {
                "description": "追加文件内容",
                "parameters": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "追加内容"},
                },
                "required": ["path", "content"],
            },
            "edit_file": {
                "description": "编辑文件",
                "parameters": {
                    "path": {"type": "string", "description": "文件路径"},
                    "old_string": {"type": "string", "description": "原文本"},
                    "new_string": {"type": "string", "description": "新文本"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            "read_file": {
                "description": "读取文件",
                "parameters": {"path": {"type": "string", "description": "文件路径"}},
                "required": ["path"],
            },
            "list_directory": {
                "description": "列出目录内容",
                "parameters": {"path": {"type": "string", "description": "目录路径"}},
                "required": ["path"],
            },
            "exists": {
                "description": "检查文件是否存在",
                "parameters": {"path": {"type": "string", "description": "文件路径"}},
                "required": ["path"],
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.pop("action", "write_file")
        path = kwargs.get("path", "unknown")
        return ToolResult.ok(
            tool=self.name,
            action=action,
            result={"path": str(path), "mock": True},
            message=f"[mock] {action} {path}",
        )


# ============================================================================
# ResultsWriter — 统一结果输出
# ============================================================================

def save_results(
    path: str | Path,
    summary: dict[str, Any],
    per_sample: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> None:
    """保存标准格式的评测结果 JSON 文件。

    输出格式::

        {"summary": {...}, "per_sample": [...], "config": {...}}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": summary,
        "per_sample": per_sample,
        "config": config or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
