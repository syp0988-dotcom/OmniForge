# OmniForge (AgentFlow) 修改建议报告

> 生成时间: 2026-07-06 | 基于代码分析 v0.1.0

---

## 一、Agent 统一接口契约

### 问题

所有 Agent 都通过约定实现了 `run(state) -> dict` 方法，但没有抽象基类或 Protocol 约束。类型系统无法校验新增 Agent 是否遗漏方法或参数类型不匹配。

### 推荐方案：Protocol（鸭子类型）而非 ABC

**选择 Protocol 的原因**：

1. 各 Agent 的 `__init__` 签名不同（依赖注入方式不同），ABC 强制统一构造签名会破坏现有设计
2. 当前各 Agent 没有需要共享的默认方法实现
3. Protocol 不影响继承树，迁移成本为零

### 具体修改

#### 1.1 新增 `agentflow/agents/base.py`

```python
"""Agent base contract — Protocol + error handling decorator."""

from __future__ import annotations

import functools
from typing import Any, Protocol, runtime_checkable

from agentflow.utils.logging import build_logger

logger = build_logger("agent")


@runtime_checkable
class AgentProtocol(Protocol):
    """Minimal contract every agent must fulfil.

    All agents accept a shared ``state`` dict and return an updated dict.
    The ``state`` dict is the LangGraph ``WorkflowState`` — agents communicate
    exclusively through it (no inter-agent calls).
    """

    def run(self, state: dict) -> dict:
        """Process the workflow state and return the updated state."""
        ...


def safe_run(agent_name: str | None = None):
    """Wrap an agent's run() so unexpected exceptions don't crash the workflow.

    Usage::

        class MyAgent:
            @safe_run("my_agent")
            def run(self, state: dict) -> dict:
                ...
    """
    def decorator(func) -> Any:
        @functools.wraps(func)
        def wrapper(self, state: dict, *args, **kwargs) -> dict:
            name = agent_name or type(self).__name__
            try:
                return func(self, state, *args, **kwargs)
            except Exception as exc:
                logger.exception("[%s] run() failed: %s", name, exc)
                state.setdefault("errors", []).append({
                    "agent": name,
                    "error": str(exc),
                })
                return state   # 返回原 state，工作流继续
        return wrapper
    return decorator
```

#### 1.2 为每个 Agent 添加 `@safe_run` 装饰器

| 文件 | 修改 |
|------|------|
| `agentflow/agents/router/agent.py:102` | `@safe_run("router")` 装饰 `run()` |
| `agentflow/agents/planner/agent.py:78` | `@safe_run("planner")` 装饰 `run()` |
| `agentflow/agents/knowledge/agent.py:21` | `@safe_run("knowledge")` 装饰 `run()` |
| `agentflow/agents/search/agent.py:20` | `@safe_run("search")` 装饰 `run()` |
| `agentflow/agents/python/agent.py:27` | `@safe_run("python")` 装饰 `run()` |
| `agentflow/agents/answer/agent.py:150` | `@safe_run("answer")` 装饰 `run()` |
| `agentflow/agents/memory/agent.py:30` | `@safe_run("memory")` 装饰 `run()` |
| `agentflow/agents/report/agent.py:15` | `@safe_run("report")` 装饰 `run()`（可选） |

#### 1.3 新增测试 `tests/test_agent_contract.py`

```python
"""Verify all registered agents comply with AgentProtocol."""

import importlib

import pytest

from agentflow.agents.base import AgentProtocol
from agentflow.agents.registry import _registry


def _find_agent_class(module):
    """Find the class with a ``run`` method in the given module."""
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and hasattr(obj, "run"):
            return obj
    return None


def test_all_registered_agents_implement_protocol():
    """Every registered agent module must contain a class implementing AgentProtocol."""
    for key, info in _registry.items():
        mod = importlib.import_module(info.module_path)
        cls = _find_agent_class(mod)
        assert cls is not None, f"{key}: no class with run() found in {info.module_path}"
        assert isinstance(cls, AgentProtocol), (
            f"{key}: {cls.__name__} does not implement AgentProtocol"
        )


def test_all_agent_run_return_dict():
    """Smoke test: each agent's run() returns a dict when given a minimal state."""
    for key, info in _registry.items():
        if info.status != "active":
            continue
        mod = importlib.import_module(info.module_path)
        cls = _find_agent_class(mod)
        instance = cls()
        result = instance.run({"question": "test"})
        assert isinstance(result, dict), (
            f"{key}: run() returned {type(result)}, expected dict"
        )
```

#### 1.4 可选：registry 注册时校验

```python
# agentflow/agents/registry.py
def register(info: AgentInfo) -> None:
    _registry[info.key] = info
    # 可在此处添加 import 和 isinstance 校验（通过配置开关）
```

### 迁移步骤

```
Phase 1: 创建 base.py（新增文件，零改动现有代码）
Phase 2: 为每个 Agent 逐一添加 @safe_run（可分批 PR）
Phase 3: 添加 tests/test_agent_contract.py
Phase 4: （可选）registry 注册时校验
```

---

## 二、LLMService 添加重试机制

### 问题

`llm_service.py:96-111` 的 `complete()` 方法在 LLM 调用异常时仅记录日志并直接返回 fallback，没有任何重试。一次网络抖动即导致 Planner 走规则回退路径。

### 修改方案

```python
# agentflow/services/llm_service.py
import time
from functools import wraps

def retry_on_failure(max_retries: int = 2, base_delay: float = 1.0):
    """指数退避重试装饰器."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                            attempt + 1, max_retries + 1, exc, delay,
                        )
                        time.sleep(delay)
            logger.error("LLM call failed after %d retries: %s", max_retries, last_exc)
            raise last_exc
        return wrapper
    return decorator
```

### 修改范围

| 文件 | 改动 |
|------|------|
| `agentflow/services/llm_service.py` | 添加 `retry_on_failure` 装饰器，应用到 `complete()` 方法 |
| `agentflow/services/llm_service.py` | 添加超时配置（`timeout` 参数透传到 OpenAI 客户端） |

### 预计工作量

**0.5 天**

---

## 三、session_state 类型统一

### 问题

`session_state` 在 WorkflowState 中定义为 `dict[str, Any]`，但实际代码中有时是 `dict`、有时是 `SessionState` 对象，多处需要 `isinstance`/`hasattr` 做运行时判断。

### 修改方案

**核心理念：工作流内部统一使用 `SessionState` 对象，仅在序列化/反序列化边界转为 `dict`。**

#### 3.1 修改 WorkflowState 类型定义

```python
# agentflow/graph/workflow.py
class WorkflowState(TypedDict, total=False):
    # ...
    session_state: Any   # 内部流转时始终是 SessionState 对象
    # ...
```

不再声明为 `dict[str, Any]`，避免类型检查的误导。

#### 3.2 所有 Agent 内部直接使用 SessionState

```python
# MemoryAgent 修改前
ss_raw = state.get("session_state")
if isinstance(ss_raw, dict):
    ss = SessionState.from_dict(ss_raw)
else:
    ss = SessionState()

# MemoryAgent 修改后
ss = state.get("session_state")
if not isinstance(ss, SessionState):
    ss = SessionState()
```

#### 3.3 ConversationManager 节点负责转换

```python
# workflow.py _make_conversation_manager_node
raw = state.get("session_state")
session_state = SessionState.from_dict(raw) if isinstance(raw, dict) else SessionState()

# ... 处理后 ...
result["session_state"] = session_state  # 直接存对象，不 to_dict()
```

#### 3.4 API 边界处统一序列化

```python
# api/routes.py — 返回给前端时
response["session_state"] = (
    state["session_state"].to_dict()
    if isinstance(state.get("session_state"), SessionState)
    else state.get("session_state", {})
)
```

### 修改范围

| 文件 | 改动 |
|------|------|
| `agentflow/graph/workflow.py` | WorkflowState 类型注解；_make_conversation_manager_node 存对象而非 dict |
| `agentflow/agents/memory/agent.py` | 移除 isinstance(dict) 判断分支，直接作为 SessionState 使用 |
| `agentflow/agents/answer/agent.py` | ContextBuilder 中简化 session_context 获取 |
| `agentflow/api/routes.py` | 返回前统一 to_dict() |
| `agentflow/graph/context.py` | `to_dict()` 中处理 SessionState 序列化 |

### 预计工作量

**1 天**

---

## 四、知识库搜索全表扫描优化

### 问题

`KnowledgeStore.get_all_embeddings_with_chunk()` 每次搜索时从 SQLite 加载所有 embedding 向量到内存，逐一计算余弦相似度。文档量增大时性能和内存消耗线性增长。

### 修改方案

#### 短期：TF-IDF 倒排索引初步筛选（P1）

```python
# agentflow/knowledge/store.py
class KnowledgeStore:
    def __init__(self):
        # 新增：TF-IDF 倒排索引缓存
        self._inverted_index: dict[str, list[int]] = {}  # term -> [chunk_ids]
        self._idf_cache: dict[str, float] = {}            # term -> idf
        self._index_loaded = False

    def _build_inverted_index(self) -> None:
        """从所有 chunk 构建 TF-IDF 倒排索引。"""
        chunks = self._get_all_chunks()
        doc_count = len(chunks)
        for chunk in chunks:
            terms = set(self._tokenize(chunk["content"]))
            for term in terms:
                self._inverted_index.setdefault(term, []).append(chunk["id"])
        # 计算 IDF
        for term, posting in self._inverted_index.items():
            self._idf_cache[term] = math.log(doc_count / (1 + len(posting)))

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05) -> list[dict]:
        # 先通过倒排索引筛选候选 chunk
        query_terms = self._tokenize(query)
        candidate_ids = set()
        for term in query_terms:
            if term in self._inverted_index:
                candidate_ids.update(self._inverted_index[term])

        if not candidate_ids:
            return []  # 无匹配，提前返回

        # 仅对候选 chunk 计算 embedding 相似度
        candidates = self._get_embeddings_by_ids(candidate_ids)
        return self._rank_candidates(query, candidates, top_k, min_score)
```

#### 中长期：FAISS 近似最近邻搜索（P2）

```python
# agentflow/knowledge/faiss_store.py
import faiss
import numpy as np

class FaissStore:
    """向量索引，替代全表扫描的暴力搜索。"""

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)  # 内积（余弦相似度）
        self.id_map: list[int] = []                 # FAISS position -> chunk_id

    def add(self, embedding: list[float], chunk_id: int) -> None:
        self.index.add(np.array([embedding], dtype=np.float32))
        self.id_map.append(chunk_id)

    def search(self, query_embedding: list[float], top_k: int) -> list[int]:
        scores, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), top_k
        )
        return [self.id_map[i] for i in indices[0]]
```

### 预计工作量

- TF-IDF 倒排索引：**1 天**
- FAISS 集成：**2 天**

---

## 五、流式传输（SSE/WebSocket）

### 问题

POST `/chat` 是同步请求/响应模式，用户需等待整个工作流完成才能看到结果。

### 修改方案

#### 5.1 EventBus 增强，支持事件流

```python
# agentflow/graph/event.py (新增 StreamingEvent 支持)
class EventBus:
    def __init__(self):
        self._handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self._stream_handlers: list[Callable] = []  # 流式输出处理器

    def on_stream(self, handler: Callable) -> None:
        """注册流式事件处理器（SSE 推送用）。"""
        self._stream_handlers.append(handler)

    def emit_stream(self, event_type: str, data: dict) -> None:
        """向所有流式处理器广播事件。"""
        for handler in self._stream_handlers:
            handler({"type": event_type, **data, "timestamp": time.time()})
```

#### 5.2 新增 WebSocket 端点

```python
# agentflow/api/routes.py
@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        # 创建带有流式回调的 EventBus
        event_bus = EventBus()
        async def stream_callback(event: dict):
            await websocket.send_json(event)
        event_bus.on_stream(stream_callback)

        # 执行工作流（EventBus 在内部节点触发时 emit 事件）
        result = run_workflow(graph, data["message"], ..., event_bus=event_bus)
        await websocket.send_json({"type": "complete", "data": result})
```

#### 5.3 工作流节点中 emit 事件

```python
# 各 Agent 的 run() 方法中
state.get("_event_bus", EventBus()).emit_stream("node.started", {
    "node": "planner", "question": question
})
```

### 前端适配

```typescript
// frontend/src/composables/useChatState.ts
const ws = new WebSocket(`ws://${host}/ws/chat`);
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    switch (msg.type) {
        case "node.started":
            thinkingSteps.value.push({ node: msg.node, status: "running" });
            break;
        case "node.completed":
            // 更新对应节点状态为 completed
            break;
        case "complete":
            // 最终答案
            break;
    }
};
```

### 预计工作量

**3 天**

---

## 六、搜索服务多 Provider 支持

### 问题

仅 DuckDuckGo HTML 爬取，单一依赖风险大。

### 修改方案

架构已正确（`BaseSearchProvider` ABC 存在），只需新增实现：

```python
# agentflow/services/search_providers/brave.py
class BraveSearchProvider(BaseSearchProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            params={"q": query, "count": max_results},
        )
        response.raise_for_status()
        return self._normalize(response.json())

    def _normalize(self, raw: dict) -> list[dict]:
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
            for r in raw.get("web", {}).get("results", [])
        ]


# agentflow/services/search_providers/tavily.py
class TavilySearchProvider(BaseSearchProvider):
    # 类似实现
    ...
```

配置切换：

```python
# agentflow/config/settings.py
class Settings(BaseSettings):
    search_provider: str = "duckduckgo"  # duckduckgo | brave | tavily | serper
    brave_api_key: str = ""
    tavily_api_key: str = ""

# agentflow/services/search_provider.py
def create_provider(settings: Settings) -> BaseSearchProvider:
    providers = {
        "duckduckgo": DuckDuckGoProvider,
        "brave": BraveSearchProvider,
        "tavily": TavilySearchProvider,
    }
    cls = providers.get(settings.search_provider, DuckDuckGoProvider)
    if settings.search_provider == "brave":
        return cls(api_key=settings.brave_api_key)
    return cls()
```

### 预计工作量

**2 天**

---

## 七、知识库 RAG 增强

### 问题

仅 TF-IDF 词法匹配，无语义理解，文档解析基础。

### 修改方案

#### 7.1 添加语义嵌入支持

```python
# agentflow/knowledge/semantic_embedder.py
from sentence_transformers import SentenceTransformer

class SemanticEmbedder:
    """基于 sentence-transformers 的语义嵌入。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
```

#### 7.2 倒排索引 + 语义检索双通道

```python
# 策略：TF-IDF 快速召回 → 语义重排序
class HybridRetriever:
    def __init__(self):
        self.tfidf = TfidfEmbedder()
        self.semantic = SemanticEmbedder()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        # Stage 1: TF-IDF 快速召回 top 20
        candidates = self.tfidf.search(query, top_k=20, min_score=0.02)

        # Stage 2: 语义重排序 top 5
        if candidates:
            query_emb = self.semantic.embed(query)
            for c in candidates:
                chunk_emb = self._load_embedding(c["chunk_id"])
                c["semantic_score"] = cosine_similarity(query_emb, chunk_emb)
            candidates.sort(key=lambda x: x["semantic_score"], reverse=True)

        return candidates[:top_k]
```

#### 7.3 文档解析增强

```python
# agentflow/knowledge/parser.py 增强
class DocumentParser:
    def parse_pdf(self, path: str) -> str:
        # 当前：仅文本提取
        # 增强目标：
        #   - 表格提取（camelot / tabula-py）
        #   - 图片 OCR（paddleocr / easyocr）
        #   - 多列布局检测与重排
        ...
```

### 预计工作量

- 语义嵌入集成：**3 天**
- re-ranking：**1 天**
- 文档解析增强：**2 天**

---

## 八、Agent 层统一错误处理

### 问题

大部分 Agent 的 `run()` 方法没有 try/except 包裹，任何一个未捕获异常都会导致 LangGraph 工作流崩溃。

### 修改方案

#### 8.1 `@safe_run` 装饰器（同第一章，详见 1.1）

每个 Agent 的 `run()` 方法添加 `@safe_run`，出现异常时：
1. 记录异常堆栈
2. 将错误信息写入 `state["errors"]` 列表
3. 返回原始 state，工作流继续执行

#### 8.2 工作流层面添加全局错误捕获

```python
# agentflow/graph/workflow.py
def _error_handler(state: WorkflowState) -> dict:
    """全局错误处理节点——收集所有 Agent 的错误信息并生成用户反馈。"""
    errors = state.get("errors", [])
    if errors:
        error_summary = "\n".join(
            f"- [{e['agent']}] {e['error'][:200]}"
            for e in errors
        )
        logger.error("Workflow completed with errors:\n%s", error_summary)
        state["answer"] = (
            state.get("answer", "")
            + f"\n\n> ⚠️ 处理过程中遇到以下问题：\n{error_summary}"
        )
    return state
```

### 预计工作量

**1 天**

---

## 九、前端错误反馈修复

### 问题

`frontend/src/composables/useChatState.ts` 中大量 catch 块使用空语句或仅注释 "silently fail"。

### 修改方案

```typescript
// 修改前
try {
    await api.deleteSession(sessionId);
} catch {
    // silently fail
}

// 修改后
import { useToast } from '@/composables/useToast';

try {
    await api.deleteSession(sessionId);
} catch (error) {
    console.error("[useChatState] deleteSession failed:", error);
    toast.show("删除会话失败，请稍后重试", "error");
}
```

### 错误分类

| 类型 | 用户反馈 | 调试输出 |
|------|---------|---------|
| 消息发送失败 | Toast 提示"发送失败，请重试" | 打印完整错误栈 |
| 会话列表刷新失败 | Toast 提示"刷新失败" | 打印错误栈 |
| 知识库文档删除失败 | Toast 提示"删除失败" | 打印错误栈 |
| 模型配置更新失败 | Toast 提示"保存失败" | 打印错误栈 |
| 网络错误（临时） | Toast 提示"网络异常" | 静默 |

### 预计工作量

**1 天**

---

## 十、Python 子进程环境修复

### 问题

`python_tool.py:73` 使用 `env={}` 执行 Python 子进程，完全清空环境变量，破坏需要系统 PATH 或 SSL 证书的库。

### 修改方案

```python
# agentflow/tools/python_tool.py
import os

# 允许保留的安全环境变量
_SAFE_ENV_VARS = {
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", ""),
    "USERPROFILE": os.environ.get("USERPROFILE", ""),  # Windows
    "TMP": os.environ.get("TMP", ""),
    "TEMP": os.environ.get("TEMP", ""),  # Windows
    # SSL 证书路径（某些系统需要）
    "SSL_CERT_FILE": os.environ.get("SSL_CERT_FILE", ""),
    "REQUESTS_CA_BUNDLE": os.environ.get("REQUESTS_CA_BUNDLE", ""),
}

# 明确排除的敏感变量
_SENSITIVE_VARS = {"API_KEY", "SECRET", "TOKEN", "PASSWORD", "OPENAI_API_KEY"}

def _build_sandbox_env() -> dict[str, str]:
    """构建沙箱环境：保留安全变量，排除敏感变量。"""
    env = dict(_SAFE_ENV_VARS)
    for key, value in os.environ.items():
        if not any(s in key.upper() for s in _SENSITIVE_VARS):
            env[key] = value
    return env
```

### 预计工作量

**0.5 天**

---

## 十一、测试覆盖补充计划

### 优先级顺序

| 优先级 | 模块 | 关键测试点 | 预计用例数 |
|--------|------|-----------|-----------|
| P0 | PlannerAgent | JSON 解析（含异常格式）、规则回退路径、capability 解析 | 8-10 |
| P0 | LLMService | fallback 行为、重试逻辑、模型切换 | 5-8 |
| P1 | KnowledgeStore | 文档增删改查、embedding 序列化/反序列化、搜索排序 | 8-10 |
| P1 | SearchTool | DuckDuckGo 响应解析、空结果处理、异常处理 | 5-8 |
| P1 | PythonTool | 代码执行、超时处理、输出截断、语法校验 | 5-8 |
| P2 | 各 Agent run() | 输入 state 有效性、错误路径、边界值 | 各 3-5 |
| P2 | Agent Protocol 合规性 | 所有注册 Agent 实现接口 | 1 |

### 预计总工作量

**3 天**

---

## 十二、死代码清理清单

| 死代码 | 文件行 | 建议操作 |
|--------|--------|---------|
| `ConversationContext.from_dict()` | `context.py:63-76` | 删除（从未被调用） |
| `ContextBuilder._format_history()` | `answer/agent.py:152-173` | 删除或整合到 `build_user_prompt()` |
| `ConversationState.facts` 字段 | `state.py:36` | 删除字段定义 |
| `ConversationState.tool_result` 字段 | `state.py:37` | 删除字段定义 |
| `ConversationManager.build_continue_context()` | `manager.py:199-210` | 删除（从未被调用） |
| `CLARIFICATION` 枚举值 | `context.py:19` | 删除或实现对应逻辑 |

### 预计工作量

**0.5 天**

---

## 十三、各问题工作量汇总

| 优先级 | 问题 | 预计工作量 |
|--------|------|-----------|
| P0 | LLMService 重试机制 | 0.5 天 |
| P0 | Agent 层统一错误处理（@safe_run） | 1 天（含测试） |
| P0 | 知识库全表扫描优化（TF-IDF 倒排索引） | 1 天 |
| P1 | 流式传输 SSE/WebSocket | 3 天 |
| P1 | session_state 类型统一 | 1 天 |
| P1 | Agent 统一接口契约（Protocol） | 0.5 天 |
| P1 | Token 窗口管理 | 1 天 |
| P1 | 多搜索 Provider 支持 | 2 天 |
| P1 | 前端错误反馈 | 1 天 |
| P1 | Python 子进程环境修复 | 0.5 天 |
| P2 | 语义嵌入支持 | 3 天 |
| P2 | 多 LLM Provider | 2 天 |
| P2 | 跨会话长期记忆 | 2 天 |
| P2 | 知识库 re-ranking | 1 天 |
| P2 | 文档解析增强（表格/OCR） | 2 天 |
| P2 | 测试覆盖补充 | 3 天 |
| P2 | 前端状态管理拆分 | 1 天 |
| P2 | 死代码清理 | 0.5 天 |
| P2 | Dockerfile 修复 | 0.5 天 |
| P2 | 数据库索引 | 0.5 天 |
| | **P0+P1 总计** | **约 11.5 天** |
| | **含 P2 总计** | **约 26 天** |

---

## 附录：文件目录索引

```
agentflow/
├── agents/
│   ├── base.py                  ← 新增：AgentProtocol + safe_run
│   ├── registry.py              ← 修改：可选 Protocol 校验
│   ├── router/agent.py          ← 修改：@safe_run("router")
│   ├── planner/agent.py         ← 修改：@safe_run("planner")
│   ├── knowledge/agent.py       ← 修改：@safe_run("knowledge")
│   ├── search/agent.py          ← 修改：@safe_run("search")
│   ├── python/agent.py          ← 修改：@safe_run("python")
│   ├── answer/agent.py          ← 修改：@safe_run("answer") + 死代码清理
│   ├── memory/agent.py          ← 修改：@safe_run("memory") + session_state 类型简化
│   ├── report/agent.py          ← 可选：@safe_run("report")
│   └── observer/                ← 建议：移除或明确规划
├── conversation/
│   ├── session_state.py         ← 修改：add version field, 乐观锁
│   ├── manager.py               ← 修改：死代码清理
│   ├── context.py               ← 修改：死代码清理
│   └── state.py                 ← 修改：删除无用字段
├── services/
│   ├── llm_service.py           ← 修改：添加重试 + 超时
│   ├── search_provider.py       ← 修改：Provider 工厂
│   ├── search_service.py        ← 修改：可选缓存
│   └── search_providers/        ← 新增：brave.py, tavily.py 等
├── knowledge/
│   ├── store.py                 ← 修改：倒排索引优化
│   ├── embedder.py              ← 修改：可选语义嵌入
│   └── parser.py                ← 修改：可选表格/OCR 增强
├── graph/
│   ├── workflow.py              ← 修改：session_state 类型、@safe_run、全局错误处理
│   ├── event.py                 ← 修改：流式事件支持
│   └── executor.py              ← 无需修改
├── tools/
│   └── python_tool.py           ← 修改：沙箱环境策略
├── api/
│   └── routes.py                ← 修改：WebSocket 端点 + session_state 序列化
└── config/
    └── settings.py              ← 修改：搜索 Provider 配置项

frontend/
└── src/
    └── composables/
        └── useChatState.ts      ← 修改：错误反馈 + 拆分建议

tests/
├── test_agent_contract.py       ← 新增：Protocol 合规性测试
├── test_knowledge_store.py      ← 新增
├── test_llm_service.py          ← 新增
├── test_planner.py              ← 新增
├── test_search_tool.py          ← 新增
└── test_python_tool.py          ← 新增
```
