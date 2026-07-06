# OmniForge (AgentFlow) 项目未实现与待改进模块报告

> 生成时间: 2026-07-06 | 基于代码分析 v0.1.0

---

## 一、完全未实现的模块

以下模块有计划或目录骨架，但完全没有功能实现：

### 1.1 Observer Agent

| 项目 | 内容 |
|------|------|
| **位置** | `agentflow/agents/observer/` |
| **状态** | 目录存在，仅有 `__pycache__`，无任何源码 |
| **影响** | 低。属于项目中的"幽灵模块"，造成困惑 |
| **建议** | 短期内不开发则移除目录，或明确规划其职责（监控/审计 Agent 行为） |

### 1.2 跨会话长期记忆（Long-term Memory）

| 项目 | 内容 |
|------|------|
| **位置** | 全局缺失 |
| **状态** | 当前记忆仅维持在同一 session 内（MemoryAgent 的滑动窗口），无法跨 session 持久化用户偏好、常用话题、重要事实 |
| **影响** | 中。多轮对话体验受影响，跨 session 记忆是产品差异化功能 |
| **建议** | 引入向量数据库或 KV 存储，按用户/话题持久化关键实体和摘要 |

### 1.3 流式传输（SSE/WebSocket）

| 项目 | 内容 |
|------|------|
| **位置** | `agentflow/api/routes.py`（全局缺失） |
| **状态** | 全部 API 都是同步请求/响应模式。POST `/chat` 需等待整个工作流（Router → Planner → Search/Execute → Answer → Memory）执行完毕才能看到结果 |
| **影响** | 高。直接影响用户体验，最明显的产品化短板 |
| **建议** | 添加 WebSocket 端点 `/ws/chat`，工作流节点完成时通过 EventBus 发射事件，前端逐步展示各阶段结果 |

### 1.4 用户认证与权限系统

| 项目 | 内容 |
|------|------|
| **位置** | 全局缺失 |
| **状态** | 无注册、登录、JWT、RBAC，只适合本地单用户使用 |
| **影响** | 中。无法对外部署为多用户服务 |
| **建议** | 引入 FastAPI 依赖注入式认证中间件，支持 JWT + OAuth2 |

### 1.5 缓存层（Redis）

| 项目 | 内容 |
|------|------|
| **位置** | 全局缺失 |
| **状态** | 无任何缓存机制。每次 LLM 调用、每次知识库搜索都是实时计算，不支持水平扩展 |
| **影响** | 中。当前单用户场景下可接受，面向产品化不足 |
| **建议** | P3 引入 Redis，支持 session 缓存、消息队列、速率限制 |

### 1.6 可观测性体系（Observability）

| 项目 | 内容 |
|------|------|
| **位置** | 全局缺失 |
| **状态** | 无 Prometheus 指标、OpenTelemetry 追踪、Grafana 仪表板、LLM 调用审计日志 |
| **影响** | 中。生产环境排障和性能分析困难 |
| **建议** | 集成 OpenTelemetry SDK，关键路径埋点（LLM 调用延迟、Agent 执行耗时、知识库搜索耗时） |

### 1.7 Docker 容器级 Python 沙箱

| 项目 | 内容 |
|------|------|
| **位置** | `agentflow/tools/python_tool.py` |
| **状态** | Python 代码执行仅靠子进程 + `env={}` 隔离，无 CPU/内存限制，无网络隔离 |
| **影响** | 中。安全性和资源隔离不足 |
| **建议** | 使用 Docker-in-Docker 或 gVisor 实现容器级执行沙箱 |

### 1.8 Workflow UI 可视化

| 项目 | 内容 |
|------|------|
| **位置** | 前端全局缺失 |
| **状态** | 无 LangGraph 工作流流程图的可视化展示 |
| **影响** | 低。当前可通过日志了解工作流执行路径 |
| **建议** | 使用 ReactFlow 或自定义 DAG 渲染组件展示工作流执行状态 |

### 1.9 知识图谱自动构建

| 项目 | 内容 |
|------|------|
| **位置** | 全局缺失 |
| **状态** | 未开始 |
| **影响** | 低。长期规划功能 |
| **建议** | 从对话记录和知识库文档中自动抽取实体、关系，构建领域知识图谱 |

---

## 二、已实现但不完整的模块

### 2.1 ReportAgent（报告生成 Agent）

| 项目 | 内容 |
|------|------|
| **位置** | `agentflow/agents/report/agent.py` |
| **状态** | 代码已完整实现，但在 registry 中注册为 `inactive`，未接入 LangGraph 工作流 |
| **问题** | 与 AnswerAgent 功能重叠（都是生成最终回答），项目对二者关系没有明确定义 |
| **影响** | 低（未使用）。但属于未完成的功能模块 |
| **建议** | 明确决策——要么删除，要么改造成长文本/报告专用生成器，由 Planner 按需选择 |

### 2.2 KnowledgeAgent 与 Executor 集成

| 项目 | 内容 |
|------|------|
| **位置** | `agentflow/agents/planner/capability.py:30` |
| **状态** | `knowledge.retrieve` 能力已注册但 `tool_name=None`，未绑定任何 Tool |
| **问题** | KnowledgeAgent 绕过 Executor 直接作为工作流节点调用，与统一 Tool 调度体系不一致 |
| **影响** | 中。限制了 Executor 的统一调度能力 |
| **建议** | 创建 `KnowledgeTool(BaseTool)`，将知识检索纳入 Executor 体系 |

### 2.3 搜索服务（Search Service）

| 项目 | 内容 |
|------|------|
| **位置** | `agentflow/services/search_provider.py` |
| **状态** | 仅实现了 DuckDuckGo HTML 爬取作为唯一 Provider |
| **问题** | 单一依赖——DuckDuckGo 被封或限流时搜索完全不可用；无备选降级；无结果缓存 |
| **影响** | 中高。搜索是核心功能之一，单一依赖风险大 |
| **建议** | 实现 BraveSearch / Tavily / Serper / Google 等 Provider，通过配置切换；添加搜索结果缓存 |

### 2.4 LLM 服务

| 项目 | 内容 |
|------|------|
| **位置** | `agentflow/services/llm_service.py` |
| **状态** | 仅支持 OpenAI 兼容 API |
| **问题** | 无多模型路由（不能按任务类型分配不同模型）、无重试机制、无流式支持 |
| **影响** | 高。LLM 是系统智能核心，当前封装过于简单 |
| **建议** | 添加指数退避重试；支持按任务分配模型（简单任务用小模型、复杂用大模型）；支持 SSE 流式输出 |

### 2.5 知识库 RAG

| 维度 | 状态 | 问题 |
|------|------|------|
| 检索方式 | 仅 TF-IDF 词法匹配 | 中文单字切分，无法理解语义；"机器学习算法"匹配不到"监督学习" |
| 搜索性能 | 全表扫描 | 每次搜索加载所有向量到内存，文档增多时性能线性下降 |
| 排序 | 无 re-ranking | 一次 TF-IDF 排序即输出，没有二次精排 |
| 索引维护 | 无增量索引 | 每次新增文档后需重建整个词汇表 |
| 文档解析 | 基础文本提取 | PDF 无表格提取、无图片 OCR、无多列布局处理 |
| **影响** | **高** | 知识检索质量直接影响答案质量 |

### 2.6 Python 执行沙箱

| 项目 | 内容 |
|------|------|
| **位置** | `agentflow/tools/python_tool.py:73` |
| **状态** | 使用 `env={}` 执行 Python 子进程，完全清空环境变量 |
| **问题** | 破坏需要系统 PATH 或 SSL 证书的库（如 `requests`、`ssl` 模块） |
| **影响** | 中。沙箱安全性提升有限，但功能破坏却很显著 |
| **建议** | 保留基本环境变量（`PATH`、`HOME`），仅清除敏感变量 |

### 2.7 对话记忆

| 问题 | 位置 | 说明 |
|------|------|------|
| 会话内记忆有限 | `agents/memory/agent.py` | 仅维持最近 `max_turns*2` 条消息 |
| 无跨会话长期记忆 | 全局 | 用户偏好、常用话题、重要事实无法跨 session 持久化 |
| 无 Token 管理 | `answer/agent.py:258-269` | 无 token 计数和智能截断 |

---

## 三、架构层面缺失

### 3.1 Agent 无统一基类/接口契约

| 项目 | 内容 |
|------|------|
| **问题** | 所有 Agent 都实现了 `run(state) -> dict` 方法，但没有抽象基类或 Protocol 约束 |
| **影响** | 签名全靠开发约定保障，类型系统无法校验。新增 Agent 时容易遗漏方法或参数类型不匹配 |
| **建议** | 抽取 `AgentProtocol`，定义 `run(state: dict) -> dict` 接口。参见 `修改建议报告` 详细策略 |

### 3.2 无统一错误处理

| 项目 | 内容 |
|------|------|
| **问题** | 大部分 Agent 的 `run()` 方法没有 try/except 包裹。任何一个 Agent 抛出未捕获异常都会导致整个 LangGraph 工作流崩溃 |
| **影响** | 高。工作流关键路径上缺乏容错能力 |
| **建议** | 抽取统一错误处理装饰器 `@safe_run`；在 `build_workflow()` 层面添加全局错误捕获节点 |

### 3.3 `session_state` 类型不统一

| 阶段 | session_state 类型 | 位置 |
|------|-------------------|------|
| `run_workflow()` 入口 | `dict \| None` | `workflow.py:319-320` |
| `_conversation_manager_node` 内部 | `SessionState` | `workflow.py:175-178` |
| MemoryAgent.run() 内部 | `dict \| SessionState` | `memory/agent.py:68-76` |
| `WorkflowContext.to_dict()` 输出 | `dict` | `graph/context.py:72-73` |
| `routes.py` 接收后 | `dict \| SessionState` | `api/routes.py:82-84` |

**影响**: 高。每次访问 session_state 都需要防御性的类型检查，是 bug 的常见来源。

### 3.4 遗留 prompt 模板未清理

| 项目 | 内容 |
|------|------|
| **位置** | `agentflow/prompts/*.md` |
| **问题** | 4 个 Markdown prompt 模板文件（planner.md、knowledge.md、search.md、report.md）未被代码引用，与代码中内联 prompt 不同步 |
| **影响** | 中。修改一处遗漏另一处，导致行为不一致 |

---

## 四、测试覆盖严重不足

| 应测模块 | 状态 |
|---------|------|
| `PlannerAgent`（JSON 解析、规则回退） | ❌ 无测试 |
| `KnowledgeStore`（增删搜索、embedding 序列化） | ❌ 无测试 |
| `LLMService`（fallback、模型切换） | ❌ 无测试 |
| `SearchTool` / `PythonTool` | ❌ 无测试 |
| 各 Agent 的 `run()` 方法 | ❌ 无测试 |
| 前端组件 | ❌ 无测试 |
| Agent Protocol 合规性 | ❌ 无测试 |
| **已覆盖** | `test_conversation_runtime.py`、`test_workflow.py` |

---

## 五、基础设施短板

| 问题 | 说明 |
|------|------|
| **数据库索引缺失** | `sessions` 表仅主键索引，按 `updated_at` 排序查询无索引；`chats` 表按 `session_id` 查询无索引 |
| **Dockerfile 路径问题** | `COPY pyproject.toml README.md ./` 因 README.md 在根目录位置问题会失败 |
| **无用户认证系统** | 无注册/登录/JWT，只适合本地单用户 |
| **无可观测性** | 无 Prometheus 指标、OpenTelemetry 追踪、Grafana 仪表板、LLM 调用审计日志 |
| **无沙箱增强** | Python 执行仅 `env={}` 隔离，无 Docker 容器级沙箱、无 CPU/内存资源限制 |

---

## 六、代码质量债务

| 问题 | 位置 | 说明 | 优先级 |
|------|------|------|--------|
| 遗留 prompt 模板 | `agentflow/prompts/*.md` | 未被代码引用，与代码中 prompt 不同步 | P2 |
| 前后端重复类型定义 | `frontend/src/types/index.ts` ↔ `agentflow/models/` | TypeScript 和 Python 类型需手工同步 | P2 |
| `_TOOL_TO_NODE` 硬编码 | `agentflow/graph/workflow.py:53-56` | Tool→节点名映射需随新 Tool 手动更新 | P2 |
| session_state 类型不一致 | `graph/workflow.py:143-169` | dict/SessionState 互转散布在多个方法中 | P1 |
| `match_any` 命名误导 | `agents/router/agent.py:137-139` | 实为 `re.search` 而非 `re.fullmatch` | P3 |
| 前端 `useChatState` 臃肿 | `frontend/src/composables/useChatState.ts` | 470+ 行，应拆分 | P2 |
| 前端大量静默失败 | `frontend/src/composables/useChatState.ts` | catch 块空语句或仅注释 "silently fail" | P1 |
| 多处死代码 | 见 issue_report.md 7.7 | `from_dict()`、`_format_history()` 等从未被调用 | P2 |

---

## 七、优先级汇总

### P0 — 立即处理（影响核心功能）

| # | 问题 | 预计工作量 |
|---|------|-----------|
| 1 | LLMService 添加重试机制 | 0.5 天 |
| 2 | Agent 层统一错误处理 | 1 天 |
| 3 | 知识库搜索全表扫描优化 | 2 天 |

### P1 — 重要（产品化关键）

| # | 问题 | 预计工作量 |
|---|------|-----------|
| 1 | 流式传输 SSE/WebSocket | 3 天 |
| 2 | session_state 类型统一 | 1 天 |
| 3 | Agent 统一接口契约（Protocol） | 0.5 天 |
| 4 | 对话历史 Token 窗口管理 | 1 天 |
| 5 | 添加更多搜索 Provider | 2 天 |
| 6 | 前端错误反馈完善 | 1 天 |
| 7 | Python 子进程环境修复 | 0.5 天 |

### P2 — 建议（质量改进）

| # | 问题 | 预计工作量 |
|---|------|-----------|
| 1 | 语义嵌入支持（sentence-transformers） | 3 天 |
| 2 | 多 LLM Provider 支持 | 2 天 |
| 3 | 跨会话长期记忆 | 2 天 |
| 4 | 知识库 re-ranking | 1 天 |
| 5 | 测试覆盖补充 | 3 天 |
| 6 | 前端状态管理拆分 | 1 天 |
| 7 | Dockerfile 修复 | 0.5 天 |
| 8 | 数据库索引添加 | 0.5 天 |
| 9 | 清理遗留 prompt 模板 | 0.5 天 |

### P3 — 未来（长期规划）

| # | 问题 |
|---|------|
| 1 | Redis 缓存/消息队列 |
| 2 | Agent 间事件驱动架构 |
| 3 | 用户认证与权限系统 |
| 4 | Docker 容器级沙箱 |
| 5 | 可观测性（Prometheus/OpenTelemetry） |
| 6 | Agent 市场/插件系统 |
| 7 | 知识图谱自动构建 |
| 8 | 工作流 UI 可视化 |
