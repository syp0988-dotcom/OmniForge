# OmniForge (multi_agent) 项目全面评审报告

> 评审日期：2026-08-14 ｜ 评审范围：`agentflow/`（137 个 Python 文件）、`frontend/src`（38 个源文件）、`tests/`（48 个测试文件）、`deploy/`、`scripts/`、`eval/`
> 评审方法：静态编译 + Ruff + 全量测试套件 + 6 路并行模块深审（agents / tools+graph / knowledge+db+conversation / api+services / frontend / tests+eval+deploy）+ 关键结论人工复核（含漏洞复现）

---

## 0. 总体结论

这是一个**架构成熟、代码质量高于平均水平**的多智能体项目（FastAPI + LangGraph + Vue3），
具备：统一错误通道、终止策略、电路熔断、指数退避重试、工具注册表动态路由、沙箱化文件系统工具、
RAG 混合检索 + 离线评测、反馈闭环、CI（含 Windows）等工程化设施。

**验证结果：**

| 检查项 | 结果 |
|---|---|
| `python -m compileall agentflow` | ✅ 全部通过 |
| `ruff check agentflow tests scripts` | ✅ 0 告警 |
| 测试套件（本沙箱环境） | ⚠️ 439 passed / 5 failed（环境性）/ 83 errors（Windows 文件锁） |
| 代码风格 | 无明显 `TODO/FIXME` 残留，注释质量好 |

**主要问题集中在四类：**

1. **安全漏洞（高危 4 个）**：Python 沙箱可逃逸、DocxTool 路径穿越、`/workspace/set` 任意文件读写链、前端存储型 XSS。
2. **异步模型误用**：同步 LLM/SQLite/文件解析阻塞事件循环、`time.sleep` 退避、SSE 流未 `aclose()`。
3. **README 与实现漂移**：Browser/MCP/Database 工具是空壳、截图是占位、任务数 8≠3-5、目标类型 10≠6。
4. **测试基建**：测试污染真实数据库/反馈文件、CI 在无密钥环境下必红、Windows 上 83 个临时目录清理错误。

---

## 1. 模块完整性评估（对照 README 声明）

| 模块 | 完整性 | 说明 |
|---|---|---|
| GoalAnalyzer 目标分析 | ✅ 完整 | 嵌入快路径 + LLM 兜底，10 种目标类型（README 写 6 种） |
| Planner 规划器 | ✅ 完整 | 蓝图→模板→函数调用→JSON 四级降级；任务上限 8（README 写 3–5） |
| Knowledge Retriever RAG | ✅ 完整 | 解析/分块/嵌入/Qdrant+FTS5 RRF/评测齐全；`.doc` 声明支持但实际解析出乱码 |
| Web Search | ✅ 完整 | DuckDuckGo HTML 抓取（脆弱）+ Tavily；无缓存 |
| Python Executor | ⚠️ 有缺陷 | 沙箱存在可验证的逃逸（见 Bug H2），临时目录清理异常会冒泡 |
| Tool Executor | ✅ 完整 | 动态注册表 + 并行批次 + 参数修复 |
| **Browser 工具** | ❌ **空壳** | `browser_tool.py` 全部 action 返回 "not yet implemented"；README 却宣称 "Browser automation" |
| **MCP 工具** | ❌ **空壳** | `mcp_tool.py` discover/call 未实现；`list_tools` 却返回 ok（行为不一致） |
| **Database 工具** | ❌ **空壳** | `database_tool.py` 4 个 action 全部 "not yet implemented" |
| Composio 工具 | ✅ 已实现 | 需 SDK + key；`_ACTION_MAP` 是死代码 |
| Reflection 反思 | ✅ 完整 | 规则优先 + LLM 兜底 + 终止策略防死循环 |
| Answer 回答 | ⚠️ 有缺陷 | `build_prompt`/`format_search_results` 是空桩；降级路径错误分类恒为 unknown |
| Conversation Manager | ✅ 完整 | 槽位填充/指代消解/续聊旁路/上下文重写 |
| Conversation Memory | ⚠️ 部分 | 滚动摘要**不跨轮持久化**；20 条上限使压缩几乎失效 |
| ProjectStructurePlanner | ❌ **死代码** | 完整实现但从未接入 workflow 图 |
| 前端聊天 SSE | ⚠️ 部分 | 流式可用；任务树/会话管理缺 rename、工作区浏览器未接线 |
| 前端知识库管理 | ⚠️ 部分 | 无重建索引 UI（后端有 `/knowledge/rebuild`） |
| 前端模型配置 | ✅ 完整 | CRUD + 激活；但错误被静默吞掉 |
| 前端工作区浏览器 | ❌ **未接线** | `browseDirectory` 等实现存在，无任何组件调用 |
| 部署 nginx/k8s | ✅ 完整 | SSE 反代配置正确（`proxy_buffering off`）；k8s 缺 EMBEDDING_API_KEY 会 CrashLoopBackOff |
| 测试套件 | ⚠️ 有污染 | 会写真实 `agentflow.db` 与 `data/feedback/feedback.jsonl`（见 §4） |

---

## 2. Bug 清单（按严重度）

### 🔴 严重（必须优先修复）

| # | 位置 | 问题 |
|---|---|---|
| H1 | `api/files_workspace.py:267-283` + `api/routes.py:103-120` | **任意文件读写链**：`POST /workspace/set` 可将全局工作区根设为任意绝对路径（如 `/` 或 `C:\`），之后 `/files/read`、`/files/create`、`/workspace/browse` 的包含校验全部相对该根放行 → 任意文件读取/写入/列目录。默认无 AUTH_TOKEN，任何人都可调用。 |
| H2 | `tools/python_tool.py:43-55,122-128` | **Python 沙箱逃逸（已复现）**：`io` 不在 `_BLOCKED_IMPORTS`、`open` 不在 `_BLOCKED_ATTRS`，wrapper 只重绑了 `builtins.open`，`io.open()` 保留原始 C 函数 → 可读写宿主机任意文件（AST 校验通过）。`sys.modules['io']` 也不会被清理。 |
| H3 | `tools/docx_tool.py:72-76` | **DocxTool 无路径包含校验**：`_resolve()` 对绝对路径原样放行、相对路径不做 `relative_to` 检查，且无 `validate()` 覆盖 → 可读写磁盘任意 `.docx`（与 FileSystemTool 的沙箱不一致）。 |
| H4 | `frontend/src/components/markdown/MarkdownRenderer.vue:2,51-52` | **存储型 XSS**：`markdown-it({html:true})` + `v-html` 无任何净化（无 DOMPurify）。知识库文档/网页搜索结果可经 LLM 回答注入 `<img onerror=...>` 等可执行 HTML。 |

### 🟠 高危

| # | 位置 | 问题 |
|---|---|---|
| M1 | `api/chat.py:275-285` + `services/llm_service.py:420-491` | **同步 LLM 流式生成器在 async SSE 中直接迭代**：每个 token 可阻塞事件循环最多 60s，整个服务对其他请求冻结；断连检测在此期间失效。 |
| M2 | `api/chat.py:226-292` | **`workflow.astream()` 从不 `aclose()`**：客户端断连/超时/异常时后台节点继续执行，浪费 LLM token、泄漏任务。 |
| M3 | `services/llm_service.py:347,566` | **`time.sleep()` 同步退避阻塞事件循环**：LLM 失败时每个重试最多睡 10s，期间所有并发请求卡死。 |
| M4 | `services/llm_service.py:413-418,481-486` | **`[fallback] 用户输入前160字` 作为"答案"直接返回给用户**：LLM 失败时用户看到的是回声文本而非错误提示，且不走降级状态通道。 |
| M5 | `api/chat.py:147,299`、`knowledge.py:105,235`、`files_workspace.py` 多处 | **内部异常 `str(exc)` 直接泄漏给客户端**：可能包含绝对路径、DB 错误、provider 信息。 |
| M6 | `api/executions.py:107-135` | **`resume_execution` 语义错误**：用空 question 重跑整个 workflow 再覆盖 task_queue，并非从 checkpoint 恢复。 |
| M7 | `api/knowledge.py:83,141-179` | **`/upload` 与 ZIP 摄取是阻塞同步代码**：解析/分块/嵌入可能耗时数分钟，直接在 async handler 内执行，阻塞事件循环。 |
| M8 | `api/chat.py:57,198` | **损坏的 session_state JSON 导致 500**：`json.loads` 无防护。 |
| M9 | `app/main.py:151-157,172-194` | **中间件顺序问题**：认证在 CORS 之前执行，`AUTH_TOKEN` 开启时浏览器 OPTIONS 预检拿 401 且无 CORS 头 → 前端完全不可用。 |
| M10 | `api/sessions.py:20-22`、`sqlite.py:381-391`、`api/knowledge.py:216-224` | **负/大 limit 无上限**：`GET /history?limit=-1` 在 SQLite 中等价于无限制，返回全表；`top_k=-5` 返回几乎全部结果。 |
| M11 | `database/sqlite.py:65-67`（已复现） | **`with sqlite3.connect(...)` 只 commit 不 close**：每次 `SQLiteStore()` 构造泄漏一个连接；线程本地连接成功路径也从不关闭 → Windows 上 DB 文件永久锁定（已用最小脚本复现 WinError 32），是 83 个测试错误与 Windows CI 失败的根因。 |
| M12 | `graph/workflow.py:660-665,691-696` | **任务状态错乱**：query_rewriter 节点把列表中第一个 `todo` 任务标为 running（而非路由实际选中的最高优先级任务），search 节点把所有 running 任务标 done → 未执行的任务被标记为完成。 |
| M13 | `graph/workflow.py:398-400,437-439,491-493` | **`tool=="knowledge"` 任务永远无法执行**：三个路由器都把它踢回 reflector/planner，形成 LLM 浪费循环，直到终止策略强停。 |
| M14 | `frontend/src/composables/useChatActions.ts:40-43` | **10 秒固定看门狗**：不随活动重置，任何 >10s 的正常工作流被强制 abort，然后静默走非流式 `/chat` 重跑 → 双倍 LLM 开销、答案不一致、进度丢失。 |
| M15 | `frontend/src/composables/useChatActions.ts:206-218` | **`newChat` 不中断在途流**：旧流继续向新会话追加内容，`done` 还会把 sessionId 改回去 → 会话串扰。 |
| M16 | `frontend/src/components/chat/ChatInput.vue:135-149` | **中文输入法组合态回车误发送**：无 `e.isComposing` 判断，候选确认回车会提前发送半截内容（中文应用的高频问题）。 |
| M17 | `agentflow/agents/answer/agent.py:70` | **降级错误分类恒为 "unknown"**：`classify_error(Exception(llm_error字符串))` 永远不匹配 openai 异常类型 → 用户永远看到通用提示。 |

### 🟡 中危（节选）

- **检索正确性**：`knowledge/retrieval.py:126-150` 文档过滤在 RRF 截断之后应用，`document_ids` 过滤下可能返回 0 结果；FTS5 查询含内嵌引号（如 `12" 屏幕`）→ 语法错误被静默吞掉返回空（`retrieval.py:205`、`sqlite.py:601`）。
- **SQLite**：无 `busy_timeout`（并发写锁错误）、复用连接无 rollback、`search_long_term_memory` 用 `%query%` LIKE 全表扫描。
- **LLM 预算**：`estimate_tokens` 用 `len//4` 估算，中文实际约 1.5 token/字 → 会话 token 预算形同虚设（低估约 6 倍）。
- **OpenAI 客户端**：`OpenAI(timeout=60)` 未设 `max_retries=0`，SDK 自带 2 次重试与业务层 2 次重试叠加 → 单次调用最坏 ~9 次 × 60s ≈ 9 分钟。
- **`complete_stream` 熔断节点写死**：`llm_service.py:465` 忽略 `node_name` 参数；熔断只包 `create()` 不包流迭代，流中网络失败不计入熔断。
- **GitTool 参数注入**：`git_tool.py` 的 branch/revision/files 以 `-` 开头可被当作 git 选项（无 shell 注入但有选项注入）；且 `GitTool` 默认 repo 为服务进程 cwd（项目根目录），未绑定沙箱 outputs/。
- **Composio 参数污染**：`registry.py:257-260` 把 `action`/`task_id` 等元字段注入第三方 API 参数（`composio_tool.py:202`）。
- **文件系统工具**：`read_file`/`tree`/`list_directory` 无输出上限（对比 git 有 50KB 上限）；`code_prompt` 参数被声明但不读取 → 只传 `code_prompt` 的代码文件会写出**空文件且报成功**（`filesystem_tool.py:109-120,293-327`）。
- **前端**：`api/sse.ts:47-53` CRLF 流永不触发事件分发；`chatState.ts:37-39` 每 token 全量序列化 sessionStorage（O(n)）；`MarkdownRenderer.vue:90` 每 token 全量重渲染（O(n²)）；`MessageItem.vue:62-73` 每条消息常驻 2s interval 定时器；`fetchWithRetry` 对 POST 网络错误重试会重复执行服务端工作流；SSE `error` 事件被前端吞掉。
- **安全加固缺口**：`/knowledge/rebuild`、`/sessions/cleanup`、`/executions/*/resume` 无节流/确认；`/health` 未认证暴露工具清单；`/files` 响应返回绝对路径；预览读取 `read_bytes()` 全量读入内存后再截断（OOM 风险，`files_workspace.py:90`、`knowledge.py:261`）。
- **安全与配置**：`/workspace/set` 会向任意目录写入探针文件 `.omni_forge_write_test`；`LLMModelUpdate` 无边界（temperature=99、max_tokens=-5）；`message` 无长度上限。

### 🔵 低危（节选）

- `registry.py:76-78` 重复注册静默覆盖（文档声称 ValueError）；`registry` 非线程安全（首次并发请求可能重复构建）。
- `graph/task.py:191` `TaskStatus(未知字符串)` 直接抛 ValueError。
- `graph/context.py:93-97` `to_dict()` 对嵌套对象原样拷贝，bytes/numpy 会破坏 FastAPI 序列化。
- `workflow.py:157-172` 路由函数内原地改 `_trace`，LangGraph 状态合并后 route 字段可能丢失。
- `knowledge/embedding_cache.py:119` `merge()` 在缺失项交错时可能错位截断向量。
- `conversation/manager.py:258-263` `finalize_turn` 把编号计划步骤误判为选项提示。
- `search_provider.py:86` `min(max_results,20)` 未从下方钳制（-5 传入 API）；`search_service.py:95` 把 Tavily 结果硬标 "duckduckgo"。
- `main.py:186-189` 指标按原始 path 打标签，基数无限增长；每个 logger 各开一个日志文件句柄（20+ 个）。
- 前端：`TaskTree` 按 task.id 做 key 而 planner 复用 id；`ModelsSettings` 错误全静默；上传 accept 列表与后端不一致；`languageMode` 无实际作用；无障碍缺失（aria-label/dialog/focus trap）。

---

## 3. 优化清单（按收益排序）

### 性能 / 延迟
1. **异步化 LLM 调用**：换 `AsyncOpenAI`，`complete_stream` 改 async generator，`chat.py` 用 `async for` 迭代 → 解决事件循环冻结 + 断连检测 + 并发聊天（收益最高）。
2. **工作区扫描缓存**：`context_builder.py:372-400` 每次 planner 调用全量 `rglob("*")`（`settings.max_workspace_files` 定义了但从未使用）；按目录 mtime 失效做 memoize。
3. **搜索缓存**：SearchTool 无缓存，DuckDuckGo 抓取每次 1-3s；加 TTL 查询缓存。
4. **Git 状态缓存**：git status/diff 每次 subprocess，规划/反思多轮重复调用；按 index mtime 缓存。
5. **复用 ThreadPoolExecutor**：`executor.py:311` 每次并行批次新建 6 线程池。
6. **嵌入批量化 + 缓存**：`embedding_cache` 已存在但 `lookup` 逐条查询，改批量；解析按 content_hash 缓存避免重复解析。
7. **Python 工具**：持久化沙箱子进程复用，避免每次 100-300ms 解释器启动。
8. **前端**：markdown 渲染节流（60-100ms）、sessionStorage 深度 watch 防抖、消息列表虚拟滚动、hljs 语言按需注册（当前 11 种语言全量打进首屏 bundle）。

### 工程化
9. **统一异常处理**：`app.add_exception_handler`，绝不回显 `str(exc)`；全路由补 `response_model` 与 Pydantic 请求体（修复负 limit、越界参数、无上限 message 等一批问题）。
10. **事件循环卸载**：`asyncio.to_thread` 包 SQLite/解析/嵌入调用；`/upload` 摄取改 BackgroundTasks + 状态端点。
11. **SSE 加固**：加 `Cache-Control: no-cache`、`X-Accel-Buffering: no`、周期 `: ping` 心跳；移除 15 字符/20ms 的人工切块。
12. **`/workspace/set` 改造**：会话级工作区根 + 限制在配置的基目录（如 `data/workspaces/`）下。
13. **中间件顺序修复**：认证放到 CORS 之后或豁免 OPTIONS 预检。
14. **速率限制**：`/chat`、`/upload`、`/knowledge/rebuild` 加 per-IP 限额。
15. **OpenAI 客户端** `max_retries=0` + 单次调用总 deadline（当前 ~9 分钟最坏情况远超 `max_request_seconds=300`）。
16. **SQLite**：`busy_timeout`、`PRAGMA optimize`/WAL checkpoint、复用单连接 + 事务化批量插入。
17. **索引与查询**：FTS5 查询转义内嵌引号；`/history` 按 session 过滤；`delete_document` 去掉 O(n) 全表扫描。
18. **依赖瘦身**：axios（15 个 REST 调用）可换 fetch；`omniforge/` 空包删除；`agentflow/docker/` 旧布局与根 docker-compose 重复。

---

## 4. 测试与 CI 问题

1. **测试污染真实数据（严重）**：`test_chat_endpoints.py` 直接写 `agentflow/database/agentflow.db` 和 `data/feedback/feedback.jsonl`（反馈文件里已有测试消息原文）。
2. **CI 必红**：`test_workflow.py::test_health_endpoint` 断言 `status=="ok"`，但 CI 无 `DEEPSEEK_API_KEY` → `/health` 返回 `degraded`；`test_workflow_produces_answer` 完全未 mock（真实 LLM + 网络）。
3. **Windows CI 必红**：83 个 tmp 清理错误（Bug M11 的连接泄漏根因）。
4. **顺序依赖**：`_workflow_cache`、`_executor_instance`、`routes._workspace_root`、`set_feedback_path` 都是模块级全局，多数测试文件不重置。
5. **Eval 框架**：`eval/run_all.py:176-177` RAG 套件加载不存在的 `eval_dataset.jsonl`；所有异常被吞且退出码恒 0；`completion_eval/runner.py:182` 每轮 `mock_llm.reset()` 使多轮模拟失效；调参无留出集（同一数据集调参+评估，报告的准确率偏乐观）。
6. **测试缺口**：llm_service（重试/熔断/预算）、search_provider/search_service、executions/sessions/memory/models/system 路由、api 级路径穿越链、SSE 断连/超时路径、Python 沙箱 `io.open` 逃逸、DocxTool 穿越、并发上传同名竞态、前端组件（无 @vue/test-utils/jsdom，仅 1 个 sse.test.ts）。

---

## 5. 优先修复建议（Top 10）

1. `MarkdownRenderer.vue` 关闭 `html:true` 或接入 DOMPurify（H4）
2. `PythonTool` 封锁 `io`/`_io`/`codecs` + `open` 入 `_BLOCKED_ATTRS` + wrapper 清理 `sys.modules`（H2）
3. `DocxTool` 复用 FileSystemTool 的 `_resolve`/`validate` 包含校验（H3）
4. `/workspace/set` 会话化 + 基目录限制（H1）
5. `chat.py` 加 `try/finally await stream.aclose()` + `AsyncOpenAI` 流式（M1/M2）
6. `SQLiteStore._initialize` 真正关闭连接 + `close()` 幂等 + 测试用 conftest 隔离（M11）
7. `ChatInput` 加 `isComposing` 守卫；`useChatActions` 修 10s 看门狗与 `newChat` 竞态（M14/M15/M16）
8. 修 `test_health_endpoint`/`test_workflow_produces_answer`，加 `tests/conftest.py` 统一隔离（§4.1-4.4）
9. 路由层补 Pydantic 校验 + 负 limit 钳制 + 统一异常出口（M5/M10）
10. README 与实现对齐：删除/标注空壳工具、修正任务数/目标类型声明

---

*本报告由 6 路并行子代理深审 + 人工复核生成；所有关键漏洞均已对照源码确认，其中 SQLite 连接泄漏（M11）与 Python 沙箱逃逸（H2，子代理执行验证）已实际复现。*

---

## 附录 A：Agents 模块深审补充（完整版新增项）

以下为 agents 模块完整深审报告中主表未收录的细节（编号沿用子代理报告）。

### 补充 Bug

| 级别 | 位置 | 问题 |
|---|---|---|
| 高 | `agents/python/agent.py:100-101` | 未找到代码块时任务被标 **failed** → 路由进 reflector → replan 循环（"无需运行代码"的正常场景被当作失败，仅靠终止策略兜底）；且该 agent 执行 `state["question"]`，忽略 planner 放在任务 input 里的 `code` |
| 高 | `agents/planner/agent.py:440-480` | `_handle_non_project` 用新计划的 tasks **整体替换** `task_queue`（L478），replan 时丢弃在途 TODO/FAILED 任务（合并逻辑只存在于 project 流程 `_merge_into_queue`） |
| 中 | `agents/reflection/agent.py:416-422` | 工作区扫描 `iterdir()` 在 try **之外**，OSError/权限错误（或 project_name 为空扫 CWD）直接冒泡 → safe_run 错误态而非回退规则评估 |
| 中 | `agents/reflection/agent.py:530-536` | 任务匹配是子串匹配 `task_name in t.goal`：`write_file` 会命中第一个含该词的 goal，多个结果塌缩到同一任务，后续任务永不匹配 |
| 中 | `agents/answer/agent.py:160-163` | `ss.metadata.pop("last_failure_reason")` **就地变异** session_state（删除重试可能需要的信息）；dict 形态的 session_state（测试/eval 中使用）会 AttributeError——违反 base.py "不得就地修改"契约 |
| 中 | `agents/memory/agent.py:50,54` | `question == ""` 时仍追加 user 消息（流式路径 answer="" 时 assistant 空轮被跳过）→ 历史角色错位，污染后续 prompt |
| 中 | `agents/goal_analyzer/intent_index.py:179-214` | `_ensure_ready()` 无锁：并发首个请求重复嵌入全部 ~90 个锚点短语（重复 API 开销）；`216-218` 一次瞬时嵌入失败会从 `match()` 抛出，本请求**不走 LLM 兜底** |
| 中 | `agents/search/agent.py:34-35` | 工具失败被吞进 `SearchResult.metadata["error"]` 且无人读取 → 用户看到空结果却不知道搜索失败 |
| 中 | `agents/python/agent.py:107-113` | 无队列任务时结果只落 `state["python_result"]`，而该键没有任何 agent/ContextBuilder 读取 → 结果对反思/回答不可见 |
| 中 | `agents/python/agent.py:132-136` | 无代码块时对**整个输入**做 `ast.parse` 兜底执行——粘贴的任意合法 Python（如 `2**10`）会被意外执行 |
| 中 | `agents/planner/task_queue.py:66-84` | `update()` 用 `setattr` 接受任意键（拼写错误静默变新属性）；`TaskStatus(value.lower())` 遇未知字符串抛 ValueError——reflection 的 `TaskUpdate.status` 无枚举校验，一个坏状态就能让 reflection 崩溃 |
| 中 | `agents/knowledge/agent.py:48-49` | 直接 `r['filename']`/`r['content']` 索引（缺键即 KeyError），与 answer/agent.py 的防御式写法不一致 |
| 中 | `agents/reflection/agent.py:255`、`planner/agent.py:375,445` | `int(state.get("_replan_count", 0))` 遇非 int 值抛 ValueError |
| 低 | `agents/planner/agent.py:973-980` | 无 code_prompt 时用裸 `task.goal`（如 `"write_file: app/main.py"`）做代码生成指令，提示质量差 |
| 低 | `agents/planner/prompt.py:81 vs 129` | JSON 示例含 `"current_stage": ""`，规则却写"不要输出 stage 名称"——prompt 自相矛盾 |
| 低 | `agents/reflection/agent.py:444-446` | 日志打印原始 LLM 输出前 300 字符，可能含用户 PII/代码 |
| 低 | `agents/answer/agent.py:76` | `knowledge_results == []` 判断：knowledge 节点崩溃（safe_run 无该键）或为 None 时跳过"无参考"守卫，LLM 零上下文作答 |

### 补充优化

1. **代码生成串行瓶颈**：`planner/agent.py:989-1008` 每轮最多 8 文件 × 2 重试 = **16 次串行 LLM 调用**；文件间相互独立（共享 project brief），应并行化。
2. **4 处重复的 `_parse_json`**（goal_analyzer/planner/reflection/project_structure_planner）+ **2 处 `_fix_json_newlines`** → 抽到 `utils/json_utils.py`，统一做 dict 类型校验（顺带修复"非 dict JSON 崩溃"两个高危）。
3. **`numpy` 顶层导入**（intent_index.py:14）拖慢应用启动 → 首次嵌入时惰性导入。
4. **`_shared_knowledge_store()` 每请求重导入 `agentflow.api.routes`**（knowledge/agent.py:70-77）→ 缓存引用并移到 service 层，切断 agent→api 耦合。
5. **设计调用无条件执行**（planner/agent.py:155）：蓝图中已有实质内容或 `plan.direct_answer` 时跳过；按 goal memoize（replan 重复推导同一设计）。
6. **规则评估重复扫描**：`reflection/agent.py:553-569,587` 每次评估都跑 `match_template` + 全目录扫描，队列无文件创建任务时应跳过。
7. **死代码清理**：`_workspace_has_content`（reflection/agent.py:865-880）、`_UNUSED`（planner/agent.py:316）、`capability_registry`/`list_grouped`（capability.py）、answer 空桩（answer/agent.py:391-400）。

### 补充测试缺口

- **SearchAgent.run 与 PythonAgent.run 完全没有测试**（test_python_sandbox 只测 PythonTool）。
- **registry.py 无测试**；project_structure_planner 的 `run()`/LLM 路径无测试。
- goal_analyzer：非 dict JSON（数组/字符串）路径无测试；answer：LLM 抛异常 / BudgetExceeded / `[fallback]` 泄漏路径无测试。
- reflection：工作区扫描 OSError、子串匹配歧义、非 dict 队列条目均无测试。
- `test_answer_agent.py:96-98` 断言 `clean_answer(...) is not None`（空洞断言）；planner 的 `_generate_project_design`/`_fill_code_content` 重试校验循环无测试。

### Agents 模块最优先修复（子代理建议）

1. goal_analyzer + planner 校验 LLM JSON 必须是 dict（修复非 dict 崩溃）→ 失败转为正常降级而非整节点报错。
2. answer 模式 LLM 调用包 try/except 并识别 `[fallback]` 响应（停止向用户泄漏回显/空答案）。
3. 设计调用按需执行 + 非降级时才调用。
4. `_handle_non_project` 改为合并而非替换 task_queue。
5. reflection 工作区扫描包进 try + 修复子串任务匹配。

---

## 附录 B：Knowledge / Database / Conversation 模块深审补充（完整版新增项）

### 补充 Bug（编号沿用子代理报告）

**Knowledge**
| 级别 | 位置 | 问题 |
|---|---|---|
| 中 | `knowledge/embedder.py:110-117` + `embedding_cache.py:110-119` | `merge()` 丢弃 None 占位：API 返回向量少于请求数时列表变短 → `store.add_document` 构造 ragged `vectors_array`，Qdrant `add` 用 `zip` 静默截断 → 向量与元数据错位 |
| 中 | `knowledge/store.py:136-138` | 嵌入/Qdrant 失败时只清理 SQLite；Qdrant 已部分 upsert 的向量**永久残留**（孤儿向量） |
| 中 | `knowledge/store.py:92-115` | `add_document` 非原子：文档行先提交，chunk 循环（在 try 之外）异常会留下半成品文档 + 孤儿 chunk |
| 中 | `knowledge/retrieval.py:264-273` + `settings.py:48-51` | `min_score=0.10` 作用于加权混合分 `α·vs+β·ls`，与 RRF 排序不一致：仅 FTS5 命中的 chunk（得分≈0.005-0.06）即使被 RRF 排第 1 也被过滤 |
| 中 | `knowledge/retrieval.py:191-223` | FTS5 只剥首尾引号：词内嵌 `"`（如 `say "hi"`）→ 非法 FTS5 语法 → `search_chunks_fts` 静默返回 []（合法查询得空结果且无提示） |
| 中 | `knowledge/reranker.py:105-121` | `LLMReranker.rerank` 固定返回 `knowledge_rerank_top_k`（默认 3），无视请求的 `top_k`：`KNOWLEDGE_RERANKER=llm` 时 `search(top_k=5)` 最多返回 3 条 |
| 低 | `knowledge/embedder.py:131-134` | `_embed_api` 按 provider 返回顺序追加、忽略 `item.index`：乱序 provider 静默错位向量 |
| 低 | `knowledge/embedder.py:139` | `embed_query` = `self.embed([text])[0]`：API 返回空列表 → IndexError |
| 低 | `knowledge/retrieval.py:97` | `top_k is None` 时 `min_score` 被静默忽略（`min_score or 0.0`）——直接调用 `HybridRetriever.search` 与 `KnowledgeStore.search` 默认值不一致 |
| 低 | `knowledge/retrieval.py:291,330` | `_format_vector_only`/`_format_lexical_only` 的 doc 过滤循环逐条 `get_chunk_with_document`（N+1 查询） |
| 低 | `knowledge/store.py:203,209` | 每次 search 全表加载 `get_all_documents()`（仅判空）+ `_check_embedding_model()` 额外 DB 读 |
| 低 | `knowledge/index.py:158-164` | 每次 search 调远程 Qdrant `count()`（每查询 2 次远程往返） |
| 低 | `knowledge/parser.py:275` + `api/knowledge.py:156` | ZIP 内不同子目录同名文件在 `UPLOAD_DIR / fname.name` 下互相覆盖 → 第二个被误判为 duplicate |
| 低 | `knowledge/embedding_cache.py:37` | 缓存库无 WAL/busy_timeout：多进程（uvicorn workers）→ "database is locked" |
| 低 | `knowledge/embedding_cache.py:129-143` | `get_shared_cache` 惰性初始化有竞态（双线程可建两个缓存）；无淘汰策略，缓存无限增长 |

**Database（sqlite.py）**
| 级别 | 位置 | 问题 |
|---|---|---|
| 中高 | `sqlite.py:49-55` | `_connect` 只在新建连接时 close：**复用连接上抛异常不 rollback** → 残留中毒事务，同线程后续查询报 "cannot commit - no transaction is active"/读到旧数据 |
| 中 | `sqlite.py:393-404` | `get_session_messages` 返回整会话无上限消息列表，长会话撑爆 API 响应与内存 |
| 中 | `chat.py:120` + `sqlite.py:339-347` | `update_session_state` 整块 JSON 后写覆盖：同一会话并发两轮会丢一轮状态（无合并/版本） |
| 低 | `sqlite.py:365-367` | `delete_session` 手动删 chats，而 FK `ON DELETE CASCADE` 已做（双重删除） |
| 低 | `sqlite.py:550-557` | `IN (?,...)` 占位符 >999 在旧 SQLite 构建上超 `MAX_VARIABLE_NUMBER` |
| 低 | `sqlite.py:44-55` | 线程本地连接在 worker 线程死亡时永不关闭（线程池 churn 泄漏句柄） |
| 低 | `sqlite.py:744-751` | `set_active_model` 两条 UPDATE 无事务：中间失败 → 无激活模型 |
| 低 | `sqlite.py:281` | f-string SQL（`PRAGMA user_version = {常量}`，不可注入但模式存在）；迁移其实未按版本门控（每次启动无条件跑 `ALTER TABLE`） |

**Conversation**
| 级别 | 位置 | 问题 |
|---|---|---|
| 中高 | `manager.py:257-263` | `finalize_turn` 把回答里**任意连续编号列表**（如 "1. 创建项目 2. …" 计划步骤）当作待选选项 → 会话被误置为 waiting_user |
| 中 | `memory/agent.py:56-58` | 历史在压缩前就被硬截断为 `max_turns*2`=20 条：8000 token 预算下压缩几乎不触发，旧轮次直接丢弃（与"分层记忆"声明矛盾） |
| 中 | `manager.py:141` | "继续"信号把下游 question 替换为 `session_state.waiting_for` 的字面字符串（如 "选择一个选项"），原样进入 LLM |
| 中 | `chat.py:51-53,69-70` | 每轮历史**完全来自客户端** `request.history`（服务端从不加载 `get_session_messages`）：客户端截断的历史静默丢失上下文；`chats` 表无限增长（仅会话级 TTL 清理） |
| 低 | `manager.py:149-169` | 槽位填充把原始输入塞进第一个空槽，无校验/类型检查；`has_unfilled_slots` 用 `v == ""`，`None`/`False` 槽值被算作已填 |
| 低 | `session_state.py:137-147` | `reset()` 保留 `token_usage` 但 docstring 声称"Reset all fields" |
| 低 | `rewrite.py:31-34` | `^[1-9][0-9]?$` 把任意 1-2 位数字（如 "99"）当序数选项 |
| 低 | `rewrite.py:279-316` vs `manager.py:525-564` | `_TASK_INQUIRY_PATTERNS` 与 `_looks_like_new_task_request` 两处逐字重复（漂移风险） |
| 低 | `manager.py:396-408` | `_extract_entities` 仅 CJK：Latin 实体（"Java"/"Python"）从不跟踪，"改成 Java" 的指代丢失目标 |
| 低 | `manager.py:514-520,648-657,671-672` | 热路径方法内每次调用 `re.compile` |
| 低 | `compression.py:150-153` | `min_keep_messages=0` 时 `history[:-0] == []` → 全部历史被压缩成一条摘要（上下文全丢；设置默认 6，属边界） |
| 低 | `compression.py:42-56` | `_transcript_text` 只保留**最后** 12000 字符，最早（通常是目标所在）的上下文被丢 |
| 低 | `compression.py:141-142` | 预算内时返回全量历史、忽略 `min_keep_messages`，与超预算路径语义不一致 |

### 补充优化
1. **检索结果 LRU 缓存**：`retrieval.py:68` 按 `(query, top_k, doc_ids)` 短 TTL 缓存——多智能体规划循环中重复问题常见，一次命中省掉 Qdrant+FTS5 两次远程调用。
2. **复用已存 `embedding_dimension`**（store.py:132-135）跳过维度探测 API 调用（embedder.py:141-146）。
3. **解析结果按 content_hash 缓存**：`rebuild` 会重解析重分块全部文档，缓存后 rebuild 近乎免费。
4. **嵌入并行化**：`batch_size=20` 固定（store.py:119），提到 32-64 并用 ThreadPoolExecutor 并行（embedder 无状态）。
5. **`PRAGMA wal_checkpoint(TRUNCATE)`** 放进现有清理任务，防 `-wal` 无限增长。
6. **`get_session_messages` 分页**（LIMIT/OFFSET 或 since-id）。
7. **Qdrant 集合大小内存缓存**替代每次 search 的 `count()`（省一半远程往返）。
8. **预编译正则**移到模块作用域（manager.py/rewrite.py 热路径）。

### 补充测试缺口
- 混合检索仅 1 个测试：无 RRF 数学、`_to_fts_query` 转义（内嵌引号/操作符/CJK）、混合模式 min_score 过滤、doc 过滤在截断后（B1/B2）、α/β 加权。
- embedding_cache 2 个测试：无 `merge()` 错位（B6）、多模型键、并发、禁用路径。
- sqlite 8 个 CRUD 测试：无 busy-timeout/锁、复用连接回滚（D1）、WAL、FTS5 MATCH 错误路径、`IN()` 参数上限、`/history` 跨会话混合。
- conversation 116 个测试但无：编号计划列表被误判选项（C1）、滚动摘要跨轮持久化（C3）、`min_keep_messages=0`（C13）、并发 `update_session_state`（D9）。
- 上传 5 个测试：无 zip 含代码文件、PDF/DOCX 走 API、失败清理断言（B9/B10）。
- 无任何"async 处理器内不得有阻塞网络调用"的结构性测试（A1/A2）。

### 该模块最优先修复（子代理建议）
1. A1/A2 — 阻塞图执行与上传摄取移出事件循环。
2. D2/D1 — SQLite `busy_timeout` + `_connect` 异常路径 rollback（并发正确性）。
3. B1/B2 — 混合检索过滤/评分一致性（融合后 doc 过滤、单一来源分数下限）。
4. B4/D3 — FTS5 转义（引号翻倍或拒绝）且不再静默吞 FTS 错误。
5. C2/C3 — 分层压缩落地：去掉/提高 20 条上限 + 滚动摘要持久化进会话状态。
6. C1 — 不把编号计划步骤当选项提示。
7. B6/B9 — 向量对齐守卫 + 摄取失败时清理 Qdrant。

---

## 附录 C：测试 / Eval / 部署模块深审补充（完整版新增项）

> 修正：tests/ 实为 **48 个 .py（519 个测试函数）+ 50 个 .pyc 字节码**，"98 个测试文件"的说法被字节码灌水；README"500+ 测试"大致属实。

### 测试基建补充 Bug

| # | 位置 | 问题 |
|---|---|---|
| 13 | `tests/test_feedback_collector.py:64` | **空洞断言**：`assert record_feedback(...) is True or True` 永远通过，什么都没测 |
| 14 | `agentflow/eval/feedback.py:36-42` | `set_feedback_path` 测试后**从不还原为 None**：后续测试可能写到已删除的临时目录或真实文件（顺序依赖） |
| 16 | `workflow.py:1012,1023-1025` | `_executor_instance` 单例 + 真实 `outputs/` 目录：跑 tool_executor 节点的测试会往仓库 `outputs/` 写真实文件 |
| 18 | `tests/test_security.py:35` | 直接改 `settings.auth_token = ""`（非 monkeypatch），且 auth 测试仍通过 `get_store()` 初始化真实 DB |
| 19 | `tests/test_upload_route.py:22,58` | 用**已弃用的 `tempfile.mktemp`**；构造 `KnowledgeStore` 不传 embedder → 默认 `QwenEmbedder()` → 隐藏的嵌入 API/网络依赖 |
| 20 | `tests/test_chat_endpoints.py:119` | 弱断言：`status_code in (200, 422)` 两种不同行为都"通过" |
| 21 | `pytest.ini:4` | `-p no:cacheprovider` 禁用了失败缓存；无 `--strict-markers`、无 marker 注册、无超时插件 → 挂起的网络测试会阻塞 CI 直到作业超时 |
| 22 | `tests/test_python_sandbox.py:64-67` | 测试真的连 localhost:8080 的 socket —— 本机恰好有服务监听时行为变化 |

### Eval / 部署补充问题

| # | 位置 | 问题 |
|---|---|---|
| 28 | `tests/mock_llm.py` vs `agentflow/eval/common.py:96` | **两个同名 `MockLLMService` 且 API 不同**；eval runner 绕过测试 mock 直接改 `llm_mod._llm_service`（模块全局，pytest monkeypatch 不可见） |
| 29 | `agentflow/eval/common.py:33-44` | 数据集校验语义不统一：子类有的 raise、有的返回 False，基类不归一化 |
| 30 | `knowledge/eval/data/` | 三份数据集（150/24/90 条）重叠且无版本来源；LLM 生成的 ground truth（`relevant_chunk_ids`）可能内嵌生成模型偏见 |
| 31 | `blueprints/loader.py:213-261`、`configurator.py:146-186` | 蓝图 LLM 路径（`_llm_confirm`/`_llm_fill`）零测试，走真实 LLM |
| 32 | `blueprints/tests.py:46-48` | 硬编码"5 个蓝图"且重复 test_blueprints.py；`python_files = test_*.py` 规则下 pytest 从不收集它 |
| 33 | `blueprints/loader.py:35-38` | 死 `templates/` 目录：Jinja loader 指向不存在的目录，无 YAML 用 `template_ref`，外部模板静默回退内联（Jinja2 沙箱化 ✅） |
| 34 | `Dockerfile:22-23` | HEALTHCHECK 的 `urlopen` 无超时，仅靠 Docker 自身 5s kill 兜底（compose 里重复声明的带 timeout 版本使其冗余） |
| 37 | `deploy/k8s/deployment.yaml:34-38` | 生产启动要求 DEEPSEEK + EMBEDDING 双 key，但 secret 只接入了 DEEPSEEK → **CrashLoopBackOff** 直到手工补全 |
| 38 | `deploy/k8s/*.yaml` | 镜像 `:latest` 无 `imagePullPolicy`；web-deployment 无资源限制、无 liveness（只有 readiness） |
| 39 | 仓库内 | 提交了 `intent_results.json`/`planner_results.json`/`mock_tool_results.json`/`eval_results.json`（78KB）一次性运行产物，冒充参考结果 |

### 部署正面项（已核实）
- nginx SSE 三处配置全部正确（`proxy_buffering off` + `proxy_cache off` + 3600s 读写超时 + ingress 注解）——经典 SSE 反代坑已被避开。
- Jinja2 蓝图沙箱化；Dockerfile 非 root 用户 + 持久卷；主 deployment 有探针与资源限制。

### 补充优化（按收益排序）
1. **根 `tests/conftest.py` 一个 autouse fixture**：重定向 `routes._store/_knowledge_store` 到 tmp SQLite + `set_feedback_path(tmp_path)` + `reset_workflow_cache()` + 还原 `_workspace_root` —— 单文件修复 11-17 项，套件变隔离且与顺序无关（ROI 最高）。
2. **修 `test_health_endpoint`**（接受 `ok|degraded` 或 monkeypatch 一个 key）+ `test_workflow_produces_answer` 包 `MockLLMService.as_default()` —— 这就是当前 CI 红绿的分水岭。
3. **eval 全离线化**：所有数据集样本补 `mock_llm_response`/`bypass_llm`，runner 改用 context-manager 模式而非直接改 `_llm_service`。
4. **修 `run_all.py`**：RAG 数据集路径改正、失败时 `return 1`（可作 CI 门槛）、结果写进 gitignored 的 `data/eval/`。
5. **调参留出集**：`tune.py` 网格搜索前拆分 train/val，最终超参在留出集上评估并双报数字。
6. **修 completion_eval 复位 bug**（runner.py:182 每轮 `reset()`）并补回归测试断言第 2 轮用 response[1]。
7. **CI 加固**：pytest-timeout `--timeout=60`、`pytest-randomly` 暴露顺序依赖、上传覆盖率产物、追加 `agentflow.blueprints.tests` 作业、k8s secret 补 `EMBEDDING_API_KEY`、镜像 pin 版本。
8. 删空洞断言、`tempfile.mktemp` 换 `tmp_path` fixture + 注入 `FakeEmbedder`；Dockerfile HEALTHCHECK 加 `timeout=5`；web-deployment 补资源与 liveness。
