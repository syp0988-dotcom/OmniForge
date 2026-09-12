# 变更日志

本文件记录所有值得注意的变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 变更（仓库清理）
- 移除三个空壳工具（Browser / MCP / Database）——它们的所有 action 均返回"未实现"，
  保留只会造成"文档承诺 ≠ 实际能力"；相关能力改由 `docs/product/backlog.md` 跟踪。
- 删除死代码：`agentflow/agents/project_structure_planner/`（实现完整但从未接入工作流）。
- 移除历史遗留的重复部署配置 `agentflow/docker/`（根目录 Dockerfile + docker-compose.yml 为准）。
- 删除孤儿评测数据集 `eval_data_new.jsonl`；修正 `run_all.py` 中 RAG 评测脚本的数据集路径
  （原先指向不存在的 `eval_dataset.jsonl`，会导致全套评测在第 5 步报错）。
- 向量索引 `data/qdrant/` 与评测输出 `*_results.json` 移出版本库（改为运行时生成 + 忽略规则）。
- `agentflow/blueprints/tests.py` 移入 `tests/test_blueprints_integration.py`，其 22 个用例
  现由 pytest/CI 收集（此前躺在包内、既不进 CI 也会被打进安装包）。

### 修复
- 子进程输出编码：`python` / `git` / `docx` 三个工具以 `text=True` 读取子进程输出时未指定
  编码，Windows 下按 GBK 解码 UTF-8 字节会抛 `UnicodeDecodeError`（子进程输出含中文即触发，
  且发生在后台线程中，表现为"任务无响应"）。现统一按 UTF-8 解码并容错，Python 沙箱同时
  注入 `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1`，跨平台行为一致。
- **CI 长期红灯的根因**：4 个测试隐含依赖开发者本机的 `.env` 密钥，在无密钥的 CI 环境
  必然失败（`Backend tests` 在 ubuntu / windows 双平台同时 red）。修复内容：
  - 意图分析中"只记一次"的 `_embedding_unavailable_logged` 同时抑制了日志与结构化错误
    上报，导致行为随请求顺序变化；现改为日志进程级节流、错误逐请求记录。
  - 上传路由测试改为注入本地确定性 embedder，不再依赖 `EMBEDDING_API_KEY`，也不再发起
    真实嵌入 API 调用（此前既慢又依赖网络）。
  - `/health` 测试原先断言 `status == "ok"`（仅在有密钥的机器上成立），现改为断言接口契约，
    并分别覆盖"已配置 / 未配置密钥"两种状态。
  测试数随之从 536 增至 538。

### 新增
- 项目管理交付物：项目章程、路线图、风险登记册、运维手册、架构文档、ADR、Issue/PR 模板。

### 文档
- README 重写为面向评审的项目主页：新增状态徽章、中文速览、目录导航、核心亮点、
  请求生命周期说明、ADR 决策表、前置条件与安装校验步骤、评测体系（5 套套件 / 332 条样本）
  与 CI 质量门禁说明；截图改为内嵌展示，并修正"截图为占位图"等与事实不符的旧表述。
- README 中的数据统一为仓库实测口径：6 个工具、9 个 Agent、11 个节点、42 个 HTTP 操作、
  538 项测试 / 48 个测试文件。

### 待办
- 安全与可信修复（详见 [docs/product/backlog.md](docs/product/backlog.md) P0 项）。

## [0.1.0] - 2026-08-13

首个可用版本：完成多 Agent 编排、RAG 检索、工具系统、评测框架与生产部署产物。

### 新增
- **多 Agent 编排**：基于 LangGraph 的目标驱动工作流（意图分析 → 规划 → 工具执行 → 反思 → 回答 → 记忆）。
- **动态任务队列规划**：蓝图 → 模板 → 函数调用 → JSON 四级降级；任务生命周期与优先级调度。
- **工具系统**：插件化注册表（自动发现 + 动态 schema + 动态路由），含文件系统、搜索、Python、Git、DOCX、Composio 等工具。
- **RAG 知识库**：10+ 格式解析、结构感知分块、Qwen 向量嵌入、Qdrant 向量库 + SQLite FTS5 词法检索、RRF 融合。
- **会话记忆**：槽位填充、指代消解、续问改写、分层记忆压缩、跨会话长期记忆。
- **评测框架**：意图 / 规划 / 工具 / 完成度 / RAG 五套离线评测与指标（recall@k、NDCG、MRR 等）。
- **运行时反馈闭环**：聊天请求的失败与完成样本自动落盘，可导出回流评测集。
- **生产部署**：多阶段 Dockerfile（非 root + 健康检查）、docker-compose（API + Nginx）、K8s 清单（ConfigMap/PVC/Ingress）、HTTPS 配置与证书说明。
- **可观测性**：JSON 结构化日志 + trace_id 关联 + 按天轮转；`/health`、`/metrics`。
- **可靠性机制**：终止策略集中化、LLM 熔断与指数退避、降级模式、工具参数自动修复、Qdrant 故障降级为词法检索。
- **代码生成质量**：设计先行（项目设计简报）+ 全文件上下文注入 + 生成后语法自检与重试；可选更强代码模型。

### 变更
- 意图识别：embedding 多锚点快路径 + LLM 兜底；失败后按冷却期自动重试。
- 规划器：移除全部硬编码目标特判（贪吃蛇、DOCX 报告），统一走通用规划链路。
- 工具调用：并行执行补齐参数修复与事件追踪；中文动作名统一翻译。
- API 层按领域拆分（chat / knowledge / sessions / files / models / memory / executions / system）。

### 修复
- 前端 AI 回复无法渲染（Markdown 组件异步 setup 缺少 Suspense）与流式传输兜底。
- Markdown frontmatter 剥离错误、反思器纠错重试缺口、评测指标函数命名问题。
- 文件系统路径穿越、NUL/ADS 注入等安全问题（部分），会话状态丢失、LLM 误判完成等历史缺陷。

### 已知问题
- 技术评审确认尚存高危项：Python 沙箱逃逸、DocxTool 路径穿越、`/workspace/set` 任意路径读写、前端存储型 XSS（见评审报告与 backlog）。
- Browser / MCP / Database 工具为接口占位；README 声明与实现存在漂移。
