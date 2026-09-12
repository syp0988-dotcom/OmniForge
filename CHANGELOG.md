# 变更日志

本文件记录所有值得注意的变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增
- 项目管理交付物：项目章程、路线图、风险登记册、运维手册、架构文档、ADR、Issue/PR 模板。

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
