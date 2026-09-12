# 变更日志

本文件记录所有值得注意的变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 安全（技术评审中列出的高危项已修复）
- **沙箱逃逸（H2）**：仅替换 `open` 内置函数不够——`io.open` / `_io.open` 可读取任意文件
  （已复现）。现将 `io`/`_io`/`codecs`/`fileinput`/`mmap`/`pickle`/`shelve`/`marshal`/`dbm`
  加入导入黑名单，并阻断 `.open`/`.fdopen`/`.FileIO`/`.read_bytes`/`.write_bytes` 属性调用。
- **前端存储型 XSS（H4）**：Markdown 渲染由 `html: true` 改为 `html: false`，助手回复中的
  原始 HTML 一律转义；新增 5 条 vitest 用例（含 `javascript:` 链接、`<script>`、事件属性）。
  渲染配置抽到 `frontend/src/components/markdown/markdown.ts` 以便无 DOM 单测。
- **DocxTool 路径穿越（H3）**：`_resolve` 原先直接放行绝对路径与 `..` 遍历。现与
  `FileSystemTool` 共用新的 `agentflow/tools/path_safety.py`，`execute()` 内强制校验，
  越界路径一律拒绝（补 4 条用例）。
- **`/workspace/set` 任意路径（H1）**：新增 `WORKSPACE_ALLOWED_ROOTS` 白名单（默认项目根 +
  用户主目录 + 系统临时目录），系统目录（`/etc`、`C:\Windows` 等）直接拒绝，避免文件 API
  变成任意读写原语。临时目录必须包含在内：POSIX 上 pytest 的 `tmp_path` 位于 `/tmp`，
  而不是 `$HOME` 之下，漏掉它会让所有使用临时工作区的用例在 Ubuntu runner 上返回 403。

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
- **Python 执行节点默认不执行**：`ALLOW_UNSAFE_PYTHON_TOOL` 的语义被写反——该开关本意是
  "跳过沙箱校验"，节点却在开关为 **False（默认）** 时直接拒绝执行，于是代码执行能力默认
  全部走"blocked"分支，想用还得先关掉沙箱。现已移除该反转门禁，沙箱始终生效。
- **无代码任务被判失败（P1-PY-1）**：没有代码块时 `success` 仍为 False，任务被标记 failed
  并触发重规划；现已视为完成（`status="no_code"`）。同时任务自带 `input.code` 优先于从问题
  文本里解析，并修掉"中文句子被当成 Python 表达式执行"的误判（补 5 条用例）。
- **反思结果匹配塌缩（P1-REF-1）**：按子串匹配时多个结果会全部落到第一个任务上，其余任务
  永不更新。现按 task_id → 精确匹配 → 子串的顺序匹配，且每个任务只消费一次；工作区扫描
  异常改为降级而非冒泡。
- **未知任务状态中断规划（P1-QUEUE-1）**：`TaskQueue.update(status=...)` 遇到未知状态会抛
  `ValueError`，使整轮规划失败；现记录告警并保留原状态。
- **降级错误分类恒为 unknown（P1-ANS-1）**：answer 节点用错误文本重建 `Exception` 再分类，
  必然得到 unknown；新增 `classify_error_message()` 按文本关键字分类，`classify_error()`
  也以此为兜底。
- **后续轮次丢失滚轮摘要（P1-MEM-1）**：记忆窗口先按 20 条截断、之后才做压缩，导致被截掉的
  旧对话永远进不了摘要；且滚动摘要每轮从头重算、不落回状态。现改为"先压缩后截断"，并把
  `rolled_summary` 跨轮携带（补 2 条用例）。
- **重规划丢弃在途任务（P1-PLAN-1）**：非项目路径用新计划**整体替换**任务队列，已完成/在途
  任务随之消失；现与项目路径一致改为合并（补 1 条用例）。
- **limit 参数未校验（P1-API-1）**：`/history`、`/sessions`、`/memory`、`/memory/search`、
  `/executions` 的 `limit` 直接进 SQL，而 SQLite 把负数视为"不限制"；现加 1..N 校验
  （非法值返回 422），并在存储层统一钳制。`/memory` 此前还完全忽略了 `limit`。
- **SQLite 连接泄漏（P1-ASYNC-2）**：`_initialize` 用 `with sqlite3.connect(...)`
  只提交不关闭，每次构造 `SQLiteStore` 都泄漏一个文件句柄（Windows 上还会锁库）；
  现改用 `contextlib.closing`。
- **构建配置缺陷**：hatchling 默认按项目名查找包目录，而导入包是 `agentflow`，本地那个
  `omniforge` 兼容别名目录又被 `.gitignore` 排除——结果是本机 `uv build` 成功、
  **全新克隆后 `uv sync` / `uv build` 必然失败**（README 推荐的安装方式不可用）。
  现显式声明 `[tool.hatch.build.targets.wheel] packages = ["agentflow"]` 并删除该别名目录。
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
  测试数随之从 536 增至 539（新增一条损坏 .docx 的校验用例）。
- **CI 红灯的第二层根因（更致命）**：`python-multipart` 依赖缺失。FastAPI 解析表单/文件上传
  需要该包，而它既不在 `requirements.txt` 也不在 `pyproject.toml` 中——本机虚拟环境恰好装了
  它，所以本地全绿、CI 却在**收集阶段**就报
  `RuntimeError: Form data requires "python-multipart" to be installed.`，
  `test_chat_endpoints` / `test_file_interactions` / `test_metrics` / `test_security` /
  `test_upload_route` / `test_workflow` 六个文件直接无法收集，整个后端 job 失败。
  同时修正两份清单的漂移：`jinja2`（blueprints 渲染必需）只写在 `requirements.txt`、
  `pytest-cov`（CI 的 `--cov` 参数必需）未列入 `pyproject.toml` 的 dev 依赖，会导致
  `uv sync` 用户拿到一份跑不起来的环境。

### 新增
- 项目管理交付物：项目章程、路线图、风险登记册、运维手册、架构文档、ADR、Issue/PR 模板。

### 文档
- README 重写为面向评审的项目主页：新增状态徽章、中文速览、目录导航、核心亮点、
  请求生命周期说明、ADR 决策表、前置条件与安装校验步骤、评测体系（5 套套件 / 332 条样本）
  与 CI 质量门禁说明；截图改为内嵌展示，并修正"截图为占位图"等与事实不符的旧表述。
- README 中的数据统一为仓库实测口径：6 个工具、9 个 Agent、11 个节点、42 个 HTTP 操作、
  571 项测试 / 49 个测试文件。

### 清理
- 删除死代码：composio 中从未使用的 `_ACTION_MAP`、answer 两个恒返回空串的"向后兼容"桩。
- 测试隔离：新增 `tests/conftest.py`，在导入应用前把数据库、知识库原文、生成产物与日志
  重定向到会话级临时目录；运行整套测试后仓库内运行时文件零改动（关闭 P0-CI-1）。
  运行时路径同时开放 `DATABASE_PATH` / `OUTPUTS_DIR` / `KNOWLEDGE_FILES_DIR` / `LOGS_DIR`
  环境变量覆盖，便于部署时挂载数据卷。
- 清理本地冗余：`tmp/`、`_tmptest/`、`batch/`、`batch_test/`、`exec_test/`、`test_dir/`、
  `build-check/`、`.playwright-mcp/`、`.pytest_cache_local/`、`.ruff_cache/`、`.vite/`、
  `.coverage` 等临时目录与缓存，并在 `.gitignore` 中补齐对应规则。

### 修复（可移植性）
- `DocxTool.validate` 此前**只会**调用外部 skill 包（`from office.validate import main`），
  而该包位于本机个人路径 `%USERPROFILE%\.claude\skills\docx\scripts`——干净环境、CI 与容器中
  必然报 "Validation dependency unavailable"，README 承诺的 validate 能力实际不可用。
  现改为内置结构化校验（ZIP 容器 + `[Content_Types].xml` / `word/document.xml` + 全部 XML
  部件格式 + python-docx 可打开），无第三方依赖；仅在发现损坏时才调用可选的外部 skill 做一次
  auto-repair，并回报是否修复。skill 路径同时改为跨平台且可用 `DOCX_SKILL_SCRIPTS` 覆盖。
- 两个环境耦合的测试断言改为断言行为契约：
  `test_symlink_escape_blocked` 原先断言拒绝信息里必须出现目标文件名（各平台文案不同），
  现断言"读取被拒 + 错误非空 + 目标内容未泄漏"；`test_validate_docx` 原先在失败分支断言
  错误信息必须包含 `defusedxml`（只有未安装 skill 的机器才走到该分支），现断言创建出的文档
  必须校验通过，并新增损坏 .docx 必须被判为无效的用例。

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
