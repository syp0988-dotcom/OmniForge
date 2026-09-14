# 贡献指南

## 环境要求

| 组件 | 版本 |
|---|---|
| Python | 3.12+ |
| Node.js | 20+（前端） |
| 包管理 | `uv`（推荐）或 `pip` |

## 快速开始

```bash
# 后端
cp .env.example .env        # 填写 DEEPSEEK_API_KEY 与 EMBEDDING_API_KEY
uv sync --dev
uv run uvicorn agentflow.app.main:app --reload --port 8000

# 前端
cd frontend && npm install && npm run dev
```

> Windows 上也可使用 `python scripts/run_backend.py` 与 `python scripts/run_frontend.py` 以分离进程方式启动。

## 分支与提交

- 分支命名：`codex/<主题>`（例如 `codex/fix-sandbox-escape`）；修复类也可用 `fix/<主题>`。
- 提交信息：使用中文祈使句，格式建议 `类型：简述`，类型取 `新增 / 修复 / 优化 / 文档 / 测试 / 重构 / 构建`。
  示例：`修复：Python 沙箱允许通过 io 模块打开文件`。
- 一次提交只做一件事；重构与行为变更尽量分开，便于回滚。

## 质量门禁（Definition of Done）

提交前必须全部满足：

1. `python -m pytest -q` 全绿（当前基线：572 passed / 1 skipped）
2. `python -m ruff check agentflow tests scripts` 0 告警
3. 覆盖率不低于 64%（CI 用 `--cov-fail-under=64` 卡口，当前约 66%）；核心模块的新代码建议自带单测
4. 行为变更同步更新文档：`README.md`、`docs/architecture.md`、`CHANGELOG.md`
5. 涉及部署/配置的变更，同步更新 `docs/deployment.md` 与 `.env.example`

常用命令：

```bash
python -m pytest -q --cov=agentflow --cov-report=term   # 覆盖率
python -m pytest tests/test_workflow_routing.py -q      # 单文件
ruff check agentflow tests scripts                      # 静态检查
```

## 代码规范

- 类型注解优先；公共函数写清参数与返回；异常不要静默吞掉（记录日志或进入统一错误通道）。
- Agent 必须实现 `AgentProtocol.run(state)`，并且**不得就地修改输入**（返回新的/更新后的 state）。
- 新增工具：继承 `BaseTool`，实现 `actions()` 与 `execute()`；注册走自动发现，无需改注册表。
- 保留既有中文注释风格；新增注释解释"为什么"而不是"做了什么"。
- 不要提交：`.env`、`logs/`、`outputs/`、私有数据、模型密钥。

## 提交 Pull Request

1. 确认上面的质量门禁全部通过。
2. 按 [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) 填写：变更目的、验证方式、风险与回滚。
3. 涉及安全修复的 PR，请附复现步骤与修复后验证结果。
4. 涉及架构决策的变更，请新增/更新 `docs/decisions/` 下的 ADR。

## 报告问题

使用 [.github/ISSUE_TEMPLATE/](.github/ISSUE_TEMPLATE) 中的模板提交，
并尽量附上：复现步骤、期望与实际结果、`trace_id`、相关日志片段。
