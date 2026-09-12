# 运维手册（Runbook）

> 面向值班/运维。部署步骤见 [deployment.md](deployment.md)。

## 1. 日常巡检（建议每日）

| 项 | 命令/位置 | 关注点 |
|---|---|---|
| 服务健康 | `GET /health` | `status`、`database`、`llm_config` 检查项 |
| 指标 | `GET /metrics` | 请求量、错误率、LLM 调用与耗时 |
| 日志 | `logs/*.log`（JSON） | `level=ERROR/WARNING`、`trace_id` |
| 磁盘 | `data/`、`logs/`、`outputs/` | 向量库与日志增长；阈值 80% |
| 容器 | `docker compose ps` / `kubectl get pods` | 重启次数、就绪状态 |

排查任何问题的第一步：**拿到 `trace_id`**，它可以串起同一请求在所有节点的日志。

## 2. 故障处置表

| 现象 | 可能原因 | 处置 |
|---|---|---|
| 回答是"服务暂时不可用/受限模式" | LLM 不可用或熔断打开 | 查 `llm.log`；确认余额/网络/密钥；熔断会在阈值后自动半开重试 |
| 知识库检索结果为空但库里有文档 | Embedding 或 Qdrant 不可用（已降级为词法检索） | 查 `knowledge.log`；确认 `EMBEDDING_API_KEY` 与 `QDRANT_URL`；降级本身是预期行为 |
| 任务反复重规划直到强制收尾 | 规划/执行持续失败 | 查 `planner.log` + `reflection.log`；关注是否工具参数错误或目标不明确 |
| SSE 流中断/前端一直转圈 | 反向代理缓冲、连接被中断 | 确认 Nginx `proxy_buffering off` 与超时；浏览器 Network 面板看事件流 |
| 容器反复重启（CrashLoopBackOff） | 生产模式缺少必需密钥 | 补齐 `DEEPSEEK_API_KEY` / `EMBEDDING_API_KEY`，或用 `ENFORCE_REQUIRED_ENV=false` 临时降级 |
| 写入文件失败 | 工作区路径越界/权限/磁盘满 | 查 `filesystem_tool.log`；确认工作区根与卷挂载 |
| 数据库锁错误 / 写入变慢 | SQLite 单写 + 并发高 | 降低并发；确认只有 1 个副本在写；必要时迁 Postgres |
| 上传失败 | 超出大小/条目限制 | 对照 `MAX_UPLOAD_BYTES` / `MAX_ZIP_*` 配置 |
| 端口占用 | 旧进程未退出 | Windows 用 `Get-NetTCPConnection -LocalPort 8000` 定位并结束；或改用其他端口 |

## 3. 备份

**必须备份的内容**（顺序停止写入后备份更安全）：

| 数据 | 路径 |
|---|---|
| 数据库（含会话/文档元数据/长期记忆） | `agentflow/database/agentflow.db`（含 `-wal`/`-shm`） |
| 向量索引 | `data/qdrant/` |
| 嵌入缓存 | `data/embedding_cache.db` |
| 知识库原文 | `knowledge_files/` |
| 生成产物（可选） | `outputs/` |
| 配置（含密钥，注意安全） | `.env` |

Docker 部署示例（导出命名卷）：

```bash
docker compose stop app
docker run --rm -v multi_agent_app-data:/data -v "$PWD/backup:/backup" alpine \
  tar czf /backup/data-$(date +%F).tgz -C /data .
docker compose start app
```

非容器部署示例（Windows PowerShell）：

```powershell
Compress-Archive -Path agentflow\database\agentflow.db* , data, knowledge_files `
  -DestinationPath ("backup\omniforge-" + (Get-Date -Format yyyy-MM-dd) + ".zip")
```

## 4. 恢复演练（建议每季度一次）

1. 停止 `app`（避免写入冲突）。
2. 将备份内容还原到对应目录（数据库、`data/qdrant`、`knowledge_files`）。
3. 启动 `app`，检查 `/health` 与日志中的 schema 迁移信息。
4. 验证：历史会话可打开；知识库检索有结果；新建会话可正常对话。
5. 记录演练结论到 [risk-register.md](risk-register.md) 对应条目。

## 5. 事故响应流程

1. **发现**：告警或用户反馈；记录时间、`trace_id`、影响范围。
2. **止血**：优先恢复可用性（回滚镜像、临时关闭高负载功能、切换降级模式）。
3. **定位**：按 `trace_id` 追日志；对照本手册故障表。
4. **修复**：小步变更，附验证证据。
5. **复盘**：24 小时内产出简短复盘（时间线、根因、改进项），改进项进入 [product/backlog.md](product/backlog.md)。

## 6. 成本与配额

- 每个节点都有 token 上限配置（`PLANNER_MAX_TOKENS`、`ANSWER_MAX_TOKENS` 等）；调低可立刻降本。
- 建议按日统计 token 与费用（可从 `llm.log` 的用量记录聚合），并设置预算告警。
- 代码生成是最贵环节（单文件一次调用），必要时为 `CODEGEN_MODEL` 指定更便宜模型。

## 7. 待补齐的运维能力（见 backlog）

- 集中式监控与告警（Prometheus/Grafana + 规则）
- 自动化备份任务与恢复脚本
- 成本看板与日预算告警
