# OmniForge 项目章程

| 项 | 内容 |
|---|---|
| 项目名 | OmniForge（仓库 `multi_agent`） |
| 当前版本 | 0.1.0 |
| 技术栈 | Python 3.12 + FastAPI + LangGraph；Vue 3 + TypeScript + Vite |
| 章程版本 | v1.0（2026-09-11） |
| 状态 | 开发中，未发布 |

---

## 1. 背景与要解决的问题

大模型能对话，但"把一个目标做完"需要拆解任务、调用工具、检查结果、必要时自我纠正。
OmniForge 把这件事工程化：**一句话目标 → 意图识别 → 任务规划 → 工具执行 → 结果反思 → 最终回答**，
全过程带会话记忆与可观测性。

当前已具备：多 Agent 编排（LangGraph）、动态任务队列、6 个工具（插件化注册）、
RAG 混合检索（Qdrant + SQLite FTS5 + RRF）、离线评测框架、Docker/K8s 部署产物、CI。

## 2. 项目定位（待确认，决定后续一切取舍）

三种定位的准备重点完全不同，**需要先定一个**：

| 定位 | 成功标准 | 安全/合规要求 |
|---|---|---|
| A. 个人作品 / 求职展示 | 能讲清架构与踩坑，代码与文档自洽 | 可标注"仅本机单用户使用" |
| B. 内部工具（自己或小团队用） | 稳定可用、失败可见、数据不丢 | 需要鉴权、备份、监控 |
| C. 对外产品 | 有人持续使用 | 安全红线必须清零：沙箱、路径、XSS、鉴权、合规 |

> 章程默认按 **A + B 双定位**推进（作品级代码质量 + 内部可用的部署能力）；
> 若目标是 C，必须在路线图 M0 阶段完成 `docs/product/backlog.md` 中全部 P0 安全项。

## 3. 目标（可度量）

1. **可信**：CI 在无密钥环境下可跑通；测试不污染真实数据；高危安全问题为零。
2. **可用**：按 `docs/runbook.md` 能从零部署、能恢复数据；"文档承诺 = 实际实现"。
3. **可讲**：架构、决策、踩坑全部有文档（`docs/architecture.md`、`docs/decisions/`）。
4. **可演进**：每个里程碑有明确退出标准，评测指标可对比（现有 5 套 eval）。

## 4. 非目标（本期明确不做）

- 多租户 / 计费 / 组织权限体系
- 自研模型或微调；模型能力依赖外部 API
- 移动端、原生客户端
- 分布式任务调度（当前为单实例 + SQLite）
- Browser / MCP / Database 工具的完整实现（占位实现已于 2026-09 移除，能力实现排到 P2-TOOL-2）

## 5. 成功标准（验收口径）

| 维度 | 指标 | 目标值 |
|---|---|---|
| 质量门禁 | 单测 | 全绿（基线：572 passed / 1 skipped；CI 在 ubuntu + windows 双平台） |
| 质量门禁 | 覆盖率 | ≥ 当前基线 64%（CI 已按此卡口），核心模块 ≥ 75% |
| 质量门禁 | lint | ruff 0 告警 |
| 交付 | 端到端冒烟 | 部署后可完成"对话 → 知识库 → 生成文件"全链路 |
| 可靠 | 数据恢复 | 从备份恢复演练成功一次 |
| 文档 | 一致性 | README 声明与实现零漂移 |
| 成本 | 可观测 | 每日 token/费用有报表，可设上限 |

## 6. 干系人与职责

| 角色 | 人 | 职责 |
|---|---|---|
| 产品/项目负责人（PM） | 待定 | 范围、排期、风险、验收 |
| 技术负责人 | 待定 | 架构决策（ADR）、代码评审 |
| 安全/代码评审人 | 待定 | 安全项关闭确认 |
| 用户代表 | 待定 | 演示验收、反馈提供 |

> 当前实际为单人多角色，**bus factor = 1** 已登记为风险（见 `docs/risk-register.md`）。

## 7. 约束与假设

- **环境**：Windows 为主要开发环境（CI 同时跑 Ubuntu + Windows）；生产建议 Linux + 容器。
- **依赖**：核心能力依赖外部 API（DeepSeek / DashScope Embedding / Tavily / Composio），需接受网络与额度约束。
- **存储**：SQLite（单写）+ 本地磁盘向量库（`data/qdrant`），决定了单副本部署。
- **产能假设**：以每周 1 个可交付里程碑批次（1–2 天）估算；若为业余投入需重新评估排期。

## 8. 待决策项（需要项目负责人拍板）

| # | 决策 | 影响 | 建议 |
|---|---|---|---|
| D1 | 项目定位（A / B / C） | 决定安全投入与里程碑顺序 | 先定 A+B，C 作为后续 |
| D2 | 许可证 `LICENSE` | 他人可否使用/商用 | 开源用 MIT/Apache-2.0；闭源则不加 LICENSE |
| D3 | 空壳工具（Browser / MCP / Database） | 文档可信度与用户预期 | ✅ 已决策并执行（2026-09 从 README 下线，实现需求记入 backlog P2-TOOL-2） |
| D4 | 是否需要多副本/多租户 | 影响数据库与存储选型 | 本期保持单实例，横向扩展另立项目 |
| D5 | 是否对外开放访问 | 决定鉴权与合规要求 | 默认仅内网访问，开放前完成 P0 安全项 |

## 9. 相关文档

- 架构与数据流：[docs/architecture.md](docs/architecture.md)
- 部署与回滚：[docs/deployment.md](docs/deployment.md)
- 运维手册：[docs/runbook.md](docs/runbook.md)
- 风险登记册：[docs/risk-register.md](docs/risk-register.md)
- 承诺 vs 实现差异：[docs/product/status.md](docs/product/status.md)
- 待办与优先级：[docs/product/backlog.md](docs/product/backlog.md)
- 里程碑：[ROADMAP.md](ROADMAP.md)
- 技术评审报告：[docs/reviews/2026-08-14-full-review.md](docs/reviews/2026-08-14-full-review.md)
