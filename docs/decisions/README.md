# 架构决策记录（ADR）

重大架构或技术选型变更需新增一份 ADR。命名：`ADR-XXXX-<短标题>.md`，编号连续。

| 编号 | 决策 | 状态 |
|---|---|---|
| [ADR-0001](ADR-0001-langgraph-state-graph.md) | 用 LangGraph StateGraph 编排多 Agent 工作流 | 已采纳 |
| [ADR-0002](ADR-0002-sqlite-and-local-qdrant.md) | 单实例存储：SQLite + 本地 Qdrant | 已采纳 |
| [ADR-0003](ADR-0003-pluginnable-tool-registry.md) | 插件化工具注册表作为工具元数据唯一来源 | 已采纳 |
| [ADR-0004](ADR-0004-graceful-degradation.md) | 统一降级策略：熔断 + 规则兜底 + 终止上限 | 已采纳 |

## 模板

```markdown
# ADR-XXXX：<决策标题>

- 状态：提议 / 已采纳 / 已废弃 / 被替代（被 ADR-YYYY 替代）
- 日期：YYYY-MM-DD
- 决策人：

## 背景
为什么现在必须做这个决策？（约束、问题、现状）

## 决策
我们决定做什么。

## 理由
关键权衡与证据（数据、实验、评审结论）。

## 后果
正面影响 / 负面影响 / 需要跟进的事项。

## 备选方案
考虑过但未采用的方案，以及未采用的原因。
```
