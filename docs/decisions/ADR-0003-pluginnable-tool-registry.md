# ADR-0003：插件化工具注册表作为工具元数据的唯一来源

- 状态：已采纳
- 日期：2026-07
- 相关代码：`agentflow/tools/base.py`、`agentflow/tools/registry.py`、`agentflow/graph/workflow.py`

## 背景

工具数量持续增长（文件、搜索、代码、Git、DOCX、Composio…），早期实现把工具的动作清单、
LLM 函数 schema、路由映射写死在多个文件里，新增一个动作要改 5–7 处，容易漂移：
写出 schema 却没有对应实现，或实现了却不会被规划器调用。

## 决策

工具元数据**只在一处声明**：`BaseTool.actions()`。其余全部派生：

- LLM 函数 schema → `tool_schemas()` 由 actions 生成；
- 能力清单 → `capabilities()` = `{tool}.{action}`；
- 图节点路由 → `routing_node()`（默认 `tool_executor`，搜索走 `query_rewriter`，代码走 `python`）；
- 工具注册 → 扫描 `tools/` 目录自动发现，无需维护注册表清单。

## 理由

1. 新增工具 = 写一个类 + 实现 `actions()`/`execute()`，不会漏改 schema 或路由。
2. 规划器看到的可用动作永远与真实实现一致（接口占位工具会被标记并排除在 schema 之外）。
3. 动态路由让路由函数不再硬编码工具名单——执行器与工作流都从注册表查询。

## 后果

- 正面：扩展成本低、声明与实现难以漂移、单个工具可独立单测。
- 负面：`actions()` 必须写准确，否则会误导 LLM（参数描述错误会导致调用失败）；
  参数修复机制（`graph/workflow.py`）用于兜底这一类失败。
- 负面：自动发现依赖目录约定（工具类必须定义在 `tools/*.py` 且非私有文件）。

## 备选方案

- **静态注册表 + 手写 schema**：一致性与维护成本都更差，已废弃。
- **由 LLM 直接生成工具描述**：不可控，无法保证与实现一致。
