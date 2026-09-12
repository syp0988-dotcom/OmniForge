## 变更目的

解决了什么问题？（关联 Issue：`#`）

## 变更内容

-

## 验证方式

- [ ] `python -m pytest -q` 全绿（当前基线：526 passed / 1 skipped）
- [ ] `python -m ruff check agentflow tests scripts` 0 告警
- [ ] 覆盖率不低于基线（如涉及核心模块，已补单测）
- [ ] 前端：`npm run typecheck` / `npm run test` / `npm run build` 通过（如涉及）
- [ ] 手动验证步骤与结果：

## 风险与回滚

- 风险：
- 回滚方式：

## 文档同步

- [ ] 更新了 README / docs（架构、部署、运维）
- [ ] 更新了 CHANGELOG.md
- [ ] 如为架构决策，新增或更新了 `docs/decisions/` 下的 ADR

## 安全相关（如适用）

- [ ] 本 PR 修复了安全项，并附复现与修复后验证
- [ ] 本 PR 未引入新的外部依赖或密钥
