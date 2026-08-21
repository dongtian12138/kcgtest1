# 当前计划入口

此文件保留在仓库根目录，是旧证据完整性工具要求的兼容路径，不再承载独立计划。

普通工作先读取：

- `CURRENT_CONTEXT_CN.md`

它给出当前任务、证据边界和最小读取路线。历史授权审计、状态冲突、正式 gate 变更或
动态启动时，再按任务 ID 定向读取 `WORK_QUEUE.yaml`、`CURRENT_TASK.md`、
`AUTONOMOUS_RESUME_CN.md`、`MASTER_STATE.json` 及相应原始台账；不要默认整批载入。

2026-08-19 清理前的旧 `TASK-R12-MULTILAYER` 计划不再保留活动路径副本；需要审计时用：

```bash
git show pre-contract-cleanup-20260821:legacy_archive/outdated_docs/TASK_R12_MULTILAYER_PLAN_legacy.md
```

该历史文本仅供追溯，不构成运行授权。
