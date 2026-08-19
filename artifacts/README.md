# artifacts

此目录是运行证据与模型数据区，不是源码目录。绝大多数内容由 Git 忽略，但不得因此
视为可删除缓存。

- 当前状态入口：`agent_control/CURRENT_STATUS_CN.md`
- 当前任务：`agent_control/CURRENT_TASK.md`
- Agent 恢复入口：`agent_control/AUTONOMOUS_RESUME_CN.md`
- 队列：`agent_control/WORK_QUEUE.yaml`
- 主状态：`agent_control/MASTER_STATE.json`
- D38999 证据与模型：`kcg_connector/`

保留、备份和未来空间回收规则见 `../docs/ARTIFACT_RETENTION_CN.md`。冻结模型、失败
证据、旧运行和已交付压缩包禁止仅凭文件名或日期删除。
