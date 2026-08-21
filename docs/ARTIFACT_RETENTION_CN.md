# artifacts 证据保留规则

2026-08-19 盘点时，`artifacts/` 约 18 GB、16826 个文件。这里混有不可替代的动态
失败证据和可再生成的大型轨迹，因此不能用“目录很大”作为批量删除依据。

## 永久保留且原路径只读

- `artifacts/agent_control/` 的控制面、台账和任务历史。
- 所有正式/诊断 Isaac 运行的计划、日志、轨迹、分析和报告。
- 高精细基线、否决候选、三种冻结表示及 `MODEL_MAPPING.json`。
- 已交付 ZIP、内部 manifest、checksum 和回放快照。

清理前抽查摘要：

```text
5eb9ad82940e58a1592b6a66fd824c480ba24268cb1c20bcc84de653bb12c995  keyed_v3_physical_r12/candidates/r12_candidate_02/r12_candidate_02.usda
d41477ee18052662904212444b907607874a8c6c27399d3d344e44ee4fd18d67  TASK-R12-006B/candidate/task_r12_006b_local_candidate_01.usda
69fe6dc3ca9caace8bb26cd0cfad68c0eb84111f09697da6068cd91802d65c0a  D38999_VISUAL_COMPLETE_V1.usda
94b9cea0a7bb1e4d4a7c6583819abe1c722e252ae45012ba78f1a396b0a5ab85  D38999_LOCAL_CONTACT_REFERENCE_V1.usda
a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783  MODEL_MAPPING.json
```

完整绝对路径和清理前基线见
`artifacts/agent_control/tasks/PROJECT-CLEANUP-20260819/CLEANUP_PLAN_CN.md`。

## 可在未来做的空间回收

只有同时满足下列条件，才可把大型原始轨迹列为删除候选：

1. 已有不可变摘要报告能回答该运行的验收问题；
2. 原始文件已进入带 manifest/checksum 的独立备份；
3. 所有引用该路径的台账和回放程序都已审计；
4. 用户对精确路径清单再次确认；
5. 使用可恢复方式处理，并记录释放空间和恢复位置。

本轮清理不执行这类删除。

## 可直接重建项与本机环境边界

根目录的 `build/`、`install/`、`log/`、`.pytest_cache/`、`__pycache__/` 和
`.venv/` 已由 Git 忽略。Python/pytest 缓存可直接删除并自动重建；`.venv/` 与
`build/install/log` 可能影响本机开发环境或 ROS 工作区，未经单独审计不批量删除。

## J599 独立证据归档

`j599_25_35_standard_interface_v1/evidence/` 的接受摘要和两组接受报告继续展开保留。
原始接触流、轨迹、临时物理场景和中间运行已进入
`evidence_archive/J599_25_35_FULL_EVIDENCE_20260820.tar.gz`；压缩包自身哈希、原始
54 个文件逐文件哈希和恢复命令位于同目录。该归档只减少重复展开数据，不改变报告
结论，也不能把 connector-only 仿真通过升级为机器人装配或真实硬件通过。
