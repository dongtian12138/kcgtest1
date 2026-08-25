# kcgtest1 当前轻量上下文

> 快照时间：2026-08-25T15:07:33Z
> 当前分支：`carts-grasp-full-palm-search-20260825`
> 原始运行产物和哈希优先于本摘要。

## 一句话状态

已按用户指令进入 `SAFE_PAUSE_FOR_TASK_MODEL_REVISION`：旧“带指甲、仅蓝色 PAD、候选先选后检查最低高度”任务模型只保留为离线对照，不再继续；Isaac 从未在本轮启动。

## 六行恢复摘要

- 最初目标：A/B 用同一方法和主要参数，由三块真实有效接触面夹持后离桌、抬升至少 50 mm、保持至少 2 s，且无未授权穿透。
- 当前已完成：官方 GraspGenX A/B 各 5824 条六维提案、91 个掌面描述器、真实手部网格与旧接触语义下对象 A 的 89/91 个掌面角离线级联。
- 当前真实物理结果：没有启动 Isaac，也没有连接器离桌；89 个完整角仅构成旧任务模型的离线对照，不能解释为新任务模型的抓取结果。
- 当前唯一阻塞：用户确认真实任务改为“无指甲、最低高度优先投影/筛选及修正后的接触语义”，旧生产方法已被替代。
- 最近用户指令：立即安全暂停、封存证据、提交并停止；停止原因是 `TASK_MODEL_SUPERSEDED_BY_USER_CONFIRMED_NAIL_FREE_HEIGHT_FIRST_GRASP_DEFINITION`。
- 下一步：不自主继续；等待用户新的总提示词后，从新任务模型定义与几何身份核对开始。

## 暂停边界

- 状态：`PARKED`，里程碑：`SAFE_PAUSE_FOR_TASK_MODEL_REVISION`。
- `hardware_authorized=false`；旧正式动态门关闭；研究型与正式动态均为 false。
- 当前旧模型级联最后完整保存：89/91，最后掌面角 `1.5351111111111113 rad`。
- 旧模型精筛累计 712 条，旧模型任务评价候选 0；这不是抓取失败，也不是三指手机械结构无解。
- 第 90 个掌面角在最近安全检查点后中断，没有写入半角结果。
- 不再启动掌面角、全级联、局部精修、机械臂规划或 Isaac。

## 已封存事实

- 检查点：`artifacts/carts_v2/full_palm_search/offline_A/full_palm_cascade_checkpoint.json`
  - SHA-256：`b7a1c2df12e0d7729831fee613ada3f9c2afb1c5653ed55e79665d40d26d642d`
  - 逻辑状态：12,023,451；真实几何查询：3,099,159；相同状态复用：8,924,292。
- 对象 A 提案：5824 条，SHA-256 `aed02e498d476a79dd7e62c24e2edb6931f33c0862f31f97d298713acc4297d4`。
- 对象 B 提案：5824 条，SHA-256 `0086d1be25fb3f74ef292a43676da5900c0bab881923e38eddcfcedf235caeea`。
- 当前配置 SHA-256：`e3789927e6527609c9850ebe85affd3b4326f081e8585d7173295c01dc56c579`。
- 暂停证据清单：`artifacts/carts_v2/full_palm_search/SAFE_PAUSE_TASK_MODEL_REVISION.json`。
- 可用终端输出与中断 stderr：`artifacts/carts_v2/full_palm_search/SAFE_PAUSE_RUN_IO.txt`。

## 证据边界

- 89/91 表示旧任务模型下完成了 89 个原子角，不是完整 91 角搜索。
- `task_candidates=0` 只属于已完成的旧语义级联，不能外推到无指甲、最低高度优先的新任务模型。
- GraspGenX 分数、离线碰撞查询和程序退出均不证明真实三指接触、离桌、50 mm 或保持 2 s。
- 本轮 Isaac 未启动，研究动态与正式动态均未通过，硬件仍未授权。

## 固定恢复读取顺序

1. `AGENTS.md`
2. `docs/carts_v2/NORTH_STAR_CN.md`
3. 用户下一份总提示词
4. 本文件
5. `docs/carts_v2/DECISIONS_CN.md` 最新部分
6. `artifacts/carts_v2/STATE.json`
7. `artifacts/carts_v2/full_palm_search/SAFE_PAUSE_TASK_MODEL_REVISION.json`
8. `git status`
9. `git log -8`
10. 新任务模型直接相关的几何、接触与高度筛选源码

## 禁止自动恢复

- 未收到新的总提示词前，不启动离线级联、GraspGenX 推理、局部精修或 Isaac。
- 不删除或改写旧候选、89 角检查点和旧模型离线结果。
- 不把旧模型的零候选写成新方法失败或机械结构无解。
