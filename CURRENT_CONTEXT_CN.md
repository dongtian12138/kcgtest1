# kcgtest1 当前轻量上下文

> 快照时间：2026-08-25T01:45:00Z
> 当前分支：`carts-grasp-graspgenx-route1-20260824`
> 原始产物和哈希优先于本摘要。

## 一句话状态

官方 GraspGenX 和 5 个 KCG 描述器已在 GPU 上真实运行；修正 OBB 提案遗漏的描述器横向工作区偏移后，对象 A 仍无三指闭合候选，对象 B 有 2 个三指闭合候选，但两项在第一指闭合时已开始穿桌、第三指阶段最深约 4.46 mm，因此当前没有可执行候选，Isaac 未启动。

## 六行恢复摘要

- 最初目标：A/B 用同一方法和主要参数，由三块真实指腹接触后离桌、抬升至少 50 mm、保持至少 2 s，且无未授权穿透。
- 当前已完成：官方正对照、5 个对象无关描述器、双对象完整网格六维生成、描述器/坐标链审计、OBB 横向接口修正和完整控制步闭合扫掠。
- 当前真实物理结果：仅离线运动学与网格距离；A 为 `256→场景粗筛100→三指闭合0`，B 为 `256→116→闭合2→桌面安全0`；连接器未在 Isaac 中运动。
- 当前唯一主要阻塞：当前固定生成预算内，没有同时满足三块允许指腹接触和完整顺序闭合不扫桌的姿态。
- 最近用户指令：`METHOD_REVISION + NEW_AUTONOMOUS_EXECUTION_WINDOW + UNATTENDED_EXECUTION`，不是硬件授权，也未降低碰撞、力量、50 mm、2 s 或在线真值边界。
- 下一步：完成定向复核、双监督、提交和事实报告；不以临时局部优化器改变本轮已登记方法。

## 当前权威事实

- 状态：`IMPLEMENTING`
- 里程碑：`GRASPGENX_ROUTE1_OBB_INTERFACE_FIX_CLOSEOUT`
- 窗口：`2026-08-24T14:46:08Z` 至 `2026-08-25T02:46:08Z`，不自动延长。
- 官方代码：commit `b9429097728cb1c430dd78b92edf17ba318aad03`，Apache-2.0。
- 运行时：Python `3.10.12`、PyTorch `2.7.0+cu128`、RTX 5070 Ti `sm_120`；checkpoint 树 SHA-256 `f301fbd…`。
- 官方正对照：`robotiq_3f + banana` 返回 20/20 个有限位姿；只证明模型与 GPU 链联通。
- KCG 描述器：5 个；manifest SHA-256 `694bc748…`；A/B 同集合、种子 `20260824`、预算和推理参数。
- 输入：完整登记对象网格按面积固定采样 8192 点；允许/禁止面只由下游 PAD 语义门使用。
- 生成：每对象文件 640 条，六维去重和固定预算后各 256 条；五个描述器均出现，斜向、侧向、roll、pitch 和径向距离均有变化。覆盖是诊断，不是抓取证据。
- 模型分数：`grasp_threshold=-1.0`，不设绝对分数物理硬门；分数只用于每描述器固定 128 项的预算优先级，最终物理排序中仅作后置并列信息。
- OBB 接口：只对 `branch=obb` 在生成器坐标系补偿 open sweep box 的负 x/y 偏移；diffusion 与 z 均不改，A/B 同规则。
- 对象 A：640→256→场景粗筛 100→三指闭合 0；任务、IK、机械臂路径、Isaac 未到达。
- 对象 B：640→256→场景粗筛 116→三指闭合 2；两项均在 `FINGER_1_CLOSURE_0164` 由 `f1Link3` 首次越过容差约 `-0.000032455 m`，全路径最差在 `FINGER_3_CLOSURE_0267` 由 `f3Link3` 达到 `-0.004459948 m`，完整扫掠通过 0。
- B 两条闭合见证的最大单周期独立关节增量为 `0.001498176 rad`，未超过冻结的 `0.0015 rad`。
- 12 N 名义任务门、鲁棒任务门、IK、Top-3 与 Isaac：均未到达；不能写成评价失败或动态失败。
- 研究动态：`false`；正式动态：`false`；旧正式门关闭；`hardware_authorized=false`。

## 冻结边界

- 对象身份、网格、质量、质心、惯量、摩擦和误差范围不变。
- 12 N 仿真研究操作上限不变；旧 8 N 只作基线，不扫描力量。
- `3 rad/s`、`0.18 rad/s`、`1/120 s`、`0.0015 rad/周期`、50 mm、2 s 不变。
- 不使用磁吸、隐藏固定、物体位姿写入或在线对象/接触/PhysX 真值。
- 旧轴对称生成器与 H102 只作基线；不修改旧连续碰撞器。

## 动态入口

候选必须先通过三块允许指腹接触、完整几何路径、12 N 名义任务、机械臂 IK 和接近/抬升路径，才允许研究型 Isaac。当前数量为 0，所以失败关闭；这不是 Isaac 已运行后失败。

## 固定恢复读取顺序

1. `AGENTS.md`
2. `docs/carts_v2/NORTH_STAR_CN.md`
3. `docs/carts_v2/GRASPGENX_ROUTE1_PLAN_CN.md`
4. 本文件
5. `docs/carts_v2/DECISIONS_CN.md` 最新部分
6. `artifacts/carts_v2/STATE.json`
7. `artifacts/carts_v2/graspgenx/INTEGRATION_MANIFEST.json`
8. `git status`
9. `git log -8`
10. 当前唯一阻塞相关源码/产物

## 当前直接证据

- 提案：`artifacts/carts_v2/graspgenx/proposals_obb_xy_fixed_v5/`
- A 离线：`artifacts/carts_v2/graspgenx/offline_obb_xy_fixed_v5_A/`
- B 离线：`artifacts/carts_v2/graspgenx/offline_obb_xy_fixed_v5_B/`
- 描述器：`artifacts/carts_v2/graspgenx/descriptors/descriptor_manifest.json`
- 身份清单：`artifacts/carts_v2/graspgenx/INTEGRATION_MANIFEST.json`

## 动作纪律

- 结论按“物理现象→数据→证据边界”表达；六维覆盖、退出码和测试不冒充抓取。
- Isaac 前重新核对源码、配置、模型、候选、GPU、预检查、物理门和真值隔离。
- 只提交本任务文件；第三方仓库、权重、缓存和逐周期大 trace 不提交。
