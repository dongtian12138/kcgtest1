# kcgtest1 当前轻量上下文

> 快照时间：2026-08-24T23:45:19Z
> 当前分支：`carts-grasp-graspgenx-route1-20260824`
> 本文件只做恢复路由；物理结论以哈希绑定的原始产物为准。

## 一句话状态

官方 GraspGenX、GPU 和 5 个 KCG 手描述器已真实运行；纠正“点云可见性误作硬门”和“25 点闭合采样过粗”后，A/B 的完整六维池分别有 256/231 个候选，但在由冻结控制参数派生的 `0.0015 rad` 离线关节步长下均没有三指允许接触候选，因此任务、IK 和 Isaac 尚未到达。

## 六行恢复摘要

- 最初目标：两个 D38999/J599 对象使用同一方法与主要参数，由三块真实指腹接触后离桌、抬升至少 50 mm、保持至少 2 s，且无未授权穿透。
- 当前已完成：官方正对照、5 个对象无关描述器、双对象六维生成、坐标/TCP 独立复核、真实张开手—有限桌面粗筛，以及与动态控制步长一致的顺序闭合预测。
- 当前真实物理结果：只有离线运动学与网格接近计算；A `256→86→闭合0`、B `231→34→闭合0`，而且过滤前全池重放也都是闭合 0；连接器没有在 Isaac 中运动。
- 当前唯一主要阻塞：这批固定官方提案在真实 KCG 手闭合范围内不能形成三块允许指腹接触；尚需区分每描述器 Top-128 截断是否漏掉较低分但可接触的原始提案。
- 最近用户指令：`METHOD_REVISION + NEW_AUTONOMOUS_EXECUTION_WINDOW + UNATTENDED_EXECUTION`，不是硬件授权，也未降低碰撞、力量、50 mm、2 s 或在线真值边界。
- 下一步：只读检查阈值后的全部 A/B 原始提案；有闭合幸存者则审查固定预算语义，没有则转入描述器/模型几何适配的事实收口。

## 当前权威事实

- 状态：`IMPLEMENTING`
- 当前里程碑：`GRASPGENX_ROUTE1_RAW_UNTRUNCATED_CLOSURE_DIAGNOSTIC`
- 窗口：`GRASPGENX_ROUTE1_12H_UNATTENDED`
- 开始：`2026-08-24T14:46:08Z`
- 硬截止：`2026-08-25T02:46:08Z`，不自动延长。
- 官方 GraspGenX：commit `b9429097728cb1c430dd78b92edf17ba318aad03`，隔离运行时 Python `3.10.12`、PyTorch `2.7.0+cu128`、RTX 5070 Ti `sm_120`。
- 官方正对照：`robotiq_3f + banana` 返回 `20/20` 个有限位姿；官方手在 A/B 同一对象点云上也各返回 `256/256`，只证明模型与输入链联通。
- KCG 描述器：固定 5 个；清单 SHA-256 `694bc748…`；两个对象使用同一集合、种子和推理参数。
- 生成顺序修正：官方 open/half 点云可见性仅作审计；每描述器按 `(-score, raw_index)` 保留最高 128，再做六维去重和最多 256 个合并。
- 当前提案：A 文件 288 条、适配去重后 256 条，描述器数 `12/31/114/16/83`；B 文件及适配后 231 条，描述器数 `4/2/128/0/97`。两对象六维覆盖诊断均通过。
- 场景点云粗筛：真实张开手与有限桌面同规则，A `256→86`、B `231→34`；它只是提前拒绝，不是完整安全证明。
- 闭合采样修正：旧 25 点意味着每步约 `0.030–0.040 rad`；当前按 `0.18 rad/s × 1/120 s = 0.0015 rad` 派生 486–644 等实际区间数，不能跳过一个真实控制周期。
- 当前离线结果：A 的 86 个粗筛保留候选中 82 个第一指无有效接触、2 个第二指无有效接触、2 个先碰禁抓面；B 为 33 个第一指无有效接触、1 个第二指无有效接触。
- 过滤前全池检查：A 256、B 231 的三指闭合幸存数也均为 0，因此当前零候选不是 scene-PC 粗筛误删造成。
- 指腹距离诊断：A 最接近任意面 `2.384 µm` 但属于禁抓面，最近允许面约 `15.183 mm`；B 相应为 `32.481 µm` 禁抓面与约 `12.218 mm` 允许面。
- 坐标/TCP 独立复核：`object_from_handbase = object_from_graspgenx @ graspgenx_from_handbase` 正确，A/B 全部 487 条变换闭合误差最大约 `55/63 nm`；当前无接触分布不是 TCP 重复或矩阵方向错误。
- 已知语义隐患：描述器输出的 `base_rotation` 含平移，而官方字段语义是纯旋转；当前 sweep-only API 不读取它，未影响现有证据，后续若改用官方 mesh/viewer 必须拆分后重绑定。
- 当前配置 SHA-256：`84eb806a5ef9aaabb5e40e97cc14b4d431afb0adeca9be9e6be12eb10ef8c18d`。
- 当前 A/B result SHA-256：`01a32e46…`、`a0f27246…`；全池闭合诊断 SHA-256：`d2345ab1…`。
- 12 N 名义任务门：未到达，不是评价后失败。
- IK/机械臂路径：未到达。
- Isaac：未启动；没有高度、保持、滑移或接触数据。
- 研究动态：`false`；正式动态：`false`；旧正式门关闭；真实硬件：`hardware_authorized=false`。

## 冻结边界

- 对象身份、真实登记网格、质量、质心、惯量、摩擦和误差范围不变。
- 主研究力量上限保持 `12 N`；旧 `8 N` 只作消融，不扫描增力。
- `3 rad/s` 安全限速、`0.18 rad/s` 手指目标速度、`1/120 s`、`0.0015 rad/周期`、50 mm 和 2 s 不变。
- 在线控制不得读取对象真实位姿、接触名称、精确接触点/法向、指腹标签或 PhysX 真值。
- 旧轴对称生成器、旧候选编号和 H102 只作基线；不修改旧大型连续碰撞器。
- 不把点云粗筛、覆盖图、退出码或静态测试写成抓取成功。

## 动态入口

只有候选同时通过完整几何路径、三块允许指腹名义接触、12 N 名义任务、机械臂 IK 与接近/抬升路径，才允许进入研究型 Isaac。当前候选数为 0，所以动态失败关闭；这不是 Isaac 运行失败。

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

- 生产提案：`artifacts/carts_v2/graspgenx/proposals_score_per_descriptor_v2/`
- A/B 当前离线：`artifacts/carts_v2/graspgenx/offline_control_step_bounded_A/`、`offline_control_step_bounded_B/`
- 可见性根因：`artifacts/carts_v2/graspgenx/visibility_root_cause_diagnostic.json`
- 指腹距离：`artifacts/carts_v2/graspgenx/closure_no_contact_diagnostic.json`
- 过滤前后全池闭合：`artifacts/carts_v2/graspgenx/prefilter_closure_stratified_diagnostic.json`
- 集成身份：`artifacts/carts_v2/graspgenx/INTEGRATION_MANIFEST.json`

## 动作纪律

- 超过 30 分钟的工作先写任务卡；候选错误回退到上一有效阶段，不提前 PARK。
- Isaac 前重新核对源码、配置、模型、候选、GPU、预检查、物理门和真值隔离。
- 每 45～60 分钟及大里程碑更新本文件、计划、状态和 manifest；只提交本任务路径。
