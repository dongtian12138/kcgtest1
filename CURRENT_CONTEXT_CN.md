# kcgtest1 当前轻量上下文

> 快照时间：2026-08-25T00:58:00Z
> 当前分支：`carts-grasp-graspgenx-route1-20260824`
> 本文件只做恢复路由；物理结论以哈希绑定的原始产物为准。

## 一句话状态

官方 GraspGenX、GPU 和 5 个 KCG 手描述器已真实运行；完整登记网格和不设模型分数硬门的 A/B 固定池各保留 256 个六维候选，但完整逐指闭合均为 0 个三指允许接触，任务、IK 和 Isaac 尚未到达。

## 六行恢复摘要

- 最初目标：两个 D38999/J599 对象使用同一方法与主要参数，由三块真实指腹接触后离桌、抬升至少 50 mm、保持至少 2 s，且无未授权穿透。
- 当前已完成：官方正对照、5 个对象无关描述器、双对象完整网格六维生成、坐标/TCP 复核、模型输入张量绑定、真实手逐周期闭合以及过滤前全 256 池重放。
- 当前真实物理结果：只有离线运动学与网格接近计算；A/B 各 `256→闭合0`，连接器没有在 Isaac 中运动。
- 当前唯一主要阻塞：模型抽象工作箱在 A/B 分别覆盖 69/56 个候选，但当前固定池真实闭合最多只有一块指腹接触；尚不能唯一归因于方盒表达、模型域外泛化或剩余描述器语义。
- 最近用户指令：`METHOD_REVISION + NEW_AUTONOMOUS_EXECUTION_WINDOW + UNATTENDED_EXECUTION`，不是硬件授权，也未降低碰撞、力量、50 mm、2 s 或在线真值边界。
- 下一步：只审查描述器规范与原始固定提案池是否还存在系统性接口遗漏；没有真实可执行候选时不启动 Isaac，并在硬截止前保存事实、提交和推送。

## 当前权威事实

- 状态：`IMPLEMENTING`
- 当前里程碑：`GRASPGENX_ROUTE1_FIXED_POOL_CONTACT_GAP_REVIEW`
- 窗口：`GRASPGENX_ROUTE1_12H_UNATTENDED`
- 开始：`2026-08-24T14:46:08Z`
- 硬截止：`2026-08-25T02:46:08Z`，不自动延长。
- 官方 GraspGenX：commit `b9429097728cb1c430dd78b92edf17ba318aad03`，隔离运行时 Python `3.10.12`、PyTorch `2.7.0+cu128`、RTX 5070 Ti `sm_120`。
- 官方正对照：`robotiq_3f + banana` 返回 `20/20` 个有限位姿；官方手在 A/B 同一对象点云上也各返回 `256/256`，只证明模型与输入链联通。
- KCG 描述器：固定 5 个；清单 SHA-256 `694bc748…`；两个对象使用同一集合、种子和推理参数。
- 模型条件绑定：generator/discriminator 均为 `sweep_volume_v2`；五个描述器输入逐项等于 manifest，且得到 5 个不同 SHA-256，排除共享模型覆盖手描述器。
- 生产输入：从完整登记网格按面积、固定种子采样 8192 点；允许/禁止面标签只用于下游真实指腹语义。
- 模型分数：官方规划接口默认 `grasp_threshold=-1.0`；本路线不再用 `0.7` 在物理检查前硬删候选，仍保持每描述器最高 128、合并最多 256。
- 当前提案：A/B 文件各 640 条，六维去重和分层多样性后各 256 条；两对象均覆盖全部 5 个描述器，六维覆盖诊断通过。
- 场景点云粗筛：真实张开手与有限桌面同规则，A `256→100`、B `256→116`；它只是提前拒绝，不是完整安全证明。
- 粗筛后闭合：A 100 项为第一指无接触 98、第二指无接触 2；B 116 项为 109/7；三指幸存均为 0。
- 粗筛前完整池：A 256 项为第一指无接触 220、第二指无接触 24、禁面首触 12；B 为 240、12、4；三指幸存仍均为 0。
- 抽象工作箱交叉表：A 69 项、B 56 项确有对象网格进入 open/half 工作箱；其中仅 24/12 项形成一块允许指腹接触，0 项形成两块或三块。
- 坐标/TCP 独立复核：`object_from_handbase = object_from_graspgenx @ graspgenx_from_handbase` 正确，逆变换误差在数值容差内；工作箱已有占据也排除整体坐标漂离对象。
- 旧 `0.7` 完整网格结果：A `207→闭合0`、B `200→闭合0`，保留为分数硬门消融。
- 旧允许面 ROI 未截断结果：A 547、B 589 个提案仍为闭合 0，保留为输入消融。
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
- 不把工作箱占据、六维覆盖、退出码或静态测试写成抓取成功。

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

- 生产提案：`artifacts/carts_v2/graspgenx/proposals_full_object_score_post_v4/`
- A/B 场景粗筛后：`offline_score_post_v4_A/`、`offline_score_post_v4_B/`
- A/B 粗筛前：`offline_score_post_prefilter_A/`、`offline_score_post_prefilter_B/`
- 5 个描述器：`artifacts/carts_v2/graspgenx/descriptors/descriptor_manifest.json`
- 集成身份：`artifacts/carts_v2/graspgenx/INTEGRATION_MANIFEST.json`

## 动作纪律

- 超过 30 分钟的工作先写任务卡；候选错误回退到上一有效阶段，不提前 PARK。
- Isaac 前重新核对源码、配置、模型、候选、GPU、预检查、物理门和真值隔离。
- 每 45～60 分钟及大里程碑更新本文件、计划、状态和 manifest；只提交本任务路径。
