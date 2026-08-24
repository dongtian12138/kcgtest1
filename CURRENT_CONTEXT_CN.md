# kcgtest1 当前轻量上下文

> 快照时间：2026-08-24T13:08:26Z
> 当前分支：`carts-grasp-v2-rebuild-20260823`
> 本文件只做恢复路由；物理结论以 V2 原始运行产物为准。

## 一句话状态

最终同源码、同配置离线重算没有找到可执行候选：对象 A 的 69 个闭合预测幸存种子全部在桌面快筛中被拒绝，对象 B 有 6 个路径幸存者但在 8 N 基线和一次固定 12 N 只读敏感性中均未达到任务载荷门；因此没有从本次新结果启动 Isaac，项目按用户停止条件 6 收口。

## 六行恢复摘要

- 总目标：真实允许表面候选 → 任务载荷鲁棒 Top-3 → Isaac 三指接触、离桌、抬升 50 mm、保持至少 2 s；两个对象同算法同主要参数。
- 当前里程碑：`V2_NO_FEASIBLE_CANDIDATE_ANALYSIS_COMPLETE`；状态 `PARKED`，原因是预先规定的候选空间内没有安全且任务可行的候选，不是 Isaac 或时间耗尽。
- 已经完成：两个对象各 384 个固定原始种子完成闭合预测和完整顺序扫掠；旧 `candidate_11` 端点净空 `+0.705 mm`、完整路径最小 `−1.838 mm`，已永久失效。
- 当前物理/算法阻塞：A 的闭合预测幸存者中，66 个在预抓/端点碰桌、3 个在闭合中途扫桌；B 的 6 个路径安全候选在 8 N 和固定 12 N 敏感性下最差余量均小于 1，限制原因仍未区分。
- 下一步最简单动作：本执行窗口不再改代码或启动 Isaac；若用户以后明确重开方法设计，应先区分 A 的手掌/指系几何可达性，以及 B 的力量上限、三接触几何与任务模型约束，而不是调增益或候选编号。
- 绝对禁止跑偏方向：续修 H102、对象专用坐标/阈值、磁吸/隐藏固定、在线读取对象/接触/PhysX 真值、写物体位姿、把退出 0 冒充物理成功。

## 当前权威字段

- 状态：`PARKED`
- 当前里程碑：`V2_NO_FEASIBLE_CANDIDATE_ANALYSIS_COMPLETE`
- V2 正式候选：空
- V2 研究动态门：`allowed=false`；必须先由新方法产生至少一个 `executable_candidate`
- 旧正式动态门：`dynamic_launch_allowed=false`（保持，不由 V2 研究线改写）
- 正式动态：`FORMAL_DYNAMIC_PASS=false`
- 研究型动态：`RESEARCH_DYNAMIC_PASS=false`
- 真实硬件：`hardware_authorized=false`
- 最终离线重算：A 为 384 → 69 闭合 → 0 路径安全 → 0 可执行；B 为 384 → 109 闭合 → 6 路径安全 → 0 个 8 N 任务合格 → 0 可执行。B 的固定 12 N 只读敏感性最佳最差余量为 0.51175，仍未达到 1。
- Isaac：没有从本次新离线结果启动新的运行；历史 `candidate_11` 第一指运行及其 4.1243 rad/s 急停只保留为旧端点方法反例，不再是当前可执行路线。
- 引擎：run16 预检被严格接受；run17 GPU 配对峰值 1/7644 低于 8192/16384，容量警告和 PhysX Error 均为 0，因此本次失败不是 GPU 容量不足
- 历史 Recovery 2 收口时全回归：1506 项通过、0 失败、982.86 s；本次扫掠/selector 修改后只跑 4 个定向回归，没有重跑全量，测试通过不改变动态失败状态。
- run16/run17 小型 `evaluation.json` 纳入本次收口；4.72/5.54 MB 逐步 trace 只留本地，SHA-256 与字节数登记在 `STATE.json`
- 完整扫掠实现提交 `4248187`、可执行/诊断分流提交 `dadab16` 已普通推送到远端同名分支；最终两对象证据等待本次收口提交。
- 可恢复增量 bundle：`artifacts/carts_v2/CARTS_V2_INTEGRATION_57efc16_INCREMENTAL.bundle`（要求基线 `698a2bb`，SHA-256 `74e1464c…dfcc68`；文件按忽略规则只留本地）
- 本轮开始：`2026-08-23T14:41:38Z`
- 硬截止：`2026-08-24T02:41:38Z`
- 补足窗口：`V2_RECOVERY_1`，开始 `2026-08-24T07:05:10Z`，原截止 `12:05:10Z`；因 GPU 碰撞容量恢复失败于 `07:49:45Z` 提前停止，不得自动重开
- 当前方法回退阶段：从 `2026-08-24T12:11:41Z` 开始，13:08:26Z 在预算内达到用户停止条件 6；没有自动新开 Recovery 窗口。
- 候选状态：旧 `candidate_11=INVALIDATED_BY_INTERMEDIATE_HAND_TABLE_SWEEP` 且不得重跑；旧 `candidate_33` 与旧 Top-3 已按新方法作废。当前 `selected_executable_candidate=null`。

## 冻结事实与复用边界

- 两个对象身份、登记网格、质量/质心/材料、8 N、0.30 N·m、50 µm、5 µm 及既有模型映射不得被本轮擅改；A 是公开规格仿真模型，B 是官方 STEP 派生模型。
- 8 N、0.30 N·m 等旧值先按来源分类，不自动声称是真实硬件极限，也不为跑通而删除。
- 旧 `ray_closure.py`、`continuous_collision.py`、`full_hand_collision.py`、`bounded_topk_exact.py` 冻结；V2 只允许一个薄公开适配器调用严格碰撞后端。
- 可复用：真实手/PAD/对象模型，`grasp_optimizer.py` 的候选与 Sobol，`robust_wrench.py` 的接触力 LP，`pareto_ranker.py` 的尾部统计/排序，以及现有路线/场景资产。
- 旧固定 Top-4、圆柱代理和 H 编号只作 baseline/reference，不得成为 V2 生产候选。

## 证据边界

- 必须分开：代码检查、离线算法、研究型动态、正式动态、真实硬件。
- 快筛只能写 `FAST_REJECT` 或 `FAST_SURVIVE`，不能声称严格无碰撞。
- 严格检查的 `UNRESOLVED` 保持不确定；它不等于安全，也不永久阻止满足独立研究门的动态观察。
- 在线控制只用关节侧现实可获得信号；对象位姿、接触名称、精确接触点/法向和 PhysX 真值仅供事后评价。
- 动态成功必须同时检查物理时间、三指有效接触、离桌、50 mm、2 s、滑移、姿态、桌面接触和穿透。

## 固定恢复读取顺序

1. `AGENTS.md`
2. `docs/carts_v2/NORTH_STAR_CN.md`
3. 本文件
4. `docs/carts_v2/SPRINT_12H_CN.md`
5. `docs/carts_v2/DECISIONS_CN.md` 最新部分
6. `artifacts/carts_v2/STATE.json`
7. `git status`、`git log -5`、当前 diff
8. 当前里程碑直接相关源码

## 当前直接路径

- 北极星：`docs/carts_v2/NORTH_STAR_CN.md`
- 计划/任务卡：`docs/carts_v2/SPRINT_12H_CN.md`
- 决策：`docs/carts_v2/DECISIONS_CN.md`
- 机器状态：`artifacts/carts_v2/STATE.json`
- V2 抓取源码：`src/kcg_connector/kcg_connector/grasp/carts_v2/`
- V2 Isaac 源码：`src/kcg_connector/isaac/carts_v2/`
- 单一配置：`src/kcg_connector/config/carts_grasp_v2.yaml`
- 方法与局限：`docs/carts_v2/METHOD_CN.md`
- 旧参考：`src/kcg_connector/kcg_connector/grasp/robust/`

## 动作前检查

- 文件修改前：说明它解决的物理/工程问题及不改变的冻结边界。
- 超过 15 分钟任务前：先在 `SPRINT_12H_CN.md` 写任务卡和停止条件。
- Isaac 前：当前研究动态门关闭；只有未来经明确方法阶段授权并产生新的 `executable_candidate`，再核对源码、配置、预检查和真值隔离后才可开放。
- 每个大里程碑后：更新本文件、Sprint、State；必要时追加 Decisions；监督审查后只提交本任务路径。
