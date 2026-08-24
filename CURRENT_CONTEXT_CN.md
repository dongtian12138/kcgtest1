# kcgtest1 当前轻量上下文

> 快照时间：2026-08-24T11:00:24Z
> 当前分支：`carts-grasp-v2-rebuild-20260823`
> 本文件只做恢复路由；物理结论以 V2 原始运行产物为准。

## 一句话状态

对象 A 的新源码绑定预检查已被严格接受；下一步只运行第一指 0.5 s 接触切换实验，不启动第二指、预紧或抬升。

## 六行恢复摘要

- 总目标：真实允许表面候选 → 任务载荷鲁棒 Top-3 → Isaac 三指接触、离桌、抬升 50 mm、保持至少 2 s；两个对象同算法同主要参数。
- 当前里程碑：`V2_OBJECT_A_FIRST_FINGER_CONTACT_SETTLE`；四状态切换及逐周期诊断已实现，新源码预检查已接受。
- 已经完成：预检查物理推进 7.75 s，控制器完成；最大关节速度 1.4252 rad/s，容量峰值 1/7644 低于 8192/16384，容量警告、PhysX Error 和三类未授权接触均为 0。
- 当前物理/算法阻塞：尚未证明第一指接触切换能消除对象场景 `f1j3` 超速，也未证明关节侧代理对应允许 PAD patch 接触。
- 下一步最简单动作：只运行对象 A、candidate_11 第一指 CONTACT_SETTLE/HOLD 0.5 s，并事后检查速度、受力、mimic 误差、对象位移和真实接触部位。
- 绝对禁止跑偏方向：续修 H102、对象专用坐标/阈值、磁吸/隐藏固定、在线读取对象/接触/PhysX 真值、写物体位姿、把退出 0 冒充物理成功。

## 当前权威字段

- 状态：`IMPLEMENTING`
- 当前里程碑：`V2_OBJECT_A_FIRST_FINGER_CONTACT_SETTLE`
- V2 正式候选：空
- V2 研究动态门：`allowed=true`，scope=`OBJECT_A_CANDIDATE11_FIRST_FINGER_DIAGNOSTIC_ONLY`；禁止第二指、预紧、抬升和对象 B
- 旧正式动态门：`dynamic_launch_allowed=false`（保持，不由 V2 研究线改写）
- 正式动态：`FORMAL_DYNAMIC_PASS=false`
- 研究型动态：`RESEARCH_DYNAMIC_PASS=false`
- 真实硬件：`hardware_authorized=false`
- Isaac：6.0.1.0；最新源码绑定预检查推进 7.75 s、最大关节速度 1.4252 rad/s、最大手部投影关节力矩 0.1317 N·m，已严格接受；它没有闭指、手—连接器接触、抬升或保持
- 引擎：run16 预检被严格接受；run17 GPU 配对峰值 1/7644 低于 8192/16384，容量警告和 PhysX Error 均为 0，因此本次失败不是 GPU 容量不足
- 最终 CPU 回归：1506 项通过、0 失败、982.86 s；测试通过不改变动态失败状态
- run16/run17 小型 `evaluation.json` 纳入本次收口；4.72/5.54 MB 逐步 trace 只留本地，SHA-256 与字节数登记在 `STATE.json`
- Recovery 2 生产代码截至 `03594b8` 已普通推送到远端同名分支；本次失败证据与收口文档将在最终单职责提交中推送
- 可恢复增量 bundle：`artifacts/carts_v2/CARTS_V2_INTEGRATION_57efc16_INCREMENTAL.bundle`（要求基线 `698a2bb`，SHA-256 `74e1464c…dfcc68`；文件按忽略规则只留本地）
- 本轮开始：`2026-08-23T14:41:38Z`
- 硬截止：`2026-08-24T02:41:38Z`
- 补足窗口：`V2_RECOVERY_1`，开始 `2026-08-24T07:05:10Z`，原截止 `12:05:10Z`；因 GPU 碰撞容量恢复失败于 `07:49:45Z` 提前停止，不得自动重开
- 当前补足窗口：`V2_RECOVERY_2`，截至本快照累计已用约 7597 s、余约 7703 s；连续执行截止仍为 `13:08:47Z`

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
- Isaac 前：当前只允许离线复盘；连续性实现与定向测试通过后，仍须先核对精确进程、正式字段、研究门、目标 run_id 和产物时间，才可开放一次第一指 0.5 s 实验。
- 每个大里程碑后：更新本文件、Sprint、State；必要时追加 Decisions；监督审查后只提交本任务路径。
