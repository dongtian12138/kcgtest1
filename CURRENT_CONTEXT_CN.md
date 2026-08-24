# kcgtest1 当前轻量上下文

> 快照时间：2026-08-24T09:29:31Z
> 当前分支：`carts-grasp-v2-rebuild-20260823`
> 本文件只做恢复路由；物理结论以 V2 原始运行产物为准。

## 一句话状态

“接触确认后保持实测位置”的最小修复已通过严格 run14 预检查；当前只允许绑定 run14 的一次对象 A 抓取验证，对象 B 仍关闭。

## 六行恢复摘要

- 总目标：真实允许表面候选 → 任务载荷鲁棒 Top-3 → Isaac 三指接触、离桌、抬升 50 mm、保持至少 2 s；两个对象同算法同主要参数。
- 当前里程碑：`V2_RECOVERY_2_OBJECT_A_CONTACT_STOP_DYNAMIC_VALIDATION`；对象 B 关闭，只允许绑定 run14 的一次 A 抓取。
- 已经完成：run13 排除相同目标序列本身足以触发尖峰；接触保持修复提交 `f2ee26f`；严格 run14 推进 7.75 s、最大速度 1.4252 rad/s、未授权接触与容量错误均为 0。
- 当前物理/算法阻塞：尚未证明有限刚度接触保持能避免对象场景的 `f1j3` 尖峰并形成三指有效接触。
- 下一步最简单动作：不再改源码，绑定 run14 原样执行一次对象 A 抓取并事后评价。
- 绝对禁止跑偏方向：续修 H102、对象专用坐标/阈值、磁吸/隐藏固定、在线读取对象/接触/PhysX 真值、写物体位姿、把退出 0 冒充物理成功。

## 当前权威字段

- 状态：`IMPLEMENTING`
- 当前里程碑：`V2_RECOVERY_2_OBJECT_A_ACCEPTED_PREFLIGHT`
- V2 正式候选：空
- V2 研究动态门：`allowed=true`，scope=`OBJECT_A_CONTACT_STOP_GRASP_LIFT_ONLY`；只允许绑定 run14 的一次 A 抓取，对象 B 关闭
- 旧正式动态门：`dynamic_launch_allowed=false`（保持，不由 V2 研究线改写）
- 正式动态：`FORMAL_DYNAMIC_PASS=false`
- 研究型动态：`RESEARCH_DYNAMIC_PASS=false`
- 真实硬件：`hardware_authorized=false`
- Isaac：6.0.1.0；run12 在对象场景 `f1j3=48.013 rad/s` 急停；run13 无对象/桌面、相同 1104 个目标下 `f1j3=0.1588 rad/s`，故对象场景是必要条件；尚无三指接触、离桌、抬升或保持
- 最终 CPU 回归：1506 项通过、0 失败、982.86 s；测试通过不改变动态失败状态
- run05 小型 `evaluation.json` 纳入交付；1.58 MiB 逐步 trace 只留本地，SHA-256 与字节数登记在 `STATE.json`
- V2 恢复提交 `0848abe`、mimic 修复提交 `7633f3a` 和失败收口 `c127bea` 已普通推送到远端同名分支
- 可恢复增量 bundle：`artifacts/carts_v2/CARTS_V2_INTEGRATION_57efc16_INCREMENTAL.bundle`（要求基线 `698a2bb`，SHA-256 `74e1464c…dfcc68`；文件按忽略规则只留本地）
- 本轮开始：`2026-08-23T14:41:38Z`
- 硬截止：`2026-08-24T02:41:38Z`
- 补足窗口：`V2_RECOVERY_1`，开始 `2026-08-24T07:05:10Z`，原截止 `12:05:10Z`；因 GPU 碰撞容量恢复失败于 `07:49:45Z` 提前停止，不得自动重开
- 当前补足窗口：`V2_RECOVERY_2`，开始 `2026-08-24T08:13:09Z`，硬截止 `12:28:09Z`；这是资源误分类纠正后的剩余执行时间，不是新 12 小时窗口

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
- Isaac 前：先核对精确进程、旧正式字段、V2 研究门、目标 run_id 和产物时间；当前只允许对象 A 一次预检查，抓取仍未开放。
- 每个大里程碑后：更新本文件、Sprint、State；必要时追加 Decisions；监督审查后只提交本任务路径。
