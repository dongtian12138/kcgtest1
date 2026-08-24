# kcgtest1 当前轻量上下文

> 快照时间：2026-08-24T16:39:00Z
> 当前分支：`carts-grasp-graspgenx-route1-20260824`
> 本文件只做恢复路由；物理结论以原始运行产物为准。

## 一句话状态

GraspGenX 官方模型已在 RTX 5070 Ti 上一次加载并对官方三指手样例输出 20 个有效六维位姿；5 个跨对象 KCG 手描述器和两个真实对象网格也已绑定，但仍无连接器候选、接触、离桌或抬升结果。

## 六行恢复摘要

- 最初目标：两个 D38999/J599 对象同一方法与主要参数，三根真实指腹接触、离桌、抬升至少 50 mm、保持至少 2 s，无未授权穿透。
- 当前已完成：固定官方 commit `b942909`、checkpoint 和官方描述器 revision；官方 `robotiq_3f + banana` 正对照一次加载模型并输出 20 个有限有效六维位姿；5 个对象无关描述器和两个登记网格已绑定。
- 当前真实物理结果：描述器相位 `0.50` 下 9 个预构型共检查 4203 个空手状态且无自碰；新路线尚未启动 Isaac，旧 A/B 结果只证明轴对称窄候选族失败。
- 当前唯一主要阻塞：用固定 5 个 KCG 描述器对两个真实连接器运行同一进程推理，并验证坐标链、尺度和六维覆盖没有退化。
- 最近用户指令：`METHOD_REVISION + NEW_AUTONOMOUS_EXECUTION_WINDOW + UNATTENDED_EXECUTION`，不是硬件授权或降低成功标准。
- 下一步：5 个固定 KCG 描述器对两个真实网格推理 → 六维覆盖 → V2 物理筛选、IK/路径预检查与 Isaac。

## 当前权威字段

- 状态：`IMPLEMENTING`
- 当前里程碑：`GRASPGENX_KCG_OBJECT_PROPOSALS`
- 窗口：`GRASPGENX_ROUTE1_12H_UNATTENDED`
- 开始：`2026-08-24T14:46:08Z`
- 硬截止：`2026-08-25T02:46:08Z`
- 官方 GraspGenX：本地隔离克隆，commit `b9429097728cb1c430dd78b92edf17ba318aad03`
- 隔离运行时：Python `3.10.12`、PyTorch `2.7.0+cu128`、torchvision `0.22.0+cu128`、`sm_120` 实际 GPU 张量通过。
- 官方正对照：`robotiq_3f + banana` 返回 `20/20` 个有限位姿；模型加载 `1.596 s`、推理 `0.522 s`、分数 `0.18669–0.98670`；只证明官方推理链联通。
- KCG 描述器：5 个；清单 SHA-256 `f60d89a29ff1a728d6126c1e28ab7babe09e58585391ea26cca0ab476bfe564e`
- 描述器空手检查：9 个固定 `f1j1` 样本、4203 个状态、最大独立关节步长 `0.0015 rad`、未发现非相邻连杆自碰。
- 对象输入：A `88,078/145,588` 顶点/面；B `343,520/687,036` 顶点/面；均由登记网格原样导出为米制 NPZ。
- 研究动态门：`allowed=false`；必须先产生并绑定 `research_executable_candidate`
- 旧正式动态门：保持关闭
- 正式动态：`false`
- 研究型动态：`false`
- 真实硬件：`hardware_authorized=false`
- 主研究力量上限：`12 N` 仿真操作上限；旧 `8 N` 仅作消融；不得扫描增力
- 动态目标：三指允许指腹接触、离桌、50 mm、至少 2 s、无未授权穿透

## 冻结事实与复用边界

- 两个对象身份、登记网格、质量、质心、惯量、摩擦和误差范围不变。
- 3 rad/s 关节安全限速、0.18 rad/s 手指目标速度、1/120 s 周期、0.0015 rad/周期、50 mm 和 2 s 不变。
- 在线控制不得读取对象位姿、接触名称、精确接触点/法向或 PhysX 真值；这些只供事后评价。
- 旧轴对称生成器、旧 Top-3 和 `candidate_11/33` 只作历史消融，不是本轮生产候选。
- 旧 H102 与大型连续碰撞实现冻结；Top-3 只通过现有薄适配器调用。
- 复用 V2 真实手/PAD/对象模型、完整顺序闭合扫掠、任务力学、selector、bounded IK、Isaac 控制器和评价器。
- GraspGenX 隔离环境只输出版本化文件，不污染 Isaac Python；第三方仓库、权重和缓存不提交。

## 证据边界

- 官方样例推理只证明模型环境联通，不证明 KCG 抓取可行。
- 六维覆盖只证明搜索姿态族变宽，不证明不碰撞或抓得住。
- `research_executable_candidate` 必须通过完整几何路径、三指名义接触、IK/路径和 12 N 名义任务门，才可进入研究型 Isaac。
- `NOMINAL_RESEARCH_DYNAMIC_PASS` 必须由物理时间、三指允许接触、离桌、50 mm、2 s、滑移、桌面接触和穿透共同判定。
- 正式动态还需预登记误差、严格证据、重复、扰动和双对象同参数；本轮旧正式证据线保持失败关闭。

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
10. 当前唯一阻塞相关源码

## 当前直接路径

- 路线计划：`docs/carts_v2/GRASPGENX_ROUTE1_PLAN_CN.md`
- 决策：`docs/carts_v2/DECISIONS_CN.md`
- 机器状态：`artifacts/carts_v2/STATE.json`
- 集成清单：`artifacts/carts_v2/graspgenx/INTEGRATION_MANIFEST.json`
- V2 抓取源码：`src/kcg_connector/kcg_connector/grasp/carts_v2/`
- V2 Isaac：`src/kcg_connector/isaac/carts_v2/`

## 动作前检查

- 超过 30 分钟的工作先在路线计划写物理问题、唯一改变量、观察量、失败判据和时间上限。
- Isaac 前核对候选、源码、配置、模型、GPU 资源、预检查与真值隔离；研究门和正式门分开。
- 每 45～60 分钟及大里程碑更新本文件、计划、状态和集成清单；只提交本任务路径。
