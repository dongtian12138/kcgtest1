# kcgtest1 当前轻量上下文

> 快照时间：2026-08-26T14:38:00Z
> 当前分支：`carts-grasp-contactopt-1488-fast6h-20260826`  
> 原始产物、源码、哈希、Git 与进程事实优先于本摘要。

## 一句话状态

对象 B 已严格登记 1488 个结构化规格并得到 1150 个三点—法向六维位姿；当前进入全量便宜几何评价，Isaac 尚未在本窗口启动，不能声称三指接触或抓取成功。

## 六行恢复摘要

- 最初目标：同一无指甲三指手、同一方法和主要参数用于 A/B，使三块真实任务抓持面接触，连接器离桌、抬升至少 50 mm、保持至少 2 s，且无未授权穿透。
- 当前已完成：旧 Surface V2 已完整收口；对象 B 的 1040 个全局规格和 448 个加密规格全部登记，1150 个生成位姿、338 个几何拒绝，覆盖图和逐项原因已保存。
- 当前真实物理结果：上一窗口对象 B 第一指 Isaac 真实推进 3.317 s，但手—物接触为 0；第二/三指未启动，抬升 0 mm、保持 0 s。本窗口尚未启动 Isaac。
- 当前唯一主要任务：对 1150 个已生成位姿运行缓存近邻/有限桌面/自碰代理，连同 338 个生成拒绝形成全 1488 分类并选择多样性 Top120。
- 最近用户指令：`METHOD_REBUILD + 1488_STRUCTURED_SEEDS + TRI_FINGER_CONTACT_CONSTRAINED_OPTIMIZATION + HIERARCHICAL_VALIDATION + SIX_HOUR_ISAAC_CLOSURE`；不是硬件授权。
- 下一步：运行便宜几何和多样性 Top120；90 分钟前保存全 1488 分类、层级计数和耗时，再进入逐指代理安全区间。

## 当前窗口和冻结边界

- 窗口：`CONTACTOPT_1488_FAST6H`；开始 `2026-08-26T14:16:36Z`；硬截止 `2026-08-26T20:16:36Z`；不自动延长。
- 状态：`IMPLEMENTING`；里程碑：`CONTACTOPT_1488_BATCHED_CHEAP_GEOMETRY`。
- `hardware_authorized=false`、`formal_dynamic_pass=false`、`research_dynamic_pass=false`。
- 固定层级：1488→Top120→Top48→Top12→Top3；第 3 小时前应有安全候选进入对象 B 局部 Isaac。
- 保持 12 N 仿真操作上限、3 rad/s 急停、0.18 rad/s 手指目标速度、0.0015 rad/控制步、50 mm、2 s、无指甲手型、对象物性和在线真值隔离。
- 不回到 H102，不运行旧 GraspGenX 主链，不重建已通过导入的无指甲 Isaac 碰撞资产。

## 固定 1488 设计

- 全局 1040：13 个掌面角 × 16 个 22.5°圆周方向 × 5 个轴向层 × `P0=[0.10,0.10,0.10]`。
- 加密 448：45°～75°共 7 个掌面角 × 每组 8 个特征感知方向 × 4 个轴向层 × `P1/P2` 两种对向预构型。
- 六维抓取姿态必须由真实手侧三点/法向与对象允许区域三点/材料绑定外法向经 Kabsch/SVD 和小型法向精修得到；不得随机、Sobol 或手填对象坐标。
- 1488 项均需留下 `POSE_GENERATED` 或唯一 `SEED_GEOMETRY_REJECT`；生成完整和优化收敛均不是抓取成功。

## 监督约束

- 前级只用缓存 FK、64～128 个代表点和对象 KDTree；Top12 才绑定原始无指甲网格与 Isaac 复合凸双几何。
- 每指输出 `q_expected` 和 `q_safe_max`；前级必须标为代理区间，最终角只可由 Top12 双几何确定。`q_safe_max` 是首个几何失败边界前一个不超过 0.0015 rad 的控制步。
- 只使用一条优化路线；不轮试多套求解器。冻结现有临界大文件，新代码保持四个数值模块和薄入口。
- 第 3 小时只有双几何、三指语义、安全区间和 12 N 名义余量通过的候选才能进 Isaac；否则失败关闭为没有安全动态输入。

## 旧基线事实

- Surface V2 40,824 个组合、189 组 FK 缓存，快搜约 85 s；两轮各 24 项原始网格精查和两轮各 8 项固定精修均为 0 个完整几何幸存者。
- 旧最佳三指精确重放桌面间隙 0.634889 mm，距冻结 1 mm 操作余量差 0.365111 mm。
- 这些只否定固定离散设计和登记邻域，不证明三指手机械结构或完整连续空间无解。
- 旧分支最终提交：`ea56c5798104b3a80094b290273e2c79a967e314`，已推送；旧证据不覆盖、不改写。

## 当前证据入口

- 新计划：`docs/carts_v2/CONTACTOPT_1488_FAST6H_PLAN_CN.md`
- 新清单：`artifacts/carts_v2/contactopt_1488_fast6h/MANIFEST.json`
- 旧最终报告：`docs/carts_v2/SURFACE_V2_FAST6H_FINAL_REPORT_CN.md`
- 旧第一指动态评价：`artifacts/carts_v2/surface_v2_fast6h/object_b_top1_first_finger_run01/evaluation_run02.json`

## 恢复读取顺序

1. `AGENTS.md`
2. `docs/carts_v2/NORTH_STAR_CN.md`
3. `docs/carts_v2/CONTACTOPT_1488_FAST6H_PLAN_CN.md`
4. 本文件
5. `docs/carts_v2/DECISIONS_CN.md` 最新部分
6. `artifacts/carts_v2/STATE.json`
7. `artifacts/carts_v2/contactopt_1488_fast6h/MANIFEST.json`
8. `git status`、`git log -8`
9. 当前唯一阻塞相关源码和新产物

## 证据边界

- 便宜近邻、代理区间、离线三指见证、FCL、IK、图片、测试和退出码都不是 Isaac 真实接触或抬升。
- Isaac 研究动态只有在物理时间真实推进、三指允许面接触、离桌、50 mm、2 s、无未授权穿透且在线未读真值时才可通过。
- 正式动态与硬件均保持否；用户未授权任何真实硬件动作。
