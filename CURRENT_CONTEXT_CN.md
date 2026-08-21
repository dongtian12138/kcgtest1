# kcgtest1 当前轻量上下文

> 快照时间：2026-08-21T14:38:24Z
> 用途：新对话、上下文压缩恢复和普通源码工作先读本文件，避免默认加载完整历史台账。
> 边界：本文件是当前状态的轻量路由，不覆盖追加式历史证据；出现冲突、时间漂移或准备动态运行时必须回查原始控制文件和运行产物。

## 一句话状态

当前研究主线是 `CARTS-GRASP-CROSS-OBJECT-V1`：三指接触范围计划已进入顶层生成器、断点和一部分手部连续防碰撞检查，当前完整静态回归 `437 passed`。但检查范围还不包括实体包含关系、权威整机/环境全路径和接触范围受力评价，非摩擦误差也未标定，因此没有正式候选且禁止启动 Isaac。

## 当前权威字段

- 总任务：`D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2`
- 当前任务：`CARTS-GRASP-CROSS-OBJECT-V1`
- 状态：`IMPLEMENTING`
- 当前前沿：`POLICY_AWARE_COLLISION_AND_WRENCH_CONSUMERS_THEN_COMPLETE_COLLISION_AND_NONFRICTION_UNCERTAINTY_CERTIFICATION`
- 当前假设：`CARTS-H1-OBJECT-SPECIFIC-CANDIDATES-AND-EMPIRICAL-GATES-CAUSE-NONTRANSFER`
- 正式候选：`selected_candidate=null`
- 完整碰撞证书：`NOT_CERTIFIABLE`
- 动态启动：`dynamic_launch_allowed=false`
- B阶段正式通过：`formal_b_passed=false`
- 真实硬件授权：`hardware_authorized=false`
- 进程快照：2026-08-21T14:38:24Z 未发现活动 Isaac、目标 runner 或 pytest；动态前必须重新核对，不能沿用本快照。
- 最新 CARTS 完整静态回归：`437 passed in 414.35s`，测试前后受检文件 SHA-256 清单一致。

## 已有证据与不能声称的内容

- 任务控制面的最新持久快照仍记录 `420 passed in 410.94s`；其后本地检查点 `53411e4` 在稳定工作树实测 `437 passed in 414.35s`。两者都只是静态程序证据，不升级为仿真或动态通过。
- study canonical SHA-256：`fd3ee314f6b956a22e3c5c5eba95a68b1aaddab630d500c75da5a0fcca7b5dab`，`freeze_eligible=false`。
- 三根手指的蓝色接触表面均从原始末端STL精确提取为2479面；旧程序报告的每指11个缺失面已确认是假缺失，当前遗漏为0且原始顶点未移动。
- 已完成 V9 多精度区间顺序闭合、对象/手/PAD/方向模型绑定、四 lane 固定预算生成、共同 QMC 的 rank-only 绑定和连续表面碰撞内核。
- 三个抓法入口现在把三指接触范围组成不可变顺序闭合计划并传到顶层生成器与断点；正式`candidate`仍为空，display-only近似值仍不能进入正式轨迹复核或排名。
- 手部非PAD表面对对象和自碰撞现已能覆盖注册接触范围及两个独立闭合阶段的完整乘积；当前仍是 `NOT_CERTIFIABLE`，因为实体包含、权威整机/环境全路径和接触范围受力评价尚未接通。
- 以上只是静态方法与协议证据，不证明 current/TE 双对象离线候选通过，更不证明 Isaac 抓取、离桌、40 mm 抬升、保持、扰动或真实硬件成功。

## 当前三个阻断条件

1. `MISSING_FORMAL_ROOT_INTERVAL_CANDIDATE_PROPAGATION`
   - 接触范围计划已到达顶层、断点和部分手部防碰撞；还需让受力评价和正式排序直接检查整个范围，禁止使用显示用近似值。
2. `MISSING_COMPLETE_HAND_ENVIRONMENT_CONTINUOUS_COLLISION_BINDING`
   - 还需完成 PAD 连续表面、实体外部/包含关系、权威 17-link 和环境碰撞覆盖证书。
3. `MISSING_CALIBRATED_NONFRICTION_UNCERTAINTY_BOUNDS`
   - 还需给位姿、质心、执行误差等非摩擦不确定性建立可追溯、经校准的边界。

任一条件缺失时，正式排名必须为空，禁止选择候选或启动动态验证。

## 当前主线文件

- 任务切换合同：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TASK_SWITCH_PLAN.json`
- 最新静态快照：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/STATIC_PROGRESS_20260821T134214Z.json`
- 接触范围计划最终静态证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_U_RESULT_20260821T134214Z.json`
- 接触范围计划传递证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_Q_RESULT_20260821T130539Z.json`
- 顶层生成与断点证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_R_RESULT_20260821T131631Z.json`
- 排名器闭门拒绝证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_S_RESULT_20260821T131843Z.json`
- 方法实现：`src/kcg_connector/kcg_connector/grasp/robust/`
- Isaac 接口：`src/kcg_connector/isaac/robust_grasp/`
- 当前测试：`src/kcg_connector/test/robust_grasp/`
- 当前配置：`src/kcg_connector/config/carts_grasp_v1.yaml`、`carts_grasp_objects_v1.yaml`、`carts_hand_contact_v1.yaml`、`carts_collision_roster_v1.yaml`
- 第二对象来源：`j599_25_35_standard_interface_v1/` 与 TE/J35 派生脚本；既有 `TE_J35_PHYSX_V1.usda` 只是装配 smoke 夹具，不是抓取通过证据。

## 2026-08-21 Python、文档与归档清理

- 工程内 `.py` 从清理前 2852 个降到 1500 个；其中 1040 个属于 `.venv`，当前普通源码区451个，`src/kcg_connector/` 399个。第二轮另外退役52个无当前入向依赖的 Python、8份旧 YAML、1份旧 JSON 和2份旧说明。
- 当前 CARTS 范围为28个方法模块、2个 Isaac 接口、24个测试和4份配置；新对话只按当前任务定向读取，不遍历历史台账或旧下游实现。
- 第二轮删除63个文件、44763行、约1.84 MB；冻结合同的478 KB Python 数据字面量改为约3 KB加载器加42.7 KB压缩包，解压前后10组冻结值规范哈希一致。
- 第二轮非 CARTS 套件实测 `2494 passed in 100.42s`；冻结合同/安装树与方法合同定向测试 `176 passed`；CARTS 完整套件 `437 passed in 414.35s`。均为静态证据。
- Git 回退点依次为 `pre-contract-cleanup-20260821`、`post-contract-cleanup-20260821` 和 `current-carts-j599-checkpoint-20260821`；本地检查点 `53411e4` 保存最新 CARTS 与压缩快照，尚未推送远端。
- J599 原54个展开证据已形成可恢复压缩归档并逐文件复核，直接保留7份接受摘要/报告；这不改变其公共标准模型、非厂商原始 CAD、非真实硬件验收的边界。
- 当前活动源码没有 `/home/noob/...` 运行时绑定；J599 已接受报告、生成资产和恢复示例中的绝对路径只作不可改写的来源记录，不参与在线控制。

## 永久边界

- 工程仍为 simulation-only；真实硬件授权保持 false。
- 在线控制不得读取对象真实位姿、碰撞体名称、接触法向、事件真值或 PhysX 接触真值；这些只能用于运行后独立评价。
- 禁止磁吸、隐藏固定、虚构支撑、运行后写活动端位姿、放宽正式门限或把生成文件/退出码/`passed=true` 升级成动态成功。
- 冻结连接器身份、几何、质量、质心、材料类别、七事件位置和正式安全门限不得因清理任务改变。
- 旧 `B_GOAL_MODE`、固定 `CAD_*` 候选、圆柱近似和 H1-H25 路线只作历史 baseline，不得重新成为当前候选、评分或控制输入。

## 最小读取路线

1. 新对话或压缩恢复：读 `AGENTS.md` 和本文件。
2. 准备处理当前 CARTS-Grasp：再读上面的任务切换合同、最新静态快照及将要修改的源码/测试。
3. 只有以下情况才读取完整控制面或历史台账：
   - 本文件缺失、时间明显落后或与磁盘产物冲突；
   - 审计历史授权、run_id、失败谱系、冻结摘要或正式 gate；
   - 准备启动 Isaac、改变任务状态或更新正式验收结论。
4. 不得为“保险”一次性加载整个 `MASTER_STATE.json`、`TASK_GRAPH.yaml`、`BLOCKER_LEDGER.jsonl` 和 `GATE_LEDGER.csv`；应先用任务 ID、blocker ID 或 gate ID 定向检索。

## 动作前检查

- 修改文件前：用中文说明为什么改、服务哪个物理/工程问题以及不改变哪些冻结边界。
- 普通静态测试前：确认测试只覆盖当前修改，不生成或改写冻结证据。
- Isaac 前：重新读取完整当前控制字段，精确检查 PID、runner、run_id、日志和报告时间；本快照明确不构成启动授权。
- 历史审计文件保持原路径，不因轻量入口而删除或改写。
