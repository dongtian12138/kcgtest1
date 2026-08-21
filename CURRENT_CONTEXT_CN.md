# kcgtest1 当前轻量上下文

> 快照时间：2026-08-21T13:28:34Z
> 用途：新对话、上下文压缩恢复和普通源码工作先读本文件，避免默认加载完整历史台账。
> 边界：本文件是当前状态的轻量路由，不覆盖追加式历史证据；出现冲突、时间漂移或准备动态运行时必须回查原始控制文件和运行产物。

## 一句话状态

当前研究主线是 `CARTS-GRASP-CROSS-OBJECT-V1`：三个入口生成的三指接触范围抓取计划已传到顶层固定预算生成器和断点，排名器也会在误用这种计划时显式拒绝；当前完整静态回归 `420 passed`。但整机防碰撞、接触范围受力评价和非摩擦误差标定尚未完成，因此没有正式候选且禁止启动 Isaac。

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
- 进程快照：2026-08-21T12:40:28Z 未发现活动 Isaac/目标 runner；动态前必须重新核对，不能沿用本快照。
- 工作树状态：当前 CARTS-Grasp、TE/J35 脚本与 J599 标准模型尚未纳入 Git；本次整理正在对它们建立可恢复检查点。最新 CARTS 源码已完成 `420 passed in 415.80s`，且测试前后源码哈希一致。

## 已有证据与不能声称的内容

- 任务控制面的最新完整持久快照仍记录 `414 passed in 392.53s`；其后 Q/R/S 三个子节点均有独立 `STATIC_PASS` 结果。当前工作树另行实测 `420 passed in 415.80s`，退出码0，且测试前后受检源码哈希完全一致。这是当前静态回归证据，不写回历史控制面，也不升级为动态通过。
- study canonical SHA-256：`eb7fbebc008cdc7dc48fe58475c165bec2f0a9614d03d607087d6ceb205da9be`，`freeze_eligible=false`。
- 三根手指的蓝色接触表面均从原始末端STL精确提取为2479面；旧程序报告的每指11个缺失面已确认是假缺失，当前遗漏为0且原始顶点未移动。
- 已完成 V9 多精度区间顺序闭合、对象/手/PAD/方向模型绑定、四 lane 固定预算生成、共同 QMC 的 rank-only 绑定和连续表面碰撞内核。
- 先前18个合成抓法入口测试失败的原因已确认并修正：三个入口现在保留display-only近似抓法和三指接触范围；正式`candidate`仍为空，正式轨迹复核仍以`REPRESENTATIVE_PROPOSAL_ONLY_PENDING_ROOT_INTERVAL_PROPAGATION`拒绝近似值。
- 以上只是静态方法与协议证据，不证明 current/TE 双对象离线候选通过，更不证明 Isaac 抓取、离桌、40 mm 抬升、保持、扰动或真实硬件成功。

## 当前三个阻断条件

1. `MISSING_FORMAL_ROOT_INTERVAL_CANDIDATE_PROPAGATION`
   - 还需把三指首次接触的可能范围形成一份可供防碰撞、受力评价和排序共同消费的正式抓法；禁止直接使用显示用近似值。
2. `MISSING_COMPLETE_HAND_ENVIRONMENT_CONTINUOUS_COLLISION_BINDING`
   - 还需完成 PAD 连续表面、实体外部/包含关系、权威 17-link 和环境碰撞覆盖证书。
3. `MISSING_CALIBRATED_NONFRICTION_UNCERTAINTY_BOUNDS`
   - 还需给位姿、质心、执行误差等非摩擦不确定性建立可追溯、经校准的边界。

任一条件缺失时，正式排名必须为空，禁止选择候选或启动动态验证。

## 当前主线文件

- 任务切换合同：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TASK_SWITCH_PLAN.json`
- 最新静态快照：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/STATIC_PROGRESS_20260821T123331Z.json`
- 接触范围计划传递证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_Q_RESULT_20260821T130539Z.json`
- 顶层生成与断点证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_R_RESULT_20260821T131631Z.json`
- 排名器闭门拒绝证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_S_RESULT_20260821T131843Z.json`
- 方法实现：`src/kcg_connector/kcg_connector/grasp/robust/`
- Isaac 接口：`src/kcg_connector/isaac/robust_grasp/`
- 当前测试：`src/kcg_connector/test/robust_grasp/`
- 当前配置：`src/kcg_connector/config/carts_grasp_v1.yaml`、`carts_grasp_objects_v1.yaml`、`carts_hand_contact_v1.yaml`、`carts_collision_roster_v1.yaml`
- 第二对象来源：`j599_25_35_standard_interface_v1/` 与 TE/J35 派生脚本；既有 `TE_J35_PHYSX_V1.usda` 只是装配 smoke 夹具，不是抓取通过证据。

## 2026-08-21 Python 与归档清理

- 仓库中的 `.py` 总数从 2852 降到 1552；其中 1040 个是 `.venv` 第三方依赖，`src/kcg_connector/` 当前有451个。两轮合计让1300个 Python 文件退出普通检索。
- 当前 CARTS 版本化范围为28个方法模块、2个 Isaac 接口和24个测试；当前完整静态回归为 `420 passed`。新对话默认只读这些路径，不遍历其他历史/下游实现。
- 已退役57个绑定旧单体运行器、源码文本或历史 SHA-256 的测试，以及13个无当前调用者/依赖测试文件才能加载的历史模块；剩余非 CARTS 套件实测 `2580 passed in 108.72s`，退出码0。
- Git 标签 `pre-contract-cleanup-20260821` 保存清理前的已提交历史。主分支不再复制保存已跟踪源码 ZIP；唯一未被旧提交覆盖的 B 路线封装为 `legacy_archive/RETIRED_B_GRASP_ROUTES_20260820.zip`，138个成员、616538 bytes、SHA-256 `626c55f8beeb2285bf19247766545356f31fc60641878fbd5d70c5cd1235a6f4`。
- 已删除多组与现有 ZIP/tar 逐字节相同的交付包解压副本；原压缩包、SHA-256 旁车、内部 manifest 和唯一运行数据保留。
- `.vscode/settings.json` 会隐藏可再生成目录，并把 `.venv`、`artifacts`、`legacy_archive`、`build/install/log` 排除出搜索和文件监视；证据目录仍可在资源管理器中手动展开。
- `kcg_connector.grasp` 公共 API 改为按需导入；导入 `grasp.robust` 不再顺带加载 7 个旧 B 控制模块。兼容 API 和相关回归为 `141 passed`。
- 活动源码和说明中不再存在 `/home/noob/...` 的机器绝对路径绑定；仓库适用范围说明和个人学习文档中的路径文字不作为运行路径。

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
