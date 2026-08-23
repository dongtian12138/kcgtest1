# kcgtest1 当前轻量上下文

> 快照时间：2026-08-23T12:47:40Z
> 用途：新对话、上下文压缩恢复和普通源码工作先读本文件，避免默认加载完整历史台账。
> 边界：本文件是当前状态的轻量路由，不覆盖追加式历史证据；出现冲突、时间漂移或准备动态运行时必须回查原始控制文件和运行产物。

## 一句话状态

当前研究主线是 `CARTS-GRASP-CROSS-OBJECT-V1`：256种便宜初筛已实测约11.99秒，并固定Top-4为155、174、14、198。H94发现V9的2479面蓝色PAD接触表面与终端碰撞检查的16面许可表面身份不一致；H96已把允许接触的PAD skin与禁止穿透的closed shell分成独立角色。H97的正确路线把candidate 14的`f1Link3`从36.85秒降到21.22秒。H99延迟窄相导致重复终点工作，H100证明该策略结构性不适用。H101恢复每次细分前完整窄相并预计算两侧自身轴投影，相关192项测试通过；但candidate 14为24.65秒、Top-4 90秒仍超时。根因是BVH叶分别触发21444个窄相小包，平均每包393.23对，65536对上限没有被利用。H102尝试跨root叶汇总pair，但用户已在正式结果登记前叫停；本次只保存实现快照和假设，不声称H102通过。生产路线仍为0，非摩擦误差未标定，正式结果为空且禁止启动 Isaac。

## 当前权威字段

- 总任务：`D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2`
- 当前任务：`CARTS-GRASP-CROSS-OBJECT-V1`
- 状态：`IMPLEMENTING`
- 当前前沿：`H102_STOPPED_RESULT_UNREGISTERED`
- 当前假设：`CARTS-H1-OBJECT-SPECIFIC-CANDIDATES-AND-EMPIRICAL-GATES-CAUSE-NONTRANSFER`
- 正式候选：`selected_candidate=null`
- 正式接触范围计划：`formal_selected_contact_range_policy=null`
- 完整碰撞证书：`NOT_CERTIFIABLE`
- 动态启动：`dynamic_launch_allowed=false`
- B阶段正式通过：`formal_b_passed=false`
- 真实硬件授权：`hardware_authorized=false`
- 本轮进程事实：未启动 Isaac、目标 runner、机械臂或三指手；动态前仍必须重新核对进程，不能把本条当成未来授权。
- 最新保留的 CARTS 完整静态回归：`535 passed in 775.75s`；它早于H101，不能代表H101完整回归。
- H101相关的连续碰撞、完整手碰撞和study-contract定向测试：`192 passed in 9.82s`；聚焦测试`7 passed in 0.14s`。这只是离线静态证据。
- 最新正式生成入口定向测试：`2 passed in 86.55s`。精确缓存定向测试为`2 passed in 8.14s`；最后两文件测试的原始退出行在上下文压缩时丢失，测试缓存登记45项且失败清单为空，只能称“支持通过”，不能称保留了原始退出码。当前对象旧第0号未提交且断点不可恢复，两个对象的正式完成尝试数均为0。

## 已有证据与不能声称的内容

- 面向 ChatGPT Pro 的当前算法审查入口：`docs/reviews/CARTS_GRASP_CHATGPT_PRO_REVIEW_20260823_CN.md`。它汇总H93-H101实测瓶颈、源码阅读路线和不允许改变的边界；不构成新验收结论。
- H100已验证H99的延迟窄相会保留本可排除的三角形对，使两个子区间重复执行终点窄相；下一路线必须在继续细分前完成完整窄相，性能只能从单次窄相吞吐、批处理和安全缓存取得。
- H101的90秒Top-4对比返回124：candidate 155的`f1Link3`为4.84秒，candidate 14为24.65秒，candidate 198只开始未完成；正式结果明确是性能目标未达，不能选择候选。

- 任务控制面当前持久快照记录 `535 passed in 775.75s`；这只是静态程序证据，不升级为仿真或动态通过。
- study canonical SHA-256：`fd421d99b7aca4a03d20c1d5e51454c4f3b7072ad5d0bed3aca971e47aded467`，`freeze_eligible=false`；H96仅同步完整指腹与封闭外壳的独立角色规则。
- 三根手指的蓝色接触表面均从原始末端STL精确提取为2479面；旧程序报告的每指11个缺失面已确认是假缺失，当前遗漏为0且原始顶点未移动。
- 已完成 V9 多精度区间顺序闭合、对象/手/PAD/方向模型绑定、四 lane 固定预算生成、共同 QMC 的 rank-only 绑定和连续表面碰撞内核。
- 三个抓法入口现在把三指接触范围组成不可变顺序闭合计划并传到顶层生成器与断点；正式`candidate`仍为空，display-only近似值仍不能进入正式轨迹复核或排名。
- 手部非PAD表面对对象和自碰撞现已能覆盖注册接触范围及两个独立闭合阶段的完整乘积；受力模块会覆盖每指全部最早接触、三指完整组合、关节传力、摩擦变化和12个任务负载方向，不读取显示中间值。
- 顶层保存的接触范围计划现已逐项进入一次范围碰撞和一次范围受力检查，并可进入诊断排序；完整条件未满足时正式排序和选择仍强制为空。
- 17个权威机械臂/三指手碰撞部件均已通过精确闭合、无自穿和材料内外证明；三根末端各有16个碰撞面与授权蓝色指垫完全一致且朝向一致，每根其余1084面全部禁止接触，未使用距离或最近面映射。
- 已把11关节完整运动链、17外壳、三末端角色、136组机器人部件对、对象材料范围以及桌面/夹具24个环境三角面组成同一份静态输入；当前对象记录为 `e18438c7…15d7`，TE对象记录为 `dfb16a08…872e`。
- 当前对象的7642个来源实体已逐一证明为封闭且不自穿的正材料，全部145588个面恰好覆盖一次；TE对象的单一整体也通过精确材料边界检查。两对象材料范围均已接入各自对象合同。
- 当前对象的7642个来源节点已证明主要是程序分件；严格公共面对消无法安全删面，因此碰撞几何保持不变。派生显示文件只按7种原颜色合并节点，145588个三角面全部保留，Isaac读取和渲染性能仍未验证。
- 桌面与夹具已从两份既有来源交叉核对并登记：桌面顶面0.20米、夹具顶面0.24米，共享环境证书 `1c471e90…dd908`；它不含固定插座、活动插头位姿或候选路线。
- 两个活动插头已按同一规则落在桌面上：共享XY为 `(0.520, -0.210)` 米、绕X轴180度；当前对象世界原点Z为 `0.23050000144867228` 米，TE对象为 `0.20` 米，两者模型最低点都落在桌面顶面 `0.20` 米。当前/TE位置证书分别为 `a71e0282…d51d` 与 `f8ed7346…e2d`，未读取在线真值，也未使用旧抓取候选。
- 路线依赖顺序已经纠正：每份静态通过的接触范围计划先单独生成并检查路线，只有可执行路线才能参加最终选择；不再要求先有正式最佳计划才生成路线。
- 通用路线程序使用完整URDF模型反算机械臂7关节角：HOME到预抓取最大相邻关节步不超过0.02弧度，40毫米竖直抬升拆成9个目标，三指闭合继续保留接触触发的角度范围。当前/TE测试路线最大位置误差小于8e-13米、最大姿态误差小于3.2e-12弧度。
- 上述测试证明方法能生成限位内路线，但正式固定预算候选尚未运行，所以生产关节路线数量仍为0，控制器执行授权仍为false。
- 完整路线碰撞程序对136组基础自身碰撞对保留完整来源，只排除15组由URDF直接父子关系独立证明的结构连接，路线仍检查其余121组；精确HOME分离不冒充整段运动通过。
- 正式生成运行器已对当前与TE两个真实对象完成同算法装配，并能写入固定256次的空断点；当前仍没有任何正式尝试结果。旧版本和精确缓存版本都在各自第0号未提交后永久停用，不得从任一断点重试。
- 精确缓存版本第0号满负荷运行1500秒、返回124；内存观测峰值约16.9 GB，随后稳定在约11.8 GB，没有系统内存错误。最新断点为`6ea0b641…bf60`和0/256；运行前`STATUS.json`是旧快照，不能冒充完成。
- 高精度区间运动学现在会在同一闭合小段内复用完全相同的关节链变换：7437次同输入调用得到7436次命中、1次实际计算，总耗时2.09秒；按此前单次测量推算约快34倍。缓存不跨闭合小段和运行，不删除表面点或对象面，也不改变精度和细分上限。
- 当前总碰撞结论仍是 `NOT_CERTIFIABLE`：测试路线尚未逐段进入完整机器人、插头、桌面和夹具的连续碰撞检查，允许指垫接触也未证明；非摩擦误差还没有完成标定。
- 以上只是静态方法与协议证据，不证明 current/TE 双对象离线候选通过，更不证明 Isaac 抓取、离桌、40 mm 抬升、保持、扰动或真实硬件成功。

## 当前两个阻断条件

1. `MISSING_COMPLETE_HAND_ENVIRONMENT_CONTINUOUS_COLLISION_BINDING`
   - 机器人、对象、桌面、夹具、两个活动插头起始位置、六阶段目标和通用11关节路线程序已就绪；还需把路线逐段接入完整碰撞检查并证明允许指垫连续接触。
2. `MISSING_CALIBRATED_NONFRICTION_UNCERTAINTY_BOUNDS`
   - 还需给位姿、质心、执行误差等非摩擦不确定性建立可追溯、经校准的边界。

任一条件缺失时，正式排名必须为空，禁止选择候选或启动动态验证。

## 当前主线文件

- 任务切换合同：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TASK_SWITCH_PLAN.json`
- 最新静态快照：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/STATIC_PROGRESS_20260822T141640Z.json`
- 11关节路线结果：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/CANDIDATE_JOINT_ROUTE_RESULT_20260822T141640Z.json`
- 六阶段路线规则结果：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/CANDIDATE_ROUTE_CONTRACT_RESULT_20260822T132335Z.json`
- 最新位置结果：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/OBJECT_WORLD_POSE_RESULT_20260822T123537Z.json`
- 双对象静态场景组合结果：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/COMPLETE_COLLISION_SUBNODE_AW_RESULT_20260822T115252Z.json`
- 双对象材料范围结果：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/COMPLETE_COLLISION_SUBNODE_AV_RESULT_20260822T111649Z.json`
- 显示节点简化结果：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/DISPLAY_SIMPLIFICATION_SUBNODE_AU_RESULT_20260822T101336Z.json`
- 派生显示文件：`artifacts/carts_grasp/CARTS_GRASP_V1/object_models/current_d38999/display/D38999_LOOSE_PLUG_RENDER_MERGED_V1.usda`
- 共享桌面/夹具与对象组成诊断：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/COMPLETE_COLLISION_SUBNODE_AT_RESULT_20260822T090855Z.json`
- 完整机器人模型与路线检查输入证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/COMPLETE_COLLISION_SUBNODE_AS_RESULT_20260822T084012Z.json`
- 17部件材料边界与精确指垫角色证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/COMPLETE_COLLISION_SUBNODE_AR_RESULT_20260822T075624Z.json`
- 接触范围进入最终筛选证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_AP_RESULT_20260822T064600Z.json`
- 接触范围计划最终静态证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_U_RESULT_20260821T134214Z.json`
- 接触范围计划传递证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_Q_RESULT_20260821T130539Z.json`
- 顶层生成与断点证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_R_RESULT_20260821T131631Z.json`
- 排名器闭门拒绝证据：`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/FORMAL_CANDIDATE_SUBNODE_S_RESULT_20260821T131843Z.json`
- 方法实现：`src/kcg_connector/kcg_connector/grasp/robust/`
- Isaac 接口：`src/kcg_connector/isaac/robust_grasp/`
- 当前测试：`src/kcg_connector/test/robust_grasp/`
- 当前配置：`src/kcg_connector/config/carts_grasp_v1.yaml`、`carts_grasp_objects_v1.yaml`、`carts_hand_contact_v1.yaml`、`carts_collision_roster_v1.yaml`、`carts_candidate_route_v1.yaml`、`carts_candidate_joint_route_v1.yaml`
- 第二对象来源：`j599_25_35_standard_interface_v1/` 与 TE/J35 派生脚本；既有 `TE_J35_PHYSX_V1.usda` 只是装配 smoke 夹具，不是抓取通过证据。

## 2026-08-21 Python、文档与归档清理

- 工程内 `.py` 从清理前 2852 个降到 1500 个；其中 1040 个属于 `.venv`，当前普通源码区451个，`src/kcg_connector/` 399个。第二轮另外退役52个无当前入向依赖的 Python、8份旧 YAML、1份旧 JSON 和2份旧说明。
- 当前 CARTS 源码、测试和配置继续按当前任务定向读取，不因本轮状态更新遍历或恢复旧下游实现。
- 第二轮删除63个文件、44763行、约1.84 MB；冻结合同的478 KB Python 数据字面量改为约3 KB加载器加42.7 KB压缩包，解压前后10组冻结值规范哈希一致。
- 第二轮非 CARTS 套件实测 `2494 passed in 100.42s`；冻结合同/安装树与方法合同定向测试 `176 passed`；最新 CARTS 完整套件 `501 passed in 439.27s`。均为静态证据。
- Git 回退点依次为 `pre-contract-cleanup-20260821`、`post-contract-cleanup-20260821`、`current-carts-j599-checkpoint-20260821` 和 `post-deep-cleanup-20260821`；本地检查点 `259de95` 保存最新 CARTS 静态集成，尚未推送远端。
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
