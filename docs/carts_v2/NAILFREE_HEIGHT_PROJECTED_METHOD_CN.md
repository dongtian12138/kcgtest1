# 无指甲高度投影抓取方法

## 一句话说明

本方法先让 GraspGenX 在连接器周围提出多方向手掌位姿，再为每个手掌位姿寻找一个既不让三指手扫到有限桌面、又保留三块真实内侧抓取面接触机会的高度；投影后必须重新计算逐指接触和整手几何，最后才按 12 N 任务承载能力排序并检查机械臂逆解。

这是一条仿真研究方法，不是新算法创新性声明，也不是 Isaac 抓取、正式动态或硬件成功证明。

## 机器人实际会怎样使用这条方法

1. GraspGenX 给出连接器坐标系中的六维手掌位姿。
2. 同一手掌位姿绑定一个真实掌形角，并尝试三指各自 0、0.1、0.2 的固定预闭合组合。
3. 程序先问两个简单物理问题：整手沿完整采样动作至少要抬到多高才不会碰桌，三块允许抓取面又必须处于什么高度范围才可能碰到连接器。
4. 两个高度条件没有交集时立即淘汰；有交集时，只沿世界坐标 Z 轴把手掌移到最近的可行高度。
5. 投影后的旧接触点一律作废，重新进行三指逐指闭合、接触即停和整手网格检查。
6. 只有三指都在登记的允许内侧表面形成新接触，并通过桌面、非允许面和自碰检查，候选才进入任务受力评价。
7. 任务评价先检查 12 N 单指仿真操作上限下的重力与抬升，再检查预登记误差场景。
8. 排名后的 Top-3 只进行有界逆解和 50 mm 离散抬升路点检查；完整机械臂碰撞和 Isaac 动态仍是后续独立门。

## 输入

### 对象输入

- 两个连接器各自登记的真实三维网格、任务坐标系和冻结的桌面初始位姿；
- 允许指腹接触的对象表面标签，以及其余禁止或非抓取表面；
- 质量、质心、惯量和摩擦范围；
- 有限桌面的 X/Y 边界与桌面顶面高度。

### 三指手输入

- 真实手部关节、机械耦合、关节限位和正运动学；
- 91 个均匀掌形角，每个掌形角最多接收 64 个 GraspGenX 候选；
- 去除可拆卸指甲后的三个末节精确三角网格；
- 三块哈希绑定的 `TASK_GRIP_SURFACE`；
- 每个控制状态最大独立关节变化 0.0015 rad；
- 三指顺序闭合、各自接触即停的既定顺序。

### 任务输入

- 单指法向力仿真操作上限 12 N；
- 重力、抬升峰值加速度和预登记的小力/小转动力矩方向；
- 16 个固定 Sobol 误差场景；
- 50 mm 抬升目标和至少 2 s 保持目标；
- 固定随机种子、配置和两个对象共用的主要参数。

## 输出

每个保留候选至少输出：

- 手掌六维位姿、掌形角和三指预闭合相位；
- 原始高度、投影高度和 Z 向平移量；
- 三指重新预测的接触停止角、对象面与手侧允许面身份；
- 完整采样手部路径的桌面最小净空、对应连杆和阶段；
- 12 N 名义任务余量、最差误差场景余量、较差 20% 场景平均余量和最大单指所需力；
- 名义任务候选、误差场景候选和诊断候选三个分离列表；
- Top-3 的机械臂接近与 50 mm 抬升离散逆解路点；
- 所有未闭合检查和当前证据等级。

输出中的 `research_task_candidates` 只表示名义重力和抬升任务可平衡；`formal_task_candidates` 表示还通过了当前预登记误差场景。二者在完整机械臂路径、Isaac 和动态接触验证前都不是可执行候选。

## TASK_GRIP_SURFACE：什么表面才算有效接触

`TASK_GRIP_SURFACE` 是三根无指甲末节上允许用于连接器抓取的、具有源面索引和哈希身份的三角面集合。它的作用不是把“末节碰到物体”都算成接触，而是把手侧接触限制在三块登记内侧表面。

离线闭合预测与运行后评价执行以下约束：

- 三根手指必须各自产生一个有效接触；
- 接触必须落在对应 `TASK_GRIP_SURFACE` 上；
- 手侧表面法向、对象表面法向和当前闭合运动必须相容；
- 末节其余表面仍按非允许表面处理；
- 指甲安装面、外侧面、其他连杆或内部约束信号不能代替有效接触；
- 该语义只用于离线几何和事后评价，禁止提供给在线控制器。

当前三个登记面分别绑定 `f1Link3`、`f2Link2` 和 `f3Link3` 的无指甲源网格。绑定检查覆盖文件哈希、源面索引、三角坐标、法向、面数和三指身份；它证明语义数据没有漂移，不证明 Isaac 中已经实际接触。

## 高度协调的两个必要条件

### H_table：桌面要求的最低手掌高度

对预闭合远端、接近、预抓取、第一至第三指顺序闭合、预紧结束和抬升起点的每个登记控制状态，程序用真实正运动学变换各连杆的本地支撑盒。只有支撑盒 X/Y 与有限桌面 X/Y 相交时，该连杆才对桌面高度门产生约束。

对每个有关状态和连杆，最低手掌高度为：

`桌面顶面 + 操作净空 - 连杆支撑盒相对手掌的最低 Z`

所有有关状态中的最大值构成 `H_table`。由于支撑盒包住注册网格，这是一条保守的采样路径桌面下界；它可能淘汰靠近桌边但真实网格仍能通过的候选，不能称连续碰撞证明。

### H_reach_outer：三指允许面仍可能接触的高度外包区间

对每根手指，程序按 0.0015 rad 的控制步长扫过其剩余闭合行程，把 `TASK_GRIP_SURFACE` 的本地包围盒变换到世界坐标，并形成该指的扫掠高度区间。该区间再与对象允许表面的世界包围盒相交。

三根手指区间的共同交集构成 `H_reach_outer`。如果某指在 X/Y 上完全够不到允许对象表面，或三指高度区间没有共同部分，候选直接失败。

`H_reach_outer` 只是“仍有接触可能”的必要条件。包围盒重叠不等于三角面真实接触，因此不能用它直接生成接触点或宣布抓取成立；投影后的完整网格接触预测才决定候选是否离线幸存。

## 高度投影和重新验证

可行高度集合为：

`H_projectable = H_reach_outer ∩ [H_table, +∞)`

若集合为空，候选以“桌面安全与三指接触机会不能同时满足”失败关闭。若非空，程序把原手掌高度投影到距离最近的可行区间；距离相同则选择更高的一侧，以保留更多桌面净空。

投影只通过世界坐标左乘改变 Z 平移，不改变 X、Y、手掌方向、掌形角、对象模型或碰撞容差。

投影完成后必须重新执行：

1. 远端预闭合到目标预构型的有界路径；
2. 目标预构型到抓取位姿的完整采样接近；
3. 三指顺序闭合和各指接触即停预测；
4. 每个登记状态的桌面、非允许对象面和非相邻自碰查询；
5. 三块 `TASK_GRIP_SURFACE` 的新接触身份与停止角检查。

投影前预测的接触点、接触停止角和碰撞结果绝不复用。只有新的三接触预测与新的 FCL 快筛同时存活，投影候选才成立。

## 为什么高度门必须在每掌角 Top-8 之前

每个掌形角最多有 64 个输入候选。全部候选先经过上述高度必要条件与投影后精确重算，然后才按每角最多 8 个的预算截取。

固定顺序是：

`91角 × 每角最多64候选 → 全部高度可行性 → 投影后重新验证 → 每角Top-8 → 任务受力`

这样不会因为 GraspGenX 分数较高但高度错误的候选提前占满昂贵验证预算。Top-8 排序先看投影后的物理几何结果，再把 GraspGenX 分数作为后置确定性信息；预算之外的候选标记为后置，不冒充物理失败。

## 12 N任务评价与字典序

12 N 是本轮仿真研究操作上限，不是硬件硬极限，也不授权真实手执行。每个三接触候选先求名义重力与抬升载荷下的接触力分配；名义不平衡者只能进入诊断列表。

名义通过后，再用同一组 16 个固定误差场景改变接触位置、接触法向、物体位姿、摩擦、质量和质心。`lambda = 1` 表示刚好承受登记任务，`lambda > 1` 表示仍有余量，`lambda < 1` 表示不足。

候选不把不同单位指标加权相加，而按固定字典序排列：

1. 最大化最差误差场景任务余量；
2. 最大化较差 20% 场景的平均余量；
3. 最小化最大单指所需法向力；
4. 最小化最大关节负载利用率；
5. 最大化已检查手部路径的最小桌面净空；
6. 最小化对误差的敏感性；
7. 在物理指标并列后优先较高 GraspGenX 置信度；
8. 最后用确定性候选 ID 打破并列。

当前真实硬件关节和腕部负载上限尚未完整标定，相关利用率字段可能为未知；未知值不能被写成已通过硬件门。

## bounded IK：只回答机械臂能不能到达离散路点

Top-3 使用现有 `build_joint_motion_plan()` 和 `solve_bounded_hand_base_ik()`：

- 预抓取手形下求解 5 个接近路点；
- 以相邻解为下一路点的固定种子；
- 抓取手形下求解 11 个从当前高度到 +50 mm 的抬升路点；
- 记录位置、姿态误差和最大相邻关节步长；
- 全程使用冻结的对象初始位姿，不读取运行中的对象真值。

bounded IK 通过只证明离散位姿在关节限位内运动学可达。它没有检查完整七轴机械臂沿路碰撞、离散路点之间连续碰撞、抓取后携带物体碰撞或动力学稳定性。

## 技术流程伪代码

```text
for palm_angle in fixed_91_angle_grid:
    seeds = at_most_64_graspgenx_seeds(palm_angle)
    for seed in seeds:
        variants = fixed_27_pregrasp_phase_combinations(seed)
        shortlist = choose_at_most_2_by_contact_and_table_bounds(variants)
        for variant in shortlist:
            H_contact = swept_task_surface_reach_interval(variant)
            H_table = sampled_full_hand_path_table_lower_bound(variant)
            H_feasible = intersect(H_contact, [H_table, +inf))
            if H_feasible is empty:
                reject
            else:
                projected = project_world_Z_to_nearest_interval(seed, H_feasible)
                prediction = recompute_sequential_contact(projected)
                geometry = recompute_pregrasp_approach_closure_FCL(projected)
                keep only if three allowed contacts and geometry survive
    retain at most 8 projected survivors for this palm angle

evaluate nominal 12 N task and fixed uncertainty scenarios
select nominal, robust and diagnostic lists by lexicographic order
solve bounded approach and 50 mm lift IK for robust Top-3
```

设 `N` 为输入候选数、`V=27` 为固定预闭合组合数、`E≤2` 为每候选精确高度变体预算、`S` 为有界控制状态数。便宜预选约为 `O(NV)`，精确几何约为 `O(NES)`；每角 Top-8 使后续任务受力规模保持有界。本方法没有创建新的连续碰撞证明器。

## 证据边界和可迁移条件

当前方法可以支持的结论：

- 两对象使用同一代码、同一 91 角、同一每角预算、同一高度规则和同一 12 N操作上限；
- 某候选在已登记离散手部状态下具有三块允许面接触预测、桌面净空和任务余量；
- 某 Top-3 候选具有接近与 50 mm 抬升的离散逆解路点。

当前方法不能单独支持的结论：

- 完整机械臂路径无碰撞；
- PhysX 中无指甲碰撞形状已经正确生效；
- 三指在 Isaac 中实际接触、连接器离桌、抬升 50 mm或保持 2 s；
- 对扰动的动态鲁棒性；
- 正式动态通过或真实硬件通过；
- 相对已有文献具有创新性或显著优越性。

迁移到另一连接器时，只允许替换对象网格、任务坐标系、允许/禁止表面、质量/质心/惯量/摩擦和冻结场景位姿；不得按对象修改高度规则、掌角网格、Top-8预算、12 N上限或排序顺序。

## 关键实现与原始证据

- 高度数值内核：`height_projection.py`
- 高度候选搜索和投影后重算：`height_projected_search.py`
- 91角与 Top-8 级联：`full_palm_search.py`
- 三指接触预测：`closure_predictor.py`
- 整手快速几何：`fast_filter.py`
- 任务受力：`task_quality.py`
- 字典序：`selector.py`
- bounded IK 路点：`isaac/carts_v2/controller.py::build_joint_motion_plan`
- 生产离线入口：`scripts/carts_v2/run_graspgenx_offline.py`
- 手侧语义清单：`artifacts/carts_v2/nailfree_height_projected/task_grip_surface_audit/TASK_GRIP_SURFACE_MANIFEST.json`
- 对象 A 完整结果：`artifacts/carts_v2/nailfree_height_projected/offline_A/result.json`
- 对象 B 完整结果：`artifacts/carts_v2/nailfree_height_projected/offline_B/result.json`
- 双对象离散 IK：`artifacts/carts_v2/nailfree_height_projected/offline_kinematic_routes/`
