# GraspGenX-CARTS 方法说明

## 白话方法

GraspGenX-CARTS 不把神经网络分数当作“抓得住”的证明。官方预训练模型先围绕允许表面 ROI 点云提出完整六维手掌姿态，随后用 KCG 三指手的真实关节关系、指腹面、完整对象网格、整手网格、桌面和任务载荷逐层淘汰错误姿态。只有完成三指允许接触、闭合路径、机械臂逆解与完整路径检查的候选，才可能进入 Isaac；动态结果还要独立检查离桌、50 mm、2 s、滑移和穿透。

## 问题定义

给定连接器真实登记网格、任务坐标系、允许/禁止接触面、质量/质心/惯量/摩擦范围，以及同一 KCG 三指手模型，寻找手掌六维位姿和离散预构型，使三块真实指腹能够接触允许表面，并能在操作上限内平衡重力、抬升加速度和预登记扰动。

两个对象使用同一生成模型、5 个对象无关预构型、随机种子 `20260824`、候选预算、12 N 仿真研究操作上限、碰撞规则和排序规则。

## 输入

- 对象 A：`current_d38999_26kj61sn_public_spec` 的完整登记网格和物性。
- 对象 B：`te_deutsch_d38999_26fj35pn_step` 的完整登记网格和物性。
- 每个对象唯一的允许面索引集合；其身份由面数、索引和 SHA-256 绑定。
- KCG 手 URDF、mimic 关系、关节限位、注册碰撞网格和 PAD 源面。
- 官方 GraspGenX commit、checkpoint、5 个固定描述器及其坐标变换。
- 同一任务、误差、动态与安全配置。

## 输出

- 每个对象最多 256 个去重后的六维候选及来源身份。
- 每个候选的三指闭合、桌面扫掠、整手碰撞和任务载荷结果。
- `research_task_candidates`：12 N 下名义重力与抬升可平衡，但还不是可执行候选。
- `formal_task_candidates`：同时通过预登记误差/扰动任务评价，但还不是动态或正式通过。
- `executable_candidates`：只有机械臂逆解、接近和抬升路径也闭环后才能写入；当前为空。
- Isaac 运行后才可能产生高度、离桌、保持、滑移、姿态和穿透证据。

## 算法流程

```text
真实对象和 KCG 手
        ↓
官方 GraspGenX + 5 个固定预构型提出六维姿态
        ↓
点云可见性只审计；每描述器按分数保留最多 128 个
        ↓
跨描述器六维去重和多样性合并，限制为每对象 256 个
        ↓
三指按真实顺序和 0.0015 rad 控制步预测首次允许 PAD 接触
        ↓
每个 0.0015 rad 控制步检查整手—桌面
        ↓
离散 FCL 检查非相邻自碰和非 PAD—对象碰撞
        ↓
名义重力+抬升任务门（研究）
        ↓
16 个固定 Sobol 误差场景与任务方向（正式任务门）
        ↓
机械臂 IK、接近/抬升路径、Top-3 严格检查
        ↓
Isaac 逐指接触、离桌、50 mm、保持至少 2 s
```

## 伪代码

```text
load official model once
for descriptor in frozen_descriptor_library:
    proposals += GraspGenX(object_points, descriptor, frozen_seed)

    keep at most 128 by (-model_score, raw_index)
    record open/half point-cloud visibility as audit only

candidates = descriptor_aware_deduplicate_6d_and_limit(proposals, 256)
for candidate in candidates:
    contacts = predict_sequential_first_pad_contacts(candidate, max_joint_step=0.0015)
    if contacts are not three allowed PAD contacts: reject
    if any sampled closure state intersects table: reject
    if non-PAD/object or nonadjacent self collision: reject

    nominal = solve_task_allocation(center_parameters, 12 N)
    if nominal base load cannot balance: diagnostic only
    robust = solve_same_allocation(all preregistered error rows)
    rank nominal and robust lists lexicographically

for candidate in task_top3:
    require arm IK, approach/lift path and bound preflight
    only then run Isaac and evaluate physical motion afterward
```

## 任务力学与排序

每个接触力必须位于多边形摩擦锥内，单指法向力不得超过 12 N 仿真研究操作上限，关节约束沿用登记模型。载荷放大倍数 `lambda=1` 表示恰好覆盖规定的任务载荷；大于 1 才有余量。

名义研究门额外使用一次全部误差参数为中心值的求解。它只回答“当前接触能否在 12 N 下平衡名义重力与 minimum-jerk 抬升加速度”，不包含完整误差鲁棒结论。

固定字典序依次为：最差误差场景余量、较差 20% 场景平均余量、最大单指力、最大关节/腕部利用率、路径最小间隙、误差敏感性、GraspGenX 分数、候选 ID。不同单位不加权相加；GraspGenX 分数只打破物理指标之后的并列。

## 复杂度

设阈值后原始提案数为 `R`、保留候选数为 `N≤256`，每个候选三指闭合离散状态总数为 `S`，注册手部碰撞三角面和对象查询成本由 FCL/BVH 记为 `C_mesh`，误差场景数为 `U=16`。分数截断和六维去重约为 `O(R log R + N²)`；候选物理筛选约为 `O(N·S·C_mesh)`，任务评价约为 `O(N_survive·(U+1)·C_LP)`。昂贵严格检查只服务最多 3 个候选。

## 可迁移条件

- 新对象只需真实网格、任务轴、允许/禁止面、物性和初始场景位姿。
- 手描述器库、GraspGenX 权重、候选预算、闭合顺序、碰撞容差、12 N 和动态参数不随对象改变。
- 对象 B 的结果不能反向修改候选、描述器、摩擦、接近高度或阈值。
- 如果新手的模型条件配置箱显著超出已审计官方资产配置范围，只能报告为域外推假设；没有完整训练数据时，不得把它升级为训练分布边界，也不能把生成失败解释为手本身不可抓。

## 对照与消融

1. 旧轴对称 384 种子生成器 vs GraspGenX 六维提案：比较六维覆盖、真实闭合、完整路径和动态结果。
2. 单一名义描述器 vs 固定 5 描述器库：比较方向与预构型多样性、路径安全率和最佳任务余量。
3. GraspGenX 分数排序 vs 任务载荷字典序：比较 Top-3 组成和 Isaac 物理结果。
4. 名义任务门 vs 16 场景鲁棒门：区分研究观察资格与正式鲁棒证据。

本轮已有第 1 项的离线覆盖对照，以及“点云可见性硬门 vs 审计-only”和“25 点粗闭合 vs 控制步长闭合”两项方法接线消融；没有动态成功，因此不能完成路径或动态优劣结论。

## 当前限制

- KCG 描述器 open 条件箱的 `extents[1]` 为 `0.117526–0.305499 m`；本轮扫描的 32 个官方程序化配置中 open/half-open 同分量最大为 `0.067305 m`，26 个运行时描述器的 open 最大为 `0.060 m`。这是模型条件配置箱的尺度对照，不是 `points.json` 整手点云跨度或真实可达工作区；它支持但不证明域外推或失败因果。
- 当前每描述器最高分保留后，A/B 适配池分别为 256/231 个候选且多描述器六维覆盖通过；控制步长闭合的三指允许接触幸存数均为 0。旧混合参数的 A `256→3→桌面安全0`、B `256→闭合0` 只作消融，不代表当前生产池。
- 官方 scene-PC 粗筛前重放当前全部 256/231 个候选仍为闭合 0，故当前主因不是粗筛误删；任务、IK 和路径尚未到达。
- 当前描述器 `base_rotation` 带平移，不符合官方字段的纯旋转语义；sweep-only API 未读取它，所以现有结果不受影响。未来使用官方 mesh/viewer 路径前必须拆分该字段并重新绑定。
- 当前没有任务评价、机械臂 IK、Isaac 接触、离桌、50 mm 或 2 s 证据。
- 旧动态配置含尚无来源的 `2 mm` 对象—桌面事后接受阈值；本路线未执行该门，未来动态前必须先完成分阶段容差与来源审查，不能让它支持“无穿透”。
- 离散控制步碰撞检查是快速失败关闭，不是状态间连续数学证明。
- 不能声称“新算法”“显著优于”“鲁棒成功”“跨对象动态成功”或硬件有效。

## 官方依赖

- GraspGenX 官方仓库：<https://github.com/NVlabs/GraspGenX>
- 官方论文：<https://arxiv.org/abs/2606.00998>
- 官方 checkpoint：<https://huggingface.co/adithyamurali/GraspGenXModel>
- 官方 gripper descriptions：<https://huggingface.co/datasets/adithyamurali/gripper_descriptions>

本项目没有发明或训练 GraspGenX；本项目适配贡献只可能是固定多预构型描述器、真实 PAD/完整闭合筛选、任务字典序重排和现实可观测顺序接触控制，且仍需动态数据支持。
