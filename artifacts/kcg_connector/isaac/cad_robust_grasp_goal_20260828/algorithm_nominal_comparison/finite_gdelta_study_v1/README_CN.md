# 无指甲三指有限抓法算法：双型号名义结果与冻结说明

## 结论先行

**只给定连接器 CAD、无指甲三指手模型和冻结物理约束，当前算法已经为两款连接器自动产生抓法；两套自动抓法都在 Isaac Sim 中形成三块完整指腹合法接触、离桌、抬升超过 50 mm，并保持 2 s。**

这支持的准确结论是：

> 在下面明确声明的轴对齐有限抓取集合 \(G_\delta\) 内，按名义“保持与竖直抬升任务承载倍率”完整评价后得到的最佳抓法，已经在两个参与开发的型号上获得 simulation-only 动态成功。

它不支持以下结论：

- 不是无限连续抓取空间的绝对全局最优；
- 不是硬件验证；
- 不是严格 TE 盲测，因为 TE 数据已经参与过通用方法修正；
- 不是正式鲁棒性结论；
- 四次动态评价的 `formal_dynamic_pass` 仍为 `false`。

## 实际动态结果

| 型号与抓法 | 三块完整指腹 | 离桌 | 抬升 mm | 保持 s | 滑移 mm | 姿态变化 ° | 最大手指力矩 Nm | 未授权接触 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前型号 baseline | 是 | 是 | 55.732 | 2.000 | 0.407 | 4.422 | 0.656 | 0 |
| 当前型号有限集最佳 | 是 | 是 | 55.430 | 2.000 | 0.097 | 5.029 | 0.661 | 0 |
| TE baseline | 是 | 是 | 55.379 | 2.000 | 0.430 | 0.078 | 0.644 | 0 |
| TE 有限集最佳 | 是 | 是 | 54.660 | 2.000 | 2.097 | 2.457 | 0.649 | 0 |

两个自动抓法运行中，连接器保持阶段均未重新接触桌面，控制器完成全部动作，没有错误手—物、手—桌、手—夹具或未分类接触。当前型号自动抓法稳定后桌面穿透为 0.000643 mm；TE 为 0.000206 mm。

必须正视一个反例：TE 自动抓法的解析任务承载倍率比 baseline 高，但 Isaac 中的滑移和姿态变化明显更大。当前数学目标因此只能说明“对所建名义受力模型更优”，不能说明每个动态指标都更优。这正是算法冻结后鲁棒性阶段要检验的模型缺口。

动态原始评价：

- [当前型号 baseline](../../nailfree_current_com_height_preload050_lift056_regression_run02/grasp_lift/evaluation.json)
- [当前型号有限集最佳](../current_auto/grasp_lift_generic_route_run02/evaluation.json)
- [TE baseline](../../te_d38999_26fj35pn_final_shared_method_regression_run01/grasp_lift/evaluation.json)
- [TE 有限集最佳](../te_auto/grasp_lift_safe_ik_run03/evaluation.json)

四次运行的旧“登记抬升峰值加速度一致性”条件都没有闭合，因此程序顶层正式状态仍为 false。这个状态限制结论强度，但不能推翻直接观测到的完整指腹接触、离桌、抬升高度和保持时间。

## 文献如何实际改变了算法

本轮没有复刻多个完整抓取系统。采用或排除每项方法的依据如下。

| 原始工作 | 它解决的问题与核心思想 | 对本工程的实际影响 |
|---|---|---|
| [Ferrari & Canny, *Planning Optimal Grasps*, ICRA 1992](https://people.eecs.berkeley.edu/~jfc/papers/92/FCicra92.pdf) | 用接触原始力/力矩构造抓取力空间，力闭合表示能平衡任意外部力和力矩，并讨论受手指力上限约束的质量。 | 保留六维接触力矩矩阵和力闭合诊断；没有用各向同性最坏方向距离作为主分数，因为本任务明确是重力保持和竖直抬升。 |
| [Borst, Fischer & Hirzinger, *Grasp Planning: How to Choose a Suitable Task Wrench Space*, ICRA 2004](https://robotic.de/fileadmin/robotic/borst/Borst-ICRA2004-TaskWrenchSpace.pdf) | 指出力和力矩直接混合会带来尺度问题，应由真实任务定义需要抵抗的力矩空间。 | 不建立任意加权综合分数；直接把保持和抬升需要的外力写入约束，分别求最大承载倍率。 |
| [Fakhari et al., *Computing a Task-Dependent Grasp Metric Using Second-Order Cone Programs*, IROS 2021](https://arxiv.org/abs/2104.12158) | 用凸优化评价指定运动方向，能够同时包含重力、每个接触的不同力上限和关节力矩约束。 | 成为主质量指标的直接依据。当前使用八边摩擦锥的保守多面体近似，因此用现有 LP，而没有为了形式复杂改用 SOCP。 |
| [Li et al., *FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric*](https://alberthli.github.io/frogger/) 与[官方代码](https://github.com/alberthli/frogger) | 用可微的 min-weight 力闭合近似快速连续优化接触位置。 | 只借鉴可解释的力闭合 LP 诊断；没有采用连续非线性优化，因为当前研究问题要求完整评价预先冻结的有限集合，局部连续优化不能提供这项有限域覆盖结论。 |
| [Charusta et al., *Independent Contact Regions based on a Patch Contact Model*, ICRA 2012](https://doi.org/10.1109/ICRA.2012.6225325) | 把可变形指腹视为接触区域，并用区域容忍手指定位误差。 | 直接拒绝“两个三角片代表整个指腹”的旧错误；碰撞和接触均使用用户确认的整块蓝色指腹，质量计算保留每个接触片中的全部 CAD/FCL witness。 |
| [Zheng & Qian, *Coping with the Grasping Uncertainties in Force-closure Analysis*, IJRR 2005](https://journals.sagepub.com/doi/10.1177/0278364905049469) 与 [PONG](https://arxiv.org/abs/2309.16930) | 分别研究摩擦/接触位置不确定性和表面法向不确定性如何使名义力闭合失效。 | 它们不进入本次名义排名，避免在没有动态扰动证据前堆鲁棒优化；它们决定下一阶段必须显式扰动摩擦、位置和法向，并用最坏工况而非名义分数评价。 |

## 数学目标

对第 \(j\) 个 CAD/FCL 接触 witness，\(c_j\) 是连接器坐标系中的接触点，\(n_j\) 是 CAD 外法向。取切向正交基 \(t_{j1},t_{j2}\)，摩擦系数使用对象合同下界 \(\mu=0.45\)。八边摩擦锥生成元为：

\[
b_{jk}=-n_j+\mu\left(\cos\phi_k\,t_{j1}+\sin\phi_k\,t_{j2}\right),
\qquad \phi_k=\frac{2\pi k}{8}.
\]

接触力写成非负组合：

\[
f_j=\sum_k \alpha_{jk} b_{jk},\qquad \alpha_{jk}\ge 0.
\]

白话解释：力只能从指腹推向连接器，不能“拉住”连接器；切向力必须位于摩擦锥内。

接触对质心 \(o\) 产生的六维力和力矩为：

\[
w_{jk}=\begin{bmatrix}
b_{jk}\\
(c_j-o)\times b_{jk}
\end{bmatrix}.
\]

对保持和抬升阶段分别构造真实任务载荷 \(d_s\)。抬升加速度来自冻结的 50 mm 竖直轨迹，不人为添加横向“万能扰动”。每个阶段解一个 LP：

\[
\begin{aligned}
\max_{\alpha,\lambda_s}\quad & \lambda_s \\
\text{s.t.}\quad
& W\alpha=\lambda_s d_s,\\
& \alpha\ge0,\\
& \sum_{j\in\text{pad }i,k}\alpha_{jk}\le 8\ \text{N},\quad i=1,2,3,\\
& -\tau_{\max}\le T\alpha\le\tau_{\max}.
\end{aligned}
\]

其中 \(T\) 由完整手运动学的接触雅可比得到。名义抓取质量定义为：

\[
\rho_{\mathrm{nom}}(g)=\min\{\lambda_{\mathrm{hold}},\lambda_{\mathrm{lift}}\}.
\]

- \(\rho_{\mathrm{nom}}=1\)：解析模型刚好抵抗规定任务；
- \(\rho_{\mathrm{nom}}>1\)：在当前接触和力矩上限下仍有承载余量；
- 它不是动态滑移、碰撞或不确定性的替代品，候选仍必须单独通过全部硬约束和 Isaac 实验。

没有把 clearance、力矩余量、滑移和姿态变化随意加权成一个总分。主排序只有 \(\rho_{\mathrm{nom}}\)；数值相同才依次用禁止碰撞间隙、剩余闭合行程和固定网格次序打破平局。

## 冻结有限集合 \(G_\delta\)

本轮只声明轴对齐、无横向偏移、无倾角的有限集合：

\[
G_\delta=P_\delta\times\Theta_\delta\times Z_\delta.
\]

| 维度 | 冻结定义 |
|---|---|
| 手掌关节 \(P_\delta\) | 0 至 1.5 rad，步长 0.1 rad，并包含上限 1.57 rad，共 17 点 |
| 连接器轴向转角 \(\Theta_\delta\) | 0° 至 345°，步长 15°，共 24 点 |
| 当前型号轴向截面 \(Z_\delta\) | 由允许接触 CAD 的轴向范围和质心锚点自动产生，约 1 mm 步长并含两个端点，共 34 点 |
| TE 轴向截面 \(Z_\delta\) | 同一规则从 TE CAD 自动产生，共 33 点 |
| 横向偏移、倾角 | 本版固定为 0；因此结论不覆盖有横移或倾斜的连续抓法 |

候选总数：当前型号 \(17\times24\times34=13{,}872\)；TE \(17\times24\times33=13{,}464\)。没有使用对象对称性合并候选，所有 24 个转角都实际评价。

每个网格点由连接器该轴向 CAD 截面和完整指腹运动学自动得到径向手—物位置、三指闭合角和三指分配；没有输入对象专用世界位姿、接触坐标或关节角。

每个成员依次经过以下硬判据：

1. CAD 截面与三块完整指腹能否产生预抓和顺序闭合；
2. 51 个接近采样点以及按 Isaac 指速和物理步长离散的完整闭合路径；
3. 完整手—物、手—桌和手自身碰撞；
4. 三块完整指腹是否都先接触允许的连接器表面；
5. 单向摩擦、每指 8 N 上限和关节力矩上限下是否有 \(\rho_{\mathrm{nom}}\ge1\)；
6. 机械臂能否到达预抓并沿冻结路径抬升。

本版完整评价结果：

| 型号 | 总数 | 可执行 | 生成不可行 | 路径/碰撞不可行 | 任务受力不可行 |
|---|---:|---:|---:|---:|---:|
| 当前 | 13,872 | 8,808 | 2,280 | 2,784 | 0 |
| TE | 13,464 | 8,384 | 3,192 | 1,840 | 48 |

这里的“完整”只表示每个已声明有限成员都得到上述离散评价或明确硬约束结果；不把离散路径采样扩张成连续时间碰撞证明，也不把 \(G_\delta\) 扩张成无限连续空间。

### 机械臂冗余分支

同一手—物抓取姿态可能有多个七关节机械臂解。最终抓法使用一条对象无关规则：对固定的 9 个 IK 初值求解；删除关节越限、非相邻机械臂自碰或最小间隙低于 0.1 mm 的分支；剩余分支先最大化最小归一化关节限位余量，平局时选择离 home 最短者。没有用候选编号或 TE 专用关节角决定分支。

该规则给当前型号选择的路径最小机械臂自碰间隙为 16.127 mm，最小归一化关节限位余量为 0.0783；给 TE 选择的对应值为 10.676 mm 和 0.0722。两条分支随后分别获得了上表的真实 Isaac 动态成功。

## 搜索结果与 baseline 对比

| 型号与抓法 | 预测承载倍率 | 接触/碰撞间隙 mm | 闭合余量 rad | 单位任务关节力矩余量 Nm | 单位任务指腹力余量 N |
|---|---:|---:|---:|---:|---:|
| 当前 baseline | 1.6166 | 0.8149 | 0.2295 | 0.3433 | 3.0514 |
| 当前有限集最佳 | 4.3253 | 0.4234 | 0.2297 | 0.6919 | 6.1504 |
| TE baseline | 1.3673 | 0.8468 | 0.2297 | 0.2417 | 4.3842 |
| TE 有限集最佳 | 2.5038 | 1.1085 | 0.2372 | 0.5405 | 4.8049 |

CAD 接触位置、外法向、源面号和八边摩擦约束的逐点数据见 [contact_witnesses.csv](contact_witnesses.csv)。自动抓法使用搜索结果中原样保存的全部 witness；baseline 原搜索为了缩小文件只保存了质心和数量，因此图和 CSV 中的 baseline 逐点法向是对那个唯一冻结格点的确定性离线重算，三块接触数量与原记录逐一一致。

这些点和法向是 CAD/FCL 预测，不是 Isaac 实测接触坐标。Isaac 动态评价只直接证明三块完整指腹身份、接触计数、运动和安全结果。

## 真实数据图

- [CAD、三指接触 witness、外法向与八边单向摩擦锥](cad_contacts_normals_friction.png)
- [冻结搜索空间与候选质量分布](gdelta_search_and_quality_distribution.png)
- [baseline 与有限集最佳的独立指标对比](baseline_vs_finite_best.png)
- [两个型号自动抓法的真实 Isaac 最终保持帧](isaac_two_model_final_hold.png)
- [完整数值对比 CSV](comparison.csv)
- [紧凑机器可读摘要](summary.json)

CAD 图为了可读性只对显示用三角形做确定性抽样；所有碰撞、接触和受力计算使用完整 CAD。每个接触 witness 的八条摩擦射线都来自与求解器相同的 \(\mu=0.45\) 和八边生成公式。

## 可复现命令

### 1. 从 CAD 完整评价冻结 \(G_\delta\)

完整搜索分别约需几十分钟，输出目录必须不存在：

```bash
REPO=/home/noob/WorkPlace/kcgtest1
PY="$REPO/.venv/bin/python"

env PYTHONPATH="$REPO/src/kcg_connector" "$PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/finite_cad_search.py" \
  --object-id current_d38999_26kj61sn_public_spec \
  --config "$REPO/src/kcg_connector/config/carts_grasp_v2.yaml" \
  --output-directory /tmp/current_finite_gdelta

env PYTHONPATH="$REPO/src/kcg_connector" "$PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/finite_cad_search.py" \
  --object-id te_deutsch_d38999_26fj35pn_step \
  --config "$REPO/src/kcg_connector/config/carts_grasp_v2.yaml" \
  --output-directory /tmp/te_finite_gdelta
```

### 2. 复现自动抓法的 Isaac 动态运行

以下以当前型号为例；TE 只需替换对象 ID、selected config 和输出目录。预抓运行应先完成，再把它的评价文件绑定给抓取运行。

```bash
REPO=/home/noob/WorkPlace/kcgtest1
ISAAC_PY=/home/noob/WorkPlace/isaacsim/.conda-env/bin/python
ROBOT="$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/nailfree_three_finger_direct_v2_shrinkwrap/handarm_nailfree_three_direct.usda"
CFG="$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/finite_gdelta_v2/current_run05/selected_config.yaml"

env PYTHONPATH="$REPO/src/kcg_connector" "$ISAAC_PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/run_grasp_lift.py" \
  --mode preflight \
  --object-id current_d38999_26kj61sn_public_spec \
  --config "$CFG" \
  --robot-asset "$ROBOT" \
  --output-directory /tmp/current_auto/preflight \
  --omit-trace-json

env PYTHONPATH="$REPO/src/kcg_connector" "$ISAAC_PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/run_grasp_lift.py" \
  --mode grasp-lift \
  --object-id current_d38999_26kj61sn_public_spec \
  --config "$CFG" \
  --robot-asset "$ROBOT" \
  --preflight-evaluation /tmp/current_auto/preflight/evaluation.json \
  --output-directory /tmp/current_auto/grasp_lift \
  --capture-visual-evidence \
  --omit-trace-json
```

### 3. 只读重建本目录图表

```bash
REPO=/home/noob/WorkPlace/kcgtest1
env PYTHONPATH="$REPO/src/kcg_connector" "$REPO/.venv/bin/python" \
  "$REPO/src/kcg_connector/isaac/carts_v2/render_finite_gdelta_evidence.py" \
  --config "$REPO/src/kcg_connector/config/carts_grasp_v2.yaml" \
  --current-search "$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/finite_gdelta_v2/current_run05/search_result.json" \
  --te-search "$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/finite_gdelta_v2/te_run01/search_result.json" \
  --current-baseline-eval "$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/nailfree_current_com_height_preload050_lift056_regression_run02/grasp_lift/evaluation.json" \
  --current-auto-run "$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/algorithm_nominal_comparison/current_auto/grasp_lift_generic_route_run02" \
  --te-baseline-eval "$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/te_d38999_26fj35pn_final_shared_method_regression_run01/grasp_lift/evaluation.json" \
  --te-auto-run "$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/algorithm_nominal_comparison/te_auto/grasp_lift_safe_ik_run03" \
  --output-directory /tmp/finite_gdelta_study
```

## 名义算法冻结边界

本说明生成后，名义算法冻结为 `COMPLETE_AXIS_ALIGNED_FULL_PAD_GRID_V2`：冻结 \(G_\delta\)、完整指腹语义、路径采样、硬碰撞判据、\(\mu=0.45\) 八边摩擦锥、每指 8 N 上限、关节力矩上限、保持/抬升承载倍率和通用机械臂冗余分支规则。

后续正式鲁棒性实验不允许悄悄修改这些定义后仍沿用本版最优性结论。若真实扰动失败证明某个通用假设需要修改，应保留本版失败数据、只改一个有因果依据的因素、重新冻结并同时复验两个型号。

当前最早科学未知量已从“自动抓法能否执行”转为：

> 为什么 TE 抓法的名义解析承载倍率提高了，但动态滑移和姿态变化变差；在位姿、摩擦、质量、质心和执行误差的预先声明边界内，哪一个变量最先造成失败？

因此下一步是算法冻结后的单变量扰动，不是继续扩大 \(G_\delta\) 或调整名义抓法。
