# 无指甲三指有限抓法 V4：双型号名义对比与冻结说明

## 实际结果

**三块完整无指甲指腹在两个型号上都形成合法接触，两个连接器都离桌、抬升超过 50 mm，并保持 2 s；本阶段没有物理失败。**

准确结论是：

> 只给定连接器 CAD、无指甲三指手模型和共同物理约束，V4 自动产生的抓法在当前连接器和 TE/DEUTSCH D38999/26FJ35PN 上均获得 Isaac Sim 名义动态成功；它是冻结有限集合 \(G_\delta\) 内的最佳抓法，不是无限连续空间的绝对全局最优。

全部结果均为 simulation-only，不是硬件验证；TE 数据此前已参与方法修正，因此只能称跨型号验证，不能称严格盲测。已有扰动运行属于先导数据，不是正式鲁棒性结论。

## 手工 baseline 与 V4 自动抓法

| 型号与抓法 | 预测任务承载倍率 | 碰撞/禁止接触余量 mm | 单位任务关节力矩余量 N·m | 三指合法接触 | 离桌 | 抬升 mm | 保持 s | 滑移 mm | 姿态变化 ° | 峰值手指力矩 N·m | 错误接触 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前型号 baseline | 1.2773 | 0.8149 | 0.1954 | 是 | 是 | 55.732 | 2.000 | 0.407 | 4.422 | 0.656 | 0 |
| 当前型号 V4 自动 | 1.9678 | 0.5426 | 0.4426 | 是 | 是 | 55.585 | 2.000 | 0.461 | 0.773 | 0.476 | 0 |
| TE baseline | 0.5552 | 0.8468 | -0.7211 | 是 | 是 | 55.379 | 2.000 | 0.430 | 0.078 | 0.644 | 0 |
| TE V4 自动 | 1.8780 | 0.7510 | 0.4208 | 是 | 是 | 54.556 | 2.000 | 0.515 | 0.172 | 0.729 | 0 |

动态原始评价：

- [当前型号 baseline](../../nailfree_current_com_height_preload050_lift056_regression_run02/grasp_lift/evaluation.json)
- [当前型号 V4 自动抓法](../current_auto/grasp_lift_run02/evaluation.json)
- [TE baseline](../../te_d38999_26fj35pn_final_shared_method_regression_run01/grasp_lift/evaluation.json)
- [TE V4 自动抓法](../te_auto/grasp_lift_run02/evaluation.json)

这组数据支持三个结论：

1. V4 的解析任务承载余量和关节力矩余量在两个型号上都高于 baseline。
2. 四次运行都完成同一物理任务，但解析分数没有保证每个动态指标都更优。当前型号的姿态变化显著减小；两个型号的滑移都比各自 baseline 略大，TE baseline 的姿态保持也仍更好。
3. TE baseline 在真实名义仿真中成功，却在使用 \(\mu=0.45\)、每指 8 N 和 0.90 N·m 上限的 V4 保守模型中得到 \(\rho<1\)。这说明模型可以用于共同约束下的候选排序，但不能反过来否定已经观测到的动态成功。

两个 baseline 的对应运行没有保存图片，因此这里只使用其原始数值，不借用其他运行图片。V4 图片则与上表两次 `run02` 成功运行严格绑定。

## 首个搜索矛盾怎样被解决

已知手工抓法能够成功，但旧搜索曾把候选判为碰撞。V4 没有扩大姿态网格来掩盖这个矛盾，而是先把已知成功抓法送回同一个几何入口，逐级对齐 Isaac 场景与离线模型。最终修正点是：

- 合法接触语义始终是用户确认的**整块蓝色指腹**，不是两个三角片；
- 手—物、手—桌、手自碰和闭合路径使用完整几何；
- 接触斑中可能有多个 FCL 碰撞 witness，但它们不是多根独立手指，也不能各自获得一整套独立力上限；
- V4 在受力模型中把每块完整指腹检测到的接触斑压缩成一个合力点，并禁止凭空加入独立扭转力矩；
- 0.075 rad 是由 0.90 N·m 实测保护上限除以 12 N·m/rad 手指刚度得到的最大预紧行程，不是固定工作点；控制器逐指达到离线所需力矩后停止闭合。

固定使用完整 0.075 rad 的 TE 名义运行曾在抬升起点因第三指约 0.97 N·m 触发保护，几乎没有离桌。换成上述逐指停止规则后，同一自动姿态达到 54.556 mm 和 2 s，峰值降到 0.729 N·m。这个对照是控制规则的直接动态依据。

## 完整指腹与单合力点的边界

必须区分两个层次：

- **几何和语义层**：每根蓝色指腹是完整有限区域；碰撞、合法性和接触 witness 都在整块区域上计算。
- **任务受力层**：每块已检测接触斑暂用一个合力点近似。其位置和法向为

\[
c_i=\frac{1}{m_i}\sum_{j=1}^{m_i}c_{ij},\qquad
n_i=\frac{\sum_{j=1}^{m_i}n_{ij}}
{\left\|\sum_{j=1}^{m_i}n_{ij}\right\|}.
\]

这里的算术平均是明确的工程建模假设，不是论文证明的真实软指腹等效定律。它不把整块指腹缩成小三角片，也不声称能表示压力分布、柔顺变形或接触斑扭转摩擦。

## 数学目标

对第 \(i\) 块指腹的合力点 \(c_i\)，取连接器 CAD 外法向 \(n_i\) 及正交切向基 \(t_{i1},t_{i2}\)。对象合同的摩擦下界为 \(\mu=0.45\)，用八边内接摩擦锥：

\[
b_{ik}=-n_i+\mu\left(\cos\phi_k\,t_{i1}+\sin\phi_k\,t_{i2}\right),
\qquad \phi_k=\frac{2\pi k}{8},
\]

\[
f_i=\sum_k\alpha_{ik}b_{ik},\qquad \alpha_{ik}\ge0.
\]

白话解释：指腹只能推连接器，不能拉；切向力不能跑出摩擦锥。

以连接器质心 \(o\) 为力矩原点，每条锥边形成六维力和力矩：

\[
w_{ik}=\begin{bmatrix}
b_{ik}\\
(c_i-o)\times b_{ik}
\end{bmatrix}.
\]

接触雅可比把接触力映射到手指关节力矩：

\[
\tau=T\alpha,
\qquad -\tau_{\max}\le T\alpha\le\tau_{\max}.
\]

保持和竖直抬升分别使用由质量、重力和冻结抬升轨迹峰值加速度得到的任务载荷 \(d_s\)，对每个阶段求一个线性规划：

\[
\begin{aligned}
\max_{\alpha,\lambda_s}\quad & \lambda_s\\
\text{s.t.}\quad
&W\alpha=\lambda_s d_s,\\
&\alpha\ge0,\\
&\sum_k\alpha_{ik}\le8\ \mathrm{N},\quad i=1,2,3,\\
&-\tau_{\max}\le T\alpha\le\tau_{\max}.
\end{aligned}
\]

名义主分数只有一个：

\[
\rho_{\mathrm{nom}}(g)=
\min\{\lambda_{\mathrm{hold}},\lambda_{\mathrm{lift}}\}.
\]

没有把碰撞余量、闭合余量、滑移和姿态变化任意加权成一个总分。只有 \(\rho_{\mathrm{nom}}\) 相同时，才依次用禁止接触间隙、关节限位余量、剩余闭合行程和固定网格顺序打破平局。

离线 LP 还输出每根闭合关节在单位任务下所需的最大力矩。动态控制在抬升前逐指闭合，直到连续样本达到该需求或达到 0.075 rad 最大行程；随后保持当时关节目标完成抬升。LP 不包含手指重力、惯性、阻尼、材料柔顺性或压力分布，因此这一步是静态预测与动态执行之间的受限接口，不是动力学证明。

## 冻结有限集合 \(G_\delta\)

\[
G_\delta=P_{\mathrm{palm}}\times
Y_{\mathrm{yaw}}\times Z_{\mathrm{axial}}.
\]

| 维度 | 冻结定义 |
|---|---|
| 手掌关节 | 0 至 1.5 rad，步长 0.1 rad，另含 1.57 rad，共 17 点 |
| 绕连接器轴转角 | 0° 至 345°，步长 15°，共 24 点 |
| 当前型号轴向截面 | 从允许接触 CAD 自动取范围，约 1 mm 步长并含端点，共 34 点 |
| TE 轴向截面 | 同一规则从 TE CAD 自动取得，共 33 点 |
| 横向偏移与倾角 | 本版均固定为 0 |

当前型号共 \(17\times24\times34=13{,}872\) 个成员，TE 共 \(17\times24\times33=13{,}464\) 个成员。所有声明成员均被评价，没有用启发式删除成员；“完整”只指这个离散集合，不是连续姿态或连续时间碰撞证明。

| 型号 | 生成不可行 | 路径/碰撞不可行 | 任务受力不可行 | 离线可执行 | 总数 |
|---|---:|---:|---:|---:|---:|
| 当前 | 2,280 | 3,456 | 264 | 7,872 | 13,872 |
| TE | 3,192 | 2,448 | 6,251 | 1,573 | 13,464 |

每个成员依次检查：完整指腹几何与手运动学、接近和顺序闭合路径、手—物/桌/自碰、三块指腹是否先接触允许表面、单向摩擦与力矩约束、机械臂预抓和抬升路径。选中结果为：

| 型号 | 手掌关节 rad | 轴向转角 | CAD 轴向截面 mm | \(\rho_{\mathrm{nom}}\) | 最小关节限位余量 rad | 最小禁止接触间隙 mm |
|---|---:|---:|---:|---:|---:|---:|
| 当前 | 1.50 | 195° | 4.992 | 1.9678 | 0.070 | 0.543 |
| TE | 1.40 | 225° | -5.135 | 1.8780 | 0.170 | 0.751 |

## 文献如何真正改变了算法

| 原始工作 | 当前问题与核心思想 | 本工程实际采用或排除的内容 |
|---|---|---|
| [Ferrari & Canny, *Planning Optimal Grasps*, ICRA 1992](https://www.research.unipd.it/bitstream/11577/2532028/1/PlanningOptimal.pdf) | 用接触原始力构造六维抓取力空间，并用凸包描述力闭合。 | 保留 \(\operatorname{rank}(W)=6\) 和原点位于离散摩擦锥凸包内部的力闭合诊断；没有用各向同性球半径作主分数，因为当前任务是明确的竖直抬升。 |
| [Borst, Fischer & Hirzinger, *How to Choose a Suitable Task Wrench Space*, ICRA 2004](https://www.robotic.dlr.de/fileadmin/robotic/borst/Borst-ICRA2004-TaskWrenchSpace.pdf) | 任务应定义需要抵抗的力和力矩，避免随意混合不同量纲。 | 直接建立保持与竖直抬升载荷，不设置任意力/力矩加权总分；八边摩擦锥是精确库仑圆锥的保守近似。 |
| [Fakhari et al., *Computing a Task-Dependent Grasp Metric Using Second-Order Cone Programs*, IROS 2021](https://arxiv.org/html/2104.12158) | 在指定任务方向下联合接触力、各接触上限和关节力矩评价抓法。 | 采用任务承载倍率、雅可比力矩和逐指力上限；因为八边锥已经把问题线性化，使用一个 LP，而没有为形式复杂改成多个优化器。 |
| [Zheng, *Computing the Best Grasp in a Discrete Point Set*, Autonomous Robots 2019](https://link.springer.com/article/10.1007/s10514-018-9788-4) | 在离散接触集合中寻找最佳抓法，并明确离散域与连续域的区别。 | 冻结 \(G_\delta\) 后直接穷举每个成员；没有采用论文的支撑函数加速，因此结论只写成冻结有限域内词典序最佳。 |
| [Zheng & Qian, *Coping with the Grasping Uncertainties in Force-Closure Analysis*, IJRR 2005](https://guppy.mpe.nus.edu.sg/~legged_group/nusbip/members/yuzheng/Coping%20with%20the%20Grasping%20Uncertainties%20in%20Force-Closure%20Analysis.pdf) | 研究摩擦衰减和接触位置误差怎样破坏名义力闭合。 | 名义阶段只采用摩擦区间下界，并报告力、力矩、碰撞、关节和闭合余量；尚未实现其连续接触位置鲁棒半径，因此不能称鲁棒性证书。 |
| [Charusta et al., *Independent Contact Regions Based on a Patch Contact Model*, ICRA 2012](https://doi.org/10.1109/ICRA.2012.6225325) | 把指腹看成可容纳接触变化的有限区域，而不是一个固定三角面。 | 支持“整块蓝色指腹是语义和碰撞区域”；不支持 V4 的 witness 算术平均规则，该规则仍明确标为工程假设。 |

FRoGGeR、PONG 等连续或概率鲁棒方法只作为后续参考，当前实现没有采用，因而不把它们列成已经实现的算法组成。

## 真实数据图与表

- [CAD、三指接触点、外法向、摩擦约束及 baseline/V4 几何对比](cad_contacts_normals_friction.png)
- [完整有限集合状态与可执行候选质量分布](gdelta_search_and_quality_distribution.png)
- [baseline 与 V4 的独立指标对比](baseline_vs_finite_best.png)
- [两个型号 V4 自动抓法的真实 Isaac 最终保持帧](isaac_two_model_final_hold.png)
- [完整数值对比 CSV](comparison.csv)
- [CAD/FCL 接触 witness CSV](contact_witnesses.csv)
- [机器可读摘要](summary.json)

CAD 图只为显示抽样网格三角形；碰撞和搜索使用完整 CAD。图中的接触点和法向是 CAD/FCL 预测，不是 Isaac 实测坐标。Isaac 运行直接验证的是完整指腹身份、接触计数、离桌、抬升、保持、滑移、姿态变化和错误接触。

## 可复现命令

### 1. 重新评价冻结 \(G_\delta\)

输出目录必须不存在；每个型号需要几十分钟。

```bash
REPO=/home/noob/WorkPlace/kcgtest1
PY="$REPO/.venv/bin/python"

env PYTHONPATH="$REPO/src/kcg_connector" "$PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/finite_cad_search.py" \
  --object-id current_d38999_26kj61sn_public_spec \
  --config "$REPO/src/kcg_connector/config/carts_grasp_v2.yaml" \
  --output-directory /tmp/current_finite_gdelta_v4

env PYTHONPATH="$REPO/src/kcg_connector" "$PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/finite_cad_search.py" \
  --object-id te_deutsch_d38999_26fj35pn_step \
  --config "$REPO/src/kcg_connector/config/carts_grasp_v2.yaml" \
  --output-directory /tmp/te_finite_gdelta_v4
```

### 2. 复现 Isaac 动态抓取

下面以当前型号为例。TE 使用对象 ID `te_deutsch_d38999_26fj35pn_step` 和 `/tmp/te_finite_gdelta_v4/selected_config.yaml`；其余控制器和安全边界相同。

```bash
REPO=/home/noob/WorkPlace/kcgtest1
ISAAC_PY=/home/noob/WorkPlace/isaacsim/.conda-env/bin/python
ROBOT="$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/nailfree_three_finger_direct_v2_shrinkwrap/handarm_nailfree_three_direct.usda"
CFG=/tmp/current_finite_gdelta_v4/selected_config.yaml

env PYTHONPATH="$REPO/src/kcg_connector" "$ISAAC_PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/run_grasp_lift.py" \
  --mode preflight \
  --object-id current_d38999_26kj61sn_public_spec \
  --config "$CFG" \
  --runtime-resources "$REPO/src/kcg_connector/config/carts_v2_isaac_runtime.json" \
  --robot-asset "$ROBOT" \
  --preload-increment-rad 0.075 \
  --output-directory /tmp/current_v4/preflight \
  --omit-trace-json

env PYTHONPATH="$REPO/src/kcg_connector" "$ISAAC_PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/run_grasp_lift.py" \
  --mode grasp-lift \
  --object-id current_d38999_26kj61sn_public_spec \
  --config "$CFG" \
  --runtime-resources "$REPO/src/kcg_connector/config/carts_v2_isaac_runtime.json" \
  --robot-asset "$ROBOT" \
  --preflight-evaluation /tmp/current_v4/preflight/evaluation.json \
  --preload-increment-rad 0.075 \
  --output-directory /tmp/current_v4/grasp_lift \
  --capture-visual-evidence \
  --omit-trace-json
```

### 3. 重建图表

```bash
REPO=/home/noob/WorkPlace/kcgtest1
BASE="$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828"

env PYTHONPATH="$REPO/src/kcg_connector" "$REPO/.venv/bin/python" \
  "$REPO/src/kcg_connector/isaac/carts_v2/render_finite_gdelta_evidence.py" \
  --config "$REPO/src/kcg_connector/config/carts_grasp_v2.yaml" \
  --current-search "$BASE/finite_gdelta_v4/current_run01/search_result.json" \
  --te-search "$BASE/finite_gdelta_v4/te_run01/search_result.json" \
  --current-baseline-eval "$BASE/nailfree_current_com_height_preload050_lift056_regression_run02/grasp_lift/evaluation.json" \
  --current-auto-run "$BASE/algorithm_nominal_comparison_v4/current_auto/grasp_lift_run02" \
  --te-baseline-eval "$BASE/te_d38999_26fj35pn_final_shared_method_regression_run01/grasp_lift/evaluation.json" \
  --te-auto-run "$BASE/algorithm_nominal_comparison_v4/te_auto/grasp_lift_run02" \
  --output-directory /tmp/finite_gdelta_study_v4
```

## 冻结边界与下一阶段

本说明对应的名义算法冻结为 `COMPLETE_AXIS_ALIGNED_FULL_PAD_GRID_V4`：冻结上述 \(G_\delta\)、完整指腹语义、每块接触斑单合力点、无独立扭转力矩、路径采样、碰撞判据、\(\mu=0.45\) 八边摩擦锥、每指 8 N、关节力矩上限、保持/抬升承载倍率、0.075 rad 最大预紧行程和逐指所需力矩停止规则。

后续扰动实验不得改动这些定义却继续沿用本版最优性结论。若单变量动态失败证明某个通用假设必须修改，应保留 V4 失败原始数据，只改一个有因果依据的因素，重新冻结，并在两个型号和未参与调参的工况上验证。

当前最早科学未知量已经从“自动抓法能否执行”转为：

> 在明确标注为仿真合同或研究假设的位姿、摩擦、质量、质心和关节执行误差边界内，哪一个单变量最先破坏合法接触、离桌、50 mm 或 2 s？

