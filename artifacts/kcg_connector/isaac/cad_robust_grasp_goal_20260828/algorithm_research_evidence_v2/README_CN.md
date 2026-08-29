# 无指甲三指抓法：有限域最优、双型号动态对比与鲁棒性失败边界

## 实际结果先行

**V5 抓法在两个型号上都由三块完整蓝色无指甲指腹形成合法接触，连接器离桌、抬升超过 50 mm、保持 2 s，错误接触为 0；名义动态安全判据均通过。**

准确结论是：

> 无指甲三指名义抓取在当前 D38999/26KJ61SN 和
> TE/DEUTSCH D38999/26FJ35PN 两个型号上动态成功。

| 型号，V5 抓法 | 三块完整指腹 | 离桌 | 抬升 mm | 保持 s | 滑移 mm | 姿态变化 ° | 峰值抬升加速度 m/s² | 动态安全 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前 D38999/26KJ61SN | 是 | 是 | 55.831 | 2.000 | 0.399 | 4.772 | 0.011164 | 通过 |
| TE D38999/26FJ35PN | 是 | 是 | 55.770 | 2.000 | 0.509 | 0.156 | 0.014645 | 通过 |

两次运行使用相同的冻结控制器和 0.020832 m/s² 抬升加速度安全上限。
原始评价分别在：

- [当前型号 V5 名义运行](../robustness_v5_development/payload_feedforward_v2/current/nominal/grasp_run01/evaluation.json)
- [TE V5 名义运行](../robustness_v5_development/payload_feedforward_v2/te/nominal/grasp_run01/evaluation.json)

这两次名义运行的原始命令和输入路径可以恢复，但评价器当时的源码哈希没有对应 Git 提交，因而不能
保证位级复现历史 JSON。冻结后的 held-out 运行绑定到提交 `5612883`，可以恢复相同源码和配置。

结论边界：全部结果均为 **simulation-only**，不是硬件验证；TE 数据在方法开发期间已经被观察过，
所以是跨型号验证，不是严格盲测；当前结果不是连续抓取空间的绝对全局最优，也不是正式鲁棒性证书。

## 公平算法对比

V4 是冻结有限集合中最大化名义任务承载倍率的抓法。V5 不再追求最高名义承载倍率，而是在同一
有限集合中优先保留扰动后的禁止接触间隙，再比较最坏任务承载倍率。下表四次 Isaac 运行使用相同
控制器、对象属性、完整指腹语义和安全判据，因此 V4 与 V5 可以直接比较。

| 型号与抓法 | 预测任务承载倍率 | 禁止接触余量 mm | 关节力矩余量 N·m | 指腹法向力余量 N | 抬升 mm | 保持 s | 滑移 mm | 姿态变化 ° | 峰值加速度 m/s² | 动态安全 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前 V4 名义最优 | 1.9678 | 0.5426 | 0.4426 | 5.4798 | 55.774 | 2.000 | 0.500 | 5.300 | 0.036026 | **未通过** |
| 当前 V5 鲁棒选择 | 1.3693 | 1.9115 | 0.2427 | 4.6427 | 55.831 | 2.000 | 0.399 | 4.772 | 0.011164 | **通过** |
| TE V4 名义最优 | 1.8780 | 0.7510 | 0.4208 | 5.3458 | 54.644 | 2.000 | 0.590 | 0.216 | 0.059502 | **未通过** |
| TE V5 鲁棒选择 | 1.2816 | 0.9136 | 0.1977 | 4.4311 | 55.770 | 2.000 | 0.509 | 0.156 | 0.014645 | **通过** |

四次运行都完成了三指合法接触、离桌、50 mm 和 2 s，错误接触均为 0。V4 的失败不是抓不住，
而是抬升瞬态加速度越过冻结安全上限。数据支持“V5 在这四次同条件运行中同时通过两个型号的名义
动态安全”，但不能单凭四次运行证明更大的几何间隙必然造成更低加速度。

原始公平评价：

- [当前 V4](../algorithm_fair_comparison_v1/v4_nominal/current/grasp_run01/evaluation.json)
- [TE V4](../algorithm_fair_comparison_v1/v4_nominal/te/grasp_run01/evaluation.json)
- [当前 V5](../robustness_v5_development/payload_feedforward_v2/current/nominal/grasp_run01/evaluation.json)
- [TE V5](../robustness_v5_development/payload_feedforward_v2/te/nominal/grasp_run01/evaluation.json)

### 历史手工 baseline 只能作描述

| 型号，历史手工抓法 | 预测任务承载倍率 | 禁止接触余量 mm | 抬升 mm | 保持 s | 滑移 mm | 姿态变化 ° | 峰值加速度 m/s² |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前 | 1.2773 | 0.8149 | 55.732 | 2.000 | 0.407 | 4.422 | 0.013233 |
| TE | 0.5552 | 0.8468 | 55.379 | 2.000 | 0.430 | 0.078 | 0.046431 |

这两次历史运行使用更早的控制器，不能与上表作控制变量齐全的因果比较。它们只证明手工名义抓法
曾真实抓起并保持，不能用于声称自动算法在所有动态指标上胜过手工抓法。

## 从搜索矛盾到可执行算法

最早的算法矛盾是：Isaac 中已经成功的抓法被旧离线入口判为碰撞。解决办法不是扩大搜索，而是把
已知成功抓法送入同一几何入口，依次对齐坐标、完整指腹碰撞、闭合路径和接触受力语义。最终规则是：

- 合法接触区域是每根**完整蓝色指腹**，不是两个三角片；
- 手—物、手—桌、手自碰和闭合路径使用完整几何；
- 一块指腹上的多个 FCL 碰撞 witness 只表示一个接触斑，不能各自获得一套独立力上限；
- 任务受力模型把每块接触斑压缩成一个合力点，不允许凭空加入独立扭转力矩；
- 动态闭合逐指达到离线所需力矩后停止，最大预紧增量仍为 0.075 rad，实测保护上限仍为 0.90 N·m。

这里没有把整块指腹缩成一个小面。几何和语义层始终使用整块指腹；只有静态受力层把已检测到的
接触斑近似为一个合力：

\[
c_i=\frac{1}{m_i}\sum_{j=1}^{m_i}c_{ij},\qquad
n_i=\frac{\sum_j n_{ij}}{\left\|\sum_j n_{ij}\right\|}.
\]

这个 witness 位置与法向的算术平均是工程建模假设，不是论文证明的软指腹等效定律；它不表示真实
压力分布、柔顺变形或接触斑扭转摩擦。

## 数学目标

对第 \(i\) 块指腹的合力点 \(c_i\)，取连接器 CAD 外法向 \(n_i\) 和两个切向基
\(t_{i1},t_{i2}\)。使用摩擦合同下界 \(\mu=0.45\) 的八边内接锥：

\[
b_{ik}=-n_i+\mu\left(\cos\phi_k\,t_{i1}+\sin\phi_k\,t_{i2}\right),
\qquad \phi_k=\frac{2\pi k}{8},
\]

\[
f_i=\sum_k\alpha_{ik}b_{ik},\qquad \alpha_{ik}\ge0.
\]

白话解释：指腹只能推连接器，不能拉；切向力必须留在摩擦允许范围内。八边锥是精确库仑圆锥的
保守多面体近似，不是精确圆锥。

以连接器质心 \(o\) 为力矩原点，每条锥边形成六维力和力矩列：

\[
w_{ik}=\begin{bmatrix}b_{ik}\\(c_i-o)\times b_{ik}\end{bmatrix}.
\]

保持和竖直抬升阶段分别求一个线性规划：

\[
\begin{aligned}
\max_{\alpha,\lambda_s}\quad &\lambda_s\\
\text{s.t.}\quad
&W\alpha=\lambda_s d_s,\\
&\alpha\ge0,\\
&\sum_k\alpha_{ik}\le8\ \mathrm{N},\quad i=1,2,3,\\
&-\tau_{\max}\le T\alpha\le\tau_{\max}.
\end{aligned}
\]

其中 \(d_s\) 由对象质量、重力和冻结抬升轨迹峰值加速度得到，\(T\) 由手指雅可比把接触力映射到
关节力矩。V4 唯一的名义主分数是：

\[
\rho_{\mathrm{nom}}(g)=
\min\{\lambda_{\mathrm{hold}},\lambda_{\mathrm{lift}}\}.
\]

没有把碰撞、滑移、姿态变化和力矩任意加权成一个总分。V4 只在 \(\rho_{\mathrm{nom}}\) 相同时，
才依次用禁止接触间隙、关节限位余量、闭合余量和固定网格顺序打破平局。

V5 对同一有限集合和离线可表达的冻结扰动集合 \(S\) 使用词典序选择：

\[
g_{V5}=\operatorname{lexmax}_{g\in G_\delta}
\left[
\min_{s\in S}c_{\mathrm{forbid}}(g,s),
\min_{s\in S}\rho(g,s),
\min_{s\in S}m_{\mathrm{joint}}(g,s),
\min_{s\in S}m_{\mathrm{closure}}(g,s)
\right].
\]

因此 V5 是“冻结场景下的鲁棒选择器”，不是 V4 的名义任务承载最优，也不是连续不确定集合上的
鲁棒最优证明。

## 冻结有限集合 \(G_\delta\)

\[
G_\delta=P_{\mathrm{palm}}\times Y_{\mathrm{yaw}}\times Z_{\mathrm{axial}}.
\]

| 维度 | 冻结定义 |
|---|---|
| 手掌关节 | 0 至 1.5 rad，步长 0.1 rad，另含 1.57 rad，共 17 点 |
| 绕连接器轴转角 | 0° 至 345°，步长 15°，共 24 点 |
| 当前型号轴向截面 | -2.000 至 30.500 mm，约 1 mm 步长并含端点，共 34 点 |
| TE 轴向截面 | -31.013 至 0 mm，约 1 mm 步长并含端点，共 33 点 |
| 横向偏移与倾角 | 均固定为 0 |

当前型号完整评价 \(17\times24\times34=13{,}872\) 个成员；TE 完整评价
\(17\times24\times33=13{,}464\) 个成员。这里的“完整”只指已冻结的轴对齐离散集合，
不覆盖连续姿态、横移、倾角或连续时间碰撞。

| 型号 | 生成不可行 | 路径/碰撞不可行 | 任务受力不可行 | 离线可执行 | 总数 |
|---|---:|---:|---:|---:|---:|
| 当前 | 2,280 | 3,456 | 264 | 7,872 | 13,872 |
| TE | 3,192 | 2,448 | 6,251 | 1,573 | 13,464 |

V4 名义最优：

| 型号 | 手掌关节 rad | 轴向转角 | CAD 轴向截面 mm | 名义承载倍率 | 禁止接触余量 mm |
|---|---:|---:|---:|---:|---:|
| 当前 | 1.50 | 195° | 4.992 | 1.9678 | 0.543 |
| TE | 1.40 | 225° | -5.135 | 1.8780 | 0.751 |

V5 鲁棒选择：

| 型号 | 手掌关节 rad | 轴向转角 | CAD 轴向截面 mm | 名义承载倍率 | 离线最坏承载倍率 | 名义禁止接触余量 mm | 离线最坏禁止接触余量 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前 | 1.00 | 150° | 11.992 | 1.3693 | 1.2403 | 1.911 | 1.909 |
| TE | 0.50 | 30° | -5.135 | 1.2816 | 1.0906 | 0.914 | 0.889 |

V5 离线场景包含名义、位姿 x 正负 0.17 mm、轴向角正负 2°、摩擦 0.45/0.55、质量正负 5%、
质心 x 正负 1 mm、第三指关节目标正负 0.004 rad，以及两个预先定义的组合场景。第三指预紧比例
属于动态变量，没有进入离线选择。

## 文献怎样改变了算法

| 原始工作 | 解决的问题与核心思想 | 本工程采用、排除及原因 |
|---|---|---|
| [Ferrari & Canny, *Planning Optimal Grasps*, ICRA 1992](https://people.eecs.berkeley.edu/~jfc/papers/92/FCicra92.pdf) | 用离散接触原始力构造六维抓取力空间和力闭合质量。 | 采用六维秩与原点位于离散摩擦锥凸包内部的力闭合诊断；没有采用各向同性球半径作主分数，因为本任务明确要求保持和竖直抬升。 |
| [Borst, Fischer & Hirzinger, *How to Choose a Suitable Task Wrench Space*, ICRA 2004](https://robotic.de/fileadmin/robotic/borst/Borst-ICRA2004-TaskWrenchSpace.pdf) | 用任务真正需要抵抗的力和力矩定义质量，避免随意混合量纲。 | 直接建立保持和竖直抬升任务载荷；采用八边摩擦锥。论文也指出八边近似仍有离散误差，所以不把 LP 写成精确圆锥结论。 |
| [Fakhari et al., *Computing a Task-Dependent Grasp Metric Using Second-Order Cone Programs*, IROS 2021](https://doi.org/10.1109/IROS51168.2021.9636197) | 联合任务方向、摩擦、逐接触力上限、重力和关节力矩约束。 | 采用任务承载倍率、逐指力上限和雅可比力矩约束；因为本工程已经用八边锥线性化，只用一个 LP，没有为了形式复杂再叠加 SOCP。 |
| [Zheng, *Computing the Best Grasp in a Discrete Point Set*, Autonomous Robots 2019](https://doi.org/10.1007/s10514-018-9788-4) | 区分离散点集最佳与连续空间最佳，并在离散集合中评价 wrench 能力。 | 冻结 \(G_\delta\) 后逐项穷举。没有采用论文的支撑函数剪枝；启发式不删除成员，所以只声称冻结有限域内最佳。 |
| [Zheng & Qian, *Coping with the Grasping Uncertainties in Force-Closure Analysis*, IJRR 2005](https://doi.org/10.1177/0278364905049469) | 定量研究摩擦衰减与接触位置误差怎样破坏力闭合。 | 用它确定要显式检查摩擦和接触位置余量；没有实现论文的连续接触位置鲁棒半径，因此不能声称连续鲁棒性证书。 |
| [Liu & Carpin, *Global Grasp Planning Using Triangular Meshes*, ICRA 2015](https://robotics.ucmerced.edu/sites/robotics.ucmerced.edu/files/page/documents/graspingicra2015.pdf) | 直接在 CAD 常用三角网格上生成接触点，并允许跨网格边移动。 | 采用“真实 CAD 网格是几何输入”的原则；没有采用其迭代优化，因为当前研究需要对预先冻结的有限域作完整评价，而迭代结果不能提供该完整性结论。 |
| [Charusta et al., *Independent Contact Regions Based on a Patch Contact Model*, ICRA 2012](https://doi.org/10.1109/ICRA.2012.6225325) | 把可容纳接触变化的指腹视为有限接触区域。 | 支持整块蓝色指腹语义；不支持 FCL witness 算术平均等价于真实软接触，该规则仍只标为工程假设。 |

## 真实图和机器可读数据

- [CAD、三指接触点、法向与八边摩擦约束](cad_contacts_normals_friction.png)
- [有限集合状态和可执行候选质量分布](gdelta_search_and_quality_distribution.png)
- [历史 baseline 与 V4 的描述性对比](baseline_vs_finite_best.png)
- [V4 与 V5 同控制器公平动态对比](v4_nominal_vs_v5_robust_fair.png)
- [两个型号 V4 自动抓法的真实 Isaac 最终保持帧](isaac_two_model_final_hold.png)
- [历史 baseline/V4 数值](comparison.csv)
- [V4/V5 公平数值](v4_vs_v5_fair_comparison.csv)
- [CAD/FCL 接触 witness](contact_witnesses.csv)
- [机器可读摘要](summary.json)

接触图中的点和法向是 CAD/FCL 预测，不是 Isaac 实测接触坐标；半透明网格只为显示，搜索和碰撞
使用完整 CAD。最终保持图是两次真实 V4 Isaac 运行的保存帧，标题数值绑定各自图像运行；公平对比表
使用后来在冻结共同控制器下重跑的 V4 数值，所以不能把图像标题数值替换为公平重跑数值。

## 冻结后鲁棒性：实际通过项与失败反例

冻结研究边界是：对象 x 位置 ±0.17 mm、轴向角 ±2°、摩擦系数 0.45 至 0.55、质量 ±5%、
质心 x ±1 mm、第三指关节目标 ±0.004 rad、第三指预紧比例 ±5%。除摩擦区间来自共享仿真材料
合同外，其余都是明确标注的研究假设，不是硬件标定误差。

最终未参与调参的场景在控制器冻结后预先写入两个型号的配置。实际按单变量顺序运行到第一个安全
失败后停止，结果如下；所有已运行场景均形成三块完整指腹接触、离桌、抬升超过 50 mm、保持 2 s，
错误接触为 0。

| 型号与冻结后场景 | 抬升 mm | 保持 s | 滑移 mm | 姿态变化 ° | 峰值加速度 m/s² | 0.020832 上限 |
|---|---:|---:|---:|---:|---:|---:|
| 当前，名义 | 55.831 | 2.000 | 0.399 | 4.772 | 0.011164 | 通过 |
| TE，名义 | 55.770 | 2.000 | 0.509 | 0.156 | 0.014645 | 通过 |
| 当前，x = -0.12 mm | 55.835 | 2.000 | 0.441 | 4.369 | 0.012746 | 通过 |
| TE，x = -0.12 mm | 55.784 | 2.000 | 0.541 | 0.127 | 0.018665 | 通过 |
| 当前，轴向角 = +0.75° | 55.849 | 2.000 | 0.441 | 4.255 | 0.012711 | 通过 |
| TE，轴向角 = +0.75° | 55.788 | 2.000 | 0.532 | 0.118 | 0.020053 | 通过，余量很小 |
| 当前，摩擦系数 = 0.49 | 55.306 | 2.000 | 0.694 | 5.418 | 0.024180 | **未通过** |

摩擦 0.49 的当前型号运行被原样重复一次，关键数值和时序完全一致。它不是“抓取失败”：连接器仍
离桌、达到 55.306 mm 并保持 2 s；最早失败是抬升阶段的瞬态加速度超过安全上限。因此现有证据只
支持以下结论：

> V5 在双型号名义工况和已运行的位姿单变量上动态成功，但冻结 V5 在当前型号、摩擦系数 0.49
> 的未参与调参工况上越过抬升加速度安全边界；正式鲁棒性结论不成立。

原始冻结后评价：

- [当前 x = -0.12 mm](../robustness_v5_final_validation_v2/v5/current/heldout_pose_x_minus_0p12mm/grasp_run01/evaluation.json)
- [TE x = -0.12 mm](../robustness_v5_final_validation_v2/v5/te/heldout_pose_x_minus_0p12mm/grasp_run01/evaluation.json)
- [当前 yaw = +0.75°](../robustness_v5_final_validation_v2/v5/current/heldout_yaw_plus_0p75deg/grasp_run01/evaluation.json)
- [TE yaw = +0.75°](../robustness_v5_final_validation_v2/v5/te/heldout_yaw_plus_0p75deg/grasp_run01/evaluation.json)
- [当前 μ = 0.49，第一次](../robustness_v5_final_validation_v2/v5/current/heldout_friction_0p49/grasp_run01/evaluation.json)
- [当前 μ = 0.49，精确重复](../robustness_v5_final_validation_v2/v5/current/heldout_friction_0p49/grasp_run02/evaluation.json)

### 相同工况下的优化前后对比

开发数据先定位到“接触、离桌和保持都成功，但抬升瞬态加速度越界”。随后只改变一次控制因素并在
相同的当前型号、摩擦 0.475 工况复测：

| 控制方式 | 三块完整指腹/离桌/2 s | 抬升 mm | 滑移 mm | 姿态变化 ° | 峰值加速度 m/s² | 相对冻结 V5 |
|---|---:|---:|---:|---:|---:|---:|
| 冻结 V5 控制器 | 是 | 55.759 | 0.624 | 4.266 | 0.027859 | 基准，未通过 |
| 抬升前一次性转移完整载荷 | 是 | 55.417 | 0.624 | 4.846 | 0.038233 | 更差 |
| 投影关节载荷反馈并限速 | 是 | 55.444 | 0.624 | 4.832 | 0.033604 | 更差 |

作为边界参考，同一冻结控制器在摩擦 0.45 时峰值为 0.020055 m/s²，刚好通过；摩擦 0.475 和
0.49 均失败，且响应并不随摩擦值简单单调。数据说明接触建立时序与载荷转移瞬态耦合，不能把
“摩擦越大一定越安全”当作控制规律。

两次有直接因果依据的控制修改都没有改善安全指标，反而恶化。按停止规则，控制器已恢复到冻结版本，
没有继续改名调参，也没有运行后续质量、质心、关节误差和组合未参与调参场景。因而不得声称完成了
鲁棒优化，也不得把未运行场景写成通过。

开发评价：

- [μ = 0.45，冻结 V5](../robustness_controller_development_v2/current/friction_lower_0p45/frozen_v5_grasp_run01/evaluation.json)
- [μ = 0.475，冻结 V5](../robustness_controller_development_v2/current/final_friction_0p475/frozen_v5_grasp_run01/evaluation.json)
- [μ = 0.475，一次性载荷转移](../robustness_controller_development_v2/current/final_friction_0p475/prelift_transfer_grasp_run01/evaluation.json)
- [μ = 0.475，投影关节载荷反馈](../robustness_controller_development_v2/current/final_friction_0p475/joint_load_feedback_grasp_run01/evaluation.json)

## 可复现命令

命令中的输出目录必须不存在。所有动态命令均为 simulation-only，并保持
`hardware_authorized=false`。

### 1. 重算 V4 有限集合和 V5 鲁棒选择

```bash
REPO=/home/noob/WorkPlace/kcgtest1
PY="$REPO/.venv/bin/python"
CFG="$REPO/src/kcg_connector/config/carts_grasp_v2.yaml"

env PYTHONPATH="$REPO/src/kcg_connector" "$PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/finite_cad_search.py" \
  --object-id current_d38999_26kj61sn_public_spec \
  --config "$CFG" --selection-mode nominal \
  --output-directory /tmp/current_finite_gdelta_v4

env PYTHONPATH="$REPO/src/kcg_connector" "$PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/finite_cad_search.py" \
  --object-id te_deutsch_d38999_26fj35pn_step \
  --config "$CFG" --selection-mode nominal \
  --output-directory /tmp/te_finite_gdelta_v4

env PYTHONPATH="$REPO/src/kcg_connector" "$PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/finite_cad_search.py" \
  --object-id current_d38999_26kj61sn_public_spec \
  --config "$CFG" --selection-mode robust \
  --output-directory /tmp/current_finite_gdelta_v5

env PYTHONPATH="$REPO/src/kcg_connector" "$PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/finite_cad_search.py" \
  --object-id te_deutsch_d38999_26fj35pn_step \
  --config "$CFG" --selection-mode robust \
  --output-directory /tmp/te_finite_gdelta_v5
```

### 2. 复现 V4/V5 名义或单个冻结扰动动态运行

下面以当前型号 V5 的 held-out 摩擦场景为例。TE 只替换对象 ID 和 TE V5
`selected_config.yaml`。V5 名义运行删除两处 `--robustness-scenario`；V4 公平名义运行把配置换成
对应 V4 `selected_config.yaml`，并把场景显式写成 `nominal`。每个 `grasp-lift` 必须绑定同一对象、
同一场景新生成的 preflight；评价由 runner 在进程内完成，不存在独立的 `evaluate_run.py` 命令。

```bash
REPO=/home/noob/WorkPlace/kcgtest1
BASE="$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828"
ISAAC_PY=/home/noob/WorkPlace/isaacsim/.conda-env/bin/python
ROBOT="$BASE/nailfree_three_finger_direct_v2_shrinkwrap/handarm_nailfree_three_direct.usda"
CFG="$BASE/finite_gdelta_v5_robust/current_run01/selected_config.yaml"
SCENARIO=heldout_friction_0p49

env PYTHONPATH="$REPO/src/kcg_connector" "$ISAAC_PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/run_grasp_lift.py" \
  --mode preflight \
  --object-id current_d38999_26kj61sn_public_spec \
  --config "$CFG" \
  --runtime-resources "$REPO/src/kcg_connector/config/carts_v2_isaac_runtime.json" \
  --robot-asset "$ROBOT" \
  --robustness-scenario "$SCENARIO" \
  --output-directory /tmp/current_v5_preflight \
  --omit-trace-json

env PYTHONPATH="$REPO/src/kcg_connector" "$ISAAC_PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/run_grasp_lift.py" \
  --mode grasp-lift \
  --object-id current_d38999_26kj61sn_public_spec \
  --config "$CFG" \
  --runtime-resources "$REPO/src/kcg_connector/config/carts_v2_isaac_runtime.json" \
  --robot-asset "$ROBOT" \
  --robustness-scenario "$SCENARIO" \
  --preflight-evaluation /tmp/current_v5_preflight/evaluation.json \
  --output-directory /tmp/current_v5_grasp \
  --omit-trace-json
```

这与冻结后数值运行的参数形式一致。若需要新的真实 Isaac 图片，可在 `grasp-lift` 命令中加入
`--capture-visual-evidence`；这会成为一次新的可视运行，图片及其评价必须与该新运行绑定，不能借用
现有无图公平评价的数值。安全判据未通过时 runner 会返回非零，但仍会保留评价 JSON；退出码不替代
其中的接触、离桌、抬升、保持和安全数据。

### 3. 重建本目录图表

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
  --current-auto-eval "$BASE/algorithm_fair_comparison_v1/v4_nominal/current/grasp_run01/evaluation.json" \
  --current-robust-search "$BASE/finite_gdelta_v5_robust/current_run01/search_result.json" \
  --current-robust-eval "$BASE/robustness_v5_development/payload_feedforward_v2/current/nominal/grasp_run01/evaluation.json" \
  --te-baseline-eval "$BASE/te_d38999_26fj35pn_final_shared_method_regression_run01/grasp_lift/evaluation.json" \
  --te-auto-run "$BASE/algorithm_nominal_comparison_v4/te_auto/grasp_lift_run02" \
  --te-auto-eval "$BASE/algorithm_fair_comparison_v1/v4_nominal/te/grasp_run01/evaluation.json" \
  --te-robust-search "$BASE/finite_gdelta_v5_robust/te_run01/search_result.json" \
  --te-robust-eval "$BASE/robustness_v5_development/payload_feedforward_v2/te/nominal/grasp_run01/evaluation.json" \
  --output-directory /tmp/algorithm_research_evidence_v2
```

## 最终证据边界与真实阻塞

已解决：

- 两个型号的无指甲三指名义动态抓取；
- 只输入 CAD、手模型和共同物理约束的自动抓法生成；
- 冻结 \(G_\delta\) 内的 V4 名义任务承载最优；
- 同控制器下 V4/V5 双型号动态对比；
- CAD 接触、法向、摩擦锥、质量分布、真实 Isaac 图片、公式和复现命令。

尚未解决：

- 连续抓取空间的绝对全局最优；
- 完整不确定性范围内的正式鲁棒性；
- 能降低已复现抬升瞬态加速度、且在未参与调参数据上通过的通用控制改进；
- 硬件验证。

当前最早阻塞是：冻结 V5 在当前型号、摩擦系数 0.49 时，三指接触、离桌、50 mm 和 2 s 均成功，
但抬升瞬态加速度为 0.024180 m/s²，高于 0.020832 m/s² 安全上限；两种由现有数据支持的单因素
控制修改分别升到 0.038233 和 0.033604 m/s²。继续在同一路线上改控制器已没有新的数据支持，
所以停止，而不是用更多代码维持“鲁棒优化仍在推进”的外观。
