# CARTS-Grasp V1：CAD 条件化不确定性鲁棒任务扳手三指抓取

## 1. 研究问题与声明边界

`CARTS-Grasp`（CAD-conditioned, uncertainty-aware robust task-wrench grasp synthesis）面向三指手抓取轴类电连接器。输入是可追溯 CAD/STEP 派生表面、手运动学与 PAD、物理属性及不确定性、任务扳手集合；输出是手相对对象位姿、关节配置、计划 PAD 接触、内部力和顺应参数。

本方法不读取连接器型号对应的手填 candidate、不使用旧 `CAD_*` 接触坐标、不导入 H1-H25 控制补丁。当前 public-spec D38999 与 TE/DEUTSCH J35 共享算法、风险定义、候选预算、求解精度和控制参数。对象间仅允许更换对象合同中的几何、质量、质心、惯量、材料后验和来源。

本项目已在设计前看到 TE J35 资产，因此最终只能表述为“冻结算法后的零对象调参跨模型验证”，不能声称前瞻性双盲测试。仿真结果不能升级为实体硬件结论，`hardware_authorized=false`。

## 2. 输入与决策变量

第 (j) 个对象的输入为

\[
\mathcal I_j=\{\mathcal S_j,\mathcal M_j,\mathcal H,\mathcal U_j,\mathcal T\},
\]

其中 \(\mathcal S_j\) 是 STEP/B-Rep 的确定性三角化外表面及允许/禁止功能面；\(\mathcal M_j=(m,c,I)\) 是质量、质心和惯量；\(\mathcal H\) 是 URDF、真实 PAD 表面、Jacobian、关节和执行器能力；\(\mathcal U_j\) 是位姿、表面、摩擦、质心和执行误差模型；\(\mathcal T\) 是抓取、抬升、保持与扰动任务。

规划变量为

\[
x=({}^{O}T_H,q,K,D,f_{\mathrm{int}}).
\]

分别表示手相对对象的 6D 位姿、关节配置、接触顺应刚度/阻尼和内部夹持力。对象身份、candidate ID 或历史运行编号不是决策变量。

## 3. CAD 条件化表面与顺应闭合

对象合同显式给出装配轴、单位、CAD 到对象坐标变换和功能面语义。算法从外部可接触表面提取局部曲面片；禁止承载的配合面、插针/插孔、开放螺纹和锐边由通用语义规则排除。PAD 足迹必须能在局部曲面片内放置，表面抽样密度由 PAD 尺寸和网格收敛性决定，而不是对象料号。

计划接触不被假定为必然实现。不确定状态

\[
\xi=(\Delta T_O,\Delta p,\Delta n,\mu,m,c,I,\Delta q,\Delta\tau)
\]

经顺应闭合映射产生实际接触

\[
C(x,\xi)=\Phi(\mathcal S_j,\mathcal H,x,\xi).
\]

V1 的离线接触内核是 `CARTS_SEQUENTIAL_MP_INTERVAL_FINITE_PAD_WITNESS_CLOSURE_CERTIFIER_V9`。它沿预注册的三条独立闭合关节路径运行 exact FK，并对全部有限 PAD 内部 witness 与全部对象三角面建立候选对。每个被接受的横截根都必须同时具有：根区间两端严格异号、整个区间内平面导数严格同号、三条三角形边半空间严格为正，以及不存在可能更早的禁止或未解析事件。运算由定向多精度区间外包；区间不能与零严格分离、根二分预算耗尽或竞争根顺序不能证明时一律返回 `UNRESOLVED`。Isaac 动态阶段再用真实手指顺应闭合验证，但 PhysX 接触真值只进入事后评价。

对象三角面在接触生成中按无向法线 \([n]=\{n,-n\}\) 处理，不能仅凭源面绕序称为物理外法向。令

\[
f(s)=n^T\bigl(p_P(s)-x_O\bigr),
\]

并令 \(\sigma=\operatorname{sign}f(s^-)\) 为根前局部平面侧的符号。当 \(\sigma\) 可区间证号且

\[
-\sigma\,n^T v_P>0
\]

在整个根区间严格成立时，冻结局部来向侧法向 \(n_{\mathrm{local}}=\sigma n\)。三角绕序翻转会同时使 \(n\) 与 \(\sigma\) 反号，因此 \(n_{\mathrm{local}}\) 及后续候选力锥不变。这个局部平面证书本身不证明根前整个手/对象无碰撞；只有起点在外部且完整根前缀另有连续碰撞证书时，才可升级为 `PATH_LOCAL_FREE_SIDE_NORMAL` 或实体外法向。对于开放或多壳 triangle soup，未取得该附加证据前只能声明 `MOTION_OPPOSING_LOCAL_PLANE_NORMAL`，不能升级成全局 solid-outward 结论。

PAD 自身仍使用哈希绑定源三角绕序法向 \(n_P\)，并要求 \(n_P^Tv_P\) 的定向区间严格为正。这里没有角度、距离或对齐接受阈值；binary64 前向误差或多精度区间只决定符号能否证明，不能把接近零的值判成接触。PAD 网格不是水密体，因此其字节哈希、相邻面绕序一致性和来源 manifest 必须同时绑定，旧 `pad_contact_face_ids` 与 `0.90` 对齐规则不得参与候选面选择。

V9 使用的是每个 PAD 三角形内的有限、确定性 witness 集，而不是完整 triangle-triangle CCD。它能认证所报告 witness 的首个横截接触，但不能单独排除薄片、edge-edge 或未采样面内位置的更早接触。正式全手碰撞门必须另外覆盖完整 PAD/非 PAD 表面；在该门完成前，V9 结果只能称静态闭合候选，不能称完整无碰撞抓取。

手的平移候选域也不等于对象自身 AABB。设对象在预注册 task frame 第 \(k\) 轴的区间为 \([o_k^-,o_k^+]\)，第 \(i\) 枚 PAD 在完整注册闭合路径 \(s\in[0,1]\) 上相对三指等权焦点的 swept 坐标集合，具有经 URDF 链速度上界和 binary64 前向误差认证的外包区间 \([\underline p_{ik},\overline p_{ik}]\)。要使每枚 PAD 在闭合途中都有可能与对象 AABB 重叠，焦点坐标必须属于

\[
[\ell_k,u_k]
=\bigcap_i [o_k^- - \overline p_{ik},\;o_k^+ - \underline p_{ik}],
\]

即 \(\ell_k=\max_i(o_k^- - \overline p_{ik})\)、\(u_k=\min_i(o_k^+ - \underline p_{ik})\)。实现以路径中点的 exact FK 为中心，并用从 URDF 祖先链、关节全行程和 PAD 全顶点半径推导的 Lipschitz 速度上界外包每个坐标；这里没有手填毫米扩张量。该 swept Minkowski 区间由对象 CAD、PAD 全表面、注册闭合路径、yaw 和预抓关节共同推导，允许焦点位于对象包络之外。它只是“存在路径接触”的必要 broadphase 条件，不能替代完整连续碰撞或可达性证书。周期 yaw 使用半开单位域 \([0,1)\)；若某个平移区间退化为单点，该轴只接受规范坐标 0，从而避免一对多参数化。

## 4. 可实现接触力与任务扳手裕度

对实际接触集合 \(C\)，定义 grasp map \(G(C)\)。可实现接触力集合为

\[
\mathcal F(x,\xi)=\left\{f\ \middle|\
f_{n,i}\ge0,\ \|f_{t,i}\|_2\le\mu_i f_{n,i},\
J_C(q)^Tf\in[\tau^-,\tau^+],\ f_{n,i}\le f_{n,i}^{\max}\right\}.
\]

数值实现采用摩擦锥的内接正多边形，并由允许的相对保守误差 \(\varepsilon_{\mathrm{cone}}\) 自动计算边数：

\[
1-\cos(\pi/E)\le\varepsilon_{\mathrm{cone}}.
\]

因此边数是求解精度的结果，不是物理门限；发布结果必须包含边数加倍收敛研究。对象可抵抗扳手集合为

\[
\mathcal A(x,\xi)=-G(C(x,\xi))\mathcal F(x,\xi).
\]

\(\mathcal W_0(\xi)\) 表示重力和计划加速度必需扳手。\(\mathcal D\) 是中心对称的 6D 单位扰动体：力分量以对象重量 \(mg\) 归一化，力矩分量以 \(mgr_m\) 归一化。刚体惯量均表达在整体质心处，并由恒等式

\[
\operatorname{tr}(I_C)=2\int\|r\|^2\,\mathrm dm
\]

唯一给出质量分布 RMS 半径

\[
r_m=\sqrt{\frac{\operatorname{tr}(I_C)}{2m}}.
\]

该尺度与表面三角化密度、可抓面面积和刚体坐标变换无关；不再给不同对象手填横向力、弯矩权重或名义圆柱半径。任务鲁棒裕度定义为

\[
\rho(x,\xi)=\sup\{\gamma\ge0:\mathcal W_0(\xi)\oplus\gamma\mathcal D\subseteq\mathcal A(x,\xi)\}.
\]

\(\rho\) 是连续量，不设置二元质量通过阈值。\(\rho=1\) 只表示覆盖一个归一化单位扰动体；这里的 1 来自量纲定义，而不是看结果设置的门槛。每个扰动顶点通过线性规划求最大尺度，\(\rho\) 取所有顶点中的最小值。候选预算、QMC 场景数、下尾描述比例和数值容差属于预注册的计算协议，必须做预算、场景数、摩擦锥边数与网格分辨率的收敛研究；它们不得被解释成物理安全阈值。

## 5. 集合鲁棒性、QMC 灵敏度与候选选择

主研究的候选生成不再使用连续局部精修。顶层生成器预注册四个相互独立的 scrambled Sobol lane：一个直接覆盖 V9 五维规范域，另外三个分别以三枚 PAD 为显式锚点，通过规范无向三角形面积测度提案后映回同一 V9 五维域。四路按 `DIRECT, PAD_A, PAD_B, PAD_C` 固定交错，提案失败或重复均消耗一个预算位且不补点；映回后的五个 binary64 参数以规范字节去重，每个唯一参数最多调用一次 V9。主预算为 256 次 attempt，128/256/512 是共用最大设计前缀的计算收敛层级；局部精修评价预算为 0。这些是两个对象共用的预注册计算协议，不是根据某个连接器结果调整的物理阈值。

当前对象合同只为摩擦给出有界区间，没有给出可校准的概率分布。因此 V1 不把 Sobol 点解释成随机样本，也不声明 CVaR。其主鲁棒量是在已认证集合 \(\Xi\) 上的最坏情形：

\[
R_{\min}(x)=\min_{\xi\in\Xi}\rho(x,\xi).
\]

固定 seed 的 Sobol 低差异设计只用于区间内部的确定性灵敏度覆盖。对排序后的 \(N\) 个 QMC 裕度 \(\rho_{(1)}\le\cdots\le\rho_{(N)}\)，报告最差 \(\alpha N\) 个设计质量的分数加权下尾均值 \(Q_\alpha\)。它是有限设计的描述统计，不是概率风险量；\(\alpha N\) 非整数时仅对边界次序统计量使用相应分数权重。

当前实现的认证集合仅含来源明确的摩擦区间。位姿、表面法向、质量、质心、惯量、关节跟踪与执行误差在独立标定并冻结前不得写入鲁棒性声明，也不得用占位数补齐。

候选选择使用字典序而非混合单位加权和：

1. 最大化 \(R_{\min}\)；
2. 最大化 QMC 下尾描述量 \(Q_\alpha\)；
3. 最小化完成任务所需峰值法向力；
4. 最小化关节力矩利用率；
5. 最大化已认证的轨迹净空下界。

当前有限 PAD witness 闭合代理的有效路径终点就是接触，因而其完整 witness 路径净空严格下界为 0，暂时不能区分候选；完整手、机械臂、桌面以及 approach/closure/lift 的 swept-collision 证书是冻结动态候选前的独立必需门。

算法输出只能称为“预注册有限候选设计中，通过全部强制证书且字典序鲁棒指标最优的抓姿”；没有全局最优证明时不得称数学全局最优。任一候选在碰撞、不确定性范围、力/力矩或执行合同上失败，都必须保留完整谱系并从正式排名中失败关闭，不得用补采样替换。

## 6. 在线控制

规划与执行分离。给定期望物体扳手 \(w_d\)，接触力分配为

\[
f_d=G^\dagger w_d+N_G\eta^*,
\]

其中 \(N_G\eta^*\) 仅改变内部力。\(\eta^*\) 由约束 QP 求得，目标是最小接触力/力矩负担，约束是单边接触、摩擦锥、关节力矩和已标定的法向力能力。关节执行采用物体级无源阻抗与局部顺应闭合；发生可观测的摩擦裕度下降时重新求解力分配，而不是固定增加 1 N。

在线允许输入仅为视觉估计、关节位置/速度/力矩、腕部 F/T 和可用触觉。对象 ground-truth pose、碰撞体名称、接触点/法向和 PhysX 接触报告只能用于 post-hoc evaluator，且必须记录在线真值使用计数为 0。

## 7. 预注册验证协议

两个对象必须使用配对随机种子和相同共享配置：

1. 独立从各自 CAD 生成候选，禁止复用 candidate ID、接触坐标或旧对象关节角；
2. 静态验证刚体变换等变性、尺度/质量一致性、摩擦锥收敛、任务裕度单调性、碰撞覆盖和真值防火墙；
3. 独立 Isaac 进程完成顺应闭合、离桌、0.5/2/10/40 mm 抬升和保持；
4. 稳定后施加预注册方向设计的有限时长 6D 扰动，强度按重力和特征力臂无量纲化连续扫描；
5. 报告保持概率曲线、临界扰动生存曲线、AURC、滑移/漂移、峰值力/矩、预测 \(\rho\) 与实测临界扰动校准关系；
6. 二元结果报告配对成功数和 95% 区间；连接器/SKU 是实验单位，物理时间步不是独立样本。

基线包括：旧固定候选+旧控制器、圆柱三指均布、nominal Ferrari-Canny、nominal task-wrench。消融包括 `-TWS`、`-Uncertainty`、`-CompliantClosure`、`-Actuation`、`-InternalForceQP` 和允许新对象调参的过拟合反例。

## 8. 证据与图形

GUI、截图和期刊图必须程序化读取冻结 CAD、candidate bundle、raw CSV/JSON 和统计结果。至少包括算法图、两个对象的接触/PAD/摩擦锥、任务扳手截面、扰动保持概率、预测-实测校准和消融。生成式图片不得作为实验图或动态证据。

## 9. 学术来源

- Li and Sastry, *Task-Oriented Optimal Grasping by Multifingered Robot Hands*, IEEE Journal on Robotics and Automation, 1988, DOI `10.1109/56.769`。
- Ferrari and Canny, *Planning Optimal Grasps*, ICRA 1992, DOI `10.1109/ROBOT.1992.219918`。
- Borst et al., *Grasp Planning: How to Choose a Suitable Task Wrench Space*, ICRA 2004。
- Haas-Heger et al., *Passive Reaction Analysis for Grasp Stability*, IEEE T-ASE, DOI `10.1109/TASE.2018.2803620`。
- Li, Culbertson and Ames, *Toward An Analytic Theory of Intrinsic Robustness for Dexterous Grasping*, IROS 2024。
- SpringGrasp, *An Optimization Pipeline for Robust and Compliant Dexterous Pre-Grasp Synthesis*, RSS 2024。
- Wimböck et al., object-level impedance control for dexterous hands, IROS 2006。

这些来源建立方法依据；是否达到期刊发表水平仍取决于跨多个独立 SKU、真实硬件、充分样本和可复现实验，不能由算法文档预先宣称。
