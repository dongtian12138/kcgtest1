# KCG 电连接器装配任务

本包是 KUKA iiwa 14 + KCG 空间三指手的第二阶段课程，目标是逐步实现：

```text
GRASP → PREALIGN → INSERT → ENGAGE → SCREW → HOLD → PASSED
```

最终目标仍是用 residual RL，并在更后期加入 VLA 高层技能选择，完成面向航天任务的
MIL-DTL-38999 类圆形电连接器抓取、对准、插入和旋拧。当前模型不代表航天认证、
space-grade 等级或制造件几何精度。

## 最终端到端任务定义

最终场景不是“手里已有连接器、只执行旋拧”，而是桌面上同时存在自由端和固定端：
固定端安装在台座上，自由端的位置、朝向和型号可变化。完整状态机定义为：

```text
HOME
→ LOCATE_FREE（识别自由端型号、6D 位姿、插入轴和键位）
→ PLAN_AND_PICK（无碰接近并抓取）
→ VERIFY_GRASP（抬升并验证未滑落）
→ RELOCALIZE_IN_HAND（重新估计自由端相对三指手的实际位姿）
→ LOCATE_FIXED（识别匹配的固定端及其装配坐标系）
→ PREALIGN（移动到预插入位）
→ VISUAL_SERVO（近距离修正横向、倾角和键位误差）
→ INSERT_AND_ENGAGE（接触插入并确认已正确啮合）
→ SEGMENTED_TWIST（q7 分段旋拧，三指保持防滑）
→ VERIFY_ASSEMBLY（机械、视觉及可用电气信号联合验收）
→ RETURN_HOME
```

远距离搜索、抓取、搬运和回零由视觉、标定、运动规划及安全状态机完成；接触阶段
再由有界 residual RL 修正可靠的确定性基线。VLA 后续只负责识别任务语义、选择
型号参数、调用技能和决定失败恢复，不直接接管毫米级装配或 240 Hz 安全控制。

成功不能仅由“q7 已转到目标角度”判定。至少应同时检查最终相对位姿和端面间隙、
实际螺母转角与轴向进给的一致性、三路指根力矩保持且无滑移、机械臂可用的关节
力矩或电流无卡滞异常，以及腕部传感器的轴向力、横向力、弯矩和旋拧轴力矩均在
该型号允许区间，并通过停止后的稳定保持。若目标是确认电气功能而不只是机械到位，
还必须加入导通、接触电阻、锁止开关或该型号规定的电气验收信号。

多型号通过“共享接口 + 型号注册表”实现，不能让策略自行猜测安全参数。每种型号
必须登记公母配对、CAD/装配坐标系、几何对称性、键位、允许抓取区域、插入深度、
螺距或卡口形式、旋向、锁紧角、终拧区间和损伤阈值，并提供相应视觉与失败样本。

## 第一版边界

- 唯一机器人来源是现有 KUKA/三指手模型，不使用 ABB 的 URDF、网格或控制参数。
- 公头必须依靠三指接触和摩擦抓持，禁止吸附、焊接或隐藏固定约束。
- 母座固定，初态近似同轴，装配控制的位姿仍来自仿真真值；已有固定全局虚拟
  RGB-D 预检，但第一版仍不是视觉驱动装配。
- Plug 分为 `plug_body / coupling_nut / mating_insert`；Receptacle 分为
  `receptacle_body / flange / mating_insert`。
- 第一版不求解真实螺纹牙面接触。啮合后使用
  `axial_travel = lead * angle / (2π)` 的显式螺旋关系。
- `coupling_nut` 与 `plug_body` 是两个普通刚体，并由 Z 轴转动关节连接；连接器
  自身不设为 articulation。只有对准、深度和低速条件连续满足后，才在物理步边界
  创建 Z 轴移动关节与 PhysX rack-and-pinion 耦合。其比例使用
  `ratio = direction * 360 * meters_per_stage_unit / lead_meters`，单位是度/场景距离。
- ENGAGE 只增加关节约束，不改公头当前位姿；插入和抓持阶段禁止逐帧写物体位姿。
- 可观测的真实手部力信号只有 `f1j2、f2j1、f3j2` 三路单轴指根力矩；没有
  指尖触觉。当前 `v0` 尚未把腕部六维力输入策略；后续直接复用现有零变换 fixed
  joint `hand2arm` 作为默认的零厚度、零质量虚拟测量边界，不新增传感器实体、不改
  TCP 或机器人 USD。真实系统可让物理腕部 F/T 或七关节力矩反演发布同一标准
  wrench 接口，它不属于任何单根手指。默认禁用的 `v1` 接口契约见
  [`config/wrist_ft_v1_contract.yaml`](config/wrist_ft_v1_contract.yaml)，实施与标定门
  见 [`docs/wrist_ft_v1_design.md`](docs/wrist_ft_v1_design.md)。
- RL 不直接输出关节努力。当前 `v0` 只在已抓紧、已插入、已啮合的单行程中输出
  有界 q7 速度残差和三个指根关节位置残差；4D 动作/24D 观测保持不变，默认
  `stage20`，也可显式选择 `stage60` 或 `stage120`。自由空间接近、抓取、插入、
  啮合和重抓仍由确定性控制器负责。

## 技术分工与长期目标

- Gazebo/ROS 2：继续承担模型、控制器、MoveIt、话题和圆柱抓取回归，不作为复杂
  螺纹任务的主训练场。
- Isaac Sim/Isaac Lab：承担连接器接触任务、GPU 批量环境、相机数据和域随机化。
- 传统控制器：提供自由空间轨迹、抓取、基础插入/旋拧和安全边界。
- Residual RL：学习小范围对准、轴向推进、旋转和失败恢复残差，不直接控制全部
  关节努力。
- VLA：待基础装配与视觉数据稳定后，用于对象/键位理解、高层 skill 选择和恢复
  决策；毫米级接触闭环仍由传统控制器和 residual RL 执行。

如果未来任务退化为单型号、固定工装且传统控制器已接近满成功率，则没有必要为了
使用学习算法而强行加入 RL/VLA。当前项目保留学习路线，是因为最终目标包含初始
偏差、摩擦/间隙变化、多型号视觉泛化和失败恢复。

## 桌面场景、公开规格代理与抓取进度

当前桌面场景采用 `0.80 × 0.90 m` 台面、`0.20 m` 台面高度；机器人基座侧留出
明确空隙，固定端通过独立夹具安装，自由端位于相距至少 `0.30 m` 的抓取区并在重力
下自然落桌。桌面与夹具是静态碰撞体，不再只是视觉背景。

当前 public-spec 仿真身份固定为：

- 自由端：`D38999/26KJ61SN`，straight plug、socket、shell `J/25`、insert `61`、
  N 键位；
- 固定端：`D38999/20KJ61PN`，wall-mount receptacle、pin、shell `J/25`、insert
  `61`、N 键位。

两者均是 Series III threaded 类型且公母互补；TE 官方目录也能逐项对应这两个完整
PIN。DLA 的公开 `/20H Amendment 1` 和 `/26G Amendment 4` 规格页保存在
[`assets/public_specs/mil_dtl_38999`](assets/public_specs/mil_dtl_38999/SOURCE.md)，
每个文件都有来源 URL、页数和公开发布说明。现在已把 TE 对应候选料号的官方
Customer View Model STEP/DXF 下载到本地忽略的 `artifacts` 目录做来源审计；这些文件
没有纳入仓库，许可和再分发边界也尚未确认。两端 STEP 和 DXF 的正脸都是空白接口，
DXF 还明确声明并非受控图纸，因此没有 25-61 触点、主/次键或可用键槽碰撞细节，不能
单独作为 keyed-v2 的内部接口真值。来源和范围审计记录在
[`d38999_keyed_v2_source_candidate.yaml`](config/d38999_keyed_v2_source_candidate.yaml)。

没有实物时，新的
[`d38999_keyed_public_spec_v2.yaml`](config/d38999_keyed_public_spec_v2.yaml)
直接以 MIL-DTL-38999N Amendment 2 的壳体/五键接口尺寸和 MIL-STD-1560C Change 3
的 25-61 全部坐标建立独立仿真身份：主键为 `0°`，四个副键为
`80°/142°/196°/293°`，并带匹配键槽碰撞。生成的
`d38999_shell25j_25_61_n_keyed_public_spec_v2.usda` 与旧 v1 并存，不覆盖旧模型。
它是可追溯的公开规格仿真模型，不是制造商内部 CAD、实物计量结果或硬件资格记录。

当前 `d38999_shell25j_61_pair_proxy_v1.usda` 仍由公开规格尺寸生成；它包含 61 个可视
pin/socket、简化法兰、可碰撞外壳和独立 revolute coupling nut，但触点排布和键几何
不是 25-61 受控模型。螺纹牙、精确 insert、真实质量/惯量、接触材料和损伤阈值仍
未知，因此只能叫 public-dimensional visual/physics proxy，不能声称厂商精确、
MIL-DTL qualified 或 space-qualified。

CPU 侧已经有严格拒绝的掌心 RGB-D 五键/主键方向检测器和 C2 两分支选择器；遮挡、
出画、缺深度、低置信或非 N 五键图样都会拒绝。公开尺寸推导的 yaw 窗口分为 nominal、
tight-size 和 adversarial-GD&T 三档；最严的 `0.06055°` 是显式工程压力假设，不是军标
直接给出的实测间隙。当前只允许以它的一半 `0.03028°` 作为仿真 shadow p95 门，仍未
接入插入控制，旧 v1 仍必须返回 `KEYED_GEOMETRY_UNAVAILABLE`。

当前证据仍分成两段，不能拼成整链通过：旧桌面场景的 `camera_rig_probe_v5` 已验证
Palm/Wrist 都是 `handbase_link` 的固定子相机，并在 Home 到 pregrasp 的 3072 个采样
点保持固定外参；独立 keyed-v2 Palm 正脸探针 v3 已在实际渲染 RGB-D 中识别五键并只
选择 shadow C2 分支。前者看的仍是旧 v1 插头，后者没有移动机器人和 Wrist，因此新的
keyed-v2 双相机静止/移动探针仍是下一道门。两份报告都保持 `control_authorized=false`。

keyed-v2 r2 的固定开环 yaw 碰撞扫描显示：正确 N 方向以及 `±0.35°` 能越过仿真视觉
触点面；`±0.5°` 和 `180°` 在键槽入口首次发生键/键槽接触，距视觉触点面代理仍有
`12 mm`。这只验证公开规格键槽代理的几何顺序；coupling nut/thread 在扫描中被隔离，
pin/socket 和 insert face 仍是 visual-only，所以不能宣称螺纹初始啮合或真实电触点
先后顺序已经物理验证，也不能据此开放插入控制。

这里选用的 class K 是不锈钢钝化、防火墙、导电等级，不是 MIL-DTL 中的
space-grade。若后续目标严格变成航天器真空/低放气资格，应另建 class G/H 或明确的
space-profile 身份；不能把当前 K 型号改名为 space-qualified。

已经通过的桌面节点如下：

- synthetic 和 D38999 两种自由端都从台面上方 `15 mm` 自然下落并稳定落桌；固定端
  漂移为 0，仿真开始后物体 pose write 为 0。
- D38999 + KUKA Home 同场通过，机器人与桌面、夹具和两端连接器的外部接触均为 0。
- KUKA 从 Home 经保守 minimum-jerk 关节路点移动到 synthetic 自由端上方预抓位；
  TCP 末态误差约 `1.36 mm / 0.15°`，外部误碰为 0。
- synthetic 抓取链完成“下降 → 去皮 → 闭手 → 三指预载 → 抬升 → 4 s 保持”，
  headless 连续两次逐字段一致地通过。自由端抬升约 `69.03 mm`、body-TCP 滑移约
  `0.014 mm`，三路指根力矩全部加载且全程最大约 `0.740 Nm < 1 Nm`；最终没有桌面
  承托，也没有 attachment、object drive 或运行中物体位姿写入。
- 抓取保持阶段把实际位置差分速度与 PhysX post-solver velocity 分开门控：最后
  120 步的可观察关节速度峰值约 `0.00020 rad/s`，post-solver 峰值约
  `0.03896 rad/s`，分别低于 `0.03/0.05 rad/s` 门。后者是闭环接触求解器健康证据，
  不再被错误解释为物体真的以该速度运动。
- D38999 代理已完成真实接触抓取、抬升和保持；随后在预置 `3 mm` engage gap 下
  完成 `open → 重新定位 → Nut-only 三指重抓`，headless 两次逐字段一致通过。最终
  三指都有 CouplingNut 接触且 BodyAssembly 接触为 0，全程最大三路指根力矩约
  `1.142/1.475/1.230 Nm`，低于 `1.8 Nm` 运行目标和 `2.0 Nm` 硬停止。
- 同一 Nut-only 状态下的 D38999 q7 探针已分别完成 `20°` 和单行程 `120°`，且
  每个阶段都 headless 两次逐字段一致通过。`120°` 阶段实测 q7 `-119.9994°`、
  Nut 相对 Body `+120.2771°`、Body 轴向进给 `-1.00228 mm`，螺旋误差约
  `0.028 µm`；三路最大指根力矩仍为 `1.142/1.475/1.230 Nm`。它使用
  `3 mm/rev` 解析代理，不是真实螺距；为消除分段环形
  碰撞体吃掉 `0.30 mm` 公称径向间隙造成的假接触，只过滤 500 个精确列举的代理
  pair，其余碰撞和横向/倾角门保持启用。该节点仍从预置 engage 状态开始，不含真实
  插入、键位、完整一圈或装配成功判定；超过 `120°` 仍需要释放、q7 回卷和重抓。
- 首个 `120° → 松手 → q7 回卷 → Nut-only 再抓` 循环也已 headless 两次逐字段
  一致通过。q7 回卷 `119.9992°`，释放期间 Nut 最大漂移 `0.00359°`、Body 轴向
  漂移 `0.000834 mm`，再抓后 Nut 进度损失约 `0.000070°`，三指接触与三路力矩门
  都恢复。这里必须明确：无阻力时 Nut 会以约 `101 rad/s` 数值发散；三档普通粘性
  阻力又无法同时满足“120° 可旋拧”和“松手自锁”。因此当前分段间使用一个只在松手
  期间启用、再抓后移除的 `0.05 Nm` 自锁制动代理。它不是 attachment 或位姿写入，
  但也没有真实硬件标定，不能当成真实螺纹自锁证据。
- 当前同一 World 的完整确定性代理节点已连续完成
  `Home → 抓取 → 抬升 → 固定端对准 → 9 mm物理插入 → Nut-only重抓 → 3×120°旋拧`
  `→ 2次回卷重抓 → 代理就位判定 → 松手撤离 → Home`。最终源码已分别完成一次
  headless 与一次 GUI 全链 PASS。GUI 实测 Nut 总进度 `6.306797 rad`、轴向进给
  `3.000021 mm`、最终 gap `0.000508 mm`、回 Home 关节误差 `0.0001644 rad`；全链
  三路力矩最大 `0.350459 Nm`。插入后换抓期间的 `0.05 Nm` 临时防转代理将 Nut 净
  漂移限制为 `0.0537°`，并在正式螺旋代理激活前移除；最终座止另有同样限力的移动
  稳定代理。没有机器人碰桌面、夹具或固定端，也没有 attachment 和运行中物体 pose
  write。这里的键位、螺纹和自锁仍是显式代理，控制位姿仍为 Isaac 真值，因此
  `assembly_success_claimed=false`。
- 不覆盖冻结基线的 `--smooth-demo` 已完成 clean headless 与 GUI 全链 PASS。headless
  phase step 从
  `65311` 降到 `52547`，仿真时间从 `272.129 s` 降到 `218.945833 s`，约减少
  `19.54%`。它只压缩 Home 开手和 q7 旋拧、回卷、安全高度返回；接触、插入、
  HOLD、重抓、撤离和全部安全阈值不变。
- 固定全局虚拟 RGB-D 使用同一渲染产物的 RGB、深度和 renderer semantic mask。
  独立 bootstrap 以 mask 中位像素射线与注册高度平面相交，只估计世界系 XY；
  自由端/固定端误差为 `2.500 mm / 1.824 mm`，均通过 `10 mm` 门。深度数据只用于
  可见性和表面诊断，注册高度来自已知 D38999 代理几何，没有使用真值 XY。
- 同 World `masked-rgbd` 预检已在首个机械臂动作前通过；它不 reset/clear World、
  不写物体 pose，清理渲染资源后 timeline 仍为 playing，随后 smooth 完整 E2E 继续
  PASS。clean 日志的 invalid `UsdAttribute.Get()` 警告为 0。控制仍明确报告
  `control_pose_provider=sim_ground_truth` 和 `masked_rgbd_xy_used_for_control=false`。

上述首次完整成功状态已冻结到
`artifacts/kcg_connector/d38999_end_to_end_v1/baseline_20260812T111941Z/`。
快照包含源码、配置、资产、最终 GUI/headless 日志、环境指纹和隔离回放脚本；归档
SHA256 为 `c7125ee77de65befc0bb927da7c543c1d864ed5e536101820db2813983b57e31`。
从归档解压的冻结源码已重新得到 `918 passed`，后续运动提速和视觉改动不会覆盖这份
基线。

D38999 型号 ID 已登记到共享 6D pose contract。当前 tabletop smoke 会把 Isaac
`[w,x,y,z]` 真值严格转换为契约的 `[x,y,z,w]`，并验证自由/固定端型号、角色、公共
world frame 和配对关系。严格 `PoseProvider` 样本还门控 source/frame/clock/capture、时间戳、
标定 hash 和 JSON-safe provenance。`masked RGB-D XY + truth orientation` 只能用于 PRECHECK，
不能授权 CONTROL；只有显式的仿真真值或未来不依赖真值的完整 6D 样本才可通过各自
用途的门。当前不是 FoundationPose、点云 6D、手眼相机或视觉控制，键位/方向仍使用
真值，object→grasp/assembly 标定也尚未伪造。

当前运动仍是 `joint_interpolation_screening_not_collision_planned`。机器人 USD 中
self collision 关闭；现有 SRDF 的 136 个候选碰撞对里，80 对被标为 `Never`、16 对
为 `Adjacent`，只剩 40 对默认检查。因此所有“外部误碰为 0”的结果都不能当作
自碰安全证明，必须在候选 ACM 重建和完整 PlanningScene 路径检查后才可解除该门。

## 当前已验收里程碑

当前已经建立与模拟器无关的几何、转角展开、螺旋进度、阶段和失败原因判定。
Isaac Sim 6.0.1 的兼容性、RTX/CUDA 无头启动、刚体落体、连接器 USD 拓扑和独立
螺旋约束均已在本机通过。在此基础上，以下机器人在环物理节点也已通过：

- 三指手仅依靠刚体接触和摩擦抓紧螺母，无吸附、无隐藏固定约束。
- 只驱动 KUKA `iiwa_joint_7` 的 20° 旋拧探测通过；三指手只负责夹持和防滑。
- `q7-static-axial` 和 `open-hand` 两个反事实通过，排除了“单独轴向
  命令或张开手也会产生同样旋拧”的假阳性。
- `3 × 120°` 分段基线完成 360°：夹紧 → q7 旋拧 → 释放/抬起 → q7 回卷
  → 重抓，最终保持 2 s。
- residual `v0` 的两回合零残差基线通过：两回合均在 26 个策略步后
  达到 `20.0025°` 实际螺母转角、`0.21437 mm` 轴向进给和 0.5 s 稳定保持。
- seed 随机化 `v1` 的零残差序列门也通过：以 base seed `1201` 启动后，两回合沿
  同一可重放 RNG 流抽到不同参数，分别覆盖 0/1 步动作延迟，`2/2` 均安全成功。
  这只证明第一批小范围扰动没有破坏基线，不证明学习或泛化得到提升。
- residual 动作因果验收通过：q7 slow/fast 的实际尾段速度约为
  `8.00/12.00°/s`，同方向进度差约 `3.90°`；三个夹持动作均对对应
  指根力矩通道产生了可分辨响应。
- hard reset 会在复位前移除运行时螺旋约束，依靠接触恢复到 checkpoint，
  再按当前位姿重建 prismatic/rack。复位验收不只看写入后瞬时 pose：最后 10 个
  solver settle step 的 body/nut 线速度、角速度和 q7 速度都必须低于冻结门限；字段
  缺失、类型错误或非有限值均失效关闭。最新物理回归也通过了首步 snap 门槛。
- episode 安全证据直接取自 `raw_physics`：初始快照和每个 240 Hz 物理子步累计 q7
  速度、关节限位偏差、三路指根力矩等峰值，10 Hz 边界另累计螺母速度、q7 跟踪和
  抓持漂移。正式报告严格复制并校验 backend 的 raw safety report，不再根据终止
  原因反推“安全”；缺字段、非法类型、NaN/Inf 或投影不一致都会失败。
- CUDA SAC 32-step 训练链路通过：Gymnasium checker、replay buffer、CUDA 参数
  更新、模型保存和重载均已验证。当次 actor 参数最大变化为
  `0.0059614`，replay size 和 timesteps 均为 32，原始 PyTorch 仍为
  `2.11.0+cu128`。
- 当前 final stage20 32-step dry-run 产物为
  `train_seed42_20260811T172410410847Z`。训练阶段 raw safety 审计为
  `passed=true`，覆盖 32 个策略步（1 个完整 episode + 1 个 partial episode），失败
  原因为空；记录的峰值包括三路指根力矩 `0.73928255 Nm`、q7 速度
  `0.18252415 rad/s`、螺母速度 `0.19348097 rad/s`、q7 跟踪误差
  `0.00515053 rad`、抓持平移/旋转漂移 `4.9777e-5 m / 0.000248894 rad`。该运行仍为
  `optimizer_updates=0`、actor delta 0、training evidence false，不能宣称学到了策略。
- 由该 train 绑定生成的 final paired20 为
  `paired_evaluate_seed10000_20260811T172501510645Z`：工程完整性通过，zero/trained
  都是 `20/20`，双方 raw safety failure 0；randomization、reset 初始签名和 seed
  均为 `20/20`，counterbalanced order `10/10`，reset/螺旋代理重建 `40/40`。
  改善为 0，单侧 McNemar `p=1`，trained 单侧 95% Clopper-Pearson 下界仅
  `0.8608916593`，所以 improvement、competence 和 generalization 声明均为 false。
- 现有 `hand2arm` 虚拟腕部六维测量边界实测通过：reaction wrench 为 6 维；在
  `handbase_link` 原点施加 `±4 N`、`±0.4 Nm` 和半量程载荷后，确认无轴交换且
  `canonical = -raw`。六轴绝对增益为 `0.993799..1.000000`，最大同类轴串扰约
  `0.0189%`；`grasp_tcp` 与机器人 USD/资产清单哈希均保持不变，相关 joint、
  disjoint 和 teardown assignment warning 为 0。
- 七个 iiwa 关节力矩到工具六维 wrench 的纯 NumPy 估计核心已实现并通过失效安全
  测试。它要求显式动力学补偿、6D 任务尺度、阻尼、condition 与 7D 投影 residual
  阈值；缺失标定、Jacobian 秩低于 6、近奇异或 residual 超限时返回 invalid，当前
  默认关闭且不接入 4D/24D v0。
- 多型号基础契约已建立：`config/connector_model_registry_v1.yaml` 目前只登记当前
  synthetic plug/receptacle，registry/profile 双重关闭，四项未标定腕力限保持
  `null`，所以不能误启用。`workflow_contract.py` 冻结了从 `DETECT_LOOSE`、抓取、
  手内重定位、检测固定端、对准/插入/旋拧/验证到 `HOME` 的 11 个阶段及逐阶段证据
  门；它尚未接相机、ROS、Isaac 或 VLA。
- 已新增不改变 4D/24D 接口的单行程课程契约
  `config/connector_residual_curriculum_v1.yaml`：stage20/60/120 分别要求
  20°/60°/120°、0.5/1.0/2.0 s 保持和 40/100/180 个最大策略步。按当前 q7 初值，
  120° 计划终点距下限仍有 `20.572°`，并强制保留至少 `10°` 命令余量。该节点目前
  已接入唯一 Isaac backend。最终物理复核中 stage20 两回合约为 `20.002°`、
  `0.214 mm`、0.5 s HOLD，stage120 两回合约为 `120.223°`、`1.319 mm`、2.0 s
  HOLD；stage60 也已通过两回合课程门，约为 `60.075°`、`0.652 mm`、1.0 s HOLD。
  这些是简化螺旋代理的零残差物理基线，不是训练有效性证据。

纯 Python 回归命令为：

```bash
cd ~/WorkPlace/kcgtest1
source /opt/ros/humble/setup.bash
colcon build --packages-select kcg_connector --symlink-install
source install/setup.bash
python3 -m pytest -q src/kcg_connector/test
```

当前开发树的纯回归为 `985 passed`，覆盖 D38999 桌面/抓取/物理插入/完整代理旋拧与回 Home、
RGB-D/PoseProvider、residual v0、seed 随机化、腕部 F/T、关节力矩 wrench 估计、
多型号注册表/工作流和 20/60/120 课程契约。上文的 `918 passed` 只属于冻结快照，
不用它代替当前开发树的全量计数。

### CouplingNut 单齿画面抖诊断边界

针对“螺母某个齿转到画面背后后仍在抖”的观察，现已完成 baseline、RTX transform
history=512、以及只在 session layer 为 Segment_00 补齐显式 `rotateZ=0` 的三组
四相机同步诊断。每组在 240 Hz 物理轨迹上按 30 Hz 取 265 个同 step 样本；rear/front
和 left/right 四视角共 1060 张 PNG，均与 global step、phase、物理报告、summary、
capture helper SHA256 逐项绑定。三组物理 trace 相同，各自连续 5590 steps 跟踪全部
24 个齿且 transform anomaly 为 0；最大相对旋转误差 `2.98e-8 rad`，远低于
`1e-5 rad` 诊断门，相对平移误差为 0。因此可以排除“某个齿作为独立物理零件相对
CouplingNut 自己运动”这一解释，但不能据此否认用户看到的渲染画面现象。

四视角使 baseline-vs-history512 和 baseline-vs-Segment00-normalized 两组严格
same-view A/B 都能完成；不过全序列可测 identity union 仍只有 16/24，缺失
`Segment_05/06/07、13/14、21/22/23`，且没有任何相邻采样 transition 达到 24/24。
另外没有在看数据前冻结“可接受视觉 residual”阈值，30 Hz 也不能覆盖采样间伪影。
所以正式聚合报告是 `VALID_LIMITED_VISUAL_JITTER_UNRESOLVED`，并明确写出
`render_jitter_absence_claim_authorized=false`，不能把“四相机分析通过”说成“单齿抖
已经修好”。可审计的 [report](../../artifacts/kcg_connector/d38999_nut_tooth_jitter/four_synced_evidence_v3/report.json)
和 [manifest](../../artifacts/kcg_connector/d38999_nut_tooth_jitter/four_synced_evidence_v3/manifest.json)
保留了完整覆盖缺口与哈希。

三组接触聚合完全相同：Segment_04/21 的最大 contact impulse norm 分别约
`0.04302/0.04204`，最低 separation 约 `-24.11/-14.74 um`，并分别含 f3/f1 指尖
counterpart；Segment_00 只有约 `0.01160` 和 `-0.429 um`，没有手指 counterpart。
Segment_21 同时属于上述视觉缺失齿。这削弱了“Segment_00 缺失零角度 op 是唯一根因”
的假设，但不能证明 Segment_04/21 的接触就是画面抖原因。下一次最有信息量的实验是
保持同一物理 trace，增加隐藏/半透明手爪的无遮挡诊断 pass，再预注册视觉 residual
门并做接触激励 A/B；不要通过放宽当前四齿/12 px 可测门来制造通过结果。

## Residual RL v0 接口

策略频率为 10 Hz，每个策略步在 Isaac 中执行 24 个 240 Hz 物理子步。当前任务是
已啮合状态下的单行程：默认 `stage20`，可选 `stage60`/`stage120`；它不是端到端
学习接近、插入或整圈重抓。

4 维归一化动作均在 `[-1, 1]`：

1. q7 旋紧速度残差：在 10°/s 标称速度上加 `±2°/s`，不会反向，也不会
   超过现有 20°/s 安全上限。
2. `f1j2` 夹持位置残差，标称 `0.750 rad`，范围 `±0.020 rad`。
3. `f2j1` 夹持位置残差，标称 `0.500 rad`，范围 `±0.020 rad`。
4. `f3j2` 夹持位置残差，标称 `0.750 rad`，范围 `±0.020 rad`。

24 维观测由以下物理量组成：

- 阶段进度 1 维；
- q7 跟踪误差和旋紧方向速度 2 维；
- 实际螺母转角/速度 2 维；
- 实际轴向进给/速度 2 维和螺旋误差 1 维；
- TCP 相对螺母的平移漂移 3 维和旋转漂移 3 维；
- `f1j2、f2j1、f3j2` 三路单轴指根力矩 3 维及其差分 3 维；
- 三个夹持关节相对标称位置 3 维；
- 剩余螺母转角 1 维。

成功由实测螺母转角和轴向螺旋进给判定，不使用 q7 命令累加值替代物理进度。
目标角、HOLD 时间、最大步数和最小轴向进度比例来自严格解析后的 stage；三个阶段
都要求角度误差不超过 `0.5°`、螺旋误差不超过 `0.1 mm`，并同时满足至少两路有载
力矩、抓持、q7 跟踪及 raw safety 门。

### Seed 随机化 v1

独立配置 `config/connector_residual_randomization_v1.yaml` 保持 4D 动作和 24D
观测不变，每回合由 seed 确定地抽取：

- `f1j2、f2j1、f3j2` 三个夹持标称位置各 `±0.010 rad`；
- 三个指根轴的 Kp/Kd 比例各在 `[0.95, 1.05]`；
- 三路指根力矩观测 bias 各 `±0.005 Nm`，高斯噪声
  `σ=0.002 Nm`，截断到 `3σ`；
- residual 动作延迟为 0 或 1 个 10 Hz policy step。

观测 bias/noise 只进入策略看到的 24D 向量。过载、丢失抓持、终止、reward 和
episode safety 仍使用未污染的 `raw_physics` 状态。第一版没有随机化连接器质量、
摩擦或螺距，metadata 会将 `mass/friction/thread_lead` 明确记录为 `false`。
`residual-zero`、`residual-action-effect` 和 `residual-sac-smoke` 默认保持固定域，只有
显式传入 `--residual-randomization-config` 才启用；formal `residual-train`、
`residual-evaluate` 和 `residual-paired-evaluate` 默认启用该 v1 配置，可用
`--fixed-residual-domain` 显式选择当前版本的固定域。

### 单行程课程 v1

`config/connector_residual_curriculum_v1.yaml` 只解析任务长度和验收门，不改变
动作、观测、资产或物理约束：

- stage20：20°、保持 0.5 s、最多 40 个 10 Hz 策略步、轴向进度至少理论值的 75%；
- stage60：60°、保持 1.0 s、最多 100 步、轴向进度至少 90%；
- stage120：120°、保持 2.0 s、最多 180 步、轴向进度至少 90%。

三个阶段统一要求角度误差不超过 0.5°、螺旋误差不超过 0.1 mm。backend 会同时
校验 q7 初值和计划终点：任一非有限、越界，或终点在关节限位外不能再保留至少
10° 命令余量，都会在动作执行前失效关闭。`--residual-stage` 已把严格、可追溯的
resolved 配置接入 zero/formal 共用 backend；训练、单策略评估和 paired 评估会冻结
curriculum YAML bytes/SHA256、resolver 源码和 resolved stage 文档/SHA256，并要求
训练与评估逐项相等。超过 120° 仍应使用已验证的确定性释放、抬起、q7 回卷和重抓
状态机。

### 默认禁用的可学习性 challenge v1

当前 operational stage20 给最慢 8°/s 动作保留了完整时间余量，因此小范围 seed v1
下 zero residual 必然容易饱和。`config/connector_residual_learning_challenge_v1.yaml`
为后续物理定标冻结了一套独立、默认 `enabled: false` 的工程挑战，不会被现有 backend
或 formal runner 自动读取：

- 同样保持 residual-v0、4D/24D、20° 目标和 0.5 s HOLD，但 deadline 为 28 步；
- q7 control-path 速度比例候选 `[0.85, 1.15]`，三路夹持独立零偏候选
  `±0.015 rad`，质量、摩擦和螺距仍固定；
- 纯 oracle 证明有效 q7 速度始终位于 `[6.8, 13.8]°/s`，四维补偿动作不越界；
  oracle 需要 26/28 步，最慢 zero 需要 30 步；
- 调参、冻结验证和最终 paired 分别使用互不重叠的 64/128/100 个 seed；只有 zero
  成功率位于 65–85%、oracle 至少 98%、两侧 raw safety failure 均为 0、最终 100
  对 randomization 全匹配，并且 zero 的失败原因只有 `time_limit`，才允许后续评审
  接入 runtime。

这套 challenge 的 deadline penalty 固定为 `-10`，deadline 失败只可记为普通
`time_limit`，不属于 safety failure；不能通过制造失抓、过载或 cross-thread 来降低
zero 成功率。在实际 seed 扫描前不能用于宣称“任务可学习”。

## 物理与 RL 节点命令

所有命令均从工作空间根目录执行。默认是无界面验收；物理抓取和旋拧命令可追加
`--gui --keep-open` 进行目视检查。

```bash
cd ~/WorkPlace/kcgtest1

# D38999 桌面 + 夹具 + KUKA Home 全景
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_tabletop_robot_smoke.py
# 期待：ISAAC D38999 TABLETOP ROBOT HOME V1 PASSED

# 固定全局虚拟 RGB-D 独立 bootstrap
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_rgbd_bootstrap_smoke.py
# 期待：ISAAC D38999 RGBD BOOTSTRAP V1 PASSED

# 最小相机安装探针：固定机器人、移动到安全路点并保存 Palm/Wrist 画面，随后早退
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py \
  --camera-rig-probe --no-live-telemetry \
  --output-dir artifacts/kcg_connector/d38999_camera_rig_probe_next
# 只验证仿真候选固定 T_HC 和视角；不抓取、不插入、不能代替真实手眼标定

# 旧版真值/FixedJoint/螺纹代理的 headless 回归；不是视觉或正式装配验收
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py \
  --end-to-end-probe --sim-truth-proxy-regression --smooth-demo
# 期待：ISAAC D38999 SIM GROUND TRUTH PROXY END TO END REGRESSION V1 PASSED

# 同一旧代理回归的 GUI 播放；仍不产生正式装配证据
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py \
  --end-to-end-probe --sim-truth-proxy-regression --smooth-demo --gui

# 可选：只在同一已验收流程结束后保留最终 Home 静态画面
# 在上条命令末尾追加 --keep-open，查看完后关闭 Isaac 窗口退出。

# D38999：预置3 mm engage、Nut-only重抓、q7负向20°螺旋代理探针
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_nut_regrasp_smoke.py --twist-probe
# 期待：ISAAC D38999 Q7 TWIST PROBE V1 PASSED

# 同一状态下的单行程120°探针；仍不是完整装配
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_nut_regrasp_smoke.py \
  --twist-probe \
  --twist-config \
  src/kcg_connector/config/d38999_q7_twist_probe_stage120_v1.yaml
# 期待：ISAAC D38999 Q7 TWIST PROBE V1 PASSED

# 120°后松手、q7回卷并Nut-only再抓；显式使用未标定的分段间自锁制动代理
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_nut_regrasp_smoke.py \
  --twist-probe \
  --twist-config \
  src/kcg_connector/config/d38999_q7_twist_probe_stage120_v1.yaml \
  --rewind-probe
# 期待：ISAAC D38999 Q7 REWIND PROBE V1 PASSED

# Home 到 synthetic 自由端上方预抓位
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_tabletop_home_to_pregrasp_smoke.py
# 期待：ISAAC CONNECTOR HOME TO PREGRASP V1 PASSED

# synthetic 自由端的真实接触抓取、抬升和保持
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_tabletop_pick_smoke.py
# 期待：ISAAC CONNECTOR TABLETOP PICK V1 PASSED

# 三指接触/摩擦物理抓取
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_grasp_smoke.py
# 期待：ISAAC CONNECTOR PHYSICAL GRASP PASSED

# q7 单轴 20° 物理旋拧
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py --mode twist
# 期待：ISAAC CONNECTOR Q7 PHYSICAL TWIST PASSED

# 反事实：q7 不动只施加轴向命令
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode q7-static-axial
# 期待：ISAAC CONNECTOR Q7-STATIC AXIAL COUNTERFACTUAL PASSED

# 反事实：张开三指手
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py --mode open-hand
# 期待：ISAAC CONNECTOR OPEN-HAND COUNTERFACTUAL PASSED

# 3 × 120° 的确定性分段 360° 基线
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py --mode segmented
# 期待：ISAAC CONNECTOR Q7 SEGMENTED 360 TWIST PASSED

# 两回合零残差 reset/step 基线
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-zero --residual-stage stage20 --episodes 2
# 期待：ISAAC CONNECTOR ZERO-RESIDUAL 2-EPISODE PASSED

# 60° 单行程课程：1 s 保持、90% 轴向进给门
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-zero --residual-stage stage60 --episodes 2
# 期待：ISAAC CONNECTOR ZERO-RESIDUAL 2-EPISODE PASSED

# 120° 单行程课程：2 s 保持、90% 轴向进给门
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-zero --residual-stage stage120 --episodes 2
# 期待：ISAAC CONNECTOR ZERO-RESIDUAL 2-EPISODE PASSED

# q7 速度与三路夹持残差的物理因果性
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-action-effect --residual-stage stage20 \
  --action-effect-steps 10
# 期待：ISAAC CONNECTOR RESIDUAL ACTION EFFECT PASSED

# seed 1201：随机化零残差短门，覆盖 0-step 动作延迟
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-zero --residual-stage stage20 --episodes 2 \
  --residual-randomization-config \
    src/kcg_connector/config/connector_residual_randomization_v1.yaml \
  --reset-seed 1201
# 期待：ISAAC CONNECTOR ZERO-RESIDUAL 2-EPISODE PASSED

# seed 1204：随机化零残差短门，覆盖 1-step 动作延迟
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-zero --residual-stage stage20 --episodes 2 \
  --residual-randomization-config \
    src/kcg_connector/config/connector_residual_randomization_v1.yaml \
  --reset-seed 1204
# 期待：ISAAC CONNECTOR ZERO-RESIDUAL 2-EPISODE PASSED

# 同一 seed/物理条件下验证 q7 与三路夹持动作因果性
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-action-effect --residual-stage stage20 \
  --action-effect-steps 10 \
  --residual-randomization-config \
    src/kcg_connector/config/connector_residual_randomization_v1.yaml \
  --reset-seed 1204
# 期待：ISAAC CONNECTOR RESIDUAL ACTION EFFECT PASSED

# Isaac Python/CUDA/Gymnasium/SB3 运行时
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/verify_isaac_rl_runtime.py
# 期待：ISAAC GPU RL RUNTIME PASSED

# 32 步 CUDA SAC 训练链路冒烟验收
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-sac-smoke --residual-stage stage20 \
  --training-timesteps 32
# 期待：ISAAC CONNECTOR CUDA SAC TRAIN SMOKE PASSED

# 正式配置/产物契约的 32-step dry-run（默认启用 seed v1；不做更新）
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-train --residual-stage stage20 \
  --formal-timesteps 32
# 期待：ISAAC CONNECTOR FORMAL RESIDUAL SAC TRAIN PASSED

# 当前版本训练产物的 stage20 确定性评估（替换 <UTC>）
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-evaluate --residual-stage stage20 \
  --formal-run-dir \
    artifacts/kcg_connector/residual_sac_v0/train_seed42_20260811T172410410847Z \
  --evaluation-episodes 20
# 期待：ISAAC CONNECTOR FORMAL RESIDUAL SAC EVALUATION PASSED

# 20 对 counterbalanced 工程门；不满足正向策略改善声明的样本量
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-paired-evaluate --residual-stage stage20 \
  --formal-run-dir \
    artifacts/kcg_connector/residual_sac_v0/train_seed42_20260811T172410410847Z \
  --evaluation-episodes 20
# 期待：ISAAC CONNECTOR FORMAL PAIRED ZERO VS MODEL BENCHMARK PASSED

# 零实体、零厚度的 hand2arm 虚拟腕部六维反力边界
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/virtual_wrist_ft_smoke.py
# 期待：ISAAC VIRTUAL WRIST FT PASSED
```

上述桌面物理入口都可追加 `--gui --keep-open`。`--insertion-probe` 和
`--end-to-end-probe` 当前都必须显式追加 `--sim-truth-proxy-regression`，并且只允许
legacy 抓取；该路径使用物体 pose/contact 真值反馈和 FixedJoint/key/thread 代理，报告
即使 PASS 也只能算仿真回归。正式抓取、视觉报告和 `masked-rgbd` 视觉预检不能与它
组合。`--smooth-demo` 只改变旧代理流程的展示节奏，`keep-open` 只保留最终 Home 画面，
不会自动重播。RGB-D 产物位于
[`artifacts/kcg_connector/d38999_rgbd_bootstrap_v1`](../../artifacts/kcg_connector/d38999_rgbd_bootstrap_v1/report.json)，
包含 RGB、semantic/depth preview、原始 depth NumPy 和 JSON 报告。

当前自碰 fail-closed 审计不需要启动 Isaac：

```bash
cd ~/WorkPlace/kcgtest1
PYTHONPATH=src/kcg_connector \
python3 -m kcg_connector.self_collision_audit --report-only

# 自动门模式；在候选 ACM 与路径验证完成前会有意返回 exit 2
PYTHONPATH=src/kcg_connector \
python3 -m kcg_connector.self_collision_audit --json
```

SAC 冒烟模型与可追溯运行元数据位于
`artifacts/kcg_connector/residual_sac_smoke/`。元数据保存了随机种子、源脚本/配置
SHA256、reset snap 指标和 CUDA 软件版本。

formal dry-run 写入唯一的
`artifacts/kcg_connector/residual_sac_v0/train_seed42_<UTC>/`；除 model/replay 外，
还保存 requested/resolved training YAML、requested curriculum YAML、resolved stage、
resolved randomization、`monitor.monitor.csv` 和 metadata。metadata 绑定源码、资产、
所有配置、模型、初始/最终/重载 actor state、raw safety、post-solver reset 证据以及
实际 Isaac 运行时。当前可复核目录是
`train_seed42_20260811T172410410847Z`；它的 32-step training raw safety 已审计通过，
但 optimizer update 和 actor delta 都是 0。这些证据仍不证明策略已经学会或比零
残差基线更能泛化。完整 formal 契约见
`src/kcg_rl/README.md`。

paired 目录使用
`artifacts/kcg_connector/residual_sac_v0/paired_evaluate_seed10000_<UTC>/`。当前 final
目录是 `paired_evaluate_seed10000_20260811T172501510645Z`，严格绑定
`train_seed42_20260811T172410410847Z`。randomization/signature/seed 均 match
`20/20`，order `10/10`、reset/rebuild `40/40`，zero/trained 均 `20/20` 且双方 raw
safety failure 0，因此 `benchmark_integrity_passed=true`。但 improvement 0、
McNemar `p=1`、Clopper-Pearson lower `0.8608916593`，故
`policy_improvement_claim=false`；20 对工程门不能单独支持正向声明。

正向改善声明至少需要 100 对，并同时通过：真实 optimizer/actor 变化与重载 hash
证据、trained 原始成功率和单侧 95% Clopper-Pearson 下界均不低于 95%、相对 zero
至少提升 10 个百分点、单侧 exact McNemar `p <= 0.05`、无回退、双方 raw safety
failure 为 0，以及全部 seed/randomization/reset signature/order/provenance/runtime 门。
少于 100 对即使终端打印 paired benchmark `PASSED`，也只能解释为工程完整性通过。

评估会精确比较训练时记录的 Python、Isaac Sim、NumPy、Gymnasium、SB3、PyTorch、
CUDA build 和 GPU，并校验当前源码/资产/config/curriculum/resolved stage/actor hash。
缺少这些字段或 hash 已过期的旧 artifact 会失效关闭；`--fixed-residual-domain` 只能
匹配由当前版本生成的固定域产物，不能让旧产物重新有效。

## 当前限制与不可宣称的结论

- 当前使用的是 synthetic `world-prismatic + rack-and-pinion` 螺旋代理，没有真实
  螺纹牙面、间隙、碰撞、乱扣和咬死物理；`cross_thread` 目前只是测量一致性门槛。
- 旧 v1 的键位/螺距及质量、摩擦仍是 synthetic；keyed public-spec v2 的壳体接口
  尺寸、五键 N 图样和 25-61 坐标来自公开军标，但倒角、材料接触、质量/惯量、螺纹
  和损伤阈值仍是仿真假设，不能当制造件真值。
- 当前是单 Isaac Sim 环境。SAC actor/critic 在 CUDA 上更新，但还不是 Isaac Lab
  GPU 批量并行训练。
- `residual-sac-smoke` 的 32-step 更新只证明 `Env → replay → CUDA update →
  save/reload` 链路接通；当前 formal 32-step dry-run 则是 0 optimizer update、actor
  delta 0。两者都不代表 SAC 已经学会旋拧、已收敛或优于确定性基线。
- 当前 seed v1 下零动作已经 `20/20`，没有剩余可测的成功率提升空间；在加入经过
  因果验收、策略可观测且可补偿的扰动以前，不启动长训练，也不以 reward 差异替代
  paired 成功率证据。
- residual RL 尚未接管接近、插入、啮合和重抓；20/60/120° 已接入同一 residual
  环境，但当前通过的是零残差 nominal baseline。完整 360° 过程仍是确定性分段基线。
- 尚未导入真实连接器硬件、指根应变片标定、真实相机和手眼/外参标定、学习式
  目标检测、FoundationPose 或其他 truth-free keyed 6D、视觉伺服、经过标定的质量/摩擦/螺距
  物理随机化、sim-to-real 或 VLA。当前 seed v1 仅覆盖小范围控制、观测和延迟扰动，
  结果不构成航天级精度或认证声明。

## Isaac Sim 环境

Isaac Sim 使用独立的 Python 3.12 环境，不与 ROS 2 Humble 的 Python 3.10 或
现有 HaMeR 环境混装。GPU PyTorch 和 Isaac Sim 分两步安装，以确保不会回退到
CPU wheel：

```bash
cd ~/WorkPlace/isaacsim
.conda-env/bin/python -m pip install \
  -r ~/WorkPlace/kcgtest1/src/kcg_connector/requirements-torch-cu128.txt
.conda-env/bin/python -m pip install \
  -r ~/WorkPlace/kcgtest1/src/kcg_connector/requirements-isaacsim.txt
.conda-env/bin/python -m pip install -e ~/WorkPlace/kcgtest1/src/kcg_connector
~/WorkPlace/kcgtest1/src/kcg_connector/isaac/run_isaac_python.sh \
  -m pip install --no-deps -e ~/WorkPlace/kcgtest1/src/kcg_rl
```

当前 formal artifact 冻结的精确运行时为：Python `3.12.13`、Isaac Sim
`6.0.1.0`、NumPy `2.3.1`、Gymnasium `1.2.3`、Stable-Baselines3 `2.7.1`、
PyTorch `2.11.0+cu128`、CUDA build `12.8`、GPU
`NVIDIA GeForce RTX 5070 Ti`。formal evaluate/paired 会逐字段精确匹配；版本缺失或
不一致会在加载模型前拒绝，而不会以“应该兼容”继续运行。

Isaac Sim 进程与系统 ROS 2 Humble 节点后续通过 ROS 2 Bridge/DDS 通信；不要在
Isaac 的 Python 3.12 进程中直接导入系统 Python 3.10 的 `rclpy`。

机器人 USD 始终从当前 KUKA/三指手 Xacro 生成，避免维护第二份机器人模型：

```bash
cd ~/WorkPlace/kcgtest1
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run kcg_connector export_isaac_urdf \
  --output artifacts/kcg_connector/urdf/handarm.urdf
check_urdf artifacts/kcg_connector/urdf/handarm.urdf
```

导出器会解析所有 `package://iiwa_description` 网格路径，并移除只属于 Gazebo/ROS
控制层的标签；关节、惯量、碰撞网格、mimic 关系和 `grasp_tcp` 仍来自唯一的原始
Xacro。

本机 Conda 环境的 ICU 需要该环境内新版 `libstdc++`。不要替换系统库，也不要把
`LD_LIBRARY_PATH` 全局写入 `.bashrc`；统一使用包装器，只对单个 Isaac 进程设置
动态库路径。安装完成并接受 NVIDIA EULA 后，先运行最小 headless 物理验收，再
导入机器人：

```bash
cd ~/WorkPlace/kcgtest1

src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/headless_smoke.py

src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/import_robot.py \
  --urdf artifacts/kcg_connector/urdf/handarm.urdf \
  --usd-directory artifacts/kcg_connector/isaac/robot

src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/create_connector_asset.py \
  --config src/kcg_connector/config/connector_task.yaml \
  --output artifacts/kcg_connector/isaac/connector_pair.usda

src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/validate_connector_asset.py \
  --asset artifacts/kcg_connector/isaac/connector_pair.usda

# 独立两刚体单元测试；它只验收螺旋约束，不代表机械手已经完成旋拧。
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/thread_proxy_smoke.py
```
