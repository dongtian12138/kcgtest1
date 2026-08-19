# KCG KUKA iiwa 14 + 空间三指手（ROS 2 Humble）

本工作空间只针对 KUKA iiwa 14 与 KCG 空间三指手。ABB 工程不属于本项目，也没有被合并进来。

## 当前适配基线

- Ubuntu 22.04
- ROS 2 Humble
- MoveIt 2.5.9
- Gazebo Classic 11
- `ros2_control` + `gazebo_ros2_control`

工作空间根目录为 `~/WorkPlace/kcgtest1`。原 ROS 1 工程完整保存在
`ros1_original/kcgtest1`，并由 `COLCON_IGNORE` 与 ROS 2 构建隔离。

## 构建

```bash
cd ~/WorkPlace/kcgtest1
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

每个新终端都需要执行：

```bash
source /opt/ros/humble/setup.bash
source ~/WorkPlace/kcgtest1/install/setup.bash
```

## 常用启动方式

MoveIt + 模拟硬件（适合先检查模型、规划和 RViz）：

```bash
ros2 launch kcg_moveit1 demo.launch.py
```

Gazebo + MoveIt + RViz 全系统：

```bash
ros2 launch kcg_moveit1 gazebo.launch.py
```

无界面运行：

```bash
ros2 launch kcg_moveit1 gazebo.launch.py gui:=false use_rviz:=false
```

首个任务“抓住圆柱体—抬起—保持”的诊断运行：

```bash
ros2 launch kcg_grasping cylinder_grasp.launch.py \
  gui:=false use_rviz:=false \
  run_baseline:=true shutdown_on_completion:=true
```

只有终端打印 `SCRIPTED GRASP PASSED` 才算通过。当前物理基线已通过独立冷启动
验收。默认就是自然演示：圆柱从启动开始位于台座上，机械臂从零位经上方安全位
接近，不会在 `RESET` 阶段传送圆柱。需要观察全过程时使用：

```bash
ros2 launch kcg_grasping cylinder_grasp.launch.py \
  gui:=true use_rviz:=false run_baseline:=true
```

RL 启动文件则自动保留原有预抓位启动和圆柱传送复位，以缩短每回合初始化时间。
任务指标、27 维观测和成功判据见
[`src/kcg_grasping/README.md`](src/kcg_grasping/README.md)。

同一套物理任务的 RL reset/step 冒烟测试：

```bash
ros2 launch kcg_rl cylinder_rl.launch.py \
  gui:=false use_rviz:=false \
  run_smoke:=true smoke_episodes:=2 shutdown_on_completion:=true
```

两回合都打印 `RL STEP SMOKE PASSED` 才表示 RL 环境的连续复位、动作、奖励和
终止链路通过。接口定义见 [`src/kcg_rl/README.md`](src/kcg_rl/README.md)。

## 电连接器与 Isaac Sim 阶段

圆柱任务验收后，第二阶段在独立的 `kcg_connector` 包中开发，不修改已经通过的
圆柱回归任务。目标阶段为：

```text
GRASP → PREALIGN → INSERT → ENGAGE → SCREW → HOLD → PASSED
```

第一版使用仿真真值、固定母座、近同轴初态和简化螺旋关系；不把 ABB 机器人资产
混入 KUKA 系统，也不声称简化模型具备真实航天连接器认证精度。当前几何/成功判据、
Isaac 环境和机器人导出说明见
[`src/kcg_connector/README.md`](src/kcg_connector/README.md)。

最终任务不是只在手中旋拧一个已知圆柱，而是：视觉识别桌面自由端的型号和 6D
位姿 → 规划抓取并做手内重定位 → 识别固定端及装配坐标系 → 视觉伺服对准 →
插入/啮合 → 由 `iiwa_joint_7` 分段旋拧、三指手只负责夹紧防滑 → 以相对位姿、
螺旋进给、三路指根力矩、虚拟腕部六维 wrench 和可用电气信号判断成功 → 退回
Home。多型号的配对、抓取区、插入轴、旋向、角度和损伤阈值由默认禁用的型号注册表
逐项提供，不能让 RL/VLA 自行猜测安全参数。

当前简化螺旋代理已通过 20°、60°、120° 单行程两回合物理课程。最终复核中
stage20 约为 `20.002° / 0.214 mm / 0.5 s HOLD`，stage120 约为
`120.223° / 1.319 mm / 2.0 s HOLD`；stage60 先前也以约
`60.075° / 0.652 mm / 1.0 s HOLD` 通过。每个 stage 都检查实际螺母角和轴向进给，
并要求 q7 初值/计划终点合法且终点外仍有至少 10° 命令余量。复核命令为：

```bash
cd ~/WorkPlace/kcgtest1

# 默认 stage20
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-zero --residual-stage stage20 --episodes 2

# 最长的已验收单行程 stage120
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-zero --residual-stage stage120 --episodes 2
```

当前 final formal stage20 训练目录为
`train_seed42_20260811T172410410847Z`：32 个策略步的 training raw safety 全部纳入
审计并通过，failure reasons 为空；仍为 optimizer update 0、actor delta 0，因此
training evidence false，不能宣称策略已学习。pre-fix train/paired 因 runner 源码
hash 改变已经失效。由新训练目录绑定生成的 final paired20 为
`paired_evaluate_seed10000_20260811T172501510645Z`：zero/trained 均 `20/20`，双方
raw safety failure 0，randomization/signature/seed 均 `20/20`、order `10/10`、
reset/rebuild `40/40`，工程完整性通过；但 improvement 0、McNemar `p=1`、
Clopper-Pearson lower `0.8608916593`，所以 improvement/competence/generalization
声明均为 false。任何正向改善声明仍至少需要 100 对和真实训练更新、统计门、无回退
及全部 raw safety/provenance/runtime 门。

formal reset 另有最后 10 个 solver step 的 post-solver 速度门；episode 安全从
`raw_physics` 子步累计，不再根据终止原因反推。训练与评估还精确绑定 Isaac Sim
运行时、curriculum/resolved stage 和 actor hash，旧 artifact 缺字段或源码 hash
过期会被拒绝。learnability challenge 当前默认禁用，尚未接入 backend/formal，也
未做物理 seed 扫描。

这些结果仍不等于真实螺纹、真实 CAD、视觉闭环、多型号装配或已学会的 RL 策略。
超过 120° 的完整一圈现已采用“释放/q7 回卷/Nut-only 重抓”确定性状态机完成；
详细物理与 formal 契约分别见 [`src/kcg_connector/README.md`](src/kcg_connector/README.md)
和 [`src/kcg_rl/README.md`](src/kcg_rl/README.md)。

### 桌面抓取与 D38999 代理

Isaac 场景现在不再是悬空的连接器：工作台尺寸为 `0.80 × 0.90 m`、台面高度
`0.20 m`，固定端安装在独立夹具上，自由端在相隔约 `0.30 m` 的抓取区自然落桌。
新增的展示/验收入口已经分别覆盖：自由端落桌、KUKA 在 Home 保持、Home 到预抓位、
下降闭手、抬起和 4 s 无支撑保持。

当前型号候选为 MIL-DTL-38999 Series III 的 `D38999/26KJ61SN` 自由插头和
`D38999/20KJ61PN` 固定插座：同为 shell `J/25`、insert `61`、N 键位，且 socket/pin
互补。工程没有复制许可边界不清晰的厂商 STEP；它保存 DLA 公开发布的 `/20H` 与
`/26G` 规格页，并据此生成独立的 public-dimensional proxy。该代理有 61 个可视触点、
简化碰撞和独立 coupling nut，但不是厂商 CAD、制造模型、QPL 记录或航天认证模型。

synthetic connector 的完整抓取物理门仍保留；D38999 代理现已在同一个 Isaac World
中连续完成：Home 出发、真实接触抓取、抬升、移动到固定端、实测位姿对准、物理插入
`8.999 mm` 到 `3.001 mm` engage gap、松手换抓 CouplingNut、三段 `120°` 旋拧与
两次回卷重抓、代理就位判定、松手撤离并返回 Home。最终源码已分别完成一次 headless
和一次 GUI 全链 PASS；GUI 实测 Nut 总进度 `6.3068 rad`、轴向进给 `3.0000 mm`，
回 Home 的机械臂最大关节误差 `0.000165 rad`。

这仍是固定摆位、Isaac 真值位姿和 `3 mm/rev` 解析螺旋代理。键位、螺纹和自锁分别
使用明确标注的代理约束。现已有固定全局虚拟 RGB-D 相机的可选预检，但它没有
接管装配控制；完整链仍使用 `sim_ground_truth` 位姿。没有真实螺纹牙、连续碰撞
证明或真实硬件标定，报告也固定写出 `assembly_success_claimed=false`。因此它
证明的是用户可见的完整代理控制链已经跑通，不能称为真实 D38999 已具备
可靠装配能力。

完整链没有 attachment，也没有在物理开始后写物体 pose。无阻力的解析螺旋会在插入
后换抓和最终座止时自转，因此这些窗口以及分段回卷/最终释放均使用明确标注、最大
`0.05 Nm` 的低力防转/自锁代理；它们不是实际键位或螺纹自锁证据。最终 headless 与
GUI 日志和源码/配置哈希保存在
[`artifacts/kcg_connector/d38999_end_to_end_v1`](artifacts/kcg_connector/d38999_end_to_end_v1/REPORT.md)。

第一份完整流程已经冻结为可回放工程基线：
[`baseline_20260812T111941Z`](artifacts/kcg_connector/d38999_end_to_end_v1/baseline_20260812T111941Z/BASELINE.md)。
该目录包含约 `50 MB` 的源码、配置、资产与证据快照，归档 SHA256 为
`c7125ee77de65befc0bb927da7c543c1d864ed5e536101820db2813983b57e31`；从快照
独立解压后的纯测试仍为 `918 passed`。`replay.sh` 会在 `/tmp` 创建隔离工作区，
强制使用冻结 Python 包并精确校验 Isaac/Python/CUDA/GPU/关键库环境，不会覆盖当前
开发目录。

不覆盖这份冻结基线的 `--smooth-demo` 只压缩 Home 开手和 q7 旋拧、回卷、安全高度
返回的展示时间；接触、插入、HOLD、重抓、撤离和全部安全阈值不变。最新 clean
headless 全链从原基线的 `65311` 个 phase step / `272.129 s` 仿真时间降到
`52547` / `218.945833 s`，约减少 `19.54%`，仍通过完整物理和安全门。
同一 smooth + RGB-D preflight 组合也已完成一次 GUI 全链 PASS。

固定全局虚拟 RGB-D 预检使用同一渲染产物的 RGB、深度和 renderer semantic mask，
以 mask 中位像素射线与已登记高度平面相交，只估计世界系 XY。独立验收中自由端/
固定端 XY 误差分别为 `2.500 mm / 1.824 mm`，小于 `10 mm` 门。同 World 预检也已在
首个机械臂动作前通过，没有 reset/clear World 或写物体 pose，随后的 smooth 完整链继续
PASS。它不是 FoundationPose、点云 6D、手眼相机或视觉控制；键位/方向仍未由视觉
观测。可直接查看 [RGB](artifacts/kcg_connector/d38999_rgbd_bootstrap_v1/rgb.png)、
[semantic mask](artifacts/kcg_connector/d38999_rgbd_bootstrap_v1/semantic_preview.png) 和
[depth preview](artifacts/kcg_connector/d38999_rgbd_bootstrap_v1/depth_preview.png)。

可直接目视检查：

```bash
cd ~/WorkPlace/kcgtest1

# D38999 两端、桌面、夹具和 KUKA Home
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_tabletop_robot_smoke.py \
  --gui --keep-open

# 固定全局虚拟 RGB-D 独立预检和可视化产物
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_rgbd_bootstrap_smoke.py
# 期待：ISAAC D38999 RGBD BOOTSTRAP V1 PASSED

# 已验收的 clean headless：同World RGB-D预检→Home→抓取→插入→旋拧→返回Home
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py \
  --end-to-end-probe --smooth-demo --pose-preflight masked-rgbd
# 期待：ISAAC D38999 END TO END V1 PASSED

# 已验收 GUI：完整播放后自动关闭
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py \
  --end-to-end-probe --smooth-demo --pose-preflight masked-rgbd --gui

# 若要保留最终 Home 静态画面，在上条命令末尾追加 --keep-open。

# D38999 预置engage + Nut-only重抓 + q7负向20°代理探针
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_nut_regrasp_smoke.py \
  --twist-probe --gui --keep-open

# 同一状态下的q7负向120°单行程代理探针
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_nut_regrasp_smoke.py \
  --twist-probe \
  --twist-config \
  src/kcg_connector/config/d38999_q7_twist_probe_stage120_v1.yaml \
  --gui --keep-open

# 120°后松手、q7回卷并Nut-only再抓（含明确标注的分段间自锁制动代理）
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_nut_regrasp_smoke.py \
  --twist-probe \
  --twist-config \
  src/kcg_connector/config/d38999_q7_twist_probe_stage120_v1.yaml \
  --rewind-probe --gui --keep-open

# KUKA 从 Home 移动到 synthetic 自由端上方预抓位
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_tabletop_home_to_pregrasp_smoke.py \
  --gui --keep-open

# 下降、真实三指接触抓取、抬起和保持
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_tabletop_pick_smoke.py \
  --gui --keep-open
```

单齿画面抖的最新结论仍是“物理已排除、视觉未闭环”：baseline、RTX history=512、
Segment_00 schema-normalized 三组相同物理轨迹中，24 齿各跟踪 5590 steps 且独立
transform anomaly 为 0；但四个同步固定视角的全序列可测 union 只有 16/24，没有
任何 transition 达到 24/24，因此不能说用户看到的背面单齿抖已经消失。Segment_21
既属于视觉缺失齿，又是高 contact impulse outlier；这只支持下一步做隐藏/半透明
手爪的无遮挡诊断和接触激励 A/B，不构成因果结论。正式 [report](artifacts/kcg_connector/d38999_nut_tooth_jitter/four_synced_evidence_v3/report.json)
固定为 `VALID_LIMITED_VISUAL_JITTER_UNRESOLVED`，并拒绝 no-jitter claim。

这里的 Home→抓取→装配→Home 仍是固定场景中的确定性关节轨迹和 Isaac 真值位姿，
不是视觉驱动的碰撞规划。严格 `PoseProvider` 已统一 source/frame/clock/capture、时间戳、
标定 hash 和 JSON-safe provenance 验证；`masked-rgbd + truth orientation` 只能用于预检，
不能通过 CONTROL 用途的门。
当前机器人 USD 关闭自碰，旧 SRDF 又把 80 对 link 标记为 `Never`，所以不能由“外部
误碰为 0”推导整条路径自碰安全。6D 接口已经允许 D38999 的 `sim_ground_truth`，并为
未来 `vision` 保留同一严格 schema；真实相机、手眼/外参标定、学习式目标检测、
FoundationPose 或其他 truth-free keyed 6D、近距离视觉伺服和 object→grasp/assembly
标定变换仍未实现。

30 秒无控制物理稳定性验收：

```bash
ros2 launch kcg_moveit1 physics_sanity.launch.py duration:=30.0
```

在 Gazebo 全系统已经启动的另一个终端中，执行 MoveIt 规划与控制链路冒烟测试：

```bash
ros2 run kcg_moveit1 moveit_smoke_test.py
```

## 控制接口

- 机械臂控制器：`/controller_gazebo_kuka/follow_joint_trajectory`
- 三指手控制器：`/controller_gazebo_hand/follow_joint_trajectory`
- 标准状态：`/joint_states`，发布 7 个机械臂关节和 8 个物理手指关节
- 单轴力矩：`/finger_torque_broadcaster/joint_states`，顺序为
  `f1j2, f2j1, f3j2`，数值位于 `effort` 字段
- 物理验收状态：`/physics_joint_state_broadcaster/joint_states`，仅在稳定性测试中启动

三指手共有 4 个主动关节和 4 个从动关节：

| 从动关节 | 主动关节 |
| --- | --- |
| `f1j3` | `f1j2` |
| `f2j2` | `f2j1` |
| `f3j1` | `f1j1` |
| `f3j3` | `f3j2` |

## Gazebo 启动参数

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `gui` | `true` | 启动 Gazebo 图形界面 |
| `use_rviz` | `true` | 启动 RViz |
| `start_moveit` | `true` | 启动 `move_group` |
| `start_trajectory_controllers` | `true` | 启动机械臂和手部轨迹控制器 |
| `run_stability_check` | `false` | 自动执行物理稳定性检查并退出 |
| `stability_duration` | `30.0` | 检查时长（仿真秒） |
| `world` | `physics.world` | 指定 Gazebo world 文件 |

## 已知提示

- ROS/MoveIt/Gazebo 链路当前没有深度相机或点云传感器，因此 MoveIt 启动时会提示
  没有 Octomap 3D sensor plugin；这不影响当前关节规划和轨迹执行。上述 Isaac 固定全局
  虚拟 RGB-D 预检是独立链路，并没有接入 MoveIt Octomap。
- 本机 MoveIt 2.5.9 在 `Ctrl+C` 退出 `move_group` / MoveIt RViz 插件的析构阶段
  可能打印 `class_loader` 警告并以 `-11` 退出。规划、执行、显示与 Gazebo
  运行期间已通过测试；该现象只发生在关闭阶段。
- 手掌、近端指节和三个末端均使用由原始网格生成的凸包碰撞体；没有额外指腹盒体，
  也没有指尖触觉传感器。力感知仅模拟 `f1j2`、`f2j1`、`f3j2` 三处惠斯通全桥的
  单轴力矩。对接实体前仍需标定零点、增益、温漂、重力补偿和量程。
- RViz 默认可能选中 `hand` 规划组；该组没有逆运动学交互标记。需要拖动机械臂末端时，
  在 MotionPlanning 面板中选择 `kuka`。

完整迁移内容和验证结果见 [MIGRATION_ROS2.md](MIGRATION_ROS2.md)。
