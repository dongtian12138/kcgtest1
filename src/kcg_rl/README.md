# KCG 强化学习适配层

`kcg_rl` 同时保存旧的 Gazebo 圆柱课程适配器，以及 Isaac 电连接器 residual
环境的严格 Gymnasium 边界。两套运行时隔离：ROS 2 Humble/Gazebo 使用系统
Python 3.10，Isaac Sim 使用独立 Python 3.12 和它自带的 GPU PyTorch。

## Gazebo 圆柱课程

圆柱适配器复用 `kcg_grasping` 的同一个 Gazebo 世界、27 维任务观测和成功判据，
不复制机器人模型，也不改变脚本基线的验收标准。

该启动文件固定启用 `fast_reset:=true` 和
`start_at_cylinder_pregrasp:=true`。因此 RL 仍使用原有快速模式：圆柱启动时停放在
工作区外，每回合 reset 才传送到预抓位；不会执行演示模式的远距离分段接近。自然
演示和训练复位由启动参数隔离，物理抓取与成功判据保持相同。

## 当前接口

- 动作：5 维归一化连续动作，范围均为 `[-1, 1]`。
  - `0–3`：`f1j1, f1j2, f2j1, f3j2` 的绝对目标位置，分别映射到
    `[0.96,1.04]、[0.70,0.80]、[0.46,0.54]、[0.70,0.80] rad`。这是第一阶段
    围绕已验证抓取姿态的安全残差范围，后续课程再逐步放宽；
  - `4`：抬升触发量，达到 `0.5` 后执行一次已验证的完整抬升轨迹。
- 观测：41 维 `kcg_cylinder_rl_observation_v1`。
  - 前 27 维与 `kcg_grasping` 完全相同；
  - 后 14 维为 7 个 iiwa 关节的位置和速度。
- 步进：手指目标变化时执行一条完整的 5 s 物理轨迹；如果同一步还请求抬升，环境
  会先额外等待 1 s 预紧，再执行完整的 8 s 抬升轨迹，绝不并发执行两条轨迹；目标
  不变时推进 1 s 用于观测和保持判定。
- 同步：环境根据 `/clock` 等待所需仿真时间，episode 内保持 Gazebo 连续运行。
  Gazebo Classic 在刚性接触中频繁暂停/恢复会导致旧手模型数值发散，因此不能把
  每一个策略决策实现成一次物理暂停。
- 终止：任务成功、圆柱掉落、抓取丢失或出现非有限物理状态。
- 截断：达到 episode 最大步数。

没有使用吸附、临时固定或指尖触觉。力观测仍然只有 `f1j2, f2j1, f3j2`
三路单轴力矩。

## ROS 2/NumPy 冒烟测试

```bash
cd ~/WorkPlace/kcgtest1
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch kcg_rl cylinder_rl.launch.py \
  gui:=false use_rviz:=false \
  run_smoke:=true smoke_episodes:=2 shutdown_on_completion:=true
```

连续看到两次 `RL STEP SMOKE PASSED`，才表示同一个环境对象能够完成两次
reset/step/奖励/终止流程；这同时检查第二回合不会继承上一回合的抓取状态。

`KcgCylinderEnv.reset()` 和 `step()` 已采用 Gymnasium 返回约定；训练时使用
`GymnasiumKcgCylinderEnv`。

## GPU 训练环境

本机不重复安装 PyTorch。项目 `.venv` 继承已经为 RTX 5070 Ti 验证过的 HaMeR
环境，其中为 `torch 2.7.0+cu128`；Gymnasium 1.2.3 和
Stable-Baselines3 2.7.1 只安装在本项目 `.venv` 中，不写入 HaMeR 环境。

当前已验证：

- `torch.cuda.is_available() == True`；
- GPU 名称为 `NVIDIA GeForce RTX 5070 Ti`；
- CUDA 张量计算、ROS 2 `rclpy` 初始化以及 Gymnasium/SB3 导入均正常；
- 训练程序默认要求 `cuda`，GPU 不可用时直接报错，不会静默退回 CPU。

需要重建训练环境时执行：

```bash
cd ~/WorkPlace/kcgtest1
export HAMER_ENV_PREFIX=/path/to/hamer/.conda-env
"${HAMER_ENV_PREFIX}/bin/python" -m venv \
  --system-site-packages .venv
.venv/bin/python -m pip install \
  -r src/kcg_rl/requirements-training.txt
```

训练分两个终端运行。终端 A 启动任务仿真：

```bash
cd ~/WorkPlace/kcgtest1
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch kcg_rl cylinder_rl.launch.py \
  gui:=false use_rviz:=false run_smoke:=false \
  shutdown_on_completion:=false
```

终端 B 先做两步 GPU 反向传播冒烟测试：

```bash
cd ~/WorkPlace/kcgtest1
source /opt/ros/humble/setup.bash
source install/setup.bash
source .venv/bin/activate
python -m kcg_rl.sac_train --mode smoke --timesteps 2
```

看到 `SAC TRAIN SMOKE PASSED` 才表示 Gazebo → Gymnasium → SB3 replay buffer →
GPU SAC 更新 → 模型保存的整条链路通过。模型和运行元数据写入
`artifacts/kcg_rl/cylinder_sac/`。

正式训练与评估入口为：

```bash
python -m kcg_rl.sac_train --mode train --timesteps 10000
python -m kcg_rl.sac_train --mode evaluate --episodes 10 \
  --model-path artifacts/kcg_rl/cylinder_sac/cylinder_sac.zip
```

当前仍是单 Gazebo 实例训练。并行实例需要先完成 ROS/Gazebo 命名空间和端口隔离，
不能直接启动多个同名节点。

## Isaac Python 导入边界

`kcg_rl` 的包初始化不会主动导入 `rclpy`。因此 Isaac Sim 的 Python 3.12 可以
导入后续的纯 RL 任务模块；只有显式访问旧 Gazebo 适配器
`CylinderEnvConfig`/`KcgCylinderEnv` 时才会加载 ROS 2 Python 模块。

如果 Isaac 环境尚缺 Gymnasium/SB3，只安装专用 requirements 中逐项锁定的
非 PyTorch 包，并禁止 pip 自行解析依赖：

```bash
cd ~/WorkPlace/kcgtest1
src/kcg_connector/isaac/run_isaac_python.sh -m pip install --no-deps \
  -r src/kcg_rl/requirements-isaac-rl.txt
src/kcg_connector/isaac/run_isaac_python.sh -m pip install --no-deps \
  -e src/kcg_rl

src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/verify_isaac_rl_runtime.py
```

这里的 `--no-deps` 是必要保护：它确保 pip 不会替换 Isaac 环境自带的 GPU
PyTorch。第二条命令只把本工作区的 `kcg_rl` 以 editable 方式注册给 Isaac
Python；专用 requirements 已把本机缺少的 Gymnasium/SB3 运行依赖逐项锁定，
并且不声明、下载或安装任何 PyTorch 版本。安装后必须运行 CUDA 反向传播检查。

当前 formal artifact 记录的精确 Isaac 运行时为：Python `3.12.13`、Isaac Sim
`6.0.1.0`、NumPy `2.3.1`、Gymnasium `1.2.3`、Stable-Baselines3 `2.7.1`、
PyTorch `2.11.0+cu128`、CUDA build `12.8`、GPU
`NVIDIA GeForce RTX 5070 Ti`。训练会冻结这些字段，evaluate/paired 会在加载模型前
逐字段精确匹配；字段缺失或版本不一致均失效关闭。这里是 Isaac 专用环境，不要与
前面 Gazebo `.venv` 继承的 HaMeR `torch 2.7.0+cu128` 混为一套运行时。

## Isaac 电连接器 residual 环境

`kcg_rl.connector_residual_env.ConnectorResidualEnv` 是正式的 Gymnasium 薄边界：

- 动作严格为 `float32 (4,)`，观测严格为 `float32 (24,)`，值域均为
  `[-1, 1]`；
- 拒绝 shape、dtype、有限值、reward、终止标志或 info 不符合约定的后端输出；
- 强制 `reset()` 后才能 `step()`，episode 结束后必须重新 reset；
- 顶层不导入 ROS、Isaac 或 PyTorch；物理场景、reset、观测、奖励和安全统计全部
  由唯一的 `kcg_connector.isaac_residual_backend` 实现；
- Gym `close()` 只关闭这个借用者，`SimulationApp` 生命周期仍由外层 Isaac
  runner 统一管理。

纯边界测试命令：

```bash
cd ~/WorkPlace/kcgtest1
source /opt/ros/humble/setup.bash
source install/setup.bash
export PYTHONPATH="$PWD/src/kcg_rl:$PWD/src/kcg_connector${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest -q src/kcg_rl/test
```

当前基线为 `109 passed, 1 skipped`。跳过项只表示系统 Python 没有可选的
Gymnasium；Isaac Python 中的实际 Gym/SB3 链路由下面的物理命令验收。实际
Isaac/SB3 联调入口仍由连接器 runner 创建唯一场景：

```bash
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-sac-smoke --residual-stage stage20 \
  --training-timesteps 32
```

这条 32-step 命令只验收 Gym → replay → CUDA 更新 → 保存/重载链路；不代表策略
已经学会旋拧或已经收敛。

## Already-engaged residual v0 正式 SAC 运行契约

`config/connector_residual_sac.yaml` 固定了 already-engaged residual v0 的 SAC、
GPU、seed、产物和确定性评估约定；`--residual-stage` 默认 `stage20`，也可显式选择
`stage60`/`stage120`，三者共用冻结的 4D 动作、24D 观测、唯一 backend/场景和同一
简化螺旋代理，不增加重抓。课程定义来自
`../kcg_connector/config/connector_residual_curriculum_v1.yaml`；独立的
`../kcg_connector/config/connector_residual_randomization_v1.yaml` 定义第一批
seed 驱动扰动。算法编排在纯 Python 模块
`kcg_rl.connector_residual_sac` 中，物理仍只由唯一的
`ConnectorResidualIsaacBackend + ConnectorResidualEnv` 实现。当前场景构建尚未
抽成独立 builder，因此 `connector_q7_twist_smoke.py` 暂时也是 formal 模式的唯一
场景 owner；formal 入口没有复制第二份 Isaac 场景。

resolved stage 决定目标角、HOLD、最大步数和最小轴向进度比例。backend 在执行前
同时检查 q7 初值与计划终点；非有限、越界或终点不能保留至少 `10°` 关节命令余量
都会失效关闭。training/evaluate/paired 归档 curriculum YAML bytes/SHA256、resolver
源码和 resolved stage 文档/SHA256，评估时要求 stage 与全部 provenance 精确相等。

随机化 v1 不改变 4D 动作或 24D 观测尺寸，每回合抽取三轴夹持标称位置
`±0.010 rad`、三轴 Kp/Kd `±5%`、三路力矩 bias `±0.005 Nm`、力矩噪声
`σ=0.002 Nm`（`3σ` 截断），以及 0/1 个 policy-step 动作延迟。所有过载、
丢失抓持、终止、reward 和 episode safety 判据继续读取无噪声的
`raw_physics`；质量、摩擦和螺距仍固定，不能称为真实物理参数域随机化。每个
episode 的 raw safety report 从初始快照起累计 240 Hz 物理子步和 10 Hz policy
边界峰值；formal 报告严格校验字段、类型、有限值、signal source 及 final info
投影，不再从 termination reason 反推安全。

每次 hard reset 还必须通过 post-solver settle 门：最后 10 个 settle step 中 body
线/角速度、nut 线/角速度和 q7 速度分别不超过 `0.010 m/s`、`0.005 rad/s`、
`0.060 m/s`、`0.25 rad/s` 和 `0.5 deg/s`。任一字段缺失、非有限或超限都会拒绝
checkpoint，不能进入训练/评估。

先运行两类短门。零动作使用标准 seed 流：第一回合显式设 base seed，后续回合不
重新播种，因此同一运行会得到不同参数、重跑同一 base seed 又能重现完整序列。
最新 seed `1201` 两回合分别覆盖 0/1-step 延迟并 `2/2` 成功；动作因果仍让六个
case 使用同一个 seed `1204`，避免域差异污染动作比较：

```bash
cd ~/WorkPlace/kcgtest1

src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-zero --residual-stage stage20 --episodes 2 \
  --residual-randomization-config \
    src/kcg_connector/config/connector_residual_randomization_v1.yaml \
  --reset-seed 1201

src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-action-effect --residual-stage stage20 \
  --action-effect-steps 10 \
  --residual-randomization-config \
    src/kcg_connector/config/connector_residual_randomization_v1.yaml \
  --reset-seed 1204
```

两条命令分别期待 `ZERO-RESIDUAL ... PASSED` 和
`RESIDUAL ACTION EFFECT PASSED`。它们只验证可复现采样、基线安全和动作因果，
尚未证明策略学习、成功率或泛化能力有所提高。

再做不会执行优化更新的 32-step formal dry-run。`residual-train`、
`residual-evaluate` 和 `residual-paired-evaluate` 默认加载 seed v1；如需创建或匹配
当前版本的完全固定域 artifact，必须显式加入 `--fixed-residual-domain`：

```bash
cd ~/WorkPlace/kcgtest1
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-train --residual-stage stage20 \
  --formal-timesteps 32
```

这一步验证正式配置解析、Gym checker、GPU/有限值 CUDA backward、32 个 replay
样本、模型和 replay buffer 分别保存、模型重载、reset snap/post-solver、raw safety、
唯一场景生命周期以及源码/配置/curriculum/USD 的 SHA256。正式配置的
`learning_starts=1000`，所以这个 dry-run 应当是 `optimizer_updates=0`、actor 参数
不变；它不能替代上面的 SAC32 更新冒烟，更不能说明策略已经学会任务。

每次 formal 训练都会建立唯一目录：

```text
artifacts/kcg_connector/residual_sac_v0/train_seed42_<UTC>/
  requested_curriculum_config.yaml
  requested_training_config.yaml
  resolved_curriculum_stage.yaml
  resolved_training_config.yaml
  resolved_randomization_config.yaml
  monitor.monitor.csv
  final_model.zip
  replay_buffer.pkl
  training_metadata.json
```

源码和 YAML 的 bytes/SHA256 在 `check_env` 和 `model.learn` 前一次性冻结；实际
加载的 curriculum stage 与随机化 dataclass 分别另存 resolved YAML，防止长运行
期间文件变化造成错误 provenance。元数据记录 policy/environment seed、4D/24D
接口、精确 Isaac 运行时、全部 actor/critic 参数设备、reset snap/post-solver、raw
safety、模型/replay/source/config/curriculum/asset hash，以及初始、最终、保存后重载
actor state SHA256，并明确 `VecNormalize` 未使用。Gym checker 的 preflight reset
与真正进入训练阶段的 reset 分开计数；seed v1 启用时，训练首/末样本只来自
`model.learn`，同时明确 `mass/friction/thread_lead=false`。同 seed 的采样参数可
复现，但不能宣称 PhysX/GPU 跨运行逐 bit 重现或策略具有泛化能力。

当前可复核训练目录为
`artifacts/kcg_connector/residual_sac_v0/train_seed42_20260811T172410410847Z/`。
training raw safety 为 `passed=true`，32 个策略步全部纳入审计，覆盖 1 个 complete
和 1 个 partial episode，failure reasons 为空；峰值为三路指根力矩
`0.73928255 Nm`、q7 速度 `0.18252415 rad/s`、螺母速度 `0.19348097 rad/s`、q7
跟踪误差 `0.00515053 rad`、抓持平移/旋转漂移
`4.9777e-5 m / 0.000248894 rad`。该 dry-run 仍是 `optimizer_updates=0`、actor delta
0，所以正向策略改善所需的 training evidence 为 false。

确定性评估只接受完整训练目录，不接受脱离元数据的裸模型：

```bash
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-evaluate --residual-stage stage20 \
  --formal-run-dir \
    artifacts/kcg_connector/residual_sac_v0/train_seed42_20260811T172410410847Z \
  --evaluation-episodes 20
```

评估固定调用 `predict(..., deterministic=True)`，并复用零残差基线的物理成功和
失败统计：目标螺母角、q7/螺母耦合、螺旋进给误差、三路已加载指根力矩、raw
safety、有限状态和关节限位。它还校验加载 actor state 与 training metadata 中的
最终/重载 hash。源码、任务/formal/curriculum/resolved stage 配置、USD 或精确
运行时与训练时不一致，都会在 episode 前拒绝评估。

评估默认使用与当前 formal 训练相同的 seed v1；训练 metadata 与评估时的随机化
开关或 provenance 不一致会被拒绝。`--fixed-residual-domain` 只用于匹配由当前代码
和当前 provenance 契约生成的固定域产物，不能让缺少 curriculum/runtime/raw
safety/actor hash 证据或源码 hash 已变化的旧 artifact 重新有效。

只有当前源码生成、通过 raw safety/provenance/runtime/actor binding 全部门的完整
训练目录才可进入评估；旧目录的历史成功计数不能当作当前证据。

正式 paired benchmark 对顺序做确定性 counterbalance：一基计数的奇数 pair 运行
`zero → deterministic model`，偶数 pair 运行反向顺序。pair 内两次 reset 显式使用
同一 seed，比较完整实际随机参数和 reset 初始 body/nut/q7 签名，并核对预期顺序、
reset 次数和螺旋代理重建次数：

```bash
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-paired-evaluate --residual-stage stage20 \
  --formal-run-dir \
    artifacts/kcg_connector/residual_sac_v0/train_seed42_20260811T172410410847Z \
  --evaluation-episodes 20
```

当前 final paired 产物为
`artifacts/kcg_connector/residual_sac_v0/paired_evaluate_seed10000_20260811T172501510645Z/`，
严格绑定 `train_seed42_20260811T172410410847Z`。20 对结果为：zero/trained 都是
`20/20`，双方 raw safety failure 0；randomization、reset 初始签名和 seed 均匹配
`20/20`，两种顺序各 `10/10`，reset/螺旋代理重建 `40/40`。因此
`benchmark_integrity_passed=true`；但改善为 0、单侧 exact McNemar `p=1`、trained
单侧 95% Clopper-Pearson 下界为 `0.8608916593`，training evidence 也为 false，故
policy improvement、competence 和 generalization claim 全部为 false。

这里终端的 paired benchmark `PASSED` 只表示 20 对工程完整性门通过。正向策略改善
声明至少需要 `--evaluation-episodes 100`，还必须同时满足：

- training metadata 证明越过 `learning_starts`、optimizer updates 足够、actor 参数
  有非零有限变化、初始/最终 hash 不同且最终与重载 hash 一致；
- trained 原始成功率和单侧 95% Clopper-Pearson 下界均至少 95%，相对 zero 至少
  提升 10 个百分点，单侧 exact McNemar `p <= 0.05`，且没有回退 pair；
- zero/trained 双方 raw safety failure 均为 0，全部 seed、randomization、reset 初始
  签名、counterbalanced order、provenance、actor binding 和 runtime 门通过。

任何字段缺失、非法布尔/数值、NaN/Inf 或跨 episode 状态不一致都会让声明失效关闭。
当前第一版扰动使 zero baseline 饱和；在加入策略可观测、可补偿且通过物理因果门的
更难扰动以前，不允许启动长训练。

### 默认禁用的 learnability challenge

`../kcg_connector/config/connector_residual_learning_challenge_v1.yaml` 与纯 Python
契约只冻结后续可学习性定标方案，默认 `enabled: false`，现有 backend/formal runner
不会读取它。它保持 residual-v0 4D/24D、20°/0.5 s/10 Hz，使用 28 步 deadline、
q7 control-path scale `[0.85, 1.15]` 和三路 clamp offset `±0.015 rad`；质量、摩擦、
螺距仍固定。只有 zero 成功率 65–85%、oracle 至少 98%、双方 raw safety failure 0、
最终 100 对 randomization 全匹配且 zero 失败仅为 `time_limit`，才允许评审接入
runtime。deadline penalty `-10` 不是 safety failure；当前尚未做物理 seed 扫描，
不能据此宣称任务可学习。

formal 训练不提供默认步数；超过 32 步还必须同时显式加入
`--allow-long-training`。在真实物理参数随机化和有学习更新的多 seed 评估接入以前，
不应启动长训练。当前尚不支持可复现续训；虽然 replay buffer 单独保存，但没有恢复所有
Python/NumPy/Gym/PyTorch/PhysX RNG 状态。`PhysicsUSD disjointed body transforms`
也尚不能在进程内可靠计数，formal 元数据将该值写为 `null`，必须另行检查对应 Kit
日志，不能把 `null` 解读为零 warning。

## 完整视觉装配的 RL 训练门禁

现有 residual v0 不是完整装配环境：每回合从已插合、已抓稳的位置开始，只学习
q7 旋紧速度和三路夹持残差；它不负责从 Home 找零件、视觉定位、抓取、插入或回
Home，24 维观测也没有 RGB-D pose 或腕部六维 wrench。不得把 v0 的 SAC smoke、
`20/20` already-engaged 成功或 GPU 反向传播说成完整视觉装配已经可训练。

`config/d38999_full_skill_rl_readiness_v1.yaml` 定义了独立、默认关闭的分层 v1：
确定性 FSM 负责 RGB-D 定位、抓取、自由空间运动、预对准、插入、验收和回 Home；
4D residual 策略只在 `ENGAGE/SCREW` 阶段启用。策略接口沿用 4D 动作，不修改 v0，
但使用独立的 `kcg_connector_twist_residual_wrist_ft_v1` 30D 观测，即 v0 的 24D
末尾追加补偿后、变换到连接器任务系的六维腕部 wrench。当前只冻结接口和训练门，
没有在补偿、时序和安全反事实证据齐备前伪造一个补零的 30D consumer。

长训练前必须运行纯 Python 检查：

```bash
cd ~/WorkPlace/kcgtest1
export PYTHONPATH="$PWD/src/kcg_rl:$PWD/src/kcg_connector${PYTHONPATH:+:$PYTHONPATH}"
python3 -m kcg_rl.full_skill_readiness \
  --config src/kcg_rl/config/d38999_full_skill_rl_readiness_v1.yaml \
  --repo-root "$PWD"
```

只有输出 `FULL SKILL RL READINESS: READY` 且退出码为 0，完整技能 runner 才能开始
长训练。当前仍预期输出 `BLOCKED` 和退出码 1。检查器现在会显示四组
`VALID LIMITED`：多位置 RGB-D 5/5、三次 monitor-only 腕部 F/T 重复性、三次
smooth headless E2E，以及主动的六视角+Segment_23 identity v2 齿诊断。v2 对真实
执行链逐层重哈希：base/axial capture manifest、ghost manifest、run log、physics
report/summary、1590 张 RGB 的间接逐 PNG 哈希图，以及 capture/ghost/prepared runner
在 import/start/finalize 时记录的源码 SHA 都必须与 readiness 外层冻结字节一致；源码
发生漂移时直接失效，不能刷新 hash 来“追上”当前文件。

齿物理证据是单次 `5590 × 24` trace、`anomaly_steps=0`。原生六视角 RGB transition
分析的全序列 identity union 是 23/24，只缺 `Segment_23`。后处理使用 connector CAD
局部中心、每帧真实 CouplingNut parent physics pose 和固定相机投影，通过 mutual-nearest、
小于 1/3 projected pitch 的误差以及大于 1/2 pitch 的 identity margin，在已有 265 帧中
恢复 `Segment_23`；因此“六视角 union + CAD/physics-assisted posthoc identity”的序列
union 可以写成 24/24，但它明确不是 RGB-only。逐 transition 重算仍是 0 个 24/24；
阈值是事后 identity 对应门，不是预注册的 render-jitter 接受门。故检查器强制保留
`render_jitter_absence_claim_authorized=false` 和 `render_jitter_remains_unresolved`，
不会把 identity 恢复写成“用户看到的单齿画面抖已经排除”。

`VALID LIMITED` 绝不等于正式 gate PASS，也不会成为 training-ready 声明。当前 RGB-D
仍只有 mask-derived XY，
没有 keyed yaw/full 6D，也没有进入控制；smooth E2E 仍用仿真真值控制，连续碰撞
验证为 false，并且只有三个固定场景 headless 样本；F/T 仍是 monitor-only，阈值为
null，动态补偿、安全门和 residual v1 都未启用；齿画面诊断也没有预注册可接受
residual 阈值，30 Hz 不能排除采样间渲染伪影。物理 240 Hz 零异常只说明没有观测到
齿相对 CouplingNut 的独立物理运动，不能代替未完成的 renderer 结论。桌面随机化闭环
也尚未完成。因此
`training.enabled=false` 保持不变，八个正式 gate 仍要求各自完整证据。缺失、NaN、
越界、schema 越权、路径逃逸或任一已绑定产物改动都会失败关闭。

这套门禁保护的是“完整视觉装配长训练”。它不改变冻结基线，也不阻止继续运行短的
residual-v0 回归和物理诊断。最终启用时仍应在实际训练入口最前面调用
`kcg_rl.full_skill_readiness.require_training_ready()`，不能只依赖人工看日志。
