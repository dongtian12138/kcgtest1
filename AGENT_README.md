# 复现指令（给新电脑的 Agent / 执行者）

本包是 KCG 项目的**便携复现包**（生成于 2026-08-14，源机器基线：
Ubuntu 22.04 / ROS 2 Humble / MoveIt 2.5.9 / Gazebo Classic 11 /
Isaac Sim 6.0.1 / torch 2.7.0+cu128（.venv）与 2.11.0+cu128（Isaac））。

## 目标

在新电脑上从零复现当前工程进度，并用项目自带的验收门逐项确认。

## 步骤（按顺序执行，不要跳过）

### 0. 解包与校验

```bash
cd ~
# 校验压缩包未损坏（输出应为 OK）
sha256sum -c kcgtest1_core_20260814.tar.gz.sha256
# 解压到标准位置（必须保持 ~/WorkPlace/kcgtest1 布局）
mkdir -p ~/WorkPlace
tar -xzf kcgtest1_core_20260814.tar.gz -C ~/WorkPlace
cd ~/WorkPlace/kcgtest1
```

### 1. 环境引导（一次性，约 20-60 分钟，视网速）

```bash
bash scripts/bootstrap.sh
```

- 脚本会依次：检测 ROS/GPU → 安装 apt 依赖（需 sudo 密码）→ rosdep →
  colcon 构建 → 创建 .venv（自包含 torch 2.7.0）→ 有 GPU 时创建 Isaac 环境。
- 全部日志在 bootstrap.log。任何一行 `[bootstrap] 失败` 都要先解决再继续。
- 若某阶段已手动完成，可用 `--skip-apt` / `--skip-venv` / `--skip-isaac` /
  `--skip-build` 跳过。
- 无 NVIDIA GPU 的机器：ROS 门可全部通过，Isaac 门不能运行（脚本会自动跳过
  Isaac 环境创建）。

### 2. 验收门 A：CPU/ROS（必做，headless）

```bash
bash scripts/verify_all.sh
```

预期输出（4 个阶段全部 [PASS]，最后一行 VERIFY_ALL_OK）：

```text
[PASS] build
[PASS] physics    （Physics stability check PASSED）
[PASS] grasp      （SCRIPTED GRASP PASSED）
[PASS] rl         （RL STEP SMOKE PASSED x2）
VERIFY_ALL_OK
```

各阶段完整日志在 log_verify/<阶段>.log。总耗时约 10-30 分钟。

### 3. 验收门 B：GPU/Isaac（有 NVIDIA GPU 时必做）

```bash
bash scripts/verify_isaac.sh
```

预期输出（5 个阶段全部 [PASS]，最后一行 VERIFY_ISAAC_OK）：

```text
[PASS] stage20    （ISAAC CONNECTOR ZERO-RESIDUAL 2-EPISODE PASSED）
[PASS] stage120
[PASS] rgbd       （ISAAC D38999 RGBD BOOTSTRAP V1 PASSED）
[PASS] e2e        （ISAAC D38999 END TO END V1 PASSED）
[PASS] replay     （918 passed）
VERIFY_ISAAC_OK
```

各阶段日志在 log_verify_isaac/<阶段>.log。总耗时约 1-3 小时
（Isaac 首次运行要下载 assets，replay 会独立校验环境）。

## 失败时的排查顺序

1. 看失败阶段日志尾部 50 行：`tail -50 log_verify/<阶段>.log`。
2. bootstrap 阶段失败：看 bootstrap.log 对应阶段的报错。
3. 常见原因与对策：
   - 缺 ROS2：装 ros-humble-desktop 后重跑 bootstrap.sh。
   - 缺 sudo/网络：apt 与 pip 阶段都需要外网。
   - Isaac 环境缺失：确认 bootstrap.sh 阶段 4 是否执行成功，
     `ls \${ISAAC_ENV_PREFIX:-~/WorkPlace/isaacsim/.conda-env}/bin/python`。
   - GPU 驱动不对：`nvidia-smi` 必须可用且 CUDA >= 12.8（torch cu128）。
   - replay 失败但其余通过：多为环境校验差异，replay 日志会打印缺哪一项。
4. 不要自行修改源码去"绕过"任何 FAIL——修改后通过不代表复现成功。

## 完成后必须回报的内容

1. `verify_all.sh` 与 `verify_isaac.sh` 的**完整输出**（含汇总行）；
2. 失败阶段的日志尾部（如有）；
3. `nvidia-smi --query-gpu=name,driver_version --format=csv` 与
   `lsb_release -ds` 的输出；
4. bootstrap.log 最后 20 行。

## 包的组成（背景知识）

- `src/`：6 个 ROS2 包 + Isaac 脚本（全部源码）；
- `ros1_original/`：原 ROS1 Catkin 工程（追溯用，不参与构建）；
- `artifacts/`：关键实验产物——冻结可回放基线、RL 训练/评估目录、
  FoundationPose 模型、RGB-D 预检产物、各正式报告。
  **注意**：单齿抖调查/后抓取力数据等约 6.4G 的原始抓图与 steps.jsonl
  不在本核心包内（在完整备份包 kcgtest1_full_20260814.tar 中），
  复现进度不需要它们；
- `requirements_venv_freeze.txt` / `requirements_isaac_conda_freeze.txt`：
  两个环境的完整包版本快照（查版本用，bootstrap 已按关键版本自动安装）；
- `sha256_manifest.txt`：本包全量文件校验清单，
  `sha256sum -c sha256_manifest.txt` 可核对解压后无损；
- 更完整的背景见 `README.md`（项目总说明）与 `备份说明.md`。
