# 当前三指手与连接器：保存版复现说明

> 本改进分支默认采用 10 毫秒负载补偿、关节接触最后求解和无损记录回收调度，最多六次抓握，总腕部指令仍限 360°。最新连续运行的深度、原 2 微米键槽检查和松手记录见 [完整验证说明](ASSEMBLY_COMPLETE_VALIDATION_20260917_CN.md)。原保存版仍在 `codex/connector-assembly-baseline-20260916` / `assembly-baseline-20260916`；下面标为原版的历史结果与耗时属于该保存版。

当前已取得两次同场景完整通过，实际结果和控制解释见[重复验收说明](ASSEMBLY_REPEATABILITY_20260917_CN.md)。随后加入的改动只减少审计检查的中间数组和收尾读取开销，保持原控制、物理及数据。1mm/1°初始位姿变化已经执行，在初始手指接触阶段因关节速度保护停止，未完成抓起；不能把它计为第三次装配成功。[四项工作及变化失败说明](ASSEMBLY_FOUR_ITEMS_DELIVERY_20260917_CN.md)、[第二轮成功及变化测试视频](https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-repeatability-20260917)均单独保留结果边界。

第一次完整验收通过的公开交付：[assembly-first-verified-20260917](https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-first-verified-20260917)。含本轮 307 秒完整视频、21 秒末段片段、实际末段图片和验收证据包；下载附件即可观看，不需要先花数小时重新计算仿真。发布标签为 `baf6d4f5c2c7035b4fd17ba38e0a19bd04ea7455`，其中记录的实际物理运行提交为 `25a0d6f`；两者仅差记录收尾异常处理和交付证据，控制与物理配置相同。

该标签已从公开 GitHub 克隆到另一干净目录，并重新下载核验 297 个运行资产，准备固定 SAM-6D 源码/补丁与四个经校验的缓存权重，构建机器人 ROS 资源，通过名义和 1mm/1°配方的启动文件检查。合计 768 项文件绑定，其中 471 项随 Git 发布的文件、297 项单独打包的运行资产。此次复用了本机三个第三方 Python 环境，没有另跑物理预检或完整装配，也没有声称全新机器安装已验证；详细记录见 [干净目录检查](../reproducibility/improvements_20260916/first_verified_fresh_checkout_check.json)。

这份基线保留 2026-09-16 的同一回合装配：从桌面视觉识别、抓取、搬运、插入键槽、换抓螺母、五段旋拧，到张手后保持 3 秒。最终本体深度 **14.604491442 mm**，目标是 14.605 mm。

**结果有明确边界：机械到位和真实松手已经发生；原键槽 2 µm 数值比较带仍有一项未通过。** 第三键最小间隙 −2.147659492 µm，超带 0.147659492 µm。保存版没有改阈值，也没有把这项失败改成通过。它是后续提速和排查的比较基线，不是硬件验证或重复成功率保证。

## 1. 下载指定版本与运行资产

复现当前改进版时，先下载改进分支，再按下文准备同一套运行资产和环境：

```bash
git clone --branch codex/connector-assembly-improvements-20260916 https://github.com/dongtian12138/kcgtest1.git kcgtest1-improvements
cd kcgtest1-improvements
python3 scripts/prepare_assembly_reproduction.py --assets
```

下面这组命令则用于复现原封存基线，二者选择一个独立目录：

```bash
git clone --branch codex/connector-assembly-baseline-20260916 https://github.com/dongtian12138/kcgtest1.git kcgtest1-baseline
cd kcgtest1-baseline
python3 scripts/prepare_assembly_reproduction.py --assets
```

资产来自同仓库 `assembly-baseline-20260916` Release。脚本检查压缩包和每个解压文件的 SHA256，已有不同内容的文件会报错，不会被覆盖。约 121 MB 压缩包包含实际使用的连接器、机器人 USD 引用链、原始 STEP/碰撞面、视觉模板和配置。已跟踪的源码与机器人网格随 Git 下载。

原始运行源码另存于 `reproducibility/assembly_20260916/execution_sources/`。可移植版本只修正路径与环境选择；原始参考路径元数据保留在 `original_path_metadata/`。不改几何、质量、惯量、摩擦、关节、控制参数或原检查带。

## 2. 使用本机已验证的三个环境

三个环境用途不同，不要合并安装或顺手升级到最新版。当前本机可直接使用：

```bash
export ISAAC_ENV_PREFIX=/home/noob/WorkPlace/isaacsim/.conda-env
export KCG_PLANNER_PYTHON=/home/noob/WorkPlace/kcgtest1/.venv/bin/python
export KCG_SAM6D_PYTHON=/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python
```

上面两处旧工程/缓存路径**只选择已安装的第三方 Python 环境**。启动脚本强制从当前下载目录加载项目源码，机器人资源也在当前目录重新构建；不使用旧工程的模型或旧运行结果来冒充新运行。

| 用途 | 已验证版本 |
| --- | --- |
| Isaac Sim | Python 3.12.13；Isaac Sim 6.0.1.0；NumPy 2.3.1；SciPy 1.17.0；python-fcl 0.7.0.11；trimesh 4.11.1；msgpack 1.2.2 |
| 视觉 SAM-6D | Python 3.10.12；PyTorch 2.7.1+cu128；torchvision 0.22.1+cu128；NumPy 1.26.4；SciPy 1.15.3；trimesh 4.0.8；已编译 PointNet2 |
| 规划 Tesseract | Python 3.10.20；tesseract-robotics-nanobind 0.35.0.7；NumPy 1.23.5；SciPy 1.10.1；python-fcl 0.7.0.8；xacro 2.1.1 |

完整实际包清单在 `reproducibility/assembly_20260916/*_python_environment_packages.json`。本机为 Ubuntu 22.04、ROS Humble、RTX 5070 Ti 16 GB、驱动 595.91.07，ffmpeg/ffplay 4.4.2。物理求解在 CPU，渲染和视觉网络需要 GPU。不要为了照抄版本自动替换系统驱动。

在不同机器新建环境时，Isaac 环境须为 Python 3.12，官方固定版本安装入口为：

```bash
python3.12 -m venv /your/path/isaac-env
/your/path/isaac-env/bin/python -m pip install 'isaacsim[all,extscache]==6.0.1.0' --extra-index-url https://pypi.nvidia.com
/your/path/isaac-env/bin/python -m pip install numpy==2.3.1 scipy==1.17.0 python-fcl==0.7.0.11 trimesh==4.11.1 msgpack==1.2.2 opencv-python-headless==4.13.0.90 PyYAML==6.0.3
```

这是 [NVIDIA 6.0.1 官方安装方式](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html)，其余实际包以随版清单核对。规划与视觉分别使用 Python 3.10；SAM-6D 采用下述固定源码和本项目补丁，PointNet2 按其源码 `Pose_Estimation_Model/model/pointnet2/setup.py` 在匹配 CUDA/PyTorch 环境中编译。[SAM-6D 上游说明](https://github.com/JiehongLin/SAM-6D/blob/1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2/README.md)。不要直接套用上游旧 Python 3.9 / torch 2.0 配方来声称等于本次环境。

跨机器从零安装全部环境尚未验证；随版检查结果明确记录同机、独立工程目录复现所使用的环境。首次完整运行可能需要下载 NVIDIA 扩展，缓存、视频和原始逐拍数据也需磁盘空间；原版一次逐拍数据约 19.6 GB。

## 3. 准备同版视觉源码、权重与 ROS 资源

本机已有缓存，可避免重新下载 5.42 GB 权重：

```bash
python3 scripts/prepare_assembly_reproduction.py --sam6d --reuse-sam6d-cache /home/noob/.cache/kcgtest1-sam6d
source /opt/ros/humble/setup.bash
colcon build --packages-select iiwa_description --symlink-install
```

新机器没有缓存时：

```bash
# gdown 用于上游 Google Drive 的 PEM 权重；安装在所选普通 Python 环境中。
python3 -m pip install gdown==6.1.0
python3 scripts/prepare_assembly_reproduction.py --sam6d
```

脚本将 SAM-6D 固定到 `1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2`，应用本次实际使用的两个推理脚本补丁；权重从上游地址取得，四个权重都核对 SHA256。SAM 源码默认使用当前目录 `.deps/SAM-6D/SAM-6D`，也可用 `KCG_SAM6D_ROOT` 指定经过同版检查的安装。ROS 安装位置可通过 `KCG_ROS_SETUP` 指定。

## 4. 检查、打开窗口并运行

```bash
# 只检查文件和环境、打印实际命令，不启动物理动作。
python3 scripts/run_current_hand_assembly.py --check

# 只执行新预检。
python3 scripts/run_current_hand_assembly.py --preflight-only

# 带 Isaac Sim 窗口：新预检通过后，执行一整轮。
python3 scripts/run_current_hand_assembly.py --run --gui

# 无窗口运行并录像。
python3 scripts/run_current_hand_assembly.py --run
```

初始位置变化测试可用 `python3 scripts/run_current_hand_assembly.py --run --pose-variation` 复现；这个明确场景已观察到初始手指接触速度保护停止，不是保证成功的选项。查看已生成的视频无需重新进行数小时仿真。

每次创建新的 `artifacts/reproductions/current_hand_<UTC时间>/`，拒绝覆盖已有结果。`--output-root` 可指定新的输出目录。预检失败会停止，完整动作不会启动。原版 291 秒仿真用了约 257 分钟现实时间；当前改进版这次约 307 秒仿真执行到运动结束用了约 243 分钟，尚不能据此把不同轨迹的耗时差当成同条件提速。当前改进版完整动作使用有限 21600 秒墙钟预算，其中 1200 秒留给保存与收尾，不会自动无限延长。原封存分支仍保留其 18000 秒预算。

完成后可看：

```bash
ffplay artifacts/reproductions/current_hand_<UTC时间>/run/video/assembly_four_view.mp4
```

控制器、接触数据与视频都会写入这一轮新目录。退出码 0、2 或程序写了 `completed` 都不能代替实际装配验收。应重新核对本体深度、松手后手部接触是否为零、保持期间原止挡和簧套是否真实承载、源指甲/指腹、原五键 2 µm 比较带，以及视频。保存版 `evidence/` 中的报告只描述原成功回合，不是新回合的自动通过凭证。

原回合完整视频与末段松手片段随同 Release 提供：`assembly_four_view.mp4`、`final_turn_and_release_actual_clip.mp4`。完整片 291.4 秒，片段 9.6 秒；都来自原实际记录，不是重画的动画。19.6 GB 原始真值档在本机保留，不是启动新回合的必要输入。

仅用于仿真：`hardware_authorized=false`。
