# 高位全局相机1、四相机完整装配基线

本分支：`codex/connector-four-camera-baseline-20260920`。这是2026-09-19实际从桌面抓取到最后3秒完全松手均完成，并通过原物理/几何/影像检查的版本。全过程录像和附件见[本次发布页](https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-four-camera-high-global1-20260920)。

结果边界必须保留：实际运动执行提交为`686734f4d4324993e1fbd2024e95caf2ed437ec6`，末端判定修复为`84e534c96760c59997fad938fbac07d16331188b`。旧软件把松手前后7.36微米落座位移当成释放后不稳定，故原退出码2和“未确认”记录保留；完整物理过程确已成功。末端修复只作用于全部动作结束后的判断，并已用本轮当时可用的15帧视觉数据重放验证，没有声称修复后重新跑过整轮。详见[完整验收](../reproducibility/high_global1_acceptance_20260919/result_CN.md)。

## 1. 下载固定分支和运行输入

```bash
git clone --branch codex/connector-four-camera-baseline-20260920 \
  https://github.com/dongtian12138/kcgtest1.git kcgtest1-four-camera
cd kcgtest1-four-camera
python3 scripts/prepare_assembly_reproduction.py --assets
```

保留完整Git历史，可以检查实际运行提交和判定修复提交。默认启动的是本分支已经修复末端判定的版本。

运行输入包括：现有Release中的297个资产（压缩121.46MB、解压369.34MB），以及随Git提交的高位G1标定模板和背景深度2个文件。下载和启动时均校验文件。高位相机的两个输入与成功回合逐字节一致；不需要复制本机整个`artifacts/`。原约24GB整轮原始数据本机保留，不是启动新一轮的输入；本次发布包含验收记录、同回合图像和完整录像。

## 2. 环境要求和固定版本

本机验证环境为Ubuntu 22.04 x86_64、ROS 2 Humble、RTX 5070 Ti 16GB、NVIDIA驱动595.91.07。物理在CPU运行，视觉和渲染使用GPU。按所用Isaac版本满足显卡/驱动要求，不为照抄数字直接替换已有系统驱动。

需要已安装ROS 2 Humble、`colcon`、`ffmpeg`、`g++`、`fonts-noto-cjk`、Python 3.12与3.10环境管理工具。机器人资源构建使用`iiwa_description`；后续命令假定ROS位于`/opt/ros/humble`，可用`KCG_ROS_SETUP`指定其他安装路径。

三个Python环境应分开。本分支附有发布时机器的完整可见包清单和按角色固定的运行依赖，见[environments](../reproducibility/four_camera_baseline_20260920/environments)。完整清单可能包含ROS系统包，用于核对，不应把其中所有条目直接作为PyPI安装表。

| 环境 | Python | 关键版本 |
|---|---|---|
| Isaac | 3.12.13 | Isaac Sim 6.0.1.0；Torch 2.11.0/cu128；NumPy 2.3.1；Warp 1.13.0 |
| 规划 | 3.10.20 | Tesseract 0.35.0.7；NumPy 1.23.5；SciPy 1.10.1 |
| SAM-6D | 3.10.12 | Torch 2.7.1/cu128；torchvision 0.22.1；NumPy 1.26.4；SciPy 1.15.3 |

若已有匹配环境，直接设置下列三个环境变量，然后跳到第3节。新环境可使用Miniforge/Conda建立独立前缀：

```bash
conda create -y -p "$PWD/.deps/envs/isaac" -c conda-forge python=3.12.13
conda create -y -p "$PWD/.deps/envs/planner" -c conda-forge python=3.10.20
conda create -y -p "$PWD/.deps/envs/sam6d" -c conda-forge python=3.10.12

export ISAAC_ENV_PREFIX="$PWD/.deps/envs/isaac"
export KCG_PLANNER_PYTHON="$PWD/.deps/envs/planner/bin/python"
export KCG_SAM6D_PYTHON="$PWD/.deps/envs/sam6d/bin/python"

"$ISAAC_ENV_PREFIX/bin/python" -m pip install -r src/kcg_connector/requirements-torch-cu128.txt
"$ISAAC_ENV_PREFIX/bin/python" -m pip install \
  -r reproducibility/four_camera_baseline_20260920/environments/isaac-runtime.txt
"$KCG_PLANNER_PYTHON" -m pip install \
  -r reproducibility/four_camera_baseline_20260920/environments/planner-runtime.txt

# 旧Lightning 1.8.1的发布元数据含旧式版本约束，视觉环境固定使用兼容的pip。
"$KCG_SAM6D_PYTHON" -m pip install pip==23.3.2
"$KCG_SAM6D_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
"$KCG_SAM6D_PYTHON" -m pip install \
  -r reproducibility/four_camera_baseline_20260920/environments/sam6d-runtime.txt
```

Isaac 6.0.1的Python 3.12、CUDA版Torch和NVIDIA包源安装方式依据[NVIDIA官方说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html)。Lightning的旧约束可查看[PyPI版本元数据](https://pypi.org/pypi/pytorch-lightning/1.8.1/json)。不要运行旧`bootstrap.sh`并据此认为已经配置好这三个专用环境；该脚本面向早期通用工程。

上述新环境安装步骤用于复现准备。此次验证使用了同机现有第三方环境，并在新工程目录重新恢复输入、编译记录模块和构建资源；没有声称在另一台机器从零安装并跑完了整轮。

## 3. 固定视觉源码、权重及记录依赖

```bash
"$KCG_SAM6D_PYTHON" scripts/prepare_assembly_reproduction.py --sam6d

# 使用匹配cu128的CUDA Toolkit编译PointNet2，CUDA_HOME应指向本机实际安装。
# 例如：export CUDA_HOME=/usr/local/cuda-12.8
"$KCG_SAM6D_PYTHON" -m pip install --no-build-isolation \
  .deps/SAM-6D/SAM-6D/Pose_Estimation_Model/model/pointnet2

"$ISAAC_ENV_PREFIX/bin/python" -m pip install --target .deps/pybind11-2.13.6 \
  -r reproducibility/performance_20260917/requirements-build.txt
"$ISAAC_ENV_PREFIX/bin/python" -m pip install --target .deps/recording \
  -r reproducibility/performance_20260917/requirements-recording.txt
"$ISAAC_ENV_PREFIX/bin/python" scripts/build_contact_copy.py

source /opt/ros/humble/setup.bash
colcon build --packages-select iiwa_description --symlink-install
```

SAM-6D固定提交`1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2`，准备脚本自动应用本项目两个推理补丁，4个上游权重合计约5.42GB，逐一核验SHA256。可选`--reuse-sam6d-cache /已有的/SAM-6D目录`复用同版源码和权重。源码默认从当前工程`.deps/SAM-6D/SAM-6D`加载。

PointNet2编译需要CUDA编译工具链，PyTorch自带的CUDA运行库本身不提供`nvcc`。原生记录模块按已验证PhysX ABI在目标环境重新编译，不复制另一台机器的`.so`。中文字体使用`fonts-noto-cjk`提供的`NotoSansCJK-Regular.ttc`。

## 4. 检查和运行

在后续每个新终端重新设置三个环境变量。然后：

```bash
python3 scripts/run_current_hand_assembly.py --check
python3 scripts/run_current_hand_assembly.py --preflight-only
python3 scripts/run_current_hand_assembly.py --run
# 需要窗口时用 --run --gui；成功基线使用的是无窗口录像方式。
```

`--check`只校验输入和依赖并打印命令，不启动物理。`--run`总是先执行新预检，失败则不开始装配；每次创建新的`artifacts/reproductions/current_hand_<时间>/`。`--output-root`可指定尚不存在的输出目录。预检预算300秒、收尾30秒；完整回合预算21600秒、收尾300秒，沿用成功回合的有限预算。预算是计算机运行时间，不是提高运动速度或力限。首次Isaac启动的缓存准备耗时受机器影响，预检超时会正常停止，应查看日志定位原因。

默认输入明确固定为：

- `reproducibility/initial_contact_fix_20260919/visual_body_balanced_band.yaml`
- `reproducibility/two_key_20260919/assembly_high_global1_plus20.yaml`
- `reproducibility/two_key_20260919/four_camera_two_key_high_global1.yaml`

保持960Hz、64/4求解、原材料/质量/几何、原受力保护和键槽2微米检查。高位全局1、固定全局2、掌心和腕部共4台功能相机；全局2两次观键；两段搬运；录像额外包含仅供观察的主视角。该固定名义场景包含启动前的插座+20°摆放，控制转角由本轮图像计算，不读取该场景设置作为观测。

停止当前回合可在另一个终端创建`本次目录/run/STOP_REQUEST`，等待原有机制保存并收尾。全过程视频位于：

```text
本次目录/run/video/assembly_five_view.mp4
```

本次成功记录是313.8秒录像，计算用了2小时51分38秒；其他机器耗时和接触细节不保证相同，此基线不声明已达到5倍提速。

## 5. 验收新回合

退出码或`completed`不能代替实际装配验收。运行下面的脚本，必须使用**本次新生成**的`run`目录：

```bash
python3 scripts/review_current_hand_assembly.py /本次结果目录/run
```

脚本运行原源面、键槽、四相机、两次观键、搬运及严格3秒释放检查，并从本次原视频抽出3张末段图片到`postrun_evidence/`。先查看这些图片，确认主视角/全局2中红色指示带没有露出；确认后执行：

```bash
python3 scripts/review_current_hand_assembly.py /本次结果目录/run --confirm-band-hidden
```

该明确的图像确认只记录本人实际看过的结果，再执行全量整体审核。最终应同时有`whole_assembly_review.json`的`complete_visual_assembly_verified=true`和`three_second_release_review.json`的`accepted=true`。任何失败都应保留并定位，脚本不会复制封存成功回合的PASS文件到新回合。

## 6. 封存证据

[基线验收报告](../reproducibility/high_global1_acceptance_20260919/result_CN.md)及其`evidence/`随Git发布；[Release](https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-four-camera-high-global1-20260920)提供原始完整五视角录像、三张实际末段图像和校验清单。历史报告中的`/home/noob/...`是原运行证据路径，不是新机器启动参数。

本分支的移植改动仅涉及输入恢复、运行/审查入口和说明；控制源码、物理参数、两个高位相机输入的字节均保留。机器人资源从当前目录重建，避免复用旧工程的`install/`。仅仿真，`hardware_authorized=false`。
