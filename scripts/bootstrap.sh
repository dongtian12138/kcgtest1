#!/usr/bin/env bash
# ============================================================
# kcgtest1 一键环境引导脚本（在新电脑上运行一次即可）
# 用法: bash scripts/bootstrap.sh [--skip-apt] [--skip-venv] [--skip-isaac] [--skip-build]
# 日志: bootstrap.log；各阶段幂等，可反复运行
# ============================================================
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${WS}/bootstrap.log"
: > "${LOG}"
log()  { echo "[bootstrap] $*" | tee -a "${LOG}"; }
fail() { echo "[bootstrap] 失败: $*" | tee -a "${LOG}"; exit 1; }

SKIP_APT=0; SKIP_VENV=0; SKIP_ISAAC=0; SKIP_BUILD=0
for a in "$@"; do
  case "$a" in
    --skip-apt) SKIP_APT=1;;   --skip-venv) SKIP_VENV=1;;
    --skip-isaac) SKIP_ISAAC=1;; --skip-build) SKIP_BUILD=1;;
    *) fail "未知参数: $a";;
  esac
done
log "工作区: ${WS}"

# ---------- 阶段 0: 基线检查 ----------
if [ -f /opt/ros/humble/setup.bash ]; then
  HAS_ROS=1; log "检测到 ROS 2 Humble"
else
  HAS_ROS=0; log "未检测到 /opt/ros/humble"
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1; log "检测到 NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  HAS_GPU=0; log "未检测到可用 NVIDIA GPU（ROS 门可跑；Isaac 门需要 GPU）"
fi

# ---------- 阶段 1: 系统依赖（apt，需要 sudo） ----------
if [ "${SKIP_APT}" = "1" ]; then
  log "跳过 apt 阶段（--skip-apt）"
else
  if [ "${HAS_ROS}" = "0" ]; then
    fail "ROS 2 Humble 未安装。先按官方文档安装 ros-humble-desktop：
https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
（只做纯 Python 静态检查时可用 --skip-apt 跳过本阶段）"
  fi
  log "安装 ROS2/MoveIt/Gazebo 相关 apt 包 ..."
  sudo apt-get update
  sudo apt-get install -y \
    ros-humble-moveit ros-humble-ros2-control ros-humble-ros2-controllers \
    ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control \
    ros-humble-joint-state-publisher ros-humble-xacro \
    python3-colcon-common-extensions python3-rosdep python3-pip python3-venv
  sudo rosdep init || true
  rosdep update
  log "apt 阶段完成"
fi

# ---------- 阶段 2: 依赖解析 + colcon 构建 ----------
if [ "${SKIP_BUILD}" = "1" ]; then
  log "跳过构建阶段（--skip-build）"
else
  source /opt/ros/humble/setup.bash
  cd "${WS}"
  log "rosdep install ..."
  rosdep install --from-paths src --ignore-src -r -y
  log "colcon build --symlink-install ..."
  colcon build --symlink-install
  log "构建完成"
fi

# ---------- 阶段 3: 项目 .venv（当前静态源码依赖） ----------
if [ "${SKIP_VENV}" = "1" ]; then
  log "跳过 .venv 阶段（--skip-venv）"
elif [ -x "${WS}/.venv/bin/python" ]; then
  log ".venv 已存在，跳过"
else
  cd "${WS}"
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -e src/kcg_connector pytest
  log ".venv 完成（kcg_connector 静态依赖与 pytest）"
fi

# ---------- 阶段 4: Isaac Sim 环境（GPU 机器才需要） ----------
if [ "${SKIP_ISAAC}" = "1" ] || [ "${HAS_GPU}" = "0" ]; then
  log "跳过 Isaac 环境阶段"
else
  ISAAC_PREFIX="${ISAAC_ENV_PREFIX:-${HOME}/WorkPlace/isaacsim/.conda-env}"
  if [ -x "${ISAAC_PREFIX}/bin/python" ]; then
    log "Isaac 环境已存在: ${ISAAC_PREFIX}"
  else
    CREATED=0
    if command -v conda >/dev/null 2>&1; then
      log "用 conda 创建 Isaac 环境（python 3.12）..."
      conda create -y -p "${ISAAC_PREFIX}" python=3.12 && CREATED=1
    elif command -v python3.12 >/dev/null 2>&1; then
      log "用系统 python3.12 创建 Isaac venv ..."
      mkdir -p "$(dirname "${ISAAC_PREFIX}")"
      python3.12 -m venv "${ISAAC_PREFIX}" && CREATED=1
    fi
    if [ "${CREATED}" = "1" ]; then
      "${ISAAC_PREFIX}/bin/pip" install -r src/kcg_connector/requirements-torch-cu128.txt
      "${ISAAC_PREFIX}/bin/pip" install -r src/kcg_connector/requirements-isaacsim.txt
      log "Isaac Sim 6.0.1 安装完成（首次运行会下载大量 assets，属正常）"
    else
      log "本机没有 conda 也没有 python3.12，Isaac 环境未创建。手动步骤："
      log "  1) 安装 Miniforge: https://github.com/conda-forge/miniforge"
      log "  2) conda create -y -p ${ISAAC_PREFIX} python=3.12"
      log "  3) ${ISAAC_PREFIX}/bin/pip install -r src/kcg_connector/requirements-torch-cu128.txt"
      log "  4) ${ISAAC_PREFIX}/bin/pip install -r src/kcg_connector/requirements-isaacsim.txt"
    fi
  fi
  cat > "${WS}/scripts/isaac_env.sh" <<'INNER'
# 用法: source scripts/isaac_env.sh（为 Isaac 冒烟设置环境）
export ISAAC_ENV_PREFIX="${ISAAC_ENV_PREFIX:-${HOME}/WorkPlace/isaacsim/.conda-env}"
INNER
  log "Isaac 环境前缀: ${ISAAC_PREFIX}（可用 ISAAC_ENV_PREFIX 覆盖，或 source scripts/isaac_env.sh）"
fi

log "BOOTSTRAP 全部完成。下一步先读 CURRENT_CONTEXT_CN.md。"
log "纯 CPU 基础检查：PYTHONPATH=src/kcg_connector python3 -m pytest -q src/kcg_connector/test/test_geometry.py"
log "不要从 bootstrap 直接启动 Isaac；动态动作必须通过当前控制面授权。"
