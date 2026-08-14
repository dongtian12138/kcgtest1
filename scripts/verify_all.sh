#!/usr/bin/env bash
# ============================================================
# kcgtest1 验收门 A：CPU/ROS（无 GPU 可跑，headless）
# 用法: bash scripts/verify_all.sh
# 输出: 每阶段一行 [PASS]/[FAIL]，日志在 log_verify/
# 全部 PASS 时退出码 0，否则 1
# ============================================================
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${WS}/log_verify"
mkdir -p "${LOGDIR}"
cd "${WS}"
source /opt/ros/humble/setup.bash 2>/dev/null || true
source install/setup.bash 2>/dev/null || true

PASS=0; FAIL=0
FAILED_STAGES=""

run_stage() {
  local name="$1" marker="$2" count="${3:-1}" timeout_s="$4"; shift 4
  local logf="${LOGDIR}/${name}.log"
  echo "=== STAGE ${name} 开始 ($(date -u +%FT%TZ)) ===" | tee "${logf}"
  local rc=0
  timeout "${timeout_s}" "$@" >>"${logf}" 2>&1 || rc=$?
  local n; n="$(grep -Fc -- "${marker}" "${logf}" || true)"
  if [ "${rc}" -eq 0 ] && [ "${n}" -ge "${count}" ]; then
    echo "[PASS] ${name}（marker 出现 ${n} 次）"
    PASS=$((PASS+1))
  else
    echo "[FAIL] ${name}（退出码 ${rc}，marker 出现 ${n} 次，期望 >= ${count}；日志: ${logf}）"
    FAIL=$((FAIL+1)); FAILED_STAGES="${FAILED_STAGES} ${name}"
  fi
}

echo "===== kcgtest1 verify_all 开始（工作区: ${WS}）====="

# 1) 构建门
run_stage build "packages finished" 1 1800 \
  bash -c "colcon build --symlink-install"

# 2) 30 秒无控制物理稳定性
run_stage physics "Physics stability check PASSED" 1 900 \
  ros2 launch kcg_moveit1 physics_sanity.launch.py duration:=30.0

# 3) 圆柱抓取脚本基线（自动结束）
run_stage grasp "SCRIPTED GRASP PASSED" 1 1200 \
  ros2 launch kcg_grasping cylinder_grasp.launch.py \
  gui:=false use_rviz:=false run_baseline:=true shutdown_on_completion:=true

# 4) RL reset/step 冒烟（两回合，自动结束）
run_stage rl "RL STEP SMOKE PASSED" 2 1200 \
  ros2 launch kcg_rl cylinder_rl.launch.py \
  gui:=false use_rviz:=false run_smoke:=true smoke_episodes:=2 shutdown_on_completion:=true

echo "===== verify_all 汇总: PASS=${PASS} FAIL=${FAIL}====="
if [ "${FAIL}" -ne 0 ]; then
  echo "失败阶段:${FAILED_STAGES}"
  exit 1
fi
echo "VERIFY_ALL_OK"
exit 0
