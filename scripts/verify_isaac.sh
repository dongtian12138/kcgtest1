#!/usr/bin/env bash
# ============================================================
# kcgtest1 验收门 B：GPU/Isaac（需要 NVIDIA GPU 与 Isaac 环境）
# 用法: bash scripts/verify_isaac.sh
# 前置: bootstrap.sh 已创建 Isaac 环境，或手动设置 ISAAC_ENV_PREFIX
# 输出: 每阶段一行 [PASS]/[FAIL]，日志在 log_verify_isaac/
# ============================================================
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${WS}/log_verify_isaac"
mkdir -p "${LOGDIR}"
cd "${WS}"

ISAAC_PREFIX="${ISAAC_ENV_PREFIX:-${HOME}/WorkPlace/isaacsim/.conda-env}"
if [ ! -x "${ISAAC_PREFIX}/bin/python" ]; then
  echo "[FAIL] 未找到 Isaac 环境: ${ISAAC_PREFIX}"
  echo "       先运行 bash scripts/bootstrap.sh（GPU 机器），或 export ISAAC_ENV_PREFIX=<isaac 环境前缀>"
  exit 2
fi
export ISAAC_ENV_PREFIX="${ISAAC_PREFIX}"
RUNNER="src/kcg_connector/isaac/run_isaac_python.sh"

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

echo "===== kcgtest1 verify_isaac 开始（Isaac 环境: ${ISAAC_PREFIX}）====="

# 1) q7 单行程 20°（已验收最短课程）
run_stage stage20 "ISAAC CONNECTOR ZERO-RESIDUAL 2-EPISODE PASSED" 1 1800 \
  "${RUNNER}" src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-zero --residual-stage stage20 --episodes 2

# 2) q7 单行程 120°（已验收最长单行程）
run_stage stage120 "ISAAC CONNECTOR ZERO-RESIDUAL 2-EPISODE PASSED" 1 1800 \
  "${RUNNER}" src/kcg_connector/isaac/connector_q7_twist_smoke.py \
  --mode residual-zero --residual-stage stage120 --episodes 2

# 3) 固定全局虚拟 RGB-D 预检
run_stage rgbd "ISAAC D38999 RGBD BOOTSTRAP V1 PASSED" 1 1500 \
  "${RUNNER}" src/kcg_connector/isaac/d38999_rgbd_bootstrap_smoke.py

# 4) D38999 clean headless 完整链（smooth demo + masked-rgbd 预检）
run_stage e2e "ISAAC D38999 END TO END V1 PASSED" 1 3600 \
  "${RUNNER}" src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py \
  --end-to-end-probe --smooth-demo --pose-preflight masked-rgbd

# 5) 冻结工程基线独立回放（918 passed，自动校验 Isaac/Python/CUDA 环境）
run_stage replay "918 passed" 1 3600 \
  bash artifacts/kcg_connector/d38999_end_to_end_v1/baseline_20260812T111941Z/replay.sh

echo "===== verify_isaac 汇总: PASS=${PASS} FAIL=${FAIL}====="
if [ "${FAIL}" -ne 0 ]; then
  echo "失败阶段:${FAILED_STAGES}"
  exit 1
fi
echo "VERIFY_ISAAC_OK"
exit 0
