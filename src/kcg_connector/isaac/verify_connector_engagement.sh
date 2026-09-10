#!/usr/bin/env bash
# First short turn, passive side support and low-torque nut oscillation.
# This entry does not run a full assembly or request hardware motion.
set -euo pipefail
connector_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
connector_repository="$(cd -- "${connector_script_dir}/../../.." && pwd)"
connector_run_output="${1:-${connector_repository}/artifacts/kcg_connector/engagement_checks/run_$(date -u +%Y%m%d_%H%M%S)}"
cd -- "${connector_repository}"
exec timeout --signal=TERM --kill-after=2 175 \
  "${connector_script_dir}/run_isaac_python.sh" \
  "${connector_script_dir}/te_engagement_wrench_probe.py" \
  --model artifacts/kcg_connector/first_turn_support_20260910/connector_engagement_repaired/connector_model.usdc \
  --install-into-source artifacts/kcg_connector/model_delivery_20260908/package/source_scene.usda \
  --initial-pose artifacts/kcg_connector/first_turn_support_20260910/first_thread_start_pose.json \
  --first-turn-deg 40 --output "${connector_run_output}"
