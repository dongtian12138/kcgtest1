#!/usr/bin/env bash
# Independent holding test: the external assembly tool stays disabled.
set -euo pipefail
te_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
te_repository="$(cd -- "${te_script_dir}/../../.." && pwd)"
te_run_output="${1:-${te_repository}/artifacts/kcg_connector/retention_checks/run_$(date -u +%Y%m%d_%H%M%S)}"
cd -- "${te_repository}"
exec timeout --signal=TERM --kill-after=10s 260s \
  "${te_script_dir}/run_isaac_python.sh" \
  "${te_script_dir}/te_complete_mating_probe.py" \
  --model artifacts/kcg_connector/te_connector_contact_repaired_20260911/connector_model.usdc \
  --initial-pose artifacts/kcg_connector/te_connector_contact_repaired_20260911/observed_mated_pose.json \
  --tool-axis X --torque-cap-nm 3 --position-iterations 64 --velocity-iterations 4 --solver TGS \
  --retention --retention-set all --contact-report-hz 5 --wall-limit-s 240 \
  --output "${te_run_output}"
