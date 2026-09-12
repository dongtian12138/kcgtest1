#!/usr/bin/env bash
# Continuous, contact-driven mating from the shallow keyed state.
# The visible external spindle is a test apparatus, not a connector part.
set -euo pipefail
te_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
te_repository="$(cd -- "${te_script_dir}/../../.." && pwd)"
te_run_output="${1:-${te_repository}/artifacts/kcg_connector/complete_mating_checks/run_$(date -u +%Y%m%d_%H%M%S)}"
cd -- "${te_repository}"
exec timeout --signal=TERM --kill-after=10s 380s \
  "${te_script_dir}/run_isaac_python.sh" \
  "${te_script_dir}/te_complete_mating_probe.py" \
  --model artifacts/kcg_connector/te_connector_contact_repaired_20260911/connector_model.usdc \
  --initial-pose artifacts/kcg_connector/te_connector_contact_repaired_20260911/shallow_initial_pose.json \
  --tool-axis X --torque-cap-nm 3 --position-iterations 64 --velocity-iterations 4 \
  --solver TGS --couple-duration-s 7 --rotary-tool-inertia-from-hand --torque-transducer \
  --spec-speed-profile --spindle-encoder-drive --engagement-axial-assist --encoder-forward-completion \
  --contact-report-hz 5 --wall-limit-s 360 --output "${te_run_output}"
