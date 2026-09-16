#!/usr/bin/env bash

# Launch one Isaac Sim Python script with this workstation's isolated runtime.
# Keep the library path process-local: exporting it globally can disturb ROS 2
# and Gazebo, which intentionally use Ubuntu's system C++ runtime.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/../../.." && pwd)"
workspace_root="$(dirname -- "${repository_root}")"

if [[ -n "${ISAAC_ENV_PREFIX:-}" ]]; then
  isaac_env_prefix="${ISAAC_ENV_PREFIX}"
else
  isaac_env_prefix="${workspace_root}/isaacsim/.conda-env"
fi
isaac_python="${isaac_env_prefix}/bin/python"
export ISAAC_ENV_PREFIX="${isaac_env_prefix}"
# An environment may have an editable install pointing at another checkout.
# This invocation must import the source belonging to its selected repository.
export PYTHONPATH="${repository_root}/src/kcg_connector${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${isaac_python}" ]]; then
  echo "Isaac Sim Python was not found: ${isaac_python}" >&2
  echo "Set ISAAC_ENV_PREFIX to the Isaac Sim environment prefix." >&2
  echo "The portable fallback checked: ${isaac_env_prefix}" >&2
  exit 2
fi

# An unattended driver update can replace userspace libraries while its old
# kernel module remains loaded. Optional, version-scoped private libraries
# restore this process without replacing a system driver or rebooting. A new
# kernel version naturally stops selecting the old private directory.
nvidia_kernel_version="$(cat /sys/module/nvidia/version 2>/dev/null || true)"
isaac_nvidia_libraries="${workspace_root}/isaacsim/.nvidia-userspace/${nvidia_kernel_version}"
if [[ -n "${nvidia_kernel_version}" && -f "${isaac_nvidia_libraries}/libcuda.so.1" && -f "${isaac_nvidia_libraries}/libnvidia-ml.so.1" ]]; then
  export LD_LIBRARY_PATH="${isaac_env_prefix}/lib:${isaac_nvidia_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  export LD_LIBRARY_PATH="${isaac_env_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

case "${1:-}" in
  */run_body_assembly_with_video.py|*/diagnose_saved_hand_wrench.py|*/carts_v2/run_grasp_lift.py)
    if [[ "${KCG_BOUNDED_EXPERIMENT:-0}" != "1" ]]; then
      exec python3 "${script_dir}/bounded_experiment.py" "${isaac_python}" "$@"
    fi
    ;;
esac
exec "${isaac_python}" "$@"
