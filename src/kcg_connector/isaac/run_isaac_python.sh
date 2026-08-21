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

if [[ ! -x "${isaac_python}" ]]; then
  echo "Isaac Sim Python was not found: ${isaac_python}" >&2
  echo "Set ISAAC_ENV_PREFIX to the Isaac Sim environment prefix." >&2
  echo "The portable fallback checked: ${isaac_env_prefix}" >&2
  exit 2
fi

export LD_LIBRARY_PATH="${isaac_env_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

exec "${isaac_python}" "$@"
