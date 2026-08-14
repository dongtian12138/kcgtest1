#!/usr/bin/env bash

# Launch one Isaac Sim Python script with this workstation's isolated runtime.
# Keep the library path process-local: exporting it globally can disturb ROS 2
# and Gazebo, which intentionally use Ubuntu's system C++ runtime.
set -euo pipefail

isaac_env_prefix="${ISAAC_ENV_PREFIX:-/home/noob/WorkPlace/isaacsim/.conda-env}"
isaac_python="${isaac_env_prefix}/bin/python"

if [[ ! -x "${isaac_python}" ]]; then
  echo "Isaac Sim Python was not found: ${isaac_python}" >&2
  echo "Set ISAAC_ENV_PREFIX to the Isaac Sim environment prefix." >&2
  exit 2
fi

export LD_LIBRARY_PATH="${isaac_env_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

exec "${isaac_python}" "$@"
