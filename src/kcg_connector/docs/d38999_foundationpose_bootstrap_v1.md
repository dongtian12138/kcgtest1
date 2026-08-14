# D38999 FoundationPose bootstrap v1

This is an isolated, disabled preparation path.  It does not modify the active
USD, import the current E2E runner, install host packages, build TensorRT
engines, or claim that FoundationPose has run.

## Prepared now

- NVIDIA NGC `1.0.1_onnx` refine and score models are stored only in the
  gitignored local artifact directory and match NGC's exact sizes and hashes.
- Three deterministic metre-scale OBJ files map separately to the loose body,
  independently rotating coupling nut, and fixed receptacle prims.
- `SOURCE.md`, the model manifest, mesh hashes, runtime probes, and a strict
  readiness report make the boundary reproducible.
- The readiness gate remains blocked for inference and control.

Run the artifact-only check (expected exit code `0` on this workstation):

```bash
PYTHONPATH=src/kcg_connector python3 \
  src/kcg_connector/isaac/d38999_foundationpose_readiness.py \
  --repository "$PWD" \
  --require artifacts
```

Run the stricter runtime check (currently expected exit code `2`):

```bash
PYTHONPATH=src/kcg_connector python3 \
  src/kcg_connector/isaac/d38999_foundationpose_readiness.py \
  --repository "$PWD" \
  --require runtime
```

## Current blockers

The workstation has a 16 GB RTX 5070 Ti and driver 595.84, which passes the
current Isaac ROS GPU, memory, and driver floors.  However, it has Ubuntu 22.04
and no Docker/Podman, TensorRT `trtexec`, or Isaac ROS FoundationPose runtime.
Current Isaac ROS release-4.5 officially supports x86 on Ubuntu 24.04 with ROS
Jazzy, so the existing host is outside that support matrix.

More importantly, the simulation proxy contains no measured unique
polarization key.  Its full appearance has at least a two-fold yaw symmetry;
the coupling nut alone has 24-fold symmetry.  FoundationPose can estimate a
pose hypothesis modulo those symmetries, but it cannot infer physical key yaw
that is absent from the CAD and observations.  Such an output must not
authorize connector assembly control.

## Next isolated execution step

This step needs explicit authority to provide a supported Ubuntu 24.04 Isaac
ROS release-4.5 environment and container runtime.  Once that environment
exists:

1. Mount the gitignored artifact bundle into `ISAAC_ROS_WS`; do not copy the
   NVIDIA weights into the repository or an image layer.
2. Activate the official environment and install/build
   `isaac_ros_foundationpose` inside it.
3. Build both FP32 plans using the release-4.5 quickstart shapes:

```bash
artifact_root="$ISAAC_ROS_WS/kcgtest1/artifacts/kcg_connector/foundationpose_1.0.1_onnx_local_v1"
mkdir -p "$artifact_root/engines"

/usr/src/tensorrt/bin/trtexec \
  --onnx="$artifact_root/models/refine_model.onnx" \
  --saveEngine="$artifact_root/engines/refine_trt_engine.plan" \
  --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --maxShapes=input1:42x160x160x6,input2:42x160x160x6

/usr/src/tensorrt/bin/trtexec \
  --onnx="$artifact_root/models/score_model.onnx" \
  --saveEngine="$artifact_root/engines/score_trt_engine.plan" \
  --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --maxShapes=input1:252x160x160x6,input2:252x160x160x6
```

4. First pass NVIDIA's official rosbag quickstart.  Only then connect the
   simulator's registered RGB, depth, camera info, instance mask, and one of
   the hash-bound OBJ meshes.
5. Score translation and axis error against withheld simulation truth while
   treating symmetry-equivalent yaw hypotheses as equivalent.  Add measured
   key geometry and a view that observes it before testing keyed yaw.

Reasonable planning estimates are 1–3 hours to create the supported isolated
environment, 10–30 minutes to build the two engine plans, and another 1–3
hours to obtain and validate the first simulation inference after the
environment exists.  These are estimates, not completed work.

Official references:

- <https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/>
- <https://nvidia-isaac-ros.github.io/getting_started/index.html>
- <https://catalog.ngc.nvidia.com/orgs/nvidia/isaac/models/foundationpose>
- <https://developer.download.nvidia.com/licenses/tao_toolkit_21-08_models_eula.pdf>
