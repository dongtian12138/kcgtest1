"""Opt-in post-grasp shadow capture hook for the tabletop pick smoke.

Formal capture uses only the raw RGB+depth runtime path.  The hand-eye
transform is frozen from the configured camera mount before any view motion;
camera world poses are computed as ``T_WC = T_WH(actual_joint_FK) @ T_HC`` and
never back-derived from a USD camera prim world transform.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from kcg_connector.d38999_cad_registration import fixed_camera_model
from kcg_connector.d38999_key_branch_selector import (
    SUPPORTED_KEYED_PLUG_MODEL_IDS,
    blocked_key_branch_selection,
)
from kcg_connector.d38999_key_yaw_acceptance import (
    THRESHOLD_LABEL as KEY_YAW_THRESHOLD_LABEL,
)
from kcg_connector.grasp.grasp_stability_monitor import (
    evaluate_wrist_moment_safety,
)
from kcg_connector.postgrasp_shadow_estimator import (
    FormalView,
    estimate_postgrasp_T_HP,
    write_formal_archive,
)
from kcg_connector.postgrasp_snapshot_truth import (
    capture_truth_snapshot,
    probe_restore_api,
    write_truth_snapshot,
)
from kcg_connector.postgrasp_shadow_view_planner import (
    DEFAULT_POSTGRASP_PLANS,
    score_formal_view,
)


# The current key detector stage is shadow-only by contract.  This must stay
# false until the real keyed geometry, measured yaw clearance and withheld-set
# p95 gate have all been independently accepted.
KEYED_INSERTION_CONTROL_PROMOTION_ENABLED = False


def _require_frozen_camera_optics(stage, camera_path, UsdGeom):
    """Read back the canonical camera without authoring any stage opinion."""
    prim = stage.GetPrimAtPath(camera_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"canonical camera prim is unavailable: {camera_path}")
    if prim.GetTypeName() != "Camera":
        raise RuntimeError(f"canonical camera prim has wrong type: {camera_path}")
    expected = {
        "focalLength": (24.0,),
        "horizontalAperture": (20.955,),
        "verticalAperture": (11.7871875,),
        "clippingRange": (0.02, 10.0),
    }
    for name, wanted in expected.items():
        attribute = prim.GetAttribute(name)
        if attribute is None or not attribute.IsValid():
            raise RuntimeError(
                f"canonical camera attribute is unavailable: {camera_path}.{name}"
            )
        actual = np.asarray(attribute.Get(), dtype=np.float64).ravel()
        if actual.shape != (len(wanted),) or not np.allclose(
            actual, wanted, rtol=0.0, atol=1.0e-9
        ):
            raise RuntimeError(
                f"canonical camera attribute changed: {camera_path}.{name}"
            )
    return UsdGeom.Camera(prim)


def _current_proxy_key_branch_block() -> dict[str, Any]:
    return blocked_key_branch_selection(
        "KEYED_GEOMETRY_UNAVAILABLE",
        "current Shell25J C2 proxy has no traceable unique key geometry",
        current_model_id="d38999_shell25j_proxy_v1",
    )


def evaluate_view_motion_safety(
    *,
    positions,
    velocities,
    arm_target,
    arm_indices,
    measured_efforts,
    tare_efforts,
    wrist_canonical,
    wrist_payload_reference,
    wrist_ft_monitor_error,
    tracking_limit_rad,
    torque_limit_nm,
    force_limit_n=8.0,
    moment_limit_nm=0.30,
) -> str | None:
    """Return a failure reason or None.  Frozen 8 N / 0.30 N*m gates are never
    increased; moment uses the existing three-component frozen evaluator."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    arm_target = np.asarray(arm_target, dtype=np.float64)
    indices = np.asarray(arm_indices, dtype=np.int64)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        return "NON_FINITE_ROBOT_STATE"
    if not np.all(np.isfinite(arm_target)):
        return "NON_FINITE_ARM_TARGET"
    if float(np.max(np.abs(positions[indices] - arm_target))) > float(
        tracking_limit_rad
    ):
        return "ARM_TRACKING_GATE"
    root_delta = np.asarray(measured_efforts, dtype=np.float64) - np.asarray(
        tare_efforts, dtype=np.float64
    )
    if not np.all(np.isfinite(root_delta)):
        return "NON_FINITE_FINGER_TORQUE"
    if float(np.max(np.abs(root_delta))) > float(torque_limit_nm):
        return "FINGER_TORQUE_GATE"
    if wrist_ft_monitor_error is not None:
        return "WRIST_FT_STALE_OR_FAILED"
    wrist = np.asarray(wrist_canonical, dtype=np.float64).ravel()
    reference = np.asarray(wrist_payload_reference, dtype=np.float64).ravel()
    if wrist.shape != (6,) or reference.shape != (6,):
        return "NON_FINITE_WRIST_WRENCH"
    wrist_delta = wrist - reference
    if float(np.linalg.norm(wrist_delta[:3])) > float(force_limit_n):
        return "WRIST_FORCE_GATE"
    moment_evidence = evaluate_wrist_moment_safety(
        tuple(float(v) for v in wrist[3:]),
        tuple(float(v) for v in reference[3:]),
        float(moment_limit_nm),
    )
    if moment_evidence["triggered"] is True:
        return "WRIST_MOMENT_GATE"
    return None


class ShadowViewMotionAbort(RuntimeError):
    """Raised by the guarded view motion before any formal capture."""


def _method_or_none(obj, name):
    value = getattr(obj, name, None)
    return value if callable(value) else None


def _as_vector3(value) -> list[float]:
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("expected finite 3-vector")
    return array.tolist()


def _as_quaternion(value) -> list[float]:
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise ValueError("expected finite 4-vector quaternion")
    norm = float(np.linalg.norm(array))
    if abs(norm - 1.0) > 1.0e-6:
        raise ValueError("quaternion must be unit norm")
    return array.tolist()


def _camera_model_from_eye_target(eye, target, resolution):
    return fixed_camera_model(
        eye=tuple(float(v) for v in eye),
        target=tuple(float(v) for v in target),
        resolution=tuple(int(v) for v in resolution),
    )


def _t_wc_from_camera_model(camera) -> np.ndarray:
    rows = np.asarray(camera.world_to_camera, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rows.T
    transform[:3, 3] = np.asarray(camera.position_world, dtype=np.float64)
    return transform


def _camera_cv_pose_from_eye_target(
    eye, target, *, down_hint=(0.0, 1.0, 0.0)
) -> np.ndarray:
    """Build a finite CV optical pose with an explicit image-down hint."""

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    down_hint = np.asarray(down_hint, dtype=np.float64)
    if eye.shape != (3,) or target.shape != (3,) or down_hint.shape != (3,):
        raise ValueError("camera eye, target and down hint must be 3-vectors")
    if not all(
        np.all(np.isfinite(value)) for value in (eye, target, down_hint)
    ):
        raise ValueError("camera eye, target and down hint must be finite")
    forward = target - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1.0e-9:
        raise ValueError("camera eye and target must be distinct")
    forward /= forward_norm
    right = np.cross(down_hint, forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-9:
        raise ValueError("camera down hint must not parallel the optical axis")
    right /= right_norm
    down = np.cross(forward, right)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((right, down, forward))
    transform[:3, 3] = eye
    return transform


def _calibrated_hand_camera_from_nominal_plug(
    nominal_hand_to_plug,
    eye_plug,
    target_plug,
    resolution,
) -> np.ndarray:
    """Freeze camera->hand from the task/CAD nominal grasp transform.

    The configured eye and target are authored in the Plug mating frame.  The
    episode-specific Plug pose is deliberately not accepted by this API.
    """
    nominal_hp = np.asarray(nominal_hand_to_plug, dtype=np.float64)
    if nominal_hp.shape != (4, 4) or not np.all(np.isfinite(nominal_hp)):
        raise ValueError("nominal_hand_to_plug must be a finite 4x4 transform")
    camera_in_plug = _camera_model_from_eye_target(
        eye_plug, target_plug, resolution
    )
    return nominal_hp @ _t_wc_from_camera_model(camera_in_plug)


def _camera_target_from_t_wc(t_wc: np.ndarray) -> np.ndarray:
    rotation = np.asarray(t_wc, dtype=np.float64)[:3, :3]
    forward = rotation @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    return np.asarray(t_wc[:3, 3], dtype=np.float64) + forward


def _camera_cv_pose_to_usd(t_wc: np.ndarray) -> np.ndarray:
    """Convert a +Z-forward CV camera pose to the USD camera convention.

    ``fixed_camera_model`` uses x-right, y-down and z-forward. A USD camera
    looks down local -Z with local +Y up, so both the y and z basis vectors
    must be negated. Translation is unchanged.
    """

    cv_pose = np.asarray(t_wc, dtype=np.float64)
    if cv_pose.shape != (4, 4) or not np.all(np.isfinite(cv_pose)):
        raise ValueError("T_WC must be a finite 4x4 transform")
    usd_pose = cv_pose.copy()
    usd_pose[:3, :3] = cv_pose[:3, :3] @ np.diag((1.0, -1.0, -1.0))
    return usd_pose


def selected_t_hp_control_pose(result: Mapping[str, Any]):
    """Return the explicitly selected finite 6D pose, or fail closed.

    Capture/optimizer completion is deliberately insufficient. Runtime
    control requires calibrated covariance, an observed C2 resolution, a
    validated real keyed model and an explicit selected hypothesis in
    addition to both authorization booleans.
    """

    if not KEYED_INSERTION_CONTROL_PROMOTION_ENABLED:
        return None
    if not isinstance(result, Mapping):
        return None
    if result.get("capture_status") != "GPU_CAPTURE_VALID":
        return None
    if result.get("pose_valid") is not True:
        return None
    if result.get("control_authorized") is not True:
        return None
    if result.get("covariance_calibration_status") != "CALIBRATED":
        return None
    if result.get("c2_resolution") != "C2_RESOLVED_BY_OBSERVATION":
        return None
    if result.get("real_keying_modeled") is not True:
        return None
    keying_model_id = result.get("keying_model_id")
    if keying_model_id not in SUPPORTED_KEYED_PLUG_MODEL_IDS:
        return None
    selected_id = result.get("selected_c2_hypothesis_id")
    if not isinstance(selected_id, str) or not selected_id:
        return None
    key_branch = result.get("key_branch_selection")
    if not isinstance(key_branch, Mapping):
        return None
    if key_branch.get("status") != "SHADOW_BRANCH_SELECTED":
        return None
    if key_branch.get("passed") is not True:
        return None
    if key_branch.get("shadow_selected_hypothesis_id") != selected_id:
        return None
    if key_branch.get("control_authorized") is not False:
        return None
    if "selected_for_control" in key_branch:
        return None
    yaw_gate = result.get("key_yaw_acceptance")
    if not isinstance(yaw_gate, Mapping):
        return None
    if yaw_gate.get("status") != "PASSED_EVALUATION_ONLY":
        return None
    if yaw_gate.get("passed") is not True or yaw_gate.get("withheld_truth") is not True:
        return None
    if yaw_gate.get("threshold_label") != KEY_YAW_THRESHOLD_LABEL:
        return None
    if yaw_gate.get("control_authorized") is not False:
        return None
    try:
        observed_p95 = float(yaw_gate["observed_yaw_error_p95_deg"])
        required_p95 = float(yaw_gate["required_yaw_error_p95_deg"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (
        math.isfinite(observed_p95)
        and math.isfinite(required_p95)
        and observed_p95 < required_p95
    ):
        return None
    hypotheses = result.get("c2_hypotheses")
    if not isinstance(hypotheses, list):
        return None
    matches = [
        item
        for item in hypotheses
        if isinstance(item, Mapping) and item.get("id") == selected_id
    ]
    if len(matches) != 1:
        return None
    pose = np.asarray(matches[0].get("T_hand_plug_xyz_rpy"), dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        return None
    return pose.copy()


def _iiwa_tcp_fk(arm_q):
    from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform

    return np.asarray(
        iiwa14_grasp_tcp_transform(tuple(float(value) for value in arm_q)),
        dtype=np.float64,
    )


def _solve_arm(seed_arm, target_position, target_rotation):
    from kcg_connector.d38999_physical_insertion import solve_fixed_q7_tcp_pose

    try:
        solved_values = solve_fixed_q7_tcp_pose(
            tuple(float(value) for value in seed_arm),
            tuple(float(value) for value in target_position),
            target_rotation=target_rotation,
            maximum_iterations=120,
            damping=1.0e-3,
        )
    except Exception as exception:
        raise ShadowViewMotionAbort(
            f"view IK target unreachable: {type(exception).__name__}"
        ) from exception
    solved = np.asarray(solved_values, dtype=np.float64)
    if not np.all(np.isfinite(solved)) or solved.shape != (7,):
        raise ShadowViewMotionAbort("view IK target is non-finite or malformed")
    return solved


def validate_planned_view_motion(
    start_arm, target_arm, h0_arm, motion_config, *, phase_label
) -> dict[str, Any]:
    """Validate one planned observation motion against the conservative budget.

    The original per-plan ``0.05 rad`` and cumulative ``max|q-H0| <= 0.20 rad``
    contracts are restored.  Per-command interpolation is checked separately
    during execution.
    """
    start = np.asarray(start_arm, dtype=np.float64)
    target = np.asarray(target_arm, dtype=np.float64)
    h0 = np.asarray(h0_arm, dtype=np.float64)
    if (
        start.shape != target.shape
        or target.shape != h0.shape
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(h0))
    ):
        raise ShadowViewMotionAbort("non-finite planned view motion")
    plan_inf = float(np.max(np.abs(target - start)))
    cumulative_inf = float(np.max(np.abs(target - h0)))
    limits = {
        "per_command_max_joint_delta_rad": float(
            motion_config["per_command_max_joint_delta_rad"]
        ),
        "planned_max_joint_inf_rad": float(
            motion_config["planned_max_joint_inf_rad"]
        ),
        "episode_max_joint_inf_rad": float(
            motion_config["episode_max_joint_inf_rad"]
        ),
    }
    if plan_inf > limits["planned_max_joint_inf_rad"]:
        raise ShadowViewMotionAbort("view planned joint delta exceeded 0.05 rad")
    if cumulative_inf > limits["episode_max_joint_inf_rad"]:
        raise ShadowViewMotionAbort("view cumulative |q-H0| budget exceeded 0.20 rad")
    return {
        "phase": phase_label,
        "planned_max_abs_dq_rad": plan_inf,
        "planned_cumulative_max_abs_from_h0_rad": cumulative_inf,
        "limits": limits,
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }


def _read_latest_wrist_state(get_latest_wrist_state, previous_step):
    state = get_latest_wrist_state()
    if not isinstance(state, Mapping):
        raise ShadowViewMotionAbort("WRIST_FT_STALE_OR_FAILED")
    step = state.get("global_step")
    error = state.get("error")
    canonical = np.asarray(state.get("canonical"), dtype=np.float64).ravel()
    if error is not None:
        raise ShadowViewMotionAbort("WRIST_FT_STALE_OR_FAILED")
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(float(step))
        or int(step) <= int(previous_step)
    ):
        raise ShadowViewMotionAbort("WRIST_FT_STALE_OR_FAILED")
    if canonical.shape != (6,) or not np.all(np.isfinite(canonical)):
        raise ShadowViewMotionAbort("WRIST_FT_STALE_OR_FAILED")
    return int(step), canonical, None


def run_postgrasp_shadow_capture(
    *,
    repository: Path,
    arguments,
    Gf,
    Usd,
    UsdGeom,
    UsdLux,
    stage,
    world,
    simulation_app,
    tcp_prim,
    tabletop,
    body,
    robot,
    arm_indices,
    current_arm_target,
    current_hand_target,
    observe_and_step,
    sample_post_tare_efforts,
    rate_hz,
    global_step,
    pick,
    physical_grasp,
    tare_efforts,
    formal_wrist_payload_reference,
    get_latest_wrist_state,
    nominal_hand_to_plug,
    nut_joint_state_provider=None,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output_root = (
        Path(arguments.output_dir).expanduser().resolve() / "postgrasp_shadow"
    )
    result: dict[str, Any] = {
        "requested": True,
        "passed_frozen_before_shadow": True,
        "shadow_authorized": False,
        "control_authorized": False,
        "key_branch_selection": _current_proxy_key_branch_block(),
        "snapshot_restore_verified": False,
        "status": "STARTED",
    }
    if output_root.exists():
        result["status"] = "SHADOW_ABORT_SAFE"
        result["error"] = "postgrasp_shadow output directory already exists"
        return result
    output_root.mkdir(parents=True, exist_ok=False)
    h0_arm = np.asarray(current_arm_target, dtype=np.float64).copy()
    current_arm = h0_arm.copy()
    try:
        initial_wrist_step, initial_wrist_canonical, _ = _read_latest_wrist_state(
            get_latest_wrist_state, -1
        )
    except Exception as exception:
        return {
            "requested": True,
            "passed_frozen_before_shadow": True,
            "shadow_authorized": False,
            "control_authorized": False,
            "key_branch_selection": _current_proxy_key_branch_block(),
            "snapshot_restore_verified": False,
            "status": "SHADOW_ABORT_SAFE",
            "error": f"{type(exception).__name__}: {exception}",
        }
    wrist_sample_step = initial_wrist_step
    motion_records = []
    views = []
    view_records = []
    moved_views = []
    return_to_h0 = {"attempted": False, "completed": False, "error": None}
    move_guarded = None
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        plug_position, plug_orientation = body.get_world_pose()
        joint_state = None
        joint_provider_status = "NO_GPU_JOINT_STATE_PROVIDER"
        if nut_joint_state_provider is not None and callable(
            getattr(nut_joint_state_provider, "get_state", None)
        ):
            raw = nut_joint_state_provider.get_state()
            q = float(np.asarray(raw[0], dtype=np.float64).ravel()[0])
            qd = float(np.asarray(raw[1], dtype=np.float64).ravel()[0])
            joint_state = {"q_rad": q, "qd_rad_s": qd}
            joint_provider_status = "JOINT_STATE_PROVIDED"
        truth_document = capture_truth_snapshot(
            snapshot_id=f"postgrasp_shadow_{arguments.seed:06d}",
            timestamp_utc=timestamp,
            episode=f"seed{arguments.seed:06d}",
            seed=int(arguments.seed),
            global_step=int(global_step),
            plug_root_getters={
                "position": lambda: _as_vector3(plug_position),
                "orientation": lambda: _as_quaternion(plug_orientation),
                "linear_velocity": lambda: _as_vector3(body.get_linear_velocity()),
                "angular_velocity": lambda: _as_vector3(body.get_angular_velocity()),
            },
            nut_joint_getters=(
                None
                if joint_state is None
                else {
                    "q": lambda: float(joint_state["q_rad"]),
                    "qd": lambda: float(joint_state["qd_rad_s"]),
                }
            ),
            robot_getters={
                "q": lambda: np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                ).ravel().tolist(),
                "qd": lambda: np.asarray(
                    robot.get_joint_velocities(), dtype=np.float64
                ).ravel().tolist(),
            },
            frozen_command={
                "arm_q_target_rad": h0_arm.tolist(),
                "hand_q_target_rad": np.asarray(
                    current_hand_target, dtype=np.float64
                ).tolist(),
                "physical_grasp_method": arguments.physical_grasp_method,
                "wrist_ft_canonical": initial_wrist_canonical.tolist(),
                "wrist_ft_payload_reference": np.asarray(
                    formal_wrist_payload_reference, dtype=np.float64
                ).ravel().tolist(),
            },
            source_hashes=dict(source_hashes or {}),
        )
        snapshot_status = (
            "SNAPSHOT_CAPTURE_ONLY_API_UNSUPPORTED"
            if joint_state is None
            else "SNAPSHOT_CAPTURE_WITH_JOINT_STATE"
        )
        write_truth_snapshot(
            output_root / "snapshot_truth_restore.posthoc.json", truth_document
        )
        restore_probe = probe_restore_api({})
        (output_root / "restore_probe.json").write_text(
            json.dumps(
                {
                    "status": restore_probe.status,
                    "mode": None,
                    "available_apis": [],
                    "missing_apis": [
                        "plug_root_set_pose",
                        "plug_root_set_linear_velocity",
                        "plug_root_set_angular_velocity",
                        "nut_joint_set_state",
                    ],
                    "snapshot_capture_status": snapshot_status,
                    "snapshot_restore_verified": False,
                    "joint_state_provider_status": joint_provider_status,
                    "gpu_readonly_probe_required": True,
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result["snapshot"] = {
            "status": snapshot_status,
            "snapshot_restore_verified": False,
            "joint_state_provider_status": joint_provider_status,
        }

        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        from PIL import Image

        from kcg_connector.isaac_d38999_rgbd_runtime import (
            capture_d38999_rgbd_raw_formal,
        )
        from kcg_connector.rgbd_pose_bootstrap import load_rgbd_bootstrap

        config_doc = yaml.safe_load(
            (
                repository
                / "src/kcg_connector/config/d38999_postgrasp_shadow_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        motion_config = config_doc["motion"]
        mount = config_doc["views"]["wrist_camera_mount"]
        handbase_to_tcp = float(pick.geometry_candidate.handbase_to_tcp_m)
        tcp_from_handbase = np.eye(4, dtype=np.float64)
        tcp_from_handbase[2, 3] = -handbase_to_tcp
        if mount.get("control_authorized") is not False:
            raise RuntimeError("simulation wrist mount must stay unauthorized")
        T_HC_fixed_mount_candidate = _camera_cv_pose_from_eye_target(
            mount["eye_handbase_m"],
            mount["target_handbase_m"],
        )
        rgbd_base = load_rgbd_bootstrap(
            Path(arguments.rgbd_config).expanduser().resolve()
        )
        handbase_path = pick.scene.grasp_tcp_prim_path.rsplit("/", 1)[0]
        camera_path = handbase_path + "/WristCamera"
        _require_frozen_camera_optics(stage, camera_path, UsdGeom)

        def move_guarded(target_arm, phase_label):
            nonlocal current_arm
            nonlocal motion_records
            nonlocal wrist_sample_step
            start_arm = current_arm.copy()
            target = np.asarray(target_arm, dtype=np.float64)
            try:
                planned = validate_planned_view_motion(
                    start_arm,
                    target,
                    h0_arm,
                    motion_config,
                    phase_label=phase_label,
                )
            except ShadowViewMotionAbort as exception:
                motion_records.append(
                    {
                        "phase": phase_label,
                        "status": "PLANNED_MOTION_REJECTED",
                        "start_arm": start_arm.tolist(),
                        "target_arm": target.tolist(),
                        "reason": str(exception),
                    }
                )
                raise
            motion_records.append(
                {
                    "phase": phase_label,
                    "status": "PLANNED",
                    **planned,
                }
            )
            move_steps = max(
                1, round(float(motion_config["move_duration_s"]) * rate_hz)
            )
            tracking_limit = float(
                physical_grasp.stability.maximum_arm_tracking_error_rad
            )
            torque_limit = float(pick.sensing.maximum_absolute_torque_delta_nm)
            force_limit = 8.0
            moment_limit = 0.30
            previous_command = start_arm.copy()
            for step in range(move_steps):
                blend = float(step + 1) / float(move_steps)
                current_arm = np.asarray(
                    start_arm + blend * (target - start_arm),
                    dtype=np.float64,
                )
                command_delta = float(
                    np.max(np.abs(current_arm - previous_command))
                )
                if command_delta > float(
                    motion_config["per_command_max_joint_delta_rad"]
                ):
                    raise ShadowViewMotionAbort(
                        "view per-command joint delta exceeded"
                    )
                previous_command = current_arm.copy()
                try:
                    positions, velocities = observe_and_step(
                        current_arm, current_hand_target, True
                    )
                    measured = sample_post_tare_efforts()
                except Exception as exception:
                    raise ShadowViewMotionAbort(
                        f"motion sensor/step failure: {type(exception).__name__}"
                    ) from exception
                wrist_sample_step, wrist_canonical, wrist_error = (
                    _read_latest_wrist_state(
                        get_latest_wrist_state, wrist_sample_step
                    )
                )
                reason = evaluate_view_motion_safety(
                    positions=positions,
                    velocities=velocities,
                    arm_target=current_arm,
                    arm_indices=arm_indices,
                    measured_efforts=measured,
                    tare_efforts=tare_efforts,
                    wrist_canonical=wrist_canonical,
                    wrist_payload_reference=formal_wrist_payload_reference,
                    wrist_ft_monitor_error=wrist_error,
                    tracking_limit_rad=tracking_limit,
                    torque_limit_nm=torque_limit,
                    force_limit_n=force_limit,
                    moment_limit_nm=moment_limit,
                )
                if reason is not None:
                    raise ShadowViewMotionAbort(reason)
            settle_steps = max(
                1, round(float(motion_config["settle_duration_s"]) * rate_hz)
            )
            settle_command_delta = float(
                np.max(np.abs(target - previous_command))
            )
            if settle_command_delta > float(
                motion_config["per_command_max_joint_delta_rad"]
            ):
                raise ShadowViewMotionAbort(
                    "view settle per-command joint delta exceeded"
                )
            for _ in range(settle_steps):
                try:
                    positions, velocities = observe_and_step(
                        target, current_hand_target, True
                    )
                    measured = sample_post_tare_efforts()
                except Exception as exception:
                    raise ShadowViewMotionAbort(
                        f"settle sensor/step failure: {type(exception).__name__}"
                    ) from exception
                wrist_sample_step, wrist_canonical, wrist_error = (
                    _read_latest_wrist_state(
                        get_latest_wrist_state, wrist_sample_step
                    )
                )
                reason = evaluate_view_motion_safety(
                    positions=positions,
                    velocities=velocities,
                    arm_target=target,
                    arm_indices=arm_indices,
                    measured_efforts=measured,
                    tare_efforts=tare_efforts,
                    wrist_canonical=wrist_canonical,
                    wrist_payload_reference=formal_wrist_payload_reference,
                    wrist_ft_monitor_error=wrist_error,
                    tracking_limit_rad=tracking_limit,
                    torque_limit_nm=torque_limit,
                    force_limit_n=force_limit,
                    moment_limit_nm=moment_limit,
                )
                if reason is not None:
                    raise ShadowViewMotionAbort(reason)
            current_arm = target.copy()

        def capture_view(plan_index, plan):
            nonlocal current_arm
            arm_q_actual = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )[np.asarray(arm_indices, dtype=np.int64)]
            tcp_fk = _iiwa_tcp_fk(arm_q_actual)
            T_WH = tcp_fk @ tcp_from_handbase
            T_WC = T_WH @ T_HC_fixed_mount_candidate
            eye_world = T_WC[:3, 3]
            target_world = _camera_target_from_t_wc(T_WC)
            camera_model = _camera_model_from_eye_target(
                eye_world, target_world, (1280, 720)
            )
            camera_model = replace(
                camera_model,
                position_world=tuple(float(value) for value in T_WC[:3, 3]),
                world_to_camera=tuple(
                    tuple(float(value) for value in row)
                    for row in T_WC[:3, :3].T
                ),
            )
            wrist_rgbd = replace(
                rgbd_base,
                camera=replace(
                    rgbd_base.camera,
                    prim_path=camera_path,
                    frame_id="postgrasp_wrist_rgbd_camera_optical",
                    eye_m=tuple(float(v) for v in eye_world),
                    target_m=tuple(float(v) for v in target_world),
                    resolution=(1280, 720),
                ),
            )
            capture_timestamp = datetime.now(timezone.utc).isoformat()
            capture = capture_d38999_rgbd_raw_formal(
                bindings={
                    "Camera": Camera,
                    "Gf": Gf,
                    "Image": Image,
                    "Usd": Usd,
                    "UsdGeom": UsdGeom,
                    "UsdLux": UsdLux,
                    "rep": rep,
                },
                simulation_app=simulation_app,
                world=world,
                stage=stage,
                tabletop=tabletop,
                rgbd=wrist_rgbd,
                output_dir=output_root / ("raw_rgbd_" + plan.view_id.lower()),
            )
            if not capture.passed:
                raise RuntimeError("raw formal capture failed")
            view = FormalView(
                view_id=plan.view_id,
                timestamp_utc=capture_timestamp,
                rgb=capture.rgb,
                depth=capture.depth,
                camera=camera_model,
                T_WH=T_WH,
                T_HC=T_HC_fixed_mount_candidate.copy(),
                T_WC=T_WC,
                group="postgrasp_inhand_views",
            )
            views.append(view)
            view_records.append(
                {
                    "phase": f"postgrasp_shadow_view_{plan.view_id.lower()}",
                    "view_id": plan.view_id,
                    "score": score_formal_view(
                        view_id=plan.view_id,
                        timestamp_utc=capture_timestamp,
                        rgb=view.rgb,
                        depth=view.depth,
                        camera=camera_model,
                    ),
                }
            )
            moved_views.append(plan.view_id)

        # H0 contract: no V1/V2 arm observation motion is attempted.
        capture_view(0, DEFAULT_POSTGRASP_PLANS[0])

        fixed_camera = fixed_camera_model(
            eye=tuple(float(v) for v in rgbd_base.camera.eye_m),
            target=tuple(float(v) for v in rgbd_base.camera.target_m),
            resolution=tuple(int(v) for v in rgbd_base.camera.resolution),
        )
        T_WC_fixed = _t_wc_from_camera_model(fixed_camera)
        fixed_capture_timestamp = datetime.now(timezone.utc).isoformat()
        fixed_capture = capture_d38999_rgbd_raw_formal(
            bindings={
                "Camera": Camera,
                "Gf": Gf,
                "Image": Image,
                "Usd": Usd,
                "UsdGeom": UsdGeom,
                "UsdLux": UsdLux,
                "rep": rep,
            },
            simulation_app=simulation_app,
            world=world,
            stage=stage,
            tabletop=tabletop,
            rgbd=rgbd_base,
            output_dir=output_root / "raw_rgbd_fixed_world_camera",
        )
        if not fixed_capture.passed:
            raise RuntimeError("fixed world raw formal capture failed")
        fixed_arm_q = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )[np.asarray(arm_indices, dtype=np.int64)]
        fixed_tcp_fk = _iiwa_tcp_fk(fixed_arm_q)
        fixed_T_WH = fixed_tcp_fk @ tcp_from_handbase
        fixed_view = FormalView(
            view_id="FIXED_WORLD_CAMERA_V0",
            timestamp_utc=fixed_capture_timestamp,
            rgb=fixed_capture.rgb,
            depth=fixed_capture.depth,
            camera=fixed_camera,
            T_WH=fixed_T_WH,
            T_HC=None,
            T_WC=T_WC_fixed,
            group="fixed_world_camera_views",
            extrinsic_source="fixed_camera_config_T_WC",
        )
        views.append(fixed_view)
        view_records.append(
            {
                "phase": "postgrasp_shadow_fixed_world_camera_h0",
                "view_id": fixed_view.view_id,
                "extrinsic_source": fixed_view.extrinsic_source,
                "score": score_formal_view(
                    view_id=fixed_view.view_id,
                    timestamp_utc=fixed_capture_timestamp,
                    rgb=fixed_capture.rgb,
                    depth=fixed_capture.depth,
                    camera=fixed_camera,
                ),
            }
        )

        formal_root = output_root / "formal_views"
        write_formal_archive(formal_root, views)
        (output_root / "control_and_obs.json").write_text(
            json.dumps(
                {
                    "schema_version": "kcg_d38999_postgrasp_shadow_v1",
                    "role": "formal_observation",
                    "snapshot_id": f"postgrasp_shadow_{arguments.seed:06d}",
                    "timestamp_utc": timestamp,
                    "formal_views": str(formal_root),
                    "frozen_command": truth_document["frozen_command"],
                    "shadow_authorized": False,
                    "control_authorized": False,
                },
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        nominal_hp = np.asarray(nominal_hand_to_plug, dtype=np.float64)
        # H0 phase has no valid receptacle-frame prior.  Estimate T_HP only;
        # do not claim T_RP and do not run the joint estimator.
        initial_state = np.concatenate(
            (np.zeros(6, dtype=np.float64), np.zeros(6, dtype=np.float64))
        )
        initial_state[:6] = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        initial_state[:6] = np.asarray(
            _pose_from_matrix(nominal_hp), dtype=np.float64
        )
        shadow_result = estimate_postgrasp_T_HP(views, initial_state)
        (output_root / "shadow_result.json").write_text(
            json.dumps(
                shadow_result, allow_nan=False, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        result.update(
            {
                "views": view_records,
                "formal_archive": str(formal_root),
                "shadow_result_path": str(output_root / "shadow_result.json"),
                "shadow_estimate_status": shadow_result.get("status"),
                "T_RP_estimated": False,
                "T_RP_status": "NOT_ESTIMATED_H0_PHASE_NO_VALID_PRIOR",
                "shadow_result": shadow_result,
                "motion_records": motion_records,
                "status": "COMPLETED_SHADOW_ONLY",
                "shadow_authorized": False,
                "control_authorized": False,
            }
        )
    except ShadowViewMotionAbort as exception:
        result["partial_archive"] = _preserve_partial_formal_archive(
            output_root, views
        )
        _attempt_return_to_h0(
            move_guarded, h0_arm, current_arm, return_to_h0
        )
        result["status"] = "SHADOW_ABORT_SAFE_VIEW_MOTION"
        result["error"] = f"{type(exception).__name__}: {exception}"
        result["return_to_h0"] = return_to_h0
    except Exception as exception:
        result["partial_archive"] = _preserve_partial_formal_archive(
            output_root, views
        )
        if move_guarded is not None and np.max(np.abs(current_arm - h0_arm)) > 1.0e-9:
            _attempt_return_to_h0(
                move_guarded, h0_arm, current_arm, return_to_h0
            )
        result["status"] = "SHADOW_ABORT_SAFE"
        result["error"] = f"{type(exception).__name__}: {exception}"
        result["return_to_h0"] = return_to_h0
    result["motion_records"] = motion_records
    return result


def _preserve_partial_formal_archive(output_root, views):
    if not views:
        return None
    formal_root = output_root / "formal_views"
    if formal_root.exists():
        return str(formal_root)
    try:
        write_formal_archive(formal_root, views)
        (output_root / "partial_shadow_archive_status.json").write_text(
            json.dumps(
                {
                    "status": "PARTIAL_VIEWS_PRESERVED",
                    "view_ids": [view.view_id for view in views],
                    "shadow_authorized": False,
                    "control_authorized": False,
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(formal_root)
    except Exception as exception:
        return f"PARTIAL_ARCHIVE_WRITE_FAILED:{type(exception).__name__}"


def _attempt_return_to_h0(move_guarded, h0_arm, current_arm, record):
    record["attempted"] = True
    try:
        if move_guarded is not None:
            move_guarded(h0_arm, "return_to_h0")
            record["completed"] = True
    except Exception as return_error:
        record["error"] = (
            f"{type(return_error).__name__}: {return_error}"
        )


def _pose_from_matrix(matrix: np.ndarray) -> np.ndarray:
    from kcg_connector.d38999_inhand_multiview import matrix_pose

    return np.asarray(
        matrix_pose(np.asarray(matrix, dtype=np.float64)), dtype=np.float64
    )


__all__ = [
    "run_postgrasp_shadow_capture",
    "run_inline_palm_t_hp_capture",
    "selected_t_hp_control_pose",
    "ShadowViewMotionAbort",
]


def run_inline_palm_t_hp_capture(
    *,
    repository: Path,
    arguments,
    Gf,
    Usd,
    UsdGeom,
    UsdLux,
    stage,
    world,
    simulation_app,
    tabletop,
    pick,
    robot,
    arm_indices,
    actual_arm_q,
    palm_t_hc,
    rate_hz,
) -> dict[str, Any]:
    """Capture the palm view of the Plug mating face in the SAME episode
    and estimate T_HP in-process.  No object truth, no contact report."""
    result: dict[str, Any] = {
        "requested": True,
        "capture_status": "NOT_RUN",
        "estimate_status": "NOT_RUN",
        "pose_valid": False,
        "covariance_calibration_status": "UNVALIDATED",
        "c2_resolution": "NOT_EVALUATED",
        "selected_c2_hypothesis_id": None,
        "real_keying_modeled": False,
        "keying_model_id": None,
        "key_branch_selection": _current_proxy_key_branch_block(),
        "control_authorized": False,
        "status": "STARTED",
    }
    output_root = (
        Path(arguments.output_dir).expanduser().resolve()
        / "inline_palm_t_hp"
    )
    if output_root.exists():
        result["status"] = "ABORT_SAFE_OUTPUT_EXISTS"
        return result
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        from PIL import Image

        from kcg_connector.isaac_d38999_rgbd_runtime import (
            capture_d38999_rgbd_raw_formal,
        )
        from kcg_connector.rgbd_pose_bootstrap import load_rgbd_bootstrap
        from postgrasp_shadow_capture_runtime import (
            _camera_target_from_t_wc,
        )

        rgbd_base = load_rgbd_bootstrap(
            Path(arguments.rgbd_config).expanduser().resolve()
        )
        t_hc = np.asarray(palm_t_hc, dtype=np.float64)
        if t_hc.shape != (4, 4) or not np.all(np.isfinite(t_hc)):
            raise ValueError("palm_t_hc must be a finite 4x4 transform")

        from kcg_connector.d38999_tabletop_pick import (
            iiwa14_grasp_tcp_transform,
        )

        actual_arm_q = np.asarray(actual_arm_q, dtype=np.float64).ravel()
        if actual_arm_q.shape != (7,) or not np.all(np.isfinite(actual_arm_q)):
            raise ValueError("actual_arm_q must be a finite 7-vector")
        tcp_fk = np.asarray(
            iiwa14_grasp_tcp_transform(tuple(float(v) for v in actual_arm_q))
        )
        handbase_to_tcp = float(pick.geometry_candidate.handbase_to_tcp_m)
        tcp_from_handbase = np.eye(4, dtype=np.float64)
        tcp_from_handbase[2, 3] = -handbase_to_tcp
        t_wh = tcp_fk @ tcp_from_handbase
        t_wc = t_wh @ t_hc
        eye_world = tuple(float(v) for v in t_wc[:3, 3])
        target_world = tuple(float(v) for v in _camera_target_from_t_wc(t_wc))

        handbase_path = pick.scene.grasp_tcp_prim_path.rsplit("/", 1)[0]
        camera_path = handbase_path + "/PalmCamera"
        _require_frozen_camera_optics(stage, camera_path, UsdGeom)
        # Capture reuses the canonical fixed child camera.  The FK-derived
        # world pose below is a readback expectation only; it is never written
        # to the stage after physics has started.
        from dataclasses import replace

        # FAST_INLINE_CANDIDATE: 2 warmup frames and 2 RT subframes per
        # render instead of the validated 5x4, and 9 optimizer starts
        # instead of 17.  The estimate quality gates (RMS < 5, plug
        # support fraction) are unchanged and are re-checked on every run;
        # this only shortens the serial capture/estimation dead time in
        # the grasp->transport span.
        palm_rgbd = replace(
            rgbd_base,
            camera=replace(
                rgbd_base.camera,
                prim_path=camera_path,
                frame_id="postgrasp_palm_rgbd_camera_optical",
                eye_m=eye_world,
                target_m=target_world,
                resolution=(1280, 720),
                warmup_frames=2,
            ),
        )
        view_dir = output_root / "PALM_INLINE"
        capture = capture_d38999_rgbd_raw_formal(
            bindings={"Camera": Camera, "Gf": Gf, "Image": Image,
                      "Usd": Usd, "UsdGeom": UsdGeom, "UsdLux": UsdLux,
                      "rep": rep},
            simulation_app=simulation_app,
            world=world,
            stage=stage,
            tabletop=tabletop,
            rgbd=palm_rgbd,
            output_dir=view_dir,
            camera_clipping_range_m=(0.02, 10.0),
            rt_subframes=2,
        )
        if capture.passed is not True:
            raise RuntimeError("inline palm capture failed")
        import json

        (view_dir / "fk.json").write_text(
            json.dumps(
                {
                    "arm_q_actual_rad": actual_arm_q.tolist(),
                    "T_HC_4x4": t_hc.tolist(),
                    "T_WH_4x4": t_wh.tolist(),
                    "T_WC_4x4": t_wc.tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (view_dir / "camera.json").write_text(
            json.dumps(
                {
                    "prim_path": camera_path,
                    "eye_m": list(eye_world),
                    "target_m": list(target_world),
                    "intrinsics": capture.metrics["camera"]["intrinsics"],
                    "T_HC_4x4": t_hc.tolist(),
                    "mount_contract": (
                        "SIM_VISUAL_MOUNT_CANDIDATE_FIXED_T_HC"
                    ),
                    "control_authorized": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # in-process formal estimation (same config as the validated chain)
        import math

        import cv2

        from kcg_connector.d38999_cad_registration import (
            CameraModel,
            proxy_cad_points,
            shell25j_plug_cad_profile,
        )
        from kcg_connector.hand_occluder_cad import (
            build_hand_occluder_cad,
        )
        from kcg_connector.postgrasp_shadow_estimator import (
            FormalView,
            estimate_postgrasp_T_HP,
        )

        rgb = cv2.cvtColor(
            cv2.imread(str(view_dir / "rgb.png"), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        depth = np.load(view_dir / "depth_m.npy").astype(np.float32)
        intrinsics = np.asarray(capture.metrics["camera"]["intrinsics"])
        camera = CameraModel(
            1280, 720,
            float(intrinsics[0, 0]), float(intrinsics[1, 1]),
            float(intrinsics[0, 2]), float(intrinsics[1, 2]),
            tuple(float(value) for value in t_wc[:3, 3]),
            tuple(
                tuple(float(value) for value in row)
                for row in t_wc[:3, :3].T
            ),
        )
        view = FormalView(
            view_id="PALM_INLINE",
            timestamp_utc="inline",
            rgb=rgb,
            depth=depth,
            camera=camera,
            T_WH=t_wh,
            T_WC=t_wc,
            group="postgrasp_inhand_views",
            extrinsic_source="T_HC_fixed_mount_candidate",
        )
        initial_state = np.zeros(12, dtype=np.float64)
        initial_state[:6] = np.array([0.0, 0.0, 0.4485, math.pi, 0.0, 0.0])
        legacy_plug, legacy_receptacle = proxy_cad_points()
        shell_profile = shell25j_plug_cad_profile(
            feature_set="shell_plus_socket"
        )
        hand_occluder = build_hand_occluder_cad(
            (1.0, 0.7587, 1.0, 0.5721, 0.5721, 1.0, 0.7601, 0.7601),
            repository / "artifacts/kcg_connector/urdf/handarm.urdf",
            repository / "src/iiwa_description/meshes/hand",
        )
        t_hp_result = estimate_postgrasp_T_HP(
            [view],
            initial_state,
            plug_cad=legacy_plug,
            receptacle_cad=legacy_receptacle,
            plug_occluder_cad=shell_profile.plug_occluders,
            hand_occluder_cad=hand_occluder,
            occlusion_policy="baseline",
            edge_policy="depth_gated",
            optimizer_variant="multistart_physical_jacobian",
            multistart_count=5,
        )
        result["capture_status"] = "GPU_CAPTURE_VALID"
        result["estimate_status"] = t_hp_result.get("status")
        result["optimizer_converged"] = t_hp_result["optimizer_converged"]
        result["support_gate_failed"] = t_hp_result["plug_support_gate_failed"]
        result["residual_rms"] = [
            h["residual_rms"] for h in t_hp_result["c2"]["hypotheses"]
        ]
        result["c2_hypotheses"] = [
            {
                "id": h["id"],
                "T_hand_plug_xyz_rpy": h["T_hand_plug_xyz_rpy"],
            }
            for h in t_hp_result["c2"]["hypotheses"]
        ]
        result["c2_resolution"] = t_hp_result["c2"]["resolution"]
        result["pose_valid"] = t_hp_result["pose_valid"]
        result["pose_valid_reasons"] = t_hp_result["pose_valid_reasons"]
        result["covariance_calibration_status"] = t_hp_result.get(
            "covariance_calibration_status", "UNVALIDATED"
        )
        result["control_authorized"] = (
            t_hp_result.get("control_authorized") is True
        )
        result["selected_c2_hypothesis_id"] = t_hp_result["c2"].get(
            "selected_for_control"
        )
        result["real_keying_modeled"] = (
            t_hp_result.get("real_keying_modeled") is True
        )
        result["keying_model_id"] = t_hp_result.get("keying_model_id")
        result["key_branch_selection"] = t_hp_result["key_branch_selection"]
        result["status"] = (
            "CONTROL_AUTHORIZED"
            if selected_t_hp_control_pose(result) is not None
            else "SHADOW_ESTIMATE_REJECTED"
        )
        (output_root / "inline_palm_t_hp_result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exception:  # noqa: BLE001
        result["status"] = "ABORT_SAFE"
        result["error"] = f"{type(exception).__name__}: {exception}"
    return result
