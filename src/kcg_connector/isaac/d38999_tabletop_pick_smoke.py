#!/usr/bin/env python3

"""Physically screen an independent tabletop pick of the D38999 proxy.

The robot uses real PhysX contact and the three scalar finger-base effort
channels.  No attachment, object pose write, kinematic object drive, or
fingertip sensor is used.  The joint interpolation is not collision planning,
and self collision remains disabled/unverified in the imported robot asset.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from numbers import Real
from pathlib import Path
import traceback

import numpy as np

EXPECTED_DOF_NAMES = tuple(f"iiwa_joint_{index}" for index in range(1, 8)) + (
    "f1j1",
    "f1j2",
    "f1j3",
    "f2j1",
    "f2j2",
    "f3j1",
    "f3j2",
    "f3j3",
)
CAMERA_EYE_M = (1.55, 1.25, 0.85)
CAMERA_TARGET_M = (0.43, -0.08, 0.38)


def _path_is_at_or_below(path, root):
    value = str(path)
    prefix = str(root)
    return value == prefix or value.startswith(prefix + "/")


def _pair_contains_subtree(paths, root):
    return any(_path_is_at_or_below(path, root) for path in paths)


def _classify_robot_external_contact(
    paths,
    robot_root,
    table_path,
    fixture_path,
    receptacle_path,
    plug_path,
):
    """Classify only contacts containing the exact robot subtree."""

    values = tuple(str(path) for path in paths)
    if not _pair_contains_subtree(values, robot_root):
        return None
    for category, path in (
        ("table", table_path),
        ("fixture", fixture_path),
        ("fixed_endpoint", receptacle_path),
        ("loose_plug", plug_path),
    ):
        if _pair_contains_subtree(values, path):
            return category
    return None


def _is_plug_table_contact(paths, plug_path, table_path):
    values = tuple(str(path) for path in paths)
    return bool(
        _pair_contains_subtree(values, plug_path)
        and _pair_contains_subtree(values, table_path)
    )


def _is_finger_plug_contact(paths, robot_root, plug_path):
    """Require both D38999 loose plug and a modeled finger-link subtree."""

    values = tuple(str(path) for path in paths)
    finger_path_present = any(
        _path_is_at_or_below(path, robot_root)
        and any(
            link_name in path
            for link_name in ("/f1Link", "/f2Link", "/f3Link")
        )
        for path in values
    )
    return bool(
        finger_path_present and _pair_contains_subtree(values, plug_path)
    )


def _gf_quaternion_error_radians(first, second):
    relative = first.GetInverse() * second
    real = max(-1.0, min(1.0, abs(float(relative.GetReal()))))
    return 2.0 * math.acos(real)


def _gf_quaternion_finite(value):
    imaginary = value.GetImaginary()
    return all(
        math.isfinite(item)
        for item in (
            float(value.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    )


def _array_quaternion_error_radians(first, second):
    """Return shortest rotation angle for scalar-first quaternions."""

    first_values = tuple(float(value) for value in first)
    second_values = tuple(float(value) for value in second)
    first_norm = math.sqrt(sum(value * value for value in first_values))
    second_norm = math.sqrt(sum(value * value for value in second_values))
    if first_norm <= 0.0 or second_norm <= 0.0:
        return float("nan")
    dot = sum(
        left * right for left, right in zip(first_values, second_values)
    ) / (first_norm * second_norm)
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(dot))))


def _world_pose(Gf, Usd, UsdGeom, prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform(matrix)
    return transform.GetTranslation(), transform.GetRotation().GetQuat()


def _quaternion_world_z_axis(value):
    imaginary = value.GetImaginary()
    values = (
        float(value.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 0.0 or not math.isfinite(norm):
        return (float("nan"),) * 3
    w, x, y, z = (item / norm for item in values)
    return (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


def _axis_error_radians(first, second):
    cosine = sum(left * right for left, right in zip(first, second))
    return math.acos(max(-1.0, min(1.0, cosine)))


def _wrapped_relative_z_angle(Gf, Usd, UsdGeom, body_prim, nut_prim):
    """Return CouplingNut yaw relative to BodyAssembly in [-pi, pi]."""

    # Reuse the exact quaternion convention from the independently validated
    # D38999 q7 runner.  Gf matrix multiplication follows row-vector
    # conventions, so deriving yaw from ``nut * inverse(body)`` reverses the
    # sign even though the physical rack motion itself is correct.
    body_matrix = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    nut_matrix = UsdGeom.Xformable(nut_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    body_quaternion = Gf.Transform(body_matrix).GetRotation().GetQuat()
    nut_quaternion = Gf.Transform(nut_matrix).GetRotation().GetQuat()
    relative = body_quaternion.GetInverse() * nut_quaternion
    imaginary = relative.GetImaginary()
    angle = 2.0 * math.atan2(
        float(imaginary[2]), float(relative.GetReal())
    )
    return math.atan2(math.sin(angle), math.cos(angle))


def _unwrap_angle(previous, wrapped):
    """Accumulate a wrapped angle without losing full-turn progress."""

    previous_wrapped = math.atan2(math.sin(previous), math.cos(previous))
    delta = wrapped - previous_wrapped
    if delta > math.pi:
        delta -= math.tau
    elif delta < -math.pi:
        delta += math.tau
    return previous + delta


def _d38999_loose_collider_group(path, body_root, nut_root):
    """Classify one loose collider into the two physical rigid bodies."""

    if _path_is_at_or_below(path, body_root) and str(path) != str(body_root):
        return "body"
    if _path_is_at_or_below(path, nut_root) and str(path) != str(nut_root):
        return "nut"
    return None


def _finger_loose_contact_group(paths, robot_root, body_root, nut_root):
    """Return the modeled finger and loose rigid-body group for a contact."""

    finger_names = []
    loose_groups = []
    for raw_path in paths:
        path = str(raw_path)
        if _path_is_at_or_below(path, robot_root):
            for finger_name in ("f1", "f2", "f3"):
                if f"/{finger_name}Link" in path:
                    finger_names.append(finger_name)
        group = _d38999_loose_collider_group(path, body_root, nut_root)
        if group is not None:
            loose_groups.append(group)
    unique_fingers = tuple(sorted(set(finger_names)))
    unique_groups = tuple(sorted(set(loose_groups)))
    if len(unique_fingers) != 1 or len(unique_groups) != 1:
        return None
    return unique_fingers[0], unique_groups[0]


def _all_fingers_have_body_contact(contact_snapshot):
    """Require each modeled finger to contact the plug body assembly."""

    if not isinstance(contact_snapshot, Mapping):
        return False
    grouped_records = contact_snapshot.get("finger_body_group_records")
    if not isinstance(grouped_records, Mapping):
        return False
    for finger_name in ("f1", "f2", "f3"):
        finger_records = grouped_records.get(finger_name)
        if not isinstance(finger_records, Mapping):
            return False
        body_records = finger_records.get("body")
        if isinstance(body_records, bool) or not isinstance(
            body_records, Real
        ):
            return False
        if not math.isfinite(float(body_records)) or body_records <= 0:
            return False
    return True


def _all_fingers_have_nut_contact(contact_snapshot):
    """Require an actual CouplingNut contact record for every finger."""

    if not isinstance(contact_snapshot, Mapping):
        return False
    grouped = contact_snapshot.get("finger_body_group_records")
    if not isinstance(grouped, Mapping):
        return False
    for finger_name in ("f1", "f2", "f3"):
        records = grouped.get(finger_name)
        if not isinstance(records, Mapping):
            return False
        value = records.get("nut")
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            return False
    return True


def _zero_finger_body_contact(contact_snapshot):
    """Reject a nut-only grip that still clamps BodyAssembly."""

    if not isinstance(contact_snapshot, Mapping):
        return False
    grouped = contact_snapshot.get("finger_body_group_records")
    if not isinstance(grouped, Mapping):
        return False
    return all(
        isinstance(grouped.get(finger_name), Mapping)
        and grouped[finger_name].get("body") == 0
        for finger_name in ("f1", "f2", "f3")
    )


def _zero_finger_endpoint_contact(contact_snapshot):
    """Verify that an opened hand is clear of both loose rigid bodies."""

    if not isinstance(contact_snapshot, Mapping):
        return False
    grouped = contact_snapshot.get("finger_body_group_records")
    if not isinstance(grouped, Mapping):
        return False
    return all(
        isinstance(grouped.get(finger_name), Mapping)
        and grouped[finger_name].get("body") == 0
        and grouped[finger_name].get("nut") == 0
        for finger_name in ("f1", "f2", "f3")
    )


def _array_quaternion_conjugate(value):
    values = np.asarray(value, dtype=np.float64)
    return np.asarray(
        (values[0], -values[1], -values[2], -values[3]),
        dtype=np.float64,
    )


def _array_quaternion_multiply(first, second):
    aw, ax, ay, az = (float(value) for value in first)
    bw, bx, by, bz = (float(value) for value in second)
    return np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=np.float64,
    )


def _relative_array_quaternion(first, second):
    """Return normalized first^-1 * second for scalar-first quaternions."""

    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first_values))
    second_norm = float(np.linalg.norm(second_values))
    if (
        first_norm <= 0.0
        or second_norm <= 0.0
        or not math.isfinite(first_norm)
        or not math.isfinite(second_norm)
    ):
        return np.full(4, float("nan"), dtype=np.float64)
    first_values = first_values / first_norm
    second_values = second_values / second_norm
    result = _array_quaternion_multiply(
        _array_quaternion_conjugate(first_values), second_values
    )
    result_norm = float(np.linalg.norm(result))
    if result_norm <= 0.0 or not math.isfinite(result_norm):
        return np.full(4, float("nan"), dtype=np.float64)
    return result / result_norm


def _speed_statistics(samples):
    """Return JSON-safe scalar statistics for a non-empty speed series."""

    values = np.asarray(samples, dtype=np.float64)
    return {
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rms": float(math.sqrt(float(np.mean(values * values)))),
    }


def _json_safe(value):
    """Replace every non-finite scalar before fail-closed JSON output."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    return value


def _metrics_json(metrics):
    return json.dumps(_json_safe(metrics), allow_nan=False, sort_keys=True)


def main():
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
        ),
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument(
        "--insertion-probe",
        action="store_true",
        help=(
            "continue the validated physical pick through high transport, "
            "12 mm preinsert and 3 mm engage in the same World"
        ),
    )
    parser.add_argument(
        "--insertion-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_physical_insertion_v1.yaml"
        ),
    )
    parser.add_argument(
        "--end-to-end-probe",
        action="store_true",
        help=(
            "continue the same physical scene through insertion, nut-only "
            "regrasp, full rotation, release, retreat and Home"
        ),
    )
    parser.add_argument(
        "--pose-preflight",
        choices=("none", "masked-rgbd"),
        default="none",
        help=(
            "run the selected pose preflight in this same World after the "
            "initial settle and before intentional robot motion"
        ),
    )
    parser.add_argument(
        "--rgbd-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_rgbd_bootstrap_v1.yaml"
        ),
    )
    parser.add_argument(
        "--smooth-demo",
        action="store_true",
        help=(
            "use an unvalidated visual-demo timing profile: Home hand opening "
            "is 4x faster, while q7 twist, released-hand rewind and the "
            "safe-height q7 return are 1.4x faster; the loaded twist retains "
            "all existing safety gates"
        ),
    )
    parser.add_argument(
        "--regrasp-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_nut_regrasp_physx_v1.yaml"
        ),
    )
    parser.add_argument(
        "--twist-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_q7_twist_probe_stage120_v1.yaml"
        ),
    )
    parser.add_argument(
        "--rewind-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_q7_rewind_probe_v1.yaml"
        ),
    )
    parser.add_argument(
        "--wrist-ft-monitor",
        action="store_true",
        help=(
            "opt-in observation only: record the existing hand2arm 6D "
            "reaction wrench; it does not modify E2E pass/fail or residual-v0"
        ),
    )
    parser.add_argument(
        "--wrist-ft-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_wrist_ft_monitor_v1.yaml"
        ),
    )
    parser.add_argument(
        "--nut-tooth-jitter-output",
        help=(
            "opt-in: sample all 24 CouplingNut teeth at 240 Hz during the "
            "end-to-end insertion-to-twist window and write CSV/JSON here"
        ),
    )
    parser.add_argument(
        "--nut-tooth-jitter-normalize-segment00-op",
        action="store_true",
        help=(
            "A/B only: add explicit Segment_00 rotateZ=0 in the session "
            "layer before play; the checked-in USDA is never edited"
        ),
    )
    parser.add_argument(
        "--nut-tooth-jitter-colorize",
        action="store_true",
        help="author deterministic 24-tooth displayColor IDs in-session",
    )
    arguments = parser.parse_args()
    if arguments.smooth_demo and not arguments.end_to_end_probe:
        parser.error("--smooth-demo requires --end-to-end-probe")
    if (
        arguments.pose_preflight == "masked-rgbd"
        and not arguments.end_to_end_probe
    ):
        parser.error(
            "--pose-preflight masked-rgbd requires --end-to-end-probe"
        )
    # Keep the validated baseline as the default.  The unvalidated visual
    # profile shortens Home hand opening by 4x and q7 twist, released-hand
    # rewind and safe-height q7 return by 1.4x.  Twist remains a loaded contact
    # motion: its contact, torque, tracking and hold gates are unchanged.
    home_hand_open_speedup = 4.0 if arguments.smooth_demo else 1.0
    q7_motion_speedup = 1.4 if arguments.smooth_demo else 1.0
    if arguments.end_to_end_probe:
        # End-to-end extends the exact insertion path.  It cannot select a
        # prepared-state shortcut or create a second simulation scene.
        arguments.insertion_probe = True
        result_marker = "ISAAC D38999 END TO END V1 "
    elif arguments.insertion_probe:
        result_marker = "ISAAC D38999 TABLETOP PICK TO ENGAGE V1 "
    else:
        result_marker = "ISAAC D38999 TABLETOP PICK V1 "
    if arguments.keep_open and not arguments.gui:
        parser.error("--keep-open requires --gui")
    if arguments.nut_tooth_jitter_output and not arguments.end_to_end_probe:
        parser.error("--nut-tooth-jitter-output requires --end-to-end-probe")
    if arguments.wrist_ft_monitor and not arguments.end_to_end_probe:
        parser.error("--wrist-ft-monitor requires --end-to-end-probe")
    if (
        arguments.nut_tooth_jitter_normalize_segment00_op
        or arguments.nut_tooth_jitter_colorize
    ) and not arguments.nut_tooth_jitter_output:
        parser.error(
            "nut-tooth A/B/color options require "
            "--nut-tooth-jitter-output"
        )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not arguments.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    passed = False
    tooth_probe = None
    wrist_ft_monitor = None
    wrist_ft_monitor_error = None
    wrist_ft_sensor_prim = None
    pose_preflight_gate = arguments.pose_preflight == "none"
    metrics = {
        "attachment": "none",
        "candidate_kind": "geometry_screened_not_dynamics_validated",
        "control_pose_provider": "sim_ground_truth",
        "fingertip_tactile_sensors": "none",
        "foundation_pose": False,
        "gui": arguments.gui,
        "insertion_probe_requested": arguments.insertion_probe,
        "end_to_end_probe_requested": arguments.end_to_end_probe,
        "keep_open": arguments.keep_open,
        "motion_profile": (
            "smooth_demo_v1" if arguments.smooth_demo else "baseline_v1"
        ),
        "motion_speedups": {
            "home_hand_open": home_hand_open_speedup,
            "q7_twist_rewind_and_return": q7_motion_speedup,
        },
        "object_pose_writes_after_start": 0,
        "object_drive": "none",
        "masked_rgbd_xy_used_for_control": False,
        "passed": False,
        "pose_preflight_requested": arguments.pose_preflight,
        "scene": "kcg_d38999_tabletop_pick_v1",
        "self_collision_enabled_in_asset": False,
        "self_collision_verified": False,
        "torque_channels": ["f1j2", "f2j1", "f3j2"],
        "operational_torque_target_nm": 1.8,
        "trajectory_kind": (
            "joint_interpolation_screening_not_collision_planned"
        ),
        "truth_orientation_used": True,
        "wrist_ft_monitor_requested": arguments.wrist_ft_monitor,
    }
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
        )
        from isaacsim.core.utils.types import ArticulationAction
        from omni.physx import get_physx_simulation_interface
        from omni.physx.scripts import physicsUtils
        from pxr import (
            Gf,
            PhysxSchema,
            PhysicsSchemaTools,
            Sdf,
            Usd,
            UsdGeom,
            UsdLux,
            UsdPhysics,
            UsdShade,
        )

        from kcg_connector.d38999_tabletop_pick import (
            iiwa14_grasp_tcp_transform,
            interpolate_arm,
            load_d38999_tabletop_pick_config,
            minimum_jerk_blend,
            verify_d38999_pick_dependencies,
        )
        from kcg_connector.d38999_assembly_baseline import (
            load_d38999_assembly_baseline,
        )
        from kcg_connector.d38999_full_rotation import (
            evaluate_d38999_full_rotation,
            validate_final_seating_contact_pairs,
        )
        from kcg_connector.d38999_physical_insertion import (
            axial_gap_waypoints,
            compensated_tcp_transform,
            load_d38999_physical_insertion,
            measure_alignment,
            pose_transform,
            solve_fixed_q7_tcp_pose,
            verify_insertion_inputs,
        )
        from kcg_connector.d38999_proxy_collision_filter import (
            apply_proxy_collision_filter,
            build_proxy_collision_filter_plan,
        )
        from kcg_connector.d38999_tabletop_scene import (
            author_d38999_tabletop_scene,
        )
        from d38999_nut_tooth_jitter_probe import (
            NutToothJitterProbe,
            colorize_segments_session,
            normalize_segment00_rotate_z_session,
        )

        config_path = Path(arguments.config).expanduser().resolve()
        pick = load_d38999_tabletop_pick_config(config_path)
        dependencies = verify_d38999_pick_dependencies(
            pick, config_path, repository
        )
        tabletop = dependencies["tabletop"]
        d38999_asset = dependencies["d38999_asset"]
        robot_asset = dependencies["robot_asset"]
        rate_hz = tabletop.physics.rate_hz
        rgbd = None
        if arguments.pose_preflight == "masked-rgbd":
            # Keep the long-validated default path free of camera/Replicator
            # imports.  The opt-in path loads these only after CLI validation.
            from isaacsim.core.experimental.utils.semantics import (
                add_labels,
                get_labels,
            )
            from isaacsim.sensors.camera import Camera
            import omni.replicator.core as rep
            from PIL import Image

            from kcg_connector.isaac_d38999_rgbd_runtime import (
                capture_d38999_rgbd_runtime,
            )
            from kcg_connector.rgbd_pose_bootstrap import load_rgbd_bootstrap

            rgbd_path = Path(arguments.rgbd_config).expanduser().resolve()
            rgbd = load_rgbd_bootstrap(rgbd_path)
            bound_tabletop_path = (
                repository / rgbd.tabletop_config
            ).resolve()
            active_tabletop_path = (
                config_path.parent / pick.scene.tabletop_config
            ).resolve()
            if bound_tabletop_path != active_tabletop_path:
                raise RuntimeError(
                    "masked RGB-D preflight does not bind the active "
                    "tabletop config"
                )
        insertion = None
        assembly = None
        regrasp_contract = None
        twist_contract = None
        rewind_contract = None
        wrist_ft_config = None
        if arguments.insertion_probe:
            insertion_path = (
                Path(arguments.insertion_config).expanduser().resolve()
            )
            insertion = load_d38999_physical_insertion(insertion_path)
            insertion_inputs = verify_insertion_inputs(
                insertion, repository
            )
            if insertion_inputs["tabletop_pick"] != config_path:
                raise RuntimeError(
                    "insertion contract does not bind the active pick config"
                )
            assembly = load_d38999_assembly_baseline(
                insertion_inputs["assembly_baseline"]
            )
        if arguments.end_to_end_probe:
            # Reuse the strict loaders and exact parameter documents from the
            # independently proven prepared-engage pipeline.  Only scene setup
            # is skipped; all motion, sensing and acceptance values remain the
            # same in this continued physical episode.
            from d38999_nut_regrasp_smoke import _load_contract
            from kcg_connector.d38999_rewind_probe import (
                load_d38999_rewind_probe_contract,
            )
            from kcg_connector.d38999_twist_probe import (
                load_d38999_twist_probe_contract,
            )

            regrasp_path = (
                Path(arguments.regrasp_config).expanduser().resolve()
            )
            regrasp_contract, regrasp_inputs, _ = _load_contract(
                regrasp_path, repository
            )
            if regrasp_inputs["tabletop_pick"] != config_path:
                raise RuntimeError(
                    "end-to-end regrasp does not bind the active pick config"
                )
            twist_path = Path(arguments.twist_config).expanduser().resolve()
            twist_contract, twist_inputs = load_d38999_twist_probe_contract(
                twist_path, repository=repository
            )
            if twist_inputs["nut_regrasp_physx"] != regrasp_path:
                raise RuntimeError(
                    "end-to-end twist does not bind the active regrasp"
                )
            rewind_path = Path(arguments.rewind_config).expanduser().resolve()
            rewind_contract, rewind_inputs = (
                load_d38999_rewind_probe_contract(
                    rewind_path, repository=repository
                )
            )
            if (
                rewind_inputs["nut_regrasp_physx"] != regrasp_path
                or rewind_inputs["stage120_twist_contract"] != twist_path
            ):
                raise RuntimeError(
                    "end-to-end rewind inputs do not bind active contracts"
                )
        if arguments.wrist_ft_monitor:
            # This remains an observation-only branch.  The default E2E path
            # does not import, construct or sample the virtual wrench helper.
            from kcg_connector.virtual_wrist_ft_runtime import (
                VirtualWristFtMonitor,
                column_rotation_from_gf_matrix3d,
                load_virtual_wrist_ft_monitor_config,
                reaction_row_index,
                verify_virtual_wrist_ft_monitor_inputs,
            )

            wrist_ft_path = (
                Path(arguments.wrist_ft_config).expanduser().resolve()
            )
            wrist_ft_config = load_virtual_wrist_ft_monitor_config(
                wrist_ft_path
            )
            wrist_ft_inputs = verify_virtual_wrist_ft_monitor_inputs(
                wrist_ft_config, repository
            )
            if wrist_ft_inputs["robot_asset"] != robot_asset:
                raise RuntimeError(
                    "wrist FT monitor does not bind the active robot asset"
                )

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / rate_hz,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        stage = get_current_stage()
        metrics["d38999_authoring"] = author_d38999_tabletop_scene(
            stage,
            tabletop,
            d38999_asset,
            add_reference_to_stage=add_reference_to_stage,
            Gf=Gf,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
            physics_utils=physicsUtils,
        )
        add_reference_to_stage(
            str(robot_asset), pick.scene.robot_root_prim_path
        )
        tcp_prim = stage.GetPrimAtPath(pick.scene.grasp_tcp_prim_path)
        fixed_prim = stage.GetPrimAtPath(
            tabletop.asset.fixed_receptacle_prim_path
        )
        for path, prim in (
            (pick.scene.grasp_tcp_prim_path, tcp_prim),
            (tabletop.asset.fixed_receptacle_prim_path, fixed_prim),
        ):
            if not prim.IsValid():
                raise RuntimeError(f"required scene prim is missing: {path}")

        grip_material_path = "/World/D38999PickGripMaterial"
        grip_material = UsdShade.Material.Define(stage, grip_material_path)
        grip_api = UsdPhysics.MaterialAPI.Apply(grip_material.GetPrim())
        grip_api.CreateStaticFrictionAttr(pick.motion.grip_static_friction)
        grip_api.CreateDynamicFrictionAttr(pick.motion.grip_dynamic_friction)
        grip_api.CreateRestitutionAttr(pick.motion.grip_restitution)
        finger_collision_anchors = []
        plug_collision_prims = {"body": [], "nut": []}
        robot_root = pick.scene.robot_root_prim_path
        plug_root = tabletop.asset.loose_plug_prim_path
        body_root = tabletop.asset.body_prim_path
        nut_root = tabletop.asset.nut_prim_path
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            finger_anchor = bool(
                prim_path.startswith(robot_root + "/")
                and prim.GetName().endswith("_convex")
                and any(
                    link_name in prim_path
                    for link_name in ("/f1Link", "/f2Link", "/f3Link")
                )
            )
            plug_collision = bool(
                prim.HasAPI(UsdPhysics.CollisionAPI)
                and prim_path.startswith(plug_root + "/")
            )
            if finger_anchor or plug_collision:
                physicsUtils.add_physics_material_to_prim(
                    stage, prim, Sdf.Path(grip_material_path)
                )
                if finger_anchor:
                    finger_collision_anchors.append(prim_path)
                elif (
                    _d38999_loose_collider_group(
                        prim_path, body_root, nut_root
                    )
                    is not None
                ):
                    group = _d38999_loose_collider_group(
                        prim_path, body_root, nut_root
                    )
                    plug_collision_prims[group].append(prim_path)
                else:
                    raise RuntimeError(
                        "D38999 loose collider is outside body/nut: "
                        f"{prim_path}"
                    )
        if len(finger_collision_anchors) != 8:
            raise RuntimeError(
                "expected 8 finger collision anchors, found "
                f"{len(finger_collision_anchors)}"
            )
        expected_plug_collision_counts = {
            # One rear-body cylinder plus 20 mating-shell segments.
            "body": 21,
            # Twenty-four coupling-nut grip segments.
            "nut": 24,
        }
        plug_collision_counts = {
            key: len(value) for key, value in plug_collision_prims.items()
        }
        if plug_collision_counts != expected_plug_collision_counts:
            raise RuntimeError(
                "unexpected D38999 loose collider groups: "
                f"expected={expected_plug_collision_counts}, "
                f"found={plug_collision_counts}"
            )

        # Apply only the exact 500-pair segmented-proxy exception already used
        # by the validated q7 twist probe.  This happens before world.reset(),
        # so no collision topology is mutated during the physical episode.
        if insertion is not None:
            filter_contract = insertion.proxy_collision_filter
            filter_plan = build_proxy_collision_filter_plan(
                body_root,
                nut_root,
                tabletop.asset.fixed_receptacle_prim_path,
                body_mating_segment_count=(
                    filter_contract.expected_body_mating_segment_count
                ),
                nut_segment_count=(
                    filter_contract.expected_nut_segment_count
                ),
                fixed_entry_segment_count=(
                    filter_contract.expected_fixed_entry_segment_count
                ),
            )
            proxy_collision_filter = apply_proxy_collision_filter(
                stage, UsdPhysics, Sdf, filter_plan
            )
            if (
                proxy_collision_filter["pair_count"]
                != filter_contract.expected_filtered_pair_count
            ):
                raise RuntimeError(
                    "D38999 proxy collision filter count changed"
                )
        else:
            proxy_collision_filter = {
                "body_mating_segment_count": 0,
                "enabled": False,
                "fixed_entry_segment_count": 0,
                "mode": "none",
                "nut_segment_count": 0,
                "pair_count": 0,
            }
        metrics["proxy_collision_filter"] = proxy_collision_filter

        proxy_material_bindings = {}
        robot_prim = stage.GetPrimAtPath(robot_root)
        for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
            prim_path = str(prim.GetPath())
            if not (
                prim.IsInstanceProxy()
                and prim.HasAPI(UsdPhysics.CollisionAPI)
                and any(
                    link_name in prim_path
                    for link_name in ("/f1Link", "/f2Link", "/f3Link")
                )
            ):
                continue
            bound_material, _ = UsdShade.MaterialBindingAPI(
                prim
            ).ComputeBoundMaterial("physics")
            proxy_material_bindings[prim_path] = (
                str(bound_material.GetPath()) if bound_material else None
            )
        proxy_material_binding_ok = bool(
            len(proxy_material_bindings) == 8
            and all(
                material == grip_material_path
                for material in proxy_material_bindings.values()
            )
        )

        contact_report_body_count = 0
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            is_robot_body = bool(
                prim_path.startswith(robot_root + "/")
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            is_loose_body = prim_path in (
                tabletop.asset.body_prim_path,
                tabletop.asset.nut_prim_path,
            )
            if is_robot_body or is_loose_body:
                report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                report.CreateThresholdAttr().Set(0.0)
                if is_robot_body:
                    contact_report_body_count += 1
        if contact_report_body_count < 17:
            raise RuntimeError("robot contact reporting is incomplete")

        tooth_normalization_report = None
        tooth_color_report = None
        if arguments.nut_tooth_jitter_output:
            # Both optional overrides are session-layer opinions authored
            # before physics starts.  Normal E2E runs never execute this block.
            if arguments.nut_tooth_jitter_normalize_segment00_op:
                tooth_normalization_report = (
                    normalize_segment00_rotate_z_session(
                        stage, nut_root, UsdGeom
                    )
                )
            if arguments.nut_tooth_jitter_colorize:
                tooth_color_report = colorize_segments_session(
                    stage, nut_root, UsdGeom, Gf
                )

        if arguments.gui:
            from isaacsim.core.rendering_manager import ViewportManager

            lighting_root = "/World/D38999PickGuiLighting"
            UsdGeom.Xform.Define(stage, lighting_root)
            dome = UsdLux.DomeLight.Define(stage, lighting_root + "/Fill")
            dome.CreateIntensityAttr(tabletop.render.dome_light_intensity)
            dome.CreateColorAttr(
                Gf.Vec3f(*tabletop.render.dome_light_color_rgb)
            )
            key = UsdLux.DistantLight.Define(stage, lighting_root + "/Key")
            key.CreateIntensityAttr(tabletop.render.key_light_intensity)
            key.CreateAngleAttr(2.0)
            key.CreateColorAttr(Gf.Vec3f(*tabletop.render.key_light_color_rgb))
            UsdGeom.Xformable(key).AddRotateXYZOp().Set(
                Gf.Vec3f(*tabletop.render.key_light_rotation_degrees_xyz)
            )
            ViewportManager.set_camera_view(
                camera="/OmniverseKit_Persp",
                eye=np.asarray(CAMERA_EYE_M, dtype=np.float64),
                target=np.asarray(CAMERA_TARGET_M, dtype=np.float64),
            )
            simulation_app.update()

        robot = world.scene.add(
            SingleArticulation(
                prim_path=pick.scene.articulation_prim_path,
                name="d38999_tabletop_pick_handarm",
            )
        )
        body = world.scene.add(
            SingleRigidPrim(
                prim_path=tabletop.asset.body_prim_path,
                name="d38999_tabletop_pick_body",
            )
        )
        nut = world.scene.add(
            SingleRigidPrim(
                prim_path=tabletop.asset.nut_prim_path,
                name="d38999_tabletop_pick_nut",
            )
        )
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError(
                "robot articulation handles were not initialized"
            )
        if wrist_ft_config is not None:
            # Isaac indexes reaction rows one higher than the corresponding
            # metadata joint index.  Bind and cross-check that exact row once
            # before any tare or contact observation is accepted.
            metadata_joint_indices = dict(
                robot._articulation_view._metadata.joint_indices
            )
            wrist_ft_reaction_row = reaction_row_index(
                metadata_joint_indices, wrist_ft_config
            )
            selected_wrench = np.asarray(
                robot.get_measured_joint_forces(
                    joint_indices=np.asarray(
                        [wrist_ft_reaction_row], dtype=np.int32
                    )
                ),
                dtype=np.float64,
            )
            all_wrenches = np.asarray(
                robot.get_measured_joint_forces(), dtype=np.float64
            )
            if (
                selected_wrench.shape != (1, 6)
                or wrist_ft_reaction_row >= all_wrenches.shape[0]
                or not np.array_equal(
                    selected_wrench[0],
                    all_wrenches[wrist_ft_reaction_row],
                )
            ):
                raise RuntimeError(
                    "hand2arm reaction row failed joint-index-plus-one check"
                )
            hand2arm_joints = [
                prim
                for prim in stage.Traverse()
                if prim.GetName() == wrist_ft_config.measurement_joint
                and prim.IsA(UsdPhysics.FixedJoint)
            ]
            if len(hand2arm_joints) != 1:
                raise RuntimeError(
                    "expected exactly one fixed hand2arm measurement joint"
                )
            body1_targets = [
                str(path)
                for path in UsdPhysics.FixedJoint(
                    hand2arm_joints[0]
                ).GetBody1Rel().GetTargets()
            ]
            if (
                len(body1_targets) != 1
                or not body1_targets[0].endswith(
                    "/" + wrist_ft_config.raw_frame
                )
            ):
                raise RuntimeError(
                    "hand2arm child does not match configured raw frame"
                )
            wrist_ft_sensor_prim = stage.GetPrimAtPath(body1_targets[0])
            if not wrist_ft_sensor_prim.IsValid():
                raise RuntimeError("wrist FT sensor-frame prim is missing")
            wrist_ft_monitor = VirtualWristFtMonitor(
                wrist_ft_config,
                reaction_row=wrist_ft_reaction_row,
                task_origin_world=assembly.datums.fixed.position_world_m,
                task_z_axis_world=assembly.datums.fixed.axis_world,
            )
            metrics["virtual_wrist_ft_monitor"] = {
                "status": "INITIALIZED_MONITOR_ONLY",
                "measurement_joint": wrist_ft_config.measurement_joint,
                "metadata_joint_index": int(
                    metadata_joint_indices[
                        wrist_ft_config.measurement_joint
                    ]
                ),
                "reaction_row_index": wrist_ft_reaction_row,
                "joint_index_plus_one_verified": True,
                "raw_frame": wrist_ft_config.raw_frame,
                "task_frame": wrist_ft_config.task_frame_id,
                "modifies_e2e_pass_gate": False,
                "residual_v1_enabled": False,
            }
        if arguments.nut_tooth_jitter_output:
            tooth_probe = NutToothJitterProbe(
                stage=stage,
                nut_root=nut_root,
                parent_rigid=nut,
                output_directory=arguments.nut_tooth_jitter_output,
                Gf=Gf,
                Usd=Usd,
                UsdGeom=UsdGeom,
                PhysicsSchemaTools=PhysicsSchemaTools,
                normalization_report=tooth_normalization_report,
                color_report=tooth_color_report,
            )
        dof_names = tuple(robot.dof_names)
        if set(dof_names) != set(EXPECTED_DOF_NAMES) or len(dof_names) != 15:
            raise RuntimeError("unexpected articulation DOF layout")
        name_to_index = {name: index for index, name in enumerate(dof_names)}
        arm_indices = np.asarray(
            [name_to_index[name] for name in pick.robot.arm_joint_names],
            dtype=np.int32,
        )
        hand_indices = np.asarray(
            [
                name_to_index[name]
                for name in pick.robot.active_hand_joint_names
            ],
            dtype=np.int32,
        )
        sensor_indices = np.asarray(
            [name_to_index[name] for name in pick.sensing.torque_joint_names],
            dtype=np.int32,
        )
        controlled_indices = np.concatenate((arm_indices, hand_indices))

        zero_positions = np.zeros(robot.num_dof, dtype=np.float32)
        robot.set_joint_positions(zero_positions)
        robot.set_joint_velocities(zero_positions)
        controller = robot.get_articulation_controller()
        kps = np.zeros(robot.num_dof, dtype=np.float32)
        kds = np.zeros(robot.num_dof, dtype=np.float32)
        kps[arm_indices] = pick.robot.arm_stiffness
        kds[arm_indices] = pick.robot.arm_damping
        kps[hand_indices] = pick.robot.hand_stiffness
        kds[hand_indices] = pick.robot.hand_damping
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        world.get_physics_context().set_gravity(tabletop.physics.gravity_m_s2)

        home_arm = np.asarray(pick.robot.home_arm_rad, dtype=np.float64)
        open_hand = np.asarray(pick.robot.open_hand_rad, dtype=np.float64)
        grasp_hand = np.asarray(pick.motion.grasp_hand_rad, dtype=np.float64)
        pregrasp_arm = np.asarray(
            pick.motion.approach_segments[-1].target_arm_rad,
            dtype=np.float64,
        )
        grasp_arm = np.asarray(pick.motion.grasp_arm_rad, dtype=np.float64)
        closure_clearance_arm = np.asarray(
            pick.motion.closure_clearance_arm_rad, dtype=np.float64
        )
        current_arm_target = home_arm.copy()
        current_hand_target = np.zeros(4, dtype=np.float64)
        dof_properties = robot.dof_properties

        finite_throughout = True
        maximum_joint_limit_violation = 0.0
        maximum_joint_speed = 0.0
        maximum_arm_tracking_error = 0.0
        external_contact_records = {
            "table": 0,
            "fixture": 0,
            "fixed_endpoint": 0,
            "loose_plug_preclosure": 0,
            "loose_plug_allowed": 0,
            "loose_plug_unexpected_robot_link": 0,
        }
        external_contact_headers = {key: 0 for key in external_contact_records}
        grip_material_contact_records = 0
        phase_steps = {}
        global_step = 0
        phase = "initial_settle"

        fixed_initial_position, fixed_initial_orientation = _world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )

        def body_in_tcp_frame(body_position):
            tcp_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
                tcp_prim
            )
            point = tcp_matrix.GetInverse().Transform(
                Gf.Vec3d(*(float(value) for value in body_position))
            )
            return np.asarray(point, dtype=np.float64)

        def contact_snapshot():
            headers, contacts, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            result = {
                "finger_body_group_records": {
                    finger: {"body": 0, "nut": 0}
                    for finger in ("f1", "f2", "f3")
                },
                "grip_material_records": 0,
                "finger_loose_plug_records": 0,
                "plug_table_records": 0,
                "robot_loose_plug_records": 0,
                "unexpected_robot_link_records": 0,
            }
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                if _is_plug_table_contact(
                    paths,
                    tabletop.asset.loose_plug_prim_path,
                    tabletop.table.prim_path,
                ):
                    result["plug_table_records"] += int(
                        header.num_contact_data
                    )
                category = _classify_robot_external_contact(
                    paths,
                    robot_root,
                    tabletop.table.prim_path,
                    tabletop.fixed_endpoint.fixture_prim_path,
                    tabletop.asset.fixed_receptacle_prim_path,
                    tabletop.asset.loose_plug_prim_path,
                )
                if category != "loose_plug":
                    continue
                result["robot_loose_plug_records"] += int(
                    header.num_contact_data
                )
                is_finger_contact = _is_finger_plug_contact(
                    paths,
                    robot_root,
                    tabletop.asset.loose_plug_prim_path,
                )
                if not is_finger_contact:
                    result["unexpected_robot_link_records"] += int(
                        header.num_contact_data
                    )
                    continue
                result["finger_loose_plug_records"] += int(
                    header.num_contact_data
                )
                contact_group = _finger_loose_contact_group(
                    paths, robot_root, body_root, nut_root
                )
                if contact_group is None:
                    raise RuntimeError(
                        "finger/loose contact cannot be assigned to one "
                        f"finger and rigid-body group: {paths}"
                    )
                finger_name, loose_group = contact_group
                result["finger_body_group_records"][finger_name][
                    loose_group
                ] += int(header.num_contact_data)
                for index in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data,
                ):
                    contact = contacts[index]
                    materials = (
                        str(
                            PhysicsSchemaTools.intToSdfPath(contact.material0)
                        ),
                        str(
                            PhysicsSchemaTools.intToSdfPath(contact.material1)
                        ),
                    )
                    if materials == (
                        grip_material_path,
                        grip_material_path,
                    ):
                        result["grip_material_records"] += 1
            return result

        def observe_and_step(arm_target, hand_target, allow_loose_contact):
            nonlocal finite_throughout
            nonlocal maximum_joint_limit_violation
            nonlocal maximum_joint_speed
            nonlocal maximum_arm_tracking_error
            nonlocal grip_material_contact_records
            nonlocal global_step
            nonlocal wrist_ft_monitor_error
            target = np.concatenate((arm_target, hand_target)).astype(
                np.float32
            )
            robot.apply_action(
                ArticulationAction(
                    joint_positions=target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            global_step += 1
            phase_steps[phase] = phase_steps.get(phase, 0) + 1
            phase_step = phase_steps[phase]
            if (
                wrist_ft_monitor is not None
                and wrist_ft_monitor_error is None
            ):
                try:
                    raw_wrench = np.asarray(
                        robot.get_measured_joint_forces(
                            joint_indices=np.asarray(
                                [wrist_ft_monitor.reaction_row],
                                dtype=np.int32,
                            )
                        ),
                        dtype=np.float64,
                    )
                    if raw_wrench.shape != (1, 6):
                        raise RuntimeError(
                            "unexpected hand2arm reaction wrench shape: "
                            f"{raw_wrench.shape}"
                        )
                    sensor_matrix = UsdGeom.Xformable(
                        wrist_ft_sensor_prim
                    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    sensor_transform = Gf.Transform(sensor_matrix)
                    # Isaac Sim 6 exposes Gf.Rotation without GetMatrix().
                    # Matrix3d uses USD's row-vector convention, so the pure
                    # runtime adapter transposes it for NumPy column vectors.
                    sensor_rotation = column_rotation_from_gf_matrix3d(
                        np.asarray(
                            Gf.Matrix3d(sensor_transform.GetRotation()),
                            dtype=np.float64,
                        )
                    )
                    wrist_ft_monitor.observe(
                        raw_wrench[0],
                        global_step=global_step,
                        runtime_phase=phase,
                        sensor_position_world=np.asarray(
                            sensor_transform.GetTranslation(),
                            dtype=np.float64,
                        ),
                        sensor_rotation_world=sensor_rotation,
                    )
                except Exception as monitor_error:
                    # The opt-in monitor cannot weaken or replace any proven
                    # E2E gate.  Preserve its failure as explicit evidence and
                    # let the unchanged physical validation finish.
                    wrist_ft_monitor_error = (
                        f"{type(monitor_error).__name__}: {monitor_error}"
                    )
                    metrics["virtual_wrist_ft_monitor"].update(
                        {
                            "status": "MONITOR_FAILED",
                            "runtime_error": wrist_ft_monitor_error,
                            "safety_gate_claimed": False,
                        }
                    )
            positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            body_position, body_orientation = body.get_world_pose()
            nut_position, nut_orientation = nut.get_world_pose()
            body_linear = np.asarray(
                body.get_linear_velocity(), dtype=np.float64
            )
            body_angular = np.asarray(
                body.get_angular_velocity(), dtype=np.float64
            )
            sampled = np.concatenate(
                (
                    positions,
                    velocities,
                    np.asarray(body_position, dtype=np.float64),
                    np.asarray(body_orientation, dtype=np.float64),
                    np.asarray(nut_position, dtype=np.float64),
                    np.asarray(nut_orientation, dtype=np.float64),
                    body_linear,
                    body_angular,
                )
            )
            sample_is_finite = bool(np.all(np.isfinite(sampled)))
            finite_throughout = bool(finite_throughout and sample_is_finite)
            if not sample_is_finite:
                nonfinite_indices = np.flatnonzero(~np.isfinite(sampled))
                metrics["first_nonfinite_state"] = {
                    "global_step": global_step,
                    "phase": phase,
                    "phase_step": phase_step,
                    "sample_indices": [
                        int(value) for value in nonfinite_indices[:16]
                    ],
                    "sample_nonfinite_count": int(nonfinite_indices.size),
                }
                raise RuntimeError(
                    "non-finite physical state at "
                    f"global_step={global_step}, phase={phase}, "
                    f"phase_step={phase_step}, "
                    f"sample_nonfinite_count={nonfinite_indices.size}"
                )
            maximum_joint_speed = max(
                maximum_joint_speed,
                float(np.max(np.abs(velocities))),
            )
            maximum_arm_tracking_error = max(
                maximum_arm_tracking_error,
                float(np.max(np.abs(positions[arm_indices] - arm_target))),
            )
            for dof_index in range(robot.num_dof):
                if bool(dof_properties[dof_index]["hasLimits"]):
                    lower = float(dof_properties[dof_index]["lower"])
                    upper = float(dof_properties[dof_index]["upper"])
                    maximum_joint_limit_violation = max(
                        maximum_joint_limit_violation,
                        lower - float(positions[dof_index]),
                        float(positions[dof_index]) - upper,
                    )

            headers, contacts, friction_anchors = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                category = _classify_robot_external_contact(
                    paths,
                    robot_root,
                    tabletop.table.prim_path,
                    tabletop.fixed_endpoint.fixture_prim_path,
                    tabletop.asset.fixed_receptacle_prim_path,
                    tabletop.asset.loose_plug_prim_path,
                )
                if category is None:
                    continue
                key = category
                finger_contact = False
                if category == "loose_plug":
                    if not allow_loose_contact:
                        key += "_preclosure"
                    elif _is_finger_plug_contact(
                        paths,
                        robot_root,
                        tabletop.asset.loose_plug_prim_path,
                    ):
                        key += "_allowed"
                        finger_contact = True
                    else:
                        key += "_unexpected_robot_link"
                external_contact_headers[key] += 1
                external_contact_records[key] += int(header.num_contact_data)
                if not (category == "loose_plug" and finger_contact):
                    metrics["first_forbidden_contact"] = {
                        "category": key,
                        "contact_record_count": int(header.num_contact_data),
                        "global_step": global_step,
                        "paths": list(paths),
                        "phase": phase,
                        "phase_step": phase_step,
                    }
                    raise RuntimeError(
                        "forbidden contact at "
                        f"global_step={global_step}, phase={phase}, "
                        f"phase_step={phase_step}, category={key}, "
                        f"paths={paths}"
                    )
                if key != "loose_plug_allowed":
                    continue
                for index in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data,
                ):
                    contact = contacts[index]
                    materials = (
                        str(
                            PhysicsSchemaTools.intToSdfPath(contact.material0)
                        ),
                        str(
                            PhysicsSchemaTools.intToSdfPath(contact.material1)
                        ),
                    )
                    if materials == (
                        grip_material_path,
                        grip_material_path,
                    ):
                        grip_material_contact_records += 1
            if (
                tooth_probe is not None
                and str(phase).startswith("end_to_end_")
            ):
                # The requested window starts only after physical insertion;
                # default E2E has no probe calls or additional contact reads.
                tooth_probe.sample(
                    global_step=global_step,
                    phase=phase,
                    phase_step=phase_step,
                    contact_report=(headers, contacts, friction_anchors),
                )
            return positions, velocities

        phase = "initial_settle"
        for _ in range(tabletop.physics.settle_steps):
            observe_and_step(current_arm_target, current_hand_target, False)
        if wrist_ft_monitor is not None and wrist_ft_monitor_error is None:
            try:
                wrist_ft_monitor.capture_home_tare()
            except Exception as monitor_error:
                wrist_ft_monitor_error = (
                    f"{type(monitor_error).__name__}: {monitor_error}"
                )
                metrics["virtual_wrist_ft_monitor"].update(
                    {
                        "status": "HOME_TARE_FAILED",
                        "runtime_error": wrist_ft_monitor_error,
                        "safety_gate_claimed": False,
                    }
                )
        settled_body_position, _ = body.get_world_pose()
        settled_nut_position, _ = nut.get_world_pose()
        settled_bottom = min(
            float(settled_body_position[2])
            + tabletop.loose_endpoint.body_bottom_offset_m,
            float(settled_nut_position[2])
            + tabletop.loose_endpoint.nut_bottom_offset_m,
        )
        settled_on_table = bool(
            -tabletop.physics.maximum_transient_table_penetration_m
            <= settled_bottom - tabletop.table.top_z_m
            <= tabletop.physics.maximum_final_surface_gap_m
        )

        if arguments.pose_preflight == "masked-rgbd":
            # Capture after the physical initial settle and before the first
            # intentional hand/arm motion.  The helper reuses this exact World
            # and episode: no report is read, no second World is made, and no
            # reset or endpoint pose write is allowed.  Its estimated XY is an
            # observation milestone only; the validated fixed trajectory still
            # uses simulation-ground-truth pose and orientation for control.
            preflight = capture_d38999_rgbd_runtime(
                bindings={
                    "Camera": Camera,
                    "Gf": Gf,
                    "Image": Image,
                    "Usd": Usd,
                    "UsdGeom": UsdGeom,
                    "UsdLux": UsdLux,
                    "add_labels": add_labels,
                    "get_labels": get_labels,
                    "rep": rep,
                },
                simulation_app=simulation_app,
                world=world,
                stage=stage,
                tabletop=tabletop,
                rgbd=rgbd,
                loose_prim=stage.GetPrimAtPath(
                    tabletop.asset.loose_plug_prim_path
                ),
                fixed_prim=fixed_prim,
                body=body,
                output_dir=None,
            )
            metrics["pose_preflight"] = {
                **preflight.metrics,
                "control_pose_provider": "sim_ground_truth",
                "foundation_pose": False,
                "masked_rgbd_xy_used_for_control": False,
                "truth_orientation_used": True,
            }
            pose_preflight_gate = preflight.passed
            if preflight.passed is not True:
                raise RuntimeError(
                    "masked RGB-D pose preflight failed before intentional "
                    "robot motion"
                )

        phase = "home_hand_open"
        closed_home_hand = current_hand_target.copy()
        hand_open_steps = round(
            pick.motion.hand_open_duration_s
            * rate_hz
            / home_hand_open_speedup
        )
        for index in range(hand_open_steps):
            blend = minimum_jerk_blend(
                float(index + 1) / float(hand_open_steps)
            )
            current_hand_target = closed_home_hand + blend * (
                open_hand - closed_home_hand
            )
            observe_and_step(current_arm_target, current_hand_target, False)
        current_hand_target = open_hand.copy()

        for segment in pick.motion.approach_segments:
            phase = segment.name
            start_arm = current_arm_target.copy()
            final_arm = np.asarray(segment.target_arm_rad, dtype=np.float64)
            segment_steps = round(segment.duration_s * rate_hz)
            for index in range(segment_steps):
                current_arm_target = np.asarray(
                    interpolate_arm(
                        tuple(float(value) for value in start_arm),
                        tuple(float(value) for value in final_arm),
                        float(index + 1) / float(segment_steps),
                    ),
                    dtype=np.float64,
                )
                observe_and_step(
                    current_arm_target, current_hand_target, False
                )
            current_arm_target = final_arm.copy()

        phase = "pregrasp_hold"
        hold_steps = round(pick.motion.pregrasp_hold_duration_s * rate_hz)
        for _ in range(hold_steps):
            observe_and_step(current_arm_target, current_hand_target, False)

        phase = "open_hand_descent"
        descent_start = current_arm_target.copy()
        descent_steps = round(pick.motion.descent_duration_s * rate_hz)
        for index in range(descent_steps):
            current_arm_target = np.asarray(
                interpolate_arm(
                    tuple(float(value) for value in descent_start),
                    tuple(float(value) for value in closure_clearance_arm),
                    float(index + 1) / float(descent_steps),
                ),
                dtype=np.float64,
            )
            observe_and_step(current_arm_target, current_hand_target, False)
        current_arm_target = closure_clearance_arm.copy()

        kps[hand_indices] = pick.motion.grip_hand_stiffness
        kds[hand_indices] = pick.motion.grip_hand_damping
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        phase = "open_grasp_tare"
        tare_effort_samples = []
        tare_steps = round(pick.motion.open_tare_duration_s * rate_hz)
        for _ in range(tare_steps):
            observe_and_step(current_arm_target, current_hand_target, False)
            tare_effort_samples.append(
                np.asarray(
                    robot.get_measured_joint_efforts(
                        joint_indices=sensor_indices
                    ),
                    dtype=np.float64,
                )
            )
        tare_efforts = np.mean(np.stack(tare_effort_samples), axis=0)
        maximum_post_tare_absolute_delta_by_channel = np.zeros(
            len(sensor_indices), dtype=np.float64
        )

        def sample_post_tare_efforts():
            nonlocal finite_throughout
            measured = np.asarray(
                robot.get_measured_joint_efforts(joint_indices=sensor_indices),
                dtype=np.float64,
            )
            delta = measured - tare_efforts
            finite_throughout = bool(
                finite_throughout
                and np.all(np.isfinite(measured))
                and np.all(np.isfinite(delta))
            )
            if not (
                np.all(np.isfinite(measured)) and np.all(np.isfinite(delta))
            ):
                metrics["first_nonfinite_effort"] = {
                    "global_step": global_step,
                    "phase": phase,
                    "phase_step": phase_steps[phase],
                }
                raise RuntimeError(
                    "non-finite measured effort at "
                    f"global_step={global_step}, phase={phase}, "
                    f"phase_step={phase_steps[phase]}"
                )
            np.maximum(
                maximum_post_tare_absolute_delta_by_channel,
                np.abs(delta),
                out=maximum_post_tare_absolute_delta_by_channel,
            )
            if np.any(
                np.abs(delta) > pick.sensing.maximum_absolute_torque_delta_nm
            ):
                metrics["first_torque_safety_violation"] = {
                    "delta_nm": {
                        name: float(value)
                        for name, value in zip(
                            pick.sensing.torque_joint_names, delta
                        )
                    },
                    "global_step": global_step,
                    "limit_nm": (
                        pick.sensing.maximum_absolute_torque_delta_nm
                    ),
                    "phase": phase,
                    "phase_step": phase_steps[phase],
                }
                raise RuntimeError(
                    "finger-base torque safety violation at "
                    f"global_step={global_step}, phase={phase}, "
                    f"phase_step={phase_steps[phase]}"
                )
            return measured

        closure_tcp_position, closure_tcp_orientation = _world_pose(
            Gf, Usd, UsdGeom, tcp_prim
        )
        closure_tcp_position_error = float(
            np.linalg.norm(
                np.asarray(closure_tcp_position, dtype=np.float64)
                - np.asarray(
                    pick.motion.closure_clearance_tcp_position_m,
                    dtype=np.float64,
                )
            )
        )
        closure_tcp_axis = _quaternion_world_z_axis(closure_tcp_orientation)
        closure_tcp_axis_error = _axis_error_radians(
            closure_tcp_axis, pick.motion.grasp_tcp_down_axis_world
        )
        positions_at_closure = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        closure_endpoint_arm_error = float(
            np.max(
                np.abs(
                    positions_at_closure[arm_indices] - closure_clearance_arm
                )
            )
        )

        phase = "physical_hand_closure"
        closure_start = current_hand_target.copy()
        closure_steps = round(pick.motion.closure_duration_s * rate_hz)
        for index in range(closure_steps):
            blend = minimum_jerk_blend(float(index + 1) / float(closure_steps))
            current_hand_target = closure_start + blend * (
                grasp_hand - closure_start
            )
            observe_and_step(current_arm_target, current_hand_target, True)
            sample_post_tare_efforts()
        current_hand_target = grasp_hand.copy()

        phase = "closed_hand_seating"
        seating_start = current_arm_target.copy()
        seating_steps = round(pick.motion.closed_seating_duration_s * rate_hz)
        for index in range(seating_steps):
            current_arm_target = np.asarray(
                interpolate_arm(
                    tuple(float(value) for value in seating_start),
                    tuple(float(value) for value in grasp_arm),
                    float(index + 1) / float(seating_steps),
                ),
                dtype=np.float64,
            )
            observe_and_step(current_arm_target, current_hand_target, True)
            sample_post_tare_efforts()
        current_arm_target = grasp_arm.copy()

        grasp_tcp_position, grasp_tcp_orientation = _world_pose(
            Gf, Usd, UsdGeom, tcp_prim
        )
        grasp_tcp_position_error = float(
            np.linalg.norm(
                np.asarray(grasp_tcp_position, dtype=np.float64)
                - np.asarray(
                    pick.motion.grasp_tcp_position_m, dtype=np.float64
                )
            )
        )
        grasp_tcp_axis = _quaternion_world_z_axis(grasp_tcp_orientation)
        grasp_tcp_axis_error = _axis_error_radians(
            grasp_tcp_axis, pick.motion.grasp_tcp_down_axis_world
        )
        positions_at_grasp = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        grasp_endpoint_arm_error = float(
            np.max(np.abs(positions_at_grasp[arm_indices] - grasp_arm))
        )

        phase = "physical_grip_preload"
        preload_effort_samples = []
        preload_steps = round(pick.motion.preload_duration_s * rate_hz)
        for _ in range(preload_steps):
            observe_and_step(current_arm_target, current_hand_target, True)
            preload_effort_samples.append(sample_post_tare_efforts())
        contact_efforts = np.mean(np.stack(preload_effort_samples), axis=0)
        torque_deltas = contact_efforts - tare_efforts
        loaded_channels = int(
            np.count_nonzero(
                np.abs(torque_deltas)
                >= pick.sensing.loaded_torque_threshold_nm
            )
        )
        maximum_absolute_torque_delta = float(np.max(np.abs(torque_deltas)))
        postclosure_body_position, _ = body.get_world_pose()
        postclosure_nut_position, _ = nut.get_world_pose()
        postclosure_body_in_tcp = body_in_tcp_frame(postclosure_body_position)
        postclosure_body_nut_separation = float(
            np.linalg.norm(
                postclosure_nut_position - postclosure_body_position
            )
        )
        postclosure_contacts = contact_snapshot()

        phase = "physical_grip_lift"
        lift_start = current_arm_target.copy()
        lift_steps = round(pick.motion.lift_duration_s * rate_hz)
        for index in range(lift_steps):
            current_arm_target = np.asarray(
                interpolate_arm(
                    tuple(float(value) for value in lift_start),
                    tuple(float(value) for value in pregrasp_arm),
                    float(index + 1) / float(lift_steps),
                ),
                dtype=np.float64,
            )
            observe_and_step(current_arm_target, current_hand_target, True)
            sample_post_tare_efforts()
        current_arm_target = pregrasp_arm.copy()

        phase = "unsupported_final_hold"
        hold_start_body_position, _ = body.get_world_pose()
        maximum_final_hold_displacement = 0.0
        final_effort_samples = []
        final_hold_steps = round(pick.motion.final_hold_duration_s * rate_hz)
        tail_window_steps = min(120, final_hold_steps)
        tail_solver_velocity_samples = []
        tail_pose_difference_velocity_samples = []
        tail_body_pose_difference_linear_speeds = []
        tail_body_pose_difference_angular_speeds = []
        tail_nut_pose_difference_angular_speeds = []
        tail_relative_pose_difference_angular_speeds = []
        tail_body_solver_angular_speeds = []
        tail_nut_solver_angular_speeds = []
        tail_start_body_orientation = None
        tail_start_nut_orientation = None
        tail_start_relative_orientation = None
        tail_end_body_orientation = None
        tail_end_nut_orientation = None
        tail_end_relative_orientation = None
        previous_tail_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        previous_tail_body_position, previous_tail_body_orientation = (
            body.get_world_pose()
        )
        previous_tail_body_position = np.asarray(
            previous_tail_body_position, dtype=np.float64
        )
        previous_tail_body_orientation = np.asarray(
            previous_tail_body_orientation, dtype=np.float64
        )
        _, previous_tail_nut_orientation = nut.get_world_pose()
        previous_tail_nut_orientation = np.asarray(
            previous_tail_nut_orientation, dtype=np.float64
        )
        previous_tail_relative_orientation = _relative_array_quaternion(
            previous_tail_body_orientation, previous_tail_nut_orientation
        )
        effort_sample_steps = round(
            pick.motion.effort_sample_duration_s * rate_hz
        )
        for index in range(final_hold_steps):
            observe_and_step(current_arm_target, current_hand_target, True)
            measured_final_effort = sample_post_tare_efforts()
            current_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            current_solver_velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            current_body_position, current_body_orientation = (
                body.get_world_pose()
            )
            _, current_nut_orientation = nut.get_world_pose()
            current_body_position = np.asarray(
                current_body_position, dtype=np.float64
            )
            current_body_orientation = np.asarray(
                current_body_orientation, dtype=np.float64
            )
            current_nut_orientation = np.asarray(
                current_nut_orientation, dtype=np.float64
            )
            current_relative_orientation = _relative_array_quaternion(
                current_body_orientation, current_nut_orientation
            )
            if index >= final_hold_steps - tail_window_steps:
                if tail_start_body_orientation is None:
                    tail_start_body_orientation = (
                        previous_tail_body_orientation.copy()
                    )
                    tail_start_nut_orientation = (
                        previous_tail_nut_orientation.copy()
                    )
                    tail_start_relative_orientation = (
                        previous_tail_relative_orientation.copy()
                    )
                tail_solver_velocity_samples.append(
                    current_solver_velocities.copy()
                )
                tail_pose_difference_velocity_samples.append(
                    (current_positions - previous_tail_positions) * rate_hz
                )
                tail_body_pose_difference_linear_speeds.append(
                    float(
                        np.linalg.norm(
                            current_body_position - previous_tail_body_position
                        )
                        * rate_hz
                    )
                )
                tail_body_pose_difference_angular_speeds.append(
                    _array_quaternion_error_radians(
                        previous_tail_body_orientation,
                        current_body_orientation,
                    )
                    * rate_hz
                )
                tail_nut_pose_difference_angular_speeds.append(
                    _array_quaternion_error_radians(
                        previous_tail_nut_orientation,
                        current_nut_orientation,
                    )
                    * rate_hz
                )
                tail_relative_pose_difference_angular_speeds.append(
                    _array_quaternion_error_radians(
                        previous_tail_relative_orientation,
                        current_relative_orientation,
                    )
                    * rate_hz
                )
                tail_body_solver_angular_speeds.append(
                    float(np.linalg.norm(body.get_angular_velocity()))
                )
                tail_nut_solver_angular_speeds.append(
                    float(np.linalg.norm(nut.get_angular_velocity()))
                )
                tail_end_body_orientation = current_body_orientation.copy()
                tail_end_nut_orientation = current_nut_orientation.copy()
                tail_end_relative_orientation = (
                    current_relative_orientation.copy()
                )
            previous_tail_positions = current_positions
            previous_tail_body_position = current_body_position
            previous_tail_body_orientation = current_body_orientation
            previous_tail_nut_orientation = current_nut_orientation
            previous_tail_relative_orientation = current_relative_orientation
            maximum_final_hold_displacement = max(
                maximum_final_hold_displacement,
                float(
                    np.linalg.norm(
                        current_body_position - hold_start_body_position
                    )
                ),
            )
            if index >= final_hold_steps - effort_sample_steps:
                final_effort_samples.append(measured_final_effort.copy())

        if wrist_ft_monitor is not None and wrist_ft_monitor_error is None:
            try:
                wrist_ft_monitor.capture_payload_baseline()
            except Exception as monitor_error:
                wrist_ft_monitor_error = (
                    f"{type(monitor_error).__name__}: {monitor_error}"
                )
                metrics["virtual_wrist_ft_monitor"].update(
                    {
                        "status": "PAYLOAD_BASELINE_FAILED",
                        "runtime_error": wrist_ft_monitor_error,
                        "safety_gate_claimed": False,
                    }
                )

        final_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        final_velocities = np.asarray(
            robot.get_joint_velocities(), dtype=np.float64
        )
        tail_solver_velocities = np.abs(np.stack(tail_solver_velocity_samples))
        tail_pose_difference_velocities = np.abs(
            np.stack(tail_pose_difference_velocity_samples)
        )
        maximum_final_observable_joint_speed = float(
            np.max(tail_pose_difference_velocities)
        )
        maximum_final_post_solver_joint_speed = float(
            np.max(tail_solver_velocities)
        )
        final_body_position, final_body_orientation = body.get_world_pose()
        final_nut_position, final_nut_orientation = nut.get_world_pose()
        final_body_post_solver_linear_speed = float(
            np.linalg.norm(body.get_linear_velocity())
        )
        final_body_post_solver_angular_speed = float(
            np.linalg.norm(body.get_angular_velocity())
        )
        final_body_observable_linear_speed = float(
            np.max(tail_body_pose_difference_linear_speeds)
        )
        final_body_observable_angular_speed = float(
            np.max(tail_body_pose_difference_angular_speeds)
        )
        tail_body_observable_angular_statistics = _speed_statistics(
            tail_body_pose_difference_angular_speeds
        )
        tail_nut_observable_angular_statistics = _speed_statistics(
            tail_nut_pose_difference_angular_speeds
        )
        tail_relative_observable_angular_statistics = _speed_statistics(
            tail_relative_pose_difference_angular_speeds
        )
        tail_body_solver_angular_statistics = _speed_statistics(
            tail_body_solver_angular_speeds
        )
        tail_nut_solver_angular_statistics = _speed_statistics(
            tail_nut_solver_angular_speeds
        )
        tail_net_rotation = {
            "body_rad": _array_quaternion_error_radians(
                tail_start_body_orientation, tail_end_body_orientation
            ),
            "nut_rad": _array_quaternion_error_radians(
                tail_start_nut_orientation, tail_end_nut_orientation
            ),
            "nut_relative_to_body_rad": _array_quaternion_error_radians(
                tail_start_relative_orientation,
                tail_end_relative_orientation,
            ),
        }
        fixed_final_position, fixed_final_orientation = _world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )
        final_contacts = contact_snapshot()
        final_body_in_tcp = body_in_tcp_frame(final_body_position)
        body_tcp_slip = float(
            np.linalg.norm(final_body_in_tcp - postclosure_body_in_tcp)
        )
        body_nut_separation_change = abs(
            float(
                np.linalg.norm(final_nut_position - final_body_position)
                - postclosure_body_nut_separation
            )
        )
        body_lift = float(final_body_position[2] - settled_body_position[2])
        final_bottom = min(
            float(final_body_position[2])
            + tabletop.loose_endpoint.body_bottom_offset_m,
            float(final_nut_position[2])
            + tabletop.loose_endpoint.nut_bottom_offset_m,
        )
        final_bottom_clearance = final_bottom - tabletop.table.top_z_m
        final_arm_tracking_error = float(
            np.max(np.abs(final_positions[arm_indices] - current_arm_target))
        )
        fixed_translation_drift = float(
            np.linalg.norm(
                np.asarray(fixed_final_position, dtype=np.float64)
                - np.asarray(fixed_initial_position, dtype=np.float64)
            )
        )
        fixed_rotation_drift = _gf_quaternion_error_radians(
            fixed_initial_orientation, fixed_final_orientation
        )
        final_efforts = np.mean(np.stack(final_effort_samples), axis=0)
        final_torque_deltas = final_efforts - tare_efforts
        final_loaded_channels = int(
            np.count_nonzero(
                np.abs(final_torque_deltas)
                >= pick.sensing.loaded_torque_threshold_nm
            )
        )
        final_maximum_absolute_torque_delta = float(
            np.max(np.abs(final_torque_deltas))
        )
        maximum_post_tare_absolute_torque_delta = float(
            np.max(maximum_post_tare_absolute_delta_by_channel)
        )
        finite_final = bool(
            np.all(np.isfinite(final_positions))
            and np.all(np.isfinite(final_velocities))
            and np.all(np.isfinite(final_body_position))
            and np.all(np.isfinite(final_body_orientation))
            and np.all(np.isfinite(final_nut_position))
            and np.all(np.isfinite(final_nut_orientation))
            and np.all(np.isfinite(tare_efforts))
            and np.all(np.isfinite(contact_efforts))
            and np.all(np.isfinite(final_efforts))
            and _gf_quaternion_finite(fixed_final_orientation)
        )
        tail_diagnostics_finite = bool(
            tail_window_steps == 120
            and len(tail_solver_velocity_samples) == tail_window_steps
            and len(tail_pose_difference_velocity_samples) == tail_window_steps
            and len(tail_body_pose_difference_linear_speeds)
            == tail_window_steps
            and len(tail_body_pose_difference_angular_speeds)
            == tail_window_steps
            and len(tail_nut_pose_difference_angular_speeds)
            == tail_window_steps
            and len(tail_relative_pose_difference_angular_speeds)
            == tail_window_steps
            and len(tail_body_solver_angular_speeds) == tail_window_steps
            and len(tail_nut_solver_angular_speeds) == tail_window_steps
            and np.all(np.isfinite(tail_solver_velocities))
            and np.all(np.isfinite(tail_pose_difference_velocities))
            and np.all(np.isfinite(tail_body_pose_difference_linear_speeds))
            and np.all(np.isfinite(tail_body_pose_difference_angular_speeds))
            and np.all(np.isfinite(tail_nut_pose_difference_angular_speeds))
            and np.all(
                np.isfinite(tail_relative_pose_difference_angular_speeds)
            )
            and np.all(np.isfinite(tail_body_solver_angular_speeds))
            and np.all(np.isfinite(tail_nut_solver_angular_speeds))
            and all(
                math.isfinite(value)
                for value in tail_net_rotation.values()
            )
        )

        acceptance = pick.acceptance
        zero_forbidden_contacts = bool(
            external_contact_records["table"] == 0
            and external_contact_records["fixture"] == 0
            and external_contact_records["fixed_endpoint"] == 0
            and external_contact_records["loose_plug_preclosure"] == 0
            and external_contact_records["loose_plug_unexpected_robot_link"]
            == 0
        )
        force_gate = bool(
            loaded_channels >= pick.sensing.minimum_loaded_channels
            and maximum_absolute_torque_delta
            <= pick.sensing.maximum_absolute_torque_delta_nm
            and final_loaded_channels >= pick.sensing.minimum_loaded_channels
            and final_maximum_absolute_torque_delta
            <= pick.sensing.maximum_absolute_torque_delta_nm
            and maximum_post_tare_absolute_torque_delta
            <= pick.sensing.maximum_absolute_torque_delta_nm
        )
        postclosure_all_fingers_body_contact = (
            _all_fingers_have_body_contact(postclosure_contacts)
        )
        final_all_fingers_body_contact = _all_fingers_have_body_contact(
            final_contacts
        )
        body_contact_gate = bool(
            postclosure_all_fingers_body_contact
            and final_all_fingers_body_contact
        )
        physical_contact_gate = bool(
            proxy_material_binding_ok
            and grip_material_contact_records > 0
            and postclosure_contacts["grip_material_records"] > 0
            and final_contacts["grip_material_records"] > 0
            and final_contacts["finger_loose_plug_records"] > 0
            and final_contacts["unexpected_robot_link_records"] == 0
            and body_contact_gate
        )
        final_unsupported = bool(
            final_contacts["plug_table_records"] == 0
            and body_lift >= acceptance.minimum_body_lift_m
            and final_bottom_clearance
            >= acceptance.minimum_final_bottom_clearance_m
        )
        passed = bool(
            finite_throughout
            and finite_final
            and tail_diagnostics_finite
            and settled_on_table
            and pose_preflight_gate
            and maximum_joint_limit_violation
            <= acceptance.maximum_joint_limit_violation_rad
            and maximum_joint_speed
            <= acceptance.maximum_observed_joint_speed_rad_s
            and maximum_arm_tracking_error
            <= acceptance.maximum_arm_tracking_error_rad
            and closure_endpoint_arm_error
            <= acceptance.maximum_endpoint_arm_tracking_error_rad
            and grasp_endpoint_arm_error
            <= acceptance.maximum_endpoint_arm_tracking_error_rad
            and final_arm_tracking_error
            <= acceptance.maximum_endpoint_arm_tracking_error_rad
            and maximum_final_observable_joint_speed
            <= acceptance.maximum_final_observable_joint_speed_rad_s
            and maximum_final_post_solver_joint_speed
            <= acceptance.maximum_final_post_solver_joint_speed_rad_s
            and closure_tcp_position_error
            <= acceptance.maximum_grasp_tcp_position_error_m
            and closure_tcp_axis_error
            <= acceptance.maximum_grasp_tcp_axis_error_rad
            and grasp_tcp_position_error
            <= acceptance.maximum_grasp_tcp_position_error_m
            and grasp_tcp_axis_error
            <= acceptance.maximum_grasp_tcp_axis_error_rad
            and force_gate
            and physical_contact_gate
            and zero_forbidden_contacts
            and final_unsupported
            and body_tcp_slip <= acceptance.maximum_body_tcp_slip_m
            and body_nut_separation_change
            <= acceptance.maximum_body_nut_separation_change_m
            and maximum_final_hold_displacement
            <= acceptance.maximum_final_hold_displacement_m
            and final_body_observable_linear_speed
            <= acceptance.maximum_final_body_observable_linear_speed_m_s
            and final_body_observable_angular_speed
            <= acceptance.maximum_final_body_observable_angular_speed_rad_s
            and final_body_post_solver_linear_speed
            <= acceptance.maximum_final_body_post_solver_linear_speed_m_s
            and final_body_post_solver_angular_speed
            <= acceptance.maximum_final_body_post_solver_angular_speed_rad_s
            and fixed_translation_drift
            <= acceptance.maximum_fixed_translation_drift_m
            and fixed_rotation_drift
            <= acceptance.maximum_fixed_rotation_drift_rad
        )
        metrics.update(
            {
                "body_lift_m": body_lift,
                "body_nut_separation_change_m": (body_nut_separation_change),
                "body_tcp_slip_m": body_tcp_slip,
                "body_contact_gate": body_contact_gate,
                "closure_endpoint_arm_error_rad": (closure_endpoint_arm_error),
                "closure_tcp_axis_error_rad": closure_tcp_axis_error,
                "closure_tcp_position_error_m": closure_tcp_position_error,
                "contact_torque_deltas_nm": {
                    name: float(value)
                    for name, value in zip(
                        pick.sensing.torque_joint_names, torque_deltas
                    )
                },
                "external_contact_headers": external_contact_headers,
                "external_contact_records": external_contact_records,
                "final_arm_tracking_error_rad": (final_arm_tracking_error),
                "final_body_observable_angular_speed_rad_s": (
                    final_body_observable_angular_speed
                ),
                "final_body_observable_linear_speed_m_s": (
                    final_body_observable_linear_speed
                ),
                "final_body_post_solver_angular_speed_rad_s": (
                    final_body_post_solver_angular_speed
                ),
                "final_body_post_solver_linear_speed_m_s": (
                    final_body_post_solver_linear_speed
                ),
                "final_bottom_clearance_m": final_bottom_clearance,
                "final_contacts": final_contacts,
                "final_all_fingers_body_contact": (
                    final_all_fingers_body_contact
                ),
                "final_hold_displacement_m": (maximum_final_hold_displacement),
                "final_loaded_torque_channels": final_loaded_channels,
                "final_maximum_absolute_torque_delta_nm": (
                    final_maximum_absolute_torque_delta
                ),
                "final_observable_joint_speed_peak_rad_s": (
                    maximum_final_observable_joint_speed
                ),
                "final_post_solver_joint_speed_peak_rad_s": (
                    maximum_final_post_solver_joint_speed
                ),
                "final_tail_window_steps": tail_window_steps,
                "final_tail_diagnostics_finite": (tail_diagnostics_finite),
                "final_tail_net_rotation_rad": tail_net_rotation,
                "final_tail_observable_angular_speed_rad_s": {
                    "body": tail_body_observable_angular_statistics,
                    "nut": tail_nut_observable_angular_statistics,
                    "nut_relative_to_body": (
                        tail_relative_observable_angular_statistics
                    ),
                },
                "final_tail_solver_angular_speed_rad_s": {
                    "body": tail_body_solver_angular_statistics,
                    "nut": tail_nut_solver_angular_statistics,
                },
                "final_torque_deltas_nm": {
                    name: float(value)
                    for name, value in zip(
                        pick.sensing.torque_joint_names,
                        final_torque_deltas,
                    )
                },
                "final_unsupported": final_unsupported,
                "finite_final": finite_final,
                "finite_throughout": finite_throughout,
                "fixed_rotation_drift_rad": fixed_rotation_drift,
                "fixed_translation_drift_m": fixed_translation_drift,
                "force_gate": force_gate,
                "grasp_endpoint_arm_error_rad": (grasp_endpoint_arm_error),
                "grasp_material_contact_records_total": (
                    grip_material_contact_records
                ),
                "grasp_tcp_axis_error_rad": grasp_tcp_axis_error,
                "grasp_tcp_position_error_m": grasp_tcp_position_error,
                "joint_limit_violation_rad": max(
                    0.0, maximum_joint_limit_violation
                ),
                "loaded_torque_channels": loaded_channels,
                "maximum_absolute_torque_delta_nm": (
                    maximum_absolute_torque_delta
                ),
                "maximum_arm_tracking_error_rad": (maximum_arm_tracking_error),
                "maximum_joint_speed_rad_s": maximum_joint_speed,
                "maximum_post_tare_absolute_delta_by_channel_nm": {
                    name: float(value)
                    for name, value in zip(
                        pick.sensing.torque_joint_names,
                        maximum_post_tare_absolute_delta_by_channel,
                    )
                },
                "maximum_post_tare_absolute_torque_delta_nm": (
                    maximum_post_tare_absolute_torque_delta
                ),
                "operational_torque_target_nm": (
                    pick.sensing.operational_torque_target_nm
                ),
                "operational_torque_target_exceeded": bool(
                    maximum_post_tare_absolute_torque_delta
                    > pick.sensing.operational_torque_target_nm
                ),
                "nut_contacts_allowed": True,
                "phase_steps": phase_steps,
                "physical_contact_gate": physical_contact_gate,
                "postclosure_all_fingers_body_contact": (
                    postclosure_all_fingers_body_contact
                ),
                "postclosure_contacts": postclosure_contacts,
                "pose_preflight_gate": pose_preflight_gate,
                "proxy_material_binding_ok": proxy_material_binding_ok,
                "settled_on_table": settled_on_table,
                "zero_forbidden_contacts": zero_forbidden_contacts,
                "passed": passed,
            }
        )
        if arguments.insertion_probe:
            # Continue from the physical pick state in this exact World.  The
            # complete measured TCP-to-body transform is the hand-off contract;
            # no object pose is authored, attached or driven here.
            insertion_contact_records = {
                "loose_fixed": 0,
                "loose_fixture": 0,
                "loose_table": 0,
            }

            def gf_quaternion_tuple(value):
                imaginary = value.GetImaginary()
                return (
                    float(value.GetReal()),
                    float(imaginary[0]),
                    float(imaginary[1]),
                    float(imaginary[2]),
                )

            def insertion_forbidden_contacts():
                headers, _, _ = (
                    get_physx_simulation_interface().get_full_contact_report()
                )
                found = []
                for header in headers:
                    paths = (
                        str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                        str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                        str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                        str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                    )
                    if not _pair_contains_subtree(paths, plug_root):
                        continue
                    category = None
                    for candidate, target_path in (
                        (
                            "loose_fixed",
                            tabletop.asset.fixed_receptacle_prim_path,
                        ),
                        (
                            "loose_fixture",
                            tabletop.fixed_endpoint.fixture_prim_path,
                        ),
                        ("loose_table", tabletop.table.prim_path),
                    ):
                        if _pair_contains_subtree(paths, target_path):
                            category = candidate
                            break
                    if category is None:
                        continue
                    records = int(header.num_contact_data)
                    insertion_contact_records[category] += records
                    found.append((category, records, paths))
                return found

            def insertion_step(arm_target):
                observe_and_step(arm_target, current_hand_target, True)
                sample_post_tare_efforts()
                forbidden = insertion_forbidden_contacts()
                if forbidden:
                    category, records, paths = forbidden[0]
                    contact_alignment = alignment_at_current_body()
                    metrics["first_insertion_forbidden_contact"] = {
                        "axis_error_rad": contact_alignment.axis_error_rad,
                        "category": category,
                        "combined_entry_error_m": (
                            contact_alignment.combined_entry_error_m
                        ),
                        "contact_record_count": records,
                        "gap_m": contact_alignment.gap_m,
                        "global_step": global_step,
                        "lateral_error_m": (
                            contact_alignment.lateral_error_m
                        ),
                        "paths": list(paths),
                        "phase": phase,
                        "phase_step": phase_steps[phase],
                        "alignment_progress": metrics.get(
                            "insertion_alignment_progress", {}
                        ),
                    }
                    raise RuntimeError(
                        "forbidden loose-endpoint contact during insertion "
                        f"at phase={phase}, phase_step={phase_steps[phase]}, "
                        f"category={category}, paths={paths}"
                    )

            def move_insertion_arm(start, target, duration_s):
                steps = round(duration_s * rate_hz)
                result = np.asarray(start, dtype=np.float64)
                for index in range(steps):
                    result = np.asarray(
                        interpolate_arm(
                            tuple(float(value) for value in start),
                            tuple(float(value) for value in target),
                            float(index + 1) / float(steps),
                        ),
                        dtype=np.float64,
                    )
                    insertion_step(result)
                return np.asarray(target, dtype=np.float64)

            def measured_tcp_body_transforms():
                tcp_position, tcp_orientation = _world_pose(
                    Gf, Usd, UsdGeom, tcp_prim
                )
                body_position, body_orientation = body.get_world_pose()
                tcp_transform = pose_transform(
                    tuple(float(value) for value in tcp_position),
                    gf_quaternion_tuple(tcp_orientation),
                )
                body_transform = pose_transform(
                    tuple(float(value) for value in body_position),
                    tuple(float(value) for value in body_orientation),
                )
                return tcp_transform, body_transform

            fixed_position = np.asarray(
                assembly.datums.fixed.position_world_m, dtype=np.float64
            )
            fixed_axis = np.asarray(
                assembly.datums.fixed.axis_world, dtype=np.float64
            )
            desired_body_orientation = (1.0, 0.0, 0.0, 0.0)

            def desired_body_transform(gap_m):
                position = fixed_position + float(gap_m) * fixed_axis
                return pose_transform(position, desired_body_orientation)

            def alignment_at_current_body():
                body_position, body_orientation = body.get_world_pose()
                body_transform = pose_transform(
                    tuple(float(value) for value in body_position),
                    tuple(float(value) for value in body_orientation),
                )
                return measure_alignment(
                    body_position,
                    body_transform[:3, 2],
                    fixed_position,
                    fixed_axis,
                    insertion.acceptance.entry_evaluation_length_m,
                )

            def command_for_measured_grasp(seed_arm, desired_body):
                measured_tcp, measured_body = measured_tcp_body_transforms()
                desired_tcp = compensated_tcp_transform(
                    desired_body, measured_tcp, measured_body
                )
                return np.asarray(
                    solve_fixed_q7_tcp_pose(
                        seed_arm,
                        desired_tcp[:3, 3],
                        target_rotation=desired_tcp[:3, :3],
                    ),
                    dtype=np.float64,
                )

            def alignment_passes(measurement, gap_m, gap_tolerance):
                return bool(
                    abs(measurement.gap_m - gap_m) <= gap_tolerance
                    and measurement.lateral_error_m
                    <= insertion.acceptance.maximum_lateral_error_m
                    and measurement.axis_error_rad
                    <= insertion.acceptance.maximum_axis_error_rad
                    and measurement.combined_entry_error_m
                    <= insertion.acceptance.maximum_combined_entry_error_m
                )

            def converge_body_target(
                start_arm,
                seed_arm,
                gap_m,
                move_duration_s,
                hold_s,
                gap_tolerance,
                phase_name,
            ):
                """Use at most two ground-truth corrections.

                Corrections move robot joints only; they never write an object
                pose or create an attachment.
                """

                nonlocal phase
                target_body = desired_body_transform(gap_m)
                current = np.asarray(start_arm, dtype=np.float64)
                corrections = []
                command_tcp = None
                for correction_index in range(3):
                    # Map the current measured body pose onto the next desired
                    # body pose, then apply that world correction to the
                    # command TCP.  This carries forward steady PD tracking
                    # compensation instead of relearning it after every
                    # 0.25 mm axial step.
                    _, measured_body = measured_tcp_body_transforms()
                    world_correction = (
                        target_body @ np.linalg.inv(measured_body)
                    )
                    if command_tcp is None:
                        command_tcp = np.asarray(
                            iiwa14_grasp_tcp_transform(
                                tuple(float(value) for value in current)
                            ),
                            dtype=np.float64,
                        )
                    corrected_tcp = world_correction @ command_tcp
                    target = np.asarray(
                        solve_fixed_q7_tcp_pose(
                            seed_arm if correction_index == 0 else current,
                            corrected_tcp[:3, 3],
                            target_rotation=corrected_tcp[:3, :3],
                        ),
                        dtype=np.float64,
                    )
                    command_tcp = np.asarray(
                        iiwa14_grasp_tcp_transform(
                            tuple(float(value) for value in target)
                        ),
                        dtype=np.float64,
                    )
                    phase = (
                        phase_name
                        if correction_index == 0
                        else f"{phase_name}_correction_{correction_index}"
                    )
                    duration = (
                        move_duration_s if correction_index == 0 else 0.75
                    )
                    current = move_insertion_arm(current, target, duration)
                    hold_steps = round(hold_s * rate_hz)
                    for _ in range(hold_steps):
                        insertion_step(current)
                    measurement = alignment_at_current_body()
                    corrections.append(
                        {
                            "axis_error_rad": measurement.axis_error_rad,
                            "combined_entry_error_m": (
                                measurement.combined_entry_error_m
                            ),
                            "correction_index": correction_index,
                            "gap_m": measurement.gap_m,
                            "lateral_error_m": (
                                measurement.lateral_error_m
                            ),
                            "target_arm_rad": [
                                float(value) for value in current
                            ],
                        }
                    )
                    metrics.setdefault(
                        "insertion_alignment_progress", {}
                    )[phase_name] = list(corrections)
                    if alignment_passes(
                        measurement, gap_m, gap_tolerance
                    ):
                        return current, measurement, corrections
                return current, measurement, corrections

            def follow_axial_gap_path(
                start_arm,
                start_gap_m,
                end_gap_m,
                total_duration_s,
                final_hold_s,
                gap_tolerance,
                phase_prefix,
            ):
                """Servo along the connector axis in bounded 0.25 mm steps."""

                gaps = axial_gap_waypoints(
                    start_gap_m, end_gap_m, 0.00025
                )
                duration = total_duration_s / float(len(gaps))
                current = np.asarray(start_arm, dtype=np.float64)
                all_corrections = []
                for index, gap in enumerate(gaps, start=1):
                    hold = final_hold_s if index == len(gaps) else 0.05
                    current, measurement, corrections = (
                        converge_body_target(
                            current,
                            current,
                            gap,
                            duration,
                            hold,
                            gap_tolerance,
                            f"{phase_prefix}_{index:02d}",
                        )
                    )
                    all_corrections.extend(corrections)
                    if not alignment_passes(
                        measurement, gap, gap_tolerance
                    ):
                        raise RuntimeError(
                            "D38999 incremental axial alignment failed: "
                            f"gap_target={gap}, measurement={measurement}"
                        )
                return current, measurement, all_corrections

            if not passed:
                insertion_report = {
                    "passed": False,
                    "reason": "physical_pick_prerequisite_failed",
                }
            else:
                insertion_motion = insertion.motion
                lift_tcp, lift_body = measured_tcp_body_transforms()
                measured_tcp_to_body = np.linalg.inv(lift_tcp) @ lift_body
                nominal_safe_extra_gap = (
                    insertion_motion.transport_safe_tcp_position_m[2]
                    - insertion_motion.preinsert_tcp_position_m[2]
                )
                safe_gap = (
                    assembly.axial_plan.preinsert_gap_m
                    + nominal_safe_extra_gap
                )
                safe_body = desired_body_transform(safe_gap)
                safe_tcp = safe_body @ np.linalg.inv(measured_tcp_to_body)
                safe_arm = np.asarray(
                    solve_fixed_q7_tcp_pose(
                        insertion_motion.transport_safe_arm_rad,
                        safe_tcp[:3, 3],
                        target_rotation=safe_tcp[:3, :3],
                    ),
                    dtype=np.float64,
                )
                phase = "mixed_grip_transport_to_fixed_safe"
                current_arm_target = move_insertion_arm(
                    current_arm_target,
                    safe_arm,
                    insertion_motion.transport_duration_s,
                )

                (
                    current_arm_target,
                    axis_high_alignment,
                    axis_high_corrections,
                ) = converge_body_target(
                    current_arm_target,
                    insertion_motion.axis_high_arm_rad,
                    insertion_motion.axis_high_gap_m,
                    insertion_motion.axis_high_duration_s,
                    insertion_motion.axis_high_hold_s,
                    insertion.acceptance.maximum_axis_high_gap_error_m,
                    "mixed_grip_align_above_entry",
                )
                if not alignment_passes(
                    axis_high_alignment,
                    insertion_motion.axis_high_gap_m,
                    insertion.acceptance.maximum_axis_high_gap_error_m,
                ):
                    raise RuntimeError(
                        "D38999 above-entry alignment failed closed: "
                        f"{axis_high_alignment}"
                    )

                (
                    current_arm_target,
                    preinsert_alignment,
                    preinsert_corrections,
                ) = follow_axial_gap_path(
                    current_arm_target,
                    insertion_motion.axis_high_gap_m,
                    assembly.axial_plan.preinsert_gap_m,
                    insertion_motion.preinsert_duration_s,
                    insertion_motion.preinsert_hold_s,
                    insertion.acceptance.maximum_preinsert_gap_error_m,
                    "mixed_grip_preinsert",
                )
                if not alignment_passes(
                    preinsert_alignment,
                    assembly.axial_plan.preinsert_gap_m,
                    insertion.acceptance.maximum_preinsert_gap_error_m,
                ):
                    raise RuntimeError(
                        "D38999 preinsert alignment failed closed: "
                        f"{preinsert_alignment}"
                    )
                preinsert_body_position, _ = body.get_world_pose()
                preinsert_body_in_tcp = body_in_tcp_frame(
                    preinsert_body_position
                )
                preinsert_contacts = contact_snapshot()

                (
                    current_arm_target,
                    engage_alignment,
                    engage_corrections,
                ) = follow_axial_gap_path(
                    current_arm_target,
                    assembly.axial_plan.preinsert_gap_m,
                    assembly.axial_plan.engage_gap_m,
                    insertion_motion.insertion_duration_s,
                    insertion_motion.engage_hold_s,
                    insertion.acceptance.maximum_engage_gap_error_m,
                    "mixed_grip_physical_insert",
                )
                engage_body_position, _ = body.get_world_pose()
                engage_body_in_tcp = body_in_tcp_frame(engage_body_position)
                engage_contacts = contact_snapshot()
                insertion_travel = (
                    preinsert_alignment.gap_m - engage_alignment.gap_m
                )
                insertion_body_tcp_slip = float(
                    np.linalg.norm(
                        engage_body_in_tcp - preinsert_body_in_tcp
                    )
                )
                insertion_force_peak = float(
                    np.max(maximum_post_tare_absolute_delta_by_channel)
                )
                insertion_passed = bool(
                    alignment_passes(
                        engage_alignment,
                        assembly.axial_plan.engage_gap_m,
                        insertion.acceptance.maximum_engage_gap_error_m,
                    )
                    and abs(
                        insertion_travel
                        - assembly.axial_plan.insertion_travel_m
                    )
                    <= insertion.acceptance.maximum_insertion_travel_error_m
                    and insertion_body_tcp_slip
                    <= insertion.acceptance.maximum_body_tcp_slip_m
                    and _all_fingers_have_body_contact(preinsert_contacts)
                    and _all_fingers_have_body_contact(engage_contacts)
                    and insertion_force_peak
                    <= pick.sensing.maximum_absolute_torque_delta_nm
                    and not any(insertion_contact_records.values())
                    and metrics["d38999_authoring"][
                        "object_pose_writes_after_start"
                    ]
                    == 0
                )
                insertion_report = {
                    "assembly_success_claimed": False,
                    "attachment": "none",
                    "collision_planned": False,
                    "continuous_collision_verified": False,
                    "axis_high_alignment": {
                        "axis_error_rad": axis_high_alignment.axis_error_rad,
                        "combined_entry_error_m": (
                            axis_high_alignment.combined_entry_error_m
                        ),
                        "gap_m": axis_high_alignment.gap_m,
                        "lateral_error_m": (
                            axis_high_alignment.lateral_error_m
                        ),
                    },
                    "axis_high_corrections": axis_high_corrections,
                    "engage_alignment": {
                        "axis_error_rad": engage_alignment.axis_error_rad,
                        "combined_entry_error_m": (
                            engage_alignment.combined_entry_error_m
                        ),
                        "gap_m": engage_alignment.gap_m,
                        "lateral_error_m": (
                            engage_alignment.lateral_error_m
                        ),
                    },
                    "engage_corrections": engage_corrections,
                    "engage_contacts": engage_contacts,
                    "insertion_body_tcp_slip_m": (
                        insertion_body_tcp_slip
                    ),
                    "insertion_contact_records": (
                        dict(insertion_contact_records)
                    ),
                    "insertion_travel_m": insertion_travel,
                    "maximum_post_tare_torque_delta_nm": (
                        insertion_force_peak
                    ),
                    "object_drive": "none",
                    "object_pose_writes_after_start": 0,
                    "passed": insertion_passed,
                    "physical_insertion_included": True,
                    "pose_source": insertion.boundaries.pose_source,
                    "preinsert_alignment": {
                        "axis_error_rad": (
                            preinsert_alignment.axis_error_rad
                        ),
                        "combined_entry_error_m": (
                            preinsert_alignment.combined_entry_error_m
                        ),
                        "gap_m": preinsert_alignment.gap_m,
                        "lateral_error_m": (
                            preinsert_alignment.lateral_error_m
                        ),
                    },
                    "preinsert_corrections": preinsert_corrections,
                    "preinsert_contacts": preinsert_contacts,
                    "real_keying_modeled": False,
                    "thread_teeth_modeled": False,
                    "vision_included": False,
                }
                passed = bool(passed and insertion_passed)
            metrics["physical_insertion"] = insertion_report
            if arguments.end_to_end_probe:
                if not passed:
                    metrics["nut_only_regrasp"] = {
                        "passed": False,
                        "reason": "physical_insertion_prerequisite_failed",
                    }
                else:
                    # The real connector keyway is not modeled.  After the
                    # measured physical insertion has reached the 3 mm engage
                    # datum, this fixed joint is the explicit v0 keying proxy
                    # that lets the hand safely release BodyAssembly and move
                    # onto CouplingNut.  Its activation jump is measured.
                    regrasp_prepared = regrasp_contract["prepared_engage"]
                    regrasp_candidate = regrasp_contract[
                        "nut_only_candidate"
                    ]
                    regrasp_control = regrasp_contract["control"]
                    regrasp_sensing = regrasp_contract["sensing"]
                    regrasp_acceptance = regrasp_contract["acceptance"]
                    if (
                        regrasp_control["physics_rate_hz"] != rate_hz
                        or regrasp_sensing["hard_stop_nm"]
                        != pick.sensing.maximum_absolute_torque_delta_nm
                        or regrasp_sensing["operational_torque_target_nm"]
                        != pick.sensing.operational_torque_target_nm
                    ):
                        raise RuntimeError(
                            "end-to-end regrasp/runtime safety mismatch"
                        )

                    def set_end_to_end_hand_gains(stiffness, damping):
                        kps[hand_indices] = stiffness
                        kds[hand_indices] = damping
                        controller.set_gains(
                            kps=kps, kds=kds, save_to_usd=False
                        )

                    # Monitor the otherwise-free coupling nut during the
                    # inserted release/regrasp window.  A real connector has
                    # key/thread friction here; the proxy revolute joint does
                    # not, so its missing resistance must be explicit and
                    # measurable instead of appearing as unexplained spin.
                    pre_twist_stability = {
                        "active": False,
                        "initial_angle_rad": 0.0,
                        "last_angle_rad": 0.0,
                        "maximum_drift_rad": 0.0,
                        "maximum_observable_speed_rad_s": 0.0,
                        "maximum_post_solver_speed_rad_s": 0.0,
                    }
                    monitor_body_prim = stage.GetPrimAtPath(body_root)
                    monitor_nut_prim = stage.GetPrimAtPath(nut_root)

                    def end_to_end_efforts(tare):
                        measured = np.asarray(
                            robot.get_measured_joint_efforts(
                                joint_indices=sensor_indices
                            ),
                            dtype=np.float64,
                        )
                        delta = measured - tare
                        if not (
                            np.all(np.isfinite(measured))
                            and np.all(np.isfinite(delta))
                        ):
                            raise RuntimeError(
                                "non-finite end-to-end finger-base effort"
                            )
                        np.maximum(
                            maximum_post_tare_absolute_delta_by_channel,
                            np.abs(delta),
                            out=(
                                maximum_post_tare_absolute_delta_by_channel
                            ),
                        )
                        if np.any(
                            np.abs(delta) > regrasp_sensing["hard_stop_nm"]
                        ):
                            raise RuntimeError(
                                "end-to-end finger-base torque exceeded 2 Nm"
                            )
                        return measured

                    def end_to_end_step(
                        contact_mode,
                        tare=None,
                        *,
                        allow_fixed_contact=False,
                    ):
                        """Step the existing World with phase contact gates."""

                        allow_contact = contact_mode != "clear"
                        observe_and_step(
                            current_arm_target,
                            current_hand_target,
                            allow_contact,
                        )
                        measured = (
                            end_to_end_efforts(tare)
                            if tare is not None
                            else None
                        )
                        snapshot = contact_snapshot()
                        if pre_twist_stability["active"]:
                            wrapped_angle = _wrapped_relative_z_angle(
                                Gf,
                                Usd,
                                UsdGeom,
                                monitor_body_prim,
                                monitor_nut_prim,
                            )
                            current_angle = _unwrap_angle(
                                pre_twist_stability["last_angle_rad"],
                                wrapped_angle,
                            )
                            observable_speed = abs(
                                current_angle
                                - pre_twist_stability["last_angle_rad"]
                            ) * rate_hz
                            pre_twist_stability["last_angle_rad"] = (
                                current_angle
                            )
                            pre_twist_stability["maximum_drift_rad"] = max(
                                pre_twist_stability[
                                    "maximum_drift_rad"
                                ],
                                abs(
                                    current_angle
                                    - pre_twist_stability[
                                        "initial_angle_rad"
                                    ]
                                ),
                            )
                            pre_twist_stability[
                                "maximum_observable_speed_rad_s"
                            ] = max(
                                pre_twist_stability[
                                    "maximum_observable_speed_rad_s"
                                ],
                                observable_speed,
                            )
                            pre_twist_stability[
                                "maximum_post_solver_speed_rad_s"
                            ] = max(
                                pre_twist_stability[
                                    "maximum_post_solver_speed_rad_s"
                                ],
                                float(
                                    np.linalg.norm(
                                        np.asarray(
                                            nut.get_angular_velocity(),
                                            dtype=np.float64,
                                        )
                                    )
                                ),
                            )
                        if snapshot["unexpected_robot_link_records"] != 0:
                            raise RuntimeError(
                                "nonfinger contact during end-to-end regrasp"
                            )
                        if (
                            contact_mode == "clear"
                            and not _zero_finger_endpoint_contact(snapshot)
                        ):
                            raise RuntimeError(
                                "opened hand still contacts D38999 endpoint"
                            )
                        if (
                            contact_mode == "nut_only"
                            and not _zero_finger_body_contact(snapshot)
                        ):
                            raise RuntimeError(
                                "nut-only grip touched BodyAssembly"
                            )
                        world_contacts = insertion_forbidden_contacts()
                        forbidden = [
                            item
                            for item in world_contacts
                            if not (
                                allow_fixed_contact
                                and item[0] == "loose_fixed"
                            )
                        ]
                        if forbidden:
                            category, _, paths = forbidden[0]
                            raise RuntimeError(
                                "loose endpoint touched world during regrasp: "
                                f"{category}, {paths}"
                            )
                        return measured, snapshot

                    constraint_path = (
                        "/World/D38999EndToEnd/EngagedKeyingProxy"
                    )
                    phase = "engaged_keying_proxy_activation"
                    key_before_position, key_before_orientation = (
                        body.get_world_pose()
                    )
                    world.pause()
                    UsdGeom.Scope.Define(stage, "/World/D38999EndToEnd")
                    keying_joint = UsdPhysics.FixedJoint.Define(
                        stage, constraint_path
                    )
                    keying_joint.CreateBody1Rel().SetTargets(
                        [Sdf.Path(body_root)]
                    )
                    keying_joint.CreateLocalPos0Attr(
                        Gf.Vec3f(
                            *(float(value) for value in key_before_position)
                        )
                    )
                    keying_joint.CreateLocalRot0Attr(
                        Gf.Quatf(
                            float(key_before_orientation[0]),
                            Gf.Vec3f(
                                float(key_before_orientation[1]),
                                float(key_before_orientation[2]),
                                float(key_before_orientation[3]),
                            ),
                        )
                    )
                    keying_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
                    keying_joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
                    keying_joint.CreateCollisionEnabledAttr(False)
                    metrics["object_drive"] = (
                        "engaged_keying_proxy_fixed_joint"
                    )
                    world.play()
                    simulation_app.update()
                    end_to_end_step("mixed", tare_efforts)
                    key_after_position, key_after_orientation = (
                        body.get_world_pose()
                    )
                    key_translation_jump = float(
                        np.linalg.norm(
                            key_after_position - key_before_position
                        )
                    )
                    key_rotation_jump = _array_quaternion_error_radians(
                        key_before_orientation, key_after_orientation
                    )

                    # Hold the measured nut angle only while the inserted
                    # hand opens and moves to its Nut-only grasp.  This is the
                    # same 0.05 Nm, force-limited proxy used between rotation
                    # strokes.  It is removed before the runtime thread proxy
                    # and therefore never drives the commanded screw motion.
                    pre_twist_brake = rewind_contract[
                        "interstroke_self_lock_brake_proxy"
                    ]
                    pre_twist_angle = _wrapped_relative_z_angle(
                        Gf,
                        Usd,
                        UsdGeom,
                        monitor_body_prim,
                        monitor_nut_prim,
                    )
                    pre_twist_stability.update(
                        {
                            "active": True,
                            "initial_angle_rad": pre_twist_angle,
                            "last_angle_rad": pre_twist_angle,
                        }
                    )
                    pre_twist_hinge_prim = stage.GetPrimAtPath(
                        tabletop.asset.joint_prim_path
                    )
                    world.pause()
                    pre_twist_drive = UsdPhysics.DriveAPI.Apply(
                        pre_twist_hinge_prim,
                        UsdPhysics.Tokens.angular,
                    )
                    pre_twist_drive.CreateTypeAttr(
                        UsdPhysics.Tokens.force
                    )
                    pre_twist_drive.CreateStiffnessAttr(
                        pre_twist_brake["stiffness"]
                    )
                    pre_twist_drive.CreateDampingAttr(
                        pre_twist_brake["damping"]
                    )
                    pre_twist_drive.CreateTargetPositionAttr(
                        math.degrees(pre_twist_angle)
                    )
                    pre_twist_drive.CreateTargetVelocityAttr(
                        pre_twist_brake[
                            "target_velocity_degrees_per_second"
                        ]
                    )
                    pre_twist_drive.CreateMaxForceAttr(
                        pre_twist_brake["maximum_force_nm"]
                    )
                    world.play()
                    simulation_app.update()

                    open_regrasp_hand = np.asarray(
                        regrasp_prepared["open_hand_rad"], dtype=np.float64
                    )
                    nut_regrasp_arm = np.asarray(
                        regrasp_candidate["arm_rad"], dtype=np.float64
                    )
                    nut_regrasp_hand = np.asarray(
                        regrasp_candidate["hand_rad"], dtype=np.float64
                    )

                    phase = "end_to_end_release_mixed_grip"
                    release_start = current_hand_target.copy()
                    release_steps = round(
                        regrasp_control["release_s"] * rate_hz
                    )
                    for index in range(release_steps):
                        blend = minimum_jerk_blend(
                            float(index + 1) / float(release_steps)
                        )
                        current_hand_target = release_start + blend * (
                            open_regrasp_hand - release_start
                        )
                        end_to_end_step("mixed", tare_efforts)
                    current_hand_target = open_regrasp_hand.copy()
                    set_end_to_end_hand_gains(
                        regrasp_control["open_hand_stiffness"],
                        regrasp_control["open_hand_damping"],
                    )

                    phase = "end_to_end_open_reposition_to_nut"
                    arm_start = current_arm_target.copy()
                    reposition_steps = round(
                        regrasp_control["open_reposition_s"] * rate_hz
                    )
                    for index in range(reposition_steps):
                        current_arm_target = np.asarray(
                            interpolate_arm(
                                tuple(float(value) for value in arm_start),
                                tuple(
                                    float(value)
                                    for value in nut_regrasp_arm
                                ),
                                float(index + 1)
                                / float(reposition_steps),
                            ),
                            dtype=np.float64,
                        )
                        end_to_end_step("clear", tare_efforts)
                    current_arm_target = nut_regrasp_arm.copy()
                    open_reposition_snapshot = contact_snapshot()

                    phase = "end_to_end_nut_open_tare"
                    nut_tare_samples = []
                    for _ in range(
                        round(
                            regrasp_control["second_open_tare_s"] * rate_hz
                        )
                    ):
                        observe_and_step(
                            current_arm_target,
                            current_hand_target,
                            False,
                        )
                        nut_tare_samples.append(
                            np.asarray(
                                robot.get_measured_joint_efforts(
                                    joint_indices=sensor_indices
                                ),
                                dtype=np.float64,
                            )
                        )
                    tare_nut = np.mean(
                        np.stack(nut_tare_samples), axis=0
                    )
                    set_end_to_end_hand_gains(
                        regrasp_control["grip_hand_stiffness"],
                        regrasp_control["grip_hand_damping"],
                    )

                    phase = "end_to_end_nut_only_closure"
                    closure_start = current_hand_target.copy()
                    closure_steps = round(
                        regrasp_control["nut_closure_s"] * rate_hz
                    )
                    for index in range(closure_steps):
                        blend = minimum_jerk_blend(
                            float(index + 1) / float(closure_steps)
                        )
                        current_hand_target = closure_start + blend * (
                            nut_regrasp_hand - closure_start
                        )
                        end_to_end_step("nut_only", tare_nut)
                    current_hand_target = nut_regrasp_hand.copy()

                    phase = "end_to_end_nut_only_preload"
                    preload_samples = []
                    for _ in range(
                        round(regrasp_control["nut_preload_s"] * rate_hz)
                    ):
                        measured, _ = end_to_end_step(
                            "nut_only", tare_nut
                        )
                        preload_samples.append(measured.copy())
                    postclosure_snapshot = contact_snapshot()

                    phase = "end_to_end_nut_only_hold"
                    final_regrasp_samples = []
                    for _ in range(
                        round(regrasp_control["final_hold_s"] * rate_hz)
                    ):
                        measured, _ = end_to_end_step(
                            "nut_only", tare_nut
                        )
                        final_regrasp_samples.append(measured.copy())
                    final_regrasp_snapshot = contact_snapshot()
                    final_regrasp_effort = np.mean(
                        np.stack(final_regrasp_samples), axis=0
                    )
                    final_regrasp_delta = final_regrasp_effort - tare_nut
                    final_loaded_channels = int(
                        np.count_nonzero(
                            np.abs(final_regrasp_delta)
                            >= regrasp_sensing[
                                "loaded_torque_threshold_nm"
                            ]
                        )
                    )
                    # Zero force before API removal.  Isaac/PhysX may retain
                    # a removed live drive for one solver step; this mirrors
                    # the proven inter-stroke cleanup and prevents the
                    # anti-spin proxy from contaminating the first twist.
                    world.pause()
                    pre_twist_drive.GetStiffnessAttr().Set(0.0)
                    pre_twist_drive.GetDampingAttr().Set(0.0)
                    pre_twist_drive.GetMaxForceAttr().Set(0.0)
                    world.play()
                    simulation_app.update()
                    end_to_end_step("nut_only", tare_nut)
                    world.pause()
                    if not pre_twist_hinge_prim.RemoveAPI(
                        UsdPhysics.DriveAPI,
                        UsdPhysics.Tokens.angular,
                    ):
                        raise RuntimeError(
                            "pre-twist anti-spin brake removal failed"
                        )
                    world.play()
                    simulation_app.update()
                    pre_twist_stability["active"] = False
                    pre_twist_drift_gate = bool(
                        pre_twist_stability["maximum_drift_rad"]
                        <= rewind_contract["acceptance"][
                            "maximum_released_nut_drift_rad"
                        ]
                    )
                    regrasp_body_position, regrasp_body_orientation = (
                        body.get_world_pose()
                    )
                    regrasp_body_drift = float(
                        np.linalg.norm(
                            regrasp_body_position - key_after_position
                        )
                    )
                    regrasp_body_rotation_drift = (
                        _array_quaternion_error_radians(
                            key_after_orientation,
                            regrasp_body_orientation,
                        )
                    )
                    regrasp_tcp_position, regrasp_tcp_orientation = (
                        _world_pose(Gf, Usd, UsdGeom, tcp_prim)
                    )
                    regrasp_tcp_error = float(
                        np.linalg.norm(
                            np.asarray(regrasp_tcp_position)
                            - np.asarray(
                                regrasp_candidate[
                                    "tcp_position_world_m"
                                ]
                            )
                        )
                    )
                    desired_tcp_error = float(
                        np.linalg.norm(
                            np.asarray(regrasp_tcp_position)
                            - np.asarray(
                                regrasp_candidate[
                                    "desired_physical_tcp_position_world_m"
                                ]
                            )
                        )
                    )
                    regrasp_axis_error = _axis_error_radians(
                        _quaternion_world_z_axis(regrasp_tcp_orientation),
                        regrasp_candidate["tcp_down_axis_world"],
                    )
                    regrasp_passed = bool(
                        key_translation_jump
                        <= twist_contract["acceptance"][
                            "maximum_constraint_activation_translation_jump_m"
                        ]
                        and key_rotation_jump
                        <= twist_contract["acceptance"][
                            "maximum_constraint_activation_rotation_jump_rad"
                        ]
                        and _zero_finger_endpoint_contact(
                            open_reposition_snapshot
                        )
                        and _all_fingers_have_nut_contact(
                            postclosure_snapshot
                        )
                        and _zero_finger_body_contact(
                            postclosure_snapshot
                        )
                        and _all_fingers_have_nut_contact(
                            final_regrasp_snapshot
                        )
                        and _zero_finger_body_contact(
                            final_regrasp_snapshot
                        )
                        and final_loaded_channels
                        >= regrasp_sensing["minimum_loaded_channels"]
                        and regrasp_body_drift
                        <= regrasp_acceptance[
                            "maximum_body_translation_drift_m"
                        ]
                        and regrasp_body_rotation_drift
                        <= regrasp_acceptance[
                            "maximum_body_rotation_drift_rad"
                        ]
                        and regrasp_tcp_error
                        <= regrasp_acceptance[
                            "maximum_tcp_position_error_m"
                        ]
                        and desired_tcp_error
                        <= regrasp_acceptance[
                            "maximum_desired_physical_tcp_position_error_m"
                        ]
                        and regrasp_axis_error
                        <= regrasp_acceptance[
                            "maximum_tcp_axis_error_rad"
                        ]
                        and float(
                            np.max(
                                maximum_post_tare_absolute_delta_by_channel
                            )
                        )
                        <= regrasp_sensing["hard_stop_nm"]
                        and finite_throughout
                        and pre_twist_drift_gate
                    )
                    metrics["nut_only_regrasp"] = {
                        "activation_rotation_jump_rad": (
                            key_rotation_jump
                        ),
                        "activation_translation_jump_m": (
                            key_translation_jump
                        ),
                        "assembly_success_claimed": False,
                        "constraint_is_real_keying_claim": False,
                        "constraint_path": constraint_path,
                        "desired_physical_tcp_position_error_m": (
                            desired_tcp_error
                        ),
                        "final_all_fingers_nut_contact": (
                            _all_fingers_have_nut_contact(
                                final_regrasp_snapshot
                            )
                        ),
                        "final_contacts": final_regrasp_snapshot,
                        "final_loaded_channels": final_loaded_channels,
                        "final_torque_deltas_nm": dict(
                            zip(
                                regrasp_sensing["torque_joint_names"],
                                final_regrasp_delta,
                            )
                        ),
                        "final_zero_finger_body_contact": (
                            _zero_finger_body_contact(
                                final_regrasp_snapshot
                            )
                        ),
                        "open_reposition_zero_endpoint_contact": (
                            _zero_finger_endpoint_contact(
                                open_reposition_snapshot
                            )
                        ),
                        "passed": regrasp_passed,
                        "pre_twist_anti_spin_proxy": {
                            "maximum_force_nm": pre_twist_brake[
                                "maximum_force_nm"
                            ],
                            "maximum_nut_drift_rad": (
                                pre_twist_stability[
                                    "maximum_drift_rad"
                                ]
                            ),
                            "maximum_nut_observable_speed_rad_s": (
                                pre_twist_stability[
                                    "maximum_observable_speed_rad_s"
                                ]
                            ),
                            "maximum_nut_post_solver_speed_rad_s": (
                                pre_twist_stability[
                                    "maximum_post_solver_speed_rad_s"
                                ]
                            ),
                            "passed": pre_twist_drift_gate,
                            "removed_before_thread_activation": True,
                        },
                        "tcp_axis_error_rad": regrasp_axis_error,
                        "tcp_position_error_m": regrasp_tcp_error,
                    }
                    passed = bool(passed and regrasp_passed)

                    if regrasp_passed:
                        # Continue the independently proven 3 x 120 degree
                        # schedule in this same World.  The fixed keying proxy
                        # is replaced at the measured pose by the validated
                        # one-axis rack/prismatic thread proxy; neither change
                        # writes an object pose or attaches the plug to hand.
                        thread = twist_contract["runtime_thread"]
                        probe = twist_contract["probe"]
                        twist_sensing = twist_contract["sensing"]
                        twist_acceptance = twist_contract["acceptance"]
                        rewind_control = rewind_contract["control"]
                        rewind_sensing = rewind_contract["sensing"]
                        rewind_acceptance = rewind_contract["acceptance"]
                        if (
                            twist_contract["probe_id"] != "stage120"
                            or rewind_control["physics_rate_hz"] != rate_hz
                            or twist_sensing != rewind_sensing
                            or twist_sensing["hard_stop_nm"]
                            != pick.sensing.maximum_absolute_torque_delta_nm
                        ):
                            raise RuntimeError(
                                "full-rotation runtime contract mismatch"
                            )

                        hinge_prim = stage.GetPrimAtPath(
                            tabletop.asset.joint_prim_path
                        )
                        if not hinge_prim or not hinge_prim.IsA(
                            UsdPhysics.RevoluteJoint
                        ):
                            raise RuntimeError(
                                "D38999 passive nut hinge is missing"
                            )
                        if any(
                            str(schema).startswith("PhysicsDriveAPI")
                            for schema in hinge_prim.GetAppliedSchemas()
                        ):
                            raise RuntimeError(
                                "D38999 passive nut hinge already has a drive"
                            )

                        phase = "end_to_end_thread_proxy_activation"
                        thread_before_position, thread_before_orientation = (
                            body.get_world_pose()
                        )
                        world.pause()
                        if not stage.RemovePrim(constraint_path):
                            raise RuntimeError(
                                "engaged keying proxy removal failed"
                            )
                        UsdGeom.Scope.Define(stage, thread["root_prim_path"])
                        prismatic = UsdPhysics.PrismaticJoint.Define(
                            stage, thread["prismatic_prim_path"]
                        )
                        prismatic.CreateAxisAttr(thread["prismatic_axis"])
                        prismatic.CreateBody1Rel().SetTargets(
                            [Sdf.Path(body_root)]
                        )
                        prismatic.CreateLocalPos0Attr(
                            Gf.Vec3f(
                                *(
                                    float(value)
                                    for value in thread_before_position
                                )
                            )
                        )
                        prismatic.CreateLocalRot0Attr(
                            Gf.Quatf(
                                float(thread_before_orientation[0]),
                                Gf.Vec3f(
                                    float(thread_before_orientation[1]),
                                    float(thread_before_orientation[2]),
                                    float(thread_before_orientation[3]),
                                ),
                            )
                        )
                        prismatic.CreateLocalPos1Attr(Gf.Vec3f(0.0))
                        prismatic.CreateLocalRot1Attr(Gf.Quatf(1.0))
                        prismatic.CreateLowerLimitAttr(
                            thread["lower_limit_m"]
                        )
                        prismatic.CreateUpperLimitAttr(
                            thread["upper_limit_m"]
                        )
                        prismatic.CreateCollisionEnabledAttr(False)
                        rack = (
                            PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(
                                stage, thread["rack_prim_path"]
                            )
                        )
                        rack.CreateBody0Rel().SetTargets([Sdf.Path(nut_root)])
                        rack.CreateBody1Rel().SetTargets([Sdf.Path(body_root)])
                        rack.CreateHingeRel().SetTargets(
                            [Sdf.Path(tabletop.asset.joint_prim_path)]
                        )
                        rack.CreatePrismaticRel().SetTargets(
                            [Sdf.Path(thread["prismatic_prim_path"])]
                        )
                        rack.CreateRatioAttr(
                            thread["rack_ratio_degrees_per_meter"]
                        )
                        world.play()
                        simulation_app.update()
                        end_to_end_step("nut_only", tare_nut)
                        thread_after_position, thread_after_orientation = (
                            body.get_world_pose()
                        )
                        thread_translation_jump = float(
                            np.linalg.norm(
                                thread_after_position
                                - thread_before_position
                            )
                        )
                        thread_rotation_jump = (
                            _array_quaternion_error_radians(
                                thread_before_orientation,
                                thread_after_orientation,
                            )
                        )
                        for _ in range(9):
                            end_to_end_step("nut_only", tare_nut)
                        thread_activation_passed = bool(
                            thread_translation_jump
                            <= twist_acceptance[
                                "maximum_constraint_activation_"
                                "translation_jump_m"
                            ]
                            and thread_rotation_jump
                            <= twist_acceptance[
                                "maximum_constraint_activation_"
                                "rotation_jump_rad"
                            ]
                        )
                        metrics["runtime_thread_proxy"] = {
                            "activation_rotation_jump_rad": (
                                thread_rotation_jump
                            ),
                            "activation_translation_jump_m": (
                                thread_translation_jump
                            ),
                            "collision_filter_pair_count": (
                                proxy_collision_filter["pair_count"]
                            ),
                            "passed": thread_activation_passed,
                            "prismatic_path": thread[
                                "prismatic_prim_path"
                            ],
                            "rack_path": thread["rack_prim_path"],
                        }
                        if not thread_activation_passed:
                            raise RuntimeError(
                                "runtime thread proxy activation gate failed"
                            )
                        metrics["object_drive"] = (
                            "engaged_keying_then_runtime_thread_"
                            "and_brake_proxies"
                        )

                        q7_index = name_to_index[probe["q7_joint_name"]]
                        q7_arm_offset = pick.robot.arm_joint_names.index(
                            probe["q7_joint_name"]
                        )
                        q7_properties = dof_properties[q7_index]
                        body_prim = stage.GetPrimAtPath(body_root)
                        nut_prim = stage.GetPrimAtPath(nut_root)
                        fixed_rear_path = (
                            tabletop.asset.fixed_receptacle_prim_path
                            + "/RearBody"
                        )
                        body_mating_root = body_root + "/MatingShell"
                        initial_nut_angle = _wrapped_relative_z_angle(
                            Gf, Usd, UsdGeom, body_prim, nut_prim
                        )
                        unwrapped_nut_angle = initial_nut_angle
                        rotation_start_body_position, _ = (
                            body.get_world_pose()
                        )
                        rotation_start_fixed_position, (
                            rotation_start_fixed_orientation
                        ) = _world_pose(Gf, Usd, UsdGeom, fixed_prim)
                        loose_fixed_pair_records = {}

                        def update_nut_angle():
                            """Update the continuous measured nut progress."""

                            nonlocal unwrapped_nut_angle
                            wrapped = _wrapped_relative_z_angle(
                                Gf, Usd, UsdGeom, body_prim, nut_prim
                            )
                            unwrapped_nut_angle = _unwrap_angle(
                                unwrapped_nut_angle, wrapped
                            )

                        def count_loose_fixed_contacts():
                            """Count exact collider pairs for final seating."""

                            records = 0
                            headers, _, _ = (
                                get_physx_simulation_interface()
                                .get_full_contact_report()
                            )
                            fixed_root = (
                                tabletop.asset.fixed_receptacle_prim_path
                            )
                            for header in headers:
                                paths = tuple(
                                    str(
                                        PhysicsSchemaTools.intToSdfPath(
                                            value
                                        )
                                    )
                                    for value in (
                                        header.actor0,
                                        header.actor1,
                                        header.collider0,
                                        header.collider1,
                                    )
                                )
                                has_loose = _pair_contains_subtree(
                                    paths, plug_root
                                )
                                has_fixed = _pair_contains_subtree(
                                    paths, fixed_root
                                )
                                if not (has_loose and has_fixed):
                                    continue
                                count = int(header.num_contact_data)
                                records += count
                                key = " <-> ".join(
                                    sorted((paths[2], paths[3]))
                                )
                                loose_fixed_pair_records[key] = (
                                    loose_fixed_pair_records.get(key, 0)
                                    + count
                                )
                            return records

                        def remove_interstroke_brake(drive, active_tare):
                            """Zero the brake before removing its USD API.

                            PhysX can retain a removed live drive for one
                            solver step.  The explicit zero-force step is the
                            validated workaround and keeps the next stroke
                            from fighting a stale brake.
                            """

                            world.pause()
                            drive.GetStiffnessAttr().Set(0.0)
                            drive.GetDampingAttr().Set(0.0)
                            drive.GetMaxForceAttr().Set(0.0)
                            world.play()
                            simulation_app.update()
                            end_to_end_step("nut_only", active_tare)
                            update_nut_angle()
                            world.pause()
                            if not hinge_prim.RemoveAPI(
                                UsdPhysics.DriveAPI,
                                UsdPhysics.Tokens.angular,
                            ):
                                raise RuntimeError(
                                    "interstroke brake API removal failed"
                                )
                            world.play()
                            simulation_app.update()

                        def run_rotation_stroke(stroke_index, rewind_after):
                            """Execute one physical 120 degree stroke.

                            The first two strokes must remain contact-free at
                            the fixed endpoint.  Only the third stroke may end
                            on the exact twenty-segment rear seating stop.
                            """

                            nonlocal current_arm_target
                            nonlocal current_hand_target
                            nonlocal phase
                            nonlocal tare_nut

                            pair_start = dict(loose_fixed_pair_records)
                            stroke_contact_records = 0
                            start_positions = np.asarray(
                                robot.get_joint_positions(),
                                dtype=np.float64,
                            )
                            start_body_position, _ = body.get_world_pose()
                            start_fixed_position, start_fixed_orientation = (
                                _world_pose(Gf, Usd, UsdGeom, fixed_prim)
                            )
                            start_nut_angle = float(unwrapped_nut_angle)
                            command_start = float(
                                current_arm_target[q7_arm_offset]
                            )
                            command_target = (
                                command_start + probe["q7_delta_rad"]
                            )
                            if bool(q7_properties["hasLimits"]) and not (
                                float(q7_properties["lower"])
                                <= command_target
                                <= float(q7_properties["upper"])
                            ):
                                raise RuntimeError(
                                    f"stroke {stroke_index} exceeds q7 limits"
                                )

                            # The analytic rack reaches a hard axial stop on
                            # stroke three, but the proxy nut hinge has none
                            # of the real thread's preload/self-lock friction.
                            # A very low-force moving target prevents that
                            # free revolute from numerically spinning at the
                            # stop.  q7 and the rack still perform the motion;
                            # this explicit proxy is capped at 0.05 Nm and is
                            # reported separately from real assembly evidence.
                            final_seating_stabilizer = None
                            if stroke_index == 3:
                                stabilizer = rewind_contract[
                                    "interstroke_self_lock_brake_proxy"
                                ]
                                world.pause()
                                final_seating_stabilizer = (
                                    UsdPhysics.DriveAPI.Apply(
                                        hinge_prim,
                                        UsdPhysics.Tokens.angular,
                                    )
                                )
                                final_seating_stabilizer.CreateTypeAttr(
                                    UsdPhysics.Tokens.force
                                )
                                final_seating_stabilizer.CreateStiffnessAttr(
                                    stabilizer["stiffness"]
                                )
                                final_seating_stabilizer.CreateDampingAttr(
                                    stabilizer["damping"]
                                )
                                final_seating_stabilizer.CreateTargetPositionAttr(  # noqa: E501
                                    math.degrees(start_nut_angle)
                                )
                                final_seating_stabilizer.CreateTargetVelocityAttr(  # noqa: E501
                                    stabilizer[
                                        "target_velocity_degrees_per_second"
                                    ]
                                )
                                final_seating_stabilizer.CreateMaxForceAttr(
                                    stabilizer["maximum_force_nm"]
                                )
                                world.play()
                                simulation_app.update()

                            phase = (
                                f"end_to_end_rotation_{stroke_index}_motion"
                            )
                            motion_steps = round(
                                probe["motion_duration_s"]
                                * rate_hz
                                / q7_motion_speedup
                            )
                            for index in range(motion_steps):
                                fraction = float(index + 1) / float(
                                    motion_steps
                                )
                                current_arm_target[q7_arm_offset] = (
                                    command_start
                                    + minimum_jerk_blend(fraction)
                                    * probe["q7_delta_rad"]
                                )
                                if final_seating_stabilizer is not None:
                                    expected_nut_angle = (
                                        start_nut_angle
                                        - (
                                            current_arm_target[q7_arm_offset]
                                            - command_start
                                        )
                                    )
                                    target_position_attr = (
                                        final_seating_stabilizer
                                        .GetTargetPositionAttr()
                                    )
                                    target_position_attr.Set(
                                        math.degrees(expected_nut_angle)
                                    )
                                end_to_end_step(
                                    "nut_only",
                                    tare_nut,
                                    allow_fixed_contact=(
                                        stroke_index == 3
                                    ),
                                )
                                stroke_contact_records += (
                                    count_loose_fixed_contacts()
                                )
                                update_nut_angle()
                            current_arm_target[q7_arm_offset] = command_target

                            # The rewind contract says that its force-limited
                            # brake is applied *after the commanded twist*.
                            # Install it before the hold window, not after the
                            # measurements.  This matters at the final axial
                            # stop: without a real self-locking thread the
                            # free revolute nut can spin numerically while the
                            # hand is holding it, especially with GUI render
                            # scheduling.  The 0.05 Nm cap remains far below
                            # the 2.0 Nm finger hard stop and all original
                            # angle, helix, contact and velocity gates remain
                            # active.
                            brake = rewind_contract[
                                "interstroke_self_lock_brake_proxy"
                            ]
                            if final_seating_stabilizer is None:
                                brake_target_degrees = math.degrees(
                                    unwrapped_nut_angle
                                )
                                world.pause()
                                brake_drive = UsdPhysics.DriveAPI.Apply(
                                    hinge_prim, UsdPhysics.Tokens.angular
                                )
                                brake_drive.CreateTypeAttr(
                                    UsdPhysics.Tokens.force
                                )
                                brake_drive.CreateStiffnessAttr(
                                    brake["stiffness"]
                                )
                                brake_drive.CreateDampingAttr(
                                    brake["damping"]
                                )
                                brake_drive.CreateTargetPositionAttr(
                                    brake_target_degrees
                                )
                                brake_drive.CreateTargetVelocityAttr(
                                    brake[
                                        "target_velocity_degrees_per_second"
                                    ]
                                )
                                brake_drive.CreateMaxForceAttr(
                                    brake["maximum_force_nm"]
                                )
                                world.play()
                                simulation_app.update()
                            else:
                                brake_drive = final_seating_stabilizer

                            phase = (
                                f"end_to_end_rotation_{stroke_index}_hold"
                            )
                            hold_steps = round(
                                probe["total_hold_duration_s"] * rate_hz
                            )
                            evaluation_steps = round(
                                probe["hold_evaluation_duration_s"]
                                * rate_hz
                            )
                            previous_positions = np.asarray(
                                robot.get_joint_positions(),
                                dtype=np.float64,
                            )
                            previous_body_position, _ = body.get_world_pose()
                            previous_nut_angle = float(unwrapped_nut_angle)
                            q7_speeds = []
                            nut_speeds = []
                            body_z_positions = [
                                float(previous_body_position[2])
                            ]
                            effort_samples = []
                            for _ in range(hold_steps):
                                measured, _ = end_to_end_step(
                                    "nut_only",
                                    tare_nut,
                                    allow_fixed_contact=(
                                        stroke_index == 3
                                    ),
                                )
                                effort_samples.append(measured.copy())
                                stroke_contact_records += (
                                    count_loose_fixed_contacts()
                                )
                                positions = np.asarray(
                                    robot.get_joint_positions(),
                                    dtype=np.float64,
                                )
                                update_nut_angle()
                                body_position, _ = body.get_world_pose()
                                q7_speeds.append(
                                    abs(
                                        float(
                                            positions[q7_index]
                                            - previous_positions[q7_index]
                                        )
                                    )
                                    * rate_hz
                                )
                                nut_speeds.append(
                                    abs(
                                        unwrapped_nut_angle
                                        - previous_nut_angle
                                    )
                                    * rate_hz
                                )
                                body_z_positions.append(
                                    float(body_position[2])
                                )
                                previous_positions = positions
                                previous_body_position = body_position
                                previous_nut_angle = float(
                                    unwrapped_nut_angle
                                )

                            final_positions = np.asarray(
                                robot.get_joint_positions(),
                                dtype=np.float64,
                            )
                            final_body_position, final_body_orientation = (
                                body.get_world_pose()
                            )
                            final_fixed_position, final_fixed_orientation = (
                                _world_pose(Gf, Usd, UsdGeom, fixed_prim)
                            )
                            q7_delta = float(
                                final_positions[q7_index]
                                - start_positions[q7_index]
                            )
                            nut_delta = float(
                                unwrapped_nut_angle - start_nut_angle
                            )
                            axial_delta = float(
                                final_body_position[2]
                                - start_body_position[2]
                            )
                            expected_axial = float(
                                -probe["lead_m_per_revolution"]
                                * nut_delta
                                / math.tau
                            )
                            final_efforts = np.mean(
                                np.stack(effort_samples), axis=0
                            )
                            final_deltas = final_efforts - tare_nut
                            loaded_channels = int(
                                np.count_nonzero(
                                    np.abs(final_deltas)
                                    >= twist_sensing[
                                        "loaded_torque_threshold_nm"
                                    ]
                                )
                            )
                            final_snapshot = contact_snapshot()
                            pair_records = {
                                key: value - pair_start.get(key, 0)
                                for key, value in (
                                    loose_fixed_pair_records.items()
                                )
                                if value - pair_start.get(key, 0) > 0
                            }
                            final_seating_gate = bool(
                                stroke_index == 3
                                and not rewind_after
                                and validate_final_seating_contact_pairs(
                                    pair_records,
                                    fixed_rear_path=fixed_rear_path,
                                    body_mating_root=body_mating_root,
                                )
                            )
                            contact_gate = bool(
                                _all_fingers_have_nut_contact(
                                    final_snapshot
                                )
                                and _zero_finger_body_contact(final_snapshot)
                                and final_snapshot[
                                    "unexpected_robot_link_records"
                                ]
                                == 0
                                and (
                                    stroke_contact_records == 0
                                    or final_seating_gate
                                )
                            )
                            evaluation_q7 = q7_speeds[-evaluation_steps:]
                            evaluation_nut = nut_speeds[-evaluation_steps:]
                            axial_window = twist_acceptance[
                                "hold_axial_observable_window_steps"
                            ]
                            evaluation_start = (
                                hold_steps - evaluation_steps
                            )
                            evaluation_axial = [
                                abs(
                                    body_z_positions[index]
                                    - body_z_positions[index - axial_window]
                                )
                                * rate_hz
                                / axial_window
                                for index in range(
                                    evaluation_start + axial_window,
                                    hold_steps + 1,
                                )
                            ]
                            axis_error = _axis_error_radians(
                                _quaternion_world_z_axis(
                                    Gf.Quatd(
                                        float(final_body_orientation[0]),
                                        Gf.Vec3d(
                                            float(
                                                final_body_orientation[1]
                                            ),
                                            float(
                                                final_body_orientation[2]
                                            ),
                                            float(
                                                final_body_orientation[3]
                                            ),
                                        ),
                                    )
                                ),
                                (0.0, 0.0, 1.0),
                            )
                            lateral_drift = float(
                                np.linalg.norm(
                                    final_body_position[:2]
                                    - start_body_position[:2]
                                )
                            )
                            fixed_translation_drift = float(
                                np.linalg.norm(
                                    np.asarray(final_fixed_position)
                                    - np.asarray(start_fixed_position)
                                )
                            )
                            fixed_rotation_drift = (
                                _gf_quaternion_error_radians(
                                    start_fixed_orientation,
                                    final_fixed_orientation,
                                )
                            )
                            non_q7_indices = [
                                index
                                for index in arm_indices
                                if index != q7_index
                            ]
                            non_q7_drift = float(
                                np.max(
                                    np.abs(
                                        final_positions[non_q7_indices]
                                        - start_positions[non_q7_indices]
                                    )
                                )
                            )
                            progress_fraction = abs(axial_delta) / abs(
                                probe["expected_axial_travel_m"]
                            )
                            stroke_passed = bool(
                                finite_throughout
                                and abs(
                                    q7_delta - probe["q7_delta_rad"]
                                )
                                <= twist_acceptance[
                                    "maximum_q7_tracking_error_rad"
                                ]
                                and twist_acceptance[
                                    "minimum_nut_progress_rad"
                                ]
                                <= nut_delta
                                <= twist_acceptance[
                                    "maximum_nut_progress_rad"
                                ]
                                and abs(q7_delta + nut_delta)
                                <= twist_acceptance[
                                    "maximum_q7_to_nut_slip_rad"
                                ]
                                and axial_delta < 0.0
                                and progress_fraction
                                >= twist_acceptance[
                                    "minimum_axial_progress_fraction"
                                ]
                                and abs(axial_delta - expected_axial)
                                <= twist_acceptance[
                                    "maximum_helical_error_m"
                                ]
                                and non_q7_drift
                                <= twist_acceptance[
                                    "maximum_non_q7_arm_drift_rad"
                                ]
                                and lateral_drift
                                <= twist_acceptance[
                                    "maximum_body_lateral_drift_m"
                                ]
                                and axis_error
                                <= twist_acceptance[
                                    "maximum_body_axis_error_rad"
                                ]
                                and max(evaluation_q7)
                                <= twist_acceptance[
                                    "maximum_hold_q7_speed_rad_s"
                                ]
                                and max(evaluation_nut)
                                <= twist_acceptance[
                                    "maximum_hold_nut_speed_rad_s"
                                ]
                                and max(evaluation_axial)
                                <= twist_acceptance[
                                    "maximum_hold_axial_speed_m_s"
                                ]
                                and fixed_translation_drift
                                <= twist_acceptance[
                                    "maximum_fixed_translation_drift_m"
                                ]
                                and fixed_rotation_drift
                                <= twist_acceptance[
                                    "maximum_fixed_rotation_drift_rad"
                                ]
                                and loaded_channels
                                >= twist_sensing[
                                    "minimum_loaded_channels"
                                ]
                                and float(
                                    np.max(
                                        maximum_post_tare_absolute_delta_by_channel  # noqa: E501
                                    )
                                )
                                <= twist_sensing["hard_stop_nm"]
                                and contact_gate
                                and metrics[
                                    "object_pose_writes_after_start"
                                ]
                                == 0
                            )
                            stroke_report = {
                                "actual_axial_travel_m": axial_delta,
                                "actual_nut_delta_rad": nut_delta,
                                "actual_q7_delta_rad": q7_delta,
                                "body_axis_error_rad": axis_error,
                                "body_lateral_drift_m": lateral_drift,
                                "final_contacts": final_snapshot,
                                "final_loaded_channels": loaded_channels,
                                "final_seating_contact_gate": (
                                    final_seating_gate
                                ),
                                "final_seating_stabilizer_proxy_used": (
                                    final_seating_stabilizer is not None
                                ),
                                "final_seating_stabilizer_maximum_force_nm": (
                                    brake["maximum_force_nm"]
                                    if final_seating_stabilizer is not None
                                    else 0.0
                                ),
                                "final_torque_deltas_nm": dict(
                                    zip(
                                        twist_sensing[
                                            "torque_joint_names"
                                        ],
                                        final_deltas,
                                    )
                                ),
                                "helical_error_m": (
                                    axial_delta - expected_axial
                                ),
                                "hold_max_axial_speed_m_s": max(
                                    evaluation_axial
                                ),
                                "hold_max_nut_speed_rad_s": max(
                                    evaluation_nut
                                ),
                                "hold_max_q7_speed_rad_s": max(
                                    evaluation_q7
                                ),
                                "loose_fixed_contact_pair_records": dict(
                                    sorted(pair_records.items())
                                ),
                                "loose_fixed_contact_records": (
                                    stroke_contact_records
                                ),
                                "passed": stroke_passed,
                                "q7_command_start_rad": command_start,
                                "q7_command_target_rad": command_target,
                                "stroke_index": stroke_index,
                                "twist_contact_gate": contact_gate,
                            }
                            if not rewind_after:
                                return stroke_report, None

                            released_nut_start = float(
                                unwrapped_nut_angle
                            )
                            released_body_start = (
                                final_body_position.copy()
                            )
                            maximum_nut_drift = 0.0
                            maximum_axial_drift = 0.0

                            def update_release_drift():
                                nonlocal maximum_nut_drift
                                nonlocal maximum_axial_drift
                                update_nut_angle()
                                released_position, _ = body.get_world_pose()
                                maximum_nut_drift = max(
                                    maximum_nut_drift,
                                    abs(
                                        unwrapped_nut_angle
                                        - released_nut_start
                                    ),
                                )
                                maximum_axial_drift = max(
                                    maximum_axial_drift,
                                    abs(
                                        float(
                                            released_position[2]
                                            - released_body_start[2]
                                        )
                                    ),
                                )

                            phase = (
                                f"end_to_end_rewind_{stroke_index}_release"
                            )
                            release_start = current_hand_target.copy()
                            release_steps = round(
                                rewind_control["release_s"] * rate_hz
                            )
                            for index in range(release_steps):
                                blend = minimum_jerk_blend(
                                    float(index + 1) / float(release_steps)
                                )
                                current_hand_target = release_start + blend * (
                                    open_regrasp_hand - release_start
                                )
                                end_to_end_step("nut_only", tare_nut)
                                update_release_drift()
                            current_hand_target = open_regrasp_hand.copy()
                            set_end_to_end_hand_gains(
                                regrasp_control["open_hand_stiffness"],
                                regrasp_control["open_hand_damping"],
                            )

                            phase = (
                                f"end_to_end_rewind_{stroke_index}_open"
                            )
                            for _ in range(
                                round(
                                    rewind_control["open_settle_s"]
                                    * rate_hz
                                )
                            ):
                                end_to_end_step("clear", tare_nut)
                                update_release_drift()
                            open_snapshot = contact_snapshot()

                            before_rewind_positions = np.asarray(
                                robot.get_joint_positions(),
                                dtype=np.float64,
                            )
                            rewind_start = float(
                                current_arm_target[q7_arm_offset]
                            )
                            rewind_target = (
                                rewind_start
                                + rewind_control["rewind_delta_rad"]
                            )
                            if bool(q7_properties["hasLimits"]) and not (
                                float(q7_properties["lower"])
                                <= rewind_target
                                <= float(q7_properties["upper"])
                            ):
                                raise RuntimeError(
                                    f"rewind {stroke_index} exceeds q7 limits"
                                )
                            phase = (
                                f"end_to_end_rewind_{stroke_index}_motion"
                            )
                            rewind_steps = round(
                                rewind_control["rewind_duration_s"]
                                * rate_hz
                                / q7_motion_speedup
                            )
                            for index in range(rewind_steps):
                                blend = minimum_jerk_blend(
                                    float(index + 1) / float(rewind_steps)
                                )
                                current_arm_target[q7_arm_offset] = (
                                    rewind_start
                                    + blend
                                    * rewind_control["rewind_delta_rad"]
                                )
                                end_to_end_step("clear", tare_nut)
                                update_release_drift()
                            current_arm_target[q7_arm_offset] = rewind_target
                            after_rewind_positions = np.asarray(
                                robot.get_joint_positions(),
                                dtype=np.float64,
                            )
                            actual_rewind = float(
                                after_rewind_positions[q7_index]
                                - before_rewind_positions[q7_index]
                            )
                            tracking_error = (
                                actual_rewind
                                - rewind_control["rewind_delta_rad"]
                            )

                            phase = (
                                f"end_to_end_rewind_{stroke_index}_settle"
                            )
                            settle_q7 = []
                            settle_nut = []
                            previous_positions = (
                                after_rewind_positions.copy()
                            )
                            previous_nut = float(unwrapped_nut_angle)
                            for _ in range(
                                round(
                                    rewind_control[
                                        "post_rewind_settle_s"
                                    ]
                                    * rate_hz
                                )
                            ):
                                end_to_end_step("clear", tare_nut)
                                update_release_drift()
                                positions = np.asarray(
                                    robot.get_joint_positions(),
                                    dtype=np.float64,
                                )
                                settle_q7.append(
                                    abs(
                                        float(
                                            positions[q7_index]
                                            - previous_positions[q7_index]
                                        )
                                    )
                                    * rate_hz
                                )
                                settle_nut.append(
                                    abs(
                                        unwrapped_nut_angle
                                        - previous_nut
                                    )
                                    * rate_hz
                                )
                                previous_positions = positions
                                previous_nut = float(unwrapped_nut_angle)

                            phase = (
                                f"end_to_end_rewind_{stroke_index}_tare"
                            )
                            tare_samples = []
                            for _ in range(
                                round(
                                    rewind_control["second_open_tare_s"]
                                    * rate_hz
                                )
                            ):
                                observe_and_step(
                                    current_arm_target,
                                    current_hand_target,
                                    False,
                                )
                                update_release_drift()
                                tare_samples.append(
                                    np.asarray(
                                        robot.get_measured_joint_efforts(
                                            joint_indices=sensor_indices
                                        ),
                                        dtype=np.float64,
                                    )
                                )
                            tare_nut = np.mean(
                                np.stack(tare_samples), axis=0
                            )
                            set_end_to_end_hand_gains(
                                regrasp_control["grip_hand_stiffness"],
                                regrasp_control["grip_hand_damping"],
                            )

                            phase = (
                                f"end_to_end_rewind_{stroke_index}_regrasp"
                            )
                            reclose_start = current_hand_target.copy()
                            reclose_steps = round(
                                rewind_control["reclosure_s"] * rate_hz
                            )
                            for index in range(reclose_steps):
                                blend = minimum_jerk_blend(
                                    float(index + 1) / float(reclose_steps)
                                )
                                current_hand_target = reclose_start + blend * (
                                    nut_regrasp_hand - reclose_start
                                )
                                end_to_end_step("nut_only", tare_nut)
                                update_nut_angle()
                            current_hand_target = nut_regrasp_hand.copy()

                            phase = (
                                f"end_to_end_rewind_{stroke_index}_preload"
                            )
                            for _ in range(
                                round(
                                    rewind_control["preload_s"] * rate_hz
                                )
                            ):
                                end_to_end_step("nut_only", tare_nut)
                                update_nut_angle()

                            remove_interstroke_brake(
                                brake_drive, tare_nut
                            )
                            set_end_to_end_hand_gains(
                                regrasp_control["grip_hand_stiffness"],
                                regrasp_control["grip_hand_damping"],
                            )

                            # A physical regrasp can relax the nut by a few
                            # degrees when the temporary self-lock brake is
                            # removed.  Do not hide that loss or loosen the
                            # 2-degree retention gate: measure it and perform
                            # a slow q7 correction under the same nut-only,
                            # contact and 2 N*m gates before continuing.
                            # First let the new finger contacts reach their
                            # unbraked equilibrium.  Measuring immediately
                            # after DriveAPI removal underestimates the loss;
                            # the remaining relaxation would otherwise occur
                            # during the final acceptance hold.
                            phase = (
                                f"end_to_end_rewind_{stroke_index}_relax"
                            )
                            for _ in range(
                                round(
                                    rewind_control["final_hold_s"]
                                    * rate_hz
                                )
                            ):
                                end_to_end_step("nut_only", tare_nut)
                                update_nut_angle()
                            progress_before_recovery = float(
                                unwrapped_nut_angle - start_nut_angle
                            )
                            loss_before_recovery = float(
                                nut_delta - progress_before_recovery
                            )
                            maximum_recovery = abs(
                                probe["q7_delta_rad"]
                            ) * 0.10
                            if abs(loss_before_recovery) > maximum_recovery:
                                raise RuntimeError(
                                    "post-regrasp progress recovery exceeds "
                                    "10 percent of one stroke"
                                )
                            recovery_q7_delta = -loss_before_recovery
                            recovery_start = float(
                                current_arm_target[q7_arm_offset]
                            )
                            recovery_target = (
                                recovery_start + recovery_q7_delta
                            )
                            if bool(q7_properties["hasLimits"]) and not (
                                float(q7_properties["lower"])
                                <= recovery_target
                                <= float(q7_properties["upper"])
                            ):
                                raise RuntimeError(
                                    "post-regrasp recovery exceeds q7 limits"
                                )
                            recovery_steps = max(
                                1,
                                round(
                                    abs(recovery_q7_delta)
                                    / probe["q7_speed_rad_s"]
                                    * rate_hz
                                ),
                            )
                            phase = (
                                f"end_to_end_rewind_{stroke_index}_recovery"
                            )
                            for index in range(recovery_steps):
                                blend = minimum_jerk_blend(
                                    float(index + 1)
                                    / float(recovery_steps)
                                )
                                current_arm_target[q7_arm_offset] = (
                                    recovery_start
                                    + blend * recovery_q7_delta
                                )
                                end_to_end_step("nut_only", tare_nut)
                                update_nut_angle()
                            current_arm_target[q7_arm_offset] = (
                                recovery_target
                            )

                            phase = (
                                f"end_to_end_rewind_{stroke_index}_hold"
                            )
                            final_samples = []
                            for _ in range(
                                round(
                                    rewind_control["final_hold_s"]
                                    * rate_hz
                                )
                            ):
                                measured, _ = end_to_end_step(
                                    "nut_only", tare_nut
                                )
                                final_samples.append(measured.copy())
                                update_nut_angle()
                            final_snapshot_after_rewind = (
                                contact_snapshot()
                            )
                            final_efforts_after_rewind = np.mean(
                                np.stack(final_samples), axis=0
                            )
                            final_deltas_after_rewind = (
                                final_efforts_after_rewind - tare_nut
                            )
                            final_loaded_after_rewind = int(
                                np.count_nonzero(
                                    np.abs(final_deltas_after_rewind)
                                    >= rewind_sensing[
                                        "loaded_torque_threshold_nm"
                                    ]
                                )
                            )
                            progress_loss = abs(
                                (
                                    unwrapped_nut_angle - start_nut_angle
                                )
                                - nut_delta
                            )
                            rewind_contact_gate = bool(
                                _zero_finger_endpoint_contact(open_snapshot)
                                and _all_fingers_have_nut_contact(
                                    final_snapshot_after_rewind
                                )
                                and _zero_finger_body_contact(
                                    final_snapshot_after_rewind
                                )
                                and final_snapshot_after_rewind[
                                    "unexpected_robot_link_records"
                                ]
                                == 0
                            )
                            rewind_passed = bool(
                                finite_throughout
                                and abs(tracking_error)
                                <= rewind_acceptance[
                                    "maximum_q7_rewind_tracking_error_rad"
                                ]
                                and maximum_nut_drift
                                <= rewind_acceptance[
                                    "maximum_released_nut_drift_rad"
                                ]
                                and maximum_axial_drift
                                <= rewind_acceptance[
                                    "maximum_released_body_axial_drift_m"
                                ]
                                and max(settle_q7)
                                <= rewind_acceptance[
                                    "maximum_settle_q7_observable_speed_rad_s"
                                ]
                                and max(settle_nut)
                                <= rewind_acceptance[
                                    "maximum_settle_nut_observable_speed_rad_s"
                                ]
                                and progress_loss
                                <= rewind_acceptance[
                                    "maximum_final_nut_progress_loss_rad"
                                ]
                                and final_loaded_after_rewind
                                >= rewind_sensing[
                                    "minimum_loaded_channels"
                                ]
                                and float(
                                    np.max(
                                        maximum_post_tare_absolute_delta_by_channel  # noqa: E501
                                    )
                                )
                                <= rewind_sensing["hard_stop_nm"]
                                and rewind_contact_gate
                            )
                            rewind_report = {
                                "actual_q7_rewind_delta_rad": (
                                    actual_rewind
                                ),
                                "final_contacts": (
                                    final_snapshot_after_rewind
                                ),
                                "final_loaded_channels": (
                                    final_loaded_after_rewind
                                ),
                                "final_nut_progress_loss_rad": (
                                    progress_loss
                                ),
                                "nut_progress_loss_before_recovery_rad": (
                                    loss_before_recovery
                                ),
                                "q7_recovery_delta_rad": (
                                    recovery_q7_delta
                                ),
                                "maximum_released_body_axial_drift_m": (
                                    maximum_axial_drift
                                ),
                                "maximum_released_nut_drift_rad": (
                                    maximum_nut_drift
                                ),
                                "passed": rewind_passed,
                                "q7_rewind_tracking_error_rad": (
                                    tracking_error
                                ),
                                "rewind_contact_gate": (
                                    rewind_contact_gate
                                ),
                                "stroke_index": stroke_index,
                            }
                            return stroke_report, rewind_report

                        stroke_reports = []
                        rewind_reports = []
                        for stroke_index in (1, 2, 3):
                            stroke_report, rewind_report = (
                                run_rotation_stroke(
                                    stroke_index, stroke_index < 3
                                )
                            )
                            stroke_reports.append(stroke_report)
                            # Persist partial evidence before fail-fast so a
                            # rejected long run identifies the exact physical
                            # gate instead of leaving only an exception name.
                            metrics["full_rotation_partial"] = {
                                "rewind_reports": list(rewind_reports),
                                "stroke_reports": list(stroke_reports),
                            }
                            if stroke_report["passed"] is not True:
                                raise RuntimeError(
                                    "full rotation stroke "
                                    f"{stroke_index} failed"
                                )
                            if rewind_report is not None:
                                rewind_reports.append(rewind_report)
                                metrics["full_rotation_partial"] = {
                                    "rewind_reports": list(rewind_reports),
                                    "stroke_reports": list(stroke_reports),
                                }
                                if rewind_report["passed"] is not True:
                                    raise RuntimeError(
                                        "full rotation rewind "
                                        f"{stroke_index} failed"
                                    )

                        full_rotation_evidence = (
                            evaluate_d38999_full_rotation(
                                stroke_reports,
                                rewind_reports,
                                expected_stroke_progress_rad=probe[
                                    "expected_nut_delta_rad"
                                ],
                                expected_stroke_axial_travel_m=probe[
                                    "expected_axial_travel_m"
                                ],
                            )
                        )
                        rotation_final_body_position, _ = (
                            body.get_world_pose()
                        )
                        measured_total_nut_progress = float(
                            unwrapped_nut_angle - initial_nut_angle
                        )
                        measured_total_axial_travel = float(
                            rotation_final_body_position[2]
                            - rotation_start_body_position[2]
                        )
                        metrics["full_rotation"] = {
                            "assembly_success_claimed": False,
                            "continuous_collision_verified": False,
                            "cumulative_axial_travel_m": (
                                full_rotation_evidence
                                .cumulative_axial_travel_m
                            ),
                            "cumulative_nut_progress_rad": (
                                full_rotation_evidence
                                .cumulative_nut_progress_rad
                            ),
                            "interstroke_brake_proxy_used": True,
                            "measured_total_axial_travel_m": (
                                measured_total_axial_travel
                            ),
                            "measured_total_nut_progress_rad": (
                                measured_total_nut_progress
                            ),
                            "missing_or_failed": list(
                                full_rotation_evidence.missing_or_failed
                            ),
                            "passed": full_rotation_evidence.passed,
                            "physical_insertion_included": True,
                            "rewind_count": (
                                full_rotation_evidence.rewind_count
                            ),
                            "rewind_reports": rewind_reports,
                            "stroke_count": (
                                full_rotation_evidence.stroke_count
                            ),
                            "stroke_reports": stroke_reports,
                        }
                        passed = bool(
                            passed and full_rotation_evidence.passed
                        )

                        if full_rotation_evidence.passed:
                            # The real connector's self-locking thread is not
                            # modeled.  Keep an explicit low-force brake at
                            # the measured final angle while the hand releases
                            # and returns Home; the report never presents this
                            # proxy as real hardware self-lock evidence.
                            final_brake = rewind_contract[
                                "interstroke_self_lock_brake_proxy"
                            ]
                            final_brake_target_degrees = math.degrees(
                                unwrapped_nut_angle
                            )
                            world.pause()
                            final_drive = UsdPhysics.DriveAPI.Apply(
                                hinge_prim, UsdPhysics.Tokens.angular
                            )
                            final_drive.CreateTypeAttr(
                                UsdPhysics.Tokens.force
                            )
                            final_drive.CreateStiffnessAttr(
                                final_brake["stiffness"]
                            )
                            final_drive.CreateDampingAttr(
                                final_brake["damping"]
                            )
                            final_drive.CreateTargetPositionAttr(
                                final_brake_target_degrees
                            )
                            final_drive.CreateTargetVelocityAttr(
                                final_brake[
                                    "target_velocity_degrees_per_second"
                                ]
                            )
                            final_drive.CreateMaxForceAttr(
                                final_brake["maximum_force_nm"]
                            )
                            world.play()
                            simulation_app.update()

                            phase = "end_to_end_final_release"
                            release_start = current_hand_target.copy()
                            release_steps = round(
                                rewind_control["release_s"] * rate_hz
                            )
                            for index in range(release_steps):
                                blend = minimum_jerk_blend(
                                    float(index + 1) / float(release_steps)
                                )
                                current_hand_target = release_start + blend * (
                                    open_regrasp_hand - release_start
                                )
                                end_to_end_step(
                                    "nut_only",
                                    tare_nut,
                                    allow_fixed_contact=True,
                                )
                                update_nut_angle()
                            current_hand_target = open_regrasp_hand.copy()
                            set_end_to_end_hand_gains(
                                regrasp_control["open_hand_stiffness"],
                                regrasp_control["open_hand_damping"],
                            )

                            phase = "end_to_end_final_open_settle"
                            for _ in range(
                                round(
                                    rewind_control["open_settle_s"]
                                    * rate_hz
                                )
                            ):
                                end_to_end_step(
                                    "clear",
                                    tare_nut,
                                    allow_fixed_contact=True,
                                )
                                update_nut_angle()
                            final_release_snapshot = contact_snapshot()

                            def move_open_arm(
                                target_arm, duration_s, phase_name
                            ):
                                """Reverse validated waypoints.

                                The hand stays open and every physical step
                                continues to use the same contact gates.
                                """

                                nonlocal current_arm_target
                                nonlocal phase
                                start_arm = current_arm_target.copy()
                                target = np.asarray(
                                    target_arm, dtype=np.float64
                                )
                                steps = round(duration_s * rate_hz)
                                phase = phase_name
                                for index in range(steps):
                                    blend = minimum_jerk_blend(
                                        float(index + 1) / float(steps)
                                    )
                                    current_arm_target = start_arm + blend * (
                                        target - start_arm
                                    )
                                    end_to_end_step(
                                        "clear",
                                        tare_nut,
                                        allow_fixed_contact=True,
                                    )
                                    update_nut_angle()
                                current_arm_target = target.copy()

                            # First lift straight back to the high fixed-axis
                            # waypoint while retaining the current q7 angle.
                            # Then restore q7 to the exact transport waypoint
                            # and reverse the already executed transport/pick
                            # waypoints instead of inventing a new return path.
                            safe_with_current_q7 = np.asarray(
                                insertion.motion.transport_safe_arm_rad,
                                dtype=np.float64,
                            )
                            safe_with_current_q7[-1] = (
                                current_arm_target[-1]
                            )
                            move_open_arm(
                                safe_with_current_q7,
                                insertion.motion.axis_high_duration_s,
                                "end_to_end_retreat_above_fixed",
                            )
                            transport_safe_arm = np.asarray(
                                insertion.motion.transport_safe_arm_rad,
                                dtype=np.float64,
                            )
                            q7_return_delta = float(
                                transport_safe_arm[-1]
                                - current_arm_target[-1]
                            )
                            q7_return_duration = (
                                abs(q7_return_delta)
                                / rewind_control["rewind_speed_rad_s"]
                                / q7_motion_speedup
                            )
                            move_open_arm(
                                transport_safe_arm,
                                q7_return_duration,
                                "end_to_end_q7_return_at_safe_height",
                            )
                            move_open_arm(
                                pregrasp_arm,
                                insertion.motion.transport_duration_s,
                                "end_to_end_reverse_transport",
                            )
                            approach_segments = pick.motion.approach_segments
                            move_open_arm(
                                approach_segments[-2].target_arm_rad,
                                approach_segments[-1].duration_s,
                                "end_to_end_reverse_pregrasp_to_high",
                            )
                            move_open_arm(
                                approach_segments[0].target_arm_rad,
                                approach_segments[-2].duration_s,
                                "end_to_end_reverse_high_to_mid",
                            )
                            move_open_arm(
                                home_arm,
                                approach_segments[0].duration_s,
                                "end_to_end_reverse_mid_to_home",
                            )

                            phase = "end_to_end_home_hold"
                            previous_home_positions = np.asarray(
                                robot.get_joint_positions(),
                                dtype=np.float64,
                            )
                            home_observable_speeds = []
                            home_solver_speeds = []
                            for _ in range(rate_hz):
                                end_to_end_step(
                                    "clear",
                                    tare_nut,
                                    allow_fixed_contact=True,
                                )
                                update_nut_angle()
                                positions = np.asarray(
                                    robot.get_joint_positions(),
                                    dtype=np.float64,
                                )
                                home_observable_speeds.append(
                                    float(
                                        np.max(
                                            np.abs(
                                                positions
                                                - previous_home_positions
                                            )
                                        )
                                    )
                                    * rate_hz
                                )
                                solver_velocities = np.asarray(
                                    robot.get_joint_velocities(),
                                    dtype=np.float64,
                                )
                                home_solver_speeds.append(
                                    float(
                                        np.max(
                                            np.abs(solver_velocities)
                                        )
                                    )
                                )
                                previous_home_positions = positions

                            final_positions = np.asarray(
                                robot.get_joint_positions(),
                                dtype=np.float64,
                            )
                            final_body_position, final_body_orientation = (
                                body.get_world_pose()
                            )
                            final_fixed_position, final_fixed_orientation = (
                                _world_pose(Gf, Usd, UsdGeom, fixed_prim)
                            )
                            final_gap = assembly.axial_gap_m(
                                tuple(
                                    float(value)
                                    for value in final_body_position
                                )
                            )
                            final_lateral_error = float(
                                np.linalg.norm(
                                    np.asarray(final_body_position[:2])
                                    - np.asarray(
                                        assembly.datums.fixed.position_world_m[
                                            :2
                                        ]
                                    )
                                )
                            )
                            final_axis_error = _axis_error_radians(
                                _quaternion_world_z_axis(
                                    Gf.Quatd(
                                        float(final_body_orientation[0]),
                                        Gf.Vec3d(
                                            float(final_body_orientation[1]),
                                            float(final_body_orientation[2]),
                                            float(final_body_orientation[3]),
                                        ),
                                    )
                                ),
                                assembly.datums.fixed.axis_world,
                            )
                            final_fixed_translation_drift = float(
                                np.linalg.norm(
                                    np.asarray(final_fixed_position)
                                    - np.asarray(
                                        rotation_start_fixed_position
                                    )
                                )
                            )
                            final_fixed_rotation_drift = (
                                _gf_quaternion_error_radians(
                                    rotation_start_fixed_orientation,
                                    final_fixed_orientation,
                                )
                            )
                            final_home_arm_error = float(
                                np.max(
                                    np.abs(
                                        final_positions[arm_indices]
                                        - home_arm
                                    )
                                )
                            )
                            final_home_hand_error = float(
                                np.max(
                                    np.abs(
                                        final_positions[hand_indices]
                                        - open_regrasp_hand
                                    )
                                )
                            )
                            final_release_clear = (
                                _zero_finger_endpoint_contact(
                                    final_release_snapshot
                                )
                            )
                            final_snapshot = contact_snapshot()
                            final_clear = _zero_finger_endpoint_contact(
                                final_snapshot
                            )
                            final_seating_gate = stroke_reports[-1][
                                "final_seating_contact_gate"
                            ]
                            final_post_solver_speed_limit = (
                                pick.acceptance
                                .maximum_final_post_solver_joint_speed_rad_s
                            )
                            home_return_passed = bool(
                                final_release_clear
                                and final_clear
                                and final_home_arm_error
                                <= (
                                    pick.acceptance
                                    .maximum_arm_tracking_error_rad
                                )
                                and final_home_hand_error
                                <= (
                                    pick.acceptance
                                    .maximum_arm_tracking_error_rad
                                )
                                and max(home_observable_speeds)
                                <= (
                                    pick.acceptance
                                    .maximum_final_observable_joint_speed_rad_s
                                )
                                and max(home_solver_speeds)
                                <= final_post_solver_speed_limit
                            )
                            proxy_assembly_verified = bool(
                                abs(
                                    final_gap
                                    - assembly.axial_plan.final_gap_m
                                )
                                <= twist_acceptance[
                                    "maximum_helical_error_m"
                                ]
                                and final_lateral_error
                                <= twist_acceptance[
                                    "maximum_body_lateral_drift_m"
                                ]
                                and final_axis_error
                                <= twist_acceptance[
                                    "maximum_body_axis_error_rad"
                                ]
                                and final_fixed_translation_drift
                                <= twist_acceptance[
                                    "maximum_fixed_translation_drift_m"
                                ]
                                and final_fixed_rotation_drift
                                <= twist_acceptance[
                                    "maximum_fixed_rotation_drift_rad"
                                ]
                                and final_seating_gate is True
                                and metrics[
                                    "object_pose_writes_after_start"
                                ]
                                == 0
                            )
                            metrics["release_retreat_home"] = {
                                "final_clear_of_endpoint": final_clear,
                                "final_hand_error_rad": (
                                    final_home_hand_error
                                ),
                                "final_home_arm_error_rad": (
                                    final_home_arm_error
                                ),
                                "maximum_home_observable_speed_rad_s": max(
                                    home_observable_speeds
                                ),
                                "maximum_home_post_solver_speed_rad_s": max(
                                    home_solver_speeds
                                ),
                                "passed": home_return_passed,
                                "release_clear_of_endpoint": (
                                    final_release_clear
                                ),
                            }
                            metrics["proxy_assembly_verification"] = {
                                "assembly_success_claimed": False,
                                "final_axis_error_rad": final_axis_error,
                                "final_gap_m": final_gap,
                                "final_lateral_error_m": (
                                    final_lateral_error
                                ),
                                "final_seating_contact_gate": (
                                    final_seating_gate
                                ),
                                "fixed_rotation_drift_rad": (
                                    final_fixed_rotation_drift
                                ),
                                "fixed_translation_drift_m": (
                                    final_fixed_translation_drift
                                ),
                                "passed": proxy_assembly_verified,
                                "real_keying_modeled": False,
                                "real_thread_self_lock_verified": False,
                                "self_lock_brake_proxy_active": True,
                            }
                            metrics["end_to_end"] = {
                                "assembly_success_claimed": False,
                                "continuous_collision_verified": False,
                                "control_pose_provider": "sim_ground_truth",
                                "foundation_pose": False,
                                "ground_truth_pose_used": True,
                                "masked_rgbd_preflight_included": bool(
                                    arguments.pose_preflight == "masked-rgbd"
                                ),
                                "masked_rgbd_xy_used_for_control": False,
                                "passed": bool(
                                    home_return_passed
                                    and proxy_assembly_verified
                                    and pose_preflight_gate
                                ),
                                "pose_preflight_passed": (
                                    pose_preflight_gate
                                ),
                                "real_vision_included": False,
                                "truth_orientation_used": True,
                            }
                            passed = bool(
                                passed
                                and metrics["end_to_end"]["passed"]
                            )
            if tooth_probe is not None:
                metrics["nut_tooth_jitter_probe"] = tooth_probe.finalize()
            if wrist_ft_monitor is not None:
                wrist_report = wrist_ft_monitor.report()
                if wrist_ft_monitor_error is not None:
                    wrist_report.update(
                        {
                            "status": "MONITOR_FAILED",
                            "runtime_error": wrist_ft_monitor_error,
                        }
                    )
                metrics["virtual_wrist_ft_monitor"] = wrist_report
            metrics["passed"] = passed
        print(_metrics_json(metrics), flush=True)
        print(
            result_marker + ("PASSED" if passed else "FAILED"),
            flush=True,
        )
    except BaseException as exception:
        metrics.update(
            {
                "error": f"{type(exception).__name__}: {exception}",
                "passed": False,
            }
        )
        if tooth_probe is not None:
            metrics["nut_tooth_jitter_probe"] = tooth_probe.finalize()
        if wrist_ft_monitor is not None:
            wrist_report = wrist_ft_monitor.report()
            wrist_report.update(
                {
                    "status": "E2E_ABORTED_MONITOR_ONLY",
                    "runtime_error": wrist_ft_monitor_error,
                }
            )
            metrics["virtual_wrist_ft_monitor"] = wrist_report
        traceback.print_exc()
        print(_metrics_json(metrics), flush=True)
        print(result_marker + "FAILED", flush=True)
    finally:
        if arguments.keep_open and arguments.gui:
            print(
                result_marker + "GUI REMAINS OPEN; "
                "close the window to exit",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
        simulation_app.close(exit_code=0 if passed else 1)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
