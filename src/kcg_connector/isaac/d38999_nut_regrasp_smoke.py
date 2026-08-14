#!/usr/bin/env python3

"""PhysX A/B for a prepared-engage, nut-only D38999 regrasp.

This deliberately starts from a pre-authored 3 mm engage gap and uses a
temporary world-to-BodyAssembly fixed joint as a stand-in for verified keying.
By default it stops after regrasp.  ``--twist-probe`` explicitly replaces the
temporary fixed joint with a one-way prismatic+rack proxy and performs only a
20 degree q7 direction probe.  It never claims real thread contact or assembly
success.  No object pose is written after physics starts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import traceback

import numpy as np
import yaml

from d38999_tabletop_pick_smoke import (
    EXPECTED_DOF_NAMES,
    _array_quaternion_error_radians,
    _axis_error_radians,
    _classify_robot_external_contact,
    _d38999_loose_collider_group,
    _finger_loose_contact_group,
    _gf_quaternion_error_radians,
    _is_finger_plug_contact,
    _metrics_json,
    _quaternion_world_z_axis,
    _world_pose,
)


SCHEMA_VERSION = "kcg_d38999_nut_regrasp_physx_v1"
CAMERA_EYE_M = (1.30, 0.92, 0.66)
CAMERA_TARGET_M = (0.55, 0.185, 0.29)
RTX_HISTORY_SETTING = "/rtx/scenedb/maxHistoryTransformCount"
FABRIC_SCENE_DELEGATE_SETTING = "/app/useFabricSceneDelegate"
DIAGNOSTIC_RTX_HISTORY_COUNT = 512


def _wrapped_relative_z_angle(Gf, Usd, UsdGeom, body_prim, nut_prim):
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


def _unwrap(previous, wrapped):
    previous_wrapped = math.atan2(math.sin(previous), math.cos(previous))
    delta = math.atan2(
        math.sin(wrapped - previous_wrapped),
        math.cos(wrapped - previous_wrapped),
    )
    return previous + delta


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite_vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} numbers")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{label}[{index}] must be numeric")
        item = float(item)
        if not math.isfinite(item):
            raise ValueError(f"{label}[{index}] must be finite")
        result.append(item)
    return tuple(result)


def _load_contract(path, repository):
    document = _strict_mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")), "contract"
    )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("nut regrasp PhysX schema mismatch")
    if document.get("enabled") is not True:
        raise ValueError("nut regrasp PhysX smoke is not explicitly enabled")
    inputs = _strict_mapping(document.get("inputs"), "inputs")
    expected_inputs = {
        "tabletop_pick",
        "tabletop_scene",
        "shell_proxy",
        "assembly_baseline",
        "cpu_search_config",
        "cpu_search_report",
        "cpu_command_compensation_config",
        "cpu_command_compensation_report",
        "uncompensated_physx_log",
        "robot_asset",
    }
    if set(inputs) != expected_inputs:
        raise ValueError("nut regrasp PhysX input set is not exact")
    resolved = {}
    for name, raw in inputs.items():
        item = _strict_mapping(raw, f"inputs.{name}")
        if set(item) != {"path", "sha256"}:
            raise ValueError(f"inputs.{name} keys are not exact")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"inputs.{name}.path must be repository-relative")
        target = (repository / relative).resolve()
        if not target.is_file() or repository not in target.parents:
            raise ValueError(f"inputs.{name} is missing or escapes repository")
        if _sha256(target) != item["sha256"]:
            raise ValueError(f"inputs.{name} SHA-256 mismatch")
        resolved[name] = target

    report = _strict_mapping(
        yaml.safe_load(resolved["cpu_search_report"].read_text()),
        "cpu search report",
    )
    if (
        report.get("status")
        != "FAIL_CLOSED_CONTINUOUS_COLLISION_UNVERIFIED"
        or report.get("nut_only_command_candidate_found") is not True
        or report.get("candidate_may_proceed_to_physx_static_ab") is not True
        or report.get("continuous_collision_verified") is not False
        or report.get("discrete_regrasp_path", {}).get("passed") is not True
    ):
        raise ValueError("CPU search report does not authorize a static A/B")
    compensated_report = _strict_mapping(
        yaml.safe_load(
            resolved["cpu_command_compensation_report"].read_text()
        ),
        "CPU command compensation report",
    )
    if (
        compensated_report.get("status")
        != "FAIL_CLOSED_CONTINUOUS_COLLISION_UNVERIFIED"
        or compensated_report.get("nut_only_command_candidate_found")
        is not True
        or compensated_report.get("candidate_may_proceed_to_physx_static_ab")
        is not True
        or compensated_report.get("continuous_collision_verified") is not False
        or compensated_report.get("discrete_regrasp_path", {}).get("passed")
        is not True
        or compensated_report.get("config_sha256")
        != _sha256(resolved["cpu_command_compensation_config"])
    ):
        raise ValueError(
            "CPU command compensation report does not authorize a static A/B"
        )
    seed = _strict_mapping(
        document.get("uncompensated_seed"), "uncompensated_seed"
    )
    seed_best = _strict_mapping(report.get("best_feasible"), "best_feasible")
    if (
        _finite_vector(seed.get("arm_rad"), 7, "seed arm")
        != tuple(seed_best.get("arm_command_rad", ()))
        or _finite_vector(seed.get("hand_rad"), 4, "seed hand")
        != tuple(seed_best.get("hand_command_rad", ()))
        or _finite_vector(seed.get("tcp_position_world_m"), 3, "seed TCP")[2]
        != seed_best.get("tcp_z_command_m")
    ):
        raise ValueError("uncompensated seed differs from CPU search result")
    candidate = _strict_mapping(
        document.get("nut_only_candidate"), "nut_only_candidate"
    )
    best = _strict_mapping(
        compensated_report.get("best_feasible"),
        "compensated best_feasible",
    )
    if (
        _finite_vector(candidate.get("arm_rad"), 7, "candidate arm")
        != tuple(best.get("arm_command_rad", ()))
        or any(
            not math.isclose(left, right, abs_tol=1e-14)
            for left, right in zip(
                _finite_vector(candidate.get("hand_rad"), 4, "candidate hand"),
                tuple(best.get("hand_command_rad", ())),
            )
        )
        or _finite_vector(
            candidate.get("tcp_position_world_m"), 3, "candidate TCP"
        )[2]
        != best.get("tcp_z_command_m")
    ):
        raise ValueError(
            "PhysX candidate differs from compensated CPU state"
        )
    log_metrics = None
    for line in reversed(
        resolved["uncompensated_physx_log"].read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    ):
        if line.startswith("{"):
            log_metrics = json.loads(line)
            break
    if not isinstance(log_metrics, Mapping):
        raise ValueError("uncompensated PhysX log has no metrics JSON")
    compensation = _strict_mapping(
        document.get("tracking_compensation"), "tracking_compensation"
    )
    measured = _finite_vector(
        compensation.get("measured_uncompensated_tcp_position_world_m"),
        3,
        "measured uncompensated TCP",
    )
    logged_tcp = tuple(
        log_metrics.get("nut_regrasp_tcp_position_world_m", ())
    )
    if measured != logged_tcp:
        raise ValueError("uncompensated PhysX TCP evidence differs from log")
    desired = _finite_vector(
        candidate.get("desired_physical_tcp_position_world_m"),
        3,
        "desired physical TCP",
    )
    seed_tcp = _finite_vector(
        seed.get("tcp_position_world_m"), 3, "seed TCP"
    )
    errors = _finite_vector(
        compensation.get("measured_uncompensated_error_world_m"),
        3,
        "measured uncompensated error",
    )
    fraction = compensation.get("fraction")
    if (
        desired != seed_tcp
        or isinstance(fraction, bool)
        or not isinstance(fraction, Real)
        or float(fraction) != 0.75
        or any(
            not math.isclose(
                measured[index] - desired[index], errors[index], abs_tol=1e-15
            )
            for index in range(3)
        )
    ):
        raise ValueError("tracking compensation evidence is inconsistent")
    commanded = _finite_vector(
        candidate.get("tcp_position_world_m"), 3, "candidate TCP"
    )
    if any(
        not math.isclose(
            desired[index] - float(fraction) * errors[index],
            commanded[index],
            abs_tol=1e-15,
        )
        for index in range(3)
    ):
        raise ValueError("tracking compensation command is inconsistent")
    return document, resolved, compensated_report


def _minimum_jerk(fraction):
    return fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)


def _interpolate(start, target, fraction):
    blend = _minimum_jerk(fraction)
    return start + blend * (target - start)


def _all_fingers_endpoint_contact(snapshot):
    records = snapshot.get("finger_body_group_records", {})
    return all(
        records.get(finger, {}).get("body", 0)
        + records.get(finger, {}).get("nut", 0)
        > 0
        for finger in ("f1", "f2", "f3")
    )


def _all_fingers_nut_contact(snapshot):
    records = snapshot.get("finger_body_group_records", {})
    return all(
        records.get(finger, {}).get("nut", 0) > 0
        for finger in ("f1", "f2", "f3")
    )


def _zero_finger_body_contact(snapshot):
    records = snapshot.get("finger_body_group_records", {})
    return all(
        records.get(finger, {}).get("body", 0) == 0
        for finger in ("f1", "f2", "f3")
    )


def _zero_endpoint_contact(snapshot):
    records = snapshot.get("finger_body_group_records", {})
    return all(
        records.get(finger, {}).get("body", 0) == 0
        and records.get(finger, {}).get("nut", 0) == 0
        for finger in ("f1", "f2", "f3")
    )


def _build_argument_parser(repository):
    """Build the CLI without importing or starting Isaac Sim."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_nut_regrasp_physx_v1.yaml"
        ),
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--twist-probe", action="store_true")
    parser.add_argument(
        "--twist-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_q7_twist_probe_v1.yaml"
        ),
    )
    parser.add_argument("--rewind-probe", action="store_true")
    parser.add_argument(
        "--rewind-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_q7_rewind_probe_v1.yaml"
        ),
    )
    parser.add_argument(
        "--full-rotation-probe",
        action="store_true",
        help=(
            "reuse the validated stage120 and rewind controls for exactly "
            "three 120-degree strokes in one physical scene"
        ),
    )
    parser.add_argument(
        "--nut-tooth-jitter-output",
        help=(
            "enable the opt-in 24-tooth 240 Hz diagnostic and write its "
            "compact CSV/JSON evidence into this directory"
        ),
    )
    parser.add_argument(
        "--nut-tooth-jitter-normalize-segment00-op",
        action="store_true",
        help=(
            "A/B only: author explicit Segment_00 rotateZ=0 in the session "
            "layer before physics starts; never edits the checked-in USDA"
        ),
    )
    parser.add_argument(
        "--nut-tooth-jitter-colorize",
        action="store_true",
        help="author deterministic 24-tooth displayColor IDs in-session",
    )
    parser.add_argument(
        "--nut-tooth-sync-capture-output",
        help=(
            "opt-in fixed rear-oblique 30 Hz RGB frames plus an exact "
            "frame/global-step/phase sidecar for prepared hold/twist/hold"
        ),
    )
    parser.add_argument(
        "--nut-tooth-ghost-fingers-output",
        help=(
            "opt-in occlusion control: hide only the three finger render "
            "roots in the anonymous session layer while preserving physics"
        ),
    )
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument(
        "--nut-tooth-jitter-rtx-history",
        type=int,
        choices=(DIAGNOSTIC_RTX_HISTORY_COUNT,),
        help=(
            "GUI diagnostic A/B: launch Kit with RTX transform history 512; "
            "this tests whether long-exposure history causes the visible tooth"
        ),
    )
    render_group.add_argument(
        "--nut-tooth-jitter-disable-fabric-scene-delegate",
        action="store_true",
        help=(
            "GUI diagnostic A/B: launch Kit with Fabric Scene Delegate off to "
            "separate USD/Fabric presentation from physical tooth motion"
        ),
    )
    return parser


def _parse_arguments(repository, argv=None):
    parser = _build_argument_parser(repository)
    arguments = parser.parse_args(argv)
    if arguments.full_rotation_probe:
        # Full rotation is an extension of the same stage120 and rewind
        # contracts; setting both flags avoids a second execution path.
        arguments.twist_probe = True
        arguments.rewind_probe = True
    if arguments.keep_open and not arguments.gui:
        parser.error("--keep-open requires --gui")
    if arguments.rewind_probe and not arguments.twist_probe:
        parser.error("--rewind-probe requires --twist-probe")
    if arguments.nut_tooth_jitter_output and not arguments.twist_probe:
        parser.error("--nut-tooth-jitter-output requires --twist-probe")
    if (
        arguments.nut_tooth_jitter_normalize_segment00_op
        or arguments.nut_tooth_jitter_colorize
    ) and not arguments.nut_tooth_jitter_output:
        parser.error(
            "nut-tooth A/B/color options require "
            "--nut-tooth-jitter-output"
        )
    render_variant = bool(
        arguments.nut_tooth_jitter_rtx_history is not None
        or arguments.nut_tooth_jitter_disable_fabric_scene_delegate
    )
    if render_variant and not arguments.nut_tooth_jitter_output:
        parser.error(
            "nut-tooth render A/B options require "
            "--nut-tooth-jitter-output"
        )
    if render_variant and not arguments.gui:
        parser.error("nut-tooth render A/B options require --gui")
    if arguments.nut_tooth_sync_capture_output and not arguments.gui:
        parser.error("--nut-tooth-sync-capture-output requires --gui")
    if arguments.nut_tooth_sync_capture_output and not arguments.twist_probe:
        parser.error(
            "--nut-tooth-sync-capture-output requires --twist-probe"
        )
    if (
        arguments.nut_tooth_sync_capture_output
        and not arguments.nut_tooth_jitter_output
    ):
        parser.error(
            "--nut-tooth-sync-capture-output requires "
            "--nut-tooth-jitter-output"
        )
    if arguments.nut_tooth_sync_capture_output:
        # The synchronized frames require the stable deterministic per-tooth
        # IDs; make that requirement explicit without a second user flag.
        arguments.nut_tooth_jitter_colorize = True
    if (
        arguments.nut_tooth_ghost_fingers_output
        and not arguments.nut_tooth_sync_capture_output
    ):
        parser.error(
            "--nut-tooth-ghost-fingers-output requires "
            "--nut-tooth-sync-capture-output"
        )
    if arguments.nut_tooth_ghost_fingers_output and (
        render_variant
        or arguments.nut_tooth_jitter_normalize_segment00_op
        or arguments.rewind_probe
        or arguments.full_rotation_probe
    ):
        parser.error(
            "tooth ghost v1 requires the baseline prepared twist capture "
            "without renderer/schema/rewind variants"
        )
    return arguments


def _nut_tooth_jitter_render_extra_args(arguments):
    """Return only explicit diagnostic Kit overrides; baseline is empty."""

    if arguments.nut_tooth_jitter_rtx_history is not None:
        return [
            f"--{RTX_HISTORY_SETTING}="
            f"{arguments.nut_tooth_jitter_rtx_history}"
        ]
    if arguments.nut_tooth_jitter_disable_fabric_scene_delegate:
        return [f"--{FABRIC_SCENE_DELEGATE_SETTING}=false"]
    return []


def _nut_tooth_jitter_render_settings_report(arguments, settings):
    """Read actual post-launch Kit settings for one controlled render A/B.

    RTX transform history and Fabric Scene Delegate both affect presentation,
    so changing them at runtime would not be an equivalent A/B.  They are
    passed through ``SimulationApp.extra_args`` and then read back here after
    Kit starts.  The caller must reject ``exact_match=false`` before building
    or stepping the physics scene.
    """

    requested = {}
    mode = "baseline"
    if arguments.nut_tooth_jitter_rtx_history is not None:
        mode = "rtx_history_512"
        requested[RTX_HISTORY_SETTING] = (
            arguments.nut_tooth_jitter_rtx_history
        )
    elif arguments.nut_tooth_jitter_disable_fabric_scene_delegate:
        mode = "fabric_scene_delegate_disabled"
        requested[FABRIC_SCENE_DELEGATE_SETTING] = False
    actual = {
        RTX_HISTORY_SETTING: settings.get(RTX_HISTORY_SETTING),
        FABRIC_SCENE_DELEGATE_SETTING: settings.get(
            FABRIC_SCENE_DELEGATE_SETTING
        ),
    }
    mismatches = []
    for path, expected in requested.items():
        observed = actual[path]
        # Exact type checking avoids accepting bool as integer or 0 as False.
        if type(observed) is not type(expected) or observed != expected:
            mismatches.append(
                {
                    "actual": observed,
                    "path": path,
                    "requested": expected,
                }
            )
    return {
        "actual": actual,
        "extra_args": _nut_tooth_jitter_render_extra_args(arguments),
        "exact_match": not mismatches,
        "mismatches": mismatches,
        "mode": mode,
        "requested": requested,
        "validated_after_simulation_app_start": True,
    }


def _require_exact_nut_tooth_jitter_render_settings(report):
    """Fail closed instead of silently running a mislabeled render A/B."""

    if report.get("exact_match") is not True:
        raise RuntimeError(
            "nut-tooth render A/B launch settings do not exactly match "
            f"request: {report.get('mismatches')}"
        )


def _finalize_tooth_probe(tooth_probe, output_directory, render_report):
    """Bind the verified render launch state into the probe's report.json."""

    report = dict(tooth_probe.finalize())
    report["render_ab_launch"] = render_report
    report_path = Path(output_directory).expanduser().resolve() / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main():
    repository = Path(__file__).resolve().parents[3]
    arguments = _parse_arguments(repository)
    render_extra_args = _nut_tooth_jitter_render_extra_args(arguments)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not arguments.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
            "extra_args": render_extra_args,
        }
    )
    passed = False
    tooth_probe = None
    tooth_sync_capture = None
    tooth_ghost_runtime = None
    metrics = {
        "assembly_success_claimed": False,
        "attachment": "none",
        "continuous_collision_verified": False,
        "gui": arguments.gui,
        "keep_open": arguments.keep_open,
        "object_drive": "none",
        "object_pose_writes_after_start": 0,
        "passed": False,
        "physical_initial_mixed_closure_included": True,
        "physical_insertion_included": False,
        "q7_twist_included": arguments.twist_probe,
        "q7_rewind_included": arguments.rewind_probe,
        "full_rotation_requested": arguments.full_rotation_probe,
        "real_keying_modeled": False,
        "scene": SCHEMA_VERSION,
        "self_collision_enabled_in_asset": False,
        "temporary_world_body_constraint": "fixed_joint",
        "thread_teeth_modeled": False,
    }
    try:
        import carb

        render_settings_report = _nut_tooth_jitter_render_settings_report(
            arguments, carb.settings.get_settings()
        )
        metrics["nut_tooth_jitter_render_ab"] = render_settings_report
        _require_exact_nut_tooth_jitter_render_settings(
            render_settings_report
        )
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
            UsdPhysics,
            UsdShade,
        )

        from kcg_connector.d38999_tabletop_pick import (
            load_d38999_tabletop_pick_config,
            verify_d38999_pick_dependencies,
        )
        from kcg_connector.d38999_tabletop_scene import (
            author_d38999_tabletop_scene,
        )
        from kcg_connector.d38999_twist_probe import (
            load_d38999_twist_probe_contract,
        )
        from kcg_connector.d38999_rewind_probe import (
            load_d38999_rewind_probe_contract,
        )
        from kcg_connector.d38999_full_rotation import (
            evaluate_d38999_full_rotation,
            validate_final_seating_contact_pairs,
        )
        from kcg_connector.robot_model import named_joint_target
        from d38999_nut_tooth_jitter_probe import (
            NutToothJitterProbe,
            colorize_segments_session,
            normalize_segment00_rotate_z_session,
        )
        from d38999_tooth_sync_capture import D38999ToothSyncCapture
        from d38999_tooth_ghost_runtime import D38999ToothGhostRuntime

        config_path = Path(arguments.config).expanduser().resolve()
        contract, inputs, cpu_report = _load_contract(
            config_path, repository
        )
        twist_contract = None
        twist_inputs = None
        rewind_contract = None
        rewind_inputs = None
        if arguments.twist_probe:
            twist_config_path = Path(
                arguments.twist_config
            ).expanduser().resolve()
            twist_contract, twist_inputs = load_d38999_twist_probe_contract(
                twist_config_path,
                repository=repository,
            )
            if twist_inputs["nut_regrasp_physx"] != config_path:
                raise RuntimeError(
                    "twist probe does not bind the active regrasp contract"
                )
            metrics["twist_contract"] = {
                "input_sha256": {
                    name: _sha256(path)
                    for name, path in sorted(twist_inputs.items())
                },
                "path": str(twist_config_path.relative_to(repository)),
                "probe_id": twist_contract["probe_id"],
                "schema_version": twist_contract["schema_version"],
                "sha256": _sha256(twist_config_path),
            }
        if arguments.rewind_probe:
            rewind_config_path = Path(
                arguments.rewind_config
            ).expanduser().resolve()
            rewind_contract, rewind_inputs = (
                load_d38999_rewind_probe_contract(
                    rewind_config_path,
                    repository=repository,
                )
            )
            if rewind_inputs["nut_regrasp_physx"] != config_path:
                raise RuntimeError(
                    "rewind probe does not bind the active regrasp contract"
                )
            if rewind_inputs["stage120_twist_contract"] != (
                twist_config_path
            ):
                raise RuntimeError(
                    "rewind probe requires the active stage120 contract"
                )
            metrics["rewind_contract"] = {
                "input_sha256": {
                    name: _sha256(path)
                    for name, path in sorted(rewind_inputs.items())
                },
                "path": str(rewind_config_path.relative_to(repository)),
                "schema_version": rewind_contract["schema_version"],
                "sha256": _sha256(rewind_config_path),
            }
        pick = load_d38999_tabletop_pick_config(inputs["tabletop_pick"])
        dependencies = verify_d38999_pick_dependencies(
            pick, inputs["tabletop_pick"], repository
        )
        tabletop = dependencies["tabletop"]
        d38999_asset = dependencies["d38999_asset"]
        if dependencies["robot_asset"] != inputs["robot_asset"]:
            raise RuntimeError("robot asset dependency mismatch")

        prepared = contract["prepared_engage"]
        candidate = contract["nut_only_candidate"]
        control = contract["control"]
        sensing = contract["sensing"]
        acceptance = contract["acceptance"]
        boundaries = contract["boundaries"]
        if (
            control["physics_rate_hz"] != tabletop.physics.rate_hz
            or sensing["hard_stop_nm"] != 2.0
            or sensing["operational_torque_target_nm"] != 1.8
            or sensing["fingertip_tactile_available"] is not False
            or boundaries["assembly_success_claimed"] is not False
        ):
            raise RuntimeError("PhysX A/B contract does not match v1")
        rate_hz = control["physics_rate_hz"]

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
            str(inputs["robot_asset"]), pick.scene.robot_root_prim_path
        )

        loose_prim = stage.GetPrimAtPath(tabletop.asset.loose_plug_prim_path)
        loose_ops = UsdGeom.Xformable(loose_prim).GetOrderedXformOps()
        if (
            len(loose_ops) != 1
            or loose_ops[0].GetOpType() != UsdGeom.XformOp.TypeTranslate
        ):
            raise RuntimeError("loose plug transform stack is not canonical")
        body_root_world = _finite_vector(
            prepared["body_root_world_m"], 3, "body_root_world_m"
        )
        loose_ops[0].Set(Gf.Vec3d(*body_root_world))
        metrics["object_pose_writes_before_start"] = 1

        constraint_path = "/World/D38999NutRegrasp/PreparedBodyConstraint"
        UsdGeom.Scope.Define(stage, "/World/D38999NutRegrasp")
        constraint = UsdPhysics.FixedJoint.Define(stage, constraint_path)
        constraint.CreateBody1Rel().SetTargets(
            [Sdf.Path(tabletop.asset.body_prim_path)]
        )
        constraint.CreateLocalPos0Attr(Gf.Vec3f(*body_root_world))
        constraint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        constraint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        constraint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        constraint.CreateCollisionEnabledAttr(False)

        grip_material_path = "/World/D38999NutRegrasp/GripMaterial"
        grip_material = UsdShade.Material.Define(stage, grip_material_path)
        grip_api = UsdPhysics.MaterialAPI.Apply(grip_material.GetPrim())
        grip_api.CreateStaticFrictionAttr(pick.motion.grip_static_friction)
        grip_api.CreateDynamicFrictionAttr(pick.motion.grip_dynamic_friction)
        grip_api.CreateRestitutionAttr(pick.motion.grip_restitution)

        robot_root = pick.scene.robot_root_prim_path
        plug_root = tabletop.asset.loose_plug_prim_path
        body_root = tabletop.asset.body_prim_path
        nut_root = tabletop.asset.nut_prim_path
        finger_collision_count = 0
        plug_collision_counts = {"body": 0, "nut": 0}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            finger_anchor = bool(
                path.startswith(robot_root + "/")
                and prim.GetName().endswith("_convex")
                and any(
                    name in path
                    for name in ("/f1Link", "/f2Link", "/f3Link")
                )
            )
            plug_collision = bool(
                prim.HasAPI(UsdPhysics.CollisionAPI)
                and path.startswith(plug_root + "/")
            )
            if not (finger_anchor or plug_collision):
                continue
            physicsUtils.add_physics_material_to_prim(
                stage, prim, Sdf.Path(grip_material_path)
            )
            if finger_anchor:
                finger_collision_count += 1
            else:
                group = _d38999_loose_collider_group(path, body_root, nut_root)
                if group is None:
                    raise RuntimeError("loose collider is outside body/nut")
                plug_collision_counts[group] += 1
        if finger_collision_count != 8 or plug_collision_counts != {
            "body": 21,
            "nut": 24,
        }:
            raise RuntimeError("D38999 contact geometry counts changed")

        filtered_pair_count = 0
        filtered_nut_segment_count = 0
        filtered_body_mating_segment_count = 0
        filtered_fixed_entry_segment_count = 0
        if arguments.twist_probe:
            thread_contract = twist_contract["runtime_thread"]
            fixed_entry_root = (
                tabletop.asset.fixed_receptacle_prim_path + "/EntryShell"
            )
            nut_segments = sorted(
                (
                    prim
                    for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith(nut_root + "/Segment_")
                    and prim.HasAPI(UsdPhysics.CollisionAPI)
                ),
                key=lambda prim: str(prim.GetPath()),
            )
            body_mating_root = body_root + "/MatingShell"
            body_mating_segments = sorted(
                (
                    prim
                    for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith(
                        body_mating_root + "/Segment_"
                    )
                    and prim.HasAPI(UsdPhysics.CollisionAPI)
                ),
                key=lambda prim: str(prim.GetPath()),
            )
            fixed_entry_segments = sorted(
                (
                    prim
                    for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith(
                        fixed_entry_root + "/Segment_"
                    )
                    and prim.HasAPI(UsdPhysics.CollisionAPI)
                ),
                key=lambda prim: str(prim.GetPath()),
            )
            filtered_nut_segment_count = len(nut_segments)
            filtered_body_mating_segment_count = len(body_mating_segments)
            filtered_fixed_entry_segment_count = len(fixed_entry_segments)
            if (
                filtered_nut_segment_count
                != thread_contract["expected_nut_segment_count"]
                or filtered_body_mating_segment_count
                != thread_contract[
                    "expected_body_mating_segment_count"
                ]
                or filtered_fixed_entry_segment_count
                != thread_contract[
                    "expected_fixed_entry_segment_count"
                ]
            ):
                raise RuntimeError(
                    "D38999 thread proxy collision segment counts changed"
                )
        metrics["proxy_collision_filter"] = {
            "body_mating_segment_count": (
                filtered_body_mating_segment_count
            ),
            "enabled": False,
            "fixed_entry_segment_count": filtered_fixed_entry_segment_count,
            "mode": (
                twist_contract["runtime_thread"]["filtered_pair_mode"]
                if arguments.twist_probe
                else "none"
            ),
            "nut_segment_count": filtered_nut_segment_count,
            "pair_count": filtered_pair_count,
        }

        contact_report_body_count = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            is_robot_body = bool(
                path.startswith(robot_root + "/")
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            is_endpoint_body = path in (body_root, nut_root)
            if is_robot_body or is_endpoint_body:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                report_api.CreateThresholdAttr().Set(0.0)
                if is_robot_body:
                    contact_report_body_count += 1
        if contact_report_body_count < 17:
            raise RuntimeError("robot contact reporting is incomplete")

        tooth_normalization_report = None
        tooth_color_report = None
        if arguments.nut_tooth_jitter_output:
            # These optional visual/schema overrides live only in the stage's
            # anonymous session layer and are authored before world.reset().
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

        if arguments.nut_tooth_ghost_fingers_output:
            # Presentation-only occlusion control.  This exact source order is
            # tested: the three inherited visibility opinions are authored
            # after the robot reference exists but before world.reset/play.
            # The helper has no transform, PhysX, collision or material API.
            tooth_ghost_runtime = D38999ToothGhostRuntime(
                stage=stage,
                robot_root=robot_root,
                output_directory=(
                    arguments.nut_tooth_ghost_fingers_output
                ),
                runner_source_path=Path(__file__).resolve(),
                Sdf=Sdf,
                UsdGeom=UsdGeom,
            )
            metrics["nut_tooth_ghost_fingers"] = (
                tooth_ghost_runtime.active_report()
            )

        if arguments.gui:
            from isaacsim.core.rendering_manager import ViewportManager
            from pxr import UsdLux

            lighting_root = "/World/D38999NutRegrasp/GuiLighting"
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

        if arguments.nut_tooth_sync_capture_output:
            # This camera/render product is presentation-only and fixed before
            # world.reset/play.  The capture class never advances physics.
            from PIL import Image

            import omni.replicator.core as rep

            tooth_sync_capture = D38999ToothSyncCapture(
                output_directory=arguments.nut_tooth_sync_capture_output,
                physics_rate_hz=rate_hz,
                bindings={
                    "Gf": Gf,
                    "Image": Image,
                    "UsdGeom": UsdGeom,
                    "rep": rep,
                },
                stage=stage,
                render_settings=render_settings_report,
            )

        robot = world.scene.add(
            SingleArticulation(
                prim_path=pick.scene.articulation_prim_path,
                name="d38999_nut_regrasp_handarm",
            )
        )
        body = world.scene.add(
            SingleRigidPrim(
                prim_path=body_root, name="d38999_nut_regrasp_body"
            )
        )
        nut = world.scene.add(
            SingleRigidPrim(
                prim_path=nut_root, name="d38999_nut_regrasp_nut"
            )
        )
        fixed_prim = stage.GetPrimAtPath(
            tabletop.asset.fixed_receptacle_prim_path
        )
        tcp_prim = stage.GetPrimAtPath(pick.scene.grasp_tcp_prim_path)
        world.reset()
        world.get_physics_context().set_gravity(tabletop.physics.gravity_m_s2)
        if not robot.handles_initialized:
            raise RuntimeError(
                "robot articulation handles were not initialized"
            )
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
            [name_to_index[name] for name in sensing["torque_joint_names"]],
            dtype=np.int32,
        )
        controlled_indices = np.concatenate((arm_indices, hand_indices))
        carry_arm = np.asarray(prepared["carry_arm_rad"], dtype=np.float64)
        carry_hand = np.asarray(prepared["carry_hand_rad"], dtype=np.float64)
        open_hand = np.asarray(prepared["open_hand_rad"], dtype=np.float64)
        nut_arm = np.asarray(candidate["arm_rad"], dtype=np.float64)
        nut_hand = np.asarray(candidate["hand_rad"], dtype=np.float64)

        initial_positions = named_joint_target(
            dof_names, carry_arm, open_hand
        ).astype(np.float32)
        robot.set_joint_positions(initial_positions)
        robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))
        metrics["initial_joint_position_error_before_first_step_rad"] = float(
            np.max(
                np.abs(
                    np.asarray(robot.get_joint_positions(), dtype=np.float64)
                    - initial_positions.astype(np.float64)
                )
            )
        )
        controller = robot.get_articulation_controller()
        kps = np.zeros(robot.num_dof, dtype=np.float32)
        kds = np.zeros(robot.num_dof, dtype=np.float32)
        kps[arm_indices] = control["arm_stiffness"]
        kds[arm_indices] = control["arm_damping"]
        kps[hand_indices] = control["open_hand_stiffness"]
        kds[hand_indices] = control["open_hand_damping"]
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)

        def set_hand_gains(stiffness, damping):
            kps[hand_indices] = stiffness
            kds[hand_indices] = damping
            controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        dof_properties = robot.dof_properties

        initial_body_position, initial_body_orientation = body.get_world_pose()
        initial_nut_position, _ = nut.get_world_pose()
        initial_body_nut_separation = float(
            np.linalg.norm(initial_nut_position - initial_body_position)
        )
        fixed_initial_position, fixed_initial_orientation = _world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )

        phase = "not_started"
        global_step = 0
        phase_steps = {}
        finite_throughout = True
        maximum_joint_limit_violation = 0.0
        maximum_joint_speed = 0.0
        maximum_arm_tracking_error = 0.0
        maximum_body_translation_drift = 0.0
        maximum_body_rotation_drift = 0.0
        maximum_post_tare_by_channel = np.zeros(3, dtype=np.float64)
        external_contact_records = {
            "table": 0,
            "fixture": 0,
            "fixed_endpoint": 0,
            "endpoint_nonfinger": 0,
            "body_forbidden_nut_only": 0,
            "endpoint_forbidden_clear": 0,
            "endpoint_allowed": 0,
        }
        hard_torque_violation = None

        def contact_snapshot():
            headers, contacts, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            result = {
                "finger_body_group_records": {
                    finger: {"body": 0, "nut": 0}
                    for finger in ("f1", "f2", "f3")
                },
                "finger_endpoint_records": 0,
                "grip_material_records": 0,
                "nonfinger_endpoint_records": 0,
            }
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
                    plug_root,
                )
                if category != "loose_plug":
                    continue
                if not _is_finger_plug_contact(paths, robot_root, plug_root):
                    result["nonfinger_endpoint_records"] += int(
                        header.num_contact_data
                    )
                    continue
                group = _finger_loose_contact_group(
                    paths, robot_root, body_root, nut_root
                )
                if group is None:
                    raise RuntimeError(
                        "finger endpoint contact is unclassified"
                    )
                finger, endpoint = group
                result["finger_body_group_records"][finger][endpoint] += int(
                    header.num_contact_data
                )
                result["finger_endpoint_records"] += int(
                    header.num_contact_data
                )
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
                    if materials == (grip_material_path, grip_material_path):
                        result["grip_material_records"] += 1
            return result

        current_arm_target = carry_arm.copy()
        current_hand_target = open_hand.copy()

        def observe_and_step(contact_mode):
            nonlocal finite_throughout
            nonlocal global_step
            nonlocal maximum_arm_tracking_error
            nonlocal maximum_body_rotation_drift
            nonlocal maximum_body_translation_drift
            nonlocal maximum_joint_limit_violation
            nonlocal maximum_joint_speed
            target = np.concatenate(
                (current_arm_target, current_hand_target)
            ).astype(np.float32)
            robot.apply_action(
                ArticulationAction(
                    joint_positions=target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            global_step += 1
            phase_steps[phase] = phase_steps.get(phase, 0) + 1
            positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            body_position, body_orientation = body.get_world_pose()
            nut_position, nut_orientation = nut.get_world_pose()
            sampled = np.concatenate(
                (
                    positions,
                    velocities,
                    body_position,
                    body_orientation,
                    nut_position,
                    nut_orientation,
                    body.get_linear_velocity(),
                    body.get_angular_velocity(),
                    nut.get_linear_velocity(),
                    nut.get_angular_velocity(),
                )
            )
            sample_finite = bool(np.all(np.isfinite(sampled)))
            finite_throughout = bool(finite_throughout and sample_finite)
            if not sample_finite:
                raise RuntimeError(f"non-finite physics state in {phase}")
            maximum_joint_speed = max(
                maximum_joint_speed, float(np.max(np.abs(velocities)))
            )
            maximum_arm_tracking_error = max(
                maximum_arm_tracking_error,
                float(
                    np.max(
                        np.abs(positions[arm_indices] - current_arm_target)
                    )
                ),
            )
            for index in range(robot.num_dof):
                if bool(dof_properties[index]["hasLimits"]):
                    lower = float(dof_properties[index]["lower"])
                    upper = float(dof_properties[index]["upper"])
                    maximum_joint_limit_violation = max(
                        maximum_joint_limit_violation,
                        lower - float(positions[index]),
                        float(positions[index]) - upper,
                    )
            maximum_body_translation_drift = max(
                maximum_body_translation_drift,
                float(np.linalg.norm(body_position - initial_body_position)),
            )
            maximum_body_rotation_drift = max(
                maximum_body_rotation_drift,
                _array_quaternion_error_radians(
                    initial_body_orientation, body_orientation
                ),
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
                    plug_root,
                )
                if category is None:
                    continue
                records = int(header.num_contact_data)
                if records <= 0:
                    continue
                if category in ("table", "fixture", "fixed_endpoint"):
                    external_contact_records[category] += records
                    tcp_at_contact, _ = _world_pose(
                        Gf, Usd, UsdGeom, tcp_prim
                    )
                    metrics["first_forbidden_contact"] = {
                        "category": category,
                        "global_step": global_step,
                        "hand_joint_positions_rad": {
                            name: float(positions[index])
                            for name, index in zip(
                                pick.robot.active_hand_joint_names,
                                hand_indices,
                            )
                        },
                        "hand_joint_targets_rad": dict(
                            zip(
                                pick.robot.active_hand_joint_names,
                                current_hand_target,
                            )
                        ),
                        "hand_joint_velocities_rad_s": {
                            name: float(velocities[index])
                            for name, index in zip(
                                pick.robot.active_hand_joint_names,
                                hand_indices,
                            )
                        },
                        "maximum_hand_tracking_error_rad": float(
                            np.max(
                                np.abs(
                                    positions[hand_indices]
                                    - current_hand_target
                                )
                            )
                        ),
                        "paths": list(paths),
                        "phase": phase,
                        "phase_step": phase_steps[phase],
                        "records": records,
                        "tcp_position_world_m": list(tcp_at_contact),
                    }
                    raise RuntimeError(
                        f"forbidden robot {category} contact in {phase}"
                    )
                if not _is_finger_plug_contact(paths, robot_root, plug_root):
                    external_contact_records["endpoint_nonfinger"] += records
                    metrics["first_forbidden_contact"] = {
                        "category": "endpoint_nonfinger",
                        "global_step": global_step,
                        "paths": list(paths),
                        "phase": phase,
                        "phase_step": phase_steps[phase],
                        "records": records,
                    }
                    raise RuntimeError(
                        f"forbidden nonfinger endpoint contact in {phase}"
                    )
                group = _finger_loose_contact_group(
                    paths, robot_root, body_root, nut_root
                )
                if group is None:
                    raise RuntimeError("endpoint contact group is ambiguous")
                _, endpoint = group
                if contact_mode == "clear":
                    external_contact_records[
                        "endpoint_forbidden_clear"
                    ] += records
                    metrics["first_forbidden_contact"] = {
                        "category": "endpoint_forbidden_clear",
                        "global_step": global_step,
                        "paths": list(paths),
                        "phase": phase,
                        "phase_step": phase_steps[phase],
                        "records": records,
                    }
                    raise RuntimeError(
                        f"endpoint contact forbidden while open in {phase}"
                    )
                if contact_mode == "nut_only" and endpoint == "body":
                    external_contact_records[
                        "body_forbidden_nut_only"
                    ] += records
                    metrics["first_forbidden_contact"] = {
                        "category": "body_forbidden_nut_only",
                        "global_step": global_step,
                        "paths": list(paths),
                        "phase": phase,
                        "phase_step": phase_steps[phase],
                        "records": records,
                    }
                    raise RuntimeError(
                        f"finger BodyAssembly contact forbidden in {phase}"
                    )
                external_contact_records["endpoint_allowed"] += records
            if tooth_probe is not None:
                # Sampling occurs after the physics step and its contact
                # report, exactly once per 240 Hz step in the prepared scene.
                tooth_probe.sample(
                    global_step=global_step,
                    phase=phase,
                    phase_step=phase_steps[phase],
                    contact_report=(headers, contacts, friction_anchors),
                )
            if tooth_sync_capture is not None:
                # Capture reads the frame presented by the world.step above,
                # then binds it to this exact completed 240 Hz physics step.
                tooth_sync_capture.maybe_capture(
                    global_step=global_step,
                    phase=phase,
                    phase_step=phase_steps[phase],
                )
            return positions, velocities

        def sample_efforts(tare):
            nonlocal hard_torque_violation
            measured = np.asarray(
                robot.get_measured_joint_efforts(
                    joint_indices=sensor_indices
                ),
                dtype=np.float64,
            )
            delta = measured - tare
            if not np.all(np.isfinite(delta)):
                raise RuntimeError("non-finite finger-base torque")
            np.maximum(
                maximum_post_tare_by_channel,
                np.abs(delta),
                out=maximum_post_tare_by_channel,
            )
            if np.any(np.abs(delta) > sensing["hard_stop_nm"]):
                hard_torque_violation = {
                    name: float(value)
                    for name, value in zip(
                        sensing["torque_joint_names"], delta
                    )
                }
                raise RuntimeError(
                    f"finger-base torque exceeded 2 Nm in {phase}"
                )
            return measured

        def run_hold(duration, contact_mode, tare=None, samples=None):
            steps = round(duration * rate_hz)
            for _ in range(steps):
                observe_and_step(contact_mode)
                if tare is not None:
                    measured = sample_efforts(tare)
                    if samples is not None:
                        samples.append(measured.copy())

        phase = "initial_open_settle"
        run_hold(control["initial_open_settle_s"], "clear")
        phase = "initial_open_tare"
        tare_samples = []
        for _ in range(round(control["open_tare_s"] * rate_hz)):
            observe_and_step("clear")
            tare_samples.append(
                np.asarray(
                    robot.get_measured_joint_efforts(
                        joint_indices=sensor_indices
                    ),
                    dtype=np.float64,
                )
            )
        tare_mixed = np.mean(np.stack(tare_samples), axis=0)

        phase = "prepare_mixed_carry_grip"
        start_hand = current_hand_target.copy()
        steps = round(control["mixed_closure_s"] * rate_hz)
        for index in range(steps):
            current_hand_target = _interpolate(
                start_hand, carry_hand, float(index + 1) / float(steps)
            )
            observe_and_step("mixed")
            sample_efforts(tare_mixed)
        current_hand_target = carry_hand.copy()
        set_hand_gains(
            control["grip_hand_stiffness"],
            control["grip_hand_damping"],
        )
        phase = "mixed_preload"
        run_hold(
            control["mixed_preload_s"], "mixed", tare=tare_mixed
        )
        mixed_snapshot = contact_snapshot()

        phase = "release_mixed_grip"
        start_hand = current_hand_target.copy()
        steps = round(control["release_s"] * rate_hz)
        for index in range(steps):
            current_hand_target = _interpolate(
                start_hand, open_hand, float(index + 1) / float(steps)
            )
            observe_and_step("mixed")
            sample_efforts(tare_mixed)
        current_hand_target = open_hand.copy()

        set_hand_gains(
            control["open_hand_stiffness"],
            control["open_hand_damping"],
        )

        phase = "open_reposition_to_nut_band"
        start_arm = current_arm_target.copy()
        steps = round(control["open_reposition_s"] * rate_hz)
        for index in range(steps):
            current_arm_target = _interpolate(
                start_arm, nut_arm, float(index + 1) / float(steps)
            )
            observe_and_step("clear")
            sample_efforts(tare_mixed)
        current_arm_target = nut_arm.copy()
        open_reposition_snapshot = contact_snapshot()

        phase = "second_open_tare"
        second_tare_samples = []
        for _ in range(round(control["second_open_tare_s"] * rate_hz)):
            observe_and_step("clear")
            second_tare_samples.append(
                np.asarray(
                    robot.get_measured_joint_efforts(
                        joint_indices=sensor_indices
                    ),
                    dtype=np.float64,
                )
            )
        tare_nut = np.mean(np.stack(second_tare_samples), axis=0)

        set_hand_gains(
            control["grip_hand_stiffness"],
            control["grip_hand_damping"],
        )

        tcp_position, tcp_orientation = _world_pose(
            Gf, Usd, UsdGeom, tcp_prim
        )
        regrasp_joint_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        metrics["nut_regrasp_arm_positions_rad"] = {
            name: float(regrasp_joint_positions[index])
            for name, index in zip(
                pick.robot.arm_joint_names, arm_indices
            )
        }
        metrics["nut_regrasp_hand_positions_rad"] = {
            name: float(regrasp_joint_positions[index])
            for name, index in zip(
                pick.robot.active_hand_joint_names, hand_indices
            )
        }
        metrics["nut_regrasp_tcp_position_world_m"] = list(tcp_position)
        tcp_position_error = float(
            np.linalg.norm(
                np.asarray(tcp_position, dtype=np.float64)
                - np.asarray(candidate["tcp_position_world_m"])
            )
        )
        desired_physical_tcp_position_error = float(
            np.linalg.norm(
                np.asarray(tcp_position, dtype=np.float64)
                - np.asarray(
                    candidate["desired_physical_tcp_position_world_m"],
                    dtype=np.float64,
                )
            )
        )
        tcp_axis_error = _axis_error_radians(
            _quaternion_world_z_axis(tcp_orientation),
            candidate["tcp_down_axis_world"],
        )

        phase = "nut_only_closure"
        start_hand = current_hand_target.copy()
        steps = round(control["nut_closure_s"] * rate_hz)
        for index in range(steps):
            current_hand_target = _interpolate(
                start_hand, nut_hand, float(index + 1) / float(steps)
            )
            observe_and_step("nut_only")
            sample_efforts(tare_nut)
        current_hand_target = nut_hand.copy()

        phase = "nut_only_preload"
        preload_samples = []
        run_hold(
            control["nut_preload_s"],
            "nut_only",
            tare=tare_nut,
            samples=preload_samples,
        )
        postclosure_snapshot = contact_snapshot()
        preload_efforts = np.mean(np.stack(preload_samples), axis=0)
        preload_deltas = preload_efforts - tare_nut
        preload_loaded_channels = int(
            np.count_nonzero(
                np.abs(preload_deltas)
                >= sensing["loaded_torque_threshold_nm"]
            )
        )

        phase = "nut_only_final_hold"
        final_effort_samples = []
        tail_steps = min(120, round(control["final_hold_s"] * rate_hz))
        tail_joint_solver_speeds = []
        tail_joint_observable_speeds = []
        tail_nut_solver_speeds = []
        tail_nut_observable_speeds = []
        previous_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        _, previous_nut_orientation = nut.get_world_pose()
        final_hold_steps = round(control["final_hold_s"] * rate_hz)
        effort_sample_steps = round(control["effort_sample_s"] * rate_hz)
        for index in range(final_hold_steps):
            observe_and_step("nut_only")
            measured = sample_efforts(tare_nut)
            positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            _, nut_orientation = nut.get_world_pose()
            if index >= final_hold_steps - tail_steps:
                tail_joint_solver_speeds.append(
                    float(np.max(np.abs(velocities)))
                )
                tail_joint_observable_speeds.append(
                    float(np.max(np.abs(positions - previous_positions)))
                    * rate_hz
                )
                tail_nut_solver_speeds.append(
                    float(np.linalg.norm(nut.get_angular_velocity()))
                )
                tail_nut_observable_speeds.append(
                    _array_quaternion_error_radians(
                        previous_nut_orientation, nut_orientation
                    )
                    * rate_hz
                )
            previous_positions = positions
            previous_nut_orientation = nut_orientation
            if index >= final_hold_steps - effort_sample_steps:
                final_effort_samples.append(measured.copy())

        final_snapshot = contact_snapshot()
        final_efforts = np.mean(np.stack(final_effort_samples), axis=0)
        final_deltas = final_efforts - tare_nut
        final_loaded_channels = int(
            np.count_nonzero(
                np.abs(final_deltas)
                >= sensing["loaded_torque_threshold_nm"]
            )
        )
        final_body_position, final_body_orientation = body.get_world_pose()
        final_nut_position, _ = nut.get_world_pose()
        body_nut_separation_change = abs(
            float(
                np.linalg.norm(final_nut_position - final_body_position)
                - initial_body_nut_separation
            )
        )
        fixed_final_position, fixed_final_orientation = _world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )
        fixed_translation_drift = float(
            np.linalg.norm(
                np.asarray(fixed_final_position)
                - np.asarray(fixed_initial_position)
            )
        )
        fixed_rotation_drift = _gf_quaternion_error_radians(
            fixed_initial_orientation, fixed_final_orientation
        )
        final_body_translation_drift = float(
            np.linalg.norm(final_body_position - initial_body_position)
        )
        final_body_rotation_drift = _array_quaternion_error_radians(
            initial_body_orientation, final_body_orientation
        )
        final_nut_observable_speed = max(tail_nut_observable_speeds)
        final_nut_solver_speed = max(tail_nut_solver_speeds)
        final_observable_joint_speed = max(tail_joint_observable_speeds)
        final_post_solver_joint_speed = max(tail_joint_solver_speeds)

        mixed_gate = _all_fingers_endpoint_contact(mixed_snapshot)
        open_gate = _zero_endpoint_contact(open_reposition_snapshot)
        post_all_nut = _all_fingers_nut_contact(postclosure_snapshot)
        post_zero_body = _zero_finger_body_contact(postclosure_snapshot)
        final_all_nut = _all_fingers_nut_contact(final_snapshot)
        final_zero_body = _zero_finger_body_contact(final_snapshot)
        operational_target_exceeded = bool(
            np.any(
                maximum_post_tare_by_channel
                > sensing["operational_torque_target_nm"]
            )
        )
        torque_gate = bool(
            hard_torque_violation is None
            and np.max(maximum_post_tare_by_channel)
            <= sensing["hard_stop_nm"]
            and preload_loaded_channels >= sensing["minimum_loaded_channels"]
            and final_loaded_channels >= sensing["minimum_loaded_channels"]
        )
        contact_gate = bool(
            mixed_gate
            and open_gate
            and post_all_nut
            and post_zero_body
            and final_all_nut
            and final_zero_body
            and postclosure_snapshot["nonfinger_endpoint_records"] == 0
            and final_snapshot["nonfinger_endpoint_records"] == 0
        )
        zero_forbidden_contacts = bool(
            all(
                external_contact_records[key] == 0
                for key in (
                    "table",
                    "fixture",
                    "fixed_endpoint",
                    "endpoint_nonfinger",
                    "body_forbidden_nut_only",
                    "endpoint_forbidden_clear",
                )
            )
        )
        regrasp_passed = bool(
            finite_throughout
            and maximum_joint_limit_violation
            <= acceptance["maximum_joint_limit_violation_rad"]
            and maximum_joint_speed
            <= acceptance["maximum_joint_speed_rad_s"]
            and maximum_arm_tracking_error
            <= acceptance["maximum_arm_tracking_error_rad"]
            and tcp_position_error
            <= acceptance["maximum_tcp_position_error_m"]
            and desired_physical_tcp_position_error
            <= acceptance[
                "maximum_desired_physical_tcp_position_error_m"
            ]
            and tcp_axis_error <= acceptance["maximum_tcp_axis_error_rad"]
            and maximum_body_translation_drift
            <= acceptance["maximum_body_translation_drift_m"]
            and maximum_body_rotation_drift
            <= acceptance["maximum_body_rotation_drift_rad"]
            and body_nut_separation_change
            <= acceptance["maximum_body_nut_separation_change_m"]
            and final_observable_joint_speed
            <= acceptance["maximum_final_observable_joint_speed_rad_s"]
            and final_post_solver_joint_speed
            <= acceptance["maximum_final_post_solver_joint_speed_rad_s"]
            and final_nut_observable_speed
            <= acceptance[
                "maximum_final_nut_observable_angular_speed_rad_s"
            ]
            and final_nut_solver_speed
            <= acceptance[
                "maximum_final_nut_post_solver_angular_speed_rad_s"
            ]
            and fixed_translation_drift == 0.0
            and fixed_rotation_drift == 0.0
            and contact_gate
            and torque_gate
            and zero_forbidden_contacts
            and metrics["object_pose_writes_after_start"] == 0
        )

        twist_passed = None
        twist_metrics = None
        rewind_passed = None
        rewind_metrics = None
        if arguments.twist_probe:
            thread = twist_contract["runtime_thread"]
            probe = twist_contract["probe"]
            twist_sensing = twist_contract["sensing"]
            twist_acceptance = twist_contract["acceptance"]
            if not regrasp_passed:
                raise RuntimeError(
                    "nut-only regrasp must pass before the q7 probe"
                )
            if (
                twist_sensing["hard_stop_nm"] != sensing["hard_stop_nm"]
                or twist_sensing["operational_torque_target_nm"]
                != sensing["operational_torque_target_nm"]
                or twist_sensing["torque_joint_names"]
                != sensing["torque_joint_names"]
            ):
                raise RuntimeError("twist and regrasp sensing differ")

            hinge_prim = stage.GetPrimAtPath(tabletop.asset.joint_prim_path)
            if not hinge_prim or not hinge_prim.IsA(
                UsdPhysics.RevoluteJoint
            ):
                raise RuntimeError("D38999 passive nut hinge is missing")
            hinge_drives = [
                str(schema)
                for schema in hinge_prim.GetAppliedSchemas()
                if str(schema).startswith("PhysicsDriveAPI")
            ]
            if hinge_drives:
                raise RuntimeError("D38999 passive nut hinge has a drive")

            phase = "twist_constraint_activation"
            activation_before_position, activation_before_orientation = (
                body.get_world_pose()
            )
            world.pause()
            for nut_segment in nut_segments:
                filtered_api = UsdPhysics.FilteredPairsAPI.Apply(nut_segment)
                relation = filtered_api.CreateFilteredPairsRel()
                for fixed_segment in fixed_entry_segments:
                    relation.AddTarget(fixed_segment.GetPath())
                    filtered_pair_count += 1
            for body_segment, fixed_segment in zip(
                body_mating_segments, fixed_entry_segments
            ):
                if body_segment.GetName() != fixed_segment.GetName():
                    raise RuntimeError(
                        "D38999 matching ring segment identities changed"
                    )
                filtered_api = UsdPhysics.FilteredPairsAPI.Apply(body_segment)
                filtered_api.CreateFilteredPairsRel().AddTarget(
                    fixed_segment.GetPath()
                )
                filtered_pair_count += 1
            if filtered_pair_count != thread[
                "expected_filtered_pair_count"
            ]:
                raise RuntimeError(
                    "D38999 proxy collision filter is incomplete"
                )
            metrics["proxy_collision_filter"].update(
                {"enabled": True, "pair_count": filtered_pair_count}
            )
            if not stage.RemovePrim(constraint_path):
                raise RuntimeError("temporary body constraint removal failed")
            UsdGeom.Scope.Define(stage, thread["root_prim_path"])
            prismatic = UsdPhysics.PrismaticJoint.Define(
                stage, thread["prismatic_prim_path"]
            )
            prismatic.CreateAxisAttr(thread["prismatic_axis"])
            prismatic.CreateBody1Rel().SetTargets([Sdf.Path(body_root)])
            activation_position_values = [
                float(value) for value in activation_before_position
            ]
            prismatic.CreateLocalPos0Attr(
                Gf.Vec3f(*activation_position_values)
            )
            prismatic.CreateLocalRot0Attr(
                Gf.Quatf(
                    float(activation_before_orientation[0]),
                    Gf.Vec3f(
                        float(activation_before_orientation[1]),
                        float(activation_before_orientation[2]),
                        float(activation_before_orientation[3]),
                    ),
                )
            )
            prismatic.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            prismatic.CreateLocalRot1Attr(Gf.Quatf(1.0))
            prismatic.CreateLowerLimitAttr(thread["lower_limit_m"])
            prismatic.CreateUpperLimitAttr(thread["upper_limit_m"])
            prismatic.CreateCollisionEnabledAttr(False)
            rack = PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(
                stage, thread["rack_prim_path"]
            )
            rack.CreateBody0Rel().SetTargets([Sdf.Path(nut_root)])
            rack.CreateBody1Rel().SetTargets([Sdf.Path(body_root)])
            rack.CreateHingeRel().SetTargets(
                [Sdf.Path(tabletop.asset.joint_prim_path)]
            )
            rack.CreatePrismaticRel().SetTargets(
                [Sdf.Path(thread["prismatic_prim_path"])]
            )
            rack.CreateRatioAttr(thread["rack_ratio_degrees_per_meter"])
            world.play()
            simulation_app.update()
            observe_and_step("nut_only")
            sample_efforts(tare_nut)
            activation_after_position, activation_after_orientation = (
                body.get_world_pose()
            )
            activation_translation_jump = float(
                np.linalg.norm(
                    activation_after_position - activation_before_position
                )
            )
            activation_rotation_jump = _array_quaternion_error_radians(
                activation_before_orientation, activation_after_orientation
            )
            for _ in range(9):
                observe_and_step("nut_only")
                sample_efforts(tare_nut)

            q7_index = name_to_index[probe["q7_joint_name"]]
            q7_arm_offset = pick.robot.arm_joint_names.index(
                probe["q7_joint_name"]
            )
            pre_twist_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            pre_twist_body_position, pre_twist_body_orientation = (
                body.get_world_pose()
            )
            pre_twist_fixed_position, pre_twist_fixed_orientation = (
                _world_pose(Gf, Usd, UsdGeom, fixed_prim)
            )
            body_prim = stage.GetPrimAtPath(body_root)
            nut_prim = stage.GetPrimAtPath(nut_root)
            motion_start_nut_angle = _wrapped_relative_z_angle(
                Gf, Usd, UsdGeom, body_prim, nut_prim
            )
            unwrapped_nut_angle = motion_start_nut_angle
            q7_command_start = float(current_arm_target[q7_arm_offset])
            q7_command_target = q7_command_start + probe["q7_delta_rad"]
            q7_properties = dof_properties[q7_index]
            if bool(q7_properties["hasLimits"]) and not (
                float(q7_properties["lower"])
                <= q7_command_target
                <= float(q7_properties["upper"])
            ):
                raise RuntimeError("q7 twist target is outside joint limits")

            loose_fixed_contact_records = 0
            loose_fixed_contact_pair_records = {}

            def count_loose_fixed_contacts():
                records = 0
                headers, _, _ = (
                    get_physx_simulation_interface().get_full_contact_report()
                )
                fixed_root = tabletop.asset.fixed_receptacle_prim_path
                for header in headers:
                    paths = tuple(
                        str(PhysicsSchemaTools.intToSdfPath(value))
                        for value in (
                            header.actor0,
                            header.actor1,
                            header.collider0,
                            header.collider1,
                        )
                    )
                    has_loose = any(
                        path == plug_root or path.startswith(plug_root + "/")
                        for path in paths
                    )
                    has_fixed = any(
                        path == fixed_root or path.startswith(fixed_root + "/")
                        for path in paths
                    )
                    if has_loose and has_fixed:
                        count = int(header.num_contact_data)
                        records += count
                        collider_pair = tuple(sorted((paths[2], paths[3])))
                        key = " <-> ".join(collider_pair)
                        loose_fixed_contact_pair_records[key] = (
                            loose_fixed_contact_pair_records.get(key, 0)
                            + count
                        )
                return records

            phase = "q7_twist_probe_motion"
            twist_steps = round(probe["motion_duration_s"] * rate_hz)
            for index in range(twist_steps):
                fraction = float(index + 1) / float(twist_steps)
                current_arm_target[q7_arm_offset] = (
                    q7_command_start
                    + _minimum_jerk(fraction) * probe["q7_delta_rad"]
                )
                observe_and_step("nut_only")
                sample_efforts(tare_nut)
                loose_fixed_contact_records += count_loose_fixed_contacts()
                wrapped = _wrapped_relative_z_angle(
                    Gf, Usd, UsdGeom, body_prim, nut_prim
                )
                unwrapped_nut_angle = _unwrap(unwrapped_nut_angle, wrapped)
            current_arm_target[q7_arm_offset] = q7_command_target

            phase = "q7_twist_probe_hold"
            hold_steps = round(probe["total_hold_duration_s"] * rate_hz)
            hold_evaluation_steps = round(
                probe["hold_evaluation_duration_s"] * rate_hz
            )
            hold_q7_observable_speeds = []
            hold_nut_observable_speeds = []
            hold_axial_observable_speeds = []
            hold_joint_solver_speeds = []
            hold_nut_solver_speeds = []
            hold_body_z_positions = []
            previous_twist_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            previous_twist_body_position, _ = body.get_world_pose()
            hold_start_body_position = previous_twist_body_position.copy()
            hold_body_z_positions.append(float(hold_start_body_position[2]))
            previous_twist_nut_angle = unwrapped_nut_angle
            twist_effort_samples = []
            for _ in range(hold_steps):
                observe_and_step("nut_only")
                measured = sample_efforts(tare_nut)
                twist_effort_samples.append(measured.copy())
                loose_fixed_contact_records += count_loose_fixed_contacts()
                positions = np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                )
                wrapped = _wrapped_relative_z_angle(
                    Gf, Usd, UsdGeom, body_prim, nut_prim
                )
                unwrapped_nut_angle = _unwrap(unwrapped_nut_angle, wrapped)
                body_position, _ = body.get_world_pose()
                hold_body_z_positions.append(float(body_position[2]))
                hold_q7_observable_speeds.append(
                    abs(
                        float(
                            positions[q7_index]
                            - previous_twist_positions[q7_index]
                        )
                    )
                    * rate_hz
                )
                hold_nut_observable_speeds.append(
                    abs(unwrapped_nut_angle - previous_twist_nut_angle)
                    * rate_hz
                )
                hold_axial_observable_speeds.append(
                    abs(
                        float(
                            body_position[2]
                            - previous_twist_body_position[2]
                        )
                    )
                    * rate_hz
                )
                hold_joint_solver_speeds.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    robot.get_joint_velocities(),
                                    dtype=np.float64,
                                )
                            )
                        )
                    )
                )
                hold_nut_solver_speeds.append(
                    float(np.linalg.norm(nut.get_angular_velocity()))
                )
                previous_twist_positions = positions
                previous_twist_body_position = body_position
                previous_twist_nut_angle = unwrapped_nut_angle

            twist_final_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            twist_final_body_position, twist_final_body_orientation = (
                body.get_world_pose()
            )
            twist_final_fixed_position, twist_final_fixed_orientation = (
                _world_pose(Gf, Usd, UsdGeom, fixed_prim)
            )
            twist_final_efforts = np.mean(
                np.stack(twist_effort_samples), axis=0
            )
            twist_final_deltas = twist_final_efforts - tare_nut
            twist_loaded_channels = int(
                np.count_nonzero(
                    np.abs(twist_final_deltas)
                    >= twist_sensing["loaded_torque_threshold_nm"]
                )
            )
            q7_delta = float(
                twist_final_positions[q7_index]
                - pre_twist_positions[q7_index]
            )
            nut_delta = float(
                unwrapped_nut_angle - motion_start_nut_angle
            )
            axial_travel = float(
                twist_final_body_position[2] - pre_twist_body_position[2]
            )
            expected_axial_from_nut = float(
                -probe["lead_m_per_revolution"] * nut_delta / math.tau
            )
            helical_error = axial_travel - expected_axial_from_nut
            q7_tracking_error = q7_delta - probe["q7_delta_rad"]
            q7_to_nut_slip = q7_delta + nut_delta
            non_q7_arm_offsets = [
                index for index in arm_indices if index != q7_index
            ]
            maximum_non_q7_arm_drift = float(
                np.max(
                    np.abs(
                        twist_final_positions[non_q7_arm_offsets]
                        - pre_twist_positions[non_q7_arm_offsets]
                    )
                )
            )
            body_lateral_drift = float(
                np.linalg.norm(
                    twist_final_body_position[:2]
                    - pre_twist_body_position[:2]
                )
            )
            body_axis_error = _axis_error_radians(
                _quaternion_world_z_axis(
                    Gf.Quatd(
                        float(twist_final_body_orientation[0]),
                        Gf.Vec3d(
                            float(twist_final_body_orientation[1]),
                            float(twist_final_body_orientation[2]),
                            float(twist_final_body_orientation[3]),
                        ),
                    )
                ),
                (0.0, 0.0, 1.0),
            )
            twist_fixed_translation_drift = float(
                np.linalg.norm(
                    np.asarray(twist_final_fixed_position)
                    - np.asarray(pre_twist_fixed_position)
                )
            )
            twist_fixed_rotation_drift = _gf_quaternion_error_radians(
                pre_twist_fixed_orientation, twist_final_fixed_orientation
            )
            axial_progress_fraction = (
                abs(axial_travel) / abs(probe["expected_axial_travel_m"])
            )
            twist_final_snapshot = contact_snapshot()
            twist_all_nut = _all_fingers_nut_contact(twist_final_snapshot)
            twist_zero_body = _zero_finger_body_contact(twist_final_snapshot)
            twist_contact_gate = bool(
                twist_all_nut
                and twist_zero_body
                and twist_final_snapshot["nonfinger_endpoint_records"] == 0
                and loose_fixed_contact_records == 0
            )
            evaluation_q7_speeds = hold_q7_observable_speeds[
                -hold_evaluation_steps:
            ]
            evaluation_nut_speeds = hold_nut_observable_speeds[
                -hold_evaluation_steps:
            ]
            evaluation_axial_speeds = hold_axial_observable_speeds[
                -hold_evaluation_steps:
            ]
            axial_window_steps = twist_acceptance[
                "hold_axial_observable_window_steps"
            ]
            evaluation_start_step = hold_steps - hold_evaluation_steps
            evaluation_windowed_axial_speeds = [
                abs(
                    hold_body_z_positions[index]
                    - hold_body_z_positions[index - axial_window_steps]
                )
                * rate_hz
                / axial_window_steps
                for index in range(
                    evaluation_start_step + axial_window_steps,
                    hold_steps + 1,
                )
            ]
            twist_passed = bool(
                finite_throughout
                and activation_translation_jump
                <= twist_acceptance[
                    "maximum_constraint_activation_translation_jump_m"
                ]
                and activation_rotation_jump
                <= twist_acceptance[
                    "maximum_constraint_activation_rotation_jump_rad"
                ]
                and abs(q7_tracking_error)
                <= twist_acceptance["maximum_q7_tracking_error_rad"]
                and twist_acceptance["minimum_nut_progress_rad"]
                <= nut_delta
                <= twist_acceptance["maximum_nut_progress_rad"]
                and abs(q7_to_nut_slip)
                <= twist_acceptance["maximum_q7_to_nut_slip_rad"]
                and axial_travel < 0.0
                and axial_progress_fraction
                >= twist_acceptance["minimum_axial_progress_fraction"]
                and abs(helical_error)
                <= twist_acceptance["maximum_helical_error_m"]
                and maximum_non_q7_arm_drift
                <= twist_acceptance["maximum_non_q7_arm_drift_rad"]
                and body_lateral_drift
                <= twist_acceptance["maximum_body_lateral_drift_m"]
                and body_axis_error
                <= twist_acceptance["maximum_body_axis_error_rad"]
                and max(evaluation_q7_speeds)
                <= twist_acceptance["maximum_hold_q7_speed_rad_s"]
                and max(evaluation_nut_speeds)
                <= twist_acceptance["maximum_hold_nut_speed_rad_s"]
                and max(evaluation_windowed_axial_speeds)
                <= twist_acceptance["maximum_hold_axial_speed_m_s"]
                and twist_fixed_translation_drift
                <= twist_acceptance["maximum_fixed_translation_drift_m"]
                and twist_fixed_rotation_drift
                <= twist_acceptance["maximum_fixed_rotation_drift_rad"]
                and twist_loaded_channels
                >= twist_sensing["minimum_loaded_channels"]
                and hard_torque_violation is None
                and np.max(maximum_post_tare_by_channel)
                <= twist_sensing["hard_stop_nm"]
                and twist_contact_gate
                and all(
                    external_contact_records[key] == 0
                    for key in (
                        "table",
                        "fixture",
                        "fixed_endpoint",
                        "endpoint_nonfinger",
                        "body_forbidden_nut_only",
                        "endpoint_forbidden_clear",
                    )
                )
                and metrics["object_pose_writes_after_start"] == 0
            )
            twist_metrics = {
                "activation_rotation_jump_rad": activation_rotation_jump,
                "activation_translation_jump_m": activation_translation_jump,
                "actual_axial_travel_m": axial_travel,
                "actual_nut_delta_rad": nut_delta,
                "actual_q7_delta_rad": q7_delta,
                "axial_progress_fraction": axial_progress_fraction,
                "body_axis_error_rad": body_axis_error,
                "body_lateral_drift_m": body_lateral_drift,
                "expected_axial_from_measured_nut_m": expected_axial_from_nut,
                "expected_axial_target_m": probe["expected_axial_travel_m"],
                "filtered_pair_count": filtered_pair_count,
                "final_all_fingers_nut_contact": twist_all_nut,
                "final_contacts": twist_final_snapshot,
                "final_loaded_channels": twist_loaded_channels,
                "final_torque_deltas_nm": dict(
                    zip(
                        twist_sensing["torque_joint_names"],
                        twist_final_deltas,
                    )
                ),
                "final_zero_finger_body_contact": twist_zero_body,
                "helical_error_m": helical_error,
                "hold_max_axial_observable_speed_m_s": max(
                    evaluation_windowed_axial_speeds
                ),
                "hold_max_one_step_axial_observable_speed_m_s": max(
                    evaluation_axial_speeds
                ),
                "hold_axial_observable_window_steps": axial_window_steps,
                "hold_evaluation_duration_s": probe[
                    "hold_evaluation_duration_s"
                ],
                "hold_settle_duration_s": probe["hold_settle_duration_s"],
                "hold_total_window_max_axial_observable_speed_m_s": max(
                    hold_axial_observable_speeds
                ),
                "hold_median_axial_observable_speed_m_s": float(
                    np.median(evaluation_axial_speeds)
                ),
                "hold_net_axial_drift_m": float(
                    twist_final_body_position[2]
                    - hold_start_body_position[2]
                ),
                "hold_max_joint_post_solver_speed_rad_s": max(
                    hold_joint_solver_speeds
                ),
                "hold_max_nut_observable_speed_rad_s": max(
                    evaluation_nut_speeds
                ),
                "hold_max_nut_post_solver_speed_rad_s": max(
                    hold_nut_solver_speeds
                ),
                "hold_max_q7_observable_speed_rad_s": max(
                    evaluation_q7_speeds
                ),
                "loose_fixed_contact_records": loose_fixed_contact_records,
                "loose_fixed_contact_pair_records": dict(
                    sorted(loose_fixed_contact_pair_records.items())
                ),
                "maximum_non_q7_arm_drift_rad": maximum_non_q7_arm_drift,
                "passed": twist_passed,
                "q7_command_start_rad": q7_command_start,
                "q7_command_target_rad": q7_command_target,
                "q7_to_nut_slip_rad": q7_to_nut_slip,
                "q7_tracking_error_rad": q7_tracking_error,
                "rack_ratio_degrees_per_meter": thread[
                    "rack_ratio_degrees_per_meter"
                ],
                "runtime_prismatic_path": thread["prismatic_prim_path"],
                "runtime_rack_path": thread["rack_prim_path"],
                "twist_contact_gate": twist_contact_gate,
                "fixed_rotation_drift_rad": twist_fixed_rotation_drift,
                "fixed_translation_drift_m": twist_fixed_translation_drift,
            }

            if arguments.rewind_probe:
                if twist_passed is not True:
                    metrics["twist_probe"] = twist_metrics
                    metrics["rewind_probe"] = {
                        "blocked_reason": "stage120_twist_failed",
                        "passed": False,
                    }
                    raise RuntimeError(
                        "stage120 twist must pass before q7 rewind"
                    )
                rewind_control = rewind_contract["control"]
                rewind_sensing = rewind_contract["sensing"]
                rewind_acceptance = rewind_contract["acceptance"]
                if (
                    twist_contract["probe_id"] != "stage120"
                    or rewind_control["physics_rate_hz"] != rate_hz
                    or rewind_sensing != twist_sensing
                ):
                    raise RuntimeError("rewind runtime contract mismatch")

                brake = rewind_contract[
                    "interstroke_self_lock_brake_proxy"
                ]
                brake_target_degrees = math.degrees(unwrapped_nut_angle)
                world.pause()
                brake_drive = UsdPhysics.DriveAPI.Apply(
                    hinge_prim, UsdPhysics.Tokens.angular
                )
                brake_drive.CreateTypeAttr(UsdPhysics.Tokens.force)
                brake_drive.CreateStiffnessAttr(brake["stiffness"])
                brake_drive.CreateDampingAttr(brake["damping"])
                brake_drive.CreateTargetPositionAttr(
                    brake_target_degrees
                )
                brake_drive.CreateTargetVelocityAttr(
                    brake["target_velocity_degrees_per_second"]
                )
                brake_drive.CreateMaxForceAttr(brake["maximum_force_nm"])
                world.play()
                simulation_app.update()
                metrics["interstroke_self_lock_brake_proxy"] = {
                    "applied_after_twist": True,
                    "calibrated_from_real_hardware": False,
                    "damping": brake["damping"],
                    "drive_type": brake["drive_type"],
                    "maximum_force_nm": brake["maximum_force_nm"],
                    "removed_after_regrasp_preload": False,
                    "stiffness": brake["stiffness"],
                    "target_position_degrees": brake_target_degrees,
                    "target_velocity_degrees_per_second": brake[
                        "target_velocity_degrees_per_second"
                    ],
                }
                metrics["object_drive"] = (
                    "interstroke_self_lock_brake_proxy"
                )

                def deactivate_and_remove_brake(drive_api, tare):
                    """Propagate a zero-force drive before removing its API.

                    Removing a live DriveAPI from USD does not guarantee that
                    an already-created PhysX joint forgets the drive in the
                    same step.  Zeroing every force-producing attribute first
                    makes the transition observable and prevents the next q7
                    stroke from fighting a stale interstroke brake.
                    """

                    nonlocal loose_fixed_contact_records
                    world.pause()
                    drive_api.GetStiffnessAttr().Set(0.0)
                    drive_api.GetDampingAttr().Set(0.0)
                    drive_api.GetMaxForceAttr().Set(0.0)
                    world.play()
                    simulation_app.update()
                    observe_and_step("nut_only")
                    sample_efforts(tare)
                    loose_fixed_contact_records += (
                        count_loose_fixed_contacts()
                    )
                    world.pause()
                    if not hinge_prim.RemoveAPI(
                        UsdPhysics.DriveAPI, UsdPhysics.Tokens.angular
                    ):
                        raise RuntimeError(
                            "interstroke brake API removal failed"
                        )
                    world.play()
                    simulation_app.update()

                rewind_loose_fixed_start = loose_fixed_contact_records
                released_start_nut_angle = float(unwrapped_nut_angle)
                released_start_body_position = (
                    twist_final_body_position.copy()
                )
                released_nut_drift_max = 0.0
                released_body_axial_drift_max = 0.0

                def update_rewind_object_state():
                    nonlocal unwrapped_nut_angle
                    wrapped_angle = _wrapped_relative_z_angle(
                        Gf, Usd, UsdGeom, body_prim, nut_prim
                    )
                    unwrapped_nut_angle = _unwrap(
                        unwrapped_nut_angle, wrapped_angle
                    )
                    body_position, body_orientation = body.get_world_pose()
                    return body_position, body_orientation

                phase = "q7_rewind_release"
                release_start_hand = current_hand_target.copy()
                release_steps = round(
                    rewind_control["release_s"] * rate_hz
                )
                for index in range(release_steps):
                    current_hand_target = _interpolate(
                        release_start_hand,
                        open_hand,
                        float(index + 1) / float(release_steps),
                    )
                    observe_and_step("nut_only")
                    sample_efforts(tare_nut)
                    loose_fixed_contact_records += (
                        count_loose_fixed_contacts()
                    )
                    released_position, _ = update_rewind_object_state()
                    released_nut_drift_max = max(
                        released_nut_drift_max,
                        abs(unwrapped_nut_angle - released_start_nut_angle),
                    )
                    released_body_axial_drift_max = max(
                        released_body_axial_drift_max,
                        abs(
                            float(
                                released_position[2]
                                - released_start_body_position[2]
                            )
                        ),
                    )
                current_hand_target = open_hand.copy()
                set_hand_gains(
                    control["open_hand_stiffness"],
                    control["open_hand_damping"],
                )

                phase = "q7_rewind_open_settle"
                open_settle_steps = round(
                    rewind_control["open_settle_s"] * rate_hz
                )
                for _ in range(open_settle_steps):
                    observe_and_step("clear")
                    sample_efforts(tare_nut)
                    loose_fixed_contact_records += (
                        count_loose_fixed_contacts()
                    )
                    released_position, _ = update_rewind_object_state()
                    released_nut_drift_max = max(
                        released_nut_drift_max,
                        abs(unwrapped_nut_angle - released_start_nut_angle),
                    )
                    released_body_axial_drift_max = max(
                        released_body_axial_drift_max,
                        abs(
                            float(
                                released_position[2]
                                - released_start_body_position[2]
                            )
                        ),
                    )
                released_open_snapshot = contact_snapshot()
                released_open_zero_contact = _zero_endpoint_contact(
                    released_open_snapshot
                ) and released_open_snapshot["nonfinger_endpoint_records"] == 0

                phase = "q7_rewind_motion"
                rewind_before_positions = np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                )
                rewind_command_start = float(
                    current_arm_target[q7_arm_offset]
                )
                rewind_command_target = (
                    rewind_command_start
                    + rewind_control["rewind_delta_rad"]
                )
                if bool(q7_properties["hasLimits"]) and not (
                    float(q7_properties["lower"])
                    <= rewind_command_target
                    <= float(q7_properties["upper"])
                ):
                    raise RuntimeError("q7 rewind target is outside limits")
                rewind_steps = round(
                    rewind_control["rewind_duration_s"] * rate_hz
                )
                for index in range(rewind_steps):
                    fraction = float(index + 1) / float(rewind_steps)
                    current_arm_target[q7_arm_offset] = (
                        rewind_command_start
                        + _minimum_jerk(fraction)
                        * rewind_control["rewind_delta_rad"]
                    )
                    observe_and_step("clear")
                    sample_efforts(tare_nut)
                    loose_fixed_contact_records += (
                        count_loose_fixed_contacts()
                    )
                    released_position, _ = update_rewind_object_state()
                    released_nut_drift_max = max(
                        released_nut_drift_max,
                        abs(unwrapped_nut_angle - released_start_nut_angle),
                    )
                    released_body_axial_drift_max = max(
                        released_body_axial_drift_max,
                        abs(
                            float(
                                released_position[2]
                                - released_start_body_position[2]
                            )
                        ),
                    )
                current_arm_target[q7_arm_offset] = rewind_command_target
                rewind_after_positions = np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                )
                actual_rewind_delta = float(
                    rewind_after_positions[q7_index]
                    - rewind_before_positions[q7_index]
                )
                rewind_tracking_error = (
                    actual_rewind_delta
                    - rewind_control["rewind_delta_rad"]
                )

                phase = "q7_rewind_post_settle"
                settle_q7_speeds = []
                settle_nut_speeds = []
                previous_rewind_positions = rewind_after_positions.copy()
                previous_rewind_nut_angle = float(unwrapped_nut_angle)
                post_settle_steps = round(
                    rewind_control["post_rewind_settle_s"] * rate_hz
                )
                for _ in range(post_settle_steps):
                    observe_and_step("clear")
                    sample_efforts(tare_nut)
                    loose_fixed_contact_records += (
                        count_loose_fixed_contacts()
                    )
                    released_position, _ = update_rewind_object_state()
                    positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    settle_q7_speeds.append(
                        abs(
                            float(
                                positions[q7_index]
                                - previous_rewind_positions[q7_index]
                            )
                        )
                        * rate_hz
                    )
                    settle_nut_speeds.append(
                        abs(unwrapped_nut_angle - previous_rewind_nut_angle)
                        * rate_hz
                    )
                    previous_rewind_positions = positions
                    previous_rewind_nut_angle = float(unwrapped_nut_angle)
                    released_nut_drift_max = max(
                        released_nut_drift_max,
                        abs(unwrapped_nut_angle - released_start_nut_angle),
                    )
                    released_body_axial_drift_max = max(
                        released_body_axial_drift_max,
                        abs(
                            float(
                                released_position[2]
                                - released_start_body_position[2]
                            )
                        ),
                    )

                phase = "q7_rewind_second_open_tare"
                rewind_tare_samples = []
                for _ in range(
                    round(
                        rewind_control["second_open_tare_s"] * rate_hz
                    )
                ):
                    observe_and_step("clear")
                    loose_fixed_contact_records += (
                        count_loose_fixed_contacts()
                    )
                    update_rewind_object_state()
                    rewind_tare_samples.append(
                        np.asarray(
                            robot.get_measured_joint_efforts(
                                joint_indices=sensor_indices
                            ),
                            dtype=np.float64,
                        )
                    )
                tare_regrasp = np.mean(
                    np.stack(rewind_tare_samples), axis=0
                )

                set_hand_gains(
                    control["grip_hand_stiffness"],
                    control["grip_hand_damping"],
                )
                phase = "q7_rewind_reclosure"
                reclosure_start_hand = current_hand_target.copy()
                reclosure_steps = round(
                    rewind_control["reclosure_s"] * rate_hz
                )
                for index in range(reclosure_steps):
                    current_hand_target = _interpolate(
                        reclosure_start_hand,
                        nut_hand,
                        float(index + 1) / float(reclosure_steps),
                    )
                    observe_and_step("nut_only")
                    sample_efforts(tare_regrasp)
                    loose_fixed_contact_records += (
                        count_loose_fixed_contacts()
                    )
                    update_rewind_object_state()
                current_hand_target = nut_hand.copy()

                phase = "q7_rewind_regrasp_preload"
                regrasp_effort_samples = []
                for _ in range(
                    round(rewind_control["preload_s"] * rate_hz)
                ):
                    observe_and_step("nut_only")
                    regrasp_effort_samples.append(
                        sample_efforts(tare_regrasp).copy()
                    )
                    loose_fixed_contact_records += (
                        count_loose_fixed_contacts()
                    )
                    update_rewind_object_state()

                deactivate_and_remove_brake(brake_drive, tare_regrasp)
                metrics["interstroke_self_lock_brake_proxy"][
                    "removed_after_regrasp_preload"
                ] = True
                metrics["interstroke_self_lock_brake_proxy"][
                    "zero_force_step_before_remove"
                ] = True

                phase = "q7_rewind_regrasp_hold"
                final_regrasp_effort_samples = []
                for _ in range(
                    round(rewind_control["final_hold_s"] * rate_hz)
                ):
                    observe_and_step("nut_only")
                    final_regrasp_effort_samples.append(
                        sample_efforts(tare_regrasp).copy()
                    )
                    loose_fixed_contact_records += (
                        count_loose_fixed_contacts()
                    )
                    update_rewind_object_state()

                rewind_final_positions = np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                )
                rewind_final_body_position, rewind_final_body_orientation = (
                    body.get_world_pose()
                )
                rewind_final_fixed_position, rewind_final_fixed_orientation = (
                    _world_pose(Gf, Usd, UsdGeom, fixed_prim)
                )
                rewind_final_snapshot = contact_snapshot()
                rewind_final_all_nut = _all_fingers_nut_contact(
                    rewind_final_snapshot
                )
                rewind_final_zero_body = _zero_finger_body_contact(
                    rewind_final_snapshot
                )
                rewind_final_efforts = np.mean(
                    np.stack(final_regrasp_effort_samples), axis=0
                )
                rewind_final_deltas = rewind_final_efforts - tare_regrasp
                rewind_loaded_channels = int(
                    np.count_nonzero(
                        np.abs(rewind_final_deltas)
                        >= rewind_sensing["loaded_torque_threshold_nm"]
                    )
                )
                retained_nut_progress = float(
                    unwrapped_nut_angle - motion_start_nut_angle
                )
                final_nut_progress_loss = abs(
                    retained_nut_progress - nut_delta
                )
                rewind_body_lateral_drift = float(
                    np.linalg.norm(
                        rewind_final_body_position[:2]
                        - released_start_body_position[:2]
                    )
                )
                rewind_body_axis_error = _axis_error_radians(
                    _quaternion_world_z_axis(
                        Gf.Quatd(
                            float(rewind_final_body_orientation[0]),
                            Gf.Vec3d(
                                float(rewind_final_body_orientation[1]),
                                float(rewind_final_body_orientation[2]),
                                float(rewind_final_body_orientation[3]),
                            ),
                        )
                    ),
                    (0.0, 0.0, 1.0),
                )
                rewind_fixed_translation_drift = float(
                    np.linalg.norm(
                        np.asarray(rewind_final_fixed_position)
                        - np.asarray(pre_twist_fixed_position)
                    )
                )
                rewind_fixed_rotation_drift = _gf_quaternion_error_radians(
                    pre_twist_fixed_orientation,
                    rewind_final_fixed_orientation,
                )
                rewind_loose_fixed_contacts = (
                    loose_fixed_contact_records - rewind_loose_fixed_start
                )
                rewind_contact_gate = bool(
                    released_open_zero_contact
                    and rewind_final_all_nut
                    and rewind_final_zero_body
                    and rewind_final_snapshot["nonfinger_endpoint_records"]
                    == 0
                    and rewind_loose_fixed_contacts == 0
                )
                rewind_passed = bool(
                    finite_throughout
                    and abs(rewind_tracking_error)
                    <= rewind_acceptance[
                        "maximum_q7_rewind_tracking_error_rad"
                    ]
                    and released_nut_drift_max
                    <= rewind_acceptance["maximum_released_nut_drift_rad"]
                    and released_body_axial_drift_max
                    <= rewind_acceptance[
                        "maximum_released_body_axial_drift_m"
                    ]
                    and rewind_body_lateral_drift
                    <= rewind_acceptance["maximum_body_lateral_drift_m"]
                    and rewind_body_axis_error
                    <= rewind_acceptance["maximum_body_axis_error_rad"]
                    and rewind_fixed_translation_drift
                    <= rewind_acceptance[
                        "maximum_fixed_translation_drift_m"
                    ]
                    and rewind_fixed_rotation_drift
                    <= rewind_acceptance[
                        "maximum_fixed_rotation_drift_rad"
                    ]
                    and max(settle_q7_speeds)
                    <= rewind_acceptance[
                        "maximum_settle_q7_observable_speed_rad_s"
                    ]
                    and max(settle_nut_speeds)
                    <= rewind_acceptance[
                        "maximum_settle_nut_observable_speed_rad_s"
                    ]
                    and final_nut_progress_loss
                    <= rewind_acceptance[
                        "maximum_final_nut_progress_loss_rad"
                    ]
                    and rewind_loaded_channels
                    >= rewind_sensing["minimum_loaded_channels"]
                    and hard_torque_violation is None
                    and np.max(maximum_post_tare_by_channel)
                    <= rewind_sensing["hard_stop_nm"]
                    and rewind_contact_gate
                    and all(
                        external_contact_records[key] == 0
                        for key in (
                            "table",
                            "fixture",
                            "fixed_endpoint",
                            "endpoint_nonfinger",
                            "body_forbidden_nut_only",
                            "endpoint_forbidden_clear",
                        )
                    )
                    and metrics["object_pose_writes_after_start"] == 0
                )
                rewind_metrics = {
                    "actual_q7_rewind_delta_rad": actual_rewind_delta,
                    "body_axis_error_rad": rewind_body_axis_error,
                    "body_lateral_drift_m": rewind_body_lateral_drift,
                    "final_all_fingers_nut_contact": rewind_final_all_nut,
                    "final_contacts": rewind_final_snapshot,
                    "final_loaded_channels": rewind_loaded_channels,
                    "final_nut_progress_loss_rad": final_nut_progress_loss,
                    "final_retained_nut_progress_rad": (
                        retained_nut_progress
                    ),
                    "final_torque_deltas_nm": dict(
                        zip(
                            rewind_sensing["torque_joint_names"],
                            rewind_final_deltas,
                        )
                    ),
                    "final_zero_finger_body_contact": (
                        rewind_final_zero_body
                    ),
                    "fixed_rotation_drift_rad": (
                        rewind_fixed_rotation_drift
                    ),
                    "fixed_translation_drift_m": (
                        rewind_fixed_translation_drift
                    ),
                    "loose_fixed_contact_records": (
                        rewind_loose_fixed_contacts
                    ),
                    "maximum_released_body_axial_drift_m": (
                        released_body_axial_drift_max
                    ),
                    "maximum_released_nut_drift_rad": (
                        released_nut_drift_max
                    ),
                    "maximum_settle_nut_observable_speed_rad_s": max(
                        settle_nut_speeds
                    ),
                    "maximum_settle_q7_observable_speed_rad_s": max(
                        settle_q7_speeds
                    ),
                    "open_settle_contacts": released_open_snapshot,
                    "open_settle_zero_endpoint_contact": (
                        released_open_zero_contact
                    ),
                    "passed": rewind_passed,
                    "q7_command_target_rad": rewind_command_target,
                    "q7_final_measured_rad": float(
                        rewind_final_positions[q7_index]
                    ),
                    "q7_rewind_tracking_error_rad": (
                        rewind_tracking_error
                    ),
                    "rewind_contact_gate": rewind_contact_gate,
                }

            additional_twist_reports = []
            additional_rewind_reports = []
            if arguments.full_rotation_probe:
                if twist_passed is not True or rewind_passed is not True:
                    raise RuntimeError(
                        "the first twist/rewind cycle must pass before 360 "
                        "degree continuation"
                    )

                # A 15 N*m/rad A/B barely changed stroke-2 progress, so grip
                # stiffness is restored to the validated compliant baseline.
                # The full-rotation experiment now changes only brake teardown.
                continuation_grip_stiffness = control[
                    "grip_hand_stiffness"
                ]
                continuation_grip_damping = control["grip_hand_damping"]
                set_hand_gains(
                    continuation_grip_stiffness,
                    continuation_grip_damping,
                )
                metrics["full_rotation_grip_ab"] = {
                    "baseline_stiffness": control[
                        "grip_hand_stiffness"
                    ],
                    "continuation_damping": continuation_grip_damping,
                    "continuation_stiffness": (
                        continuation_grip_stiffness
                    ),
                    "hard_stop_nm": sensing["hard_stop_nm"],
                    "operational_target_nm": sensing[
                        "operational_torque_target_nm"
                    ],
                }

                def run_continuation_stroke(stroke_index, rewind_after):
                    """Run one more stroke without rebuilding the scene.

                    The first cycle above remains the golden, detailed probe.
                    Continuation reuses the exact same callbacks, gains, thread
                    proxy, contact classification, torque tare and acceptance
                    limits.  Constraint activation is intentionally absent:
                    the rack/prismatic pair remains live for all three strokes.
                    """

                    nonlocal current_arm_target
                    nonlocal current_hand_target
                    nonlocal loose_fixed_contact_records
                    nonlocal phase
                    nonlocal tare_nut
                    nonlocal unwrapped_nut_angle

                    stroke_contact_start = loose_fixed_contact_records
                    stroke_pair_start = dict(
                        loose_fixed_contact_pair_records
                    )
                    stroke_start_positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    stroke_start_body_position, _ = body.get_world_pose()
                    (
                        stroke_start_fixed_position,
                        stroke_start_fixed_orientation,
                    ) = _world_pose(Gf, Usd, UsdGeom, fixed_prim)
                    stroke_start_nut_angle = float(unwrapped_nut_angle)
                    command_start = float(
                        current_arm_target[q7_arm_offset]
                    )
                    command_target = command_start + probe["q7_delta_rad"]
                    if bool(q7_properties["hasLimits"]) and not (
                        float(q7_properties["lower"])
                        <= command_target
                        <= float(q7_properties["upper"])
                    ):
                        raise RuntimeError(
                            f"stroke {stroke_index} q7 target exceeds limits"
                        )

                    phase = f"full_rotation_stroke_{stroke_index}_motion"
                    motion_steps = round(
                        probe["motion_duration_s"] * rate_hz
                    )
                    for index in range(motion_steps):
                        fraction = float(index + 1) / float(motion_steps)
                        current_arm_target[q7_arm_offset] = (
                            command_start
                            + _minimum_jerk(fraction)
                            * probe["q7_delta_rad"]
                        )
                        observe_and_step("nut_only")
                        sample_efforts(tare_nut)
                        loose_fixed_contact_records += (
                            count_loose_fixed_contacts()
                        )
                        wrapped = _wrapped_relative_z_angle(
                            Gf, Usd, UsdGeom, body_prim, nut_prim
                        )
                        unwrapped_nut_angle = _unwrap(
                            unwrapped_nut_angle, wrapped
                        )
                    current_arm_target[q7_arm_offset] = command_target

                    phase = f"full_rotation_stroke_{stroke_index}_hold"
                    hold_steps = round(
                        probe["total_hold_duration_s"] * rate_hz
                    )
                    evaluation_steps = round(
                        probe["hold_evaluation_duration_s"] * rate_hz
                    )
                    previous_positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    previous_body_position, _ = body.get_world_pose()
                    previous_nut_angle = float(unwrapped_nut_angle)
                    q7_speeds = []
                    nut_speeds = []
                    body_z_positions = [
                        float(previous_body_position[2])
                    ]
                    stroke_efforts = []
                    for _ in range(hold_steps):
                        observe_and_step("nut_only")
                        stroke_efforts.append(
                            sample_efforts(tare_nut).copy()
                        )
                        loose_fixed_contact_records += (
                            count_loose_fixed_contacts()
                        )
                        positions = np.asarray(
                            robot.get_joint_positions(), dtype=np.float64
                        )
                        wrapped = _wrapped_relative_z_angle(
                            Gf, Usd, UsdGeom, body_prim, nut_prim
                        )
                        unwrapped_nut_angle = _unwrap(
                            unwrapped_nut_angle, wrapped
                        )
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
                            abs(unwrapped_nut_angle - previous_nut_angle)
                            * rate_hz
                        )
                        body_z_positions.append(float(body_position[2]))
                        previous_positions = positions
                        previous_body_position = body_position
                        previous_nut_angle = float(unwrapped_nut_angle)

                    final_positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    final_body_position, final_body_orientation = (
                        body.get_world_pose()
                    )
                    final_fixed_position, final_fixed_orientation = (
                        _world_pose(Gf, Usd, UsdGeom, fixed_prim)
                    )
                    q7_delta_now = float(
                        final_positions[q7_index]
                        - stroke_start_positions[q7_index]
                    )
                    nut_delta_now = float(
                        unwrapped_nut_angle - stroke_start_nut_angle
                    )
                    axial_delta_now = float(
                        final_body_position[2]
                        - stroke_start_body_position[2]
                    )
                    expected_axial_now = float(
                        -probe["lead_m_per_revolution"]
                        * nut_delta_now
                        / math.tau
                    )
                    final_efforts = np.mean(
                        np.stack(stroke_efforts), axis=0
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
                    final_snapshot_now = contact_snapshot()
                    stroke_pair_records = {
                        key: value - stroke_pair_start.get(key, 0)
                        for key, value in (
                            loose_fixed_contact_pair_records.items()
                        )
                        if value - stroke_pair_start.get(key, 0) > 0
                    }
                    stroke_loose_fixed_contacts = (
                        loose_fixed_contact_records - stroke_contact_start
                    )
                    # At the end of the third 1 mm stroke the proxy reaches
                    # gap=0.  Its only valid seating evidence is all twenty
                    # loose mating-shell segments touching the fixed rear
                    # stop.  Earlier strokes still require zero endpoint
                    # contact; every other final collider pair remains fatal.
                    fixed_rear_path = (
                        tabletop.asset.fixed_receptacle_prim_path
                        + "/RearBody"
                    )
                    final_seating_contact_gate = bool(
                        stroke_index == 3
                        and not rewind_after
                        and validate_final_seating_contact_pairs(
                            stroke_pair_records,
                            fixed_rear_path=fixed_rear_path,
                            body_mating_root=body_mating_root,
                        )
                    )
                    contact_gate_now = bool(
                        _all_fingers_nut_contact(final_snapshot_now)
                        and _zero_finger_body_contact(final_snapshot_now)
                        and final_snapshot_now[
                            "nonfinger_endpoint_records"
                        ]
                        == 0
                        and (
                            stroke_loose_fixed_contacts == 0
                            or final_seating_contact_gate
                        )
                    )
                    evaluation_q7 = q7_speeds[-evaluation_steps:]
                    evaluation_nut = nut_speeds[-evaluation_steps:]
                    axial_window = twist_acceptance[
                        "hold_axial_observable_window_steps"
                    ]
                    evaluation_start = hold_steps - evaluation_steps
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
                    axis_error_now = _axis_error_radians(
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
                        (0.0, 0.0, 1.0),
                    )
                    lateral_drift_now = float(
                        np.linalg.norm(
                            final_body_position[:2]
                            - stroke_start_body_position[:2]
                        )
                    )
                    fixed_translation_now = float(
                        np.linalg.norm(
                            np.asarray(final_fixed_position)
                            - np.asarray(stroke_start_fixed_position)
                        )
                    )
                    fixed_rotation_now = _gf_quaternion_error_radians(
                        stroke_start_fixed_orientation,
                        final_fixed_orientation,
                    )
                    non_q7_indices = [
                        index for index in arm_indices if index != q7_index
                    ]
                    non_q7_drift_now = float(
                        np.max(
                            np.abs(
                                final_positions[non_q7_indices]
                                - stroke_start_positions[non_q7_indices]
                            )
                        )
                    )
                    progress_fraction_now = abs(axial_delta_now) / abs(
                        probe["expected_axial_travel_m"]
                    )
                    stroke_passed_now = bool(
                        finite_throughout
                        and abs(q7_delta_now - probe["q7_delta_rad"])
                        <= twist_acceptance[
                            "maximum_q7_tracking_error_rad"
                        ]
                        and twist_acceptance["minimum_nut_progress_rad"]
                        <= nut_delta_now
                        <= twist_acceptance["maximum_nut_progress_rad"]
                        and abs(q7_delta_now + nut_delta_now)
                        <= twist_acceptance[
                            "maximum_q7_to_nut_slip_rad"
                        ]
                        and axial_delta_now < 0.0
                        and progress_fraction_now
                        >= twist_acceptance[
                            "minimum_axial_progress_fraction"
                        ]
                        and abs(axial_delta_now - expected_axial_now)
                        <= twist_acceptance["maximum_helical_error_m"]
                        and non_q7_drift_now
                        <= twist_acceptance[
                            "maximum_non_q7_arm_drift_rad"
                        ]
                        and lateral_drift_now
                        <= twist_acceptance["maximum_body_lateral_drift_m"]
                        and axis_error_now
                        <= twist_acceptance["maximum_body_axis_error_rad"]
                        and max(evaluation_q7)
                        <= twist_acceptance["maximum_hold_q7_speed_rad_s"]
                        and max(evaluation_nut)
                        <= twist_acceptance["maximum_hold_nut_speed_rad_s"]
                        and max(evaluation_axial)
                        <= twist_acceptance[
                            "maximum_hold_axial_speed_m_s"
                        ]
                        and fixed_translation_now
                        <= twist_acceptance[
                            "maximum_fixed_translation_drift_m"
                        ]
                        and fixed_rotation_now
                        <= twist_acceptance[
                            "maximum_fixed_rotation_drift_rad"
                        ]
                        and loaded_channels
                        >= twist_sensing["minimum_loaded_channels"]
                        and hard_torque_violation is None
                        and np.max(maximum_post_tare_by_channel)
                        <= twist_sensing["hard_stop_nm"]
                        and contact_gate_now
                        and all(
                            external_contact_records[key] == 0
                            for key in (
                                "table",
                                "fixture",
                                "fixed_endpoint",
                                "endpoint_nonfinger",
                                "body_forbidden_nut_only",
                                "endpoint_forbidden_clear",
                            )
                        )
                        and metrics["object_pose_writes_after_start"] == 0
                    )
                    stroke_report = {
                        "actual_axial_travel_m": axial_delta_now,
                        "actual_nut_delta_rad": nut_delta_now,
                        "actual_q7_delta_rad": q7_delta_now,
                        "body_axis_error_rad": axis_error_now,
                        "body_lateral_drift_m": lateral_drift_now,
                        "final_contacts": final_snapshot_now,
                        "final_loaded_channels": loaded_channels,
                        "final_torque_deltas_nm": dict(
                            zip(
                                twist_sensing["torque_joint_names"],
                                final_deltas,
                            )
                        ),
                        "final_seating_contact_gate": (
                            final_seating_contact_gate
                        ),
                        "final_seating_expected_pair_count": 20,
                        "helical_error_m": (
                            axial_delta_now - expected_axial_now
                        ),
                        "hold_max_axial_observable_speed_m_s": max(
                            evaluation_axial
                        ),
                        "hold_max_nut_observable_speed_rad_s": max(
                            evaluation_nut
                        ),
                        "hold_max_q7_observable_speed_rad_s": max(
                            evaluation_q7
                        ),
                        "loose_fixed_contact_pair_records": dict(
                            sorted(stroke_pair_records.items())
                        ),
                        "loose_fixed_contact_records": (
                            stroke_loose_fixed_contacts
                        ),
                        "passed": stroke_passed_now,
                        "q7_command_start_rad": command_start,
                        "q7_command_target_rad": command_target,
                        "stroke_index": stroke_index,
                        "twist_contact_gate": contact_gate_now,
                    }
                    if not rewind_after:
                        return stroke_report, None

                    brake_config = rewind_contract[
                        "interstroke_self_lock_brake_proxy"
                    ]
                    brake_target = math.degrees(unwrapped_nut_angle)
                    world.pause()
                    drive = UsdPhysics.DriveAPI.Apply(
                        hinge_prim, UsdPhysics.Tokens.angular
                    )
                    drive.CreateTypeAttr(UsdPhysics.Tokens.force)
                    drive.CreateStiffnessAttr(brake_config["stiffness"])
                    drive.CreateDampingAttr(brake_config["damping"])
                    drive.CreateTargetPositionAttr(brake_target)
                    drive.CreateTargetVelocityAttr(
                        brake_config[
                            "target_velocity_degrees_per_second"
                        ]
                    )
                    drive.CreateMaxForceAttr(
                        brake_config["maximum_force_nm"]
                    )
                    world.play()
                    simulation_app.update()

                    released_nut_start = float(unwrapped_nut_angle)
                    released_body_start = final_body_position.copy()
                    maximum_nut_drift = 0.0
                    maximum_axial_drift = 0.0

                    def update_released_state():
                        nonlocal unwrapped_nut_angle
                        wrapped_angle = _wrapped_relative_z_angle(
                            Gf, Usd, UsdGeom, body_prim, nut_prim
                        )
                        unwrapped_nut_angle = _unwrap(
                            unwrapped_nut_angle, wrapped_angle
                        )
                        return body.get_world_pose()[0]

                    def update_release_drift():
                        nonlocal maximum_nut_drift
                        nonlocal maximum_axial_drift
                        released_position = update_released_state()
                        maximum_nut_drift = max(
                            maximum_nut_drift,
                            abs(unwrapped_nut_angle - released_nut_start),
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

                    phase = f"full_rotation_rewind_{stroke_index}_release"
                    release_start = current_hand_target.copy()
                    release_steps = round(
                        rewind_control["release_s"] * rate_hz
                    )
                    for index in range(release_steps):
                        current_hand_target = _interpolate(
                            release_start,
                            open_hand,
                            float(index + 1) / float(release_steps),
                        )
                        observe_and_step("nut_only")
                        sample_efforts(tare_nut)
                        loose_fixed_contact_records += (
                            count_loose_fixed_contacts()
                        )
                        update_release_drift()
                    current_hand_target = open_hand.copy()
                    set_hand_gains(
                        control["open_hand_stiffness"],
                        control["open_hand_damping"],
                    )

                    phase = f"full_rotation_rewind_{stroke_index}_open"
                    for _ in range(
                        round(rewind_control["open_settle_s"] * rate_hz)
                    ):
                        observe_and_step("clear")
                        sample_efforts(tare_nut)
                        loose_fixed_contact_records += (
                            count_loose_fixed_contacts()
                        )
                        update_release_drift()
                    open_snapshot = contact_snapshot()
                    open_gate_now = bool(
                        _zero_endpoint_contact(open_snapshot)
                        and open_snapshot["nonfinger_endpoint_records"] == 0
                    )

                    before_rewind_positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    rewind_start = float(
                        current_arm_target[q7_arm_offset]
                    )
                    rewind_target = (
                        rewind_start + rewind_control["rewind_delta_rad"]
                    )
                    if bool(q7_properties["hasLimits"]) and not (
                        float(q7_properties["lower"])
                        <= rewind_target
                        <= float(q7_properties["upper"])
                    ):
                        raise RuntimeError(
                            f"stroke {stroke_index} rewind exceeds limits"
                        )
                    phase = f"full_rotation_rewind_{stroke_index}_motion"
                    rewind_steps = round(
                        rewind_control["rewind_duration_s"] * rate_hz
                    )
                    for index in range(rewind_steps):
                        fraction = float(index + 1) / float(rewind_steps)
                        current_arm_target[q7_arm_offset] = (
                            rewind_start
                            + _minimum_jerk(fraction)
                            * rewind_control["rewind_delta_rad"]
                        )
                        observe_and_step("clear")
                        sample_efforts(tare_nut)
                        loose_fixed_contact_records += (
                            count_loose_fixed_contacts()
                        )
                        update_release_drift()
                    current_arm_target[q7_arm_offset] = rewind_target
                    after_rewind_positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    actual_rewind = float(
                        after_rewind_positions[q7_index]
                        - before_rewind_positions[q7_index]
                    )
                    tracking_error = (
                        actual_rewind - rewind_control["rewind_delta_rad"]
                    )

                    phase = f"full_rotation_rewind_{stroke_index}_settle"
                    settle_q7 = []
                    settle_nut = []
                    previous_positions = after_rewind_positions.copy()
                    previous_nut = float(unwrapped_nut_angle)
                    for _ in range(
                        round(
                            rewind_control["post_rewind_settle_s"]
                            * rate_hz
                        )
                    ):
                        observe_and_step("clear")
                        sample_efforts(tare_nut)
                        loose_fixed_contact_records += (
                            count_loose_fixed_contacts()
                        )
                        update_release_drift()
                        positions = np.asarray(
                            robot.get_joint_positions(), dtype=np.float64
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
                            abs(unwrapped_nut_angle - previous_nut) * rate_hz
                        )
                        previous_positions = positions
                        previous_nut = float(unwrapped_nut_angle)

                    phase = f"full_rotation_rewind_{stroke_index}_tare"
                    tare_samples_now = []
                    for _ in range(
                        round(
                            rewind_control["second_open_tare_s"] * rate_hz
                        )
                    ):
                        observe_and_step("clear")
                        loose_fixed_contact_records += (
                            count_loose_fixed_contacts()
                        )
                        update_release_drift()
                        tare_samples_now.append(
                            np.asarray(
                                robot.get_measured_joint_efforts(
                                    joint_indices=sensor_indices
                                ),
                                dtype=np.float64,
                            )
                        )
                    tare_nut = np.mean(np.stack(tare_samples_now), axis=0)
                    set_hand_gains(
                        control["grip_hand_stiffness"],
                        control["grip_hand_damping"],
                    )

                    phase = f"full_rotation_rewind_{stroke_index}_regrasp"
                    reclose_start = current_hand_target.copy()
                    reclose_steps = round(
                        rewind_control["reclosure_s"] * rate_hz
                    )
                    for index in range(reclose_steps):
                        current_hand_target = _interpolate(
                            reclose_start,
                            nut_hand,
                            float(index + 1) / float(reclose_steps),
                        )
                        observe_and_step("nut_only")
                        sample_efforts(tare_nut)
                        loose_fixed_contact_records += (
                            count_loose_fixed_contacts()
                        )
                        update_released_state()
                    current_hand_target = nut_hand.copy()

                    phase = f"full_rotation_rewind_{stroke_index}_preload"
                    preload_samples_now = []
                    for _ in range(
                        round(rewind_control["preload_s"] * rate_hz)
                    ):
                        observe_and_step("nut_only")
                        preload_samples_now.append(
                            sample_efforts(tare_nut).copy()
                        )
                        loose_fixed_contact_records += (
                            count_loose_fixed_contacts()
                        )
                        update_released_state()

                    deactivate_and_remove_brake(drive, tare_nut)
                    set_hand_gains(
                        continuation_grip_stiffness,
                        continuation_grip_damping,
                    )

                    phase = f"full_rotation_rewind_{stroke_index}_hold"
                    final_samples_now = []
                    for _ in range(
                        round(rewind_control["final_hold_s"] * rate_hz)
                    ):
                        observe_and_step("nut_only")
                        final_samples_now.append(
                            sample_efforts(tare_nut).copy()
                        )
                        loose_fixed_contact_records += (
                            count_loose_fixed_contacts()
                        )
                        update_released_state()
                    rewind_snapshot_now = contact_snapshot()
                    rewind_efforts_now = np.mean(
                        np.stack(final_samples_now), axis=0
                    )
                    rewind_deltas_now = rewind_efforts_now - tare_nut
                    rewind_loaded_now = int(
                        np.count_nonzero(
                            np.abs(rewind_deltas_now)
                            >= rewind_sensing[
                                "loaded_torque_threshold_nm"
                            ]
                        )
                    )
                    progress_loss = abs(
                        float(unwrapped_nut_angle - stroke_start_nut_angle)
                        - nut_delta_now
                    )
                    rewind_contact_gate_now = bool(
                        open_gate_now
                        and _all_fingers_nut_contact(rewind_snapshot_now)
                        and _zero_finger_body_contact(rewind_snapshot_now)
                        and rewind_snapshot_now[
                            "nonfinger_endpoint_records"
                        ]
                        == 0
                    )
                    rewind_passed_now = bool(
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
                        and rewind_loaded_now
                        >= rewind_sensing["minimum_loaded_channels"]
                        and hard_torque_violation is None
                        and np.max(maximum_post_tare_by_channel)
                        <= rewind_sensing["hard_stop_nm"]
                        and rewind_contact_gate_now
                        and metrics["object_pose_writes_after_start"] == 0
                    )
                    rewind_report = {
                        "actual_q7_rewind_delta_rad": actual_rewind,
                        "final_contacts": rewind_snapshot_now,
                        "final_loaded_channels": rewind_loaded_now,
                        "final_nut_progress_loss_rad": progress_loss,
                        "maximum_released_body_axial_drift_m": (
                            maximum_axial_drift
                        ),
                        "maximum_released_nut_drift_rad": maximum_nut_drift,
                        "passed": rewind_passed_now,
                        "q7_rewind_tracking_error_rad": tracking_error,
                        "rewind_contact_gate": rewind_contact_gate_now,
                        "stroke_index": stroke_index,
                    }
                    return stroke_report, rewind_report

                stroke_2, rewind_2 = run_continuation_stroke(2, True)
                metrics["full_rotation_partial"] = {
                    "rewind_reports": [rewind_metrics, rewind_2],
                    "stroke_reports": [twist_metrics, stroke_2],
                }
                if stroke_2["passed"] is not True:
                    raise RuntimeError("full rotation stroke 2 failed")
                if rewind_2["passed"] is not True:
                    raise RuntimeError("full rotation rewind 2 failed")
                additional_twist_reports.append(stroke_2)
                additional_rewind_reports.append(rewind_2)
                stroke_3, unused_rewind = run_continuation_stroke(3, False)
                if unused_rewind is not None:
                    raise RuntimeError("final stroke must not rewind")
                additional_twist_reports.append(stroke_3)

                full_rotation_evidence = evaluate_d38999_full_rotation(
                    [twist_metrics, *additional_twist_reports],
                    [rewind_metrics, *additional_rewind_reports],
                    expected_stroke_progress_rad=probe[
                        "expected_nut_delta_rad"
                    ],
                    expected_stroke_axial_travel_m=probe[
                        "expected_axial_travel_m"
                    ],
                )
                measured_total_nut_progress = float(
                    unwrapped_nut_angle - motion_start_nut_angle
                )
                final_full_body_position, _ = body.get_world_pose()
                measured_total_axial_travel = float(
                    final_full_body_position[2]
                    - pre_twist_body_position[2]
                )
                metrics["full_rotation"] = {
                    "assembly_success_claimed": False,
                    "continuous_collision_verified": False,
                    "cumulative_axial_travel_m": (
                        full_rotation_evidence.cumulative_axial_travel_m
                    ),
                    "cumulative_nut_progress_rad": (
                        full_rotation_evidence.cumulative_nut_progress_rad
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
                    "physical_insertion_included": False,
                    "rewind_count": full_rotation_evidence.rewind_count,
                    "rewind_reports": [
                        rewind_metrics,
                        *additional_rewind_reports,
                    ],
                    "stroke_count": full_rotation_evidence.stroke_count,
                    "stroke_reports": [
                        twist_metrics,
                        *additional_twist_reports,
                    ],
                }

        passed = bool(
            regrasp_passed
            and (twist_passed is True if arguments.twist_probe else True)
            and (rewind_passed is True if arguments.rewind_probe else True)
            and (
                metrics["full_rotation"]["passed"] is True
                if arguments.full_rotation_probe
                else True
            )
        )
        metrics.update(
            {
                "body_nut_separation_change_m": body_nut_separation_change,
                "candidate_arm_rad": list(nut_arm),
                "candidate_hand_rad": list(nut_hand),
                "desired_physical_tcp_position_error_m": (
                    desired_physical_tcp_position_error
                ),
                "desired_physical_tcp_position_world_m": candidate[
                    "desired_physical_tcp_position_world_m"
                ],
                "contact_gate": contact_gate,
                "cpu_search_config_sha256": _sha256(
                    inputs["cpu_search_config"]
                ),
                "cpu_search_report_sha256": _sha256(
                    inputs["cpu_search_report"]
                ),
                "external_contact_records": external_contact_records,
                "final_all_fingers_nut_contact": final_all_nut,
                "final_body_rotation_drift_rad": (
                    final_body_rotation_drift
                ),
                "final_body_translation_drift_m": (
                    final_body_translation_drift
                ),
                "final_contacts": final_snapshot,
                "final_loaded_channels": final_loaded_channels,
                "final_nut_observable_angular_speed_rad_s": (
                    final_nut_observable_speed
                ),
                "final_nut_post_solver_angular_speed_rad_s": (
                    final_nut_solver_speed
                ),
                "final_observable_joint_speed_rad_s": (
                    final_observable_joint_speed
                ),
                "final_post_solver_joint_speed_rad_s": (
                    final_post_solver_joint_speed
                ),
                "final_torque_deltas_nm": dict(
                    zip(sensing["torque_joint_names"], final_deltas)
                ),
                "final_zero_finger_body_contact": final_zero_body,
                "finite_throughout": finite_throughout,
                "fixed_rotation_drift_rad": fixed_rotation_drift,
                "fixed_translation_drift_m": fixed_translation_drift,
                "hard_stop_nm": sensing["hard_stop_nm"],
                "hard_torque_violation": hard_torque_violation,
                "maximum_arm_tracking_error_rad": (
                    maximum_arm_tracking_error
                ),
                "maximum_body_rotation_drift_rad": (
                    maximum_body_rotation_drift
                ),
                "maximum_body_translation_drift_m": (
                    maximum_body_translation_drift
                ),
                "maximum_joint_limit_violation_rad": (
                    maximum_joint_limit_violation
                ),
                "maximum_joint_speed_rad_s": maximum_joint_speed,
                "maximum_post_tare_absolute_delta_nm": dict(
                    zip(
                        sensing["torque_joint_names"],
                        maximum_post_tare_by_channel,
                    )
                ),
                "mixed_preload_all_fingers_endpoint_contact": mixed_gate,
                "mixed_preload_contacts": mixed_snapshot,
                "open_hand_gains": {
                    "damping": control["open_hand_damping"],
                    "stiffness": control["open_hand_stiffness"],
                },
                "open_reposition_zero_endpoint_contact": open_gate,
                "open_reposition_contacts": open_reposition_snapshot,
                "operational_target_exceeded": operational_target_exceeded,
                "operational_torque_target_nm": (
                    sensing["operational_torque_target_nm"]
                ),
                "passed": passed,
                "regrasp_passed_before_twist": regrasp_passed,
                "phase_steps": phase_steps,
                "postclosure_all_fingers_nut_contact": post_all_nut,
                "postclosure_contacts": postclosure_snapshot,
                "postclosure_zero_finger_body_contact": post_zero_body,
                "preload_loaded_channels": preload_loaded_channels,
                "preload_torque_deltas_nm": dict(
                    zip(sensing["torque_joint_names"], preload_deltas)
                ),
                "grip_hand_gains": {
                    "damping": control["grip_hand_damping"],
                    "stiffness": control["grip_hand_stiffness"],
                },
                "temporary_constraint_path": constraint_path,
                "tcp_axis_error_rad": tcp_axis_error,
                "tcp_position_error_m": tcp_position_error,
                "torque_gate": torque_gate,
                "zero_forbidden_contacts": zero_forbidden_contacts,
                "twist_probe": twist_metrics,
                "rewind_probe": rewind_metrics,
            }
        )
        if tooth_probe is not None:
            metrics["nut_tooth_jitter_probe"] = _finalize_tooth_probe(
                tooth_probe,
                arguments.nut_tooth_jitter_output,
                render_settings_report,
            )
        if tooth_sync_capture is not None:
            metrics["nut_tooth_sync_capture"] = (
                tooth_sync_capture.finalize(
                    physics_report_path=(
                        Path(arguments.nut_tooth_jitter_output)
                        / "report.json"
                    ),
                    physics_summary_path=(
                        Path(arguments.nut_tooth_jitter_output)
                        / "summary.csv"
                    ),
                )
            )
            if metrics["nut_tooth_sync_capture"]["passed"] is not True:
                raise RuntimeError(
                    "nut-tooth synchronized capture cleanup failed"
                )
        if tooth_ghost_runtime is not None:
            ghost_result = tooth_ghost_runtime.finalize(
                capture_manifest_path=(
                    Path(arguments.nut_tooth_sync_capture_output)
                    / "video_capture_manifest.json"
                ),
                physics_report_path=(
                    Path(arguments.nut_tooth_jitter_output) / "report.json"
                ),
                physics_summary_path=(
                    Path(arguments.nut_tooth_jitter_output) / "summary.csv"
                ),
            )
            metrics["nut_tooth_ghost_fingers"] = ghost_result["report"]
            if ghost_result["report"]["passed"] is not True:
                raise RuntimeError("nut-tooth ghost evidence failed")
        print(_metrics_json(metrics), flush=True)
        marker = (
            "ISAAC D38999 Q7 FULL ROTATION PROBE V1"
            if arguments.full_rotation_probe
            else (
                "ISAAC D38999 Q7 REWIND PROBE V1"
                if arguments.rewind_probe
                else (
                    "ISAAC D38999 Q7 TWIST PROBE V1"
                    if arguments.twist_probe
                    else "ISAAC D38999 NUT REGRASP V1"
                )
            )
        )
        print(f"{marker} " + ("PASSED" if passed else "FAILED"), flush=True)
        if arguments.keep_open:
            print(
                "ISAAC D38999 NUT REGRASP V1 GUI REMAINS OPEN; "
                "Ctrl+C to close",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
    except Exception as error:
        metrics["error"] = str(error)
        metrics["passed"] = False
        if tooth_probe is not None:
            metrics["nut_tooth_jitter_probe"] = _finalize_tooth_probe(
                tooth_probe,
                arguments.nut_tooth_jitter_output,
                render_settings_report,
            )
        if tooth_sync_capture is not None:
            try:
                metrics["nut_tooth_sync_capture"] = (
                    tooth_sync_capture.finalize(
                        physics_report_path=(
                            Path(arguments.nut_tooth_jitter_output)
                            / "report.json"
                        ),
                        physics_summary_path=(
                            Path(arguments.nut_tooth_jitter_output)
                            / "summary.csv"
                        ),
                    )
                )
            except Exception as capture_error:
                metrics["nut_tooth_sync_capture_error"] = str(capture_error)
        if tooth_ghost_runtime is not None:
            try:
                metrics["nut_tooth_ghost_cleanup"] = (
                    tooth_ghost_runtime.restore()
                )
            except Exception as ghost_error:
                metrics["nut_tooth_ghost_cleanup_error"] = str(ghost_error)
        print(_metrics_json(metrics), flush=True)
        traceback.print_exc()
        marker = (
            "ISAAC D38999 Q7 FULL ROTATION PROBE V1"
            if arguments.full_rotation_probe
            else (
                "ISAAC D38999 Q7 REWIND PROBE V1"
                if arguments.rewind_probe
                else (
                    "ISAAC D38999 Q7 TWIST PROBE V1"
                    if arguments.twist_probe
                    else "ISAAC D38999 NUT REGRASP V1"
                )
            )
        )
        print(f"{marker} FAILED", flush=True)
        passed = False
    finally:
        # Last-resort idempotent cleanup covers failures inside another
        # diagnostic finalizer.  A successful ghost finalize already restored
        # these opinions before the PASSED marker was emitted.
        if tooth_ghost_runtime is not None:
            try:
                tooth_ghost_runtime.restore()
            except Exception as ghost_error:  # pragma: no cover - Isaac edge
                passed = False
                print(
                    "TOOTH GHOST VISIBILITY RESTORE FAILED: "
                    f"{type(ghost_error).__name__}: {ghost_error}",
                    flush=True,
                )
        simulation_app.close(exit_code=0 if passed else 1)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
