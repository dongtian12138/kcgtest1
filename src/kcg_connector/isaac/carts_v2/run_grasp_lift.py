#!/usr/bin/env python3

"""Run a bounded, truth-isolated CARTS-Grasp V2 preflight or grasp-lift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from typing import Mapping

import numpy as np

if __package__:
    from . import controller as control
    from .engine_health import (
        PhysxStatsMonitor, current_engine_log_path, finalize_engine_evaluation,
        gpu_backend_record, gpu_world_parameters, identity_hashes_match,
        load_runtime_resources, preflight_is_accepted, synchronize_engine_log,
    )
    from .evaluate_run import (
        IsolatedHandRecorder, TruthAuditRecorder, audit_initial_joint_state,
        audit_mimic_schema, compare_reference_targets,
        evaluate_isolated_hand_trace, evaluate_trace,
    )
else:
    import controller as control
    from engine_health import (
        PhysxStatsMonitor, current_engine_log_path, finalize_engine_evaluation,
        gpu_backend_record, gpu_world_parameters, identity_hashes_match,
        load_runtime_resources, preflight_is_accepted, synchronize_engine_log,
    )
    from evaluate_run import (
        IsolatedHandRecorder, TruthAuditRecorder, audit_initial_joint_state,
        audit_mimic_schema, compare_reference_targets,
        evaluate_isolated_hand_trace, evaluate_trace,
    )
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.task_grip_surface import (
    generate_axial_pad_intersection_grasp,
)
from kcg_connector.grasp.robust.object_model import file_sha256
from kcg_connector.robot_model import MIMIC_HAND_JOINTS


ROBOT_ROOT = "/World/HandArm"
ARTICULATION_PATH = ROBOT_ROOT + "/Geometry/world"
HAND_BASE_PATH = ARTICULATION_PATH + (
    "/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3/iiwa_link_4/"
    "iiwa_link_5/iiwa_link_6/iiwa_link_7/iiwa_link_ee/handbase_link"
)
EXPECTED_DOF_NAMES = control.ARM_JOINT_NAMES + (
    "f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3",
)
TENSOR_CONTACT_MAX_COUNT = 4096
def _json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _arguments(repository: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("isolated-hand", "preflight", "first-finger-diagnostic", "grasp-lift"),
                        required=True)
    parser.add_argument("--object-id", default="current_d38999_26kj61sn_public_spec")
    parser.add_argument("--config", default=str(
        repository / "src/kcg_connector/config/carts_grasp_v2.yaml"))
    parser.add_argument("--runtime-resources", default=str(
        repository / "src/kcg_connector/config/carts_v2_isaac_runtime.json"))
    parser.add_argument(
        "--robot-asset",
        help="explicit simulation robot asset; defaults to dynamic.robot_asset",
    )
    parser.add_argument("--preflight-evaluation")
    parser.add_argument("--reference-trace")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--initialize-at-pregrasp", action="store_true")
    parser.add_argument(
        "--capture-visual-evidence", action="store_true",
        help="save four post-step Isaac viewport frames without feeding image truth to control",
    )
    parser.add_argument(
        "--omit-trace-json", action="store_true",
        help="evaluate in memory but do not duplicate the large trace.json",
    )
    arguments = parser.parse_args()
    if arguments.mode in ("first-finger-diagnostic", "grasp-lift") and not arguments.preflight_evaluation:
        parser.error("object contact execution requires --preflight-evaluation")
    if arguments.mode == "isolated-hand" and not arguments.reference_trace:
        parser.error("isolated-hand requires --reference-trace")
    if arguments.capture_visual_evidence and arguments.mode == "isolated-hand":
        parser.error("visual evidence capture is only supported in the object scene")
    if arguments.initialize_at_pregrasp and arguments.mode not in (
        "preflight", "first-finger-diagnostic"
    ):
        parser.error("pregrasp initialization is diagnostic-only")
    return arguments


def _registered_grasp(inputs, object_id: str) -> Mapping[str, object]:
    dynamic = inputs.config.section("dynamic")
    generation = generate_axial_pad_intersection_grasp(inputs)
    grasp = {
        "schema_version": "carts_v2_registered_grasp_v1",
        "grasp_id": f"axial_full_pad_first_intersection_v1__{object_id}",
        "object_id": object_id,
        "hardware_authorized": False,
        "construction_method": generation["method"],
        "control_plan": generation["control_plan"],
        "generation_evidence": generation["evidence"],
        "closure_control": {
            "method": "SEQUENTIAL_LOW_SPEED_JOINT_EFFORT_CONTACT_THEN_FINITE_PRELOAD",
            "finger_maximum_speed_rad_s": dynamic["finger_maximum_speed_rad_s"],
            "contact_detection_effort_rise_nm": dynamic["contact_effort_rise_nm"],
        },
        "finite_clamp_target": {
            "preload_increment_rad": dynamic["preload_increment_rad"],
            "hand_drive_maximum_effort_nm": dynamic[
                "hand_drive_maximum_effort_nm"
            ],
            "measured_effort_abort_nm": dynamic["measured_effort_abort_nm"],
        },
        "lift_trajectory": {
            "direction_world": [0.0, 0.0, 1.0],
            "distance_m": dynamic["lift_command_distance_m"],
            "duration_s": dynamic["lift_duration_s"],
            "arm_damping_nm_s_rad": dynamic["lift_arm_damping_nm_s_rad"],
            "registered_peak_acceleration_m_s2": dynamic[
                "registered_lift_peak_acceleration_m_s2"
            ],
        },
        "hold_control": {
            "method": "HOLD_FINAL_ARM_AND_FINITE_PRELOAD_TARGETS",
            "duration_s": dynamic["hold_duration_s"],
        },
    }
    if not isinstance(grasp, Mapping):
        raise ValueError("object has no directly registered grasp")
    if (
        grasp.get("schema_version") != "carts_v2_registered_grasp_v1"
        or grasp.get("object_id") != object_id
        or grasp.get("hardware_authorized") is not False
        or not isinstance(grasp.get("grasp_id"), str)
        or not isinstance(grasp.get("control_plan"), Mapping)
    ):
        raise ValueError("registered grasp identity or authorization is invalid")
    control_plan = grasp["control_plan"]
    object_from_hand = np.asarray(
        control_plan.get("object_from_hand_row_major"), dtype=np.float64
    )
    pregrasp = np.asarray(
        control_plan.get("pregrasp_joint_positions_rad"), dtype=np.float64
    )
    final = np.asarray(
        control_plan.get("final_joint_positions_rad"), dtype=np.float64
    )
    if (
        object_from_hand.shape != (16,)
        or pregrasp.shape != (4,)
        or final.shape != (4,)
        or not all(np.all(np.isfinite(row)) for row in (object_from_hand, pregrasp, final))
        or not np.allclose(object_from_hand.reshape(4, 4)[3], (0.0, 0.0, 0.0, 1.0))
    ):
        raise ValueError("registered grasp control plan is not finite and rigid-shaped")
    clamp = grasp.get("finite_clamp_target")
    lift = grasp.get("lift_trajectory")
    hold = grasp.get("hold_control")
    if not all(isinstance(row, Mapping) for row in (clamp, lift, hold)):
        raise ValueError("registered grasp omits finite clamp, lift, or hold control")
    expected = (
        (clamp["preload_increment_rad"], dynamic["preload_increment_rad"]),
        (clamp["hand_drive_maximum_effort_nm"], dynamic["hand_drive_maximum_effort_nm"]),
        (clamp["measured_effort_abort_nm"], dynamic["measured_effort_abort_nm"]),
        (lift["distance_m"], dynamic["lift_command_distance_m"]),
        (lift["duration_s"], dynamic["lift_duration_s"]),
        (
            lift["arm_damping_nm_s_rad"],
            dynamic["lift_arm_damping_nm_s_rad"],
        ),
        (hold["duration_s"], dynamic["hold_duration_s"]),
    )
    if any(float(left) != float(right) for left, right in expected):
        raise ValueError("registered grasp differs from bounded dynamic control settings")
    return grasp


def _load_plan_inputs(repository: Path, arguments: argparse.Namespace):
    config_path = Path(arguments.config).resolve()
    arguments.runtime_resources_path = Path(arguments.runtime_resources).resolve()
    arguments.runtime_resources_document = load_runtime_resources(
        arguments.runtime_resources_path)
    inputs = load_v2_inputs(repository, config_path=config_path,
                            object_id=arguments.object_id)
    dynamic = inputs.config.section("dynamic")
    registered_robot_asset = (repository / dynamic["robot_asset"]).resolve()
    arguments.robot_asset_path = (
        Path(arguments.robot_asset).expanduser().resolve()
        if arguments.robot_asset
        else registered_robot_asset
    )
    arguments.robot_asset_override_used = bool(arguments.robot_asset)
    grasp = _registered_grasp(inputs, arguments.object_id)
    scene_entry = dynamic["object_scenes"].get(arguments.object_id)
    if not isinstance(scene_entry, dict):
        raise ValueError("object has no registered free tabletop dynamic scene")
    if arguments.mode in ("first-finger-diagnostic", "grasp-lift"):
        arguments.preflight_evaluation_path = Path(
            arguments.preflight_evaluation).resolve()
        preflight = json.loads(arguments.preflight_evaluation_path.read_text(
            encoding="utf-8"))
        arguments.preflight_document = preflight
        expected = (arguments.object_id, grasp["grasp_id"])
        observed = (preflight.get("object_id"), preflight.get("candidate_id"))
        if (
            observed != expected
            or not preflight_is_accepted(preflight)
            or bool(preflight.get("initialized_at_pregrasp"))
            != bool(arguments.initialize_at_pregrasp)
        ):
            raise ValueError("matching independent preflight did not pass")
    if arguments.mode == "isolated-hand":
        arguments.reference_document = json.loads(
            Path(arguments.reference_trace).read_text(encoding="utf-8"))
        motion_plan = arguments.reference_document["motion_plan"]
    else:
        world_from_object = np.asarray(scene_entry[
            "frozen_settled_world_from_object_row_major"], dtype=np.float64).reshape(4, 4)
        motion_plan = control.build_joint_motion_plan(
            repository, inputs, grasp["control_plan"], world_from_object,
            include_lift=arguments.mode == "grasp-lift",
        )
    return inputs, grasp, scene_entry, motion_plan


def prepare_dynamic_scene(
    repository: Path, stage, entry, add_reference_to_stage
) -> dict[str, object]:
    from omni.physx.scripts import physicsUtils
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    from kcg_connector.d38999_tabletop_scene import (
        author_d38999_tabletop_environment,
        author_d38999_tabletop_scene,
        load_d38999_tabletop_scene,
        verify_d38999_tabletop_asset,
    )

    kind = str(entry["scene_kind"])
    if kind == "D38999_PAIR_TABLETOP":
        scene_path = (repository / entry["scene_config"]).resolve()
        scene = load_d38999_tabletop_scene(scene_path)
        asset = verify_d38999_tabletop_asset(scene, repository)
        author_d38999_tabletop_scene(
            stage, scene, asset, add_reference_to_stage=add_reference_to_stage,
            Gf=Gf, Sdf=Sdf, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics,
            UsdShade=UsdShade, physics_utils=physicsUtils)
        return {
            "object_asset": asset,
            "part_prim_paths": (scene.asset.body_prim_path, scene.asset.nut_prim_path),
            "part_bottom_offsets_m": (
                scene.loose_endpoint.body_bottom_offset_m,
                scene.loose_endpoint.nut_bottom_offset_m,
            ),
            "roots": {
                "object": scene.asset.loose_plug_prim_path,
                "table": scene.table.prim_path,
                "fixture": scene.fixed_endpoint.fixture_prim_path,
            },
            "table_top_z_m": scene.table.top_z_m,
            "gravity_m_s2": scene.physics.gravity_m_s2,
            "render": scene.render,
            "evidence_paths": (scene_path,),
            "environment_scope": "FULL_TABLE_FIXTURE_AND_FIXED_RECEPTACLE",
        }
    if kind != "FREE_SINGLE_RIGID_ON_SHARED_FINITE_TABLE":
        raise ValueError(f"unsupported dynamic scene kind: {kind}")

    environment_path = (repository / entry["environment_scene_config"]).resolve()
    environment = load_d38999_tabletop_scene(environment_path)
    manifest_path = (repository / entry["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset = (repository / entry["asset"]).resolve()
    if (
        manifest.get("hardware_authorized") is not False
        or manifest.get("product_id") != "D38999/26FJ35PN"
        or manifest.get("asset_sha256") != file_sha256(asset)
    ):
        raise ValueError("free rigid asset differs from its registered manifest")
    root = str(entry["reference_prim_path"])
    author_d38999_tabletop_environment(
        stage, environment, Gf=Gf, Sdf=Sdf, UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics, UsdShade=UsdShade, physics_utils=physicsUtils)
    add_reference_to_stage(str(asset), root)
    object_prim = stage.GetPrimAtPath(root)
    if not object_prim.IsValid() or not object_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("free object rigid body is missing")
    matrix = np.asarray(entry["frozen_settled_world_from_object_row_major"],
                        dtype=np.float64).reshape(4, 4)
    xformable = UsdGeom.Xformable(object_prim)
    if xformable.GetOrderedXformOps():
        raise RuntimeError("free object root already has a transform stack")
    xformable.AddTransformOp().Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))
    return {
        "object_asset": asset,
        "part_prim_paths": (root,),
        "part_bottom_offsets_m": tuple(entry["component_bottom_offsets_m"]),
        "roots": {
            "object": root,
            "table": environment.table.prim_path,
            "fixture": environment.fixed_endpoint.fixture_prim_path,
        },
        "table_top_z_m": environment.table.top_z_m,
        "gravity_m_s2": environment.physics.gravity_m_s2,
        "render": environment.render,
        "evidence_paths": (environment_path, manifest_path),
        "environment_scope": "SHARED_FINITE_TABLE_AND_FIXTURE_WITHOUT_FIXED_RECEPTACLE",
    }


def _initial_trace(arguments, inputs, grasp, motion_plan, dynamic):
    criteria = {
        key: dynamic[key]
        for key in (
            "lift_distance_m", "lift_tolerance_m", "hold_duration_s",
            "table_release_clearance_m", "maximum_table_penetration_m",
            "lift_acceleration_difference_window_samples",
            "lift_acceleration_tolerance_m_s2",
        )
    }
    criteria["sustained_three_contact_samples"] = int(
        dynamic["contact_consecutive_samples"]
    )
    criteria["registered_lift_peak_acceleration_m_s2"] = float(
        grasp["lift_trajectory"]["registered_peak_acceleration_m_s2"]
    )
    criteria["first_finger_diagnostic_duration_s"] = float(dynamic["preload_duration_s"])
    criteria["maximum_finger_target_increment_rad"] = float(dynamic[
        "finger_maximum_speed_rad_s"]) * float(dynamic["physics_dt_s"])
    return {
        "schema_version": "carts_grasp_v2_dynamic_trace_v1",
        "object_id": arguments.object_id, "candidate_id": grasp["grasp_id"],
        "mode": arguments.mode,
        "initialized_at_pregrasp": bool(arguments.initialize_at_pregrasp),
        "config_sha256": file_sha256(inputs.config.path),
        "registered_grasp": dict(grasp),
        "offline_worst_task_margin": None,
        "offline_task_gate_passed": False,
        "hardware_authorized": False, "formal_dynamic_pass": False,
        "physics_dt_s": float(dynamic["physics_dt_s"]),
        "maximum_joint_speed_limit_rad_s": float(
            dynamic["maximum_joint_speed_rad_s"]
        ),
        "object_pose_writes_after_start": 0,
        "controller_online_signals": list(motion_plan["online_signals"]),
        "online_object_or_contact_truth_used": False,
        "truth_audit_data_returned_to_controller": False,
        "disturbance_executed": bool(dynamic["disturbance_executed"]),
        "motion_plan": motion_plan,
        "criteria": criteria,
        "controller_outcome": {"completed": False, "failure_reason": None},
        "samples": [],
    }


def _initial_isolated_trace(arguments, inputs, grasp, motion_plan, dynamic):
    reference_path = Path(arguments.reference_trace).resolve()
    reference = arguments.reference_document
    observed = (reference.get("object_id"), reference.get("candidate_id"))
    config_sha256 = file_sha256(inputs.config.path)
    if (observed != (arguments.object_id, grasp["grasp_id"])
            or reference.get("mode") not in ("preflight", "first-finger-diagnostic", "grasp-lift")
            or reference.get("config_sha256") != config_sha256):
        raise ValueError("isolated diagnostic reference differs from the failed run")
    if (
        float(reference.get("physics_dt_s", -1.0)) != float(dynamic["physics_dt_s"])
        or _json_sha256(reference.get("motion_plan")) != _json_sha256(motion_plan)
    ):
        raise ValueError("isolated diagnostic trajectory differs from the failed run")
    return {
        "schema_version": "carts_grasp_v2_isolated_hand_diagnostic_v1",
        "object_id": arguments.object_id, "candidate_id": grasp["grasp_id"],
        "mode": arguments.mode,
        "config_sha256": config_sha256,
        "physics_dt_s": float(dynamic["physics_dt_s"]),
        "maximum_joint_speed_limit_rad_s": float(
            dynamic["maximum_joint_speed_rad_s"]
        ),
        "hardware_authorized": False, "formal_dynamic_pass": False,
        "research_dynamic_pass": False,
        "object_loaded": False, "table_loaded": False,
        "online_object_or_contact_truth_used": False,
        "object_pose_writes_after_start": 0,
        "reference_trace": str(reference_path),
        "reference_trace_sha256": file_sha256(reference_path),
        "reference_failure_reason": reference["controller_outcome"]["failure_reason"],
        "reference_maximum_joint_speed_rad_s": reference["controller_outcome"][
            "maximum_joint_speed_rad_s"],
        "reference_controller_source_sha256": reference["evidence_binding"][
            "controller_source_sha256"],
        "reference_robot_asset_sha256": reference["evidence_binding"][
            "robot_asset_sha256"],
        "reference_active_drive_audit_sha256": _json_sha256(
            reference["controller_outcome"]["native_drive_audit"]),
        "motion_plan": motion_plan,
        "samples": [],
    }


def _full_drive_audit(robot, dof_names):
    stiffnesses, dampings = robot.get_dof_gains(indices=0)
    efforts = robot.get_dof_max_efforts(indices=0)
    drive_types = robot.get_dof_drive_types(indices=0)[0]
    stiffnesses = stiffnesses.numpy()[0]
    dampings = dampings.numpy()[0]
    efforts = efforts.numpy()[0]
    return {
        name: {
            "drive_type": drive_types[index],
            "stiffness": float(stiffnesses[index]),
            "damping": float(dampings[index]),
            "maximum_effort_nm": float(efforts[index]),
            "mimic_source": MIMIC_HAND_JOINTS.get(name),
        }
        for index, name in enumerate(dof_names)
    }


def _isolated_gravity(repository, scene_entry):
    from kcg_connector.d38999_tabletop_scene import load_d38999_tabletop_scene

    key = (
        "scene_config" if scene_entry["scene_kind"] == "D38999_PAIR_TABLETOP"
        else "environment_scene_config"
    )
    scene_path = (repository / scene_entry[key]).resolve()
    return float(load_d38999_tabletop_scene(scene_path).physics.gravity_m_s2), scene_path


def _create_isolated_runtime(repository, arguments, inputs, scene_entry, trace):
    from isaacsim.core.api import World
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

    dynamic = inputs.config.section("dynamic")
    robot_asset = arguments.robot_asset_path
    gravity, gravity_source = _isolated_gravity(repository, scene_entry)
    World.clear_instance()
    SimulationManager.set_physics_sim_device("cuda:0")
    world = World(
        stage_units_in_meters=1.0, physics_dt=float(dynamic["physics_dt_s"]),
        rendering_dt=1.0 / 60.0, backend="numpy", device="cuda:0",
        sim_params={"use_gpu_pipeline": True},
    )
    context = world.get_physics_context()
    add_reference_to_stage(str(robot_asset), ROBOT_ROOT)
    context.set_gravity(gravity)
    world.reset()
    robot_data = control.create_native_gravity_compensated_robot(
        ARTICULATION_PATH, EXPECTED_DOF_NAMES, dynamic)
    robot, active_indices, arm_indices, lower, upper, active_audit = robot_data
    backend = gpu_backend_record(world, context)
    if not backend["pass"]:
        raise RuntimeError(f"GPU physics backend audit failed: {backend}")
    trace["physics_backend"] = backend
    trace["gravity_m_s2"] = gravity
    trace["gravity_source"] = str(gravity_source)
    trace["robot_asset"] = str(robot_asset)
    trace["robot_asset_sha256"] = file_sha256(robot_asset)
    trace["active_drive_audit"] = active_audit
    if (
        trace["robot_asset_sha256"] != trace["reference_robot_asset_sha256"]
        or _json_sha256(active_audit) != trace["reference_active_drive_audit_sha256"]
    ):
        raise RuntimeError("isolated robot asset or active drive differs from reference")
    trace["all_dof_drive_audit"] = _full_drive_audit(robot, robot.dof_names)
    trace["initial_joint_audit"] = audit_initial_joint_state(robot, robot.dof_names)
    trace["mimic_schema_audit"] = audit_mimic_schema(
        get_current_stage(), ROBOT_ROOT, MIMIC_HAND_JOINTS
    )
    recorder = IsolatedHandRecorder(
        robot=robot, dof_names=robot.dof_names,
        active_names=control.ARM_JOINT_NAMES + control.ACTIVE_HAND_JOINT_NAMES,
        hand_names=EXPECTED_DOF_NAMES[7:], physics_dt_s=dynamic["physics_dt_s"],
        drive_settings=dynamic,
    )
    return world, recorder, robot_data


def _execute_isolated(repository, arguments, output, inputs, scene_entry, motion_plan, trace):
    dynamic = inputs.config.section("dynamic")
    world, recorder, robot_data = _create_isolated_runtime(
        repository, arguments, inputs, scene_entry, trace
    )
    robot, active_indices, arm_indices, lower, upper, drive_audit = robot_data
    stepper = control.JointSignalStepper(
        robot=robot, world=world, auditor=recorder, active_indices=active_indices,
        arm_indices=arm_indices, arm_lower_limits=lower, arm_upper_limits=upper,
        settings=dynamic, render=arguments.gui)
    pregrasp = control.run_pregrasp_sequence(stepper, motion_plan, dynamic)
    if arguments.reference_document.get("mode") in ("first-finger-diagnostic", "grasp-lift"):
        for row in arguments.reference_document["samples"][stepper.step_index:]:
            target = np.asarray(row["active_targets_rad"], dtype=np.float64)
            stepper.advance(f"replay_{row['phase']}", target[:7], target[7:])
            if stepper.abort_reason is not None:
                break
    outcome = control.controller_outcome(
        stepper, mode="preflight", native_drive_audit=drive_audit,
        pregrasp=pregrasp,
        grasp={"contact_controller": None, "failure_reason": stepper.abort_reason})
    trace["samples"] = recorder.samples
    trace["controller_outcome"] = outcome
    trace["reference_target_comparison"] = compare_reference_targets(
        arguments.reference_document, trace["samples"])
    trace["runtime"] = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
            check=True, capture_output=True,
        ).stdout.strip(),
        "runner_source_sha256": file_sha256(Path(__file__)),
        "controller_source_sha256": file_sha256(Path(__file__).with_name("controller.py")),
    }
    metrics = evaluate_isolated_hand_trace(trace)
    (output / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["diagnostic_pass"] else 2


def _evidence_binding(repository, arguments, inputs, grasp, scene, robot_asset):
    object_asset = scene["object_asset"]
    evidence_paths = tuple(scene["evidence_paths"])
    return {
        "config_sha256": file_sha256(inputs.config.path),
        "registered_grasp_sha256": _json_sha256(grasp),
        "control_plan_sha256": _json_sha256(grasp["control_plan"]),
        "runtime_resources_sha256": file_sha256(arguments.runtime_resources_path),
        "capacity_audit_sha256": arguments.runtime_resources_document[
            "capacity_audit_sha256"],
        "scene_evidence_sha256": {
            str(path.relative_to(repository)): file_sha256(path) for path in evidence_paths
        },
        "environment_scope": scene["environment_scope"],
        "object_asset_sha256": file_sha256(object_asset),
        "robot_asset_sha256": file_sha256(robot_asset),
        "controller_source_sha256": file_sha256(Path(__file__).with_name("controller.py")),
        "runner_source_sha256": file_sha256(Path(__file__)),
        "evaluator_source_sha256": file_sha256(Path(__file__).with_name("evaluate_run.py")),
        "engine_health_source_sha256": file_sha256(Path(__file__).with_name("engine_health.py")),
    }


def _create_runtime(
    repository, arguments, inputs, grasp, scene_entry, motion_plan, trace
):
    import carb.settings
    from isaacsim.core.api import World
    from isaacsim.core.experimental.prims import RigidPrim as TensorRigidPrim
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdLux, UsdPhysics

    dynamic = inputs.config.section("dynamic")
    robot_asset = arguments.robot_asset_path
    if not robot_asset.is_file():
        raise ValueError("selected robot asset is missing")
    trace["robot_asset"] = str(robot_asset)
    trace["robot_asset_override_used"] = arguments.robot_asset_override_used
    settings = carb.settings.get_settings()
    contact_processing_before = settings.get(
        SETTING_DISABLE_CONTACT_PROCESSING
    )
    settings.set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
    contact_processing_before_world = settings.get_as_bool(
        SETTING_DISABLE_CONTACT_PROCESSING
    )
    trace["contact_processing_setting_audit"] = {
        "path": SETTING_DISABLE_CONTACT_PROCESSING,
        "before": contact_processing_before,
        "required": False,
        "before_world": contact_processing_before_world,
    }
    if contact_processing_before_world:
        raise RuntimeError("PhysX contact processing was not enabled before World creation")
    World.clear_instance()
    SimulationManager.set_physics_sim_device("cuda:0")
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=float(dynamic["physics_dt_s"]),
        rendering_dt=1.0 / 60.0,
        **gpu_world_parameters(arguments.runtime_resources_document),
    )
    context = world.get_physics_context()
    stage = get_current_stage()
    physics_scene_prim = stage.GetPrimAtPath(context.prim_path)
    physics_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    minimum_velocity_iterations = (
        physics_scene_api.GetMinVelocityIterationCountAttr().Get()
    )
    physics_scene_api.CreateMinVelocityIterationCountAttr().Set(8)
    observed_minimum_velocity_iterations = (
        physics_scene_api.GetMinVelocityIterationCountAttr().Get()
    )
    if observed_minimum_velocity_iterations != 8:
        raise RuntimeError("physics scene minimum velocity iterations did not read back as 8")
    trace["physics_scene_velocity_iteration_audit"] = {
        "before": minimum_velocity_iterations,
        "required": 8,
        "observed": observed_minimum_velocity_iterations,
    }
    scene = prepare_dynamic_scene(repository, stage, scene_entry, add_reference_to_stage)
    if arguments.capture_visual_evidence:
        render = scene["render"]
        lighting_root = "/World/CARTSGraspVisualEvidenceLights"
        dome = UsdLux.DomeLight.Define(stage, lighting_root + "/Fill")
        dome.CreateIntensityAttr(float(render.dome_light_intensity))
        dome.CreateColorAttr(Gf.Vec3f(*render.dome_light_color_rgb))
        key = UsdLux.DistantLight.Define(stage, lighting_root + "/Key")
        key.CreateIntensityAttr(float(render.key_light_intensity))
        key.CreateColorAttr(Gf.Vec3f(*render.key_light_color_rgb))
        key.AddRotateXYZOp().Set(
            Gf.Vec3f(*render.key_light_rotation_degrees_xyz)
        )
    trace["evidence_binding"] = _evidence_binding(
        repository, arguments, inputs, grasp, scene, robot_asset)
    if arguments.mode in ("first-finger-diagnostic", "grasp-lift"):
        preflight = arguments.preflight_document
        if preflight.get("evidence_binding") != trace["evidence_binding"]:
            raise ValueError("preflight evidence binding does not match this run")
        trace["accepted_preflight_bound"] = True
        trace["accepted_preflight_evaluation_sha256"] = file_sha256(
            arguments.preflight_evaluation_path)
    add_reference_to_stage(str(robot_asset), ROBOT_ROOT)
    rigid_body_prims, contact_report_prims = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_prims.append(str(prim.GetPath()))
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
            contact_report_prims.append(str(prim.GetPath()))
    trace["contact_report_api_audit"] = {
        "before_reset_rigid_body_paths": rigid_body_prims,
        "before_reset_reporter_paths": contact_report_prims,
    }
    hand_base_prim = stage.GetPrimAtPath(HAND_BASE_PATH)
    if not hand_base_prim.IsValid():
        raise RuntimeError("hand base prim is missing")
    object_parts = tuple(
        world.scene.add(
            SingleRigidPrim(prim_path=path, name=f"carts_v2_object_part_{index}")
        )
        for index, path in enumerate(scene["part_prim_paths"])
    )
    robot_contact_paths = tuple(
        path for path in rigid_body_prims
        if path == ROBOT_ROOT or path.startswith(ROBOT_ROOT + "/")
    )
    object_contact_paths = tuple(map(str, scene["part_prim_paths"]))
    tensor_contact_sensor_paths = robot_contact_paths + object_contact_paths
    if (
        len(set(tensor_contact_sensor_paths)) != len(tensor_contact_sensor_paths)
        or not set(object_contact_paths).issubset(rigid_body_prims)
    ):
        raise RuntimeError("tensor contact sensor paths do not match audited rigid bodies")
    tensor_contact_prim = TensorRigidPrim(
        list(tensor_contact_sensor_paths),
        resolve_paths=False,
        max_contact_count=TENSOR_CONTACT_MAX_COUNT,
    )
    trace["tensor_contact_view_audit"] = {
        "robot_sensor_paths": list(robot_contact_paths),
        "object_sensor_paths": list(object_contact_paths),
        "sensor_paths": list(tensor_contact_sensor_paths),
        "max_contact_count": TENSOR_CONTACT_MAX_COUNT,
    }
    context.set_gravity(float(scene["gravity_m_s2"]))
    world.reset()
    tensor_contact_view_valid = (
        tensor_contact_prim.is_physics_tensor_entity_valid()
    )
    trace["tensor_contact_view_audit"]["valid_after_reset"] = (
        tensor_contact_view_valid
    )
    if not tensor_contact_view_valid:
        raise RuntimeError("tensor contact view is invalid after reset")
    contact_processing_after_reset = settings.get_as_bool(
        SETTING_DISABLE_CONTACT_PROCESSING
    )
    trace["contact_processing_setting_audit"]["after_reset"] = (
        contact_processing_after_reset
    )
    if contact_processing_after_reset:
        raise RuntimeError("PhysX contact processing was disabled during reset")
    after_rigid = [str(prim.GetPath()) for prim in stage.Traverse()
                   if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    after_reporters = [str(prim.GetPath()) for prim in stage.Traverse()
                       if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)]
    trace["contact_report_api_audit"].update({
        "after_reset_rigid_body_paths": after_rigid,
        "after_reset_reporter_paths": after_reporters,
        "complete": (set(rigid_body_prims) == set(contact_report_prims)
                     == set(after_rigid) == set(after_reporters)),
    })
    backend = gpu_backend_record(world, context)
    if not backend["pass"]:
        raise RuntimeError(f"GPU physics backend audit failed: {backend}")
    trace["physics_backend"] = backend
    engine_monitor = PhysxStatsMonitor(context)
    robot_data = control.create_native_gravity_compensated_robot(
        ARTICULATION_PATH,
        EXPECTED_DOF_NAMES,
        dynamic,
        initial_arm_positions=(
            motion_plan["pregrasp_arm_positions_rad"]
            if arguments.initialize_at_pregrasp
            else None
        ),
        initial_hand_positions=(
            motion_plan["pregrasp_hand_positions_rad"]
            if arguments.initialize_at_pregrasp
            else None
        ),
    )
    trace["initial_joint_audit"] = audit_initial_joint_state(
        robot_data[0], robot_data[0].dof_names
    )
    auditor = TruthAuditRecorder(
        object_parts=object_parts,
        hand_base_prim=hand_base_prim,
        robot_model=inputs.robot_model,
        stage_modules=(Gf, Usd, UsdGeom),
        contact_interface=get_physx_simulation_interface(),
        path_decoder=PhysicsSchemaTools.intToSdfPath,
        roots={"robot": ROBOT_ROOT, **scene["roots"]},
        expected_total_mass_kg=inputs.object_contract.model.mass_kg,
        part_bottom_offsets_m=scene["part_bottom_offsets_m"],
        table_top_z_m=scene["table_top_z_m"],
        physics_dt_s=float(dynamic["physics_dt_s"]),
        engine_monitor=engine_monitor,
        physics_step_interface=get_physx_interface(),
        tensor_contact_prim=tensor_contact_prim,
        tensor_contact_sensor_paths=tensor_contact_sensor_paths,
        tensor_contact_max_count=TENSOR_CONTACT_MAX_COUNT,
    )
    return {
        "world": world, "scene": scene, "robot_asset": robot_asset,
        "auditor": auditor, "robot_data": robot_data,
        "engine_monitor": engine_monitor,
        "runtime_resources_path": arguments.runtime_resources_path,
        "capacity_audit_sha256": arguments.runtime_resources_document[
            "capacity_audit_sha256"],
        "registered_grasp": grasp,
        "control_plan": grasp["control_plan"],
    }


class _VisualEvidenceCapture:
    """Capture post-step viewport frames; never return image truth to control."""

    def __init__(self, *, world, auditor, output: Path, physics_dt_s: float) -> None:
        import omni.kit.renderer_capture
        from omni.kit.viewport.utility import get_active_viewport

        self.world = world
        self.auditor = auditor
        self.output = output / "visuals"
        self.output.mkdir(parents=True, exist_ok=False)
        self.physics_dt_s = float(physics_dt_s)
        self.viewport = get_active_viewport()
        if self.viewport is None:
            raise RuntimeError("visual evidence requested but no Isaac viewport is active")
        self.viewport.resolution = (1600, 900)
        self.renderer_capture = (
            omni.kit.renderer_capture.acquire_renderer_capture_interface()
        )
        self.records: list[dict[str, object]] = []
        self.pregrasp_object_z_m: float | None = None
        self._captured: set[str] = set()
        self._truth_capture = auditor.capture
        auditor.capture = self.capture

    def capture(self, **kwargs) -> None:
        self._truth_capture(**kwargs)
        row = self.auditor.samples[-1]
        phase = str(row["phase"])
        if phase == "pregrasp_hold":
            if self.pregrasp_object_z_m is None:
                self.pregrasp_object_z_m = float(row["object_center_m"][2])
            self._capture_once("01_pregrasp", row)
        elif phase == "preload":
            self._capture_once("02_three_finger_clamp", row)
        elif (
            phase == "lift"
            and self.pregrasp_object_z_m is not None
            and float(row["object_center_m"][2]) - self.pregrasp_object_z_m >= 0.020
            and int(row["contacts"]["object_table"]) == 0
        ):
            self._capture_once("03_table_released_20mm", row)

    def capture_run_end(self) -> None:
        if not self.auditor.samples:
            return
        row = self.auditor.samples[-1]
        name = "04_final_hold" if row["phase"] == "hold" else "04_run_end_failure"
        self._capture_once(name, row)

    def _capture_once(self, name: str, row: Mapping[str, object]) -> None:
        if name in self._captured:
            return
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import capture_viewport_to_file

        center = np.asarray(row["object_center_m"], dtype=np.float64)
        target = center + np.asarray((0.0, 0.0, 0.015), dtype=np.float64)
        eye = target + np.asarray((0.38, 0.34, 0.25), dtype=np.float64)
        set_camera_view(eye=eye, target=target, viewport_api=self.viewport)
        for _ in range(8):
            self.world.render()
        image_path = self.output / f"{name}.png"
        capture_viewport_to_file(self.viewport, file_path=str(image_path))
        for _ in range(16):
            self.world.render()
        self.renderer_capture.wait_async_capture()
        for _ in range(8):
            if image_path.is_file() and image_path.stat().st_size > 0:
                break
            self.world.render()
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            raise RuntimeError(f"Isaac viewport capture did not produce {image_path}")
        terminal = row["contacts"]["terminal_link_object"]
        self.records.append({
            "file": str(image_path),
            "step": int(row["step"]),
            "simulation_time_s": float(row["step"]) * self.physics_dt_s,
            "phase": str(row["phase"]),
            "object_center_m": list(map(float, row["object_center_m"])),
            "object_lift_from_pregrasp_m": (
                None if self.pregrasp_object_z_m is None else
                float(row["object_center_m"][2]) - self.pregrasp_object_z_m
            ),
            "object_table_contact_count": int(row["contacts"]["object_table"]),
            "terminal_link_object_contact_counts": list(map(int, terminal)),
            "camera_eye_m": eye.tolist(),
            "camera_target_m": target.tolist(),
        })
        self._captured.add(name)


def _run_controller(runtime, arguments, motion_plan, dynamic):
    robot, active_indices, arm_indices, lower, upper, drive_audit = runtime["robot_data"]
    stepper = control.JointSignalStepper(
        robot=robot, world=runtime["world"], auditor=runtime["auditor"],
        active_indices=active_indices, arm_indices=arm_indices,
        arm_lower_limits=lower, arm_upper_limits=upper,
        settings=dynamic, render=arguments.gui,
    )
    pregrasp = control.run_pregrasp_sequence(
        stepper,
        motion_plan,
        dynamic,
        initialized_at_pregrasp=arguments.initialize_at_pregrasp,
    )
    grasp = (
        control.run_grasp_lift_sequence(
            stepper, motion_plan, dynamic, pregrasp,
            first_finger_only=arguments.mode == "first-finger-diagnostic")
        if arguments.mode in ("first-finger-diagnostic", "grasp-lift")
        else {"contact_controller": None, "failure_reason": stepper.abort_reason}
    )
    outcome = control.controller_outcome(
        stepper, mode=arguments.mode, native_drive_audit=drive_audit,
        pregrasp=pregrasp, grasp=grasp,
    )
    return stepper, outcome


def _runtime_record(repository, inputs, runtime):
    scene = runtime["scene"]
    return {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
            check=True, capture_output=True,
        ).stdout.strip(),
        "config_path": str(inputs.config.path),
        "config_sha256": file_sha256(inputs.config.path),
        "registered_grasp_sha256": _json_sha256(runtime["registered_grasp"]),
        "control_plan_sha256": _json_sha256(runtime["control_plan"]),
        "runtime_resources_sha256": file_sha256(runtime["runtime_resources_path"]),
        "capacity_audit_sha256": runtime["capacity_audit_sha256"],
        "scene_evidence_paths": [str(path) for path in scene["evidence_paths"]],
        "scene_evidence_sha256": {
            str(path.relative_to(repository)): file_sha256(path)
            for path in scene["evidence_paths"]
        },
        "robot_asset_sha256": file_sha256(runtime["robot_asset"]),
        "object_asset_sha256": file_sha256(scene["object_asset"]),
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "controller.py", "run_grasp_lift.py", "evaluate_run.py",
                "engine_health.py",
            )
        },
    }


def _finish_run(repository, inputs, runtime, trace, outcome):
    trace["controller_outcome"] = outcome
    trace["samples"] = runtime["auditor"].samples
    trace["audit_roots"] = {"robot": ROBOT_ROOT, **runtime["scene"]["roots"]}
    visual_capture = runtime.get("visual_capture")
    if visual_capture is not None:
        trace["visual_evidence"] = {
            "schema_version": "carts_grasp_v2_visual_evidence_v1",
            "post_step_observation_only": True,
            "returned_to_controller": False,
            "records": visual_capture.records,
        }
    trace["runtime"] = _runtime_record(repository, inputs, runtime)
    trace["identity_hash_check_pass"] = identity_hashes_match(trace)
    engine_runtime = runtime["engine_monitor"].summary()
    engine_runtime["gpu_backend_pass"] = trace["physics_backend"]["pass"]
    return trace, evaluate_trace(
        trace, robot_asset_path=runtime["robot_asset"], inputs=inputs
    ), engine_runtime


def _execute(repository, arguments, output, inputs, grasp, scene_entry, motion_plan, trace):
    if arguments.mode == "isolated-hand":
        return _execute_isolated(
            repository, arguments, output, inputs, scene_entry, motion_plan, trace
        )
    runtime = _create_runtime(
        repository, arguments, inputs, grasp, scene_entry, motion_plan, trace
    )
    dynamic = inputs.config.section("dynamic")
    if arguments.capture_visual_evidence:
        runtime["visual_capture"] = _VisualEvidenceCapture(
            world=runtime["world"], auditor=runtime["auditor"], output=output,
            physics_dt_s=float(dynamic["physics_dt_s"]),
        )
    _, outcome = _run_controller(runtime, arguments, motion_plan, dynamic)
    if arguments.capture_visual_evidence and arguments.mode == "grasp-lift":
        runtime["visual_capture"].capture_run_end()
    return _finish_run(repository, inputs, runtime, trace, outcome)


def _write_failure(output: Path, error: Exception) -> None:
    payload = {"error_type": type(error).__name__, "error": str(error),
               "traceback": traceback.format_exc(), "hardware_authorized": False}
    (output / "failure.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    repository = Path(__file__).resolve().parents[4]
    arguments = _arguments(repository)
    output = Path(arguments.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs, grasp, scene_entry, motion_plan = _load_plan_inputs(
        repository, arguments)
    dynamic = inputs.config.section("dynamic")
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({
        "headless": not (arguments.gui or arguments.capture_visual_evidence),
        "multi_gpu": False,
        "active_gpu": 0, "physics_gpu": 0, "fast_shutdown": True,
    })
    engine_log_path = current_engine_log_path()
    trace = (_initial_isolated_trace(arguments, inputs, grasp, motion_plan, dynamic)
             if arguments.mode == "isolated-hand"
             else _initial_trace(arguments, inputs, grasp, motion_plan, dynamic))
    try:
        result = _execute(repository, arguments, output, inputs, grasp,
                          scene_entry, motion_plan, trace)
    except Exception as error:
        _write_failure(output, error)
        traceback.print_exc()
        simulation_app.close(exit_code=1)
        return 1
    if isinstance(result, int):
        simulation_app.close(exit_code=result)
        return result
    try:
        trace, evaluation, engine_runtime = result
        engine_runtime["engine_log_sync"] = synchronize_engine_log(engine_log_path)
        evaluation = finalize_engine_evaluation(evaluation, engine_runtime, engine_log_path)
        if not arguments.omit_trace_json:
            (output / "trace.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if "visual_evidence" in trace:
            (output / "visual_evidence.json").write_text(
                json.dumps(trace["visual_evidence"], ensure_ascii=False, indent=2)
                + "\n", encoding="utf-8")
        (output / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        key = ("accepted_preflight_pass" if arguments.mode == "preflight" else
               "first_finger_diagnostic_pass" if arguments.mode == "first-finger-diagnostic"
               else "nominal_research_dynamic_pass")
        exit_code = 0 if evaluation[key] else 2
    except Exception as error:
        _write_failure(output, error)
        traceback.print_exc()
        exit_code = 1
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
