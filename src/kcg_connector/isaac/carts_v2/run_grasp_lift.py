#!/usr/bin/env python3

"""Run a bounded, truth-isolated CARTS-Grasp V2 preflight or grasp-lift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

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
def _json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _arguments(repository: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("isolated-hand", "preflight", "grasp-lift"),
                        required=True)
    parser.add_argument("--object-id", default="current_d38999_26kj61sn_public_spec")
    parser.add_argument("--config", default=str(
        repository / "src/kcg_connector/config/carts_grasp_v2.yaml"))
    parser.add_argument("--offline-result", default=str(
        repository / "artifacts/carts_v2/offline/current_d38999_26kj61sn_public_spec/result.json"))
    parser.add_argument("--runtime-resources", default=str(
        repository / "src/kcg_connector/config/carts_v2_isaac_runtime.json"))
    parser.add_argument("--preflight-evaluation")
    parser.add_argument("--reference-trace")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--gui", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "grasp-lift" and not arguments.preflight_evaluation:
        parser.error("grasp-lift requires --preflight-evaluation")
    if arguments.mode == "isolated-hand" and not arguments.reference_trace:
        parser.error("isolated-hand requires --reference-trace")
    return arguments


def _load_plan_inputs(repository: Path, arguments: argparse.Namespace):
    config_path = Path(arguments.config).resolve()
    arguments.runtime_resources_path = Path(arguments.runtime_resources).resolve()
    arguments.runtime_resources_document = load_runtime_resources(
        arguments.runtime_resources_path)
    report_path = Path(arguments.offline_result).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["object_id"] != arguments.object_id:
        raise ValueError("offline result object does not match requested object")
    if report.get("hardware_authorized") is not False:
        raise ValueError("offline report changed hardware authorization")
    if report["config_sha256"] != file_sha256(config_path):
        raise ValueError("offline result is stale relative to V2 config")
    if not report["top_candidates"]:
        raise ValueError("offline report contains no Top candidate")
    selected = report["top_candidates"][0]
    if selected["rank"] != 1 or not selected["three_effective_pad_contacts"]:
        raise ValueError("dynamic candidate must be the official three-contact Top-1")
    inputs = load_v2_inputs(repository, config_path=config_path,
                            object_id=arguments.object_id)
    dynamic = inputs.config.section("dynamic")
    scene_entry = dynamic["object_scenes"].get(arguments.object_id)
    if not isinstance(scene_entry, dict):
        raise ValueError("object has no registered free tabletop dynamic scene")
    if arguments.mode == "grasp-lift":
        arguments.preflight_evaluation_path = Path(
            arguments.preflight_evaluation).resolve()
        preflight = json.loads(arguments.preflight_evaluation_path.read_text(
            encoding="utf-8"))
        arguments.preflight_document = preflight
        expected = (arguments.object_id, selected["candidate_id"])
        observed = (preflight.get("object_id"), preflight.get("candidate_id"))
        if observed != expected or not preflight_is_accepted(preflight):
            raise ValueError("matching independent preflight did not pass")
    if arguments.mode == "isolated-hand":
        arguments.reference_document = json.loads(
            Path(arguments.reference_trace).read_text(encoding="utf-8"))
        motion_plan = arguments.reference_document["motion_plan"]
    else:
        world_from_object = np.asarray(scene_entry[
            "frozen_settled_world_from_object_row_major"], dtype=np.float64).reshape(4, 4)
        motion_plan = control.build_joint_motion_plan(
            repository, inputs, selected["control_plan"], world_from_object)
    return inputs, report, selected, scene_entry, motion_plan


def _prepare_dynamic_scene(
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
        "evidence_paths": (environment_path, manifest_path),
        "environment_scope": "SHARED_FINITE_TABLE_AND_FIXTURE_WITHOUT_FIXED_RECEPTACLE",
    }


def _initial_trace(arguments, report, selected, motion_plan, dynamic):
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
        report["lambda_one_task_load"]["lift_peak_acceleration_m_s2"]
    )
    return {
        "schema_version": "carts_grasp_v2_dynamic_trace_v1",
        "object_id": arguments.object_id, "candidate_id": selected["candidate_id"],
        "mode": arguments.mode,
        "config_sha256": report["config_sha256"],
        "offline_worst_task_margin": selected["worst_task_margin"],
        "offline_task_gate_passed": selected["offline_task_gate_passed"],
        "hardware_authorized": False, "formal_dynamic_pass": False,
        "physics_dt_s": float(dynamic["physics_dt_s"]),
        "maximum_joint_speed_limit_rad_s": float(
            dynamic["maximum_joint_speed_rad_s"]
        ),
        "object_pose_writes_after_start": 0,
        "controller_online_signals": list(motion_plan["online_signals"]),
        "online_object_or_contact_truth_used": False,
        "truth_audit_data_returned_to_controller": False,
        "pad_surface_identity_verified": False,
        "disturbance_executed": bool(dynamic["disturbance_executed"]),
        "motion_plan": motion_plan,
        "criteria": criteria,
        "controller_outcome": {"completed": False, "failure_reason": None},
        "samples": [],
    }


def _initial_isolated_trace(arguments, report, selected, motion_plan, dynamic):
    reference_path = Path(arguments.reference_trace).resolve()
    reference = arguments.reference_document
    observed = (reference.get("object_id"), reference.get("candidate_id"))
    if (observed != (arguments.object_id, selected["candidate_id"])
            or reference.get("mode") not in ("preflight", "grasp-lift")
            or reference.get("config_sha256") != report["config_sha256"]):
        raise ValueError("isolated diagnostic reference differs from the failed run")
    if (
        float(reference.get("physics_dt_s", -1.0)) != float(dynamic["physics_dt_s"])
        or _json_sha256(reference.get("motion_plan")) != _json_sha256(motion_plan)
    ):
        raise ValueError("isolated diagnostic trajectory differs from the failed run")
    return {
        "schema_version": "carts_grasp_v2_isolated_hand_diagnostic_v1",
        "object_id": arguments.object_id, "candidate_id": selected["candidate_id"],
        "mode": arguments.mode,
        "config_sha256": report["config_sha256"],
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


def _create_isolated_runtime(repository, inputs, scene_entry, trace):
    from isaacsim.core.api import World
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

    dynamic = inputs.config.section("dynamic")
    robot_asset = (repository / dynamic["robot_asset"]).resolve()
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
    world, recorder, robot_data = _create_isolated_runtime(repository, inputs, scene_entry, trace)
    robot, active_indices, arm_indices, lower, upper, drive_audit = robot_data
    stepper = control.JointSignalStepper(
        robot=robot, world=world, auditor=recorder, active_indices=active_indices,
        arm_indices=arm_indices, arm_lower_limits=lower, arm_upper_limits=upper,
        settings=dynamic, render=arguments.gui)
    pregrasp = control.run_pregrasp_sequence(stepper, motion_plan, dynamic)
    if arguments.reference_document.get("mode") == "grasp-lift":
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


def _evidence_binding(repository, arguments, report, selected, scene, robot_asset):
    object_asset = scene["object_asset"]
    evidence_paths = tuple(scene["evidence_paths"])
    return {
        "config_sha256": report["config_sha256"],
        "offline_result_sha256": file_sha256(Path(arguments.offline_result).resolve()),
        "control_plan_sha256": _json_sha256(selected["control_plan"]),
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


def _create_runtime(repository, arguments, inputs, report, selected, scene_entry, trace):
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics

    dynamic = inputs.config.section("dynamic")
    robot_asset = (repository / dynamic["robot_asset"]).resolve()
    if not robot_asset.is_file():
        raise ValueError("registered robot asset is missing")
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
    scene = _prepare_dynamic_scene(repository, stage, scene_entry, add_reference_to_stage)
    trace["evidence_binding"] = _evidence_binding(
        repository, arguments, report, selected, scene, robot_asset)
    if arguments.mode == "grasp-lift":
        preflight = arguments.preflight_document
        if preflight.get("evidence_binding") != trace["evidence_binding"]:
            raise ValueError("preflight evidence binding does not match this run")
        trace["accepted_preflight_bound"] = True
        trace["accepted_preflight_evaluation_sha256"] = file_sha256(
            arguments.preflight_evaluation_path)
    add_reference_to_stage(str(robot_asset), ROBOT_ROOT)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    hand_base_prim = stage.GetPrimAtPath(HAND_BASE_PATH)
    if not hand_base_prim.IsValid():
        raise RuntimeError("hand base prim is missing")
    object_parts = tuple(
        world.scene.add(
            SingleRigidPrim(prim_path=path, name=f"carts_v2_object_part_{index}")
        )
        for index, path in enumerate(scene["part_prim_paths"])
    )
    context.set_gravity(float(scene["gravity_m_s2"]))
    world.reset()
    backend = gpu_backend_record(world, context)
    if not backend["pass"]:
        raise RuntimeError(f"GPU physics backend audit failed: {backend}")
    trace["physics_backend"] = backend
    engine_monitor = PhysxStatsMonitor(context)
    robot_data = control.create_native_gravity_compensated_robot(
        ARTICULATION_PATH, EXPECTED_DOF_NAMES, dynamic)
    auditor = TruthAuditRecorder(
        object_parts=object_parts,
        hand_base_prim=hand_base_prim,
        stage_modules=(Gf, Usd, UsdGeom),
        contact_interface=get_physx_simulation_interface(),
        path_decoder=PhysicsSchemaTools.intToSdfPath,
        roots={"robot": ROBOT_ROOT, **scene["roots"]},
        expected_total_mass_kg=inputs.object_contract.model.mass_kg,
        part_bottom_offsets_m=scene["part_bottom_offsets_m"],
        table_top_z_m=scene["table_top_z_m"],
        physics_dt_s=float(dynamic["physics_dt_s"]),
        engine_monitor=engine_monitor,
    )
    return {
        "world": world, "scene": scene, "robot_asset": robot_asset,
        "auditor": auditor, "robot_data": robot_data,
        "engine_monitor": engine_monitor,
        "runtime_resources_path": arguments.runtime_resources_path,
        "capacity_audit_sha256": arguments.runtime_resources_document[
            "capacity_audit_sha256"],
        "offline_result_path": Path(arguments.offline_result).resolve(),
        "control_plan": selected["control_plan"],
    }


def _run_controller(runtime, arguments, motion_plan, dynamic):
    robot, active_indices, arm_indices, lower, upper, drive_audit = runtime["robot_data"]
    stepper = control.JointSignalStepper(
        robot=robot, world=runtime["world"], auditor=runtime["auditor"],
        active_indices=active_indices, arm_indices=arm_indices,
        arm_lower_limits=lower, arm_upper_limits=upper,
        settings=dynamic, render=arguments.gui,
    )
    pregrasp = control.run_pregrasp_sequence(stepper, motion_plan, dynamic)
    grasp = (
        control.run_grasp_lift_sequence(stepper, motion_plan, dynamic, pregrasp)
        if arguments.mode == "grasp-lift"
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
        "offline_result_sha256": file_sha256(runtime["offline_result_path"]),
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
    trace["runtime"] = _runtime_record(repository, inputs, runtime)
    trace["identity_hash_check_pass"] = identity_hashes_match(trace)
    engine_runtime = runtime["engine_monitor"].summary()
    engine_runtime["gpu_backend_pass"] = trace["physics_backend"]["pass"]
    return trace, evaluate_trace(trace), engine_runtime


def _execute(repository, arguments, output, inputs, report, selected, scene_entry, motion_plan, trace):
    if arguments.mode == "isolated-hand":
        return _execute_isolated(
            repository, arguments, output, inputs, scene_entry, motion_plan, trace
        )
    runtime = _create_runtime(
        repository, arguments, inputs, report, selected, scene_entry, trace
    )
    dynamic = inputs.config.section("dynamic")
    _, outcome = _run_controller(runtime, arguments, motion_plan, dynamic)
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
    inputs, report, selected, scene_entry, motion_plan = _load_plan_inputs(
        repository, arguments)
    dynamic = inputs.config.section("dynamic")
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({
        "headless": not arguments.gui, "multi_gpu": False,
        "active_gpu": 0, "physics_gpu": 0, "fast_shutdown": True,
    })
    engine_log_path = current_engine_log_path()
    trace = (_initial_isolated_trace(arguments, report, selected, motion_plan, dynamic)
             if arguments.mode == "isolated-hand"
             else _initial_trace(arguments, report, selected, motion_plan, dynamic))
    try:
        result = _execute(repository, arguments, output, inputs, report, selected,
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
        (output / "trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        key = ("accepted_preflight_pass" if arguments.mode == "preflight"
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
