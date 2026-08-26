#!/usr/bin/env python3
"""Replay the opposition-60 preshape or first-finger controller in Isaac."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import sys, traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
ISAAC_V2 = ROOT / "src/kcg_connector/isaac/carts_v2"
CONTROLLER_SOURCE = ISAAC_V2 / "controller.py"
sys.path[:0] = [str(ROOT / "src/kcg_connector"), str(ISAAC_V2)]
CONFIG = ROOT / "src/kcg_connector/config/carts_nailfree_height_projected.yaml"
HAND_ROOT = "/World/Opposition60LocalHand"
ACTIVE = ("f1j1", "f1j2", "f2j1", "f3j2")
MIMIC = {"f1j3": "f1j2", "f2j2": "f2j1", "f3j1": "f1j1", "f3j3": "f3j2"}
TERMINALS = ("f1Link3", "f2Link2", "f3Link3")
DT_S, MAXIMUM_INCREMENT_RAD = 1.0 / 120.0, 0.0015
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preshape-replay", "first-finger-diagnostic"))
    parser.add_argument("--task-ik", required=True, type=Path)
    parser.add_argument("--initial-trace", required=True, type=Path)
    parser.add_argument("--initial-exact-replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--preshape-trace", type=Path)
    parser.add_argument("--contact-endpoint-plan", type=Path)
    args = parser.parse_args()
    if args.mode == "first-finger-diagnostic" and args.preshape_trace is None: parser.error("first-finger-diagnostic requires --preshape-trace")
    if args.mode == "first-finger-diagnostic" and args.contact_endpoint_plan is None: parser.error("first-finger-diagnostic requires --contact-endpoint-plan")
    return args
def _sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _load(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def _require(condition: bool, message: str) -> None:
    if not condition: raise ValueError(message)
def _resolve(path: Path | str) -> Path:
    value = Path(path).expanduser(); return value.resolve() if value.is_absolute() else (ROOT / value).resolve()
def _bound(row: dict, label: str) -> Path:
    path = _resolve(row.get("path", ""))
    _require(path.is_file() and _sha256(path) == row.get("sha256"), f"{label} path or hash changed")
    return path
def _host(value) -> np.ndarray:
    for method in ("detach", "cpu"):
        if hasattr(value, method): value = getattr(value, method)()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)
def _values(value) -> list[float | None]:
    return [float(item) if math.isfinite(float(item)) else None
            for item in np.asarray(value).reshape(-1)]
def _finite_or_none(value) -> float | None:
    return float(value) if math.isfinite(float(value)) else None
def _maximum_abs_mapping(row: dict) -> float:
    values = tuple(row.values())
    return (max(abs(float(value)) for value in values)
            if values and all(value is not None for value in values) else math.inf)
def _world_matrix(Usd, UsdGeom, prim) -> np.ndarray:
    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).T
def _verify(args: argparse.Namespace) -> dict:
    from kcg_connector.d38999_tabletop_scene import load_d38999_tabletop_scene
    from kcg_connector.grasp.carts_v2.models import joint_positions_for_phases, load_v2_inputs
    paths = {"task_ik": _resolve(args.task_ik), "initial_trace": _resolve(args.initial_trace),
             "initial_exact_replay": _resolve(args.initial_exact_replay),
             "config": CONFIG.resolve(), "controller_source": CONTROLLER_SOURCE.resolve()}
    _require(all(path.is_file() for path in paths.values()), "fixed evidence input is missing")
    task, initial, exact = map(_load, (paths["task_ik"], paths["initial_trace"], paths["initial_exact_replay"]))
    gates = initial.get("runtime_gates") or {}
    _require(all((initial.get("status") == "INITIAL_PENETRATION_PASS",
             gates.get("ISAAC_IMPORT") is True, gates.get("INITIAL_PENETRATION") is True,
             gates.get("OPPOSITION60_REPLAY") is False, initial.get("online_truth_used_for_control") is False,
             initial.get("closure_command_count") == initial.get("lift_command_count") == 0,
             initial.get("hardware_authorized") is False, initial.get("formal_dynamic_pass") is False,
             initial.get("research_dynamic_pass") is False)),
             "candidate initial-penetration boundary changed")
    trace_row = (exact.get("evidence_binding") or {}).get("trace") or {}
    _require(all((exact.get("status") == "OFFLINE_EXACT_REPLAY_ACCEPTED",
             exact.get("accepted_initial_penetration_pass") is True,
             _resolve(trace_row.get("path", "")) == paths["initial_trace"],
             trace_row.get("sha256") == _sha256(paths["initial_trace"]))),
             "exact-mesh replay no longer binds the candidate initial trace")
    evidence = initial.get("evidence_binding") or {}
    bound = {name: _bound(row, f"initial trace {name}") for name, row in evidence.items()}
    _require(bound["task_ik"] == paths["task_ik"] and bound["config"] == paths["config"],
             "initial trace does not bind the supplied task/configuration")
    anchor, survivor = task["selected_geometric_anchor"], task["survivor_candidates"][0]
    row = task["task_and_bounded_ik"][0]
    height = task["height_search"]
    object_id = anchor["object_id"]
    palm_angle_deg = float(task.get("requested_palm_angle_deg", math.nan))
    _require(all((isinstance(task.get("selected_geometric_anchor_index"), int),
             0 <= task["selected_geometric_anchor_index"] < 12,
             45.0 <= palm_angle_deg <= 75.0,
             abs(palm_angle_deg - round(palm_angle_deg)) < 1e-9,
             abs(float(anchor["palm_configuration_deg"]) - palm_angle_deg) < 1e-9,
             task.get("survivor_count") == height.get("survivor_count") == 1,
             anchor["candidate_id"] == survivor["candidate_id"] == row["candidate_id"],
             survivor["object_id"] == object_id, row.get("fresh_contact_count") == 3,
             row.get("research_task_eligible_not_executable") is True,
             row["bounded_ik"]["status"] == "BOUNDED_IK_PASS_NOT_PATH_COLLISION",
             row["task_quality"]["nominal_gravity_lift_balance_pass"] is True)),
             "height-projected opposition survivor identity changed")
    _require(task.get("config_sha256") == _sha256(paths["config"]),
             "height-projected survivor configuration hash changed")
    inputs = load_v2_inputs(ROOT, config_path=paths["config"], object_id=object_id)
    dynamic = inputs.config.section("dynamic")
    _require(all((float(dynamic["physics_dt_s"]) == DT_S,
             float(dynamic["finger_maximum_speed_rad_s"]) * DT_S == MAXIMUM_INCREMENT_RAD,
             float(dynamic["maximum_joint_speed_rad_s"]) == 3.0, float(dynamic["measured_effort_abort_nm"]) == 0.9,
             round(float(dynamic["effort_tare_duration_s"]) / DT_S) == 60,
             round(float(dynamic["preload_duration_s"]) / DT_S) == 60)),
             "frozen local-contact control limits changed")
    pre = np.asarray(anchor["pregrasp_joint_positions_rad"], np.float64)
    stop = joint_positions_for_phases(inputs, tuple(row["fresh_contact_stop_phases"]),
                                      reference_joint_positions_rad=pre)
    target = np.asarray(row["target_world_from_handbase_row_major"], np.float64).reshape(4, 4)
    projected = inputs.frozen_world_from_object @ np.asarray(
        survivor["object_from_hand_row_major"], np.float64).reshape(4, 4)
    _require(np.allclose(target, projected, atol=1e-12, rtol=0.0)
             and np.allclose(target, np.asarray(initial["pose_binding"][
                 "world_from_handbase_target_row_major"]).reshape(4, 4), atol=1e-12),
             "runtime pose is not the height-projected survivor pose")
    direction = inputs.frozen_world_from_object[:3, :3] @ np.asarray(
        anchor["approach_direction_object"], np.float64)
    far = target.copy()
    far[:3, 3] -= direction * float(dynamic["approach_clearance_height_m"])
    environment = load_d38999_tabletop_scene(
        ROOT / dynamic["object_scenes"][object_id]["environment_scene_config"])
    endpoint_plan = None
    if args.contact_endpoint_plan is not None:
        plan_path = _resolve(args.contact_endpoint_plan); endpoint_plan = _load(plan_path)
        binding = endpoint_plan.get("evidence_binding") or {}
        bound_plan = {name: _bound(value, f"endpoint {name}")
                      for name, value in binding.items()}
        execution = float(endpoint_plan.get("execution_target_rad", math.nan))
        upper = float(endpoint_plan.get("first_nonexecutable_target_rad", math.nan))
        bound_kind = endpoint_plan.get("endpoint_upper_bound_kind")
        _require(all((endpoint_plan.get("status") == "OFFLINE_LAST_SEMANTICALLY_VALID_ENDPOINT_ACCEPTED",
            endpoint_plan.get("candidate_id") == row["candidate_id"], endpoint_plan.get("object_id") == object_id,
            endpoint_plan.get("online_truth_used") is False, endpoint_plan.get("hardware_authorized") is False,
            endpoint_plan.get("formal_dynamic_pass") is False, endpoint_plan.get("research_dynamic_pass") is False,
            endpoint_plan.get("maximum_joint_increment_rad") == MAXIMUM_INCREMENT_RAD,
            endpoint_plan.get("endpoint_definition") == "LAST_SEMANTIC_VALID_NONINTERSECTING_CONTROL_STEP",
            endpoint_plan.get("selection_rule") == "SEMANTIC_VALIDITY_PRECEDES_RAW_FREE_SPACE",
            set(bound_plan) == {"task_ik", "config", "builder_source", "evaluator_source", "runner_source"},
            np.isfinite(execution), np.isfinite(upper), stop[1] <= execution < upper,
            upper - execution <= MAXIMUM_INCREMENT_RAD + 1e-12,
            bound_kind in {"FORBIDDEN_OBJECT_SURFACE_FIRST", "RAW_TASK_SURFACE_INTERSECTION"},
            bound_plan.get("task_ik") == paths["task_ik"], bound_plan.get("config") == paths["config"],
            bound_plan.get("runner_source") == Path(__file__).resolve(),
            np.allclose(np.asarray(endpoint_plan.get("fixed_world_from_handbase_row_major"), np.float64)
                        .reshape(4, 4), target, atol=1e-12, rtol=0.0))),
            "contact endpoint plan is not the bound last semantic-valid state")
        stop = stop.copy(); stop[1] = execution; paths["contact_endpoint_plan"] = plan_path
    if args.preshape_trace is not None:
        preshape_path = _resolve(args.preshape_trace)
        preshape = _load(preshape_path)
        binding = preshape.get("evidence_binding") or {}
        expected = {"task_ik": paths["task_ik"], "initial_trace": paths["initial_trace"],
            "initial_exact_replay": paths["initial_exact_replay"], "config": paths["config"],
            "controller_source": paths["controller_source"],
            "runner_source": Path(__file__).resolve(), "runtime_hand_asset": bound["hand_asset"],
            "runtime_binding": bound["runtime_binding"], "runtime_resources": bound["runtime_resources"]}
        bound_preshape = all(name in binding and _resolve(binding[name].get("path", "")) == path
                             and binding[name].get("sha256") == _sha256(path)
                             for name, path in expected.items())
        controller = preshape.get("controller") or {}
        _require(all((preshape.get("mode") == "preshape-replay", preshape.get("candidate_id") == row["candidate_id"],
                 preshape.get("status") == "PRESHAPE_REPLAY_TRACE_COMPLETE",
                 preshape.get("preshape_replay_pass") is True, preshape.get("atomic_result_pass") is True,
                 preshape.get("research_dynamic_pass") is False, preshape.get("formal_dynamic_pass") is False,
                 preshape.get("hardware_authorized") is False, preshape.get("online_truth_used_for_control") is False,
                 preshape.get("closure_command_count") == preshape.get("second_third_finger_command_count") == 0,
                 preshape.get("preload_command_count") == preshape.get("lift_command_count") == 0,
                 preshape.get("object_pose_writes_after_first_step") == 0,
                 preshape.get("hand_root_pose_writes_after_first_step") == 0,
                 preshape.get("asset_identity", {}).get("pass") is True,
                 preshape.get("physics", {}).get("engine_observation_pass_for_this_atomic_run") is True,
                 controller.get("abort_reason") is None,
                 controller.get("maximum_target_increment_rad", math.inf) <= MAXIMUM_INCREMENT_RAD + 1e-12,
                 controller.get("maximum_joint_speed_rad_s", math.inf) <= 3.0,
                 controller.get("maximum_absolute_joint_effort_nm", math.inf) <= 0.9,
                 controller.get("preshape_hold_maximum_joint_speed_rad_s", math.inf)
                    <= float(dynamic["finger_maximum_speed_rad_s"]),
                 controller.get("preshape_hold_error_reduced_and_hold_speed_bounded") is True,
                 np.allclose(np.asarray(preshape.get("fixed_world_from_handbase_row_major"), np.float64)
                             .reshape(4, 4), far, atol=1e-7, rtol=0.0), bound_preshape)),
                 "first-finger mode lacks one matching accepted preshape trace")
        paths["preshape_trace"] = preshape_path
    return {"paths": paths, "bound": bound, "task": task, "row": row,
            "object_id": object_id, "inputs": inputs,
            "dynamic": dynamic, "pre": pre, "stop": stop,
            "target": target, "far": far, "gravity": environment.physics.gravity_m_s2,
            "endpoint_plan": endpoint_plan}
def _contacts(interface, decoder, roots: dict[str, str]) -> dict:
    headers, _, _ = interface.get_full_contact_report()
    counts = {name: 0 for name in ("hand_object", "hand_table", "hand_fixture", "hand_self", "hand_unclassified", "object_table")}
    rows = []
    for header in headers:
        sides = [tuple(str(decoder(value)) for value in pair) for pair in ((header.actor0, header.collider0), (header.actor1, header.collider1))]
        hits = {name: tuple(any(path == root or path.startswith(root + "/")
                               for path in side) for side in sides)
                for name, root in roots.items()}
        has = lambda name: any(hits.get(name, (False, False)))
        category = ("hand_self" if hits["hand"] == (True, True) else
                    "hand_object" if has("hand") and has("object") else
                    "hand_table" if has("hand") and has("table") else
                    "hand_fixture" if has("hand") and has("fixture") else
                    "hand_unclassified" if has("hand") else
                    "object_table" if has("object") and has("table") else None)
        records = int(header.num_contact_data)
        if category:
            counts[category] += records
        rows.append({"side_0": sides[0], "side_1": sides[1], "record_count": records, "category": category or "other"})
    return {"counts": counts, "paths": rows}
def _run(verified: dict, mode: str, report: dict) -> None:
    import torch
    from isaacsim.core.api import World; from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.types import ArticulationAction; from omni.physx import get_physx_simulation_interface
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics
    from controller import SequentialEffortContactController
    from engine_health import (PhysxStatsMonitor, audit_physx_log, current_engine_log_path, gpu_backend_record,
        gpu_world_parameters, load_runtime_resources, synchronize_engine_log)
    from run_grasp_lift import prepare_dynamic_scene
    dynamic = verified["dynamic"]
    resources = load_runtime_resources(verified["bound"]["runtime_resources"])
    log_path = current_engine_log_path()
    World.clear_instance(); SimulationManager.set_physics_sim_device("cuda:0")
    parameters = gpu_world_parameters(resources); parameters["backend"] = "torch"
    world = World(stage_units_in_meters=1.0, physics_dt=DT_S,
                  rendering_dt=1.0 / 60.0, **parameters)
    context, stage = world.get_physics_context(), get_current_stage()
    scene, objects = None, []
    if mode == "first-finger-diagnostic":
        scene = prepare_dynamic_scene(ROOT, stage, dynamic["object_scenes"][verified["object_id"]],
                                      add_reference_to_stage)
        objects = [world.scene.add(SingleRigidPrim(prim_path=path, name=f"opposition60_contact_object_{index}"))
            for index, path in enumerate(scene["part_prim_paths"])]
    pose = verified["far"] if mode == "preshape-replay" else verified["target"]
    add_reference_to_stage(str(verified["bound"]["hand_asset"]), HAND_ROOT); root = stage.GetPrimAtPath(HAND_ROOT)
    _require(root.IsValid() and not UsdGeom.Xformable(root).GetOrderedXformOps(),
             "local hand root cannot receive the fixed experiment pose")
    UsdGeom.Xformable(root).AddTransformOp().Set(Gf.Matrix4d(*pose.T.ravel().tolist()))
    articulations = [prim for prim in stage.Traverse() if prim.GetPath().HasPrefix(
        Sdf.Path(HAND_ROOT)) and prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    _require(len(articulations) == 1, "local hand articulation identity changed")
    collisions = [prim for prim in stage.Traverse(Usd.TraverseInstanceProxies()) if
        prim.GetPath().HasPrefix(Sdf.Path(HAND_ROOT)) and prim.HasAPI(UsdPhysics.CollisionAPI)]
    terminals = {name: sum(name in str(prim.GetPath()).split("/") for prim in collisions)
                 for name in TERMINALS}
    mimic_live = {follower: list(stage.GetPrimAtPath(f"{HAND_ROOT}/Physics/{follower}")
        .GetRelationship("newton:mimicJoint").GetTargets()) ==
        [Sdf.Path(f"{HAND_ROOT}/Physics/{source}")] for follower, source in MIMIC.items()}
    identity_pass = (len(collisions) == 198 and terminals == {name: 64 for name in TERMINALS}
                     and all(mimic_live.values()))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    robot = world.scene.add(SingleArticulation(
        prim_path=str(articulations[0].GetPath()), name="opposition60_local_contact_hand"))
    context.set_gravity(float(scene["gravity_m_s2"] if scene else verified["gravity"])); world.reset()
    _require(robot.handles_initialized and robot.num_dof == 8, "8-DOF local hand failed to initialize")
    dof_names = tuple(robot.dof_names); name_to_index = {name: i for i, name in enumerate(dof_names)}
    _require(set(dof_names) == set(ACTIVE) | set(MIMIC), "runtime hand DOF names changed")
    active_indices = np.asarray([name_to_index[name] for name in ACTIVE], np.int32)
    follower_indices = np.asarray([name_to_index[name] for name in MIMIC], np.int32)
    initial_active = np.zeros(4) if mode == "preshape-replay" else verified["pre"]
    initial_by_name = dict(zip(ACTIVE, initial_active)); initial_by_name.update(
        {name: initial_by_name[source] for name, source in MIMIC.items()})
    initial = np.asarray([initial_by_name[name] for name in dof_names], np.float32)
    zeros = np.zeros(8, np.float32)
    runtime = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype,
                                                                  device=world.device)
    robot.set_joints_default_state(positions=runtime(initial), velocities=runtime(zeros),
                                   efforts=runtime(zeros))
    robot.set_joint_positions(runtime(initial)); robot.set_joint_velocities(runtime(zeros))
    joint_controller = robot.get_articulation_controller()
    kp = np.zeros(8, np.float32); kd = kp.copy(); cap = kp.copy()
    kp[active_indices] = float(dynamic["hand_stiffness"])
    kd[active_indices] = float(dynamic["hand_damping"])
    cap[active_indices] = float(dynamic["hand_drive_maximum_effort_nm"])
    joint_controller.set_gains(kps=runtime(kp), kds=runtime(kd), save_to_usd=False); joint_controller.set_max_efforts(cap.tolist())
    robot.apply_action(ArticulationAction(joint_positions=runtime(initial_active),
        joint_indices=runtime(active_indices, torch.int64)))
    observed_kp, observed_kd = map(_host, joint_controller.get_gains())
    observed_cap = _host(joint_controller.get_max_efforts())
    drive_pass = bool(np.allclose(observed_kp[active_indices], kp[active_indices])
        and np.allclose(observed_kd[active_indices], kd[active_indices])
        and np.allclose(observed_cap[active_indices], cap[active_indices])
        and np.allclose(observed_kp[follower_indices], 0.0)
        and np.allclose(observed_kd[follower_indices], 0.0)
        and np.allclose(observed_cap[follower_indices], 0.0))
    monitor, interface = PhysxStatsMonitor(context), get_physx_simulation_interface()
    roots = {"hand": HAND_ROOT, **({} if scene is None else scene["roots"])}
    samples, totals, abort = [], {name: 0 for name in ("hand_object", "hand_table", "hand_fixture",
        "hand_self", "hand_unclassified", "object_table")}, None
    maximum_speed = maximum_effort = maximum_delta = other_finger_delta = 0.0
    maximum_mimic_error = 0.0
    previous_target = initial_active.copy()
    latest_positions = initial.copy(); latest_efforts = zeros.copy()
    def sample(target, phase, state, active_finger, effort_input=None):
        nonlocal abort, maximum_speed, maximum_effort, maximum_delta
        nonlocal maximum_mimic_error
        nonlocal other_finger_delta, previous_target, latest_positions, latest_efforts
        target = np.asarray(target, np.float64)
        if target.shape != (4,) or not np.all(np.isfinite(target)):
            abort = abort or "NONFINITE_OR_MALFORMED_TARGET_ABORT"
            return latest_positions, latest_efforts
        delta = float(np.max(np.abs(target - previous_target)))
        maximum_delta = max(maximum_delta, delta)
        if delta > MAXIMUM_INCREMENT_RAD + 1e-12:
            abort = abort or "TARGET_INCREMENT_ABORT"
            return latest_positions, latest_efforts
        other_finger_delta = max(other_finger_delta,
            float(np.max(np.abs(target[[0, 2, 3]] - initial_active[[0, 2, 3]]))))
        robot.apply_action(ArticulationAction(joint_positions=runtime(target),
            joint_indices=runtime(active_indices, torch.int64)))
        world.step(render=False); monitor.sample()
        positions = _host(robot.get_joint_positions()).reshape(-1); velocities = _host(robot.get_joint_velocities()).reshape(-1)
        efforts = _host(robot.get_measured_joint_efforts()).reshape(-1)
        contacts = _contacts(interface, PhysicsSchemaTools.intToSdfPath, roots)
        for name, count in contacts["counts"].items(): totals[name] += count
        object_poses = [tuple(_host(value).reshape(-1) for value in item.get_world_pose())
                        for item in objects]
        finite = all(np.isfinite(row).all() for row in (positions, velocities, efforts))
        speed = float(np.max(np.abs(velocities))) if finite else math.inf; effort = float(np.max(np.abs(efforts))) if finite else math.inf
        maximum_speed, maximum_effort = max(maximum_speed, speed), max(maximum_effort, effort)
        mimic_errors = {name: float(positions[name_to_index[name]]-
            positions[name_to_index[source]]) for name, source in MIMIC.items()}
        if finite: maximum_mimic_error = max(maximum_mimic_error,
            max(abs(value) for value in mimic_errors.values()))
        if not finite: abort = abort or "NONFINITE_JOINT_SIGNAL_ABORT"
        elif speed > 3.0: abort = abort or "JOINT_SPEED_ABORT"
        elif effort > 0.9: abort = abort or "HAND_MEASURED_EFFORT_ABORT"
        elif (verified["endpoint_plan"] is not None and
              positions[name_to_index["f1j2"]] >= verified["endpoint_plan"][
                  "first_nonexecutable_target_rad"]):
            abort = abort or "NONEXECUTABLE_ENDPOINT_OVERSHOOT_ABORT"
        samples.append({"step": len(samples), "simulation_time_s": (len(samples)+1)*DT_S,
            "phase": phase, "controller_state": state, "active_finger": active_finger,
            "joint_positions_rad": dict(zip(dof_names, _values(positions))),
            "joint_velocities_rad_s": dict(zip(dof_names, _values(velocities))),
            "joint_efforts_nm": dict(zip(dof_names, _values(efforts))),
            "active_targets_rad": dict(zip(ACTIVE, _values(target))),
            "active_target_delta_rad": dict(zip(ACTIVE, _values(target-previous_target))),
            "active_target_errors_rad": dict(zip(ACTIVE,
                _values(target-positions[active_indices]))),
            "controller_input_tare_subtracted_effort_nm": None if effort_input is None else
                dict(zip(ACTIVE, _values(effort_input))),
            "mimic_position_errors_rad": {name: _finite_or_none(value)
                for name, value in mimic_errors.items()},
            "mimic_velocity_errors_rad_s": {name: _finite_or_none(
                velocities[name_to_index[name]]-velocities[name_to_index[source]])
                for name, source in MIMIC.items()},
            "world_from_handbase_row_major": _values(_world_matrix(
                Usd, UsdGeom, articulations[0])),
            "object_poses": [{"position_m": _values(pose[0]),
                              "orientation_wxyz": _values(pose[1])} for pose in object_poses],
            "post_step_contact_truth_audit_only": contacts, "safety_abort_reason": abort})
        previous_target = target.copy(); latest_positions, latest_efforts = positions, efforts
        return positions, efforts
    contact = None; hold_steps = closure_commands = 0
    if mode == "preshape-replay":
        palm = np.asarray([verified["pre"][0], 0.0, 0.0, 0.0])
        stages = (("palm_far", initial_active, palm), ("preshape_far", palm, verified["pre"]))
        for phase, start, goal in stages:
            count = int(math.ceil(float(np.max(np.abs(goal-start))) / MAXIMUM_INCREMENT_RAD))
            for index in range(1, count + 1):
                sample(start + (index/count)*(goal-start), phase, "BOUNDED_REPLAY", None)
                if abort: break
            if abort: break
        for _ in range(60 if abort is None else 0):
            sample(verified["pre"], "preshape_hold", "HOLD", None)
    else:
        tare_rows = []
        for _ in range(60):
            _, effort = sample(verified["pre"], "effort_tare", "TARE", 1)
            if abort: break
            tare_rows.append(effort[active_indices].copy())
        tare = np.mean(np.stack(tare_rows), axis=0) if len(tare_rows) == 60 else np.zeros(4)
        contact = SequentialEffortContactController(verified["pre"], verified["stop"],
            effort_rise_nm=float(dynamic["contact_effort_rise_nm"]),
            position_error_rad=float(dynamic["contact_position_error_rad"]),
            consecutive_samples=int(dynamic["contact_consecutive_samples"]),
            endpoint_timeout_samples=round(float(dynamic["contact_endpoint_timeout_s"])/DT_S),
            hand_stiffness=float(dynamic["hand_stiffness"]))
        budget = int(math.ceil(abs(verified["stop"][1]-verified["pre"][1]) /
                               MAXIMUM_INCREMENT_RAD)) + 2*contact.endpoint_timeout_samples + 20
        for _ in range(budget if abort is None else 0):
            effort_delta = latest_efforts[active_indices] - tare
            target = contact.step(latest_positions[active_indices], effort_delta,
                                  MAXIMUM_INCREMENT_RAD, advance_after_hold=False)
            sample(target, f"finger_1_{contact.last_output_state.lower()}",
                   contact.last_output_state, 1, effort_delta); closure_commands += 1
            if abort or contact.failed or contact.state == "HOLD": break
        for _ in range(60 if abort is None and contact and contact.state == "HOLD" else 0):
            effort_delta = latest_efforts[active_indices] - tare
            target = contact.step(latest_positions[active_indices], effort_delta,
                                  MAXIMUM_INCREMENT_RAD, advance_after_hold=False)
            sample(target, "finger_1_hold", "HOLD", 1, effort_delta)
            closure_commands += 1; hold_steps += 1
            if abort: break
    sync = synchronize_engine_log(log_path); log = audit_physx_log(
        log_path, cutoff_bytes=sync["audit_byte_count"], required_marker=sync["marker"])
    stats, backend = monitor.summary(), gpu_backend_record(world, context)
    engine_pass = bool(backend["pass"] and stats["physx_statistics_sample_count"] > 0
        and stats["physx_statistics_read_failures"] == 0
        and stats["observed_gpu_found_lost_aggregate_pairs_peak"] <
            stats["configured_gpu_found_lost_aggregate_pairs_capacity"]
        and stats["observed_gpu_total_aggregate_pairs_peak"] <
            stats["configured_gpu_total_aggregate_pairs_capacity"]
        and log.get("scan_complete") is True and log.get("capacity_warning_count") == 0
        and log.get("physx_error_lines") == [])
    final_error = (float(np.max(np.abs(latest_positions[active_indices]-verified["pre"])))
                   if np.isfinite(latest_positions).all() else math.inf)
    hold_rows = [row for row in samples if row["phase"] == "preshape_hold"]
    hold_start_errors = (hold_rows[0]["active_target_errors_rad"]
                         if hold_rows else {})
    hold_end_errors = (hold_rows[-1]["active_target_errors_rad"]
                       if hold_rows else {})
    hold_start_error = (_maximum_abs_mapping(hold_rows[0][
        "active_target_errors_rad"]) if hold_rows else math.inf)
    final_speed = (_maximum_abs_mapping(samples[-1][
        "joint_velocities_rad_s"]) if samples else math.inf)
    hold_maximum_speed = max((_maximum_abs_mapping(row[
        "joint_velocities_rad_s"]) for row in hold_rows), default=math.inf)
    per_joint_error_reduced = bool(hold_rows and all(
        hold_start_errors.get(name) is not None
        and hold_end_errors.get(name) is not None
        and abs(float(hold_end_errors[name])) <= abs(float(hold_start_errors[name])) + 1e-12
        for name in ACTIVE))
    hold_converged = bool(len(hold_rows) == 60 and per_joint_error_reduced
        and final_error < hold_start_error
        and hold_maximum_speed <= float(dynamic["finger_maximum_speed_rad_s"]))
    forbidden = sum(totals[name] for name in ("hand_table", "hand_fixture", "hand_self", "hand_unclassified"))
    root_error = max(float(np.max(np.abs(np.asarray(row[
        "world_from_handbase_row_major"]).reshape(4, 4)-pose))) for row in samples)
    preshape_pass = bool(mode == "preshape-replay" and abort is None and identity_pass
        and drive_pass and engine_pass and forbidden == 0 and maximum_delta <= MAXIMUM_INCREMENT_RAD+1e-12
        and hold_converged
        and maximum_mimic_error <= float(dynamic["contact_position_error_rad"])
        and root_error <= 1e-7)
    controller_pass = bool(mode == "first-finger-diagnostic" and abort is None
        and contact is not None and not contact.failed and contact.state == "HOLD"
        and len(contact.contact_targets_rad) == 1 and hold_steps == 60
        and other_finger_delta <= 1e-12 and forbidden == 0 and identity_pass and drive_pass
        and engine_pass and maximum_delta <= MAXIMUM_INCREMENT_RAD+1e-12
        and maximum_mimic_error <= float(dynamic["contact_position_error_rad"])
        and root_error <= 1e-7)
    report.update({"status": ("PRESHAPE_REPLAY_TRACE_COMPLETE" if preshape_pass else
        "FIRST_FINGER_CONTROLLER_TRACE_COMPLETE" if controller_pass else "FAILED_CLOSED"),
        "preshape_replay_pass": preshape_pass,
        "first_finger_controller_trace_pass": controller_pass,
        "first_finger_diagnostic_pass": False,
        "post_run_task_grip_surface_evaluation_pending": mode == "first-finger-diagnostic",
        "atomic_result_pass": preshape_pass or controller_pass,
        "runtime_gates": {"ISAAC_IMPORT": True, "INITIAL_PENETRATION": True,
                          "OPPOSITION60_REPLAY": False, "PHYSX_HEALTH": False},
        "asset_identity": {"pass": identity_pass, "dof_names": dof_names,
                           "collision_count": len(collisions), "terminal_counts": terminals,
                           "mimic_live": mimic_live, "drive_pass": drive_pass},
        "physics": {"dt_s": DT_S, "step_count": len(samples),
                    "physics_time_advanced_s": len(samples)*DT_S, "backend": backend,
                    "statistics": stats, "log": log,
                    "engine_observation_pass_for_this_atomic_run": engine_pass},
        "controller": {"abort_reason": abort or (None if contact is None else contact.failure_reason),
            "maximum_target_increment_rad": maximum_delta,
            "maximum_other_finger_target_change_rad": other_finger_delta,
            "maximum_joint_speed_rad_s": _finite_or_none(maximum_speed),
            "maximum_absolute_joint_effort_nm": _finite_or_none(maximum_effort),
            "maximum_mimic_position_error_rad": _finite_or_none(maximum_mimic_error),
            "mimic_error_gate_rad": float(dynamic["contact_position_error_rad"]),
            "preshape_hold_start_maximum_target_error_rad": _finite_or_none(hold_start_error),
            "preshape_final_maximum_target_error_rad": _finite_or_none(final_error),
            "preshape_hold_start_target_error_rad": hold_start_errors,
            "preshape_hold_final_target_error_rad": hold_end_errors,
            "preshape_hold_each_active_joint_error_reduced": per_joint_error_reduced,
            "preshape_hold_maximum_joint_speed_rad_s": _finite_or_none(hold_maximum_speed),
            "preshape_final_maximum_joint_speed_rad_s": _finite_or_none(final_speed),
            "preshape_hold_error_reduced_and_hold_speed_bounded": hold_converged,
            "contact_targets_rad": [] if contact is None else list(contact.contact_targets_rad),
            "final_contact_state": None if contact is None else contact.state,
            "first_finger_hold_steps": hold_steps,
            "predicted_proximity_target_rad": None if verified["endpoint_plan"] is None else
                verified["endpoint_plan"]["predicted_proximity_target_rad"],
            "bound_execution_target_rad": None if verified["endpoint_plan"] is None else
                verified["endpoint_plan"]["execution_target_rad"],
            "first_semantically_invalid_target_rad": None if verified["endpoint_plan"] is None else
                verified["endpoint_plan"]["first_semantically_invalid_target_rad"],
            "first_nonexecutable_target_rad": None if verified["endpoint_plan"] is None else
                verified["endpoint_plan"]["first_nonexecutable_target_rad"]},
        "contact_totals": totals, "forbidden_contact_record_count": forbidden,
        "fixed_world_from_handbase_row_major": pose.ravel().tolist(),
        "maximum_handbase_readback_error": root_error, "samples": samples,
        "closure_command_count": closure_commands, "second_third_finger_command_count": 0,
        "preload_command_count": 0, "lift_command_count": 0,
        "object_pose_writes_after_first_step": 0, "hand_root_pose_writes_after_first_step": 0})
def main() -> int:
    args = _arguments(); output = _resolve(args.output)
    _require(not output.exists(), f"refusing to overwrite evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "carts_opposition60_local_contact_v1",
        "status": "FAILED_CLOSED", "mode": args.mode, "object_id": None,
        "hardware_authorized": False, "formal_dynamic_pass": False,
        "research_dynamic_pass": False, "runtime_binding_accepted": False,
        "online_truth_used_for_control": False,
        "truth_evaluation_timing": "POST_STEP_LOGGING_AND_POST_RUN_GATE_ONLY_NO_TARGET_FEEDBACK",
        "preshape_replay_pass": False, "first_finger_controller_trace_pass": False,
        "first_finger_diagnostic_pass": False, "atomic_result_pass": False, "errors": []}
    app = None
    try:
        verified = _verify(args)
        report["object_id"] = verified["object_id"]
        report["candidate_id"] = verified["row"]["candidate_id"]
        report["evidence_binding"] = {name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in verified["paths"].items()}
        for name in ("hand_asset", "runtime_binding", "runtime_resources"):
            path = verified["bound"][name]
            report["evidence_binding"][f"runtime_{name}" if name == "hand_asset" else name] = {
                "path": str(path), "sha256": _sha256(path)}
        report["evidence_binding"]["runner_source"] = {
            "path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())}
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "multi_gpu": False,
                             "active_gpu": 0, "physics_gpu": 0})
        _run(verified, args.mode, report)
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error),
                                 "traceback": traceback.format_exc()})
    finally:
        output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "status": report["status"],
                          "atomic_result_pass": report["atomic_result_pass"]}, sort_keys=True),
              flush=True)
        if app is not None: app.close()
    return 0 if report["atomic_result_pass"] else 2
if __name__ == "__main__":
    raise SystemExit(main())
