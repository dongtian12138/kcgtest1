#!/usr/bin/env python3
"""Run the object-B opposition-60 initial-penetration gate, and nothing else."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ISAAC_V2 = ROOT / "src/kcg_connector/isaac/carts_v2"
sys.path[:0] = [str(ROOT / "src/kcg_connector"), str(ISAAC_V2)]

HAND_ROOT = "/World/Opposition60LocalHand"
ACTIVE = ("f1j1", "f1j2", "f2j1", "f3j2")
MIMIC = {"f1j3": "f1j2", "f2j2": "f2j1", "f3j1": "f1j1", "f3j3": "f3j2"}
EXPECTED_DOFS = set(ACTIVE) | set(MIMIC)
TERMINALS = ("f1Link3", "f2Link2", "f3Link3")
DT_S, DURATION_S = 1.0 / 120.0, 0.5
USD_READBACK_TOLERANCE = 1.0e-7


def _arguments() -> argparse.Namespace:
    base = ROOT / "artifacts/carts_v2/opposition60_isaac"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-binding", type=Path, default=base /
        "runtime_handarm_connector_no_nail_residual_vertex64_20260826_run03/"
        "RUNTIME_URDF_BINDING.json")
    parser.add_argument("--import-readback", type=Path,
                        default=base / "local_hand_import_readback_run01.json")
    parser.add_argument("--task-ik", type=Path, default=base /
        "qp60_anchor_a02_task_ik/"
        "opposition60_anchor_a02_exact_offset_00_count_01_static_control.json")
    parser.add_argument("--config", type=Path, default=ROOT /
        "src/kcg_connector/config/carts_nailfree_height_projected.yaml")
    parser.add_argument("--runtime-resources", type=Path, default=ROOT /
        "src/kcg_connector/config/carts_v2_isaac_runtime.json")
    parser.add_argument("--hand-asset", type=Path, default=ROOT /
        "artifacts/kcg_connector/isaac/robot/hand_connector_no_nail_local/"
        "hand_connector_no_nail_local.usda")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bound(path: Path | str, expected: str, label: str) -> Path:
    value = _resolve(path)
    _require(value.is_file() and _sha256(value) == expected, f"{label} hash changed")
    return value


def _host(value) -> np.ndarray:
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _values(value) -> list[float | None]:
    return [float(item) if math.isfinite(float(item)) else None
            for item in np.asarray(value).reshape(-1)]


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _candidate_inputs(task: dict, paths: dict[str, Path]):
    schema = task.get("schema_version")
    if schema == "carts_contactopt_b0_recheck_v1":
        rows = task.get("candidates"); _require(isinstance(rows, list), "B0 candidate rows are missing")
        ready = [row for row in rows if isinstance(row, dict) and row.get("local_isaac_input_ready") is True]
        row = ready[0] if len(ready) == 1 else {}
        seed, quality = row.get("input_seed") or {}, row.get("task_quality") or {}
        audit, source = task.get("b0_surface_audit") or {}, task.get("source") or {}
        _require(all((task.get("hardware_authorized") is False,
            task.get("formal_dynamic_pass") is False, task.get("research_dynamic_pass") is False,
            task.get("local_isaac_input_count") == len(ready) == 1,
            task.get("object_id") == seed.get("object_id") == "te_deutsch_d38999_26fj35pn_step",
            row.get("candidate_id") == seed.get("candidate_id"),
            row.get("sampled_raw_mesh_geometry_pass") is True,
            row.get("sampled_table_operation_clearance_pass") is True,
            row.get("nominal_12n_task_pass") is True,
            quality.get("nominal_gravity_lift_balance_pass") is True,
            float(quality.get("nominal_operation_force_cap_n", math.nan)) == 12.0,
            (row.get("bounded_ik") or {}).get("status") == "BOUNDED_IK_PASS_NOT_PATH_COLLISION",
            row.get("full_arm_path_collision_checked") is False,
            audit.get("method") == "EXTERNAL_LOAD_BEARING_SURFACE_B0",
            audit.get("legacy_primary_secondary_are_hard_gates") is False,
            audit.get("normal_alignment_is_object_semantic_hard_gate") is False)),
            "B0 local Isaac candidate gate changed")
        producer = _bound(source.get("path", ""), source.get("sha256", ""), "B0 recheck producer")
        base = _bound(task.get("base_physical_config", ""), task.get("base_physical_config_sha256", ""), "B0 base config")
        _bound(task.get("method_config", ""), task.get("method_config_sha256", ""), "B0 method config")
        _bound(task.get("seed_manifest", ""), task.get("seed_manifest_sha256", ""), "B0 seed manifest")
        expected = (ROOT / "scripts/carts_v2/run_contactopt_b0_recheck.py").resolve()
        _require(producer == expected and base == paths["config"], "B0 source or supplied config changed")
        return row, task["object_id"], seed["pregrasp_joint_positions_rad"], row["target_world_from_handbase_row_major"], schema

    palm_angle_deg = float(task.get("requested_palm_angle_deg", math.nan))
    _require(isinstance(task.get("selected_geometric_anchor_index"), int)
             and 0 <= task["selected_geometric_anchor_index"] < 12 and 45.0 <= palm_angle_deg <= 75.0
             and abs(palm_angle_deg - round(palm_angle_deg)) < 1e-9
             and task.get("survivor_count") == 1 and len(task.get("survivor_candidates") or []) == 1
             and len(task.get("task_and_bounded_ik") or []) == 1
             and task.get("hardware_authorized") is False and task.get("isaac_started") is False,
             "task/IK document is not one bounded opposition-60 survivor")
    anchor = task["selected_geometric_anchor"]
    survivor, row = task["survivor_candidates"][0], task["task_and_bounded_ik"][0]
    object_id = anchor["object_id"]
    _require(anchor["candidate_id"] == survivor["candidate_id"] == row["candidate_id"]
             and survivor["object_id"] == object_id
             and abs(float(anchor["palm_configuration_deg"]) - palm_angle_deg) < 1e-9
             and row["bounded_ik"]["status"] == "BOUNDED_IK_PASS_NOT_PATH_COLLISION"
             and row["task_quality"]["nominal_gravity_lift_balance_pass"] is True,
             "opposition-60 task/IK candidate identity changed")
    _bound(ROOT / "scripts/carts_v2/run_opposition60_static_control.py", task["script_sha256"], "task/IK producer")
    _require(_sha256(paths["config"]) == task["config_sha256"], "task/IK configuration hash changed")
    return row, object_id, anchor["pregrasp_joint_positions_rad"], row["target_world_from_handbase_row_major"], schema


def _verify_inputs(args: argparse.Namespace) -> dict:
    paths = {key: _resolve(getattr(args, key)) for key in
             ("runtime_binding", "import_readback", "task_ik", "config",
              "runtime_resources", "hand_asset")}
    for label, path in paths.items():
        _require(path.is_file(), f"{label} is missing: {path}")
    binding, readback, task = (_load(paths[name]) for name in
                               ("runtime_binding", "import_readback", "task_ik"))
    _require(binding.get("schema_version") == "carts_opposition60_runtime_urdf_binding_v1"
             and binding.get("hardware_authorized") is False
             and binding.get("formal_dynamic_pass") is False
             and binding.get("runtime_binding_accepted") is False,
             "run03 runtime binding boundary changed")
    local = binding.get("local_hand_urdf") or {}
    _require(local.get("revolute_joint_count") == 8
             and local.get("mimic_joint_count") == 4
             and local.get("terminal_compound_collision_count") == 192
             and local.get("total_collision_count") == 198,
             "run03 local-hand identity is not 8/4/192/198")
    _bound(local["path"], local["sha256"], "run03 local hand URDF")
    _bound(binding["generator"], binding["generator_sha256"], "runtime generator")
    _bound(binding["collision_manifest"], binding["collision_manifest_sha256"],
           "collision manifest")
    expected_gates = {"ISAAC_IMPORT": True, "INITIAL_PENETRATION": False,
                      "OPPOSITION60_REPLAY": False, "PHYSX_HEALTH": False}
    _require(readback.get("status") == "ISAAC_IMPORT_GATE_PASS"
             and readback.get("asset_scope") == "local-hand"
             and readback.get("runtime_gates") == expected_gates
             and readback.get("runtime_binding_accepted") is False
             and readback.get("hardware_authorized") is False
             and readback.get("formal_dynamic_pass") is False
             and readback.get("research_dynamic_pass") is False
             and readback.get("object_asset_loaded") is False,
             "local-hand import readback gate changed")
    _require(readback["runtime_binding"]["sha256"] == _sha256(paths["runtime_binding"]),
             "import readback does not bind current run03 manifest")
    rb = readback["readback"]
    _require(rb["runtime_dof_count"] == 8 and rb["runtime_dof_pass"] is True
             and len(rb["mimic"]["usd_rows"]) == 4 and rb["mimic"]["pass"] is True
             and rb["terminal_collisions"]["terminal_total"] == 192
             and rb["terminal_collisions"]["pass"] is True
             and readback["import_process_physx"]["statistics"]["scene_counts"]
                     ["physx_collision_shape_count"] == 198,
             "local import report is not the accepted 8/4/192/198 identity")
    layers = readback["usd_layers"]
    _require(_resolve(layers["root_usd"]["path"]) == paths["hand_asset"],
             "configured hand USD differs from import readback")
    _bound(paths["hand_asset"], layers["root_usd"]["sha256"], "local hand root USD")
    for row in layers["payload_layers"]:
        _bound(paths["hand_asset"].parent / row["path"], row["sha256"],
               f"USD payload {row['path']}")
    task_row, object_id, q, target_values, source_schema = _candidate_inputs(task, paths)
    _require(len(q) == 4, "active pregrasp joint vector changed")
    initial = {"f1j1": q[0], "f1j2": q[1], "f2j1": q[2], "f3j2": q[3]}
    initial.update({follower: initial[source] for follower, source in MIMIC.items()})
    target = np.asarray(target_values, dtype=np.float64).reshape(4, 4)
    _require(np.isfinite(target).all() and np.allclose(target[3], (0, 0, 0, 1)),
             "opposition handbase transform is invalid")
    return {"paths": paths, "binding": binding, "readback": readback,
            "task": task, "task_row": task_row, "object_id": object_id,
            "initial": initial, "target": target,
            "candidate_source_schema": source_schema}


def _world_matrix(Usd, UsdGeom, prim) -> np.ndarray:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.asarray(matrix, dtype=np.float64).T


def _contacts(interface, decoder, roots: dict[str, str]) -> dict:
    headers, _, _ = interface.get_full_contact_report()
    counts = {name: 0 for name in
              ("hand_object", "hand_table", "hand_fixture", "hand_self",
               "hand_unclassified", "object_table")}
    rows = []
    for header in headers:
        left = tuple(str(decoder(value)) for value in (header.actor0, header.collider0))
        right = tuple(str(decoder(value)) for value in (header.actor1, header.collider1))
        flags = {name: (any(_under(path, root) for path in left),
                        any(_under(path, root) for path in right))
                 for name, root in roots.items()}
        has = lambda name: flags[name][0] or flags[name][1]
        category = None
        if flags["hand"] == (True, True): category = "hand_self"
        elif has("hand") and has("object"): category = "hand_object"
        elif has("hand") and has("table"): category = "hand_table"
        elif has("hand") and has("fixture"): category = "hand_fixture"
        elif has("hand"): category = "hand_unclassified"
        elif has("object") and has("table"): category = "object_table"
        records = int(header.num_contact_data)
        if category:
            counts[category] += records
        rows.append({"side_0": list(left), "side_1": list(right),
                     "record_count": records, "category": category or "other"})
    return {"counts": counts, "paths": rows}


def _run(verified: dict, report: dict) -> None:
    import torch

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics
    from engine_health import (PhysxStatsMonitor, audit_physx_log,
        current_engine_log_path, gpu_backend_record, gpu_world_parameters,
        load_runtime_resources, synchronize_engine_log)
    from run_grasp_lift import prepare_dynamic_scene
    from kcg_connector.grasp.carts_v2.models import load_v2_inputs

    paths = verified["paths"]
    object_id = verified["object_id"]
    inputs = load_v2_inputs(ROOT, config_path=paths["config"], object_id=object_id)
    dynamic = inputs.config.section("dynamic")
    _require(float(dynamic["physics_dt_s"]) == DT_S
             and float(dynamic["maximum_joint_speed_rad_s"]) == 3.0
             and float(dynamic["measured_effort_abort_nm"]) == 0.9,
             "frozen dt/speed/effort gate changed")
    resources = load_runtime_resources(paths["runtime_resources"])
    log_path = current_engine_log_path()
    World.clear_instance()
    SimulationManager.set_physics_sim_device("cuda:0")
    world_parameters = gpu_world_parameters(resources)
    world_parameters["backend"] = "torch"
    world = World(stage_units_in_meters=1.0, physics_dt=DT_S,
                  rendering_dt=1.0 / 60.0, **world_parameters)
    context, stage = world.get_physics_context(), get_current_stage()
    entry = dynamic["object_scenes"][object_id]
    scene = prepare_dynamic_scene(ROOT, stage, entry, add_reference_to_stage)
    report["scene_binding"] = {
        "environment_scope": scene["environment_scope"],
        "object_asset": {"path": str(scene["object_asset"]),
                         "sha256": _sha256(Path(scene["object_asset"]))},
        "scene_evidence": [{"path": str(path), "sha256": _sha256(Path(path))}
                           for path in scene["evidence_paths"]],
        "table_top_z_m": float(scene["table_top_z_m"]),
    }
    add_reference_to_stage(str(paths["hand_asset"]), HAND_ROOT)
    root_prim = stage.GetPrimAtPath(HAND_ROOT)
    _require(root_prim.IsValid(), "local hand reference root is missing")
    root_xform = UsdGeom.Xformable(root_prim)
    _require(not root_xform.GetOrderedXformOps(), "local hand root has an unexpected transform")
    root_xform.AddTransformOp().Set(Gf.Matrix4d(*verified["target"].T.ravel().tolist()))
    articulation_roots = [prim for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(Sdf.Path(HAND_ROOT))
        and prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    _require(len(articulation_roots) == 1, "local hand articulation root count changed")
    hand_collisions = [prim for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.GetPath().HasPrefix(Sdf.Path(HAND_ROOT))
        and prim.HasAPI(UsdPhysics.CollisionAPI)]
    terminal_counts = {name: sum(name in str(prim.GetPath()).split("/")
        for prim in hand_collisions) for name in TERMINALS}
    live_mimic = {}
    for follower, source in MIMIC.items():
        prim = stage.GetPrimAtPath(f"{HAND_ROOT}/Physics/{follower}")
        relation = prim.GetRelationship("newton:mimicJoint") if prim.IsValid() else None
        targets = [] if relation is None else [str(item) for item in relation.GetTargets()]
        live_mimic[follower] = targets == [f"{HAND_ROOT}/Physics/{source}"]
    identity_stage_pass = (len(hand_collisions) == 198
                           and terminal_counts == {name: 64 for name in TERMINALS}
                           and all(live_mimic.values()))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    robot = world.scene.add(SingleArticulation(
        prim_path=str(articulation_roots[0].GetPath()), name="opposition60_local_hand"))
    objects = [world.scene.add(SingleRigidPrim(prim_path=path,
        name=f"opposition60_object_{index}"))
        for index, path in enumerate(scene["part_prim_paths"])]
    context.set_gravity(float(scene["gravity_m_s2"]))
    object_authored = _world_matrix(Usd, UsdGeom,
                                    stage.GetPrimAtPath(scene["roots"]["object"]))
    object_target = np.asarray(entry["frozen_settled_world_from_object_row_major"],
                               dtype=np.float64).reshape(4, 4)
    world.reset()
    _require(robot.handles_initialized, "local hand articulation did not initialize")
    dof_names = tuple(robot.dof_names)
    name_to_index = {name: index for index, name in enumerate(dof_names)}
    _require(robot.num_dof == 8 and set(dof_names) == EXPECTED_DOFS,
             "runtime local-hand DOF identity changed")
    initial = np.asarray([verified["initial"][name] for name in dof_names], np.float32)
    zeros = np.zeros(robot.num_dof, dtype=np.float32)
    runtime = lambda values, dtype=torch.float32: torch.as_tensor(
        values, dtype=dtype, device=world.device)
    robot.set_joints_default_state(positions=runtime(initial),
                                   velocities=runtime(zeros), efforts=runtime(zeros))
    robot.set_joint_positions(runtime(initial))
    robot.set_joint_velocities(runtime(zeros))
    active_indices = np.asarray([name_to_index[name] for name in ACTIVE], np.int32)
    follower_indices = np.asarray([name_to_index[name] for name in MIMIC], np.int32)
    active_kp = np.full(4, float(dynamic["hand_stiffness"]), np.float32)
    active_kd = np.full(4, float(dynamic["hand_damping"]), np.float32)
    active_cap = np.full(4, float(dynamic["hand_drive_maximum_effort_nm"]), np.float32)
    all_kp, all_kd, all_cap = zeros.copy(), zeros.copy(), zeros.copy()
    all_kp[active_indices], all_kd[active_indices] = active_kp, active_kd
    all_cap[active_indices] = active_cap
    joint_controller = robot.get_articulation_controller()
    joint_controller.set_gains(kps=runtime(all_kp), kds=runtime(all_kd),
                               save_to_usd=False)
    joint_controller.set_max_efforts(all_cap.tolist())
    robot.apply_action(ArticulationAction(
        joint_positions=runtime(initial[active_indices]),
        joint_indices=runtime(active_indices, torch.int64)))
    observed_kp, observed_kd = map(_host, joint_controller.get_gains())
    observed_caps = _host(joint_controller.get_max_efforts())
    drive_audit_pass = bool(
        np.allclose(observed_kp[active_indices], active_kp)
        and np.allclose(observed_kd[active_indices], active_kd)
        and np.allclose(observed_caps[active_indices], active_cap)
        and np.allclose(observed_kp[follower_indices], 0.0)
        and np.allclose(observed_kd[follower_indices], 0.0)
        and np.allclose(observed_caps[follower_indices], 0.0))
    before_root = _world_matrix(Usd, UsdGeom, articulation_roots[0])
    before_positions = _host(robot.get_joint_positions()).reshape(-1)
    monitor = PhysxStatsMonitor(context)
    interface = get_physx_simulation_interface()
    roots = {"hand": HAND_ROOT, **scene["roots"]}
    samples, contact_totals = [], {name: 0 for name in
        ("hand_object", "hand_table", "hand_fixture", "hand_self",
         "hand_unclassified", "object_table")}
    finite = True
    maximum_speed = maximum_effort = 0.0
    steps = int(round(DURATION_S / DT_S))
    for step in range(steps):
        world.step(render=False)
        monitor.sample()
        positions = _host(robot.get_joint_positions()).reshape(-1)
        velocities = _host(robot.get_joint_velocities()).reshape(-1)
        efforts = _host(robot.get_measured_joint_efforts()).reshape(-1)
        object_poses = [tuple(_host(value).reshape(-1) for value in item.get_world_pose())
                        for item in objects]
        contacts = _contacts(interface, PhysicsSchemaTools.intToSdfPath, roots)
        for name, count in contacts["counts"].items(): contact_totals[name] += count
        row_finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all()
                          and np.isfinite(efforts).all()
                          and all(np.isfinite(value).all() for pose in object_poses
                                  for value in pose))
        finite &= row_finite
        if np.isfinite(velocities).all(): maximum_speed = max(maximum_speed,
            float(np.max(np.abs(velocities))))
        if np.isfinite(efforts).all(): maximum_effort = max(maximum_effort,
            float(np.max(np.abs(efforts))))
        samples.append({"step": step, "simulation_time_s": (step + 1) * DT_S,
            "joint_positions_rad": dict(zip(dof_names, _values(positions))),
            "joint_velocities_rad_s": dict(zip(dof_names, _values(velocities))),
            "joint_efforts_nm": dict(zip(dof_names, _values(efforts))),
            "active_targets_rad": dict(zip(ACTIVE, _values(initial[active_indices]))),
            "active_target_errors_rad": dict(zip(ACTIVE, _values(
                initial[active_indices] - positions[active_indices]))),
            "mimic_errors_rad": {follower: (float(positions[name_to_index[follower]]
                - positions[name_to_index[source]]) if row_finite else None)
                for follower, source in MIMIC.items()},
            "object_poses": [{"position_m": _values(pose[0]),
                              "orientation_wxyz": _values(pose[1])}
                             for pose in object_poses],
            "contacts": contacts})
    after_root = _world_matrix(Usd, UsdGeom, articulation_roots[0])
    sync = synchronize_engine_log(log_path)
    log = audit_physx_log(log_path, cutoff_bytes=sync["audit_byte_count"],
                          required_marker=sync["marker"])
    stats, backend = monitor.summary(), gpu_backend_record(world, context)
    capacity_pass = bool(stats["physx_statistics_sample_count"] > 0
        and stats["physx_statistics_read_failures"] == 0
        and stats["observed_gpu_found_lost_aggregate_pairs_peak"]
            < stats["configured_gpu_found_lost_aggregate_pairs_capacity"]
        and stats["observed_gpu_total_aggregate_pairs_peak"]
            < stats["configured_gpu_total_aggregate_pairs_capacity"])
    engine_observation_pass = bool(backend["pass"] and capacity_pass
        and log.get("scan_complete") is True and log.get("capacity_warning_count") == 0
        and log.get("physx_error_lines") == [])
    forbidden_contacts = sum(contact_totals[name] for name in contact_totals
                             if name.startswith("hand_"))
    pose_finite = all(np.isfinite(value).all() for value in
                      (before_root, after_root, object_authored, object_target))
    transform_error = (max(float(np.max(np.abs(before_root - verified["target"]))),
                           float(np.max(np.abs(after_root - verified["target"]))))
                       if pose_finite else None)
    object_transform_error = (float(np.max(np.abs(object_authored - object_target)))
                              if pose_finite else None)
    initial_q_error = (float(np.max(np.abs(before_positions - initial)))
                       if np.isfinite(before_positions).all() else None)
    object_motion = None
    if samples and finite:
        first_pose, last_pose = (samples[index]["object_poses"][0]
                                 for index in (0, -1))
        first_position = np.asarray(first_pose["position_m"], np.float64)
        last_position = np.asarray(last_pose["position_m"], np.float64)
        first_quaternion = np.asarray(first_pose["orientation_wxyz"], np.float64)
        last_quaternion = np.asarray(last_pose["orientation_wxyz"], np.float64)
        cosine = abs(float(np.dot(first_quaternion, last_quaternion)
                           / np.linalg.norm(first_quaternion)
                           / np.linalg.norm(last_quaternion)))
        angle = 2.0 * math.acos(np.clip(cosine, 0.0, 1.0))
        radius = float(np.max(np.linalg.norm(
            inputs.object_contract.model.mesh.vertices_m, axis=1)))
        motion_bound = (float(np.linalg.norm(last_position - first_position))
                        + 2.0 * radius * math.sin(0.5 * angle))
        object_motion = {"position_delta_m": float(np.linalg.norm(
            last_position - first_position)), "orientation_delta_rad": angle,
            "maximum_surface_motion_bound_m": motion_bound,
            "registered_limit_m": float(dynamic["lift_tolerance_m"]),
            "pass": motion_bound <= float(dynamic["lift_tolerance_m"])}
    identity_pass = bool(identity_stage_pass and robot.num_dof == 8
                         and len(MIMIC) == 4)
    gate = bool(identity_pass and drive_audit_pass and pose_finite
        and transform_error is not None and transform_error <= 1e-7
        and initial_q_error is not None and initial_q_error <= 1e-6
        and object_transform_error is not None
        and object_transform_error <= USD_READBACK_TOLERANCE
        and object_motion is not None and object_motion["pass"]
        and finite and maximum_speed <= 3.0 and maximum_effort <= 0.9
        and forbidden_contacts == 0 and engine_observation_pass)
    report.update({"status": "INITIAL_PENETRATION_PASS" if gate else
                   "INITIAL_PENETRATION_FAILED", "runtime_gates": {
        "ISAAC_IMPORT": True, "INITIAL_PENETRATION": gate,
        "OPPOSITION60_REPLAY": False, "PHYSX_HEALTH": False},
        "physics": {"dt_s": DT_S, "step_count": steps,
                    "physics_time_advanced_s": steps * DT_S,
                    "backend": backend, "statistics": stats, "log": log,
                    "engine_observation_pass_for_this_gate": engine_observation_pass},
        "asset_identity": {"pass": identity_pass, "dof_count": robot.num_dof,
            "dof_names": list(dof_names), "active_drive_count": 4,
            "mimic_count": 4, "terminal_collision_count": sum(terminal_counts.values()),
            "total_hand_collision_count": len(hand_collisions),
            "terminal_counts": terminal_counts, "live_mimic": live_mimic},
        "initial_state": {"target_by_dof_rad": verified["initial"],
            "maximum_joint_error_before_first_step_rad": initial_q_error,
            "active_joint_names": list(ACTIVE),
            "active_stiffness": float(dynamic["hand_stiffness"]),
            "active_damping": float(dynamic["hand_damping"]),
            "active_maximum_effort_nm": float(dynamic["hand_drive_maximum_effort_nm"]),
            "drive_audit_pass": drive_audit_pass,
            "mimic_followers_have_zero_independent_drive": drive_audit_pass},
        "pose_binding": {"world_from_handbase_target_row_major": verified["target"].ravel().tolist(),
            "world_from_handbase_before_row_major": _values(before_root),
            "world_from_handbase_after_row_major": _values(after_root),
            "maximum_absolute_transform_error": transform_error,
            "object_authored_transform_error": object_transform_error,
            "usd_readback_tolerance": USD_READBACK_TOLERANCE,
            "hand_root_pose_writes_after_first_step": 0,
            "object_pose_writes_after_first_step": 0},
        "safety": {"all_samples_finite": finite,
            "maximum_absolute_joint_speed_rad_s": maximum_speed,
            "maximum_absolute_joint_effort_nm": maximum_effort,
            "speed_limit_rad_s": 3.0, "effort_limit_nm": 0.9,
            "object_stationary_hold": object_motion,
            "contact_totals": contact_totals,
            "forbidden_hand_contact_record_count": forbidden_contacts},
        "samples": samples})


def main() -> int:
    args = _arguments()
    output_dir = _resolve(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "initial_penetration.json"
    _require(not output.exists(), f"refusing to overwrite evidence: {output}")
    report = {"schema_version": "carts_opposition60_initial_penetration_v1",
        "status": "INITIAL_PENETRATION_FAILED", "object_id": None,
        "mode": "initial-penetration", "hardware_authorized": False,
        "formal_dynamic_pass": False, "research_dynamic_pass": False,
        "runtime_binding_accepted": False, "online_truth_used_for_control": False,
        "truth_evaluation_timing": "POST_STEP_LOGGING_AND_POST_RUN_GATE_ONLY_NO_TARGET_FEEDBACK",
        "closure_command_count": 0, "lift_command_count": 0,
        "runtime_gates": {"ISAAC_IMPORT": False, "INITIAL_PENETRATION": False,
                          "OPPOSITION60_REPLAY": False, "PHYSX_HEALTH": False},
        "errors": []}
    app = None
    try:
        verified = _verify_inputs(args)
        report["object_id"] = verified["object_id"]
        report["candidate_source_schema"] = verified["candidate_source_schema"]
        report["evidence_binding"] = {name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in verified["paths"].items()}
        report["evidence_binding"]["runner_source"] = {
            "path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())}
        report["candidate_id"] = verified["task_row"]["candidate_id"]
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "multi_gpu": False,
                             "active_gpu": 0, "physics_gpu": 0})
        _run(verified, report)
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error),
                                 "traceback": traceback.format_exc()})
    finally:
        output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "status": report["status"],
                          "runtime_binding_accepted": False}, sort_keys=True), flush=True)
        if app is not None: app.close()
    return 0 if report["runtime_gates"]["INITIAL_PENETRATION"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
