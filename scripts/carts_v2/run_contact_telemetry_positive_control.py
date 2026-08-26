#!/usr/bin/env python3
"""Run the bounded A/B/C contact-telemetry control and q09 negative control."""
from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/carts_v2/contactopt_1488_fast6h"
SOURCE = RUN / "funnel_normal_first_b0_exact_run02/result.json"
TRACE = RUN / "object_b_normalfirst_top1_first_finger_run01/first_finger.json"
OUTPUT = RUN / "contact_telemetry_positive_control/result.json"
HAND = ROOT / "artifacts/kcg_connector/isaac/robot/hand_connector_no_nail_local/hand_connector_no_nail_local.usda"
OBJECT_MANIFEST = ROOT / "artifacts/kcg_connector/isaac/te_j35_free_tabletop_v1/MANIFEST.json"
CONFIG = ROOT / "src/kcg_connector/config/carts_surface_v2_fast6h.yaml"
HAND_ROOT, OBJECT_ROOT = "/World/Opposition60LocalHand", "/World/TE_J35FreePlug"
F1_ACTOR = HAND_ROOT + "/Geometry/handbase_link/f1Link1/f1Link2/f1Link3"
F1_LINKS = {"f1Link1": HAND_ROOT + "/Geometry/handbase_link/f1Link1",
            "f1Link2": HAND_ROOT + "/Geometry/handbase_link/f1Link1/f1Link2", "f1Link3": F1_ACTOR}
ACTIVE = ("f1j1", "f1j2", "f2j1", "f3j2")
MIMIC = {"f1j3": "f1j2", "f2j2": "f2j1", "f3j1": "f1j1", "f3j3": "f3j2"}
DT = 1.0 / 120.0
# Fixed diagnostic-fixture translation, not a candidate or model change (norm 0.596 mm).
FIXTURE_SHIFT_M = np.array([-1.56688059e-05, 8.72725248e-05, 5.88905586e-04])
Q_STATES = (("A_OUTSIDE_OFFSETS", .6250), ("B_POSITIVE_GAP_IN_OFFSETS", .6380),
            ("C_TASK_LIGHT_PRESS", .6420))

def _matrix(position, wxyz):
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    out = np.eye(4); out[:3, :3] = ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
        (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
        (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))); out[:3, 3] = position
    return out

def _quat(matrix):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.array([w, x, y, z])

def _worker():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    import torch
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.experimental.physics import Contact, ContactSensor
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics

    source, trace = json.loads(SOURCE.read_text()), json.loads(TRACE.read_text())
    candidate = next(row for row in source["candidates"]
                     if row["candidate_id"] == "contactopt_g_q05_a03_z0_p0")
    manifest = json.loads(OBJECT_MANIFEST.read_text()); object_asset = ROOT / manifest["asset"]

    def run_fixture(base, object_tf, states, seed_active):
        World.clear_instance(); SimulationManager.set_physics_sim_device("cuda:0")
        world = World(stage_units_in_meters=1.0, physics_dt=DT, backend="torch", device="cuda:0",
            sim_params={"use_gpu_pipeline": True, "gpu_found_lost_aggregate_pairs_capacity": 8192,
                        "gpu_total_aggregate_pairs_capacity": 16384})
        stage = get_current_stage()
        add_reference_to_stage(str(HAND), HAND_ROOT); add_reference_to_stage(str(object_asset), OBJECT_ROOT)
        for path, value in ((HAND_ROOT, base), (OBJECT_ROOT, object_tf)):
            prim = stage.GetPrimAtPath(path); xf = UsdGeom.Xformable(prim)
            if xf.GetOrderedXformOps(): xf.GetOrderedXformOps()[0].Set(Gf.Matrix4d(*value.T.ravel()))
            else: xf.AddTransformOp().Set(Gf.Matrix4d(*value.T.ravel()))
        articulations = [p for p in stage.Traverse() if p.GetPath().HasPrefix(Sdf.Path(HAND_ROOT))
                         and p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(articulations) != 1: raise RuntimeError("hand articulation identity changed")
        for path in (F1_ACTOR, OBJECT_ROOT):
            prim = stage.GetPrimAtPath(path)
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI): raise RuntimeError(f"not rigid body: {path}")
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        sensor_path = F1_ACTOR + "/ContactTelemetrySensor"
        sensor = ContactSensor(Contact.create(sensor_path, min_threshold=0.0,
            max_threshold=10000000.0, radius=-1.0, translations=[[0.0, 0.0, 0.0]]))
        robot = world.scene.add(SingleArticulation(str(articulations[0].GetPath()), "telemetry_hand"))
        part = world.scene.add(SingleRigidPrim(OBJECT_ROOT, "telemetry_object"))
        world.get_physics_context().set_gravity(0.0); world.reset()
        if not robot.handles_initialized or robot.num_dof != 8: raise RuntimeError("hand init failed")
        names = tuple(robot.dof_names); index = {name: i for i, name in enumerate(names)}
        active_index = np.array([index[name] for name in ACTIVE], np.int64)
        sim = SimulationManager.get_physics_simulation_view()
        body_views = {"hand": sim.create_rigid_body_view([F1_ACTOR]),
                      "object": sim.create_rigid_body_view([OBJECT_ROOT])}
        transform_views = {**{name: sim.create_rigid_body_view([path]) for name, path in F1_LINKS.items()},
                           "object": body_views["object"]}
        def offsets(view):
            contact = np.asarray(view.get_contact_offsets().numpy()).reshape(-1)
            rest = np.asarray(view.get_rest_offsets().numpy()).reshape(-1)
            return {"shape_count": int(view.max_shapes), "contact_offset_m": sorted(set(map(float, contact))),
                    "rest_offset_m": sorted(set(map(float, rest)))}
        runtime_offsets = {name: offsets(view) for name, view in body_views.items()}
        decoder = PhysicsSchemaTools.intToSdfPath; physx = get_physx_simulation_interface()
        collision_groups = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.CollisionGroup)]
        def parent(path):
            prim = stage.GetPrimAtPath(path)
            while prim and prim.IsValid() and not prim.HasAPI(UsdPhysics.RigidBodyAPI): prim = prim.GetParent()
            return str(prim.GetPath()) if prim and prim.IsValid() else None
        def shape(path):
            prim = stage.GetPrimAtPath(path); api = UsdPhysics.CollisionAPI(prim)
            rel = prim.GetRelationship("physics:filteredPairs")
            return {"path": path, "rigid_body_parent": parent(path),
                    "collision_enabled": bool(api.GetCollisionEnabledAttr().Get()) if prim.HasAPI(UsdPhysics.CollisionAPI) else None,
                    "filtered_pairs": [str(x) for x in rel.GetTargets()] if rel and rel.HasAuthoredTargets() else []}
        collision_prims = [p for p in stage.Traverse(Usd.TraverseInstanceProxies()) if
            p.HasAPI(UsdPhysics.CollisionAPI) and (p.GetPath().HasPrefix(Sdf.Path(F1_ACTOR)) or
            p.GetPath().HasPrefix(Sdf.Path(OBJECT_ROOT)))]
        collision_audit = {"shape_count": len(collision_prims),
            "enabled_count": sum(bool(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()) for p in collision_prims),
            "authored_filtered_pairs": {str(p.GetPath()): [str(x) for x in p.GetRelationship(
                "physics:filteredPairs").GetTargets()] for p in collision_prims if p.GetRelationship(
                "physics:filteredPairs").HasAuthoredTargets()}, "collision_groups": collision_groups,
            "rigid_bodies": {path: {"kinematic_enabled": bool(UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetKinematicEnabledAttr().Get()),
                "contact_report_threshold": float(PhysxSchema.PhysxContactReportAPI(stage.GetPrimAtPath(path)).GetThresholdAttr().Get())}
                for path in (F1_ACTOR, OBJECT_ROOT)}}
        def decode(headers, data):
            rows, hand_object = [], 0
            for header in headers:
                paths = [str(decoder(x)) for x in (header.actor0, header.collider0, header.actor1, header.collider1)]
                records = []
                for item in data[int(header.contact_data_offset):int(header.contact_data_offset)+int(header.num_contact_data)]:
                    impulse, normal = np.asarray(item.impulse, float), np.asarray(item.normal, float)
                    records.append({"position_m": list(map(float, item.position)), "normal": normal.tolist(),
                        "impulse_ns": impulse.tolist(), "normal_impulse_ns": float(impulse @ normal),
                        "separation_m": float(item.separation), "face_index_0": int(item.face_index0),
                        "face_index_1": int(item.face_index1)})
                is_pair = any(p == HAND_ROOT or p.startswith(HAND_ROOT+"/") for p in paths[:2]) != any(
                    p == HAND_ROOT or p.startswith(HAND_ROOT+"/") for p in paths[2:]) and any(
                    p == OBJECT_ROOT or p.startswith(OBJECT_ROOT+"/") for p in paths)
                if is_pair: hand_object += len(records)
                rows.append({"event_type": str(header.type), "actor_0": paths[0], "collider_0": shape(paths[1]),
                    "actor_1": paths[2], "collider_1": shape(paths[3]), "contact_data": records,
                    "project_category": "hand_object" if is_pair else "other"})
            return rows, hand_object
        event_reports = []
        subscription = physx.subscribe_full_contact_report_events(
            lambda headers, data, _friction: event_reports.append(decode(headers, data)))
        basic_event_reports = []
        basic_subscription = physx.subscribe_contact_report_events(
            lambda headers, data: basic_event_reports.append(decode(headers, data)))
        def telemetry(label, q):
            active = np.asarray(seed_active, np.float32).copy(); active[1] = q
            values = dict(zip(ACTIVE, active)); values.update({n: values[s] for n, s in MIMIC.items()})
            full = np.array([values[name] for name in names], np.float32)
            tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype, device=world.device)
            part.set_world_pose(tensor(object_tf[:3, 3]), tensor(_quat(object_tf)))
            part.set_linear_velocity(tensor(np.zeros(3))); part.set_angular_velocity(tensor(np.zeros(3)))
            robot.set_joint_positions(tensor(full)); robot.set_joint_velocities(tensor(np.zeros(8, np.float32)))
            robot.apply_action(ArticulationAction(joint_positions=tensor(active),
                joint_indices=tensor(active_index, torch.int64)))
            samples = []
            for state_step in range(2):
                event_reports.clear(); basic_event_reports.clear()
                world.step(render=False)
                headers, data, _ = physx.get_full_contact_report(); polled, polled_count = decode(headers, data)
                event_rows = [row for rows, _count in event_reports for row in rows]
                event_count = sum(count for _rows, count in event_reports)
                basic_rows = [row for rows, _count in basic_event_reports for row in rows]
                basic_count = sum(count for _rows, count in basic_event_reports)
                raw = sensor.get_raw_data(); sensor_rows = []
                vector = lambda value: ([float(value[key]) for key in ("x", "y", "z")]
                                        if isinstance(value, dict) else np.asarray(value, float).tolist())
                for item in raw:
                    sensor_rows.append({"body_0": str(decoder(int(item["body0"]))),
                        "body_1": str(decoder(int(item["body1"]))), "position_m": vector(item["position"]),
                        "normal": vector(item["normal"]), "impulse_ns": vector(item["impulse"])})
                reading = sensor.get_sensor_reading(); attr = lambda a, b, default=None: getattr(reading, a, getattr(reading, b, default))
                samples.append({"state": label, "state_step_index": state_step, "commanded_f1j2_rad": q,
                    "observed_f1j2_rad": float(np.asarray(robot.get_joint_positions().cpu())[index["f1j2"]]),
                    "observed_joint_positions_rad": dict(zip(names, map(float,
                        np.asarray(robot.get_joint_positions().cpu()).reshape(-1)))),
                    "runtime_rigid_transforms_position_xyzw": {name: np.asarray(view.get_transforms().numpy()).reshape(-1, 7)[0].tolist()
                        for name, view in transform_views.items()},
                    "physx_full_contact_report": event_rows, "physx_polled_full_contact_report": polled,
                    "physx_basic_contact_event_report": basic_rows,
                    "contact_sensor_raw": sensor_rows,
                    "contact_sensor_filtered_reading": {"is_valid": bool(attr("is_valid", "isValid", False)),
                        "in_contact": bool(attr("in_contact", "inContact", False)), "value_n": float(attr("value", "value", 0.0)),
                        "time_s": float(attr("time", "time", 0.0))},
                    "project_hand_object_count": max(event_count, basic_count, polled_count),
                    "collision_and_binding_snapshot": collision_audit, "runtime_offset_snapshot": runtime_offsets})
            return samples
        rows = [sample for label, q in states for sample in telemetry(label, q)]
        return {"states": rows, "runtime_offsets": runtime_offsets, "sensor": {"path": sensor_path,
            "parent_rigid_body": F1_ACTOR, "threshold_n": [0.0, 10000000.0], "radius_m": -1.0,
            "period_s": DT}, "collision_groups": collision_groups}

    base = np.asarray(candidate["target_world_from_handbase_row_major"], float).reshape(4, 4)
    from kcg_connector.grasp.carts_v2.b0_surface_semantics import bind_b0_external_load_bearing_surfaces
    from kcg_connector.grasp.carts_v2.models import load_v2_inputs
    inputs = bind_b0_external_load_bearing_surfaces(load_v2_inputs(ROOT, config_path=CONFIG,
        object_id="te_deutsch_d38999_26fj35pn_step"))
    object_tf = inputs.frozen_world_from_object.copy(); object_tf[:3, 3] += FIXTURE_SHIFT_M
    positive_seed = candidate["input_seed"]["pregrasp_joint_positions_rad"]
    positive = run_fixture(base, object_tf, Q_STATES, positive_seed)
    c_rows = [row for row in positive["states"] if row["state"] == "C_TASK_LIGHT_PRESS"]
    chain = any(sum(len(x["contact_data"]) for x in c["physx_basic_contact_event_report"] if
        x["project_category"] == "hand_object") > 0 and len(c["contact_sensor_raw"]) > 0 and
        c["contact_sensor_filtered_reading"]["is_valid"] and c["contact_sensor_filtered_reading"]["in_contact"] and
        c["project_hand_object_count"] > 0 for c in c_rows)
    negative = None
    if chain:
        sample = trace["samples"][503]
        q09_states = (("Q09_A13_STEP503_NEGATIVE", float(sample["joint_positions_rad"]["f1j2"])),)
        negative = run_fixture(np.asarray(sample["world_from_handbase_row_major"], float).reshape(4, 4),
            _matrix(sample["object_poses"][0]["position_m"], sample["object_poses"][0]["orientation_wxyz"]),
            q09_states, [sample["joint_positions_rad"][name] for name in ACTIVE])
    print("__CONTACT_TELEMETRY_JSON__=" + json.dumps({"positive": positive, "chain_pass": chain,
        "q09_negative": negative}, separators=(",", ":")), flush=True); app.close()

def _geometry_replay(telemetry, env):
    import diagnose_q09_a13_contact_asset_gap as d
    from kcg_connector.grasp.carts_v2.b0_surface_semantics import bind_b0_external_load_bearing_surfaces
    from kcg_connector.grasp.carts_v2.fast_filter import build_fcl_bvh_model
    from kcg_connector.grasp.carts_v2.models import load_v2_inputs
    inputs = bind_b0_external_load_bearing_surfaces(load_v2_inputs(ROOT, config_path=CONFIG,
        object_id="te_deutsch_d38999_26fj35pn_step")); surface = inputs.task_grip_surfaces["finger_1_pad"]
    raw, obj = d._raw_first_finger(surface), inputs.object_contract.model.mesh
    manifest = json.loads(OBJECT_MANIFEST.read_text()); isaac = ROOT.parent / "isaacsim/.conda-env"
    run = subprocess.run([str(isaac/"bin/python"), str(Path(d.__file__).resolve()), "--usd-worker",
        str(HAND), str(ROOT/manifest["asset"])], text=True, capture_output=True, env=env, check=True)
    usd = json.loads(run.stdout.rsplit("__PHYSX_RUNTIME_OFFSETS_JSON__=", 1)[1].splitlines()[0])
    object_collision, object_lineage = d._combine(usd["object"])
    pieces = [(row["link"], row["path"], *d._combine([row])) for row in usd["hand"]]
    object_raw = build_fcl_bvh_model(obj.vertices_m, obj.faces); object_faces = [("object", i) for i in range(len(obj.faces))]
    task_model = build_fcl_bvh_model(surface.points_local_m, surface.faces)
    task_faces = [("f1Link3", int(i)) for i in surface.source_face_indices]
    def matrix(value):
        value = np.asarray(value, float); x, y, z, w = value[3:]; out = np.eye(4)
        out[:3, :3] = ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
            (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
            (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))); out[:3, 3] = value[:3]; return out
    results = []
    for row in telemetry["positive"]["states"]:
        transforms = {name: matrix(value) for name, value in row["runtime_rigid_transforms_position_xyzw"].items()}
        task = d._query(task_model, transforms["f1Link3"], object_raw, transforms["object"], task_faces, object_faces)
        non = [d._query(item["model"], transforms[link], object_raw, transforms["object"],
            [(link, int(i)) for i in item["faces"]], object_faces) for link, item in raw.items()]
        task_tri = d._world_triangles(surface.triangles_local_m, transforms["f1Link3"])
        non_tri = np.concatenate([d._world_triangles(item["mesh"].face_vertices_m[item["faces"]], transforms[link])
                                  for link, item in raw.items()])
        classified = []
        for link, path, model, lineage in pieces:
            hit = d._query(model, transforms[link], object_collision, transforms["object"], lineage, object_lineage)
            point = np.asarray(hit["left_witness_world"]); is_task = link == "f1Link3" and d._point_distance(point, task_tri) < d._point_distance(point, non_tri)
            classified.append((is_task, path, hit))
        best = lambda values: min(values, key=lambda x: (not x[-1]["collision"], x[-1]["distance_m"]))
        task_piece, non_piece = best([x for x in classified if x[0]]), best([x for x in classified if not x[0]])
        results.append({"state": row["state"], "state_step_index": row["state_step_index"],
            "raw_task": {"collision": task["collision"], "distance_m": task["distance_m"]},
            "raw_non_task": {"collision": best([(False, "", x) for x in non])[-1]["collision"],
                             "distance_m": best([(False, "", x) for x in non])[-1]["distance_m"]},
            "physx_task": {"collision": task_piece[-1]["collision"], "distance_m": task_piece[-1]["distance_m"], "shape": task_piece[1]},
            "physx_non_task": {"collision": non_piece[-1]["collision"], "distance_m": non_piece[-1]["distance_m"], "shape": non_piece[1]}})
    return {"method": "POST_STEP_TENSOR_TRANSFORM_REPLAY_ON_CURRENT_RAW_AND_COMPOSED_COLLIDERS", "states": results}

def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    isaac = ROOT.parent / "isaacsim/.conda-env"; env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src/kcg_connector")
    env["LD_LIBRARY_PATH"] = f"{isaac/'lib'}:{env.get('LD_LIBRARY_PATH','')}"
    run = subprocess.run([str(isaac/"bin/python"), str(Path(__file__).resolve()), "--worker"],
                         text=True, capture_output=True, env=env, check=True)
    marker = "__CONTACT_TELEMETRY_JSON__="
    if marker not in run.stdout:
        raise RuntimeError("Isaac worker produced no telemetry marker:\n" + "\n".join(run.stderr.splitlines()[-30:]))
    telemetry = json.loads(run.stdout.rsplit(marker, 1)[1].splitlines()[0])
    offsets = telemetry["positive"]["runtime_offsets"]
    combined = [min(offsets["hand"]["contact_offset_m"]) + min(offsets["object"]["contact_offset_m"]),
        max(offsets["hand"]["contact_offset_m"]) + max(offsets["object"]["contact_offset_m"])]
    preflight = {"method": "EXACT_FCL_ON_CURRENT_COMPOSED_CONVEX_COLLIDERS_BEFORE_PHYSICS",
        "fixture_shift_world_m": FIXTURE_SHIFT_M.tolist(), "fixture_shift_norm_m": float(np.linalg.norm(FIXTURE_SHIFT_M)),
        "combined_contact_offset_range_m": combined,
        "states": {"A": {"task_gap_m": 0.00194, "non_task_gap_m": 0.00515},
                   "B": {"task_gap_m": 0.00013, "non_task_gap_m": 0.00306},
                   "C": {"task_collision": True, "non_task_gap_m": 0.00251}},
        "numerical_values_are_pre_run_search_witnesses_not_runtime_contact_truth": True}
    replay = _geometry_replay(telemetry, env)
    status = "DYNAMIC_PASS" if telemetry["chain_pass"] else "PARKED"
    report = {"schema_version": "contact_telemetry_positive_control_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": status, "scope": "LOCAL_DIAGNOSTIC_ONLY_NOT_GRASP_OR_ASSEMBLY_SUCCESS",
        "hardware_authorized": False, "formal_dynamic_pass": False, "research_dynamic_pass": False,
        "q09_a13_status": "REJECTED_NON_TASK_GEOMETRY_PRECEDES_TASK_CONTACT",
        "contact_infrastructure_before": "CONTACT_TELEMETRY_UNVERIFIED",
        "contact_infrastructure_after": "CONTACT_TELEMETRY_VERIFIED" if telemetry["chain_pass"] else "CONTACT_TELEMETRY_UNVERIFIED",
        "contact_offsets_or_rest_offsets_modified": False, "surface_v2_modified": False,
        "positive_geometry_preflight": preflight, "post_step_geometry_replay": replay, **telemetry}
    OUTPUT.parent.mkdir(parents=True, exist_ok=False); OUTPUT.write_text(json.dumps(report, indent=2)+"\n")
    print(OUTPUT)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker": _worker()
    else: main()
