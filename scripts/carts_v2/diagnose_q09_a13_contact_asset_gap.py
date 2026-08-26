#!/usr/bin/env python3
"""Offline A/B/C mesh-distance diagnostic for the saved q09_a13 trace."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path; from datetime import datetime, timezone
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; RUN = ROOT / "artifacts/carts_v2/contactopt_1488_fast6h"
TRACE = RUN / "object_b_normalfirst_top1_first_finger_run01/first_finger.json"; EVALUATION = TRACE.with_name("evaluation.json")
CONFIG = ROOT / "src/kcg_connector/config/carts_surface_v2_fast6h.yaml"; HAND_USD = ROOT / "artifacts/kcg_connector/isaac/robot/hand_connector_no_nail_local/hand_connector_no_nail_local.usda"
OBJECT_MANIFEST = ROOT / "artifacts/kcg_connector/isaac/te_j35_free_tabletop_v1/MANIFEST.json"; RUNTIME_BINDING = ROOT / "artifacts/carts_v2/opposition60_isaac/runtime_handarm_connector_no_nail_residual_vertex64_20260826_run03/RUNTIME_URDF_BINDING.json"
TASK_MANIFEST = ROOT / "artifacts/carts_v2/nailfree_height_projected/task_grip_surface_audit/TASK_GRIP_SURFACE_MANIFEST.json"; OUTPUT = RUN / "q09_a13_contact_asset_differential"
OBJECT_ID, LINKS = "te_deutsch_d38999_26fj35pn_step", ("f1Link1", "f1Link2", "f1Link3")
def _usd_worker(hand_path, object_path):
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    from isaacsim.core.api import World; from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from pxr import Usd, UsdGeom, UsdPhysics
    World.clear_instance(); SimulationManager.set_physics_sim_device("cuda:0"); world = World(stage_units_in_meters=1.0, physics_dt=1/120, backend="torch", device="cuda:0", sim_params={"use_gpu_pipeline": True, "gpu_found_lost_aggregate_pairs_capacity": 8192, "gpu_total_aggregate_pairs_capacity": 16384})
    for path, asset in (("/World/Opposition60LocalHand", hand_path), ("/World/TE_J35FreePlug", object_path)): add_reference_to_stage(asset, path)
    world.reset(); stage = get_current_stage()
    def matrix(prim):
        value = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.asarray(value, dtype=np.float64).T
    def geometry(prim, frame):
        mesh = UsdGeom.Mesh(prim); points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform = np.linalg.inv(matrix(frame)) @ matrix(prim)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64); indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        faces, cursor = [], 0
        for count in counts:
            polygon = indices[cursor:cursor + count]
            faces.extend((int(polygon[0]), int(polygon[i]), int(polygon[i + 1])) for i in range(1, int(count) - 1))
            cursor += int(count)
        return points.tolist(), faces
    hand_root = stage.GetPrimAtPath("/World/Opposition60LocalHand"); object_root = stage.GetPrimAtPath("/World/TE_J35FreePlug")
    rows, filters, groups = [], [], []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        relation = prim.GetRelationship("physics:filteredPairs")
        if relation and relation.HasAuthoredTargets():
            filters.append({"prim": str(prim.GetPath()), "targets": [str(item) for item in relation.GetTargets()]})
        if prim.IsA(UsdPhysics.CollisionGroup):
            groups.append(str(prim.GetPath()))
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if path.startswith(str(object_root.GetPath())):
            owner, link, side = object_root, "object", "object"
        elif path.startswith(str(hand_root.GetPath())):
            link = next((name for name in reversed(LINKS) if f"/{name}" in path), "")
            if not link:
                continue
            owner, side = prim, "hand"
            while owner and owner.GetName() != link:
                owner = owner.GetParent()
        else:
            continue
        points, faces = geometry(prim, owner)
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr(); approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        authored = lambda name: bool(prim.GetAttribute(name).HasAuthoredValueOpinion())
        rows.append({"side": side, "link": link, "path": path, "points": points, "faces": faces,
            "collision_enabled": bool(enabled.Get()), "collision_enabled_authored": enabled.HasAuthoredValueOpinion(), "approximation": approximation.Get(),
            "contact_offset_authored": authored("physxCollision:contactOffset"), "rest_offset_authored": authored("physxCollision:restOffset")})
    hand = [row for row in rows if row["side"] == "hand"]; obj = [row for row in rows if row["side"] == "object"]
    if len(hand) != 66 or len(obj) != 64:
        raise RuntimeError(f"unexpected collider counts hand={len(hand)} object={len(obj)}")
    base = "/World/Opposition60LocalHand/Geometry/handbase_link/f1Link1"
    actor_paths = {"f1Link1": base, "f1Link2": base + "/f1Link2", "f1Link3": base + "/f1Link2/f1Link3", "object": "/World/TE_J35FreePlug"}
    runtime, sim = {}, SimulationManager.get_physics_simulation_view()
    for name, path in actor_paths.items():
        view = sim.create_rigid_body_view([path])
        contact = view.get_contact_offsets().numpy().reshape(view.count, view.max_shapes)[0]; rest = view.get_rest_offsets().numpy().reshape(view.count, view.max_shapes)[0]
        runtime[name] = {"actor_path": path, "shape_count": int(view.max_shapes),
            "contact_offsets_m": contact.tolist(), "rest_offsets_m": rest.tolist()}
    context = world.get_physics_context(); backend = {"actual_data_backend": str(world.backend), "world_device": str(world.device), "physics_context_device": str(context.device), "gpu_sim": bool(context.use_gpu_sim), "gpu_pipeline": bool(context.use_gpu_pipeline), "gpu_dynamics_enabled": bool(context.is_gpu_dynamics_enabled()), "broadphase_type": str(context.get_broadphase_type())}
    payload = {"hand": hand, "object": obj, "filtered_pairs": filters, "collision_groups": groups, "runtime_offsets": runtime, "runtime_backend": backend}
    print("__PHYSX_RUNTIME_OFFSETS_JSON__=" + json.dumps(payload, separators=(",", ":")), flush=True); app.close()
if len(sys.argv) > 1 and sys.argv[1] == "--usd-worker":
    _usd_worker(sys.argv[2], sys.argv[3]); raise SystemExit(0)
import fcl, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull; from trimesh.triangles import closest_point
from kcg_connector.grasp.carts_v2.b0_surface_semantics import bind_b0_external_load_bearing_surfaces; from kcg_connector.grasp.carts_v2.fast_filter import build_fcl_bvh_model; from kcg_connector.grasp.carts_v2.models import load_v2_inputs; from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh
def _transform(position, quaternion):
    w, x, y, z = np.asarray(quaternion, np.float64) / np.linalg.norm(quaternion)
    out = np.eye(4)
    out[:3, :3] = ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)), (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
                    (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))
    out[:3, 3] = position
    return out
def _fk(model, sample):
    result = {model.base_link: np.asarray(sample["world_from_handbase_row_major"], np.float64).reshape(4, 4)}
    for name in model.joint_order:
        joint = model.joints[name]
        position = sample["joint_positions_rad"][name] if joint.movable else 0.0
        result[joint.child_link] = result[joint.parent_link] @ joint.origin_transform() @ joint.motion_transform(position)
    return result
def _query(left_model, left_tf, right_model, right_tf, left_faces, right_faces):
    def collision_object(model, transform):
        value = fcl.CollisionObject(model); value.setTransform(fcl.Transform(transform[:3, :3], transform[:3, 3]))
        return value
    left, right = collision_object(left_model, left_tf), collision_object(right_model, right_tf); collision = fcl.CollisionResult()
    hit = bool(fcl.collide(left, right, fcl.CollisionRequest(num_max_contacts=1, enable_contact=True), collision))
    if hit:
        contact, distance = collision.contacts[0], 0.0; left_point = right_point = np.asarray(contact.pos, np.float64).tolist()
        left_face, right_face = int(contact.b1), int(contact.b2)
    else:
        result = fcl.DistanceResult(); distance = float(fcl.distance(left, right, fcl.DistanceRequest(enable_nearest_points=True), result))
        if distance < 0 or not np.isfinite(distance) or len(result.nearest_points) != 2:
            raise RuntimeError("invalid FCL free-space result")
        left_point, right_point = [np.asarray(point).tolist() for point in result.nearest_points]
        left_face, right_face = int(result.b1), int(result.b2)
    return {"distance_m": distance, "collision": hit,
            "left_witness_world": left_point, "right_witness_world": right_point,
            "left_face": left_faces[left_face], "right_face": right_faces[right_face]}
def _combine(rows):
    points, faces, lineage, offset = [], [], [], 0
    for row in rows:
        vertex, face = np.asarray(row["points"]), np.asarray(row["faces"])
        points.append(vertex); faces.append(face + offset); offset += len(vertex)
        lineage.extend((row["path"], int(index)) for index in range(len(face)))
    return build_fcl_bvh_model(np.vstack(points), np.vstack(faces)), lineage
def _raw_first_finger(surface):
    paths = {"f1Link1": "src/iiwa_description/meshes/hand/f1Link1.STL",
             "f1Link2": "src/iiwa_description/meshes/hand/f1Link2.STL",
             "f1Link3": "src/iiwa_description/meshes/hand/connector_no_nail/f1Link3_nailfree.stl"}
    result = {}
    for link, relative in paths.items():
        mesh, provenance = load_stl_mesh(ROOT / relative, unit="m", orient_outward=False); keep = np.ones(len(mesh.faces), dtype=bool)
        if link == "f1Link3":
            keep[surface.source_face_indices] = False
        source = np.flatnonzero(keep)
        result[link] = {"mesh": mesh, "faces": source,
            "model": build_fcl_bvh_model(mesh.vertices_m, mesh.faces[source]),
            "path": str(ROOT / relative), "sha256": provenance.source_sha256}
    return result
def _world_triangles(triangles, transform): return triangles @ transform[:3, :3].T + transform[:3, 3]
def _point_distance(point, triangles): return float(np.min(np.linalg.norm(closest_point(triangles, np.repeat(np.asarray(point)[None, :], len(triangles), axis=0)) - point, axis=1)))
def _asset(path): return {"path": str(path), "sha256": file_sha256(path)}
def _query_fields(task, physx, non_task):
    return {"task_exact_min_distance_m": task["distance_m"], "task_exact_collision": task["collision"],
        "task_exact_hand_witness_world": task["left_witness_world"], "task_exact_object_witness_world": task["right_witness_world"], "task_exact_hand_face_id": task["left_face"][1], "task_exact_object_face_id": task["right_face"][1],
        "physx_collider_min_distance_m": physx["distance_m"], "physx_collider_collision": physx["collision"],
        "closest_hand_collision_piece": physx["left_face"][0], "closest_object_collision_piece": physx["right_face"][0], "physx_hand_witness_world": physx["left_witness_world"], "physx_object_witness_world": physx["right_witness_world"],
        "non_task_exact_min_distance_m": non_task["distance_m"], "non_task_exact_collision": non_task["collision"],
        "non_task_hand_witness_world": non_task["left_witness_world"], "non_task_object_witness_world": non_task["right_witness_world"], "non_task_hand_link": non_task["left_face"][0], "non_task_hand_face_id": non_task["left_face"][1], "non_task_object_face_id": non_task["right_face"][1], "asset_gap_bias_m": physx["distance_m"] - task["distance_m"],
        "task_vs_non_task_lead_m": non_task["distance_m"] - task["distance_m"]}
def main():
    if OUTPUT.exists(): raise FileExistsError("diagnostic output directory already exists")
    trace, evaluation = [json.loads(path.read_text()) for path in (TRACE, EVALUATION)]
    geometry = evaluation["geometry"]; failures = geometry["safety_failure_indices"]
    if not (evaluation["sample_count"] == evaluation["safety_evaluation_count"] == len(trace["samples"]) == 504 and failures == geometry["full_evaluation_failure_indices"] == [503] and geometry["minimum_clearances"]["self_m"] > 0 and geometry["minimum_clearances"]["table_m"] > 0):
        raise ValueError("trace-bound full-state safety scan changed")
    main_index, indices = failures[0] - 1, range(failures[0] - 1, failures[0] + 1)
    inputs = bind_b0_external_load_bearing_surfaces(load_v2_inputs(
        ROOT, config_path=CONFIG, object_id=OBJECT_ID))
    surface = inputs.task_grip_surfaces["finger_1_pad"]
    raw, object_mesh = _raw_first_finger(surface), inputs.object_contract.model.mesh
    containment = {}
    for link in LINKS[:2]:
        path = ROOT / f"src/iiwa_description/meshes/hand/collision/{link}_convex.stl"; convex, provenance = load_stl_mesh(path, unit="m", orient_outward=False)
        hull = ConvexHull(convex.vertices_m); violation = float(np.max(raw[link]["mesh"].vertices_m @ hull.equations[:, :3].T + hull.equations[:, 3]))
        containment[link] = {"raw_within_collision_convex": violation <= 1e-10, "maximum_halfspace_violation_m": violation, "tolerance_m": 1e-10, "convex_asset": {"path": str(path), "sha256": provenance.source_sha256}}
    if not all(row["raw_within_collision_convex"] for row in containment.values()): raise RuntimeError("full-trace convex selector does not contain proximal raw visual mesh")
    object_model = build_fcl_bvh_model(object_mesh.vertices_m, object_mesh.faces); object_faces = [("object", int(index)) for index in range(len(object_mesh.faces))]
    task_model = build_fcl_bvh_model(surface.points_local_m, surface.faces); task_faces = [(surface.link_name, int(index)) for index in surface.source_face_indices]
    object_manifest, binding = [json.loads(path.read_text())
                                for path in (OBJECT_MANIFEST, RUNTIME_BINDING)]
    object_usd, collision_manifest = (ROOT / object_manifest["asset"],
                                       Path(binding["collision_manifest"]))
    isaac, env = ROOT.parent / "isaacsim/.conda-env", dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{isaac / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    worker = subprocess.run([str(isaac / "bin/python"), str(Path(__file__).resolve()),
        "--usd-worker", str(HAND_USD), str(object_usd)], check=True,
        text=True, capture_output=True, env=env)
    marker = "__PHYSX_RUNTIME_OFFSETS_JSON__="; usd = json.loads(worker.stdout.rsplit(marker, 1)[1].splitlines()[0])
    colliders = {link: _combine([row for row in usd["hand"] if row["link"] == link])
                 for link in LINKS}
    object_collider, object_lineage = _combine(usd["object"])
    metadata = usd["hand"] + usd["object"]
    if (not all(row["collision_enabled"] and row["approximation"] == "convexHull"
                for row in metadata)
            or any(row["contact_offset_authored"] or row["rest_offset_authored"]
                   for row in metadata)):
        raise RuntimeError("composed collider metadata changed")
    rows = []
    for index in indices:
        sample, candidates = trace["samples"][index], {"non_task": [], "physx": []}
        fk, pose = _fk(inputs.hand_model, sample), sample["object_poses"][0]
        world_object = _transform(pose["position_m"], pose["orientation_wxyz"])
        task = _query(task_model, fk["f1Link3"], object_model, world_object,
                      task_faces, object_faces)
        for link in LINKS:
            item, (collider, lineage) = raw[link], colliders[link]
            source = [(link, int(face)) for face in item["faces"]]
            candidates["non_task"].append(_query(
                item["model"], fk[link], object_model, world_object, source, object_faces))
            physx = _query(collider, fk[link], object_collider, world_object,
                           lineage, object_lineage)
            physx["closest_hand_link"] = link
            candidates["physx"].append(physx)
        best = lambda values: min(values, key=lambda row: (not row["collision"], row["distance_m"]))
        rows.append({"sample_index": index, "step": sample["step"],
            "simulation_time_s": sample["simulation_time_s"], "task": task,
            "physx": best(candidates["physx"]), "non_task": best(candidates["non_task"])})
    main_row, first = rows[-2], rows[-1]
    if main_row["sample_index"] != main_index or not first["non_task"]["collision"] or first["task"]["collision"]:
        raise RuntimeError("adjacent-state geometric-invalid discriminator failed")
    usd_filter_found = bool(usd["filtered_pairs"] or usd["collision_groups"])
    runtime = usd["runtime_offsets"]; contact_values = [value for row in runtime.values() for value in row["contact_offsets_m"]]
    rest_values = [value for row in runtime.values() for value in row["rest_offsets_m"]]
    if ([runtime[name]["shape_count"] for name in (*LINKS, "object")] != [1, 1, 64, 64] or len(set(contact_values)) != 1 or len(set(rest_values)) != 1 or usd_filter_found
            or usd["runtime_backend"] != {"actual_data_backend": "torch", "world_device": "cuda:0", "physics_context_device": "cuda:0", "gpu_sim": True, "gpu_pipeline": True, "gpu_dynamics_enabled": True, "broadphase_type": "GPU"}):
        raise RuntimeError("runtime shape offset mapping is not uniquely resolved")
    hand_contact = contact_values[0]; object_contact = runtime["object"]["contact_offsets_m"][0]
    hand_rest = rest_values[0]; object_rest = runtime["object"]["rest_offsets_m"][0]
    classification = "PHYSX_CONTACT_REPORTING_OR_FILTERING_ERROR"
    next_action = ("CHECK_ONLY_COLLISION_ENABLE_FILTER_CONTACT_REPORT_API_SUBSCRIPTION_"
                   "ARTICULATION_RIGID_BODY_BINDING_OFFSETS_AND_PRIM_PATHS; DO_NOT_CHANGE_POSE_OR_GENERATOR")
    sample, fk = trace["samples"][main_index], _fk(inputs.hand_model, trace["samples"][main_index])
    witness = np.asarray(main_row["physx"]["left_witness_world"])
    witness_link = main_row["physx"]["closest_hand_link"]
    witness_local = (np.linalg.inv(fk[witness_link]) @ np.r_[witness, 1])[:3]
    task_world = _world_triangles(surface.triangles_local_m, fk["f1Link3"])
    non_task_world = np.concatenate([_world_triangles(
        item["mesh"].face_vertices_m[item["faces"]], fk[link]) for link, item in raw.items()])
    task_distance, non_task_distance = (_point_distance(witness, triangles)
                                        for triangles in (task_world, non_task_world))
    offset = {"usd_authored": False, "schema_fallback": "-inf",
              "effective_source": "PHYSX_TENSOR_BACKEND_OBSERVED_AFTER_RESET"}
    task, physx, non_task = (main_row[name] for name in ("task", "physx", "non_task"))
    in_contact_range = physx["distance_m"] <= hand_contact + object_contact
    if not in_contact_range or evaluation["physx"]["hand_object_record_count"] != 0:
        raise RuntimeError("saved zero-contact/runtime contact-range discriminator changed")
    report = {"schema_version": "q09_a13_contact_asset_differential_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate_id": trace["candidate_id"], "status": "OFFLINE_PASS",
        "hardware_authorized": False, "formal_dynamic_pass": False, "research_dynamic_pass": False,
        "state_selection": {"method": "LAST_DISCRETE_STATE_BEFORE_FIRST_RAW_NON_TASK_INTERSECTION", "full_trace_selector": "BOUND_504_STATE_CONVEX_CONTAINS_PROXIMAL_RAW_PLUS_RAW_TERMINAL_SCAN_AND_LOCAL_CAUSE_QUERY", "proximal_raw_visual_containment_proof": containment, "first_intersection_sample_index": failures[0], "q_safe_max_fallback_used": False, "checked_existing_sample_indices": list(indices), "proximity_0p75mm_used_for_selection": False},
        "last_exactly_nonintersecting_state": {"sample_index": main_index, "step": sample["step"],
            "simulation_time_s": sample["simulation_time_s"], "active_joint": "f1j2", "active_joint_angle_rad": sample["joint_positions_rad"]["f1j2"], "joint_positions_rad": sample["joint_positions_rad"],
            "world_from_handbase_row_major": sample["world_from_handbase_row_major"],
            "world_from_object_row_major": _transform(sample["object_poses"][0]["position_m"],
                sample["object_poses"][0]["orientation_wxyz"]).ravel().tolist(),
            "table_pose_source": "STATIC_SCENE_CONTRACT_NOT_STEP_TELEMETRY", "world_from_table_row_major": [1, 0, 0, 0.55, 0, 1, 0, 0, 0, 0, 1, 0.16, 0, 0, 0, 1], "table_center_world_m": [0.55, 0.0, 0.16], "table_size_m": [0.8, 0.9, 0.08]},
        **_query_fields(task, physx, non_task),
        "collision_enabled": True, "collision_enabled_scope": "ALL_66_HAND_AND_64_OBJECT_COLLIDERS", "collision_filter_allows_pair": True,
        "usd_authored_pair_filter_found": usd_filter_found, "collision_filter_runtime_validation": "USD_RULES_INFERRED_POST_RESET_ACTIVE_STAGE_EXTERNAL_PAIR_DEFAULT_ALLOW",
        "collision_filter_scope": "NO_FILTERED_PAIRS_OR_COLLISION_GROUPS; BACKEND_PAIR_FILTER_GETTER_UNAVAILABLE", "collision_filter_details": {"filtered_pairs": usd["filtered_pairs"], "collision_groups": usd["collision_groups"]},
        "hand_contact_offset": hand_contact, "object_contact_offset": object_contact,
        "hand_rest_offset": hand_rest, "object_rest_offset": object_rest,
        "combined_contact_offset_m": hand_contact + object_contact,
        "physx_distance_within_combined_contact_offset": in_contact_range,
        "offset_metadata": {"hand": {**offset, "value": hand_contact}, "object": {**offset, "value": object_contact},
            "runtime_query_api": "omni.physics.tensors.RigidBodyView.get_contact_offsets/get_rest_offsets", "runtime_query_backend": usd["runtime_backend"],
            "runtime_query_point": "AFTER_WORLD_RESET_BEFORE_ANY_ADDITIONAL_WORLD_STEP",
            "runtime_actor_results": runtime, "runtime_values_identical_across_130_shapes": True,
            "collider_path_mapping_basis": "ALL_RUNTIME_VALUES_IDENTICAL_MAPPING_ORDER_IRRELEVANT"},
        "collision_approximation": "convexHull", "hand_collider_piece_count": 66,
        "object_collider_piece_count": 64, "physx_witness_hand_link": witness_link,
        "physx_witness_hand_link_local_m": witness_local.tolist(),
        "physx_witness_distance_to_raw_task_m": task_distance,
        "physx_witness_distance_to_raw_non_task_m": non_task_distance,
        "physx_witness_physical_class": "TASK" if task_distance <= non_task_distance else "NON_TASK",
        "adjacent_existing_state_results": rows, "classification": classification,
        "classification_basis": "STEP_502_COMPOSED_COLLIDER_DISTANCE_IS_WITHIN_OBSERVED_COMBINED_CONTACT_OFFSET_AND_USD_RULES_ALLOW_PAIR_BUT_SAVED_RUNTIME_HAND_OBJECT_RECORD_COUNT_IS_ZERO",
        "classification_precedence": "MAIN_SELECTED_STATE_MATCHES_PHYSX_REPORTING_OR_FILTERING_BRANCH; STEP_503_RAW_NON_TASK_INTERSECTION_IS_SECONDARY_SAFETY_EVIDENCE",
        "secondary_physx_observation": "STEP_503_RAW_NON_TASK_AND_COMPOSED_COLLIDERS_INTERSECT_WHILE_RAW_TASK_REMAINS_FREE",
        "next_action_only": next_action,
        "preexisting_terminal_asset_audit": {"task_raw_to_compound_p95_m": 0.0015908483608893618,
            "absolute_limit_m": 0.002, "baseline_degradation_limit_m": 0.00025,
            "use": "BACKGROUND_REGION_P95_NOT_A_STATE_SPECIFIC_GATE"},
        "assets": {"diagnostic_script": _asset(Path(__file__)), "trace": _asset(TRACE), "evaluation": _asset(EVALUATION), "config": _asset(CONFIG),
            "task_surface_manifest": _asset(TASK_MANIFEST), "runtime_binding": _asset(RUNTIME_BINDING), "collision_manifest": _asset(collision_manifest), "hand_usd": _asset(HAND_USD),
            "object_manifest": _asset(OBJECT_MANIFEST), "object_usd": _asset(object_usd), "object_raw": _asset(ROOT / object_manifest["source_stl"]),
            "raw_first_finger": {link: {"path": item["path"], "sha256": item["sha256"]}
                                 for link, item in raw.items()}},
        "evidence_boundary": ["OFFLINE_EXACT_RAW_MESH_AND_COMPOSED_USD_COLLIDER_QUERY",
            "HEADLESS_WORLD_RESET_FOR_BACKEND_OFFSET_READBACK_NO_ADDITIONAL_PHYSICS_STEP",
            "NO_CONTROL_COMMANDS_NO_GRASP_OR_LIFT_CLAIM",
            "COMPOSED_CONVEX_MESHES_DO_NOT_EXPOSE_PHYSX_INTERNAL_COOKED_SHAPES",
            "0.75MM_IS_DIAGNOSTIC_PROXIMITY_ONLY_NOT_A_NON_TASK_HARD_GATE"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "diagnosis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    colors = (("TASK", "#f59e0b", task), ("PhysX", "#0891b2", physx), ("non-TASK", "#db2777", non_task))
    pose = _transform(sample["object_poses"][0]["position_m"], sample["object_poses"][0]["orientation_wxyz"])
    vertices = object_mesh.vertices_m @ pose[:3, :3].T + pose[:3, 3]
    vertices = vertices[np.linspace(0, len(vertices)-1, 3000, dtype=int)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (a, b) in zip(axes, ((0, 1), (0, 2), (1, 2))):
        axis.scatter(vertices[:, a], vertices[:, b], s=1, c="0.75", alpha=.35)
        for label, color, row in colors:
            p, q = np.asarray(row["left_witness_world"]), np.asarray(row["right_witness_world"])
            axis.plot((p[a], q[a]), (p[b], q[b]), "-o", color=color, ms=4,
                      label=f"{label}: {row['distance_m']*1e3:.4f} mm")
        axis.set_aspect("equal"); axis.set_xlabel("xyz"[a] + " / m"); axis.set_ylabel("xyz"[b] + " / m"); axis.legend(fontsize=7); axis.grid(alpha=.2)
    fig.suptitle("q09_a13 step 502 — same pose: raw TASK / composed PhysX / raw non-TASK")
    fig.savefig(OUTPUT / "witness_overlay.png", dpi=180); plt.close(fig)
if __name__ == "__main__": main()
