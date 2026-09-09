"""Inspect original Body/key/Socket surfaces at two completed release poses.

The signed-distance samples are not a continuous penetration bound. Original
surfaces must not be substituted for a subsequently deformed compliant model.
"""
import argparse
import json
from pathlib import Path

import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from evaluate_source_key_contacts import mesh_model


def evaluate(directory):
    repo = Path(__file__).resolve().parents[3]
    output = directory / "source_support_endpoints_posthoc_v1.json"
    if output.exists():
        raise FileExistsError(output)
    support = json.loads((directory / "support_posthoc_v3.json").read_text())
    steps = {support["before_unload"]["step"], support["final"]["step"]}
    body_file = repo / "artifacts/carts_grasp/CARTS_GRASP_V1/object_models/te_j35/plug_body_visual_mesh.npz"
    socket_file = repo / "artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/D38999_20FJ35SN_VISUAL.obj"
    data = np.load(body_file)
    vertices, faces = data["vertices_m"], data["faces"]
    body_object = fcl.CollisionObject(mesh_model(vertices, faces))
    socket = trimesh.load(socket_file, process=True)
    if not socket.is_watertight or not socket.is_winding_consistent:
        raise ValueError("source Socket must be a closed consistently wound solid")
    socket_object = fcl.CollisionObject(mesh_model(socket.vertices*.001, socket.faces))
    slab = (vertices[:, 2] >= -.007646) & (vertices[:, 2] <= -.000761)
    protruding = slab & (np.linalg.norm(vertices[:, :2], axis=1) > .0183)
    key_faces = faces[np.any(protruding[faces], axis=1) & np.all(slab[faces], axis=1)]
    if len(key_faces) != 106:
        raise ValueError("original key face selection changed")
    keys = np.array([10, 90, 157, 254, 308])
    centers = vertices[key_faces].mean(1)
    angles = np.rad2deg(np.arctan2(centers[:, 1], centers[:, 0]))
    groups = np.argmin(np.abs((angles[:, None]-keys+180)%360-180), axis=1)
    key_objects = []
    for k in range(5):
        indices, inverse = np.unique(key_faces[groups == k], return_inverse=True)
        key_objects.append(fcl.CollisionObject(mesh_model(vertices[indices], inverse.reshape(-1, 3))))
    origin = np.array(json.loads((directory / "assembly_scene.json").read_text())["socket_initial_position_world_m"])
    states = []
    with (directory / "truth_samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["step"] not in steps:
                continue
            rotation = Rotation.from_quat(np.roll(row["object_part_orientations_wxyz"][0], -1)).as_matrix()
            position = np.asarray(row["object_part_positions_m"][0])-origin
            transform = fcl.Transform(rotation, position)
            body_object.setTransform(transform)
            result = fcl.CollisionResult()
            count = fcl.collide(body_object, socket_object,
                                fcl.CollisionRequest(num_max_contacts=4096, enable_contact=True), result)
            if count >= 4096:
                raise ValueError("intersection contact enumeration reached its capacity")
            indices = sorted({int(c.b1) for c in result.contacts})
            hits = {}
            for angle, obj in zip(keys, key_objects):
                obj.setTransform(transform)
                hits[str(angle)] = bool(fcl.collide(obj, socket_object,
                    fcl.CollisionRequest(num_max_contacts=1), fcl.CollisionResult()))
            record = {"step": row["step"], "phase": row["phase"],
                      "source_body_surfaces_intersect": bool(count),
                      "source_key_intersection_by_angle": hits,
                      "intersecting_body_triangle_indices": indices}
            if indices:
                tri = vertices[faces[indices]]
                query = np.unique(np.vstack((tri.reshape(-1, 3), tri.mean(1),
                    (tri[:, 0]+tri[:, 1])/2, (tri[:, 1]+tri[:, 2])/2,
                    (tri[:, 2]+tri[:, 0])/2)), axis=0)
                distance = trimesh.proximity.signed_distance(
                    socket, (query@rotation.T+position)*1000)/1000
                record.update(sample_count=len(query),
                    inside_sample_count=int(np.sum(distance > 0)),
                    maximum_sampled_inside_distance_m=float(max(0., distance.max())))
            states.append(record)
    if len(states) != len(steps):
        raise ValueError("required completed-state poses are missing")
    result = {"scope": "POSTRUN_ORIGINAL_SURFACE_INTERSECTION_AND_MM_CONDITIONED_SAMPLES",
              "body_source": str(body_file), "socket_source": str(socket_file),
              "nominal_body_sdf_cell_spacing_m": float(np.ptp(vertices, axis=0).max()/1024),
              "continuous_penetration_bound": False, "native_physx_contact_measurement": False,
              "online_control_used": False, "states": states}
    output.write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = evaluate(parser.parse_args().directory.resolve())
    print(json.dumps({**result, "states": [
        {k: v for k, v in s.items() if k != "intersecting_body_triangle_indices"}
        for s in result["states"]]}, indent=2))
