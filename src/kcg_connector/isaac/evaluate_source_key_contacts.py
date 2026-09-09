#!/usr/bin/env python3
"""Check original key triangles against the original socket after motion stops."""

import argparse
import json
from pathlib import Path

import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def mesh_model(vertices, faces):
    model = fcl.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(np.ascontiguousarray(vertices), np.ascontiguousarray(faces, dtype=np.int32))
    model.endModel()
    return model


def evaluate(directory):
    repository = Path(__file__).resolve().parents[3]
    output = directory / "source_key_contacts_full_extent_posthoc_v1.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite source-contact evidence")
    extent = json.loads((directory / "full_key_axial_extent_posthoc_v1.json").read_text())
    if extent["first_all_keys_behind_mouth"] is None:
        raise ValueError("the episode never reached complete axial key extent")
    first = extent["first_all_keys_behind_mouth"]["step"]
    last = extent["final"]["step"]
    source = Path(extent["source_mesh"])
    data = np.load(source)
    vertices, faces = data["vertices_m"], data["faces"]
    slab = (vertices[:, 2] >= -.007646) & (vertices[:, 2] <= -.000761)
    protruding = slab & (np.linalg.norm(vertices[:, :2], axis=1) > .0183)
    key_faces = faces[np.any(protruding[faces], axis=1) & np.all(slab[faces], axis=1)]
    if len(key_faces) != 106:
        raise ValueError("original key face selection changed")
    key_angles = np.asarray(extent["key_angles_deg"])
    centers = vertices[key_faces].mean(1)
    angles = np.rad2deg(np.arctan2(centers[:, 1], centers[:, 0]))
    groups = np.argmin(np.abs((angles[:, None]-key_angles+180)%360-180), axis=1)
    key_objects = []
    for k in range(5):
        indices, inverse = np.unique(key_faces[groups == k], return_inverse=True)
        key_objects.append(fcl.CollisionObject(mesh_model(vertices[indices], inverse.reshape(-1, 3))))
    socket_path = repository / "artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/D38999_20FJ35SN_VISUAL.obj"
    socket = trimesh.load(socket_path, force="mesh", process=False)
    socket_object = fcl.CollisionObject(mesh_model(socket.vertices*.001, socket.faces))
    scene = json.loads((directory / "assembly_scene.json").read_text())
    socket_position = np.asarray(scene["socket_initial_position_world_m"])
    request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
    counts = np.zeros(5, dtype=int)
    first_intersection = [None]*5
    records, steps = [], []
    with (directory / "truth_samples.jsonl").open() as stream:
        for line in stream:
            sample = json.loads(line)
            step = int(sample["step"])
            if step < first:
                continue
            if step > last:
                break
            rotation = Rotation.from_quat(np.roll(sample["object_part_orientations_wxyz"][0], -1)).as_matrix()
            position = np.asarray(sample["object_part_positions_m"][0])-socket_position
            transform = fcl.Transform(rotation, position)
            flags = []
            for k, key in enumerate(key_objects):
                key.setTransform(transform)
                hit = bool(fcl.collide(key, socket_object, request, fcl.CollisionResult()))
                flags.append(hit)
                counts[k] += hit
                if hit and first_intersection[k] is None:
                    first_intersection[k] = step
            steps.append(step)
            records.append(flags)
    if steps != list(range(first, last+1)):
        raise ValueError("incomplete source-pose interval")
    result = {
        "scope": "POSTRUN_ORIGINAL_TRIANGLE_INTERSECTION_AT_EVERY_RECORDED_POSE_AFTER_FIRST_FULL_AXIAL_EXTENT",
        "body_mesh": str(source), "socket_mesh": str(socket_path),
        "source_key_triangle_count": len(key_faces), "key_angles_deg": key_angles.tolist(),
        "socket_axes": "FIXED_ORIGINAL_SOCKET_AXES_EQUAL_WORLD_AXES",
        "window_steps": [first, last], "sample_count": len(steps),
        "source_triangle_intersection_sample_counts_by_key": counts.tolist(),
        "first_intersection_steps_by_key": first_intersection,
        "first_full_extent_intersections_by_key": records[0],
        "final_intersections_by_key": records[-1],
        "all_sampled_original_key_surfaces_clear": bool(not counts.any()),
        "between_step_continuous_motion_checked": False,
        "intersection_is_not_a_penetration_depth_measurement": True,
        "contact_load_and_sdf_penetrations_require_separate_native_contact_evaluation": True,
        "online_control_used": False,
    }
    output.write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(output.with_suffix(".npz"), steps=steps, key_triangle_intersection=records)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(evaluate(parser.parse_args().directory.resolve()), indent=2))
