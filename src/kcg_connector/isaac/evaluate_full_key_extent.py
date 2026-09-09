#!/usr/bin/env python3
"""Evaluate complete original key axial extents after a stopped assembly run."""

import argparse
import json
from pathlib import Path

import numpy as np
from trace_metadata import iter_control_samples
from scipy.spatial.transform import Rotation


def evaluate(directory):
    repository = Path(__file__).resolve().parents[3]
    source = repository / "artifacts/carts_grasp/CARTS_GRASP_V1/object_models/te_j35/plug_body_visual_mesh.npz"
    data = np.load(source)
    vertices, faces = data["vertices_m"], data["faces"]
    slab = (vertices[:, 2] >= -.007646) & (vertices[:, 2] <= -.000761)
    protruding = slab & (np.linalg.norm(vertices[:, :2], axis=1) > .0183)
    key_faces = faces[np.any(protruding[faces], axis=1) & np.all(slab[faces], axis=1)]
    if len(key_faces) != 106:
        raise ValueError("original key face selection differs from the source geometry audit")
    key_angles = np.array([10., 90., 157., 254., 308.])
    centers = vertices[key_faces].mean(1)
    angles = np.rad2deg(np.arctan2(centers[:, 1], centers[:, 0]))
    groups = np.argmin(np.abs((angles[:, None]-key_angles+180)%360-180), axis=1)
    points = [vertices[np.unique(key_faces[groups == k])] for k in range(5)]
    entry = json.loads((directory / "socket_transport/key_entry/key_entry_controller_result.json").read_text())
    scene = json.loads((directory / "assembly_scene.json").read_text())
    socket_position = np.asarray(scene["socket_initial_position_world_m"])
    first = int(entry["contact_first_step"])-1
    rotation_path = directory / "socket_transport/nut_rotation/nut_rotation_controller_result.json"
    rotation = json.loads(rotation_path.read_text()) if rotation_path.exists() else {}
    first_turn = next((int(s["step"]) for s in (iter_control_samples(rotation) if rotation else [])
                       if abs(s["commanded_rotation_deg"]) > 1e-6), None)
    rows, first_full, before_turn, consecutive_start = [], None, None, None
    with (directory / "truth_samples.jsonl").open() as stream:
        for line in stream:
            sample = json.loads(line)
            if sample["step"] < first:
                continue
            position = np.asarray(sample["object_part_positions_m"][0])
            R = Rotation.from_quat(np.roll(sample["object_part_orientations_wxyz"][0], -1)).as_matrix()
            depths = [socket_position[2]-(p @ R.T+position)[:, 2] for p in points]
            minima = np.asarray([d.min() for d in depths])
            maxima = np.asarray([d.max() for d in depths])
            full = bool(np.all(minima > 0.))
            row = {"step": sample["step"], "phase": sample["phase"],
                   "time_s": sample["simulation_time_s"],
                   "body_depth_m": float(socket_position[2]-position[2]),
                   "body_lateral_offset_m": float(np.linalg.norm(position[:2]-socket_position[:2])),
                   "body_axis_tilt_deg": float(np.rad2deg(np.arccos(np.clip(-R[2, 2], -1, 1)))),
                   "rearmost_key_vertex_depths_m": minima.tolist(),
                   "frontmost_key_vertex_depths_m": maxima.tolist(),
                   "all_source_key_vertices_behind_mouth": full}
            if first_full is None and full:
                first_full = row.copy()
            if first_turn is not None and sample["step"] == first_turn-1:
                before_turn = row.copy()
            if full and consecutive_start is None:
                consecutive_start = sample["simulation_time_s"]
            elif not full:
                consecutive_start = None
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("no complete entry/turn interval")
    dt = float(np.median(np.diff([r["time_s"] for r in rows])))
    result = {"scope": "POSTRUN_AXIAL_EXTENT_OF_ALL_ORIGINAL_KEY_TRIANGLES",
              "source_mesh": str(source), "source_key_triangle_count": len(key_faces),
              "key_angles_deg": key_angles.tolist(), "source_vertex_count_by_key": [len(p) for p in points],
              "source_key_rear_z_m": float(min(p[:, 2].min() for p in points)),
              "socket_axis": "WORLD_PLUS_Z_FIXED_SOURCE_FIXTURE",
              "first_all_keys_behind_mouth": first_full,
              "before_first_nonzero_formal_turn_command": before_turn,
              "final": rows[-1],
              "final_contiguous_full_axial_extent_duration_s": (0. if consecutive_start is None
                  else rows[-1]["time_s"]-consecutive_start+dt),
              "source_key_slot_containment_requires_separate_contact_geometry_evaluation": True,
              "full_coupling_or_electrical_continuity_claimed": False,
              "online_control_used": False}
    stem = directory / "full_key_axial_extent_posthoc_v1"
    if stem.with_suffix(".json").exists() or stem.with_suffix(".npz").exists():
        raise FileExistsError("refusing to overwrite completed key extent evidence")
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(stem.with_suffix(".npz"),
        steps=[r["step"] for r in rows], time_s=[r["time_s"] for r in rows],
        rearmost_key_depths_m=[r["rearmost_key_vertex_depths_m"] for r in rows],
        frontmost_key_depths_m=[r["frontmost_key_vertex_depths_m"] for r in rows])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(evaluate(parser.parse_args().directory.resolve()), indent=2))
