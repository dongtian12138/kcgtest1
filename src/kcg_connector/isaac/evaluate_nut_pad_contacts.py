#!/usr/bin/env python3
"""Project completed nut-grip contacts onto the original fingertip/pad surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_metadata import read_metadata_field
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def evaluate(directory: Path, phase_kind="grip_hold", stage_name=None):
    repository = Path(__file__).resolve().parents[3]
    dt = float(read_metadata_field(directory, "physics_dt_s"))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("recorded physics time step must be positive")
    if stage_name is not None and (Path(stage_name).name != stage_name or stage_name in (".", "..")):
        raise ValueError("stage name must identify one local stage directory")
    if phase_kind == "rotation":
        stage = json.loads((directory / "socket_transport" / (stage_name or "nut_rotation") / "nut_rotation_controller_result.json").read_text())
        first, last = int(stage["first_step"]), int(stage["last_step"]) - 1
        if last < first:
            raise ValueError("the rotation stage did not execute any physical step")
        prefix = "nut_rotation_pad_surface"
    else:
        stage = json.loads((directory / "socket_transport" / (stage_name or "nut_regrasp") / "nut_regrasp_controller_result.json").read_text())
        last = int(stage["last_step"]) - 1
        first = last - round(2.0 / dt) + 1
        prefix = "nut_pad_surface"
    if stage_name is not None:
        prefix = stage_name + "_pad_surface"
    links = ("f1Link3", "f2Link2", "f3Link3")
    points, impulses, nut_points = ({link: [] for link in links} for _ in range(3))
    contact_steps = {link: [] for link in links}
    steps = []
    native_pose_samples, fk_pose_samples = 0, 0
    with (directory / "truth_samples.jsonl").open() as stream:
        for line in stream:
            sample = json.loads(line)
            step = int(sample["step"])
            if step < first:
                continue
            if step > last:
                break
            valid_phase = (sample["phase"].startswith("key_probe_nut_rotation_")
                if phase_kind == "rotation" else sample["phase"] == "key_probe_nut_grip_hold")
            if not valid_phase:
                raise ValueError(f"unexpected phase in {phase_kind}: {sample['phase']}")
            steps.append(step)
            native = sample.get("native_robot_link_pose_audit")
            if native is not None:
                if (native.get("source") != "EXISTING_NATIVE_PHYSICS_TENSOR_RIGID_LINK_TRANSFORMS"
                        or native.get("used_for_online_control") is not False
                        or any(link not in native.get("poses", {}) for link in links)):
                    raise ValueError("incomplete or unsupported native terminal-link pose record")
                native_pose_samples += 1
            else:
                fk_pose_samples += 1
            nut_rotation = Rotation.from_quat(np.roll(sample["object_part_orientations_wxyz"][1], -1)).as_matrix()
            nut_position = np.asarray(sample["object_part_positions_m"][1])
            for header in sample["contacts"]["tensor_headers"]:
                if not any(path.endswith("/CouplingNut") for path in header["paths"]):
                    continue
                link = next((name for name in links if any(path.endswith("/"+name) for path in header["paths"])), None)
                if link is None:
                    continue
                if native is None:
                    orientation = sample["terminal_link_orientations_wxyz"][link]
                    position = np.asarray(sample["terminal_link_positions_m"][link])
                else:
                    pose = native["poses"][link]
                    orientation = pose["orientation_world_wxyz"]
                    position = np.asarray(pose["position_world_m"])
                rotation = Rotation.from_quat(np.roll(orientation, -1)).as_matrix()
                for contact in header["contacts"]:
                    if contact["normal_impulse_n_s"] <= 1e-9:
                        continue
                    world_point = np.asarray(contact["position_m"])
                    points[link].append((world_point-position) @ rotation)
                    impulses[link].append(contact["normal_impulse_n_s"])
                    nut_points[link].append((world_point-nut_position) @ nut_rotation)
                    contact_steps[link].append(step)
    if steps != list(range(first, last+1)):
        raise ValueError(f"the trace does not cover the complete {last-first+1}-sample {phase_kind} window")
    base = repository / "artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TERMINAL_PAD_EXACT_SOURCE_V2"
    result = {"scope": "POSTRUN_SOURCE_SURFACE_PROJECTION_NOT_ONLINE_CONTACT_LABEL",
        "source_run": str(directory), "phase_kind": phase_kind, "window_steps": [first, last],
        "physical_sample_count": len(steps),
        "hand_pose_provenance": {
            "native_rigid_body_pose_sample_count": native_pose_samples,
            "encoder_fk_with_nominal_mimic_sample_count": fk_pose_samples,
            "all_hand_poses_independently_read_from_physics": native_pose_samples == len(steps),
            "legacy_fk_projection_limitation": "Archived FK-derived terminal poses assume the frozen robot and nominal mimic relations; they are not independent rigid-link pose measurements.",
        },
        "physics_dt_s": dt, "physical_window_duration_s": len(steps)*dt,
        "pad_source_manifest": str(base / "TERMINAL_PAD_SOURCE_MANIFEST.json"),
        "projection_algorithm": "Trimesh nearest-point queries in millimetre coordinates, preserving source face order; distances converted to metres",
        "supersedes_v1_distance_query": "Metre-coordinate queries can misclassify small CAD triangles due to fixed numeric tolerances; original v1 outputs are retained",
        "projection_limitation": "Convex-decomposition contact points are projected onto source surfaces. These are measured distances and nearest source-face memberships, not exact source-face IDs returned by PhysX.",
        "fingers": []}
    arrays = {}
    for link in links:
        raw = trimesh.load(repository / f"src/iiwa_description/meshes/hand/{link}.STL", force="mesh", process=False)
        pad = np.load(base / f"{link}_PAD_BODY_raw_source_local_m.npz")
        raw.vertices *= 1000.
        pad_mesh = trimesh.Trimesh(pad["points_local_m"]*1000., pad["faces"], process=False)
        points_local, weights, points_nut = np.asarray(points[link]), np.asarray(impulses[link]), np.asarray(nut_points[link])
        if not len(points_local):
            raise ValueError(f"no positive nut contact for {link}")
        _, distance, face = trimesh.proximity.closest_point(raw, points_local*1000.)
        _, pad_distance, _ = trimesh.proximity.closest_point(pad_mesh, points_local*1000.)
        distance /= 1000.
        pad_distance /= 1000.
        is_pad = np.isin(face, pad["source_face_indices"])
        result["fingers"].append({"link": link, "count": len(points_local),
            "source_face_count": len(raw.faces),
            "minimum_distance_to_original_pad_m": float(pad_distance.min()),
            "maximum_distance_to_original_pad_m": float(pad_distance.max()),
            "median_distance_to_original_pad_m": float(np.median(pad_distance)),
            "maximum_distance_to_full_original_fingertip_m": float(distance.max()),
            "nearest_full_source_face_is_pad_fraction": float(is_pad.mean()),
            "normal_impulse_weighted_pad_face_fraction": float(weights[is_pad].sum()/weights.sum()),
            "nut_contact_z_range_m": [float(points_nut[:, 2].min()), float(points_nut[:, 2].max())],
            "nut_contact_radius_range_m": [float(np.linalg.norm(points_nut[:, :2], axis=1).min()),
                                           float(np.linalg.norm(points_nut[:, :2], axis=1).max())]})
        arrays.update({link+"_contact_points_link_m": points_local, link+"_nut_points_m": points_nut,
                       link+"_source_face_indices": face, link+"_nearest_is_pad": is_pad,
                       link+"_contact_steps": np.asarray(contact_steps[link]),
                       link+"_normal_impulses_n_s": weights})
    summary_path, samples_path = directory / (prefix+"_posthoc_v2.json"), directory / (prefix+"_samples_v2.npz")
    if summary_path.exists() or samples_path.exists():
        raise FileExistsError("refusing to overwrite an existing contact-surface audit")
    summary_path.write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(samples_path, **arrays)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--phase-kind", choices=("grip_hold", "rotation"), default="grip_hold")
    parser.add_argument("--stage-name", help="Alternate repeated-stage directory")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.directory.resolve(), args.phase_kind, args.stage_name), indent=2))
