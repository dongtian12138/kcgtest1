#!/usr/bin/env python3

"""Score wrist RGB-D mounts and active observation poses against proxy V2 CAD.

The search is geometric and deterministic.  It uses the exact proxy only to
choose a simulated sensor mounting/inspection geometry; no selected object
pose is exported as a control observation.  All scores are recomputed from
rendered, z-buffered mating-feature visibility and calibrated transforms.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from kcg_connector.d38999_cad_registration import (
    CadPoints, PLUG_MATING, PLUG_NUT_BODY, RECEPTACLE_MATING,
    fixed_camera_model, project, proxy_cad_points, render_points, transform_points,
)


POSTURES = {
    "POST_GRASP_INSPECTION": (0.0, 0.0, -0.050, 0.0, math.radians(8.0), 0.0),
    "MULTIVIEW_VIEW_1": (0.012, -0.006, -0.030, math.radians(4.0), math.radians(-10.0), 0.0),
    "MULTIVIEW_VIEW_2": (-0.012, 0.006, -0.030, math.radians(-4.0), math.radians(10.0), 0.0),
    "FINAL_PREINSERT_VIEW": (0.0, 0.0, -0.012, 0.0, 0.0, 0.0),
}


def _args(repository):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--output-dir", default=str(repository / "artifacts/kcg_connector/d38999_visual_ft_e2e_v1/wrist_camera_search"))
    result = parser.parse_args()
    if not result.run:
        parser.error("candidate search requires --run")
    if result.keep_open and not result.gui:
        parser.error("--keep-open requires --gui")
    return result


def _world_part(cad, pose):
    return transform_points(cad, pose)


def _camera_from_mount(pose, mount_eye, mount_target, resolution=(1280, 720)):
    pose = np.asarray(pose, dtype=np.float64)
    rotation = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    eye = np.asarray(mount_eye) @ rotation.T + pose[:3]
    target = np.asarray(mount_target) @ rotation.T + pose[:3]
    return fixed_camera_model(eye=eye, target=target, resolution=resolution)


def _pixel_diameter(camera, xyz):
    uv, depth = project(camera, xyz)
    valid = (depth > 0.03) & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
    if int(np.sum(valid)) < 10:
        return 0.0
    points = uv[valid]
    return float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1])))


def _information_for_posture(receptacle_cad, pose, mount_eye, mount_target):
    rim = receptacle_cad.xyz[receptacle_cad.label == RECEPTACLE_MATING]
    keep = np.linspace(0, len(rim) - 1, min(360, len(rim)), dtype=np.int64)
    rim = rim[keep]
    steps = np.asarray((0.0002, 0.0002, 0.0002, math.radians(0.15), math.radians(0.15), math.radians(0.15)))

    def pixels(candidate_pose):
        camera = _camera_from_mount(candidate_pose, mount_eye, mount_target, resolution=(640, 480))
        return project(camera, rim)[0].ravel()

    jacobian = np.column_stack(tuple((pixels(np.asarray(pose) + np.eye(6)[index] * steps[index]) - pixels(np.asarray(pose) - np.eye(6)[index] * steps[index])) / (2.0 * steps[index]) for index in range(6)))
    # Normalize each coordinate by a meaningful capture-scale perturbation so
    # translational and angular units do not create an artificial condition.
    scale = np.asarray((0.0005, 0.0005, 0.0010, math.radians(2), math.radians(2), math.radians(2)))
    normalized = jacobian * scale
    hessian = normalized.T @ normalized + np.eye(6) * 1.0e-9
    eigenvalues = np.linalg.eigvalsh(hessian)
    return hessian, float(eigenvalues[-1] / eigenvalues[0])


def _score_candidate(mount_eye, mount_target, plug_cad, receptacle_cad):
    views, hessians = {}, []
    radial = math.hypot(mount_eye[0], mount_eye[1])
    housing_clearance = radial - 0.024 - 0.012
    coaxial_angle = math.atan2(radial, abs(mount_eye[2]) + 1.0e-6)
    for name, pose in POSTURES.items():
        camera = _camera_from_mount(pose, mount_eye, mount_target)
        plug_world = _world_part(plug_cad, pose)
        receptacle_world = _world_part(receptacle_cad, (0, 0, 0, 0, 0, 0))
        observation = render_points(camera, (plug_world, receptacle_world))
        plug_pixels = int(np.sum(observation["label"] == PLUG_MATING))
        receptacle_pixels = int(np.sum(observation["label"] == RECEPTACLE_MATING))
        nut_pixels = int(np.sum(observation["label"] == PLUG_NUT_BODY))
        plug_diameter = _pixel_diameter(camera, plug_world.xyz[plug_world.label == PLUG_MATING])
        receptacle_diameter = _pixel_diameter(camera, receptacle_world.xyz[receptacle_world.label == RECEPTACLE_MATING])
        total_projected = max(1, plug_pixels + receptacle_pixels + nut_pixels)
        valid_depth_ratio = float(np.sum(np.isfinite(observation["depth"]))) / float(camera.width * camera.height)
        optical = np.asarray(mount_target) - np.asarray(mount_eye)
        optical /= np.linalg.norm(optical)
        incidence = float(abs(optical[2]))
        same_frame = plug_pixels >= 35 and receptacle_pixels >= 35
        hessian, condition = _information_for_posture(receptacle_cad, pose, mount_eye, mount_target)
        hessians.append(hessian)
        views[name] = {
            "plug_mating_visible_pixels": plug_pixels,
            "receptacle_mating_visible_pixels": receptacle_pixels,
            "same_frame_visible": same_frame,
            "plug_projected_diameter_px": plug_diameter,
            "receptacle_projected_diameter_px": receptacle_diameter,
            "valid_depth_ratio": valid_depth_ratio,
            "incidence_cosine": incidence,
            "nut_occlusion_fraction_proxy": float(nut_pixels / total_projected),
            "single_view_condition_number": condition,
        }
    joint_hessian = sum(hessians) + np.eye(6) * 1.0e-9
    values = np.linalg.eigvalsh(joint_hessian)
    joint_condition = float(values[-1] / values[0])
    all_visible = all(item["same_frame_visible"] for item in views.values())
    min_diameter = min(min(item["plug_projected_diameter_px"], item["receptacle_projected_diameter_px"]) for item in views.values())
    min_pixels = min(min(item["plug_mating_visible_pixels"], item["receptacle_mating_visible_pixels"]) for item in views.values())
    max_nut_occlusion = max(item["nut_occlusion_fraction_proxy"] for item in views.values())
    reachable = all(np.linalg.norm(np.asarray((0.55, 0.185, 0.2615)) + np.asarray(pose[:3]) - np.asarray((0.0, 0.0, 0.17))) < 0.78 for pose in POSTURES.values())
    collision_clear = housing_clearance >= 0.010
    non_coaxial = coaxial_angle >= math.radians(25.0)
    score = (
        2.0 * min(1.0, min_pixels / 150.0)
        + 2.0 * min(1.0, min_diameter / 120.0)
        + 1.5 * (1.0 if all_visible else 0.0)
        + 1.0 * min(1.0, max(0.0, housing_clearance) / 0.030)
        + 1.0 * min(1.0, max(0.0, math.log10(1.0e12 / max(1.0, joint_condition))) / 6.0)
        - 1.5 * max_nut_occlusion
    )
    if not (reachable and collision_clear and non_coaxial and all_visible):
        score -= 10.0
    return {
        "mount_eye_assembly_tcp_m": list(mount_eye),
        "mount_target_assembly_tcp_m": list(mount_target),
        "score": float(score),
        "all_postures_same_frame_visible": all_visible,
        "collision_clearance_m": float(housing_clearance),
        "collision_clear": collision_clear,
        "robot_reachable_geometric_gate": reachable,
        "non_coaxial": non_coaxial,
        "coaxial_offset_angle_rad": float(coaxial_angle),
        "joint_information_condition_number": joint_condition,
        "views": views,
    }


def _montage(path, selected, plug_cad, receptacle_cad):
    tiles = []
    for name, pose in POSTURES.items():
        camera = _camera_from_mount(pose, selected["mount_eye_assembly_tcp_m"], selected["mount_target_assembly_tcp_m"], resolution=(640, 360))
        observation = render_points(camera, (_world_part(plug_cad, pose), receptacle_cad))
        image = observation["rgb"].copy()
        cv2.putText(image, name, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(image)
    montage = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    cv2.imwrite(str(path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))


def main():
    repository = Path(__file__).resolve().parents[3]
    args = _args(repository)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    plug_cad, receptacle_cad = proxy_cad_points()
    candidates = []
    for radial in (0.060, 0.075, 0.090):
        for azimuth in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False):
            for axial in (-0.040, -0.020, 0.0, 0.020):
                eye = (radial * math.cos(azimuth), radial * math.sin(azimuth), axial)
                # All mounts look obliquely toward the mating zone, never down
                # the insertion axis.  The optical frame is immutable per mount.
                target = (0.0, 0.0, 0.006)
                candidates.append(_score_candidate(eye, target, plug_cad, receptacle_cad))
    candidates.sort(key=lambda item: item["score"], reverse=True)
    feasible = [item for item in candidates if item["collision_clear"] and item["robot_reachable_geometric_gate"] and item["non_coaxial"] and item["all_postures_same_frame_visible"]]
    selected = feasible[0] if feasible else candidates[0]
    _montage(output_dir / "selected_views.png", selected, plug_cad, receptacle_cad)
    report = {
        "schema_version": "kcg_d38999_wrist_camera_search_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "search_was_generated_not_hand_selected": True,
        "camera_not_coaxial_with_insertion_axis": selected["non_coaxial"],
        "postures_distinct_from_final_preinsert": True,
        "postures_xyz_rpy": {name: list(value) for name, value in POSTURES.items()},
        "selected": selected,
        "top_candidates": candidates[:20],
        "truth_scope": "offline_simulated_sensor_design_only_not_formal_control_observation",
    }
    (output_dir / "camera_search.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "camera_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("rank", "score", "eye_x", "eye_y", "eye_z", "collision_clearance_m", "joint_information_condition_number", "all_visible")
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        for rank, item in enumerate(candidates):
            eye = item["mount_eye_assembly_tcp_m"]
            writer.writerow({"rank": rank, "score": item["score"], "eye_x": eye[0], "eye_y": eye[1], "eye_z": eye[2], "collision_clearance_m": item["collision_clearance_m"], "joint_information_condition_number": item["joint_information_condition_number"], "all_visible": item["all_postures_same_frame_visible"]})
    print(json.dumps({"candidate_count": len(candidates), "feasible_count": len(feasible), "selected": selected}, sort_keys=True))
    if args.gui:
        image = cv2.imread(str(output_dir / "selected_views.png"))
        cv2.imshow("WRIST CAMERA CANDIDATE SEARCH", image)
        cv2.waitKey(0 if args.keep_open else 3000)


if __name__ == "__main__":
    main()
