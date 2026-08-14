#!/usr/bin/env python3

"""Evaluate direct wrist-multiview ``T_receptacle_plug`` registration."""

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

from kcg_connector.d38999_cad_registration import proxy_cad_points, render_points, transform_points
from kcg_connector.d38999_inhand_multiview import (
    camera_from_plug_pose, compose_pose, register_inhand_relative_pose_multiview,
    relative_delta,
)


VIEW_NAMES = ("MULTIVIEW_VIEW_1", "MULTIVIEW_VIEW_2", "FINAL_PREINSERT_VIEW")


def _args(repository):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--accuracy-only", action="store_true", help="run pose/conditioning evaluation while corrected Bcapture is still pending")
    parser.add_argument("--seed", type=int, default=39001)
    parser.add_argument("--camera-search", default=str(repository / "artifacts/kcg_connector/d38999_visual_ft_e2e_v1/wrist_camera_search_20260813T0730Z/camera_search.json"))
    parser.add_argument("--bcapture", default=str(repository / "artifacts/kcg_connector/d38999_insert_proxy_v2/bcapture_corrected_20260813T0748Z/bcapture.json"))
    parser.add_argument("--output-dir", default=str(repository / "artifacts/kcg_connector/d38999_visual_ft_e2e_v1/wrist_multiview_evaluation"))
    result = parser.parse_args()
    if not result.run or result.episodes < 100:
        parser.error("multiview evaluation requires --run and >=100 episodes")
    if result.keep_open and not result.gui:
        parser.error("--keep-open requires --gui")
    return result


def _pose_error(estimate, truth):
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    translation = estimate[:3] - truth[:3]
    rotation = Rotation.from_euler("xyz", estimate[3:]).inv() * Rotation.from_euler("xyz", truth[3:])
    rv = rotation.as_rotvec()
    return {"lateral_error_m": float(np.linalg.norm(translation[:2])), "xyz_error_m": float(np.linalg.norm(translation)), "axis_error_rad": float(np.linalg.norm(rv[:2])), "yaw_error_mod_c2_rad": float(min(abs(rv[2]), abs(abs(rv[2]) - math.pi)))}


def _slice_safe_radius(records, erx, ery):
    subset = [item for item in records if item["erx_rad"] == erx and item["ery_rad"] == ery]
    levels = sorted({math.hypot(item["ex_m"], item["ey_m"]) for item in subset})
    safe = 0.0
    for level in levels:
        inside = [item for item in subset if math.hypot(item["ex_m"], item["ey_m"]) <= level + 1.0e-12]
        if inside and all(item["capture_success"] for item in inside):
            safe = level
        else:
            break
    return safe


def _capture_radius(bcapture, rx_abs, ry_abs):
    records = bcapture["results"]
    rx_values = sorted({item["erx_rad"] for item in records})
    ry_values = sorted({item["ery_rad"] for item in records})
    def enclosing(values, target):
        nonnegative = sorted({abs(value) for value in values})
        upper = next((value for value in nonnegative if value + 1e-12 >= target), None)
        return upper
    rx_upper, ry_upper = enclosing(rx_values, rx_abs), enclosing(ry_values, ry_abs)
    if rx_upper is None or ry_upper is None:
        return 0.0
    rx_candidates = (0.0,) if rx_upper == 0.0 else (-rx_upper, rx_upper)
    ry_candidates = (0.0,) if ry_upper == 0.0 else (-ry_upper, ry_upper)
    return min(_slice_safe_radius(records, float(rx), float(ry)) for rx in rx_candidates for ry in ry_candidates)


def _summary(values):
    value = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(value)), "median": float(np.median(value)), "p95": float(np.quantile(value, 0.95)), "maximum": float(np.max(value))}


def _overlay(observation, camera, receptacle_cad, path):
    predicted = render_points(camera, (receptacle_cad,))
    image = observation["rgb"].copy()
    observed_edge = cv2.Canny(observation["rgb"], 30, 90) > 0
    predicted_edge = cv2.Canny(predicted["rgb"], 30, 90) > 0
    image[observed_edge] = (30, 255, 80)
    image[predicted_edge] = (255, 60, 220)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main():
    repository = Path(__file__).resolve().parents[3]
    args = _args(repository)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    camera_search = json.loads(Path(args.camera_search).read_text(encoding="utf-8"))
    bcapture_path = Path(args.bcapture)
    if bcapture_path.is_file():
        bcapture = json.loads(bcapture_path.read_text(encoding="utf-8"))
    else:
        bcapture = None
    if not args.accuracy_only and (bcapture is None or bcapture.get("passed") is not True):
        raise RuntimeError("refusing multiview authorization without a passed corrected Bcapture run")
    mount_eye = camera_search["selected"]["mount_eye_assembly_tcp_m"]
    mount_target = camera_search["selected"]["mount_target_assembly_tcp_m"]
    postures = {key: np.asarray(value) for key, value in camera_search["postures_xyz_rpy"].items()}
    nominal = postures["FINAL_PREINSERT_VIEW"]
    deltas = [relative_delta(nominal, postures[name]) for name in VIEW_NAMES]
    plug_cad, receptacle_cad = proxy_cad_points()
    rng = np.random.default_rng(args.seed)
    records = []
    for index in range(args.episodes):
        truth = nominal + np.asarray((rng.uniform(-0.0012, 0.0012), rng.uniform(-0.0012, 0.0012), rng.uniform(-0.0008, 0.0008), rng.uniform(-math.radians(5), math.radians(5)), rng.uniform(-math.radians(5), math.radians(5)), rng.uniform(-math.radians(2), math.radians(2))))
        observations = []
        for view_index, delta in enumerate(deltas):
            view_pose = compose_pose(truth, delta)
            camera = camera_from_plug_pose(view_pose, mount_eye, mount_target, resolution=(640, 360))
            observations.append(render_points(camera, (transform_points(plug_cad, view_pose), receptacle_cad), depth_noise_std_m=0.00040, seed=args.seed + index * 7 + view_index))
        branches = [register_inhand_relative_pose_multiview(receptacle_cad, observations, deltas, nominal, mount_eye_plug=mount_eye, mount_target_plug=mount_target, yaw_hypothesis_rad=yaw, semantic_mask=False) for yaw in (0.0, math.pi)]
        valid = [item for item in branches if item.get("success")]
        selected = min(valid, key=lambda item: item["cost"]) if valid else None
        if selected is None:
            records.append({"episode": index, "success": False, "control_authorized": False, "reject_reason": "REGISTRATION_FAILED", "truth": truth.tolist(), "c2_branches": branches})
            continue
        estimate = np.asarray(selected["reference_relative_pose_xyz_rpy"])
        covariance = np.asarray(selected["conditional_covariance_xyz_rx_ry_5x5"])
        sigma_lateral = math.sqrt(max(0.0, covariance[0, 0] + covariance[1, 1]))
        sigma_axis = math.sqrt(max(0.0, covariance[3, 3] + covariance[4, 4]))
        lateral = float(np.linalg.norm(estimate[:2]))
        axis = float(np.linalg.norm(estimate[3:5]))
        effective_length = 0.012
        requested_formula = lateral - 3.0 * sigma_lateral - effective_length * math.tan(axis + 3.0 * sigma_axis)
        required_radius = lateral + 3.0 * sigma_lateral + effective_length * math.tan(axis + 3.0 * sigma_axis)
        capture_radius = 0.0 if bcapture is None or bcapture.get("passed") is not True else _capture_radius(bcapture, abs(estimate[3]) + 3.0 * sigma_axis, abs(estimate[4]) + 3.0 * sigma_axis)
        authorized = bool(math.isfinite(required_radius) and capture_radius > 0.0 and required_radius <= capture_radius)
        error = _pose_error(estimate, truth)
        branch_poses = [branch.get("reference_relative_pose_xyz_rpy") for branch in valid]
        branch_lateral_disagreement = math.inf if len(branch_poses) < 2 else float(np.linalg.norm(np.asarray(branch_poses[0])[:2] - np.asarray(branch_poses[1])[:2]))
        branch_axis_disagreement = math.inf if len(branch_poses) < 2 else float(np.linalg.norm(np.asarray(branch_poses[0])[3:5] - np.asarray(branch_poses[1])[3:5]))
        branch_consensus = bool(branch_lateral_disagreement <= 0.00050 and branch_axis_disagreement <= math.radians(1.0))
        authorized = bool(authorized and branch_consensus)
        record = {"episode": index, "success": True, "truth_posthoc_only": truth.tolist(), "estimate": estimate.tolist(), "posthoc_truth_error": error, "three_sigma_lateral_m": 3.0 * sigma_lateral, "three_sigma_axis_rad": 3.0 * sigma_axis, "effective_error_requested_formula_m": requested_formula, "safe_required_capture_radius_m": required_radius, "Bcapture_conservative_radius_m": capture_radius, "control_authorized": authorized, "c2_hypotheses_retained": branch_poses, "c2_uniquely_disambiguated": False, "c2_branch_lateral_disagreement_m": branch_lateral_disagreement, "c2_branch_axis_disagreement_rad": branch_axis_disagreement, "c2_branch_consensus_for_non_yaw_dofs": branch_consensus, "condition_number": selected["condition_number"], "conditional_condition_number_5d": selected["conditional_condition_number_5d"], "selected_branch": selected}
        records.append(record)
        if index == 0:
            for name, delta, observation in zip(VIEW_NAMES, deltas, observations):
                candidate_view_pose = compose_pose(estimate, delta)
                camera = camera_from_plug_pose(candidate_view_pose, mount_eye, mount_target, resolution=(640, 360))
                _overlay(observation, camera, receptacle_cad, output_dir / f"{name.lower()}_overlay.png")
        if (index + 1) % 10 == 0:
            print(json.dumps({"completed": index + 1, "required": args.episodes}), flush=True)
    valid = [item for item in records if item.get("success")]
    stats = {"success_count": len(valid), "authorized_count": sum(item["control_authorized"] for item in valid), "lateral_error_m": _summary([item["posthoc_truth_error"]["lateral_error_m"] for item in valid]), "axis_error_rad": _summary([item["posthoc_truth_error"]["axis_error_rad"] for item in valid]), "three_sigma_lateral_m": _summary([item["three_sigma_lateral_m"] for item in valid]), "three_sigma_axis_rad": _summary([item["three_sigma_axis_rad"] for item in valid]), "condition_number": _summary([item["condition_number"] for item in valid])}
    report = {"schema_version": "kcg_d38999_wrist_multiview_evaluation_v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "episode_count": len(records), "mode": "WRIST_MULTIVIEW", "optimized_variable": "single_reference_T_receptacle_plug", "per_view_pose_then_average": False, "camera_extrinsic_optimized": False, "formal_inputs": ["RGB_D_observations", "TCP_FK_view_deltas", "calibrated_camera_to_assembly_tcp"], "posthoc_truth_not_controller_input": True, "ideal_part_labels_offline_diagnostic": True, "authorization_evaluated": bool(bcapture is not None and bcapture.get("passed") is True), "accuracy_only": bool(args.accuracy_only), "statistics": stats, "episodes": records}
    (output_dir / "multiview_evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "multiview_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("episode", "success", "authorized", "lateral_error_m", "axis_error_rad", "three_sigma_lateral_m", "three_sigma_axis_rad", "condition_number")
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        for item in records:
            error = item.get("posthoc_truth_error", {})
            writer.writerow({"episode": item["episode"], "success": item["success"], "authorized": item.get("control_authorized", False), "lateral_error_m": error.get("lateral_error_m"), "axis_error_rad": error.get("axis_error_rad"), "three_sigma_lateral_m": item.get("three_sigma_lateral_m"), "three_sigma_axis_rad": item.get("three_sigma_axis_rad"), "condition_number": item.get("condition_number")})
    print(json.dumps(stats, sort_keys=True))
    if args.gui:
        images = [cv2.imread(str(output_dir / f"{name.lower()}_overlay.png")) for name in VIEW_NAMES]
        cv2.imshow("WRIST_MULTIVIEW JOINT REGISTRATION", np.hstack(images))
        cv2.waitKey(0 if args.keep_open else 3000)


if __name__ == "__main__":
    main()
