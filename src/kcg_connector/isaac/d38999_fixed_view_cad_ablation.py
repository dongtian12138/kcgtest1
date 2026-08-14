#!/usr/bin/env python3

"""100-pose fixed-camera observability ablation for proxy-CAD registration.

A/B/C use a deterministic CPU z-buffer of the exact V2 proxy.  Ideal masks
and ideal depth are offline diagnostics only.  D is a semantic-depth
centroid/PCA baseline.  Simulation pose is consulted only after every estimate
is immutable to compute post-hoc error.
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
    CadPoints, fixed_camera_model, overlay_registration, proxy_cad_points,
    register_relative_pose, render_points, transform_points,
)


def _args(repository):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=38999)
    parser.add_argument("--bcapture", default=str(repository / "artifacts/kcg_connector/d38999_insert_proxy_v2/bcapture_20260813T0714Z/bcapture.json"))
    parser.add_argument("--output-dir", default=str(repository / "artifacts/kcg_connector/d38999_visual_ft_e2e_v1/fixed_single_view_ablation"))
    result = parser.parse_args()
    if not result.run or result.episodes < 100:
        parser.error("ablation requires --run and at least 100 episodes")
    if result.keep_open and not result.gui:
        parser.error("--keep-open requires --gui")
    return result


def _world_part(cad, relative, receptacle):
    local = transform_points(cad, relative)
    rotation = Rotation.from_euler("xyz", receptacle[3:]).as_matrix()
    return CadPoints(local.xyz @ rotation.T + receptacle[:3], local.normal @ rotation.T, local.label, local.edge)


def _backproject(camera, depth, mask):
    rows, columns = np.nonzero(mask & np.isfinite(depth))
    values = depth[rows, columns]
    camera_points = np.column_stack(((columns - camera.cx) * values / camera.fx, (rows - camera.cy) * values / camera.fy, values))
    return camera_points @ np.asarray(camera.world_to_camera) + np.asarray(camera.position_world)


def _coarse_initial(camera, observation, receptacle):
    label = observation["label"]
    mask = (label == 1) | (label == 2)
    points = _backproject(camera, observation["depth"], mask)
    if len(points) < 20:
        raise RuntimeError("semantic coarse initializer has insufficient points")
    # The semantic label contains both mating body and coupling nut.  A whole
    # image centroid is badly biased by their 30+ mm axial separation in this
    # oblique view.  Use only the leading axial depth band selected in 3-D by
    # the calibrated fixture/assembly-axis prior.  This is the existing
    # semantic-depth coarse observation, not a truth pose or CAD centroid.
    axial = points[:, 2]
    leading = points[axial >= np.quantile(axial, 0.86)]
    if len(leading) < 20:
        leading = points[np.argsort(axial)[-20:]]
    world = np.median(leading, axis=0)
    relative_xyz = Rotation.from_euler("xyz", receptacle[3:]).inv().apply(world - receptacle[:3])
    return np.asarray((relative_xyz[0], relative_xyz[1], -0.012, 0.0, 0.0, 0.0))


def _pose_error(estimate, truth):
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    translation = estimate[:3] - truth[:3]
    rotation = Rotation.from_euler("xyz", estimate[3:]).inv() * Rotation.from_euler("xyz", truth[3:])
    rotvec = rotation.as_rotvec()
    return {
        "xyz_error_m": float(np.linalg.norm(translation)),
        "lateral_error_m": float(np.linalg.norm(translation[:2])),
        "axis_error_rad": float(np.linalg.norm(rotvec[:2])),
        "yaw_error_mod_c2_rad": float(min(abs(rotvec[2]), abs(abs(rotvec[2]) - math.pi))),
    }


def _point_fit(camera, observation, receptacle):
    plug = _backproject(camera, observation["depth"], (observation["label"] == 1) | (observation["label"] == 2))
    fixed = _backproject(camera, observation["depth"], (observation["label"] == 3) | (observation["label"] == 4))
    def fit(points):
        center = np.median(points, axis=0)
        covariance = np.cov(points - center, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, np.argmax(np.abs(vectors.T @ np.asarray((0.0, 0.0, 1.0))))]
        if axis[2] < 0.0: axis = -axis
        return center, axis
    plug_center, plug_axis = fit(plug)
    fixed_center, fixed_axis = fit(fixed)
    relative_world = plug_center - fixed_center
    relative = Rotation.from_euler("xyz", receptacle[3:]).inv().apply(relative_world)
    axis_local = Rotation.from_euler("xyz", receptacle[3:]).inv().apply(plug_axis)
    return np.asarray((relative[0], relative[1], -0.012, -axis_local[1], axis_local[0], 0.0))


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)), "maximum": float(np.max(values))}


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def _save_residuals(output_dir, prefix, observation, overlay, estimate_render):
    cv2.imwrite(str(output_dir / f"{prefix}_aligned_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    observed_depth, predicted_depth = observation["depth"], estimate_render["depth"]
    valid = np.isfinite(observed_depth) & np.isfinite(predicted_depth)
    residual = np.zeros_like(observed_depth, dtype=np.float32)
    residual[valid] = np.clip(np.abs(observed_depth[valid] - predicted_depth[valid]) / 0.005, 0.0, 1.0)
    cv2.imwrite(str(output_dir / f"{prefix}_depth_residual.png"), np.uint8(255 * residual))
    observed_edge = cv2.Canny(observation["rgb"], 30, 90)
    predicted_edge = cv2.Canny(estimate_render["rgb"], 30, 90)
    edge = np.zeros((*observed_edge.shape, 3), dtype=np.uint8)
    edge[observed_edge > 0] = (0, 255, 0); edge[predicted_edge > 0] = (255, 0, 255)
    cv2.imwrite(str(output_dir / f"{prefix}_edge_residual.png"), cv2.cvtColor(edge, cv2.COLOR_RGB2BGR))


def main():
    repository = Path(__file__).resolve().parents[3]
    args = _args(repository)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    camera = fixed_camera_model()
    plug_cad, receptacle_cad = proxy_cad_points()
    receptacle = np.asarray((0.550, 0.185, 0.2615, 0.0, 0.0, 0.0))
    fixed_world = transform_points(receptacle_cad, receptacle)
    rng = np.random.default_rng(args.seed)
    records = []
    for index in range(args.episodes):
        truth = np.asarray((rng.uniform(-0.002, 0.002), rng.uniform(-0.002, 0.002), -0.012 + rng.uniform(-0.001, 0.001), rng.uniform(-math.radians(4), math.radians(4)), rng.uniform(-math.radians(4), math.radians(4)), rng.uniform(-math.radians(2), math.radians(2))))
        plug_world = _world_part(plug_cad, truth, receptacle)
        ideal = render_points(camera, (plug_world, fixed_world), seed=args.seed + index)
        noisy = render_points(camera, (plug_world, fixed_world), depth_noise_std_m=0.00075, seed=args.seed + index)
        initial = _coarse_initial(camera, ideal, receptacle)
        episode = {"episode": index, "posthoc_truth_relative_pose": truth.tolist(), "coarse_initial": initial.tolist(), "modes": {}}
        for mode, observation in (("A_IDEAL_PART_MASK_IDEAL_DEPTH_EXACT_CAD", ideal), ("B_SEMANTIC_MASK_IDEAL_DEPTH_EXACT_CAD", ideal), ("C_SEMANTIC_MASK_NOISY_DEPTH_EXACT_CAD", noisy)):
            registration_mode = "ideal_part" if mode.startswith("A_") else ("semantic_noisy_depth" if mode.startswith("C_") else "semantic_ideal_depth")
            branches = []
            for branch_yaw in (initial[5], initial[5] + math.pi):
                branch = register_relative_pose(camera, plug_cad, receptacle, observation, initial, mode=registration_mode, yaw_hypothesis_rad=float(branch_yaw))
                branches.append(branch)
            valid = [item for item in branches if item.get("success")]
            if valid:
                selected = min(valid, key=lambda item: item["cost"])
                error = _pose_error(selected["relative_pose_xyz_rotvec"], truth)
                covariance = np.asarray(selected["covariance_6x6"])
                uncertainty = {"three_sigma_lateral_m": float(3.0 * math.sqrt(max(0.0, covariance[0, 0] + covariance[1, 1]))), "three_sigma_axis_rad": float(3.0 * math.sqrt(max(0.0, covariance[3, 3] + covariance[4, 4])))}
                selected = {**selected, "posthoc_truth_error": error, "uncertainty": uncertainty}
            else:
                selected = {"success": False, "reject_reason": branches[0].get("reject_reason", "REGISTRATION_FAILED"), "condition_number": math.inf, "posthoc_truth_error": {"xyz_error_m": math.inf, "lateral_error_m": math.inf, "axis_error_rad": math.inf, "yaw_error_mod_c2_rad": math.inf}, "uncertainty": {"three_sigma_lateral_m": math.inf, "three_sigma_axis_rad": math.inf}}
            episode["modes"][mode] = {"selected": selected, "c2_branches": branches, "ideal_inputs_offline_diagnostic_only": mode.startswith("A_")}
            if index == 0 and selected.get("success"):
                estimate = np.asarray(selected["relative_pose_xyz_rotvec"])
                estimate_world = _world_part(plug_cad, estimate, receptacle)
                predicted = render_points(camera, (estimate_world, fixed_world))
                overlay = overlay_registration(camera, plug_cad, receptacle, observation, estimate)
                _save_residuals(output_dir, mode[0], observation, overlay, predicted)
        point_estimate = _point_fit(camera, ideal, receptacle)
        episode["modes"]["D_CURRENT_CIRCLE_CYLINDER_POINT_FIT"] = {"selected": {"success": True, "relative_pose_xyz_rotvec": point_estimate.tolist(), "posthoc_truth_error": _pose_error(point_estimate, truth), "condition_number": 1.0e18, "uncertainty": {"three_sigma_lateral_m": 1.0e18, "three_sigma_axis_rad": 1.0e18}}, "whole_mask_centroid_used": False}
        records.append(episode)
        if (index + 1) % 10 == 0:
            print(json.dumps({"completed": index + 1, "required": args.episodes}), flush=True)
    modes = list(records[0]["modes"])
    statistics = {}
    for mode in modes:
        selected = [item["modes"][mode]["selected"] for item in records]
        statistics[mode] = {
            "success_count": sum(item.get("success") is True for item in selected),
            "lateral_error_m": _summary([item["posthoc_truth_error"]["lateral_error_m"] for item in selected]),
            "axis_error_rad": _summary([item["posthoc_truth_error"]["axis_error_rad"] for item in selected]),
            "condition_number": _summary([min(float(item.get("condition_number", math.inf)), 1e18) for item in selected]),
        }
    bcapture_path = Path(args.bcapture)
    bcapture = json.loads(bcapture_path.read_text(encoding="utf-8")) if bcapture_path.is_file() else None
    nominal_capture_radius = None
    if bcapture:
        nominal = bcapture["Bcapture_slices"].get("erx_+0.000000_ery_+0.000000")
        nominal_capture_radius = nominal["maximum_success_radius_m"] if nominal else None
    a = statistics["A_IDEAL_PART_MASK_IDEAL_DEPTH_EXACT_CAD"]
    fixed_ill_conditioned = bool(
        nominal_capture_radius is None
        or a["lateral_error_m"]["p95"] > nominal_capture_radius
        or a["axis_error_rad"]["p95"] > math.radians(5.0)
    )
    report = {
        "schema_version": "kcg_d38999_fixed_single_view_cad_ablation_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(records), "same_random_poses_across_modes": True,
        "camera_extrinsic_optimized": False, "formal_control_uses_ideal_inputs": False,
        "mating_feature_roles": {"plug": ["body_nose", "guide_shell", "C2_keys"], "plug_occluder_only": ["coupling_nut", "rear_body"], "receptacle": ["entry_rim", "entrance_chamfer", "guide_bore"], "receptacle_occluder_only": ["rear_shell", "flange"]},
        "statistics": statistics, "nominal_Bcapture_radius_m": nominal_capture_radius,
        "fixed_single_view_conclusion": "FIXED_SINGLE_VIEW_GEOMETRICALLY_ILL_CONDITIONED" if fixed_ill_conditioned else "FIXED_SINGLE_VIEW_IDEAL_CASE_OBSERVABLE",
        "episodes": records,
    }
    (output_dir / "ablation.json").write_text(json.dumps(_finite_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "ablation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["episode", "mode", "success", "lateral_error_m", "axis_error_rad", "condition_number"]
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        for episode in records:
            for mode, value in episode["modes"].items():
                selected = value["selected"]; error = selected["posthoc_truth_error"]
                writer.writerow({"episode": episode["episode"], "mode": mode, "success": selected.get("success"), "lateral_error_m": error["lateral_error_m"], "axis_error_rad": error["axis_error_rad"], "condition_number": selected.get("condition_number")})
    print(json.dumps({"conclusion": report["fixed_single_view_conclusion"], "statistics": statistics}, sort_keys=True))
    if args.gui:
        cv2.imshow("FIXED_SINGLE_VIEW CAD registration overlay", cv2.imread(str(output_dir / "A_aligned_overlay.png")))
        cv2.waitKey(0 if args.keep_open else 3000)


if __name__ == "__main__":
    main()
