#!/usr/bin/env python3
"""Conditional Body yaw observation from visible rear recesses and RGB-D.

No simulator pose is an input. Wires or a backshell can hide these recesses;
this measurement is not a required source of yaw for a wired connector.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


class RearArrayNotObservable(ValueError):
    """The current image contains too few usable rear features."""


def optional_estimate(frame, source_geometry, pixel_center_offset=.5):
    """Record unavailable features without inventing a Body yaw measurement.

    File/configuration and other unexpected errors still propagate. This only
    handles the estimator's explicit visibility failure; it does not authorize
    motion or replace the separate position/axis and guide-entry checks.
    """
    try:
        return estimate(frame, source_geometry, pixel_center_offset)
    except RearArrayNotObservable as error:
        return {
            "scope": "CURRENT_RGBD_OPTIONAL_REAR_ARRAY_OBSERVATION",
            "source_frame": str(frame), "source_geometry": str(source_geometry),
            "estimated_world_from_body": None,
            "candidate_yaw_solutions": [],
            "yaw_identifiability": {"status": "INSUFFICIENT_VISIBLE_REAR_FEATURES",
                                    "body_pose_returned": False},
            "visibility_failure": str(error),
            "pose_or_contact_truth_used_for_estimation": False,
            "body_yaw_inferred_from_hand_or_nut_rotation": False,
            "wired_connector_visibility_validated": False,
        }


def estimate(frame, source_geometry, pixel_center_offset=0.):
    observation = json.loads((frame / "camera_and_estimate.json").read_text())
    source = json.loads(source_geometry.read_text())
    depth = np.load(frame / "rgbd/depth_m.npy")
    K = np.asarray(observation["intrinsics_3x3"])
    camera = np.asarray(observation["world_from_camera_cv"])
    basis = np.asarray(observation["world_from_plug_five_dof"])
    yy, xx = np.indices(depth.shape)
    rays = np.stack(((xx+pixel_center_offset-K[0, 2])/K[0, 0],
                     (yy+pixel_center_offset-K[1, 2])/K[1, 1], np.ones_like(xx)), -1)
    world = (rays*depth[..., None]) @ camera[:3, :3].T + camera[:3, 3]
    local = (world-basis[:3, 3]) @ basis[:3, :3]
    mask = (np.isfinite(depth) & (depth > 0)
            & (np.abs(local[..., 2]-source["recess_cap_plane_z_m"]) < .00005)
            & (np.linalg.norm(local[..., :2], axis=2) < .0155))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    measured, pixels, areas = [], [], []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not 20 <= area <= 10000:
            continue
        y, x = np.nonzero(labels == label)
        points = local[y, x, :2]
        covariance = np.linalg.eigvalsh(np.cov(points.T))
        # Reject long narrow remnants; retain measurements and diagnostics.
        if covariance[0] <= 0 or covariance[1] > 4*covariance[0]:
            continue
        measured.append(points.mean(0))
        pixels.append([float(x.mean()), float(y.mean())])
        areas.append(area)
    measured = np.asarray(measured)
    if len(measured) < 6:
        raise RearArrayNotObservable(f"only {len(measured)} rear-cap components were measured")
    model = np.asarray(source["cap_centers_body_m"])[:, :2]
    tree = cKDTree(model)

    def rotate(theta):
        return np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def residual(parameters):
        theta, dx, dy = parameters
        R = rotate(theta)
        _, nearest = tree.query((measured-[dx, dy]) @ R)
        return (model[nearest] @ R.T + [dx, dy] - measured).ravel()

    grid = np.deg2rad(np.arange(0., 360., .5))
    scores = np.asarray([np.mean(np.minimum(tree.query(measured @ rotate(a))[0], .001)**2)
                         for a in grid])
    minima = np.flatnonzero((scores < np.roll(scores, 1)) & (scores < np.roll(scores, -1)))
    minima = minima[np.argsort(scores[minima])[:8]]
    candidates = []
    for index in minima:
        theta = grid[index]
        fit = least_squares(residual, [theta, 0., 0.],
                            bounds=([theta-.035, -.0002, -.0002], [theta+.035, .0002, .0002]),
                            loss="soft_l1", f_scale=.00005,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=80)
        distances = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
        candidates.append({"yaw_relative_to_five_dof_basis_deg": float(np.rad2deg(fit.x[0]) % 360),
                           "centre_refinement_in_basis_m": fit.x[1:].tolist(),
                           "nearest_match_rms_m": float(np.sqrt(np.mean(distances**2))),
                           "median_match_distance_m": float(np.median(distances)),
                           "maximum_match_distance_m": float(distances.max()),
                           "matches_within_50um": int(np.sum(distances < .00005)),
                           "optimizer_success": bool(fit.success)})
    # Use the complete visible pattern residual to distinguish the near-180
    # degree alias. A hard per-point inlier cutoff can reverse this ranking
    # when recess occlusion biases otherwise correctly matched centroids.
    candidates.sort(key=lambda c: c["nearest_match_rms_m"])
    best = candidates[0]
    Rz = np.eye(3)
    Rz[:2, :2] = rotate(np.deg2rad(best["yaw_relative_to_five_dof_basis_deg"]))
    estimated = basis.copy()
    estimated[:3, :3] = basis[:3, :3] @ Rz
    estimated[:3, 3] += basis[:3, :3] @ np.r_[best["centre_refinement_in_basis_m"], 0.]
    _, matched = tree.query((measured-best["centre_refinement_in_basis_m"]) @ Rz[:2, :2])
    unique_matches = np.unique(matched)
    half_turn_discrepancy, _ = tree.query(-model)
    # The original 114 symmetric CAD centres agree far below one micrometre;
    # this numerical equivalence test is not a physical sensor accuracy gate.
    asymmetric = half_turn_discrepancy > 1e-6
    distinguishing_matches = unique_matches[asymmetric[unique_matches]]
    duplicated_components = len(measured)-len(unique_matches)
    yaw_status = ("REJECTED_DUPLICATE_CAP_COMPONENTS" if duplicated_components else
                  "AMBIGUOUS_180_DEG_SOURCE_PATTERN" if not len(distinguishing_matches) else
                  "SOURCE_ASYMMETRY_OBSERVED_NOISE_CONFIDENCE_NOT_CALIBRATED")
    usable_candidate = yaw_status == "SOURCE_ASYMMETRY_OBSERVED_NOISE_CONFIDENCE_NOT_CALIBRATED"
    return {"scope": "OFFLINE_SAVED_RGBD_REAR_CAP_ARRAY_ESTIMATION",
            "source_frame": str(frame), "source_geometry": str(source_geometry),
            "captured_physics_time_s": observation["physics_time_s"],
            "detected_cap_components": len(measured), "component_pixel_counts": areas,
            "observed_centres_in_five_dof_basis_m": measured.tolist(),
            "observed_centres_image_px": pixels,
            "cap_plane_band_m": .00005,
            "backprojection_pixel_center_offset_px": pixel_center_offset,
            "candidate_selection": "MINIMUM_COMPLETE_VISIBLE_PATTERN_RMS",
            "centroid_occlusion_bias_not_yet_calibrated": True,
            "candidate_yaw_solutions": candidates,
            "estimated_world_from_body": estimated.tolist() if usable_candidate else None,
            "best_candidate_world_from_body_for_diagnostics": estimated.tolist(),
            "yaw_identifiability": {
                "status": yaw_status,
                "unique_matched_model_features": int(len(unique_matches)),
                "duplicated_measured_cap_components": int(duplicated_components),
                "source_180_degree_distinguishing_feature_count": int(np.sum(asymmetric)),
                "matched_distinguishing_feature_indices": distinguishing_matches.tolist(),
                "source_symmetry_equivalence_tolerance_m": 1e-6,
                "body_pose_returned": usable_candidate,
                "statistical_noise_or_hardware_confidence_calibrated": False,
            },
            "pose_or_contact_truth_used_for_estimation": False,
            "body_yaw_inferred_from_hand_or_nut_rotation": False,
            "known_camera_extrinsics_used": True,
            "accuracy_and_partial_visibility_ambiguity_validated": False,
            "online_control_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("--source-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixel-center-offset", type=float, choices=(0., .5), default=0.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = estimate(args.frame.resolve(), args.source_geometry.resolve(), args.pixel_center_offset)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"detected": result["detected_cap_components"],
                      "candidates": result["candidate_yaw_solutions"][:3]}, indent=2))
