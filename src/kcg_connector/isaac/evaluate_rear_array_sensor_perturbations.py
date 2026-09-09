#!/usr/bin/env python3
"""Controlled offline depth-noise/visibility checks of the rear yaw module.

The preceding five-DOF observation is held fixed to isolate this module. These
synthetic depth images are not new simulator captures or a sensor calibration.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from estimate_rear_contact_array_yaw import estimate


def evaluate(directory, output, geometry):
    frame = directory / "socket_transport/nut_rotation/postgrip_palm"
    observation = json.loads((frame / "camera_and_estimate.json").read_text())
    original = np.load(frame / "rgbd/depth_m.npy")
    output.mkdir(parents=True, exist_ok=False)
    cases = [("nominal", 0., 101, None)]
    cases += [(f"noise_{sigma*1e6:.0f}um_seed{seed}", sigma, seed, None)
              for sigma in (20e-6, 50e-6, 100e-6) for seed in (101, 102, 103)]
    cases += [(f"keep_{side}_half", 0., 101, side) for side in ("left", "right", "top", "bottom")]
    yy, xx = np.indices(original.shape)
    K = np.asarray(observation["intrinsics_3x3"])
    keep_masks = {"left": xx < K[0, 2], "right": xx >= K[0, 2],
                  "top": yy < K[1, 2], "bottom": yy >= K[1, 2]}
    estimates, rows = [], []
    for name, sigma, seed, side in cases:
        case = output / name
        (case / "rgbd").mkdir(parents=True)
        depth = original.astype(np.float64)
        if sigma:
            valid = np.isfinite(depth) & (depth > 0)
            depth[valid] += np.random.default_rng(seed).normal(0., sigma, valid.sum())
        if side:
            depth[~keep_masks[side]] = np.nan
        np.save(case / "rgbd/depth_m.npy", depth.astype(np.float32))
        (case / "camera_and_estimate.json").write_text(json.dumps(observation)+"\n")
        row = {"case": name, "synthetic_depth_noise_std_m": sigma,
               "noise_seed": seed, "visible_half": side,
               "preceding_five_dof_pose_reestimated": False}
        try:
            result = estimate(case, geometry, pixel_center_offset=.5)
            (case / "estimate.json").write_text(json.dumps(result, indent=2)+"\n")
            candidates = result["candidate_yaw_solutions"]
            row.update(estimation_returned=True, detected_components=result["detected_cap_components"],
                       yaw_identifiability=result.get("yaw_identifiability"),
                       best_complete_pattern_rms_m=candidates[0]["nearest_match_rms_m"],
                       second_complete_pattern_rms_m=candidates[1]["nearest_match_rms_m"],
                       competing_yaw_separation_deg=float(abs((
                           candidates[1]["yaw_relative_to_five_dof_basis_deg"]-
                           candidates[0]["yaw_relative_to_five_dof_basis_deg"]+180)%360-180)))
            estimates.append(result)
        except Exception as error:
            row.update(estimation_returned=False, error=f"{type(error).__name__}: {error}")
            estimates.append(None)
        rows.append(row)
        print(json.dumps(row), flush=True)
    # The known pose is loaded only after all estimation calls have completed.
    epoch = json.loads((directory / "rear_array_truth_epoch_for_evaluation_v1.json").read_text())
    truth = np.asarray(epoch["world_from_body_truth"])
    for row, result in zip(rows, estimates):
        if result is None:
            continue
        predicted = np.asarray(result.get("best_candidate_world_from_body_for_diagnostics",
                                          result["estimated_world_from_body"]))
        relative = truth[:3, :3].T @ predicted[:3, :3]
        row.update(evaluated_pose_is_diagnostic_candidate_only=result["estimated_world_from_body"] is None,
                   posthoc_full_rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(relative).magnitude())),
                   posthoc_axial_yaw_error_deg=float(np.rad2deg(np.arctan2(relative[1, 0], relative[0, 0]))),
                   posthoc_translation_error_m=float(np.linalg.norm(predicted[:3, 3]-truth[:3, 3])))
    summary = {"scope": "SYNTHETIC_DEPTH_PERTURBATION_OF_OFFLINE_REAR_YAW_MODULE",
               "original_frame": str(frame), "source_geometry": str(geometry),
               "truth_step_for_evaluation_only": epoch["truth_step"],
               "preceding_five_dof_pose_held_fixed": True,
               "full_perception_pipeline_or_hardware_sensor_validated": False,
               "noise_is_an_assumed_iid_gaussian_depth_error_not_a_measured_sensor_model": True,
               "controller_changed": False, "cases": rows}
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-geometry", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.directory.resolve(), args.output.resolve(), args.source_geometry.resolve())
