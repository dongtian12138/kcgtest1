#!/usr/bin/env python3
"""Optimistic offline visibility ablations of rear-face position/axis and yaw.

Pixels are removed from a saved depth image. This is not a rendered cable,
new physical experiment, camera calibration or an end-to-end SAM evaluation.
The old SAM ROI is retained, but position and axis are reestimated per case.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from estimate_rear_contact_array_yaw import optional_estimate
from te_plug_five_dof_geometry import estimate_plug_rear_circle_from_float_depth


def evaluate(frame, geometry, output):
    output.mkdir(parents=True, exist_ok=False)
    observation = json.loads((frame / "camera_and_estimate.json").read_text())
    source = json.loads(geometry.read_text())
    depth = np.load(frame / "rgbd/depth_m.npy")
    roi = cv2.imread(observation["measurement"]["sam"]["mask"], cv2.IMREAD_GRAYSCALE) > 0
    K = np.asarray(observation["intrinsics_3x3"])
    camera = np.asarray(observation["world_from_camera_cv"])
    basis = np.asarray(observation["world_from_plug_five_dof"])
    yy, xx = np.indices(depth.shape)
    rays = np.stack(((xx+.5-K[0, 2])/K[0, 0], (yy+.5-K[1, 2])/K[1, 1], np.ones_like(xx)), -1)
    world = (rays*depth[..., None]) @ camera[:3, :3].T + camera[:3, 3]
    local = (world-basis[:3, 3]) @ basis[:3, :3]
    radius = np.linalg.norm(local[..., :2], axis=-1)
    rear_region = np.abs(local[..., 2]-source["recess_cap_plane_z_m"]) < .002
    cap_pixels = (np.abs(local[..., 2]-source["recess_cap_plane_z_m"]) < .00005) & (radius < .0155)
    masks = {
        "nominal": np.zeros(depth.shape, dtype=bool),
        "all_cap_floors_hidden": cap_pixels,
        "central_rear_disk_12mm_hidden": rear_region & (radius < .012),
        "rear_face_and_rim_18mm_hidden": rear_region & (radius < .018),
    }
    rows = []
    for name, hidden in masks.items():
        case = output / name
        (case / "rgbd").mkdir(parents=True)
        changed = depth.copy()
        changed[hidden] = np.nan
        np.save(case / "rgbd/depth_m.npy", changed)
        np.save(case / "synthetic_visibility_removed.npy", hidden)
        row = {"case": name, "removed_pixels": int(hidden.sum()),
               "position_and_axis_reestimated": True, "sam_roi_reestimated": False,
               "pose_or_contact_truth_used": False}
        try:
            result = estimate_plug_rear_circle_from_float_depth(
                depth_m=changed, mask=roi, intrinsics=K, mesh_path=Path(observation["cad_mm"]),
                pixel_center_offset_px=float(observation.get("metrics", {}).get("depth_pixel_center_offset_px", 0.)))
            new_basis = camera @ result["camera_from_object"]
            fresh = {**observation, "world_from_plug_five_dof": new_basis.tolist(),
                     "synthetic_visibility_case": name}
            (case / "camera_and_estimate.json").write_text(json.dumps(fresh, indent=2)+"\n")
            rear = optional_estimate(case, geometry, .5)
            (case / "rear_array_estimate.json").write_text(json.dumps(rear, indent=2)+"\n")
            row.update(position_and_axis_returned=True,
                       position_change_from_nominal_observation_m=float(np.linalg.norm(new_basis[:3, 3]-basis[:3, 3])),
                       axis_change_from_nominal_observation_deg=float(np.degrees(np.arccos(np.clip(
                           new_basis[:3, 2] @ basis[:3, 2], -1., 1.)))),
                       rear_yaw_returned=rear["estimated_world_from_body"] is not None,
                       rear_yaw_status=rear["yaw_identifiability"]["status"],
                       detected_rear_components=rear.get("detected_cap_components"),
                       circle_metrics=result["metrics"])
        except (ValueError, RuntimeError) as error:
            row.update(position_and_axis_returned=False, rear_yaw_returned=False,
                       reason=f"{type(error).__name__}: {error}")
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != "circle_metrics"}), flush=True)
    report = {"scope": "SYNTHETIC_DEPTH_VISIBILITY_ABLATION_NOT_PHYSICAL_CABLE_EVIDENCE",
              "source_frame": str(frame), "source_geometry": str(geometry),
              "occlusion_mask_uses_saved_image_geometry_not_truth": True,
              "occluder_shape_and_sizes_are_test_assumptions_not_measured_wiring": True,
              "sam_roi_held_fixed_optimistic_segmentation": True,
              "wired_connector_full_pipeline_validated": False,
              "controller_motion_executed": False, "cases": rows}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("--source-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.frame.resolve(), args.source_geometry.resolve(), args.output.resolve())
