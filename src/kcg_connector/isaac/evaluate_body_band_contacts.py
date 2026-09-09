#!/usr/bin/env python3
"""Locate the final stopped Body--Socket contacts on the original band surface."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    run = args.directory.resolve()
    out = run / "final_body_band_contacts_posthoc_v1.json"
    if out.exists():
        raise FileExistsError(out)
    path = run / "truth_samples.jsonl"
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size-64*1024*1024))
        lines = f.read().splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if len(rows) < 481:
        raise ValueError("tail does not cover the required stopped contact window")
    dt = float(rows[-1]["simulation_time_s"]-rows[-2]["simulation_time_s"])
    rows = rows[-round(2./dt):]
    if [s["step"] for s in rows] != list(range(rows[0]["step"], rows[-1]["step"]+1)):
        raise ValueError("contact window has missing steps")
    points, impulses, steps = [], [], []
    for sample in rows:
        rotation = Rotation.from_quat(np.roll(sample["object_part_orientations_wxyz"][0], -1)).as_matrix()
        position = np.asarray(sample["object_part_positions_m"][0])
        for header in sample["contacts"]["tensor_headers"]:
            if not (header["paths"][0].endswith("/Body")
                    and "FixedReceptaclePose" in header["paths"][1]):
                continue
            for contact in header["contacts"]:
                impulse = abs(float(contact["normal_impulse_n_s"]))
                if impulse <= 1e-10:
                    continue
                points.append((np.asarray(contact["position_m"])-position) @ rotation)
                impulses.append(impulse); steps.append(sample["step"])
    points, impulses = np.asarray(points), np.asarray(impulses)
    if len(points) == 0:
        raise ValueError("no final Body--Socket contact impulses")
    source = repo / "artifacts/kcg_connector/isaac/te_full_assembly_20260905/grounding_band_geometry_03/circumferential_band.npz"
    data = np.load(source)
    band = trimesh.Trimesh(data["vertices_m"], data["faces"], process=False)
    projection, distance, faces = trimesh.proximity.closest_point(band, points)
    region = ((np.linalg.norm(points[:, :2], axis=1) > .01735)
              & (points[:, 2] >= -.0141) & (points[:, 2] <= -.00765))
    values = np.c_[np.linalg.norm(points[:, :2], axis=1), points[:, 2]]
    result = {"scope": "POSTRUN_FINAL_CONTACT_POINTS_TO_SOURCE_GROUNDING_BAND",
              "run": str(run), "window_steps": [rows[0]["step"], rows[-1]["step"]],
              "sample_count": len(rows), "contact_point_count": len(points),
              "contact_actor_side": "BODY_SENSOR_ONLY_NO_MIRROR_DOUBLE_COUNT",
              "physics_dt_s": dt, "body_local_contact_r_z_bounds_m": [values.min(0).tolist(), values.max(0).tolist()],
              "band_partition_region_point_fraction": float(region.mean()),
              "band_partition_region_normal_impulse_fraction": float(impulses[region].sum()/impulses.sum()),
              "source_band_projection_distance_min_median_max_m": [float(distance.min()), float(np.median(distance)), float(distance.max())],
              "mean_sum_of_normal_contact_force_magnitudes_n": float(impulses.sum()/(len(rows)*dt)),
              "friction_not_included_in_this_location_audit": True,
              "source_band": str(source), "online_control_used": False,
              "physical_compliance_or_complete_assembly_validated": False}
    out.write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(run / "final_body_band_contacts_samples_v1.npz", points_body_m=points,
                        normal_impulses_n_s=impulses, steps=np.asarray(steps),
                        band_projection_m=projection, band_distance_m=distance, band_face=faces)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
