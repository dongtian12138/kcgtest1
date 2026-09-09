#!/usr/bin/env python3
"""Offline angular load check for the peak ring of the source grounding band.

This compares a declared equal-line-stiffness idealization to native contact
coupons. It does not identify the TE spring or supply online control forces.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def radial_intersections(segments, angles, *, nearest):
    p = segments[:, 0, :2]
    edge = segments[:, 1, :2]-p
    direction = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    denominator = direction[:, 0, None]*edge[None, :, 1]-direction[:, 1, None]*edge[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (p[:, 0]*edge[:, 1]-p[:, 1]*edge[:, 0])[None, :]/denominator
        fraction = (p[None, :, 0]*direction[:, 1, None]
                    - p[None, :, 1]*direction[:, 0, None])/denominator
    valid = (np.abs(denominator) > 1e-14) & (distance > .01735) & (fraction >= -1e-9) & (fraction <= 1+1e-9)
    values = np.where(valid, distance, np.inf if nearest else -np.inf)
    return values.min(axis=1) if nearest else values.max(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    root = repo / "artifacts/kcg_connector/isaac/te_full_assembly_20260905"
    recipe = json.loads((root / "grounding_band_representative_law_proposal_v1.json").read_text())
    with np.load(root / "grounding_band_geometry_03/circumferential_band.npz") as data:
        vertices, faces = data["vertices_m"], data["faces"]
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    peak_z = float(np.mean(vertices[np.isclose(radius, radius.max(), rtol=0., atol=1e-9), 2]))
    band = trimesh.Trimesh(vertices, faces, process=False)
    # A nearby plane avoids a tangent-plane degeneracy at the exact peak vertices.
    section = trimesh.intersections.mesh_plane(band, [0., 0., 1.], [0., 0., peak_z+1e-9])
    section[:, :, 0] *= -1
    angles = np.arange(7200)*2*np.pi/7200
    band_radius = radial_intersections(section, angles, nearest=False)
    socket = trimesh.load(repo / "artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/D38999_20FJ35SN_VISUAL.obj", force="mesh", process=False)
    socket.vertices *= .001
    results = []
    arrays = {"angles_rad": angles, "band_peak_radius_m": band_radius}
    for depth in (.0105, .012):
        plane_z = -depth-peak_z
        section = trimesh.intersections.mesh_plane(socket, [0., 0., 1.], [0., 0., plane_z])
        socket_radius = radial_intersections(section, angles, nearest=True)
        if not np.isfinite(band_radius).all() or not np.isfinite(socket_radius).all():
            raise RuntimeError("missing radial intersection in the declared annular domain")
        compression = np.maximum(band_radius-socket_radius, 0.)
        line_element = recipe["source_band_maximum_radius_m"]*2*np.pi/len(angles)
        force = recipe["low_deflection_line_stiffness_n_m2"]*line_element*compression
        vector = -np.column_stack((np.cos(angles), np.sin(angles)))*force[:, None]
        results.append({"body_depth_m": depth, "socket_section_z_m": plane_z,
                        "radial_force_magnitude_sum_n": float(force.sum()),
                        "net_lateral_force_world_n": vector.sum(axis=0).tolist(),
                        "net_lateral_force_norm_n": float(np.linalg.norm(vector.sum(axis=0))),
                        "fraction_of_angles_with_positive_compression": float(np.mean(compression > 0.))})
        arrays[f"socket_radius_{depth:.4f}_m"] = socket_radius
        arrays[f"compression_{depth:.4f}_m"] = compression
    np.savez_compressed(args.output / "angular_samples.npz", **arrays)
    result = {"scope": "OFFLINE_EQUAL_LINE_STIFFNESS_PEAK_RING_IDEALIZATION_NOT_TE_CALIBRATION",
              "body_peak_band_z_m": peak_z, "angle_count": len(angles),
              "rigid_source_socket_keyways_included": True,
              "online_control_used": False, "actual_dynamic_deformation_computed": False,
              "axial_force_prediction": "NOT_EVALUATED_BY_THIS_SINGLE_PEAK_SECTION",
              "line_stiffness_source": str(root / "grounding_band_representative_law_proposal_v1.json"),
              "cases": results}
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
