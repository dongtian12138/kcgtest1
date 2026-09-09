#!/usr/bin/env python3
"""Extend the source shallow socket cavities for offline model development.

The exterior openings and all mesh topology remain unchanged; only the 128 blind
floors and their attached straight walls are extended. No runtime is modified.
A physical contact law is still required before using this geometry in assembly.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    root = repo / "artifacts/kcg_connector/isaac/te_full_assembly_20260905"
    audit = json.loads((root / "source_socket_blind_bore_faces_v1.json").read_text())
    source = repo / "artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/D38999_20FJ35SN_VISUAL.obj"
    mesh = trimesh.load(source, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)*.001
    centers = np.asarray([x["center_m"] for x in audit["cap_faces"]])
    distance, cavity = cKDTree(centers[:, :2]).query(vertices[:, :2])
    old_floor = audit["source_front_bore_bottom_z_m"]
    straight_start = audit["source_front_straight_bore_axial_range_m"][1]
    radius = audit["source_front_straight_bore_radius_m"]
    # The comparable manufacturer's cavity dimension is a minimum. Applying
    # it from the source straight-bore start leaves the existing lead-in extra.
    representative_straight_depth = .166*.0254
    new_floor = straight_start-representative_straight_depth
    selected = (abs(vertices[:, 2]-old_floor) < 2e-9) & (distance <= radius+2e-9)
    counts = np.bincount(cavity[selected], minlength=len(centers))
    if len(centers) != 128 or counts.min() < 3:
        raise ValueError("the source-floor selection does not cover every cavity")
    modified = vertices.copy()
    modified[selected, 2] = new_floor
    old_triangles, new_triangles = vertices[mesh.faces], modified[mesh.faces]
    changed_faces = np.any(selected[mesh.faces], axis=1)
    original_area = np.linalg.norm(np.cross(old_triangles[:, 1]-old_triangles[:, 0],
                                           old_triangles[:, 2]-old_triangles[:, 0]), axis=1)/2
    new_area = np.linalg.norm(np.cross(new_triangles[:, 1]-new_triangles[:, 0],
                                      new_triangles[:, 2]-new_triangles[:, 0]), axis=1)/2
    if np.any(new_area[original_area > 0] <= 0):
        raise ValueError("cavity extension collapsed an original nonzero triangle")
    floor_faces = np.all(selected[mesh.faces], axis=1)
    if not np.allclose(original_area[floor_faces], new_area[floor_faces], rtol=0., atol=1e-15):
        raise ValueError("a blind floor changed radius or shape")
    path = args.output / "socket_with_representative_deep_cavities.npz"
    np.savez_compressed(path, vertices_m=modified, faces=np.asarray(mesh.faces, np.int32),
                        moved_source_vertex_indices=np.flatnonzero(selected),
                        changed_source_face_indices=np.flatnonzero(changed_faces))
    result = {"scope": "OFFLINE_REPRESENTATIVE_INTERIOR_GEOMETRY_NOT_ASSEMBLY_MODEL",
              "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "source_assets_modified": False, "installed_in_runtime": False,
              "contact_spring_law_present": False,
              "must_not_use_as_zero_resistance_physical_assembly": True,
              "cavity_count": len(centers), "moved_vertex_count": int(selected.sum()),
              "moved_vertex_count_per_cavity": counts.tolist(),
              "changed_triangle_count": int(changed_faces.sum()),
              "blind_floor_triangle_count": int(floor_faces.sum()),
              "original_floor_z_m": old_floor, "representative_floor_z_m": new_floor,
              "original_straight_bore_start_z_m": straight_start,
              "representative_depth_from_straight_bore_start_m": representative_straight_depth,
              "unchanged_straight_bore_radius_m": radius,
              "unchanged_exterior_openings_and_lead_ins": True,
              "unchanged_original_mesh_topology": True,
              "manufacturer_reference": "https://www.glenair.com/as39029-qpl-and-glenair-commercial-high-performance-connector-contacts/sae-as39029-crimp-contacts/pdf/850-001.pdf",
              "reference_role": "Comparable AS39029/56-348 cavity minimum; not the exact TE interior",
              "nominal_source_shell_stop_depth_m": .014605,
              "pin_tip_to_representative_floor_gap_at_nominal_shell_stop_m": -new_floor-(.014605-.0094615),
              "file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / "geometry_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "moved_vertex_count_per_cavity"},indent=2))


if __name__ == "__main__":
    main()
