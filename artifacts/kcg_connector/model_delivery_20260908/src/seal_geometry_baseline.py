#!/usr/bin/env python3
"""Partition the source front seal envelope while retaining metal pins and keys.

Offline geometry only. The material-region interpretation is supported by the
TE Series III exploded view, not an exact supplier material-volume CAD model.
No stiffness, friction, mass or runtime model is assigned here.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from prepare_te_grounding_band_geometry import cleaned


def prepare(repository, output):
    output.mkdir(parents=True, exist_ok=False)
    old = repository / "artifacts/kcg_connector/isaac/te_full_assembly_20260905"
    source = old / "grounding_band_geometry_03/rigid_body.npz"
    pin_map = old / "source_socket_blind_bore_faces_v1.json"
    data = np.load(source)
    vertices, faces = data["vertices_m"], data["faces"]
    # Construct and boolean in millimetres. At metre scale the installed
    # Trimesh cylinder constructor drops the tiny pin-mask end-cap triangles.
    scale = 1000.
    original = trimesh.Trimesh(vertices*scale, faces, process=False)
    if not original.is_volume:
        raise ValueError("the existing rigid core is not a closed positive volume")
    pin_data = json.loads(pin_map.read_text())
    centers = np.asarray(pin_data["source_plug_pin_centers_m"])
    if centers.shape != (128, 2):
        raise ValueError("the source pin map must retain all 128 contacts")

    # Source features: front insert plane z=-14.5923 mm, disk r=16.002 mm;
    # pin radius 0.37211 mm, raised boss front z=-13.2715 mm. These cuts
    # lie inside the source solid, away from its exterior/material seams.
    # The 0.4077 mm internal backing allocation is not a TE seal thickness.
    z_back, z_front, outer_radius = -.015, -.013270, .01605
    pin_keep_radius = .000373
    height, middle = z_front-z_back, (z_front+z_back)/2
    disk = trimesh.creation.cylinder(outer_radius*scale, height*scale, sections=140)
    disk.apply_translation([0., 0., middle*scale])
    pin_keep = []
    for center in centers:
        cylinder = trimesh.creation.cylinder(pin_keep_radius*scale, (height+.002)*scale, sections=140)
        cylinder.apply_translation([*(center*scale), middle*scale])
        pin_keep.append(cylinder)
    mask = cleaned(trimesh.boolean.difference(
        [disk, trimesh.util.concatenate(pin_keep)], engine="manifold"))
    seal = cleaned(trimesh.boolean.intersection([original, mask], engine="manifold"))
    rigid = cleaned(trimesh.boolean.difference([original, mask], engine="manifold"))
    error = abs(seal.volume+rigid.volume-original.volume)
    if error > original.volume*1e-6:
        raise ValueError("partition does not conserve the source core volume")

    distance, _ = cKDTree(centers).query(vertices[:, :2])
    pins = ((vertices[:, 2] >= -.013271501) & (vertices[:, 2] <= -.009461499)
            & (distance < .0003722))
    key_slab = (vertices[:, 2] >= -.007646) & (vertices[:, 2] <= -.000761)
    key_outer = key_slab & (np.linalg.norm(vertices[:, :2], axis=1) > .0183)
    key_faces = np.all(key_slab[faces], axis=1) & np.any(key_outer[faces], axis=1)
    keys = np.zeros(len(vertices), dtype=bool)
    keys[np.unique(faces[key_faces])] = True
    critical = vertices[pins | keys]
    if pins.sum() < 128*70 or not keys.any():
        raise ValueError("source pin/key selection is incomplete")
    distances = cKDTree(rigid.vertices).query(critical*scale)[0]/scale
    if distances.max() > 1e-8:
        raise ValueError("a critical metal pin or key vertex changed")

    meshes = {}
    for name, mesh in (("rigid_core_without_front_seal", rigid), ("front_seal_envelope", seal)):
        path = output / (name+".npz")
        np.savez_compressed(path, vertices_m=np.asarray(mesh.vertices)/scale,
                            faces=np.asarray(mesh.faces, np.int32))
        meshes[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "volume_m3": float(mesh.volume)/scale**3, "bounds_m": (mesh.bounds/scale).tolist()}
    report = {
        "scope": "OFFLINE_FRONT_SEAL_REGION_PARTITION_NOT_DYNAMIC_ASSEMBLY",
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_pin_map": str(pin_map), "source_assets_modified": False,
        "installed_in_runtime": False, "constitutive_law_assigned": False,
        "mass_inertia_or_friction_modified": False,
        "region_identity": "Inferred interfacial seal envelope; exact TE material-volume mapping unavailable",
        "manufacturer_structure_reference": "artifacts/kcg_connector/reference/j35_contact_numbering_20260905/te_deutsch_38999_catalog.pdf",
        "manufacturer_structure_pages": [9, 31],
        "source_front_insert_plane_z_m": -.0145923,
        "source_front_disk_radius_m": .016002,
        "internal_partition_z_m": [z_back, z_front], "mask_radius_m": outer_radius,
        "metal_pin_preservation_radius_m": pin_keep_radius,
        "internal_partition_is_supplier_thickness": False,
        "boolean_length_unit": "MILLIMETERS; OUTPUT_RESTORED_TO_METERS",
        "source_core_volume_m3": float(original.volume)/scale**3,
        "partition_volume_error_m3": float(error)/scale**3,
        "source_metal_pin_vertices_checked": int(pins.sum()),
        "source_key_vertices_checked": int(keys.sum()),
        "source_key_faces_checked": int(key_faces.sum()),
        "maximum_critical_vertex_distance_to_rigid_partition_m": float(distances.max()),
        "meshes": meshes,
        "unresolved": ["finite elastic and damping law", "contact force scale", "dynamic final seating and retention"],
    }
    (output / "geometry_manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(Path(__file__).resolve().parents[3], args.output.resolve())
    print(json.dumps({k: v for k, v in result.items() if k != "meshes"}, indent=2))
