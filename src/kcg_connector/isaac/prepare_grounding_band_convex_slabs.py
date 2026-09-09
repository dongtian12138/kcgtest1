#!/usr/bin/env python3
"""Convex axial slabs preserving the source band's contact-facing envelope.

Convex filling on the inner side is permitted only within the rigid-core region.
Source masses/inertias and the original sector meshes remain unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--geometry", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
geometry = json.loads(args.geometry.read_text())
root = args.geometry.parent.parent
with np.load(root / "grounding_band_geometry_03/circumferential_band.npz") as d:
    source_z = np.unique(np.round(d["vertices_m"][:, 2], 10))
records = []
for element in geometry["sectors"]:
    with np.load(element["path"]) as d:
        source = trimesh.Trimesh(d["vertices_m"], d["faces"], process=False)
    hull_records, samples = [], []
    for j, (lower, upper) in enumerate(zip(source_z[:-1], source_z[1:])):
        inside = source.vertices[(source.vertices[:, 2] >= lower-1e-10)
                                 & (source.vertices[:, 2] <= upper+1e-10)]
        points = [inside]
        for z in (lower, upper):
            section = trimesh.intersections.mesh_plane(source, [0., 0., 1.], [0., 0., z])
            points.append(section.reshape(-1, 3))
        points = np.unique(np.concatenate(points), axis=0)
        if len(points) < 4:
            continue
        hull = trimesh.convex.convex_hull(points)
        if hull.volume <= 1e-20:
            continue
        destination = args.output / f"sector_{element['index']:03d}_slab_{j:02d}.npz"
        np.savez_compressed(destination, vertices_m=hull.vertices, faces=hull.faces)
        hull_records.append({"path": str(destination.resolve()), "z_interval_m": [float(lower), float(upper)],
                             "vertices": len(hull.vertices), "triangles": len(hull.faces)})
        samples.extend((hull.vertices, hull.triangles_center))
    samples = np.concatenate(samples)
    # The original socket bore is larger than this radius. Inner convex filling
    # is not accepted as a substitute for preserving the load-bearing outside.
    outer = samples[np.linalg.norm(samples[:, :2], axis=1) >= .0178]
    # Trimesh's fixed small-number tests are not scale covariant on millimetre
    # CAD triangles expressed in metres. Query at millimetre coordinates.
    conditioned_source = trimesh.Trimesh(source.vertices*1000., source.faces, process=False)
    _, distances, _ = trimesh.proximity.closest_point(conditioned_source, outer*1000.)
    distances /= 1000.
    outside = ~conditioned_source.contains(outer*1000.)
    outward_error = float(distances[outside].max()) if outside.any() else 0.
    element["prepared_convex_collision_slabs"] = hull_records
    records.append({"sector_index": element["index"], "slabs": len(hull_records),
                    "outer_samples": len(outer), "max_sampled_outer_expansion_m": outward_error})
geometry["collision_preparation"] = {
    "method": "Convex hull of each original axial interval; original source sector mesh retained",
    "source_geometry_manifest": str(args.geometry.resolve()),
    "world_geometry_verification": "All hull vertices and triangle centroids with r>=17.8 mm; not a continuous Hausdorff bound",
    "geometry_query_coordinate_scale": "millimetres, distances converted back to metres",
    "inner_convex_filling_changes_inertial_properties": False,
    "maximum_sampled_outer_expansion_m": max(r["max_sampled_outer_expansion_m"] for r in records),
    "native_cooking_evaluated": False,
    "records": records,
}
(args.output / "geometry_manifest.json").write_text(json.dumps(geometry, indent=2)+"\n")
print(json.dumps(geometry["collision_preparation"], indent=2))
