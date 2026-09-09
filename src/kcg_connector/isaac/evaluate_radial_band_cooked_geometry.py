"""Postrun point-sampled approximation errors of native cooked band hulls."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull


def samples(mesh):
    triangles = mesh.triangles
    return np.unique(np.vstack((mesh.vertices, triangles.mean(1),
        .5*(triangles[:, 0]+triangles[:, 1]),
        .5*(triangles[:, 1]+triangles[:, 2]),
        .5*(triangles[:, 2]+triangles[:, 0]))), axis=0)


def evaluate(run, output):
    result = json.loads((run / "result.json").read_text())
    geometry = json.loads(Path(result["geometry_manifest"]).read_text())
    source_manifest = json.loads(Path(geometry["source_partition_manifest"]).read_text())
    original = np.load(source_manifest["meshes"]["circumferential_band"]["path"])
    band = trimesh.Trimesh(original["vertices_m"]*1000., original["faces"], process=False)
    if not band.is_volume:
        raise ValueError("original band must have a consistent closed outward surface")
    cooked = json.loads((run / "cooked_collision_manifest.json").read_text())
    rows = []
    for entry in cooked:
        if len(entry["hulls"]) != 1:
            raise ValueError("comparison requires one whole hull per sector")
        index = entry["sector_index"]
        native = np.load(entry["hulls"][0]["path"])
        hull = trimesh.Trimesh(native["vertices_body_m"]*1000., native["faces"], process=False)
        points = samples(hull)
        closest, distance, faces = trimesh.proximity.closest_point(band, points)
        outside = np.einsum("ij,ij->i", points-closest, band.face_normals[faces]) > 1e-9
        source = np.load(geometry["sectors"][index]["path"])
        source_mesh = trimesh.Trimesh(source["vertices_m"]*1000., source["faces"], process=False)
        original_points = samples(source_mesh)
        equations = ConvexHull(hull.vertices).equations
        source_outside_hull = np.max(original_points @ equations[:, :3].T + equations[:, 3], axis=1) > 1e-9
        _, inward_distance, _ = trimesh.proximity.closest_point(hull, original_points)
        row = {"sector_index": index, "cooked_vertices": len(hull.vertices),
               "cooked_sample_points": len(points), "source_sample_points": len(original_points)}
        for radius in (17.8, 17.95, 18.0):
            selected = outside & (np.linalg.norm(points[:, :2], axis=1) >= radius)
            row[f"sampled_outward_error_r_ge_{radius}_mm_m"] = float(distance[selected].max()/1000.) if np.any(selected) else 0.
        selected = source_outside_hull & (np.linalg.norm(original_points[:, :2], axis=1) >= 17.95)
        row["sampled_inward_error_r_ge_17.95_mm_m"] = float(inward_distance[selected].max()/1000.) if np.any(selected) else 0.
        rows.append(row)
    report = {"scope": "POSTRUN_COOKED_HULL_SOURCE_GEOMETRY_SAMPLE_COMPARISON",
              "run": str(run), "query_units": "millimetres; reported errors in metres",
              "sampling": "all vertices, face centroids, edge midpoints; no simulator pose input",
              "outward_classification": "dot with nearest outward source-face normal",
              "inward_classification": "source sample outside native convex hull halfspaces",
              "continuous_Hausdorff_bound_claimed": False, "rows": rows}
    report["maximum_sampled_errors_m"] = {key: max(row[key] for row in rows)
        for key in rows[0] if key.startswith("sampled_")}
    output.open("x").write(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["maximum_sampled_errors_m"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run.resolve(), args.output.resolve())
