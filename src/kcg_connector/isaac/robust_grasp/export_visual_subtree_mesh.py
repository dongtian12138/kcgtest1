#!/usr/bin/env python3
"""Deterministically export a visual USD mesh subtree for offline grasp planning.

The exporter reads geometry only.  It neither enables physics nor inspects
collision/contact state.  The frozen source layer is opened read-only and the
derived NPZ is accompanied by a provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--subtree", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--maximum-relative-sagitta-error", type=float, required=True
    )
    parser.add_argument("--resolution-multiplier", type=int, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triangulate_face(indices: Sequence[int]) -> list[tuple[int, int, int]]:
    if len(indices) < 3:
        raise ValueError("USD face has fewer than three vertices")
    return [
        (int(indices[0]), int(indices[index]), int(indices[index + 1]))
        for index in range(1, len(indices) - 1)
    ]


def _regular_polygon_edges(
    maximum_relative_sagitta_error: float, resolution_multiplier: int
) -> int:
    """Derive circle resolution from a relative chord-sagitta error."""

    error = float(maximum_relative_sagitta_error)
    if not math.isfinite(error) or not 0.0 < error < 1.0:
        raise ValueError("maximum relative sagitta error must lie in (0, 1)")
    if (
        isinstance(resolution_multiplier, bool)
        or not isinstance(resolution_multiplier, int)
        or resolution_multiplier < 1
    ):
        raise ValueError("resolution multiplier must be a positive integer")
    base = max(3, int(math.ceil(math.pi / math.acos(1.0 - error))))
    while 1.0 - math.cos(math.pi / base) > error:
        base += 1
    return base * resolution_multiplier


def _orient_cylinder_points(points: np.ndarray, axis: str) -> np.ndarray:
    if axis == "Z":
        return points
    if axis == "X":
        return points[:, (2, 0, 1)]
    if axis == "Y":
        return points[:, (1, 2, 0)]
    raise ValueError(f"unsupported USD cylinder axis: {axis!r}")


def _cylinder_mesh(
    *, radius: float, height: float, axis: str, edge_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a closed USD cylinder with outward winding."""

    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("USD cylinder radius must be positive and finite")
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError("USD cylinder height must be positive and finite")
    angles = 2.0 * math.pi * np.arange(edge_count) / float(edge_count)
    ring = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles))
    )
    bottom = np.column_stack((ring, np.full(edge_count, -0.5 * height)))
    top = np.column_stack((ring, np.full(edge_count, 0.5 * height)))
    points = np.vstack((bottom, top, (0.0, 0.0, -0.5 * height), (0.0, 0.0, 0.5 * height)))
    bottom_center = 2 * edge_count
    top_center = bottom_center + 1
    faces: list[tuple[int, int, int]] = []
    for index in range(edge_count):
        following = (index + 1) % edge_count
        faces.extend(
            (
                (index, following, edge_count + following),
                (index, edge_count + following, edge_count + index),
                (bottom_center, following, index),
                (top_center, edge_count + index, edge_count + following),
            )
        )
    return _orient_cylinder_points(points, axis), np.asarray(faces, dtype=np.int64)


def _sphere_mesh(*, radius: float, edge_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a closed USD sphere at the same meridional error bound."""

    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("USD sphere radius must be positive and finite")
    latitude_count = int(math.ceil(edge_count / 2.0))
    points: list[tuple[float, float, float]] = [(0.0, 0.0, radius)]
    for latitude_index in range(1, latitude_count):
        polar = math.pi * latitude_index / float(latitude_count)
        transverse = radius * math.sin(polar)
        z_value = radius * math.cos(polar)
        for longitude_index in range(edge_count):
            azimuth = 2.0 * math.pi * longitude_index / float(edge_count)
            points.append(
                (
                    transverse * math.cos(azimuth),
                    transverse * math.sin(azimuth),
                    z_value,
                )
            )
    south_index = len(points)
    points.append((0.0, 0.0, -radius))
    faces: list[tuple[int, int, int]] = []
    first_ring = 1
    for longitude_index in range(edge_count):
        following = (longitude_index + 1) % edge_count
        faces.append(
            (0, first_ring + longitude_index, first_ring + following)
        )
    for latitude_index in range(latitude_count - 2):
        first = 1 + latitude_index * edge_count
        second = first + edge_count
        for longitude_index in range(edge_count):
            following = (longitude_index + 1) % edge_count
            faces.extend(
                (
                    (
                        first + longitude_index,
                        second + longitude_index,
                        second + following,
                    ),
                    (
                        first + longitude_index,
                        second + following,
                        first + following,
                    ),
                )
            )
    last_ring = 1 + (latitude_count - 2) * edge_count
    for longitude_index in range(edge_count):
        following = (longitude_index + 1) % edge_count
        faces.append(
            (last_ring + longitude_index, south_index, last_ring + following)
        )
    return np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ transform)[:, :3]


def _mesh_arrays(
    stage_path: Path,
    subtree_path: str,
    *,
    maximum_relative_sagitta_error: float,
    resolution_multiplier: int,
):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(stage_path.resolve()), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"cannot open USD stage: {stage_path}")
    root = stage.GetPrimAtPath(subtree_path)
    if not root:
        raise ValueError(f"USD subtree does not exist: {subtree_path}")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    source_prims: list[str] = []
    unsupported_types: dict[str, int] = {}
    source_type_counts: dict[str, int] = {}
    edge_count = _regular_polygon_edges(
        maximum_relative_sagitta_error, resolution_multiplier
    )

    for prim in Usd.PrimRange(root):
        type_name = str(prim.GetTypeName())
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if type_name == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get()]
            indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get()]
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"invalid points on {prim.GetPath()}")
            if sum(counts) != len(indices):
                raise ValueError(f"face index inventory mismatch on {prim.GetPath()}")
            local_faces: list[tuple[int, int, int]] = []
            cursor = 0
            for count in counts:
                local = indices[cursor : cursor + count]
                local_faces.extend(_triangulate_face(local))
                cursor += count
            face_array = np.asarray(local_faces, dtype=np.int64)
        elif type_name == "Cylinder":
            cylinder = UsdGeom.Cylinder(prim)
            points, face_array = _cylinder_mesh(
                radius=float(cylinder.GetRadiusAttr().Get()),
                height=float(cylinder.GetHeightAttr().Get()),
                axis=str(cylinder.GetAxisAttr().Get()),
                edge_count=edge_count,
            )
        elif type_name == "Sphere":
            sphere = UsdGeom.Sphere(prim)
            points, face_array = _sphere_mesh(
                radius=float(sphere.GetRadiusAttr().Get()),
                edge_count=edge_count,
            )
        else:
            if type_name:
                name = str(prim.GetTypeName())
                unsupported_types[name] = unsupported_types.get(name, 0) + 1
            continue
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        world = _transform_points(points, transform)
        offset = sum(len(block) for block in vertices)
        faces.extend(
            tuple(offset + int(value) for value in triangle)
            for triangle in face_array
        )
        vertices.append(world)
        source_prims.append(str(prim.GetPath()))
        source_type_counts[type_name] = source_type_counts.get(type_name, 0) + 1

    if unsupported_types:
        raise ValueError(
            "selected visual subtree contains unsupported non-mesh Gprims: "
            f"{unsupported_types}"
        )
    if not vertices or not faces:
        raise ValueError("selected visual subtree contains no triangulated meshes")
    return (
        np.concatenate(vertices, axis=0),
        np.asarray(faces, dtype=np.int64),
        source_prims,
        source_type_counts,
        edge_count,
    )


def export_visual_subtree(
    *,
    stage_path: Path,
    subtree_path: str,
    output_path: Path,
    manifest_path: Path,
    maximum_relative_sagitta_error: float,
    resolution_multiplier: int,
) -> dict[str, object]:
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    vertices, faces, source_prims, source_type_counts, edge_count = _mesh_arrays(
        stage_path,
        subtree_path,
        maximum_relative_sagitta_error=maximum_relative_sagitta_error,
        resolution_multiplier=resolution_multiplier,
    )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("visual subtree contains non-finite vertices")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        vertices_m=vertices,
        faces=faces,
        source_prim_paths=np.asarray(source_prims, dtype=np.str_),
    )
    document: dict[str, object] = {
        "schema_version": "carts_grasp_visual_subtree_mesh_v1",
        "scope": "OFFLINE_VISUAL_GEOMETRY_ONLY_NO_PHYSX_TRUTH",
        "source_stage": str(stage_path.resolve()),
        "source_stage_sha256": _sha256(stage_path),
        "source_subtree": subtree_path,
        "source_mesh_prim_count": len(source_prims),
        "source_gprim_type_counts": source_type_counts,
        "analytic_primitive_tessellation": {
            "maximum_relative_sagitta_error": maximum_relative_sagitta_error,
            "resolution_multiplier": resolution_multiplier,
            "circle_edge_count": edge_count,
            "derivation": "1-cos(pi/E)<=maximum_relative_sagitta_error",
        },
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "output": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "physics_loaded": False,
        "collision_or_contact_truth_read": False,
        "source_modified": False,
    }
    manifest_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return document


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    report = export_visual_subtree(
        stage_path=arguments.stage,
        subtree_path=arguments.subtree,
        output_path=arguments.output,
        manifest_path=arguments.manifest,
        maximum_relative_sagitta_error=arguments.maximum_relative_sagitta_error,
        resolution_multiplier=arguments.resolution_multiplier,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
