#!/usr/bin/env python3
"""Build hash-bound continuous PAD meshes from the active DIRECT hand source.

This builder does not infer PAD semantics or remesh a surface.  It calls the
same ``DIRECT_USER_NAILFREE_STL`` binding used by the offline/Isaac path,
selects the already user-confirmed source-face interval, and only welds
vertices with exactly equal transformed coordinates.  Triangle order and
winding therefore remain identical to the dynamic task surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from kcg_connector.grasp.carts_v2.models import load_v2_config
from kcg_connector.grasp.carts_v2.task_grip_surface import (
    bind_task_hand_variant,
)


_HAND_VARIANT = "DIRECT_USER_NAILFREE_STL"
_SOURCE_SHA256 = (
    "d2dc05c9367075acbf1f4970297b4d3ec6778a067a119fc04ee6c286d1961c58"
)
_SOURCE_FACE_COUNT = 11354
_PAD_FACE_RANGE = (8024, 10465)
_PAD_FACE_COUNT = 2442
_PAD_VERTEX_COUNT = 1233
_PAD_BOUNDARY_EDGE_COUNT = 24
_PAD_BOUNDARY_LOOP_COUNT = 2
_EXPECTED_TRANSFORMS = {
    "f1Link3": {
        "yaw_rad": 2.34686878808,
        "translation_m": [0.02049125144, -0.00300734189, 0.0],
    },
    "f2Link2": {
        "yaw_rad": 2.539984764379217,
        "translation_m": [0.020687502006, 0.000981199056, -0.023999999874],
    },
    "f3Link3": {
        "yaw_rad": 2.34686878808,
        "translation_m": [0.02049125144, -0.00300734189, 0.0],
    },
}
_PAD_ORDER = (
    (1, "finger_1_pad", "f1Link3"),
    (2, "finger_2_pad", "f2Link2"),
    (3, "finger_3_pad", "f3Link3"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_geometry_sha256(points: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256(b"KCG_DIRECT_NAILFREE_PAD_GEOMETRY_V1\0")
    for name, value, dtype in (
        (b"points_local_m", points, np.dtype("<f8")),
        (b"faces", faces, np.dtype("<i8")),
    ):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(name + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _weld_exact_first_occurrence(
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weld numerically equal coordinates without moving or reordering faces."""

    flat = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    vertex_by_coordinate: dict[tuple[float, float, float], int] = {}
    points: list[np.ndarray] = []
    indices = np.empty(len(flat), dtype=np.int64)
    for index, point in enumerate(flat):
        key = (float(point[0]), float(point[1]), float(point[2]))
        vertex = vertex_by_coordinate.get(key)
        if vertex is None:
            vertex = len(points)
            vertex_by_coordinate[key] = vertex
            points.append(np.array(point, dtype=np.float64, copy=True))
        indices[index] = vertex
    welded = np.ascontiguousarray(points, dtype=np.float64)
    faces = np.ascontiguousarray(indices.reshape(-1, 3), dtype=np.int64)
    if not np.array_equal(welded[faces], triangles):
        raise ValueError("exact welding changed a DIRECT PAD triangle coordinate")
    return welded, faces


def _topology(points: np.ndarray, faces: np.ndarray) -> Mapping[str, Any]:
    triangle_points = points[faces]
    doubled_area = np.linalg.norm(
        np.cross(
            triangle_points[:, 1] - triangle_points[:, 0],
            triangle_points[:, 2] - triangle_points[:, 0],
        ),
        axis=1,
    )
    if np.any(doubled_area <= 0.0):
        raise ValueError("DIRECT PAD contains a degenerate triangle")

    edge_rows: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, (a, b, c) in enumerate(np.asarray(faces, dtype=np.int64)):
        for start, stop in ((a, b), (b, c), (c, a)):
            key = (int(min(start, stop)), int(max(start, stop)))
            edge_rows.setdefault(key, []).append(
                (face_index, int(start), int(stop))
            )

    boundary = [edge for edge, rows in edge_rows.items() if len(rows) == 1]
    nonmanifold = [edge for edge, rows in edge_rows.items() if len(rows) > 2]
    winding_conflicts = 0
    parent = np.arange(len(faces), dtype=np.int64)

    def find(value: int) -> int:
        while int(parent[value]) != value:
            parent[value] = parent[int(parent[value])]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for rows in edge_rows.values():
        for row in rows[1:]:
            union(rows[0][0], row[0])
        if len(rows) == 2:
            _face_a, start_a, stop_a = rows[0]
            _face_b, start_b, stop_b = rows[1]
            if start_a != stop_b or stop_a != start_b:
                winding_conflicts += 1

    boundary_graph: dict[int, set[int]] = {}
    for first, second in boundary:
        boundary_graph.setdefault(first, set()).add(second)
        boundary_graph.setdefault(second, set()).add(first)
    if any(len(neighbours) != 2 for neighbours in boundary_graph.values()):
        raise ValueError("DIRECT PAD boundary is not a union of closed loops")
    unvisited = set(boundary_graph)
    boundary_loops = 0
    while unvisited:
        boundary_loops += 1
        stack = [unvisited.pop()]
        while stack:
            current = stack.pop()
            for neighbour in boundary_graph[current]:
                if neighbour in unvisited:
                    unvisited.remove(neighbour)
                    stack.append(neighbour)

    return {
        "area_m2": float(0.5 * np.sum(doubled_area)),
        "boundary_edge_count": len(boundary),
        "boundary_loop_count": boundary_loops,
        "component_count": len({find(index) for index in range(len(faces))}),
        "nonmanifold_edge_count": len(nonmanifold),
        "winding_conflict_count": winding_conflicts,
    }


def _validate_source_binding(inputs: Mapping[str, Any]) -> Path:
    if inputs.get("hand_variant") != _HAND_VARIANT:
        raise ValueError("active hand variant is not DIRECT_USER_NAILFREE_STL")
    if inputs.get("direct_nailfree_stl_sha256") != _SOURCE_SHA256:
        raise ValueError("DIRECT nail-free source hash contract changed")
    if inputs.get("direct_nailfree_stl_unit") != "mm":
        raise ValueError("DIRECT nail-free source unit changed")
    pad = inputs.get("direct_nailfree_pad_surface")
    if not isinstance(pad, Mapping) or tuple(
        pad.get("source_face_index_range_zero_based_inclusive", ())
    ) != _PAD_FACE_RANGE:
        raise ValueError("DIRECT nail-free PAD source-face interval changed")
    if inputs.get("direct_nailfree_link_transforms") != _EXPECTED_TRANSFORMS:
        raise ValueError("DIRECT nail-free link transforms changed")
    source = Path(str(inputs.get("direct_nailfree_stl", ""))).expanduser()
    source = source.resolve(strict=True)
    if not source.is_file() or _sha256(source) != _SOURCE_SHA256:
        raise ValueError("DIRECT nail-free source file is unavailable or changed")
    return source


def build_assets(
    repository_root: Path,
    source_config: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    root = repository_root.resolve(strict=True)
    config_path = (
        source_config.resolve()
        if source_config.is_absolute()
        else (root / source_config).resolve()
    )
    config = load_v2_config(config_path)
    inputs = config.section("inputs")
    source = _validate_source_binding(inputs)
    surfaces, registered, variant = bind_task_hand_variant(root, inputs, {})
    if variant != _HAND_VARIANT or surfaces is None:
        raise ValueError("DIRECT task surfaces were not produced")

    output = (
        output_directory.resolve()
        if output_directory.is_absolute()
        else (root / output_directory).resolve()
    )
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError("output directory must remain inside the repository") from error
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    links: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary_text:
        temporary = Path(temporary_text)
        for finger_number, pad_name, link_name in _PAD_ORDER:
            surface = surfaces[pad_name]
            full_triangles = np.asarray(registered[link_name], dtype=np.float64)
            source_faces = np.asarray(surface.source_face_indices, dtype=np.int64)
            triangles = np.asarray(surface.triangles_local_m, dtype=np.float64)
            if (
                full_triangles.shape != (_SOURCE_FACE_COUNT, 3, 3)
                or len(source_faces) != _PAD_FACE_COUNT
                or not np.array_equal(
                    source_faces,
                    np.arange(_PAD_FACE_RANGE[0], _PAD_FACE_RANGE[1] + 1),
                )
                or not np.array_equal(full_triangles[source_faces], triangles)
            ):
                raise ValueError(f"{link_name}: DIRECT PAD lineage changed")

            points, faces = _weld_exact_first_occurrence(triangles)
            topology = _topology(points, faces)
            if (
                len(points) != _PAD_VERTEX_COUNT
                or len(faces) != _PAD_FACE_COUNT
                or topology["boundary_edge_count"] != _PAD_BOUNDARY_EDGE_COUNT
                or topology["boundary_loop_count"] != _PAD_BOUNDARY_LOOP_COUNT
                or topology["component_count"] != 1
                or topology["nonmanifold_edge_count"] != 0
                or topology["winding_conflict_count"] != 0
            ):
                raise ValueError(
                    f"{link_name}: DIRECT PAD topology differs from the audited source"
                )

            filename = f"{link_name}_DIRECT_NAILFREE_PAD_SOURCE_local_m.npz"
            asset_path = temporary / filename
            with asset_path.open("wb") as stream:
                np.savez(
                    stream,
                    points_local_m=points,
                    faces=faces,
                    source_face_indices=source_faces,
                )
            asset_hash = _sha256(asset_path)
            transform = _EXPECTED_TRANSFORMS[link_name]
            links.append(
                {
                    "finger_number": finger_number,
                    "link_name": link_name,
                    "pad_source_arrays": filename,
                    "pad_source_arrays_sha256": asset_hash,
                    "source_mesh": str(source),
                    "source_mesh_sha256": _SOURCE_SHA256,
                    "diagnostics": {
                        "area_m2": topology["area_m2"],
                        "boundary_edge_count": topology["boundary_edge_count"],
                        "boundary_loop_count": topology["boundary_loop_count"],
                        "component_count": topology["component_count"],
                        "coordinate_tolerance_used": False,
                        "dynamic_use_allowed": False,
                        "exact_source_face_ordinal_lineage_complete": True,
                        "geometry_sha256": _array_geometry_sha256(points, faces),
                        "link_name": link_name,
                        "link_transform": transform,
                        "method_id": (
                            "CARTS_DIRECT_USER_NAILFREE_EXACT_SOURCE_FACE_PAD_V1"
                        ),
                        "nonmanifold_edge_count": topology[
                            "nonmanifold_edge_count"
                        ],
                        "pad_component_is_winding_consistent": True,
                        "pad_source_face_count": len(faces),
                        "pad_source_vertex_count": len(points),
                        "source_face_count": _SOURCE_FACE_COUNT,
                        "source_face_index_range_zero_based_inclusive": list(
                            _PAD_FACE_RANGE
                        ),
                        "source_path": str(source),
                        "source_sha256": _SOURCE_SHA256,
                        "source_vertex_changed": False,
                        "winding_conflict_count": topology[
                            "winding_conflict_count"
                        ],
                    },
                }
            )

        manifest = {
            "coordinate_tolerance_used": False,
            "dynamic_use_allowed": False,
            "links": links,
            "local_points_unit": "metre",
            "method_id": "CARTS_DIRECT_USER_NAILFREE_EXACT_SOURCE_FACE_PAD_V1",
            "online_control_role_truth_allowed": False,
            "schema": "CARTS_EXACT_SOURCE_TERMINAL_PAD_V1",
            "semantic_authority": "USER_CONFIRMED_HAND_GEOMETRY_SEMANTICS",
            "source_authority": "AUTHORED_HAND_STL_GEOMETRY",
            "source_config": str(config_path.relative_to(root)),
            "source_config_sha256": _sha256(config_path),
            "source_face_index_range_zero_based_inclusive": list(_PAD_FACE_RANGE),
            "source_mesh": str(source),
            "source_mesh_sha256": _SOURCE_SHA256,
            "source_vertex_changed": False,
            "status": "STATIC_EXACT_SOURCE_PAD_LINEAGE_VERIFIED",
        }
        manifest_path = temporary / "DIRECT_NAILFREE_PAD_SOURCE_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)

    return {
        "output_directory": str(output.relative_to(root)),
        "manifest": str(
            (output / "DIRECT_NAILFREE_PAD_SOURCE_MANIFEST.json").relative_to(root)
        ),
        "manifest_sha256": _sha256(
            output / "DIRECT_NAILFREE_PAD_SOURCE_MANIFEST.json"
        ),
        "pad_face_count_per_finger": _PAD_FACE_COUNT,
        "pad_vertex_count_per_finger": _PAD_VERTEX_COUNT,
        "source_mesh_sha256": _SOURCE_SHA256,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("src/kcg_connector/config/carts_grasp_v2.yaml"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "artifacts/kcg_connector/continuous_grasp_global_v1/hand/"
            "direct_nailfree_pad_contract_v1"
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    report = build_assets(
        arguments.repository_root,
        arguments.source_config,
        arguments.output_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_assets"]
