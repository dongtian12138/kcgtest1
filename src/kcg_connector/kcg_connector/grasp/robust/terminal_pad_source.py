"""Exact source-face extraction for the user-confirmed three-finger PAD bodies.

The authored terminal-link STL files contain several surface sheets joined by
nonmanifold edges.  Joining through those edges merges distinct drawing
components.  This module therefore connects faces only across an edge owned by
exactly two triangles.  The selected PAD coordinates are copied from the
hash-bound STL triangles; no vertex welding, distance tolerance, or geometric
repair is applied.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh


METHOD_ID = "CARTS_EXACT_RAW_STL_MANIFOLD_EDGE_PAD_SOURCE_V1"


class TerminalPadSourceError(ValueError):
    """Raised when exact PAD source lineage cannot be established."""


def _immutable(value: object, dtype: np.dtype[object]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    result = result.reshape(array.shape)
    result.flags.writeable = False
    return result


def manifold_edge_face_components(faces: object) -> tuple[np.ndarray, ...]:
    """Return face components joined only through exactly two-face edges."""

    face_array = np.asarray(faces)
    if (
        face_array.ndim != 2
        or face_array.shape[1:] != (3,)
        or not np.issubdtype(face_array.dtype, np.integer)
        or len(face_array) == 0
        or int(np.min(face_array)) < 0
    ):
        raise TerminalPadSourceError("faces must be a non-empty nonnegative Mx3 integer array")
    face_array = np.asarray(face_array, dtype=np.int64)
    parent = np.arange(len(face_array), dtype=np.int64)

    def find(index: int) -> int:
        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    edge_owners: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, (first, second, third) in enumerate(face_array):
        for start, end in (
            (first, second),
            (second, third),
            (third, first),
        ):
            edge_owners[tuple(sorted((int(start), int(end))))].append(face_index)
    for owners in edge_owners.values():
        if len(owners) == 2:
            union(owners[0], owners[1])

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for face_index in range(len(face_array)):
        groups[find(face_index)].append(face_index)
    rows = tuple(
        _immutable(indices, np.dtype("<i8"))
        for indices in sorted(groups.values(), key=lambda row: min(row))
    )
    if sum(len(row) for row in rows) != len(face_array):
        raise TerminalPadSourceError("face components do not cover the source mesh")
    return rows


def _indexed_exact_triangles(
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    by_bytes: dict[bytes, int] = {}
    for triangle in np.asarray(triangles, dtype=np.float64):
        face: list[int] = []
        for vertex in triangle:
            canonical = np.asarray(vertex, dtype="<f8")
            key = canonical.tobytes(order="C")
            index = by_bytes.get(key)
            if index is None:
                index = len(points)
                by_bytes[key] = index
                points.append(tuple(float(item) for item in canonical))
            face.append(index)
        faces.append((face[0], face[1], face[2]))
    point_array = _immutable(points, np.dtype("<f8"))
    face_array = _immutable(faces, np.dtype("<i8"))
    if not np.array_equal(point_array[face_array], triangles):
        raise TerminalPadSourceError("indexed PAD triangles changed source coordinates")
    return point_array, face_array


def _surface_edge_counts(faces: np.ndarray) -> tuple[int, int, int]:
    edge_rows: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for first, second, third in np.asarray(faces, dtype=np.int64):
        for start, end in (
            (first, second),
            (second, third),
            (third, first),
        ):
            edge = tuple(sorted((int(start), int(end))))
            edge_rows[edge].append(1 if int(start) < int(end) else -1)
    boundary = sum(len(rows) == 1 for rows in edge_rows.values())
    nonmanifold = sum(len(rows) > 2 for rows in edge_rows.values())
    winding_conflicts = sum(
        len(rows) == 2 and rows[0] == rows[1]
        for rows in edge_rows.values()
    )
    return boundary, nonmanifold, winding_conflicts


def _geometry_sha256(
    source_sha256: str,
    source_face_indices: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(METHOD_ID.encode("ascii") + b"\0")
    digest.update(source_sha256.encode("ascii") + b"\0")
    for value, dtype in (
        (source_face_indices, np.dtype("<i8")),
        (points, np.dtype("<f8")),
        (faces, np.dtype("<i8")),
    ):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ExactTerminalPadSource:
    link_name: str
    source_path: Path
    source_sha256: str
    source_face_count: int
    source_component_count: int
    source_nonmanifold_edge_count: int
    component_area_rank: int
    source_face_indices: np.ndarray
    points_local_m: np.ndarray
    faces: np.ndarray
    boundary_edge_count: int
    nonmanifold_edge_count: int
    winding_conflict_count: int
    geometry_sha256: str

    @property
    def vertex_count(self) -> int:
        return int(len(self.points_local_m))

    @property
    def face_count(self) -> int:
        return int(len(self.faces))

    @property
    def audit(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "method_id": METHOD_ID,
                "link_name": self.link_name,
                "source_path": str(self.source_path),
                "source_sha256": self.source_sha256,
                "source_face_count": self.source_face_count,
                "source_component_count": self.source_component_count,
                "source_nonmanifold_edge_count": self.source_nonmanifold_edge_count,
                "component_area_rank": self.component_area_rank,
                "pad_source_face_count": self.face_count,
                "pad_source_vertex_count": self.vertex_count,
                "boundary_edge_count": self.boundary_edge_count,
                "nonmanifold_edge_count": self.nonmanifold_edge_count,
                "winding_conflict_count": self.winding_conflict_count,
                "geometry_sha256": self.geometry_sha256,
                "coordinate_tolerance_used": False,
                "source_vertex_changed": False,
            }
        )


def extract_exact_terminal_pad_source(
    *,
    link_name: str,
    source_stl_path: str | Path,
    source_stl_sha256: str,
    expected_source_face_count: int,
    expected_pad_face_count: int,
    expected_pad_vertex_count: int,
    expected_component_area_rank: int,
) -> ExactTerminalPadSource:
    """Extract one hash-bound PAD component without moving source vertices."""

    source_path = Path(source_stl_path).resolve()
    if not source_path.is_file() or file_sha256(source_path) != source_stl_sha256:
        raise TerminalPadSourceError("terminal source STL is missing or its SHA-256 changed")
    mesh, provenance = load_stl_mesh(
        source_path,
        unit="m",
        orient_outward=False,
    )
    if (
        provenance.source_sha256 != source_stl_sha256
        or provenance.dropped_degenerate_face_count != 0
        or len(mesh.faces) != expected_source_face_count
    ):
        raise TerminalPadSourceError("terminal source STL structure changed")

    components = manifold_edge_face_components(mesh.faces)
    triangles = mesh.face_vertices_m
    areas: list[float] = []
    candidates: list[np.ndarray] = []
    for indices in components:
        component_triangles = triangles[indices]
        areas.append(
            float(
                np.sum(
                    0.5
                    * np.linalg.norm(
                        np.cross(
                            component_triangles[:, 1] - component_triangles[:, 0],
                            component_triangles[:, 2] - component_triangles[:, 0],
                        ),
                        axis=1,
                    )
                )
            )
        )
        if (
            len(indices) == expected_pad_face_count
            and len(np.unique(mesh.faces[indices])) == expected_pad_vertex_count
        ):
            candidates.append(indices)
    if len(candidates) != 1:
        raise TerminalPadSourceError(
            f"expected one exact PAD component; found {len(candidates)}"
        )
    selected = candidates[0]
    area_order = sorted(range(len(components)), key=lambda index: (-areas[index], index))
    selected_component_index = next(
        index
        for index, component in enumerate(components)
        if np.array_equal(component, selected)
    )
    area_rank = area_order.index(selected_component_index)
    if area_rank != expected_component_area_rank:
        raise TerminalPadSourceError("user-confirmed PAD component area rank changed")

    selected_triangles = np.asarray(triangles[selected], dtype=np.float64)
    points, faces = _indexed_exact_triangles(selected_triangles)
    boundary, nonmanifold, winding_conflicts = _surface_edge_counts(faces)
    if winding_conflicts != 0:
        raise TerminalPadSourceError("exact PAD source has inconsistent triangle winding")

    full_edge_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for first, second, third in np.asarray(mesh.faces, dtype=np.int64):
        for start, end in ((first, second), (second, third), (third, first)):
            full_edge_counts[tuple(sorted((int(start), int(end))))] += 1
    source_nonmanifold = sum(count > 2 for count in full_edge_counts.values())
    source_face_indices = _immutable(selected, np.dtype("<i8"))
    geometry_sha256 = _geometry_sha256(
        source_stl_sha256,
        source_face_indices,
        points,
        faces,
    )
    return ExactTerminalPadSource(
        link_name=link_name,
        source_path=source_path,
        source_sha256=source_stl_sha256,
        source_face_count=len(mesh.faces),
        source_component_count=len(components),
        source_nonmanifold_edge_count=source_nonmanifold,
        component_area_rank=area_rank,
        source_face_indices=source_face_indices,
        points_local_m=points,
        faces=faces,
        boundary_edge_count=boundary,
        nonmanifold_edge_count=nonmanifold,
        winding_conflict_count=winding_conflicts,
        geometry_sha256=geometry_sha256,
    )


__all__ = [
    "ExactTerminalPadSource",
    "METHOD_ID",
    "TerminalPadSourceError",
    "extract_exact_terminal_pad_source",
    "manifold_edge_face_components",
]
