"""Task-frame canonicalisation of unoriented triangle vertex permutations.

The source mesh winding is not a contact-normal oracle.  This module maps all
six vertex permutations of one physical triangle to one representation in an
explicitly registered task frame.  It does not infer a frame from world axes,
PCA, object symmetry, or triangle winding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


UNORIENTED_CANONICAL_REPRESENTATIVE_NORMAL_DIAGNOSTIC_ONLY = (
    "UNORIENTED_CANONICAL_REPRESENTATIVE_NORMAL_DIAGNOSTIC_ONLY"
)
_COMMON_SE3_TASK_COORDINATE_OPERATION_COUNT = 23


class TriangleCanonicalizationError(ValueError):
    """Raised when registered geometry cannot be canonicalised safely."""


def _readonly_finite(
    value: object,
    *,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise TriangleCanonicalizationError(
            f"{label} must be a finite array with shape {shape}"
        )
    array.setflags(write=False)
    return array


def _rotation_tolerance(rotation: np.ndarray) -> float:
    scale = max(1.0, float(np.linalg.norm(rotation, ord=np.inf)))
    return 512.0 * np.finfo(np.float64).eps * scale


def _gamma(operation_count: int) -> float:
    """Higham's standard floating-point forward-error factor gamma_n."""

    unit_roundoff = 0.5 * np.finfo(np.float64).eps
    product = float(operation_count) * unit_roundoff
    if product >= 1.0:
        raise TriangleCanonicalizationError(
            "floating-point error model operation count is invalid"
        )
    return product / (1.0 - product)


def _registered_task_coordinate_intervals(
    triangle: np.ndarray,
    task_frame: "RegisteredTaskFrame",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enclose task coordinates through one common SE(3) frame change.

    The bound follows the actual public equivariance path: six rounded
    operations for a transformed vertex, six for the transformed origin, five
    for a transformed basis entry, then one subtraction and a five-operation
    three-term dot product.  The resulting gamma_23 is an execution-graph
    bound, not an angular, length, or object-specific acceptance tolerance.
    """

    centers = np.empty((3, 3), dtype=np.float64)
    magnitudes = np.empty((3, 3), dtype=np.float64)
    origin = task_frame.origin_object_m
    basis = task_frame.basis_object
    for vertex_index in range(3):
        delta = np.empty(3, dtype=np.float64)
        absolute_delta_bound = np.empty(3, dtype=np.float64)
        for component in range(3):
            delta[component] = (
                float(triangle[vertex_index, component])
                - float(origin[component])
            )
            absolute_delta_bound[component] = (
                abs(float(triangle[vertex_index, component]))
                + abs(float(origin[component]))
            )
        for task_axis in range(3):
            products = tuple(
                float(delta[component]) * float(basis[component, task_axis])
                for component in range(3)
            )
            centers[vertex_index, task_axis] = (
                products[0] + products[1]
            ) + products[2]
            magnitudes[vertex_index, task_axis] = sum(
                absolute_delta_bound[component]
                * abs(float(basis[component, task_axis]))
                for component in range(3)
            )
    error = _gamma(_COMMON_SE3_TASK_COORDINATE_OPERATION_COUNT) * magnitudes
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(error)):
        raise TriangleCanonicalizationError(
            "task-coordinate interval evaluation overflowed"
        )
    lower = np.nextafter(centers - error, -np.inf)
    upper = np.nextafter(centers + error, np.inf)
    return centers, lower, upper


def _certified_lexicographic_order(
    centers: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return a unique forward-error-equivalence-class lexicographic order.

    An axis is skipped for a tied group only when *all* member intervals have
    a common intersection.  Pairwise overlap alone is not transitive and is
    therefore insufficient to define an equivalence class.
    """

    del centers  # Interval endpoints, not point estimates, decide ordering.
    ordered_groups: list[tuple[int, ...]] = [(0, 1, 2)]
    for axis in range(3):
        refined: list[tuple[int, ...]] = []
        for group in ordered_groups:
            if len(group) == 1:
                refined.append(group)
                continue
            by_lower = sorted(
                group,
                key=lambda index: (
                    float(lower[index, axis]),
                    float(upper[index, axis]),
                ),
            )
            clusters: list[list[int]] = []
            current = [by_lower[0]]
            common_lower = float(lower[by_lower[0], axis])
            common_upper = float(upper[by_lower[0], axis])
            maximum_upper = common_upper
            for index in by_lower[1:]:
                candidate_lower = float(lower[index, axis])
                candidate_upper = float(upper[index, axis])
                new_common_lower = max(common_lower, candidate_lower)
                new_common_upper = min(common_upper, candidate_upper)
                if new_common_lower <= new_common_upper:
                    current.append(index)
                    common_lower = new_common_lower
                    common_upper = new_common_upper
                    maximum_upper = max(maximum_upper, candidate_upper)
                    continue
                if candidate_lower <= maximum_upper:
                    raise TriangleCanonicalizationError(
                        "task-coordinate intervals have non-transitive overlap"
                    )
                clusters.append(current)
                current = [index]
                common_lower = candidate_lower
                common_upper = candidate_upper
                maximum_upper = candidate_upper
            clusters.append(current)
            refined.extend(tuple(cluster) for cluster in clusters)
        ordered_groups = refined

    if any(len(group) != 1 for group in ordered_groups):
        raise TriangleCanonicalizationError(
            "triangle vertex order is unresolved by task-coordinate intervals"
        )
    order = np.asarray(
        tuple(group[0] for group in ordered_groups),
        dtype=np.int64,
    )
    order.setflags(write=False)
    return order


def _proper_rotation(value: object, *, label: str) -> np.ndarray:
    rotation = _readonly_finite(value, shape=(3, 3), label=label)
    tolerance = _rotation_tolerance(rotation)
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=tolerance,
    ) or not np.isclose(
        np.linalg.det(rotation),
        1.0,
        rtol=0.0,
        atol=tolerance,
    ):
        raise TriangleCanonicalizationError(f"{label} must belong to SO(3)")
    return rotation


@dataclass(frozen=True)
class RegisteredTaskFrame:
    """A contract-supplied task frame expressed in the object coordinates.

    Columns of ``basis_object`` are the task-frame axes expressed in the
    object frame.  Consequently, row-vector points are mapped into task
    coordinates by ``(point_object - origin_object_m) @ basis_object``.
    """

    origin_object_m: np.ndarray
    basis_object: np.ndarray
    source: str

    def __post_init__(self) -> None:
        origin = _readonly_finite(
            self.origin_object_m,
            shape=(3,),
            label="task-frame origin_object_m",
        )
        basis = _proper_rotation(
            self.basis_object,
            label="task-frame basis_object",
        )
        if not str(self.source):
            raise TriangleCanonicalizationError(
                "registered task-frame source must be non-empty"
            )
        object.__setattr__(self, "origin_object_m", origin)
        object.__setattr__(self, "basis_object", basis)
        object.__setattr__(self, "source", str(self.source))

    def transformed(
        self,
        transform: Sequence[Sequence[float]],
    ) -> "RegisteredTaskFrame":
        """Express the same registered frame after a proper SE(3) change."""

        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise TriangleCanonicalizationError(
                "task-frame transform must be a finite 4x4 matrix"
            )
        tolerance = 512.0 * np.finfo(np.float64).eps * max(
            1.0, float(np.linalg.norm(matrix, ord=np.inf))
        )
        if not np.allclose(
            matrix[3],
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            rtol=0.0,
            atol=tolerance,
        ):
            raise TriangleCanonicalizationError(
                "task-frame transform must have homogeneous last row"
            )
        rotation = _proper_rotation(
            matrix[:3, :3],
            label="task-frame transform rotation",
        )
        return RegisteredTaskFrame(
            origin_object_m=(rotation @ self.origin_object_m + matrix[:3, 3]),
            basis_object=rotation @ self.basis_object,
            source=self.source,
        )


def canonicalize_unoriented_triangles(
    triangles_m: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    *,
    task_frame: RegisteredTaskFrame,
) -> np.ndarray:
    """Return a read-only S3-invariant order in registered coordinates."""

    if not isinstance(task_frame, RegisteredTaskFrame):
        raise TriangleCanonicalizationError(
            "canonicalisation requires an explicit RegisteredTaskFrame"
        )
    triangles = np.asarray(triangles_m, dtype=np.float64)
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or len(triangles) == 0
        or not np.all(np.isfinite(triangles))
    ):
        raise TriangleCanonicalizationError(
            "unoriented triangles must be a non-empty finite array with "
            "shape (F, 3, 3)"
        )
    canonical = np.empty_like(triangles)
    for face_index, triangle in enumerate(triangles):
        centers, lower, upper = _registered_task_coordinate_intervals(
            triangle,
            task_frame,
        )
        order = _certified_lexicographic_order(centers, lower, upper)
        ordered = triangle[order]
        with np.errstate(over="ignore", invalid="ignore"):
            edges = ordered[1:] - ordered[0]
        if not np.all(np.isfinite(edges)):
            raise TriangleCanonicalizationError(
                "canonical triangle edge arithmetic overflowed"
            )
        edge_scale = float(np.max(np.abs(edges)))
        if edge_scale == 0.0:
            raise TriangleCanonicalizationError(
                "degenerate triangle cannot be canonicalised"
            )
        scaled_cross = np.cross(edges[0] / edge_scale, edges[1] / edge_scale)
        if not np.all(np.isfinite(scaled_cross)) or float(
            np.linalg.norm(scaled_cross)
        ) == 0.0:
            raise TriangleCanonicalizationError(
                "degenerate triangle cannot be canonicalised"
            )
        canonical[face_index] = ordered
    canonical.setflags(write=False)
    return canonical


def canonical_representative_normals(
    canonical_triangles_m: Sequence[Sequence[Sequence[float]]] | np.ndarray,
) -> np.ndarray:
    """Return read-only deterministic representatives of unoriented normal lines.

    The sign follows only the canonical vertex order.  It is diagnostic and
    must never be interpreted as outward, inward, or as a force-cone axis.
    """

    triangles = np.asarray(canonical_triangles_m, dtype=np.float64)
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or len(triangles) == 0
        or not np.all(np.isfinite(triangles))
    ):
        raise TriangleCanonicalizationError(
            "canonical triangles must be a non-empty finite array with "
            "shape (F, 3, 3)"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        first_edges = triangles[:, 1] - triangles[:, 0]
        second_edges = triangles[:, 2] - triangles[:, 0]
    edge_scale = np.max(
        np.abs(np.concatenate((first_edges, second_edges), axis=1)),
        axis=1,
    )
    if np.any(~np.isfinite(edge_scale)) or np.any(edge_scale == 0.0):
        raise TriangleCanonicalizationError(
            "canonical triangle edge arithmetic is invalid"
        )
    crosses = np.cross(
        first_edges / edge_scale[:, None],
        second_edges / edge_scale[:, None],
    )
    lengths = np.linalg.norm(crosses, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths == 0.0):
        raise TriangleCanonicalizationError(
            "canonical triangles contain degenerate normals"
        )
    normals = np.array(crosses / lengths[:, None], copy=True)
    normals.setflags(write=False)
    return normals


__all__ = [
    "RegisteredTaskFrame",
    "TriangleCanonicalizationError",
    "UNORIENTED_CANONICAL_REPRESENTATIVE_NORMAL_DIAGNOSTIC_ONLY",
    "canonical_representative_normals",
    "canonicalize_unoriented_triangles",
]
