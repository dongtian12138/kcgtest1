"""Directed interval wrench matrices from certified three-PAD root domains.

This module performs only a physical-input conversion.  It uses the lower
friction bound to construct fixed inner-cone force rays, then encloses
``(p - origin) cross f`` and ``J_linear.T @ f`` by directed interval
arithmetic.  It does not choose a grasp, solve equilibrium, or use contact
midpoints/samples as formal evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Protocol, Sequence

from mpmath.ctx_iv import MPIntervalContext
import numpy as np

from kcg_connector.grasp.robust.interval_kinematics import (
    INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID,
    IntervalBounds,
    IntervalGeometricJacobian,
)
from kcg_connector.grasp.robust.robust_wrench import (
    friction_cone_inner_relative_error,
    minimum_regular_polygon_edges,
)


METHOD_ID = "CARTS_DIRECTED_INTERVAL_CONTACT_ROOT_WRENCH_MATRIX_V1"
INTERVAL_ARITHMETIC_METHOD_ID = "MPMATH_DIRECTED_INTERVAL_LINEAR_PRODUCTS_V1"
FORMAL_DOMAIN_RULE = (
    "NO_CONTACT_MIDPOINT_OR_FINITE_GEOMETRY_SAMPLE;"
    "DIRECTED_INTERVAL_POSITION_CROSS_FORCE_AND_JACOBIAN_TRANSPOSE_FORCE"
)


class IntervalContactWrenchError(ValueError):
    """Raised when physical root domains cannot form one bound matrix."""


class ContactRootWrenchDomain(Protocol):
    pad_name: str
    formal_root_sha256: str
    position_object_m: tuple[IntervalBounds, IntervalBounds, IntervalBounds]
    path_local_free_side_normal_object: tuple[float, float, float]
    pad_link_name: str
    interval_geometric_jacobian: IntervalGeometricJacobian


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class IntervalContactWrenchMatrices:
    """Immutable interval G/T matrices for one complete three-root choice."""

    grasp_matrix_intervals: tuple[tuple[IntervalBounds, ...], ...]
    joint_torque_from_ray_intervals: tuple[
        tuple[IntervalBounds, ...], ...
    ]
    force_ray_vectors_object: tuple[tuple[float, float, float], ...]
    ray_owner: tuple[int, ...]
    pad_order: tuple[str, str, str]
    independent_joint_names: tuple[str, ...]
    normal_force_caps_n: tuple[float, float, float]
    formal_root_sha256: tuple[str, str, str]
    wrench_origin_object_m: tuple[float, float, float]
    hard_bound_friction_coefficient: float
    base_edges_per_contact: int
    edges_per_contact: int
    maximum_inner_relative_error: float
    method_id: str
    interval_arithmetic_method_id: str
    interval_geometric_jacobian_method_id: str
    formal_domain_rule: str
    decimal_precision: int
    display_approximation_used_as_formal_evidence: bool
    finite_contact_geometry_sampling_used_as_formal_evidence: bool

    def __post_init__(self) -> None:
        grasp = tuple(tuple(row) for row in self.grasp_matrix_intervals)
        torque = tuple(
            tuple(row) for row in self.joint_torque_from_ray_intervals
        )
        forces = tuple(tuple(float(value) for value in row) for row in self.force_ray_vectors_object)
        owners = tuple(self.ray_owner)
        ray_count = len(owners)
        if (
            len(grasp) != 6
            or ray_count < 9
            or any(len(row) != ray_count for row in grasp)
            or not all(
                isinstance(bounds, IntervalBounds)
                for row in grasp
                for bounds in row
            )
            or not self.independent_joint_names
            or len(torque) != len(self.independent_joint_names)
            or any(len(row) != ray_count for row in torque)
            or not all(
                isinstance(bounds, IntervalBounds)
                for row in torque
                for bounds in row
            )
            or len(forces) != ray_count
            or any(
                len(row) != 3
                or not all(math.isfinite(value) for value in row)
                for row in forces
            )
            or len(self.pad_order) != 3
            or len(set(self.pad_order)) != 3
            or any(owner not in (0, 1, 2) for owner in owners)
            or set(owners) != {0, 1, 2}
            or len(self.normal_force_caps_n) != 3
            or not all(
                math.isfinite(value) and value > 0.0
                for value in self.normal_force_caps_n
            )
            or len(self.formal_root_sha256) != 3
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.formal_root_sha256
            )
            or len(self.wrench_origin_object_m) != 3
            or not all(
                math.isfinite(value) for value in self.wrench_origin_object_m
            )
            or not math.isfinite(self.hard_bound_friction_coefficient)
            or self.hard_bound_friction_coefficient < 0.0
            or self.base_edges_per_contact < 3
            or self.edges_per_contact < self.base_edges_per_contact
            or ray_count != 3 * self.edges_per_contact
            or not math.isfinite(self.maximum_inner_relative_error)
            or not 0.0 < self.maximum_inner_relative_error < 1.0
            or self.method_id != METHOD_ID
            or self.interval_arithmetic_method_id
            != INTERVAL_ARITHMETIC_METHOD_ID
            or self.interval_geometric_jacobian_method_id
            != INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID
            or self.formal_domain_rule != FORMAL_DOMAIN_RULE
            or not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
            or self.display_approximation_used_as_formal_evidence
            or self.finite_contact_geometry_sampling_used_as_formal_evidence
        ):
            raise IntervalContactWrenchError(
                "interval contact wrench matrix provenance is malformed"
            )
        object.__setattr__(self, "grasp_matrix_intervals", grasp)
        object.__setattr__(
            self,
            "joint_torque_from_ray_intervals",
            torque,
        )
        object.__setattr__(self, "force_ray_vectors_object", forces)
        object.__setattr__(self, "ray_owner", owners)

    @property
    def ray_count(self) -> int:
        return len(self.ray_owner)

    @property
    def matrix_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._evidence_dict()).encode("utf-8")
        ).hexdigest()

    def _evidence_dict(self) -> dict[str, object]:
        return {
            "grasp_matrix_intervals": [
                [bounds.as_dict() for bounds in row]
                for row in self.grasp_matrix_intervals
            ],
            "joint_torque_from_ray_intervals": [
                [bounds.as_dict() for bounds in row]
                for row in self.joint_torque_from_ray_intervals
            ],
            "force_ray_vectors_object_binary64_hex": [
                [float(value).hex() for value in row]
                for row in self.force_ray_vectors_object
            ],
            "ray_owner": list(self.ray_owner),
            "pad_order": list(self.pad_order),
            "independent_joint_names": list(self.independent_joint_names),
            "normal_force_caps_n_binary64_hex": [
                float(value).hex() for value in self.normal_force_caps_n
            ],
            "formal_root_sha256": list(self.formal_root_sha256),
            "wrench_origin_object_m_binary64_hex": [
                float(value).hex() for value in self.wrench_origin_object_m
            ],
            "hard_bound_friction_coefficient_binary64_hex": float(
                self.hard_bound_friction_coefficient
            ).hex(),
            "base_edges_per_contact": self.base_edges_per_contact,
            "edges_per_contact": self.edges_per_contact,
            "maximum_inner_relative_error_binary64_hex": float(
                self.maximum_inner_relative_error
            ).hex(),
            "method_id": self.method_id,
            "interval_arithmetic_method_id": (
                self.interval_arithmetic_method_id
            ),
            "interval_geometric_jacobian_method_id": (
                self.interval_geometric_jacobian_method_id
            ),
            "formal_domain_rule": self.formal_domain_rule,
            "decimal_precision": self.decimal_precision,
            "display_approximation_used_as_formal_evidence": (
                self.display_approximation_used_as_formal_evidence
            ),
            "finite_contact_geometry_sampling_used_as_formal_evidence": (
                self.finite_contact_geometry_sampling_used_as_formal_evidence
            ),
        }

    def as_dict(self) -> dict[str, object]:
        result = self._evidence_dict()
        result["matrix_sha256"] = self.matrix_sha256
        return result


def _outward_bounds(value: object) -> IntervalBounds:
    lower = float(value.a)
    upper = float(value.b)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise IntervalContactWrenchError(
            "directed interval wrench arithmetic became non-finite"
        )
    return IntervalBounds(
        float(np.nextafter(lower, -math.inf)),
        float(np.nextafter(upper, math.inf)),
    )


def build_interval_contact_wrench_matrices(
    *,
    root_domains: Sequence[ContactRootWrenchDomain],
    pad_order: Sequence[str],
    normal_force_caps_n: Sequence[float],
    wrench_origin_object_m: Sequence[float],
    task_frame_rotation_object: Sequence[Sequence[float]],
    hard_bound_friction_coefficient: float,
    maximum_inner_approximation_relative_error: float,
    cone_edge_multiplier: int,
) -> IntervalContactWrenchMatrices:
    """Build directed interval G and joint-ray matrices for three roots."""

    roots = tuple(root_domains)
    names = tuple(pad_order)
    caps = tuple(float(value) for value in normal_force_caps_n)
    origin = np.asarray(wrench_origin_object_m, dtype=np.float64)
    task_rotation = np.asarray(task_frame_rotation_object, dtype=np.float64)
    friction = float(hard_bound_friction_coefficient)
    maximum_error = float(maximum_inner_approximation_relative_error)
    if (
        len(roots) != 3
        or len(names) != 3
        or len(set(names)) != 3
        or tuple(root.pad_name for root in roots) != names
        or len(caps) != 3
        or not all(math.isfinite(value) and value > 0.0 for value in caps)
        or origin.shape != (3,)
        or not np.all(np.isfinite(origin))
        or task_rotation.shape != (3, 3)
        or not np.all(np.isfinite(task_rotation))
        or not np.allclose(
            task_rotation.T @ task_rotation,
            np.eye(3),
            rtol=0.0,
            atol=256.0 * np.finfo(np.float64).eps,
        )
        or not math.isclose(
            float(np.linalg.det(task_rotation)),
            1.0,
            rel_tol=0.0,
            abs_tol=256.0 * np.finfo(np.float64).eps,
        )
        or not math.isfinite(friction)
        or friction < 0.0
        or not 0.0 < maximum_error < 1.0
        or not isinstance(cone_edge_multiplier, int)
        or isinstance(cone_edge_multiplier, bool)
        or cone_edge_multiplier < 1
    ):
        raise IntervalContactWrenchError(
            "three ordered roots, positive PAD caps, SO(3) task frame, explicit friction and cone contract are required"
        )

    jacobians = tuple(root.interval_geometric_jacobian for root in roots)
    joint_names = jacobians[0].independent_joint_names
    precisions = {jacobian.decimal_precision for jacobian in jacobians}
    if (
        not joint_names
        or len(set(joint_names)) != len(joint_names)
        or any(
            jacobian.independent_joint_names != joint_names
            or jacobian.method_id != INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID
            or jacobian.link_name != root.pad_link_name
            or jacobian.point_object_m != root.position_object_m
            for root, jacobian in zip(roots, jacobians)
        )
        or len(precisions) != 1
    ):
        raise IntervalContactWrenchError(
            "contact roots must share one ordered interval Jacobian domain and precision"
        )
    decimal_precision = precisions.pop()
    free_side_normals = np.asarray(
        [root.path_local_free_side_normal_object for root in roots],
        dtype=np.float64,
    )
    normal_norms = np.linalg.norm(free_side_normals, axis=1)
    if (
        free_side_normals.shape != (3, 3)
        or not np.all(np.isfinite(free_side_normals))
        or not np.allclose(
            normal_norms,
            np.ones(3),
            rtol=0.0,
            atol=64.0 * np.finfo(np.float64).eps,
        )
    ):
        raise IntervalContactWrenchError(
            "every contact root must carry one unit free-side normal"
        )
    inward_normals = -free_side_normals / normal_norms[:, None]

    base_edges = minimum_regular_polygon_edges(maximum_error)
    edges = base_edges * cone_edge_multiplier
    force_rays: list[tuple[float, float, float]] = []
    owners: list[int] = []
    for contact_index, inward in enumerate(inward_normals):
        tangent = None
        for axis_index in range(3):
            axis = task_rotation[:, axis_index]
            proposal = axis - float(axis @ inward) * inward
            proposal_norm = float(np.linalg.norm(proposal))
            if proposal_norm > np.finfo(np.float64).eps:
                tangent = proposal / proposal_norm
                break
        if tangent is None:
            raise IntervalContactWrenchError(
                "task frame produced no contact tangent"
            )
        second_tangent = np.cross(inward, tangent)
        for edge_index in range(edges):
            angle = 2.0 * math.pi * edge_index / float(edges)
            force = inward + friction * (
                math.cos(angle) * tangent
                + math.sin(angle) * second_tangent
            )
            force_rays.append(tuple(float(value) for value in force))
            owners.append(contact_index)

    context = MPIntervalContext()
    context.dps = decimal_precision

    def exact(value: float) -> object:
        return context.mpf([float(value), float(value)])

    def interval(bounds: IntervalBounds) -> object:
        return context.mpf([bounds.lower, bounds.upper])

    grasp_columns: list[tuple[IntervalBounds, ...]] = []
    torque_columns: list[tuple[IntervalBounds, ...]] = []
    for force_tuple, owner in zip(force_rays, owners):
        force = tuple(exact(value) for value in force_tuple)
        relative_position = tuple(
            interval(bounds) - exact(float(origin[axis]))
            for axis, bounds in enumerate(roots[owner].position_object_m)
        )
        moment = (
            relative_position[1] * force[2]
            - relative_position[2] * force[1],
            relative_position[2] * force[0]
            - relative_position[0] * force[2],
            relative_position[0] * force[1]
            - relative_position[1] * force[0],
        )
        grasp_columns.append(
            tuple(IntervalBounds(value, value) for value in force_tuple)
            + tuple(_outward_bounds(value) for value in moment)
        )
        jacobian = jacobians[owner]
        joint_values: list[IntervalBounds] = []
        for joint_index in range(len(joint_names)):
            value = exact(0.0)
            for axis in range(3):
                value += interval(jacobian.elements[axis][joint_index]) * force[axis]
            joint_values.append(_outward_bounds(value))
        torque_columns.append(tuple(joint_values))

    grasp_rows = tuple(
        tuple(column[row] for column in grasp_columns) for row in range(6)
    )
    torque_rows = tuple(
        tuple(column[row] for column in torque_columns)
        for row in range(len(joint_names))
    )
    return IntervalContactWrenchMatrices(
        grasp_matrix_intervals=grasp_rows,
        joint_torque_from_ray_intervals=torque_rows,
        force_ray_vectors_object=tuple(force_rays),
        ray_owner=tuple(owners),
        pad_order=names,  # type: ignore[arg-type]
        independent_joint_names=tuple(joint_names),
        normal_force_caps_n=caps,  # type: ignore[arg-type]
        formal_root_sha256=tuple(
            root.formal_root_sha256 for root in roots
        ),  # type: ignore[arg-type]
        wrench_origin_object_m=tuple(float(value) for value in origin),
        hard_bound_friction_coefficient=friction,
        base_edges_per_contact=base_edges,
        edges_per_contact=edges,
        maximum_inner_relative_error=friction_cone_inner_relative_error(edges),
        method_id=METHOD_ID,
        interval_arithmetic_method_id=INTERVAL_ARITHMETIC_METHOD_ID,
        interval_geometric_jacobian_method_id=(
            INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID
        ),
        formal_domain_rule=FORMAL_DOMAIN_RULE,
        decimal_precision=decimal_precision,
        display_approximation_used_as_formal_evidence=False,
        finite_contact_geometry_sampling_used_as_formal_evidence=False,
    )


__all__ = [
    "FORMAL_DOMAIN_RULE",
    "INTERVAL_ARITHMETIC_METHOD_ID",
    "IntervalContactWrenchError",
    "IntervalContactWrenchMatrices",
    "METHOD_ID",
    "build_interval_contact_wrench_matrices",
]
