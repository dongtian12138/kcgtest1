"""Directed interval predicates for finite-witness contact certificates.

The backend evaluates a URDF serial chain with second-order interval jets.
It encloses position, velocity, and acceleration over a closed scalar path
interval, including revolute ``sin``/``cos`` through an independent mpmath
interval context.  No distance, angle, or residual acceptance threshold is
used: callers may rely only on strict separation of an interval from zero.

Object triangles are treated as unoriented normal lines.  A transverse root
freezes the source-winding sign selected by the certified closing velocity;
flipping a triangle therefore flips only that stored sign, not the physical
motion-opposing contact normal constructed by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
from typing import Sequence

from mpmath.ctx_iv import MPIntervalContext
import numpy as np

from kcg_connector.grasp.robust.hand_model import (
    HandModelError,
    JointSpec,
    ThreeFingerHandModel,
)


METHOD_ID = "MPMATH_DIRECTED_INTERVAL_SECOND_ORDER_URDF_JET_V1"
BATCH_POINT_MOTION_METHOD_ID = (
    "MPMATH_DIRECTED_INTERVAL_LINK_JET_WITH_OUTWARD_BINARY64_"
    "BATCH_POINT_AFFINE_EVALUATION_V1"
)
BATCH_POINT_VELOCITY_VECTOR_METHOD_ID = (
    "MPMATH_DIRECTED_INTERVAL_LINK_JET_WITH_OUTWARD_BINARY64_"
    "BATCH_POINT_VELOCITY_VECTOR_EVALUATION_V1"
)
INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID = (
    "MPMATH_DIRECTED_INTERVAL_JOINT_BOX_GEOMETRIC_JACOBIAN_V1"
)
INTERVAL_RIGID_TRANSFORM_METHOD_ID = (
    "MPMATH_DIRECTED_INTERVAL_JOINT_BOX_RIGID_TRANSFORM_V1"
)
IMPLICIT_ROOT_METHOD_ID = "CARTS_CERTIFIED_IMPLICIT_MONOTONE_INTERVAL_ROOT_V1"
IMPLICIT_ROOT_FEATURE_TYPE = (
    "PAD_FINITE_INTERIOR_WITNESS_X_OBJECT_TRIANGLE_PLANE_V1"
)
DISPLAY_APPROXIMATION_ROLE = (
    "DISPLAY_GUI_ONLY_NON_EVIDENTIARY_APPROXIMATION"
)
_BINARY64_STRUCTURE_GAMMA = (
    64.0 * np.finfo(np.float64).eps
    / (1.0 - 64.0 * np.finfo(np.float64).eps)
)
_EARLY_TRIANGLE_ROOT_BRACKET_CHECK_ITERATIONS = frozenset(
    range(8, 49, 8)
)
_MAX_CERTIFIED_INTERVAL_NEWTON_ITERATIONS = 4
_INTERVAL_NEWTON_ENDPOINT_ULP_PADDING = 32
MULTIPHASE_TRANSFORM_CACHE_CAPACITY = 256
MULTIPHASE_POINT_CACHE_CAPACITY = 2048
MULTIPHASE_PAD_AREA_CACHE_CAPACITY = 1024
COMPILED_POINT_PLANE_BINDING_CACHE_CAPACITY = 16
NOMINAL_ROOT_SEED_MAXIMUM_ITERATIONS = 12
NOMINAL_ROOT_SEED_ENDPOINT_ULP_PADDING = 64
NOMINAL_ROOT_SEED_POLICY = (
    "BINARY64_FK_SECANT_BISECTION_HINT_EXACT_INTERVAL_ENDPOINT_"
    "REVERIFICATION_OR_FULL_FALLBACK_V1"
)


class IntervalKinematicsError(ValueError):
    """Raised when an interval kinematics input or enclosure is invalid."""


@dataclass(frozen=True)
class IntervalArithmeticOptions:
    decimal_precision: int
    maximum_root_bisection_iterations: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
        ):
            raise IntervalKinematicsError(
                "decimal_precision must be an explicit positive integer"
            )
        if (
            not isinstance(self.maximum_root_bisection_iterations, int)
            or isinstance(self.maximum_root_bisection_iterations, bool)
            or self.maximum_root_bisection_iterations <= 0
        ):
            raise IntervalKinematicsError(
                "maximum_root_bisection_iterations must be an explicit "
                "positive integer"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "decimal_precision": self.decimal_precision,
            "maximum_root_bisection_iterations": (
                self.maximum_root_bisection_iterations
            ),
        }


@dataclass(frozen=True)
class IntervalBounds:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.lower)
            or not math.isfinite(self.upper)
            or self.lower > self.upper
        ):
            raise IntervalKinematicsError(
                "interval bounds must be finite and ordered"
            )

    @property
    def strictly_positive(self) -> bool:
        return self.lower > 0.0

    @property
    def strictly_negative(self) -> bool:
        return self.upper < 0.0

    @property
    def contains_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper

    def as_dict(self) -> dict[str, str]:
        return {
            "lower_binary64_hex": float(self.lower).hex(),
            "upper_binary64_hex": float(self.upper).hex(),
        }


def _outward_interval_divide(
    numerator: IntervalBounds,
    denominator: IntervalBounds,
) -> IntervalBounds:
    """Divide two binary64 enclosures with one outward rounding step."""

    if denominator.contains_zero:
        raise IntervalKinematicsError(
            "interval Newton denominator must exclude zero"
        )
    quotients = (
        numerator.lower / denominator.lower,
        numerator.lower / denominator.upper,
        numerator.upper / denominator.lower,
        numerator.upper / denominator.upper,
    )
    if not all(math.isfinite(value) for value in quotients):
        raise IntervalKinematicsError(
            "interval Newton quotient must remain finite"
        )
    return IntervalBounds(
        float(np.nextafter(min(quotients), -math.inf)),
        float(np.nextafter(max(quotients), math.inf)),
    )


def _exact_oriented_plane_coefficients(
    triangle_m: np.ndarray,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Return exact rational coefficients for one binary64 triangle plane."""

    triangle = np.asarray(triangle_m, dtype=np.float64)
    if triangle.shape != (3, 3) or not np.all(np.isfinite(triangle)):
        raise IntervalKinematicsError(
            "exact plane coefficients require one finite triangle"
        )
    points = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in triangle
    )
    first_edge = tuple(
        points[1][index] - points[0][index] for index in range(3)
    )
    second_edge = tuple(
        points[2][index] - points[0][index] for index in range(3)
    )
    normal = (
        first_edge[1] * second_edge[2]
        - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0]
        - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1]
        - first_edge[1] * second_edge[0],
    )
    if all(value == 0 for value in normal):
        raise IntervalKinematicsError(
            "exact plane coefficients require a non-degenerate triangle"
        )
    offset = -sum(
        (normal[index] * points[0][index] for index in range(3)),
        Fraction(0),
    )
    return normal[0], normal[1], normal[2], offset


def _exact_same_plane_scale(
    representative_triangle_m: np.ndarray,
    actual_triangle_m: np.ndarray,
) -> Fraction | None:
    """Return actual/representative plane scale only for exact coplanarity."""

    representative = _exact_oriented_plane_coefficients(
        representative_triangle_m
    )
    actual = _exact_oriented_plane_coefficients(actual_triangle_m)
    pivot = next(
        index for index, value in enumerate(representative) if value != 0
    )
    scale = actual[pivot] / representative[pivot]
    if scale == 0 or any(
        actual_value != scale * representative_value
        for actual_value, representative_value in zip(actual, representative)
    ):
        return None
    return scale


@dataclass(frozen=True)
class CertifiedImplicitRoot:
    """A unique mathematical root represented only by an isolating interval.

    ``display_approximation`` is deliberately not evidence of the root value.
    The evidence is the strict endpoint signs, the non-zero derivative
    interval, and the immutable equation/feature identities.
    """

    method_id: str
    equation_sha256: str
    feature_identity_sha256: str
    feature_type: str
    isolating_interval: IntervalBounds
    value_at_lower: IntervalBounds
    value_at_upper: IntervalBounds
    derivative: IntervalBounds
    uniqueness_proven: bool
    display_approximation: float
    display_approximation_role: str

    def __post_init__(self) -> None:
        for label, digest in (
            ("equation", self.equation_sha256),
            ("feature identity", self.feature_identity_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise IntervalKinematicsError(
                    f"implicit root {label} SHA-256 is invalid"
                )
        if (
            self.method_id != IMPLICIT_ROOT_METHOD_ID
            or self.feature_type != IMPLICIT_ROOT_FEATURE_TYPE
            or type(self.uniqueness_proven) is not bool
            or not self.uniqueness_proven
        ):
            raise IntervalKinematicsError(
                "implicit root method, feature, and uniqueness must be explicit"
            )
        for label, bounds in (
            ("isolating interval", self.isolating_interval),
            ("lower endpoint value", self.value_at_lower),
            ("upper endpoint value", self.value_at_upper),
            ("derivative", self.derivative),
        ):
            if not isinstance(bounds, IntervalBounds):
                raise IntervalKinematicsError(
                    f"implicit root {label} must be IntervalBounds"
                )
        lower_sign = (
            1 if self.value_at_lower.strictly_positive else
            -1 if self.value_at_lower.strictly_negative else 0
        )
        upper_sign = (
            1 if self.value_at_upper.strictly_positive else
            -1 if self.value_at_upper.strictly_negative else 0
        )
        derivative_sign = (
            1 if self.derivative.strictly_positive else
            -1 if self.derivative.strictly_negative else 0
        )
        if (
            lower_sign == 0
            or upper_sign != -lower_sign
            or derivative_sign != upper_sign
        ):
            raise IntervalKinematicsError(
                "implicit root endpoint and derivative signs are inconsistent"
            )
        if (
            not math.isfinite(self.display_approximation)
            or not (
                self.isolating_interval.lower
                <= self.display_approximation
                <= self.isolating_interval.upper
            )
            or self.display_approximation_role != DISPLAY_APPROXIMATION_ROLE
        ):
            raise IntervalKinematicsError(
                "implicit root display approximation is not explicitly non-evidentiary"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "equation_sha256": self.equation_sha256,
            "feature_identity_sha256": self.feature_identity_sha256,
            "feature_type": self.feature_type,
            "isolating_interval": self.isolating_interval.as_dict(),
            "value_at_lower": self.value_at_lower.as_dict(),
            "value_at_upper": self.value_at_upper.as_dict(),
            "derivative": self.derivative.as_dict(),
            "uniqueness_proven": self.uniqueness_proven,
            "display_approximation_binary64_hex": (
                float(self.display_approximation).hex()
            ),
            "display_approximation_role": self.display_approximation_role,
        }


@dataclass(frozen=True)
class IntervalContactPredicates:
    phase: IntervalBounds
    position_object_m: tuple[IntervalBounds, IntervalBounds, IntervalBounds]
    plane_value: IntervalBounds
    plane_derivative: IntervalBounds
    plane_second_derivative: IntervalBounds
    triangle_edge_halfspaces: tuple[
        IntervalBounds, IntervalBounds, IntervalBounds
    ]
    pad_approach: IntervalBounds
    object_plane_transversality: IntervalBounds
    method_id: str
    decimal_precision: int


@dataclass(frozen=True)
class IntervalPointMotion:
    phase: IntervalBounds
    position_object_m: tuple[IntervalBounds, IntervalBounds, IntervalBounds]
    velocity_object_m_per_unit: tuple[
        IntervalBounds, IntervalBounds, IntervalBounds
    ]
    acceleration_object_m_per_unit_squared: tuple[
        IntervalBounds, IntervalBounds, IntervalBounds
    ]
    method_id: str
    decimal_precision: int


@dataclass(frozen=True)
class IntervalPointMotionBatch:
    """Array enclosure for many points rigidly attached to one link."""

    phase: IntervalBounds
    position_lower_object_m: np.ndarray
    position_upper_object_m: np.ndarray
    velocity_lower_object_m_per_unit: np.ndarray
    velocity_upper_object_m_per_unit: np.ndarray
    acceleration_lower_object_m_per_unit_squared: np.ndarray
    acceleration_upper_object_m_per_unit_squared: np.ndarray
    method_id: str
    decimal_precision: int

    def __post_init__(self) -> None:
        if not isinstance(self.phase, IntervalBounds):
            raise IntervalKinematicsError(
                "batch point motion phase must be IntervalBounds"
            )
        arrays: list[np.ndarray] = []
        for label, value in (
            ("position lower", self.position_lower_object_m),
            ("position upper", self.position_upper_object_m),
            ("velocity lower", self.velocity_lower_object_m_per_unit),
            ("velocity upper", self.velocity_upper_object_m_per_unit),
            (
                "acceleration lower",
                self.acceleration_lower_object_m_per_unit_squared,
            ),
            (
                "acceleration upper",
                self.acceleration_upper_object_m_per_unit_squared,
            ),
        ):
            array = np.asarray(value, dtype=np.float64)
            if (
                array.ndim != 2
                or array.shape[1:] != (3,)
                or len(array) == 0
                or not np.all(np.isfinite(array))
            ):
                raise IntervalKinematicsError(
                    f"batch point motion {label} must have finite non-empty shape (N, 3)"
                )
            frozen = np.frombuffer(
                np.ascontiguousarray(array).tobytes(order="C"),
                dtype=np.float64,
            ).reshape(array.shape)
            frozen.setflags(write=False)
            arrays.append(frozen)
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise IntervalKinematicsError(
                "batch point motion arrays must share one shape"
            )
        for lower_index, upper_index in ((0, 1), (2, 3), (4, 5)):
            if np.any(arrays[lower_index] > arrays[upper_index]):
                raise IntervalKinematicsError(
                    "batch point motion bounds must be ordered"
                )
        if (
            self.method_id != BATCH_POINT_MOTION_METHOD_ID
            or not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
        ):
            raise IntervalKinematicsError(
                "batch point motion method and precision must be explicit"
            )
        for field_name, array in zip(
            (
                "position_lower_object_m",
                "position_upper_object_m",
                "velocity_lower_object_m_per_unit",
                "velocity_upper_object_m_per_unit",
                "acceleration_lower_object_m_per_unit_squared",
                "acceleration_upper_object_m_per_unit_squared",
            ),
            arrays,
        ):
            object.__setattr__(self, field_name, array)


@dataclass(frozen=True)
class IntervalPointVelocityVectorBatch:
    """Velocity bounds for attached points and orientation bounds for vectors."""

    phase: IntervalBounds
    point_velocity_lower_object_m_per_unit: np.ndarray
    point_velocity_upper_object_m_per_unit: np.ndarray
    vector_lower_object: np.ndarray
    vector_upper_object: np.ndarray
    method_id: str
    decimal_precision: int

    def __post_init__(self) -> None:
        if not isinstance(self.phase, IntervalBounds):
            raise IntervalKinematicsError(
                "batch point velocity/vector phase must be IntervalBounds"
            )
        arrays: list[np.ndarray] = []
        for label, value in (
            (
                "point velocity lower",
                self.point_velocity_lower_object_m_per_unit,
            ),
            (
                "point velocity upper",
                self.point_velocity_upper_object_m_per_unit,
            ),
            ("vector lower", self.vector_lower_object),
            ("vector upper", self.vector_upper_object),
        ):
            array = np.asarray(value, dtype=np.float64)
            if (
                array.ndim != 2
                or array.shape[1:] != (3,)
                or len(array) == 0
                or not np.all(np.isfinite(array))
            ):
                raise IntervalKinematicsError(
                    "batch point velocity/vector "
                    f"{label} must have finite non-empty shape (N, 3)"
                )
            frozen = np.frombuffer(
                np.ascontiguousarray(array).tobytes(order="C"),
                dtype=np.float64,
            ).reshape(array.shape)
            frozen.setflags(write=False)
            arrays.append(frozen)
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise IntervalKinematicsError(
                "batch point velocity/vector arrays must share one shape"
            )
        if np.any(arrays[0] > arrays[1]) or np.any(
            arrays[2] > arrays[3]
        ):
            raise IntervalKinematicsError(
                "batch point velocity/vector bounds must be ordered"
            )
        if (
            self.method_id != BATCH_POINT_VELOCITY_VECTOR_METHOD_ID
            or not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
        ):
            raise IntervalKinematicsError(
                "batch point velocity/vector method and precision must be explicit"
            )
        for field_name, array in zip(
            (
                "point_velocity_lower_object_m_per_unit",
                "point_velocity_upper_object_m_per_unit",
                "vector_lower_object",
                "vector_upper_object",
            ),
            arrays,
        ):
            object.__setattr__(self, field_name, array)


@dataclass(frozen=True)
class IntervalRigidTransform:
    """Three-by-four rigid-transform enclosure over one joint box."""

    link_name: str
    independent_joint_names: tuple[str, ...]
    joint_position_intervals: tuple[IntervalBounds, ...]
    elements: tuple[tuple[IntervalBounds, ...], ...]
    method_id: str
    decimal_precision: int

    def __post_init__(self) -> None:
        if not isinstance(self.link_name, str) or not self.link_name:
            raise IntervalKinematicsError(
                "interval rigid transform link must be named"
            )
        if (
            not self.independent_joint_names
            or any(
                not isinstance(name, str) or not name
                for name in self.independent_joint_names
            )
            or len(set(self.independent_joint_names))
            != len(self.independent_joint_names)
            or len(self.joint_position_intervals)
            != len(self.independent_joint_names)
            or not all(
                isinstance(bounds, IntervalBounds)
                for bounds in self.joint_position_intervals
            )
            or len(self.elements) != 3
            or any(
                len(row) != 4
                or not all(
                    isinstance(bounds, IntervalBounds) for bounds in row
                )
                for row in self.elements
            )
            or self.method_id != INTERVAL_RIGID_TRANSFORM_METHOD_ID
            or not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
        ):
            raise IntervalKinematicsError(
                "interval rigid transform contract is malformed"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "link_name": self.link_name,
            "independent_joint_names": list(self.independent_joint_names),
            "joint_position_intervals": {
                name: bounds.as_dict()
                for name, bounds in zip(
                    self.independent_joint_names,
                    self.joint_position_intervals,
                )
            },
            "elements": [
                [bounds.as_dict() for bounds in row]
                for row in self.elements
            ],
            "method_id": self.method_id,
            "decimal_precision": self.decimal_precision,
        }


@dataclass(frozen=True)
class IntervalGeometricJacobian:
    """Six-by-N geometric Jacobian enclosure over a joint/contact box."""

    link_name: str
    independent_joint_names: tuple[str, ...]
    joint_position_intervals: tuple[IntervalBounds, ...]
    point_object_m: tuple[IntervalBounds, IntervalBounds, IntervalBounds]
    elements: tuple[tuple[IntervalBounds, ...], ...]
    method_id: str
    decimal_precision: int

    def __post_init__(self) -> None:
        if not isinstance(self.link_name, str) or not self.link_name:
            raise IntervalKinematicsError(
                "interval geometric Jacobian link must be named"
            )
        if (
            not self.independent_joint_names
            or any(not isinstance(name, str) or not name for name in self.independent_joint_names)
            or len(set(self.independent_joint_names))
            != len(self.independent_joint_names)
        ):
            raise IntervalKinematicsError(
                "interval geometric Jacobian joint names must be unique and named"
            )
        if (
            len(self.joint_position_intervals)
            != len(self.independent_joint_names)
            or not all(
                isinstance(bounds, IntervalBounds)
                for bounds in self.joint_position_intervals
            )
        ):
            raise IntervalKinematicsError(
                "interval geometric Jacobian joint bounds must match independent joints"
            )
        if len(self.point_object_m) != 3 or not all(
            isinstance(bounds, IntervalBounds) for bounds in self.point_object_m
        ):
            raise IntervalKinematicsError(
                "interval geometric Jacobian point must contain three intervals"
            )
        if len(self.elements) != 6 or any(
            len(row) != len(self.independent_joint_names)
            or not all(isinstance(bounds, IntervalBounds) for bounds in row)
            for row in self.elements
        ):
            raise IntervalKinematicsError(
                "interval geometric Jacobian elements must be a 6-by-N interval matrix"
            )
        if (
            self.method_id != INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID
            or not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
        ):
            raise IntervalKinematicsError(
                "interval geometric Jacobian numerical provenance is invalid"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "link_name": self.link_name,
            "independent_joint_names": list(self.independent_joint_names),
            "joint_position_intervals": {
                name: bounds.as_dict()
                for name, bounds in zip(
                    self.independent_joint_names,
                    self.joint_position_intervals,
                )
            },
            "point_object_m": [bounds.as_dict() for bounds in self.point_object_m],
            "elements": [
                [bounds.as_dict() for bounds in row] for row in self.elements
            ],
            "method_id": self.method_id,
            "decimal_precision": self.decimal_precision,
        }


class IntervalRootState(str, Enum):
    CERTIFIED_FREE = "CERTIFIED_FREE"
    CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT = (
        "CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT"
    )
    CERTIFIED_PAD_DIRECTION_REJECTED_TRANSVERSE_ROOT = (
        "CERTIFIED_PAD_DIRECTION_REJECTED_TRANSVERSE_ROOT"
    )
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class IntervalTransverseRootCertificate:
    implicit_root: CertifiedImplicitRoot
    triangle_edge_halfspaces: tuple[
        IntervalBounds, IntervalBounds, IntervalBounds
    ]
    pad_approach: IntervalBounds
    path_local_free_side_approach: IntervalBounds
    object_source_winding_free_side_sign: int
    position_object_m: tuple[
        IntervalBounds, IntervalBounds, IntervalBounds
    ]
    bisection_iterations: int
    method_id: str
    decimal_precision: int

    @property
    def phase(self) -> IntervalBounds:
        return self.implicit_root.isolating_interval

    @property
    def plane_value_at_lower(self) -> IntervalBounds:
        return self.implicit_root.value_at_lower

    @property
    def plane_value_at_upper(self) -> IntervalBounds:
        return self.implicit_root.value_at_upper

    @property
    def plane_derivative(self) -> IntervalBounds:
        return self.implicit_root.derivative

    @property
    def representative_phase(self) -> float:
        """Compatibility view; never evidence of an exact contact endpoint."""

        return self.implicit_root.display_approximation

    @property
    def representative_phase_role(self) -> str:
        return self.implicit_root.display_approximation_role

    def __post_init__(self) -> None:
        if not isinstance(self.implicit_root, CertifiedImplicitRoot):
            raise IntervalKinematicsError(
                "transverse root must bind a CertifiedImplicitRoot"
            )
        for label, rows in (
            ("triangle edge", self.triangle_edge_halfspaces),
            ("position", self.position_object_m),
        ):
            if len(rows) != 3 or not all(
                isinstance(row, IntervalBounds) for row in rows
            ):
                raise IntervalKinematicsError(
                    f"root certificate {label} bounds must contain three intervals"
                )
        for label, row in (
            ("phase", self.phase),
            ("lower plane", self.plane_value_at_lower),
            ("upper plane", self.plane_value_at_upper),
            ("plane derivative", self.plane_derivative),
            ("PAD approach", self.pad_approach),
            ("path-local free-side approach", self.path_local_free_side_approach),
        ):
            if not isinstance(row, IntervalBounds):
                raise IntervalKinematicsError(
                    f"root certificate {label} must be IntervalBounds"
                )
        lower_sign = (
            1
            if self.plane_value_at_lower.strictly_positive
            else -1
            if self.plane_value_at_lower.strictly_negative
            else 0
        )
        upper_sign = (
            1
            if self.plane_value_at_upper.strictly_positive
            else -1
            if self.plane_value_at_upper.strictly_negative
            else 0
        )
        derivative_sign = (
            1
            if self.plane_derivative.strictly_positive
            else -1
            if self.plane_derivative.strictly_negative
            else 0
        )
        if (
            lower_sign == 0
            or upper_sign != -lower_sign
            or derivative_sign != upper_sign
        ):
            raise IntervalKinematicsError(
                "root endpoint and derivative signs are inconsistent"
            )
        if not all(
            edge.strictly_positive
            for edge in self.triangle_edge_halfspaces
        ):
            raise IntervalKinematicsError(
                "certified root must be strictly inside its triangle"
            )
        if self.pad_approach.contains_zero:
            raise IntervalKinematicsError(
                "certified root contact direction must exclude zero"
            )
        if not self.path_local_free_side_approach.strictly_positive:
            raise IntervalKinematicsError(
                "certified path-local free-side approach must be positive"
            )
        if self.object_source_winding_free_side_sign not in (-1, 1):
            raise IntervalKinematicsError(
                "object source-winding free-side sign must be minus or plus one"
            )
        if self.object_source_winding_free_side_sign != lower_sign:
            raise IntervalKinematicsError(
                "object source-winding free-side sign contradicts the root prefix"
            )
        if (
            not isinstance(self.bisection_iterations, int)
            or isinstance(self.bisection_iterations, bool)
            or self.bisection_iterations < 0
            or self.method_id != METHOD_ID
            or not isinstance(self.decimal_precision, int)
            or self.decimal_precision <= 0
        ):
            raise IntervalKinematicsError(
                "root certificate numerical provenance is invalid"
            )


@dataclass(frozen=True)
class IntervalRootClassification:
    state: IntervalRootState
    searched_phase: IntervalBounds
    certificate: IntervalTransverseRootCertificate | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, IntervalRootState) or not self.reason:
            raise IntervalKinematicsError(
                "root classification state and reason must be explicit"
            )
        root_states = {
            IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT,
            IntervalRootState.CERTIFIED_PAD_DIRECTION_REJECTED_TRANSVERSE_ROOT,
        }
        if (self.state in root_states) != (self.certificate is not None):
            raise IntervalKinematicsError(
                "only certified root states may carry a root certificate"
            )
        if self.certificate is None:
            return
        if (
            self.certificate.phase.lower < self.searched_phase.lower
            or self.certificate.phase.upper > self.searched_phase.upper
        ):
            raise IntervalKinematicsError(
                "root certificate bracket lies outside the searched phase"
            )
        if self.state is (
            IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT
        ):
            if not (
                self.certificate.pad_approach.strictly_positive
                and self.certificate.path_local_free_side_approach.strictly_positive
            ):
                raise IntervalKinematicsError(
                    "directional root state contradicts approach intervals"
                )
        elif self.state is (
            IntervalRootState.CERTIFIED_PAD_DIRECTION_REJECTED_TRANSVERSE_ROOT
        ) and not self.certificate.pad_approach.strictly_negative:
            raise IntervalKinematicsError(
                "direction-rejected root state lacks a rejected approach"
            )


class IntervalPlaneRootState(str, Enum):
    CERTIFIED_FREE = "CERTIFIED_FREE"
    CERTIFIED_TRANSVERSE_PLANE_ROOT = "CERTIFIED_TRANSVERSE_PLANE_ROOT"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class IntervalTransversePlaneRoot:
    """Triangle-independent root data for one exact oriented plane."""

    searched_phase: IntervalBounds
    isolating_interval: IntervalBounds
    value_at_lower: IntervalBounds
    value_at_upper: IntervalBounds
    plane_derivative: IntervalBounds
    position_object_m: tuple[
        IntervalBounds, IntervalBounds, IntervalBounds
    ]
    pad_approach: IntervalBounds
    object_plane_transversality: IntervalBounds
    object_source_winding_free_side_sign: int
    interpolation_iterations: int
    interval_newton_iterations: int
    bisection_iterations: int
    method_id: str
    decimal_precision: int

    def __post_init__(self) -> None:
        for label, value in (
            ("searched phase", self.searched_phase),
            ("isolating interval", self.isolating_interval),
            ("lower plane value", self.value_at_lower),
            ("upper plane value", self.value_at_upper),
            ("plane derivative", self.plane_derivative),
            ("PAD approach", self.pad_approach),
            (
                "object plane transversality",
                self.object_plane_transversality,
            ),
        ):
            if not isinstance(value, IntervalBounds):
                raise IntervalKinematicsError(
                    f"transverse plane root {label} must be IntervalBounds"
                )
        if (
            len(self.position_object_m) != 3
            or not all(
                isinstance(value, IntervalBounds)
                for value in self.position_object_m
            )
        ):
            raise IntervalKinematicsError(
                "transverse plane root position needs three intervals"
            )
        lower_sign = (
            1 if self.value_at_lower.strictly_positive else
            -1 if self.value_at_lower.strictly_negative else 0
        )
        upper_sign = (
            1 if self.value_at_upper.strictly_positive else
            -1 if self.value_at_upper.strictly_negative else 0
        )
        derivative_sign = (
            1 if self.plane_derivative.strictly_positive else
            -1 if self.plane_derivative.strictly_negative else 0
        )
        if (
            lower_sign == 0
            or upper_sign != -lower_sign
            or derivative_sign != upper_sign
            or self.object_source_winding_free_side_sign != lower_sign
            or not self.object_plane_transversality.strictly_positive
            or self.isolating_interval.lower < self.searched_phase.lower
            or self.isolating_interval.upper > self.searched_phase.upper
            or not isinstance(self.interpolation_iterations, int)
            or isinstance(self.interpolation_iterations, bool)
            or self.interpolation_iterations < 0
            or not isinstance(self.interval_newton_iterations, int)
            or isinstance(self.interval_newton_iterations, bool)
            or self.interval_newton_iterations < 0
            or not isinstance(self.bisection_iterations, int)
            or isinstance(self.bisection_iterations, bool)
            or self.bisection_iterations < 0
            or self.method_id != METHOD_ID
            or not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
        ):
            raise IntervalKinematicsError(
                "transverse plane root contract is inconsistent"
            )


@dataclass(frozen=True)
class IntervalPlaneRootClassification:
    state: IntervalPlaneRootState
    searched_phase: IntervalBounds
    root: IntervalTransversePlaneRoot | None
    reason: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.state, IntervalPlaneRootState)
            or not isinstance(self.searched_phase, IntervalBounds)
            or not self.reason
            or (
                self.state is IntervalPlaneRootState.CERTIFIED_TRANSVERSE_PLANE_ROOT
            ) != (self.root is not None)
        ):
            raise IntervalKinematicsError(
                "plane root classification contract is malformed"
            )
        if self.root is not None and self.root.searched_phase != self.searched_phase:
            raise IntervalKinematicsError(
                "plane root classification searched phase changed"
            )


@dataclass(frozen=True)
class _Jet:
    value: object
    first: object
    second: object


@dataclass(frozen=True)
class _ObjectPlaneValueData:
    triangle_binary64: bytes
    origin: tuple[object, object, object]
    area: tuple[object, object, object]


@dataclass(frozen=True)
class _IntervalPlaneMotion:
    """Only the two strict plane quantities needed before root isolation."""

    phase: IntervalBounds
    plane_value: IntervalBounds
    plane_derivative: IntervalBounds


@dataclass
class _CompiledPointPlaneBinding:
    """Compiled exact-point evaluator with immutable path identity."""

    evaluator: object
    link_name: str
    q_start_binary64: bytes
    direction_binary64: bytes
    base_transform_binary64: bytes
    witness_binary64: bytes
    triangle_binary64: bytes
    enabled: bool = True
    failure_reason: str | None = None

    @staticmethod
    def _identity(value: object) -> bytes:
        return np.asarray(value, dtype=">f8").tobytes(order="C")

    def matches(
        self,
        *,
        link_name: str,
        q_start: object,
        direction: object,
        base_transform: object,
        witness: object,
        triangle_binary64: bytes,
    ) -> bool:
        return (
            link_name == self.link_name
            and self._identity(q_start) == self.q_start_binary64
            and self._identity(direction) == self.direction_binary64
            and self._identity(base_transform)
            == self.base_transform_binary64
            and self._identity(witness) == self.witness_binary64
            and triangle_binary64 == self.triangle_binary64
        )


@dataclass(frozen=True)
class _CompiledPointPlaneLinkPlan:
    """Read-only compiled payload shared by every witness on one link."""

    joint_types: np.ndarray
    source_indices: np.ndarray
    origins_xyz_m: np.ndarray
    origins_rpy_rad: np.ndarray
    axes: np.ndarray
    multipliers: np.ndarray
    offsets: np.ndarray


class IntervalLinkTransformCache:
    """Per-call cache for identical directed interval kinematic primitives.

    The cache is created by exactly one DirectedIntervalKinematics instance.
    Keys retain every binary64 input bit.  Second-order and value-only link
    matrices are frozen as tuples and copied back to mutable outer containers
    for callers.  Multiple phase intervals may be reused because the same
    endpoints and midpoint are shared by many witness/plane queries.  Every
    category has a fixed FIFO capacity, so one difficult root search cannot
    grow memory without a bound.
    """

    def __init__(self, owner_token: object) -> None:
        self._owner_token = owner_token
        self._entries: dict[
            tuple[str, bytes, bytes, bytes, bytes],
            tuple[tuple[_Jet, ...], ...],
        ] = {}
        self._value_entries: dict[
            tuple[str, bytes, bytes, bytes, bytes],
            tuple[tuple[object, ...], ...],
        ] = {}
        self._value_point_entries: dict[
            tuple[
                tuple[str, bytes, bytes, bytes, bytes],
                bytes,
            ],
            tuple[object, object, object],
        ] = {}
        self._point_entries: dict[
            tuple[
                tuple[str, bytes, bytes, bytes, bytes],
                bytes,
            ],
            tuple[_Jet, _Jet, _Jet],
        ] = {}
        self._pad_area_entries: dict[
            tuple[
                tuple[str, bytes, bytes, bytes, bytes],
                bytes,
            ],
            tuple[_Jet, _Jet, _Jet],
        ] = {}
        self._primary_phase_key: bytes | None = None
        self.hit_count = 0
        self.miss_count = 0
        self.nonprimary_transform_bypass_count = 0
        self.point_hit_count = 0
        self.point_miss_count = 0
        self.point_nonprimary_bypass_count = 0
        self.pad_area_hit_count = 0
        self.pad_area_miss_count = 0
        self.pad_area_nonprimary_bypass_count = 0
        self.transform_eviction_count = 0
        self.point_eviction_count = 0
        self.pad_area_eviction_count = 0

    @staticmethod
    def _bounded_store(
        entries: dict[object, object],
        key: object,
        value: object,
        capacity: int,
    ) -> bool:
        """Store one immutable value and report whether FIFO eviction occurred."""

        if key in entries:
            entries.pop(key)
        evicted = False
        while len(entries) >= capacity:
            entries.pop(next(iter(entries)))
            evicted = True
        entries[key] = value
        return evicted

    @property
    def entry_count(self) -> int:
        return len(self._entries) + len(self._value_entries)

    @property
    def point_entry_count(self) -> int:
        return len(self._point_entries) + len(self._value_point_entries)

    @property
    def pad_area_entry_count(self) -> int:
        return len(self._pad_area_entries)


def _add(first: _Jet, second: _Jet) -> _Jet:
    return _Jet(
        first.value + second.value,
        first.first + second.first,
        first.second + second.second,
    )


def _neg(value: _Jet) -> _Jet:
    return _Jet(-value.value, -value.first, -value.second)


def _sub(first: _Jet, second: _Jet) -> _Jet:
    return _add(first, _neg(second))


def _mul(first: _Jet, second: _Jet) -> _Jet:
    return _Jet(
        first.value * second.value,
        first.first * second.value + first.value * second.first,
        first.second * second.value
        + 2 * first.first * second.first
        + first.value * second.second,
    )


def _sin(context: MPIntervalContext, value: _Jet) -> _Jet:
    sine = context.sin(value.value)
    cosine = context.cos(value.value)
    return _Jet(
        sine,
        cosine * value.first,
        -sine * value.first * value.first + cosine * value.second,
    )


def _cos(context: MPIntervalContext, value: _Jet) -> _Jet:
    sine = context.sin(value.value)
    cosine = context.cos(value.value)
    return _Jet(
        cosine,
        -sine * value.first,
        -cosine * value.first * value.first - sine * value.second,
    )


def _binary64_array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise IntervalKinematicsError(
            "implicit-root identity arrays must be finite binary64"
        )
    return {
        "shape": list(array.shape),
        "values_binary64_hex": [
            float(item).hex() for item in array.ravel(order="C")
        ],
    }


def _optional_binary64(value: float | None) -> str | None:
    return None if value is None else float(value).hex()


def _joint_identity(joint: JointSpec) -> dict[str, object]:
    limit = None
    if joint.limit is not None:
        limit = {
            "lower_binary64_hex": float(joint.limit.lower).hex(),
            "upper_binary64_hex": float(joint.limit.upper).hex(),
            "effort_binary64_hex": _optional_binary64(joint.limit.effort),
            "velocity_binary64_hex": _optional_binary64(joint.limit.velocity),
        }
    mimic = None
    if joint.mimic is not None:
        mimic = {
            "source_joint": joint.mimic.source_joint,
            "multiplier_binary64_hex": float(joint.mimic.multiplier).hex(),
            "offset_binary64_hex": float(joint.mimic.offset).hex(),
        }
    return {
        "name": joint.name,
        "joint_type": joint.joint_type,
        "parent_link": joint.parent_link,
        "child_link": joint.child_link,
        "origin_xyz_m": _binary64_array_identity(joint.origin_xyz_m),
        "origin_rpy_rad": _binary64_array_identity(joint.origin_rpy_rad),
        "axis": _binary64_array_identity(joint.axis),
        "limit": limit,
        "mimic": mimic,
    }


def _canonical_sha256(document: dict[str, object]) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _implicit_root_identities(
    *,
    hand_model: ThreeFingerHandModel,
    link_name: str,
    q_start: np.ndarray,
    direction: np.ndarray,
    base_transform: np.ndarray,
    witness_point_local_m: np.ndarray,
    pad_triangle_local_m: np.ndarray,
    object_triangle_m: np.ndarray,
) -> tuple[str, str]:
    feature_document: dict[str, object] = {
        "schema": IMPLICIT_ROOT_FEATURE_TYPE,
        "link_name": link_name,
        "witness_point_local_m": _binary64_array_identity(
            witness_point_local_m
        ),
        "pad_triangle_local_m": _binary64_array_identity(
            pad_triangle_local_m
        ),
        "object_triangle_m": _binary64_array_identity(object_triangle_m),
    }
    feature_identity_sha256 = _canonical_sha256(feature_document)
    equation_document: dict[str, object] = {
        "schema": IMPLICIT_ROOT_METHOD_ID,
        "interval_kinematics_method_id": METHOD_ID,
        "feature_identity_sha256": feature_identity_sha256,
        "hand": {
            "base_link": hand_model.base_link,
            "joint_order": list(hand_model.joint_order),
            "independent_joint_names": list(
                hand_model.independent_joint_names
            ),
            "joints": [
                _joint_identity(hand_model.joints[name])
                for name in hand_model.joint_order
            ],
        },
        "q_start": _binary64_array_identity(q_start),
        "direction": _binary64_array_identity(direction),
        "base_transform": _binary64_array_identity(base_transform),
    }
    return _canonical_sha256(equation_document), feature_identity_sha256


class DirectedIntervalKinematics:
    """Second-order interval FK and contact predicates for one hand model."""

    def __init__(
        self,
        hand_model: ThreeFingerHandModel,
        options: IntervalArithmeticOptions,
    ) -> None:
        if not isinstance(hand_model, ThreeFingerHandModel):
            raise IntervalKinematicsError(
                "interval kinematics requires ThreeFingerHandModel"
            )
        if not isinstance(options, IntervalArithmeticOptions):
            raise IntervalKinematicsError(
                "interval arithmetic options must be explicit"
            )
        context = MPIntervalContext()
        context.dps = options.decimal_precision
        for joint in hand_model.joints.values():
            if not joint.movable:
                continue
            axis = np.asarray(joint.axis, dtype=np.float64)
            squared_norm = math.fsum(float(value * value) for value in axis)
            structure_bound = _BINARY64_STRUCTURE_GAMMA * max(
                1.0, abs(squared_norm)
            )
            if (
                axis.shape != (3,)
                or not np.all(np.isfinite(axis))
                or abs(squared_norm - 1.0) > structure_bound
            ):
                raise IntervalKinematicsError(
                    f"joint {joint.name} axis is not a binary64 unit vector"
                )
        self.hand_model = hand_model
        self.options = options
        self.context = context
        self._link_transform_cache_owner = object()
        self._identity_cache: tuple[tuple[_Jet, ...], ...] | None = None
        self._value_identity_cache: tuple[tuple[object, ...], ...] | None = None
        self._base_transform_cache: dict[
            bytes, tuple[tuple[_Jet, ...], ...]
        ] = {}
        self._value_base_transform_cache: dict[
            bytes, tuple[tuple[object, ...], ...]
        ] = {}
        self._origin_transform_cache: dict[
            str, tuple[tuple[_Jet, ...], ...]
        ] = {}
        self._value_origin_transform_cache: dict[
            str, tuple[tuple[object, ...], ...]
        ] = {}
        self._ancestor_joint_name_cache: dict[str, tuple[str, ...]] = {}
        affine_cache: dict[str, tuple[str, float, float]] = {}
        independent_index = {
            name: index
            for index, name in enumerate(
                self.hand_model.independent_joint_names
            )
        }
        compiled_plans: dict[str, tuple[int, float, float]] = {}
        for name in self.hand_model.joint_order:
            joint = self.hand_model.joints[name]
            if not joint.movable:
                continue
            source, multiplier, offset = self._affine_map(
                name, affine_cache, set()
            )
            try:
                source_index = independent_index[source]
            except KeyError as error:  # pragma: no cover - model validates this
                raise IntervalKinematicsError(
                    f"mimic source {source} is not an independent joint"
                ) from error
            compiled_plans[name] = (source_index, multiplier, offset)
        self._compiled_joint_affine_plans = compiled_plans
        self._compiled_point_plane_link_plan_cache: dict[
            str, _CompiledPointPlaneLinkPlan
        ] = {}
        self._compiled_point_plane_binding_cache: dict[
            tuple[str, bytes, bytes, bytes, bytes],
            _CompiledPointPlaneBinding,
        ] = {}
        self.compiled_point_backend_status = "NOT_REQUESTED"
        self.compiled_point_backend_failure_reason: str | None = None
        self.compiled_point_evaluation_count = 0
        self.compiled_root_transaction_count = 0
        self.compiled_interval_position_evaluation_count = 0
        self.compiled_point_binding_cache_hit_count = 0
        self.compiled_point_binding_cache_miss_count = 0
        self.compiled_point_binding_triangle_rebind_count = 0
        self.compiled_point_binding_cache_eviction_count = 0
        for joint in self.hand_model.joints.values():
            self._origin_transform(joint)
            self._value_origin_transform(joint)

    def new_link_transform_cache(self) -> IntervalLinkTransformCache:
        """Return one empty cache that cannot be shared with another backend."""

        return IntervalLinkTransformCache(self._link_transform_cache_owner)

    def _interval(self, lower: float, upper: float | None = None) -> object:
        upper_value = lower if upper is None else upper
        if (
            not math.isfinite(float(lower))
            or not math.isfinite(float(upper_value))
            or float(lower) > float(upper_value)
        ):
            raise IntervalKinematicsError(
                "interval endpoints must be finite and ordered"
            )
        return self.context.mpf([float(lower), float(upper_value)])

    def _constant(self, value: float) -> _Jet:
        interval = self._interval(float(value))
        zero = self._interval(0.0)
        return _Jet(interval, zero, zero)

    def _phase(self, lower: float, upper: float) -> _Jet:
        return _Jet(
            self._interval(lower, upper),
            self._interval(1.0),
            self._interval(0.0),
        )

    def _bounded_constant(self, bounds: IntervalBounds) -> _Jet:
        if not isinstance(bounds, IntervalBounds):
            raise IntervalKinematicsError(
                "joint and point boxes must contain IntervalBounds"
            )
        zero = self._interval(0.0)
        return _Jet(self._interval(bounds.lower, bounds.upper), zero, zero)

    def _identity(self) -> list[list[_Jet]]:
        if self._identity_cache is None:
            self._identity_cache = tuple(
                tuple(
                    self._constant(1.0 if row == column else 0.0)
                    for column in range(4)
                )
                for row in range(4)
            )
        return [list(row) for row in self._identity_cache]

    def _validated_base_transform(self, value: np.ndarray) -> np.ndarray:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise IntervalKinematicsError(
                "base transform must be a finite 4x4 matrix"
            )
        if not np.array_equal(
            matrix[3], np.asarray((0.0, 0.0, 0.0, 1.0))
        ):
            raise IntervalKinematicsError(
                "base transform must have an exact homogeneous final row"
            )
        determinant = float(np.linalg.det(matrix[:3, :3]))
        if not math.isfinite(determinant) or determinant <= 0.0:
            raise IntervalKinematicsError(
                "base transform linear part must preserve orientation"
            )
        rotation = matrix[:3, :3]
        gram = rotation.T @ rotation
        structure_scale = max(
            1.0,
            float(np.linalg.norm(rotation, ord=np.inf)) ** 2,
        )
        structure_bound = _BINARY64_STRUCTURE_GAMMA * structure_scale
        if (
            not np.all(np.isfinite(gram))
            or np.max(np.abs(gram - np.eye(3))) > structure_bound
            or abs(determinant - 1.0) > structure_bound
        ):
            raise IntervalKinematicsError(
                "base transform linear part is not a binary64 proper rotation"
            )
        return matrix

    def _matrix_from_float(self, value: np.ndarray) -> list[list[_Jet]]:
        candidate = np.asarray(value, dtype=np.float64)
        if candidate.shape == (4, 4) and np.all(np.isfinite(candidate)):
            key = np.asarray(candidate, dtype=">f8").tobytes(order="C")
            cached = self._base_transform_cache.get(key)
            if cached is not None:
                return [list(row) for row in cached]
        matrix = self._validated_base_transform(candidate)
        key = np.asarray(matrix, dtype=">f8").tobytes(order="C")
        result = [
            [self._constant(float(matrix[row, column])) for column in range(4)]
            for row in range(4)
        ]
        self._base_transform_cache[key] = tuple(
            tuple(row) for row in result
        )
        return result

    def _value_identity(self) -> list[list[object]]:
        if self._value_identity_cache is None:
            self._value_identity_cache = tuple(
                tuple(
                    self._interval(1.0 if row == column else 0.0)
                    for column in range(4)
                )
                for row in range(4)
            )
        return [list(row) for row in self._value_identity_cache]

    def _value_matrix_from_float(
        self, value: np.ndarray
    ) -> list[list[object]]:
        candidate = np.asarray(value, dtype=np.float64)
        if candidate.shape == (4, 4) and np.all(np.isfinite(candidate)):
            key = np.asarray(candidate, dtype=">f8").tobytes(order="C")
            cached = self._value_base_transform_cache.get(key)
            if cached is not None:
                return [list(row) for row in cached]
        matrix = self._validated_base_transform(candidate)
        key = np.asarray(matrix, dtype=">f8").tobytes(order="C")
        result = [
            [
                self._interval(float(matrix[row, column]))
                for column in range(4)
            ]
            for row in range(4)
        ]
        self._value_base_transform_cache[key] = tuple(
            tuple(row) for row in result
        )
        return result

    def _value_matmul(
        self,
        first: list[list[object]],
        second: list[list[object]],
    ) -> list[list[object]]:
        result = self._value_identity()
        for row in range(3):
            for column in range(3):
                value = self._interval(0.0)
                for inner in range(3):
                    value = (
                        value
                        + first[row][inner] * second[inner][column]
                    )
                result[row][column] = value
            translation = self._interval(0.0)
            for inner in range(3):
                translation = (
                    translation
                    + first[row][inner] * second[inner][3]
                )
            result[row][3] = translation + first[row][3]
        return result

    def _matmul(
        self, first: list[list[_Jet]], second: list[list[_Jet]]
    ) -> list[list[_Jet]]:
        result = self._identity()
        for row in range(3):
            for column in range(3):
                value = self._constant(0.0)
                for inner in range(3):
                    value = _add(
                        value, _mul(first[row][inner], second[inner][column])
                    )
                result[row][column] = value
            translation = self._constant(0.0)
            for inner in range(3):
                translation = _add(
                    translation,
                    _mul(first[row][inner], second[inner][3]),
                )
            result[row][3] = _add(translation, first[row][3])
        return result

    def _rpy_rotation(
        self, rpy_rad: Sequence[float]
    ) -> list[list[_Jet]]:
        roll, pitch, yaw = (self._constant(float(value)) for value in rpy_rad)
        cr, sr = _cos(self.context, roll), _sin(self.context, roll)
        cp, sp = _cos(self.context, pitch), _sin(self.context, pitch)
        cy, sy = _cos(self.context, yaw), _sin(self.context, yaw)
        return [
            [
                _mul(cy, cp),
                _sub(_mul(_mul(cy, sp), sr), _mul(sy, cr)),
                _add(_mul(_mul(cy, sp), cr), _mul(sy, sr)),
            ],
            [
                _mul(sy, cp),
                _add(_mul(_mul(sy, sp), sr), _mul(cy, cr)),
                _sub(_mul(_mul(sy, sp), cr), _mul(cy, sr)),
            ],
            [_neg(sp), _mul(cp, sr), _mul(cp, cr)],
        ]

    def _origin_transform(self, joint: JointSpec) -> list[list[_Jet]]:
        cached = self._origin_transform_cache.get(joint.name)
        if cached is not None:
            return [list(row) for row in cached]
        result = self._identity()
        rotation = self._rpy_rotation(joint.origin_rpy_rad)
        for row in range(3):
            for column in range(3):
                result[row][column] = rotation[row][column]
            result[row][3] = self._constant(joint.origin_xyz_m[row])
        self._origin_transform_cache[joint.name] = tuple(
            tuple(row) for row in result
        )
        return result

    def _axis_rotation(
        self, axis: Sequence[float], angle: _Jet
    ) -> list[list[_Jet]]:
        axis_tuple = tuple(float(value) for value in axis)
        cardinal = {
            (1.0, 0.0, 0.0): (0, 1.0),
            (-1.0, 0.0, 0.0): (0, -1.0),
            (0.0, 1.0, 0.0): (1, 1.0),
            (0.0, -1.0, 0.0): (1, -1.0),
            (0.0, 0.0, 1.0): (2, 1.0),
            (0.0, 0.0, -1.0): (2, -1.0),
        }.get(axis_tuple)
        if cardinal is not None:
            axis_index, sign = cardinal
            sine = _sin(self.context, angle)
            if sign < 0.0:
                sine = _neg(sine)
            cosine = _cos(self.context, angle)
            zero = self._constant(0.0)
            one = self._constant(1.0)
            if axis_index == 0:
                return [
                    [one, zero, zero],
                    [zero, cosine, _neg(sine)],
                    [zero, sine, cosine],
                ]
            if axis_index == 1:
                return [
                    [cosine, zero, sine],
                    [zero, one, zero],
                    [_neg(sine), zero, cosine],
                ]
            return [
                [cosine, _neg(sine), zero],
                [sine, cosine, zero],
                [zero, zero, one],
            ]
        x, y, z = (self._constant(float(value)) for value in axis)
        zero = self._constant(0.0)
        skew = [
            [zero, _neg(z), y],
            [z, zero, _neg(x)],
            [_neg(y), x, zero],
        ]
        skew_squared: list[list[_Jet]] = []
        for row in range(3):
            output_row: list[_Jet] = []
            for column in range(3):
                value = self._constant(0.0)
                for inner in range(3):
                    value = _add(
                        value, _mul(skew[row][inner], skew[inner][column])
                    )
                output_row.append(value)
            skew_squared.append(output_row)
        sine = _sin(self.context, angle)
        one_minus_cosine = _sub(self._constant(1.0), _cos(self.context, angle))
        rotation: list[list[_Jet]] = []
        for row in range(3):
            output_row = []
            for column in range(3):
                value = self._constant(1.0 if row == column else 0.0)
                value = _add(value, _mul(sine, skew[row][column]))
                value = _add(
                    value,
                    _mul(one_minus_cosine, skew_squared[row][column]),
                )
                output_row.append(value)
            rotation.append(output_row)
        return rotation

    def _motion_transform(
        self, joint: JointSpec, position: _Jet
    ) -> list[list[_Jet]]:
        result = self._identity()
        if joint.joint_type in ("revolute", "continuous"):
            rotation = self._axis_rotation(joint.axis, position)
            for row in range(3):
                for column in range(3):
                    result[row][column] = rotation[row][column]
        elif joint.joint_type == "prismatic":
            for row in range(3):
                result[row][3] = _mul(
                    self._constant(joint.axis[row]), position
                )
        elif joint.joint_type != "fixed":
            raise IntervalKinematicsError(
                f"unsupported joint type for interval FK: {joint.joint_type}"
            )
        return result

    def _apply_joint_step(
        self,
        transform: list[list[_Jet]],
        joint: JointSpec,
        position: _Jet,
    ) -> list[list[_Jet]]:
        result = self._matmul(transform, self._origin_transform(joint))
        if joint.joint_type == "fixed":
            return result
        if joint.joint_type in ("revolute", "continuous"):
            rotation = self._axis_rotation(joint.axis, position)
            output = [list(row) for row in result]
            for row in range(3):
                for column in range(3):
                    value = self._constant(0.0)
                    for inner in range(3):
                        value = _add(
                            value,
                            _mul(result[row][inner], rotation[inner][column]),
                        )
                    output[row][column] = value
            return output
        if joint.joint_type == "prismatic":
            output = [list(row) for row in result]
            local_displacement = tuple(
                _mul(self._constant(component), position)
                for component in joint.axis
            )
            for row in range(3):
                translation = result[row][3]
                for inner, component in enumerate(joint.axis):
                    if component == 0.0:
                        continue
                    translation = _add(
                        translation,
                        _mul(result[row][inner], local_displacement[inner]),
                    )
                output[row][3] = translation
            return output
        raise IntervalKinematicsError(
            f"unsupported joint type for interval FK: {joint.joint_type}"
        )

    def _value_rpy_rotation(
        self, rpy_rad: Sequence[float]
    ) -> list[list[object]]:
        roll, pitch, yaw = (
            self._interval(float(value)) for value in rpy_rad
        )
        cr, sr = self.context.cos(roll), self.context.sin(roll)
        cp, sp = self.context.cos(pitch), self.context.sin(pitch)
        cy, sy = self.context.cos(yaw), self.context.sin(yaw)
        return [
            [
                cy * cp,
                (cy * sp) * sr - sy * cr,
                (cy * sp) * cr + sy * sr,
            ],
            [
                sy * cp,
                (sy * sp) * sr + cy * cr,
                (sy * sp) * cr - cy * sr,
            ],
            [-sp, cp * sr, cp * cr],
        ]

    def _value_origin_transform(
        self, joint: JointSpec
    ) -> list[list[object]]:
        cached = self._value_origin_transform_cache.get(joint.name)
        if cached is not None:
            return [list(row) for row in cached]
        result = self._value_identity()
        rotation = self._value_rpy_rotation(joint.origin_rpy_rad)
        for row in range(3):
            for column in range(3):
                result[row][column] = rotation[row][column]
            result[row][3] = self._interval(joint.origin_xyz_m[row])
        self._value_origin_transform_cache[joint.name] = tuple(
            tuple(row) for row in result
        )
        return result

    def _value_axis_rotation(
        self, axis: Sequence[float], angle: object
    ) -> list[list[object]]:
        axis_tuple = tuple(float(value) for value in axis)
        cardinal = {
            (1.0, 0.0, 0.0): (0, 1.0),
            (-1.0, 0.0, 0.0): (0, -1.0),
            (0.0, 1.0, 0.0): (1, 1.0),
            (0.0, -1.0, 0.0): (1, -1.0),
            (0.0, 0.0, 1.0): (2, 1.0),
            (0.0, 0.0, -1.0): (2, -1.0),
        }.get(axis_tuple)
        if cardinal is not None:
            axis_index, sign = cardinal
            sine = self.context.sin(angle)
            if sign < 0.0:
                sine = -sine
            cosine = self.context.cos(angle)
            zero = self._interval(0.0)
            one = self._interval(1.0)
            if axis_index == 0:
                return [
                    [one, zero, zero],
                    [zero, cosine, -sine],
                    [zero, sine, cosine],
                ]
            if axis_index == 1:
                return [
                    [cosine, zero, sine],
                    [zero, one, zero],
                    [-sine, zero, cosine],
                ]
            return [
                [cosine, -sine, zero],
                [sine, cosine, zero],
                [zero, zero, one],
            ]
        x, y, z = (
            self._interval(float(value)) for value in axis
        )
        zero = self._interval(0.0)
        skew = [
            [zero, -z, y],
            [z, zero, -x],
            [-y, x, zero],
        ]
        skew_squared: list[list[object]] = []
        for row in range(3):
            output_row: list[object] = []
            for column in range(3):
                value = self._interval(0.0)
                for inner in range(3):
                    value = (
                        value
                        + skew[row][inner] * skew[inner][column]
                    )
                output_row.append(value)
            skew_squared.append(output_row)
        sine = self.context.sin(angle)
        one_minus_cosine = self._interval(1.0) - self.context.cos(angle)
        rotation: list[list[object]] = []
        for row in range(3):
            output_row = []
            for column in range(3):
                value = self._interval(
                    1.0 if row == column else 0.0
                )
                value = value + sine * skew[row][column]
                value = (
                    value
                    + one_minus_cosine * skew_squared[row][column]
                )
                output_row.append(value)
            rotation.append(output_row)
        return rotation

    def _value_motion_transform(
        self, joint: JointSpec, position: object
    ) -> list[list[object]]:
        result = self._value_identity()
        if joint.joint_type in ("revolute", "continuous"):
            rotation = self._value_axis_rotation(joint.axis, position)
            for row in range(3):
                for column in range(3):
                    result[row][column] = rotation[row][column]
        elif joint.joint_type == "prismatic":
            for row in range(3):
                result[row][3] = (
                    self._interval(joint.axis[row]) * position
                )
        elif joint.joint_type != "fixed":
            raise IntervalKinematicsError(
                f"unsupported joint type for interval FK: {joint.joint_type}"
            )
        return result

    def _value_apply_joint_step(
        self,
        transform: list[list[object]],
        joint: JointSpec,
        position: object,
    ) -> list[list[object]]:
        result = self._value_matmul(
            transform, self._value_origin_transform(joint)
        )
        if joint.joint_type == "fixed":
            return result
        if joint.joint_type in ("revolute", "continuous"):
            rotation = self._value_axis_rotation(joint.axis, position)
            output = [list(row) for row in result]
            for row in range(3):
                for column in range(3):
                    value = self._interval(0.0)
                    for inner in range(3):
                        value = (
                            value
                            + result[row][inner] * rotation[inner][column]
                        )
                    output[row][column] = value
            return output
        if joint.joint_type == "prismatic":
            output = [list(row) for row in result]
            local_displacement = tuple(
                self._interval(component) * position
                for component in joint.axis
            )
            for row in range(3):
                translation = result[row][3]
                for inner, component in enumerate(joint.axis):
                    if component == 0.0:
                        continue
                    translation = (
                        translation
                        + result[row][inner] * local_displacement[inner]
                    )
                output[row][3] = translation
            return output
        raise IntervalKinematicsError(
            f"unsupported joint type for interval FK: {joint.joint_type}"
        )

    def _affine_map(
        self,
        joint_name: str,
        cache: dict[str, tuple[str, float, float]],
        active: set[str],
    ) -> tuple[str, float, float]:
        cached = cache.get(joint_name)
        if cached is not None:
            return cached
        if joint_name in active:
            raise IntervalKinematicsError("cyclic mimic relation")
        active.add(joint_name)
        joint = self.hand_model.joints[joint_name]
        if joint.mimic is None:
            result = (joint_name, 1.0, 0.0)
        else:
            source, multiplier, offset = self._affine_map(
                joint.mimic.source_joint, cache, active
            )
            result = (
                source,
                joint.mimic.multiplier * multiplier,
                joint.mimic.multiplier * offset + joint.mimic.offset,
            )
        active.remove(joint_name)
        cache[joint_name] = result
        return result

    def _ancestor_joint_names(self, link_name: str) -> tuple[str, ...]:
        cached = self._ancestor_joint_name_cache.get(link_name)
        if cached is not None:
            return cached
        by_child = {
            joint.child_link: name
            for name, joint in self.hand_model.joints.items()
        }
        names: list[str] = []
        cursor = link_name
        while cursor != self.hand_model.base_link:
            name = by_child.get(cursor)
            if name is None:
                raise IntervalKinematicsError(
                    f"link {link_name} is disconnected from hand base"
                )
            names.append(name)
            cursor = self.hand_model.joints[name].parent_link
        names.reverse()
        result = tuple(names)
        self._ancestor_joint_name_cache[link_name] = result
        return result

    @staticmethod
    def _directed_transform_cache_key(
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
    ) -> tuple[str, bytes, bytes, bytes, bytes]:
        return (
            str(link_name),
            np.asarray(q_start, dtype=">f8").tobytes(order="C"),
            np.asarray(direction, dtype=">f8").tobytes(order="C"),
            np.asarray(
                (phase_lower, phase_upper), dtype=">f8"
            ).tobytes(order="C"),
            np.asarray(base_transform, dtype=">f8").tobytes(order="C"),
        )

    def _link_transform(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> list[list[_Jet]]:
        q_start = np.asarray(q_start, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        base_transform = np.asarray(base_transform, dtype=np.float64)
        joint_count = len(self.hand_model.independent_joint_names)
        if (
            q_start.shape != (joint_count,)
            or direction.shape != (joint_count,)
            or not np.all(np.isfinite(q_start))
            or not np.all(np.isfinite(direction))
        ):
            raise IntervalKinematicsError(
                "q_start and direction must match independent joints"
            )
        phase = self._phase(phase_lower, phase_upper)
        cache_key: tuple[str, bytes, bytes, bytes, bytes] | None = None
        if transform_cache is not None:
            if (
                type(transform_cache) is not IntervalLinkTransformCache
                or transform_cache._owner_token
                is not self._link_transform_cache_owner
            ):
                raise IntervalKinematicsError(
                    "link transform cache belongs to another interval backend"
                )
            cache_key = self._directed_transform_cache_key(
                link_name=link_name,
                q_start=q_start,
                direction=direction,
                phase_lower=phase_lower,
                phase_upper=phase_upper,
                base_transform=base_transform,
            )
            if transform_cache._primary_phase_key is None:
                transform_cache._primary_phase_key = cache_key[3]
            cached = transform_cache._entries.get(cache_key)
            if cached is not None:
                transform_cache.hit_count += 1
                return [list(row) for row in cached]
            transform_cache.miss_count += 1
        independent: list[_Jet] = []
        for start, rate in zip(q_start, direction):
            independent.append(_add(
                self._constant(float(start)),
                _mul(self._constant(float(rate)), phase),
            ))
        for name in self.hand_model.joint_order:
            joint = self.hand_model.joints[name]
            if not joint.movable:
                continue
            source_index, multiplier, offset = (
                self._compiled_joint_affine_plans[name]
            )
            position = _add(
                _mul(
                    self._constant(multiplier),
                    independent[source_index],
                ),
                self._constant(offset),
            )
            assert joint.limit is not None
            lower_limit = self.context.mpf(joint.limit.lower)
            upper_limit = self.context.mpf(joint.limit.upper)
            if (
                position.value.a < lower_limit
                or position.value.b > upper_limit
            ):
                raise IntervalKinematicsError(
                    "interval joint path violates the URDF limit contract"
                )
        transform = self._matrix_from_float(base_transform)
        for name in self._ancestor_joint_names(link_name):
            joint = self.hand_model.joints[name]
            if joint.movable:
                source_index, multiplier, offset = (
                    self._compiled_joint_affine_plans[name]
                )
                position = _add(
                    _mul(
                        self._constant(multiplier),
                        independent[source_index],
                    ),
                    self._constant(offset),
                )
            else:
                position = self._constant(0.0)
            transform = self._apply_joint_step(
                transform, joint, position
            )
        if transform_cache is not None:
            assert cache_key is not None
            evicted = transform_cache._bounded_store(
                transform_cache._entries,
                cache_key,
                tuple(tuple(row) for row in transform),
                MULTIPHASE_TRANSFORM_CACHE_CAPACITY,
            )
            transform_cache.transform_eviction_count += int(evicted)
            transform_cache._value_entries.pop(cache_key, None)
        return transform

    def _value_link_transform(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> list[list[object]]:
        q_start = np.asarray(q_start, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        base_transform = np.asarray(base_transform, dtype=np.float64)
        joint_count = len(self.hand_model.independent_joint_names)
        if (
            q_start.shape != (joint_count,)
            or direction.shape != (joint_count,)
            or not np.all(np.isfinite(q_start))
            or not np.all(np.isfinite(direction))
        ):
            raise IntervalKinematicsError(
                "q_start and direction must match independent joints"
            )
        phase = self._interval(phase_lower, phase_upper)
        cache_key: tuple[str, bytes, bytes, bytes, bytes] | None = None
        if transform_cache is not None:
            if (
                type(transform_cache) is not IntervalLinkTransformCache
                or transform_cache._owner_token
                is not self._link_transform_cache_owner
            ):
                raise IntervalKinematicsError(
                    "link transform cache belongs to another interval backend"
                )
            cache_key = self._directed_transform_cache_key(
                link_name=link_name,
                q_start=q_start,
                direction=direction,
                phase_lower=phase_lower,
                phase_upper=phase_upper,
                base_transform=base_transform,
            )
            if transform_cache._primary_phase_key is None:
                transform_cache._primary_phase_key = cache_key[3]
            cached_value = transform_cache._value_entries.get(cache_key)
            if cached_value is not None:
                transform_cache.hit_count += 1
                return [list(row) for row in cached_value]
            cached_jet = transform_cache._entries.get(cache_key)
            if cached_jet is not None:
                transform_cache.hit_count += 1
                return [
                    [component.value for component in row]
                    for row in cached_jet
                ]
            transform_cache.miss_count += 1
        independent: list[object] = []
        for start, rate in zip(q_start, direction):
            independent.append(
                self._interval(float(start))
                + self._interval(float(rate)) * phase
            )
        for name in self.hand_model.joint_order:
            joint = self.hand_model.joints[name]
            if not joint.movable:
                continue
            source_index, multiplier, offset = (
                self._compiled_joint_affine_plans[name]
            )
            position = (
                self._interval(multiplier) * independent[source_index]
                + self._interval(offset)
            )
            assert joint.limit is not None
            if (
                position.a < self.context.mpf(joint.limit.lower)
                or position.b > self.context.mpf(joint.limit.upper)
            ):
                raise IntervalKinematicsError(
                    "interval joint path violates the URDF limit contract"
                )
        transform = self._value_matrix_from_float(base_transform)
        for name in self._ancestor_joint_names(link_name):
            joint = self.hand_model.joints[name]
            if joint.movable:
                source_index, multiplier, offset = (
                    self._compiled_joint_affine_plans[name]
                )
                position = (
                    self._interval(multiplier)
                    * independent[source_index]
                    + self._interval(offset)
                )
            else:
                position = self._interval(0.0)
            transform = self._value_apply_joint_step(
                transform, joint, position
            )
        if transform_cache is not None:
            assert cache_key is not None
            evicted = transform_cache._bounded_store(
                transform_cache._value_entries,
                cache_key,
                tuple(tuple(row) for row in transform),
                MULTIPHASE_TRANSFORM_CACHE_CAPACITY,
            )
            transform_cache.transform_eviction_count += int(evicted)
        return transform

    def _link_transform_over_joint_box(
        self,
        *,
        link_name: str,
        independent_joint_intervals: Sequence[IntervalBounds],
        base_transform: np.ndarray,
    ) -> list[list[_Jet]]:
        joint_names = self.hand_model.independent_joint_names
        intervals = tuple(independent_joint_intervals)
        if len(intervals) != len(joint_names) or not all(
            isinstance(bounds, IntervalBounds) for bounds in intervals
        ):
            raise IntervalKinematicsError(
                "joint interval box must match independent joints"
            )
        independent = {
            name: self._bounded_constant(bounds)
            for name, bounds in zip(joint_names, intervals)
        }
        affine_cache: dict[str, tuple[str, float, float]] = {}
        resolved: dict[str, _Jet] = {}
        for name in self.hand_model.joint_order:
            joint = self.hand_model.joints[name]
            if not joint.movable:
                continue
            source, multiplier, offset = self._affine_map(
                name, affine_cache, set()
            )
            try:
                source_position = independent[source]
            except KeyError as error:
                raise IntervalKinematicsError(
                    f"mimic source {source} is not an independent joint"
                ) from error
            position = _add(
                _mul(self._constant(multiplier), source_position),
                self._constant(offset),
            )
            assert joint.limit is not None
            if (
                position.value.a < self.context.mpf(joint.limit.lower)
                or position.value.b > self.context.mpf(joint.limit.upper)
            ):
                raise IntervalKinematicsError(
                    "interval joint box violates the URDF limit contract"
                )
            resolved[name] = position

        transform = self._matrix_from_float(base_transform)
        for name in self._ancestor_joint_names(link_name):
            joint = self.hand_model.joints[name]
            position = (
                resolved[name] if joint.movable else self._constant(0.0)
            )
            transform = self._apply_joint_step(
                transform, joint, position
            )
        return transform

    def _point_jet(
        self, transform: list[list[_Jet]], point_local_m: np.ndarray
    ) -> tuple[_Jet, _Jet, _Jet]:
        point = np.asarray(point_local_m, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise IntervalKinematicsError(
                "local witness point must be a finite three-vector"
            )
        result: list[_Jet] = []
        for row in range(3):
            value = transform[row][3]
            for column in range(3):
                value = _add(
                    value,
                    _mul(
                        transform[row][column],
                        self._constant(float(point[column])),
                    ),
                )
            result.append(value)
        return result[0], result[1], result[2]

    def _value_point(
        self,
        transform: list[list[object]],
        point_local_m: np.ndarray,
    ) -> tuple[object, object, object]:
        point = np.asarray(point_local_m, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise IntervalKinematicsError(
                "local witness point must be a finite three-vector"
            )
        result: list[object] = []
        for row in range(3):
            value = transform[row][3]
            for column in range(3):
                value = (
                    value
                    + transform[row][column]
                    * self._interval(float(point[column]))
                )
            result.append(value)
        return result[0], result[1], result[2]

    @staticmethod
    def _value_cross(
        first: Sequence[object], second: Sequence[object]
    ) -> tuple[object, object, object]:
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    def _value_dot(
        self, first: Sequence[object], second: Sequence[object]
    ) -> object:
        value = self._interval(0.0)
        for first_value, second_value in zip(first, second):
            value = value + first_value * second_value
        return value

    def _value_point_with_cache(
        self,
        *,
        transform: list[list[object]],
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        point_local_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None,
    ) -> tuple[object, object, object]:
        point = np.asarray(point_local_m, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise IntervalKinematicsError(
                "local witness point must be a finite three-vector"
            )
        if transform_cache is None:
            return self._value_point(transform, point)
        if (
            type(transform_cache) is not IntervalLinkTransformCache
            or transform_cache._owner_token
            is not self._link_transform_cache_owner
        ):
            raise IntervalKinematicsError(
                "link transform cache belongs to another interval backend"
            )
        transform_key = self._directed_transform_cache_key(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
        )
        key = (
            transform_key,
            np.asarray(point, dtype=">f8").tobytes(order="C"),
        )
        cached_value = transform_cache._value_point_entries.get(key)
        if cached_value is not None:
            transform_cache.point_hit_count += 1
            return cached_value
        cached_jet = transform_cache._point_entries.get(key)
        if cached_jet is not None:
            transform_cache.point_hit_count += 1
            return tuple(component.value for component in cached_jet)  # type: ignore[return-value]
        transform_cache.point_miss_count += 1
        result = self._value_point(transform, point)
        evicted = transform_cache._bounded_store(
            transform_cache._value_point_entries,
            key,
            result,
            MULTIPHASE_POINT_CACHE_CAPACITY,
        )
        transform_cache.point_eviction_count += int(evicted)
        return result

    def _point_jet_with_cache(
        self,
        *,
        transform: list[list[_Jet]],
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        point_local_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None,
    ) -> tuple[_Jet, _Jet, _Jet]:
        point = np.asarray(point_local_m, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise IntervalKinematicsError(
                "local witness point must be a finite three-vector"
            )
        if transform_cache is None:
            return self._point_jet(transform, point)
        if (
            type(transform_cache) is not IntervalLinkTransformCache
            or transform_cache._owner_token
            is not self._link_transform_cache_owner
        ):
            raise IntervalKinematicsError(
                "link transform cache belongs to another interval backend"
            )
        transform_key = self._directed_transform_cache_key(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
        )
        key = (
            transform_key,
            np.asarray(point, dtype=">f8").tobytes(order="C"),
        )
        cached = transform_cache._point_entries.get(key)
        if cached is not None:
            transform_cache.point_hit_count += 1
            return cached
        transform_cache.point_miss_count += 1
        result = self._point_jet(transform, point)
        evicted = transform_cache._bounded_store(
            transform_cache._point_entries,
            key,
            result,
            MULTIPHASE_POINT_CACHE_CAPACITY,
        )
        transform_cache.point_eviction_count += int(evicted)
        transform_cache._value_point_entries.pop(key, None)
        return result

    def _vector_jet(
        self, transform: list[list[_Jet]], vector_local: Sequence[_Jet]
    ) -> tuple[_Jet, _Jet, _Jet]:
        result: list[_Jet] = []
        for row in range(3):
            value = self._constant(0.0)
            for column in range(3):
                value = _add(
                    value,
                    _mul(transform[row][column], vector_local[column]),
                )
            result.append(value)
        return result[0], result[1], result[2]

    def _pad_area_jet_with_cache(
        self,
        *,
        transform: list[list[_Jet]],
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        pad_triangle_local_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None,
    ) -> tuple[_Jet, _Jet, _Jet]:
        pad_triangle = np.asarray(pad_triangle_local_m, dtype=np.float64)
        if pad_triangle.shape != (3, 3) or not np.all(
            np.isfinite(pad_triangle)
        ):
            raise IntervalKinematicsError(
                "contact predicate triangles must be finite shape (3, 3)"
            )

        def compute() -> tuple[_Jet, _Jet, _Jet]:
            pad_vertices = tuple(
                self._constant_vector(row) for row in pad_triangle
            )
            pad_edge_one = tuple(
                _sub(second, first)
                for first, second in zip(
                    pad_vertices[0], pad_vertices[1]
                )
            )
            pad_edge_two = tuple(
                _sub(third, first)
                for first, third in zip(
                    pad_vertices[0], pad_vertices[2]
                )
            )
            return self._cross(
                self._vector_jet(transform, pad_edge_one),
                self._vector_jet(transform, pad_edge_two),
            )

        if transform_cache is None:
            return compute()
        if (
            type(transform_cache) is not IntervalLinkTransformCache
            or transform_cache._owner_token
            is not self._link_transform_cache_owner
        ):
            raise IntervalKinematicsError(
                "link transform cache belongs to another interval backend"
            )
        transform_key = self._directed_transform_cache_key(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
        )
        key = (
            transform_key,
            np.asarray(pad_triangle, dtype=">f8").tobytes(order="C"),
        )
        cached = transform_cache._pad_area_entries.get(key)
        if cached is not None:
            transform_cache.pad_area_hit_count += 1
            return cached
        transform_cache.pad_area_miss_count += 1
        result = compute()
        evicted = transform_cache._bounded_store(
            transform_cache._pad_area_entries,
            key,
            result,
            MULTIPHASE_PAD_AREA_CACHE_CAPACITY,
        )
        transform_cache.pad_area_eviction_count += int(evicted)
        return result

    @staticmethod
    def _cross(
        first: Sequence[_Jet], second: Sequence[_Jet]
    ) -> tuple[_Jet, _Jet, _Jet]:
        return (
            _sub(_mul(first[1], second[2]), _mul(first[2], second[1])),
            _sub(_mul(first[2], second[0]), _mul(first[0], second[2])),
            _sub(_mul(first[0], second[1]), _mul(first[1], second[0])),
        )

    def _dot(self, first: Sequence[_Jet], second: Sequence[_Jet]) -> _Jet:
        value = self._constant(0.0)
        for first_value, second_value in zip(first, second):
            value = _add(value, _mul(first_value, second_value))
        return value

    def _constant_vector(
        self, values: Sequence[float]
    ) -> tuple[_Jet, _Jet, _Jet]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise IntervalKinematicsError(
                "interval vector input must be finite shape (3,)"
            )
        return tuple(
            self._constant(float(value)) for value in array
        )  # type: ignore[return-value]

    def _bounds(self, value: object) -> IntervalBounds:
        lower = float(value.a)
        upper = float(value.b)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise IntervalKinematicsError(
                "interval arithmetic produced a non-finite enclosure"
            )
        return IntervalBounds(
            float(np.nextafter(lower, -math.inf)),
            float(np.nextafter(upper, math.inf)),
        )

    @staticmethod
    def _outward_affine_points_many(
        *,
        coefficient_lower: np.ndarray,
        coefficient_upper: np.ndarray,
        points_local_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one interval 3x4 affine map for many exact points.

        Every binary64 multiplication and accumulation is expanded by one
        representable value in the required direction.  The result therefore
        encloses evaluating the same high-precision interval coefficients for
        every supplied point, while avoiding one Python/mpmath loop per point.
        """

        coefficient_lower = np.asarray(
            coefficient_lower, dtype=np.float64
        )
        coefficient_upper = np.asarray(
            coefficient_upper, dtype=np.float64
        )
        points = np.asarray(points_local_m, dtype=np.float64)
        if (
            coefficient_lower.shape != (3, 4)
            or coefficient_upper.shape != (3, 4)
            or points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) == 0
            or not np.all(np.isfinite(coefficient_lower))
            or not np.all(np.isfinite(coefficient_upper))
            or np.any(coefficient_lower > coefficient_upper)
            or not np.all(np.isfinite(points))
        ):
            raise IntervalKinematicsError(
                "batch affine point inputs are malformed"
            )
        homogeneous = np.concatenate(
            (points, np.ones((len(points), 1), dtype=np.float64)),
            axis=1,
        )
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            first_products = (
                homogeneous[:, None, :] * coefficient_lower[None, :, :]
            )
            second_products = (
                homogeneous[:, None, :] * coefficient_upper[None, :, :]
            )
            product_lower = np.nextafter(
                np.minimum(first_products, second_products), -math.inf
            )
            product_upper = np.nextafter(
                np.maximum(first_products, second_products), math.inf
            )
            lower = np.zeros((len(points), 3), dtype=np.float64)
            upper = np.zeros((len(points), 3), dtype=np.float64)
            for column in range(4):
                lower = np.nextafter(
                    lower + product_lower[:, :, column], -math.inf
                )
                upper = np.nextafter(
                    upper + product_upper[:, :, column], math.inf
                )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise IntervalKinematicsError(
                "batch affine point evaluation produced non-finite bounds"
            )
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    @staticmethod
    def _outward_linear_vectors_many(
        *,
        coefficient_lower: np.ndarray,
        coefficient_upper: np.ndarray,
        vectors_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one interval 3x3 linear map for many exact vectors."""

        coefficient_lower = np.asarray(
            coefficient_lower, dtype=np.float64
        )
        coefficient_upper = np.asarray(
            coefficient_upper, dtype=np.float64
        )
        vectors = np.asarray(vectors_local, dtype=np.float64)
        if (
            coefficient_lower.shape != (3, 3)
            or coefficient_upper.shape != (3, 3)
            or vectors.ndim != 2
            or vectors.shape[1:] != (3,)
            or len(vectors) == 0
            or not np.all(np.isfinite(coefficient_lower))
            or not np.all(np.isfinite(coefficient_upper))
            or np.any(coefficient_lower > coefficient_upper)
            or not np.all(np.isfinite(vectors))
        ):
            raise IntervalKinematicsError(
                "batch linear vector inputs are malformed"
            )
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            first_products = (
                vectors[:, None, :] * coefficient_lower[None, :, :]
            )
            second_products = (
                vectors[:, None, :] * coefficient_upper[None, :, :]
            )
            product_lower = np.nextafter(
                np.minimum(first_products, second_products), -math.inf
            )
            product_upper = np.nextafter(
                np.maximum(first_products, second_products), math.inf
            )
            lower = np.zeros((len(vectors), 3), dtype=np.float64)
            upper = np.zeros((len(vectors), 3), dtype=np.float64)
            for column in range(3):
                lower = np.nextafter(
                    lower + product_lower[:, :, column], -math.inf
                )
                upper = np.nextafter(
                    upper + product_upper[:, :, column], math.inf
                )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise IntervalKinematicsError(
                "batch linear vector evaluation produced non-finite bounds"
            )
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def point_velocity_and_vector_many(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        points_local_m: np.ndarray,
        vectors_local: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> IntervalPointVelocityVectorBatch:
        """Enclose attached-point velocities and rotated local vectors."""

        points = np.asarray(points_local_m, dtype=np.float64)
        vectors = np.asarray(vectors_local, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or vectors.shape != points.shape
            or len(points) == 0
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(vectors))
        ):
            raise IntervalKinematicsError(
                "batch point velocity/vector inputs must be aligned finite (N, 3)"
            )
        transform = self._link_transform(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            transform_cache=transform_cache,
        )
        velocity_lower = np.empty((3, 4), dtype=np.float64)
        velocity_upper = np.empty((3, 4), dtype=np.float64)
        rotation_lower = np.empty((3, 3), dtype=np.float64)
        rotation_upper = np.empty((3, 3), dtype=np.float64)
        for row in range(3):
            for column in range(4):
                bounds = self._bounds(transform[row][column].first)
                velocity_lower[row, column] = bounds.lower
                velocity_upper[row, column] = bounds.upper
                if column < 3:
                    bounds = self._bounds(transform[row][column].value)
                    rotation_lower[row, column] = bounds.lower
                    rotation_upper[row, column] = bounds.upper
        point_velocity_lower, point_velocity_upper = (
            self._outward_affine_points_many(
                coefficient_lower=velocity_lower,
                coefficient_upper=velocity_upper,
                points_local_m=points,
            )
        )
        vector_lower, vector_upper = self._outward_linear_vectors_many(
            coefficient_lower=rotation_lower,
            coefficient_upper=rotation_upper,
            vectors_local=vectors,
        )
        return IntervalPointVelocityVectorBatch(
            phase=IntervalBounds(phase_lower, phase_upper),
            point_velocity_lower_object_m_per_unit=point_velocity_lower,
            point_velocity_upper_object_m_per_unit=point_velocity_upper,
            vector_lower_object=vector_lower,
            vector_upper_object=vector_upper,
            method_id=BATCH_POINT_VELOCITY_VECTOR_METHOD_ID,
            decimal_precision=self.options.decimal_precision,
        )

    def point_motion_many(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        points_local_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> IntervalPointMotionBatch:
        """Enclose many rigid-link point paths after one interval FK."""

        points = np.asarray(points_local_m, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) == 0
            or not np.all(np.isfinite(points))
        ):
            raise IntervalKinematicsError(
                "local witness points must have finite non-empty shape (N, 3)"
            )
        transform = self._link_transform(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            transform_cache=transform_cache,
        )

        component_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for attribute in ("value", "first", "second"):
            coefficient_lower = np.empty((3, 4), dtype=np.float64)
            coefficient_upper = np.empty((3, 4), dtype=np.float64)
            for row in range(3):
                for column in range(4):
                    bounds = self._bounds(
                        getattr(transform[row][column], attribute)
                    )
                    coefficient_lower[row, column] = bounds.lower
                    coefficient_upper[row, column] = bounds.upper
            component_pairs.append(
                self._outward_affine_points_many(
                    coefficient_lower=coefficient_lower,
                    coefficient_upper=coefficient_upper,
                    points_local_m=points,
                )
            )

        return IntervalPointMotionBatch(
            phase=IntervalBounds(phase_lower, phase_upper),
            position_lower_object_m=component_pairs[0][0],
            position_upper_object_m=component_pairs[0][1],
            velocity_lower_object_m_per_unit=component_pairs[1][0],
            velocity_upper_object_m_per_unit=component_pairs[1][1],
            acceleration_lower_object_m_per_unit_squared=(
                component_pairs[2][0]
            ),
            acceleration_upper_object_m_per_unit_squared=(
                component_pairs[2][1]
            ),
            method_id=BATCH_POINT_MOTION_METHOD_ID,
            decimal_precision=self.options.decimal_precision,
        )

    def point_motion(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        point_local_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> IntervalPointMotion:
        """Enclose one exact-FK point path and its first two derivatives."""

        transform = self._link_transform(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            transform_cache=transform_cache,
        )
        point = self._point_jet_with_cache(
            transform=transform,
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            point_local_m=point_local_m,
            transform_cache=transform_cache,
        )
        return IntervalPointMotion(
            phase=IntervalBounds(phase_lower, phase_upper),
            position_object_m=tuple(
                self._bounds(component.value) for component in point
            ),  # type: ignore[arg-type]
            velocity_object_m_per_unit=tuple(
                self._bounds(component.first) for component in point
            ),  # type: ignore[arg-type]
            acceleration_object_m_per_unit_squared=tuple(
                self._bounds(component.second) for component in point
            ),  # type: ignore[arg-type]
            method_id=METHOD_ID,
            decimal_precision=self.options.decimal_precision,
        )

    def link_transform_over_joint_box(
        self,
        *,
        link_name: str,
        independent_joint_intervals: Sequence[IntervalBounds],
        base_transform: np.ndarray,
    ) -> IntervalRigidTransform:
        """Enclose a link transform over all supplied joint combinations.

        This evaluates the complete Cartesian product of the joint intervals;
        it never substitutes interval midpoints or a finite set of endpoints.
        """

        intervals = tuple(independent_joint_intervals)
        transform = self._link_transform_over_joint_box(
            link_name=link_name,
            independent_joint_intervals=intervals,
            base_transform=base_transform,
        )
        return IntervalRigidTransform(
            link_name=str(link_name),
            independent_joint_names=tuple(
                self.hand_model.independent_joint_names
            ),
            joint_position_intervals=intervals,
            elements=tuple(
                tuple(
                    self._bounds(transform[row][column].value)
                    for column in range(4)
                )
                for row in range(3)
            ),  # type: ignore[arg-type]
            method_id=INTERVAL_RIGID_TRANSFORM_METHOD_ID,
            decimal_precision=self.options.decimal_precision,
        )

    def independent_joint_box_from_paths(
        self,
        *,
        initial_independent_joint_positions: Sequence[float],
        directions: Sequence[Sequence[float]],
        phase_intervals: Sequence[IntervalBounds],
    ) -> tuple[IntervalBounds, ...]:
        """Enclose the sum of all registered independent-joint paths."""

        joint_names = self.hand_model.independent_joint_names
        initial = np.asarray(
            initial_independent_joint_positions, dtype=np.float64
        )
        direction_rows = np.asarray(directions, dtype=np.float64)
        phases = tuple(phase_intervals)
        if (
            initial.shape != (len(joint_names),)
            or not np.all(np.isfinite(initial))
            or not phases
            or direction_rows.shape != (len(phases), len(joint_names))
            or not np.all(np.isfinite(direction_rows))
            or not all(isinstance(bounds, IntervalBounds) for bounds in phases)
        ):
            raise IntervalKinematicsError(
                "joint path box inputs must match independent joints and phases"
            )

        independent: dict[str, _Jet] = {}
        for column, (name, initial_value) in enumerate(
            zip(joint_names, initial)
        ):
            position = self._constant(float(initial_value))
            for row, phase in enumerate(phases):
                position = _add(
                    position,
                    _mul(
                        self._constant(float(direction_rows[row, column])),
                        self._bounded_constant(phase),
                    ),
                )
            independent[name] = position

        affine_cache: dict[str, tuple[str, float, float]] = {}
        for name in self.hand_model.joint_order:
            joint = self.hand_model.joints[name]
            if not joint.movable:
                continue
            source, multiplier, offset = self._affine_map(
                name, affine_cache, set()
            )
            position = _add(
                _mul(self._constant(multiplier), independent[source]),
                self._constant(offset),
            )
            assert joint.limit is not None
            if (
                position.value.a < self.context.mpf(joint.limit.lower)
                or position.value.b > self.context.mpf(joint.limit.upper)
            ):
                raise IntervalKinematicsError(
                    "interval joint path box violates the URDF limit contract"
                )
        return tuple(
            self._bounds(independent[name].value) for name in joint_names
        )

    def geometric_jacobian_bounds(
        self,
        *,
        link_name: str,
        independent_joint_intervals: Sequence[IntervalBounds],
        point_object_m: Sequence[IntervalBounds],
        base_transform: np.ndarray,
    ) -> IntervalGeometricJacobian:
        """Enclose a point geometric Jacobian over independent joint boxes.

        The point is supplied directly in the object/base coordinate system.
        This deliberately permits a certified contact-position interval to be
        combined with a certified joint-position interval without replacing
        either interval by a display midpoint.
        """

        joint_intervals = tuple(independent_joint_intervals)
        point_intervals = tuple(point_object_m)
        joint_names = self.hand_model.independent_joint_names
        if len(joint_intervals) != len(joint_names) or not all(
            isinstance(bounds, IntervalBounds) for bounds in joint_intervals
        ):
            raise IntervalKinematicsError(
                "joint interval box must match independent joints"
            )
        if len(point_intervals) != 3 or not all(
            isinstance(bounds, IntervalBounds) for bounds in point_intervals
        ):
            raise IntervalKinematicsError(
                "object-frame point box must contain three intervals"
            )

        independent = {
            name: self._bounded_constant(bounds)
            for name, bounds in zip(joint_names, joint_intervals)
        }
        affine_cache: dict[str, tuple[str, float, float]] = {}
        resolved: dict[str, _Jet] = {}
        for name in self.hand_model.joint_order:
            joint = self.hand_model.joints[name]
            if not joint.movable:
                continue
            source, multiplier, offset = self._affine_map(
                name, affine_cache, set()
            )
            try:
                source_position = independent[source]
            except KeyError as exc:
                raise IntervalKinematicsError(
                    f"mimic source {source} is not an independent joint"
                ) from exc
            position = _add(
                _mul(self._constant(multiplier), source_position),
                self._constant(offset),
            )
            assert joint.limit is not None
            lower_limit = self.context.mpf(joint.limit.lower)
            upper_limit = self.context.mpf(joint.limit.upper)
            if position.value.a < lower_limit or position.value.b > upper_limit:
                raise IntervalKinematicsError(
                    "interval joint box violates the URDF limit contract"
                )
            resolved[name] = position

        transform = self._matrix_from_float(base_transform)
        point = tuple(
            self._bounded_constant(bounds) for bounds in point_intervals
        )
        column_by_name = {
            name: index for index, name in enumerate(joint_names)
        }
        jacobian: list[list[_Jet]] = [
            [self._constant(0.0) for _name in joint_names]
            for _row in range(6)
        ]
        for name in self._ancestor_joint_names(link_name):
            joint = self.hand_model.joints[name]
            joint_frame = self._matmul(
                transform, self._origin_transform(joint)
            )
            if joint.movable:
                axis_object = self._vector_jet(
                    joint_frame, self._constant_vector(joint.axis)
                )
                joint_origin = tuple(joint_frame[row][3] for row in range(3))
                source, multiplier, _offset = self._affine_map(
                    name, affine_cache, set()
                )
                column = column_by_name[source]
                if joint.joint_type in ("revolute", "continuous"):
                    point_from_joint = tuple(
                        _sub(point_component, origin_component)
                        for point_component, origin_component in zip(
                            point, joint_origin
                        )
                    )
                    linear = self._cross(axis_object, point_from_joint)
                    angular = axis_object
                elif joint.joint_type == "prismatic":
                    linear = axis_object
                    angular = (
                        self._constant(0.0),
                        self._constant(0.0),
                        self._constant(0.0),
                    )
                else:
                    raise IntervalKinematicsError(
                        f"unsupported movable joint type: {joint.joint_type}"
                    )
                contribution = (*linear, *angular)
                for row, component in enumerate(contribution):
                    jacobian[row][column] = _add(
                        jacobian[row][column],
                        _mul(self._constant(multiplier), component),
                    )
                position = resolved[name]
            else:
                position = self._constant(0.0)
            transform = self._matmul(
                joint_frame, self._motion_transform(joint, position)
            )

        return IntervalGeometricJacobian(
            link_name=link_name,
            independent_joint_names=tuple(joint_names),
            joint_position_intervals=joint_intervals,
            point_object_m=point_intervals,  # type: ignore[arg-type]
            elements=tuple(
                tuple(self._bounds(component.value) for component in row)
                for row in jacobian
            ),
            method_id=INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID,
            decimal_precision=self.options.decimal_precision,
        )

    def _compiled_point_plane_link_plan(
        self, link_name: str
    ) -> _CompiledPointPlaneLinkPlan:
        """Prepare one immutable C payload from the URDF chain exactly once."""

        cached = self._compiled_point_plane_link_plan_cache.get(link_name)
        if cached is not None:
            return cached
        names = self._ancestor_joint_names(link_name)
        if not names:
            raise IntervalKinematicsError(
                "compiled contact link has no serial joint chain"
            )
        type_codes: list[int] = []
        source_indices: list[int] = []
        origins_xyz: list[tuple[float, float, float]] = []
        origins_rpy: list[tuple[float, float, float]] = []
        axes: list[tuple[float, float, float]] = []
        multipliers: list[float] = []
        offsets: list[float] = []
        for name in names:
            joint = self.hand_model.joints[name]
            if joint.joint_type == "fixed":
                type_codes.append(0)
                source_indices.append(-1)
                multipliers.append(0.0)
                offsets.append(0.0)
            elif joint.joint_type in ("revolute", "continuous"):
                type_codes.append(1)
                source, multiplier, offset = (
                    self._compiled_joint_affine_plans[name]
                )
                source_indices.append(source)
                multipliers.append(multiplier)
                offsets.append(offset)
            elif joint.joint_type == "prismatic":
                type_codes.append(2)
                source, multiplier, offset = (
                    self._compiled_joint_affine_plans[name]
                )
                source_indices.append(source)
                multipliers.append(multiplier)
                offsets.append(offset)
            else:  # pragma: no cover - model validation owns this boundary
                raise IntervalKinematicsError(
                    f"unsupported joint type for compiled FK: {joint.joint_type}"
                )
            origins_xyz.append(joint.origin_xyz_m)
            origins_rpy.append(joint.origin_rpy_rad)
            axes.append(joint.axis)

        def frozen(value: object, dtype: np.dtype) -> np.ndarray:
            result = np.ascontiguousarray(value, dtype=dtype)
            result.setflags(write=False)
            return result

        plan = _CompiledPointPlaneLinkPlan(
            joint_types=frozen(type_codes, np.dtype(np.int32)),
            source_indices=frozen(source_indices, np.dtype(np.int32)),
            origins_xyz_m=frozen(origins_xyz, np.dtype(np.float64)),
            origins_rpy_rad=frozen(origins_rpy, np.dtype(np.float64)),
            axes=frozen(axes, np.dtype(np.float64)),
            multipliers=frozen(multipliers, np.dtype(np.float64)),
            offsets=frozen(offsets, np.dtype(np.float64)),
        )
        self._compiled_point_plane_link_plan_cache[link_name] = plan
        return plan

    def _new_compiled_point_plane_binding(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        object_triangle_m: np.ndarray,
    ) -> _CompiledPointPlaneBinding | None:
        """Reuse one exact path evaluator and rebind only its object plane."""

        from kcg_connector.grasp.robust.mpfr_point_plane_backend import (
            MpfrPointPlaneBackendError,
            MpfrPointPlaneBackendUnavailable,
            MpfrPointPlaneEvaluator,
        )

        names = self._ancestor_joint_names(link_name)
        if not names:
            self.compiled_point_backend_status = "FALLBACK_MPMATH"
            self.compiled_point_backend_failure_reason = (
                "contact link has no serial joint chain"
            )
            return None

        q_start_array = np.asarray(q_start, dtype=np.float64)
        direction_array = np.asarray(direction, dtype=np.float64)
        base_array = np.asarray(base_transform, dtype=np.float64)
        witness_array = np.asarray(
            witness_point_local_m, dtype=np.float64
        )
        triangle_array = np.asarray(object_triangle_m, dtype=np.float64)
        q_identity = _CompiledPointPlaneBinding._identity(q_start_array)
        direction_identity = _CompiledPointPlaneBinding._identity(
            direction_array
        )
        base_identity = _CompiledPointPlaneBinding._identity(base_array)
        witness_identity = _CompiledPointPlaneBinding._identity(
            witness_array
        )
        triangle_identity = _CompiledPointPlaneBinding._identity(
            triangle_array
        )
        cache_key = (
            link_name,
            q_identity,
            direction_identity,
            base_identity,
            witness_identity,
        )
        cached = self._compiled_point_plane_binding_cache.get(cache_key)
        if cached is not None:
            if not cached.enabled:
                self.compiled_point_backend_status = (
                    "FALLBACK_MPMATH_AFTER_CACHED_EVALUATOR_ERROR"
                )
                self.compiled_point_backend_failure_reason = (
                    cached.failure_reason
                )
                return None
            rebound = False
            try:
                if cached.triangle_binary64 != triangle_identity:
                    cached.evaluator.rebind_object_triangle(triangle_array)
                    cached.triangle_binary64 = triangle_identity
                    self.compiled_point_binding_triangle_rebind_count += 1
                    rebound = True
            except MpfrPointPlaneBackendError as error:
                cached.enabled = False
                cached.failure_reason = str(error)
                self.compiled_point_backend_status = (
                    "FALLBACK_MPMATH_AFTER_TRIANGLE_REBIND_ERROR"
                )
                self.compiled_point_backend_failure_reason = str(error)
                return None
            self._compiled_point_plane_binding_cache.pop(cache_key)
            self._compiled_point_plane_binding_cache[cache_key] = cached
            self.compiled_point_binding_cache_hit_count += 1
            self.compiled_point_backend_status = (
                "ACTIVE_REBOUND" if rebound else "ACTIVE_CACHED"
            )
            self.compiled_point_backend_failure_reason = None
            return cached

        self.compiled_point_binding_cache_miss_count += 1
        try:
            plan = self._compiled_point_plane_link_plan(link_name)
        except IntervalKinematicsError as error:
            self.compiled_point_backend_status = "FALLBACK_MPMATH"
            self.compiled_point_backend_failure_reason = str(error)
            return None
        try:
            evaluator = MpfrPointPlaneEvaluator(
                precision_bits=max(64, int(self.context.prec)),
                joint_types=plan.joint_types,
                source_indices=plan.source_indices,
                origins_xyz_m=plan.origins_xyz_m,
                origins_rpy_rad=plan.origins_rpy_rad,
                axes=plan.axes,
                multipliers=plan.multipliers,
                offsets=plan.offsets,
                q_start=q_start_array,
                direction=direction_array,
                base_transform_3x4=base_array[:3, :4],
                witness_point_local_m=witness_array,
                object_triangle_m=triangle_array,
            )
        except (MpfrPointPlaneBackendUnavailable, MpfrPointPlaneBackendError) as error:
            self.compiled_point_backend_status = "FALLBACK_MPMATH"
            self.compiled_point_backend_failure_reason = str(error)
            return None
        self.compiled_point_backend_status = "ACTIVE"
        self.compiled_point_backend_failure_reason = None
        binding = _CompiledPointPlaneBinding(
            evaluator=evaluator,
            link_name=link_name,
            q_start_binary64=q_identity,
            direction_binary64=direction_identity,
            base_transform_binary64=base_identity,
            witness_binary64=witness_identity,
            triangle_binary64=triangle_identity,
        )
        while (
            len(self._compiled_point_plane_binding_cache)
            >= COMPILED_POINT_PLANE_BINDING_CACHE_CAPACITY
        ):
            oldest_key = next(iter(self._compiled_point_plane_binding_cache))
            evicted = self._compiled_point_plane_binding_cache.pop(oldest_key)
            evicted.evaluator.close()
            self.compiled_point_binding_cache_eviction_count += 1
        self._compiled_point_plane_binding_cache[cache_key] = binding
        return binding

    def _object_plane_value_data(
        self, object_triangle_m: np.ndarray
    ) -> _ObjectPlaneValueData:
        object_triangle = np.asarray(object_triangle_m, dtype=np.float64)
        if (
            object_triangle.shape != (3, 3)
            or not np.all(np.isfinite(object_triangle))
        ):
            raise IntervalKinematicsError(
                "object triangle must be finite shape (3, 3)"
            )
        object_vertices = tuple(
            tuple(self._interval(float(value)) for value in row)
            for row in object_triangle
        )
        object_edge_one = tuple(
            second - first
            for first, second in zip(
                object_vertices[0], object_vertices[1]
            )
        )
        object_edge_two = tuple(
            third - first
            for first, third in zip(
                object_vertices[0], object_vertices[2]
            )
        )
        return _ObjectPlaneValueData(
            triangle_binary64=np.asarray(
                object_triangle, dtype=">f8"
            ).tobytes(order="C"),
            origin=object_vertices[0],
            area=self._value_cross(object_edge_one, object_edge_two),
        )

    def contact_plane_motion(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        object_triangle_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> _IntervalPlaneMotion:
        """Enclose plane value and derivative without triangle/PAD work."""

        object_triangle = np.asarray(object_triangle_m, dtype=np.float64)
        if (
            object_triangle.shape != (3, 3)
            or not np.all(np.isfinite(object_triangle))
        ):
            raise IntervalKinematicsError(
                "object triangle must be finite shape (3, 3)"
            )
        transform = self._link_transform(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            transform_cache=transform_cache,
        )
        point = self._point_jet_with_cache(
            transform=transform,
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            point_local_m=witness_point_local_m,
            transform_cache=transform_cache,
        )
        vertices = tuple(
            self._constant_vector(row) for row in object_triangle
        )
        edge_one = tuple(
            _sub(second, first)
            for first, second in zip(vertices[0], vertices[1])
        )
        edge_two = tuple(
            _sub(third, first)
            for first, third in zip(vertices[0], vertices[2])
        )
        area = self._cross(edge_one, edge_two)
        relative = tuple(
            _sub(point_value, vertex_value)
            for point_value, vertex_value in zip(point, vertices[0])
        )
        plane = self._dot(area, relative)
        return _IntervalPlaneMotion(
            phase=IntervalBounds(phase_lower, phase_upper),
            plane_value=self._bounds(plane.value),
            plane_derivative=self._bounds(plane.first),
        )

    def contact_pad_approach(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        pad_triangle_local_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> IntervalBounds:
        """Enclose PAD-normal closing motion independently of object faces."""

        pad_triangle = np.asarray(pad_triangle_local_m, dtype=np.float64)
        if (
            pad_triangle.shape != (3, 3)
            or not np.all(np.isfinite(pad_triangle))
        ):
            raise IntervalKinematicsError(
                "PAD triangle must be finite shape (3, 3)"
            )
        transform = self._link_transform(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            transform_cache=transform_cache,
        )
        point = self._point_jet_with_cache(
            transform=transform,
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            point_local_m=witness_point_local_m,
            transform_cache=transform_cache,
        )
        pad_area = self._pad_area_jet_with_cache(
            transform=transform,
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            pad_triangle_local_m=pad_triangle,
            transform_cache=transform_cache,
        )
        approach = self._interval(0.0)
        for pad_component, point_component in zip(pad_area, point):
            approach += pad_component.value * point_component.first
        return self._bounds(approach)

    def contact_plane_value(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        object_triangle_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
        _object_plane_value_data: _ObjectPlaneValueData | None = None,
        _compiled_point_evaluator: _CompiledPointPlaneBinding | None = None,
    ) -> IntervalBounds:
        """Enclose only the object-plane value needed by root bisection."""

        object_triangle = np.asarray(object_triangle_m, dtype=np.float64)
        if (
            object_triangle.shape != (3, 3)
            or not np.all(np.isfinite(object_triangle))
        ):
            raise IntervalKinematicsError(
                "object triangle must be finite shape (3, 3)"
            )
        triangle_binary64 = np.asarray(
            object_triangle, dtype=">f8"
        ).tobytes(order="C")
        if _object_plane_value_data is None:
            plane_data = self._object_plane_value_data(object_triangle)
        elif (
            not isinstance(_object_plane_value_data, _ObjectPlaneValueData)
            or _object_plane_value_data.triangle_binary64
            != triangle_binary64
        ):
            raise IntervalKinematicsError(
                "precompiled object plane differs from object triangle"
            )
        else:
            plane_data = _object_plane_value_data
        if _compiled_point_evaluator is not None:
            if (
                not isinstance(
                    _compiled_point_evaluator,
                    _CompiledPointPlaneBinding,
                )
                or not _compiled_point_evaluator.matches(
                    link_name=link_name,
                    q_start=q_start,
                    direction=direction,
                    base_transform=base_transform,
                    witness=witness_point_local_m,
                    triangle_binary64=triangle_binary64,
                )
            ):
                raise IntervalKinematicsError(
                    "compiled point-plane evaluator differs from equation"
                )
            if (
                phase_lower == phase_upper
                and _compiled_point_evaluator.enabled
            ):
                try:
                    lower, upper = (
                        _compiled_point_evaluator.evaluator.evaluate(
                            phase_lower
                        )
                    )
                except RuntimeError as error:
                    _compiled_point_evaluator.enabled = False
                    _compiled_point_evaluator.failure_reason = str(error)
                    self.compiled_point_backend_status = (
                        "FALLBACK_MPMATH_AFTER_EVALUATION_ERROR"
                    )
                    self.compiled_point_backend_failure_reason = str(error)
                else:
                    self.compiled_point_evaluation_count += 1
                    return IntervalBounds(lower, upper)
        transform = self._value_link_transform(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            transform_cache=transform_cache,
        )
        point = self._value_point_with_cache(
            transform=transform,
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            point_local_m=witness_point_local_m,
            transform_cache=transform_cache,
        )
        relative = tuple(
            point_value - vertex_value
            for point_value, vertex_value in zip(point, plane_data.origin)
        )
        return self._bounds(self._value_dot(plane_data.area, relative))

    def object_triangle_affine_form_bounds(
        self,
        object_triangle_m: np.ndarray,
    ) -> tuple[
        tuple[IntervalBounds, IntervalBounds, IntervalBounds, IntervalBounds],
        tuple[IntervalBounds, IntervalBounds, IntervalBounds, IntervalBounds],
        tuple[IntervalBounds, IntervalBounds, IntervalBounds, IntervalBounds],
        tuple[IntervalBounds, IntervalBounds, IntervalBounds, IntervalBounds],
    ]:
        """Precompute outward affine bounds for one plane and three edges.

        Each returned row encloses ``a*x + b*y + c*z + d``.  Row zero is the
        object-plane predicate; rows one through three are the same oriented
        triangle-edge halfspaces used by :meth:`contact_predicates`.  The
        geometry-only coefficients can be reused for every witness path.
        """

        object_triangle = np.asarray(object_triangle_m, dtype=np.float64)
        if (
            object_triangle.shape != (3, 3)
            or not np.all(np.isfinite(object_triangle))
        ):
            raise IntervalKinematicsError(
                "object triangle must be finite shape (3, 3)"
            )
        vertices = tuple(
            tuple(self._interval(float(value)) for value in row)
            for row in object_triangle
        )
        edge_one = tuple(
            second - first
            for first, second in zip(vertices[0], vertices[1])
        )
        edge_two = tuple(
            third - first
            for first, third in zip(vertices[0], vertices[2])
        )
        area = self._value_cross(edge_one, edge_two)

        def affine_row(
            coefficients: tuple[object, object, object],
            origin: tuple[object, object, object],
        ) -> tuple[
            IntervalBounds,
            IntervalBounds,
            IntervalBounds,
            IntervalBounds,
        ]:
            offset = -self._value_dot(coefficients, origin)
            return tuple(
                self._bounds(value) for value in (*coefficients, offset)
            )  # type: ignore[return-value]

        rows = [affine_row(area, vertices[0])]
        for index in range(3):
            following = (index + 1) % 3
            edge = tuple(
                second - first
                for first, second in zip(
                    vertices[index], vertices[following]
                )
            )
            rows.append(
                affine_row(self._value_cross(area, edge), vertices[index])
            )
        return tuple(rows)  # type: ignore[return-value]

    def contact_triangle_edge_halfspaces(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        object_triangle_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> tuple[IntervalBounds, IntervalBounds, IntervalBounds]:
        """Enclose only triangle-edge values for an already isolated root."""

        object_triangle = np.asarray(object_triangle_m, dtype=np.float64)
        if (
            object_triangle.shape != (3, 3)
            or not np.all(np.isfinite(object_triangle))
        ):
            raise IntervalKinematicsError(
                "object triangle must be finite shape (3, 3)"
            )
        transform = self._value_link_transform(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            transform_cache=transform_cache,
        )
        point = self._value_point_with_cache(
            transform=transform,
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            point_local_m=witness_point_local_m,
            transform_cache=transform_cache,
        )
        vertices = tuple(
            tuple(self._interval(float(value)) for value in row)
            for row in object_triangle
        )
        edge_one = tuple(
            second - first
            for first, second in zip(vertices[0], vertices[1])
        )
        edge_two = tuple(
            third - first
            for first, third in zip(vertices[0], vertices[2])
        )
        area = self._value_cross(edge_one, edge_two)
        rows: list[IntervalBounds] = []
        for index in range(3):
            following = (index + 1) % 3
            edge = tuple(
                second - first
                for first, second in zip(
                    vertices[index], vertices[following]
                )
            )
            relative = tuple(
                point_value - vertex_value
                for point_value, vertex_value in zip(
                    point, vertices[index]
                )
            )
            rows.append(
                self._bounds(
                    self._value_dot(
                        area, self._value_cross(edge, relative)
                    )
                )
            )
        return tuple(rows)  # type: ignore[return-value]

    def contact_predicates(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        pad_triangle_local_m: np.ndarray,
        object_triangle_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> IntervalContactPredicates:
        """Enclose all predicates needed for a transverse contact proof."""

        pad_triangle = np.asarray(pad_triangle_local_m, dtype=np.float64)
        object_triangle = np.asarray(object_triangle_m, dtype=np.float64)
        if (
            pad_triangle.shape != (3, 3)
            or object_triangle.shape != (3, 3)
            or not np.all(np.isfinite(pad_triangle))
            or not np.all(np.isfinite(object_triangle))
        ):
            raise IntervalKinematicsError(
                "contact predicate triangles must be finite shape (3, 3)"
            )
        transform = self._link_transform(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            transform_cache=transform_cache,
        )
        point = self._point_jet_with_cache(
            transform=transform,
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            point_local_m=witness_point_local_m,
            transform_cache=transform_cache,
        )
        pad_area_object = self._pad_area_jet_with_cache(
            transform=transform,
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
            pad_triangle_local_m=pad_triangle,
            transform_cache=transform_cache,
        )

        object_vertices = tuple(
            self._constant_vector(row) for row in object_triangle
        )
        object_edge_one = tuple(
            _sub(second, first)
            for first, second in zip(
                object_vertices[0], object_vertices[1]
            )
        )
        object_edge_two = tuple(
            _sub(third, first)
            for first, third in zip(
                object_vertices[0], object_vertices[2]
            )
        )
        object_area = self._cross(object_edge_one, object_edge_two)
        relative = tuple(
            _sub(point_value, vertex_value)
            for point_value, vertex_value in zip(point, object_vertices[0])
        )
        plane = self._dot(object_area, relative)

        edge_rows: list[_Jet] = []
        for index in range(3):
            following = (index + 1) % 3
            edge = tuple(
                _sub(second, first)
                for first, second in zip(
                    object_vertices[index], object_vertices[following]
                )
            )
            edge_relative = tuple(
                _sub(point_value, vertex_value)
                for point_value, vertex_value in zip(
                    point, object_vertices[index]
                )
            )
            edge_rows.append(
                self._dot(object_area, self._cross(edge, edge_relative))
            )

        pad_approach = self._interval(0.0)
        for pad_component, point_component in zip(
            pad_area_object, point
        ):
            pad_approach += pad_component.value * point_component.first

        return IntervalContactPredicates(
            phase=IntervalBounds(phase_lower, phase_upper),
            position_object_m=tuple(
                self._bounds(component.value) for component in point
            ),  # type: ignore[arg-type]
            plane_value=self._bounds(plane.value),
            plane_derivative=self._bounds(plane.first),
            plane_second_derivative=self._bounds(plane.second),
            triangle_edge_halfspaces=tuple(
                self._bounds(row.value) for row in edge_rows
            ),  # type: ignore[arg-type]
            pad_approach=self._bounds(pad_approach),
            object_plane_transversality=self._bounds(abs(plane.first)),
            method_id=METHOD_ID,
            decimal_precision=self.options.decimal_precision,
        )

    @staticmethod
    def _strict_sign(bounds: IntervalBounds) -> int:
        if bounds.strictly_positive:
            return 1
        if bounds.strictly_negative:
            return -1
        return 0

    @staticmethod
    def _strict_absolute_bounds(bounds: IntervalBounds) -> IntervalBounds:
        if bounds.strictly_positive:
            return bounds
        if bounds.strictly_negative:
            return IntervalBounds(-bounds.upper, -bounds.lower)
        raise IntervalKinematicsError(
            "absolute transversality requires a strict derivative sign"
        )

    def _nominal_contact_plane_value(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase: float,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        object_triangle_m: np.ndarray,
    ) -> float | None:
        """Return an untrusted binary64 root-location hint.

        This value is never used as acceptance or rejection evidence.  The
        caller must re-evaluate both proposed bracket endpoints with the
        interval backend before it may contract a certified bracket.
        """

        try:
            q = (
                np.asarray(q_start, dtype=np.float64)
                + float(phase) * np.asarray(direction, dtype=np.float64)
            )
            transform = self.hand_model.forward_kinematics(
                q,
                base_transform=np.asarray(base_transform, dtype=np.float64),
            )[link_name]
        except (HandModelError, KeyError, ValueError, FloatingPointError):
            return None
        witness = np.asarray(witness_point_local_m, dtype=np.float64)
        triangle = np.asarray(object_triangle_m, dtype=np.float64)
        if (
            witness.shape != (3,)
            or triangle.shape != (3, 3)
            or not np.all(np.isfinite(witness))
            or not np.all(np.isfinite(triangle))
        ):
            return None
        point = transform[:3, :3] @ witness + transform[:3, 3]
        normal = np.cross(
            triangle[1] - triangle[0], triangle[2] - triangle[0]
        )
        value = float(np.dot(normal, point - triangle[0]))
        return value if math.isfinite(value) else None

    def _nominal_root_seed_bracket(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        object_triangle_m: np.ndarray,
        certified_lower_sign: int,
        certified_upper_sign: int,
    ) -> tuple[float, float] | None:
        """Propose a small bracket; exact interval signs remain mandatory."""

        arguments = {
            "link_name": link_name,
            "q_start": q_start,
            "direction": direction,
            "base_transform": base_transform,
            "witness_point_local_m": witness_point_local_m,
            "object_triangle_m": object_triangle_m,
        }
        lower = float(phase_lower)
        upper = float(phase_upper)
        lower_value = self._nominal_contact_plane_value(
            phase=lower, **arguments
        )
        upper_value = self._nominal_contact_plane_value(
            phase=upper, **arguments
        )
        if lower_value is None or upper_value is None:
            return None
        lower_nominal_sign = 1 if lower_value > 0.0 else -1 if lower_value < 0.0 else 0
        upper_nominal_sign = 1 if upper_value > 0.0 else -1 if upper_value < 0.0 else 0
        if (
            lower_nominal_sign != certified_lower_sign
            or upper_nominal_sign != certified_upper_sign
        ):
            return None

        for _index in range(NOMINAL_ROOT_SEED_MAXIMUM_ITERATIONS):
            if np.nextafter(lower, upper) >= upper:
                break
            denominator = upper_value - lower_value
            proposed = (
                lower - lower_value * (upper - lower) / denominator
                if denominator != 0.0 and math.isfinite(denominator)
                else math.nan
            )
            central_lower = lower + 0.125 * (upper - lower)
            central_upper = upper - 0.125 * (upper - lower)
            if (
                not math.isfinite(proposed)
                or proposed <= central_lower
                or proposed >= central_upper
            ):
                proposed = lower + 0.5 * (upper - lower)
            proposed_value = self._nominal_contact_plane_value(
                phase=proposed, **arguments
            )
            if proposed_value is None:
                return None
            proposed_sign = (
                1 if proposed_value > 0.0
                else -1 if proposed_value < 0.0
                else 0
            )
            if proposed_sign == 0:
                lower = float(np.nextafter(proposed, phase_lower))
                upper = float(np.nextafter(proposed, phase_upper))
                break
            if proposed_sign == certified_lower_sign:
                lower = proposed
                lower_value = proposed_value
            elif proposed_sign == certified_upper_sign:
                upper = proposed
                upper_value = proposed_value
            else:
                return None

        for _index in range(NOMINAL_ROOT_SEED_ENDPOINT_ULP_PADDING):
            lower = max(
                phase_lower,
                float(np.nextafter(lower, phase_lower)),
            )
            upper = min(
                phase_upper,
                float(np.nextafter(upper, phase_upper)),
            )
        if lower >= upper:
            return None
        return lower, upper

    def certify_transverse_plane_root(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        pad_triangle_local_m: np.ndarray,
        object_triangle_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
        _use_interval_newton_contraction: bool = True,
        _use_nominal_root_seed: bool = True,
        _use_compiled_root_transaction: bool = True,
        _precertified_plane_derivative: IntervalBounds | None = None,
        _precertified_lower_value: IntervalBounds | None = None,
        _precertified_upper_value: IntervalBounds | None = None,
    ) -> IntervalPlaneRootClassification:
        """Isolate one transverse plane root without triangle-edge work."""

        if type(_use_interval_newton_contraction) is not bool:
            raise IntervalKinematicsError(
                "interval Newton contraction switch must be boolean"
            )
        if type(_use_nominal_root_seed) is not bool:
            raise IntervalKinematicsError(
                "nominal root-seed switch must be boolean"
            )
        if type(_use_compiled_root_transaction) is not bool:
            raise IntervalKinematicsError(
                "compiled root-transaction switch must be boolean"
            )
        precertified_values = (
            _precertified_plane_derivative,
            _precertified_lower_value,
            _precertified_upper_value,
        )
        precertified = all(value is not None for value in precertified_values)
        if precertified != any(
            value is not None for value in precertified_values
        ) or any(
            value is not None and not isinstance(value, IntervalBounds)
            for value in precertified_values
        ):
            raise IntervalKinematicsError(
                "precertified plane gate requires derivative and both "
                "endpoint intervals together"
            )

        arguments = {
            "link_name": link_name,
            "q_start": q_start,
            "direction": direction,
            "base_transform": base_transform,
            "witness_point_local_m": witness_point_local_m,
            "pad_triangle_local_m": pad_triangle_local_m,
            "object_triangle_m": object_triangle_m,
            "transform_cache": transform_cache,
        }
        plane_value_data = self._object_plane_value_data(
            object_triangle_m
        )
        plane_arguments = {
            "link_name": link_name,
            "q_start": q_start,
            "direction": direction,
            "base_transform": base_transform,
            "witness_point_local_m": witness_point_local_m,
            "object_triangle_m": object_triangle_m,
            "transform_cache": transform_cache,
            "_object_plane_value_data": plane_value_data,
        }
        searched = IntervalBounds(phase_lower, phase_upper)
        if precertified:
            assert _precertified_plane_derivative is not None
            assert _precertified_lower_value is not None
            assert _precertified_upper_value is not None
            plane_derivative = _precertified_plane_derivative
            lower_value = _precertified_lower_value
            upper_value = _precertified_upper_value
            compiled_point_binding = self._new_compiled_point_plane_binding(
                link_name=link_name,
                q_start=q_start,
                direction=direction,
                base_transform=base_transform,
                witness_point_local_m=witness_point_local_m,
                object_triangle_m=object_triangle_m,
            )
            if compiled_point_binding is not None:
                plane_arguments["_compiled_point_evaluator"] = (
                    compiled_point_binding
                )
        else:
            plane_only = self.contact_plane_value(
                phase_lower=phase_lower,
                phase_upper=phase_upper,
                **plane_arguments,
            )
            if plane_only.strictly_positive or plane_only.strictly_negative:
                return IntervalPlaneRootClassification(
                    IntervalPlaneRootState.CERTIFIED_FREE,
                    searched,
                    None,
                    "OBJECT_PLANE_VALUE_EXCLUDES_ZERO",
                )
            whole = self.contact_plane_motion(
                link_name=link_name,
                q_start=q_start,
                direction=direction,
                phase_lower=phase_lower,
                phase_upper=phase_upper,
                base_transform=base_transform,
                witness_point_local_m=witness_point_local_m,
                object_triangle_m=object_triangle_m,
                transform_cache=transform_cache,
            )
            if (
                whole.plane_value.strictly_positive
                or whole.plane_value.strictly_negative
            ):
                return IntervalPlaneRootClassification(
                    IntervalPlaneRootState.CERTIFIED_FREE,
                    searched,
                    None,
                    "OBJECT_PLANE_VALUE_EXCLUDES_ZERO",
                )
            plane_derivative = whole.plane_derivative
            compiled_point_binding = self._new_compiled_point_plane_binding(
                link_name=link_name,
                q_start=q_start,
                direction=direction,
                base_transform=base_transform,
                witness_point_local_m=witness_point_local_m,
                object_triangle_m=object_triangle_m,
            )
            if compiled_point_binding is not None:
                plane_arguments["_compiled_point_evaluator"] = (
                    compiled_point_binding
                )
            lower_value = self.contact_plane_value(
                phase_lower=phase_lower,
                phase_upper=phase_lower,
                **plane_arguments,
            )
            upper_value = self.contact_plane_value(
                phase_lower=phase_upper,
                phase_upper=phase_upper,
                **plane_arguments,
            )

        derivative_sign = self._strict_sign(plane_derivative)
        if derivative_sign == 0:
            return IntervalPlaneRootClassification(
                IntervalPlaneRootState.UNRESOLVED,
                searched,
                None,
                "NONTRANSVERSE_OR_MULTIPLE_PLANE_ROOTS",
            )

        lower_sign = self._strict_sign(lower_value)
        upper_sign = self._strict_sign(upper_value)
        if lower_sign == 0 or upper_sign == 0:
            return IntervalPlaneRootClassification(
                IntervalPlaneRootState.UNRESOLVED,
                searched,
                None,
                "PLANE_ENDPOINT_SIGN_UNRESOLVED",
            )
        if lower_sign == upper_sign:
            return IntervalPlaneRootClassification(
                IntervalPlaneRootState.CERTIFIED_FREE,
                searched,
                None,
                "STRICTLY_MONOTONE_PLANE_WITH_SAME_SIDE_ENDPOINTS",
            )
        if lower_sign != -derivative_sign or upper_sign != derivative_sign:
            return IntervalPlaneRootClassification(
                IntervalPlaneRootState.UNRESOLVED,
                searched,
                None,
                "DERIVATIVE_AND_ENDPOINT_SIGNS_INCONSISTENT",
            )

        if (
            compiled_point_binding is not None
            and _use_interval_newton_contraction
            and _use_nominal_root_seed
            and _use_compiled_root_transaction
        ):
            compiled_transaction = None
            try:
                compiled_transaction = (
                    compiled_point_binding.evaluator.
                    isolate_monotone_root(
                        phase_lower=phase_lower,
                        phase_upper=phase_upper,
                        derivative_lower=plane_derivative.lower,
                        derivative_upper=plane_derivative.upper,
                        lower_sign=lower_sign,
                        upper_sign=upper_sign,
                        maximum_iterations=(
                            self.options.maximum_root_bisection_iterations
                        ),
                    )
                )
            except RuntimeError as error:
                compiled_point_binding.enabled = False
                compiled_point_binding.failure_reason = str(error)
                self.compiled_point_backend_status = (
                    "FALLBACK_MPMATH_AFTER_ROOT_TRANSACTION_ERROR"
                )
                self.compiled_point_backend_failure_reason = str(error)
            if compiled_transaction is not None:
                (
                    compiled_lower,
                    compiled_upper,
                    compiled_lower_value,
                    compiled_upper_value,
                    compiled_interpolation_iterations,
                    compiled_newton_iterations,
                    compiled_bisection_iterations,
                ) = compiled_transaction
                try:
                    compiled_plane, compiled_positions = (
                        compiled_point_binding.evaluator.evaluate_interval(
                            compiled_lower,
                            compiled_upper,
                        )
                    )
                except RuntimeError as error:
                    compiled_point_binding.enabled = False
                    compiled_point_binding.failure_reason = str(error)
                    self.compiled_point_backend_status = (
                        "FALLBACK_MPMATH_AFTER_INTERVAL_POSITION_ERROR"
                    )
                    self.compiled_point_backend_failure_reason = str(error)
                else:
                    compiled_plane_bounds = IntervalBounds(*compiled_plane)
                    if compiled_plane_bounds.contains_zero:
                        pad_approach = self.contact_pad_approach(
                            link_name=link_name,
                            q_start=q_start,
                            direction=direction,
                            phase_lower=phase_lower,
                            phase_upper=phase_upper,
                            base_transform=base_transform,
                            witness_point_local_m=witness_point_local_m,
                            pad_triangle_local_m=pad_triangle_local_m,
                            transform_cache=transform_cache,
                        )
                        root = IntervalTransversePlaneRoot(
                            searched_phase=searched,
                            isolating_interval=IntervalBounds(
                                compiled_lower, compiled_upper
                            ),
                            value_at_lower=IntervalBounds(
                                *compiled_lower_value
                            ),
                            value_at_upper=IntervalBounds(
                                *compiled_upper_value
                            ),
                            plane_derivative=plane_derivative,
                            position_object_m=tuple(
                                IntervalBounds(*row)
                                for row in compiled_positions
                            ),  # type: ignore[arg-type]
                            pad_approach=pad_approach,
                            object_plane_transversality=(
                                self._strict_absolute_bounds(
                                    plane_derivative
                                )
                            ),
                            object_source_winding_free_side_sign=(
                                lower_sign
                            ),
                            interpolation_iterations=(
                                compiled_interpolation_iterations
                            ),
                            interval_newton_iterations=(
                                compiled_newton_iterations
                            ),
                            bisection_iterations=(
                                compiled_bisection_iterations
                            ),
                            method_id=METHOD_ID,
                            decimal_precision=(
                                self.options.decimal_precision
                            ),
                        )
                        self.compiled_root_transaction_count += 1
                        self.compiled_interval_position_evaluation_count += 1
                        return IntervalPlaneRootClassification(
                            IntervalPlaneRootState.
                            CERTIFIED_TRANSVERSE_PLANE_ROOT,
                            searched,
                            root,
                            "STRICT_IVT_MONOTONE_TRANSVERSE_PLANE_ROOT",
                        )
                    compiled_point_binding.enabled = False
                    compiled_point_binding.failure_reason = (
                        "compiled root-position plane interval excludes zero"
                    )
                    self.compiled_point_backend_status = (
                        "FALLBACK_MPMATH_AFTER_INTERVAL_POSITION_ERROR"
                    )
                    self.compiled_point_backend_failure_reason = (
                        compiled_point_binding.failure_reason
                    )

        bracket_lower = phase_lower
        bracket_upper = phase_upper
        if _use_nominal_root_seed:
            nominal_bracket = self._nominal_root_seed_bracket(
                link_name=link_name,
                q_start=q_start,
                direction=direction,
                phase_lower=phase_lower,
                phase_upper=phase_upper,
                base_transform=base_transform,
                witness_point_local_m=witness_point_local_m,
                object_triangle_m=object_triangle_m,
                certified_lower_sign=lower_sign,
                certified_upper_sign=upper_sign,
            )
            if nominal_bracket is not None:
                proposed_lower, proposed_upper = nominal_bracket
                proposed_lower_value = self.contact_plane_value(
                    phase_lower=proposed_lower,
                    phase_upper=proposed_lower,
                    **plane_arguments,
                )
                proposed_upper_value = self.contact_plane_value(
                    phase_lower=proposed_upper,
                    phase_upper=proposed_upper,
                    **plane_arguments,
                )
                if (
                    self._strict_sign(proposed_lower_value) == lower_sign
                    and self._strict_sign(proposed_upper_value) == upper_sign
                ):
                    bracket_lower = proposed_lower
                    bracket_upper = proposed_upper
                    lower_value = proposed_lower_value
                    upper_value = proposed_upper_value
        interval_newton_iterations = 0
        for _index in range(
            _MAX_CERTIFIED_INTERVAL_NEWTON_ITERATIONS
            if _use_interval_newton_contraction
            else 0
        ):
            if np.nextafter(bracket_lower, bracket_upper) >= bracket_upper:
                break
            midpoint = bracket_lower + 0.5 * (
                bracket_upper - bracket_lower
            )
            middle_value = self.contact_plane_value(
                phase_lower=midpoint,
                phase_upper=midpoint,
                **plane_arguments,
            )
            middle_sign = self._strict_sign(middle_value)
            if middle_sign == 0:
                break
            quotient = _outward_interval_divide(
                middle_value, plane_derivative
            )
            candidate_lower = float(
                np.nextafter(midpoint - quotient.upper, -math.inf)
            )
            candidate_upper = float(
                np.nextafter(midpoint - quotient.lower, math.inf)
            )
            if not (
                math.isfinite(candidate_lower)
                and math.isfinite(candidate_upper)
                and candidate_lower <= candidate_upper
            ):
                break
            ulp_padding = _INTERVAL_NEWTON_ENDPOINT_ULP_PADDING * max(
                math.ulp(midpoint),
                math.ulp(candidate_lower),
                math.ulp(candidate_upper),
            )
            proposed_lower = max(
                bracket_lower,
                float(
                    np.nextafter(
                        candidate_lower - ulp_padding, -math.inf
                    )
                ),
            )
            proposed_upper = min(
                bracket_upper,
                float(
                    np.nextafter(
                        candidate_upper + ulp_padding, math.inf
                    )
                ),
            )
            old_width = bracket_upper - bracket_lower
            new_width = proposed_upper - proposed_lower
            if (
                proposed_lower > proposed_upper
                or not math.isfinite(new_width)
                or new_width >= old_width
            ):
                break
            proposed_lower_value = self.contact_plane_value(
                phase_lower=proposed_lower,
                phase_upper=proposed_lower,
                **plane_arguments,
            )
            proposed_upper_value = self.contact_plane_value(
                phase_lower=proposed_upper,
                phase_upper=proposed_upper,
                **plane_arguments,
            )
            interval_newton_iterations += 1
            if (
                self._strict_sign(proposed_lower_value) != lower_sign
                or self._strict_sign(proposed_upper_value) != upper_sign
            ):
                break
            bracket_lower = proposed_lower
            bracket_upper = proposed_upper
            lower_value = proposed_lower_value
            upper_value = proposed_upper_value

        iterations = 0
        while np.nextafter(bracket_lower, bracket_upper) < bracket_upper:
            if iterations >= self.options.maximum_root_bisection_iterations:
                return IntervalPlaneRootClassification(
                    IntervalPlaneRootState.UNRESOLVED,
                    IntervalBounds(bracket_lower, bracket_upper),
                    None,
                    "ROOT_BISECTION_COMPUTATION_BUDGET_EXHAUSTED",
                )
            midpoint = bracket_lower + 0.5 * (
                bracket_upper - bracket_lower
            )
            middle_value = self.contact_plane_value(
                phase_lower=midpoint,
                phase_upper=midpoint,
                **plane_arguments,
            )
            middle_sign = self._strict_sign(middle_value)
            iterations += 1
            if middle_sign == 0:
                predecessor = float(np.nextafter(midpoint, bracket_lower))
                successor = float(np.nextafter(midpoint, bracket_upper))
                predecessor_value = self.contact_plane_value(
                    phase_lower=predecessor,
                    phase_upper=predecessor,
                    **plane_arguments,
                )
                successor_value = self.contact_plane_value(
                    phase_lower=successor,
                    phase_upper=successor,
                    **plane_arguments,
                )
                if (
                    self._strict_sign(predecessor_value) != lower_sign
                    or self._strict_sign(successor_value) != upper_sign
                ):
                    return IntervalPlaneRootClassification(
                        IntervalPlaneRootState.UNRESOLVED,
                        IntervalBounds(bracket_lower, bracket_upper),
                        None,
                        "ROOT_NEIGHBOR_SIGN_UNRESOLVED",
                    )
                bracket_lower = predecessor
                bracket_upper = successor
                lower_value = predecessor_value
                upper_value = successor_value
                break
            if middle_sign == lower_sign:
                bracket_lower = midpoint
                lower_value = middle_value
            else:
                bracket_upper = midpoint
                upper_value = middle_value

        root_row = self.contact_predicates(
            phase_lower=bracket_lower,
            phase_upper=bracket_upper,
            **arguments,
        )
        root = IntervalTransversePlaneRoot(
            searched_phase=searched,
            isolating_interval=IntervalBounds(
                bracket_lower, bracket_upper
            ),
            value_at_lower=lower_value,
            value_at_upper=upper_value,
            plane_derivative=root_row.plane_derivative,
            position_object_m=root_row.position_object_m,
            pad_approach=root_row.pad_approach,
            object_plane_transversality=(
                root_row.object_plane_transversality
            ),
            object_source_winding_free_side_sign=lower_sign,
            interpolation_iterations=0,
            interval_newton_iterations=interval_newton_iterations,
            bisection_iterations=iterations,
            method_id=METHOD_ID,
            decimal_precision=self.options.decimal_precision,
        )
        return IntervalPlaneRootClassification(
            IntervalPlaneRootState.CERTIFIED_TRANSVERSE_PLANE_ROOT,
            searched,
            root,
            (
                "STRICT_IVT_MONOTONE_TRANSVERSE_PLANE_ROOT_WITH_"
                "CERTIFIED_INTERVAL_NEWTON_CONTRACTION"
                if interval_newton_iterations > 0
                else "STRICT_IVT_MONOTONE_TRANSVERSE_PLANE_ROOT"
            ),
        )

    def finalize_transverse_plane_root_for_triangle(
        self,
        *,
        plane_classification: IntervalPlaneRootClassification,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        pad_triangle_local_m: np.ndarray,
        object_triangle_m: np.ndarray,
        representative_object_triangle_m: np.ndarray | None = None,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> IntervalRootClassification:
        """Apply one isolated exact plane root to one actual triangle."""

        if not isinstance(
            plane_classification, IntervalPlaneRootClassification
        ):
            raise IntervalKinematicsError(
                "triangle finalization requires a plane root classification"
            )
        supplied_phase = IntervalBounds(phase_lower, phase_upper)
        if supplied_phase != plane_classification.searched_phase:
            raise IntervalKinematicsError(
                "triangle finalization phase differs from the isolated plane root"
            )
        if plane_classification.state is IntervalPlaneRootState.CERTIFIED_FREE:
            return IntervalRootClassification(
                IntervalRootState.CERTIFIED_FREE,
                plane_classification.searched_phase,
                None,
                plane_classification.reason,
            )
        if plane_classification.state is IntervalPlaneRootState.UNRESOLVED:
            return IntervalRootClassification(
                IntervalRootState.UNRESOLVED,
                plane_classification.searched_phase,
                None,
                plane_classification.reason,
            )
        plane_root = plane_classification.root
        if plane_root is None:  # pragma: no cover - dataclass enforces this
            raise IntervalKinematicsError("certified plane root is missing")
        bracket = plane_root.isolating_interval
        if representative_object_triangle_m is None:
            root_row = self.contact_predicates(
                link_name=link_name,
                q_start=q_start,
                direction=direction,
                phase_lower=bracket.lower,
                phase_upper=bracket.upper,
                base_transform=base_transform,
                witness_point_local_m=witness_point_local_m,
                pad_triangle_local_m=pad_triangle_local_m,
                object_triangle_m=object_triangle_m,
                transform_cache=transform_cache,
            )
            actual_plane_arguments = {
                "link_name": link_name,
                "q_start": q_start,
                "direction": direction,
                "base_transform": base_transform,
                "witness_point_local_m": witness_point_local_m,
                "object_triangle_m": object_triangle_m,
                "transform_cache": transform_cache,
            }
            actual_lower_value = self.contact_plane_value(
                phase_lower=bracket.lower,
                phase_upper=bracket.lower,
                **actual_plane_arguments,
            )
            actual_upper_value = self.contact_plane_value(
                phase_lower=bracket.upper,
                phase_upper=bracket.upper,
                **actual_plane_arguments,
            )
            actual_plane_derivative = root_row.plane_derivative
            triangle_edge_halfspaces = root_row.triangle_edge_halfspaces
            pad_approach = root_row.pad_approach
            object_plane_transversality = (
                root_row.object_plane_transversality
            )
            position_object_m = root_row.position_object_m
            root_plane_excludes_zero = (
                root_row.plane_value.strictly_positive
                or root_row.plane_value.strictly_negative
            )
        else:
            scale = _exact_same_plane_scale(
                np.asarray(
                    representative_object_triangle_m, dtype=np.float64
                ),
                np.asarray(object_triangle_m, dtype=np.float64),
            )
            if scale is None:
                return IntervalRootClassification(
                    IntervalRootState.UNRESOLVED,
                    bracket,
                    None,
                    "REUSED_EXACT_PLANE_ROOT_NOT_CERTIFIED_FOR_TRIANGLE_PLANE",
                )
            scale_interval = (
                self.context.mpf(scale.numerator)
                / self.context.mpf(scale.denominator)
            )

            def scaled(bounds: IntervalBounds) -> IntervalBounds:
                return self._bounds(
                    self._interval(bounds.lower, bounds.upper)
                    * scale_interval
                )

            actual_lower_value = scaled(plane_root.value_at_lower)
            actual_upper_value = scaled(plane_root.value_at_upper)
            actual_plane_derivative = scaled(plane_root.plane_derivative)
            object_plane_transversality = self._strict_absolute_bounds(
                actual_plane_derivative
            )
            position_object_m = plane_root.position_object_m
            point = tuple(
                self._interval(bounds.lower, bounds.upper)
                for bounds in position_object_m
            )
            vertices = tuple(
                tuple(self._interval(float(value)) for value in row)
                for row in np.asarray(object_triangle_m, dtype=np.float64)
            )
            first_edge = tuple(
                second - first
                for first, second in zip(vertices[0], vertices[1])
            )
            second_edge = tuple(
                third - first
                for first, third in zip(vertices[0], vertices[2])
            )
            area = self._value_cross(first_edge, second_edge)
            edge_rows: list[IntervalBounds] = []
            for index in range(3):
                following = (index + 1) % 3
                edge = tuple(
                    second - first
                    for first, second in zip(
                        vertices[index], vertices[following]
                    )
                )
                relative = tuple(
                    point_value - vertex_value
                    for point_value, vertex_value in zip(
                        point, vertices[index]
                    )
                )
                edge_rows.append(
                    self._bounds(
                        self._value_dot(
                            area, self._value_cross(edge, relative)
                        )
                    )
                )
            triangle_edge_halfspaces = tuple(edge_rows)
            pad_approach = plane_root.pad_approach
            if not (
                pad_approach.strictly_positive
                or pad_approach.strictly_negative
            ):
                pad_approach = self.contact_pad_approach(
                    link_name=link_name,
                    q_start=q_start,
                    direction=direction,
                    phase_lower=bracket.lower,
                    phase_upper=bracket.upper,
                    base_transform=base_transform,
                    witness_point_local_m=witness_point_local_m,
                    pad_triangle_local_m=pad_triangle_local_m,
                    transform_cache=transform_cache,
                )
            root_plane_excludes_zero = False
        actual_lower_sign = self._strict_sign(actual_lower_value)
        actual_upper_sign = self._strict_sign(actual_upper_value)
        actual_derivative_sign = self._strict_sign(
            actual_plane_derivative
        )
        if (
            root_plane_excludes_zero
            or actual_lower_sign == 0
            or actual_upper_sign != -actual_lower_sign
            or actual_derivative_sign != actual_upper_sign
        ):
            return IntervalRootClassification(
                IntervalRootState.UNRESOLVED,
                bracket,
                None,
                "REUSED_EXACT_PLANE_ROOT_NOT_CERTIFIED_FOR_TRIANGLE_PLANE",
            )
        if any(
            edge.strictly_negative
            for edge in triangle_edge_halfspaces
        ):
            return IntervalRootClassification(
                IntervalRootState.CERTIFIED_FREE,
                bracket,
                None,
                "UNIQUE_PLANE_ROOT_STRICTLY_OUTSIDE_TRIANGLE",
            )
        if not all(
            edge.strictly_positive
            for edge in triangle_edge_halfspaces
        ):
            return IntervalRootClassification(
                IntervalRootState.UNRESOLVED,
                bracket,
                None,
                "TRIANGLE_BOUNDARY_NOT_STRICTLY_INTERIOR",
            )
        equation_sha256, feature_identity_sha256 = (
            _implicit_root_identities(
                hand_model=self.hand_model,
                link_name=link_name,
                q_start=q_start,
                direction=direction,
                base_transform=base_transform,
                witness_point_local_m=witness_point_local_m,
                pad_triangle_local_m=pad_triangle_local_m,
                object_triangle_m=object_triangle_m,
            )
        )
        implicit_root = CertifiedImplicitRoot(
            method_id=IMPLICIT_ROOT_METHOD_ID,
            equation_sha256=equation_sha256,
            feature_identity_sha256=feature_identity_sha256,
            feature_type=IMPLICIT_ROOT_FEATURE_TYPE,
            isolating_interval=bracket,
            value_at_lower=actual_lower_value,
            value_at_upper=actual_upper_value,
            derivative=actual_plane_derivative,
            uniqueness_proven=True,
            display_approximation=(
                bracket.lower + 0.5 * (bracket.upper - bracket.lower)
            ),
            display_approximation_role=DISPLAY_APPROXIMATION_ROLE,
        )
        certificate = IntervalTransverseRootCertificate(
            implicit_root=implicit_root,
            triangle_edge_halfspaces=triangle_edge_halfspaces,
            pad_approach=pad_approach,
            path_local_free_side_approach=(
                object_plane_transversality
            ),
            object_source_winding_free_side_sign=(
                actual_lower_sign
            ),
            position_object_m=position_object_m,
            bisection_iterations=plane_root.bisection_iterations,
            method_id=METHOD_ID,
            decimal_precision=self.options.decimal_precision,
        )
        if (
            pad_approach.strictly_positive
            and object_plane_transversality.strictly_positive
        ):
            state = IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT
            reason = "STRICT_IVT_MONOTONE_INTERIOR_PAD_DIRECTIONAL_TRANSVERSE_ROOT"
        elif pad_approach.strictly_negative:
            state = (
                IntervalRootState.CERTIFIED_PAD_DIRECTION_REJECTED_TRANSVERSE_ROOT
            )
            reason = "STRICT_IVT_MONOTONE_INTERIOR_PAD_REVERSE_TRANSVERSE_ROOT"
        else:
            return IntervalRootClassification(
                IntervalRootState.UNRESOLVED,
                bracket,
                None,
                "CONTACT_DIRECTION_SIGN_UNRESOLVED",
            )
        return IntervalRootClassification(
            state,
            plane_classification.searched_phase,
            certificate,
            reason,
        )

    def certify_transverse_contact_root(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
        witness_point_local_m: np.ndarray,
        pad_triangle_local_m: np.ndarray,
        object_triangle_m: np.ndarray,
        transform_cache: IntervalLinkTransformCache | None = None,
    ) -> IntervalRootClassification:
        """Classify one witness/face interval without residual tolerances."""

        arguments = {
            "link_name": link_name,
            "q_start": q_start,
            "direction": direction,
            "base_transform": base_transform,
            "witness_point_local_m": witness_point_local_m,
            "pad_triangle_local_m": pad_triangle_local_m,
            "object_triangle_m": object_triangle_m,
            "transform_cache": transform_cache,
        }
        plane_arguments = {
            "link_name": link_name,
            "q_start": q_start,
            "direction": direction,
            "base_transform": base_transform,
            "witness_point_local_m": witness_point_local_m,
            "object_triangle_m": object_triangle_m,
            "transform_cache": transform_cache,
        }
        searched = IntervalBounds(phase_lower, phase_upper)
        plane_only = self.contact_plane_value(
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            **plane_arguments,
        )
        if plane_only.strictly_positive or plane_only.strictly_negative:
            return IntervalRootClassification(
                IntervalRootState.CERTIFIED_FREE,
                searched,
                None,
                "OBJECT_PLANE_VALUE_EXCLUDES_ZERO",
            )
        whole = self.contact_predicates(
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            **arguments,
        )
        if whole.plane_value.strictly_positive or (
            whole.plane_value.strictly_negative
        ):
            return IntervalRootClassification(
                IntervalRootState.CERTIFIED_FREE,
                searched,
                None,
                "OBJECT_PLANE_VALUE_EXCLUDES_ZERO",
            )
        if any(
            edge.strictly_negative
            for edge in whole.triangle_edge_halfspaces
        ):
            return IntervalRootClassification(
                IntervalRootState.CERTIFIED_FREE,
                searched,
                None,
                "TRIANGLE_EDGE_HALFSPACE_EXCLUDES_PATH",
            )
        derivative_sign = self._strict_sign(whole.plane_derivative)
        if derivative_sign == 0:
            return IntervalRootClassification(
                IntervalRootState.UNRESOLVED,
                searched,
                None,
                "NONTRANSVERSE_OR_MULTIPLE_PLANE_ROOTS",
            )

        lower_value = self.contact_plane_value(
            phase_lower=phase_lower,
            phase_upper=phase_lower,
            **plane_arguments,
        )
        upper_value = self.contact_plane_value(
            phase_lower=phase_upper,
            phase_upper=phase_upper,
            **plane_arguments,
        )
        lower_sign = self._strict_sign(lower_value)
        upper_sign = self._strict_sign(upper_value)
        if lower_sign == 0 or upper_sign == 0:
            return IntervalRootClassification(
                IntervalRootState.UNRESOLVED,
                searched,
                None,
                "PLANE_ENDPOINT_SIGN_UNRESOLVED",
            )
        if lower_sign == upper_sign:
            return IntervalRootClassification(
                IntervalRootState.CERTIFIED_FREE,
                searched,
                None,
                "STRICTLY_MONOTONE_PLANE_WITH_SAME_SIDE_ENDPOINTS",
            )
        if lower_sign != -derivative_sign or upper_sign != derivative_sign:
            return IntervalRootClassification(
                IntervalRootState.UNRESOLVED,
                searched,
                None,
                "DERIVATIVE_AND_ENDPOINT_SIGNS_INCONSISTENT",
            )

        bracket_lower = phase_lower
        bracket_upper = phase_upper
        iterations = 0
        while np.nextafter(bracket_lower, bracket_upper) < bracket_upper:
            if iterations >= self.options.maximum_root_bisection_iterations:
                return IntervalRootClassification(
                    IntervalRootState.UNRESOLVED,
                    IntervalBounds(bracket_lower, bracket_upper),
                    None,
                    "ROOT_BISECTION_COMPUTATION_BUDGET_EXHAUSTED",
                )
            midpoint = bracket_lower + 0.5 * (
                bracket_upper - bracket_lower
            )
            middle_value = self.contact_plane_value(
                phase_lower=midpoint,
                phase_upper=midpoint,
                **plane_arguments,
            )
            middle_sign = self._strict_sign(middle_value)
            iterations += 1
            if middle_sign == 0:
                predecessor = float(
                    np.nextafter(midpoint, bracket_lower)
                )
                successor = float(
                    np.nextafter(midpoint, bracket_upper)
                )
                predecessor_value = self.contact_plane_value(
                    phase_lower=predecessor,
                    phase_upper=predecessor,
                    **plane_arguments,
                )
                successor_value = self.contact_plane_value(
                    phase_lower=successor,
                    phase_upper=successor,
                    **plane_arguments,
                )
                if (
                    self._strict_sign(predecessor_value)
                    != lower_sign
                    or self._strict_sign(successor_value)
                    != upper_sign
                ):
                    return IntervalRootClassification(
                        IntervalRootState.UNRESOLVED,
                        IntervalBounds(bracket_lower, bracket_upper),
                        None,
                        "ROOT_NEIGHBOR_SIGN_UNRESOLVED",
                    )
                bracket_lower = predecessor
                bracket_upper = successor
                lower_value = predecessor_value
                upper_value = successor_value
                break
            if middle_sign == lower_sign:
                bracket_lower = midpoint
                lower_value = middle_value
            else:
                bracket_upper = midpoint
                upper_value = middle_value
            if iterations in _EARLY_TRIANGLE_ROOT_BRACKET_CHECK_ITERATIONS:
                bracket_edges = self.contact_triangle_edge_halfspaces(
                    phase_lower=bracket_lower,
                    phase_upper=bracket_upper,
                    **plane_arguments,
                )
                if any(edge.strictly_negative for edge in bracket_edges):
                    return IntervalRootClassification(
                        IntervalRootState.CERTIFIED_FREE,
                        IntervalBounds(bracket_lower, bracket_upper),
                        None,
                        "UNIQUE_PLANE_ROOT_BRACKET_STRICTLY_OUTSIDE_TRIANGLE",
                    )

        root_row = self.contact_predicates(
            phase_lower=bracket_lower,
            phase_upper=bracket_upper,
            **arguments,
        )
        if any(
            edge.strictly_negative
            for edge in root_row.triangle_edge_halfspaces
        ):
            return IntervalRootClassification(
                IntervalRootState.CERTIFIED_FREE,
                IntervalBounds(bracket_lower, bracket_upper),
                None,
                "UNIQUE_PLANE_ROOT_STRICTLY_OUTSIDE_TRIANGLE",
            )
        if not all(
            edge.strictly_positive
            for edge in root_row.triangle_edge_halfspaces
        ):
            return IntervalRootClassification(
                IntervalRootState.UNRESOLVED,
                IntervalBounds(bracket_lower, bracket_upper),
                None,
                "TRIANGLE_BOUNDARY_NOT_STRICTLY_INTERIOR",
            )
        equation_sha256, feature_identity_sha256 = (
            _implicit_root_identities(
                hand_model=self.hand_model,
                link_name=link_name,
                q_start=q_start,
                direction=direction,
                base_transform=base_transform,
                witness_point_local_m=witness_point_local_m,
                pad_triangle_local_m=pad_triangle_local_m,
                object_triangle_m=object_triangle_m,
            )
        )
        implicit_root = CertifiedImplicitRoot(
            method_id=IMPLICIT_ROOT_METHOD_ID,
            equation_sha256=equation_sha256,
            feature_identity_sha256=feature_identity_sha256,
            feature_type=IMPLICIT_ROOT_FEATURE_TYPE,
            isolating_interval=IntervalBounds(
                bracket_lower, bracket_upper
            ),
            value_at_lower=lower_value,
            value_at_upper=upper_value,
            derivative=root_row.plane_derivative,
            uniqueness_proven=True,
            display_approximation=(
                bracket_lower + 0.5 * (bracket_upper - bracket_lower)
            ),
            display_approximation_role=DISPLAY_APPROXIMATION_ROLE,
        )
        certificate = IntervalTransverseRootCertificate(
            implicit_root=implicit_root,
            triangle_edge_halfspaces=root_row.triangle_edge_halfspaces,
            pad_approach=root_row.pad_approach,
            path_local_free_side_approach=(
                root_row.object_plane_transversality
            ),
            object_source_winding_free_side_sign=lower_sign,
            position_object_m=root_row.position_object_m,
            bisection_iterations=iterations,
            method_id=METHOD_ID,
            decimal_precision=self.options.decimal_precision,
        )
        if (
            root_row.pad_approach.strictly_positive
            and root_row.object_plane_transversality.strictly_positive
        ):
            root_state = (
                IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT
            )
            root_reason = (
                "STRICT_IVT_MONOTONE_INTERIOR_PAD_DIRECTIONAL_TRANSVERSE_ROOT"
            )
        elif root_row.pad_approach.strictly_negative:
            root_state = (
                IntervalRootState.CERTIFIED_PAD_DIRECTION_REJECTED_TRANSVERSE_ROOT
            )
            root_reason = (
                "STRICT_IVT_MONOTONE_INTERIOR_PAD_REVERSE_TRANSVERSE_ROOT"
            )
        else:
            return IntervalRootClassification(
                IntervalRootState.UNRESOLVED,
                IntervalBounds(bracket_lower, bracket_upper),
                None,
                "CONTACT_DIRECTION_SIGN_UNRESOLVED",
            )
        return IntervalRootClassification(
            root_state,
            searched,
            certificate,
            root_reason,
        )


__all__ = [
    "CertifiedImplicitRoot",
    "BATCH_POINT_VELOCITY_VECTOR_METHOD_ID",
    "COMPILED_POINT_PLANE_BINDING_CACHE_CAPACITY",
    "DISPLAY_APPROXIMATION_ROLE",
    "DirectedIntervalKinematics",
    "INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID",
    "INTERVAL_RIGID_TRANSFORM_METHOD_ID",
    "IMPLICIT_ROOT_FEATURE_TYPE",
    "IMPLICIT_ROOT_METHOD_ID",
    "IntervalArithmeticOptions",
    "IntervalBounds",
    "IntervalContactPredicates",
    "IntervalGeometricJacobian",
    "IntervalKinematicsError",
    "IntervalLinkTransformCache",
    "IntervalPointMotion",
    "IntervalPointVelocityVectorBatch",
    "IntervalPlaneRootClassification",
    "IntervalPlaneRootState",
    "IntervalRigidTransform",
    "IntervalRootClassification",
    "IntervalRootState",
    "IntervalTransversePlaneRoot",
    "IntervalTransverseRootCertificate",
    "METHOD_ID",
]
