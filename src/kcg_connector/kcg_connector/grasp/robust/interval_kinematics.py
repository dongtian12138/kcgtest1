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
import hashlib
import json
import math
from typing import Sequence

from mpmath.ctx_iv import MPIntervalContext
import numpy as np

from kcg_connector.grasp.robust.hand_model import (
    JointSpec,
    ThreeFingerHandModel,
)


METHOD_ID = "MPMATH_DIRECTED_INTERVAL_SECOND_ORDER_URDF_JET_V1"
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


@dataclass(frozen=True)
class _Jet:
    value: object
    first: object
    second: object


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

    def _identity(self) -> list[list[_Jet]]:
        return [
            [
                self._constant(1.0 if row == column else 0.0)
                for column in range(4)
            ]
            for row in range(4)
        ]

    def _matrix_from_float(self, value: np.ndarray) -> list[list[_Jet]]:
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
        return [
            [self._constant(float(matrix[row, column])) for column in range(4)]
            for row in range(4)
        ]

    def _matmul(
        self, first: list[list[_Jet]], second: list[list[_Jet]]
    ) -> list[list[_Jet]]:
        result: list[list[_Jet]] = []
        for row in range(4):
            output_row: list[_Jet] = []
            for column in range(4):
                value = self._constant(0.0)
                for inner in range(4):
                    value = _add(
                        value, _mul(first[row][inner], second[inner][column])
                    )
                output_row.append(value)
            result.append(output_row)
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
        result = self._identity()
        rotation = self._rpy_rotation(joint.origin_rpy_rad)
        for row in range(3):
            for column in range(3):
                result[row][column] = rotation[row][column]
            result[row][3] = self._constant(joint.origin_xyz_m[row])
        return result

    def _axis_rotation(
        self, axis: Sequence[float], angle: _Jet
    ) -> list[list[_Jet]]:
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
        return tuple(names)

    def _link_transform(
        self,
        *,
        link_name: str,
        q_start: np.ndarray,
        direction: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        base_transform: np.ndarray,
    ) -> list[list[_Jet]]:
        q_start = np.asarray(q_start, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
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
        independent: dict[str, _Jet] = {}
        for name, start, rate in zip(
            self.hand_model.independent_joint_names, q_start, direction
        ):
            independent[name] = _add(
                self._constant(float(start)),
                _mul(self._constant(float(rate)), phase),
            )
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
                source, multiplier, offset = self._affine_map(
                    name, affine_cache, set()
                )
                position = _add(
                    _mul(self._constant(multiplier), independent[source]),
                    self._constant(offset),
                )
            else:
                position = self._constant(0.0)
            transform = self._matmul(
                transform, self._origin_transform(joint)
            )
            transform = self._matmul(
                transform, self._motion_transform(joint, position)
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
    ) -> IntervalPointMotion:
        """Enclose one exact-FK point path and its first two derivatives."""

        transform = self._link_transform(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            base_transform=base_transform,
        )
        point = self._point_jet(transform, point_local_m)
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
        )
        point = self._point_jet(transform, witness_point_local_m)

        pad_vertices = tuple(
            self._constant_vector(row) for row in pad_triangle
        )
        pad_edge_one = tuple(
            _sub(second, first)
            for first, second in zip(pad_vertices[0], pad_vertices[1])
        )
        pad_edge_two = tuple(
            _sub(third, first)
            for first, third in zip(pad_vertices[0], pad_vertices[2])
        )
        pad_edge_one_object = self._vector_jet(transform, pad_edge_one)
        pad_edge_two_object = self._vector_jet(transform, pad_edge_two)
        pad_area_object = self._cross(
            pad_edge_one_object, pad_edge_two_object
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
        }
        searched = IntervalBounds(phase_lower, phase_upper)
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

        lower_row = self.contact_predicates(
            phase_lower=phase_lower,
            phase_upper=phase_lower,
            **arguments,
        )
        upper_row = self.contact_predicates(
            phase_lower=phase_upper,
            phase_upper=phase_upper,
            **arguments,
        )
        lower_sign = self._strict_sign(lower_row.plane_value)
        upper_sign = self._strict_sign(upper_row.plane_value)
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
            middle_row = self.contact_predicates(
                phase_lower=midpoint,
                phase_upper=midpoint,
                **arguments,
            )
            middle_sign = self._strict_sign(middle_row.plane_value)
            iterations += 1
            if middle_sign == 0:
                predecessor = float(
                    np.nextafter(midpoint, bracket_lower)
                )
                successor = float(
                    np.nextafter(midpoint, bracket_upper)
                )
                predecessor_row = self.contact_predicates(
                    phase_lower=predecessor,
                    phase_upper=predecessor,
                    **arguments,
                )
                successor_row = self.contact_predicates(
                    phase_lower=successor,
                    phase_upper=successor,
                    **arguments,
                )
                if (
                    self._strict_sign(predecessor_row.plane_value)
                    != lower_sign
                    or self._strict_sign(successor_row.plane_value)
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
                lower_row = predecessor_row
                upper_row = successor_row
                break
            if middle_sign == lower_sign:
                bracket_lower = midpoint
                lower_row = middle_row
            else:
                bracket_upper = midpoint
                upper_row = middle_row

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
            value_at_lower=lower_row.plane_value,
            value_at_upper=upper_row.plane_value,
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
    "DISPLAY_APPROXIMATION_ROLE",
    "DirectedIntervalKinematics",
    "IMPLICIT_ROOT_FEATURE_TYPE",
    "IMPLICIT_ROOT_METHOD_ID",
    "IntervalArithmeticOptions",
    "IntervalBounds",
    "IntervalContactPredicates",
    "IntervalKinematicsError",
    "IntervalPointMotion",
    "IntervalRootClassification",
    "IntervalRootState",
    "IntervalTransverseRootCertificate",
    "METHOD_ID",
]
