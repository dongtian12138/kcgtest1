"""Deterministic bounded IK for the aggregate robot hand-base target.

This module owns the small numerical kernel shared by the legacy route builder
and CARTS-Grasp V2.  It deliberately has no dependency on candidate contracts,
collision aggregation, closure prediction, or continuous collision checking.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


EXPECTED_HAND_BASE_LINK = "handbase_link"
EXPECTED_SOLVER_METHOD = "SCIPY_DOGBOX_BOUNDED_POSE_ANALYTIC_JACOBIAN_V3"
EXPECTED_SEED_RULE = "HOME_THEN_UNSCRAMBLED_SOBOL_INTERIOR_JOINT_LIMITS"
EXPECTED_FEASIBLE_CHOICE_RULE = "FIRST_FEASIBLE_IN_FIXED_SEED_ORDER"


class CandidateJointRouteError(ValueError):
    """Raised when bounded IK inputs or convergence fail closed."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("candidate joint-route error fields cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


@dataclass(frozen=True)
class BoundedIKSettings:
    sobol_point_count: int
    sobol_interior_lower_fraction: float
    sobol_interior_upper_fraction: float
    orientation_residual_length_scale_m_per_rad: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    function_tolerance: float
    step_tolerance: float
    gradient_tolerance: float
    maximum_function_evaluations: int


def bounded_ik_settings(value: Mapping[str, Any]) -> BoundedIKSettings:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CandidateJointRouteError(
            "MAPPING_REQUIRED", "solver must be a string-keyed mapping"
        )
    expected = {
        "method",
        "pregrasp_seed_rule",
        "pregrasp_sobol_point_count",
        "sobol_interior_lower_fraction",
        "sobol_interior_upper_fraction",
        "feasible_choice_rule",
        "orientation_residual_length_scale_m_per_rad",
        "position_tolerance_m",
        "orientation_tolerance_rad",
        "function_tolerance",
        "step_tolerance",
        "gradient_tolerance",
        "maximum_function_evaluations",
    }
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise CandidateJointRouteError(
            "SCHEMA_MISMATCH", f"solver missing={missing}, extra={extra}"
        )
    sobol_count = int(value["pregrasp_sobol_point_count"])
    lower_fraction = float(value["sobol_interior_lower_fraction"])
    upper_fraction = float(value["sobol_interior_upper_fraction"])
    numerical = tuple(
        float(value[name])
        for name in (
            "orientation_residual_length_scale_m_per_rad",
            "position_tolerance_m",
            "orientation_tolerance_rad",
            "function_tolerance",
            "step_tolerance",
            "gradient_tolerance",
        )
    )
    evaluations = int(value["maximum_function_evaluations"])
    if (
        value["method"] != EXPECTED_SOLVER_METHOD
        or value["pregrasp_seed_rule"] != EXPECTED_SEED_RULE
        or value["feasible_choice_rule"] != EXPECTED_FEASIBLE_CHOICE_RULE
        or sobol_count <= 0
        or sobol_count & (sobol_count - 1)
        or not (0.0 < lower_fraction < upper_fraction < 1.0)
        or any(not math.isfinite(item) or item <= 0.0 for item in numerical)
        or evaluations <= 0
    ):
        raise CandidateJointRouteError(
            "SOLVER_CONFIGURATION_INVALID",
            "bounded deterministic IK settings are incomplete",
        )
    return BoundedIKSettings(
        sobol_count,
        lower_fraction,
        upper_fraction,
        *numerical,
        evaluations,
    )


def _pose_error(
    model: object,
    arm_positions: Sequence[float],
    hand_positions: Sequence[float],
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    complete = tuple(float(value) for value in (*arm_positions, *hand_positions))
    actual = model.forward_kinematics(complete)[EXPECTED_HAND_BASE_LINK]
    position = actual[:3, 3] - target[:3, 3]
    orientation = Rotation.from_matrix(
        target[:3, :3].T @ actual[:3, :3]
    ).as_rotvec()
    return position, orientation


def _skew_symmetric(vector: Sequence[float]) -> np.ndarray:
    x, y, z = (float(value) for value in vector)
    return np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=np.float64,
    )


def _so3_left_jacobian_inverse(rotation_vector: Sequence[float]) -> np.ndarray:
    """Map a left SO(3) perturbation to the derivative of its log vector."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    skew = _skew_symmetric(vector)
    skew_squared = skew @ skew
    if angle < 1.0e-4:
        coefficient = 1.0 / 12.0 + angle**2 / 720.0 + angle**4 / 30240.0
    else:
        coefficient = (
            1.0 - angle / (2.0 * math.tan(0.5 * angle))
        ) / angle**2
    return np.eye(3, dtype=np.float64) - 0.5 * skew + coefficient * skew_squared


def solve_bounded_target(
    *,
    model: object,
    hand_positions: Sequence[float],
    target: np.ndarray,
    seeds: Sequence[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    settings: BoundedIKSettings,
    label: str,
) -> tuple[tuple[float, ...], float, float, int]:
    best_failure: tuple[float, float] | None = None
    for seed_index, seed in enumerate(seeds):
        initial = np.asarray(seed, dtype=np.float64)
        if initial.shape != (7,) or not np.all(np.isfinite(initial)):
            raise CandidateJointRouteError(
                "IK_SEED_INVALID", f"{label}:{seed_index}"
            )
        initial = np.clip(initial, lower, upper)
        cached_arm_positions: np.ndarray | None = None
        cached_position: np.ndarray | None = None
        cached_orientation: np.ndarray | None = None

        def pose_error(
            arm_positions: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            nonlocal cached_arm_positions, cached_position, cached_orientation
            if (
                cached_arm_positions is None
                or not np.array_equal(arm_positions, cached_arm_positions)
            ):
                cached_position, cached_orientation = _pose_error(
                    model, arm_positions, hand_positions, target
                )
                cached_arm_positions = np.array(
                    arm_positions, dtype=np.float64, copy=True
                )
            assert cached_position is not None and cached_orientation is not None
            return cached_position, cached_orientation

        def residual(arm_positions: np.ndarray) -> np.ndarray:
            position, orientation = pose_error(arm_positions)
            return np.concatenate(
                (
                    position,
                    settings.orientation_residual_length_scale_m_per_rad
                    * orientation,
                )
            )

        def jacobian(arm_positions: np.ndarray) -> np.ndarray:
            complete = tuple(
                float(value) for value in (*arm_positions, *hand_positions)
            )
            _position, orientation = pose_error(arm_positions)
            geometric = np.asarray(
                model.geometric_jacobian(EXPECTED_HAND_BASE_LINK, complete),
                dtype=np.float64,
            )
            if geometric.shape[0] != 6 or geometric.shape[1] < 7:
                raise CandidateJointRouteError(
                    "IK_JACOBIAN_INVALID", f"{label}:{geometric.shape}"
                )
            result = np.empty((6, 7), dtype=np.float64)
            result[:3] = geometric[:3, :7]
            result[3:] = (
                settings.orientation_residual_length_scale_m_per_rad
                * _so3_left_jacobian_inverse(orientation)
                @ target[:3, :3].T
                @ geometric[3:, :7]
            )
            return result

        try:
            result = least_squares(
                residual,
                initial,
                jac=jacobian,
                bounds=(lower, upper),
                method="dogbox",
                ftol=settings.function_tolerance,
                xtol=settings.step_tolerance,
                gtol=settings.gradient_tolerance,
                max_nfev=settings.maximum_function_evaluations,
                x_scale="jac",
            )
        except (FloatingPointError, ValueError):
            best_failure = best_failure or (math.inf, math.inf)
            continue
        position, orientation = _pose_error(
            model, result.x, hand_positions, target
        )
        position_error = float(np.linalg.norm(position))
        orientation_error = float(np.linalg.norm(orientation))
        score = (position_error, orientation_error)
        if best_failure is None or score < best_failure:
            best_failure = score
        complete = tuple(float(value) for value in (*result.x, *hand_positions))
        if (
            result.success
            and position_error <= settings.position_tolerance_m
            and orientation_error <= settings.orientation_tolerance_rad
            and model.within_joint_limits(complete)
        ):
            return (
                tuple(float(value) for value in result.x),
                position_error,
                orientation_error,
                seed_index,
            )
    raise CandidateJointRouteError(
        "IK_TARGET_UNREACHABLE",
        f"{label}:best_position_orientation_error={best_failure}",
    )


def pregrasp_seeds(
    *,
    home_arm: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: BoundedIKSettings,
) -> tuple[np.ndarray, ...]:
    exponent = int(math.log2(settings.sobol_point_count))
    unit = qmc.Sobol(d=7, scramble=False).random_base2(m=exponent)
    span = (
        settings.sobol_interior_lower_fraction
        + (
            settings.sobol_interior_upper_fraction
            - settings.sobol_interior_lower_fraction
        )
        * unit
    )
    rows = [np.array(home_arm, dtype=np.float64, copy=True)]
    rows.extend(lower + row * (upper - lower) for row in span)
    return tuple(rows)


def solve_bounded_hand_base_ik(
    solver_settings: Mapping[str, Any],
    *,
    model: object,
    hand_positions: Sequence[float],
    target_world_from_hand_base: Sequence[Sequence[float]],
    seed_arm_positions: Sequence[Sequence[float]] | None = None,
    label: str = "V2_TARGET",
) -> tuple[tuple[float, ...], float, float, int]:
    """Solve one hand-base target without loading legacy collision modules."""

    settings = bounded_ik_settings(solver_settings)
    lower, upper = model.joint_limit_vectors()
    arm_lower = np.asarray(lower[:7], dtype=np.float64)
    arm_upper = np.asarray(upper[:7], dtype=np.float64)
    seeds = (
        pregrasp_seeds(
            home_arm=np.zeros(7, dtype=np.float64),
            lower=arm_lower,
            upper=arm_upper,
            settings=settings,
        )
        if seed_arm_positions is None
        else tuple(
            np.asarray(row, dtype=np.float64) for row in seed_arm_positions
        )
    )
    target = np.asarray(target_world_from_hand_base, dtype=np.float64)
    if target.shape != (4, 4) or not np.all(np.isfinite(target)):
        raise CandidateJointRouteError("IK_TARGET_INVALID", label)
    return solve_bounded_target(
        model=model,
        hand_positions=hand_positions,
        target=target,
        seeds=seeds,
        lower=arm_lower,
        upper=arm_upper,
        settings=settings,
        label=label,
    )


__all__ = [
    "BoundedIKSettings",
    "CandidateJointRouteError",
    "bounded_ik_settings",
    "pregrasp_seeds",
    "solve_bounded_target",
    "solve_bounded_hand_base_ik",
]
