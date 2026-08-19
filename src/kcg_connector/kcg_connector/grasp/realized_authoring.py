'''Pure planning/validation helpers for realized physical-grasp authoring.

Everything here is dependency-free (numpy + stdlib only): no Omni/pxr/
isaacsim imports and no RNG.  The column-vector vs Gf row-vector boundary is
handled exclusively by the runner, which transposes these matrices next to
the USD calls.
'''

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

HAND_FINGER_JOINT_INDICES = (1, 2, 3)
SPREAD_JOINT_INDEX = 0


@dataclass(frozen=True)
class RandomizationValidationConfig:
    '''Frozen SIM_TUNING_ONLY validation gates for realized arm targets.

    These bounds are fail-closed validation gates only: they do not enter the
    realized payload/hash and they do not depend on the method.
    '''

    threshold_label: str = "SIM_TUNING_ONLY"
    maximum_arm_joint_delta_rad: float = 0.05
    maximum_fk_position_error_m: float = 1.0e-7
    maximum_fk_rotation_error_rad: float = 1.0e-7

    def __post_init__(self) -> None:
        if self.threshold_label != "SIM_TUNING_ONLY":
            raise ValueError(
                "randomization validation gates must remain SIM_TUNING_ONLY"
            )
        for name in (
            "maximum_arm_joint_delta_rad",
            "maximum_fk_position_error_m",
            "maximum_fk_rotation_error_rad",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{label} must be one finite {length}-vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def minimum_jerk_blend(fraction: float) -> float:
    value = float(fraction)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    return value ** 3 * (10.0 + value * (-15.0 + 6.0 * value))


def compose_loose_plug_transform(
    nominal_origin_xyz: Sequence[float],
    x_offset_m: float,
    y_offset_m: float,
    yaw_deg: float,
) -> np.ndarray:
    '''Compose the loose-plug root transform as a column-vector 4x4.

    Translation is the nominal origin plus the realized world X/Y offsets;
    Z is preserved exactly.  The realized yaw rotates only about the root
    local Z axis.  Returns a finite 4x4 numpy array in column-vector
    convention; the runner transposes it into Gf's row-vector convention.
    '''
    origin = _finite_vector(nominal_origin_xyz, 3, "nominal_origin_xyz")
    for name, value in (("x_offset_m", x_offset_m), ("y_offset_m", y_offset_m),
                        ("yaw_deg", yaw_deg)):
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite and non-bool")
    yaw = math.radians(float(yaw_deg))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = (
        origin[0] + float(x_offset_m),
        origin[1] + float(y_offset_m),
        origin[2],
    )
    return result


def _contains_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(
            isinstance(item, (bool, np.bool_)) for item in value
        )
    if isinstance(value, np.ndarray):
        return any(
            isinstance(item, (bool, np.bool_))
            for item in np.ravel(value.astype(object)).tolist()
        )
    return False


def float32_readback_evidence(
    intended: Any,
    readback: Any,
    *,
    label: str,
) -> dict[str, Any]:
    '''Verify that a readback equals the float32 storage of the intended value.

    USD Float/Point3f attributes store float32, so an exact float64 comparison
    against the Python intended value is wrong.  This helper quantizes the
    intended value to float32 first, then requires the readback to carry the
    SAME float32 representation (compared bitwise).  Returns JSON-safe
    evidence; it never raises for a mismatching readback, only for malformed
    inputs.
    '''
    if _contains_bool(intended) or _contains_bool(readback):
        raise ValueError(f"{label}: values must not be booleans")
    intended_array = np.atleast_1d(
        np.asarray(intended, dtype=np.float64)
    )
    readback_array = np.atleast_1d(
        np.asarray(readback, dtype=np.float64)
    )
    if intended_array.shape != readback_array.shape:
        raise ValueError(f"{label}: intended/readback shapes differ")
    if not np.all(np.isfinite(intended_array)) or not np.all(
        np.isfinite(readback_array)
    ):
        raise ValueError(f"{label}: values must be finite")
    storage_expected = intended_array.astype(np.float32)
    readback_float32 = readback_array.astype(np.float32)
    same_representation = bool(
        np.array_equal(
            readback_float32.view(np.uint32),
            storage_expected.view(np.uint32),
        )
    )
    quantization_error = (
        storage_expected.astype(np.float64) - intended_array
    )
    storage_error = (
        readback_array - storage_expected.astype(np.float64)
    )
    scalar_shape = intended_array.shape == (1,)
    return {
        "label": label,
        "shape": list(intended_array.shape),
        "storage_type": "float32",
        "intended": (
            float(intended_array[0])
            if scalar_shape
            else [float(value) for value in intended_array]
        ),
        "storage_expected": (
            float(storage_expected[0])
            if scalar_shape
            else [float(value) for value in storage_expected]
        ),
        "readback": (
            float(readback_array[0])
            if scalar_shape
            else [float(value) for value in readback_array]
        ),
        "quantization_error": (
            float(quantization_error[0])
            if scalar_shape
            else [float(value) for value in quantization_error]
        ),
        "storage_error": (
            float(storage_error[0])
            if scalar_shape
            else [float(value) for value in storage_error]
        ),
        "maximum_quantization_error": float(
            np.max(np.abs(quantization_error))
        ),
        "maximum_storage_error": float(np.max(np.abs(storage_error))),
        "verified": same_representation,
    }


def _rotation_error_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = max(
        -1.0, min(1.0, (float(np.trace(relative)) - 1.0) * 0.5)
    )
    return math.acos(cosine)


def validate_offset_arm_targets(
    nominal_arm_rad: Sequence[float],
    realized_arm_rad: Sequence[float],
    config: RandomizationValidationConfig,
    fk: Callable[[Sequence[float]], Any],
    *,
    requested_position_m: Sequence[float],
) -> dict[str, float]:
    '''Validate one realized arm target against its requested TCP.

    Requires: seven finite joints, q7 preserved, bounded per-joint delta, a
    bounded FK position residual between the realized joints and the
    REQUESTED TCP position (the nominal FK TCP shifted by the realized world
    X/Y), and a bounded FK rotation residual against the nominal rotation.
    Returns the residual evidence as a JSON-safe dict.
    '''
    nominal = _finite_vector(nominal_arm_rad, 7, "nominal_arm_rad")
    realized = _finite_vector(realized_arm_rad, 7, "realized_arm_rad")
    requested = _finite_vector(
        requested_position_m, 3, "requested_position_m"
    )
    delta = realized - nominal
    q7_delta = abs(float(delta[6]))
    if q7_delta > 1.0e-9:
        raise ValueError(f"realized arm target moved q7 by {q7_delta} rad")
    maximum_delta = float(np.max(np.abs(delta)))
    if maximum_delta > config.maximum_arm_joint_delta_rad:
        raise ValueError(
            "realized arm target exceeds maximum_arm_joint_delta_rad: "
            f"{maximum_delta} > {config.maximum_arm_joint_delta_rad}"
        )
    nominal_tcp = np.asarray(fk(tuple(float(v) for v in nominal)),
                             dtype=np.float64)
    realized_tcp = np.asarray(fk(tuple(float(v) for v in realized)),
                              dtype=np.float64)
    position_error = float(
        np.linalg.norm(realized_tcp[:3, 3] - requested)
    )
    if position_error > config.maximum_fk_position_error_m:
        raise ValueError(
            "realized arm target FK position residual is unbounded: "
            f"{position_error} > {config.maximum_fk_position_error_m}"
        )
    rotation_error = _rotation_error_rad(
        nominal_tcp[:3, :3], realized_tcp[:3, :3]
    )
    if rotation_error > config.maximum_fk_rotation_error_rad:
        raise ValueError(
            "realized arm target FK rotation residual is unbounded: "
            f"{rotation_error} > {config.maximum_fk_rotation_error_rad}"
        )
    return {
        "maximum_joint_delta_rad": maximum_delta,
        "q7_delta_rad": q7_delta,
        "position_residual_m": position_error,
        "rotation_residual_rad": rotation_error,
    }


def closure_onset_plan(
    closure_steps: int,
    onset_delay_steps: Sequence[int],
) -> tuple[int, tuple[tuple[float | None, ...], ...]]:
    '''Plan a staggered-onset synchronous closure over the three finger
    joints (hand indices 1, 2, 3); the f1j1 spread joint is not part of the
    plan.  Each channel holds open for its onset delay and then plays the
    complete nominal minimum-jerk closure duration, so the total length is
    closure_steps + max(delay).  None means "hold open" for that channel.
    '''
    if (
        isinstance(closure_steps, bool)
        or not isinstance(closure_steps, int)
        or closure_steps < 1
    ):
        raise ValueError("closure_steps must be a positive integer")
    delays = tuple(onset_delay_steps)
    if len(delays) != 3:
        raise ValueError("onset_delay_steps must contain three values")
    for value in delays:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "onset_delay_steps must contain non-negative integers"
            )
    total_steps = closure_steps + max(delays)
    plan = []
    for step in range(total_steps):
        row = []
        for delay in delays:
            progress = step - delay
            if progress < 0:
                row.append(None)
            else:
                # A channel whose onset is earlier than the maximum delay
                # finishes its nominal duration before the loop ends and
                # then simply holds at the closed target.
                fraction = min(
                    1.0, float(progress + 1) / float(closure_steps)
                )
                row.append(minimum_jerk_blend(fraction))
        plan.append(tuple(row))
    return total_steps, tuple(plan)


def synchronous_contact_stability(
    contact_order: Sequence[str],
    final_states: Mapping[str, str],
) -> bool:
    '''Synchronous closure is stable only when every finger appeared in the
    contact order AND every finger's final detector state is still
    CONTACT_CONFIRMED.  The historical contact order alone must never mask a
    terminal SLIP/FAILED state.
    '''
    if set(contact_order) != {"f1", "f2", "f3"}:
        return False
    for finger in ("f1", "f2", "f3"):
        if final_states.get(finger) != "CONTACT_CONFIRMED":
            return False
    return True
