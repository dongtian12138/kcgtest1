"""Bounded six-axis wrist-FT compliant insertion policy.

This pure module has no simulator API and its observation schema cannot carry
object truth, contact normals, collider paths, or contact points.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from kcg_connector.contact_classifier import ContactClass, classify_contact
from kcg_connector.virtual_wrist_ft_runtime import transform_wrench_to_task


SCHEMA_VERSION = "kcg_d38999_compliant_insertion_v2"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/d38999_compliant_insertion_v2.yaml"


class InsertionState(str, Enum):
    PREINSERT_READY = "PREINSERT_READY"
    GUARDED_APPROACH = "GUARDED_APPROACH"
    FIRST_CONTACT = "FIRST_CONTACT"
    CONTACT_HOLD = "CONTACT_HOLD"
    CONTACT_CLASSIFY = "CONTACT_CLASSIFY"
    CONTACT_UNLOAD = "CONTACT_UNLOAD"
    UNLOADED_CENTERING = "UNLOADED_CENTERING"
    ACTIVE_PROBE = "ACTIVE_PROBE"
    COMPLIANT_CENTERING = "COMPLIANT_CENTERING"
    TILT_CORRECTION = "TILT_CORRECTION"
    BOUNDED_RZ_SEARCH = "BOUNDED_RZ_SEARCH"
    INSERT_ADVANCE = "INSERT_ADVANCE"
    INSERT_VERIFY = "INSERT_VERIFY"
    SEATED = "SEATED"
    SUCCESS = "SUCCESS"
    BACKOFF = "BACKOFF"
    REOBSERVE = "REOBSERVE"
    REALIGN = "REALIGN"
    RETRY = "RETRY"
    SAFE_ABORT = "SAFE_ABORT"


@dataclass(frozen=True)
class InsertionObservation:
    timestamp_s: float
    sample_age_s: float
    wrench_assembly: tuple[float, float, float, float, float, float]
    tcp_position_assembly_m: tuple[float, float, float]
    tcp_rotation_vector_assembly_rad: tuple[float, float, float]
    vision_control_authorized: bool
    synchronized_capture: bool
    ft_valid: bool
    ft_tared: bool
    payload_compensated: bool


@dataclass(frozen=True)
class ControllerState:
    phase: InsertionState = InsertionState.PREINSERT_READY
    step_count: int = 0
    retry_count: int = 0
    filtered_wrench: tuple[float, ...] = (0.0,) * 6
    last_twist_command: tuple[float, ...] = (0.0,) * 6
    start_position_m: tuple[float, float, float] | None = None
    progress_window_origin_m: float = 0.0
    progress_window_step: int = 0
    xy_search_offset_m: tuple[float, float] = (0.0, 0.0)
    rz_search_angle_rad: float = 0.0
    phase_step: int = 0
    probe_leg: int = 0
    probe_leg_step: int = 0
    probe_total_steps: int = 0
    probe_origin_xy_m: tuple[float, float] | None = None
    probe_origin_tilt_rad: tuple[float, float] | None = None
    probe_baseline_score: float = math.inf
    probe_scores: tuple[float, ...] = (math.inf,) * 8
    probe_selected_xy: tuple[float, float] = (0.0, 0.0)
    probe_selected_tilt: tuple[float, float] = (0.0, 0.0)
    retained_probe_xy: tuple[float, float] = (0.0, 0.0)
    # Preserve the first force-probed angular direction across re-contact.
    # This is controller history, not object-pose or contact-geometry truth.
    retained_probe_tilt: tuple[float, float] = (0.0, 0.0)
    cumulative_unloaded_tilt_rad: float = 0.0
    probe_selection_history: tuple[tuple[float, float, float, float], ...] = ()
    probe_direction_order: tuple[int, ...] = tuple(range(8))
    latched_contact_class: str = ContactClass.NO_CONTACT.value
    latched_contact_score: float = math.inf
    latched_contact_wrench: tuple[float, ...] = (0.0,) * 6
    contact_unload_origin_z_m: float | None = None
    unloaded_centering_origin_xy_m: tuple[float, float] | None = None
    unloaded_centering_origin_tilt_rad: tuple[float, float] | None = None
    contact_realign_count: int = 0
    abort_reason: str | None = None


@dataclass(frozen=True)
class InsertionCommand:
    next_state: ControllerState
    twist_assembly: tuple[float, float, float, float, float, float]
    contact_class: ContactClass
    stop_motion: bool
    request_reobserve: bool
    status: str


def load_compliant_insertion_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Mapping[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if document.get("enabled") is not True:
        raise ValueError("compliant insertion config is disabled")
    boundaries = document["boundaries"]
    required_forbidden = {"object_truth", "physx_contact_normal", "collider_identity", "contact_point_truth", "penetration_depth_truth", "fingertip_tactile"}
    if set(boundaries["forbidden_inputs"]) != required_forbidden:
        raise ValueError("forbidden compliant insertion inputs changed")
    if boundaries["root_joint_torque_name"] != "root_joint_torque_proxy":
        raise ValueError("root joint torque must remain a proxy")
    safety = document["safety"]
    if not math.isclose(float(safety["soft_fraction"]), 0.70, abs_tol=1e-12):
        raise ValueError("soft safety gate must remain 70% of hard gate")
    response = document["local_response_model"]
    matrix = np.asarray(
        response["response_matrix_wrench_per_motion"], dtype=np.float64
    )
    if (
        response["enabled"] is not True
        or response["motion_order"] != ["X", "Y", "Rx", "Ry"]
        or response["wrench_rows"] != ["Fx", "Fy", "Mx", "My"]
        or matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("local response model is invalid")
    if (
        float(response["effective_length_m"]) <= 0.0
        or float(response["damping"]) <= 0.0
        or float(response["angular_regularization_weight"]) < 1.0
        or float(response["correction_rate_hz"]) <= 0.0
        or float(response["maximum_model_tilt_speed_rad_s"]) <= 0.0
    ):
        raise ValueError("local response scaling and damping must be positive")
    probe = document["active_probe"]
    if (
        len(probe["sequence"]) != 16
        or float(probe["angular_amplitude_rad"]) <= 0.0
        or float(probe["angular_speed_rad_s"]) <= 0.0
        or float(probe["unloaded_tilt_distance_rad"]) <= 0.0
        or float(probe["unloaded_tilt_distance_rad"])
        > float(document["motion"]["maximum_search_angle_rad"])
        or float(probe["early_pair_selection_improvement_n"])
        < float(probe["minimum_score_improvement_n"])
        or not 0.0 < float(probe["early_return_fraction_of_soft"]) < 1.0
    ):
        raise ValueError("bounded 4D active-probe configuration is invalid")
    return document


def wrench_sensor_to_assembly(
    wrench_sensor, sensor_position_world, sensor_rotation_world,
    assembly_origin_world, assembly_rotation_world,
) -> np.ndarray:
    """Full wrench adjoint including the sensor-to-assembly lever arm."""
    return transform_wrench_to_task(
        wrench_sensor, sensor_position_world, sensor_rotation_world,
        assembly_origin_world, assembly_rotation_world,
    )


def full_seated_posthoc(
    body_depth_m: float,
    configured_insertion_depth_m: float,
    physically_guided: bool,
    *,
    depth_tolerance_m: float = 0.00010,
) -> bool:
    """Conservative posthoc seated gate; never a controller observation."""

    values = (body_depth_m, configured_insertion_depth_m, depth_tolerance_m)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if configured_insertion_depth_m <= 0.0 or depth_tolerance_m < 0.0:
        return False
    return bool(
        physically_guided
        and body_depth_m
        >= configured_insertion_depth_m - depth_tolerance_m
    )


def effective_lateral_posthoc(
    lateral_error_m: float,
    axis_error_rad: float,
    effective_length_m: float,
) -> float:
    """Worst-case guide offset used only by posthoc/Bcapture evaluation."""

    values = (lateral_error_m, axis_error_rad, effective_length_m)
    if not all(math.isfinite(float(value)) for value in values):
        return math.inf
    if lateral_error_m < 0.0 or effective_length_m < 0.0:
        return math.inf
    if abs(axis_error_rad) >= 0.5 * math.pi:
        return math.inf
    return float(
        lateral_error_m
        + effective_length_m * math.tan(abs(axis_error_rad))
    )


def _slew(target: np.ndarray, previous: np.ndarray, linear_accel: float, angular_accel: float, dt: float) -> np.ndarray:
    limit = np.asarray((linear_accel,) * 3 + (angular_accel,) * 3) * dt
    return previous + np.clip(target - previous, -limit, limit)


def response_model_twist(config: Mapping[str, Any], wrench) -> np.ndarray:
    """Bounded X/Y/Rx/Ry velocity from a measured local response matrix."""

    response = config["local_response_model"]
    matrix = np.asarray(
        response["response_matrix_wrench_per_motion"], dtype=np.float64
    )
    values = np.asarray(wrench, dtype=np.float64)
    if matrix.shape != (4, 4) or values.shape != (6,):
        raise ValueError("response model expects a 4x4 matrix and 6D wrench")
    effective_length = float(response["effective_length_m"])
    normalized_matrix = matrix.copy()
    # q = [x, y, L*rx, L*ry], w = [Fx, Fy, Mx/L, My/L].
    normalized_matrix[:, 2:] /= effective_length
    normalized_matrix[2:, :] /= effective_length
    normalized_wrench = np.asarray(
        (
            values[0],
            values[1],
            values[3] / effective_length,
            values[4] / effective_length,
        ),
        dtype=np.float64,
    )
    damping = float(response["damping"])
    regularization = np.diag(
        (
            1.0,
            1.0,
            float(response["angular_regularization_weight"]),
            float(response["angular_regularization_weight"]),
        )
    )
    delta_normalized = -np.linalg.solve(
        normalized_matrix.T @ normalized_matrix
        + damping * damping * regularization,
        normalized_matrix.T @ normalized_wrench,
    )
    correction_rate = float(response["correction_rate_hz"])
    velocity = correction_rate * np.asarray(
        (
            delta_normalized[0],
            delta_normalized[1],
            delta_normalized[2] / effective_length,
            delta_normalized[3] / effective_length,
        )
    )
    motion = config["motion"]
    result = np.zeros(6, dtype=np.float64)
    xy_limit = float(motion["maximum_xy_speed_m_s"])
    xy_norm = float(np.linalg.norm(velocity[:2]))
    result[:2] = (
        velocity[:2]
        if xy_norm <= xy_limit
        else velocity[:2] * (xy_limit / xy_norm)
    )
    tilt_limit = min(
        float(motion["maximum_tilt_speed_rad_s"]),
        float(response["maximum_model_tilt_speed_rad_s"]),
    )
    tilt_norm = float(np.linalg.norm(velocity[2:]))
    result[3:5] = (
        velocity[2:]
        if tilt_norm <= tilt_limit
        else velocity[2:] * (tilt_limit / tilt_norm)
    )
    return result


def _abort(state: ControllerState, reason: str, contact: ContactClass = ContactClass.UNKNOWN) -> InsertionCommand:
    next_state = replace(state, phase=InsertionState.SAFE_ABORT, abort_reason=reason, step_count=state.step_count + 1, last_twist_command=(0.0,) * 6)
    return InsertionCommand(next_state, (0.0,) * 6, contact, True, False, reason)


def _probe_command(config, state, position, rotation, filtered):
    """Run symmetric 4D X/Y/Rx/Ry probing without contact geometry input."""
    probe = config["active_probe"]
    origin_xy = position[:2] if state.probe_origin_xy_m is None else np.asarray(state.probe_origin_xy_m)
    origin_tilt = rotation[:2] if state.probe_origin_tilt_rad is None else np.asarray(state.probe_origin_tilt_rad)
    origin = np.concatenate((origin_xy, origin_tilt))
    amplitudes = np.asarray(
        (
            float(probe["amplitude_m"]),
            float(probe["amplitude_m"]),
            float(probe["angular_amplitude_rad"]),
            float(probe["angular_amplitude_rad"]),
        )
    )
    canonical_directions = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, -1.0),
        )
    )
    directions = canonical_directions[
        np.asarray(state.probe_direction_order, dtype=np.int64)
    ]
    # Outbound and return-to-origin alternate, giving genuinely symmetric
    # perturbations from the same measured TCP origin.
    outbound = state.probe_leg % 2 == 0
    direction_index = min(7, state.probe_leg // 2)
    scores = list(state.probe_scores)
    score = (
        abs(float(filtered[2]))
        + float(probe["lateral_weight"]) * float(np.linalg.norm(filtered[:2]))
        + float(probe["bending_weight_n_per_nm"])
        * float(np.linalg.norm(filtered[3:5]))
    )
    baseline_score = state.probe_baseline_score
    probe_leg = state.probe_leg
    early_scalars = np.asarray(
        (
            abs(float(filtered[2])),
            float(np.linalg.norm(filtered[:2])),
            float(np.linalg.norm(filtered[3:5])),
            abs(float(filtered[5])),
        )
    )
    hard = np.asarray(
        (
            float(config["safety"]["hard_axial_force_n"]),
            float(config["safety"]["hard_lateral_force_n"]),
            float(config["safety"]["hard_bending_moment_nm"]),
            float(config["safety"]["hard_torsional_moment_nm"]),
        )
    )
    early_return = bool(
        outbound
        and np.any(
            early_scalars
            >= float(config["safety"]["soft_fraction"])
            * float(probe["early_return_fraction_of_soft"])
            * hard
        )
    )
    if early_return:
        # Record the already-observed bad direction and return to the exact
        # probe origin before the formal soft gate is reached.
        scores[direction_index] = score - baseline_score
        outbound = False
        probe_leg += 1
        direction_index = min(7, probe_leg // 2)
    destination = origin + (
        directions[direction_index] * amplitudes if outbound else 0.0
    )
    current = np.concatenate((position[:2], rotation[:2]))
    delta = destination - current
    command = np.zeros(6)
    active_dimension = int(np.argmax(np.abs(directions[direction_index])))
    distance = abs(float(delta[active_dimension]))
    tolerance = (
        float(probe["position_tolerance_m"])
        if active_dimension < 2
        else float(probe["angular_position_tolerance_rad"])
    )
    speed = (
        float(probe["speed_m_s"])
        if active_dimension < 2
        else float(probe["angular_speed_rad_s"])
    )
    reached = distance <= tolerance
    if not reached:
        command_index = active_dimension if active_dimension < 2 else active_dimension + 1
        command[command_index] = math.copysign(speed, delta[active_dimension])
    leg_step = 1 if early_return else state.probe_leg_step + 1
    max_leg_steps = max(1, int(round(float(probe["maximum_leg_duration_s"]) * float(config["control_rate_hz"]))))
    timed_out = leg_step >= max_leg_steps
    if outbound and (reached or timed_out):
        scores[direction_index] = score - baseline_score
    elif not outbound and (reached or timed_out):
        baseline_score = score
    leg = probe_leg
    if reached or timed_out:
        leg += 1
        leg_step = 0
    selected = state.probe_selected_xy
    selected_tilt = state.probe_selected_tilt
    phase = InsertionState.ACTIVE_PROBE
    status = "ACTIVE_PROBE_EARLY_RETURN" if early_return else "ACTIVE_PROBE"
    if leg == 4 and all(math.isfinite(value) for value in scores[:2]):
        pair = np.asarray(scores[:2], dtype=np.float64)
        pair_best = int(np.argmin(pair))
        if (
            -float(pair[pair_best])
            >= float(probe["early_pair_selection_improvement_n"])
            and float(pair[pair_best ^ 1]) >= 0.0
        ):
            selected_direction = directions[pair_best]
            selected = tuple(float(value) for value in selected_direction[:2])
            selected_tilt = tuple(
                float(value) for value in selected_direction[2:]
            )
            phase = InsertionState.CONTACT_UNLOAD
            command[:] = 0.0
            status = "PROBE_EARLY_PAIR_SELECTED"
    if leg >= 16:
        best_index = int(np.argmin(np.asarray(scores)))
        improvement = float(-scores[best_index])
        if not math.isfinite(scores[best_index]) or improvement < float(probe["minimum_score_improvement_n"]):
            phase = InsertionState.BACKOFF
            command[:] = (0.0, 0.0, -float(config["recovery"]["backoff_speed_m_s"]), 0.0, 0.0, 0.0)
            status = "PROBE_NO_SAFE_IMPROVEMENT_BACKOFF"
        else:
            selected_direction = directions[best_index]
            selected = tuple(float(value) for value in selected_direction[:2])
            selected_tilt = tuple(
                float(value) for value in selected_direction[2:]
            )
            phase = InsertionState.CONTACT_UNLOAD
            command[:] = 0.0
            status = "PROBE_DIRECTION_SELECTED"
    next_state = replace(
        state,
        phase=phase,
        step_count=state.step_count + 1,
        phase_step=0 if phase is not InsertionState.ACTIVE_PROBE else state.phase_step + 1,
        probe_leg=leg,
        probe_leg_step=leg_step,
        probe_total_steps=state.probe_total_steps + 1,
        probe_origin_xy_m=tuple(float(value) for value in origin_xy),
        probe_origin_tilt_rad=tuple(float(value) for value in origin_tilt),
        probe_baseline_score=baseline_score,
        probe_scores=tuple(float(value) for value in scores),
        probe_selected_xy=selected,
        probe_selected_tilt=selected_tilt,
        retained_probe_xy=(
            selected
            if float(np.linalg.norm(selected)) > 0.5
            else state.retained_probe_xy
        ),
        retained_probe_tilt=(
            selected_tilt
            if float(np.linalg.norm(selected_tilt)) > 0.5
            else state.retained_probe_tilt
        ),
        probe_selection_history=(
            state.probe_selection_history
            + (
                tuple(float(value) for value in selected_direction),
            )
            if status in {
                "PROBE_DIRECTION_SELECTED",
                "PROBE_EARLY_PAIR_SELECTED",
            }
            else state.probe_selection_history
        ),
        contact_unload_origin_z_m=None,
        filtered_wrench=tuple(float(value) for value in filtered),
        last_twist_command=tuple(float(value) for value in command),
    )
    return next_state, command, status


def step_compliant_insertion(config: Mapping[str, Any], state: ControllerState, observation: InsertionObservation) -> InsertionCommand:
    rate = float(config["control_rate_hz"])
    dt = 1.0 / rate
    motion, gains, safety = config["motion"], config["admittance"], config["safety"]
    wrench = np.asarray(observation.wrench_assembly, dtype=np.float64)
    position = np.asarray(observation.tcp_position_assembly_m, dtype=np.float64)
    rotation = np.asarray(
        observation.tcp_rotation_vector_assembly_rad, dtype=np.float64
    )
    if wrench.shape != (6,) or position.shape != (3,) or rotation.shape != (3,) or not np.all(np.isfinite(wrench)) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
        return _abort(state, "NONFINITE_OBSERVATION")
    if observation.sample_age_s > float(safety["maximum_sample_age_s"]):
        return _abort(state, "STALE_WRENCH")
    if not observation.ft_valid or not observation.ft_tared or not observation.payload_compensated:
        return _abort(state, "FT_TARE_OR_PAYLOAD_INVALID")
    alpha = float(config["filter"]["wrench_lowpass_alpha"])
    filtered = alpha * wrench + (1.0 - alpha) * np.asarray(state.filtered_wrench)
    axial, lateral = abs(float(filtered[2])), float(np.linalg.norm(filtered[:2]))
    bending, torsion = float(np.linalg.norm(filtered[3:5])), abs(float(filtered[5]))
    hard = (float(safety["hard_axial_force_n"]), float(safety["hard_lateral_force_n"]), float(safety["hard_bending_moment_nm"]), float(safety["hard_torsional_moment_nm"]))
    if any(value > limit for value, limit in zip((axial, lateral, bending, torsion), hard)):
        return _abort(replace(state, filtered_wrench=tuple(filtered)), "HARD_SAFETY_GATE")
    start = position if state.start_position_m is None else np.asarray(state.start_position_m)
    progress = float(position[2] - start[2])
    window_steps = max(1, int(round(float(config["contact_classifier"]["stalled_window_s"]) * rate)))
    if state.step_count - state.progress_window_step >= window_steps:
        window_origin, window_step = float(position[2]), state.step_count
    else:
        window_origin, window_step = state.progress_window_origin_m, state.progress_window_step
    progress_window = float(position[2] - window_origin)
    contact = classify_contact(tuple(filtered), axial_progress_m=progress, progress_in_window_m=progress_window, rz_search_active=state.phase is InsertionState.BOUNDED_RZ_SEARCH, thresholds=config["contact_classifier"])
    if state.step_count > int(float(motion["maximum_duration_s"]) * rate):
        return _abort(state, "MAXIMUM_DURATION", contact)
    if float(np.linalg.norm(position - start)) > float(motion["maximum_total_travel_m"]):
        return _abort(state, "MAXIMUM_TRAVEL", contact)
    if state.phase is InsertionState.PREINSERT_READY:
        if not observation.vision_control_authorized or not observation.synchronized_capture:
            next_state = replace(state, phase=InsertionState.REOBSERVE, step_count=state.step_count + 1, filtered_wrench=tuple(filtered), start_position_m=tuple(start))
            return InsertionCommand(next_state, (0.0,) * 6, contact, True, True, "VISION_REOBSERVE_REQUIRED")
        phase = InsertionState.GUARDED_APPROACH
    else:
        phase = state.phase
    soft = tuple(float(safety["soft_fraction"]) * item for item in hard)
    if any(value > limit for value, limit in zip((axial, lateral, bending, torsion), soft)):
        next_state = replace(state, phase=InsertionState.BACKOFF, step_count=state.step_count + 1, filtered_wrench=tuple(filtered), start_position_m=tuple(start))
        return InsertionCommand(next_state, (0.0, 0.0, -float(config["recovery"]["backoff_speed_m_s"]), 0.0, 0.0, 0.0), contact, False, False, "SOFT_GATE_BACKOFF")
    target = np.zeros(6)
    if phase is InsertionState.ACTIVE_PROBE:
        maximum_probe_steps = int(config["active_probe"]["maximum_total_steps"])
        if state.probe_total_steps >= maximum_probe_steps:
            next_state = replace(
                state,
                phase=InsertionState.BACKOFF,
                step_count=state.step_count + 1,
                phase_step=0,
                filtered_wrench=tuple(float(value) for value in filtered),
            )
            twist = (0.0, 0.0, -float(config["recovery"]["backoff_speed_m_s"]), 0.0, 0.0, 0.0)
            return InsertionCommand(next_state, twist, contact, False, False, "PROBE_STEP_LIMIT_BACKOFF")
        next_state, target, status = _probe_command(
            config, state, position, rotation, filtered
        )
        command = _slew(target, np.asarray(state.last_twist_command), float(motion["maximum_linear_acceleration_m_s2"]), float(motion["maximum_angular_acceleration_rad_s2"]), dt)
        next_state = replace(next_state, last_twist_command=tuple(float(value) for value in command), start_position_m=tuple(float(value) for value in start), progress_window_origin_m=window_origin, progress_window_step=window_step)
        return InsertionCommand(next_state, tuple(float(value) for value in command), contact, False, False, status)
    if phase is InsertionState.CONTACT_UNLOAD:
        origin_z = (
            float(position[2])
            if state.contact_unload_origin_z_m is None
            else float(state.contact_unload_origin_z_m)
        )
        unloaded = origin_z - float(position[2])
        angular_selection = bool(
            np.linalg.norm(np.asarray(state.probe_selected_tilt)) > 0.5
        )
        unload_distance = float(
            config["active_probe"][
                "angular_contact_unload_distance_m"
                if angular_selection
                else "contact_unload_distance_m"
            ]
        )
        if unloaded < unload_distance:
            target = np.zeros(6)
            target[2] = -float(
                config["active_probe"][
                    "angular_contact_unload_speed_m_s"
                    if angular_selection
                    else "contact_unload_speed_m_s"
                ]
            )
            command = _slew(
                target,
                np.asarray(state.last_twist_command),
                float(motion["maximum_linear_acceleration_m_s2"]),
                float(motion["maximum_angular_acceleration_rad_s2"]),
                dt,
            )
            next_state = replace(
                state,
                step_count=state.step_count + 1,
                phase_step=state.phase_step + 1,
                filtered_wrench=tuple(float(value) for value in filtered),
                last_twist_command=tuple(float(value) for value in command),
                start_position_m=tuple(float(value) for value in start),
                progress_window_origin_m=window_origin,
                progress_window_step=window_step,
                contact_unload_origin_z_m=origin_z,
            )
            return InsertionCommand(
                next_state,
                tuple(float(value) for value in command),
                contact,
                False,
                False,
                "CONTACT_UNLOAD",
            )
        next_state = replace(
            state,
            phase=InsertionState.UNLOADED_CENTERING,
            step_count=state.step_count + 1,
            phase_step=0,
            filtered_wrench=tuple(float(value) for value in filtered),
            last_twist_command=(0.0,) * 6,
            start_position_m=tuple(float(value) for value in start),
            progress_window_origin_m=window_origin,
            progress_window_step=window_step,
            unloaded_centering_origin_xy_m=tuple(
                float(value) for value in position[:2]
            ),
            unloaded_centering_origin_tilt_rad=tuple(
                float(value) for value in rotation[:2]
            ),
        )
        return InsertionCommand(
            next_state,
            (0.0,) * 6,
            contact,
            False,
            False,
            "CONTACT_UNLOAD_COMPLETE_BEGIN_UNLOADED_CENTERING",
        )
    if phase is InsertionState.UNLOADED_CENTERING:
        origin_xy = (
            np.asarray(position[:2])
            if state.unloaded_centering_origin_xy_m is None
            else np.asarray(state.unloaded_centering_origin_xy_m)
        )
        selected_xy = np.asarray(state.probe_selected_xy)
        origin_tilt = (
            np.asarray(rotation[:2])
            if state.unloaded_centering_origin_tilt_rad is None
            else np.asarray(state.unloaded_centering_origin_tilt_rad)
        )
        selected_tilt = np.asarray(state.probe_selected_tilt)
        angular_selection = bool(np.linalg.norm(selected_tilt) > 0.5)
        shifted = float(
            np.dot(
                rotation[:2] - origin_tilt,
                selected_tilt,
            )
            if angular_selection
            else np.dot(position[:2] - origin_xy, selected_xy)
        )
        shift_limit = float(
            config["active_probe"][
                "unloaded_tilt_distance_rad"
                if angular_selection
                else "unloaded_centering_distance_m"
            ]
        )
        if angular_selection:
            shift_limit = min(
                shift_limit,
                max(
                    0.0,
                    float(motion["maximum_search_angle_rad"])
                    - max(
                        float(state.cumulative_unloaded_tilt_rad),
                        float(np.linalg.norm(origin_tilt)),
                    )
                    - 2.0
                    * float(
                        config["active_probe"][
                            "angular_position_tolerance_rad"
                        ]
                    ),
                ),
            )
        if shifted < shift_limit:
            target = np.zeros(6)
            if angular_selection:
                target[3:5] = selected_tilt * float(
                    config["active_probe"]["selected_tilt_speed_rad_s"]
                )
            else:
                target[:2] = selected_xy * float(
                    config["active_probe"]["selected_centering_speed_m_s"]
                )
            command = _slew(
                target,
                np.asarray(state.last_twist_command),
                float(motion["maximum_linear_acceleration_m_s2"]),
                float(motion["maximum_angular_acceleration_rad_s2"]),
                dt,
            )
            next_state = replace(
                state,
                step_count=state.step_count + 1,
                phase_step=state.phase_step + 1,
                filtered_wrench=tuple(float(value) for value in filtered),
                last_twist_command=tuple(float(value) for value in command),
                start_position_m=tuple(float(value) for value in start),
                progress_window_origin_m=window_origin,
                progress_window_step=window_step,
                unloaded_centering_origin_xy_m=tuple(
                    float(value) for value in origin_xy
                ),
                unloaded_centering_origin_tilt_rad=tuple(
                    float(value) for value in origin_tilt
                ),
            )
            return InsertionCommand(
                next_state,
                tuple(float(value) for value in command),
                contact,
                False,
                False,
                "UNLOADED_CENTERING",
            )
        next_state = replace(
            state,
            phase=InsertionState.INSERT_ADVANCE,
            step_count=state.step_count + 1,
            phase_step=0,
            filtered_wrench=tuple(float(value) for value in filtered),
            last_twist_command=(0.0,) * 6,
            start_position_m=tuple(float(value) for value in start),
            progress_window_origin_m=window_origin,
            progress_window_step=window_step,
            probe_selected_xy=(0.0, 0.0),
            probe_selected_tilt=(0.0, 0.0),
            contact_unload_origin_z_m=None,
            unloaded_centering_origin_xy_m=None,
            unloaded_centering_origin_tilt_rad=None,
            contact_realign_count=state.contact_realign_count + 1,
            cumulative_unloaded_tilt_rad=(
                float(state.cumulative_unloaded_tilt_rad)
                + max(0.0, shifted)
                if angular_selection
                else float(state.cumulative_unloaded_tilt_rad)
            ),
        )
        return InsertionCommand(
            next_state,
            (0.0,) * 6,
            contact,
            False,
            False,
            "UNLOADED_CENTERING_COMPLETE_REAPPROACH",
        )
    if phase is InsertionState.FIRST_CONTACT:
        phase = InsertionState.CONTACT_HOLD
        target[:] = 0.0
    elif phase is InsertionState.CONTACT_HOLD:
        target[:] = 0.0
        hold_steps = max(1, int(round(float(config["active_probe"]["contact_hold_s"]) * rate)))
        if state.phase_step >= hold_steps:
            phase = InsertionState.CONTACT_CLASSIFY
    elif phase is InsertionState.CONTACT_CLASSIFY:
        latched_contact = ContactClass(state.latched_contact_class)
        if latched_contact in {ContactClass.AXIAL_CONTACT, ContactClass.SINGLE_EDGE_CONTACT, ContactClass.DOUBLE_EDGE_OR_JAM}:
            if state.contact_realign_count > int(motion["maximum_retries"]):
                next_state = replace(
                    state,
                    phase=InsertionState.BACKOFF,
                    step_count=state.step_count + 1,
                    phase_step=0,
                    filtered_wrench=tuple(float(value) for value in filtered),
                    last_twist_command=(0.0, 0.0, -float(config["recovery"]["backoff_speed_m_s"]), 0.0, 0.0, 0.0),
                )
                return InsertionCommand(
                    next_state,
                    next_state.last_twist_command,
                    contact,
                    False,
                    False,
                    "CONTACT_REALIGN_RETRY_LIMIT_BACKOFF",
                )
            retained_xy = np.asarray(state.retained_probe_xy)
            if (
                state.contact_realign_count > 0
                and float(np.linalg.norm(retained_xy)) > 0.5
            ):
                xy_offset = position[:2] - start[:2]
                remaining_xy = (
                    float(motion["maximum_search_radius_m"])
                    - float(np.linalg.norm(xy_offset))
                )
                if remaining_xy <= float(
                    config["active_probe"]["position_tolerance_m"]
                ):
                    next_state = replace(
                        state,
                        phase=InsertionState.BACKOFF,
                        step_count=state.step_count + 1,
                        phase_step=0,
                        filtered_wrench=tuple(float(value) for value in filtered),
                        last_twist_command=(
                            0.0,
                            0.0,
                            -float(config["recovery"]["backoff_speed_m_s"]),
                            0.0,
                            0.0,
                            0.0,
                        ),
                    )
                    return InsertionCommand(
                        next_state,
                        next_state.last_twist_command,
                        contact,
                        False,
                        False,
                        "PERSISTENT_XY_BUDGET_EXHAUSTED_BACKOFF",
                    )
                next_state = replace(
                    state,
                    phase=InsertionState.CONTACT_UNLOAD,
                    step_count=state.step_count + 1,
                    phase_step=0,
                    probe_selected_xy=tuple(
                        float(value) for value in retained_xy
                    ),
                    probe_selected_tilt=(0.0, 0.0),
                    contact_unload_origin_z_m=None,
                    filtered_wrench=tuple(float(value) for value in filtered),
                    last_twist_command=(0.0,) * 6,
                )
                return InsertionCommand(
                    next_state,
                    (0.0,) * 6,
                    contact,
                    False,
                    False,
                    "REUSE_MEASURED_XY_DIRECTION",
                )
            retained_tilt = np.asarray(state.retained_probe_tilt)
            if (
                state.contact_realign_count > 0
                and float(np.linalg.norm(retained_tilt)) > 0.5
            ):
                remaining_tilt = (
                    float(motion["maximum_search_angle_rad"])
                    - max(
                        float(state.cumulative_unloaded_tilt_rad),
                        float(np.linalg.norm(rotation[:2])),
                    )
                )
                if remaining_tilt <= float(
                    config["active_probe"]["angular_position_tolerance_rad"]
                ):
                    next_state = replace(
                        state,
                        phase=InsertionState.BACKOFF,
                        step_count=state.step_count + 1,
                        phase_step=0,
                        filtered_wrench=tuple(float(value) for value in filtered),
                        last_twist_command=(
                            0.0,
                            0.0,
                            -float(config["recovery"]["backoff_speed_m_s"]),
                            0.0,
                            0.0,
                            0.0,
                        ),
                    )
                    return InsertionCommand(
                        next_state,
                        next_state.last_twist_command,
                        contact,
                        False,
                        False,
                        "PERSISTENT_TILT_BUDGET_EXHAUSTED_BACKOFF",
                    )
                # Reuse the direction that the first symmetric force probe
                # measured as load-reducing.  A later side contact may change
                # the instantaneous score ordering; allowing it to reverse
                # the prior correction caused bounded moves to cancel.  The
                # total motion remains capped by maximum_search_angle_rad.
                next_state = replace(
                    state,
                    phase=InsertionState.CONTACT_UNLOAD,
                    step_count=state.step_count + 1,
                    phase_step=0,
                    probe_selected_xy=(0.0, 0.0),
                    probe_selected_tilt=tuple(
                        float(value) for value in retained_tilt
                    ),
                    contact_unload_origin_z_m=None,
                    filtered_wrench=tuple(float(value) for value in filtered),
                    last_twist_command=(0.0,) * 6,
                )
                return InsertionCommand(
                    next_state,
                    (0.0,) * 6,
                    contact,
                    False,
                    False,
                    "REUSE_MEASURED_TILT_DIRECTION",
                )
            seed_twist = response_model_twist(
                config, np.asarray(state.latched_contact_wrench)
            )
            seed_motion = np.asarray(
                (seed_twist[0], seed_twist[1], seed_twist[3], seed_twist[4])
            )
            equivalent_seed = seed_motion.copy()
            equivalent_seed[2:] *= float(
                config["local_response_model"]["effective_length_m"]
            )
            direction_order_list = []
            for motion_axis in np.argsort(-np.abs(equivalent_seed)):
                preferred = 2 * int(motion_axis) + int(
                    seed_motion[motion_axis] < 0.0
                )
                direction_order_list.extend((preferred, preferred ^ 1))
            direction_order = tuple(direction_order_list)
            phase = InsertionState.ACTIVE_PROBE
            next_state = replace(
                state,
                phase=phase,
                step_count=state.step_count + 1,
                phase_step=0,
                probe_leg=0,
                probe_leg_step=0,
                probe_total_steps=0,
                probe_origin_xy_m=tuple(float(value) for value in position[:2]),
                probe_origin_tilt_rad=tuple(
                    float(value) for value in rotation[:2]
                ),
                probe_baseline_score=state.latched_contact_score,
                probe_scores=(math.inf,) * 8,
                probe_direction_order=direction_order,
                filtered_wrench=tuple(float(value) for value in filtered),
                last_twist_command=(0.0,) * 6,
                start_position_m=tuple(float(value) for value in start),
                progress_window_origin_m=window_origin,
                progress_window_step=window_step,
            )
            return InsertionCommand(
                next_state,
                (0.0,) * 6,
                contact,
                False,
                False,
                "CONTACT_CLASSIFIED_BEGIN_ACTIVE_PROBE",
            )
        phase = InsertionState.INSERT_ADVANCE
    if phase in {InsertionState.GUARDED_APPROACH, InsertionState.INSERT_ADVANCE, InsertionState.COMPLIANT_CENTERING, InsertionState.TILT_CORRECTION}:
        if contact is ContactClass.SINGLE_EDGE_CONTACT:
            if (
                phase is InsertionState.COMPLIANT_CENTERING
                and any(
                    abs(value) > 0.0 for value in state.probe_selected_xy
                )
            ):
                target[:2] = np.asarray(state.probe_selected_xy) * float(
                    config["active_probe"]["selected_centering_speed_m_s"]
                )
            else:
                target += response_model_twist(config, filtered)
        # +Z is insertion, therefore a fixture reaction on the tool is -Fz.
        # The direction-calibration suite establishes this sign explicitly.
        # A positive +Fz assists insertion and must not be mistaken for a
        # resisting load.
        axial_resistance = max(0.0, -float(filtered[2]))
        target[2] = np.clip(
            float(gains["k_fz_m_s_n"])
            * (float(gains["fz_reference_n"]) - axial_resistance),
            0.0,
            float(motion["axial_speed_m_s"]),
        )
        if contact is ContactClass.SINGLE_EDGE_CONTACT:
            target[2] = 0.0
    if phase in {InsertionState.CONTACT_HOLD, InsertionState.CONTACT_CLASSIFY}:
        pass
    elif phase is InsertionState.GUARDED_APPROACH and contact is not ContactClass.NO_CONTACT:
        score = abs(float(filtered[2])) + float(
            config["active_probe"]["bending_weight_n_per_nm"]
        ) * float(np.linalg.norm(filtered[3:5]))
        next_state = replace(
            state,
            phase=InsertionState.FIRST_CONTACT,
            step_count=state.step_count + 1,
            phase_step=0,
            filtered_wrench=tuple(float(value) for value in filtered),
            last_twist_command=(0.0,) * 6,
            start_position_m=tuple(float(value) for value in start),
            progress_window_origin_m=window_origin,
            progress_window_step=window_step,
            latched_contact_class=contact.value,
            latched_contact_score=score,
            latched_contact_wrench=tuple(float(value) for value in filtered),
        )
        return InsertionCommand(
            next_state,
            (0.0,) * 6,
            contact,
            False,
            False,
            "FIRST_CONTACT_LATCHED",
        )
    elif contact is ContactClass.SINGLE_EDGE_CONTACT and phase not in {InsertionState.FIRST_CONTACT, InsertionState.CONTACT_HOLD, InsertionState.CONTACT_CLASSIFY}:
        if any(abs(value) > 0.0 for value in state.probe_selected_xy):
            phase = InsertionState.COMPLIANT_CENTERING if lateral >= float(config["contact_classifier"]["lateral_contact_n"]) else InsertionState.TILT_CORRECTION
        else:
            score = abs(float(filtered[2])) + float(
                config["active_probe"]["bending_weight_n_per_nm"]
            ) * float(np.linalg.norm(filtered[3:5]))
            next_state = replace(
                state,
                phase=InsertionState.FIRST_CONTACT,
                step_count=state.step_count + 1,
                phase_step=0,
                filtered_wrench=tuple(float(value) for value in filtered),
                last_twist_command=(0.0,) * 6,
                start_position_m=tuple(float(value) for value in start),
                progress_window_origin_m=window_origin,
                progress_window_step=window_step,
                latched_contact_class=contact.value,
                latched_contact_score=score,
                latched_contact_wrench=tuple(
                    float(value) for value in filtered
                ),
            )
            return InsertionCommand(
                next_state,
                (0.0,) * 6,
                contact,
                False,
                False,
                "FIRST_CONTACT_LATCHED",
            )
    elif contact is ContactClass.KEY_MISMATCH:
        if abs(state.rz_search_angle_rad) >= float(motion["maximum_search_angle_rad"]):
            phase = InsertionState.BACKOFF
            target[:] = (0.0, 0.0, -float(config["recovery"]["backoff_speed_m_s"]), 0.0, 0.0, 0.0)
        else:
            phase = InsertionState.BOUNDED_RZ_SEARCH
            target[:] = 0.0
            target[5] = float(motion["maximum_rz_search_speed_rad_s"])
    elif contact is ContactClass.DOUBLE_EDGE_OR_JAM:
        phase = InsertionState.BACKOFF
        target[:] = (0.0, 0.0, -float(config["recovery"]["backoff_speed_m_s"]), 0.0, 0.0, 0.0)
    elif contact in {ContactClass.GUIDED_ENTRY, ContactClass.INSERTING}:
        phase = InsertionState.INSERT_ADVANCE
    elif contact is ContactClass.SEATED:
        phase = InsertionState.SUCCESS
        target[:] = 0.0
    xy_offset = position[:2] - start[:2]
    if float(np.linalg.norm(xy_offset)) >= float(motion["maximum_search_radius_m"]) and float(np.dot(target[:2], xy_offset)) > 0.0:
        # Keep the tangential component but remove any further radial growth.
        unit = xy_offset / np.linalg.norm(xy_offset)
        target[:2] -= unit * float(np.dot(target[:2], unit))
    previous = np.asarray(state.last_twist_command)
    command = _slew(target, previous, float(motion["maximum_linear_acceleration_m_s2"]), float(motion["maximum_angular_acceleration_rad_s2"]), dt)
    phase_step = state.phase_step + 1 if phase is state.phase else 0
    next_state = replace(state, phase=phase, step_count=state.step_count + 1, phase_step=phase_step, filtered_wrench=tuple(float(value) for value in filtered), last_twist_command=tuple(float(value) for value in command), start_position_m=tuple(float(value) for value in start), progress_window_origin_m=window_origin, progress_window_step=window_step, xy_search_offset_m=tuple(float(value) for value in xy_offset), rz_search_angle_rad=float(state.rz_search_angle_rad + command[5] * dt))
    stop = phase in {InsertionState.SUCCESS, InsertionState.SAFE_ABORT, InsertionState.REOBSERVE}
    return InsertionCommand(next_state, tuple(float(value) for value in command), contact, stop, phase is InsertionState.REOBSERVE, phase.value)


__all__ = ["ControllerState", "InsertionCommand", "InsertionObservation", "InsertionState", "effective_lateral_posthoc", "full_seated_posthoc", "load_compliant_insertion_config", "response_model_twist", "step_compliant_insertion", "wrench_sensor_to_assembly"]
