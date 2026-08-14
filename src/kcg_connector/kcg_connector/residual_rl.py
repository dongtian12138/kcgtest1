"""Pure NumPy residual-RL contract for one engaged q7 twist probe.

The simulator adapter owns reset and physics.  This module only defines the
versioned action/observation mapping, measured-progress reward and termination
rules, so it can be tested without importing Isaac Sim, ROS 2, Gymnasium or
PyTorch.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import yaml

from .geometry import helical_travel


RESIDUAL_ACTION_NAMES = (
    "tightening_speed_residual",
    "f1j2_clamp_residual",
    "f2j1_clamp_residual",
    "f3j2_clamp_residual",
)

RESIDUAL_OBSERVATION_NAMES = (
    "phase_progress",
    "q7_tracking_error",
    "q7_tightening_velocity",
    "nut_progress",
    "nut_tightening_velocity",
    "axial_progress",
    "axial_tightening_velocity",
    "helical_error",
    "grasp_translation_x",
    "grasp_translation_y",
    "grasp_translation_z",
    "grasp_rotation_x",
    "grasp_rotation_y",
    "grasp_rotation_z",
    "f1j2_torque",
    "f2j1_torque",
    "f3j2_torque",
    "f1j2_torque_delta",
    "f2j1_torque_delta",
    "f3j2_torque_delta",
    "f1j2_position_residual",
    "f2j1_position_residual",
    "f3j2_position_residual",
    "remaining_nut_angle",
)

RESIDUAL_ACTION_SIZE = len(RESIDUAL_ACTION_NAMES)
RESIDUAL_OBSERVATION_SIZE = len(RESIDUAL_OBSERVATION_NAMES)


def _finite_float(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_tuple(name, values, size):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numbers") from error
    if len(result) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class ConnectorResidualConfig:
    """Validated stage-1 contract, with all angular values in radians."""

    interface_version: str
    policy_rate_hz: float
    target_angle_rad: float
    tightening_direction: int
    nominal_q7_speed_rad_s: float
    q7_speed_residual_rad_s: float
    maximum_q7_speed_rad_s: float
    q7_safe_lower_rad: float
    q7_safe_upper_rad: float
    clamp_joint_names: tuple
    clamp_nominal_positions_rad: tuple
    clamp_position_residual_limits_rad: tuple
    loaded_torque_threshold_nm: float
    minimum_loaded_torque_channels: int
    maximum_absolute_finger_torque_nm: float
    helical_lead_m: float
    helical_error_tolerance_m: float
    q7_tracking_error_scale_rad: float
    grasp_translation_error_scale_m: float
    grasp_rotation_error_scale_rad: float
    torque_delta_scale_nm: float
    minimum_axial_progress_fraction: float
    success_angle_tolerance_rad: float
    success_hold_duration_s: float
    maximum_grasp_translation_error_m: float
    maximum_grasp_rotation_error_rad: float
    maximum_q7_tracking_error_rad: float
    hold_q7_velocity_tolerance_rad_s: float
    hold_nut_velocity_tolerance_rad_s: float
    hold_axial_velocity_tolerance_m_s: float

    @property
    def expected_axial_travel_m(self):
        return helical_travel(self.target_angle_rad, self.helical_lead_m)

    @property
    def maximum_axial_speed_m_s(self):
        return helical_travel(
            self.maximum_q7_speed_rad_s, self.helical_lead_m
        )

    def helical_tolerance_m(self, nut_angle_rad):
        """Match the proven short-probe gate at the measured nut angle."""
        expected = helical_travel(
            max(0.0, float(nut_angle_rad)), self.helical_lead_m
        )
        return min(
            self.helical_error_tolerance_m,
            max(0.00005, 0.25 * expected),
        )


def load_connector_residual_config(config_path):
    """Load the residual contract and cross-check it against task limits."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        raw = document["residual_rl"]
        q7 = document["q7_twist"]
        success = document["success"]
        config = ConnectorResidualConfig(
            interface_version=str(raw["interface_version"]),
            policy_rate_hz=float(raw["policy_rate_hz"]),
            target_angle_rad=math.radians(
                float(raw["target_segment_degrees"])
            ),
            tightening_direction=int(q7["tightening_direction"]),
            nominal_q7_speed_rad_s=math.radians(
                float(raw["nominal_q7_speed_degrees_per_second"])
            ),
            q7_speed_residual_rad_s=math.radians(
                float(raw["q7_speed_residual_degrees_per_second"])
            ),
            maximum_q7_speed_rad_s=math.radians(
                float(q7["maximum_speed"])
            ),
            q7_safe_lower_rad=float(q7["safe_lower_rad"]),
            q7_safe_upper_rad=float(q7["safe_upper_rad"]),
            clamp_joint_names=tuple(raw["clamp_joint_names"]),
            clamp_nominal_positions_rad=_finite_tuple(
                "clamp_nominal_positions_rad",
                raw["clamp_nominal_positions_rad"],
                3,
            ),
            clamp_position_residual_limits_rad=_finite_tuple(
                "clamp_position_residual_limits_rad",
                raw["clamp_position_residual_limits_rad"],
                3,
            ),
            loaded_torque_threshold_nm=float(
                raw["loaded_torque_threshold_nm"]
            ),
            minimum_loaded_torque_channels=int(
                success["minimum_loaded_torque_channels"]
            ),
            maximum_absolute_finger_torque_nm=float(
                success["maximum_absolute_finger_torque"]
            ),
            helical_lead_m=float(
                success["helical_lead_per_revolution"]
            ),
            helical_error_tolerance_m=float(
                success["helical_error_tolerance"]
            ),
            q7_tracking_error_scale_rad=math.radians(
                float(raw["q7_tracking_error_scale_degrees"])
            ),
            grasp_translation_error_scale_m=float(
                raw["grasp_translation_error_scale_m"]
            ),
            grasp_rotation_error_scale_rad=math.radians(
                float(raw["grasp_rotation_error_scale_degrees"])
            ),
            torque_delta_scale_nm=float(raw["torque_delta_scale_nm"]),
            minimum_axial_progress_fraction=float(
                raw["minimum_axial_progress_fraction"]
            ),
            success_angle_tolerance_rad=math.radians(
                float(raw["success_angle_tolerance_degrees"])
            ),
            success_hold_duration_s=float(
                raw["success_hold_duration_s"]
            ),
            maximum_grasp_translation_error_m=float(
                raw["maximum_grasp_translation_error_m"]
            ),
            maximum_grasp_rotation_error_rad=math.radians(
                float(raw["maximum_grasp_rotation_error_degrees"])
            ),
            maximum_q7_tracking_error_rad=math.radians(
                float(raw["maximum_q7_tracking_error_degrees"])
            ),
            hold_q7_velocity_tolerance_rad_s=math.radians(
                float(
                    raw[
                        "hold_q7_velocity_tolerance_degrees_per_second"
                    ]
                )
            ),
            hold_nut_velocity_tolerance_rad_s=math.radians(
                float(
                    raw[
                        "hold_nut_velocity_tolerance_degrees_per_second"
                    ]
                )
            ),
            hold_axial_velocity_tolerance_m_s=float(
                raw["hold_axial_velocity_tolerance_m_per_second"]
            ),
        )
        maximum_segment_angle = math.radians(
            float(q7["maximum_segment_degrees"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid residual RL config: {path}") from error

    if not config.interface_version.strip():
        raise ValueError("residual interface version must not be empty")
    if config.tightening_direction not in (-1, 1):
        raise ValueError("tightening direction must be exactly -1 or 1")
    if len(config.clamp_joint_names) != 3 or len(
        set(config.clamp_joint_names)
    ) != 3:
        raise ValueError(
            "residual clamp joint names must be three unique names"
        )
    if tuple(config.clamp_joint_names) != ("f1j2", "f2j1", "f3j2"):
        raise ValueError(
            "stage-1 residual clamp joints must match the three torque axes"
        )

    positive_values = (
        config.policy_rate_hz,
        config.target_angle_rad,
        config.nominal_q7_speed_rad_s,
        config.q7_speed_residual_rad_s,
        config.maximum_q7_speed_rad_s,
        config.loaded_torque_threshold_nm,
        config.maximum_absolute_finger_torque_nm,
        config.helical_lead_m,
        config.helical_error_tolerance_m,
        config.q7_tracking_error_scale_rad,
        config.grasp_translation_error_scale_m,
        config.grasp_rotation_error_scale_rad,
        config.torque_delta_scale_nm,
        config.minimum_axial_progress_fraction,
        config.success_angle_tolerance_rad,
        config.success_hold_duration_s,
        config.maximum_grasp_translation_error_m,
        config.maximum_grasp_rotation_error_rad,
        config.maximum_q7_tracking_error_rad,
        config.hold_q7_velocity_tolerance_rad_s,
        config.hold_nut_velocity_tolerance_rad_s,
        config.hold_axial_velocity_tolerance_m_s,
    )
    if not all(
        math.isfinite(value) and value > 0.0
        for value in positive_values
    ):
        raise ValueError("residual RL scales and limits must be positive")
    if not config.q7_safe_lower_rad < config.q7_safe_upper_rad:
        raise ValueError("q7 residual safe window is invalid")
    if config.minimum_axial_progress_fraction > 1.0:
        raise ValueError(
            "minimum axial progress fraction must be in (0, 1]"
        )
    if config.target_angle_rad > maximum_segment_angle:
        raise ValueError("residual target exceeds one deterministic q7 stroke")
    if config.success_angle_tolerance_rad >= config.target_angle_rad:
        raise ValueError("residual success angle tolerance is too large")
    if (
        config.nominal_q7_speed_rad_s
        - config.q7_speed_residual_rad_s
        <= 0.0
    ):
        raise ValueError("q7 residual can reverse or stop tightening motion")
    if (
        config.nominal_q7_speed_rad_s
        + config.q7_speed_residual_rad_s
        > config.maximum_q7_speed_rad_s
    ):
        raise ValueError("q7 residual can exceed the configured speed ceiling")
    if not 1 <= config.minimum_loaded_torque_channels <= 3:
        raise ValueError(
            "loaded torque channels must be between one and three"
        )
    if (
        config.loaded_torque_threshold_nm
        >= config.maximum_absolute_finger_torque_nm
    ):
        raise ValueError(
            "loaded torque threshold must be below overload torque"
        )
    if not all(
        value > 0.0
        for value in config.clamp_position_residual_limits_rad
    ):
        raise ValueError("clamp residual limits must be positive")
    return config


@dataclass(frozen=True)
class ConnectorResidualState:
    """One measured, already-engaged twist state in deployment units.

    ``nut_angle_rad``, ``axial_travel_m`` and their velocities are positive in
    the tightening task direction.  The Isaac adapter therefore converts the
    observed downward world-Z motion to positive axial travel.
    """

    phase_progress: float
    q7_position_rad: float
    q7_tracking_error_rad: float
    q7_velocity_rad_s: float
    nut_angle_rad: float
    nut_angular_velocity_rad_s: float
    axial_travel_m: float
    axial_velocity_m_s: float
    grasp_translation_error_m: tuple
    grasp_rotation_error_rad: tuple
    finger_torques_nm: tuple
    finger_torque_deltas_nm: tuple
    clamp_positions_rad: tuple
    stable_hold_seconds: float = 0.0


@dataclass(frozen=True)
class DecodedResidualAction:
    """Safe physical targets produced from one normalized policy action."""

    normalized: tuple
    q7_velocity_target_rad_s: float
    clamp_position_targets_rad: tuple


@dataclass(frozen=True)
class ResidualTermination:
    terminated: bool
    success: bool
    reason: str


@dataclass(frozen=True)
class ResidualReward:
    total: float
    terms: dict


def _state_arrays(state):
    return (
        np.asarray(state.grasp_translation_error_m, dtype=np.float64),
        np.asarray(state.grasp_rotation_error_rad, dtype=np.float64),
        np.asarray(state.finger_torques_nm, dtype=np.float64),
        np.asarray(state.finger_torque_deltas_nm, dtype=np.float64),
        np.asarray(state.clamp_positions_rad, dtype=np.float64),
    )


def _validate_state(state):
    scalars = np.asarray(
        [
            state.phase_progress,
            state.q7_position_rad,
            state.q7_tracking_error_rad,
            state.q7_velocity_rad_s,
            state.nut_angle_rad,
            state.nut_angular_velocity_rad_s,
            state.axial_travel_m,
            state.axial_velocity_m_s,
            state.stable_hold_seconds,
        ],
        dtype=np.float64,
    )
    arrays = _state_arrays(state)
    if any(array.shape != (3,) for array in arrays):
        raise ValueError("residual state vector fields must have shape (3,)")
    if not np.all(np.isfinite(scalars)) or not all(
        np.all(np.isfinite(array)) for array in arrays
    ):
        raise ValueError("residual state must contain only finite values")
    return arrays


def decode_residual_action(action, config):
    """Map a normalized action to q7 speed and three clamp positions."""
    normalized = np.asarray(action, dtype=np.float64)
    if normalized.shape != (RESIDUAL_ACTION_SIZE,):
        raise ValueError(
            f"residual action must have shape ({RESIDUAL_ACTION_SIZE},)"
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("residual action must contain finite values")
    normalized = np.clip(normalized, -1.0, 1.0)
    tightening_speed = (
        config.nominal_q7_speed_rad_s
        + normalized[0] * config.q7_speed_residual_rad_s
    )
    q7_velocity = config.tightening_direction * tightening_speed
    clamp_targets = np.asarray(
        config.clamp_nominal_positions_rad, dtype=np.float64
    ) + normalized[1:] * np.asarray(
        config.clamp_position_residual_limits_rad, dtype=np.float64
    )
    return DecodedResidualAction(
        normalized=tuple(float(value) for value in normalized),
        q7_velocity_target_rad_s=float(q7_velocity),
        clamp_position_targets_rad=tuple(
            float(value) for value in clamp_targets
        ),
    )


def residual_observation(state, config):
    """Encode the documented 24-D, bounded, hardware-compatible vector."""
    translation, rotation, torques, torque_deltas, clamp_positions = (
        _validate_state(state)
    )
    helical_error = state.axial_travel_m - helical_travel(
        max(0.0, state.nut_angle_rad), config.helical_lead_m
    )
    values = np.concatenate(
        (
            np.asarray(
                [
                    np.clip(state.phase_progress, 0.0, 1.0),
                    state.q7_tracking_error_rad
                    / config.q7_tracking_error_scale_rad,
                    config.tightening_direction
                    * state.q7_velocity_rad_s
                    / config.maximum_q7_speed_rad_s,
                    state.nut_angle_rad / config.target_angle_rad,
                    state.nut_angular_velocity_rad_s
                    / config.maximum_q7_speed_rad_s,
                    state.axial_travel_m
                    / config.expected_axial_travel_m,
                    state.axial_velocity_m_s
                    / config.maximum_axial_speed_m_s,
                    helical_error
                    / config.helical_tolerance_m(state.nut_angle_rad),
                ],
                dtype=np.float64,
            ),
            translation / config.grasp_translation_error_scale_m,
            rotation / config.grasp_rotation_error_scale_rad,
            torques / config.maximum_absolute_finger_torque_nm,
            torque_deltas / config.torque_delta_scale_nm,
            (
                clamp_positions
                - np.asarray(
                    config.clamp_nominal_positions_rad, dtype=np.float64
                )
            )
            / np.asarray(
                config.clamp_position_residual_limits_rad,
                dtype=np.float64,
            ),
            np.asarray(
                [
                    (config.target_angle_rad - state.nut_angle_rad)
                    / config.target_angle_rad
                ],
                dtype=np.float64,
            ),
        )
    )
    if values.shape != (RESIDUAL_OBSERVATION_SIZE,):
        raise RuntimeError("residual observation contract size changed")
    return np.clip(values, -1.0, 1.0).astype(np.float32)


def loaded_torque_channels(state, config):
    """Count only the three real one-dimensional base torque channels."""
    _, _, torques, _, _ = _validate_state(state)
    return int(
        np.count_nonzero(
            np.abs(torques) >= config.loaded_torque_threshold_nm
        )
    )


def evaluate_residual_state(state, config):
    """Terminate from measured nut/travel/grasp state, never q7 command sum."""
    try:
        translation, _, torques, _, _ = _validate_state(state)
    except ValueError:
        return ResidualTermination(True, False, "invalid_physics")
    if not (
        config.q7_safe_lower_rad
        <= state.q7_position_rad
        <= config.q7_safe_upper_rad
    ):
        return ResidualTermination(True, False, "q7_limit")
    if abs(state.q7_velocity_rad_s) > config.maximum_q7_speed_rad_s * 1.10:
        return ResidualTermination(True, False, "q7_overspeed")
    if (
        abs(state.nut_angular_velocity_rad_s)
        > config.maximum_q7_speed_rad_s * 1.25
    ):
        return ResidualTermination(True, False, "nut_overspeed")
    if (
        abs(state.q7_tracking_error_rad)
        > config.maximum_q7_tracking_error_rad
    ):
        return ResidualTermination(True, False, "q7_tracking")
    if (
        float(np.max(np.abs(torques)))
        > config.maximum_absolute_finger_torque_nm
    ):
        return ResidualTermination(True, False, "finger_overload")
    if (
        loaded_torque_channels(state, config)
        < config.minimum_loaded_torque_channels
    ):
        return ResidualTermination(True, False, "lost_grasp")
    if float(np.linalg.norm(translation)) > (
        config.maximum_grasp_translation_error_m
    ):
        return ResidualTermination(True, False, "lost_grasp")
    rotation = np.asarray(
        state.grasp_rotation_error_rad, dtype=np.float64
    )
    if float(np.linalg.norm(rotation)) > (
        config.maximum_grasp_rotation_error_rad
    ):
        return ResidualTermination(True, False, "lost_grasp")

    progress = max(0.0, state.nut_angle_rad) / config.target_angle_rad
    expected_travel = helical_travel(
        max(0.0, state.nut_angle_rad), config.helical_lead_m
    )
    helical_error = state.axial_travel_m - expected_travel
    helical_tolerance = config.helical_tolerance_m(
        state.nut_angle_rad
    )
    if state.nut_angle_rad < -config.success_angle_tolerance_rad:
        return ResidualTermination(True, False, "reverse_progress")
    if state.nut_angle_rad > (
        config.target_angle_rad + config.success_angle_tolerance_rad
    ):
        return ResidualTermination(True, False, "overtwist")
    if (
        progress >= 0.25
        and (
            abs(helical_error) > helical_tolerance
            or state.axial_travel_m
            < config.minimum_axial_progress_fraction * expected_travel
        )
    ):
        return ResidualTermination(True, False, "cross_thread")
    angle_complete = abs(
        state.nut_angle_rad - config.target_angle_rad
    ) <= config.success_angle_tolerance_rad
    helix_complete = bool(
        expected_travel > 0.0
        and state.axial_travel_m > 0.0
        and state.axial_travel_m
        >= config.minimum_axial_progress_fraction * expected_travel
        and abs(helical_error) <= helical_tolerance
    )
    hold_stable = bool(
        abs(state.q7_velocity_rad_s)
        <= config.hold_q7_velocity_tolerance_rad_s
        and abs(state.nut_angular_velocity_rad_s)
        <= config.hold_nut_velocity_tolerance_rad_s
        and abs(state.axial_velocity_m_s)
        <= config.hold_axial_velocity_tolerance_m_s
    )
    if angle_complete and helix_complete:
        if (
            hold_stable
            and state.stable_hold_seconds
            >= config.success_hold_duration_s
        ):
            return ResidualTermination(True, True, "success")
        return ResidualTermination(False, False, "holding")
    return ResidualTermination(False, False, "")


def calculate_residual_reward(previous, current, action, config):
    """Reward measured connector progress and penalize unsafe residuals."""
    _validate_state(previous)
    _validate_state(current)
    decoded = decode_residual_action(action, config)
    previous_helix_error = previous.axial_travel_m - helical_travel(
        max(0.0, previous.nut_angle_rad), config.helical_lead_m
    )
    current_helix_error = current.axial_travel_m - helical_travel(
        max(0.0, current.nut_angle_rad), config.helical_lead_m
    )
    nut_delta = current.nut_angle_rad - previous.nut_angle_rad
    axial_delta = current.axial_travel_m - previous.axial_travel_m
    termination = evaluate_residual_state(current, config)
    terms = {
        "nut_progress": 8.0
        * float(np.clip(nut_delta / config.target_angle_rad, -0.10, 0.10)),
        "axial_progress": 2.0
        * float(
            np.clip(
                axial_delta / config.expected_axial_travel_m,
                -0.10,
                0.10,
            )
        ),
        "helix_improvement": 0.20
        * float(
            np.clip(
                (abs(previous_helix_error) - abs(current_helix_error))
                / config.helical_tolerance_m(current.nut_angle_rad),
                -1.0,
                1.0,
            )
        ),
        "loaded_channels": 0.005
        * loaded_torque_channels(current, config),
        "q7_tracking": -0.01
        * min(
            abs(current.q7_tracking_error_rad)
            / config.q7_tracking_error_scale_rad,
            2.0,
        ),
        "control": -0.002
        * float(np.dot(decoded.normalized, decoded.normalized)),
        "living": -0.020,
        "terminal": 0.0,
    }
    if termination.success:
        terms["terminal"] = 25.0
    elif termination.terminated:
        terms["terminal"] = (
            -25.0
            if termination.reason in {"invalid_physics", "q7_overspeed"}
            else -10.0
        )
    return ResidualReward(float(sum(terms.values())), terms)
