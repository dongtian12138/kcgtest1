"""Versioned state and success criteria for the connector prototype."""

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path

import yaml

from .geometry import helical_travel


class ConnectorPhase(str, Enum):
    """Deterministic first-stage connector curriculum phases."""

    GRASP = "GRASP"
    PREALIGN = "PREALIGN"
    INSERT = "INSERT"
    ENGAGE = "ENGAGE"
    SCREW = "SCREW"
    HOLD = "HOLD"
    PASSED = "PASSED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ConnectorTaskConfig:
    """Synthetic prototype thresholds, not certified connector dimensions."""

    maximum_grasp_distance: float = 0.015
    minimum_loaded_torque_channels: int = 2
    maximum_absolute_finger_torque: float = 1.0
    lateral_alignment_tolerance: float = 0.0015
    angular_alignment_tolerance: float = math.radians(2.0)
    key_alignment_tolerance: float = math.radians(5.0)
    engage_depth: float = 0.010
    target_coupling_angle: float = 2.0 * math.pi
    coupling_angle_tolerance: float = math.radians(2.0)
    helical_lead: float = 0.004
    helical_error_tolerance: float = 0.0005
    hold_duration: float = 2.0


def load_connector_task_config(config_path):
    """Load the versioned success thresholds from the task YAML."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        success = document["success"]
        config = ConnectorTaskConfig(
            maximum_grasp_distance=float(success["maximum_grasp_distance"]),
            minimum_loaded_torque_channels=int(
                success["minimum_loaded_torque_channels"]
            ),
            maximum_absolute_finger_torque=float(
                success["maximum_absolute_finger_torque"]
            ),
            lateral_alignment_tolerance=float(
                success["lateral_alignment_tolerance"]
            ),
            angular_alignment_tolerance=math.radians(
                float(success["angular_alignment_tolerance_degrees"])
            ),
            key_alignment_tolerance=math.radians(
                float(success["key_alignment_tolerance_degrees"])
            ),
            engage_depth=float(success["engage_depth"]),
            target_coupling_angle=math.radians(
                float(success["target_coupling_angle_degrees"])
            ),
            coupling_angle_tolerance=math.radians(
                float(success["coupling_angle_tolerance_degrees"])
            ),
            helical_lead=float(success["helical_lead_per_revolution"]),
            helical_error_tolerance=float(
                success["helical_error_tolerance"]
            ),
            hold_duration=float(success["hold_duration"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid connector task config: {path}") from error

    positive_values = (
        config.maximum_grasp_distance,
        config.maximum_absolute_finger_torque,
        config.lateral_alignment_tolerance,
        config.angular_alignment_tolerance,
        config.key_alignment_tolerance,
        config.engage_depth,
        config.target_coupling_angle,
        config.coupling_angle_tolerance,
        config.helical_lead,
        config.helical_error_tolerance,
        config.hold_duration,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in positive_values):
        raise ValueError("connector success thresholds must be finite and positive")
    if not 1 <= config.minimum_loaded_torque_channels <= 3:
        raise ValueError("loaded torque channels must be between one and three")
    if config.coupling_angle_tolerance >= config.target_coupling_angle:
        raise ValueError("coupling angle tolerance must be smaller than the target")
    return config


@dataclass(frozen=True)
class ConnectorMetrics:
    """Minimal truth metrics shared by scripted control and later RL."""

    grasp_distance: float
    loaded_torque_channels: int
    maximum_absolute_finger_torque: float
    lateral_error: float
    angular_error: float
    key_error: float
    insertion_depth: float
    coupling_angle: float
    axial_lock_travel: float
    hold_seconds: float
    engaged: bool


@dataclass(frozen=True)
class TaskEvaluation:
    """Result of evaluating one connector task state."""

    phase: ConnectorPhase
    success: bool
    failure_reason: str
    expected_lock_travel: float
    helical_error: float


def _finite(metrics):
    values = (
        metrics.grasp_distance,
        metrics.maximum_absolute_finger_torque,
        metrics.lateral_error,
        metrics.angular_error,
        metrics.key_error,
        metrics.insertion_depth,
        metrics.coupling_angle,
        metrics.axial_lock_travel,
        metrics.hold_seconds,
    )
    return all(math.isfinite(value) for value in values)


def _result(phase, expected, error, success=False, failure_reason=""):
    return TaskEvaluation(
        phase=phase,
        success=success,
        failure_reason=failure_reason,
        expected_lock_travel=expected,
        helical_error=error,
    )


def evaluate_connector_task(metrics, config=ConnectorTaskConfig()):
    """Evaluate the current deterministic curriculum phase and termination."""
    if not _finite(metrics):
        return _result(
            ConnectorPhase.FAILED, 0.0, math.inf, failure_reason="invalid_physics"
        )

    coupling_angle = max(0.0, metrics.coupling_angle)
    expected = helical_travel(coupling_angle, config.helical_lead)
    helical_error = metrics.axial_lock_travel - expected

    if metrics.grasp_distance > config.maximum_grasp_distance:
        return _result(
            ConnectorPhase.FAILED,
            expected,
            helical_error,
            failure_reason="lost_grasp",
        )
    if (
        metrics.maximum_absolute_finger_torque
        > config.maximum_absolute_finger_torque
    ):
        return _result(
            ConnectorPhase.FAILED,
            expected,
            helical_error,
            failure_reason="finger_overload",
        )
    if (
        metrics.loaded_torque_channels
        < config.minimum_loaded_torque_channels
    ):
        return _result(ConnectorPhase.GRASP, expected, helical_error)

    aligned = (
        metrics.lateral_error <= config.lateral_alignment_tolerance
        and metrics.angular_error <= config.angular_alignment_tolerance
        and abs(metrics.key_error) <= config.key_alignment_tolerance
    )
    if not aligned:
        if metrics.insertion_depth > 0.001:
            return _result(
                ConnectorPhase.FAILED,
                expected,
                helical_error,
                failure_reason="misaligned",
            )
        return _result(ConnectorPhase.PREALIGN, expected, helical_error)

    if metrics.insertion_depth < config.engage_depth:
        return _result(ConnectorPhase.INSERT, expected, helical_error)
    if not metrics.engaged:
        return _result(ConnectorPhase.ENGAGE, expected, helical_error)

    angle_complete = coupling_angle >= (
        config.target_coupling_angle - config.coupling_angle_tolerance
    )
    helix_valid = abs(helical_error) <= config.helical_error_tolerance
    if coupling_angle >= 0.25 * config.target_coupling_angle and not helix_valid:
        return _result(
            ConnectorPhase.FAILED,
            expected,
            helical_error,
            failure_reason="cross_thread",
        )
    if not angle_complete:
        return _result(ConnectorPhase.SCREW, expected, helical_error)
    if not helix_valid:
        return _result(
            ConnectorPhase.FAILED,
            expected,
            helical_error,
            failure_reason="stalled",
        )
    if metrics.hold_seconds < config.hold_duration:
        return _result(ConnectorPhase.HOLD, expected, helical_error)
    return _result(
        ConnectorPhase.PASSED,
        expected,
        helical_error,
        success=True,
    )
