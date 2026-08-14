"""Strict pure contract for deterministic tabletop pick smoke v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import yaml


PICK_SCHEMA_VERSION = "kcg_connector_tabletop_pick_v1"
DEFAULT_PICK_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/connector_tabletop_pick_v1.yaml"
)
EXPECTED_GRASP_ARM_RAD = (
    -0.164155590,
    0.426740717,
    -0.376494151,
    -0.980754913,
    0.155526632,
    1.758561897,
    -0.096543358,
)
EXPECTED_TORQUE_JOINT_NAMES = ("f1j2", "f2j1", "f3j2")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_keys(value, expected, name):
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise ValueError(
            f"{name} keys are invalid: "
            f"missing={sorted(required - actual)}; "
            f"unexpected={sorted(actual - required)}"
        )


def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value, name):
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value, name):
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    return tuple(
        _finite_float(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _names(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} names")
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or len(set(result)) != length:
        raise ValueError(f"{name} names must be unique and nonempty")
    return result


def _strict_bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class PickPregrasp:
    config: str


@dataclass(frozen=True)
class PickMotion:
    interpolation: str
    grasp_arm_rad: tuple[float, ...]
    grasp_tcp_position_m: tuple[float, float, float]
    grasp_tcp_down_axis_world: tuple[float, float, float]
    descent_duration_s: float
    open_tare_duration_s: float
    grasp_hand_rad: tuple[float, ...]
    closure_duration_s: float
    preload_duration_s: float
    lift_duration_s: float
    final_hold_duration_s: float
    effort_sample_duration_s: float
    grip_hand_stiffness: float
    grip_hand_damping: float
    grip_static_friction: float
    grip_dynamic_friction: float
    grip_restitution: float


@dataclass(frozen=True)
class PickSensing:
    torque_joint_names: tuple[str, ...]
    loaded_torque_threshold_nm: float
    minimum_loaded_channels: int
    maximum_absolute_torque_delta_nm: float
    fingertip_tactile_available: bool


@dataclass(frozen=True)
class PickAcceptance:
    maximum_grasp_tcp_position_error_m: float
    maximum_grasp_tcp_axis_error_rad: float
    maximum_arm_tracking_error_rad: float
    maximum_endpoint_arm_tracking_error_rad: float
    maximum_joint_limit_violation_rad: float
    maximum_observed_joint_speed_rad_s: float
    maximum_final_observable_joint_speed_rad_s: float
    maximum_final_post_solver_joint_speed_rad_s: float
    minimum_body_lift_m: float
    minimum_final_bottom_clearance_m: float
    maximum_body_tcp_slip_m: float
    maximum_body_nut_separation_change_m: float
    maximum_final_hold_displacement_m: float
    maximum_final_body_linear_speed_m_s: float
    maximum_final_body_angular_speed_rad_s: float
    maximum_fixed_translation_drift_m: float
    maximum_fixed_rotation_drift_rad: float
    require_zero_preclosure_robot_connector_contacts: bool
    require_zero_robot_table_contacts: bool
    require_zero_robot_fixture_contacts: bool
    require_zero_robot_fixed_endpoint_contacts: bool
    require_zero_final_plug_table_contacts: bool
    require_physical_grip_contact: bool


@dataclass(frozen=True)
class TabletopPickConfig:
    schema_version: str
    pregrasp: PickPregrasp
    motion: PickMotion
    sensing: PickSensing
    acceptance: PickAcceptance

    def as_dict(self):
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


def _load_motion(value):
    document = _mapping(value, "motion")
    fields = (
        "interpolation",
        "grasp_arm_rad",
        "grasp_tcp_position_m",
        "grasp_tcp_down_axis_world",
        "descent_duration_s",
        "open_tare_duration_s",
        "grasp_hand_rad",
        "closure_duration_s",
        "preload_duration_s",
        "lift_duration_s",
        "final_hold_duration_s",
        "effort_sample_duration_s",
        "grip_hand_stiffness",
        "grip_hand_damping",
        "grip_static_friction",
        "grip_dynamic_friction",
        "grip_restitution",
    )
    _exact_keys(document, fields, "motion")
    if document["interpolation"] != "minimum_jerk":
        raise ValueError("motion interpolation must be minimum_jerk")
    duration_fields = (
        "descent_duration_s",
        "open_tare_duration_s",
        "closure_duration_s",
        "preload_duration_s",
        "lift_duration_s",
        "final_hold_duration_s",
        "effort_sample_duration_s",
    )
    result = PickMotion(
        interpolation="minimum_jerk",
        grasp_arm_rad=_vector(
            document["grasp_arm_rad"], 7, "motion.grasp_arm_rad"
        ),
        grasp_tcp_position_m=_vector(
            document["grasp_tcp_position_m"],
            3,
            "motion.grasp_tcp_position_m",
        ),
        grasp_tcp_down_axis_world=_vector(
            document["grasp_tcp_down_axis_world"],
            3,
            "motion.grasp_tcp_down_axis_world",
        ),
        grasp_hand_rad=_vector(
            document["grasp_hand_rad"], 4, "motion.grasp_hand_rad"
        ),
        **{
            field: _positive_float(document[field], f"motion.{field}")
            for field in duration_fields
        },
        grip_hand_stiffness=_positive_float(
            document["grip_hand_stiffness"],
            "motion.grip_hand_stiffness",
        ),
        grip_hand_damping=_positive_float(
            document["grip_hand_damping"],
            "motion.grip_hand_damping",
        ),
        grip_static_friction=_nonnegative_float(
            document["grip_static_friction"],
            "motion.grip_static_friction",
        ),
        grip_dynamic_friction=_nonnegative_float(
            document["grip_dynamic_friction"],
            "motion.grip_dynamic_friction",
        ),
        grip_restitution=_nonnegative_float(
            document["grip_restitution"], "motion.grip_restitution"
        ),
    )
    if result.grasp_arm_rad != EXPECTED_GRASP_ARM_RAD:
        raise ValueError("motion grasp arm seed is not canonical v1")
    if result.grasp_tcp_position_m != (0.520, -0.210, 0.291):
        raise ValueError("motion grasp TCP position is not canonical v1")
    if result.grasp_tcp_down_axis_world != (0.0, 0.0, -1.0):
        raise ValueError("motion grasp TCP axis must point down")
    if result.grasp_hand_rad != (1.0, 0.75, 0.50, 0.75):
        raise ValueError("motion grasp hand target is not canonical v1")
    expected_durations = (2.5, 0.5, 3.0, 0.5, 2.5, 4.0, 0.5)
    actual_durations = tuple(
        getattr(result, field) for field in duration_fields
    )
    if actual_durations != expected_durations:
        raise ValueError("motion phase durations are not canonical v1")
    for duration in actual_durations:
        samples = duration * 240.0
        if not math.isclose(samples, round(samples), abs_tol=1.0e-9):
            raise ValueError(
                "motion durations must contain whole 240 Hz steps"
            )
    if (
        result.effort_sample_duration_s > result.open_tare_duration_s
        or result.effort_sample_duration_s > result.preload_duration_s
        or result.effort_sample_duration_s > result.final_hold_duration_s
    ):
        raise ValueError("effort sample window exceeds a sampling phase")
    expected_physics = (5.0, 1.0, 1.4, 1.4, 0.0)
    actual_physics = (
        result.grip_hand_stiffness,
        result.grip_hand_damping,
        result.grip_static_friction,
        result.grip_dynamic_friction,
        result.grip_restitution,
    )
    if actual_physics != expected_physics:
        raise ValueError("motion grip physics values are not canonical v1")
    pregrasp_arm = (
        -0.2266304250,
        0.4839301430,
        -0.3433435570,
        -0.7101551580,
        0.1705757170,
        1.9666191770,
        -0.0829773380,
    )
    descent_peak = max(
        1.875 * abs(end - start) / result.descent_duration_s
        for start, end in zip(pregrasp_arm, result.grasp_arm_rad)
    )
    lift_peak = max(
        1.875 * abs(end - start) / result.lift_duration_s
        for start, end in zip(result.grasp_arm_rad, pregrasp_arm)
    )
    closure_peak = max(
        1.875 * abs(end - start) / result.closure_duration_s
        for start, end in zip((1.0, 0.0, 0.0, 0.0), result.grasp_hand_rad)
    )
    if max(descent_peak, lift_peak) > 0.3 + 1.0e-9:
        raise ValueError("motion arm phase exceeds 0.3 rad/s peak speed")
    if closure_peak > 0.5 + 1.0e-9:
        raise ValueError("motion closure exceeds 0.5 rad/s peak speed")
    return result


def _load_sensing(value):
    document = _mapping(value, "sensing")
    fields = (
        "torque_joint_names",
        "loaded_torque_threshold_nm",
        "minimum_loaded_channels",
        "maximum_absolute_torque_delta_nm",
        "fingertip_tactile_available",
    )
    _exact_keys(document, fields, "sensing")
    result = PickSensing(
        torque_joint_names=_names(
            document["torque_joint_names"], 3, "sensing.torque_joint_names"
        ),
        loaded_torque_threshold_nm=_positive_float(
            document["loaded_torque_threshold_nm"],
            "sensing.loaded_torque_threshold_nm",
        ),
        minimum_loaded_channels=_positive_integer(
            document["minimum_loaded_channels"],
            "sensing.minimum_loaded_channels",
        ),
        maximum_absolute_torque_delta_nm=_positive_float(
            document["maximum_absolute_torque_delta_nm"],
            "sensing.maximum_absolute_torque_delta_nm",
        ),
        fingertip_tactile_available=_strict_bool(
            document["fingertip_tactile_available"],
            "sensing.fingertip_tactile_available",
        ),
    )
    if result.torque_joint_names != EXPECTED_TORQUE_JOINT_NAMES:
        raise ValueError("sensing torque channels are not the real base axes")
    if result.loaded_torque_threshold_nm != 0.020:
        raise ValueError("sensing loaded threshold is not canonical v1")
    if result.minimum_loaded_channels != 2:
        raise ValueError("sensing must require exactly two loaded channels")
    if result.maximum_absolute_torque_delta_nm != 1.0:
        raise ValueError("sensing torque safety limit is not canonical v1")
    if result.fingertip_tactile_available:
        raise ValueError("fingertip tactile must remain unavailable")
    return result


def _load_acceptance(value):
    document = _mapping(value, "acceptance")
    numeric_fields = (
        "maximum_grasp_tcp_position_error_m",
        "maximum_grasp_tcp_axis_error_rad",
        "maximum_arm_tracking_error_rad",
        "maximum_endpoint_arm_tracking_error_rad",
        "maximum_joint_limit_violation_rad",
        "maximum_observed_joint_speed_rad_s",
        "maximum_final_observable_joint_speed_rad_s",
        "maximum_final_post_solver_joint_speed_rad_s",
        "minimum_body_lift_m",
        "minimum_final_bottom_clearance_m",
        "maximum_body_tcp_slip_m",
        "maximum_body_nut_separation_change_m",
        "maximum_final_hold_displacement_m",
        "maximum_final_body_linear_speed_m_s",
        "maximum_final_body_angular_speed_rad_s",
        "maximum_fixed_translation_drift_m",
        "maximum_fixed_rotation_drift_rad",
    )
    boolean_fields = (
        "require_zero_preclosure_robot_connector_contacts",
        "require_zero_robot_table_contacts",
        "require_zero_robot_fixture_contacts",
        "require_zero_robot_fixed_endpoint_contacts",
        "require_zero_final_plug_table_contacts",
        "require_physical_grip_contact",
    )
    _exact_keys(document, numeric_fields + boolean_fields, "acceptance")
    result = PickAcceptance(
        **{
            field: _positive_float(document[field], f"acceptance.{field}")
            for field in numeric_fields
        },
        **{
            field: _strict_bool(document[field], f"acceptance.{field}")
            for field in boolean_fields
        },
    )
    upper_bounds = {
        "maximum_grasp_tcp_position_error_m": 0.003,
        "maximum_grasp_tcp_axis_error_rad": math.radians(2.0),
        "maximum_arm_tracking_error_rad": 0.020,
        "maximum_endpoint_arm_tracking_error_rad": 0.020,
        "maximum_joint_limit_violation_rad": 0.020,
        "maximum_observed_joint_speed_rad_s": 1.0,
        "maximum_final_observable_joint_speed_rad_s": 0.030,
        "maximum_final_post_solver_joint_speed_rad_s": 0.050,
        "maximum_body_tcp_slip_m": 0.005,
        "maximum_body_nut_separation_change_m": 0.001,
        "maximum_final_hold_displacement_m": 0.002,
        "maximum_final_body_linear_speed_m_s": 0.020,
        "maximum_final_body_angular_speed_rad_s": 0.200,
        "maximum_fixed_translation_drift_m": 0.000001,
        "maximum_fixed_rotation_drift_rad": 0.00001,
    }
    for field, upper in upper_bounds.items():
        if getattr(result, field) > upper:
            raise ValueError(f"acceptance {field} exceeds safety bound")
    if result.minimum_body_lift_m < 0.040:
        raise ValueError("acceptance body lift must be at least 40 mm")
    if result.minimum_final_bottom_clearance_m < 0.030:
        raise ValueError("acceptance bottom clearance must be at least 30 mm")
    if not all(getattr(result, field) for field in boolean_fields):
        raise ValueError("all physical contact requirements must be enabled")
    return result


def load_tabletop_pick_config(
    config_path: str | Path = DEFAULT_PICK_CONFIG_PATH,
):
    """Load and fail-closed validate tabletop pick v1 YAML."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "pick config")
    _exact_keys(
        document,
        ("schema_version", "pregrasp", "motion", "sensing", "acceptance"),
        "pick config",
    )
    if document["schema_version"] != PICK_SCHEMA_VERSION:
        raise ValueError("unsupported tabletop pick schema")
    pregrasp_document = _mapping(document["pregrasp"], "pregrasp")
    _exact_keys(pregrasp_document, ("config",), "pregrasp")
    if pregrasp_document["config"] != (
        "connector_home_to_pregrasp_v1.yaml"
    ):
        raise ValueError("pick pregrasp config is not canonical v1")
    result = TabletopPickConfig(
        schema_version=PICK_SCHEMA_VERSION,
        pregrasp=PickPregrasp(config=pregrasp_document["config"]),
        motion=_load_motion(document["motion"]),
        sensing=_load_sensing(document["sensing"]),
        acceptance=_load_acceptance(document["acceptance"]),
    )
    result.as_dict()
    return result


__all__ = [
    "DEFAULT_PICK_CONFIG_PATH",
    "EXPECTED_GRASP_ARM_RAD",
    "EXPECTED_TORQUE_JOINT_NAMES",
    "PICK_SCHEMA_VERSION",
    "TabletopPickConfig",
    "load_tabletop_pick_config",
]
