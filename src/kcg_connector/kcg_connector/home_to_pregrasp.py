"""Strict pure contract for deterministic Home-to-pregrasp motion v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


PREGRASP_SCHEMA_VERSION = "kcg_connector_home_to_pregrasp_v1"
DEFAULT_PREGRASP_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/connector_home_to_pregrasp_v1.yaml"
)
_PRIM_PATH_PATTERN = re.compile(
    r"^/World(?:/[A-Za-z_][A-Za-z0-9_]*)+$"
)
EXPECTED_ARM_JOINT_NAMES = tuple(
    f"iiwa_joint_{index}" for index in range(1, 8)
)
EXPECTED_ACTIVE_HAND_JOINT_NAMES = ("f1j1", "f1j2", "f2j1", "f3j2")
EXPECTED_SEGMENT_NAMES = (
    "home_to_safe_mid",
    "safe_mid_to_high_approach",
    "high_approach_to_pregrasp",
)
EXPECTED_SEGMENT_TARGETS = (
    (
        -0.1133152125,
        0.2419650715,
        -0.1716717785,
        -0.3550775790,
        0.0852878585,
        0.9833095885,
        -0.0414886690,
    ),
    (
        -0.1813043400,
        0.3871441144,
        -0.2746748456,
        -0.5681241264,
        0.1364605736,
        1.5732953416,
        -0.0663818704,
    ),
    (
        -0.2266304250,
        0.4839301430,
        -0.3433435570,
        -0.7101551580,
        0.1705757170,
        1.9666191770,
        -0.0829773380,
    ),
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_keys(value, expected, name):
    actual = set(value)
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        raise ValueError(
            f"{name} keys are invalid: missing={missing}; "
            f"unexpected={unexpected}"
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


def _vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    return tuple(
        _finite_float(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _text_tuple(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} names")
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or len(set(result)) != length:
        raise ValueError(f"{name} must contain unique nonempty names")
    return result


def _prim_path(value, name):
    if not isinstance(value, str) or not _PRIM_PATH_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be an absolute /World prim path")
    return value


def _strict_bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class PregraspScene:
    tabletop_config: str
    robot_root_prim_path: str
    articulation_prim_path: str
    grasp_tcp_prim_path: str


@dataclass(frozen=True)
class PregraspRobot:
    arm_joint_names: tuple[str, ...]
    active_hand_joint_names: tuple[str, ...]
    home_arm_rad: tuple[float, ...]
    open_hand_rad: tuple[float, ...]
    arm_stiffness: float
    arm_damping: float
    hand_stiffness: float
    hand_damping: float


@dataclass(frozen=True)
class PregraspSegment:
    name: str
    duration_s: float
    target_arm_rad: tuple[float, ...]


@dataclass(frozen=True)
class PregraspMotion:
    interpolation: str
    hand_open_duration_s: float
    segments: tuple[PregraspSegment, ...]
    hold_duration_s: float
    target_tcp_position_m: tuple[float, float, float]
    target_tcp_down_axis_world: tuple[float, float, float]


@dataclass(frozen=True)
class PregraspAcceptance:
    maximum_tcp_position_error_m: float
    maximum_tcp_axis_error_rad: float
    maximum_observed_arm_tracking_error_rad: float
    maximum_observed_hand_tracking_error_rad: float
    maximum_final_arm_tracking_error_rad: float
    maximum_joint_limit_violation_rad: float
    maximum_observed_joint_speed_rad_s: float
    maximum_final_joint_speed_rad_s: float
    maximum_loose_endpoint_xy_drift_m: float
    maximum_loose_endpoint_tail_displacement_m: float
    maximum_fixed_translation_drift_m: float
    maximum_fixed_rotation_drift_rad: float
    require_zero_robot_table_contacts: bool
    require_zero_robot_fixture_contacts: bool
    require_zero_robot_connector_contacts: bool


@dataclass(frozen=True)
class HomeToPregraspConfig:
    schema_version: str
    scene: PregraspScene
    robot: PregraspRobot
    motion: PregraspMotion
    acceptance: PregraspAcceptance

    def as_dict(self):
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


def minimum_jerk_blend(fraction):
    """Quintic blend with zero endpoint velocity and acceleration."""

    value = _finite_float(fraction, "fraction")
    if not 0.0 <= value <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    return value * value * value * (
        10.0 + value * (-15.0 + 6.0 * value)
    )


def interpolate_segment(start, target, fraction):
    """Interpolate one exact seven-joint minimum-jerk setpoint."""

    first = _vector(start, 7, "start")
    second = _vector(target, 7, "target")
    blend = minimum_jerk_blend(fraction)
    return tuple(
        initial + blend * (final - initial)
        for initial, final in zip(first, second)
    )


def _load_scene(value):
    document = _mapping(value, "scene")
    _exact_keys(
        document,
        (
            "tabletop_config",
            "robot_root_prim_path",
            "articulation_prim_path",
            "grasp_tcp_prim_path",
        ),
        "scene",
    )
    tabletop_config = document["tabletop_config"]
    if tabletop_config != "connector_tabletop_scene_v1.yaml":
        raise ValueError("scene tabletop config is not canonical v1")
    result = PregraspScene(
        tabletop_config=tabletop_config,
        robot_root_prim_path=_prim_path(
            document["robot_root_prim_path"], "scene.robot_root_prim_path"
        ),
        articulation_prim_path=_prim_path(
            document["articulation_prim_path"],
            "scene.articulation_prim_path",
        ),
        grasp_tcp_prim_path=_prim_path(
            document["grasp_tcp_prim_path"],
            "scene.grasp_tcp_prim_path",
        ),
    )
    if result.robot_root_prim_path != "/World/HandArm":
        raise ValueError("scene robot root is not canonical")
    expected_articulation = "/World/HandArm/Geometry/world"
    if result.articulation_prim_path != expected_articulation:
        raise ValueError("scene articulation path is not canonical")
    expected_tcp = (
        expected_articulation
        + "/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3"
        + "/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7"
        + "/iiwa_link_ee/handbase_link/grasp_tcp"
    )
    if result.grasp_tcp_prim_path != expected_tcp:
        raise ValueError("scene TCP path is not canonical")
    return result


def _load_robot(value):
    document = _mapping(value, "robot")
    fields = (
        "arm_joint_names",
        "active_hand_joint_names",
        "home_arm_rad",
        "open_hand_rad",
        "arm_stiffness",
        "arm_damping",
        "hand_stiffness",
        "hand_damping",
    )
    _exact_keys(document, fields, "robot")
    result = PregraspRobot(
        arm_joint_names=_text_tuple(
            document["arm_joint_names"], 7, "robot.arm_joint_names"
        ),
        active_hand_joint_names=_text_tuple(
            document["active_hand_joint_names"],
            4,
            "robot.active_hand_joint_names",
        ),
        home_arm_rad=_vector(
            document["home_arm_rad"], 7, "robot.home_arm_rad"
        ),
        open_hand_rad=_vector(
            document["open_hand_rad"], 4, "robot.open_hand_rad"
        ),
        **{
            field: _positive_float(document[field], f"robot.{field}")
            for field in fields[-4:]
        },
    )
    if result.arm_joint_names != EXPECTED_ARM_JOINT_NAMES:
        raise ValueError("robot arm joint names are not canonical")
    if result.active_hand_joint_names != EXPECTED_ACTIVE_HAND_JOINT_NAMES:
        raise ValueError("robot active hand joint names are not canonical")
    if result.home_arm_rad != (0.0,) * 7:
        raise ValueError("robot Home arm must be all zero")
    if result.open_hand_rad != (1.0, 0.0, 0.0, 0.0):
        raise ValueError("robot open hand target is not canonical")
    expected_gains = (24000.0, 400.0, 25.0, 2.0)
    actual_gains = (
        result.arm_stiffness,
        result.arm_damping,
        result.hand_stiffness,
        result.hand_damping,
    )
    if actual_gains != expected_gains:
        raise ValueError("robot PD gains are not canonical v1 values")
    return result


def _load_motion(value):
    document = _mapping(value, "motion")
    _exact_keys(
        document,
        (
            "interpolation",
            "hand_open_duration_s",
            "segments",
            "hold_duration_s",
            "target_tcp_position_m",
            "target_tcp_down_axis_world",
        ),
        "motion",
    )
    if document["interpolation"] != "minimum_jerk":
        raise ValueError("motion interpolation must be minimum_jerk")
    raw_segments = document["segments"]
    if not isinstance(raw_segments, list) or len(raw_segments) != 3:
        raise ValueError("motion must contain exactly three segments")
    segments = []
    for index, raw in enumerate(raw_segments):
        segment = _mapping(raw, f"motion.segments[{index}]")
        _exact_keys(
            segment,
            ("name", "duration_s", "target_arm_rad"),
            f"motion.segments[{index}]",
        )
        segments.append(
            PregraspSegment(
                name=str(segment["name"]),
                duration_s=_positive_float(
                    segment["duration_s"],
                    f"motion.segments[{index}].duration_s",
                ),
                target_arm_rad=_vector(
                    segment["target_arm_rad"],
                    7,
                    f"motion.segments[{index}].target_arm_rad",
                ),
            )
        )
    result = PregraspMotion(
        interpolation="minimum_jerk",
        hand_open_duration_s=_positive_float(
            document["hand_open_duration_s"],
            "motion.hand_open_duration_s",
        ),
        segments=tuple(segments),
        hold_duration_s=_positive_float(
            document["hold_duration_s"], "motion.hold_duration_s"
        ),
        target_tcp_position_m=_vector(
            document["target_tcp_position_m"],
            3,
            "motion.target_tcp_position_m",
        ),
        target_tcp_down_axis_world=_vector(
            document["target_tcp_down_axis_world"],
            3,
            "motion.target_tcp_down_axis_world",
        ),
    )
    if tuple(segment.name for segment in result.segments) != (
        EXPECTED_SEGMENT_NAMES
    ):
        raise ValueError("motion segment names or order are not canonical")
    if tuple(
        segment.target_arm_rad for segment in result.segments
    ) != EXPECTED_SEGMENT_TARGETS:
        raise ValueError("motion joint waypoints are not canonical v1 seeds")
    if any(segment.duration_s < 2.5 for segment in result.segments):
        raise ValueError(
            "motion segment duration must be at least 2.5 seconds"
        )
    if result.hand_open_duration_s < 2.0:
        raise ValueError("Home hand opening must be at least two seconds")
    durations = [
        (result.hand_open_duration_s, "hand_open_duration_s"),
        (result.hold_duration_s, "hold_duration_s"),
    ]
    durations.extend(
        (segment.duration_s, segment.name)
        for segment in result.segments
    )
    for duration, name in durations:
        samples = duration * 240.0
        if not math.isclose(samples, round(samples), abs_tol=1.0e-9):
            raise ValueError(f"motion {name} must contain whole 240 Hz steps")

    # Minimum-jerk peak speed is exactly 1.875 * |delta| / duration.
    # Keep arm joints at or below the explicit 0.3 rad/s screening bound and
    # the active hand below 1.0 rad/s; this is approximately 0.2 of the arm's
    # 1.5 rad/s MoveIt override, not a collision-planner velocity claim.
    previous = (0.0,) * 7
    for segment in result.segments:
        peak = max(
            1.875 * abs(target - start) / segment.duration_s
            for start, target in zip(previous, segment.target_arm_rad)
        )
        if peak > 0.300 + 1.0e-9:
            raise ValueError("motion segment exceeds 0.3 rad/s peak speed")
        previous = segment.target_arm_rad
    hand_peak = 1.875 / result.hand_open_duration_s
    if hand_peak > 1.0:
        raise ValueError("Home hand opening exceeds 1.0 rad/s peak speed")
    if result.hold_duration_s < 1.0:
        raise ValueError("pregrasp hold must be at least one second")
    if result.target_tcp_position_m != (0.520, -0.210, 0.360):
        raise ValueError("motion TCP target is not tabletop-v1 pregrasp")
    axis_norm = math.sqrt(
        sum(value * value for value in result.target_tcp_down_axis_world)
    )
    if not math.isclose(axis_norm, 1.0, abs_tol=1.0e-9):
        raise ValueError("motion TCP down axis must be unit length")
    if result.target_tcp_down_axis_world != (0.0, 0.0, -1.0):
        raise ValueError("motion TCP target axis must point down")
    return result


def _load_acceptance(value):
    document = _mapping(value, "acceptance")
    numeric_fields = (
        "maximum_tcp_position_error_m",
        "maximum_tcp_axis_error_rad",
        "maximum_observed_arm_tracking_error_rad",
        "maximum_observed_hand_tracking_error_rad",
        "maximum_final_arm_tracking_error_rad",
        "maximum_joint_limit_violation_rad",
        "maximum_observed_joint_speed_rad_s",
        "maximum_final_joint_speed_rad_s",
        "maximum_loose_endpoint_xy_drift_m",
        "maximum_loose_endpoint_tail_displacement_m",
        "maximum_fixed_translation_drift_m",
        "maximum_fixed_rotation_drift_rad",
    )
    boolean_fields = (
        "require_zero_robot_table_contacts",
        "require_zero_robot_fixture_contacts",
        "require_zero_robot_connector_contacts",
    )
    _exact_keys(document, numeric_fields + boolean_fields, "acceptance")
    result = PregraspAcceptance(
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
        "maximum_tcp_position_error_m": 0.002,
        "maximum_tcp_axis_error_rad": math.radians(2.0),
        "maximum_observed_arm_tracking_error_rad": 0.020,
        "maximum_observed_hand_tracking_error_rad": 0.020,
        "maximum_final_arm_tracking_error_rad": 0.020,
        "maximum_joint_limit_violation_rad": 0.020,
        "maximum_observed_joint_speed_rad_s": 1.0,
        "maximum_final_joint_speed_rad_s": 0.030,
        "maximum_loose_endpoint_xy_drift_m": 0.001,
        "maximum_loose_endpoint_tail_displacement_m": 0.002,
        "maximum_fixed_translation_drift_m": 0.000001,
        "maximum_fixed_rotation_drift_rad": 0.00001,
    }
    for field, upper in upper_bounds.items():
        if getattr(result, field) > upper:
            raise ValueError(f"acceptance {field} exceeds safety bound")
    if not all(getattr(result, field) for field in boolean_fields):
        raise ValueError("all zero-contact requirements must be enabled")
    return result


def load_home_to_pregrasp_config(
    config_path: str | Path = DEFAULT_PREGRASP_CONFIG_PATH,
):
    """Load and fail-closed validate Home-to-pregrasp v1 YAML."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "pregrasp config")
    _exact_keys(
        document,
        ("schema_version", "scene", "robot", "motion", "acceptance"),
        "pregrasp config",
    )
    if document["schema_version"] != PREGRASP_SCHEMA_VERSION:
        raise ValueError("unsupported Home-to-pregrasp schema")
    result = HomeToPregraspConfig(
        schema_version=PREGRASP_SCHEMA_VERSION,
        scene=_load_scene(document["scene"]),
        robot=_load_robot(document["robot"]),
        motion=_load_motion(document["motion"]),
        acceptance=_load_acceptance(document["acceptance"]),
    )
    result.as_dict()
    return result


__all__ = [
    "DEFAULT_PREGRASP_CONFIG_PATH",
    "EXPECTED_ACTIVE_HAND_JOINT_NAMES",
    "EXPECTED_ARM_JOINT_NAMES",
    "HomeToPregraspConfig",
    "PREGRASP_SCHEMA_VERSION",
    "interpolate_segment",
    "load_home_to_pregrasp_config",
    "minimum_jerk_blend",
]
