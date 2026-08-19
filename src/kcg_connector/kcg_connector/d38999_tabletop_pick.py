"""Pure fail-closed contract for the independent D38999 pick candidate.

The candidate is intentionally separate from the accepted synthetic pick.
It contains a new IK endpoint and a geometry-only finger-envelope screen, but
does not claim Isaac dynamics, collision planning, or self-collision proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

D38999_PICK_SCHEMA_VERSION = "kcg_d38999_tabletop_pick_v1"
# FAST_CANDIDATE: identical geometry, IK targets and acceptance gates as
# v1; only motion.final_hold_duration_s may differ (2.0 s instead of
# 4.0 s).  The canonical duration tuple below switches on this exact
# schema string, so a v1 config never silently accepts shortened holds.
D38999_PICK_SCHEMA_VERSION_V2_FAST = "kcg_d38999_tabletop_pick_v2_fast"
D38999_PICK_SCHEMA_VERSION_KEYED_V2 = (
    "kcg_d38999_keyed_v2_tabletop_pick_v1"
)
D38999_PICK_SCHEMA_VERSION_KEYED_V2_XYCOMP_CANDIDATE = (
    "kcg_d38999_keyed_v2_tabletop_pick_xycomp_candidate_v1"
)
D38999_PICK_SCHEMA_VERSION_MULTILAYER_GRASP = (
    "kcg_d38999_multilayer_tabletop_pick_grasp_v1"
)
_KEYED_V2_PICK_SCHEMAS = frozenset(
    (
        D38999_PICK_SCHEMA_VERSION_KEYED_V2,
        D38999_PICK_SCHEMA_VERSION_KEYED_V2_XYCOMP_CANDIDATE,
        D38999_PICK_SCHEMA_VERSION_MULTILAYER_GRASP,
    )
)
DEFAULT_D38999_PICK_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config/d38999_tabletop_pick_v1.yaml"
)
EXPECTED_D38999_GRASP_ARM_RAD = (
    -0.129948040338,
    0.415863528857,
    -0.404365705983,
    -1.108380667037,
    0.160080395004,
    1.646562177393,
    -0.107043622479,
)
EXPECTED_D38999_CLOSURE_CLEARANCE_ARM_RAD = (
    -0.129948040338,
    0.415863528857,
    -0.404365705983,
    -1.108380667037,
    0.160080395004,
    1.646562177393,
    -0.107043622479,
)
EXPECTED_D38999_KEYED_V2_GRASP_ARM_RAD = (
    -0.129340616881,
    0.415154160813,
    -0.403066037272,
    -1.119324443931,
    0.159214032504,
    1.636235747425,
    -0.107043622479,
)
EXPECTED_D38999_KEYED_V2_GRASP_TCP_POSITION_M = (
    0.520,
    -0.210,
    0.24448,
)
EXPECTED_D38999_KEYED_V2_XYCOMP_GRASP_ARM_RAD = (
    -0.130492893831,
    0.416770989849,
    -0.402233115109,
    -1.116975714771,
    0.159495264823,
    1.636923213041,
    -0.107043622479,
)
EXPECTED_D38999_KEYED_V2_XYCOMP_GRASP_TCP_POSITION_M = (
    0.520574651334,
    -0.210265856035,
    0.24448,
)
EXPECTED_TORQUE_JOINT_NAMES = ("f1j2", "f2j1", "f3j2")
EXPECTED_ARM_JOINT_NAMES = tuple(
    f"iiwa_joint_{index}" for index in range(1, 8)
)
EXPECTED_ACTIVE_HAND_JOINT_NAMES = ("f1j1", "f1j2", "f2j1", "f3j2")
EXPECTED_APPROACH_NAMES = (
    "home_to_safe_mid",
    "safe_mid_to_high_approach",
    "high_approach_to_pregrasp",
)
EXPECTED_APPROACH_TARGETS = (
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
_PRIM_PATH_PATTERN = re.compile(r"^/World(?:/[A-Za-z_][A-Za-z0-9_]*)+$")
_KEYED_V2_PROFILE_ID = (
    "d38999_shell25j_25_61_n_keyed_physical_pair_v3"
)
_MULTILAYER_GRASP_PROFILE_ID = "D38999_ASSEMBLY_CONTROL_V1"
_PICK_PROFILE_ALLOWLIST = {
    D38999_PICK_SCHEMA_VERSION: {
        "tabletop_config": "d38999_tabletop_scene_v1.yaml",
        "source_config": "d38999_shell25j_proxy_v1.yaml",
        "asset_profile_id": "d38999_shell25j_61_pair_proxy_v1",
    },
    D38999_PICK_SCHEMA_VERSION_V2_FAST: {
        "tabletop_config": "d38999_tabletop_scene_v1.yaml",
        "source_config": "d38999_shell25j_proxy_v1.yaml",
        "asset_profile_id": "d38999_shell25j_61_pair_proxy_v1",
    },
    D38999_PICK_SCHEMA_VERSION_KEYED_V2: {
        "tabletop_config": "d38999_keyed_v2_tabletop_scene_v1.yaml",
        "source_config": "d38999_keyed_v2_physical_model_contract_v1.yaml",
        "asset_profile_id": _KEYED_V2_PROFILE_ID,
    },
    D38999_PICK_SCHEMA_VERSION_KEYED_V2_XYCOMP_CANDIDATE: {
        "tabletop_config": "d38999_keyed_v2_tabletop_scene_v1.yaml",
        "source_config": "d38999_keyed_v2_physical_model_contract_v1.yaml",
        "asset_profile_id": _KEYED_V2_PROFILE_ID,
    },
    D38999_PICK_SCHEMA_VERSION_MULTILAYER_GRASP: {
        "tabletop_config": "d38999_multilayer_tabletop_scene_grasp_v1.yaml",
        "source_config": "d38999_master_model_contract_v1.yaml",
        "asset_profile_id": _MULTILAYER_GRASP_PROFILE_ID,
    },
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected, label: str) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ValueError(
            f"{label} keys are invalid: "
            f"missing={sorted(wanted - actual)}; "
            f"unexpected={sorted(actual - wanted)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} numbers")
    return tuple(
        _finite(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


def _names(value: Any, length: int, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} names")
    result = tuple(
        _text(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{label} names must be unique")
    return result


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _prim_path(value: Any, label: str) -> str:
    result = _text(value, label)
    if not _PRIM_PATH_PATTERN.fullmatch(result):
        raise ValueError(f"{label} must be an absolute /World prim path")
    return result


@dataclass(frozen=True)
class D38999PickScene:
    tabletop_config: str
    proxy_config: str
    robot_asset: str
    robot_root_prim_path: str
    articulation_prim_path: str
    grasp_tcp_prim_path: str


@dataclass(frozen=True)
class D38999GeometryCandidate:
    loose_settled_origin_m: tuple[float, float, float]
    rear_body_radius_m: float
    coupling_nut_outer_radius_m: float
    rear_body_world_z_interval_m: tuple[float, float]
    coupling_nut_world_z_interval_m: tuple[float, float]
    grip_local_z_interval_m: tuple[float, float]
    handbase_to_tcp_m: float
    maximum_closed_finger_local_z_m: float
    maximum_closure_swept_finger_local_z_m: float
    minimum_predicted_terminal_finger_table_clearance_m: float
    predicted_closure_sweep_table_clearance_m: float
    closure_sweep_collision_free: bool
    proposed_clearance_tcp_position_m: tuple[float, float, float]
    proposed_clearance_arm_rad: tuple[float, ...]
    proposed_clearance_nominal_table_margin_m: float
    proposed_motion_implemented: bool
    screening_source: str
    dynamics_validated: bool


@dataclass(frozen=True)
class D38999PickRobot:
    arm_joint_names: tuple[str, ...]
    active_hand_joint_names: tuple[str, ...]
    home_arm_rad: tuple[float, ...]
    open_hand_rad: tuple[float, ...]
    arm_stiffness: float
    arm_damping: float
    hand_stiffness: float
    hand_damping: float


@dataclass(frozen=True)
class D38999ApproachSegment:
    name: str
    duration_s: float
    target_arm_rad: tuple[float, ...]


@dataclass(frozen=True)
class D38999PickMotion:
    interpolation: str
    hand_open_duration_s: float
    approach_segments: tuple[D38999ApproachSegment, ...]
    pregrasp_hold_duration_s: float
    pregrasp_tcp_position_m: tuple[float, float, float]
    grasp_arm_rad: tuple[float, ...]
    grasp_tcp_position_m: tuple[float, float, float]
    closure_clearance_arm_rad: tuple[float, ...]
    closure_clearance_tcp_position_m: tuple[float, float, float]
    grasp_tcp_down_axis_world: tuple[float, float, float]
    descent_duration_s: float
    open_tare_duration_s: float
    grasp_hand_rad: tuple[float, ...]
    closure_duration_s: float
    closed_seating_duration_s: float
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
class D38999PickSensing:
    torque_joint_names: tuple[str, ...]
    loaded_torque_threshold_nm: float
    minimum_loaded_channels: int
    operational_torque_target_nm: float
    maximum_absolute_torque_delta_nm: float
    fingertip_tactile_available: bool


@dataclass(frozen=True)
class D38999PickAcceptance:
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
    maximum_final_body_observable_linear_speed_m_s: float
    maximum_final_body_observable_angular_speed_rad_s: float
    maximum_final_body_post_solver_linear_speed_m_s: float
    maximum_final_body_post_solver_angular_speed_rad_s: float
    maximum_fixed_translation_drift_m: float
    maximum_fixed_rotation_drift_rad: float
    require_zero_preclosure_robot_loose_plug_contacts: bool
    require_zero_robot_table_contacts: bool
    require_zero_robot_fixture_contacts: bool
    require_zero_robot_fixed_endpoint_contacts: bool
    require_zero_final_plug_table_contacts: bool
    require_only_finger_loose_plug_contacts: bool
    require_physical_grip_contact: bool


@dataclass(frozen=True)
class D38999PickBoundaries:
    attachment_allowed: bool
    object_drive_allowed: bool
    object_pose_writes_after_start_allowed: bool
    collision_planned: bool
    self_collision_verified: bool


@dataclass(frozen=True)
class D38999TabletopPickConfig:
    schema_version: str
    scene: D38999PickScene
    geometry_candidate: D38999GeometryCandidate
    robot: D38999PickRobot
    motion: D38999PickMotion
    sensing: D38999PickSensing
    acceptance: D38999PickAcceptance
    boundaries: D38999PickBoundaries

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


def minimum_jerk_blend(fraction: Any) -> float:
    value = _finite(fraction, "fraction")
    if not 0.0 <= value <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def interpolate_arm(
    start: Any, target: Any, fraction: Any
) -> tuple[float, ...]:
    first = _vector(start, 7, "start")
    second = _vector(target, 7, "target")
    blend = minimum_jerk_blend(fraction)
    return tuple(a + blend * (b - a) for a, b in zip(first, second))


def _matrix_multiply(first, second):
    return tuple(
        tuple(
            sum(
                first[row][index] * second[index][column] for index in range(4)
            )
            for column in range(4)
        )
        for row in range(4)
    )


def _transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return (
        (*rotation[0], float(xyz[0])),
        (*rotation[1], float(xyz[1])),
        (*rotation[2], float(xyz[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def iiwa14_grasp_tcp_transform(arm_rad: Any):
    """Return pure URDF FK for iiwa14 link_0 to grasp_tcp."""

    joints = _vector(arm_rad, 7, "arm_rad")
    origins = (
        ((0.0, 0.0, 0.1575), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.2025), (math.pi / 2.0, 0.0, math.pi)),
        ((0.0, 0.2045, 0.0), (math.pi / 2.0, 0.0, math.pi)),
        ((0.0, 0.0, 0.2155), (math.pi / 2.0, 0.0, 0.0)),
        ((0.0, 0.1845, 0.0), (-math.pi / 2.0, math.pi, 0.0)),
        ((0.0, 0.0, 0.2155), (math.pi / 2.0, 0.0, 0.0)),
        ((0.0, 0.081, 0.0), (-math.pi / 2.0, math.pi, 0.0)),
    )
    result = _transform()
    for (xyz, rpy), angle in zip(origins, joints):
        result = _matrix_multiply(result, _transform(xyz, rpy))
        result = _matrix_multiply(result, _transform(rpy=(0.0, 0.0, angle)))
    # iiwa_link_7 -> iiwa_link_ee is 45 mm; grasp_tcp is another 400 mm.
    return _matrix_multiply(result, _transform((0.0, 0.0, 0.445)))


def _load_scene(value: Any, schema_version: str) -> D38999PickScene:
    document = _mapping(value, "scene")
    fields = tuple(D38999PickScene.__dataclass_fields__)
    _exact_keys(document, fields, "scene")
    asset = _text(document["robot_asset"], "scene.robot_asset")
    if Path(asset).is_absolute() or ".." in Path(asset).parts:
        raise ValueError("scene.robot_asset must be repository-relative")
    result = D38999PickScene(
        tabletop_config=_text(
            document["tabletop_config"], "scene.tabletop_config"
        ),
        proxy_config=_text(document["proxy_config"], "scene.proxy_config"),
        robot_asset=asset,
        robot_root_prim_path=_prim_path(
            document["robot_root_prim_path"], "scene.robot_root_prim_path"
        ),
        articulation_prim_path=_prim_path(
            document["articulation_prim_path"], "scene.articulation_prim_path"
        ),
        grasp_tcp_prim_path=_prim_path(
            document["grasp_tcp_prim_path"], "scene.grasp_tcp_prim_path"
        ),
    )
    profile = _PICK_PROFILE_ALLOWLIST[schema_version]
    if result.tabletop_config != profile["tabletop_config"]:
        if schema_version in (
            D38999_PICK_SCHEMA_VERSION,
            D38999_PICK_SCHEMA_VERSION_V2_FAST,
        ):
            raise ValueError("scene must use the D38999 tabletop v1")
        raise ValueError("scene tabletop config differs from keyed-v2 profile")
    if result.proxy_config != profile["source_config"]:
        if schema_version in (
            D38999_PICK_SCHEMA_VERSION,
            D38999_PICK_SCHEMA_VERSION_V2_FAST,
        ):
            raise ValueError("scene must use the D38999 shell-25/J proxy v1")
        raise ValueError("scene source config differs from keyed-v2 profile")
    expected_robot_asset = (
        "artifacts/kcg_connector/isaac/robot/handarm_keyed_v3_physical_r7/handarm.usda"
        if schema_version in _KEYED_V2_PICK_SCHEMAS
        else "artifacts/kcg_connector/isaac/robot/handarm/handarm.usda"
    )
    if result.robot_asset != expected_robot_asset:
        raise ValueError("scene robot asset is not canonical")
    expected_root = "/World/HandArm"
    if result.robot_root_prim_path != expected_root:
        raise ValueError("scene robot root is not canonical")
    expected_articulation = expected_root + "/Geometry/world"
    if result.articulation_prim_path != expected_articulation:
        raise ValueError("scene articulation path is not canonical")
    expected_tcp = (
        expected_articulation
        + "/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3"
        + "/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7"
        + "/iiwa_link_ee/handbase_link/grasp_tcp"
    )
    if result.grasp_tcp_prim_path != expected_tcp:
        raise ValueError("scene grasp TCP path is not canonical")
    return result


def _load_geometry(
    value: Any, schema_version: str
) -> D38999GeometryCandidate:
    document = _mapping(value, "geometry_candidate")
    fields = tuple(D38999GeometryCandidate.__dataclass_fields__)
    _exact_keys(document, fields, "geometry_candidate")
    result = D38999GeometryCandidate(
        loose_settled_origin_m=_vector(
            document["loose_settled_origin_m"],
            3,
            "geometry_candidate.loose_settled_origin_m",
        ),
        rear_body_radius_m=_positive(
            document["rear_body_radius_m"],
            "geometry_candidate.rear_body_radius_m",
        ),
        coupling_nut_outer_radius_m=_positive(
            document["coupling_nut_outer_radius_m"],
            "geometry_candidate.coupling_nut_outer_radius_m",
        ),
        rear_body_world_z_interval_m=_vector(
            document["rear_body_world_z_interval_m"],
            2,
            "geometry_candidate.rear_body_world_z_interval_m",
        ),
        coupling_nut_world_z_interval_m=_vector(
            document["coupling_nut_world_z_interval_m"],
            2,
            "geometry_candidate.coupling_nut_world_z_interval_m",
        ),
        grip_local_z_interval_m=_vector(
            document["grip_local_z_interval_m"],
            2,
            "geometry_candidate.grip_local_z_interval_m",
        ),
        handbase_to_tcp_m=_positive(
            document["handbase_to_tcp_m"],
            "geometry_candidate.handbase_to_tcp_m",
        ),
        maximum_closed_finger_local_z_m=_positive(
            document["maximum_closed_finger_local_z_m"],
            "geometry_candidate.maximum_closed_finger_local_z_m",
        ),
        maximum_closure_swept_finger_local_z_m=_positive(
            document["maximum_closure_swept_finger_local_z_m"],
            "geometry_candidate.maximum_closure_swept_finger_local_z_m",
        ),
        minimum_predicted_terminal_finger_table_clearance_m=_positive(
            document["minimum_predicted_terminal_finger_table_clearance_m"],
            "geometry_candidate."
            "minimum_predicted_terminal_finger_table_clearance_m",
        ),
        predicted_closure_sweep_table_clearance_m=_finite(
            document["predicted_closure_sweep_table_clearance_m"],
            "geometry_candidate." "predicted_closure_sweep_table_clearance_m",
        ),
        closure_sweep_collision_free=_strict_bool(
            document["closure_sweep_collision_free"],
            "geometry_candidate.closure_sweep_collision_free",
        ),
        proposed_clearance_tcp_position_m=_vector(
            document["proposed_clearance_tcp_position_m"],
            3,
            "geometry_candidate.proposed_clearance_tcp_position_m",
        ),
        proposed_clearance_arm_rad=_vector(
            document["proposed_clearance_arm_rad"],
            7,
            "geometry_candidate.proposed_clearance_arm_rad",
        ),
        proposed_clearance_nominal_table_margin_m=_positive(
            document["proposed_clearance_nominal_table_margin_m"],
            "geometry_candidate." "proposed_clearance_nominal_table_margin_m",
        ),
        proposed_motion_implemented=_strict_bool(
            document["proposed_motion_implemented"],
            "geometry_candidate.proposed_motion_implemented",
        ),
        screening_source=_text(
            document["screening_source"], "geometry_candidate.screening_source"
        ),
        dynamics_validated=_strict_bool(
            document["dynamics_validated"],
            "geometry_candidate.dynamics_validated",
        ),
    )
    keyed_v2 = schema_version in _KEYED_V2_PICK_SCHEMAS
    xycomp_candidate = (
        schema_version
        == D38999_PICK_SCHEMA_VERSION_KEYED_V2_XYCOMP_CANDIDATE
    )
    keyed_grasp_position = (
        EXPECTED_D38999_KEYED_V2_XYCOMP_GRASP_TCP_POSITION_M
        if xycomp_candidate
        else EXPECTED_D38999_KEYED_V2_GRASP_TCP_POSITION_M
    )
    keyed_grasp_arm = (
        EXPECTED_D38999_KEYED_V2_XYCOMP_GRASP_ARM_RAD
        if xycomp_candidate
        else EXPECTED_D38999_KEYED_V2_GRASP_ARM_RAD
    )
    expected_profile_values = (
        {
            "loose_settled_origin_m": (0.520, -0.210, 0.2305),
            "rear_body_radius_m": 0.02220,
            "coupling_nut_outer_radius_m": 0.024,
            "rear_body_world_z_interval_m": (0.200, 0.21545),
            "coupling_nut_world_z_interval_m": (0.200, 0.2325),
            "grip_local_z_interval_m": (0.42903, 0.44448),
            "screening_source": (
                "physical_r7_frozen_blueprint_geometry_only_A2_A3_not_run"
            ),
        }
        if keyed_v2
        else {
            "loose_settled_origin_m": (0.520, -0.210, 0.200),
            "rear_body_radius_m": 0.02215,
            "coupling_nut_outer_radius_m": 0.024,
            "rear_body_world_z_interval_m": (0.217, 0.231),
            "coupling_nut_world_z_interval_m": (0.207, 0.224),
            "grip_local_z_interval_m": (0.41748, 0.44148),
            "screening_source": (
                "iiwa14_urdf_fk_and_checked_in_finger_collision_stl"
            ),
        }
    )
    expected = (
        result.loose_settled_origin_m
        == expected_profile_values["loose_settled_origin_m"]
        and result.rear_body_radius_m
        == expected_profile_values["rear_body_radius_m"]
        and result.coupling_nut_outer_radius_m
        == expected_profile_values["coupling_nut_outer_radius_m"]
        and result.rear_body_world_z_interval_m
        == expected_profile_values["rear_body_world_z_interval_m"]
        and result.coupling_nut_world_z_interval_m
        == expected_profile_values["coupling_nut_world_z_interval_m"]
        and result.grip_local_z_interval_m
        == expected_profile_values["grip_local_z_interval_m"]
        and result.handbase_to_tcp_m == 0.400
        and result.maximum_closed_finger_local_z_m == 0.41760
        and result.maximum_closure_swept_finger_local_z_m == 0.436673
        and result.minimum_predicted_terminal_finger_table_clearance_m == 0.010
        and result.predicted_closure_sweep_table_clearance_m
        == (0.007807 if keyed_v2 else 0.011807)
        and result.proposed_clearance_tcp_position_m
        == (
            keyed_grasp_position
            if keyed_v2
            else (0.520, -0.210, 0.24848)
        )
        and result.proposed_clearance_arm_rad
        == (
            keyed_grasp_arm
            if keyed_v2
            else EXPECTED_D38999_GRASP_ARM_RAD
        )
        and result.proposed_clearance_nominal_table_margin_m
        == (0.007807 if keyed_v2 else 0.011807)
        and result.screening_source
        == expected_profile_values["screening_source"]
    )
    if not expected:
        label = "keyed-v2" if keyed_v2 else "v1"
        raise ValueError(f"geometry candidate is not canonical {label}")
    if result.dynamics_validated:
        raise ValueError(
            "geometry candidate must not claim dynamics validation"
        )
    if not result.closure_sweep_collision_free:
        raise ValueError(
            "implemented geometry candidate must clear the table"
        )
    if not result.proposed_motion_implemented:
        raise ValueError("proposed clearance motion must remain implemented")
    for interval in (
        result.rear_body_world_z_interval_m,
        result.coupling_nut_world_z_interval_m,
        result.grip_local_z_interval_m,
    ):
        if interval[0] >= interval[1]:
            raise ValueError("geometry intervals must be increasing")
    return result


def _load_robot(value: Any) -> D38999PickRobot:
    document = _mapping(value, "robot")
    fields = tuple(D38999PickRobot.__dataclass_fields__)
    _exact_keys(document, fields, "robot")
    result = D38999PickRobot(
        arm_joint_names=_names(
            document["arm_joint_names"], 7, "robot.arm_joint_names"
        ),
        active_hand_joint_names=_names(
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
        arm_stiffness=_positive(
            document["arm_stiffness"], "robot.arm_stiffness"
        ),
        arm_damping=_positive(document["arm_damping"], "robot.arm_damping"),
        hand_stiffness=_positive(
            document["hand_stiffness"], "robot.hand_stiffness"
        ),
        hand_damping=_positive(document["hand_damping"], "robot.hand_damping"),
    )
    if result.arm_joint_names != EXPECTED_ARM_JOINT_NAMES:
        raise ValueError("robot arm joint names are not canonical")
    if result.active_hand_joint_names != EXPECTED_ACTIVE_HAND_JOINT_NAMES:
        raise ValueError("robot active hand joint names are not canonical")
    if result.home_arm_rad != (0.0,) * 7 or result.open_hand_rad != (
        1.0,
        0.0,
        0.0,
        0.0,
    ):
        raise ValueError("robot Home/open-hand targets are not canonical")
    if (
        result.arm_stiffness,
        result.arm_damping,
        result.hand_stiffness,
        result.hand_damping,
    ) != (24000.0, 400.0, 25.0, 2.0):
        raise ValueError("robot drive gains are not canonical v1")
    return result


def _load_motion(
    value: Any, robot: D38999PickRobot, schema_version: str
) -> D38999PickMotion:
    document = _mapping(value, "motion")
    fields = tuple(D38999PickMotion.__dataclass_fields__)
    _exact_keys(document, fields, "motion")
    raw_segments = document["approach_segments"]
    if not isinstance(raw_segments, list) or len(raw_segments) != 3:
        raise ValueError(
            "motion.approach_segments must contain three segments"
        )
    segments = []
    for index, raw in enumerate(raw_segments):
        item = _mapping(raw, f"motion.approach_segments[{index}]")
        _exact_keys(
            item,
            ("name", "duration_s", "target_arm_rad"),
            f"motion.approach_segments[{index}]",
        )
        segments.append(
            D38999ApproachSegment(
                name=_text(
                    item["name"], f"motion.approach_segments[{index}].name"
                ),
                duration_s=_positive(
                    item["duration_s"],
                    f"motion.approach_segments[{index}].duration_s",
                ),
                target_arm_rad=_vector(
                    item["target_arm_rad"],
                    7,
                    f"motion.approach_segments[{index}].target_arm_rad",
                ),
            )
        )
    duration_names = (
        "hand_open_duration_s",
        "pregrasp_hold_duration_s",
        "descent_duration_s",
        "open_tare_duration_s",
        "closure_duration_s",
        "closed_seating_duration_s",
        "preload_duration_s",
        "lift_duration_s",
        "final_hold_duration_s",
        "effort_sample_duration_s",
    )
    result = D38999PickMotion(
        interpolation=_text(document["interpolation"], "motion.interpolation"),
        approach_segments=tuple(segments),
        pregrasp_tcp_position_m=_vector(
            document["pregrasp_tcp_position_m"],
            3,
            "motion.pregrasp_tcp_position_m",
        ),
        grasp_arm_rad=_vector(
            document["grasp_arm_rad"], 7, "motion.grasp_arm_rad"
        ),
        grasp_tcp_position_m=_vector(
            document["grasp_tcp_position_m"], 3, "motion.grasp_tcp_position_m"
        ),
        closure_clearance_arm_rad=_vector(
            document["closure_clearance_arm_rad"],
            7,
            "motion.closure_clearance_arm_rad",
        ),
        closure_clearance_tcp_position_m=_vector(
            document["closure_clearance_tcp_position_m"],
            3,
            "motion.closure_clearance_tcp_position_m",
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
            name: _positive(document[name], f"motion.{name}")
            for name in duration_names
        },
        grip_hand_stiffness=_positive(
            document["grip_hand_stiffness"], "motion.grip_hand_stiffness"
        ),
        grip_hand_damping=_positive(
            document["grip_hand_damping"], "motion.grip_hand_damping"
        ),
        grip_static_friction=_positive(
            document["grip_static_friction"], "motion.grip_static_friction"
        ),
        grip_dynamic_friction=_positive(
            document["grip_dynamic_friction"], "motion.grip_dynamic_friction"
        ),
        grip_restitution=_finite(
            document["grip_restitution"], "motion.grip_restitution"
        ),
    )
    if result.interpolation != "minimum_jerk":
        raise ValueError("motion interpolation must be minimum_jerk")
    if (
        tuple(segment.name for segment in result.approach_segments)
        != EXPECTED_APPROACH_NAMES
    ):
        raise ValueError("motion approach names are not canonical v1")
    if (
        tuple(segment.target_arm_rad for segment in result.approach_segments)
        != EXPECTED_APPROACH_TARGETS
    ):
        raise ValueError("motion approach targets are not canonical v1")
    if tuple(segment.duration_s for segment in result.approach_segments) != (
        6.2,
        3.7,
        2.5,
    ):
        raise ValueError("motion approach durations are not canonical v1")
    if result.pregrasp_tcp_position_m != (0.520, -0.210, 0.360):
        raise ValueError("motion pregrasp TCP is not canonical v1")
    keyed_v2 = schema_version in _KEYED_V2_PICK_SCHEMAS
    xycomp_candidate = (
        schema_version
        == D38999_PICK_SCHEMA_VERSION_KEYED_V2_XYCOMP_CANDIDATE
    )
    expected_grasp_arm = (
        (
            EXPECTED_D38999_KEYED_V2_XYCOMP_GRASP_ARM_RAD
            if xycomp_candidate
            else EXPECTED_D38999_KEYED_V2_GRASP_ARM_RAD
        )
        if keyed_v2
        else EXPECTED_D38999_GRASP_ARM_RAD
    )
    expected_grasp_position = (
        (
            EXPECTED_D38999_KEYED_V2_XYCOMP_GRASP_TCP_POSITION_M
            if xycomp_candidate
            else EXPECTED_D38999_KEYED_V2_GRASP_TCP_POSITION_M
        )
        if keyed_v2
        else (0.520, -0.210, 0.24848)
    )
    if result.grasp_arm_rad != expected_grasp_arm:
        raise ValueError("motion D38999 grasp IK is not canonical v1")
    if result.grasp_tcp_position_m != expected_grasp_position:
        raise ValueError("motion D38999 grasp TCP is not canonical v1")
    if (
        result.closure_clearance_arm_rad
        != (
            EXPECTED_D38999_KEYED_V2_GRASP_ARM_RAD
            if keyed_v2 and not xycomp_candidate
            else EXPECTED_D38999_KEYED_V2_XYCOMP_GRASP_ARM_RAD
            if xycomp_candidate
            else EXPECTED_D38999_CLOSURE_CLEARANCE_ARM_RAD
        )
    ):
        raise ValueError("motion closure-clearance IK is not canonical v1")
    if result.closure_clearance_tcp_position_m != expected_grasp_position:
        raise ValueError("motion closure-clearance TCP is not canonical v1")
    if result.grasp_tcp_down_axis_world != (0.0, 0.0, -1.0):
        raise ValueError("motion grasp TCP axis must point down")
    if result.grasp_hand_rad != (1.0, 0.765, 0.595, 0.765):
        raise ValueError("motion D38999 hand target is not canonical v1")
    expected_durations = (
        (16.0, 1.0, 3.0, 0.5, 3.5, 1.5, 0.5, 3.0, 2.0, 0.5)
        if schema_version == D38999_PICK_SCHEMA_VERSION_V2_FAST
        else (16.0, 1.0, 3.0, 0.5, 3.5, 1.5, 0.5, 3.0, 4.0, 0.5)
    )
    if (
        tuple(getattr(result, name) for name in duration_names)
        != expected_durations
    ):
        raise ValueError("motion phase durations are not canonical v1")
    if (
        result.grip_hand_stiffness,
        result.grip_hand_damping,
        result.grip_static_friction,
        result.grip_dynamic_friction,
        result.grip_restitution,
    ) != (5.0, 1.0, 1.4, 1.4, 0.0):
        raise ValueError("motion grip physics is not canonical v1")
    all_durations = [
        segment.duration_s for segment in result.approach_segments
    ]
    all_durations.extend(getattr(result, name) for name in duration_names)
    if any(
        not math.isclose(value * 240.0, round(value * 240.0), abs_tol=1.0e-9)
        for value in all_durations
    ):
        raise ValueError("motion durations must contain whole 240 Hz steps")
    if result.effort_sample_duration_s > min(
        result.open_tare_duration_s,
        result.preload_duration_s,
        result.final_hold_duration_s,
    ):
        raise ValueError(
            "motion effort sample window exceeds a sampling phase"
        )
    starts = (robot.home_arm_rad,) + tuple(
        segment.target_arm_rad for segment in result.approach_segments[:-1]
    )
    for start, segment in zip(starts, result.approach_segments):
        peak = max(
            1.875 * abs(end - begin) / segment.duration_s
            for begin, end in zip(start, segment.target_arm_rad)
        )
        if peak > 0.3 + 1.0e-9:
            raise ValueError("motion approach exceeds 0.3 rad/s peak speed")
    arm_segments = (
        (
            result.approach_segments[-1].target_arm_rad,
            result.closure_clearance_arm_rad,
            result.descent_duration_s,
        ),
        (
            result.closure_clearance_arm_rad,
            result.grasp_arm_rad,
            result.closed_seating_duration_s,
        ),
        (
            result.grasp_arm_rad,
            result.approach_segments[-1].target_arm_rad,
            result.lift_duration_s,
        ),
    )
    for start, target, duration in arm_segments:
        peak = max(
            1.875 * abs(end - begin) / duration
            for begin, end in zip(start, target)
        )
        if peak > 0.3 + 1.0e-9:
            raise ValueError(
                "motion grasp arm segment exceeds 0.3 rad/s peak speed"
            )
    closure_peak = max(
        1.875 * abs(end - begin) / result.closure_duration_s
        for begin, end in zip(robot.open_hand_rad, result.grasp_hand_rad)
    )
    if closure_peak > 0.5 + 1.0e-9:
        raise ValueError("motion hand closure exceeds 0.5 rad/s peak speed")
    transform = iiwa14_grasp_tcp_transform(result.grasp_arm_rad)
    position = tuple(transform[index][3] for index in range(3))
    z_axis = tuple(transform[index][2] for index in range(3))
    if math.dist(position, result.grasp_tcp_position_m) > 1.0e-6:
        raise ValueError("motion grasp IK fails pure iiwa14 FK position")
    if math.dist(z_axis, result.grasp_tcp_down_axis_world) > 1.0e-6:
        raise ValueError("motion grasp IK fails pure iiwa14 FK orientation")
    transform = iiwa14_grasp_tcp_transform(result.closure_clearance_arm_rad)
    position = tuple(transform[index][3] for index in range(3))
    z_axis = tuple(transform[index][2] for index in range(3))
    if math.dist(position, result.closure_clearance_tcp_position_m) > 1.0e-6:
        raise ValueError("motion closure-clearance IK fails pure FK position")
    if math.dist(z_axis, result.grasp_tcp_down_axis_world) > 1.0e-6:
        raise ValueError(
            "motion closure-clearance IK fails pure FK orientation"
        )
    return result


def _load_sensing(value: Any) -> D38999PickSensing:
    document = _mapping(value, "sensing")
    fields = tuple(D38999PickSensing.__dataclass_fields__)
    _exact_keys(document, fields, "sensing")
    result = D38999PickSensing(
        torque_joint_names=_names(
            document["torque_joint_names"], 3, "sensing.torque_joint_names"
        ),
        loaded_torque_threshold_nm=_positive(
            document["loaded_torque_threshold_nm"],
            "sensing.loaded_torque_threshold_nm",
        ),
        minimum_loaded_channels=_positive_integer(
            document["minimum_loaded_channels"],
            "sensing.minimum_loaded_channels",
        ),
        operational_torque_target_nm=_positive(
            document["operational_torque_target_nm"],
            "sensing.operational_torque_target_nm",
        ),
        maximum_absolute_torque_delta_nm=_positive(
            document["maximum_absolute_torque_delta_nm"],
            "sensing.maximum_absolute_torque_delta_nm",
        ),
        fingertip_tactile_available=_strict_bool(
            document["fingertip_tactile_available"],
            "sensing.fingertip_tactile_available",
        ),
    )
    if result.torque_joint_names != EXPECTED_TORQUE_JOINT_NAMES:
        raise ValueError("sensing must use the three real base torque axes")
    if (
        result.loaded_torque_threshold_nm != 0.020
        or result.minimum_loaded_channels != 2
        or result.operational_torque_target_nm != 1.8
        or result.maximum_absolute_torque_delta_nm != 2.0
        or result.operational_torque_target_nm
        >= result.maximum_absolute_torque_delta_nm
    ):
        raise ValueError("sensing thresholds are not canonical v1")
    if result.fingertip_tactile_available:
        raise ValueError("fingertip tactile must remain unavailable")
    return result


def _load_acceptance(value: Any) -> D38999PickAcceptance:
    document = _mapping(value, "acceptance")
    fields = tuple(D38999PickAcceptance.__dataclass_fields__)
    _exact_keys(document, fields, "acceptance")
    boolean_fields = tuple(
        name for name in fields if name.startswith("require_")
    )
    numeric_fields = tuple(
        name for name in fields if name not in boolean_fields
    )
    result = D38999PickAcceptance(
        **{
            name: _positive(document[name], f"acceptance.{name}")
            for name in numeric_fields
        },
        **{
            name: _strict_bool(document[name], f"acceptance.{name}")
            for name in boolean_fields
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
        "maximum_final_body_observable_linear_speed_m_s": 0.020,
        "maximum_final_body_observable_angular_speed_rad_s": 0.200,
        "maximum_final_body_post_solver_linear_speed_m_s": 0.020,
        "maximum_final_body_post_solver_angular_speed_rad_s": 0.200,
        "maximum_fixed_translation_drift_m": 0.000001,
        "maximum_fixed_rotation_drift_rad": 0.00001,
    }
    for name, upper in upper_bounds.items():
        if getattr(result, name) > upper:
            raise ValueError(f"acceptance {name} exceeds safety bound")
    if result.minimum_body_lift_m < 0.080:
        raise ValueError("acceptance body lift must be at least 80 mm")
    if result.minimum_final_bottom_clearance_m < 0.070:
        raise ValueError("acceptance bottom clearance must be at least 70 mm")
    if not all(getattr(result, name) for name in boolean_fields):
        raise ValueError(
            "all D38999 physical contact requirements must be enabled"
        )
    return result


def _load_boundaries(value: Any) -> D38999PickBoundaries:
    document = _mapping(value, "boundaries")
    fields = tuple(D38999PickBoundaries.__dataclass_fields__)
    _exact_keys(document, fields, "boundaries")
    result = D38999PickBoundaries(
        **{
            name: _strict_bool(document[name], f"boundaries.{name}")
            for name in fields
        }
    )
    if any(getattr(result, name) for name in fields):
        raise ValueError("D38999 pick boundaries must remain fail-closed")
    return result


def _validate_cross_contract(config: D38999TabletopPickConfig) -> None:
    geometry = config.geometry_candidate
    motion = config.motion
    handbase_world_z = (
        motion.grasp_tcp_position_m[2] + geometry.handbase_to_tcp_m
    )
    expected_local = (
        handbase_world_z - geometry.rear_body_world_z_interval_m[1],
        handbase_world_z - geometry.coupling_nut_world_z_interval_m[0],
    )
    if math.dist(expected_local, geometry.grip_local_z_interval_m) > 1.0e-9:
        raise ValueError("geometry grip interval does not match grasp TCP")
    predicted_terminal_clearance = (
        handbase_world_z - geometry.maximum_closed_finger_local_z_m - 0.200
    )
    if (
        predicted_terminal_clearance + 1.0e-12
        < geometry.minimum_predicted_terminal_finger_table_clearance_m
    ):
        raise ValueError("geometry terminal candidate touches the table")
    predicted_sweep_clearance = (
        handbase_world_z
        - geometry.maximum_closure_swept_finger_local_z_m
        - 0.200
    )
    if not math.isclose(
        predicted_sweep_clearance,
        geometry.predicted_closure_sweep_table_clearance_m,
        abs_tol=1.0e-9,
    ):
        raise ValueError("geometry closure sweep clearance is inconsistent")
    if predicted_sweep_clearance <= 0.0:
        raise ValueError("implemented v1 closure sweep must clear the table")
    if not geometry.closure_sweep_collision_free:
        raise ValueError("implemented v1 closure sweep flag must be true")
    proposed_transform = iiwa14_grasp_tcp_transform(
        geometry.proposed_clearance_arm_rad
    )
    proposed_position = tuple(
        proposed_transform[index][3] for index in range(3)
    )
    proposed_axis = tuple(proposed_transform[index][2] for index in range(3))
    if (
        math.dist(
            proposed_position, geometry.proposed_clearance_tcp_position_m
        )
        > 1.0e-6
        or math.dist(proposed_axis, (0.0, 0.0, -1.0)) > 1.0e-6
    ):
        raise ValueError("proposed clearance IK fails pure iiwa14 FK")
    proposed_handbase_z = (
        geometry.proposed_clearance_tcp_position_m[2]
        + geometry.handbase_to_tcp_m
    )
    proposed_margin = (
        proposed_handbase_z
        - geometry.maximum_closure_swept_finger_local_z_m
        - 0.200
    )
    if not math.isclose(
        proposed_margin,
        geometry.proposed_clearance_nominal_table_margin_m,
        abs_tol=1.0e-9,
    ):
        raise ValueError("proposed closure margin is inconsistent")
    if (
        motion.closure_clearance_arm_rad != geometry.proposed_clearance_arm_rad
        or motion.closure_clearance_tcp_position_m
        != geometry.proposed_clearance_tcp_position_m
    ):
        raise ValueError(
            "implemented closure-clearance motion differs from geometry"
        )
    if not geometry.proposed_motion_implemented:
        raise ValueError("closure-clearance motion must be implemented")
    if geometry.coupling_nut_outer_radius_m * 2.0 != 0.048:
        raise ValueError("geometry candidate must retain the 48 mm nut")


def load_d38999_tabletop_pick_config(
    config_path: Path | str = DEFAULT_D38999_PICK_CONFIG_PATH,
) -> D38999TabletopPickConfig:
    """Load the independent D38999 pick candidate without Isaac imports."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "D38999 pick config")
    fields = (
        "schema_version",
        "scene",
        "geometry_candidate",
        "robot",
        "motion",
        "sensing",
        "acceptance",
        "boundaries",
    )
    _exact_keys(document, fields, "D38999 pick config")
    schema_version = document["schema_version"]
    if schema_version not in _PICK_PROFILE_ALLOWLIST:
        raise ValueError("unsupported D38999 tabletop pick schema")
    robot = _load_robot(document["robot"])
    result = D38999TabletopPickConfig(
        schema_version=schema_version,
        scene=_load_scene(document["scene"], schema_version),
        geometry_candidate=_load_geometry(
            document["geometry_candidate"], schema_version
        ),
        robot=robot,
        motion=_load_motion(document["motion"], robot, schema_version),
        sensing=_load_sensing(document["sensing"]),
        acceptance=_load_acceptance(document["acceptance"]),
        boundaries=_load_boundaries(document["boundaries"]),
    )
    _validate_cross_contract(result)
    result.as_dict()
    return result


def verify_d38999_pick_dependencies(
    config: D38999TabletopPickConfig,
    config_path: Path | str = DEFAULT_D38999_PICK_CONFIG_PATH,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Cross-check the allowlisted local dependencies before Isaac starts."""

    from .d38999_tabletop_scene import (
        load_d38999_tabletop_scene,
        verify_d38999_tabletop_asset,
    )

    path = Path(config_path).expanduser().resolve()
    repository = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else path.parents[3]
    )
    tabletop_path = path.parent / config.scene.tabletop_config
    source_path = path.parent / config.scene.proxy_config
    tabletop = load_d38999_tabletop_scene(tabletop_path)
    expected_profile = _PICK_PROFILE_ALLOWLIST[config.schema_version]
    if tabletop.asset_profile.profile_id != expected_profile["asset_profile_id"]:
        raise ValueError("pick and tabletop asset profiles differ")
    d38999_asset = verify_d38999_tabletop_asset(tabletop, repository)
    robot_asset = (repository / config.scene.robot_asset).resolve()
    try:
        robot_asset.relative_to(repository)
    except ValueError as error:
        raise ValueError("robot asset escapes repository") from error
    if not robot_asset.is_file():
        raise ValueError("robot asset is missing")
    candidate = config.geometry_candidate
    source_key = "proxy"
    if config.schema_version == D38999_PICK_SCHEMA_VERSION_MULTILAYER_GRASP:
        source_model = _mapping(
            yaml.safe_load(source_path.read_text(encoding="utf-8")),
            "multilayer master model contract",
        )
        if (
            source_model.get("schema_version")
            != "kcg_d38999_master_model_contract_v1"
            or source_model.get("status") != "FROZEN_FOR_MULTILAYER_V1"
            or tabletop.asset.proxy_id != _MULTILAYER_GRASP_PROFILE_ID
        ):
            raise ValueError(
                "multilayer master identity differs from tabletop"
            )
        expected_rear_radius = 0.02220
        expected_nut_radius = 0.024
        if not math.isclose(
            candidate.rear_body_radius_m,
            expected_rear_radius,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("pick rear body differs from multilayer authority")
        if not math.isclose(
            candidate.coupling_nut_outer_radius_m,
            expected_nut_radius,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("pick coupling nut differs from human authority")
        source_key = "source_model"
    elif config.schema_version in _KEYED_V2_PICK_SCHEMAS:
        from .d38999_keyed_v2_physical_model_contract import (
            load_physical_model_contract,
        )

        source_model = load_physical_model_contract(source_path)
        physical_pair_model_id = source_model.document["identity"][
            "pair_model_id"
        ]
        if (
            physical_pair_model_id != _KEYED_V2_PROFILE_ID
            or tabletop.asset.proxy_id != physical_pair_model_id
        ):
            raise ValueError("keyed-v2 source identity differs from tabletop")
        shells_and_keying = source_model.document[
            "a2_collision_authoring_blueprint"
        ]["connector_shells_and_keying"]
        rear_bands = shells_and_keying["body_assembly_rear_body"][
            "local_z_profile_bands"
        ]
        expected_rear_radius = max(
            float(band["outer_radius_m"]) for band in rear_bands
        )
        expected_nut_radius = float(
            shells_and_keying["coupling_nut"][
                "public_outer_radius_max_selected_as_proxy_m"
            ]
        )
        if not math.isclose(
            candidate.rear_body_radius_m,
            expected_rear_radius,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("pick rear body differs from frozen physical r7")
        if not math.isclose(
            candidate.coupling_nut_outer_radius_m,
            expected_nut_radius,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "pick coupling nut differs from frozen physical r7"
            )
        source_key = "source_model"
    else:
        from .d38999_proxy import load_d38999_shell25j_proxy

        source_model = load_d38999_shell25j_proxy(source_path)
        if source_model.identity.proxy_id != tabletop.asset.proxy_id:
            raise ValueError("D38999 proxy identity differs from tabletop asset")
        if (
            candidate.rear_body_radius_m
            != source_model.plug_geometry_m.rear_body_radius
            or candidate.coupling_nut_outer_radius_m
            != source_model.plug_geometry_m.coupling_nut_outer_radius
        ):
            raise ValueError("pick geometry differs from D38999 proxy contract")
    if candidate.loose_settled_origin_m[:2] != (
        tabletop.loose_endpoint.initial_origin_m[:2]
    ):
        raise ValueError("pick XY center differs from D38999 tabletop")
    if not math.isclose(
        candidate.loose_settled_origin_m[2],
        tabletop.loose_settled_origin_m[2],
        abs_tol=1.0e-12,
    ):
        raise ValueError("pick settled origin differs from tabletop profile")
    result = {
        "d38999_asset": d38999_asset,
        "robot_asset": robot_asset,
        "tabletop": tabletop,
        source_key: source_model,
    }
    return result


__all__ = [
    "DEFAULT_D38999_PICK_CONFIG_PATH",
    "D38999_PICK_SCHEMA_VERSION",
    "D38999_PICK_SCHEMA_VERSION_KEYED_V2",
    "D38999_PICK_SCHEMA_VERSION_KEYED_V2_XYCOMP_CANDIDATE",
    "D38999_PICK_SCHEMA_VERSION_MULTILAYER_GRASP",
    "D38999TabletopPickConfig",
    "EXPECTED_D38999_CLOSURE_CLEARANCE_ARM_RAD",
    "EXPECTED_D38999_GRASP_ARM_RAD",
    "EXPECTED_TORQUE_JOINT_NAMES",
    "iiwa14_grasp_tcp_transform",
    "interpolate_arm",
    "load_d38999_tabletop_pick_config",
    "minimum_jerk_blend",
    "verify_d38999_pick_dependencies",
]
