"""Deterministic arm IK and complete 11-joint route for one accepted policy.

The input is the fail-closed six-stage :class:`CandidateRouteStateContract`.
This module uses the same hash-bound aggregate URDF model for both study
objects, solves the seven arm joints, retains contact-triggered hand intervals,
and emits bounded HOME-to-pregrasp and vertical-lift waypoint sequences.

The result is kinematic input for a later continuous collision checker.  It is
not collision clearance, controller authorization, Isaac evidence, hardware
evidence, or a formal grasp selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from kcg_connector.grasp.robust.aggregate_collision_inputs import (
    EXPECTED_INDEPENDENT_JOINTS,
    AggregateCollisionRuntimeInputCertificate,
)
from kcg_connector.grasp.robust.bounded_hand_base_ik import (
    EXPECTED_FEASIBLE_CHOICE_RULE,
    EXPECTED_SEED_RULE,
    EXPECTED_SOLVER_METHOD,
    CandidateJointRouteError,
    bounded_ik_settings,
    pregrasp_seeds,
    solve_bounded_target,
    solve_bounded_hand_base_ik,
)
from kcg_connector.grasp.robust.candidate_route import (
    EXPECTED_ARM_JOINT_NAMES,
    EXPECTED_HAND_BASE_LINK,
    EXPECTED_HAND_JOINT_NAMES,
    EXPECTED_ROUTE_CONTRACT_ID,
    LIFT_TRANSLATION_WORLD_M,
    METHOD_ID as ROUTE_STATE_METHOD_ID,
    CandidateRouteStateContract,
)
from kcg_connector.grasp.robust.interval_kinematics import IntervalBounds


METHOD_ID = "CARTS_HASH_BOUND_BOUNDED_ARM_IK_AND_11_JOINT_ROUTE_V1"
EXPECTED_SCHEMA_VERSION = "carts_candidate_joint_route_v1"
EXPECTED_JOINT_ROUTE_CONTRACT_ID = (
    "CARTS_PER_ACCEPTED_POLICY_KINEMATIC_11_JOINT_ROUTE_V1"
)
EXPECTED_CLAIM_SCOPE = "STATIC_KINEMATIC_JOINT_ROUTE_BINDING_ONLY"
EXPECTED_APPROACH_RULE = (
    "MINIMUM_JERK_JOINT_INTERPOLATION_HOME_TO_PREGRASP"
)
EXPECTED_CLOSURE_RULE = (
    "FIXED_ARM_WITH_CUMULATIVE_CONTACT_TRIGGERED_HAND_INTERVALS"
)
EXPECTED_LIFT_RULE = (
    "WORLD_Z_CARTESIAN_WAYPOINT_IK_WARM_STARTED_FROM_PREVIOUS_WAYPOINT"
)
EXPECTED_CLOSURE_STAGE_NAMES = (
    "CLOSURE_FINGER_1",
    "CLOSURE_FINGER_2",
    "CLOSURE_FINGER_3",
)
CLAIM_LIMITATIONS = (
    "STATIC_KINEMATIC_11_JOINT_ROUTE_ONLY",
    "PER_STATIC_ACCEPTED_POLICY_NOT_POST_SELECTION_SHORTCUT",
    "CONTACT_TRIGGERED_HAND_ENDPOINTS_REMAIN_CERTIFIED_INTERVALS",
    "NO_CONTINUOUS_COLLISION_OR_ALLOWED_PAD_CONTACT_CERTIFICATE_YET",
    "NO_CONTROLLER_ISAAC_HARDWARE_OR_FORMAL_SELECTION_AUTHORIZATION",
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise CandidateJointRouteError(
                "DUPLICATE_YAML_KEY", f"duplicate key {key!r}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CandidateJointRouteError(
            "MAPPING_REQUIRED", f"{label} must be a string-keyed mapping"
        )
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    missing = sorted(set(expected).difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise CandidateJointRouteError(
            "SCHEMA_MISMATCH", f"{label} missing={missing}, extra={extra}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _float_hex(value: float) -> str:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CandidateJointRouteError("NONFINITE_VALUE", repr(value))
    return parsed.hex()


def _proper_transform(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    tolerance = 128.0 * np.finfo(np.float64).eps
    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or float(np.linalg.norm(matrix[3] - (0.0, 0.0, 0.0, 1.0)))
        > tolerance
        or float(np.linalg.norm(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3)))
        > tolerance
        or abs(float(np.linalg.det(matrix[:3, :3])) - 1.0) > tolerance
    ):
        raise CandidateJointRouteError(
            "PROPER_TRANSFORM_REQUIRED", f"{label} must be one rigid transform"
        )
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _interval_document(interval: IntervalBounds) -> list[str]:
    return [_float_hex(interval.lower), _float_hex(interval.upper)]


def _interval_rows_document(
    rows: Sequence[Sequence[IntervalBounds]],
) -> list[list[list[str]]]:
    return [
        [_interval_document(interval) for interval in row]
        for row in rows
    ]


@dataclass(frozen=True)
class _JointRouteSettings:
    config_sha256: str
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
    approach_maximum_arm_joint_step_rad: float
    lift_cartesian_step_m: float
    lift_waypoint_count: int
    lift_maximum_arm_joint_step_rad: float


def _validated_config(path: Path) -> _JointRouteSettings:
    try:
        config = _mapping(
            yaml.load(
                path.read_text(encoding="utf-8"),
                Loader=_UniqueKeySafeLoader,
            ),
            "candidate joint-route config",
        )
    except yaml.YAMLError as error:
        raise CandidateJointRouteError("INVALID_YAML", str(error)) from error
    _exact_keys(
        config,
        (
            "schema_version",
            "joint_route_contract_id",
            "claim_scope",
            "input_binding",
            "solver",
            "route_sampling",
            "truth_firewall",
        ),
        "candidate joint-route config",
    )
    if (
        config["schema_version"] != EXPECTED_SCHEMA_VERSION
        or config["joint_route_contract_id"]
        != EXPECTED_JOINT_ROUTE_CONTRACT_ID
        or config["claim_scope"] != EXPECTED_CLAIM_SCOPE
    ):
        raise CandidateJointRouteError(
            "CONTRACT_IDENTITY_MISMATCH",
            "candidate joint-route identity changed",
        )

    inputs = _mapping(config["input_binding"], "input_binding")
    _exact_keys(
        inputs,
        (
            "candidate_route_method_id",
            "candidate_route_contract_id",
            "aggregate_robot_base_link",
            "target_link",
            "arm_joint_names",
            "hand_joint_names",
        ),
        "input_binding",
    )
    if (
        inputs["candidate_route_method_id"] != ROUTE_STATE_METHOD_ID
        or inputs["candidate_route_contract_id"] != EXPECTED_ROUTE_CONTRACT_ID
        or inputs["aggregate_robot_base_link"] != "world"
        or inputs["target_link"] != EXPECTED_HAND_BASE_LINK
        or tuple(inputs["arm_joint_names"]) != EXPECTED_ARM_JOINT_NAMES
        or tuple(inputs["hand_joint_names"]) != EXPECTED_HAND_JOINT_NAMES
    ):
        raise CandidateJointRouteError(
            "INPUT_BINDING_CHANGED",
            "route-state or aggregate robot identity changed",
        )

    bounded = bounded_ik_settings(config["solver"])

    route = _mapping(config["route_sampling"], "route_sampling")
    _exact_keys(
        route,
        (
            "approach_rule",
            "approach_maximum_arm_joint_step_rad",
            "closure_rule",
            "lift_rule",
            "lift_translation_world_m",
            "lift_cartesian_step_m",
            "lift_waypoint_count",
            "lift_maximum_arm_joint_step_rad",
        ),
        "route_sampling",
    )
    approach_step = float(route["approach_maximum_arm_joint_step_rad"])
    lift_step = float(route["lift_cartesian_step_m"])
    lift_count = int(route["lift_waypoint_count"])
    lift_joint_step = float(route["lift_maximum_arm_joint_step_rad"])
    lift_translation = tuple(
        float(value) for value in route["lift_translation_world_m"]
    )
    if (
        route["approach_rule"] != EXPECTED_APPROACH_RULE
        or route["closure_rule"] != EXPECTED_CLOSURE_RULE
        or route["lift_rule"] != EXPECTED_LIFT_RULE
        or lift_translation != LIFT_TRANSLATION_WORLD_M
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in (approach_step, lift_step, lift_joint_step)
        )
        or lift_count < 2
        or not math.isclose(
            lift_step * (lift_count - 1),
            LIFT_TRANSLATION_WORLD_M[2],
            rel_tol=0.0,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        )
    ):
        raise CandidateJointRouteError(
            "ROUTE_SAMPLING_INVALID",
            "approach, closure, or 40 mm lift sampling changed",
        )

    firewall = _mapping(config["truth_firewall"], "truth_firewall")
    expected_firewall = (
        "legacy_joint_waypoints_allowed",
        "display_only_proposal_allowed",
        "formal_selected_policy_required_as_input",
        "online_object_ground_truth_allowed",
        "online_contact_ground_truth_allowed",
        "post_start_object_pose_write_allowed",
        "collision_result_claimed",
        "controller_execution_authorized",
        "isaac_dynamic_state_allowed",
        "hardware_state_allowed",
    )
    _exact_keys(firewall, expected_firewall, "truth_firewall")
    if any(firewall[name] is not False for name in expected_firewall):
        raise CandidateJointRouteError(
            "TRUTH_FIREWALL_CHANGED",
            "joint-route truth or execution firewall was relaxed",
        )

    return _JointRouteSettings(
        config_sha256=_file_sha256(path),
        sobol_point_count=bounded.sobol_point_count,
        sobol_interior_lower_fraction=bounded.sobol_interior_lower_fraction,
        sobol_interior_upper_fraction=bounded.sobol_interior_upper_fraction,
        orientation_residual_length_scale_m_per_rad=(
            bounded.orientation_residual_length_scale_m_per_rad
        ),
        position_tolerance_m=bounded.position_tolerance_m,
        orientation_tolerance_rad=bounded.orientation_tolerance_rad,
        function_tolerance=bounded.function_tolerance,
        step_tolerance=bounded.step_tolerance,
        gradient_tolerance=bounded.gradient_tolerance,
        maximum_function_evaluations=bounded.maximum_function_evaluations,
        approach_maximum_arm_joint_step_rad=approach_step,
        lift_cartesian_step_m=lift_step,
        lift_waypoint_count=lift_count,
        lift_maximum_arm_joint_step_rad=lift_joint_step,
    )


def _field(source: object, name: str) -> object:
    if isinstance(source, Mapping):
        return source[name]
    return getattr(source, name)


def _contract_document(source: object) -> Mapping[str, object]:
    vector_names = (
        "complete_joint_lower_limits_rad",
        "complete_joint_upper_limits_rad",
        "home_independent_joint_positions_rad",
        "pregrasp_independent_joint_positions_rad",
        "lift_path_parameters",
        "lift_position_errors_m",
        "lift_orientation_errors_rad",
    )
    scalar_names = (
        "pregrasp_position_error_m",
        "pregrasp_orientation_error_rad",
        "position_tolerance_m",
        "orientation_tolerance_rad",
        "approach_maximum_arm_joint_step_rad",
        "maximum_observed_approach_arm_joint_step_rad",
        "lift_cartesian_step_m",
        "lift_maximum_arm_joint_step_rad",
        "maximum_observed_lift_arm_joint_step_rad",
    )
    flag_names = (
        "arm_ik_solution_complete",
        "candidate_specific_motion_binding_complete",
        "complete_eleven_joint_route_binding_complete",
        "contact_triggered_hand_intervals_preserved",
        "joint_limits_complete",
        "joint_step_bound_complete",
        "continuous_collision_complete",
        "controller_execution_authorized",
        "formal_selection_input_used",
        "legacy_joint_waypoints_used",
        "display_only_proposal_used",
        "online_truth_used",
        "isaac_dynamic_state_used",
        "hardware_state_used",
    )
    return {
        "method_id": _field(source, "method_id"),
        "joint_route_contract_id": _field(source, "joint_route_contract_id"),
        "claim_scope": _field(source, "claim_scope"),
        "joint_route_config_sha256": _field(
            source, "joint_route_config_sha256"
        ),
        "candidate_route_state_sha256": _field(
            source, "candidate_route_state_sha256"
        ),
        "object_id": _field(source, "object_id"),
        "aggregate_collision_input_sha256": _field(
            source, "aggregate_collision_input_sha256"
        ),
        "aggregate_robot_model_sha256": _field(
            source, "aggregate_robot_model_sha256"
        ),
        "v9_parameter_key_hex": _field(source, "v9_parameter_key_hex"),
        "policy_sha256": _field(source, "policy_sha256"),
        "arm_joint_names": list(_field(source, "arm_joint_names")),
        "hand_joint_names": list(_field(source, "hand_joint_names")),
        "complete_joint_names": list(_field(source, "complete_joint_names")),
        "hand_base_link": _field(source, "hand_base_link"),
        **{
            name + "_binary64_hex": [
                _float_hex(value) for value in _field(source, name)
            ]
            for name in vector_names
        },
        "pregrasp_seed_index": int(_field(source, "pregrasp_seed_index")),
        "approach_independent_joint_waypoints_rad_binary64_hex": [
            [_float_hex(value) for value in row]
            for row in _field(source, "approach_independent_joint_waypoints_rad")
        ],
        "closure_stage_names": list(_field(source, "closure_stage_names")),
        "closure_stage_swept_independent_joint_intervals_rad_binary64_hex": (
            _interval_rows_document(
                _field(
                    source,
                    "closure_stage_swept_independent_joint_intervals_rad",
                )
            )
        ),
        "closure_stage_endpoint_independent_joint_intervals_rad_binary64_hex": (
            _interval_rows_document(
                _field(
                    source,
                    "closure_stage_endpoint_independent_joint_intervals_rad",
                )
            )
        ),
        "lift_independent_joint_interval_waypoints_rad_binary64_hex": (
            _interval_rows_document(
                _field(
                    source,
                    "lift_independent_joint_interval_waypoints_rad",
                )
            )
        ),
        "lift_world_from_hand_targets_binary64_hex": [
            [[_float_hex(value) for value in row] for row in matrix]
            for matrix in _field(source, "lift_world_from_hand_targets")
        ],
        **{
            name + "_binary64_hex": _float_hex(_field(source, name))
            for name in scalar_names
        },
        **{name: _field(source, name) for name in flag_names},
        "claim_limitations": list(_field(source, "claim_limitations")),
    }


def _certificate_sha256(source: object) -> str:
    payload = json.dumps(
        _contract_document(source),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _maximum_arm_step(rows: Sequence[Sequence[float]]) -> float:
    if len(rows) < 2:
        return 0.0
    values = np.asarray(rows, dtype=np.float64)
    return float(np.max(np.abs(np.diff(values[:, :7], axis=0))))


@dataclass(frozen=True)
class CandidateJointRouteContract:
    method_id: str
    joint_route_contract_id: str
    claim_scope: str
    joint_route_config_sha256: str
    candidate_route_state_sha256: str
    object_id: str
    aggregate_collision_input_sha256: str
    aggregate_robot_model_sha256: str
    v9_parameter_key_hex: str
    policy_sha256: str
    arm_joint_names: tuple[str, ...]
    hand_joint_names: tuple[str, ...]
    complete_joint_names: tuple[str, ...]
    hand_base_link: str
    complete_joint_lower_limits_rad: tuple[float, ...]
    complete_joint_upper_limits_rad: tuple[float, ...]
    home_independent_joint_positions_rad: tuple[float, ...]
    pregrasp_independent_joint_positions_rad: tuple[float, ...]
    pregrasp_seed_index: int
    pregrasp_position_error_m: float
    pregrasp_orientation_error_rad: float
    approach_independent_joint_waypoints_rad: tuple[tuple[float, ...], ...]
    closure_stage_names: tuple[str, ...]
    closure_stage_swept_independent_joint_intervals_rad: tuple[
        tuple[IntervalBounds, ...], ...
    ]
    closure_stage_endpoint_independent_joint_intervals_rad: tuple[
        tuple[IntervalBounds, ...], ...
    ]
    lift_path_parameters: tuple[float, ...]
    lift_independent_joint_interval_waypoints_rad: tuple[
        tuple[IntervalBounds, ...], ...
    ]
    lift_world_from_hand_targets: tuple[np.ndarray, ...]
    lift_position_errors_m: tuple[float, ...]
    lift_orientation_errors_rad: tuple[float, ...]
    position_tolerance_m: float
    orientation_tolerance_rad: float
    approach_maximum_arm_joint_step_rad: float
    maximum_observed_approach_arm_joint_step_rad: float
    lift_cartesian_step_m: float
    lift_maximum_arm_joint_step_rad: float
    maximum_observed_lift_arm_joint_step_rad: float
    arm_ik_solution_complete: bool
    candidate_specific_motion_binding_complete: bool
    complete_eleven_joint_route_binding_complete: bool
    contact_triggered_hand_intervals_preserved: bool
    joint_limits_complete: bool
    joint_step_bound_complete: bool
    continuous_collision_complete: bool
    controller_execution_authorized: bool
    formal_selection_input_used: bool
    legacy_joint_waypoints_used: bool
    display_only_proposal_used: bool
    online_truth_used: bool
    isaac_dynamic_state_used: bool
    hardware_state_used: bool
    claim_limitations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        transforms = tuple(
            _proper_transform(value, "lift_world_from_hand_target")
            for value in self.lift_world_from_hand_targets
        )
        if (
            self.method_id != METHOD_ID
            or self.joint_route_contract_id
            != EXPECTED_JOINT_ROUTE_CONTRACT_ID
            or self.claim_scope != EXPECTED_CLAIM_SCOPE
            or any(
                not _is_sha256(value)
                for value in (
                    self.joint_route_config_sha256,
                    self.candidate_route_state_sha256,
                    self.aggregate_collision_input_sha256,
                    self.aggregate_robot_model_sha256,
                    self.policy_sha256,
                    self.certificate_sha256,
                )
            )
            or len(self.v9_parameter_key_hex) != 80
            or any(
                character not in "0123456789abcdef"
                for character in self.v9_parameter_key_hex
            )
            or self.arm_joint_names != EXPECTED_ARM_JOINT_NAMES
            or self.hand_joint_names != EXPECTED_HAND_JOINT_NAMES
            or self.complete_joint_names != EXPECTED_INDEPENDENT_JOINTS
            or self.hand_base_link != EXPECTED_HAND_BASE_LINK
            or len(self.complete_joint_lower_limits_rad) != 11
            or len(self.complete_joint_upper_limits_rad) != 11
            or len(self.home_independent_joint_positions_rad) != 11
            or len(self.pregrasp_independent_joint_positions_rad) != 11
            or self.pregrasp_seed_index < 0
            or len(self.approach_independent_joint_waypoints_rad) < 2
            or self.closure_stage_names != EXPECTED_CLOSURE_STAGE_NAMES
            or len(
                self.closure_stage_swept_independent_joint_intervals_rad
            )
            != 3
            or len(
                self.closure_stage_endpoint_independent_joint_intervals_rad
            )
            != 3
            or len(self.lift_path_parameters) < 2
            or len(self.lift_path_parameters)
            != len(self.lift_independent_joint_interval_waypoints_rad)
            or len(self.lift_path_parameters) != len(transforms)
            or len(self.lift_path_parameters)
            != len(self.lift_position_errors_m)
            or len(self.lift_path_parameters)
            != len(self.lift_orientation_errors_rad)
            or self.lift_path_parameters[0] != 0.0
            or self.lift_path_parameters[-1] != 1.0
            or any(
                right <= left
                for left, right in zip(
                    self.lift_path_parameters,
                    self.lift_path_parameters[1:],
                )
            )
            or any(
                not math.isfinite(float(value))
                for value in (
                    *self.complete_joint_lower_limits_rad,
                    *self.complete_joint_upper_limits_rad,
                    *self.home_independent_joint_positions_rad,
                    *self.pregrasp_independent_joint_positions_rad,
                    self.pregrasp_position_error_m,
                    self.pregrasp_orientation_error_rad,
                    *self.lift_position_errors_m,
                    *self.lift_orientation_errors_rad,
                    self.position_tolerance_m,
                    self.orientation_tolerance_rad,
                    self.approach_maximum_arm_joint_step_rad,
                    self.maximum_observed_approach_arm_joint_step_rad,
                    self.lift_cartesian_step_m,
                    self.lift_maximum_arm_joint_step_rad,
                    self.maximum_observed_lift_arm_joint_step_rad,
                )
            )
            or any(
                lower >= upper
                for lower, upper in zip(
                    self.complete_joint_lower_limits_rad,
                    self.complete_joint_upper_limits_rad,
                )
            )
            or self.position_tolerance_m <= 0.0
            or self.orientation_tolerance_rad <= 0.0
            or self.approach_maximum_arm_joint_step_rad <= 0.0
            or self.lift_cartesian_step_m <= 0.0
            or self.lift_maximum_arm_joint_step_rad <= 0.0
            or self.pregrasp_position_error_m > self.position_tolerance_m
            or self.pregrasp_orientation_error_rad
            > self.orientation_tolerance_rad
            or any(
                value > self.position_tolerance_m
                for value in self.lift_position_errors_m
            )
            or any(
                value > self.orientation_tolerance_rad
                for value in self.lift_orientation_errors_rad
            )
            or self.claim_limitations != CLAIM_LIMITATIONS
            or any(
                value is not True
                for value in (
                    self.arm_ik_solution_complete,
                    self.candidate_specific_motion_binding_complete,
                    self.complete_eleven_joint_route_binding_complete,
                    self.contact_triggered_hand_intervals_preserved,
                    self.joint_limits_complete,
                    self.joint_step_bound_complete,
                )
            )
            or any(
                value is not False
                for value in (
                    self.continuous_collision_complete,
                    self.controller_execution_authorized,
                    self.formal_selection_input_used,
                    self.legacy_joint_waypoints_used,
                    self.display_only_proposal_used,
                    self.online_truth_used,
                    self.isaac_dynamic_state_used,
                    self.hardware_state_used,
                )
            )
        ):
            raise ValueError("candidate joint-route contract is incomplete")

        approach = self.approach_independent_joint_waypoints_rad
        if (
            any(len(row) != 11 for row in approach)
            or approach[0] != self.home_independent_joint_positions_rad
            or approach[-1] != self.pregrasp_independent_joint_positions_rad
            or any(
                row[7:] != self.pregrasp_independent_joint_positions_rad[7:]
                for row in approach
            )
            or self.maximum_observed_approach_arm_joint_step_rad
            != _maximum_arm_step(approach)
            or self.maximum_observed_approach_arm_joint_step_rad
            > self.approach_maximum_arm_joint_step_rad
            + 128.0 * np.finfo(np.float64).eps
            or any(
                value < lower or value > upper
                for row in approach
                for value, lower, upper in zip(
                    row,
                    self.complete_joint_lower_limits_rad,
                    self.complete_joint_upper_limits_rad,
                )
            )
        ):
            raise ValueError("HOME-to-pregrasp joint interpolation is invalid")

        for row in (
            *self.closure_stage_swept_independent_joint_intervals_rad,
            *self.closure_stage_endpoint_independent_joint_intervals_rad,
            *self.lift_independent_joint_interval_waypoints_rad,
        ):
            if (
                len(row) != 11
                or any(not isinstance(value, IntervalBounds) for value in row)
                or any(
                    value.lower < lower or value.upper > upper
                    for value, lower, upper in zip(
                        row,
                        self.complete_joint_lower_limits_rad,
                        self.complete_joint_upper_limits_rad,
                    )
                )
            ):
                raise ValueError("joint interval route is outside arm limits")

        lift_arm_rows = tuple(
            tuple(interval.lower for interval in row[:7])
            for row in self.lift_independent_joint_interval_waypoints_rad
        )
        if (
            any(
                interval.lower != interval.upper
                for row in self.lift_independent_joint_interval_waypoints_rad
                for interval in row[:7]
            )
            or lift_arm_rows[0]
            != self.pregrasp_independent_joint_positions_rad[:7]
            or self.maximum_observed_lift_arm_joint_step_rad
            != _maximum_arm_step(lift_arm_rows)
            or self.maximum_observed_lift_arm_joint_step_rad
            > self.lift_maximum_arm_joint_step_rad
            + 128.0 * np.finfo(np.float64).eps
        ):
            raise ValueError("vertical lift joint route is invalid")

        first_target = transforms[0]
        for parameter, transform in zip(self.lift_path_parameters, transforms):
            expected = np.array(first_target, copy=True)
            expected[:3, 3] += (
                np.asarray(LIFT_TRANSLATION_WORLD_M, dtype=np.float64)
                * float(parameter)
            )
            tolerance = (
                128.0
                * np.finfo(np.float64).eps
                * max(1.0, float(np.max(np.abs(expected))))
            )
            if not np.allclose(
                transform, expected, rtol=0.0, atol=tolerance
            ):
                raise ValueError("lift target is not the declared world-Z line")

        if self.certificate_sha256 != _certificate_sha256(self):
            raise ValueError("candidate joint-route certificate digest changed")
        object.__setattr__(self, "lift_world_from_hand_targets", transforms)

    @property
    def audit(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "object_id": self.object_id,
                "v9_parameter_key_hex": self.v9_parameter_key_hex,
                "pregrasp_seed_index": self.pregrasp_seed_index,
                "approach_waypoint_count": len(
                    self.approach_independent_joint_waypoints_rad
                ),
                "lift_waypoint_count": len(self.lift_path_parameters),
                "arm_ik_solution_complete": True,
                "complete_eleven_joint_route_binding_complete": True,
                "continuous_collision_complete": False,
                "controller_execution_authorized": False,
                "formal_selection_input_used": False,
                "certificate_sha256": self.certificate_sha256,
            }
        )


def _minimum_jerk_approach(
    home: Sequence[float],
    pregrasp: Sequence[float],
    maximum_arm_step_rad: float,
) -> tuple[tuple[float, ...], ...]:
    start = np.asarray(home, dtype=np.float64)
    goal = np.asarray(pregrasp, dtype=np.float64)
    maximum_delta = float(np.max(np.abs(goal[:7] - start[:7])))
    segment_count = max(
        1,
        int(math.ceil(1.875 * maximum_delta / maximum_arm_step_rad)),
    )
    parameters = np.linspace(0.0, 1.0, segment_count + 1)
    blend = (
        10.0 * parameters**3
        - 15.0 * parameters**4
        + 6.0 * parameters**5
    )
    rows = tuple(
        tuple(float(value) for value in start + value * (goal - start))
        for value in blend
    )
    rows = (tuple(float(value) for value in start), *rows[1:-1], tuple(float(value) for value in goal))
    if (
        _maximum_arm_step(rows)
        > maximum_arm_step_rad + 128.0 * np.finfo(np.float64).eps
    ):
        raise CandidateJointRouteError(
            "APPROACH_JOINT_STEP_BOUND_FAILED",
            repr(_maximum_arm_step(rows)),
        )
    return rows


def _complete_interval_row(
    arm_positions: Sequence[float],
    hand_intervals: Sequence[IntervalBounds],
) -> tuple[IntervalBounds, ...]:
    return tuple(
        IntervalBounds(float(value), float(value))
        for value in arm_positions
    ) + tuple(hand_intervals)


def _closure_route_intervals(
    route_state: CandidateRouteStateContract,
    pregrasp_arm: Sequence[float],
) -> tuple[
    tuple[tuple[IntervalBounds, ...], ...],
    tuple[tuple[IntervalBounds, ...], ...],
]:
    initial = tuple(
        IntervalBounds(float(value), float(value))
        for value in route_state.pregrasp_hand_joint_positions_rad
    )
    swept_rows: list[tuple[IntervalBounds, ...]] = []
    endpoint_rows: list[tuple[IntervalBounds, ...]] = []
    completed = list(initial)
    for hand_index in (1, 2, 3):
        moving = list(completed)
        moving[hand_index] = (
            route_state.closure_swept_hand_joint_intervals_rad[hand_index]
        )
        swept_rows.append(_complete_interval_row(pregrasp_arm, moving))
        completed[hand_index] = (
            route_state.closure_contact_hand_joint_intervals_rad[hand_index]
        )
        endpoint_rows.append(_complete_interval_row(pregrasp_arm, completed))
    return tuple(swept_rows), tuple(endpoint_rows)


def build_candidate_joint_route_contract(
    config_path: Path | str,
    *,
    route_state: object,
    aggregate_inputs: AggregateCollisionRuntimeInputCertificate,
    repository_root: Path | str,
) -> CandidateJointRouteContract:
    """Solve one accepted policy's arm IK and bind its complete joint route."""

    if type(route_state) is not CandidateRouteStateContract:
        raise CandidateJointRouteError(
            "CANDIDATE_ROUTE_STATE_REQUIRED",
            "input must be one exact six-stage route-state contract",
        )
    if type(aggregate_inputs) is not AggregateCollisionRuntimeInputCertificate:
        raise CandidateJointRouteError(
            "VERIFIED_AGGREGATE_INPUT_REQUIRED",
            "aggregate collision input must be the exact certified type",
        )
    root = Path(repository_root).resolve()
    raw_path = Path(config_path)
    path = (raw_path if raw_path.is_absolute() else root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CandidateJointRouteError(
            "CONFIG_OUTSIDE_REPOSITORY", str(path)
        ) from error
    if not path.is_file():
        raise CandidateJointRouteError("CONFIG_MISSING", str(path))
    settings = _validated_config(path)

    model = aggregate_inputs.kinematic_binding.model
    if (
        route_state.object_id != aggregate_inputs.object_id
        or route_state.aggregate_collision_input_sha256
        != aggregate_inputs.certificate_sha256
        or route_state.aggregate_robot_model_sha256
        != aggregate_inputs.kinematic_binding.model_sha256
        or route_state.complete_joint_names
        != tuple(aggregate_inputs.kinematic_binding.independent_joint_names)
        or tuple(model.independent_joint_names) != EXPECTED_INDEPENDENT_JOINTS
        or model.base_link != "world"
    ):
        raise CandidateJointRouteError(
            "ROUTE_AND_AGGREGATE_BINDING_MISMATCH",
            "route state and aggregate model are not the same object and robot",
        )

    lower, upper = model.joint_limit_vectors()
    arm_lower = np.asarray(lower[:7], dtype=np.float64)
    arm_upper = np.asarray(upper[:7], dtype=np.float64)
    home = tuple(route_state.home_independent_joint_positions_rad)
    home_arm = np.asarray(home[:7], dtype=np.float64)
    hand_open = tuple(route_state.pregrasp_hand_joint_positions_rad)
    seeds = pregrasp_seeds(
        home_arm=home_arm,
        lower=arm_lower,
        upper=arm_upper,
        settings=settings,
    )
    pregrasp_arm, pregrasp_position_error, pregrasp_orientation_error, seed_index = (
        solve_bounded_target(
            model=model,
            hand_positions=hand_open,
            target=route_state.world_from_hand_pregrasp_target,
            seeds=seeds,
            lower=arm_lower,
            upper=arm_upper,
            settings=settings,
            label="PREGRASP",
        )
    )
    pregrasp_complete = tuple(pregrasp_arm) + hand_open
    approach = _minimum_jerk_approach(
        home,
        pregrasp_complete,
        settings.approach_maximum_arm_joint_step_rad,
    )
    closure_swept, closure_endpoints = _closure_route_intervals(
        route_state, pregrasp_arm
    )

    lift_parameters = tuple(
        float(value)
        for value in np.linspace(0.0, 1.0, settings.lift_waypoint_count)
    )
    lift_targets: list[np.ndarray] = []
    lift_arm_rows: list[tuple[float, ...]] = [tuple(pregrasp_arm)]
    lift_position_errors: list[float] = [pregrasp_position_error]
    lift_orientation_errors: list[float] = [pregrasp_orientation_error]
    for parameter in lift_parameters:
        target = np.array(
            route_state.world_from_hand_pregrasp_target,
            dtype=np.float64,
            copy=True,
        )
        target[:3, 3] += (
            np.asarray(LIFT_TRANSLATION_WORLD_M, dtype=np.float64)
            * parameter
        )
        target.setflags(write=False)
        lift_targets.append(target)
    previous = np.asarray(pregrasp_arm, dtype=np.float64)
    for index, target in enumerate(lift_targets[1:], start=1):
        arm, position_error, orientation_error, _seed_index = solve_bounded_target(
            model=model,
            hand_positions=hand_open,
            target=target,
            seeds=(previous,),
            lower=arm_lower,
            upper=arm_upper,
            settings=settings,
            label=f"LIFT_{index}",
        )
        lift_arm_rows.append(arm)
        lift_position_errors.append(position_error)
        lift_orientation_errors.append(orientation_error)
        previous = np.asarray(arm, dtype=np.float64)
    maximum_lift_step = _maximum_arm_step(lift_arm_rows)
    if (
        maximum_lift_step
        > settings.lift_maximum_arm_joint_step_rad
        + 128.0 * np.finfo(np.float64).eps
    ):
        raise CandidateJointRouteError(
            "LIFT_JOINT_STEP_BOUND_FAILED", repr(maximum_lift_step)
        )

    lift_hand_intervals = tuple(
        route_state.closure_contact_hand_joint_intervals_rad
    )
    lift_intervals = tuple(
        _complete_interval_row(arm, lift_hand_intervals)
        for arm in lift_arm_rows
    )
    values: dict[str, object] = {
        "method_id": METHOD_ID,
        "joint_route_contract_id": EXPECTED_JOINT_ROUTE_CONTRACT_ID,
        "claim_scope": EXPECTED_CLAIM_SCOPE,
        "joint_route_config_sha256": settings.config_sha256,
        "candidate_route_state_sha256": route_state.certificate_sha256,
        "object_id": route_state.object_id,
        "aggregate_collision_input_sha256": (
            aggregate_inputs.certificate_sha256
        ),
        "aggregate_robot_model_sha256": (
            aggregate_inputs.kinematic_binding.model_sha256
        ),
        "v9_parameter_key_hex": route_state.v9_parameter_key_hex,
        "policy_sha256": route_state.policy_sha256,
        "arm_joint_names": EXPECTED_ARM_JOINT_NAMES,
        "hand_joint_names": EXPECTED_HAND_JOINT_NAMES,
        "complete_joint_names": EXPECTED_INDEPENDENT_JOINTS,
        "hand_base_link": EXPECTED_HAND_BASE_LINK,
        "complete_joint_lower_limits_rad": tuple(
            float(value) for value in lower
        ),
        "complete_joint_upper_limits_rad": tuple(
            float(value) for value in upper
        ),
        "home_independent_joint_positions_rad": home,
        "pregrasp_independent_joint_positions_rad": pregrasp_complete,
        "pregrasp_seed_index": seed_index,
        "pregrasp_position_error_m": pregrasp_position_error,
        "pregrasp_orientation_error_rad": pregrasp_orientation_error,
        "approach_independent_joint_waypoints_rad": approach,
        "closure_stage_names": EXPECTED_CLOSURE_STAGE_NAMES,
        "closure_stage_swept_independent_joint_intervals_rad": closure_swept,
        "closure_stage_endpoint_independent_joint_intervals_rad": (
            closure_endpoints
        ),
        "lift_path_parameters": lift_parameters,
        "lift_independent_joint_interval_waypoints_rad": lift_intervals,
        "lift_world_from_hand_targets": tuple(lift_targets),
        "lift_position_errors_m": tuple(lift_position_errors),
        "lift_orientation_errors_rad": tuple(lift_orientation_errors),
        "position_tolerance_m": settings.position_tolerance_m,
        "orientation_tolerance_rad": settings.orientation_tolerance_rad,
        "approach_maximum_arm_joint_step_rad": (
            settings.approach_maximum_arm_joint_step_rad
        ),
        "maximum_observed_approach_arm_joint_step_rad": (
            _maximum_arm_step(approach)
        ),
        "lift_cartesian_step_m": settings.lift_cartesian_step_m,
        "lift_maximum_arm_joint_step_rad": (
            settings.lift_maximum_arm_joint_step_rad
        ),
        "maximum_observed_lift_arm_joint_step_rad": maximum_lift_step,
        "arm_ik_solution_complete": True,
        "candidate_specific_motion_binding_complete": True,
        "complete_eleven_joint_route_binding_complete": True,
        "contact_triggered_hand_intervals_preserved": True,
        "joint_limits_complete": True,
        "joint_step_bound_complete": True,
        "continuous_collision_complete": False,
        "controller_execution_authorized": False,
        "formal_selection_input_used": False,
        "legacy_joint_waypoints_used": False,
        "display_only_proposal_used": False,
        "online_truth_used": False,
        "isaac_dynamic_state_used": False,
        "hardware_state_used": False,
        "claim_limitations": CLAIM_LIMITATIONS,
    }
    return CandidateJointRouteContract(
        **values,
        certificate_sha256=_certificate_sha256(values),
    )


__all__ = [
    "CLAIM_LIMITATIONS",
    "METHOD_ID",
    "CandidateJointRouteContract",
    "CandidateJointRouteError",
    "build_candidate_joint_route_contract",
    "solve_bounded_hand_base_ik",
]
