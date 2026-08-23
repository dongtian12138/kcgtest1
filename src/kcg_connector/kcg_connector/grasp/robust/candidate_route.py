"""Fail-closed route-state binding for every statically accepted grasp policy.

Formal selection must happen *after* each candidate has a complete route and
collision result.  This module therefore consumes one ``StaticV9AcceptedPolicy``
instead of the still-null final selection.  It binds HOME, the world-frame
pregrasp target, all three sequential closure ranges, and the 40 mm lift target.

It deliberately does not solve arm inverse kinematics or claim continuous
collision freedom.  Those remain explicit output fields and blockers rather
than silently importing the legacy hand-written arm waypoints.
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
from kcg_connector.grasp.robust.interval_kinematics import IntervalBounds
from kcg_connector.grasp.robust.object_world_pose import (
    SettledObjectWorldPoseCertificate,
)
from kcg_connector.grasp.robust.ray_closure import (
    CANDIDATE_REPRESENTATIVE_ROLE,
    CLOSURE_PARAMETER_DOMAIN_ID,
    CertifiedSequentialClosurePolicy,
    METHOD_ID as RAY_CLOSURE_METHOD_ID,
    MODEL_BINDING_COMPLETE_STATUS,
    REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
)
from kcg_connector.grasp.robust.top_level_candidate_generator import (
    CandidateLineage,
    StaticV9AcceptedPolicy,
    V9InvocationAuditBinding,
    canonicalize_v9_parameters,
)


METHOD_ID = "CARTS_HASH_BOUND_PER_ACCEPTED_POLICY_ROUTE_STATE_CONTRACT_V1"
EXPECTED_SCHEMA_VERSION = "carts_candidate_route_v1"
EXPECTED_ROUTE_CONTRACT_ID = (
    "CARTS_PER_ACCEPTED_POLICY_HOME_PREGRASP_CLOSURE_LIFT_ROUTE_V1"
)
EXPECTED_CLAIM_SCOPE = "STATIC_CANDIDATE_ROUTE_STATE_AND_TARGET_BINDING_ONLY"
EXPECTED_ARM_JOINT_NAMES = EXPECTED_INDEPENDENT_JOINTS[:7]
EXPECTED_HAND_JOINT_NAMES = EXPECTED_INDEPENDENT_JOINTS[7:]
EXPECTED_HAND_BASE_LINK = "handbase_link"
EXPECTED_HOME_ARM_JOINT_RULE = "ZERO_POSITION_OF_HASH_BOUND_URDF_ARM_JOINTS"
EXPECTED_STAGE_ORDER = (
    "HOME",
    "PREGRASP",
    "CLOSURE_FINGER_1",
    "CLOSURE_FINGER_2",
    "CLOSURE_FINGER_3",
    "LIFT_40MM",
)
EXPECTED_PREGRASP_TARGET_RULE = (
    "WORLD_FROM_OBJECT_TIMES_OBJECT_FROM_HAND_WITH_POLICY_OPEN_JOINTS"
)
EXPECTED_CLOSURE_SWEPT_RULE = (
    "INITIAL_TO_GUARANTEED_EARLIEST_PHASE_UPPER_FOR_EACH_DISJOINT_SUPPORT"
)
EXPECTED_CLOSURE_CONTACT_RULE = (
    "MINIMUM_POSSIBLE_EARLIEST_ROOT_LOWER_TO_GUARANTEED_EARLIEST_PHASE_UPPER"
)
EXPECTED_APPROACH_PATH_RULE = (
    "PENDING_DETERMINISTIC_IK_AND_CONTINUOUS_COLLISION_CERTIFICATE"
)
EXPECTED_LIFT_PATH_RULE = (
    "PENDING_DETERMINISTIC_IK_WITH_CONTACT_ENDPOINT_INTERVALS_AND_CONTINUOUS_COLLISION_CERTIFICATE"
)
LIFT_TRANSLATION_WORLD_M = (0.0, 0.0, 0.04)
CLAIM_LIMITATIONS = (
    "STATIC_ROUTE_STATE_AND_WORLD_TARGET_BINDING_ONLY",
    "PER_STATIC_ACCEPTED_POLICY_NOT_POST_SELECTION_SHORTCUT",
    "NO_ARM_IK_OR_EXECUTABLE_JOINT_TRAJECTORY_YET",
    "NO_CONTINUOUS_COLLISION_OR_ALLOWED_PAD_CONTACT_CERTIFICATE_YET",
    "NO_ISAAC_DYNAMIC_HARDWARE_OR_FORMAL_GRASP_SELECTION_CLAIM",
)


class CandidateRouteError(ValueError):
    """Raised when route-state inputs are absent, stale, or mismatched."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("candidate-route error fields cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


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
            raise CandidateRouteError(
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
        raise CandidateRouteError(
            "MAPPING_REQUIRED", f"{label} must be a string-keyed mapping"
        )
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set.difference(value))
    extra = sorted(set(value).difference(expected_set))
    if missing or extra:
        raise CandidateRouteError(
            "SCHEMA_MISMATCH", f"{label} missing={missing}, extra={extra}"
        )


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        return _mapping(
            yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader),
            "candidate route config",
        )
    except yaml.YAMLError as error:
        raise CandidateRouteError("INVALID_YAML", str(error)) from error


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
        raise CandidateRouteError("NONFINITE_VALUE", repr(value))
    return parsed.hex()


def _proper_transform(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape((4, 4))
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
        raise CandidateRouteError(
            "PROPER_TRANSFORM_REQUIRED", f"{label} must be one rigid transform"
        )
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _interval_document(rows: Sequence[IntervalBounds]) -> list[list[str]]:
    return [[_float_hex(row.lower), _float_hex(row.upper)] for row in rows]


def _contract_document(
    source: "CandidateRouteStateContract | Mapping[str, object]",
) -> dict[str, object]:
    def field(name: str) -> object:
        return source[name] if isinstance(source, Mapping) else getattr(source, name)

    return {
        "method_id": field("method_id"),
        "route_contract_id": field("route_contract_id"),
        "claim_scope": field("claim_scope"),
        "route_config_sha256": field("route_config_sha256"),
        "object_id": field("object_id"),
        "aggregate_collision_input_sha256": field(
            "aggregate_collision_input_sha256"
        ),
        "aggregate_robot_model_sha256": field("aggregate_robot_model_sha256"),
        "object_world_pose_sha256": field("object_world_pose_sha256"),
        "object_surface_geometry_sha256": field(
            "object_surface_geometry_sha256"
        ),
        "ray_closure_object_geometry_sha256": field(
            "ray_closure_object_geometry_sha256"
        ),
        "v9_parameter_key_hex": field("v9_parameter_key_hex"),
        "policy_sha256": field("policy_sha256"),
        "v9_model_contract_sha256": field("v9_model_contract_sha256"),
        "arm_joint_names": list(field("arm_joint_names")),
        "hand_joint_names": list(field("hand_joint_names")),
        "complete_joint_names": list(field("complete_joint_names")),
        "hand_base_link": field("hand_base_link"),
        "stage_order": list(field("stage_order")),
        "home_independent_joint_positions_rad": [
            _float_hex(value)
            for value in field("home_independent_joint_positions_rad")
        ],
        "pregrasp_hand_joint_positions_rad": [
            _float_hex(value)
            for value in field("pregrasp_hand_joint_positions_rad")
        ],
        "world_from_hand_pregrasp_target": [
            [_float_hex(value) for value in row]
            for row in field("world_from_hand_pregrasp_target")
        ],
        "world_from_hand_lift_target": [
            [_float_hex(value) for value in row]
            for row in field("world_from_hand_lift_target")
        ],
        "world_from_object_lift_target": [
            [_float_hex(value) for value in row]
            for row in field("world_from_object_lift_target")
        ],
        "closure_swept_hand_joint_intervals_rad": _interval_document(
            field("closure_swept_hand_joint_intervals_rad")
        ),
        "closure_contact_hand_joint_intervals_rad": _interval_document(
            field("closure_contact_hand_joint_intervals_rad")
        ),
        "lift_translation_world_m": [
            _float_hex(value) for value in field("lift_translation_world_m")
        ],
        "route_state_binding_complete": field("route_state_binding_complete"),
        "arm_ik_solution_complete": field("arm_ik_solution_complete"),
        "candidate_specific_motion_binding_complete": field(
            "candidate_specific_motion_binding_complete"
        ),
        "continuous_collision_complete": field("continuous_collision_complete"),
        "formal_selection_input_used": field("formal_selection_input_used"),
        "selection_occurs_after_route_evaluation": field(
            "selection_occurs_after_route_evaluation"
        ),
        "legacy_joint_waypoints_used": field("legacy_joint_waypoints_used"),
        "display_only_proposal_used": field("display_only_proposal_used"),
        "online_truth_used": field("online_truth_used"),
        "isaac_dynamic_state_used": field("isaac_dynamic_state_used"),
        "hardware_state_used": field("hardware_state_used"),
        "claim_limitations": list(field("claim_limitations")),
    }


def _certificate_sha256(
    source: "CandidateRouteStateContract | Mapping[str, object]",
) -> str:
    return hashlib.sha256(
        json.dumps(
            _contract_document(source),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CandidateRouteStateContract:
    method_id: str
    route_contract_id: str
    claim_scope: str
    route_config_sha256: str
    object_id: str
    aggregate_collision_input_sha256: str
    aggregate_robot_model_sha256: str
    object_world_pose_sha256: str
    object_surface_geometry_sha256: str
    ray_closure_object_geometry_sha256: str
    v9_parameter_key_hex: str
    policy_sha256: str
    v9_model_contract_sha256: str
    arm_joint_names: tuple[str, ...]
    hand_joint_names: tuple[str, ...]
    complete_joint_names: tuple[str, ...]
    hand_base_link: str
    stage_order: tuple[str, ...]
    home_independent_joint_positions_rad: tuple[float, ...]
    pregrasp_hand_joint_positions_rad: tuple[float, ...]
    world_from_hand_pregrasp_target: np.ndarray
    world_from_hand_lift_target: np.ndarray
    world_from_object_lift_target: np.ndarray
    closure_swept_hand_joint_intervals_rad: tuple[IntervalBounds, ...]
    closure_contact_hand_joint_intervals_rad: tuple[IntervalBounds, ...]
    lift_translation_world_m: tuple[float, float, float]
    route_state_binding_complete: bool
    arm_ik_solution_complete: bool
    candidate_specific_motion_binding_complete: bool
    continuous_collision_complete: bool
    formal_selection_input_used: bool
    selection_occurs_after_route_evaluation: bool
    legacy_joint_waypoints_used: bool
    display_only_proposal_used: bool
    online_truth_used: bool
    isaac_dynamic_state_used: bool
    hardware_state_used: bool
    claim_limitations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        pregrasp = _proper_transform(
            self.world_from_hand_pregrasp_target,
            "world_from_hand_pregrasp_target",
        )
        lift_hand = _proper_transform(
            self.world_from_hand_lift_target,
            "world_from_hand_lift_target",
        )
        lift_object = _proper_transform(
            self.world_from_object_lift_target,
            "world_from_object_lift_target",
        )
        if (
            self.method_id != METHOD_ID
            or self.route_contract_id != EXPECTED_ROUTE_CONTRACT_ID
            or self.claim_scope != EXPECTED_CLAIM_SCOPE
            or any(
                not _is_sha256(value)
                for value in (
                    self.route_config_sha256,
                    self.aggregate_collision_input_sha256,
                    self.aggregate_robot_model_sha256,
                    self.object_world_pose_sha256,
                    self.object_surface_geometry_sha256,
                    self.ray_closure_object_geometry_sha256,
                    self.policy_sha256,
                    self.v9_model_contract_sha256,
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
            or self.stage_order != EXPECTED_STAGE_ORDER
            or len(self.home_independent_joint_positions_rad) != 11
            or len(self.pregrasp_hand_joint_positions_rad) != 4
            or self.home_independent_joint_positions_rad[:7] != (0.0,) * 7
            or self.home_independent_joint_positions_rad[7:]
            != self.pregrasp_hand_joint_positions_rad
            or len(self.closure_swept_hand_joint_intervals_rad) != 4
            or len(self.closure_contact_hand_joint_intervals_rad) != 4
            or any(
                not isinstance(row, IntervalBounds)
                for row in (
                    *self.closure_swept_hand_joint_intervals_rad,
                    *self.closure_contact_hand_joint_intervals_rad,
                )
            )
            or self.lift_translation_world_m != LIFT_TRANSLATION_WORLD_M
            or self.route_state_binding_complete is not True
            or self.selection_occurs_after_route_evaluation is not True
            or any(
                value is not False
                for value in (
                    self.arm_ik_solution_complete,
                    self.candidate_specific_motion_binding_complete,
                    self.continuous_collision_complete,
                    self.formal_selection_input_used,
                    self.legacy_joint_waypoints_used,
                    self.display_only_proposal_used,
                    self.online_truth_used,
                    self.isaac_dynamic_state_used,
                    self.hardware_state_used,
                )
            )
            or self.claim_limitations != CLAIM_LIMITATIONS
        ):
            raise ValueError("candidate route-state contract is incomplete")
        expected_lift_hand = np.array(pregrasp, copy=True)
        expected_lift_hand[:3, 3] += np.asarray(LIFT_TRANSLATION_WORLD_M)
        if not np.array_equal(lift_hand, expected_lift_hand):
            raise ValueError("lift hand target is not exactly 40 mm above pregrasp")
        if self.certificate_sha256 != _certificate_sha256(self):
            raise ValueError("candidate route-state certificate digest changed")
        object.__setattr__(self, "world_from_hand_pregrasp_target", pregrasp)
        object.__setattr__(self, "world_from_hand_lift_target", lift_hand)
        object.__setattr__(self, "world_from_object_lift_target", lift_object)

    @property
    def audit(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "object_id": self.object_id,
                "v9_parameter_key_hex": self.v9_parameter_key_hex,
                "policy_sha256": self.policy_sha256,
                "stage_order": self.stage_order,
                "lift_distance_m": self.lift_translation_world_m[2],
                "route_state_binding_complete": True,
                "arm_ik_solution_complete": False,
                "candidate_specific_motion_binding_complete": False,
                "continuous_collision_complete": False,
                "formal_selection_input_used": False,
                "selection_occurs_after_route_evaluation": True,
                "dynamic_claimed": False,
                "certificate_sha256": self.certificate_sha256,
            }
        )


def _validated_config(path: Path) -> Mapping[str, Any]:
    config = _load_config(path)
    _exact_keys(
        config,
        (
            "schema_version",
            "route_contract_id",
            "claim_scope",
            "robot_binding",
            "route_semantics",
            "truth_firewall",
        ),
        "candidate route config",
    )
    if (
        config["schema_version"] != EXPECTED_SCHEMA_VERSION
        or config["route_contract_id"] != EXPECTED_ROUTE_CONTRACT_ID
        or config["claim_scope"] != EXPECTED_CLAIM_SCOPE
    ):
        raise CandidateRouteError(
            "CONTRACT_IDENTITY_MISMATCH", "candidate route identity changed"
        )
    robot = _mapping(config["robot_binding"], "robot_binding")
    _exact_keys(
        robot,
        (
            "arm_joint_names",
            "hand_joint_names",
            "hand_base_link",
            "home_arm_joint_rule",
            "home_arm_joint_positions_rad",
        ),
        "robot_binding",
    )
    if (
        tuple(robot["arm_joint_names"]) != EXPECTED_ARM_JOINT_NAMES
        or tuple(robot["hand_joint_names"]) != EXPECTED_HAND_JOINT_NAMES
        or robot["hand_base_link"] != EXPECTED_HAND_BASE_LINK
        or robot["home_arm_joint_rule"] != EXPECTED_HOME_ARM_JOINT_RULE
        or tuple(float(value) for value in robot["home_arm_joint_positions_rad"])
        != (0.0,) * 7
    ):
        raise CandidateRouteError(
            "ROBOT_BINDING_CHANGED", "joint order, hand base, or HOME changed"
        )
    route = _mapping(config["route_semantics"], "route_semantics")
    _exact_keys(
        route,
        (
            "stage_order",
            "pregrasp_target_rule",
            "closure_swept_interval_rule",
            "closure_contact_interval_rule",
            "lift_translation_world_m",
            "lift_distance_m",
            "approach_arm_path_rule",
            "lift_arm_path_rule",
            "formal_selection_occurs_after_route_evaluation",
        ),
        "route_semantics",
    )
    lift = tuple(float(value) for value in route["lift_translation_world_m"])
    if (
        tuple(route["stage_order"]) != EXPECTED_STAGE_ORDER
        or route["pregrasp_target_rule"] != EXPECTED_PREGRASP_TARGET_RULE
        or route["closure_swept_interval_rule"] != EXPECTED_CLOSURE_SWEPT_RULE
        or route["closure_contact_interval_rule"]
        != EXPECTED_CLOSURE_CONTACT_RULE
        or lift != LIFT_TRANSLATION_WORLD_M
        or float(route["lift_distance_m"]) != LIFT_TRANSLATION_WORLD_M[2]
        or route["approach_arm_path_rule"] != EXPECTED_APPROACH_PATH_RULE
        or route["lift_arm_path_rule"] != EXPECTED_LIFT_PATH_RULE
        or route["formal_selection_occurs_after_route_evaluation"] is not True
    ):
        raise CandidateRouteError(
            "ROUTE_SEMANTICS_CHANGED", "route stages or target rules changed"
        )
    firewall = _mapping(config["truth_firewall"], "truth_firewall")
    _exact_keys(
        firewall,
        (
            "legacy_joint_waypoints_allowed",
            "display_only_proposal_allowed",
            "formal_selected_policy_required_as_input",
            "per_static_accepted_policy_required_as_input",
            "online_object_ground_truth_allowed",
            "online_contact_ground_truth_allowed",
            "post_start_object_pose_write_allowed",
            "isaac_dynamic_state_allowed",
            "hardware_state_allowed",
        ),
        "truth_firewall",
    )
    if (
        firewall["per_static_accepted_policy_required_as_input"] is not True
        or any(
            firewall[name] is not False
            for name in firewall
            if name != "per_static_accepted_policy_required_as_input"
        )
    ):
        raise CandidateRouteError(
            "TRUTH_FIREWALL_CHANGED", "route input firewall was relaxed"
        )
    return config


def _validated_policy(
    accepted: object,
) -> tuple[CertifiedSequentialClosurePolicy, object]:
    if type(accepted) is not StaticV9AcceptedPolicy:
        raise CandidateRouteError(
            "STATIC_ACCEPTED_POLICY_REQUIRED",
            "route input must be one exact per-candidate StaticV9AcceptedPolicy",
        )
    policy = accepted.sequential_closure_policy
    audit = accepted.v9_audit
    binding = accepted.invocation_binding
    if type(policy) is not CertifiedSequentialClosurePolicy:
        raise CandidateRouteError(
            "CERTIFIED_CONTACT_RANGE_POLICY_REQUIRED",
            "display-only or exact-point substitutes are forbidden",
        )
    if type(binding) is not V9InvocationAuditBinding:
        raise CandidateRouteError(
            "V9_INVOCATION_BINDING_REQUIRED", "accepted policy lacks V9 lineage"
        )
    required = (
        "method_id",
        "closure_parameter_domain_id",
        "parameter_layout",
        "failure_reason",
        "model_binding_complete",
        "model_binding_status",
        "object_geometry_sha256",
        "model_contract_sha256",
        "pad_order",
        "pad_link_names",
        "independent_actuation_supports",
        "closing_directions_physical",
        "possible_first_contact_set_sha256",
        "candidate_role",
        "candidate_exact_contact_endpoint_certified",
        "full_verified_pad_mesh_used",
        "pad_face_subset_input_allowed",
        "subdivision_budget_exhausted",
        "model_contract_canonical_json",
    )
    if audit is None or any(not hasattr(audit, name) for name in required):
        raise CandidateRouteError(
            "V9_AUDIT_INCOMPLETE", "accepted policy audit lacks formal bindings"
        )
    try:
        canonical = canonicalize_v9_parameters(
            accepted.v9_parameters_unit,
            parameter_layout=tuple(audit.parameter_layout),
        )
    except Exception as error:
        raise CandidateRouteError(
            "V9_PARAMETER_BINDING_INVALID", str(error)
        ) from error
    if (
        canonical.exact_key_hex != accepted.v9_parameter_key_hex
        or binding.raw_v9_audit is not audit
        or binding.method_id != audit.method_id
        or binding.parameter_domain_id != audit.closure_parameter_domain_id
        or tuple(binding.parameter_layout) != tuple(audit.parameter_layout)
        or tuple(binding.requested_parameters_unit)
        != tuple(accepted.v9_parameters_unit)
        or binding.requested_parameter_key_hex
        != accepted.v9_parameter_key_hex
        or not accepted.lineage
        or any(type(row) is not CandidateLineage for row in accepted.lineage)
    ):
        raise CandidateRouteError(
            "ACCEPTED_POLICY_LINEAGE_MISMATCH",
            "accepted policy does not bind one canonical V9 request",
        )
    if (
        audit.method_id != RAY_CLOSURE_METHOD_ID
        or audit.closure_parameter_domain_id != CLOSURE_PARAMETER_DOMAIN_ID
        or audit.failure_reason != REPRESENTATIVE_PROPOSAL_FAILURE_REASON
        or audit.model_binding_complete is not True
        or audit.model_binding_status != MODEL_BINDING_COMPLETE_STATUS
        or audit.candidate_role != CANDIDATE_REPRESENTATIVE_ROLE
        or audit.candidate_exact_contact_endpoint_certified is not False
        or audit.full_verified_pad_mesh_used is not True
        or audit.pad_face_subset_input_allowed is not False
        or audit.subdivision_budget_exhausted is not False
        or policy.object_geometry_sha256 != audit.object_geometry_sha256
        or policy.model_contract_sha256 != audit.model_contract_sha256
        or policy.pad_order != tuple(audit.pad_order)
        or policy.independent_actuation_supports
        != tuple(tuple(row) for row in audit.independent_actuation_supports)
        or policy.closing_directions_physical
        != tuple(tuple(row) for row in audit.closing_directions_physical)
        or tuple(row.set_sha256 for row in policy.possible_first_contact_sets)
        != tuple(audit.possible_first_contact_set_sha256)
    ):
        raise CandidateRouteError(
            "POLICY_V9_AUDIT_MISMATCH",
            "contact ranges differ from the accepted V9 evidence",
        )
    return policy, audit


def _closure_intervals(
    policy: CertifiedSequentialClosurePolicy,
) -> tuple[tuple[IntervalBounds, ...], tuple[IntervalBounds, ...]]:
    initial = tuple(policy.initial_independent_joint_positions_rad)
    index_by_name = {
        name: index for index, name in enumerate(policy.independent_joint_names)
    }
    swept = [[value, value] for value in initial]
    contact = [[value, value] for value in initial]
    for support, direction, contact_set in zip(
        policy.independent_actuation_supports,
        policy.closing_directions_physical,
        policy.possible_first_contact_sets,
    ):
        if len(support) != 1:
            raise CandidateRouteError(
                "SINGLE_SUPPORT_ROUTE_V1_REQUIRED",
                "each closure stage must move one independent finger joint",
            )
        joint_index = index_by_name[support[0]]
        upper = float(contact_set.guaranteed_earliest_phase_upper)
        roots = contact_set.possible_earliest_roots
        lower = min(float(row.certificate.phase.lower) for row in roots)
        if not math.isfinite(lower) or not math.isfinite(upper) or not (
            0.0 <= lower <= upper
        ):
            raise CandidateRouteError(
                "CONTACT_PHASE_RANGE_INVALID", contact_set.pad_name
            )
        rate = float(direction[joint_index])
        start = initial[joint_index]
        swept_endpoint = start + upper * rate
        contact_first = start + lower * rate
        swept[joint_index] = [min(start, swept_endpoint), max(start, swept_endpoint)]
        contact[joint_index] = [
            min(contact_first, swept_endpoint),
            max(contact_first, swept_endpoint),
        ]
    return (
        tuple(IntervalBounds(*row) for row in swept),
        tuple(IntervalBounds(*row) for row in contact),
    )


def build_candidate_route_state_contract(
    config_path: Path | str,
    *,
    accepted_policy: object,
    aggregate_inputs: AggregateCollisionRuntimeInputCertificate,
    object_world_pose: SettledObjectWorldPoseCertificate,
    repository_root: Path | str,
) -> CandidateRouteStateContract:
    """Bind one accepted policy to world targets without inventing arm IK."""

    if not isinstance(
        aggregate_inputs, AggregateCollisionRuntimeInputCertificate
    ):
        raise CandidateRouteError(
            "VERIFIED_AGGREGATE_INPUT_REQUIRED",
            "aggregate collision input must be the exact certified type",
        )
    if not isinstance(object_world_pose, SettledObjectWorldPoseCertificate):
        raise CandidateRouteError(
            "VERIFIED_OBJECT_WORLD_POSE_REQUIRED",
            "object pose must be the exact settled-pose certificate",
        )
    root = Path(repository_root).resolve()
    raw_path = Path(config_path)
    path = (raw_path if raw_path.is_absolute() else root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CandidateRouteError(
            "CONFIG_OUTSIDE_REPOSITORY", str(path)
        ) from error
    if not path.is_file():
        raise CandidateRouteError("CONFIG_MISSING", str(path))
    config = _validated_config(path)
    policy, audit = _validated_policy(accepted_policy)

    aggregate_model = aggregate_inputs.kinematic_binding.model
    aggregate_links = {aggregate_model.base_link}
    aggregate_links.update(
        joint.parent_link for joint in aggregate_model.joints.values()
    )
    aggregate_links.update(
        joint.child_link for joint in aggregate_model.joints.values()
    )
    if (
        aggregate_inputs.object_id != object_world_pose.object_id
        or object_world_pose.aggregate_collision_input_sha256
        != aggregate_inputs.certificate_sha256
        or object_world_pose.shared_environment_certificate_sha256
        != aggregate_inputs.shared_environment.certificate_sha256
        or object_world_pose.object_source_asset_sha256
        != aggregate_inputs.object_surface.source_asset_sha256
        or object_world_pose.object_surface_geometry_sha256
        != aggregate_inputs.object_surface.geometry_sha256
        or policy.object_geometry_sha256
        != aggregate_inputs.object_surface.ray_closure_object_geometry_sha256
    ):
        raise CandidateRouteError(
            "CROSS_OBJECT_OR_SCENE_BINDING_MISMATCH",
            "policy, object pose, and collision scene are not the same object",
        )
    if (
        tuple(aggregate_inputs.kinematic_binding.independent_joint_names)
        != EXPECTED_INDEPENDENT_JOINTS
        or policy.independent_joint_names != EXPECTED_HAND_JOINT_NAMES
        or object_world_pose.root_frame != aggregate_model.base_link
        or EXPECTED_HAND_BASE_LINK not in aggregate_links
    ):
        raise CandidateRouteError(
            "AGGREGATE_AND_POLICY_JOINT_BINDING_MISMATCH",
            "four hand joints cannot be mapped into the eleven-joint robot",
        )
    try:
        manifest = json.loads(audit.model_contract_canonical_json)
        manifest_hand = manifest["hand"]
        manifest_joint_names = tuple(manifest_hand["independent_joint_names"])
        manifest_base_link = str(manifest_hand["base_link"])
        manifest_pad_names = tuple(
            row["name"] for row in manifest["verified_pads"]
        )
        manifest_pad_links = tuple(
            row["link_name"] for row in manifest["verified_pads"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CandidateRouteError(
            "V9_MODEL_MANIFEST_INVALID", str(error)
        ) from error
    if (
        manifest_joint_names != EXPECTED_HAND_JOINT_NAMES
        or manifest_base_link != EXPECTED_HAND_BASE_LINK
        or hashlib.sha256(
            audit.model_contract_canonical_json.encode("utf-8")
        ).hexdigest()
        != policy.model_contract_sha256
        or manifest_pad_names != policy.pad_order
        or manifest_pad_links != tuple(audit.pad_link_names)
        or tuple(audit.pad_link_names)
        != tuple(row.link_name for row in aggregate_inputs.terminal_roles)
    ):
        raise CandidateRouteError(
            "V9_HAND_MANIFEST_MISMATCH",
            "V9 hand base, joints, or PAD links differ from the aggregate robot",
        )

    home_arm = tuple(
        float(value)
        for value in config["robot_binding"]["home_arm_joint_positions_rad"]
    )
    home = home_arm + tuple(policy.initial_independent_joint_positions_rad)
    try:
        aggregate_model.resolve_joint_positions(home)
    except ValueError as error:
        raise CandidateRouteError(
            "HOME_OR_POLICY_OPEN_STATE_OUTSIDE_LIMITS", str(error)
        ) from error

    object_from_hand = _proper_transform(policy.object_from_hand, "object_from_hand")
    world_from_hand = _proper_transform(
        object_world_pose.world_from_object @ object_from_hand,
        "world_from_hand_pregrasp_target",
    )
    lift_transform = np.eye(4, dtype=np.float64)
    lift_transform[:3, 3] = np.asarray(LIFT_TRANSLATION_WORLD_M)
    world_from_hand_lift = _proper_transform(
        lift_transform @ world_from_hand,
        "world_from_hand_lift_target",
    )
    world_from_object_lift = _proper_transform(
        lift_transform @ object_world_pose.world_from_object,
        "world_from_object_lift_target",
    )
    swept_intervals, contact_intervals = _closure_intervals(policy)

    values: dict[str, object] = {
        "method_id": METHOD_ID,
        "route_contract_id": EXPECTED_ROUTE_CONTRACT_ID,
        "claim_scope": EXPECTED_CLAIM_SCOPE,
        "route_config_sha256": _file_sha256(path),
        "object_id": aggregate_inputs.object_id,
        "aggregate_collision_input_sha256": aggregate_inputs.certificate_sha256,
        "aggregate_robot_model_sha256": (
            aggregate_inputs.kinematic_binding.model_sha256
        ),
        "object_world_pose_sha256": object_world_pose.certificate_sha256,
        "object_surface_geometry_sha256": (
            aggregate_inputs.object_surface.geometry_sha256
        ),
        "ray_closure_object_geometry_sha256": (
            aggregate_inputs.object_surface.ray_closure_object_geometry_sha256
        ),
        "v9_parameter_key_hex": accepted_policy.v9_parameter_key_hex,
        "policy_sha256": policy.policy_sha256,
        "v9_model_contract_sha256": policy.model_contract_sha256,
        "arm_joint_names": EXPECTED_ARM_JOINT_NAMES,
        "hand_joint_names": EXPECTED_HAND_JOINT_NAMES,
        "complete_joint_names": EXPECTED_INDEPENDENT_JOINTS,
        "hand_base_link": EXPECTED_HAND_BASE_LINK,
        "stage_order": EXPECTED_STAGE_ORDER,
        "home_independent_joint_positions_rad": home,
        "pregrasp_hand_joint_positions_rad": tuple(
            policy.initial_independent_joint_positions_rad
        ),
        "world_from_hand_pregrasp_target": world_from_hand,
        "world_from_hand_lift_target": world_from_hand_lift,
        "world_from_object_lift_target": world_from_object_lift,
        "closure_swept_hand_joint_intervals_rad": swept_intervals,
        "closure_contact_hand_joint_intervals_rad": contact_intervals,
        "lift_translation_world_m": LIFT_TRANSLATION_WORLD_M,
        "route_state_binding_complete": True,
        "arm_ik_solution_complete": False,
        "candidate_specific_motion_binding_complete": False,
        "continuous_collision_complete": False,
        "formal_selection_input_used": False,
        "selection_occurs_after_route_evaluation": True,
        "legacy_joint_waypoints_used": False,
        "display_only_proposal_used": False,
        "online_truth_used": False,
        "isaac_dynamic_state_used": False,
        "hardware_state_used": False,
        "claim_limitations": CLAIM_LIMITATIONS,
    }
    return CandidateRouteStateContract(
        **values,
        certificate_sha256=_certificate_sha256(values),
    )


__all__ = [
    "CLAIM_LIMITATIONS",
    "EXPECTED_STAGE_ORDER",
    "METHOD_ID",
    "CandidateRouteError",
    "CandidateRouteStateContract",
    "build_candidate_route_state_contract",
]
