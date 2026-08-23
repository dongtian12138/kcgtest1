"""Fail-closed collision coverage for one complete 11-joint grasp route.

The checker consumes only hash-bound static inputs and a deterministic joint
route.  Every route segment is enclosed by a joint-position box.  Interval FK
then encloses every robot triangle; fixed binary64 projection axes may prove
two triangle hulls strictly separated, but a missing strict axis is never
treated as clearance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from kcg_connector.grasp.robust.aggregate_collision_inputs import (
    AggregateCollisionRuntimeInputCertificate,
)
from kcg_connector.grasp.robust.candidate_joint_route import (
    CandidateJointRouteContract,
)
from kcg_connector.grasp.robust.candidate_route import (
    CandidateRouteStateContract,
)
from kcg_connector.grasp.robust.hand_contract import CARTSHandContract
from kcg_connector.grasp.robust.interval_kinematics import (
    INTERVAL_RIGID_TRANSFORM_METHOD_ID,
    DirectedIntervalKinematics,
    IntervalBounds,
    IntervalKinematicsError,
    IntervalRigidTransform,
)
from kcg_connector.grasp.robust.object_world_pose import (
    SettledObjectWorldPoseCertificate,
)
from kcg_connector.grasp.robust.ray_closure import (
    CertifiedSequentialClosurePolicy,
)
from kcg_connector.grasp.robust.self_collision_execution_policy import (
    EXPECTED_BASE_PAIR_COUNT,
    EXPECTED_FORBIDDEN_PAIR_COUNT,
    EXPECTED_STRUCTURAL_INTERFACE_PAIR_COUNT,
    METHOD_ID as SELF_COLLISION_EXECUTION_POLICY_METHOD_ID,
    STRUCTURAL_INTERFACE_RULE,
    build_self_collision_execution_policy,
)
from kcg_connector.grasp.robust.top_level_candidate_generator import (
    StaticV9AcceptedPolicy,
)


METHOD_ID = "CARTS_COMPLETE_11_JOINT_ROUTE_INTERVAL_COLLISION_V2"
PAD_ROOT_BINDING_METHOD_ID = (
    "CARTS_FULL_AUTHORIZED_PAD_ROOT_INDEPENDENT_CLOSED_SHELL_ROLE_V1"
)
EXPECTED_SCHEMA_VERSION = "carts_candidate_route_collision_v1"
EXPECTED_CONTRACT_ID = "CARTS_COMPLETE_11_JOINT_ROUTE_COLLISION_V1"
EXPECTED_CLAIM_SCOPE = (
    "STATIC_CONTINUOUS_ROBOT_OBJECT_ENVIRONMENT_COLLISION_ONLY"
)
EXPECTED_STAGE_ORDER = (
    "HOME",
    "PREGRASP",
    "FINGER_1_CONTACT_TRIGGERED_CLOSE",
    "FINGER_2_CONTACT_TRIGGERED_CLOSE",
    "FINGER_3_CONTACT_TRIGGERED_CLOSE",
    "LIFT_40_MM",
)
EXPECTED_INTERVAL_METHOD = (
    "JOINT_BOX_INTERVAL_FK_FIXED_AXIS_TRIANGLE_SEPARATION_V1"
)
EXPECTED_SPLIT_RULE = (
    "WIDEST_NORMALIZED_NONZERO_JOINT_INTERVAL_LEFT_FIRST"
)
EXPECTED_TABLE_RELEASE_RULE = (
    "INITIAL_EXACT_SUPPORT_THEN_MONOTONE_WORLD_POSITIVE_Z"
)
CLAIM_LIMITATIONS = (
    "STATIC_HASH_BOUND_GEOMETRY_AND_DECLARED_JOINT_ROUTE_ONLY",
    "JOINT_BOXES_CONSERVATIVELY_INCLUDE_OFF_PATH_COMBINATIONS",
    "STRICT_FIXED_AXIS_SEPARATION_IS_SUFFICIENT_NOT_NECESSARY",
    "FULL_AUTHORIZED_PAD_ROOT_IDENTITY_IS_INDEPENDENT_OF_CLOSED_SHELL_"
    "FACE_ROLE_PARTITION",
    "ALLOWED_PAD_CONTACT_DOES_NOT_PROVE_DYNAMIC_ATTACHMENT",
    "NO_ISAAC_PHYSX_HARDWARE_OR_CONTROLLER_EXECUTION_CLAIM",
    "NO_FORMAL_SELECTION_OR_NONFRICTION_UNCERTAINTY_CLAIM",
)


class CandidateRouteCollisionError(ValueError):
    """Fail-closed route-collision input or contract error."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("route-collision error fields cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


class CandidateRouteCollisionState(str, Enum):
    CERTIFIED_FREE = "CERTIFIED_FREE"
    NOT_CERTIFIABLE = "NOT_CERTIFIABLE"


@dataclass(frozen=True)
class RouteCollisionFailure:
    stage_name: str
    segment_index: int
    first_body: str
    second_body: str
    reason: str

    def __post_init__(self) -> None:
        if (
            not self.stage_name
            or type(self.segment_index) is not int
            or self.segment_index < 0
            or not self.first_body
            or not self.second_body
            or not self.reason
        ):
            raise CandidateRouteCollisionError(
                "MALFORMED_COLLISION_FAILURE",
                "stage, segment, bodies, and reason must be explicit",
            )


@dataclass(frozen=True)
class CandidateRouteCollisionCertificate:
    method_id: str
    route_collision_contract_id: str
    claim_scope: str
    route_collision_config_sha256: str
    candidate_route_state_sha256: str
    candidate_joint_route_sha256: str
    aggregate_collision_input_sha256: str
    self_collision_execution_policy_sha256: str
    object_world_pose_sha256: str
    hand_contract_sha256: str
    object_id: str
    policy_sha256: str
    v9_parameter_key_hex: str
    collision_link_count: int
    self_pair_count: int
    structural_interface_pair_count: int
    route_checked_self_pair_count: int
    environment_obstacle_count: int
    expected_approach_segment_count: int
    expected_closure_segment_count: int
    expected_lift_segment_count: int
    evaluated_approach_segment_count: int
    evaluated_closure_segment_count: int
    evaluated_lift_segment_count: int
    processed_joint_box_count: int
    certified_free_joint_box_count: int
    maximum_subdivision_boxes_per_route_segment: int
    pad_root_binding_method_id: str
    possible_earliest_pad_root_count: int
    authorized_full_pad_root_count: int
    possible_earliest_pad_roots_bound_to_authorized_full_pad: bool
    robot_object_coverage_complete: bool
    robot_environment_coverage_complete: bool
    robot_self_coverage_complete: bool
    object_table_release_coverage_complete: bool
    complete_route_collision_coverage: bool
    state: CandidateRouteCollisionState
    first_failure: RouteCollisionFailure | None
    blockers: tuple[str, ...]
    display_approximation_used: bool
    finite_sampling_used_as_proof: bool
    legacy_waypoints_used: bool
    online_truth_used: bool
    isaac_or_physx_state_used: bool
    hardware_state_used: bool
    controller_execution_authorized: bool
    dynamic_launch_allowed: bool
    claim_limitations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        digest_fields = (
            self.route_collision_config_sha256,
            self.candidate_route_state_sha256,
            self.candidate_joint_route_sha256,
            self.aggregate_collision_input_sha256,
            self.self_collision_execution_policy_sha256,
            self.object_world_pose_sha256,
            self.hand_contract_sha256,
            self.policy_sha256,
            self.certificate_sha256,
        )
        counts = (
            self.collision_link_count,
            self.self_pair_count,
            self.structural_interface_pair_count,
            self.route_checked_self_pair_count,
            self.environment_obstacle_count,
            self.expected_approach_segment_count,
            self.expected_closure_segment_count,
            self.expected_lift_segment_count,
            self.evaluated_approach_segment_count,
            self.evaluated_closure_segment_count,
            self.evaluated_lift_segment_count,
            self.processed_joint_box_count,
            self.certified_free_joint_box_count,
            self.maximum_subdivision_boxes_per_route_segment,
            self.possible_earliest_pad_root_count,
            self.authorized_full_pad_root_count,
        )
        complete_counts = (
            self.evaluated_approach_segment_count
            == self.expected_approach_segment_count
            and self.evaluated_closure_segment_count
            == self.expected_closure_segment_count
            and self.evaluated_lift_segment_count
            == self.expected_lift_segment_count
        )
        complete_flags = (
            self.possible_earliest_pad_roots_bound_to_authorized_full_pad
            and self.robot_object_coverage_complete
            and self.robot_environment_coverage_complete
            and self.robot_self_coverage_complete
            and self.object_table_release_coverage_complete
            and complete_counts
        )
        if (
            self.method_id != METHOD_ID
            or self.pad_root_binding_method_id != PAD_ROOT_BINDING_METHOD_ID
            or self.route_collision_contract_id != EXPECTED_CONTRACT_ID
            or self.claim_scope != EXPECTED_CLAIM_SCOPE
            or any(not _is_sha256(value) for value in digest_fields)
            or not self.object_id
            or len(self.v9_parameter_key_hex) != 80
            or any(
                character not in "0123456789abcdef"
                for character in self.v9_parameter_key_hex
            )
            or any(type(value) is not int or value < 0 for value in counts)
            or self.collision_link_count != 17
            or self.self_pair_count != EXPECTED_BASE_PAIR_COUNT
            or self.structural_interface_pair_count
            != EXPECTED_STRUCTURAL_INTERFACE_PAIR_COUNT
            or self.route_checked_self_pair_count
            != EXPECTED_FORBIDDEN_PAIR_COUNT
            or self.self_pair_count
            != self.structural_interface_pair_count
            + self.route_checked_self_pair_count
            or self.environment_obstacle_count != 2
            or self.expected_closure_segment_count != 3
            or self.maximum_subdivision_boxes_per_route_segment <= 0
            or self.certified_free_joint_box_count
            > self.processed_joint_box_count
            or self.authorized_full_pad_root_count
            > self.possible_earliest_pad_root_count
            or self.blockers != tuple(sorted(set(self.blockers)))
            or self.claim_limitations != CLAIM_LIMITATIONS
            or any(
                value is not False
                for value in (
                    self.display_approximation_used,
                    self.finite_sampling_used_as_proof,
                    self.legacy_waypoints_used,
                    self.online_truth_used,
                    self.isaac_or_physx_state_used,
                    self.hardware_state_used,
                    self.controller_execution_authorized,
                    self.dynamic_launch_allowed,
                )
            )
            or self.complete_route_collision_coverage is not complete_flags
        ):
            raise ValueError("candidate route-collision certificate is malformed")
        if self.state is CandidateRouteCollisionState.CERTIFIED_FREE:
            if (
                not complete_flags
                or self.first_failure is not None
                or self.blockers
            ):
                raise ValueError("free route-collision state lacks full evidence")
        elif (
            self.state is not CandidateRouteCollisionState.NOT_CERTIFIABLE
            or not self.blockers
        ):
            raise ValueError("unresolved route-collision state lacks blockers")
        if self.first_failure is not None and not isinstance(
            self.first_failure, RouteCollisionFailure
        ):
            raise ValueError("route-collision first failure has wrong type")
        if self.certificate_sha256 != _certificate_sha256(self):
            raise ValueError("route-collision certificate digest changed")

    @property
    def audit(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "method_id": self.method_id,
                "object_id": self.object_id,
                "state": self.state.value,
                "expected_segment_counts": {
                    "approach": self.expected_approach_segment_count,
                    "closure": self.expected_closure_segment_count,
                    "lift": self.expected_lift_segment_count,
                },
                "evaluated_segment_counts": {
                    "approach": self.evaluated_approach_segment_count,
                    "closure": self.evaluated_closure_segment_count,
                    "lift": self.evaluated_lift_segment_count,
                },
                "processed_joint_box_count": self.processed_joint_box_count,
                "self_collision_pair_counts": {
                    "authoritative": self.self_pair_count,
                    "structural_interfaces": (
                        self.structural_interface_pair_count
                    ),
                    "route_checked": self.route_checked_self_pair_count,
                },
                "pad_root_binding_method_id": self.pad_root_binding_method_id,
                "authorized_full_pad_root_count": (
                    self.authorized_full_pad_root_count
                ),
                "possible_earliest_pad_root_count": (
                    self.possible_earliest_pad_root_count
                ),
                "complete_route_collision_coverage": (
                    self.complete_route_collision_coverage
                ),
                "dynamic_launch_allowed": False,
                "blockers": list(self.blockers),
                "certificate_sha256": self.certificate_sha256,
            }
        )


@dataclass(frozen=True)
class _Settings:
    config_sha256: str
    bvh_leaf_triangle_count: int
    maximum_subdivision_boxes_per_route_segment: int


@dataclass(frozen=True)
class _BVHNode:
    lower_m: np.ndarray
    upper_m: np.ndarray
    face_indices: tuple[int, ...]
    subtree_face_count: int
    left: "_BVHNode | None" = None
    right: "_BVHNode | None" = None

    @property
    def leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass(frozen=True)
class _IntervalSurface:
    name: str
    lower_m: np.ndarray
    upper_m: np.ndarray
    nominal_m: np.ndarray
    tree: _BVHNode


@dataclass(frozen=True)
class _BoxCheckResult:
    free: bool
    failure: RouteCollisionFailure | None
    robot_object_checked: bool
    robot_environment_checked: bool
    robot_self_checked: bool
    object_table_release_checked: bool


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
            raise CandidateRouteCollisionError(
                "DUPLICATE_YAML_KEY", repr(key)
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    missing = sorted(set(expected).difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise CandidateRouteCollisionError(
            "SCHEMA_MISMATCH",
            f"{label} missing={missing}, extra={extra}",
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CandidateRouteCollisionError(
            "MAPPING_REQUIRED", f"{label} must be a string-keyed mapping"
        )
    return value


def _load_settings(path: Path) -> _Settings:
    try:
        document = _mapping(
            yaml.load(
                path.read_text(encoding="utf-8"),
                Loader=_UniqueKeySafeLoader,
            ),
            "route collision config",
        )
    except yaml.YAMLError as error:
        raise CandidateRouteCollisionError(
            "INVALID_YAML", str(error)
        ) from error
    _exact_keys(
        document,
        (
            "schema_version",
            "route_collision_contract_id",
            "claim_scope",
            "required_coverage",
            "interval_collision",
            "allowed_contact",
            "truth_firewall",
        ),
        "route collision config",
    )
    coverage = _mapping(document["required_coverage"], "required_coverage")
    interval = _mapping(document["interval_collision"], "interval_collision")
    allowed = _mapping(document["allowed_contact"], "allowed_contact")
    firewall = _mapping(document["truth_firewall"], "truth_firewall")
    _exact_keys(
        coverage,
        (
            "stage_order",
            "collision_link_count",
            "authoritative_self_pair_count",
            "structural_interface_pair_count",
            "forbidden_route_check_self_pair_count",
            "structural_interface_policy_method_id",
            "structural_interface_rule",
            "terminal_allowed_pad_count",
            "environment_obstacle_count",
            "table_release_rule",
        ),
        "required_coverage",
    )
    _exact_keys(
        interval,
        (
            "method",
            "bvh_leaf_triangle_count",
            "maximum_subdivision_boxes_per_route_segment",
            "split_rule",
            "unresolved_policy",
        ),
        "interval_collision",
    )
    _exact_keys(
        allowed,
        (
            "policy",
            "possible_earliest_contact_roots_must_bind_to_full_authorized_pad",
            "closed_shell_forbidden_subset_must_remain_collision_free",
            "pad_contact_does_not_exempt_robot_self_or_environment_collision",
        ),
        "allowed_contact",
    )
    _exact_keys(
        firewall,
        (
            "display_approximation_allowed",
            "finite_joint_or_contact_sampling_as_proof_allowed",
            "legacy_waypoints_allowed",
            "online_object_ground_truth_allowed",
            "online_contact_or_collider_truth_allowed",
            "isaac_or_physx_state_allowed",
            "hardware_state_allowed",
            "controller_execution_authorized",
        ),
        "truth_firewall",
    )
    leaf_count = interval["bvh_leaf_triangle_count"]
    maximum_boxes = interval["maximum_subdivision_boxes_per_route_segment"]
    if (
        document["schema_version"] != EXPECTED_SCHEMA_VERSION
        or document["route_collision_contract_id"] != EXPECTED_CONTRACT_ID
        or document["claim_scope"] != EXPECTED_CLAIM_SCOPE
        or tuple(coverage["stage_order"]) != EXPECTED_STAGE_ORDER
        or coverage["collision_link_count"] != 17
        or coverage["authoritative_self_pair_count"]
        != EXPECTED_BASE_PAIR_COUNT
        or coverage["structural_interface_pair_count"]
        != EXPECTED_STRUCTURAL_INTERFACE_PAIR_COUNT
        or coverage["forbidden_route_check_self_pair_count"]
        != EXPECTED_FORBIDDEN_PAIR_COUNT
        or coverage["structural_interface_policy_method_id"]
        != SELF_COLLISION_EXECUTION_POLICY_METHOD_ID
        or coverage["structural_interface_rule"]
        != STRUCTURAL_INTERFACE_RULE
        or coverage["terminal_allowed_pad_count"] != 3
        or coverage["environment_obstacle_count"] != 2
        or coverage["table_release_rule"] != EXPECTED_TABLE_RELEASE_RULE
        or interval["method"] != EXPECTED_INTERVAL_METHOD
        or interval["split_rule"] != EXPECTED_SPLIT_RULE
        or interval["unresolved_policy"]
        != "FAIL_CLOSED_WITH_STAGE_SEGMENT_AND_PAIR"
        or type(leaf_count) is not int
        or leaf_count <= 0
        or type(maximum_boxes) is not int
        or maximum_boxes <= 0
        or allowed["policy"]
        != (
            "FULL_AUTHORIZED_PAD_ROOTS_WITH_INDEPENDENT_CLOSED_SHELL_"
            "FACE_ROLES_AT_REGISTERED_CONTACT_STAGES"
        )
        or any(
            allowed[name] is not True
            for name in (
                "possible_earliest_contact_roots_must_bind_to_full_authorized_pad",
                "closed_shell_forbidden_subset_must_remain_collision_free",
                "pad_contact_does_not_exempt_robot_self_or_environment_collision",
            )
        )
        or any(value is not False for value in firewall.values())
    ):
        raise CandidateRouteCollisionError(
            "CONFIG_POLICY_MISMATCH",
            "route collision coverage or truth firewall changed",
        )
    return _Settings(
        config_sha256=_file_sha256(path),
        bvh_leaf_triangle_count=leaf_count,
        maximum_subdivision_boxes_per_route_segment=maximum_boxes,
    )


def _failure_document(
    value: RouteCollisionFailure | None,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    return {
        "stage_name": value.stage_name,
        "segment_index": value.segment_index,
        "first_body": value.first_body,
        "second_body": value.second_body,
        "reason": value.reason,
    }


def _certificate_document(source: object) -> dict[str, object]:
    fields = CandidateRouteCollisionCertificate.__dataclass_fields__
    result: dict[str, object] = {}
    for name in fields:
        if name == "certificate_sha256":
            continue
        value = source[name] if isinstance(source, Mapping) else getattr(source, name)
        if isinstance(value, Enum):
            result[name] = value.value
        elif name == "first_failure":
            result[name] = _failure_document(value)
        elif isinstance(value, tuple):
            result[name] = list(value)
        else:
            result[name] = value
    return result


def _certificate_sha256(source: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _certificate_document(source),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _expanded_exact_surface(name: str, triangles: np.ndarray, leaf: int) -> _IntervalSurface:
    nominal = np.asarray(triangles, dtype=np.float64)
    lower = np.nextafter(nominal, -np.inf)
    upper = np.nextafter(nominal, np.inf)
    return _make_surface(name, lower, upper, nominal, leaf)


def _transform_exact_surface(
    name: str,
    triangles: np.ndarray,
    transform: np.ndarray,
    leaf: int,
) -> _IntervalSurface:
    matrix = np.asarray(transform, dtype=np.float64)
    source = np.asarray(triangles, dtype=np.float64)
    nominal = source @ matrix[:3, :3].T + matrix[:3, 3]
    lower = np.empty_like(nominal)
    upper = np.empty_like(nominal)
    for row in range(3):
        lower[..., row] = np.nextafter(matrix[row, 3], -np.inf)
        upper[..., row] = np.nextafter(matrix[row, 3], np.inf)
        for column in range(3):
            product_value = matrix[row, column] * source[..., column]
            product_lower = np.nextafter(product_value, -np.inf)
            product_upper = np.nextafter(product_value, np.inf)
            lower[..., row] = np.nextafter(
                lower[..., row] + product_lower, -np.inf
            )
            upper[..., row] = np.nextafter(
                upper[..., row] + product_upper, np.inf
            )
    return _make_surface(name, lower, upper, nominal, leaf)


def _transform_interval_surface(
    name: str,
    triangles: np.ndarray,
    transform: IntervalRigidTransform,
    nominal_transform: np.ndarray,
    leaf: int,
) -> _IntervalSurface:
    if transform.method_id != INTERVAL_RIGID_TRANSFORM_METHOD_ID:
        raise CandidateRouteCollisionError(
            "INTERVAL_TRANSFORM_METHOD_MISMATCH", transform.method_id
        )
    source = np.asarray(triangles, dtype=np.float64)
    nominal = source @ nominal_transform[:3, :3].T + nominal_transform[:3, 3]
    lower = np.empty_like(nominal)
    upper = np.empty_like(nominal)
    for row in range(3):
        translation = transform.elements[row][3]
        lower[..., row] = translation.lower
        upper[..., row] = translation.upper
        for column in range(3):
            rotation = transform.elements[row][column]
            coordinate = source[..., column]
            first = rotation.lower * coordinate
            second = rotation.upper * coordinate
            product_lower = np.nextafter(np.minimum(first, second), -np.inf)
            product_upper = np.nextafter(np.maximum(first, second), np.inf)
            lower[..., row] = np.nextafter(
                lower[..., row] + product_lower, -np.inf
            )
            upper[..., row] = np.nextafter(
                upper[..., row] + product_upper, np.inf
            )
    return _make_surface(name, lower, upper, nominal, leaf)


def _translated_surface(
    source: _IntervalSurface,
    name: str,
    translation_lower_m: np.ndarray,
    translation_upper_m: np.ndarray,
    translation_nominal_m: np.ndarray,
    leaf: int,
) -> _IntervalSurface:
    lower = np.nextafter(
        source.lower_m + translation_lower_m.reshape(1, 1, 3), -np.inf
    )
    upper = np.nextafter(
        source.upper_m + translation_upper_m.reshape(1, 1, 3), np.inf
    )
    nominal = source.nominal_m + translation_nominal_m.reshape(1, 1, 3)
    return _make_surface(name, lower, upper, nominal, leaf)


def _subset_surface(
    source: _IntervalSurface,
    name: str,
    indices: Sequence[int],
    leaf: int,
) -> _IntervalSurface:
    index = np.asarray(tuple(indices), dtype=np.int64)
    return _make_surface(
        name,
        source.lower_m[index],
        source.upper_m[index],
        source.nominal_m[index],
        leaf,
    )


def _make_surface(
    name: str,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: np.ndarray,
    leaf: int,
) -> _IntervalSurface:
    lower_value = np.asarray(lower, dtype=np.float64)
    upper_value = np.asarray(upper, dtype=np.float64)
    nominal_value = np.asarray(nominal, dtype=np.float64)
    if (
        not name
        or lower_value.ndim != 3
        or lower_value.shape[1:] != (3, 3)
        or len(lower_value) == 0
        or upper_value.shape != lower_value.shape
        or nominal_value.shape != lower_value.shape
        or not np.all(np.isfinite(lower_value))
        or not np.all(np.isfinite(upper_value))
        or not np.all(np.isfinite(nominal_value))
        or np.any(lower_value > upper_value)
    ):
        raise CandidateRouteCollisionError(
            "MALFORMED_INTERVAL_SURFACE", name
        )
    face_lower = np.min(lower_value, axis=1)
    face_upper = np.max(upper_value, axis=1)
    centroids = np.mean(nominal_value, axis=1)
    tree = _build_bvh(
        face_lower,
        face_upper,
        centroids,
        tuple(range(len(lower_value))),
        leaf,
    )
    return _IntervalSurface(
        name=str(name),
        lower_m=lower_value,
        upper_m=upper_value,
        nominal_m=nominal_value,
        tree=tree,
    )


def _build_bvh(
    face_lower: np.ndarray,
    face_upper: np.ndarray,
    centroids: np.ndarray,
    indices: tuple[int, ...],
    leaf_count: int,
) -> _BVHNode:
    index = np.asarray(indices, dtype=np.int64)
    lower = np.min(face_lower[index], axis=0)
    upper = np.max(face_upper[index], axis=0)
    if len(indices) <= leaf_count:
        return _BVHNode(lower, upper, indices, len(indices))
    spread = np.max(centroids[index], axis=0) - np.min(
        centroids[index], axis=0
    )
    axis = int(np.argmax(spread))
    ordered = tuple(
        sorted(indices, key=lambda row: (float(centroids[row, axis]), row))
    )
    middle = len(ordered) // 2
    left = _build_bvh(
        face_lower, face_upper, centroids, ordered[:middle], leaf_count
    )
    right = _build_bvh(
        face_lower, face_upper, centroids, ordered[middle:], leaf_count
    )
    return _BVHNode(lower, upper, (), len(indices), left, right)


def _strict_aabb_separation(
    first_lower: np.ndarray,
    first_upper: np.ndarray,
    second_lower: np.ndarray,
    second_upper: np.ndarray,
) -> bool:
    return bool(
        np.any(first_upper < second_lower)
        or np.any(second_upper < first_lower)
    )


def _candidate_axes(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, ...]:
    first_edges = (
        first[1] - first[0],
        first[2] - first[1],
        first[0] - first[2],
    )
    second_edges = (
        second[1] - second[0],
        second[2] - second[1],
        second[0] - second[2],
    )
    first_normal = np.cross(first_edges[0], first_edges[1])
    second_normal = np.cross(second_edges[0], second_edges[1])
    raw = [
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        first_normal,
        second_normal,
        np.mean(second, axis=0) - np.mean(first, axis=0),
    ]
    raw.extend(
        np.cross(first_edge, second_edge)
        for first_edge in first_edges
        for second_edge in second_edges
    )
    raw.extend(np.cross(first_normal, edge) for edge in first_edges)
    raw.extend(np.cross(second_normal, edge) for edge in second_edges)
    result: list[np.ndarray] = []
    seen: set[bytes] = set()
    for axis in raw:
        norm = float(np.linalg.norm(axis))
        if not math.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            continue
        unit = np.asarray(axis / norm, dtype=np.float64)
        first_nonzero = next(
            (value for value in unit if value != 0.0), 1.0
        )
        if first_nonzero < 0.0:
            unit = -unit
        key = np.asarray(unit, dtype="<f8").tobytes()
        if key not in seen:
            seen.add(key)
            result.append(unit)
    return tuple(result)


def _projection_interval(
    lower: np.ndarray, upper: np.ndarray, axis: np.ndarray
) -> tuple[float, float]:
    vertex_lowers: list[float] = []
    vertex_uppers: list[float] = []
    for vertex_lower, vertex_upper in zip(lower, upper):
        low_terms = np.where(
            axis >= 0.0, axis * vertex_lower, axis * vertex_upper
        )
        high_terms = np.where(
            axis >= 0.0, axis * vertex_upper, axis * vertex_lower
        )
        vertex_lowers.append(
            float(np.nextafter(math.fsum(low_terms), -math.inf))
        )
        vertex_uppers.append(
            float(np.nextafter(math.fsum(high_terms), math.inf))
        )
    return min(vertex_lowers), max(vertex_uppers)


def _triangle_pair_strictly_separated(
    first: _IntervalSurface,
    first_index: int,
    second: _IntervalSurface,
    second_index: int,
) -> bool:
    first_lower = first.lower_m[first_index]
    first_upper = first.upper_m[first_index]
    second_lower = second.lower_m[second_index]
    second_upper = second.upper_m[second_index]
    if _strict_aabb_separation(
        np.min(first_lower, axis=0),
        np.max(first_upper, axis=0),
        np.min(second_lower, axis=0),
        np.max(second_upper, axis=0),
    ):
        return True
    for axis in _candidate_axes(
        first.nominal_m[first_index], second.nominal_m[second_index]
    ):
        first_projection = _projection_interval(
            first_lower, first_upper, axis
        )
        second_projection = _projection_interval(
            second_lower, second_upper, axis
        )
        if (
            first_projection[1] < second_projection[0]
            or second_projection[1] < first_projection[0]
        ):
            return True
    return False


def _surfaces_strictly_separated(
    first: _IntervalSurface, second: _IntervalSurface
) -> tuple[bool, str]:
    pending = [(first.tree, second.tree)]
    while pending:
        first_node, second_node = pending.pop()
        if _strict_aabb_separation(
            first_node.lower_m,
            first_node.upper_m,
            second_node.lower_m,
            second_node.upper_m,
        ):
            continue
        if first_node.leaf and second_node.leaf:
            for first_index in first_node.face_indices:
                for second_index in second_node.face_indices:
                    if not _triangle_pair_strictly_separated(
                        first, first_index, second, second_index
                    ):
                        return (
                            False,
                            "NO_STRICT_FIXED_AXIS_TRIANGLE_SEPARATION:"
                            f"{first_index}:{second_index}",
                        )
            continue
        if second_node.leaf or (
            not first_node.leaf
            and first_node.subtree_face_count
            >= second_node.subtree_face_count
        ):
            if first_node.left is None or first_node.right is None:
                raise CandidateRouteCollisionError(
                    "MALFORMED_BVH", first.name
                )
            pending.append((first_node.right, second_node))
            pending.append((first_node.left, second_node))
        else:
            if second_node.left is None or second_node.right is None:
                raise CandidateRouteCollisionError(
                    "MALFORMED_BVH", second.name
                )
            pending.append((first_node, second_node.right))
            pending.append((first_node, second_node.left))
    return True, "NONE"


def _joint_box_from_rows(
    first: Sequence[IntervalBounds], second: Sequence[IntervalBounds]
) -> tuple[IntervalBounds, ...]:
    if len(first) != len(second):
        raise CandidateRouteCollisionError(
            "JOINT_BOX_ROW_MISMATCH", "route endpoint widths differ"
        )
    return tuple(
        IntervalBounds(
            min(left.lower, right.lower), max(left.upper, right.upper)
        )
        for left, right in zip(first, second)
    )


def _point_row_box(row: Sequence[float]) -> tuple[IntervalBounds, ...]:
    return tuple(IntervalBounds(float(value), float(value)) for value in row)


def _split_joint_box(
    box: tuple[IntervalBounds, ...],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> tuple[tuple[IntervalBounds, ...], tuple[IntervalBounds, ...]] | None:
    normalized = tuple(
        (bounds.upper - bounds.lower) / (upper - lower)
        for bounds, lower, upper in zip(box, lower_limits, upper_limits)
    )
    index = int(np.argmax(np.asarray(normalized, dtype=np.float64)))
    selected = box[index]
    midpoint = selected.lower + 0.5 * (selected.upper - selected.lower)
    if not selected.lower < midpoint < selected.upper:
        return None
    left = list(box)
    right = list(box)
    left[index] = IntervalBounds(selected.lower, midpoint)
    right[index] = IntervalBounds(midpoint, selected.upper)
    return tuple(left), tuple(right)


def _pad_root_binding(
    accepted: StaticV9AcceptedPolicy,
    aggregate: AggregateCollisionRuntimeInputCertificate,
    hand_contract: CARTSHandContract,
) -> tuple[int, int, tuple[str, ...]]:
    policy = accepted.sequential_closure_policy
    role_by_pad = {row.pad_name: row for row in aggregate.terminal_roles}
    pad_by_name = {row.name: row for row in hand_contract.pads}
    possible_count = 0
    authorized_count = 0
    blockers: list[str] = []
    for pad_name, contact_set in zip(
        policy.pad_order, policy.possible_first_contact_sets
    ):
        role = role_by_pad.get(pad_name)
        pad = pad_by_name.get(pad_name)
        if role is None or pad is None:
            blockers.append(f"PAD_ROLE_OR_SOURCE_MISSING:{pad_name}")
            continue
        if role.terminal_certificate.pad_mesh_sha256 != pad.mesh.sha256:
            blockers.append(f"PAD_SOURCE_BINDING_MISMATCH:{pad_name}")
            continue
        pad_triangles = pad.points_local_m[pad.faces]
        roots = tuple(contact_set.possible_earliest_roots)
        possible_count += len(roots)
        for root in roots:
            index = root.pad_triangle_index
            if index < 0 or index >= len(pad_triangles):
                blockers.append(
                    f"PAD_ROOT_INDEX_OUTSIDE_SOURCE:{pad_name}:{index}"
                )
                continue
            if root.semantic_classification != (
                "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
            ):
                blockers.append(
                    "POSSIBLE_EARLIEST_PAD_ROOT_NOT_AUTHORIZED_CONTACT:"
                    f"{pad_name}:{index}"
                )
                continue
            authorized_count += 1
    if possible_count == 0:
        blockers.append("NO_POSSIBLE_EARLIEST_PAD_CONTACT_ROOT")
    return possible_count, authorized_count, tuple(sorted(set(blockers)))


def _check_table_release(
    object_surface: _IntervalSurface,
    table_surface: _IntervalSurface,
) -> tuple[bool, str]:
    object_lower_z = float(np.min(object_surface.lower_m[..., 2]))
    table_upper_z = float(np.max(table_surface.upper_m[..., 2]))
    tolerance = 256.0 * np.finfo(np.float64).eps * max(
        1.0, abs(object_lower_z), abs(table_upper_z)
    )
    if object_lower_z + tolerance >= table_upper_z:
        return True, "MONOTONE_POSITIVE_Z_SUPPORT_RELEASE"
    return False, "OBJECT_INTERVAL_PENETRATES_BELOW_TABLE_TOP"


def _check_joint_box(
    *,
    backend: DirectedIntervalKinematics,
    aggregate: AggregateCollisionRuntimeInputCertificate,
    box: tuple[IntervalBounds, ...],
    stage_name: str,
    segment_index: int,
    contact_stage: bool,
    lift_stage: bool,
    object_surface: _IntervalSurface,
    environment_surfaces: Mapping[str, _IntervalSurface],
    self_collision_pairs: Sequence[tuple[str, str]],
    leaf_count: int,
) -> _BoxCheckResult:
    midpoint = tuple(
        bounds.lower + 0.5 * (bounds.upper - bounds.lower)
        for bounds in box
    )
    model = aggregate.kinematic_binding.model
    nominal_transforms = model.forward_kinematics(midpoint)
    full_surfaces: dict[str, _IntervalSurface] = {}
    for source in aggregate.link_surfaces:
        try:
            transform = backend.link_transform_over_joint_box(
                link_name=source.link_name,
                independent_joint_intervals=box,
                base_transform=np.eye(4),
            )
        except IntervalKinematicsError as error:
            failure = RouteCollisionFailure(
                stage_name,
                segment_index,
                source.link_name,
                "world",
                "INTERVAL_KINEMATICS_UNRESOLVED:"
                + type(error).__name__,
            )
            return _BoxCheckResult(False, failure, False, False, False, False)
        full_surfaces[source.link_name] = _transform_interval_surface(
            source.link_name,
            source.triangles_link_m,
            transform,
            nominal_transforms[source.link_name],
            leaf_count,
        )

    terminal_by_name = {
        row.link_name: row for row in aggregate.terminal_roles
    }
    object_domains: dict[str, _IntervalSurface] = {}
    for name, surface in full_surfaces.items():
        terminal = terminal_by_name.get(name)
        if contact_stage and terminal is not None:
            object_domains[name] = _subset_surface(
                surface,
                name + ":EXACT_NONPAD_FORBIDDEN",
                terminal.terminal_certificate.forbidden_collision_face_indices,
                leaf_count,
            )
        else:
            object_domains[name] = surface

    for name in aggregate.kinematic_binding.collision_link_names:
        free, reason = _surfaces_strictly_separated(
            object_domains[name], object_surface
        )
        if not free:
            return _BoxCheckResult(
                False,
                RouteCollisionFailure(
                    stage_name, segment_index, name, aggregate.object_id, reason
                ),
                True,
                False,
                False,
                False,
            )

    for name in aggregate.kinematic_binding.collision_link_names:
        for obstacle_name, obstacle in environment_surfaces.items():
            free, reason = _surfaces_strictly_separated(
                full_surfaces[name], obstacle
            )
            if not free:
                return _BoxCheckResult(
                    False,
                    RouteCollisionFailure(
                        stage_name, segment_index, name, obstacle_name, reason
                    ),
                    True,
                    True,
                    False,
                    False,
                )

    for first_name, second_name in self_collision_pairs:
        free, reason = _surfaces_strictly_separated(
            full_surfaces[first_name], full_surfaces[second_name]
        )
        if not free:
            return _BoxCheckResult(
                False,
                RouteCollisionFailure(
                    stage_name,
                    segment_index,
                    first_name,
                    second_name,
                    reason,
                ),
                True,
                True,
                True,
                False,
            )

    table_checked = not lift_stage
    if lift_stage:
        table = environment_surfaces["table"]
        table_checked, reason = _check_table_release(object_surface, table)
        if not table_checked:
            return _BoxCheckResult(
                False,
                RouteCollisionFailure(
                    stage_name,
                    segment_index,
                    aggregate.object_id,
                    "table",
                    reason,
                ),
                True,
                True,
                True,
                True,
            )
        fixture = environment_surfaces["fixture"]
        free, reason = _surfaces_strictly_separated(object_surface, fixture)
        if not free:
            return _BoxCheckResult(
                False,
                RouteCollisionFailure(
                    stage_name,
                    segment_index,
                    aggregate.object_id,
                    "fixture",
                    reason,
                ),
                True,
                True,
                True,
                True,
            )
    return _BoxCheckResult(True, None, True, True, True, table_checked)


def _certify_segment(
    *,
    backend: DirectedIntervalKinematics,
    aggregate: AggregateCollisionRuntimeInputCertificate,
    initial_box: tuple[IntervalBounds, ...],
    stage_name: str,
    segment_index: int,
    contact_stage: bool,
    lift_stage: bool,
    object_surface_factory: Any,
    environment_surfaces: Mapping[str, _IntervalSurface],
    self_collision_pairs: Sequence[tuple[str, str]],
    settings: _Settings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> tuple[bool, int, int, _BoxCheckResult]:
    pending = [initial_box]
    processed = 0
    free_leaves = 0
    last = _BoxCheckResult(False, None, False, False, False, False)
    while pending:
        box = pending.pop()
        if processed >= settings.maximum_subdivision_boxes_per_route_segment:
            failure = last.failure or RouteCollisionFailure(
                stage_name,
                segment_index,
                "route_joint_box",
                "collision_coverage",
                "SUBDIVISION_BOX_BUDGET_EXHAUSTED",
            )
            return False, processed, free_leaves, _BoxCheckResult(
                False,
                failure,
                last.robot_object_checked,
                last.robot_environment_checked,
                last.robot_self_checked,
                last.object_table_release_checked,
            )
        processed += 1
        object_surface = object_surface_factory(box)
        last = _check_joint_box(
            backend=backend,
            aggregate=aggregate,
            box=box,
            stage_name=stage_name,
            segment_index=segment_index,
            contact_stage=contact_stage,
            lift_stage=lift_stage,
            object_surface=object_surface,
            environment_surfaces=environment_surfaces,
            self_collision_pairs=self_collision_pairs,
            leaf_count=settings.bvh_leaf_triangle_count,
        )
        if last.free:
            free_leaves += 1
            continue
        split = _split_joint_box(box, lower_limits, upper_limits)
        if (
            split is None
            or processed >= settings.maximum_subdivision_boxes_per_route_segment
        ):
            return False, processed, free_leaves, last
        left, right = split
        pending.append(right)
        pending.append(left)
    return True, processed, free_leaves, last


def build_candidate_route_collision_certificate(
    config_path: Path | str,
    *,
    accepted_policy: object,
    route_state: object,
    joint_route: object,
    aggregate_inputs: AggregateCollisionRuntimeInputCertificate,
    object_world_pose: SettledObjectWorldPoseCertificate,
    hand_contract: CARTSHandContract,
    repository_root: Path | str,
) -> CandidateRouteCollisionCertificate:
    """Check one accepted policy's complete route without simulator truth."""

    if type(accepted_policy) is not StaticV9AcceptedPolicy:
        raise CandidateRouteCollisionError(
            "STATIC_ACCEPTED_POLICY_REQUIRED", type(accepted_policy).__name__
        )
    if type(route_state) is not CandidateRouteStateContract:
        raise CandidateRouteCollisionError(
            "CANDIDATE_ROUTE_STATE_REQUIRED", type(route_state).__name__
        )
    if type(joint_route) is not CandidateJointRouteContract:
        raise CandidateRouteCollisionError(
            "CANDIDATE_JOINT_ROUTE_REQUIRED", type(joint_route).__name__
        )
    if type(aggregate_inputs) is not AggregateCollisionRuntimeInputCertificate:
        raise CandidateRouteCollisionError(
            "AGGREGATE_COLLISION_INPUT_REQUIRED",
            type(aggregate_inputs).__name__,
        )
    if type(object_world_pose) is not SettledObjectWorldPoseCertificate:
        raise CandidateRouteCollisionError(
            "SETTLED_OBJECT_POSE_REQUIRED", type(object_world_pose).__name__
        )
    if type(hand_contract) is not CARTSHandContract:
        raise CandidateRouteCollisionError(
            "CARTS_HAND_CONTRACT_REQUIRED", type(hand_contract).__name__
        )
    root = Path(repository_root).resolve()
    raw_path = Path(config_path)
    path = (raw_path if raw_path.is_absolute() else root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CandidateRouteCollisionError(
            "CONFIG_OUTSIDE_REPOSITORY", str(path)
        ) from error
    if not path.is_file():
        raise CandidateRouteCollisionError("CONFIG_MISSING", str(path))
    settings = _load_settings(path)
    policy = accepted_policy.sequential_closure_policy
    if type(policy) is not CertifiedSequentialClosurePolicy:
        raise CandidateRouteCollisionError(
            "CERTIFIED_SEQUENTIAL_POLICY_REQUIRED", type(policy).__name__
        )
    hand_sha = _file_sha256(hand_contract.contract_path)
    if (
        route_state.certificate_sha256
        != joint_route.candidate_route_state_sha256
        or route_state.aggregate_collision_input_sha256
        != aggregate_inputs.certificate_sha256
        or joint_route.aggregate_collision_input_sha256
        != aggregate_inputs.certificate_sha256
        or route_state.object_world_pose_sha256
        != object_world_pose.certificate_sha256
        or route_state.object_id != aggregate_inputs.object_id
        or route_state.object_id != object_world_pose.object_id
        or joint_route.object_id != route_state.object_id
        or route_state.policy_sha256 != policy.policy_sha256
        or joint_route.policy_sha256 != policy.policy_sha256
        or route_state.v9_parameter_key_hex
        != accepted_policy.v9_parameter_key_hex
        or joint_route.v9_parameter_key_hex
        != accepted_policy.v9_parameter_key_hex
        or any(
            row.terminal_certificate.hand_contract_sha256 != hand_sha
            for row in aggregate_inputs.terminal_roles
        )
    ):
        raise CandidateRouteCollisionError(
            "CROSS_ROUTE_OBJECT_MODEL_BINDING_MISMATCH",
            "policy, route, robot, object pose, or hand bytes differ",
        )

    self_collision_policy = build_self_collision_execution_policy(
        kinematic_binding=aggregate_inputs.kinematic_binding,
        base_inventory=aggregate_inputs.self_pair_inventory,
    )

    possible_roots, authorized_roots, pad_blockers = _pad_root_binding(
        accepted_policy, aggregate_inputs, hand_contract
    )
    pad_complete = (
        possible_roots > 0 and possible_roots == authorized_roots
    )
    blockers = list(pad_blockers)
    first_failure: RouteCollisionFailure | None = None
    processed_boxes = 0
    free_boxes = 0
    evaluated_approach = 0
    evaluated_closure = 0
    evaluated_lift = 0
    robot_object_complete = True
    robot_environment_complete = True
    robot_self_complete = True
    object_table_complete = True

    leaf = settings.bvh_leaf_triangle_count
    initial_object = _transform_exact_surface(
        aggregate_inputs.object_id,
        aggregate_inputs.object_surface.triangles_object_m,
        object_world_pose.world_from_object,
        leaf,
    )
    environment = {
        obstacle.name: _expanded_exact_surface(
            obstacle.name, obstacle.triangles_world_m, leaf
        )
        for obstacle in aggregate_inputs.shared_environment.obstacles
    }
    lower_limits = joint_route.complete_joint_lower_limits_rad
    upper_limits = joint_route.complete_joint_upper_limits_rad
    backend = aggregate_inputs.kinematic_binding.new_interval_backend()

    def initial_object_factory(_box: object) -> _IntervalSurface:
        return initial_object

    approach = joint_route.approach_independent_joint_waypoints_rad
    stopped = False
    for index, (first, second) in enumerate(zip(approach, approach[1:])):
        box = _joint_box_from_rows(_point_row_box(first), _point_row_box(second))
        free, processed, leaves, result = _certify_segment(
            backend=backend,
            aggregate=aggregate_inputs,
            initial_box=box,
            stage_name="HOME_TO_PREGRASP",
            segment_index=index,
            contact_stage=False,
            lift_stage=False,
            object_surface_factory=initial_object_factory,
            environment_surfaces=environment,
            self_collision_pairs=self_collision_policy.forbidden_pairs,
            settings=settings,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )
        processed_boxes += processed
        free_boxes += leaves
        if not free:
            first_failure = result.failure
            blockers.append(
                "APPROACH_COLLISION_UNRESOLVED:"
                + ("UNKNOWN" if first_failure is None else first_failure.reason)
            )
            robot_object_complete &= result.robot_object_checked
            robot_environment_complete &= result.robot_environment_checked
            robot_self_complete &= result.robot_self_checked
            stopped = True
            break
        evaluated_approach += 1

    if not stopped:
        closure_rows = joint_route.closure_stage_swept_independent_joint_intervals_rad
        for index, box in enumerate(closure_rows):
            stage_name = joint_route.closure_stage_names[index]
            free, processed, leaves, result = _certify_segment(
                backend=backend,
                aggregate=aggregate_inputs,
                initial_box=tuple(box),
                stage_name=stage_name,
                segment_index=index,
                contact_stage=True,
                lift_stage=False,
                object_surface_factory=initial_object_factory,
                environment_surfaces=environment,
                self_collision_pairs=self_collision_policy.forbidden_pairs,
                settings=settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
            )
            processed_boxes += processed
            free_boxes += leaves
            if not free:
                first_failure = result.failure
                blockers.append(
                    "CLOSURE_COLLISION_UNRESOLVED:"
                    + ("UNKNOWN" if first_failure is None else first_failure.reason)
                )
                robot_object_complete &= result.robot_object_checked
                robot_environment_complete &= result.robot_environment_checked
                robot_self_complete &= result.robot_self_checked
                stopped = True
                break
            evaluated_closure += 1

    if not stopped:
        lift_rows = joint_route.lift_independent_joint_interval_waypoints_rad
        lift_parameters = joint_route.lift_path_parameters
        for index, (first, second) in enumerate(zip(lift_rows, lift_rows[1:])):
            box = _joint_box_from_rows(first, second)
            parameter_lower = lift_parameters[index]
            parameter_upper = lift_parameters[index + 1]

            def lift_object_factory(
                _box: object,
                lower: float = parameter_lower,
                upper: float = parameter_upper,
            ) -> _IntervalSurface:
                lower_translation = np.asarray((0.0, 0.0, 0.04 * lower))
                upper_translation = np.asarray((0.0, 0.0, 0.04 * upper))
                nominal_translation = 0.5 * (
                    lower_translation + upper_translation
                )
                return _translated_surface(
                    initial_object,
                    aggregate_inputs.object_id + ":LIFT",
                    lower_translation,
                    upper_translation,
                    nominal_translation,
                    leaf,
                )

            free, processed, leaves, result = _certify_segment(
                backend=backend,
                aggregate=aggregate_inputs,
                initial_box=box,
                stage_name="LIFT_40_MM",
                segment_index=index,
                contact_stage=True,
                lift_stage=True,
                object_surface_factory=lift_object_factory,
                environment_surfaces=environment,
                self_collision_pairs=self_collision_policy.forbidden_pairs,
                settings=settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
            )
            processed_boxes += processed
            free_boxes += leaves
            if not free:
                first_failure = result.failure
                blockers.append(
                    "LIFT_COLLISION_UNRESOLVED:"
                    + ("UNKNOWN" if first_failure is None else first_failure.reason)
                )
                robot_object_complete &= result.robot_object_checked
                robot_environment_complete &= result.robot_environment_checked
                robot_self_complete &= result.robot_self_checked
                object_table_complete &= result.object_table_release_checked
                stopped = True
                break
            evaluated_lift += 1

    if stopped:
        blockers.append("ROUTE_COLLISION_EVALUATION_STOPPED_AT_FIRST_UNRESOLVED_SEGMENT")
    expected_approach = len(approach) - 1
    expected_closure = len(joint_route.closure_stage_names)
    expected_lift = len(joint_route.lift_path_parameters) - 1
    complete_counts = (
        evaluated_approach == expected_approach
        and evaluated_closure == expected_closure
        and evaluated_lift == expected_lift
    )
    if not complete_counts:
        robot_object_complete = False
        robot_environment_complete = False
        robot_self_complete = False
        object_table_complete = False
    complete = (
        pad_complete
        and complete_counts
        and robot_object_complete
        and robot_environment_complete
        and robot_self_complete
        and object_table_complete
    )
    if not pad_complete:
        blockers.append("AUTHORIZED_FULL_PAD_CONTACT_ROOT_BINDING_INCOMPLETE")
    state = (
        CandidateRouteCollisionState.CERTIFIED_FREE
        if complete
        else CandidateRouteCollisionState.NOT_CERTIFIABLE
    )
    if not complete and not blockers:
        blockers.append("COMPLETE_ROUTE_COLLISION_COVERAGE_INCOMPLETE")
    values: dict[str, object] = {
        "method_id": METHOD_ID,
        "route_collision_contract_id": EXPECTED_CONTRACT_ID,
        "claim_scope": EXPECTED_CLAIM_SCOPE,
        "route_collision_config_sha256": settings.config_sha256,
        "candidate_route_state_sha256": route_state.certificate_sha256,
        "candidate_joint_route_sha256": joint_route.certificate_sha256,
        "aggregate_collision_input_sha256": aggregate_inputs.certificate_sha256,
        "self_collision_execution_policy_sha256": (
            self_collision_policy.certificate_sha256
        ),
        "object_world_pose_sha256": object_world_pose.certificate_sha256,
        "hand_contract_sha256": hand_sha,
        "object_id": aggregate_inputs.object_id,
        "policy_sha256": policy.policy_sha256,
        "v9_parameter_key_hex": accepted_policy.v9_parameter_key_hex,
        "collision_link_count": aggregate_inputs.collision_link_count,
        "self_pair_count": aggregate_inputs.self_pair_count,
        "structural_interface_pair_count": (
            self_collision_policy.structural_interface_pair_count
        ),
        "route_checked_self_pair_count": (
            self_collision_policy.forbidden_pair_count
        ),
        "environment_obstacle_count": aggregate_inputs.shared_environment.obstacle_count,
        "expected_approach_segment_count": expected_approach,
        "expected_closure_segment_count": expected_closure,
        "expected_lift_segment_count": expected_lift,
        "evaluated_approach_segment_count": evaluated_approach,
        "evaluated_closure_segment_count": evaluated_closure,
        "evaluated_lift_segment_count": evaluated_lift,
        "processed_joint_box_count": processed_boxes,
        "certified_free_joint_box_count": free_boxes,
        "maximum_subdivision_boxes_per_route_segment": settings.maximum_subdivision_boxes_per_route_segment,
        "pad_root_binding_method_id": PAD_ROOT_BINDING_METHOD_ID,
        "possible_earliest_pad_root_count": possible_roots,
        "authorized_full_pad_root_count": authorized_roots,
        "possible_earliest_pad_roots_bound_to_authorized_full_pad": pad_complete,
        "robot_object_coverage_complete": robot_object_complete,
        "robot_environment_coverage_complete": robot_environment_complete,
        "robot_self_coverage_complete": robot_self_complete,
        "object_table_release_coverage_complete": object_table_complete,
        "complete_route_collision_coverage": complete,
        "state": state,
        "first_failure": first_failure,
        "blockers": tuple(sorted(set(blockers))),
        "display_approximation_used": False,
        "finite_sampling_used_as_proof": False,
        "legacy_waypoints_used": False,
        "online_truth_used": False,
        "isaac_or_physx_state_used": False,
        "hardware_state_used": False,
        "controller_execution_authorized": False,
        "dynamic_launch_allowed": False,
        "claim_limitations": CLAIM_LIMITATIONS,
    }
    return CandidateRouteCollisionCertificate(
        **values,
        certificate_sha256=_certificate_sha256(values),
    )


__all__ = [
    "CLAIM_LIMITATIONS",
    "METHOD_ID",
    "PAD_ROOT_BINDING_METHOD_ID",
    "CandidateRouteCollisionCertificate",
    "CandidateRouteCollisionError",
    "CandidateRouteCollisionState",
    "RouteCollisionFailure",
    "build_candidate_route_collision_certificate",
]
