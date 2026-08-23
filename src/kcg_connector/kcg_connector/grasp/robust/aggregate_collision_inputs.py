"""Verified aggregate-robot and runtime collision-surface inputs.

The collision roster and material certificates identify the authoritative
files, but a trajectory checker needs two additional deterministic objects:

* one world-to-arm-to-hand kinematic tree;
* link-local triangle arrays carrying the exact terminal PAD/non-PAD roles; and
* the hash-bound table and fixture surfaces shared by both study objects.

This module builds those objects only from already verified bytes.  It also
binds one study object's planning surface and persisted material-boundary
certificate.  It deliberately does not invent a loose-object world pose,
candidate-specific arm path, or lift attachment model.  Those remain explicit
blockers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from kcg_connector.grasp.robust.collision_contract import (
    SelfCollisionPairInventory,
    build_self_collision_pair_inventory,
)
from kcg_connector.grasp.robust.collision_geometry_binding import (
    CollisionGeometryBindingCertificate,
)
from kcg_connector.grasp.robust.collision_roster import (
    AuthoritativeCollisionLinkRoster,
    EXPECTED_AGGREGATE_SOURCE,
    EXPECTED_INCLUDE_SOURCES,
    EXPECTED_INDEPENDENT_JOINTS,
    build_verified_aggregate_robot_xml,
)
from kcg_connector.grasp.robust.full_hand_collision import (
    HashBoundLinkSurface,
    HashBoundObjectSurface,
    triangle_surface_geometry_sha256,
)
from kcg_connector.grasp.robust.hand_contract import CARTSHandContract
from kcg_connector.grasp.robust.hand_model import (
    ThreeFingerHandModel,
    rpy_rotation,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    DirectedIntervalKinematics,
    IntervalArithmeticOptions,
)
from kcg_connector.grasp.robust.object_contract import LoadedObjectContract
from kcg_connector.grasp.robust.object_material_boundary import (
    ObjectMaterialBoundaryEvidence,
)
from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh
from kcg_connector.grasp.robust.shared_environment import (
    SharedTableFixtureWorldCertificate,
)
from kcg_connector.grasp.robust.terminal_collision_boundary import (
    TerminalCollisionBoundaryCertificate,
)


METHOD_ID = "CARTS_AGGREGATE_KINEMATICS_OBJECT_AND_WORLD_COLLISION_INPUTS_V2"
KINEMATIC_ASSEMBLY_POLICY = (
    "HASH_VERIFIED_AGGREGATE_XACRO_INCLUDE_ORDER_X_LINK_AND_JOINT_ELEMENTS_ONLY"
)
REMAINING_BLOCKERS = (
    "CANDIDATE_SPECIFIC_LOOSE_OBJECT_WORLD_POSE_AND_HOME_PREGRASP_CLOSURE_LIFT_TRAJECTORY_UNAVAILABLE",
    "CONTINUOUS_ALLOWED_PAD_CONTACT_PATH_AND_ENDPOINT_UNAVAILABLE",
)
CLAIM_LIMITATIONS = (
    "AGGREGATE_KINEMATIC_OBJECT_AND_STATIC_WORLD_GEOMETRY_INPUT_BINDING_ONLY",
    "OBJECT_MATERIAL_BOUNDARY_IS_STATIC_MESH_LOCAL_EVIDENCE_ONLY",
    "TABLE_AND_FIXTURE_ARE_STATIC_WORLD_FRAME_SURFACES_ONLY",
    "NO_LOOSE_OBJECT_WORLD_POSE_OR_CANDIDATE_SPECIFIC_IK_PATH",
    "NO_CONTINUOUS_ALLOWED_PAD_CONTACT_OR_LIFT_ATTACHMENT_PROOF",
    "NO_ISAAC_HARDWARE_OR_FORMAL_GRASP_CANDIDATE_CLAIM",
)


class AggregateCollisionInputError(ValueError):
    """Fail-closed aggregate-model or runtime-surface input error."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("aggregate collision input error cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _float_hex(value: float) -> str:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AggregateCollisionInputError(
            "NONFINITE_KINEMATIC_VALUE",
            "aggregate kinematic model contains a non-finite value",
        )
    return parsed.hex()


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _aggregate_model_document(
    model: ThreeFingerHandModel,
    *,
    aggregate_source_sha256: str,
    include_source_bindings: tuple[tuple[str, str], ...],
    hand_contract_sha256: str,
) -> dict[str, object]:
    joints: list[object] = []
    for name in model.joint_order:
        joint = model.joints[name]
        joints.append(
            {
                "name": joint.name,
                "joint_type": joint.joint_type,
                "parent_link": joint.parent_link,
                "child_link": joint.child_link,
                "origin_xyz_m": [_float_hex(row) for row in joint.origin_xyz_m],
                "origin_rpy_rad": [_float_hex(row) for row in joint.origin_rpy_rad],
                "axis": [_float_hex(row) for row in joint.axis],
                "limit": (
                    None
                    if joint.limit is None
                    else {
                        "lower": _float_hex(joint.limit.lower),
                        "upper": _float_hex(joint.limit.upper),
                        "effort": (
                            None
                            if joint.limit.effort is None
                            else _float_hex(joint.limit.effort)
                        ),
                        "velocity": (
                            None
                            if joint.limit.velocity is None
                            else _float_hex(joint.limit.velocity)
                        ),
                    }
                ),
                "mimic": (
                    None
                    if joint.mimic is None
                    else {
                        "source_joint": joint.mimic.source_joint,
                        "multiplier": _float_hex(joint.mimic.multiplier),
                        "offset": _float_hex(joint.mimic.offset),
                    }
                ),
            }
        )
    return {
        "method_id": METHOD_ID,
        "kinematic_assembly_policy": KINEMATIC_ASSEMBLY_POLICY,
        "aggregate_source_sha256": aggregate_source_sha256,
        "include_source_bindings": [list(row) for row in include_source_bindings],
        "hand_contract_sha256": hand_contract_sha256,
        "base_link": model.base_link,
        "joint_order": list(model.joint_order),
        "independent_joint_names": list(model.independent_joint_names),
        "joints": joints,
        "pads": [
            {
                "name": name,
                "finger_name": model.pads[name].finger_name,
                "link_name": model.pads[name].link_name,
            }
            for name in sorted(model.pads)
        ],
    }


def _aggregate_model_sha256(
    model: ThreeFingerHandModel,
    *,
    aggregate_source_sha256: str,
    include_source_bindings: tuple[tuple[str, str], ...],
    hand_contract_sha256: str,
) -> str:
    return _canonical_sha256(
        _aggregate_model_document(
            model,
            aggregate_source_sha256=aggregate_source_sha256,
            include_source_bindings=include_source_bindings,
            hand_contract_sha256=hand_contract_sha256,
        )
    )


@dataclass(frozen=True)
class AggregateRobotKinematicBinding:
    method_id: str
    kinematic_assembly_policy: str
    aggregate_source_sha256: str
    include_source_bindings: tuple[tuple[str, str], ...]
    hand_contract_sha256: str
    model: ThreeFingerHandModel
    interval_options: IntervalArithmeticOptions
    collision_link_names: tuple[str, ...]
    independent_joint_names: tuple[str, ...]
    collision_link_count: int
    independent_joint_count: int
    every_collision_link_connected_to_world: bool
    model_sha256: str

    def __post_init__(self) -> None:
        if (
            self.method_id != METHOD_ID
            or self.kinematic_assembly_policy != KINEMATIC_ASSEMBLY_POLICY
            or not _is_sha256(self.aggregate_source_sha256)
            or not _is_sha256(self.hand_contract_sha256)
            or not _is_sha256(self.model_sha256)
            or not isinstance(self.model, ThreeFingerHandModel)
            or not isinstance(self.interval_options, IntervalArithmeticOptions)
            or self.model.base_link != "world"
            or self.include_source_bindings
            != tuple(
                zip(
                    EXPECTED_INCLUDE_SOURCES,
                    (row[1] for row in self.include_source_bindings),
                )
            )
            or any(not _is_sha256(row[1]) for row in self.include_source_bindings)
            or self.independent_joint_names != EXPECTED_INDEPENDENT_JOINTS
            or tuple(self.model.independent_joint_names)
            != self.independent_joint_names
            or self.independent_joint_count != len(EXPECTED_INDEPENDENT_JOINTS)
            or self.collision_link_count != 17
            or len(self.collision_link_names) != self.collision_link_count
            or len(set(self.collision_link_names)) != self.collision_link_count
            or self.every_collision_link_connected_to_world is not True
        ):
            raise ValueError("aggregate robot kinematic binding is incomplete")
        reachable = {self.model.base_link}
        reachable.update(joint.child_link for joint in self.model.joints.values())
        if not set(self.collision_link_names) <= reachable:
            raise ValueError("aggregate model does not connect every collision link")
        expected_digest = _aggregate_model_sha256(
            self.model,
            aggregate_source_sha256=self.aggregate_source_sha256,
            include_source_bindings=self.include_source_bindings,
            hand_contract_sha256=self.hand_contract_sha256,
        )
        if self.model_sha256 != expected_digest:
            raise ValueError("aggregate robot model digest changed")

    def new_interval_backend(self) -> DirectedIntervalKinematics:
        """Return a fresh interval backend over the verified aggregate tree."""

        return DirectedIntervalKinematics(self.model, self.interval_options)


@dataclass(frozen=True)
class TerminalRuntimeCollisionRole:
    link_name: str
    pad_name: str
    terminal_certificate: TerminalCollisionBoundaryCertificate
    full_surface: HashBoundLinkSurface
    allowed_pad_surface: HashBoundLinkSurface
    forbidden_nonpad_surface: HashBoundLinkSurface

    def __post_init__(self) -> None:
        certificate = self.terminal_certificate
        surfaces = (
            self.full_surface,
            self.allowed_pad_surface,
            self.forbidden_nonpad_surface,
        )
        if (
            not isinstance(certificate, TerminalCollisionBoundaryCertificate)
            or any(not isinstance(row, HashBoundLinkSurface) for row in surfaces)
            or self.link_name != certificate.link_name
            or self.pad_name != certificate.pad_name
            or any(row.link_name != self.link_name for row in surfaces)
            or any(
                row.source_asset_sha256 != certificate.collision_mesh_sha256
                for row in surfaces
            )
            or len(self.full_surface.triangles_link_m)
            != certificate.collision_face_count
            or len(self.allowed_pad_surface.triangles_link_m)
            != certificate.exact_shared_collision_face_count
            or len(self.forbidden_nonpad_surface.triangles_link_m)
            != len(certificate.forbidden_collision_face_indices)
        ):
            raise ValueError("terminal runtime collision role is malformed")
        allowed = self.full_surface.triangles_link_m[
            np.asarray(certificate.allowed_collision_face_indices, dtype=np.int64)
        ]
        forbidden = self.full_surface.triangles_link_m[
            np.asarray(certificate.forbidden_collision_face_indices, dtype=np.int64)
        ]
        if not np.array_equal(
            allowed, self.allowed_pad_surface.triangles_link_m
        ) or not np.array_equal(
            forbidden, self.forbidden_nonpad_surface.triangles_link_m
        ):
            raise ValueError("terminal runtime role differs from exact face indices")


@dataclass(frozen=True)
class AggregateCollisionRuntimeInputCertificate:
    method_id: str
    kinematic_binding: AggregateRobotKinematicBinding
    geometry_binding: CollisionGeometryBindingCertificate
    link_surfaces: tuple[HashBoundLinkSurface, ...]
    terminal_roles: tuple[
        TerminalRuntimeCollisionRole,
        TerminalRuntimeCollisionRole,
        TerminalRuntimeCollisionRole,
    ]
    self_pair_inventory: SelfCollisionPairInventory
    object_surface: HashBoundObjectSurface
    object_material_boundary: ObjectMaterialBoundaryEvidence
    shared_environment: SharedTableFixtureWorldCertificate
    object_id: str
    object_contract_source_sha256: str
    collision_link_count: int
    terminal_role_count: int
    self_pair_count: int
    aggregate_robot_kinematics_binding_complete: bool
    runtime_link_surface_binding_complete: bool
    terminal_runtime_role_binding_complete: bool
    object_planning_surface_binding_complete: bool
    object_material_boundary_binding_complete: bool
    candidate_specific_motion_binding_complete: bool
    table_fixture_environment_binding_complete: bool
    continuous_pad_contact_binding_complete: bool
    formal_complete_collision_input_eligible: bool
    remaining_blockers: tuple[str, ...]
    claim_limitations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        links = self.link_surfaces
        terminals = self.terminal_roles
        if (
            self.method_id != METHOD_ID
            or not isinstance(self.kinematic_binding, AggregateRobotKinematicBinding)
            or not isinstance(self.geometry_binding, CollisionGeometryBindingCertificate)
            or not isinstance(self.self_pair_inventory, SelfCollisionPairInventory)
            or not isinstance(self.object_surface, HashBoundObjectSurface)
            or not isinstance(
                self.object_material_boundary, ObjectMaterialBoundaryEvidence
            )
            or not isinstance(
                self.shared_environment, SharedTableFixtureWorldCertificate
            )
            or not _is_sha256(self.object_contract_source_sha256)
            or not _is_sha256(self.certificate_sha256)
            or self.object_id != self.object_surface.object_id
            or self.object_id != self.object_material_boundary.object_id
            or self.object_surface.source_asset_sha256
            != self.object_material_boundary.source_asset_sha256
            or self.object_id not in self.shared_environment.registered_object_ids
            or self.kinematic_binding.model.base_link != "world"
            or self.shared_environment.root_frame != "world"
            or self.shared_environment.robot_base_origin_m != (0.0, 0.0, 0.0)
            or self.shared_environment.obstacle_count != 2
            or len(self.shared_environment.obstacles) != 2
            or sum(
                len(obstacle.triangles_world_m)
                for obstacle in self.shared_environment.obstacles
            )
            != 24
            or self.shared_environment.table_fixture_world_binding_complete is not True
            or self.shared_environment.loose_object_initial_pose_included is not False
            or self.shared_environment.candidate_specific_robot_route_included
            is not False
            or self.shared_environment.isaac_dynamic_state_included is not False
            or self.shared_environment.hardware_state_included is not False
            or self.collision_link_count != 17
            or self.terminal_role_count != 3
            or self.self_pair_count != 136
            or len(links) != self.collision_link_count
            or len(terminals) != self.terminal_role_count
            or tuple(row.link_name for row in links)
            != self.kinematic_binding.collision_link_names
            or tuple(row.link_name for row in terminals)
            != ("f1Link3", "f2Link2", "f3Link3")
            or self.self_pair_inventory.link_names
            != tuple(sorted(row.link_name for row in links))
            or len(self.self_pair_inventory.all_pairs) != self.self_pair_count
        ):
            raise ValueError("aggregate collision runtime input coverage is incomplete")
        material_by_name = {
            row.link_name: row
            for row in self.geometry_binding.collision_link_material_bindings
        }
        surface_by_name = {row.link_name: row for row in links}
        if set(material_by_name) != set(surface_by_name) or any(
            surface_by_name[name].source_asset_sha256
            != material_by_name[name].collision_mesh_sha256
            for name in material_by_name
        ):
            raise ValueError("runtime surfaces diverged from material bindings")
        terminal_by_name = {
            row.link_name: row for row in self.geometry_binding.terminal_role_bindings
        }
        if any(
            row.terminal_certificate is not terminal_by_name[row.link_name]
            for row in terminals
        ):
            raise ValueError("runtime terminal roles diverged from exact certificates")
        if (
            self.aggregate_robot_kinematics_binding_complete is not True
            or self.runtime_link_surface_binding_complete is not True
            or self.terminal_runtime_role_binding_complete is not True
            or self.object_planning_surface_binding_complete is not True
            or self.object_material_boundary_binding_complete is not True
            or self.candidate_specific_motion_binding_complete is not False
            or self.table_fixture_environment_binding_complete is not True
            or self.continuous_pad_contact_binding_complete is not False
            or self.formal_complete_collision_input_eligible is not False
            or self.remaining_blockers != REMAINING_BLOCKERS
            or self.claim_limitations != CLAIM_LIMITATIONS
        ):
            raise ValueError("aggregate collision input claim boundary changed")
        if self.certificate_sha256 != _certificate_digest(self):
            raise ValueError("aggregate collision runtime certificate digest changed")

    @property
    def audit(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "method_id": self.method_id,
                "aggregate_model_sha256": self.kinematic_binding.model_sha256,
                "geometry_binding_certificate_sha256": (
                    self.geometry_binding.certificate_sha256
                ),
                "object_id": self.object_id,
                "object_surface_geometry_sha256": self.object_surface.geometry_sha256,
                "object_material_boundary_certificate_sha256": (
                    self.object_material_boundary.certificate_sha256
                ),
                "shared_environment_certificate_sha256": (
                    self.shared_environment.certificate_sha256
                ),
                "environment_root_frame": self.shared_environment.root_frame,
                "environment_obstacle_count": self.shared_environment.obstacle_count,
                "environment_triangle_count": sum(
                    len(obstacle.triangles_world_m)
                    for obstacle in self.shared_environment.obstacles
                ),
                "collision_link_count": self.collision_link_count,
                "terminal_role_count": self.terminal_role_count,
                "self_pair_count": self.self_pair_count,
                "aggregate_robot_kinematics_binding_complete": True,
                "runtime_link_surface_binding_complete": True,
                "terminal_runtime_role_binding_complete": True,
                "object_planning_surface_binding_complete": True,
                "object_material_boundary_binding_complete": True,
                "candidate_specific_motion_binding_complete": False,
                "table_fixture_environment_binding_complete": True,
                "continuous_pad_contact_binding_complete": False,
                "formal_complete_collision_input_eligible": False,
                "remaining_blockers": list(self.remaining_blockers),
                "claim_limitations": list(self.claim_limitations),
                "certificate_sha256": self.certificate_sha256,
            }
        )


def _certificate_digest(
    certificate: AggregateCollisionRuntimeInputCertificate,
) -> str:
    document = {
        "method_id": METHOD_ID,
        "aggregate_model_sha256": certificate.kinematic_binding.model_sha256,
        "geometry_binding_certificate_sha256": (
            certificate.geometry_binding.certificate_sha256
        ),
        "link_surfaces": [
            [row.link_name, row.source_asset_sha256, row.geometry_sha256]
            for row in certificate.link_surfaces
        ],
        "terminal_roles": [
            {
                "link_name": row.link_name,
                "terminal_certificate_sha256": (
                    row.terminal_certificate.certificate_sha256
                ),
                "full_surface_geometry_sha256": row.full_surface.geometry_sha256,
                "allowed_pad_surface_geometry_sha256": (
                    row.allowed_pad_surface.geometry_sha256
                ),
                "forbidden_nonpad_surface_geometry_sha256": (
                    row.forbidden_nonpad_surface.geometry_sha256
                ),
            }
            for row in certificate.terminal_roles
        ],
        "self_pair_inventory_sha256": (
            certificate.self_pair_inventory.inventory_sha256
        ),
        "object_id": certificate.object_id,
        "object_contract_source_sha256": certificate.object_contract_source_sha256,
        "object_source_asset_sha256": certificate.object_surface.source_asset_sha256,
        "object_surface_geometry_sha256": certificate.object_surface.geometry_sha256,
        "object_material_boundary_evidence_sha256": (
            certificate.object_material_boundary.evidence_sha256
        ),
        "object_material_boundary_certificate_sha256": (
            certificate.object_material_boundary.certificate_sha256
        ),
        "shared_environment_certificate_sha256": (
            certificate.shared_environment.certificate_sha256
        ),
        "shared_environment_id": certificate.shared_environment.environment_id,
        "shared_environment_root_frame": certificate.shared_environment.root_frame,
        "shared_environment_robot_base_origin_m": [
            _float_hex(value)
            for value in certificate.shared_environment.robot_base_origin_m
        ],
        "shared_environment_registered_object_ids": list(
            certificate.shared_environment.registered_object_ids
        ),
        "shared_environment_obstacles": [
            [
                obstacle.name,
                obstacle.role,
                obstacle.prim_path,
                obstacle.geometry_sha256,
                len(obstacle.triangles_world_m),
            ]
            for obstacle in certificate.shared_environment.obstacles
        ],
        "binding_status": {
            "aggregate_robot_kinematics_binding_complete": True,
            "runtime_link_surface_binding_complete": True,
            "terminal_runtime_role_binding_complete": True,
            "object_planning_surface_binding_complete": True,
            "object_material_boundary_binding_complete": True,
            "table_fixture_environment_binding_complete": True,
            "candidate_specific_motion_binding_complete": False,
            "continuous_pad_contact_binding_complete": False,
            "formal_complete_collision_input_eligible": False,
        },
        "remaining_blockers": list(REMAINING_BLOCKERS),
        "claim_limitations": list(CLAIM_LIMITATIONS),
    }
    return _canonical_sha256(document)


def _build_aggregate_kinematic_binding(
    hand_contract: CARTSHandContract,
    roster: AuthoritativeCollisionLinkRoster,
    interval_options: IntervalArithmeticOptions,
) -> AggregateRobotKinematicBinding:
    combined_xml = build_verified_aggregate_robot_xml(roster)
    try:
        model = ThreeFingerHandModel.from_urdf(
            combined_xml,
            pad_geometry_contract=hand_contract.to_hand_model_pad_contract(),
            base_link="world",
        )
    except ValueError as error:
        raise AggregateCollisionInputError(
            "AGGREGATE_KINEMATIC_MODEL_BUILD_FAILED",
            str(error),
        ) from error
    include_bindings = tuple(
        (row.repository_path, row.sha256) for row in roster.include_sources
    )
    hand_contract_sha256 = file_sha256(hand_contract.contract_path)
    model_sha256 = _aggregate_model_sha256(
        model,
        aggregate_source_sha256=roster.aggregate_source.sha256,
        include_source_bindings=include_bindings,
        hand_contract_sha256=hand_contract_sha256,
    )
    return AggregateRobotKinematicBinding(
        method_id=METHOD_ID,
        kinematic_assembly_policy=KINEMATIC_ASSEMBLY_POLICY,
        aggregate_source_sha256=roster.aggregate_source.sha256,
        include_source_bindings=include_bindings,
        hand_contract_sha256=hand_contract_sha256,
        model=model,
        interval_options=interval_options,
        collision_link_names=roster.link_names,
        independent_joint_names=tuple(model.independent_joint_names),
        collision_link_count=len(roster.links),
        independent_joint_count=len(model.independent_joint_names),
        every_collision_link_connected_to_world=True,
        model_sha256=model_sha256,
    )


def _transformed_link_triangles(link: object) -> np.ndarray:
    mesh, provenance = load_stl_mesh(
        link.absolute_path,
        unit=link.unit,
        orient_outward=False,
    )
    if provenance.source_sha256 != link.sha256:
        raise AggregateCollisionInputError(
            "COLLISION_MESH_PROVENANCE_MISMATCH",
            link.link_name,
        )
    triangles = np.asarray(mesh.face_vertices_m, dtype=np.float64)
    scale = np.asarray(link.scale, dtype=np.float64)
    rotation = rpy_rotation(link.origin_rpy_rad)
    translation = np.asarray(link.origin_xyz_m, dtype=np.float64)
    transformed = triangles * scale
    transformed = transformed @ rotation.T + translation
    if not np.all(np.isfinite(transformed)):
        raise AggregateCollisionInputError(
            "COLLISION_LOCAL_TRANSFORM_NONFINITE",
            link.link_name,
        )
    return transformed


def build_carts_aggregate_collision_runtime_inputs(
    *,
    hand_contract: CARTSHandContract,
    collision_roster: AuthoritativeCollisionLinkRoster,
    geometry_binding: CollisionGeometryBindingCertificate,
    object_contract: LoadedObjectContract,
    shared_environment: SharedTableFixtureWorldCertificate,
    interval_options: IntervalArithmeticOptions,
) -> AggregateCollisionRuntimeInputCertificate:
    """Build exact static-scene inputs without claiming a candidate route."""

    if not isinstance(hand_contract, CARTSHandContract):
        raise AggregateCollisionInputError(
            "VERIFIED_HAND_CONTRACT_REQUIRED",
            "hand contract must come from its strict loader",
        )
    if not isinstance(collision_roster, AuthoritativeCollisionLinkRoster):
        raise AggregateCollisionInputError(
            "VERIFIED_COLLISION_ROSTER_REQUIRED",
            "collision roster must come from its strict loader",
        )
    if not isinstance(geometry_binding, CollisionGeometryBindingCertificate):
        raise AggregateCollisionInputError(
            "CERTIFIED_COLLISION_GEOMETRY_BINDING_REQUIRED",
            "geometry binding must be the exact certified type",
        )
    if not isinstance(object_contract, LoadedObjectContract):
        raise AggregateCollisionInputError(
            "VERIFIED_OBJECT_CONTRACT_REQUIRED",
            "object contract must come from its loader",
        )
    if not isinstance(shared_environment, SharedTableFixtureWorldCertificate):
        raise AggregateCollisionInputError(
            "VERIFIED_SHARED_ENVIRONMENT_REQUIRED",
            "shared environment must come from its strict loader",
        )
    if not isinstance(interval_options, IntervalArithmeticOptions):
        raise AggregateCollisionInputError(
            "EXPLICIT_INTERVAL_OPTIONS_REQUIRED",
            "interval arithmetic options cannot be defaulted",
        )
    if (
        geometry_binding.hand_contract_sha256
        != file_sha256(hand_contract.contract_path)
        or geometry_binding.collision_roster_sha256
        != collision_roster.roster_sha256
    ):
        raise AggregateCollisionInputError(
            "GEOMETRY_BINDING_INPUT_MISMATCH",
            "geometry certificate belongs to different hand or roster bytes",
        )
    if object_contract.object_id not in shared_environment.registered_object_ids:
        raise AggregateCollisionInputError(
            "OBJECT_NOT_REGISTERED_FOR_SHARED_ENVIRONMENT",
            object_contract.object_id,
        )
    if (
        shared_environment.root_frame != "world"
        or shared_environment.robot_base_origin_m != (0.0, 0.0, 0.0)
        or shared_environment.table_fixture_world_binding_complete is not True
        or shared_environment.loose_object_initial_pose_included is not False
        or shared_environment.candidate_specific_robot_route_included is not False
        or shared_environment.isaac_dynamic_state_included is not False
        or shared_environment.hardware_state_included is not False
    ):
        raise AggregateCollisionInputError(
            "SHARED_ENVIRONMENT_SCOPE_OR_FRAME_MISMATCH",
            shared_environment.environment_id,
        )

    kinematics = _build_aggregate_kinematic_binding(
        hand_contract,
        collision_roster,
        interval_options,
    )
    material_by_name = {
        row.link_name: row
        for row in geometry_binding.collision_link_material_bindings
    }
    surfaces: list[HashBoundLinkSurface] = []
    transformed_by_name: dict[str, np.ndarray] = {}
    for link in collision_roster.links:
        material = material_by_name.get(link.link_name)
        if (
            material is None
            or material.collision_mesh_sha256 != link.sha256
            or not material.material_boundary.formal_material_boundary_eligible
        ):
            raise AggregateCollisionInputError(
                "UNCERTIFIED_COLLISION_LINK_RUNTIME_INPUT",
                link.link_name,
            )
        triangles = _transformed_link_triangles(link)
        transformed_by_name[link.link_name] = triangles
        surfaces.append(
            HashBoundLinkSurface(
                link_name=link.link_name,
                source_asset_sha256=link.sha256,
                geometry_sha256=triangle_surface_geometry_sha256(triangles),
                triangles_link_m=triangles,
            )
        )

    surface_by_name = {row.link_name: row for row in surfaces}
    terminal_roles: list[TerminalRuntimeCollisionRole] = []
    for terminal in geometry_binding.terminal_role_bindings:
        full = surface_by_name[terminal.link_name]
        triangles = transformed_by_name[terminal.link_name]
        allowed = triangles[
            np.asarray(terminal.allowed_collision_face_indices, dtype=np.int64)
        ]
        forbidden = triangles[
            np.asarray(terminal.forbidden_collision_face_indices, dtype=np.int64)
        ]
        terminal_roles.append(
            TerminalRuntimeCollisionRole(
                link_name=terminal.link_name,
                pad_name=terminal.pad_name,
                terminal_certificate=terminal,
                full_surface=full,
                allowed_pad_surface=HashBoundLinkSurface(
                    link_name=terminal.link_name,
                    source_asset_sha256=terminal.collision_mesh_sha256,
                    geometry_sha256=triangle_surface_geometry_sha256(allowed),
                    triangles_link_m=allowed,
                ),
                forbidden_nonpad_surface=HashBoundLinkSurface(
                    link_name=terminal.link_name,
                    source_asset_sha256=terminal.collision_mesh_sha256,
                    geometry_sha256=triangle_surface_geometry_sha256(forbidden),
                    triangles_link_m=forbidden,
                ),
            )
        )

    inventory = build_self_collision_pair_inventory(
        link_names=collision_roster.link_names,
        srdf_assertions=(),
    )
    if inventory.all_pairs != collision_roster.all_self_pairs:
        raise AggregateCollisionInputError(
            "SELF_PAIR_INVENTORY_MISMATCH",
            "runtime all-pairs order differs from the authoritative roster",
        )

    object_source_sha256 = object_contract.verified_source_sha256.get(
        "planning_geometry"
    )
    if (
        not _is_sha256(object_source_sha256)
        or object_contract.model.provenance.source_sha256 != object_source_sha256
    ):
        raise AggregateCollisionInputError(
            "OBJECT_PLANNING_SURFACE_PROVENANCE_MISMATCH",
            object_contract.object_id,
        )
    object_triangles = np.asarray(
        object_contract.model.mesh.face_vertices_m,
        dtype=np.float64,
    )
    object_surface = HashBoundObjectSurface(
        object_id=object_contract.object_id,
        source_asset_sha256=object_source_sha256,
        geometry_sha256=triangle_surface_geometry_sha256(object_triangles),
        ray_closure_object_geometry_sha256=(
            object_contract.model.geometry_sha256
        ),
        triangles_object_m=object_triangles,
    )
    object_material_boundary = object_contract.material_boundary_evidence
    if (
        object_material_boundary.object_id != object_contract.object_id
        or object_material_boundary.source_asset_sha256 != object_source_sha256
        or object_material_boundary.formal_material_boundary_eligible is not True
        or object_material_boundary.certificate.source_face_count
        != len(object_triangles)
    ):
        raise AggregateCollisionInputError(
            "OBJECT_MATERIAL_BOUNDARY_BINDING_MISMATCH",
            object_contract.object_id,
        )

    values = {
        "method_id": METHOD_ID,
        "kinematic_binding": kinematics,
        "geometry_binding": geometry_binding,
        "link_surfaces": tuple(surfaces),
        "terminal_roles": tuple(terminal_roles),
        "self_pair_inventory": inventory,
        "object_surface": object_surface,
        "object_material_boundary": object_material_boundary,
        "shared_environment": shared_environment,
        "object_id": object_contract.object_id,
        "object_contract_source_sha256": object_source_sha256,
        "collision_link_count": len(surfaces),
        "terminal_role_count": len(terminal_roles),
        "self_pair_count": len(inventory.all_pairs),
        "aggregate_robot_kinematics_binding_complete": True,
        "runtime_link_surface_binding_complete": True,
        "terminal_runtime_role_binding_complete": True,
        "object_planning_surface_binding_complete": True,
        "object_material_boundary_binding_complete": True,
        "candidate_specific_motion_binding_complete": False,
        "table_fixture_environment_binding_complete": True,
        "continuous_pad_contact_binding_complete": False,
        "formal_complete_collision_input_eligible": False,
        "remaining_blockers": REMAINING_BLOCKERS,
        "claim_limitations": CLAIM_LIMITATIONS,
    }
    provisional = object.__new__(AggregateCollisionRuntimeInputCertificate)
    for name in AggregateCollisionRuntimeInputCertificate.__dataclass_fields__:
        if name != "certificate_sha256":
            object.__setattr__(provisional, name, values[name])
    object.__setattr__(provisional, "certificate_sha256", "0" * 64)
    return AggregateCollisionRuntimeInputCertificate(
        **values,
        certificate_sha256=_certificate_digest(provisional),
    )


__all__ = [
    "CLAIM_LIMITATIONS",
    "EXPECTED_AGGREGATE_SOURCE",
    "EXPECTED_INCLUDE_SOURCES",
    "EXPECTED_INDEPENDENT_JOINTS",
    "KINEMATIC_ASSEMBLY_POLICY",
    "METHOD_ID",
    "REMAINING_BLOCKERS",
    "AggregateCollisionInputError",
    "AggregateCollisionRuntimeInputCertificate",
    "AggregateRobotKinematicBinding",
    "TerminalRuntimeCollisionRole",
    "build_carts_aggregate_collision_runtime_inputs",
]
