"""Roster-wide material and terminal contact-role input certificate.

This static extension consumes the existing aggregate-xacro collision roster
without changing it.  Every registered collision mesh must first be proved an
embedded material boundary, and the three terminal meshes must additionally
carry exact authorized PAD/forbidden face partitions.  The result closes the
old *geometry-input* gaps while deliberately retaining motion and environment
blockers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Mapping

from kcg_connector.grasp.robust.collision_roster import (
    MOTION_MISSING_EVIDENCE,
    AuthoritativeCollisionLinkRoster,
)
from kcg_connector.grasp.robust.hand_contract import CARTSHandContract
from kcg_connector.grasp.robust.material_boundary import (
    MaterialBoundaryCertificate,
    certify_single_embedded_material_boundary,
)
from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh
from kcg_connector.grasp.robust.terminal_collision_boundary import (
    TerminalCollisionBoundaryCertificate,
    certify_carts_terminal_collision_boundaries,
)


METHOD_ID = "CARTS_COLLISION_GEOMETRY_AND_TERMINAL_ROLE_BINDING_V1"
ENVIRONMENT_MISSING_EVIDENCE = (
    "HASH_BOUND_OBJECT_PLUG_TABLE_AND_ENVIRONMENT_COLLISION_ROSTER"
)
REMAINING_BLOCKERS = (
    MOTION_MISSING_EVIDENCE,
    ENVIRONMENT_MISSING_EVIDENCE,
)
CLAIM_LIMITATIONS = (
    "STATIC_COLLISION_GEOMETRY_INPUT_BINDING_ONLY",
    "NO_ARM_IK_OR_APPROACH_CLOSURE_LIFT_TRAJECTORY_CERTIFICATE",
    "NO_OBJECT_PLUG_TABLE_OR_ENVIRONMENT_COLLISION_ROSTER_CERTIFICATE",
    "NO_FULL_SCENE_PAIR_OR_CONTINUOUS_TIME_COVERAGE",
    "NO_ISAAC_HARDWARE_OR_GRASP_CANDIDATE_CLAIM",
)


class CollisionGeometryBindingError(ValueError):
    """Raised when the complete static collision-geometry input cannot bind."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("collision geometry binding error cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class CollisionLinkMaterialBinding:
    ordinal: int
    link_name: str
    collision_mesh_sha256: str
    origin_xyz_m: tuple[float, float, float]
    origin_rpy_rad: tuple[float, float, float]
    scale: tuple[float, float, float]
    material_boundary: MaterialBoundaryCertificate

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or not self.link_name
            or not _is_sha256(self.collision_mesh_sha256)
            or not isinstance(self.material_boundary, MaterialBoundaryCertificate)
            or len(self.origin_xyz_m) != 3
            or len(self.origin_rpy_rad) != 3
            or len(self.scale) != 3
            or any(value <= 0.0 for value in self.scale)
        ):
            raise ValueError("collision link material binding is malformed")


@dataclass(frozen=True)
class CollisionGeometryBindingCertificate:
    method_id: str
    hand_contract_sha256: str
    collision_roster_sha256: str
    collision_link_material_bindings: tuple[CollisionLinkMaterialBinding, ...]
    terminal_role_bindings: tuple[
        TerminalCollisionBoundaryCertificate,
        TerminalCollisionBoundaryCertificate,
        TerminalCollisionBoundaryCertificate,
    ]
    collision_link_count: int
    self_pair_count: int
    verified_material_boundary_count: int
    verified_terminal_role_binding_count: int
    all_registered_collision_links_bound: bool
    solid_boundary_binding_complete: bool
    terminal_pad_role_binding_complete: bool
    motion_binding_complete: bool
    environment_binding_complete: bool
    formal_complete_collision_input_eligible: bool
    remaining_blockers: tuple[str, ...]
    claim_limitations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        if self.method_id != METHOD_ID:
            raise ValueError("collision geometry binding method changed")
        if not _is_sha256(self.hand_contract_sha256) or not _is_sha256(
            self.collision_roster_sha256
        ) or not _is_sha256(self.certificate_sha256):
            raise ValueError("collision geometry binding digest is invalid")
        bindings = self.collision_link_material_bindings
        terminals = self.terminal_role_bindings
        if (
            self.collision_link_count != 17
            or self.self_pair_count != 136
            or len(bindings) != self.collision_link_count
            or tuple(row.ordinal for row in bindings) != tuple(range(17))
            or len({row.link_name for row in bindings}) != 17
            or self.verified_material_boundary_count != len(bindings)
            or len(terminals) != 3
            or self.verified_terminal_role_binding_count != len(terminals)
            or tuple(row.link_name for row in terminals)
            != ("f1Link3", "f2Link2", "f3Link3")
        ):
            raise ValueError("collision geometry binding coverage is incomplete")
        binding_by_name = {row.link_name: row for row in bindings}
        if any(
            terminal.collision_mesh_sha256
            != binding_by_name[terminal.link_name].collision_mesh_sha256
            or terminal.material_boundary_certificate_sha256
            != binding_by_name[
                terminal.link_name
            ].material_boundary.certificate_sha256
            for terminal in terminals
        ):
            raise ValueError("terminal role and roster material evidence diverged")
        if (
            self.all_registered_collision_links_bound is not True
            or self.solid_boundary_binding_complete is not True
            or self.terminal_pad_role_binding_complete is not True
            or self.motion_binding_complete is not False
            or self.environment_binding_complete is not False
            or self.formal_complete_collision_input_eligible is not False
            or self.remaining_blockers != REMAINING_BLOCKERS
            or self.claim_limitations != CLAIM_LIMITATIONS
        ):
            raise ValueError("collision geometry binding claim boundary changed")
        if self.certificate_sha256 != _certificate_digest(self):
            raise ValueError("collision geometry binding certificate digest changed")

    @property
    def audit(self) -> Mapping[str, object]:
        return {
            "method_id": self.method_id,
            "hand_contract_sha256": self.hand_contract_sha256,
            "collision_roster_sha256": self.collision_roster_sha256,
            "collision_link_count": self.collision_link_count,
            "self_pair_count": self.self_pair_count,
            "verified_material_boundary_count": (
                self.verified_material_boundary_count
            ),
            "verified_terminal_role_binding_count": (
                self.verified_terminal_role_binding_count
            ),
            "solid_boundary_binding_complete": True,
            "terminal_pad_role_binding_complete": True,
            "motion_binding_complete": False,
            "environment_binding_complete": False,
            "formal_complete_collision_input_eligible": False,
            "remaining_blockers": list(self.remaining_blockers),
            "claim_limitations": list(self.claim_limitations),
            "certificate_sha256": self.certificate_sha256,
        }


def _certificate_digest(certificate: CollisionGeometryBindingCertificate) -> str:
    digest = hashlib.sha256()
    digest.update(METHOD_ID.encode("ascii") + b"\0")
    digest.update(bytes.fromhex(certificate.hand_contract_sha256))
    digest.update(bytes.fromhex(certificate.collision_roster_sha256))
    for binding in certificate.collision_link_material_bindings:
        digest.update(struct.pack("<Q", binding.ordinal))
        digest.update(binding.link_name.encode("ascii") + b"\0")
        digest.update(bytes.fromhex(binding.collision_mesh_sha256))
        for vector in (
            binding.origin_xyz_m,
            binding.origin_rpy_rad,
            binding.scale,
        ):
            digest.update(struct.pack("<3d", *vector))
        digest.update(bytes.fromhex(binding.material_boundary.certificate_sha256))
    for terminal in certificate.terminal_role_bindings:
        digest.update(bytes.fromhex(terminal.certificate_sha256))
    for blocker in REMAINING_BLOCKERS:
        digest.update(blocker.encode("ascii") + b"\0")
    for limitation in CLAIM_LIMITATIONS:
        digest.update(limitation.encode("ascii") + b"\0")
    return digest.hexdigest()


def certify_carts_collision_geometry_bindings(
    hand_contract: CARTSHandContract,
    collision_roster: AuthoritativeCollisionLinkRoster,
) -> CollisionGeometryBindingCertificate:
    """Certify all 17 roster meshes and all three exact terminal PAD roles."""

    if not isinstance(hand_contract, CARTSHandContract):
        raise CollisionGeometryBindingError(
            "VERIFIED_HAND_CONTRACT_REQUIRED",
            "hand contract must come from its strict loader",
        )
    if not isinstance(collision_roster, AuthoritativeCollisionLinkRoster):
        raise CollisionGeometryBindingError(
            "VERIFIED_COLLISION_ROSTER_REQUIRED",
            "collision roster must come from its strict loader",
        )
    link_bindings: list[CollisionLinkMaterialBinding] = []
    for link in collision_roster.links:
        mesh, provenance = load_stl_mesh(
            link.absolute_path,
            unit=link.unit,
            orient_outward=False,
        )
        if provenance.source_sha256 != link.sha256:
            raise CollisionGeometryBindingError(
                "COLLISION_MESH_PROVENANCE_MISMATCH",
                f"link={link.link_name}",
            )
        material_boundary = certify_single_embedded_material_boundary(
            mesh.vertices_m, mesh.faces
        )
        link_bindings.append(
            CollisionLinkMaterialBinding(
                ordinal=link.ordinal,
                link_name=link.link_name,
                collision_mesh_sha256=link.sha256,
                origin_xyz_m=link.origin_xyz_m,
                origin_rpy_rad=link.origin_rpy_rad,
                scale=link.scale,
                material_boundary=material_boundary,
            )
        )
    terminal_bindings = certify_carts_terminal_collision_boundaries(
        hand_contract, collision_roster
    )
    values = {
        "method_id": METHOD_ID,
        "hand_contract_sha256": file_sha256(hand_contract.contract_path),
        "collision_roster_sha256": collision_roster.roster_sha256,
        "collision_link_material_bindings": tuple(link_bindings),
        "terminal_role_bindings": terminal_bindings,
        "collision_link_count": len(collision_roster.links),
        "self_pair_count": len(collision_roster.all_self_pairs),
        "verified_material_boundary_count": len(link_bindings),
        "verified_terminal_role_binding_count": len(terminal_bindings),
        "all_registered_collision_links_bound": True,
        "solid_boundary_binding_complete": True,
        "terminal_pad_role_binding_complete": True,
        "motion_binding_complete": False,
        "environment_binding_complete": False,
        "formal_complete_collision_input_eligible": False,
        "remaining_blockers": REMAINING_BLOCKERS,
        "claim_limitations": CLAIM_LIMITATIONS,
    }
    provisional = object.__new__(CollisionGeometryBindingCertificate)
    for name in CollisionGeometryBindingCertificate.__dataclass_fields__:
        if name != "certificate_sha256":
            object.__setattr__(provisional, name, values[name])
    object.__setattr__(provisional, "certificate_sha256", "0" * 64)
    digest = _certificate_digest(provisional)
    return CollisionGeometryBindingCertificate(
        **values,
        certificate_sha256=digest,
    )


__all__ = [
    "CLAIM_LIMITATIONS",
    "ENVIRONMENT_MISSING_EVIDENCE",
    "METHOD_ID",
    "REMAINING_BLOCKERS",
    "CollisionGeometryBindingCertificate",
    "CollisionGeometryBindingError",
    "CollisionLinkMaterialBinding",
    "certify_carts_collision_geometry_bindings",
]
