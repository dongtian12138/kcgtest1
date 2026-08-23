"""Exact terminal PAD roles on certified collision-material boundaries.

The user-authorized blue PAD meshes and the URDF collision meshes are separate
assets.  A nearest-surface transfer would silently turn nearby red-tip or side
faces into allowed object contacts.  This module instead admits only collision
triangles whose three binary64 vertices exactly match an authorized PAD
triangle with unambiguous multiset occurrence and the same material-outward
winding.  Every other collision triangle remains forbidden.

The returned certificate closes only the local terminal-link material and
contact-role input.  It does not certify an arm pose, approach, closure, lift,
object/environment separation, force feasibility, simulation, or a grasp
candidate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import struct
from pathlib import Path
from typing import Mapping

import numpy as np

from kcg_connector.grasp.robust.collision_contract import (
    canonical_oriented_triangle_sha256,
    canonical_unoriented_triangle_sha256,
)
from kcg_connector.grasp.robust.collision_roster import (
    AuthoritativeCollisionLinkRoster,
    CollisionLinkBinding,
)
from kcg_connector.grasp.robust.hand_contract import (
    CARTSHandContract,
    VerifiedPad,
)
from kcg_connector.grasp.robust.material_boundary import (
    MaterialBoundaryCertificate,
    certify_single_embedded_material_boundary,
)
from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh
from kcg_connector.grasp.robust.surface_orientation import (
    SurfaceBoundaryRole,
    audit_surface_orientation,
)


METHOD_ID = "CARTS_EXACT_TERMINAL_COLLISION_MATERIAL_AND_PAD_ROLE_V1"
ROLE_POLICY = (
    "EXACT_UNORIENTED_BINARY64_TRIANGLE_MULTISET_X_AMBIGUOUS_FORBIDDEN_X_"
    "MATERIAL_OUTWARD_WINDING_MUST_MATCH_AUTHORIZED_PAD_SOURCE"
)
CLAIM_LIMITATIONS = (
    "ONLY_EXACTLY_SHARED_COLLISION_TRIANGLES_ARE_ALLOWED_PAD_CONTACT",
    "UNMAPPED_AUTHORIZED_PAD_TRIANGLES_REMAIN_FORBIDDEN_NOT_INTERPOLATED",
    "TERMINAL_LINK_LOCAL_COLLISION_BOUNDARY_ONLY",
    "NO_ARM_POSE_APPROACH_CLOSURE_LIFT_OBJECT_OR_ENVIRONMENT_CLAIM",
    "NO_ISAAC_HARDWARE_FORCE_OR_GRASP_CANDIDATE_CLAIM",
)


class TerminalCollisionBoundaryError(ValueError):
    """Fail-closed terminal collision role or lineage error."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("terminal collision error fields cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class TerminalCollisionBoundaryCertificate:
    method_id: str
    role_policy: str
    link_name: str
    pad_name: str
    hand_contract_sha256: str
    pad_source_manifest_sha256: str
    pad_mesh_sha256: str
    collision_roster_sha256: str
    collision_mesh_sha256: str
    collision_source_indexed_mesh_sha256: str
    material_boundary_certificate_sha256: str
    collision_face_count: int
    pad_face_count: int
    exact_shared_collision_face_count: int
    allowed_collision_face_indices: tuple[int, ...]
    forbidden_collision_face_indices: tuple[int, ...]
    ambiguous_collision_face_count: int
    unmapped_pad_face_count: int
    same_material_outward_winding_count: int
    reversed_material_outward_winding_count: int
    exact_coordinate_frame_binding_complete: bool
    collision_face_partition_complete: bool
    source_lineage_complete: bool
    nearest_surface_or_tolerance_mapping_used: bool
    formal_solid_coverage: bool
    formal_terminal_role_binding_eligible: bool
    material_boundary: MaterialBoundaryCertificate
    claim_limitations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        if self.method_id != METHOD_ID or self.role_policy != ROLE_POLICY:
            raise ValueError("terminal collision method or role policy changed")
        if not self.link_name or not self.pad_name:
            raise ValueError("terminal collision link and PAD names are required")
        for digest in (
            self.hand_contract_sha256,
            self.pad_source_manifest_sha256,
            self.pad_mesh_sha256,
            self.collision_roster_sha256,
            self.collision_mesh_sha256,
            self.collision_source_indexed_mesh_sha256,
            self.material_boundary_certificate_sha256,
            self.certificate_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("terminal collision digest is invalid")
        count_fields = (
            self.collision_face_count,
            self.pad_face_count,
            self.exact_shared_collision_face_count,
            self.ambiguous_collision_face_count,
            self.unmapped_pad_face_count,
            self.same_material_outward_winding_count,
            self.reversed_material_outward_winding_count,
        )
        if any(type(value) is not int or value < 0 for value in count_fields):
            raise ValueError("terminal collision counts must be nonnegative integers")
        allowed = self.allowed_collision_face_indices
        forbidden = self.forbidden_collision_face_indices
        if (
            allowed != tuple(sorted(allowed))
            or forbidden != tuple(sorted(forbidden))
            or len(set(allowed)) != len(allowed)
            or len(set(forbidden)) != len(forbidden)
            or set(allowed) & set(forbidden)
            or tuple(sorted(allowed + forbidden))
            != tuple(range(self.collision_face_count))
            or len(allowed) != self.exact_shared_collision_face_count
            or self.unmapped_pad_face_count
            != self.pad_face_count - self.exact_shared_collision_face_count
            or self.same_material_outward_winding_count
            + self.reversed_material_outward_winding_count
            != self.exact_shared_collision_face_count
        ):
            raise ValueError("terminal collision face partition is inconsistent")
        if (
            not isinstance(self.material_boundary, MaterialBoundaryCertificate)
            or self.material_boundary.source_face_count != self.collision_face_count
            or self.material_boundary.source_indexed_mesh_sha256
            != self.collision_source_indexed_mesh_sha256
            or self.material_boundary.certificate_sha256
            != self.material_boundary_certificate_sha256
        ):
            raise ValueError("terminal collision material certificate is not bound")
        if (
            self.ambiguous_collision_face_count != 0
            or self.exact_shared_collision_face_count == 0
            or self.reversed_material_outward_winding_count != 0
            or self.exact_coordinate_frame_binding_complete is not True
            or self.collision_face_partition_complete is not True
            or self.source_lineage_complete is not True
            or self.nearest_surface_or_tolerance_mapping_used is not False
            or self.formal_solid_coverage
            is not self.material_boundary.formal_material_boundary_eligible
            or self.formal_solid_coverage is not True
            or self.formal_terminal_role_binding_eligible is not True
            or self.claim_limitations != CLAIM_LIMITATIONS
        ):
            raise ValueError("terminal collision formal claim boundary is incomplete")
        if self.certificate_sha256 != _certificate_digest(self):
            raise ValueError("terminal collision certificate digest is not canonical")

    @property
    def audit(self) -> Mapping[str, object]:
        return {
            "method_id": self.method_id,
            "role_policy": self.role_policy,
            "link_name": self.link_name,
            "pad_name": self.pad_name,
            "collision_face_count": self.collision_face_count,
            "pad_face_count": self.pad_face_count,
            "exact_shared_collision_face_count": (
                self.exact_shared_collision_face_count
            ),
            "allowed_collision_face_indices": list(
                self.allowed_collision_face_indices
            ),
            "forbidden_collision_face_count": len(
                self.forbidden_collision_face_indices
            ),
            "unmapped_pad_face_count": self.unmapped_pad_face_count,
            "same_material_outward_winding_count": (
                self.same_material_outward_winding_count
            ),
            "nearest_surface_or_tolerance_mapping_used": False,
            "formal_solid_coverage": self.formal_solid_coverage,
            "formal_terminal_role_binding_eligible": (
                self.formal_terminal_role_binding_eligible
            ),
            "material_boundary_certificate_sha256": (
                self.material_boundary_certificate_sha256
            ),
            "certificate_sha256": self.certificate_sha256,
            "claim_limitations": list(self.claim_limitations),
        }


def _certificate_digest(
    certificate: TerminalCollisionBoundaryCertificate,
) -> str:
    digest = hashlib.sha256()
    digest.update(METHOD_ID.encode("ascii") + b"\0")
    digest.update(ROLE_POLICY.encode("ascii") + b"\0")
    for value in (
        certificate.link_name,
        certificate.pad_name,
        certificate.hand_contract_sha256,
        certificate.pad_source_manifest_sha256,
        certificate.pad_mesh_sha256,
        certificate.collision_roster_sha256,
        certificate.collision_mesh_sha256,
        certificate.collision_source_indexed_mesh_sha256,
        certificate.material_boundary_certificate_sha256,
    ):
        digest.update(value.encode("ascii") + b"\0")
    for value in (
        certificate.collision_face_count,
        certificate.pad_face_count,
        certificate.exact_shared_collision_face_count,
        certificate.ambiguous_collision_face_count,
        certificate.unmapped_pad_face_count,
        certificate.same_material_outward_winding_count,
        certificate.reversed_material_outward_winding_count,
    ):
        digest.update(struct.pack("<Q", value))
    for label, indices in (
        (b"A", certificate.allowed_collision_face_indices),
        (b"F", certificate.forbidden_collision_face_indices),
    ):
        for index in indices:
            digest.update(label + struct.pack("<Q", index))
    for limitation in CLAIM_LIMITATIONS:
        digest.update(limitation.encode("ascii") + b"\0")
    return digest.hexdigest()


def _identity_terminal_frames(
    pad: VerifiedPad,
    collision_link: CollisionLinkBinding,
) -> bool:
    return (
        pad.link_name == collision_link.link_name
        and pad.coordinate_frame == pad.link_name
        and pad.unit == "m"
        and pad.origin_xyz_m == (0.0, 0.0, 0.0)
        and pad.origin_rpy_rad == (0.0, 0.0, 0.0)
        and collision_link.unit == "m"
        and collision_link.origin_xyz_m == (0.0, 0.0, 0.0)
        and collision_link.origin_rpy_rad == (0.0, 0.0, 0.0)
        and collision_link.scale == (1.0, 1.0, 1.0)
    )


def certify_terminal_collision_boundary(
    *,
    pad: VerifiedPad,
    collision_link: CollisionLinkBinding,
    hand_contract_sha256: str,
    pad_source_manifest_sha256: str,
    collision_roster_sha256: str,
) -> TerminalCollisionBoundaryCertificate:
    """Bind one verified PAD to one exact embedded terminal collision mesh."""

    for label, digest in (
        ("hand contract", hand_contract_sha256),
        ("PAD source manifest", pad_source_manifest_sha256),
        ("collision roster", collision_roster_sha256),
    ):
        if not _is_sha256(digest):
            raise TerminalCollisionBoundaryError(
                "INVALID_SOURCE_DIGEST", f"{label} SHA-256 is invalid"
            )
    if not _identity_terminal_frames(pad, collision_link):
        raise TerminalCollisionBoundaryError(
            "TERMINAL_FRAME_BINDING_MISMATCH",
            f"PAD and collision mesh are not exact identity-frame metres: "
            f"{pad.name}:{collision_link.link_name}",
        )
    if file_sha256(pad.mesh.absolute_path) != pad.mesh.sha256:
        raise TerminalCollisionBoundaryError(
            "PAD_SOURCE_SHA256_MISMATCH", f"PAD bytes changed: {pad.name}"
        )
    try:
        with np.load(pad.mesh.absolute_path, allow_pickle=False) as archive:
            stored_points = np.asarray(archive["points_local_m"], dtype="<f8")
            stored_faces = np.asarray(archive["faces"], dtype="<i8")
    except (OSError, ValueError, KeyError) as error:
        raise TerminalCollisionBoundaryError(
            "PAD_SOURCE_ARRAYS_UNAVAILABLE",
            f"PAD arrays cannot be reloaded: {pad.name}",
        ) from error
    if (
        stored_points.shape != pad.points_local_m.shape
        or stored_faces.shape != pad.faces.shape
        or stored_points.tobytes(order="C")
        != np.asarray(pad.points_local_m, dtype="<f8").tobytes(order="C")
        or stored_faces.tobytes(order="C")
        != np.asarray(pad.faces, dtype="<i8").tobytes(order="C")
    ):
        raise TerminalCollisionBoundaryError(
            "PAD_RUNTIME_ARRAYS_DIFFER_FROM_HASH_BOUND_SOURCE",
            f"PAD runtime arrays changed: {pad.name}",
        )
    if file_sha256(collision_link.absolute_path) != collision_link.sha256:
        raise TerminalCollisionBoundaryError(
            "COLLISION_SOURCE_SHA256_MISMATCH",
            f"collision bytes changed: {collision_link.link_name}",
        )

    collision_mesh, provenance = load_stl_mesh(
        collision_link.absolute_path,
        unit="m",
        orient_outward=False,
    )
    if provenance.source_sha256 != collision_link.sha256:
        raise TerminalCollisionBoundaryError(
            "COLLISION_STL_PROVENANCE_MISMATCH",
            f"STL loader hash changed: {collision_link.link_name}",
        )
    material_boundary = certify_single_embedded_material_boundary(
        collision_mesh.vertices_m, collision_mesh.faces
    )
    orientation = audit_surface_orientation(
        collision_mesh.vertices_m,
        collision_mesh.faces,
        role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
    )
    collision_triangles = collision_mesh.face_vertices_m
    oriented_collision_triangles = tuple(
        triangle if sign == 1 else triangle[[0, 2, 1]]
        for triangle, sign in zip(
            collision_triangles,
            orientation.positive_volume_winding_sign_by_source_face,
        )
    )
    pad_triangles = pad.points_local_m[pad.faces]

    collision_indices_by_geometry: defaultdict[str, list[int]] = defaultdict(list)
    for face_index, triangle in enumerate(collision_triangles):
        collision_indices_by_geometry[
            canonical_unoriented_triangle_sha256(triangle)
        ].append(face_index)
    pad_orientation_by_geometry: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for triangle in pad_triangles:
        geometry = canonical_unoriented_triangle_sha256(triangle)
        pad_orientation_by_geometry[geometry][
            canonical_oriented_triangle_sha256(triangle)
        ] += 1

    allowed: list[int] = []
    forbidden: list[int] = []
    ambiguous: list[int] = []
    shared_occurrence_count = 0
    for geometry, collision_indices in collision_indices_by_geometry.items():
        pad_count = sum(pad_orientation_by_geometry.get(geometry, {}).values())
        shared_occurrence_count += min(len(collision_indices), pad_count)
        if pad_count == 0:
            forbidden.extend(collision_indices)
        elif pad_count < len(collision_indices):
            ambiguous.extend(collision_indices)
            forbidden.extend(collision_indices)
        else:
            allowed.extend(collision_indices)
    if ambiguous:
        raise TerminalCollisionBoundaryError(
            "AMBIGUOUS_CROSS_SOURCE_TRIANGLE_MULTIPLICITY",
            f"link={collision_link.link_name}, "
            f"ambiguous_collision_face_count={len(ambiguous)}",
        )
    if not allowed:
        raise TerminalCollisionBoundaryError(
            "NO_EXACT_AUTHORIZED_PAD_COLLISION_FACE",
            f"link={collision_link.link_name}",
        )

    remaining_pad_orientations = {
        geometry: Counter(counter)
        for geometry, counter in pad_orientation_by_geometry.items()
    }
    same_winding = 0
    reversed_winding = 0
    for face_index in sorted(allowed):
        triangle = oriented_collision_triangles[face_index]
        geometry = canonical_unoriented_triangle_sha256(triangle)
        same = canonical_oriented_triangle_sha256(triangle)
        reverse = canonical_oriented_triangle_sha256(triangle[[0, 2, 1]])
        available = remaining_pad_orientations[geometry]
        if available.get(same, 0) > 0:
            available[same] -= 1
            same_winding += 1
        elif available.get(reverse, 0) > 0:
            available[reverse] -= 1
            reversed_winding += 1
        else:
            raise TerminalCollisionBoundaryError(
                "PAD_COLLISION_ORIENTATION_OCCURRENCE_MISMATCH",
                f"link={collision_link.link_name}, face={face_index}",
            )
    if reversed_winding:
        raise TerminalCollisionBoundaryError(
            "PAD_MATERIAL_OUTWARD_WINDING_MISMATCH",
            f"link={collision_link.link_name}, reversed={reversed_winding}",
        )
    allowed_indices = tuple(sorted(allowed))
    forbidden_indices = tuple(sorted(forbidden))
    if shared_occurrence_count != len(allowed_indices):
        raise TerminalCollisionBoundaryError(
            "INTERNAL_SHARED_OCCURRENCE_MISMATCH",
            f"link={collision_link.link_name}",
        )

    values = {
        "method_id": METHOD_ID,
        "role_policy": ROLE_POLICY,
        "link_name": collision_link.link_name,
        "pad_name": pad.name,
        "hand_contract_sha256": hand_contract_sha256,
        "pad_source_manifest_sha256": pad_source_manifest_sha256,
        "pad_mesh_sha256": pad.mesh.sha256,
        "collision_roster_sha256": collision_roster_sha256,
        "collision_mesh_sha256": collision_link.sha256,
        "collision_source_indexed_mesh_sha256": (
            material_boundary.source_indexed_mesh_sha256
        ),
        "material_boundary_certificate_sha256": (
            material_boundary.certificate_sha256
        ),
        "collision_face_count": len(collision_triangles),
        "pad_face_count": len(pad_triangles),
        "exact_shared_collision_face_count": len(allowed_indices),
        "allowed_collision_face_indices": allowed_indices,
        "forbidden_collision_face_indices": forbidden_indices,
        "ambiguous_collision_face_count": 0,
        "unmapped_pad_face_count": len(pad_triangles) - len(allowed_indices),
        "same_material_outward_winding_count": same_winding,
        "reversed_material_outward_winding_count": 0,
        "exact_coordinate_frame_binding_complete": True,
        "collision_face_partition_complete": True,
        "source_lineage_complete": True,
        "nearest_surface_or_tolerance_mapping_used": False,
        "formal_solid_coverage": True,
        "formal_terminal_role_binding_eligible": True,
        "material_boundary": material_boundary,
        "claim_limitations": CLAIM_LIMITATIONS,
    }
    provisional = object.__new__(TerminalCollisionBoundaryCertificate)
    for name in TerminalCollisionBoundaryCertificate.__dataclass_fields__:
        if name != "certificate_sha256":
            object.__setattr__(provisional, name, values[name])
    object.__setattr__(provisional, "certificate_sha256", "0" * 64)
    digest = _certificate_digest(provisional)
    return TerminalCollisionBoundaryCertificate(
        **values,
        certificate_sha256=digest,
    )


def certify_carts_terminal_collision_boundaries(
    hand_contract: CARTSHandContract,
    collision_roster: AuthoritativeCollisionLinkRoster,
) -> tuple[
    TerminalCollisionBoundaryCertificate,
    TerminalCollisionBoundaryCertificate,
    TerminalCollisionBoundaryCertificate,
]:
    """Join the verified hand and aggregate-xacro roster for all three PADs."""

    if not isinstance(hand_contract, CARTSHandContract):
        raise TerminalCollisionBoundaryError(
            "VERIFIED_HAND_CONTRACT_REQUIRED",
            "hand_contract must be loaded by load_carts_hand_contract",
        )
    if not isinstance(collision_roster, AuthoritativeCollisionLinkRoster):
        raise TerminalCollisionBoundaryError(
            "VERIFIED_COLLISION_ROSTER_REQUIRED",
            "collision_roster must be loaded by its verified loader",
        )
    link_by_name = {link.link_name: link for link in collision_roster.links}
    expected_links = tuple(pad.link_name for pad in hand_contract.pads)
    if expected_links != ("f1Link3", "f2Link2", "f3Link3"):
        raise TerminalCollisionBoundaryError(
            "TERMINAL_PAD_LINK_SET_CHANGED",
            f"terminal links={expected_links}",
        )
    try:
        collision_links = tuple(link_by_name[name] for name in expected_links)
    except KeyError as error:
        raise TerminalCollisionBoundaryError(
            "TERMINAL_COLLISION_LINK_MISSING",
            f"collision roster lacks {error.args[0]}",
        ) from error
    hand_contract_sha256 = file_sha256(hand_contract.contract_path)
    certificates = tuple(
        certify_terminal_collision_boundary(
            pad=pad,
            collision_link=collision_link,
            hand_contract_sha256=hand_contract_sha256,
            pad_source_manifest_sha256=hand_contract.source_manifest.sha256,
            collision_roster_sha256=collision_roster.roster_sha256,
        )
        for pad, collision_link in zip(hand_contract.pads, collision_links)
    )
    return certificates  # type: ignore[return-value]


__all__ = [
    "CLAIM_LIMITATIONS",
    "METHOD_ID",
    "ROLE_POLICY",
    "TerminalCollisionBoundaryCertificate",
    "TerminalCollisionBoundaryError",
    "certify_carts_terminal_collision_boundaries",
    "certify_terminal_collision_boundary",
]
