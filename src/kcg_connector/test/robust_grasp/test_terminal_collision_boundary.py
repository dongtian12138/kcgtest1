from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.collision_contract import (
    canonical_oriented_triangle_sha256,
    canonical_unoriented_triangle_sha256,
)
from kcg_connector.grasp.robust.collision_roster import (
    load_authoritative_collision_link_roster,
)
from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract
from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh
from kcg_connector.grasp.robust.surface_orientation import (
    SurfaceBoundaryRole,
    audit_surface_orientation,
)
from kcg_connector.grasp.robust.terminal_collision_boundary import (
    CLAIM_LIMITATIONS,
    TerminalCollisionBoundaryError,
    certify_carts_terminal_collision_boundaries,
    certify_terminal_collision_boundary,
)


REPOSITORY = Path(__file__).resolve().parents[4]
HAND_CONTRACT_PATH = REPOSITORY / "src/kcg_connector/config/carts_hand_contact_v1.yaml"
COLLISION_ROSTER_PATH = (
    REPOSITORY / "src/kcg_connector/config/carts_collision_roster_v1.yaml"
)


@pytest.fixture(scope="module")
def verified_inputs():
    hand = load_carts_hand_contract(
        HAND_CONTRACT_PATH, repository_root=REPOSITORY
    )
    roster = load_authoritative_collision_link_roster(
        COLLISION_ROSTER_PATH, repository_root=REPOSITORY
    )
    certificates = certify_carts_terminal_collision_boundaries(hand, roster)
    return hand, roster, certificates


def test_real_three_terminal_boundaries_have_only_sixteen_exact_pad_faces(
    verified_inputs,
) -> None:
    _hand, _roster, certificates = verified_inputs
    assert tuple(row.link_name for row in certificates) == (
        "f1Link3",
        "f2Link2",
        "f3Link3",
    )
    assert tuple(row.pad_name for row in certificates) == (
        "finger_1_pad",
        "finger_2_pad",
        "finger_3_pad",
    )
    for certificate in certificates:
        assert certificate.collision_face_count == 1100
        assert certificate.pad_face_count == 2479
        assert certificate.exact_shared_collision_face_count == 16
        assert len(certificate.allowed_collision_face_indices) == 16
        assert len(certificate.forbidden_collision_face_indices) == 1084
        assert certificate.unmapped_pad_face_count == 2463
        assert certificate.same_material_outward_winding_count == 16
        assert certificate.reversed_material_outward_winding_count == 0
        assert certificate.nearest_surface_or_tolerance_mapping_used is False
        assert certificate.formal_solid_coverage is True
        assert certificate.formal_terminal_role_binding_eligible is True
        assert certificate.claim_limitations == CLAIM_LIMITATIONS


def test_every_allowed_real_collision_face_is_exact_pad_geometry_and_winding(
    verified_inputs,
) -> None:
    hand, roster, certificates = verified_inputs
    pad_by_link = {pad.link_name: pad for pad in hand.pads}
    collision_by_link = {link.link_name: link for link in roster.links}
    for certificate in certificates:
        pad = pad_by_link[certificate.link_name]
        collision = collision_by_link[certificate.link_name]
        mesh, _provenance = load_stl_mesh(
            collision.absolute_path, unit="m", orient_outward=False
        )
        orientation = audit_surface_orientation(
            mesh.vertices_m,
            mesh.faces,
            role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
        )
        pad_triangles = pad.points_local_m[pad.faces]
        pad_unoriented = {
            canonical_unoriented_triangle_sha256(triangle)
            for triangle in pad_triangles
        }
        pad_oriented = {
            canonical_oriented_triangle_sha256(triangle)
            for triangle in pad_triangles
        }
        for face_index in certificate.allowed_collision_face_indices:
            triangle = mesh.face_vertices_m[face_index]
            if orientation.positive_volume_winding_sign_by_source_face[face_index] == -1:
                triangle = triangle[[0, 2, 1]]
            assert canonical_unoriented_triangle_sha256(triangle) in pad_unoriented
            assert canonical_oriented_triangle_sha256(triangle) in pad_oriented


def test_certificate_is_deterministic_immutable_and_cannot_expand_pad_role(
    verified_inputs,
) -> None:
    hand, roster, certificates = verified_inputs
    repeated = certify_carts_terminal_collision_boundaries(hand, roster)
    assert certificates == repeated
    first = certificates[0]
    with pytest.raises(FrozenInstanceError):
        first.formal_terminal_role_binding_eligible = False  # type: ignore[misc]
    with pytest.raises(ValueError):
        replace(
            first,
            allowed_collision_face_indices=tuple(range(1100)),
            forbidden_collision_face_indices=(),
            exact_shared_collision_face_count=1100,
        )


def test_nonidentity_terminal_frame_fails_closed(verified_inputs) -> None:
    hand, roster, _certificates = verified_inputs
    pad = replace(hand.pads[0], origin_xyz_m=(0.001, 0.0, 0.0))
    collision = next(link for link in roster.links if link.link_name == pad.link_name)
    with pytest.raises(TerminalCollisionBoundaryError) as error:
        certify_terminal_collision_boundary(
            pad=pad,
            collision_link=collision,
            hand_contract_sha256=file_sha256(hand.contract_path),
            pad_source_manifest_sha256=hand.source_manifest.sha256,
            collision_roster_sha256=roster.roster_sha256,
        )
    assert error.value.code == "TERMINAL_FRAME_BINDING_MISMATCH"


def test_runtime_pad_array_cannot_differ_from_hash_bound_npz(
    verified_inputs,
) -> None:
    hand, roster, _certificates = verified_inputs
    shifted = np.array(hand.pads[0].points_local_m, copy=True)
    shifted[0, 0] += 0.001
    shifted.flags.writeable = False
    pad = replace(hand.pads[0], points_local_m=shifted)
    collision = next(link for link in roster.links if link.link_name == pad.link_name)
    with pytest.raises(TerminalCollisionBoundaryError) as error:
        certify_terminal_collision_boundary(
            pad=pad,
            collision_link=collision,
            hand_contract_sha256=file_sha256(hand.contract_path),
            pad_source_manifest_sha256=hand.source_manifest.sha256,
            collision_roster_sha256=roster.roster_sha256,
        )
    assert error.value.code == (
        "PAD_RUNTIME_ARRAYS_DIFFER_FROM_HASH_BOUND_SOURCE"
    )


def test_invalid_upstream_digest_cannot_enter_terminal_certificate(
    verified_inputs,
) -> None:
    hand, roster, _certificates = verified_inputs
    pad = hand.pads[0]
    collision = next(link for link in roster.links if link.link_name == pad.link_name)
    with pytest.raises(TerminalCollisionBoundaryError) as error:
        certify_terminal_collision_boundary(
            pad=pad,
            collision_link=collision,
            hand_contract_sha256="not-a-digest",
            pad_source_manifest_sha256=hand.source_manifest.sha256,
            collision_roster_sha256=roster.roster_sha256,
        )
    assert error.value.code == "INVALID_SOURCE_DIGEST"
