from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.object_contract import load_object_contract
from kcg_connector.grasp.robust.surface_orientation import (
    ExactDyadic,
    SurfaceBoundaryRole,
    SurfaceOrientationAuditError,
    UNVERIFIED,
    VERIFIED,
    audit_surface_orientation,
)


REPOSITORY = Path(__file__).resolve().parents[4]
OBJECT_CONTRACT = REPOSITORY / "src/kcg_connector/config/carts_grasp_objects_v1.yaml"
CURRENT_OBJECT = "current_d38999_26kj61sn_public_spec"
TRANSFER_OBJECT = "te_deutsch_d38999_26fj35pn_step"


def _tetrahedron(
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (scale, 0.0, 0.0),
            (0.0, scale, 0.0),
            (0.0, 0.0, scale),
        ),
        dtype=np.float64,
    )
    vertices += np.asarray(offset, dtype=np.float64)
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 1, 3),
            (0, 3, 2),
            (1, 2, 3),
        ),
        dtype=np.int64,
    )
    return vertices, faces


def _apply_winding_signs(faces: np.ndarray, signs: tuple[int, ...]) -> np.ndarray:
    oriented = np.array(faces, dtype=np.int64, copy=True)
    flipped = np.flatnonzero(np.asarray(signs, dtype=np.int8) == -1)
    oriented[flipped, 1], oriented[flipped, 2] = (
        oriented[flipped, 2].copy(),
        oriented[flipped, 1].copy(),
    )
    return oriented


def test_whole_component_and_single_face_flips_are_oriented_without_epsilon() -> None:
    vertices, outward_faces = _tetrahedron()
    outward = audit_surface_orientation(
        vertices,
        outward_faces,
        role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
    )
    assert outward.components[0].positive_signed_six_volume == ExactDyadic(1, 0)
    assert outward.positive_volume_winding_sign_by_source_face == (1, 1, 1, 1)
    assert outward.components[0].source_winding_consistent
    assert (
        outward.components[0].source_consistent_winding_sign_to_positive_volume
        == 1
    )

    reversed_faces = outward_faces[:, (0, 2, 1)]
    reversed_certificate = audit_surface_orientation(
        vertices,
        reversed_faces,
        role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
    )
    assert reversed_certificate.components[0].positive_signed_six_volume == ExactDyadic(
        1, 0
    )
    assert reversed_certificate.positive_volume_winding_sign_by_source_face == (
        -1,
        -1,
        -1,
        -1,
    )
    assert (
        reversed_certificate.components[
            0
        ].source_consistent_winding_sign_to_positive_volume
        == -1
    )
    assert np.array_equal(
        _apply_winding_signs(
            reversed_faces,
            reversed_certificate.positive_volume_winding_sign_by_source_face,
        ),
        outward_faces,
    )

    locally_reversed = np.array(outward_faces, copy=True)
    locally_reversed[2, 1], locally_reversed[2, 2] = (
        locally_reversed[2, 2],
        locally_reversed[2, 1],
    )
    repaired = audit_surface_orientation(
        vertices,
        locally_reversed,
        role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
    )
    assert repaired.source_same_direction_edge_count == 3
    assert not repaired.components[0].source_winding_consistent
    assert (
        repaired.components[0].source_consistent_winding_sign_to_positive_volume
        == 0
    )
    assert np.array_equal(
        _apply_winding_signs(
            locally_reversed,
            repaired.positive_volume_winding_sign_by_source_face,
        ),
        outward_faces,
    )


def test_binary64_subnormal_signed_volume_is_exact_and_has_no_threshold() -> None:
    smallest_subnormal = float(
        np.nextafter(np.float64(0.0), np.float64(1.0))
    )
    vertices, faces = _tetrahedron(scale=smallest_subnormal)
    certificate = audit_surface_orientation(
        vertices,
        faces,
        role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
    )
    assert certificate.components[0].positive_signed_six_volume == ExactDyadic(
        1, -3222
    )


def test_open_and_non_manifold_source_index_topologies_fail_closed() -> None:
    vertices, faces = _tetrahedron()
    with pytest.raises(SurfaceOrientationAuditError) as open_error:
        audit_surface_orientation(
            vertices,
            faces[:-1],
            role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
        )
    assert open_error.value.code == "OPEN_SOURCE_INDEX_TOPOLOGY"
    assert "boundary_edge_count=3" in open_error.value.detail

    non_manifold_vertices = np.vstack((vertices, (0.0, -1.0, 0.5)))
    non_manifold_faces = np.vstack((faces, (0, 1, 4)))
    with pytest.raises(SurfaceOrientationAuditError) as non_manifold_error:
        audit_surface_orientation(
            non_manifold_vertices,
            non_manifold_faces,
            role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
        )
    assert non_manifold_error.value.code == "NON_MANIFOLD_SOURCE_INDEX_TOPOLOGY"
    assert "non_manifold_edge_count=1" in non_manifold_error.value.detail


def test_non_orientable_source_winding_constraints_fail_closed() -> None:
    # Minimal six-vertex triangulation of the real projective plane.  Every
    # source-indexed edge is incident to two faces, but no global face winding
    # can make all shared-edge directions opposite.
    parameters = np.arange(1.0, 7.0, dtype=np.float64)
    vertices = np.column_stack((parameters, parameters**2, parameters**3))
    faces = np.asarray(
        (
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 4),
            (0, 3, 5),
            (0, 4, 5),
            (1, 2, 5),
            (1, 3, 4),
            (1, 4, 5),
            (2, 3, 4),
            (2, 3, 5),
        ),
        dtype=np.int64,
    )
    with pytest.raises(SurfaceOrientationAuditError) as contradiction:
        audit_surface_orientation(
            vertices,
            faces,
            role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
        )
    assert contradiction.value.code == "WINDING_CONSTRAINT_CONTRADICTION"


def test_exact_zero_component_volume_fails_without_an_epsilon() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    _, tetrahedron_faces = _tetrahedron()
    with pytest.raises(SurfaceOrientationAuditError) as zero_volume:
        audit_surface_orientation(
            vertices,
            tetrahedron_faces,
            role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
        )
    assert zero_volume.value.code == "EXACT_ZERO_COMPONENT_SIGNED_VOLUME"


def test_component_order_and_certificate_digest_are_deterministic() -> None:
    first_vertices, first_faces = _tetrahedron(offset=(4.0, 0.0, 0.0))
    second_vertices, second_faces = _tetrahedron(offset=(-4.0, 0.0, 0.0))
    vertices = np.vstack((first_vertices, second_vertices))
    shifted_second_faces = second_faces + 4
    faces = np.empty((8, 3), dtype=np.int64)
    faces[0::2] = shifted_second_faces
    faces[1::2] = first_faces

    first = audit_surface_orientation(
        vertices,
        faces,
        role=SurfaceBoundaryRole.SOURCE_INDEXED_CLOSED_COMPONENT_SOUP,
    )
    second = audit_surface_orientation(
        vertices,
        faces,
        role=SurfaceBoundaryRole.SOURCE_INDEXED_CLOSED_COMPONENT_SOUP,
    )
    assert first == second
    assert [record.minimum_source_face_index for record in first.components] == [
        0,
        1,
    ]
    assert first.canonical_component_index_by_source_face == (
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    )
    assert len(first.canonical_sha256) == 64
    with pytest.raises(SurfaceOrientationAuditError) as wrong_role:
        audit_surface_orientation(
            vertices,
            faces,
            role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
        )
    assert wrong_role.value.code == "BOUNDARY_ROLE_COMPONENT_COUNT_MISMATCH"


def test_v1_certificate_is_immutable_and_never_claims_formal_outward_normals() -> None:
    vertices, faces = _tetrahedron()
    certificate = audit_surface_orientation(
        vertices,
        faces,
        role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
    )
    assert certificate.source_index_topology_status == VERIFIED
    assert certificate.closed_two_manifold_status == VERIFIED
    assert certificate.local_orientability_status == VERIFIED
    assert certificate.exact_dyadic_component_volume_status == VERIFIED
    assert certificate.component_positive_volume_orientation_status == VERIFIED
    assert certificate.self_intersection_status == UNVERIFIED
    assert certificate.nesting_parity_status == UNVERIFIED
    assert not certificate.formal_outward_eligible
    assert certificate.formal_outward_ineligibility_reasons == (
        "SELF_INTERSECTION_UNVERIFIED",
        "NESTING_PARITY_UNVERIFIED",
    )
    with pytest.raises(FrozenInstanceError):
        certificate.formal_outward_eligible = True  # type: ignore[misc]


def test_boundary_role_and_binary64_payload_must_be_explicit() -> None:
    vertices, faces = _tetrahedron()
    with pytest.raises(SurfaceOrientationAuditError) as role_error:
        audit_surface_orientation(
            vertices,
            faces,
            role="SINGLE_CLOSED_BOUNDARY",  # type: ignore[arg-type]
        )
    assert role_error.value.code == "EXPLICIT_BOUNDARY_ROLE_REQUIRED"

    with pytest.raises(SurfaceOrientationAuditError) as dtype_error:
        audit_surface_orientation(
            vertices.astype(np.float32),
            faces,
            role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
        )
    assert dtype_error.value.code == "BINARY64_VERTICES_REQUIRED"


@pytest.mark.parametrize(
    ("object_id", "role", "expected"),
    (
        (
            CURRENT_OBJECT,
            SurfaceBoundaryRole.SOURCE_INDEXED_CLOSED_COMPONENT_SOUP,
            {
                "vertices": 88078,
                "faces": 145588,
                "edges": 218382,
                "components": 7642,
                "component_face_histogram": {
                    8: 1985,
                    12: 5264,
                    16: 378,
                    284: 3,
                    4970: 12,
                },
            },
        ),
        (
            TRANSFER_OBJECT,
            SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
            {
                "vertices": 343520,
                "faces": 687036,
                "edges": 1030554,
                "components": 1,
                "component_face_histogram": {687036: 1},
            },
        ),
    ),
)
def test_real_source_indexed_meshes_have_frozen_linear_topology_counts(
    object_id: str,
    role: SurfaceBoundaryRole,
    expected: dict[str, object],
) -> None:
    loaded = load_object_contract(
        OBJECT_CONTRACT,
        object_id=object_id,
        repository_root=REPOSITORY,
    )
    certificate = loaded.orientation_certificate
    assert certificate.role is role
    assert certificate.source_vertex_count == expected["vertices"]
    assert certificate.source_face_count == expected["faces"]
    assert certificate.source_edge_count == expected["edges"]
    assert certificate.component_count == expected["components"]
    assert certificate.boundary_edge_count == 0
    assert certificate.non_manifold_edge_count == 0
    assert certificate.winding_constraint_contradiction_count == 0
    assert certificate.source_same_direction_edge_count == 0
    assert Counter(
        record.source_face_count for record in certificate.components
    ) == expected["component_face_histogram"]
    assert all(
        record.positive_signed_six_volume.sign == 1
        for record in certificate.components
    )
    assert certificate.self_intersection_status == UNVERIFIED
    assert certificate.nesting_parity_status == UNVERIFIED
    assert not certificate.formal_outward_eligible
