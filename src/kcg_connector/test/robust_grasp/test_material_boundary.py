from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.material_boundary import (
    CLAIM_LIMITATIONS,
    MaterialBoundaryError,
    certify_single_embedded_material_boundary,
)
from kcg_connector.grasp.robust.object_model import load_stl_mesh
from kcg_connector.grasp.robust.surface_orientation import (
    SurfaceOrientationAuditError,
)


REPOSITORY = Path(__file__).resolve().parents[4]
TERMINAL_LINKS = ("f1Link3", "f2Link2", "f3Link3")


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
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


def _triangulated_cube() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ),
        dtype=np.int64,
    )
    return vertices, faces


def test_tetrahedron_produces_complete_exact_pair_certificate() -> None:
    vertices, faces = _tetrahedron()
    certificate = certify_single_embedded_material_boundary(vertices, faces)

    assert certificate.source_vertex_count == 4
    assert certificate.source_face_count == 4
    assert certificate.source_face_pair_count == 6
    assert certificate.pair_coverage_count == 6
    assert certificate.allowed_shared_edge_pair_count == 6
    assert certificate.allowed_shared_vertex_pair_count == 0
    assert certificate.self_intersection_count == 0
    assert certificate.formal_material_boundary_eligible is True
    assert certificate.claim_limitations == CLAIM_LIMITATIONS
    assert len(certificate.certificate_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        certificate.formal_material_boundary_eligible = False  # type: ignore[misc]
    with pytest.raises(ValueError):
        replace(certificate, pair_coverage_count=5)


def test_open_surface_still_fails_at_the_topology_boundary() -> None:
    vertices, faces = _tetrahedron()
    with pytest.raises(SurfaceOrientationAuditError) as error:
        certify_single_embedded_material_boundary(vertices, faces[:-1])
    assert error.value.code == "OPEN_SOURCE_INDEX_TOPOLOGY"


def test_duplicate_or_unused_source_vertex_is_rejected() -> None:
    vertices, faces = _tetrahedron()
    duplicate = np.vstack((vertices, vertices[0]))
    with pytest.raises(MaterialBoundaryError) as duplicate_error:
        certify_single_embedded_material_boundary(duplicate, faces)
    assert duplicate_error.value.code == "DUPLICATE_GEOMETRIC_VERTEX_INDEX"

    unused = np.vstack((vertices, (2.0, 2.0, 2.0)))
    with pytest.raises(MaterialBoundaryError) as unused_error:
        certify_single_embedded_material_boundary(unused, faces)
    assert unused_error.value.code == "UNUSED_SOURCE_VERTEX"


def test_coplanar_face_triangulation_is_only_shared_edge_contact() -> None:
    vertices, faces = _triangulated_cube()
    certificate = certify_single_embedded_material_boundary(vertices, faces)

    assert certificate.source_face_count == 12
    assert certificate.source_face_pair_count == 66
    assert certificate.pair_coverage_count == 66
    assert certificate.allowed_shared_edge_pair_count == 18
    assert certificate.self_intersection_count == 0


def test_closed_two_manifold_with_a_vertex_pulled_through_surface_is_rejected() -> None:
    vertices, faces = _triangulated_cube()
    crossed = np.array(vertices, copy=True)
    crossed[6] = (0.0, 0.0, -2.0)
    with pytest.raises(MaterialBoundaryError) as error:
        certify_single_embedded_material_boundary(crossed, faces)
    assert error.value.code == "SOURCE_TRIANGLE_SELF_INTERSECTION"
    assert "first_face=" in error.value.detail
    assert "second_face=" in error.value.detail


@pytest.mark.parametrize("link_name", TERMINAL_LINKS)
def test_real_terminal_collision_mesh_is_one_exact_embedded_boundary(
    link_name: str,
) -> None:
    path = (
        REPOSITORY
        / f"src/iiwa_description/meshes/hand/collision/{link_name}_convex.stl"
    )
    mesh, _provenance = load_stl_mesh(
        path,
        unit="m",
        orient_outward=False,
    )
    first = certify_single_embedded_material_boundary(
        mesh.vertices_m, mesh.faces
    )
    second = certify_single_embedded_material_boundary(
        mesh.vertices_m, mesh.faces
    )

    assert first == second
    assert first.source_vertex_count == 552
    assert first.source_face_count == 1100
    assert first.source_face_pair_count == 604450
    assert first.pair_coverage_count == 604450
    assert first.formal_material_boundary_eligible is True
