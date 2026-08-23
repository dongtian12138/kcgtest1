from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.positive_solid_union import (
    CLAIM_LIMITATIONS,
    MATERIAL_OPERATION,
    PositiveSolidUnionError,
    bind_positive_usda_component_ids,
    certify_positive_solid_component_union,
    parse_usda_gprim_solid_roles,
)


ROLE_SHA256 = "1" * 64
ASSET_SHA256 = "2" * 64


def _tetrahedron(
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    ) + np.asarray(offset, dtype=np.float64)
    faces = np.asarray(
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        dtype=np.int64,
    )
    return vertices, faces


def _two_tetrahedra(
    second_offset: tuple[float, float, float] = (0.25, 0.25, 0.25),
) -> tuple[np.ndarray, np.ndarray]:
    first_vertices, first_faces = _tetrahedron()
    second_vertices, second_faces = _tetrahedron(second_offset)
    return (
        np.vstack((first_vertices, second_vertices)),
        np.vstack((first_faces, second_faces + 4)),
    )


def _certify(vertices: np.ndarray, faces: np.ndarray):
    return certify_positive_solid_component_union(
        vertices,
        faces,
        source_asset_sha256=ASSET_SHA256,
        source_component_ids=("/Source/A", "/Source/B"),
        positive_solid_component_ids=("/Source/A", "/Source/B"),
        role_authority_kind="TEST_EXPLICIT_POSITIVE_SOLID_ROLE_V1",
        role_authority_sha256=ROLE_SHA256,
    )


def test_overlapping_closed_components_are_a_positive_solid_union() -> None:
    vertices, faces = _two_tetrahedra()
    first = _certify(vertices, faces)
    second = _certify(vertices, faces)

    assert first == second
    assert first.material_operation == MATERIAL_OPERATION
    assert first.source_component_count == 2
    assert first.embedded_positive_solid_count == 2
    assert first.source_face_count == 8
    assert first.within_component_source_face_pair_count == 12
    assert first.inter_component_face_pair_count == 16
    assert first.component_face_count_histogram == ((4, 2),)
    assert first.formal_material_boundary_eligible is True
    assert first.claim_limitations == CLAIM_LIMITATIONS
    assert len(first.certificate_sha256) == 64
    assert first.as_dict()["certificate_sha256"] == first.certificate_sha256
    with pytest.raises(FrozenInstanceError):
        first.formal_material_boundary_eligible = False  # type: ignore[misc]
    with pytest.raises(ValueError):
        replace(first, inter_component_face_pair_count=0)


def test_positive_role_inventory_must_cover_every_component_exactly() -> None:
    vertices, faces = _two_tetrahedra((3.0, 0.0, 0.0))
    with pytest.raises(PositiveSolidUnionError) as missing:
        certify_positive_solid_component_union(
            vertices,
            faces,
            source_asset_sha256=ASSET_SHA256,
            source_component_ids=("/Source/A", "/Source/B"),
            positive_solid_component_ids=("/Source/A",),
            role_authority_kind="TEST_EXPLICIT_POSITIVE_SOLID_ROLE_V1",
            role_authority_sha256=ROLE_SHA256,
        )
    assert missing.value.code == "INCOMPLETE_POSITIVE_SOLID_ROLE_COVERAGE"

    with pytest.raises(PositiveSolidUnionError) as duplicate:
        certify_positive_solid_component_union(
            vertices,
            faces,
            source_asset_sha256=ASSET_SHA256,
            source_component_ids=("/Source/A", "/Source/A"),
            positive_solid_component_ids=("/Source/A",),
            role_authority_kind="TEST_EXPLICIT_POSITIVE_SOLID_ROLE_V1",
            role_authority_sha256=ROLE_SHA256,
        )
    assert duplicate.value.code == "INVALID_COMPONENT_IDENTITY_INVENTORY"


def test_interleaved_component_faces_are_rejected_as_ambiguous_source_order() -> None:
    vertices, faces = _two_tetrahedra((3.0, 0.0, 0.0))
    interleaved = np.empty_like(faces)
    interleaved[0::2] = faces[:4]
    interleaved[1::2] = faces[4:]
    with pytest.raises(PositiveSolidUnionError) as error:
        _certify(vertices, interleaved)
    assert error.value.code == "NONCONTIGUOUS_SOURCE_COMPONENT_FACE_RANGE"


def test_one_self_intersecting_component_fails_the_whole_union() -> None:
    first_vertices, first_faces = _tetrahedron()
    cube_vertices = np.asarray(
        (
            (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
            (0.0, 0.0, -2.0), (-1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    ) + np.asarray((4.0, 0.0, 0.0), dtype=np.float64)
    cube_faces = np.asarray(
        (
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        ),
        dtype=np.int64,
    )
    vertices = np.vstack((first_vertices, cube_vertices))
    faces = np.vstack((first_faces, cube_faces + 4))
    with pytest.raises(PositiveSolidUnionError) as error:
        _certify(vertices, faces)
    assert error.value.code == "COMPONENT_MATERIAL_BOUNDARY_FAILURE"
    assert "component_index=1" in error.value.detail


def test_usda_roles_are_bound_by_exact_prim_path_and_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "roles.usda"
    source.write_text(
        """#usda 1.0
def Xform "World"
{
    def Mesh "Good"
    {
        custom bool kcg:closedManifold = 1
        custom bool kcg:positiveVolume = 1
    }
    def Mesh "Void"
    {
        custom bool kcg:closedManifold = 1
        custom bool kcg:positiveVolume = 0
    }
}
""",
        encoding="utf-8",
    )
    roles = parse_usda_gprim_solid_roles(source)
    assert bind_positive_usda_component_ids(("/World/Good",), roles) == (
        "/World/Good",
    )
    with pytest.raises(PositiveSolidUnionError) as subtractive:
        bind_positive_usda_component_ids(("/World/Void",), roles)
    assert subtractive.value.code == "SOURCE_COMPONENT_NOT_POSITIVE_SOLID"
    with pytest.raises(PositiveSolidUnionError) as missing:
        bind_positive_usda_component_ids(("/World/Missing",), roles)
    assert missing.value.code == "SOURCE_COMPONENT_ROLE_MISSING"
