from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.object_model import (
    AssetProvenance,
    CARTS_VISUAL_SUBTREE_NPZ,
    ObjectGraspModel,
    TriangleMesh,
)
from kcg_connector.grasp.robust.surface_sampling import (
    PadNormalCone,
    eligible_lateral_face_mask,
)
from kcg_connector.grasp.robust.surface_visibility import (
    DirectionalFirstHitVisibilityPredicate,
    EXTERNAL_FIRST_HIT_METHOD,
    NUMERICAL_POLICY,
    TriangleFirstHitIntersector,
    external_first_hit_face_visibility,
)


def _model(vertices: np.ndarray, faces: np.ndarray) -> ObjectGraspModel:
    semantics = tuple("external_surface" for _ in range(len(faces)))
    mesh = TriangleMesh(vertices, faces, semantics)
    digest = hashlib.sha256()
    digest.update(np.asarray(vertices, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(faces, dtype="<i8").tobytes(order="C"))
    provenance = AssetProvenance(
        source_path=str(Path("/synthetic/visibility_fixture.npz")),
        source_sha256=digest.hexdigest(),
        source_class="SYNTHETIC_GEOMETRY_TEST",
        source_format=CARTS_VISUAL_SUBTREE_NPZ,
        source_unit="m",
        meters_per_source_unit=1.0,
    )
    return ObjectGraspModel(
        mesh=mesh,
        provenance=provenance,
        assembly_axis=np.asarray((0.0, 0.0, 1.0)),
        mass_kg=1.0,
        center_of_mass_m=np.mean(vertices, axis=0),
        inertia_kg_m2=np.eye(3),
        allowed_contact_semantics=frozenset(("external_surface",)),
    )


def _box() -> tuple[np.ndarray, np.ndarray]:
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
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        ),
        dtype=np.int64,
    )
    return vertices, faces


def _append_quad(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z: float,
) -> tuple[int, int]:
    start_vertex = len(vertices)
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    vertices.extend(
        ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))
    )
    start_face = len(faces)
    faces.extend(
        (
            (start_vertex, start_vertex + 1, start_vertex + 2),
            (start_vertex, start_vertex + 2, start_vertex + 3),
        )
    )
    return start_face, start_face + 2


def _open_recess_and_hidden_layer(
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    _append_quad(vertices, faces, (-3.0, -1.0), (-1.0, 1.0), 0.0)
    hidden = _append_quad(vertices, faces, (-3.0, -1.0), (-1.0, 1.0), 1.0)

    # Four strips form a front plane with a finite central opening.
    _append_quad(vertices, faces, (0.0, 1.0), (-1.5, 1.5), 0.0)
    _append_quad(vertices, faces, (2.0, 3.0), (-1.5, 1.5), 0.0)
    _append_quad(vertices, faces, (1.0, 2.0), (-1.5, -0.5), 0.0)
    _append_quad(vertices, faces, (1.0, 2.0), (0.5, 1.5), 0.0)
    recess = _append_quad(vertices, faces, (1.1, 1.9), (-0.4, 0.4), 1.0)
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        tuple(range(*hidden)),
        tuple(range(*recess)),
    )


def test_convex_box_keeps_only_the_entry_faces_and_adapts_surface_sampling() -> None:
    vertices, faces = _box()
    model = _model(vertices, faces)
    predicate = DirectionalFirstHitVisibilityPredicate(((1.0, 0.0, 0.0),))

    result = predicate.evaluate(model)

    assert result.audit.method_id == EXTERNAL_FIRST_HIT_METHOD
    assert result.audit.numerical_policy == NUMERICAL_POLICY
    assert result.audit.ray_origin_padding_m > 0.0
    assert result.audit.distance_error_bound_m > 0.0
    assert result.audit.visible_face_count == 2
    assert np.array_equal(np.flatnonzero(result.face_mask), (8, 9))
    assert result.face_mask.flags.writeable is False
    assert np.array_equal(predicate(model), result.face_mask)

    # One homogeneous hand-derived half-space is enough for this fixture; the
    # visibility object is passed unchanged through the public callback API.
    cone = PadNormalCone(
        halfspaces_local=np.asarray(((0.0, 0.0, 1.0),)),
        source="synthetic_hand_kinematic_domain",
    )
    selected = eligible_lateral_face_mask(
        model,
        cone,
        visibility_predicate=predicate,
    )
    assert np.array_equal(np.flatnonzero(selected), (8, 9))


def test_open_recess_is_reachable_but_occluded_inner_layer_is_removed() -> None:
    vertices, faces, hidden_faces, recess_faces = _open_recess_and_hidden_layer()
    model = _model(vertices, faces)

    result = external_first_hit_face_visibility(model, ((0.0, 0.0, 7.0),))

    assert np.all(result.face_mask[np.asarray(recess_faces)])
    assert not np.any(result.face_mask[np.asarray(hidden_faces)])
    assert result.audit.rays_with_no_hit == 0
    assert result.audit.ray_triangle_tests > 0


def test_visibility_is_equivariant_under_a_common_rigid_transform() -> None:
    vertices, faces, _hidden_faces, _recess_faces = _open_recess_and_hidden_layer()
    model = _model(vertices, faces)
    direction = np.asarray((0.0, 0.0, 1.0))
    reference = external_first_hit_face_visibility(model, (direction,))

    yaw = math.radians(31.0)
    pitch = math.radians(-23.0)
    rz = np.asarray(
        (
            (math.cos(yaw), -math.sin(yaw), 0.0),
            (math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    ry = np.asarray(
        (
            (math.cos(pitch), 0.0, math.sin(pitch)),
            (0.0, 1.0, 0.0),
            (-math.sin(pitch), 0.0, math.cos(pitch)),
        )
    )
    rotation = rz @ ry
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = (4.75, -3.25, 2.5)
    transformed = external_first_hit_face_visibility(
        model.transformed(transform),
        (rotation @ direction,),
    )

    assert np.array_equal(transformed.face_mask, reference.face_mask)
    assert transformed.audit.visible_face_count == reference.audit.visible_face_count
    assert transformed.audit.characteristic_length_m == pytest.approx(
        reference.audit.characteristic_length_m,
        rel=64.0 * np.finfo(np.float64).eps,
        abs=0.0,
    )


def test_direction_and_face_order_do_not_change_geometric_classification() -> None:
    vertices, faces = _box()
    model = _model(vertices, faces)
    directions_first = ((0.0, 0.0, 3.0), (2.0, 0.0, 0.0))
    directions_second = ((5.0, 0.0, 0.0), (0.0, 0.0, 11.0))
    first = external_first_hit_face_visibility(model, directions_first)
    second = external_first_hit_face_visibility(model, directions_second)

    assert np.array_equal(first.face_mask, second.face_mask)
    assert (
        first.audit.canonical_directions_sha256
        == second.audit.canonical_directions_sha256
    )
    assert first.audit.visible_face_count_by_direction == (
        second.audit.visible_face_count_by_direction
    )

    permutation = np.asarray((7, 2, 11, 0, 8, 4, 10, 1, 9, 6, 3, 5))
    permuted_model = _model(vertices, faces[permutation])
    permuted = external_first_hit_face_visibility(permuted_model, directions_first)
    remapped = np.empty_like(permuted.face_mask)
    remapped[permutation] = permuted.face_mask
    assert np.array_equal(remapped, first.face_mask)


def test_invalid_directions_fail_closed_without_a_fallback() -> None:
    vertices, faces = _box()
    model = _model(vertices, faces)

    with pytest.raises(ValueError, match="non-empty"):
        external_first_hit_face_visibility(model, ())
    with pytest.raises(ValueError, match="non-zero"):
        external_first_hit_face_visibility(model, ((0.0, 0.0, 0.0),))
    with pytest.raises(ValueError, match="non-finite"):
        external_first_hit_face_visibility(model, ((math.nan, 0.0, 1.0),))


def test_reusable_intersector_reports_first_surface_and_deterministic_tie() -> None:
    vertices, faces = _box()
    model = _model(vertices, faces)
    intersector = TriangleFirstHitIntersector(model)

    hit = intersector.first_hit((-3.0, 0.25, -0.25), (7.0, 0.0, 0.0))
    repeated = intersector.first_hit((-3.0, 0.25, -0.25), (1.0, 0.0, 0.0))

    assert hit.hit
    assert hit.face_index in (8, 9)
    assert hit.distance_m == pytest.approx(2.0)
    assert np.allclose(hit.position_m, (-1.0, 0.25, -0.25))
    assert np.allclose(hit.outward_normal, (-1.0, 0.0, 0.0))
    assert hit.face_index == repeated.face_index
    assert hit.distance_m == repeated.distance_m
    assert hit.distance_error_bound_m > 0.0
    assert hit.ray_triangle_tests > 0
    assert hit.position_m.flags.writeable is False

    miss = intersector.first_hit((-3.0, 4.0, 0.0), (1.0, 0.0, 0.0))
    assert not miss.hit
    assert miss.face_index is None
    assert miss.distance_m is None


def test_reusable_intersector_is_rigid_transform_equivariant() -> None:
    vertices, faces = _box()
    model = _model(vertices, faces)
    origin = np.asarray((-3.0, 0.25, -0.25))
    direction = np.asarray((1.0, 0.0, 0.0))
    reference = TriangleFirstHitIntersector(model).first_hit(origin, direction)

    angle = 0.63
    rotation = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    translation = np.asarray((2.0, -4.0, 0.75))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transformed = TriangleFirstHitIntersector(model.transformed(transform)).first_hit(
        rotation @ origin + translation,
        rotation @ direction,
    )

    assert reference.hit and transformed.hit
    assert transformed.face_index == reference.face_index
    assert transformed.distance_m == pytest.approx(
        reference.distance_m,
        rel=128.0 * np.finfo(np.float64).eps,
    )
    assert np.allclose(
        transformed.position_m,
        rotation @ reference.position_m + translation,
        rtol=0.0,
        atol=256.0 * np.finfo(np.float64).eps * np.linalg.norm(translation),
    )
    assert np.allclose(transformed.outward_normal, rotation @ reference.outward_normal)


def test_reusable_intersector_batch_shape_and_invalid_rays_fail_closed() -> None:
    vertices, faces = _box()
    intersector = TriangleFirstHitIntersector(_model(vertices, faces))
    hits = intersector.first_hits(
        ((-3.0, 0.0, 0.0), (0.0, -3.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    assert len(hits) == 2
    assert all(hit.hit for hit in hits)

    with pytest.raises(ValueError, match="match origins"):
        intersector.first_hits(((-3.0, 0.0, 0.0),), ((1.0, 0.0, 0.0),) * 2)
    with pytest.raises(ValueError, match="non-zero"):
        intersector.first_hit((-3.0, 0.0, 0.0), (0.0, 0.0, 0.0))
