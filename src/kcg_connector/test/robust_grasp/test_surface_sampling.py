"""Winding-independent task-frame surface-sampling certificates."""

from __future__ import annotations

from itertools import permutations
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
    UNORIENTED_CANONICAL_REPRESENTATIVE_NORMAL_DIAGNOSTIC_ONLY,
    eligible_lateral_face_mask,
    extract_lateral_contact_patches,
    sample_mesh_faces_area_stratified,
)
from kcg_connector.grasp.robust.triangle_canonicalization import (
    RegisteredTaskFrame,
    TriangleCanonicalizationError,
    canonicalize_unoriented_triangles,
)


_VERTICES = np.asarray(
    (
        (2.0, -1.0, -1.0),
        (2.0, 1.0, -1.0),
        (2.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


def _gamma(operation_count: int) -> float:
    unit_roundoff = 0.5 * np.finfo(np.float64).eps
    product = float(operation_count) * unit_roundoff
    return product / (1.0 - product)


def _model(permutation: tuple[int, int, int]) -> ObjectGraspModel:
    mesh = TriangleMesh(
        vertices_m=_VERTICES,
        faces=np.asarray((permutation,), dtype=np.int64),
        face_semantics=("grip_surface",),
    )
    provenance = AssetProvenance(
        source_path=str(Path("/synthetic/task_frame_triangle_fixture.npz")),
        source_sha256="7" * 64,
        source_class="SYNTHETIC_UNORIENTED_TRIANGLE_TEST_FIXTURE",
        source_format=CARTS_VISUAL_SUBTREE_NPZ,
        source_unit="m",
        meters_per_source_unit=1.0,
    )
    return ObjectGraspModel(
        mesh=mesh,
        provenance=provenance,
        assembly_axis=np.asarray((0.0, 0.0, 1.0)),
        assembly_axis_origin_m=np.zeros(3),
        mass_kg=0.2,
        center_of_mass_m=np.asarray((2.0, 0.0, -1.0 / 3.0)),
        inertia_kg_m2=np.diag((0.1, 0.2, 0.3)),
        allowed_contact_semantics=frozenset(("grip_surface",)),
    )


def _nontrivial_task_frame() -> RegisteredTaskFrame:
    yaw = math.radians(19.0)
    basis = np.asarray(
        (
            (math.cos(yaw), -math.sin(yaw), 0.0),
            (math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return RegisteredTaskFrame(
        origin_object_m=np.asarray((0.25, -0.5, 0.75)),
        basis_object=basis,
        source="SYNTHETIC_PRE_REGISTERED_TASK_DATUM",
    )


def _positive_radial_asymmetric_cone() -> PadNormalCone:
    # A @ n <= 0 accepts +radial but rejects -radial.  This deliberately is
    # not a symmetric cone, so abs(A @ n) cannot satisfy the test by accident.
    return PadNormalCone(
        halfspaces_local=np.asarray(((-1.0, 0.0, 0.0),)),
        source="SYNTHETIC_ASYMMETRIC_PAD_CONE",
    )


def test_all_six_vertex_permutations_have_exact_canonical_samples() -> None:
    task_frame = _nontrivial_task_frame()
    reference_triangles: np.ndarray | None = None
    reference_positions: np.ndarray | None = None
    reference_barycentric: np.ndarray | None = None
    reference_normals: np.ndarray | None = None

    for permutation in permutations((0, 1, 2)):
        model = _model(permutation)
        canonical = canonicalize_unoriented_triangles(
            model.mesh.face_vertices_m,
            task_frame=task_frame,
        )
        samples = sample_mesh_faces_area_stratified(
            model,
            task_frame=task_frame,
            face_indices=(0,),
            sample_count=32,
            seed=1943,
        )
        assert canonical.flags.writeable is False
        assert samples.positions_m.flags.writeable is False
        assert samples.normals.flags.writeable is False
        assert samples.normal_semantics == (
            UNORIENTED_CANONICAL_REPRESENTATIVE_NORMAL_DIAGNOSTIC_ONLY
        )
        if reference_triangles is None:
            reference_triangles = canonical
            reference_positions = samples.positions_m
            reference_barycentric = samples.barycentric_coordinates
            reference_normals = samples.normals
            continue
        assert np.array_equal(canonical, reference_triangles)
        assert np.array_equal(samples.positions_m, reference_positions)
        assert np.array_equal(
            samples.barycentric_coordinates,
            reference_barycentric,
        )
        assert np.array_equal(samples.normals, reference_normals)


def test_asymmetric_cone_queries_both_normal_orientations_separately() -> None:
    cone = _positive_radial_asymmetric_cone()
    directed = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    assert np.array_equal(cone.contains(directed), (True, False))
    assert np.array_equal(cone.intersects_normal_lines(directed), (True, True))

    reference_second_moment: np.ndarray | None = None
    for permutation in permutations((0, 1, 2)):
        model = _model(permutation)
        mask = eligible_lateral_face_mask(
            model,
            cone,
            require_watertight=False,
        )
        assert np.array_equal(mask, (True,))
        patches = extract_lateral_contact_patches(
            model,
            cone,
            require_watertight=False,
        )
        assert len(patches) == 1
        second_moment = patches[0].area_normal_second_moment_m2
        assert second_moment.flags.writeable is False
        assert np.trace(second_moment) == pytest.approx(patches[0].area_m2)
        if reference_second_moment is None:
            reference_second_moment = second_moment
        else:
            assert np.array_equal(second_moment, reference_second_moment)


def test_samples_are_equivariant_under_common_proper_se3_transform() -> None:
    model = _model((2, 0, 1))
    task_frame = _nontrivial_task_frame()
    reference = sample_mesh_faces_area_stratified(
        model,
        task_frame=task_frame,
        face_indices=(0,),
        sample_count=64,
        seed=811,
    )

    yaw, pitch, roll = (
        math.radians(31.0),
        math.radians(-23.0),
        math.radians(11.0),
    )
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
    rx = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(roll), -math.sin(roll)),
            (0.0, math.sin(roll), math.cos(roll)),
        )
    )
    rotation = rz @ ry @ rx
    translation = np.asarray((3.25, -4.75, 2.5))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    transformed = sample_mesh_faces_area_stratified(
        model.transformed(transform),
        task_frame=task_frame.transformed(transform),
        face_indices=(0,),
        sample_count=64,
        seed=811,
    )
    assert np.array_equal(transformed.face_indices, reference.face_indices)
    assert np.array_equal(
        transformed.barycentric_coordinates,
        reference.barycentric_coordinates,
    )
    expected_positions = reference.positions_m @ rotation.T + translation
    position_scale = max(
        1.0,
        float(np.max(np.abs(reference.positions_m))),
        float(np.max(np.abs(expected_positions))),
        float(np.max(np.abs(transformed.positions_m))),
    )
    # At most 12 operations occur on either side of the common-SE(3)
    # comparison (barycentric interpolation plus rigid transform).
    position_error_bound = _gamma(24) * position_scale
    assert float(
        np.max(np.abs(transformed.positions_m - expected_positions))
    ) <= position_error_bound

    expected_normals = reference.normals @ rotation.T
    # The 128-operation envelope covers both finite-precision paths through
    # transformed vertices, two edges, scaled cross product, and unit normal.
    normal_error_bound = _gamma(128) * max(
        1.0,
        float(np.linalg.norm(rotation, ord=np.inf)),
    )
    assert float(
        np.max(np.abs(transformed.normals - expected_normals))
    ) <= normal_error_bound


def test_nontransitive_interval_overlap_fails_closed() -> None:
    root_half = math.sqrt(0.5)
    task_frame = RegisteredTaskFrame(
        origin_object_m=np.zeros(3),
        basis_object=np.asarray(
            (
                (root_half, -root_half, 0.0),
                (root_half, root_half, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        source="SYNTHETIC_CANCELLATION_TASK_DATUM",
    )
    triangle = np.asarray(
        (
            (1.0e16, -1.0e16, 0.0),
            (1.0e16, -1.0e16 + 60.0, 1.0),
            (1.0e16, -1.0e16 + 120.0, 0.0),
        )
    )
    with pytest.raises(
        TriangleCanonicalizationError,
        match="non-transitive overlap",
    ):
        canonicalize_unoriented_triangles(
            triangle[None, :, :],
            task_frame=task_frame,
        )


def test_missing_or_improper_task_frame_fails_closed() -> None:
    model = _model((0, 1, 2))
    with pytest.raises(TypeError, match="task_frame"):
        sample_mesh_faces_area_stratified(
            model,
            face_indices=(0,),
            sample_count=1,
            seed=0,
        )
    with pytest.raises(TriangleCanonicalizationError, match=r"SO\(3\)"):
        RegisteredTaskFrame(
            origin_object_m=np.zeros(3),
            basis_object=np.diag((1.0, 1.0, -1.0)),
            source="IMPROPER_FRAME_MUST_FAIL",
        )
