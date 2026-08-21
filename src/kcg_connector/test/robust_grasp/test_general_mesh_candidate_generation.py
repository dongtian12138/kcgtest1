"""Object-agnostic geometry tests for the CARTS-Grasp surface layer."""

from __future__ import annotations

import math
from pathlib import Path
import struct

import numpy as np
import pytest

from kcg_connector.grasp.robust.object_model import ObjectGraspModel
from kcg_connector.grasp.robust.surface_sampling import (
    AREA_STRATIFIED_SOBOL,
    PadNormalCone,
    eligible_lateral_face_mask,
    extract_lateral_contact_patches,
    sample_contact_surfaces,
)
from kcg_connector.grasp.robust.triangle_canonicalization import (
    RegisteredTaskFrame,
)


def _box_triangles(half_extent: float) -> np.ndarray:
    h = float(half_extent)
    vertices = np.asarray(
        (
            (-h, -h, -h),
            (+h, -h, -h),
            (+h, +h, -h),
            (-h, +h, -h),
            (-h, -h, +h),
            (+h, -h, +h),
            (+h, +h, +h),
            (-h, +h, +h),
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
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ),
        dtype=np.int64,
    )
    return vertices[faces]


def _write_ascii_stl(path: Path, triangles: np.ndarray) -> None:
    lines = ["solid object_agnostic_fixture"]
    for triangle in triangles:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        lines.append(f"  facet normal {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}")
        lines.append("    outer loop")
        for vertex in triangle:
            lines.append(f"      vertex {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}")
        lines.extend(("    endloop", "  endfacet"))
    lines.append("endsolid object_agnostic_fixture")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    payload = bytearray(b"CARTS-Grasp deterministic fixture".ljust(80, b"\0"))
    payload.extend(struct.pack("<I", len(triangles)))
    for triangle in triangles:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        values = tuple(float(value) for value in normal)
        values += tuple(float(value) for value in triangle.reshape(-1))
        payload.extend(struct.pack("<12fH", *values, 0))
    path.write_bytes(bytes(payload))


def _model(path: Path, *, unit: str, semantics: tuple[str, ...]) -> ObjectGraspModel:
    return ObjectGraspModel.from_stl(
        path,
        unit=unit,
        source_class="SYNTHETIC_CLOSED_SOLID_TEST_FIXTURE",
        assembly_axis=(0.0, 0.0, 1.0),
        assembly_axis_origin_m=(0.0, 0.0, 0.0),
        mass_kg=0.12,
        center_of_mass_m=(0.0, 0.0, 0.0),
        inertia_kg_m2=np.diag((2.0e-4, 3.0e-4, 4.0e-4)),
        face_semantics=semantics,
        allowed_contact_semantics=("grip_surface",),
        forbidden_contact_semantics=("mating_surface",),
        require_watertight=True,
    )


def _fixture_pad_normal_cone() -> PadNormalCone:
    # This aperture belongs to the synthetic hand fixture, not to the object.
    tangent_of_pad_aperture = math.tan(math.radians(40.0))
    return PadNormalCone(
        halfspaces_local=np.asarray(
            (
                (-tangent_of_pad_aperture, +1.0, 0.0),
                (-tangent_of_pad_aperture, -1.0, 0.0),
                (-tangent_of_pad_aperture, 0.0, +1.0),
                (-tangent_of_pad_aperture, 0.0, -1.0),
                (-1.0, 0.0, 0.0),
            ),
            dtype=np.float64,
        ),
        source="SYNTHETIC_HAND_CONTACT_MODEL_PAD_ORIENTATION_CONE",
    )


def _registered_task_frame() -> RegisteredTaskFrame:
    return RegisteredTaskFrame(
        origin_object_m=np.zeros(3),
        basis_object=np.eye(3),
        source="SYNTHETIC_FIXTURE_REGISTERED_TASK_DATUM",
    )


def _gamma(operation_count: int) -> float:
    unit_roundoff = 0.5 * np.finfo(np.float64).eps
    product = float(operation_count) * unit_roundoff
    return product / (1.0 - product)


def _semantics() -> tuple[str, ...]:
    return ("mating_surface",) * 4 + ("grip_surface",) * 8


def test_ascii_and_binary_stl_are_deterministic_and_provenance_bound(
    tmp_path: Path,
) -> None:
    triangles_mm = _box_triangles(1000.0)
    ascii_path = tmp_path / "fixture_ascii.stl"
    binary_path = tmp_path / "fixture_binary.stl"
    _write_ascii_stl(ascii_path, triangles_mm)
    _write_binary_stl(binary_path, triangles_mm)

    ascii_model = _model(ascii_path, unit="mm", semantics=_semantics())
    binary_model = _model(binary_path, unit="mm", semantics=_semantics())
    repeated_model = _model(ascii_path, unit="mm", semantics=_semantics())

    assert ascii_model.provenance.source_format == "ASCII_STL"
    assert binary_model.provenance.source_format == "BINARY_STL"
    assert ascii_model.provenance.source_sha256 != binary_model.provenance.source_sha256
    assert ascii_model.provenance.meters_per_source_unit == 1.0e-3
    assert ascii_model.provenance["source_class"] == "SYNTHETIC_CLOSED_SOLID_TEST_FIXTURE"
    assert ascii_model.mesh.is_watertight
    assert binary_model.mesh.is_watertight
    assert np.array_equal(ascii_model.mesh.faces, binary_model.mesh.faces)
    assert np.array_equal(ascii_model.mesh.vertices_m, binary_model.mesh.vertices_m)
    assert ascii_model.geometry_sha256 == binary_model.geometry_sha256
    assert ascii_model.geometry_sha256 == repeated_model.geometry_sha256

    with pytest.raises(ValueError, match="implicit STL unit"):
        _model(ascii_path, unit="unspecified", semantics=_semantics())


def test_explicit_stl_units_produce_the_same_si_geometry_and_scale(
    tmp_path: Path,
) -> None:
    millimeter_path = tmp_path / "fixture_mm.stl"
    meter_path = tmp_path / "fixture_m.stl"
    _write_ascii_stl(millimeter_path, _box_triangles(1000.0))
    _write_ascii_stl(meter_path, _box_triangles(1.0))

    millimeter_model = _model(millimeter_path, unit="mm", semantics=_semantics())
    meter_model = _model(meter_path, unit="m", semantics=_semantics())

    assert np.array_equal(millimeter_model.mesh.vertices_m, meter_model.mesh.vertices_m)
    assert np.array_equal(millimeter_model.mesh.faces, meter_model.mesh.faces)
    assert np.allclose(
        millimeter_model.mesh.face_areas_m2,
        meter_model.mesh.face_areas_m2,
        rtol=0.0,
        atol=1.0e-15,
    )
    lower, upper = meter_model.mesh.bounds_m
    assert np.array_equal(lower, (-1.0, -1.0, -1.0))
    assert np.array_equal(upper, (+1.0, +1.0, +1.0))


def test_semantics_and_hand_normal_cone_select_then_sample_lateral_surface(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.stl"
    _write_ascii_stl(source, _box_triangles(1.0))
    model = _model(source, unit="m", semantics=_semantics())
    cone = _fixture_pad_normal_cone()

    mask = eligible_lateral_face_mask(model, cone)
    assert np.array_equal(np.flatnonzero(mask), np.arange(4, 12))
    patches = extract_lateral_contact_patches(model, cone)
    assert len(patches) == 1
    assert patches[0].semantic == "grip_surface"
    assert np.array_equal(patches[0].face_indices, np.arange(4, 12))
    assert patches[0].area_m2 == pytest.approx(16.0)

    first = sample_contact_surfaces(
        model,
        cone,
        task_frame=_registered_task_frame(),
        sample_count=128,
        seed=7411,
    )
    second = sample_contact_surfaces(
        model,
        cone,
        task_frame=_registered_task_frame(),
        sample_count=128,
        seed=7411,
    )
    assert first.method == AREA_STRATIFIED_SOBOL
    assert np.array_equal(first.face_indices, second.face_indices)
    assert np.array_equal(first.barycentric_coordinates, second.barycentric_coordinates)
    assert np.array_equal(first.positions_m, second.positions_m)
    assert set(first.semantics) == {"grip_surface"}
    assert set(first.patch_ids) == {patches[0].patch_id}
    assert np.sum(first.integration_weights_m2) == pytest.approx(16.0)
    assert np.all(np.abs(first.normals[:, 2]) == 0.0)


def test_contract_mask_and_visibility_predicate_are_explicitly_composed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.stl"
    _write_ascii_stl(source, _box_triangles(1.0))
    model = _model(source, unit="m", semantics=_semantics())
    cone = _fixture_pad_normal_cone()
    contract_mask = np.zeros(len(model.mesh.faces), dtype=bool)
    contract_mask[4:8] = True

    def visible_first_intersections(_model: ObjectGraspModel) -> np.ndarray:
        visible = np.zeros(len(_model.mesh.faces), dtype=bool)
        visible[6:12] = True
        return visible

    selected = eligible_lateral_face_mask(
        model,
        cone,
        allowed_face_mask=contract_mask,
        visibility_predicate=visible_first_intersections,
    )
    assert np.array_equal(np.flatnonzero(selected), (6, 7))
    samples = sample_contact_surfaces(
        model,
        cone,
        task_frame=_registered_task_frame(),
        sample_count=32,
        seed=9187,
        allowed_face_mask=contract_mask,
        visibility_predicate=visible_first_intersections,
    )
    assert set(samples.face_indices) <= {6, 7}
    assert np.sum(samples.integration_weights_m2) == pytest.approx(4.0)


def test_object_patches_and_samples_are_rigid_transform_equivariant(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.stl"
    _write_binary_stl(source, _box_triangles(1.0))
    model = _model(source, unit="m", semantics=_semantics())
    cone = _fixture_pad_normal_cone()
    task_frame = _registered_task_frame()
    samples = sample_contact_surfaces(
        model,
        cone,
        task_frame=task_frame,
        sample_count=64,
        seed=3191,
    )
    patches = extract_lateral_contact_patches(model, cone)

    yaw, pitch = math.radians(31.0), math.radians(-23.0)
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
    translation = np.asarray((0.37, -0.19, 0.42))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    transformed_model = model.transformed(transform)
    transformed_samples = sample_contact_surfaces(
        transformed_model,
        cone,
        task_frame=task_frame.transformed(transform),
        sample_count=64,
        seed=3191,
    )
    transformed_patches = extract_lateral_contact_patches(transformed_model, cone)

    assert len(transformed_patches) == len(patches)
    assert transformed_patches[0].patch_id == patches[0].patch_id
    assert np.array_equal(transformed_patches[0].face_indices, patches[0].face_indices)
    assert transformed_patches[0].area_m2 == pytest.approx(patches[0].area_m2)
    assert np.array_equal(transformed_samples.face_indices, samples.face_indices)
    assert np.array_equal(
        transformed_samples.barycentric_coordinates, samples.barycentric_coordinates
    )
    expected_positions = samples.positions_m @ rotation.T + translation
    position_scale = max(
        1.0,
        float(np.max(np.abs(samples.positions_m))),
        float(np.max(np.abs(expected_positions))),
        float(np.max(np.abs(transformed_samples.positions_m))),
    )
    assert float(
        np.max(np.abs(transformed_samples.positions_m - expected_positions))
    ) <= _gamma(24) * position_scale
    expected_normals = samples.normals @ rotation.T
    assert float(
        np.max(np.abs(transformed_samples.normals - expected_normals))
    ) <= _gamma(128) * max(
        1.0,
        float(np.linalg.norm(rotation, ord=np.inf)),
    )
    assert np.allclose(transformed_model.assembly_axis, rotation @ model.assembly_axis)
    assert np.allclose(transformed_model.center_of_mass_m, translation)
    assert np.allclose(
        transformed_model.inertia_kg_m2,
        rotation @ model.inertia_kg_m2 @ rotation.T,
    )


def test_surface_layer_contains_no_connector_specific_candidate_coordinates() -> None:
    implementation_paths = (
        Path(__file__).resolve().parents[2]
        / "kcg_connector/grasp/robust/object_model.py",
        Path(__file__).resolve().parents[2]
        / "kcg_connector/grasp/robust/surface_sampling.py",
    )
    forbidden_tokens = ("D38999", "DEUTSCH", "J35", "CAD_P", "candidate_id")
    for implementation_path in implementation_paths:
        source = implementation_path.read_text(encoding="utf-8")
        assert all(token not in source for token in forbidden_tokens)
