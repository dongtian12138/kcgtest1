"""Regressions for full PAD mesh contact and fail-closed motion compatibility."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.candidate_generator import generate_raw_candidates
from kcg_connector.grasp.carts_v2.closure_predictor import (
    SequentialClosurePredictor,
)
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.surface_contact import (
    ExactPadSurfaceQuery,
    NearestSurface,
    nearest_motion_compatible_index,
)
from kcg_connector.grasp.carts_v2.task_grip_surface import (
    motion_compatible_with_object_witness,
)
from kcg_connector.grasp.robust.object_model import TriangleMesh


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/kcg_connector/config/carts_grasp_v2.yaml"
OBJECT_A = "current_d38999_26kj61sn_public_spec"
REGION_CONFIG = SimpleNamespace(section=lambda name: {
    "minimum_three_contact_triangle_area_m2": 1.0e-8})


def test_exact_pad_mesh_query_returns_full_triangle_witness_and_normal() -> None:
    pytest.importorskip("fcl")
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    mesh = TriangleMesh(vertices, faces, ("allowed",))
    orientation_sha = "a" * 64
    inputs = SimpleNamespace(
        object_contract=SimpleNamespace(
            model=SimpleNamespace(mesh=mesh),
            orientation_certificate=SimpleNamespace(
                positive_volume_winding_sign_by_source_face=(1,),
                canonical_sha256=orientation_sha,
            ),
            material_boundary_evidence=SimpleNamespace(
                formal_material_boundary_eligible=True,
                certificate=SimpleNamespace(
                    orientation_certificate_sha256=orientation_sha
                ),
            ),
        ),
        hand_contract=SimpleNamespace(
            pads=(SimpleNamespace(name="pad", points_local_m=vertices, faces=faces),)
        ),
    )
    index = ExactPadSurfaceQuery(inputs)
    transform = np.eye(4)
    transform[2, 3] = 0.1
    nearest, pad_point, normal = index.query_pad("pad", transform)
    assert nearest.face_index.tolist() == [0]
    assert nearest.distance_m[0] == pytest.approx(0.1)
    assert pad_point[2] == pytest.approx(0.1)
    assert np.allclose(normal, (0.0, 0.0, 1.0))
    touching, _pad_point, _normal = index.query_pad("pad", np.eye(4))
    assert touching.intersecting is True


def test_outward_normal_sign_is_bound_to_material_certificate() -> None:
    pytest.importorskip("fcl")
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    faces = np.asarray(((0, 2, 1),), dtype=np.int64)
    mesh = TriangleMesh(vertices, faces, ("allowed",))
    orientation_sha = "b" * 64
    inputs = SimpleNamespace(
        object_contract=SimpleNamespace(
            model=SimpleNamespace(mesh=mesh),
            orientation_certificate=SimpleNamespace(
                positive_volume_winding_sign_by_source_face=(-1,),
                canonical_sha256=orientation_sha,
            ),
            material_boundary_evidence=SimpleNamespace(
                formal_material_boundary_eligible=True,
                certificate=SimpleNamespace(
                    orientation_certificate_sha256=orientation_sha
                ),
            ),
        ),
        hand_contract=SimpleNamespace(
            pads=(SimpleNamespace(name="pad", points_local_m=vertices, faces=faces),)
        ),
    )
    assert np.allclose(ExactPadSurfaceQuery(inputs).normal(0), (0.0, 0.0, 1.0))


def test_task_grip_query_returns_hand_face_identity_and_real_normal() -> None:
    pytest.importorskip("fcl")
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    mesh = TriangleMesh(vertices, faces, ("allowed",))
    orientation_sha = "c" * 64
    surface = SimpleNamespace(
        points_local_m=vertices,
        faces=faces,
        face_normals_local=np.asarray(((0.0, 0.0, 1.0),)),
        source_face_indices=np.asarray((7,), dtype=np.int64),
        legacy_blue_pad_face_mask=np.asarray((False,), dtype=np.bool_),
        patch_indices=np.asarray((0,), dtype=np.int64),
    )
    inputs = SimpleNamespace(
        object_contract=SimpleNamespace(
            model=SimpleNamespace(mesh=mesh),
            orientation_certificate=SimpleNamespace(
                positive_volume_winding_sign_by_source_face=(1,),
                canonical_sha256=orientation_sha,
            ),
            material_boundary_evidence=SimpleNamespace(
                formal_material_boundary_eligible=True,
                certificate=SimpleNamespace(
                    orientation_certificate_sha256=orientation_sha
                ),
            ),
        ),
        hand_contract=SimpleNamespace(pads=()),
        task_grip_surfaces={"finger_1_pad": surface},
        config=REGION_CONFIG,
    )
    transform = np.eye(4)
    transform[2, 3] = 0.1
    nearest, _point, _normal = ExactPadSurfaceQuery(inputs).query_pad(
        "finger_1_pad", transform
    )
    assert nearest.surface_face_index.tolist() == [7]
    assert np.allclose(nearest.surface_normal_m, ((0.0, 0.0, 1.0),))
    assert nearest.surface_legacy_blue_pad.tolist() == [False]


def test_second_nearest_motion_compatible_witness_is_not_lost() -> None:
    selected = nearest_motion_compatible_index(
        np.asarray((1.0e-4, 2.0e-4)),
        np.asarray((-1.0e-3, 1.0e-3)),
        1.0e-5,
    )
    assert selected == 1
    assert nearest_motion_compatible_index(
        np.asarray((1.0e-4, 2.0e-4)),
        np.asarray((-1.0e-3, 0.0)),
        1.0e-5,
    ) == -1


def test_task_surface_second_patch_witness_reaches_production_selector() -> None:
    hand_points = np.asarray(((0.0, 0.0, 0.1), (0.1, 0.0, 0.1)))
    object_points = hand_points - np.asarray((0.0, 0.0, 0.1))
    hand_normals = np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)))
    object_normals = np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)))
    motion = np.asarray(((0.0, 0.0, -0.01), (0.0, 0.0, -0.01)))
    compatible = motion_compatible_with_object_witness(
        hand_points, object_points, hand_normals, object_normals, motion,
        np.asarray((0.05, 0.0, 0.0)), 1.0e-5,
    )
    selected = nearest_motion_compatible_index(
        np.asarray((1.0e-4, 2.0e-4)),
        np.where(compatible, 0.01, -np.inf), 1.0e-5,
    )
    assert compatible.tolist() == [False, True]
    assert selected == 1


def test_task_surface_query_returns_one_witness_per_registered_patch() -> None:
    pytest.importorskip("fcl")
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0),
                           (2, 0, 0), (3, 0, 0), (2, 1, 0)), dtype=float)
    faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    mesh = TriangleMesh(vertices, faces, ("allowed", "allowed"))
    orientation_sha = "d" * 64
    surface = SimpleNamespace(
        points_local_m=vertices, faces=faces,
        face_normals_local=np.asarray(((0.0, 0.0, -1.0),) * 2),
        source_face_indices=np.asarray((7, 8), dtype=np.int64),
        legacy_blue_pad_face_mask=np.asarray((False, False), dtype=np.bool_),
        patch_indices=np.asarray((0, 1), dtype=np.int64),
    )
    inputs = SimpleNamespace(
        object_contract=SimpleNamespace(
            model=SimpleNamespace(mesh=mesh),
            orientation_certificate=SimpleNamespace(
                positive_volume_winding_sign_by_source_face=(1, 1),
                canonical_sha256=orientation_sha),
            material_boundary_evidence=SimpleNamespace(
                formal_material_boundary_eligible=True,
                certificate=SimpleNamespace(
                    orientation_certificate_sha256=orientation_sha))),
        face_roles=SimpleNamespace(face_is_allowed=np.asarray((True, True))),
        hand_contract=SimpleNamespace(pads=()),
        task_grip_surfaces={"finger_1_pad": surface},
        config=REGION_CONFIG,
    )
    transform = np.eye(4)
    transform[2, 3] = 0.1
    nearest, hand_points, normals = ExactPadSurfaceQuery(
        inputs).query_task_surface_witnesses("finger_1_pad", transform, 2)
    assert len(nearest.distance_m) == len(hand_points) == len(normals) == 2
    assert set(nearest.surface_face_index.tolist()) == {7, 8}
    assert np.allclose(nearest.distance_m, (0.1, 0.1))
    assert nearest.registered_patch_count == nearest.finite_patch_witness_count == 2


def test_full_surface_far_gate_skips_patch_queries_and_preserves_distance(
    monkeypatch,
) -> None:
    query = object.__new__(ExactPadSurfaceQuery)
    full = NearestSurface(
        point_m=np.asarray(((0.0, 0.0, 0.0),)),
        distance_m=np.asarray((1.0e-2,)),
        face_index=np.asarray((3,), dtype=np.int64),
        surface_face_index=np.asarray((7,), dtype=np.int64),
        surface_normal_m=np.asarray(((0.0, 0.0, -1.0),)),
    )
    responses = [full]
    calls = {"full": 0, "patch": 0}

    def query_pad(_pad_name, _transform):
        calls["full"] += 1
        return responses[0], np.zeros(3), np.asarray((0.0, 0.0, 1.0))

    def query_patches(*_args, **_kwargs):
        calls["patch"] += 1
        pytest.fail("far or intersecting full-surface states must skip patch queries")

    monkeypatch.setattr(query, "query_pad", query_pad)
    monkeypatch.setattr(query, "query_task_surface_witnesses", query_patches)
    selected, nearest, _normals, _inward = query.select_task_surface_contact(
        "finger_1_pad", np.eye(4), np.eye(4), np.zeros(3),
        1.0e-5, 0.1, 7.5e-4,
    )
    assert selected == -1
    assert nearest is full
    assert nearest.distance_m[0] == pytest.approx(1.0e-2)
    assert calls == {"full": 1, "patch": 0}

    responses[0] = NearestSurface(
        point_m=full.point_m,
        distance_m=np.asarray((0.0,)),
        face_index=full.face_index,
        intersecting=True,
    )
    selected, nearest, _normals, _inward = query.select_task_surface_contact(
        "finger_1_pad", np.eye(4), np.eye(4), np.zeros(3),
        1.0e-5, 0.1, 7.5e-4,
    )
    assert selected == -1
    assert nearest.intersecting is True
    assert calls == {"full": 2, "patch": 0}


def test_full_surface_boundary_still_runs_multi_witness_query(monkeypatch) -> None:
    query = object.__new__(ExactPadSurfaceQuery)
    contact_distance = 7.5e-4
    tolerance = 64.0 * np.finfo(np.float64).eps
    full = NearestSurface(
        point_m=np.asarray(((0.0, 0.0, 0.0),)),
        distance_m=np.asarray((contact_distance + tolerance,)),
        face_index=np.asarray((0,), dtype=np.int64),
    )
    witness = NearestSurface(
        point_m=np.asarray(((0.0, 0.0, -5.0e-4),
                            (1.0e-3, 0.0, -5.0e-4),
                            (0.0, 1.0e-3, -5.0e-4))),
        distance_m=np.asarray((5.0e-4,) * 3),
        face_index=np.asarray((0, 0, 0), dtype=np.int64),
        surface_normal_m=np.asarray(((0.0, 0.0, -1.0),) * 3),
        surface_patch_index=np.asarray((0, 1, 2), dtype=np.int64),
        surface_patch_area_m2=np.asarray((1.0, 1.0, 1.0)),
        object_role_code=np.asarray((0, 1, 1), dtype=np.uint8),
        forbidden_distance_m=6.0e-4,
    )
    query._minimum_region_area_m2 = 1.0e-8
    calls = {"patch": 0}

    monkeypatch.setattr(
        query, "query_pad",
        lambda *_args: (full, np.zeros(3), np.asarray((0.0, 0.0, 1.0))),
    )

    object_normals = np.asarray(((0.0, 0.0, 1.0),
                                 (0.1, 0.0, np.sqrt(0.99)),
                                 (0.0, 0.1, np.sqrt(0.99))))

    def query_patches(*_args, **_kwargs):
        calls["patch"] += 1
        return witness, np.zeros((3, 3)), object_normals

    monkeypatch.setattr(query, "query_task_surface_witnesses", query_patches)
    monkeypatch.setattr(
        "kcg_connector.grasp.carts_v2.task_grip_surface."
        "motion_compatible_with_object_witness",
        lambda *_args: np.asarray((True, True, True)),
    )
    moved = np.eye(4)
    moved[2, 3] = -1.0e-3
    selected, nearest, returned_normals, _inward = query.select_task_surface_contact(
        "finger_1_pad", np.eye(4), moved, np.zeros(3),
        1.0e-5, 0.1, contact_distance,
    )
    assert selected == 0
    assert nearest.distance_m[0] == pytest.approx(5.0e-4)
    assert nearest.forbidden_first_contact is False
    assert nearest.contact_region_pass is True
    assert nearest.region_primary_sampled_hand_patch_area_fraction == pytest.approx(1 / 3)
    assert nearest.region_secondary_sampled_hand_patch_area_fraction == pytest.approx(2 / 3)
    assert returned_normals[selected] == pytest.approx(object_normals[selected])
    assert calls["patch"] == 1

    query._minimum_region_area_m2 = 1.0
    selected, nearest, _normals, _inward = query.select_task_surface_contact(
        "finger_1_pad", np.eye(4), moved, np.zeros(3),
        1.0e-5, 0.1, contact_distance,
    )
    assert selected == -1
    assert nearest.contact_region_pass is False
    assert nearest.forbidden_first_contact is False


def test_forbidden_object_surface_blocks_farther_allowed_witness(monkeypatch) -> None:
    pytest.importorskip("fcl")
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0),
                           (0, 0, 5e-4), (1, 0, 5e-4), (0, 1, 5e-4)), dtype=float)
    faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    mesh = TriangleMesh(vertices, faces, ("allowed", "forbidden"))
    orientation_sha = "e" * 64
    surface = SimpleNamespace(
        points_local_m=vertices[:3], faces=np.asarray(((0, 1, 2),)),
        face_normals_local=np.asarray(((0.0, 0.0, -1.0),)),
        source_face_indices=np.asarray((7,), dtype=np.int64),
        legacy_blue_pad_face_mask=np.asarray((False,), dtype=np.bool_),
        patch_indices=np.asarray((0,), dtype=np.int64),
    )
    inputs = SimpleNamespace(
        object_contract=SimpleNamespace(
            model=SimpleNamespace(mesh=mesh),
            orientation_certificate=SimpleNamespace(
                positive_volume_winding_sign_by_source_face=(1, 1),
                canonical_sha256=orientation_sha),
            material_boundary_evidence=SimpleNamespace(
                formal_material_boundary_eligible=True,
                certificate=SimpleNamespace(
                    orientation_certificate_sha256=orientation_sha))),
        face_roles=SimpleNamespace(face_is_allowed=np.asarray((True, False))),
        hand_contract=SimpleNamespace(pads=()),
        task_grip_surfaces={"finger_1_pad": surface},
        config=REGION_CONFIG,
    )
    current, moved = np.eye(4), np.eye(4)
    current[2, 3], moved[2, 3] = 1e-3, 9e-4
    query = ExactPadSurfaceQuery(inputs)
    forbidden_calls = 0
    original_forbidden_distance = query._forbidden_surface_distance

    def counted_forbidden_distance(*args):
        nonlocal forbidden_calls
        forbidden_calls += 1
        return original_forbidden_distance(*args)

    monkeypatch.setattr(
        query, "_forbidden_surface_distance", counted_forbidden_distance)
    selected, nearest, _normals, _inward = query.select_task_surface_contact(
            "finger_1_pad", current, moved, np.asarray((0.2, 0.2, 0.0)),
            1e-5, 0.1, 7.5e-4)
    assert selected == -1
    assert nearest.forbidden_first_contact is True
    assert nearest.forbidden_face_index == 1
    assert nearest.forbidden_distance_m == pytest.approx(5e-4)
    assert forbidden_calls == 1


def test_empty_motion_compatible_contact_fails_closed(monkeypatch) -> None:
    inputs = load_v2_inputs(ROOT, config_path=CONFIG, object_id=OBJECT_A)
    seed = generate_raw_candidates(inputs)[0]
    predictor = SequentialClosurePredictor(inputs)
    nearest = NearestSurface(
        point_m=np.asarray(((0.01, 0.0, 0.0),)),
        distance_m=np.asarray((1.0e-4,)),
        face_index=np.asarray((int(inputs.face_roles.allowed_face_indices[0]),)),
    )
    monkeypatch.setattr(predictor, "_initial_clearance", lambda *_args: 0.002)
    monkeypatch.setattr(
        predictor,
        "_contact_at_phase",
        lambda *_args: (-1, nearest, np.asarray(((1.0, 0.0, 0.0),)), np.asarray((-1.0,))),
    )
    prediction = predictor.predict(seed)
    assert prediction.status == "CLOSURE_REJECT"
    assert prediction.reason == "NO_MOTION_COMPATIBLE_PAD_POINT_finger_1_pad"
