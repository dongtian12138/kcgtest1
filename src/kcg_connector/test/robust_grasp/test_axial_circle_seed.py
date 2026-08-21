from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.axial_circle_seed import (
    METHOD_ID,
    SEED_ROLE,
    AxialCircleNumericalOptions,
    AxialCircleSeedError,
    AxialConditionedCircleTriangleSeedModel,
)
from kcg_connector.grasp.robust.hand_contract import (
    OBJECT_CONTACT_NORMAL_POLICY,
    PAD_SURFACE_NORMAL_POLICY,
    VerifiedFileReference,
    VerifiedPad,
)
from kcg_connector.grasp.robust.hand_model import (
    GeometrySpec,
    JointLimit,
    JointSpec,
    PadGeometry,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.object_model import (
    AssetProvenance,
    CARTS_VISUAL_SUBTREE_NPZ,
    ObjectGraspModel,
    TriangleMesh,
)
from kcg_connector.grasp.robust.ray_closure import (
    PreRegisteredTaskFrame,
    REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
    RayClosureSurfaceModel,
)


_FLAT_AXIAL_COMPONENT_PARAMETERS = np.asarray(
    (
        0.7611236572265625,
        0.2285454124212265,
        0.8873859643936157,
        0.17561988532543182,
        0.28081122040748596,
        0.8158808350563049,
        0.17835277318954468,
        0.4516959488391876,
        0.6772855520248413,
        0.34752920269966125,
    ),
    dtype=np.float64,
)
_FEASIBLE_PARAMETERS = np.asarray(
    (
        0.05,
        0.8990384615384616,
        0.5,
        0.7110290598290597,
        0.8597972972972975,
        0.05,
        0.05,
        0.25,
        0.3,
        0.75,
    ),
    dtype=np.float64,
)


def _reference(
    name: str, points: np.ndarray, faces: np.ndarray
) -> VerifiedFileReference:
    digest = hashlib.sha256()
    digest.update(np.asarray(points, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(faces, dtype="<i8").tobytes(order="C"))
    return VerifiedFileReference(
        repository_relative_path=f"synthetic/{name}.npz",
        absolute_path=Path(f"/synthetic/{name}.npz"),
        sha256=digest.hexdigest(),
        byte_count=int(points.nbytes + faces.nbytes),
    )


def _verified_pad(name: str, finger: str, link: str) -> VerifiedPad:
    points = np.asarray(
        (
            (-0.04, -0.04, 0.0),
            (0.04, -0.04, 0.0),
            (0.04, 0.04, 0.0),
            (-0.04, 0.04, 0.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 2, 1), (0, 3, 2)), dtype=np.int64)
    return VerifiedPad(
        name=name,
        finger_name=finger,
        link_name=link,
        origin_xyz_m=(0.0, 0.0, 0.0),
        origin_rpy_rad=(0.0, 0.0, 0.0),
        mesh=_reference(name, points, faces),
        coordinate_frame=link,
        unit="m",
        normal_force_capacity_n=1.0,
        points_local_m=points,
        faces=faces,
    )


def _hand_and_pads() -> tuple[ThreeFingerHandModel, tuple[VerifiedPad, ...]]:
    definitions = (
        (
            "finger_a",
            "joint_a",
            "link_a",
            "pad_a",
            (1.5, 0.0, 0.0),
            (0.0, math.pi / 2.0, 0.0),
            (0.0, 0.0, -1.0),
        ),
        (
            "finger_b",
            "joint_b",
            "link_b",
            "pad_b",
            (0.0, 1.5, 0.0),
            (-math.pi / 2.0, 0.0, 0.0),
            (0.0, 1.0 / math.sqrt(17.0), -4.0 / math.sqrt(17.0)),
        ),
        (
            "finger_c",
            "joint_c",
            "link_c",
            "pad_c",
            (-1.5, 0.0, 0.0),
            (0.0, -math.pi / 2.0, 0.0),
            (0.0, 0.0, -1.0),
        ),
    )
    joints = {}
    pad_geometry = {}
    finger_joints = {}
    pads = []
    for finger, joint, link, pad_name, origin, rpy, axis in definitions:
        joints[joint] = JointSpec(
            name=joint,
            joint_type="prismatic",
            parent_link="hand_base",
            child_link=link,
            origin_xyz_m=origin,
            origin_rpy_rad=rpy,
            axis=axis,
            limit=JointLimit(0.0, 2.0),
        )
        finger_joints[finger] = (joint,)
        pad_geometry[pad_name] = PadGeometry(
            name=pad_name,
            finger_name=finger,
            link_name=link,
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            geometry=GeometrySpec("box", (0.08, 0.08, 0.001)),
        )
        pads.append(_verified_pad(pad_name, finger, link))
    return (
        ThreeFingerHandModel(
            base_link="hand_base",
            joints=joints,
            joint_order=tuple(joints),
            finger_joint_names=finger_joints,
            pads=pad_geometry,
        ),
        tuple(pads),
    )


def _box() -> ObjectGraspModel:
    vertices = np.asarray(
        (
            (-1.0, -0.75, -0.5),
            (1.0, -0.75, -0.5),
            (1.0, 0.75, -0.5),
            (-1.0, 0.75, -0.5),
            (-1.0, -0.75, 0.5),
            (1.0, -0.75, 0.5),
            (1.0, 0.75, 0.5),
            (-1.0, 0.75, 0.5),
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
    mesh = TriangleMesh(
        vertices,
        faces,
        tuple("external_surface" for _face in faces),
    )
    digest = hashlib.sha256(
        vertices.tobytes() + faces.tobytes()
    ).hexdigest()
    return ObjectGraspModel(
        mesh=mesh,
        provenance=AssetProvenance(
            source_path="/synthetic/box.npz",
            source_sha256=digest,
            source_class="SYNTHETIC_GEOMETRY_TEST",
            source_format=CARTS_VISUAL_SUBTREE_NPZ,
            source_unit="m",
            meters_per_source_unit=1.0,
        ),
        assembly_axis=np.asarray((0.0, 0.0, 1.0)),
        assembly_axis_origin_m=np.zeros(3),
        mass_kg=1.0,
        center_of_mass_m=np.zeros(3),
        inertia_kg_m2=np.eye(3),
        allowed_contact_semantics=frozenset(("external_surface",)),
    )


def _model(
    object_transform: np.ndarray | None = None,
) -> tuple[
    AxialConditionedCircleTriangleSeedModel,
    ThreeFingerHandModel,
]:
    hand, pads = _hand_and_pads()
    object_model = _box()
    transverse_axis = np.asarray((1.0, 0.0, 0.0))
    if object_transform is not None:
        object_model = object_model.transformed(object_transform)
        transverse_axis = object_transform[:3, :3] @ transverse_axis
    closure = RayClosureSurfaceModel(
        object_model=object_model,
        hand_model=hand,
        verified_pads=pads,
        task_frame=PreRegisteredTaskFrame(
            transverse_axis_object=tuple(transverse_axis),
            source="SYNTHETIC_TASK_FRAME",
        ),
        closing_actuation_directions_unit=np.eye(3),
        object_contact_normal_policy=OBJECT_CONTACT_NORMAL_POLICY,
        pad_surface_normal_policy=PAD_SURFACE_NORMAL_POLICY,
        maximum_subdivision_intervals=4096,
        interval_decimal_precision=80,
        maximum_root_bisection_iterations=256,
    )
    return (
        AxialConditionedCircleTriangleSeedModel(
            closure,
            numerical_options=AxialCircleNumericalOptions(
                axial_phase_cell_count=64,
                axial_bisection_iterations=64,
            ),
        ),
        hand,
    )


def _representative_candidate(evaluation: object):
    assert evaluation.candidate is None
    assert evaluation.feasible is False
    assert evaluation.representative_proposal_available
    assert evaluation.display_only_proposal is not None
    assert evaluation.static_policy_available
    assert evaluation.sequential_closure_policy is not None
    assert len(evaluation.possible_first_contact_sets) == 3
    assert evaluation.sequential_closure_policy.possible_first_contact_sets == (
        evaluation.possible_first_contact_sets
    )
    assert evaluation.audit.failure_reason == (
        "DELEGATED_CLOSURE_REJECTED:"
        f"{REPRESENTATIVE_PROPOSAL_FAILURE_REASON}"
    )
    assert evaluation.audit.delegated_closure_audit is not None
    assert evaluation.audit.delegated_closure_audit.failure_reason == (
        REPRESENTATIVE_PROPOSAL_FAILURE_REASON
    )
    assert evaluation.sequential_closure_policy.pad_order == (
        evaluation.audit.delegated_closure_audit.pad_order
    )
    return evaluation.display_only_proposal.grasp_candidate


def test_axial_circle_seed_is_deterministic_and_delegates_truth() -> None:
    model, hand = _model()

    first = model.evaluate_unit_parameters(_FEASIBLE_PARAMETERS, hand)
    second = model.evaluate_unit_parameters(
        _FEASIBLE_PARAMETERS.copy(), hand
    )

    assert first == second
    representative = _representative_candidate(first)
    assert first.audit.method_id == METHOD_ID
    assert first.audit.seed_role == SEED_ROLE
    assert len(first.audit.phase_endpoints) == 1
    assert len(first.audit.circle_triangle_endpoints) == 2
    assert first.audit.numerically_unresolved_face_count == 0
    assert first.audit.selected_phase_endpoint_index == 0
    assert first.audit.selected_circle_endpoint_index == 1
    assert len(representative.planned_pad_contacts) == 3
    endpoint = first.audit.phase_endpoints[0]
    assert endpoint.grid_cell_index == 25
    assert endpoint.construction == (
        "FORWARD_ERROR_SEPARATED_SIGN_CHANGE_BRACKET"
    )
    lower = endpoint.bracket_lower_residual_interval_m
    upper = endpoint.bracket_upper_residual_interval_m
    assert (lower[1] < 0.0 < upper[0]) or (
        upper[1] < 0.0 < lower[0]
    )
    decoded = model.candidate_from_unit_parameters(
        _FEASIBLE_PARAMETERS, hand
    )
    assert decoded is None


def test_axial_donor_is_allowed_surface_area_pushforward() -> None:
    model, hand = _model()
    result = model.evaluate_unit_parameters(_FEASIBLE_PARAMETERS, hand)

    _representative_candidate(result)
    audit = result.audit
    assert audit.sampled_axial_donor_object_face_index is not None
    donor_face = audit.sampled_axial_donor_object_face_index
    assert model.object_model.contact_face_mask[donor_face]
    barycentric = np.asarray(
        audit.sampled_axial_donor_object_barycentric
    )
    triangle = model.object_model.mesh.face_vertices_m[donor_face]
    donor_position = barycentric @ triangle
    assert tuple(donor_position) == pytest.approx(
        audit.sampled_axial_donor_object_position_m
    )
    donor_task = model.closure_model.task_basis_object.T @ (
        donor_position - model.object_model.assembly_axis_origin_m
    )
    assert donor_task[2] == pytest.approx(
        audit.sampled_axial_donor_coordinate_m
    )
    assert len(audit.allowed_face_domain_sha256) == 64
    assert all(len(value) == 64 for value in audit.pad_witness_domain_sha256)


def test_circle_endpoints_lie_on_reported_object_triangles() -> None:
    model, hand = _model()
    result = model.evaluate_unit_parameters(_FEASIBLE_PARAMETERS, hand)

    _representative_candidate(result)
    for endpoint in result.audit.circle_triangle_endpoints:
        triangle = model.object_model.mesh.face_vertices_m[
            endpoint.object_face_index
        ]
        barycentric = np.asarray(endpoint.barycentric)
        reconstructed = barycentric @ triangle
        reported = np.asarray(endpoint.position_object_m)
        scale = max(
            1.0,
            float(np.linalg.norm(triangle, ord=np.inf)),
            float(np.linalg.norm(reported, ord=np.inf)),
        )
        arithmetic_bound = (
            512.0 * np.finfo(np.float64).eps * scale
        )
        assert float(np.linalg.norm(reconstructed - reported, ord=np.inf)) < (
            arithmetic_bound
        )
        assert endpoint.proposal_classification.endswith("_SEED")


def test_phase_residual_and_sampled_normals_are_not_acceptance_gates() -> None:
    model, hand = _model()
    result = model.evaluate_unit_parameters(_FEASIBLE_PARAMETERS, hand)

    _representative_candidate(result)
    contract = dict(result.audit.numerical_contract)
    assert contract["phase_acceptance_residual_threshold"] is None
    assert contract["sampled_direction_physical_acceptance_gate"] is False
    assert any(
        row.startswith("SAMPLED_ANCHORS_MUST_NOT_FEED")
        for row in result.audit.claim_limitations
    )


def test_flat_axial_component_is_not_promoted_by_roundoff() -> None:
    model, hand = _model()
    result = model.evaluate_unit_parameters(
        _FLAT_AXIAL_COMPONENT_PARAMETERS, hand
    )

    assert result.candidate is None
    assert result.audit.failure_reason == (
        "AXIAL_PHASE_COMPONENT_OR_ARITHMETIC_UNRESOLVED"
    )
    assert result.audit.phase_endpoints == ()
    assert result.audit.unresolved_phase_interval_count == 64
    assert result.audit.axial_component_or_arithmetic_unresolved is True


def test_non_degenerate_phase_and_representative_are_se3_equivariant() -> None:
    base_model, base_hand = _model()
    cosine_x = math.cos(-0.37)
    sine_x = math.sin(-0.37)
    rotation_x = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, cosine_x, -sine_x),
            (0.0, sine_x, cosine_x),
        )
    )
    cosine_z = math.cos(0.43)
    sine_z = math.sin(0.43)
    rotation_z = np.asarray(
        (
            (cosine_z, -sine_z, 0.0),
            (sine_z, cosine_z, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation_z @ rotation_x
    transform[:3, 3] = (3.0, -2.0, 0.5)
    changed_model, changed_hand = _model(transform)

    base = base_model.evaluate_unit_parameters(
        _FEASIBLE_PARAMETERS, base_hand
    )
    changed = changed_model.evaluate_unit_parameters(
        _FEASIBLE_PARAMETERS, changed_hand
    )

    base_candidate = _representative_candidate(base)
    changed_candidate = _representative_candidate(changed)
    base_endpoint = base.audit.phase_endpoints[0]
    changed_endpoint = changed.audit.phase_endpoints[0]
    assert base_endpoint.grid_cell_index == changed_endpoint.grid_cell_index
    assert base_endpoint.bracket_lower_phase == (
        changed_endpoint.bracket_lower_phase
    )
    assert base_endpoint.bracket_upper_phase == (
        changed_endpoint.bracket_upper_phase
    )
    assert base_endpoint.selected_phase == changed_endpoint.selected_phase
    for endpoint in (base_endpoint, changed_endpoint):
        lower = endpoint.bracket_lower_residual_interval_m
        upper = endpoint.bracket_upper_residual_interval_m
        assert (lower[1] < 0.0 < upper[0]) or (
            upper[1] < 0.0 < lower[0]
        )
    base_pose = np.asarray(base_candidate.object_from_hand).reshape((4, 4))
    changed_pose = np.asarray(
        changed_candidate.object_from_hand
    ).reshape((4, 4))
    expected_pose = transform @ base_pose
    pose_scale = max(
        1.0, float(np.linalg.norm(expected_pose, ord=np.inf))
    )
    numerical_bound = (
        4096.0 * np.finfo(np.float64).eps * pose_scale
    )
    np.testing.assert_allclose(
        changed_pose,
        expected_pose,
        rtol=0.0,
        atol=numerical_bound,
    )
    np.testing.assert_allclose(
        changed_candidate.independent_joint_positions_rad,
        base_candidate.independent_joint_positions_rad,
        rtol=0.0,
        atol=numerical_bound,
    )
    for base_contact, changed_contact in zip(
        base_candidate.planned_pad_contacts,
        changed_candidate.planned_pad_contacts,
    ):
        assert base_contact.pad_name == changed_contact.pad_name
        np.testing.assert_allclose(
            changed_contact.position_object_m,
            transform[:3, :3] @ np.asarray(base_contact.position_object_m)
            + transform[:3, 3],
            rtol=0.0,
            atol=numerical_bound,
        )
        np.testing.assert_allclose(
            changed_contact.path_local_free_side_normal_object,
            transform[:3, :3]
            @ np.asarray(base_contact.path_local_free_side_normal_object),
            rtol=0.0,
            atol=numerical_bound,
        )


def test_parameter_seams_and_numerical_options_fail_closed() -> None:
    model, hand = _model()
    seam = _FEASIBLE_PARAMETERS.copy()
    seam[9] = 1.0
    rejected = model.evaluate_unit_parameters(seam, hand)

    assert rejected.candidate is None
    assert rejected.audit.failure_reason is not None
    assert "half-open" in rejected.audit.failure_reason
    with pytest.raises(TypeError):
        AxialCircleNumericalOptions()  # type: ignore[call-arg]
    with pytest.raises(AxialCircleSeedError, match="positive integer"):
        AxialCircleNumericalOptions(
            axial_phase_cell_count=0,
            axial_bisection_iterations=64,
        )


def test_contract_has_no_connector_or_legacy_candidate_tokens() -> None:
    model, _hand = _model()
    text = repr(dict(model.contract))
    for forbidden in (
        "CAD_",
        "PAD_Z",
        "candidate-",
        "D38999",
        "J35",
        "alignment_0p90",
    ):
        assert forbidden not in text
