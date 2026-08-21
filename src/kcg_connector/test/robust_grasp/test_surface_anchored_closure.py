from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

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
    RayClosureError,
    RayClosureSurfaceModel,
)
from kcg_connector.grasp.robust.surface_anchored_closure import (
    ANCHOR_ROLE,
    FIXED_ANCHOR_METHOD_ID,
    FIXED_ANCHOR_PARAMETER_DOMAIN_ID,
    METHOD_ID,
    PARAMETER_DOMAIN_ID,
    SurfaceAnchoredRayClosureModel,
)


def _reference(name: str, points: np.ndarray, faces: np.ndarray) -> VerifiedFileReference:
    digest = hashlib.sha256()
    digest.update(np.asarray(points, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(faces, dtype="<i8").tobytes(order="C"))
    return VerifiedFileReference(
        repository_relative_path=f"synthetic/{name}.npz",
        absolute_path=Path(f"/synthetic/{name}.npz"),
        sha256=digest.hexdigest(),
        byte_count=int(points.nbytes + faces.nbytes),
    )


def _pad(name: str, finger: str, link: str) -> VerifiedPad:
    points = np.asarray(
        ((-0.04, -0.04, 0.0), (0.04, -0.04, 0.0),
         (0.04, 0.04, 0.0), (-0.04, 0.04, 0.0)),
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
        ("finger_a", "joint_a", "link_a", "pad_a", (1.5, 0.0, 0.0),
         (0.0, math.pi / 2.0, 0.0)),
        ("finger_b", "joint_b", "link_b", "pad_b", (0.0, 1.5, 0.0),
         (-math.pi / 2.0, 0.0, 0.0)),
        ("finger_c", "joint_c", "link_c", "pad_c", (-1.5, 0.0, 0.0),
         (0.0, -math.pi / 2.0, 0.0)),
    )
    joints = {}
    pad_geometry = {}
    finger_joints = {}
    pads = []
    for finger, joint, link, pad_name, origin, rpy in definitions:
        joints[joint] = JointSpec(
            name=joint,
            joint_type="prismatic",
            parent_link="hand_base",
            child_link=link,
            origin_xyz_m=origin,
            origin_rpy_rad=rpy,
            axis=(0.0, 0.0, -1.0),
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
        pads.append(_pad(pad_name, finger, link))
    hand = ThreeFingerHandModel(
        base_link="hand_base",
        joints=joints,
        joint_order=tuple(joints),
        finger_joint_names=finger_joints,
        pads=pad_geometry,
    )
    return hand, tuple(pads)


def _box(
    *,
    face_permutation: tuple[int, int, int] = (0, 1, 2),
    selected_face_indices: tuple[int, ...] | None = None,
) -> ObjectGraspModel:
    vertices = np.asarray(
        ((-1.0, -0.75, -0.5), (1.0, -0.75, -0.5),
         (1.0, 0.75, -0.5), (-1.0, 0.75, -0.5),
         (-1.0, -0.75, 0.5), (1.0, -0.75, 0.5),
         (1.0, 0.75, 0.5), (-1.0, 0.75, 0.5)),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
         (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
         (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)),
        dtype=np.int64,
    )
    selected = (
        np.arange(len(faces), dtype=np.int64)
        if selected_face_indices is None
        else np.asarray(selected_face_indices, dtype=np.int64)
    )
    faces = np.array(faces, copy=True)
    faces[selected] = faces[selected][:, face_permutation]
    semantics = tuple("external_surface" for _ in faces)
    mesh = TriangleMesh(vertices, faces, semantics)
    digest = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()
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
    object_model: ObjectGraspModel | None = None,
    *,
    common_transform: np.ndarray | None = None,
) -> tuple[SurfaceAnchoredRayClosureModel, ThreeFingerHandModel]:
    hand, pads = _hand_and_pads()
    selected_object = _box() if object_model is None else object_model
    transverse_axis = np.asarray((1.0, 0.0, 0.0))
    if common_transform is not None:
        transform = np.asarray(common_transform, dtype=np.float64)
        selected_object = selected_object.transformed(transform)
        transverse_axis = transform[:3, :3] @ transverse_axis
    closure = RayClosureSurfaceModel(
        object_model=selected_object,
        hand_model=hand,
        verified_pads=pads,
        task_frame=PreRegisteredTaskFrame(
            transverse_axis_object=tuple(float(row) for row in transverse_axis),
            source="SYNTHETIC_TASK_FRAME",
        ),
        closing_actuation_directions_unit=np.eye(3),
        object_contact_normal_policy=OBJECT_CONTACT_NORMAL_POLICY,
        pad_surface_normal_policy=PAD_SURFACE_NORMAL_POLICY,
        maximum_subdivision_intervals=4096,
        interval_decimal_precision=80,
        maximum_root_bisection_iterations=256,
    )
    return SurfaceAnchoredRayClosureModel(closure), hand


def _surface_coordinate_for_face(
    model: SurfaceAnchoredRayClosureModel,
    face_index: int,
    *,
    residual: float,
) -> float:
    row = int(np.flatnonzero(model.allowed_face_indices == face_index)[0])
    lower = 0.0 if row == 0 else model.cumulative_allowed_area_m2[row - 1]
    upper = model.cumulative_allowed_area_m2[row]
    return float(
        (lower + residual * (upper - lower))
        / model.allowed_surface_area_m2
    )


def _parameters(
    model: SurfaceAnchoredRayClosureModel,
    *,
    pad_selector: float,
    face_index: int,
) -> np.ndarray:
    return np.asarray(
        (
            0.0,
            pad_selector,
            _surface_coordinate_for_face(
                model, face_index, residual=0.25
            ),
            0.5,
            0.25,
            0.0,
        ),
        dtype=np.float64,
    )


def _fixed_parameters(
    model: SurfaceAnchoredRayClosureModel,
    *,
    face_index: int,
) -> np.ndarray:
    selector_bearing = _parameters(
        model,
        pad_selector=0.0,
        face_index=face_index,
    )
    return np.concatenate((selector_bearing[:1], selector_bearing[2:]))


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


def test_surface_anchor_is_deterministic_and_delegates_final_contacts() -> None:
    model, hand = _model()
    parameters = _parameters(model, pad_selector=0.1, face_index=10)

    first = model.evaluate_unit_parameters(parameters, hand)
    second = model.evaluate_unit_parameters(parameters.copy(), hand)
    audit_before_runtime_provenance_access = first.audit.as_dict()

    assert first == second
    assert SurfaceAnchoredRayClosureModel.fixed_anchor_method_id == (
        FIXED_ANCHOR_METHOD_ID
    )
    assert model.fixed_anchor_method_id == FIXED_ANCHOR_METHOD_ID
    assert (
        SurfaceAnchoredRayClosureModel.fixed_anchor_parameter_domain_id
        == FIXED_ANCHOR_PARAMETER_DOMAIN_ID
    )
    assert model.fixed_anchor_parameter_domain_id == (
        FIXED_ANCHOR_PARAMETER_DOMAIN_ID
    )
    assert "fixed_anchor_method_id" not in vars(model)
    assert "fixed_anchor_parameter_domain_id" not in vars(model)
    assert first.audit.as_dict() == audit_before_runtime_provenance_access
    representative = _representative_candidate(first)
    assert first.audit.method_id == METHOD_ID
    assert first.audit.parameter_domain_id == PARAMETER_DOMAIN_ID
    assert first.audit.anchor_role == ANCHOR_ROLE
    assert first.audit.anchor_pad_name == "pad_a"
    assert first.audit.anchor_object_face_index == 10
    assert (
        first.audit.selected_normalised_pad_source_winding_approach_margin
        is not None
    )
    assert (
        first.audit.selected_normalised_pad_source_winding_approach_margin
        > 0.0
    )
    assert first.audit.compatible_witness_count is not None
    assert first.audit.compatible_witness_count > 1
    assert first.audit.selected_witness_branch_index is not None
    assert len(representative.planned_pad_contacts) == 3
    assert model.candidate_from_unit_parameters(parameters, hand) is None
    assert first.display_only_proposal is not None
    with pytest.raises(
        RayClosureError,
        match=REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
    ):
        model.trajectory_clearance_m(representative, hand)


def test_anchor_pad_selector_covers_all_three_fingers_symmetrically() -> None:
    model, hand = _model()
    rows = ((0.1, 10, "pad_a"), (0.5, 6, "pad_b"), (0.9, 8, "pad_c"))

    for selector, face, expected_pad in rows:
        evaluation = model.evaluate_unit_parameters(
            _parameters(model, pad_selector=selector, face_index=face), hand
        )
        assert evaluation.audit.anchor_pad_name == expected_pad
        assert (
            evaluation.audit.
            selected_normalised_pad_source_winding_approach_margin
            is not None
        )
        assert (
            evaluation.audit.
            selected_normalised_pad_source_winding_approach_margin
            > 0.0
        )


def test_object_source_normal_is_not_a_proposal_gate_and_chart_seams_fail_closed() -> None:
    model, hand = _model()
    top = model.evaluate_unit_parameters(
        _parameters(model, pad_selector=0.1, face_index=2), hand
    )
    assert top.audit.compatible_witness_count is not None
    assert top.audit.compatible_witness_count > 0
    assert top.audit.failure_reason != "NO_KINEMATICALLY_COMPATIBLE_PAD_WITNESS"

    seam = _parameters(model, pad_selector=0.1, face_index=10)
    seam[0] = 1.0
    rejected = model.evaluate_unit_parameters(seam, hand)
    assert rejected.candidate is None
    assert rejected.audit.failure_reason is not None
    assert "half-open" in rejected.audit.failure_reason

    degenerate = _parameters(model, pad_selector=0.1, face_index=10)
    degenerate[2] = 0.0
    degenerate[3] = 0.5
    rejected_degenerate = model.evaluate_unit_parameters(degenerate, hand)
    assert rejected_degenerate.candidate is None
    assert rejected_degenerate.audit.failure_reason is not None
    assert "canonical split" in rejected_degenerate.audit.failure_reason


def test_contract_contains_no_object_specific_cutoff_or_legacy_candidate() -> None:
    model, _hand = _model()
    text = repr(dict(model.contract))
    for forbidden in (
        "CAD_",
        "PAD_Z",
        "candidate-",
        "D38999",
        "J35",
        "24_mm",
        "alignment_0p90",
    ):
        assert forbidden not in text


def test_fixed_anchor_mapper_is_selector_free_v9_free_and_immutable_for_all_pads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, hand = _model()
    delegated_calls: list[tuple[object, ...]] = []

    def forbidden_v9(*args: object, **kwargs: object) -> object:
        delegated_calls.append(args + tuple(kwargs.items()))
        raise AssertionError("fixed-anchor mapper must not call delegated V9")

    monkeypatch.setattr(
        model.closure_model,
        "evaluate_unit_parameters",
        forbidden_v9,
    )
    rows = (("pad_a", 10), ("pad_b", 6), ("pad_c", 8))
    proposals = []
    for pad_name, face_index in rows:
        parameters6 = _fixed_parameters(model, face_index=face_index)
        proposal = model.propose_fixed_anchor(parameters6, pad_name, hand)
        proposals.append(proposal)

        assert proposal.feasible
        assert isinstance(proposal.v9_parameters_unit, tuple)
        assert len(proposal.v9_parameters_unit) == (
            model.closure_model.parameter_dimension
        )
        assert all(0.0 <= value <= 1.0 for value in proposal.v9_parameters_unit)
        assert proposal.audit.method_id == FIXED_ANCHOR_METHOD_ID
        assert (
            proposal.audit.parameter_domain_id
            == FIXED_ANCHOR_PARAMETER_DOMAIN_ID
        )
        assert proposal.audit.parameter_layout == (
            model.fixed_anchor_parameter_layout
        )
        assert len(parameters6) == model.fixed_anchor_parameter_dimension
        assert all(
            "selector" not in label
            for label in proposal.audit.parameter_layout
        )
        assert proposal.audit.anchor_pad_name == pad_name
        assert proposal.audit.delegated_closure_audit is None
        assert proposal.audit.delegated_volume_parameters_unit == (
            proposal.v9_parameters_unit
        )
    assert delegated_calls == []

    with pytest.raises(FrozenInstanceError):
        setattr(proposals[0], "v9_parameters_unit", None)
    with pytest.raises(FrozenInstanceError):
        setattr(proposals[0].audit, "failure_reason", "MUTATED")


def test_selector_adapter_calls_delegated_v9_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, hand = _model()
    delegated = model.closure_model.evaluate_unit_parameters
    delegated_calls: list[tuple[float, ...]] = []

    def counted_v9(
        parameters_unit: object,
        hand_model: ThreeFingerHandModel | None = None,
    ) -> object:
        delegated_calls.append(
            tuple(float(value) for value in np.asarray(parameters_unit))
        )
        return delegated(parameters_unit, hand_model)

    monkeypatch.setattr(
        model.closure_model,
        "evaluate_unit_parameters",
        counted_v9,
    )
    evaluation = model.evaluate_unit_parameters(
        _parameters(model, pad_selector=0.1, face_index=10),
        hand,
    )

    _representative_candidate(evaluation)
    assert len(delegated_calls) == 1
    assert delegated_calls[0] == (
        evaluation.audit.delegated_volume_parameters_unit
    )


def test_fixed_anchor_failures_are_explicit_and_do_not_call_v9(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, hand = _model()
    delegated_call_count = 0

    def forbidden_v9(*args: object, **kwargs: object) -> object:
        nonlocal delegated_call_count
        delegated_call_count += 1
        raise AssertionError("failed mapper must not call delegated V9")

    monkeypatch.setattr(
        model.closure_model,
        "evaluate_unit_parameters",
        forbidden_v9,
    )
    parameters6 = _fixed_parameters(model, face_index=10)
    unknown = model.propose_fixed_anchor(parameters6, "PAD_A", hand)
    selector_bearing = _parameters(
        model,
        pad_selector=0.1,
        face_index=10,
    )
    wrong_dimension = model.propose_fixed_anchor(
        selector_bearing,
        "pad_a",
        hand,
    )

    assert unknown.v9_parameters_unit is None
    assert unknown.audit.failure_reason is not None
    assert unknown.audit.failure_reason.startswith("ANCHOR_PAD_REJECTED:")
    assert wrong_dimension.v9_parameters_unit is None
    assert wrong_dimension.audit.failure_reason is not None
    assert "finite shape" in wrong_dimension.audit.failure_reason
    assert delegated_call_count == 0


def test_fixed_anchor_mapper_is_equivariant_under_exact_common_proper_se3() -> None:
    transform = np.asarray(
        (
            (0.0, -1.0, 0.0, 4.0),
            (1.0, 0.0, 0.0, -2.0),
            (0.0, 0.0, 1.0, 0.5),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    reference_model, reference_hand = _model()
    transformed_model, transformed_hand = _model(common_transform=transform)
    parameters6 = _fixed_parameters(reference_model, face_index=10)

    reference = reference_model.propose_fixed_anchor(
        parameters6,
        "pad_a",
        reference_hand,
    )
    transformed = transformed_model.propose_fixed_anchor(
        parameters6,
        "pad_a",
        transformed_hand,
    )

    assert reference.v9_parameters_unit is not None
    assert transformed.v9_parameters_unit is not None
    # The signed-permutation SE(3) is binary-exact; the two equivalent
    # normalised-coordinate paths can differ only in their final rounding.
    np.testing.assert_array_max_ulp(
        np.asarray(transformed.v9_parameters_unit),
        np.asarray(reference.v9_parameters_unit),
        maxulp=1,
    )
    assert transformed.audit.object_geometry_sha256 != (
        reference.audit.object_geometry_sha256
    )
    reference_position = np.asarray(reference.audit.anchor_object_position_m)
    expected_position = (
        transform[:3, :3] @ reference_position + transform[:3, 3]
    )
    assert np.array_equal(
        np.asarray(transformed.audit.anchor_object_position_m),
        expected_position,
    )
    for field in (
        "anchor_object_face_index",
        "anchor_object_barycentric",
        "anchor_closure_phase",
        "selected_pad_triangle_index",
        "selected_pad_witness_index",
        "selected_normalised_pad_source_winding_approach_margin",
        "compatible_witness_count",
        "selected_witness_branch_index",
    ):
        assert getattr(transformed.audit, field) == getattr(reference.audit, field)


def _delegated_v9_classifications(evaluation: object) -> tuple[str, ...]:
    audit = evaluation.audit.delegated_closure_audit
    assert audit is not None
    return tuple(row.first_contact_classification for row in audit.pad_audits)


def _assert_same_physical_proposal(reference: object, variant: object) -> None:
    assert variant.audit.anchor_object_face_index == (
        reference.audit.anchor_object_face_index
    )
    assert variant.audit.anchor_object_barycentric == (
        reference.audit.anchor_object_barycentric
    )
    assert variant.audit.anchor_object_position_m == (
        reference.audit.anchor_object_position_m
    )
    assert variant.audit.anchor_closure_phase == reference.audit.anchor_closure_phase
    assert variant.audit.selected_pad_triangle_index == (
        reference.audit.selected_pad_triangle_index
    )
    assert variant.audit.selected_pad_witness_index == (
        reference.audit.selected_pad_witness_index
    )
    assert (
        variant.audit.selected_normalised_pad_source_winding_approach_margin
        == reference.audit.selected_normalised_pad_source_winding_approach_margin
    )
    assert variant.audit.compatible_witness_count == (
        reference.audit.compatible_witness_count
    )
    assert variant.audit.selected_witness_branch_index == (
        reference.audit.selected_witness_branch_index
    )
    assert variant.audit.delegated_volume_parameters_unit == (
        reference.audit.delegated_volume_parameters_unit
    )
    assert variant.audit.failure_reason == reference.audit.failure_reason
    assert _delegated_v9_classifications(variant) == (
        _delegated_v9_classifications(reference)
    )
    assert _representative_candidate(variant) == (
        _representative_candidate(reference)
    )


def _assert_same_fixed_anchor_proposal(reference: object, variant: object) -> None:
    assert variant.v9_parameters_unit == reference.v9_parameters_unit
    assert variant.audit.delegated_closure_audit is None
    assert reference.audit.delegated_closure_audit is None
    for field in (
        "anchor_pad_name",
        "anchor_object_face_index",
        "anchor_object_barycentric",
        "anchor_object_position_m",
        "anchor_closure_phase",
        "selected_pad_triangle_index",
        "selected_pad_witness_index",
        "selected_normalised_pad_source_winding_approach_margin",
        "compatible_witness_count",
        "selected_witness_branch_index",
        "delegated_volume_parameters_unit",
        "failure_reason",
    ):
        assert getattr(variant.audit, field) == getattr(reference.audit, field)


@pytest.mark.parametrize(
    "selected_face_indices",
    ((10,), None),
    ids=("single_anchor_face_winding_flip", "all_face_winding_flip"),
)
def test_source_winding_flips_preserve_physical_anchor_candidate_and_v9_classification(
    selected_face_indices: tuple[int, ...] | None,
) -> None:
    reference_model, reference_hand = _model()
    parameters = _parameters(
        reference_model,
        pad_selector=0.1,
        face_index=10,
    )
    reference = reference_model.evaluate_unit_parameters(
        parameters,
        reference_hand,
    )
    reference_proposal = reference_model.propose_fixed_anchor(
        np.concatenate((parameters[:1], parameters[2:])),
        "pad_a",
        reference_hand,
    )
    _representative_candidate(reference)

    variant_model, variant_hand = _model(
        _box(
            face_permutation=(0, 2, 1),
            selected_face_indices=selected_face_indices,
        )
    )
    assert _surface_coordinate_for_face(
        variant_model,
        10,
        residual=0.25,
    ) == parameters[2]
    variant = variant_model.evaluate_unit_parameters(parameters, variant_hand)
    variant_proposal = variant_model.propose_fixed_anchor(
        np.concatenate((parameters[:1], parameters[2:])),
        "pad_a",
        variant_hand,
    )

    assert variant.audit.object_geometry_sha256 != (
        reference.audit.object_geometry_sha256
    )
    _assert_same_physical_proposal(reference, variant)
    _assert_same_fixed_anchor_proposal(reference_proposal, variant_proposal)


@pytest.mark.parametrize(
    "face_permutation",
    (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ),
)
def test_all_s3_source_face_permutations_preserve_physical_proposal(
    face_permutation: tuple[int, int, int],
) -> None:
    reference_model, reference_hand = _model()
    parameters = _parameters(
        reference_model,
        pad_selector=0.1,
        face_index=10,
    )
    reference = reference_model.evaluate_unit_parameters(
        parameters,
        reference_hand,
    )
    reference_proposal = reference_model.propose_fixed_anchor(
        np.concatenate((parameters[:1], parameters[2:])),
        "pad_a",
        reference_hand,
    )
    _representative_candidate(reference)

    variant_model, variant_hand = _model(
        _box(face_permutation=face_permutation)
    )
    variant = variant_model.evaluate_unit_parameters(parameters, variant_hand)
    variant_proposal = variant_model.propose_fixed_anchor(
        np.concatenate((parameters[:1], parameters[2:])),
        "pad_a",
        variant_hand,
    )

    if face_permutation == (0, 1, 2):
        assert variant.audit.object_geometry_sha256 == (
            reference.audit.object_geometry_sha256
        )
    else:
        assert variant.audit.object_geometry_sha256 != (
            reference.audit.object_geometry_sha256
        )
    _assert_same_physical_proposal(reference, variant)
    _assert_same_fixed_anchor_proposal(reference_proposal, variant_proposal)
