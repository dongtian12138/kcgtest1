from __future__ import annotations

from dataclasses import replace
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
from kcg_connector.grasp.robust.pair_conditioned_seed import (
    METHOD_ID,
    PAIR_INVARIANTS,
    SEED_ROLE,
    PairConditionedInvariantSeedModel,
    PairConditionedSeedError,
    PairSeedSolverOptions,
)
from kcg_connector.grasp.robust.ray_closure import (
    PreRegisteredTaskFrame,
    REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
    RayClosureSurfaceModel,
)


_FEASIBLE_PARAMETERS = np.asarray(
    (
        0.5219444036483765,
        0.5175994038581848,
        0.2938779890537262,
        0.756528377532959,
        0.5244523286819458,
        0.08676660805940628,
        0.8521717190742493,
        0.3579062521457672,
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
        ),
        (
            "finger_b",
            "joint_b",
            "link_b",
            "pad_b",
            (0.0, 1.5, 0.0),
            (-math.pi / 2.0, 0.0, 0.0),
        ),
        (
            "finger_c",
            "joint_c",
            "link_c",
            "pad_c",
            (-1.5, 0.0, 0.0),
            (0.0, -math.pi / 2.0, 0.0),
        ),
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
        tuple("external_surface" for _ in faces),
    )
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


def _model() -> tuple[PairConditionedInvariantSeedModel, ThreeFingerHandModel]:
    hand, pads = _hand_and_pads()
    closure = RayClosureSurfaceModel(
        object_model=_box(),
        hand_model=hand,
        verified_pads=pads,
        task_frame=PreRegisteredTaskFrame(
            transverse_axis_object=(1.0, 0.0, 0.0),
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
        PairConditionedInvariantSeedModel(
            closure,
            solver_options=PairSeedSolverOptions(
                multistart_count=8,
                sobol_seed=71,
                maximum_function_evaluations=128,
                function_tolerance=1.0e-12,
                step_tolerance=1.0e-12,
                gradient_tolerance=1.0e-12,
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


def test_pair_seed_is_deterministic_and_delegates_contact_truth() -> None:
    model, hand = _model()

    first = model.evaluate_unit_parameters(_FEASIBLE_PARAMETERS, hand)
    second = model.evaluate_unit_parameters(_FEASIBLE_PARAMETERS.copy(), hand)

    assert first == second
    representative = _representative_candidate(first)
    assert first.audit.method_id == METHOD_ID
    assert first.audit.seed_role == SEED_ROLE
    assert first.audit.pair_invariants == PAIR_INVARIANTS
    assert first.audit.eligible_endpoint_count > 0
    assert first.audit.selected_endpoint_index is not None
    assert all(
        row.first_contact_classification
        == "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
        for row in first.audit.delegated_closure_audit.pad_audits
    )
    assert len(representative.planned_pad_contacts) == 3
    decoded = model.candidate_from_unit_parameters(
        _FEASIBLE_PARAMETERS, hand
    )
    assert decoded is None


def test_solver_residual_is_a_report_only_dimensionless_quantity() -> None:
    model, hand = _model()
    result = model.evaluate_unit_parameters(_FEASIBLE_PARAMETERS, hand)

    _representative_candidate(result)
    selected = result.audit.solver_endpoints[
        result.audit.selected_endpoint_index
    ]
    assert math.isfinite(selected.residual_infinity_norm)
    assert (
        dict(result.audit.solver_contract)["physical_acceptance_gate"]
        is False
    )
    assert any(
        row.startswith("ENDPOINT_RESIDUAL_IS_REPORTED_ONLY")
        for row in result.audit.claim_limitations
    )


def test_solver_diagnostics_cannot_gate_delegated_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, hand = _model()
    original = model._solver_endpoints

    def diagnostics_forced_to_fail(**kwargs):
        return tuple(
            replace(
                endpoint,
                solver_success=False,
                invariant_residual_dimensionless=(123.0, -456.0),
                residual_infinity_norm=456.0,
                sampled_anchor_directions_compatible=False,
            )
            for endpoint in original(**kwargs)
        )

    monkeypatch.setattr(
        model, "_solver_endpoints", diagnostics_forced_to_fail
    )
    result = model.evaluate_unit_parameters(_FEASIBLE_PARAMETERS, hand)

    _representative_candidate(result)
    assert result.audit.sampled_direction_compatible_endpoint_count == 0
    assert all(
        not endpoint.solver_success
        for endpoint in result.audit.solver_endpoints
    )
    assert all(
        endpoint.residual_infinity_norm == 456.0
        for endpoint in result.audit.solver_endpoints
    )
    assert result.audit.delegated_closure_audit is not None


def test_area_measure_and_parameter_domain_fail_closed() -> None:
    model, hand = _model()
    for pad_row, prepared in enumerate(model.closure_model.prepared_pads):
        triangles = prepared.verified.points_local_m[prepared.verified.faces]
        expected_area = math.fsum(
            float(value)
            for value in 0.5
            * np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                ),
                axis=1,
            )
        )
        assert model.witness_total_measures[pad_row] == pytest.approx(
            expected_area
        )

    seam = _FEASIBLE_PARAMETERS.copy()
    seam[7] = 1.0
    rejected = model.evaluate_unit_parameters(seam, hand)
    assert rejected.candidate is None
    assert rejected.audit.failure_reason is not None
    assert "half-open" in rejected.audit.failure_reason


def test_solver_numerics_have_no_silent_defaults() -> None:
    with pytest.raises(TypeError):
        PairSeedSolverOptions()  # type: ignore[call-arg]
    with pytest.raises(PairConditionedSeedError, match="binary64 epsilon"):
        PairSeedSolverOptions(
            multistart_count=8,
            sobol_seed=71,
            maximum_function_evaluations=128,
            function_tolerance=np.finfo(np.float64).eps,
            step_tolerance=1.0e-12,
            gradient_tolerance=1.0e-12,
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
