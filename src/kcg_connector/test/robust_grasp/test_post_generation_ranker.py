"""Fail-closed tests for the post-generation CARTS rank-only bridge."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib
import json
from types import MappingProxyType

import numpy as np
import pytest

from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
)
from kcg_connector.grasp.robust.object_model import (
    AssetProvenance,
    ObjectGraspModel,
    TriangleMesh,
)
from kcg_connector.grasp.robust.post_generation_ranker import (
    COMPLETE_CLEARANCE_METHOD_ID,
    COMPLETE_CLEARANCE_SCOPE,
    CandidateEvaluationState,
    CompleteTrajectoryCollisionCertificate,
    PostGenerationRankOnlyPipeline,
    PostGenerationRankingError,
    SCENARIO_COUNT,
    SCENARIO_DESIGN_SHA256,
    candidate_sha256,
    v9_evidence_sha256,
)
from kcg_connector.grasp.robust.ray_closure import (
    CANDIDATE_REPRESENTATIVE_ROLE,
    CLAIM_LIMITATIONS as RAY_CLOSURE_CLAIM_LIMITATIONS,
    CLOSURE_PARAMETER_DOMAIN_ID,
    CertifiedSequentialClosurePolicy,
    DISPLAY_APPROXIMATION_ROLE,
    METHOD_ID as RAY_CLOSURE_METHOD_ID,
    MODEL_BINDING_COMPLETE_STATUS,
    MODEL_CONTRACT_DIGEST_METHOD_ID,
    POSSIBLE_EARLIEST_ORDERING_POLICY,
    POSSIBLE_FIRST_CONTACT_SET_METHOD_ID,
    RayClosureAudit,
)
from kcg_connector.grasp.robust.robust_wrench import (
    LinearProgramSolverOptions,
)
from kcg_connector.grasp.robust.task_wrench_evaluator import (
    FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE,
    TaskWrenchEvaluator,
    TaskWrenchOnlyEvaluation,
)
from kcg_connector.grasp.robust.top_level_candidate_generator import (
    AttemptStatus,
    CandidateAttemptAudit,
    CandidateLane,
    CandidateLineage,
    StaticV9AcceptedCandidate,
    StaticV9AcceptedPolicy,
    TopLevelGenerationResult,
    UniqueV9Evaluation,
    V9InvocationAuditBinding,
    METHOD_ID as TOP_LEVEL_METHOD_ID,
)


GENERATION_SHA256 = "a" * 64
PAD_NAMES = ("pad_0", "pad_1", "pad_2")
PAD_LINKS = ("link_0", "link_1", "link_2")
V9_LAYOUT = (
    "assembly_axis_yaw_unit",
    "axial_target_unit",
    "lateral_task_x_unit",
    "lateral_task_y_unit",
    "preshape_joint_unit:q",
)


def _object_model() -> ObjectGraspModel:
    vertices = np.asarray(
        (
            (+0.20, +0.00, -0.10),
            (-0.10, +0.18, -0.10),
            (-0.10, -0.18, -0.10),
            (+0.00, +0.00, +0.20),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        dtype=np.int64,
    )
    mesh = TriangleMesh(
        vertices_m=vertices,
        faces=faces,
        face_semantics=("external",) * 4,
    )
    return ObjectGraspModel(
        mesh=mesh,
        provenance=AssetProvenance(
            source_path="rank_fixture.stl",
            source_sha256="0" * 64,
            source_class="SYNTHETIC_ANALYTIC_TEST_FIXTURE",
            source_format="ASCII_STL",
            source_unit="m",
            meters_per_source_unit=1.0,
        ),
        assembly_axis=np.asarray((0.0, 0.0, 1.0)),
        mass_kg=1.0,
        center_of_mass_m=np.zeros(3),
        inertia_kg_m2=np.diag((0.01, 0.01, 0.02)),
        allowed_contact_semantics=frozenset(("external",)),
    )


@dataclass(frozen=True)
class _Pad:
    link_name: str
    normal_force_capacity_n: float = 10.0


class _Hand:
    independent_joint_names = ("q",)

    def __init__(self) -> None:
        self.pads = MappingProxyType(
            {
                name: _Pad(link)
                for name, link in zip(PAD_NAMES, PAD_LINKS)
            }
        )


LP_OPTIONS = LinearProgramSolverOptions.from_mapping(
    {
        "solver": "SCIPY_HIGHS",
        "constraint_scaling": "ROW_AND_COLUMN_INF_NORM",
        "maximum_iterations": 10000,
        "primal_feasibility_tolerance": 1.0e-9,
        "dual_feasibility_tolerance": 1.0e-9,
        "ipm_optimality_tolerance": 1.0e-10,
        "physical_acceptance_gate": False,
    }
)


def _evaluator(model: ObjectGraspModel) -> TaskWrenchEvaluator:
    return TaskWrenchEvaluator(
        object_model=model,
        characteristic_radius_m=0.1,
        friction_coefficient_interval=(0.2, 0.8),
        uncertainty_claim_scope=(
            FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ),
        gravity_direction_object=(0.0, 0.0, -1.0),
        task_frame_rotation_object=np.eye(3),
        gravity_acceleration_m_s2=9.80665,
        lift_acceleration_m_s2=0.1,
        maximum_inner_approximation_relative_error=0.001,
        cone_edge_multiplier=2,
        solver_options=LP_OPTIONS,
    )


def _candidate(value: float) -> GraspCandidate:
    contacts = tuple(
        PlannedPadContact(
            pad_name=name,
            position_object_m=(0.0, float(index) * 0.01, 0.0),
            path_local_free_side_normal_object=(1.0, 0.0, 0.0),
            surface_coordinates=(0.25, 0.25, 0.50, value),
        )
        for index, name in enumerate(PAD_NAMES)
    )
    return GraspCandidate.from_matrix(
        object_from_hand=np.eye(4),
        independent_joint_positions_rad=(value,),
        planned_pad_contacts=contacts,
        internal_normal_forces_n=(0.0, 0.0, 0.0),
    )


def _bound_audit(
    model: ObjectGraspModel,
) -> tuple[RayClosureAudit, str]:
    pad_source = tuple(str(index + 1) * 64 for index in range(3))
    pad_runtime = tuple(str(index + 4) * 64 for index in range(3))
    directions = ((1.0,), (1.0,), (1.0,))
    interval_method = "FIXTURE_INTERVAL_METHOD"
    document = {
        "schema": MODEL_CONTRACT_DIGEST_METHOD_ID,
        "object": {"geometry_sha256": model.geometry_sha256},
        "task_frame": {"source": "FIXTURE_TASK_FRAME"},
        "hand": {"independent_joint_names": ["q"]},
        "verified_pads": [
            {
                "name": name,
                "link_name": link,
                "source_mesh_sha256": source,
                "runtime_geometry_sha256": runtime,
            }
            for name, link, source, runtime in zip(
                PAD_NAMES, PAD_LINKS, pad_source, pad_runtime
            )
        ],
        "closure": {
            "closing_directions_physical": [
                [float(value).hex() for value in row]
                for row in directions
            ]
        },
        "ray_closure": {
            "method_id": RAY_CLOSURE_METHOD_ID,
            "object_contact_normal_policy": "FIXTURE_OBJECT_NORMAL",
            "pad_surface_normal_policy": "FIXTURE_PAD_NORMAL",
            "maximum_subdivision_intervals": 4096,
            "possible_first_contact_set_method_id": (
                POSSIBLE_FIRST_CONTACT_SET_METHOD_ID
            ),
            "possible_earliest_ordering_policy": (
                POSSIBLE_EARLIEST_ORDERING_POLICY
            ),
            "representative_proposal_role": (
                CANDIDATE_REPRESENTATIVE_ROLE
            ),
            "display_approximation_role": DISPLAY_APPROXIMATION_ROLE,
        },
        "interval_backend": {
            "method_id": interval_method,
            "decimal_precision": 80,
            "maximum_root_bisection_iterations": 256,
        },
    }
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    model_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    audit = RayClosureAudit(
        method_id=RAY_CLOSURE_METHOD_ID,
        numerical_policy="fixture",
        witness_rule="fixture",
        interval_rule="fixture",
        distance_bvh_rule="fixture",
        ray_evaluation_policy="fixture",
        feature_root_policy="fixture",
        object_contact_normal_policy="FIXTURE_OBJECT_NORMAL",
        pad_surface_normal_policy="FIXTURE_PAD_NORMAL",
        parameter_layout=V9_LAYOUT,
        pad_order=PAD_NAMES,
        full_verified_pad_mesh_used=True,
        pad_face_subset_input_allowed=False,
        independent_actuation_supports=(("q",), ("q",), ("q",)),
        closure_parameter_domain_id=CLOSURE_PARAMETER_DOMAIN_ID,
        closure_suffix_dominance_argument="fixture",
        preshape_joint_names=("q",),
        closure_open_joint_positions_rad=(0.0,),
        maximum_subdivision_intervals=4096,
        interval_arithmetic_method_id=interval_method,
        interval_decimal_precision=80,
        maximum_root_bisection_iterations=256,
        subdivision_intervals_used=1,
        subdivision_budget_exhausted=False,
        internal_force_role="fixture",
        trajectory_clearance_m=0.0,
        trajectory_clearance_role="fixture",
        task_frame_source="FIXTURE_TASK_FRAME",
        closure_focus_method="fixture",
        distance_bvh_node_count=1,
        pad_audits=(),
        claim_limitations=RAY_CLOSURE_CLAIM_LIMITATIONS,
        failure_reason=None,
        model_binding_complete=True,
        model_binding_status=MODEL_BINDING_COMPLETE_STATUS,
        object_geometry_sha256=model.geometry_sha256,
        model_contract_sha256=model_sha,
        pad_geometry_sha256=pad_source,
        pad_runtime_geometry_sha256=pad_runtime,
        pad_link_names=PAD_LINKS,
        closing_directions_physical=directions,
        model_contract_canonical_json=canonical,
    )
    return audit, model_sha


def _accepted_row(
    *,
    index: int,
    value: float,
    audit: RayClosureAudit,
) -> tuple[
    CandidateAttemptAudit,
    UniqueV9Evaluation,
    StaticV9AcceptedCandidate,
]:
    parameters = (value, 0.0, 0.0, 0.0, value)
    key = np.asarray(parameters, dtype=">f8").tobytes(order="C").hex()
    candidate = _candidate(value)
    lineage = CandidateLineage(
        attempt_index=index,
        lane=CandidateLane.DIRECT_V9,
        lane_point_index=index,
        sobol_seed=20260820,
        sobol_parameters_unit=parameters,
        anchor_pad_name=None,
        proposal_audit=None,
        proposal_failure_reason=None,
    )
    binding = V9InvocationAuditBinding(
        method_id=RAY_CLOSURE_METHOD_ID,
        parameter_domain_id=CLOSURE_PARAMETER_DOMAIN_ID,
        parameter_layout=V9_LAYOUT,
        requested_parameters_unit=parameters,
        requested_parameter_key_hex=key,
        raw_v9_audit=audit,
    )
    attempt = CandidateAttemptAudit(
        lineage=lineage,
        status=AttemptStatus.STATIC_V9_ACCEPTED,
        v9_parameters_unit=parameters,
        v9_parameter_key_hex=key,
        duplicate_of_attempt_index=None,
        v9_audit=audit,
        invocation_binding=binding,
        v9_failure_reason=None,
        failure_reason=None,
    )
    unique = UniqueV9Evaluation(
        v9_parameters_unit=parameters,
        v9_parameter_key_hex=key,
        first_attempt_index=index,
        lineage=(lineage,),
        candidate=candidate,
        v9_audit=audit,
        invocation_binding=binding,
        status=AttemptStatus.STATIC_V9_ACCEPTED,
        v9_failure_reason=None,
    )
    accepted = StaticV9AcceptedCandidate(
        v9_parameters_unit=parameters,
        v9_parameter_key_hex=key,
        candidate=candidate,
        v9_audit=audit,
        invocation_binding=binding,
        lineage=(lineage,),
    )
    return attempt, unique, accepted


def _generation_result(
    model: ObjectGraspModel,
    values=(0.1, 0.2),
) -> tuple[TopLevelGenerationResult, str]:
    audit, model_sha = _bound_audit(model)
    accepted_rows = tuple(
        _accepted_row(index=index, value=value, audit=audit)
        for index, value in enumerate(values)
    )
    attempts = [row[0] for row in accepted_rows]
    for index in range(len(attempts), 128):
        lineage = CandidateLineage(
            attempt_index=index,
            lane=CandidateLane.SURFACE_PAD_A,
            lane_point_index=index // 4,
            sobol_seed=20260821,
            sobol_parameters_unit=(0.0,) * 6,
            anchor_pad_name=PAD_NAMES[0],
            proposal_audit=None,
            proposal_failure_reason="FIXTURE_NO_PROPOSAL",
        )
        attempts.append(
            CandidateAttemptAudit(
                lineage=lineage,
                status=AttemptStatus.PROPOSAL_REJECTED,
                v9_parameters_unit=None,
                v9_parameter_key_hex=None,
                duplicate_of_attempt_index=None,
                v9_audit=None,
                invocation_binding=None,
                v9_failure_reason=None,
                failure_reason="FIXTURE_NO_PROPOSAL",
            )
        )
    return (
        TopLevelGenerationResult(
            method_id=TOP_LEVEL_METHOD_ID,
            contract_hash_sha256=GENERATION_SHA256,
            total_attempt_budget=128,
            attempts_per_lane=32,
            local_refinement_evaluation_budget=0,
            attempts=tuple(attempts),
            unique_v9_evaluations=tuple(row[1] for row in accepted_rows),
            accepted_candidates=tuple(row[2] for row in accepted_rows),
            v9_evaluation_count=len(accepted_rows),
            duplicate_attempt_count=0,
            proposal_failure_count=128 - len(accepted_rows),
        ),
        model_sha,
    )


def _install_wrench_stub(
    evaluator: TaskWrenchEvaluator,
    calls: list[tuple[float, int, bool, str]],
    *,
    fail_value: float | None = None,
) -> None:
    def evaluate_task_wrench(candidate, scenarios, *, hand_model):
        del hand_model
        value = candidate.independent_joint_positions_rad[0]
        digest = hashlib.sha256(
            np.asarray(scenarios, dtype=">f8").tobytes(order="C")
        ).hexdigest()
        calls.append(
            (
                value,
                id(scenarios),
                bool(scenarios.flags.writeable),
                digest,
            )
        )
        if fail_value is not None and value == fail_value:
            raise RuntimeError("fixture wrench failure")
        hard = 1.0 + value
        return TaskWrenchOnlyEvaluation(
            task_margins=tuple(hard + 0.01 for _ in range(SCENARIO_COUNT)),
            hard_bound_minimum_task_margin=hard,
            peak_normal_force_n=100.0 - value,
            joint_torque_utilization=10.0 - value,
            diagnostics={
                "certified_uncertainty_scope": (
                    FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
                )
            },
        )

    evaluator.evaluate_task_wrench = evaluate_task_wrench


def _complete_collision(accepted):
    return CompleteTrajectoryCollisionCertificate(
        method_id=COMPLETE_CLEARANCE_METHOD_ID,
        claim_scope=COMPLETE_CLEARANCE_SCOPE,
        source_certificate_sha256="b" * 64,
        candidate_sha256=candidate_sha256(accepted.candidate),
        v9_evidence_sha256=v9_evidence_sha256(
            accepted.candidate, accepted.v9_audit
        ),
        model_contract_sha256=accepted.v9_audit.model_contract_sha256,
        trajectory_clearance_lower_bound_m=0.001,
    )


def test_common_design_exactly_once_and_hard_bound_first_diagnostic_rank(
) -> None:
    model = _object_model()
    generation, model_sha = _generation_result(model)
    evaluator = _evaluator(model)
    wrench_calls: list[tuple[float, int, bool, str]] = []
    collision_calls: list[str] = []
    _install_wrench_stub(evaluator, wrench_calls)

    def collision(accepted):
        collision_calls.append(accepted.v9_parameter_key_hex)
        return _complete_collision(accepted)

    pipeline = PostGenerationRankOnlyPipeline(
        expected_generation_contract_sha256=GENERATION_SHA256,
        expected_model_contract_sha256=model_sha,
        wrench_evaluator=evaluator,
        hand_model=_Hand(),
        collision_certifier=collision,
    )
    result = pipeline.evaluate(generation)

    assert len(collision_calls) == 2
    assert len(wrench_calls) == 2
    assert len(set(row[1] for row in wrench_calls)) == 1
    assert all(not row[2] for row in wrench_calls)
    assert all(row[3] == SCENARIO_DESIGN_SHA256 for row in wrench_calls)
    assert len(result.generation_attempt_lineage) == 128
    assert len(result.candidate_records) == 2
    assert result.diagnostic_ranked_keys[0] == (
        generation.accepted_candidates[1].v9_parameter_key_hex
    )
    assert tuple(row.diagnostic_rank for row in result.candidate_records) == (
        2,
        1,
    )
    assert all(
        row.state is CandidateEvaluationState.UNCERTAINTY_SCOPE_INCOMPLETE
        for row in result.candidate_records
    )
    assert not result.formal_ranked_keys
    assert result.selected_candidate is None
    assert result.selected_v9_parameter_key_hex is None
    assert "MISSING_CALIBRATED_NONFRICTION_UNCERTAINTY_BOUNDS" in (
        result.selection_blockers
    )
    assert hashlib.sha256(result.canonical_json_bytes).hexdigest() == (
        result.canonical_sha256
    )
    with pytest.raises(FrozenInstanceError):
        result.canonical_sha256 = "0" * 64


def test_contact_range_policy_stops_before_collision_or_wrench_calls() -> None:
    model = _object_model()
    generation, model_sha = _generation_result(model, values=(0.1,))
    unique = generation.unique_v9_evaluations[0]
    accepted = generation.accepted_candidates[0]
    policy = object.__new__(CertifiedSequentialClosurePolicy)
    policy_unique = replace(
        unique,
        candidate=None,
        status=AttemptStatus.STATIC_V9_POLICY_ACCEPTED,
        sequential_closure_policy=policy,
    )
    policy_attempt = replace(
        generation.attempts[0],
        status=AttemptStatus.STATIC_V9_POLICY_ACCEPTED,
    )
    policy_row = StaticV9AcceptedPolicy(
        v9_parameters_unit=accepted.v9_parameters_unit,
        v9_parameter_key_hex=accepted.v9_parameter_key_hex,
        sequential_closure_policy=policy,
        v9_audit=accepted.v9_audit,
        invocation_binding=accepted.invocation_binding,
        lineage=accepted.lineage,
    )
    policy_generation = replace(
        generation,
        attempts=(policy_attempt, *generation.attempts[1:]),
        unique_v9_evaluations=(policy_unique,),
        accepted_candidates=(),
        accepted_policies=(policy_row,),
    )
    evaluator = _evaluator(model)
    wrench_calls: list[tuple[float, int, bool, str]] = []
    collision_calls: list[str] = []
    _install_wrench_stub(evaluator, wrench_calls)

    def collision(accepted_row):
        collision_calls.append(accepted_row.v9_parameter_key_hex)
        return _complete_collision(accepted_row)

    pipeline = PostGenerationRankOnlyPipeline(
        expected_generation_contract_sha256=GENERATION_SHA256,
        expected_model_contract_sha256=model_sha,
        wrench_evaluator=evaluator,
        hand_model=_Hand(),
        collision_certifier=collision,
    )
    with pytest.raises(
        PostGenerationRankingError,
        match="POLICY_AWARE_COLLISION_AND_WRENCH_REQUIRED",
    ):
        pipeline.evaluate(policy_generation)

    assert collision_calls == []
    assert wrench_calls == []


def test_unresolved_collision_and_wrench_failures_are_retained_without_retry(
) -> None:
    model = _object_model()
    generation, model_sha = _generation_result(model)
    evaluator = _evaluator(model)
    wrench_calls: list[tuple[float, int, bool, str]] = []
    collision_calls: list[str] = []
    _install_wrench_stub(evaluator, wrench_calls, fail_value=0.2)

    def collision(accepted):
        collision_calls.append(accepted.v9_parameter_key_hex)
        if accepted.candidate.independent_joint_positions_rad[0] == 0.1:
            raise RuntimeError("fixture collision failure")
        return object()

    result = PostGenerationRankOnlyPipeline(
        expected_generation_contract_sha256=GENERATION_SHA256,
        expected_model_contract_sha256=model_sha,
        wrench_evaluator=evaluator,
        hand_model=_Hand(),
        collision_certifier=collision,
    ).evaluate(generation)

    assert len(collision_calls) == 2
    assert len(wrench_calls) == 2
    assert result.candidate_records[0].diagnostic_wrench is not None
    assert result.candidate_records[0].state is (
        CandidateEvaluationState.UNRESOLVED_COLLISION
    )
    assert result.candidate_records[1].diagnostic_wrench is None
    assert result.candidate_records[1].state is (
        CandidateEvaluationState.UNRESOLVED_WRENCH
    )
    assert any(
        "fixture collision failure" in blocker
        for blocker in result.candidate_records[0].blockers
    )
    assert any(
        "fixture wrench failure" in blocker
        for blocker in result.candidate_records[1].blockers
    )
    assert not result.all_unique_candidates_resolved
    assert not result.diagnostic_ranked_keys
    assert result.selected_candidate is None


def test_model_contract_mismatch_rejects_before_any_physical_evaluation(
) -> None:
    model = _object_model()
    generation, _model_sha = _generation_result(model)
    evaluator = _evaluator(model)
    wrench_calls: list[tuple[float, int, bool, str]] = []
    collision_calls: list[str] = []
    _install_wrench_stub(evaluator, wrench_calls)

    def collision(accepted):
        collision_calls.append(accepted.v9_parameter_key_hex)
        return _complete_collision(accepted)

    pipeline = PostGenerationRankOnlyPipeline(
        expected_generation_contract_sha256=GENERATION_SHA256,
        expected_model_contract_sha256="f" * 64,
        wrench_evaluator=evaluator,
        hand_model=_Hand(),
        collision_certifier=collision,
    )
    with pytest.raises(
        PostGenerationRankingError,
        match="expected real model",
    ):
        pipeline.evaluate(generation)
    assert not collision_calls
    assert not wrench_calls
