from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.grasp_optimizer import (
    CARTSGraspOptimizer,
    GraspCandidate,
    OptimizationConfig,
    OptimizationError,
    PlannedPadContact,
    WrenchEvaluation,
    deterministic_sobol,
)
from kcg_connector.grasp.robust.hand_model import (
    HandModelError,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.pareto_ranker import (
    CandidateMetrics,
    ScoredCandidate,
    lower_tail_cvar,
    qmc_lower_tail_mean,
    rank_candidates,
)


REPOSITORY = Path(__file__).resolve().parents[4]
HAND_URDF = REPOSITORY / "src/iiwa_description/urdf/hand.xacro"


def _pad_contract() -> dict:
    return {
        "schema_version": "test_hand_pad_geometry_v1",
        "pads": [
            {
                "name": "finger_1_pad",
                "finger": "f1",
                "link": "f1Link3",
                "geometry_source": "URDF_COLLISION",
                "contact_normal_pad": [1.0, 0.0, 0.0],
                "normal_force_capacity_n": 15.0,
            },
            {
                "name": "finger_2_pad",
                "finger": "f2",
                "link": "f2Link2",
                "geometry_source": "URDF_COLLISION",
                "contact_normal_pad": [1.0, 0.0, 0.0],
                "normal_force_capacity_n": 15.0,
            },
            {
                "name": "finger_3_pad",
                "finger": "f3",
                "link": "f3Link3",
                "geometry_source": "URDF_COLLISION",
                "contact_normal_pad": [1.0, 0.0, 0.0],
                "normal_force_capacity_n": 15.0,
            },
        ],
    }


@pytest.fixture(scope="module")
def hand_model() -> ThreeFingerHandModel:
    return ThreeFingerHandModel.from_urdf(
        HAND_URDF,
        pad_geometry_contract=_pad_contract(),
    )


def test_real_urdf_fk_mimic_and_pad_frames_are_deterministic(
    hand_model: ThreeFingerHandModel,
) -> None:
    assert hand_model.independent_joint_names == (
        "f1j1",
        "f1j2",
        "f2j1",
        "f3j2",
    )
    positions = (0.30, 0.45, 0.50, 0.55)
    resolved = hand_model.resolve_joint_positions(positions)
    assert resolved["f1j3"] == pytest.approx(resolved["f1j2"], abs=0.0)
    assert resolved["f2j2"] == pytest.approx(resolved["f2j1"], abs=0.0)
    assert resolved["f3j1"] == pytest.approx(resolved["f1j1"], abs=0.0)
    assert resolved["f3j3"] == pytest.approx(resolved["f3j2"], abs=0.0)

    first = hand_model.forward_kinematics(positions)
    second = hand_model.forward_kinematics(positions)
    assert first.keys() == second.keys()
    for link_name in first:
        assert np.array_equal(first[link_name], second[link_name])

    pad_first = hand_model.pad_transforms(positions)
    pad_second = hand_model.pad_transforms(positions)
    assert set(pad_first) == {"finger_1_pad", "finger_2_pad", "finger_3_pad"}
    for pad_name in pad_first:
        assert np.array_equal(pad_first[pad_name], pad_second[pad_name])

    jacobian_first = hand_model.geometric_jacobian(
        "f3Link3", positions, point_local_m=(0.001, -0.002, 0.003)
    )
    jacobian_second = hand_model.geometric_jacobian(
        "f3Link3", positions, point_local_m=(0.001, -0.002, 0.003)
    )
    assert jacobian_first.shape == (6, 4)
    assert np.array_equal(jacobian_first, jacobian_second)


def test_joint_position_velocity_limits_and_kinematic_normal_domain(
    hand_model: ThreeFingerHandModel,
) -> None:
    lower, upper = hand_model.joint_limit_vectors()
    assert np.array_equal(lower, np.zeros(4))
    assert np.allclose(upper, (1.57, 1.3963, 1.3963, 1.3963), atol=0.0)
    assert hand_model.within_joint_limits((0.30, 0.45, 0.50, 0.55))
    assert not hand_model.within_joint_limits((1.58, 0.45, 0.50, 0.55))
    with pytest.raises(HandModelError, match="violates"):
        hand_model.resolve_joint_positions((0.30, 1.50, 0.50, 0.55))
    with pytest.raises(HandModelError, match="exceeds"):
        hand_model.resolve_joint_velocities((4.01, 0.1, 0.1, 0.1))

    domains = hand_model.pad_kinematic_normal_domains(
        (0.30, 0.45, 0.50, 0.55),
        (0.1, 0.1, 0.1, 0.1),
    )
    for domain in domains.values():
        velocity = np.asarray(domain.closing_velocity_base_m_s)
        assert np.linalg.norm(velocity) > 0.0
        approached_normal = -velocity / np.linalg.norm(velocity)
        separating_normal = -approached_normal
        assert domain.contains(approached_normal)
        assert not domain.contains(separating_normal)
        assert domain.numerical_tolerance_m_s == pytest.approx(
            64.0 * np.finfo(np.float64).eps * np.linalg.norm(velocity),
            rel=8.0 * np.finfo(np.float64).eps,
        )


def test_lexicographic_ranking_is_not_a_mixed_unit_weighted_sum() -> None:
    qmc_statistic_winner = CandidateMetrics(
        hard_bound_minimum_task_margin=0.1,
        qmc_lower_tail_mean_task_margin=2.0,
        peak_normal_force_n=1.0e12,
        joint_torque_utilization=1.0e9,
        trajectory_clearance_m=-1.0,
    )
    hard_bound_winner = CandidateMetrics(
        hard_bound_minimum_task_margin=100.0,
        qmc_lower_tail_mean_task_margin=1.0,
        peak_normal_force_n=1.0e-12,
        joint_torque_utilization=1.0e-12,
        trajectory_clearance_m=100.0,
    )
    rows = (
        ScoredCandidate("qmc", qmc_statistic_winner, source_index=1),
        ScoredCandidate("hard_bound", hard_bound_winner, source_index=0),
    )
    assert rank_candidates(rows)[0].candidate == "hard_bound"

    tied_hard_bound = replace(
        qmc_statistic_winner,
        hard_bound_minimum_task_margin=hard_bound_winner.hard_bound_minimum_task_margin,
    )
    assert rank_candidates(
        (
            ScoredCandidate("qmc", tied_hard_bound, source_index=1),
            ScoredCandidate("hard_bound", hard_bound_winner, source_index=0),
        )
    )[0].candidate == "qmc"

    force_unit_rescaled = tuple(
        ScoredCandidate(
            row.candidate,
            replace(row.metrics, peak_normal_force_n=row.metrics.peak_normal_force_n * 1.0e9),
            source_index=row.source_index,
        )
        for row in rows
    )
    clearance_unit_rescaled = tuple(
        ScoredCandidate(
            row.candidate,
            replace(row.metrics, trajectory_clearance_m=row.metrics.trajectory_clearance_m * 1.0e-6),
            source_index=row.source_index,
        )
        for row in rows
    )
    assert rank_candidates(force_unit_rescaled)[0].candidate == "hard_bound"
    assert rank_candidates(clearance_unit_rescaled)[0].candidate == "hard_bound"
    expected_lower_tail_mean = pytest.approx(1.0 / 3.0)
    assert qmc_lower_tail_mean((0.0, 1.0, 2.0, 3.0), 0.375) == (
        expected_lower_tail_mean
    )
    assert lower_tail_cvar((0.0, 1.0, 2.0, 3.0), 0.375) == (
        expected_lower_tail_mean
    )


def test_production_metric_names_express_interval_not_probability_semantics() -> None:
    metric_names = tuple(field.name for field in fields(CandidateMetrics))
    assert metric_names[:2] == (
        "hard_bound_minimum_task_margin",
        "qmc_lower_tail_mean_task_margin",
    )
    assert all("cvar" not in name.lower() for name in metric_names)


def test_candidate_contract_has_no_model_or_historical_lookup_fields() -> None:
    forbidden = {
        "object_id",
        "candidate_id",
        "history_run_id",
        "h_chain_id",
        "stored_contact_coordinates",
    }
    for contract_type in (GraspCandidate, PlannedPadContact, CandidateMetrics):
        assert forbidden.isdisjoint(field.name for field in fields(contract_type))


def test_sobol_and_optimizer_are_reproducible(
    hand_model: ThreeFingerHandModel,
) -> None:
    first_design = deterministic_sobol(dimension=2, count=8, seed=71)
    second_design = deterministic_sobol(dimension=2, count=8, seed=71)
    assert np.array_equal(first_design, second_design)

    class Surface:
        parameter_dimension = 1

        def candidate_from_unit_parameters(self, parameters_unit, model):
            value = float(parameters_unit[0])
            contacts = tuple(
                PlannedPadContact(
                    pad_name=name,
                    position_object_m=(0.0, 0.0, float(index) * 0.001),
                    path_local_free_side_normal_object=(1.0, 0.0, 0.0),
                )
                for index, name in enumerate(model.pads)
            )
            return GraspCandidate.from_matrix(
                object_from_hand=np.eye(4),
                independent_joint_positions_rad=(
                    0.2 + 0.1 * value,
                    0.4,
                    0.4,
                    0.4,
                ),
                planned_pad_contacts=contacts,
                internal_normal_forces_n=(3.0, 3.0, 3.0),
            )

    class Evaluator:
        uncertainty_dimension = 1

        def evaluate(
            self,
            candidate,
            scenario_parameters_unit,
            *,
            surface_model,
            hand_model,
        ):
            del surface_model, hand_model
            decision = candidate.independent_joint_positions_rad[0]
            margins = tuple(
                decision - 0.01 * float(row[0]) for row in scenario_parameters_unit
            )
            return WrenchEvaluation(
                task_margins=margins,
                peak_normal_force_n=4.0 - decision,
                joint_torque_utilization=0.2,
                trajectory_clearance_m=0.001,
                hard_bound_minimum_task_margin=decision - 0.01,
            )

    optimizer = CARTSGraspOptimizer(
        OptimizationConfig(
            candidate_budget=4,
            continuous_refinement_multistarts=1,
            candidate_sobol_seed=19,
            maximum_solver_iterations=8,
            relative_objective_tolerance=1.0e-6,
            scenario_count=4,
            scenario_sobol_seed=23,
            lower_tail_fraction=0.5,
        )
    )
    first = optimizer.optimize(
        surface_model=Surface(), wrench_evaluator=Evaluator(), hand_model=hand_model
    )
    second = optimizer.optimize(
        surface_model=Surface(), wrench_evaluator=Evaluator(), hand_model=hand_model
    )
    assert first.candidate_design_unit == second.candidate_design_unit
    assert first.scenario_design_unit == second.scenario_design_unit
    assert first.selected.parameters_unit == second.selected.parameters_unit
    assert first.ranked[0].metrics == second.ranked[0].metrics

    class NegativeClearanceEvaluator(Evaluator):
        def evaluate(self, *args, **kwargs):
            value = super().evaluate(*args, **kwargs)
            return replace(
                value,
                trajectory_clearance_m=np.nextafter(0.0, -np.inf),
            )

    with pytest.raises(OptimizationError, match="no feasible grasp"):
        optimizer.optimize(
            surface_model=Surface(),
            wrench_evaluator=NegativeClearanceEvaluator(),
            hand_model=hand_model,
        )

    class MissingHardBoundEvaluator(Evaluator):
        def evaluate(self, *args, **kwargs):
            value = super().evaluate(*args, **kwargs)
            return replace(value, hard_bound_minimum_task_margin=None)

    with pytest.raises(OptimizationError, match="hard_bound_minimum_task_margin"):
        optimizer.optimize(
            surface_model=Surface(),
            wrench_evaluator=MissingHardBoundEvaluator(),
            hand_model=hand_model,
        )


def test_shared_optimization_mapping_is_fail_closed() -> None:
    complete = {
        "candidate_optimization": {
            "candidate_budget": 8,
            "continuous_refinement_multistarts": 2,
            "sobol_seed": 31,
            "maximum_solver_iterations": 12,
            "relative_objective_tolerance": 1.0e-8,
            "clearance_feasibility_policy": "NONNEGATIVE_CERTIFIED_LOWER_BOUND",
            "selection": "LEXICOGRAPHIC",
            "selection_order": [
                "hard_bound_minimum_task_margin",
                "qmc_lower_tail_mean_task_margin",
                "minimum_peak_normal_force_n",
                "minimum_joint_torque_utilization",
                "maximum_trajectory_clearance_m",
            ],
        },
        "uncertainty": {
            "scenario_design": "SCRAMBLED_SOBOL",
            "scenario_count": 8,
            "sobol_seed": 37,
            "lower_tail_fraction": 0.1,
        },
    }
    parsed = OptimizationConfig.from_mapping(complete)
    assert parsed.candidate_sobol_seed == 31
    assert parsed.scenario_sobol_seed == 37

    wrong_clearance_policy = {
        **complete,
        "candidate_optimization": {
            **complete["candidate_optimization"],
            "clearance_feasibility_policy": "ALLOW_ONE_MICROMETER_PENETRATION",
        },
    }
    with pytest.raises(ValueError, match="nonnegative certified lower bound"):
        OptimizationConfig.from_mapping(wrong_clearance_policy)

    incomplete = {
        **complete,
        "candidate_optimization": {
            key: value
            for key, value in complete["candidate_optimization"].items()
            if key != "candidate_budget"
        },
    }
    with pytest.raises(ValueError, match="candidate_budget"):
        OptimizationConfig.from_mapping(incomplete)

    legacy_probability_order = {
        **complete,
        "candidate_optimization": {
            **complete["candidate_optimization"],
            "selection_order": [
                "lower_tail_cvar_task_margin",
                "hard_bound_minimum_task_margin",
                "minimum_peak_normal_force_n",
                "minimum_joint_torque_utilization",
                "maximum_trajectory_clearance_m",
            ],
        },
    }
    with pytest.raises(ValueError, match="selection_order"):
        OptimizationConfig.from_mapping(legacy_probability_order)
