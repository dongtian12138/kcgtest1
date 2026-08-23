import math
from types import SimpleNamespace

import numpy as np
import pytest

import kcg_connector.grasp.robust.robust_wrench as wrench_module
from kcg_connector.grasp.robust.robust_wrench import (
    LinearProgramSolverOptions,
    PolyhedralContactWrenchModel,
    build_polyhedral_contact_wrench_model,
    friction_cone_inner_relative_error,
    maximum_task_wrench_polytope_margin,
    minimum_regular_polygon_edges,
    prescribed_task_scale_burden,
)
from kcg_connector.grasp.robust.uncertainty import (
    lower_tail_cvar,
    scrambled_sobol_scenarios,
    summarize_lower_tail_risk,
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


def _single_contact_model(
    *,
    normal_force_cap_n: float,
    friction: float = 1.0,
    relative_error: float = 0.01,
    edge_multiplier: int = 1,
    point=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),
    tangent=(1.0, 0.0, 0.0),
    origin=(0.0, 0.0, 0.0),
):
    return build_polyhedral_contact_wrench_model(
        contact_points_m=[point],
        inward_normals=[normal],
        tangent_directions=[tangent],
        friction_coefficients=friction,
        normal_force_caps_n=normal_force_cap_n,
        wrench_origin_m=origin,
        maximum_inner_approximation_relative_error=relative_error,
        cone_edge_multiplier=edge_multiplier,
    )


def _lexicographic_load_groups(model):
    signed_contact_force_components = np.vstack(
        (model.contact_force_matrix, -model.contact_force_matrix)
    )
    return (
        model.normal_force_matrix,
        signed_contact_force_components,
    )


def _three_level_degenerate_model():
    """Analytic three-PAD cone with a strict three-level optimum."""

    contact_count = 3
    edges_per_contact = 3
    ray_count = contact_count * edges_per_contact
    grasp_matrix = np.zeros((6, ray_count), dtype=np.float64)
    contact_force_matrix = np.zeros(
        (3 * contact_count, ray_count), dtype=np.float64
    )
    normal_force_matrix = np.zeros(
        (contact_count, ray_count), dtype=np.float64
    )
    ray_owner = []
    for contact_index in range(contact_count):
        for edge_index in range(edges_per_contact):
            angle = 2.0 * math.pi * edge_index / edges_per_contact
            force = np.asarray((math.cos(angle), math.sin(angle), 1.0))
            ray_index = contact_index * edges_per_contact + edge_index
            grasp_matrix[:3, ray_index] = force
            contact_force_matrix[
                3 * contact_index : 3 * contact_index + 3, ray_index
            ] = force
            normal_force_matrix[contact_index, ray_index] = 1.0
            ray_owner.append(contact_index)

    # Individual caps are deliberately inactive.  The algebraic total-normal
    # capacity fixes gamma=1 while leaving many Stage-1 allocations.
    ray_constraint_matrix = np.vstack(
        (normal_force_matrix, np.ones((1, ray_count), dtype=np.float64))
    )
    return PolyhedralContactWrenchModel(
        grasp_matrix=grasp_matrix,
        contact_force_matrix=contact_force_matrix,
        ray_constraint_matrix=ray_constraint_matrix,
        ray_constraint_upper_bounds=np.asarray((10.0, 10.0, 10.0, 1.0)),
        normal_force_matrix=normal_force_matrix,
        ray_owner=tuple(ray_owner),
        contact_points_m=np.zeros((contact_count, 3), dtype=np.float64),
        inward_normals=np.tile((0.0, 0.0, 1.0), (contact_count, 1)),
        tangent_directions=np.tile((1.0, 0.0, 0.0), (contact_count, 1)),
        friction_coefficients=np.ones(contact_count, dtype=np.float64),
        normal_force_caps_n=np.full(contact_count, 10.0, dtype=np.float64),
        wrench_origin_m=np.zeros(3, dtype=np.float64),
        base_edges_per_contact=edges_per_contact,
        edges_per_contact=edges_per_contact,
        maximum_inner_relative_error=friction_cone_inner_relative_error(
            edges_per_contact
        ),
    )


def _three_level_load_groups(model):
    # Six unit-effort generalized coordinates independently observe each PAD's
    # x/y force.  This has exactly the evaluator form +/-T F / effort.
    torque_from_contact_forces = np.zeros((6, 9), dtype=np.float64)
    for contact_index in range(3):
        torque_from_contact_forces[2 * contact_index, 3 * contact_index] = 1.0
        torque_from_contact_forces[
            2 * contact_index + 1, 3 * contact_index + 1
        ] = 1.0
    torque_utilization_from_rays = (
        torque_from_contact_forces @ model.contact_force_matrix
    )
    return (
        model.normal_force_matrix,
        np.vstack(
            (torque_utilization_from_rays, -torque_utilization_from_rays)
        ),
    )


def _certificate(model, *, nominal, vertices):
    result = maximum_task_wrench_polytope_margin(
        model,
        nominal_external_wrench=nominal,
        disturbance_vertices=vertices,
        solver_options=LP_OPTIONS,
        lexicographic_ray_load_groups=_lexicographic_load_groups(model),
    )
    assert result.solver_success, result.solver_message
    assert result.maximum_margin is not None
    assert result.lexicographic_optimal_loads is not None
    assert len(result.lexicographic_stage_results) == 3
    assert all(stage.solver_success for stage in result.lexicographic_stage_results)
    assert all(
        stage.maximum_scaled_equilibrium_residual
        <= LP_OPTIONS.primal_feasibility_tolerance
        for stage in result.lexicographic_stage_results
    )
    assert all(
        stage.maximum_scaled_inequality_violation
        <= LP_OPTIONS.primal_feasibility_tolerance
        for stage in result.lexicographic_stage_results
    )
    assert (
        result.maximum_scaled_equilibrium_residual
        <= LP_OPTIONS.primal_feasibility_tolerance
    )
    assert (
        result.maximum_scaled_inequality_violation
        <= LP_OPTIONS.primal_feasibility_tolerance
    )
    assert result.constraint_scaling_implementation == (
        "EXPLICIT_AUGMENTED_ROW_AND_COLUMN_INF_NORM_V1_"
        "PLUS_HIGHS_INTERNAL_AUTOMATIC"
    )
    return result


def _margin(model, *, nominal, vertices) -> float:
    result = _certificate(model, nominal=nominal, vertices=vertices)
    return float(result.maximum_margin)


def test_edge_count_is_derived_from_certified_inner_error() -> None:
    epsilon = 0.001
    edges = minimum_regular_polygon_edges(epsilon)
    assert friction_cone_inner_relative_error(edges) <= epsilon
    assert friction_cone_inner_relative_error(edges - 1) > epsilon
    model = _single_contact_model(
        normal_force_cap_n=1.0, relative_error=epsilon
    )
    assert model.base_edges_per_contact == edges
    assert model.edges_per_contact == edges


def test_task_margin_is_monotone_in_declared_normal_force_capacity() -> None:
    low_capacity = _single_contact_model(normal_force_cap_n=1.25)
    high_capacity = _single_contact_model(normal_force_cap_n=2.50)
    downward_unit_wrench = [[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]
    low_result = _certificate(
        low_capacity, nominal=np.zeros(6), vertices=downward_unit_wrench
    )
    high_result = _certificate(
        high_capacity, nominal=np.zeros(6), vertices=downward_unit_wrench
    )
    low_margin = low_result.maximum_margin
    high_margin = high_result.maximum_margin
    assert low_margin == pytest.approx(1.25)
    assert high_margin == pytest.approx(2.50)
    assert high_margin >= low_margin
    assert low_result.lexicographic_optimal_loads == pytest.approx((1.25, 1.25))
    assert high_result.lexicographic_optimal_loads == pytest.approx((2.50, 2.50))
    assert (
        high_result.lexicographic_optimal_loads[0]
        >= low_result.lexicographic_optimal_loads[0]
    )


def test_wrench_margin_is_equivariant_under_rigid_rotation_and_translation() -> None:
    point = np.asarray((0.20, -0.03, 0.04))
    origin = np.asarray((0.01, 0.02, -0.05))
    normal = np.asarray((0.0, 0.0, 1.0))
    tangent = np.asarray((1.0, 0.0, 0.0))
    model = _single_contact_model(
        normal_force_cap_n=3.0,
        point=point,
        origin=origin,
        normal=normal,
        tangent=tangent,
    )
    unit_contact_wrench = np.concatenate(
        (normal, np.cross(point - origin, normal))
    )
    disturbance = -unit_contact_wrench
    reference_margin = _margin(
        model, nominal=np.zeros(6), vertices=[disturbance]
    )

    cz, sz = math.cos(0.61), math.sin(0.61)
    cy, sy = math.cos(-0.37), math.sin(-0.37)
    rotation = np.asarray(
        (
            (cz * cy, -sz, cz * sy),
            (sz * cy, cz, sz * sy),
            (-sy, 0.0, cy),
        )
    )
    translation = np.asarray((1.7, -0.8, 0.35))
    transformed = _single_contact_model(
        normal_force_cap_n=3.0,
        point=rotation @ point + translation,
        origin=rotation @ origin + translation,
        normal=rotation @ normal,
        tangent=rotation @ tangent,
    )
    transformed_disturbance = np.concatenate(
        (rotation @ disturbance[:3], rotation @ disturbance[3:])
    )
    transformed_margin = _margin(
        transformed, nominal=np.zeros(6), vertices=[transformed_disturbance]
    )
    assert transformed_margin == pytest.approx(reference_margin, rel=1.0e-9)
    assert transformed_margin == pytest.approx(3.0, rel=1.0e-9)


@pytest.mark.parametrize("length_scale", (1.0e-6, 1.0, 1.0e6))
def test_explicit_equilibration_preserves_margin_across_length_units(
    length_scale: float,
) -> None:
    """Equivalent moment-row units must not change the certified margin."""

    point = np.asarray((0.17, -0.09, 0.04)) * length_scale
    origin = np.asarray((-0.03, 0.02, 0.01)) * length_scale
    normal = np.asarray((0.0, 0.0, 1.0))
    model = _single_contact_model(
        normal_force_cap_n=2.75,
        point=point,
        origin=origin,
        normal=normal,
    )
    contact_wrench = np.concatenate(
        (normal, np.cross(point - origin, normal))
    )
    result = maximum_task_wrench_polytope_margin(
        model,
        nominal_external_wrench=np.zeros(6),
        disturbance_vertices=[-contact_wrench],
        solver_options=LP_OPTIONS,
        lexicographic_ray_load_groups=_lexicographic_load_groups(model),
    )
    assert result.solver_success, result.solver_message
    assert result.maximum_margin == pytest.approx(2.75, rel=1.0e-9)
    assert result.maximum_scaled_equilibrium_residual is not None
    assert (
        result.maximum_scaled_equilibrium_residual
        <= LP_OPTIONS.primal_feasibility_tolerance
    )
    assert not result.equality_augmented_row_inf_norms.flags.writeable
    assert not result.column_inf_norms_after_row_scaling.flags.writeable
    assert result.lexicographic_optimal_loads == pytest.approx(
        (2.75, 2.75), rel=1.0e-9
    )
    assert len(result.lexicographic_stage_results) == 3
    assert all(
        not stage.equality_augmented_row_inf_norms.flags.writeable
        and not stage.column_inf_norms_after_row_scaling.flags.writeable
        for stage in result.lexicographic_stage_results
    )


def test_three_level_degeneracy_has_unique_stable_lexicographic_certificate() -> None:
    model = _three_level_degenerate_model()
    arguments = {
        "nominal_external_wrench": np.zeros(6),
        "disturbance_vertices": [[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]],
        "solver_options": LP_OPTIONS,
        "lexicographic_ray_load_groups": _three_level_load_groups(model),
    }
    first = maximum_task_wrench_polytope_margin(model, **arguments)
    second = maximum_task_wrench_polytope_margin(model, **arguments)
    assert first.solver_success, first.solver_message
    assert second.solver_success, second.solver_message
    assert first.maximum_margin == pytest.approx(1.0)
    assert first.lexicographic_optimal_loads == pytest.approx(
        (1.0 / 3.0, 0.0), abs=LP_OPTIONS.primal_feasibility_tolerance
    )
    assert second.lexicographic_optimal_loads == first.lexicographic_optimal_loads
    assert tuple(
        stage.stage_name for stage in first.lexicographic_stage_results
    ) == (
        "MAXIMIZE_SHARED_TASK_MARGIN",
        "MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_0",
        "MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_1",
    )
    np.testing.assert_allclose(
        first.ray_coefficients_by_vertex,
        np.full((1, 9), 1.0 / 9.0),
        atol=LP_OPTIONS.primal_feasibility_tolerance,
    )
    np.testing.assert_allclose(
        first.contact_forces_by_vertex,
        np.asarray((((0.0, 0.0, 1.0 / 3.0),) * 3,)),
        atol=LP_OPTIONS.primal_feasibility_tolerance,
    )
    final_coefficients = first.ray_coefficients_by_vertex[0]
    final_group_maxima = tuple(
        float(np.max(group @ final_coefficients))
        for group in arguments["lexicographic_ray_load_groups"]
    )
    assert final_group_maxima == pytest.approx(
        first.lexicographic_optimal_loads,
        abs=LP_OPTIONS.primal_feasibility_tolerance,
    )
    np.testing.assert_array_equal(
        first.ray_coefficients_by_vertex,
        second.ray_coefficients_by_vertex,
    )
    np.testing.assert_array_equal(
        first.contact_forces_by_vertex,
        second.contact_forces_by_vertex,
    )


@pytest.mark.parametrize("failed_call", (2, 3))
def test_later_lexicographic_stage_failure_discards_margin_and_allocations(
    monkeypatch,
    failed_call: int,
) -> None:
    model = _single_contact_model(normal_force_cap_n=2.0)
    real_linprog = wrench_module.linprog
    calls = 0

    def fail_selected_stage(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            return SimpleNamespace(
                success=False,
                status=2,
                message="synthetic later-stage failure",
            )
        return real_linprog(*args, **kwargs)

    monkeypatch.setattr(wrench_module, "linprog", fail_selected_stage)
    result = maximum_task_wrench_polytope_margin(
        model,
        nominal_external_wrench=np.zeros(6),
        disturbance_vertices=[[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]],
        solver_options=LP_OPTIONS,
        lexicographic_ray_load_groups=_lexicographic_load_groups(model),
    )
    assert calls == failed_call
    assert not result.solver_success
    assert result.maximum_margin is None
    assert result.lexicographic_optimal_loads is None
    assert result.ray_coefficients_by_vertex is None
    assert result.contact_forces_by_vertex is None
    assert result.normal_forces_by_vertex is None
    assert len(result.lexicographic_stage_results) == failed_call
    assert not result.lexicographic_stage_results[-1].solver_success


def test_cone_edge_doubling_converges_monotonically_from_inside() -> None:
    # epsilon=0.30 derives four edges.  The chosen direction lies halfway
    # between the 4- and 8-edge rays, and is exactly a 16-edge ray.
    direction_angle = math.pi / 8.0
    tangent_disturbance = [
        math.cos(direction_angle),
        math.sin(direction_angle),
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    nominal = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
    models = [
        _single_contact_model(
            normal_force_cap_n=1.0,
            friction=1.0,
            relative_error=0.30,
            edge_multiplier=multiplier,
        )
        for multiplier in (1, 2, 4)
    ]
    assert [model.edges_per_contact for model in models] == [4, 8, 16]
    margins = [
        _margin(model, nominal=nominal, vertices=[tangent_disturbance])
        for model in models
    ]
    assert margins[0] < margins[1] < margins[2]
    assert 1.0 - margins[1] < 1.0 - margins[0]
    assert margins[2] == pytest.approx(1.0, rel=1.0e-9)


def test_arbitrary_contact_count_is_not_hard_coded_to_three() -> None:
    points = np.asarray(
        (
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        )
    )
    normals = -points / np.linalg.norm(points, axis=1)[:, None]
    tangents = []
    for normal in normals:
        trial = np.asarray((1.0, 0.0, 0.0))
        if abs(float(trial @ normal)) > 0.9:
            trial = np.asarray((0.0, 1.0, 0.0))
        tangents.append(trial)
    model = build_polyhedral_contact_wrench_model(
        contact_points_m=points,
        inward_normals=normals,
        tangent_directions=tangents,
        friction_coefficients=[0.4, 0.5, 0.6, 0.7],
        normal_force_caps_n=[3.0, 4.0, 5.0, 6.0],
        wrench_origin_m=(0.0, 0.0, 0.0),
        maximum_inner_approximation_relative_error=0.01,
    )
    assert model.contact_count == 4
    assert model.grasp_matrix.shape == (6, 4 * model.edges_per_contact)


def test_lower_tail_cvar_matches_fractional_hand_calculation() -> None:
    # Four equally weighted outcomes and alpha=3/8 give 1.5 samples of mass:
    # (1 + 0.5 * 2) / 1.5 = 4/3.
    margins = [8.0, 1.0, 4.0, 2.0]
    value = lower_tail_cvar(margins, lower_tail_fraction=3.0 / 8.0)
    assert value == pytest.approx(4.0 / 3.0)
    summary = summarize_lower_tail_risk(
        margins, lower_tail_fraction=3.0 / 8.0
    )
    assert summary.lower_tail_cvar == pytest.approx(4.0 / 3.0)
    assert summary.lower_tail_var == pytest.approx(2.0)
    assert summary.hard_minimum == pytest.approx(1.0)
    assert summary.mean_margin == pytest.approx(3.75)


def test_scrambled_sobol_scenarios_are_seeded_balanced_and_bounded() -> None:
    bounds = {"mu": (0.35, 0.65), "mass_kg": (0.25, 0.40)}
    first = scrambled_sobol_scenarios(bounds, scenario_count=8, seed=20260820)
    second = scrambled_sobol_scenarios(
        dict(reversed(tuple(bounds.items()))),
        scenario_count=8,
        seed=20260820,
    )
    assert first.parameter_names == ("mass_kg", "mu")
    np.testing.assert_array_equal(first.unit_scenarios, second.unit_scenarios)
    np.testing.assert_array_equal(first.values, second.values)
    assert np.all(first.values >= first.lower_bounds)
    assert np.all(first.values <= first.upper_bounds)
    assert len(first.records()) == 8


def _prescribed_arguments(model, *, scale):
    return {
        "nominal_external_wrench": (0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        "disturbance_vertices": [[0.0, 0.0, -0.1, 0.0, 0.0, 0.0]],
        "prescribed_task_scale": scale,
        "solver_options": LP_OPTIONS,
        "lexicographic_ray_load_groups": _three_level_load_groups(model),
    }


def test_prescribed_task_scale_burden_pins_scale_not_maximum_margin() -> None:
    model = _three_level_degenerate_model()
    margin_result = maximum_task_wrench_polytope_margin(
        model,
        nominal_external_wrench=(0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        disturbance_vertices=[[0.0, 0.0, -0.1, 0.0, 0.0, 0.0]],
        solver_options=LP_OPTIONS,
        lexicographic_ray_load_groups=_three_level_load_groups(model),
    )
    assert margin_result.solver_success, margin_result.solver_message
    assert margin_result.maximum_margin == pytest.approx(5.0)
    assert margin_result.lexicographic_optimal_loads == pytest.approx(
        (1.0 / 3.0, 0.0), abs=LP_OPTIONS.primal_feasibility_tolerance
    )
    at_max = prescribed_task_scale_burden(
        model,
        **_prescribed_arguments(
            model, scale=float(margin_result.maximum_margin)
        ),
    )
    assert at_max.solver_success, at_max.solver_message
    assert at_max.feasible_at_prescribed_scale
    assert at_max.prescribed_task_scale == pytest.approx(5.0)
    assert (
        at_max.peak_normal_force_n,
        at_max.peak_joint_torque_utilization,
    ) == pytest.approx(
        margin_result.lexicographic_optimal_loads,
        abs=LP_OPTIONS.primal_feasibility_tolerance,
    )
    at_one = prescribed_task_scale_burden(
        model, **_prescribed_arguments(model, scale=1.0)
    )
    assert at_one.solver_success, at_one.solver_message
    assert at_one.feasible_at_prescribed_scale
    assert at_one.prescribed_task_scale == pytest.approx(1.0)
    assert at_one.peak_normal_force_n == pytest.approx(0.2, abs=1.0e-8)
    assert at_one.peak_joint_torque_utilization == pytest.approx(0.0, abs=1.0e-8)
    assert at_one.peak_normal_force_n < (
        margin_result.lexicographic_optimal_loads[0] - 1.0e-6
    )
    assert tuple(
        stage.stage_name for stage in at_one.lexicographic_stage_results
    ) == (
        "CHECK_FEASIBILITY_AT_PRESCRIBED_TASK_SCALE",
        "MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_0_AT_PRESCRIBED_TASK_SCALE",
        "MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_1_AT_PRESCRIBED_TASK_SCALE",
    )


def test_prescribed_task_scale_infeasible_scale_fails_closed() -> None:
    model = _three_level_degenerate_model()
    result = prescribed_task_scale_burden(
        model, **_prescribed_arguments(model, scale=6.0)
    )
    assert not result.solver_success
    assert not result.feasible_at_prescribed_scale
    assert result.peak_normal_force_n is None
    assert result.peak_joint_torque_utilization is None
    assert (
        result.lexicographic_stage_results[0].stage_name
        == "CHECK_FEASIBILITY_AT_PRESCRIBED_TASK_SCALE"
    )
    assert "CHECK_FEASIBILITY_AT_PRESCRIBED_TASK_SCALE" in result.solver_message


@pytest.mark.parametrize("scale", (-1.0, float("nan"), float("inf")))
def test_prescribed_task_scale_rejects_invalid_scale(scale) -> None:
    model = _three_level_degenerate_model()
    with pytest.raises(ValueError):
        prescribed_task_scale_burden(
            model, **_prescribed_arguments(model, scale=scale)
        )
