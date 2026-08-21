from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from kcg_connector.grasp.robust.force_controller import (
    ForceAllocationSolverOptions,
    JointTorqueLinearConstraint,
    OnlineControlObservation,
    PassiveImpedanceNumerics,
    PassiveObjectImpedance,
    allocate_grasp_forces,
)
from kcg_connector.grasp.robust.robust_wrench import (
    build_polyhedral_contact_wrench_model,
)


IMPEDANCE_NUMERICS = PassiveImpedanceNumerics.from_mapping(
    {
        "matrix_symmetry_absolute_tolerance": 1.0e-12,
        "semidefinite_eigenvalue_tolerance": 1.0e-12,
        "rotation_orthogonality_tolerance": 1.0e-12,
        "homogeneous_row_tolerance": 1.0e-12,
        "passivity_balance_tolerance": 1.0e-12,
    }
)

QP_OPTIONS = ForceAllocationSolverOptions.from_mapping(
    {
        "solver": "SCIPY_SLSQP_WITH_HIGHS_FEASIBILITY",
        "constraint_scaling": "CALLER_SUPPLIED_WRENCH_SCALES",
        "maximum_iterations": 2000,
        "objective_tolerance": 1.0e-11,
        "equality_tolerance": 1.0e-8,
        "inequality_tolerance": 1.0e-8,
        "linear_independence_tolerance": 1.0e-10,
        "feasibility_dual_tolerance": 1.0e-9,
        "regularization": 1.0e-8,
        "physical_acceptance_gate": False,
    }
)


def test_real_shared_config_supplies_all_controller_numerics() -> None:
    repository = Path(__file__).resolve().parents[4]
    document = yaml.safe_load(
        (repository / "src/kcg_connector/config/carts_grasp_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    impedance = PassiveImpedanceNumerics.from_mapping(
        document["passive_impedance_numerics"]
    )
    force_qp = ForceAllocationSolverOptions.from_mapping(document["force_qp"])
    assert impedance == IMPEDANCE_NUMERICS
    assert force_qp == QP_OPTIONS
    assert document["numerical_convergence"]["physical_acceptance_gate"] is False


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def _symmetric_model(*, normal_force_cap_n: float, rotation=None, translation=None):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    translation = np.zeros(3) if translation is None else np.asarray(translation)
    angles = 2.0 * math.pi * np.arange(3) / 3.0
    points = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(3)))
    normals = -points
    tangents = np.tile((0.0, 0.0, 1.0), (3, 1))
    return build_polyhedral_contact_wrench_model(
        contact_points_m=(points @ rotation.T + translation),
        inward_normals=(normals @ rotation.T),
        tangent_directions=(tangents @ rotation.T),
        friction_coefficients=[0.6, 0.6, 0.6],
        normal_force_caps_n=[normal_force_cap_n] * 3,
        wrench_origin_m=translation,
        maximum_inner_approximation_relative_error=0.05,
    )


def _valid_observation(*, pose=None, twist=None) -> OnlineControlObservation:
    return OnlineControlObservation.from_mapping(
        {
            "estimated_pose_world_from_object": (
                np.eye(4) if pose is None else pose
            ),
            "estimated_twist_world": (
                np.zeros(6) if twist is None else twist
            ),
            "estimator_source": "rgbd_wrist_ft_estimator",
            "joint_positions": [0.1, -0.2],
            "joint_velocities": [0.0, 0.0],
            "joint_torques": [0.3, -0.1],
            "wrist_wrench": np.zeros(6),
            "tactile_measurements": [0.2, 0.3, 0.4],
            "timestamp_s": 2.0,
        },
        numerics=IMPEDANCE_NUMERICS,
    )


def _allocate(
    model,
    *,
    preferred,
    desired=None,
    joint_constraint=None,
    wrench_scaling=None,
):
    observation = _valid_observation()
    result = allocate_grasp_forces(
        model,
        desired_object_wrench=(np.zeros(6) if desired is None else desired),
        preferred_normal_forces=preferred,
        contact_force_quadratic_weights=np.ones(3 * model.contact_count),
        normal_tracking_quadratic_weights=np.ones(model.contact_count),
        wrench_scaling=(
            np.ones(6) if wrench_scaling is None else wrench_scaling
        ),
        joint_torque_constraint=joint_constraint,
        solver_options=QP_OPTIONS,
        truth_firewall_audit=observation.truth_firewall_audit,
    )
    assert result.solver_success, result.solver_message
    assert QP_OPTIONS.physical_acceptance_gate is False
    assert result.maximum_solver_coordinate_equality_residual is not None
    assert result.maximum_solver_coordinate_inequality_violation is not None
    assert (
        result.maximum_solver_coordinate_equality_residual
        <= QP_OPTIONS.equality_tolerance
    )
    assert (
        result.maximum_solver_coordinate_inequality_violation
        <= QP_OPTIONS.inequality_tolerance
    )
    assert result.solver_coordinate_equality_residuals is not None
    assert result.solver_coordinate_inequality_residuals is not None
    assert len(result.inequality_row_labels) == len(
        result.solver_coordinate_inequality_residuals
    )
    assert len(result.inequality_augmented_row_inf_norms) == len(
        result.inequality_row_labels
    )
    return result


def test_qp_balances_wrench_and_recovers_symmetric_internal_force() -> None:
    model = _symmetric_model(normal_force_cap_n=4.0)
    result = _allocate(model, preferred=[2.0, 2.0, 2.0])
    assert result.physical_equilibrium_force_residual_n is not None
    assert result.physical_equilibrium_moment_residual_nm is not None
    np.testing.assert_array_equal(
        result.physical_equilibrium_force_residual_n,
        result.achieved_object_wrench[:3],
    )
    np.testing.assert_array_equal(
        result.physical_equilibrium_moment_residual_nm,
        result.achieved_object_wrench[3:],
    )
    np.testing.assert_allclose(
        result.normal_forces / model.normal_force_caps_n,
        np.repeat(
            result.normal_forces.mean() / model.normal_force_caps_n[0], 3
        ),
        rtol=0.0,
        atol=QP_OPTIONS.equality_tolerance,
    )
    assert result.normal_forces.mean() > 0.0
    assert result.solver_coordinate_internal_wrench_residuals is not None
    assert np.max(np.abs(result.solver_coordinate_internal_wrench_residuals)) <= (
        QP_OPTIONS.equality_tolerance
    )
    assert result.physical_internal_force_residual_n is not None
    assert result.physical_internal_moment_residual_nm is not None
    np.testing.assert_array_equal(
        np.concatenate(
            (
                result.physical_internal_force_residual_n,
                result.physical_internal_moment_residual_nm,
            )
        ),
        model.grasp_matrix @ result.internal_ray_component,
    )
    assert not hasattr(result, "maximum_equilibrium_residual")
    assert not hasattr(result, "maximum_inequality_violation")
    assert not hasattr(result, "maximum_internal_wrench_residual")
    assert result.truth_firewall_audit is not None
    assert result.truth_firewall_audit.ground_truth_pose_used is False


def test_normal_force_cap_relaxation_is_monotone_without_a_pass_threshold() -> None:
    low_model = _symmetric_model(normal_force_cap_n=0.75)
    high_model = _symmetric_model(normal_force_cap_n=1.50)
    low = _allocate(
        low_model,
        preferred=[3.0, 3.0, 3.0],
    )
    high = _allocate(
        high_model,
        preferred=[3.0, 3.0, 3.0],
    )
    np.testing.assert_array_equal(
        low.physical_normal_cap_residuals_n,
        low.normal_forces - low_model.normal_force_caps_n,
    )
    np.testing.assert_array_equal(
        high.physical_normal_cap_residuals_n,
        high.normal_forces - high_model.normal_force_caps_n,
    )
    assert np.all(high.normal_forces > low.normal_forces)
    assert high.objective_value < low.objective_value


def test_optional_joint_torque_linear_constraint_is_enforced() -> None:
    model = _symmetric_model(normal_force_cap_n=4.0)
    # The first row projects the first contact force onto its inward normal.
    torque_map = np.zeros((1, 9))
    torque_map[0, :3] = model.inward_normals[0]
    constraint = JointTorqueLinearConstraint.from_arrays(
        contact_force_to_joint_torque=torque_map,
        bias_torque=[0.0],
        lower_torque=[0.0],
        upper_torque=[0.80],
    )
    result = _allocate(
        model,
        preferred=[3.0, 3.0, 3.0],
        joint_constraint=constraint,
    )
    assert result.joint_torques is not None
    np.testing.assert_array_equal(
        result.physical_joint_upper_torque_residuals_nm,
        result.joint_torques - constraint.upper_torque,
    )
    np.testing.assert_array_equal(
        result.physical_joint_lower_torque_residuals_nm,
        constraint.lower_torque - result.joint_torques,
    )
    upper_row = result.inequality_row_labels.index("joint_upper[0]")
    assert abs(result.solver_coordinate_inequality_residuals[upper_row]) <= (
        QP_OPTIONS.inequality_tolerance
    )
    assert result.normal_forces[1] == pytest.approx(
        result.normal_forces[2], abs=QP_OPTIONS.equality_tolerance
    )
    assert result.normal_forces[1] >= result.normal_forces[0]


def test_caller_wrench_length_scale_choice_is_coordinate_equivalent() -> None:
    model = _symmetric_model(normal_force_cap_n=4.0)
    desired = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.30))
    one_metre_characteristic_length = np.ones(6)
    one_millimetre_characteristic_length = np.asarray(
        (1.0, 1.0, 1.0, 1.0e-3, 1.0e-3, 1.0e-3)
    )
    metre = _allocate(
        model,
        preferred=[2.0, 2.0, 2.0],
        desired=desired,
        wrench_scaling=one_metre_characteristic_length,
    )
    millimetre = _allocate(
        model,
        preferred=[2.0, 2.0, 2.0],
        desired=desired,
        wrench_scaling=one_millimetre_characteristic_length,
    )
    np.testing.assert_allclose(
        millimetre.contact_forces / model.normal_force_caps_n[:, None],
        metre.contact_forces / model.normal_force_caps_n[:, None],
        rtol=0.0,
        atol=QP_OPTIONS.equality_tolerance,
    )
    np.testing.assert_allclose(
        millimetre.equality_augmented_row_inf_norms_after_wrench_scaling
        * one_millimetre_characteristic_length,
        metre.equality_augmented_row_inf_norms_after_wrench_scaling
        * one_metre_characteristic_length,
        rtol=0.0,
        atol=QP_OPTIONS.equality_tolerance,
    )
    np.testing.assert_allclose(
        millimetre.solver_coordinate_equality_residuals,
        metre.solver_coordinate_equality_residuals,
        rtol=0.0,
        atol=QP_OPTIONS.equality_tolerance,
    )


@pytest.mark.parametrize("row_multiplier", (1.0e-6, 1.0e6))
def test_inequality_augmented_row_scaling_is_representation_invariant(
    row_multiplier: float,
) -> None:
    model = _symmetric_model(normal_force_cap_n=4.0)
    additional_row = model.normal_force_matrix[[0]]
    additional_bound = np.asarray((3.0,))
    augmented = replace(
        model,
        ray_constraint_matrix=np.vstack(
            (model.ray_constraint_matrix, additional_row)
        ),
        ray_constraint_upper_bounds=np.concatenate(
            (model.ray_constraint_upper_bounds, additional_bound)
        ),
    )
    scaled_matrix = np.array(augmented.ray_constraint_matrix, copy=True)
    scaled_bounds = np.array(
        augmented.ray_constraint_upper_bounds, copy=True
    )
    scaled_matrix[-1] *= row_multiplier
    scaled_bounds[-1] *= row_multiplier
    rescaled = replace(
        augmented,
        ray_constraint_matrix=scaled_matrix,
        ray_constraint_upper_bounds=scaled_bounds,
    )
    baseline = _allocate(augmented, preferred=[2.0, 2.0, 2.0])
    transformed = _allocate(rescaled, preferred=[2.0, 2.0, 2.0])
    additional_index = augmented.contact_count
    np.testing.assert_allclose(
        transformed.contact_forces / augmented.normal_force_caps_n[:, None],
        baseline.contact_forces / augmented.normal_force_caps_n[:, None],
        rtol=0.0,
        atol=QP_OPTIONS.equality_tolerance,
    )
    assert transformed.inequality_augmented_row_inf_norms[
        additional_index
    ] == pytest.approx(
        row_multiplier
        * baseline.inequality_augmented_row_inf_norms[additional_index],
        rel=0.0,
        abs=0.0,
    )
    np.testing.assert_allclose(
        transformed.solver_coordinate_inequality_residuals[additional_index],
        baseline.solver_coordinate_inequality_residuals[additional_index],
        rtol=0.0,
        atol=QP_OPTIONS.inequality_tolerance,
    )
    np.testing.assert_allclose(
        transformed.physical_additional_model_inequality_residuals_by_row,
        row_multiplier
        * baseline.physical_additional_model_inequality_residuals_by_row,
        rtol=0.0,
        atol=(
            row_multiplier
            * baseline.inequality_augmented_row_inf_norms[additional_index]
            * QP_OPTIONS.inequality_tolerance
        ),
    )


def test_force_allocation_is_equivariant_under_one_rigid_transform() -> None:
    base_model = _symmetric_model(normal_force_cap_n=3.0)
    base = _allocate(base_model, preferred=[1.2, 1.2, 1.2])
    rotation = _rotation_z(0.73)
    translation = np.asarray((0.4, -0.2, 0.7))
    transformed_model = _symmetric_model(
        normal_force_cap_n=3.0,
        rotation=rotation,
        translation=translation,
    )
    transformed = _allocate(
        transformed_model, preferred=[1.2, 1.2, 1.2]
    )
    np.testing.assert_allclose(
        transformed.contact_forces,
        base.contact_forces @ rotation.T,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        transformed.normal_forces, base.normal_forces, atol=2.0e-7
    )


def test_object_impedance_reports_nonincreasing_mechanical_energy() -> None:
    stiffness = np.diag((120.0, 100.0, 80.0, 4.0, 3.0, 2.0))
    damping = np.diag((8.0, 7.0, 6.0, 0.8, 0.7, 0.6))
    controller = PassiveObjectImpedance(
        stiffness=stiffness,
        damping=damping,
        numerics=IMPEDANCE_NUMERICS,
    )
    error = np.asarray((0.01, -0.02, 0.005, 0.02, -0.01, 0.03))
    twist = np.asarray((0.2, -0.1, 0.05, -0.03, 0.02, 0.04))
    result = controller.evaluate(
        pose_error=error,
        object_twist=twist,
        truth_firewall_audit=_valid_observation().truth_firewall_audit,
    )
    assert result.audit.elastic_storage_j >= 0.0
    assert result.audit.damping_dissipation_rate_w >= 0.0
    assert result.audit.closed_loop_mechanical_energy_rate_w <= 0.0
    assert result.audit.passivity_identity_within_numerical_tolerance
    assert abs(result.audit.passivity_balance_residual_w) <= (
        IMPEDANCE_NUMERICS.passivity_balance_tolerance
    )
    assert result.audit.closed_loop_mechanical_energy_rate_w == pytest.approx(
        -result.audit.damping_dissipation_rate_w,
        abs=IMPEDANCE_NUMERICS.passivity_balance_tolerance,
    )


def test_object_impedance_is_rigid_rotation_equivariant() -> None:
    rotation = _rotation_z(-0.41)
    spatial_rotation = np.zeros((6, 6))
    spatial_rotation[:3, :3] = rotation
    spatial_rotation[3:, 3:] = rotation
    stiffness = np.diag((90.0, 90.0, 70.0, 5.0, 5.0, 3.0))
    damping = np.diag((7.0, 7.0, 5.0, 0.7, 0.7, 0.5))
    base_controller = PassiveObjectImpedance(
        stiffness=stiffness,
        damping=damping,
        numerics=IMPEDANCE_NUMERICS,
    )
    transformed_controller = PassiveObjectImpedance(
        stiffness=spatial_rotation @ stiffness @ spatial_rotation.T,
        damping=spatial_rotation @ damping @ spatial_rotation.T,
        numerics=IMPEDANCE_NUMERICS,
    )
    error = np.asarray((0.02, -0.01, 0.03, 0.1, -0.04, 0.06))
    twist = np.asarray((-0.2, 0.05, 0.08, 0.02, 0.03, -0.01))
    base = base_controller.evaluate(
        pose_error=error,
        object_twist=twist,
        truth_firewall_audit=None,
    )
    transformed = transformed_controller.evaluate(
        pose_error=spatial_rotation @ error,
        object_twist=spatial_rotation @ twist,
        truth_firewall_audit=None,
    )
    np.testing.assert_allclose(
        transformed.commanded_object_wrench,
        spatial_rotation @ base.commanded_object_wrench,
        atol=1.0e-12,
    )
    assert transformed.audit.elastic_storage_j == pytest.approx(
        base.audit.elastic_storage_j, abs=1.0e-12
    )
    assert transformed.audit.damping_dissipation_rate_w == pytest.approx(
        base.audit.damping_dissipation_rate_w, abs=1.0e-12
    )


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "ground_truth_pose_world_from_object",
        "contact_normal",
        "collider_name",
        "physx_contact_report",
    ),
)
def test_online_truth_fields_fail_closed(forbidden_field: str) -> None:
    document = {
        "estimated_pose_world_from_object": np.eye(4),
        "estimated_twist_world": np.zeros(6),
        "estimator_source": "rgbd_estimator",
        forbidden_field: np.zeros(6),
    }
    with pytest.raises(ValueError, match="online observation rejected"):
        OnlineControlObservation.from_mapping(
            document, numerics=IMPEDANCE_NUMERICS
        )


def test_stiffness_and_damping_contracts_fail_closed() -> None:
    nonsymmetric = np.eye(6)
    nonsymmetric[0, 1] = 0.1
    with pytest.raises(ValueError, match="stiffness must be symmetric"):
        PassiveObjectImpedance(
            stiffness=nonsymmetric,
            damping=np.eye(6),
            numerics=IMPEDANCE_NUMERICS,
        )
    negative_damping = np.eye(6)
    negative_damping[0, 0] = -0.1
    with pytest.raises(ValueError, match="damping must be positive semidefinite"):
        PassiveObjectImpedance(
            stiffness=np.eye(6),
            damping=negative_damping,
            numerics=IMPEDANCE_NUMERICS,
        )
