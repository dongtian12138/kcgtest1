import numpy as np
import pytest

from kcg_connector.joint_torque_wrench import estimate_tool_wrench


BASE_JACOBIAN = np.column_stack(
    (
        np.eye(6, dtype=np.float64),
        np.array([0.2, -0.3, 0.1, 0.25, -0.15, 0.2]),
    )
)
ZERO_INTERNAL_TORQUE = np.zeros(7, dtype=np.float64)
SOLVER_LIMITS = {
    "wrench_scales": np.ones(6, dtype=np.float64),
    "damping": 1.0e-10,
    "maximum_condition_number": 100.0,
    "maximum_projection_residual_nm": 1.0e-8,
}


def estimate(jacobian, measured, modeled=ZERO_INTERNAL_TORQUE, **kwargs):
    limits = dict(SOLVER_LIMITS)
    limits.update(kwargs)
    return estimate_tool_wrench(
        jacobian,
        measured,
        modeled,
        **limits,
    )


def test_exact_full_rank_wrench_is_recovered_after_internal_compensation():
    expected = np.array([12.0, -3.0, 25.0, 0.4, -0.7, 1.2])
    modeled = np.array([2.0, -1.0, 3.0, 0.2, 0.1, -0.4, 0.8])
    measured = modeled + BASE_JACOBIAN.T @ expected

    result = estimate(BASE_JACOBIAN, measured, modeled)

    assert result.valid is True
    assert result.reason == "ok"
    assert result.rank == 6
    assert result.condition_number < 2.0
    assert result.wrench is not None
    assert result.wrench.shape == (6,)
    assert np.allclose(result.wrench, expected, atol=1.0e-10)
    assert result.projection_residual is not None
    assert result.projection_residual.shape == (7,)
    assert result.projection_residual_norm_nm < 1.0e-10


def test_positive_weights_reduce_influence_of_a_noisy_joint_channel():
    expected = np.array([4.0, -2.0, 6.0, 0.2, -0.3, 0.5])
    measured = BASE_JACOBIAN.T @ expected
    noisy = measured.copy()
    noisy[-1] += 0.2

    unweighted = estimate(
        BASE_JACOBIAN,
        noisy,
        maximum_projection_residual_nm=1.0,
    )
    weighted = estimate(
        BASE_JACOBIAN,
        noisy,
        weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.001],
        maximum_projection_residual_nm=1.0,
    )

    assert unweighted.valid is True
    assert weighted.valid is True
    assert np.linalg.norm(weighted.wrench - expected) < np.linalg.norm(
        unweighted.wrench - expected
    )
    assert weighted.projection_residual.shape == (7,)


def test_unexplainable_seventh_dimension_fails_projection_gate():
    expected = np.array([2.0, 3.0, -5.0, 0.1, 0.2, -0.4])
    _, _, right_vectors_t = np.linalg.svd(
        BASE_JACOBIAN,
        full_matrices=True,
    )
    joint_null_vector = right_vectors_t[-1]
    measured = (
        BASE_JACOBIAN.T @ expected + 0.2 * joint_null_vector
    )

    result = estimate(
        BASE_JACOBIAN,
        measured,
        maximum_projection_residual_nm=0.05,
    )

    assert result.valid is False
    assert result.reason == "projection_residual_exceeded"
    assert result.wrench is None
    assert result.rank == 6
    assert result.projection_residual.shape == (7,)
    assert result.projection_residual_norm_nm == pytest.approx(0.2)
    assert np.allclose(
        result.external_torque
        - result.projected_torque,
        result.projection_residual,
    )


def test_rank_deficient_jacobian_fails_closed():
    singular = BASE_JACOBIAN.copy()
    singular[-1, :] = 0.0
    measured = singular.T @ np.ones(6)

    result = estimate(singular, measured)

    assert result.valid is False
    assert result.reason == "rank_below_six"
    assert result.wrench is None
    assert result.rank == 5
    assert np.isposinf(result.condition_number)
    assert result.projection_residual.shape == (7,)


def test_full_rank_but_near_singular_jacobian_fails_condition_gate():
    near_singular = BASE_JACOBIAN.copy()
    near_singular[-1, :] *= 1.0e-8
    measured = near_singular.T @ np.ones(6)

    result = estimate(
        near_singular,
        measured,
        maximum_condition_number=1.0e6,
        maximum_projection_residual_nm=1.0,
    )

    assert result.valid is False
    assert result.reason == "condition_number_exceeded"
    assert result.wrench is None
    assert result.rank == 6
    assert result.condition_number > 1.0e6
    assert result.projection_residual.shape == (7,)


def test_task_wrench_scales_define_the_condition_number_metric():
    imbalanced = BASE_JACOBIAN.copy()
    imbalanced[-1, :] *= 1.0e-3
    expected = np.array([2.0, -3.0, 4.0, 0.1, -0.2, 0.3])
    measured = imbalanced.T @ expected

    unscaled_metric = estimate(
        imbalanced,
        measured,
        maximum_condition_number=100.0,
        maximum_projection_residual_nm=1.0,
    )
    task_metric = estimate(
        imbalanced,
        measured,
        wrench_scales=[1.0, 1.0, 1.0, 1.0, 1.0, 1000.0],
        maximum_condition_number=100.0,
        maximum_projection_residual_nm=1.0,
    )

    assert unscaled_metric.valid is False
    assert unscaled_metric.reason == "condition_number_exceeded"
    assert unscaled_metric.condition_number > 100.0
    assert task_metric.valid is True
    assert task_metric.condition_number < 2.0
    assert np.allclose(task_metric.wrench, expected, atol=1.0e-9)


@pytest.mark.parametrize(
    ("field", "replacement", "expected_reason"),
    (
        ("jacobian", np.zeros((7, 6)), "invalid_jacobian_shape"),
        ("jacobian", np.full((6, 7), np.nan), "nonfinite_jacobian"),
        ("measured", np.zeros(6), "invalid_measured_torque_shape"),
        (
            "measured",
            np.full(7, np.inf),
            "nonfinite_measured_torque",
        ),
        ("modeled", np.zeros(8), "invalid_modeled_internal_torque_shape"),
        (
            "modeled",
            np.full(7, np.nan),
            "nonfinite_modeled_internal_torque",
        ),
        ("weights", np.ones(6), "invalid_weights_shape"),
        ("weights", [1.0] * 6 + [np.nan], "nonfinite_weights"),
        ("weights", [1.0] * 6 + [0.0], "nonpositive_weights"),
        (
            "wrench_scales",
            np.ones(7),
            "invalid_wrench_scales_shape",
        ),
        (
            "wrench_scales",
            [1.0] * 5 + [np.inf],
            "nonfinite_wrench_scales",
        ),
        (
            "wrench_scales",
            [1.0] * 5 + [-1.0],
            "nonpositive_wrench_scales",
        ),
        ("damping", 0.0, "invalid_damping"),
        (
            "maximum_condition_number",
            0.5,
            "invalid_maximum_condition_number",
        ),
        (
            "maximum_projection_residual_nm",
            -0.1,
            "invalid_maximum_projection_residual_nm",
        ),
        ("rank_tolerance", -1.0, "invalid_rank_tolerance"),
    ),
)
def test_invalid_inputs_fail_closed(field, replacement, expected_reason):
    arguments = {
        "jacobian": BASE_JACOBIAN,
        "measured_torque": np.zeros(7),
        "modeled_internal_torque": np.zeros(7),
        **SOLVER_LIMITS,
    }
    if field == "measured":
        field = "measured_torque"
    elif field == "modeled":
        field = "modeled_internal_torque"
    arguments[field] = replacement

    result = estimate_tool_wrench(**arguments)

    assert result.valid is False
    assert result.reason == expected_reason
    assert result.wrench is None


@pytest.mark.parametrize(
    "missing_name",
    (
        "wrench_scales",
        "damping",
        "maximum_condition_number",
        "maximum_projection_residual_nm",
    ),
)
def test_uncalibrated_safety_limit_keeps_estimator_disabled(missing_name):
    arguments = dict(SOLVER_LIMITS)
    arguments[missing_name] = None

    result = estimate_tool_wrench(
        BASE_JACOBIAN,
        np.zeros(7),
        np.zeros(7),
        **arguments,
    )

    assert result.valid is False
    assert result.reason == "uncalibrated_safety_limits"
    assert result.wrench is None


def test_missing_dynamics_compensation_can_look_like_a_valid_contact():
    """The algebra cannot distinguish omitted dynamics from contact."""
    internal_wrench_equivalent = np.array(
        [18.0, -7.0, 31.0, 0.5, -0.2, 1.7]
    )
    internal_torque = BASE_JACOBIAN.T @ internal_wrench_equivalent

    uncompensated = estimate(
        BASE_JACOBIAN,
        internal_torque,
        ZERO_INTERNAL_TORQUE,
    )
    compensated = estimate(
        BASE_JACOBIAN,
        internal_torque,
        internal_torque,
    )

    assert uncompensated.valid is True
    assert np.allclose(
        uncompensated.wrench,
        internal_wrench_equivalent,
        atol=1.0e-10,
    )
    assert compensated.valid is True
    assert np.allclose(compensated.wrench, np.zeros(6), atol=1.0e-12)


def test_invalid_result_arrays_are_read_only_to_preserve_diagnostics():
    expected = np.ones(6)
    measured = BASE_JACOBIAN.T @ expected
    result = estimate(BASE_JACOBIAN, measured)

    with pytest.raises(ValueError):
        result.wrench[0] = 99.0
    with pytest.raises(ValueError):
        result.projection_residual[0] = 99.0
