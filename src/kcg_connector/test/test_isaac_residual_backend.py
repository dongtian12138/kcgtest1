"""Pure fail-closed tests for Isaac residual reset diagnostics."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.isaac_residual_backend import (
    EpisodeSafetyStats,
    ConnectorResidualIsaacBackend,
    _RAW_SAFETY_JOINT_LIMIT_TOLERANCE_RAD,
    _RESET_DIAGNOSTIC_COMPATIBILITY_ALIAS_KEYS,
    _RESET_DIAGNOSTIC_INTENDED_DISTANCE_KEYS,
    _RESET_DIAGNOSTIC_LIMITS,
    _RESET_DIAGNOSTIC_REQUIRED_KEYS,
    _maximum_finite_tail,
    raw_episode_safety_report,
    summarize_reset_diagnostics,
)


EXPECTED_LIMITS = {
    "body_position_error_m": 0.00005,
    "nut_position_error_m": 0.00005,
    "body_orientation_error_degrees": 0.05,
    "nut_orientation_error_degrees": 0.05,
    "q7_checkpoint_error_degrees": 0.05,
    "preconstraint_recovery_body_displacement_m": 0.0001,
    "preconstraint_recovery_nut_displacement_m": 0.0001,
    "preconstraint_recovery_body_rotation_degrees": 0.1,
    "preconstraint_recovery_nut_rotation_degrees": 0.1,
    "first_step_body_jump_m": 0.0001,
    "first_step_nut_jump_m": 0.0001,
    "first_step_body_orientation_jump_degrees": 0.1,
    "first_step_nut_orientation_jump_degrees": 0.1,
    "first_step_q7_jump_degrees": 0.05,
    "settled_body_position_error_m": 0.0001,
    "settled_nut_position_error_m": 0.0001,
    "settled_body_orientation_error_degrees": 0.1,
    "settled_nut_orientation_error_degrees": 0.1,
    "settled_q7_error_degrees": 0.1,
    "settled_body_linear_speed_m_s": 0.001,
    "settled_nut_linear_speed_m_s": 0.001,
    "settled_body_angular_speed_rad_s": 0.02,
    "settled_nut_angular_speed_rad_s": 0.02,
    "settled_q7_speed_degrees_s": 0.5,
    "first_ten_peak_body_linear_speed_m_s": 0.025,
    "first_ten_peak_nut_linear_speed_m_s": 0.025,
    "first_ten_peak_body_angular_speed_rad_s": 0.25,
    "first_ten_peak_nut_angular_speed_rad_s": 0.25,
    "first_ten_peak_q7_speed_degrees_s": 5.0,
    "post_solver_tail_peak_body_linear_speed_m_s": 0.010,
    "post_solver_tail_peak_nut_linear_speed_m_s": 0.060,
    "post_solver_tail_peak_body_angular_speed_rad_s": 0.005,
    "post_solver_tail_peak_nut_angular_speed_rad_s": 0.25,
    "post_solver_tail_peak_q7_speed_degrees_s": 0.5,
}


def _diagnostic_at_limits():
    diagnostic = dict(EXPECTED_LIMITS)
    diagnostic.update(
        {
            "body_intended_reset_distance_m": 1.0e6,
            "nut_intended_reset_distance_m": 2.0e6,
            "solver_body_linear_speed_m_s": 10.0,
            "solver_body_angular_speed_rad_s": 20.0,
            "solver_nut_linear_speed_m_s": 30.0,
            "solver_nut_angular_speed_rad_s": 40.0,
            "solver_q7_speed_degrees_s": 50.0,
        }
    )
    return diagnostic


def test_complete_schema_preserves_all_observable_and_post_solver_limits():
    assert _RESET_DIAGNOSTIC_LIMITS == EXPECTED_LIMITS
    assert _RESET_DIAGNOSTIC_REQUIRED_KEYS == frozenset(
        EXPECTED_LIMITS
    ).union(
        _RESET_DIAGNOSTIC_INTENDED_DISTANCE_KEYS,
        _RESET_DIAGNOSTIC_COMPATIBILITY_ALIAS_KEYS,
    )


def test_exact_boundaries_pass_and_non_gating_fields_are_not_thresholded():
    diagnostic = _diagnostic_at_limits()
    maxima, passed = summarize_reset_diagnostics([diagnostic])
    assert passed is True
    assert not _RESET_DIAGNOSTIC_INTENDED_DISTANCE_KEYS.intersection(
        maxima
    )
    for name in _RESET_DIAGNOSTIC_COMPATIBILITY_ALIAS_KEYS:
        assert maxima[name] == diagnostic[name]


@pytest.mark.parametrize("name", tuple(EXPECTED_LIMITS))
def test_one_ulp_above_each_gate_fails(name):
    diagnostic = _diagnostic_at_limits()
    diagnostic[name] = math.nextafter(
        EXPECTED_LIMITS[name], math.inf
    )
    _, passed = summarize_reset_diagnostics([diagnostic])
    assert passed is False


@pytest.mark.parametrize(
    "stage_name", ("stage20", "stage60", "stage120")
)
def test_curriculum_stages_share_one_reset_gate(stage_name):
    maxima, passed = summarize_reset_diagnostics(
        [_diagnostic_at_limits()]
    )
    assert passed is True, stage_name
    assert maxima[
        "post_solver_tail_peak_nut_linear_speed_m_s"
    ] == pytest.approx(0.060)


@pytest.mark.parametrize(
    "missing_name", sorted(_RESET_DIAGNOSTIC_REQUIRED_KEYS)
)
def test_every_required_field_is_required(missing_name):
    diagnostic = _diagnostic_at_limits()
    del diagnostic[missing_name]
    maxima, passed = summarize_reset_diagnostics([diagnostic])
    assert maxima == {}
    assert passed is False


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
@pytest.mark.parametrize("name", tuple(EXPECTED_LIMITS))
def test_every_gating_field_rejects_nonfinite_values(name, value):
    diagnostic = _diagnostic_at_limits()
    diagnostic[name] = value
    maxima, passed = summarize_reset_diagnostics([diagnostic])
    assert maxima == {}
    assert passed is False


@pytest.mark.parametrize(
    "name",
    tuple(_RESET_DIAGNOSTIC_INTENDED_DISTANCE_KEYS)
    + tuple(_RESET_DIAGNOSTIC_COMPATIBILITY_ALIAS_KEYS),
)
def test_non_gating_required_fields_must_still_be_finite(name):
    diagnostic = _diagnostic_at_limits()
    diagnostic[name] = math.nan
    maxima, passed = summarize_reset_diagnostics([diagnostic])
    assert maxima == {}
    assert passed is False


def test_schema_is_exact_and_rejects_extra_fields():
    diagnostic = _diagnostic_at_limits()
    diagnostic["unexpected"] = 0.0
    maxima, passed = summarize_reset_diagnostics([diagnostic])
    assert maxima == {}
    assert passed is False


def test_schema_mismatch_between_resets_fails_closed():
    complete = _diagnostic_at_limits()
    incomplete = _diagnostic_at_limits()
    del incomplete["solver_q7_speed_degrees_s"]
    maxima, passed = summarize_reset_diagnostics(
        [complete, incomplete]
    )
    assert maxima == {}
    assert passed is False


@pytest.mark.parametrize("invalid", (None, "0.0", True, np.bool_(False)))
def test_non_numeric_and_boolean_values_fail_closed(invalid):
    diagnostic = _diagnostic_at_limits()
    diagnostic["body_position_error_m"] = invalid
    maxima, passed = summarize_reset_diagnostics([diagnostic])
    assert maxima == {}
    assert passed is False


def test_numpy_real_scalars_are_accepted_and_normalized_to_float():
    diagnostic = _diagnostic_at_limits()
    diagnostic["body_position_error_m"] = np.float32(0.00001)
    maxima, passed = summarize_reset_diagnostics([diagnostic])
    assert passed is True
    assert isinstance(maxima["body_position_error_m"], float)


def test_empty_and_non_mapping_diagnostics_fail_closed():
    assert summarize_reset_diagnostics([]) == ({}, False)
    assert summarize_reset_diagnostics([None]) == ({}, False)


def test_multiple_resets_are_gated_by_their_maximum():
    lower = _diagnostic_at_limits()
    lower.update({name: 0.0 for name in EXPECTED_LIMITS})
    boundary = _diagnostic_at_limits()
    maxima, passed = summarize_reset_diagnostics([lower, boundary])
    assert passed is True
    assert maxima[
        "post_solver_tail_peak_body_linear_speed_m_s"
    ] == pytest.approx(0.010)


def test_tail_peak_uses_exactly_the_final_ten_samples():
    values = [99.0, 98.0] + [float(value) for value in range(10)]
    assert _maximum_finite_tail(values, 10) == pytest.approx(9.0)
    assert _maximum_finite_tail([1.0, 3.0, 2.0], 10) == pytest.approx(
        3.0
    )


@pytest.mark.parametrize(
    "values,count",
    (
        ([], 10),
        ([1.0], 0),
        ([1.0, math.nan], 10),
        ([1.0, math.inf], 10),
        ([1.0, -math.inf], 10),
    ),
)
def test_unusable_tail_windows_fail_closed_as_infinity(values, count):
    assert _maximum_finite_tail(values, count) == math.inf


RAW_CONFIG_VALUES = {
    "policy_rate_hz": 10.0,
    "maximum_q7_speed_rad_s": 0.4,
    "maximum_absolute_finger_torque_nm": 1.0,
    "maximum_q7_tracking_error_rad": 0.03,
    "maximum_grasp_translation_error_m": 0.005,
    "maximum_grasp_rotation_error_rad": 0.08,
}

RAW_PEAK_ATTRIBUTES = {
    "physics_substep_max_abs_joint_velocity_rad_s": (
        "max_abs_velocity"
    ),
    "physics_substep_max_abs_q7_velocity_rad_s": (
        "max_abs_q7_velocity"
    ),
    "physics_substep_max_joint_limit_violation_rad": (
        "max_limit_violation"
    ),
    "physics_substep_max_abs_finger_base_torque_nm": (
        "max_finger_torque_delta"
    ),
    "policy_boundary_max_abs_nut_angular_velocity_rad_s": (
        "max_abs_nut_angular_velocity_policy_boundary"
    ),
    "policy_boundary_max_abs_q7_tracking_error_rad": (
        "max_abs_q7_tracking_error_policy_boundary"
    ),
    "policy_boundary_max_grasp_translation_error_m": (
        "max_grasp_translation_error_policy_boundary"
    ),
    "policy_boundary_max_grasp_rotation_error_rad": (
        "max_grasp_rotation_error_policy_boundary"
    ),
}

RAW_LIMITS = {
    "physics_substep_max_abs_q7_velocity_rad_s": (
        RAW_CONFIG_VALUES["maximum_q7_speed_rad_s"] * 1.10
    ),
    "physics_substep_max_joint_limit_violation_rad": 0.02,
    "physics_substep_max_abs_finger_base_torque_nm": 1.0,
    "policy_boundary_max_abs_nut_angular_velocity_rad_s": (
        RAW_CONFIG_VALUES["maximum_q7_speed_rad_s"] * 1.25
    ),
    "policy_boundary_max_abs_q7_tracking_error_rad": 0.03,
    "policy_boundary_max_grasp_translation_error_m": 0.005,
    "policy_boundary_max_grasp_rotation_error_rad": 0.08,
}

RAW_FAILURE_REASONS = {
    "physics_substep_max_abs_q7_velocity_rad_s": (
        "physics_substep_q7_speed_limit_exceeded"
    ),
    "physics_substep_max_joint_limit_violation_rad": (
        "physics_substep_joint_limit_tolerance_exceeded"
    ),
    "physics_substep_max_abs_finger_base_torque_nm": (
        "physics_substep_finger_base_torque_limit_exceeded"
    ),
    "policy_boundary_max_abs_nut_angular_velocity_rad_s": (
        "policy_boundary_nut_speed_limit_exceeded"
    ),
    "policy_boundary_max_abs_q7_tracking_error_rad": (
        "policy_boundary_q7_tracking_limit_exceeded"
    ),
    "policy_boundary_max_grasp_translation_error_m": (
        "policy_boundary_grasp_translation_limit_exceeded"
    ),
    "policy_boundary_max_grasp_rotation_error_rad": (
        "policy_boundary_grasp_rotation_limit_exceeded"
    ),
}


def _raw_config(**overrides):
    values = dict(RAW_CONFIG_VALUES)
    values.update(overrides)
    return SimpleNamespace(**values)


def _raw_safety_at_limits():
    return EpisodeSafetyStats(
        # There is deliberately no invented all-joint speed gate.
        max_abs_velocity=1.0e6,
        max_abs_q7_velocity=RAW_LIMITS[
            "physics_substep_max_abs_q7_velocity_rad_s"
        ],
        max_limit_violation=RAW_LIMITS[
            "physics_substep_max_joint_limit_violation_rad"
        ],
        max_finger_torque_delta=RAW_LIMITS[
            "physics_substep_max_abs_finger_base_torque_nm"
        ],
        max_abs_nut_angular_velocity_policy_boundary=RAW_LIMITS[
            "policy_boundary_max_abs_nut_angular_velocity_rad_s"
        ],
        max_abs_q7_tracking_error_policy_boundary=RAW_LIMITS[
            "policy_boundary_max_abs_q7_tracking_error_rad"
        ],
        max_grasp_translation_error_policy_boundary=RAW_LIMITS[
            "policy_boundary_max_grasp_translation_error_m"
        ],
        max_grasp_rotation_error_policy_boundary=RAW_LIMITS[
            "policy_boundary_max_grasp_rotation_error_rad"
        ],
        finite_throughout=True,
        physics_substep_samples=24,
        policy_boundary_samples=1,
    )


def _raw_report(safety=None, config=None, physics_rate_hz=240.0):
    return raw_episode_safety_report(
        _raw_safety_at_limits() if safety is None else safety,
        _raw_config() if config is None else config,
        physics_rate_hz,
    )


def test_raw_safety_report_has_stable_json_safe_schema_and_sampling():
    report = _raw_report()
    assert report == {
        "passed": True,
        "failure_reasons": [],
        "finite_throughout": True,
        "limits": RAW_LIMITS,
        "metrics": {
            name: float(getattr(_raw_safety_at_limits(), attribute))
            for name, attribute in RAW_PEAK_ATTRIBUTES.items()
        },
        "sampling": {
            "physics_substep": {
                "includes_episode_initial_snapshot": True,
                "rate_hz": 240.0,
                "samples": 24,
            },
            "policy_boundary": {
                "includes_episode_initial_snapshot": True,
                "rate_hz": 10.0,
                "samples": 1,
            },
        },
        "signal_source": "raw_physics",
    }
    json.dumps(report, allow_nan=False)
    assert (
        _RAW_SAFETY_JOINT_LIMIT_TOLERANCE_RAD
        == RAW_LIMITS[
            "physics_substep_max_joint_limit_violation_rad"
        ]
    )


@pytest.mark.parametrize("metric_name", tuple(RAW_LIMITS))
def test_raw_safety_exact_gate_boundaries_pass(metric_name):
    report = _raw_report()
    assert report["metrics"][metric_name] == RAW_LIMITS[metric_name]
    assert report["passed"] is True


@pytest.mark.parametrize("metric_name", tuple(RAW_LIMITS))
def test_raw_safety_one_ulp_above_each_gate_fails(metric_name):
    safety = _raw_safety_at_limits()
    attribute = RAW_PEAK_ATTRIBUTES[metric_name]
    setattr(
        safety,
        attribute,
        math.nextafter(RAW_LIMITS[metric_name], math.inf),
    )
    report = _raw_report(safety=safety)
    assert report["passed"] is False
    assert report["failure_reasons"] == [
        RAW_FAILURE_REASONS[metric_name]
    ]


def test_all_joint_speed_is_raw_evidence_without_an_invented_limit():
    report = _raw_report()
    name = "physics_substep_max_abs_joint_velocity_rad_s"
    assert report["metrics"][name] == 1.0e6
    assert name not in report["limits"]
    assert report["passed"] is True


def test_raw_safety_ignores_termination_labels_and_noisy_observations():
    safety = _raw_safety_at_limits()
    safety.termination_reason = "success"
    safety.last_observed_state = SimpleNamespace(
        finger_torques_nm=(math.nan, math.nan, math.nan)
    )
    assert _raw_report(safety=safety) == _raw_report()


@pytest.mark.parametrize(
    "missing_name",
    tuple(RAW_PEAK_ATTRIBUTES.values())
    + (
        "finite_throughout",
        "physics_substep_samples",
        "policy_boundary_samples",
    ),
)
def test_raw_safety_missing_stat_fields_fail_closed(missing_name):
    values = vars(_raw_safety_at_limits()).copy()
    del values[missing_name]
    report = _raw_report(safety=SimpleNamespace(**values))
    assert report["passed"] is False
    assert "raw_safety_evidence_invalid" in report["failure_reasons"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
@pytest.mark.parametrize(
    "metric_name,attribute", tuple(RAW_PEAK_ATTRIBUTES.items())
)
def test_raw_safety_nonfinite_peak_fields_fail_closed_and_json_safe(
    metric_name, attribute, value
):
    safety = _raw_safety_at_limits()
    setattr(safety, attribute, value)
    report = _raw_report(safety=safety)
    assert report["passed"] is False
    assert report["metrics"][metric_name] is None
    assert "raw_safety_evidence_invalid" in report["failure_reasons"]
    json.dumps(report, allow_nan=False)


def test_raw_safety_explicit_nonfinite_latch_fails_independently():
    safety = _raw_safety_at_limits()
    safety.finite_throughout = False
    report = _raw_report(safety=safety)
    assert report["passed"] is False
    assert report["failure_reasons"] == ["raw_physics_nonfinite"]


@pytest.mark.parametrize("count_name", (
    "physics_substep_samples",
    "policy_boundary_samples",
))
@pytest.mark.parametrize("invalid", (0, -1, 1.0, True, np.bool_(True)))
def test_raw_safety_invalid_or_empty_sampling_fails_closed(
    count_name, invalid
):
    safety = _raw_safety_at_limits()
    setattr(safety, count_name, invalid)
    report = _raw_report(safety=safety)
    assert report["passed"] is False
    assert report["sampling"][
        "physics_substep"
        if count_name == "physics_substep_samples"
        else "policy_boundary"
    ]["samples"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("config_name", tuple(RAW_CONFIG_VALUES))
def test_raw_safety_missing_configured_limits_or_rates_fail_closed(
    config_name,
):
    values = dict(RAW_CONFIG_VALUES)
    del values[config_name]
    report = _raw_report(config=SimpleNamespace(**values))
    assert report["passed"] is False
    assert "raw_safety_evidence_invalid" in report["failure_reasons"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf, 0.0))
@pytest.mark.parametrize("config_name", tuple(RAW_CONFIG_VALUES))
def test_raw_safety_nonfinite_or_nonpositive_config_fails_closed(
    config_name, invalid
):
    report = _raw_report(config=_raw_config(**{config_name: invalid}))
    assert report["passed"] is False
    assert "raw_safety_evidence_invalid" in report["failure_reasons"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf, 0.0))
def test_raw_safety_invalid_physics_rate_fails_closed(invalid):
    report = _raw_report(physics_rate_hz=invalid)
    assert report["passed"] is False
    assert report["sampling"]["physics_substep"]["rate_hz"] is None
    json.dumps(report, allow_nan=False)


def test_backend_property_and_info_fields_share_the_same_raw_report():
    backend = object.__new__(ConnectorResidualIsaacBackend)
    backend.episode_safety = _raw_safety_at_limits()
    backend.active_residual_config = _raw_config()
    backend.scene = SimpleNamespace(physics_rate_hz=240.0)
    report = backend.raw_safety_report
    assert report == _raw_report()
    assert backend._raw_safety_info_fields() == {
        "raw_safety_passed": report["passed"],
        "raw_safety_failure_reasons": report["failure_reasons"],
        "raw_safety_peaks": report["metrics"],
    }


def test_episode_safety_accumulators_are_independent_instances():
    first = EpisodeSafetyStats()
    second = EpisodeSafetyStats()
    first.max_abs_q7_velocity = 123.0
    first.physics_substep_samples = 99
    assert second.max_abs_q7_velocity == 0.0
    assert second.physics_substep_samples == 0


class _FakeRobot:
    num_dof = 3

    def __init__(self):
        self.positions = np.asarray(
            (0.0, 1.03, 0.0), dtype=np.float64
        )
        self.velocities = np.asarray(
            (2.0, -0.44, 1.0), dtype=np.float64
        )
        self.efforts = np.asarray(
            (0.20, -0.30, 1.20), dtype=np.float64
        )

    def get_joint_positions(self):
        return self.positions

    def get_joint_velocities(self):
        return self.velocities

    def get_measured_joint_efforts(self, joint_indices):
        assert tuple(joint_indices) == (0, 1, 2)
        return self.efforts


class _FakeRigidBody:
    def get_world_pose(self):
        return (
            np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        )


def test_observe_accumulates_true_240hz_raw_physics_peaks():
    backend = object.__new__(ConnectorResidualIsaacBackend)
    backend.episode_safety = EpisodeSafetyStats()
    backend.scene = SimpleNamespace(
        robot=_FakeRobot(),
        sensor_indices=np.asarray((0, 1, 2), dtype=np.int64),
        tare_efforts=np.asarray((0.10, 0.10, 0.10), dtype=np.float64),
        body=_FakeRigidBody(),
        nut=_FakeRigidBody(),
        q7_index=1,
        dof_properties=(
            {"hasLimits": True, "lower": -1.0, "upper": 1.0},
            {"hasLimits": True, "lower": -1.0, "upper": 1.0},
            {"hasLimits": False, "lower": 0.0, "upper": 0.0},
        ),
    )
    backend._observe()
    safety = backend.episode_safety
    assert safety.physics_substep_samples == 1
    assert safety.max_abs_velocity == pytest.approx(2.0)
    assert safety.max_abs_q7_velocity == pytest.approx(0.44)
    assert safety.max_limit_violation == pytest.approx(0.03)
    assert safety.max_finger_torque_delta == pytest.approx(1.10)


def test_one_substep_violation_stays_latched_after_raw_values_recover():
    robot = _FakeRobot()
    robot.positions[:] = 0.0
    robot.efforts[:] = 0.10
    robot.velocities[:] = (0.0, 0.45, 0.0)
    backend = object.__new__(ConnectorResidualIsaacBackend)
    backend.episode_safety = EpisodeSafetyStats(
        policy_boundary_samples=1
    )
    backend.active_residual_config = _raw_config()
    backend.scene = SimpleNamespace(
        robot=robot,
        sensor_indices=np.asarray((0, 1, 2), dtype=np.int64),
        tare_efforts=np.asarray((0.10, 0.10, 0.10), dtype=np.float64),
        body=_FakeRigidBody(),
        nut=_FakeRigidBody(),
        q7_index=1,
        dof_properties=(
            {"hasLimits": True, "lower": -1.0, "upper": 1.0},
            {"hasLimits": True, "lower": -1.0, "upper": 1.0},
            {"hasLimits": True, "lower": -1.0, "upper": 1.0},
        ),
        physics_rate_hz=240.0,
    )
    backend._observe()
    robot.velocities[:] = 0.0
    backend._observe()
    assert backend.episode_safety.max_abs_q7_velocity == pytest.approx(
        0.45
    )
    assert backend.raw_safety_report["passed"] is False
    assert backend.raw_safety_report["failure_reasons"] == [
        "physics_substep_q7_speed_limit_exceeded"
    ]


def test_policy_boundary_accumulator_uses_raw_state_at_10hz():
    backend = object.__new__(ConnectorResidualIsaacBackend)
    backend.episode_safety = EpisodeSafetyStats()
    raw_state = SimpleNamespace(
        nut_angular_velocity_rad_s=-0.4,
        q7_tracking_error_rad=-0.02,
        grasp_translation_error_m=(0.003, 0.004, 0.0),
        grasp_rotation_error_rad=(0.0, 0.06, 0.08),
    )
    backend._record_policy_boundary_safety(raw_state)
    safety = backend.episode_safety
    assert safety.policy_boundary_samples == 1
    assert (
        safety.max_abs_nut_angular_velocity_policy_boundary
        == pytest.approx(0.4)
    )
    assert (
        safety.max_abs_q7_tracking_error_policy_boundary
        == pytest.approx(0.02)
    )
    assert safety.max_grasp_translation_error_policy_boundary == (
        pytest.approx(0.005)
    )
    assert safety.max_grasp_rotation_error_policy_boundary == (
        pytest.approx(0.10)
    )
