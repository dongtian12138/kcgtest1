from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.lift_x_force_admittance import (
    LiftXForceAdmittanceConfig,
    MAXIMUM_STEP_CORRECTION_M,
    MAXIMUM_TOTAL_CORRECTION_M,
    SOURCE_TARGET_STIFFNESS_N_M,
    TASK_X_COMPLIANCE_M_N,
    derive_lift_x_force_admittance_step,
    load_lift_x_force_admittance_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_multilayer_tabletop_physical_grasp_v2.yaml"
)
RUNNER = REPOSITORY / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"


def _config():
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return load_lift_x_force_admittance_config(
        document["lift_x_force_admittance"]
    )


def _historical_active_config():
    values = dict(_config().__dict__)
    values["enabled"] = True
    return LiftXForceAdmittanceConfig(**values)


def test_h8_is_the_single_run05_derived_parameter_set():
    config = _config()
    assert config.enabled is False
    assert config.source_run_id == "B-V2-GRASP-05"
    assert config.source_target_stiffness_n_m == pytest.approx(
        SOURCE_TARGET_STIFFNESS_N_M
    )
    assert config.task_x_compliance_m_n == pytest.approx(
        1.0 / SOURCE_TARGET_STIFFNESS_N_M
    )
    assert config.task_x_compliance_m_n == pytest.approx(
        TASK_X_COMPLIANCE_M_N
    )
    assert config.maximum_total_correction_m == pytest.approx(
        MAXIMUM_TOTAL_CORRECTION_M
    )
    assert config.maximum_step_correction_m == pytest.approx(
        MAXIMUM_STEP_CORRECTION_M
    )


def test_h8_opposes_force_and_applies_both_bounds():
    config = _historical_active_config()
    first = derive_lift_x_force_admittance_step(-1.0851328280521557, 0.0, config)
    assert first["desired_unbounded_correction_m"] > 0.0
    assert first["desired_bounded_correction_m"] == pytest.approx(15.0e-6)
    assert first["applied_correction_m"] == pytest.approx(1.0e-6)
    assert first["total_bound_active"] is True
    assert first["rate_bound_active"] is True

    correction = 0.0
    for _ in range(20):
        step = derive_lift_x_force_admittance_step(-1.0851328280521557, correction, config)
        correction = float(step["applied_correction_m"])
    assert correction == pytest.approx(15.0e-6)


def test_h8_returns_toward_zero_without_a_position_jump():
    config = _historical_active_config()
    step = derive_lift_x_force_admittance_step(0.0, 15.0e-6, config)
    assert step["applied_delta_m"] == pytest.approx(-1.0e-6)
    assert step["applied_correction_m"] == pytest.approx(14.0e-6)


def test_h8_rejects_parameter_drift_and_over_gate_input():
    values = dict(_config().__dict__)
    values["maximum_total_correction_m"] = 20.0e-6
    with pytest.raises(ValueError, match="frozen"):
        LiftXForceAdmittanceConfig(**values)
    with pytest.raises(ValueError, match="force gate"):
        derive_lift_x_force_admittance_step(
            8.01, 0.0, _historical_active_config()
        )


def test_runner_keeps_h8_truth_free_and_sensor_guarded():
    source = RUNNER.read_text(encoding="utf-8")
    assert "derive_lift_x_force_admittance_step" in source
    assert '"lift_x_force_admittance"' in source
    assert '"sensor_origin_hard_gate_unchanged": True' in source
    assert '"object_truth_used": False' in source
    assert '"contact_truth_used": False' in source
    assert '"force_input_frame": "robot_fk_grasp_tcp_frame"' in source
    assert '"command_target_frame": "world"' in source
