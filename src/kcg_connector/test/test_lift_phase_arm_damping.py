from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.lift_phase_arm_damping import (
    FINAL_DAMPING_NM_S_RAD,
    LiftPhaseArmDampingConfig,
    SOURCE_MEDIAN_VELOCITY_RATIO,
    derive_lift_phase_arm_damping_step,
    load_lift_phase_arm_damping_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_multilayer_tabletop_physical_grasp_v2.yaml"
)
RUNNER = REPOSITORY / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"


def _config():
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return load_lift_phase_arm_damping_config(
        document["lift_phase_arm_damping"]
    )


def test_h13_value_is_the_run13_h12_joint_velocity_ratio_median():
    config = _config()
    assert config.enabled is True
    assert config.source_lift_run_id == "B-V2-GRASP-13"
    assert config.source_hold_run_id == "B-V2-H12-ZERO-LIFT-01"
    assert config.source_median_velocity_ratio == pytest.approx(
        SOURCE_MEDIAN_VELOCITY_RATIO
    )
    assert config.final_damping_nm_s_rad == pytest.approx(
        400.0 * SOURCE_MEDIAN_VELOCITY_RATIO
    )
    assert FINAL_DAMPING_NM_S_RAD == pytest.approx(875.29978045654)


def test_h13_transition_is_minimum_jerk_monotonic_and_bounded():
    config = _config()
    values = [
        derive_lift_phase_arm_damping_step(step, config)[
            "applied_damping_nm_s_rad"
        ]
        for step in range(config.transition_steps)
    ]
    assert values[0] > config.initial_damping_nm_s_rad
    assert values[-1] == pytest.approx(config.final_damping_nm_s_rad)
    assert all(left <= right for left, right in zip(values, values[1:]))


def test_h13_rejects_a_second_parameter_set():
    values = dict(_config().__dict__)
    values["final_damping_nm_s_rad"] += 1.0
    with pytest.raises(ValueError, match="frozen"):
        LiftPhaseArmDampingConfig(**values)


def test_historical_missing_section_is_disabled():
    assert load_lift_phase_arm_damping_config(None).enabled is False


def test_runner_applies_h13_only_after_zero_lift_branch():
    source = RUNNER.read_text(encoding="utf-8")
    zero_lift = source.index("if zero_lift_hold_mode:")
    h13 = source.index("# B-V2-H13 is applied only here")
    staged = source.index("# Staged mode below", zero_lift)
    assert zero_lift < staged < h13
    assert "derive_lift_phase_arm_damping_step" in source
    assert 'phase = "pre_lift_arm_damping_transition"' in source
    assert '"arm_target_modified": False' in source
    assert '"sensor_origin_hard_gate_unchanged": True' in source
