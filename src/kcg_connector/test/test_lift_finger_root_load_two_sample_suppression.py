from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.lift_finger_root_load_two_sample_suppression import (
    LiftFingerRootLoadTwoSampleSuppression,
    load_lift_finger_root_load_two_sample_suppression_config,
    two_sample_root_load_control_input,
)
from kcg_connector.grasp.physical_grasp_config import (
    load_physical_grasp_experiment_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_multilayer_tabletop_physical_grasp_v2.yaml"
)
RUNNER = REPOSITORY / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"


def _section():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[
        "lift_finger_root_load_two_sample_suppression"
    ]


def _enabled_section():
    section = _section()
    section["enabled"] = True
    return section


def test_h24_strict_historical_section_is_disabled_and_has_no_tunable_numeric_parameter():
    section = _section()
    config = load_lift_finger_root_load_two_sample_suppression_config(section)
    physical = load_physical_grasp_experiment_config(CONFIG)
    assert config.enabled is False
    assert physical.lift_finger_root_load_two_sample_suppression == config
    assert config.hard_gate_detection_delay_steps == 0
    assert not any(
        key in section
        for key in ("gain", "cutoff_hz", "force_n", "geometry", "coefficient")
    )


def test_h24_two_sample_mean_has_exact_nyquist_zero_and_preserves_raw_values():
    result = two_sample_root_load_control_input(
        (0.2, -0.3, 0.4), (-0.2, 0.3, -0.4)
    )
    assert result["filtered_h17_control_root_torque_nm"] == pytest.approx(
        (0.0, 0.0, 0.0)
    )
    assert result["raw_current_root_torque_nm"] == pytest.approx(
        (0.2, -0.3, 0.4)
    )
    assert result["raw_previous_root_torque_nm"] == pytest.approx(
        (-0.2, 0.3, -0.4)
    )
    assert result["nyquist_alternating_component_gain"] == 0.0
    assert result["raw_sensor_hard_gate_unchanged"] is True
    assert result["hard_gate_detection_delay_steps"] == 0


def test_h24_first_sample_duplication_has_no_zero_start_transient():
    controller = LiftFingerRootLoadTwoSampleSuppression(
        load_lift_finger_root_load_two_sample_suppression_config(
            _enabled_section()
        )
    )
    result = controller.update((0.2, 0.3, 0.4), 100)
    assert result["filtered_h17_control_root_torque_nm"] == pytest.approx(
        (0.2, 0.3, 0.4)
    )
    assert result["initialized_by_current_sample_duplication"] is True
    assert result["newest_control_sample_age_steps"] == 1
    assert result["oldest_control_sample_age_steps"] == 1


def test_h24_requires_consecutive_samples_and_resets_only_for_reference_refresh():
    controller = LiftFingerRootLoadTwoSampleSuppression(
        load_lift_finger_root_load_two_sample_suppression_config(
            _enabled_section()
        )
    )
    controller.update((0.2, 0.3, 0.4), 100)
    second = controller.update((0.4, 0.5, 0.6), 101)
    assert second["filtered_h17_control_root_torque_nm"] == pytest.approx(
        (0.3, 0.4, 0.5)
    )
    assert second["previous_input_global_step"] == 100
    with pytest.raises(ValueError, match="consecutive"):
        controller.update((0.5, 0.6, 0.7), 103)
    controller.reset_after_h17_reference_refresh()
    reset = controller.update((0.7, 0.8, 0.9), 500)
    assert reset["initialized_by_current_sample_duplication"] is True
    assert controller.summary()["reference_refresh_reset_count"] == 1


@pytest.mark.parametrize(
    "key",
    (
        "two_sample_arithmetic_mean_required",
        "h17_control_input_only",
        "initialize_by_current_sample_duplication",
        "reset_after_h17_reference_refresh",
        "raw_finger_root_samples_recorded",
        "raw_sensor_hard_gate_unchanged",
        "no_new_gain_force_or_geometry_parameter",
    ),
)
def test_h24_truth_and_parameter_boundaries_fail_closed(key):
    section = _section()
    section[key] = False
    with pytest.raises(ValueError, match="must remain true"):
        load_lift_finger_root_load_two_sample_suppression_config(section)


def test_h24_rejects_delay_unknown_fields_and_nonfinite_inputs():
    section = _section()
    section["hard_gate_detection_delay_steps"] = 1
    with pytest.raises(ValueError, match="must remain zero"):
        load_lift_finger_root_load_two_sample_suppression_config(section)
    section = _section()
    section["coefficient"] = 0.5
    with pytest.raises(ValueError, match="unknown keys"):
        load_lift_finger_root_load_two_sample_suppression_config(section)
    with pytest.raises(ValueError, match="finite"):
        two_sample_root_load_control_input((float("nan"), 0.2, 0.3), None)


def test_h24_runner_has_exact_three_control_sites_and_one_reference_reset():
    source = RUNNER.read_text(encoding="utf-8")
    assert source.count("h24_h17_control_input(") == 4  # definition + 3 calls
    assert source.count("finalize_h24_record(") == 4  # definition + 3 calls
    assert source.count(
        "lift_finger_root_load_suppression_controller.reset_after_h17_reference_refresh()"
    ) == 1
    assert "H24_TWO_SAMPLE_ROOT_LOAD" in source
    assert '"raw_hard_gate_sample_filtered": False' in source
    assert '"raw_hard_gate_detection_delay_steps": 0' in source
    assert "formal_lift_monitor.update(\n                            root_delta," in source


def test_h24_runner_binds_derivation_hash_and_keeps_truth_firewall():
    source = RUNNER.read_text(encoding="utf-8")
    assert "B_H24_TWO_SAMPLE_FINGER_ROOT_LOAD_SUPPRESSION_DERIVATION.json" in source
    assert "h24_source_derivation_actual_sha256" in source
    assert "h24_source_derivation_expected_sha256" in source
    assert '"object_truth_used": False' in source
    assert '"contact_truth_used": False' in source
    assert '"event_truth_used": False' in source
    assert '"post_physics_object_pose_writes": 0' in source
