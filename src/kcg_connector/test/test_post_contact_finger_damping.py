from pathlib import Path

import pytest

from kcg_connector.grasp.post_contact_finger_damping import (
    FINAL_DAMPING_NM_S_RAD,
    PostContactFingerDampingConfig,
    derive_post_contact_finger_damping_step,
    load_post_contact_finger_damping_config,
    select_post_contact_damping_hand_indices,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_multilayer_tabletop_physical_grasp_v2.yaml"
)
RUNNER = REPOSITORY / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"


def _config():
    import yaml

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return load_post_contact_finger_damping_config(
        document["post_contact_finger_damping"]
    )


def test_h7_value_is_the_single_run08_energy_dissipation_derivation():
    config = _config()
    expected = 1.0 + 0.04907255567783536 / 0.05643138289451599
    assert config.enabled is True
    assert config.source_run_id == "B-V2-GRASP-08"
    assert config.include_preshape_joint is True
    assert config.preshape_joint_name == "f1j1"
    assert config.preshape_extension_source_run_id == "B-V2-GRASP-11-IFIX02"
    assert config.final_damping_nm_s_rad == pytest.approx(expected)
    assert FINAL_DAMPING_NM_S_RAD == pytest.approx(expected)


def test_h7_transition_is_minimum_jerk_monotonic_and_bounded():
    config = _config()
    values = [
        derive_post_contact_finger_damping_step(step, config)
        ["applied_damping_nm_s_rad"]
        for step in range(config.transition_steps)
    ]
    assert values[0] > config.initial_damping_nm_s_rad
    assert values[-1] == pytest.approx(config.final_damping_nm_s_rad)
    assert all(left <= right for left, right in zip(values, values[1:]))


def test_h7_rejects_a_second_parameter_set():
    values = dict(_config().__dict__)
    values["final_damping_nm_s_rad"] = 2.0
    with pytest.raises(ValueError, match="frozen"):
        PostContactFingerDampingConfig(**values)


def test_h10_extends_the_same_h7_curve_to_only_the_preshape_joint():
    config = _config()
    assert select_post_contact_damping_hand_indices(
        ("f1j1", "f1j2", "f2j1", "f3j2"),
        (1, 2, 3),
        config,
    ) == (0, 1, 2, 3)


def test_historical_default_keeps_preshape_extension_disabled():
    config = load_post_contact_finger_damping_config(None)
    assert config.include_preshape_joint is False
    assert select_post_contact_damping_hand_indices(
        ("f1j1", "f1j2", "f2j1", "f3j2"),
        (1, 2, 3),
        config,
    ) == (1, 2, 3)


def test_runner_keeps_formal_channels_separate_from_damping_selection():
    source = RUNNER.read_text(encoding="utf-8")
    assert "derive_post_contact_finger_damping_step" in source
    assert "finger_damping_dof_indices" in source
    assert '"formal_finger_root_channels_unchanged": True' in source
    assert "post_contact_finger_damping" in source
    assert '"finger_drive_damping_nm_s_rad"' in source
