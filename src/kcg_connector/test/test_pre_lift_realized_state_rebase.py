import numpy as np
import pytest

from kcg_connector.grasp.pre_lift_realized_state_rebase import (
    PreLiftRealizedStateRebaseConfig,
    validate_realized_state_rebase,
)


def _config():
    return PreLiftRealizedStateRebaseConfig(
        enabled=True,
        threshold_label="SIM_TUNING_ONLY_B_V2_H5",
        reference_window_steps=240,
        maximum_rebase_joint_delta_rad=0.005,
        maximum_rebase_tcp_translation_m=0.002,
        maximum_rebase_tcp_rotation_rad=0.01,
        maximum_entry_moment_score_nm=0.24,
        maximum_entry_load_imbalance=0.18,
    )


def test_robot_only_rebase_reports_joint_and_fk_deltas():
    commanded = np.zeros(7)
    realized = np.array([0.001, 0.0, -0.0015, 0.0, 0.0, 0.0, 0.0])
    target_tcp = np.eye(4)
    realized_tcp = np.eye(4)
    realized_tcp[:3, 3] = (-0.000539, 0.000269, -0.001202)
    result = validate_realized_state_rebase(
        commanded, realized, target_tcp, realized_tcp, _config()
    )
    assert result["maximum_joint_delta_rad"] == pytest.approx(0.0015)
    assert result["tcp_translation_delta_m"] == pytest.approx(
        [-0.000539, 0.000269, -0.001202]
    )
    assert result["robot_joint_state_only"] is True
    assert result["object_truth_used"] is False
    assert result["contact_truth_used"] is False


def test_rebase_fails_closed_outside_internal_joint_bound():
    with pytest.raises(ValueError, match="joint delta"):
        validate_realized_state_rebase(
            np.zeros(7),
            np.array([0.006, 0, 0, 0, 0, 0, 0]),
            np.eye(4),
            np.eye(4),
            _config(),
        )


def test_rebase_config_cannot_reach_formal_hard_gates():
    with pytest.raises(ValueError, match="0.03"):
        PreLiftRealizedStateRebaseConfig(
            **{**_config().__dict__, "maximum_rebase_joint_delta_rad": 0.03}
        )
    with pytest.raises(ValueError, match="0.30"):
        PreLiftRealizedStateRebaseConfig(
            **{**_config().__dict__, "maximum_entry_moment_score_nm": 0.30}
        )
