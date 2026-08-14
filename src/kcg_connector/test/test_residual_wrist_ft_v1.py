"""Tests for the strict wrist-F/T residual-v1 observation boundary."""

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from kcg_connector.residual_rl import (
    ConnectorResidualState,
    load_connector_residual_config,
    residual_observation,
)
from kcg_connector.residual_wrist_ft_v1 import (
    TASK_FRAME,
    WRENCH_NAMES,
    CompensatedTaskWrench,
    WristFtV1BlockedError,
    WristFtV1Config,
    default_wrist_ft_v1_config_path,
    load_wrist_ft_v1_config,
    residual_wrist_ft_observation,
)


PACKAGE_ROOT = Path(__file__).parents[1]
RESIDUAL_CONFIG = load_connector_residual_config(
    PACKAGE_ROOT / "config" / "connector_task.yaml"
)
FT_CONFIG = WristFtV1Config(
    lateral_force_n=10.0,
    axial_force_n=20.0,
    bending_torque_nm=2.0,
    tightening_torque_nm=4.0,
    stale_timeout_s=0.1,
)


def _state(**overrides):
    values = {
        "phase_progress": 0.5,
        "q7_position_rad": -0.05,
        "q7_tracking_error_rad": 0.01,
        "q7_velocity_rad_s": (
            RESIDUAL_CONFIG.tightening_direction
            * RESIDUAL_CONFIG.nominal_q7_speed_rad_s
        ),
        "nut_angle_rad": RESIDUAL_CONFIG.target_angle_rad / 2.0,
        "nut_angular_velocity_rad_s": (
            RESIDUAL_CONFIG.nominal_q7_speed_rad_s
        ),
        "axial_travel_m": RESIDUAL_CONFIG.expected_axial_travel_m / 2.0,
        "axial_velocity_m_s": RESIDUAL_CONFIG.maximum_axial_speed_m_s / 2.0,
        "grasp_translation_error_m": (0.001, -0.001, 0.0),
        "grasp_rotation_error_rad": (0.01, 0.0, -0.01),
        "finger_torques_nm": (0.10, 0.12, 0.11),
        "finger_torque_deltas_nm": (0.01, -0.01, 0.0),
        "clamp_positions_rad": RESIDUAL_CONFIG.clamp_nominal_positions_rad,
    }
    values.update(overrides)
    return ConnectorResidualState(**values)


def _sample(**overrides):
    values = {
        "values": (5.0, -10.0, 20.0, 1.0, -2.0, 4.0),
        "timestamp_s": 10.0,
        "frame_id": TASK_FRAME,
        "health": "OK",
        "baseline_ready": True,
        "compensation_ready": True,
    }
    values.update(overrides)
    return CompensatedTaskWrench(**values)


def _encode(sample=None, config=FT_CONFIG, now_s=10.05):
    return residual_wrist_ft_observation(
        _state(),
        RESIDUAL_CONFIG,
        sample or _sample(),
        config,
        now_s=now_s,
    )


def test_v1_shape_dtype_range_and_v0_prefix_is_byte_exact():
    state = _state()
    base = residual_observation(state, RESIDUAL_CONFIG)
    result = residual_wrist_ft_observation(
        state,
        RESIDUAL_CONFIG,
        _sample(),
        FT_CONFIG,
        now_s=10.05,
    )
    assert result.shape == (30,)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)
    assert result[:24].tobytes() == base.tobytes()


def test_six_axis_order_and_scale_mapping_are_exact():
    assert WRENCH_NAMES == (
        "wrist_force_x",
        "wrist_force_y",
        "wrist_force_z",
        "wrist_torque_x",
        "wrist_torque_y",
        "wrist_torque_z",
    )
    assert _encode()[24:] == pytest.approx(
        [0.5, -1.0, 1.0, 0.5, -1.0, 1.0]
    )


def test_current_default_contract_is_blocked_by_null_scales_and_age_limit():
    with pytest.raises(WristFtV1BlockedError) as caught:
        load_wrist_ft_v1_config()
    message = str(caught.value)
    assert default_wrist_ft_v1_config_path().is_file()
    assert "BLOCKED" in message
    assert "normalization_scales.lateral_force_n is null" in message
    assert "safety_limits.stale_timeout_s is null" in message


def test_loader_rejects_zero_scale(tmp_path):
    with default_wrist_ft_v1_config_path().open(
        "r", encoding="utf-8"
    ) as stream:
        document = yaml.safe_load(stream)
    document["enabled"] = True
    document["residual_v1"]["normalization_scales"] = {
        "lateral_force_n": 0.0,
        "axial_force_n": 20.0,
        "bending_torque_nm": 2.0,
        "tightening_torque_nm": 4.0,
    }
    document["safety_limits"]["stale_timeout_s"] = 0.1
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(WristFtV1BlockedError, match="lateral_force_n"):
        load_wrist_ft_v1_config(path)


@pytest.mark.parametrize(
    "sample, now_s, message",
    (
        (_sample(health="DEGRADED"), 10.0, "health"),
        (_sample(baseline_ready=False), 10.0, "baseline"),
        (_sample(compensation_ready=False), 10.0, "compensation"),
        (_sample(frame_id="handbase_link"), 10.0, "task_frame"),
        (_sample(timestamp_s=9.0), 10.0, "stale"),
        (_sample(timestamp_s=11.0), 10.0, "future"),
        (_sample(timestamp_s=math.nan), 10.0, "finite"),
        (_sample(values=(0.0, 0.0, math.inf, 0.0, 0.0, 0.0)), 10.0,
         "finite"),
    ),
)
def test_health_age_baseline_frame_and_finite_fail_closed(
    sample, now_s, message
):
    with pytest.raises(ValueError, match=message):
        _encode(sample=sample, now_s=now_s)


@pytest.mark.parametrize(
    "field",
    (
        "simulation_ground_truth_control_authority_count",
        "privileged_contact_wrench_control_authority_count",
    ),
)
@pytest.mark.parametrize("value", (1, -1, False, 0.0))
def test_truth_and_privileged_authority_counters_must_be_integer_zero(
    field, value
):
    with pytest.raises(ValueError, match="integer zero"):
        _encode(sample=_sample(**{field: value}))


@pytest.mark.parametrize(
    "config",
    (
        replace(FT_CONFIG, lateral_force_n=0.0),
        replace(FT_CONFIG, axial_force_n=-1.0),
        replace(FT_CONFIG, bending_torque_nm=math.nan),
        replace(FT_CONFIG, tightening_torque_nm=math.inf),
        replace(FT_CONFIG, stale_timeout_s=0.0),
    ),
)
def test_direct_config_null_equivalents_and_nonpositive_values_are_rejected(
    config,
):
    with pytest.raises(ValueError, match="scale|positive|stale"):
        _encode(config=config)
