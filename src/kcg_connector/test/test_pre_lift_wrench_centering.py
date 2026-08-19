from dataclasses import replace

import numpy as np
import pytest

from kcg_connector.grasp.pre_lift_wrench_centering import (
    load_pre_lift_wrench_centering_config,
    solve_bounded_xy_centering,
)


def _config():
    return replace(
        load_pre_lift_wrench_centering_config(None), enabled=True
    )


def _wrench_from_objective(value):
    return np.asarray(
        (value[0] * 8.0, value[1] * 8.0, 0.0,
         value[2] * 0.30, value[3] * 0.30, 0.0),
        dtype=np.float64,
    )


def test_central_difference_recovers_known_bounded_xy_correction():
    config = _config()
    expected = np.asarray((0.00012, -0.00008))
    jacobian = np.asarray(
        ((800.0, 100.0), (-50.0, 900.0),
         (500.0, -120.0), (80.0, 650.0))
    )
    center = -(jacobian @ expected)
    dx = config.probe_offset_m
    result = solve_bounded_xy_centering(
        _wrench_from_objective(center),
        _wrench_from_objective(center + jacobian[:, 0] * dx),
        _wrench_from_objective(center - jacobian[:, 0] * dx),
        _wrench_from_objective(center + jacobian[:, 1] * dx),
        _wrench_from_objective(center - jacobian[:, 1] * dx),
        replace(config, damping_ratio=1.0e-6),
    )
    assert result["correction_xy_m"] == pytest.approx(expected, abs=1.0e-9)
    assert result["correction_clipped"] is False
    assert result["predicted_objective_norm"] <= 1.0e-6


def test_correction_is_norm_clipped_without_direction_change():
    config = replace(_config(), maximum_correction_m=0.00010)
    jacobian = np.asarray(
        ((1000.0, 0.0), (0.0, 1000.0),
         (500.0, 0.0), (0.0, 500.0))
    )
    center = np.asarray((-0.4, 0.3, -0.2, 0.15))
    dx = config.probe_offset_m
    result = solve_bounded_xy_centering(
        _wrench_from_objective(center),
        _wrench_from_objective(center + jacobian[:, 0] * dx),
        _wrench_from_objective(center - jacobian[:, 0] * dx),
        _wrench_from_objective(center + jacobian[:, 1] * dx),
        _wrench_from_objective(center - jacobian[:, 1] * dx),
        config,
    )
    assert result["correction_clipped"] is True
    assert result["correction_norm_m"] == pytest.approx(0.00010)


def test_degenerate_or_nonfinite_probe_set_fails_closed():
    config = _config()
    zero = np.zeros(6)
    with pytest.raises(ValueError, match="lacks two independent"):
        solve_bounded_xy_centering(zero, zero, zero, zero, zero, config)
    bad = zero.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite 6-vector"):
        solve_bounded_xy_centering(zero, bad, zero, zero, zero, config)
