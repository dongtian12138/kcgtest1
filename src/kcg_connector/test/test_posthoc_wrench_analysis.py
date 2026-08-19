'''Pure tests for the E0 posthoc wrench diagnostics module and CLI.'''

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.posthoc_wrench_analysis import (
    analyze_episode,
    channel_window_statistics,
    force_norm,
    moment_magnitude_increase,
    moment_norm,
    per_step_series,
    perpendicular_moment_delta,
    reference_window_statistics,
    signed_delta,
)

REPOSITORY = Path(__file__).resolve().parents[3]
ISAAC_DIR = REPOSITORY / "src" / "kcg_connector" / "isaac"

REFERENCE = [0.9945919121, -0.6336179830, 20.1982088884,
             0.3077882620, 0.4752902940, 0.0016043479]
CURRENT = [0.4174154103, -0.2389231920, 21.0850448608,
           0.1346363127, 0.2252950370, 0.0018836792]


def test_norms_and_signed_delta():
    assert force_norm(REFERENCE) == pytest.approx(20.2326, abs=1.0e-3)
    assert moment_norm(REFERENCE) == pytest.approx(0.56625, abs=1.0e-4)
    assert moment_norm(CURRENT) == pytest.approx(0.26247, abs=1.0e-4)
    delta = signed_delta(CURRENT, REFERENCE)
    assert delta.shape == (6,)
    assert delta[2] == pytest.approx(CURRENT[2] - REFERENCE[2])


def test_magnitude_increase_candidate_is_zero_on_normal_unload():
    # The rep05 evidence: absolute moment norm dropped from 0.566 to 0.262.
    # The magnitude-increase candidate must not flag that as dangerous.
    assert moment_magnitude_increase(CURRENT, REFERENCE) == 0.0


def test_magnitude_increase_candidate_flags_true_increase():
    increased = list(CURRENT)
    increased[3] = 0.8
    assert moment_magnitude_increase(increased, REFERENCE) == pytest.approx(
        moment_norm(increased) - moment_norm(REFERENCE)
    )


def test_perpendicular_delta_decomposes_radial_and_perpendicular():
    # Nearly radial delta: tiny perpendicular component, angle close to pi.
    radial = perpendicular_moment_delta(CURRENT, REFERENCE)
    delta = np.asarray(CURRENT)[3:] - np.asarray(REFERENCE)[3:]
    assert radial["radial_component_nm"] == pytest.approx(
        -np.linalg.norm(delta), abs=1.0e-3
    )
    assert radial["perpendicular_component_norm_nm"] == pytest.approx(
        0.00952, abs=1.0e-3
    )
    assert radial["delta_to_reference_angle_rad"] == pytest.approx(
        math.pi, abs=5.0e-2
    )
    # The rep05 delta is nearly anti-parallel: the perpendicular candidate is
    # tiny (about 0.0095 N*m) even though the vector-delta norm trips 0.30.
    assert radial["perpendicular_component_norm_nm"] < 0.02


def test_perpendicular_delta_pure_perpendicular_case():
    reference = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0]
    current = [1.0, 2.0, 3.0, 1.0, 0.5, 0.0]
    result = perpendicular_moment_delta(current, reference)
    assert result["perpendicular_component_norm_nm"] == pytest.approx(0.5)
    assert result["radial_component_nm"] == pytest.approx(0.0, abs=1.0e-9)
    assert result["delta_to_reference_angle_rad"] == pytest.approx(
        math.pi / 2.0
    )


def test_perpendicular_delta_degenerate_zero_reference():
    result = perpendicular_moment_delta(
        [0.0, 0.0, 0.0, 0.3, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert result["perpendicular_component_norm_nm"] == pytest.approx(0.5)
    assert result["delta_to_reference_angle_rad"] is None


def test_reference_window_statistics_records_drift_and_norms():
    samples = [REFERENCE] * 120
    stats = reference_window_statistics(
        samples, force_drift_bound_n=0.5, moment_drift_bound_nm=0.15
    )
    assert stats["window_steps"] == 120
    assert stats["evidence_only"] is True
    assert stats["per_channel"]["mean"] == pytest.approx(REFERENCE)
    assert np.allclose(stats["per_channel"]["std"], 0.0)
    assert np.allclose(stats["first_to_second_half_drift"], 0.0)
    assert stats["absolute_moment_norm_nm"]["mean"] == pytest.approx(
        moment_norm(REFERENCE)
    )
    assert stats["bounded_drift_indicators"]["force"]["within_bound"] is True
    assert stats["bounded_drift_indicators"]["moment"]["within_bound"] is True
    assert stats["bounded_drift_indicators"]["gates_control"] is False


def test_reference_window_statistics_marks_unbounded_drift():
    drifting = []
    for index in range(120):
        sample = list(REFERENCE)
        sample[0] += 0.01 * index
        drifting.append(sample)
    stats = reference_window_statistics(
        drifting, force_drift_bound_n=0.05
    )
    assert stats["bounded_drift_indicators"]["force"]["within_bound"] is False


def test_reference_window_statistics_validates():
    with pytest.raises(ValueError, match="at least two"):
        reference_window_statistics([REFERENCE])
    with pytest.raises(ValueError, match="finite"):
        reference_window_statistics([REFERENCE, [float("nan")] * 6])
    with pytest.raises(ValueError, match="N x 6"):
        reference_window_statistics([REFERENCE[:3], REFERENCE[:3]])


def _synthetic_report():
    return {
        "passed": False,
        "formal_payload_wrist_reference": REFERENCE,
        "formal_payload_wrist_reference_sample_count": 120,
        "formal_payload_wrist_reference_statistics": {
            "window_steps": 120,
            "evidence_only": True,
        },
        "formal_lift_failure": {
            "reason": "wrist_moment_limit",
            "global_step": 9722,
            "wrist_wrench_canonical": CURRENT,
        },
    }


def _synthetic_steps():
    return [
        {
            "global_step": 1,
            "phase": "physical_grip_lift_stage_1",
            "lift_stage": 1,
            "lift_stage_step": 0,
            "wrist_wrench_canonical": REFERENCE,
        },
        {
            "global_step": 2,
            "phase": "physical_grip_lift_stage_1",
            "lift_stage": 1,
            "lift_stage_step": 1,
            "wrist_wrench_canonical": CURRENT,
        },
        {
            "global_step": 3,
            "phase": "formal_lift_recovery_return",
            "recovery": True,
            "wrist_wrench_canonical": CURRENT,
        },
    ]


def test_analyze_episode_computes_frozen_diagnostics():
    result = analyze_episode(
        _synthetic_report(), _synthetic_steps(), plug_nut_mass_kg=0.12
    )
    assert result["posthoc_diagnostics_only"] is True
    assert result["changes_pass"] is False
    assert result["original_pass_unchanged"] is False
    assert result["reference_absolute_moment_norm_nm"] == pytest.approx(
        0.56625, abs=1.0e-4
    )
    assert result["current_absolute_moment_norm_nm"] == pytest.approx(
        0.26247, abs=1.0e-4
    )
    assert result["moment_magnitude_increase_candidate_nm"] == 0.0
    assert result["vector_delta_moment_norm_nm"] == pytest.approx(
        0.3041, abs=1.0e-4
    )
    assert result["delta_fz_n"] == pytest.approx(
        CURRENT[2] - REFERENCE[2]
    )
    assert result["plug_nut_weight_n"] == pytest.approx(0.12 * 9.81)
    assert result["delta_fz_to_plug_nut_weight_ratio"] == pytest.approx(
        (CURRENT[2] - REFERENCE[2]) / (0.12 * 9.81)
    )
    assert result["reference_statistics_available"] is True
    # Only the two staged lift steps enter the series; recovery is excluded.
    assert len(result["per_step_series"]) == 2
    assert result["per_step_series"][1]["absolute_moment_norm_nm"] == (
        pytest.approx(0.26247, abs=1.0e-4)
    )


def test_per_step_series_filters_non_lift_phases():
    assert len(per_step_series(_synthetic_steps())) == 2


def test_analyze_episode_validates_mass_and_report():
    with pytest.raises(ValueError, match="mass"):
        analyze_episode(_synthetic_report(), [], plug_nut_mass_kg=0.0)
    with pytest.raises(ValueError, match="formal_payload_wrist_reference"):
        analyze_episode({}, [], plug_nut_mass_kg=0.12)
    with pytest.raises(ValueError, match="formal_lift_failure"):
        analyze_episode(
            {"formal_payload_wrist_reference": REFERENCE},
            [],
            plug_nut_mass_kg=0.12,
        )


def test_checked_in_config_chain_yields_012_kg_and_9_81_gravity():
    import sys

    sys.path.insert(0, str(ISAAC_DIR))
    try:
        from d38999_tabletop_posthoc_analyze import (
            load_gravity_m_s2,
            load_plug_nut_mass_kg,
        )
    finally:
        sys.path.remove(str(ISAAC_DIR))
    config = (
        REPOSITORY
        / "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
    )
    assert load_plug_nut_mass_kg(REPOSITORY, config) == pytest.approx(0.12)
    assert load_gravity_m_s2(REPOSITORY, config) == pytest.approx(9.81)


def test_e0_cli_refuses_to_overwrite_original_report_or_steps(tmp_path):
    # The frozen requirement: E0 writes a NEW json and can never overwrite
    # nominal_physics_report.json or controller_steps.jsonl.
    (tmp_path / "nominal_physics_report.json").write_text(
        json.dumps(_synthetic_report()), encoding="utf-8"
    )
    steps_path = tmp_path / "controller_steps.jsonl"
    steps_path.write_text(
        "\n".join(
            json.dumps(record) for record in _synthetic_steps()
        )
        + "\n",
        encoding="utf-8",
    )
    import os
    import subprocess
    import sys

    environment = dict(os.environ)
    source_root = str(REPOSITORY / "src" / "kcg_connector")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    report_before = (tmp_path / "nominal_physics_report.json").read_bytes()
    steps_before = steps_path.read_bytes()
    for target in ("nominal_physics_report.json", "controller_steps.jsonl"):
        result = subprocess.run(
            [
                sys.executable,
                str(ISAAC_DIR / "d38999_tabletop_posthoc_analyze.py"),
                "--episode-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / target),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode != 0
        assert "refusing to overwrite" in result.stderr
    assert (tmp_path / "nominal_physics_report.json").read_bytes() == (
        report_before
    )
    assert steps_path.read_bytes() == steps_before


def test_analyze_episode_never_mutates_inputs(tmp_path):
    report = _synthetic_report()
    steps = _synthetic_steps()
    before_report = json.dumps(report, sort_keys=True)
    before_steps = json.dumps(steps, sort_keys=True)
    analyze_episode(report, steps, plug_nut_mass_kg=0.12)
    assert json.dumps(report, sort_keys=True) == before_report
    assert json.dumps(steps, sort_keys=True) == before_steps


def test_channel_window_statistics_accepts_240x3_root_proxy_window():
    rng = np.random.default_rng(7)
    samples = 0.12 + 0.06 * rng.standard_normal((240, 3))
    result = channel_window_statistics(samples)
    assert result["window_steps"] == 240
    assert result["evidence_only"] is True
    assert result["threshold_label"] == "SIM_TUNING_ONLY_A_CANDIDATE"
    assert set(result["per_channel"]) == {"f1", "f2", "f3"}
    for name, stats in result["per_channel"].items():
        for key in (
            "mean",
            "std",
            "min",
            "max",
            "first_half_mean",
            "second_half_mean",
            "first_to_second_half_drift",
            "first_to_second_half_slope_per_sample",
        ):
            assert key in stats, key
            assert math.isfinite(float(stats[key])), key
    # No wrist-specific force/moment norm outputs may leak in.
    assert "absolute_force_norm_n" not in result
    assert "absolute_moment_norm_nm" not in result


def test_channel_window_statistics_matches_reference_math():
    samples = [
        [0.10, 0.20, 0.30],
        [0.11, 0.21, 0.31],
        [0.12, 0.22, 0.32],
        [0.13, 0.23, 0.33],
    ]
    result = channel_window_statistics(samples)
    data = np.asarray(samples, dtype=np.float64)
    for index, name in enumerate(("f1", "f2", "f3")):
        stats = result["per_channel"][name]
        assert stats["mean"] == pytest.approx(float(np.mean(data[:, index])))
        assert stats["min"] == pytest.approx(float(np.min(data[:, index])))
        assert stats["max"] == pytest.approx(float(np.max(data[:, index])))
        assert stats["first_half_mean"] == pytest.approx(
            float(np.mean(data[:2, index]))
        )
        assert stats["second_half_mean"] == pytest.approx(
            float(np.mean(data[2:, index]))
        )


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros((10, 6)),
        np.zeros(10),
        np.zeros((10, 4)),
        np.zeros((1, 3)),
    ],
)
def test_channel_window_statistics_rejects_wrong_shapes(samples):
    with pytest.raises(ValueError):
        channel_window_statistics(samples)


def test_channel_window_statistics_rejects_nonfinite():
    samples = np.zeros((10, 3))
    samples[4, 1] = np.nan
    with pytest.raises(ValueError):
        channel_window_statistics(samples)


def test_channel_window_statistics_rejects_wrong_channel_names():
    with pytest.raises(ValueError):
        channel_window_statistics(
            np.zeros((10, 3)), channel_names=("f1", "f2")
        )


def test_reference_window_statistics_still_rejects_nx3():
    # The wrist contract stays N x 6; root windows must use the new helper.
    with pytest.raises(ValueError):
        reference_window_statistics(np.zeros((10, 3)))
