'''Pure tests for the GUI/headless single-finger consistency evaluator.'''

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from kcg_connector.grasp.posthoc_wrench_analysis import (
    window_statistics_block,
)
from kcg_connector.grasp.single_finger_gui_consistency import (
    compare_gui_headless_consistency,
)

REPOSITORY = Path(__file__).resolve().parents[3]
COMPARATOR = (
    REPOSITORY
    / "src/kcg_connector/isaac/single_finger_gui_consistency_compare.py"
)
SOURCE_ROOT = str(REPOSITORY / "src" / "kcg_connector")

WRIST_CHANNELS = ("fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm")
FINGER_CHANNELS = ("f1j2", "f2j1", "f3j2")
WRIST_MEANS = [0.1, 0.2, 20.5, 0.01, 0.02, 0.001]


def _digest(index=0):
    return f"{index:064x}"[-64:]


def _provenance(**overrides):
    values = {
        "audit_mode": "capture",
        "finger": "f1",
        "seed": 0,
        "payload_sha256": _digest(10),
        "physical_grasp_config_sha256": _digest(1),
        "pick_config_sha256": _digest(2),
        "tabletop_scene_config_sha256": _digest(3),
        "runner_sha256": _digest(4),
        "wrapper_sha256": _digest(5),
        "finger_contact_detector_sha256": _digest(6),
        "single_finger_contact_test_sha256": _digest(7),
        "single_finger_posthoc_audit_sha256": _digest(8),
        "single_finger_posthoc_audit_compare_sha256": _digest(9),
    }
    values.update(overrides)
    return values


def _channel_stats(mean, std, base=0.0):
    return {
        "mean": mean,
        "std": std,
        "rms": (mean * mean + std * std) ** 0.5 if std >= 0 else std,
        "min": mean - 2.0 * std,
        "max": mean + 2.0 * std,
        "first_half_mean": mean - base,
        "second_half_mean": mean + base,
        "first_to_second_half_drift": 2.0 * base,
    }


def _stats_block(
    channels,
    *,
    count=120,
    means=None,
    std=0.001,
    nan_channel=None,
    subtraction=None,
):
    means = means if means is not None else [0.0] * len(channels)
    per_channel = {}
    for index, channel in enumerate(channels):
        channel_std = float("nan") if channel == nan_channel else std
        per_channel[channel] = _channel_stats(means[index], channel_std)
    block = {
        "sample_count": count,
        "std_ddof": 1,
        "std_ddof_note": (
            "sample standard deviation (ddof=1) over the window"
        ),
        "threshold_label": "SIM_TUNING_ONLY",
        "evidence_only": True,
        "baseline_subtraction": subtraction,
        "frame": "evidence window",
        "source": "test fixture",
        "per_channel": per_channel,
    }
    return block


def _finger_block(count=120, nan_channel=None, **overrides):
    values = _stats_block(
        FINGER_CHANNELS, count=count, std=0.0001, nan_channel=nan_channel
    )
    values.update(overrides)
    return values


def _wrist_raw_block(means=None, count=120, **overrides):
    values = _stats_block(
        WRIST_CHANNELS,
        means=WRIST_MEANS if means is None else means,
        count=count,
    )
    values["absolute_force_norm_n"] = {
        "mean": 20.6, "std": 0.001, "minimum": 20.59, "maximum": 20.61,
    }
    values["absolute_moment_norm_nm"] = {
        "mean": 0.03, "std": 0.001, "minimum": 0.028, "maximum": 0.032,
    }
    values.update(overrides)
    return values


def _wrist_residual_block(subtraction="window_mean", **overrides):
    values = _stats_block(
        WRIST_CHANNELS, means=[0.0] * 6, std=0.001,
        subtraction=subtraction,
    )
    values["absolute_force_norm_n"] = {
        "mean": 0.002, "std": 0.001, "minimum": 0.0, "maximum": 0.004,
    }
    values["absolute_moment_norm_nm"] = {
        "mean": 0.0001, "std": 0.0001, "minimum": 0.0, "maximum": 0.0002,
    }
    values.update(overrides)
    return values


def _runtime_block(gui):
    return {
        "evidence_only": True,
        "threshold_label": "SIM_TUNING_ONLY",
        "report_gui": gui,
        "physics_rate_hz": 240,
        "world_physics_dt_s": 1.0 / 240.0,
        "world_physics_dt_readback_s": 1.0 / 240.0,
        "world_rendering_dt_s": 1.0 / 60.0,
        "world_rendering_dt_readback_s": 1.0 / 60.0,
        "world_step_render_enabled": gui,
        "open_tare_duration_s": 0.5,
        "tare_actual_steps": 120,
        "wrist_tare_actual_samples": 120,
        "controller_updates_per_physics_step": 1,
        "controller_update_source": (
            "one controller update per observe_and_step"
        ),
        "solver_substep_contract": {
            "status": "not_explicitly_authored_or_unavailable",
            "note": "no PhysxSceneAPI solver settings authored",
        },
        "dt_readback_status": ["physics_dt_ok", "rendering_dt_ok"],
    }


def _snapshot(*, body=0, nut=0, other=0, unexpected=0, plug_table=1, material=None):
    snap = {
        "finger_body_group_records": {
            "f1": {"body": body, "nut": nut},
            "f2": {"body": other, "nut": 0},
            "f3": {"body": 0, "nut": 0},
        },
        "plug_table_records": plug_table,
        "unexpected_robot_link_records": unexpected,
    }
    if material is not None:
        snap["material_evidence"] = material
    return snap


def _material(total, unresolved=0):
    return {
        "available": True,
        "resolved_records": total,
        "unresolved_records": unresolved,
        "grip_grip_records": total,
        "resolved_non_grip_records": 0,
    }


TRANSITION_EVENTS = [
    {"from": "APPROACH", "to": "CONTACT_CANDIDATE", "step": 2},
    {"from": "CONTACT_CANDIDATE", "to": "CONTACT_CONFIRMED", "step": 13},
    {"from": "CONTACT_CONFIRMED", "to": "SOFT_HOLD", "step": 13},
    {"from": "SOFT_HOLD", "to": "RELEASE_COMMANDED", "step": 38},
    {"from": "RELEASE_COMMANDED", "to": "RELEASE_CONFIRMED", "step": 118},
]


def _steps(confirm_step=13, release_steps=80, **overrides):
    records = []
    for step in range(1, confirm_step + 1):
        target = 0.00075 * step
        if step <= confirm_step - 12:
            state = "APPROACH"
            candidate_steps = 0
            observation_state = "APPROACH"
        elif step < confirm_step:
            state = "APPROACH"
            candidate_steps = step - (confirm_step - 12)
            observation_state = "CONTACT_CANDIDATE"
        else:
            state = "SOFT_HOLD"
            candidate_steps = 12
            observation_state = "CONTACT_CONFIRMED"
        records.append(
            _record(
                step, state, target,
                soft_hold=0, release=0,
                observation_state=observation_state,
                candidate_steps=candidate_steps,
                stiffness=0.35 if state == "SOFT_HOLD" else 1.0,
            )
        )
    for hold in range(1, 25):
        records.append(
            _record(
                confirm_step + hold, "SOFT_HOLD", 0.00075 * confirm_step,
                soft_hold=hold, release=0,
                observation_state="SOFT_HOLD",
                candidate_steps=12,
                stiffness=0.35,
            )
        )
    for release in range(1, release_steps):
        target = 0.00075 * confirm_step - 0.00075 * release
        release_confirm = max(0, release - (release_steps - 18))
        records.append(
            _record(
                confirm_step + 24 + release, "RELEASE_COMMANDED", target,
                soft_hold=24, release=release,
                observation_state="RELEASE_COMMANDED",
                candidate_steps=12,
                stiffness=0.35,
                release_confirm=release_confirm,
                start_target=0.00075 * confirm_step,
            )
        )
    final_target = 0.00075 * confirm_step - 0.00075 * release_steps
    records.append(
        _record(
            confirm_step + 24 + release_steps, "RELEASE_CONFIRMED",
            final_target,
            soft_hold=24, release=release_steps,
            observation_state="RELEASE_CONFIRMED",
            candidate_steps=12,
            stiffness=0.35,
            release_confirm=18,
            start_target=0.00075 * confirm_step,
            final=True,
        )
    )
    for record in records:
        record.update(overrides)
    return records


def _record(
    step, state, target, *,
    soft_hold, release, observation_state, candidate_steps,
    stiffness, release_confirm=0, start_target=None, final=False,
):
    return {
        "global_step": step,
        "phase": "single_finger_contact_characterization",
        "method": "single-finger",
        "state": state,
        "selected_finger": "f1",
        "selected_joint_name": "f1j2",
        "selected_hand_local_index": 1,
        "selected_robot_dof_index": 10,
        "selected_target_rad": target,
        "selected_stiffness_scale": stiffness,
        "selected_q_rad": 0.99 * target,
        "selected_qd_rad_s": 0.0,
        "soft_hold_step": soft_hold,
        "release_step": release,
        "failed": False,
        "failure_reason": None,
        "detector_test_passed": final,
        "transition_events": [],
        "hand_target_rad": [1.0, target, 0.0, 0.0],
        "other_fingers_open_target_invariant": True,
        "release_conditions": (
            {}
            if not final
            else {
                "load_ok": True,
                "travel_ok": True,
                "tracking_ok": True,
                "absolute_load_nm": 0.008,
                "release_threshold_nm": 0.009,
                "travel_rad": release * 0.00075,
                "tracking_error_rad": 0.04,
            }
        ),
        "finger_root_torque_proxy_nm": {"f1": 0.1, "f2": 0.0, "f3": 0.0},
        "hand_q_rad": [1.0, 0.99 * target, 0.0, 0.0],
        "hand_qd_rad_s": [0.0, 0.0, 0.0, 0.0],
        "observation": {
            "state": observation_state,
            "step": step,
            "candidate_steps": candidate_steps,
            "contact_threshold_nm": 0.02,
            "filtered_delta_nm": 0.1 if state in ("SOFT_HOLD", "RELEASE_COMMANDED", "RELEASE_CONFIRMED") else 0.001,
            "filtered_rate_nm_s": 0.0,
            "absolute_load_nm": 0.1 if state in ("SOFT_HOLD", "RELEASE_COMMANDED", "RELEASE_CONFIRMED") else 0.001,
            "load_score": 0.0,
            "raw_delta_nm": 0.12 if state in ("SOFT_HOLD", "RELEASE_COMMANDED", "RELEASE_CONFIRMED") else 0.001,
            "release_steps": 0,
            "release_threshold_nm": 0.009,
            "stalled": False,
        },
        "controller_evidence": {
            "pre_state": state,
            "post_state": state,
            "step": step,
            "release_confirm": release_confirm,
            "release_start_target_rad": start_target,
            "transition_events": [],
        },
        "wrist_wrench_raw_sensor_frame": [0.0] * 6,
        "wrist_wrench_canonical": [0.0, 0.0, 20.5, 0.0, 0.0, 0.0],
        "wrist_wrench_empty_baseline_compensated": [0.0] * 6,
    }


def _audit_points(confirm_step, release_steps, body, nut):
    final_step = confirm_step + 24 + release_steps
    total = body + nut
    return {
        "pre_approach": {
            "point": "pre_approach",
            "global_step": 0,
            "selected_finger": "f1",
            "controller_state": "APPROACH",
            "soft_hold_step": 0,
            "release_step": 0,
            "snapshot": _snapshot(plug_table=1),
        },
        "contact_confirmed": {
            "point": "contact_confirmed",
            "global_step": confirm_step,
            "selected_finger": "f1",
            "controller_state": "SOFT_HOLD",
            "soft_hold_step": 0,
            "release_step": 0,
            "snapshot": _snapshot(
                body=body, nut=nut, material=_material(total)
            ),
        },
        "soft_hold_complete": {
            "point": "soft_hold_complete",
            "global_step": confirm_step + 24,
            "selected_finger": "f1",
            "controller_state": "SOFT_HOLD",
            "soft_hold_step": 24,
            "release_step": 0,
            "snapshot": _snapshot(
                body=body, nut=nut, material=_material(total)
            ),
        },
        "release_confirmed": {
            "point": "release_confirmed",
            "global_step": final_step,
            "selected_finger": "f1",
            "controller_state": "RELEASE_CONFIRMED",
            "soft_hold_step": 24,
            "release_step": release_steps,
            "snapshot": _snapshot(plug_table=1),
        },
    }


def _report(
    *,
    gui=False,
    confirm_step=13,
    release_steps=80,
    body=2,
    nut=3,
    **overrides,
):
    values = {
        "provenance": _provenance(),
        "seed": 0,
        "physical_grasp_method": "single-finger",
        "formal_lift_mode": "zero-lift-hold",
        "attachment": "none",
        "torque_channels": ["f1j2", "f2j1", "f3j2"],
        "gui": gui,
        "process_exit_code": 3,
        "passed": False,
        "grasp_success_claimed": False,
        "control_reads_object_truth": False,
        "control_reads_contact_report": False,
        "posthoc_audit_consumed_by_control": False,
        "posthoc_audit_reads_contact_report": True,
        "formal_truth_firewall_enabled": True,
        "object_pose_writes_after_start": 0,
        "d38999_authoring": {
            "asset_sha256": _digest(11),
            "object_pose_writes_after_start": 0,
        },
        "realized_randomization": {
            "payload_sha256": _digest(10),
            "seed": 0,
            "active_fields": ["plug_mass_scale"],
            "canonical_payload": {"plug_mass_scale": 1.1, "seed": 0},
        },
        "realized_usd_authoring": {
            "usd_authoring_verified": True,
            "material_binding_identity": {
                "grip_material_path": "/World/D38999PickGripMaterial",
                "finger_proxy_count": 8,
                "finger_proxy_all_grip": True,
                "plug_collider_count": 45,
                "plug_collider_all_grip": True,
                "all_bindings_ok": True,
            },
        },
        "proxy_collision_filter": {
            "enabled": False, "mode": "none", "pair_count": 0,
        },
        "physical_grasp_contract": {
            "schema_version": "kcg_d38999_tabletop_physical_grasp_v1",
            "post_grasp_stabilization_proxy_enabled": False,
            "threshold_label": "SIM_TUNING_ONLY",
        },
        "single_finger_release_budget_preflight": {
            "configured_steps": 1200,
            "confirm_steps": 18,
            "feasible": True,
            "filter_tail_steps": 28,
            "headroom_steps": 67,
            "lowpass_alpha": 0.18,
            "maximum_span_rad": 0.765,
            "maximum_torque_delta_gate_nm": 2.0,
            "minimum_possible_release_threshold_nm": 0.009,
            "required_steps": 1133,
            "step_rad": 0.00075,
            "tracking_lag_steps": 67,
            "travel_steps": 1020,
        },
        "single_finger": {
            "selected_finger": "f1",
            "selected_joint_name": "f1j2",
            "selected_hand_local_index": 1,
            "selected_robot_dof_index": 10,
            "detector_test_passed": True,
            "posthoc_contact_audit_passed": None,
            "single_finger_validation_passed": None,
            "soft_hold_step": 24,
            "release_step": release_steps,
            "release_conditions": {
                "load_ok": True,
                "travel_ok": True,
                "tracking_ok": True,
                "absolute_load_nm": 0.008,
                "release_threshold_nm": 0.009,
                "travel_rad": release_steps * 0.00075,
                "tracking_error_rad": 0.04,
            },
            "transition_events": [dict(event) for event in TRANSITION_EVENTS],
            "maximum_post_tare_absolute_delta_by_channel_nm": {
                "f1j2": 0.2, "f2j1": 0.0, "f3j2": 0.0,
            },
            "maximum_post_tare_absolute_delta_nm": 0.2,
        },
        "virtual_wrist_ft_monitor": {
            "status": "SINGLE_FINGER_CONTROL_COMPLETED_POSTHOC_PENDING",
            "last_sample": {
                "global_step": confirm_step + 24 + release_steps,
            },
        },
        "posthoc_audit": {
            "mode": "capture",
            "read_contact_report": True,
            "consumed_by_control": False,
            "points": _audit_points(confirm_step, release_steps, body, nut),
        },
        "runtime_mode_evidence": _runtime_block(gui),
        "finger_root_torque_proxy_baseline_statistics": _finger_block(),
        "formal_empty_wrist_reference_statistics_raw": _wrist_raw_block(),
        "formal_empty_wrist_reference_statistics_residual": (
            _wrist_residual_block()
        ),
        "formal_pregrasp_empty_wrist_baseline": list(WRIST_MEANS),
    }
    values.update(overrides)
    return values


def _pair(**gui_overrides):
    headless = _report(gui=False, confirm_step=13)
    gui_report = _report(gui=True, confirm_step=17, body=1, nut=2)
    gui_report.update(gui_overrides)
    return (
        headless,
        _steps(confirm_step=13, release_steps=80),
        gui_report,
        _steps(confirm_step=17, release_steps=80),
    )


def _run(**kwargs):
    return compare_gui_headless_consistency(*_pair(), **kwargs)


def test_window_statistics_block_schema_and_residual():
    samples = np.linspace(-0.001, 0.001, 120)[:, None] * np.ones((1, 3))
    block = window_statistics_block(
        samples, channel_names=FINGER_CHANNELS,
        frame="tare window", source="test",
    )
    assert block["sample_count"] == 120
    assert block["std_ddof"] == 1
    assert block["evidence_only"] is True
    assert block["threshold_label"] == "SIM_TUNING_ONLY"
    assert block["baseline_subtraction"] is None
    assert set(block["per_channel"]) == set(FINGER_CHANNELS)
    for channel, stats in block["per_channel"].items():
        for name in (
            "mean", "std", "rms", "min", "max",
            "first_half_mean", "second_half_mean",
            "first_to_second_half_drift",
        ):
            assert np.isfinite(stats[name])
    residual_samples = samples - np.mean(samples, axis=0)
    residual = window_statistics_block(
        residual_samples, channel_names=FINGER_CHANNELS,
        frame="tare window", source="test",
        baseline_subtraction="window_mean",
    )
    assert residual["baseline_subtraction"] == "window_mean"
    for stats in residual["per_channel"].values():
        assert abs(stats["mean"]) < 1e-12


def test_window_statistics_block_rejects_bad_inputs():
    with pytest.raises(ValueError):
        window_statistics_block(
            [[0.0]], channel_names=("a",), frame="f", source="s"
        )
    with pytest.raises(ValueError):
        window_statistics_block(
            [[0.0, float("nan")]], channel_names=("a", "b"),
            frame="f", source="s",
        )
    with pytest.raises(ValueError):
        window_statistics_block(
            np.zeros((3, 2)), channel_names=("a",), frame="f", source="s"
        )


def test_happy_path_different_numbers_exits_zero_and_claims_nothing_quantitative():
    summary = _run()
    assert summary["exit_code"] == 0
    assert summary["failure_reason"] is None
    assert summary["functional_structural_gui_headless_consistency_passed"]
    assert summary["quantitative_equivalence_claimed"] is False
    assert summary["quantitative_gates_registered"] is False
    assert summary["input_problems"] == []
    assert summary["schema_problems"] == []
    assert summary["headless_functional_structural_problems"] == []
    assert summary["gui_functional_structural_problems"] == []
    assert summary["evidence_problems"] == []
    statistics = summary["quantitative_statistics"]
    assert statistics["quantitative_equivalence_claimed"] is False
    assert statistics["quantitative_gates_registered"] is False
    assert statistics["values"]["confirm_step"] == {
        "headless": 13, "gui": 17, "delta": 4,
    }
    assert statistics["values"]["confirm_target_rad"]["delta"] == (
        pytest.approx(0.003)
    )
    assert statistics["values"]["contact_records_confirm_body"]["delta"] == -1
    assert "gui_headless_equivalent" not in summary


def test_gui_flag_must_form_false_true_pair():
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["gui"] = False
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 2
    assert "gui_flag_not_false_true_pair" in summary["input_problems"]
    gui_report["gui"] = True
    headless["gui"] = True
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 2


@pytest.mark.parametrize(
    "field, value, side",
    [
        ("seed", 1, "gui"),
        ("finger", "f2", "headless"),
        ("payload_sha256", _digest(99), "gui"),
        ("physical_grasp_config_sha256", _digest(99), "gui"),
        ("runner_sha256", _digest(99), "gui"),
        ("audit_mode", "skip", "gui"),
    ],
)
def test_provenance_mismatch_exits_two(field, value, side):
    headless, headless_steps, gui_report, gui_steps = _pair()
    target = gui_report if side == "gui" else headless
    target["provenance"] = dict(target["provenance"])
    target["provenance"][field] = value
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("control_reads_object_truth", True),
        ("control_reads_contact_report", True),
        ("process_exit_code", 0),
        ("passed", True),
        ("grasp_success_claimed", True),
        ("formal_truth_firewall_enabled", False),
        ("object_pose_writes_after_start", 1),
        ("posthoc_audit_reads_contact_report", False),
    ],
)
def test_safety_truth_boundary_mismatch_exits_two(field, value):
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report[field] = value
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 2


def test_proxy_collision_filter_mismatch_exits_two():
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["proxy_collision_filter"] = {
        "enabled": True, "mode": "nut_only", "pair_count": 3,
    }
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("physics_rate_hz", 480),
        ("tare_actual_steps", 121),
        ("world_physics_dt_s", 1.0 / 480.0),
        ("rendering_dt_probe", None),
    ],
)
def test_runtime_contract_mismatch_exits_two(field, value):
    headless, headless_steps, gui_report, gui_steps = _pair()
    if field == "rendering_dt_probe":
        gui_report["runtime_mode_evidence"] = dict(
            gui_report["runtime_mode_evidence"]
        )
        gui_report["runtime_mode_evidence"]["world_rendering_dt_s"] = (
            1.0 / 30.0
        )
    else:
        gui_report["runtime_mode_evidence"] = dict(
            gui_report["runtime_mode_evidence"]
        )
        gui_report["runtime_mode_evidence"][field] = value
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 2


def test_runtime_gui_descriptor_mismatch_exits_two():
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["runtime_mode_evidence"] = dict(
        gui_report["runtime_mode_evidence"]
    )
    gui_report["runtime_mode_evidence"]["world_step_render_enabled"] = False
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 2
    assert "gui_runtime_render_enabled_mismatch" in summary["input_problems"]


@pytest.mark.parametrize(
    "override, problem_fragment",
    [
        ("confirm_zero_contact", "gui_contact_gate_failed_contact_confirmed"),
        ("confirm_other_finger", "gui_contact_gate_failed_contact_confirmed"),
        ("confirm_unresolved_material", "gui_contact_gate_failed_contact_confirmed"),
        ("release_nonzero_contact", "gui_contact_gate_failed_release_confirmed"),
    ],
)
def test_capture_contact_gate_failure_exits_one(override, problem_fragment):
    headless, headless_steps, gui_report, gui_steps = _pair()
    points = dict(gui_report["posthoc_audit"]["points"])
    if override == "confirm_zero_contact":
        marker = dict(points["contact_confirmed"])
        marker["snapshot"] = _snapshot(
            body=0, nut=0, material=_material(0)
        )
        points["contact_confirmed"] = marker
    elif override == "confirm_other_finger":
        marker = dict(points["contact_confirmed"])
        marker["snapshot"] = _snapshot(
            body=1, nut=2, other=1, material=_material(3)
        )
        points["contact_confirmed"] = marker
    elif override == "confirm_unresolved_material":
        marker = dict(points["contact_confirmed"])
        marker["snapshot"] = _snapshot(
            body=1, nut=2, material={
                "available": True,
                "resolved_records": 0,
                "unresolved_records": 3,
                "grip_grip_records": 0,
                "resolved_non_grip_records": 0,
            },
        )
        points["contact_confirmed"] = marker
    else:
        marker = dict(points["release_confirmed"])
        marker["snapshot"] = _snapshot(body=1, nut=0, plug_table=1)
        points["release_confirmed"] = marker
    gui_report["posthoc_audit"] = dict(gui_report["posthoc_audit"])
    gui_report["posthoc_audit"]["points"] = points
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert any(
        problem_fragment in problem
        for problem in summary["gui_functional_structural_problems"]
    )


def test_transition_skeleton_violation_exits_one():
    headless, headless_steps, gui_report, gui_steps = _pair()
    events = list(gui_report["single_finger"]["transition_events"])
    events[2] = {"from": "CONTACT_CANDIDATE", "to": "SOFT_HOLD", "step": 13}
    gui_report["single_finger"] = dict(gui_report["single_finger"])
    gui_report["single_finger"]["transition_events"] = events
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert any(
        "transition_skeleton" in problem
        for problem in summary["gui_functional_structural_problems"]
    )


def test_confirm_debounce_violations_exit_one():
    headless, headless_steps, gui_report, gui_steps = _pair()
    confirm = next(
        record for record in gui_steps
        if record["state"] == "SOFT_HOLD" and record["soft_hold_step"] == 0
    )
    confirm["observation"]["candidate_steps"] = 11
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert "gui_confirm_debounce_not_12" in (
        summary["gui_functional_structural_problems"]
    )
    confirm["observation"]["candidate_steps"] = 12
    gui_steps[11]["observation"]["candidate_steps"] = 8
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert "gui_confirm_debounce_sequence_wrong" in (
        summary["gui_functional_structural_problems"]
    )


def test_hold_off_by_one_exits_one():
    headless, headless_steps, gui_report, gui_steps = _pair()
    for record in gui_steps:
        if record["state"] == "SOFT_HOLD" and record["soft_hold_step"] >= 1:
            record["soft_hold_step"] -= 1
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert any(
        "hold_step_sequence_wrong" in problem
        for problem in summary["gui_functional_structural_problems"]
    )


def test_release_confirm_insufficient_exits_one():
    headless, headless_steps, gui_report, gui_steps = _pair()
    final = gui_steps[-1]
    final["controller_evidence"]["release_confirm"] = 17
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert "gui_release_confirm_not_18" in (
        summary["gui_functional_structural_problems"]
    )


def test_hold_target_advance_exits_one():
    headless, headless_steps, gui_report, gui_steps = _pair()
    for record in gui_steps:
        if record["state"] == "SOFT_HOLD" and record["soft_hold_step"] == 24:
            record["selected_target_rad"] += 0.00075
            record["hand_target_rad"] = [
                1.0, record["selected_target_rad"], 0.0, 0.0,
            ]
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert "gui_hold_target_not_exactly_frozen" in (
        summary["gui_functional_structural_problems"]
    )


def test_hold_stiffness_violation_exits_one():
    headless, headless_steps, gui_report, gui_steps = _pair()
    for record in gui_steps:
        if record["state"] == "SOFT_HOLD":
            record["selected_stiffness_scale"] = 0.34
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert "gui_hold_stiffness_not_0.35" in (
        summary["gui_functional_structural_problems"]
    )


def test_release_direction_reversal_exits_one():
    headless, headless_steps, gui_report, gui_steps = _pair()
    releases = [
        record for record in gui_steps
        if record["state"] == "RELEASE_COMMANDED"
    ]
    releases[5]["selected_target_rad"] = (
        releases[4]["selected_target_rad"] + 0.0015
    )
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert "gui_release_target_not_monotonic_toward_open" in (
        summary["gui_functional_structural_problems"]
    )


@pytest.mark.parametrize("gate", ["load_ok", "travel_ok", "tracking_ok"])
def test_terminal_release_gate_false_exits_one(gate):
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_steps[-1]["release_conditions"][gate] = False
    gui_report["single_finger"] = dict(gui_report["single_finger"])
    gui_report["single_finger"]["release_conditions"] = dict(
        gui_report["single_finger"]["release_conditions"]
    )
    gui_report["single_finger"]["release_conditions"][gate] = False
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert f"gui_final_release_gate_{gate}_false" in (
        summary["gui_functional_structural_problems"]
    )


def test_other_finger_invariant_false_exits_one():
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_steps[3]["other_fingers_open_target_invariant"] = False
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert "gui_other_finger_invariant_false_at_3" in (
        summary["gui_functional_structural_problems"]
    )


def test_missing_statistics_block_exits_three():
    headless, headless_steps, gui_report, gui_steps = _pair()
    del gui_report["finger_root_torque_proxy_baseline_statistics"]
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3
    assert summary["failure_reason"] == "evidence_incomplete_or_inconsistent"
    assert "gui_missing_finger_root_torque_proxy_baseline_statistics" in (
        summary["evidence_problems"]
    )
    assert summary["quantitative_statistics"] is None


def test_statistics_nan_exits_three():
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["finger_root_torque_proxy_baseline_statistics"] = (
        _finger_block(nan_channel="f2j1")
    )
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3
    assert any(
        "not_finite" in problem for problem in summary["evidence_problems"]
    )


def test_statistics_count_mismatch_exits_three():
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["formal_empty_wrist_reference_statistics_raw"] = (
        _wrist_raw_block(count=121)
    )
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3
    assert "gui_stats_counts_inconsistent" in summary["evidence_problems"]


def test_residual_mean_relation_violation_exits_three():
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["formal_empty_wrist_reference_statistics_residual"] = (
        _wrist_residual_block()
    )
    gui_report["formal_empty_wrist_reference_statistics_residual"][
        "per_channel"
    ]["fx_n"]["mean"] = 0.01
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3
    assert "gui_residual_mean_not_zero_fx_n" in summary["evidence_problems"]


def test_residual_subtraction_label_violation_exits_three():
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["formal_empty_wrist_reference_statistics_residual"] = (
        _wrist_residual_block(subtraction="payload_reference")
    )
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3
    assert "gui_residual_baseline_subtraction_wrong" in (
        summary["evidence_problems"]
    )


def test_raw_mean_vs_report_baseline_violation_exits_three():
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["formal_empty_wrist_reference_statistics_raw"] = (
        _wrist_raw_block(means=[0.11, 0.2, 20.5, 0.01, 0.02, 0.001])
    )
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3
    assert "gui_raw_mean_vs_report_baseline_fx_n" in (
        summary["evidence_problems"]
    )


def test_request_quantitative_equivalence_exits_three():
    summary = _run(request_quantitative_gates=True)
    assert summary["exit_code"] == 3
    assert summary["failure_reason"] == (
        "quantitative_equivalence_requested_without_registered_gates"
    )
    assert summary["functional_structural_gui_headless_consistency_passed"] is False


def _write_episode(directory: Path, report, steps):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nominal_physics_report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    (directory / "controller_steps.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in steps),
        encoding="utf-8",
    )


def _run_cli(*arguments):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = SOURCE_ROOT + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    return subprocess.run(
        [sys.executable, str(COMPARATOR), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_cli_happy_path_exits_zero_and_writes_json(tmp_path):
    headless_dir = tmp_path / "headless_episode"
    gui_dir = tmp_path / "gui_episode"
    output_dir = tmp_path / "comparison_output"
    headless, headless_steps, gui_report, gui_steps = _pair()
    _write_episode(headless_dir, headless, headless_steps)
    _write_episode(gui_dir, gui_report, gui_steps)
    result = _run_cli(
        "--headless-dir", str(headless_dir),
        "--gui-dir", str(gui_dir),
        "--output-dir", str(output_dir),
    )
    assert result.returncode == 0
    summary = json.loads(
        (output_dir / "gui_headless_consistency_comparison.json").read_text(
            encoding="utf-8",
        )
    )
    assert summary["exit_code"] == 0
    assert summary["functional_structural_gui_headless_consistency_passed"]
    assert summary["quantitative_equivalence_claimed"] is False
    assert "gui_headless_equivalent" not in summary
    source = summary["source"]
    assert len(source["cli_sha256"]) == 64
    assert len(source["evaluator_module_sha256"]) == 64
    assert source["cli_sha256"] != source["evaluator_module_sha256"]


def test_cli_quantitative_request_exits_three(tmp_path):
    headless_dir = tmp_path / "headless_episode"
    gui_dir = tmp_path / "gui_episode"
    output_dir = tmp_path / "comparison_output"
    headless, headless_steps, gui_report, gui_steps = _pair()
    _write_episode(headless_dir, headless, headless_steps)
    _write_episode(gui_dir, gui_report, gui_steps)
    result = _run_cli(
        "--headless-dir", str(headless_dir),
        "--gui-dir", str(gui_dir),
        "--output-dir", str(output_dir),
        "--request-quantitative-equivalence",
    )
    assert result.returncode == 3


def test_cli_output_inside_episode_refused(tmp_path):
    headless_dir = tmp_path / "headless_episode"
    gui_dir = tmp_path / "gui_episode"
    headless, headless_steps, gui_report, gui_steps = _pair()
    _write_episode(headless_dir, headless, headless_steps)
    _write_episode(gui_dir, gui_report, gui_steps)
    result = _run_cli(
        "--headless-dir", str(headless_dir),
        "--gui-dir", str(gui_dir),
        "--output-dir", str(headless_dir / "nested"),
    )
    assert result.returncode == 2


def test_cli_nonempty_output_refused(tmp_path):
    headless_dir = tmp_path / "headless_episode"
    gui_dir = tmp_path / "gui_episode"
    output_dir = tmp_path / "comparison_output"
    headless, headless_steps, gui_report, gui_steps = _pair()
    _write_episode(headless_dir, headless, headless_steps)
    _write_episode(gui_dir, gui_report, gui_steps)
    output_dir.mkdir(parents=True)
    (output_dir / "occupied.txt").write_text("x", encoding="utf-8")
    result = _run_cli(
        "--headless-dir", str(headless_dir),
        "--gui-dir", str(gui_dir),
        "--output-dir", str(output_dir),
    )
    assert result.returncode == 2


def test_cli_missing_files_exits_two(tmp_path):
    headless_dir = tmp_path / "headless_episode"
    gui_dir = tmp_path / "gui_episode"
    output_dir = tmp_path / "comparison_output"
    headless_dir.mkdir(parents=True)
    gui_dir.mkdir(parents=True)
    result = _run_cli(
        "--headless-dir", str(headless_dir),
        "--gui-dir", str(gui_dir),
        "--output-dir", str(output_dir),
    )
    assert result.returncode == 2



def test_raw_mean_missing_is_evidence_problem_not_exception():
    headless, headless_steps, gui_report, gui_steps = _pair()
    block = gui_report["formal_empty_wrist_reference_statistics_raw"]
    del block["per_channel"]["fx_n"]["mean"]
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3
    assert any(
        "not_finite" in problem for problem in summary["evidence_problems"]
    )


def test_raw_mean_wrong_type_is_evidence_problem_not_exception():
    headless, headless_steps, gui_report, gui_steps = _pair()
    block = gui_report["formal_empty_wrist_reference_statistics_raw"]
    block["per_channel"]["fx_n"]["mean"] = "not-a-number"
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3
    assert any(
        "not_finite" in problem for problem in summary["evidence_problems"]
    )


@pytest.mark.parametrize("bad", [float("nan"), "bad", True])
def test_baseline_element_invalid_is_evidence_problem(bad):
    headless, headless_steps, gui_report, gui_steps = _pair()
    baseline = list(WRIST_MEANS)
    baseline[0] = bad
    gui_report["formal_pregrasp_empty_wrist_baseline"] = baseline
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3
    assert any(
        "baseline" in problem for problem in summary["evidence_problems"]
    )


@pytest.mark.parametrize("bad", ["bad", [0.0], (0.1, 0.2), None])
def test_baseline_whole_invalid_is_evidence_problem(bad):
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["formal_pregrasp_empty_wrist_baseline"] = bad
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3
    assert "gui_wrist_baseline_invalid" in summary["evidence_problems"]


def test_per_channel_stats_not_mapping_is_evidence_problem():
    headless, headless_steps, gui_report, gui_steps = _pair()
    block = gui_report["formal_empty_wrist_reference_statistics_raw"]
    block["per_channel"]["fx_n"] = "not-a-mapping"
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 3


def test_comparator_never_raises_on_malformed_statistics():
    headless, headless_steps, gui_report, gui_steps = _pair()
    malformations = [
        ("formal_empty_wrist_reference_statistics_raw", "per_channel", "fx_n"),
        ("formal_empty_wrist_reference_statistics_raw", "per_channel", None),
        ("finger_root_torque_proxy_baseline_statistics", "per_channel", "f1j2"),
    ]
    for block_name, section, channel in malformations:
        candidate = dict(gui_report)
        if channel is None:
            candidate[block_name] = {
                **candidate[block_name],
                section: "not-a-mapping",
            }
        else:
            candidate[block_name] = dict(candidate[block_name])
            candidate[block_name][section] = dict(
                candidate[block_name][section]
            )
            candidate[block_name][section][channel] = {
                "mean": "not-a-number", "std": 0.001,
                "rms": 0.001, "min": -0.002, "max": 0.002,
                "first_half_mean": 0.0, "second_half_mean": 0.0,
                "first_to_second_half_drift": 0.0,
            }
        summary = compare_gui_headless_consistency(
            headless, headless_steps, candidate, gui_steps
        )
        assert summary["exit_code"] == 3


def test_cli_malformed_statistics_yields_json_exit3_no_traceback(tmp_path):
    headless_dir = tmp_path / "headless_episode"
    gui_dir = tmp_path / "gui_episode"
    output_dir = tmp_path / "comparison_output"
    headless, headless_steps, gui_report, gui_steps = _pair()
    block = gui_report["formal_empty_wrist_reference_statistics_raw"]
    block["per_channel"]["fx_n"]["mean"] = "not-a-number"
    _write_episode(headless_dir, headless, headless_steps)
    _write_episode(gui_dir, gui_report, gui_steps)
    result = _run_cli(
        "--headless-dir", str(headless_dir),
        "--gui-dir", str(gui_dir),
        "--output-dir", str(output_dir),
    )
    assert result.returncode == 3
    assert "Traceback" not in result.stderr
    summary = json.loads(
        (output_dir / "gui_headless_consistency_comparison.json").read_text(
            encoding="utf-8",
        )
    )
    assert summary["exit_code"] == 3
    assert summary["failure_reason"] == "evidence_incomplete_or_inconsistent"
    assert summary["quantitative_equivalence_claimed"] is False


def _marker_problems(summary):
    return (
        summary["headless_functional_structural_problems"]
        + summary["gui_functional_structural_problems"]
    )


# Frozen exit semantics: marker input-schema violations are exit 2
# (input/schema/provenance contract invalid), while marker/trace structure
# and frozen-value mismatches on schema-valid markers are exit 1.


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        ("extra_key", "marker_extra_keys"),
        ("snapshot_with_error", "snapshot_with_error"),
        ("snapshot_not_mapping", "snapshot_not_mapping"),
        ("soft_hold_bool", "soft_hold_step_type"),
        ("soft_hold_string", "soft_hold_step_type"),
        ("point_type", "point_type"),
        ("global_step_bool", "global_step_type"),
        ("global_step_string", "global_step_type"),
        ("release_step_string", "release_step_type"),
        ("release_step_bool", "release_step_type"),
        ("controller_state_nonstring", "state_type"),
        ("marker_set_missing", "marker_set_not_exact"),
    ],
)
def test_marker_schema_violations_exit_two(mutate, fragment):
    headless, headless_steps, gui_report, gui_steps = _pair()
    points = gui_report["posthoc_audit"]["points"]
    if mutate in ("release_step_string", "release_step_bool"):
        marker = points["release_confirmed"]
        marker["release_step"] = (
            "bad" if mutate == "release_step_string" else False
        )
    else:
        marker = points["contact_confirmed"]
        if mutate == "extra_key":
            marker["bogus_field"] = 1
        elif mutate == "snapshot_with_error":
            marker["error"] = "boom"
        elif mutate == "snapshot_not_mapping":
            marker["snapshot"] = None
        elif mutate == "soft_hold_bool":
            marker["soft_hold_step"] = False
        elif mutate == "soft_hold_string":
            marker["soft_hold_step"] = "0"
        elif mutate == "point_type":
            marker["point"] = 5
        elif mutate == "global_step_bool":
            marker["global_step"] = True
        elif mutate == "global_step_string":
            marker["global_step"] = "bad"
        elif mutate == "controller_state_nonstring":
            marker["controller_state"] = 5
        else:
            del points["soft_hold_complete"]
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 2
    assert any(fragment in problem for problem in summary["schema_problems"])


def test_marker_trace_selected_finger_mismatch_fails_closed():
    headless, headless_steps, gui_report, gui_steps = _pair()
    for record in gui_steps:
        if record["global_step"] == 17:
            record["selected_finger"] = "f2"
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    # The trace record itself violates the per-step schema (exit 2
    # precedence), but the shared helper must classify the marker/trace
    # finger mismatch in the functional bucket.
    assert summary["exit_code"] == 2
    assert any(
        "finger_misaligned" in problem for problem in _marker_problems(summary)
    )


def test_marker_trace_state_misalignment_exits_one():
    headless, headless_steps, gui_report, gui_steps = _pair()
    final = gui_steps[-1]["global_step"]
    gui_report["posthoc_audit"]["points"]["contact_confirmed"][
        "global_step"
    ] = final
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert any(
        "trace_state_misaligned" in problem
        for problem in _marker_problems(summary)
    )


def test_marker_duplicate_global_steps_fail_closed():
    headless, headless_steps, gui_report, gui_steps = _pair()
    points = gui_report["posthoc_audit"]["points"]
    points["soft_hold_complete"]["global_step"] = points[
        "contact_confirmed"
    ]["global_step"]
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert any(
        "not_increasing" in problem for problem in _marker_problems(summary)
    )


def test_marker_reversed_global_steps_fail_closed():
    headless, headless_steps, gui_report, gui_steps = _pair()
    points = gui_report["posthoc_audit"]["points"]
    first = points["contact_confirmed"]["global_step"]
    second = points["soft_hold_complete"]["global_step"]
    points["contact_confirmed"]["global_step"] = second
    points["soft_hold_complete"]["global_step"] = first
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps
    )
    assert summary["exit_code"] == 1
    assert any(
        "not_increasing" in problem for problem in _marker_problems(summary)
    )


def test_cli_marker_schema_error_exits_two_without_traceback(tmp_path):
    headless_dir = tmp_path / "headless_episode"
    gui_dir = tmp_path / "gui_episode"
    output_dir = tmp_path / "comparison_output"
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["posthoc_audit"]["points"]["contact_confirmed"][
        "bogus_field"
    ] = 1
    _write_episode(headless_dir, headless, headless_steps)
    _write_episode(gui_dir, gui_report, gui_steps)
    result = _run_cli(
        "--headless-dir", str(headless_dir),
        "--gui-dir", str(gui_dir),
        "--output-dir", str(output_dir),
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    summary = json.loads(
        (output_dir / "gui_headless_consistency_comparison.json").read_text(
            encoding="utf-8",
        )
    )
    assert summary["exit_code"] == 2
    assert any(
        "marker_extra_keys" in problem
        for problem in summary["schema_problems"]
    )



def test_compare_never_raises_on_structurally_malformed_reports():
    headless, headless_steps, gui_report, gui_steps = _pair()
    cases = []
    case_a = dict(gui_report)
    case_a["single_finger"] = ["not", "a", "mapping"]
    cases.append(case_a)
    case_b = dict(gui_report)
    case_b["posthoc_audit"] = "not-a-mapping"
    cases.append(case_b)
    case_c = dict(gui_report)
    case_c["virtual_wrist_ft_monitor"] = []
    cases.append(case_c)
    for candidate in cases:
        summary = compare_gui_headless_consistency(
            headless, headless_steps, candidate, gui_steps
        )
        assert summary["exit_code"] == 2
        assert summary["quantitative_statistics"] is None


def test_compare_never_raises_on_non_mapping_step_records():
    headless, headless_steps, gui_report, gui_steps = _pair()
    summary = compare_gui_headless_consistency(
        headless, headless_steps, gui_report, gui_steps + [[1, 2, 3]]
    )
    assert summary["exit_code"] == 2
    assert any(
        "unexpected_evaluation_error" in problem
        for problem in summary["input_problems"]
    )


def test_cli_structurally_malformed_report_exits_two_without_traceback(tmp_path):
    headless_dir = tmp_path / "headless_episode"
    gui_dir = tmp_path / "gui_episode"
    output_dir = tmp_path / "comparison_output"
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["single_finger"] = ["not", "a", "mapping"]
    _write_episode(headless_dir, headless, headless_steps)
    _write_episode(gui_dir, gui_report, gui_steps)
    result = _run_cli(
        "--headless-dir", str(headless_dir),
        "--gui-dir", str(gui_dir),
        "--output-dir", str(output_dir),
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    summary = json.loads(
        (output_dir / "gui_headless_consistency_comparison.json").read_text(
            encoding="utf-8",
        )
    )
    assert summary["exit_code"] == 2



def test_cli_release_step_string_schema_error_exit_two_no_traceback(tmp_path):
    headless_dir = tmp_path / "headless_episode"
    gui_dir = tmp_path / "gui_episode"
    output_dir = tmp_path / "comparison_output"
    headless, headless_steps, gui_report, gui_steps = _pair()
    gui_report["posthoc_audit"]["points"]["release_confirmed"][
        "release_step"
    ] = "bad"
    _write_episode(headless_dir, headless, headless_steps)
    _write_episode(gui_dir, gui_report, gui_steps)
    result = _run_cli(
        "--headless-dir", str(headless_dir),
        "--gui-dir", str(gui_dir),
        "--output-dir", str(output_dir),
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    summary = json.loads(
        (output_dir / "gui_headless_consistency_comparison.json").read_text(
            encoding="utf-8",
        )
    )
    assert summary["exit_code"] == 2
    assert any(
        "release_step_type" in problem
        for problem in summary["schema_problems"]
    )

