'''Pure tests for the single-finger posthoc audit evaluator and comparator.'''

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kcg_connector.grasp.single_finger_posthoc_audit import (
    canonical_sha256,
    compare_episodes,
    evaluate_audit_point,
    verify_capture_episode_markers,
    verify_capture_episode_markers_classified,
    verify_markers,
)

SHA = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def _digest(index=0):
    return f"{index:064x}"[-64:]


def _snapshot(
    *,
    selected_body=0,
    selected_nut=0,
    other_total=0,
    unexpected=0,
    plug_table=1,
    material=None,
):
    snapshot = {
        "finger_body_group_records": {
            "f1": {"body": 0, "nut": 0},
            "f2": {"body": 0, "nut": 0},
            "f3": {"body": 0, "nut": 0},
        },
        "plug_table_records": plug_table,
        "unexpected_robot_link_records": unexpected,
    }
    snapshot["finger_body_group_records"]["f1"] = {
        "body": selected_body,
        "nut": selected_nut,
    }
    snapshot["finger_body_group_records"]["f2"] = {
        "body": other_total,
        "nut": 0,
    }
    if material is not None:
        snapshot["material_evidence"] = material
    return snapshot


def _point_evaluation(point, **kwargs):
    if (
        point in ("contact_confirmed", "soft_hold_complete")
        and "material" not in kwargs
    ):
        total = kwargs.get("selected_body", 0) + kwargs.get(
            "selected_nut", 0
        )
        kwargs["material"] = {
            "available": total > 0,
            "resolved_records": total,
            "unresolved_records": 0,
            "grip_grip_records": total,
            "resolved_non_grip_records": 0,
        }
    return evaluate_audit_point(point, _snapshot(**kwargs), "f1")


def test_contact_points_accept_body_or_nut_contact():
    assert _point_evaluation("contact_confirmed", selected_body=3)["passed"]
    assert _point_evaluation("contact_confirmed", selected_nut=2)["passed"]
    assert _point_evaluation(
        "contact_confirmed", selected_body=1, selected_nut=1
    )["passed"]


def test_contact_points_reject_other_finger_and_unexpected_link():
    assert not _point_evaluation("contact_confirmed", other_total=1)["passed"]
    assert not _point_evaluation(
        "contact_confirmed", unexpected=1
    )["passed"]
    assert not _point_evaluation("contact_confirmed")["passed"]


def test_pre_and_release_require_no_contact_and_plug_on_table():
    assert _point_evaluation("pre_approach", plug_table=3)["passed"]
    assert not _point_evaluation("pre_approach", selected_body=1)["passed"]
    assert not _point_evaluation("pre_approach", plug_table=0)["passed"]
    assert _point_evaluation("release_confirmed", plug_table=2)["passed"]
    assert not _point_evaluation(
        "release_confirmed", selected_nut=1
    )["passed"]


def test_soft_hold_plug_table_is_context_not_gate():
    hold = _point_evaluation(
        "soft_hold_complete", selected_body=1, plug_table=0
    )
    assert hold["passed"]
    assert hold["gates"]["plug_table_required"] is False
    assert hold["gates"]["plug_table_present"] is False


def test_material_modes_fully_resolved_unresolved_partial():
    grip = {
        "available": True,
        "resolved_records": 2,
        "unresolved_records": 0,
        "grip_grip_records": 2,
        "resolved_non_grip_records": 0,
    }
    ok = _point_evaluation(
        "contact_confirmed", selected_body=2, material=grip
    )
    assert ok["gates"]["material_evidence_consistent"] is True
    assert ok["material_evidence"]["mode"] == "fully_resolved"
    non_grip = dict(grip, resolved_non_grip_records=1, grip_grip_records=1)
    assert not _point_evaluation(
        "contact_confirmed", selected_body=2, material=non_grip
    )["gates"]["material_evidence_consistent"]
    # Partial resolution must fail closed, never degrade silently.
    partial = dict(grip, unresolved_records=1)
    partial_eval = _point_evaluation(
        "contact_confirmed", selected_body=2, material=partial
    )
    assert partial_eval["material_evidence"]["mode"] == (
        "partial_unresolved_fail_closed"
    )
    assert not partial_eval["gates"]["material_evidence_consistent"]
    unresolved = {
        "available": False,
        "resolved_records": 0,
        "unresolved_records": 2,
        "grip_grip_records": 0,
        "resolved_non_grip_records": 0,
    }
    fallback_fail = evaluate_audit_point(
        "contact_confirmed",
        _snapshot(selected_body=2, material=unresolved),
        "f1",
        binding_identity_ok=False,
    )
    assert not fallback_fail["gates"]["material_evidence_consistent"]
    fallback_ok = evaluate_audit_point(
        "contact_confirmed",
        _snapshot(selected_body=2, material=unresolved),
        "f1",
        binding_identity_ok=True,
    )
    assert fallback_ok["gates"]["material_evidence_consistent"] is True
    assert fallback_ok["material_evidence"]["mode"] == (
        "fully_unresolved_binding_fallback"
    )


def test_evaluate_audit_point_validates_point_and_finger():
    with pytest.raises(ValueError, match="unknown audit point"):
        evaluate_audit_point("nonsense", _snapshot(), "f1")
    with pytest.raises(ValueError, match="selected finger"):
        evaluate_audit_point("pre_approach", _snapshot(), "f9")


def test_missing_snapshot_fails_all_gates():
    result = evaluate_audit_point("contact_confirmed", None, "f1")
    assert result["passed"] is False
    assert result["snapshot_missing"] is True


def _provenance(audit_mode="skip", **overrides):
    values = {
        "seed": 0,
        "finger": "f1",
        "audit_mode": audit_mode,
        "payload_sha256": SHA,
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


def _marker(point, global_step, state, soft_hold, release, finger="f1"):
    return {
        "point": point,
        "global_step": global_step,
        "selected_finger": finger,
        "controller_state": state,
        "soft_hold_step": soft_hold,
        "release_step": release,
    }


def _step(state, *, step, soft_hold=0, release=0, **overrides):
    values = {
        "global_step": step,
        "phase": "single_finger_contact_characterization",
        "state": state,
        "selected_finger": "f1",
        "selected_joint_name": "f1j2",
        "selected_hand_local_index": 1,
        "selected_robot_dof_index": 8,
        "selected_target_rad": 0.5,
        "selected_stiffness_scale": 1.0,
        "soft_hold_step": soft_hold,
        "release_step": release,
        "failed": False,
        "failure_reason": None,
        "detector_test_passed": state == "RELEASE_CONFIRMED",
        "transition_events": [],
        "hand_target_rad": [1.0, 0.0, 0.0, 0.0],
        "other_fingers_open_target_invariant": True,
        "release_conditions": {},
        "selected_q_rad": 0.4,
        "selected_qd_rad_s": 0.0,
        "finger_root_torque_proxy_nm": {"f1": 0.1, "f2": 0.0, "f3": 0.0},
        "hand_q_rad": [1.0, 0.4, 0.0, 0.0],
        "hand_qd_rad_s": [0.0, 0.0, 0.0, 0.0],
        "observation": {"state": state, "step": step},
        "controller_evidence": {"post_state": state},
        "wrist_wrench_raw_sensor_frame": [0.0] * 6,
        "wrist_wrench_canonical": [0.0] * 6,
        "wrist_wrench_empty_baseline_compensated": [0.0] * 6,
    }
    values.update(overrides)
    return values


# Frozen controller timeline used by the synthetic episodes:
#   step 10 APPROACH, 20 SOFT_HOLD/0 (confirm), 44 SOFT_HOLD/24 (hold done),
#   50 RELEASE_CONFIRMED/24/6.
def _control_steps():
    return [
        _step("APPROACH", step=10),
        _step("SOFT_HOLD", step=20, soft_hold=0),
        _step("SOFT_HOLD", step=44, soft_hold=24),
        _step(
            "RELEASE_CONFIRMED",
            step=50,
            soft_hold=24,
            release=6,
            release_conditions={
                "load_ok": True,
                "travel_ok": True,
                "tracking_ok": True,
            },
        ),
    ]


def _audit_points(capture=False):
    grip = {
        "available": True,
        "resolved_records": 2,
        "unresolved_records": 0,
        "grip_grip_records": 2,
        "resolved_non_grip_records": 0,
    }
    points = {
        "pre_approach": {
            **_marker("pre_approach", 9, "APPROACH", 0, 0),
            "snapshot": _snapshot(plug_table=2) if capture else None,
        },
        "contact_confirmed": {
            **_marker("contact_confirmed", 20, "SOFT_HOLD", 0, 0),
            "snapshot": (
                _snapshot(selected_body=2, material=grip)
                if capture
                else None
            ),
        },
        "soft_hold_complete": {
            **_marker("soft_hold_complete", 44, "SOFT_HOLD", 24, 0),
            "snapshot": (
                _snapshot(selected_body=2, material=grip)
                if capture
                else None
            ),
        },
        "release_confirmed": {
            **_marker(
                "release_confirmed", 50, "RELEASE_CONFIRMED", 24, 6
            ),
            "snapshot": _snapshot(plug_table=2) if capture else None,
        },
    }
    return points


def _binding_identity(ok=True):
    return {
        "grip_material_path": "/World/D38999PickGripMaterial",
        "finger_proxy_count": 8,
        "finger_proxy_all_grip": ok,
        "plug_collider_count": 45,
        "plug_collider_all_grip": ok,
        "all_bindings_ok": ok,
    }


def _report(audit_mode="capture", points=None, **overrides):
    values = {
        "provenance": _provenance(audit_mode),
        "seed": 0,
        "physical_grasp_method": "single-finger",
        "formal_lift_mode": "zero-lift-hold",
        "gui": False,
        "process_exit_code": 3,
        "passed": False,
        "grasp_success_claimed": False,
        "control_reads_object_truth": False,
        "control_reads_contact_report": False,
        "posthoc_audit_reads_contact_report": audit_mode == "capture",
        "posthoc_audit_consumed_by_control": False,
        "formal_truth_firewall_enabled": True,
        "object_pose_writes_after_start": 0,
        "physical_grasp_contract": {
            "post_grasp_stabilization_proxy_enabled": False,
        },
        "realized_usd_authoring": {
            "usd_authoring_verified": True,
            "material_binding_identity": _binding_identity(),
        },
        "single_finger": {
            "selected_finger": "f1",
            "detector_test_passed": True,
            "posthoc_contact_audit_passed": None,
            "single_finger_validation_passed": None,
            "release_step": 6,
            "soft_hold_step": 24,
            "release_conditions": {
                "load_ok": True,
                "travel_ok": True,
                "tracking_ok": True,
            },
            "transition_events": [],
            "maximum_post_tare_absolute_delta_by_channel_nm": {
                "f1j2": 0.2,
                "f2j1": 0.0,
                "f3j2": 0.0,
            },
            "maximum_post_tare_absolute_delta_nm": 0.2,
        },
        "virtual_wrist_ft_monitor": {
            "status": "SINGLE_FINGER_CONTROL_COMPLETED_POSTHOC_PENDING",
            "last_sample": {"global_step": 50},
        },
        "posthoc_audit": {
            "mode": audit_mode,
            "read_contact_report": audit_mode == "capture",
            "points": (
                points
                if points is not None
                else _audit_points(audit_mode == "capture")
            ),
            "consumed_by_control": False,
        },
    }
    values.update(overrides)
    return values


def test_comparator_happy_path_exits_zero_and_validates():
    summary = compare_episodes(
        _report("capture"),
        _control_steps(),
        _report("skip"),
        _control_steps(),
    )
    assert summary["exit_code"] == 0, summary
    assert summary["single_finger_validation_passed"] is True
    assert summary["grasp_success_claimed"] is False
    assert summary["marker_problems"] == []
    assert summary["decision_trace_identical"] is True
    assert summary["sensor_trace_identical"] is True
    assert summary["report_terminal_identical"] is True


def test_comparator_gate_failure_exits_one():
    points = _audit_points(True)
    points["contact_confirmed"]["snapshot"] = _snapshot(selected_body=0)
    summary = compare_episodes(
        _report("capture", points=points),
        _control_steps(),
        _report("skip"),
        _control_steps(),
    )
    assert summary["exit_code"] == 1
    assert summary["failure_reason"] == "contact_gate_failed"
    assert summary["single_finger_validation_passed"] is False


def test_comparator_partial_material_unresolved_is_gate_failure():
    points = _audit_points(True)
    partial = {
        "available": True,
        "resolved_records": 1,
        "unresolved_records": 1,
        "grip_grip_records": 1,
        "resolved_non_grip_records": 0,
    }
    points["contact_confirmed"]["snapshot"] = _snapshot(
        selected_body=2, material=partial
    )
    summary = compare_episodes(
        _report("capture", points=points),
        _control_steps(),
        _report("skip"),
        _control_steps(),
    )
    assert summary["exit_code"] == 1
    assert summary["failure_reason"] == "contact_gate_failed"


def test_comparator_unresolved_fallback_requires_binding_identity():
    points = _audit_points(True)
    unresolved = {
        "available": False,
        "resolved_records": 0,
        "unresolved_records": 2,
        "grip_grip_records": 0,
        "resolved_non_grip_records": 0,
    }
    for point in ("contact_confirmed", "soft_hold_complete"):
        points[point]["snapshot"] = _snapshot(
            selected_body=2, material=unresolved
        )
    broken_report = _report("capture", points=points)
    broken_report["realized_usd_authoring"]["material_binding_identity"] = (
        _binding_identity(ok=False)
    )
    broken_summary = compare_episodes(
        broken_report, _control_steps(), _report("skip"), _control_steps()
    )
    assert broken_summary["exit_code"] == 3
    assert "material_binding_identity_contract_failed" in (
        broken_summary["capture_contract_problems"]
    )
    ok_summary = compare_episodes(
        _report("capture", points=points),
        _control_steps(),
        _report("skip"),
        _control_steps(),
    )
    assert ok_summary["exit_code"] == 0


def test_comparator_marker_off_by_one_is_inconclusive():
    points = _audit_points(True)
    points["contact_confirmed"] = {
        **_marker("contact_confirmed", 21, "SOFT_HOLD", 1, 0),
        "snapshot": _snapshot(selected_body=2),
    }
    summary = compare_episodes(
        _report("capture", points=points),
        _control_steps(),
        _report("skip"),
        _control_steps(),
    )
    assert summary["exit_code"] == 3
    assert summary["marker_problems"]
    assert "contact_confirmed" in " ".join(summary["marker_problems"])


def test_comparator_marker_disagreement_between_modes_is_inconclusive():
    capture_points = _audit_points(True)
    skip_points = _audit_points(False)
    skip_points["contact_confirmed"]["global_step"] = 21
    summary = compare_episodes(
        _report("capture", points=capture_points),
        _control_steps(),
        _report("skip", points=skip_points),
        _control_steps(),
    )
    assert summary["exit_code"] == 3
    assert any(
        "marker_mismatch" in problem for problem in summary["marker_problems"]
    )


def test_comparator_marker_order_violation_is_inconclusive():
    points = _audit_points(True)
    points["pre_approach"]["global_step"] = 20
    points["contact_confirmed"]["global_step"] = 9
    summary = compare_episodes(
        _report("capture", points=points),
        _control_steps(),
        _report("skip"),
        _control_steps(),
    )
    assert summary["exit_code"] == 3
    assert "marker_global_steps_not_increasing" in summary["marker_problems"]


def test_comparator_single_sensor_field_mismatch_reports_path():
    capture_steps = _control_steps()
    skip_steps = _control_steps()
    skip_steps[1]["hand_q_rad"] = [1.0, 0.3999999, 0.0, 0.0]
    summary = compare_episodes(
        _report("capture"),
        capture_steps,
        _report("skip"),
        skip_steps,
    )
    assert summary["exit_code"] == 3
    assert summary["failure_reason"] == "inconclusive_query_noninterference"
    mismatch = summary["first_sensor_mismatch"]
    assert mismatch is not None
    assert mismatch["path"][:2] == ["1", "hand_q_rad"]
    assert summary["sensor_trace_sha256_capture"] != (
        summary["sensor_trace_sha256_skip"]
    )


def test_comparator_report_terminal_mismatch_is_inconclusive():
    skip_report = _report("skip")
    skip_report["single_finger"]["maximum_post_tare_absolute_delta_nm"] = 0.3
    summary = compare_episodes(
        _report("capture"),
        _control_steps(),
        skip_report,
        _control_steps(),
    )
    assert summary["exit_code"] == 3
    assert summary["first_terminal_mismatch"] is not None


def test_comparator_episode_contract_violation_is_inconclusive():
    capture_report = _report("capture")
    capture_report["process_exit_code"] = 0
    summary = compare_episodes(
        capture_report, _control_steps(), _report("skip"), _control_steps()
    )
    assert summary["exit_code"] == 3
    assert "exit_code_not_3" in summary["capture_contract_problems"]

    capture_report = _report("capture")
    capture_report["control_reads_object_truth"] = True
    summary = compare_episodes(
        capture_report, _control_steps(), _report("skip"), _control_steps()
    )
    assert summary["exit_code"] == 3
    assert "control_reads_object_truth_not_false" in (
        summary["capture_contract_problems"]
    )

    capture_report = _report("capture")
    capture_report["passed"] = True
    summary = compare_episodes(
        capture_report, _control_steps(), _report("skip"), _control_steps()
    )
    assert summary["exit_code"] == 3
    assert "passed_not_false" in summary["capture_contract_problems"]


def test_comparator_provenance_contract_mismatches_exit_two():
    with pytest.raises(ValueError, match="input contract mismatch"):
        compare_episodes(
            _report("capture"),
            _control_steps(),
            _report("skip", provenance=_provenance("skip", finger="f2")),
            _control_steps(),
        )
    with pytest.raises(ValueError, match="input contract mismatch"):
        compare_episodes(
            _report(
                "capture",
                provenance=_provenance("capture", payload_sha256="abc"),
            ),
            _control_steps(),
            _report("skip"),
            _control_steps(),
        )
    with pytest.raises(ValueError, match="input contract mismatch"):
        compare_episodes(
            _report(
                "capture",
                provenance=_provenance(
                    "capture", payload_sha256=SHA.upper()
                ),
            ),
            _control_steps(),
            _report("skip"),
            _control_steps(),
        )
    with pytest.raises(ValueError, match="input contract mismatch"):
        compare_episodes(
            _report(
                "capture",
                provenance=_provenance("capture", seed=True),
            ),
            _control_steps(),
            _report("skip"),
            _control_steps(),
        )
    with pytest.raises(ValueError, match="input contract mismatch"):
        compare_episodes(
            _report("capture", seed=7),
            _control_steps(),
            _report("skip"),
            _control_steps(),
        )
    with pytest.raises(ValueError, match="input contract mismatch"):
        compare_episodes(
            _report("capture", gui=True),
            _control_steps(),
            _report("skip", gui=False),
            _control_steps(),
        )


def test_verify_markers_pre_approach_must_precede_first_control_step():
    capture_points = _audit_points(True)
    skip_points = _audit_points(False)
    capture_points["pre_approach"]["global_step"] = 10
    skip_points["pre_approach"]["global_step"] = 10
    capture_report = _report("capture", points=capture_points)
    skip_report = _report("skip", points=skip_points)
    problems, _ = verify_markers(
        capture_report, skip_report, _control_steps()
    )
    assert "marker_pre_approach_not_first" in problems


def test_canonical_sha256_is_deterministic():
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256(
        {"b": 2, "a": 1}
    )
    assert canonical_sha256({"a": 1}) != canonical_sha256({"a": 2})


def _write_episode(directory, mode, steps=None):
    directory.mkdir(parents=True)
    report = _report(mode)
    (directory / "nominal_physics_report.json").write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )
    with (directory / "controller_steps.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for step in steps if steps is not None else _control_steps():
            stream.write(json.dumps(step, sort_keys=True) + "\n")


def _comparator_env():
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    return environment


def _run_comparator(capture_dir, skip_dir, output_dir):
    return subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "isaac"
                / "single_finger_posthoc_audit_compare.py"
            ),
            "--capture-dir",
            str(capture_dir),
            "--skip-dir",
            str(skip_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_comparator_env(),
    )


def test_comparator_cli_happy_path_and_no_episode_rewrite(tmp_path):
    capture_dir = tmp_path / "capture"
    skip_dir = tmp_path / "skip"
    out_dir = tmp_path / "validation"
    _write_episode(capture_dir, "capture")
    _write_episode(skip_dir, "skip")
    before_capture = (
        capture_dir / "nominal_physics_report.json"
    ).read_bytes()
    before_skip = (skip_dir / "nominal_physics_report.json").read_bytes()
    result = _run_comparator(capture_dir, skip_dir, out_dir)
    assert result.returncode == 0
    summary = json.loads(
        (out_dir / "posthoc_audit_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["single_finger_validation_passed"] is True
    assert (capture_dir / "nominal_physics_report.json").read_bytes() == (
        before_capture
    )
    assert (skip_dir / "nominal_physics_report.json").read_bytes() == (
        before_skip
    )


def test_comparator_cli_gate_failure_exit_one(tmp_path):
    capture_dir = tmp_path / "capture"
    skip_dir = tmp_path / "skip"
    out_dir = tmp_path / "validation"
    _write_episode(skip_dir, "skip")
    capture_dir.mkdir()
    points = _audit_points(True)
    points["contact_confirmed"]["snapshot"] = _snapshot(selected_body=0)
    report = _report("capture", points=points)
    (capture_dir / "nominal_physics_report.json").write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )
    with (capture_dir / "controller_steps.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for step in _control_steps():
            stream.write(json.dumps(step, sort_keys=True) + "\n")
    result = _run_comparator(capture_dir, skip_dir, out_dir)
    assert result.returncode == 1


def test_comparator_cli_output_nested_or_nonempty_refused(tmp_path):
    capture_dir = tmp_path / "capture"
    skip_dir = tmp_path / "skip"
    _write_episode(capture_dir, "capture")
    _write_episode(skip_dir, "skip")
    nested = capture_dir / "nested"
    result = _run_comparator(capture_dir, skip_dir, nested)
    assert result.returncode == 2
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "junk.txt").write_text("x", encoding="utf-8")
    result = _run_comparator(capture_dir, skip_dir, nonempty)
    assert result.returncode == 2
    assert "non-empty" in result.stdout


def test_review_counterexample_skip_contains_snapshots():
    # skip must never carry truth payloads: snapshot must be None.
    summary = compare_episodes(
        _report("capture"),
        _control_steps(),
        _report("skip", points=_audit_points(True)),
        _control_steps(),
    )
    assert summary["exit_code"] == 3
    assert any(
        "snapshot_not_none" in problem
        for problem in summary["marker_problems"]
    )


def test_review_counterexample_extra_marker():
    points = _audit_points(True)
    points["bogus_point"] = {
        **_marker("bogus_point", 99, "APPROACH", 0, 0),
        "snapshot": None,
    }
    summary = compare_episodes(
        _report("capture", points=points),
        _control_steps(),
        _report("skip"),
        _control_steps(),
    )
    assert summary["exit_code"] == 3
    assert "capture_marker_set_not_exact" in summary["marker_problems"]


def test_review_counterexample_marker_wrong_finger():
    capture_points = _audit_points(True)
    skip_points = _audit_points(False)
    capture_points["contact_confirmed"]["selected_finger"] = "f2"
    skip_points["contact_confirmed"]["selected_finger"] = "f2"
    summary = compare_episodes(
        _report("capture", points=capture_points),
        _control_steps(),
        _report("skip", points=skip_points),
        _control_steps(),
    )
    assert summary["exit_code"] == 3
    assert any(
        "marker_wrong_finger" in problem
        for problem in summary["marker_problems"]
    )


def test_review_counterexample_marker_trace_step_misaligned():
    capture_points = _audit_points(True)
    skip_points = _audit_points(False)
    # Confirm marker claims soft_hold_step 1 while the controller record at
    # the same global step has 0.
    capture_points["contact_confirmed"]["soft_hold_step"] = 1
    skip_points["contact_confirmed"]["soft_hold_step"] = 1
    summary = compare_episodes(
        _report("capture", points=capture_points),
        _control_steps(),
        _report("skip", points=skip_points),
        _control_steps(),
    )
    assert summary["exit_code"] == 3
    assert any(
        "trace_soft_hold_misaligned" in problem
        for problem in summary["marker_problems"]
    )


def test_review_counterexample_material_count_inconsistent():
    points = _audit_points(True)
    inconsistent = {
        "available": True,
        "resolved_records": 3,
        "unresolved_records": 0,
        "grip_grip_records": 3,
        "resolved_non_grip_records": 0,
    }
    # selected contact total is 2 but material claims 3 resolved records.
    points["contact_confirmed"]["snapshot"] = _snapshot(
        selected_body=2, material=inconsistent
    )
    summary = compare_episodes(
        _report("capture", points=points),
        _control_steps(),
        _report("skip"),
        _control_steps(),
    )
    assert summary["exit_code"] == 1
    assert summary["failure_reason"] == "contact_gate_failed"
    assert not summary["audit_points"]["contact_confirmed"]["gates"][
        "material_evidence_consistent"
    ]


def test_review_counterexample_missing_per_step_torque_channel():
    capture_steps = []
    skip_steps = []
    for step in _control_steps():
        capture_record = dict(step)
        capture_record["finger_root_torque_proxy_nm"] = {
            "f1": 0.1,
            "f2": 0.0,
        }
        capture_steps.append(capture_record)
        skip_record = dict(step)
        skip_record["finger_root_torque_proxy_nm"] = {
            "f1": 0.1,
            "f2": 0.0,
        }
        skip_steps.append(skip_record)
    summary = compare_episodes(
        _report("capture"), capture_steps, _report("skip"), skip_steps
    )
    assert summary["exit_code"] == 3
    assert any(
        "torque_channels_wrong" in problem
        for problem in summary["schema_problems"]
    )


def test_review_counterexample_nested_skip_contract_wrong():
    skip_report = _report("skip")
    skip_report["posthoc_audit"]["mode"] = "capture"
    summary = compare_episodes(
        _report("capture"),
        _control_steps(),
        skip_report,
        _control_steps(),
    )
    assert summary["exit_code"] == 3
    assert "posthoc_audit_mode_wrong" in summary["skip_contract_problems"]


def test_schema_nan_torque_value_rejected():
    steps = []
    for step in _control_steps():
        record = dict(step)
        if record["state"] == "APPROACH":
            record["finger_root_torque_proxy_nm"] = {
                "f1": float("nan"),
                "f2": 0.0,
                "f3": 0.0,
            }
        steps.append(record)
    summary = compare_episodes(
        _report("capture"), steps, _report("skip"), _control_steps()
    )
    assert summary["exit_code"] == 3
    assert any(
        "torque_values_invalid" in problem
        for problem in summary["schema_problems"]
    )


def test_schema_wrong_hand_q_length_rejected():
    steps = []
    for step in _control_steps():
        record = dict(step)
        record["hand_q_rad"] = [1.0, 0.4, 0.0]
        steps.append(record)
    summary = compare_episodes(
        _report("capture"), _control_steps(), _report("skip"), steps
    )
    assert summary["exit_code"] == 3
    assert any(
        "hand_q_rad_invalid" in problem
        for problem in summary["schema_problems"]
    )


def test_schema_missing_observation_rejected():
    steps = []
    for step in _control_steps():
        record = dict(step)
        record.pop("observation")
        steps.append(record)
    summary = compare_episodes(
        _report("capture"), _control_steps(), _report("skip"), steps
    )
    assert summary["exit_code"] == 3
    assert any(
        "observation_missing" in problem
        for problem in summary["schema_problems"]
    )


def test_skip_marker_with_error_payload_rejected():
    skip_points = _audit_points(False)
    skip_points["contact_confirmed"]["error"] = "boom"
    summary = compare_episodes(
        _report("capture"),
        _control_steps(),
        _report("skip", points=skip_points),
        _control_steps(),
    )
    assert summary["exit_code"] == 3
    assert any(
        "unexpected_error" in problem
        for problem in summary["marker_problems"]
    )


def test_capture_marker_missing_snapshot_fails_gate_not_pass():
    capture_points = _audit_points(True)
    capture_points["contact_confirmed"]["snapshot"] = None
    capture_points["contact_confirmed"]["error"] = "query_failed"
    summary = compare_episodes(
        _report("capture", points=capture_points),
        _control_steps(),
        _report("skip"),
        _control_steps(),
    )
    # Marker itself is legal (explicit error), but the point must fail its
    # contact gate, never pass.
    assert summary["exit_code"] == 1
    assert not summary["audit_points"]["contact_confirmed"]["passed"]


def test_capture_episode_marker_helper_accepts_clean_episode():
    problems, evidence = verify_capture_episode_markers(
        _report("capture"), _control_steps(), "capture"
    )
    assert problems == []
    assert set(evidence) == {
        "pre_approach",
        "contact_confirmed",
        "soft_hold_complete",
        "release_confirmed",
    }


def test_capture_episode_marker_helper_rejects_extra_key():
    report = _report("capture")
    marker = report["posthoc_audit"]["points"]["contact_confirmed"]
    marker["bogus_field"] = 1
    problems, _ = verify_capture_episode_markers(
        report, _control_steps(), "capture"
    )
    assert any("marker_extra_keys" in problem for problem in problems)


def test_capture_episode_marker_helper_rejects_snapshot_with_error():
    report = _report("capture")
    marker = report["posthoc_audit"]["points"]["contact_confirmed"]
    marker["error"] = "boom"
    problems, _ = verify_capture_episode_markers(
        report, _control_steps(), "capture"
    )
    assert any(
        "snapshot_with_error" in problem for problem in problems
    )


def test_capture_episode_marker_helper_rejects_non_mapping_snapshot():
    report = _report("capture")
    marker = report["posthoc_audit"]["points"]["contact_confirmed"]
    marker["snapshot"] = None
    problems, _ = verify_capture_episode_markers(
        report, _control_steps(), "capture"
    )
    assert any(
        "snapshot_not_mapping" in problem for problem in problems
    )


def test_capture_episode_marker_helper_rejects_bool_hold_step():
    report = _report("capture")
    marker = report["posthoc_audit"]["points"]["contact_confirmed"]
    marker["soft_hold_step"] = False
    problems, _ = verify_capture_episode_markers(
        report, _control_steps(), "capture"
    )
    assert any("soft_hold_step_type" in problem for problem in problems)


def test_capture_episode_marker_helper_rejects_duplicate_steps():
    report = _report("capture")
    points = report["posthoc_audit"]["points"]
    points["soft_hold_complete"]["global_step"] = points[
        "contact_confirmed"
    ]["global_step"]
    problems, _ = verify_capture_episode_markers(
        report, _control_steps(), "capture"
    )
    assert any("not_increasing" in problem for problem in problems)



def test_classified_marker_helper_separates_schema_and_functional():
    report = _report("capture")
    marker = report["posthoc_audit"]["points"]["contact_confirmed"]
    marker["bogus_field"] = 1
    schema_problems, functional_problems, evidence = (
        verify_capture_episode_markers_classified(
            report, _control_steps(), "capture"
        )
    )
    assert any("marker_extra_keys" in p for p in schema_problems)
    assert functional_problems == []
    assert set(evidence) == {"pre_approach", "contact_confirmed", "soft_hold_complete", "release_confirmed"}


def test_classified_marker_helper_trace_misalignment_is_functional():
    report = _report("capture")
    marker = report["posthoc_audit"]["points"]["contact_confirmed"]
    marker["global_step"] = 50
    schema_problems, functional_problems, _ = (
        verify_capture_episode_markers_classified(
            report, _control_steps(), "capture"
        )
    )
    assert schema_problems == []
    assert any(
        "trace_state_misaligned" in p for p in functional_problems
    )


def test_classified_marker_helper_frozen_value_error_is_functional():
    report = _report("capture")
    marker = report["posthoc_audit"]["points"]["contact_confirmed"]
    marker["soft_hold_step"] = 5
    schema_problems, functional_problems, _ = (
        verify_capture_episode_markers_classified(
            report, _control_steps(), "capture"
        )
    )
    assert schema_problems == []
    assert any(
        "marker_contact_confirmed_state" in p
        for p in functional_problems
    )


def test_compat_entry_flat_list_contains_both_categories():
    # A schema-invalid marker contributes only its precise schema problems
    # (the value checker is skipped for it), while a schema-valid marker
    # with a frozen-value error contributes functional problems.
    report = _report("capture")
    marker = report["posthoc_audit"]["points"]["contact_confirmed"]
    marker["bogus_field"] = 1
    problems, evidence = verify_capture_episode_markers(
        report, _control_steps(), "capture"
    )
    assert any("marker_extra_keys" in p for p in problems)
    assert not any("marker_contact_confirmed_state" in p for p in problems)
    assert set(evidence) == {
        "pre_approach",
        "contact_confirmed",
        "soft_hold_complete",
        "release_confirmed",
    }
    del marker["bogus_field"]
    marker["soft_hold_step"] = 5
    problems, _ = verify_capture_episode_markers(
        report, _control_steps(), "capture"
    )
    assert not any("marker_extra_keys" in p for p in problems)
    assert any("marker_contact_confirmed_state" in p for p in problems)



@pytest.mark.parametrize(
    "point, field, value",
    [
        ("release_confirmed", "release_step", "bad"),
        ("release_confirmed", "release_step", False),
        ("contact_confirmed", "global_step", "bad"),
        ("contact_confirmed", "soft_hold_step", "bad"),
        ("contact_confirmed", "controller_state", 5),
        ("contact_confirmed", "snapshot", None),
    ],
)
def test_classified_marker_helper_invalid_marker_is_schema_only_no_raise(
    point, field, value
):
    report = _report("capture")
    marker = report["posthoc_audit"]["points"][point]
    marker[field] = value
    schema_problems, functional_problems, evidence = (
        verify_capture_episode_markers_classified(
            report, _control_steps(), "capture"
        )
    )
    assert schema_problems, "invalid marker must produce schema problems"
    assert functional_problems == []
    assert set(evidence) == {
        "pre_approach",
        "contact_confirmed",
        "soft_hold_complete",
        "release_confirmed",
    }

