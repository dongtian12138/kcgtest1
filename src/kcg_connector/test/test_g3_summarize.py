'''Targeted tests for the G3 posthoc collection summarizer CLI.

The CLI is a fail-closed acceptance verifier and statistics summarizer
for the final five-headless + one-GUI synchronous staged grasp
collection.  These tests exercise the good collection, missing/duplicate
seeds, hash mismatch, truth/proxy violations, false PASS, missing GPU
log markers, GUI payload/hash mismatch, units, the n=5 linear P95, and
failure-reason preservation (never filtered or overwritten).
'''

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
CLI = (
    REPOSITORY
    / "src/kcg_connector/isaac/d38999_physical_grasp_g3_summarize.py"
)
SOURCE_ROOT = str(REPOSITORY / "src" / "kcg_connector")

GPU_LOG_TEXT = (
    "Active GPU: NVIDIA GeForce RTX 5070 Ti\n"
    "| 0 | NVIDIA GeForce RTX 5070 Ti | Yes: 0 |\n"
    'some warp evidence "cuda:0"\n'
)

OTHER_PROVENANCE_KEYS = {
    "pick_config_sha256": "d" * 64,
    "tabletop_scene_config_sha256": "e" * 64,
    "wrapper_sha256": "f" * 64,
    "finger_contact_detector_sha256": "a" * 64,
    "single_finger_contact_test_sha256": "b" * 64,
    "single_finger_posthoc_audit_sha256": "c" * 64,
    "single_finger_posthoc_audit_compare_sha256": "d" * 64,
}


def _disk_hashes() -> dict[str, str]:
    def sha(relative: str) -> str:
        return hashlib.sha256(
            (REPOSITORY / relative).read_bytes()
        ).hexdigest()

    return {
        "runner_sha256": sha(
            "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"
        ),
        "grasp_stability_monitor_sha256": sha(
            "src/kcg_connector/kcg_connector/grasp/"
            "grasp_stability_monitor.py"
        ),
        "physical_grasp_config_loader_sha256": sha(
            "src/kcg_connector/kcg_connector/grasp/physical_grasp_config.py"
        ),
        "physical_grasp_config_sha256": sha(
            "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
        ),
    }


def _payload(seed: int) -> str:
    return hashlib.sha256(f"payload-{seed}".encode()).hexdigest()


def _make_report(seed: int, *, gui: bool = False, payload=None, **overrides):
    if payload is None:
        payload = _payload(seed)
    provenance = {
        "seed": seed,
        "finger": None,
        "audit_mode": None,
        "payload_sha256": payload,
        **_disk_hashes(),
        **OTHER_PROVENANCE_KEYS,
    }
    report = {
        "seed": seed,
        "gui": gui,
        "physical_grasp_method": "synchronous",
        "formal_lift_mode": "staged",
        "process_exit_code": 0,
        "passed": True,
        "realized_randomization": {
            "mode": "staged",
            "seed": seed,
            "payload_sha256": payload,
        },
        "formal_acceptance": {
            "passed": True,
            "sensor_lift_gate": True,
            "episode_end_contact_gate": True,
            "controller_stable": True,
            "post_grasp_stabilization_proxy_used": False,
            "minimum_total_lift_m": 0.045,
            "actual_body_lift_m": 0.052,
            "recovery_record": None,
        },
        "formal_lift_stages": [
            {
                "stage": 1,
                "increment_m": 0.002,
                "passed_sensor_gate": True,
            },
            {
                "stage": 2,
                "increment_m": 0.010,
                "passed_sensor_gate": True,
            },
            {
                "stage": 3,
                "increment_m": 0.040,
                "passed_sensor_gate": True,
            },
        ],
        "formal_lift_monitor": {
            "failed": False,
            "failure_reason": None,
            "steps": 2972,
            "peak_wrist_force_increment_n": 2.64,
            "peak_moment_safety_score_nm": 0.086,
        },
        "finite_throughout": True,
        "finite_final": True,
        "final_tail_diagnostics_finite": True,
        "final_all_fingers_body_contact": True,
        "zero_forbidden_contacts": True,
        "final_unsupported": False,
        "final_contacts": {
            "plug_table_records": 0,
            "unexpected_robot_link_records": 0,
            "finger_body_group_records": {
                "f1": {"body": 1, "nut": 4},
                "f2": {"body": 3, "nut": 2},
                "f3": {"body": 1, "nut": 5},
            },
            "material_evidence": {
                "available": True,
                "grip_grip_records": 16,
                "resolved_records": 16,
                "unresolved_records": 0,
            },
            "finger_loose_plug_records": 16,
        },
        "external_contact_records": {
            "fixed_endpoint": 0,
            "fixture": 0,
            "loose_plug_allowed": 0,
            "loose_plug_preclosure": 0,
            "loose_plug_unexpected_robot_link": 0,
            "table": 0,
        },
        "control_reads_object_truth": False,
        "control_reads_contact_report": False,
        "truth_orientation_used": False,
        "object_pose_writes_after_start": 0,
        "attachment": "none",
        "object_drive": "none",
        "proxy_collision_filter": {
            "enabled": False,
            "mode": "none",
            "pair_count": 0,
        },
        "posthoc_truth_evaluation_only": True,
        "formal_truth_firewall_enabled": True,
        "provenance": provenance,
        "table_stage": {
            "translation_xy_m": 2.6e-05 + seed * 1.0e-06,
            "yaw_delta_rad": -0.001 + seed * 0.0001,
        },
        "body_lift_m": 0.051 + seed * 0.001,
        "body_tcp_slip_m": 0.0001 + seed * 1.0e-05,
        "body_nut_separation_change_m": 1.0e-06 + seed * 1.0e-07,
        "posthoc_pose_error": {
            "dx_m": 0.0005 + seed * 0.0001,
            "dy_m": -0.0003 + seed * 1.0e-05,
            "dz_m": 0.0013,
            "drx_rad": 0.003,
            "dry_rad": -0.001,
            "drz_rad": 0.05 + seed * 0.005,
        },
        "posthoc_lift_relative_slip": {
            "dx_m": 3.0e-05,
            "dy_m": -0.0001,
            "dz_m": 4.0e-06,
            "drx_rad": 0.003,
            "dry_rad": 0.001,
            "drz_rad": -0.009,
        },
    }
    for key, value in overrides.items():
        if (
            key in report
            and isinstance(report[key], dict)
            and isinstance(value, dict)
        ):
            report[key].update(value)
        else:
            report[key] = value
    return report


def _write_episode(
    directory: Path,
    seed: int,
    *,
    gui: bool = False,
    payload=None,
    order=("f2", "f3", "f1"),
    **overrides,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    report = _make_report(seed, gui=gui, payload=payload, **overrides)
    (directory / "nominal_physics_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    records = []
    for index, _finger in enumerate(order):
        records.append(
            json.dumps(
                {
                    "global_step": 9000 + index,
                    "contact_order": list(order),
                    "phase": "physical_hand_closure",
                    "states": {
                        finger: (
                            "CONTACT_CONFIRMED"
                            if order.index(finger) <= index
                            else "APPROACH"
                        )
                        for finger in ("f1", "f2", "f3")
                    },
                }
            )
        )
    (directory / "controller_steps.jsonl").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )


def _write_logs(
    logs_dir: Path,
    seeds=(0, 1, 2, 3, 4),
    *,
    gui: bool = True,
    headless_text: str = GPU_LOG_TEXT,
    gui_text: str = GPU_LOG_TEXT,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        (logs_dir / f"seed{seed:03d}_headless_kit.log").write_text(
            headless_text, encoding="utf-8"
        )
    if gui:
        (logs_dir / "seed000_gui_kit.log").write_text(
            gui_text, encoding="utf-8"
        )


def _run_cli(headless_dirs, gui_dir, logs_dir, output_dir):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = SOURCE_ROOT + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    command = [sys.executable, str(CLI)]
    for directory in headless_dirs:
        command += ["--headless", str(directory)]
    command += [
        "--gui",
        str(gui_dir),
        "--runtime-logs",
        str(logs_dir),
        "--output",
        str(output_dir),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def _make_collection(tmp_path, seeds=(0, 1, 2, 3, 4), **kwargs):
    headless = []
    payloads = {seed: _payload(seed) for seed in seeds}
    for seed in seeds:
        directory = tmp_path / f"headless_{seed}"
        _write_episode(
            directory,
            seed,
            payload=payloads[seed],
            **kwargs.get(f"seed{seed}", {}),
        )
        headless.append(directory)
    gui_dir = tmp_path / "gui_0"
    _write_episode(
        gui_dir,
        0,
        gui=True,
        payload=payloads.get(0),
        **kwargs.get("gui", {}),
    )
    logs = tmp_path / "logs"
    _write_logs(logs, seeds=seeds, **kwargs.get("logs", {}))
    output = tmp_path / "summary_out"
    return headless, gui_dir, logs, output


def test_good_collection_exits_zero_and_writes_outputs(tmp_path):
    headless, gui_dir, logs, output = _make_collection(tmp_path)
    code, stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code == 0, stderr
    summary_path = output / "summary.json"
    manifest_path = output / "input_manifest.json"
    assert summary_path.is_file()
    assert manifest_path.is_file()
    assert (output / "SUMMARY_CN.md").is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["g3_complete"] is True
    assert summary["verdicts"]["headless_accepted"] == 5
    assert summary["verdicts"]["gui_accepted"] is True
    assert summary["statistics"]["episode_count"] == 5
    assert summary["statistics"]["accepted_count"] == 5
    assert summary["statistics"]["first_finger_distribution"] == {"f2": 5}
    assert summary["statistics"]["contact_order_distribution"] == {
        "f2-f3-f1": 5
    }
    assert summary["source_hashes"]["identical_across_episodes"] is True
    gui_functional = summary["gui_vs_headless_seed0"]["functional"]
    assert gui_functional["payload_equal"] is True
    assert gui_functional["provenance_equal"] is True
    assert summary["p95_n_equals_5_preliminary"] is True
    assert summary["continuous_contact_path_verified"] is False
    recomputed = hashlib.sha256(
        json.dumps(
            {"inputs": manifest["inputs"]}, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["manifest_content_sha256"] == recomputed
    assert summary["input_manifest_sha256"] == recomputed
    assert "5/5" in stdout or "accepted" in stdout


def test_missing_seed_rejected(tmp_path):
    headless, gui_dir, logs, output = _make_collection(tmp_path)
    path = tmp_path / "headless_4" / "nominal_physics_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["seed"] = 5
    path.write_text(json.dumps(report), encoding="utf-8")
    (logs / "seed005_headless_kit.log").write_text(
        GPU_LOG_TEXT, encoding="utf-8"
    )
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code != 0
    assert "0-4" in stderr


def test_duplicate_seed_rejected(tmp_path):
    headless, gui_dir, logs, output = _make_collection(
        tmp_path, seeds=(0, 1, 2, 3, 4)
    )
    headless[4] = headless[2]
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code != 0
    assert "distinct" in stderr


def test_hash_mismatch_rejected(tmp_path):
    headless, gui_dir, logs, output = _make_collection(
        tmp_path,
        seed2={
            "provenance": {"runner_sha256": "f" * 64},
        },
    )
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code == 1
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["g3_complete"] is False
    reasons = summary["verdicts"]["failure_reasons"]
    assert "seed2" in reasons
    assert any(
        "runner_sha256" in problem
        for entry in reasons["seed2"]
        for problem in entry["verification_problems"]
    )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (
            {"control_reads_object_truth": True},
            "control reads object truth",
        ),
        (
            {"proxy_collision_filter": {"enabled": True}},
            "proxy collision filter must be disabled",
        ),
    ],
)
def test_truth_or_proxy_violation_rejected(tmp_path, overrides, expected):
    headless, gui_dir, logs, output = _make_collection(
        tmp_path, seed3=overrides
    )
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code == 1
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["g3_complete"] is False
    problems = summary["verdicts"]["failure_reasons"]["seed3"][0][
        "verification_problems"
    ]
    assert expected in problems


@pytest.mark.parametrize(
    "overrides",
    [
        {"formal_acceptance": {"sensor_lift_gate": False}},
        {"formal_acceptance": {"actual_body_lift_m": 0.044}},
        {"formal_acceptance": {"episode_end_contact_gate": False}},
        {
            "passed": True,
            "process_exit_code": 1,
            "formal_lift_monitor": {
                "failed": True,
                "failure_reason": "wrist_moment_limit",
            },
        },
    ],
)
def test_false_pass_rejected(tmp_path, overrides):
    headless, gui_dir, logs, output = _make_collection(
        tmp_path, seed1=overrides
    )
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code == 1
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["g3_complete"] is False
    assert summary["verdicts"]["headless_accepted"] == 4
    assert summary["verdicts"]["headless_rejected"] == 1


def test_failure_reasons_not_filtered_or_overwritten(tmp_path):
    headless, gui_dir, logs, output = _make_collection(
        tmp_path,
        seed4={
            "passed": True,
            "process_exit_code": 1,
            "formal_lift_monitor": {
                "failed": True,
                "failure_reason": "wrist_moment_limit",
            },
        },
    )
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code == 1
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    reasons = summary["verdicts"]["failure_reasons"]
    assert reasons["seed4"][0]["failure_reason"] == "wrist_moment_limit"
    assert "wrist_moment_limit" in (output / "SUMMARY_CN.md").read_text(
        encoding="utf-8"
    )


def test_gpu_log_missing_marker_rejected(tmp_path):
    headless, gui_dir, logs, output = _make_collection(tmp_path)
    # remove the Yes: 0 marker from seed2's log
    (logs / "seed002_headless_kit.log").write_text(
        "Active GPU: NVIDIA GeForce RTX 5070 Ti\n" 'warp "cuda:0"\n',
        encoding="utf-8",
    )
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code == 1
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    problems = summary["verdicts"]["failure_reasons"]["seed2"][0][
        "verification_problems"
    ]
    assert any("Yes: 0" in problem for problem in problems)


def test_gui_payload_mismatch_rejected(tmp_path):
    headless, gui_dir, logs, output = _make_collection(tmp_path)
    _write_episode(gui_dir, 0, gui=True, payload=_payload(99))
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code == 1
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["verdicts"]["gui_accepted"] is False
    problems = summary["verdicts"]["gui_failure_reasons"]["seed0_gui"]
    assert any("payload_sha256" in problem for problem in problems)


def test_gui_hash_mismatch_rejected(tmp_path):
    headless, gui_dir, logs, output = _make_collection(tmp_path)
    report = json.loads(
        (gui_dir / "nominal_physics_report.json").read_text(encoding="utf-8")
    )
    report["provenance"]["wrapper_sha256"] = "9" * 64
    (gui_dir / "nominal_physics_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code == 1
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    problems = summary["verdicts"]["gui_failure_reasons"]["seed0_gui"]
    assert any("wrapper_sha256" in problem for problem in problems)


def test_units_and_n5_linear_p95(tmp_path):
    lift_values = [0.051, 0.052, 0.053, 0.054, 0.055]
    headless, gui_dir, logs, output = _make_collection(tmp_path)
    # craft body_lift_m per seed after writing, then rewrite reports
    for seed, lift in zip((0, 1, 2, 3, 4), lift_values):
        path = tmp_path / f"headless_{seed}" / "nominal_physics_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["body_lift_m"] = lift
        path.write_text(json.dumps(report), encoding="utf-8")
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, output)
    assert code == 0, stderr
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    block = summary["statistics"]["body_lift"]
    expected_p95 = float(np.percentile(np.asarray(lift_values), 95))
    assert block["signed"]["p95"] == pytest.approx(expected_p95)
    assert block["unit"] == "m"
    assert block["readable_unit"] == "mm"
    assert block["signed_readable"]["mean"] == pytest.approx(
        block["signed"]["mean"] * 1000.0
    )
    pose = summary["statistics"]["posthoc_pose_error"]
    assert pose["dx_m"]["unit"] == "m"
    assert pose["dx_m"]["readable_unit"] == "mm"
    assert pose["drx_rad"]["unit"] == "rad"
    assert pose["drx_rad"]["readable_unit"] == "deg"
    assert pose["drx_rad"]["absolute_readable"]["maximum"] == pytest.approx(
        pose["drx_rad"]["absolute"]["maximum"] * 180.0 / np.pi
    )
    assert summary["statistics"]["episode_count"] == 5
    assert summary["percentile_method"] == (
        "numpy_linear_default_np_percentile_95"
    )


def test_output_dir_must_not_overwrite_input(tmp_path):
    headless, gui_dir, logs, output = _make_collection(tmp_path)
    code, _stdout, stderr = _run_cli(headless, gui_dir, logs, headless[0])
    assert code != 0
    assert "overwrite" in stderr
