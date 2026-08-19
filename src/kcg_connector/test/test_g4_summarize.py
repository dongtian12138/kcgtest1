'''Targeted tests for the offline G4 summarizer CLI (035 contracts).'''

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
    / "src/kcg_connector/isaac/d38999_physical_grasp_g4_summarize.py"
)
SOURCE_ROOT = str(REPOSITORY / "src" / "kcg_connector")

GPU_LOG = (
    "Active GPU: NVIDIA GeForce RTX 5070 Ti\n"
    "| 0 | NVIDIA GeForce RTX 5070 Ti | Yes: 0 |\n"
    'warp "cuda:0"\n'
    "PhysXFoundation: 1 CUDA device(s) available, selecting device 0\n"
)

FINGERS = ("f1", "f2", "f3")
WRIST = [0.3, -0.1, 19.0, 0.1, 0.2, 0.01]


def _disk_hashes() -> dict[str, str]:
    def sha(relative: str) -> str:
        return hashlib.sha256(
            (REPOSITORY / relative).read_bytes()
        ).hexdigest()

    return {
        "runner_sha256": sha(
            "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"
        ),
        "wrapper_sha256": sha(
            "src/kcg_connector/isaac/d38999_tabletop_physical_grasp_v1.py"
        ),
        "three_finger_sequential_grasp_sha256": sha(
            "src/kcg_connector/kcg_connector/grasp/"
            "three_finger_sequential_grasp.py"
        ),
        "finger_contact_detector_sha256": sha(
            "src/kcg_connector/kcg_connector/grasp/finger_contact_detector.py"
        ),
        "grasp_stability_monitor_sha256": sha(
            "src/kcg_connector/kcg_connector/grasp/grasp_stability_monitor.py"
        ),
        "physical_grasp_config_loader_sha256": sha(
            "src/kcg_connector/kcg_connector/grasp/physical_grasp_config.py"
        ),
        "physical_grasp_config_sha256": sha(
            "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
        ),
        "pick_config_sha256": sha(
            "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
        ),
        "tabletop_scene_config_sha256": sha(
            "src/kcg_connector/config/d38999_tabletop_scene_v1.yaml"
        ),
    }


def _payload(seed: int) -> str:
    return hashlib.sha256(f"payload-{seed}".encode()).hexdigest()


def _base_report(seed: int, passed: bool) -> dict:
    payload = _payload(seed)
    return {
        "seed": seed,
        "gui": False,
        "physical_grasp_method": "sequential-compliant",
        "formal_lift_mode": "staged",
        "process_exit_code": 0 if passed else 1,
        "passed": passed,
        "error": None if passed else "sequential lift failed closed",
        "provenance": {
            "seed": seed,
            "finger": None,
            "audit_mode": None,
            "payload_sha256": payload,
            **_disk_hashes(),
            "single_finger_contact_test_sha256": "a" * 64,
            "single_finger_posthoc_audit_sha256": "b" * 64,
            "single_finger_posthoc_audit_compare_sha256": "c" * 64,
        },
        "realized_randomization": {
            "seed": seed,
            "payload_sha256": payload,
            "mode": "staged",
        },
        "control_reads_object_truth": False,
        "control_reads_contact_report": False,
        "truth_orientation_used": False,
        "object_pose_writes_after_start": 0,
        "attachment": "none",
        "object_drive": "none",
        "proxy_collision_filter": {"enabled": False},
        "posthoc_truth_evaluation_only": True,
    }


def _success_report(seed: int, **overrides) -> dict:
    report = _base_report(seed, True)
    report.update(
        {
            "formal_acceptance": {
                "passed": True,
                "sensor_lift_gate": True,
                "episode_end_contact_gate": True,
                "controller_stable": True,
                "post_grasp_stabilization_proxy_used": False,
                "actual_body_lift_m": 0.052,
                "minimum_total_lift_m": 0.045,
            },
            "finite_throughout": True,
            "finite_final": True,
            "final_tail_diagnostics_finite": True,
            "final_all_fingers_body_contact": True,
            "zero_forbidden_contacts": True,
            "final_unsupported": False,
            "final_contacts": {
                "finger_body_group_records": {
                    "f1": {"body": 1, "nut": 4},
                    "f2": {"body": 3, "nut": 2},
                    "f3": {"body": 1, "nut": 5},
                },
                "plug_table_records": 0,
                "unexpected_robot_link_records": 0,
                "material_evidence": {
                    "available": True,
                    "grip_grip_records": 16,
                    "resolved_records": 16,
                    "unresolved_records": 0,
                },
            },
            "formal_lift_stages": [
                {"stage": 1, "increment_m": 0.002, "passed_sensor_gate": True},
                {"stage": 2, "increment_m": 0.010, "passed_sensor_gate": True},
                {"stage": 3, "increment_m": 0.040, "passed_sensor_gate": True},
            ],
            "formal_lift_monitor": {
                "failed": False,
                "failure_reason": None,
                "steps": 2972,
                "force_gate_n": 8.0,
                "moment_gate_nm": 0.30,
                "peak_wrist_force_increment_n": 2.5,
                "peak_moment_safety_score_nm": 0.15,
                "moment_trigger_component": None,
            },
            "grasp_controller": {
                "contact_order": ["f2", "f3", "f1"],
                "contact_global_steps": {
                    "f1": 9371,
                    "f2": 9149,
                    "f3": 9370,
                },
                "failure_reason": None,
                "method": "sequential-compliant",
                "stable": True,
            },
            "sequential_consolidation": {
                "completed": True,
                "lift_ready": True,
                "targets_frozen_exact": True,
                "applied_scale_min": 0.35,
                "applied_scale_max": 1.0,
                "commanded_scale_monotonic": True,
                "ramp_steps": 120,
                "window_steps": 240,
                "final_window_sample_count": 240,
                "final_stiffness_scale": 1.0,
                "soft_hold_stiffness_scale": 0.35,
                "first_window_sample_applied_scale": 1.0,
                "final_root_reference_nm": [0.25, 0.26, 0.25],
                "final_wrist_reference": list(WRIST),
            },
            "body_tcp_slip_m": 0.0002,
            "final_hold_displacement_m": 0.000001,
            "table_stage": {
                "translation_xy_m": 0.000004,
                "yaw_delta_rad": -0.0001,
            },
            "posthoc_pose_error": {
                "dx_m": 0.001,
                "dy_m": -0.0005,
                "dz_m": 0.0013,
                "drx_rad": 0.003,
                "dry_rad": -0.001,
                "drz_rad": 0.05,
            },
            "posthoc_lift_relative_slip": {
                "dx_m": 0.00003,
                "dy_m": -0.0001,
                "dz_m": 0.000004,
                "drx_rad": 0.003,
                "dry_rad": 0.001,
                "drz_rad": -0.009,
            },
            "final_tail_observable_angular_speed_rad_s": {
                "nut_relative_to_body": {
                    "mean": 0.001,
                    "median": 0.0006,
                    "rms": 0.0018,
                    "maximum": 0.011,
                },
            },
            "final_tail_net_rotation_rad": {
                "body_rad": 8.4e-06,
                "nut_rad": 1.5e-06,
                "nut_relative_to_body_rad": 8.0e-06,
            },
            "pre_lift_grasp_controller_evidence": {
                "sequential_final_summary": {
                    "normalized_loads": [0.85, 0.86, 0.85],
                    "normalized_load_imbalance": 0.01,
                },
                "contact_order": ["f2", "f3", "f1"],
                "lift_ready": True,
                "failure_reason": None,
            },
        }
    )
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


def _failure_report(seed: int) -> dict:
    # Real s065-shaped failure schema: terminal/acceptance/final-contact
    # fields are null because the episode never reached them.
    report = _base_report(seed, False)
    report.update(
        {
            "formal_acceptance": None,
            "grasp_controller": None,
            "final_contacts": None,
            "finite_throughout": None,
            "finite_final": None,
            "final_tail_diagnostics_finite": None,
            "final_all_fingers_body_contact": None,
            "zero_forbidden_contacts": None,
            "final_unsupported": None,
            "first_torque_safety_violation": None,
            "formal_lift_monitor": {
                "failed": True,
                "failure_reason": "f2_load_lost",
                "steps": 1400,
                "force_gate_n": 8.0,
                "moment_gate_nm": 0.30,
                "peak_wrist_force_increment_n": 3.2,
                "peak_moment_safety_score_nm": 0.28,
                "moment_trigger_component": None,
            },
            "formal_lift_failure": {
                "reason": "f2_load_lost",
                "stage": 3,
                "stage_step": 558,
                "global_step": 12090,
                "controller_terminal": True,
            },
            "formal_recovery": {
                "requested": True,
                "completed": False,
                "original_failure_reason": "f2_load_lost",
                "failure_stage": 3,
                "failure_stage_step": 558,
                "failure_global_step": 12090,
                "interrupted_by": (
                    "RuntimeError: finger-base torque safety violation at "
                    "global_step=13482, phase=formal_lift_recovery_return, "
                    "phase_step=1392"
                ),
                "return_completed": False,
                "open_completed": False,
                "steps": {"return": 1391, "open": 0},
                "traversed_waypoint_count": 1445,
            },
            "pre_lift_grasp_controller_evidence": {
                "contact_order": ["f2", "f3", "f1"],
                "lift_ready": True,
                "failure_reason": None,
                "finger_states_final": {
                    f: "STABLE_CONTACT" for f in FINGERS
                },
            },
            "sequential_consolidation": {
                "completed": True,
                "lift_ready": True,
                "targets_frozen_exact": True,
                "applied_scale_min": 0.35,
                "applied_scale_max": 1.0,
                "commanded_scale_monotonic": True,
                "ramp_steps": 120,
                "window_steps": 240,
                "final_window_sample_count": 240,
                "final_stiffness_scale": 1.0,
                "soft_hold_stiffness_scale": 0.35,
                "final_root_reference_nm": [0.25, 0.26, 0.25],
                "final_wrist_reference": list(WRIST),
            },
            "table_stage": {
                "translation_xy_m": 0.000004,
                "yaw_delta_rad": -0.0001,
            },
        }
    )
    return report


def _write_steps(
    directory: Path,
    *,
    window_overrides=None,
    window_count=240,
    include_contact_order=True,
) -> None:
    records = []
    step = 9000
    for index in range(1, 121):
        step += 1
        record = {
            "global_step": step,
            "phase": "physical_grip_consolidation",
            "applied_finger_stiffness_scale": 0.35 + 0.65 * index / 120,
            "next_command_finger_stiffness_scale": [1.0, 1.0, 1.0],
            "finger_root_torque_proxy_nm": {
                "f1": 0.25,
                "f2": 0.26,
                "f3": 0.25,
            },
            "finger_targets_rad": [0.7, 0.5, 0.7],
            "controller_evidence": {
                "consolidation_ramp_step": index,
                "consolidation_window_step": 0,
                "soft_hold_stiffness_scale_configured": 0.35,
                "consolidation_final_stiffness_scale_configured": 1.0,
                "consolidation_ramp_steps_configured": 120,
                "consolidation_window_steps_configured": 240,
            },
        }
        if include_contact_order:
            record["contact_order"] = ["f2", "f3", "f1"]
        records.append(json.dumps(record))
    for index in range(1, window_count + 1):
        step += 1
        record = {
            "global_step": step,
            "phase": "physical_grip_consolidation",
            "applied_finger_stiffness_scale": 1.0,
            "next_command_finger_stiffness_scale": [1.0, 1.0, 1.0],
            "finger_root_torque_proxy_nm": {
                "f1": 0.25,
                "f2": 0.26,
                "f3": 0.25,
            },
            "finger_targets_rad": [0.7, 0.5, 0.7],
            "controller_evidence": {
                "consolidation_ramp_step": 120,
                "consolidation_window_step": index,
                "soft_hold_stiffness_scale_configured": 0.35,
                "consolidation_final_stiffness_scale_configured": 1.0,
                "consolidation_ramp_steps_configured": 120,
                "consolidation_window_steps_configured": 240,
                "targets_match_frozen": True,
                "frozen_targets_rad": [0.7, 0.5, 0.7],
                "post_states": {f: "STABLE_CONTACT" for f in FINGERS},
            },
        }
        if include_contact_order:
            record["contact_order"] = ["f2", "f3", "f1"]
        if window_overrides is not None:
            override = window_overrides.get(index)
            if isinstance(override, dict):
                record.update(override)
            elif override == "skip":
                continue
        records.append(json.dumps(record))
    (directory / "controller_steps.jsonl").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )


def _write_episode(directory: Path, seed: int, report: dict, **steps_kwargs):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nominal_physics_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    _write_steps(directory, **steps_kwargs)


def _write_log(path: Path, text: str = GPU_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_cli(pairs, output_dir, extra=()):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = SOURCE_ROOT + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    command = [sys.executable, str(CLI)]
    for episode, log in pairs:
        command += ["--episode", str(episode), "--kit-log", str(log)]
    command += ["--output", str(output_dir), *extra]
    result = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    return result.returncode, result.stdout, result.stderr


def _summary(output_dir) -> dict:
    return json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )


def test_happy_path_two_pass_episodes(tmp_path):
    pairs = []
    for seed in (0, 1):
        directory = tmp_path / f"ep{seed}"
        _write_episode(directory, seed, _success_report(seed))
        log = tmp_path / f"seed{seed}_kit.log"
        _write_log(log)
        pairs.append((directory, log))
    output = tmp_path / "out"
    code, _stdout, stderr = _run_cli(pairs, output)
    assert code == 0, stderr
    summary = _summary(output)
    assert summary["structural_valid"] is True
    assert summary["all_physical_pass"] is True
    assert summary["statistics"]["physical_success_count"] == 2
    assert summary["statistics"]["drop_count"] == 0
    manifest = json.loads(
        (output / "input_manifest.json").read_text(encoding="utf-8")
    )
    recomputed = hashlib.sha256(
        json.dumps(
            {"inputs": manifest["inputs"]}, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["manifest_content_sha256"] == recomputed
    assert summary["input_manifest_sha256"] == recomputed
    assert summary["statistics"]["n_less_than_30_preliminary"] is True
    root = summary["statistics"]["sensors"]["final_root_reference"]
    assert set(root) == {"root_f1", "root_f2", "root_f3"}
    assert (
        summary["statistics"]["posthoc_only"]["pose_error_abs"]["drz_rad"][
            "readable"
        ]["maximum"]
        == pytest.approx(0.05 * 180 / np.pi)
    )


def test_real_shaped_failure_preserved_and_require_all_pass(tmp_path):
    directory = tmp_path / "ep"
    _write_episode(directory, 1, _failure_report(1))
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 0, stderr
    summary = _summary(tmp_path / "out")
    episode = summary["episodes"][0]
    assert episode["structural_ok"] is True
    assert episode["physical_success"] is False
    failure = episode["failure_record"]
    assert failure["primary_failure"] == "f2_load_lost"
    assert failure["first_failure_stage"] == 3
    assert failure["first_failure_stage_step"] == 558
    assert failure["first_failure_global_step"] == 12090
    assert failure["recovery_requested"] is True
    assert failure["recovery_completed"] is False
    assert failure["recovery_return_completed"] is False
    assert "finger-base torque safety violation" in failure[
        "recovery_secondary_failure"
    ]
    assert summary["statistics"]["failure_reason_distribution"] == {
        "f2_load_lost": 1
    }
    assert summary["statistics"]["drop_count"] == 1
    assert summary["statistics"]["recovery_requested_count"] == 1
    assert summary["statistics"]["recovery_incomplete_count"] == 1
    code2, _stdout2, _stderr2 = _run_cli(
        [(directory, log)], tmp_path / "out2", extra=("--require-all-pass",)
    )
    assert code2 == 2


def test_conflicting_primary_reasons_rejected(tmp_path):
    directory = tmp_path / "ep"
    report = _failure_report(1)
    report["formal_lift_monitor"]["failure_reason"] = "f3_load_lost"
    _write_episode(directory, 1, report)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1
    summary = _summary(tmp_path / "out")
    assert any(
        "conflicting primary" in problem
        for episode in summary["episodes"]
        for problem in episode["structural_problems"]
    )


def test_failure_without_any_reason_rejected(tmp_path):
    directory = tmp_path / "ep"
    report = _failure_report(1)
    report["formal_lift_monitor"]["failure_reason"] = None
    report["formal_recovery"]["original_failure_reason"] = None
    report["formal_lift_failure"]["reason"] = None
    report["error"] = None
    _write_episode(directory, 1, report)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


def test_duplicate_seed_rejected(tmp_path):
    directory_a = tmp_path / "a"
    directory_b = tmp_path / "b"
    _write_episode(directory_a, 1, _success_report(1))
    _write_episode(directory_b, 1, _success_report(1))
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli(
        [(directory_a, log), (directory_b, log)], tmp_path / "out"
    )
    assert code == 1


def test_hash_mismatch_rejected(tmp_path):
    directory = tmp_path / "ep"
    report = _success_report(0)
    report["provenance"]["runner_sha256"] = "f" * 64
    _write_episode(directory, 0, report)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1
    summary = _summary(tmp_path / "out")
    assert any(
        "runner_sha256" in problem
        for episode in summary["episodes"]
        for problem in episode["structural_problems"]
    )


@pytest.mark.parametrize(
    "log_text",
    [
        GPU_LOG.replace("NVIDIA GeForce RTX 5070 Ti", "NVIDIA GTX 1080"),
        GPU_LOG.replace("Yes: 0", "Yes: 1"),
        GPU_LOG + "Failed to create any GPU devices\n",
        GPU_LOG + "[Fatal] something\n",
        GPU_LOG + "[Error] something\n",
        GPU_LOG + "CPU fallback enabled\n",
        GPU_LOG + "Warp initialized on cpu\n",
    ],
)
def test_gpu_log_violations_rejected(tmp_path, log_text):
    directory = tmp_path / "ep"
    _write_episode(directory, 0, _success_report(0))
    log = tmp_path / "log"
    _write_log(log, log_text)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"control_reads_object_truth": True},
        {"control_reads_contact_report": True},
        {"proxy_collision_filter": {"enabled": True}},
        {"object_pose_writes_after_start": 1},
        {
            "formal_acceptance": {"post_grasp_stabilization_proxy_used": True}
        },
    ],
)
def test_truth_or_proxy_violations_rejected(tmp_path, overrides):
    directory = tmp_path / "ep"
    _write_episode(directory, 0, _success_report(0, **overrides))
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


def test_bad_jsonl_line_rejected(tmp_path):
    directory = tmp_path / "ep"
    _write_episode(directory, 0, _success_report(0))
    with (directory / "controller_steps.jsonl").open("a") as handle:
        handle.write("{broken json\n")
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


def test_non_increasing_steps_rejected(tmp_path):
    directory = tmp_path / "ep"
    _write_episode(directory, 0, _success_report(0))
    path = directory / "controller_steps.jsonl"
    lines = path.read_text().splitlines()
    lines[10] = json.dumps(json.loads(lines[9]))
    path.write_text("\n".join(lines) + "\n")
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


def test_contact_order_mismatch_rejected(tmp_path):
    directory = tmp_path / "ep"
    report = _success_report(0)
    report["grasp_controller"]["contact_order"] = ["f1", "f2", "f3"]
    _write_episode(directory, 0, report)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


@pytest.mark.parametrize(
    "steps_kwargs",
    [
        {"window_overrides": {5: "skip"}},
        {
            "window_overrides": {
                5: {"controller_evidence": {
                    "consolidation_window_step": 4,
                    "consolidation_ramp_step": 120,
                    "soft_hold_stiffness_scale_configured": 0.35,
                    "consolidation_final_stiffness_scale_configured": 1.0,
                    "consolidation_ramp_steps_configured": 120,
                    "consolidation_window_steps_configured": 240,
                    "targets_match_frozen": True,
                    "post_states": {f: "STABLE_CONTACT" for f in FINGERS},
                }}
            }
        },
        {
            "window_overrides": {
                3: {"applied_finger_stiffness_scale": 0.9}
            }
        },
        {
            "window_overrides": {
                3: {"finger_targets_rad": [0.1, 0.2, 0.3]}
            }
        },
    ],
)
def test_steps_cross_violations_rejected(tmp_path, steps_kwargs):
    directory = tmp_path / "ep"
    _write_episode(directory, 0, _success_report(0), **steps_kwargs)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


def test_evidence_config_mismatch_rejected(tmp_path):
    directory = tmp_path / "ep"
    _write_episode(
        directory,
        0,
        _success_report(0),
        window_overrides={
            3: {
                "controller_evidence": {
                    "consolidation_ramp_step": 120,
                    "consolidation_window_step": 3,
                    "soft_hold_stiffness_scale_configured": 0.35,
                    "consolidation_final_stiffness_scale_configured": 0.65,
                    "consolidation_ramp_steps_configured": 120,
                    "consolidation_window_steps_configured": 240,
                    "targets_match_frozen": True,
                    "post_states": {f: "STABLE_CONTACT" for f in FINGERS},
                }
            }
        },
    )
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1
    summary = _summary(tmp_path / "out")
    assert any(
        "config mismatch" in problem
        for episode in summary["episodes"]
        for problem in episode["structural_problems"]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "formal_acceptance": {"passed": True, "sensor_lift_gate": False}
        },
        {"formal_acceptance": {"actual_body_lift_m": 0.044}},
        {"final_all_fingers_body_contact": False},
        {"zero_forbidden_contacts": False},
        {"final_unsupported": True},
        {"final_tail_diagnostics_finite": False},
        {
            "formal_lift_monitor": {
                "peak_moment_safety_score_nm": 0.31,
                "failed": False,
                "failure_reason": None,
                "steps": 2972,
                "force_gate_n": 8.0,
                "moment_gate_nm": 0.30,
                "peak_wrist_force_increment_n": 2.5,
                "moment_trigger_component": None,
            }
        },
        {
            "formal_lift_monitor": {
                "peak_wrist_force_increment_n": 8.5,
                "failed": False,
                "failure_reason": None,
                "steps": 2972,
                "force_gate_n": 8.0,
                "moment_gate_nm": 0.30,
                "peak_moment_safety_score_nm": 0.15,
                "moment_trigger_component": None,
            }
        },
        {"formal_lift_monitor": {"moment_gate_nm": 0.35}},
        {"sequential_consolidation": {"applied_scale_min": 0.3525}},
        {"sequential_consolidation": {"ramp_steps": 121}},
        {"sequential_consolidation": {"window_steps": 241}},
        {"sequential_consolidation": {"final_window_sample_count": 239}},
        {"sequential_consolidation": {"commanded_scale_monotonic": False}},
        {"sequential_consolidation": {"targets_frozen_exact": False}},
        {"sequential_consolidation": {"final_stiffness_scale": 0.65}},
    ],
)
def test_success_acceptance_violations_rejected(tmp_path, overrides):
    directory = tmp_path / "ep"
    _write_episode(directory, 0, _success_report(0, **overrides))
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


def test_contradictory_passed_exit_rejected(tmp_path):
    directory = tmp_path / "ep"
    report = _success_report(0)
    report["passed"] = True
    report["process_exit_code"] = 1
    _write_episode(directory, 0, report)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


def test_posthoc_never_recomputes_pass(tmp_path):
    directory = tmp_path / "ep"
    report = _success_report(0)
    report["posthoc_pose_error"] = {
        "dx_m": 0.5,
        "dy_m": 0.5,
        "dz_m": 0.5,
        "drx_rad": 0.5,
        "dry_rad": 0.5,
        "drz_rad": 0.5,
    }
    _write_episode(directory, 0, report)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 0
    summary = _summary(tmp_path / "out")
    assert summary["episodes"][0]["physical_success"] is True
    translation = summary["statistics"]["posthoc_only"]["pose_error_signed"][
        "translation_norm"
    ]
    assert translation["readable"]["mean"] == pytest.approx(866.0254, rel=1e-3)


def test_unavailable_metrics_listed(tmp_path):
    directory = tmp_path / "ep"
    report = _success_report(0)
    del report["final_tail_observable_angular_speed_rad_s"]
    del report["final_tail_net_rotation_rad"]
    _write_episode(directory, 0, report)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 0
    summary = _summary(tmp_path / "out")
    unavailable = summary["episodes"][0]["unavailable_metrics"]
    assert "nut_relative_net_rotation_deg" in unavailable
    assert "nut_relative_tail_speed_mean_rad_s" in unavailable
    assert (
        summary["statistics"]["posthoc_only"]["nut_relative_tail_speed"][
            "mean"
        ]["all_valid_available"]
        is False
    )


def test_units_and_linear_p95(tmp_path):
    pairs = []
    for seed in range(5):
        directory = tmp_path / f"ep{seed}"
        _write_episode(directory, seed, _success_report(seed))
        log = tmp_path / f"log{seed}"
        _write_log(log)
        pairs.append((directory, log))
    code, _stdout, stderr = _run_cli(pairs, tmp_path / "out")
    assert code == 0, stderr
    summary = _summary(tmp_path / "out")
    moment = summary["statistics"]["sensors"]["moment_score_peak"]["all_valid"]
    assert moment["p95"] == pytest.approx(
        float(np.percentile(np.asarray([0.15] * 5), 95))
    )


def test_early_failure_without_contact_order_is_valid(tmp_path):
    directory = tmp_path / "ep"
    report = _failure_report(1)
    _write_episode(
        directory, 1, report, include_contact_order=False
    )
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 0, stderr
    summary = _summary(tmp_path / "out")
    episode = summary["episodes"][0]
    assert episode["structural_ok"] is True
    assert episode["physical_success"] is False
    assert episode["failure_record"]["primary_failure"] == "f2_load_lost"
    assert episode["steps_cross"]["final_contact_order"] is None


def test_failure_partial_valid_window_is_valid(tmp_path):
    directory = tmp_path / "ep"
    report = _failure_report(1)
    report["sequential_consolidation"]["completed"] = False
    _write_episode(directory, 1, report, window_count=60)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 0, stderr
    summary = _summary(tmp_path / "out")
    assert summary["episodes"][0]["structural_ok"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_overrides": {5: "skip"}},
        {
            "window_overrides": {
                5: {
                    "controller_evidence": {
                        "consolidation_ramp_step": 120,
                        "consolidation_window_step": 4,
                        "soft_hold_stiffness_scale_configured": 0.35,
                        "consolidation_final_stiffness_scale_configured": 1.0,
                        "consolidation_ramp_steps_configured": 120,
                        "consolidation_window_steps_configured": 240,
                        "targets_match_frozen": True,
                        "post_states": {f: "STABLE_CONTACT" for f in FINGERS},
                    }
                }
            }
        },
        {"window_overrides": {3: {"applied_finger_stiffness_scale": 0.9}}},
        {"window_overrides": {3: {"finger_targets_rad": [0.1, 0.2, 0.3]}}},
        {
            "window_overrides": {
                3: {
                    "controller_evidence": {
                        "consolidation_ramp_step": 120,
                        "consolidation_window_step": 3,
                        "soft_hold_stiffness_scale_configured": 0.35,
                        "consolidation_final_stiffness_scale_configured": 0.65,
                        "consolidation_ramp_steps_configured": 120,
                        "consolidation_window_steps_configured": 240,
                        "targets_match_frozen": True,
                        "post_states": {f: "STABLE_CONTACT" for f in FINGERS},
                    }
                }
            }
        },
    ],
)
def test_failure_partial_window_violations_rejected(tmp_path, kwargs):
    directory = tmp_path / "ep"
    report = _failure_report(1)
    report["sequential_consolidation"]["completed"] = False
    _write_episode(directory, 1, report, window_count=60, **kwargs)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


def test_failure_claims_completed_but_short_window_rejected(tmp_path):
    directory = tmp_path / "ep"
    report = _failure_report(1)
    report["sequential_consolidation"]["completed"] = True
    _write_episode(directory, 1, report, window_count=60)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


def test_success_missing_config_field_rejected(tmp_path):
    directory = tmp_path / "ep"
    _write_episode(
        directory,
        0,
        _success_report(0),
        window_overrides={
            3: {
                "controller_evidence": {
                    "consolidation_ramp_step": 120,
                    "consolidation_window_step": 3,
                    "soft_hold_stiffness_scale_configured": 0.35,
                    "consolidation_ramp_steps_configured": 120,
                    "consolidation_window_steps_configured": 240,
                    "targets_match_frozen": True,
                    "post_states": {f: "STABLE_CONTACT" for f in FINGERS},
                }
            }
        },
    )
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"formal_lift_monitor": {
            "failed": True,
            "failure_reason": "f2_load_lost",
            "steps": 1400,
            "force_gate_n": 8.5,
            "moment_gate_nm": 0.30,
            "peak_wrist_force_increment_n": 3.2,
            "peak_moment_safety_score_nm": 0.28,
            "moment_trigger_component": None,
        }},
        {"formal_lift_monitor": {
            "failed": True,
            "failure_reason": "f2_load_lost",
            "steps": 1400,
            "force_gate_n": 8.0,
            "moment_gate_nm": 0.30,
            "peak_wrist_force_increment_n": float("nan"),
            "peak_moment_safety_score_nm": 0.28,
            "moment_trigger_component": None,
        }},
    ],
)
def test_failure_monitor_bounds_rejected(tmp_path, overrides):
    directory = tmp_path / "ep"
    report = _failure_report(1)
    report.update(overrides)
    _write_episode(directory, 1, report)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, _stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 1


def test_unstructured_error_as_primary_source(tmp_path):
    directory = tmp_path / "ep"
    report = _failure_report(1)
    report["formal_lift_monitor"]["failure_reason"] = None
    report["formal_recovery"]["original_failure_reason"] = None
    report["formal_lift_failure"]["reason"] = None
    report["error"] = "sequential-compliant grasp failed closed"
    _write_episode(directory, 1, report)
    log = tmp_path / "log"
    _write_log(log)
    code, _stdout, stderr = _run_cli([(directory, log)], tmp_path / "out")
    assert code == 0, stderr
    summary = _summary(tmp_path / "out")
    failure = summary["episodes"][0]["failure_record"]
    assert failure["primary_failure"] == (
        "sequential-compliant grasp failed closed"
    )
    assert failure["primary_source"] == "report.error_unstructured"
    assert summary["episodes"][0]["physical_success"] is False
