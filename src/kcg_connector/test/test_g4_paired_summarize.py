"""Evidence-integrity tests for the offline paired summarizer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPOSITORY / "src/kcg_connector/isaac/d38999_g4_paired_summarize.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "paired_summarize", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _console(method: str) -> str:
    return (
        "NVIDIA GeForce RTX 5070 Ti\n"
        "| 0 | NVIDIA GeForce RTX 5070 Ti | Yes: 0 |\n"
        '"cuda:0" CUDA Toolkit 12.9\n'
        f"--physical-grasp-method {method}\n"
    )


def _report(module, method: str, seed: int, *, passed: bool) -> dict:
    payload = hashlib.sha256(f"payload-{seed}".encode()).hexdigest()
    report = {
        "seed": seed,
        "gui": False,
        "physical_grasp_method": method,
        "formal_lift_mode": "staged",
        "passed": passed,
        "process_exit_code": 0 if passed else 1,
        "formal_acceptance": {"passed": True} if passed else None,
        "formal_lift_monitor": {
            "failed": not passed,
            "failure_reason": None if passed else "f2_load_lost",
            "peak_wrist_force_increment_n": 2.0,
            "peak_moment_safety_score_nm": 0.1,
        },
        "provenance": {
            **module.current_source_hashes(),
            "payload_sha256": payload,
        },
        "realized_randomization": {
            "seed": seed,
            "method": method,
            "payload_sha256": payload,
            "canonical_payload": {
                "seed": seed,
                "plug_x_offset_m": 0.0001,
            },
        },
        "realized_arm_targets": {"grasp": [0.1, 0.2, 0.3]},
        "control_reads_object_truth": False,
        "control_reads_contact_report": False,
        "object_pose_writes_after_start": 0,
        "attachment": "none",
        "posthoc_truth_evaluation_only": True,
        "table_stage": {
            "translation_xy_m": (
                0.00001 if method == "synchronous" else 0.000005
            ),
            "yaw_delta_rad": 0.001 if method == "synchronous" else 0.0005,
        },
        "posthoc_pose_error": {
            "dx_m": 0.001 if method == "synchronous" else 0.0007,
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
        "phase_steps": {"physical_hand_closure": 1000, "lift": 500},
        "grasp_controller": {
            "contact_order": ["f2", "f3", "f1"],
            "contact_global_steps": {"f1": 30, "f2": 10, "f3": 20},
        },
    }
    if method == "sequential-compliant":
        report["pre_lift_grasp_controller_evidence"] = {
            "sequential_final_summary": {
                "normalized_load_imbalance": 0.02
            }
        }
    return report


def _write_pair(tmp_path: Path, *, passed=(True, True)):
    module = _module()
    pair_dir = tmp_path / "seed000"
    pair_dir.mkdir()
    sides = []
    normalized = [
        "/wrapper",
        "/runner",
        "--formal-lift-mode",
        "staged",
        "--seed",
        "0",
    ]
    for index, method in enumerate(module.SIDE_METHODS):
        side_dir = pair_dir / ("sync" if index == 0 else "sequential")
        side_dir.mkdir()
        report_path = side_dir / "nominal_physics_report.json"
        trace_path = side_dir / "controller_steps.jsonl"
        console_path = side_dir / "side_console.log"
        report_path.write_text(
            json.dumps(
                _report(module, method, 0, passed=passed[index])
            )
            + "\n",
            encoding="utf-8",
        )
        trace_path.write_text(
            json.dumps({"global_step": 1, "method": method}) + "\n",
            encoding="utf-8",
        )
        console_path.write_text(_console(method), encoding="utf-8")
        sides.append(
            {
                "method": method,
                "kind": "fresh",
                "normalized_argv": list(normalized),
                "report_file": str(report_path),
                "report_sha256": _sha(report_path),
                "trace_file": str(trace_path),
                "trace_sha256": _sha(trace_path),
                "side_console": str(console_path),
                "side_console_sha256": _sha(console_path),
            }
        )
    manifest = {
        "schema_version": "kcg_g4_paired_batch_v1",
        "seed": 0,
        "gui": False,
        "batch_runner_sha256": _sha(module.BATCH_RUNNER_PATH),
        "source_hashes": module.current_source_hashes(),
        "sides": sides,
    }
    manifest["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(
            {"seed": 0, "gui": False, "sides": sides},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = pair_dir / "pair_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return module, pair_dir, manifest


def _rewrite_manifest(pair_dir: Path, manifest: dict) -> None:
    manifest["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "seed": manifest.get("seed"),
                "gui": manifest.get("gui"),
                "sides": manifest.get("sides"),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    (pair_dir / "pair_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def test_valid_pair_produces_method_and_paired_statistics(tmp_path):
    module, pair_dir, _manifest = _write_pair(tmp_path)
    output = tmp_path / "out"
    code = module.main(
        [
            "--pair-dir",
            str(pair_dir),
            "--require-complete-pairs",
            "--require-all-pass",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["structural_valid"] is True
    assert summary["all_sides_physical_pass"] is True
    delta = summary["paired_deltas"]["posthoc_abs_dx_m_mm"]
    assert delta["paired_sample_count"] == 1
    assert delta["stats"]["mean"] < 0.0
    assert summary["preliminary"] is True
    assert summary["regrasp"] == "unavailable_and_not_inferred"


def test_physical_failure_is_structural_but_require_all_pass_returns_two(
    tmp_path,
):
    module, pair_dir, _manifest = _write_pair(
        tmp_path, passed=(False, True)
    )
    assert module.main(
        ["--pair-dir", str(pair_dir), "--output", str(tmp_path / "out1")]
    ) == 0
    assert module.main(
        [
            "--pair-dir",
            str(pair_dir),
            "--require-all-pass",
            "--output",
            str(tmp_path / "out2"),
        ]
    ) == 2


def test_report_hash_tamper_fails_closed(tmp_path):
    module, pair_dir, _manifest = _write_pair(tmp_path)
    report = pair_dir / "sync/nominal_physics_report.json"
    report.write_text(
        report.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    assert module.main(
        ["--pair-dir", str(pair_dir), "--output", str(tmp_path / "out")]
    ) == 1


def test_payload_and_normalized_argv_mismatch_fail_closed(tmp_path):
    module, pair_dir, manifest = _write_pair(tmp_path)
    report_path = pair_dir / "sequential/nominal_physics_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["provenance"]["payload_sha256"] = "f" * 64
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    manifest["sides"][1]["report_sha256"] = _sha(report_path)
    manifest["sides"][1]["normalized_argv"].append("--extra")
    _rewrite_manifest(pair_dir, manifest)
    assert module.main(
        ["--pair-dir", str(pair_dir), "--output", str(tmp_path / "out")]
    ) == 1


def test_duplicate_or_missing_side_fails_closed(tmp_path):
    module, pair_dir, manifest = _write_pair(tmp_path)
    manifest["sides"] = [manifest["sides"][0], manifest["sides"][0]]
    _rewrite_manifest(pair_dir, manifest)
    assert module.main(
        ["--pair-dir", str(pair_dir), "--output", str(tmp_path / "out")]
    ) == 1


def test_nonfinite_json_is_rejected_even_when_hash_is_updated(tmp_path):
    module, pair_dir, manifest = _write_pair(tmp_path)
    report_path = pair_dir / "sync/nominal_physics_report.json"
    text = report_path.read_text(encoding="utf-8").rstrip()
    report_path.write_text(text[:-1] + ', "bad": NaN}\n', encoding="utf-8")
    manifest["sides"][0]["report_sha256"] = _sha(report_path)
    _rewrite_manifest(pair_dir, manifest)
    assert module.main(
        ["--pair-dir", str(pair_dir), "--output", str(tmp_path / "out")]
    ) == 1


def test_missing_gpu_marker_and_source_mismatch_fail_closed(tmp_path):
    module, pair_dir, manifest = _write_pair(tmp_path)
    console = pair_dir / "sync/side_console.log"
    console.write_text("synchronous only\n", encoding="utf-8")
    manifest["sides"][0]["side_console_sha256"] = _sha(console)
    manifest["source_hashes"]["runner_sha256"] = "0" * 64
    _rewrite_manifest(pair_dir, manifest)
    assert module.main(
        ["--pair-dir", str(pair_dir), "--output", str(tmp_path / "out")]
    ) == 1
