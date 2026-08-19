"""Fail-closed tests for the pure outer paired-batch orchestrator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


REPOSITORY = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPOSITORY / "src/kcg_connector/isaac/d38999_g4_paired_batch.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("paired_batch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _console(method: str) -> bytes:
    return (
        "NVIDIA GeForce RTX 5070 Ti\n"
        "| 0 | NVIDIA GeForce RTX 5070 Ti | Yes: 0 |\n"
        '"cuda:0" CUDA Toolkit 12.9\n'
        f"--physical-grasp-method {method}\n"
    ).encode("utf-8")


def _fake_subprocess(module, calls, *, fail_sync=False):
    def run(argv, *, stdout, stderr, check):
        assert stderr is module.subprocess.STDOUT
        assert check is False
        method = argv[argv.index("--physical-grasp-method") + 1]
        seed = int(argv[argv.index("--seed") + 1])
        output = Path(argv[argv.index("--output-dir") + 1])
        calls.append(method)
        stdout.write(_console(method))
        stdout.flush()
        report = {
            "seed": seed,
            "physical_grasp_method": method,
            "formal_lift_mode": "staged",
            "gui": False,
            "passed": not (fail_sync and method == "synchronous"),
            "process_exit_code": (
                1 if fail_sync and method == "synchronous" else 0
            ),
            "provenance": module.current_source_hashes(),
        }
        (output / "nominal_physics_report.json").write_text(
            json.dumps(report) + "\n", encoding="utf-8"
        )
        (output / "controller_steps.jsonl").write_text(
            json.dumps({"global_step": 1, "method": method}) + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=report["process_exit_code"])

    return run


def test_normalized_argv_drops_only_method_and_output():
    module = _module()
    first = module.build_argv(
        wrapper=Path("/wrapper"),
        runner=Path("/runner"),
        method="synchronous",
        seed=7,
        formal_lift_mode="staged",
        gui=False,
        output_dir=Path("/sync"),
    )
    second = module.build_argv(
        wrapper=Path("/wrapper"),
        runner=Path("/runner"),
        method="sequential-compliant",
        seed=7,
        formal_lift_mode="staged",
        gui=False,
        output_dir=Path("/seq"),
    )
    assert module.normalized_argv(first) == module.normalized_argv(second)
    assert "--seed" in module.normalized_argv(first)


def test_batch_runs_both_sides_even_when_sync_physically_fails(
    tmp_path, monkeypatch
):
    module = _module()
    wrapper = tmp_path / "wrapper.sh"
    runner = tmp_path / "runner.py"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    runner.write_text("# fake\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        _fake_subprocess(module, calls, fail_sync=True),
    )
    output = tmp_path / "batch"
    code = module.main(
        [
            "--seeds",
            "0",
            "--base-output-dir",
            str(output),
            "--isaac-wrapper",
            str(wrapper),
            "--runner-py",
            str(runner),
        ]
    )
    assert code == 0
    assert calls == ["synchronous", "sequential-compliant"]
    manifest = json.loads(
        (output / "seed000/pair_manifest.json").read_text(encoding="utf-8")
    )
    assert [side["method"] for side in manifest["sides"]] == list(
        module.SIDE_METHODS
    )
    assert manifest["sides"][0]["exit_code"] == 1
    assert manifest["sides"][1]["exit_code"] == 0
    assert manifest["sides"][0]["normalized_argv"] == (
        manifest["sides"][1]["normalized_argv"]
    )
    assert (output / "seed000/sync/execution_record.json").is_file()
    assert (output / "seed000/sequential/execution_record.json").is_file()


def test_finalized_pair_requires_resume_and_reverifies_without_rerun(
    tmp_path, monkeypatch
):
    module = _module()
    wrapper = tmp_path / "wrapper.sh"
    runner = tmp_path / "runner.py"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    runner.write_text("# fake\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        module.subprocess, "run", _fake_subprocess(module, calls)
    )
    args = [
        "--seeds",
        "2",
        "--base-output-dir",
        str(tmp_path / "batch"),
        "--isaac-wrapper",
        str(wrapper),
        "--runner-py",
        str(runner),
    ]
    assert module.main(args) == 0
    assert module.main(args) == 1
    calls.clear()
    assert module.main([*args, "--resume"]) == 0
    assert calls == []


def test_trace_and_console_tamper_fail_closed(tmp_path):
    module = _module()
    side = tmp_path / "side"
    side.mkdir()
    report = {
        "seed": 0,
        "physical_grasp_method": "synchronous",
        "formal_lift_mode": "staged",
        "gui": False,
        "provenance": module.current_source_hashes(),
    }
    (side / "nominal_physics_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (side / "controller_steps.jsonl").write_text(
        '{"global_step":1}\n{"global_step":1}\n', encoding="utf-8"
    )
    console = side / "side_console.log"
    console.write_bytes(
        _console("synchronous") + b"Failed to create any GPU devices\n"
    )
    problems = module.verify_side_evidence(
        side,
        console,
        expected_seed=0,
        expected_method="synchronous",
        expected_gui=False,
        source_hashes=module.current_source_hashes(),
    )
    assert any("not increasing" in problem for problem in problems)
    assert any("forbidden GPU marker" in problem for problem in problems)


def test_source_keys_are_actual_report_provenance_names():
    module = _module()
    assert "three_finger_sequential_grasp_sha256" in module.SOURCE_FILES
    assert "physical_grasp_config_sha256" in module.SOURCE_FILES
    assert "controller_sha256" not in module.SOURCE_FILES
    assert "yaml_sha256" not in module.SOURCE_FILES
