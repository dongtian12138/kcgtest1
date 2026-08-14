"""Gate-order and side-effect tests for the full-skill training stub."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from kcg_rl import full_skill_train


def test_blocked_gate_is_first_call_and_creates_no_output(
    tmp_path, monkeypatch
):
    calls = []
    output = tmp_path / "must_not_exist"

    def blocked(config, repo_root):
        calls.append(("gate", config, repo_root, output.exists()))
        raise RuntimeError("deliberately blocked")

    def backend():
        calls.append(("backend",))

    monkeypatch.setattr(full_skill_train, "require_training_ready", blocked)
    monkeypatch.setattr(full_skill_train, "_load_training_backend", backend)
    result = full_skill_train.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(output),
        ]
    )
    assert result == full_skill_train.EXIT_BLOCKED
    assert calls == [
        (
            "gate",
            tmp_path / "config.yaml",
            tmp_path,
            False,
        )
    ]
    assert not output.exists()


def test_ready_gate_reaches_distinct_unimplemented_state_without_output(
    tmp_path, monkeypatch, capsys
):
    calls = []
    output = tmp_path / "must_not_exist"

    def ready(config, repo_root):
        del config, repo_root
        calls.append("gate")
        return object()

    def backend():
        calls.append("backend")
        raise full_skill_train.FullSkillBackendNotImplementedError("missing")

    monkeypatch.setattr(full_skill_train, "require_training_ready", ready)
    monkeypatch.setattr(full_skill_train, "_load_training_backend", backend)
    result = full_skill_train.main(
        ["--repo-root", str(tmp_path), "--output-dir", str(output)]
    )
    assert result == full_skill_train.EXIT_BACKEND_NOT_IMPLEMENTED
    assert calls == ["gate", "backend"]
    assert (
        "READY_GATE_PASSED_BACKEND_NOT_IMPLEMENTED"
        in capsys.readouterr().err
    )
    assert not output.exists()


def test_checked_in_blocked_cli_loads_no_training_or_simulator_modules(
    tmp_path,
):
    package_root = Path(__file__).resolve().parents[1]
    repository = package_root.parents[1]
    output = tmp_path / "must_not_exist"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)
    script = """
import importlib
import pathlib
import sys

module = importlib.import_module("kcg_rl.full_skill_train")
result = module.main([
    "--repo-root", sys.argv[1],
    "--output-dir", sys.argv[2],
])
assert result == module.EXIT_BLOCKED
assert not pathlib.Path(sys.argv[2]).exists()
for name in tuple(sys.modules):
    root = name.split(".", 1)[0]
    assert root not in {
        "torch", "stable_baselines3", "gym", "gymnasium", "omni",
        "isaacsim", "rclpy"
    }, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(repository), str(output)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert not output.exists()


def test_default_backend_loader_never_imports_training_packages(monkeypatch):
    imported = []

    def forbidden_import(*args, **kwargs):
        imported.append((args, kwargs))
        raise AssertionError("unexpected import")

    monkeypatch.setattr("builtins.__import__", forbidden_import)
    with pytest.raises(
        full_skill_train.FullSkillBackendNotImplementedError,
        match="not implemented",
    ):
        full_skill_train._load_training_backend()
    assert imported == []
