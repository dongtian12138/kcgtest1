'''Pure tests for zero-lift hold CLI/result semantics and config bounds.'''

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.physical_grasp_config import (
    load_physical_grasp_experiment_config,
)

REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
)
RUNNER = (
    Path(__file__).resolve().parents[1]
    / "isaac"
    / "d38999_tabletop_pick_smoke.py"
)


def test_zero_lift_hold_config_is_bounded_and_characterization_only():
    config = load_physical_grasp_experiment_config(CONFIG)
    assert config.zero_lift_hold_duration_s == pytest.approx(5.0)
    assert config.zero_lift_hold_maximum_duration_s == pytest.approx(30.0)
    assert config.reference_window_steps == 120
    assert config.reference_window_steps == round(
        config.synchronous_preload_duration_s * config.physics_rate_hz
    )
    assert config.terminal_evaluator_lift_started_dz_m > 0.0


def test_reference_window_must_match_preload_duration(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["reference"]["window_steps"] = 119
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="must match"):
        load_physical_grasp_experiment_config(changed)


def test_zero_lift_hold_duration_must_stay_bounded(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["zero_lift_hold"]["duration_s"] = 31.0
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="bounded"):
        load_physical_grasp_experiment_config(changed)


def test_reference_evidence_and_terminal_truth_must_stay_log_only(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["reference"]["evidence_only"] = False
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence-only"):
        load_physical_grasp_experiment_config(changed)
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["terminal_evaluator"]["posthoc_truth_evaluation_only"] = False
    changed = tmp_path / "changed2.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="log-only"):
        load_physical_grasp_experiment_config(changed)


def _runner_argument_kwargs():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    results = {}
    for call in ast.walk(function):
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "add_argument"
        ):
            continue
        option = None
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                option = arg.value
        if option:
            results[option] = {kw.arg: kw.value for kw in call.keywords}
    return results


def test_formal_lift_mode_cli_defaults_to_staged_with_bounded_choices():
    arguments = _runner_argument_kwargs()
    option = arguments["--formal-lift-mode"]
    assert ast.literal_eval(option["choices"]) == (
        "staged",
        "zero-lift-hold",
        "gravity-transfer-hold",
        "stiffness-restore-hold",
    )
    assert ast.literal_eval(option["default"]) == "staged"


def test_zero_lift_hold_requires_formal_grasp_method():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    messages = []
    for call in ast.walk(function):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "error"
        ):
            messages.append(ast.unparse(call.args[0]))
    assert any(
        "zero-lift-hold" in message and "formal" in message
        for message in messages
    )


def test_zero_lift_hold_never_becomes_a_grasp_pass():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'phase = "physical_grip_zero_lift_hold"' in source
    assert '"CHARACTERIZATION_ONLY": True' in source
    assert '"characterization_completed": False' in source
    assert '"passed": False' in source
    assert "run_formal_failure_recovery()" in source
    assert "zero-lift hold characterization completed;" in source
    assert "CHARACTERIZATION_ONLY episode is not a grasp PASS" in source


def test_zero_lift_hold_gate_trigger_still_fails_closed():
    source = RUNNER.read_text(encoding="utf-8")
    # A gate trigger inside the hold must run the same fail-closed recovery.
    assert '"gate_triggered": True' in source
    assert "if formal_lift_failure is not None:" in source
    # The hold reuses the original monitor update, never a new gate.
    assert "formal_lift_monitor.update(" in source
    assert "moment_magnitude_increase" in source
    assert "candidates_gate_control" in source


def test_zero_lift_hold_holds_the_exact_grasp_commands():
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index("if zero_lift_hold_mode:")
    region = source[start : source.index("# Staged mode below", start)]
    # H6 may leave a non-zero drive target bias relative to lift_start.
    # Freeze the exact pre-branch command instead of dropping that bias.
    assert "zero_lift_arm_target = current_arm_target.copy()" in region
    assert "zero_lift_arm_target, current_hand_target, True" in region
    assert '"holds_exact_prebranch_arm_command": True' in region
    assert "lift_start, current_hand_target, True" not in region
    assert "hold_steps = max(" in region
    assert "zero_lift_hold_duration_s * rate_hz" in region
