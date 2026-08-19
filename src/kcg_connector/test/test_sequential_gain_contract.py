'''Contract tests for the sequential hand-gain consistency (028/030).

The physical-grasp YAML declares approach/soft-hold stiffness, damping
and the consolidation stiffness scales/steps; the runner applies
pick.motion.grip_hand_stiffness scaled per finger by the declared soft
scale (or the consolidation trajectory) and pick.motion.grip_hand_damping
through controller.set_gains.  These tests pin the fail-closed validator,
the config-driven controller scales, the runner wiring order, the report
evidence and the YAML numbers so no gain/config drift can happen
silently before the G4 GPU smoke.
'''

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
RUNNER = (
    REPOSITORY
    / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"
)
SEQUENTIAL = (
    REPOSITORY
    / "src/kcg_connector/kcg_connector/grasp"
    / "three_finger_sequential_grasp.py"
)
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
)

DECLARED = {
    "approach_stiffness": 5.0,
    "soft_hold_stiffness": 1.75,
    "damping": 1.0,
    "soft_hold_stiffness_scale": 0.35,
    "consolidation_final_stiffness_scale": 1.0,
    "consolidation_ramp_steps": 120,
    "consolidation_window_steps": 240,
    "consolidation_threshold_label": "SIM_TUNING_ONLY_A_CANDIDATE",
}


def _runner_module():
    spec = importlib.util.spec_from_file_location(
        "d38999_tabletop_pick_smoke", RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _pick_motion(stiffness=5.0, damping=1.0):
    return SimpleNamespace(
        grip_hand_stiffness=stiffness, grip_hand_damping=damping
    )


def _sequential_config(**overrides):
    values = {
        "soft_hold_stiffness_scale": 0.35,
        "consolidation_final_stiffness_scale": 1.0,
        "consolidation_ramp_steps": 120,
        "consolidation_window_steps": 240,
        "consolidation_threshold_label": "SIM_TUNING_ONLY_A_CANDIDATE",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validator_accepts_current_wiring():
    runner = _runner_module()
    problems, evidence = runner.validate_sequential_gain_consistency(
        DECLARED,
        _pick_motion(),
        label="physical_grasp.sequential",
        sequential_config=_sequential_config(),
    )
    assert problems == []
    assert evidence["consistent"] is True
    assert evidence["declared_soft_hold_stiffness_scale"] == 0.35
    assert evidence["declared_consolidation_final_stiffness_scale"] == 1.0
    assert evidence["loaded_sequential_config"] is not None


@pytest.mark.parametrize(
    "declared, pick_motion, expected_token",
    [
        (
            {**DECLARED, "approach_stiffness": 5.1},
            _pick_motion(),
            "approach_stiffness",
        ),
        (
            {**DECLARED, "soft_hold_stiffness": 1.8},
            _pick_motion(),
            "soft_hold_stiffness",
        ),
        (
            {**DECLARED, "damping": 1.2},
            _pick_motion(),
            "damping",
        ),
        (
            DECLARED,
            _pick_motion(stiffness=5.5),
            "grip_hand_stiffness",
        ),
        (
            {**DECLARED, "soft_hold_stiffness_scale": 0.4},
            _pick_motion(),
            "soft_hold_stiffness",
        ),
    ],
)
def test_validator_rejects_drift(declared, pick_motion, expected_token):
    runner = _runner_module()
    problems, evidence = runner.validate_sequential_gain_consistency(
        declared, pick_motion, label="physical_grasp.sequential"
    )
    assert problems, expected_token
    assert evidence["consistent"] is False
    assert any(expected_token in problem for problem in problems)


@pytest.mark.parametrize(
    "declared",
    [
        {
            **DECLARED,
            "soft_hold_stiffness_scale": 0.7,
            "consolidation_final_stiffness_scale": 0.5,
        },
        {
            **DECLARED,
            "soft_hold_stiffness_scale": 0.0,
        },
        {
            **DECLARED,
            "consolidation_final_stiffness_scale": 1.2,
        },
        {
            **DECLARED,
            "consolidation_final_stiffness_scale": 0.35,
        },
        {
            **DECLARED,
            "soft_hold_stiffness_scale": True,
        },
        {
            "approach_stiffness": 5.0,
            "soft_hold_stiffness": 1.75,
            "damping": 1.0,
        },
    ],
)
def test_validator_rejects_scale_violations_and_missing(declared):
    runner = _runner_module()
    problems, evidence = runner.validate_sequential_gain_consistency(
        declared, _pick_motion(), label="physical_grasp.sequential"
    )
    assert problems
    assert evidence["consistent"] is False


def test_validator_cross_checks_loaded_sequential_config():
    runner = _runner_module()
    for overrides in (
        {"consolidation_ramp_steps": 121},
        {"consolidation_final_stiffness_scale": 0.9},
        {
            "consolidation_threshold_label": "SIM_TUNING_ONLY",
        },
    ):
        problems, evidence = runner.validate_sequential_gain_consistency(
            DECLARED,
            _pick_motion(),
            label="physical_grasp.sequential",
            sequential_config=_sequential_config(**overrides),
        )
        assert problems, overrides
        assert evidence["consistent"] is False


def test_controller_scales_are_config_driven_not_literal():
    tree = ast.parse(SEQUENTIAL.read_text(encoding="utf-8"))
    commands = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_command"
    ]
    assert len(commands) == 1
    scale_nodes = [
        node
        for node in ast.walk(commands[0])
        if isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Constant)
        and isinstance(node.body.value, float)
    ]
    # The only float literal stiffness in _command is the 1.0 non-contact
    # fallback; the soft scale must come from the config object.
    for node in scale_nodes:
        assert node.body.value == 1.0
    unparsed = ast.unparse(commands[0])
    assert "soft_hold_stiffness_scale" in unparsed
    assert "consolidation_stiffness_scale" in unparsed
    module = ast.parse(SEQUENTIAL.read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: node.value.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert constants["DEFAULT_SOFT_HOLD_STIFFNESS_SCALE"] == 0.35
    assert constants["DEFAULT_CONSOLIDATION_FINAL_STIFFNESS_SCALE"] == 1.0
    assert constants["DEFAULT_CONSOLIDATION_RAMP_STEPS"] == 120
    assert constants["DEFAULT_CONSOLIDATION_WINDOW_STEPS"] == 240


def test_runner_has_no_hard_coded_scale_constant():
    source = RUNNER.read_text(encoding="utf-8")
    assert "SEQUENTIAL_SOFT_HOLD_STIFFNESS_SCALE" not in source
    assert "validate_sequential_gain_consistency(" in source
    assert "sequential_config=physical_grasp.sequential" in source


def test_runner_applies_scaled_gains_per_step_in_sequential_loop():
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index(
        'if formal_grasp and arguments.physical_grasp_method'
        ' == "sequential-compliant":'
    )
    end = source.index(
        "synchronous_detectors = None",
        start,
    )
    region = source[start:end]
    update = region.index("sequential_command = sequential_controller.update(")
    gains = region.index("pick.motion.grip_hand_stiffness * stiffness_scale")
    set_gains = region.index(
        "controller.set_gains(kps=kps, kds=kds, save_to_usd=False)"
    )
    assert update < gains < set_gains
    assert "kds[hand_indices] = pick.motion.grip_hand_damping" in source


def test_runner_consistency_check_precedes_world_reset():
    source = RUNNER.read_text(encoding="utf-8")
    check = source.index("validate_sequential_gain_consistency(")
    reset = source.index("\n        world.reset()")
    assert check < reset
    assert "sequential hand-gain consistency failed closed" in source
    assert 'metrics["sequential_gain_consistency"]' in source


def test_yaml_gain_and_consolidation_values_unchanged():
    document = CONFIG.read_text(encoding="utf-8")
    for expected in (
        "approach_stiffness: 5.0",
        "soft_hold_stiffness: 1.75",
        "damping: 1.0",
        "consolidation_threshold_label: SIM_TUNING_ONLY_A_CANDIDATE",
        "soft_hold_stiffness_scale: 0.35",
        "consolidation_final_stiffness_scale: 1.0",
        "consolidation_ramp_steps: 120",
        "consolidation_window_steps: 240",
    ):
        assert expected in document, expected


def test_pre_lift_evidence_records_final_sequential_summary():
    source = RUNNER.read_text(encoding="utf-8")
    assert "sequential_final_summary = None" in source
    branch = source.index("sequential_final_summary = {")
    for key in (
        '"probe_response_nm"',
        '"normalized_loads"',
        '"normalized_load_imbalance"',
        '"balance_total_rad"',
        '"soft_hold_window_complete"',
    ):
        assert key in source[branch:branch + 2000], key
    evidence = source.index(
        '"sequential_final_summary": sequential_final_summary,'
    )
    lift = source.index('phase = "physical_grip_lift"')
    assert evidence < lift
