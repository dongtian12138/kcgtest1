"""CPU-only tests for the hierarchical full-skill boundary."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from kcg_rl.full_skill_env import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    POLICY_ACTIVE_STAGES,
    WORKFLOW_STAGES,
    FullSkillBoundary,
)


class FakeFullSkillBackend:
    def __init__(self):
        self.stage_index = 0
        self.actions = []
        self.close_calls = 0
        self.reset_result = None
        self.step_result = None

    def reset(self, seed=None, options=None):
        del seed, options
        self.stage_index = 0
        if self.reset_result is not None:
            return self.reset_result
        return (
            np.zeros(OBSERVATION_SIZE, dtype=np.float32),
            {"workflow_stage": WORKFLOW_STAGES[0]},
        )

    def step(self, action):
        self.actions.append(
            None if action is None else np.array(action, copy=True)
        )
        if self.step_result is not None:
            return self.step_result
        self.stage_index = min(self.stage_index + 1, len(WORKFLOW_STAGES) - 1)
        return (
            np.zeros(OBSERVATION_SIZE, dtype=np.float32),
            0.0,
            False,
            False,
            {"workflow_stage": WORKFLOW_STAGES[self.stage_index]},
        )

    def close(self):
        self.close_calls += 1


def _advance_to(boundary, stage):
    action = np.full(ACTION_SIZE, 0.5, dtype=np.float32)
    while boundary.workflow_stage != stage:
        boundary.step(action)


def test_contract_shapes_dtypes_ranges_and_initial_stage():
    backend = FakeFullSkillBackend()
    boundary = FullSkillBoundary(backend)
    observation, info = boundary.reset(seed=9, options={})
    assert ACTION_SIZE == 4
    assert OBSERVATION_SIZE == 30
    assert observation.shape == (30,)
    assert observation.dtype == np.float32
    assert info == {
        "workflow_stage": "DETECT_LOOSE",
        "policy_active": False,
        "residual_action_applied": False,
    }


def test_policy_action_is_forwarded_only_during_engage_and_screw():
    backend = FakeFullSkillBackend()
    boundary = FullSkillBoundary(backend)
    boundary.reset()
    action = np.asarray([0.2, -0.3, 0.4, -0.5], dtype=np.float32)
    applied_by_stage = {}
    while boundary.workflow_stage != "VERIFY":
        current_stage = boundary.workflow_stage
        _, _, _, _, info = boundary.step(action)
        applied_by_stage[current_stage] = backend.actions[-1]
        assert info["residual_action_applied"] == (
            current_stage in POLICY_ACTIVE_STAGES
        )
    assert set(POLICY_ACTIVE_STAGES) == {"ENGAGE", "SCREW"}
    for stage, applied in applied_by_stage.items():
        if stage in POLICY_ACTIVE_STAGES:
            assert np.array_equal(applied, action)
            assert applied is not action
        else:
            assert applied is None


@pytest.mark.parametrize(
    "action, error, message",
    (
        (np.zeros(3, dtype=np.float32), ValueError, "shape"),
        (np.zeros(4, dtype=np.float64), TypeError, "dtype"),
        (
            np.asarray([0.0, np.nan, 0.0, 0.0], dtype=np.float32),
            ValueError,
            "finite",
        ),
        (
            np.asarray([0.0, 1.01, 0.0, 0.0], dtype=np.float32),
            ValueError,
            "within",
        ),
        ([0.0] * 4, TypeError, "numpy.ndarray"),
    ),
)
def test_strict_action_boundary(action, error, message):
    backend = FakeFullSkillBackend()
    boundary = FullSkillBoundary(backend)
    boundary.reset()
    with pytest.raises(error, match=message):
        boundary.step(action)
    assert backend.actions == []


@pytest.mark.parametrize(
    "observation, error, message",
    (
        (np.zeros(29, dtype=np.float32), ValueError, "shape"),
        (np.zeros(30, dtype=np.float64), TypeError, "dtype"),
        (np.full(30, np.inf, dtype=np.float32), ValueError, "finite"),
        (np.full(30, -1.01, dtype=np.float32), ValueError, "within"),
    ),
)
def test_strict_observation_boundary(observation, error, message):
    backend = FakeFullSkillBackend()
    backend.reset_result = observation, {"workflow_stage": "DETECT_LOOSE"}
    with pytest.raises(error, match=message):
        FullSkillBoundary(backend).reset()


def test_reset_stage_and_fsm_transitions_are_strict():
    backend = FakeFullSkillBackend()
    backend.reset_result = (
        np.zeros(30, dtype=np.float32),
        {"workflow_stage": "ENGAGE"},
    )
    with pytest.raises(ValueError, match="DETECT_LOOSE"):
        FullSkillBoundary(backend).reset()

    backend = FakeFullSkillBackend()
    boundary = FullSkillBoundary(backend)
    boundary.reset()
    backend.step_result = (
        np.zeros(30, dtype=np.float32),
        0.0,
        False,
        False,
        {"workflow_stage": "ENGAGE"},
    )
    with pytest.raises(ValueError, match="advance one"):
        boundary.step(np.zeros(4, dtype=np.float32))
    with pytest.raises(RuntimeError, match="reset"):
        boundary.step(np.zeros(4, dtype=np.float32))


def test_lifecycle_requires_reset_after_terminal_and_close_is_idempotent():
    backend = FakeFullSkillBackend()
    boundary = FullSkillBoundary(backend)
    with pytest.raises(RuntimeError, match="reset"):
        boundary.step(np.zeros(4, dtype=np.float32))
    boundary.reset()
    backend.step_result = (
        np.zeros(30, dtype=np.float32),
        np.float32(0.0),
        np.bool_(True),
        np.bool_(False),
        {"workflow_stage": "DETECT_LOOSE"},
    )
    assert boundary.step(np.zeros(4, dtype=np.float32))[2] is True
    with pytest.raises(RuntimeError, match="reset"):
        boundary.step(np.zeros(4, dtype=np.float32))
    boundary.close()
    boundary.close()
    assert backend.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        boundary.reset()


def test_module_import_stays_cpu_only_and_does_not_import_gym():
    package_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)
    script = """
import importlib
import sys

module = importlib.import_module("kcg_rl.full_skill_env")
assert module.ACTION_SIZE == 4
assert module.OBSERVATION_SIZE == 30
for name in ("gym", "gymnasium", "torch", "stable_baselines3",
             "omni", "isaacsim", "rclpy"):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
