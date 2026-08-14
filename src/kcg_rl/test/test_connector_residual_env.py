"""Pure tests for the connector residual Gymnasium boundary."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from kcg_rl.connector_residual_env import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    ConnectorResidualBoundary,
    ConnectorResidualEnv,
)


class FakeBackend:
    def __init__(self):
        self.reset_calls = 0
        self.reset_arguments = []
        self.step_actions = []
        self.close_calls = 0
        self.reset_result = (
            np.zeros(OBSERVATION_SIZE, dtype=np.float32),
            {"reset": "fake"},
        )
        self.step_result = (
            np.full(OBSERVATION_SIZE, 0.25, dtype=np.float32),
            1.5,
            False,
            False,
            {"step": 1},
        )

    def reset(self, seed=None, options=None):
        self.reset_calls += 1
        self.reset_arguments.append((seed, options))
        return self.reset_result

    def step(self, action):
        self.step_actions.append(action)
        return self.step_result

    def close(self):
        self.close_calls += 1


def test_boundary_forwards_one_canonical_action_and_results():
    backend = FakeBackend()
    boundary = ConnectorResidualBoundary(backend)
    observation, info = boundary.reset()
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert info == {"reset": "fake"}

    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    result = boundary.step(action)
    assert result[0].shape == (OBSERVATION_SIZE,)
    assert result[0].dtype == np.float32
    assert result[1:] == (1.5, False, False, {"step": 1})
    assert backend.step_actions[0] is not action
    assert np.array_equal(backend.step_actions[0], action)


def test_boundary_forwards_reset_seed_and_options():
    backend = FakeBackend()
    boundary = ConnectorResidualBoundary(backend)
    boundary.reset(seed=23, options={})
    assert backend.reset_arguments == [(23, {})]


def test_gym_environment_seeds_spaces_and_backend():
    pytest.importorskip("gymnasium")
    backend = FakeBackend()
    environment = ConnectorResidualEnv(backend)
    environment.reset(seed=71, options={})
    first_action = environment.action_space.sample()
    environment.reset(seed=71, options={})
    second_action = environment.action_space.sample()
    assert np.array_equal(first_action, second_action)
    assert backend.reset_arguments == [(71, {}), (71, {})]
    environment.close()


@pytest.mark.parametrize(
    "action, error, message",
    (
        (np.zeros(3, dtype=np.float32), ValueError, "shape"),
        (np.zeros(ACTION_SIZE, dtype=np.float64), TypeError, "dtype"),
        (
            np.asarray([0.0, 0.0, np.nan, 0.0], dtype=np.float32),
            ValueError,
            "finite",
        ),
        (
            np.asarray([0.0, 0.0, 1.01, 0.0], dtype=np.float32),
            ValueError,
            "within",
        ),
        ([0.0] * ACTION_SIZE, TypeError, "numpy.ndarray"),
    ),
)
def test_boundary_rejects_invalid_actions(action, error, message):
    backend = FakeBackend()
    boundary = ConnectorResidualBoundary(backend)
    boundary.reset()
    with pytest.raises(error, match=message):
        boundary.step(action)
    assert backend.step_actions == []


@pytest.mark.parametrize(
    "observation, error, message",
    (
        (np.zeros(23, dtype=np.float32), ValueError, "shape"),
        (np.zeros(OBSERVATION_SIZE, dtype=np.float64), TypeError, "dtype"),
        (
            np.full(OBSERVATION_SIZE, np.inf, dtype=np.float32),
            ValueError,
            "finite",
        ),
        (
            np.full(OBSERVATION_SIZE, -1.01, dtype=np.float32),
            ValueError,
            "within",
        ),
    ),
)
def test_boundary_rejects_invalid_reset_observations(
    observation, error, message
):
    backend = FakeBackend()
    backend.reset_result = observation, {}
    boundary = ConnectorResidualBoundary(backend)
    with pytest.raises(error, match=message):
        boundary.reset()


@pytest.mark.parametrize(
    "step_result, error, message",
    (
        (
            (np.zeros(OBSERVATION_SIZE, dtype=np.float32), np.inf,
             False, False, {}),
            ValueError,
            "reward",
        ),
        (
            (np.zeros(OBSERVATION_SIZE, dtype=np.float32), 0.0,
             1, False, {}),
            TypeError,
            "terminated",
        ),
        (
            (np.zeros(OBSERVATION_SIZE, dtype=np.float32), 0.0,
             False, False, []),
            TypeError,
            "info",
        ),
        ((np.zeros(OBSERVATION_SIZE, dtype=np.float32), 0.0),
         TypeError, "backend step"),
    ),
)
def test_boundary_rejects_invalid_step_results(step_result, error, message):
    backend = FakeBackend()
    backend.step_result = step_result
    boundary = ConnectorResidualBoundary(backend)
    boundary.reset()
    with pytest.raises(error, match=message):
        boundary.step(np.zeros(ACTION_SIZE, dtype=np.float32))
    with pytest.raises(RuntimeError, match="reset"):
        boundary.step(np.zeros(ACTION_SIZE, dtype=np.float32))


def test_boundary_enforces_reset_and_idempotent_close():
    backend = FakeBackend()
    boundary = ConnectorResidualBoundary(backend)
    with pytest.raises(RuntimeError, match="reset"):
        boundary.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    boundary.reset()
    backend.step_result = (
        np.zeros(OBSERVATION_SIZE, dtype=np.float32),
        np.float32(0.0),
        np.bool_(True),
        np.bool_(False),
        {},
    )
    assert boundary.step(np.zeros(ACTION_SIZE, dtype=np.float32))[2]
    with pytest.raises(RuntimeError, match="reset"):
        boundary.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    boundary.close()
    boundary.close()
    assert backend.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        boundary.reset()


@pytest.mark.parametrize("missing", ("reset", "step", "close"))
def test_boundary_requires_all_backend_methods(missing):
    backend = FakeBackend()
    setattr(backend, missing, None)
    with pytest.raises(TypeError, match=missing):
        ConnectorResidualBoundary(backend)


def test_module_import_does_not_load_ros_or_isaac_modules():
    package_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)
    script = """
import importlib
import sys

module = importlib.import_module("kcg_rl.connector_residual_env")
assert module.ACTION_SIZE == 4
assert module.OBSERVATION_SIZE == 24
for name in ("rclpy", "omni", "isaacsim"):
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
