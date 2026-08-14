"""Strict Gymnasium boundary for the connector residual task.

The physics backend is injected through a small structural protocol.  Keeping
that boundary free of Isaac Sim and ROS imports lets this module load in both
the Isaac Sim Python 3.12 runtime and ordinary unit-test interpreters.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Protocol

import numpy as np


ACTION_SIZE = 4
OBSERVATION_SIZE = 24


class ConnectorResidualBackend(Protocol):
    """Structural interface implemented by a connector physics adapter."""

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Hard-reset the physics task and return observation and metadata."""

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one policy step using a normalized residual action."""

    def close(self) -> None:
        """Release resources owned by the backend."""


def _require_backend(backend: ConnectorResidualBackend) -> None:
    missing = tuple(
        name
        for name in ("reset", "step", "close")
        if not callable(getattr(backend, name, None))
    )
    if missing:
        raise TypeError(
            "connector residual backend is missing callable method(s): "
            + ", ".join(missing)
        )


def _validated_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if value.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have dtype float32")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(value < -1.0) or np.any(value > 1.0):
        raise ValueError(f"{name} values must stay within [-1, 1]")
    return np.array(value, dtype=np.float32, copy=True)


def _validated_info(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("backend info must be a dict")
    return dict(value)


def _validated_flag(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool")
    return bool(value)


def _validated_reward(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("reward must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("reward must be finite")
    return result


class ConnectorResidualBoundary:
    """Framework-independent validation and lifecycle boundary."""

    def __init__(self, backend: ConnectorResidualBackend):
        _require_backend(backend)
        self._backend = backend
        self._closed = False
        self._needs_reset = True

    @property
    def backend(self) -> ConnectorResidualBackend:
        return self._backend

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._require_open()
        result = self._backend.reset(seed=seed, options=options)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("backend reset must return (observation, info)")
        observation, info = result
        validated = _validated_vector(
            observation,
            name="observation",
            size=OBSERVATION_SIZE,
        )
        validated_info = _validated_info(info)
        self._needs_reset = False
        return validated, validated_info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._require_open()
        if self._needs_reset:
            raise RuntimeError("reset must be called before step")
        validated_action = _validated_vector(
            action,
            name="action",
            size=ACTION_SIZE,
        )
        result = self._backend.step(validated_action)
        if not isinstance(result, tuple) or len(result) != 5:
            self._needs_reset = True
            raise TypeError(
                "backend step must return "
                "(observation, reward, terminated, truncated, info)"
            )
        observation, reward, terminated, truncated, info = result
        try:
            validated_observation = _validated_vector(
                observation,
                name="observation",
                size=OBSERVATION_SIZE,
            )
            validated_reward = _validated_reward(reward)
            validated_terminated = _validated_flag(
                terminated, name="terminated"
            )
            validated_truncated = _validated_flag(
                truncated, name="truncated"
            )
            validated_info = _validated_info(info)
        except (TypeError, ValueError):
            self._needs_reset = True
            raise
        if validated_terminated or validated_truncated:
            self._needs_reset = True
        return (
            validated_observation,
            validated_reward,
            validated_terminated,
            validated_truncated,
            validated_info,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._needs_reset = True
        self._backend.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("connector residual environment is closed")


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - base ROS environment has no Gymnasium
    gym = None
    spaces = None


if gym is not None:

    class ConnectorResidualEnv(gym.Env):
        """Gymnasium facade over one connector residual physics backend."""

        metadata = {"render_modes": []}

        def __init__(self, backend: ConnectorResidualBackend):
            super().__init__()
            self._boundary = ConnectorResidualBoundary(backend)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(ACTION_SIZE,),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(OBSERVATION_SIZE,),
                dtype=np.float32,
            )

        @property
        def backend(self) -> ConnectorResidualBackend:
            return self._boundary.backend

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.action_space.seed(seed)
                self.observation_space.seed(seed)
            return self._boundary.reset(seed=seed, options=options)

        def step(self, action):
            return self._boundary.step(action)

        def close(self):
            self._boundary.close()

else:

    class ConnectorResidualEnv:  # pragma: no cover - clear optional error
        """Placeholder used when the optional Gymnasium package is absent."""

        def __init__(self, backend: ConnectorResidualBackend):
            del backend
            raise ImportError(
                "Gymnasium is required for ConnectorResidualEnv; install the "
                "pinned Isaac RL requirements in the Isaac Python runtime"
            )


__all__ = [
    "ACTION_SIZE",
    "OBSERVATION_SIZE",
    "ConnectorResidualBackend",
    "ConnectorResidualBoundary",
    "ConnectorResidualEnv",
]
