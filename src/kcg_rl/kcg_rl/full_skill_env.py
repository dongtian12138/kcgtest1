"""CPU-only boundary for the hierarchical D38999 full-skill task.

The deterministic backend owns perception, picking, free-space motion and
workflow transitions.  A 4-D residual action is forwarded only while the
current stage is ``ENGAGE`` or ``SCREW``.  No simulator or RL framework is
imported here, which keeps the interface independently testable.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Protocol

import numpy as np


ACTION_SIZE = 4
OBSERVATION_SIZE = 30
WORKFLOW_STAGES = (
    "DETECT_LOOSE",
    "PICK",
    "IN_HAND_RELOCALIZE",
    "DETECT_FIXED",
    "PREALIGN",
    "INSERT",
    "ENGAGE",
    "SCREW",
    "VERIFY",
    "RETREAT",
    "HOME",
)
POLICY_ACTIVE_STAGES = frozenset({"ENGAGE", "SCREW"})


class FullSkillBackend(Protocol):
    """Structural interface for a future full-skill physics backend."""

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to ``DETECT_LOOSE`` and return a strict 30-D observation."""

    def step(
        self, residual_action: np.ndarray | None
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the FSM; ``None`` means deterministic-FSM ownership."""

    def close(self) -> None:
        """Release resources owned by the backend."""


def _require_backend(backend: FullSkillBackend) -> None:
    missing = tuple(
        name
        for name in ("reset", "step", "close")
        if not callable(getattr(backend, name, None))
    )
    if missing:
        raise TypeError(
            "full-skill backend is missing callable method(s): "
            + ", ".join(missing)
        )


def _vector(value: Any, *, name: str, size: int) -> np.ndarray:
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


def _info(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("backend info must be a dict")
    return dict(value)


def _flag(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool")
    return bool(value)


def _reward(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("reward must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("reward must be finite")
    return result


def _stage(info: dict[str, Any]) -> str:
    value = info.get("workflow_stage")
    if not isinstance(value, str) or value not in WORKFLOW_STAGES:
        raise ValueError(
            "backend info.workflow_stage must be a supported workflow stage"
        )
    return value


def _decorate_info(
    info: dict[str, Any], *, stage: str, action_applied: bool
) -> dict[str, Any]:
    result = dict(info)
    result["workflow_stage"] = stage
    result["policy_active"] = stage in POLICY_ACTIVE_STAGES
    result["residual_action_applied"] = action_applied
    return result


class FullSkillBoundary:
    """Strict Gym-like lifecycle and policy-authority boundary."""

    def __init__(self, backend: FullSkillBackend):
        _require_backend(backend)
        self._backend = backend
        self._closed = False
        self._needs_reset = True
        self._stage: str | None = None

    @property
    def backend(self) -> FullSkillBackend:
        return self._backend

    @property
    def workflow_stage(self) -> str | None:
        return self._stage

    @property
    def policy_active(self) -> bool:
        return self._stage in POLICY_ACTIVE_STAGES

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
        observation, raw_info = result
        validated_observation = _vector(
            observation, name="observation", size=OBSERVATION_SIZE
        )
        validated_info = _info(raw_info)
        stage = _stage(validated_info)
        if stage != WORKFLOW_STAGES[0]:
            raise ValueError("full-skill reset must start at DETECT_LOOSE")
        self._stage = stage
        self._needs_reset = False
        return validated_observation, _decorate_info(
            validated_info, stage=stage, action_applied=False
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._require_open()
        if self._needs_reset or self._stage is None:
            raise RuntimeError("reset must be called before step")
        validated_action = _vector(action, name="action", size=ACTION_SIZE)
        action_applied = self._stage in POLICY_ACTIVE_STAGES
        backend_action = validated_action if action_applied else None
        result = self._backend.step(backend_action)
        if not isinstance(result, tuple) or len(result) != 5:
            self._invalidate_episode()
            raise TypeError(
                "backend step must return "
                "(observation, reward, terminated, truncated, info)"
            )
        observation, reward, terminated, truncated, raw_info = result
        try:
            validated_observation = _vector(
                observation, name="observation", size=OBSERVATION_SIZE
            )
            validated_reward = _reward(reward)
            validated_terminated = _flag(terminated, name="terminated")
            validated_truncated = _flag(truncated, name="truncated")
            validated_info = _info(raw_info)
            next_stage = _stage(validated_info)
            self._validate_transition(self._stage, next_stage)
        except (TypeError, ValueError):
            self._invalidate_episode()
            raise
        self._stage = next_stage
        result_info = _decorate_info(
            validated_info,
            stage=next_stage,
            action_applied=action_applied,
        )
        if validated_terminated or validated_truncated:
            self._invalidate_episode()
        return (
            validated_observation,
            validated_reward,
            validated_terminated,
            validated_truncated,
            result_info,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._invalidate_episode()
        self._backend.close()

    @staticmethod
    def _validate_transition(current: str, following: str) -> None:
        current_index = WORKFLOW_STAGES.index(current)
        following_index = WORKFLOW_STAGES.index(following)
        if following_index not in (current_index, current_index + 1):
            raise ValueError(
                "backend workflow transition must stay or advance one stage"
            )

    def _invalidate_episode(self) -> None:
        self._needs_reset = True
        self._stage = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("full-skill environment is closed")


# The boundary itself is the framework-independent environment.  This alias
# keeps the public name natural without importing Gymnasium.
FullSkillEnv = FullSkillBoundary


__all__ = [
    "ACTION_SIZE",
    "OBSERVATION_SIZE",
    "POLICY_ACTIVE_STAGES",
    "WORKFLOW_STAGES",
    "FullSkillBackend",
    "FullSkillBoundary",
    "FullSkillEnv",
]
