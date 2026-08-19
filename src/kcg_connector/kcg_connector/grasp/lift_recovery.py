"""Bounded sensor-only recovery planning for a failed staged lift.

When the staged-lift sensor gate fails closed, the episode replays the
already-commanded arm targets in reverse (so the robot returns along the
trajectory it actually traversed), settles at the pre-lift target, and then
opens the fingers on a minimum-jerk profile.  This module is pure planning:
it accepts only robot targets and returns only robot targets.  It must never
see object pose, contact reports, colliders, or contact normals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LiftRecoveryConfig:
    return_steps_per_waypoint: int
    settle_steps: int
    open_duration_s: float

    def __post_init__(self) -> None:
        if self.return_steps_per_waypoint < 1 or self.settle_steps < 1:
            raise ValueError("recovery step counts must be positive integers")
        if not math.isfinite(self.open_duration_s) or self.open_duration_s <= 0.0:
            raise ValueError("open_duration_s must be positive and finite")


def _arm_waypoint(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be one finite 7-vector")
    return result


def _hand_target(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be one finite 4-vector")
    return result


def _minimum_jerk(fraction: float) -> float:
    return fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))


def plan_recovery_return(
    traversed_waypoints: Sequence[Sequence[float]],
    prelift_target: Sequence[float],
    config: LiftRecoveryConfig,
) -> tuple[np.ndarray, ...]:
    """Reverse the traversed arm targets and finish at the pre-lift target.

    Every traversed waypoint is replayed ``return_steps_per_waypoint`` times,
    so the return is at most half as fast as the forward motion and always
    bounded by the forward step count.  The plan ends with ``settle_steps``
    copies of ``prelift_target`` so the payload can resettle before opening.
    An empty traversal (zero-lift hold) yields a settle-only plan at the
    pre-lift target.
    """

    traversed = tuple(
        _arm_waypoint(waypoint, "traversed arm waypoint")
        for waypoint in traversed_waypoints
    )
    target = _arm_waypoint(prelift_target, "pre-lift target")
    waypoints: list[np.ndarray] = []
    if not traversed:
        # Zero-lift hold characterization: the arm command was held constant,
        # so the only safe return is settling at the pre-lift target.
        for _ in range(config.settle_steps):
            waypoints.append(target.copy())
        return tuple(waypoints)
    for waypoint in reversed(traversed):
        for _ in range(config.return_steps_per_waypoint):
            waypoints.append(waypoint.copy())
    for _ in range(config.settle_steps):
        waypoints.append(target.copy())
    return tuple(waypoints)


def plan_recovery_open(
    held_hand: Sequence[float],
    open_hand: Sequence[float],
    duration_s: float,
    rate_hz: float,
) -> tuple[np.ndarray, ...]:
    """Plan a bounded minimum-jerk finger opening from held to open targets."""

    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("open duration must be positive and finite")
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("physics rate must be positive and finite")
    held = _hand_target(held_hand, "held hand target")
    open_target = _hand_target(open_hand, "open hand target")
    steps = max(1, round(duration_s * rate_hz))
    return tuple(
        held
        + _minimum_jerk(float(index + 1) / float(steps))
        * (open_target - held)
        for index in range(steps)
    )
