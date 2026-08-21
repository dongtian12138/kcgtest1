"""Unit-safe lexicographic and Pareto ranking for CARTS-Grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Generic, Iterable, Sequence, TypeVar

import numpy as np


CandidateT = TypeVar("CandidateT")


@dataclass(frozen=True)
class CandidateMetrics:
    """Pre-registered candidate metrics in their physical units.

    The certified interval minimum is the primary maximand.  The lower-tail
    mean over the finite QMC design is only a secondary interval-sensitivity
    statistic.  Clearance is maximised; required normal force and joint-torque
    utilisation are minimised.  The quantities are intentionally not
    normalised into, or combined by, a weighted sum.
    """

    hard_bound_minimum_task_margin: float
    qmc_lower_tail_mean_task_margin: float
    peak_normal_force_n: float
    joint_torque_utilization: float
    trajectory_clearance_m: float

    def __post_init__(self) -> None:
        values = (
            self.hard_bound_minimum_task_margin,
            self.qmc_lower_tail_mean_task_margin,
            self.peak_normal_force_n,
            self.joint_torque_utilization,
            self.trajectory_clearance_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("candidate metrics must be finite")
        if self.peak_normal_force_n < 0.0:
            raise ValueError("peak normal force cannot be negative")
        if self.joint_torque_utilization < 0.0:
            raise ValueError("joint torque utilization cannot be negative")

    def lexicographic_key(self) -> tuple[float, float, float, float, float]:
        """Ascending sort key implementing the frozen selection order."""

        return (
            -float(self.hard_bound_minimum_task_margin),
            -float(self.qmc_lower_tail_mean_task_margin),
            float(self.peak_normal_force_n),
            float(self.joint_torque_utilization),
            -float(self.trajectory_clearance_m),
        )

    def desirability_vector(self) -> tuple[float, float, float, float, float]:
        """Return components oriented so that larger is always better."""

        return (
            float(self.hard_bound_minimum_task_margin),
            float(self.qmc_lower_tail_mean_task_margin),
            -float(self.peak_normal_force_n),
            -float(self.joint_torque_utilization),
            float(self.trajectory_clearance_m),
        )


@dataclass(frozen=True)
class ScoredCandidate(Generic[CandidateT]):
    candidate: CandidateT
    metrics: CandidateMetrics
    source: str = "UNSPECIFIED"
    source_index: int = 0


@dataclass(frozen=True)
class RankedCandidate(Generic[CandidateT]):
    candidate: CandidateT
    metrics: CandidateMetrics
    source: str
    source_index: int
    rank: int
    pareto_layer: int


def qmc_lower_tail_mean(samples: Sequence[float], tail_fraction: float) -> float:
    """Lower-tail mean of a finite deterministic QMC scenario design.

    Fractional sample counts are handled by assigning the boundary order
    statistic its fractional design weight.  This is an order statistic of the
    supplied design only; it does not introduce a probability distribution.
    """

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "QMC samples must be a non-empty one-dimensional sequence"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("QMC samples must be finite")
    if not math.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")

    ordered = np.sort(values)
    effective_count = float(tail_fraction) * float(ordered.size)
    complete_count = int(math.floor(effective_count))
    fractional_count = effective_count - complete_count
    total = float(np.sum(ordered[:complete_count], dtype=np.float64))
    if fractional_count > 0.0:
        total += fractional_count * float(ordered[complete_count])
    return total / effective_count


def lower_tail_cvar(samples: Sequence[float], tail_fraction: float) -> float:
    """Compatibility wrapper for the same finite-sample lower-tail arithmetic.

    Production candidate ranking uses :func:`qmc_lower_tail_mean` because the
    bounded friction contract does not claim a probability distribution.
    """

    return qmc_lower_tail_mean(samples, tail_fraction)


def dominates(left: CandidateMetrics, right: CandidateMetrics) -> bool:
    """Return true when ``left`` Pareto-dominates ``right`` exactly."""

    left_values = left.desirability_vector()
    right_values = right.desirability_vector()
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_layers(metrics: Sequence[CandidateMetrics]) -> tuple[tuple[int, ...], ...]:
    """Deterministically compute non-dominated layers by input index."""

    remaining = list(range(len(metrics)))
    layers: list[tuple[int, ...]] = []
    while remaining:
        front = tuple(
            index
            for index in remaining
            if not any(
                other != index and dominates(metrics[other], metrics[index])
                for other in remaining
            )
        )
        if not front:  # Defensive guard; strict dominance cannot form a cycle.
            raise RuntimeError("Pareto layering failed to produce a front")
        layers.append(front)
        front_set = set(front)
        remaining = [index for index in remaining if index not in front_set]
    return tuple(layers)


def rank_candidates(
    candidates: Iterable[ScoredCandidate[CandidateT]],
) -> tuple[RankedCandidate[CandidateT], ...]:
    """Rank candidates by exact lexicographic order with stable tie breaking."""

    rows = tuple(candidates)
    if not rows:
        return ()
    layers = pareto_layers(tuple(row.metrics for row in rows))
    layer_by_index = {
        index: layer_number
        for layer_number, layer in enumerate(layers, start=1)
        for index in layer
    }
    ordered_indices = sorted(
        range(len(rows)),
        key=lambda index: (
            rows[index].metrics.lexicographic_key(),
            int(rows[index].source_index),
            index,
        ),
    )
    return tuple(
        RankedCandidate(
            candidate=rows[index].candidate,
            metrics=rows[index].metrics,
            source=rows[index].source,
            source_index=rows[index].source_index,
            rank=rank,
            pareto_layer=layer_by_index[index],
        )
        for rank, index in enumerate(ordered_indices, start=1)
    )


def lexicographically_better(left: CandidateMetrics, right: CandidateMetrics) -> bool:
    """Strict comparator used by deterministic continuous refinement."""

    return left.lexicographic_key() < right.lexicographic_key()


__all__ = [
    "CandidateMetrics",
    "RankedCandidate",
    "ScoredCandidate",
    "dominates",
    "lexicographically_better",
    "lower_tail_cvar",
    "pareto_layers",
    "qmc_lower_tail_mean",
    "rank_candidates",
]
