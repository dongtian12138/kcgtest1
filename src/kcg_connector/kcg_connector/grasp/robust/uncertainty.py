"""Deterministic low-discrepancy scenarios and lower-tail risk statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import qmc


def _readonly(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.float64)
    value.setflags(write=False)
    return value


def scrambled_sobol_unit_scenarios(
    *, dimension: int, scenario_count: int, seed: int
) -> np.ndarray:
    """Generate a reproducible balanced scrambled Sobol design on [0, 1)^d.

    A power-of-two count is required because ``random_base2`` preserves the
    balance properties used by the preregistered 128/256-scenario protocol.
    """

    if isinstance(dimension, bool) or int(dimension) != dimension or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if (
        isinstance(scenario_count, bool)
        or int(scenario_count) != scenario_count
        or scenario_count < 1
        or int(scenario_count) & (int(scenario_count) - 1)
    ):
        raise ValueError("scenario_count must be a positive power of two")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    exponent = int(math.log2(int(scenario_count)))
    sampler = qmc.Sobol(d=int(dimension), scramble=True, seed=int(seed))
    return _readonly(sampler.random_base2(m=exponent))


@dataclass(frozen=True)
class SobolScenarioSet:
    """Named hyper-rectangular parameter scenarios with their unit design."""

    parameter_names: tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    unit_scenarios: np.ndarray
    values: np.ndarray
    seed: int

    @property
    def scenario_count(self) -> int:
        return int(self.values.shape[0])

    def records(self) -> tuple[dict[str, float], ...]:
        return tuple(
            {
                name: float(row[index])
                for index, name in enumerate(self.parameter_names)
            }
            for row in self.values
        )


def scrambled_sobol_scenarios(
    parameter_bounds: Mapping[str, Sequence[float]],
    *,
    scenario_count: int,
    seed: int,
) -> SobolScenarioSet:
    """Map a scrambled Sobol design through declared independent intervals.

    Parameter names are sorted so the same contract is reproducible regardless
    of mapping insertion order.  More general marginals can use the returned
    unit design and apply their preregistered inverse distribution transforms.
    """

    if not parameter_bounds:
        raise ValueError("parameter_bounds cannot be empty")
    names = tuple(sorted(parameter_bounds))
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("parameter names must be non-empty strings")
    lower: list[float] = []
    upper: list[float] = []
    for name in names:
        bounds = np.asarray(parameter_bounds[name], dtype=np.float64)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError(f"parameter_bounds[{name!r}] must contain two finite values")
        if float(bounds[1]) < float(bounds[0]):
            raise ValueError(f"parameter_bounds[{name!r}] has upper < lower")
        lower.append(float(bounds[0]))
        upper.append(float(bounds[1]))
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    unit = scrambled_sobol_unit_scenarios(
        dimension=len(names), scenario_count=scenario_count, seed=seed
    )
    values = lower_array + unit * (upper_array - lower_array)
    return SobolScenarioSet(
        parameter_names=names,
        lower_bounds=_readonly(lower_array),
        upper_bounds=_readonly(upper_array),
        unit_scenarios=unit,
        values=_readonly(values),
        seed=int(seed),
    )


def lower_tail_cvar(
    margins: Sequence[float], *, lower_tail_fraction: float
) -> float:
    """Exact empirical mean over the lowest ``alpha`` probability mass.

    Equal mass is assigned to every scenario.  When ``alpha * N`` is not an
    integer, the boundary order statistic contributes only the required
    fractional mass.  This is the finite-sample form of

        max_t [t - E[(t - rho)_+] / alpha].
    """

    values = np.asarray(margins, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("margins must be one non-empty finite vector")
    alpha = float(lower_tail_fraction)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("lower_tail_fraction must lie in (0, 1]")
    ordered = np.sort(values)
    tail_mass_in_samples = alpha * values.size
    full_count = int(math.floor(tail_mass_in_samples))
    fractional_count = tail_mass_in_samples - full_count
    total = float(np.sum(ordered[:full_count]))
    if fractional_count > 0.0:
        total += fractional_count * float(ordered[full_count])
    return total / tail_mass_in_samples


@dataclass(frozen=True)
class LowerTailRiskSummary:
    scenario_count: int
    lower_tail_fraction: float
    lower_tail_cvar: float
    lower_tail_var: float
    hard_minimum: float
    mean_margin: float


def summarize_lower_tail_risk(
    margins: Sequence[float], *, lower_tail_fraction: float
) -> LowerTailRiskSummary:
    """Report CVaR together with the independently interpretable hard minimum."""

    values = np.asarray(margins, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("margins must be one non-empty finite vector")
    alpha = float(lower_tail_fraction)
    cvar = lower_tail_cvar(values, lower_tail_fraction=alpha)
    ordered = np.sort(values)
    var_index = min(values.size - 1, int(math.ceil(alpha * values.size)) - 1)
    return LowerTailRiskSummary(
        scenario_count=int(values.size),
        lower_tail_fraction=alpha,
        lower_tail_cvar=cvar,
        lower_tail_var=float(ordered[var_index]),
        hard_minimum=float(ordered[0]),
        mean_margin=float(np.mean(values)),
    )
