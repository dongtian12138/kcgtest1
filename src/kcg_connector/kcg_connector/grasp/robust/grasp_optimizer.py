"""Deterministic Sobol/multistart optimisation skeleton for CARTS-Grasp."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
import warnings

import numpy as np

from kcg_connector.grasp.robust.hand_model import (
    HandModelError,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.pareto_ranker import (
    CandidateMetrics,
    RankedCandidate,
    ScoredCandidate,
    lexicographically_better,
    qmc_lower_tail_mean,
    rank_candidates,
)


class OptimizationError(RuntimeError):
    """Raised when the optimisation protocol cannot produce a feasible grasp."""


def _finite_tuple(
    value: Sequence[float], *, length: int | None, label: str
) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if length is not None and len(result) != length:
        raise ValueError(f"{label} must contain {length} values")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


@dataclass(frozen=True)
class PlannedPadContact:
    """A planned finite-PAD contact expressed in the object contract frame.

    The normal points toward the path-local side occupied immediately before
    the certified transverse root.  It is not a global solid-outward claim.
    """

    pad_name: str
    position_object_m: tuple[float, float, float]
    path_local_free_side_normal_object: tuple[float, float, float]
    surface_coordinates: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.pad_name:
            raise ValueError("planned contact must name a PAD")
        position = _finite_tuple(
            self.position_object_m, length=3, label="contact position"
        )
        normal = np.asarray(
            _finite_tuple(
                self.path_local_free_side_normal_object,
                length=3,
                label="path-local free-side contact normal",
            ),
            dtype=np.float64,
        )
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= np.finfo(np.float64).eps:
            raise ValueError("contact normal must be non-zero")
        object.__setattr__(self, "position_object_m", position)
        object.__setattr__(
            self,
            "path_local_free_side_normal_object",
            tuple(float(item) for item in normal / normal_norm),
        )
        object.__setattr__(
            self,
            "surface_coordinates",
            _finite_tuple(
                self.surface_coordinates,
                length=None,
                label="surface coordinates",
            ),
        )


@dataclass(frozen=True)
class GraspCandidate:
    """Object-independent optimisation decision returned by a surface model."""

    object_from_hand: tuple[float, ...]
    independent_joint_positions_rad: tuple[float, ...]
    planned_pad_contacts: tuple[PlannedPadContact, ...]
    internal_normal_forces_n: tuple[float, ...]
    stiffness_diagonal: tuple[float, ...] = ()
    damping_diagonal: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        transform_values = _finite_tuple(
            self.object_from_hand, length=16, label="object_from_hand"
        )
        transform = np.asarray(transform_values, dtype=np.float64).reshape(4, 4)
        homogeneous_error = float(
            np.linalg.norm(transform[3] - np.asarray((0.0, 0.0, 0.0, 1.0)))
        )
        numerical_tolerance = 64.0 * np.finfo(np.float64).eps
        if homogeneous_error > numerical_tolerance:
            raise ValueError("object_from_hand must be a homogeneous transform")
        rotation_error = float(
            np.linalg.norm(transform[:3, :3].T @ transform[:3, :3] - np.eye(3))
        )
        determinant_error = abs(float(np.linalg.det(transform[:3, :3])) - 1.0)
        if rotation_error > numerical_tolerance or determinant_error > numerical_tolerance:
            raise ValueError("object_from_hand rotation must be orthonormal and proper")
        joints = _finite_tuple(
            self.independent_joint_positions_rad,
            length=None,
            label="independent joint positions",
        )
        contacts = tuple(self.planned_pad_contacts)
        forces = _finite_tuple(
            self.internal_normal_forces_n,
            length=len(contacts),
            label="internal normal forces",
        )
        if len(contacts) != 3:
            raise ValueError("a three-finger grasp must plan exactly three PAD contacts")
        if any(force < 0.0 for force in forces):
            raise ValueError("internal normal forces cannot be negative")
        if len({contact.pad_name for contact in contacts}) != len(contacts):
            raise ValueError("planned PAD contacts must be unique")
        stiffness = _finite_tuple(
            self.stiffness_diagonal, length=None, label="stiffness diagonal"
        )
        damping = _finite_tuple(
            self.damping_diagonal, length=None, label="damping diagonal"
        )
        if any(item < 0.0 for item in stiffness + damping):
            raise ValueError("stiffness and damping entries cannot be negative")
        object.__setattr__(self, "object_from_hand", transform_values)
        object.__setattr__(self, "independent_joint_positions_rad", joints)
        object.__setattr__(self, "planned_pad_contacts", contacts)
        object.__setattr__(self, "internal_normal_forces_n", forces)
        object.__setattr__(self, "stiffness_diagonal", stiffness)
        object.__setattr__(self, "damping_diagonal", damping)

    @classmethod
    def from_matrix(
        cls,
        *,
        object_from_hand: np.ndarray,
        independent_joint_positions_rad: Sequence[float],
        planned_pad_contacts: Sequence[PlannedPadContact],
        internal_normal_forces_n: Sequence[float],
        stiffness_diagonal: Sequence[float] = (),
        damping_diagonal: Sequence[float] = (),
    ) -> "GraspCandidate":
        transform = np.asarray(object_from_hand, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("object_from_hand matrix must have shape (4, 4)")
        return cls(
            object_from_hand=tuple(float(item) for item in transform.ravel()),
            independent_joint_positions_rad=tuple(independent_joint_positions_rad),
            planned_pad_contacts=tuple(planned_pad_contacts),
            internal_normal_forces_n=tuple(internal_normal_forces_n),
            stiffness_diagonal=tuple(stiffness_diagonal),
            damping_diagonal=tuple(damping_diagonal),
        )

    def object_from_hand_matrix(self) -> np.ndarray:
        return np.asarray(self.object_from_hand, dtype=np.float64).reshape(4, 4)


@dataclass(frozen=True)
class WrenchEvaluation:
    """Scenario margins and physical burden metrics from a wrench evaluator."""

    task_margins: tuple[float, ...]
    peak_normal_force_n: float
    joint_torque_utilization: float
    trajectory_clearance_m: float
    hard_bound_minimum_task_margin: float | None = None
    feasible: bool = True
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        margins = _finite_tuple(
            self.task_margins, length=None, label="task wrench margins"
        )
        if not margins:
            raise ValueError("wrench evaluation requires at least one scenario margin")
        scalar_values = (
            self.peak_normal_force_n,
            self.joint_torque_utilization,
            self.trajectory_clearance_m,
        )
        if not all(math.isfinite(float(value)) for value in scalar_values):
            raise ValueError("wrench burden metrics must be finite")
        if self.peak_normal_force_n < 0.0:
            raise ValueError("peak normal force cannot be negative")
        if self.joint_torque_utilization < 0.0:
            raise ValueError("joint torque utilization cannot be negative")
        hard_minimum = self.hard_bound_minimum_task_margin
        if hard_minimum is not None and not math.isfinite(float(hard_minimum)):
            raise ValueError("hard-bound task margin must be finite when provided")
        object.__setattr__(self, "task_margins", margins)


@runtime_checkable
class ObjectSurfaceModel(Protocol):
    """Protocol mapping a unit hypercube point to a physical grasp decision."""

    @property
    def parameter_dimension(self) -> int:
        ...

    def candidate_from_unit_parameters(
        self,
        parameters_unit: np.ndarray,
        hand_model: ThreeFingerHandModel,
    ) -> GraspCandidate | None:
        """Return ``None`` only when geometry makes the sampled point infeasible."""


@runtime_checkable
class WrenchEvaluator(Protocol):
    """Protocol for robust task-wrench and actuation evaluation."""

    @property
    def uncertainty_dimension(self) -> int:
        ...

    def evaluate(
        self,
        candidate: GraspCandidate,
        scenario_parameters_unit: np.ndarray,
        *,
        surface_model: ObjectSurfaceModel,
        hand_model: ThreeFingerHandModel,
    ) -> WrenchEvaluation:
        ...


@dataclass(frozen=True)
class OptimizationConfig:
    candidate_budget: int
    continuous_refinement_multistarts: int
    candidate_sobol_seed: int
    maximum_solver_iterations: int
    relative_objective_tolerance: float
    scenario_count: int
    scenario_sobol_seed: int
    lower_tail_fraction: float

    def __post_init__(self) -> None:
        integer_values = (
            self.candidate_budget,
            self.maximum_solver_iterations,
            self.scenario_count,
        )
        if any(not isinstance(value, int) or value <= 0 for value in integer_values):
            raise ValueError("candidate, iteration and scenario counts must be positive integers")
        if (
            not isinstance(self.continuous_refinement_multistarts, int)
            or self.continuous_refinement_multistarts < 0
        ):
            raise ValueError("continuous_refinement_multistarts cannot be negative")
        if self.continuous_refinement_multistarts > self.candidate_budget:
            raise ValueError("continuous refinement starts cannot exceed candidate budget")
        for label, seed in (
            ("candidate_sobol_seed", self.candidate_sobol_seed),
            ("scenario_sobol_seed", self.scenario_sobol_seed),
        ):
            if not isinstance(seed, int) or seed < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if (
            not math.isfinite(self.relative_objective_tolerance)
            or self.relative_objective_tolerance <= 0.0
        ):
            raise ValueError("relative objective tolerance must be positive")
        if not math.isfinite(self.lower_tail_fraction) or not (
            0.0 < self.lower_tail_fraction <= 1.0
        ):
            raise ValueError("lower tail fraction must lie in (0, 1]")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "OptimizationConfig":
        candidate = config.get("candidate_optimization")
        uncertainty = config.get("uncertainty")
        if not isinstance(candidate, Mapping) or not isinstance(uncertainty, Mapping):
            raise ValueError("candidate_optimization and uncertainty must be mappings")
        candidate_required = (
            "candidate_budget",
            "continuous_refinement_multistarts",
            "sobol_seed",
            "maximum_solver_iterations",
            "relative_objective_tolerance",
            "clearance_feasibility_policy",
            "selection",
            "selection_order",
        )
        uncertainty_required = (
            "scenario_design",
            "scenario_count",
            "sobol_seed",
            "lower_tail_fraction",
        )
        missing_candidate = [key for key in candidate_required if key not in candidate]
        missing_uncertainty = [key for key in uncertainty_required if key not in uncertainty]
        if missing_candidate or missing_uncertainty:
            raise ValueError(
                "incomplete shared optimization config: "
                f"candidate_optimization missing {missing_candidate}; "
                f"uncertainty missing {missing_uncertainty}"
            )
        expected_order = (
            "hard_bound_minimum_task_margin",
            "qmc_lower_tail_mean_task_margin",
            "minimum_peak_normal_force_n",
            "minimum_joint_torque_utilization",
            "maximum_trajectory_clearance_m",
        )
        if candidate["selection"] != "LEXICOGRAPHIC":
            raise ValueError("candidate selection must be explicitly LEXICOGRAPHIC")
        if tuple(candidate["selection_order"]) != expected_order:
            raise ValueError("candidate selection_order differs from the frozen protocol")
        if (
            candidate["clearance_feasibility_policy"]
            != "NONNEGATIVE_CERTIFIED_LOWER_BOUND"
        ):
            raise ValueError(
                "clearance feasibility must use a nonnegative certified lower bound"
            )
        if uncertainty["scenario_design"] != "SCRAMBLED_SOBOL":
            raise ValueError("uncertainty scenario_design must be SCRAMBLED_SOBOL")
        return cls(
            candidate_budget=int(candidate["candidate_budget"]),
            continuous_refinement_multistarts=int(
                candidate["continuous_refinement_multistarts"]
            ),
            candidate_sobol_seed=int(candidate["sobol_seed"]),
            maximum_solver_iterations=int(candidate["maximum_solver_iterations"]),
            relative_objective_tolerance=float(
                candidate["relative_objective_tolerance"]
            ),
            scenario_count=int(uncertainty["scenario_count"]),
            scenario_sobol_seed=int(uncertainty["sobol_seed"]),
            lower_tail_fraction=float(uncertainty["lower_tail_fraction"]),
        )


@dataclass(frozen=True)
class EvaluatedGrasp:
    candidate: GraspCandidate
    evaluation: WrenchEvaluation
    parameters_unit: tuple[float, ...]


@dataclass(frozen=True)
class OptimizationResult:
    ranked: tuple[RankedCandidate[EvaluatedGrasp], ...]
    candidate_design_unit: tuple[tuple[float, ...], ...]
    scenario_design_unit: tuple[tuple[float, ...], ...]
    decoded_candidate_count: int
    feasible_evaluation_count: int
    refinement_evaluation_count: int

    @property
    def selected(self) -> EvaluatedGrasp:
        if not self.ranked:
            raise OptimizationError("optimization result has no selected grasp")
        return self.ranked[0].candidate


def deterministic_sobol(
    *, dimension: int, count: int, seed: int
) -> np.ndarray:
    """Generate a reproducible scrambled Sobol design in ``[0, 1)^d``."""

    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Sobol dimension must be a positive integer")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Sobol count must be a positive integer")
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover - scipy is an offline dependency
        raise OptimizationError("deterministic Sobol design requires scipy.stats.qmc") from exc

    sobol_parameters = inspect.signature(qmc.Sobol).parameters
    sobol_keywords: dict[str, Any] = {
        "d": dimension,
        "scramble": True,
        "seed": seed,
    }
    if "bits" in sobol_parameters:
        sobol_keywords["bits"] = max(1, int(math.ceil(math.log2(count))))
    if "optimization" in sobol_parameters:
        sobol_keywords["optimization"] = None
    engine = qmc.Sobol(**sobol_keywords)
    if count & (count - 1) == 0:
        points = engine.random_base2(int(math.log2(count)))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            points = engine.random(count)
    result = np.asarray(points, dtype=np.float64)
    if result.shape != (count, dimension) or not np.all(
        (0.0 <= result) & (result < 1.0)
    ):
        raise OptimizationError("Sobol implementation returned an invalid design")
    return result


class CARTSGraspOptimizer:
    """Common-QMC-design search with lexicographic local refinement."""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config

    def _metrics(self, evaluation: WrenchEvaluation) -> CandidateMetrics:
        hard_minimum = evaluation.hard_bound_minimum_task_margin
        if hard_minimum is None:
            raise OptimizationError(
                "wrench evaluator must explicitly provide a certified "
                "hard_bound_minimum_task_margin"
            )
        return CandidateMetrics(
            hard_bound_minimum_task_margin=float(hard_minimum),
            qmc_lower_tail_mean_task_margin=qmc_lower_tail_mean(
                evaluation.task_margins, self.config.lower_tail_fraction
            ),
            peak_normal_force_n=float(evaluation.peak_normal_force_n),
            joint_torque_utilization=float(evaluation.joint_torque_utilization),
            trajectory_clearance_m=float(evaluation.trajectory_clearance_m),
        )

    def _evaluate_point(
        self,
        *,
        parameters_unit: np.ndarray,
        scenario_parameters_unit: np.ndarray,
        surface_model: ObjectSurfaceModel,
        wrench_evaluator: WrenchEvaluator,
        hand_model: ThreeFingerHandModel,
    ) -> tuple[EvaluatedGrasp, CandidateMetrics] | None:
        if parameters_unit.shape != (surface_model.parameter_dimension,):
            raise OptimizationError("surface model received a point of the wrong dimension")
        if not np.all((0.0 <= parameters_unit) & (parameters_unit <= 1.0)):
            raise OptimizationError("candidate parameters left the unit hypercube")
        expected_scenario_shape = (
            self.config.scenario_count,
            int(wrench_evaluator.uncertainty_dimension),
        )
        if scenario_parameters_unit.shape != expected_scenario_shape:
            raise OptimizationError(
                "scenario design has shape "
                f"{scenario_parameters_unit.shape}, expected {expected_scenario_shape}"
            )
        candidate = surface_model.candidate_from_unit_parameters(
            parameters_unit.copy(), hand_model
        )
        if candidate is None:
            return None
        if len(candidate.independent_joint_positions_rad) != len(
            hand_model.independent_joint_names
        ):
            raise OptimizationError("surface model returned the wrong number of hand joints")
        try:
            hand_model.resolve_joint_positions(candidate.independent_joint_positions_rad)
        except HandModelError:
            return None
        if {contact.pad_name for contact in candidate.planned_pad_contacts} != set(
            hand_model.pads
        ):
            raise OptimizationError("surface model must plan one contact for every PAD")
        evaluation = wrench_evaluator.evaluate(
            candidate,
            scenario_parameters_unit,
            surface_model=surface_model,
            hand_model=hand_model,
        )
        if not evaluation.feasible:
            return None
        if evaluation.trajectory_clearance_m < 0.0:
            return None
        evaluated = EvaluatedGrasp(
            candidate=candidate,
            evaluation=evaluation,
            parameters_unit=tuple(float(item) for item in parameters_unit),
        )
        return evaluated, self._metrics(evaluation)

    def _refine_start(
        self,
        *,
        start: ScoredCandidate[EvaluatedGrasp],
        initial_step: float,
        surface_model: ObjectSurfaceModel,
        wrench_evaluator: WrenchEvaluator,
        hand_model: ThreeFingerHandModel,
        next_source_index: int,
        scenario_parameters_unit: np.ndarray,
    ) -> tuple[
        ScoredCandidate[EvaluatedGrasp],
        int,
        int,
        int,
    ]:
        current = start
        point = np.asarray(start.candidate.parameters_unit, dtype=np.float64)
        step = float(initial_step)
        evaluation_count = 0
        feasible_count = 0
        source_index = next_source_index
        stop_resolution = max(
            self.config.relative_objective_tolerance,
            64.0 * np.finfo(np.float64).eps,
        )

        while (
            evaluation_count < self.config.maximum_solver_iterations
            and step > stop_resolution
        ):
            improved_in_sweep = False
            for axis in range(surface_model.parameter_dimension):
                for direction in (-1.0, 1.0):
                    proposal = point.copy()
                    proposal[axis] = float(np.clip(proposal[axis] + direction * step, 0.0, 1.0))
                    if proposal[axis] == point[axis]:
                        continue
                    result = self._evaluate_point(
                        parameters_unit=proposal,
                        scenario_parameters_unit=scenario_parameters_unit,
                        surface_model=surface_model,
                        wrench_evaluator=wrench_evaluator,
                        hand_model=hand_model,
                    )
                    evaluation_count += 1
                    if result is not None:
                        evaluated, metrics = result
                        row = ScoredCandidate(
                            candidate=evaluated,
                            metrics=metrics,
                            source="CONTINUOUS_REFINEMENT",
                            source_index=source_index,
                        )
                        source_index += 1
                        feasible_count += 1
                        if lexicographically_better(metrics, current.metrics):
                            current = row
                            point = proposal
                            improved_in_sweep = True
                    if evaluation_count >= self.config.maximum_solver_iterations:
                        break
                if evaluation_count >= self.config.maximum_solver_iterations:
                    break
            if not improved_in_sweep:
                step *= 0.5
        return current, source_index, evaluation_count, feasible_count

    def optimize(
        self,
        *,
        surface_model: ObjectSurfaceModel,
        wrench_evaluator: WrenchEvaluator,
        hand_model: ThreeFingerHandModel,
    ) -> OptimizationResult:
        parameter_dimension = int(surface_model.parameter_dimension)
        uncertainty_dimension = int(wrench_evaluator.uncertainty_dimension)
        if parameter_dimension <= 0 or uncertainty_dimension <= 0:
            raise OptimizationError("surface and uncertainty dimensions must be positive")

        candidate_design = deterministic_sobol(
            dimension=parameter_dimension,
            count=self.config.candidate_budget,
            seed=self.config.candidate_sobol_seed,
        )
        scenario_design = deterministic_sobol(
            dimension=uncertainty_dimension,
            count=self.config.scenario_count,
            seed=self.config.scenario_sobol_seed,
        )

        rows: list[ScoredCandidate[EvaluatedGrasp]] = []
        decoded_count = 0
        for index, point in enumerate(candidate_design):
            result = self._evaluate_point(
                parameters_unit=point,
                scenario_parameters_unit=scenario_design,
                surface_model=surface_model,
                wrench_evaluator=wrench_evaluator,
                hand_model=hand_model,
            )
            if result is None:
                continue
            decoded_count += 1
            evaluated, metrics = result
            rows.append(
                ScoredCandidate(
                    candidate=evaluated,
                    metrics=metrics,
                    source="SOBOL_DESIGN",
                    source_index=index,
                )
            )
        if not rows:
            raise OptimizationError("candidate design produced no feasible grasp")

        initial_ranking = rank_candidates(rows)
        multistart_count = min(
            self.config.continuous_refinement_multistarts,
            len(initial_ranking),
        )
        # Unit-cube cell width implied by the finite design budget; no physical
        # alignment or model-specific angular cut-off is introduced here.
        initial_step = self.config.candidate_budget ** (-1.0 / parameter_dimension)
        source_index = self.config.candidate_budget
        refinement_rows: list[ScoredCandidate[EvaluatedGrasp]] = []
        refinement_evaluations = 0
        refinement_feasible_evaluations = 0
        for ranked_start in initial_ranking[:multistart_count]:
            start = ScoredCandidate(
                candidate=ranked_start.candidate,
                metrics=ranked_start.metrics,
                source=ranked_start.source,
                source_index=ranked_start.source_index,
            )
            best, source_index, attempted, feasible = self._refine_start(
                start=start,
                initial_step=initial_step,
                surface_model=surface_model,
                wrench_evaluator=wrench_evaluator,
                hand_model=hand_model,
                next_source_index=source_index,
                scenario_parameters_unit=scenario_design,
            )
            if best.source == "CONTINUOUS_REFINEMENT":
                refinement_rows.append(best)
            refinement_evaluations += attempted
            refinement_feasible_evaluations += feasible

        unique_rows: dict[tuple[float, ...], ScoredCandidate[EvaluatedGrasp]] = {
            row.candidate.parameters_unit: row for row in rows
        }
        for row in refinement_rows:
            existing = unique_rows.get(row.candidate.parameters_unit)
            if existing is None or lexicographically_better(
                row.metrics, existing.metrics
            ):
                unique_rows[row.candidate.parameters_unit] = row
        all_rows = list(unique_rows.values())
        final_ranking = rank_candidates(all_rows)
        return OptimizationResult(
            ranked=final_ranking,
            candidate_design_unit=tuple(
                tuple(float(item) for item in row) for row in candidate_design
            ),
            scenario_design_unit=tuple(
                tuple(float(item) for item in row) for row in scenario_design
            ),
            decoded_candidate_count=decoded_count,
            feasible_evaluation_count=(
                len(rows) + refinement_feasible_evaluations
            ),
            refinement_evaluation_count=refinement_evaluations,
        )


__all__ = [
    "CARTSGraspOptimizer",
    "EvaluatedGrasp",
    "GraspCandidate",
    "ObjectSurfaceModel",
    "OptimizationConfig",
    "OptimizationError",
    "OptimizationResult",
    "PlannedPadContact",
    "WrenchEvaluation",
    "WrenchEvaluator",
    "deterministic_sobol",
]
