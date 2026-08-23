"""Object-independent polyhedral contact-wrench calculations for CARTS-Grasp.

The module consumes planned contact geometry and explicit physical contracts.
It has no simulator dependency and deliberately keeps object identification out
of the mathematical interface.

For each contact, the circular Coulomb cone is replaced by a certified inner
regular polygon.  Non-negative combinations of its rays produce contact forces;
the sum of a contact's ray coefficients is its normal force.  A single linear
program then certifies the largest ``gamma`` for which every vertex of

    nominal_external_wrench + gamma * disturbance_polytope

can be balanced under the same contact and actuator constraints.  Each vertex
has its own feasible contact-force allocation, while ``gamma`` is shared.
After maximising ``gamma``, two explicitly supplied ray-load groups are
minimised in strict lexicographic stages.  No weighted-sum approximation is
used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag, csr_matrix, diags, hstack, vstack


_FLOAT_EPS = np.finfo(np.float64).eps
_SCALING_IMPLEMENTATION = (
    "EXPLICIT_AUGMENTED_ROW_AND_COLUMN_INF_NORM_V1_"
    "PLUS_HIGHS_INTERNAL_AUTOMATIC"
)


def _readonly(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float array that callers cannot mutate."""

    value = np.ascontiguousarray(array, dtype=np.float64)
    value.setflags(write=False)
    return value


def _finite_vector(
    value: Sequence[float], *, shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _per_contact_values(
    value: float | Sequence[float],
    *,
    contact_count: int,
    name: str,
    allow_zero: bool,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(contact_count, float(array), dtype=np.float64)
    if array.shape != (contact_count,) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be one finite scalar or {contact_count} values"
        )
    if allow_zero:
        valid = np.all(array >= 0.0)
        qualifier = "non-negative"
    else:
        valid = np.all(array > 0.0)
        qualifier = "positive"
    if not valid:
        raise ValueError(f"{name} must be {qualifier}")
    return array


def friction_cone_inner_relative_error(edges: int) -> float:
    """Worst radial loss of an E-sided unit-circle inner polygon."""

    if isinstance(edges, bool) or int(edges) != edges or int(edges) < 3:
        raise ValueError("edges must be an integer of at least three")
    return 1.0 - math.cos(math.pi / int(edges))


def minimum_regular_polygon_edges(maximum_relative_error: float) -> int:
    """Derive the smallest E satisfying 1 - cos(pi/E) <= epsilon.

    ``maximum_relative_error`` is a numerical approximation contract, not a
    grasp success threshold.
    """

    epsilon = float(maximum_relative_error)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("maximum_relative_error must lie strictly in (0, 1)")
    denominator = math.acos(1.0 - epsilon)
    if denominator == 0.0:
        raise ValueError("maximum_relative_error is below float64 resolution")
    edges = max(3, int(math.ceil(math.pi / denominator)))
    # Correct a possible one-ulp underestimate in the closed-form evaluation.
    while friction_cone_inner_relative_error(edges) > epsilon:
        edges += 1
    return edges


@dataclass(frozen=True)
class PolyhedralContactWrenchModel:
    """Immutable linear contact model expressed about ``wrench_origin_m``.

    ``inward_normals`` point along the compressive force exerted on the object.
    ``tangent_directions`` provide a material/contact-frame direction and make
    the finite polygon exactly equivariant when geometry is rigidly transformed.
    Optional force constraints act on stacked world-frame contact forces
    ``[f_0x, f_0y, f_0z, ..., f_(N-1)z]``; lower bounds can be represented by
    negating a row.
    """

    grasp_matrix: np.ndarray
    contact_force_matrix: np.ndarray
    ray_constraint_matrix: np.ndarray
    ray_constraint_upper_bounds: np.ndarray
    normal_force_matrix: np.ndarray
    ray_owner: tuple[int, ...]
    contact_points_m: np.ndarray
    inward_normals: np.ndarray
    tangent_directions: np.ndarray
    friction_coefficients: np.ndarray
    normal_force_caps_n: np.ndarray
    wrench_origin_m: np.ndarray
    base_edges_per_contact: int
    edges_per_contact: int
    maximum_inner_relative_error: float

    @property
    def contact_count(self) -> int:
        return int(self.contact_points_m.shape[0])

    @property
    def ray_count(self) -> int:
        return int(self.grasp_matrix.shape[1])


@dataclass(frozen=True)
class LinearProgramSolverOptions:
    """Numerical contract copied explicitly from the shared method config.

    CARTS explicitly equilibrates the augmented constraint rows and then the
    variable columns before calling SciPy/HiGHS.  HiGHS may additionally apply
    its internal scaling.  The feasibility tolerances therefore describe the
    normalized solver-coordinate problem, not a physical grasp-success gate.
    """

    solver: str
    requested_constraint_scaling: str
    maximum_iterations: int
    primal_feasibility_tolerance: float
    dual_feasibility_tolerance: float
    ipm_optimality_tolerance: float
    physical_acceptance_gate: bool

    def __post_init__(self) -> None:
        if self.solver != "SCIPY_HIGHS":
            raise ValueError("only the frozen SCIPY_HIGHS solver is supported")
        if self.requested_constraint_scaling != "ROW_AND_COLUMN_INF_NORM":
            raise ValueError(
                "constraint_scaling must be ROW_AND_COLUMN_INF_NORM"
            )
        if (
            isinstance(self.maximum_iterations, bool)
            or int(self.maximum_iterations) != self.maximum_iterations
            or self.maximum_iterations < 1
        ):
            raise ValueError("maximum_iterations must be a positive integer")
        for name in (
            "primal_feasibility_tolerance",
            "dual_feasibility_tolerance",
            "ipm_optimality_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.physical_acceptance_gate is not False:
            raise ValueError("the mathematical margin cannot be a physical acceptance gate")

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "LinearProgramSolverOptions":
        """Build from the ``linear_program`` mapping without silent defaults."""

        required = {
            "solver",
            "constraint_scaling",
            "maximum_iterations",
            "primal_feasibility_tolerance",
            "dual_feasibility_tolerance",
            "ipm_optimality_tolerance",
            "physical_acceptance_gate",
        }
        missing = sorted(required.difference(document))
        if missing:
            raise ValueError(f"linear_program config is missing fields: {missing}")
        return cls(
            solver=str(document["solver"]),
            requested_constraint_scaling=str(document["constraint_scaling"]),
            maximum_iterations=int(document["maximum_iterations"]),
            primal_feasibility_tolerance=float(
                document["primal_feasibility_tolerance"]
            ),
            dual_feasibility_tolerance=float(
                document["dual_feasibility_tolerance"]
            ),
            ipm_optimality_tolerance=float(document["ipm_optimality_tolerance"]),
            physical_acceptance_gate=document["physical_acceptance_gate"],
        )

    def scipy_options(self) -> dict[str, float | int]:
        return {
            "maxiter": int(self.maximum_iterations),
            "primal_feasibility_tolerance": float(
                self.primal_feasibility_tolerance
            ),
            "dual_feasibility_tolerance": float(self.dual_feasibility_tolerance),
            "ipm_optimality_tolerance": float(self.ipm_optimality_tolerance),
        }


def build_polyhedral_contact_wrench_model(
    *,
    contact_points_m: Sequence[Sequence[float]],
    inward_normals: Sequence[Sequence[float]],
    tangent_directions: Sequence[Sequence[float]],
    friction_coefficients: float | Sequence[float],
    normal_force_caps_n: float | Sequence[float],
    wrench_origin_m: Sequence[float],
    maximum_inner_approximation_relative_error: float,
    cone_edge_multiplier: int = 1,
    contact_force_inequality_matrix: Sequence[Sequence[float]] | None = None,
    contact_force_inequality_upper_bounds: Sequence[float] | None = None,
) -> PolyhedralContactWrenchModel:
    """Construct an arbitrary-N certified inner friction-cone wrench model."""

    points = np.asarray(contact_points_m, dtype=np.float64)
    normals_raw = np.asarray(inward_normals, dtype=np.float64)
    tangents_raw = np.asarray(tangent_directions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 1:
        raise ValueError("contact_points_m must have finite shape (N, 3), N >= 1")
    if (
        normals_raw.shape != points.shape
        or tangents_raw.shape != points.shape
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(normals_raw))
        or not np.all(np.isfinite(tangents_raw))
    ):
        raise ValueError("contact geometry must contain finite matching (N, 3) arrays")
    contact_count = int(points.shape[0])
    origin = _finite_vector(wrench_origin_m, shape=(3,), name="wrench_origin_m")
    friction = _per_contact_values(
        friction_coefficients,
        contact_count=contact_count,
        name="friction_coefficients",
        allow_zero=True,
    )
    caps = _per_contact_values(
        normal_force_caps_n,
        contact_count=contact_count,
        name="normal_force_caps_n",
        allow_zero=False,
    )
    if (
        isinstance(cone_edge_multiplier, bool)
        or int(cone_edge_multiplier) != cone_edge_multiplier
        or int(cone_edge_multiplier) < 1
    ):
        raise ValueError("cone_edge_multiplier must be a positive integer")
    base_edges = minimum_regular_polygon_edges(
        maximum_inner_approximation_relative_error
    )
    edges = base_edges * int(cone_edge_multiplier)

    normals: list[np.ndarray] = []
    tangents_1: list[np.ndarray] = []
    tangents_2: list[np.ndarray] = []
    numerical_collinearity_limit = math.sqrt(_FLOAT_EPS)
    for index, (normal_raw, tangent_raw) in enumerate(
        zip(normals_raw, tangents_raw)
    ):
        normal_norm = float(np.linalg.norm(normal_raw))
        if normal_norm <= numerical_collinearity_limit:
            raise ValueError(f"inward_normals[{index}] is numerically zero")
        normal = normal_raw / normal_norm
        tangent = tangent_raw - float(tangent_raw @ normal) * normal
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= numerical_collinearity_limit:
            raise ValueError(
                f"tangent_directions[{index}] is collinear with its normal"
            )
        tangent_1 = tangent / tangent_norm
        tangent_2 = np.cross(normal, tangent_1)
        normals.append(normal)
        tangents_1.append(tangent_1)
        tangents_2.append(tangent_2)

    ray_count = contact_count * edges
    grasp_matrix = np.zeros((6, ray_count), dtype=np.float64)
    contact_force_matrix = np.zeros(
        (3 * contact_count, ray_count), dtype=np.float64
    )
    normal_force_matrix = np.zeros((contact_count, ray_count), dtype=np.float64)
    owners: list[int] = []
    for contact_index in range(contact_count):
        radius = points[contact_index] - origin
        for edge_index in range(edges):
            angle = 2.0 * math.pi * edge_index / float(edges)
            force = normals[contact_index] + friction[contact_index] * (
                math.cos(angle) * tangents_1[contact_index]
                + math.sin(angle) * tangents_2[contact_index]
            )
            ray_index = contact_index * edges + edge_index
            grasp_matrix[:3, ray_index] = force
            grasp_matrix[3:, ray_index] = np.cross(radius, force)
            contact_force_matrix[
                3 * contact_index : 3 * contact_index + 3, ray_index
            ] = force
            normal_force_matrix[contact_index, ray_index] = 1.0
            owners.append(contact_index)

    constraint_rows = [normal_force_matrix]
    constraint_bounds = [caps]
    if (contact_force_inequality_matrix is None) != (
        contact_force_inequality_upper_bounds is None
    ):
        raise ValueError(
            "contact force inequality matrix and upper bounds must be supplied together"
        )
    if contact_force_inequality_matrix is not None:
        force_matrix = np.asarray(
            contact_force_inequality_matrix, dtype=np.float64
        )
        force_bounds = np.asarray(
            contact_force_inequality_upper_bounds, dtype=np.float64
        )
        if (
            force_matrix.ndim != 2
            or force_matrix.shape[1] != 3 * contact_count
            or force_bounds.shape != (force_matrix.shape[0],)
            or not np.all(np.isfinite(force_matrix))
            or not np.all(np.isfinite(force_bounds))
        ):
            raise ValueError(
                "contact force inequalities must be finite A:(M, 3N), b:(M,)"
            )
        constraint_rows.append(force_matrix @ contact_force_matrix)
        constraint_bounds.append(force_bounds)

    return PolyhedralContactWrenchModel(
        grasp_matrix=_readonly(grasp_matrix),
        contact_force_matrix=_readonly(contact_force_matrix),
        ray_constraint_matrix=_readonly(np.vstack(constraint_rows)),
        ray_constraint_upper_bounds=_readonly(np.concatenate(constraint_bounds)),
        normal_force_matrix=_readonly(normal_force_matrix),
        ray_owner=tuple(owners),
        contact_points_m=_readonly(points),
        inward_normals=_readonly(np.asarray(normals)),
        tangent_directions=_readonly(np.asarray(tangents_1)),
        friction_coefficients=_readonly(friction),
        normal_force_caps_n=_readonly(caps),
        wrench_origin_m=_readonly(origin),
        base_edges_per_contact=base_edges,
        edges_per_contact=edges,
        maximum_inner_relative_error=friction_cone_inner_relative_error(edges),
    )


@dataclass(frozen=True)
class TaskWrenchLexicographicStageResult:
    """Numerical report for one explicitly equilibrated LP stage."""

    stage_name: str
    solver_success: bool
    solver_status: int
    solver_message: str
    optimal_value: float | None
    equality_augmented_row_inf_norms: np.ndarray
    inequality_augmented_row_inf_norms: np.ndarray
    column_inf_norms_after_row_scaling: np.ndarray
    maximum_equilibrium_residual: float | None
    maximum_inequality_violation: float | None
    maximum_scaled_equilibrium_residual: float | None
    maximum_scaled_inequality_violation: float | None


@dataclass(frozen=True)
class TaskWrenchMarginResult:
    """Strict lexicographic certificate and its final-stage allocations."""

    solver_success: bool
    solver_status: int
    solver_message: str
    solver_options: LinearProgramSolverOptions
    constraint_scaling_implementation: str
    maximum_margin: float | None
    nominal_external_wrench: np.ndarray
    disturbance_vertices: np.ndarray
    ray_coefficients_by_vertex: np.ndarray | None
    contact_forces_by_vertex: np.ndarray | None
    normal_forces_by_vertex: np.ndarray | None
    lexicographic_optimal_loads: tuple[float, float] | None
    lexicographic_stage_results: tuple[TaskWrenchLexicographicStageResult, ...]
    equality_augmented_row_inf_norms: np.ndarray
    inequality_augmented_row_inf_norms: np.ndarray
    column_inf_norms_after_row_scaling: np.ndarray
    maximum_equilibrium_residual: float | None
    maximum_inequality_violation: float | None
    maximum_scaled_equilibrium_residual: float | None
    maximum_scaled_inequality_violation: float | None


def _sparse_row_absolute_maximum(matrix: csr_matrix) -> np.ndarray:
    """Return exact per-row maxima, with empty rows represented by zero."""

    absolute = abs(matrix).tocsr()
    maxima = np.zeros(absolute.shape[0], dtype=np.float64)
    for row_index in range(absolute.shape[0]):
        start = int(absolute.indptr[row_index])
        stop = int(absolute.indptr[row_index + 1])
        if start != stop:
            maxima[row_index] = float(np.max(absolute.data[start:stop]))
    return maxima


def _sparse_column_absolute_maximum(matrix: csr_matrix) -> np.ndarray:
    """Return exact per-column maxima, with empty columns represented by zero."""

    return _sparse_row_absolute_maximum(matrix.transpose().tocsr())


def _explicitly_equilibrate_linear_program(
    *,
    objective: np.ndarray,
    equality: csr_matrix,
    equality_target: np.ndarray,
    inequality: csr_matrix,
    inequality_target: np.ndarray,
) -> tuple[
    np.ndarray,
    csr_matrix,
    np.ndarray,
    csr_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Equilibrate an LP without empirical constants or hidden thresholds.

    For every constraint row, the divisor is the infinity norm of its
    augmented row ``[A_i, b_i]``.  A structurally zero augmented row uses the
    algebraic identity divisor one.  Columns are then divided by their
    infinity norm across the row-scaled constraints and objective; a
    structurally zero column likewise uses one.  If ``y`` is the solver
    variable and ``d`` contains the reported column norms, the physical LP
    variable is ``x = y / d``.
    """

    equality_row_norms = np.maximum(
        _sparse_row_absolute_maximum(equality), np.abs(equality_target)
    )
    inequality_row_norms = np.maximum(
        _sparse_row_absolute_maximum(inequality), np.abs(inequality_target)
    )
    equality_row_norms[equality_row_norms == 0.0] = 1.0
    inequality_row_norms[inequality_row_norms == 0.0] = 1.0

    row_scaled_equality = (
        diags(1.0 / equality_row_norms, format="csr") @ equality
    ).tocsr()
    row_scaled_inequality = (
        diags(1.0 / inequality_row_norms, format="csr") @ inequality
    ).tocsr()
    row_scaled_equality_target = equality_target / equality_row_norms
    row_scaled_inequality_target = inequality_target / inequality_row_norms

    coefficient_and_objective = vstack(
        (
            row_scaled_equality,
            row_scaled_inequality,
            csr_matrix(objective.reshape(1, -1)),
        ),
        format="csr",
    )
    column_norms = _sparse_column_absolute_maximum(
        coefficient_and_objective
    )
    column_norms[column_norms == 0.0] = 1.0
    physical_from_solver = diags(1.0 / column_norms, format="csr")

    return (
        objective / column_norms,
        (row_scaled_equality @ physical_from_solver).tocsr(),
        row_scaled_equality_target,
        (row_scaled_inequality @ physical_from_solver).tocsr(),
        row_scaled_inequality_target,
        equality_row_norms,
        inequality_row_norms,
        column_norms,
    )


@dataclass(frozen=True)
class _ScaledLinearProgramSolve:
    """Internal physical-coordinate solution and its scaling audit."""

    solver_success: bool
    solver_status: int
    solver_message: str
    physical_solution: np.ndarray | None
    equality_augmented_row_inf_norms: np.ndarray
    inequality_augmented_row_inf_norms: np.ndarray
    column_inf_norms_after_row_scaling: np.ndarray
    maximum_equilibrium_residual: float | None
    maximum_inequality_violation: float | None
    maximum_scaled_equilibrium_residual: float | None
    maximum_scaled_inequality_violation: float | None


def _solve_explicitly_equilibrated_linear_program(
    *,
    objective: np.ndarray,
    equality: csr_matrix,
    equality_target: np.ndarray,
    inequality: csr_matrix,
    inequality_target: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    solver_options: LinearProgramSolverOptions,
) -> _ScaledLinearProgramSolve:
    """Solve and certify one stage in the same scaled coordinates."""

    (
        scaled_objective,
        scaled_equality,
        scaled_equality_target,
        scaled_inequality,
        scaled_inequality_target,
        equality_row_norms,
        inequality_row_norms,
        column_norms,
    ) = _explicitly_equilibrate_linear_program(
        objective=objective,
        equality=equality,
        equality_target=equality_target,
        inequality=inequality,
        inequality_target=inequality_target,
    )
    result = linprog(
        scaled_objective,
        A_ub=scaled_inequality,
        b_ub=scaled_inequality_target,
        A_eq=scaled_equality,
        b_eq=scaled_equality_target,
        bounds=tuple(bounds),
        method="highs",
        options=solver_options.scipy_options(),
    )
    if not result.success:
        return _ScaledLinearProgramSolve(
            solver_success=False,
            solver_status=int(result.status),
            solver_message=str(result.message),
            physical_solution=None,
            equality_augmented_row_inf_norms=_readonly(equality_row_norms),
            inequality_augmented_row_inf_norms=_readonly(
                inequality_row_norms
            ),
            column_inf_norms_after_row_scaling=_readonly(column_norms),
            maximum_equilibrium_residual=None,
            maximum_inequality_violation=None,
            maximum_scaled_equilibrium_residual=None,
            maximum_scaled_inequality_violation=None,
        )

    solver_solution = np.asarray(result.x, dtype=np.float64)
    if (
        solver_solution.shape != objective.shape
        or not np.all(np.isfinite(solver_solution))
    ):
        return _ScaledLinearProgramSolve(
            solver_success=False,
            solver_status=int(result.status),
            solver_message="solver returned a non-finite or malformed solution",
            physical_solution=None,
            equality_augmented_row_inf_norms=_readonly(equality_row_norms),
            inequality_augmented_row_inf_norms=_readonly(
                inequality_row_norms
            ),
            column_inf_norms_after_row_scaling=_readonly(column_norms),
            maximum_equilibrium_residual=None,
            maximum_inequality_violation=None,
            maximum_scaled_equilibrium_residual=None,
            maximum_scaled_inequality_violation=None,
        )

    solution = solver_solution / column_norms
    equilibrium_residual = np.asarray(
        equality @ solution - equality_target, dtype=np.float64
    ).reshape(-1)
    inequality_residual = np.asarray(
        inequality @ solution - inequality_target, dtype=np.float64
    ).reshape(-1)
    scaled_equilibrium_residual = equilibrium_residual / equality_row_norms
    scaled_inequality_residual = inequality_residual / inequality_row_norms
    maximum_equilibrium_residual = float(
        np.max(np.abs(equilibrium_residual), initial=0.0)
    )
    maximum_inequality_violation = float(
        max(0.0, np.max(inequality_residual, initial=0.0))
    )
    maximum_scaled_equilibrium_residual = float(
        np.max(np.abs(scaled_equilibrium_residual), initial=0.0)
    )
    maximum_scaled_inequality_violation = float(
        max(0.0, np.max(scaled_inequality_residual, initial=0.0))
    )
    certified = (
        maximum_scaled_equilibrium_residual
        <= solver_options.primal_feasibility_tolerance
        and maximum_scaled_inequality_violation
        <= solver_options.primal_feasibility_tolerance
    )
    message = str(result.message)
    if not certified:
        message = (
            f"{message}; explicit scaled primal residual exceeds configured "
            "tolerance"
        )
    return _ScaledLinearProgramSolve(
        solver_success=certified,
        solver_status=int(result.status),
        solver_message=message,
        physical_solution=_readonly(solution),
        equality_augmented_row_inf_norms=_readonly(equality_row_norms),
        inequality_augmented_row_inf_norms=_readonly(inequality_row_norms),
        column_inf_norms_after_row_scaling=_readonly(column_norms),
        maximum_equilibrium_residual=maximum_equilibrium_residual,
        maximum_inequality_violation=maximum_inequality_violation,
        maximum_scaled_equilibrium_residual=(
            maximum_scaled_equilibrium_residual
        ),
        maximum_scaled_inequality_violation=(
            maximum_scaled_inequality_violation
        ),
    )


def _stage_result(
    *,
    stage_name: str,
    solve: _ScaledLinearProgramSolve,
    optimal_value: float | None,
) -> TaskWrenchLexicographicStageResult:
    """Project an internal solve into the immutable public stage report."""

    return TaskWrenchLexicographicStageResult(
        stage_name=stage_name,
        solver_success=solve.solver_success,
        solver_status=solve.solver_status,
        solver_message=solve.solver_message,
        optimal_value=(
            float(optimal_value)
            if solve.solver_success and optimal_value is not None
            else None
        ),
        equality_augmented_row_inf_norms=(
            solve.equality_augmented_row_inf_norms
        ),
        inequality_augmented_row_inf_norms=(
            solve.inequality_augmented_row_inf_norms
        ),
        column_inf_norms_after_row_scaling=(
            solve.column_inf_norms_after_row_scaling
        ),
        maximum_equilibrium_residual=solve.maximum_equilibrium_residual,
        maximum_inequality_violation=solve.maximum_inequality_violation,
        maximum_scaled_equilibrium_residual=(
            solve.maximum_scaled_equilibrium_residual
        ),
        maximum_scaled_inequality_violation=(
            solve.maximum_scaled_inequality_violation
        ),
    )


def _task_scale_inputs(model, nominal_external_wrench, disturbance_vertices, groups):
    nominal = _finite_vector(
        nominal_external_wrench, shape=(6,), name="nominal_external_wrench"
    )
    vertices = np.asarray(disturbance_vertices, dtype=np.float64)
    if vertices.shape == (6,):
        vertices = vertices.reshape(1, 6)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (6,)
        or len(vertices) < 1
        or not np.all(np.isfinite(vertices))
        or not np.any(vertices != 0.0)
    ):
        raise ValueError("disturbance_vertices must be finite non-zero (K, 6)")
    load_groups = tuple(np.asarray(group, dtype=np.float64) for group in groups)
    if len(load_groups) != 2 or any(
        group.ndim != 2
        or len(group) < 1
        or group.shape[1] != model.ray_count
        or not np.all(np.isfinite(group))
        for group in load_groups
    ):
        raise ValueError("exactly two finite ray-load groups are required")
    return nominal, vertices, load_groups


def _fixed_scale_program(model, nominal, vertices, scale):
    vertex_count = len(vertices)
    equality = block_diag(
        [csr_matrix(model.grasp_matrix)] * vertex_count, format="csr"
    )
    equality_target = np.tile(-nominal, vertex_count) - scale * vertices.reshape(-1)
    inequality = block_diag(
        [csr_matrix(model.ray_constraint_matrix)] * vertex_count, format="csr"
    )
    inequality_target = np.tile(model.ray_constraint_upper_bounds, vertex_count)
    return equality, equality_target, inequality, inequality_target


def _solve_fixed_scale_burdens(
    *, model, nominal, vertices, scale, solver_options, load_groups,
    check_feasibility, stage_suffix,
):
    equality, equality_target, inequality, inequality_target = _fixed_scale_program(
        model, nominal, vertices, scale
    )
    base_count = equality.shape[1]
    stages: list[TaskWrenchLexicographicStageResult] = []
    if check_feasibility:
        solve = _solve_explicitly_equilibrated_linear_program(
            objective=np.zeros(base_count), equality=equality,
            equality_target=equality_target, inequality=inequality,
            inequality_target=inequality_target,
            bounds=((0.0, None),) * base_count, solver_options=solver_options,
        )
        stages.append(_stage_result(
            stage_name="CHECK_FEASIBILITY_AT_PRESCRIBED_TASK_SCALE",
            solve=solve, optimal_value=0.0 if solve.solver_success else None,
        ))
        if not solve.solver_success:
            return stages, [], None
    group_blocks = tuple(
        block_diag([csr_matrix(group)] * len(vertices), format="csr")
        for group in load_groups
    )
    optimal_loads: list[float] = []
    final_solve = None
    for index, current in enumerate(group_blocks):
        rows = [hstack([inequality, csr_matrix((inequality.shape[0], 1))])]
        targets = [inequality_target]
        for previous, optimum in zip(group_blocks, optimal_loads):
            rows.append(hstack([previous, csr_matrix((previous.shape[0], 1))]))
            targets.append(np.full(previous.shape[0], optimum))
        rows.append(hstack([current, -np.ones((current.shape[0], 1))]))
        targets.append(np.zeros(current.shape[0]))
        objective = np.zeros(base_count + 1)
        objective[-1] = 1.0
        solve = _solve_explicitly_equilibrated_linear_program(
            objective=objective,
            equality=hstack([equality, csr_matrix((equality.shape[0], 1))]),
            equality_target=equality_target,
            inequality=vstack(rows, format="csr"),
            inequality_target=np.concatenate(targets),
            bounds=((0.0, None),) * (base_count + 1),
            solver_options=solver_options,
        )
        optimum = (
            float(solve.physical_solution[-1])
            if solve.solver_success and solve.physical_solution is not None else None
        )
        stages.append(_stage_result(
            stage_name=f"MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_{index}{stage_suffix}",
            solve=solve, optimal_value=optimum,
        ))
        if optimum is None:
            return stages, optimal_loads, None
        optimal_loads.append(optimum)
        final_solve = solve
    return stages, optimal_loads, final_solve


def maximum_task_wrench_polytope_margin(
    model: PolyhedralContactWrenchModel, *, nominal_external_wrench,
    disturbance_vertices, solver_options, lexicographic_ray_load_groups,
) -> TaskWrenchMarginResult:
    """Maximize task scale, then minimize the two physical burden groups."""

    nominal, vertices, groups = _task_scale_inputs(
        model, nominal_external_wrench, disturbance_vertices,
        lexicographic_ray_load_groups,
    )
    base_equality, target, base_inequality, inequality_target = (
        _fixed_scale_program(model, nominal, vertices, 0.0)
    )
    equality = hstack([base_equality, csr_matrix(vertices.reshape(-1, 1))])
    inequality = hstack(
        [base_inequality, csr_matrix((base_inequality.shape[0], 1))]
    )
    objective = np.zeros(equality.shape[1])
    objective[-1] = -1.0
    margin_solve = _solve_explicitly_equilibrated_linear_program(
        objective=objective, equality=equality, equality_target=target,
        inequality=inequality, inequality_target=inequality_target,
        bounds=((0.0, None),) * len(objective), solver_options=solver_options,
    )
    maximum_margin = (
        float(margin_solve.physical_solution[-1])
        if margin_solve.solver_success and margin_solve.physical_solution is not None
        else None
    )
    margin_stage = _stage_result(
        stage_name="MAXIMIZE_SHARED_TASK_MARGIN", solve=margin_solve,
        optimal_value=maximum_margin,
    )
    stages = [margin_stage]
    if maximum_margin is not None:
        load_stages, loads, final_solve = _solve_fixed_scale_burdens(
            model=model, nominal=nominal, vertices=vertices, scale=maximum_margin,
            solver_options=solver_options, load_groups=groups,
            check_feasibility=False, stage_suffix="",
        )
        stages.extend(load_stages)
    else:
        loads, final_solve = [], None
    final_stage = stages[-1]
    success = final_solve is not None and final_solve.physical_solution is not None
    if success:
        rays = final_solve.physical_solution[:-1].reshape(len(vertices), model.ray_count)
        forces = np.asarray([
            (model.contact_force_matrix @ row).reshape(model.contact_count, 3)
            for row in rays
        ])
        normals = np.asarray([model.normal_force_matrix @ row for row in rays])
    else:
        rays = forces = normals = None
    return TaskWrenchMarginResult(
        solver_success=success, solver_status=final_stage.solver_status,
        solver_message="; ".join(
            f"{stage.stage_name}: {stage.solver_message}" for stage in stages
        ), solver_options=solver_options,
        constraint_scaling_implementation=_SCALING_IMPLEMENTATION,
        maximum_margin=maximum_margin if success else None,
        nominal_external_wrench=_readonly(nominal),
        disturbance_vertices=_readonly(vertices),
        ray_coefficients_by_vertex=None if rays is None else _readonly(rays),
        contact_forces_by_vertex=None if forces is None else _readonly(forces),
        normal_forces_by_vertex=None if normals is None else _readonly(normals),
        lexicographic_optimal_loads=None if not success else tuple(loads),
        lexicographic_stage_results=tuple(stages),
        equality_augmented_row_inf_norms=final_stage.equality_augmented_row_inf_norms,
        inequality_augmented_row_inf_norms=final_stage.inequality_augmented_row_inf_norms,
        column_inf_norms_after_row_scaling=final_stage.column_inf_norms_after_row_scaling,
        maximum_equilibrium_residual=final_stage.maximum_equilibrium_residual,
        maximum_inequality_violation=final_stage.maximum_inequality_violation,
        maximum_scaled_equilibrium_residual=final_stage.maximum_scaled_equilibrium_residual,
        maximum_scaled_inequality_violation=final_stage.maximum_scaled_inequality_violation,
    )


@dataclass(frozen=True)
class PrescribedTaskScaleBurdenResult:
    """Lexicographic burden certificate at one fixed task scale.

    ``gamma`` is pinned to ``prescribed_task_scale`` instead of being
    maximised, so the two burdens are minima needed at exactly that task
    scale, not at the maximum margin.
    """

    solver_success: bool
    solver_status: int
    solver_message: str
    prescribed_task_scale: float
    feasible_at_prescribed_scale: bool
    peak_normal_force_n: float | None
    peak_joint_torque_utilization: float | None
    lexicographic_stage_results: tuple[TaskWrenchLexicographicStageResult, ...]


def _prescribed_task_scale_failure(
    *,
    scale: float,
    stages: Sequence[TaskWrenchLexicographicStageResult],
    feasible: bool,
) -> PrescribedTaskScaleBurdenResult:
    stage = stages[-1]
    return PrescribedTaskScaleBurdenResult(
        solver_success=False,
        solver_status=stage.solver_status,
        solver_message=f"{stage.stage_name} failed closed: {stage.solver_message}",
        prescribed_task_scale=scale,
        feasible_at_prescribed_scale=feasible,
        peak_normal_force_n=None,
        peak_joint_torque_utilization=None,
        lexicographic_stage_results=tuple(stages),
    )


def prescribed_task_scale_burden(
    model: PolyhedralContactWrenchModel,
    *,
    nominal_external_wrench: Sequence[float],
    disturbance_vertices: Sequence[Sequence[float]],
    prescribed_task_scale: float,
    solver_options: LinearProgramSolverOptions,
    lexicographic_ray_load_groups: Sequence[Sequence[Sequence[float]]],
) -> PrescribedTaskScaleBurdenResult:
    """Minimum burdens with the shared task scale pinned to one value.

    Reuses the contact model and the explicitly equilibrated LP machinery;
    the coverage variable is fixed, so every vertex equilibrium becomes
    ``G r = -w - scale * d``.  A zero-objective feasibility phase fails
    closed before any burden stage runs.
    """

    scale = float(prescribed_task_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("prescribed_task_scale must be finite and non-negative")
    nominal, vertices, groups = _task_scale_inputs(
        model, nominal_external_wrench, disturbance_vertices,
        lexicographic_ray_load_groups,
    )
    stages, loads, final_solve = _solve_fixed_scale_burdens(
        model=model, nominal=nominal, vertices=vertices, scale=scale,
        solver_options=solver_options, load_groups=groups,
        check_feasibility=True, stage_suffix="_AT_PRESCRIBED_TASK_SCALE",
    )
    if final_solve is None:
        return _prescribed_task_scale_failure(
            scale=scale, stages=stages, feasible=len(stages) > 1
        )
    final_stage = stages[-1]
    return PrescribedTaskScaleBurdenResult(
        solver_success=True,
        solver_status=final_stage.solver_status,
        solver_message="; ".join(f"{stage.stage_name}: {stage.solver_message}" for stage in stages),
        prescribed_task_scale=scale,
        feasible_at_prescribed_scale=True,
        peak_normal_force_n=float(loads[0]),
        peak_joint_torque_utilization=float(loads[1]),
        lexicographic_stage_results=tuple(stages),
    )
