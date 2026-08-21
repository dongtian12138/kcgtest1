"""Pair-conditioned pose seeds followed by exact-FK first-contact validation.

Two object-surface points and two finite-PAD witnesses define a small
dimensionless invariant problem.  Its bounded least-squares solver endpoints
are *pose proposals*, not roots or contact certificates.  Every accepted grasp
is recomputed by
``RayClosureSurfaceModel`` over all finite PAD witnesses and the complete
object triangle mesh.

No residual cutoff is used.  In particular, a local least-squares residual is
reported for diagnosis and branch ordering is independent of that residual;
it never decides whether a grasp is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    deterministic_sobol,
)
from kcg_connector.grasp.robust.hand_model import ThreeFingerHandModel
from kcg_connector.grasp.robust.ray_closure import (
    CertifiedSequentialClosurePolicy,
    DisplayOnlyGraspProposal,
    PossibleFirstContactSet,
    RayClosureAudit,
    RayClosureError,
    RayClosureSurfaceModel,
    _DOT_ERROR,
    _FK_ERROR,
    _gamma,
    _recover_closed_unit_coordinate,
)


METHOD_ID = "CARTS_PAIR_CONDITIONED_INVARIANT_POSE_SEED_V1"
PARAMETER_DOMAIN_ID = (
    "THREE_UNORDERED_PAD_PAIRS_X_TWO_RESIDUAL_CDF_AREA_CHARTS_X_"
    "TWO_PAD_AREA_WITNESS_CDFS_X_MULTISTART_ENDPOINT_BRANCH_X_"
    "NONCLOSURE_PRESHAPE_V1"
)
SEED_ROLE = "POSE_SEED_ONLY_FINAL_CONTACT_TRUTH_FROM_DELEGATED_RAY_CLOSURE"
PAIR_INVARIANTS = (
    "TASK_AXIS_CHORD_COMPONENT_EQUALITY",
    "TASK_TRANSVERSE_CHORD_SQUARED_NORM_EQUALITY",
)
PAD_WITNESS_MEASURE = "PAD_TRIANGLE_AREA_DIVIDED_EQUALLY_AMONG_FIXED_WITNESSES"
OBJECT_SURFACE_MEASURE = "CONTRACT_ALLOWED_TRIANGLE_AREA_MEASURE"
MULTISTART_ENDPOINT_POLICY = (
    "ALL_FINITE_NONDEGENERATE_BOUNDED_LEAST_SQUARES_ENDPOINTS_"
    "IN_SOBOL_START_ORDER_WITH_DUPLICATES_RETAINED"
)
PARAMETER_LAYOUT_PREFIX = (
    "unordered_sampled_anchor_pad_pair_selector_unit",
    "object_anchor_1_residual_cdf_unit",
    "object_anchor_1_split_unit",
    "object_anchor_2_residual_cdf_unit",
    "object_anchor_2_split_unit",
    "pad_anchor_1_area_witness_cdf_unit",
    "pad_anchor_2_area_witness_cdf_unit",
    "multistart_endpoint_branch_unit",
)
CLAIM_LIMITATIONS = (
    "SAMPLED_PAIR_ANCHORS_ARE_NOT_CONTACT_TRUTH",
    "SAMPLED_ANCHORS_MUST_NOT_FEED_WRENCH_OR_CONTROLLER",
    "LOCAL_MULTISTART_ENDPOINTS_ARE_NOT_ROOT_CERTIFICATES",
    "ENDPOINT_RESIDUAL_IS_REPORTED_ONLY_AND_IS_NOT_AN_ACCEPTANCE_GATE",
    "SOLVER_SUCCESS_IS_DIAGNOSTIC_ONLY",
    "FINITE_PAD_WITNESS_AND_OBJECT_MESH_CONVERGENCE_REQUIRED",
    "DELEGATED_RAY_CLOSURE_AND_COMPLETE_COLLISION_LIMITATIONS_APPLY",
)


class PairConditionedSeedError(ValueError):
    """Raised when the immutable pair-conditioned proposal is malformed."""


@dataclass(frozen=True)
class PairSeedSolverOptions:
    """Explicit numerical protocol for the dimensionless two-phase solve."""

    multistart_count: int
    sobol_seed: int
    maximum_function_evaluations: int
    function_tolerance: float
    step_tolerance: float
    gradient_tolerance: float

    def __post_init__(self) -> None:
        for name, value in (
            ("multistart_count", self.multistart_count),
            (
                "maximum_function_evaluations",
                self.maximum_function_evaluations,
            ),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise PairConditionedSeedError(
                    f"{name} must be a positive integer"
                )
        if not isinstance(self.sobol_seed, int) or isinstance(
            self.sobol_seed, bool
        ):
            raise PairConditionedSeedError("sobol_seed must be an integer")
        machine_epsilon = np.finfo(np.float64).eps
        for name, value in (
            ("function_tolerance", self.function_tolerance),
            ("step_tolerance", self.step_tolerance),
            ("gradient_tolerance", self.gradient_tolerance),
        ):
            if (
                not math.isfinite(float(value))
                or float(value) <= machine_epsilon
            ):
                raise PairConditionedSeedError(
                    f"{name} must be finite and greater than binary64 epsilon"
                )

    @property
    def contract(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "solver": "SCIPY_TRF_BOUNDED_WITH_EXACT_INVARIANT_JACOBIAN",
                "residual_scaling": (
                    "LENGTH_BY_OBJECT_CHARACTERISTIC_LENGTH_AND_SQUARED_"
                    "LENGTH_BY_ITS_SQUARE"
                ),
                "multistart_count": self.multistart_count,
                "sobol_seed": self.sobol_seed,
                "maximum_function_evaluations": (
                    self.maximum_function_evaluations
                ),
                "function_tolerance": self.function_tolerance,
                "step_tolerance": self.step_tolerance,
                "gradient_tolerance": self.gradient_tolerance,
                "physical_acceptance_gate": False,
            }
        )


@dataclass(frozen=True)
class PairSolverEndpointAudit:
    start_index: int
    solver_success: bool
    solver_status: int
    function_evaluations: int
    jacobian_evaluations: int | None
    phases: tuple[float, float]
    invariant_residual_dimensionless: tuple[float, float]
    residual_infinity_norm: float
    first_order_optimality: float
    yaw_rad: float | None
    sampled_pad_winding_rates_m_per_unit: tuple[float, float] | None
    sampled_object_normal_rates_m_per_unit: tuple[float, float] | None
    sampled_anchor_directions_compatible: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "start_index": self.start_index,
            "solver_success": self.solver_success,
            "solver_status": self.solver_status,
            "function_evaluations": self.function_evaluations,
            "jacobian_evaluations": self.jacobian_evaluations,
            "phases": list(self.phases),
            "invariant_residual_dimensionless": list(
                self.invariant_residual_dimensionless
            ),
            "residual_infinity_norm": self.residual_infinity_norm,
            "first_order_optimality": self.first_order_optimality,
            "yaw_rad": self.yaw_rad,
            "sampled_pad_winding_rates_m_per_unit": (
                None
                if self.sampled_pad_winding_rates_m_per_unit is None
                else list(self.sampled_pad_winding_rates_m_per_unit)
            ),
            "sampled_object_normal_rates_m_per_unit": (
                None
                if self.sampled_object_normal_rates_m_per_unit is None
                else list(self.sampled_object_normal_rates_m_per_unit)
            ),
            "sampled_anchor_directions_compatible": (
                self.sampled_anchor_directions_compatible
            ),
        }


@dataclass(frozen=True)
class PairConditionedSeedAudit:
    method_id: str
    parameter_domain_id: str
    parameter_layout: tuple[str, ...]
    seed_role: str
    pair_invariants: tuple[str, ...]
    pad_witness_measure: str
    object_surface_measure: str
    solver_endpoint_policy: str
    parameters_unit: tuple[float, ...] | None
    object_geometry_sha256: str
    sampled_anchor_pad_names: tuple[str, str] | None
    sampled_anchor_object_face_indices: tuple[int, int] | None
    sampled_anchor_object_barycentric: (
        tuple[tuple[float, float, float], ...] | None
    )
    sampled_anchor_object_positions_m: (
        tuple[tuple[float, float, float], ...] | None
    )
    sampled_anchor_pad_triangle_indices: tuple[int, int] | None
    sampled_anchor_pad_witness_indices: tuple[int, int] | None
    solver_endpoints: tuple[PairSolverEndpointAudit, ...]
    eligible_endpoint_count: int
    sampled_direction_compatible_endpoint_count: int
    selected_endpoint_index: int | None
    delegated_volume_parameters_unit: tuple[float, ...] | None
    delegated_closure_audit: RayClosureAudit | None
    solver_contract: Mapping[str, object]
    claim_limitations: tuple[str, ...]
    failure_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "parameter_domain_id": self.parameter_domain_id,
            "parameter_layout": list(self.parameter_layout),
            "seed_role": self.seed_role,
            "pair_invariants": list(self.pair_invariants),
            "pad_witness_measure": self.pad_witness_measure,
            "object_surface_measure": self.object_surface_measure,
            "solver_endpoint_policy": self.solver_endpoint_policy,
            "parameters_unit": (
                None
                if self.parameters_unit is None
                else list(self.parameters_unit)
            ),
            "object_geometry_sha256": self.object_geometry_sha256,
            "sampled_anchor_pad_names": (
                None
                if self.sampled_anchor_pad_names is None
                else list(self.sampled_anchor_pad_names)
            ),
            "sampled_anchor_object_face_indices": (
                None
                if self.sampled_anchor_object_face_indices is None
                else list(self.sampled_anchor_object_face_indices)
            ),
            "sampled_anchor_object_barycentric": (
                None
                if self.sampled_anchor_object_barycentric is None
                else [
                    list(row)
                    for row in self.sampled_anchor_object_barycentric
                ]
            ),
            "sampled_anchor_object_positions_m": (
                None
                if self.sampled_anchor_object_positions_m is None
                else [
                    list(row)
                    for row in self.sampled_anchor_object_positions_m
                ]
            ),
            "sampled_anchor_pad_triangle_indices": (
                None
                if self.sampled_anchor_pad_triangle_indices is None
                else list(self.sampled_anchor_pad_triangle_indices)
            ),
            "sampled_anchor_pad_witness_indices": (
                None
                if self.sampled_anchor_pad_witness_indices is None
                else list(self.sampled_anchor_pad_witness_indices)
            ),
            "solver_endpoints": [
                endpoint.as_dict() for endpoint in self.solver_endpoints
            ],
            "eligible_endpoint_count": self.eligible_endpoint_count,
            "sampled_direction_compatible_endpoint_count": (
                self.sampled_direction_compatible_endpoint_count
            ),
            "selected_endpoint_index": self.selected_endpoint_index,
            "delegated_volume_parameters_unit": (
                None
                if self.delegated_volume_parameters_unit is None
                else list(self.delegated_volume_parameters_unit)
            ),
            "delegated_closure_audit": (
                None
                if self.delegated_closure_audit is None
                else self.delegated_closure_audit.as_dict()
            ),
            "solver_contract": dict(self.solver_contract),
            "claim_limitations": list(self.claim_limitations),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class PairConditionedSeedEvaluation:
    candidate: GraspCandidate | None
    audit: PairConditionedSeedAudit
    possible_first_contact_sets: tuple[PossibleFirstContactSet, ...] = ()
    display_only_proposal: DisplayOnlyGraspProposal | None = None
    sequential_closure_policy: CertifiedSequentialClosurePolicy | None = None

    @property
    def feasible(self) -> bool:
        return self.candidate is not None

    @property
    def representative_proposal_available(self) -> bool:
        """Whether V9 returned a non-evidentiary display proposal."""

        return self.display_only_proposal is not None

    @property
    def static_policy_available(self) -> bool:
        return self.sequential_closure_policy is not None


@dataclass(frozen=True)
class _SurfacePoint:
    face_index: int
    barycentric: np.ndarray
    position_m: np.ndarray
    outward_normal: np.ndarray


@dataclass(frozen=True)
class _WitnessState:
    position_hand_m: np.ndarray
    velocity_hand_per_unit: np.ndarray
    source_winding_normal_hand: np.ndarray


class PairConditionedInvariantSeedModel:
    """ObjectSurfaceModel-compatible dual-surface invariant pose proposal."""

    def __init__(
        self,
        closure_model: RayClosureSurfaceModel,
        *,
        solver_options: PairSeedSolverOptions,
    ) -> None:
        if not isinstance(closure_model, RayClosureSurfaceModel):
            raise PairConditionedSeedError(
                "closure_model must be a RayClosureSurfaceModel"
            )
        if not isinstance(solver_options, PairSeedSolverOptions):
            raise PairConditionedSeedError(
                "solver_options must be an explicit PairSeedSolverOptions"
            )
        face_indices = np.flatnonzero(
            closure_model.object_model.contact_face_mask
        ).astype(np.int64)
        if len(face_indices) == 0:
            raise PairConditionedSeedError(
                "object contract exposes no allowed contact faces"
            )
        face_areas = np.asarray(
            closure_model.object_model.mesh.face_areas_m2[face_indices],
            dtype=np.float64,
        )
        if np.any(face_areas <= 0.0) or not np.all(np.isfinite(face_areas)):
            raise PairConditionedSeedError(
                "allowed object faces need positive finite area"
            )
        cumulative_face_area = np.cumsum(face_areas, dtype=np.float64)
        total_face_area = math.fsum(float(value) for value in face_areas)
        cumulative_face_area[-1] = total_face_area

        witness_cumulative_measures: list[np.ndarray] = []
        witness_total_measures: list[float] = []
        for prepared in closure_model.prepared_pads:
            triangles = prepared.verified.points_local_m[
                prepared.verified.faces
            ]
            areas = 0.5 * np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                ),
                axis=1,
            )
            counts = np.bincount(
                prepared.triangle_indices,
                minlength=len(prepared.verified.faces),
            )
            if (
                np.any(areas <= 0.0)
                or np.any(counts <= 0)
                or not np.all(np.isfinite(areas))
            ):
                raise PairConditionedSeedError(
                    f"verified PAD {prepared.verified.name} has invalid "
                    "witness measure"
                )
            weights = (
                areas[prepared.triangle_indices]
                / counts[prepared.triangle_indices]
            )
            cumulative = np.cumsum(weights, dtype=np.float64)
            total = math.fsum(float(value) for value in weights)
            cumulative[-1] = total
            cumulative.setflags(write=False)
            witness_cumulative_measures.append(cumulative)
            witness_total_measures.append(total)

        self.closure_model = closure_model
        self.hand_model = closure_model.hand_model
        self.object_model = closure_model.object_model
        self.solver_options = solver_options
        self.allowed_face_indices = face_indices
        self.allowed_face_areas_m2 = face_areas
        self.cumulative_allowed_face_area_m2 = cumulative_face_area
        self.allowed_surface_area_m2 = total_face_area
        self.witness_cumulative_measures = tuple(witness_cumulative_measures)
        self.witness_total_measures = tuple(witness_total_measures)
        self.pad_pairs = ((0, 1), (0, 2), (1, 2))
        self.characteristic_length_m = float(
            closure_model.intersector.characteristic_length_m
        )
        if self.characteristic_length_m <= 0.0:
            raise PairConditionedSeedError(
                "pair invariants need a positive object characteristic length"
            )
        self.phase_start_design = deterministic_sobol(
            dimension=2,
            count=solver_options.multistart_count,
            seed=solver_options.sobol_seed,
        )
        self.phase_start_design.setflags(write=False)
        self.parameter_layout = PARAMETER_LAYOUT_PREFIX + tuple(
            f"preshape_joint_unit:{name}"
            for name in closure_model.preshape_joint_names
        )
        for array in (
            self.allowed_face_indices,
            self.allowed_face_areas_m2,
            self.cumulative_allowed_face_area_m2,
        ):
            array.setflags(write=False)

    @property
    def parameter_dimension(self) -> int:
        return len(self.parameter_layout)

    def _surface_point(
        self, residual_unit: float, split_unit: float
    ) -> _SurfacePoint:
        area_coordinate = residual_unit * self.allowed_surface_area_m2
        local_face = min(
            int(
                np.searchsorted(
                    self.cumulative_allowed_face_area_m2,
                    area_coordinate,
                    side="right",
                )
            ),
            len(self.allowed_face_indices) - 1,
        )
        preceding_area = (
            0.0
            if local_face == 0
            else float(self.cumulative_allowed_face_area_m2[local_face - 1])
        )
        residual = (
            area_coordinate - preceding_area
        ) / float(self.allowed_face_areas_m2[local_face])
        residual = min(1.0, max(0.0, residual))
        if residual == 0.0 and split_unit != 0.0:
            raise PairConditionedSeedError(
                "zero residual-CDF radius requires canonical split "
                "coordinate 0"
            )
        radial = math.sqrt(residual)
        barycentric = np.asarray(
            (
                1.0 - radial,
                radial * (1.0 - split_unit),
                radial * split_unit,
            ),
            dtype=np.float64,
        )
        face_index = int(self.allowed_face_indices[local_face])
        triangle = self.object_model.mesh.face_vertices_m[face_index]
        return _SurfacePoint(
            face_index=face_index,
            barycentric=barycentric,
            position_m=barycentric @ triangle,
            outward_normal=self.object_model.mesh.face_normals[face_index],
        )

    def _witness_flat_index(self, pad_row: int, unit: float) -> int:
        cumulative = self.witness_cumulative_measures[pad_row]
        coordinate = unit * self.witness_total_measures[pad_row]
        return min(
            int(np.searchsorted(cumulative, coordinate, side="right")),
            len(cumulative) - 1,
        )

    def _witness_state(
        self,
        *,
        pad_row: int,
        witness_flat_index: int,
        q_start: np.ndarray,
        phase: float,
    ) -> _WitnessState:
        prepared = self.closure_model.prepared_pads[pad_row]
        direction = self.closure_model.closing_directions_physical[pad_row]
        q = q_start + phase * direction
        transform = self.hand_model.forward_kinematics(q)[
            prepared.verified.link_name
        ]
        point_local = prepared.witness_points_link_m[witness_flat_index]
        position = transform[:3, :3] @ point_local + transform[:3, 3]
        jacobian = self.hand_model.geometric_jacobian(
            prepared.verified.link_name,
            q,
            point_local_m=point_local,
        )
        velocity = jacobian[:3] @ direction
        normal = transform[:3, :3] @ prepared.witness_normals_link[
            witness_flat_index
        ]
        return _WitnessState(position, velocity, normal)

    def _solver_endpoints(
        self,
        *,
        pad_pair: tuple[int, int],
        witness_indices: tuple[int, int],
        object_points: tuple[_SurfacePoint, _SurfacePoint],
        q_start: np.ndarray,
    ) -> tuple[PairSolverEndpointAudit, ...]:
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:  # pragma: no cover - offline dependency
            raise PairConditionedSeedError(
                "pair-conditioned seeds require scipy.optimize.least_squares"
            ) from exc

        first_row, second_row = pad_pair
        first_direction = self.closure_model.closing_directions_physical[
            first_row
        ]
        second_direction = self.closure_model.closing_directions_physical[
            second_row
        ]
        maximum = np.asarray(
            (
                self.closure_model._maximum_path_parameter(
                    q_start, first_direction
                ),
                self.closure_model._maximum_path_parameter(
                    q_start, second_direction
                ),
            ),
            dtype=np.float64,
        )
        if np.any(maximum <= 0.0):
            return ()
        target_chord_task = self.closure_model.task_basis_object.T @ (
            object_points[1].position_m - object_points[0].position_m
        )
        length = self.characteristic_length_m
        cache_key: bytes | None = None
        cache_value: (
            tuple[np.ndarray, np.ndarray, _WitnessState, _WitnessState]
            | None
        ) = None

        def invariant_and_jacobian(
            phases: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, _WitnessState, _WitnessState]:
            nonlocal cache_key, cache_value
            key = np.asarray(phases, dtype="<f8").tobytes(order="C")
            if key == cache_key and cache_value is not None:
                return cache_value
            first = self._witness_state(
                pad_row=first_row,
                witness_flat_index=witness_indices[0],
                q_start=q_start,
                phase=float(phases[0]),
            )
            second = self._witness_state(
                pad_row=second_row,
                witness_flat_index=witness_indices[1],
                q_start=q_start,
                phase=float(phases[1]),
            )
            chord = second.position_hand_m - first.position_hand_m
            residual = np.asarray(
                (
                    (chord[2] - target_chord_task[2]) / length,
                    (
                        float(chord[:2] @ chord[:2])
                        - float(target_chord_task[:2] @ target_chord_task[:2])
                    )
                    / (length * length),
                ),
                dtype=np.float64,
            )
            jacobian = np.asarray(
                (
                    (
                        -first.velocity_hand_per_unit[2] / length,
                        second.velocity_hand_per_unit[2] / length,
                    ),
                    (
                        -2.0
                        * float(
                            chord[:2] @ first.velocity_hand_per_unit[:2]
                        )
                        / (length * length),
                        2.0
                        * float(
                            chord[:2] @ second.velocity_hand_per_unit[:2]
                        )
                        / (length * length),
                    ),
                ),
                dtype=np.float64,
            )
            cache_key = key
            cache_value = residual, jacobian, first, second
            return cache_value

        endpoints: list[PairSolverEndpointAudit] = []
        for start_index, start_unit in enumerate(self.phase_start_design):
            result = least_squares(
                lambda value: invariant_and_jacobian(value)[0],
                np.asarray(start_unit * maximum, dtype=np.float64),
                jac=lambda value: invariant_and_jacobian(value)[1],
                bounds=(np.zeros(2, dtype=np.float64), maximum),
                method="trf",
                ftol=self.solver_options.function_tolerance,
                xtol=self.solver_options.step_tolerance,
                gtol=self.solver_options.gradient_tolerance,
                max_nfev=self.solver_options.maximum_function_evaluations,
            )
            phases = np.asarray(result.x, dtype=np.float64)
            residual, _jacobian, first, second = invariant_and_jacobian(phases)
            chord = second.position_hand_m - first.position_hand_m
            transverse_cross = float(
                chord[0] * target_chord_task[1]
                - chord[1] * target_chord_task[0]
            )
            transverse_dot = float(
                chord[0] * target_chord_task[0]
                + chord[1] * target_chord_task[1]
            )
            transverse_scale = max(
                float(np.linalg.norm(chord[:2]))
                * float(np.linalg.norm(target_chord_task[:2])),
                np.finfo(np.float64).tiny,
            )
            degeneracy_bound = _gamma(64) * transverse_scale
            yaw: float | None
            pad_rates: tuple[float, float] | None
            object_rates: tuple[float, float] | None
            sampled_directions_compatible = False
            if (
                not np.all(np.isfinite(phases))
                or not np.all(np.isfinite(residual))
                or math.hypot(transverse_cross, transverse_dot)
                <= degeneracy_bound
            ):
                yaw = None
                pad_rates = None
                object_rates = None
            else:
                yaw = math.atan2(transverse_cross, transverse_dot) % (
                    2.0 * math.pi
                )
                cosine = math.cos(yaw)
                sine = math.sin(yaw)
                rotation = self.closure_model.task_basis_object @ np.asarray(
                    (
                        (cosine, -sine, 0.0),
                        (sine, cosine, 0.0),
                        (0.0, 0.0, 1.0),
                    ),
                    dtype=np.float64,
                )
                velocities = (
                    rotation @ first.velocity_hand_per_unit,
                    rotation @ second.velocity_hand_per_unit,
                )
                normals = (
                    rotation @ first.source_winding_normal_hand,
                    rotation @ second.source_winding_normal_hand,
                )
                pad_rates = tuple(
                    float(normal @ velocity)
                    for normal, velocity in zip(normals, velocities)
                )
                object_rates = tuple(
                    -float(point.outward_normal @ velocity)
                    for point, velocity in zip(object_points, velocities)
                )
                speed_errors = tuple(
                    _DOT_ERROR * float(np.linalg.norm(velocity))
                    for velocity in velocities
                )
                sampled_directions_compatible = all(
                    pad_rate > error and object_rate > error
                    for pad_rate, object_rate, error in zip(
                        pad_rates, object_rates, speed_errors
                    )
                )
            endpoints.append(
                PairSolverEndpointAudit(
                    start_index=start_index,
                    solver_success=bool(result.success),
                    solver_status=int(result.status),
                    function_evaluations=int(result.nfev),
                    jacobian_evaluations=(
                        None if result.njev is None else int(result.njev)
                    ),
                    phases=(float(phases[0]), float(phases[1])),
                    invariant_residual_dimensionless=(
                        float(residual[0]),
                        float(residual[1]),
                    ),
                    residual_infinity_norm=float(
                        np.linalg.norm(residual, ord=np.inf)
                    ),
                    first_order_optimality=float(result.optimality),
                    yaw_rad=yaw,
                    sampled_pad_winding_rates_m_per_unit=pad_rates,
                    sampled_object_normal_rates_m_per_unit=object_rates,
                    sampled_anchor_directions_compatible=(
                        sampled_directions_compatible
                    ),
                )
            )
        return tuple(endpoints)

    def _audit(
        self,
        *,
        parameters: np.ndarray | None = None,
        pad_pair: tuple[int, int] | None = None,
        object_points: tuple[_SurfacePoint, _SurfacePoint] | None = None,
        witness_indices: tuple[int, int] | None = None,
        solver_endpoints: tuple[PairSolverEndpointAudit, ...] = (),
        selected_endpoint_index: int | None = None,
        volume_parameters: np.ndarray | None = None,
        closure_audit: RayClosureAudit | None = None,
        failure_reason: str | None = None,
    ) -> PairConditionedSeedAudit:
        sampled_anchor_pad_names = None
        triangles = None
        witnesses = None
        if pad_pair is not None:
            sampled_anchor_pad_names = tuple(
                self.closure_model.prepared_pads[row].verified.name
                for row in pad_pair
            )
        if pad_pair is not None and witness_indices is not None:
            triangles = tuple(
                int(
                    self.closure_model.prepared_pads[row].triangle_indices[
                        witness
                    ]
                )
                for row, witness in zip(pad_pair, witness_indices)
            )
            witnesses = tuple(
                int(
                    self.closure_model.prepared_pads[row].witness_indices[
                        witness
                    ]
                )
                for row, witness in zip(pad_pair, witness_indices)
            )
        eligible_count = sum(
            endpoint.yaw_rad is not None for endpoint in solver_endpoints
        )
        sampled_direction_compatible_count = sum(
            endpoint.sampled_anchor_directions_compatible
            for endpoint in solver_endpoints
        )
        return PairConditionedSeedAudit(
            method_id=METHOD_ID,
            parameter_domain_id=PARAMETER_DOMAIN_ID,
            parameter_layout=self.parameter_layout,
            seed_role=SEED_ROLE,
            pair_invariants=PAIR_INVARIANTS,
            pad_witness_measure=PAD_WITNESS_MEASURE,
            object_surface_measure=OBJECT_SURFACE_MEASURE,
            solver_endpoint_policy=MULTISTART_ENDPOINT_POLICY,
            parameters_unit=(
                None
                if parameters is None
                else tuple(float(value) for value in parameters)
            ),
            object_geometry_sha256=self.object_model.geometry_sha256,
            sampled_anchor_pad_names=sampled_anchor_pad_names,
            sampled_anchor_object_face_indices=(
                None
                if object_points is None
                else tuple(point.face_index for point in object_points)
            ),
            sampled_anchor_object_barycentric=(
                None
                if object_points is None
                else tuple(
                    tuple(float(value) for value in point.barycentric)
                    for point in object_points
                )
            ),
            sampled_anchor_object_positions_m=(
                None
                if object_points is None
                else tuple(
                    tuple(float(value) for value in point.position_m)
                    for point in object_points
                )
            ),
            sampled_anchor_pad_triangle_indices=triangles,
            sampled_anchor_pad_witness_indices=witnesses,
            solver_endpoints=solver_endpoints,
            eligible_endpoint_count=eligible_count,
            sampled_direction_compatible_endpoint_count=(
                sampled_direction_compatible_count
            ),
            selected_endpoint_index=selected_endpoint_index,
            delegated_volume_parameters_unit=(
                None
                if volume_parameters is None
                else tuple(float(value) for value in volume_parameters)
            ),
            delegated_closure_audit=closure_audit,
            solver_contract=self.solver_options.contract,
            claim_limitations=CLAIM_LIMITATIONS,
            failure_reason=failure_reason,
        )

    def _parameter_failure(
        self, reason: str, parameters: np.ndarray | None = None
    ) -> PairConditionedSeedEvaluation:
        return PairConditionedSeedEvaluation(
            None,
            self._audit(
                parameters=parameters,
                failure_reason=f"PARAMETER_DOMAIN_REJECTED:{reason}",
            ),
        )

    def evaluate_unit_parameters(
        self,
        parameters_unit: Sequence[float],
        hand_model: ThreeFingerHandModel | None = None,
    ) -> PairConditionedSeedEvaluation:
        supplied_hand = self.hand_model if hand_model is None else hand_model
        self.closure_model._validate_hand(supplied_hand)
        parameters = np.asarray(parameters_unit, dtype=np.float64)
        if parameters.shape != (self.parameter_dimension,) or not np.all(
            np.isfinite(parameters)
        ):
            return self._parameter_failure(
                f"parameters need finite shape ({self.parameter_dimension},)"
            )
        if np.any(parameters < 0.0) or np.any(parameters > 1.0):
            return self._parameter_failure(
                "parameters must lie within [0, 1]", parameters
            )
        for index, label in enumerate(PARAMETER_LAYOUT_PREFIX):
            if parameters[index] >= 1.0:
                return self._parameter_failure(
                    f"{label} uses a canonical half-open unit interval",
                    parameters,
                )
        try:
            object_points = (
                self._surface_point(
                    float(parameters[1]), float(parameters[2])
                ),
                self._surface_point(
                    float(parameters[3]), float(parameters[4])
                ),
            )
        except PairConditionedSeedError as error:
            return self._parameter_failure(str(error), parameters)
        pair_index = min(
            int(float(parameters[0]) * len(self.pad_pairs)),
            len(self.pad_pairs) - 1,
        )
        pad_pair = self.pad_pairs[pair_index]
        witness_indices = (
            self._witness_flat_index(pad_pair[0], float(parameters[5])),
            self._witness_flat_index(pad_pair[1], float(parameters[6])),
        )
        q_start = np.array(self.closure_model.open_joint_template, copy=True)
        for unit_value, joint_index in zip(
            parameters[len(PARAMETER_LAYOUT_PREFIX):],
            self.closure_model.preshape_joint_indices,
        ):
            q_start[joint_index] = (
                self.closure_model.lower_joint_limits[joint_index]
                + float(unit_value)
                * self.closure_model.joint_spans[joint_index]
            )
        solver_endpoints = self._solver_endpoints(
            pad_pair=pad_pair,
            witness_indices=witness_indices,
            object_points=object_points,
            q_start=q_start,
        )
        eligible_indices = tuple(
            index
            for index, endpoint in enumerate(solver_endpoints)
            if endpoint.yaw_rad is not None
        )
        if not eligible_indices:
            return PairConditionedSeedEvaluation(
                None,
                self._audit(
                    parameters=parameters,
                    pad_pair=pad_pair,
                    object_points=object_points,
                    witness_indices=witness_indices,
                    solver_endpoints=solver_endpoints,
                    failure_reason=(
                        "NO_FINITE_NONDEGENERATE_SOLVER_ENDPOINT"
                    ),
                ),
            )
        branch_index = min(
            int(float(parameters[7]) * len(eligible_indices)),
            len(eligible_indices) - 1,
        )
        selected_endpoint_index = eligible_indices[branch_index]
        selected = solver_endpoints[selected_endpoint_index]
        assert selected.yaw_rad is not None
        first_state = self._witness_state(
            pad_row=pad_pair[0],
            witness_flat_index=witness_indices[0],
            q_start=q_start,
            phase=selected.phases[0],
        )
        cosine = math.cos(selected.yaw_rad)
        sine = math.sin(selected.yaw_rad)
        rotation = self.closure_model.task_basis_object @ np.asarray(
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        translation = (
            object_points[0].position_m
            - rotation @ first_state.position_hand_m
        )
        focus_result = self.closure_model._closure_focus_hand(q_start)
        if focus_result is None:
            return PairConditionedSeedEvaluation(
                None,
                self._audit(
                    parameters=parameters,
                    pad_pair=pad_pair,
                    object_points=object_points,
                    witness_indices=witness_indices,
                    solver_endpoints=solver_endpoints,
                    selected_endpoint_index=selected_endpoint_index,
                    failure_reason=(
                        "CLOSING_FOCUS_UNDEFINED_FROM_PAD_KINEMATICS"
                    ),
                ),
            )
        focus_hand, hand_extent = focus_result
        target = rotation @ focus_hand + translation
        try:
            lower, upper = self.closure_model._placement_coordinate_bounds(
                q_start, rotation
            )
            target_coordinates = self.closure_model.task_basis_object.T @ (
                target - self.object_model.assembly_axis_origin_m
            )
            comparison_error_m = (
                self.closure_model.intersector.distance_error_bound_m
                + self.closure_model.distance_bvh.aabb_error_bound_m
                + _FK_ERROR
                * max(
                    self.characteristic_length_m,
                    hand_extent,
                    float(np.linalg.norm(target, ord=np.inf)),
                )
            )
            volume_parameters = np.empty(
                self.closure_model.parameter_dimension, dtype=np.float64
            )
            volume_parameters[0] = selected.yaw_rad / (2.0 * math.pi)
            volume_parameters[1] = _recover_closed_unit_coordinate(
                float(target_coordinates[2]),
                float(lower[2]),
                float(upper[2]),
                absolute_error=comparison_error_m,
                label="pair-conditioned axial target",
            )
            volume_parameters[2] = _recover_closed_unit_coordinate(
                float(target_coordinates[0]),
                float(lower[0]),
                float(upper[0]),
                absolute_error=comparison_error_m,
                label="pair-conditioned first lateral target",
            )
            volume_parameters[3] = _recover_closed_unit_coordinate(
                float(target_coordinates[1]),
                float(lower[1]),
                float(upper[1]),
                absolute_error=comparison_error_m,
                label="pair-conditioned second lateral target",
            )
            volume_parameters[4:] = parameters[
                len(PARAMETER_LAYOUT_PREFIX):
            ]
        except RayClosureError as error:
            return PairConditionedSeedEvaluation(
                None,
                self._audit(
                    parameters=parameters,
                    pad_pair=pad_pair,
                    object_points=object_points,
                    witness_indices=witness_indices,
                    solver_endpoints=solver_endpoints,
                    selected_endpoint_index=selected_endpoint_index,
                    failure_reason=f"SEED_OUTSIDE_COMMON_SWEPT_DOMAIN:{error}",
                ),
            )
        closure = self.closure_model.evaluate_unit_parameters(
            volume_parameters, supplied_hand
        )
        failure_reason = (
            None
            if closure.candidate is not None
            else f"DELEGATED_CLOSURE_REJECTED:{closure.audit.failure_reason}"
        )
        return PairConditionedSeedEvaluation(
            closure.candidate,
            self._audit(
                parameters=parameters,
                pad_pair=pad_pair,
                object_points=object_points,
                witness_indices=witness_indices,
                solver_endpoints=solver_endpoints,
                selected_endpoint_index=selected_endpoint_index,
                volume_parameters=volume_parameters,
                closure_audit=closure.audit,
                failure_reason=failure_reason,
            ),
            possible_first_contact_sets=closure.possible_first_contact_sets,
            display_only_proposal=closure.display_only_proposal,
            sequential_closure_policy=closure.sequential_closure_policy,
        )

    def candidate_from_unit_parameters(
        self,
        parameters_unit: np.ndarray,
        hand_model: ThreeFingerHandModel,
    ) -> GraspCandidate | None:
        return self.evaluate_unit_parameters(
            parameters_unit, hand_model
        ).candidate

    def trajectory_clearance_m(
        self,
        candidate: GraspCandidate,
        hand_model: ThreeFingerHandModel,
    ) -> float:
        return self.closure_model.trajectory_clearance_m(candidate, hand_model)

    @property
    def contract(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "method_id": METHOD_ID,
                "parameter_domain_id": PARAMETER_DOMAIN_ID,
                "parameter_layout": self.parameter_layout,
                "seed_role": SEED_ROLE,
                "pair_invariants": PAIR_INVARIANTS,
                "pad_witness_measure": PAD_WITNESS_MEASURE,
                "object_surface_measure": OBJECT_SURFACE_MEASURE,
                "solver_endpoint_policy": MULTISTART_ENDPOINT_POLICY,
                "object_geometry_sha256": self.object_model.geometry_sha256,
                "solver_contract": self.solver_options.contract,
                "claim_limitations": CLAIM_LIMITATIONS,
                "delegated_closure_contract": self.closure_model.contract,
            }
        )


__all__ = [
    "CLAIM_LIMITATIONS",
    "METHOD_ID",
    "OBJECT_SURFACE_MEASURE",
    "PAD_WITNESS_MEASURE",
    "PAIR_INVARIANTS",
    "PARAMETER_DOMAIN_ID",
    "PARAMETER_LAYOUT_PREFIX",
    "PairConditionedInvariantSeedModel",
    "PairConditionedSeedAudit",
    "PairConditionedSeedError",
    "PairConditionedSeedEvaluation",
    "PairSolverEndpointAudit",
    "PairSeedSolverOptions",
    "MULTISTART_ENDPOINT_POLICY",
    "SEED_ROLE",
]
