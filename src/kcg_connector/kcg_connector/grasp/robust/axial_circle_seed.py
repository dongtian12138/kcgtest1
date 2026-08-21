"""Axially conditioned circle--triangle pose proposals for CARTS-Grasp.

The first finite-PAD witness is placed exactly on an area-sampled allowed
object point.  A second area-sampled surface point supplies only an axial
coordinate and conditions the second finger's closure phase through
deterministic binary64 sign-change bisection.
For every selected phase, the second witness traces a circle under the
assembly-axis yaw.  Analytic binary64 circle--triangle-plane rows provide
mesh-labelled pose seeds against every contract-allowed object triangle;
they are not intersection certificates.

These constructions are pose proposals, not contact truth.  Neither a phase
residual nor a sampled-normal diagnostic is an acceptance gate.  Every
accepted grasp is recomputed by ``RayClosureSurfaceModel`` using exact hand FK,
all finite PAD witnesses, and the complete object triangle mesh.  Wrench and
controller modules may consume only the delegated contacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.grasp_optimizer import GraspCandidate
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


METHOD_ID = "CARTS_AXIAL_CONDITIONED_CIRCLE_TRIANGLE_POSE_SEED_V2"
PARAMETER_DOMAIN_ID = (
    "SIX_ORDERED_PAD_PAIRS_X_TWO_ALLOWED_SURFACE_AREA_CHARTS_X_"
    "TWO_PAD_AREA_WITNESS_CDFS_X_FIRST_PHASE_X_SURFACE_PUSHFORWARD_AXIS_X_"
    "BRACKETED_PHASE_ENDPOINT_X_CIRCLE_TRIANGLE_ENDPOINT_X_PRESHAPE_V1"
)
SEED_ROLE = "POSE_SEED_ONLY_FINAL_CONTACT_TRUTH_FROM_DELEGATED_RAY_CLOSURE"
PHASE_ENDPOINT_POLICY = (
    "UNIFORM_PHASE_CELLS_X_FORWARD_ERROR_SEPARATED_SIGN_CHANGE_"
    "FIXED_BISECTION_X_BRACKETS_ORDERED_BY_PHASE"
)
CIRCLE_PROPOSAL_POLICY = (
    "ANALYTIC_BINARY64_NONCOPLANAR_CIRCLE_TRIANGLE_PLANE_"
    "ENDPOINT_PROPOSALS_X_BARYCENTRIC_FILTER_X_MESH_LABEL_ORDER"
)
OBJECT_SURFACE_MEASURE = "CONTRACT_ALLOWED_TRIANGLE_AREA_MEASURE"
PAD_WITNESS_MEASURE = (
    "PAD_TRIANGLE_AREA_DIVIDED_EQUALLY_AMONG_FIXED_WITNESSES"
)
PARAMETER_LAYOUT_PREFIX = (
    "ordered_sampled_anchor_pad_pair_selector_unit",
    "sampled_anchor_1_object_residual_cdf_unit",
    "sampled_anchor_1_object_split_unit",
    "sampled_axial_donor_object_residual_cdf_unit",
    "sampled_axial_donor_object_split_unit",
    "sampled_anchor_1_pad_area_witness_cdf_unit",
    "sampled_anchor_2_pad_area_witness_cdf_unit",
    "sampled_anchor_1_closure_phase_unit",
    "bracketed_phase_endpoint_branch_unit",
    "circle_triangle_endpoint_branch_unit",
)
CLAIM_LIMITATIONS = (
    "SAMPLED_PAIR_ANCHORS_ARE_NOT_CONTACT_TRUTH",
    "SAMPLED_ANCHORS_MUST_NOT_FEED_WRENCH_OR_CONTROLLER",
    "CERTIFIED_SIGN_CHANGE_PHASE_BRACKETS_PROVE_EXISTENCE_BUT_NOT_"
    "ROOT_LOCATION",
    "EVEN_MULTIPLE_OR_TANGENTIAL_PHASE_ROOTS_WITHOUT_GRID_SIGN_CHANGE_"
    "CAN_BE_MISSED",
    "CIRCLE_TRIANGLE_ROWS_ARE_BINARY64_SEEDS_NOT_ROOT_CERTIFICATES",
    "COPLANAR_CIRCLE_TRIANGLE_ARCS_ARE_NOT_COVERED",
    "SHARED_EDGE_AND_YAW_SEAM_DUPLICATES_CAN_CHANGE_BRANCH_MULTIPLICITY",
    "AMBIGUOUS_CIRCLE_OR_BARYCENTRIC_ROWS_MAY_BE_SKIPPED_AS_SEEDS",
    "FINITE_PAD_WITNESS_AND_OBJECT_MESH_CONVERGENCE_REQUIRED",
    "DELEGATED_RAY_CLOSURE_AND_COMPLETE_COLLISION_LIMITATIONS_APPLY",
)


class AxialCircleSeedError(ValueError):
    """Raised when the immutable axial-circle proposal is malformed."""


@dataclass(frozen=True)
class AxialCircleNumericalOptions:
    """Explicit computation budgets; neither field is a physical threshold."""

    axial_phase_cell_count: int
    axial_bisection_iterations: int

    def __post_init__(self) -> None:
        for name, value in (
            ("axial_phase_cell_count", self.axial_phase_cell_count),
            ("axial_bisection_iterations", self.axial_bisection_iterations),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise AxialCircleSeedError(
                    f"{name} must be an explicit positive integer"
                )

    @property
    def contract(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "axial_phase_cell_count": self.axial_phase_cell_count,
                "axial_bisection_iterations": (
                    self.axial_bisection_iterations
                ),
                "phase_acceptance_residual_threshold": None,
                "sampled_direction_physical_acceptance_gate": False,
                "physical_acceptance_owner": (
                    "DELEGATED_RAY_CLOSURE_THEN_WRENCH_AND_COLLISION_"
                    "CERTIFICATES"
                ),
            }
        )


@dataclass(frozen=True)
class AxialPhaseEndpointAudit:
    grid_cell_index: int
    bracket_lower_phase: float
    bracket_upper_phase: float
    bracket_lower_residual_interval_m: tuple[float, float]
    bracket_upper_residual_interval_m: tuple[float, float]
    selected_phase: float
    raw_axial_residual_m: float
    selected_residual_interval_m: tuple[float, float]
    construction: str

    def as_dict(self) -> dict[str, object]:
        return {
            "grid_cell_index": self.grid_cell_index,
            "bracket_lower_phase": self.bracket_lower_phase,
            "bracket_upper_phase": self.bracket_upper_phase,
            "bracket_lower_residual_interval_m": list(
                self.bracket_lower_residual_interval_m
            ),
            "bracket_upper_residual_interval_m": list(
                self.bracket_upper_residual_interval_m
            ),
            "selected_phase": self.selected_phase,
            "raw_axial_residual_m": self.raw_axial_residual_m,
            "selected_residual_interval_m": list(
                self.selected_residual_interval_m
            ),
            "construction": self.construction,
        }


@dataclass(frozen=True)
class CircleTriangleEndpointAudit:
    object_face_index: int
    yaw_rad: float
    position_object_m: tuple[float, float, float]
    barycentric: tuple[float, float, float]
    proposal_classification: str
    sampled_pad_winding_rates_m_per_unit: tuple[float, float]
    sampled_object_normal_rates_m_per_unit: tuple[float, float]
    sampled_anchor_directions_compatible: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "object_face_index": self.object_face_index,
            "yaw_rad": self.yaw_rad,
            "position_object_m": list(self.position_object_m),
            "barycentric": list(self.barycentric),
            "proposal_classification": self.proposal_classification,
            "sampled_pad_winding_rates_m_per_unit": list(
                self.sampled_pad_winding_rates_m_per_unit
            ),
            "sampled_object_normal_rates_m_per_unit": list(
                self.sampled_object_normal_rates_m_per_unit
            ),
            "sampled_anchor_directions_compatible": (
                self.sampled_anchor_directions_compatible
            ),
        }


@dataclass(frozen=True)
class AxialCircleSeedAudit:
    method_id: str
    parameter_domain_id: str
    parameter_layout: tuple[str, ...]
    seed_role: str
    phase_endpoint_policy: str
    circle_proposal_policy: str
    object_surface_measure: str
    pad_witness_measure: str
    parameters_unit: tuple[float, ...] | None
    object_geometry_sha256: str
    allowed_face_domain_sha256: str
    pad_witness_domain_sha256: tuple[str, ...]
    sampled_anchor_pad_names: tuple[str, str] | None
    sampled_anchor_1_object_face_index: int | None
    sampled_anchor_1_object_barycentric: (
        tuple[float, float, float] | None
    )
    sampled_anchor_1_object_position_m: (
        tuple[float, float, float] | None
    )
    sampled_axial_donor_object_face_index: int | None
    sampled_axial_donor_object_barycentric: (
        tuple[float, float, float] | None
    )
    sampled_axial_donor_object_position_m: (
        tuple[float, float, float] | None
    )
    sampled_pad_triangle_indices: tuple[int, int] | None
    sampled_pad_witness_indices: tuple[int, int] | None
    sampled_anchor_1_closure_phase: float | None
    sampled_axial_donor_coordinate_m: float | None
    phase_endpoints: tuple[AxialPhaseEndpointAudit, ...]
    unresolved_phase_interval_count: int
    axial_component_or_arithmetic_unresolved: bool
    selected_phase_endpoint_index: int | None
    circle_triangle_endpoints: tuple[CircleTriangleEndpointAudit, ...]
    selected_circle_endpoint_index: int | None
    coplanar_face_count: int
    numerically_unresolved_face_count: int
    delegated_volume_parameters_unit: tuple[float, ...] | None
    delegated_closure_audit: RayClosureAudit | None
    numerical_contract: Mapping[str, object]
    claim_limitations: tuple[str, ...]
    failure_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "parameter_domain_id": self.parameter_domain_id,
            "parameter_layout": list(self.parameter_layout),
            "seed_role": self.seed_role,
            "phase_endpoint_policy": self.phase_endpoint_policy,
            "circle_proposal_policy": self.circle_proposal_policy,
            "object_surface_measure": self.object_surface_measure,
            "pad_witness_measure": self.pad_witness_measure,
            "parameters_unit": (
                None
                if self.parameters_unit is None
                else list(self.parameters_unit)
            ),
            "object_geometry_sha256": self.object_geometry_sha256,
            "allowed_face_domain_sha256": self.allowed_face_domain_sha256,
            "pad_witness_domain_sha256": list(
                self.pad_witness_domain_sha256
            ),
            "sampled_anchor_pad_names": (
                None
                if self.sampled_anchor_pad_names is None
                else list(self.sampled_anchor_pad_names)
            ),
            "sampled_anchor_1_object_face_index": (
                self.sampled_anchor_1_object_face_index
            ),
            "sampled_anchor_1_object_barycentric": (
                None
                if self.sampled_anchor_1_object_barycentric is None
                else list(self.sampled_anchor_1_object_barycentric)
            ),
            "sampled_anchor_1_object_position_m": (
                None
                if self.sampled_anchor_1_object_position_m is None
                else list(self.sampled_anchor_1_object_position_m)
            ),
            "sampled_axial_donor_object_face_index": (
                self.sampled_axial_donor_object_face_index
            ),
            "sampled_axial_donor_object_barycentric": (
                None
                if self.sampled_axial_donor_object_barycentric is None
                else list(self.sampled_axial_donor_object_barycentric)
            ),
            "sampled_axial_donor_object_position_m": (
                None
                if self.sampled_axial_donor_object_position_m is None
                else list(self.sampled_axial_donor_object_position_m)
            ),
            "sampled_pad_triangle_indices": (
                None
                if self.sampled_pad_triangle_indices is None
                else list(self.sampled_pad_triangle_indices)
            ),
            "sampled_pad_witness_indices": (
                None
                if self.sampled_pad_witness_indices is None
                else list(self.sampled_pad_witness_indices)
            ),
            "sampled_anchor_1_closure_phase": (
                self.sampled_anchor_1_closure_phase
            ),
            "sampled_axial_donor_coordinate_m": (
                self.sampled_axial_donor_coordinate_m
            ),
            "phase_endpoints": [row.as_dict() for row in self.phase_endpoints],
            "unresolved_phase_interval_count": (
                self.unresolved_phase_interval_count
            ),
            "axial_component_or_arithmetic_unresolved": (
                self.axial_component_or_arithmetic_unresolved
            ),
            "selected_phase_endpoint_index": (
                self.selected_phase_endpoint_index
            ),
            "circle_triangle_endpoints": [
                row.as_dict() for row in self.circle_triangle_endpoints
            ],
            "selected_circle_endpoint_index": (
                self.selected_circle_endpoint_index
            ),
            "coplanar_face_count": self.coplanar_face_count,
            "numerically_unresolved_face_count": (
                self.numerically_unresolved_face_count
            ),
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
            "numerical_contract": dict(self.numerical_contract),
            "claim_limitations": list(self.claim_limitations),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class AxialCircleSeedEvaluation:
    candidate: GraspCandidate | None
    audit: AxialCircleSeedAudit
    possible_first_contact_sets: tuple[PossibleFirstContactSet, ...] = ()
    display_only_proposal: DisplayOnlyGraspProposal | None = None
    sequential_closure_policy: CertifiedSequentialClosurePolicy | None = None

    @property
    def delegated_closure_candidate_available(self) -> bool:
        """Whether delegated finite-witness closure returned a candidate."""

        return self.candidate is not None

    @property
    def representative_proposal_available(self) -> bool:
        """Whether V9 returned a non-evidentiary display proposal."""

        return self.display_only_proposal is not None

    @property
    def static_policy_available(self) -> bool:
        return self.sequential_closure_policy is not None

    @property
    def feasible(self) -> bool:
        """Optimizer protocol alias; never a physical-feasibility claim."""

        return self.delegated_closure_candidate_available


@dataclass(frozen=True)
class _SurfacePoint:
    face_index: int
    barycentric: np.ndarray
    position_object_m: np.ndarray
    position_task_m: np.ndarray
    outward_normal_object: np.ndarray


@dataclass(frozen=True)
class _WitnessState:
    position_hand_m: np.ndarray
    velocity_hand_per_unit: np.ndarray
    source_winding_normal_hand: np.ndarray


def _immutable_digest(*arrays: np.ndarray, labels: Sequence[str] = ()) -> str:
    digest = hashlib.sha256()
    for label in labels:
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    for array in arrays:
        canonical = np.ascontiguousarray(array)
        digest.update(canonical.dtype.str.encode("ascii"))
        digest.update(repr(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


class AxialConditionedCircleTriangleSeedModel:
    """ObjectSurfaceModel-compatible exact-geometry pose proposal."""

    def __init__(
        self,
        closure_model: RayClosureSurfaceModel,
        *,
        numerical_options: AxialCircleNumericalOptions,
    ) -> None:
        if not isinstance(closure_model, RayClosureSurfaceModel):
            raise AxialCircleSeedError(
                "closure_model must be a RayClosureSurfaceModel"
            )
        if not isinstance(numerical_options, AxialCircleNumericalOptions):
            raise AxialCircleSeedError(
                "numerical_options must be explicit "
                "AxialCircleNumericalOptions"
            )
        object_model = closure_model.object_model
        face_indices = np.flatnonzero(
            object_model.contact_face_mask
        ).astype(np.int64)
        if len(face_indices) == 0:
            raise AxialCircleSeedError(
                "object contract exposes no allowed contact faces"
            )
        face_areas = np.asarray(
            object_model.mesh.face_areas_m2[face_indices], dtype=np.float64
        )
        if np.any(face_areas <= 0.0) or not np.all(np.isfinite(face_areas)):
            raise AxialCircleSeedError(
                "allowed object faces need positive finite area"
            )
        cumulative_face_area = np.cumsum(face_areas, dtype=np.float64)
        total_face_area = math.fsum(float(value) for value in face_areas)
        cumulative_face_area[-1] = total_face_area
        if np.any(np.diff(cumulative_face_area) <= 0.0):
            raise AxialCircleSeedError(
                "allowed object face area CDF is not strictly increasing"
            )

        witness_cumulative_measures: list[np.ndarray] = []
        witness_total_measures: list[float] = []
        witness_domain_hashes: list[str] = []
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
                raise AxialCircleSeedError(
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
            if np.any(np.diff(cumulative) <= 0.0):
                raise AxialCircleSeedError(
                    f"verified PAD {prepared.verified.name} witness CDF "
                    "is not strictly increasing"
                )
            cumulative.setflags(write=False)
            witness_cumulative_measures.append(cumulative)
            witness_total_measures.append(total)
            witness_domain_hashes.append(
                _immutable_digest(
                    prepared.triangle_indices,
                    prepared.witness_indices,
                    prepared.barycentric_coordinates,
                    labels=(
                        prepared.verified.name,
                        prepared.verified.mesh.sha256,
                    ),
                )
            )

        basis = closure_model.task_basis_object
        origin = object_model.assembly_axis_origin_m
        allowed_triangles_object = object_model.mesh.face_vertices_m[
            face_indices
        ]
        allowed_triangles_task = (
            allowed_triangles_object - origin[None, None, :]
        ) @ basis
        allowed_normals_object = object_model.mesh.face_normals[face_indices]
        allowed_normals_task = allowed_normals_object @ basis
        allowed_plane_offsets_task = np.sum(
            allowed_normals_task * allowed_triangles_task[:, 0, :], axis=1
        )
        characteristic_length = float(
            closure_model.intersector.characteristic_length_m
        )
        if (
            not math.isfinite(characteristic_length)
            or characteristic_length <= 0
        ):
            raise AxialCircleSeedError(
                "circle equations need a positive characteristic length"
            )

        allowed_mask = np.asarray(
            object_model.contact_face_mask, dtype=np.uint8
        )
        self.allowed_face_domain_sha256 = _immutable_digest(
            allowed_mask,
            labels=tuple(sorted(object_model.allowed_contact_semantics)),
        )
        self.closure_model = closure_model
        self.hand_model = closure_model.hand_model
        self.object_model = object_model
        self.numerical_options = numerical_options
        self.allowed_face_indices = face_indices
        self.allowed_face_areas_m2 = face_areas
        self.cumulative_allowed_face_area_m2 = cumulative_face_area
        self.allowed_surface_area_m2 = total_face_area
        self.witness_cumulative_measures = tuple(witness_cumulative_measures)
        self.witness_total_measures = tuple(witness_total_measures)
        self.pad_witness_domain_sha256 = tuple(witness_domain_hashes)
        self.allowed_triangles_task_m = allowed_triangles_task
        self.allowed_normals_task = allowed_normals_task
        self.allowed_normals_object = allowed_normals_object
        self.allowed_plane_offsets_task_m = allowed_plane_offsets_task
        self.characteristic_length_m = characteristic_length
        self.ordered_pad_pairs = tuple(
            (first, second)
            for first in range(3)
            for second in range(3)
            if first != second
        )
        self.parameter_layout = PARAMETER_LAYOUT_PREFIX + tuple(
            f"preshape_joint_unit:{name}"
            for name in closure_model.preshape_joint_names
        )
        for array in (
            self.allowed_face_indices,
            self.allowed_face_areas_m2,
            self.cumulative_allowed_face_area_m2,
            self.allowed_triangles_task_m,
            self.allowed_normals_task,
            self.allowed_normals_object,
            self.allowed_plane_offsets_task_m,
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
            raise AxialCircleSeedError(
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
        position_object = barycentric @ triangle
        position_task = self.closure_model.task_basis_object.T @ (
            position_object - self.object_model.assembly_axis_origin_m
        )
        return _SurfacePoint(
            face_index=face_index,
            barycentric=barycentric,
            position_object_m=position_object,
            position_task_m=position_task,
            outward_normal_object=(
                self.object_model.mesh.face_normals[face_index]
            ),
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

    def _axial_phase_endpoints(
        self,
        *,
        second_pad_row: int,
        second_witness_flat_index: int,
        q_start: np.ndarray,
        first_state: _WitnessState,
        first_object_axial_m: float,
        second_object_axial_m: float,
    ) -> tuple[tuple[AxialPhaseEndpointAudit, ...], int, bool]:
        direction = self.closure_model.closing_directions_physical[
            second_pad_row
        ]
        maximum = self.closure_model._maximum_path_parameter(
            q_start, direction
        )
        if maximum <= 0.0:
            return (), 0, False
        target_delta = second_object_axial_m - first_object_axial_m

        def residual_with_error(phase: float) -> tuple[float, float]:
            state = self._witness_state(
                pad_row=second_pad_row,
                witness_flat_index=second_witness_flat_index,
                q_start=q_start,
                phase=phase,
            )
            value = float(
                state.position_hand_m[2]
                - first_state.position_hand_m[2]
                - target_delta
            )
            coordinate_scale = (
                self.characteristic_length_m
                + abs(float(state.position_hand_m[2]))
                + abs(float(first_state.position_hand_m[2]))
                + abs(target_delta)
            )
            return value, _FK_ERROR * coordinate_scale

        def outward_interval(
            value: float, error: float
        ) -> tuple[float, float]:
            if not math.isfinite(value) or not math.isfinite(error):
                return -math.inf, math.inf
            outward_error = float(np.nextafter(error, math.inf))
            return (
                float(np.nextafter(value - outward_error, -math.inf)),
                float(np.nextafter(value + outward_error, math.inf)),
            )

        def certified_sign(interval: tuple[float, float]) -> int:
            if interval[0] > 0.0:
                return 1
            if interval[1] < 0.0:
                return -1
            return 0

        cell_count = self.numerical_options.axial_phase_cell_count
        phases = np.linspace(0.0, maximum, cell_count + 1)
        rows = tuple(
            residual_with_error(float(phase)) for phase in phases
        )
        values = np.asarray([row[0] for row in rows], dtype=np.float64)
        errors = np.asarray([row[1] for row in rows], dtype=np.float64)
        intervals = tuple(
            outward_interval(float(value), float(error))
            for value, error in zip(values, errors)
        )
        signs = np.asarray(
            [certified_sign(interval) for interval in intervals],
            dtype=np.int8,
        )
        endpoints: list[AxialPhaseEndpointAudit] = []
        seen_bracket_keys: set[bytes] = set()
        unresolved_interval_count = 0

        def append_endpoint(
            *,
            cell: int,
            lower: float,
            upper: float,
            selected: float,
            construction: str,
        ) -> None:
            key = np.asarray((lower, upper), dtype="<f8").tobytes()
            if key in seen_bracket_keys:
                return
            seen_bracket_keys.add(key)
            lower_residual, lower_error = residual_with_error(lower)
            upper_residual, upper_error = residual_with_error(upper)
            selected_residual, selected_error = residual_with_error(selected)
            endpoints.append(
                AxialPhaseEndpointAudit(
                    grid_cell_index=cell,
                    bracket_lower_phase=lower,
                    bracket_upper_phase=upper,
                    bracket_lower_residual_interval_m=outward_interval(
                        lower_residual, lower_error
                    ),
                    bracket_upper_residual_interval_m=outward_interval(
                        upper_residual, upper_error
                    ),
                    selected_phase=selected,
                    raw_axial_residual_m=selected_residual,
                    selected_residual_interval_m=outward_interval(
                        selected_residual, selected_error
                    ),
                    construction=construction,
                )
            )

        for cell in range(cell_count):
            lower = float(phases[cell])
            upper = float(phases[cell + 1])
            lower_sign = int(signs[cell])
            upper_sign = int(signs[cell + 1])
            if lower_sign == 0 or upper_sign == 0:
                unresolved_interval_count += 1
                continue
            if lower_sign == upper_sign:
                continue
            for _iteration in range(
                self.numerical_options.axial_bisection_iterations
            ):
                middle = lower + 0.5 * (upper - lower)
                if middle == lower or middle == upper:
                    break
                middle_value, middle_error = residual_with_error(middle)
                middle_sign = certified_sign(
                    outward_interval(middle_value, middle_error)
                )
                if middle_sign == 0:
                    break
                if middle_sign == lower_sign:
                    lower = middle
                    lower_sign = middle_sign
                else:
                    upper = middle
                    upper_sign = middle_sign
            selected = lower + 0.5 * (upper - lower)
            append_endpoint(
                cell=cell,
                lower=lower,
                upper=upper,
                selected=selected,
                construction=(
                    "FORWARD_ERROR_SEPARATED_SIGN_CHANGE_BRACKET"
                ),
            )
        endpoints.sort(
            key=lambda row: (row.selected_phase, row.grid_cell_index)
        )
        unresolved = unresolved_interval_count > 0
        return tuple(endpoints), unresolved_interval_count, unresolved

    @staticmethod
    def _barycentric_with_error(
        triangle: np.ndarray, point: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        first = triangle[1] - triangle[0]
        second = triangle[2] - triangle[0]
        relative = point - triangle[0]
        d00 = float(first @ first)
        d01 = float(first @ second)
        d11 = float(second @ second)
        d20 = float(relative @ first)
        d21 = float(relative @ second)
        denominator = d00 * d11 - d01 * d01
        denominator_scale = abs(d00 * d11) + abs(d01 * d01)
        denominator_error = _gamma(128) * denominator_scale
        if denominator <= denominator_error:
            return None
        numerator_one = d11 * d20 - d01 * d21
        numerator_two = d00 * d21 - d01 * d20
        coordinate_one = numerator_one / denominator
        coordinate_two = numerator_two / denominator
        coordinate_zero = 1.0 - coordinate_one - coordinate_two
        numerator_scale = max(
            abs(d11 * d20) + abs(d01 * d21),
            abs(d00 * d21) + abs(d01 * d20),
        )
        error = _gamma(512) * (
            1.0
            + numerator_scale / (denominator - denominator_error)
            + denominator_scale / (denominator - denominator_error)
        )
        return (
            np.asarray(
                (coordinate_zero, coordinate_one, coordinate_two),
                dtype=np.float64,
            ),
            error,
        )

    def _circle_triangle_endpoints(
        self,
        *,
        first_point: _SurfacePoint,
        first_state: _WitnessState,
        second_state: _WitnessState,
    ) -> tuple[
        tuple[CircleTriangleEndpointAudit, ...],
        int,
        int,
    ]:
        chord_hand = (
            second_state.position_hand_m - first_state.position_hand_m
        )
        chord_transverse = chord_hand[:2]
        radius_squared = float(chord_transverse @ chord_transverse)
        radius_error = _gamma(128) * max(
            radius_squared, self.characteristic_length_m**2
        )
        if radius_squared <= radius_error:
            return (), 0, len(self.allowed_face_indices)

        center = first_point.position_task_m
        normals = self.allowed_normals_task
        dx = float(chord_transverse[0])
        dy = float(chord_transverse[1])
        circle_axial = float(center[2] + chord_hand[2])
        coefficient_cosine = normals[:, 0] * dx + normals[:, 1] * dy
        coefficient_sine = -normals[:, 0] * dy + normals[:, 1] * dx
        constant = (
            normals[:, 0] * center[0]
            + normals[:, 1] * center[1]
            + normals[:, 2] * circle_axial
            - self.allowed_plane_offsets_task_m
        )
        amplitude_squared = (
            coefficient_cosine * coefficient_cosine
            + coefficient_sine * coefficient_sine
        )
        equation_scale = (
            np.abs(coefficient_cosine)
            + np.abs(coefficient_sine)
            + np.abs(constant)
            + np.sum(
                np.abs(
                    normals
                    * self.allowed_triangles_task_m[:, 0, :]
                ),
                axis=1,
            )
        )
        plane_error = _gamma(256) * equation_scale
        amplitude_error = _gamma(256) * (
            coefficient_cosine * coefficient_cosine
            + coefficient_sine * coefficient_sine
        )
        discriminant = amplitude_squared - constant * constant
        discriminant_error = (
            _gamma(512)
            * (amplitude_squared + constant * constant)
            + 2.0 * np.abs(constant) * plane_error
            + plane_error * plane_error
            + amplitude_error
        )

        coplanar = (
            amplitude_squared <= amplitude_error
        ) & (np.abs(constant) <= plane_error)
        unresolved = (
            (amplitude_squared > amplitude_error)
            & (np.abs(discriminant) <= discriminant_error)
        )
        regular = (
            amplitude_squared > amplitude_error
        ) & (discriminant > discriminant_error)
        local_faces = np.flatnonzero(regular)
        endpoints: list[CircleTriangleEndpointAudit] = []
        numerically_unresolved = int(np.count_nonzero(unresolved))

        for local_face in local_faces:
            amplitude = math.sqrt(float(amplitude_squared[local_face]))
            ratio = -float(constant[local_face]) / amplitude
            ratio = min(1.0, max(-1.0, ratio))
            centre_angle = math.atan2(
                float(coefficient_sine[local_face]),
                float(coefficient_cosine[local_face]),
            )
            offset = math.acos(ratio)
            for yaw in (
                (centre_angle - offset) % (2.0 * math.pi),
                (centre_angle + offset) % (2.0 * math.pi),
            ):
                cosine = math.cos(yaw)
                sine = math.sin(yaw)
                rotated_chord = np.asarray(
                    (
                        cosine * dx - sine * dy,
                        sine * dx + cosine * dy,
                        chord_hand[2],
                    ),
                    dtype=np.float64,
                )
                point_task = center + rotated_chord
                barycentric_result = self._barycentric_with_error(
                    self.allowed_triangles_task_m[local_face], point_task
                )
                if barycentric_result is None:
                    numerically_unresolved += 1
                    continue
                barycentric, barycentric_error = barycentric_result
                if (
                    np.min(barycentric) < -barycentric_error
                    or np.max(barycentric) > 1.0 + barycentric_error
                ):
                    continue
                rotation_about_axis = np.asarray(
                    (
                        (cosine, -sine, 0.0),
                        (sine, cosine, 0.0),
                        (0.0, 0.0, 1.0),
                    ),
                    dtype=np.float64,
                )
                rotation = (
                    self.closure_model.task_basis_object
                    @ rotation_about_axis
                )
                velocities = (
                    rotation @ first_state.velocity_hand_per_unit,
                    rotation @ second_state.velocity_hand_per_unit,
                )
                pad_normals = (
                    rotation @ first_state.source_winding_normal_hand,
                    rotation @ second_state.source_winding_normal_hand,
                )
                object_normals = (
                    first_point.outward_normal_object,
                    self.allowed_normals_object[local_face],
                )
                pad_rates = tuple(
                    float(normal @ velocity)
                    for normal, velocity in zip(pad_normals, velocities)
                )
                object_rates = tuple(
                    -float(normal @ velocity)
                    for normal, velocity in zip(object_normals, velocities)
                )
                direction_errors = tuple(
                    _DOT_ERROR * float(np.linalg.norm(velocity))
                    for velocity in velocities
                )
                compatible = all(
                    pad_rate > error and object_rate > error
                    for pad_rate, object_rate, error in zip(
                        pad_rates, object_rates, direction_errors
                    )
                )
                position_object = (
                    self.object_model.assembly_axis_origin_m
                    + self.closure_model.task_basis_object @ point_task
                )
                endpoints.append(
                    CircleTriangleEndpointAudit(
                        object_face_index=int(
                            self.allowed_face_indices[local_face]
                        ),
                        yaw_rad=yaw,
                        position_object_m=tuple(
                            float(value) for value in position_object
                        ),
                        barycentric=tuple(
                            float(value) for value in barycentric
                        ),
                        proposal_classification=(
                            "REGULAR_NONCOPLANAR_CIRCLE_TRIANGLE_SEED"
                        ),
                        sampled_pad_winding_rates_m_per_unit=pad_rates,
                        sampled_object_normal_rates_m_per_unit=object_rates,
                        sampled_anchor_directions_compatible=compatible,
                    )
                )
        endpoints.sort(
            key=lambda row: (row.object_face_index, row.yaw_rad)
        )
        return (
            tuple(endpoints),
            int(np.count_nonzero(coplanar)),
            numerically_unresolved,
        )

    def _audit(
        self,
        *,
        parameters: np.ndarray | None = None,
        pad_pair: tuple[int, int] | None = None,
        first_point: _SurfacePoint | None = None,
        axial_donor_point: _SurfacePoint | None = None,
        witness_indices: tuple[int, int] | None = None,
        first_phase: float | None = None,
        phase_endpoints: tuple[AxialPhaseEndpointAudit, ...] = (),
        unresolved_phase_interval_count: int = 0,
        axial_component_or_arithmetic_unresolved: bool = False,
        selected_phase_endpoint_index: int | None = None,
        circle_endpoints: tuple[CircleTriangleEndpointAudit, ...] = (),
        selected_circle_endpoint_index: int | None = None,
        coplanar_face_count: int = 0,
        numerically_unresolved_face_count: int = 0,
        volume_parameters: np.ndarray | None = None,
        closure_audit: RayClosureAudit | None = None,
        failure_reason: str | None = None,
    ) -> AxialCircleSeedAudit:
        names = None
        triangle_indices = None
        witness_numbers = None
        if pad_pair is not None:
            names = tuple(
                self.closure_model.prepared_pads[row].verified.name
                for row in pad_pair
            )
        if pad_pair is not None and witness_indices is not None:
            triangle_indices = tuple(
                int(
                    self.closure_model.prepared_pads[row].triangle_indices[
                        witness
                    ]
                )
                for row, witness in zip(pad_pair, witness_indices)
            )
            witness_numbers = tuple(
                int(
                    self.closure_model.prepared_pads[row].witness_indices[
                        witness
                    ]
                )
                for row, witness in zip(pad_pair, witness_indices)
            )
        return AxialCircleSeedAudit(
            method_id=METHOD_ID,
            parameter_domain_id=PARAMETER_DOMAIN_ID,
            parameter_layout=self.parameter_layout,
            seed_role=SEED_ROLE,
            phase_endpoint_policy=PHASE_ENDPOINT_POLICY,
            circle_proposal_policy=CIRCLE_PROPOSAL_POLICY,
            object_surface_measure=OBJECT_SURFACE_MEASURE,
            pad_witness_measure=PAD_WITNESS_MEASURE,
            parameters_unit=(
                None
                if parameters is None
                else tuple(float(value) for value in parameters)
            ),
            object_geometry_sha256=self.object_model.geometry_sha256,
            allowed_face_domain_sha256=self.allowed_face_domain_sha256,
            pad_witness_domain_sha256=self.pad_witness_domain_sha256,
            sampled_anchor_pad_names=names,
            sampled_anchor_1_object_face_index=(
                None if first_point is None else first_point.face_index
            ),
            sampled_anchor_1_object_barycentric=(
                None
                if first_point is None
                else tuple(float(value) for value in first_point.barycentric)
            ),
            sampled_anchor_1_object_position_m=(
                None
                if first_point is None
                else tuple(
                    float(value) for value in first_point.position_object_m
                )
            ),
            sampled_axial_donor_object_face_index=(
                None
                if axial_donor_point is None
                else axial_donor_point.face_index
            ),
            sampled_axial_donor_object_barycentric=(
                None
                if axial_donor_point is None
                else tuple(
                    float(value) for value in axial_donor_point.barycentric
                )
            ),
            sampled_axial_donor_object_position_m=(
                None
                if axial_donor_point is None
                else tuple(
                    float(value)
                    for value in axial_donor_point.position_object_m
                )
            ),
            sampled_pad_triangle_indices=triangle_indices,
            sampled_pad_witness_indices=witness_numbers,
            sampled_anchor_1_closure_phase=first_phase,
            sampled_axial_donor_coordinate_m=(
                None
                if axial_donor_point is None
                else float(axial_donor_point.position_task_m[2])
            ),
            phase_endpoints=phase_endpoints,
            unresolved_phase_interval_count=(
                unresolved_phase_interval_count
            ),
            axial_component_or_arithmetic_unresolved=(
                axial_component_or_arithmetic_unresolved
            ),
            selected_phase_endpoint_index=selected_phase_endpoint_index,
            circle_triangle_endpoints=circle_endpoints,
            selected_circle_endpoint_index=selected_circle_endpoint_index,
            coplanar_face_count=coplanar_face_count,
            numerically_unresolved_face_count=(
                numerically_unresolved_face_count
            ),
            delegated_volume_parameters_unit=(
                None
                if volume_parameters is None
                else tuple(float(value) for value in volume_parameters)
            ),
            delegated_closure_audit=closure_audit,
            numerical_contract=self.numerical_options.contract,
            claim_limitations=CLAIM_LIMITATIONS,
            failure_reason=failure_reason,
        )

    def _parameter_failure(
        self, reason: str, parameters: np.ndarray | None = None
    ) -> AxialCircleSeedEvaluation:
        return AxialCircleSeedEvaluation(
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
    ) -> AxialCircleSeedEvaluation:
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
            if index != 7 and parameters[index] >= 1.0:
                return self._parameter_failure(
                    f"{label} uses a canonical half-open unit interval",
                    parameters,
                )

        try:
            first_point = self._surface_point(
                float(parameters[1]), float(parameters[2])
            )
            axial_donor_point = self._surface_point(
                float(parameters[3]), float(parameters[4])
            )
        except AxialCircleSeedError as error:
            return self._parameter_failure(str(error), parameters)
        pair_index = min(
            int(float(parameters[0]) * len(self.ordered_pad_pairs)),
            len(self.ordered_pad_pairs) - 1,
        )
        pad_pair = self.ordered_pad_pairs[pair_index]
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
        first_direction = self.closure_model.closing_directions_physical[
            pad_pair[0]
        ]
        first_maximum = self.closure_model._maximum_path_parameter(
            q_start, first_direction
        )
        first_phase = float(parameters[7]) * first_maximum
        first_state = self._witness_state(
            pad_row=pad_pair[0],
            witness_flat_index=witness_indices[0],
            q_start=q_start,
            phase=first_phase,
        )
        second_object_axial = float(
            axial_donor_point.position_task_m[2]
        )
        (
            phase_endpoints,
            unresolved_phase_interval_count,
            axial_component_or_arithmetic_unresolved,
        ) = self._axial_phase_endpoints(
            second_pad_row=pad_pair[1],
            second_witness_flat_index=witness_indices[1],
            q_start=q_start,
            first_state=first_state,
            first_object_axial_m=float(first_point.position_task_m[2]),
            second_object_axial_m=second_object_axial,
        )
        if not phase_endpoints:
            return AxialCircleSeedEvaluation(
                None,
                self._audit(
                    parameters=parameters,
                    pad_pair=pad_pair,
                    first_point=first_point,
                    axial_donor_point=axial_donor_point,
                    witness_indices=witness_indices,
                    first_phase=first_phase,
                    unresolved_phase_interval_count=(
                        unresolved_phase_interval_count
                    ),
                    axial_component_or_arithmetic_unresolved=(
                        axial_component_or_arithmetic_unresolved
                    ),
                    failure_reason=(
                        "AXIAL_PHASE_COMPONENT_OR_ARITHMETIC_UNRESOLVED"
                        if axial_component_or_arithmetic_unresolved
                        else "NO_BRACKETED_AXIAL_PHASE_ENDPOINT"
                    ),
                ),
            )
        selected_phase_index = min(
            int(float(parameters[8]) * len(phase_endpoints)),
            len(phase_endpoints) - 1,
        )
        selected_phase = phase_endpoints[selected_phase_index]
        second_state = self._witness_state(
            pad_row=pad_pair[1],
            witness_flat_index=witness_indices[1],
            q_start=q_start,
            phase=selected_phase.selected_phase,
        )
        (
            circle_endpoints,
            coplanar_face_count,
            numerically_unresolved_face_count,
        ) = self._circle_triangle_endpoints(
            first_point=first_point,
            first_state=first_state,
            second_state=second_state,
        )
        if not circle_endpoints:
            return AxialCircleSeedEvaluation(
                None,
                self._audit(
                    parameters=parameters,
                    pad_pair=pad_pair,
                    first_point=first_point,
                    axial_donor_point=axial_donor_point,
                    witness_indices=witness_indices,
                    first_phase=first_phase,
                    phase_endpoints=phase_endpoints,
                    unresolved_phase_interval_count=(
                        unresolved_phase_interval_count
                    ),
                    axial_component_or_arithmetic_unresolved=(
                        axial_component_or_arithmetic_unresolved
                    ),
                    selected_phase_endpoint_index=selected_phase_index,
                    coplanar_face_count=coplanar_face_count,
                    numerically_unresolved_face_count=(
                        numerically_unresolved_face_count
                    ),
                    failure_reason="NO_REGULAR_CIRCLE_TRIANGLE_ENDPOINT",
                ),
            )
        selected_circle_index = min(
            int(float(parameters[9]) * len(circle_endpoints)),
            len(circle_endpoints) - 1,
        )
        selected_circle = circle_endpoints[selected_circle_index]
        yaw = selected_circle.yaw_rad
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        rotation_about_axis = np.asarray(
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        rotation = self.closure_model.task_basis_object @ rotation_about_axis
        translation = (
            first_point.position_object_m
            - rotation @ first_state.position_hand_m
        )
        focus_result = self.closure_model._closure_focus_hand(q_start)
        if focus_result is None:
            return AxialCircleSeedEvaluation(
                None,
                self._audit(
                    parameters=parameters,
                    pad_pair=pad_pair,
                    first_point=first_point,
                    axial_donor_point=axial_donor_point,
                    witness_indices=witness_indices,
                    first_phase=first_phase,
                    phase_endpoints=phase_endpoints,
                    unresolved_phase_interval_count=(
                        unresolved_phase_interval_count
                    ),
                    axial_component_or_arithmetic_unresolved=(
                        axial_component_or_arithmetic_unresolved
                    ),
                    selected_phase_endpoint_index=selected_phase_index,
                    circle_endpoints=circle_endpoints,
                    selected_circle_endpoint_index=selected_circle_index,
                    coplanar_face_count=coplanar_face_count,
                    numerically_unresolved_face_count=(
                        numerically_unresolved_face_count
                    ),
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
            volume_parameters[0] = yaw / (2.0 * math.pi)
            volume_parameters[1] = _recover_closed_unit_coordinate(
                float(target_coordinates[2]),
                float(lower[2]),
                float(upper[2]),
                absolute_error=comparison_error_m,
                label="axial-circle axial target",
            )
            volume_parameters[2] = _recover_closed_unit_coordinate(
                float(target_coordinates[0]),
                float(lower[0]),
                float(upper[0]),
                absolute_error=comparison_error_m,
                label="axial-circle first lateral target",
            )
            volume_parameters[3] = _recover_closed_unit_coordinate(
                float(target_coordinates[1]),
                float(lower[1]),
                float(upper[1]),
                absolute_error=comparison_error_m,
                label="axial-circle second lateral target",
            )
            volume_parameters[4:] = parameters[
                len(PARAMETER_LAYOUT_PREFIX):
            ]
        except RayClosureError as error:
            return AxialCircleSeedEvaluation(
                None,
                self._audit(
                    parameters=parameters,
                    pad_pair=pad_pair,
                    first_point=first_point,
                    axial_donor_point=axial_donor_point,
                    witness_indices=witness_indices,
                    first_phase=first_phase,
                    phase_endpoints=phase_endpoints,
                    unresolved_phase_interval_count=(
                        unresolved_phase_interval_count
                    ),
                    axial_component_or_arithmetic_unresolved=(
                        axial_component_or_arithmetic_unresolved
                    ),
                    selected_phase_endpoint_index=selected_phase_index,
                    circle_endpoints=circle_endpoints,
                    selected_circle_endpoint_index=selected_circle_index,
                    coplanar_face_count=coplanar_face_count,
                    numerically_unresolved_face_count=(
                        numerically_unresolved_face_count
                    ),
                    failure_reason=(
                        f"SEED_OUTSIDE_COMMON_SWEPT_DOMAIN:{error}"
                    ),
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
        return AxialCircleSeedEvaluation(
            closure.candidate,
            self._audit(
                parameters=parameters,
                pad_pair=pad_pair,
                first_point=first_point,
                axial_donor_point=axial_donor_point,
                witness_indices=witness_indices,
                first_phase=first_phase,
                phase_endpoints=phase_endpoints,
                unresolved_phase_interval_count=(
                    unresolved_phase_interval_count
                ),
                axial_component_or_arithmetic_unresolved=(
                    axial_component_or_arithmetic_unresolved
                ),
                selected_phase_endpoint_index=selected_phase_index,
                circle_endpoints=circle_endpoints,
                selected_circle_endpoint_index=selected_circle_index,
                coplanar_face_count=coplanar_face_count,
                numerically_unresolved_face_count=(
                    numerically_unresolved_face_count
                ),
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
                "phase_endpoint_policy": PHASE_ENDPOINT_POLICY,
                "circle_proposal_policy": CIRCLE_PROPOSAL_POLICY,
                "object_surface_measure": OBJECT_SURFACE_MEASURE,
                "pad_witness_measure": PAD_WITNESS_MEASURE,
                "object_geometry_sha256": self.object_model.geometry_sha256,
                "allowed_face_domain_sha256": (
                    self.allowed_face_domain_sha256
                ),
                "pad_witness_domain_sha256": (
                    self.pad_witness_domain_sha256
                ),
                "numerical_contract": self.numerical_options.contract,
                "claim_limitations": CLAIM_LIMITATIONS,
                "delegated_closure_contract": self.closure_model.contract,
            }
        )


__all__ = [
    "AxialCircleNumericalOptions",
    "AxialCircleSeedAudit",
    "AxialCircleSeedError",
    "AxialCircleSeedEvaluation",
    "AxialConditionedCircleTriangleSeedModel",
    "AxialPhaseEndpointAudit",
    "CIRCLE_PROPOSAL_POLICY",
    "CLAIM_LIMITATIONS",
    "CircleTriangleEndpointAudit",
    "METHOD_ID",
    "OBJECT_SURFACE_MEASURE",
    "PAD_WITNESS_MEASURE",
    "PARAMETER_DOMAIN_ID",
    "PARAMETER_LAYOUT_PREFIX",
    "PHASE_ENDPOINT_POLICY",
    "SEED_ROLE",
]
