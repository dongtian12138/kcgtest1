"""Canonical area-measure pose proposals for exact-FK PAD closure.

The volume chart in :mod:`ray_closure` is a conservative completeness chart,
not an efficient proposal distribution.  This module maps a unit-hypercube
point directly to a canonical unoriented object-triangle anchor, constructs a
hand translation from one PAD-winding-compatible finite witness, and delegates
every object-contact and direction claim to the V9 closure certifier.

The anchor is only a pose seed.  It is never accepted as contact truth: a
hidden face, an earlier obstruction, or a different first contact is handled
by the delegated closure evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import ClassVar, Mapping, Sequence

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
    _FK_ERROR,
    _gamma,
    _recover_closed_unit_coordinate,
)


METHOD_ID = (
    "CARTS_CANONICAL_AREA_MEASURE_SURFACE_ANCHORED_"
    "DELEGATED_V9_CLOSURE_POSE_SEED_V3"
)
PARAMETER_DOMAIN_ID = (
    "HALF_OPEN_ASSEMBLY_YAW_X_THREE_PAD_SELECTOR_X_TWO_DIMENSIONAL_"
    "RESIDUAL_CDF_CANONICAL_UNORIENTED_TRIANGLE_AREA_CHART_X_"
    "CLOSURE_PHASE_X_UNIFORM_PAD_SOURCE_WINDING_SIGN_CERTIFIED_"
    "WITNESS_BRANCH_X_NONCLOSURE_PRESHAPE_V3"
)
FIXED_ANCHOR_METHOD_ID = (
    "CARTS_CANONICAL_AREA_MEASURE_FIXED_PAD_SURFACE_ANCHOR_"
    "V9_PARAMETER_POSE_PROPOSAL_V1"
)
FIXED_ANCHOR_PARAMETER_DOMAIN_ID = (
    "HALF_OPEN_ASSEMBLY_YAW_X_TWO_DIMENSIONAL_RESIDUAL_CDF_"
    "CANONICAL_UNORIENTED_TRIANGLE_AREA_CHART_X_CLOSURE_PHASE_X_"
    "UNIFORM_PAD_SOURCE_WINDING_SIGN_CERTIFIED_WITNESS_BRANCH_X_"
    "NONCLOSURE_PRESHAPE_WITH_EXTERNAL_EXACT_PREPARED_PAD_NAME_V1"
)
SURFACE_MEASURE = "ALLOWED_CANONICAL_UNORIENTED_FACE_TRIANGLE_AREA_MEASURE"
ANCHOR_ROLE = (
    "POSE_PROPOSAL_ONLY_OBJECT_CONTACT_AND_DIRECTION_FROM_DELEGATED_V9"
)
WITNESS_SELECTION = (
    "UNIFORM_BRANCH_OVER_ALL_NONZERO_SPEED_HASH_BOUND_PAD_SOURCE_"
    "WINDING_APPROACH_CERTIFIED_FINITE_WITNESSES"
)
PARAMETER_LAYOUT_PREFIX = (
    "assembly_axis_yaw_unit",
    "anchor_pad_selector_unit",
    "object_surface_residual_cdf_unit",
    "object_surface_split_unit",
    "anchor_closure_phase_unit",
    "anchor_witness_branch_unit",
)
FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX = (
    "assembly_axis_yaw_unit",
    "object_surface_residual_cdf_unit",
    "object_surface_split_unit",
    "anchor_closure_phase_unit",
    "anchor_witness_branch_unit",
)
CLAIM_LIMITATIONS = (
    "ANCHOR_POSITION_IS_A_PROPOSAL_NOT_OBJECT_CONTACT_TRUTH",
    "FIXED_ANCHOR_MAPPER_DOES_NOT_CALL_OR_CERTIFY_DELEGATED_V9",
    "OBJECT_SOURCE_FACE_WINDING_AND_NORMAL_NOT_CONSUMED_BY_PROPOSAL",
    "PAD_SOURCE_WINDING_APPROACH_IS_PROPOSAL_COMPATIBILITY_ONLY",
    "OBJECT_CONTACT_AND_DIRECTION_CLASSIFICATION_ONLY_FROM_DELEGATED_V9",
    "FINITE_PAD_WITNESS_AND_OBJECT_MESH_CONVERGENCE_REQUIRED",
    "AREA_MEASURE_PROPOSAL_IS_NOT_A_GLOBAL_OPTIMALITY_CERTIFICATE",
    "GENERATION_PARAMETERS_REQUIRED_FOR_GENERATION_REPLAY",
    "FINITE_WITNESS_BRANCHES_NOT_EXHAUSTIVELY_EVALUATED_AT_FINITE_BUDGET",
    "DELEGATED_V9_CLOSURE_AND_COMPLETE_COLLISION_LIMITATIONS_APPLY",
)


class SurfaceAnchoredClosureError(ValueError):
    """Raised when the immutable anchor model contract is invalid."""


@dataclass(frozen=True)
class SurfaceAnchorAudit:
    method_id: str
    parameter_domain_id: str
    parameter_layout: tuple[str, ...]
    surface_measure: str
    anchor_role: str
    witness_selection: str
    parameters_unit: tuple[float, ...] | None
    object_geometry_sha256: str
    allowed_face_count: int
    allowed_surface_area_m2: float
    anchor_pad_name: str | None
    anchor_object_face_index: int | None
    anchor_object_barycentric: tuple[float, float, float] | None
    anchor_object_position_m: tuple[float, float, float] | None
    anchor_closure_phase: float | None
    selected_pad_triangle_index: int | None
    selected_pad_witness_index: int | None
    selected_normalised_pad_source_winding_approach_margin: float | None
    compatible_witness_count: int | None
    selected_witness_branch_index: int | None
    delegated_volume_parameters_unit: tuple[float, ...] | None
    delegated_closure_audit: RayClosureAudit | None
    claim_limitations: tuple[str, ...]
    failure_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "parameter_domain_id": self.parameter_domain_id,
            "parameter_layout": list(self.parameter_layout),
            "surface_measure": self.surface_measure,
            "anchor_role": self.anchor_role,
            "witness_selection": self.witness_selection,
            "parameters_unit": (
                None
                if self.parameters_unit is None
                else list(self.parameters_unit)
            ),
            "object_geometry_sha256": self.object_geometry_sha256,
            "allowed_face_count": self.allowed_face_count,
            "allowed_surface_area_m2": self.allowed_surface_area_m2,
            "anchor_pad_name": self.anchor_pad_name,
            "anchor_object_face_index": self.anchor_object_face_index,
            "anchor_object_barycentric": (
                None
                if self.anchor_object_barycentric is None
                else list(self.anchor_object_barycentric)
            ),
            "anchor_object_position_m": (
                None
                if self.anchor_object_position_m is None
                else list(self.anchor_object_position_m)
            ),
            "anchor_closure_phase": self.anchor_closure_phase,
            "selected_pad_triangle_index": self.selected_pad_triangle_index,
            "selected_pad_witness_index": self.selected_pad_witness_index,
            "selected_normalised_pad_source_winding_approach_margin": (
                self.selected_normalised_pad_source_winding_approach_margin
            ),
            "compatible_witness_count": self.compatible_witness_count,
            "selected_witness_branch_index": (
                self.selected_witness_branch_index
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
            "claim_limitations": list(self.claim_limitations),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class SurfaceAnchoredClosureEvaluation:
    candidate: GraspCandidate | None
    audit: SurfaceAnchorAudit
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
class SurfaceAnchorProposal:
    """Selector-free pose seed; never a contact or closure certificate."""

    v9_parameters_unit: tuple[float, ...] | None
    audit: SurfaceAnchorAudit

    @property
    def feasible(self) -> bool:
        return self.v9_parameters_unit is not None


class SurfaceAnchoredRayClosureModel:
    """ObjectSurfaceModel-compatible area-measure contact-conditioned seeds."""

    fixed_anchor_method_id: ClassVar[str] = FIXED_ANCHOR_METHOD_ID
    fixed_anchor_parameter_domain_id: ClassVar[str] = (
        FIXED_ANCHOR_PARAMETER_DOMAIN_ID
    )

    def __init__(self, closure_model: RayClosureSurfaceModel) -> None:
        if not isinstance(closure_model, RayClosureSurfaceModel):
            raise SurfaceAnchoredClosureError(
                "closure_model must be a RayClosureSurfaceModel"
            )
        face_indices = np.flatnonzero(
            closure_model.object_model.contact_face_mask
        ).astype(np.int64)
        if len(face_indices) == 0:
            raise SurfaceAnchoredClosureError(
                "object contract exposes no allowed contact faces"
            )
        canonical_triangles = np.asarray(
            closure_model.canonical_object_face_vertices_m[face_indices],
            dtype=np.float64,
        )
        canonical_area_vectors = 0.5 * np.cross(
            canonical_triangles[:, 1] - canonical_triangles[:, 0],
            canonical_triangles[:, 2] - canonical_triangles[:, 0],
        )
        areas = np.linalg.norm(canonical_area_vectors, axis=1)
        total_area = math.fsum(float(value) for value in areas)
        if (
            not math.isfinite(total_area)
            or total_area <= 0.0
            or np.any(areas <= 0.0)
            or not np.all(np.isfinite(areas))
        ):
            raise SurfaceAnchoredClosureError(
                "allowed object faces need positive finite area"
            )
        cumulative = np.cumsum(areas, dtype=np.float64)
        cumulative[-1] = total_area
        for array in (face_indices, areas, cumulative):
            array.setflags(write=False)
        self.closure_model = closure_model
        self.hand_model = closure_model.hand_model
        self.object_model = closure_model.object_model
        self.allowed_face_indices = face_indices
        self.allowed_face_areas_m2 = areas
        self.cumulative_allowed_area_m2 = cumulative
        self.allowed_surface_area_m2 = total_area
        self.parameter_layout = PARAMETER_LAYOUT_PREFIX + tuple(
            f"preshape_joint_unit:{name}"
            for name in closure_model.preshape_joint_names
        )
        self.fixed_anchor_parameter_layout = (
            FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX
            + tuple(
                f"preshape_joint_unit:{name}"
                for name in closure_model.preshape_joint_names
            )
        )
        prepared_pad_names = tuple(
            row.verified.name for row in closure_model.prepared_pads
        )
        if len(set(prepared_pad_names)) != len(prepared_pad_names):
            raise SurfaceAnchoredClosureError(
                "prepared PAD names must be unique for fixed-anchor lanes"
            )
        self.prepared_pad_names = prepared_pad_names
        self.prepared_pad_rows = MappingProxyType(
            {name: index for index, name in enumerate(prepared_pad_names)}
        )

    @property
    def parameter_dimension(self) -> int:
        return len(self.parameter_layout)

    @property
    def fixed_anchor_parameter_dimension(self) -> int:
        return len(self.fixed_anchor_parameter_layout)

    def _audit(
        self,
        *,
        parameters_unit: np.ndarray | None = None,
        anchor_pad_name: str | None = None,
        object_face_index: int | None = None,
        object_barycentric: np.ndarray | None = None,
        object_position_m: np.ndarray | None = None,
        closure_phase: float | None = None,
        witness_flat_index: int | None = None,
        normalised_pad_source_winding_approach_margin: float | None = None,
        compatible_witness_count: int | None = None,
        witness_branch_index: int | None = None,
        volume_parameters: np.ndarray | None = None,
        closure_audit: RayClosureAudit | None = None,
        failure_reason: str | None = None,
        method_id: str = METHOD_ID,
        parameter_domain_id: str = PARAMETER_DOMAIN_ID,
        parameter_layout: tuple[str, ...] | None = None,
    ) -> SurfaceAnchorAudit:
        triangle_index = None
        witness_index = None
        if witness_flat_index is not None and anchor_pad_name is not None:
            prepared = next(
                row
                for row in self.closure_model.prepared_pads
                if row.verified.name == anchor_pad_name
            )
            triangle_index = int(
                prepared.triangle_indices[witness_flat_index]
            )
            witness_index = int(
                prepared.witness_indices[witness_flat_index]
            )
        return SurfaceAnchorAudit(
            method_id=method_id,
            parameter_domain_id=parameter_domain_id,
            parameter_layout=(
                self.parameter_layout
                if parameter_layout is None
                else parameter_layout
            ),
            surface_measure=SURFACE_MEASURE,
            anchor_role=ANCHOR_ROLE,
            witness_selection=WITNESS_SELECTION,
            parameters_unit=(
                None
                if parameters_unit is None
                else tuple(float(value) for value in parameters_unit)
            ),
            object_geometry_sha256=self.object_model.geometry_sha256,
            allowed_face_count=len(self.allowed_face_indices),
            allowed_surface_area_m2=self.allowed_surface_area_m2,
            anchor_pad_name=anchor_pad_name,
            anchor_object_face_index=object_face_index,
            anchor_object_barycentric=(
                None
                if object_barycentric is None
                else tuple(float(value) for value in object_barycentric)
            ),
            anchor_object_position_m=(
                None
                if object_position_m is None
                else tuple(float(value) for value in object_position_m)
            ),
            anchor_closure_phase=closure_phase,
            selected_pad_triangle_index=triangle_index,
            selected_pad_witness_index=witness_index,
            selected_normalised_pad_source_winding_approach_margin=(
                normalised_pad_source_winding_approach_margin
            ),
            compatible_witness_count=compatible_witness_count,
            selected_witness_branch_index=witness_branch_index,
            delegated_volume_parameters_unit=(
                None
                if volume_parameters is None
                else tuple(float(value) for value in volume_parameters)
            ),
            delegated_closure_audit=closure_audit,
            claim_limitations=CLAIM_LIMITATIONS,
            failure_reason=failure_reason,
        )

    def _parameter_failure(
        self, reason: str, parameters: np.ndarray | None = None
    ) -> SurfaceAnchoredClosureEvaluation:
        return SurfaceAnchoredClosureEvaluation(
            None,
            self._audit(
                parameters_unit=parameters,
                failure_reason=f"PARAMETER_DOMAIN_REJECTED:{reason}",
            ),
        )

    def _proposal_audit(
        self,
        **kwargs: object,
    ) -> SurfaceAnchorAudit:
        return self._audit(
            method_id=FIXED_ANCHOR_METHOD_ID,
            parameter_domain_id=FIXED_ANCHOR_PARAMETER_DOMAIN_ID,
            parameter_layout=self.fixed_anchor_parameter_layout,
            **kwargs,
        )

    def _proposal_failure(
        self,
        reason: str,
        *,
        parameters: np.ndarray | None = None,
        anchor_pad_name: str | None = None,
    ) -> SurfaceAnchorProposal:
        return SurfaceAnchorProposal(
            None,
            self._proposal_audit(
                parameters_unit=parameters,
                anchor_pad_name=anchor_pad_name,
                failure_reason=reason,
            ),
        )

    def _evaluation_audit_from_proposal(
        self,
        proposal_audit: SurfaceAnchorAudit,
        parameters: np.ndarray,
        *,
        closure_audit: RayClosureAudit | None = None,
        failure_reason: str | None = None,
    ) -> SurfaceAnchorAudit:
        return replace(
            proposal_audit,
            method_id=METHOD_ID,
            parameter_domain_id=PARAMETER_DOMAIN_ID,
            parameter_layout=self.parameter_layout,
            parameters_unit=tuple(float(value) for value in parameters),
            delegated_closure_audit=closure_audit,
            failure_reason=(
                proposal_audit.failure_reason
                if failure_reason is None
                else failure_reason
            ),
        )

    def propose_fixed_anchor(
        self,
        parameters6: Sequence[float],
        anchor_pad_name: str,
        hand_model: ThreeFingerHandModel,
    ) -> SurfaceAnchorProposal:
        self.closure_model._validate_hand(hand_model)
        parameters = np.asarray(parameters6, dtype=np.float64)
        if parameters.shape != (
            self.fixed_anchor_parameter_dimension,
        ) or not np.all(
            np.isfinite(parameters)
        ):
            return self._proposal_failure(
                "PARAMETER_DOMAIN_REJECTED:parameters need finite shape "
                f"({self.fixed_anchor_parameter_dimension},)",
                anchor_pad_name=anchor_pad_name,
            )
        if np.any(parameters < 0.0) or np.any(parameters > 1.0):
            return self._proposal_failure(
                "PARAMETER_DOMAIN_REJECTED:parameters must lie within [0, 1]",
                parameters=parameters,
                anchor_pad_name=anchor_pad_name,
            )
        for index, label in (
            (0, "assembly-axis yaw"),
            (1, "object residual-CDF surface"),
            (2, "object surface split"),
            (4, "anchor witness branch"),
        ):
            if parameters[index] >= 1.0:
                return self._proposal_failure(
                    "PARAMETER_DOMAIN_REJECTED:"
                    f"{label} uses a canonical half-open unit interval",
                    parameters=parameters,
                    anchor_pad_name=anchor_pad_name,
                )
        if (
            not isinstance(anchor_pad_name, str)
            or anchor_pad_name not in self.prepared_pad_rows
        ):
            return self._proposal_failure(
                "ANCHOR_PAD_REJECTED:anchor_pad_name must exactly match one "
                "prepared PAD name",
                parameters=parameters,
                anchor_pad_name=(
                    anchor_pad_name
                    if isinstance(anchor_pad_name, str)
                    else None
                ),
            )
        pad_row = self.prepared_pad_rows[anchor_pad_name]
        prepared = self.closure_model.prepared_pads[pad_row]

        yaw = 2.0 * math.pi * float(parameters[0])
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        rotation_about_axis = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        rotation = self.closure_model.task_basis_object @ rotation_about_axis
        q_start = np.array(self.closure_model.open_joint_template, copy=True)
        for unit_value, joint_index in zip(
            parameters[len(FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX):],
            self.closure_model.preshape_joint_indices,
        ):
            q_start[joint_index] = (
                self.closure_model.lower_joint_limits[joint_index]
                + float(unit_value)
                * self.closure_model.joint_spans[joint_index]
            )

        area_coordinate = float(parameters[1]) * self.allowed_surface_area_m2
        local_face = int(
            np.searchsorted(
                self.cumulative_allowed_area_m2,
                area_coordinate,
                side="right",
            )
        )
        local_face = min(local_face, len(self.allowed_face_indices) - 1)
        object_face_index = int(self.allowed_face_indices[local_face])
        preceding_area = (
            0.0
            if local_face == 0
            else float(self.cumulative_allowed_area_m2[local_face - 1])
        )
        residual = (
            area_coordinate - preceding_area
        ) / float(self.allowed_face_areas_m2[local_face])
        residual = min(1.0, max(0.0, residual))
        split = float(parameters[2])
        if residual == 0.0 and split != 0.0:
            return self._proposal_failure(
                "PARAMETER_DOMAIN_REJECTED:zero residual-CDF radius requires "
                "canonical split coordinate 0",
                parameters=parameters,
                anchor_pad_name=anchor_pad_name,
            )
        radial = math.sqrt(residual)
        barycentric = np.asarray(
            (1.0 - radial, radial * (1.0 - split), radial * split),
            dtype=np.float64,
        )
        object_triangle = self.closure_model.canonical_object_face_vertices_m[
            object_face_index
        ]
        object_position = barycentric @ object_triangle

        direction = self.closure_model.closing_directions_physical[pad_row]
        maximum_parameter = self.closure_model._maximum_path_parameter(
            q_start, direction
        )
        closure_phase = float(parameters[3]) * maximum_parameter
        q_anchor = q_start + closure_phase * direction
        orientation_only = np.eye(4, dtype=np.float64)
        orientation_only[:3, :3] = rotation
        states = self.closure_model._witness_states(
            prepared, q_anchor, direction, orientation_only
        )
        velocities = states.velocities_object_per_unit
        speeds = np.linalg.norm(velocities, axis=1)
        pad_approach = np.sum(
            states.pad_source_winding_normals_object * velocities, axis=1
        )
        errors = states.leading_error_bounds
        compatible = (
            (speeds > 0.0)
            & (pad_approach > errors)
        )
        compatible_indices = np.flatnonzero(compatible)
        if len(compatible_indices) == 0:
            return SurfaceAnchorProposal(
                None,
                self._proposal_audit(
                    anchor_pad_name=anchor_pad_name,
                    parameters_unit=parameters,
                    object_face_index=object_face_index,
                    object_barycentric=barycentric,
                    object_position_m=object_position,
                    closure_phase=closure_phase,
                    failure_reason="NO_KINEMATICALLY_COMPATIBLE_PAD_WITNESS",
                ),
            )
        branch_index = min(
            int(float(parameters[4]) * len(compatible_indices)),
            len(compatible_indices) - 1,
        )
        witness_flat_index = int(compatible_indices[branch_index])
        selected_speed = float(speeds[witness_flat_index])
        selected_pad_margin = float(
            pad_approach[witness_flat_index] / selected_speed
        )

        anchor_without_translation = states.positions_object_m[
            witness_flat_index
        ]
        translation = object_position - anchor_without_translation
        focus_result = self.closure_model._closure_focus_hand(q_start)
        if focus_result is None:
            return SurfaceAnchorProposal(
                None,
                self._proposal_audit(
                    anchor_pad_name=anchor_pad_name,
                    parameters_unit=parameters,
                    object_face_index=object_face_index,
                    object_barycentric=barycentric,
                    object_position_m=object_position,
                    closure_phase=closure_phase,
                    witness_flat_index=witness_flat_index,
                    normalised_pad_source_winding_approach_margin=(
                        selected_pad_margin
                    ),
                    compatible_witness_count=len(compatible_indices),
                    witness_branch_index=branch_index,
                    failure_reason="CLOSING_FOCUS_UNDEFINED_FROM_PAD_KINEMATICS",
                ),
            )
        focus_hand, hand_extent = focus_result
        target = rotation @ focus_hand + translation

        try:
            lower, upper = self.closure_model._placement_coordinate_bounds(
                q_start,
                rotation,
                focus_result=focus_result,
            )
            target_coordinates = self.closure_model.task_basis_object.T @ (
                target - self.object_model.assembly_axis_origin_m
            )
            comparison_error_m = (
                self.closure_model.intersector.distance_error_bound_m
                + self.closure_model.distance_bvh.aabb_error_bound_m
                + _FK_ERROR
                * max(
                    self.closure_model.intersector.characteristic_length_m,
                    hand_extent,
                    float(np.linalg.norm(target, ord=np.inf)),
                )
            )
            volume_parameters = np.empty(
                self.closure_model.parameter_dimension, dtype=np.float64
            )
            volume_parameters[0] = parameters[0]
            volume_parameters[1] = _recover_closed_unit_coordinate(
                float(target_coordinates[2]),
                float(lower[2]),
                float(upper[2]),
                absolute_error=comparison_error_m,
                label="anchored axial target",
            )
            volume_parameters[2] = _recover_closed_unit_coordinate(
                float(target_coordinates[0]),
                float(lower[0]),
                float(upper[0]),
                absolute_error=comparison_error_m,
                label="anchored first lateral target",
            )
            volume_parameters[3] = _recover_closed_unit_coordinate(
                float(target_coordinates[1]),
                float(lower[1]),
                float(upper[1]),
                absolute_error=comparison_error_m,
                label="anchored second lateral target",
            )
            volume_parameters[4:] = parameters[
                len(FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX):
            ]
        except RayClosureError as error:
            return SurfaceAnchorProposal(
                None,
                self._proposal_audit(
                    anchor_pad_name=anchor_pad_name,
                    parameters_unit=parameters,
                    object_face_index=object_face_index,
                    object_barycentric=barycentric,
                    object_position_m=object_position,
                    closure_phase=closure_phase,
                    witness_flat_index=witness_flat_index,
                    normalised_pad_source_winding_approach_margin=(
                        selected_pad_margin
                    ),
                    compatible_witness_count=len(compatible_indices),
                    witness_branch_index=branch_index,
                    failure_reason=f"ANCHOR_OUTSIDE_COMMON_SWEPT_DOMAIN:{error}",
                ),
            )

        v9_parameters_unit = tuple(float(value) for value in volume_parameters)
        return SurfaceAnchorProposal(
            v9_parameters_unit,
            self._proposal_audit(
                anchor_pad_name=anchor_pad_name,
                parameters_unit=parameters,
                object_face_index=object_face_index,
                object_barycentric=barycentric,
                object_position_m=object_position,
                closure_phase=closure_phase,
                witness_flat_index=witness_flat_index,
                normalised_pad_source_winding_approach_margin=(
                    selected_pad_margin
                ),
                compatible_witness_count=len(compatible_indices),
                witness_branch_index=branch_index,
                volume_parameters=volume_parameters,
            ),
        )

    def evaluate_unit_parameters(
        self,
        parameters_unit: Sequence[float],
        hand_model: ThreeFingerHandModel | None = None,
    ) -> SurfaceAnchoredClosureEvaluation:
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
        if parameters[1] >= 1.0:
            return self._parameter_failure(
                "anchor PAD selector uses a canonical half-open unit interval",
                parameters,
            )
        pad_row = min(
            int(float(parameters[1]) * len(self.closure_model.prepared_pads)),
            len(self.closure_model.prepared_pads) - 1,
        )
        anchor_pad_name = self.prepared_pad_names[pad_row]
        fixed_parameters = np.concatenate((parameters[:1], parameters[2:]))
        proposal = self.propose_fixed_anchor(
            fixed_parameters,
            anchor_pad_name,
            supplied_hand,
        )
        if proposal.v9_parameters_unit is None:
            return SurfaceAnchoredClosureEvaluation(
                None,
                self._evaluation_audit_from_proposal(
                    proposal.audit,
                    parameters,
                ),
            )

        anchor_audit = proposal.audit
        if (
            anchor_audit.anchor_object_face_index is None
            or anchor_audit.anchor_object_position_m is None
            or anchor_audit.anchor_closure_phase is None
            or anchor_audit.selected_pad_triangle_index is None
            or anchor_audit.selected_pad_witness_index is None
        ):
            return SurfaceAnchoredClosureEvaluation(
                None,
                self._evaluation_audit_from_proposal(
                    anchor_audit,
                    parameters,
                    failure_reason="FIXED_ANCHOR_PROPOSAL_AUDIT_INCOMPLETE",
                ),
            )

        closure = self.closure_model.evaluate_unit_parameters(
            proposal.v9_parameters_unit,
            supplied_hand,
        )
        failure_reason = None
        candidate = closure.candidate
        if candidate is None:
            failure_reason = (
                f"DELEGATED_CLOSURE_REJECTED:{closure.audit.failure_reason}"
            )
        else:
            pad_audit = closure.audit.pad_audits[pad_row]
            contact = next(
                (
                    row
                    for row in candidate.planned_pad_contacts
                    if row.pad_name == anchor_pad_name
                ),
                None,
            )
            closure_phase = anchor_audit.anchor_closure_phase
            object_position = np.asarray(
                anchor_audit.anchor_object_position_m,
                dtype=np.float64,
            )
            phase_error = (
                0.5 * float(pad_audit.selected_closure_interval_width or 0.0)
                + _gamma(256) * max(1.0, abs(closure_phase))
            )
            position_error = (
                self.closure_model.intersector.distance_error_bound_m
                + float(pad_audit.selected_spatial_error_bound_m or 0.0)
                + _FK_ERROR
                * max(
                    self.closure_model.intersector.characteristic_length_m,
                    float(np.linalg.norm(object_position, ord=np.inf)),
                )
            )
            anchor_matches = (
                pad_audit.selected_triangle_index
                == anchor_audit.selected_pad_triangle_index
                and pad_audit.selected_witness_index
                == anchor_audit.selected_pad_witness_index
                and pad_audit.selected_object_face_index
                == anchor_audit.anchor_object_face_index
                and pad_audit.selected_normalized_closure is not None
                and abs(
                    float(pad_audit.selected_normalized_closure)
                    - closure_phase
                )
                <= phase_error
                and contact is not None
                and np.linalg.norm(
                    np.asarray(contact.position_object_m) - object_position
                )
                <= position_error
            )
            if not anchor_matches:
                candidate = None
                failure_reason = "ANCHOR_NOT_ACTUAL_FIRST_CONTACT"
        return SurfaceAnchoredClosureEvaluation(
            candidate,
            self._evaluation_audit_from_proposal(
                anchor_audit,
                parameters,
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
        return self.evaluate_unit_parameters(parameters_unit, hand_model).candidate

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
                "fixed_anchor_method_id": FIXED_ANCHOR_METHOD_ID,
                "fixed_anchor_parameter_domain_id": (
                    FIXED_ANCHOR_PARAMETER_DOMAIN_ID
                ),
                "fixed_anchor_parameter_layout": (
                    self.fixed_anchor_parameter_layout
                ),
                "fixed_anchor_pad_selection": (
                    "EXACT_PREPARED_PAD_NAME_FROM_EXTERNAL_DISCRETE_LANE"
                ),
                "prepared_pad_names": self.prepared_pad_names,
                "surface_measure": SURFACE_MEASURE,
                "anchor_role": ANCHOR_ROLE,
                "witness_selection": WITNESS_SELECTION,
                "object_geometry_sha256": self.object_model.geometry_sha256,
                "allowed_face_count": len(self.allowed_face_indices),
                "allowed_surface_area_m2": self.allowed_surface_area_m2,
                "claim_limitations": CLAIM_LIMITATIONS,
                "delegated_closure_contract": self.closure_model.contract,
            }
        )


__all__ = [
    "ANCHOR_ROLE",
    "CLAIM_LIMITATIONS",
    "FIXED_ANCHOR_METHOD_ID",
    "FIXED_ANCHOR_PARAMETER_DOMAIN_ID",
    "FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX",
    "METHOD_ID",
    "PARAMETER_DOMAIN_ID",
    "PARAMETER_LAYOUT_PREFIX",
    "SURFACE_MEASURE",
    "SurfaceAnchorAudit",
    "SurfaceAnchorProposal",
    "SurfaceAnchoredClosureError",
    "SurfaceAnchoredClosureEvaluation",
    "SurfaceAnchoredRayClosureModel",
    "WITNESS_SELECTION",
]
