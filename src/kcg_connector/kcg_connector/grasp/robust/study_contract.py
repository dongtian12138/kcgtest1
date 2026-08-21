"""Fail-closed preregistration boundary for the CARTS-Grasp study.

The three YAML documents in this study have different responsibilities:

* the shared method document fixes every algorithmic choice;
* the hand document fixes kinematics, finite PAD geometry and actuation; and
* the object document contains only object evidence and study roles.

This module canonicalises those documents without YAML-order dependence,
verifies every file that the study consumes, and rejects attempts to move an
algorithm parameter into an object row.  A :class:`FrozenStudyContract` is
created only when *all* preregistration bindings are explicit.  The RayClosure
subdivision budget and closure parameter domain are bound explicitly to the
production implementation; unrelated, still-open study bindings continue to
fail closed and prevent a premature formal freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml
import numpy as np
import scipy

from .continuous_collision import INDEPENDENT_MOVING_PAIR_METHOD_ID
from .full_hand_collision import (
    CONTACT_RANGE_POLICY_CLAIM_LIMITATIONS,
    CONTACT_RANGE_POLICY_MANDATORY_BLOCKERS,
    CONTACT_RANGE_POLICY_METHOD_ID,
    ContactRangePolicyCollisionCertificate,
)
from .grasp_optimizer import deterministic_sobol
from .hand_contract import (
    CARTSHandContract,
    HandContractError,
    load_carts_hand_contract,
)
from .interval_kinematics import METHOD_ID as INTERVAL_KINEMATICS_METHOD_ID
from .post_generation_ranker import (
    COMPLETE_CLEARANCE_SCOPE,
    FORMAL_UNCERTAINTY_BLOCKER,
    METHOD_ID as POST_GENERATION_RANKER_METHOD_ID,
    PostGenerationRankOnlyPipeline,
    PostGenerationRankResult,
    SCENARIO_COUNT as POST_GENERATION_SCENARIO_COUNT,
    SCENARIO_DESIGN_SHA256 as POST_GENERATION_SCENARIO_DESIGN_SHA256,
    SCENARIO_DIMENSION as POST_GENERATION_SCENARIO_DIMENSION,
    SCENARIO_METHOD_ID as POST_GENERATION_SCENARIO_METHOD_ID,
    SCENARIO_SCIPY_VERSION as POST_GENERATION_SCIPY_VERSION,
    SCENARIO_SOBOL_SEED as POST_GENERATION_SCENARIO_SOBOL_SEED,
    SELECTION_ORDER as POST_GENERATION_SELECTION_ORDER,
    TIE_BREAK_RULE as POST_GENERATION_TIE_BREAK_RULE,
    POLICY_AWARE_RANKING_GUARD,
)
from .ray_closure import (
    CLOSURE_PARAMETER_DOMAIN_ID,
    METHOD_ID as RAY_CLOSURE_METHOD_ID,
    PARAMETER_LAYOUT_PREFIX,
    RayClosureSurfaceModel,
)
from .surface_anchored_closure import (
    FIXED_ANCHOR_METHOD_ID,
    FIXED_ANCHOR_PARAMETER_DOMAIN_ID,
    FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX,
    SurfaceAnchoredRayClosureModel,
)
from .task_wrench_evaluator import (
    FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE,
)
from .top_level_candidate_generator import (
    ALLOWED_TOTAL_ATTEMPT_BUDGETS,
    CONTACT_RANGE_POLICY_DOWNSTREAM_STATUS,
    CONTACT_RANGE_POLICY_OUTPUT_CHANNEL,
    DEDUPLICATION_RULE,
    EXACT_CANDIDATE_OUTPUT_CHANNEL,
    FROZEN_FIXED_ANCHOR_PARAMETER_DIMENSION,
    FROZEN_V9_PARAMETER_DIMENSION,
    LANE_SPECS,
    LOCAL_REFINEMENT_EVALUATION_BUDGET,
    MAIN_TOTAL_ATTEMPT_BUDGET,
    MAXIMUM_POINTS_PER_LANE,
    METHOD_ID as TOP_LEVEL_CANDIDATE_GENERATOR_METHOD_ID,
    SCHEDULE_RULE,
    TOP_LEVEL_OUTPUT_CLAIM,
    TopLevelGenerationResult,
)


_LEAF = object()
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_METHOD = "CARTS-Grasp"
_DEVELOPMENT_OBJECT = "current_d38999_26kj61sn_public_spec"
_TRANSFER_OBJECT = "te_deutsch_d38999_26fj35pn_step"
_TRANSFER_ROLE = "ZERO_OBJECT_TUNING_HELD_OUT_CASE_STUDY"
_NUMERICAL_ROLE = "NUMERICAL_SOLVER_PROTOCOL_NOT_PHYSICAL_ACCEPTANCE_GATE"
_FROZEN_SCHEMA = "carts_frozen_study_contract_v1"
_SCIPY_VERSION = "1.8.0"
_SETUP_MANIFEST_SHA256 = (
    "bde1b207d15a35018de4ea91c05bf26d8b143366dcb3eadec4346f2456913765"
)
_PACKAGE_XML_SHA256 = (
    "3e37fbe688ad739472109af4da6f39b4a03e09eef07eb456ddddd64acd95340b"
)

_MISSING_COMPLETE_COLLISION_BINDING = (
    "MISSING_COMPLETE_HAND_ENVIRONMENT_CONTINUOUS_COLLISION_BINDING"
)
_MISSING_FORMAL_ROOT_INTERVAL_CANDIDATE_PROPAGATION = (
    "MISSING_FORMAL_ROOT_INTERVAL_CANDIDATE_PROPAGATION"
)
_FORMAL_EMPTY_STATUS = (
    "EMPTY_UNTIL_FORMAL_ROOT_INTERVAL_CANDIDATE_PROPAGATION_"
    "COMPLETE_COLLISION_AND_CALIBRATED_FULL_UNCERTAINTY"
)
_NESTED_CONVERGENCE_ROLE = (
    "SAME_SEED_SCRAMBLED_SOBOL_PREFIX_EXTENSION_NOT_INDEPENDENT_VALIDATION"
)
_CURRENT_PREREGISTRATION_BLOCKERS = (
    _MISSING_FORMAL_ROOT_INTERVAL_CANDIDATE_PROPAGATION,
    _MISSING_COMPLETE_COLLISION_BINDING,
    FORMAL_UNCERTAINTY_BLOCKER,
)


class StudyContractError(ValueError):
    """Raised when a study document cannot cross the freeze boundary."""


class StudyFreezeIncompleteError(StudyContractError):
    """Raised after a valid audit still has explicit preregistration gaps."""

    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(blockers)
        super().__init__(
            "study is not freeze-eligible; preregistration blockers="
            + ",".join(self.blockers)
        )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every nesting level."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise StudyContractError(
                "YAML mapping keys must be scalar and hashable"
            ) from exc
        if duplicate:
            raise StudyContractError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise StudyContractError(f"{label} must be a string-keyed mapping")
    return value


def _load_unique_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except StudyContractError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise StudyContractError(f"cannot read {label}: {path}") from exc
    return _mapping(value, f"{label} root")


def _validate_json_leaf(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StudyContractError(f"{label} must not contain NaN or infinity")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                raise StudyContractError(
                    f"{label}[{index}] contains an unregistered mapping schema"
                )
            _validate_json_leaf(item, f"{label}[{index}]")
        return
    raise StudyContractError(f"{label} is not a canonical JSON value")


def _validate_schema(value: Any, schema: Any, label: str) -> None:
    if schema is _LEAF:
        _validate_json_leaf(value, label)
        return
    document = _mapping(value, label)
    expected = set(schema)
    actual = set(document)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise StudyContractError(
            f"{label} schema mismatch; missing={missing}, extra={extra}"
        )
    for key, child_schema in schema.items():
        _validate_schema(document[key], child_schema, f"{label}.{key}")


_METHOD_SCHEMA = {
    "schema_version": _LEAF,
    "method": _LEAF,
    "claim_scope": _LEAF,
    "shared_protocol": {
        "object_specific_algorithm_tuning_allowed": _LEAF,
        "legacy_candidate_import_allowed": _LEAF,
        "legacy_h_chain_import_allowed": _LEAF,
        "online_object_ground_truth_allowed": _LEAF,
        "online_contact_truth_allowed": _LEAF,
        "hardware_authorized": _LEAF,
    },
    "surface_model": {
        "assembly_axis_source": _LEAF,
        "deterministic_surface_sampling": _LEAF,
        "pad_footprint_source": _LEAF,
        "functional_surface_mask_source": _LEAF,
        "mesh_convergence_levels": _LEAF,
        "contact_orientation_feasible_set": _LEAF,
        "object_specific_normal_component_thresholds_allowed": _LEAF,
        "analytic_primitive_tessellation": {
            "maximum_relative_sagitta_error": _LEAF,
            "convergence_resolution_multipliers": _LEAF,
            "edge_count_derivation": _LEAF,
        },
    },
    "candidate_optimization": {
        "method_id": _LEAF,
        "v9_certifier": {
            "implementation_type_id": _LEAF,
            "method_id": _LEAF,
            "parameter_domain_id": _LEAF,
            "parameter_layout": _LEAF,
        },
        "fixed_anchor_mapper": {
            "implementation_type_id": _LEAF,
            "method_id": _LEAF,
            "parameter_domain_id": _LEAF,
            "parameter_layout": _LEAF,
        },
        "hand_binding": {
            "dimension_and_layout_source": _LEAF,
            "runtime_object_identity_required": _LEAF,
            "preshape_joint_names": _LEAF,
            "prepared_pad_order": _LEAF,
        },
        "lane_order": _LEAF,
        "lanes": {
            "DIRECT_V9": {
                "dimension": _LEAF,
                "sobol_seed": _LEAF,
                "anchor_pad_ordinal": _LEAF,
                "anchor_pad_name": _LEAF,
                "maximum_prefix_design_sha256": _LEAF,
            },
            "SURFACE_PAD_A": {
                "dimension": _LEAF,
                "sobol_seed": _LEAF,
                "anchor_pad_ordinal": _LEAF,
                "anchor_pad_name": _LEAF,
                "maximum_prefix_design_sha256": _LEAF,
            },
            "SURFACE_PAD_B": {
                "dimension": _LEAF,
                "sobol_seed": _LEAF,
                "anchor_pad_ordinal": _LEAF,
                "anchor_pad_name": _LEAF,
                "maximum_prefix_design_sha256": _LEAF,
            },
            "SURFACE_PAD_C": {
                "dimension": _LEAF,
                "sobol_seed": _LEAF,
                "anchor_pad_ordinal": _LEAF,
                "anchor_pad_name": _LEAF,
                "maximum_prefix_design_sha256": _LEAF,
            },
        },
        "schedule_rule": _LEAF,
        "allowed_total_attempt_budgets": _LEAF,
        "main_total_attempt_budget": _LEAF,
        "maximum_points_per_lane": _LEAF,
        "proposal_failure_consumes_attempt": _LEAF,
        "duplicate_consumes_attempt": _LEAF,
        "replacement_sampling_allowed": _LEAF,
        "maximum_v9_evaluations_equals_attempt_budget": _LEAF,
        "deduplication_rule": _LEAF,
        "sobol_design": {
            "generator": _LEAF,
            "scipy_version": _LEAF,
            "dependency": {
                "package": _LEAF,
                "version": _LEAF,
                "python_manifest_path": _LEAF,
                "python_manifest_sha256": _LEAF,
                "ros_package": _LEAF,
                "ros_manifest_path": _LEAF,
                "ros_manifest_sha256": _LEAF,
            },
            "scramble": _LEAF,
            "optimization": _LEAF,
            "generate_common_maximum_then_take_lane_prefix": _LEAF,
            "maximum_points_per_lane": _LEAF,
            "realized_design_hashes_are_contract_bound": _LEAF,
        },
        "local_refinement": {
            "method_id": _LEAF,
            "execution_status": _LEAF,
            "evaluation_budget": _LEAF,
            "ranking_eligible": _LEAF,
        },
        "external_lane_registry_supported": _LEAF,
        "legacy_grasp_optimizer_formal_eligible": _LEAF,
        "accepted_output_channels": _LEAF,
        "candidate_and_policy_mutually_exclusive": _LEAF,
        "display_only_proposal_formal_eligible": _LEAF,
        "contact_range_policy_downstream_status": _LEAF,
        "output_claim": _LEAF,
    },
    "contact_range_policy_collision": {
        "implementation_type_id": _LEAF,
        "method_id": _LEAF,
        "independent_two_phase_method_id": _LEAF,
        "phase_domain_rule": _LEAF,
        "link_dependency_rule": _LEAF,
        "same_support_rule": _LEAF,
        "cross_support_rule": _LEAF,
        "display_approximation_used_as_formal_evidence": _LEAF,
        "endpoint_or_finite_corner_sampling_allowed": _LEAF,
        "srdf_exemptions_applied": _LEAF,
        "exact_candidate_collision_path_changed": _LEAF,
        "current_state": _LEAF,
        "checkable_scope": _LEAF,
        "mandatory_blockers": _LEAF,
        "claim_limitations": _LEAF,
        "formal_selection_allowed": _LEAF,
        "isaac_launch_allowed": _LEAF,
    },
    "post_generation_ranking": {
        "implementation_type_id": _LEAF,
        "method_id": _LEAF,
        "input_type_id": _LEAF,
        "output_type_id": _LEAF,
        "execution_role": _LEAF,
        "selection": _LEAF,
        "selection_order": _LEAF,
        "tie_break_rule": _LEAF,
        "common_scenarios": {
            "method_id": _LEAF,
            "source": _LEAF,
            "identical_realizations_for_every_candidate": _LEAF,
            "candidate_specific_resampling_allowed": _LEAF,
            "scenario_design": _LEAF,
            "scipy_version": _LEAF,
            "dimension": _LEAF,
            "scenario_count": _LEAF,
            "sobol_seed": _LEAF,
            "scramble": _LEAF,
            "optimization": _LEAF,
            "identity_encoding": _LEAF,
            "design_sha256": _LEAF,
        },
        "failure_retention": {
            "retain_every_generation_attempt": _LEAF,
            "retain_every_unique_accepted_candidate": _LEAF,
            "retain_every_unique_accepted_policy": _LEAF,
            "retain_every_generation_and_evaluation_failure": _LEAF,
            "failure_reason_or_exception_required": _LEAF,
            "failed_candidate_drop_allowed": _LEAF,
            "collision_invocations_per_unique_accepted_candidate": _LEAF,
            "wrench_invocations_per_unique_accepted_candidate": _LEAF,
            "retry_allowed": _LEAF,
            "replacement_after_failure_allowed": _LEAF,
        },
        "formal_selection": {
            "status": _LEAF,
            "allowed_with_current_bindings": _LEAF,
            "formal_ranked_keys_must_be_empty": _LEAF,
            "selected_candidate_must_be_none": _LEAF,
            "contact_range_policy_handling": _LEAF,
            "contact_range_policy_collision_invocations_before_support": _LEAF,
            "contact_range_policy_wrench_invocations_before_support": _LEAF,
            "required_collision_claim_scope": _LEAF,
            "current_uncertainty_claim_scope": _LEAF,
            "required_additional_uncertainty_binding": _LEAF,
        },
    },
    "ray_closure": {
        "method_id": _LEAF,
        "maximum_subdivision_intervals": _LEAF,
        "maximum_subdivision_intervals_role": _LEAF,
        "subdivision_budget_exhaustion_policy": _LEAF,
        "budget_convergence_values": _LEAF,
        "budget_convergence_role": _LEAF,
        "adjacent_budget_result_stability_report_required": _LEAF,
        "physical_acceptance_gate": _LEAF,
        "interval_backend": {
            "method_id": _LEAF,
            "dependency": {
                "package": _LEAF,
                "version": _LEAF,
                "manifest_path": _LEAF,
                "manifest_sha256": _LEAF,
            },
            "decimal_precision": _LEAF,
            "decimal_precision_role": _LEAF,
            "precision_convergence_values": _LEAF,
            "precision_convergence_role": _LEAF,
            "maximum_root_bisection_iterations": _LEAF,
            "maximum_root_bisection_iterations_role": _LEAF,
            "root_budget_convergence_values": _LEAF,
            "root_budget_exhaustion_policy": _LEAF,
            "predicate_policy": _LEAF,
            "adjacent_precision_result_stability_report_required": _LEAF,
            "physical_acceptance_gate": _LEAF,
        },
        "closure_parameter_domain": {
            "domain_id": _LEAF,
            "coordinate_space": _LEAF,
            "dimension": _LEAF,
            "parameter_layout": _LEAF,
            "dimension_and_layout_source": _LEAF,
            "assembly_axis_yaw": {
                "coordinate_name": _LEAF,
                "unit_interval": _LEAF,
                "physical_mapping": _LEAF,
                "source": _LEAF,
            },
            "placement": {
                "coordinate_names": _LEAF,
                "unit_interval": _LEAF,
                "physical_bounds_source": _LEAF,
                "zero_width_axis_convention": _LEAF,
            },
            "preshape": {
                "coordinate_prefix": _LEAF,
                "joint_names": _LEAF,
                "unit_interval": _LEAF,
                "joint_selection_source": _LEAF,
                "physical_bounds_source": _LEAF,
            },
        },
    },
    "linear_program": {
        "solver": _LEAF,
        "constraint_scaling": _LEAF,
        "maximum_iterations": _LEAF,
        "primal_feasibility_tolerance": _LEAF,
        "dual_feasibility_tolerance": _LEAF,
        "ipm_optimality_tolerance": _LEAF,
        "tolerance_convergence_multipliers": _LEAF,
        "physical_acceptance_gate": _LEAF,
    },
    "contact_model": {
        "type": _LEAF,
        "friction_cone_approximation": _LEAF,
        "maximum_inner_approximation_relative_error": _LEAF,
        "cone_edge_convergence_multiplier": _LEAF,
        "normal_force_capacity_source": _LEAF,
        "joint_torque_limits_source": _LEAF,
    },
    "uncertainty": {
        "scenario_design": _LEAF,
        "scenario_role": _LEAF,
        "probability_distribution_claimed": _LEAF,
        "scenario_count": _LEAF,
        "nested_convergence_scenario_count": _LEAF,
        "nested_convergence_role": _LEAF,
        "sobol_seed": _LEAF,
        "lower_tail_fraction": _LEAF,
        "report_hard_bound_minimum": _LEAF,
        "per_object_interval_source": _LEAF,
        "post_result_interval_changes_allowed": _LEAF,
    },
    "task_wrench": {
        "representation": _LEAF,
        "gravity_acceleration_m_s2": _LEAF,
        "gravity_acceleration_source": _LEAF,
        "task_frame_source": _LEAF,
        "required_wrench": {
            "gravity": _LEAF,
            "lift_acceleration_m_s2": _LEAF,
        },
        "disturbance_body": _LEAF,
        "force_normalization": _LEAF,
        "moment_normalization": _LEAF,
        "characteristic_radius_source": _LEAF,
        "binary_quality_pass_threshold_allowed": _LEAF,
    },
    "controller": {
        "architecture": _LEAF,
        "force_target_source": _LEAF,
        "fixed_force_increment_allowed": _LEAF,
        "object_pose_feedback_source": _LEAF,
        "contact_feedback_source": _LEAF,
        "qp_regularization_source": _LEAF,
    },
    "passive_impedance_numerics": {
        "matrix_symmetry_absolute_tolerance": _LEAF,
        "semidefinite_eigenvalue_tolerance": _LEAF,
        "rotation_orthogonality_tolerance": _LEAF,
        "homogeneous_row_tolerance": _LEAF,
        "passivity_balance_tolerance": _LEAF,
    },
    "force_qp": {
        "solver": _LEAF,
        "constraint_scaling": _LEAF,
        "maximum_iterations": _LEAF,
        "objective_tolerance": _LEAF,
        "equality_tolerance": _LEAF,
        "inequality_tolerance": _LEAF,
        "linear_independence_tolerance": _LEAF,
        "feasibility_dual_tolerance": _LEAF,
        "regularization": _LEAF,
        "physical_acceptance_gate": _LEAF,
    },
    "numerical_convergence": {
        "tolerance_multipliers": _LEAF,
        "result_stability_required": _LEAF,
        "physical_acceptance_gate": _LEAF,
    },
    "dynamic_protocol": {
        "lift_targets_m": _LEAF,
        "hold_duration_s": _LEAF,
        "disturbance_direction_design": _LEAF,
        "disturbance_direction_count": _LEAF,
        "dimensionless_amplitudes": _LEAF,
        "pulse_duration_s": _LEAF,
        "paired_random_seeds": _LEAF,
        "independent_isaac_process_per_trial": _LEAF,
    },
    "reporting": {
        "confidence_level": _LEAF,
        "report_all_failures": _LEAF,
        "success_unit": _LEAF,
        "physics_step_is_independent_sample": _LEAF,
        "generated_images_as_evidence_allowed": _LEAF,
        "figure_data_sources": _LEAF,
    },
}


_FRAMES_CURRENT_SCHEMA = {
    "length_unit": _LEAF,
    "assembly_axis_object": _LEAF,
    "task_frame_rotation_object": _LEAF,
    "task_frame_source": _LEAF,
    "nominal_validation_gravity_direction_object": _LEAF,
}
_FRAMES_TRANSFER_SCHEMA = {
    **_FRAMES_CURRENT_SCHEMA,
    "source_step_length_unit": _LEAF,
}
_UNCERTAINTY_SCHEMA = {
    "simulation_mass_com_inertia_randomized": _LEAF,
    "perception_pose_residual_source": _LEAF,
    "surface_normal_residual_source": _LEAF,
    "joint_and_torque_residual_source": _LEAF,
    "placeholder_numeric_bounds_allowed": _LEAF,
}
_DYNAMIC_ELIGIBILITY_SCHEMA = {"allowed": _LEAF, "reason": _LEAF}


_OBJECT_SCHEMA = {
    "schema_version": _LEAF,
    "study_id": _LEAF,
    "claim_scope": _LEAF,
    "shared_method_config": _LEAF,
    "shared_method_config_required_for_every_object": _LEAF,
    "object_specific_algorithm_hyperparameters_allowed": _LEAF,
    "hardware_authorized": _LEAF,
    "transfer_protocol": {
        "development_object": _LEAF,
        "frozen_transfer_object": _LEAF,
        "transfer_object_geometry_seen_before_method_freeze": _LEAF,
        "prospective_double_blind_claim_allowed": _LEAF,
        "candidate_ids_or_contact_coordinates_shared_between_objects": _LEAF,
    },
    "objects": {
        _DEVELOPMENT_OBJECT: {
            "identity": {
                "connector": _LEAF,
                "shell_size": _LEAF,
                "contact_count": _LEAF,
                "source_class": _LEAF,
            },
            "frames": _FRAMES_CURRENT_SCHEMA,
            "planning_geometry": {
                "format": _LEAF,
                "path": _LEAF,
                "sha256": _LEAF,
                "manifest": _LEAF,
                "manifest_sha256": _LEAF,
                "source_stage": _LEAF,
                "source_stage_sha256": _LEAF,
                "source_subtree": _LEAF,
                "allowed_surface_semantic": _LEAF,
                "semantic_authority": _LEAF,
                "simulator_truth_used": _LEAF,
            },
            "physical_properties": {
                "source_class": _LEAF,
                "source_contract": _LEAF,
                "source_contract_sha256": _LEAF,
                "component_composition": {
                    "body_assembly": {
                        "mass_kg": _LEAF,
                        "center_of_mass_m": _LEAF,
                        "diagonal_inertia_kg_m2": _LEAF,
                    },
                    "coupling_nut": {
                        "mass_kg": _LEAF,
                        "center_of_mass_m": _LEAF,
                        "diagonal_inertia_kg_m2": _LEAF,
                    },
                },
                "planning_rigid_composition": {
                    "method": _LEAF,
                    "mass_kg": _LEAF,
                    "center_of_mass_m": _LEAF,
                    "inertia_kg_m2": _LEAF,
                    "vendor_hardware_truth_claimed": _LEAF,
                },
            },
            "contact_material_uncertainty": {
                "model": _LEAF,
                "friction_coefficient": _LEAF,
                "source_class": _LEAF,
                "source": _LEAF,
                "source_sha256": _LEAF,
                "vendor_friction_claimed": _LEAF,
                "probability_distribution_claimed": _LEAF,
            },
            "uncertainty_calibration": _UNCERTAINTY_SCHEMA,
            "dynamic_eligibility": _DYNAMIC_ELIGIBILITY_SCHEMA,
        },
        _TRANSFER_OBJECT: {
            "identity": {
                "connector": _LEAF,
                "manufacturer": _LEAF,
                "te_catalog_number": _LEAF,
                "shell_size": _LEAF,
                "insert_arrangement": _LEAF,
                "contact_count": _LEAF,
                "source_class": _LEAF,
            },
            "frames": _FRAMES_TRANSFER_SCHEMA,
            "original_cad": {
                "path": _LEAF,
                "sha256": _LEAF,
                "solid_count": _LEAF,
                "face_count": _LEAF,
                "geometry_audit": _LEAF,
                "geometry_audit_sha256": _LEAF,
            },
            "planning_geometry": {
                "format": _LEAF,
                "path": _LEAF,
                "sha256": _LEAF,
                "source_unit": _LEAF,
                "watertight": _LEAF,
                "winding_consistent": _LEAF,
                "allowed_surface_semantic": _LEAF,
                "semantic_derivation": _LEAF,
                "part_number_specific_z_or_radius_cutoffs_allowed": _LEAF,
                "simulator_truth_used": _LEAF,
            },
            "physical_properties": {
                "mass_source_class": _LEAF,
                "mass_source": _LEAF,
                "mass_source_sha256": _LEAF,
                "mass_kg": _LEAF,
                "com_and_inertia_source_class": _LEAF,
                "uniform_density_kg_m3": _LEAF,
                "center_of_mass_m": _LEAF,
                "inertia_kg_m2": _LEAF,
                "vendor_hardware_truth_claimed": _LEAF,
            },
            "contact_material_uncertainty": {
                "model": _LEAF,
                "friction_coefficient": _LEAF,
                "source_class": _LEAF,
                "source": _LEAF,
                "source_sha256": _LEAF,
                "vendor_friction_claimed": _LEAF,
                "probability_distribution_claimed": _LEAF,
            },
            "uncertainty_calibration": _UNCERTAINTY_SCHEMA,
            "dynamic_model": {
                "existing_te_j35_physx_v1_eligible": _LEAF,
                "required_representation": _LEAF,
                "current_status": _LEAF,
            },
            "dynamic_eligibility": _DYNAMIC_ELIGIBILITY_SCHEMA,
        },
    },
}


def _exact(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise StudyContractError(f"{label} must be exactly {expected!r}")


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudyContractError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        qualifier = "positive " if positive else ""
        raise StudyContractError(f"{label} must be a finite {qualifier}number")
    return parsed


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StudyContractError(f"{label} must be a positive integer")
    return value


def _number_sequence(
    value: Any,
    label: str,
    *,
    nonempty: bool = True,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StudyContractError(f"{label} must be a numeric sequence")
    if nonempty and not value:
        raise StudyContractError(f"{label} cannot be empty")
    parsed = tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    if positive and any(item <= 0.0 for item in parsed):
        raise StudyContractError(f"{label} values must be positive")
    if nonnegative and any(item < 0.0 for item in parsed):
        raise StudyContractError(f"{label} values must be nonnegative")
    return parsed


def _implementation_type_id(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _top_level_design_sha256() -> dict[str, str]:
    result: dict[str, str] = {}
    for spec in LANE_SPECS:
        design = deterministic_sobol(
            dimension=spec.dimension,
            count=MAXIMUM_POINTS_PER_LANE,
            seed=spec.sobol_seed,
        )
        encoded = np.asarray(design, dtype=">f8").tobytes(order="C")
        result[spec.lane.value] = hashlib.sha256(encoded).hexdigest()
    return result


def _post_generation_design_sha256() -> str:
    design = deterministic_sobol(
        dimension=POST_GENERATION_SCENARIO_DIMENSION,
        count=POST_GENERATION_SCENARIO_COUNT,
        seed=POST_GENERATION_SCENARIO_SOBOL_SEED,
    )
    encoded = np.asarray(design, dtype=">f8").tobytes(order="C")
    return hashlib.sha256(encoded).hexdigest()


def _validate_nested_post_generation_design(
    nested_count: int,
) -> None:
    base = deterministic_sobol(
        dimension=POST_GENERATION_SCENARIO_DIMENSION,
        count=POST_GENERATION_SCENARIO_COUNT,
        seed=POST_GENERATION_SCENARIO_SOBOL_SEED,
    )
    nested = deterministic_sobol(
        dimension=POST_GENERATION_SCENARIO_DIMENSION,
        count=nested_count,
        seed=POST_GENERATION_SCENARIO_SOBOL_SEED,
    )
    if nested_count <= POST_GENERATION_SCENARIO_COUNT or not np.array_equal(
        np.asarray(base, dtype=np.float64),
        np.asarray(nested[:POST_GENERATION_SCENARIO_COUNT], dtype=np.float64),
    ):
        raise StudyContractError(
            "method.uncertainty nested convergence is not an exact same-seed "
            "Sobol prefix extension"
        )


def _validate_method(document: Mapping[str, Any]) -> None:
    _validate_schema(document, _METHOD_SCHEMA, "method contract")
    _exact(document["schema_version"], "carts_grasp_v1", "method.schema_version")
    _exact(document["method"], _METHOD, "method.method")
    _exact(
        document["claim_scope"],
        "SIMULATION_ONLY_ZERO_OBJECT_TUNING_CROSS_MODEL_STUDY",
        "method.claim_scope",
    )

    shared = _mapping(document["shared_protocol"], "method.shared_protocol")
    for field in shared:
        _exact(shared[field], False, f"method.shared_protocol.{field}")

    surface = _mapping(document["surface_model"], "method.surface_model")
    for field, expected in {
        "assembly_axis_source": "OBJECT_CONTRACT",
        "deterministic_surface_sampling": "AREA_STRATIFIED_SOBOL",
        "pad_footprint_source": "HAND_CONTACT_MODEL",
        "functional_surface_mask_source": "OBJECT_CONTRACT_PLUS_GENERAL_TOPOLOGY",
        "contact_orientation_feasible_set": (
            "DERIVED_FROM_PAD_NORMAL_CONE_JOINT_LIMITS_AND_ASSEMBLY_AXIS"
        ),
    }.items():
        _exact(surface[field], expected, f"method.surface_model.{field}")
    _exact(
        surface["object_specific_normal_component_thresholds_allowed"],
        False,
        "method.surface_model.object_specific_normal_component_thresholds_allowed",
    )
    _number_sequence(
        surface["mesh_convergence_levels"],
        "method.surface_model.mesh_convergence_levels",
        positive=True,
    )
    tessellation = _mapping(
        surface["analytic_primitive_tessellation"],
        "method.surface_model.analytic_primitive_tessellation",
    )
    relative_error = _number(
        tessellation["maximum_relative_sagitta_error"],
        "method.surface_model.analytic_primitive_tessellation.maximum_relative_sagitta_error",
        positive=True,
    )
    if relative_error >= 1.0:
        raise StudyContractError("analytic primitive relative error must be below one")
    _number_sequence(
        tessellation["convergence_resolution_multipliers"],
        "method.surface_model.analytic_primitive_tessellation.convergence_resolution_multipliers",
        positive=True,
    )
    _exact(
        tessellation["edge_count_derivation"],
        "1_MINUS_COS_PI_OVER_E_LE_MAXIMUM_RELATIVE_SAGITTA_ERROR",
        "method.surface_model.analytic_primitive_tessellation.edge_count_derivation",
    )

    candidate = _mapping(
        document["candidate_optimization"],
        "method.candidate_optimization",
    )
    _exact(
        candidate["method_id"],
        TOP_LEVEL_CANDIDATE_GENERATOR_METHOD_ID,
        "method.candidate_optimization.method_id",
    )
    v9 = _mapping(
        candidate["v9_certifier"],
        "method.candidate_optimization.v9_certifier",
    )
    _exact(
        v9["implementation_type_id"],
        _implementation_type_id(RayClosureSurfaceModel),
        "method.candidate_optimization.v9_certifier.implementation_type_id",
    )
    _exact(
        v9["method_id"],
        RAY_CLOSURE_METHOD_ID,
        "method.candidate_optimization.v9_certifier.method_id",
    )
    _exact(
        v9["parameter_domain_id"],
        CLOSURE_PARAMETER_DOMAIN_ID,
        "method.candidate_optimization.v9_certifier.parameter_domain_id",
    )
    hand_binding = _mapping(
        candidate["hand_binding"],
        "method.candidate_optimization.hand_binding",
    )
    _exact(
        hand_binding["dimension_and_layout_source"],
        "HASH_BOUND_HAND_URDF_PLUS_CLOSURE_ACTUATION_AND_PAD_ORDER",
        "method.candidate_optimization.hand_binding."
        "dimension_and_layout_source",
    )
    _exact(
        hand_binding["runtime_object_identity_required"],
        True,
        "method.candidate_optimization.hand_binding."
        "runtime_object_identity_required",
    )
    preshape_names = hand_binding["preshape_joint_names"]
    if not isinstance(preshape_names, Sequence):
        raise StudyContractError(
            "method.candidate_optimization.hand_binding.preshape_joint_names "
            "must be non-empty strings"
        )
    if isinstance(preshape_names, (str, bytes, bytearray)):
        raise StudyContractError(
            "method.candidate_optimization.hand_binding.preshape_joint_names "
            "must be non-empty strings"
        )
    if any(
        not isinstance(name, str) or not name
        for name in preshape_names
    ):
        raise StudyContractError(
            "method.candidate_optimization.hand_binding.preshape_joint_names "
            "must be non-empty strings"
        )
    prepared_pad_order = hand_binding["prepared_pad_order"]
    if not isinstance(prepared_pad_order, Sequence):
        raise StudyContractError(
            "method.candidate_optimization.hand_binding.prepared_pad_order "
            "must contain three distinct non-empty PAD names"
        )
    if isinstance(prepared_pad_order, (str, bytes, bytearray)):
        raise StudyContractError(
            "method.candidate_optimization.hand_binding.prepared_pad_order "
            "must contain three distinct non-empty PAD names"
        )
    if len(prepared_pad_order) != 3 or len(set(prepared_pad_order)) != 3:
        raise StudyContractError(
            "method.candidate_optimization.hand_binding.prepared_pad_order "
            "must contain three distinct non-empty PAD names"
        )
    if any(
        not isinstance(name, str) or not name
        for name in prepared_pad_order
    ):
        raise StudyContractError(
            "method.candidate_optimization.hand_binding.prepared_pad_order "
            "must contain three distinct non-empty PAD names"
        )
    expected_v9_layout = PARAMETER_LAYOUT_PREFIX + tuple(
        f"preshape_joint_unit:{name}" for name in preshape_names
    )
    _exact(
        v9["parameter_layout"],
        list(expected_v9_layout),
        "method.candidate_optimization.v9_certifier.parameter_layout",
    )
    _exact(
        len(expected_v9_layout),
        FROZEN_V9_PARAMETER_DIMENSION,
        "method.candidate_optimization.v9_certifier dimension",
    )
    fixed_anchor = _mapping(
        candidate["fixed_anchor_mapper"],
        "method.candidate_optimization.fixed_anchor_mapper",
    )
    _exact(
        fixed_anchor["implementation_type_id"],
        _implementation_type_id(SurfaceAnchoredRayClosureModel),
        "method.candidate_optimization.fixed_anchor_mapper."
        "implementation_type_id",
    )
    _exact(
        fixed_anchor["method_id"],
        FIXED_ANCHOR_METHOD_ID,
        "method.candidate_optimization.fixed_anchor_mapper.method_id",
    )
    _exact(
        fixed_anchor["parameter_domain_id"],
        FIXED_ANCHOR_PARAMETER_DOMAIN_ID,
        "method.candidate_optimization.fixed_anchor_mapper."
        "parameter_domain_id",
    )
    expected_fixed_layout = FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX + tuple(
        f"preshape_joint_unit:{name}" for name in preshape_names
    )
    _exact(
        fixed_anchor["parameter_layout"],
        list(expected_fixed_layout),
        "method.candidate_optimization.fixed_anchor_mapper.parameter_layout",
    )
    _exact(
        len(expected_fixed_layout),
        FROZEN_FIXED_ANCHOR_PARAMETER_DIMENSION,
        "method.candidate_optimization.fixed_anchor_mapper dimension",
    )
    expected_lane_order = [spec.lane.value for spec in LANE_SPECS]
    _exact(
        candidate["lane_order"],
        expected_lane_order,
        "method.candidate_optimization.lane_order",
    )
    lanes = _mapping(
        candidate["lanes"],
        "method.candidate_optimization.lanes",
    )
    realized_hashes = _top_level_design_sha256()
    for spec in LANE_SPECS:
        lane_name = spec.lane.value
        lane = _mapping(
            lanes[lane_name],
            f"method.candidate_optimization.lanes.{lane_name}",
        )
        anchor_name = (
            None
            if spec.anchor_pad_ordinal is None
            else prepared_pad_order[spec.anchor_pad_ordinal]
        )
        for field, expected in {
            "dimension": spec.dimension,
            "sobol_seed": spec.sobol_seed,
            "anchor_pad_ordinal": spec.anchor_pad_ordinal,
            "anchor_pad_name": anchor_name,
            "maximum_prefix_design_sha256": realized_hashes[lane_name],
        }.items():
            _exact(
                lane[field],
                expected,
                f"method.candidate_optimization.lanes.{lane_name}.{field}",
            )
    for field, expected in {
        "schedule_rule": SCHEDULE_RULE,
        "allowed_total_attempt_budgets": list(
            ALLOWED_TOTAL_ATTEMPT_BUDGETS
        ),
        "main_total_attempt_budget": MAIN_TOTAL_ATTEMPT_BUDGET,
        "maximum_points_per_lane": MAXIMUM_POINTS_PER_LANE,
        "proposal_failure_consumes_attempt": True,
        "duplicate_consumes_attempt": True,
        "replacement_sampling_allowed": False,
        "maximum_v9_evaluations_equals_attempt_budget": True,
        "deduplication_rule": DEDUPLICATION_RULE,
        "external_lane_registry_supported": False,
        "legacy_grasp_optimizer_formal_eligible": False,
        "accepted_output_channels": [
            EXACT_CANDIDATE_OUTPUT_CHANNEL,
            CONTACT_RANGE_POLICY_OUTPUT_CHANNEL,
        ],
        "candidate_and_policy_mutually_exclusive": True,
        "display_only_proposal_formal_eligible": False,
        "contact_range_policy_downstream_status": (
            CONTACT_RANGE_POLICY_DOWNSTREAM_STATUS
        ),
        "output_claim": TOP_LEVEL_OUTPUT_CLAIM,
    }.items():
        _exact(
            candidate[field],
            expected,
            f"method.candidate_optimization.{field}",
        )

    policy_collision = _mapping(
        document["contact_range_policy_collision"],
        "method.contact_range_policy_collision",
    )
    for field, expected in {
        "implementation_type_id": _implementation_type_id(
            ContactRangePolicyCollisionCertificate
        ),
        "method_id": CONTACT_RANGE_POLICY_METHOD_ID,
        "independent_two_phase_method_id": (
            INDEPENDENT_MOVING_PAIR_METHOD_ID
        ),
        "phase_domain_rule": (
            "COMPLETE_CARTESIAN_PRODUCT_OF_BOTH_REGISTERED_PHASE_INTERVALS"
        ),
        "link_dependency_rule": (
            "EVERY_LINK_MUST_DEPEND_ON_AT_MOST_ONE_CLOSURE_SUPPORT"
        ),
        "same_support_rule": "SHARED_OR_SINGLE_SUPPORT_PHASE_PATH",
        "cross_support_rule": "INDEPENDENT_SUPPORT_PHASE_PRODUCT",
        "display_approximation_used_as_formal_evidence": False,
        "endpoint_or_finite_corner_sampling_allowed": False,
        "srdf_exemptions_applied": False,
        "exact_candidate_collision_path_changed": False,
        "current_state": "NOT_CERTIFIABLE",
        "checkable_scope": (
            "HAND_LINK_NONPAD_OBJECT_AND_SELF_PAIR_CONTACT_RANGE_"
            "CLOSURE_ONLY"
        ),
        "mandatory_blockers": list(
            CONTACT_RANGE_POLICY_MANDATORY_BLOCKERS
        ),
        "claim_limitations": list(
            CONTACT_RANGE_POLICY_CLAIM_LIMITATIONS
        ),
        "formal_selection_allowed": False,
        "isaac_launch_allowed": False,
    }.items():
        _exact(
            policy_collision[field],
            expected,
            f"method.contact_range_policy_collision.{field}",
        )
    sobol_design = _mapping(
        candidate["sobol_design"],
        "method.candidate_optimization.sobol_design",
    )
    for field, expected in {
        "generator": "SCIPY_STATS_QMC_SOBOL_VIA_DETERMINISTIC_SOBOL",
        "scipy_version": _SCIPY_VERSION,
        "scramble": True,
        "optimization": None,
        "generate_common_maximum_then_take_lane_prefix": True,
        "maximum_points_per_lane": MAXIMUM_POINTS_PER_LANE,
        "realized_design_hashes_are_contract_bound": True,
    }.items():
        _exact(
            sobol_design[field],
            expected,
            f"method.candidate_optimization.sobol_design.{field}",
        )
    _exact(
        scipy.__version__,
        _SCIPY_VERSION,
        "runtime scipy.__version__",
    )
    qmc_dependency = _mapping(
        sobol_design["dependency"],
        "method.candidate_optimization.sobol_design.dependency",
    )
    for field, expected in {
        "package": "scipy",
        "version": _SCIPY_VERSION,
        "python_manifest_path": "src/kcg_connector/setup.py",
        "python_manifest_sha256": _SETUP_MANIFEST_SHA256,
        "ros_package": "python3-scipy",
        "ros_manifest_path": "src/kcg_connector/package.xml",
        "ros_manifest_sha256": _PACKAGE_XML_SHA256,
    }.items():
        _exact(
            qmc_dependency[field],
            expected,
            "method.candidate_optimization.sobol_design.dependency."
            f"{field}",
        )
    local_refinement = _mapping(
        candidate["local_refinement"],
        "method.candidate_optimization.local_refinement",
    )
    for field, expected in {
        "method_id": "CANONICAL_V9_DYADIC_STENCIL_V1",
        "execution_status": "DISABLED_FOR_V1",
        "evaluation_budget": LOCAL_REFINEMENT_EVALUATION_BUDGET,
        "ranking_eligible": False,
    }.items():
        _exact(
            local_refinement[field],
            expected,
            f"method.candidate_optimization.local_refinement.{field}",
        )
    ray_closure = _mapping(document["ray_closure"], "method.ray_closure")
    _exact(
        ray_closure["method_id"],
        RAY_CLOSURE_METHOD_ID,
        "method.ray_closure.method_id",
    )
    _exact(
        ray_closure["maximum_subdivision_intervals"],
        4096,
        "method.ray_closure.maximum_subdivision_intervals",
    )
    _exact(
        ray_closure["maximum_subdivision_intervals_role"],
        "PRE_REGISTERED_COMPUTE_BUDGET_NOT_PHYSICAL_ACCEPTANCE_THRESHOLD",
        "method.ray_closure.maximum_subdivision_intervals_role",
    )
    _exact(
        ray_closure["subdivision_budget_exhaustion_policy"],
        "FAIL_CLOSED_REJECT_CANDIDATE_WITH_AUDIT",
        "method.ray_closure.subdivision_budget_exhaustion_policy",
    )
    _exact(
        ray_closure["budget_convergence_values"],
        [1024, 2048, 4096, 8192],
        "method.ray_closure.budget_convergence_values",
    )
    _exact(
        ray_closure["budget_convergence_role"],
        "COMPUTE_CONVERGENCE_AUDIT_NOT_PHYSICAL_PASS_GATE",
        "method.ray_closure.budget_convergence_role",
    )
    _exact(
        ray_closure["adjacent_budget_result_stability_report_required"],
        True,
        "method.ray_closure.adjacent_budget_result_stability_report_required",
    )
    _exact(
        ray_closure["physical_acceptance_gate"],
        False,
        "method.ray_closure.physical_acceptance_gate",
    )
    interval_backend = _mapping(
        ray_closure["interval_backend"],
        "method.ray_closure.interval_backend",
    )
    _exact(
        interval_backend["method_id"],
        INTERVAL_KINEMATICS_METHOD_ID,
        "method.ray_closure.interval_backend.method_id",
    )
    dependency = _mapping(
        interval_backend["dependency"],
        "method.ray_closure.interval_backend.dependency",
    )
    for field, expected in {
        "package": "mpmath",
        "version": "1.2.1",
        "manifest_path": "src/kcg_connector/setup.py",
        "manifest_sha256": _SETUP_MANIFEST_SHA256,
    }.items():
        _exact(
            dependency[field],
            expected,
            f"method.ray_closure.interval_backend.dependency.{field}",
        )
    for field, expected in {
        "decimal_precision": 80,
        "decimal_precision_role": (
            "NUMERICAL_INTERVAL_ARITHMETIC_PRECISION_NOT_PHYSICAL_"
            "ACCEPTANCE_THRESHOLD"
        ),
        "precision_convergence_values": [50, 80, 120],
        "precision_convergence_role": (
            "NUMERICAL_PRECISION_CONVERGENCE_AUDIT_NOT_PHYSICAL_PASS_GATE"
        ),
        "maximum_root_bisection_iterations": 256,
        "maximum_root_bisection_iterations_role": (
            "PRE_REGISTERED_COMPUTE_BUDGET_NOT_PHYSICAL_ACCEPTANCE_THRESHOLD"
        ),
        "root_budget_convergence_values": [128, 256, 512],
        "root_budget_exhaustion_policy": "FAIL_CLOSED_UNRESOLVED",
        "predicate_policy": (
            "STRICT_INTERVAL_SEPARATION_FROM_ZERO_NO_GEOMETRIC_TOLERANCE"
        ),
        "adjacent_precision_result_stability_report_required": True,
        "physical_acceptance_gate": False,
    }.items():
        _exact(
            interval_backend[field],
            expected,
            f"method.ray_closure.interval_backend.{field}",
        )
    domain = _mapping(
        ray_closure["closure_parameter_domain"],
        "method.ray_closure.closure_parameter_domain",
    )
    _exact(
        domain["domain_id"],
        CLOSURE_PARAMETER_DOMAIN_ID,
        "method.ray_closure.closure_parameter_domain.domain_id",
    )
    _exact(
        domain["coordinate_space"],
        "UNIT_HYPERCUBE",
        "method.ray_closure.closure_parameter_domain.coordinate_space",
    )
    dimension = _positive_integer(
        domain["dimension"],
        "method.ray_closure.closure_parameter_domain.dimension",
    )
    layout = domain["parameter_layout"]
    if (
        not isinstance(layout, Sequence)
        or isinstance(layout, (str, bytes, bytearray))
        or any(not isinstance(name, str) or not name for name in layout)
    ):
        raise StudyContractError(
            "method.ray_closure.closure_parameter_domain.parameter_layout "
            "must be a sequence of non-empty strings"
        )
    if len(layout) != dimension or len(set(layout)) != len(layout):
        raise StudyContractError(
            "method.ray_closure.closure_parameter_domain.parameter_layout "
            "must be unique and match dimension"
        )
    _exact(
        list(layout[: len(PARAMETER_LAYOUT_PREFIX)]),
        list(PARAMETER_LAYOUT_PREFIX),
        "method.ray_closure.closure_parameter_domain.parameter_layout prefix",
    )
    _exact(
        domain["dimension_and_layout_source"],
        "HAND_URDF_JOINT_LIMITS_PLUS_CLOSURE_ACTUATION_CONTRACT",
        "method.ray_closure.closure_parameter_domain."
        "dimension_and_layout_source",
    )
    yaw = _mapping(
        domain["assembly_axis_yaw"],
        "method.ray_closure.closure_parameter_domain.assembly_axis_yaw",
    )
    for field, expected in {
        "coordinate_name": "assembly_axis_yaw_unit",
        "unit_interval": "HALF_OPEN_ZERO_TO_ONE",
        "physical_mapping": "TWO_PI_ROTATION_ABOUT_OBJECT_ASSEMBLY_AXIS",
        "source": "OBJECT_CONTRACT_ASSEMBLY_AXIS_AND_TASK_FRAME",
    }.items():
        _exact(
            yaw[field],
            expected,
            f"method.ray_closure.closure_parameter_domain.assembly_axis_yaw.{field}",
        )
    placement = _mapping(
        domain["placement"],
        "method.ray_closure.closure_parameter_domain.placement",
    )
    for field, expected in {
        "coordinate_names": [
            "axial_target_unit",
            "lateral_task_x_unit",
            "lateral_task_y_unit",
        ],
        "unit_interval": "CLOSED_ZERO_TO_ONE",
        "physical_bounds_source": (
            "OBJECT_CONTRACT_PLANNING_GEOMETRY_BOUNDS_INTERSECTED_WITH_"
            "FULL_CLOSURE_SWEPT_PAD_AABB"
        ),
        "zero_width_axis_convention": "UNIT_COORDINATE_MUST_EQUAL_ZERO",
    }.items():
        _exact(
            placement[field],
            expected,
            f"method.ray_closure.closure_parameter_domain.placement.{field}",
        )
    preshape = _mapping(
        domain["preshape"],
        "method.ray_closure.closure_parameter_domain.preshape",
    )
    for field, expected in {
        "coordinate_prefix": "preshape_joint_unit",
        "unit_interval": "CLOSED_ZERO_TO_ONE",
        "joint_selection_source": (
            "POSITIVE_SPAN_NONCLOSURE_INDEPENDENT_JOINTS_FROM_HAND_URDF_"
            "AND_CLOSURE_ACTUATION_CONTRACT"
        ),
        "physical_bounds_source": "FULL_URDF_JOINT_LIMITS",
    }.items():
        _exact(
            preshape[field],
            expected,
            f"method.ray_closure.closure_parameter_domain.preshape.{field}",
        )
    joint_names = preshape["joint_names"]
    if (
        not isinstance(joint_names, Sequence)
        or isinstance(joint_names, (str, bytes, bytearray))
        or any(not isinstance(name, str) or not name for name in joint_names)
        or len(set(joint_names)) != len(joint_names)
    ):
        raise StudyContractError(
            "method.ray_closure.closure_parameter_domain.preshape.joint_names "
            "must be a unique sequence of non-empty strings"
        )

    linear = _mapping(document["linear_program"], "method.linear_program")
    _exact(linear["solver"], "SCIPY_HIGHS", "method.linear_program.solver")
    _exact(
        linear["constraint_scaling"],
        "ROW_AND_COLUMN_INF_NORM",
        "method.linear_program.constraint_scaling",
    )
    _positive_integer(linear["maximum_iterations"], "method.linear_program.maximum_iterations")
    for field in (
        "primal_feasibility_tolerance",
        "dual_feasibility_tolerance",
        "ipm_optimality_tolerance",
    ):
        _number(linear[field], f"method.linear_program.{field}", positive=True)
    _number_sequence(
        linear["tolerance_convergence_multipliers"],
        "method.linear_program.tolerance_convergence_multipliers",
        positive=True,
    )
    _exact(
        linear["physical_acceptance_gate"],
        False,
        "method.linear_program.physical_acceptance_gate",
    )

    contact = _mapping(document["contact_model"], "method.contact_model")
    for field, expected in {
        "type": "POINT_CONTACT_WITH_FRICTION_AND_FINITE_PAD_FEASIBILITY",
        "friction_cone_approximation": "CERTIFIED_INNER_REGULAR_POLYGON",
        "normal_force_capacity_source": "HAND_CONTACT_MODEL",
        "joint_torque_limits_source": "URDF_AND_ACTUATOR_CONTRACT",
    }.items():
        _exact(contact[field], expected, f"method.contact_model.{field}")
    cone_error = _number(
        contact["maximum_inner_approximation_relative_error"],
        "method.contact_model.maximum_inner_approximation_relative_error",
        positive=True,
    )
    if cone_error >= 1.0:
        raise StudyContractError("friction-cone inner approximation error must be below one")
    if _positive_integer(
        contact["cone_edge_convergence_multiplier"],
        "method.contact_model.cone_edge_convergence_multiplier",
    ) <= 1:
        raise StudyContractError("friction-cone convergence multiplier must exceed one")

    uncertainty = _mapping(document["uncertainty"], "method.uncertainty")
    for field, expected in {
        "scenario_design": "SCRAMBLED_SOBOL",
        "scenario_role": "BOUNDED_INTERVAL_QMC_SENSITIVITY_NOT_PROBABILITY_RISK",
        "per_object_interval_source": "OBJECT_CONTRACT",
    }.items():
        _exact(uncertainty[field], expected, f"method.uncertainty.{field}")
    _exact(uncertainty["probability_distribution_claimed"], False, "method.uncertainty.probability_distribution_claimed")
    _exact(uncertainty["post_result_interval_changes_allowed"], False, "method.uncertainty.post_result_interval_changes_allowed")
    _exact(uncertainty["report_hard_bound_minimum"], True, "method.uncertainty.report_hard_bound_minimum")
    _exact(
        uncertainty["scenario_count"],
        POST_GENERATION_SCENARIO_COUNT,
        "method.uncertainty.scenario_count",
    )
    nested_count = _positive_integer(
        uncertainty["nested_convergence_scenario_count"],
        "method.uncertainty.nested_convergence_scenario_count",
    )
    _exact(
        nested_count,
        256,
        "method.uncertainty.nested_convergence_scenario_count",
    )
    _exact(
        uncertainty["nested_convergence_role"],
        _NESTED_CONVERGENCE_ROLE,
        "method.uncertainty.nested_convergence_role",
    )
    _exact(
        uncertainty["sobol_seed"],
        POST_GENERATION_SCENARIO_SOBOL_SEED,
        "method.uncertainty.sobol_seed",
    )
    _validate_nested_post_generation_design(nested_count)
    lower_tail = _number(uncertainty["lower_tail_fraction"], "method.uncertainty.lower_tail_fraction", positive=True)
    if lower_tail >= 1.0:
        raise StudyContractError("method.uncertainty.lower_tail_fraction must be below one")

    ranking = _mapping(
        document["post_generation_ranking"],
        "method.post_generation_ranking",
    )
    for field, expected in {
        "implementation_type_id": _implementation_type_id(
            PostGenerationRankOnlyPipeline
        ),
        "method_id": POST_GENERATION_RANKER_METHOD_ID,
        "input_type_id": _implementation_type_id(TopLevelGenerationResult),
        "output_type_id": _implementation_type_id(PostGenerationRankResult),
        "execution_role": (
            "RANK_ONLY_WITHOUT_GENERATION_REFINEMENT_REPLACEMENT_OR_RETRY"
        ),
        "tie_break_rule": POST_GENERATION_TIE_BREAK_RULE,
    }.items():
        _exact(
            ranking[field],
            expected,
            f"method.post_generation_ranking.{field}",
        )
    _exact(
        ranking["selection"],
        "LEXICOGRAPHIC",
        "method.post_generation_ranking.selection",
    )
    _exact(
        ranking["selection_order"],
        list(POST_GENERATION_SELECTION_ORDER),
        "method.post_generation_ranking.selection_order",
    )
    common_scenarios = _mapping(
        ranking["common_scenarios"],
        "method.post_generation_ranking.common_scenarios",
    )
    for field, expected in {
        "method_id": POST_GENERATION_SCENARIO_METHOD_ID,
        "source": "SHARED_UNCERTAINTY_PROTOCOL",
        "identical_realizations_for_every_candidate": True,
        "candidate_specific_resampling_allowed": False,
        "scenario_design": uncertainty["scenario_design"],
        "scipy_version": POST_GENERATION_SCIPY_VERSION,
        "dimension": POST_GENERATION_SCENARIO_DIMENSION,
        "scenario_count": POST_GENERATION_SCENARIO_COUNT,
        "sobol_seed": POST_GENERATION_SCENARIO_SOBOL_SEED,
        "scramble": True,
        "optimization": None,
        "identity_encoding": "BIG_ENDIAN_BINARY64_ROW_MAJOR",
    }.items():
        _exact(
            common_scenarios[field],
            expected,
            f"method.post_generation_ranking.common_scenarios.{field}",
        )
    realized_post_generation_sha256 = _post_generation_design_sha256()
    _exact(
        realized_post_generation_sha256,
        POST_GENERATION_SCENARIO_DESIGN_SHA256,
        "production post-generation common scenario design SHA-256",
    )
    _exact(
        common_scenarios["design_sha256"],
        realized_post_generation_sha256,
        "method.post_generation_ranking.common_scenarios.design_sha256",
    )
    failure_retention = _mapping(
        ranking["failure_retention"],
        "method.post_generation_ranking.failure_retention",
    )
    for field, expected in {
        "retain_every_generation_attempt": True,
        "retain_every_unique_accepted_candidate": True,
        "retain_every_unique_accepted_policy": True,
        "retain_every_generation_and_evaluation_failure": True,
        "failure_reason_or_exception_required": True,
        "failed_candidate_drop_allowed": False,
        "collision_invocations_per_unique_accepted_candidate": 1,
        "wrench_invocations_per_unique_accepted_candidate": 1,
        "retry_allowed": False,
        "replacement_after_failure_allowed": False,
    }.items():
        _exact(
            failure_retention[field],
            expected,
            f"method.post_generation_ranking.failure_retention.{field}",
        )
    formal_selection = _mapping(
        ranking["formal_selection"],
        "method.post_generation_ranking.formal_selection",
    )
    for field, expected in {
        "status": _FORMAL_EMPTY_STATUS,
        "allowed_with_current_bindings": False,
        "formal_ranked_keys_must_be_empty": True,
        "selected_candidate_must_be_none": True,
        "contact_range_policy_handling": (
            "FAIL_CLOSED_BEFORE_COLLISION_OR_WRENCH_UNTIL_"
            "POLICY_AWARE_CONSUMERS_EXIST"
        ),
        "contact_range_policy_collision_invocations_before_support": 0,
        "contact_range_policy_wrench_invocations_before_support": 0,
        "required_collision_claim_scope": COMPLETE_CLEARANCE_SCOPE,
        "current_uncertainty_claim_scope": (
            FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ),
        "required_additional_uncertainty_binding": (
            FORMAL_UNCERTAINTY_BLOCKER
        ),
    }.items():
        _exact(
            formal_selection[field],
            expected,
            f"method.post_generation_ranking.formal_selection.{field}",
        )

    wrench = _mapping(document["task_wrench"], "method.task_wrench")
    for field, expected in {
        "representation": "CONVEX_POLYTOPE",
        "gravity_acceleration_source": "EXACT_CONVENTIONAL_STANDARD_GRAVITY",
        "task_frame_source": "OBJECT_CONTRACT_CAD_DATUM",
        "disturbance_body": "CENTRALLY_SYMMETRIC_UNIT_6D_CROSS_POLYTOPE",
        "force_normalization": "OBJECT_WEIGHT",
        "moment_normalization": "OBJECT_WEIGHT_TIMES_CHARACTERISTIC_RADIUS",
        "characteristic_radius_source": "MASS_DISTRIBUTION_RMS_RADIUS_SQRT_TRACE_I_COM_OVER_2M",
    }.items():
        _exact(wrench[field], expected, f"method.task_wrench.{field}")
    _number(wrench["gravity_acceleration_m_s2"], "method.task_wrench.gravity_acceleration_m_s2", positive=True)
    required = _mapping(wrench["required_wrench"], "method.task_wrench.required_wrench")
    _exact(required["gravity"], True, "method.task_wrench.required_wrench.gravity")
    _number(required["lift_acceleration_m_s2"], "method.task_wrench.required_wrench.lift_acceleration_m_s2", positive=True)
    _exact(wrench["binary_quality_pass_threshold_allowed"], False, "method.task_wrench.binary_quality_pass_threshold_allowed")

    controller = _mapping(document["controller"], "method.controller")
    for field, expected in {
        "architecture": "PASSIVE_OBJECT_IMPEDANCE_PLUS_GRASP_NULLSPACE_FORCE_QP",
        "force_target_source": "ROBUST_OPTIMIZER",
        "object_pose_feedback_source": "ESTIMATOR_ONLY",
        "contact_feedback_source": "PROPRIOCEPTION_WRIST_FT_OR_PHYSICAL_TACTILE_ONLY",
        "qp_regularization_source": "FORCE_QP_NUMERICAL_CONTRACT",
    }.items():
        _exact(controller[field], expected, f"method.controller.{field}")
    _exact(controller["fixed_force_increment_allowed"], False, "method.controller.fixed_force_increment_allowed")

    passive = _mapping(document["passive_impedance_numerics"], "method.passive_impedance_numerics")
    for field, value in passive.items():
        _number(value, f"method.passive_impedance_numerics.{field}", positive=True)

    force_qp = _mapping(document["force_qp"], "method.force_qp")
    _exact(force_qp["solver"], "SCIPY_SLSQP_WITH_HIGHS_FEASIBILITY", "method.force_qp.solver")
    _exact(force_qp["constraint_scaling"], "CALLER_SUPPLIED_WRENCH_SCALES", "method.force_qp.constraint_scaling")
    _positive_integer(force_qp["maximum_iterations"], "method.force_qp.maximum_iterations")
    for field in (
        "objective_tolerance",
        "equality_tolerance",
        "inequality_tolerance",
        "linear_independence_tolerance",
        "feasibility_dual_tolerance",
        "regularization",
    ):
        _number(force_qp[field], f"method.force_qp.{field}", positive=True)
    _exact(force_qp["physical_acceptance_gate"], False, "method.force_qp.physical_acceptance_gate")

    convergence = _mapping(document["numerical_convergence"], "method.numerical_convergence")
    _number_sequence(convergence["tolerance_multipliers"], "method.numerical_convergence.tolerance_multipliers", positive=True)
    _exact(convergence["result_stability_required"], True, "method.numerical_convergence.result_stability_required")
    _exact(convergence["physical_acceptance_gate"], False, "method.numerical_convergence.physical_acceptance_gate")

    dynamic = _mapping(document["dynamic_protocol"], "method.dynamic_protocol")
    _number_sequence(dynamic["lift_targets_m"], "method.dynamic_protocol.lift_targets_m", positive=True)
    _number(dynamic["hold_duration_s"], "method.dynamic_protocol.hold_duration_s", positive=True)
    _exact(dynamic["disturbance_direction_design"], "SPHERICAL_FIBONACCI_PAIRED_6D", "method.dynamic_protocol.disturbance_direction_design")
    _positive_integer(dynamic["disturbance_direction_count"], "method.dynamic_protocol.disturbance_direction_count")
    _number_sequence(dynamic["dimensionless_amplitudes"], "method.dynamic_protocol.dimensionless_amplitudes", nonnegative=True)
    _number(dynamic["pulse_duration_s"], "method.dynamic_protocol.pulse_duration_s", positive=True)
    seeds = dynamic["paired_random_seeds"]
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or not seeds:
        raise StudyContractError("method.dynamic_protocol.paired_random_seeds must be non-empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise StudyContractError("method.dynamic_protocol.paired_random_seeds must be integers")
    _exact(dynamic["independent_isaac_process_per_trial"], True, "method.dynamic_protocol.independent_isaac_process_per_trial")

    reporting = _mapping(document["reporting"], "method.reporting")
    confidence = _number(reporting["confidence_level"], "method.reporting.confidence_level", positive=True)
    if confidence >= 1.0:
        raise StudyContractError("method.reporting.confidence_level must be below one")
    _exact(reporting["report_all_failures"], True, "method.reporting.report_all_failures")
    _exact(reporting["success_unit"], "CONNECTOR_TRIAL", "method.reporting.success_unit")
    _exact(reporting["physics_step_is_independent_sample"], False, "method.reporting.physics_step_is_independent_sample")
    _exact(reporting["generated_images_as_evidence_allowed"], False, "method.reporting.generated_images_as_evidence_allowed")
    _exact(reporting["figure_data_sources"], ["RAW_JSON", "RAW_CSV", "FROZEN_CAD", "CANDIDATE_BUNDLE"], "method.reporting.figure_data_sources")


def _reject_object_algorithm_overrides(value: Any, path: str) -> None:
    forbidden_fragments = (
        "algorithm",
        "hyperparameter",
        "candidate_budget",
        "selection_order",
        "solver_tolerance",
        "subdivision_interval",
        "parameter_domain",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in forbidden_fragments):
                raise StudyContractError(
                    f"object-specific algorithm override is forbidden at {path}.{key}"
                )
            _reject_object_algorithm_overrides(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_object_algorithm_overrides(child, f"{path}[{index}]")


def _validate_ray_closure_hand_binding(
    method_document: Mapping[str, Any],
    hand_contract: CARTSHandContract,
) -> None:
    """Bind both candidate domains to the production hand derivation."""

    try:
        hand_model = hand_contract.build_hand_model()
        directions = hand_contract.closing_actuation_directions_unit(hand_model)
    except HandContractError as exc:
        raise StudyContractError(
            "cannot derive the RayClosure parameter domain from the hand contract"
        ) from exc
    lower, upper = hand_model.joint_limit_vectors()
    spans = upper - lower
    used_closure_indices: set[int] = set()
    for row_index, row in enumerate(directions):
        support = tuple(int(index) for index in np.flatnonzero(row != 0.0))
        if len(support) != 1:
            raise StudyContractError(
                "RayClosure production domain requires one exclusive closure "
                f"joint in hand closure row {row_index}"
            )
        if support[0] in used_closure_indices:
            raise StudyContractError(
                "RayClosure production domain closure supports must be disjoint"
            )
        used_closure_indices.add(support[0])
    preshape_joint_names = tuple(
        name
        for index, name in enumerate(hand_model.independent_joint_names)
        if index not in used_closure_indices and spans[index] > 0.0
    )
    expected_layout = PARAMETER_LAYOUT_PREFIX + tuple(
        f"preshape_joint_unit:{name}" for name in preshape_joint_names
    )
    domain = method_document["ray_closure"]["closure_parameter_domain"]
    _exact(
        domain["dimension"],
        len(expected_layout),
        "method.ray_closure.closure_parameter_domain.dimension "
        "derived from hand",
    )
    _exact(
        domain["parameter_layout"],
        list(expected_layout),
        "method.ray_closure.closure_parameter_domain.parameter_layout "
        "derived from hand",
    )
    _exact(
        domain["preshape"]["joint_names"],
        list(preshape_joint_names),
        "method.ray_closure.closure_parameter_domain.preshape.joint_names "
        "derived from hand",
    )
    candidate = _mapping(
        method_document["candidate_optimization"],
        "method.candidate_optimization",
    )
    candidate_hand = _mapping(
        candidate["hand_binding"],
        "method.candidate_optimization.hand_binding",
    )
    pad_names = tuple(pad.name for pad in hand_contract.pads)
    if len(pad_names) != 3 or len(set(pad_names)) != len(pad_names):
        raise StudyContractError(
            "top-level generator requires three distinct hand-derived "
            "PAD names"
        )
    _exact(
        candidate_hand["preshape_joint_names"],
        list(preshape_joint_names),
        "method.candidate_optimization.hand_binding.preshape_joint_names "
        "derived from hand",
    )
    _exact(
        candidate_hand["prepared_pad_order"],
        list(pad_names),
        "method.candidate_optimization.hand_binding.prepared_pad_order "
        "derived from hand",
    )
    candidate_v9 = _mapping(
        candidate["v9_certifier"],
        "method.candidate_optimization.v9_certifier",
    )
    _exact(
        candidate_v9["parameter_layout"],
        list(expected_layout),
        "method.candidate_optimization.v9_certifier.parameter_layout "
        "derived from hand",
    )
    expected_fixed_layout = FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX + tuple(
        f"preshape_joint_unit:{name}" for name in preshape_joint_names
    )
    candidate_fixed = _mapping(
        candidate["fixed_anchor_mapper"],
        "method.candidate_optimization.fixed_anchor_mapper",
    )
    _exact(
        candidate_fixed["parameter_layout"],
        list(expected_fixed_layout),
        "method.candidate_optimization.fixed_anchor_mapper.parameter_layout "
        "derived from hand",
    )
    lanes = _mapping(
        candidate["lanes"],
        "method.candidate_optimization.lanes",
    )
    for spec in LANE_SPECS:
        lane = _mapping(
            lanes[spec.lane.value],
            f"method.candidate_optimization.lanes.{spec.lane.value}",
        )
        expected_pad_name = (
            None
            if spec.anchor_pad_ordinal is None
            else pad_names[spec.anchor_pad_ordinal]
        )
        _exact(
            lane["anchor_pad_name"],
            expected_pad_name,
            "method.candidate_optimization.lanes."
            f"{spec.lane.value}.anchor_pad_name derived from hand",
        )


def _validate_vector3(value: Any, label: str) -> tuple[float, float, float]:
    parsed = _number_sequence(value, label)
    if len(parsed) != 3:
        raise StudyContractError(f"{label} must contain exactly three numbers")
    return parsed[0], parsed[1], parsed[2]


def _validate_matrix3(value: Any, label: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise StudyContractError(f"{label} must contain three rows")
    for index, row in enumerate(value):
        _validate_vector3(row, f"{label}[{index}]")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise StudyContractError(f"{label} must be a non-empty string")
    return value


def _validate_unit_vector(value: Any, label: str) -> tuple[float, float, float]:
    parsed = _validate_vector3(value, label)
    if math.fsum(component * component for component in parsed) != 1.0:
        raise StudyContractError(f"{label} must be an exactly registered unit vector")
    return parsed


def _validate_rotation_and_axis(
    value: Any,
    axis: tuple[float, float, float],
    label: str,
) -> None:
    _validate_matrix3(value, label)
    rotation = tuple(tuple(float(component) for component in row) for row in value)
    columns = tuple(tuple(rotation[row][column] for row in range(3)) for column in range(3))
    for first in range(3):
        for second in range(3):
            dot = math.fsum(
                columns[first][index] * columns[second][index]
                for index in range(3)
            )
            expected = 1.0 if first == second else 0.0
            if dot != expected:
                raise StudyContractError(
                    f"{label} must be an exactly registered orthonormal rotation"
                )
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if determinant != 1.0 or columns[2] != axis:
        raise StudyContractError(
            f"{label} must be proper and its third axis must equal assembly_axis_object"
        )


def _validate_rigid_inertia(value: Any, label: str) -> None:
    _validate_matrix3(value, label)
    inertia = np.asarray(value, dtype=np.float64)
    if not np.array_equal(inertia, inertia.T):
        raise StudyContractError(f"{label} must be exactly symmetric")
    principal = np.linalg.eigvalsh(inertia)
    scale = max(float(np.linalg.norm(inertia, ord=2)), np.finfo(np.float64).tiny)
    numerical_bound = 256.0 * np.finfo(np.float64).eps * scale
    if principal[0] <= numerical_bound:
        raise StudyContractError(f"{label} must be positive definite")
    if principal[-1] > principal[0] + principal[1] + numerical_bound:
        raise StudyContractError(f"{label} violates the rigid-body triangle inequality")


def _validate_object_document(document: Mapping[str, Any]) -> None:
    objects_unchecked = document.get("objects")
    if isinstance(objects_unchecked, Mapping):
        for object_id, value in objects_unchecked.items():
            _reject_object_algorithm_overrides(value, f"objects.{object_id}")
    _validate_schema(document, _OBJECT_SCHEMA, "object contract")
    _exact(document["schema_version"], "carts_grasp_objects_v1", "objects.schema_version")
    _exact(document["study_id"], "CARTS-GRASP-CROSS-OBJECT-V1", "objects.study_id")
    _exact(document["claim_scope"], "SIMULATION_ONLY_ZERO_OBJECT_TUNING_CROSS_MODEL_CASE_STUDY", "objects.claim_scope")
    _exact(document["shared_method_config_required_for_every_object"], True, "objects.shared_method_config_required_for_every_object")
    _exact(document["object_specific_algorithm_hyperparameters_allowed"], False, "objects.object_specific_algorithm_hyperparameters_allowed")
    _exact(document["hardware_authorized"], False, "objects.hardware_authorized")

    transfer = _mapping(document["transfer_protocol"], "objects.transfer_protocol")
    _exact(transfer["development_object"], _DEVELOPMENT_OBJECT, "objects.transfer_protocol.development_object")
    _exact(transfer["frozen_transfer_object"], _TRANSFER_OBJECT, "objects.transfer_protocol.frozen_transfer_object")
    _exact(transfer["transfer_object_geometry_seen_before_method_freeze"], True, "objects.transfer_protocol.transfer_object_geometry_seen_before_method_freeze")
    _exact(transfer["prospective_double_blind_claim_allowed"], False, "objects.transfer_protocol.prospective_double_blind_claim_allowed")
    _exact(transfer["candidate_ids_or_contact_coordinates_shared_between_objects"], False, "objects.transfer_protocol.candidate_ids_or_contact_coordinates_shared_between_objects")

    objects = _mapping(document["objects"], "objects.objects")
    current = _mapping(objects[_DEVELOPMENT_OBJECT], f"objects.{_DEVELOPMENT_OBJECT}")
    transfer_object = _mapping(objects[_TRANSFER_OBJECT], f"objects.{_TRANSFER_OBJECT}")
    for object_id, identity_value in (
        (_DEVELOPMENT_OBJECT, current["identity"]),
        (_TRANSFER_OBJECT, transfer_object["identity"]),
    ):
        identity = _mapping(identity_value, f"objects.{object_id}.identity")
        _nonempty_string(identity["connector"], f"objects.{object_id}.identity.connector")
        _positive_integer(identity["shell_size"], f"objects.{object_id}.identity.shell_size")
        _positive_integer(identity["contact_count"], f"objects.{object_id}.identity.contact_count")
    _positive_integer(
        transfer_object["identity"]["insert_arrangement"],
        f"objects.{_TRANSFER_OBJECT}.identity.insert_arrangement",
    )
    _nonempty_string(
        transfer_object["identity"]["te_catalog_number"],
        f"objects.{_TRANSFER_OBJECT}.identity.te_catalog_number",
    )
    _exact(current["identity"]["source_class"], "PUBLIC_SPEC_SIMULATION_MODEL_NOT_VENDOR_CAD", f"objects.{_DEVELOPMENT_OBJECT}.identity.source_class")
    _exact(transfer_object["identity"]["manufacturer"], "TE_CONNECTIVITY_DEUTSCH", f"objects.{_TRANSFER_OBJECT}.identity.manufacturer")
    _exact(transfer_object["identity"]["source_class"], "OFFICIAL_TE_CUSTOMER_VIEW_MODEL_STEP", f"objects.{_TRANSFER_OBJECT}.identity.source_class")

    friction_intervals: dict[str, tuple[float, ...]] = {}
    for object_id, object_row in objects.items():
        row = _mapping(object_row, f"objects.{object_id}")
        frames = _mapping(row["frames"], f"objects.{object_id}.frames")
        _exact(frames["length_unit"], "m", f"objects.{object_id}.frames.length_unit")
        axis = _validate_unit_vector(frames["assembly_axis_object"], f"objects.{object_id}.frames.assembly_axis_object")
        _validate_rotation_and_axis(
            frames["task_frame_rotation_object"],
            axis,
            f"objects.{object_id}.frames.task_frame_rotation_object",
        )
        _validate_unit_vector(frames["nominal_validation_gravity_direction_object"], f"objects.{object_id}.frames.nominal_validation_gravity_direction_object")
        _nonempty_string(frames["task_frame_source"], f"objects.{object_id}.frames.task_frame_source")
        geometry = _mapping(row["planning_geometry"], f"objects.{object_id}.planning_geometry")
        _exact(geometry["allowed_surface_semantic"], "EXTERNALLY_FIRST_VISIBLE_PAD_REACHABLE_SURFACE", f"objects.{object_id}.planning_geometry.allowed_surface_semantic")
        _exact(geometry["simulator_truth_used"], False, f"objects.{object_id}.planning_geometry.simulator_truth_used")
        material = _mapping(row["contact_material_uncertainty"], f"objects.{object_id}.contact_material_uncertainty")
        friction = _number_sequence(material["friction_coefficient"], f"objects.{object_id}.contact_material_uncertainty.friction_coefficient", positive=True)
        if len(friction) != 2 or friction[0] > friction[1]:
            raise StudyContractError(f"objects.{object_id} friction coefficient must be an ordered closed interval")
        friction_intervals[object_id] = friction
        _exact(material["probability_distribution_claimed"], False, f"objects.{object_id}.contact_material_uncertainty.probability_distribution_claimed")
        _exact(material["vendor_friction_claimed"], False, f"objects.{object_id}.contact_material_uncertainty.vendor_friction_claimed")
        calibration = _mapping(row["uncertainty_calibration"], f"objects.{object_id}.uncertainty_calibration")
        _exact(calibration["simulation_mass_com_inertia_randomized"], False, f"objects.{object_id}.uncertainty_calibration.simulation_mass_com_inertia_randomized")
        _exact(calibration["placeholder_numeric_bounds_allowed"], False, f"objects.{object_id}.uncertainty_calibration.placeholder_numeric_bounds_allowed")
        for field in (
            "perception_pose_residual_source",
            "surface_normal_residual_source",
            "joint_and_torque_residual_source",
        ):
            source = calibration[field]
            if not isinstance(source, str) or not source.startswith("PENDING_"):
                raise StudyContractError(f"objects.{object_id}.uncertainty_calibration.{field} must remain explicitly pending")
        eligibility = _mapping(row["dynamic_eligibility"], f"objects.{object_id}.dynamic_eligibility")
        _exact(eligibility["allowed"], False, f"objects.{object_id}.dynamic_eligibility.allowed")
        reason = eligibility["reason"]
        if not isinstance(reason, str) or not reason.startswith("PENDING_"):
            raise StudyContractError(f"objects.{object_id}.dynamic_eligibility.reason must be a PENDING_ reason")

    current_material = _mapping(
        current["contact_material_uncertainty"], "current contact material"
    )
    transfer_material = _mapping(
        transfer_object["contact_material_uncertainty"], "transfer contact material"
    )
    _exact(
        current_material["source_class"],
        "FROZEN_SIMULATION_MATERIAL_ROLE_NOT_HARDWARE_CALIBRATION",
        "current contact material source_class",
    )
    _exact(
        transfer_material["source_class"],
        "SHARED_STUDY_SIMULATION_ASSUMPTION_FROM_DEVELOPMENT_MATERIAL_ROLE",
        "transfer contact material source_class",
    )
    if friction_intervals[_DEVELOPMENT_OBJECT] != friction_intervals[_TRANSFER_OBJECT]:
        raise StudyContractError(
            "development and transfer objects must consume one shared friction interval"
        )
    if current_material["source"] != transfer_material["source"]:
        raise StudyContractError(
            "development and transfer objects must consume one shared material source"
        )
    if current_material["source_sha256"] != transfer_material["source_sha256"]:
        raise StudyContractError(
            "development and transfer material source SHA-256 must match"
        )

    current_geometry = _mapping(current["planning_geometry"], "current planning geometry")
    _exact(current_geometry["format"], "CARTS_GRASP_VISUAL_SUBTREE_NPZ_V1", "current planning geometry format")
    _exact(current_geometry["semantic_authority"], "SHARED_DIRECTIONAL_FIRST_HIT_WITH_TASK_FUNCTIONAL_MASK_PENDING", "current planning geometry semantic_authority")
    current_physical = _mapping(current["physical_properties"], "current physical properties")
    _exact(current_physical["source_class"], "FROZEN_EQUIVALENT_SIMULATION_ASSUMPTION", "current physical source_class")
    _exact(current_physical["planning_rigid_composition"]["method"], "PARALLEL_AXIS_COMPOSITION_OF_FROZEN_COMPONENTS", "current planning rigid composition method")
    _exact(current_physical["planning_rigid_composition"]["vendor_hardware_truth_claimed"], False, "current vendor_hardware_truth_claimed")
    for component_name, component in current_physical["component_composition"].items():
        _number(component["mass_kg"], f"current component {component_name} mass_kg", positive=True)
        _validate_vector3(component["center_of_mass_m"], f"current component {component_name} center_of_mass_m")
        diagonal = _number_sequence(component["diagonal_inertia_kg_m2"], f"current component {component_name} diagonal_inertia_kg_m2", positive=True)
        if len(diagonal) != 3:
            raise StudyContractError(f"current component {component_name} diagonal inertia must have three values")

    transfer_frames = _mapping(transfer_object["frames"], "transfer frames")
    _exact(transfer_frames["source_step_length_unit"], "mm", "transfer source_step_length_unit")
    transfer_geometry = _mapping(transfer_object["planning_geometry"], "transfer planning geometry")
    _exact(transfer_geometry["format"], "BINARY_STL_TESSELLATION_FROM_ORIGINAL_STEP", "transfer planning geometry format")
    _exact(transfer_geometry["source_unit"], "mm", "transfer planning geometry source_unit")
    _exact(transfer_geometry["watertight"], True, "transfer planning geometry watertight")
    _exact(transfer_geometry["winding_consistent"], True, "transfer planning geometry winding_consistent")
    _exact(transfer_geometry["part_number_specific_z_or_radius_cutoffs_allowed"], False, "transfer part_number_specific_z_or_radius_cutoffs_allowed")
    _exact(transfer_geometry["semantic_derivation"], "FIRST_INTERSECTION_FROM_HAND_CLOSING_RAYS_PLUS_HAND_KINEMATIC_FEASIBILITY", "transfer semantic_derivation")
    original = _mapping(transfer_object["original_cad"], "transfer original CAD")
    _positive_integer(original["solid_count"], "transfer original CAD solid_count")
    _positive_integer(original["face_count"], "transfer original CAD face_count")
    transfer_physical = _mapping(transfer_object["physical_properties"], "transfer physical properties")
    _exact(transfer_physical["mass_source_class"], "INITIAL_SIMULATION_VALUE_NOT_VENDOR_MEASUREMENT", "transfer mass_source_class")
    _exact(transfer_physical["com_and_inertia_source_class"], "UNIFORM_DENSITY_EQUIVALENT_FROM_WATERTIGHT_STEP_TESSELLATION", "transfer com_and_inertia_source_class")
    _number(transfer_physical["uniform_density_kg_m3"], "transfer uniform_density_kg_m3", positive=True)
    _exact(transfer_physical["vendor_hardware_truth_claimed"], False, "transfer vendor_hardware_truth_claimed")
    _exact(transfer_object["contact_material_uncertainty"]["vendor_friction_claimed"], False, "transfer vendor_friction_claimed")
    _exact(transfer_object["dynamic_model"]["existing_te_j35_physx_v1_eligible"], False, "transfer existing PhysX eligibility")
    _exact(transfer_object["dynamic_model"]["required_representation"], "FREE_ACTIVITY_PLUG_WITH_COMPLETE_EXTERNAL_COLLISION_TABLE_AND_HAND", "transfer dynamic required_representation")
    _exact(transfer_object["dynamic_model"]["current_status"], "NOT_BUILT", "transfer dynamic model current_status")

    for object_id, object_row in objects.items():
        physical = _mapping(object_row["physical_properties"], f"objects.{object_id}.physical_properties")
        values = physical.get("planning_rigid_composition", physical)
        _number(values["mass_kg"], f"objects.{object_id}.mass_kg", positive=True)
        _validate_vector3(values["center_of_mass_m"], f"objects.{object_id}.center_of_mass_m")
        _validate_rigid_inertia(values["inertia_kg_m2"], f"objects.{object_id}.inertia_kg_m2")


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StudyContractError("contract is not canonical JSON serialisable") from exc
    return text.encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract_path(repository_root: Path, supplied: str | Path, label: str) -> Path:
    path = Path(supplied)
    path = path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    if not path.is_file():
        raise StudyContractError(f"{label} is unavailable: {path}")
    return path


def _normalised_repository_reference(
    repository_root: Path,
    reference: Any,
    label: str,
) -> tuple[Path, str, str | None]:
    if not isinstance(reference, str) or not reference:
        raise StudyContractError(f"{label} must be a non-empty repository-relative path")
    file_text, separator, fragment = reference.partition("#")
    if separator and not fragment:
        raise StudyContractError(f"{label} has an empty fragment")
    if "\\" in file_text:
        raise StudyContractError(f"{label} must use POSIX path syntax")
    pure = PurePosixPath(file_text)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.as_posix() != file_text
    ):
        raise StudyContractError(f"{label} must be normalized and repository-relative")
    try:
        absolute = (repository_root / Path(*pure.parts)).resolve(strict=True)
        absolute.relative_to(repository_root)
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        raise StudyContractError(f"{label} referenced file is unavailable or escapes repository") from exc
    if not absolute.is_file():
        raise StudyContractError(f"{label} does not reference a regular file")
    return absolute, file_text, fragment if separator else None


@dataclass(frozen=True)
class FrozenFileDigest:
    """One byte stream consumed by the audited study."""

    path: str
    sha256: str
    byte_count: int
    fragment: str | None = None
    declared_sha256: str | None = None


def _external_or_repository_digest(path: Path, repository_root: Path) -> FrozenFileDigest:
    try:
        name = path.relative_to(repository_root).as_posix()
    except ValueError:
        name = path.as_posix()
    return FrozenFileDigest(
        path=name,
        sha256=_sha256_file(path),
        byte_count=path.stat().st_size,
    )


def _verified_digest(
    repository_root: Path,
    reference: Any,
    declared_sha256: Any,
    label: str,
) -> FrozenFileDigest:
    path, relative, fragment = _normalised_repository_reference(
        repository_root, reference, label
    )
    if not isinstance(declared_sha256, str) or _HEX_SHA256.fullmatch(declared_sha256) is None:
        raise StudyContractError(f"{label} declared SHA-256 must be 64 lowercase hexadecimal digits")
    actual = _sha256_file(path)
    if actual != declared_sha256:
        raise StudyContractError(
            f"{label} SHA-256 mismatch: declared={declared_sha256}, actual={actual}"
        )
    return FrozenFileDigest(
        path=relative,
        sha256=actual,
        byte_count=path.stat().st_size,
        fragment=fragment,
        declared_sha256=declared_sha256,
    )


def _derived_geometry_hashes(
    method: Mapping[str, Any],
    hand: Mapping[str, Any],
    objects_document: Mapping[str, Any],
    files: Mapping[str, FrozenFileDigest],
) -> dict[str, str]:
    result = {
        "hand.closure_geometry_descriptor": _sha256_bytes(
            _canonical_bytes(
                {
                    "kinematics": hand["kinematics"],
                    "closure_actuation": hand["closure_actuation"],
                    "pads": hand["pads"],
                    "surface_model": method["surface_model"],
                    "contact_model": method["contact_model"],
                }
            )
        )
    }
    for pad_name in ("finger_1_pad", "finger_2_pad", "finger_3_pad"):
        result[f"hand.{pad_name}.finite_footprint_asset"] = files[
            f"hand.{pad_name}.finite_footprint"
        ].sha256
    objects = _mapping(objects_document["objects"], "objects")
    for object_id, row in objects.items():
        result[f"objects.{object_id}.planning_geometry_asset"] = files[
            f"objects.{object_id}.planning_geometry"
        ].sha256
        result[f"objects.{object_id}.grasp_model_descriptor"] = _sha256_bytes(
            _canonical_bytes(
                {
                    "frames": row["frames"],
                    "planning_geometry": row["planning_geometry"],
                    "physical_properties": row["physical_properties"],
                    "contact_material_uncertainty": row[
                        "contact_material_uncertainty"
                    ],
                }
            )
        )
    result[f"objects.{_TRANSFER_OBJECT}.original_step_asset"] = files[
        f"objects.{_TRANSFER_OBJECT}.original_cad"
    ].sha256
    return dict(sorted(result.items()))


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(value[key]) for key in sorted(value)}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class StudyContractAudit:
    """Immutable, signable audit manifest that is not a formal freeze."""

    input_files: Mapping[str, FrozenFileDigest]
    input_file_sha256: Mapping[str, str]
    derived_geometry_sha256: Mapping[str, str]
    preregistration_blockers: tuple[str, ...]
    canonical_manifest: Mapping[str, Any]
    canonical_json_bytes: bytes
    canonical_sha256: str

    @property
    def freeze_eligible(self) -> bool:
        return not self.preregistration_blockers


@dataclass(frozen=True)
class FrozenStudyContract:
    """Formal immutable contract, constructible only after all gates pass."""

    schema_version: str
    study_id: str
    method: str
    claim_scope: str
    development_object: str
    transfer_object: str
    transfer_object_role: str
    transfer_geometry_seen_before_method_freeze: bool
    prospective_double_blind_claim_allowed: bool
    hardware_authorized: bool
    dynamic_eligibility: Mapping[str, bool]
    numerical_tolerance_role: str
    input_files: Mapping[str, FrozenFileDigest]
    input_file_sha256: Mapping[str, str]
    derived_geometry_sha256: Mapping[str, str]
    canonical_manifest: Mapping[str, Any]
    canonical_json_bytes: bytes
    canonical_sha256: str


def audit_study_contract(
    shared_method_path: str | Path,
    hand_contract_path: str | Path,
    object_contract_path: str | Path,
    *,
    repository_root: str | Path,
) -> StudyContractAudit:
    """Validate and canonicalise the current study without declaring a freeze.

    No algorithmic or numerical value is defaulted.  The returned blockers are
    part of the signed audit content so an incomplete audit cannot be mistaken
    for a formal preregistration.
    """

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise StudyContractError("repository_root must be an existing directory")
    method_path = _contract_path(root, shared_method_path, "shared method YAML")
    hand_path = _contract_path(root, hand_contract_path, "hand YAML")
    object_path = _contract_path(root, object_contract_path, "object YAML")

    method_document = _load_unique_yaml(method_path, "shared method YAML")
    hand_document = _load_unique_yaml(hand_path, "hand YAML")
    object_document = _load_unique_yaml(object_path, "object YAML")
    _validate_method(method_document)
    _validate_object_document(object_document)

    declared_method_path, _relative, _fragment = _normalised_repository_reference(
        root,
        object_document["shared_method_config"],
        "objects.shared_method_config",
    )
    declared_method_document = _load_unique_yaml(
        declared_method_path, "objects.shared_method_config"
    )
    _validate_method(declared_method_document)
    if _canonical_bytes(method_document) != _canonical_bytes(declared_method_document):
        raise StudyContractError(
            "supplied shared method YAML differs from objects.shared_method_config"
        )

    try:
        hand_contract = load_carts_hand_contract(hand_path, repository_root=root)
    except HandContractError as exc:
        raise StudyContractError(f"hand contract rejected: {exc}") from exc
    if hand_contract.method != method_document["method"]:
        raise StudyContractError("hand and shared method identifiers differ")
    if not hand_contract.truth_firewall_all_false:
        raise StudyContractError("hand online truth firewall is not closed")
    if hand_contract.dynamic_use_allowed:
        raise StudyContractError("static hand force capacity cannot be upgraded")
    _validate_ray_closure_hand_binding(method_document, hand_contract)

    files: dict[str, FrozenFileDigest] = {
        "config.shared_method": _external_or_repository_digest(method_path, root),
        "config.hand_contract": _external_or_repository_digest(hand_path, root),
        "config.object_contract": _external_or_repository_digest(object_path, root),
        "hand.urdf": FrozenFileDigest(
            path=hand_contract.urdf.repository_relative_path,
            sha256=hand_contract.urdf.sha256,
            byte_count=hand_contract.urdf.byte_count,
            declared_sha256=hand_contract.urdf.sha256,
        ),
        "hand.pad_source_manifest": FrozenFileDigest(
            path=hand_contract.source_manifest.repository_relative_path,
            sha256=hand_contract.source_manifest.sha256,
            byte_count=hand_contract.source_manifest.byte_count,
            declared_sha256=hand_contract.source_manifest.sha256,
        ),
    }
    interval_dependency = _mapping(
        _mapping(
            method_document["ray_closure"],
            "method.ray_closure",
        )["interval_backend"],
        "method.ray_closure.interval_backend",
    )["dependency"]
    interval_dependency = _mapping(
        interval_dependency,
        "method.ray_closure.interval_backend.dependency",
    )
    files["method.interval_backend_dependency_manifest"] = _verified_digest(
        root,
        interval_dependency["manifest_path"],
        interval_dependency["manifest_sha256"],
        "interval backend dependency manifest",
    )
    candidate_dependency = _mapping(
        _mapping(
            _mapping(
                method_document["candidate_optimization"],
                "method.candidate_optimization",
            )["sobol_design"],
            "method.candidate_optimization.sobol_design",
        )["dependency"],
        "method.candidate_optimization.sobol_design.dependency",
    )
    files[
        "method.candidate_qmc_python_dependency_manifest"
    ] = _verified_digest(
        root,
        candidate_dependency["python_manifest_path"],
        candidate_dependency["python_manifest_sha256"],
        "candidate QMC Python dependency manifest",
    )
    files["method.candidate_qmc_ros_dependency_manifest"] = _verified_digest(
        root,
        candidate_dependency["ros_manifest_path"],
        candidate_dependency["ros_manifest_sha256"],
        "candidate QMC ROS dependency manifest",
    )
    if declared_method_path != method_path:
        files["config.declared_shared_method"] = _external_or_repository_digest(
            declared_method_path, root
        )
    for pad in hand_contract.pads:
        files[f"hand.{pad.name}.finite_footprint"] = FrozenFileDigest(
            path=pad.mesh.repository_relative_path,
            sha256=pad.mesh.sha256,
            byte_count=pad.mesh.byte_count,
            declared_sha256=pad.mesh.sha256,
        )

    objects = _mapping(object_document["objects"], "objects")
    current = _mapping(objects[_DEVELOPMENT_OBJECT], f"objects.{_DEVELOPMENT_OBJECT}")
    current_geometry = _mapping(current["planning_geometry"], "current planning geometry")
    files[f"objects.{_DEVELOPMENT_OBJECT}.planning_geometry"] = _verified_digest(
        root, current_geometry["path"], current_geometry["sha256"], "current planning geometry"
    )
    files[f"objects.{_DEVELOPMENT_OBJECT}.planning_geometry_manifest"] = _verified_digest(
        root, current_geometry["manifest"], current_geometry["manifest_sha256"], "current planning geometry manifest"
    )
    files[f"objects.{_DEVELOPMENT_OBJECT}.source_stage"] = _verified_digest(
        root, current_geometry["source_stage"], current_geometry["source_stage_sha256"], "current source stage"
    )
    current_physical = _mapping(current["physical_properties"], "current physical properties")
    files[f"objects.{_DEVELOPMENT_OBJECT}.physical_source"] = _verified_digest(
        root, current_physical["source_contract"], current_physical["source_contract_sha256"], "current physical source"
    )
    current_material = _mapping(current["contact_material_uncertainty"], "current material")
    files[f"objects.{_DEVELOPMENT_OBJECT}.contact_material_source"] = _verified_digest(
        root,
        current_material["source"],
        current_material["source_sha256"],
        "current contact material source",
    )

    transfer = _mapping(objects[_TRANSFER_OBJECT], f"objects.{_TRANSFER_OBJECT}")
    transfer_geometry = _mapping(transfer["planning_geometry"], "transfer planning geometry")
    files[f"objects.{_TRANSFER_OBJECT}.planning_geometry"] = _verified_digest(
        root, transfer_geometry["path"], transfer_geometry["sha256"], "transfer planning geometry"
    )
    original = _mapping(transfer["original_cad"], "transfer original CAD")
    files[f"objects.{_TRANSFER_OBJECT}.original_cad"] = _verified_digest(
        root, original["path"], original["sha256"], "transfer original CAD"
    )
    files[f"objects.{_TRANSFER_OBJECT}.geometry_audit"] = _verified_digest(
        root, original["geometry_audit"], original["geometry_audit_sha256"], "transfer geometry audit"
    )
    transfer_physical = _mapping(transfer["physical_properties"], "transfer physical properties")
    files[f"objects.{_TRANSFER_OBJECT}.mass_source"] = _verified_digest(
        root, transfer_physical["mass_source"], transfer_physical["mass_source_sha256"], "transfer mass source"
    )
    transfer_material = _mapping(
        transfer["contact_material_uncertainty"], "transfer material"
    )
    files[f"objects.{_TRANSFER_OBJECT}.contact_material_source"] = _verified_digest(
        root,
        transfer_material["source"],
        transfer_material["source_sha256"],
        "transfer contact material source",
    )

    files = dict(sorted(files.items()))
    geometry_hashes = _derived_geometry_hashes(
        method_document, hand_document, object_document, files
    )
    asset_inputs = {
        key: {
            "path": digest.path,
            "fragment": digest.fragment,
            "sha256": digest.sha256,
            "byte_count": digest.byte_count,
            "declared_sha256": digest.declared_sha256,
        }
        for key, digest in files.items()
        if not key.startswith("config.")
    }
    blockers = _CURRENT_PREREGISTRATION_BLOCKERS
    manifest = {
        "schema_version": _FROZEN_SCHEMA,
        "status": "FROZEN" if not blockers else "NOT_FREEZE_ELIGIBLE",
        "study": {
            "study_id": object_document["study_id"],
            "method": method_document["method"],
            "claim_scope": object_document["claim_scope"],
            "development_object": _DEVELOPMENT_OBJECT,
            "transfer_object": _TRANSFER_OBJECT,
            "transfer_object_role": _TRANSFER_ROLE,
        },
        "claim_boundaries": {
            "simulation_only": True,
            "zero_object_tuning": True,
            "transfer_geometry_seen_before_method_freeze": True,
            "prospective_double_blind_claim_allowed": False,
            "hardware_authorized": False,
            "online_truth_firewall_closed": True,
        },
        "dynamic_eligibility": {
            _DEVELOPMENT_OBJECT: False,
            _TRANSFER_OBJECT: False,
        },
        "protocol_interpretation": {
            "numerical_tolerances": _NUMERICAL_ROLE,
            "ray_closure_subdivision_budget": (
                "PRE_REGISTERED_COMPUTE_BUDGET_NOT_PHYSICAL_ACCEPTANCE_THRESHOLD"
            ),
            "ray_closure_budget_convergence": (
                "COMPUTE_CONVERGENCE_AUDIT_NOT_PHYSICAL_PASS_GATE"
            ),
            "ray_closure_adjacent_budget_result_stability_report_required": True,
            "ray_closure_physical_acceptance_gate": False,
            "interval_backend_decimal_precision": 80,
            "interval_backend_decimal_precision_role": (
                "NUMERICAL_INTERVAL_ARITHMETIC_PRECISION_NOT_PHYSICAL_"
                "ACCEPTANCE_THRESHOLD"
            ),
            "interval_backend_precision_convergence": (
                "NUMERICAL_PRECISION_CONVERGENCE_AUDIT_NOT_PHYSICAL_PASS_GATE"
            ),
            "interval_backend_root_bisection_budget": (
                "PRE_REGISTERED_COMPUTE_BUDGET_NOT_PHYSICAL_ACCEPTANCE_THRESHOLD"
            ),
            "interval_backend_root_budget_exhaustion": (
                "FAIL_CLOSED_UNRESOLVED"
            ),
            "interval_backend_physical_acceptance_gate": False,
            "top_level_candidate_generator_method_id": (
                TOP_LEVEL_CANDIDATE_GENERATOR_METHOD_ID
            ),
            "top_level_candidate_design_role": (
                "PRE_REGISTERED_FINITE_QMC_COMPUTE_DESIGN_NOT_PHYSICAL_"
                "ACCEPTANCE_THRESHOLD"
            ),
            "top_level_candidate_main_total_attempt_budget": (
                MAIN_TOTAL_ATTEMPT_BUDGET
            ),
            "top_level_candidate_allowed_total_attempt_budgets": list(
                ALLOWED_TOTAL_ATTEMPT_BUDGETS
            ),
            "top_level_candidate_schedule_rule": SCHEDULE_RULE,
            "top_level_candidate_no_replacement": True,
            "top_level_candidate_failure_and_duplicate_consume_attempt": True,
            "top_level_candidate_exact_deduplication_rule": DEDUPLICATION_RULE,
            "top_level_candidate_scipy_version": _SCIPY_VERSION,
            "top_level_candidate_realized_design_hashes_bound": True,
            "top_level_candidate_local_refinement_execution_status": (
                "DISABLED_FOR_V1"
            ),
            "top_level_candidate_output_claim": TOP_LEVEL_OUTPUT_CLAIM,
            "top_level_candidate_accepted_output_channels": [
                EXACT_CANDIDATE_OUTPUT_CHANNEL,
                CONTACT_RANGE_POLICY_OUTPUT_CHANNEL,
            ],
            "top_level_candidate_and_policy_mutually_exclusive": True,
            "top_level_display_only_proposal_formal_eligible": False,
            "top_level_contact_range_policy_downstream_status": (
                CONTACT_RANGE_POLICY_DOWNSTREAM_STATUS
            ),
            "contact_range_policy_collision_implementation_type_id": (
                _implementation_type_id(
                    ContactRangePolicyCollisionCertificate
                )
            ),
            "contact_range_policy_collision_method_id": (
                CONTACT_RANGE_POLICY_METHOD_ID
            ),
            "contact_range_policy_collision_independent_two_phase_method_id": (
                INDEPENDENT_MOVING_PAIR_METHOD_ID
            ),
            "contact_range_policy_collision_phase_domain_rule": (
                "COMPLETE_CARTESIAN_PRODUCT_OF_BOTH_REGISTERED_"
                "PHASE_INTERVALS"
            ),
            "contact_range_policy_collision_display_approximation_used": False,
            "contact_range_policy_collision_finite_sampling_allowed": False,
            "contact_range_policy_collision_current_state": (
                "NOT_CERTIFIABLE"
            ),
            "contact_range_policy_collision_checkable_scope": (
                "HAND_LINK_NONPAD_OBJECT_AND_SELF_PAIR_CONTACT_RANGE_"
                "CLOSURE_ONLY"
            ),
            "contact_range_policy_collision_mandatory_blockers": list(
                CONTACT_RANGE_POLICY_MANDATORY_BLOCKERS
            ),
            "contact_range_policy_collision_formal_selection_allowed": False,
            "contact_range_policy_collision_isaac_launch_allowed": False,
            "legacy_grasp_optimizer_formal_eligible": False,
            "post_generation_ranking_binding_status": (
                "BOUND_TO_PRODUCTION_POST_GENERATION_RANK_ONLY_PIPELINE"
            ),
            "post_generation_ranking_implementation_type_id": (
                _implementation_type_id(PostGenerationRankOnlyPipeline)
            ),
            "post_generation_ranking_method_id": (
                POST_GENERATION_RANKER_METHOD_ID
            ),
            "post_generation_ranking_role": (
                "RANK_ONLY_WITHOUT_CANDIDATE_MUTATION_OR_REPLACEMENT"
            ),
            "post_generation_ranking_common_scenario_method_id": (
                POST_GENERATION_SCENARIO_METHOD_ID
            ),
            "post_generation_ranking_common_scenario_dimension": (
                POST_GENERATION_SCENARIO_DIMENSION
            ),
            "post_generation_ranking_common_scenario_count": (
                POST_GENERATION_SCENARIO_COUNT
            ),
            "post_generation_ranking_common_scenario_sobol_seed": (
                POST_GENERATION_SCENARIO_SOBOL_SEED
            ),
            "post_generation_ranking_common_scenario_design_sha256": (
                POST_GENERATION_SCENARIO_DESIGN_SHA256
            ),
            "post_generation_ranking_common_scenarios_required": True,
            "post_generation_ranking_failure_retention_required": True,
            "post_generation_ranking_failed_candidate_drop_allowed": False,
            "post_generation_policy_aware_ranking_guard": (
                POLICY_AWARE_RANKING_GUARD
            ),
            "post_generation_policy_collision_invocations_before_support": 0,
            "post_generation_policy_wrench_invocations_before_support": 0,
            "post_generation_ranking_formal_selection_status": (
                _FORMAL_EMPTY_STATUS
            ),
            "post_generation_ranking_formal_selection_allowed": False,
            "post_generation_ranking_required_collision_scope": (
                COMPLETE_CLEARANCE_SCOPE
            ),
            "post_generation_ranking_current_uncertainty_scope": (
                FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
            ),
            "uncertainty_nested_convergence_scenario_count": 256,
            "uncertainty_nested_convergence_role": (
                _NESTED_CONVERGENCE_ROLE
            ),
            "linear_program_physical_acceptance_gate": False,
            "force_qp_physical_acceptance_gate": False,
            "numerical_convergence_physical_acceptance_gate": False,
        },
        "preregistration_blockers": list(blockers),
        "source_documents": {
            "shared_method": method_document,
            "hand_contract": hand_document,
            "object_contract": object_document,
        },
        "consumed_referenced_inputs": asset_inputs,
        "derived_geometry_sha256": geometry_hashes,
    }
    encoded = _canonical_bytes(manifest)
    frozen_files = MappingProxyType(files)
    input_hashes = MappingProxyType(
        {key: value.sha256 for key, value in files.items()}
    )
    frozen_geometry = MappingProxyType(geometry_hashes)
    return StudyContractAudit(
        input_files=frozen_files,
        input_file_sha256=input_hashes,
        derived_geometry_sha256=frozen_geometry,
        preregistration_blockers=blockers,
        canonical_manifest=_deep_freeze(manifest),
        canonical_json_bytes=encoded,
        canonical_sha256=_sha256_bytes(encoded),
    )


def build_frozen_study_contract(
    shared_method_path: str | Path,
    hand_contract_path: str | Path,
    object_contract_path: str | Path,
    *,
    repository_root: str | Path,
) -> FrozenStudyContract:
    """Build the formal freeze or fail closed on any unresolved binding."""

    audit = audit_study_contract(
        shared_method_path,
        hand_contract_path,
        object_contract_path,
        repository_root=repository_root,
    )
    if audit.preregistration_blockers:
        raise StudyFreezeIncompleteError(audit.preregistration_blockers)
    study = audit.canonical_manifest["study"]
    boundaries = audit.canonical_manifest["claim_boundaries"]
    return FrozenStudyContract(
        schema_version=_FROZEN_SCHEMA,
        study_id=study["study_id"],
        method=study["method"],
        claim_scope=study["claim_scope"],
        development_object=study["development_object"],
        transfer_object=study["transfer_object"],
        transfer_object_role=study["transfer_object_role"],
        transfer_geometry_seen_before_method_freeze=boundaries[
            "transfer_geometry_seen_before_method_freeze"
        ],
        prospective_double_blind_claim_allowed=boundaries[
            "prospective_double_blind_claim_allowed"
        ],
        hardware_authorized=boundaries["hardware_authorized"],
        dynamic_eligibility=audit.canonical_manifest["dynamic_eligibility"],
        numerical_tolerance_role=audit.canonical_manifest[
            "protocol_interpretation"
        ]["numerical_tolerances"],
        input_files=audit.input_files,
        input_file_sha256=audit.input_file_sha256,
        derived_geometry_sha256=audit.derived_geometry_sha256,
        canonical_manifest=audit.canonical_manifest,
        canonical_json_bytes=audit.canonical_json_bytes,
        canonical_sha256=audit.canonical_sha256,
    )


load_frozen_study_contract = build_frozen_study_contract


__all__ = [
    "FrozenFileDigest",
    "FrozenStudyContract",
    "StudyContractAudit",
    "StudyContractError",
    "StudyFreezeIncompleteError",
    "audit_study_contract",
    "build_frozen_study_contract",
    "load_frozen_study_contract",
]
