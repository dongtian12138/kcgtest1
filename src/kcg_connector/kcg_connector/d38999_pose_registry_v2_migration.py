"""Pure-CPU, fail-closed D38999 pose-registry v2 migration audit.

The active pose contract remains v1.  This module validates an independent,
disabled migration design and hash-binds the evidence that makes the current
``keyed_order_1`` declarations unsafe for D38999 vision.  It deliberately has
no Isaac, ROS, GPU, FoundationPose-runtime, or E2E imports.

The v2 design represents a pose as an equivalence class under a finite group
acting in the object frame.  A convenient quaternion representative may be
published for visualization, but it is never evidence of a unique key yaw.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "kcg_d38999_pose_registry_v2_migration_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "d38999_pose_registry_v2_migration_v1.yaml"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_INPUTS = {
    "active_pose_registry_v1",
    "proxy_definition",
    "loose_body_obj",
    "fixed_receptacle_obj",
    "coupling_nut_obj",
}
_EXPECTED_AFFECTED_MODELS = {
    "d38999_26kj61sn_proxy_v1",
    "d38999_20kj61pn_proxy_v1",
}
_EXPECTED_BOUNDARIES = {
    "active_registry_modified",
    "active_e2e_modified",
    "active_asset_modified",
    "gpu_execution_performed",
    "model_or_geometry_download_performed",
    "v2_runtime_activated",
    "pose_pair_publication_authorized",
    "robot_control_authorized",
    "unique_key_geometry_claimed",
    "unique_key_yaw_claimed",
    "real_assembly_success_claimed",
}


@dataclass(frozen=True)
class BoundInput:
    """One repository-relative, content-addressed migration input."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class AxialSymmetry:
    """Finite rotation group around local object-frame +Z."""

    object_id: str
    geometry_input: str
    order: int
    required_verified_orders: tuple[int, ...]
    unique_key_geometry_present: bool


@dataclass(frozen=True)
class ProposedModel:
    """One endpoint registration proposed for the side-by-side v2 schema."""

    model_id: str
    role: str
    object_frame_id: str
    geometry_input: str
    compatible_model_ids: tuple[str, ...]
    symmetry: AxialSymmetry


@dataclass(frozen=True)
class MigrationContract:
    """Validated migration design; loading it cannot activate any runtime."""

    config_path: Path
    schema_version: str
    enabled: bool
    status: str
    inputs: dict[str, BoundInput]
    active_declared_symmetry_class: str
    affected_model_ids: tuple[str, ...]
    coordinate_decimal_places: int
    observed_symmetry: dict[str, AxialSymmetry]
    proposed_models: dict[str, ProposedModel]
    proposed_component: AxialSymmetry
    control_requirements: dict[str, tuple[str, ...]]
    migration_phases: dict[str, bool]
    boundaries: dict[str, bool]
    document: dict[str, Any]


@dataclass(frozen=True)
class OrientationModuloSymmetry:
    """Geodesic orientation error after choosing the best group element."""

    error_rad: float
    selected_symmetry_index: int
    equivalent_yaw_offset_rad: float


@dataclass(frozen=True)
class StageAuthorization:
    """One stage-scoped gate result, never a global control assertion."""

    stage: str
    all_required_gates_passed: bool
    migration_activated: bool
    would_authorize_in_future_active_contract: bool
    authorized: bool
    missing_gates: tuple[str, ...]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{label} keys differ; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty unpadded text")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha(value: Any, label: str) -> str:
    result = _text(value, label)
    if not _SHA256.fullmatch(result) or result == "0" * 64:
        raise ValueError(f"{label} must be a non-zero lowercase SHA-256")
    return result


def _relative_path(value: Any, label: str) -> Path:
    result = Path(_text(value, label))
    if result.is_absolute() or ".." in result.parts:
        raise ValueError(f"{label} must be repository-relative")
    return result


def _unique_text_list(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} must be non-empty and unique")
    return result


def _positive_int_list(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    result = tuple(_positive_int(item, f"{label}[]") for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} must be non-empty and unique")
    return result


def _axis(value: Any, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be object-frame +Z")
    result = tuple(_finite(item, f"{label}[]") for item in value)
    if result != (0.0, 0.0, 1.0):
        raise ValueError(f"{label} must be object-frame +Z")
    return result  # type: ignore[return-value]


def _parse_input(value: Any, label: str) -> BoundInput:
    document = _mapping(value, label)
    _exact(document, {"path", "sha256"}, label)
    return BoundInput(
        path=_relative_path(document["path"], f"{label}.path"),
        sha256=_sha(document["sha256"], f"{label}.sha256"),
    )


def _parse_observed_symmetry(
    object_id: str, value: Any
) -> AxialSymmetry:
    label = f"current_mismatch.observed_objects.{object_id}"
    document = _mapping(value, label)
    _exact(
        document,
        {
            "mesh_input",
            "minimum_rotational_symmetry_order",
            "required_verified_orders",
            "unique_key_geometry_present",
        },
        label,
    )
    order = _positive_int(
        document["minimum_rotational_symmetry_order"], f"{label}.order"
    )
    verified = _positive_int_list(
        document["required_verified_orders"], f"{label}.orders"
    )
    if order not in verified:
        raise ValueError(f"{label} must verify its declared minimum order")
    keyed = _boolean(
        document["unique_key_geometry_present"], f"{label}.unique key"
    )
    if keyed:
        raise ValueError("current proxy cannot claim unique key geometry")
    return AxialSymmetry(
        object_id=object_id,
        geometry_input=_text(document["mesh_input"], f"{label}.mesh"),
        order=order,
        required_verified_orders=verified,
        unique_key_geometry_present=keyed,
    )


def _parse_symmetry(
    object_id: str, geometry_input: str, value: Any, *, expected_order: int
) -> AxialSymmetry:
    label = f"proposed_v2.symmetry.{object_id}"
    document = _mapping(value, label)
    _exact(
        document,
        {
            "kind",
            "axis_in_object_frame_xyz",
            "order",
            "equivalent_yaw_period_rad",
            "unique_key_geometry_present",
            "pose_semantics",
        },
        label,
    )
    if document["kind"] != "discrete_axial":
        raise ValueError(f"{label}.kind must be discrete_axial")
    _axis(document["axis_in_object_frame_xyz"], f"{label}.axis")
    order = _positive_int(document["order"], f"{label}.order")
    if order != expected_order:
        raise ValueError(f"{label}.order differs from audited geometry")
    period = _finite(
        document["equivalent_yaw_period_rad"], f"{label}.period"
    )
    if abs(period - 2.0 * math.pi / order) > 1.0e-12:
        raise ValueError(f"{label}.period is inconsistent with order")
    keyed = _boolean(
        document["unique_key_geometry_present"], f"{label}.unique key"
    )
    if keyed or document["pose_semantics"] != "equivalence_class":
        raise ValueError(f"{label} must remain a keyless equivalence class")
    return AxialSymmetry(
        object_id=object_id,
        geometry_input=geometry_input,
        order=order,
        required_verified_orders=(order,),
        unique_key_geometry_present=keyed,
    )


def _parse_proposed_model(value: Any) -> ProposedModel:
    document = _mapping(value, "proposed_v2.model_registry[]")
    _exact(
        document,
        {
            "model_id",
            "role",
            "object_frame_id",
            "geometry_input",
            "compatible_model_ids",
            "symmetry",
        },
        "proposed_v2.model_registry[]",
    )
    model_id = _text(document["model_id"], "model_id")
    role = _text(document["role"], f"{model_id}.role")
    expected_geometry = {
        "loose_plug": "loose_body_obj",
        "fixed_receptacle": "fixed_receptacle_obj",
    }
    if role not in expected_geometry:
        raise ValueError("proposed v2 endpoint role is unsupported")
    geometry = _text(document["geometry_input"], f"{model_id}.geometry")
    if geometry != expected_geometry[role]:
        raise ValueError("proposed v2 endpoint geometry differs")
    symmetry = _parse_symmetry(
        role, geometry, document["symmetry"], expected_order=2
    )
    return ProposedModel(
        model_id=model_id,
        role=role,
        object_frame_id=_text(
            document["object_frame_id"], f"{model_id}.object_frame_id"
        ),
        geometry_input=geometry,
        compatible_model_ids=_unique_text_list(
            document["compatible_model_ids"], f"{model_id}.mates"
        ),
        symmetry=symmetry,
    )


def _require_exact_list(
    value: Any, expected: set[str], label: str
) -> tuple[str, ...]:
    result = _unique_text_list(value, label)
    if set(result) != expected:
        raise ValueError(f"{label} differs from fail-closed requirements")
    return result


def load_pose_registry_v2_migration_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> MigrationContract:
    """Load and strictly validate the disabled migration design."""

    config_path = Path(path).expanduser().resolve()
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "document"
    )
    _exact(
        document,
        {
            "schema_version",
            "enabled",
            "status",
            "inputs",
            "current_mismatch",
            "proposed_v2",
            "key_geometry_upgrade",
            "foundationpose_evaluation",
            "control_authorization",
            "migration_phases",
            "boundaries",
        },
        "document",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    enabled = _boolean(document["enabled"], "enabled")
    if enabled:
        raise ValueError("migration contract must remain disabled")
    status = _text(document["status"], "status")
    if status != "audit_confirmed_v2_designed_not_activated":
        raise ValueError("migration status cannot imply activation")

    input_document = _mapping(document["inputs"], "inputs")
    _exact(input_document, _EXPECTED_INPUTS, "inputs")
    inputs = {
        name: _parse_input(input_document[name], f"inputs.{name}")
        for name in sorted(input_document)
    }

    mismatch = _mapping(document["current_mismatch"], "current_mismatch")
    _exact(
        mismatch,
        {
            "active_declared_symmetry_class",
            "affected_model_ids",
            "geometry_audit_method",
            "coordinate_decimal_places",
            "observed_objects",
            "consequences",
        },
        "current_mismatch",
    )
    if mismatch["active_declared_symmetry_class"] != "keyed_order_1":
        raise ValueError("audit must identify the active keyed_order_1 claim")
    affected = _unique_text_list(
        mismatch["affected_model_ids"], "current_mismatch.affected models"
    )
    if set(affected) != _EXPECTED_AFFECTED_MODELS:
        raise ValueError("affected D38999 model set differs")
    if mismatch["geometry_audit_method"] != (
        "quantized_obj_vertex_multiset_under_axial_rotation"
    ):
        raise ValueError("geometry audit method differs")
    decimal_places = _positive_int(
        mismatch["coordinate_decimal_places"], "coordinate decimal places"
    )
    if decimal_places != 7:
        raise ValueError("geometry audit precision must remain seven decimals")
    observed_document = _mapping(
        mismatch["observed_objects"], "current_mismatch.observed_objects"
    )
    _exact(
        observed_document,
        {"loose_plug", "fixed_receptacle", "coupling_nut"},
        "current_mismatch.observed_objects",
    )
    observed = {
        name: _parse_observed_symmetry(name, value)
        for name, value in observed_document.items()
    }
    if observed["loose_plug"].order != 2:
        raise ValueError("loose proxy must remain at least order-2")
    if observed["fixed_receptacle"].order != 2:
        raise ValueError("fixed proxy must remain at least order-2")
    if observed["coupling_nut"].order != 24:
        raise ValueError("coupling nut must remain order-24")
    consequences = _mapping(mismatch["consequences"], "consequences")
    _exact(
        consequences,
        {
            "v1_d38999_vision_pair_publication_safe",
            "v1_d38999_vision_control_safe",
            "polarization_label_counts_as_geometry_evidence",
        },
        "consequences",
    )
    if any(
        _boolean(value, f"consequences.{name}")
        for name, value in consequences.items()
    ):
        raise ValueError("v1 mismatch consequences must remain fail-closed")

    proposed = _mapping(document["proposed_v2"], "proposed_v2")
    _exact(
        proposed,
        {
            "contract_schema_version",
            "observation_schema_version",
            "symmetry_schema_version",
            "migration_mode",
            "v1_remains_runtime_default",
            "v2_runtime_registration_enabled",
            "group_action",
            "model_registry",
            "component_registry",
            "v1_bridge",
        },
        "proposed_v2",
    )
    expected_scalars = {
        "contract_schema_version": "kcg_connector_pose_contract_v2",
        "observation_schema_version": "kcg_connector_pose_observation_v2",
        "symmetry_schema_version": "kcg_connector_object_symmetry_v1",
        "migration_mode": "side_by_side_no_in_place_upgrade",
        "group_action": "right_multiply_object_frame_rotation",
    }
    for name, expected in expected_scalars.items():
        if proposed[name] != expected:
            raise ValueError(f"proposed_v2.{name} differs")
    if not _boolean(proposed["v1_remains_runtime_default"], "v1 default"):
        raise ValueError("v1 must remain runtime default during migration")
    if _boolean(proposed["v2_runtime_registration_enabled"], "v2 enabled"):
        raise ValueError("v2 runtime registration must remain disabled")
    model_values = proposed["model_registry"]
    if not isinstance(model_values, list) or len(model_values) != 2:
        raise ValueError("proposed v2 registry must contain two endpoints")
    parsed_models = tuple(_parse_proposed_model(item) for item in model_values)
    if {item.model_id for item in parsed_models} != _EXPECTED_AFFECTED_MODELS:
        raise ValueError("proposed v2 model IDs differ from audited geometry")
    models = {item.model_id: item for item in parsed_models}
    for model in parsed_models:
        if len(model.compatible_model_ids) != 1:
            raise ValueError("proposed endpoint must have exactly one mate")
        mate = models.get(model.compatible_model_ids[0])
        if mate is None or model.model_id not in mate.compatible_model_ids:
            raise ValueError("proposed v2 compatibility must be reciprocal")
        if mate.role == model.role:
            raise ValueError("proposed v2 compatible roles must differ")

    components = proposed["component_registry"]
    if not isinstance(components, list) or len(components) != 1:
        raise ValueError("proposed v2 component registry must contain the nut")
    component_document = _mapping(components[0], "component_registry[0]")
    _exact(
        component_document,
        {
            "component_id",
            "parent_model_id",
            "independent_rigid_body",
            "geometry_input",
            "symmetry",
        },
        "component_registry[0]",
    )
    if (
        component_document["component_id"]
        != "d38999_26kj61sn_coupling_nut_proxy_v1"
        or component_document["parent_model_id"]
        != "d38999_26kj61sn_proxy_v1"
        or component_document["geometry_input"] != "coupling_nut_obj"
        or not _boolean(
            component_document["independent_rigid_body"],
            "component independent body",
        )
    ):
        raise ValueError("proposed nut component identity differs")
    component = _parse_symmetry(
        "coupling_nut",
        "coupling_nut_obj",
        component_document["symmetry"],
        expected_order=24,
    )
    bridge = _mapping(proposed["v1_bridge"], "proposed_v2.v1_bridge")
    _exact(
        bridge,
        {
            "affected_d38999_observations_are_lossless",
            "allow_evaluation_only_diagnostic_bridge",
            "allow_pair_publication",
            "allow_control",
            "required_bridge_diagnostic",
        },
        "proposed_v2.v1_bridge",
    )
    if _boolean(
        bridge["affected_d38999_observations_are_lossless"],
        "bridge lossless",
    ):
        raise ValueError("v1-to-v2 D38999 bridge cannot claim lossless")
    if not _boolean(
        bridge["allow_evaluation_only_diagnostic_bridge"],
        "diagnostic bridge",
    ):
        raise ValueError("migration must retain an evaluation diagnostic")
    if _boolean(bridge["allow_pair_publication"], "bridge publication") or _boolean(
        bridge["allow_control"], "bridge control"
    ):
        raise ValueError("v1 bridge cannot publish or authorize control")

    _validate_key_upgrade(document["key_geometry_upgrade"])
    _validate_foundationpose_policy(document["foundationpose_evaluation"])
    control_requirements = _parse_control_policy(
        document["control_authorization"]
    )

    phase_values = document["migration_phases"]
    if not isinstance(phase_values, list) or len(phase_values) != 5:
        raise ValueError("migration must retain five ordered phases")
    phase_ids = (
        "phase_0_quarantine",
        "phase_1_side_by_side_schema",
        "phase_2_shadow_pose_evaluation",
        "phase_3_keyed_geometry_revision",
        "phase_4_stage_scoped_control",
    )
    phases: dict[str, bool] = {}
    for expected_id, raw in zip(phase_ids, phase_values):
        item = _mapping(raw, f"migration_phases.{expected_id}")
        _exact(item, {"id", "complete", "exit_criterion"}, f"phase {expected_id}")
        if item["id"] != expected_id:
            raise ValueError("migration phase order or ID differs")
        phases[expected_id] = _boolean(item["complete"], f"{expected_id}.complete")
        _text(item["exit_criterion"], f"{expected_id}.exit_criterion")
    if phases[phase_ids[0]] is not True or any(
        phases[name] for name in phase_ids[1:]
    ):
        raise ValueError("only migration quarantine phase may be complete")

    boundary_document = _mapping(document["boundaries"], "boundaries")
    _exact(boundary_document, _EXPECTED_BOUNDARIES, "boundaries")
    boundaries = {
        name: _boolean(value, f"boundaries.{name}")
        for name, value in boundary_document.items()
    }
    if any(boundaries.values()):
        raise ValueError("migration boundaries must all remain false")

    return MigrationContract(
        config_path=config_path,
        schema_version=SCHEMA_VERSION,
        enabled=enabled,
        status=status,
        inputs=inputs,
        active_declared_symmetry_class="keyed_order_1",
        affected_model_ids=affected,
        coordinate_decimal_places=decimal_places,
        observed_symmetry=observed,
        proposed_models=models,
        proposed_component=component,
        control_requirements=control_requirements,
        migration_phases=phases,
        boundaries=boundaries,
        document=dict(document),
    )


def _validate_key_upgrade(value: Any) -> None:
    label = "key_geometry_upgrade"
    document = _mapping(value, label)
    _exact(
        document,
        {
            "status",
            "selected_identity",
            "acceptable_primary_geometry_sources",
            "public_shell_outline_drawings_sufficient",
            "required_source_records",
            "required_key_records",
            "keyed_upgrade_model_ids",
            "reuse_v1_model_ids_for_changed_geometry_allowed",
            "manually_drawn_or_guessed_key_allowed",
        },
        label,
    )
    if document["status"] != "blocked_missing_hash_bound_key_geometry":
        raise ValueError("key geometry status must remain blocked")
    identity = _mapping(document["selected_identity"], f"{label}.identity")
    _exact(
        identity,
        {
            "loose_part_number",
            "fixed_part_number",
            "insert_arrangement",
            "polarization",
            "identity_label_is_geometry_evidence",
        },
        f"{label}.identity",
    )
    if (
        identity["loose_part_number"] != "D38999/26KJ61SN"
        or identity["fixed_part_number"] != "D38999/20KJ61PN"
        or identity["insert_arrangement"] != 61
        or identity["polarization"] != "N"
        or _boolean(
            identity["identity_label_is_geometry_evidence"],
            "identity geometry evidence",
        )
    ):
        raise ValueError("selected identity or evidence boundary differs")
    _require_exact_list(
        document["acceptable_primary_geometry_sources"],
        {
            "licensed_vendor_cad_for_exact_orderable_part_numbers",
            "calibrated_metrology_or_scan_of_traceable_physical_mates",
        },
        f"{label}.acceptable sources",
    )
    if _boolean(
        document["public_shell_outline_drawings_sufficient"],
        "public drawings",
    ):
        raise ValueError("public shell outline drawings are insufficient")
    source_records = {
        "source_or_specification_identifier",
        "license_and_redistribution_boundary",
        "exact_part_number_and_polarization_traceability",
        "units_scale_and_coordinate_frame",
        "original_file_sha256",
        "derived_mesh_sha256_and_reproducible_conversion",
        "dimensional_or_scan_uncertainty",
    }
    _require_exact_list(
        document["required_source_records"], source_records, "source records"
    )
    key_records = {
        "asymmetric_plug_polarization_feature_geometry",
        "matching_receptacle_polarization_feature_geometry",
        "plug_object_frame_key_reference_direction",
        "receptacle_object_frame_key_reference_direction",
        "mating_key_angle_and_tolerance",
        "calibrated_object_to_assembly_transforms",
    }
    _require_exact_list(
        document["required_key_records"], key_records, "key records"
    )
    model_ids = _mapping(document["keyed_upgrade_model_ids"], "keyed model IDs")
    _exact(model_ids, {"loose_plug", "fixed_receptacle"}, "keyed model IDs")
    new_ids = {_text(value, "keyed model ID") for value in model_ids.values()}
    if new_ids & _EXPECTED_AFFECTED_MODELS or len(new_ids) != 2:
        raise ValueError("keyed geometry must use distinct new model IDs")
    if _boolean(
        document["reuse_v1_model_ids_for_changed_geometry_allowed"],
        "ID reuse",
    ):
        raise ValueError("changed keyed geometry cannot reuse v1 model IDs")
    if _boolean(
        document["manually_drawn_or_guessed_key_allowed"], "guessed key"
    ):
        raise ValueError("key geometry cannot be guessed")


def _validate_foundationpose_policy(value: Any) -> None:
    label = "foundationpose_evaluation"
    document = _mapping(value, label)
    _exact(
        document,
        {
            "output_interpretation",
            "endpoint_metric",
            "group_action",
            "translation_metric",
            "axis_metric",
            "raw_yaw_error_reportable_as_keyed_accuracy",
            "canonical_representative_may_authorize_control",
            "pair_hypothesis_policy",
            "temporal_policy",
            "coupling_nut_policy",
            "keyed_yaw_field_for_current_proxy",
            "required_shadow_evidence",
        },
        label,
    )
    expected = {
        "output_interpretation": "pose_equivalence_class_not_unique_keyed_pose",
        "endpoint_metric": "minimum_se3_orientation_error_over_declared_symmetry_group",
        "group_action": "estimated_parent_T_object_right_multiply_object_T_symmetry",
        "translation_metric": "euclidean_3d",
        "axis_metric": "unsigned_object_axis_angle",
        "pair_hypothesis_policy": "retain_all_loose_fixed_symmetry_combinations",
        "temporal_policy": (
            "preserve_hypothesis_identity_and_reject_unexplained_branch_flips"
        ),
        "coupling_nut_policy": "evaluate_separately_modulo_order_24",
    }
    for name, expected_value in expected.items():
        if document[name] != expected_value:
            raise ValueError(f"{label}.{name} differs")
    if _boolean(
        document["raw_yaw_error_reportable_as_keyed_accuracy"], "raw yaw"
    ):
        raise ValueError("raw yaw cannot be reported as keyed accuracy")
    if _boolean(
        document["canonical_representative_may_authorize_control"],
        "canonical pose",
    ):
        raise ValueError("a canonical representative cannot authorize control")
    if document["keyed_yaw_field_for_current_proxy"] is not None:
        raise ValueError("current proxy keyed yaw must be null")
    _require_exact_list(
        document["required_shadow_evidence"],
        {
            "withheld_truth_not_used_by_candidate_generation",
            "all_required_randomized_anchor_poses",
            "translation_axis_and_modulo_symmetry_orientation_metrics",
            "covariance_or_hypothesis_confidence_calibration",
            "occlusion_and_out_of_distribution_rejection",
        },
        f"{label}.required_shadow_evidence",
    )


def _parse_control_policy(value: Any) -> dict[str, tuple[str, ...]]:
    label = "control_authorization"
    document = _mapping(value, label)
    _exact(
        document,
        {
            "current_authorized",
            "evaluation_or_preflight_pair_publication",
            "symmetry_invariant_pick_only",
            "keyed_assembly_full_workflow",
            "force_guided_key_search_alternative_covered_by_this_contract",
        },
        label,
    )
    if _boolean(document["current_authorized"], f"{label}.current"):
        raise ValueError("current control must remain unauthorized")
    if _boolean(
        document["force_guided_key_search_alternative_covered_by_this_contract"],
        f"{label}.force guided alternative",
    ):
        raise ValueError("force-guided key search is outside this contract")
    expected = {
        "evaluation_or_preflight_pair_publication": {
            "content_addressed_v2_registry_loaded_by_v2_parser",
            "both_endpoint_outputs_publish_equivalence_classes",
            "modulo_symmetry_accuracy_passes_withheld_truth_thresholds",
            "timing_calibration_confidence_and_ood_gates_pass",
        },
        "symmetry_invariant_pick_only": {
            "every_grasp_and_collision_result_valid_for_every_symmetry_hypothesis",
            "no_unique_keyed_yaw_claim",
            "stage_scoped_controller_cannot_enter_insertion_or_twist",
            "repeated_closed_loop_pick_regression_passes",
        },
        "keyed_assembly_full_workflow": {
            "hash_bound_keyed_geometry_uses_new_model_ids",
            "key_feature_observed_or_fixed_mate_key_frame_independently_calibrated",
            "all_symmetry_equivalent_false_hypotheses_rejected",
            "relative_key_yaw_accuracy_passes_withheld_randomized_trials",
            "qualified_object_to_grasp_and_object_to_assembly_transforms",
            "collision_force_torque_and_repeatability_gates_pass",
            "pose_provider_control_purpose_has_no_truth_pose_components",
        },
    }
    result = {}
    possible_fields = {
        "evaluation_or_preflight_pair_publication": (
            "possible_after_v2_registry_activation"
        ),
        "symmetry_invariant_pick_only": "possible_in_future",
        "keyed_assembly_full_workflow": "possible_in_future",
    }
    for stage, required in expected.items():
        stage_document = _mapping(document[stage], f"{label}.{stage}")
        possible_name = possible_fields[stage]
        _exact(
            stage_document,
            {possible_name, "required_gates"},
            f"{label}.{stage}",
        )
        if not _boolean(
            stage_document[possible_name], f"{label}.{stage}.possible"
        ):
            raise ValueError(f"{stage} future path must remain explicit")
        result[stage] = _require_exact_list(
            stage_document["required_gates"], required, f"{label}.{stage}.gates"
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bound_input(
    repository: Path, item: BoundInput, label: str
) -> Path:
    root = repository.expanduser().resolve()
    path = (root / item.path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} resolves outside repository")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {item.path}")
    actual = _sha256_file(path)
    if actual != item.sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={item.sha256}, actual={actual}"
        )
    return path


def audit_obj_vertex_symmetry(
    path: str | Path,
    *,
    orders: Sequence[int],
    decimal_places: int = 7,
) -> dict[int, bool]:
    """Check vertex-multiset invariance for selected local +Z rotations.

    This does not pretend to prove manufacturing symmetry.  It is a
    deterministic audit of the exact simplified OBJ consumed by the current
    FoundationPose preparation path.
    """

    mesh_path = Path(path)
    vertices: list[tuple[float, float, float]] = []
    face_count = 0
    for line_number, raw in enumerate(
        mesh_path.read_text(encoding="ascii").splitlines(), start=1
    ):
        fields = raw.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            if len(fields) != 4:
                raise ValueError(f"invalid OBJ vertex at line {line_number}")
            vertex = tuple(float(value) for value in fields[1:])
            if not all(math.isfinite(value) for value in vertex):
                raise ValueError(f"non-finite OBJ vertex at line {line_number}")
            vertices.append(vertex)  # type: ignore[arg-type]
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError(
                    f"OBJ face is not triangular at line {line_number}"
                )
            face_count += 1
    if not vertices or not face_count:
        raise ValueError("OBJ must contain vertices and triangular faces")
    precision = _positive_int(decimal_places, "decimal_places")

    def quantized(point: Sequence[float]) -> tuple[float, float, float]:
        return tuple(  # type: ignore[return-value]
            round(float(value), precision) for value in point
        )

    baseline = Counter(quantized(vertex) for vertex in vertices)
    result: dict[int, bool] = {}
    for raw_order in orders:
        order = _positive_int(raw_order, "symmetry order")
        angle = 2.0 * math.pi / order
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotated = Counter(
            quantized(
                (
                    cosine * x_value - sine * y_value,
                    sine * x_value + cosine * y_value,
                    z_value,
                )
            )
            for x_value, y_value, z_value in vertices
        )
        result[order] = rotated == baseline
    return result


def _quaternion(value: Sequence[Real], label: str) -> tuple[float, ...]:
    if len(value) != 4:
        raise ValueError(f"{label} must contain four values")
    result = tuple(_finite(item, f"{label}[]") for item in value)
    norm = math.sqrt(sum(item * item for item in result))
    if norm == 0.0 or abs(norm - 1.0) > 1.0e-6:
        raise ValueError(f"{label} must be a unit quaternion")
    return tuple(item / norm for item in result)


def _multiply_quaternion(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def orientation_error_modulo_axial_symmetry(
    estimated_quaternion_xyzw: Sequence[Real],
    truth_quaternion_xyzw: Sequence[Real],
    *,
    symmetry_order: int,
) -> OrientationModuloSymmetry:
    """Return ``min_k angle(q_truth^-1 * q_est * Rz(2*pi*k/n))``."""

    estimated = _quaternion(estimated_quaternion_xyzw, "estimated quaternion")
    truth = _quaternion(truth_quaternion_xyzw, "truth quaternion")
    order = _positive_int(symmetry_order, "symmetry_order")
    truth_inverse = (-truth[0], -truth[1], -truth[2], truth[3])
    candidates = []
    for index in range(order):
        yaw = 2.0 * math.pi * index / order
        symmetry = (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        equivalent = _multiply_quaternion(estimated, symmetry)
        relative = _multiply_quaternion(truth_inverse, equivalent)
        error = 2.0 * math.acos(min(1.0, max(-1.0, abs(relative[3]))))
        candidates.append((error, index, yaw))
    error, index, yaw = min(candidates, key=lambda item: (item[0], item[1]))
    return OrientationModuloSymmetry(
        error_rad=error,
        selected_symmetry_index=index,
        equivalent_yaw_offset_rad=yaw,
    )


def relative_yaw_hypotheses(
    loose_symmetry_order: int, fixed_symmetry_order: int
) -> tuple[float, ...]:
    """Enumerate distinct relative-yaw branches for an endpoint pair."""

    loose_order = _positive_int(loose_symmetry_order, "loose order")
    fixed_order = _positive_int(fixed_symmetry_order, "fixed order")
    quantized = {
        round(
            (
                2.0 * math.pi * loose_index / loose_order
                - 2.0 * math.pi * fixed_index / fixed_order
            )
            % (2.0 * math.pi),
            14,
        )
        for loose_index in range(loose_order)
        for fixed_index in range(fixed_order)
    }
    return tuple(sorted(quantized))


def evaluate_stage_authorization(
    contract: MigrationContract,
    *,
    stage: str,
    passed_gates: Sequence[str],
    migration_activated: bool,
) -> StageAuthorization:
    """Evaluate a hypothetical v2 stage without changing runtime state."""

    if stage not in contract.control_requirements:
        raise ValueError(f"unsupported authorization stage: {stage!r}")
    if isinstance(passed_gates, (str, bytes)) or not isinstance(
        passed_gates, Sequence
    ):
        raise ValueError("passed_gates must be a sequence")
    passed_items = tuple(
        _text(item, "passed_gates[]") for item in passed_gates
    )
    if len(passed_items) != len(set(passed_items)):
        raise ValueError("passed_gates must be unique")
    passed = set(passed_items)
    required = contract.control_requirements[stage]
    unknown = passed - set(required)
    if unknown:
        raise ValueError(f"passed_gates contains unknown gates: {sorted(unknown)}")
    missing = tuple(item for item in required if item not in passed)
    active = _boolean(migration_activated, "migration_activated")
    future_ready = active and not missing
    return StageAuthorization(
        stage=stage,
        all_required_gates_passed=not missing,
        migration_activated=active,
        would_authorize_in_future_active_contract=future_ready,
        # This migration artifact is intentionally and immutably disabled.
        # A future production v2 contract needs its own activation boundary.
        authorized=contract.enabled and future_ready,
        missing_gates=missing,
    )


def evaluate_pose_registry_v2_migration(
    contract: MigrationContract,
    repository: str | Path,
) -> dict[str, Any]:
    """Verify bound evidence and return a JSON-safe, non-activating report."""

    root = Path(repository).expanduser().resolve()
    resolved = {
        name: _resolve_bound_input(root, item, f"inputs.{name}")
        for name, item in contract.inputs.items()
    }
    active = _mapping(
        yaml.safe_load(
            resolved["active_pose_registry_v1"].read_text(encoding="utf-8")
        ),
        "active pose registry",
    )
    registrations = active.get("model_registry")
    if not isinstance(registrations, list):
        raise ValueError("active pose registry has no model_registry list")
    active_classes = {
        item.get("model_id"): item.get("symmetry_class")
        for item in registrations
        if isinstance(item, Mapping)
        and item.get("model_id") in _EXPECTED_AFFECTED_MODELS
    }
    if set(active_classes) != _EXPECTED_AFFECTED_MODELS or any(
        value != contract.active_declared_symmetry_class
        for value in active_classes.values()
    ):
        raise ValueError("active D38999 symmetry mismatch no longer matches audit")

    proxy = _mapping(
        yaml.safe_load(resolved["proxy_definition"].read_text(encoding="utf-8")),
        "proxy definition",
    )
    identity = _mapping(proxy.get("identity"), "proxy identity")
    if (
        identity.get("loose_part_number") != "D38999/26KJ61SN"
        or identity.get("fixed_part_number") != "D38999/20KJ61PN"
        or identity.get("polarization") != "N"
    ):
        raise ValueError("proxy identity differs from migration design")

    geometry: dict[str, Any] = {}
    for object_id, spec in contract.observed_symmetry.items():
        path = resolved[spec.geometry_input]
        results = audit_obj_vertex_symmetry(
            path,
            orders=spec.required_verified_orders,
            decimal_places=contract.coordinate_decimal_places,
        )
        if not all(results.values()):
            failures = [order for order, passed in results.items() if not passed]
            raise ValueError(
                f"{object_id} OBJ symmetry audit failed for orders {failures}"
            )
        geometry[object_id] = {
            "mesh_path": str(contract.inputs[spec.geometry_input].path),
            "mesh_sha256": contract.inputs[spec.geometry_input].sha256,
            "required_orders_verified": {
                str(order): passed for order, passed in results.items()
            },
            "declared_minimum_order": spec.order,
            "unique_key_geometry_present": False,
        }

    pair_hypotheses = relative_yaw_hypotheses(2, 2)
    gates = {
        "content_addressed_inputs_verified": True,
        "active_v1_mismatch_confirmed": True,
        "loose_order_2_vertex_symmetry_verified": True,
        "fixed_order_2_vertex_symmetry_verified": True,
        "nut_order_24_vertex_symmetry_verified": True,
        "v2_migration_contract_parser_ready": True,
        "production_v2_pose_parser_ready": False,
        "hash_bound_unique_key_geometry_available": False,
        "foundationpose_shadow_accuracy_qualified": False,
        "v2_runtime_activated": False,
        "vision_control_authorized": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "AUDIT_CONFIRMED_V2_DESIGNED_NOT_ACTIVATED",
        "contract_enabled": contract.enabled,
        "active_registry": {
            "path": str(contract.inputs["active_pose_registry_v1"].path),
            "sha256": contract.inputs["active_pose_registry_v1"].sha256,
            "affected_symmetry_classes": active_classes,
            "modified_by_migration": False,
        },
        "geometry_audit": geometry,
        "proposed_v2": {
            "contract_schema_version": "kcg_connector_pose_contract_v2",
            "runtime_registration_enabled": False,
            "endpoint_symmetry_orders": {
                model.role: model.symmetry.order
                for model in contract.proposed_models.values()
            },
            "coupling_nut_symmetry_order": contract.proposed_component.order,
            "loose_fixed_relative_yaw_hypotheses_rad": pair_hypotheses,
            "loose_fixed_relative_yaw_hypothesis_count": len(pair_hypotheses),
        },
        "key_geometry": {
            "available": False,
            "polarization_N_label_counts_as_geometry_evidence": False,
            "acceptable_sources": list(
                contract.document["key_geometry_upgrade"][
                    "acceptable_primary_geometry_sources"
                ]
            ),
            "new_model_ids_required": dict(
                contract.document["key_geometry_upgrade"][
                    "keyed_upgrade_model_ids"
                ]
            ),
        },
        "foundationpose_output_policy": {
            "current_output_semantics": "pose_equivalence_class",
            "endpoint_orientation_metric": (
                "minimum_over_right_multiplied_object_symmetry_group"
            ),
            "raw_yaw_is_keyed_accuracy": False,
            "canonical_representative_can_authorize_control": False,
            "coupling_nut_evaluated_separately_modulo_order": 24,
        },
        "control": {
            "current_authorized": False,
            "stage_requirements": {
                stage: list(requirements)
                for stage, requirements in contract.control_requirements.items()
            },
            "force_guided_key_search_is_separate_contract": True,
        },
        "migration_phases": contract.migration_phases,
        "gates": gates,
        "blockers": [
            "production_v2_pose_contract_and_parser_not_implemented",
            "foundationpose_shadow_inference_not_run_or_qualified",
            "hash_bound_unique_polarization_key_geometry_missing",
            "key_reference_frames_and_target_transforms_not_calibrated",
            "stage_scoped_control_regression_not_qualified",
        ],
        "next_actions": [
            "implement_connector_pose_v2_parser_side_by_side_without_touching_v1",
            "publish_foundationpose_hypothesis_sets_for_evaluation_only",
            (
                "evaluate_translation_axis_and_orientation_modulo_symmetry_"
                "on_withheld_truth"
            ),
            "obtain_licensed_exact_part_CAD_or_traceable_mated_part_metrology",
            "create_new_keyed_model_ids_and_calibrate_key_and_target_frames",
            "authorize_only_the_stage_whose_complete_gate_set_passes",
        ],
        "boundaries": contract.boundaries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CPU-only readiness audit and optionally persist its report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        contract = load_pose_registry_v2_migration_contract(arguments.config)
        report = evaluate_pose_registry_v2_migration(
            contract, arguments.repository
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(json.dumps({"status": "INVALID", "error": str(error)}))
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    # The audit itself passes, while runtime/control correctly remain disabled.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
