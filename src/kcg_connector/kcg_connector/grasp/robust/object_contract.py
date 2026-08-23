"""Strict object-contract loading for the CARTS-Grasp cross-model study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from kcg_connector.grasp.robust.object_material_boundary import (
    CURRENT_REPRESENTATION,
    SINGLE_REPRESENTATION,
    ObjectMaterialBoundaryError,
    ObjectMaterialBoundaryEvidence,
    load_object_material_boundary_evidence,
)
from kcg_connector.grasp.robust.object_model import ObjectGraspModel
from kcg_connector.grasp.robust.surface_orientation import (
    SurfaceBoundaryRole,
    SurfaceOrientationAuditError,
    SurfaceOrientationCertificate,
    audit_surface_orientation,
)


_SCHEMA_VERSION = "carts_grasp_objects_v1"
_STUDY_ID = "CARTS-GRASP-CROSS-OBJECT-V1"
_CLAIM_SCOPE = "SIMULATION_ONLY_ZERO_OBJECT_TUNING_CROSS_MODEL_CASE_STUDY"
_MATERIAL_MODELS = frozenset(
    {
        "BOUNDED_COULOMB_INTERVAL_FROM_FROZEN_STATIC_AND_DYNAMIC_VALUES",
        "SHARED_STUDY_MATERIAL_CLASS_BOUNDED_COULOMB_INTERVAL",
    }
)
_ORIENTATION_ROLE_BY_GEOMETRY_FORMAT: Mapping[
    str, SurfaceBoundaryRole
] = MappingProxyType(
    {
        "CARTS_GRASP_VISUAL_SUBTREE_NPZ_V1": (
            SurfaceBoundaryRole.SOURCE_INDEXED_CLOSED_COMPONENT_SOUP
        ),
        "BINARY_STL_TESSELLATION_FROM_ORIGINAL_STEP": (
            SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY
        ),
    }
)


class ObjectContractError(ValueError):
    """Raised when the object-study evidence boundary is incomplete."""


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
        except TypeError as error:
            raise ObjectContractError(
                "YAML mapping keys must be scalar and hashable"
            ) from error
        if duplicate:
            raise ObjectContractError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _exact_keys(
    document: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set.difference(document))
    extra = sorted(set(document).difference(expected_set))
    if missing or extra:
        raise ObjectContractError(
            f"{label} schema mismatch; missing={missing}, extra={extra}"
        )


def _exact_bool(value: Any, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise ObjectContractError(f"{label} must be exactly {expected}")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ObjectContractError(f"{label} must be a non-empty string")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ObjectContractError(f"{label} keys must be strings")
    return value


def _required(document: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in document:
        raise ValueError(f"{label} is missing required field {key!r}")
    return document[key]


def _repository_path(repository_root: Path, relative: Any, label: str) -> Path:
    raw = Path(str(relative))
    if raw.is_absolute():
        raise ValueError(f"{label} must be repository-relative")
    root = repository_root.resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(
    repository_root: Path,
    document: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    label: str,
) -> Path:
    path = _repository_path(
        repository_root, _required(document, path_key, label), f"{label}.{path_key}"
    )
    expected = str(_required(document, sha_key, label)).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{label}.{sha_key} must be one SHA-256 digest")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return path


def _verified_fragmented_reference(
    repository_root: Path,
    reference: Any,
    expected_sha256: Any,
    label: str,
) -> Path:
    raw = str(reference)
    file_part, separator, fragment = raw.partition("#")
    if not file_part or (separator and not fragment):
        raise ValueError(f"{label} must contain a file and optional non-empty fragment")
    document = {"path": file_part, "sha256": expected_sha256}
    return _verified_file(
        repository_root,
        document,
        path_key="path",
        sha_key="sha256",
        label=label,
    )


def _resolve_mapping_fragment(
    document: Mapping[str, Any], fragment: str, label: str
) -> Mapping[str, Any]:
    current: Any = document
    for component in fragment.split("."):
        if not component or not isinstance(current, Mapping) or component not in current:
            raise ObjectContractError(
                f"{label} fragment does not resolve at {component!r}"
            )
        current = current[component]
    return _mapping(current, f"{label} fragment")


@dataclass(frozen=True)
class ContactMaterialUncertainty:
    """Hash-bound Coulomb interval consumed by the robust evaluator.

    This is an epistemic simulation contract, not a vendor measurement or a
    probability distribution.  Keeping it on ``LoadedObjectContract`` closes
    the former integration loophole in which a runner had to re-read raw YAML.
    """

    model: str
    friction_coefficient_interval: tuple[float, float]
    source_class: str
    source_reference: str
    source_sha256: str
    probability_distribution_claimed: bool
    vendor_friction_claimed: bool


def _contact_material_uncertainty(
    *,
    repository_root: Path,
    object_document: Mapping[str, Any],
    object_id: str,
) -> ContactMaterialUncertainty:
    label = f"{object_id}.contact_material_uncertainty"
    material = _mapping(
        _required(object_document, "contact_material_uncertainty", object_id),
        label,
    )
    _exact_keys(
        material,
        (
            "model",
            "friction_coefficient",
            "source_class",
            "source",
            "source_sha256",
            "vendor_friction_claimed",
            "probability_distribution_claimed",
        ),
        label,
    )
    model = _required_string(material["model"], f"{label}.model")
    if model not in _MATERIAL_MODELS:
        raise ObjectContractError(f"{label}.model is not a registered V1 model")
    interval_array = _vector(
        material["friction_coefficient"],
        (2,),
        f"{label}.friction_coefficient",
    )
    lower, upper = (float(interval_array[0]), float(interval_array[1]))
    if lower < 0.0 or upper < lower:
        raise ObjectContractError(
            f"{label}.friction_coefficient must be a nonnegative ordered interval"
        )
    source_class = _required_string(
        material["source_class"], f"{label}.source_class"
    )
    source_reference = _required_string(material["source"], f"{label}.source")
    file_part, separator, fragment = source_reference.partition("#")
    if not separator or not file_part or not fragment:
        raise ObjectContractError(
            f"{label}.source must bind a repository file and mapping fragment"
        )
    source_path = _verified_fragmented_reference(
        repository_root,
        source_reference,
        material["source_sha256"],
        f"{label}.source",
    )
    try:
        source_document = _mapping(
            yaml.load(
                source_path.read_text(encoding="utf-8"),
                Loader=_UniqueKeySafeLoader,
            ),
            f"{label}.source document",
        )
    except yaml.YAMLError as error:
        raise ObjectContractError(f"{label}.source is not valid unique-key YAML") from error
    role = _resolve_mapping_fragment(source_document, fragment, f"{label}.source")
    try:
        source_dynamic = float(
            _required(role, "dynamic_friction", f"{label}.source")
        )
        source_static = float(_required(role, "static_friction", f"{label}.source"))
    except (TypeError, ValueError) as error:
        raise ObjectContractError(
            f"{label}.source friction values must be numeric"
        ) from error
    if not (
        math.isfinite(source_dynamic)
        and math.isfinite(source_static)
        and lower == source_dynamic
        and upper == source_static
    ):
        raise ObjectContractError(
            f"{label} interval must equal the bound source dynamic/static values"
        )
    probability_claimed = _exact_bool(
        material["probability_distribution_claimed"],
        False,
        f"{label}.probability_distribution_claimed",
    )
    vendor_claimed = _exact_bool(
        material["vendor_friction_claimed"],
        False,
        f"{label}.vendor_friction_claimed",
    )
    return ContactMaterialUncertainty(
        model=model,
        friction_coefficient_interval=(lower, upper),
        source_class=source_class,
        source_reference=source_reference,
        source_sha256=_sha256(source_path),
        probability_distribution_claimed=probability_claimed,
        vendor_friction_claimed=vendor_claimed,
    )


def _vector(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must have finite shape {shape}")
    return array


def _unit_vector(value: Any, label: str) -> np.ndarray:
    vector = _vector(value, (3,), label)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{label} must be non-zero")
    return vector / norm


def _proper_rotation(value: Any, label: str) -> np.ndarray:
    rotation = _vector(value, (3, 3), label)
    numerical_bound = (
        128.0 * np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(rotation)))
    )
    if (
        float(np.linalg.norm(rotation.T @ rotation - np.eye(3))) > numerical_bound
        or abs(float(np.linalg.det(rotation)) - 1.0) > numerical_bound
    ):
        raise ValueError(f"{label} must be a proper orthonormal rotation")
    return rotation


def _contains_pending(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("PENDING") or value in {"NOT_BUILT", "NOT_CALIBRATED"}
    if isinstance(value, Mapping):
        return any(_contains_pending(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return any(_contains_pending(item) for item in value)
    return False


def mass_distribution_rms_radius(model: ObjectGraspModel) -> float:
    """Return the exact mass RMS radius implied by inertia about the COM.

    For a rigid body, ``trace(I_COM) = 2 integral ||r||^2 dm``.  Therefore
    ``sqrt(trace(I_COM)/(2m))`` is independent of mesh tessellation, surface
    semantics and rigid coordinate transforms.  This makes it a common
    task-wrench moment scale even when two CAD sources have different surface
    decompositions.
    """

    inertia_trace = float(np.trace(model.inertia_kg_m2))
    radius_squared = inertia_trace / (2.0 * float(model.mass_kg))
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise ValueError("mass-distribution RMS radius must be positive")
    return math.sqrt(radius_squared)


def _validate_visual_subtree_manifest(
    *,
    manifest_path: Path,
    geometry_path: Path,
    source_stage_path: Path,
    geometry_contract: Mapping[str, Any],
    model: ObjectGraspModel,
) -> None:
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("planning geometry manifest is not valid JSON") from error
    manifest = _mapping(document, "planning geometry manifest")
    required = {
        "schema_version",
        "scope",
        "source_stage",
        "source_stage_sha256",
        "source_subtree",
        "source_mesh_prim_count",
        "source_gprim_type_counts",
        "analytic_primitive_tessellation",
        "vertex_count",
        "triangle_count",
        "bounds_m",
        "output",
        "output_sha256",
        "physics_loaded",
        "collision_or_contact_truth_read",
        "source_modified",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"planning geometry manifest is missing fields: {missing}")
    if manifest["schema_version"] != "carts_grasp_visual_subtree_mesh_v1":
        raise ValueError("planning geometry manifest schema changed")
    if manifest["scope"] != "OFFLINE_VISUAL_GEOMETRY_ONLY_NO_PHYSX_TRUTH":
        raise ValueError("planning geometry manifest scope is not offline visual geometry")
    if Path(str(manifest["source_stage"])).resolve() != source_stage_path.resolve():
        raise ValueError("planning geometry manifest source stage path mismatch")
    if str(manifest["source_stage_sha256"]) != _sha256(source_stage_path):
        raise ValueError("planning geometry manifest source stage SHA-256 mismatch")
    if str(manifest["source_subtree"]) != str(
        _required(geometry_contract, "source_subtree", "planning_geometry")
    ):
        raise ValueError("planning geometry manifest source subtree mismatch")
    if Path(str(manifest["output"])).resolve() != geometry_path.resolve():
        raise ValueError("planning geometry manifest output path mismatch")
    if str(manifest["output_sha256"]) != _sha256(geometry_path):
        raise ValueError("planning geometry manifest output SHA-256 mismatch")
    for field in (
        "physics_loaded",
        "collision_or_contact_truth_read",
        "source_modified",
    ):
        if manifest[field] is not False:
            raise ValueError(f"planning geometry manifest {field} must be false")
    if int(manifest["vertex_count"]) != len(model.mesh.vertices_m):
        raise ValueError("planning geometry manifest vertex count mismatch")
    if int(manifest["triangle_count"]) != len(model.mesh.faces):
        raise ValueError("planning geometry manifest triangle count mismatch")
    bounds = np.asarray(manifest["bounds_m"], dtype=np.float64)
    actual_bounds = np.vstack(
        (np.min(model.mesh.vertices_m, axis=0), np.max(model.mesh.vertices_m, axis=0))
    )
    if bounds.shape != (2, 3) or not np.array_equal(bounds, actual_bounds):
        raise ValueError("planning geometry manifest bounds mismatch")
    type_counts = _mapping(
        manifest["source_gprim_type_counts"],
        "planning geometry manifest source_gprim_type_counts",
    )
    if (
        not type_counts
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for count in type_counts.values()
        )
        or sum(type_counts.values()) != int(manifest["source_mesh_prim_count"])
    ):
        raise ValueError("planning geometry manifest Gprim inventory mismatch")
    tessellation = _mapping(
        manifest["analytic_primitive_tessellation"],
        "planning geometry manifest analytic_primitive_tessellation",
    )
    relative_error = float(
        _required(
            tessellation,
            "maximum_relative_sagitta_error",
            "analytic_primitive_tessellation",
        )
    )
    multiplier = int(
        _required(
            tessellation,
            "resolution_multiplier",
            "analytic_primitive_tessellation",
        )
    )
    edge_count = int(
        _required(
            tessellation,
            "circle_edge_count",
            "analytic_primitive_tessellation",
        )
    )
    if not 0.0 < relative_error < 1.0 or multiplier < 1:
        raise ValueError("planning geometry tessellation contract is invalid")
    base_edges = edge_count // multiplier
    if (
        base_edges * multiplier != edge_count
        or base_edges < 3
        or 1.0 - math.cos(math.pi / base_edges) > relative_error
        or (
            base_edges > 3
            and 1.0 - math.cos(math.pi / (base_edges - 1)) <= relative_error
        )
    ):
        raise ValueError("planning geometry tessellation edge derivation mismatch")


@dataclass(frozen=True)
class LoadedObjectContract:
    object_id: str
    model: ObjectGraspModel
    characteristic_radius_m: float
    task_frame_rotation_object: np.ndarray
    nominal_validation_gravity_direction_object: np.ndarray
    task_frame_source: str
    contact_material_uncertainty: ContactMaterialUncertainty
    orientation_certificate: SurfaceOrientationCertificate
    material_boundary_evidence: ObjectMaterialBoundaryEvidence
    verified_source_sha256: Mapping[str, str]
    geometry_contract: Mapping[str, Any]
    physical_contract: Mapping[str, Any]
    uncertainty_contract: Mapping[str, Any]
    uncertainty_calibrated: bool
    dynamic_eligible: bool
    dynamic_ineligibility_reason: str


def _orientation_role_for_geometry_format(
    geometry_format: str,
) -> SurfaceBoundaryRole:
    """Map an explicit geometry source/type to its registered topology role."""

    try:
        return _ORIENTATION_ROLE_BY_GEOMETRY_FORMAT[geometry_format]
    except KeyError as error:
        raise ObjectContractError(
            "planning_geometry.format has no registered surface-boundary role: "
            f"{geometry_format!r}"
        ) from error


def _physical_values(
    object_document: Mapping[str, Any], label: str
) -> tuple[float, np.ndarray, np.ndarray, Mapping[str, Any]]:
    physical = _mapping(_required(object_document, "physical_properties", label), f"{label}.physical_properties")
    if "planning_rigid_composition" in physical:
        values = _mapping(physical["planning_rigid_composition"], f"{label}.planning_rigid_composition")
    else:
        values = physical
    mass = float(_required(values, "mass_kg", f"{label}.physical_properties"))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"{label} mass must be finite and positive")
    com = _vector(
        _required(values, "center_of_mass_m", f"{label}.physical_properties"),
        (3,),
        f"{label}.center_of_mass_m",
    )
    inertia = _vector(
        _required(values, "inertia_kg_m2", f"{label}.physical_properties"),
        (3, 3),
        f"{label}.inertia_kg_m2",
    )
    principal = np.linalg.eigvalsh(0.5 * (inertia + inertia.T))
    inertia_scale = max(float(np.linalg.norm(inertia, ord=2)), np.finfo(np.float64).tiny)
    numerical_bound = 256.0 * np.finfo(np.float64).eps * inertia_scale
    if principal[-1] <= numerical_bound:
        raise ValueError(f"{label} inertia must represent a non-zero rigid body")
    if principal[-1] > principal[0] + principal[1] + numerical_bound:
        raise ValueError(f"{label} principal inertia violates the rigid-body triangle inequality")
    return mass, com, inertia, physical


def load_object_contract(
    contract_path: Path | str,
    *,
    object_id: str,
    repository_root: Path | str,
) -> LoadedObjectContract:
    """Load one object with no algorithm or physical-property defaults."""

    root = Path(repository_root).resolve()
    path = Path(contract_path)
    if not path.is_absolute():
        path = (root / path).resolve()
    try:
        document = _mapping(
            yaml.load(
                path.read_text(encoding="utf-8"),
                Loader=_UniqueKeySafeLoader,
            ),
            "object contract",
        )
    except yaml.YAMLError as error:
        raise ObjectContractError("object contract is not valid unique-key YAML") from error
    _exact_keys(
        document,
        (
            "schema_version",
            "study_id",
            "claim_scope",
            "shared_method_config",
            "shared_method_config_required_for_every_object",
            "object_specific_algorithm_hyperparameters_allowed",
            "hardware_authorized",
            "transfer_protocol",
            "objects",
        ),
        "object contract",
    )
    if document["schema_version"] != _SCHEMA_VERSION:
        raise ObjectContractError("object contract schema_version changed")
    if document["study_id"] != _STUDY_ID:
        raise ObjectContractError("object contract study_id changed")
    if document["claim_scope"] != _CLAIM_SCOPE:
        raise ObjectContractError("object contract claim_scope changed")
    _repository_path(
        root,
        document["shared_method_config"],
        "object contract.shared_method_config",
    )
    _exact_bool(
        document["shared_method_config_required_for_every_object"],
        True,
        "object contract.shared_method_config_required_for_every_object",
    )
    _exact_bool(
        document["object_specific_algorithm_hyperparameters_allowed"],
        False,
        "object contract.object_specific_algorithm_hyperparameters_allowed",
    )
    _exact_bool(
        document["hardware_authorized"],
        False,
        "object contract.hardware_authorized",
    )
    transfer = _mapping(document["transfer_protocol"], "transfer_protocol")
    _exact_keys(
        transfer,
        (
            "development_object",
            "frozen_transfer_object",
            "transfer_object_geometry_seen_before_method_freeze",
            "prospective_double_blind_claim_allowed",
            "candidate_ids_or_contact_coordinates_shared_between_objects",
        ),
        "transfer_protocol",
    )
    development_object = _required_string(
        transfer["development_object"], "transfer_protocol.development_object"
    )
    transfer_object = _required_string(
        transfer["frozen_transfer_object"],
        "transfer_protocol.frozen_transfer_object",
    )
    if development_object == transfer_object:
        raise ObjectContractError("development and transfer objects must be distinct")
    _exact_bool(
        transfer["transfer_object_geometry_seen_before_method_freeze"],
        True,
        "transfer_protocol.transfer_object_geometry_seen_before_method_freeze",
    )
    _exact_bool(
        transfer["prospective_double_blind_claim_allowed"],
        False,
        "transfer_protocol.prospective_double_blind_claim_allowed",
    )
    _exact_bool(
        transfer["candidate_ids_or_contact_coordinates_shared_between_objects"],
        False,
        "transfer_protocol.candidate_ids_or_contact_coordinates_shared_between_objects",
    )
    objects = _mapping(_required(document, "objects", "object contract"), "objects")
    if set(objects) != {development_object, transfer_object}:
        raise ObjectContractError(
            "objects must be exactly the registered development and transfer objects"
        )
    if object_id not in objects:
        raise KeyError(f"object contract has no object {object_id!r}")
    object_document = _mapping(objects[object_id], f"objects.{object_id}")
    frames = _mapping(_required(object_document, "frames", object_id), f"{object_id}.frames")
    axis = _unit_vector(
        _required(frames, "assembly_axis_object", f"{object_id}.frames"),
        f"{object_id}.assembly_axis_object",
    )
    if str(_required(frames, "length_unit", f"{object_id}.frames")) != "m":
        raise ValueError(f"{object_id} object frame must use SI meters")
    task_frame = _proper_rotation(
        _required(frames, "task_frame_rotation_object", f"{object_id}.frames"),
        f"{object_id}.task_frame_rotation_object",
    )
    frame_axis_error = float(np.linalg.norm(task_frame[:, 2] - axis))
    frame_axis_bound = 128.0 * np.finfo(np.float64).eps
    if frame_axis_error > frame_axis_bound:
        raise ValueError(
            f"{object_id} task-frame third axis must equal assembly_axis_object"
        )
    gravity_direction = _unit_vector(
        _required(
            frames,
            "nominal_validation_gravity_direction_object",
            f"{object_id}.frames",
        ),
        f"{object_id}.nominal_validation_gravity_direction_object",
    )
    task_frame_source = str(
        _required(frames, "task_frame_source", f"{object_id}.frames")
    )
    if not task_frame_source:
        raise ValueError(f"{object_id}.frames.task_frame_source cannot be empty")
    task_frame = np.array(task_frame, dtype=np.float64, copy=True)
    gravity_direction = np.array(gravity_direction, dtype=np.float64, copy=True)
    task_frame.setflags(write=False)
    gravity_direction.setflags(write=False)
    mass, com, inertia, physical = _physical_values(object_document, object_id)
    contact_material = _contact_material_uncertainty(
        repository_root=root,
        object_document=object_document,
        object_id=object_id,
    )
    geometry = _mapping(
        _required(object_document, "planning_geometry", object_id),
        f"{object_id}.planning_geometry",
    )
    geometry_path = _verified_file(
        root, geometry, path_key="path", sha_key="sha256", label=f"{object_id}.planning_geometry"
    )
    verified_sources: dict[str, str] = {
        "planning_geometry": _sha256(geometry_path),
        "contact_material_source": contact_material.source_sha256,
    }
    semantic = str(_required(geometry, "allowed_surface_semantic", f"{object_id}.planning_geometry"))
    geometry_format = str(_required(geometry, "format", f"{object_id}.planning_geometry"))
    orientation_role = _orientation_role_for_geometry_format(geometry_format)

    if geometry_format == "CARTS_GRASP_VISUAL_SUBTREE_NPZ_V1":
        manifest_path = _verified_file(
            root,
            geometry,
            path_key="manifest",
            sha_key="manifest_sha256",
            label=f"{object_id}.planning_geometry_manifest",
        )
        source_stage_path = _verified_file(
            root,
            geometry,
            path_key="source_stage",
            sha_key="source_stage_sha256",
            label=f"{object_id}.planning_geometry_source_stage",
        )
        verified_sources["planning_geometry_manifest"] = _sha256(manifest_path)
        verified_sources["planning_geometry_source_stage"] = _sha256(
            source_stage_path
        )
        model = ObjectGraspModel.from_visual_subtree_npz(
            geometry_path,
            source_class=str(_mapping(_required(object_document, "identity", object_id), f"{object_id}.identity")["source_class"]),
            assembly_axis=axis,
            mass_kg=mass,
            center_of_mass_m=com,
            inertia_kg_m2=inertia,
            allowed_contact_semantics=(semantic,),
            default_face_semantic=semantic,
            orient_outward=False,
            require_watertight=False,
        )
        _validate_visual_subtree_manifest(
            manifest_path=manifest_path,
            geometry_path=geometry_path,
            source_stage_path=source_stage_path,
            geometry_contract=geometry,
            model=model,
        )
    elif geometry_format == "BINARY_STL_TESSELLATION_FROM_ORIGINAL_STEP":
        source_unit = str(_required(geometry, "source_unit", f"{object_id}.planning_geometry"))
        model = ObjectGraspModel.from_stl(
            geometry_path,
            unit=source_unit,
            source_class=str(_mapping(_required(object_document, "identity", object_id), f"{object_id}.identity")["source_class"]),
            assembly_axis=axis,
            mass_kg=mass,
            center_of_mass_m=com,
            inertia_kg_m2=inertia,
            allowed_contact_semantics=(semantic,),
            default_face_semantic=semantic,
            orient_outward=False,
            require_watertight=bool(_required(geometry, "watertight", f"{object_id}.planning_geometry")),
        )
        original = _mapping(_required(object_document, "original_cad", object_id), f"{object_id}.original_cad")
        original_path = _verified_file(
            root, original, path_key="path", sha_key="sha256", label=f"{object_id}.original_cad"
        )
        audit_path = _verified_file(
            root,
            original,
            path_key="geometry_audit",
            sha_key="geometry_audit_sha256",
            label=f"{object_id}.original_cad_geometry_audit",
        )
        verified_sources["original_cad"] = _sha256(original_path)
        verified_sources["original_cad_geometry_audit"] = _sha256(audit_path)
    else:
        raise ValueError(f"unsupported planning geometry format {geometry_format!r}")

    geometry_sha256 = verified_sources["planning_geometry"]
    if model.provenance.source_sha256 != geometry_sha256:
        raise ObjectContractError(
            f"{object_id}.planning_geometry loader provenance is not bound to "
            "the verified geometry SHA-256"
        )
    try:
        orientation_certificate = audit_surface_orientation(
            model.mesh.vertices_m,
            model.mesh.faces,
            role=orientation_role,
        )
    except SurfaceOrientationAuditError as error:
        raise ObjectContractError(
            f"{object_id}.planning_geometry surface orientation audit failed: {error}"
        ) from error

    material_evidence_contract = _mapping(
        _required(
            geometry,
            "material_boundary_evidence",
            f"{object_id}.planning_geometry",
        ),
        f"{object_id}.planning_geometry.material_boundary_evidence",
    )
    _exact_keys(
        material_evidence_contract,
        ("representation", "path", "sha256"),
        f"{object_id}.planning_geometry.material_boundary_evidence",
    )
    expected_representation = (
        CURRENT_REPRESENTATION
        if geometry_format == "CARTS_GRASP_VISUAL_SUBTREE_NPZ_V1"
        else SINGLE_REPRESENTATION
    )
    if material_evidence_contract["representation"] != expected_representation:
        raise ObjectContractError(
            f"{object_id}.planning_geometry material representation changed"
        )
    material_evidence_path = _verified_file(
        root,
        material_evidence_contract,
        path_key="path",
        sha_key="sha256",
        label=f"{object_id}.planning_geometry.material_boundary_evidence",
    )
    try:
        material_boundary_evidence = load_object_material_boundary_evidence(
            material_evidence_path,
            repository_root=root,
            expected_object_id=object_id,
            expected_source_asset_path=geometry_path,
            expected_source_asset_sha256=geometry_sha256,
            orientation_certificate=orientation_certificate,
        )
    except (ObjectMaterialBoundaryError, ValueError) as error:
        raise ObjectContractError(
            f"{object_id}.planning_geometry material boundary failed: {error}"
        ) from error
    verified_sources["material_boundary_evidence"] = _sha256(
        material_evidence_path
    )
    verified_sources["material_boundary_role_authority"] = (
        material_boundary_evidence.role_authority_sha256
    )

    if "source_contract" in physical:
        source_contract_path = _verified_fragmented_reference(
            root,
            physical["source_contract"],
            _required(physical, "source_contract_sha256", f"{object_id}.physical_properties"),
            f"{object_id}.physical_source_contract",
        )
        verified_sources["physical_source_contract"] = _sha256(source_contract_path)
    elif "mass_source" in physical:
        mass_source_path = _verified_fragmented_reference(
            root,
            physical["mass_source"],
            _required(physical, "mass_source_sha256", f"{object_id}.physical_properties"),
            f"{object_id}.mass_source",
        )
        verified_sources["mass_source"] = _sha256(mass_source_path)
    else:
        raise ValueError(f"{object_id} physical properties lack a verified source file")

    uncertainty = _mapping(
        _required(object_document, "uncertainty_calibration", object_id),
        f"{object_id}.uncertainty_calibration",
    )
    uncertainty_calibrated = not _contains_pending(uncertainty)
    eligibility = _mapping(
        _required(object_document, "dynamic_eligibility", object_id),
        f"{object_id}.dynamic_eligibility",
    )
    dynamic_eligible = _required(eligibility, "allowed", f"{object_id}.dynamic_eligibility")
    if not isinstance(dynamic_eligible, bool):
        raise ValueError(f"{object_id}.dynamic_eligibility.allowed must be Boolean")
    reason = str(_required(eligibility, "reason", f"{object_id}.dynamic_eligibility"))
    if dynamic_eligible and (not uncertainty_calibrated or _contains_pending(object_document)):
        raise ValueError(f"{object_id} cannot be dynamically eligible with pending contracts")
    return LoadedObjectContract(
        object_id=object_id,
        model=model,
        characteristic_radius_m=mass_distribution_rms_radius(model),
        task_frame_rotation_object=task_frame,
        nominal_validation_gravity_direction_object=gravity_direction,
        task_frame_source=task_frame_source,
        contact_material_uncertainty=contact_material,
        orientation_certificate=orientation_certificate,
        material_boundary_evidence=material_boundary_evidence,
        verified_source_sha256=MappingProxyType(dict(verified_sources)),
        geometry_contract=geometry,
        physical_contract=physical,
        uncertainty_contract=uncertainty,
        uncertainty_calibrated=uncertainty_calibrated,
        dynamic_eligible=dynamic_eligible,
        dynamic_ineligibility_reason=reason,
    )


__all__ = [
    "ContactMaterialUncertainty",
    "LoadedObjectContract",
    "ObjectContractError",
    "mass_distribution_rms_radius",
    "load_object_contract",
]
