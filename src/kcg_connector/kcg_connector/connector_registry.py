"""Fail-closed connector model registry without ROS or simulator imports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


REGISTRY_SCHEMA_VERSION = "kcg_connector_model_registry_v1"
REGISTRY_SCOPE = "synthetic_curriculum_only"
_EPSILON = 1.0e-9


class ConnectorRole(str, Enum):
    """How a connector end participates in the tabletop workflow."""

    LOOSE = "loose"
    FIXED = "fixed"


class ConnectorGender(str, Enum):
    """Minimal mating-gender vocabulary for registry validation."""

    MALE = "male"
    FEMALE = "female"


@dataclass(frozen=True)
class AssemblyFrame:
    """A connector mating datum expressed in a named component frame."""

    frame_id: str
    parent_component: str
    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class RotationalSymmetry:
    """Pose equivalence and optional key reference about the mating axis."""

    kind: str
    order: Optional[int]
    axis_in_assembly_frame: tuple[float, float, float]
    key_reference_direction_in_assembly_frame: Optional[
        tuple[float, float, float]
    ]


@dataclass(frozen=True)
class GraspRegion:
    """An allowed cylindrical grasp surface on a loose connector."""

    region_id: str
    parent_component: str
    shape: str
    axis_in_region_frame: tuple[float, float, float]
    axial_bounds_m: tuple[float, float]
    radius_bounds_m: tuple[float, float]


@dataclass(frozen=True)
class ConnectorModel:
    """One loose or fixed connector-end model."""

    model_id: str
    designation: str
    provenance: str
    deployment_role: ConnectorRole
    gender: ConnectorGender
    compatible_model_ids: tuple[str, ...]
    assembly_frame: Optional[AssemblyFrame]
    rotational_symmetry: Optional[RotationalSymmetry]
    grasp_regions: tuple[GraspRegion, ...]


@dataclass(frozen=True)
class InsertionSpec:
    """Insertion direction and geometric gates in the fixed assembly frame."""

    direction_in_fixed_assembly_frame: Optional[
        tuple[float, float, float]
    ]
    minimum_engage_depth_m: Optional[float]
    maximum_lateral_error_m: Optional[float]
    maximum_angular_error_degrees: Optional[float]
    maximum_key_error_degrees: Optional[float]


@dataclass(frozen=True)
class FasteningSpec:
    """Synthetic screw relation without a robot-joint direction mapping."""

    mechanism: Optional[str]
    axis_in_fixed_assembly_frame: Optional[tuple[float, float, float]]
    tightening_sign_about_axis: Optional[int]
    target_angle_degrees: Optional[float]
    angle_tolerance_degrees: Optional[float]
    lead_m_per_revolution: Optional[float]


@dataclass(frozen=True)
class ConnectorSafetyLimits:
    """Per-model damage limits required before an assembly profile can run."""

    maximum_insertion_axial_force_n: Optional[float]
    maximum_lateral_force_n: Optional[float]
    maximum_bending_moment_nm: Optional[float]
    maximum_tightening_torque_nm: Optional[float]
    maximum_finger_base_torque_nm: Optional[float]


@dataclass(frozen=True)
class AssemblyProfile:
    """A validated loose/fixed pairing and its assembly parameters."""

    profile_id: str
    enabled: bool
    fidelity: str
    loose_model_id: str
    fixed_model_id: str
    insertion: InsertionSpec
    fastening: FasteningSpec
    safety_limits: ConnectorSafetyLimits


@dataclass(frozen=True)
class ConnectorModelRegistry:
    """Immutable registry with an explicit global operational kill switch."""

    schema_version: str
    enabled: bool
    scope: str
    models: tuple[ConnectorModel, ...]
    assembly_profiles: tuple[AssemblyProfile, ...]

    def model(self, model_id: str) -> ConnectorModel:
        """Return a named model or fail instead of choosing a fallback."""
        matches = [item for item in self.models if item.model_id == model_id]
        if len(matches) != 1:
            raise ValueError(f"unknown connector model: {model_id!r}")
        return matches[0]

    def profile(self, profile_id: str) -> AssemblyProfile:
        """Return a named pair profile or fail instead of guessing by shape."""
        matches = [
            item
            for item in self.assembly_profiles
            if item.profile_id == profile_id
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown assembly profile: {profile_id!r}")
        return matches[0]

    def activation_blockers(self, profile_id: str) -> tuple[str, ...]:
        """List every reason the profile is not eligible for operation."""
        profile = self.profile(profile_id)
        loose = self.model(profile.loose_model_id)
        fixed = self.model(profile.fixed_model_id)
        blockers = []
        if not self.enabled:
            blockers.append("registry_disabled")
        if not profile.enabled:
            blockers.append("profile_disabled")
        blockers.extend(_profile_data_blockers(profile, loose, fixed))
        return tuple(blockers)

    def require_enabled_profile(self, profile_id: str) -> AssemblyProfile:
        """Return an operational profile only when every safety gate passes."""
        blockers = self.activation_blockers(profile_id)
        if blockers:
            joined = ", ".join(blockers)
            raise ValueError(
                f"assembly profile {profile_id!r} is not enabled: {joined}"
            )
        return self.profile(profile_id)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _finite_tuple(
    value: Any, size: int, label: str
) -> tuple[float, ...]:
    try:
        items = tuple(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite numbers") from error
    if any(isinstance(item, bool) for item in items):
        raise ValueError(f"{label} must contain finite numbers")
    try:
        result = tuple(float(item) for item in items)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite numbers") from error
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain {size} finite numbers")
    return result


def _optional_vector(
    value: Any, size: int, label: str
) -> Optional[tuple[float, ...]]:
    if value is None:
        return None
    return _finite_tuple(value, size, label)


def _optional_positive(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be positive and finite") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _unit_vector(value: tuple[float, ...], label: str) -> None:
    norm = math.sqrt(sum(component * component for component in value))
    if abs(norm - 1.0) > 1.0e-6:
        raise ValueError(f"{label} must be a unit vector")


def _parse_frame(value: Any, label: str) -> Optional[AssemblyFrame]:
    if value is None:
        return None
    document = _mapping(value, label)
    _exact_keys(
        document,
        {
            "frame_id",
            "parent_component",
            "translation_m",
            "quaternion_xyzw",
        },
        label,
    )
    quaternion = _finite_tuple(
        document["quaternion_xyzw"], 4, f"{label}.quaternion_xyzw"
    )
    norm = math.sqrt(sum(component * component for component in quaternion))
    if abs(norm - 1.0) > 1.0e-6:
        raise ValueError(f"{label}.quaternion_xyzw must be normalized")
    return AssemblyFrame(
        frame_id=_nonempty_text(document["frame_id"], f"{label}.frame_id"),
        parent_component=_nonempty_text(
            document["parent_component"], f"{label}.parent_component"
        ),
        translation_m=_finite_tuple(
            document["translation_m"], 3, f"{label}.translation_m"
        ),
        quaternion_xyzw=quaternion,
    )


def _parse_symmetry(value: Any, label: str) -> Optional[RotationalSymmetry]:
    if value is None:
        return None
    document = _mapping(value, label)
    _exact_keys(
        document,
        {
            "kind",
            "order",
            "axis_in_assembly_frame",
            "key_reference_direction_in_assembly_frame",
        },
        label,
    )
    kind = _nonempty_text(document["kind"], f"{label}.kind")
    if kind not in {"keyed", "cyclic", "continuous"}:
        raise ValueError(f"{label}.kind is unsupported")
    order_value = document["order"]
    order = None if order_value is None else int(order_value)
    if order_value is not None and (
        isinstance(order_value, bool) or order != order_value or order < 1
    ):
        raise ValueError(f"{label}.order must be a positive integer or null")
    axis = _finite_tuple(
        document["axis_in_assembly_frame"],
        3,
        f"{label}.axis_in_assembly_frame",
    )
    _unit_vector(axis, f"{label}.axis_in_assembly_frame")
    key = _optional_vector(
        document["key_reference_direction_in_assembly_frame"],
        3,
        f"{label}.key_reference_direction_in_assembly_frame",
    )
    if kind == "continuous":
        if order is not None or key is not None:
            raise ValueError(
                f"{label} continuous symmetry needs null order and key"
            )
    elif kind == "cyclic":
        if order is None or order < 2 or key is not None:
            raise ValueError(
                f"{label} cyclic symmetry needs order >= 2 and null key"
            )
    else:
        if order != 1 or key is None:
            raise ValueError(
                f"{label} keyed symmetry needs order 1 and a key direction"
            )
        _unit_vector(key, f"{label}.key_reference_direction")
        dot = sum(first * second for first, second in zip(axis, key))
        if abs(dot) > 1.0e-6:
            raise ValueError(f"{label} key direction must be normal to axis")
    return RotationalSymmetry(kind, order, axis, key)


def _parse_grasp_region(value: Any, label: str) -> GraspRegion:
    document = _mapping(value, label)
    _exact_keys(
        document,
        {
            "region_id",
            "parent_component",
            "shape",
            "axis_in_region_frame",
            "axial_bounds_m",
            "radius_bounds_m",
        },
        label,
    )
    shape = _nonempty_text(document["shape"], f"{label}.shape")
    if shape != "cylindrical_surface":
        raise ValueError(f"{label}.shape is unsupported")
    axis = _finite_tuple(
        document["axis_in_region_frame"],
        3,
        f"{label}.axis_in_region_frame",
    )
    _unit_vector(axis, f"{label}.axis_in_region_frame")
    axial = _finite_tuple(
        document["axial_bounds_m"], 2, f"{label}.axial_bounds_m"
    )
    radial = _finite_tuple(
        document["radius_bounds_m"], 2, f"{label}.radius_bounds_m"
    )
    if axial[0] >= axial[1]:
        raise ValueError(f"{label}.axial_bounds_m must have positive span")
    if radial[0] <= 0.0 or radial[0] > radial[1]:
        raise ValueError(f"{label}.radius_bounds_m is invalid")
    return GraspRegion(
        region_id=_nonempty_text(
            document["region_id"], f"{label}.region_id"
        ),
        parent_component=_nonempty_text(
            document["parent_component"], f"{label}.parent_component"
        ),
        shape=shape,
        axis_in_region_frame=axis,
        axial_bounds_m=axial,
        radius_bounds_m=radial,
    )


def _parse_model(value: Any, index: int) -> ConnectorModel:
    label = f"models[{index}]"
    document = _mapping(value, label)
    _exact_keys(
        document,
        {
            "model_id",
            "designation",
            "provenance",
            "deployment_role",
            "gender",
            "compatible_model_ids",
            "assembly_frame",
            "rotational_symmetry",
            "grasp_regions",
        },
        label,
    )
    try:
        role = ConnectorRole(document["deployment_role"])
        gender = ConnectorGender(document["gender"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} role or gender is invalid") from error
    compatible = tuple(
        _nonempty_text(item, f"{label}.compatible_model_ids")
        for item in document["compatible_model_ids"]
    )
    if not compatible or len(compatible) != len(set(compatible)):
        raise ValueError(
            f"{label}.compatible_model_ids must be non-empty and unique"
        )
    regions_value = document["grasp_regions"]
    if not isinstance(regions_value, list):
        raise ValueError(f"{label}.grasp_regions must be a list")
    regions = tuple(
        _parse_grasp_region(item, f"{label}.grasp_regions[{region_index}]")
        for region_index, item in enumerate(regions_value)
    )
    region_ids = [item.region_id for item in regions]
    if len(region_ids) != len(set(region_ids)):
        raise ValueError(f"{label}.grasp_regions IDs must be unique")
    return ConnectorModel(
        model_id=_nonempty_text(document["model_id"], f"{label}.model_id"),
        designation=_nonempty_text(
            document["designation"], f"{label}.designation"
        ),
        provenance=_nonempty_text(
            document["provenance"], f"{label}.provenance"
        ),
        deployment_role=role,
        gender=gender,
        compatible_model_ids=compatible,
        assembly_frame=_parse_frame(
            document["assembly_frame"], f"{label}.assembly_frame"
        ),
        rotational_symmetry=_parse_symmetry(
            document["rotational_symmetry"],
            f"{label}.rotational_symmetry",
        ),
        grasp_regions=regions,
    )


def _parse_insertion(value: Any, label: str) -> InsertionSpec:
    document = _mapping(value, label)
    _exact_keys(
        document,
        {
            "direction_in_fixed_assembly_frame",
            "minimum_engage_depth_m",
            "maximum_lateral_error_m",
            "maximum_angular_error_degrees",
            "maximum_key_error_degrees",
        },
        label,
    )
    direction = _optional_vector(
        document["direction_in_fixed_assembly_frame"],
        3,
        f"{label}.direction_in_fixed_assembly_frame",
    )
    if direction is not None:
        _unit_vector(direction, f"{label}.direction_in_fixed_assembly_frame")
    return InsertionSpec(
        direction_in_fixed_assembly_frame=direction,
        minimum_engage_depth_m=_optional_positive(
            document["minimum_engage_depth_m"],
            f"{label}.minimum_engage_depth_m",
        ),
        maximum_lateral_error_m=_optional_positive(
            document["maximum_lateral_error_m"],
            f"{label}.maximum_lateral_error_m",
        ),
        maximum_angular_error_degrees=_optional_positive(
            document["maximum_angular_error_degrees"],
            f"{label}.maximum_angular_error_degrees",
        ),
        maximum_key_error_degrees=_optional_positive(
            document["maximum_key_error_degrees"],
            f"{label}.maximum_key_error_degrees",
        ),
    )


def _optional_sign(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or value not in (-1, 1):
        raise ValueError(f"{label} must be -1 or 1")
    return int(value)


def _parse_fastening(value: Any, label: str) -> FasteningSpec:
    document = _mapping(value, label)
    _exact_keys(
        document,
        {
            "mechanism",
            "axis_in_fixed_assembly_frame",
            "tightening_sign_about_axis",
            "target_angle_degrees",
            "angle_tolerance_degrees",
            "lead_m_per_revolution",
        },
        label,
    )
    mechanism_value = document["mechanism"]
    mechanism = (
        None
        if mechanism_value is None
        else _nonempty_text(mechanism_value, f"{label}.mechanism")
    )
    if mechanism not in {None, "synthetic_helical_proxy"}:
        raise ValueError(f"{label}.mechanism is unsupported in registry v1")
    axis = _optional_vector(
        document["axis_in_fixed_assembly_frame"],
        3,
        f"{label}.axis_in_fixed_assembly_frame",
    )
    if axis is not None:
        _unit_vector(axis, f"{label}.axis_in_fixed_assembly_frame")
    target = _optional_positive(
        document["target_angle_degrees"],
        f"{label}.target_angle_degrees",
    )
    tolerance = _optional_positive(
        document["angle_tolerance_degrees"],
        f"{label}.angle_tolerance_degrees",
    )
    if target is not None and tolerance is not None and tolerance >= target:
        raise ValueError(f"{label} angle tolerance must be below target")
    return FasteningSpec(
        mechanism=mechanism,
        axis_in_fixed_assembly_frame=axis,
        tightening_sign_about_axis=_optional_sign(
            document["tightening_sign_about_axis"],
            f"{label}.tightening_sign_about_axis",
        ),
        target_angle_degrees=target,
        angle_tolerance_degrees=tolerance,
        lead_m_per_revolution=_optional_positive(
            document["lead_m_per_revolution"],
            f"{label}.lead_m_per_revolution",
        ),
    )


def _parse_safety(value: Any, label: str) -> ConnectorSafetyLimits:
    document = _mapping(value, label)
    fields = {
        "maximum_insertion_axial_force_n",
        "maximum_lateral_force_n",
        "maximum_bending_moment_nm",
        "maximum_tightening_torque_nm",
        "maximum_finger_base_torque_nm",
    }
    _exact_keys(document, fields, label)
    parsed = {
        name: _optional_positive(document[name], f"{label}.{name}")
        for name in fields
    }
    return ConnectorSafetyLimits(**parsed)


def _parse_profile(value: Any, index: int) -> AssemblyProfile:
    label = f"assembly_profiles[{index}]"
    document = _mapping(value, label)
    _exact_keys(
        document,
        {
            "profile_id",
            "enabled",
            "fidelity",
            "loose_model_id",
            "fixed_model_id",
            "insertion",
            "fastening",
            "safety_limits",
        },
        label,
    )
    return AssemblyProfile(
        profile_id=_nonempty_text(
            document["profile_id"], f"{label}.profile_id"
        ),
        enabled=_boolean(document["enabled"], f"{label}.enabled"),
        fidelity=_nonempty_text(
            document["fidelity"], f"{label}.fidelity"
        ),
        loose_model_id=_nonempty_text(
            document["loose_model_id"], f"{label}.loose_model_id"
        ),
        fixed_model_id=_nonempty_text(
            document["fixed_model_id"], f"{label}.fixed_model_id"
        ),
        insertion=_parse_insertion(
            document["insertion"], f"{label}.insertion"
        ),
        fastening=_parse_fastening(
            document["fastening"], f"{label}.fastening"
        ),
        safety_limits=_parse_safety(
            document["safety_limits"], f"{label}.safety_limits"
        ),
    )


def _profile_data_blockers(
    profile: AssemblyProfile,
    loose: ConnectorModel,
    fixed: ConnectorModel,
) -> list[str]:
    blockers = []
    for label, model in (("loose", loose), ("fixed", fixed)):
        if model.assembly_frame is None:
            blockers.append(f"{label}_assembly_frame_missing")
        if model.rotational_symmetry is None:
            blockers.append(f"{label}_rotational_symmetry_missing")
    if not loose.grasp_regions:
        blockers.append("loose_grasp_region_missing")
    for name, value in profile.insertion.__dict__.items():
        if value is None:
            blockers.append(f"insertion_{name}_missing")
    for name, value in profile.fastening.__dict__.items():
        if value is None:
            blockers.append(f"fastening_{name}_missing")
    for name, value in profile.safety_limits.__dict__.items():
        if value is None:
            blockers.append(f"safety_{name}_missing")
    return blockers


def _validate_registry_relations(registry: ConnectorModelRegistry) -> None:
    model_ids = [item.model_id for item in registry.models]
    profile_ids = [item.profile_id for item in registry.assembly_profiles]
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise ValueError("registry model IDs must be non-empty and unique")
    if not profile_ids or len(profile_ids) != len(set(profile_ids)):
        raise ValueError("assembly profile IDs must be non-empty and unique")
    known_models = set(model_ids)
    for model in registry.models:
        if not model.model_id.startswith("synthetic_"):
            raise ValueError("registry v1 accepts only synthetic model IDs")
        if model.provenance != REGISTRY_SCOPE:
            raise ValueError("registry v1 model provenance must be synthetic")
        unknown = set(model.compatible_model_ids) - known_models
        if unknown:
            raise ValueError(
                f"model {model.model_id!r} has unknown mates: "
                f"{sorted(unknown)}"
            )
    for profile in registry.assembly_profiles:
        if profile.fidelity != REGISTRY_SCOPE:
            raise ValueError("registry v1 profiles must remain synthetic")
        loose = registry.model(profile.loose_model_id)
        fixed = registry.model(profile.fixed_model_id)
        if loose.deployment_role is not ConnectorRole.LOOSE:
            raise ValueError("assembly loose model must have role loose")
        if fixed.deployment_role is not ConnectorRole.FIXED:
            raise ValueError("assembly fixed model must have role fixed")
        if loose.gender is not ConnectorGender.MALE:
            raise ValueError(
                "assembly loose model must be male in registry v1"
            )
        if fixed.gender is not ConnectorGender.FEMALE:
            raise ValueError(
                "assembly fixed model must be female in registry v1"
            )
        if fixed.model_id not in loose.compatible_model_ids:
            raise ValueError("loose model does not declare the fixed mate")
        if loose.model_id not in fixed.compatible_model_ids:
            raise ValueError("fixed model does not declare the loose mate")
        direction = profile.insertion.direction_in_fixed_assembly_frame
        axis = profile.fastening.axis_in_fixed_assembly_frame
        if direction is not None and axis is not None:
            dot = sum(first * second for first, second in zip(direction, axis))
            if abs(abs(dot) - 1.0) > 1.0e-6:
                raise ValueError(
                    "insertion direction and fastening axis must be collinear"
                )
        if profile.enabled:
            blockers = _profile_data_blockers(profile, loose, fixed)
            if blockers:
                joined = ", ".join(blockers)
                raise ValueError(
                    f"enabled profile {profile.profile_id!r} is incomplete: "
                    f"{joined}"
                )
    if registry.enabled:
        enabled_profiles = [
            item for item in registry.assembly_profiles if item.enabled
        ]
        if not enabled_profiles:
            raise ValueError("enabled registry needs an enabled profile")


def load_connector_model_registry(
    config_path: str | Path,
) -> ConnectorModelRegistry:
    """Load the v1 synthetic registry without implicit defaults."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "registry")
    _exact_keys(
        document,
        {
            "schema_version",
            "registry_enabled",
            "scope",
            "models",
            "assembly_profiles",
        },
        "registry",
    )
    schema_version = _nonempty_text(
        document["schema_version"], "registry.schema_version"
    )
    if schema_version != REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"unsupported connector registry: {schema_version!r}")
    scope = _nonempty_text(document["scope"], "registry.scope")
    if scope != REGISTRY_SCOPE:
        raise ValueError(
            "registry v1 is restricted to the synthetic curriculum"
        )
    models_value = document["models"]
    profiles_value = document["assembly_profiles"]
    if not isinstance(models_value, list):
        raise ValueError("registry.models must be a list")
    if not isinstance(profiles_value, list):
        raise ValueError("registry.assembly_profiles must be a list")
    registry = ConnectorModelRegistry(
        schema_version=schema_version,
        enabled=_boolean(
            document["registry_enabled"], "registry.registry_enabled"
        ),
        scope=scope,
        models=tuple(
            _parse_model(item, index)
            for index, item in enumerate(models_value)
        ),
        assembly_profiles=tuple(
            _parse_profile(item, index)
            for index, item in enumerate(profiles_value)
        ),
    )
    _validate_registry_relations(registry)
    return registry
