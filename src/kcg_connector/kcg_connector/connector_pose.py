"""Strict 6D connector-pose contract with no ROS or simulator imports.

The same observation schema accepts ``sim_ground_truth`` today and ``vision``
later.  Accepting the latter source only defines its interface; it does not
implement a detector, calibration, synchronization, or uncertainty estimator.

Transforms use ``parent_T_child`` and quaternions use ``[x, y, z, w]``.  The
6x6 observation covariance tangent order is
``[x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]`` in ``frame_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


POSE_CONTRACT_SCHEMA_VERSION = "kcg_connector_pose_contract_v1"
POSE_OBSERVATION_SCHEMA_VERSION = "kcg_connector_pose_observation_v1"
OBJECT_TARGET_TRANSFORM_SCHEMA_VERSION = (
    "kcg_connector_object_target_transform_v1"
)
DEFAULT_POSE_CONTRACT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "connector_pose_observation_v1.yaml"
)

_QUATERNION_CONVENTION = "xyzw"
_TRANSFORM_CONVENTION = "parent_T_child"
_COVARIANCE_TANGENT_ORDER = (
    "x_m",
    "y_m",
    "z_m",
    "rx_rad",
    "ry_rad",
    "rz_rad",
)


class ConnectorPoseRole(str, Enum):
    """Physical role represented by an observation."""

    LOOSE_PLUG = "loose_plug"
    FIXED_RECEPTACLE = "fixed_receptacle"


class ConnectorPoseSource(str, Enum):
    """Permitted producers of the shared pose schema."""

    SIM_GROUND_TRUTH = "sim_ground_truth"
    VISION = "vision"


class ObjectTargetKind(str, Enum):
    """Versioned target frame derived from a registered object frame."""

    GRASP = "grasp"
    ASSEMBLY = "assembly"


@dataclass(frozen=True)
class PoseValidationPolicy:
    """Fail-closed numerical and timing gates."""

    maximum_age_s: float
    maximum_future_offset_s: float
    maximum_pair_timestamp_skew_s: float
    minimum_confidence: float
    quaternion_norm_tolerance: float
    covariance_symmetry_tolerance: float
    covariance_psd_tolerance: float


@dataclass(frozen=True)
class PoseModelRegistration:
    """Identity, role, frame, symmetry, and compatible mates for one model."""

    model_id: str
    role: ConnectorPoseRole
    object_frame_id: str
    symmetry_class: str
    compatible_model_ids: tuple[str, ...]


@dataclass(frozen=True)
class ObjectTargetTransform:
    """A measured/versioned ``object_T_target`` transform."""

    schema_version: str
    transform_id: str
    model_id: str
    role: ConnectorPoseRole
    target_kind: ObjectTargetKind
    parent_object_frame_id: str
    child_target_frame_id: str
    translation_xyz_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class ConnectorPoseContract:
    """Loaded v1 contract, model registry, and optional calibrated targets."""

    schema_version: str
    observation_schema_version: str
    object_target_transform_schema_version: str
    policy: PoseValidationPolicy
    model_registry: tuple[PoseModelRegistration, ...]
    object_target_transforms: tuple[ObjectTargetTransform, ...]

    def model(self, model_id: str) -> PoseModelRegistration:
        """Return one exact model match or fail instead of guessing."""
        matches = [
            model for model in self.model_registry
            if model.model_id == model_id
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown pose model: {model_id!r}")
        return matches[0]

    def transform(self, transform_id: str) -> ObjectTargetTransform:
        """Return one calibrated target transform or fail closed."""
        matches = [
            transform for transform in self.object_target_transforms
            if transform.transform_id == transform_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"unknown object target transform: {transform_id!r}"
            )
        return matches[0]


@dataclass(frozen=True)
class ConnectorPoseObservation:
    """Validated and canonicalized object pose observation."""

    schema_version: str
    model_id: str
    role: ConnectorPoseRole
    timestamp_s: float
    frame_id: str
    position_xyz_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    covariance_6x6: tuple[tuple[float, ...], ...]
    confidence: float
    symmetry_class: str
    source: ConnectorPoseSource


@dataclass(frozen=True)
class ConnectorPosePair:
    """Canonical loose/fixed observations that passed pair gates."""

    loose_plug: ConnectorPoseObservation
    fixed_receptacle: ConnectorPoseObservation


@dataclass(frozen=True)
class ResolvedTargetPose:
    """Pose obtained by composing ``frame_T_object * object_T_target``."""

    transform_id: str
    model_id: str
    role: ConnectorPoseRole
    target_kind: ObjectTargetKind
    timestamp_s: float
    frame_id: str
    target_frame_id: str
    position_xyz_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    confidence: float
    source: ConnectorPoseSource


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


def _text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(f"{label} must be non-empty unpadded text")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _finite_vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {size} finite numbers")
    try:
        items = tuple(value)
    except TypeError as error:
        raise ValueError(
            f"{label} must contain {size} finite numbers"
        ) from error
    if len(items) != size:
        raise ValueError(f"{label} must contain {size} finite numbers")
    return tuple(
        _finite_float(item, f"{label}[{index}]")
        for index, item in enumerate(items)
    )


def _positive(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _closed_unit_interval(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return result


def _canonical_unit_quaternion(
    value: Any,
    tolerance: float,
    label: str,
) -> tuple[float, float, float, float]:
    quaternion = _finite_vector(value, 4, label)
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm == 0.0 or abs(norm - 1.0) > tolerance:
        raise ValueError(f"{label} must be normalized within tolerance")
    normalized = tuple(component / norm for component in quaternion)
    x_value, y_value, z_value, w_value = normalized
    vector_first_nonzero = next(
        (
            component
            for component in (x_value, y_value, z_value)
            if component != 0.0
        ),
        0.0,
    )
    if w_value < 0.0 or (
        w_value == 0.0 and vector_first_nonzero < 0.0
    ):
        normalized = tuple(-component for component in normalized)
    return normalized  # type: ignore[return-value]


def _minimum_symmetric_eigenvalue(
    matrix: tuple[tuple[float, ...], ...]
) -> float:
    """Return the smallest eigenvalue using a pure-Python Jacobi sweep."""
    values = [list(row) for row in matrix]
    size = len(values)
    for _ in range(100 * size * size):
        pivot_row = 0
        pivot_column = 1
        maximum = 0.0
        for row in range(size):
            for column in range(row + 1, size):
                candidate = abs(values[row][column])
                if candidate > maximum:
                    maximum = candidate
                    pivot_row = row
                    pivot_column = column
        scale = max(
            1.0,
            max(abs(values[index][index]) for index in range(size)),
        )
        if maximum <= 1.0e-15 * scale:
            break
        first = values[pivot_row][pivot_row]
        second = values[pivot_column][pivot_column]
        off_diagonal = values[pivot_row][pivot_column]
        tau = (second - first) / (2.0 * off_diagonal)
        tangent = math.copysign(
            1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)),
            tau,
        )
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        for index in range(size):
            if index in (pivot_row, pivot_column):
                continue
            row_value = values[index][pivot_row]
            column_value = values[index][pivot_column]
            rotated_row = cosine * row_value - sine * column_value
            rotated_column = sine * row_value + cosine * column_value
            values[index][pivot_row] = rotated_row
            values[pivot_row][index] = rotated_row
            values[index][pivot_column] = rotated_column
            values[pivot_column][index] = rotated_column
        values[pivot_row][pivot_row] = (
            cosine * cosine * first
            - 2.0 * sine * cosine * off_diagonal
            + sine * sine * second
        )
        values[pivot_column][pivot_column] = (
            sine * sine * first
            + 2.0 * sine * cosine * off_diagonal
            + cosine * cosine * second
        )
        values[pivot_row][pivot_column] = 0.0
        values[pivot_column][pivot_row] = 0.0
    return min(values[index][index] for index in range(size))


def _covariance(
    value: Any,
    symmetry_tolerance: float,
    psd_tolerance: float,
    label: str,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a 6x6 finite matrix")
    try:
        rows = tuple(value)
    except TypeError as error:
        raise ValueError(f"{label} must be a 6x6 finite matrix") from error
    if len(rows) != 6:
        raise ValueError(f"{label} must be a 6x6 finite matrix")
    matrix = tuple(
        _finite_vector(row, 6, f"{label}[{index}]")
        for index, row in enumerate(rows)
    )
    for row in range(6):
        for column in range(row + 1, 6):
            if (
                abs(matrix[row][column] - matrix[column][row])
                > symmetry_tolerance
            ):
                raise ValueError(f"{label} must be symmetric")
    symmetric = tuple(
        tuple(
            0.5 * (matrix[row][column] + matrix[column][row])
            for column in range(6)
        )
        for row in range(6)
    )
    if _minimum_symmetric_eigenvalue(symmetric) < -psd_tolerance:
        raise ValueError(f"{label} must be positive semidefinite")
    return symmetric


def _parse_role(value: Any, label: str) -> ConnectorPoseRole:
    try:
        return ConnectorPoseRole(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} has an unsupported role") from error


def _parse_source(value: Any, label: str) -> ConnectorPoseSource:
    try:
        return ConnectorPoseSource(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} has an unsupported source") from error


def _parse_target_kind(value: Any, label: str) -> ObjectTargetKind:
    try:
        return ObjectTargetKind(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} has an unsupported target kind") from error


def _parse_policy(value: Any) -> PoseValidationPolicy:
    label = "pose_contract.policy"
    document = _mapping(value, label)
    fields = {
        "maximum_age_s",
        "maximum_future_offset_s",
        "maximum_pair_timestamp_skew_s",
        "minimum_confidence",
        "quaternion_norm_tolerance",
        "covariance_symmetry_tolerance",
        "covariance_psd_tolerance",
    }
    _exact_keys(document, fields, label)
    return PoseValidationPolicy(
        maximum_age_s=_positive(
            document["maximum_age_s"], f"{label}.maximum_age_s"
        ),
        maximum_future_offset_s=_nonnegative(
            document["maximum_future_offset_s"],
            f"{label}.maximum_future_offset_s",
        ),
        maximum_pair_timestamp_skew_s=_nonnegative(
            document["maximum_pair_timestamp_skew_s"],
            f"{label}.maximum_pair_timestamp_skew_s",
        ),
        minimum_confidence=_closed_unit_interval(
            document["minimum_confidence"],
            f"{label}.minimum_confidence",
        ),
        quaternion_norm_tolerance=_positive(
            document["quaternion_norm_tolerance"],
            f"{label}.quaternion_norm_tolerance",
        ),
        covariance_symmetry_tolerance=_nonnegative(
            document["covariance_symmetry_tolerance"],
            f"{label}.covariance_symmetry_tolerance",
        ),
        covariance_psd_tolerance=_nonnegative(
            document["covariance_psd_tolerance"],
            f"{label}.covariance_psd_tolerance",
        ),
    )


def _parse_model(value: Any, index: int) -> PoseModelRegistration:
    label = f"pose_contract.model_registry[{index}]"
    document = _mapping(value, label)
    _exact_keys(
        document,
        {
            "model_id",
            "role",
            "object_frame_id",
            "symmetry_class",
            "compatible_model_ids",
        },
        label,
    )
    compatible_value = document["compatible_model_ids"]
    if not isinstance(compatible_value, list):
        raise ValueError(f"{label}.compatible_model_ids must be a list")
    compatible = tuple(
        _text(item, f"{label}.compatible_model_ids[{item_index}]")
        for item_index, item in enumerate(compatible_value)
    )
    if len(compatible) != len(set(compatible)):
        raise ValueError(f"{label}.compatible_model_ids must be unique")
    return PoseModelRegistration(
        model_id=_text(document["model_id"], f"{label}.model_id"),
        role=_parse_role(document["role"], f"{label}.role"),
        object_frame_id=_text(
            document["object_frame_id"], f"{label}.object_frame_id"
        ),
        symmetry_class=_text(
            document["symmetry_class"], f"{label}.symmetry_class"
        ),
        compatible_model_ids=compatible,
    )


def _parse_transform(
    value: Any,
    models: Mapping[str, PoseModelRegistration],
    quaternion_tolerance: float,
    label: str,
) -> ObjectTargetTransform:
    document = _mapping(value, label)
    fields = {
        "schema_version",
        "transform_id",
        "model_id",
        "role",
        "target_kind",
        "parent_object_frame_id",
        "child_target_frame_id",
        "translation_xyz_m",
        "quaternion_xyzw",
    }
    _exact_keys(document, fields, label)
    schema_version = _text(
        document["schema_version"], f"{label}.schema_version"
    )
    if schema_version != OBJECT_TARGET_TRANSFORM_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported object target transform: {schema_version!r}"
        )
    model_id = _text(document["model_id"], f"{label}.model_id")
    if model_id not in models:
        raise ValueError(f"{label} references unknown model {model_id!r}")
    model = models[model_id]
    role = _parse_role(document["role"], f"{label}.role")
    if role is not model.role:
        raise ValueError(f"{label}.role does not match model registry")
    target_kind = _parse_target_kind(
        document["target_kind"], f"{label}.target_kind"
    )
    if (
        role is ConnectorPoseRole.FIXED_RECEPTACLE
        and target_kind is ObjectTargetKind.GRASP
    ):
        raise ValueError("fixed receptacle cannot register a grasp target")
    parent_frame = _text(
        document["parent_object_frame_id"],
        f"{label}.parent_object_frame_id",
    )
    if parent_frame != model.object_frame_id:
        raise ValueError(
            f"{label}.parent_object_frame_id does not match model registry"
        )
    return ObjectTargetTransform(
        schema_version=schema_version,
        transform_id=_text(
            document["transform_id"], f"{label}.transform_id"
        ),
        model_id=model_id,
        role=role,
        target_kind=target_kind,
        parent_object_frame_id=parent_frame,
        child_target_frame_id=_text(
            document["child_target_frame_id"],
            f"{label}.child_target_frame_id",
        ),
        translation_xyz_m=_finite_vector(
            document["translation_xyz_m"],
            3,
            f"{label}.translation_xyz_m",
        ),
        quaternion_xyzw=_canonical_unit_quaternion(
            document["quaternion_xyzw"],
            quaternion_tolerance,
            f"{label}.quaternion_xyzw",
        ),
    )


def _validate_registry(
    models: tuple[PoseModelRegistration, ...],
) -> dict[str, PoseModelRegistration]:
    if not models:
        raise ValueError("pose contract model registry must not be empty")
    model_ids = [model.model_id for model in models]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("pose contract model IDs must be unique")
    registry = {model.model_id: model for model in models}
    for model in models:
        if not model.compatible_model_ids:
            raise ValueError(
                f"pose model {model.model_id!r} needs a compatible mate"
            )
        for mate_id in model.compatible_model_ids:
            if mate_id not in registry:
                raise ValueError(
                    f"pose model {model.model_id!r} has unknown mate "
                    f"{mate_id!r}"
                )
            mate = registry[mate_id]
            if mate.role is model.role:
                raise ValueError(
                    "compatible pose models must have opposite roles"
                )
            if model.model_id not in mate.compatible_model_ids:
                raise ValueError("pose model compatibility must be reciprocal")
    return registry


def load_connector_pose_contract(
    config_path: str | Path = DEFAULT_POSE_CONTRACT_CONFIG_PATH,
) -> ConnectorPoseContract:
    """Load and strictly validate the simulator-neutral v1 contract."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "pose_contract")
    fields = {
        "schema_version",
        "observation_schema_version",
        "object_target_transform_schema_version",
        "quaternion_convention",
        "transform_convention",
        "covariance_tangent_order",
        "policy",
        "model_registry",
        "object_target_transforms",
    }
    _exact_keys(document, fields, "pose_contract")
    schema_version = _text(
        document["schema_version"], "pose_contract.schema_version"
    )
    if schema_version != POSE_CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported pose contract: {schema_version!r}")
    observation_version = _text(
        document["observation_schema_version"],
        "pose_contract.observation_schema_version",
    )
    if observation_version != POSE_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported pose observation: {observation_version!r}"
        )
    transform_version = _text(
        document["object_target_transform_schema_version"],
        "pose_contract.object_target_transform_schema_version",
    )
    if transform_version != OBJECT_TARGET_TRANSFORM_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported object target transform: {transform_version!r}"
        )
    if document["quaternion_convention"] != _QUATERNION_CONVENTION:
        raise ValueError("pose contract quaternion convention must be xyzw")
    if document["transform_convention"] != _TRANSFORM_CONVENTION:
        raise ValueError(
            "pose contract transform convention must be parent_T_child"
        )
    covariance_order = document["covariance_tangent_order"]
    if not isinstance(covariance_order, list) or tuple(
        covariance_order
    ) != _COVARIANCE_TANGENT_ORDER:
        raise ValueError(
            "pose contract covariance tangent order is unsupported"
        )
    policy = _parse_policy(document["policy"])
    model_values = document["model_registry"]
    if not isinstance(model_values, list):
        raise ValueError("pose_contract.model_registry must be a list")
    models = tuple(
        _parse_model(value, index)
        for index, value in enumerate(model_values)
    )
    model_lookup = _validate_registry(models)
    transform_values = document["object_target_transforms"]
    if not isinstance(transform_values, list):
        raise ValueError(
            "pose_contract.object_target_transforms must be a list"
        )
    transforms = tuple(
        _parse_transform(
            value,
            model_lookup,
            policy.quaternion_norm_tolerance,
            f"pose_contract.object_target_transforms[{index}]",
        )
        for index, value in enumerate(transform_values)
    )
    transform_ids = [transform.transform_id for transform in transforms]
    if len(transform_ids) != len(set(transform_ids)):
        raise ValueError("object target transform IDs must be unique")
    return ConnectorPoseContract(
        schema_version=schema_version,
        observation_schema_version=observation_version,
        object_target_transform_schema_version=transform_version,
        policy=policy,
        model_registry=models,
        object_target_transforms=transforms,
    )


def _observation_mapping(
    observation: ConnectorPoseObservation,
) -> dict[str, Any]:
    return {
        "schema_version": observation.schema_version,
        "model_id": observation.model_id,
        "role": observation.role.value,
        "timestamp_s": observation.timestamp_s,
        "frame_id": observation.frame_id,
        "position_xyz_m": observation.position_xyz_m,
        "quaternion_xyzw": observation.quaternion_xyzw,
        "covariance_6x6": observation.covariance_6x6,
        "confidence": observation.confidence,
        "symmetry_class": observation.symmetry_class,
        "source": observation.source.value,
    }


def parse_connector_pose_observation(
    value: Any,
    contract: ConnectorPoseContract,
    *,
    now_s: float,
) -> ConnectorPoseObservation:
    """Parse one exact observation and enforce registry/timing/confidence."""
    label = "connector_pose_observation"
    document = _mapping(value, label)
    fields = {
        "schema_version",
        "model_id",
        "role",
        "timestamp_s",
        "frame_id",
        "position_xyz_m",
        "quaternion_xyzw",
        "covariance_6x6",
        "confidence",
        "symmetry_class",
        "source",
    }
    _exact_keys(document, fields, label)
    schema_version = _text(
        document["schema_version"], f"{label}.schema_version"
    )
    if schema_version != contract.observation_schema_version:
        raise ValueError(f"unsupported pose observation: {schema_version!r}")
    model_id = _text(document["model_id"], f"{label}.model_id")
    model = contract.model(model_id)
    role = _parse_role(document["role"], f"{label}.role")
    if role is not model.role:
        raise ValueError("observation role does not match model registry")
    symmetry_class = _text(
        document["symmetry_class"], f"{label}.symmetry_class"
    )
    if symmetry_class != model.symmetry_class:
        raise ValueError(
            "observation symmetry class does not match model registry"
        )
    timestamp = _finite_float(
        document["timestamp_s"], f"{label}.timestamp_s"
    )
    current_time = _finite_float(now_s, "now_s")
    age = current_time - timestamp
    if age > contract.policy.maximum_age_s:
        raise ValueError("connector pose observation is stale")
    if age < -contract.policy.maximum_future_offset_s:
        raise ValueError("connector pose observation is too far in the future")
    confidence = _closed_unit_interval(
        document["confidence"], f"{label}.confidence"
    )
    if confidence < contract.policy.minimum_confidence:
        raise ValueError("connector pose observation confidence is too low")
    return ConnectorPoseObservation(
        schema_version=schema_version,
        model_id=model_id,
        role=role,
        timestamp_s=timestamp,
        frame_id=_text(document["frame_id"], f"{label}.frame_id"),
        position_xyz_m=_finite_vector(
            document["position_xyz_m"], 3, f"{label}.position_xyz_m"
        ),
        quaternion_xyzw=_canonical_unit_quaternion(
            document["quaternion_xyzw"],
            contract.policy.quaternion_norm_tolerance,
            f"{label}.quaternion_xyzw",
        ),
        covariance_6x6=_covariance(
            document["covariance_6x6"],
            contract.policy.covariance_symmetry_tolerance,
            contract.policy.covariance_psd_tolerance,
            f"{label}.covariance_6x6",
        ),
        confidence=confidence,
        symmetry_class=symmetry_class,
        source=_parse_source(document["source"], f"{label}.source"),
    )


def pair_connector_pose_observations(
    first: ConnectorPoseObservation,
    second: ConnectorPoseObservation,
    contract: ConnectorPoseContract,
    *,
    now_s: float,
) -> ConnectorPosePair:
    """Validate and pair exactly one compatible loose and fixed endpoint."""
    validated = tuple(
        parse_connector_pose_observation(
            _observation_mapping(observation), contract, now_s=now_s
        )
        for observation in (first, second)
    )
    by_role = {observation.role: observation for observation in validated}
    required_roles = {
        ConnectorPoseRole.LOOSE_PLUG,
        ConnectorPoseRole.FIXED_RECEPTACLE,
    }
    if set(by_role) != required_roles or len(by_role) != 2:
        raise ValueError(
            "pose pair needs one loose plug and one fixed receptacle"
        )
    loose = by_role[ConnectorPoseRole.LOOSE_PLUG]
    fixed = by_role[ConnectorPoseRole.FIXED_RECEPTACLE]
    loose_model = contract.model(loose.model_id)
    fixed_model = contract.model(fixed.model_id)
    if (
        fixed.model_id not in loose_model.compatible_model_ids
        or loose.model_id not in fixed_model.compatible_model_ids
    ):
        raise ValueError("connector pose models are not compatible mates")
    if loose.frame_id != fixed.frame_id:
        raise ValueError("connector pose pair must use one common frame")
    skew = abs(loose.timestamp_s - fixed.timestamp_s)
    if skew > contract.policy.maximum_pair_timestamp_skew_s:
        raise ValueError("connector pose pair timestamps are too far apart")
    return ConnectorPosePair(
        loose_plug=loose,
        fixed_receptacle=fixed,
    )


def parse_object_target_transform(
    value: Any,
    contract: ConnectorPoseContract,
) -> ObjectTargetTransform:
    """Parse an explicit versioned transform against the model registry."""
    models = {model.model_id: model for model in contract.model_registry}
    return _parse_transform(
        value,
        models,
        contract.policy.quaternion_norm_tolerance,
        "object_target_transform",
    )


def _transform_mapping(transform: ObjectTargetTransform) -> dict[str, Any]:
    return {
        "schema_version": transform.schema_version,
        "transform_id": transform.transform_id,
        "model_id": transform.model_id,
        "role": transform.role.value,
        "target_kind": transform.target_kind.value,
        "parent_object_frame_id": transform.parent_object_frame_id,
        "child_target_frame_id": transform.child_target_frame_id,
        "translation_xyz_m": transform.translation_xyz_m,
        "quaternion_xyzw": transform.quaternion_xyzw,
    }


def _quaternion_multiply_xyzw(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float, float]:
    first_x, first_y, first_z, first_w = first
    second_x, second_y, second_z, second_w = second
    return (
        first_w * second_x
        + first_x * second_w
        + first_y * second_z
        - first_z * second_y,
        first_w * second_y
        - first_x * second_z
        + first_y * second_w
        + first_z * second_x,
        first_w * second_z
        + first_x * second_y
        - first_y * second_x
        + first_z * second_w,
        first_w * second_w
        - first_x * second_x
        - first_y * second_y
        - first_z * second_z,
    )


def _rotate_vector(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    qx_value, qy_value, qz_value, qw_value = quaternion
    vx_value, vy_value, vz_value = vector
    cross_x = qy_value * vz_value - qz_value * vy_value
    cross_y = qz_value * vx_value - qx_value * vz_value
    cross_z = qx_value * vy_value - qy_value * vx_value
    second_cross_x = qy_value * cross_z - qz_value * cross_y
    second_cross_y = qz_value * cross_x - qx_value * cross_z
    second_cross_z = qx_value * cross_y - qy_value * cross_x
    return (
        vx_value + 2.0 * (qw_value * cross_x + second_cross_x),
        vy_value + 2.0 * (qw_value * cross_y + second_cross_y),
        vz_value + 2.0 * (qw_value * cross_z + second_cross_z),
    )


def resolve_object_target_pose(
    observation: ConnectorPoseObservation,
    object_to_target: ObjectTargetTransform,
    contract: ConnectorPoseContract,
    *,
    now_s: float,
) -> ResolvedTargetPose:
    """Compose a fresh object observation with one explicit target transform.

    The returned target intentionally does not claim a transformed covariance;
    uncertainty propagation depends on a declared perturbation convention and
    calibrated transform uncertainty, neither of which is guessed here.
    """
    valid_observation = parse_connector_pose_observation(
        _observation_mapping(observation), contract, now_s=now_s
    )
    valid_transform = parse_object_target_transform(
        _transform_mapping(object_to_target), contract
    )
    if valid_observation.model_id != valid_transform.model_id:
        raise ValueError("observation and object target model IDs differ")
    if valid_observation.role is not valid_transform.role:
        raise ValueError("observation and object target roles differ")
    rotated_translation = _rotate_vector(
        valid_observation.quaternion_xyzw,
        valid_transform.translation_xyz_m,
    )
    position = tuple(
        first + second
        for first, second in zip(
            valid_observation.position_xyz_m,
            rotated_translation,
        )
    )
    quaternion = _quaternion_multiply_xyzw(
        valid_observation.quaternion_xyzw,
        valid_transform.quaternion_xyzw,
    )
    canonical_quaternion = _canonical_unit_quaternion(
        quaternion,
        max(contract.policy.quaternion_norm_tolerance, 1.0e-12),
        "resolved_target.quaternion_xyzw",
    )
    return ResolvedTargetPose(
        transform_id=valid_transform.transform_id,
        model_id=valid_observation.model_id,
        role=valid_observation.role,
        target_kind=valid_transform.target_kind,
        timestamp_s=valid_observation.timestamp_s,
        frame_id=valid_observation.frame_id,
        target_frame_id=valid_transform.child_target_frame_id,
        position_xyz_m=position,  # type: ignore[arg-type]
        quaternion_xyzw=canonical_quaternion,
        confidence=valid_observation.confidence,
        source=valid_observation.source,
    )
