"""Pure-CPU symmetry-aware pose evaluation and PoseProvider adapter.

This module reuses :mod:`connector_pose` and :mod:`pose_provider` for numeric,
timing, provenance, and publication checks.  It adds only the missing
object-symmetry sidecar and accuracy metrics.  The checked-in v1 contract is
disabled and can never authorize vision control.

The current shared pose registry labels both D38999 proxy models as
``keyed_order_1`` even though the generated proxy has no unique key and is at
least order-2 symmetric.  Consequently the current adapter returns a strict
diagnostic-only ``PoseProviderSample`` with ``pair=None``.  A future versioned
registry must represent the declared symmetry before pair publication can be
enabled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

from kcg_connector.connector_pose import (
    ConnectorPoseContract,
    load_connector_pose_contract,
    parse_connector_pose_observation,
)
from kcg_connector.d38999_foundationpose_bootstrap import (
    load_foundationpose_bootstrap_contract,
    validate_obj_document,
)
from kcg_connector.pose_provider import (
    POSE_PROVIDER_SAMPLE_SCHEMA_VERSION,
    PoseProviderPurpose,
    PoseProviderSample,
    parse_pose_provider_sample,
)


SCHEMA_VERSION = "kcg_d38999_symmetry_pose_readiness_v1"
CANDIDATE_SCHEMA_VERSION = "kcg_foundationpose_candidate_pair_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "d38999_symmetry_pose_readiness_v1.yaml"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_ENDPOINT_ROLES = ("loose_plug", "fixed_receptacle")


@dataclass(frozen=True)
class BoundFile:
    """One repository-relative content-addressed input."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class SymmetrySpec:
    """Discrete rotation group about the object's local +Z axis."""

    object_id: str
    model_id: str
    mesh_input: str
    rotational_symmetry_order: int
    equivalent_yaw_period_rad: float
    unique_key_geometry_present: bool
    keyed_yaw_observable: bool


@dataclass(frozen=True)
class AccuracyThresholds:
    translation_error_3d_max_m: float
    axis_error_max_rad: float
    orientation_error_modulo_symmetry_max_rad: float
    current_xy_error_max_m: float
    required_anchor_ids: tuple[str, ...]
    minimum_anchor_count: int
    current_evidence: dict[str, bool]


@dataclass(frozen=True)
class PoseProviderAdapterPolicy:
    candidate_schema_version: str
    provider_id: str
    provider_version: str
    allowed_purposes: tuple[PoseProviderPurpose, ...]
    required_clock_domain: str
    required_control_frame: str
    required_calibration_sha256: str
    required_registry_symmetry_class: str
    current_registry_symmetry_class: str
    pair_publication_enabled: bool
    full_accuracy_evidence_qualified: bool
    control_purpose_allowed: bool
    control_authorization_possible: bool


@dataclass(frozen=True)
class SymmetryPoseReadinessContract:
    schema_version: str
    enabled: bool
    status: str
    inputs: dict[str, BoundFile]
    symmetry_axis_xyz: tuple[float, float, float]
    symmetry: dict[str, SymmetrySpec]
    accuracy: AccuracyThresholds
    adapter: PoseProviderAdapterPolicy
    boundaries: dict[str, bool]


@dataclass(frozen=True)
class PoseModuloSymmetryResult:
    """Accuracy result that cannot confuse equivalent yaw with keyed yaw."""

    translation_error_m: float
    axis_error_rad: float
    orientation_error_modulo_symmetry_rad: float
    selected_symmetry_index: int
    translation_passed: bool
    axis_passed: bool
    orientation_modulo_symmetry_passed: bool
    symmetry_aware_pose_passed: bool
    unique_key_yaw_observable: bool
    unique_key_yaw_error_rad: float | None
    unique_key_yaw_rejection_reason: str | None
    vision_control_qualified: bool


@dataclass(frozen=True)
class FoundationPoseCandidateEndpoint:
    role: str
    model_id: str
    mesh_id: str
    mesh_sha256: str
    position_xyz_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    covariance_6x6: tuple[tuple[float, ...], ...]
    confidence: float


@dataclass(frozen=True)
class FoundationPoseCandidatePair:
    capture_id: str
    inference_id: str
    timestamp_s: float
    clock_domain: str
    control_frame: str
    calibration_sha256: str
    model_version: str
    refine_model_sha256: str
    score_model_sha256: str
    endpoints: dict[str, FoundationPoseCandidateEndpoint]


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


def _identifier(value: Any, label: str) -> str:
    result = _text(value, label)
    if not _IDENTIFIER.fullmatch(result):
        raise ValueError(f"{label} must be a stable identifier")
    return result


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must contain {size} finite numbers")
    if len(value) != size:
        raise ValueError(f"{label} must contain {size} finite numbers")
    return tuple(
        _finite(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


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


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} must be non-empty and unique")
    return result


def _unit_quaternion(
    value: Any, label: str
) -> tuple[float, float, float, float]:
    quaternion = _vector(value, 4, label)
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm == 0.0 or abs(norm - 1.0) > 1.0e-6:
        raise ValueError(f"{label} must be a unit quaternion")
    result = tuple(item / norm for item in quaternion)
    if result[3] < 0.0:
        result = tuple(-item for item in result)
    return result  # type: ignore[return-value]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_symmetry_pose_readiness_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> SymmetryPoseReadinessContract:
    """Load the disabled contract without touching runtime state."""

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
            "symmetry",
            "accuracy_schema",
            "pose_provider_adapter",
            "boundaries",
        },
        "document",
    )
    schema = _text(document["schema_version"], "schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    enabled = _boolean(document["enabled"], "enabled")
    if enabled:
        raise ValueError("symmetry pose readiness must remain disabled")

    input_document = _mapping(document["inputs"], "inputs")
    expected_inputs = {
        "multisite_rgbd_report",
        "foundationpose_bootstrap",
        "pose_contract",
        "loose_body_obj",
        "coupling_nut_obj",
        "fixed_receptacle_obj",
    }
    if set(input_document) != expected_inputs:
        raise ValueError("symmetry readiness input set differs")
    inputs: dict[str, BoundFile] = {}
    for name, raw in input_document.items():
        item = _mapping(raw, f"inputs.{name}")
        _exact(item, {"path", "sha256"}, f"inputs.{name}")
        inputs[name] = BoundFile(
            path=_relative_path(item["path"], f"inputs.{name}.path"),
            sha256=_sha(item["sha256"], f"inputs.{name}.sha256"),
        )

    symmetry_document = _mapping(document["symmetry"], "symmetry")
    _exact(
        symmetry_document,
        {"axis_in_object_frame_xyz", "objects"},
        "symmetry",
    )
    axis = _vector(
        symmetry_document["axis_in_object_frame_xyz"],
        3,
        "symmetry.axis",
    )
    if axis != (0.0, 0.0, 1.0):
        raise ValueError("v1 symmetry axis must be object-frame +Z")
    object_document = _mapping(
        symmetry_document["objects"], "symmetry.objects"
    )
    if set(object_document) != {
        "loose_plug",
        "fixed_receptacle",
        "coupling_nut",
    }:
        raise ValueError("symmetry object set differs")
    symmetry: dict[str, SymmetrySpec] = {}
    expected_orders = {
        "loose_plug": 2,
        "fixed_receptacle": 2,
        "coupling_nut": 24,
    }
    for object_id, raw in object_document.items():
        label = f"symmetry.objects.{object_id}"
        item = _mapping(raw, label)
        _exact(
            item,
            {
                "model_id",
                "mesh_input",
                "rotational_symmetry_order",
                "equivalent_yaw_period_rad",
                "unique_key_geometry_present",
                "keyed_yaw_observable",
            },
            label,
        )
        order = _positive_int(
            item["rotational_symmetry_order"], f"{label}.order"
        )
        if order != expected_orders[object_id]:
            raise ValueError(f"{object_id} symmetry order changed")
        period = _positive(
            item["equivalent_yaw_period_rad"], f"{label}.period"
        )
        if not math.isclose(period, 2.0 * math.pi / order, abs_tol=1.0e-15):
            raise ValueError(f"{object_id} yaw period and order disagree")
        unique_key = _boolean(
            item["unique_key_geometry_present"], f"{label}.unique key"
        )
        keyed_yaw = _boolean(
            item["keyed_yaw_observable"], f"{label}.keyed yaw"
        )
        if unique_key or keyed_yaw:
            raise ValueError("current proxy cannot claim a unique keyed yaw")
        mesh_input = _text(item["mesh_input"], f"{label}.mesh_input")
        if mesh_input not in inputs or not mesh_input.endswith("_obj"):
            raise ValueError(f"{object_id} mesh input is not content-bound")
        symmetry[object_id] = SymmetrySpec(
            object_id=object_id,
            model_id=_text(item["model_id"], f"{label}.model_id"),
            mesh_input=mesh_input,
            rotational_symmetry_order=order,
            equivalent_yaw_period_rad=period,
            unique_key_geometry_present=unique_key,
            keyed_yaw_observable=keyed_yaw,
        )

    accuracy_document = _mapping(
        document["accuracy_schema"], "accuracy_schema"
    )
    _exact(
        accuracy_document,
        {
            "translation_error_3d_max_m",
            "axis_error_max_rad",
            "orientation_error_modulo_symmetry_max_rad",
            "current_xy_error_max_m",
            "required_anchor_ids",
            "minimum_anchor_count",
            "current_evidence",
        },
        "accuracy_schema",
    )
    anchors = _string_list(
        accuracy_document["required_anchor_ids"], "required_anchor_ids"
    )
    minimum_anchors = _positive_int(
        accuracy_document["minimum_anchor_count"], "minimum_anchor_count"
    )
    if minimum_anchors != 5 or len(anchors) != 5:
        raise ValueError("v1 requires exactly five anchor placements")
    evidence_document = _mapping(
        accuracy_document["current_evidence"], "current_evidence"
    )
    evidence_keys = {
        "xy_translation_accuracy_available",
        "xyz_translation_accuracy_available",
        "axis_accuracy_available",
        "orientation_modulo_symmetry_accuracy_available",
        "unique_key_yaw_accuracy_available",
    }
    if set(evidence_document) != evidence_keys:
        raise ValueError("current accuracy evidence field set differs")
    evidence = {
        name: _boolean(value, f"current_evidence.{name}")
        for name, value in evidence_document.items()
    }
    if evidence != {
        "xy_translation_accuracy_available": True,
        "xyz_translation_accuracy_available": False,
        "axis_accuracy_available": False,
        "orientation_modulo_symmetry_accuracy_available": False,
        "unique_key_yaw_accuracy_available": False,
    }:
        raise ValueError("current evidence must remain partial XY only")
    accuracy = AccuracyThresholds(
        translation_error_3d_max_m=_positive(
            accuracy_document["translation_error_3d_max_m"],
            "translation threshold",
        ),
        axis_error_max_rad=_positive(
            accuracy_document["axis_error_max_rad"], "axis threshold"
        ),
        orientation_error_modulo_symmetry_max_rad=_positive(
            accuracy_document["orientation_error_modulo_symmetry_max_rad"],
            "orientation threshold",
        ),
        current_xy_error_max_m=_positive(
            accuracy_document["current_xy_error_max_m"], "XY threshold"
        ),
        required_anchor_ids=anchors,
        minimum_anchor_count=minimum_anchors,
        current_evidence=evidence,
    )

    adapter_document = _mapping(
        document["pose_provider_adapter"], "pose_provider_adapter"
    )
    _exact(
        adapter_document,
        {
            "candidate_schema_version",
            "provider_id",
            "provider_version",
            "allowed_purposes",
            "required_clock_domain",
            "required_control_frame",
            "required_calibration_sha256",
            "required_registry_symmetry_class",
            "current_registry_symmetry_class",
            "pair_publication_enabled",
            "full_accuracy_evidence_qualified",
            "control_purpose_allowed",
            "control_authorization_possible",
        },
        "pose_provider_adapter",
    )
    candidate_schema = _text(
        adapter_document["candidate_schema_version"], "candidate schema"
    )
    if candidate_schema != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("candidate schema version changed")
    allowed_names = _string_list(
        adapter_document["allowed_purposes"], "allowed purposes"
    )
    try:
        allowed_purposes = tuple(
            PoseProviderPurpose(item) for item in allowed_names
        )
    except ValueError as exc:
        raise ValueError("adapter purpose is unsupported") from exc
    if set(allowed_purposes) != {
        PoseProviderPurpose.PREFLIGHT,
        PoseProviderPurpose.EVALUATION,
    }:
        raise ValueError("adapter may only serve preflight and evaluation")
    pair_enabled = _boolean(
        adapter_document["pair_publication_enabled"], "pair publication"
    )
    accuracy_qualified = _boolean(
        adapter_document["full_accuracy_evidence_qualified"],
        "accuracy qualification",
    )
    control_purpose = _boolean(
        adapter_document["control_purpose_allowed"], "control purpose"
    )
    control_possible = _boolean(
        adapter_document["control_authorization_possible"],
        "control authorization",
    )
    if (
        pair_enabled
        or accuracy_qualified
        or control_purpose
        or control_possible
    ):
        raise ValueError(
            "current adapter publication/control gates must be false"
        )
    required_registry = _text(
        adapter_document["required_registry_symmetry_class"],
        "required registry symmetry",
    )
    current_registry = _text(
        adapter_document["current_registry_symmetry_class"],
        "current registry symmetry",
    )
    if required_registry != "discrete_axial_order_2":
        raise ValueError(
            "required registry must express axial order-2 symmetry"
        )
    if (
        current_registry != "keyed_order_1"
        or current_registry == required_registry
    ):
        raise ValueError("current registry mismatch must remain explicit")
    adapter = PoseProviderAdapterPolicy(
        candidate_schema_version=candidate_schema,
        provider_id=_identifier(
            adapter_document["provider_id"], "provider_id"
        ),
        provider_version=_identifier(
            adapter_document["provider_version"], "provider_version"
        ),
        allowed_purposes=allowed_purposes,
        required_clock_domain=_identifier(
            adapter_document["required_clock_domain"], "clock_domain"
        ),
        required_control_frame=_text(
            adapter_document["required_control_frame"], "control_frame"
        ),
        required_calibration_sha256=_sha(
            adapter_document["required_calibration_sha256"],
            "calibration SHA-256",
        ),
        required_registry_symmetry_class=required_registry,
        current_registry_symmetry_class=current_registry,
        pair_publication_enabled=pair_enabled,
        full_accuracy_evidence_qualified=accuracy_qualified,
        control_purpose_allowed=control_purpose,
        control_authorization_possible=control_possible,
    )

    boundary_document = _mapping(document["boundaries"], "boundaries")
    boundary_keys = {
        "gpu_required",
        "foundationpose_inference_performed",
        "truth_used_in_candidate_pose",
        "pose_provider_pair_published",
        "full_6d_keyed_pose_claimed",
        "unique_key_yaw_claimed",
        "vision_control_authorized",
        "e2e_integration_allowed",
        "real_assembly_success_claimed",
    }
    if set(boundary_document) != boundary_keys:
        raise ValueError("boundary set differs")
    boundaries = {
        name: _boolean(value, f"boundaries.{name}")
        for name, value in boundary_document.items()
    }
    if any(boundaries.values()):
        raise ValueError(
            "all current action and claim boundaries must be false"
        )

    return SymmetryPoseReadinessContract(
        schema_version=schema,
        enabled=enabled,
        status=_text(document["status"], "status"),
        inputs=inputs,
        symmetry_axis_xyz=axis,  # type: ignore[arg-type]
        symmetry=symmetry,
        accuracy=accuracy,
        adapter=adapter,
        boundaries=boundaries,
    )


def _quaternion_multiply(
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


def _quaternion_conjugate(
    quaternion: Sequence[float],
) -> tuple[float, float, float, float]:
    return (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])


def _rotate_vector(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    pure = (vector[0], vector[1], vector[2], 0.0)
    rotated = _quaternion_multiply(
        _quaternion_multiply(quaternion, pure),
        _quaternion_conjugate(quaternion),
    )
    return (rotated[0], rotated[1], rotated[2])


def _orientation_distance(
    first: Sequence[float], second: Sequence[float]
) -> float:
    dot = abs(sum(a * b for a, b in zip(first, second)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def evaluate_pose_modulo_symmetry(
    *,
    estimated_position_xyz_m: Sequence[Real],
    estimated_quaternion_xyzw: Sequence[Real],
    truth_position_xyz_m: Sequence[Real],
    truth_quaternion_xyzw: Sequence[Real],
    symmetry_order: int,
    unique_key_observed: bool,
    thresholds: AccuracyThresholds,
) -> PoseModuloSymmetryResult:
    """Score a pose against ``truth * Rz(2*pi*k/order)`` hypotheses."""

    if isinstance(symmetry_order, bool) or not isinstance(
        symmetry_order, Integral
    ):
        raise ValueError("symmetry_order must be a positive integer")
    order = int(symmetry_order)
    if order <= 0:
        raise ValueError("symmetry_order must be a positive integer")
    if type(unique_key_observed) is not bool:
        raise ValueError("unique_key_observed must be boolean")
    if unique_key_observed and order != 1:
        raise ValueError(
            "unique key observation contradicts a nontrivial symmetry group"
        )
    estimated_position = _vector(
        estimated_position_xyz_m, 3, "estimated_position_xyz_m"
    )
    truth_position = _vector(truth_position_xyz_m, 3, "truth_position_xyz_m")
    estimated_q = _unit_quaternion(
        estimated_quaternion_xyzw, "estimated_quaternion_xyzw"
    )
    truth_q = _unit_quaternion(
        truth_quaternion_xyzw, "truth_quaternion_xyzw"
    )
    translation_error = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(estimated_position, truth_position))
    )

    estimated_axis = _rotate_vector(estimated_q, (0.0, 0.0, 1.0))
    truth_axis = _rotate_vector(truth_q, (0.0, 0.0, 1.0))
    axis_dot = sum(a * b for a, b in zip(estimated_axis, truth_axis))
    axis_error = math.acos(max(-1.0, min(1.0, axis_dot)))

    distances = []
    for index in range(order):
        angle = 2.0 * math.pi * index / order
        symmetry_q = (0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle))
        equivalent_truth = _quaternion_multiply(truth_q, symmetry_q)
        distances.append(_orientation_distance(estimated_q, equivalent_truth))
    selected_index = min(range(order), key=distances.__getitem__)
    orientation_error = distances[selected_index]

    keyed_error = None
    keyed_rejection = "unique_key_not_observed"
    if unique_key_observed:
        relative = _quaternion_multiply(
            _quaternion_conjugate(truth_q), estimated_q
        )
        x_value, y_value, z_value, w_value = relative
        yaw = math.atan2(
            2.0 * (w_value * z_value + x_value * y_value),
            1.0 - 2.0 * (y_value * y_value + z_value * z_value),
        )
        keyed_error = abs(_wrap_angle(yaw))
        keyed_rejection = None

    translation_passed = (
        translation_error <= thresholds.translation_error_3d_max_m
    )
    axis_passed = axis_error <= thresholds.axis_error_max_rad
    orientation_passed = orientation_error <= (
        thresholds.orientation_error_modulo_symmetry_max_rad
    )
    return PoseModuloSymmetryResult(
        translation_error_m=translation_error,
        axis_error_rad=axis_error,
        orientation_error_modulo_symmetry_rad=orientation_error,
        selected_symmetry_index=selected_index,
        translation_passed=translation_passed,
        axis_passed=axis_passed,
        orientation_modulo_symmetry_passed=orientation_passed,
        symmetry_aware_pose_passed=(
            translation_passed and axis_passed and orientation_passed
        ),
        unique_key_yaw_observable=unique_key_observed,
        unique_key_yaw_error_rad=keyed_error,
        unique_key_yaw_rejection_reason=keyed_rejection,
        vision_control_qualified=False,
    )


def _candidate_observation_mapping(
    endpoint: Mapping[str, Any],
    *,
    role: str,
    timestamp_s: float,
    frame_id: str,
    registry_symmetry_class: str,
) -> dict[str, Any]:
    return {
        "schema_version": "kcg_connector_pose_observation_v1",
        "model_id": endpoint["model_id"],
        "role": role,
        "timestamp_s": timestamp_s,
        "frame_id": frame_id,
        "position_xyz_m": endpoint["position_xyz_m"],
        "quaternion_xyzw": endpoint["quaternion_xyzw"],
        "covariance_6x6": endpoint["covariance_6x6"],
        "confidence": endpoint["confidence"],
        # The current registry value is used only to reuse its numerical
        # parser.  Pair publication independently requires the corrected
        # discrete symmetry class and is blocked in v1.
        "symmetry_class": registry_symmetry_class,
        "source": "vision",
    }


def parse_foundationpose_candidate_pair(
    value: Any,
    readiness: SymmetryPoseReadinessContract,
    pose_contract: ConnectorPoseContract,
    *,
    now_s: Real,
) -> FoundationPoseCandidatePair:
    """Validate an inference candidate without authorizing publication."""

    label = "foundationpose_candidate_pair"
    document = _mapping(value, label)
    _exact(
        document,
        {
            "schema_version",
            "capture_id",
            "inference_id",
            "timestamp_s",
            "clock_domain",
            "control_frame",
            "calibration_sha256",
            "model_version",
            "refine_model_sha256",
            "score_model_sha256",
            "endpoints",
        },
        label,
    )
    if (
        document["schema_version"]
        != readiness.adapter.candidate_schema_version
    ):
        raise ValueError("unsupported FoundationPose candidate schema")
    clock_domain = _identifier(
        document["clock_domain"], f"{label}.clock_domain"
    )
    if clock_domain != readiness.adapter.required_clock_domain:
        raise ValueError("candidate clock domain differs from adapter")
    control_frame = _text(document["control_frame"], f"{label}.control_frame")
    if control_frame != readiness.adapter.required_control_frame:
        raise ValueError("candidate control frame differs from adapter")
    calibration = _sha(
        document["calibration_sha256"], f"{label}.calibration_sha256"
    )
    if calibration != readiness.adapter.required_calibration_sha256:
        raise ValueError("candidate calibration hash differs from adapter")

    bootstrap_path = readiness.inputs["foundationpose_bootstrap"].path
    repository = DEFAULT_CONFIG_PATH.resolve().parents[3]
    bootstrap = load_foundationpose_bootstrap_contract(
        repository / bootstrap_path
    )
    model_version = _text(document["model_version"], f"{label}.model_version")
    if model_version != bootstrap.model_version:
        raise ValueError("candidate FoundationPose model version differs")
    refine_sha = _sha(
        document["refine_model_sha256"], f"{label}.refine_model_sha256"
    )
    score_sha = _sha(
        document["score_model_sha256"], f"{label}.score_model_sha256"
    )
    if refine_sha != bootstrap.models["refine_model"].sha256:
        raise ValueError("candidate refine model hash differs")
    if score_sha != bootstrap.models["score_model"].sha256:
        raise ValueError("candidate score model hash differs")

    timestamp = _finite(document["timestamp_s"], f"{label}.timestamp_s")
    endpoint_document = _mapping(document["endpoints"], f"{label}.endpoints")
    if set(endpoint_document) != set(_ENDPOINT_ROLES):
        raise ValueError("candidate must contain both connector endpoints")
    endpoints: dict[str, FoundationPoseCandidateEndpoint] = {}
    for role in _ENDPOINT_ROLES:
        raw = _mapping(endpoint_document[role], f"{label}.endpoints.{role}")
        _exact(
            raw,
            {
                "model_id",
                "mesh_id",
                "mesh_sha256",
                "position_xyz_m",
                "quaternion_xyzw",
                "covariance_6x6",
                "confidence",
            },
            f"{label}.endpoints.{role}",
        )
        spec = readiness.symmetry[role]
        if raw["model_id"] != spec.model_id:
            raise ValueError(f"candidate {role} model ID differs")
        if raw["mesh_id"] != spec.mesh_input:
            raise ValueError(f"candidate {role} mesh ID differs")
        mesh_sha = _sha(raw["mesh_sha256"], f"candidate {role} mesh SHA")
        if mesh_sha != readiness.inputs[spec.mesh_input].sha256:
            raise ValueError(f"candidate {role} mesh hash differs")
        registry_model = pose_contract.model(spec.model_id)
        observation = parse_connector_pose_observation(
            _candidate_observation_mapping(
                raw,
                role=role,
                timestamp_s=timestamp,
                frame_id=control_frame,
                registry_symmetry_class=registry_model.symmetry_class,
            ),
            pose_contract,
            now_s=_finite(now_s, "now_s"),
        )
        endpoints[role] = FoundationPoseCandidateEndpoint(
            role=role,
            model_id=observation.model_id,
            mesh_id=spec.mesh_input,
            mesh_sha256=mesh_sha,
            position_xyz_m=observation.position_xyz_m,
            quaternion_xyzw=observation.quaternion_xyzw,
            covariance_6x6=observation.covariance_6x6,
            confidence=observation.confidence,
        )
    return FoundationPoseCandidatePair(
        capture_id=_identifier(document["capture_id"], f"{label}.capture_id"),
        inference_id=_identifier(
            document["inference_id"], f"{label}.inference_id"
        ),
        timestamp_s=timestamp,
        clock_domain=clock_domain,
        control_frame=control_frame,
        calibration_sha256=calibration,
        model_version=model_version,
        refine_model_sha256=refine_sha,
        score_model_sha256=score_sha,
        endpoints=endpoints,
    )


def _endpoint_pose_mapping(
    endpoint: FoundationPoseCandidateEndpoint,
    *,
    timestamp_s: float,
    frame_id: str,
    symmetry_class: str,
) -> dict[str, Any]:
    return {
        "schema_version": "kcg_connector_pose_observation_v1",
        "model_id": endpoint.model_id,
        "role": endpoint.role,
        "timestamp_s": timestamp_s,
        "frame_id": frame_id,
        "position_xyz_m": list(endpoint.position_xyz_m),
        "quaternion_xyzw": list(endpoint.quaternion_xyzw),
        "covariance_6x6": [list(row) for row in endpoint.covariance_6x6],
        "confidence": endpoint.confidence,
        "symmetry_class": symmetry_class,
        "source": "vision",
    }


def adapt_foundationpose_candidate_to_pose_provider(
    value: Any,
    readiness: SymmetryPoseReadinessContract,
    pose_contract: ConnectorPoseContract,
    *,
    purpose: PoseProviderPurpose | str,
    now_s: Real,
) -> PoseProviderSample:
    """Return a sample that can never authorize control."""

    try:
        requested_purpose = PoseProviderPurpose(purpose)
    except (TypeError, ValueError) as exc:
        raise ValueError("adapter purpose is unsupported") from exc
    if requested_purpose is PoseProviderPurpose.CONTROL:
        raise ValueError("symmetry adapter rejects control purpose")
    if requested_purpose not in readiness.adapter.allowed_purposes:
        raise ValueError("adapter purpose is not allowed")
    candidate = parse_foundationpose_candidate_pair(
        value, readiness, pose_contract, now_s=now_s
    )
    registry_classes = {
        role: pose_contract.model(
            readiness.symmetry[role].model_id
        ).symmetry_class
        for role in _ENDPOINT_ROLES
    }
    registry_compatible = all(
        value == readiness.adapter.required_registry_symmetry_class
        for value in registry_classes.values()
    )
    publication_blockers = []
    if not readiness.enabled:
        publication_blockers.append("adapter_contract_disabled")
    if not readiness.adapter.pair_publication_enabled:
        publication_blockers.append("pose_provider_pair_publication_disabled")
    if not readiness.adapter.full_accuracy_evidence_qualified:
        publication_blockers.append(
            "full_3d_axis_symmetry_accuracy_not_qualified"
        )
    if not registry_compatible:
        publication_blockers.append(
            "pose_registry_claims_keyed_order_1_but_proxy_is_order_2"
        )
    publish_pair = not publication_blockers
    pair = None
    if publish_pair:
        pair = {
            role: _endpoint_pose_mapping(
                candidate.endpoints[role],
                timestamp_s=candidate.timestamp_s,
                frame_id=candidate.control_frame,
                symmetry_class=(
                    readiness.adapter.required_registry_symmetry_class
                ),
            )
            for role in _ENDPOINT_ROLES
        }
    sample_document = {
        "schema_version": POSE_PROVIDER_SAMPLE_SCHEMA_VERSION,
        "purpose": requested_purpose.value,
        "provider_id": readiness.adapter.provider_id,
        "provider_version": readiness.adapter.provider_version,
        "capture_id": candidate.capture_id,
        "clock_domain": candidate.clock_domain,
        "control_frame": candidate.control_frame,
        "calibration_sha256": candidate.calibration_sha256,
        "pair": pair,
        "reference_truth_pair": None,
        # A representative quaternion exists, but missing key geometry means
        # the shared keyed/full-6D claim remains false.
        "full_6d": False,
        "keyed_orientation_observed": False,
        "uses_truth_position": False,
        "uses_truth_orientation": False,
        "control_authorized": False,
        "preflight_passed": publish_pair,
        "diagnostics": {
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "inference_id": candidate.inference_id,
            "model_version": candidate.model_version,
            "refine_model_sha256": candidate.refine_model_sha256,
            "score_model_sha256": candidate.score_model_sha256,
            "candidate_pose_modulo_symmetry_complete": True,
            "declared_symmetry_orders": {
                role: readiness.symmetry[role].rotational_symmetry_order
                for role in _ENDPOINT_ROLES
            },
            "registry_symmetry_classes": registry_classes,
            "registry_symmetry_compatible": registry_compatible,
            "pair_publication_blockers": publication_blockers,
            "unique_key_yaw_observable": False,
            "unique_key_yaw_rejection_reason": (
                "proxy_unique_polarization_key_geometry_absent"
            ),
            "vision_control_authorized": False,
        },
    }
    return parse_pose_provider_sample(
        sample_document,
        pose_contract,
        purpose=requested_purpose,
        now_s=now_s,
        expected_clock_domain=readiness.adapter.required_clock_domain,
        expected_control_frame=readiness.adapter.required_control_frame,
    )


def _verify_input(repository: Path, item: BoundFile) -> dict[str, Any]:
    path = repository / item.path
    evidence = {
        "path": item.path.as_posix(),
        "exists": path.is_file(),
        "expected_sha256": item.sha256,
        "actual_sha256": None,
        "verified": False,
    }
    if path.is_file():
        evidence["actual_sha256"] = _sha256_file(path)
        evidence["verified"] = evidence["actual_sha256"] == item.sha256
    return evidence


def _multisite_xy_evidence(
    report_path: Path, accuracy: AccuracyThresholds
) -> dict[str, Any]:
    document = _mapping(
        json.loads(report_path.read_text(encoding="utf-8")),
        "multisite report",
    )
    if document.get("schema_version") != "kcg_d38999_multisite_rgbd_report_v1":
        raise ValueError("unsupported multisite RGB-D report schema")
    if document.get("passed") is not True:
        raise ValueError("multisite RGB-D report did not pass")
    if document.get("required_trial_count") != accuracy.minimum_anchor_count:
        raise ValueError("multisite required trial count differs")
    if document.get("passed_trial_count") != accuracy.minimum_anchor_count:
        raise ValueError("multisite passed trial count differs")
    if not math.isclose(
        _finite(document.get("strict_maximum_xy_error_m"), "XY threshold"),
        accuracy.current_xy_error_max_m,
        abs_tol=1.0e-15,
    ):
        raise ValueError("multisite XY threshold differs from contract")
    scope = _mapping(document.get("pose_scope"), "multisite pose_scope")
    for key in ("control_authorized", "full_6d", "keyed_orientation_observed"):
        if scope.get(key) is not False:
            raise ValueError(f"multisite report must keep {key} false")
    trials = document.get("trials")
    if (
        not isinstance(trials, list)
        or len(trials) != accuracy.minimum_anchor_count
    ):
        raise ValueError("multisite report must contain five trials")
    by_anchor = {}
    for trial in trials:
        item = _mapping(trial, "multisite trial")
        anchor = _text(item.get("anchor_id"), "anchor_id")
        if anchor in by_anchor:
            raise ValueError("multisite anchor IDs must be unique")
        if (
            item.get("passed") is not True
            or item.get("capture_passed") is not True
        ):
            raise ValueError(f"multisite anchor {anchor} did not pass")
        if item.get("object_pose_writes_after_start") != 0:
            raise ValueError(
                "multisite evidence wrote object poses after start"
            )
        endpoint_document = _mapping(item.get("endpoints"), "trial endpoints")
        errors = {}
        for role in _ENDPOINT_ROLES:
            endpoint = _mapping(endpoint_document.get(role), f"trial {role}")
            error = _finite(endpoint.get("ray_plane_xy_error_m"), "XY error")
            if error < 0.0:
                raise ValueError("multisite XY error must be nonnegative")
            errors[role] = error
        by_anchor[anchor] = errors
    if set(by_anchor) != set(accuracy.required_anchor_ids):
        raise ValueError("multisite anchor set differs from contract")
    summary = {}
    for role in _ENDPOINT_ROLES:
        values = [
            by_anchor[anchor][role]
            for anchor in accuracy.required_anchor_ids
        ]
        summary[role] = {
            "count": len(values),
            "mean_xy_error_m": sum(values) / len(values),
            "maximum_xy_error_m": max(values),
            "passed": max(values) <= accuracy.current_xy_error_max_m,
        }
    return {
        "anchor_ids": list(accuracy.required_anchor_ids),
        "endpoints": summary,
        "five_anchor_xy_accuracy_passed": all(
            item["passed"] for item in summary.values()
        ),
        "xyz_translation_accuracy_available": False,
        "axis_accuracy_available": False,
        "orientation_modulo_symmetry_accuracy_available": False,
        "unique_key_yaw_accuracy_available": False,
    }


def evaluate_symmetry_pose_readiness(
    contract: SymmetryPoseReadinessContract,
    repository: str | Path,
) -> dict[str, Any]:
    """Aggregate existing CPU-readable evidence without running inference."""

    root = Path(repository).expanduser().resolve()
    inputs = {
        name: _verify_input(root, item)
        for name, item in contract.inputs.items()
    }
    inputs_verified = all(item["verified"] for item in inputs.values())
    xy_evidence = None
    if inputs["multisite_rgbd_report"]["verified"]:
        xy_evidence = _multisite_xy_evidence(
            root / contract.inputs["multisite_rgbd_report"].path,
            contract.accuracy,
        )
    mesh_stats = {}
    mesh_verified = True
    for name in ("loose_body_obj", "coupling_nut_obj", "fixed_receptacle_obj"):
        if not inputs[name]["verified"]:
            mesh_verified = False
            mesh_stats[name] = None
            continue
        stats = validate_obj_document(
            (root / contract.inputs[name].path).read_bytes()
        )
        mesh_stats[name] = {
            "vertex_count": stats.vertex_count,
            "triangle_count": stats.triangle_count,
            "bounds_min_xyz_m": list(stats.bounds_min_xyz_m),
            "bounds_max_xyz_m": list(stats.bounds_max_xyz_m),
        }

    pose_contract = load_connector_pose_contract(
        root / contract.inputs["pose_contract"].path
    )
    registry_classes = {
        role: pose_contract.model(
            contract.symmetry[role].model_id
        ).symmetry_class
        for role in _ENDPOINT_ROLES
    }
    registry_compatible = all(
        value == contract.adapter.required_registry_symmetry_class
        for value in registry_classes.values()
    )
    xy_passed = bool(
        xy_evidence and xy_evidence["five_anchor_xy_accuracy_passed"]
    )
    blockers = []
    if not inputs_verified:
        blockers.append("content_addressed_input_missing_or_mismatched")
    if not mesh_verified:
        blockers.append("three_obj_mesh_bundle_invalid")
    if not xy_passed:
        blockers.append("five_anchor_xy_accuracy_not_passed")
    blockers.extend(
        (
            "five_anchor_xyz_translation_accuracy_not_available",
            "five_anchor_axis_accuracy_not_available",
            "five_anchor_orientation_modulo_symmetry_not_available",
            "unique_key_yaw_unobservable_without_key_geometry",
            "pose_registry_claims_keyed_order_1_but_proxy_is_order_2",
            "adapter_contract_disabled",
            "pose_provider_pair_publication_disabled",
        )
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "METRIC_AND_ADAPTER_STATIC_READY_PAIR_BLOCKED"
            if inputs_verified and mesh_verified and xy_passed
            else "STATIC_READINESS_INPUT_INVALID"
        ),
        "config_enabled": False,
        "inputs": inputs,
        "mesh_stats": mesh_stats,
        "five_anchor_rgbd": xy_evidence,
        "symmetry": {
            name: {
                "model_id": item.model_id,
                "rotational_symmetry_order": item.rotational_symmetry_order,
                "equivalent_yaw_period_rad": item.equivalent_yaw_period_rad,
                "unique_key_geometry_present": False,
                "keyed_yaw_observable": False,
            }
            for name, item in contract.symmetry.items()
        },
        "pose_provider_registry": {
            "current_classes": registry_classes,
            "required_class": (
                contract.adapter.required_registry_symmetry_class
            ),
            "compatible": registry_compatible,
        },
        "gates": {
            "content_addressed_inputs_verified": inputs_verified,
            "three_obj_mesh_bundle_verified": mesh_verified,
            "five_anchor_xy_accuracy_passed": xy_passed,
            "pose_modulo_symmetry_metric_schema_ready": True,
            "foundationpose_candidate_schema_ready": True,
            "xyz_translation_accuracy_qualified": False,
            "axis_accuracy_qualified": False,
            "orientation_modulo_symmetry_accuracy_qualified": False,
            "unique_key_yaw_qualified": False,
            "pose_registry_symmetry_compatible": registry_compatible,
            "pose_provider_pair_ready": False,
            "vision_control_authorized": False,
        },
        "blockers": blockers,
        "claims": {
            "gpu_used": False,
            "foundationpose_inference_performed": False,
            "pose_provider_pair_published": False,
            "full_6d_keyed_pose_claimed": False,
            "unique_key_yaw_claimed": False,
            "vision_control_authorized": False,
        },
    }
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit D38999 symmetry-aware pose readiness on CPU"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--repository", default=str(Path(__file__).resolve().parents[3])
    )
    parser.add_argument("--report", help="optional JSON report output")
    parser.add_argument(
        "--require",
        choices=("assets", "pose_metrics", "provider_pair", "control"),
        default="provider_pair",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    contract = load_symmetry_pose_readiness_contract(arguments.config)
    report = evaluate_symmetry_pose_readiness(contract, arguments.repository)
    if arguments.report:
        output = Path(arguments.report).expanduser().resolve()
        if output.exists() and not output.is_file():
            raise FileExistsError(f"report path is not a file: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    gate = {
        "assets": (
            report["gates"]["content_addressed_inputs_verified"]
            and report["gates"]["three_obj_mesh_bundle_verified"]
        ),
        "pose_metrics": report["gates"][
            "pose_modulo_symmetry_metric_schema_ready"
        ],
        "provider_pair": report["gates"]["pose_provider_pair_ready"],
        "control": report["gates"]["vision_control_authorized"],
    }[arguments.require]
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
