"""Pure boundary for replaceable connector-pose providers.

The shared :mod:`kcg_connector.connector_pose` observation remains unchanged.
This module adds a strict provenance sidecar around an optional pose pair.  A
provider implements ``observe_pair(purpose, now_s)`` and returns one
``PoseProviderSample``; the consumer must validate that sample against its
requested clock domain, control frame and purpose before using it.

Partial masked RGB-D evidence belongs in ``diagnostics`` with ``pair=None``.
It may pass a preflight, but truth-filled orientation never authorizes robot
control.  A future full-6D vision provider must produce a normal ``vision``
pose pair without using simulator truth.

Simulation implementations compose this boundary with
``sim_pose_provider.make_sim_ground_truth_observation``.  This module neither
repeats Isaac's wxyz conversion nor constructs simulator-truth observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping, Protocol, runtime_checkable

from kcg_connector.connector_pose import (
    ConnectorPoseContract,
    ConnectorPoseObservation,
    ConnectorPosePair,
    ConnectorPoseSource,
    pair_connector_pose_observations,
    parse_connector_pose_observation,
)


POSE_PROVIDER_SAMPLE_SCHEMA_VERSION = "kcg_connector_pose_provider_sample_v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OBSERVATION_FIELDS = {
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
_PAIR_FIELDS = {"loose_plug", "fixed_receptacle"}
_SAMPLE_FIELDS = {
    "schema_version",
    "purpose",
    "provider_id",
    "provider_version",
    "capture_id",
    "clock_domain",
    "control_frame",
    "calibration_sha256",
    "pair",
    "reference_truth_pair",
    "full_6d",
    "keyed_orientation_observed",
    "uses_truth_position",
    "uses_truth_orientation",
    "control_authorized",
    "preflight_passed",
    "diagnostics",
}


class PoseProviderPurpose(str, Enum):
    """Consumer intent bound to one provider sample."""

    PREFLIGHT = "preflight"
    CONTROL = "control"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class PoseProviderSample:
    """Validated pose pair plus provenance and control-authorization facts."""

    schema_version: str
    purpose: PoseProviderPurpose
    provider_id: str
    provider_version: str
    capture_id: str
    clock_domain: str
    control_frame: str
    calibration_sha256: str
    pair: ConnectorPosePair | None
    reference_truth_pair: ConnectorPosePair | None
    full_6d: bool
    keyed_orientation_observed: bool
    uses_truth_position: bool
    uses_truth_orientation: bool
    control_authorized: bool
    preflight_passed: bool
    diagnostics: dict[str, Any]


@runtime_checkable
class PoseProvider(Protocol):
    """Interface implemented by simulation, RGB-D or future 6D providers."""

    def observe_pair(
        self,
        purpose: PoseProviderPurpose,
        now_s: float,
    ) -> PoseProviderSample:
        """Observe both connector roles for one explicit consumer purpose."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
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


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _purpose(value: Any, label: str) -> PoseProviderPurpose:
    try:
        return PoseProviderPurpose(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is unsupported") from error


def _json_safe(value: Any, label: str) -> Any:
    """Copy one strict JSON value while rejecting aliases and non-finites."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return _finite(value, label)
    if isinstance(value, list):
        return [
            _json_safe(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} keys must be strings")
            _text(key, f"{label} key")
            result[key] = _json_safe(item, f"{label}.{key}")
        return result
    raise ValueError(f"{label} must contain only JSON-safe values")


def _observation_mapping(
    observation: ConnectorPoseObservation,
) -> dict[str, Any]:
    return {
        "schema_version": observation.schema_version,
        "model_id": observation.model_id,
        "role": observation.role.value,
        "timestamp_s": observation.timestamp_s,
        "frame_id": observation.frame_id,
        "position_xyz_m": list(observation.position_xyz_m),
        "quaternion_xyzw": list(observation.quaternion_xyzw),
        "covariance_6x6": [
            list(row) for row in observation.covariance_6x6
        ],
        "confidence": observation.confidence,
        "symmetry_class": observation.symmetry_class,
        "source": observation.source.value,
    }


def _pair_mapping(pair: ConnectorPosePair) -> dict[str, Any]:
    return {
        "loose_plug": _observation_mapping(pair.loose_plug),
        "fixed_receptacle": _observation_mapping(pair.fixed_receptacle),
    }


def _parse_pair(
    value: Any,
    contract: ConnectorPoseContract,
    *,
    now_s: float,
    label: str,
) -> ConnectorPosePair | None:
    if value is None:
        return None
    document = _mapping(value, label)
    _exact_keys(document, _PAIR_FIELDS, label)
    observations = []
    for role_name in ("loose_plug", "fixed_receptacle"):
        observation = _mapping(
            document[role_name], f"{label}.{role_name}"
        )
        _exact_keys(
            observation,
            _OBSERVATION_FIELDS,
            f"{label}.{role_name}",
        )
        observations.append(
            parse_connector_pose_observation(
                observation, contract, now_s=now_s
            )
        )
    pair = pair_connector_pose_observations(
        observations[0], observations[1], contract, now_s=now_s
    )
    if pair.loose_plug.source is not pair.fixed_receptacle.source:
        raise ValueError(f"{label} endpoints must use one source")
    return pair


def _pair_observations(
    pair: ConnectorPosePair,
) -> tuple[ConnectorPoseObservation, ConnectorPoseObservation]:
    return (pair.loose_plug, pair.fixed_receptacle)


def _check_pair_provenance(
    pair: ConnectorPosePair,
    *,
    control_frame: str,
    uses_truth_position: bool,
    uses_truth_orientation: bool,
) -> None:
    observations = _pair_observations(pair)
    if any(item.frame_id != control_frame for item in observations):
        raise ValueError("pose pair frame differs from provider control_frame")
    source = observations[0].source
    if source is ConnectorPoseSource.VISION and (
        uses_truth_position or uses_truth_orientation
    ):
        raise ValueError("vision pose pair cannot use truth pose components")
    if source is ConnectorPoseSource.SIM_GROUND_TRUTH and not (
        uses_truth_position and uses_truth_orientation
    ):
        raise ValueError(
            "sim_ground_truth pair must disclose truth position and "
            "orientation"
        )


def _check_reference_truth_pair(
    pair: ConnectorPosePair, *, control_frame: str
) -> None:
    observations = _pair_observations(pair)
    if any(
        item.source is not ConnectorPoseSource.SIM_GROUND_TRUTH
        for item in observations
    ):
        raise ValueError("reference_truth_pair must be sim_ground_truth")
    if any(item.frame_id != control_frame for item in observations):
        raise ValueError(
            "reference_truth_pair frame differs from provider control_frame"
        )


def _check_cross_pair_contract(
    pair: ConnectorPosePair | None,
    reference: ConnectorPosePair | None,
    contract: ConnectorPoseContract,
) -> None:
    if pair is None or reference is None:
        return
    for role in ("loose_plug", "fixed_receptacle"):
        observation = getattr(pair, role)
        truth = getattr(reference, role)
        if observation.model_id != truth.model_id:
            raise ValueError("pose pair and truth reference model IDs differ")
    timestamps = [
        item.timestamp_s
        for candidate in (pair, reference)
        for item in _pair_observations(candidate)
    ]
    if max(timestamps) - min(timestamps) > (
        contract.policy.maximum_pair_timestamp_skew_s
    ):
        raise ValueError("pose pair and truth reference captures are skewed")


def parse_pose_provider_sample(
    value: Any,
    contract: ConnectorPoseContract,
    *,
    purpose: PoseProviderPurpose | str,
    now_s: Real,
    expected_clock_domain: str,
    expected_control_frame: str,
) -> PoseProviderSample:
    """Parse one exact provider sidecar and enforce control authorization."""
    label = "pose_provider_sample"
    document = _mapping(value, label)
    _exact_keys(document, _SAMPLE_FIELDS, label)
    if document["schema_version"] != POSE_PROVIDER_SAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported pose provider sample schema")
    requested_purpose = _purpose(purpose, "purpose")
    sample_purpose = _purpose(document["purpose"], f"{label}.purpose")
    if sample_purpose is not requested_purpose:
        raise ValueError("provider sample purpose differs from request")
    current_time = _finite(now_s, "now_s")
    clock_domain = _identifier(
        document["clock_domain"], f"{label}.clock_domain"
    )
    required_clock = _identifier(
        expected_clock_domain, "expected_clock_domain"
    )
    if clock_domain != required_clock:
        raise ValueError("provider sample clock domain differs from consumer")
    control_frame = _text(
        document["control_frame"], f"{label}.control_frame"
    )
    required_frame = _text(
        expected_control_frame, "expected_control_frame"
    )
    if control_frame != required_frame:
        raise ValueError("provider sample control frame differs from consumer")
    calibration_sha256 = _text(
        document["calibration_sha256"],
        f"{label}.calibration_sha256",
    )
    if not _SHA256.fullmatch(calibration_sha256):
        raise ValueError("calibration_sha256 must be lowercase SHA-256")

    pair = _parse_pair(
        document["pair"], contract, now_s=current_time, label="pair"
    )
    reference = _parse_pair(
        document["reference_truth_pair"],
        contract,
        now_s=current_time,
        label="reference_truth_pair",
    )
    full_6d = _boolean(document["full_6d"], f"{label}.full_6d")
    keyed = _boolean(
        document["keyed_orientation_observed"],
        f"{label}.keyed_orientation_observed",
    )
    truth_position = _boolean(
        document["uses_truth_position"],
        f"{label}.uses_truth_position",
    )
    truth_orientation = _boolean(
        document["uses_truth_orientation"],
        f"{label}.uses_truth_orientation",
    )
    control_authorized = _boolean(
        document["control_authorized"],
        f"{label}.control_authorized",
    )
    preflight_passed = _boolean(
        document["preflight_passed"], f"{label}.preflight_passed"
    )

    if keyed and not full_6d:
        raise ValueError("keyed orientation requires a full-6D estimate")
    if full_6d and pair is None:
        raise ValueError("full-6D provider sample requires a pose pair")
    if pair is not None:
        _check_pair_provenance(
            pair,
            control_frame=control_frame,
            uses_truth_position=truth_position,
            uses_truth_orientation=truth_orientation,
        )
    if reference is not None:
        _check_reference_truth_pair(reference, control_frame=control_frame)
    _check_cross_pair_contract(pair, reference, contract)
    if preflight_passed and pair is None and reference is None:
        raise ValueError("passing preflight needs pose or truth evidence")
    if control_authorized:
        if sample_purpose is not PoseProviderPurpose.CONTROL:
            raise ValueError("control authorization requires control purpose")
        if not preflight_passed:
            raise ValueError("control authorization requires passed preflight")
        if pair is None or not full_6d or not keyed:
            raise ValueError(
                "control authorization requires a keyed full-6D pose pair"
            )
        source = pair.loose_plug.source
        if source is ConnectorPoseSource.VISION and (
            truth_position or truth_orientation
        ):
            raise ValueError("truth-filled vision cannot authorize control")
        if source is ConnectorPoseSource.SIM_GROUND_TRUTH and not (
            truth_position and truth_orientation
        ):
            raise ValueError(
                "partial sim truth cannot authorize control"
            )
    if not preflight_passed and control_authorized:
        raise ValueError("failed preflight cannot authorize control")

    diagnostics = _json_safe(
        _mapping(document["diagnostics"], f"{label}.diagnostics"),
        f"{label}.diagnostics",
    )
    return PoseProviderSample(
        schema_version=POSE_PROVIDER_SAMPLE_SCHEMA_VERSION,
        purpose=sample_purpose,
        provider_id=_identifier(
            document["provider_id"], f"{label}.provider_id"
        ),
        provider_version=_identifier(
            document["provider_version"], f"{label}.provider_version"
        ),
        capture_id=_identifier(
            document["capture_id"], f"{label}.capture_id"
        ),
        clock_domain=clock_domain,
        control_frame=control_frame,
        calibration_sha256=calibration_sha256,
        pair=pair,
        reference_truth_pair=reference,
        full_6d=full_6d,
        keyed_orientation_observed=keyed,
        uses_truth_position=truth_position,
        uses_truth_orientation=truth_orientation,
        control_authorized=control_authorized,
        preflight_passed=preflight_passed,
        diagnostics=diagnostics,
    )


def pose_provider_sample_to_mapping(
    sample: PoseProviderSample,
) -> dict[str, Any]:
    """Return the exact JSON-safe v1 representation of a provider sample."""
    if not isinstance(sample, PoseProviderSample):
        raise ValueError("sample must be PoseProviderSample")
    return {
        "schema_version": sample.schema_version,
        "purpose": sample.purpose.value,
        "provider_id": sample.provider_id,
        "provider_version": sample.provider_version,
        "capture_id": sample.capture_id,
        "clock_domain": sample.clock_domain,
        "control_frame": sample.control_frame,
        "calibration_sha256": sample.calibration_sha256,
        "pair": None if sample.pair is None else _pair_mapping(sample.pair),
        "reference_truth_pair": (
            None
            if sample.reference_truth_pair is None
            else _pair_mapping(sample.reference_truth_pair)
        ),
        "full_6d": sample.full_6d,
        "keyed_orientation_observed": sample.keyed_orientation_observed,
        "uses_truth_position": sample.uses_truth_position,
        "uses_truth_orientation": sample.uses_truth_orientation,
        "control_authorized": sample.control_authorized,
        "preflight_passed": sample.preflight_passed,
        "diagnostics": _json_safe(sample.diagnostics, "diagnostics"),
    }


def validate_pose_provider_sample(
    sample: PoseProviderSample,
    contract: ConnectorPoseContract,
    *,
    purpose: PoseProviderPurpose | str,
    now_s: Real,
    expected_clock_domain: str,
    expected_control_frame: str,
) -> PoseProviderSample:
    """Revalidate a provider result at its point of consumption."""
    return parse_pose_provider_sample(
        pose_provider_sample_to_mapping(sample),
        contract,
        purpose=purpose,
        now_s=now_s,
        expected_clock_domain=expected_clock_domain,
        expected_control_frame=expected_control_frame,
    )
