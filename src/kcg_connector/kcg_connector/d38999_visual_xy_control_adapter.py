"""Fail-closed visual-XY to nominal-world-target adaptation.

This pure-Python module is intentionally separate from the D38999 E2E runner.
It consumes partial ray-plane XY evidence carried in a :class:`PoseProvider`
diagnostics sidecar, shifts already validated nominal TCP positions in world
XY, and preserves their Z values.  It does not estimate orientation, solve IK,
authorize production control, or claim a full 6D pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from kcg_connector.connector_pose import (
    ConnectorPoseContract,
    load_connector_pose_contract,
)
from kcg_connector.d38999_multisite_vision6d import (
    D38999MultisiteVision6dContract,
    EndpointPlacementBounds,
    load_d38999_multisite_vision6d_contract,
)
from kcg_connector.d38999_physical_insertion import (
    load_d38999_physical_insertion,
)
from kcg_connector.d38999_tabletop_pick import (
    load_d38999_tabletop_pick_config,
)
from kcg_connector.pose_provider import (
    POSE_PROVIDER_SAMPLE_SCHEMA_VERSION,
    PoseProvider,
    PoseProviderPurpose,
    PoseProviderSample,
    validate_pose_provider_sample,
)


SCHEMA_VERSION = "kcg_d38999_visual_xy_control_adapter_v1"
DIAGNOSTICS_SCHEMA_VERSION = "kcg_d38999_visual_xy_diagnostics_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_visual_xy_control_adapter_v1.yaml"
)
ENDPOINT_ROLES = ("loose_plug", "fixed_receptacle")
ORIENTATION_SOURCES = ("sim_ground_truth", "registered_nominal")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class XyAdapterGates:
    """Numerical gates inherited from existing pose/RGB-D contracts."""

    minimum_confidence: float
    maximum_xy_error_bound_m: float
    maximum_age_s: float
    maximum_future_offset_s: float
    maximum_pair_timestamp_skew_s: float


@dataclass(frozen=True)
class EndpointXyObservation:
    """One bounded ray-plane XY observation from provider diagnostics."""

    estimated_world_xy_m: tuple[float, float]
    timestamp_s: float
    confidence: float
    xy_error_bound_m: float


@dataclass(frozen=True)
class VisualXyControlAdapterContract:
    """Resolved, content-addressed adapter contract."""

    schema_version: str
    enabled_by_default: bool
    status: str
    input_paths: dict[str, Path]
    required_clock_domain: str
    required_control_frame: str
    required_calibration_sha256: str
    estimator_kind: str
    gates: XyAdapterGates
    vision_contract: D38999MultisiteVision6dContract
    pose_contract: ConnectorPoseContract
    nominal_loose_xy_m: tuple[float, float]
    nominal_fixed_xy_m: tuple[float, float]
    loose_maximum_abs_translation_xy_m: tuple[float, float]
    fixed_maximum_abs_translation_xy_m: tuple[float, float]
    nominal_targets: dict[str, tuple[float, float, float]]
    target_roles: dict[str, str]
    validation_trial_count: int
    validation_maximum_observed_xy_error_m: float
    boundaries: dict[str, bool]


@dataclass(frozen=True)
class VisualXyAdaptationResult:
    """Candidate translations or an explicit rejection with no targets."""

    status: str
    eligible_for_independent_probe: bool
    rejection_reasons: tuple[str, ...]
    capture_id: str
    translation_source: str
    orientation_source: str
    uses_truth_orientation: bool
    loose_translation_xy_m: tuple[float, float] | None
    fixed_translation_xy_m: tuple[float, float] | None
    world_targets: dict[str, tuple[float, float, float]]
    validation_maximum_observed_xy_error_m: float

    def to_mapping(self) -> dict[str, Any]:
        """Return JSON-safe evidence without upgrading its control scope."""

        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "eligible_for_independent_probe": (
                self.eligible_for_independent_probe
            ),
            "rejection_reasons": list(self.rejection_reasons),
            "capture_id": self.capture_id,
            "translation_source": self.translation_source,
            "orientation_source": self.orientation_source,
            "uses_truth_orientation": self.uses_truth_orientation,
            "loose_translation_xy_m": (
                None
                if self.loose_translation_xy_m is None
                else list(self.loose_translation_xy_m)
            ),
            "fixed_translation_xy_m": (
                None
                if self.fixed_translation_xy_m is None
                else list(self.fixed_translation_xy_m)
            ),
            "world_targets": {
                name: list(position)
                for name, position in sorted(self.world_targets.items())
            },
            "world_target_frame": "world",
            "preserves_nominal_target_z": True,
            "yaw_observed": False,
            "full_6d": False,
            "pose_provider_control_authorized": False,
            "production_control_authorized": False,
            "collision_free_ik_verified": False,
            "downstream_ik_required": True,
            "validation_maximum_observed_xy_error_m": (
                self.validation_maximum_observed_xy_error_m
            ),
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], label: str) -> None:
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


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
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


def _fraction(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return result


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {length} finite numbers")
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(
            f"{label} must contain {length} finite numbers"
        ) from error
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} finite numbers")
    return tuple(
        _finite(item, f"{label}[{index}]")
        for index, item in enumerate(result)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_paths(value: Any, repository: Path) -> dict[str, Path]:
    inputs = _mapping(value, "inputs")
    names = {
        "multisite_vision_contract",
        "multisite_rgbd_report",
        "rgbd_config",
        "pose_contract",
        "nominal_pick",
        "nominal_insertion",
    }
    _exact(inputs, names, "inputs")
    resolved = {}
    for name in sorted(names):
        label = f"inputs.{name}"
        item = _mapping(inputs[name], label)
        _exact(item, {"path", "sha256"}, label)
        relative = Path(_text(item["path"], f"{label}.path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label}.path must be repository-relative")
        expected = _text(item["sha256"], f"{label}.sha256")
        if not _SHA256.fullmatch(expected):
            raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
        target = (repository / relative).resolve()
        if not target.is_file() or repository not in target.parents:
            raise ValueError(f"{label} is missing or outside repository")
        if _sha256(target) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
        resolved[name] = target
    return resolved


def _validate_multisite_report(
    path: Path,
    *,
    vision_sha256: str,
    rgbd_sha256: str,
    expected_anchor_ids: set[str],
    maximum_xy_error_m: float,
) -> tuple[int, float]:
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "report")
    if report.get("schema_version") != (
        "kcg_d38999_multisite_rgbd_report_v1"
    ):
        raise ValueError("multisite RGB-D report schema is unsupported")
    if (
        report.get("passed") is not True
        or report.get("required_trial_count") != len(expected_anchor_ids)
        or report.get("passed_trial_count") != len(expected_anchor_ids)
        or report.get("strict_maximum_xy_error_m") != maximum_xy_error_m
        or report.get("config_sha256") != vision_sha256
        or report.get("rgbd_config_sha256") != rgbd_sha256
    ):
        raise ValueError(
            "multisite RGB-D report does not prove the 10 mm gate"
        )
    scope = _mapping(report.get("pose_scope"), "report.pose_scope")
    if (
        scope.get("control_authorized") is not False
        or scope.get("full_6d") is not False
        or scope.get("keyed_orientation_observed") is not False
        or scope.get("yaw_observed") is not False
    ):
        raise ValueError("multisite RGB-D report overclaims pose scope")
    trials = report.get("trials")
    if not isinstance(trials, list) or len(trials) != len(expected_anchor_ids):
        raise ValueError("multisite RGB-D report trial count is invalid")
    observed_anchors = set()
    errors = []
    for index, trial_raw in enumerate(trials):
        trial = _mapping(trial_raw, f"report.trials[{index}]")
        observed_anchors.add(_text(trial.get("anchor_id"), "anchor_id"))
        if trial.get("passed") is not True:
            raise ValueError("multisite RGB-D report contains a failed trial")
        endpoints = _mapping(trial.get("endpoints"), "trial.endpoints")
        if set(endpoints) != set(ENDPOINT_ROLES):
            raise ValueError("multisite RGB-D report endpoint set is invalid")
        for role in ENDPOINT_ROLES:
            endpoint = _mapping(endpoints[role], f"endpoints.{role}")
            error = _positive(
                endpoint.get("ray_plane_xy_error_m"),
                f"endpoints.{role}.ray_plane_xy_error_m",
            )
            if (
                endpoint.get("passed") is not True
                or error > maximum_xy_error_m
            ):
                raise ValueError("multisite RGB-D endpoint exceeds 10 mm")
            errors.append(error)
    if observed_anchors != expected_anchor_ids:
        raise ValueError("multisite RGB-D anchor IDs are incomplete")
    return len(trials), max(errors)


def load_visual_xy_control_adapter_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repository: str | Path | None = None,
) -> VisualXyControlAdapterContract:
    """Load all hash edges and reject any scope or gate upgrade."""

    config_path = Path(path).expanduser().resolve()
    root = (
        Path(repository).expanduser().resolve()
        if repository is not None
        else config_path.parents[3]
    )
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "document"
    )
    _exact(
        document,
        {
            "schema_version",
            "enabled_by_default",
            "status",
            "inputs",
            "provider_boundary",
            "gates",
            "bounded_reachability_screen",
            "orientation_policy",
            "target_translation",
            "boundaries",
        },
        "document",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("visual XY adapter schema is unsupported")
    if document["enabled_by_default"] is not False:
        raise ValueError("visual XY adapter must remain disabled by default")
    status = _text(document["status"], "status")
    if status != "prepared_independent_probe_adapter_not_e2e_integrated":
        raise ValueError("visual XY adapter status is unsupported")

    input_paths = _input_paths(document["inputs"], root)
    vision = load_d38999_multisite_vision6d_contract(
        input_paths["multisite_vision_contract"], repository=root
    )
    pose = load_connector_pose_contract(input_paths["pose_contract"])
    pick = load_d38999_tabletop_pick_config(input_paths["nominal_pick"])
    insertion = load_d38999_physical_insertion(
        input_paths["nominal_insertion"]
    )

    provider = _mapping(document["provider_boundary"], "provider_boundary")
    _exact(
        provider,
        {
            "pose_provider_sample_schema",
            "required_purpose",
            "required_clock_domain",
            "required_control_frame",
            "diagnostics_schema",
            "estimator_kind",
            "required_calibration_input",
        },
        "provider_boundary",
    )
    if (
        provider["pose_provider_sample_schema"]
        != POSE_PROVIDER_SAMPLE_SCHEMA_VERSION
        or provider["required_purpose"] != PoseProviderPurpose.PREFLIGHT.value
        or provider["required_control_frame"] != "world"
        or provider["diagnostics_schema"] != DIAGNOSTICS_SCHEMA_VERSION
        or provider["estimator_kind"]
        != "ray_plane_registered_model_height_world_xy_only"
        or provider["required_calibration_input"] != "rgbd_config"
    ):
        raise ValueError("visual XY provider boundary changed")

    rgbd = _mapping(
        yaml.safe_load(input_paths["rgbd_config"].read_text(encoding="utf-8")),
        "rgbd_config",
    )
    if (
        rgbd.get("position_estimator", {}).get("kind")
        != "ray_plane_registered_model_height"
        or rgbd.get("acceptance", {}).get("maximum_xy_centroid_error_m")
        != 0.010
    ):
        raise ValueError("RGB-D ray-plane 10 mm contract changed")

    gates_raw = _mapping(document["gates"], "gates")
    _exact(
        gates_raw,
        {
            "minimum_confidence",
            "maximum_xy_error_bound_m",
            "maximum_age_s",
            "maximum_future_offset_s",
            "maximum_pair_timestamp_skew_s",
            "require_both_endpoints",
            "require_footprint_inside_table",
        },
        "gates",
    )
    gates = XyAdapterGates(
        minimum_confidence=_fraction(
            gates_raw["minimum_confidence"], "gates.minimum_confidence"
        ),
        maximum_xy_error_bound_m=_positive(
            gates_raw["maximum_xy_error_bound_m"],
            "gates.maximum_xy_error_bound_m",
        ),
        maximum_age_s=_positive(gates_raw["maximum_age_s"], "maximum_age_s"),
        maximum_future_offset_s=_positive(
            gates_raw["maximum_future_offset_s"], "maximum_future_offset_s"
        ),
        maximum_pair_timestamp_skew_s=_positive(
            gates_raw["maximum_pair_timestamp_skew_s"],
            "maximum_pair_timestamp_skew_s",
        ),
    )
    if (
        gates.minimum_confidence != pose.policy.minimum_confidence
        or gates.maximum_xy_error_bound_m != 0.010
        or gates.maximum_age_s != pose.policy.maximum_age_s
        or gates.maximum_future_offset_s != pose.policy.maximum_future_offset_s
        or gates.maximum_pair_timestamp_skew_s
        != pose.policy.maximum_pair_timestamp_skew_s
        or gates_raw["require_both_endpoints"] is not True
        or gates_raw["require_footprint_inside_table"] is not True
    ):
        raise ValueError("visual XY gates cannot loosen source contracts")

    orientation = _mapping(
        document["orientation_policy"], "orientation_policy"
    )
    _exact(
        orientation,
        {
            "permitted_sources",
            "orientation_is_not_estimated_by_xy_adapter",
            "yaw_is_not_observed_by_xy_adapter",
            "full_6d",
        },
        "orientation_policy",
    )
    if (
        tuple(orientation["permitted_sources"]) != ORIENTATION_SOURCES
        or orientation["orientation_is_not_estimated_by_xy_adapter"]
        is not True
        or orientation["yaw_is_not_observed_by_xy_adapter"] is not True
        or orientation["full_6d"] is not False
    ):
        raise ValueError("visual XY orientation scope changed")

    targets = _mapping(document["target_translation"], "target_translation")
    _exact(
        targets,
        {
            "preserve_nominal_z",
            "loose_compensated_targets",
            "fixed_compensated_targets",
        },
        "target_translation",
    )
    loose_target_names = (
        "pregrasp_tcp",
        "grasp_tcp",
        "closure_clearance_tcp",
    )
    fixed_target_names = (
        "transport_safe_tcp",
        "axis_high_tcp",
        "preinsert_tcp",
        "engage_tcp",
    )
    if (
        targets["preserve_nominal_z"] is not True
        or tuple(targets["loose_compensated_targets"]) != loose_target_names
        or tuple(targets["fixed_compensated_targets"]) != fixed_target_names
    ):
        raise ValueError("visual XY target set changed")

    anchor_matches = [
        item
        for item in vision.required_anchor_pairs
        if item["id"] == "nominal"
    ]
    if len(anchor_matches) != 1:
        raise ValueError("multisite contract has no unique nominal anchor")
    nominal_loose = tuple(anchor_matches[0]["loose_xy_m"])
    nominal_fixed = tuple(anchor_matches[0]["fixed_xy_m"])
    if (
        pick.geometry_candidate.loose_settled_origin_m[:2] != nominal_loose
        or insertion.motion.engage_tcp_position_m[:2] != nominal_fixed
        or insertion.inputs.tabletop_pick.sha256 != _sha256(
            input_paths["nominal_pick"]
        )
    ):
        raise ValueError("nominal pick/insertion XY frames do not agree")

    reachability = _mapping(
        document["bounded_reachability_screen"],
        "bounded_reachability_screen",
    )
    _exact(
        reachability,
        {
            "loose_maximum_abs_translation_xy_m",
            "fixed_maximum_abs_translation_xy_m",
            "downstream_ik_required",
            "collision_free_ik_claimed",
        },
        "bounded_reachability_screen",
    )
    loose_delta = _vector(
        reachability["loose_maximum_abs_translation_xy_m"],
        2,
        "loose maximum translation",
    )
    fixed_delta = _vector(
        reachability["fixed_maximum_abs_translation_xy_m"],
        2,
        "fixed maximum translation",
    )
    expected_loose = (
        max(
            nominal_loose[0] - vision.loose_plug.x_m.lower,
            vision.loose_plug.x_m.upper - nominal_loose[0],
        ),
        max(
            nominal_loose[1] - vision.loose_plug.y_m.lower,
            vision.loose_plug.y_m.upper - nominal_loose[1],
        ),
    )
    expected_fixed = (
        max(
            nominal_fixed[0] - vision.fixed_receptacle.x_m.lower,
            vision.fixed_receptacle.x_m.upper - nominal_fixed[0],
        ),
        max(
            nominal_fixed[1] - vision.fixed_receptacle.y_m.lower,
            vision.fixed_receptacle.y_m.upper - nominal_fixed[1],
        ),
    )
    if (
        any(
            not math.isclose(actual, expected, abs_tol=1.0e-12)
            for actual, expected in zip(loose_delta, expected_loose)
        )
        or any(
            not math.isclose(actual, expected, abs_tol=1.0e-12)
            for actual, expected in zip(fixed_delta, expected_fixed)
        )
        or reachability["downstream_ik_required"] is not True
        or reachability["collision_free_ik_claimed"] is not False
    ):
        raise ValueError("bounded reachability screen changed")

    boundaries = dict(_mapping(document["boundaries"], "boundaries"))
    expected_boundaries = {
        "explicit_independent_probe_opt_in_required": True,
        "e2e_integration_allowed": False,
        "production_control_authorized": False,
        "pose_provider_control_authorized": False,
        "vision_xy_used_for_translation_only": True,
        "truth_position_used_for_translation": False,
        "full_6d_claimed": False,
        "arbitrary_pose_reachability_claimed": False,
        "collision_planned": False,
        "object_pose_writes_after_start_allowed": False,
        "assembly_success_claimed": False,
    }
    if boundaries != expected_boundaries:
        raise ValueError("visual XY adapter boundaries changed")

    report_trials, report_maximum_error = _validate_multisite_report(
        input_paths["multisite_rgbd_report"],
        vision_sha256=_sha256(input_paths["multisite_vision_contract"]),
        rgbd_sha256=_sha256(input_paths["rgbd_config"]),
        expected_anchor_ids={
            item["id"] for item in vision.required_anchor_pairs
        },
        maximum_xy_error_m=gates.maximum_xy_error_bound_m,
    )
    nominal_targets = {
        "pregrasp_tcp": pick.motion.pregrasp_tcp_position_m,
        "grasp_tcp": pick.motion.grasp_tcp_position_m,
        "closure_clearance_tcp": (
            pick.motion.closure_clearance_tcp_position_m
        ),
        "transport_safe_tcp": insertion.motion.transport_safe_tcp_position_m,
        "axis_high_tcp": insertion.motion.axis_high_tcp_position_m,
        "preinsert_tcp": insertion.motion.preinsert_tcp_position_m,
        "engage_tcp": insertion.motion.engage_tcp_position_m,
    }
    target_roles = {
        name: "loose_plug" for name in loose_target_names
    }
    target_roles.update(
        {name: "fixed_receptacle" for name in fixed_target_names}
    )
    return VisualXyControlAdapterContract(
        schema_version=SCHEMA_VERSION,
        enabled_by_default=False,
        status=status,
        input_paths=input_paths,
        required_clock_domain=_text(
            provider["required_clock_domain"], "required_clock_domain"
        ),
        required_control_frame="world",
        required_calibration_sha256=_sha256(input_paths["rgbd_config"]),
        estimator_kind=provider["estimator_kind"],
        gates=gates,
        vision_contract=vision,
        pose_contract=pose,
        nominal_loose_xy_m=nominal_loose,
        nominal_fixed_xy_m=nominal_fixed,
        loose_maximum_abs_translation_xy_m=loose_delta,
        fixed_maximum_abs_translation_xy_m=fixed_delta,
        nominal_targets=nominal_targets,
        target_roles=target_roles,
        validation_trial_count=report_trials,
        validation_maximum_observed_xy_error_m=report_maximum_error,
        boundaries=boundaries,
    )


def _parse_endpoint(value: Any, label: str) -> EndpointXyObservation:
    document = _mapping(value, label)
    _exact(
        document,
        {
            "estimated_world_xy_m",
            "timestamp_s",
            "confidence",
            "xy_error_bound_m",
        },
        label,
    )
    return EndpointXyObservation(
        estimated_world_xy_m=_vector(
            document["estimated_world_xy_m"],
            2,
            f"{label}.estimated_world_xy_m",
        ),
        timestamp_s=_finite(document["timestamp_s"], f"{label}.timestamp_s"),
        confidence=_fraction(document["confidence"], f"{label}.confidence"),
        xy_error_bound_m=_positive(
            document["xy_error_bound_m"], f"{label}.xy_error_bound_m"
        ),
    )


def _point_in_bounds(
    point: tuple[float, float], bounds: EndpointPlacementBounds
) -> bool:
    return bounds.x_m.contains(point[0]) and bounds.y_m.contains(point[1])


def _footprint_in_table(
    point: tuple[float, float],
    radius: float,
    table: tuple[float, float, float, float],
) -> bool:
    x_min, x_max, y_min, y_max = table
    return (
        point[0] - radius >= x_min
        and point[0] + radius <= x_max
        and point[1] - radius >= y_min
        and point[1] + radius <= y_max
    )


def adapt_visual_xy_to_world_targets(
    contract: VisualXyControlAdapterContract,
    sample: PoseProviderSample,
    *,
    now_s: Real,
    explicit_independent_probe_opt_in: bool,
    orientation_source: str,
) -> VisualXyAdaptationResult:
    """Translate nominal targets for a bounded independent probe.

    Schema/type errors raise immediately.  Runtime gate failures return an
    explicit rejected result with an empty target map, preventing a caller
    from accidentally consuming a stale or out-of-domain target.
    """

    current_time = _finite(now_s, "now_s")
    if not isinstance(explicit_independent_probe_opt_in, bool):
        raise ValueError("explicit_independent_probe_opt_in must be boolean")
    if orientation_source not in ORIENTATION_SOURCES:
        raise ValueError("orientation_source is unsupported")
    validated = validate_pose_provider_sample(
        sample,
        contract.pose_contract,
        purpose=PoseProviderPurpose.PREFLIGHT,
        now_s=current_time,
        expected_clock_domain=contract.required_clock_domain,
        expected_control_frame=contract.required_control_frame,
    )
    diagnostics = _mapping(validated.diagnostics, "diagnostics")
    _exact(
        diagnostics,
        {"schema_version", "estimator", "endpoints"},
        "diagnostics",
    )
    if diagnostics["schema_version"] != DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("visual XY diagnostics schema is unsupported")
    endpoints_raw = _mapping(
        diagnostics["endpoints"], "diagnostics.endpoints"
    )
    _exact(endpoints_raw, set(ENDPOINT_ROLES), "diagnostics.endpoints")
    endpoints = {
        role: _parse_endpoint(endpoints_raw[role], f"endpoints.{role}")
        for role in ENDPOINT_ROLES
    }

    reasons = []
    if not explicit_independent_probe_opt_in:
        reasons.append("explicit_independent_probe_opt_in_missing")
    if diagnostics["estimator"] != contract.estimator_kind:
        reasons.append("ray_plane_estimator_kind_mismatch")
    if validated.calibration_sha256 != contract.required_calibration_sha256:
        reasons.append("rgbd_calibration_provenance_mismatch")
    if (
        validated.pair is not None
        or validated.reference_truth_pair is not None
    ):
        reasons.append("partial_xy_must_remain_in_diagnostics_only")
    if validated.full_6d or validated.keyed_orientation_observed:
        reasons.append("partial_xy_cannot_claim_full_6d_or_keyed_yaw")
    if validated.control_authorized or validated.preflight_passed:
        reasons.append("pose_provider_must_not_authorize_partial_xy")
    if validated.uses_truth_position:
        reasons.append("truth_position_for_translation_forbidden")
    expects_truth_orientation = orientation_source == "sim_ground_truth"
    if validated.uses_truth_orientation != expects_truth_orientation:
        reasons.append("orientation_source_disclosure_mismatch")

    timestamps = []
    for role, endpoint in endpoints.items():
        timestamps.append(endpoint.timestamp_s)
        age = current_time - endpoint.timestamp_s
        if age > contract.gates.maximum_age_s:
            reasons.append(f"{role}:stale")
        if age < -contract.gates.maximum_future_offset_s:
            reasons.append(f"{role}:timestamp_too_far_in_future")
        if endpoint.confidence < contract.gates.minimum_confidence:
            reasons.append(f"{role}:confidence_below_gate")
        if endpoint.xy_error_bound_m > contract.gates.maximum_xy_error_bound_m:
            reasons.append(f"{role}:xy_error_bound_exceeds_10mm")
    if max(timestamps) - min(timestamps) > (
        contract.gates.maximum_pair_timestamp_skew_s
    ):
        reasons.append("endpoint_timestamp_skew_exceeds_gate")

    loose = endpoints["loose_plug"].estimated_world_xy_m
    fixed = endpoints["fixed_receptacle"].estimated_world_xy_m
    loose_bounds = contract.vision_contract.loose_plug
    fixed_bounds = contract.vision_contract.fixed_receptacle
    if not _point_in_bounds(loose, loose_bounds):
        reasons.append("loose_plug:outside_bounded_observation_domain")
    if not _point_in_bounds(fixed, fixed_bounds):
        reasons.append("fixed_receptacle:outside_bounded_observation_domain")
    table = contract.vision_contract.table_xy_bounds_m
    if not _footprint_in_table(loose, loose_bounds.footprint_radius_m, table):
        reasons.append("loose_plug:footprint_outside_table")
    if not _footprint_in_table(fixed, fixed_bounds.footprint_radius_m, table):
        reasons.append("fixed_receptacle:footprint_outside_table")

    loose_delta = tuple(
        value - nominal
        for value, nominal in zip(loose, contract.nominal_loose_xy_m)
    )
    fixed_delta = tuple(
        value - nominal
        for value, nominal in zip(fixed, contract.nominal_fixed_xy_m)
    )
    if any(
        abs(value) > limit + 1.0e-12
        for value, limit in zip(
            loose_delta, contract.loose_maximum_abs_translation_xy_m
        )
    ):
        reasons.append("loose_plug:outside_bounded_reachability_screen")
    if any(
        abs(value) > limit + 1.0e-12
        for value, limit in zip(
            fixed_delta, contract.fixed_maximum_abs_translation_xy_m
        )
    ):
        reasons.append("fixed_receptacle:outside_bounded_reachability_screen")

    eligible = not reasons
    world_targets = {}
    if eligible:
        for name, nominal in contract.nominal_targets.items():
            delta = (
                loose_delta
                if contract.target_roles[name] == "loose_plug"
                else fixed_delta
            )
            world_targets[name] = (
                nominal[0] + delta[0],
                nominal[1] + delta[1],
                nominal[2],
            )
    return VisualXyAdaptationResult(
        status=(
            "ELIGIBLE_FOR_INDEPENDENT_VISUAL_XY_PROBE"
            if eligible
            else "REJECTED_FAIL_CLOSED"
        ),
        eligible_for_independent_probe=eligible,
        rejection_reasons=tuple(reasons),
        capture_id=validated.capture_id,
        translation_source=(
            "vision_semantic_mask_ray_plane_registered_height_xy"
        ),
        orientation_source=orientation_source,
        uses_truth_orientation=validated.uses_truth_orientation,
        loose_translation_xy_m=loose_delta if eligible else None,
        fixed_translation_xy_m=fixed_delta if eligible else None,
        world_targets=world_targets,
        validation_maximum_observed_xy_error_m=(
            contract.validation_maximum_observed_xy_error_m
        ),
    )


def observe_and_adapt_visual_xy(
    contract: VisualXyControlAdapterContract,
    provider: PoseProvider,
    *,
    now_s: Real,
    explicit_independent_probe_opt_in: bool,
    orientation_source: str,
) -> VisualXyAdaptationResult:
    """Call one replaceable provider and adapt its preflight sample.

    This is the intended seam for the next independent probe.  Keeping the
    requested purpose at ``PREFLIGHT`` makes it impossible for this partial
    XY adapter to ask a provider for production ``CONTROL`` authorization.
    """

    if not isinstance(provider, PoseProvider):
        raise ValueError("provider must implement PoseProvider")
    current_time = _finite(now_s, "now_s")
    sample = provider.observe_pair(PoseProviderPurpose.PREFLIGHT, current_time)
    return adapt_visual_xy_to_world_targets(
        contract,
        sample,
        now_s=current_time,
        explicit_independent_probe_opt_in=(
            explicit_independent_probe_opt_in
        ),
        orientation_source=orientation_source,
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DIAGNOSTICS_SCHEMA_VERSION",
    "ENDPOINT_ROLES",
    "ORIENTATION_SOURCES",
    "SCHEMA_VERSION",
    "EndpointXyObservation",
    "VisualXyAdaptationResult",
    "VisualXyControlAdapterContract",
    "XyAdapterGates",
    "adapt_visual_xy_to_world_targets",
    "load_visual_xy_control_adapter_contract",
    "observe_and_adapt_visual_xy",
]
