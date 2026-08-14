"""Fail-closed readiness gate for full D38999 residual-RL training.

The existing residual-v0 task begins from an already-engaged connector.  It
must remain possible to train and evaluate that frozen interface independently
of the longer Home -> perception -> pick -> insert -> screw -> Home workflow.
This module therefore validates a separate, disabled-by-default v1 manifest.

The checker is intentionally pure Python: it does not start Isaac Sim, import
ROS, or infer success from filenames.  Every required gate needs a structured
evidence document, explicit positive checks, bounded numerical metrics, and
independently re-hashed artifacts.  Missing or malformed evidence always
blocks long training.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


MANIFEST_SCHEMA_VERSION = "kcg_d38999_full_skill_rl_readiness_v1"
EVIDENCE_SCHEMA_VERSION = "kcg_d38999_training_gate_evidence_v1"
FULL_SKILL_INTERFACE_VERSION = "kcg_d38999_full_skill_hierarchical_v1"
RESIDUAL_INTERFACE_VERSION = "kcg_connector_twist_residual_wrist_ft_v1"
BASE_RESIDUAL_INTERFACE_VERSION = "kcg_connector_twist_residual_v0"
ACTION_SIZE = 4
OBSERVATION_SIZE = 30
WORKFLOW_STAGES = (
    "DETECT_LOOSE",
    "PICK",
    "IN_HAND_RELOCALIZE",
    "DETECT_FIXED",
    "PREALIGN",
    "INSERT",
    "ENGAGE",
    "SCREW",
    "VERIFY",
    "RETREAT",
    "HOME",
)
POLICY_ACTIVE_STAGES = ("ENGAGE", "SCREW")
REQUIRED_GATE_CATEGORIES = frozenset(
    {
        "perception",
        "control",
        "ft",
        "jitter",
        "collision",
        "randomization",
        "artifact",
    }
)
REQUIRED_GATE_IDS = frozenset(
    {
        "perception_tabletop_pose",
        "full_workflow_control",
        "wrist_ft_contact",
        "coupling_nut_tooth_jitter",
        "collision_safety",
        "tabletop_pose_randomization",
        "repeat_physics_regression",
        "visual_evidence_archive",
    }
)
REQUIRED_LIMITED_EVIDENCE_IDS = frozenset(
    {
        "multisite_rgbd_xy_five_of_five",
        "nut_tooth_six_view_identity_limited_v2",
        "wrist_ft_monitor_three_run_repeatability",
        "smooth_e2e_three_headless_runs",
    }
)
LIMITED_EVIDENCE_DISPOSITIONS = frozenset({"active", "superseded"})
LIMITED_EVIDENCE_ARTIFACTS = {
    "multisite_rgbd_xy_v1": frozenset(
        {
            "report",
            "multisite_config",
            "rgbd_config",
            "runner_source",
        }
    ),
    "nut_tooth_four_view_sync_v1": frozenset(
        {
            "ab_residuals",
            "aggregator_source",
            "analyzer_source",
            "baseline_capture_manifest",
            "baseline_physics_report",
            "baseline_physics_summary",
            "baseline_sync_csv",
            "capture_helper",
            "evidence_manifest",
            "evidence_report",
            "history512_capture_manifest",
            "history512_physics_report",
            "history512_physics_summary",
            "history512_sync_csv",
            "normalized_capture_manifest",
            "normalized_physics_report",
            "normalized_physics_summary",
            "normalized_sync_csv",
            "run_residuals",
            "runner_source",
        }
    ),
    "nut_tooth_six_view_identity_v2": frozenset(
        {
            "axial_all_view_residuals",
            "axial_capture",
            "axial_capture_manifest",
            "axial_evidence",
            "axial_ghost_bundle",
            "axial_wrapper",
            "base_analysis",
            "base_capture",
            "base_capture_manifest",
            "connector_asset",
            "ghost_manifest",
            "ghost_runtime",
            "ghost_visibility_sidecar",
            "occlusion_control",
            "occlusion_evidence",
            "physics_report",
            "physics_summary",
            "prepared_runner_source",
            "run_log",
            "segment23_assignments",
            "segment23_manifest",
            "segment23_reanalysis_source",
            "segment23_report",
            "six_view_manifest",
            "six_view_report",
            "six_view_residuals",
            "sync_evidence",
        }
    ),
    "wrist_ft_monitor_repeatability_v1": frozenset(
        {
            "repeatability_report",
            "run_v3_log",
            "run_v4_log",
            "run_v5_log",
            "monitor_config",
            "runner_source",
        }
    ),
    "smooth_e2e_repeat_v1": frozenset(
        {
            "repeatability_report",
            "run_v3_log",
            "run_v4_log",
            "run_v5_log",
            "monitor_config",
            "runner_source",
        }
    ),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)


@dataclass(frozen=True)
class MetricRule:
    """One finite numerical bound required by a gate."""

    minimum: float | None = None
    maximum: float | None = None
    equal: float | None = None


@dataclass(frozen=True)
class GateSpec:
    """Immutable requirements for one independently recorded gate."""

    gate_id: str
    category: str
    evidence_path: Path
    required_checks: tuple[str, ...]
    required_metrics: Mapping[str, MetricRule]
    required_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactSpec:
    """One repository-relative artifact with immutable byte provenance."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class LimitedEvidenceSpec:
    """Narrow evidence that cannot by itself close a full-skill gate."""

    evidence_id: str
    gate_id: str
    validator: str
    disposition: str
    scope: tuple[str, ...]
    limitations: tuple[str, ...]
    artifacts: Mapping[str, ArtifactSpec]


@dataclass(frozen=True)
class ReadinessManifest:
    """Validated full-skill training contract and its evidence gates."""

    config_path: Path
    training_enabled: bool
    gates: tuple[GateSpec, ...]
    limited_evidence: tuple[LimitedEvidenceSpec, ...]


@dataclass(frozen=True)
class GateResult:
    """Result for one gate; reasons are stable, human-readable blockers."""

    gate_id: str
    category: str
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class LimitedEvidenceResult:
    """Validation result whose scope remains explicitly non-promotional."""

    evidence_id: str
    gate_id: str
    validator: str
    disposition: str
    valid: bool
    scope: tuple[str, ...]
    limitations: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReadinessReport:
    """Aggregate report used by the CLI and training launch guard."""

    ready: bool
    training_enabled: bool
    manifest_path: str
    gate_results: tuple[GateResult, ...]
    limited_evidence_results: tuple[LimitedEvidenceResult, ...]
    global_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report without dataclass magic."""
        return {
            "schema_version": "kcg_d38999_full_skill_readiness_report_v1",
            "ready": self.ready,
            "training_enabled": self.training_enabled,
            "manifest_path": self.manifest_path,
            "global_reasons": list(self.global_reasons),
            "limited_evidence_results": [
                {
                    "evidence_id": result.evidence_id,
                    "gate_id": result.gate_id,
                    "validator": result.validator,
                    "disposition": result.disposition,
                    "valid": result.valid,
                    "scope": list(result.scope),
                    "limitations": list(result.limitations),
                    "reasons": list(result.reasons),
                    "closes_full_skill_gate": False,
                    # A historical superseded record remains visible but is
                    # never allowed to become a positive readiness input.
                    "counts_toward_readiness": (
                        result.disposition == "active"
                    ),
                }
                for result in self.limited_evidence_results
            ],
            "gate_results": [
                {
                    "gate_id": result.gate_id,
                    "category": result.category,
                    "passed": result.passed,
                    "reasons": list(result.reasons),
                }
                for result in self.gate_results
            ],
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    name: str,
    required: set[str],
) -> None:
    keys = set(value)
    if keys != required:
        missing = sorted(required - keys)
        unexpected = sorted(keys - required)
        raise ValueError(
            f"{name} keys do not match schema; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(
        _nonempty_string(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _relative_path(value: Any, name: str) -> Path:
    raw = _nonempty_string(value, name)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a repository-relative safe path")
    return path


def _artifact_spec(value: Any, name: str) -> ArtifactSpec:
    document = _mapping(value, name)
    _exact_keys(
        document,
        name=name,
        required={"path", "sha256", "size_bytes"},
    )
    digest = _nonempty_string(document["sha256"], f"{name}.sha256")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{name}.sha256 must be lowercase SHA256")
    return ArtifactSpec(
        path=_relative_path(document["path"], f"{name}.path"),
        sha256=digest,
        size_bytes=_positive_integer(
            document["size_bytes"], f"{name}.size_bytes"
        ),
    )


def _metric_rule(value: Any, name: str) -> MetricRule:
    document = _mapping(value, name)
    allowed = {"minimum", "maximum", "equal"}
    if not document or not set(document).issubset(allowed):
        raise ValueError(
            f"{name} must contain only minimum, maximum, or equal"
        )
    if "equal" in document and len(document) != 1:
        raise ValueError(f"{name}.equal cannot be combined with bounds")
    rule = MetricRule(
        minimum=(
            _finite_number(document["minimum"], f"{name}.minimum")
            if "minimum" in document
            else None
        ),
        maximum=(
            _finite_number(document["maximum"], f"{name}.maximum")
            if "maximum" in document
            else None
        ),
        equal=(
            _finite_number(document["equal"], f"{name}.equal")
            if "equal" in document
            else None
        ),
    )
    if (
        rule.minimum is not None
        and rule.maximum is not None
        and rule.minimum > rule.maximum
    ):
        raise ValueError(f"{name} minimum exceeds maximum")
    return rule


def load_readiness_manifest(config_path: str | Path) -> ReadinessManifest:
    """Load and strictly validate the disabled-by-default v1 manifest."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        root = _mapping(yaml.safe_load(stream), "manifest")
    _exact_keys(
        root,
        name="manifest",
        required={
            "schema_version",
            "training",
            "workflow",
            "gates",
            "limited_evidence",
        },
    )
    if root["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported readiness schema: {root['schema_version']!r}"
        )

    training = _mapping(root["training"], "training")
    _exact_keys(
        training,
        name="training",
        required={
            "enabled",
            "full_skill_interface_version",
            "residual_interface_version",
            "base_residual_interface_version",
            "action_size",
            "observation_size",
            "active_v0_modified",
            "long_training_requires_readiness_pass",
        },
    )
    enabled = _strict_bool(training["enabled"], "training.enabled")
    expected_training = {
        "full_skill_interface_version": FULL_SKILL_INTERFACE_VERSION,
        "residual_interface_version": RESIDUAL_INTERFACE_VERSION,
        "base_residual_interface_version": BASE_RESIDUAL_INTERFACE_VERSION,
        "action_size": ACTION_SIZE,
        "observation_size": OBSERVATION_SIZE,
        "active_v0_modified": False,
        "long_training_requires_readiness_pass": True,
    }
    for field, expected in expected_training.items():
        if training[field] != expected:
            raise ValueError(
                f"training.{field} must be {expected!r}, got "
                f"{training[field]!r}"
            )

    workflow = _mapping(root["workflow"], "workflow")
    _exact_keys(
        workflow,
        name="workflow",
        required={
            "topology",
            "stage_order",
            "policy_active_stages",
            "fsm_owns_perception_pick_and_free_space_motion",
            "perception_control_source",
            "simulation_ground_truth_control_authority",
        },
    )
    if workflow["topology"] != "hierarchical_fsm_with_contact_residual":
        raise ValueError("workflow.topology is unsupported")
    if tuple(workflow["stage_order"]) != WORKFLOW_STAGES:
        raise ValueError("workflow.stage_order must match the full workflow")
    if tuple(workflow["policy_active_stages"]) != POLICY_ACTIVE_STAGES:
        raise ValueError(
            "policy may be active only in ENGAGE and SCREW for this 4D action"
        )
    if workflow["fsm_owns_perception_pick_and_free_space_motion"] is not True:
        raise ValueError("the deterministic FSM must own non-contact phases")
    if workflow["perception_control_source"] != "measured_rgbd_pose":
        raise ValueError(
            "perception control source must be measured RGB-D pose"
        )
    if workflow["simulation_ground_truth_control_authority"] is not False:
        raise ValueError("simulation truth cannot have control authority")

    raw_gates = root["gates"]
    if not isinstance(raw_gates, list) or not raw_gates:
        raise ValueError("gates must be a non-empty list")
    gates: list[GateSpec] = []
    for index, value in enumerate(raw_gates):
        name = f"gates[{index}]"
        gate = _mapping(value, name)
        _exact_keys(
            gate,
            name=name,
            required={
                "gate_id",
                "category",
                "evidence_path",
                "required_checks",
                "required_metrics",
                "required_artifacts",
            },
        )
        metrics = _mapping(
            gate["required_metrics"], f"{name}.required_metrics"
        )
        parsed_metrics = {
            _nonempty_string(metric_name, f"{name}.metric_name"): _metric_rule(
                rule, f"{name}.required_metrics.{metric_name}"
            )
            for metric_name, rule in metrics.items()
        }
        gates.append(
            GateSpec(
                gate_id=_nonempty_string(gate["gate_id"], f"{name}.gate_id"),
                category=_nonempty_string(
                    gate["category"], f"{name}.category"
                ),
                evidence_path=_relative_path(
                    gate["evidence_path"], f"{name}.evidence_path"
                ),
                required_checks=_string_tuple(
                    gate["required_checks"], f"{name}.required_checks"
                ),
                required_metrics=parsed_metrics,
                required_artifacts=_string_tuple(
                    gate["required_artifacts"],
                    f"{name}.required_artifacts",
                ),
            )
        )

    ids = tuple(gate.gate_id for gate in gates)
    if len(ids) != len(set(ids)):
        raise ValueError("gate_id values must be unique")
    if frozenset(ids) != REQUIRED_GATE_IDS:
        raise ValueError(
            "manifest gate ids must exactly match required full-skill gates"
        )
    categories = frozenset(gate.category for gate in gates)
    if categories != REQUIRED_GATE_CATEGORIES:
        raise ValueError(
            "manifest must cover perception, control, ft, jitter, collision, "
            "randomization, and artifact categories"
        )
    evidence_paths = tuple(gate.evidence_path for gate in gates)
    if len(evidence_paths) != len(set(evidence_paths)):
        raise ValueError("gate evidence paths must be unique")

    raw_limited = root["limited_evidence"]
    if not isinstance(raw_limited, list) or not raw_limited:
        raise ValueError("limited_evidence must be a non-empty list")
    limited: list[LimitedEvidenceSpec] = []
    for index, value in enumerate(raw_limited):
        name = f"limited_evidence[{index}]"
        item = _mapping(value, name)
        _exact_keys(
            item,
            name=name,
            required={
                "evidence_id",
                "gate_id",
                "validator",
                "disposition",
                "scope",
                "limitations",
                "artifacts",
            },
        )
        evidence_id = _nonempty_string(
            item["evidence_id"], f"{name}.evidence_id"
        )
        gate_id = _nonempty_string(item["gate_id"], f"{name}.gate_id")
        if gate_id not in REQUIRED_GATE_IDS:
            raise ValueError(f"{name}.gate_id is not a required gate")
        validator = _nonempty_string(
            item["validator"], f"{name}.validator"
        )
        if validator not in LIMITED_EVIDENCE_ARTIFACTS:
            raise ValueError(f"{name}.validator is unsupported")
        disposition = _nonempty_string(
            item["disposition"], f"{name}.disposition"
        )
        if disposition not in LIMITED_EVIDENCE_DISPOSITIONS:
            raise ValueError(
                f"{name}.disposition must be active or superseded"
            )
        raw_artifacts = _mapping(item["artifacts"], f"{name}.artifacts")
        expected_artifacts = LIMITED_EVIDENCE_ARTIFACTS[validator]
        if frozenset(raw_artifacts) != expected_artifacts:
            raise ValueError(
                f"{name}.artifacts must exactly match "
                f"{sorted(expected_artifacts)}"
            )
        artifacts = {
            key: _artifact_spec(
                raw_artifacts[key], f"{name}.artifacts.{key}"
            )
            for key in sorted(raw_artifacts)
        }
        limited.append(
            LimitedEvidenceSpec(
                evidence_id=evidence_id,
                gate_id=gate_id,
                validator=validator,
                disposition=disposition,
                scope=_string_tuple(item["scope"], f"{name}.scope"),
                limitations=_string_tuple(
                    item["limitations"], f"{name}.limitations"
                ),
                artifacts=artifacts,
            )
        )
    limited_ids = tuple(item.evidence_id for item in limited)
    if len(limited_ids) != len(set(limited_ids)):
        raise ValueError("limited evidence_id values must be unique")
    if frozenset(limited_ids) != REQUIRED_LIMITED_EVIDENCE_IDS:
        raise ValueError(
            "manifest limited evidence ids must exactly match the required set"
        )
    return ReadinessManifest(
        config_path=path,
        training_enabled=enabled,
        gates=tuple(gates),
        limited_evidence=tuple(limited),
    )


def _repository_file(repo_root: Path, relative: Path, name: str) -> Path:
    root = repo_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} resolves outside repository root") from error
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_failure(
    name: str,
    value: float,
    rule: MetricRule,
) -> str | None:
    if rule.equal is not None and value != rule.equal:
        return f"metric {name}={value:g} != {rule.equal:g}"
    if rule.minimum is not None and value < rule.minimum:
        return f"metric {name}={value:g} < {rule.minimum:g}"
    if rule.maximum is not None and value > rule.maximum:
        return f"metric {name}={value:g} > {rule.maximum:g}"
    return None


def _check_gate(gate: GateSpec, repo_root: Path) -> GateResult:
    reasons: list[str] = []
    try:
        evidence_path = _repository_file(
            repo_root, gate.evidence_path, f"{gate.gate_id}.evidence_path"
        )
    except ValueError as error:
        return GateResult(
            gate.gate_id, gate.category, False, (str(error),)
        )
    if not evidence_path.is_file():
        return GateResult(
            gate.gate_id,
            gate.category,
            False,
            (f"missing evidence file: {gate.evidence_path}",),
        )
    try:
        with evidence_path.open("r", encoding="utf-8") as stream:
            evidence = _mapping(
                yaml.safe_load(stream), f"evidence {gate.gate_id}"
            )
        _exact_keys(
            evidence,
            name=f"evidence {gate.gate_id}",
            required={
                "schema_version",
                "contract_version",
                "gate_id",
                "passed",
                "run_id",
                "generated_utc",
                "command",
                "checks",
                "metrics",
                "artifacts",
            },
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        return GateResult(
            gate.gate_id,
            gate.category,
            False,
            (f"invalid evidence document: {error}",),
        )

    if evidence["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        reasons.append("evidence schema_version mismatch")
    if evidence["contract_version"] != FULL_SKILL_INTERFACE_VERSION:
        reasons.append("evidence contract_version mismatch")
    if evidence["gate_id"] != gate.gate_id:
        reasons.append("evidence gate_id mismatch")
    try:
        if not _strict_bool(evidence["passed"], "evidence.passed"):
            reasons.append("producer marked gate passed=false")
        _nonempty_string(evidence["run_id"], "evidence.run_id")
        generated = _nonempty_string(
            evidence["generated_utc"], "evidence.generated_utc"
        )
        if not _UTC_RE.fullmatch(generated):
            reasons.append("generated_utc is not an explicit UTC timestamp")
        _nonempty_string(evidence["command"], "evidence.command")
    except ValueError as error:
        reasons.append(str(error))

    try:
        checks = _mapping(evidence["checks"], "evidence.checks")
        for name in gate.required_checks:
            if name not in checks:
                reasons.append(f"missing required check: {name}")
            elif checks[name] is not True:
                reasons.append(f"required check is not true: {name}")
        metrics = _mapping(evidence["metrics"], "evidence.metrics")
        for name, rule in gate.required_metrics.items():
            if name not in metrics:
                reasons.append(f"missing required metric: {name}")
                continue
            try:
                metric_value = _finite_number(
                    metrics[name], f"evidence.metrics.{name}"
                )
            except ValueError as error:
                reasons.append(str(error))
                continue
            failure = _metric_failure(name, metric_value, rule)
            if failure is not None:
                reasons.append(failure)
        artifacts = _mapping(evidence["artifacts"], "evidence.artifacts")
    except ValueError as error:
        reasons.append(str(error))
        artifacts = {}

    for key in gate.required_artifacts:
        if key not in artifacts:
            reasons.append(f"missing required artifact record: {key}")
            continue
        record_name = f"evidence.artifacts.{key}"
        try:
            record = _mapping(artifacts[key], record_name)
            _exact_keys(
                record,
                name=record_name,
                required={"path", "sha256", "size_bytes"},
            )
            relative = _relative_path(record["path"], f"{record_name}.path")
            expected_hash = _nonempty_string(
                record["sha256"], f"{record_name}.sha256"
            )
            if not _SHA256_RE.fullmatch(expected_hash):
                raise ValueError(
                    f"{record_name}.sha256 must be lowercase SHA256"
                )
            expected_size = _positive_integer(
                record["size_bytes"], f"{record_name}.size_bytes"
            )
            artifact_path = _repository_file(
                repo_root, relative, f"{record_name}.path"
            )
            if not artifact_path.is_file():
                reasons.append(f"artifact is missing: {relative}")
                continue
            actual_size = artifact_path.stat().st_size
            if actual_size != expected_size:
                reasons.append(
                    f"artifact size mismatch for {key}: "
                    f"{actual_size} != {expected_size}"
                )
            actual_hash = _sha256(artifact_path)
            if actual_hash != expected_hash:
                reasons.append(f"artifact SHA256 mismatch for {key}")
        except (OSError, ValueError) as error:
            reasons.append(str(error))

    return GateResult(
        gate_id=gate.gate_id,
        category=gate.category,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_document(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return _mapping(json.load(stream), name)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error


def _yaml_document(path: Path, name: str) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return _mapping(yaml.safe_load(stream), name)


def _path_has_suffix(value: Any, suffix: Path, name: str) -> None:
    raw = Path(_nonempty_string(value, name))
    _require(
        tuple(raw.parts[-len(suffix.parts):]) == tuple(suffix.parts),
        f"{name} does not end with hash-bound path {suffix}",
    )


def _verify_limited_artifacts(
    evidence: LimitedEvidenceSpec, repo_root: Path
) -> tuple[dict[str, Path], list[str]]:
    paths = {}
    reasons = []
    for key, artifact in evidence.artifacts.items():
        name = f"limited_evidence.{evidence.evidence_id}.{key}"
        try:
            path = _repository_file(repo_root, artifact.path, f"{name}.path")
            if not path.is_file():
                reasons.append(f"artifact is missing: {artifact.path}")
                continue
            actual_size = path.stat().st_size
            if actual_size != artifact.size_bytes:
                reasons.append(
                    f"artifact size mismatch for {key}: "
                    f"{actual_size} != {artifact.size_bytes}"
                )
            if _sha256(path) != artifact.sha256:
                reasons.append(f"artifact SHA256 mismatch for {key}")
            paths[key] = path
        except (OSError, ValueError) as error:
            reasons.append(str(error))
    return paths, reasons


def _validate_multisite_rgbd(
    evidence: LimitedEvidenceSpec, paths: Mapping[str, Path]
) -> None:
    multisite_config = _yaml_document(
        paths["multisite_config"], "multisite RGBD config"
    )
    rgbd_config = _yaml_document(paths["rgbd_config"], "RGBD config")
    _require(
        multisite_config.get("schema_version")
        == "kcg_d38999_multisite_vision6d_v1",
        "multisite config schema_version mismatch",
    )
    _require(
        rgbd_config.get("schema_version")
        == "kcg_d38999_rgbd_bootstrap_v1",
        "RGBD config schema_version mismatch",
    )
    source = paths["runner_source"].read_text(encoding="utf-8")
    _require(
        "kcg_d38999_multisite_rgbd_report_v1" in source,
        "multisite runner does not declare the report schema",
    )

    report = _json_document(paths["report"], "multisite RGBD report")
    _exact_keys(
        report,
        name="multisite RGBD report",
        required={
            "schema_version",
            "config_path",
            "config_sha256",
            "rgbd_config_path",
            "rgbd_config_sha256",
            "report_path",
            "gui",
            "passed",
            "required_trial_count",
            "passed_trial_count",
            "strict_maximum_xy_error_m",
            "pose_scope",
            "trials",
        },
    )
    _require(
        report["schema_version"] == "kcg_d38999_multisite_rgbd_report_v1",
        "multisite RGBD report schema_version mismatch",
    )
    _require(
        report["gui"] is False,
        "multisite RGBD evidence must be headless",
    )
    _require(report["passed"] is True, "multisite RGBD report is not passed")
    _require(
        report["required_trial_count"] == 5
        and report["passed_trial_count"] == 5,
        "multisite RGBD report must record exactly 5/5 trials",
    )
    _require(
        report["config_sha256"]
        == evidence.artifacts["multisite_config"].sha256,
        "multisite report config SHA256 does not match binding",
    )
    _require(
        report["rgbd_config_sha256"]
        == evidence.artifacts["rgbd_config"].sha256,
        "multisite report RGBD config SHA256 does not match binding",
    )
    _path_has_suffix(
        report["config_path"],
        evidence.artifacts["multisite_config"].path,
        "multisite report config_path",
    )
    _path_has_suffix(
        report["rgbd_config_path"],
        evidence.artifacts["rgbd_config"].path,
        "multisite report rgbd_config_path",
    )
    _path_has_suffix(
        report["report_path"],
        evidence.artifacts["report"].path,
        "multisite report report_path",
    )
    maximum_error = _finite_number(
        report["strict_maximum_xy_error_m"],
        "multisite report strict_maximum_xy_error_m",
    )
    _require(maximum_error == 0.01, "multisite XY limit must equal 0.01 m")

    pose_scope = _mapping(report["pose_scope"], "multisite pose_scope")
    _exact_keys(
        pose_scope,
        name="multisite pose_scope",
        required={
            "control_authorized",
            "full_6d",
            "keyed_orientation_observed",
            "uses_truth_orientation_for_vision_pose",
            "yaw_observed",
            "rejection_reasons",
        },
    )
    for key in (
        "control_authorized",
        "full_6d",
        "keyed_orientation_observed",
        "uses_truth_orientation_for_vision_pose",
        "yaw_observed",
    ):
        _require(
            pose_scope.get(key) is False,
            f"multisite pose_scope.{key} must remain false",
        )
    trials = report["trials"]
    _require(
        isinstance(trials, list) and len(trials) == 5,
        "expected 5 trials",
    )
    indices = set()
    anchors = set()
    for index, raw_trial in enumerate(trials):
        trial = _mapping(raw_trial, f"multisite trials[{index}]")
        _exact_keys(
            trial,
            name=f"multisite trials[{index}]",
            required={
                "trial_index",
                "anchor_id",
                "passed",
                "capture_passed",
                "authored_scene",
                "endpoint_authoring_before_physics",
                "physics_started_after_endpoint_authoring",
                "object_pose_writes_before_start",
                "object_pose_writes_after_start",
                "observed_semantic_ids",
                "camera_projection",
                "endpoints",
                "pose_scope",
                "timeline_pause",
                "timeline_state",
                "resource_cleanup",
            },
        )
        trial_index = _positive_integer(
            trial.get("trial_index"), f"multisite trials[{index}].trial_index"
        )
        indices.add(trial_index)
        anchors.add(
            _nonempty_string(
                trial.get("anchor_id"),
                f"multisite trials[{index}].anchor_id",
            )
        )
        for key in ("passed", "capture_passed"):
            _require(
                trial.get(key) is True,
                f"multisite trials[{index}].{key} must be true",
            )
        _require(
            trial.get("object_pose_writes_after_start") == 0,
            f"multisite trials[{index}] has runtime pose writes",
        )
        _require(
            trial.get("physics_started_after_endpoint_authoring") is True,
            f"multisite trials[{index}] authoring order is invalid",
        )
        endpoints = _mapping(
            trial.get("endpoints"), f"multisite trials[{index}].endpoints"
        )
        _require(
            set(endpoints) == {"loose_plug", "fixed_receptacle"},
            f"multisite trials[{index}] endpoints are incomplete",
        )
        for endpoint_name, raw_endpoint in endpoints.items():
            endpoint = _mapping(
                raw_endpoint,
                f"multisite trials[{index}].endpoints.{endpoint_name}",
            )
            _exact_keys(
                endpoint,
                name=f"multisite trial {index} endpoint {endpoint_name}",
                required={
                    "passed",
                    "semantic_ids",
                    "mask_pixel_count",
                    "visible_fraction",
                    "semantic_mask_center",
                    "mask_depth",
                    "ray_plane_xy_error_m",
                },
            )
            _require(
                endpoint.get("passed") is True,
                f"multisite trial {index} endpoint {endpoint_name} failed",
            )
            error = _finite_number(
                endpoint.get("ray_plane_xy_error_m"),
                f"multisite trial {index} {endpoint_name} XY error",
            )
            _require(
                error <= maximum_error,
                f"multisite trial {index} {endpoint_name} exceeds XY limit",
            )
    _require(
        indices == set(range(1, 6)),
        "multisite trial indices are invalid",
    )
    _require(len(anchors) == 5, "multisite anchors must be unique")


def _validate_tooth_report(
    report: Mapping[str, Any], mode: str, expected_requested: Mapping[str, Any]
) -> None:
    _exact_keys(
        report,
        name=f"tooth report {mode}",
        required={
            "schema_version",
            "output_directory",
            "steps",
            "phase_steps",
            "thresholds",
            "anomaly_steps",
            "segment_aggregate",
            "segment00_schema",
            "normalization_ab",
            "fabric",
            "color_identification",
            "render_ab_launch",
        },
    )
    _require(
        report["schema_version"] == "kcg_d38999_nut_tooth_jitter_probe_v1",
        f"tooth report {mode} schema_version mismatch",
    )
    _positive_integer(report["steps"], f"tooth report {mode}.steps")
    _require(
        report["anomaly_steps"] == 0,
        f"tooth report {mode} contains anomaly steps",
    )
    thresholds = _mapping(report["thresholds"], f"tooth {mode} thresholds")
    _exact_keys(
        thresholds,
        name=f"tooth {mode} thresholds",
        required={"rotation_rad", "translation_m"},
    )
    rotation_limit = _finite_number(
        thresholds["rotation_rad"], f"tooth {mode} rotation threshold"
    )
    translation_limit = _finite_number(
        thresholds["translation_m"], f"tooth {mode} translation threshold"
    )
    _require(
        rotation_limit == 0.00001 and translation_limit == 0.000001,
        f"tooth report {mode} thresholds changed",
    )
    segments = _mapping(
        report["segment_aggregate"], f"tooth {mode} segment_aggregate"
    )
    expected_segments = {f"Segment_{index:02d}" for index in range(24)}
    _require(
        set(segments) == expected_segments,
        f"tooth report {mode} must contain all 24 segments",
    )
    for segment_name, raw_segment in segments.items():
        segment = _mapping(raw_segment, f"tooth {mode} {segment_name}")
        _exact_keys(
            segment,
            name=f"tooth {mode} {segment_name}",
            required={
                "maximum_local_translation_error_m",
                "maximum_local_rotation_error_rad",
                "maximum_parent_relative_translation_error_m",
                "maximum_parent_relative_rotation_error_rad",
                "contact_records",
                "contact_counterparts",
                "minimum_contact_separation_m",
                "maximum_contact_impulse_norm",
            },
        )
        _positive_integer(
            segment["contact_records"],
            f"tooth {mode} {segment_name}.contact_records",
        )
        for key, maximum in (
            ("maximum_local_translation_error_m", translation_limit),
            ("maximum_parent_relative_translation_error_m", translation_limit),
            ("maximum_local_rotation_error_rad", rotation_limit),
            ("maximum_parent_relative_rotation_error_rad", rotation_limit),
        ):
            value = _finite_number(
                segment[key], f"tooth {mode} {segment_name}.{key}"
            )
            _require(
                value <= maximum,
                f"tooth {mode} {segment_name}.{key} exceeds threshold",
            )
    render = _mapping(report["render_ab_launch"], f"tooth {mode} render A/B")
    _exact_keys(
        render,
        name=f"tooth {mode} render A/B",
        required={
            "mode",
            "requested",
            "actual",
            "extra_args",
            "mismatches",
            "exact_match",
            "validated_after_simulation_app_start",
        },
    )
    _require(render["mode"] == mode, f"tooth render mode {mode} mismatch")
    _require(
        render["requested"] == expected_requested,
        f"tooth render mode {mode} request mismatch",
    )
    _require(
        render["exact_match"] is True
        and render["validated_after_simulation_app_start"] is True
        and render["mismatches"] == [],
        f"tooth render mode {mode} setting readback failed",
    )
    actual = _mapping(render["actual"], f"tooth {mode} actual settings")
    for key, value in expected_requested.items():
        _require(
            actual.get(key) == value,
            f"tooth render mode {mode} actual {key} mismatch",
        )


def _validate_tooth_ab(
    evidence: LimitedEvidenceSpec, paths: Mapping[str, Path]
) -> None:
    cases = (
        ("baseline_report", "baseline", {}),
        (
            "rtx_history_512_report",
            "rtx_history_512",
            {"/rtx/scenedb/maxHistoryTransformCount": 512},
        ),
        (
            "fabric_disabled_report",
            "fabric_scene_delegate_disabled",
            {"/app/useFabricSceneDelegate": False},
        ),
    )
    reports = []
    for key, mode, requested in cases:
        report = _json_document(paths[key], f"tooth report {mode}")
        _validate_tooth_report(report, mode, requested)
        _path_has_suffix(
            report["output_directory"],
            evidence.artifacts[key].path.parent,
            f"tooth report {mode}.output_directory",
        )
        reports.append(report)
    baseline = reports[0]
    for report in reports[1:]:
        _require(
            report["steps"] == baseline["steps"]
            and report["phase_steps"] == baseline["phase_steps"],
            "tooth render A/B reports do not cover identical phases",
        )


def _binding_document(artifact: ArtifactSpec) -> dict[str, Any]:
    """Return the JSON form used by the four-view evidence manifest."""

    return {
        "path": artifact.path.as_posix(),
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def _validate_four_view_coverage(value: Any, name: str) -> None:
    """Validate coverage without promoting a partial union to no-jitter."""

    coverage = _mapping(value, name)
    _exact_keys(
        coverage,
        name=name,
        required={
            "complete_24_segment_transitions",
            "every_transition_all_24",
            "identity_union_all_24",
            "minimum_segments_per_transition",
            "missing_from_identity_union",
            "per_phase",
            "per_segment_transition_counts",
            "segments_in_identity_union",
            "transitions",
        },
    )
    transitions = _positive_integer(
        coverage["transitions"], f"{name}.transitions"
    )
    complete = coverage["complete_24_segment_transitions"]
    _require(
        type(complete) is int and 0 <= complete <= transitions,
        f"{name}.complete_24_segment_transitions is invalid",
    )
    minimum = _positive_integer(
        coverage["minimum_segments_per_transition"],
        f"{name}.minimum_segments_per_transition",
    )
    _require(minimum <= 24, f"{name} minimum coverage exceeds 24")
    observed = _string_tuple(
        coverage["segments_in_identity_union"],
        f"{name}.segments_in_identity_union",
    )
    missing = _string_tuple(
        coverage["missing_from_identity_union"],
        f"{name}.missing_from_identity_union",
    )
    expected_segments = {f"Segment_{index:02d}" for index in range(24)}
    _require(
        set(observed).isdisjoint(missing)
        and set(observed) | set(missing) == expected_segments,
        f"{name} observed/missing segment partition is invalid",
    )
    # This frozen artifact deliberately remains limited: it measures 16/24
    # identities across the full sequence and zero fully-covered transitions.
    # Requiring the negative claims protects the readiness output from turning
    # parser success into a visual-stability assertion.
    expected_missing = {
        "Segment_05",
        "Segment_06",
        "Segment_07",
        "Segment_13",
        "Segment_14",
        "Segment_21",
        "Segment_22",
        "Segment_23",
    }
    _require(
        set(missing) == expected_missing,
        f"{name} frozen 16/24 identity union changed",
    )
    _require(
        coverage["identity_union_all_24"] is False
        and coverage["every_transition_all_24"] is False
        and complete == 0,
        f"{name} must not overclaim complete visual tooth coverage",
    )
    counts = _mapping(
        coverage["per_segment_transition_counts"],
        f"{name}.per_segment_transition_counts",
    )
    _require(
        set(counts) == expected_segments,
        f"{name} per-segment counts must contain 24 IDs",
    )
    for segment, raw_count in counts.items():
        _require(
            type(raw_count) is int and 0 <= raw_count <= transitions,
            f"{name}.{segment} transition count is invalid",
        )
        _require(
            (raw_count == 0) == (segment in expected_missing),
            f"{name}.{segment} count conflicts with identity union",
        )
    phases = _mapping(coverage["per_phase"], f"{name}.per_phase")
    expected_phases = {
        "nut_only_final_hold",
        "q7_twist_probe_motion",
        "q7_twist_probe_hold",
    }
    _require(set(phases) == expected_phases, f"{name} phases changed")
    for phase, raw_phase in phases.items():
        phase_value = _mapping(raw_phase, f"{name}.per_phase.{phase}")
        _exact_keys(
            phase_value,
            name=f"{name}.per_phase.{phase}",
            required={
                "complete_24_segment_transitions",
                "identity_union_all_24",
                "minimum_segments_per_transition",
                "missing_from_identity_union",
                "segments_in_identity_union",
                "transitions",
            },
        )
        _positive_integer(
            phase_value["transitions"],
            f"{name}.per_phase.{phase}.transitions",
        )
        _positive_integer(
            phase_value["minimum_segments_per_transition"],
            f"{name}.per_phase.{phase}.minimum_segments_per_transition",
        )
        _require(
            phase_value["identity_union_all_24"] is False
            and phase_value["complete_24_segment_transitions"] == 0,
            f"{name}.per_phase.{phase} overclaims 24-tooth coverage",
        )
        phase_observed = set(
            _string_tuple(
                phase_value["segments_in_identity_union"],
                f"{name}.per_phase.{phase}.segments_in_identity_union",
            )
        )
        phase_missing = set(
            _string_tuple(
                phase_value["missing_from_identity_union"],
                f"{name}.per_phase.{phase}.missing_from_identity_union",
            )
        )
        _require(
            phase_observed.isdisjoint(phase_missing)
            and phase_observed | phase_missing == expected_segments,
            f"{name}.per_phase.{phase} segment partition is invalid",
        )


def _validate_tooth_four_view_sync(
    evidence: LimitedEvidenceSpec, paths: Mapping[str, Path]
) -> None:
    """Validate the formal three-run, four-view limited tooth artifact."""

    manifest = _json_document(paths["evidence_manifest"], "tooth manifest")
    report = _json_document(paths["evidence_report"], "tooth evidence report")
    _exact_keys(
        manifest,
        name="tooth manifest",
        required={
            "schema_version",
            "status",
            "capture_bundles",
            "indirect_frame_binding",
            "outputs",
            "sources",
        },
    )
    _require(
        manifest["schema_version"]
        == "kcg_d38999_tooth_sync_evidence_manifest_v1",
        "tooth evidence manifest schema mismatch",
    )
    _require(
        manifest["status"] == "HASH_SIZE_SCHEMA_BOUND",
        "tooth evidence manifest status mismatch",
    )
    indirect = _mapping(
        manifest["indirect_frame_binding"], "tooth manifest frame binding"
    )
    _require(
        indirect
        == {
            "all_png_hashes_revalidated": True,
            "mechanism": "capture_manifest_sha256_plus_per_png_sha256_map",
        },
        "tooth PNG hash-binding claim mismatch",
    )

    output_records = _mapping(manifest["outputs"], "tooth manifest outputs")
    source_records = _mapping(manifest["sources"], "tooth manifest sources")
    capture_records = _mapping(
        manifest["capture_bundles"], "tooth manifest captures"
    )
    _require(
        set(output_records)
        == {"all_view_ab_residuals", "all_view_per_tooth_residuals", "report"},
        "tooth manifest output set changed",
    )
    _require(
        set(source_records)
        == {"analysis_source", "capture_helper", "runner_source_snapshot"},
        "tooth manifest source set changed",
    )
    _require(
        set(capture_records)
        == {"baseline", "rtx_history_512", "segment00_normalized"},
        "tooth manifest capture run set changed",
    )
    expected_records = {
        "ab_residuals": output_records["all_view_ab_residuals"],
        "evidence_report": output_records["report"],
        "run_residuals": output_records["all_view_per_tooth_residuals"],
        "analyzer_source": source_records["analysis_source"],
        "capture_helper": source_records["capture_helper"],
        "runner_source": source_records["runner_source_snapshot"],
    }
    capture_prefixes = {
        "baseline": "baseline",
        "rtx_history_512": "history512",
        "segment00_normalized": "normalized",
    }
    for run_id, prefix in capture_prefixes.items():
        record = _mapping(
            capture_records[run_id], f"tooth manifest capture {run_id}"
        )
        _exact_keys(
            record,
            name=f"tooth manifest capture {run_id}",
            required={
                "capture_directory",
                "capture_manifest",
                "physics_report",
                "physics_summary",
                "sync_csv",
            },
        )
        _relative_path(
            record["capture_directory"],
            f"tooth manifest capture {run_id}.capture_directory",
        )
        expected_records[f"{prefix}_capture_manifest"] = record[
            "capture_manifest"
        ]
        expected_records[f"{prefix}_physics_report"] = record[
            "physics_report"
        ]
        expected_records[f"{prefix}_physics_summary"] = record[
            "physics_summary"
        ]
        expected_records[f"{prefix}_sync_csv"] = record["sync_csv"]
    _require(
        set(expected_records) == set(evidence.artifacts) - {
            "aggregator_source",
            "evidence_manifest",
        },
        "tooth manifest/config artifact keys differ",
    )
    for key, raw_record in expected_records.items():
        record = _mapping(raw_record, f"tooth manifest binding {key}")
        _require(
            record == _binding_document(evidence.artifacts[key]),
            f"tooth manifest binding differs for {key}",
        )

    _exact_keys(
        report,
        name="tooth evidence report",
        required={
            "schema_version",
            "classification",
            "evidence_valid",
            "physics",
            "treatments",
            "visual",
            "scope",
            "limitations",
        },
    )
    _require(
        report["schema_version"] == "kcg_d38999_tooth_sync_evidence_v1",
        "tooth evidence report schema mismatch",
    )
    _require(
        report["classification"] == "VALID_LIMITED_VISUAL_JITTER_UNRESOLVED"
        and report["evidence_valid"] is True,
        "tooth evidence must remain valid-limited and unresolved",
    )
    physics = _mapping(report["physics"], "tooth evidence physics")
    _require(
        physics.get(
            "all_three_runs_exclude_independent_tooth_motion_above_"
            "diagnostic_threshold"
        )
        is True,
        "tooth physical relative-motion gate failed",
    )
    trace_digest = _nonempty_string(
        physics.get("identical_physics_trace_sha256"),
        "tooth evidence physics trace SHA256",
    )
    _require(
        _SHA256_RE.fullmatch(trace_digest) is not None,
        "tooth evidence physics trace SHA256 is invalid",
    )
    physics_runs = _mapping(
        physics.get("runs"), "tooth evidence physics runs"
    )
    _require(
        set(physics_runs)
        == {"baseline", "rtx_history_512", "segment00_normalized"},
        "tooth evidence physics run set changed",
    )
    for run_id, raw_run in physics_runs.items():
        run = _mapping(raw_run, f"tooth evidence physics {run_id}")
        _require(
            run.get("all_24_segments_tracked") is True
            and run.get("relative_motion_below_diagnostic_threshold") is True
            and run.get("anomaly_steps") == 0
            and run.get("steps") == 5590,
            f"tooth evidence physical gate failed for {run_id}",
        )
        _require(
            run.get("thresholds")
            == {"rotation_rad": 1.0e-5, "translation_m": 1.0e-6},
            f"tooth evidence thresholds changed for {run_id}",
        )

    visual = _mapping(report["visual"], "tooth evidence visual")
    for key in (
        "all_three_capture_bundles_hash_and_frame_validated",
        "all_three_single_run_analyses_authorized",
        "both_priority_view_ab_comparisons_authorized",
    ):
        _require(visual.get(key) is True, f"tooth visual gate failed: {key}")
    _require(
        visual.get("strict_unknown_single_tooth_temporal_coverage_passed")
        is False,
        "tooth visual coverage must remain explicitly incomplete",
    )
    _require(
        visual.get("render_jitter_absence_claim_authorized") is False,
        "tooth evidence must not authorize a no-render-jitter claim",
    )
    blockers = set(
        _string_tuple(visual.get("claim_blockers"), "tooth visual blockers")
    )
    _require(
        blockers
        == {
            "not_all_24_segment_ids_are_measurable_at_every_"
            "sampled_transition",
            "no_visual_residual_acceptance_threshold_was_preregistered",
            "30_hz_sampling_cannot_exclude_between_sample_render_artifacts",
        },
        "tooth visual claim blockers changed",
    )
    run_coverage = _mapping(
        visual.get("run_all_view_coverage"), "tooth run visual coverage"
    )
    ab_coverage = _mapping(
        visual.get("ab_all_view_coverage"), "tooth A/B visual coverage"
    )
    _require(
        set(run_coverage)
        == {"baseline", "rtx_history_512", "segment00_normalized"},
        "tooth run visual coverage set changed",
    )
    _require(
        set(ab_coverage)
        == {
            "baseline_vs_rtx_history_512",
            "baseline_vs_segment00_normalized",
        },
        "tooth A/B visual coverage set changed",
    )
    for run_id, coverage in run_coverage.items():
        _validate_four_view_coverage(coverage, f"run coverage {run_id}")
    for comparison_id, coverage in ab_coverage.items():
        _validate_four_view_coverage(
            coverage, f"A/B coverage {comparison_id}"
        )


def _validate_bound_artifact(
    value: Any, artifact: ArtifactSpec, name: str
) -> None:
    """Validate one nested path/hash/size record against the outer manifest."""

    binding = _mapping(value, name)
    _exact_keys(
        binding,
        name=name,
        required={"path", "sha256", "size_bytes"},
    )
    _path_has_suffix(binding["path"], artifact.path, f"{name}.path")
    _require(
        binding["sha256"] == artifact.sha256,
        f"{name}.sha256 does not match the outer immutable binding",
    )
    _require(
        binding["size_bytes"] == artifact.size_bytes,
        f"{name}.size_bytes does not match the outer immutable binding",
    )


def _validate_binding_section(
    section: Mapping[str, Any],
    expected: Mapping[str, str],
    evidence: LimitedEvidenceSpec,
    name: str,
) -> None:
    _require(set(section) == set(expected), f"{name} keys changed")
    for key, artifact_key in expected.items():
        _validate_bound_artifact(
            section[key], evidence.artifacts[artifact_key], f"{name}.{key}"
        )


def _validate_tooth_six_view_identity_v2(
    evidence: LimitedEvidenceSpec, paths: Mapping[str, Path]
) -> None:
    """Validate six-view plus posthoc Segment23 as limited evidence only."""

    expected_segments = {f"Segment_{index:02d}" for index in range(24)}
    expected_phases = {
        "nut_only_final_hold",
        "q7_twist_probe_motion",
        "q7_twist_probe_hold",
    }
    six_manifest = _json_document(
        paths["six_view_manifest"], "six-view tooth manifest"
    )
    _exact_keys(
        six_manifest,
        name="six-view tooth manifest",
        required={
            "schema_version",
            "status",
            "indirect_frame_binding",
            "inputs",
            "outputs",
            "sources",
        },
    )
    _require(
        six_manifest["schema_version"]
        == "kcg_d38999_tooth_axial_evidence_manifest_v1"
        and six_manifest["status"] == "HASH_SIZE_SCHEMA_BOUND",
        "six-view manifest is not hash/size/schema bound",
    )
    _validate_binding_section(
        _mapping(six_manifest["inputs"], "six-view inputs"),
        {
            "axial_capture_manifest": "axial_capture_manifest",
            "axial_ghost_bundle": "axial_ghost_bundle",
            "base_capture_manifest": "base_capture_manifest",
            "ghost_manifest": "ghost_manifest",
            "ghost_visibility_sidecar": "ghost_visibility_sidecar",
            "physics_report": "physics_report",
            "physics_summary": "physics_summary",
            "run_log": "run_log",
        },
        evidence,
        "six-view inputs",
    )
    _validate_binding_section(
        _mapping(six_manifest["outputs"], "six-view outputs"),
        {
            "axial_all_view_residuals": "axial_all_view_residuals",
            "report": "six_view_report",
            "six_view_residuals": "six_view_residuals",
        },
        evidence,
        "six-view outputs",
    )
    _validate_binding_section(
        _mapping(six_manifest["sources"], "six-view sources"),
        {
            "axial_capture": "axial_capture",
            "axial_evidence": "axial_evidence",
            "axial_wrapper": "axial_wrapper",
            "base_analysis": "base_analysis",
            "base_capture": "base_capture",
            "occlusion_evidence": "occlusion_evidence",
            "prepared_runner": "prepared_runner_source",
            "sync_evidence": "sync_evidence",
        },
        evidence,
        "six-view sources",
    )
    frame_binding = _mapping(
        six_manifest["indirect_frame_binding"],
        "six-view indirect frame binding",
    )
    _exact_keys(
        frame_binding,
        name="six-view indirect frame binding",
        required={
            "mechanism",
            "rgb_frames_revalidated",
            "all_png_hashes_revalidated",
        },
    )
    _require(
        frame_binding["mechanism"]
        == "base_and_axial_manifest_per_png_sha256_maps"
        and frame_binding["rgb_frames_revalidated"] == 1590
        and frame_binding["all_png_hashes_revalidated"] is True,
        "six-view frame hash binding is incomplete",
    )

    segment_manifest = _json_document(
        paths["segment23_manifest"], "Segment23 identity manifest"
    )
    _exact_keys(
        segment_manifest,
        name="Segment23 identity manifest",
        required={
            "schema_version",
            "status",
            "indirect_frame_binding",
            "inputs",
            "outputs",
            "sources",
        },
    )
    _require(
        segment_manifest["schema_version"]
        == "kcg_d38999_segment23_identity_reanalysis_manifest_v1"
        and segment_manifest["status"] == "HASH_SIZE_SCHEMA_BOUND",
        "Segment23 manifest is not hash/size/schema bound",
    )
    _validate_binding_section(
        _mapping(segment_manifest["inputs"], "Segment23 manifest inputs"),
        {
            "axial_capture_manifest": "axial_capture_manifest",
            "connector_asset": "connector_asset",
            "physics_report": "physics_report",
            "physics_summary": "physics_summary",
            "run_log": "run_log",
            "upstream_manifest": "six_view_manifest",
            "upstream_report": "six_view_report",
        },
        evidence,
        "Segment23 manifest inputs",
    )
    _validate_binding_section(
        _mapping(segment_manifest["outputs"], "Segment23 manifest outputs"),
        {
            "assignments": "segment23_assignments",
            "report": "segment23_report",
        },
        evidence,
        "Segment23 manifest outputs",
    )
    _validate_binding_section(
        _mapping(segment_manifest["sources"], "Segment23 manifest sources"),
        {
            "identity_reanalysis": "segment23_reanalysis_source",
            "upstream_base_analysis": "base_analysis",
        },
        evidence,
        "Segment23 manifest sources",
    )
    segment_frame_binding = _mapping(
        segment_manifest["indirect_frame_binding"],
        "Segment23 indirect frame binding",
    )
    _exact_keys(
        segment_frame_binding,
        name="Segment23 indirect frame binding",
        required={
            "mechanism",
            "segment23_png_hashes_revalidated",
            "sync_csv_sha256_revalidated",
        },
    )
    _require(
        segment_frame_binding["mechanism"]
        == "axial_manifest_per_png_sha256_map"
        and segment_frame_binding["segment23_png_hashes_revalidated"] == 265
        and segment_frame_binding["sync_csv_sha256_revalidated"] is True,
        "Segment23 frame/sync binding is incomplete",
    )

    base_capture = _json_document(
        paths["base_capture_manifest"], "base four-view capture manifest"
    )
    axial_capture = _json_document(
        paths["axial_capture_manifest"], "axial capture manifest"
    )
    _require(
        base_capture.get("schema_version")
        == "kcg_d38999_tooth_sync_capture_v3"
        and base_capture.get("passed") is True,
        "base four-view capture manifest is not a passing v3 capture",
    )
    _require(
        axial_capture.get("schema_version")
        == "kcg_d38999_tooth_axial_capture_v1"
        and axial_capture.get("passed") is True,
        "axial capture manifest is not a passing v1 capture",
    )
    base_source = _mapping(
        base_capture.get("capture_source"), "base capture source provenance"
    )
    axial_source = _mapping(
        axial_capture.get("capture_source"), "axial source provenance"
    )
    for stage in ("sha256_at_import", "sha256_at_start", "sha256_at_finalize"):
        _require(
            base_source.get(stage)
            == evidence.artifacts["base_capture"].sha256,
            f"base capture {stage} does not match immutable source bytes",
        )
        _require(
            axial_source.get(stage)
            == evidence.artifacts["axial_capture"].sha256,
            f"axial capture {stage} does not match immutable source bytes",
        )
    _require(
        base_source.get("unchanged_during_capture") is True,
        "base capture source changed during execution",
    )
    axial_provenance = _mapping(
        axial_capture.get("provenance"), "axial execution provenance"
    )
    _validate_bound_artifact(
        axial_provenance.get("prepared_runner"),
        evidence.artifacts["prepared_runner_source"],
        "axial execution provenance.prepared_runner",
    )
    _validate_bound_artifact(
        axial_provenance.get("wrapper"),
        evidence.artifacts["axial_wrapper"],
        "axial execution provenance.wrapper",
    )
    _require(
        axial_provenance.get("runner_sha256_at_start")
        == evidence.artifacts["prepared_runner_source"].sha256
        and axial_provenance.get("wrapper_sha256_at_start")
        == evidence.artifacts["axial_wrapper"].sha256
        and axial_provenance.get("unchanged_during_capture") is True,
        "axial prepared runner/wrapper execution provenance is not stable",
    )
    for capture, expected_frames, expected_views, name in (
        (
            base_capture,
            1060,
            {"rear_left", "rear_right", "front_left", "front_right"},
            "base capture",
        ),
        (
            axial_capture,
            530,
            {"axial_segment13", "axial_segment23"},
            "axial capture",
        ),
    ):
        sampling = _mapping(capture.get("sampling"), f"{name}.sampling")
        _require(
            sampling.get("capture_rate_hz") == 30
            and sampling.get("physics_rate_hz") == 240
            and sampling.get("physics_steps_per_frame") == 8,
            f"{name} sampling contract changed",
        )
        frame_capture = _mapping(
            capture.get("frame_capture"), f"{name}.frame_capture"
        )
        _require(
            frame_capture.get("frame_count") == expected_frames
            and frame_capture.get("sample_count") == 265
            and set(frame_capture.get("view_order", ())) == expected_views,
            f"{name} frame/view counts changed",
        )
        cleanup = _mapping(capture.get("cleanup"), f"{name}.cleanup")
        _require(
            cleanup.get("object_pose_writes") == 0
            and cleanup.get("errors") == []
            and cleanup.get("resources_released") is True,
            f"{name} cleanup or zero-pose-write contract failed",
        )
    _require(
        axial_capture.get("same_sample_keys_as_base_four_views") is True,
        "axial and base captures do not share the same 265 sample keys",
    )

    ghost_manifest = _json_document(paths["ghost_manifest"], "ghost manifest")
    _require(
        ghost_manifest.get("schema_version")
        == "kcg_d38999_tooth_ghost_manifest_v1"
        and ghost_manifest.get("status") == "HASH_SIZE_SCHEMA_BOUND",
        "ghost manifest is not hash/size/schema bound",
    )
    _validate_binding_section(
        _mapping(ghost_manifest.get("inputs"), "ghost inputs"),
        {
            "capture_manifest": "base_capture_manifest",
            "physics_report": "physics_report",
            "physics_summary": "physics_summary",
        },
        evidence,
        "ghost inputs",
    )
    _validate_binding_section(
        _mapping(ghost_manifest.get("outputs"), "ghost outputs"),
        {"visibility_sidecar": "ghost_visibility_sidecar"},
        evidence,
        "ghost outputs",
    )
    _validate_binding_section(
        _mapping(ghost_manifest.get("sources"), "ghost sources"),
        {
            "contact_fingerprint_contract": "occlusion_control",
            "prepared_tooth_runner": "prepared_runner_source",
            "runtime": "ghost_runtime",
        },
        evidence,
        "ghost sources",
    )

    main_reports = []
    run_content = paths["run_log"].read_text(encoding="utf-8")
    _require(
        run_content.count("ISAAC D38999 Q7 TWIST PROBE V1 PASSED") == 1
        and "Traceback (most recent call last)" not in run_content
        and "ISAAC D38999 Q7 TWIST PROBE V1 FAILED" not in run_content,
        "execution log process markers are not a unique clean PASS",
    )
    for line in run_content.splitlines():
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(candidate, Mapping)
            and candidate.get("scene") == "kcg_d38999_nut_regrasp_physx_v1"
        ):
            main_reports.append(candidate)
    _require(
        len(main_reports) == 1,
        "execution log must contain exactly one prepared main report",
    )
    main_report = main_reports[0]
    _require(
        main_report.get("passed") is True
        and main_report.get("object_pose_writes_after_start") == 0,
        "execution main report did not pass without runtime pose writes",
    )
    ghost_runtime = _mapping(
        main_report.get("nut_tooth_ghost_fingers"),
        "execution ghost runtime report",
    )
    ghost_source = _mapping(
        ghost_runtime.get("source"), "execution ghost source provenance"
    )
    ghost_mutations = _mapping(
        ghost_runtime.get("mutation_audit"), "execution ghost mutation audit"
    )
    _require(
        ghost_runtime.get("passed") is True
        and all(value == 0 for value in ghost_mutations.values()),
        "ghost runtime was not visibility-only and zero-write",
    )
    for stage in ("sha256_at_import", "sha256_at_start", "sha256_at_finalize"):
        _require(
            ghost_source.get(stage)
            == evidence.artifacts["ghost_runtime"].sha256,
            f"ghost runtime {stage} does not match immutable source bytes",
        )
    _require(
        ghost_source.get("runner_sha256_at_start")
        == evidence.artifacts["prepared_runner_source"].sha256
        and ghost_source.get("runner_sha256_at_finalize")
        == evidence.artifacts["prepared_runner_source"].sha256
        and ghost_source.get("unchanged_during_run") is True
        and ghost_source.get("runner_unchanged_during_run") is True,
        "ghost/runner source bytes changed during execution",
    )

    physics_report = _json_document(paths["physics_report"], "physics report")
    _validate_tooth_report(physics_report, "baseline", {})
    _require(
        physics_report["steps"] == 5590
        and len(physics_report["segment_aggregate"]) == 24
        and physics_report["anomaly_steps"] == 0,
        "physics evidence must remain exactly 5590x24 with anomaly_steps=0",
    )

    six_report = _json_document(paths["six_view_report"], "six-view report")
    _exact_keys(
        six_report,
        name="six-view report",
        required={
            "schema_version",
            "classification",
            "evidence_valid",
            "capture",
            "ghost_runtime",
            "limitations",
            "physics",
            "posthoc_binding",
            "process_result",
            "visual",
        },
    )
    _require(
        six_report["schema_version"] == "kcg_d38999_tooth_axial_evidence_v1"
        and six_report["classification"]
        == "VALID_SIX_VIEW_PARTIAL_COVERAGE_JITTER_UNRESOLVED"
        and six_report["evidence_valid"] is True,
        "six-view report classification/schema changed",
    )
    report_physics = _mapping(six_report["physics"], "six-view report physics")
    _require(
        report_physics.get("steps") == 5590
        and report_physics.get("all_24_segments_tracked") is True
        and report_physics.get("anomaly_steps") == 0,
        "six-view report does not project the 5590x24 anomaly-free trace",
    )
    visual = _mapping(six_report["visual"], "six-view visual evidence")
    coverage = _mapping(
        visual.get("six_view_coverage"), "six-view identity coverage"
    )
    _require(
        coverage.get("transitions") == 262
        and coverage.get("complete_24_segment_transitions") == 0
        and coverage.get("every_transition_all_24") is False
        and coverage.get("identity_union_all_24") is False
        and coverage.get("missing_from_identity_union") == ["Segment_23"]
        and set(coverage.get("segments_in_identity_union", ()))
        == expected_segments - {"Segment_23"},
        "six-view visual record must remain a 23/24 union with no complete "
        "transition",
    )
    _require(
        visual.get("render_jitter_absence_claim_authorized") is False
        and visual.get("strict_every_transition_all_24") is False,
        "six-view visual record must not authorize no-jitter or strict "
        "coverage",
    )
    six_limitations = set(
        _string_tuple(six_report["limitations"], "six-view limitations")
    )
    _require(
        {
            "identity_union_is_not_every_transition_coverage",
            "no_visual_residual_acceptance_threshold_was_preregistered",
            "30_hz_sampling_cannot_exclude_between_sample_render_artifacts",
        }.issubset(six_limitations),
        "six-view limitations omit required non-promotion boundaries",
    )

    transition_segments: dict[int, set[str]] = {}
    with paths["six_view_residuals"].open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        required_columns = {
            "global_step",
            "phase",
            "phase_step",
            "previous_global_step",
            "segment",
            "view_id",
            "residual_x_px",
            "residual_y_px",
            "residual_px",
            "tooth_pitch_px",
            "residual_pitch_fraction",
            "registered_scale",
        }
        _require(
            set(reader.fieldnames or ()) == required_columns,
            "six-view residual CSV columns changed",
        )
        residual_rows = 0
        for row in reader:
            residual_rows += 1
            step = int(row["global_step"])
            segment = row["segment"]
            _require(
                segment in expected_segments
                and row["phase"] in expected_phases,
                "six-view residual row has an invalid segment or phase",
            )
            for key in (
                "residual_x_px",
                "residual_y_px",
                "residual_px",
                "tooth_pitch_px",
                "residual_pitch_fraction",
                "registered_scale",
            ):
                _finite_number(
                    float(row[key]), f"six-view residual row {key}"
                )
            transition_segments.setdefault(step, set()).add(segment)
    _require(
        residual_rows == 4747 and len(transition_segments) == 262,
        "six-view residual CSV row/transition count changed",
    )

    segment_report = _json_document(
        paths["segment23_report"], "Segment23 identity report"
    )
    _exact_keys(
        segment_report,
        name="Segment23 identity report",
        required={
            "schema_version",
            "classification",
            "evidence_valid",
            "fresh_capture_recommendation",
            "identity_result",
            "limitations",
            "physics_result",
            "visual_diagnostics_only",
        },
    )
    _require(
        segment_report["schema_version"]
        == "kcg_d38999_segment23_identity_reanalysis_v1"
        and segment_report["classification"]
        == (
            "VALID_GEOMETRY_RECOVERED_SEGMENT23_IDENTITY_"
            "RENDER_JITTER_UNRESOLVED"
        )
        and segment_report["evidence_valid"] is True,
        "Segment23 report classification/schema changed",
    )
    identity = _mapping(
        segment_report["identity_result"], "Segment23 identity result"
    )
    _require(
        identity.get("segment") == "Segment_23"
        and identity.get("existing_frame_count") == 265
        and identity.get("geometry_identity_recovered") is True
        and identity.get("hue_identity_recovered") is False
        and identity.get("hue_gate_widened") is False
        and identity.get("pixels_relabelled_or_modified") is False
        and identity.get("per_frame_actual_physical_parent_pose_used") is True
        and identity.get("all_frames_mutual_nearest") is True
        and identity.get("all_frames_within_one_third_projected_pitch") is True
        and identity.get(
            "all_frames_with_more_than_half_pitch_identity_margin"
        )
        is True
        and identity.get("modality")
        == (
            "physics_and_CAD_projection_assisted_posthoc_identity_recovery_"
            "not_RGB_only"
        ),
        "Segment23 recovery must remain CAD/physics-assisted, posthoc, and "
        "non-RGB-only",
    )
    segment_physics = _mapping(
        segment_report["physics_result"], "Segment23 physics result"
    )
    _require(
        segment_physics.get("all_24_segments_one_rigid_parent_trace") is True
        and segment_physics.get("anomaly_steps") == 0
        and segment_physics.get(
            "independent_physical_segment23_motion_observed"
        )
        is False,
        "Segment23 physics result changed",
    )
    for key, value in _mapping(
        segment_physics.get("segment23_maximum_relative_errors"),
        "Segment23 relative errors",
    ).items():
        _require(
            _finite_number(value, f"Segment23 relative error {key}") == 0.0,
            f"Segment23 relative error {key} is not zero",
        )
    diagnostics = _mapping(
        segment_report["visual_diagnostics_only"],
        "Segment23 visual diagnostics",
    )
    _require(
        diagnostics.get("identity_match_is_render_no_jitter_claim") is False
        and diagnostics.get("render_jitter_absence_claim_authorized") is False,
        "Segment23 posthoc identity must not become a render no-jitter claim",
    )
    segment_limitations = set(
        _string_tuple(
            segment_report["limitations"], "Segment23 limitations"
        )
    )
    _require(
        {
            "posthoc_geometry_identity_is_not_preregistered_jitter_acceptance",
            "30_hz_sampling_cannot_exclude_between_sample_render_artifacts",
            "physics_zero_anomaly_does_not_prove_renderer_zero_jitter",
        }.issubset(segment_limitations),
        "Segment23 limitations omit required render-jitter boundaries",
    )

    assignment_steps = set()
    projection_fractions = []
    margin_fractions = []
    with paths["segment23_assignments"].open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        required_columns = {
            "candidate_hue_label",
            "candidate_xy_px",
            "projected_xy_px",
            "projection_error_px",
            "projection_error_pitch_fraction",
            "projected_identity_margin_px",
            "projected_identity_margin_pitch_fraction",
            "projected_neighbour_pitch_px",
            "candidate_area_px",
            "global_step",
            "phase",
            "phase_step",
            "sample_index",
        }
        _require(
            set(reader.fieldnames or ()) == required_columns,
            "Segment23 assignment CSV columns changed",
        )
        sample_indices = set()
        for row in reader:
            _require(
                row["candidate_hue_label"] == "Segment_22"
                and row["phase"] in expected_phases,
                "Segment23 assignment changed hue alias or phase",
            )
            sample_indices.add(int(row["sample_index"]))
            assignment_steps.add(int(row["global_step"]))
            projection = _finite_number(
                float(row["projection_error_pitch_fraction"]),
                "Segment23 assignment projection fraction",
            )
            margin = _finite_number(
                float(row["projected_identity_margin_pitch_fraction"]),
                "Segment23 assignment identity margin fraction",
            )
            _require(
                projection < (1.0 / 3.0) and margin > 0.5,
                "Segment23 assignment violates the posthoc correspondence "
                "gate",
            )
            for vector_key in ("candidate_xy_px", "projected_xy_px"):
                vector = json.loads(row[vector_key])
                _require(
                    isinstance(vector, list)
                    and len(vector) == 2
                    and all(
                        math.isfinite(float(component)) for component in vector
                    ),
                    f"Segment23 assignment {vector_key} is invalid",
                )
            projection_fractions.append(projection)
            margin_fractions.append(margin)
    _require(
        sample_indices == set(range(265))
        and len(assignment_steps) == 265
        and set(transition_segments).issubset(assignment_steps),
        "Segment23 assignments do not cover the existing 265 synchronized "
        "frames",
    )
    report_projection = _mapping(
        identity.get("projection_error_pitch_fraction"),
        "Segment23 report projection fractions",
    )
    report_margin = _mapping(
        identity.get("projected_identity_margin_pitch_fraction"),
        "Segment23 report identity margins",
    )
    _require(
        math.isclose(
            max(projection_fractions),
            report_projection.get("maximum"),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            min(margin_fractions),
            report_margin.get("minimum"),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "Segment23 report extrema do not match the assignment CSV",
    )
    combined_union = set(coverage["segments_in_identity_union"])
    combined_union.add(identity["segment"])
    _require(
        combined_union == expected_segments,
        "six-view plus CAD/physics-assisted posthoc identity union is not "
        "24/24",
    )
    complete_after_posthoc = sum(
        len(segments | {"Segment_23"}) == 24
        for segments in transition_segments.values()
    )
    _require(
        complete_after_posthoc == 0,
        "posthoc identity must not claim any per-transition 24/24 coverage",
    )


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_observed_scalar_statistics(value: Any, name: str) -> None:
    statistics_document = _mapping(value, name)
    _exact_keys(
        statistics_document,
        name=name,
        required={
            "sample_count",
            "mean",
            "sample_standard_deviation",
            "minimum_observed",
            "maximum_observed",
            "observed_span",
        },
    )
    _require(
        statistics_document["sample_count"] == 3,
        f"{name}.sample_count must equal 3",
    )
    numeric = {
        key: _finite_number(statistics_document[key], f"{name}.{key}")
        for key in (
            "mean",
            "sample_standard_deviation",
            "minimum_observed",
            "maximum_observed",
            "observed_span",
        )
    }
    mean_within_observed = (
        numeric["minimum_observed"] <= numeric["mean"]
        <= numeric["maximum_observed"]
        or math.isclose(
            numeric["mean"],
            numeric["minimum_observed"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or math.isclose(
            numeric["mean"],
            numeric["maximum_observed"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )
    _require(
        numeric["sample_standard_deviation"] >= 0.0
        and mean_within_observed
        and math.isclose(
            numeric["observed_span"],
            numeric["maximum_observed"] - numeric["minimum_observed"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        f"{name} observed statistics are inconsistent",
    )


def _validate_observed_statistics(value: Any) -> None:
    observed = _mapping(value, "wrist FT observed_statistics")
    _exact_keys(
        observed,
        name="wrist FT observed_statistics",
        required={"wrench_vectors", "protected_phase_absolute_peaks"},
    )
    vectors = _mapping(
        observed["wrench_vectors"], "wrist FT wrench vector statistics"
    )
    _exact_keys(
        vectors,
        name="wrist FT wrench vector statistics",
        required={
            "home_empty_baseline_canonical",
            "payload_baseline_canonical",
            "payload_increment_estimate_canonical",
            "last_sample_canonical_wrench_sensor",
        },
    )
    axis_order = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    for vector_name, raw_vector in vectors.items():
        vector = _mapping(raw_vector, f"wrist FT statistics {vector_name}")
        _exact_keys(
            vector,
            name=f"wrist FT statistics {vector_name}",
            required={
                "sample_count",
                "axis_order",
                "per_axis_observed",
                "maximum_observed_pairwise_l2_distance",
            },
        )
        _require(
            vector["sample_count"] == 3 and vector["axis_order"] == axis_order,
            f"wrist FT statistics {vector_name} run/axis contract mismatch",
        )
        distance = _finite_number(
            vector["maximum_observed_pairwise_l2_distance"],
            f"wrist FT statistics {vector_name} pairwise distance",
        )
        _require(distance >= 0.0, "pairwise distance must be non-negative")
        per_axis = _mapping(
            vector["per_axis_observed"],
            f"wrist FT statistics {vector_name}.per_axis_observed",
        )
        _exact_keys(
            per_axis,
            name=f"wrist FT statistics {vector_name}.per_axis_observed",
            required=set(axis_order),
        )
        for axis in axis_order:
            _validate_observed_scalar_statistics(
                per_axis[axis], f"wrist FT {vector_name}.{axis}"
            )
    peaks = _mapping(
        observed["protected_phase_absolute_peaks"],
        "wrist FT protected phase peak statistics",
    )
    _exact_keys(
        peaks,
        name="wrist FT protected phase peak statistics",
        required={"INSERT", "ENGAGE", "SCREW", "HOLD"},
    )
    scalar_names = {
        "lateral_force_n",
        "axial_force_n",
        "bending_torque_nm",
        "tightening_torque_nm",
    }
    for phase, raw_phase in peaks.items():
        phase_document = _mapping(raw_phase, f"wrist FT peaks {phase}")
        _exact_keys(
            phase_document,
            name=f"wrist FT peaks {phase}",
            required=scalar_names,
        )
        for scalar_name in scalar_names:
            _validate_observed_scalar_statistics(
                phase_document[scalar_name],
                f"wrist FT peaks {phase}.{scalar_name}",
            )


def _read_smooth_e2e_log(path: Path, name: str) -> Mapping[str, Any]:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    _require(
        "ISAAC D38999 END TO END V1 PASSED" in lines,
        f"{name} lacks exact E2E PASS banner",
    )
    _require("Traceback" not in content, f"{name} contains a traceback")
    candidates = []
    for line in lines:
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} contains malformed JSON") from error
        if isinstance(value, Mapping) and "end_to_end" in value:
            candidates.append(value)
    _require(
        len(candidates) == 1,
        f"{name} must contain one E2E metrics object",
    )
    metrics = candidates[0]
    for key in (
        "passed",
        "end_to_end_probe_requested",
        "wrist_ft_monitor_requested",
    ):
        _require(metrics.get(key) is True, f"{name}.{key} must be true")
    _require(metrics.get("gui") is False, f"{name}.gui must be false")
    _require(
        metrics.get("motion_profile") == "smooth_demo_v1",
        f"{name}.motion_profile must equal smooth_demo_v1",
    )
    _require(
        metrics.get("pose_preflight_requested") == "masked-rgbd",
        f"{name} must include masked RGBD preflight",
    )
    _require(
        metrics.get("control_pose_provider") == "sim_ground_truth"
        and metrics.get("masked_rgbd_xy_used_for_control") is False
        and metrics.get("truth_orientation_used") is True,
        f"{name} control-authority boundary changed",
    )
    _require(
        metrics.get("object_pose_writes_after_start") == 0,
        f"{name} contains runtime object-pose writes",
    )
    end_to_end = _mapping(metrics["end_to_end"], f"{name}.end_to_end")
    _require(end_to_end.get("passed") is True, f"{name} E2E did not pass")
    for key in (
        "assembly_success_claimed",
        "continuous_collision_verified",
        "real_vision_included",
        "masked_rgbd_xy_used_for_control",
    ):
        _require(
            end_to_end.get(key) is False,
            f"{name}.end_to_end.{key} must remain false",
        )
    _require(
        end_to_end.get("ground_truth_pose_used") is True,
        f"{name} must disclose ground-truth pose use",
    )
    monitor = _mapping(
        metrics.get("virtual_wrist_ft_monitor"), f"{name}.wrist_ft_monitor"
    )
    _require(
        monitor.get("schema_version") == "kcg_d38999_wrist_ft_monitor_v1"
        and monitor.get("status") == "MONITOR_ONLY"
        and monitor.get("monitor_only") is True,
        f"{name} wrist FT monitor schema/status mismatch",
    )
    for key in (
        "home_empty_baseline_canonical",
        "payload_baseline_canonical",
        "payload_increment_estimate_canonical",
    ):
        vector = monitor.get(key)
        _require(
            isinstance(vector, list) and len(vector) == 6,
            f"{name} wrist FT {key} is absent",
        )
        for index, value in enumerate(vector):
            _finite_number(value, f"{name} wrist FT {key}[{index}]")
    _require(
        isinstance(monitor.get("last_sample"), Mapping),
        f"{name} wrist FT last_sample is absent",
    )
    _require(
        monitor.get("calibrated_safety_limits") is None,
        f"{name} calibrated safety limits must remain null",
    )
    for key in (
        "modifies_e2e_pass_gate",
        "residual_v1_enabled",
        "safety_gate_claimed",
        "assembly_success_claimed_from_wrench",
        "dynamic_inertia_compensation_complete",
        "orientation_dependent_gravity_compensation_complete",
    ):
        _require(
            monitor.get(key) is False,
            f"{name} wrist FT {key} must remain false",
        )
    return metrics


def _validate_repeatability_report(
    evidence: LimitedEvidenceSpec,
    paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    report = _json_document(
        paths["repeatability_report"], "wrist FT repeatability report"
    )
    _exact_keys(
        report,
        name="wrist FT repeatability report",
        required={
            "schema_version",
            "status",
            "run_count",
            "minimum_required_runs",
            "axis_order",
            "provenance",
            "observed_statistics",
            "claims",
            "calibrated_safety_limits",
            "generated_safety_thresholds",
        },
    )
    _require(
        report["schema_version"]
        == "kcg_d38999_wrist_ft_repeatability_artifact_v1",
        "wrist FT repeatability schema_version mismatch",
    )
    _require(
        report["status"] == "MONITOR_ONLY_REPEATABILITY_EVIDENCE",
        "wrist FT repeatability status mismatch",
    )
    _require(
        report["run_count"] == 3 and report["minimum_required_runs"] == 3,
        "wrist FT repeatability must contain exactly three runs",
    )
    _require(
        report["axis_order"] == ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "wrist FT repeatability axis order mismatch",
    )
    expected_claims = {
        "monitor_only": True,
        "statistics_only": True,
        "safety_thresholds_generated": False,
        "calibration_claimed": False,
        "training_ready_claimed": False,
        "safety_gate_enabled": False,
        "e2e_gate_modified": False,
    }
    _require(
        report["claims"] == expected_claims,
        "wrist FT repeatability claim boundary mismatch",
    )
    _validate_observed_statistics(report["observed_statistics"])
    _require(
        report["calibrated_safety_limits"] is None
        and report["generated_safety_thresholds"] is None,
        "wrist FT repeatability thresholds must remain null",
    )
    provenance = _mapping(
        report["provenance"], "wrist FT repeatability provenance"
    )
    _exact_keys(
        provenance,
        name="wrist FT repeatability provenance",
        required={
            "runner_source",
            "monitor_config",
            "run_artifacts",
            "all_hashes_verified",
            "duplicate_artifacts_rejected",
        },
    )
    _require(
        provenance["all_hashes_verified"] is True
        and provenance["duplicate_artifacts_rejected"] is True,
        "wrist FT repeatability provenance flags must be true",
    )
    for key in ("runner_source", "monitor_config"):
        record = _mapping(provenance[key], f"repeatability {key}")
        _exact_keys(
            record,
            name=f"repeatability {key}",
            required={"path", "sha256"},
        )
        _require(
            record["path"] == evidence.artifacts[key].path.as_posix()
            and record["sha256"] == evidence.artifacts[key].sha256,
            f"repeatability {key} does not match manifest binding",
        )
    expected_logs = {
        (
            evidence.artifacts[key].path.as_posix(),
            evidence.artifacts[key].sha256,
        )
        for key in ("run_v3_log", "run_v4_log", "run_v5_log")
    }
    actual_logs = set()
    report_hashes = {}
    run_artifacts = provenance["run_artifacts"]
    _require(
        isinstance(run_artifacts, list) and len(run_artifacts) == 3,
        "repeatability provenance must list exactly three runs",
    )
    for index, raw_run in enumerate(run_artifacts):
        run = _mapping(raw_run, f"repeatability run_artifacts[{index}]")
        _exact_keys(
            run,
            name=f"repeatability run_artifacts[{index}]",
            required={"run_id", "metrics_artifact", "monitor_report_sha256"},
        )
        _nonempty_string(run["run_id"], f"repeatability run {index}.run_id")
        artifact = _mapping(
            run["metrics_artifact"], f"repeatability run {index}.artifact"
        )
        _exact_keys(
            artifact,
            name=f"repeatability run {index}.artifact",
            required={"kind", "path", "sha256"},
        )
        _require(
            artifact["kind"] == "headless_e2e_jsonl_log",
            "repeatability run artifact kind mismatch",
        )
        digest = _nonempty_string(
            run["monitor_report_sha256"],
            f"repeatability run {index}.monitor_report_sha256",
        )
        _require(
            _SHA256_RE.fullmatch(digest) is not None,
            "repeatability monitor report SHA256 is invalid",
        )
        actual_logs.add((artifact["path"], artifact["sha256"]))
        report_hashes[artifact["path"]] = digest
    _require(
        actual_logs == expected_logs,
        "repeatability run artifacts do not match manifest bindings",
    )
    return report_hashes


def _validate_ft_repeatability(
    evidence: LimitedEvidenceSpec, paths: Mapping[str, Path]
) -> None:
    report_hashes = _validate_repeatability_report(evidence, paths)
    for key in ("run_v3_log", "run_v4_log", "run_v5_log"):
        metrics = _read_smooth_e2e_log(paths[key], key)
        report_path = evidence.artifacts[key].path.as_posix()
        actual = _canonical_json_sha256(metrics["virtual_wrist_ft_monitor"])
        _require(
            actual == report_hashes[report_path],
            f"{key} monitor report hash does not match repeatability report",
        )


def _validate_smooth_e2e(
    evidence: LimitedEvidenceSpec, paths: Mapping[str, Path]
) -> None:
    report_hashes = _validate_repeatability_report(evidence, paths)
    digests = set()
    for key in ("run_v3_log", "run_v4_log", "run_v5_log"):
        metrics = _read_smooth_e2e_log(paths[key], key)
        digests.add(evidence.artifacts[key].sha256)
        report_path = evidence.artifacts[key].path.as_posix()
        _require(
            _canonical_json_sha256(metrics["virtual_wrist_ft_monitor"])
            == report_hashes[report_path],
            f"{key} is not included in the repeatability report",
        )
        for section in (
            "physical_insertion",
            "full_rotation",
            "proxy_assembly_verification",
            "release_retreat_home",
        ):
            value = _mapping(metrics.get(section), f"{key}.{section}")
            _require(value.get("passed") is True, f"{key}.{section} failed")
    _require(len(digests) == 3, "smooth E2E log hashes must be distinct")


def _check_limited_evidence(
    evidence: LimitedEvidenceSpec, repo_root: Path
) -> LimitedEvidenceResult:
    paths, reasons = _verify_limited_artifacts(evidence, repo_root)
    if evidence.disposition == "superseded":
        # Do not silently bless a historical run after its execution-time
        # source bytes have disappeared.  The stale artifact remains visible
        # for audit, but a new run is required before this ID can be active.
        reasons.insert(
            0,
            "superseded limited evidence is non-current and requires "
            "re-execution with an immutable runner source snapshot",
        )
    elif not reasons:
        validators = {
            "multisite_rgbd_xy_v1": _validate_multisite_rgbd,
            "nut_tooth_four_view_sync_v1": _validate_tooth_four_view_sync,
            "nut_tooth_six_view_identity_v2": (
                _validate_tooth_six_view_identity_v2
            ),
            "wrist_ft_monitor_repeatability_v1": _validate_ft_repeatability,
            "smooth_e2e_repeat_v1": _validate_smooth_e2e,
        }
        try:
            validators[evidence.validator](evidence, paths)
        except (OSError, ValueError, yaml.YAMLError) as error:
            reasons.append(f"schema validation failed: {error}")
    return LimitedEvidenceResult(
        evidence_id=evidence.evidence_id,
        gate_id=evidence.gate_id,
        validator=evidence.validator,
        disposition=evidence.disposition,
        valid=not reasons,
        scope=evidence.scope,
        limitations=evidence.limitations,
        reasons=tuple(reasons),
    )


def check_training_readiness(
    manifest: ReadinessManifest,
    repo_root: str | Path,
) -> ReadinessReport:
    """Evaluate all gates and the explicit long-training enable switch."""
    root = Path(repo_root).expanduser().resolve()
    global_reasons: list[str] = []
    if not root.is_dir():
        global_reasons.append(f"repository root is not a directory: {root}")
    if not manifest.training_enabled:
        global_reasons.append(
            "training.enabled is false; promote only after every gate passes"
        )
    gate_results = tuple(_check_gate(gate, root) for gate in manifest.gates)
    limited_results = tuple(
        _check_limited_evidence(evidence, root)
        for evidence in manifest.limited_evidence
    )
    ready = (
        manifest.training_enabled
        and not global_reasons
        and all(result.passed for result in gate_results)
        and all(
            result.valid or result.disposition == "superseded"
            for result in limited_results
        )
    )
    return ReadinessReport(
        ready=ready,
        training_enabled=manifest.training_enabled,
        manifest_path=str(manifest.config_path),
        gate_results=gate_results,
        limited_evidence_results=limited_results,
        global_reasons=tuple(global_reasons),
    )


def require_training_ready(
    config_path: str | Path,
    repo_root: str | Path,
) -> ReadinessReport:
    """Return a passing report or raise before any long training starts."""
    manifest = load_readiness_manifest(config_path)
    report = check_training_readiness(manifest, repo_root)
    if not report.ready:
        blockers = list(report.global_reasons)
        blockers.extend(
            f"{result.gate_id}: {reason}"
            for result in report.gate_results
            for reason in result.reasons
        )
        blockers.extend(
            f"{result.evidence_id}: {reason}"
            for result in report.limited_evidence_results
            if result.disposition == "active"
            for reason in result.reasons
        )
        raise RuntimeError(
            "full-skill RL training is not ready:\n- "
            + "\n- ".join(blockers)
        )
    return report


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / (
        "d38999_full_skill_rl_readiness_v1.yaml"
    )


def _human_report(report: ReadinessReport) -> str:
    lines = [
        "FULL SKILL RL READINESS: "
        + ("READY" if report.ready else "BLOCKED")
    ]
    for reason in report.global_reasons:
        lines.append(f"[BLOCK] {reason}")
    for result in report.limited_evidence_results:
        if result.disposition == "superseded":
            status = "INVALID SUPERSEDED"
        else:
            status = "VALID LIMITED" if result.valid else "INVALID LIMITED"
        lines.append(
            f"[{status}] {result.evidence_id} -> {result.gate_id} "
            "(never closes full gate)"
        )
        lines.extend(f"  scope: {item}" for item in result.scope)
        lines.extend(f"  limitation: {item}" for item in result.limitations)
        lines.extend(f"  - {reason}" for reason in result.reasons)
    for result in report.gate_results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"[{status}] {result.gate_id} ({result.category})")
        lines.extend(f"  - {reason}" for reason in result.reasons)
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed preflight for full D38999 hierarchical RL training"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="strict readiness manifest",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root used to resolve and hash evidence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full machine-readable report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        manifest = load_readiness_manifest(arguments.config)
        report = check_training_readiness(manifest, arguments.repo_root)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"FULL SKILL RL READINESS: INVALID\n[BLOCK] {error}")
        return 2
    if arguments.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(_human_report(report))
    return 0 if report.ready else 1


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())


__all__ = [
    "ACTION_SIZE",
    "BASE_RESIDUAL_INTERFACE_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "FULL_SKILL_INTERFACE_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "LIMITED_EVIDENCE_ARTIFACTS",
    "LIMITED_EVIDENCE_DISPOSITIONS",
    "OBSERVATION_SIZE",
    "POLICY_ACTIVE_STAGES",
    "REQUIRED_GATE_CATEGORIES",
    "REQUIRED_GATE_IDS",
    "REQUIRED_LIMITED_EVIDENCE_IDS",
    "RESIDUAL_INTERFACE_VERSION",
    "WORKFLOW_STAGES",
    "ArtifactSpec",
    "GateResult",
    "GateSpec",
    "LimitedEvidenceResult",
    "LimitedEvidenceSpec",
    "ReadinessManifest",
    "ReadinessReport",
    "check_training_readiness",
    "load_readiness_manifest",
    "require_training_ready",
]
