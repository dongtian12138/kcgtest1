"""Strict CPU-only evidence aggregation for visual-XY preinsert probes.

The Isaac runs are immutable inputs to this module.  Aggregation never starts
Isaac, imports simulator modules, or changes a run directory.  A versioned
source manifest binds every report, CPU plan and config by repository-relative
path, byte size, SHA-256 and schema.  The aggregate remains deliberately
limited to four independent visual-XY picks that stopped outside the 10 mm
entry datum; engage, insertion, twist, Home, full-6D and production claims are
always false.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


CONFIG_SCHEMA_VERSION = (
    "kcg_d38999_visual_xy_preinsert_evidence_config_v1"
)
SOURCE_MANIFEST_SCHEMA_VERSION = (
    "kcg_d38999_visual_xy_preinsert_evidence_manifest_v1"
)
EVIDENCE_SCHEMA_VERSION = (
    "kcg_d38999_visual_xy_preinsert_evidence_v1"
)
SOURCE_REPORT_SCHEMA_VERSION = "kcg_d38999_visual_xy_pick_report_v1"
PICK_PLAN_SCHEMA_VERSION = "kcg_d38999_visual_xy_pick_probe_v1"
PREINSERT_PLAN_SCHEMA_VERSION = "kcg_d38999_visual_xy_preinsert_probe_v1"
PICK_CONFIG_SCHEMA_VERSION = PICK_PLAN_SCHEMA_VERSION
PREINSERT_CONFIG_SCHEMA_VERSION = PREINSERT_PLAN_SCHEMA_VERSION

EXPECTED_RUN_COUNT = 4
DEFAULT_CONFIG_PATH = (
    "src/kcg_connector/config/"
    "d38999_visual_xy_preinsert_evidence_v1.yaml"
)
DEFAULT_MANIFEST_PATH = (
    "src/kcg_connector/config/"
    "d38999_visual_xy_preinsert_evidence_manifest_v1.json"
)
AGGREGATOR_SOURCE_PATH = (
    "src/kcg_connector/kcg_connector/"
    "d38999_visual_xy_preinsert_evidence.py"
)

MANIFEST_COMPLETE = "COMPLETE_HASH_SIZE_SCHEMA_BOUND"
MANIFEST_INCOMPLETE = "INCOMPLETE_MISSING_SOURCES"
EVIDENCE_PASS = (
    "PASS_LIMITED_FOUR_POSITION_VISUAL_XY_PREINSERT_EVIDENCE"
)

_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_BINDING_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "schema_version",
    "state",
}
_CONFIG_BINDING_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "schema_version",
}
_CONFIG_KEYS = {
    "schema_version",
    "expected_run_count",
    "source_manifest_path",
    "shared_preinsert_config_path",
    "policy",
    "thresholds",
    "runs",
}
_POLICY_KEYS = {
    "evidence_scope",
    "require_all_runs_passed",
    "require_distinct_authored_loose_xy",
    "require_same_authored_fixed_xy",
    "engage_gate_blocking_run_id",
    "statistics_only",
    "claim_engage",
    "claim_insertion",
    "claim_twist",
    "claim_home_return",
    "claim_full_6d",
    "claim_production_control",
    "claim_full_end_to_end",
}
_THRESHOLD_KEYS = {
    "strict_torque_upper_nm",
    "entry_gap_m",
    "commanded_preinsert_gap_m",
    "registered_margin_before_entry_m",
    "maximum_joint_speed_rad_s",
    "maximum_arm_tracking_error_rad",
    "maximum_joint_limit_violation_rad",
    "maximum_tcp_position_error_m",
    "maximum_tcp_axis_error_rad",
    "maximum_engage_lateral_error_m",
    "maximum_engage_axis_error_rad",
    "maximum_engage_combined_entry_error_m",
}
_RUN_KEYS = {
    "run_id",
    "expected_trial_id",
    "expected_authored_loose_xy_m",
    "expected_authored_fixed_xy_m",
    "run_directory",
    "pick_config_path",
}
_SOURCE_NAMES = (
    "report",
    "cpu_plan",
    "preinsert_cpu_plan",
    "pick_config",
    "preinsert_config",
)
_SOURCE_SCHEMAS = {
    "report": SOURCE_REPORT_SCHEMA_VERSION,
    "cpu_plan": PICK_PLAN_SCHEMA_VERSION,
    "preinsert_cpu_plan": PREINSERT_PLAN_SCHEMA_VERSION,
    "pick_config": PICK_CONFIG_SCHEMA_VERSION,
    "preinsert_config": PREINSERT_CONFIG_SCHEMA_VERSION,
}

_REPORT_KEYS = {
    "authored_before_physics",
    "body_lift_m",
    "body_nut_separation_change_m",
    "body_tcp_slip_m",
    "closure_tcp_axis_error_rad",
    "closure_tcp_position_error_m",
    "collision_planned",
    "config_path",
    "config_sha256",
    "contact_gate",
    "contact_torque_deltas_nm",
    "cpu_plan",
    "cpu_plan_path",
    "d38999_authoring",
    "engage_executed",
    "explicit_opt_in",
    "external_contact_records",
    "final_body_observable_angular_speed_rad_s",
    "final_body_observable_linear_speed_m_s",
    "final_bottom_clearance_m",
    "final_contacts",
    "final_hold_displacement_m",
    "final_loaded_torque_channels",
    "final_observable_joint_speed_rad_s",
    "final_solver_joint_speed_rad_s",
    "final_torque_deltas_nm",
    "finite_final",
    "finite_throughout",
    "fixed_rotation_drift_rad",
    "fixed_translation_drift_m",
    "full_6d",
    "grasp_tcp_axis_error_rad",
    "grasp_tcp_position_error_m",
    "gui",
    "home_return_executed",
    "insertion_executed",
    "loaded_torque_channels",
    "maximum_arm_tracking_error_rad",
    "maximum_joint_limit_violation_rad",
    "maximum_joint_speed_rad_s",
    "maximum_post_tare_absolute_delta_nm",
    "object_pose_writes_after_physics",
    "orientation_source",
    "output_directory",
    "passed",
    "pose_provider",
    "postclosure_contacts",
    "preinsert_config_path",
    "preinsert_config_sha256",
    "preinsert_cpu_plan",
    "preinsert_cpu_plan_path",
    "preinsert_probe",
    "preinsert_probe_requested",
    "production_control_authorized",
    "report_path",
    "rgbd_capture",
    "runtime_side_effects",
    "schema_version",
    "settled_on_table",
    "torque_gate",
    "trial_id",
    "truth_evaluation",
    "truth_xy_evaluation_gate",
    "truth_xy_used_for_target",
    "twist_executed",
    "unsupported_gate",
    "zero_forbidden_contacts",
}
_PICK_PLAN_KEYS = {
    "adapter",
    "arm_targets_rad",
    "capture_id",
    "collision_planned",
    "fk_orientation_errors_rad",
    "fk_position_errors_m",
    "full_6d",
    "gpu_or_physx_validated",
    "maximum_abs_joint_delta_from_nominal_rad",
    "orientation_source",
    "planned_peak_joint_speed_rad_s",
    "production_control_authorized",
    "schema_version",
    "status",
    "tcp_targets_world_m",
    "trial_id",
    "uses_truth_xy_for_target",
}
_ADAPTER_KEYS = {
    "capture_id",
    "collision_free_ik_verified",
    "downstream_ik_required",
    "eligible_for_independent_probe",
    "fixed_translation_xy_m",
    "full_6d",
    "loose_translation_xy_m",
    "orientation_source",
    "pose_provider_control_authorized",
    "preserves_nominal_target_z",
    "production_control_authorized",
    "rejection_reasons",
    "schema_version",
    "status",
    "translation_source",
    "uses_truth_orientation",
    "validation_maximum_observed_xy_error_m",
    "world_target_frame",
    "world_targets",
    "yaw_observed",
}
_PREINSERT_PLAN_KEYS = {
    "arm_targets_rad",
    "assembly_success_claimed",
    "capture_id",
    "collision_planned",
    "engage_executed",
    "fixed_translation_xy_m",
    "fk_orientation_errors_rad",
    "fk_position_errors_m",
    "full_6d",
    "gpu_or_physx_validated",
    "maximum_abs_joint_delta_from_nominal_rad",
    "orientation_source",
    "planned_peak_joint_speed_rad_s",
    "production_control_authorized",
    "registered_margin_before_entry_m",
    "schema_version",
    "status",
    "stop_stage",
    "target_joint_deltas_from_nominal_rad",
    "target_order",
    "tcp_targets_world_m",
    "transition_peak_joint_speeds_rad_s",
    "translation_source",
    "trial_id",
    "truth_pose_feedback_used_for_target",
    "truth_xy_used_for_target",
    "z_source",
}
_PREINSERT_PROBE_KEYS = {
    "assembly_success_claimed",
    "body_contact_retention_gate",
    "checked_physics_steps",
    "continuation_global_steps",
    "engage_executed",
    "final_contacts",
    "final_target_tracking_error_rad",
    "final_tcp_axis_error_rad",
    "final_tcp_position_error_m",
    "finite_preinsert_gate",
    "home_return_executed",
    "insertion_executed",
    "loose_fixed_contact_records",
    "maximum_arm_tracking_error_rad",
    "maximum_joint_limit_violation_rad",
    "maximum_joint_speed_rad_s",
    "maximum_post_tare_absolute_delta_nm",
    "minimum_body_contact_finger_count",
    "object_pose_write_gate",
    "orientation_source",
    "outside_entry_gate",
    "passed",
    "post_hoc_actual_alignment",
    "prior_visual_pick_passed",
    "production_control_authorized",
    "same_world_capture_gate",
    "status",
    "tcp_target_gate",
    "torque_hard_stop_gate",
    "tracking_and_speed_gate",
    "translation_source",
    "truth_pose_feedback_used_for_target",
    "truth_xy_used_for_target",
    "twist_executed",
    "zero_preentry_contact_gate",
}
_ALIGNMENT_KEYS = {
    "axis_error_rad",
    "combined_entry_error_m",
    "commanded_preinsert_gap_m",
    "entry_gap_m",
    "gap_m",
    "lateral_error_m",
    "scope",
}
_CONTACT_KEYS = {
    "finger_body_group_records",
    "finger_loose_plug_records",
    "grip_material_records",
    "plug_table_records",
    "robot_loose_plug_records",
    "unexpected_robot_link_records",
}
_RUNTIME_SIDE_EFFECT_KEYS = {
    "endpoint_pose_writes_after_physics",
    "playing_after_restore",
    "playing_before_capture",
    "resource_cleanup_verified",
    "timeline_state_restored",
    "truth_or_error_gate_consulted",
    "world_reset_or_clear_calls",
}
_EXTERNAL_CONTACT_KEYS = {
    "fixed_endpoint",
    "fixture",
    "loose_plug_allowed",
    "loose_plug_preclosure",
    "loose_plug_unexpected_robot_link",
    "table",
}
_RGBD_KEYS = {
    "camera",
    "camera_frame_diagnostics",
    "camera_observation_present",
    "camera_projection",
    "capture_episode",
    "detector_kind",
    "endpoint_semantic_ids",
    "fixed_inheriting_gprim_count",
    "fixed_receptacle",
    "foundation_pose_present",
    "full_keyed_6d_vision_pose_claimed",
    "learned_detector_present",
    "loose_inheriting_gprim_count",
    "loose_plug",
    "masked_rgbd_xy_used_for_control",
    "object_pose_writes_after_start",
    "observed_semantic_ids",
    "output_directory",
    "passed",
    "position_estimator",
    "real_camera_present",
    "render_pipeline",
    "resource_cleanup",
    "rgbd_position_estimate_scope",
    "semantic_id_to_labels",
    "stage_prim_lifecycle",
    "timeline_pause",
    "timeline_state",
    "world_reset_or_clear_calls",
}


class EvidenceError(ValueError):
    """Raised when an input cannot authorize the limited evidence report."""


@dataclass(frozen=True)
class RunSpec:
    """One fixed approved run and its expected authored geometry."""

    run_id: str
    expected_trial_id: str
    expected_authored_loose_xy_m: tuple[float, float]
    expected_authored_fixed_xy_m: tuple[float, float]
    run_directory: str
    pick_config_path: str


@dataclass(frozen=True)
class EvidenceConfig:
    """Validated evidence policy; booleans cannot upgrade scope."""

    path: Path
    manifest_path: str
    preinsert_config_path: str
    policy: Mapping[str, Any]
    thresholds: Mapping[str, float]
    runs: tuple[RunSpec, ...]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise EvidenceError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise EvidenceError(f"{label} must be non-empty unpadded text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise EvidenceError(f"{label} must be >= {minimum}")
    return result


def _xy(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise EvidenceError(f"{label} must be a two-element list")
    return (
        _number(value[0], f"{label}[0]"),
        _number(value[1], f"{label}[1]"),
    )


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise EvidenceError(f"{label} must remain {expected}")


def _same(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=1.0e-12)


def _same_xy(first: Sequence[float], second: Sequence[float]) -> bool:
    return len(first) == len(second) and all(
        _same(float(left), float(right))
        for left, right in zip(first, second)
    )


def _repository_path(
    raw_path: Any,
    repository: Path,
    label: str,
    *,
    require_file: bool,
) -> Path:
    relative = Path(_text(raw_path, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise EvidenceError(f"{label} must be repository-relative")
    path = (repository / relative).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise EvidenceError(f"{label} escaped repository") from error
    if require_file and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _shown_path(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository))
    except ValueError as error:
        raise EvidenceError("source path escaped repository") from error


def _reject_json_constant(value: str) -> None:
    raise EvidenceError(f"JSON contains non-finite constant {value}")


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(document, label)


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise EvidenceError(f"{label} is not valid UTF-8 YAML") from error
    return _mapping(document, label)


def _binding_for(
    path: Path,
    repository: Path,
    schema_version: str,
) -> dict[str, Any]:
    return {
        "path": _shown_path(path, repository),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema_version": schema_version,
        "state": "BOUND",
    }


def _config_binding_for(
    path: Path,
    repository: Path,
) -> dict[str, Any]:
    return {
        "path": _shown_path(path, repository),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema_version": CONFIG_SCHEMA_VERSION,
    }


def _expected_source_paths(
    config: EvidenceConfig,
    run: RunSpec,
) -> dict[str, str]:
    directory = Path(run.run_directory)
    return {
        "report": str(directory / "report.json"),
        "cpu_plan": str(directory / "cpu_plan.json"),
        "preinsert_cpu_plan": str(
            directory / "preinsert_cpu_plan.json"
        ),
        "pick_config": run.pick_config_path,
        "preinsert_config": config.preinsert_config_path,
    }


def load_evidence_config(
    path: str | Path,
    repository: str | Path,
) -> EvidenceConfig:
    """Load the exact v1 config and reject any scope upgrade."""

    root = Path(repository).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    document = _load_yaml(config_path, "config")
    _exact_keys(document, _CONFIG_KEYS, "config")
    if document["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise EvidenceError("config.schema_version mismatch")
    if document["expected_run_count"] != EXPECTED_RUN_COUNT:
        raise EvidenceError("config.expected_run_count must remain 4")

    manifest_path = _text(
        document["source_manifest_path"],
        "config.source_manifest_path",
    )
    if manifest_path != DEFAULT_MANIFEST_PATH:
        raise EvidenceError("config.source_manifest_path mismatch")
    _repository_path(
        manifest_path,
        root,
        "config.source_manifest_path",
        require_file=False,
    )
    preinsert_config_path = _text(
        document["shared_preinsert_config_path"],
        "config.shared_preinsert_config_path",
    )
    _repository_path(
        preinsert_config_path,
        root,
        "config.shared_preinsert_config_path",
        require_file=True,
    )

    policy = _mapping(document["policy"], "config.policy")
    _exact_keys(policy, _POLICY_KEYS, "config.policy")
    expected_policy = {
        "evidence_scope": (
            "independent_four_position_visual_xy_"
            "pick_to_preinsert_outside_entry_only"
        ),
        "require_all_runs_passed": True,
        "require_distinct_authored_loose_xy": True,
        "require_same_authored_fixed_xy": True,
        "engage_gate_blocking_run_id": "plus10_xy",
        "statistics_only": True,
        "claim_engage": False,
        "claim_insertion": False,
        "claim_twist": False,
        "claim_home_return": False,
        "claim_full_6d": False,
        "claim_production_control": False,
        "claim_full_end_to_end": False,
    }
    for key, expected in expected_policy.items():
        if policy[key] != expected:
            raise EvidenceError(
                f"config.policy.{key} must remain {expected!r}"
            )

    thresholds = _mapping(document["thresholds"], "config.thresholds")
    _exact_keys(thresholds, _THRESHOLD_KEYS, "config.thresholds")
    expected_thresholds = {
        "strict_torque_upper_nm": 2.0,
        "entry_gap_m": 0.010,
        "commanded_preinsert_gap_m": 0.012,
        "registered_margin_before_entry_m": 0.002,
        "maximum_joint_speed_rad_s": 1.0,
        "maximum_arm_tracking_error_rad": 0.020,
        "maximum_joint_limit_violation_rad": 0.020,
        "maximum_tcp_position_error_m": 0.003,
        "maximum_tcp_axis_error_rad": 0.03490658503988659,
        "maximum_engage_lateral_error_m": 0.00020,
        "maximum_engage_axis_error_rad": 0.008726646259971648,
        "maximum_engage_combined_entry_error_m": 0.00025,
    }
    parsed_thresholds: dict[str, float] = {}
    for key, expected in expected_thresholds.items():
        actual = _number(thresholds[key], f"config.thresholds.{key}")
        if not _same(actual, expected):
            raise EvidenceError(
                f"config.thresholds.{key} must remain {expected!r}"
            )
        parsed_thresholds[key] = actual

    raw_runs = document["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != EXPECTED_RUN_COUNT:
        raise EvidenceError("config.runs must contain exactly four runs")
    runs: list[RunSpec] = []
    for index, value in enumerate(raw_runs):
        label = f"config.runs[{index}]"
        raw = _mapping(value, label)
        _exact_keys(raw, _RUN_KEYS, label)
        run_id = _text(raw["run_id"], f"{label}.run_id")
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise EvidenceError(f"{label}.run_id is unsafe")
        run_directory = _text(
            raw["run_directory"], f"{label}.run_directory"
        )
        _repository_path(
            run_directory,
            root,
            f"{label}.run_directory",
            require_file=False,
        )
        pick_config_path = _text(
            raw["pick_config_path"], f"{label}.pick_config_path"
        )
        _repository_path(
            pick_config_path,
            root,
            f"{label}.pick_config_path",
            require_file=True,
        )
        runs.append(
            RunSpec(
                run_id=run_id,
                expected_trial_id=_text(
                    raw["expected_trial_id"],
                    f"{label}.expected_trial_id",
                ),
                expected_authored_loose_xy_m=_xy(
                    raw["expected_authored_loose_xy_m"],
                    f"{label}.expected_authored_loose_xy_m",
                ),
                expected_authored_fixed_xy_m=_xy(
                    raw["expected_authored_fixed_xy_m"],
                    f"{label}.expected_authored_fixed_xy_m",
                ),
                run_directory=run_directory,
                pick_config_path=pick_config_path,
            )
        )
    if len({run.run_id for run in runs}) != EXPECTED_RUN_COUNT:
        raise EvidenceError("config run_id values must be unique")
    if len({run.expected_trial_id for run in runs}) != EXPECTED_RUN_COUNT:
        raise EvidenceError("config trial IDs must be unique")
    if len(
        {run.expected_authored_loose_xy_m for run in runs}
    ) != EXPECTED_RUN_COUNT:
        raise EvidenceError(
            "config authored loose XY positions must be distinct"
        )
    if len(
        {run.expected_authored_fixed_xy_m for run in runs}
    ) != 1:
        raise EvidenceError(
            "config authored fixed XY positions must be identical"
        )

    return EvidenceConfig(
        path=config_path,
        manifest_path=manifest_path,
        preinsert_config_path=preinsert_config_path,
        policy=dict(policy),
        thresholds=parsed_thresholds,
        runs=tuple(runs),
    )


def build_source_manifest(
    config_path: str | Path,
    repository: str | Path,
) -> dict[str, Any]:
    """Build a manifest; absent sources are explicit and non-authorizing."""

    root = Path(repository).resolve()
    config = load_evidence_config(config_path, root)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    complete_run_count = 0
    for run in config.runs:
        sources: dict[str, dict[str, Any]] = {}
        run_complete = True
        for name, shown in _expected_source_paths(config, run).items():
            path = _repository_path(
                shown,
                root,
                f"{run.run_id}.{name}.path",
                require_file=False,
            )
            if path.is_file():
                sources[name] = _binding_for(
                    path,
                    root,
                    _SOURCE_SCHEMAS[name],
                )
            else:
                run_complete = False
                missing.append(shown)
                sources[name] = {
                    "path": shown,
                    "sha256": None,
                    "size_bytes": None,
                    "schema_version": _SOURCE_SCHEMAS[name],
                    "state": "MISSING",
                }
        if run_complete:
            complete_run_count += 1
        records.append({"run_id": run.run_id, "sources": sources})
    status = (
        MANIFEST_COMPLETE
        if complete_run_count == EXPECTED_RUN_COUNT
        else MANIFEST_INCOMPLETE
    )
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "status": status,
        "expected_run_count": EXPECTED_RUN_COUNT,
        "complete_run_count": complete_run_count,
        "config": _config_binding_for(config.path, root),
        "runs": records,
        "missing_paths": sorted(set(missing)),
        "claims_authorized": False,
    }


def write_source_manifest(
    config_path: str | Path,
    repository: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically refresh the source manifest from the fixed config paths."""

    root = Path(repository).resolve()
    config = load_evidence_config(config_path, root)
    expected_path = _repository_path(
        config.manifest_path,
        root,
        "config.source_manifest_path",
        require_file=False,
    )
    output = expected_path
    if manifest_path is not None:
        candidate = Path(manifest_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        output = candidate.resolve()
        if output != expected_path:
            raise EvidenceError("manifest output path differs from config")
    manifest = build_source_manifest(config.path, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            manifest,
            stream,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    temporary.replace(output)
    return manifest


def _verify_config_binding(
    value: Any,
    config: EvidenceConfig,
    repository: Path,
) -> None:
    binding = _mapping(value, "manifest.config")
    _exact_keys(binding, _CONFIG_BINDING_KEYS, "manifest.config")
    if binding["path"] != _shown_path(config.path, repository):
        raise EvidenceError("manifest.config.path mismatch")
    if binding["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise EvidenceError("manifest.config.schema_version mismatch")
    size = _integer(
        binding["size_bytes"], "manifest.config.size_bytes", minimum=1
    )
    if size != config.path.stat().st_size:
        raise EvidenceError("manifest.config byte-size mismatch")
    digest = _text(binding["sha256"], "manifest.config.sha256")
    if digest != sha256_file(config.path):
        raise EvidenceError("manifest.config SHA-256 mismatch")


def _verify_source_binding(
    value: Any,
    repository: Path,
    expected_path: str,
    expected_schema: str,
    label: str,
) -> Path:
    binding = _mapping(value, label)
    _exact_keys(binding, _BINDING_KEYS, label)
    if binding["state"] != "BOUND":
        raise EvidenceError(f"{label}.state must be BOUND")
    if binding["path"] != expected_path:
        raise EvidenceError(f"{label}.path mismatch")
    if binding["schema_version"] != expected_schema:
        raise EvidenceError(f"{label}.schema_version mismatch")
    size = _integer(binding["size_bytes"], f"{label}.size_bytes", minimum=1)
    digest = _text(binding["sha256"], f"{label}.sha256")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise EvidenceError(f"{label}.sha256 must be lowercase SHA-256")
    path = _repository_path(
        expected_path,
        repository,
        f"{label}.path",
        require_file=True,
    )
    if path.stat().st_size != size:
        raise EvidenceError(f"{label} byte-size mismatch")
    if sha256_file(path) != digest:
        raise EvidenceError(f"{label} SHA-256 mismatch")
    return path


def load_complete_source_manifest(
    config: EvidenceConfig,
    repository: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[Mapping[str, Any], list[dict[str, Path]]]:
    """Verify a complete manifest and return its bound source paths."""

    root = Path(repository).resolve()
    expected_manifest = _repository_path(
        config.manifest_path,
        root,
        "config.source_manifest_path",
        require_file=True,
    )
    selected = expected_manifest
    if manifest_path is not None:
        candidate = Path(manifest_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        selected = candidate.resolve()
        if selected != expected_manifest:
            raise EvidenceError("manifest path differs from config")
    manifest = _load_json(selected, "manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "status",
            "expected_run_count",
            "complete_run_count",
            "config",
            "runs",
            "missing_paths",
            "claims_authorized",
        },
        "manifest",
    )
    if manifest["schema_version"] != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise EvidenceError("manifest.schema_version mismatch")
    if manifest["status"] != MANIFEST_COMPLETE:
        raise EvidenceError(
            "source manifest is incomplete; refresh only after all four "
            "approved runs exist"
        )
    if manifest["expected_run_count"] != EXPECTED_RUN_COUNT:
        raise EvidenceError("manifest.expected_run_count must remain 4")
    if manifest["complete_run_count"] != EXPECTED_RUN_COUNT:
        raise EvidenceError("manifest.complete_run_count must be 4")
    if manifest["missing_paths"] != []:
        raise EvidenceError("complete manifest must have no missing paths")
    _require_bool(
        manifest["claims_authorized"],
        False,
        "manifest.claims_authorized",
    )
    _verify_config_binding(manifest["config"], config, root)

    raw_runs = manifest["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != EXPECTED_RUN_COUNT:
        raise EvidenceError("manifest.runs must contain exactly four runs")
    paths_by_run: list[dict[str, Path]] = []
    for index, (raw_record, spec) in enumerate(zip(raw_runs, config.runs)):
        label = f"manifest.runs[{index}]"
        record = _mapping(raw_record, label)
        _exact_keys(record, {"run_id", "sources"}, label)
        if record["run_id"] != spec.run_id:
            raise EvidenceError(f"{label}.run_id mismatch")
        sources = _mapping(record["sources"], f"{label}.sources")
        _exact_keys(sources, set(_SOURCE_NAMES), f"{label}.sources")
        expected_paths = _expected_source_paths(config, spec)
        verified: dict[str, Path] = {}
        for name in _SOURCE_NAMES:
            verified[name] = _verify_source_binding(
                sources[name],
                root,
                expected_paths[name],
                _SOURCE_SCHEMAS[name],
                f"{label}.sources.{name}",
            )
        paths_by_run.append(verified)
    return manifest, paths_by_run


def _internal_path(value: Any, label: str) -> Path:
    try:
        return Path(_text(value, label)).resolve()
    except OSError as error:
        raise EvidenceError(f"{label} cannot be resolved") from error


def _validate_pick_config(
    path: Path,
    report: Mapping[str, Any],
    spec: RunSpec,
) -> None:
    document = _load_yaml(path, f"{spec.run_id}.pick_config")
    _exact_keys(
        document,
        {
            "schema_version",
            "enabled_by_default",
            "status",
            "inputs",
            "trial",
            "rgbd_observation",
            "local_fixed_q7_ik",
            "execution",
            "output",
            "boundaries",
        },
        f"{spec.run_id}.pick_config",
    )
    if document["schema_version"] != PICK_CONFIG_SCHEMA_VERSION:
        raise EvidenceError(f"{spec.run_id}.pick_config schema mismatch")
    _require_bool(
        document["enabled_by_default"],
        False,
        f"{spec.run_id}.pick_config.enabled_by_default",
    )
    if document["status"] != "prepared_independent_visual_xy_pick_probe":
        raise EvidenceError(f"{spec.run_id}.pick_config status mismatch")
    trial = _mapping(document["trial"], f"{spec.run_id}.pick_config.trial")
    _exact_keys(
        trial,
        {
            "trial_id",
            "author_before_physics",
            "loose_plug_xy_m",
            "fixed_receptacle_xy_m",
            "orientation_source",
            "loose_yaw_rad",
            "fixed_yaw_rad",
        },
        f"{spec.run_id}.pick_config.trial",
    )
    if trial["trial_id"] != spec.expected_trial_id:
        raise EvidenceError(f"{spec.run_id}.pick_config trial_id mismatch")
    _require_bool(
        trial["author_before_physics"],
        True,
        f"{spec.run_id}.pick_config.trial.author_before_physics",
    )
    if not _same_xy(
        _xy(
            trial["loose_plug_xy_m"],
            f"{spec.run_id}.pick_config.trial.loose_plug_xy_m",
        ),
        spec.expected_authored_loose_xy_m,
    ):
        raise EvidenceError(f"{spec.run_id}.pick_config loose XY mismatch")
    if not _same_xy(
        _xy(
            trial["fixed_receptacle_xy_m"],
            f"{spec.run_id}.pick_config.trial.fixed_receptacle_xy_m",
        ),
        spec.expected_authored_fixed_xy_m,
    ):
        raise EvidenceError(f"{spec.run_id}.pick_config fixed XY mismatch")
    if trial["orientation_source"] != "registered_nominal":
        raise EvidenceError(f"{spec.run_id}.pick_config orientation mismatch")
    for name in ("loose_yaw_rad", "fixed_yaw_rad"):
        if _number(
            trial[name], f"{spec.run_id}.pick_config.trial.{name}"
        ) != 0.0:
            raise EvidenceError(
                f"{spec.run_id}.pick_config {name} must be zero"
            )

    execution = _mapping(
        document["execution"], f"{spec.run_id}.pick_config.execution"
    )
    if _number(
        execution.get("hard_stop_nm"),
        f"{spec.run_id}.pick_config.execution.hard_stop_nm",
    ) != 2.0:
        raise EvidenceError(f"{spec.run_id}.pick_config hard stop changed")
    boundaries = _mapping(
        document["boundaries"], f"{spec.run_id}.pick_config.boundaries"
    )
    for key in (
        "existing_e2e_modified",
        "frozen_baseline_modified",
        "truth_xy_used_for_target",
        "full_6d_claimed",
        "production_control_authorized",
        "collision_planned",
        "object_pose_writes_after_physics_allowed",
        "real_assembly_success_claimed",
    ):
        _require_bool(
            boundaries.get(key),
            False,
            f"{spec.run_id}.pick_config.boundaries.{key}",
        )
    if sha256_file(path) != report.get("config_sha256"):
        raise EvidenceError(
            f"{spec.run_id}.pick_config hash differs from report"
        )
    if _internal_path(
        report.get("config_path"), f"{spec.run_id}.report.config_path"
    ) != path:
        raise EvidenceError(
            f"{spec.run_id}.pick_config path differs from report"
        )


def _validate_preinsert_config(
    path: Path,
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
    run_id: str,
) -> None:
    document = _load_yaml(path, f"{run_id}.preinsert_config")
    _exact_keys(
        document,
        {
            "schema_version",
            "enabled_by_default",
            "status",
            "inputs",
            "planning",
            "local_fixed_q7_ik",
            "axial_scope",
            "runtime_failure_gates",
            "output",
            "boundaries",
        },
        f"{run_id}.preinsert_config",
    )
    if document["schema_version"] != PREINSERT_CONFIG_SCHEMA_VERSION:
        raise EvidenceError(f"{run_id}.preinsert_config schema mismatch")
    _require_bool(
        document["enabled_by_default"],
        False,
        f"{run_id}.preinsert_config.enabled_by_default",
    )
    if document["status"] != "prepared_cpu_plan_not_physx_executed":
        raise EvidenceError(f"{run_id}.preinsert_config status mismatch")

    axial = _mapping(
        document["axial_scope"], f"{run_id}.preinsert_config.axial_scope"
    )
    _exact_keys(
        axial,
        {
            "preinsert_gap_m",
            "entry_gap_m",
            "registered_margin_before_entry_m",
            "engage_target_planned",
            "insertion_target_planned",
        },
        f"{run_id}.preinsert_config.axial_scope",
    )
    for key, threshold_key in (
        ("preinsert_gap_m", "commanded_preinsert_gap_m"),
        ("entry_gap_m", "entry_gap_m"),
        (
            "registered_margin_before_entry_m",
            "registered_margin_before_entry_m",
        ),
    ):
        if not _same(
            _number(
                axial[key],
                f"{run_id}.preinsert_config.axial_scope.{key}",
            ),
            thresholds[threshold_key],
        ):
            raise EvidenceError(f"{run_id}.preinsert_config {key} mismatch")
    for key in ("engage_target_planned", "insertion_target_planned"):
        _require_bool(
            axial[key],
            False,
            f"{run_id}.preinsert_config.axial_scope.{key}",
        )

    gates = _mapping(
        document["runtime_failure_gates"],
        f"{run_id}.preinsert_config.runtime_failure_gates",
    )
    _exact_keys(
        gates,
        {
            "require_prior_visual_pick_pass",
            "require_same_world_and_capture_id",
            "require_no_object_pose_writes_after_start",
            "require_no_robot_table_fixture_or_fixed_contact",
            "require_no_loose_fixed_contact_before_entry",
            "require_all_fingers_retain_body_contact",
            "require_finger_torque_below_hard_stop",
            "actual_body_fixed_alignment_truth_evaluation_only",
            "actual_alignment_must_not_change_targets",
        },
        f"{run_id}.preinsert_config.runtime_failure_gates",
    )
    for key, value in gates.items():
        _require_bool(
            value,
            True,
            f"{run_id}.preinsert_config.runtime_failure_gates.{key}",
        )

    boundaries = _mapping(
        document["boundaries"], f"{run_id}.preinsert_config.boundaries"
    )
    for key in (
        "existing_e2e_modified",
        "existing_visual_pick_default_modified",
        "frozen_baseline_modified",
        "truth_xy_used_for_target",
        "truth_pose_feedback_used_for_target",
        "orientation_estimated_from_rgbd",
        "full_6d_claimed",
        "engage_executed",
        "insertion_executed",
        "twist_executed",
        "home_return_executed",
        "collision_planned",
        "gpu_or_physx_validated",
        "production_control_authorized",
        "assembly_success_claimed",
    ):
        _require_bool(
            boundaries.get(key),
            False,
            f"{run_id}.preinsert_config.boundaries.{key}",
        )
    if sha256_file(path) != report.get("preinsert_config_sha256"):
        raise EvidenceError(
            f"{run_id}.preinsert_config hash differs from report"
        )
    if _internal_path(
        report.get("preinsert_config_path"),
        f"{run_id}.report.preinsert_config_path",
    ) != path:
        raise EvidenceError(
            f"{run_id}.preinsert_config path differs from report"
        )


def _vector_xy(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise EvidenceError(f"{label} must be a three-element list")
    for index, item in enumerate(value):
        _number(item, f"{label}[{index}]")
    return float(value[0]), float(value[1])


def _validate_pick_plan(
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    path: Path,
    spec: RunSpec,
) -> dict[str, tuple[float, float]]:
    label = f"{spec.run_id}.cpu_plan"
    _exact_keys(plan, _PICK_PLAN_KEYS, label)
    if plan["schema_version"] != PICK_PLAN_SCHEMA_VERSION:
        raise EvidenceError(f"{label}.schema_version mismatch")
    if plan != report.get("cpu_plan"):
        raise EvidenceError(f"{label} differs from report.cpu_plan")
    if _internal_path(
        report.get("cpu_plan_path"), f"{spec.run_id}.report.cpu_plan_path"
    ) != path:
        raise EvidenceError(f"{label} path differs from report")
    if plan["status"] != "CPU_PLAN_READY_FOR_INDEPENDENT_ISAAC_PROBE":
        raise EvidenceError(f"{label}.status mismatch")
    if plan["trial_id"] != spec.expected_trial_id:
        raise EvidenceError(f"{label}.trial_id mismatch")
    for key in (
        "collision_planned",
        "full_6d",
        "gpu_or_physx_validated",
        "production_control_authorized",
        "uses_truth_xy_for_target",
    ):
        _require_bool(plan[key], False, f"{label}.{key}")
    if plan["orientation_source"] != "registered_nominal":
        raise EvidenceError(f"{label}.orientation_source mismatch")

    capture_id = _text(plan["capture_id"], f"{label}.capture_id")
    adapter = _mapping(plan["adapter"], f"{label}.adapter")
    _exact_keys(adapter, _ADAPTER_KEYS, f"{label}.adapter")
    if adapter["capture_id"] != capture_id:
        raise EvidenceError(f"{label}.adapter capture_id mismatch")
    expected_adapter_bools = {
        "eligible_for_independent_probe": True,
        "downstream_ik_required": True,
        "preserves_nominal_target_z": True,
        "collision_free_ik_verified": False,
        "full_6d": False,
        "pose_provider_control_authorized": False,
        "production_control_authorized": False,
        "uses_truth_orientation": False,
        "yaw_observed": False,
    }
    for key, expected in expected_adapter_bools.items():
        _require_bool(adapter[key], expected, f"{label}.adapter.{key}")
    if adapter["orientation_source"] != "registered_nominal":
        raise EvidenceError(f"{label}.adapter orientation mismatch")
    if adapter["world_target_frame"] != "world":
        raise EvidenceError(f"{label}.adapter target frame mismatch")
    if adapter["rejection_reasons"] != []:
        raise EvidenceError(f"{label}.adapter has rejection reasons")
    translation_source = _text(
        adapter["translation_source"], f"{label}.adapter.translation_source"
    )
    if not translation_source.startswith("vision_"):
        raise EvidenceError(f"{label}.adapter translation is not visual")

    targets = _mapping(plan["tcp_targets_world_m"], f"{label}.tcp_targets")
    _exact_keys(
        targets,
        {"pregrasp_tcp", "grasp_tcp", "closure_clearance_tcp"},
        f"{label}.tcp_targets",
    )
    loose_target = _vector_xy(targets["grasp_tcp"], f"{label}.grasp_tcp")
    for key in ("pregrasp_tcp", "closure_clearance_tcp"):
        if not _same_xy(
            _vector_xy(targets[key], f"{label}.{key}"), loose_target
        ):
            raise EvidenceError(f"{label}.{key} XY differs from grasp")

    world_targets = _mapping(
        adapter["world_targets"], f"{label}.adapter.world_targets"
    )
    _exact_keys(
        world_targets,
        {
            "axis_high_tcp",
            "closure_clearance_tcp",
            "engage_tcp",
            "grasp_tcp",
            "pregrasp_tcp",
            "preinsert_tcp",
            "transport_safe_tcp",
        },
        f"{label}.adapter.world_targets",
    )
    fixed_target = _vector_xy(
        world_targets["preinsert_tcp"],
        f"{label}.adapter.preinsert_tcp",
    )
    for key in ("axis_high_tcp", "engage_tcp", "transport_safe_tcp"):
        if not _same_xy(
            _vector_xy(world_targets[key], f"{label}.adapter.{key}"),
            fixed_target,
        ):
            raise EvidenceError(f"{label}.adapter.{key} XY mismatch")
    return {
        "loose_target": loose_target,
        "fixed_target": fixed_target,
        "capture_id": capture_id,
        "fixed_translation": _xy(
            adapter["fixed_translation_xy_m"],
            f"{label}.adapter.fixed_translation_xy_m",
        ),
    }


def _validate_preinsert_plan(
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    path: Path,
    spec: RunSpec,
    pick: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> None:
    label = f"{spec.run_id}.preinsert_cpu_plan"
    _exact_keys(plan, _PREINSERT_PLAN_KEYS, label)
    if plan["schema_version"] != PREINSERT_PLAN_SCHEMA_VERSION:
        raise EvidenceError(f"{label}.schema_version mismatch")
    if plan != report.get("preinsert_cpu_plan"):
        raise EvidenceError(f"{label} differs from report.preinsert_cpu_plan")
    if _internal_path(
        report.get("preinsert_cpu_plan_path"),
        f"{spec.run_id}.report.preinsert_cpu_plan_path",
    ) != path:
        raise EvidenceError(f"{label} path differs from report")
    if plan["status"] != "CPU_PLAN_READY_FOR_VISUAL_XY_PREINSERT_PROBE":
        raise EvidenceError(f"{label}.status mismatch")
    if plan["trial_id"] != spec.expected_trial_id:
        raise EvidenceError(f"{label}.trial_id mismatch")
    if plan["capture_id"] != pick["capture_id"]:
        raise EvidenceError(f"{label}.capture_id differs from visual pick")
    if plan["target_order"] != [
        "transport_safe",
        "axis_high",
        "preinsert",
    ]:
        raise EvidenceError(f"{label}.target_order mismatch")
    if plan["stop_stage"] != "PREINSERT":
        raise EvidenceError(f"{label}.stop_stage mismatch")
    expected_false = (
        "assembly_success_claimed",
        "collision_planned",
        "engage_executed",
        "full_6d",
        "gpu_or_physx_validated",
        "production_control_authorized",
        "truth_pose_feedback_used_for_target",
        "truth_xy_used_for_target",
    )
    for key in expected_false:
        _require_bool(plan[key], False, f"{label}.{key}")
    if plan["translation_source"] != "visual_fixed_receptacle_xy":
        raise EvidenceError(f"{label}.translation_source mismatch")
    if plan["orientation_source"] != "registered_nominal_fk":
        raise EvidenceError(f"{label}.orientation_source mismatch")
    if plan["z_source"] != "registered_nominal":
        raise EvidenceError(f"{label}.z_source mismatch")
    if not _same_xy(
        _xy(plan["fixed_translation_xy_m"], f"{label}.fixed_translation"),
        pick["fixed_translation"],
    ):
        raise EvidenceError(f"{label}.fixed_translation differs from adapter")
    margin = _number(
        plan["registered_margin_before_entry_m"],
        f"{label}.registered_margin_before_entry_m",
    )
    if not _same(margin, thresholds["registered_margin_before_entry_m"]):
        raise EvidenceError(f"{label}.registered margin mismatch")
    planned_speed = _number(
        plan["planned_peak_joint_speed_rad_s"],
        f"{label}.planned_peak_joint_speed_rad_s",
        minimum=0.0,
    )
    if planned_speed > thresholds["maximum_joint_speed_rad_s"]:
        raise EvidenceError(f"{label}.planned speed exceeds limit")
    targets = _mapping(plan["tcp_targets_world_m"], f"{label}.tcp_targets")
    _exact_keys(
        targets,
        {"transport_safe", "axis_high", "preinsert"},
        f"{label}.tcp_targets",
    )
    for key in ("transport_safe", "axis_high", "preinsert"):
        if not _same_xy(
            _vector_xy(targets[key], f"{label}.{key}"),
            pick["fixed_target"],
        ):
            raise EvidenceError(f"{label}.{key} XY is not visual fixed XY")


def _validate_contacts(value: Any, label: str) -> dict[str, Any]:
    contacts = _mapping(value, label)
    _exact_keys(contacts, _CONTACT_KEYS, label)
    fingers = _mapping(
        contacts["finger_body_group_records"],
        f"{label}.finger_body_group_records",
    )
    _exact_keys(
        fingers,
        {"f1", "f2", "f3"},
        f"{label}.finger_body_group_records",
    )
    body_counts: dict[str, int] = {}
    nut_counts: dict[str, int] = {}
    for finger in ("f1", "f2", "f3"):
        record = _mapping(fingers[finger], f"{label}.{finger}")
        _exact_keys(record, {"body", "nut"}, f"{label}.{finger}")
        body_counts[finger] = _integer(
            record["body"], f"{label}.{finger}.body", minimum=1
        )
        nut_counts[finger] = _integer(
            record["nut"], f"{label}.{finger}.nut", minimum=0
        )
    for key in (
        "finger_loose_plug_records",
        "grip_material_records",
        "robot_loose_plug_records",
    ):
        _integer(contacts[key], f"{label}.{key}", minimum=1)
    for key in ("plug_table_records", "unexpected_robot_link_records"):
        if _integer(contacts[key], f"{label}.{key}") != 0:
            raise EvidenceError(f"{label}.{key} must be zero")
    return {
        "body_counts": body_counts,
        "nut_counts": nut_counts,
        "finger_loose_plug_records": contacts[
            "finger_loose_plug_records"
        ],
        "grip_material_records": contacts["grip_material_records"],
        "robot_loose_plug_records": contacts[
            "robot_loose_plug_records"
        ],
    }


def _validate_report(
    report: Mapping[str, Any],
    paths: Mapping[str, Path],
    spec: RunSpec,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Validate exact report semantics and return aggregate-ready metrics."""

    label = f"{spec.run_id}.report"
    _exact_keys(report, _REPORT_KEYS, label)
    if report["schema_version"] != SOURCE_REPORT_SCHEMA_VERSION:
        raise EvidenceError(f"{label}.schema_version mismatch")
    if report["trial_id"] != spec.expected_trial_id:
        raise EvidenceError(f"{label}.trial_id mismatch")
    if _internal_path(report["report_path"], f"{label}.report_path") != paths[
        "report"
    ]:
        raise EvidenceError(f"{label}.report_path mismatch")
    if _internal_path(
        report["output_directory"], f"{label}.output_directory"
    ) != paths["report"].parent:
        raise EvidenceError(f"{label}.output_directory mismatch")

    for key in (
        "passed",
        "explicit_opt_in",
        "finite_final",
        "finite_throughout",
        "settled_on_table",
        "contact_gate",
        "torque_gate",
        "truth_xy_evaluation_gate",
        "unsupported_gate",
        "zero_forbidden_contacts",
        "preinsert_probe_requested",
    ):
        _require_bool(report[key], True, f"{label}.{key}")
    for key in (
        "collision_planned",
        "engage_executed",
        "full_6d",
        "gui",
        "home_return_executed",
        "insertion_executed",
        "production_control_authorized",
        "truth_xy_used_for_target",
        "twist_executed",
    ):
        _require_bool(report[key], False, f"{label}.{key}")
    if report["orientation_source"] != "registered_nominal":
        raise EvidenceError(f"{label}.orientation_source mismatch")

    authored = _mapping(
        report["authored_before_physics"],
        f"{label}.authored_before_physics",
    )
    _exact_keys(
        authored,
        {
            "fixed_receptacle_xy_m",
            "fixed_yaw_rad",
            "loose_plug_xy_m",
            "loose_yaw_rad",
        },
        f"{label}.authored_before_physics",
    )
    if not _same_xy(
        _xy(authored["loose_plug_xy_m"], f"{label}.authored.loose_xy"),
        spec.expected_authored_loose_xy_m,
    ):
        raise EvidenceError(f"{label} authored loose XY mismatch")
    if not _same_xy(
        _xy(authored["fixed_receptacle_xy_m"], f"{label}.authored.fixed_xy"),
        spec.expected_authored_fixed_xy_m,
    ):
        raise EvidenceError(f"{label} authored fixed XY mismatch")
    for key in ("loose_yaw_rad", "fixed_yaw_rad"):
        if _number(authored[key], f"{label}.authored.{key}") != 0.0:
            raise EvidenceError(f"{label}.authored.{key} must be zero")

    _validate_pick_config(paths["pick_config"], report, spec)
    _validate_preinsert_config(
        paths["preinsert_config"], report, thresholds, spec.run_id
    )
    pick_plan = _load_json(paths["cpu_plan"], f"{spec.run_id}.cpu_plan")
    pick = _validate_pick_plan(
        pick_plan, report, paths["cpu_plan"], spec
    )
    preinsert_plan = _load_json(
        paths["preinsert_cpu_plan"], f"{spec.run_id}.preinsert_cpu_plan"
    )
    _validate_preinsert_plan(
        preinsert_plan,
        report,
        paths["preinsert_cpu_plan"],
        spec,
        pick,
        thresholds,
    )

    pose = _mapping(report["pose_provider"], f"{label}.pose_provider")
    _exact_keys(
        pose,
        {
            "control_authorized",
            "diagnostics",
            "full_6d",
            "provider_id",
            "purpose",
            "uses_truth_orientation",
            "uses_truth_position",
        },
        f"{label}.pose_provider",
    )
    for key in (
        "control_authorized",
        "full_6d",
        "uses_truth_orientation",
        "uses_truth_position",
    ):
        _require_bool(pose[key], False, f"{label}.pose_provider.{key}")
    if pose["purpose"] != "preflight":
        raise EvidenceError(f"{label}.pose_provider.purpose mismatch")
    diagnostics = _mapping(
        pose["diagnostics"], f"{label}.pose_provider.diagnostics"
    )
    _exact_keys(
        diagnostics,
        {"endpoints", "estimator", "schema_version"},
        f"{label}.pose_provider.diagnostics",
    )
    endpoints = _mapping(
        diagnostics["endpoints"],
        f"{label}.pose_provider.diagnostics.endpoints",
    )
    _exact_keys(
        endpoints,
        {"loose_plug", "fixed_receptacle"},
        f"{label}.pose_provider.diagnostics.endpoints",
    )
    observed_xy: dict[str, tuple[float, float]] = {}
    for endpoint in ("loose_plug", "fixed_receptacle"):
        endpoint_record = _mapping(
            endpoints[endpoint], f"{label}.pose_provider.{endpoint}"
        )
        _exact_keys(
            endpoint_record,
            {
                "confidence",
                "estimated_world_xy_m",
                "timestamp_s",
                "xy_error_bound_m",
            },
            f"{label}.pose_provider.{endpoint}",
        )
        observed_xy[endpoint] = _xy(
            endpoint_record["estimated_world_xy_m"],
            f"{label}.pose_provider.{endpoint}.estimated_world_xy_m",
        )
    if not _same_xy(observed_xy["loose_plug"], pick["loose_target"]):
        raise EvidenceError(f"{label} visual loose XY was not used by pick")
    if not _same_xy(observed_xy["fixed_receptacle"], pick["fixed_target"]):
        raise EvidenceError(
            f"{label} visual fixed XY was not used by preinsert"
        )

    rgbd = _mapping(report["rgbd_capture"], f"{label}.rgbd_capture")
    _exact_keys(rgbd, _RGBD_KEYS, f"{label}.rgbd_capture")
    for key, expected in (
        ("passed", True),
        ("camera_observation_present", True),
        ("full_keyed_6d_vision_pose_claimed", False),
        ("masked_rgbd_xy_used_for_control", False),
    ):
        _require_bool(rgbd[key], expected, f"{label}.rgbd_capture.{key}")
    if rgbd["capture_episode"] != "caller_world_same_episode":
        raise EvidenceError(f"{label}.rgbd_capture.capture_episode mismatch")
    if _integer(
        rgbd["object_pose_writes_after_start"],
        f"{label}.rgbd_capture.object_pose_writes_after_start",
    ) != 0:
        raise EvidenceError(f"{label}.rgbd_capture has object pose writes")
    if _integer(
        rgbd["world_reset_or_clear_calls"],
        f"{label}.rgbd_capture.world_reset_or_clear_calls",
    ) != 0:
        raise EvidenceError(f"{label}.rgbd_capture reset or cleared world")
    estimator = _mapping(
        rgbd["position_estimator"],
        f"{label}.rgbd_capture.position_estimator",
    )
    _require_bool(
        estimator.get("uses_registered_truth_xy"),
        False,
        f"{label}.rgbd_capture.position_estimator.uses_registered_truth_xy",
    )
    for endpoint in ("loose_plug", "fixed_receptacle"):
        endpoint_result = _mapping(
            rgbd[endpoint], f"{label}.rgbd_capture.{endpoint}"
        )
        _require_bool(
            endpoint_result.get("passed"),
            True,
            f"{label}.rgbd_capture.{endpoint}.passed",
        )

    truth = _mapping(
        report["truth_evaluation"], f"{label}.truth_evaluation"
    )
    _exact_keys(
        truth,
        {
            "fixed_xy_error_m",
            "loose_xy_error_m",
            "scope",
            "truth_xy_used_for_target",
        },
        f"{label}.truth_evaluation",
    )
    if truth["scope"] != "post_hoc_truth_evaluation_not_target_input":
        raise EvidenceError(f"{label}.truth_evaluation.scope mismatch")
    _require_bool(
        truth["truth_xy_used_for_target"],
        False,
        f"{label}.truth_evaluation.truth_xy_used_for_target",
    )
    loose_xy_error = _number(
        truth["loose_xy_error_m"],
        f"{label}.truth_evaluation.loose_xy_error_m",
        minimum=0.0,
    )
    fixed_xy_error = _number(
        truth["fixed_xy_error_m"],
        f"{label}.truth_evaluation.fixed_xy_error_m",
        minimum=0.0,
    )

    authoring = _mapping(
        report["d38999_authoring"], f"{label}.d38999_authoring"
    )
    _exact_keys(
        authoring,
        {
            "asset_sha256",
            "body_prim_path",
            "fixed_receptacle_prim_path",
            "fixture_prim_path",
            "joint_prim_path",
            "nut_prim_path",
            "object_pose_writes_after_start",
            "table_prim_path",
        },
        f"{label}.d38999_authoring",
    )
    if _integer(
        authoring["object_pose_writes_after_start"],
        f"{label}.d38999_authoring.object_pose_writes_after_start",
    ) != 0:
        raise EvidenceError(f"{label}.d38999_authoring has pose writes")
    asset_sha256 = _text(
        authoring["asset_sha256"], f"{label}.d38999_authoring.asset_sha256"
    )
    if len(asset_sha256) != 64:
        raise EvidenceError(f"{label}.d38999_authoring asset SHA invalid")

    side_effects = _mapping(
        report["runtime_side_effects"], f"{label}.runtime_side_effects"
    )
    _exact_keys(
        side_effects,
        _RUNTIME_SIDE_EFFECT_KEYS,
        f"{label}.runtime_side_effects",
    )
    for key in (
        "playing_after_restore",
        "playing_before_capture",
        "resource_cleanup_verified",
        "timeline_state_restored",
    ):
        _require_bool(side_effects[key], True, f"{label}.side_effects.{key}")
    _require_bool(
        side_effects["truth_or_error_gate_consulted"],
        False,
        f"{label}.side_effects.truth_or_error_gate_consulted",
    )
    for key in (
        "endpoint_pose_writes_after_physics",
        "world_reset_or_clear_calls",
    ):
        if _integer(side_effects[key], f"{label}.side_effects.{key}") != 0:
            raise EvidenceError(f"{label}.side_effects.{key} must be zero")
    if _integer(
        report["object_pose_writes_after_physics"],
        f"{label}.object_pose_writes_after_physics",
    ) != 0:
        raise EvidenceError(
            f"{label}.object_pose_writes_after_physics is nonzero"
        )

    external = _mapping(
        report["external_contact_records"], f"{label}.external_contacts"
    )
    _exact_keys(external, _EXTERNAL_CONTACT_KEYS, f"{label}.external_contacts")
    for key in (
        "fixed_endpoint",
        "fixture",
        "loose_plug_preclosure",
        "loose_plug_unexpected_robot_link",
        "table",
    ):
        if _integer(external[key], f"{label}.external_contacts.{key}") != 0:
            raise EvidenceError(
                f"{label}.external_contacts.{key} must be zero"
            )
    _integer(
        external["loose_plug_allowed"],
        f"{label}.external_contacts.loose_plug_allowed",
        minimum=1,
    )

    final_contacts = _validate_contacts(
        report["final_contacts"], f"{label}.final_contacts"
    )
    probe = _mapping(report["preinsert_probe"], f"{label}.preinsert_probe")
    _exact_keys(probe, _PREINSERT_PROBE_KEYS, f"{label}.preinsert_probe")
    if probe["status"] != "PASSED_AT_PREINSERT_OUTSIDE_ENTRY":
        raise EvidenceError(f"{label}.preinsert_probe.status mismatch")
    for key in (
        "passed",
        "prior_visual_pick_passed",
        "same_world_capture_gate",
        "object_pose_write_gate",
        "zero_preentry_contact_gate",
        "body_contact_retention_gate",
        "torque_hard_stop_gate",
        "finite_preinsert_gate",
        "tracking_and_speed_gate",
        "tcp_target_gate",
        "outside_entry_gate",
    ):
        _require_bool(probe[key], True, f"{label}.preinsert_probe.{key}")
    for key in (
        "assembly_success_claimed",
        "engage_executed",
        "home_return_executed",
        "insertion_executed",
        "production_control_authorized",
        "truth_pose_feedback_used_for_target",
        "truth_xy_used_for_target",
        "twist_executed",
    ):
        _require_bool(probe[key], False, f"{label}.preinsert_probe.{key}")
    if probe["translation_source"] != "visual_fixed_receptacle_xy":
        raise EvidenceError(f"{label}.preinsert_probe translation mismatch")
    if probe["orientation_source"] != "registered_nominal_fk":
        raise EvidenceError(f"{label}.preinsert_probe orientation mismatch")
    if _integer(
        probe["checked_physics_steps"],
        f"{label}.preinsert_probe.checked_physics_steps",
        minimum=1,
    ) < 1:
        raise EvidenceError(f"{label}.preinsert_probe has no checked steps")
    _integer(
        probe["continuation_global_steps"],
        f"{label}.preinsert_probe.continuation_global_steps",
        minimum=1,
    )
    if _integer(
        probe["minimum_body_contact_finger_count"],
        f"{label}.preinsert_probe.minimum_body_contact_finger_count",
    ) != 3:
        raise EvidenceError(f"{label} did not retain all three finger bodies")
    if _integer(
        probe["loose_fixed_contact_records"],
        f"{label}.preinsert_probe.loose_fixed_contact_records",
    ) != 0:
        raise EvidenceError(f"{label} has loose/fixed pre-entry contact")
    if probe["final_contacts"] != report["final_contacts"]:
        raise EvidenceError(f"{label} final contact snapshots differ")
    _validate_contacts(
        probe["final_contacts"], f"{label}.preinsert_probe.final_contacts"
    )

    torque_peak = _number(
        probe["maximum_post_tare_absolute_delta_nm"],
        f"{label}.preinsert_probe.maximum_post_tare_absolute_delta_nm",
        minimum=0.0,
    )
    if not _same(
        torque_peak,
        _number(
            report["maximum_post_tare_absolute_delta_nm"],
            f"{label}.maximum_post_tare_absolute_delta_nm",
            minimum=0.0,
        ),
    ):
        raise EvidenceError(f"{label} torque maxima differ")
    if not torque_peak < thresholds["strict_torque_upper_nm"]:
        raise EvidenceError(f"{label} torque must remain strictly below 2 Nm")
    for field in ("contact_torque_deltas_nm", "final_torque_deltas_nm"):
        channels = _mapping(report[field], f"{label}.{field}")
        if not channels:
            raise EvidenceError(f"{label}.{field} must not be empty")
        for channel, raw_delta in channels.items():
            delta = _number(raw_delta, f"{label}.{field}.{channel}")
            if abs(delta) >= thresholds["strict_torque_upper_nm"]:
                raise EvidenceError(f"{label}.{field}.{channel} reached 2 Nm")
    loaded = _integer(
        report["loaded_torque_channels"],
        f"{label}.loaded_torque_channels",
        minimum=3,
    )
    final_loaded = _integer(
        report["final_loaded_torque_channels"],
        f"{label}.final_loaded_torque_channels",
        minimum=3,
    )

    speed = _number(
        probe["maximum_joint_speed_rad_s"],
        f"{label}.preinsert_probe.maximum_joint_speed_rad_s",
        minimum=0.0,
    )
    tracking = _number(
        probe["maximum_arm_tracking_error_rad"],
        f"{label}.preinsert_probe.maximum_arm_tracking_error_rad",
        minimum=0.0,
    )
    joint_limit = _number(
        probe["maximum_joint_limit_violation_rad"],
        f"{label}.preinsert_probe.maximum_joint_limit_violation_rad",
        minimum=0.0,
    )
    final_tracking = _number(
        probe["final_target_tracking_error_rad"],
        f"{label}.preinsert_probe.final_target_tracking_error_rad",
        minimum=0.0,
    )
    tcp_position = _number(
        probe["final_tcp_position_error_m"],
        f"{label}.preinsert_probe.final_tcp_position_error_m",
        minimum=0.0,
    )
    tcp_axis = _number(
        probe["final_tcp_axis_error_rad"],
        f"{label}.preinsert_probe.final_tcp_axis_error_rad",
        minimum=0.0,
    )
    limits = (
        (speed, "maximum_joint_speed_rad_s"),
        (tracking, "maximum_arm_tracking_error_rad"),
        (joint_limit, "maximum_joint_limit_violation_rad"),
        (final_tracking, "maximum_arm_tracking_error_rad"),
        (tcp_position, "maximum_tcp_position_error_m"),
        (tcp_axis, "maximum_tcp_axis_error_rad"),
    )
    for observed, threshold_name in limits:
        if observed > thresholds[threshold_name]:
            raise EvidenceError(
                f"{label} exceeds {threshold_name}: {observed}"
            )
    for report_key, observed in (
        ("maximum_joint_speed_rad_s", speed),
        ("maximum_arm_tracking_error_rad", tracking),
        ("maximum_joint_limit_violation_rad", joint_limit),
    ):
        if not _same(
            _number(report[report_key], f"{label}.{report_key}"), observed
        ):
            raise EvidenceError(f"{label}.{report_key} differs from probe")
    for key in (
        "final_body_observable_angular_speed_rad_s",
        "final_body_observable_linear_speed_m_s",
        "final_observable_joint_speed_rad_s",
        "final_solver_joint_speed_rad_s",
    ):
        _number(report[key], f"{label}.{key}", minimum=0.0)

    alignment = _mapping(
        probe["post_hoc_actual_alignment"],
        f"{label}.preinsert_probe.post_hoc_actual_alignment",
    )
    _exact_keys(
        alignment,
        _ALIGNMENT_KEYS,
        f"{label}.preinsert_probe.post_hoc_actual_alignment",
    )
    if alignment["scope"] != (
        "truth_evaluation_after_motion_never_target_or_correction"
    ):
        raise EvidenceError(f"{label} actual-alignment scope mismatch")
    gap = _number(alignment["gap_m"], f"{label}.alignment.gap_m")
    entry_gap = _number(
        alignment["entry_gap_m"], f"{label}.alignment.entry_gap_m"
    )
    commanded_gap = _number(
        alignment["commanded_preinsert_gap_m"],
        f"{label}.alignment.commanded_preinsert_gap_m",
    )
    if not _same(entry_gap, thresholds["entry_gap_m"]):
        raise EvidenceError(f"{label} entry gap changed")
    if not _same(
        commanded_gap, thresholds["commanded_preinsert_gap_m"]
    ):
        raise EvidenceError(f"{label} commanded preinsert gap changed")
    if gap < entry_gap:
        raise EvidenceError(f"{label} actual gap entered the 10 mm datum")
    lateral = _number(
        alignment["lateral_error_m"],
        f"{label}.alignment.lateral_error_m",
        minimum=0.0,
    )
    axis = _number(
        alignment["axis_error_rad"],
        f"{label}.alignment.axis_error_rad",
        minimum=0.0,
    )
    combined = _number(
        alignment["combined_entry_error_m"],
        f"{label}.alignment.combined_entry_error_m",
        minimum=0.0,
    )

    return {
        "run_id": spec.run_id,
        "trial_id": spec.expected_trial_id,
        "authored_loose_xy_m": list(spec.expected_authored_loose_xy_m),
        "authored_fixed_xy_m": list(spec.expected_authored_fixed_xy_m),
        "visual_loose_target_xy_m": list(pick["loose_target"]),
        "visual_fixed_target_xy_m": list(pick["fixed_target"]),
        "capture_id": pick["capture_id"],
        "asset_sha256": asset_sha256,
        "loose_xy_error_m": loose_xy_error,
        "fixed_xy_error_m": fixed_xy_error,
        "maximum_post_tare_absolute_delta_nm": torque_peak,
        "loaded_torque_channels": loaded,
        "final_loaded_torque_channels": final_loaded,
        "maximum_joint_speed_rad_s": speed,
        "maximum_arm_tracking_error_rad": tracking,
        "maximum_joint_limit_violation_rad": joint_limit,
        "final_target_tracking_error_rad": final_tracking,
        "final_tcp_position_error_m": tcp_position,
        "final_tcp_axis_error_rad": tcp_axis,
        "actual_gap_m": gap,
        "outside_entry_margin_m": gap - entry_gap,
        "lateral_error_m": lateral,
        "axis_error_rad": axis,
        "combined_entry_error_m": combined,
        "engage_lateral_margin_m": (
            thresholds["maximum_engage_lateral_error_m"] - lateral
        ),
        "engage_axis_margin_rad": (
            thresholds["maximum_engage_axis_error_rad"] - axis
        ),
        "engage_combined_margin_m": (
            thresholds["maximum_engage_combined_entry_error_m"]
            - combined
        ),
        "final_contacts": final_contacts,
    }


def _statistics(values: Sequence[Real]) -> dict[str, Any]:
    if not values:
        raise EvidenceError("cannot aggregate an empty statistic")
    numbers = [float(value) for value in values]
    for index, value in enumerate(numbers):
        if not math.isfinite(value):
            raise EvidenceError(f"statistic[{index}] must be finite")
    return {
        "count": len(numbers),
        "minimum": min(numbers),
        "maximum": max(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "sample_standard_deviation": (
            statistics.stdev(numbers) if len(numbers) > 1 else None
        ),
    }


def _manifest_file_binding(
    path: Path,
    repository: Path,
    schema_version: str,
) -> dict[str, Any]:
    return {
        "path": _shown_path(path, repository),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema_version": schema_version,
    }


def aggregate_evidence(
    config_path: str | Path,
    repository: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all four sources and return limited aggregate evidence."""

    root = Path(repository).resolve()
    config = load_evidence_config(config_path, root)
    manifest, source_paths = load_complete_source_manifest(
        config, root, manifest_path
    )
    metrics: list[dict[str, Any]] = []
    for spec, paths in zip(config.runs, source_paths):
        report = _load_json(paths["report"], f"{spec.run_id}.report")
        metrics.append(
            _validate_report(report, paths, spec, config.thresholds)
        )

    if len(metrics) != EXPECTED_RUN_COUNT:
        raise EvidenceError("exactly four validated metrics are required")
    if len({tuple(item["authored_loose_xy_m"]) for item in metrics}) != 4:
        raise EvidenceError("authored loose XY coverage is not four positions")
    if len({tuple(item["authored_fixed_xy_m"]) for item in metrics}) != 1:
        raise EvidenceError("authored fixed XY changed across runs")
    if len({item["asset_sha256"] for item in metrics}) != 1:
        raise EvidenceError("D38999 asset SHA differs across runs")
    if len({item["capture_id"] for item in metrics}) != 4:
        raise EvidenceError("capture IDs must be independent across runs")

    blocking_id = config.policy["engage_gate_blocking_run_id"]
    blocking = next(
        (item for item in metrics if item["run_id"] == blocking_id), None
    )
    if blocking is None:
        raise EvidenceError("engage-gate blocking run is absent")
    for key in (
        "engage_lateral_margin_m",
        "engage_axis_margin_rad",
        "engage_combined_margin_m",
    ):
        if not blocking[key] < 0.0:
            raise EvidenceError(
                f"blocking run no longer exceeds {key}; "
                "v1 policy requires review"
            )

    thresholds = config.thresholds
    per_run: list[dict[str, Any]] = []
    for item in metrics:
        per_run.append(
            {
                **item,
                "torque_margin_to_strict_2nm_nm": (
                    thresholds["strict_torque_upper_nm"]
                    - item["maximum_post_tare_absolute_delta_nm"]
                ),
                "joint_speed_margin_rad_s": (
                    thresholds["maximum_joint_speed_rad_s"]
                    - item["maximum_joint_speed_rad_s"]
                ),
                "arm_tracking_margin_rad": (
                    thresholds["maximum_arm_tracking_error_rad"]
                    - item["maximum_arm_tracking_error_rad"]
                ),
                "joint_limit_margin_rad": (
                    thresholds["maximum_joint_limit_violation_rad"]
                    - item["maximum_joint_limit_violation_rad"]
                ),
                "tcp_position_margin_m": (
                    thresholds["maximum_tcp_position_error_m"]
                    - item["final_tcp_position_error_m"]
                ),
                "tcp_axis_margin_rad": (
                    thresholds["maximum_tcp_axis_error_rad"]
                    - item["final_tcp_axis_error_rad"]
                ),
                "original_visual_pick_passed": True,
                "preinsert_passed": True,
                "same_world_and_capture_verified": True,
                "zero_pose_writes_verified": True,
                "zero_forbidden_preentry_contacts_verified": True,
                "three_finger_body_contact_verified": True,
                "finite_tracking_and_speed_verified": True,
                "actual_gap_outside_entry_verified": True,
            }
        )

    def stats(key: str) -> dict[str, Any]:
        return _statistics([item[key] for item in per_run])

    loose_x = [item["authored_loose_xy_m"][0] for item in per_run]
    loose_y = [item["authored_loose_xy_m"][1] for item in per_run]
    fixed_visual_x = [item["visual_fixed_target_xy_m"][0] for item in per_run]
    fixed_visual_y = [item["visual_fixed_target_xy_m"][1] for item in per_run]
    selected_manifest = _repository_path(
        config.manifest_path,
        root,
        "config.source_manifest_path",
        require_file=True,
    )
    source_path = _repository_path(
        AGGREGATOR_SOURCE_PATH,
        root,
        "aggregator_source_path",
        require_file=True,
    )

    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": EVIDENCE_PASS,
        "summary": {
            "pass_fraction": "4/4",
            "run_count": EXPECTED_RUN_COUNT,
            "passed_run_count": EXPECTED_RUN_COUNT,
            "all_original_visual_picks_passed": True,
            "all_preinsert_probes_passed": True,
            "all_paths_sizes_hashes_and_schemas_verified": True,
            "all_same_world_capture_gates_passed": True,
            "all_zero_pose_write_gates_passed": True,
            "all_zero_forbidden_contact_gates_passed": True,
            "all_three_finger_body_contact_gates_passed": True,
            "all_torque_strictly_below_2nm": True,
            "all_finite_tracking_and_speed_gates_passed": True,
            "all_actual_gaps_outside_10mm_entry": True,
        },
        "evidence_scope": {
            "classification": (
                "LIMITED_INDEPENDENT_MULTI_POSITION_VISUAL_XY_"
                "PICK_TO_PREINSERT_OUTSIDE_ENTRY"
            ),
            "statistics_only": True,
            "same_condition_repeatability_claimed": False,
            "entry_gap_m": thresholds["entry_gap_m"],
            "commanded_preinsert_gap_m": thresholds[
                "commanded_preinsert_gap_m"
            ],
        },
        "xy_coverage": {
            "authored_loose_position_m": {
                "x_minimum": min(loose_x),
                "x_maximum": max(loose_x),
                "x_span": max(loose_x) - min(loose_x),
                "y_minimum": min(loose_y),
                "y_maximum": max(loose_y),
                "y_span": max(loose_y) - min(loose_y),
                "distinct_position_count": EXPECTED_RUN_COUNT,
            },
            "visual_fixed_target_m": {
                "x_minimum": min(fixed_visual_x),
                "x_maximum": max(fixed_visual_x),
                "x_span": max(fixed_visual_x) - min(fixed_visual_x),
                "y_minimum": min(fixed_visual_y),
                "y_maximum": max(fixed_visual_y),
                "y_span": max(fixed_visual_y) - min(fixed_visual_y),
            },
            "loose_xy_error_m": stats("loose_xy_error_m"),
            "fixed_xy_error_m": stats("fixed_xy_error_m"),
        },
        "observed_margins": {
            "actual_gap_m": stats("actual_gap_m"),
            "outside_10mm_entry_margin_m": stats(
                "outside_entry_margin_m"
            ),
            "torque_margin_to_strict_2nm_nm": stats(
                "torque_margin_to_strict_2nm_nm"
            ),
            "joint_speed_margin_rad_s": stats("joint_speed_margin_rad_s"),
            "arm_tracking_margin_rad": stats("arm_tracking_margin_rad"),
            "joint_limit_margin_rad": stats("joint_limit_margin_rad"),
            "tcp_position_margin_m": stats("tcp_position_margin_m"),
            "tcp_axis_margin_rad": stats("tcp_axis_margin_rad"),
        },
        "engage_gate_assessment": {
            "ready_for_engage": False,
            "thresholds": {
                "maximum_lateral_error_m": thresholds[
                    "maximum_engage_lateral_error_m"
                ],
                "maximum_axis_error_rad": thresholds[
                    "maximum_engage_axis_error_rad"
                ],
                "maximum_combined_entry_error_m": thresholds[
                    "maximum_engage_combined_entry_error_m"
                ],
            },
            "lateral_gate_margin_m": stats("engage_lateral_margin_m"),
            "axis_gate_margin_rad": stats("engage_axis_margin_rad"),
            "combined_gate_margin_m": stats("engage_combined_margin_m"),
            "blocking_run": {
                "run_id": blocking_id,
                "observed_lateral_error_m": blocking["lateral_error_m"],
                "observed_axis_error_rad": blocking["axis_error_rad"],
                "observed_combined_entry_error_m": blocking[
                    "combined_entry_error_m"
                ],
                "lateral_excess_m": -blocking["engage_lateral_margin_m"],
                "axis_excess_rad": -blocking["engage_axis_margin_rad"],
                "combined_excess_m": -blocking[
                    "engage_combined_margin_m"
                ],
                "all_three_engage_gates_failed": True,
            },
        },
        "per_run": per_run,
        "claims": {
            "four_position_visual_xy_pick_to_preinsert_evidence": True,
            "engage_executed_or_authorized": False,
            "insertion_executed_or_authorized": False,
            "twist_executed_or_authorized": False,
            "home_return_executed_or_authorized": False,
            "full_6d_claimed": False,
            "production_control_authorized": False,
            "full_end_to_end_assembly_claimed": False,
            "same_condition_repeatability_claimed": False,
        },
        "limitations": [
            (
                "all trials stop at preinsert outside the 10 mm entry datum; "
                "engage and insertion were not executed"
            ),
            (
                "the plus10_xy run exceeds the lateral, axis and combined "
                "entry-alignment gates"
            ),
            (
                "orientation remains registered nominal FK rather than "
                "visually observed full 6-DoF"
            ),
            "twist and return Home were not executed",
            "the experiment is not production-control authorized",
            (
                "four different authored positions are multi-position "
                "coverage, not same-condition repeatability"
            ),
        ],
        "provenance": {
            "source_manifest": _manifest_file_binding(
                selected_manifest,
                root,
                SOURCE_MANIFEST_SCHEMA_VERSION,
            ),
            "config": _config_binding_for(config.path, root),
            "aggregator_source": _manifest_file_binding(
                source_path, root, "python_source_v1"
            ),
            "source_runs": manifest["runs"],
            "same_d38999_asset_sha256": metrics[0]["asset_sha256"],
        },
    }
    json.dumps(evidence, allow_nan=False, sort_keys=True)
    return evidence


def write_evidence_report(
    evidence: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write one immutable aggregate report without overwriting evidence."""

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(
            evidence,
            stream,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path(DEFAULT_CONFIG_PATH)
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repository / config_path
    try:
        if args.refresh_manifest:
            if args.output is not None:
                raise EvidenceError(
                    "--refresh-manifest and --output are separate operations"
                )
            manifest = write_source_manifest(
                config_path,
                repository,
                args.manifest,
            )
            print(json.dumps(manifest, allow_nan=False, sort_keys=True))
            if manifest["status"] != MANIFEST_COMPLETE:
                raise EvidenceError(
                    "manifest remains incomplete; all four approved runs "
                    "must exist before aggregation"
                )
            return 0
        evidence = aggregate_evidence(
            config_path,
            repository,
            args.manifest,
        )
        if args.output is not None:
            output = args.output
            if not output.is_absolute():
                output = repository / output
            write_evidence_report(evidence, output)
        print(json.dumps(evidence, allow_nan=False, sort_keys=True))
        return 0
    except (
        EvidenceError,
        FileExistsError,
        FileNotFoundError,
        OSError,
    ) as error:
        print(
            f"visual XY preinsert evidence rejected: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATOR_SOURCE_PATH",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MANIFEST_PATH",
    "EVIDENCE_PASS",
    "EVIDENCE_SCHEMA_VERSION",
    "EXPECTED_RUN_COUNT",
    "EvidenceConfig",
    "EvidenceError",
    "MANIFEST_COMPLETE",
    "MANIFEST_INCOMPLETE",
    "PREINSERT_PLAN_SCHEMA_VERSION",
    "PICK_PLAN_SCHEMA_VERSION",
    "RunSpec",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "SOURCE_REPORT_SCHEMA_VERSION",
    "aggregate_evidence",
    "build_source_manifest",
    "load_complete_source_manifest",
    "load_evidence_config",
    "main",
    "sha256_file",
    "write_evidence_report",
    "write_source_manifest",
]
