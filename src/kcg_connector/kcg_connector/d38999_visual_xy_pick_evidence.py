"""Strict CPU-only aggregation for the D38999 visual-XY pick probes.

The four source trials were executed by Isaac Sim, but this module never
imports Isaac, starts a renderer, or touches the scene.  It verifies the byte
size, SHA-256 and schema of every report, console log, config and CPU plan,
then archives the otherwise-ephemeral console logs and writes a compact
multi-position evidence report.

This is deliberately a *limited* evidence adapter.  It can show that four
independent visual-XY targets led to successful physical picks; it cannot
upgrade registered nominal orientation to visual 6-DoF, authorize production
control, close an RL readiness gate, or turn four different positions into a
same-condition repeatability study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


REQUEST_SCHEMA_VERSION = "kcg_d38999_visual_xy_pick_evidence_request_v1"
EVIDENCE_SCHEMA_VERSION = "kcg_d38999_visual_xy_pick_evidence_v1"
MANIFEST_SCHEMA_VERSION = "kcg_d38999_visual_xy_pick_evidence_manifest_v1"
SOURCE_REPORT_SCHEMA_VERSION = "kcg_d38999_visual_xy_pick_report_v1"
SOURCE_PLAN_SCHEMA_VERSION = "kcg_d38999_visual_xy_pick_probe_v1"
SOURCE_CONFIG_SCHEMA_VERSION = "kcg_d38999_visual_xy_pick_probe_v1"
SOURCE_LOG_SCHEMA_VERSION = "kcg_d38999_visual_xy_pick_console_log_v1"

EXPECTED_RUN_COUNT = 4
PASS_BANNER = "ISAAC D38999 VISUAL XY PICK PROBE V1 PASSED"
FAIL_BANNER = "ISAAC D38999 VISUAL XY PICK PROBE V1 FAILED"
AGGREGATOR_SOURCE_PATH = (
    "src/kcg_connector/kcg_connector/d38999_visual_xy_pick_evidence.py"
)
DEFAULT_REQUEST_PATH = (
    "src/kcg_connector/config/d38999_visual_xy_pick_evidence_v1.yaml"
)

_REQUEST_KEYS = {
    "schema_version",
    "expected_run_count",
    "policy",
    "runs",
}
_POLICY_KEYS = {
    "evidence_scope",
    "require_all_passed",
    "statistics_only",
    "claim_same_condition_repeatability",
    "claim_full_6d",
    "claim_arbitrary_pose",
    "claim_production_control",
    "close_rl_readiness_gate",
}
_RUN_KEYS = {
    "run_id",
    "expected_trial_id",
    "expected_loose_xy_m",
    "expected_fixed_xy_m",
    "report",
    "console_log",
    "config",
    "cpu_plan",
}
_BINDING_KEYS = {"path", "sha256", "size_bytes", "schema_version"}

# Exact root keys make an unexpected source-report schema change fail closed.
_SOURCE_REPORT_KEYS = {
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
    "unsupported_gate",
    "zero_forbidden_contacts",
}
_CPU_PLAN_KEYS = {
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
_FINAL_CONTACT_KEYS = {
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
_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class EvidenceError(ValueError):
    """Raised when a source cannot authorize the limited evidence report."""


def sha256_file(path: Path) -> str:
    """Hash a file in chunks so console logs need not be loaded at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise EvidenceError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} must be non-empty text")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise EvidenceError(f"{label} must be lowercase SHA-256")
    return digest


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
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
    return (_number(value[0], f"{label}[0]"), _number(value[1], f"{label}[1]"))


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise EvidenceError(f"{label} must remain {expected}")


def _same_xy(first: Sequence[float], second: Sequence[float]) -> bool:
    return all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-12)
        for a, b in zip(first, second)
    )


def _resolve(
    raw_path: Any,
    repository: Path,
    label: str,
    *,
    repository_only: bool,
) -> Path:
    candidate = Path(_text(raw_path, label)).expanduser()
    if not candidate.is_absolute():
        candidate = repository / candidate
    resolved = candidate.resolve()
    if repository_only:
        try:
            resolved.relative_to(repository)
        except ValueError as error:
            raise EvidenceError(f"{label} must remain inside repository") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _verify_binding(
    value: Any,
    repository: Path,
    label: str,
    *,
    expected_schema: str,
    repository_only: bool,
) -> tuple[dict[str, Any], Path]:
    binding = _mapping(value, label)
    _exact_keys(binding, _BINDING_KEYS, label)
    if binding["schema_version"] != expected_schema:
        raise EvidenceError(f"{label}.schema_version mismatch")
    expected_size = _integer(binding["size_bytes"], f"{label}.size_bytes", minimum=1)
    expected_digest = _sha256(binding["sha256"], f"{label}.sha256")
    path = _resolve(
        binding["path"], repository, f"{label}.path", repository_only=repository_only
    )
    if path.stat().st_size != expected_size:
        raise EvidenceError(f"{label} byte-size mismatch")
    if sha256_file(path) != expected_digest:
        raise EvidenceError(f"{label} SHA-256 mismatch")
    return dict(binding), path


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} is not valid UTF-8 JSON") from error
    return _mapping(value, label)


def _validate_policy(value: Any) -> None:
    policy = _mapping(value, "policy")
    _exact_keys(policy, _POLICY_KEYS, "policy")
    expected = {
        "evidence_scope": "independent_multi_position_visual_xy_pick_only",
        "require_all_passed": True,
        "statistics_only": True,
        "claim_same_condition_repeatability": False,
        "claim_full_6d": False,
        "claim_arbitrary_pose": False,
        "claim_production_control": False,
        "close_rl_readiness_gate": False,
    }
    for key, expected_value in expected.items():
        if policy[key] != expected_value:
            raise EvidenceError(f"policy.{key} must remain {expected_value!r}")


def _validate_console_log(path: Path, report: Mapping[str, Any], label: str) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{label} must be UTF-8") from error
    if lines.count(PASS_BANNER) != 1:
        raise EvidenceError(f"{label} must contain one exact PASS banner")
    if FAIL_BANNER in lines or any("Traceback (most recent call last)" in line for line in lines):
        raise EvidenceError(f"{label} contains failure evidence")

    report_lines: list[tuple[int, Mapping[str, Any]]] = []
    for index, line in enumerate(lines):
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(candidate, Mapping)
            and candidate.get("schema_version") == SOURCE_REPORT_SCHEMA_VERSION
        ):
            report_lines.append((index, candidate))
    if len(report_lines) != 1:
        raise EvidenceError(f"{label} must contain one source report JSON line")
    report_index, logged_report = report_lines[0]
    if logged_report != report:
        raise EvidenceError(f"{label} embedded report differs from report.json")
    if lines.index(PASS_BANNER) <= report_index:
        raise EvidenceError(f"{label} PASS banner must follow its report")


def _validate_config(path: Path, report: Mapping[str, Any], label: str) -> None:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise EvidenceError(f"{label} is not valid UTF-8 YAML") from error
    config = _mapping(document, label)
    if config.get("schema_version") != SOURCE_CONFIG_SCHEMA_VERSION:
        raise EvidenceError(f"{label} internal schema mismatch")
    _require_bool(config.get("enabled_by_default"), False, f"{label}.enabled_by_default")
    if config.get("status") != "prepared_independent_visual_xy_pick_probe":
        raise EvidenceError(f"{label}.status mismatch")
    if sha256_file(path) != report.get("config_sha256"):
        raise EvidenceError(f"{label} hash differs from source report")
    try:
        internal_path = Path(_text(report.get("config_path"), "report.config_path")).resolve()
    except OSError as error:
        raise EvidenceError("report.config_path cannot be resolved") from error
    if internal_path != path:
        raise EvidenceError(f"{label} path differs from source report")


def _validate_cpu_plan(
    plan: Mapping[str, Any], report: Mapping[str, Any], path: Path, label: str
) -> dict[str, tuple[float, float]]:
    _exact_keys(plan, _CPU_PLAN_KEYS, label)
    if plan.get("schema_version") != SOURCE_PLAN_SCHEMA_VERSION:
        raise EvidenceError(f"{label} internal schema mismatch")
    if plan != report.get("cpu_plan"):
        raise EvidenceError(f"{label} differs from report.cpu_plan")
    if Path(_text(report.get("cpu_plan_path"), "report.cpu_plan_path")).resolve() != path:
        raise EvidenceError(f"{label} path differs from source report")
    if plan.get("status") != "CPU_PLAN_READY_FOR_INDEPENDENT_ISAAC_PROBE":
        raise EvidenceError(f"{label}.status mismatch")
    for key in ("full_6d", "production_control_authorized", "uses_truth_xy_for_target"):
        _require_bool(plan.get(key), False, f"{label}.{key}")
    _require_bool(plan.get("collision_planned"), False, f"{label}.collision_planned")

    adapter = _mapping(plan.get("adapter"), f"{label}.adapter")
    _exact_keys(adapter, _ADAPTER_KEYS, f"{label}.adapter")
    expected_adapter_values = {
        "eligible_for_independent_probe": True,
        "full_6d": False,
        "production_control_authorized": False,
        "pose_provider_control_authorized": False,
        "uses_truth_orientation": False,
        "yaw_observed": False,
    }
    for key, expected in expected_adapter_values.items():
        _require_bool(adapter.get(key), expected, f"{label}.adapter.{key}")
    if adapter.get("orientation_source") != "registered_nominal":
        raise EvidenceError(f"{label}.adapter.orientation_source mismatch")
    translation_source = _text(
        adapter.get("translation_source"), f"{label}.adapter.translation_source"
    )
    if not translation_source.startswith("vision_"):
        raise EvidenceError(f"{label}.adapter translation is not vision-derived")
    if adapter.get("rejection_reasons") != []:
        raise EvidenceError(f"{label}.adapter has rejection reasons")

    targets = _mapping(plan.get("tcp_targets_world_m"), f"{label}.tcp_targets_world_m")
    adapter_targets = _mapping(adapter.get("world_targets"), f"{label}.adapter.world_targets")
    loose = _xy(targets.get("grasp_tcp", [])[0:2], f"{label}.grasp_target_xy")
    fixed = _xy(adapter_targets.get("engage_tcp", [])[0:2], f"{label}.engage_target_xy")
    for key in ("pregrasp_tcp", "closure_clearance_tcp"):
        if not _same_xy(_xy(targets.get(key, [])[0:2], f"{label}.{key}_xy"), loose):
            raise EvidenceError(f"{label}.{key} XY differs from visual grasp target")
    return {"loose": loose, "fixed": fixed}


def _validate_contacts(value: Any, label: str) -> dict[str, Any]:
    contacts = _mapping(value, label)
    _exact_keys(contacts, _FINAL_CONTACT_KEYS, label)
    for key in (
        "finger_loose_plug_records",
        "grip_material_records",
        "robot_loose_plug_records",
    ):
        _integer(contacts[key], f"{label}.{key}", minimum=1)
    for key in ("plug_table_records", "unexpected_robot_link_records"):
        if _integer(contacts[key], f"{label}.{key}") != 0:
            raise EvidenceError(f"{label}.{key} must be zero")
    fingers = _mapping(contacts["finger_body_group_records"], f"{label}.finger_body_group_records")
    _exact_keys(fingers, {"f1", "f2", "f3"}, f"{label}.finger_body_group_records")
    counts: dict[str, dict[str, int]] = {}
    for finger in ("f1", "f2", "f3"):
        record = _mapping(fingers[finger], f"{label}.{finger}")
        _exact_keys(record, {"body", "nut"}, f"{label}.{finger}")
        counts[finger] = {
            "body": _integer(record["body"], f"{label}.{finger}.body", minimum=1),
            "nut": _integer(record["nut"], f"{label}.{finger}.nut", minimum=1),
        }
    return {"counts": counts, **{key: contacts[key] for key in contacts if key != "finger_body_group_records"}}


def _validate_report(
    report: Mapping[str, Any],
    *,
    expected_trial_id: str,
    expected_loose_xy: tuple[float, float],
    expected_fixed_xy: tuple[float, float],
    cpu_plan_path: Path,
) -> dict[str, Any]:
    _exact_keys(report, _SOURCE_REPORT_KEYS, "report")
    if report.get("schema_version") != SOURCE_REPORT_SCHEMA_VERSION:
        raise EvidenceError("report.schema_version mismatch")
    if report.get("trial_id") != expected_trial_id:
        raise EvidenceError("report.trial_id mismatch")

    expected_true = (
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
    )
    for key in expected_true:
        _require_bool(report.get(key), True, f"report.{key}")
    for key in (
        "full_6d",
        "production_control_authorized",
        "truth_xy_used_for_target",
        "collision_planned",
        "gui",
    ):
        _require_bool(report.get(key), False, f"report.{key}")
    if report.get("orientation_source") != "registered_nominal":
        raise EvidenceError("report.orientation_source mismatch")

    authored = _mapping(report.get("authored_before_physics"), "report.authored_before_physics")
    if not _same_xy(_xy(authored.get("loose_plug_xy_m"), "authored.loose_plug_xy_m"), expected_loose_xy):
        raise EvidenceError("authored loose XY differs from request")
    if not _same_xy(_xy(authored.get("fixed_receptacle_xy_m"), "authored.fixed_receptacle_xy_m"), expected_fixed_xy):
        raise EvidenceError("authored fixed XY differs from request")
    if _number(authored.get("loose_yaw_rad"), "authored.loose_yaw_rad") != 0.0:
        raise EvidenceError("authored loose yaw must remain nominal")
    if _number(authored.get("fixed_yaw_rad"), "authored.fixed_yaw_rad") != 0.0:
        raise EvidenceError("authored fixed yaw must remain nominal")

    plan = _mapping(report.get("cpu_plan"), "report.cpu_plan")
    if plan.get("trial_id") != expected_trial_id:
        raise EvidenceError("cpu plan trial_id mismatch")
    target_xy = _validate_cpu_plan(plan, report, cpu_plan_path, "cpu_plan")

    pose_provider = _mapping(report.get("pose_provider"), "report.pose_provider")
    for key in ("uses_truth_position", "uses_truth_orientation", "full_6d", "control_authorized"):
        _require_bool(pose_provider.get(key), False, f"report.pose_provider.{key}")
    if pose_provider.get("purpose") != "preflight":
        raise EvidenceError("report.pose_provider.purpose mismatch")
    diagnostics = _mapping(pose_provider.get("diagnostics"), "report.pose_provider.diagnostics")
    endpoints = _mapping(diagnostics.get("endpoints"), "report.pose_provider.diagnostics.endpoints")
    observed_loose = _xy(
        _mapping(endpoints.get("loose_plug"), "endpoints.loose_plug").get("estimated_world_xy_m"),
        "endpoints.loose_plug.estimated_world_xy_m",
    )
    observed_fixed = _xy(
        _mapping(endpoints.get("fixed_receptacle"), "endpoints.fixed_receptacle").get("estimated_world_xy_m"),
        "endpoints.fixed_receptacle.estimated_world_xy_m",
    )
    if not _same_xy(target_xy["loose"], observed_loose):
        raise EvidenceError("visual loose XY was not preserved into grasp target")
    if not _same_xy(target_xy["fixed"], observed_fixed):
        raise EvidenceError("visual fixed XY was not preserved into adapter target")

    rgbd = _mapping(report.get("rgbd_capture"), "report.rgbd_capture")
    _require_bool(rgbd.get("passed"), True, "report.rgbd_capture.passed")
    _require_bool(rgbd.get("camera_observation_present"), True, "report.rgbd_capture.camera_observation_present")
    _require_bool(rgbd.get("full_keyed_6d_vision_pose_claimed"), False, "report.rgbd_capture.full_keyed_6d_vision_pose_claimed")
    # The reusable capture helper deliberately makes no control claim.  The
    # independent plan-use proof above is the narrower, downstream evidence.
    _require_bool(rgbd.get("masked_rgbd_xy_used_for_control"), False, "report.rgbd_capture.masked_rgbd_xy_used_for_control")
    estimator = _mapping(rgbd.get("position_estimator"), "report.rgbd_capture.position_estimator")
    _require_bool(estimator.get("uses_registered_truth_xy"), False, "report.rgbd_capture.position_estimator.uses_registered_truth_xy")

    truth = _mapping(report.get("truth_evaluation"), "report.truth_evaluation")
    if truth.get("scope") != "post_hoc_truth_evaluation_not_target_input":
        raise EvidenceError("truth evaluation scope mismatch")
    _require_bool(truth.get("truth_xy_used_for_target"), False, "truth_evaluation.truth_xy_used_for_target")
    loose_error = _number(truth.get("loose_xy_error_m"), "truth_evaluation.loose_xy_error_m", minimum=0.0)
    fixed_error = _number(truth.get("fixed_xy_error_m"), "truth_evaluation.fixed_xy_error_m", minimum=0.0)

    side_effects = _mapping(report.get("runtime_side_effects"), "report.runtime_side_effects")
    _exact_keys(side_effects, _RUNTIME_SIDE_EFFECT_KEYS, "report.runtime_side_effects")
    for key in ("playing_after_restore", "playing_before_capture", "resource_cleanup_verified", "timeline_state_restored"):
        _require_bool(side_effects.get(key), True, f"runtime_side_effects.{key}")
    _require_bool(side_effects.get("truth_or_error_gate_consulted"), False, "runtime_side_effects.truth_or_error_gate_consulted")
    for key in ("endpoint_pose_writes_after_physics", "world_reset_or_clear_calls"):
        if _integer(side_effects.get(key), f"runtime_side_effects.{key}") != 0:
            raise EvidenceError(f"runtime_side_effects.{key} must be zero")
    if _integer(report.get("object_pose_writes_after_physics"), "report.object_pose_writes_after_physics") != 0:
        raise EvidenceError("report.object_pose_writes_after_physics must be zero")

    contacts = _validate_contacts(report.get("final_contacts"), "report.final_contacts")
    loaded = _integer(report.get("loaded_torque_channels"), "report.loaded_torque_channels", minimum=1)
    final_loaded = _integer(report.get("final_loaded_torque_channels"), "report.final_loaded_torque_channels", minimum=1)
    torque_channels = _mapping(report.get("contact_torque_deltas_nm"), "report.contact_torque_deltas_nm")
    if not torque_channels:
        raise EvidenceError("report.contact_torque_deltas_nm must not be empty")
    contact_peak = max(
        _number(value, f"contact_torque_deltas_nm.{name}", minimum=0.0)
        for name, value in torque_channels.items()
    )
    return {
        "trial_id": expected_trial_id,
        "authored_loose_xy_m": list(expected_loose_xy),
        "authored_fixed_xy_m": list(expected_fixed_xy),
        "visual_loose_target_xy_m": list(target_xy["loose"]),
        "visual_fixed_target_xy_m": list(target_xy["fixed"]),
        "loose_xy_error_m": loose_error,
        "fixed_xy_error_m": fixed_error,
        "body_lift_m": _number(report.get("body_lift_m"), "report.body_lift_m", minimum=0.0),
        "final_bottom_clearance_m": _number(report.get("final_bottom_clearance_m"), "report.final_bottom_clearance_m", minimum=0.0),
        "body_tcp_slip_m": _number(report.get("body_tcp_slip_m"), "report.body_tcp_slip_m", minimum=0.0),
        "maximum_post_tare_absolute_delta_nm": _number(report.get("maximum_post_tare_absolute_delta_nm"), "report.maximum_post_tare_absolute_delta_nm", minimum=0.0),
        "contact_peak_channel_delta_nm": contact_peak,
        "loaded_torque_channels": loaded,
        "final_loaded_torque_channels": final_loaded,
        "maximum_joint_speed_rad_s": _number(report.get("maximum_joint_speed_rad_s"), "report.maximum_joint_speed_rad_s", minimum=0.0),
        "maximum_arm_tracking_error_rad": _number(report.get("maximum_arm_tracking_error_rad"), "report.maximum_arm_tracking_error_rad", minimum=0.0),
        "final_contacts": contacts,
    }


def _observed_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise EvidenceError("cannot aggregate an empty statistic")
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "minimum": min(numbers),
        "maximum": max(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "sample_standard_deviation": (
            statistics.stdev(numbers) if len(numbers) >= 2 else None
        ),
    }


def _binding_for(path: Path, schema_version: str, *, shown_path: str | None = None) -> dict[str, Any]:
    return {
        "path": shown_path if shown_path is not None else str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema_version": schema_version,
    }


def _repository_relative(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository))
    except ValueError:
        return str(path.resolve())


def _parse_request(request_path: Path, repository: Path) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    try:
        document = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise EvidenceError("request is not valid UTF-8 YAML") from error
    request = _mapping(document, "request")
    _exact_keys(request, _REQUEST_KEYS, "request")
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise EvidenceError("request.schema_version mismatch")
    if _integer(request.get("expected_run_count"), "request.expected_run_count", minimum=1) != EXPECTED_RUN_COUNT:
        raise EvidenceError(f"request.expected_run_count must equal {EXPECTED_RUN_COUNT}")
    _validate_policy(request.get("policy"))
    raw_runs = request.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != EXPECTED_RUN_COUNT:
        raise EvidenceError(f"request must contain exactly {EXPECTED_RUN_COUNT} runs")

    verified: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    trial_ids: set[str] = set()
    report_paths: set[Path] = set()
    log_paths: set[Path] = set()
    authored_positions: set[tuple[float, float]] = set()
    for index, raw_run in enumerate(raw_runs):
        label = f"runs[{index}]"
        run = _mapping(raw_run, label)
        _exact_keys(run, _RUN_KEYS, label)
        run_id = _text(run["run_id"], f"{label}.run_id")
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise EvidenceError(f"{label}.run_id is not a safe artifact name")
        trial_id = _text(run["expected_trial_id"], f"{label}.expected_trial_id")
        loose_xy = _xy(run["expected_loose_xy_m"], f"{label}.expected_loose_xy_m")
        fixed_xy = _xy(run["expected_fixed_xy_m"], f"{label}.expected_fixed_xy_m")
        if run_id in run_ids or trial_id in trial_ids:
            raise EvidenceError("run_id and expected_trial_id must be distinct")
        if loose_xy in authored_positions:
            raise EvidenceError("all four authored loose XY positions must be distinct")
        run_ids.add(run_id)
        trial_ids.add(trial_id)
        authored_positions.add(loose_xy)

        report_binding, report_path = _verify_binding(
            run["report"], repository, f"{label}.report",
            expected_schema=SOURCE_REPORT_SCHEMA_VERSION, repository_only=True,
        )
        log_binding, log_path = _verify_binding(
            run["console_log"], repository, f"{label}.console_log",
            expected_schema=SOURCE_LOG_SCHEMA_VERSION, repository_only=False,
        )
        config_binding, config_path = _verify_binding(
            run["config"], repository, f"{label}.config",
            expected_schema=SOURCE_CONFIG_SCHEMA_VERSION, repository_only=True,
        )
        plan_binding, plan_path = _verify_binding(
            run["cpu_plan"], repository, f"{label}.cpu_plan",
            expected_schema=SOURCE_PLAN_SCHEMA_VERSION, repository_only=True,
        )
        if report_path in report_paths or log_path in log_paths:
            raise EvidenceError("source report and console log paths must be distinct")
        report_paths.add(report_path)
        log_paths.add(log_path)

        report = _load_json(report_path, f"{label}.report")
        _validate_config(config_path, report, f"{label}.config")
        cpu_plan = _load_json(plan_path, f"{label}.cpu_plan")
        if cpu_plan != report.get("cpu_plan"):
            raise EvidenceError(f"{label}.cpu_plan differs from report.cpu_plan")
        metrics = _validate_report(
            report,
            expected_trial_id=trial_id,
            expected_loose_xy=loose_xy,
            expected_fixed_xy=fixed_xy,
            cpu_plan_path=plan_path,
        )
        _validate_console_log(log_path, report, f"{label}.console_log")
        verified.append(
            {
                "run_id": run_id,
                "metrics": metrics,
                "paths": {
                    "report": report_path,
                    "console_log": log_path,
                    "config": config_path,
                    "cpu_plan": plan_path,
                },
                "bindings": {
                    "report": report_binding,
                    "console_log": log_binding,
                    "config": config_binding,
                    "cpu_plan": plan_binding,
                },
                "asset_sha256": _mapping(report.get("d38999_authoring"), "report.d38999_authoring").get("asset_sha256"),
            }
        )

    asset_hashes = {_sha256(item["asset_sha256"], "asset_sha256") for item in verified}
    if len(asset_hashes) != 1:
        raise EvidenceError("all four reports must bind the same D38999 asset")
    return request, verified


def _build_evidence(
    verified: Sequence[dict[str, Any]],
    *,
    request_binding: Mapping[str, Any],
    aggregator_binding: Mapping[str, Any],
    archived_log_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = [item["metrics"] for item in verified]
    run_count = len(metrics)
    if run_count != EXPECTED_RUN_COUNT:
        raise EvidenceError("internal run-count mismatch")

    def stats(name: str) -> dict[str, Any]:
        return _observed_statistics([item[name] for item in metrics])

    all_finger_body = [
        count
        for item in metrics
        for count in (
            finger["body"]
            for finger in item["final_contacts"]["counts"].values()
        )
    ]
    all_finger_nut = [
        count
        for item in metrics
        for count in (
            finger["nut"]
            for finger in item["final_contacts"]["counts"].values()
        )
    ]
    source_runs = []
    per_trial = []
    for item in verified:
        bindings = {
            name: dict(binding) for name, binding in item["bindings"].items()
        }
        bindings["archived_console_log"] = dict(archived_log_bindings[item["run_id"]])
        source_runs.append({"run_id": item["run_id"], "bindings": bindings})
        per_trial.append({"run_id": item["run_id"], **item["metrics"]})

    loose_x = [item["authored_loose_xy_m"][0] for item in metrics]
    loose_y = [item["authored_loose_xy_m"][1] for item in metrics]
    report = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "PASS_LIMITED_MULTI_POSITION_VISUAL_XY_PICK_EVIDENCE",
        "summary": {
            "pass_fraction": f"{run_count}/{run_count}",
            "run_count": run_count,
            "passed_run_count": run_count,
            "failed_run_count": 0,
            "all_four_gpu_trials_passed": True,
            "all_hashes_sizes_and_schemas_verified": True,
        },
        "visual_xy_evidence": {
            "independent_probe_plan_consumed_visual_xy": True,
            "target_chain": (
                "semantic_mask_ray_plane_registered_height_xy -> "
                "PoseProvider diagnostics -> visual XY adapter -> "
                "fixed-q7 local IK -> physical pick"
            ),
            "truth_xy_used_for_target": False,
            "truth_scope": "post_hoc_accuracy_evaluation_only",
            "authored_loose_position_coverage_m": {
                "x_minimum": min(loose_x),
                "x_maximum": max(loose_x),
                "x_span": max(loose_x) - min(loose_x),
                "y_minimum": min(loose_y),
                "y_maximum": max(loose_y),
                "y_span": max(loose_y) - min(loose_y),
                "distinct_position_count": run_count,
            },
            "loose_xy_error_m": stats("loose_xy_error_m"),
            "fixed_xy_error_m": stats("fixed_xy_error_m"),
            "all_endpoint_xy_error_m": _observed_statistics(
                [
                    value
                    for item in metrics
                    for value in (
                        item["loose_xy_error_m"],
                        item["fixed_xy_error_m"],
                    )
                ]
            ),
        },
        "physical_pick_evidence": {
            "body_lift_m": stats("body_lift_m"),
            "final_bottom_clearance_m": stats("final_bottom_clearance_m"),
            "body_tcp_slip_m": stats("body_tcp_slip_m"),
            "maximum_post_tare_absolute_delta_nm": stats(
                "maximum_post_tare_absolute_delta_nm"
            ),
            "contact_peak_channel_delta_nm": stats(
                "contact_peak_channel_delta_nm"
            ),
            "loaded_torque_channels": stats("loaded_torque_channels"),
            "final_loaded_torque_channels": stats(
                "final_loaded_torque_channels"
            ),
            "maximum_joint_speed_rad_s": stats("maximum_joint_speed_rad_s"),
            "maximum_arm_tracking_error_rad": stats(
                "maximum_arm_tracking_error_rad"
            ),
            "final_contact_records": {
                "grip_material": _observed_statistics(
                    [item["final_contacts"]["grip_material_records"] for item in metrics]
                ),
                "robot_loose_plug": _observed_statistics(
                    [item["final_contacts"]["robot_loose_plug_records"] for item in metrics]
                ),
                "per_finger_body": _observed_statistics(all_finger_body),
                "per_finger_nut": _observed_statistics(all_finger_nut),
                "all_runs_zero_table_contacts": True,
                "all_runs_zero_unexpected_robot_link_contacts": True,
            },
            "all_reported_gates_passed": {
                "contact_gate": True,
                "torque_gate": True,
                "unsupported_gate": True,
                "truth_xy_evaluation_gate": True,
                "zero_forbidden_contacts": True,
            },
        },
        "per_trial": per_trial,
        "claims": {
            "independent_multi_position_visual_xy_pick_evidence": True,
            "same_condition_repeatability_claimed": False,
            "full_6d_vision_claimed": False,
            "arbitrary_pose_claimed": False,
            "production_control_authorized": False,
            "collision_planning_claimed": False,
            "full_end_to_end_assembly_claimed": False,
            "rl_training_ready_claimed": False,
            "rl_formal_readiness_gate_closed": False,
        },
        "rl_readiness": {
            "classification": "VALID_LIMITED_EVIDENCE_CANDIDATE",
            "candidate_key": "four_position_visual_xy_physical_pick",
            "formal_gate_closed": False,
            "why_limited": [
                "four deterministic trials at different XY positions are not a same-condition repeatability campaign",
                "orientation remains registered nominal and visually observed yaw is absent",
                "the partial PoseProvider and adapter are not production-control authorized",
                "the probe stops after lift/hold and does not demonstrate insertion, twist, or return Home",
                "collision planning is not claimed",
            ],
        },
        "provenance": {
            "request": dict(request_binding),
            "aggregator_source": dict(aggregator_binding),
            "source_runs": source_runs,
            "same_d38999_asset_sha256": verified[0]["asset_sha256"],
            "original_console_logs_archived_byte_for_byte": True,
        },
    }
    # Refuse to emit non-standard NaN/Infinity values.
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def aggregate_evidence(request_path: Path, repository: Path, output_dir: Path) -> dict[str, Any]:
    """Validate four source runs, archive logs and atomically write evidence."""

    repository = repository.resolve()
    request_path = request_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    _, verified = _parse_request(request_path, repository)
    aggregator_path = (repository / AGGREGATOR_SOURCE_PATH).resolve()
    if not aggregator_path.is_file():
        raise FileNotFoundError(aggregator_path)
    request_binding = _binding_for(
        request_path,
        REQUEST_SCHEMA_VERSION,
        shown_path=_repository_relative(request_path, repository),
    )
    aggregator_binding = _binding_for(
        aggregator_path,
        "python_source_v1",
        shown_path=AGGREGATOR_SOURCE_PATH,
    )

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)
    ) as temporary:
        staging = Path(temporary)
        logs_dir = staging / "logs"
        logs_dir.mkdir()
        archived_log_bindings: dict[str, dict[str, Any]] = {}
        for item in verified:
            archive_path = logs_dir / f"{item['run_id']}.log"
            shutil.copyfile(item["paths"]["console_log"], archive_path)
            source = item["bindings"]["console_log"]
            archived = _binding_for(
                archive_path,
                SOURCE_LOG_SCHEMA_VERSION,
                shown_path=f"logs/{archive_path.name}",
            )
            if (
                archived["sha256"] != source["sha256"]
                or archived["size_bytes"] != source["size_bytes"]
            ):
                raise EvidenceError("archived console log differs from source")
            archived_log_bindings[item["run_id"]] = archived

        evidence = _build_evidence(
            verified,
            request_binding=request_binding,
            aggregator_binding=aggregator_binding,
            archived_log_bindings=archived_log_bindings,
        )
        report_path = staging / "report.json"
        report_path.write_text(
            json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_binding = _binding_for(
            report_path, EVIDENCE_SCHEMA_VERSION, shown_path="report.json"
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "HASH_SIZE_SCHEMA_BOUND",
            "artifact_report": report_binding,
            "archived_console_logs": [
                {"run_id": run_id, "binding": archived_log_bindings[run_id]}
                for run_id in sorted(archived_log_bindings)
            ],
            "request": request_binding,
            "aggregator_source": aggregator_binding,
            "source_runs": evidence["provenance"]["source_runs"],
            "audit_without_original_tmp_logs_supported": True,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_dir)
    return evidence


def _verify_artifact_binding(
    artifact_dir: Path, value: Any, label: str, expected_schema: str
) -> Path:
    binding = _mapping(value, label)
    _exact_keys(binding, _BINDING_KEYS, label)
    if binding.get("schema_version") != expected_schema:
        raise EvidenceError(f"{label}.schema_version mismatch")
    relative = Path(_text(binding.get("path"), f"{label}.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise EvidenceError(f"{label}.path must remain artifact-relative")
    path = (artifact_dir / relative).resolve()
    try:
        path.relative_to(artifact_dir)
    except ValueError as error:
        raise EvidenceError(f"{label}.path escaped artifact directory") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != _integer(binding.get("size_bytes"), f"{label}.size_bytes", minimum=1):
        raise EvidenceError(f"{label} byte-size mismatch")
    if sha256_file(path) != _sha256(binding.get("sha256"), f"{label}.sha256"):
        raise EvidenceError(f"{label} SHA-256 mismatch")
    return path


def audit_artifact(artifact_dir: Path, repository: Path) -> dict[str, Any]:
    """Re-audit a generated bundle without depending on the original /tmp logs."""

    artifact_dir = artifact_dir.resolve()
    repository = repository.resolve()
    manifest_path = artifact_dir / "manifest.json"
    manifest = _load_json(manifest_path, "manifest")
    expected_manifest_keys = {
        "schema_version",
        "status",
        "artifact_report",
        "archived_console_logs",
        "request",
        "aggregator_source",
        "source_runs",
        "audit_without_original_tmp_logs_supported",
    }
    _exact_keys(manifest, expected_manifest_keys, "manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise EvidenceError("manifest.schema_version mismatch")
    if manifest.get("status") != "HASH_SIZE_SCHEMA_BOUND":
        raise EvidenceError("manifest.status mismatch")
    _require_bool(
        manifest.get("audit_without_original_tmp_logs_supported"),
        True,
        "manifest.audit_without_original_tmp_logs_supported",
    )
    report_path = _verify_artifact_binding(
        artifact_dir,
        manifest.get("artifact_report"),
        "manifest.artifact_report",
        EVIDENCE_SCHEMA_VERSION,
    )
    report = _load_json(report_path, "artifact report")
    if report.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceError("artifact report schema mismatch")
    if report.get("status") != "PASS_LIMITED_MULTI_POSITION_VISUAL_XY_PICK_EVIDENCE":
        raise EvidenceError("artifact report status mismatch")
    summary = _mapping(report.get("summary"), "artifact report.summary")
    if summary.get("pass_fraction") != "4/4" or summary.get("all_four_gpu_trials_passed") is not True:
        raise EvidenceError("artifact report is not 4/4 PASS")

    archived = manifest.get("archived_console_logs")
    if not isinstance(archived, list) or len(archived) != EXPECTED_RUN_COUNT:
        raise EvidenceError("manifest must bind four archived console logs")
    archived_by_run: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(archived):
        record = _mapping(item, f"manifest.archived_console_logs[{index}]")
        _exact_keys(record, {"run_id", "binding"}, f"manifest.archived_console_logs[{index}]")
        run_id = _text(record.get("run_id"), f"manifest.archived_console_logs[{index}].run_id")
        if run_id in archived_by_run:
            raise EvidenceError("duplicate archived console-log run_id")
        _verify_artifact_binding(
            artifact_dir,
            record.get("binding"),
            f"manifest.archived_console_logs[{index}].binding",
            SOURCE_LOG_SCHEMA_VERSION,
        )
        archived_by_run[run_id] = _mapping(record.get("binding"), "archived binding")

    # Recheck repository-owned immutable sources.  The original /tmp logs are
    # intentionally replaced by their byte-identical archived copies here.
    source_runs = manifest.get("source_runs")
    if source_runs != report.get("provenance", {}).get("source_runs"):
        raise EvidenceError("manifest source_runs differ from artifact report")
    if not isinstance(source_runs, list) or len(source_runs) != EXPECTED_RUN_COUNT:
        raise EvidenceError("manifest source_runs must contain four runs")
    per_trial = report.get("per_trial")
    if not isinstance(per_trial, list) or len(per_trial) != EXPECTED_RUN_COUNT:
        raise EvidenceError("artifact report must contain four per-trial records")
    per_trial_by_run: dict[str, Mapping[str, Any]] = {}
    for index, raw_trial in enumerate(per_trial):
        trial_record = _mapping(raw_trial, f"per_trial[{index}]")
        trial_run_id = _text(
            trial_record.get("run_id"), f"per_trial[{index}].run_id"
        )
        if trial_run_id in per_trial_by_run:
            raise EvidenceError("artifact report has duplicate per-trial run_id")
        per_trial_by_run[trial_run_id] = trial_record
    for index, item in enumerate(source_runs):
        record = _mapping(item, f"manifest.source_runs[{index}]")
        _exact_keys(record, {"run_id", "bindings"}, f"manifest.source_runs[{index}]")
        bindings = _mapping(record.get("bindings"), f"manifest.source_runs[{index}].bindings")
        _exact_keys(
            bindings,
            {"report", "console_log", "config", "cpu_plan", "archived_console_log"},
            f"manifest.source_runs[{index}].bindings",
        )
        repository_paths: dict[str, Path] = {}
        for name, schema in (
            ("report", SOURCE_REPORT_SCHEMA_VERSION),
            ("config", SOURCE_CONFIG_SCHEMA_VERSION),
            ("cpu_plan", SOURCE_PLAN_SCHEMA_VERSION),
        ):
            _, repository_paths[name] = _verify_binding(
                bindings[name],
                repository,
                f"manifest.source_runs[{index}].{name}",
                expected_schema=schema,
                repository_only=True,
            )
        run_id = record["run_id"]
        if run_id not in archived_by_run or bindings["archived_console_log"] != archived_by_run[run_id]:
            raise EvidenceError("source run archived-log binding mismatch")
        source_log = _mapping(
            bindings["console_log"],
            f"manifest.source_runs[{index}].console_log",
        )
        archived_log = archived_by_run[run_id]
        for key in ("sha256", "size_bytes", "schema_version"):
            if source_log.get(key) != archived_log.get(key):
                raise EvidenceError(
                    "archived console log is not byte-identical to its source binding"
                )

        trial = per_trial_by_run.get(run_id)
        if trial is None:
            raise EvidenceError("source run has no per-trial evidence record")
        source_report = _load_json(
            repository_paths["report"],
            f"manifest.source_runs[{index}].report",
        )
        _validate_config(
            repository_paths["config"],
            source_report,
            f"manifest.source_runs[{index}].config",
        )
        source_plan = _load_json(
            repository_paths["cpu_plan"],
            f"manifest.source_runs[{index}].cpu_plan",
        )
        if source_plan != source_report.get("cpu_plan"):
            raise EvidenceError("source CPU plan differs from source report")
        _validate_report(
            source_report,
            expected_trial_id=_text(
                trial.get("trial_id"), f"per_trial[{run_id}].trial_id"
            ),
            expected_loose_xy=_xy(
                trial.get("authored_loose_xy_m"),
                f"per_trial[{run_id}].authored_loose_xy_m",
            ),
            expected_fixed_xy=_xy(
                trial.get("authored_fixed_xy_m"),
                f"per_trial[{run_id}].authored_fixed_xy_m",
            ),
            cpu_plan_path=repository_paths["cpu_plan"],
        )
        archived_path = _verify_artifact_binding(
            artifact_dir,
            archived_log,
            f"manifest.source_runs[{index}].archived_console_log",
            SOURCE_LOG_SCHEMA_VERSION,
        )
        _validate_console_log(
            archived_path,
            source_report,
            f"manifest.source_runs[{index}].archived_console_log",
        )

    for name, schema, repository_only in (
        ("request", REQUEST_SCHEMA_VERSION, False),
        ("aggregator_source", "python_source_v1", True),
    ):
        _verify_binding(
            manifest[name],
            repository,
            f"manifest.{name}",
            expected_schema=schema,
            repository_only=repository_only,
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "PASS",
        "pass_fraction": "4/4",
        "artifact_report_sha256": manifest["artifact_report"]["sha256"],
        "archived_console_log_count": EXPECTED_RUN_COUNT,
        "original_tmp_logs_required": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--request", type=Path, default=Path(DEFAULT_REQUEST_PATH)
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve()
    try:
        if (args.output_dir is None) == (args.audit is None):
            raise EvidenceError("choose exactly one of --output-dir or --audit")
        if args.audit is not None:
            result = audit_artifact(args.audit, repository)
        else:
            request = args.request
            if not request.is_absolute():
                request = repository / request
            result = aggregate_evidence(request, repository, args.output_dir)
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    except (EvidenceError, FileExistsError, FileNotFoundError, OSError) as error:
        print(f"visual XY pick evidence rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATOR_SOURCE_PATH",
    "DEFAULT_REQUEST_PATH",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceError",
    "EXPECTED_RUN_COUNT",
    "MANIFEST_SCHEMA_VERSION",
    "PASS_BANNER",
    "REQUEST_SCHEMA_VERSION",
    "SOURCE_CONFIG_SCHEMA_VERSION",
    "SOURCE_LOG_SCHEMA_VERSION",
    "SOURCE_PLAN_SCHEMA_VERSION",
    "SOURCE_REPORT_SCHEMA_VERSION",
    "aggregate_evidence",
    "audit_artifact",
    "main",
    "sha256_file",
]
