'''Pure posthoc contact audit evaluation and capture/skip comparison.

The runner records RAW classified contact evidence at four fixed controller
points plus identical markers in both skip and capture modes.  This module
is the offline authority that proves: (a) the capture episode's control and
sensor traces are bit-identical to the skip episode's, so the mid-episode
contact query had no side effect; (b) the four markers sit at exactly the
frozen controller points in both episodes; (c) every per-step record obeys
the frozen schema; (d) the recorded contact evidence satisfies the frozen
per-point gates including exact material-count consistency.

Pure module: no Omni/pxr/isaacsim imports, no file IO, no RNG, and it never
starts any simulation.
'''

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

AUDIT_POINTS = (
    "pre_approach",
    "contact_confirmed",
    "soft_hold_complete",
    "release_confirmed",
)

FINGERS = ("f1", "f2", "f3")
TORQUE_CHANNEL_NAMES = ("f1j2", "f2j1", "f3j2")

DECISION_FIELDS = (
    "global_step",
    "phase",
    "state",
    "selected_target_rad",
    "selected_stiffness_scale",
    "soft_hold_step",
    "release_step",
    "failed",
    "failure_reason",
    "detector_test_passed",
    "transition_events",
    "hand_target_rad",
    "other_fingers_open_target_invariant",
    "release_conditions",
)

SENSOR_FIELDS = (
    "selected_q_rad",
    "selected_qd_rad_s",
    "finger_root_torque_proxy_nm",
    "hand_q_rad",
    "hand_qd_rad_s",
    "observation",
    "controller_evidence",
    "wrist_wrench_raw_sensor_frame",
    "wrist_wrench_canonical",
    "wrist_wrench_empty_baseline_compensated",
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

MARKER_FIELDS = (
    "point",
    "global_step",
    "selected_finger",
    "controller_state",
    "soft_hold_step",
    "release_step",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def project_record(record: Mapping[str, Any], fields: Sequence[str]) -> dict:
    projected = {}
    for name in fields:
        value = record.get(name)
        if isinstance(value, (list, dict, bool, int, float, str)):
            projected[name] = value
        elif value is None:
            projected[name] = None
        else:
            projected[name] = repr(value)
    return projected


def project_trace(steps: Sequence[Mapping[str, Any]], fields: Sequence[str]):
    return [project_record(record, fields) for record in steps]


def _first_diff(first: Any, second: Any, path: tuple = ()) -> tuple | None:
    if isinstance(first, dict) and isinstance(second, dict):
        for key in sorted(set(first) | set(second)):
            if key not in first or key not in second:
                return path + (key,), first.get(key), second.get(key)
            found = _first_diff(first[key], second[key], path + (key,))
            if found is not None:
                return found
        return None
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            return path, first, second
        for index, (left, right) in enumerate(zip(first, second)):
            found = _first_diff(left, right, path + (index,))
            if found is not None:
                return found
        return None
    if first != second:
        return path, first, second
    return None


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_finite_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(_is_finite_number(item) for item in value)
    )


def material_evidence_snapshot(snapshot: Mapping[str, Any]) -> dict:
    evidence = snapshot.get("material_evidence") or {}
    resolved = int(evidence.get("resolved_records", 0) or 0)
    unresolved = int(evidence.get("unresolved_records", 0) or 0)
    if resolved > 0 and unresolved > 0:
        mode = "partial_unresolved_fail_closed"
    elif resolved > 0:
        mode = "fully_resolved"
    elif unresolved > 0:
        mode = "fully_unresolved_binding_fallback"
    else:
        mode = "no_material_records"
    return {
        "available": bool(evidence.get("available", False)),
        "mode": mode,
        "resolved_records": resolved,
        "unresolved_records": unresolved,
        "grip_grip_records": int(
            evidence.get("grip_grip_records", 0) or 0
        ),
        "resolved_non_grip_records": int(
            evidence.get("resolved_non_grip_records", 0) or 0
        ),
    }


def evaluate_audit_point(
    point: str,
    snapshot: Mapping[str, Any],
    selected_finger: str,
    *,
    binding_identity_ok: bool = False,
) -> dict[str, Any]:
    if point not in AUDIT_POINTS:
        raise ValueError(f"unknown audit point {point!r}")
    if selected_finger not in FINGERS:
        raise ValueError(f"unknown selected finger {selected_finger!r}")
    if not isinstance(snapshot, Mapping):
        result = {
            "passed": False,
            "gates": {
                "selected_finger_contact": False,
                "other_fingers_zero": False,
                "unexpected_robot_link_zero": False,
                "plug_table_present": False,
                "plug_table_required": False,
                "material_evidence_consistent": False,
            },
            "material_evidence": material_evidence_snapshot({}),
            "snapshot_missing": True,
        }
        result.update({"point": point, "selected_finger": selected_finger})
        return result
    groups = snapshot.get("finger_body_group_records") or {}
    selected_counts = groups.get(selected_finger) or {}
    selected_body = int(selected_counts.get("body", 0) or 0)
    selected_nut = int(selected_counts.get("nut", 0) or 0)
    selected_total = selected_body + selected_nut
    other_total = 0
    for finger in FINGERS:
        if finger == selected_finger:
            continue
        counts = groups.get(finger) or {}
        other_total += int(counts.get("body", 0) or 0) + int(
            counts.get("nut", 0) or 0
        )
    unexpected = int(
        snapshot.get("unexpected_robot_link_records", 0) or 0
    )
    plug_table = int(snapshot.get("plug_table_records", 0) or 0)
    material = material_evidence_snapshot(snapshot)
    if point in ("contact_confirmed", "soft_hold_complete"):
        selected_contact_ok = selected_total >= 1
        material_ok = False
        if material["mode"] == "fully_resolved":
            material_ok = bool(
                material["available"] is True
                and material["unresolved_records"] == 0
                and material["resolved_records"]
                == material["grip_grip_records"]
                + material["resolved_non_grip_records"]
                and material["resolved_records"] == selected_total
                and material["resolved_non_grip_records"] == 0
                and material["grip_grip_records"] == selected_total
            )
        elif material["mode"] == "fully_unresolved_binding_fallback":
            material_ok = bool(
                material["available"] is False
                and material["resolved_records"] == 0
                and material["grip_grip_records"] == 0
                and material["resolved_non_grip_records"] == 0
                and material["unresolved_records"] == selected_total
                and binding_identity_ok
            )
    else:
        selected_contact_ok = selected_total == 0
        material_ok = True
    plug_table_required = point in ("pre_approach", "release_confirmed")
    plug_table_ok = plug_table > 0 if plug_table_required else True
    passed = bool(
        selected_contact_ok
        and other_total == 0
        and unexpected == 0
        and plug_table_ok
        and material_ok
    )
    return {
        "point": point,
        "selected_finger": selected_finger,
        "passed": passed,
        "gates": {
            "selected_finger_contact": selected_contact_ok,
            "other_fingers_zero": other_total == 0,
            "unexpected_robot_link_zero": unexpected == 0,
            "plug_table_present": plug_table > 0,
            "plug_table_required": plug_table_required,
            "material_evidence_consistent": material_ok,
        },
        "counts": {
            "selected_body_records": selected_body,
            "selected_nut_records": selected_nut,
            "other_finger_total_records": other_total,
            "unexpected_robot_link_records": unexpected,
            "plug_table_records": plug_table,
        },
        "material_evidence": material,
        "binding_identity_ok": binding_identity_ok,
    }


PROVENANCE_KEYS = (
    "seed",
    "finger",
    "payload_sha256",
    "physical_grasp_config_sha256",
    "pick_config_sha256",
    "tabletop_scene_config_sha256",
    "runner_sha256",
    "wrapper_sha256",
    "finger_contact_detector_sha256",
    "single_finger_contact_test_sha256",
    "single_finger_posthoc_audit_sha256",
    "single_finger_posthoc_audit_compare_sha256",
)

PROVENANCE_DIGEST_KEYS = (
    "payload_sha256",
    "physical_grasp_config_sha256",
    "pick_config_sha256",
    "tabletop_scene_config_sha256",
    "runner_sha256",
    "wrapper_sha256",
    "finger_contact_detector_sha256",
    "single_finger_contact_test_sha256",
    "single_finger_posthoc_audit_sha256",
    "single_finger_posthoc_audit_compare_sha256",
)


def _validate_provenance(provenance: Mapping[str, Any]) -> list[str]:
    problems = []
    for key in PROVENANCE_KEYS:
        if key not in provenance:
            problems.append(f"missing_{key}")
    if "audit_mode" not in provenance:
        problems.append("missing_audit_mode")
    seed = provenance.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.append("seed_not_integer")
    if provenance.get("finger") not in FINGERS:
        problems.append("finger_invalid")
    for key in PROVENANCE_DIGEST_KEYS:
        value = provenance.get(key)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            problems.append(f"digest_invalid_{key}")
    return problems


def verify_provenance_contract(
    capture_report: Mapping[str, Any],
    skip_report: Mapping[str, Any],
) -> list[str]:
    capture_provenance = capture_report.get("provenance")
    skip_provenance = skip_report.get("provenance")
    if not isinstance(capture_provenance, Mapping) or not isinstance(
        skip_provenance, Mapping
    ):
        return ["provenance_missing"]
    problems = _validate_provenance(capture_provenance)
    problems.extend(_validate_provenance(skip_provenance))
    if problems:
        return problems
    for key in PROVENANCE_KEYS:
        if capture_provenance.get(key) != skip_provenance.get(key):
            problems.append(f"mismatch_{key}")
    if capture_provenance.get("audit_mode") != "capture":
        problems.append("capture_audit_mode")
    if skip_provenance.get("audit_mode") != "skip":
        problems.append("skip_audit_mode")
    for report, label in (
        (capture_report, "capture"),
        (skip_report, "skip"),
    ):
        provenance = report.get("provenance") or {}
        if report.get("seed") != provenance.get("seed"):
            problems.append(f"{label}_report_seed_mismatch")
        if (report.get("single_finger") or {}).get(
            "selected_finger"
        ) != provenance.get("finger"):
            problems.append(f"{label}_report_finger_mismatch")
        if report.get("physical_grasp_method") != "single-finger":
            problems.append(f"{label}_report_method_mismatch")
        if report.get("formal_lift_mode") != "zero-lift-hold":
            problems.append(f"{label}_report_lift_mode_mismatch")
        if not isinstance(report.get("gui"), bool):
            problems.append(f"{label}_gui_not_bool")
    if capture_report.get("gui") != skip_report.get("gui"):
        problems.append("gui_flag_mismatch")
    return problems


def _marker_schema_problems(
    marker: Mapping[str, Any],
    *,
    expected_finger: str,
    expect_capture: bool,
    label: str,
) -> list[str]:
    problems = []
    extra_keys = set(marker) - set(MARKER_FIELDS) - {"snapshot", "error"}
    if extra_keys:
        problems.append(f"{label}_marker_extra_keys")
    for name in MARKER_FIELDS:
        if name not in marker:
            problems.append(f"{label}_marker_missing_{name}")
    if not isinstance(marker.get("point"), str):
        problems.append(f"{label}_marker_point_type")
    if isinstance(marker.get("global_step"), bool) or not isinstance(
        marker.get("global_step"), int
    ):
        problems.append(f"{label}_marker_global_step_type")
    if marker.get("selected_finger") != expected_finger:
        problems.append(f"{label}_marker_wrong_finger")
    if not isinstance(marker.get("controller_state"), str):
        problems.append(f"{label}_marker_state_type")
    for name in ("soft_hold_step", "release_step"):
        if isinstance(marker.get(name), bool) or not isinstance(
            marker.get(name), int
        ):
            problems.append(f"{label}_marker_{name}_type")
    if expect_capture:
        snapshot = marker.get("snapshot")
        error = marker.get("error")
        if isinstance(snapshot, Mapping):
            if error is not None:
                problems.append(f"{label}_marker_snapshot_with_error")
        else:
            if not isinstance(error, str):
                problems.append(f"{label}_marker_snapshot_missing")
    else:
        if marker.get("snapshot") is not None:
            problems.append(f"{label}_marker_snapshot_not_none")
        if "error" in marker:
            problems.append(f"{label}_marker_unexpected_error")
    return problems


def _marker_value_and_trace_problems(
    marker: Mapping[str, Any],
    point: str,
    *,
    control_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Shared per-point marker value semantics and trace alignment.

    Extracted verbatim from the frozen verify_markers checks so the
    capture-only episode helper and the capture/skip comparator validate
    the exact same value semantics: point identity, per-point controller
    state/hold/release frozen values, trace alignment on state/hold/
    release/selected_finger, and pre-approach strictly before the first
    control record.
    """
    problems: list[str] = []
    by_step = {
        record.get("global_step"): record for record in control_records
    }
    if marker.get("point") != point:
        problems.append(f"marker_point_mismatch_{point}")
    controller_state = marker.get("controller_state")
    soft_hold_step = marker.get("soft_hold_step")
    release_step = marker.get("release_step")
    global_step = marker.get("global_step")
    record = by_step.get(global_step)
    if point == "pre_approach":
        if (
            controller_state != "APPROACH"
            or soft_hold_step != 0
            or release_step != 0
        ):
            problems.append("marker_pre_approach_state")
        if control_records and global_step is not None:
            if global_step >= control_records[0].get("global_step"):
                problems.append("marker_pre_approach_not_first")
    else:
        if record is None:
            problems.append(f"marker_{point}_trace_misaligned")
        else:
            if record.get("state") != controller_state:
                problems.append(f"marker_{point}_trace_state_misaligned")
            if record.get("soft_hold_step") != soft_hold_step:
                problems.append(
                    f"marker_{point}_trace_soft_hold_misaligned"
                )
            if record.get("release_step") != release_step:
                problems.append(
                    f"marker_{point}_trace_release_misaligned"
                )
            if record.get("selected_finger") != marker.get(
                "selected_finger"
            ):
                problems.append(f"marker_{point}_trace_finger_misaligned")
        if point == "contact_confirmed":
            if (
                controller_state != "SOFT_HOLD"
                or soft_hold_step != 0
                or release_step != 0
            ):
                problems.append("marker_contact_confirmed_state")
        elif point == "soft_hold_complete":
            if (
                controller_state != "SOFT_HOLD"
                or soft_hold_step != 24
                or release_step != 0
            ):
                problems.append("marker_soft_hold_complete_state")
        elif point == "release_confirmed":
            if (
                controller_state != "RELEASE_CONFIRMED"
                or soft_hold_step != 24
                or release_step <= 0
            ):
                problems.append("marker_release_confirmed_state")
    return problems


def verify_capture_episode_markers_classified(
    report: Mapping[str, Any],
    control_records: Sequence[Mapping[str, Any]],
    label: str = "capture",
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Strict capture-only marker validation with explicit exit categories.

    Returns (schema_problems, functional_problems, evidence):

    schema_problems (input/schema contract invalid, exit-2 semantics):
      marker key set inexact or a marker missing; extra keys or missing
      required fields; point/global_step/selected_finger/controller_state/
      soft_hold_step/release_step type violations; selected_finger not
      matching the episode; capture snapshot not a Mapping; snapshot with
      a non-empty error payload.

    functional_problems (physical/functional structure mismatch, exit-1
    semantics): point identity or per-point frozen state/hold/release value
    errors; marker vs control-trace misalignment on state/hold/release/
    selected_finger; pre-approach position; duplicate, reversed, or
    non-increasing marker global steps.

    The categories are returned by this helper so callers never classify
    by problem-string prefixes.  This helper shares the exact
    _marker_schema_problems and _marker_value_and_trace_problems checks
    used by verify_markers; the frozen capture/skip comparator itself is
    unchanged.
    """
    schema_problems: list[str] = []
    functional_problems: list[str] = []
    expected_finger = (report.get("single_finger") or {}).get(
        "selected_finger", ""
    )
    audit = report.get("posthoc_audit") or {}
    points = (
        audit.get("points") or {} if isinstance(audit, Mapping) else {}
    )
    if set(points.keys()) != set(AUDIT_POINTS):
        schema_problems.append(f"{label}_marker_set_not_exact")
    evidence: dict[str, Any] = {}
    marker_steps: list[int] = []
    for point in AUDIT_POINTS:
        marker = points.get(point)
        if not isinstance(marker, Mapping):
            schema_problems.append(f"{label}_missing_marker_{point}")
            continue
        marker_schema_problems = _marker_schema_problems(
            marker,
            expected_finger=expected_finger,
            expect_capture=True,
            label=f"{label}_{point}",
        )
        schema_problems.extend(marker_schema_problems)
        snapshot_mapping = isinstance(marker.get("snapshot"), Mapping)
        if not snapshot_mapping:
            schema_problems.append(
                f"{label}_{point}_marker_snapshot_not_mapping"
            )
        evidence[point] = dict(marker)
        if marker_schema_problems or not snapshot_mapping:
            # Fail closed inside the helper: a schema-invalid marker must
            # produce its precise schema problems and never feed a value
            # checker that could raise on the invalid field.
            continue
        functional_problems.extend(
            _marker_value_and_trace_problems(
                marker, point, control_records=control_records
            )
        )
        global_step = marker.get("global_step")
        if isinstance(global_step, int) and not isinstance(global_step, bool):
            marker_steps.append(global_step)
    if (
        marker_steps
        and (
            marker_steps != sorted(marker_steps)
            or len(set(marker_steps)) != len(marker_steps)
        )
    ):
        functional_problems.append("marker_global_steps_not_increasing")
    return schema_problems, functional_problems, evidence


def verify_capture_episode_markers(
    report: Mapping[str, Any],
    control_records: Sequence[Mapping[str, Any]],
    label: str = "capture",
) -> tuple[list[str], dict[str, Any]]:
    """Compatibility entry: schema then functional problems, plus evidence.

    Kept for callers that only need one flat problem list.  Callers that
    must distinguish exit-2 schema violations from exit-1 structure
    mismatches should use verify_capture_episode_markers_classified
    instead; do not classify by problem-string prefixes.
    """
    schema_problems, functional_problems, evidence = (
        verify_capture_episode_markers_classified(
            report, control_records, label
        )
    )
    return schema_problems + functional_problems, evidence


def verify_markers(
    capture_report: Mapping[str, Any],
    skip_report: Mapping[str, Any],
    control_records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    problems = []
    expected_finger = capture_report.get("single_finger", {}).get(
        "selected_finger", ""
    )
    capture_audit = capture_report.get("posthoc_audit") or {}
    skip_audit = skip_report.get("posthoc_audit") or {}
    capture_points = (
        capture_audit.get("points") or {}
        if isinstance(capture_audit, Mapping)
        else {}
    )
    skip_points = (
        skip_audit.get("points") or {}
        if isinstance(skip_audit, Mapping)
        else {}
    )
    if set(capture_points.keys()) != set(AUDIT_POINTS):
        problems.append("capture_marker_set_not_exact")
    if set(skip_points.keys()) != set(AUDIT_POINTS):
        problems.append("skip_marker_set_not_exact")
    evidence = {}
    for point in AUDIT_POINTS:
        capture_marker = capture_points.get(point)
        skip_marker = skip_points.get(point)
        if not isinstance(capture_marker, Mapping):
            problems.append(f"capture_missing_marker_{point}")
            continue
        if not isinstance(skip_marker, Mapping):
            problems.append(f"skip_missing_marker_{point}")
            continue
        problems.extend(
            _marker_schema_problems(
                capture_marker,
                expected_finger=expected_finger,
                expect_capture=True,
                label=f"capture_{point}",
            )
        )
        problems.extend(
            _marker_schema_problems(
                skip_marker,
                expected_finger=expected_finger,
                expect_capture=False,
                label=f"skip_{point}",
            )
        )
        evidence[point] = {
            "capture_marker": dict(capture_marker),
            "skip_marker": dict(skip_marker),
        }
        capture_tuple = tuple(
            capture_marker.get(name) for name in MARKER_FIELDS
        )
        skip_tuple = tuple(skip_marker.get(name) for name in MARKER_FIELDS)
        if capture_tuple != skip_tuple:
            problems.append(f"marker_mismatch_{point}")
            continue
        problems.extend(
            _marker_value_and_trace_problems(
                capture_marker, point, control_records=control_records
            )
        )
    steps = [
        capture_points[point].get("global_step")
        for point in AUDIT_POINTS
        if isinstance(capture_points.get(point), Mapping)
        and isinstance(capture_points[point].get("global_step"), int)
    ]
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        problems.append("marker_global_steps_not_increasing")
    return problems, evidence


def verify_step_schema(
    record: Mapping[str, Any],
    selected_finger: str,
    label: str,
) -> list[str]:
    problems = []
    if isinstance(record.get("global_step"), bool) or not isinstance(
        record.get("global_step"), int
    ):
        problems.append(f"{label}_global_step_type")
    if not isinstance(record.get("state"), str):
        problems.append(f"{label}_state_type")
    if record.get("selected_finger") != selected_finger:
        problems.append(f"{label}_selected_finger_mismatch")
    for name in (
        "selected_hand_local_index",
        "selected_robot_dof_index",
        "soft_hold_step",
        "release_step",
    ):
        if isinstance(record.get(name), bool) or not isinstance(
            record.get(name), int
        ):
            problems.append(f"{label}_{name}_type")
    for name in (
        "selected_target_rad",
        "selected_stiffness_scale",
        "selected_q_rad",
        "selected_qd_rad_s",
    ):
        if not _is_finite_number(record.get(name)):
            problems.append(f"{label}_{name}_invalid")
    torque = record.get("finger_root_torque_proxy_nm")
    if not isinstance(torque, Mapping) or set(torque) != set(FINGERS):
        problems.append(f"{label}_torque_channels_wrong")
    elif not all(
        _is_finite_number(torque.get(name)) for name in FINGERS
    ):
        problems.append(f"{label}_torque_values_invalid")
    for name in ("hand_q_rad", "hand_qd_rad_s", "hand_target_rad"):
        if not _is_finite_vector(record.get(name), 4):
            problems.append(f"{label}_{name}_invalid")
    for name in (
        "wrist_wrench_raw_sensor_frame",
        "wrist_wrench_canonical",
        "wrist_wrench_empty_baseline_compensated",
    ):
        if not _is_finite_vector(record.get(name), 6):
            problems.append(f"{label}_{name}_invalid")
    if not isinstance(record.get("observation"), Mapping):
        problems.append(f"{label}_observation_missing")
    if not isinstance(record.get("controller_evidence"), Mapping):
        problems.append(f"{label}_controller_evidence_missing")
    return problems


def verify_report_terminal_schema(report: Mapping[str, Any]) -> list[str]:
    problems = []
    single = report.get("single_finger") or {}
    torque = single.get("maximum_post_tare_absolute_delta_by_channel_nm")
    if not isinstance(torque, Mapping) or set(torque) != set(
        TORQUE_CHANNEL_NAMES
    ):
        problems.append("report_torque_channels_wrong")
    elif not all(
        _is_finite_number(torque.get(name)) for name in TORQUE_CHANNEL_NAMES
    ):
        problems.append("report_torque_values_invalid")
    if not _is_finite_number(
        single.get("maximum_post_tare_absolute_delta_nm")
    ):
        problems.append("report_torque_overall_invalid")
    for name in ("release_step", "soft_hold_step"):
        if isinstance(single.get(name), bool) or not isinstance(
            single.get(name), int
        ):
            problems.append(f"report_{name}_invalid")
    if not isinstance(single.get("release_conditions"), Mapping):
        problems.append("report_release_conditions_missing")
    if not isinstance(single.get("transition_events"), list):
        problems.append("report_transition_events_missing")
    wrist = report.get("virtual_wrist_ft_monitor") or {}
    if not isinstance(wrist.get("status"), str):
        problems.append("report_wrist_status_missing")
    last_step = (wrist.get("last_sample") or {}).get("global_step")
    if isinstance(last_step, bool) or not isinstance(last_step, int):
        problems.append("report_wrist_last_step_missing")
    return problems


def verify_episode_safety_contract(
    report: Mapping[str, Any], expect_read_contact: bool
) -> list[str]:
    problems = []
    checks = (
        ("process_exit_code", lambda v: v == 3, "exit_code_not_3"),
        ("passed", lambda v: v is False, "passed_not_false"),
        (
            "grasp_success_claimed",
            lambda v: v is False,
            "grasp_success_claimed_not_false",
        ),
        (
            "control_reads_object_truth",
            lambda v: v is False,
            "control_reads_object_truth_not_false",
        ),
        (
            "control_reads_contact_report",
            lambda v: v is False,
            "control_reads_contact_report_not_false",
        ),
        (
            "posthoc_audit_consumed_by_control",
            lambda v: v is False,
            "posthoc_audit_consumed_by_control_not_false",
        ),
        (
            "posthoc_audit_reads_contact_report",
            lambda v: v is expect_read_contact,
            "posthoc_audit_reads_contact_report_wrong",
        ),
        (
            "formal_truth_firewall_enabled",
            lambda v: v is True,
            "formal_truth_firewall_not_enabled",
        ),
        (
            "object_pose_writes_after_start",
            lambda v: v == 0,
            "object_pose_writes_not_zero",
        ),
    )
    for key, predicate, label in checks:
        value = report.get(key)
        if not predicate(value):
            problems.append(label)
    single = report.get("single_finger") or {}
    if single.get("posthoc_contact_audit_passed") is not None:
        problems.append("posthoc_contact_audit_passed_not_null")
    if single.get("single_finger_validation_passed") is not None:
        problems.append("single_finger_validation_passed_not_null")
    if single.get("detector_test_passed") is not True:
        problems.append("controller_not_complete")
    contract = report.get("physical_grasp_contract") or {}
    if contract.get("post_grasp_stabilization_proxy_enabled") is not False:
        problems.append("stabilization_proxy_enabled")
    authoring = report.get("realized_usd_authoring") or {}
    if authoring.get("usd_authoring_verified") is not True:
        problems.append("usd_authoring_not_verified")
    binding = authoring.get("material_binding_identity") or {}
    if (
        binding.get("all_bindings_ok") is not True
        or binding.get("finger_proxy_count") != 8
        or binding.get("plug_collider_count") != 45
        or binding.get("finger_proxy_all_grip") is not True
        or binding.get("plug_collider_all_grip") is not True
    ):
        problems.append("material_binding_identity_contract_failed")
    expected_mode = "capture" if expect_read_contact else "skip"
    audit = report.get("posthoc_audit")
    if not isinstance(audit, Mapping):
        problems.append("posthoc_audit_missing")
    else:
        if audit.get("mode") != expected_mode:
            problems.append("posthoc_audit_mode_wrong")
        if audit.get("read_contact_report") is not expect_read_contact:
            problems.append("posthoc_audit_read_flag_wrong")
        if audit.get("consumed_by_control") is not False:
            problems.append("posthoc_audit_consumed_wrong")
        if not isinstance(audit.get("points"), Mapping):
            problems.append("posthoc_audit_points_missing")
    return problems


def _report_terminal_projection(report: Mapping[str, Any]) -> dict:
    single = report.get("single_finger") or {}
    wrist = report.get("virtual_wrist_ft_monitor") or {}
    return {
        "torque_by_channel": single.get(
            "maximum_post_tare_absolute_delta_by_channel_nm"
        ),
        "torque_overall": single.get("maximum_post_tare_absolute_delta_nm"),
        "final_release_conditions": single.get("release_conditions"),
        "final_release_step": single.get("release_step"),
        "final_soft_hold_step": single.get("soft_hold_step"),
        "transition_events": single.get("transition_events"),
        "wrist_status": wrist.get("status"),
        "wrist_last_sample_step": (wrist.get("last_sample") or {}).get(
            "global_step"
        ),
    }


def _format_diff(diff):
    if diff is None:
        return None
    path, capture_value, skip_value = diff
    return {
        "path": [str(part) for part in path],
        "capture_value": capture_value,
        "skip_value": skip_value,
    }


def compare_episodes(
    capture_report: Mapping[str, Any],
    capture_steps: Sequence[Mapping[str, Any]],
    skip_report: Mapping[str, Any],
    skip_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    problems = verify_provenance_contract(capture_report, skip_report)
    if problems:
        raise ValueError(
            "input contract mismatch: " + ", ".join(sorted(problems))
        )
    capture_control = [
        record
        for record in capture_steps
        if record.get("phase") == "single_finger_contact_characterization"
    ]
    skip_control = [
        record
        for record in skip_steps
        if record.get("phase") == "single_finger_contact_characterization"
    ]
    selected_finger = capture_report.get("single_finger", {}).get(
        "selected_finger", ""
    )
    marker_problems, marker_evidence = verify_markers(
        capture_report, skip_report, capture_control
    )
    capture_contract = verify_episode_safety_contract(
        capture_report, expect_read_contact=True
    )
    skip_contract = verify_episode_safety_contract(
        skip_report, expect_read_contact=False
    )
    schema_problems = []
    for index, record in enumerate(capture_control):
        schema_problems.extend(
            verify_step_schema(
                record, selected_finger, f"capture_step_{index}"
            )
        )
    for index, record in enumerate(skip_control):
        schema_problems.extend(
            verify_step_schema(
                record, selected_finger, f"skip_step_{index}"
            )
        )
    schema_problems.extend(
        verify_report_terminal_schema(capture_report)
    )
    schema_problems.extend(verify_report_terminal_schema(skip_report))
    decision_first = project_trace(capture_control, DECISION_FIELDS)
    decision_second = project_trace(skip_control, DECISION_FIELDS)
    sensor_first = project_trace(capture_control, SENSOR_FIELDS)
    sensor_second = project_trace(skip_control, SENSOR_FIELDS)
    decision_diff = _first_diff(decision_first, decision_second)
    sensor_diff = _first_diff(sensor_first, sensor_second)
    terminal_diff = _first_diff(
        _report_terminal_projection(capture_report),
        _report_terminal_projection(skip_report),
    )
    traces_identical = bool(
        decision_diff is None
        and sensor_diff is None
        and terminal_diff is None
    )
    # Canonical hashes require JSON-finite inputs; a schema violation such
    # as a NaN sensor value makes serialization impossible and is itself
    # the failure being reported, so hashes stay null in that case.
    if schema_problems:
        trace_hashes = {
            "decision_trace_sha256_capture": None,
            "decision_trace_sha256_skip": None,
            "sensor_trace_sha256_capture": None,
            "sensor_trace_sha256_skip": None,
        }
    else:
        trace_hashes = {
            "decision_trace_sha256_capture": canonical_sha256(
                decision_first
            ),
            "decision_trace_sha256_skip": canonical_sha256(
                decision_second
            ),
            "sensor_trace_sha256_capture": canonical_sha256(sensor_first),
            "sensor_trace_sha256_skip": canonical_sha256(sensor_second),
        }
    binding_identity_ok = bool(
        (capture_report.get("realized_usd_authoring") or {})
        .get("material_binding_identity", {})
        .get("all_bindings_ok")
        is True
        and (skip_report.get("realized_usd_authoring") or {})
        .get("material_binding_identity", {})
        .get("all_bindings_ok")
        is True
    )
    audit = capture_report.get("posthoc_audit") or {}
    points = audit.get("points") or {} if isinstance(audit, Mapping) else {}
    gate_results = {}
    all_gates_passed = True
    for point in AUDIT_POINTS:
        raw = points.get(point) or {}
        evaluated = evaluate_audit_point(
            point,
            raw.get("snapshot") or {},
            selected_finger,
            binding_identity_ok=binding_identity_ok,
        )
        gate_results[point] = evaluated
        all_gates_passed = bool(all_gates_passed and evaluated["passed"])
    summary = {
        "single_finger_validation_passed": False,
        "grasp_success_claimed": False,
        "marker_problems": marker_problems,
        "marker_evidence": marker_evidence,
        "capture_contract_problems": capture_contract,
        "skip_contract_problems": skip_contract,
        "schema_problems": schema_problems,
        **trace_hashes,
        "decision_trace_identical": decision_diff is None,
        "sensor_trace_identical": sensor_diff is None,
        "report_terminal_identical": terminal_diff is None,
        "first_decision_mismatch": _format_diff(decision_diff),
        "first_sensor_mismatch": _format_diff(sensor_diff),
        "first_terminal_mismatch": _format_diff(terminal_diff),
        "audit_points": gate_results,
        "inconclusive_query_noninterference": bool(
            not traces_identical or marker_problems or schema_problems
        ),
    }
    if (
        capture_contract
        or skip_contract
        or marker_problems
        or schema_problems
    ):
        summary["exit_code"] = 3
        summary["failure_reason"] = (
            "episode_contract_marker_or_schema_inconclusive"
        )
    elif not traces_identical:
        summary["exit_code"] = 3
        summary["failure_reason"] = "inconclusive_query_noninterference"
    elif not all_gates_passed:
        summary["exit_code"] = 1
        summary["failure_reason"] = "contact_gate_failed"
    else:
        summary["exit_code"] = 0
        summary["failure_reason"] = None
        summary["single_finger_validation_passed"] = True
    return summary
