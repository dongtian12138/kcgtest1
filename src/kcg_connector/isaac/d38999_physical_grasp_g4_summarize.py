#!/usr/bin/env python3
"""Offline fail-closed summarizer for the G4 sequential episodes.

Validation is outcome-aware: evidence integrity always applies (inputs,
seed/payload/provenance, on-disk source hashes, GPU markers, truth/proxy
boundaries, JSONL validity and monotonic steps); success episodes must
pass the full terminal acceptance contract; failed episodes may lack
not-yet-reached terminal fields but their present fields must be typed
and bounded, and their primary/secondary failure evidence is preserved
verbatim.  Exit 0 = structure valid (physical failures allowed),
--require-all-pass adds exit 2 for any physical failure, exit 1 =
structural/evidence violation.  Never recomputes PASS from posthoc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "kcg_d38999_physical_grasp_g4_summary_v1"
MANIFEST_SCHEMA_VERSION = "kcg_d38999_physical_grasp_g4_input_manifest_v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SOURCE_FILES = {
    "runner_sha256": "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py",
    "wrapper_sha256": (
        "src/kcg_connector/isaac/d38999_tabletop_physical_grasp_v1.py"
    ),
    "three_finger_sequential_grasp_sha256": (
        "src/kcg_connector/kcg_connector/grasp/"
        "three_finger_sequential_grasp.py"
    ),
    "finger_contact_detector_sha256": (
        "src/kcg_connector/kcg_connector/grasp/finger_contact_detector.py"
    ),
    "grasp_stability_monitor_sha256": (
        "src/kcg_connector/kcg_connector/grasp/grasp_stability_monitor.py"
    ),
    "physical_grasp_config_loader_sha256": (
        "src/kcg_connector/kcg_connector/grasp/physical_grasp_config.py"
    ),
    "physical_grasp_config_sha256": (
        "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
    ),
    "pick_config_sha256": (
        "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    ),
    "tabletop_scene_config_sha256": (
        "src/kcg_connector/config/d38999_tabletop_scene_v1.yaml"
    ),
}

GPU_REQUIRED_MARKERS = (
    "Active GPU: NVIDIA GeForce RTX 5070 Ti",
    "Yes: 0",
    '"cuda:0"',
    "selecting device 0",
)

GPU_FORBIDDEN_MARKERS = (
    "Failed to create any GPU devices",
    "[Fatal]",
    "[Error]",
    "CPU fallback",
    "cpu_fallback",
    "Warp initialized on cpu",
    "warp CPU backend",
)

FINGERS = ("f1", "f2", "f3")
STAGE_INCREMENTS_M = (0.002, 0.010, 0.040)
MINIMUM_LIFT_M = 0.045
CONSOLIDATION_RAMP_STEPS = 120
CONSOLIDATION_WINDOW_STEPS = 240
SOFT_SCALE = 0.35
FINAL_SCALE = 1.0
FORCE_GATE_N = 8.0
MOMENT_GATE_NM = 0.30

DROP_REASONS = ("f1_load_lost", "f2_load_lost", "f3_load_lost")


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"source file missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_source_hashes() -> dict[str, str]:
    return {
        key: _sha256_file(REPOSITORY_ROOT / relative)
        for key, relative in SOURCE_FILES.items()
    }


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _check(problems: list[str], condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


def _steps_document(
    episode_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    path = episode_dir / "controller_steps.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing controller steps: {path}")
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    previous_step = None
    final_order = None
    window_steps_seen: list[int] = []
    window_scale_bad = []
    window_target_bad = []
    window_states_bad = []
    ramp_steps_seen: list[int] = []
    config_mismatches: dict[str, tuple[Any, Any]] = {}
    config_missing: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                problems.append(
                    f"controller_steps line {line_number} invalid: {error}"
                )
                continue
            step = record.get("global_step")
            if isinstance(step, bool) or not isinstance(step, int):
                problems.append(
                    f"line {line_number}: global_step not a non-bool integer"
                )
            elif previous_step is not None and step <= previous_step:
                problems.append(
                    f"line {line_number}: global_step not strictly increasing"
                )
            if isinstance(step, int) and not isinstance(step, bool):
                previous_step = step
            order = record.get("contact_order")
            if (
                isinstance(order, list)
                and len(order) == 3
                and set(order) == set(FINGERS)
            ):
                final_order = list(order)
            if record.get("phase") == "physical_grip_consolidation":
                evidence = record.get("controller_evidence") or {}
                ramp = evidence.get("consolidation_ramp_step")
                window = evidence.get("consolidation_window_step")
                if (
                    isinstance(window, int)
                    and not isinstance(window, bool)
                    and window == 0
                    and isinstance(ramp, int)
                    and not isinstance(ramp, bool)
                ):
                    ramp_steps_seen.append(ramp)
                if isinstance(window, int) and not isinstance(window, bool):
                    if window > 0:
                        window_steps_seen.append(window)
                        applied = record.get("applied_finger_stiffness_scale")
                        if not _finite(applied) or not math.isclose(
                            float(applied), FINAL_SCALE, abs_tol=1e-12
                        ):
                            window_scale_bad.append(step)
                        if evidence.get("targets_match_frozen") is not True:
                            window_target_bad.append(step)
                        elif record.get("finger_targets_rad") != evidence.get(
                            "frozen_targets_rad"
                        ):
                            window_target_bad.append(step)
                        post_states = evidence.get("post_states") or {}
                        if not all(
                            post_states.get(f) == "STABLE_CONTACT"
                            for f in FINGERS
                        ):
                            window_states_bad.append(step)
                for key in (
                    "soft_hold_stiffness_scale_configured",
                    "consolidation_final_stiffness_scale_configured",
                    "consolidation_ramp_steps_configured",
                    "consolidation_window_steps_configured",
                ):
                    if evidence.get(key) is None:
                        config_missing.append((step, key))
                for key, expected in (
                    ("soft_hold_stiffness_scale_configured", SOFT_SCALE),
                    (
                        "consolidation_final_stiffness_scale_configured",
                        FINAL_SCALE,
                    ),
                    ("consolidation_ramp_steps_configured",
                     CONSOLIDATION_RAMP_STEPS),
                    ("consolidation_window_steps_configured",
                     CONSOLIDATION_WINDOW_STEPS),
                ):
                    value = evidence.get(key)
                    if value is not None and not (
                        _finite(value)
                        and math.isclose(
                            float(value), float(expected), abs_tol=1e-12
                        )
                    ):
                        config_mismatches[key] = (value, expected)
            records.append(record)
    cross: dict[str, Any] = {
        "final_contact_order": final_order,
        "contact_order_available": final_order is not None,
        "ramp_records_total": len(ramp_steps_seen),
        "window_step_sequence": list(window_steps_seen),
        "window_steps_seen_count": len(window_steps_seen),
        "window_scale_mismatch_steps": window_scale_bad,
        "window_target_mismatch_steps": window_target_bad,
        "window_states_mismatch_steps": window_states_bad,
        "window_config_missing_records": len(config_missing),
        "evidence_config_mismatches": config_mismatches,
    }
    return records, problems, cross


def verify_evidence_integrity(
    report: Mapping[str, Any],
    kit_log_text: str,
    steps_cross: Mapping[str, Any],
    disk_hashes: Mapping[str, str],
) -> list[str]:
    problems: list[str] = []
    seed = report.get("seed")
    _check(
        problems,
        isinstance(seed, int) and not isinstance(seed, bool),
        f"seed must be a non-bool integer, got {seed!r}",
    )
    prov = report.get("provenance") or {}
    _check(
        problems, prov.get("seed") == seed, "provenance.seed mismatch"
    )
    realized = report.get("realized_randomization") or {}
    _check(problems, realized.get("seed") == seed, "realized seed mismatch")
    _check(
        problems,
        prov.get("payload_sha256") == realized.get("payload_sha256"),
        "payload_sha256 mismatch inside report",
    )
    _check(
        problems,
        report.get("physical_grasp_method") == "sequential-compliant",
        "method must be sequential-compliant",
    )
    _check(problems, report.get("formal_lift_mode") == "staged", "mode staged")
    _check(problems, report.get("gui") is False, "gui must be false")
    for key, expected in disk_hashes.items():
        _check(
            problems,
            prov.get(key) == expected,
            f"provenance {key} does not match on-disk source",
        )
    for marker in GPU_REQUIRED_MARKERS:
        _check(
            problems,
            marker in kit_log_text,
            f"kit log missing GPU marker {marker!r}",
        )
    for marker in GPU_FORBIDDEN_MARKERS:
        _check(
            problems,
            marker not in kit_log_text,
            f"kit log contains forbidden marker {marker!r}",
        )
    _check(
        problems,
        report.get("control_reads_object_truth") is False,
        "control_reads_object_truth false required",
    )
    _check(
        problems,
        report.get("control_reads_contact_report") is False,
        "control_reads_contact_report false required",
    )
    _check(
        problems,
        report.get("truth_orientation_used") is False,
        "truth_orientation_used false required",
    )
    _check(
        problems,
        report.get("object_pose_writes_after_start") == 0,
        "object_pose_writes_after_start must be 0",
    )
    _check(problems, report.get("attachment") == "none", "attachment none")
    _check(problems, report.get("object_drive") == "none", "object_drive none")
    _check(
        problems,
        (report.get("proxy_collision_filter") or {}).get("enabled") is False,
        "proxy collision filter disabled required",
    )
    _check(
        problems,
        report.get("posthoc_truth_evaluation_only") is True,
        "posthoc truth evaluation only required",
    )
    return problems


def verify_success_acceptance(
    report: Mapping[str, Any], steps_cross: Mapping[str, Any]
) -> list[str]:
    problems: list[str] = []
    acceptance = report.get("formal_acceptance") or {}
    _check(
        problems,
        acceptance.get("passed") is True,
        "acceptance.passed true",
    )
    _check(
        problems,
        acceptance.get("sensor_lift_gate") is True,
        "acceptance.sensor_lift_gate true",
    )
    _check(
        problems,
        acceptance.get("episode_end_contact_gate") is True,
        "acceptance.episode_end_contact_gate true",
    )
    _check(
        problems,
        acceptance.get("controller_stable") is True,
        "acceptance.controller_stable true",
    )
    _check(
        problems,
        acceptance.get("post_grasp_stabilization_proxy_used") is False,
        "acceptance post-grasp proxy false",
    )
    actual = acceptance.get("actual_body_lift_m")
    _check(
        problems,
        _finite(actual) and float(actual) >= MINIMUM_LIFT_M,
        f"actual_body_lift_m {actual!r} below {MINIMUM_LIFT_M}",
    )
    _check(
        problems,
        report.get("finite_throughout") is True,
        "finite_throughout true",
    )
    _check(problems, report.get("finite_final") is True, "finite_final true")
    _check(
        problems,
        report.get("final_tail_diagnostics_finite") is True,
        "final_tail_diagnostics_finite true",
    )
    _check(
        problems,
        report.get("final_all_fingers_body_contact") is True,
        "final_all_fingers_body_contact true",
    )
    _check(
        problems,
        report.get("zero_forbidden_contacts") is True,
        "zero_forbidden_contacts true",
    )
    _check(
        problems,
        report.get("final_unsupported") is False,
        "final_unsupported false",
    )
    contacts = report.get("final_contacts") or {}
    finger_body = contacts.get("finger_body_group_records") or {}
    _check(
        problems,
        set(finger_body) >= set(FINGERS)
        and all(
            isinstance(finger_body[f].get("body"), int)
            and not isinstance(finger_body[f].get("body"), bool)
            and finger_body[f]["body"] >= 1
            for f in FINGERS
        ),
        "each finger needs body contact count >= 1",
    )
    material = contacts.get("material_evidence") or {}
    _check(problems, material.get("available") is True, "material available")
    _check(
        problems,
        isinstance(material.get("grip_grip_records"), int)
        and material["grip_grip_records"] >= 1,
        "grip_grip_records >= 1",
    )
    _check(
        problems,
        contacts.get("plug_table_records") == 0,
        "plug_table_records 0",
    )
    _check(
        problems,
        contacts.get("unexpected_robot_link_records") == 0,
        "unexpected_robot_link_records 0",
    )
    stages = report.get("formal_lift_stages")
    _check(
        problems,
        isinstance(stages, list) and len(stages) == 3,
        "exactly 3 lift stages",
    )
    if isinstance(stages, list) and len(stages) == 3:
        for index, stage in enumerate(stages):
            _check(
                problems,
                stage.get("stage") == index + 1,
                f"stage {index + 1} numbering",
            )
            _check(
                problems,
                _finite(stage.get("increment_m"))
                and math.isclose(
                    float(stage["increment_m"]),
                    STAGE_INCREMENTS_M[index],
                    abs_tol=1e-12,
                ),
                f"stage {index + 1} increment",
            )
            _check(
                problems,
                stage.get("passed_sensor_gate") is True,
                f"stage {index + 1} sensor gate",
            )
    monitor = report.get("formal_lift_monitor") or {}
    _check(problems, monitor.get("failed") is False, "monitor.failed false")
    _check(
        problems,
        _finite(monitor.get("force_gate_n"))
        and math.isclose(
            float(monitor["force_gate_n"]), FORCE_GATE_N, abs_tol=1e-12
        ),
        "monitor force gate must remain 8.0",
    )
    _check(
        problems,
        _finite(monitor.get("moment_gate_nm"))
        and math.isclose(
            float(monitor["moment_gate_nm"]), MOMENT_GATE_NM, abs_tol=1e-12
        ),
        "monitor moment gate must remain 0.30",
    )
    force_peak = monitor.get("peak_wrist_force_increment_n")
    moment_peak = monitor.get("peak_moment_safety_score_nm")
    _check(problems, _finite(force_peak), "force peak finite")
    _check(problems, _finite(moment_peak), "moment peak finite")
    if _finite(force_peak):
        _check(
            problems,
            float(force_peak) <= FORCE_GATE_N,
            f"force peak {force_peak!r} exceeds 8 N",
        )
    if _finite(moment_peak):
        _check(
            problems,
            float(moment_peak) <= MOMENT_GATE_NM,
            f"moment peak {moment_peak!r} exceeds 0.30",
        )
    grasp = report.get("grasp_controller") or {}
    _check(
        problems,
        isinstance(grasp.get("contact_order"), list)
        and list(grasp["contact_order"])
        == list(steps_cross.get("final_contact_order") or []),
        "grasp_controller.contact_order vs JSONL mismatch",
    )
    cross = report.get("sequential_consolidation") or {}
    _check(problems, cross.get("completed") is True, "consolidation completed")
    _check(problems, cross.get("lift_ready") is True, "lift_ready true")
    _check(
        problems,
        cross.get("targets_frozen_exact") is True,
        "targets_frozen_exact true",
    )
    _check(
        problems,
        _finite(cross.get("applied_scale_min"))
        and math.isclose(
            float(cross["applied_scale_min"]), SOFT_SCALE, abs_tol=1e-12
        ),
        "applied_scale_min must be 0.35",
    )
    _check(
        problems,
        _finite(cross.get("applied_scale_max"))
        and math.isclose(
            float(cross["applied_scale_max"]), FINAL_SCALE, abs_tol=1e-12
        ),
        "applied_scale_max must be 1.0",
    )
    _check(
        problems,
        cross.get("commanded_scale_monotonic") is True,
        "commanded scale monotonic",
    )
    _check(
        problems,
        _finite(cross.get("final_stiffness_scale"))
        and math.isclose(
            float(cross["final_stiffness_scale"]), FINAL_SCALE, abs_tol=1e-12
        ),
        "final_stiffness_scale must be 1.0",
    )
    _check(
        problems,
        _finite(cross.get("soft_hold_stiffness_scale"))
        and math.isclose(
            float(cross["soft_hold_stiffness_scale"]),
            SOFT_SCALE,
            abs_tol=1e-12,
        ),
        "soft_hold_stiffness_scale must be 0.35",
    )
    _check(
        problems,
        cross.get("ramp_steps") == CONSOLIDATION_RAMP_STEPS,
        "ramp_steps 120",
    )
    _check(
        problems,
        cross.get("window_steps") == CONSOLIDATION_WINDOW_STEPS,
        "window_steps 240",
    )
    _check(
        problems,
        cross.get("final_window_sample_count") == CONSOLIDATION_WINDOW_STEPS,
        "final_window_sample_count 240",
    )
    final_root = cross.get("final_root_reference_nm")
    _check(
        problems,
        isinstance(final_root, (list, tuple))
        and len(final_root) == 3
        and all(_finite(v) for v in final_root),
        "final_root_reference_nm three finite values",
    )
    final_wrist = cross.get("final_wrist_reference")
    _check(
        problems,
        isinstance(final_wrist, (list, tuple))
        and len(final_wrist) == 6
        and all(_finite(v) for v in final_wrist),
        "final_wrist_reference six finite values",
    )
    window_seen = steps_cross.get("window_step_sequence") or []
    _check(
        problems,
        window_seen == list(range(1, CONSOLIDATION_WINDOW_STEPS + 1)),
        (
            "window steps must be complete 1..240, got "
            f"{len(window_seen)} unique"
        ),
    )
    _check(
        problems,
        steps_cross.get("window_steps_seen_count")
        == CONSOLIDATION_WINDOW_STEPS,
        "window must have exactly 240 records",
    )
    _check(
        problems,
        steps_cross.get("ramp_records_total") == CONSOLIDATION_RAMP_STEPS,
        "ramp must have exactly 120 records",
    )
    _check(
        problems,
        steps_cross.get("window_config_missing_records") == 0,
        "every consolidation record needs all four configured fields",
    )
    _check(
        problems,
        not steps_cross.get("window_scale_mismatch_steps"),
        "window applied scale must be 1.0 everywhere",
    )
    _check(
        problems,
        not steps_cross.get("window_target_mismatch_steps"),
        "window targets must match frozen",
    )
    _check(
        problems,
        not steps_cross.get("window_states_mismatch_steps"),
        "window post_states must be STABLE_CONTACT",
    )
    _check(
        problems,
        not steps_cross.get("evidence_config_mismatches"),
        f"controller_evidence config mismatch: "
        f"{steps_cross.get('evidence_config_mismatches')}",
    )
    return problems


def failure_primary(
    report: Mapping[str, Any],
) -> tuple[str | None, list[str], str | None]:
    monitor = report.get("formal_lift_monitor") or {}
    recovery = report.get("formal_recovery") or {}
    lift_failure = report.get("formal_lift_failure") or {}
    pre_lift = report.get("pre_lift_grasp_controller_evidence") or {}
    error_field = report.get("error")
    sources = {
        "monitor": monitor.get("failure_reason"),
        "recovery": recovery.get("original_failure_reason"),
        "lift_failure": lift_failure.get("reason"),
        "pre_lift": pre_lift.get("failure_reason"),
    }
    non_null = {k: v for k, v in sources.items() if v not in (None, "")}
    problems = []
    values = {str(v) for v in non_null.values()}
    if len(values) > 1:
        problems.append(
            "conflicting primary failure reasons: " + repr(non_null)
        )
    if non_null:
        reason = str(next(iter(non_null.values())))
        return reason, problems, "structured"
    if isinstance(error_field, str) and error_field.strip():
        return error_field.strip(), problems, "report.error_unstructured"
    return None, problems, None


def verify_failure_consistency(
    report: Mapping[str, Any], steps_cross: Mapping[str, Any]
) -> list[str]:
    problems: list[str] = []
    acceptance = report.get("formal_acceptance")
    if acceptance is not None:
        if not isinstance(acceptance, Mapping):
            problems.append("formal_acceptance must be a mapping or null")
        else:
            for key in ("passed", "sensor_lift_gate", "controller_stable"):
                value = acceptance.get(key)
                if value is not None and not isinstance(value, bool):
                    problems.append(f"formal_acceptance.{key} not bool")
    contacts = report.get("final_contacts")
    if contacts is not None:
        if not isinstance(contacts, Mapping):
            problems.append("final_contacts must be a mapping or null")
        else:
            for key in ("plug_table_records", "unexpected_robot_link_records"):
                value = contacts.get(key)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int)
                ):
                    problems.append(f"final_contacts.{key} not int")
    for key in (
        "finite_throughout",
        "finite_final",
        "final_tail_diagnostics_finite",
        "final_all_fingers_body_contact",
        "zero_forbidden_contacts",
        "final_unsupported",
    ):
        value = report.get(key)
        if value is not None and not isinstance(value, bool):
            problems.append(f"{key} must be bool or null")
    monitor = report.get("formal_lift_monitor") or {}
    if monitor.get("failed") is not None and not isinstance(
        monitor.get("failed"), bool
    ):
        problems.append("formal_lift_monitor.failed not bool")
    recovery = report.get("formal_recovery")
    if recovery is not None and not isinstance(recovery, Mapping):
        problems.append("formal_recovery must be a mapping or null")
    monitor = report.get("formal_lift_monitor") or {}
    if monitor.get("force_gate_n") is not None:
        _check(
            problems,
            _finite(monitor.get("force_gate_n"))
            and math.isclose(
                float(monitor["force_gate_n"]), FORCE_GATE_N, abs_tol=1e-12
            ),
            "monitor force gate must remain 8.0 when present",
        )
    if monitor.get("moment_gate_nm") is not None:
        _check(
            problems,
            _finite(monitor.get("moment_gate_nm"))
            and math.isclose(
                float(monitor["moment_gate_nm"]),
                MOMENT_GATE_NM,
                abs_tol=1e-12,
            ),
            "monitor moment gate must remain 0.30 when present",
        )
    for key in (
        "peak_wrist_force_increment_n",
        "peak_moment_safety_score_nm",
    ):
        if monitor.get(key) is not None:
            _check(
                problems,
                _finite(monitor.get(key)),
                f"monitor {key} finite",
            )
    consolidation = report.get("sequential_consolidation")
    if consolidation is not None:
        if not isinstance(consolidation, Mapping):
            problems.append(
                "sequential_consolidation must be a mapping or null"
            )
        else:
            for key in ("completed", "lift_ready", "targets_frozen_exact"):
                value = consolidation.get(key)
                if value is not None and not isinstance(value, bool):
                    problems.append(f"sequential_consolidation.{key} not bool")
            for key in ("applied_scale_min", "applied_scale_max"):
                value = consolidation.get(key)
                if value is not None:
                    _check(
                        problems,
                        _finite(value)
                        and SOFT_SCALE - 1e-9 <= float(value),
                        f"sequential_consolidation.{key} out of bounds",
                    )
            for key in (
                "ramp_steps",
                "window_steps",
                "final_window_sample_count",
            ):
                value = consolidation.get(key)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int)
                ):
                    problems.append(f"sequential_consolidation.{key} not int")
            final_scale = consolidation.get("final_stiffness_scale")
            if final_scale is not None:
                _check(
                    problems,
                    _finite(final_scale) and float(final_scale) <= 1.0,
                    "final_stiffness_scale must be <= 1.0 when present",
                )
    pre_lift = report.get("pre_lift_grasp_controller_evidence")
    if pre_lift is not None:
        if not isinstance(pre_lift, Mapping):
            problems.append("pre_lift evidence must be a mapping or null")
        else:
            order = pre_lift.get("contact_order")
            if order is not None and not (
                isinstance(order, list) and len(order) == 3
            ):
                problems.append("pre_lift contact_order malformed")
            if pre_lift.get("lift_ready") is not None and not isinstance(
                pre_lift.get("lift_ready"), bool
            ):
                problems.append("pre_lift lift_ready not bool")
    window_seen = steps_cross.get("window_step_sequence") or []
    claims_complete = (
        isinstance(consolidation, Mapping)
        and consolidation.get("completed") is True
    )
    if claims_complete:
        _check(
            problems,
            window_seen
            == list(range(1, CONSOLIDATION_WINDOW_STEPS + 1)),
            "consolidation claims completed but window is not 1..240",
        )
    elif window_seen:
        _check(
            problems,
            window_seen == list(range(1, len(window_seen) + 1)),
            "partial window must be consecutive 1..k without gaps or "
            "duplicates",
        )
        _check(
            problems,
            not steps_cross.get("window_scale_mismatch_steps"),
            "partial window applied scale must be 1.0",
        )
        _check(
            problems,
            not steps_cross.get("window_target_mismatch_steps"),
            "partial window targets must match frozen",
        )
        _check(
            problems,
            not steps_cross.get("window_states_mismatch_steps"),
            "partial window post_states must be STABLE_CONTACT",
        )
        _check(
            problems,
            steps_cross.get("window_config_missing_records") == 0,
            "partial window records need all four configured fields",
        )
        _check(
            problems,
            not steps_cross.get("evidence_config_mismatches"),
            "partial window config mismatch",
        )
    return problems


def _stats(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray([float(v) for v in values], dtype=np.float64)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("statistics require at least one finite value")
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p50": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "maximum": float(np.max(data)),
    }


def _metric(values, unit, readable_unit, scale, absolute=False):
    data = [abs(float(v)) if absolute else float(v) for v in values]
    result = _stats(data)
    result["unit"] = unit
    result["readable_unit"] = readable_unit
    result["readable"] = {
        key: result[key] * scale
        for key in ("mean", "median", "p50", "p95", "maximum")
    }
    return result


def _seed_metrics(
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    mon = report.get("formal_lift_monitor") or {}
    cons = report.get("sequential_consolidation") or {}
    ac = report.get("formal_acceptance") or {}
    tst = report.get("table_stage") or {}
    pose = report.get("posthoc_pose_error") or {}
    slip = report.get("posthoc_lift_relative_slip") or {}
    tail = report.get("final_tail_observable_angular_speed_rad_s") or {}
    rotation = report.get("final_tail_net_rotation_rad") or {}
    summary = (
        report.get("pre_lift_grasp_controller_evidence") or {}
    ).get("sequential_final_summary") or {}
    moment_peak = mon.get("peak_moment_safety_score_nm")
    force_peak = mon.get("peak_wrist_force_increment_n")
    unavailable: list[str] = []
    metrics: dict[str, Any] = {}
    for key, value in (
        ("actual_body_lift_mm", ac.get("actual_body_lift_m")),
        ("body_tcp_slip_mm", report.get("body_tcp_slip_m")),
        (
            "final_hold_displacement_mm",
            report.get("final_hold_displacement_m"),
        ),
        ("table_xy_mm", tst.get("translation_xy_m")),
        ("abs_table_yaw_deg", tst.get("yaw_delta_rad")),
        ("wrist_force_peak_n", force_peak),
        ("moment_score_peak_nm", moment_peak),
        (
            "normalized_load_imbalance",
            summary.get("normalized_load_imbalance"),
        ),
    ):
        if _finite(value):
            if key == "abs_table_yaw_deg":
                metrics[key] = abs(float(value)) * 180.0 / math.pi
            elif key == "table_xy_mm":
                metrics[key] = float(value) * 1000.0
            elif key in (
                "actual_body_lift_mm",
                "body_tcp_slip_mm",
                "final_hold_displacement_mm",
            ):
                metrics[key] = float(value) * 1000.0
            else:
                metrics[key] = float(value)
        else:
            unavailable.append(key)
    if _finite(moment_peak):
        metrics["moment_gate_margin"] = (
            MOMENT_GATE_NM - float(moment_peak)
        ) / MOMENT_GATE_NM
    else:
        unavailable.append("moment_gate_margin")
    root_refs = cons.get("final_root_reference_nm")
    if (
        isinstance(root_refs, (list, tuple))
        and len(root_refs) == 3
        and all(_finite(v) for v in root_refs)
    ):
        metrics["final_root_reference_nm"] = {
            f: float(root_refs[i]) for i, f in enumerate(FINGERS)
        }
    else:
        unavailable.append("final_root_reference_nm")
    rot = rotation.get("nut_relative_to_body_rad")
    if _finite(rot):
        metrics["nut_relative_net_rotation_deg"] = float(rot) * 180.0 / math.pi
    else:
        unavailable.append("nut_relative_net_rotation_deg")
    tail_rel = tail.get("nut_relative_to_body") or {}
    for key in ("mean", "rms", "maximum"):
        value = tail_rel.get(key)
        if _finite(value):
            metrics[f"nut_relative_tail_speed_{key}_rad_s"] = float(value)
        else:
            unavailable.append(f"nut_relative_tail_speed_{key}_rad_s")
    for label, field in (
        ("posthoc_pose_error", pose),
        ("lift_relative_slip", slip),
    ):
        channels = {}
        for channel in (
            "dx_m",
            "dy_m",
            "dz_m",
            "drx_rad",
            "dry_rad",
            "drz_rad",
        ):
            value = field.get(channel)
            if _finite(value):
                channels[channel] = float(value)
        if channels:
            metrics[label] = channels
    return metrics, unavailable


def _pose_blocks(rows, field, absolute=False):
    result = {}
    for channel in ("dx_m", "dy_m", "dz_m", "drx_rad", "dry_rad", "drz_rad"):
        if channel.endswith("_m"):
            unit, readable, scale = "m", "mm", 1000.0
        else:
            unit, readable, scale = "rad", "deg", 180.0 / math.pi
        values = [
            row[field][channel]
            for row in rows
            if channel in row.get(field, {})
        ]
        if values:
            result[channel] = _metric(values, unit, readable, scale, absolute)
    trans = [
        math.sqrt(sum(row[field][c] ** 2 for c in ("dx_m", "dy_m", "dz_m")))
        for row in rows
        if all(c in row.get(field, {}) for c in ("dx_m", "dy_m", "dz_m"))
    ]
    rot = [
        math.sqrt(
            sum(row[field][c] ** 2 for c in ("drx_rad", "dry_rad", "drz_rad"))
        )
        for row in rows
        if all(
            c in row.get(field, {})
            for c in ("drx_rad", "dry_rad", "drz_rad")
        )
    ]
    if trans:
        result["translation_norm"] = _metric(
            trans, "m", "mm", 1000.0, absolute
        )
    if rot:
        result["rotation_norm"] = _metric(
            rot, "rad", "deg", 180.0 / math.pi, absolute
        )
    return result


def _series(rows, key):
    return [float(row[key]) for row in rows if _finite(row.get(key))]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode", action="append", required=True,
        help="episode directory (repeatable, paired with --kit-log)",
    )
    parser.add_argument(
        "--kit-log", action="append", required=True,
        help="kit log file (repeatable, paired with --episode)",
    )
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument(
        "--require-all-pass", action="store_true",
        help="exit 2 when structure valid but some episode physically failed",
    )
    arguments = parser.parse_args(argv)

    episode_dirs = [Path(raw).resolve() for raw in arguments.episode]
    kit_logs = [Path(raw).resolve() for raw in arguments.kit_log]
    if len(episode_dirs) != len(kit_logs):
        parser.error("episode and kit-log counts must be equal")
    if not episode_dirs:
        parser.error("at least one episode is required")
    output_dir = Path(arguments.output).resolve()
    if output_dir in episode_dirs:
        parser.error("output directory must not be an episode directory")

    disk_hashes = current_source_hashes()
    cli_sha256 = _sha256_file(Path(__file__).resolve())
    episodes = []
    for episode_dir, kit_log in zip(episode_dirs, kit_logs):
        problems: list[str] = []
        report = None
        steps_cross: dict[str, Any] = {}
        steps_problems: list[str] = []
        try:
            report = json.loads(
                (episode_dir / "nominal_physics_report.json").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(report, dict):
                raise ValueError("report is not a JSON object")
            if not kit_log.is_file():
                raise FileNotFoundError(f"kit log missing: {kit_log}")
            kit_text = kit_log.read_text(encoding="utf-8", errors="replace")
            _records, steps_problems, steps_cross = _steps_document(
                episode_dir
            )
            problems.extend(steps_problems)
            problems.extend(
                verify_evidence_integrity(
                    report, kit_text, steps_cross, disk_hashes
                )
            )
            passed = report.get("passed")
            exit_code = report.get("process_exit_code")
            primary, primary_problems, primary_source = failure_primary(
                report
            )
            problems.extend(primary_problems)
            if passed is True and exit_code == 0 and not primary:
                problems.extend(verify_success_acceptance(report, steps_cross))
            elif passed is False and exit_code not in (0, None) and primary:
                problems.extend(
                    verify_failure_consistency(report, steps_cross)
                )
            else:
                problems.append(
                    "contradictory passed/exit/failure fields: "
                    f"passed={passed!r} exit={exit_code!r} primary={primary!r}"
                )
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"{type(error).__name__}: {error}")
        seed = report.get("seed") if report else None
        if isinstance(seed, bool) or not isinstance(seed, int):
            seed = None
        physical_success = bool(
            not problems and report and report.get("passed") is True
        )
        failure_record = None
        if not problems and report and report.get("passed") is not True:
            monitor = report.get("formal_lift_monitor") or {}
            recovery = report.get("formal_recovery") or {}
            lift_failure = report.get("formal_lift_failure") or {}
            pre_lift = report.get("pre_lift_grasp_controller_evidence") or {}
            primary, _, primary_source = failure_primary(report)
            stage = recovery.get("failure_stage")
            if stage is None:
                stage = lift_failure.get("stage")
            stage_step = recovery.get("failure_stage_step")
            if stage_step is None:
                stage_step = lift_failure.get("stage_step")
            global_step = recovery.get("failure_global_step")
            if global_step is None:
                global_step = lift_failure.get("global_step")
            failure_record = {
                "primary_failure": primary,
                "primary_source": primary_source,
                "first_failure_stage": stage,
                "first_failure_stage_step": stage_step,
                "first_failure_global_step": global_step,
                "monitor_failure_reason": monitor.get("failure_reason"),
                "moment_trigger_component": monitor.get(
                    "moment_trigger_component"
                ),
                "pre_lift_failure_reason": pre_lift.get("failure_reason"),
                "recovery_requested": recovery.get("requested"),
                "recovery_completed": recovery.get("completed"),
                "recovery_return_completed": recovery.get("return_completed"),
                "recovery_open_completed": recovery.get("open_completed"),
                "recovery_secondary_failure": recovery.get("interrupted_by"),
                "error_field": report.get("error"),
            }
        metrics, unavailable_metrics = (
            _seed_metrics(report) if report else ({}, [])
        )
        episodes.append(
            {
                "directory": str(episode_dir),
                "kit_log": str(kit_log),
                "seed": seed,
                "structural_ok": not problems,
                "structural_problems": list(problems),
                "physical_success": physical_success,
                "failure_record": failure_record,
                "metrics": metrics,
                "unavailable_metrics": unavailable_metrics,
                "steps_cross": {
                    "window_steps_seen_count": (
                        steps_cross.get("window_steps_seen_count")
                    ),
                    "final_contact_order": steps_cross.get(
                        "final_contact_order"
                    ),
                },
            }
        )

    seeds = [e["seed"] for e in episodes]
    structural_total = sum(
        len(e["structural_problems"]) for e in episodes
    )
    if any(seed is None for seed in seeds):
        structural_total += 1
    if len(set(seeds)) != len(seeds):
        structural_total += 1

    valid = [e for e in episodes if e["structural_ok"]]
    passed_rows = [e["metrics"] for e in valid if e["physical_success"]]
    all_rows = [e["metrics"] for e in valid]

    def block(key, unit, readable, scale):
        result = {}
        all_values = _series(all_rows, key)
        passed_values = _series(passed_rows, key)
        if all_values:
            result["all_valid"] = _metric(all_values, unit, readable, scale)
        else:
            result["all_valid_available"] = False
        if passed_values:
            result["passed_only"] = _metric(
                passed_values, unit, readable, scale
            )
        else:
            result["passed_only_available"] = False
        return result

    failure_distribution: dict[str, int] = {}
    contact_orders: dict[str, int] = {}
    contact_step_deltas: list[int] = []
    for entry in valid:
        report = json.loads(
            (
                Path(entry["directory"]) / "nominal_physics_report.json"
            ).read_text(encoding="utf-8")
        )
        grasp = report.get("grasp_controller") or {}
        pre_lift = report.get("pre_lift_grasp_controller_evidence") or {}
        order = grasp.get("contact_order")
        if order is None:
            order = pre_lift.get("contact_order")
        if isinstance(order, list) and len(order) == 3:
            contact_orders["-".join(order)] = (
                contact_orders.get("-".join(order), 0) + 1
            )
            steps_map = grasp.get("contact_global_steps") or {}
            if set(steps_map) == set(FINGERS):
                ordered_steps = [steps_map[f] for f in order]
                for earlier, later in zip(ordered_steps, ordered_steps[1:]):
                    contact_step_deltas.append(int(later) - int(earlier))
        if not entry["physical_success"] and entry["failure_record"]:
            reason = entry["failure_record"]["primary_failure"] or "unknown"
            failure_distribution[reason] = (
                failure_distribution.get(reason, 0) + 1
            )

    drop_count = sum(
        count for reason, count in failure_distribution.items()
        if any(token in reason for token in DROP_REASONS)
    )
    recovery_requested = sum(
        1
        for e in valid
        if not e["physical_success"]
        and (e["failure_record"] or {}).get("recovery_requested") is True
    )
    recovery_completed = sum(
        1
        for e in valid
        if not e["physical_success"]
        and (e["failure_record"] or {}).get("recovery_completed") is True
    )
    recovery_incomplete = sum(
        1
        for e in valid
        if not e["physical_success"]
        and (e["failure_record"] or {}).get("recovery_requested") is True
        and (e["failure_record"] or {}).get("recovery_completed") is not True
    )

    root_channels = {}
    for finger in FINGERS:
        values = [
            e["metrics"]["final_root_reference_nm"][finger]
            for e in valid
            if e["metrics"].get("final_root_reference_nm")
            and _finite(e["metrics"]["final_root_reference_nm"].get(finger))
        ]
        if values:
            root_channels[f"root_{finger}"] = _metric(
                values, "N*m", "N*m", 1.0
            )
    nut_tail = {}
    for key in ("mean", "rms", "maximum"):
        field = f"nut_relative_tail_speed_{key}_rad_s"
        values = _series(all_rows, field)
        passed_values = _series(passed_rows, field)
        entry = {}
        if values:
            entry["all_valid"] = _metric(values, "rad/s", "rad/s", 1.0)
        else:
            entry["all_valid_available"] = False
        if passed_values:
            entry["passed_only"] = _metric(
                passed_values, "rad/s", "rad/s", 1.0
            )
        else:
            entry["passed_only_available"] = False
        nut_tail[key] = entry

    statistics = {
        "episode_count": len(episodes),
        "structural_valid_count": len(valid),
        "physical_success_count": sum(
            1 for e in valid if e["physical_success"]
        ),
        "physical_failure_count": sum(
            1 for e in valid if not e["physical_success"]
        ),
        "drop_count": drop_count,
        "drop_reason_definition": {
            "tokens": list(DROP_REASONS),
            "match": "primary failure reason contains any token",
        },
        "recovery_requested_count": recovery_requested,
        "recovery_completed_count": recovery_completed,
        "recovery_incomplete_count": recovery_incomplete,
        "failure_reason_distribution": failure_distribution,
        "contact_order_distribution": contact_orders,
        "contact_step_deltas": (
            _stats(contact_step_deltas) if contact_step_deltas else None
        ),
        "posthoc_only": {
            "table_xy": block("table_xy_mm", "mm", "mm", 1.0),
            "abs_table_yaw": block("abs_table_yaw_deg", "deg", "deg", 1.0),
            "pose_error_signed": _pose_blocks(all_rows, "posthoc_pose_error"),
            "pose_error_abs": _pose_blocks(
                all_rows, "posthoc_pose_error", absolute=True
            ),
            "nut_relative_net_rotation": block(
                "nut_relative_net_rotation_deg", "deg", "deg", 1.0
            ),
            "nut_relative_tail_speed": nut_tail,
        },
        "lift": {
            "actual_body_lift": block(
                "actual_body_lift_mm", "mm", "mm", 1.0
            ),
            "body_tcp_slip": block("body_tcp_slip_mm", "mm", "mm", 1.0),
            "final_hold_displacement": block(
                "final_hold_displacement_mm", "mm", "mm", 1.0
            ),
            "lift_relative_slip_signed": _pose_blocks(
                all_rows, "lift_relative_slip"
            ),
            "lift_relative_slip_abs": _pose_blocks(
                all_rows, "lift_relative_slip", absolute=True
            ),
        },
        "sensors": {
            "wrist_force_peak": block("wrist_force_peak_n", "N", "N", 1.0),
            "moment_score_peak": block(
                "moment_score_peak_nm", "N*m", "N*m", 1.0
            ),
            "moment_gate_margin": block(
                "moment_gate_margin", "1", "1", 1.0
            ),
            "normalized_load_imbalance": block(
                "normalized_load_imbalance", "1", "1", 1.0
            ),
            "final_root_reference": root_channels,
        },
        "p95_method": "numpy_linear_default_np_percentile_95",
        "n_less_than_30_preliminary": len(valid) < 30,
    }

    manifest_inputs = []
    for entry, episode_dir, kit_log in zip(episodes, episode_dirs, kit_logs):
        report_path = episode_dir / "nominal_physics_report.json"
        steps_path = episode_dir / "controller_steps.jsonl"
        manifest_inputs.append(
            {
                "role": "episode",
                "seed": entry["seed"],
                "directory": str(episode_dir),
                "report_file": str(report_path),
                "report_sha256": _sha256_file(report_path),
                "steps_file": str(steps_path),
                "steps_sha256": _sha256_file(steps_path),
                "kit_log": str(kit_log),
                "kit_log_sha256": _sha256_file(kit_log),
            }
        )
    for key, relative in SOURCE_FILES.items():
        manifest_inputs.append(
            {
                "role": "source_file",
                "key": key,
                "path": str(REPOSITORY_ROOT / relative),
                "sha256": disk_hashes[key],
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_source_sha256": cli_sha256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": manifest_inputs,
    }
    canonical = json.dumps(
        {"inputs": manifest["inputs"]}, sort_keys=True, ensure_ascii=False
    )
    manifest_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest["manifest_content_sha256"] = manifest_sha256

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_source_sha256": cli_sha256,
        "structural_valid": structural_total == 0,
        "all_physical_pass": bool(
            valid and all(e["physical_success"] for e in valid)
        ),
        "require_all_pass": bool(arguments.require_all_pass),
        "posthoc_note": (
            "posthoc pose/table/nut data are evidence-only and never "
            "recompute or alter episode PASS"
        ),
        "episodes": [
            {
                "directory": e["directory"],
                "kit_log": e["kit_log"],
                "seed": e["seed"],
                "structural_ok": e["structural_ok"],
                "structural_problems": e["structural_problems"],
                "physical_success": e["physical_success"],
                "failure_record": e["failure_record"],
                "metrics": e["metrics"],
                "unavailable_metrics": e["unavailable_metrics"],
                "steps_cross": e["steps_cross"],
            }
            for e in episodes
        ],
        "statistics": statistics,
        "input_manifest_sha256": manifest_sha256,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "input_manifest.json"
    markdown_path = output_dir / "SUMMARY_CN.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    lines = ["# G4 逐指柔顺离线汇总（只读）", ""]
    lines.append("结构有效：%s" % summary["structural_valid"])
    lines.append(
        "物理成功：%d/%d（drop=%d）"
        % (
            statistics["physical_success_count"],
            statistics["structural_valid_count"],
            drop_count,
        )
    )
    for e in episodes:
        lines.append(
            "- seed%s %s：结构%s，物理%s"
            % (
                e["seed"],
                Path(e["directory"]).name,
                (
                    "有效"
                    if e["structural_ok"]
                    else "违规: " + "; ".join(e["structural_problems"])
                ),
                "成功" if e["physical_success"] else "失败",
            )
        )
    lines.append("")
    lines.append(
        "失败原因分布："
        + json.dumps(failure_distribution, ensure_ascii=False)
    )
    lines.append(
        "接触顺序分布：" + json.dumps(contact_orders, ensure_ascii=False)
    )
    lines.append("")
    lines.append(
        "P95 方法：NumPy linear 默认；N=%d，%s"
        % (
            len(valid),
            (
                "PRELIMINARY（N<30）"
                if len(valid) < 30
                else "正式样本量待扩大"
            ),
        )
    )
    lines.append("posthoc 位姿/桌面/nut 数据仅证据，绝不重算 PASS。")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    self_check = []
    try:
        reloaded_summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        reloaded_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        recomputed = hashlib.sha256(
            json.dumps(
                {"inputs": reloaded_manifest.get("inputs", [])},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if recomputed != reloaded_manifest.get("manifest_content_sha256"):
            self_check.append("manifest content hash mismatch")
        if reloaded_summary.get("input_manifest_sha256") != recomputed:
            self_check.append("summary manifest hash mismatch")
    except (ValueError, OSError) as error:
        self_check.append(f"self-check failed: {error}")
    if self_check:
        print("SELF-CHECK FAILED: " + "; ".join(self_check))
        return 1

    if structural_total != 0:
        return 1
    if arguments.require_all_pass and not summary["all_physical_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
