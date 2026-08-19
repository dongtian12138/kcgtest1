"""Pure GUI/headless single-finger functional/structural consistency evaluator.

Frozen exit semantics (022a Codex review):
  0 = functional/structural GUI-headless consistency passed;
      quantitative equivalence explicitly NOT claimed
  1 = valid evidence but physical/functional structure mismatch
  2 = input/schema/provenance contract invalid
  3 = evidence incomplete/inconclusive, or the caller requests an
      unregistered quantitative equivalence claim

This evaluator reuses the strict per-step schema and the strict capture
contact gates of the frozen capture/skip audit module; it defines no
looser second set.  The capture/skip comparator keeps its own frozen
exit semantics unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from kcg_connector.grasp.single_finger_posthoc_audit import (
    AUDIT_POINTS,
    FINGERS,
    SHA256_PATTERN,
    _first_diff,
    _is_finite_number,
    evaluate_audit_point,
    verify_capture_episode_markers_classified,
    verify_episode_safety_contract,
    verify_report_terminal_schema,
    verify_step_schema,
)

CONTROL_PHASE = "single_finger_contact_characterization"

TRANSITION_SKELETON = (
    ("APPROACH", "CONTACT_CANDIDATE"),
    ("CONTACT_CANDIDATE", "CONTACT_CONFIRMED"),
    ("CONTACT_CONFIRMED", "SOFT_HOLD"),
    ("SOFT_HOLD", "RELEASE_COMMANDED"),
    ("RELEASE_COMMANDED", "RELEASE_CONFIRMED"),
)

CONFIRM_DEBOUNCE_STEPS = 12
SOFT_HOLD_STEPS = 24
RELEASE_CONFIRM_STEPS = 18
SOFT_HOLD_STIFFNESS_SCALE = 0.35
RELEASE_STEP_RAD = 0.00075

PROVENANCE_SHA_KEYS = (
    "finger_contact_detector_sha256",
    "payload_sha256",
    "physical_grasp_config_sha256",
    "pick_config_sha256",
    "runner_sha256",
    "single_finger_contact_test_sha256",
    "single_finger_posthoc_audit_compare_sha256",
    "single_finger_posthoc_audit_sha256",
    "tabletop_scene_config_sha256",
    "wrapper_sha256",
)

FINGER_CHANNELS = ("f1j2", "f2j1", "f3j2")
WRIST_CHANNELS = ("fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm")

FINGER_BLOCK = "finger_root_torque_proxy_baseline_statistics"
WRIST_RAW_BLOCK = "formal_empty_wrist_reference_statistics_raw"
WRIST_RESIDUAL_BLOCK = "formal_empty_wrist_reference_statistics_residual"
STATS_BLOCKS = (FINGER_BLOCK, WRIST_RAW_BLOCK, WRIST_RESIDUAL_BLOCK)

VALID_OBSERVATION_STATES = {
    "APPROACH",
    "CONTACT_CANDIDATE",
    "CONTACT_CONFIRMED",
    "SOFT_HOLD",
    "RELEASE_COMMANDED",
    "RELEASE_CONFIRMED",
}

SCHEMA_VERSION = "kcg_single_finger_gui_consistency_v1"


def _control_records(
    steps: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in steps
        if record.get("phase") == CONTROL_PHASE
    ]


def _l1_problems(report: Mapping[str, Any], label: str) -> list[str]:
    problems: list[str] = []
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        problems.append(f"{label}_provenance_missing")
    else:
        for key in PROVENANCE_SHA_KEYS:
            value = provenance.get(key)
            if (
                not isinstance(value, str)
                or not SHA256_PATTERN.match(value)
            ):
                problems.append(f"{label}_provenance_{key}_invalid")
        if (
            isinstance(provenance.get("seed"), bool)
            or not isinstance(provenance.get("seed"), int)
        ):
            problems.append(f"{label}_provenance_seed_invalid")
        if provenance.get("audit_mode") != "capture":
            problems.append(f"{label}_provenance_audit_mode_not_capture")
        if provenance.get("finger") not in FINGERS:
            problems.append(f"{label}_provenance_finger_invalid")
        else:
            single = report.get("single_finger") or {}
            if single.get("selected_finger") != provenance.get("finger"):
                problems.append(
                    f"{label}_provenance_finger_vs_report_mismatch"
                )
    if not isinstance(report.get("gui"), bool):
        problems.append(f"{label}_gui_not_bool")
    if report.get("physical_grasp_method") != "single-finger":
        problems.append(f"{label}_method_not_single_finger")
    if report.get("formal_lift_mode") != "zero-lift-hold":
        problems.append(f"{label}_lift_mode_not_zero_lift_hold")
    audit = report.get("posthoc_audit")
    if not isinstance(audit, Mapping) or audit.get("mode") != "capture":
        problems.append(f"{label}_posthoc_audit_not_capture")
    runtime = report.get("runtime_mode_evidence")
    if not isinstance(runtime, Mapping):
        problems.append(f"{label}_runtime_mode_evidence_missing")
    else:
        if not isinstance(runtime.get("report_gui"), bool):
            problems.append(f"{label}_runtime_report_gui_not_bool")
        if not isinstance(runtime.get("world_step_render_enabled"), bool):
            problems.append(f"{label}_runtime_render_enabled_not_bool")
    problems.extend(
        verify_episode_safety_contract(report, expect_read_contact=True)
    )
    return problems


def _runtime_projection(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, Mapping):
        return None
    return {
        "physics_rate_hz": block.get("physics_rate_hz"),
        "world_physics_dt_s": block.get("world_physics_dt_s"),
        "world_physics_dt_readback_s": block.get(
            "world_physics_dt_readback_s"
        ),
        "world_rendering_dt_s": block.get("world_rendering_dt_s"),
        "world_rendering_dt_readback_s": block.get(
            "world_rendering_dt_readback_s"
        ),
        "open_tare_duration_s": block.get("open_tare_duration_s"),
        "tare_actual_steps": block.get("tare_actual_steps"),
        "wrist_tare_actual_samples": block.get(
            "wrist_tare_actual_samples"
        ),
        "controller_updates_per_physics_step": block.get(
            "controller_updates_per_physics_step"
        ),
        "solver_substep_contract": block.get("solver_substep_contract"),
        "dt_readback_status": block.get("dt_readback_status"),
    }


def _l1_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    single = report.get("single_finger") or {}
    audit = report.get("posthoc_audit") or {}
    runtime = report.get("runtime_mode_evidence") or {}
    return {
        "provenance": report.get("provenance"),
        "seed": report.get("seed"),
        "method": report.get("physical_grasp_method"),
        "lift_mode": report.get("formal_lift_mode"),
        "attachment": report.get("attachment"),
        "torque_channels": report.get("torque_channels"),
        "authoring": report.get("d38999_authoring"),
        "realized_randomization": report.get("realized_randomization"),
        "realized_usd_authoring": report.get("realized_usd_authoring"),
        "proxy_collision_filter": report.get("proxy_collision_filter"),
        "contract": report.get("physical_grasp_contract"),
        "release_budget": report.get(
            "single_finger_release_budget_preflight"
        ),
        "finger_mapping": {
            "selected_finger": single.get("selected_finger"),
            "selected_joint_name": single.get("selected_joint_name"),
            "selected_hand_local_index": single.get(
                "selected_hand_local_index"
            ),
            "selected_robot_dof_index": single.get(
                "selected_robot_dof_index"
            ),
        },
        "release_threshold_nm": (single.get("release_conditions") or {}).get(
            "release_threshold_nm"
        ),
        "safety": {
            "process_exit_code": report.get("process_exit_code"),
            "passed": report.get("passed"),
            "grasp_success_claimed": report.get("grasp_success_claimed"),
            "control_reads_object_truth": report.get(
                "control_reads_object_truth"
            ),
            "control_reads_contact_report": report.get(
                "control_reads_contact_report"
            ),
            "formal_truth_firewall_enabled": report.get(
                "formal_truth_firewall_enabled"
            ),
            "object_pose_writes_after_start": report.get(
                "object_pose_writes_after_start"
            ),
            "posthoc_audit_consumed_by_control": report.get(
                "posthoc_audit_consumed_by_control"
            ),
            "posthoc_audit_reads_contact_report": report.get(
                "posthoc_audit_reads_contact_report"
            ),
        },
        "posthoc_mode": audit.get("mode"),
        "runtime_contract": _runtime_projection(runtime),
    }


def _l1_pair_problems(
    headless_report: Mapping[str, Any],
    gui_report: Mapping[str, Any],
) -> tuple[list[str], Any]:
    problems: list[str] = []
    if headless_report.get("gui") is not False or gui_report.get(
        "gui"
    ) is not True:
        problems.append("gui_flag_not_false_true_pair")
    for label, report in (
        ("headless", headless_report),
        ("gui", gui_report),
    ):
        runtime = report.get("runtime_mode_evidence")
        if isinstance(runtime, Mapping):
            if runtime.get("report_gui") is not report.get("gui"):
                problems.append(f"{label}_runtime_report_gui_mismatch")
            if runtime.get("world_step_render_enabled") is not report.get(
                "gui"
            ):
                problems.append(f"{label}_runtime_render_enabled_mismatch")
    first_mismatch = _first_diff(
        _l1_projection(headless_report),
        _l1_projection(gui_report),
    )
    return problems, first_mismatch


def _schema_problems(
    report: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    label: str,
) -> list[str]:
    problems: list[str] = []
    problems.extend(verify_report_terminal_schema(report))
    finger = (report.get("single_finger") or {}).get("selected_finger", "")
    control = _control_records(steps)
    if not control:
        problems.append(f"{label}_no_control_records")
    for index, record in enumerate(control):
        problems.extend(
            verify_step_schema(record, finger, f"{label}_step_{index}")
        )
    marker_schema_problems, _, _ = (
        verify_capture_episode_markers_classified(report, control, label)
    )
    problems.extend(marker_schema_problems)
    return problems


def _l2_episode_problems(
    report: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    label: str,
) -> list[str]:
    problems: list[str] = []
    control = _control_records(steps)
    single = report.get("single_finger") or {}
    finger = single.get("selected_finger", "")
    if not control:
        problems.append(f"{label}_no_control_records")
        return problems
    binding_identity_ok = bool(
        (report.get("realized_usd_authoring") or {})
        .get("material_binding_identity", {})
        .get("all_bindings_ok")
        is True
    )
    audit = report.get("posthoc_audit") or {}
    points = (
        audit.get("points") or {} if isinstance(audit, Mapping) else {}
    )
    for point in AUDIT_POINTS:
        raw = points.get(point)
        if not isinstance(raw, Mapping):
            problems.append(f"{label}_audit_point_missing_{point}")
            continue
        evaluated = evaluate_audit_point(
            point,
            raw.get("snapshot") or {},
            finger,
            binding_identity_ok=binding_identity_ok,
        )
        if not evaluated.get("passed"):
            problems.append(f"{label}_contact_gate_failed_{point}")
    events = single.get("transition_events")
    if not isinstance(events, list) or len(events) != len(
        TRANSITION_SKELETON
    ):
        problems.append(f"{label}_transition_skeleton_wrong")
    else:
        for index, (expected_from, expected_to) in enumerate(
            TRANSITION_SKELETON
        ):
            event = events[index]
            if (
                not isinstance(event, Mapping)
                or event.get("from") != expected_from
                or event.get("to") != expected_to
            ):
                problems.append(
                    f"{label}_transition_skeleton_wrong_at_{index}"
                )
                break
    for index, record in enumerate(control):
        if record.get("other_fingers_open_target_invariant") is not True:
            problems.append(
                f"{label}_other_finger_invariant_false_at_{index}"
            )
            break
        if record.get("failed") is not False:
            problems.append(f"{label}_failed_flag_true_at_{index}")
            break
        if record.get("failure_reason") is not None:
            problems.append(f"{label}_failure_reason_set_at_{index}")
            break
        observation = record.get("observation")
        if isinstance(observation, Mapping) and observation.get(
            "state"
        ) not in VALID_OBSERVATION_STATES:
            problems.append(f"{label}_observation_state_invalid_at_{index}")
            break
    confirm_index = next(
        (
            index
            for index, record in enumerate(control)
            if record.get("state") == "SOFT_HOLD"
            and record.get("soft_hold_step") == 0
        ),
        None,
    )
    if confirm_index is None:
        problems.append(f"{label}_confirm_record_missing")
        return problems
    confirm = control[confirm_index]
    confirm_observation = confirm.get("observation") or {}
    if confirm_observation.get("candidate_steps") != CONFIRM_DEBOUNCE_STEPS:
        problems.append(
            f"{label}_confirm_debounce_not_{CONFIRM_DEBOUNCE_STEPS}"
        )
    if not any(
        (record.get("observation") or {}).get("state") == "CONTACT_CANDIDATE"
        for record in control
    ):
        problems.append(f"{label}_candidate_state_absent_in_trace")
    else:
        # Inclusive debounce semantics: the 11 candidate updates preceding
        # the confirm record must carry candidate_steps 1..11 and the
        # confirm record itself carries 12 (the 12th candidate update is
        # the one that transitions to CONTACT_CONFIRMED).
        for offset in range(1, CONFIRM_DEBOUNCE_STEPS):
            record = (
                control[confirm_index - offset]
                if confirm_index - offset >= 0
                else None
            )
            observation = (record or {}).get("observation") or {}
            if (
                observation.get("state") != "CONTACT_CANDIDATE"
                or observation.get("candidate_steps")
                != CONFIRM_DEBOUNCE_STEPS - offset
            ):
                problems.append(f"{label}_confirm_debounce_sequence_wrong")
                break
    hold_records = control[
        confirm_index : confirm_index + SOFT_HOLD_STEPS + 1
    ]
    if len(hold_records) != SOFT_HOLD_STEPS + 1 or any(
        record.get("state") != "SOFT_HOLD" for record in hold_records
    ):
        problems.append(f"{label}_hold_window_incomplete")
    else:
        for offset, record in enumerate(hold_records):
            if record.get("soft_hold_step") != offset:
                problems.append(
                    f"{label}_hold_step_sequence_wrong_at_{offset}"
                )
                break
    next_index = confirm_index + SOFT_HOLD_STEPS + 1
    if next_index < len(control):
        next_record = control[next_index]
        if (
            next_record.get("state") != "RELEASE_COMMANDED"
            or next_record.get("release_step") != 1
        ):
            problems.append(f"{label}_release_not_started_next_update")
    else:
        problems.append(f"{label}_release_command_missing")
    hold_targets = {
        record.get("selected_target_rad") for record in hold_records
    }
    if len(hold_targets) != 1 or None in hold_targets:
        problems.append(f"{label}_hold_target_not_exactly_frozen")
    hold_stiffness = {
        record.get("selected_stiffness_scale") for record in hold_records
    }
    if hold_stiffness != {SOFT_HOLD_STIFFNESS_SCALE}:
        problems.append(
            f"{label}_hold_stiffness_not_{SOFT_HOLD_STIFFNESS_SCALE}"
        )
    frozen_target = (
        next(iter(hold_targets)) if len(hold_targets) == 1 else None
    )
    release_records = [
        record
        for record in control
        if record.get("state") == "RELEASE_COMMANDED"
    ]
    previous_target = None
    any_strict_decrease = False
    for index, record in enumerate(release_records):
        target = record.get("selected_target_rad")
        if not _is_finite_number(target):
            problems.append(f"{label}_release_target_not_finite_at_{index}")
            break
        if index == 0:
            start_target = (record.get("controller_evidence") or {}).get(
                "release_start_target_rad"
            )
            if frozen_target is not None and start_target != frozen_target:
                problems.append(
                    f"{label}_release_start_target_not_hold_target"
                )
        else:
            if target > previous_target:
                problems.append(
                    f"{label}_release_target_not_monotonic_toward_open"
                )
                break
            delta = float(previous_target) - float(target)
            if not (
                math.isclose(delta, RELEASE_STEP_RAD, abs_tol=1e-12)
                or math.isclose(delta, 0.0, abs_tol=1e-12)
            ):
                problems.append(
                    f"{label}_release_target_step_not_{RELEASE_STEP_RAD}"
                )
                break
            if delta > 0.0:
                any_strict_decrease = True
        if record.get("release_step") != index + 1:
            problems.append(f"{label}_release_step_sequence_wrong_at_{index}")
            break
        previous_target = target
    if release_records and not any_strict_decrease:
        problems.append(f"{label}_release_travel_never_strictly_decreases")
    final = control[-1]
    if final.get("state") != "RELEASE_CONFIRMED":
        problems.append(f"{label}_final_state_not_release_confirmed")
    else:
        final_evidence = final.get("controller_evidence") or {}
        if final_evidence.get("release_confirm") != RELEASE_CONFIRM_STEPS:
            problems.append(
                f"{label}_release_confirm_not_{RELEASE_CONFIRM_STEPS}"
            )
        if final.get("release_step") != single.get("release_step"):
            problems.append(f"{label}_final_release_step_vs_report_mismatch")
        if final.get("detector_test_passed") is not True:
            problems.append(f"{label}_detector_test_passed_not_true")
        conditions = final.get("release_conditions")
        if not isinstance(conditions, Mapping):
            problems.append(f"{label}_final_release_conditions_missing")
        else:
            for name in ("load_ok", "travel_ok", "tracking_ok"):
                if conditions.get(name) is not True:
                    problems.append(
                        f"{label}_final_release_gate_{name}_false"
                    )
            for name in (
                "absolute_load_nm",
                "release_threshold_nm",
                "travel_rad",
                "tracking_error_rad",
            ):
                if not _is_finite_number(conditions.get(name)):
                    problems.append(f"{label}_final_release_{name}_invalid")
    _, marker_functional_problems, _ = (
        verify_capture_episode_markers_classified(report, control, label)
    )
    problems.extend(marker_functional_problems)
    return problems


def _evidence_problems(
    headless_report: Mapping[str, Any],
    gui_report: Mapping[str, Any],
) -> list[str]:
    problems: list[str] = []
    counts: dict[tuple[str, str], int] = {}
    for label, report in (
        ("headless", headless_report),
        ("gui", gui_report),
    ):
        blocks = {name: report.get(name) for name in STATS_BLOCKS}
        for name in STATS_BLOCKS:
            block = blocks[name]
            if not isinstance(block, Mapping):
                problems.append(f"{label}_missing_{name}")
                continue
            if block.get("evidence_only") is not True:
                problems.append(f"{label}_{name}_not_evidence_only")
            if block.get("threshold_label") != "SIM_TUNING_ONLY":
                problems.append(f"{label}_{name}_threshold_label_wrong")
            count = block.get("sample_count")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 2
            ):
                problems.append(f"{label}_{name}_sample_count_invalid")
            else:
                counts[(label, name)] = count
            per_channel = block.get("per_channel")
            expected_channels = (
                FINGER_CHANNELS if name == FINGER_BLOCK else WRIST_CHANNELS
            )
            if (
                not isinstance(per_channel, Mapping)
                or set(per_channel) != set(expected_channels)
            ):
                problems.append(f"{label}_{name}_channels_wrong")
                continue
            for channel in expected_channels:
                stats = per_channel.get(channel)
                if not isinstance(stats, Mapping):
                    problems.append(f"{label}_{name}_{channel}_missing")
                    continue
                for stat_name in (
                    "mean",
                    "std",
                    "rms",
                    "min",
                    "max",
                    "first_half_mean",
                    "second_half_mean",
                    "first_to_second_half_drift",
                ):
                    if not _is_finite_number(stats.get(stat_name)):
                        problems.append(
                            f"{label}_{name}_{channel}_{stat_name}_not_finite"
                        )
                if (
                    isinstance(stats.get("std"), (int, float))
                    and not isinstance(stats.get("std"), bool)
                    and float(stats.get("std")) < 0.0
                ):
                    problems.append(f"{label}_{name}_{channel}_std_negative")
        present = [name for name in STATS_BLOCKS if (label, name) in counts]
        if present and len({counts[(label, name)] for name in present}) > 1:
            problems.append(f"{label}_stats_counts_inconsistent")
        residual = blocks[WRIST_RESIDUAL_BLOCK]
        if isinstance(residual, Mapping):
            if residual.get("baseline_subtraction") != "window_mean":
                problems.append(
                    f"{label}_residual_baseline_subtraction_wrong"
                )
            residual_channels = residual.get("per_channel")
            if isinstance(residual_channels, Mapping):
                for channel, stats in residual_channels.items():
                    if (
                        isinstance(stats, Mapping)
                        and _is_finite_number(stats.get("mean"))
                        and abs(float(stats["mean"])) > 1e-9
                    ):
                        problems.append(
                            f"{label}_residual_mean_not_zero_{channel}"
                        )
        raw = blocks[WRIST_RAW_BLOCK]
        baseline = report.get("formal_pregrasp_empty_wrist_baseline")
        if not (
            isinstance(baseline, (list, tuple))
            and len(baseline) == len(WRIST_CHANNELS)
        ):
            problems.append(f"{label}_wrist_baseline_invalid")
        elif isinstance(raw, Mapping) and isinstance(
            raw.get("per_channel"), Mapping
        ):
            for index, channel in enumerate(WRIST_CHANNELS):
                stats = raw["per_channel"].get(channel)
                if not isinstance(stats, Mapping):
                    continue
                mean = stats.get("mean")
                if not _is_finite_number(mean):
                    continue
                baseline_value = baseline[index]
                if not _is_finite_number(baseline_value):
                    problems.append(
                        f"{label}_wrist_baseline_{channel}_not_finite"
                    )
                    continue
                if not math.isclose(
                    float(mean),
                    float(baseline_value),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    problems.append(
                        f"{label}_raw_mean_vs_report_baseline_{channel}"
                    )
    for name in STATS_BLOCKS:
        headless_count = counts.get(("headless", name))
        gui_count = counts.get(("gui", name))
        if (
            headless_count is not None
            and gui_count is not None
            and headless_count != gui_count
        ):
            problems.append(f"cross_mode_{name}_counts_differ")
    return problems


def _l3_statistics(
    headless_report: Mapping[str, Any],
    headless_steps: Sequence[Mapping[str, Any]],
    gui_report: Mapping[str, Any],
    gui_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def side(report: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]):
        control = _control_records(steps)
        single = report["single_finger"]
        finger = single["selected_finger"]
        confirm = next(
            (
                record
                for record in control
                if record.get("state") == "SOFT_HOLD"
                and record.get("soft_hold_step") == 0
            ),
            None,
        )
        candidate = next(
            (
                record
                for record in control
                if (record.get("observation") or {}).get("state")
                == "CONTACT_CANDIDATE"
            ),
            None,
        )
        hold_complete = next(
            (
                record
                for record in control
                if record.get("state") == "SOFT_HOLD"
                and record.get("soft_hold_step") == SOFT_HOLD_STEPS
            ),
            None,
        )
        points = report["posthoc_audit"]["points"]
        confirm_snapshot = points["contact_confirmed"]["snapshot"]
        hold_snapshot = points["soft_hold_complete"]["snapshot"]
        by_step = {record.get("global_step"): record for record in control}
        marker_wrench = {}
        for point in AUDIT_POINTS:
            marker = points.get(point) or {}
            record = by_step.get(marker.get("global_step"))
            if record is not None:
                marker_wrench[point] = record.get("wrist_wrench_canonical")
        confirm_observation = (confirm or {}).get("observation") or {}
        return {
            "candidate_step": (
                (candidate or {}).get("observation", {}).get("step")
            ),
            "confirm_step": confirm_observation.get("step"),
            "confirm_target_rad": (confirm or {}).get(
                "selected_target_rad"
            ),
            "confirm_q_rad": (confirm or {}).get("selected_q_rad"),
            "confirm_qd_rad_s": (confirm or {}).get("selected_qd_rad_s"),
            "confirm_raw_delta_nm": confirm_observation.get("raw_delta_nm"),
            "confirm_filtered_delta_nm": confirm_observation.get(
                "filtered_delta_nm"
            ),
            "hold_complete_target_rad": (hold_complete or {}).get(
                "selected_target_rad"
            ),
            "hold_complete_q_rad": (hold_complete or {}).get(
                "selected_q_rad"
            ),
            "contact_records_confirm_body": confirm_snapshot[
                "finger_body_group_records"
            ][finger]["body"],
            "contact_records_confirm_nut": confirm_snapshot[
                "finger_body_group_records"
            ][finger]["nut"],
            "contact_records_hold_body": hold_snapshot[
                "finger_body_group_records"
            ][finger]["body"],
            "contact_records_hold_nut": hold_snapshot[
                "finger_body_group_records"
            ][finger]["nut"],
            "release_step": single.get("release_step"),
            "release_travel_rad": (single.get("release_conditions") or {}).get(
                "travel_rad"
            ),
            "release_tracking_error_rad": (single.get(
                "release_conditions"
            ) or {}).get("tracking_error_rad"),
            "release_load_nm": (single.get("release_conditions") or {}).get(
                "absolute_load_nm"
            ),
            "finger_baseline_mean": {
                channel: report[FINGER_BLOCK]["per_channel"][channel]["mean"]
                for channel in FINGER_CHANNELS
            },
            "finger_baseline_std": {
                channel: report[FINGER_BLOCK]["per_channel"][channel]["std"]
                for channel in FINGER_CHANNELS
            },
            "wrist_raw_mean": {
                channel: report[WRIST_RAW_BLOCK]["per_channel"][channel][
                    "mean"
                ]
                for channel in WRIST_CHANNELS
            },
            "wrist_raw_std": {
                channel: report[WRIST_RAW_BLOCK]["per_channel"][channel][
                    "std"
                ]
                for channel in WRIST_CHANNELS
            },
            "wrist_residual_rms": {
                channel: report[WRIST_RESIDUAL_BLOCK]["per_channel"][
                    channel
                ]["rms"]
                for channel in WRIST_CHANNELS
            },
            "marker_wrench_canonical": marker_wrench,
        }

    headless_values = side(headless_report, headless_steps)
    gui_values = side(gui_report, gui_steps)
    scalar_names = [
        "candidate_step",
        "confirm_step",
        "confirm_target_rad",
        "confirm_q_rad",
        "confirm_qd_rad_s",
        "confirm_raw_delta_nm",
        "confirm_filtered_delta_nm",
        "hold_complete_target_rad",
        "hold_complete_q_rad",
        "contact_records_confirm_body",
        "contact_records_confirm_nut",
        "contact_records_hold_body",
        "contact_records_hold_nut",
        "release_step",
        "release_travel_rad",
        "release_tracking_error_rad",
        "release_load_nm",
    ]
    channel_map_names = [
        "finger_baseline_mean",
        "finger_baseline_std",
        "wrist_raw_mean",
        "wrist_raw_std",
        "wrist_residual_rms",
    ]
    values: dict[str, Any] = {}
    for name in scalar_names:
        headless_value = headless_values.get(name)
        gui_value = gui_values.get(name)
        delta = None
        if (
            _is_finite_number(headless_value)
            and _is_finite_number(gui_value)
        ):
            delta = float(gui_value) - float(headless_value)
        values[name] = {
            "headless": headless_value,
            "gui": gui_value,
            "delta": delta,
        }
    for name in channel_map_names:
        headless_map = headless_values.get(name) or {}
        gui_map = gui_values.get(name) or {}
        values[name] = {
            "headless": dict(headless_map),
            "gui": dict(gui_map),
            "delta": {
                channel: (
                    None
                    if (
                        not _is_finite_number(headless_map.get(channel))
                        or not _is_finite_number(gui_map.get(channel))
                    )
                    else float(gui_map.get(channel))
                    - float(headless_map.get(channel))
                )
                for channel in sorted(
                    set(headless_map) | set(gui_map)
                )
            },
        }
    marker_wrench = {}
    for point in AUDIT_POINTS:
        headless_wrench = headless_values["marker_wrench_canonical"].get(
            point
        )
        gui_wrench = gui_values["marker_wrench_canonical"].get(point)
        delta = None
        if (
            isinstance(headless_wrench, (list, tuple))
            and isinstance(gui_wrench, (list, tuple))
            and len(headless_wrench) == len(gui_wrench)
        ):
            delta = [
                (
                    None
                    if (
                        not _is_finite_number(headless_wrench[index])
                        or not _is_finite_number(gui_wrench[index])
                    )
                    else float(gui_wrench[index])
                    - float(headless_wrench[index])
                )
                for index in range(len(headless_wrench))
            ]
        marker_wrench[point] = {
            "headless": headless_wrench,
            "gui": gui_wrench,
            "delta": delta,
        }
    sample_counts = {}
    for label, report in (
        ("headless", headless_report),
        ("gui", gui_report),
    ):
        sample_counts[label] = {
            "finger": report.get(FINGER_BLOCK, {}).get("sample_count"),
            "wrist": report.get(WRIST_RAW_BLOCK, {}).get("sample_count"),
        }
    return {
        "schema_version": "kcg_single_finger_gui_consistency_v1",
        "quantitative_equivalence_claimed": False,
        "quantitative_gates_registered": False,
        "note": (
            "n=1 per mode: all numbers below are evidence only; no "
            "quantitative cross-mode gate is registered"
        ),
        "sample_counts": sample_counts,
        "values": values,
        "marker_wrench_canonical": marker_wrench,
    }


def compare_gui_headless_consistency(
    headless_report: Mapping[str, Any],
    headless_steps: Sequence[Mapping[str, Any]],
    gui_report: Mapping[str, Any],
    gui_steps: Sequence[Mapping[str, Any]],
    *,
    request_quantitative_gates: bool = False,
) -> dict[str, Any]:
    """Compare one headless capture episode against one GUI capture episode.

    L1 (static/safety/provenance) and schema problems exit 2; L2 physical/
    functional structure mismatches exit 1; incomplete or inconsistent
    evidence (or an unregistered quantitative request) exits 3; only a full
    L1+L2 pass with healthy evidence exits 0, and that exit claims ONLY
    functional/structural consistency -- quantitative equivalence is
    explicitly not claimed.

    Malformed input must never raise out of this function: any unexpected
    evaluation exception is converted into a fail-closed exit-2 summary, so
    the CLI always emits valid JSON instead of a traceback.
    """
    try:
        return _evaluate_gui_headless_consistency(
            headless_report,
            headless_steps,
            gui_report,
            gui_steps,
            request_quantitative_gates=request_quantitative_gates,
        )
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "functional_structural_gui_headless_consistency_passed": False,
            "quantitative_equivalence_claimed": False,
            "quantitative_gates_registered": False,
            "exit_code": 2,
            "failure_reason": "input_schema_or_provenance_contract_invalid",
            "input_problems": [
                f"unexpected_evaluation_error_{type(error).__name__}"
            ],
            "schema_problems": [],
            "headless_functional_structural_problems": [],
            "gui_functional_structural_problems": [],
            "evidence_problems": [],
            "first_l1_mismatch": None,
            "quantitative_statistics": None,
        }


def _evaluate_gui_headless_consistency(
    headless_report: Mapping[str, Any],
    headless_steps: Sequence[Mapping[str, Any]],
    gui_report: Mapping[str, Any],
    gui_steps: Sequence[Mapping[str, Any]],
    *,
    request_quantitative_gates: bool,
) -> dict[str, Any]:
    input_problems: list[str] = []
    input_problems.extend(_l1_problems(headless_report, "headless"))
    input_problems.extend(_l1_problems(gui_report, "gui"))
    pair_problems, first_l1_mismatch = _l1_pair_problems(
        headless_report, gui_report
    )
    input_problems.extend(pair_problems)
    if first_l1_mismatch is not None:
        input_problems.append("l1_projection_mismatch")
    schema_problems: list[str] = []
    schema_problems.extend(
        _schema_problems(headless_report, headless_steps, "headless")
    )
    schema_problems.extend(
        _schema_problems(gui_report, gui_steps, "gui")
    )
    headless_functional = _l2_episode_problems(
        headless_report, headless_steps, "headless"
    )
    gui_functional = _l2_episode_problems(
        gui_report, gui_steps, "gui"
    )
    evidence_problems = _evidence_problems(headless_report, gui_report)
    if input_problems or schema_problems:
        exit_code = 2
        failure_reason = "input_schema_or_provenance_contract_invalid"
    elif headless_functional or gui_functional:
        exit_code = 1
        failure_reason = "functional_or_structural_mismatch"
    elif request_quantitative_gates:
        exit_code = 3
        failure_reason = (
            "quantitative_equivalence_requested_without_registered_gates"
        )
    elif evidence_problems:
        exit_code = 3
        failure_reason = "evidence_incomplete_or_inconsistent"
    else:
        exit_code = 0
        failure_reason = None
    statistics = None
    if exit_code == 0:
        statistics = _l3_statistics(
            headless_report, headless_steps, gui_report, gui_steps
        )
    return {
        "schema_version": "kcg_single_finger_gui_consistency_v1",
        "functional_structural_gui_headless_consistency_passed": bool(
            exit_code == 0
        ),
        "quantitative_equivalence_claimed": False,
        "quantitative_gates_registered": False,
        "exit_code": exit_code,
        "failure_reason": failure_reason,
        "input_problems": input_problems,
        "schema_problems": schema_problems,
        "headless_functional_structural_problems": headless_functional,
        "gui_functional_structural_problems": gui_functional,
        "evidence_problems": evidence_problems,
        "first_l1_mismatch": (
            None
            if first_l1_mismatch is None
            else {
                "path": list(first_l1_mismatch[0]),
                "headless": first_l1_mismatch[1],
                "gui": first_l1_mismatch[2],
            }
        ),
        "quantitative_statistics": statistics,
    }
