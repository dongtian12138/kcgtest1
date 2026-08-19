#!/usr/bin/env python3
"""Posthoc G3 same-source synchronous-grasp collection summarizer.

Read-only acceptance verifier and statistics summarizer for the final
five-headless + one-GUI synchronous staged grasp collection (G3).  It
never writes into any episode directory and never imports Isaac; it
reuses the pure statistics primitive summarize() from
kcg_connector/grasp/grasp_evidence.py without modifying it.

Fail-closed acceptance per headless episode: unique seed in {0,1,2,3,4},
headless/synchronous/staged, passed and exit 0, all three stage sensor
gates, actual lift >= 45 mm, episode-end contact gate, finite
throughout, three-finger end contact with no table support, no
unexpected contacts, no object-truth/contact-report control reads, zero
pose writes, no attachment/object drive/proxy, and the
runner/monitor/config-loader/YAML provenance hashes equal to the current
on-disk SHA-256 of the real source files.  The kit log for each episode
must contain the GPU evidence markers (Active GPU: NVIDIA GeForce RTX
5070 Ti, the Yes: 0 device line and the Warp cuda:0 token); a
CPU-fallback log is never accepted.  The GUI episode must additionally
share seed, payload hash and the full provenance hash set with the
headless seed-0 episode.

Outputs written to the output directory only:
  summary.json / SUMMARY_CN.md / input_manifest.json

Explicit scope statements recorded in the summary:
  * P95 is computed over n=5 with the NumPy linear default percentile
    and is PRELIMINARY; it is not the G6/30-pair final distribution.
  * continuous_contact_path_verified=false: control never reads contact
    truth; the episode-end contact gate is episode-end acceptance only.
  * G3 completion does not imply sequential-compliant, G4, 5+5, 30-pair
    or vision-stage completion.
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

from kcg_connector.grasp.grasp_evidence import summarize

SCHEMA_VERSION = "kcg_d38999_physical_grasp_g3_summary_v1"
MANIFEST_SCHEMA_VERSION = "kcg_d38999_physical_grasp_g3_input_manifest_v1"

PROVENANCE_IDENTITY_KEYS = (
    "runner_sha256",
    "grasp_stability_monitor_sha256",
    "physical_grasp_config_loader_sha256",
    "physical_grasp_config_sha256",
)

SOURCE_FILES = {
    "runner_sha256": "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py",
    "grasp_stability_monitor_sha256": (
        "src/kcg_connector/kcg_connector/grasp/grasp_stability_monitor.py"
    ),
    "physical_grasp_config_loader_sha256": (
        "src/kcg_connector/kcg_connector/grasp/physical_grasp_config.py"
    ),
    "physical_grasp_config_sha256": (
        "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
    ),
}

GPU_LOG_MARKERS = (
    "Active GPU: NVIDIA GeForce RTX 5070 Ti",
    "Yes: 0",
    '"cuda:0"',
)

STAGE_INCREMENTS_M = (0.002, 0.010, 0.040)
MINIMUM_TOTAL_LIFT_M = 0.045
FINGERS = ("f1", "f2", "f3")

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"source file missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_source_hashes() -> dict[str, str]:
    return {
        key: _sha256_file(REPOSITORY_ROOT / relative)
        for key, relative in SOURCE_FILES.items()
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_report(episode_dir: Path) -> dict[str, Any]:
    report_path = episode_dir / "nominal_physics_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing report: {report_path}")
    document = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"report is not a JSON object: {report_path}")
    return document


def load_contact_order(episode_dir: Path) -> list[str]:
    steps_path = episode_dir / "controller_steps.jsonl"
    if not steps_path.is_file():
        raise FileNotFoundError(f"missing controller steps: {steps_path}")
    final_order: list[str] | None = None
    with steps_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            order = record.get("contact_order")
            if (
                isinstance(order, list)
                and len(order) == len(FINGERS)
                and set(order) == set(FINGERS)
            ):
                final_order = list(order)
    if final_order is None:
        raise ValueError(f"no complete contact_order found in {steps_path}")
    return final_order


def verify_episode(
    report: Mapping[str, Any],
    *,
    expect_gui: bool,
    disk_hashes: Mapping[str, str],
    kit_log_text: str,
    contact_order: Sequence[str] | None,
) -> list[str]:
    problems: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    seed = report.get("seed")
    check(
        isinstance(seed, int) and not isinstance(seed, bool),
        f"seed must be an integer, got {seed!r}",
    )
    check(
        report.get("gui") is expect_gui,
        "gui flag does not match episode role",
    )
    check(
        report.get("physical_grasp_method") == "synchronous",
        "physical_grasp_method must be synchronous",
    )
    check(
        report.get("formal_lift_mode") == "staged",
        "formal_lift_mode must be staged",
    )
    check(
        (report.get("realized_randomization") or {}).get("mode") == "staged",
        "realized randomization mode must be staged",
    )
    check(report.get("process_exit_code") == 0, "process_exit_code must be 0")
    check(report.get("passed") is True, "report passed must be true")

    acceptance = report.get("formal_acceptance") or {}
    check(
        acceptance.get("passed") is True,
        "formal_acceptance.passed must be true",
    )
    check(
        acceptance.get("sensor_lift_gate") is True,
        "formal_acceptance.sensor_lift_gate must be true",
    )
    check(
        acceptance.get("episode_end_contact_gate") is True,
        "formal_acceptance.episode_end_contact_gate must be true",
    )
    check(
        acceptance.get("controller_stable") is True,
        "formal_acceptance.controller_stable must be true",
    )
    check(
        acceptance.get("post_grasp_stabilization_proxy_used") is False,
        "post-grasp stabilization proxy must not be used",
    )
    minimum = acceptance.get("minimum_total_lift_m")
    check(
        _finite_number(minimum)
        and abs(float(minimum) - MINIMUM_TOTAL_LIFT_M) < 1e-12,
        "minimum_total_lift_m must be 0.045",
    )
    actual = acceptance.get("actual_body_lift_m")
    check(
        _finite_number(actual) and float(actual) >= MINIMUM_TOTAL_LIFT_M,
        f"actual_body_lift_m {actual!r} is below 0.045 m",
    )

    stages = report.get("formal_lift_stages")
    check(
        isinstance(stages, list) and len(stages) == 3,
        "must have exactly 3 lift stages",
    )
    if isinstance(stages, list) and len(stages) == 3:
        for index, stage in enumerate(stages):
            check(
                stage.get("passed_sensor_gate") is True,
                f"lift stage {index + 1} sensor gate must pass",
            )
            check(
                _finite_number(stage.get("increment_m"))
                and abs(
                    float(stage["increment_m"]) - STAGE_INCREMENTS_M[index]
                )
                < 1e-12,
                f"lift stage {index + 1} increment must be "
                f"{STAGE_INCREMENTS_M[index]}",
            )

    monitor = report.get("formal_lift_monitor") or {}
    check(
        monitor.get("failed") is False,
        "formal_lift_monitor.failed must be false",
    )
    check(
        monitor.get("failure_reason") is None,
        "monitor failure_reason must be null",
    )
    check(
        isinstance(monitor.get("steps"), int) and monitor["steps"] > 0,
        "monitor steps must be a positive integer",
    )
    for key in ("peak_wrist_force_increment_n", "peak_moment_safety_score_nm"):
        check(
            _finite_number(monitor.get(key)),
            f"monitor {key} must be finite",
        )

    check(
        report.get("finite_throughout") is True,
        "finite_throughout must be true",
    )
    check(report.get("finite_final") is True, "finite_final must be true")
    check(
        report.get("final_tail_diagnostics_finite") is True,
        "final_tail_diagnostics_finite must be true",
    )
    check(
        report.get("final_all_fingers_body_contact") is True,
        "final_all_fingers_body_contact must be true",
    )
    check(
        report.get("zero_forbidden_contacts") is True,
        "zero_forbidden_contacts must be true",
    )
    check(
        report.get("final_unsupported") is False,
        "final_unsupported must be false",
    )

    contacts = report.get("final_contacts") or {}
    check(
        contacts.get("plug_table_records") == 0,
        "plug_table_records must be 0 (no table support)",
    )
    check(
        contacts.get("unexpected_robot_link_records") == 0,
        "unexpected_robot_link_records must be 0",
    )
    finger_body = contacts.get("finger_body_group_records") or {}
    check(
        set(finger_body) == set(FINGERS)
        and all(
            isinstance(entry.get("body"), int) and entry["body"] >= 1
            for entry in finger_body.values()
        ),
        "all three fingers must end with body contact records",
    )
    material = contacts.get("material_evidence") or {}
    check(
        material.get("available") is True,
        "material evidence must be available",
    )
    check(
        isinstance(material.get("grip_grip_records"), int)
        and material["grip_grip_records"] >= 1,
        "grip-grip material records must be present",
    )
    external = report.get("external_contact_records") or {}
    check(
        all(
            isinstance(value, int) and value == 0
            for value in external.values()
        ),
        "external contact records must all be zero",
    )

    check(
        report.get("control_reads_object_truth") is False,
        "control reads object truth",
    )
    check(
        report.get("control_reads_contact_report") is False,
        "control reads contact report",
    )
    check(
        report.get("truth_orientation_used") is False,
        "truth orientation used",
    )
    check(
        report.get("object_pose_writes_after_start") == 0,
        "object pose writes after start must be 0",
    )
    check(report.get("attachment") == "none", "attachment must be none")
    check(report.get("object_drive") == "none", "object drive must be none")
    check(
        (report.get("proxy_collision_filter") or {}).get("enabled") is False,
        "proxy collision filter must be disabled",
    )
    check(
        report.get("posthoc_truth_evaluation_only") is True,
        "posthoc truth must be evaluation-only",
    )
    check(
        report.get("formal_truth_firewall_enabled") is True,
        "formal truth firewall must be enabled",
    )

    provenance = report.get("provenance") or {}
    for key, expected in disk_hashes.items():
        check(
            provenance.get(key) == expected,
            f"provenance {key} does not match the on-disk source SHA-256",
        )

    for marker in GPU_LOG_MARKERS:
        check(
            marker in kit_log_text,
            f"kit log is missing GPU evidence marker {marker!r}",
        )

    if contact_order is not None:
        check(
            len(list(contact_order)) == len(FINGERS)
            and set(contact_order) == set(FINGERS),
            "contact order must cover f1, f2, f3 exactly once",
        )
    return problems


def kit_log_name(seed: int, gui: bool) -> str:
    return f"seed{seed:03d}_{'gui' if gui else 'headless'}_kit.log"


def _series(
    reports: Sequence[Mapping[str, Any]], extractor
) -> list[float]:
    values = []
    for report in reports:
        value = extractor(report)
        if not _finite_number(value):
            raise ValueError(f"non-finite statistic extracted: {value!r}")
        values.append(float(value))
    return values


def _block(
    values: Sequence[float], si_unit: str, readable_unit: str, scale: float
) -> dict[str, Any]:
    signed = summarize(values)
    absolute = summarize([abs(value) for value in values])
    result: dict[str, Any] = {
        "unit": si_unit,
        "readable_unit": readable_unit,
        "count": signed["count"],
    }
    for label, stats in (("signed", signed), ("absolute", absolute)):
        result[label] = {
            key: stats[key] for key in ("mean", "median", "p95", "maximum")
        }
        result[label + "_readable"] = {
            key: (None if stats[key] is None else stats[key] * scale)
            for key in ("mean", "median", "p95", "maximum")
        }
    return result


CHANNELS = ("dx_m", "dy_m", "dz_m", "drx_rad", "dry_rad", "drz_rad")


def _pose_block(
    reports: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for channel in CHANNELS:
        if channel.endswith("_m"):
            unit, readable_unit, scale = "m", "mm", 1000.0
        else:
            unit, readable_unit, scale = "rad", "deg", 180.0 / math.pi
        result[channel] = _block(
            _series(
                reports,
                lambda report, c=channel: report.get(field, {}).get(c),
            ),
            unit,
            readable_unit,
            scale,
        )
    translation = _series(
        reports,
        lambda report: math.sqrt(
            sum(
                float(report.get(field, {}).get(key)) ** 2
                for key in ("dx_m", "dy_m", "dz_m")
            )
        ),
    )
    rotation = _series(
        reports,
        lambda report: math.sqrt(
            sum(
                float(report.get(field, {}).get(key)) ** 2
                for key in ("drx_rad", "dry_rad", "drz_rad")
            )
        ),
    )
    result["translation_norm"] = _block(translation, "m", "mm", 1000.0)
    result["rotation_norm"] = _block(rotation, "rad", "deg", 180.0 / math.pi)
    return result


def build_summary(
    headless_dirs: Sequence[Path],
    headless_reports: Sequence[Mapping[str, Any]],
    headless_problems: Sequence[list[str]],
    headless_contact_orders: Sequence[list[str]],
    gui_dir: Path,
    gui_report: Mapping[str, Any],
    gui_problems: list[str],
    gui_contact_order: list[str],
    disk_hashes: Mapping[str, str],
    manifest_sha256: str,
    cli_sha256: str,
) -> dict[str, Any]:
    accepted = [
        (report, problems, order)
        for report, problems, order in zip(
            headless_reports, headless_problems, headless_contact_orders
        )
        if not problems
    ]
    g3_complete = bool(len(accepted) == 5 and not gui_problems)

    failure_reasons: dict[str, Any] = {}
    for report, problems, _order in zip(
        headless_reports, headless_problems, headless_contact_orders
    ):
        if problems:
            monitor_reason = (report.get("formal_lift_monitor") or {}).get(
                "failure_reason"
            )
            reason = monitor_reason or "verification_rejected"
            failure_reasons.setdefault(f"seed{report['seed']}", []).append(
                {
                    "failure_reason": reason,
                    "process_exit_code": report.get("process_exit_code"),
                    "verification_problems": list(problems),
                }
            )
    gui_failure_reasons: dict[str, Any] = {}
    if gui_problems:
        gui_failure_reasons["seed0_gui"] = list(gui_problems)

    accepted_reports = [report for report, _p, _o in accepted]
    orders = [order for _r, _p, order in accepted]
    order_distribution: dict[str, int] = {}
    first_finger_distribution: dict[str, int] = {}
    for order in orders:
        key = "-".join(order)
        order_distribution[key] = order_distribution.get(key, 0) + 1
        first = order[0]
        first_finger_distribution[first] = first_finger_distribution.get(
            first, 0
        ) + 1

    contact_records = _series(
        accepted_reports,
        lambda report: (report.get("final_contacts") or {})
        .get("material_evidence", {})
        .get("grip_grip_records"),
    )

    statistics: dict[str, Any] = {
        "episode_count": len(headless_reports),
        "accepted_count": len(accepted),
        "rejected_count": len(headless_reports) - len(accepted),
        "gui_accepted": not gui_problems,
        "contact_order_distribution": order_distribution,
        "first_finger_distribution": first_finger_distribution,
        "table_translation_xy": _block(
            _series(
                accepted_reports,
                lambda report: (report.get("table_stage") or {}).get(
                    "translation_xy_m"
                ),
            ),
            "m",
            "mm",
            1000.0,
        ),
        "table_yaw_delta": _block(
            _series(
                accepted_reports,
                lambda report: (report.get("table_stage") or {}).get(
                    "yaw_delta_rad"
                ),
            ),
            "rad",
            "deg",
            180.0 / math.pi,
        ),
        "body_lift": _block(
            _series(
                accepted_reports,
                lambda report: report.get("body_lift_m"),
            ),
            "m",
            "mm",
            1000.0,
        ),
        "body_tcp_slip": _block(
            _series(
                accepted_reports,
                lambda report: report.get("body_tcp_slip_m"),
            ),
            "m",
            "mm",
            1000.0,
        ),
        "body_nut_separation": _block(
            _series(
                accepted_reports,
                lambda report: report.get("body_nut_separation_change_m"),
            ),
            "m",
            "mm",
            1000.0,
        ),
        "posthoc_pose_error": _pose_block(
            accepted_reports, "posthoc_pose_error"
        ),
        "lift_relative_slip": _pose_block(
            accepted_reports, "posthoc_lift_relative_slip"
        ),
        "wrist_force_peak": _block(
            _series(
                accepted_reports,
                lambda report: (report.get("formal_lift_monitor") or {}).get(
                    "peak_wrist_force_increment_n"
                ),
            ),
            "N",
            "N",
            1.0,
        ),
        "moment_safety_score_peak": _block(
            _series(
                accepted_reports,
                lambda report: (report.get("formal_lift_monitor") or {}).get(
                    "peak_moment_safety_score_nm"
                ),
            ),
            "N*m",
            "N*m",
            1.0,
        ),
        "episode_end_grip_contact_records": {
            "per_episode": [
                {"seed": report["seed"], "count": int(value)}
                for report, value in zip(accepted_reports, contact_records)
            ],
            "stats": _block(contact_records, "records", "records", 1.0),
        },
    }

    seed0_index = None
    for index, report in enumerate(headless_reports):
        if report.get("seed") == 0:
            seed0_index = index
            break
    gui_comparison: dict[str, Any] = {}
    if seed0_index is not None and not headless_problems[seed0_index]:
        seed0 = headless_reports[seed0_index]
        gui_comparison["functional"] = {
            "passed_equal": gui_report.get("passed") == seed0.get("passed"),
            "method_equal": gui_report.get("physical_grasp_method")
            == seed0.get("physical_grasp_method"),
            "mode_equal": gui_report.get("formal_lift_mode")
            == seed0.get("formal_lift_mode"),
            "sensor_lift_gate_equal": (
                (gui_report.get("formal_acceptance") or {}).get(
                    "sensor_lift_gate"
                )
                == (seed0.get("formal_acceptance") or {}).get(
                    "sensor_lift_gate"
                )
            ),
            "episode_end_contact_gate_equal": (
                gui_report.get("formal_acceptance") or {}
            )
            .get("episode_end_contact_gate")
            == (seed0.get("formal_acceptance") or {}).get(
                "episode_end_contact_gate"
            ),
            "payload_equal": gui_report["provenance"].get("payload_sha256")
            == seed0["provenance"].get("payload_sha256"),
            "provenance_equal": all(
                gui_report["provenance"].get(key)
                == seed0["provenance"].get(key)
                for key in seed0["provenance"]
            ),
            "boundary_fields_equal": all(
                gui_report.get(key) == seed0.get(key)
                for key in (
                    "control_reads_object_truth",
                    "control_reads_contact_report",
                    "truth_orientation_used",
                    "object_pose_writes_after_start",
                    "attachment",
                    "object_drive",
                )
            ),
            "headless_contact_order": list(
                headless_contact_orders[seed0_index]
            ),
            "gui_contact_order": list(gui_contact_order),
        }
        numeric_metrics = [
            ("body_lift", "body_lift_m", "m", "mm", 1000.0),
            (
                "body_tcp_slip",
                "body_tcp_slip_m",
                "m",
                "mm",
                1000.0,
            ),
            (
                "body_nut_separation",
                "body_nut_separation_change_m",
                "m",
                "mm",
                1000.0,
            ),
            (
                "table_translation_xy",
                ("table_stage", "translation_xy_m"),
                "m",
                "mm",
                1000.0,
            ),
            (
                "table_yaw_delta",
                ("table_stage", "yaw_delta_rad"),
                "rad",
                "deg",
                180.0 / math.pi,
            ),
            (
                "wrist_force_peak",
                ("formal_lift_monitor", "peak_wrist_force_increment_n"),
                "N",
                "N",
                1.0,
            ),
            (
                "moment_safety_score_peak",
                ("formal_lift_monitor", "peak_moment_safety_score_nm"),
                "N*m",
                "N*m",
                1.0,
            ),
        ]
        for channel in CHANNELS:
            if channel.endswith("_m"):
                unit, readable_unit, scale = "m", "mm", 1000.0
            else:
                unit, readable_unit, scale = "rad", "deg", 180.0 / math.pi
            numeric_metrics.append(
                (
                    "pose_error_" + channel,
                    ("posthoc_pose_error", channel),
                    unit,
                    readable_unit,
                    scale,
                )
            )
            numeric_metrics.append(
                (
                    "lift_slip_" + channel,
                    ("posthoc_lift_relative_slip", channel),
                    unit,
                    readable_unit,
                    scale,
                )
            )

        def _nested(report: Mapping[str, Any], path: Any) -> float:
            if isinstance(path, tuple):
                current: Any = report
                for part in path:
                    current = current[part]
                return float(current)
            return float(report[path])

        deltas = []
        for metric, path, unit, readable_unit, scale in numeric_metrics:
            headless = _nested(seed0, path)
            gui = _nested(gui_report, path)
            delta = gui - headless
            deltas.append(
                {
                    "metric": metric,
                    "unit": unit,
                    "headless_seed0": headless,
                    "gui": gui,
                    "delta": delta,
                    "delta_readable": delta * scale,
                    "readable_unit": readable_unit,
                }
            )
        gui_comparison["numeric_deltas"] = deltas

    identity_identical = (
        len(
            {
                tuple(
                    report["provenance"].get(key)
                    for key in PROVENANCE_IDENTITY_KEYS
                )
                for report in headless_reports
            }
        )
        == 1
    )
    payloads = {
        report["seed"]: report["provenance"].get("payload_sha256")
        for report in headless_reports
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_source_sha256": cli_sha256,
        "g3_complete": g3_complete,
        "percentile_method": "numpy_linear_default_np_percentile_95",
        "p95_n_equals_5_preliminary": True,
        "continuous_contact_path_verified": False,
        "continuous_contact_path_note": (
            "control never reads contact truth; the episode-end contact "
            "gate is episode-end acceptance only"
        ),
        "scope_notes": {
            "g3_only": True,
            "sequential_compliant_not_claimed": True,
            "g4_not_claimed": True,
            "five_plus_five_not_claimed": True,
            "thirty_pair_not_claimed": True,
            "vision_stage_not_claimed": True,
            "cpu_fallback_never_accepted": True,
        },
        "verdicts": {
            "headless_episodes": 5,
            "headless_accepted": len(accepted),
            "headless_rejected": len(headless_reports) - len(accepted),
            "gui_accepted": not gui_problems,
            "failure_reasons": failure_reasons,
            "gui_failure_reasons": gui_failure_reasons,
        },
        "episodes": [
            {
                "directory": str(directory),
                "seed": report["seed"],
                "gui": report.get("gui"),
                "accepted": not problems,
                "verification_problems": list(problems),
                "key_values": {
                    "passed": report.get("passed"),
                    "body_lift_m": report.get("body_lift_m"),
                    "body_tcp_slip_m": report.get("body_tcp_slip_m"),
                    "body_nut_separation_change_m": report.get(
                        "body_nut_separation_change_m"
                    ),
                    "wrist_force_peak_n": (
                        report.get("formal_lift_monitor") or {}
                    ).get("peak_wrist_force_increment_n"),
                    "moment_safety_score_peak_nm": (
                        report.get("formal_lift_monitor") or {}
                    ).get("peak_moment_safety_score_nm"),
                    "grip_contact_records": (
                        report.get("final_contacts") or {}
                    )
                    .get("material_evidence", {})
                    .get("grip_grip_records"),
                    "contact_order": list(order),
                    "process_exit_code": report.get("process_exit_code"),
                },
            }
            for directory, report, problems, order in zip(
                headless_dirs,
                headless_reports,
                headless_problems,
                headless_contact_orders,
            )
        ]
        + [
            {
                "directory": str(gui_dir),
                "seed": gui_report.get("seed"),
                "gui": gui_report.get("gui"),
                "accepted": not gui_problems,
                "verification_problems": list(gui_problems),
                "key_values": {
                    "passed": gui_report.get("passed"),
                    "body_lift_m": gui_report.get("body_lift_m"),
                    "body_tcp_slip_m": gui_report.get("body_tcp_slip_m"),
                    "body_nut_separation_change_m": gui_report.get(
                        "body_nut_separation_change_m"
                    ),
                    "wrist_force_peak_n": (
                        gui_report.get("formal_lift_monitor") or {}
                    ).get("peak_wrist_force_increment_n"),
                    "moment_safety_score_peak_nm": (
                        gui_report.get("formal_lift_monitor") or {}
                    ).get("peak_moment_safety_score_nm"),
                    "grip_contact_records": (
                        gui_report.get("final_contacts") or {}
                    )
                    .get("material_evidence", {})
                    .get("grip_grip_records"),
                    "contact_order": list(gui_contact_order),
                    "process_exit_code": gui_report.get("process_exit_code"),
                },
            }
        ],
        "statistics": statistics,
        "source_hashes": {
            "disk": dict(disk_hashes),
            "identical_across_episodes": identity_identical,
            "payloads_by_seed": payloads,
        },
        "gui_vs_headless_seed0": gui_comparison,
        "input_manifest_sha256": manifest_sha256,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def render_summary_cn(summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# G3 最终同源同步抓取集合汇总（正式证据）")
    lines.append("")
    lines.append("生成时间（UTC）：" + str(summary.get("generated_at_utc")))
    lines.append("schema：%s" % SCHEMA_VERSION)
    lines.append(
        "P95 方法：NumPy linear 默认（np.percentile(..., 95)），n=5，"
        "**PRELIMINARY** —— 不是 G6/30-pair 最终分布。"
    )
    verdict = summary.get("g3_complete")
    if verdict:
        lines.append("")
        lines.append(
            "**结论：G3 可判完成** —— 5/5 headless 逐集 fail-closed 验收全部通过，"
            "GUI 对照通过（不计入 5 次）。"
        )
    else:
        lines.append("")
        lines.append("**结论：G3 不可判完成** —— 存在验收问题，见逐集判定与失败原因。")
    lines.append("")

    verdicts = summary.get("verdicts") or {}
    lines.append("## 逐集判定")
    lines.append("")
    lines.append(
        "| seed | GUI | passed | exit | lift(mm) | force peak(N) | "
        "moment score peak(N*m) | 接触条数 | 接触顺序 | 判定 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for episode in summary.get("episodes", []):
        key_values = episode.get("key_values", {})
        lines.append(
            "| %s | %s | %s | %s | %.3f | %.4f | %.4f | %s | %s | %s |"
            % (
                episode.get("seed"),
                "是" if episode.get("gui") else "否",
                key_values.get("passed"),
                key_values.get("process_exit_code"),
                float(key_values.get("body_lift_m") or 0.0) * 1000.0,
                float(key_values.get("wrist_force_peak_n") or 0.0),
                float(key_values.get("moment_safety_score_peak_nm") or 0.0),
                key_values.get("grip_contact_records"),
                "-".join(key_values.get("contact_order") or []),
                "通过" if episode.get("accepted") else "拒绝",
            )
        )
    lines.append("")

    failure_reasons = verdicts.get("failure_reasons") or {}
    gui_failure = verdicts.get("gui_failure_reasons") or {}
    if failure_reasons or gui_failure:
        lines.append("## 失败原因（未过滤、未覆盖）")
        lines.append("")
        lines.append(
            json.dumps(failure_reasons, ensure_ascii=False, indent=2)
        )
        lines.append(
            json.dumps(gui_failure, ensure_ascii=False, indent=2)
        )
        lines.append("")

    statistics = summary.get("statistics") or {}
    lines.append("## 接触顺序与首指分布")
    lines.append("")
    lines.append(
        "- 接触顺序分布："
        + json.dumps(
            statistics.get("contact_order_distribution"), ensure_ascii=False
        )
    )
    lines.append(
        "- 首指分布："
        + json.dumps(
            statistics.get("first_finger_distribution"), ensure_ascii=False
        )
    )
    lines.append("")

    lines.append(
        "## 数值统计（n=%s，signed 与 absolute 的 mean/median/P95/max）"
        % statistics.get("episode_count")
    )
    lines.append("")

    def emit_block(title: str, block: Mapping[str, Any]) -> None:
        lines.append("### %s" % title)
        lines.append("")
        for label in ("signed", "absolute"):
            readable = block.get(label + "_readable") or {}
            lines.append(
                "- %s（%s）：mean=%s median=%s P95=%s max=%s ｜ 易读（%s）："
                "mean=%s median=%s P95=%s max=%s"
                % (
                    label,
                    block.get("unit"),
                    _fmt(block.get(label, {}).get("mean")),
                    _fmt(block.get(label, {}).get("median")),
                    _fmt(block.get(label, {}).get("p95")),
                    _fmt(block.get(label, {}).get("maximum")),
                    block.get("readable_unit"),
                    _fmt(readable.get("mean")),
                    _fmt(readable.get("median")),
                    _fmt(readable.get("p95")),
                    _fmt(readable.get("maximum")),
                )
            )
        lines.append("")

    for title, key in (
        ("table translation (xy)", "table_translation_xy"),
        ("table yaw delta", "table_yaw_delta"),
        ("body lift", "body_lift"),
        ("body-TCP slip", "body_tcp_slip"),
        ("body-nut separation", "body_nut_separation"),
        ("wrist force peak", "wrist_force_peak"),
        ("moment safety score peak", "moment_safety_score_peak"),
    ):
        emit_block(title, statistics.get(key) or {})
    contact_stats = (
        statistics.get("episode_end_grip_contact_records") or {}
    ).get("stats") or {}
    emit_block("episode-end grip contact records", contact_stats)
    pose_error = statistics.get("posthoc_pose_error") or {}
    slip = statistics.get("lift_relative_slip") or {}
    for title, group in (
        ("posthoc pose error", pose_error),
        ("lift relative slip", slip),
    ):
        lines.append("### %s（分通道）" % title)
        lines.append("")
        for channel in CHANNELS:
            block = group.get(channel) or {}
            for label in ("signed", "absolute"):
                readable = block.get(label + "_readable") or {}
                lines.append(
                    "- %s %s（%s）：mean=%s median=%s P95=%s max=%s ｜ "
                    "易读（%s）：mean=%s median=%s P95=%s max=%s"
                    % (
                        channel,
                        label,
                        block.get("unit"),
                        _fmt(block.get(label, {}).get("mean")),
                        _fmt(block.get(label, {}).get("median")),
                        _fmt(block.get(label, {}).get("p95")),
                        _fmt(block.get(label, {}).get("maximum")),
                        block.get("readable_unit"),
                        _fmt(readable.get("mean")),
                        _fmt(readable.get("median")),
                        _fmt(readable.get("p95")),
                        _fmt(readable.get("maximum")),
                    )
                )
        for label in ("signed", "absolute"):
            for norm_key in ("translation_norm", "rotation_norm"):
                block = group.get(norm_key) or {}
                readable = block.get(label + "_readable") or {}
                lines.append(
                    "- %s %s（%s）：mean=%s median=%s P95=%s max=%s ｜ 易读（%s）："
                    "mean=%s median=%s P95=%s max=%s"
                    % (
                        norm_key,
                        label,
                        block.get("unit"),
                        _fmt(block.get(label, {}).get("mean")),
                        _fmt(block.get(label, {}).get("median")),
                        _fmt(block.get(label, {}).get("p95")),
                        _fmt(block.get(label, {}).get("maximum")),
                        block.get("readable_unit"),
                        _fmt(readable.get("mean")),
                        _fmt(readable.get("median")),
                        _fmt(readable.get("p95")),
                        _fmt(readable.get("maximum")),
                    )
                )
        lines.append("")

    gui = summary.get("gui_vs_headless_seed0") or {}
    if gui:
        lines.append("## GUI 对照（seed0，不计入 5 次）")
        lines.append("")
        functional = gui.get("functional") or {}
        lines.append(
            "- 功能一致性：" + json.dumps(functional, ensure_ascii=False)
        )
        lines.append("- 数值差异（gui - headless_seed0）：")
        for delta in gui.get("numeric_deltas", []):
            lines.append(
                "  - %s：%.6g %s → %.6g %s（delta %.6g %s）"
                % (
                    delta.get("metric"),
                    delta.get("headless_seed0"),
                    delta.get("unit"),
                    delta.get("gui"),
                    delta.get("unit"),
                    delta.get("delta_readable"),
                    delta.get("readable_unit"),
                )
            )
        lines.append("")

    hashes = summary.get("source_hashes") or {}
    lines.append("## 源码与配置哈希")
    lines.append("")
    lines.append("五集逐集一致：" + str(hashes.get("identical_across_episodes")))
    for key, value in (hashes.get("disk") or {}).items():
        lines.append("- %s = %s" % (key, value))
    lines.append("- payloads_by_seed：")
    for seed, payload in sorted(
        (hashes.get("payloads_by_seed") or {}).items()
    ):
        lines.append("  - seed%s: %s" % (seed, payload))
    lines.append("")

    lines.append("## 显式声明")
    lines.append("")
    lines.append(
        "- continuous_contact_path_verified=false：控制期不读 contact truth，"
        "末端接触只作 episode-end acceptance 证据。"
    )
    lines.append("- P95 n=5 PRELIMINARY，不是 G6/30-pair 最终分布。")
    lines.append(
        "- 本汇总只完成 G3（同源 synchronous 5 headless + 1 GUI 对照）；"
        "不代表 sequential-compliant、G4、5+5、30-pair 或视觉阶段完成。"
    )
    lines.append("- CPU fallback 一律不作为正式证据。")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="append",
        required=True,
        help="headless episode directory (exactly 5 required)",
    )
    parser.add_argument("--gui", required=True, help="GUI episode directory")
    parser.add_argument(
        "--runtime-logs",
        required=True,
        help="directory holding the copied kit logs",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output directory for summary.json / SUMMARY_CN.md / "
        "input_manifest.json",
    )
    arguments = parser.parse_args(argv)

    headless_dirs = [Path(raw).resolve() for raw in arguments.headless]
    gui_dir = Path(arguments.gui).resolve()
    logs_dir = Path(arguments.runtime_logs).resolve()
    output_dir = Path(arguments.output).resolve()
    if len(headless_dirs) != 5:
        parser.error("exactly 5 --headless directories are required")
    if len(set(headless_dirs)) != 5:
        parser.error("--headless directories must be distinct")
    if gui_dir in headless_dirs:
        parser.error("GUI directory must differ from every headless directory")
    if (
        output_dir in headless_dirs
        or output_dir == gui_dir
        or output_dir == logs_dir
    ):
        parser.error("output directory must not overwrite any input directory")
    if not logs_dir.is_dir():
        parser.error(f"runtime logs directory missing: {logs_dir}")

    cli_sha256 = _sha256_file(Path(__file__).resolve())
    disk_hashes = current_source_hashes()

    headless_reports: list[dict[str, Any]] = []
    headless_problems: list[list[str]] = []
    headless_orders: list[list[str]] = []
    for episode_dir in headless_dirs:
        report = load_report(episode_dir)
        contact_order = load_contact_order(episode_dir)
        seed = report.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            parser.error(f"invalid seed in {episode_dir}")
        log_path = logs_dir / kit_log_name(seed, False)
        if not log_path.is_file():
            parser.error(f"kit log missing: {log_path}")
        kit_log_text = log_path.read_text(encoding="utf-8", errors="replace")
        problems = verify_episode(
            report,
            expect_gui=False,
            disk_hashes=disk_hashes,
            kit_log_text=kit_log_text,
            contact_order=contact_order,
        )
        headless_reports.append(report)
        headless_problems.append(problems)
        headless_orders.append(contact_order)

    seeds = [report["seed"] for report in headless_reports]
    if sorted(seeds) != [0, 1, 2, 3, 4]:
        parser.error(
            f"headless seeds must be exactly 0-4, got {sorted(seeds)}"
        )

    gui_report = load_report(gui_dir)
    gui_order = load_contact_order(gui_dir)
    gui_seed = gui_report.get("seed")
    if isinstance(gui_seed, bool) or not isinstance(gui_seed, int):
        parser.error(f"invalid seed in {gui_dir}")
    gui_log_path = logs_dir / kit_log_name(gui_seed, True)
    if not gui_log_path.is_file():
        parser.error(f"kit log missing: {gui_log_path}")
    gui_kit_log_text = gui_log_path.read_text(
        encoding="utf-8", errors="replace"
    )
    gui_problems = verify_episode(
        gui_report,
        expect_gui=True,
        disk_hashes=disk_hashes,
        kit_log_text=gui_kit_log_text,
        contact_order=gui_order,
    )
    if gui_seed != 0:
        gui_problems.append("GUI seed must be 0 (headless seed-0 counterpart)")
    seed0_index = seeds.index(0)
    for key in headless_reports[seed0_index]["provenance"]:
        if (
            gui_report["provenance"].get(key)
            != headless_reports[seed0_index]["provenance"].get(key)
        ):
            gui_problems.append(
                f"GUI provenance {key} differs from headless seed-0"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_inputs: list[dict[str, Any]] = []
    for episode_dir, report, seed in zip(
        headless_dirs, headless_reports, seeds
    ):
        report_path = episode_dir / "nominal_physics_report.json"
        steps_path = episode_dir / "controller_steps.jsonl"
        log_path = logs_dir / kit_log_name(seed, False)
        manifest_inputs.append(
            {
                "role": "headless_episode",
                "seed": seed,
                "directory": str(episode_dir),
                "report_file": str(report_path),
                "report_sha256": _sha256_file(report_path),
                "steps_file": str(steps_path),
                "steps_sha256": _sha256_file(steps_path),
                "kit_log": str(log_path),
                "kit_log_sha256": _sha256_file(log_path),
            }
        )
    gui_report_path = gui_dir / "nominal_physics_report.json"
    gui_steps_path = gui_dir / "controller_steps.jsonl"
    manifest_inputs.append(
        {
            "role": "gui_episode",
            "seed": gui_seed,
            "directory": str(gui_dir),
            "report_file": str(gui_report_path),
            "report_sha256": _sha256_file(gui_report_path),
            "steps_file": str(gui_steps_path),
            "steps_sha256": _sha256_file(gui_steps_path),
            "kit_log": str(gui_log_path),
            "kit_log_sha256": _sha256_file(gui_log_path),
        }
    )
    for key, relative in SOURCE_FILES.items():
        source_path = REPOSITORY_ROOT / relative
        manifest_inputs.append(
            {
                "role": "source_file",
                "key": key,
                "path": str(source_path),
                "sha256": disk_hashes[key],
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_source_sha256": cli_sha256,
        "inputs": manifest_inputs,
    }
    canonical = json.dumps(
        {"inputs": manifest["inputs"]}, sort_keys=True, ensure_ascii=False
    )
    manifest_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest["manifest_content_sha256"] = manifest_sha256

    summary = build_summary(
        headless_dirs=headless_dirs,
        headless_reports=headless_reports,
        headless_problems=headless_problems,
        headless_contact_orders=headless_orders,
        gui_dir=gui_dir,
        gui_report=gui_report,
        gui_problems=gui_problems,
        gui_contact_order=gui_order,
        disk_hashes=disk_hashes,
        manifest_sha256=manifest_sha256,
        cli_sha256=cli_sha256,
    )

    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "input_manifest.json"
    summary_cn_path = output_dir / "SUMMARY_CN.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_cn_path.write_text(
        render_summary_cn(summary) + "\n", encoding="utf-8"
    )

    self_check_problems: list[str] = []
    reloaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(reloaded_summary, dict):
        self_check_problems.append("summary.json is not a JSON object")
    reloaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(reloaded_manifest, dict):
        self_check_problems.append("input_manifest.json is not a JSON object")
    else:
        recomputed = hashlib.sha256(
            json.dumps(
                {"inputs": reloaded_manifest.get("inputs", [])},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if recomputed != reloaded_manifest.get("manifest_content_sha256"):
            self_check_problems.append("input manifest content hash mismatch")
        if reloaded_summary.get("input_manifest_sha256") != recomputed:
            self_check_problems.append("summary manifest hash mismatch")
    if self_check_problems:
        print("SELF-CHECK FAILED: " + "; ".join(self_check_problems))
        return 2

    print(
        "headless accepted %d/5; gui accepted=%s; g3_complete=%s"
        % (
            summary["verdicts"]["headless_accepted"],
            summary["verdicts"]["gui_accepted"],
            summary["g3_complete"],
        )
    )
    for episode in summary["episodes"]:
        for problem in episode["verification_problems"]:
            print(f"REJECTED {episode['directory']}: {problem}")
    return 0 if summary["g3_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
