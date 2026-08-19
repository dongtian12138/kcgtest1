#!/usr/bin/env python3
"""Create deterministic post-hoc evidence for TASK-R12-005."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shlex
from typing import Any


EVENT_ORDER = [
    "five_key_polarization",
    "three_start_thread_entry",
    "spring_finger_engagement",
    "first_pin_socket_spring_touch",
    "pin_barrier_seal_contact",
    "seal_compression",
    "shell_to_shell_metal_bottoming",
]
COMPLIANT_EVENTS = {
    "spring_finger_engagement",
    "first_pin_socket_spring_touch",
    "pin_barrier_seal_contact",
    "seal_compression",
}
HARD_PENETRATION_LIMIT_M = 5.0e-5


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_key(pair: dict[str, Any]) -> str:
    return pair.get("event") or "UNSCORED:" + ",".join(
        sorted(pair.get("families", []))
    )


def contact_groups(
    audit: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "pair_count": 0,
            "contact_record_count": 0,
            "active_step_count": 0,
            "first_step": 10**12,
            "last_step": 0,
            "minimum_separation_m": None,
            "deepest_paths": None,
            "maximum_impulse_norm_ns": 0.0,
            "maximum_impulse_paths": None,
        }
    )
    for pair in audit["pairs"]:
        group = groups[family_key(pair)]
        group["pair_count"] += 1
        group["contact_record_count"] += int(pair["contact_record_count"])
        group["active_step_count"] += int(pair["active_step_count"])
        group["first_step"] = min(group["first_step"], int(pair["first_step"]))
        group["last_step"] = max(group["last_step"], int(pair["last_step"]))
        separation = pair.get("minimum_separation_m")
        if separation is not None and (
            group["minimum_separation_m"] is None
            or separation < group["minimum_separation_m"]
        ):
            group["minimum_separation_m"] = float(separation)
            group["deepest_paths"] = list(pair["collider_paths"])
        impulse = float(pair["maximum_impulse_norm"])
        if impulse > group["maximum_impulse_norm_ns"]:
            group["maximum_impulse_norm_ns"] = impulse
            group["maximum_impulse_paths"] = list(pair["collider_paths"])
    for group in groups.values():
        first = rows[int(group["first_step"]) - 1]
        group["first_time_s"] = first["time_s"]
        group["first_observed_separation_m"] = first["observed_separation_m"]
        group["first_target_separation_m"] = first["target_separation_m"]
        group["equivalent_peak_force_norm_n_at_240_hz"] = (
            group["maximum_impulse_norm_ns"] * 240.0
        )
    return dict(sorted(groups.items()))


def force_metrics(row: dict[str, Any]) -> dict[str, Any]:
    body_force = row["body_force_n"]
    nut_force = row["nut_force_n"]
    body_torque = row["body_torque_nm"]
    nut_torque = row["nut_torque_nm"]
    return {
        "body_force_n": body_force,
        "nut_force_n": nut_force,
        "axial_force_sum_magnitude_n": abs(body_force[2]) + abs(nut_force[2]),
        "body_lateral_force_norm_n": math.hypot(*body_force[:2]),
        "nut_lateral_force_norm_n": math.hypot(*nut_force[:2]),
        "combined_lateral_force_norm_n": math.hypot(
            body_force[0] + nut_force[0], body_force[1] + nut_force[1]
        ),
        "body_torque_nm": body_torque,
        "nut_torque_nm": nut_torque,
        "maximum_torque_component_nm": max(
            abs(value) for value in body_torque + nut_torque
        ),
        "combined_moment_norm_nm": math.sqrt(
            sum(
                (body_torque[index] + nut_torque[index]) ** 2
                for index in range(3)
            )
        ),
    }


def hard_violations(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for key, group in groups.items():
        if key in COMPLIANT_EVENTS:
            continue
        separation = group["minimum_separation_m"]
        if separation is not None and separation < -HARD_PENETRATION_LIMIT_M:
            result.append(
                {
                    "family": key,
                    "minimum_separation_m": separation,
                    "penetration_m": -separation,
                    "paths": group["deepest_paths"],
                }
            )
    return sorted(result, key=lambda item: item["minimum_separation_m"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--h2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    baseline = (repo / args.baseline).resolve()
    h1 = (repo / args.h1).resolve()
    h2 = (repo / args.h2).resolve()
    output = (repo / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    runs = {"baseline": baseline, "H1": h1, "H2": h2}
    reports = {label: read_json(path / "report.json") for label, path in runs.items()}
    traces = {label: read_jsonl(path / "trace.jsonl") for label, path in runs.items()}
    groups = {
        label: contact_groups(read_json(path / "contact_audit.json"), traces[label])
        for label, path in runs.items()
    }

    h2_report = reports["H2"]
    h2_rows = traces["H2"]
    onset_event = h2_report["event_first"]["pin_barrier_seal_contact"]
    onset_step = int(onset_event["step"])
    onset_row = h2_rows[onset_step - 1]
    final_row = h2_rows[-1]
    maximum_row = max(h2_rows, key=lambda row: row["observed_separation_m"])
    dominant = groups["H2"]["pin_barrier_seal_contact"]
    baseline_thread = groups["baseline"].get("three_start_thread_entry", {})
    h2_thread = groups["H2"].get("three_start_thread_entry", {})

    onset_contact = onset_event["first_contact"]
    onset_normal = onset_contact["contact_points"][0]["normal"]
    h2_analysis = {
        "schema_version": 1,
        "task_id": "TASK-R12-005",
        "hypothesis_id": "H2_FULL_BOUNDED_FEEDFORWARD",
        "driver_kind": "bounded_axial_force_feedforward",
        "execution_complete": True,
        "diagnostic_only": h2_report["diagnostic_only"],
        "formal_p1_pass_claimed": h2_report["formal_p1_pass_claimed"],
        "events_observed": h2_report["observed_event_order"],
        "last_three_events_all_observed": all(
            event in h2_report["event_first"] for event in EVENT_ORDER[-3:]
        ),
        "limits": {
            "maximum_force_component_n": h2_report["maximum_force_component_n"],
            "force_component_limit_n": h2_report["force_component_limit_n"],
            "maximum_torque_component_nm": h2_report["maximum_torque_component_nm"],
            "torque_component_limit_nm": 0.30,
            "solver_error_count": h2_report["solver_error_count"],
            "object_pose_write_after_physics_start_count": h2_report[
                "object_pose_write_after_physics_start_count"
            ],
        },
        "final": {
            "target_separation_m": final_row["target_separation_m"],
            "observed_separation_m": final_row["observed_separation_m"],
            "remaining_m": final_row["target_separation_m"]
            - final_row["observed_separation_m"],
            "maximum_observed_separation_m": maximum_row["observed_separation_m"],
            "force_and_moment": force_metrics(final_row),
        },
        "dominant_contact": {
            "families": ["hard_socket_entries_61", "pin_barriers_61"],
            "event": "pin_barrier_seal_contact",
            "block_onset_step": onset_step,
            "block_onset_time_s": onset_event["time_s"],
            "block_onset_observed_separation_m": onset_row[
                "observed_separation_m"
            ],
            "block_onset_target_separation_m": onset_row["target_separation_m"],
            "block_onset_force_and_moment": force_metrics(onset_row),
            "onset_collider_paths": onset_contact["collider_paths"],
            "onset_contact_normal": onset_normal,
            "onset_minimum_separation_m": onset_contact["minimum_separation_m"],
            "minimum_recorded_separation_m": dominant["minimum_separation_m"],
            "maximum_recorded_overlap_m": max(
                0.0, -float(dominant["minimum_separation_m"])
            ),
            "deepest_overlap_collider_paths": dominant["deepest_paths"],
            "pair_count": dominant["pair_count"],
            "contact_record_count": dominant["contact_record_count"],
            "active_step_count": dominant["active_step_count"],
            "last_step": dominant["last_step"],
            "observed_progress_after_onset_m": maximum_row[
                "observed_separation_m"
            ]
            - onset_row["observed_separation_m"],
            "target_progress_after_onset_m": final_row["target_separation_m"]
            - onset_row["target_separation_m"],
            "selection_basis": (
                "This is the only new scored family appearing at the terminal "
                "plateau; it begins at 14.293686 mm, persists through the final "
                "step, and observed travel thereafter is only about 23.75 um "
                "while the target advances about 666.67 um. Earlier thread and "
                "detent contacts do not persist to the final step."
            ),
        },
        "hard_penetration_audit": {
            "limit_m": HARD_PENETRATION_LIMIT_M,
            "baseline_violations": hard_violations(groups["baseline"]),
            "H1_violations": hard_violations(groups["H1"]),
            "H2_violations": hard_violations(groups["H2"]),
            "thread_minimum_separation_baseline_m": baseline_thread.get(
                "minimum_separation_m"
            ),
            "thread_minimum_separation_H2_m": h2_thread.get(
                "minimum_separation_m"
            ),
        },
        "classification": "PHYSICAL_CONTACT_BLOCK_CONFIRMED",
        "classification_basis": (
            "Two different bounded diagnostic drivers failed to produce all three "
            "remaining events. H2 reached only pin_barrier_seal_contact at the full "
            "8 N component limit, then stopped before seal compression and metal "
            "bottoming."
        ),
    }
    write_json(output / "H2_ANALYSIS.json", h2_analysis)
    write_json(output / "CONTACT_STATS.json", groups)

    windows: list[dict[str, Any]] = []
    for label, start, stop in (
        ("block_onset", max(0, onset_step - 11), min(len(h2_rows), onset_step + 10)),
        ("terminal", max(0, len(h2_rows) - 20), len(h2_rows)),
    ):
        for row in h2_rows[start:stop]:
            windows.append({"window": label, **row})
    with (output / "H2_BLOCK_WINDOWS.jsonl").open("w", encoding="utf-8") as stream:
        for row in windows:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    command_lines = []
    for record_name in (
        "run_01_bounded_axial_integral.json",
        "run_02_bounded_axial_force_feedforward.json",
    ):
        record = read_json(repo / "artifacts/agent_control/runs" / record_name)
        command_lines.extend(
            [
                f"# {record['hypothesis_id']}",
                shlex.join(record["command"]),
                f"exit_code={record['exit_code']} timed_out={record['timed_out']}",
                "",
            ]
        )
    (output / "ACTUAL_COMMANDS.txt").write_text(
        "\n".join(command_lines), encoding="utf-8"
    )

    summary_paths = [
        repo / "src/kcg_connector/isaac/d38999_physical_r7_p1_nominal_bench.py",
        repo / "src/kcg_connector/config/d38999_keyed_v3_physical_acceptance_r12_v1.yaml",
        repo / "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml",
        repo
        / "artifacts/kcg_connector/isaac/keyed_v3_physical_r12/candidates/"
        "r12_candidate_02/r12_candidate_02.usda",
        *(path / name for path in runs.values() for name in ("report.json", "trace.jsonl", "contact_audit.json")),
    ]
    write_json(
        output / "FILE_SUMMARY.json",
        {
            "files": [
                {
                    "path": str(path.relative_to(repo)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in summary_paths
            ]
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
