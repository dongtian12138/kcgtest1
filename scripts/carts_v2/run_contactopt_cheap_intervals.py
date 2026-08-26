#!/usr/bin/env python3
"""Run the bounded CONTACTOPT cheap screen and proxy contact intervals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time

import yaml

from kcg_connector.grasp.carts_v2.contact_interval_solver import (
    solve_proxy_contact_intervals,
)
from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed, file_sha256, load_v2_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
SEED_MANIFEST = Path("artifacts/carts_v2/contactopt_1488_fast6h/seed_manifest.json")
OUTPUT_ROOT = Path("artifacts/carts_v2/contactopt_1488_fast6h")
OBJECT_B = "te_deutsch_d38999_26fj35pn_step"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--seed-manifest", type=Path, default=SEED_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _candidate(row: dict) -> CandidateSeed:
    value = dict(row)
    for key in ("anchor_position_object_m", "object_from_hand",
                "pregrasp_joint_positions_rad", "pregrasp_closure_phases"):
        value[key] = tuple(value[key])
    if value.get("approach_direction_object") is not None:
        value["approach_direction_object"] = tuple(value["approach_direction_object"])
    return CandidateSeed(**value)


def _status_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((str(row["candidate_id"]) + "\0" + str(row["status"])
                       + "\n").encode())
    return digest.hexdigest()


def _configured_input_hashes(root: Path, base: Path) -> dict[str, dict[str, str]]:
    config = yaml.safe_load(base.read_text(encoding="utf-8"))
    _require(config.get("hardware_authorized") is False,
             "base physical config hardware boundary changed")
    inputs = config.get("inputs", {})
    result = {}
    for key in ("object_contract", "hand_contract", "collision_roster",
                "task_grip_surface_manifest"):
        path = _resolve(root, inputs[key])
        _require(path.is_file(), f"configured input is missing: {key}")
        result[key] = {"path": str(path), "sha256": file_sha256(path)}
    expected = str(inputs.get("task_grip_surface_manifest_sha256", ""))
    _require(result["task_grip_surface_manifest"]["sha256"] == expected,
             "TASK_GRIP_SURFACE manifest hash changed")
    return result


def _load_and_verify(root: Path, manifest_path: Path):
    manifest = _read_json(manifest_path)
    _require(manifest.get("schema_version") == "carts_contactopt_seed_run_v1",
             "seed manifest schema changed")
    _require(manifest.get("object_id") == OBJECT_B, "cheap B run received another object")
    _require(manifest.get("hardware_authorized") is False,
             "seed manifest hardware boundary changed")
    method = _resolve(root, manifest["method_config"])
    base = _resolve(root, manifest["base_physical_config"])
    producer = _resolve(root, manifest["source"]["path"])
    _require(file_sha256(method) == manifest["method_config_sha256"],
             "method config hash changed")
    _require(file_sha256(base) == manifest["base_physical_config_sha256"],
             "base physical config hash changed")
    _require(file_sha256(producer) == manifest["source"]["sha256"],
             "seed producer source hash changed")
    implementations = manifest.get("implementation_sources", [])
    _require(len(implementations) == 2, "seed implementation source binding is incomplete")
    for source in implementations:
        path = _resolve(root, source["path"])
        _require(file_sha256(path) == source["sha256"],
                 "seed implementation source hash changed")
    method_values = yaml.safe_load(method.read_text(encoding="utf-8"))
    _require(_resolve(root, method_values["base_physical_config"]) == base,
             "method/base config binding changed")
    _require(method_values.get("hardware_authorized") is False
             and method_values["structured_seeds"]["total_count_per_object"] == 1488,
             "CONTACTOPT method identity changed")
    inputs = load_v2_inputs(root, config_path=base, object_id=manifest["object_id"])
    _require(inputs.object_contract.model.provenance.source_sha256
             == manifest["object_mesh_sha256"], "object mesh hash changed")
    rows = manifest.get("audit", {}).get("specifications", [])
    candidates = manifest.get("generated_candidates", [])
    _require(len(rows) == 1488 and len({row["candidate_id"] for row in rows}) == 1488,
             "seed specification audit is incomplete")
    _require(_status_digest(rows) == manifest["audit"]["specification_status_sha256"],
             "seed specification status digest changed")
    pose_ids = {row["candidate_id"] for row in rows if row["status"] == "POSE_GENERATED"}
    candidate_ids = {row["candidate_id"] for row in candidates}
    _require(candidate_ids == pose_ids and len(candidate_ids) == len(candidates),
             "generated candidates disagree with the 1488-row audit")
    _require(all(row.get("object_id") == OBJECT_B for row in candidates),
             "generated candidate object identity changed")
    seeds = tuple(_candidate(row) for row in candidates)
    return manifest, method, base, inputs, rows, seeds, _configured_input_hashes(root, base)


def _top120_csv(audit: dict) -> str | None:
    intervals = audit.get("interval_rows", [])
    if not intervals:
        return None
    cheap = {row["candidate_id"]: row for row in audit["cheap_rows"]}
    fields = ("rank", "candidate_id", "family", "palm_key", "axial_key",
              "maximum_positive_gap_m", "gap_imbalance_m", "hard_margin_m",
              "table_margin_m", "self_margin_m", "interval_status", "reason",
              "finger_1_q_expected_rad", "finger_1_q_safe_max_rad",
              "finger_2_q_expected_rad", "finger_2_q_safe_max_rad",
              "finger_3_q_expected_rad", "finger_3_q_safe_max_rad")
    rows = []
    for rank, interval in enumerate(intervals, 1):
        source = cheap[interval["candidate_id"]]
        row = {key: source.get(key) for key in fields}
        row.update({"rank": rank, "candidate_id": interval["candidate_id"],
                    "interval_status": interval["status"], "reason": interval["reason"]})
        for index, finger in enumerate(interval.get("finger_intervals", []), 1):
            row[f"finger_{index}_q_expected_rad"] = finger["proxy_q_expected_rad"]
            row[f"finger_{index}_q_safe_max_rad"] = finger["proxy_q_safe_max_rad"]
        rows.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def _atomic_outputs(targets: dict[Path, str]) -> None:
    _require(all(not path.exists() for path in targets), "refusing to overwrite cheap evidence")
    staged = {}
    try:
        for path, text in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with temporary.open("x", encoding="utf-8", newline="") as stream:
                stream.write(text); stream.flush(); os.fsync(stream.fileno())
            staged[path] = temporary
        for path, temporary in staged.items():
            os.replace(temporary, path)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def main() -> int:
    args, started = _arguments(), time.perf_counter()
    root = args.repository_root.resolve()
    manifest_path = _resolve(root, args.seed_manifest)
    output_root = _resolve(root, args.output_root)
    manifest, method, base, inputs, rows, seeds, input_hashes = _load_and_verify(
        root, manifest_path)
    selected, audit = solve_proxy_contact_intervals(inputs, seeds, rows)
    report = {"schema_version": "carts_contactopt_cheap_intervals_run_v1",
              "claim_scope": "CKDTREE_CONTACT_CONVEX_HULL_TABLE_FCL_SELF_SAMPLED_NOT_EXACT_OBJECT_CONTACT_OR_DYNAMIC_SUCCESS",
              "hardware_authorized": False, "formal_dynamic_pass": False,
              "research_dynamic_pass": False, "object_id": manifest["object_id"],
              "seed_manifest": str(manifest_path),
              "seed_manifest_sha256": file_sha256(manifest_path),
              "method_config": str(method), "method_config_sha256": file_sha256(method),
              "base_physical_config": str(base), "base_physical_config_sha256": file_sha256(base),
              "object_mesh_sha256": inputs.object_contract.model.provenance.source_sha256,
              "configured_input_hashes": input_hashes,
              "seed_producer": manifest["source"],
              "solver_source": {"path": str(Path(solve_proxy_contact_intervals.__code__.co_filename).resolve()),
                                "sha256": file_sha256(Path(solve_proxy_contact_intervals.__code__.co_filename))},
              "runner_source": {"path": str(Path(__file__).resolve()),
                                "sha256": file_sha256(Path(__file__).resolve())},
              "elapsed_s": time.perf_counter() - started,
              "proxy_interval_survivors": [asdict(seed) for seed in selected],
              "audit": audit}
    targets = {output_root / "cheap_evaluation/cheap_intervals_B.json":
               json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"}
    csv_text = _top120_csv(audit)
    if csv_text is not None:
        targets[output_root / "top120/top120.csv"] = csv_text
    _atomic_outputs(targets)
    print(json.dumps({"output": str(next(iter(targets))),
                      "near_contact": audit["near_contact_seed_count"],
                      "top120": audit["top120_count"],
                      "interval_survivors": len(selected),
                      "elapsed_s": report["elapsed_s"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
