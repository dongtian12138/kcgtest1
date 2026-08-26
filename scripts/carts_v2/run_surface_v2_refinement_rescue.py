#!/usr/bin/env python3
"""Select and run bounded refinement for completed Surface-V2 exact shards."""
from __future__ import annotations

import argparse, json, math, os, time
from dataclasses import asdict, replace
from pathlib import Path
from kcg_connector.grasp.carts_v2.models import (
    CARTSV2Config, CandidateSeed, file_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
_SHARD_SCHEMA = "carts_surface_v2_exact_shortlist_shard_v1"
_FEATURE_SCHEMA = "carts_surface_v2_feature_search_run_v1"
_TRANSLATION_BOUND_M = 0.015
_ROTATION_BOUND_RAD = math.radians(8.0)

def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--exact-shard", type=Path, action="append", required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selection-only", action="store_true")
    mode.add_argument("--selection-index", type=int)
    return parser.parse_args()

def _resolve(root: Path, path: Path) -> Path: return path.resolve() if path.is_absolute() else (root / path).resolve()

def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"{path} is not one JSON object")
    return value

def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value

def _atomic_write(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(
            _finite_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _flags_are_closed(row: dict) -> bool:
    return all(row.get(name) is False for name in ("hardware_authorized", "formal_dynamic_pass", "research_dynamic_pass"))

def _valid_witnesses(candidate: dict, global_index: int) -> list[dict]:
    evaluated = candidate.get("height_search", {}).get("evaluated", [])
    if len(evaluated) != 1: return []
    row = evaluated[0]
    if (row.get("status") != "POST_PROJECTION_REVALIDATION_REJECT" or
            row.get("reason") != "CONTACT_LOST_AFTER_TABLE_PROJECTION"):
        return []
    probes = row.get("contact_height_probes", [])
    result = []
    for ordinal, witness in enumerate(row.get("contact_conditioned_iterations", [])):
        phases = witness.get("contact_stop_phases", [])
        numbers = [witness.get("handbase_world_z_m"), witness.get("minimum_table_handbase_z_m"), *phases]
        if len(phases) != 3 or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in numbers):
            continue
        contact_z, table_z = map(float, numbers[:2])
        matching = any(
            probe.get("closure_status") == "CLOSURE_SURVIVE"
            and probe.get("contact_count") == 3
            and probe.get("handbase_world_z_m") == contact_z for probe in probes)
        if not matching or table_z < contact_z:
            continue
        stop = tuple(float(value) for value in phases)
        result.append({
            "global_shortlist_index": global_index,
            "candidate_id": candidate["candidate_id"],
            "input_seed": candidate["input_seed"],
            "three_contact_world_z_m": contact_z,
            "minimum_table_handbase_z_m": table_z,
            "table_contact_conflict_m": table_z - contact_z,
            "reference_contact_stop_phases": list(stop),
            "stop_phase_range": max(stop) - min(stop),
            "source_iteration": int(witness.get("iteration", ordinal)),
        })
    return sorted(result, key=lambda row: (row["table_contact_conflict_m"], row["stop_phase_range"], row["source_iteration"]))[:1]

def build_selection_manifest(repository_root: Path, feature_report: Path,
                             exact_shards: list[Path]) -> dict:
    root, feature = repository_root.resolve(), feature_report.resolve()
    shards = [path.resolve() for path in exact_shards]
    if len(shards) != 3 or len(set(shards)) != 3:
        raise ValueError("exact rescue requires three distinct shards")
    source, feature_sha = _read(feature), file_sha256(feature)
    config = _resolve(root, Path(source.get("config", "")))
    if (source.get("schema_version") != _FEATURE_SCHEMA
            or not _flags_are_closed(source)
            or len(source.get("exact_shortlist", [])) != 24
            or not config.is_file()
            or file_sha256(config) != source.get("config_sha256")):
        raise ValueError("feature report identity or authorization changed")
    loaded = sorted(((_read(path), path) for path in shards),
                    key=lambda item: item[0].get("shortlist_offset", -1))
    if [row.get("shortlist_offset") for row, _path in loaded] != [0, 8, 16]:
        raise ValueError("exact shard offsets do not partition the Top-24")
    witnesses, seen = [], set()
    for shard, path in loaded:
        offset, candidates = int(shard["shortlist_offset"]), shard.get("candidates", [])
        if (shard.get("schema_version") != _SHARD_SCHEMA
                or not _flags_are_closed(shard)
                or shard.get("feature_report_sha256") != feature_sha
                or _resolve(root, Path(shard.get("feature_report", ""))) != feature
                or shard.get("requested_count") != 8
                or shard.get("completed_count") != 8 or len(candidates) != 8):
            raise ValueError(f"exact shard identity or completeness changed: {path}")
        expected = source["exact_shortlist"][offset:offset + 8]
        for local_index, (candidate, seed) in enumerate(zip(candidates, expected)):
            candidate_id = candidate.get("candidate_id")
            if (candidate_id in seen or candidate_id != seed.get("candidate_id")
                    or candidate.get("input_seed") != seed):
                raise ValueError("exact candidate identity or ordering changed")
            seen.add(candidate_id)
            if candidate.get("sampled_exact_geometry_pass") is not False:
                raise ValueError("rescue is invalid while an exact geometry pass exists")
            for witness in _valid_witnesses(candidate, offset + local_index):
                witness["source_shard"] = str(path)
                witness["source_shard_sha256"] = file_sha256(path)
                witnesses.append(witness)
    witnesses.sort(key=lambda row: (
        row["table_contact_conflict_m"], row["stop_phase_range"],
        row["source_iteration"], row["global_shortlist_index"], row["candidate_id"]))
    selected = [dict(row, selection_index=index) for index, row in enumerate(witnesses[:8])]
    return {
        "schema_version": "carts_surface_v2_refinement_rescue_selection_v1",
        "claim_scope": "DETERMINISTIC_OFFLINE_RESCUE_SEEDS_NOT_GEOMETRY_TASK_IK_OR_DYNAMIC_SUCCESS",
        "hardware_authorized": False, "formal_dynamic_pass": False,
        "research_dynamic_pass": False,
        "feature_report": str(feature),
        "feature_report_sha256": feature_sha,
        "config": str(config), "config_sha256": file_sha256(config),
        "object_id": source["object_id"],
        "object_mesh_sha256": source["object_mesh_sha256"],
        "exact_shards": [{"path": str(path), "sha256": file_sha256(path)}
                         for _row, path in loaded],
        "registered_candidate_count": 24,
        "eligible_contact_table_conflict_count": len(witnesses),
        "selected_count": len(selected),
        "selection_rule": ["TABLE_CONTACT_CONFLICT_M_MIN",
                           "STOP_PHASE_RANGE_MIN", "SOURCE_ITERATION_MIN",
                           "GLOBAL_SHORTLIST_INDEX_MIN", "CANDIDATE_ID_MIN"],
        "selected": selected,
    }

def _candidate(row: dict) -> CandidateSeed:
    values = dict(row)
    for name in ("anchor_position_object_m", "object_from_hand",
                 "pregrasp_joint_positions_rad", "pregrasp_closure_phases"):
        values[name] = tuple(values[name])
    if values.get("approach_direction_object") is not None:
        values["approach_direction_object"] = tuple(values["approach_direction_object"])
    return CandidateSeed(**values)

def _surface_v2_refinement_inputs(inputs):
    values = dict(inputs.config.values)
    generation = dict(values["candidate_generation"])
    refinement = dict(generation["refinement"])
    refinement["translation_bound_m"] = _TRANSLATION_BOUND_M
    refinement["rotation_bound_rad"] = _ROTATION_BOUND_RAD
    generation["refinement"] = refinement
    values["candidate_generation"] = generation
    return replace(inputs, config=CARTSV2Config(inputs.config.path, values))

def main() -> int:
    args, started = _arguments(), time.perf_counter()
    root = args.repository_root.resolve()
    script_sha_at_start = file_sha256(Path(__file__).resolve())
    refinement_source = (root / "src/kcg_connector/kcg_connector/grasp/"
                         "carts_v2/opposition_refinement_search.py")
    refinement_source_sha_at_start = file_sha256(refinement_source)
    feature = _resolve(root, args.feature_report)
    shards = [_resolve(root, path) for path in args.exact_shard]
    output, manifest = _resolve(root, args.output), build_selection_manifest(
        root, feature, shards)
    if args.selection_only:
        if args.selection_manifest is not None:
            raise ValueError("selection-only does not consume a selection manifest")
        _atomic_write(output, manifest)
        print(json.dumps({"output": str(output), "selected_count": manifest["selected_count"]}))
        return 0
    if args.selection_manifest is None:
        raise ValueError("--selection-index requires --selection-manifest")
    registered_path = _resolve(root, args.selection_manifest)
    if _read(registered_path) != manifest:
        raise ValueError("selection manifest no longer matches the exact evidence")
    index = args.selection_index
    if index is None or not 0 <= index < manifest["selected_count"]:
        raise ValueError("selection index is outside the registered rescue set")
    from kcg_connector.grasp.carts_v2.models import load_v2_inputs
    from kcg_connector.grasp.carts_v2.opposition_refinement_search import (
        refine_opposition_pose,
    )
    selected = manifest["selected"][index]
    inputs = load_v2_inputs(root, config_path=_resolve(root, Path(manifest["config"])),
                            object_id=manifest["object_id"])
    if inputs.object_contract.model.provenance.source_sha256 != manifest["object_mesh_sha256"]:
        raise ValueError("object mesh identity changed after selection")
    inputs = _surface_v2_refinement_inputs(inputs)
    status, survivors, refinement = "BOUNDED_REFINEMENT_COMPLETE", (), None
    try:
        survivors, refinement = refine_opposition_pose(
            inputs, _candidate(selected["input_seed"]),
            three_contact_world_z_m=selected["three_contact_world_z_m"],
            reference_contact_stop_phases=selected["reference_contact_stop_phases"])
    except ValueError as error:
        if str(error) != "registered three-contact witness no longer replays":
            raise
        status = "SOURCE_THREE_CONTACT_WITNESS_REPLAY_FAILED"
    result = {
        "schema_version": "carts_surface_v2_refinement_rescue_run_v1",
        "claim_scope": "OFFLINE_BOUNDED_REFINEMENT_NOT_TASK_IK_OR_DYNAMIC_SUCCESS",
        "status": status, "hardware_authorized": False,
        "formal_dynamic_pass": False, "research_dynamic_pass": False,
        "selection_manifest": str(registered_path),
        "selection_manifest_sha256": file_sha256(registered_path),
        "selection": selected, "refinement": refinement,
        "serialization_nonfinite_values_as_null": True,
        "effective_refinement_bounds": {
            "translation_each_axis_m": _TRANSLATION_BOUND_M,
            "rotation_each_axis_rad": _ROTATION_BOUND_RAD,
            "palm_configuration_rad": inputs.config.section(
                "candidate_generation")["refinement"][
                    "palm_configuration_bound_rad"],
            "source": "SURFACE_V2_FAST6H_USER_PREREGISTERED_BOUNDARY",
        },
        "b_full_pass_count": len(survivors),
        "b_full_pass_candidates": [asdict(seed) for seed in survivors],
        "isaac_started": False, "elapsed_s": time.perf_counter() - started,
        "script_sha256_at_process_start": script_sha_at_start,
        "refinement_source_sha256_at_process_start":
            refinement_source_sha_at_start,
    }
    _atomic_write(output, result)
    print(json.dumps({"output": str(output), "status": status,
                      "b_full_pass_count": len(survivors)}))
    return 0 if status == "BOUNDED_REFINEMENT_COMPLETE" else 2

if __name__ == "__main__":
    raise SystemExit(main())
