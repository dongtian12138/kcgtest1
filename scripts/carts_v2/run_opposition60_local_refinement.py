#!/usr/bin/env python3
"""Refine the registered object-B 60-degree contact/table conflict offline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.opposition_refinement_search import (
    refine_opposition_pose,
)
from kcg_connector.grasp.carts_v2.opposition_seed_generator import (
    generate_opposition_anchors,
)
from kcg_connector.grasp.robust.object_model import file_sha256


_CONFIG = Path("src/kcg_connector/config/carts_nailfree_height_projected.yaml")
_SUMMARY = Path(
    "artifacts/carts_v2/opposition60_isaac/qp60_anchor_a01_probe1/"
    "ANCHOR_A01_THREE_CONTACT_TABLE_CONFLICT_SUMMARY.json"
)
_OUTPUT = Path(
    "artifacts/carts_v2/opposition60_isaac/qp60_anchor_a01_local_refinement"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=_CONFIG)
    parser.add_argument("--witness-summary", type=Path, default=_SUMMARY)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    return parser.parse_args()


def _resolved(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _candidate_record(seed) -> dict[str, object]:
    return {
        "candidate_id": seed.candidate_id,
        "object_id": seed.object_id,
        "object_from_hand_row_major": list(seed.object_from_hand),
        "pregrasp_joint_positions_rad": list(seed.pregrasp_joint_positions_rad),
        "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
        "palm_configuration_rad": seed.palm_configuration_rad,
        "approach_direction_object": list(seed.approach_direction_object),
    }


def main() -> int:
    started = time.perf_counter()
    args = _arguments()
    root = args.repository_root.resolve()
    config = _resolved(root, args.config)
    summary_path = _resolved(root, args.witness_summary)
    output = _resolved(root, args.output)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema_version")
        != "carts_opposition60_anchor_table_conflict_v1"
        or summary.get("object_id") != "te_deutsch_d38999_26fj35pn_step"
        or summary.get("anchor_index") != 1
        or summary.get("palm_configuration_deg") != 60
        or summary.get("pregrasp_closure_phases") != [0.2, 0.2, 0.2]
        or summary.get("research_executable_candidate") is not False
    ):
        raise ValueError("registered contact/table-conflict witness identity changed")
    raw_path = summary_path.parent / str(summary["raw_result"])
    if file_sha256(raw_path) != summary["raw_result_sha256"]:
        raise ValueError("registered contact/table-conflict raw result hash changed")
    inputs = load_v2_inputs(root, config_path=config, object_id=summary["object_id"])
    seeds, anchor_audit = generate_opposition_anchors(inputs, (math.radians(60.0),))
    anchor = seeds[1]
    if anchor.candidate_id != summary["candidate_id"]:
        raise ValueError("deterministic opposition anchor identity changed")
    witness = summary["three_contact_witness"]
    survivors, refinement = refine_opposition_pose(
        inputs,
        anchor,
        three_contact_world_z_m=float(witness["handbase_world_z_m"]),
        reference_contact_stop_phases=witness["contact_stop_phases"],
    )
    result = {
        "schema_version": "carts_opposition60_local_refinement_run_v1",
        "claim_scope": "OFFLINE_BOUNDED_REFINEMENT_NOT_TASK_IK_OR_DYNAMIC_SUCCESS",
        "source_witness_summary": str(summary_path.relative_to(root)),
        "source_witness_summary_sha256": file_sha256(summary_path),
        "source_raw_result_sha256": file_sha256(raw_path),
        "config": str(config.relative_to(root)),
        "config_sha256": file_sha256(config),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "source_sha256": {
            relative: file_sha256(root / relative)
            for relative in (
                "src/kcg_connector/kcg_connector/grasp/carts_v2/"
                "opposition_refiner.py",
                "src/kcg_connector/kcg_connector/grasp/carts_v2/"
                "opposition_refinement_search.py",
                "src/kcg_connector/kcg_connector/grasp/carts_v2/"
                "height_projection.py",
                "src/kcg_connector/kcg_connector/grasp/carts_v2/"
                "height_projected_search.py",
            )
        },
        "anchor_generation_claim_scope": anchor_audit["claim_scope"],
        "anchor": _candidate_record(anchor),
        "refinement": refinement,
        "b_full_pass_candidates": [_candidate_record(seed) for seed in survivors],
        "b_full_pass_count": len(survivors),
        "research_executable_candidate": False,
        "isaac_started": False,
        "connector_moved": False,
        "hardware_authorized": False,
        "dynamic_success": False,
        "elapsed_s": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "opposition60_anchor_a01_bounded_refinement.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "result": str(destination),
        "result_sha256": file_sha256(destination),
        "guidance_evaluation_count": refinement["guidance_evaluation_count"],
        "exact_revalidation_count": refinement["exact_revalidation_count"],
        "fixed_phase_gap_witness_count": refinement[
            "fixed_phase_gap_witness_count"],
        "table_evaluated_gap_witness_count": refinement[
            "table_evaluated_gap_witness_count"],
        "b_full_pass_count": len(survivors),
        "elapsed_s": result["elapsed_s"],
        "isaac_started": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
