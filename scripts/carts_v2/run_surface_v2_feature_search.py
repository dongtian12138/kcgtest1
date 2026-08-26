#!/usr/bin/env python3
"""Run the bounded Surface-V2 feature grid and save its exact shortlist."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from kcg_connector.grasp.carts_v2.fast_surface_phase_search import (
    search_feature_aware_opposition,
)
from kcg_connector.grasp.carts_v2.models import file_sha256, load_v2_inputs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("src/kcg_connector/config/carts_surface_v2_fast6h.yaml")
DEFAULT_OBJECT = "te_deutsch_d38999_26fj35pn_step"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--object-id", default=DEFAULT_OBJECT)
    parser.add_argument("--maximum-exact-candidates", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _resolved(root: Path, supplied: Path) -> Path:
    return supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()


def main() -> int:
    args, started = _arguments(), time.perf_counter()
    root = args.repository_root.resolve()
    config, output = _resolved(root, args.config), _resolved(root, args.output)
    if output.exists():
        raise ValueError(f"refusing to overwrite evidence: {output}")
    inputs = load_v2_inputs(root, config_path=config, object_id=args.object_id)
    shortlist, audit = search_feature_aware_opposition(
        inputs, maximum_exact_candidates=args.maximum_exact_candidates)
    report = {
        "schema_version": "carts_surface_v2_feature_search_run_v1",
        "claim_scope": "FULL_CHEAP_GRID_AND_EXACT_SHORTLIST_NOT_EXACT_OR_DYNAMIC_SUCCESS",
        "hardware_authorized": False,
        "formal_dynamic_pass": False,
        "research_dynamic_pass": False,
        "config": str(config),
        "config_sha256": file_sha256(config),
        "object_id": inputs.object_contract.object_id,
        "object_mesh_sha256": inputs.object_contract.model.provenance.source_sha256,
        "search_audit": audit,
        "exact_shortlist": [asdict(seed) for seed in shortlist],
        "elapsed_s": time.perf_counter() - started,
        "source": {"script": str(Path(__file__).resolve()),
                   "script_sha256": file_sha256(Path(__file__).resolve())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "registered_candidate_count":
                      audit["registered_candidate_count"], "exact_shortlist_count":
                      len(shortlist), "elapsed_s": report["elapsed_s"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
