#!/usr/bin/env python3
"""Generate and audit the fixed CONTACTOPT-1488 contact-aligned seed design."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from kcg_connector.grasp.carts_v2.models import file_sha256, load_v2_inputs
from kcg_connector.grasp.carts_v2.structured_seed_generator import (
    generate_structured_contact_seeds,
)
from kcg_connector.grasp.carts_v2.three_contact_pose_initializer import (
    hand_contact_references,
)


ROOT = Path(__file__).resolve().parents[2]
METHOD_CONFIG = Path("src/kcg_connector/config/carts_contactopt_1488_fast6h.yaml")
OBJECT_B = "te_deutsch_d38999_26fj35pn_step"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--method-config", type=Path, default=METHOD_CONFIG)
    parser.add_argument("--object-id", default=OBJECT_B)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _method_identity(root: Path, path: Path) -> tuple[dict, Path]:
    method = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = _resolve(root, Path(method["base_physical_config"]))
    design = method["structured_seeds"]
    if (method.get("hardware_authorized") is not False
            or design["global"]["expected_count"] != 1040
            or design["dense_opposition"]["expected_count"] != 448
            or design["total_count_per_object"] != 1488):
        raise ValueError("CONTACTOPT_METHOD_IDENTITY_INVALID")
    return method, base


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _coverage_plots(output: Path, rows: list[dict]) -> None:
    colors = ["tab:blue" if row["family"] == "GLOBAL" else "tab:orange"
              for row in rows]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.scatter([row["qp_deg"] for row in rows],
                 [row["axial_ratio"] for row in rows], c=colors, s=7, alpha=0.35)
    axis.set(xlabel="palm configuration q_p (deg)", ylabel="axial ratio",
             title="CONTACTOPT-1488 registered q_p / axial coverage")
    fig.tight_layout()
    fig.savefig(output / "seed_coverage_qp_axial.png", dpi=150)
    plt.close(fig)
    generated = [row for row in rows if row["status"] == "POSE_GENERATED"]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.hist([row["azimuth_deg"] for row in generated], bins=32,
              color="tab:green", edgecolor="black")
    axis.set(xlabel="generated base azimuth (deg)", ylabel="count",
             title="CONTACTOPT-1488 generated azimuth coverage")
    fig.tight_layout()
    fig.savefig(output / "seed_coverage_azimuth.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args, started = _arguments(), time.perf_counter()
    root = args.repository_root.resolve()
    config = _resolve(root, args.method_config)
    output = _resolve(root, args.output_dir)
    targets = tuple(output / name for name in (
        "seed_manifest.json", "seed_coverage.csv", "seed_coverage_qp_axial.png",
        "seed_coverage_azimuth.png", "seed_preshape_counts.csv"))
    if any(path.exists() for path in targets):
        raise ValueError("refusing to overwrite CONTACTOPT seed evidence")
    _method, base = _method_identity(root, config)
    inputs = load_v2_inputs(root, config_path=base, object_id=args.object_id)
    seeds, audit = generate_structured_contact_seeds(inputs)
    rows = list(audit["specifications"])
    counts = Counter(row["preshape_id"] for row in rows)
    report = {
        "schema_version": "carts_contactopt_seed_run_v1",
        "claim_scope": "STRUCTURED_CONTACT_ALIGNED_POSES_NOT_COLLISION_TASK_OR_DYNAMIC_SUCCESS",
        "hardware_authorized": False,
        "formal_dynamic_pass": False,
        "research_dynamic_pass": False,
        "object_id": inputs.object_contract.object_id,
        "method_config": str(config), "method_config_sha256": file_sha256(config),
        "base_physical_config": str(base), "base_physical_config_sha256": file_sha256(base),
        "object_mesh_sha256": inputs.object_contract.model.provenance.source_sha256,
        "audit": audit, "generated_candidates": [asdict(seed) for seed in seeds],
        "elapsed_s": time.perf_counter() - started,
        "source": {"path": str(Path(__file__).resolve()),
                   "sha256": file_sha256(Path(__file__).resolve())},
        "implementation_sources": [
            {"path": str(Path(function.__code__.co_filename).resolve()),
             "sha256": file_sha256(Path(function.__code__.co_filename))}
            for function in (generate_structured_contact_seeds, hand_contact_references)
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(report, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    fields = ("candidate_id", "family", "qp_index", "qp_deg",
              "azimuth_or_peak_index", "azimuth_deg", "axial_index", "axial_ratio",
              "preshape_index", "preshape_id", "status", "reason",
              "maximum_point_residual_m", "maximum_normal_residual_rad")
    _write_csv(targets[1], rows, fields)
    _write_csv(targets[4], [{"preshape_id": key, "count": counts[key]}
                            for key in sorted(counts)], ("preshape_id", "count"))
    _coverage_plots(output, rows)
    print(json.dumps({"output": str(output), "specifications": len(rows),
                      "generated": len(seeds), "status_counts": audit["status_counts"],
                      "elapsed_s": report["elapsed_s"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
