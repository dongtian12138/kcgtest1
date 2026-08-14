#!/usr/bin/env python3

"""Derive a conservative connected Bcapture from raw four-dimensional trials."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _args(repository):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    result = parser.parse_args()
    if not result.run:
        parser.error("analysis requires --run")
    return result


def _neighbors(cell, available, dx, dy):
    x, y = cell
    result = []
    for candidate in ((x - dx, y), (x + dx, y), (x, y - dy), (x, y + dy)):
        for value in available:
            if math.isclose(value[0], candidate[0], abs_tol=1e-12) and math.isclose(value[1], candidate[1], abs_tol=1e-12):
                result.append(value)
    return result


def _connected_safe(records):
    by_cell = {(item["ex_m"], item["ey_m"]): item for item in records}
    safe = {cell for cell, item in by_cell.items() if item["capture_success"] and not item["hard_gate_triggered"] and not str(item["terminal_state"]).startswith("SAFE_ABORT")}
    origin = min(by_cell, key=lambda cell: math.hypot(*cell))
    if origin not in safe:
        return [], origin
    xs, ys = sorted({cell[0] for cell in by_cell}), sorted({cell[1] for cell in by_cell})
    dx = min(b - a for a, b in zip(xs, xs[1:])); dy = min(b - a for a, b in zip(ys, ys[1:]))
    connected, frontier = {origin}, [origin]
    while frontier:
        cell = frontier.pop()
        for neighbor in _neighbors(cell, safe, dx, dy):
            if neighbor not in connected:
                connected.add(neighbor); frontier.append(neighbor)
    return sorted(connected), origin


def main():
    repository = Path(__file__).resolve().parents[3]
    args = _args(repository)
    raw_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    records = raw["results"]
    groups = sorted({(item["erx_rad"], item["ery_rad"]) for item in records})
    slices = {}
    for rx, ry in groups:
        subset = [item for item in records if item["erx_rad"] == rx and item["ery_rad"] == ry]
        connected, origin = _connected_safe(subset)
        slices[f"erx_{rx:+.10f}_ery_{ry:+.10f}"] = {
            "erx_rad": rx, "ery_rad": ry,
            "origin_cell_m": list(origin),
            "origin_capture_success": bool(connected),
            "connected_safe_xy_cells_m": [list(cell) for cell in connected],
            "connected_safe_count": len(connected),
            "maximum_connected_radius_m": max((math.hypot(*cell) for cell in connected), default=0.0),
            "isolated_successes_excluded": sum(item["capture_success"] for item in subset) - len(connected),
        }
    nominal = next(item for item in records if all(math.isclose(item[key], 0.0, abs_tol=1e-12) for key in ("ex_m", "ey_m", "erx_rad", "ery_rad")))
    physics_transients = [item for item in records if item.get("maximum_tcp_step_m", 0.0) > 0.002 or item["terminal_state"] == "SAFE_ABORT_PHYSICS_TRANSIENT"]
    report = {
        "schema_version": "kcg_d38999_bcapture_connected_v1",
        "raw_input": str(raw_path),
        "raw_trial_count": len(records),
        "experiment_valid": not physics_transients,
        "nominal_capture_success": nominal["capture_success"],
        "nominal_terminal_state": nominal["terminal_state"],
        "hard_gate_count": sum(item["hard_gate_triggered"] for item in records),
        "safe_recovery_count": sum(item["safe_recovery"] for item in records),
        "physics_transient_count": len(physics_transients),
        "Bcapture_connected_slices": slices,
        "authorization_rule": "bilinear enclosure must lie in origin-connected successful cells; isolated success is rejected",
        "open_loop_0_26mm_gate": {
            "value_m": 0.00026,
            "derivation": "0.65 times 0.40 mm last success in single-axis one-shot 4 mm/s no-feedback free-coast sweep",
            "is_single_side_radial_clearance": False,
            "is_compliant_capture_region": False,
            "is_controller_success_boundary": False,
            "role": "legacy conservative open-loop regression gate only",
        },
    }
    (output_dir / "bcapture_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    nominal_slice = slices.get("erx_+0.0000000000_ery_+0.0000000000", {})
    text = f"""# Four-dimensional compliant capture analysis

`0.26 mm` was **not** a single-side clearance and was **not** measured with
the current compliant controller. It was 65% of the last 0.40 mm success in a
single-axis, one-shot 4 mm/s, no-feedback free-coast regression. The modeled
single-side radial clearance is `{raw.get('radial_clearance_m', 0.0)*1000:.3f} mm`.

This report defines `Bcapture(ex, ey, erx, ery)` from origin-connected successful
XY cells in each Rx/Ry slice. Isolated successes are rejected.

- raw trials: {len(records)}
- experiment valid: {report['experiment_valid']}
- nominal capture success: {report['nominal_capture_success']}
- hard safety aborts: {report['hard_gate_count']}
- safe soft-gate recoveries: {report['safe_recovery_count']}
- physics transients: {report['physics_transient_count']}
- nominal-slice connected cells: {nominal_slice.get('connected_safe_count', 0)}
- nominal-slice connected radius: {nominal_slice.get('maximum_connected_radius_m', 0.0)*1000:.3f} mm

Truth was used to author the scan and score outcomes only. Controller inputs
were delayed wrist wrench, TCP proprioception, and controller history; no
contact normal or collider identity was supplied.
"""
    (output_dir / "bcapture_analysis.md").write_text(text, encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("experiment_valid", "nominal_capture_success", "hard_gate_count", "safe_recovery_count", "physics_transient_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
