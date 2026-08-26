#!/usr/bin/env python3
"""Replay one opposition-60 Isaac trace against registered exact meshes offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.observed_state_replay import (
    ObservedHandStateEvaluator,
)
ROOT = Path(__file__).resolve().parents[2]
OBJECT_B = "te_deutsch_d38999_26fj35pn_step"
EXPECTED_CONFIG = ROOT / "src/kcg_connector/config/carts_nailfree_height_projected.yaml"
EVALUATOR_SOURCE = (ROOT / "src/kcg_connector/kcg_connector/grasp/carts_v2/"
                    "observed_state_replay.py")
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def _resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _transform(position, quaternion_wxyz) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    _require(position.shape == (3,) and quaternion.shape == (4,)
             and np.isfinite(position).all() and np.isfinite(quaternion).all(),
             "object pose must be finite xyz plus wxyz")
    norm = float(np.linalg.norm(quaternion))
    _require(norm > 0.0, "object quaternion has zero norm")
    w, x, y, z = quaternion / norm
    rotation = np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def _minimum(rows: list[dict], first: str, second: str):
    values = [row["geometry"][first][second] for row in rows
              if row["geometry"][first][second] is not None]
    return None if not values else float(min(values))


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _replay(trace_path: Path, report: dict) -> None:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    _require(trace.get("schema_version") == "carts_opposition60_initial_penetration_v1"
             and trace.get("object_id") == OBJECT_B and trace.get("mode") == "initial-penetration"
             and trace.get("hardware_authorized") is False
             and trace.get("formal_dynamic_pass") is False
             and trace.get("research_dynamic_pass") is False,
             "trace identity or authorization boundary changed")
    config_row = (trace.get("evidence_binding") or {}).get("config") or {}
    config_path = _resolve(config_row.get("path", ""))
    _require(config_path == EXPECTED_CONFIG.resolve() and config_path.is_file()
             and _sha256(config_path) == config_row.get("sha256"),
             "trace-bound configuration hash changed")
    report["evidence_binding"]["config"] = {
        "path": str(config_path), "sha256": _sha256(config_path)}
    samples = trace.get("samples")
    _require(isinstance(samples, list) and len(samples) == 60,
             "trace must contain exactly 60 physics samples")
    target = np.asarray(trace["pose_binding"][
        "world_from_handbase_target_row_major"], dtype=np.float64).reshape(4, 4)
    inputs = load_v2_inputs(ROOT, config_path=config_path, object_id=OBJECT_B)
    evaluator = ObservedHandStateEvaluator(inputs)
    rows, previous = [], None
    for index, sample in enumerate(samples):
        poses = sample.get("object_poses")
        _require(isinstance(poses, list) and len(poses) == 1,
                 f"sample {index} must contain exactly one object pose")
        world_from_object = _transform(
            poses[0]["position_m"], poses[0]["orientation_wxyz"])
        state = {"world_from_handbase": target,
                 "joint_positions_by_name": sample["joint_positions_rad"],
                 "world_from_object": world_from_object}
        geometry = evaluator.evaluate(
            target, state["joint_positions_by_name"], world_from_object,
            previous_state=previous)
        rows.append({"sample_index": index, "step": sample.get("step"),
                     "simulation_time_s": sample.get("simulation_time_s"),
                     "geometry": geometry})
        previous = state
    task_counts = [sum(bool(item["task_grip_surface_contact"])
        for item in row["geometry"]["task_grip_surface_by_finger"].values())
        for row in rows]
    all_geometry_pass = all(row["geometry"]["fail_closed"] is False for row in rows)
    physics = trace.get("physics") or {}
    trace_gate = bool((trace.get("runtime_gates") or {}).get("INITIAL_PENETRATION") is True)
    timing_pass = bool(physics.get("step_count") == 60
        and math.isclose(float(physics.get("physics_time_advanced_s", -1)), 0.5,
                         rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(float(physics.get("dt_s", -1)), 1.0 / 120.0,
                         rel_tol=0.0, abs_tol=1.0e-15))
    command_pass = trace.get("closure_command_count") == 0 and trace.get(
        "lift_command_count") == 0
    truth_pass = (trace.get("online_truth_used_for_control") is False
        and trace.get("truth_evaluation_timing") ==
            "POST_STEP_LOGGING_AND_POST_RUN_GATE_ONLY_NO_TARGET_FEEDBACK")
    accepted = bool(trace_gate and timing_pass and command_pass and truth_pass
                    and all_geometry_pass)
    report.update({"status": ("OFFLINE_EXACT_REPLAY_ACCEPTED" if accepted else
                              "OFFLINE_EXACT_REPLAY_REJECTED"),
        "trace_initial_penetration_gate": trace_gate,
        "sample_count": len(rows), "samples": rows,
        "aggregate": {"all_fail_closed_false": all_geometry_pass,
            "minimum_table_clearance_m": _minimum(rows, "table_top", "minimum_clearance_m"),
            "minimum_self_clearance_m": _minimum(rows, "self_collision", "minimum_clearance_m"),
            "minimum_non_task_clearance_m": _minimum(
                rows, "non_task_hand_object", "minimum_clearance_m"),
            "actual_task_contact_observation_count": int(sum(task_counts)),
            "maximum_simultaneous_task_contact_count": int(max(task_counts)),
            "final_task_contact_count": int(task_counts[-1])},
        "gate_inputs": {"timing_60_steps_0_5_s": timing_pass,
                        "zero_closure_and_lift_commands": command_pass,
                        "truth_not_used_for_control": truth_pass},
        "accepted_initial_penetration_pass": accepted})


def main() -> int:
    args = _arguments()
    trace_path, output = _resolve(args.trace), _resolve(args.output)
    _require(not output.exists(), f"refusing to overwrite evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "carts_opposition60_observed_trace_replay_v1",
        "status": "OFFLINE_EXACT_REPLAY_FAILED_CLOSED",
        "offline_post_run_only": True, "online_control_use_allowed": False,
        "accepted_initial_penetration_pass": False,
        "hardware_authorized": False, "formal_dynamic_pass": False,
        "research_dynamic_pass": False, "runtime_binding_accepted": False,
        "evidence_binding": {"trace": {"path": str(trace_path),
            "sha256": (_sha256(trace_path) if trace_path.is_file() else None)},
            "replay_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve())},
            "evaluator_source": {"path": str(EVALUATOR_SOURCE.resolve()),
            "sha256": _sha256(EVALUATOR_SOURCE)}}, "errors": []}
    try:
        _replay(trace_path, report)
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error),
                                 "traceback": traceback.format_exc()})
    output.write_text(json.dumps(_clean(report), indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "accepted_initial_penetration_pass": report[
                          "accepted_initial_penetration_pass"]}, sort_keys=True))
    return 0 if report["accepted_initial_penetration_pass"] else 2
if __name__ == "__main__":
    raise SystemExit(main())
