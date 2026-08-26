#!/usr/bin/env python3
"""Evaluate one saved opposition-60 first-finger trace against exact meshes."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback
import numpy as np
from kcg_connector.grasp.carts_v2.b0_surface_semantics import (
    bind_b0_external_load_bearing_surfaces,
)
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.observed_state_replay import (
    ObservedHandStateEvaluator,
)
ROOT = Path(__file__).resolve().parents[2]
OBJECT_B = "te_deutsch_d38999_26fj35pn_step"
CONFIG = ROOT / "src/kcg_connector/config/carts_nailfree_height_projected.yaml"
RUNNER = ROOT / "scripts/carts_v2/run_opposition60_local_contact.py"
EVALUATOR = (ROOT / "src/kcg_connector/kcg_connector/grasp/carts_v2/"
             "observed_state_replay.py")
B0_SOURCE = (ROOT / "src/kcg_connector/kcg_connector/grasp/carts_v2/"
             "b0_surface_semantics.py")
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--initial-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def _resolve(value: Path | str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
def _transform(position, quaternion_wxyz) -> np.ndarray:
    position, quaternion = map(lambda row: np.asarray(row, np.float64),
                               (position, quaternion_wxyz))
    _require(position.shape == (3,) and quaternion.shape == (4,)
             and np.isfinite(position).all() and np.isfinite(quaternion).all(),
             "object pose must be finite xyz plus wxyz")
    norm = float(np.linalg.norm(quaternion))
    _require(norm > 0.0, "object quaternion has zero norm")
    w, x, y, z = quaternion / norm
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
                      (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
                      (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))
    result[:3, 3] = position
    return result
def _state(sample: dict) -> dict:
    hand = np.asarray(sample["world_from_handbase_row_major"], np.float64).reshape(4, 4)
    poses = sample.get("object_poses")
    _require(np.isfinite(hand).all() and isinstance(poses, list) and len(poses) == 1,
             "each sample must bind one finite hand pose and one object pose")
    return {"world_from_handbase": hand,
            "joint_positions_by_name": sample["joint_positions_rad"],
            "world_from_object": _transform(
                poses[0]["position_m"], poses[0]["orientation_wxyz"])}
def _bound(binding: dict, name: str, expected: Path) -> None:
    row = binding.get(name) or {}
    _require(_resolve(row.get("path", "")) == expected.resolve()
             and expected.is_file() and row.get("sha256") == _sha256(expected),
             f"trace-bound {name} path or hash changed")


def _bind_object_surface_method(inputs, trace: dict, report: dict):
    task_row = (trace.get("evidence_binding") or {}).get("task_ik") or {}
    task_path = _resolve(task_row.get("path", ""))
    if not task_path.is_file():
        return inputs, False
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task.get("schema_version") != "carts_contactopt_b0_recheck_v1":
        return inputs, False
    ready = [row for row in task.get("candidates", [])
             if row.get("local_isaac_input_ready") is True]
    audit = task.get("b0_surface_audit") or {}
    _require(task_row.get("sha256") == _sha256(task_path) and len(ready) == 1
             and task.get("local_isaac_input_count") == 1
             and ready[0].get("candidate_id") == trace.get("candidate_id")
             and audit.get("method") == "EXTERNAL_LOAD_BEARING_SURFACE_B0"
             and audit.get("legacy_primary_secondary_are_hard_gates") is False
             and audit.get("normal_alignment_is_object_semantic_hard_gate") is False,
             "trace-bound B0 surface identity changed")
    report["evidence_binding"]["b0_task"] = {
        "path": str(task_path), "sha256": _sha256(task_path)}
    report["evidence_binding"]["b0_surface_source"] = {
        "path": str(B0_SOURCE.resolve()), "sha256": _sha256(B0_SOURCE)}
    report["object_surface_method"] = audit["method"]
    return bind_b0_external_load_bearing_surfaces(inputs), True
def _verify(trace: dict, initial_path: Path, config: Path) -> tuple[list[dict], dict, Path]:
    controller = trace.get("controller") or {}
    controller_complete = bool(
        trace.get("status") == "FIRST_FINGER_CONTROLLER_TRACE_COMPLETE"
        and trace.get("first_finger_controller_trace_pass") is True)
    endpoint_timeout = bool(
        trace.get("status") == "FAILED_CLOSED"
        and trace.get("first_finger_controller_trace_pass") is False
        and controller.get("abort_reason") == "FINGER_1_NO_CONTACT_SIGNAL")
    endpoint_overshoot_abort = bool(
        trace.get("status") == "FAILED_CLOSED"
        and controller.get("abort_reason") == "NONEXECUTABLE_ENDPOINT_OVERSHOOT_ABORT")
    _require(all((trace.get("schema_version") == "carts_opposition60_local_contact_v1",
                  trace.get("mode") == "first-finger-diagnostic",
                  trace.get("object_id") == OBJECT_B,
                  controller_complete or endpoint_timeout or endpoint_overshoot_abort,
                  trace.get("first_finger_diagnostic_pass") is False,
                  trace.get("hardware_authorized") is False,
                  trace.get("formal_dynamic_pass") is False,
                  trace.get("research_dynamic_pass") is False,
                  trace.get("runtime_binding_accepted") is False)),
             "first-finger trace identity or evidence boundary changed")
    binding = trace.get("evidence_binding") or {}
    for name, path in (("runner_source", RUNNER), ("config", config),
                       ("initial_trace", initial_path)):
        _bound(binding, name, path)
    plan_row = binding.get("contact_endpoint_plan") or {}
    plan_path = _resolve(plan_row.get("path", ""))
    _require(plan_path.is_file() and plan_row.get("sha256") == _sha256(plan_path),
             "trace-bound contact endpoint plan changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_binding = plan.get("evidence_binding") or {}
    for name, path in (("runner_source", RUNNER), ("config", config),
                       ("evaluator_source", EVALUATOR)):
        _bound(plan_binding, name, path)
    execution = float(plan.get("execution_target_rad", math.nan))
    upper = float(plan.get("first_nonexecutable_target_rad", math.nan))
    _require(plan.get("status") == "OFFLINE_LAST_SEMANTICALLY_VALID_ENDPOINT_ACCEPTED"
             and plan.get("object_id") == OBJECT_B and np.isfinite(execution)
             and execution < upper and controller.get("bound_execution_target_rad") == execution,
             "trace and semantic endpoint plan disagree")
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    _require((initial.get("runtime_gates") or {}).get("INITIAL_PENETRATION") is True
             and initial.get("closure_command_count") == initial.get("lift_command_count") == 0
             and initial.get("online_truth_used_for_control") is False,
             "bound initial trace no longer supplies the accepted zero-command gate")
    samples = trace.get("samples")
    _require(isinstance(samples, list) and samples, "first-finger samples are missing")
    physics = trace.get("physics") or {}
    _require(physics.get("step_count") == len(samples)
             and math.isclose(float(physics.get("dt_s", -1)), 1/120, abs_tol=1e-15),
             "trace sample count or physics period changed")
    targets = [float(row["active_targets_rad"]["f1j2"]) for row in samples]
    _require(max(targets) <= execution + 1e-12,
             "runtime commanded beyond the semantic-valid endpoint")
    return samples, plan, plan_path
def _minimum(current, value):
    return value if current is None else min(current, value)
def _evaluate(trace: dict, report: dict, initial_path: Path, config: Path) -> None:
    samples, plan, plan_path = _verify(trace, initial_path, config)
    report["evidence_binding"].update({name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in (("runner", RUNNER), ("config", config),
                          ("initial_trace", initial_path))})
    report["evidence_binding"]["contact_endpoint_plan"] = {
        "path": str(plan_path), "sha256": _sha256(plan_path)}
    inputs = load_v2_inputs(ROOT, config_path=config, object_id=OBJECT_B)
    inputs, b0_mode = _bind_object_surface_method(inputs, trace, report)
    evaluator, states = ObservedHandStateEvaluator(inputs), [_state(row) for row in samples]
    minima = {"table_m": None, "self_m": None, "non_task_m": None}
    safety_failures, table_margin_failures, non_task_band_diagnostic = [], [], []
    contact_distance = float(inputs.config.section(
        "closure_prediction")["contact_distance_m"])
    table_required = float(inputs.config.section(
        "height_projection")["table_operation_clearance_m"])
    table_tolerance = float(inputs.config.section(
        "fast_filter")["table_penetration_tolerance_m"])
    for index, state in enumerate(states):
        result = evaluator.evaluate_safety(**state)
        if result["fail_closed"]:
            safety_failures.append(index)
        table_value = result["table_top"]["minimum_clearance_m"]
        non_task_value = result["non_task_hand_object"]["minimum_clearance_m"]
        if (b0_mode and (table_value is None or float(table_value)
                        < table_required - table_tolerance)):
            table_margin_failures.append(index)
        if (b0_mode and (non_task_value is None or float(non_task_value)
                        <= contact_distance)):
            non_task_band_diagnostic.append(index)
        for key, group in (("table_m", "table_top"), ("self_m", "self_collision"),
                           ("non_task_m", "non_task_hand_object")):
            value = result[group]["minimum_clearance_m"]
            if value is not None:
                minima[key] = _minimum(minima[key], float(value))
    confirmed = [i for i, row in enumerate(samples)
                 if row.get("controller_state") == "CONTACT_CONFIRMED"]
    settle = [i for i, row in enumerate(samples)
              if row.get("controller_state") == "CONTACT_SETTLE"]
    hold = [i for i, row in enumerate(samples)
            if row.get("controller_state") == "HOLD"]
    endpoint_timeout = (trace.get("controller", {}).get("abort_reason")
                        == "FINGER_1_NO_CONTACT_SIGNAL")
    endpoint_target = float(samples[-1]["active_targets_rad"]["f1j2"])
    endpoint = [i for i, row in enumerate(samples)
                if abs(float(row["active_targets_rad"]["f1j2"])
                       - endpoint_target) <= 1e-12]
    _require((confirmed and len(hold) == 60) or (endpoint_timeout and endpoint),
             "neither accepted HOLD nor endpoint-timeout evidence is complete")
    full_indices = set(settle) | set(hold) | set(endpoint)
    for index in confirmed + endpoint[:1]:
        full_indices.update(i for i in (index-1, index, index+1)
                            if 0 <= i < len(samples))
    contacts = {name: [] for name in ("finger_1_pad", "finger_2_pad", "finger_3_pad")}
    first_contacts, motion_compatible, full_failures, forbidden_first = [], [], [], []
    for index in sorted(full_indices):
        previous = None if index == 0 else states[index-1]
        result = evaluator.evaluate(**states[index], previous_state=previous)
        if result["fail_closed"]:
            full_failures.append(index)
        for name, row in result["task_grip_surface_by_finger"].items():
            if row["task_grip_surface_contact"]:
                contacts[name].append(index)
            if row["forbidden_first"]:
                forbidden_first.append({"sample_index": index, "finger": name})
            if row["first_contact_from_previous_state"]:
                first_contacts.append({"sample_index": index, "finger": name,
                                       "motion_compatible": row["first_contact_motion_compatible"]})
                if row["first_contact_motion_compatible"]:
                    motion_compatible.append(index)
    physx_count = sum(int(row["post_step_contact_truth_audit_only"]["counts"]
                          .get("hand_object", 0)) for row in samples)
    positions = np.asarray([row["object_poses"][0]["position_m"] for row in samples], np.float64)
    object_motion = float(np.max(np.linalg.norm(positions-positions[0], axis=1)))
    target_ranges = {name: float(np.ptp([row["active_targets_rad"][name]
                                         for row in samples]))
                     for name in ("f2j1", "f3j2")}
    actual_ranges = {name: float(np.ptp([row["joint_positions_rad"][name]
                                         for row in samples]))
                     for name in ("f2j1", "f2j2", "f3j2", "f3j3")}
    physics = trace["physics"]
    engine = bool(physics.get("engine_observation_pass_for_this_atomic_run") is True
                  and physics.get("backend", {}).get("pass") is True
                  and physics.get("log", {}).get("capacity_warning_count") == 0
                  and physics.get("log", {}).get("physx_error_lines") == [])
    truth = bool(trace.get("online_truth_used_for_control") is False
                 and trace.get("truth_evaluation_timing") ==
                 "POST_STEP_LOGGING_AND_POST_RUN_GATE_ONLY_NO_TARGET_FEEDBACK")
    other_idle = bool(trace.get("second_third_finger_command_count") == 0
                      and max(target_ranges.values()) <= 1e-12)
    exact_contact = bool(hold and all(index in contacts["finger_1_pad"] for index in hold)
                         and not contacts["finger_2_pad"] and not contacts["finger_3_pad"])
    endpoint_proximity = bool(
        endpoint and any(index in contacts["finger_1_pad"] for index in endpoint))
    endpoint_limit = float(plan["first_nonexecutable_target_rad"])
    endpoint_overshoot = [index for index, row in enumerate(samples)
        if float(row["joint_positions_rad"]["f1j2"]) >= endpoint_limit]
    safe = bool(not safety_failures and not table_margin_failures
                and not full_failures
                and not forbidden_first and not endpoint_overshoot)
    accepted = bool(safe and exact_contact and motion_compatible and physx_count > 0
                    and object_motion <= 0.001 and other_idle and engine and truth)
    false_proxy = bool(trace.get("controller", {}).get("contact_targets_rad")
                       and (physx_count == 0 or not contacts["finger_1_pad"]))
    proximity_only = bool(endpoint_timeout and endpoint_proximity and physx_count == 0)
    unresolved_boundary = bool(forbidden_first and not full_failures and physx_count == 0)
    status = ("B0_TABLE_OPERATION_MARGIN_REJECT" if table_margin_failures else
              "SEMANTIC_PROXIMITY_BOUNDARY_UNRESOLVED" if unresolved_boundary else
              "SEMANTICALLY_INVALID_FORBIDDEN_FIRST_CONTACT" if forbidden_first else
              "NONEXECUTABLE_ENDPOINT_OVERSHOOT" if endpoint_overshoot else
              "FALSE_CONTACT_PROXY" if false_proxy else
              "NO_PHYSX_CONTACT_AT_LAST_SEMANTIC_VALID_ENDPOINT" if proximity_only else
              "FIRST_FINGER_OFFLINE_CONTACT_ACCEPTED" if accepted else
              "FIRST_FINGER_OFFLINE_CONTACT_REJECTED")
    report.update({"status": status, "accepted_first_finger_contact_pass": accepted,
        "sample_count": len(samples), "safety_evaluation_count": len(samples),
        "full_contact_evaluation_indices": sorted(full_indices),
        "contact_confirmed_indices": confirmed, "contact_settle_indices": settle,
        "hold_indices": hold, "endpoint_indices": endpoint,
        "geometry": {"all_cycles_safe": safe, "safety_failure_indices": safety_failures,
                     "table_operation_margin_failure_indices": table_margin_failures,
                     "non_task_contact_band_diagnostic_indices": non_task_band_diagnostic,
                     "required_table_clearance_m": table_required,
                     "table_numerical_tolerance_m": table_tolerance,
                     "non_task_contact_distance_m": contact_distance,
                     "full_evaluation_failure_indices": full_failures,
                     "minimum_clearances": minima},
        "task_contact": {"contact_indices_by_finger": contacts,
                         "forbidden_first_rows": forbidden_first,
                         "nonexecutable_endpoint_overshoot_indices": endpoint_overshoot,
                         "first_contacts": first_contacts,
                         "motion_compatible_first_contact_indices": motion_compatible,
                         "all_60_hold_samples_have_finger_1_contact": exact_contact,
                         "endpoint_has_predicted_contact_proximity": endpoint_proximity},
        "physx": {"hand_object_record_count": physx_count,
                  "trace_contact_total": trace.get("contact_totals", {}).get("hand_object"),
                  "engine_health_pass": engine},
        "object_maximum_translation_from_initial_m": object_motion,
        "object_motion_at_most_1mm": object_motion <= 0.001,
        "second_third_fingers": {"target_range_rad": target_ranges,
                                 "observed_position_range_rad": actual_ranges,
                                 "not_commanded_pass": other_idle},
        "truth_boundary_pass": truth,
        "classification_reason": ("sampled table clearance fell below the registered B0 operation margin"
                                  if table_margin_failures else
                                  "functional-protected surface entered the offline proximity band without mesh intersection or PhysX contact"
                                  if unresolved_boundary else
                                  "functional-protected surface became first or the nonexecutable bound was crossed"
                                  if forbidden_first or endpoint_overshoot else
                                  "joint-side contact proxy had no PhysX/exact TASK contact"
                                  if false_proxy else "all offline gates passed" if accepted
                                  else "last semantic-valid endpoint was reached without PhysX contact"
                                  if proximity_only else "one or more offline gates rejected")})
def main() -> int:
    args = _arguments()
    trace_path, initial_path, output = map(
        _resolve, (args.trace, args.initial_trace, args.output))
    config = _resolve(args.config)
    _require(not output.exists(), f"refusing to overwrite evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "carts_opposition60_first_finger_evaluation_v1",
        "status": "FAILED_CLOSED", "offline_post_run_only": True,
        "online_control_use_allowed": False, "accepted_first_finger_contact_pass": False,
        "hardware_authorized": False, "formal_dynamic_pass": False,
        "research_dynamic_pass": False, "runtime_binding_accepted": False,
        "evidence_binding": {"trace": {"path": str(trace_path),
            "sha256": _sha256(trace_path) if trace_path.is_file() else None},
            "evaluation_source": {"path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve())},
            "evaluator_source": {"path": str(EVALUATOR), "sha256": _sha256(EVALUATOR)}},
        "errors": []}
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        _evaluate(trace, report, initial_path, config)
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error),
                                 "traceback": traceback.format_exc()})
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "accepted": report["accepted_first_finger_contact_pass"]},
                     sort_keys=True))
    return 0 if report["accepted_first_finger_contact_pass"] else 2
if __name__ == "__main__":
    raise SystemExit(main())
