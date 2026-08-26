#!/usr/bin/env python3
"""Plan one hash-bound last semantic-valid opposition-60 first-finger endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src/kcg_connector"
sys.path.insert(0, str(PACKAGE_ROOT))

from kcg_connector.grasp.carts_v2.models import (  # noqa: E402
    joint_positions_for_phases,
    load_v2_inputs,
)
from kcg_connector.grasp.carts_v2.b0_surface_semantics import (  # noqa: E402
    b0_surface_audit,
    bind_b0_external_load_bearing_surfaces,
)
from kcg_connector.grasp.carts_v2.observed_state_replay import (  # noqa: E402
    ObservedHandStateEvaluator,
)
from kcg_connector.robot_model import expand_active_hand_positions  # noqa: E402


CONFIG = ROOT / "src/kcg_connector/config/carts_nailfree_height_projected.yaml"
RUNNER = ROOT / "scripts/carts_v2/run_opposition60_local_contact.py"
EVALUATOR = PACKAGE_ROOT / "kcg_connector/grasp/carts_v2/observed_state_replay.py"
B0_RECHECK = ROOT / "scripts/carts_v2/run_contactopt_b0_recheck.py"
B0_SURFACE = PACKAGE_ROOT / "kcg_connector/grasp/carts_v2/b0_surface_semantics.py"
PAD_NAME = "finger_1_pad"
STEP_RAD = 0.0015


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bounded_targets(start: float, stop: float) -> list[float]:
    _require(np.isfinite(start) and np.isfinite(stop) and stop >= start, "invalid target range")
    values = [float(start)]
    while values[-1] < stop - 1.0e-12:
        values.append(float(min(values[-1] + STEP_RAD, stop)))
    return values


def _minimum(rows: list[dict], *keys: str) -> float:
    values = []
    for row in rows:
        value = row
        for key in keys:
            value = value[key]
        values.append(float(value))
    return min(values)


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _bound_file(path_value: str, expected_sha256: str, label: str) -> Path:
    path = Path(path_value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    _require(path.is_file() and _sha256(path) == expected_sha256,
             f"{label} path or hash changed")
    return path


def _b0_context(task: dict, config_path: Path) -> dict:
    rows = task.get("candidates")
    _require(isinstance(rows, list), "B0 candidate rows are missing")
    ready = [row for row in rows if isinstance(row, dict)
             and row.get("local_isaac_input_ready") is True]
    _require(task.get("local_isaac_input_count") == len(ready) == 1,
             "B0 endpoint requires exactly one local-Isaac-ready candidate")
    row = ready[0]
    seed = row.get("input_seed") or {}
    quality = row.get("task_quality") or {}
    audit = task.get("b0_surface_audit") or {}
    pregrasp_filter = row.get("pregrasp_filter") or {}
    closure_filter = row.get("closure_filter") or {}
    required = float(row.get("required_table_operation_clearance_m", np.nan))
    tolerance = float(row.get("table_clearance_numerical_tolerance_m", np.nan))
    table_values = (pregrasp_filter.get("minimum_table_clearance_m"),
                    closure_filter.get("minimum_table_clearance_m"))
    _require(all((task.get("hardware_authorized") is False,
                  task.get("formal_dynamic_pass") is False,
                  task.get("research_dynamic_pass") is False,
                  row.get("candidate_id") == seed.get("candidate_id"),
                  task.get("object_id") == seed.get("object_id"),
                  row.get("sampled_raw_mesh_geometry_pass") is True,
                  row.get("sampled_table_operation_clearance_pass") is True,
                  row.get("nominal_12n_task_pass") is True,
                  quality.get("nominal_gravity_lift_balance_pass") is True,
                  float(quality.get("nominal_operation_force_cap_n", np.nan)) == 12.0,
                  (row.get("bounded_ik") or {}).get("status")
                  == "BOUNDED_IK_PASS_NOT_PATH_COLLISION",
                  row.get("full_arm_path_collision_checked") is False,
                  audit.get("method") == "EXTERNAL_LOAD_BEARING_SURFACE_B0",
                  audit.get("legacy_primary_secondary_are_hard_gates") is False,
                  audit.get("normal_alignment_is_object_semantic_hard_gate") is False,
                  np.isfinite(required), np.isfinite(tolerance), required > 0.0,
                  tolerance >= 0.0,
                  all(value is not None and np.isfinite(float(value))
                      and float(value) >= required - tolerance
                      for value in table_values))),
             "B0 identity, 12 N nominal, or sampled-table gate changed")
    base = _bound_file(task.get("base_physical_config", ""),
                       task.get("base_physical_config_sha256", ""), "B0 base config")
    method = _bound_file(task.get("method_config", ""),
                         task.get("method_config_sha256", ""), "B0 method config")
    manifest_path = _bound_file(task.get("seed_manifest", ""),
                                task.get("seed_manifest_sha256", ""), "B0 seed manifest")
    producer_row = task.get("source") or {}
    producer = _bound_file(producer_row.get("path", ""),
                           producer_row.get("sha256", ""), "B0 recheck producer")
    _require(base == config_path and producer == B0_RECHECK.resolve(),
             "B0 supplied config or producer identity changed")
    inputs = bind_b0_external_load_bearing_surfaces(load_v2_inputs(
        ROOT, config_path=base, object_id=task["object_id"]))
    _require(task.get("object_mesh_sha256")
             == inputs.object_contract.model.provenance.source_sha256
             and audit == b0_surface_audit(inputs),
             "B0 object mesh or bound surface semantics changed")
    manifest = _load(manifest_path)
    originals = [item for item in manifest.get("generated_candidates", ())
                 if item.get("candidate_id") == row["candidate_id"]]
    projections = [item for item in (task.get("table_projection") or ())
                   if item.get("candidate_id") == row["candidate_id"]]
    _require(len(originals) == len(projections) == 1
             and manifest.get("object_id") == task["object_id"]
             and manifest.get("object_mesh_sha256") == task["object_mesh_sha256"]
             and manifest.get("base_physical_config_sha256")
             == task.get("base_physical_config_sha256")
             and manifest.get("method_config_sha256")
             == task.get("method_config_sha256")
             and manifest.get("b0_surface_audit") == audit
             and projections[0].get("status") == "PROJECTED",
             "B0 projected seed lineage is incomplete")
    original, projection = originals[0], projections[0]
    projected = np.asarray(original["object_from_hand"], dtype=np.float64).reshape(4, 4)
    rise = float(projection.get("required_rise_m", np.nan))
    _require(np.isfinite(rise) and 0.0 <= rise <= 0.020,
             "B0 table projection exceeds its registered bound")
    projected[:3, 3] += inputs.frozen_world_from_object[:3, :3].T @ np.asarray(
        (0.0, 0.0, rise), dtype=np.float64)
    original_identity = {key: value for key, value in original.items()
                         if key != "object_from_hand"}
    seed_identity = {key: value for key, value in seed.items()
                     if key != "object_from_hand"}
    target = np.asarray(row.get("target_world_from_handbase_row_major"),
                        dtype=np.float64).reshape(4, 4)
    _require(original_identity == seed_identity
             and np.allclose(projected, np.asarray(seed["object_from_hand"]).reshape(4, 4),
                             atol=1.0e-12, rtol=0.0)
             and np.allclose(target, inputs.frozen_world_from_object @ projected,
                             atol=1.0e-12, rtol=0.0),
             "B0 projected candidate pose or identity changed")
    prediction = row.get("closure_prediction") or {}
    phases = tuple(prediction.get("final_closure_phases") or ())
    pre = np.asarray(seed["pregrasp_joint_positions_rad"], dtype=np.float64)
    predicted = np.asarray(prediction.get("final_joint_positions_rad"), dtype=np.float64)
    recomputed = joint_positions_for_phases(
        inputs, phases, reference_joint_positions_rad=pre)
    intervals = [item for item in (row.get("proxy_interval") or {}).get(
        "finger_intervals", ()) if item.get("finger_index") == 1]
    _require(prediction.get("status") == "CLOSURE_SURVIVE"
             and len(prediction.get("contacts") or ()) == 3
             and predicted.shape == pre.shape == (4,)
             and np.allclose(predicted, recomputed, atol=1.0e-12, rtol=0.0)
             and len(intervals) == 1,
             "B0 exact contact prediction identity changed")
    proxy_upper = float(intervals[0].get("proxy_q_safe_max_rad", np.nan))
    _require(np.isfinite(proxy_upper) and predicted[1] < proxy_upper,
             "B0 proxy q_safe_max cannot bound the raw-mesh endpoint search")
    return {"inputs": inputs, "row": row, "anchor": seed, "pre": pre,
            "predicted": predicted, "maximum_q": proxy_upper, "target": target,
            "object_id": task["object_id"], "proxy_upper": proxy_upper,
            "paths": {"b0_recheck_source": producer, "b0_surface_source": B0_SURFACE,
                      "method_config": method, "seed_manifest": manifest_path},
            "audit": audit, "sampled_table": {
                "pregrasp_minimum_clearance_m": float(table_values[0]),
                "closure_minimum_clearance_m": float(table_values[1]),
                "required_clearance_m": required,
                "numerical_tolerance_m": tolerance}}


def plan_endpoint(task_path: Path, config_path: Path = CONFIG) -> dict:
    task = _load(task_path)
    b0 = task.get("schema_version") == "carts_contactopt_b0_recheck_v1"
    b0_context = _b0_context(task, config_path) if b0 else None
    if b0:
        anchor, row = b0_context["anchor"], b0_context["row"]
        object_id, inputs = b0_context["object_id"], b0_context["inputs"]
    else:
        anchor = task["selected_geometric_anchor"]
        row = task["task_and_bounded_ik"][0]
        survivor = task["survivor_candidates"][0]
        object_id = anchor["object_id"]
        palm_angle_deg = float(task["requested_palm_angle_deg"])
        inputs = load_v2_inputs(ROOT, config_path=config_path, object_id=object_id)
        _require(
            anchor["candidate_id"] == row["candidate_id"] == survivor["candidate_id"]
            and anchor["object_id"] == survivor["object_id"] == object_id
            and 45.0 <= palm_angle_deg <= 75.0
            and abs(palm_angle_deg - round(palm_angle_deg)) < 1e-9
            and abs(float(anchor["palm_configuration_deg"]) - palm_angle_deg) < 1e-9
            and task["survivor_count"] == 1
            and len(task["task_and_bounded_ik"]) == 1
            and task["config_sha256"] == _sha256(config_path),
            "opposition 45-to-75-degree survivor task identity changed",
        )
    dynamic = inputs.config.section("dynamic")
    _require(
        float(dynamic["physics_dt_s"]) == 1.0 / 120.0
        and float(dynamic["finger_maximum_speed_rad_s"]) * float(
            dynamic["physics_dt_s"]
        ) == STEP_RAD,
        "registered control-step identity changed",
    )
    if b0:
        pre = b0_context["pre"]
        predicted = b0_context["predicted"]
        maximum_q = b0_context["maximum_q"]
        target = b0_context["target"]
    else:
        pre = np.asarray(anchor["pregrasp_joint_positions_rad"], dtype=np.float64)
        predicted = joint_positions_for_phases(
            inputs, tuple(row["fresh_contact_stop_phases"]),
            reference_joint_positions_rad=pre)
        maximum_phases = list(anchor["pregrasp_closure_phases"])
        maximum_phases[0] = float(inputs.config.section("candidate_generation")[
            "maximum_closure_phase"])
        maximum_q = joint_positions_for_phases(
            inputs, tuple(maximum_phases), reference_joint_positions_rad=pre)[1]
        target = np.asarray(
            row["target_world_from_handbase_row_major"], dtype=np.float64
        ).reshape(4, 4)
    evaluator = ObservedHandStateEvaluator(inputs)
    if b0:
        q_values = _bounded_targets(float(predicted[1]), float(maximum_q))
        previous_active = pre.copy()
        previous_active[1] = float(predicted[1] - STEP_RAD)
        _require(previous_active[1] >= pre[1],
                 "B0 predicted contact lacks one frozen-step free-side predecessor")
    else:
        q_values = _bounded_targets(float(pre[1]), float(predicted[1]))
        q_values.extend(
            float(predicted[1] + index * STEP_RAD)
            for index in range(1, 1 + int(np.ceil(
                (maximum_q - predicted[1]) / STEP_RAD)))
            if predicted[1] + index * STEP_RAD <= maximum_q + 1.0e-12)
        previous_active = None
    states, first_intersection = [], None
    for q_value in q_values:
        active = pre.copy()
        active[1] = q_value
        exact = evaluator.evaluate(
            target,
            expand_active_hand_positions(active),
            inputs.frozen_world_from_object,
            previous_state=None if previous_active is None else {
                "world_from_handbase": target,
                "joint_positions_by_name": expand_active_hand_positions(previous_active),
                "world_from_object": inputs.frozen_world_from_object,
            },
        )
        states.append({"q_rad": q_value, "exact": exact})
        intersections = exact["task_surface_intersecting_by_finger"]
        if any(intersections.values()):
            first_intersection = len(states) - 1
            break
        if b0 and exact["fail_closed"]:
            first_intersection = len(states) - 1
            break
        _require(not exact["fail_closed"], "forbidden geometry failed before TASK contact")
        previous_active = active
    _require(first_intersection is not None and first_intersection > 0,
             "no bounded last-free/first-intersecting bracket exists")
    last_free_row, next_row = states[first_intersection - 1], states[first_intersection]
    next_safety = next_row["exact"]
    if b0:
        _require(next_safety["fail_closed"] is True
                 and 0.0 < next_row["q_rad"] - last_free_row["q_rad"]
                 <= STEP_RAD + 1.0e-12,
                 "B0 search did not end at the first raw-mesh nonexecutable state")
    else:
        _require(
            next_safety["task_surface_intersecting_by_finger"] == {
                "finger_1_pad": True, "finger_2_pad": False, "finger_3_pad": False
            }
            and not next_safety["table_top"]["top_intersection_beyond_numerical_tolerance"]
            and not next_safety["self_collision"]["intersecting_pairs"]
            and not next_safety["non_task_hand_object"]["intersecting_links"]
            and 0.0 < next_row["q_rad"] - last_free_row["q_rad"] <= STEP_RAD + 1.0e-12,
            "first intersection is not the isolated finger-1 TASK surface",
        )
    predicted_index = next(
        index for index, item in enumerate(states)
        if abs(item["q_rad"] - predicted[1]) <= 1.0e-12
    )
    semantic_rows = states[predicted_index:first_intersection]
    semantic_valid = [
        row for row in semantic_rows
        if row["exact"]["task_grip_surface_by_finger"][PAD_NAME][
            "task_grip_surface_contact"] is True
    ]
    semantic_invalid = [
        row for row in semantic_rows
        if row["exact"]["task_grip_surface_by_finger"][PAD_NAME][
            "forbidden_first"] is True
    ]
    _require(semantic_valid, "no semantic-valid endpoint exists before TASK intersection")
    _require(not any(
        finger["forbidden_first"]
        for state in states[:predicted_index]
        for finger in state["exact"]["task_grip_surface_by_finger"].values()
    ), "forbidden object surface became first before predicted contact")
    endpoint = semantic_valid[-1]
    first_invalid = None if not semantic_invalid else semantic_invalid[0]
    upper_bound = next_row if first_invalid is None else first_invalid
    if first_invalid is not None:
        upper_bound_kind = "FORBIDDEN_OBJECT_SURFACE_FIRST"
    elif next_safety["non_task_hand_object"]["intersecting_links"]:
        upper_bound_kind = "NON_TASK_HAND_OBJECT_INTERSECTION"
    elif next_safety["table_top"]["top_intersection_beyond_numerical_tolerance"]:
        upper_bound_kind = "FINITE_TABLE_TOP_INTERSECTION"
    elif next_safety["self_collision"]["intersecting_pairs"]:
        upper_bound_kind = "HAND_SELF_INTERSECTION"
    else:
        upper_bound_kind = "RAW_TASK_SURFACE_INTERSECTION"
    endpoint_index = next(
        index for index, state in enumerate(states)
        if abs(state["q_rad"] - endpoint["q_rad"]) <= 1.0e-12
    )
    _require(
        endpoint["q_rad"] < upper_bound["q_rad"]
        and upper_bound["q_rad"] - endpoint["q_rad"] <= STEP_RAD + 1.0e-12
        and (first_invalid is not None or
             abs(endpoint["q_rad"] - last_free_row["q_rad"]) <= 1.0e-12),
        "semantic-valid endpoint and first nonexecutable state are not adjacent",
    )
    _require(not any(
        finger["forbidden_first"]
        for state in states[:endpoint_index + 1]
        for finger in state["exact"]["task_grip_surface_by_finger"].values()
    ), "a forbidden object surface became first before the execution endpoint")
    endpoint_exact = endpoint["exact"]
    witness = endpoint_exact["task_grip_surface_by_finger"][PAD_NAME]
    entry = next((state for state in states[:endpoint_index + 1]
                  if state["exact"]["task_grip_surface_by_finger"][PAD_NAME][
                      "first_contact_from_previous_state"] is True
                  and state["exact"]["task_grip_surface_by_finger"][PAD_NAME][
                      "first_contact_motion_compatible"] is True), None)
    _require(
        endpoint_exact["fail_closed"] is False
        and witness["task_grip_surface_contact"] is True
        and witness["forbidden_first"] is False
        and witness["full_object_intersecting"] is False,
        "selected endpoint lacks an unambiguous finger-1 TASK witness",
    )
    _require(entry is not None, "no motion-compatible contact entry precedes endpoint")
    entry_witness = entry["exact"]["task_grip_surface_by_finger"][PAD_NAME]
    safe_prefix = states[:first_intersection]
    grid_bytes = np.asarray([row["q_rad"] for row in states], dtype="<f8").tobytes()
    surfaces = {
        name: {"link_name": surface.link_name,
               "source_mesh_path": str(surface.source_mesh_path),
               "source_mesh_sha256": surface.source_mesh_sha256}
        for name, surface in sorted(inputs.task_grip_surfaces.items())
    }
    execution_active = pre.copy()
    execution_active[1] = endpoint["q_rad"]
    execution_joints = {
        name: float(value)
        for name, value in expand_active_hand_positions(execution_active).items()
    }
    joint_margins = {
        name: float(min(value - joint.limit.lower, joint.limit.upper - value))
        for name, value in execution_joints.items()
        for joint in (inputs.hand_model.joints[name],)
        if joint.limit is not None
    }
    evidence_binding = {
        "task_ik": _binding(task_path), "config": _binding(config_path),
        "builder_source": _binding(Path(__file__)),
        "evaluator_source": _binding(EVALUATOR), "runner_source": _binding(RUNNER),
    }
    if b0:
        evidence_binding.update({
            name: _binding(path) for name, path in b0_context["paths"].items()})
    return {
        "schema_version": "carts_opposition60_physical_contact_endpoint_v1",
        "source_task_schema_version": task.get("schema_version"),
        "status": "OFFLINE_LAST_SEMANTICALLY_VALID_ENDPOINT_ACCEPTED",
        "claim_scope": "DISCRETE_RAW_MESH_FIRST_FINGER_RESEARCH_ENDPOINT_NOT_PHYSICAL_CONTACT_SUCCESS",
        "object_id": object_id,
        "candidate_id": row["candidate_id"],
        "finger_index": 1,
        "pad_name": PAD_NAME,
        "hardware_authorized": False,
        "formal_dynamic_pass": False,
        "research_dynamic_pass": False,
        "online_truth_used": False,
        "maximum_joint_increment_rad": STEP_RAD,
        "predicted_proximity_target_rad": float(predicted[1]),
        "proxy_q_safe_max_upper_bound_rad": (
            float(b0_context["proxy_upper"]) if b0 else None),
        "execution_target_rad": float(endpoint["q_rad"]),
        "first_semantically_invalid_target_rad": (
            None if first_invalid is None else float(first_invalid["q_rad"])),
        "first_nonexecutable_target_rad": float(upper_bound["q_rad"]),
        "endpoint_upper_bound_kind": upper_bound_kind,
        "last_nonintersecting_target_rad": float(last_free_row["q_rad"]),
        "first_intersecting_target_rad": float(next_row["q_rad"]),
        "execution_extension_from_predicted_rad": float(endpoint["q_rad"] - predicted[1]),
        "selection_rule": "SEMANTIC_VALIDITY_PRECEDES_RAW_FREE_SPACE",
        "endpoint_definition": "LAST_SEMANTIC_VALID_NONINTERSECTING_CONTROL_STEP",
        "checked_state_count_through_first_intersection": len(states),
        "checked_target_grid_sha256": hashlib.sha256(grid_bytes).hexdigest(),
        "execution_task_gap_m": float(witness["allowed_distance_m"]),
        "execution_forbidden_object_gap_m": (
            None if witness["forbidden_distance_m"] is None
            else float(witness["forbidden_distance_m"])),
        "execution_allowed_before_forbidden_margin_m": (
            None if witness["forbidden_distance_m"] is None else float(
                witness["forbidden_distance_m"] - witness["allowed_distance_m"])),
        "execution_active_joint_positions_rad": execution_active.tolist(),
        "execution_all_joint_positions_rad_by_name": execution_joints,
        "execution_joint_limit_margin_rad_by_name": joint_margins,
        "execution_mimic_error_rad_by_joint": endpoint_exact[
            "mimic_error_rad_by_joint"],
        "motion_compatible_entry_witness": {
            "q_rad": float(entry["q_rad"]),
            "first_contact_from_previous_state": entry_witness[
                "first_contact_from_previous_state"],
            "first_contact_motion_compatible": entry_witness[
                "first_contact_motion_compatible"],
            "object_allowed_face_index": entry_witness[
                "object_allowed_face_index"],
            "hand_source_face_index": entry_witness[
                "hand_source_face_index"],
        },
        "path_minimum_table_clearance_m": _minimum(
            safe_prefix, "exact", "table_top", "minimum_clearance_m"),
        "path_minimum_self_clearance_m": _minimum(
            safe_prefix, "exact", "self_collision", "minimum_clearance_m"),
        "path_minimum_non_task_object_clearance_m": _minimum(
            safe_prefix, "exact", "non_task_hand_object", "minimum_clearance_m"),
        "first_semantic_invalid": (None if first_invalid is None else {
            "q_rad": float(first_invalid["q_rad"]),
            "task_grip_surface_by_finger": first_invalid["exact"][
                "task_grip_surface_by_finger"],
        }),
        "first_intersection": {
            "task_surface_intersecting_by_finger": next_safety[
                "task_surface_intersecting_by_finger"],
            "table_intersection": next_safety["table_top"][
                "top_intersection_beyond_numerical_tolerance"],
            "self_intersecting_pairs": next_safety["self_collision"][
                "intersecting_pairs"],
            "non_task_intersecting_links": next_safety["non_task_hand_object"][
                "intersecting_links"],
        },
        "fixed_world_from_handbase_row_major": target.ravel().tolist(),
        "fixed_world_from_object_row_major": inputs.frozen_world_from_object.ravel().tolist(),
        "b0_gate_evidence": (None if not b0 else {
            "surface_audit": b0_context["audit"],
            "sampled_table": b0_context["sampled_table"],
            "nominal_operation_force_cap_n": 12.0,
            "nominal_gravity_lift_balance_pass": True,
            "unique_local_isaac_input_ready_count": 1,
            "proxy_q_safe_max_is_search_upper_bound_only": True,
        }),
        "registered_task_grip_surfaces": surfaces,
        "object_mesh": {
            "path": str(inputs.object_contract.model.provenance.source_path),
            "sha256": inputs.object_contract.model.provenance.source_sha256,
        },
        "evidence_binding": evidence_binding,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--task-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    task_path = args.task_report.resolve()
    _require(task_path.is_file(), f"task report is missing: {task_path}")
    output = args.output.resolve()
    _require(not output.exists(), f"refusing to overwrite evidence: {output}")
    report = plan_endpoint(task_path, args.config.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "q_execute": report["execution_target_rad"],
                      "q_nonexecutable": report["first_nonexecutable_target_rad"],
                      "q_semantic_invalid": report[
                          "first_semantically_invalid_target_rad"],
                      "q_raw_intersect": report["first_intersecting_target_rad"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
