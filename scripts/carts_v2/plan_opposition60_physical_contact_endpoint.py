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
from kcg_connector.grasp.carts_v2.observed_state_replay import (  # noqa: E402
    ObservedHandStateEvaluator,
)
from kcg_connector.robot_model import expand_active_hand_positions  # noqa: E402


CONFIG = ROOT / "src/kcg_connector/config/carts_nailfree_height_projected.yaml"
RUNNER = ROOT / "scripts/carts_v2/run_opposition60_local_contact.py"
EVALUATOR = PACKAGE_ROOT / "kcg_connector/grasp/carts_v2/observed_state_replay.py"
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


def plan_endpoint(task_path: Path) -> dict:
    task = _load(task_path)
    anchor = task["selected_geometric_anchor"]
    row = task["task_and_bounded_ik"][0]
    survivor = task["survivor_candidates"][0]
    object_id = anchor["object_id"]
    inputs = load_v2_inputs(ROOT, config_path=CONFIG, object_id=object_id)
    _require(
        anchor["candidate_id"] == row["candidate_id"] == survivor["candidate_id"]
        and anchor["object_id"] == survivor["object_id"] == object_id
        and task["requested_palm_angle_deg"] == 60
        and task["survivor_count"] == 1
        and len(task["task_and_bounded_ik"]) == 1
        and task["config_sha256"] == _sha256(CONFIG),
        "opposition-60 survivor task identity changed",
    )
    dynamic = inputs.config.section("dynamic")
    _require(
        float(dynamic["physics_dt_s"]) == 1.0 / 120.0
        and float(dynamic["finger_maximum_speed_rad_s"]) * float(
            dynamic["physics_dt_s"]
        ) == STEP_RAD,
        "registered control-step identity changed",
    )
    pre = np.asarray(anchor["pregrasp_joint_positions_rad"], dtype=np.float64)
    predicted = joint_positions_for_phases(
        inputs,
        tuple(row["fresh_contact_stop_phases"]),
        reference_joint_positions_rad=pre,
    )
    maximum_phase = float(inputs.config.section("candidate_generation")[
        "maximum_closure_phase"
    ])
    maximum_phases = list(anchor["pregrasp_closure_phases"])
    maximum_phases[0] = maximum_phase
    maximum = joint_positions_for_phases(
        inputs, tuple(maximum_phases), reference_joint_positions_rad=pre)
    target = np.asarray(
        row["target_world_from_handbase_row_major"], dtype=np.float64
    ).reshape(4, 4)
    evaluator = ObservedHandStateEvaluator(inputs)
    q_values = _bounded_targets(float(pre[1]), float(predicted[1]))
    q_values.extend(
        float(predicted[1] + index * STEP_RAD)
        for index in range(1, 1 + int(np.ceil((maximum[1] - predicted[1]) / STEP_RAD)))
        if predicted[1] + index * STEP_RAD <= maximum[1] + 1.0e-12
    )
    states, first_intersection, previous_active = [], None, None
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
        _require(not exact["fail_closed"], "forbidden geometry failed before TASK contact")
        previous_active = active
    _require(first_intersection is not None and first_intersection > 0,
             "no bounded last-free/first-intersecting bracket exists")
    last_free_row, next_row = states[first_intersection - 1], states[first_intersection]
    next_safety = next_row["exact"]
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
    _require(semantic_valid and semantic_invalid, "semantic endpoint bracket is absent")
    _require(not any(
        finger["forbidden_first"]
        for state in states[:predicted_index]
        for finger in state["exact"]["task_grip_surface_by_finger"].values()
    ), "forbidden object surface became first before predicted contact")
    endpoint = semantic_valid[-1]
    first_invalid = semantic_invalid[0]
    endpoint_index = next(
        index for index, state in enumerate(states)
        if abs(state["q_rad"] - endpoint["q_rad"]) <= 1.0e-12
    )
    _require(
        endpoint["q_rad"] < first_invalid["q_rad"] <= last_free_row["q_rad"]
        and first_invalid["q_rad"] - endpoint["q_rad"] <= STEP_RAD + 1.0e-12,
        "semantic-valid and semantic-invalid endpoints are not adjacent",
    )
    _require(not any(
        finger["forbidden_first"]
        for state in states[:endpoint_index + 1]
        for finger in state["exact"]["task_grip_surface_by_finger"].values()
    ), "a forbidden object surface became first before the execution endpoint")
    endpoint_exact = endpoint["exact"]
    witness = endpoint_exact["task_grip_surface_by_finger"][PAD_NAME]
    predicted_witness = semantic_rows[0]["exact"][
        "task_grip_surface_by_finger"][PAD_NAME]
    _require(
        endpoint_exact["fail_closed"] is False
        and witness["task_grip_surface_contact"] is True
        and witness["forbidden_first"] is False
        and witness["full_object_intersecting"] is False,
        "selected endpoint lacks an unambiguous finger-1 TASK witness",
    )
    _require(
        predicted_witness["first_contact_from_previous_state"] is True
        and predicted_witness["first_contact_motion_compatible"] is True,
        "predicted contact entry is not motion compatible",
    )
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
    return {
        "schema_version": "carts_opposition60_physical_contact_endpoint_v1",
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
        "execution_target_rad": float(endpoint["q_rad"]),
        "first_semantically_invalid_target_rad": float(first_invalid["q_rad"]),
        "last_nonintersecting_target_rad": float(last_free_row["q_rad"]),
        "first_intersecting_target_rad": float(next_row["q_rad"]),
        "execution_extension_from_predicted_rad": float(endpoint["q_rad"] - predicted[1]),
        "selection_rule": "SEMANTIC_VALIDITY_PRECEDES_RAW_FREE_SPACE",
        "endpoint_definition": "LAST_SEMANTIC_VALID_NONINTERSECTING_CONTROL_STEP",
        "checked_state_count_through_first_intersection": len(states),
        "checked_target_grid_sha256": hashlib.sha256(grid_bytes).hexdigest(),
        "execution_task_gap_m": float(witness["allowed_distance_m"]),
        "execution_forbidden_object_gap_m": float(witness["forbidden_distance_m"]),
        "execution_allowed_before_forbidden_margin_m": float(
            witness["forbidden_distance_m"] - witness["allowed_distance_m"]),
        "execution_active_joint_positions_rad": execution_active.tolist(),
        "execution_all_joint_positions_rad_by_name": execution_joints,
        "execution_joint_limit_margin_rad_by_name": joint_margins,
        "execution_mimic_error_rad_by_joint": endpoint_exact[
            "mimic_error_rad_by_joint"],
        "motion_compatible_entry_witness": {
            "q_rad": float(semantic_rows[0]["q_rad"]),
            "first_contact_from_previous_state": predicted_witness[
                "first_contact_from_previous_state"],
            "first_contact_motion_compatible": predicted_witness[
                "first_contact_motion_compatible"],
            "object_allowed_face_index": predicted_witness[
                "object_allowed_face_index"],
            "hand_source_face_index": predicted_witness[
                "hand_source_face_index"],
        },
        "path_minimum_table_clearance_m": _minimum(
            safe_prefix, "exact", "table_top", "minimum_clearance_m"),
        "path_minimum_self_clearance_m": _minimum(
            safe_prefix, "exact", "self_collision", "minimum_clearance_m"),
        "path_minimum_non_task_object_clearance_m": _minimum(
            safe_prefix, "exact", "non_task_hand_object", "minimum_clearance_m"),
        "first_semantic_invalid": {
            "q_rad": float(first_invalid["q_rad"]),
            "task_grip_surface_by_finger": first_invalid["exact"][
                "task_grip_surface_by_finger"],
        },
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
        "registered_task_grip_surfaces": surfaces,
        "object_mesh": {
            "path": str(inputs.object_contract.model.provenance.source_path),
            "sha256": inputs.object_contract.model.provenance.source_sha256,
        },
        "evidence_binding": {
            "task_ik": _binding(task_path), "config": _binding(CONFIG),
            "builder_source": _binding(Path(__file__)),
            "evaluator_source": _binding(EVALUATOR), "runner_source": _binding(RUNNER),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    task_path = args.task_report.resolve()
    _require(task_path.is_file(), f"task report is missing: {task_path}")
    output = args.output.resolve()
    _require(not output.exists(), f"refusing to overwrite evidence: {output}")
    report = plan_endpoint(task_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "q_execute": report["execution_target_rad"],
                      "q_semantic_invalid": report[
                          "first_semantically_invalid_target_rad"],
                      "q_raw_intersect": report["first_intersecting_target_rad"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
