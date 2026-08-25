"""Height-feasible pregrasp coordination before the per-angle Top-8 budget."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.carts_v2.closure_predictor import (
    SequentialClosurePredictor, closure_phase_samples,
)
from kcg_connector.grasp.carts_v2.fast_filter import (
    sampled_pregrasp_path_states, sampled_sequential_closure_states,
)
from kcg_connector.grasp.carts_v2.full_palm_search import (
    PhaseTriple,
    bind_pregrasp,
    fixed_pregrasp_phase_combinations,
)
from kcg_connector.grasp.carts_v2.height_projection import (
    TableHeightRequirement,
    intersect_contact_with_table,
    project_height_to_intervals,
    translate_transform_world_z,
)
from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed,
    FastFilterResult,
    V2Inputs,
    joint_positions_for_phases,
)


PathState = tuple[str, np.ndarray, np.ndarray]
FastFilterCallback = Callable[[object], FastFilterResult]
PregraspPathCallback = Callable[[CandidateSeed], Mapping[str, object]]
PregraspContactKey = Callable[[CandidateSeed], Sequence[float]]


@dataclass(frozen=True)
class SampledPathEnvelope:
    """Existing sampled states covering the required grasp-height path."""

    states: tuple[PathState, ...]
    evidence_scope: str


PathEnvelopeCallback = Callable[[CandidateSeed], SampledPathEnvelope]


def sampled_height_path_states(inputs: V2Inputs, seed: CandidateSeed):
    pregrasp = sampled_pregrasp_path_states(
        inputs, seed, seed.pregrasp_closure_phases)
    maximum = float(seed.maximum_closure_phase or inputs.config.section(
        "candidate_generation")["maximum_closure_phase"])
    closure = sampled_sequential_closure_states(inputs, seed, (maximum,) * 3)
    closure = tuple(row for row in closure
                    if row[0].startswith(("FINGER_", "CONTACT_STOP_")))
    _stage, base, joints = closure[-1]
    lifted = np.array(base, copy=True)
    lifted[2, 3] += min(0.001, float(inputs.config.section(
        "dynamic")["lift_distance_m"]))
    return pregrasp + closure + (("PRELOAD_END", base, joints),
                                 ("LIFT_START", lifted, joints))


def _registered_link_boxes(inputs: V2Inputs):
    boxes = []
    for link, triangles in sorted(inputs.hand_collision_triangles_by_link.items()):
        points = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
        lower, upper = np.min(points, axis=0), np.max(points, axis=0)
        corners = np.asarray([[x, y, z] for x in (lower[0], upper[0])
                              for y in (lower[1], upper[1])
                              for z in (lower[2], upper[2])])
        boxes.append((link, corners))
    if not boxes:
        raise ValueError("registered hand collision triangles are required")
    return tuple(boxes)


def _finite_key(values: Sequence[float], label: str) -> tuple[float, ...]:
    key = tuple(float(value) for value in values)
    if not key or any(not math.isfinite(value) for value in key):
        raise ValueError(f"{label} must be finite and non-empty")
    return key


def _validate_complete_envelope(envelope: SampledPathEnvelope) -> None:
    stages = tuple(str(row[0]) for row in envelope.states)
    required = ("PALM_FAR_", "PRESHAPE_FAR_", "APPROACH_", "PREGRASP",
                "FINGER_1_", "FINGER_2_", "FINGER_3_", "PRELOAD_END", "LIFT_START")
    if (not envelope.evidence_scope
            or any(not any(stage.startswith(prefix) for stage in stages)
                   for prefix in required)):
        raise ValueError("height envelope does not cover the complete registered path")


def _table_requirement(
    inputs: V2Inputs,
    seed: CandidateSeed,
    envelope: SampledPathEnvelope,
    link_boxes,
    required_clearance_m: float,
    *,
    require_complete_path: bool = True,
    reject_above_handbase_z_m: float | None = None,
) -> tuple[TableHeightRequirement, str | None, int, bool]:
    """Reduce existing FK path states without materializing one giant mesh array."""

    if require_complete_path:
        _validate_complete_envelope(envelope)
    elif not envelope.states or not envelope.evidence_scope:
        raise ValueError("height preselection snapshot is empty or unidentified")
    final_base = inputs.frozen_world_from_object @ seed.object_from_hand_matrix()
    final_z = float(final_base[2, 3])
    best: TableHeightRequirement | None = None
    best_stage = None
    overlap_count = 0
    checked_state_count, early_stop = 0, False
    bounds = inputs.table_xy_bounds_m
    for state_index, (stage, _base, joints) in enumerate(envelope.states):
        checked_state_count += 1
        transforms = inputs.hand_model.forward_kinematics(joints, base_transform=_base)
        for link_index, (link, corners) in enumerate(link_boxes):
            transform = np.asarray(transforms[link], dtype=np.float64)
            world = corners @ transform[:3, :3].T + transform[:3, 3]
            lower, upper = np.min(world, axis=0), np.max(world, axis=0)
            overlap = (upper[0] >= bounds[0, 0] and lower[0] <= bounds[0, 1]
                       and upper[1] >= bounds[1, 0] and lower[1] <= bounds[1, 1])
            if not overlap:
                continue
            overlap_count += 1
            vertex = int(np.argmin(world[:, 2]))
            relative_z = float(world[vertex, 2] - final_z)
            minimum = inputs.table_top_z_m + required_clearance_m - relative_z
            if best is None or minimum > float(best.minimum_handbase_z_m):
                best = TableHeightRequirement(
                    minimum, relative_z, state_index, link_index, vertex, 1,
                    "REGISTERED_LINK_LOCAL_AABB_CONSERVATIVE",
                    "SAMPLED_CONTROL_PATH_LINK_BOX_SUPPORT_NOT_CONTINUOUS")
                best_stage = str(stage)
                if (reject_above_handbase_z_m is not None
                        and minimum > reject_above_handbase_z_m):
                    early_stop = True
                    break
        if early_stop:
            break
    if best is None:
        best = TableHeightRequirement(None, None, None, None, None, 0,
            "REGISTERED_LINK_LOCAL_AABB_CONSERVATIVE",
            "SAMPLED_CONTROL_PATH_LINK_BOX_SUPPORT_NOT_CONTINUOUS")
    return (replace(best, overlapping_primitive_count=overlap_count), best_stage,
            checked_state_count, early_stop)


def _seed_at_world_height(
    inputs: V2Inputs, seed: CandidateSeed, world_height_m: float
) -> CandidateSeed:
    world_from_object = np.asarray(inputs.frozen_world_from_object, dtype=np.float64)
    world_from_hand = world_from_object @ seed.object_from_hand_matrix()
    shifted = translate_transform_world_z(
        world_from_hand, float(world_height_m) - float(world_from_hand[2, 3]))
    object_from_hand = np.linalg.inv(world_from_object) @ shifted
    return replace(seed, object_from_hand=tuple(float(value) for value in
                                                object_from_hand.ravel()))


def _three_registered_contacts(inputs: V2Inputs, prediction: object) -> bool:
    if getattr(prediction, "status", "") != "CLOSURE_SURVIVE":
        return False
    contacts = tuple(getattr(prediction, "contacts", ()))
    expected = {pad.name for pad in inputs.hand_contract.pads}
    return len(contacts) == 3 and {contact.pad_name for contact in contacts} == expected


def _swept_contact_reach_interval(
    inputs: V2Inputs, seed: CandidateSeed
) -> tuple[tuple[float, float] | None, dict[str, object]]:
    surfaces = inputs.task_grip_surfaces
    if surfaces is None:
        raise ValueError("TASK_GRIP_SURFACE is required for height reach")
    object_mesh = inputs.object_contract.model.mesh
    allowed = np.asarray(inputs.face_roles.face_is_allowed, dtype=np.bool_)
    object_points = object_mesh.face_vertices_m[allowed].reshape(-1, 3)
    world_object = (object_points @ inputs.frozen_world_from_object[:3, :3].T
                    + inputs.frozen_world_from_object[:3, 3])
    object_lower, object_upper = np.min(world_object, axis=0), np.max(world_object, axis=0)
    base = inputs.frozen_world_from_object @ seed.object_from_hand_matrix()
    original_height = float(base[2, 3])
    reference = seed.pregrasp_joint_positions_rad
    phases = tuple(float(value) for value in seed.pregrasp_closure_phases)
    maximum = float(seed.maximum_closure_phase or inputs.config.section(
        "candidate_generation")["maximum_closure_phase"])
    clearance = float(inputs.config.section("closure_prediction")["contact_distance_m"])
    phase_by_name = {pad.name: index for index, pad in enumerate(inputs.hand_contract.pads)}
    intervals, state_count = [], 0
    for pad_name in inputs.config.section("closure_prediction")["closing_order"]:
        phase_index = phase_by_name[str(pad_name)]
        values = (phases[phase_index], *closure_phase_samples(
            inputs, phases, phase_index, maximum, reference))
        surface = surfaces[str(pad_name)]
        local_lower = np.min(surface.points_local_m, axis=0)
        local_upper = np.max(surface.points_local_m, axis=0)
        corners = np.asarray([[x, y, z] for x in (local_lower[0], local_upper[0])
                              for y in (local_lower[1], local_upper[1])
                              for z in (local_lower[2], local_upper[2])])
        swept_lower = np.full(3, np.inf)
        swept_upper = np.full(3, -np.inf)
        for value in values:
            sample = list(phases)
            sample[phase_index] = float(value)
            joints = joint_positions_for_phases(
                inputs, tuple(sample), reference_joint_positions_rad=reference)
            transform = inputs.hand_model.forward_kinematics(
                joints, base_transform=base)[surface.link_name]
            world = corners @ transform[:3, :3].T + transform[:3, 3]
            swept_lower = np.minimum(swept_lower, np.min(world, axis=0))
            swept_upper = np.maximum(swept_upper, np.max(world, axis=0))
        state_count += len(values)
        xy_overlap = (swept_upper[0] + clearance >= object_lower[0]
                      and swept_lower[0] - clearance <= object_upper[0]
                      and swept_upper[1] + clearance >= object_lower[1]
                      and swept_lower[1] - clearance <= object_upper[1])
        if not xy_overlap:
            return None, {"sampled_finger_state_count": state_count,
                          "reason": "NO_XY_TASK_SURFACE_REACH"}
        intervals.append((
            original_height + object_lower[2] - clearance - swept_upper[2],
            original_height + object_upper[2] + clearance - swept_lower[2],
        ))
    lower = max(value[0] for value in intervals)
    upper = min(value[1] for value in intervals)
    interval = None if lower > upper else (float(lower), float(upper))
    return interval, {
        "sampled_finger_state_count": state_count,
        "per_finger_world_height_intervals_m": [list(value) for value in intervals],
        "evidence_scope": "CONTROL_STEP_TASK_SURFACE_AABB_REACH_NOT_CONTACT_PROOF",
    }


def _evaluate_variant(
    inputs: V2Inputs,
    seed: CandidateSeed,
    predictor: SequentialClosurePredictor,
    pregrasp_path_callback: PregraspPathCallback,
    fast_filter_callback: FastFilterCallback,
    requirement: TableHeightRequirement,
    stage: str | None,
    bounded_reach: tuple[tuple[float, float], ...],
    reach_evidence: Mapping[str, object],
    required_table_clearance_m: float,
    table_numerical_tolerance_m: float,
) -> tuple[CandidateSeed | None, dict[str, object]]:
    feasible = intersect_contact_with_table(
        bounded_reach, requirement.minimum_handbase_z_m)
    row: dict[str, object] = {
        "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
        "minimum_table_handbase_z_m": requirement.minimum_handbase_z_m,
        "minimum_table_contributing_stage": stage,
        "contact_reach_outer_intervals_m": [list(value) for value in bounded_reach],
        "conservative_feasible_height_intervals_m": [list(value) for value in feasible],
        "height_evidence_scope": (
            "ANALYTIC_REACH_OUTER_BOUND_PLUS_EXACT_PROJECTED_REVALIDATION"),
        "contact_reach_evidence": reach_evidence,
    }
    if not feasible:
        row.update(
            status="HARD_REJECT",
            reason="EMPTY_TABLE_AND_CONTACT_HEIGHT_INTERSECTION_CONSERVATIVE_GATE")
        return None, row
    original = float((inputs.frozen_world_from_object
                      @ seed.object_from_hand_matrix())[2, 3])
    projection = project_height_to_intervals(original, feasible)
    projected = _seed_at_world_height(inputs, seed, projection.projected_height_m)
    pregrasp_result = dict(pregrasp_path_callback(projected))
    if (pregrasp_result.get("candidate_id") != seed.candidate_id
            or tuple(pregrasp_result.get("pregrasp_closure_phases", ())) !=
            projected.pregrasp_closure_phases):
        raise ValueError("projected pregrasp-path identity changed")
    pregrasp_clearance = pregrasp_result.get("minimum_table_clearance_m")
    pregrasp_clearance_pass = (pregrasp_clearance is None or
        float(pregrasp_clearance) >= required_table_clearance_m -
        table_numerical_tolerance_m)
    if pregrasp_result.get("accepted") is not True or not pregrasp_clearance_pass:
        details = tuple(str(value) for value in pregrasp_result.get("reasons", ()))
        row.update({
            "status": "POST_PROJECTION_REVALIDATION_REJECT",
            "reason": "PROJECTED_HEIGHT_REVALIDATION_FAILED",
            "revalidation_detail": (";".join(details) if details else
                                     "PREGRASP_TABLE_OPERATION_CLEARANCE"),
            "projected_pregrasp_path": pregrasp_result,
        })
        return None, row
    fresh_prediction = predictor.predict(projected)
    fast_result = fast_filter_callback(fresh_prediction)
    if fast_result.candidate_id != seed.candidate_id:
        raise ValueError("post-projection fast-filter candidate identity changed")
    closure_pass = _three_registered_contacts(inputs, fresh_prediction)
    clearance = fast_result.minimum_table_clearance_m
    clearance_pass = (clearance is None or clearance >=
                      required_table_clearance_m - table_numerical_tolerance_m)
    fast_pass = bool(fast_result.status == "FAST_SURVIVE"
                     and fast_result.sequential_closure_sweep_pass
                     and fast_result.checked_state_count > 0 and clearance_pass)
    remaining = np.asarray(fresh_prediction.final_closure_phases) - np.asarray(
        projected.pregrasp_closure_phases)
    balance = float(np.max(remaining) - np.min(remaining))
    table_key = [0.0, 0.0] if clearance is None else [1.0, -float(clearance)]
    detail = (getattr(fresh_prediction, "reason", "")
              or ";".join(fast_result.reasons) or "FRESH_REVALIDATION_REJECT")
    row.update({
        "status": "OFFLINE_SAMPLED_HAND_HEIGHT_FEASIBLE_AT_PROJECTED_Z"
                  if closure_pass and fast_pass
                  else "POST_PROJECTION_REVALIDATION_REJECT",
        "reason": "" if closure_pass and fast_pass
                  else "PROJECTED_HEIGHT_REVALIDATION_FAILED",
        "revalidation_detail": "" if closure_pass and fast_pass else detail,
        "original_handbase_world_z_m": original,
        "projected_handbase_world_z_m": projection.projected_height_m,
        "translation_world_z_m": projection.translation_world_z_m,
        "selected_height_interval_m": list(projection.selected_interval_m),
        "fresh_closure_status": getattr(fresh_prediction, "status", ""),
        "projected_pregrasp_path": pregrasp_result,
        "fresh_fast_filter_status": fast_result.status,
        "fresh_sequential_closure_sweep_pass": fast_result.sequential_closure_sweep_pass,
        "fresh_checked_state_count": fast_result.checked_state_count,
        "fresh_minimum_table_clearance_m": clearance,
        "fresh_final_closure_phases": list(fresh_prediction.final_closure_phases),
        "fresh_object_contact_face_indices": [contact.object_face_index
                                               for contact in fresh_prediction.contacts],
        "fresh_hand_surface_face_indices": [contact.hand_surface_face_index
                                             for contact in fresh_prediction.contacts],
        "fresh_hand_surface_legacy_blue_pad": [contact.hand_surface_legacy_blue_pad
                                                for contact in fresh_prediction.contacts],
        "remaining_closure_imbalance": balance,
        "selection_key": [0.0, *table_key, 0.0, balance,
                          abs(projection.translation_world_z_m)],
        "minimum_nonallowed_surface_clearance_m": None,
        "nonallowed_surface_gate": "COLLISION_FREE_BINARY_DISTANCE_NOT_YET_REPORTED",
    })
    return (projected if closure_pass and fast_pass else None), row


def search_height_projected_pregrasps(
    inputs: V2Inputs,
    seed: CandidateSeed,
    predictor: SequentialClosurePredictor,
    *,
    sampled_path_envelope: PathEnvelopeCallback,
    pregrasp_contact_key: PregraspContactKey,
    pregrasp_path_callback: PregraspPathCallback,
    fast_filter_callback: FastFilterCallback,
    contact_height_bounds_m: tuple[float, float],
    table_numerical_tolerance_m: float,
    required_table_clearance_m: float = 0.0,
    maximum_exact_variants: int = 2,
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Search 27 preshapes, precisely evaluating no more than two of them.

    ``sampled_path_envelope`` must wrap the existing bounded pregrasp and
    sequential path-state generators. Contact predictions used during the
    interval search are never reused for the mandatory projected revalidation.
    """

    if predictor.inputs is not inputs:
        raise ValueError("closure predictor is bound to different V2 inputs")
    if not 1 <= int(maximum_exact_variants) <= 2:
        raise ValueError("maximum exact pregrasp variants must lie in [1, 2]")
    prepared = []
    link_boxes = _registered_link_boxes(inputs)
    for phases in fixed_pregrasp_phase_combinations():
        bound = bind_pregrasp(inputs, seed, phases)
        contact_key = _finite_key(pregrasp_contact_key(bound), "pregrasp contact key")
        base = inputs.frozen_world_from_object @ bound.object_from_hand_matrix()
        snapshot = SampledPathEnvelope(
            (("PREGRASP", base, np.asarray(bound.pregrasp_joint_positions_rad)),),
            "SINGLE_STATE_PRESELECTION_ONLY",
        )
        requirement, _stage, _checked, _early = _table_requirement(
            inputs, bound, snapshot, link_boxes, required_table_clearance_m,
            require_complete_path=False)
        table_key = ((0.0, 0.0) if requirement.minimum_handbase_z_m is None
                     else (1.0, requirement.minimum_handbase_z_m))
        prepared.append((contact_key, table_key, phases, bound))
    contact_choice = min(prepared, key=lambda row: (row[0], row[2]))
    table_choice = min(prepared, key=lambda row: (row[1], row[0], row[2]))
    selected = [contact_choice]
    if table_choice[2] != contact_choice[2] and maximum_exact_variants > 1:
        selected.append(table_choice)
    selected_phases = {row[2] for row in selected}
    deferred = [{"pregrasp_closure_phases": list(row[2]),
                 "status": "BUDGET_NOT_EVALUATED"} for row in prepared
                if row[2] not in selected_phases]
    survivors, evaluated = [], []
    for _contact_key, _table_key, _phases, bound in selected:
        reach, reach_evidence = _swept_contact_reach_interval(inputs, bound)
        bounded_reach = ()
        if reach is not None:
            lower, upper = max(reach[0], contact_height_bounds_m[0]), min(
                reach[1], contact_height_bounds_m[1])
            if lower <= upper:
                bounded_reach = ((float(lower), float(upper)),)
        envelope = sampled_path_envelope(bound) if bounded_reach else None
        if envelope is None:
            requirement = TableHeightRequirement(None, None, None, None, None, 0,
                "NOT_EVALUATED_NO_CONTACT_REACH", "NO_CONTACT_REACH_OUTER_INTERVAL")
            stage, checked, early, scope = None, 0, False, requirement.evidence_scope
        else:
            requirement, stage, checked, early = _table_requirement(
                inputs, bound, envelope, link_boxes, required_table_clearance_m,
                reject_above_handbase_z_m=bounded_reach[0][1])
            scope = envelope.evidence_scope
        survivor, row = _evaluate_variant(
            inputs, bound, predictor, pregrasp_path_callback,
            fast_filter_callback, requirement, stage,
            bounded_reach, reach_evidence, required_table_clearance_m,
            table_numerical_tolerance_m)
        row.update(path_evidence_scope=scope, table_path_checked_state_count=checked,
                   table_path_scan_complete=bool(envelope is not None and not early),
                   table_path_early_empty_intersection=early)
        if early:
            row["height_evidence_scope"] = "OUTER_BOUND_MONOTONIC_TABLE_EARLY_REJECT"
        evaluated.append(row)
        if survivor is not None:
            survivors.append(survivor)
    audit = {
        "schema_version": "carts_height_projected_search_v1",
        "claim_scope": "OFFLINE_SAMPLED_HEIGHT_SEARCH_NOT_DYNAMIC_SUCCESS",
        "candidate_id": seed.candidate_id,
        "pregrasp_variant_count": len(prepared),
        "exact_variant_budget": int(maximum_exact_variants),
        "exact_variant_evaluated_count": len(evaluated),
        "survivor_count": len(survivors),
        "evaluated": evaluated,
        "deferred": deferred,
    }
    return tuple(survivors), audit


__all__ = ["SampledPathEnvelope", "sampled_height_path_states",
           "search_height_projected_pregrasps"]
