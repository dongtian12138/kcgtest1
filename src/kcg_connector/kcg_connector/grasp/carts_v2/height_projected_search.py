"""Height-feasible pregrasp coordination before the per-angle Top-8 budget."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
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
    minimum_z_over_finite_table_top,
    translate_transform_world_z,
)
from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed,
    FastFilterResult,
    V2Inputs,
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


PathEnvelopeCallback = Callable[
    [CandidateSeed, tuple[float, float, float]], SampledPathEnvelope
]


def sampled_height_path_states(
    inputs: V2Inputs, seed: CandidateSeed,
    final_closure_phases: tuple[float, float, float],
):
    pregrasp = sampled_pregrasp_path_states(
        inputs, seed, seed.pregrasp_closure_phases)
    closure = sampled_sequential_closure_states(
        inputs, seed, tuple(float(value) for value in final_closure_phases))
    closure = tuple(row for row in closure
                    if row[0].startswith(("FINGER_", "CONTACT_STOP_")))
    _stage, base, joints = closure[-1]
    lifted = np.array(base, copy=True)
    lifted[2, 3] += min(0.001, float(inputs.config.section(
        "dynamic")["lift_distance_m"]))
    return pregrasp + closure + (("PRELOAD_END", base, joints),
                                 ("LIFT_START", lifted, joints))


def sampled_table_path_requirement(
    inputs: V2Inputs,
    seed: CandidateSeed,
    final_closure_phases: tuple[float, float, float],
    required_clearance_m: float,
) -> tuple[TableHeightRequirement, str | None, int]:
    """Evaluate the registered complete sampled path against the finite table."""

    envelope = SampledPathEnvelope(
        tuple(sampled_height_path_states(inputs, seed, final_closure_phases)),
        "REGISTERED_CONTROL_STEPS_PALM_PRESHAPE_APPROACH_"
        "SEQUENTIAL_CLOSURE_PRELOAD_LIFT_START",
    )
    requirement, stage, checked, _early = _table_requirement(
        inputs,
        seed,
        envelope,
        _registered_link_geometry(inputs),
        float(required_clearance_m),
    )
    return requirement, stage, checked


def _registered_link_geometry(inputs: V2Inputs):
    geometry = []
    for link, triangles in sorted(inputs.hand_collision_triangles_by_link.items()):
        values = np.asarray(triangles, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (3, 3):
            raise ValueError(f"registered collision triangles malformed for {link}")
        geometry.append((link, values))
    if not geometry:
        raise ValueError("registered hand collision triangles are required")
    return tuple(geometry)


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
    link_geometry,
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
        for link_index, (link, triangles) in enumerate(link_geometry):
            transform = np.asarray(transforms[link], dtype=np.float64)
            world = triangles @ transform[:3, :3].T + transform[:3, 3]
            minimum_z, triangle_index = minimum_z_over_finite_table_top(
                world, bounds)
            if minimum_z is None:
                continue
            overlap_count += 1
            relative_z = float(minimum_z - final_z)
            minimum = inputs.table_top_z_m + required_clearance_m - relative_z
            if best is None or minimum > float(best.minimum_handbase_z_m):
                best = TableHeightRequirement(
                    minimum, relative_z, state_index, link_index,
                    triangle_index, 1,
                    "REGISTERED_EXACT_TRIANGLE_FINITE_TABLE_TOP",
                    "SAMPLED_CONTROL_PATH_EXACT_MESH_NOT_CONTINUOUS")
                best_stage = str(stage)
                if (reject_above_handbase_z_m is not None
                        and minimum > reject_above_handbase_z_m):
                    early_stop = True
                    break
        if early_stop:
            break
    if best is None:
        best = TableHeightRequirement(None, None, None, None, None, 0,
            "REGISTERED_EXACT_TRIANGLE_FINITE_TABLE_TOP",
            "SAMPLED_CONTROL_PATH_EXACT_MESH_NOT_CONTINUOUS")
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


def contact_height_bounds(inputs: V2Inputs, seed: CandidateSeed) -> tuple[float, float]:
    """Bound a vertical contact search by object height and registered hand reach."""

    transforms = inputs.hand_model.forward_kinematics(
        seed.pregrasp_joint_positions_rad)
    reach = 0.0
    for link_name, triangles in inputs.hand_collision_triangles_by_link.items():
        transform = transforms[link_name]
        points = (np.asarray(triangles).reshape(-1, 3) @ transform[:3, :3].T
                  + transform[:3, 3])
        reach = max(reach, float(np.max(np.linalg.norm(points, axis=1))))
    vertices = np.asarray(inputs.object_contract.model.mesh.vertices_m)
    world = (vertices @ inputs.frozen_world_from_object[:3, :3].T
             + inputs.frozen_world_from_object[:3, 3])
    if not math.isfinite(reach) or reach <= 0.0:
        raise ValueError("registered hand reach radius is invalid")
    return float(np.min(world[:, 2]) - reach), float(np.max(world[:, 2]) + reach)


def _three_registered_contacts(inputs: V2Inputs, prediction: object) -> bool:
    if getattr(prediction, "status", "") != "CLOSURE_SURVIVE":
        return False
    contacts = tuple(getattr(prediction, "contacts", ()))
    expected = {pad.name for pad in inputs.hand_contract.pads}
    return len(contacts) == 3 and {contact.pad_name for contact in contacts} == expected


def _height_probe_candidates(inputs, seed, bounds):
    lower, upper = (float(value) for value in bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("contact-height bounds must be finite and increasing")
    count = int(inputs.config.section("height_projection")[
        "contact_search_coarse_sample_count"])
    original = float((inputs.frozen_world_from_object
                      @ seed.object_from_hand_matrix())[2, 3])
    values = np.unique(np.append(np.linspace(lower, upper, count),
                                 np.clip(original, lower, upper)))
    return tuple(_seed_at_world_height(inputs, seed, value) for value in values)


def _prediction_signature(prediction) -> tuple[str, str, int]:
    return (str(prediction.status), str(prediction.reason), len(prediction.contacts))


def _probe_contact_height(inputs, seed, predictor, height, kind):
    candidate = _seed_at_world_height(inputs, seed, float(height))
    prediction = predictor.predict(candidate)
    row = {"handbase_world_z_m": float(height),
           "closure_status": prediction.status, "reason": prediction.reason,
           "contact_count": len(prediction.contacts), "probe_kind": kind}
    return candidate, prediction, row


def _first_contact_prediction(inputs, seed, predictor, bounds):
    probes, sampled = [], []
    for candidate in _height_probe_candidates(inputs, seed, bounds):
        prediction = predictor.predict(candidate)
        height = float((inputs.frozen_world_from_object
                        @ candidate.object_from_hand_matrix())[2, 3])
        probes.append({"handbase_world_z_m": height,
                       "closure_status": prediction.status,
                       "reason": prediction.reason,
                       "contact_count": len(prediction.contacts),
                       "probe_kind": "COARSE"})
        sampled.append((height, candidate, prediction))
        if _three_registered_contacts(inputs, prediction):
            return candidate, prediction, probes
    settings = inputs.config.section("height_projection")
    tolerance = float(settings["contact_boundary_tolerance_m"])
    maximum = int(settings["maximum_bisection_iterations"])
    if tolerance <= 0.0 or maximum < 1:
        raise ValueError("contact-height boundary search bounds are invalid")
    for left, right in zip(sampled[:-1], sampled[1:]):
        if _prediction_signature(left[2]) == _prediction_signature(right[2]):
            continue
        queue, iterations = [(left, right)], 0
        while queue and iterations < maximum:
            lower, upper = queue.pop(0)
            if upper[0] - lower[0] <= tolerance:
                continue
            middle_height = 0.5 * (lower[0] + upper[0])
            candidate, prediction, row = _probe_contact_height(
                inputs, seed, predictor, middle_height, "STATE_BOUNDARY_BISECTION")
            probes.append(row)
            middle = (middle_height, candidate, prediction)
            iterations += 1
            if _three_registered_contacts(inputs, prediction):
                return candidate, prediction, probes
            if _prediction_signature(lower[2]) != _prediction_signature(prediction):
                queue.append((lower, middle))
            if _prediction_signature(prediction) != _prediction_signature(upper[2]):
                queue.append((middle, upper))
    return None, None, probes


def _final_validation(
    inputs: V2Inputs,
    seed: CandidateSeed,
    predictor: SequentialClosurePredictor,
    pregrasp_path_callback: PregraspPathCallback,
    fast_filter_callback: FastFilterCallback,
    original_height_m: float,
    history: list[dict[str, object]],
    required_table_clearance_m: float,
    table_numerical_tolerance_m: float,
) -> tuple[CandidateSeed | None, dict[str, object]]:
    pregrasp_result = dict(pregrasp_path_callback(seed))
    if (pregrasp_result.get("candidate_id") != seed.candidate_id
            or tuple(pregrasp_result.get("pregrasp_closure_phases", ())) !=
            seed.pregrasp_closure_phases):
        raise ValueError("projected pregrasp-path identity changed")
    pregrasp_clearance = pregrasp_result.get("minimum_table_clearance_m")
    pregrasp_clearance_pass = (pregrasp_clearance is None or
        float(pregrasp_clearance) >= required_table_clearance_m -
        table_numerical_tolerance_m)
    if pregrasp_result.get("accepted") is not True or not pregrasp_clearance_pass:
        details = tuple(str(value) for value in pregrasp_result.get("reasons", ()))
        return None, {
            "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
            "status": "POST_PROJECTION_REVALIDATION_REJECT",
            "reason": "PROJECTED_HEIGHT_REVALIDATION_FAILED",
            "revalidation_detail": (";".join(details) if details else
                                     "PREGRASP_TABLE_OPERATION_CLEARANCE"),
            "projected_pregrasp_path": pregrasp_result,
            "contact_conditioned_iterations": history,
        }
    fresh_prediction = predictor.predict(seed)
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
        seed.pregrasp_closure_phases)
    balance = float(np.max(remaining) - np.min(remaining))
    table_key = [0.0, 0.0] if clearance is None else [1.0, -float(clearance)]
    detail = (getattr(fresh_prediction, "reason", "")
              or ";".join(fast_result.reasons) or "FRESH_REVALIDATION_REJECT")
    final_height = float((inputs.frozen_world_from_object
                          @ seed.object_from_hand_matrix())[2, 3])
    row = {
        "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
        "status": "OFFLINE_SAMPLED_HAND_HEIGHT_FEASIBLE_AT_PROJECTED_Z"
                  if closure_pass and fast_pass
                  else "POST_PROJECTION_REVALIDATION_REJECT",
        "reason": "" if closure_pass and fast_pass
                  else "PROJECTED_HEIGHT_REVALIDATION_FAILED",
        "revalidation_detail": "" if closure_pass and fast_pass else detail,
        "original_handbase_world_z_m": original_height_m,
        "projected_handbase_world_z_m": final_height,
        "translation_world_z_m": final_height - original_height_m,
        "contact_conditioned_iterations": history,
        "height_evidence_scope": "CONTACT_STOP_CONDITIONED_EXACT_MESH_ITERATION",
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
                          abs(final_height - original_height_m)],
        "minimum_nonallowed_surface_clearance_m": None,
        "nonallowed_surface_gate": "COLLISION_FREE_BINARY_DISTANCE_NOT_YET_REPORTED",
    }
    return (seed if closure_pass and fast_pass else None), row


def _evaluate_variant(
    inputs, seed, predictor, sampled_path_envelope, pregrasp_path_callback,
    fast_filter_callback, link_geometry, contact_height_bounds_m,
    required_table_clearance_m, table_numerical_tolerance_m,
    maximum_iterations=5,
):
    original = float((inputs.frozen_world_from_object
                      @ seed.object_from_hand_matrix())[2, 3])
    current, prediction, probes = _first_contact_prediction(
        inputs, seed, predictor, contact_height_bounds_m)
    if current is None or prediction is None:
        return None, {
            "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
            "status": "UNRESOLVED",
            "reason": "SAMPLED_NO_THREE_CONTACT_WITNESS_UNRESOLVED",
            "contact_height_probes": probes,
            "height_evidence_scope": "COARSE_AND_STATE_BOUNDARY_SAMPLES_NOT_PROOF_OF_EMPTY_SET",
        }
    history: list[dict[str, object]] = []
    for iteration in range(int(maximum_iterations)):
        final_phases = tuple(float(value) for value in prediction.final_closure_phases)
        envelope = sampled_path_envelope(current, final_phases)
        requirement, stage, checked, _early = _table_requirement(
            inputs, current, envelope, link_geometry, required_table_clearance_m)
        height = float((inputs.frozen_world_from_object
                        @ current.object_from_hand_matrix())[2, 3])
        projected = max(height, float(requirement.minimum_handbase_z_m
                                      if requirement.minimum_handbase_z_m is not None
                                      else -math.inf))
        history.append({
            "iteration": iteration, "handbase_world_z_m": height,
            "contact_stop_phases": list(final_phases),
            "minimum_table_handbase_z_m": requirement.minimum_handbase_z_m,
            "minimum_table_contributing_stage": stage,
            "table_path_checked_state_count": checked,
            "projected_handbase_world_z_m": projected,
        })
        if projected > float(contact_height_bounds_m[1]) + table_numerical_tolerance_m:
            return None, {
                "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
                "status": "HARD_REJECT",
                "reason": "EMPTY_TABLE_AND_CONTACT_HEIGHT_INTERSECTION",
                "contact_height_probes": probes,
                "contact_conditioned_iterations": history,
            }
        if abs(projected - height) <= table_numerical_tolerance_m:
            survivor, row = _final_validation(
                inputs, current, predictor, pregrasp_path_callback,
                fast_filter_callback, original, history,
                required_table_clearance_m, table_numerical_tolerance_m)
            row["contact_height_probes"] = probes
            return survivor, row
        current = _seed_at_world_height(inputs, current, projected)
        prediction = predictor.predict(current)
        if not _three_registered_contacts(inputs, prediction):
            return None, {
                "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
                "status": "POST_PROJECTION_REVALIDATION_REJECT",
                "reason": "CONTACT_LOST_AFTER_TABLE_PROJECTION",
                "contact_height_probes": probes,
                "contact_conditioned_iterations": history,
                "fresh_closure_reason": prediction.reason,
            }
    return None, {
        "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
        "status": "UNRESOLVED", "reason": "CONTACT_HEIGHT_ITERATION_LIMIT",
        "contact_height_probes": probes, "contact_conditioned_iterations": history,
    }


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
    exact_variant_offset: int = 0,
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Evaluate a bounded subset or all 27 contact-conditioned preshapes.

    ``sampled_path_envelope`` must wrap the existing bounded pregrasp and
    sequential path-state generators. Contact predictions used during the
    height search are never reused for the mandatory projected revalidation.
    """

    if predictor.inputs is not inputs:
        raise ValueError("closure predictor is bound to different V2 inputs")
    offset, count = int(exact_variant_offset), int(maximum_exact_variants)
    if not 0 <= offset < 27:
        raise ValueError("exact pregrasp variant offset must lie in [0, 26]")
    if not 1 <= count <= 27 - offset:
        raise ValueError("maximum exact pregrasp variants exceed the remaining budget")
    prepared = []
    link_geometry = _registered_link_geometry(inputs)
    for phases in fixed_pregrasp_phase_combinations():
        bound = bind_pregrasp(inputs, seed, phases)
        contact_key = _finite_key(pregrasp_contact_key(bound), "pregrasp contact key")
        base = inputs.frozen_world_from_object @ bound.object_from_hand_matrix()
        snapshot = SampledPathEnvelope(
            (("PREGRASP", base, np.asarray(bound.pregrasp_joint_positions_rad)),),
            "SINGLE_STATE_PRESELECTION_ONLY",
        )
        requirement, _stage, _checked, _early = _table_requirement(
            inputs, bound, snapshot, link_geometry, required_table_clearance_m,
            require_complete_path=False)
        table_key = ((0.0, 0.0) if requirement.minimum_handbase_z_m is None
                     else (1.0, requirement.minimum_handbase_z_m))
        prepared.append((contact_key, table_key, phases, bound))
    ordered = []
    priorities = [
        min(prepared, key=lambda row: (row[0], row[2])),
        min(prepared, key=lambda row: (row[1], row[0], row[2])),
        min(prepared, key=lambda row: (
            max(row[2]) - min(row[2]), row[0], row[2])),
    ]
    priorities.extend(min(prepared, key=lambda row, index=index: (
        -row[2][index], sum(row[2][other] for other in range(3) if other != index),
        row[0], row[2])) for index in range(3))
    priorities.extend(sorted(prepared, key=lambda row: (row[0], row[1], row[2])))
    for choice in priorities:
        if choice[2] not in {row[2] for row in ordered}:
            ordered.append(choice)
        if len(ordered) == len(prepared):
            break
    if len(ordered) != 27:
        raise RuntimeError("deterministic pregrasp priority order is incomplete")
    selected = ordered[offset:offset + count]
    selected_phases = {row[2] for row in selected}
    deferred = [{"pregrasp_closure_phases": list(row[2]),
                 "status": "BUDGET_NOT_EVALUATED"} for row in prepared
                if row[2] not in selected_phases]
    survivors, evaluated = [], []
    for _contact_key, _table_key, _phases, bound in selected:
        survivor, row = _evaluate_variant(
            inputs, bound, predictor, sampled_path_envelope,
            pregrasp_path_callback, fast_filter_callback, link_geometry,
            contact_height_bounds_m, required_table_clearance_m,
            table_numerical_tolerance_m)
        evaluated.append(row)
        if survivor is not None:
            survivors.append(survivor)
    audit = {
        "schema_version": "carts_contact_conditioned_height_search_v2",
        "claim_scope": "OFFLINE_CONTACT_CONDITIONED_HEIGHT_NOT_DYNAMIC_SUCCESS",
        "candidate_id": seed.candidate_id,
        "pregrasp_variant_count": len(prepared),
        "exact_variant_budget": count,
        "exact_variant_offset": offset,
        "exact_variant_evaluation_interval": {
            "start_inclusive": offset, "stop_exclusive": offset + count,
        },
        "exact_variant_evaluated_count": len(evaluated),
        "survivor_count": len(survivors),
        "evaluated": evaluated,
        "deferred": deferred,
    }
    return tuple(survivors), audit


__all__ = ["SampledPathEnvelope", "contact_height_bounds", "sampled_height_path_states",
           "sampled_table_path_requirement", "search_height_projected_pregrasps"]
