"""SciPy orchestration for bounded opposition-pose refinement."""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.fast_filter import (
    fast_filter_predictions, fast_filter_pregrasp_paths,
)
from kcg_connector.grasp.carts_v2.models import CandidateSeed, V2Inputs
from kcg_connector.grasp.carts_v2.opposition_refiner import (
    OppositionPoseEvaluator, REFINEMENT_VARIABLES,
    finite_refinement_vector, initial_refinement_population,
    ranked_refinement_population, refinement_public_row,
    three_registered_contacts,
)


def _refinement_bounds(inputs, anchor, settings):
    translation = float(settings["translation_bound_m"])
    rotation = float(settings["rotation_bound_rad"])
    palm_bound = float(settings["palm_configuration_bound_rad"])
    palm = float(anchor.palm_configuration_rad)
    palm_low = max(-palm_bound, float(settings["opposition_palm_lower_rad"]) - palm)
    palm_high = min(palm_bound, float(settings["opposition_palm_upper_rad"]) - palm)
    return ([(-translation, translation)] * 3 + [(-rotation, rotation)] * 3
            + [(palm_low, palm_high)])


def _witness_vectors(inputs, anchor, witness_world_z, exact_deficit):
    original_world = inputs.frozen_world_from_object @ anchor.object_from_hand_matrix()
    world_delta = np.array((0.0, 0.0,
                            float(witness_world_z) - float(original_world[2, 3])))
    task_frame = np.asarray(inputs.object_contract.task_frame_rotation_object)
    witness = np.zeros(7)
    witness[:3] = task_frame.T @ (
        inputs.frozen_world_from_object[:3, :3].T @ world_delta)
    safe = np.array(witness, copy=True)
    safe_world = np.array((0.0, 0.0, float(exact_deficit)))
    safe[:3] += task_frame.T @ (
        inputs.frozen_world_from_object[:3, :3].T @ safe_world)
    return witness, safe


def _run_global_stages(probe, bounds, population, seed, witness, safe):
    def contact_objective(vector):
        try:
            value = probe.evaluate(vector)["maximum_contact_gap_m"]
        except RuntimeError:
            return math.inf
        return math.inf if value is None else float(value)

    stage_a = differential_evolution(
        contact_objective, bounds, init=population, maxiter=3, polish=False,
        seed=seed, workers=1, updating="immediate", tol=0.0,
        callback=lambda _x, convergence=0.0: bool(probe.stop_reason))
    stage_b_population = ranked_refinement_population(
        probe.cache.values(), population)
    stage_b_population[0], stage_b_population[1] = witness, safe

    def table_objective(vector):
        try:
            row = probe.evaluate(vector)
            if row["status"] != "A_FIXED_PHASE_GAP_WITNESS":
                return math.inf
            value = probe.evaluate(vector, with_table=True)[
                "proxy_table_deficit_to_required_clearance_m"]
        except RuntimeError:
            return math.inf
        return math.inf if value is None else float(value)

    stage_b = differential_evolution(
        table_objective, bounds, init=stage_b_population, maxiter=3,
        polish=False, seed=seed, workers=1, updating="immediate", tol=0.0,
        callback=lambda _x, convergence=0.0: bool(probe.stop_reason))
    return stage_a, stage_b


def _run_constrained_local(probe, bounds, budget):
    rows = [row for row in probe.cache.values()
            if row["status"] == "A_FIXED_PHASE_GAP_WITNESS"
            and row["table_evaluated"]]
    rows.sort(key=lambda row: (
        float(row["proxy_table_deficit_to_required_clearance_m"]),
        float(row["maximum_contact_gap_m"]), row["candidate_id"]))
    if not rows or budget - len(probe.cache) - 9 < 8:
        return None
    lower = np.asarray([row[0] for row in bounds], dtype=np.float64)
    upper = np.asarray([row[1] for row in bounds], dtype=np.float64)
    center, half_span = 0.5 * (lower + upper), 0.5 * (upper - lower)
    start = np.asarray([rows[0]["variables"][name]
                        for name in REFINEMENT_VARIABLES])
    probe.budget = budget - 9

    def decode(value):
        return center + half_span * np.asarray(value, dtype=np.float64)

    def objective(value):
        try:
            row = probe.evaluate(decode(value))
            if row["maximum_contact_gap_m"] is None:
                return 1.0
            deficit = probe.evaluate(decode(value), with_table=True)[
                "proxy_table_deficit_to_required_clearance_m"]
        except RuntimeError:
            return 1.0
        return 1.0 if deficit is None else float(deficit)

    def constraint(value):
        try:
            gap = probe.evaluate(decode(value))["maximum_contact_gap_m"]
        except RuntimeError:
            return -1.0
        return -1.0 if gap is None else probe.contact_distance - float(gap)

    result = minimize(
        objective, (start - center) / half_span, method="SLSQP",
        bounds=[(-1.0, 1.0)] * 7,
        constraints=({"type": "ineq", "fun": constraint},),
        options={"maxiter": 7, "ftol": 1.0e-8, "eps": 0.02, "disp": False})
    return {
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "iterations": int(result.nit),
        "objective_m": (
            None if not np.isfinite(result.fun) or float(result.fun) >= 1.0
            else float(result.fun)
        ),
        "objective_status": (
            "BUDGET_SENTINEL_NOT_A_PHYSICAL_METRIC"
            if np.isfinite(result.fun) and float(result.fun) >= 1.0
            else "FINITE_TABLE_DEFICIT_METRES"
        ),
    }


def _exact_revalidation(inputs, probe, guidance, budget, required):
    predictor = SequentialClosurePredictor(inputs)
    tolerance = float(inputs.config.section("fast_filter")[
        "table_penetration_tolerance_m"])
    passed, records = [], []
    for row in guidance[:8]:
        if 1 + len(probe.cache) + len(records) >= budget:
            break
        if time.monotonic() >= probe.deadline:
            probe.stop_reason = "MAXIMUM_WALL_TIME"
            break
        seed = row["candidate"]
        prediction = predictor.predict(seed)
        pregrasp = fast_filter_pregrasp_paths(
            inputs, ((seed, seed.pregrasp_closure_phases),))[0]
        fast = fast_filter_predictions(inputs, (prediction,))[0]
        values = [value for value in (
            pregrasp.get("minimum_table_clearance_m"),
            fast.minimum_table_clearance_m) if value is not None]
        clearance = None if not values else float(min(values))
        table_deficit = (
            None if clearance is None else max(0.0, required - clearance)
        )
        rejection_reasons = []
        if not three_registered_contacts(inputs, prediction):
            rejection_reasons.append("THREE_REGISTERED_CONTACTS_NOT_CONFIRMED")
        if pregrasp.get("accepted") is not True:
            rejection_reasons.append("PREGRASP_PATH_REJECTED")
        if fast.status != "FAST_SURVIVE" or not fast.sequential_closure_sweep_pass:
            rejection_reasons.append("SEQUENTIAL_CLOSURE_PATH_REJECTED")
        if clearance is None:
            rejection_reasons.append("TABLE_CLEARANCE_UNRESOLVED")
        elif clearance < required - tolerance:
            rejection_reasons.append("TABLE_CLEARANCE_BELOW_REQUIRED")
        accepted = bool(
            three_registered_contacts(inputs, prediction)
            and pregrasp.get("accepted") is True
            and fast.status == "FAST_SURVIVE"
            and fast.sequential_closure_sweep_pass
            and clearance is not None and clearance >= required - tolerance)
        records.append({
            "candidate_id": seed.candidate_id,
            "status": "B_FULL_PASS" if accepted else "B_EXACT_REJECT",
            "closure_status": prediction.status, "closure_reason": prediction.reason,
            "contact_count": len(prediction.contacts),
            "final_contact_stop_phases": list(prediction.final_closure_phases),
            "pregrasp_path_accepted": pregrasp.get("accepted") is True,
            "pregrasp_path_reasons": list(pregrasp.get("reasons", ())),
            "fast_filter_status": fast.status,
            "fast_filter_reasons": list(fast.reasons),
            "minimum_table_clearance_m": clearance,
            "required_table_clearance_m": required,
            "numerical_tolerance_m": tolerance,
            "table_deficit_to_required_clearance_m": table_deficit,
            "rejection_reasons": rejection_reasons,
            "minimum_nonallowed_surface_clearance_m": None,
            "nonallowed_surface_gate": "FULL_PATH_BINARY_FCL_ONLY",
            "variables": row["variables"],
            "object_from_hand_row_major": list(seed.object_from_hand),
            "pregrasp_joint_positions_rad": list(seed.pregrasp_joint_positions_rad),
        })
        if accepted:
            passed.append(seed)
    return passed, records


def refine_opposition_pose(
    inputs: V2Inputs, anchor: CandidateSeed, *, three_contact_world_z_m: float,
    reference_contact_stop_phases: Sequence[float],
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Refine one fixed anchor/preshape without changing physical thresholds."""

    settings = inputs.config.section("candidate_generation")["refinement"]
    budget, wall = (int(settings["maximum_stage_a_plus_b_evaluations_per_seed"]),
                    float(settings["maximum_wall_time_s"]))
    if not 1 <= budget <= 300 or not 0.0 < wall <= 1500.0:
        raise ValueError("opposition refinement budget changed")
    bounds = _refinement_bounds(inputs, anchor, settings)
    phases = finite_refinement_vector(reference_contact_stop_phases, 3, "stop phases")
    probe = OppositionPoseEvaluator(
        inputs, anchor, phases, bounds, budget, time.monotonic() + wall)
    witness, _placeholder = _witness_vectors(inputs, anchor,
                                              three_contact_world_z_m, 0.0)
    witness_row = probe.evaluate(witness)
    exact_table = probe.exact_table_replay(witness_row["candidate"])
    deficit = exact_table["table_deficit_to_required_clearance_m"]
    if witness_row["status"] != "A_FIXED_PHASE_GAP_WITNESS" or deficit is None:
        raise ValueError("registered three-contact witness no longer replays")
    witness, safe = _witness_vectors(inputs, anchor, three_contact_world_z_m, deficit)
    random_seed = int(settings["optimizer_random_seed"])
    population = initial_refinement_population(
        bounds, witness, safe, random_seed)
    stage_a, stage_b = _run_global_stages(
        probe, bounds, population, random_seed, witness, safe)
    local = _run_constrained_local(probe, bounds, budget)
    gap_rows = [row for row in probe.cache.values()
                if row["status"] == "A_FIXED_PHASE_GAP_WITNESS"]
    guidance = [row for row in gap_rows if row["table_evaluated"]]
    guidance.sort(key=lambda row: (
        float(row["proxy_table_deficit_to_required_clearance_m"]),
        float(row["maximum_contact_gap_m"]), float(row["contact_gap_range_m"]),
        float(np.linalg.norm(list(row["variables"].values())[:3])),
        row["candidate_id"]))
    required = float(inputs.config.section("height_projection")[
        "table_operation_clearance_m"])
    passed, exact_rows = _exact_revalidation(
        inputs, probe, guidance, budget, required)
    audit = {
        "schema_version": "carts_opposition_bounded_refinement_v1",
        "claim_scope": "OFFLINE_BOUNDED_REFINEMENT_NOT_TASK_IK_OR_DYNAMIC_SUCCESS",
        "anchor_candidate_id": anchor.candidate_id,
        "fixed_pregrasp_closure_phases": list(anchor.pregrasp_closure_phases),
        "reference_contact_stop_phases": phases.tolist(),
        "variable_order": list(REFINEMENT_VARIABLES),
        "translation_frame": "OBJECT_TASK_FRAME",
        "rotation_composition": "ANCHOR_HAND_FRAME_RIGHT_MULTIPLY_XYZ",
        "bounds": [list(row) for row in bounds], "evaluation_budget": budget,
        "evaluation_count_unit": "UNIQUE_CANDIDATE_POSE",
        "guidance_evaluation_count": len(probe.cache),
        "source_witness_exact_table_replay_count": 1,
        "exact_revalidation_count": len(exact_rows),
        "total_evaluation_count": 1 + len(probe.cache) + len(exact_rows),
        "evaluation_count_composition": {
            "guidance_unique_candidate_pose_count": len(probe.cache),
            "source_witness_exact_table_replay_count": 1,
            "fresh_exact_revalidation_candidate_pose_count": len(exact_rows),
        },
        "stop_reason": probe.stop_reason or "BOUNDED_OPTIMIZER_COMPLETE",
        "negative_evidence_scope": (
            "NO_PASS_FOUND_NOT_CONTINUOUS_INFEASIBILITY_PROOF;"
            "FIXED_ANCHOR_FIXED_PRESHAPE_AND_REGISTERED_BOUNDS_ONLY"
        ),
        "witness_vector": witness.tolist(), "pure_table_projection_vector": safe.tolist(),
        "source_witness_exact_table_replay": exact_table,
        "table_search_metric_scope": (
            "STRATIFIED_REAL_MESH_STATES_ORDERING_ONLY_TOP8_FULL_PATH_REQUIRED"),
        "stage_a_best_contact_gap_m": float(stage_a.fun),
        "stage_b_best_table_deficit_m": (
            None if not np.isfinite(stage_b.fun) else float(stage_b.fun)),
        "stage_b_constrained_local_result": local,
        "final_best_proxy_table_deficit_m": (
            None if not guidance else float(guidance[0][
                "proxy_table_deficit_to_required_clearance_m"])),
        "fixed_phase_gap_witness_count": len(gap_rows),
        "table_evaluated_gap_witness_count": len(guidance),
        "b_full_pass_count": len(passed),
        "top_guidance": [refinement_public_row(row) for row in guidance[:16]],
        "exact_revalidation": exact_rows,
        "research_executable_candidate": False,
        "isaac_started": False, "hardware_authorized": False,
    }
    return tuple(passed), audit


__all__ = ["refine_opposition_pose"]
