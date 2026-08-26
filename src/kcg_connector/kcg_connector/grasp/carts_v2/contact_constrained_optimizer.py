"""One bounded COBYLA route for proxy three-finger contact refinement."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from kcg_connector.grasp.carts_v2.contact_interval_solver import (
    ProxyContactIntervalEvaluator,
)
from kcg_connector.grasp.carts_v2.full_palm_search import bind_pregrasp
from kcg_connector.grasp.carts_v2.models import CandidateSeed, V2Inputs


_VARIABLES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "dq_p",
              "preshape_1", "preshape_2", "preshape_3")
_MAXIMUM_CANDIDATES = 48
_MAXIMUM_EVALUATIONS = 120
_CONSTRAINT_SCALE_M = 0.001
_FAILURE_OBJECTIVE_M = 1.0


def _finite(value, fallback=-1.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _physical_bounds(inputs: V2Inputs, anchor: CandidateSeed) -> np.ndarray:
    config = inputs.config.section("contact_optimization")
    translation = float(config["translation_bound_m"])
    rotation = math.radians(float(config["rotation_bound_deg"]))
    palm_bound = math.radians(float(config["palm_configuration_bound_deg"]))
    preshape = tuple(float(value) for value in config["preclose_bounds"])
    names = tuple(inputs.hand_model.independent_joint_names)
    palm_index = names.index("f1j1")
    lower, upper = inputs.hand_model.joint_limit_vectors()
    palm = float(anchor.palm_configuration_rad)
    palm_delta = (max(-palm_bound, float(lower[palm_index]) - palm),
                  min(palm_bound, float(upper[palm_index]) - palm))
    bounds = ([(-translation, translation)] * 3
              + [(-rotation, rotation)] * 3 + [palm_delta]
              + [preshape] * 3)
    result = np.asarray(bounds, dtype=np.float64)
    if result.shape != (10, 2) or np.any(result[:, 0] >= result[:, 1]):
        raise ValueError("CONTACT_OPTIMIZATION_BOUNDS_EMPTY")
    return result


def _normalize(physical: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    center = np.mean(bounds, axis=1)
    half = 0.5 * (bounds[:, 1] - bounds[:, 0])
    return (physical - center) / half


def _denormalize(normalized: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    center = np.mean(bounds, axis=1)
    half = 0.5 * (bounds[:, 1] - bounds[:, 0])
    return center + half * normalized


def _candidate(inputs, anchor, physical):
    pose = np.array(anchor.object_from_hand_matrix(), copy=True)
    pose[:3, :3] = pose[:3, :3] @ Rotation.from_euler(
        "xyz", physical[3:6]).as_matrix()
    pose[:3, 3] += np.asarray(
        inputs.object_contract.task_frame_rotation_object) @ physical[:3]
    palm = float(anchor.palm_configuration_rad) + float(physical[6])
    reference = np.asarray(anchor.pregrasp_joint_positions_rad).copy()
    palm_index = inputs.hand_model.independent_joint_names.index("f1j1")
    reference[palm_index] = palm
    approach = np.asarray(anchor.approach_direction_object, dtype=np.float64)
    local_approach = anchor.object_from_hand_matrix()[:3, :3].T @ approach
    moved_approach = pose[:3, :3] @ local_approach
    seed = replace(
        anchor,
        object_from_hand=tuple(float(value) for value in pose.ravel()),
        pregrasp_joint_positions_rad=tuple(float(value) for value in reference),
        pregrasp_closure_phases=tuple(float(value) for value in physical[7:10]),
        palm_configuration_rad=palm,
        approach_direction_object=tuple(float(value) for value in moved_approach),
    )
    return bind_pregrasp(inputs, seed, seed.pregrasp_closure_phases)


def _feasibility(cheap: Mapping, interval: Mapping,
                 table_clearance_m: float) -> tuple[bool, tuple[str, ...]]:
    reasons = []
    fingers = tuple(cheap.get("finger_proxies", ()))
    if len(fingers) != 3 or not all(
            bool(row.get("closing_direction_reasonable")) for row in fingers):
        reasons.append("CHEAP_CLOSING_DIRECTION_REJECT")
    if _finite(cheap.get("hard_margin_m")) <= 0.0:
        reasons.append("HARD_SURFACE_MARGIN_REJECT")
    if _finite(cheap.get("table_margin_m")) < table_clearance_m:
        reasons.append("TABLE_OPERATION_CLEARANCE_REJECT")
    if _finite(cheap.get("self_margin_m")) <= 0.0:
        reasons.append("SELF_MARGIN_REJECT")
    if interval.get("status") != "PROXY_INTERVAL_SURVIVE" or len(
            interval.get("finger_intervals", ())) != 3:
        reasons.append("THREE_PROXY_INTERVALS_REJECT")
    return not reasons, tuple(reasons)


def _constraint_values(row: Mapping, table_clearance_m: float) -> np.ndarray:
    if row.get("candidate") is None:
        return np.full(5, -1.0)
    cheap, interval = row["cheap"], row["interval"]
    fingers = tuple(cheap.get("finger_proxies", ()))
    direction = 1.0 if len(fingers) == 3 and all(
        bool(item.get("closing_direction_reasonable")) for item in fingers) else -1.0
    interval_gate = 1.0 if interval.get("status") == "PROXY_INTERVAL_SURVIVE" else -1.0
    return np.asarray((direction,
        _finite(cheap.get("hard_margin_m")) / _CONSTRAINT_SCALE_M,
        (_finite(cheap.get("table_margin_m")) - table_clearance_m) /
        _CONSTRAINT_SCALE_M,
        _finite(cheap.get("self_margin_m")) / _CONSTRAINT_SCALE_M,
        interval_gate), dtype=np.float64)


def _selection_key(row: Mapping) -> tuple:
    cheap = row["cheap"]
    physical = np.asarray(row["physical_variables"], dtype=np.float64)
    return (_finite(cheap.get("maximum_positive_gap_m"), math.inf),
            _finite(cheap.get("gap_imbalance_m"), math.inf),
            -_finite(cheap.get("hard_margin_m")),
            -_finite(cheap.get("table_margin_m")),
            -_finite(cheap.get("self_margin_m")),
            float(np.linalg.norm(physical[:3])),
            float(np.linalg.norm(physical[3:6])), abs(float(physical[6])),
            tuple(float(value) for value in physical[7:10]), row["candidate_id"])


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()
                if key != "_seed"}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


class _SeedProbe:
    def __init__(self, inputs, evaluator, anchor, specification):
        self.inputs, self.evaluator, self.anchor = inputs, evaluator, anchor
        self.specification = dict(specification)
        self.table_clearance_m = float(inputs.config.section("height_projection")[
            "table_operation_clearance_m"])
        self.bounds = _physical_bounds(inputs, anchor)
        self.cache: dict[bytes, dict[str, object]] = {}
        start = np.zeros(10); start[7:10] = anchor.pregrasp_closure_phases
        self.start = _normalize(start, self.bounds)

    def evaluate(self, normalized) -> dict[str, object]:
        vector = np.asarray(normalized, dtype=np.float64)
        if vector.shape != (10,) or not np.all(np.isfinite(vector)):
            return self._invalid("NONFINITE_OR_WRONG_SIZE_VECTOR", vector)
        key = np.asarray(vector, dtype="<f8").tobytes()
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= _MAXIMUM_EVALUATIONS:
            return self._invalid("MAXIMUM_UNIQUE_EVALUATIONS", vector)
        physical = _denormalize(vector, self.bounds)
        if np.any(np.abs(vector) > 1.0 + 1.0e-12):
            row = self._invalid("NORMALIZED_BOUND_REJECT", vector, physical)
        else:
            row = self._evaluate_candidate(vector, physical)
        self.cache[key] = row
        return row

    def _invalid(self, reason, vector, physical=None):
        values = np.full(10, np.nan) if physical is None else physical
        return {"candidate_id": "", "normalized_variables": list(vector),
                "physical_variables": list(values), "candidate": None,
                "cheap": {}, "interval": {}, "feasible": False,
                "reasons": (reason,)}

    def _evaluate_candidate(self, normalized, physical):
        try:
            candidate = _candidate(self.inputs, self.anchor, physical)
            spec = {**self.specification,
                    "palm_configuration_rad": candidate.palm_configuration_rad}
            cheap, interval = self.evaluator.evaluate(candidate, spec)
            feasible, reasons = _feasibility(
                cheap, interval, self.table_clearance_m)
        except (ValueError, RuntimeError) as error:
            return self._invalid(f"CANDIDATE_EVALUATION_ERROR:{error}",
                                 normalized, physical)
        return {"candidate_id": candidate.candidate_id,
                "normalized_variables": list(normalized),
                "physical_variables": list(physical), "candidate": candidate,
                "cheap": cheap, "interval": interval,
                "feasible": feasible, "reasons": reasons}

    def objective(self, normalized):
        value = self.evaluate(normalized)["cheap"].get("maximum_positive_gap_m")
        result = _finite(value, _FAILURE_OBJECTIVE_M)
        return result if result >= 0.0 else _FAILURE_OBJECTIVE_M

    def constraints(self, normalized):
        vector = np.asarray(normalized, dtype=np.float64)
        box = 1.0 - np.abs(vector) if vector.shape == (10,) else np.full(10, -1.0)
        return np.concatenate((box, _constraint_values(
            self.evaluate(normalized), self.table_clearance_m)))


def _public_evaluation(row: Mapping) -> dict[str, object]:
    result = {key: value for key, value in row.items() if key != "candidate"}
    result["candidate"] = None if row.get("candidate") is None else asdict(row["candidate"])
    return _json_value(result)


def _optimize_one(inputs, evaluator, anchor, specification):
    probe = _SeedProbe(inputs, evaluator, anchor, specification)
    probe.evaluate(probe.start)
    error = ""
    try:
        result = minimize(
            probe.objective, probe.start, method="COBYLA",
            constraints=({"type": "ineq", "fun": probe.constraints},),
            options={"maxiter": _MAXIMUM_EVALUATIONS, "rhobeg": 0.25,
                     "tol": 1.0e-4, "catol": 1.0e-9, "disp": False})
        optimizer = {"success": bool(result.success), "status": int(result.status),
                     "message": str(result.message), "nfev": int(result.nfev),
                     "objective_m": _finite(result.fun, None)}
    except (RuntimeError, ValueError) as caught:
        error, optimizer = str(caught), {"success": False, "message": str(caught)}
    feasible = [row for row in probe.cache.values() if row["feasible"]]
    chosen = min(feasible, key=_selection_key) if feasible else None
    status = ("GENERATOR_REFINEMENT_DIAGNOSTIC_OUTPUT"
              if chosen is not None else "OPTIMIZED_PROXY_REJECT")
    record = {"source_candidate_id": anchor.candidate_id, "status": status,
              "reason": "" if chosen is not None else (error or "NO_FEASIBLE_PROXY_EVALUATION"),
              "evaluation_budget": _MAXIMUM_EVALUATIONS,
              "unique_evaluation_count": len(probe.cache), "optimizer": optimizer,
              "physical_bounds": {name: list(bound) for name, bound in zip(
                  _VARIABLES, probe.bounds)},
              "selected": None if chosen is None else _public_evaluation(chosen),
              "evaluations": [_public_evaluation(row) for row in probe.cache.values()]}
    return None if chosen is None else chosen["candidate"], record


def optimize_contact_constrained_top48(
    inputs: V2Inputs, candidates: Sequence[CandidateSeed],
    specification_rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Run one fixed COBYLA route for each preregistered proxy Top candidate."""

    anchors = tuple(candidates)
    if len(anchors) > _MAXIMUM_CANDIDATES:
        raise ValueError("CONTACT_OPTIMIZATION_INPUT_EXCEEDS_TOP48")
    identifiers = [seed.candidate_id for seed in anchors]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("CONTACT_OPTIMIZATION_CANDIDATE_IDS_NOT_UNIQUE")
    specifications = {str(row.get("candidate_id", "")): dict(row)
                      for row in specification_rows}
    if any(identifier not in specifications for identifier in identifiers):
        raise ValueError("CONTACT_OPTIMIZATION_SPECIFICATION_MISSING")
    evaluator = ProxyContactIntervalEvaluator(inputs)
    survivors, records = [], []
    for anchor in anchors:
        candidate, record = _optimize_one(
            inputs, evaluator, anchor, specifications[anchor.candidate_id])
        records.append(record)
        if candidate is not None:
            survivors.append(candidate)
    audit = {"schema_version": "carts_contact_constrained_optimizer_v1",
             "claim_scope": "PROXY_ONLY_NOT_EXACT_CONTACT_COLLISION_TASK_OR_DYNAMIC",
             "optimizer": "SCIPY_COBYLA_SINGLE_START_SINGLE_ROUTE",
             "variable_order": list(_VARIABLES),
             "input_candidate_count": len(anchors),
             "optimized_proxy_survivor_count": len(survivors),
             "maximum_candidates": _MAXIMUM_CANDIDATES,
             "maximum_unique_evaluations_per_seed": _MAXIMUM_EVALUATIONS,
             "table_operation_clearance_m": float(inputs.config.section(
                 "height_projection")["table_operation_clearance_m"]),
             "constraint_scale_m": _CONSTRAINT_SCALE_M,
             "hardware_authorized": False, "formal_dynamic_pass": False,
             "candidate_records": records}
    return tuple(survivors), audit


__all__ = ["optimize_contact_constrained_top48"]
