"""Post-hoc assembly-arrival decision for the D38999 multilayer chain.

This module independently scores an existing nominal-bench report.  Reported
booleans are never sufficient by themselves, and no post-hoc event or contact
truth is exposed as a task-controller input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .d38999_multilayer_metal_stop import load_metal_stop_contract


TASK_ID = "EIGHT-HOUR-E8-ARRIVAL-DECISION"
E7_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-E7-METAL-STOP/TASK_RESULT.json"
)
EXPECTED_BENCH_ID = "TASK-R12-MULTILAYER-005"
EXPECTED_REPORT_MODE = "formal_multilayer_nominal"

FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/isaac/d38999_multilayer_nominal_bench.py": (
        "e123e5820c1add57628f5c06127fad3dbe4088d1ef5513460b6d5d71c8b4078f"
    ),
    "src/kcg_connector/kcg_connector/d38999_multilayer_metal_stop.py": (
        "dd49f29896909b5c007dfe37e3f528ad70c263ee321f8a7fd3d360cb1694e664"
    ),
    E7_RESULT_PATH: (
        "1c771f150f6b668854acf3d1764cd6eb044a4e84024698b1984d8e365036c8a7"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class ArrivalDecisionContract:
    source_rows: tuple[tuple[str, str], ...]
    event_order: tuple[str, ...]
    nominal_event_positions_m: tuple[tuple[str, float], ...]
    required_precedence_edges: tuple[tuple[str, str], ...]
    event_position_tolerance_m: float
    force_component_limit_n_per_driven_body: float
    force_limit_semantics: str
    torque_component_limit_nm: float
    maximum_fixed_receptacle_translation_drift_m: float
    maximum_noncompliant_hard_penetration_m: float
    fixed_stop_path: str
    plug_stop_path: str
    current_e7_outcome: str
    current_e7_dynamic_metal_stop_passed: bool
    current_e7_evidence_sha256: str


@dataclass(frozen=True)
class ArrivalDecisionReadiness:
    e7_evidence_path: str
    e7_evidence_sha256: str
    e7_dynamic_metal_stop_passed: bool
    nominal_report_available: bool
    posthoc_audit_only: bool
    report_truth_routed_to_controller: bool


def load_arrival_decision_contract(
    repository_root: str | Path,
) -> ArrivalDecisionContract:
    root = Path(repository_root).resolve()
    rows: list[tuple[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"frozen E8 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen E8 source hash mismatch: {relative}")
        rows.append((relative, actual))

    master = _mapping(
        yaml.safe_load(
            (root / "src/kcg_connector/config/"
             "d38999_master_model_contract_v1.yaml").read_text(
                encoding="utf-8"
            )
        ),
        "master model contract",
    )
    events = _mapping(master.get("assembly_events"), "master.assembly_events")
    ordered = events.get("ordered")
    edges = events.get("required_precedence_edges")
    if not isinstance(ordered, list) or len(ordered) != 7:
        raise ValueError("arrival decision requires exactly seven events")
    if not isinstance(edges, list) or not edges:
        raise ValueError("arrival decision requires precedence edges")
    event_order: list[str] = []
    positions: list[tuple[str, float]] = []
    for index, raw in enumerate(ordered, start=1):
        row = _mapping(raw, f"assembly event {index}")
        if row.get("ordinal") != index:
            raise ValueError("assembly event ordinals must be contiguous")
        name = str(row.get("name"))
        event_order.append(name)
        positions.append(
            (name, _finite(row.get("nominal_separation_m"), f"{name} position"))
        )
    precedence: list[tuple[str, str]] = []
    for raw in edges:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("each precedence edge must have two event names")
        before, after = str(raw[0]), str(raw[1])
        if before not in event_order or after not in event_order:
            raise ValueError("precedence edge names must be declared events")
        precedence.append((before, after))

    limits = _mapping(master.get("acceptance_limits"), "master.acceptance_limits")
    if limits.get("force_limit_semantics") != (
        "each_driven_body_each_component_not_whole_robot_wrist_total"
    ):
        raise ValueError("driver force limit semantics changed")

    metal = load_metal_stop_contract(root)
    if metal.event_name != event_order[-1] or metal.event_ordinal != 7:
        raise ValueError("E7 metal stop does not match the seventh event")

    bench_source = (
        root / "src/kcg_connector/isaac/d38999_multilayer_nominal_bench.py"
    ).read_text(encoding="utf-8")
    required_report_fields = (
        '"event_first"',
        '"observed_event_order"',
        '"maximum_driver_force_component_n"',
        '"maximum_driver_torque_component_nm"',
        '"maximum_fixed_receptacle_translation_drift_m"',
        '"hard_penetrations_over_limit"',
        '"solver_error_count"',
        '"object_pose_write_after_physics_start_count"',
        '"control_input_policy"',
    )
    if any(field not in bench_source for field in required_report_fields):
        raise ValueError("nominal bench report schema no longer supports E8")

    e7 = _mapping(
        json.loads((root / E7_RESULT_PATH).read_text(encoding="utf-8")),
        "E7 task result",
    )
    if (
        e7.get("task_id") != "EIGHT-HOUR-E7-METAL-STOP"
        or e7.get("outcome") != "OFFLINE_PASS"
        or type(e7.get("dynamic_metal_stop_pass_claimed")) is not bool
        or e7.get("software_pose_write_requested") is not False
        or e7.get("determined_by_physical_collision_not_pose_or_boolean") is not True
    ):
        raise ValueError("E7 evidence does not support E8")

    return ArrivalDecisionContract(
        source_rows=tuple(rows),
        event_order=tuple(event_order),
        nominal_event_positions_m=tuple(positions),
        required_precedence_edges=tuple(precedence),
        event_position_tolerance_m=_finite(
            limits.get("event_position_tolerance_m"), "event tolerance"
        ),
        force_component_limit_n_per_driven_body=_finite(
            limits.get("force_component_limit_n_per_driven_body"), "force limit"
        ),
        force_limit_semantics=str(limits.get("force_limit_semantics")),
        torque_component_limit_nm=_finite(
            limits.get("torque_component_limit_nm"), "torque limit"
        ),
        maximum_fixed_receptacle_translation_drift_m=_finite(
            limits.get("maximum_fixed_receptacle_translation_drift_m"),
            "fixed drift limit",
        ),
        maximum_noncompliant_hard_penetration_m=_finite(
            limits.get("maximum_noncompliant_hard_penetration_m"),
            "hard penetration limit",
        ),
        fixed_stop_path=metal.fixed_stop_path,
        plug_stop_path=metal.plug_stop_path,
        current_e7_outcome=str(e7.get("outcome")),
        current_e7_dynamic_metal_stop_passed=e7[
            "dynamic_metal_stop_pass_claimed"
        ],
        current_e7_evidence_sha256=FROZEN_SOURCES[E7_RESULT_PATH],
    )


def evaluate_arrival_readiness(
    contract: ArrivalDecisionContract,
    readiness: ArrivalDecisionReadiness,
) -> str | None:
    if (
        readiness.e7_evidence_path != E7_RESULT_PATH
        or readiness.e7_evidence_sha256 != contract.current_e7_evidence_sha256
    ):
        return "E7_EVIDENCE_ID_MISMATCH"
    if (
        contract.current_e7_dynamic_metal_stop_passed is not True
        or readiness.e7_dynamic_metal_stop_passed is not True
    ):
        return "E7_METAL_STOP_NOT_DYNAMIC"
    if readiness.nominal_report_available is not True:
        return "NOMINAL_DYNAMIC_REPORT_MISSING"
    if readiness.posthoc_audit_only is not True:
        return "ARRIVAL_DECISION_MUST_BE_POSTHOC_ONLY"
    if readiness.report_truth_routed_to_controller is not False:
        return "REPORT_TRUTH_TO_CONTROLLER_FORBIDDEN"
    return None


def build_arrival_decision_request(
    contract: ArrivalDecisionContract,
    readiness: ArrivalDecisionReadiness,
) -> dict[str, Any]:
    """Create only an audit request; never create a controller command."""

    rejection = evaluate_arrival_readiness(contract, readiness)
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "request_ready": rejection is None,
        "rejection_code": rejection,
        "posthoc_audit_requested": rejection is None,
        "event_or_contact_truth_routed_to_controller": False,
        "position_boolean_requested": False,
        "software_pose_write_requested": False,
        "force_or_moment_command_requested": False,
        "robot_commands_emitted": 0,
        "assembly_arrival_claimed": False,
        "dynamic_assembly_pass_claimed": False,
        "control_authorized": False,
    }


def audit_posthoc_arrival_report(
    contract: ArrivalDecisionContract,
    report: Mapping[str, Any],
    *,
    evidence_is_offline_fixture: bool,
) -> dict[str, Any]:
    """Independently evaluate one completed nominal report."""

    data = _mapping(report, "nominal report")
    reasons: list[str] = []
    if evidence_is_offline_fixture is not False:
        reasons.append("OFFLINE_FIXTURE_NOT_DYNAMIC_EVIDENCE")
    if data.get("bench_id") != EXPECTED_BENCH_ID:
        reasons.append("BENCH_ID_MISMATCH")
    if data.get("mode") != EXPECTED_REPORT_MODE:
        reasons.append("REPORT_MODE_MISMATCH")
    if data.get("simulation_started") is not True:
        reasons.append("SIMULATION_NOT_STARTED")

    event_first = data.get("event_first")
    if not isinstance(event_first, Mapping):
        event_first = {}
        reasons.append("EVENT_FIRST_MAP_MISSING")
    observed = data.get("observed_event_order")
    if observed != list(contract.event_order):
        reasons.append("EVENT_ORDER_INVALID")
    nominal_positions = dict(contract.nominal_event_positions_m)
    observed_steps: dict[str, int] = {}
    for event in contract.event_order:
        raw = event_first.get(event)
        if not isinstance(raw, Mapping):
            reasons.append(f"EVENT_MISSING:{event}")
            continue
        step = raw.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            reasons.append(f"EVENT_STEP_INVALID:{event}")
        else:
            observed_steps[event] = step
        try:
            position = _finite(
                raw.get("datum_B_separation_m"), f"{event} observed position"
            )
        except ValueError:
            reasons.append(f"EVENT_POSITION_INVALID:{event}")
        else:
            if abs(position - nominal_positions[event]) > contract.event_position_tolerance_m:
                reasons.append(f"EVENT_POSITION_OUT_OF_TOLERANCE:{event}")
        if event == contract.event_order[-1] and raw.get("source") != (
            "physx_continuous_real_collision"
        ):
            reasons.append("METAL_STOP_SOURCE_NOT_PHYSICAL_COLLISION")
    for before, after in contract.required_precedence_edges:
        if before in observed_steps and after in observed_steps:
            if observed_steps[before] >= observed_steps[after]:
                reasons.append(f"PRECEDENCE_VIOLATION:{before}->{after}")

    contact_pairs = data.get("contact_pairs")
    exact_stop_pair_found = False
    if isinstance(contact_pairs, list):
        expected_pair = {contract.fixed_stop_path, contract.plug_stop_path}
        for raw in contact_pairs:
            if not isinstance(raw, Mapping):
                continue
            paths = raw.get("collider_paths")
            if isinstance(paths, list) and len(paths) == 2 and set(paths) == expected_pair:
                active = raw.get("active_step_count")
                if isinstance(active, int) and not isinstance(active, bool) and active > 0:
                    exact_stop_pair_found = True
                    break
    if not exact_stop_pair_found:
        reasons.append("EXACT_PHYSICAL_METAL_STOP_PAIR_NOT_PROVEN")

    numeric_limits = (
        (
            "maximum_driver_force_component_n",
            contract.force_component_limit_n_per_driven_body,
            "DRIVER_FORCE_COMPONENT_LIMIT_EXCEEDED",
        ),
        (
            "maximum_driver_torque_component_nm",
            contract.torque_component_limit_nm,
            "DRIVER_TORQUE_COMPONENT_LIMIT_EXCEEDED",
        ),
        (
            "maximum_fixed_receptacle_translation_drift_m",
            contract.maximum_fixed_receptacle_translation_drift_m,
            "FIXED_RECEPTACLE_DRIFT_LIMIT_EXCEEDED",
        ),
    )
    for field, limit, code in numeric_limits:
        try:
            value = _finite(data.get(field), field)
        except ValueError:
            reasons.append(f"INVALID_NUMERIC_FIELD:{field}")
        else:
            if value < 0.0 or value > limit:
                reasons.append(code)

    penetrations = data.get("hard_penetrations_over_limit")
    if not isinstance(penetrations, list) or penetrations:
        reasons.append("HARD_PENETRATION_GATE_FAILED")
    try:
        solver_errors = _nonnegative_integer(
            data.get("solver_error_count"), "solver error count"
        )
    except ValueError:
        reasons.append("SOLVER_ERROR_COUNT_INVALID")
    else:
        if solver_errors != 0:
            reasons.append("SOLVER_ERROR_PRESENT")
    try:
        pose_writes = _nonnegative_integer(
            data.get("object_pose_write_after_physics_start_count"),
            "post-start pose write count",
        )
    except ValueError:
        reasons.append("POSE_WRITE_COUNT_INVALID")
    else:
        if pose_writes != 0:
            reasons.append("POST_START_POSE_WRITE_PRESENT")

    policy = data.get("control_input_policy")
    if not isinstance(policy, Mapping):
        reasons.append("CONTROL_INPUT_POLICY_MISSING")
    else:
        required_false = (
            "contact_object_name_used",
            "contact_normal_used",
            "contact_manifold_used",
            "event_truth_used",
        )
        if any(policy.get(field) is not False for field in required_false):
            reasons.append("CONTROL_TRUTH_FIREWALL_FAILED")
        if policy.get("posthoc_contact_truth_for_scoring_only") is not True:
            reasons.append("POSTHOC_ONLY_BOUNDARY_FAILED")

    if data.get("formal_p1_pass_claimed") is not False:
        reasons.append("FORMAL_P1_CLAIM_FORBIDDEN_IN_E8")
    if data.get("formal_r12_generated") is not False:
        reasons.append("FORMAL_R12_CLAIM_FORBIDDEN_IN_E8")
    if data.get("hardware_authorized") is not False:
        reasons.append("HARDWARE_AUTHORIZATION_FORBIDDEN")
    independent_pass = not reasons
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "posthoc_only": True,
        "reported_status": data.get("status"),
        "reported_passed": data.get("passed"),
        "self_report_trusted_without_recomputation": False,
        "accepted": independent_pass,
        "rejection_codes": reasons,
        "assembly_arrival_proven": independent_pass,
        "event_or_contact_truth_routed_to_controller": False,
        "control_authorized": False,
    }


def current_readiness(
    contract: ArrivalDecisionContract,
) -> ArrivalDecisionReadiness:
    return ArrivalDecisionReadiness(
        e7_evidence_path=E7_RESULT_PATH,
        e7_evidence_sha256=contract.current_e7_evidence_sha256,
        e7_dynamic_metal_stop_passed=contract.current_e7_dynamic_metal_stop_passed,
        nominal_report_available=False,
        posthoc_audit_only=True,
        report_truth_routed_to_controller=False,
    )


__all__ = [
    "E7_RESULT_PATH",
    "ArrivalDecisionContract",
    "ArrivalDecisionReadiness",
    "audit_posthoc_arrival_report",
    "build_arrival_decision_request",
    "current_readiness",
    "evaluate_arrival_readiness",
    "load_arrival_decision_contract",
]
