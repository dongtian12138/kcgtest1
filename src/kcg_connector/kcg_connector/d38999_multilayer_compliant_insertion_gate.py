"""Evidence-bound D7 gate around the existing compliant insertion policy."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from kcg_connector.compliant_insertion import (
    ControllerState,
    InsertionObservation,
    InsertionState,
    load_compliant_insertion_config,
    step_compliant_insertion,
)


SCHEMA_VERSION = "kcg_d38999_multilayer_compliant_insertion_gate_v1"
FORMAL_FORCE_COMPONENT_LIMIT_N = 8.0
FORMAL_MOMENT_COMPONENT_LIMIT_NM = 0.30

FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/config/d38999_compliant_insertion_v2.yaml": (
        "0f16e9b2fc5d615a4e8035dfa21c4ec9a18a341b4320ecbc3e66b874be489703"
    ),
    "src/kcg_connector/kcg_connector/compliant_insertion.py": (
        "2885a9960592221c6a26059b20e4c2a7da9aaf85b0b2109ce349c952a9fa187e"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-A1-INIT-DATUM/TASK_RESULT.json": (
        "3c90b7edabdfbcc685fd0dea8222538ccb074d6654593c4059c028f407574476"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D1-RECEPTACLE-SAFE-STANDOFF/"
    "TASK_RESULT.json": (
        "64d373c590e7b1ddb88ff961451367e369aaf677450ec796cd4f115d955179cc"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D2-VISUAL-PREALIGN/"
    "TASK_RESULT.json": (
        "cfb2866fdf7d1a1e553404aff475cefedfb0dbc791808e78f7a38e8fbb27b7ee"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D3-LIGHT-FACE-CONTACT/"
    "TASK_RESULT.json": (
        "822af4fcc991c1bd7428ea74aeed91259bcac1d45f1652b0c2d4b03fbbdb99a2"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D4-LATERAL-CENTERING/"
    "TASK_RESULT.json": (
        "fc69c579df982f8584373564d95ed2656336c02f8d41438d93bcfdc561c94cf3"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D5-ATTITUDE-LEVELING/"
    "TASK_RESULT.json": (
        "cfe15363e9cd2d64cd32ce37057ebf40cec61015db1660f75ce7f55a9073b9c5"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D6-SAFE-KEY-SEARCH/"
    "TASK_RESULT.json": (
        "07f7831dd1367658f80e5d0407a408d7f2b900dd6451fd7369695293caa0d96b"
    ),
}


@dataclass(frozen=True)
class CompliantInsertionReadiness:
    """Only explicit evidence gates; no object/contact/event truth fields."""

    a2_nominal_insertion_dynamic_pass: bool
    d1_safe_standoff_dynamic_pass: bool
    d2_visual_prealign_dynamic_pass: bool
    d3_light_contact_dynamic_pass: bool
    d4_lateral_centering_dynamic_pass: bool
    d5_attitude_leveling_dynamic_pass: bool
    d6_safe_key_search_dynamic_pass: bool


GATE_ORDER = (
    ("a2_nominal_insertion_dynamic_pass", "A2_NOMINAL_INSERTION_NOT_DYNAMIC"),
    ("d1_safe_standoff_dynamic_pass", "D1_SAFE_STANDOFF_NOT_DYNAMIC"),
    ("d2_visual_prealign_dynamic_pass", "D2_VISUAL_PREALIGN_NOT_DYNAMIC"),
    ("d3_light_contact_dynamic_pass", "D3_LIGHT_CONTACT_NOT_DYNAMIC"),
    ("d4_lateral_centering_dynamic_pass", "D4_CENTERING_NOT_DYNAMIC"),
    ("d5_attitude_leveling_dynamic_pass", "D5_LEVELING_NOT_DYNAMIC"),
    ("d6_safe_key_search_dynamic_pass", "D6_KEY_SEARCH_NOT_DYNAMIC"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "REJECTED_SAFE_STOP",
        "rejection_code": None,
        "twist_candidate_task": [0.0] * 6,
        "next_state": None,
        "contact_class": None,
        "stop_motion": True,
        "request_reobserve": False,
        "state_machine_evaluated": False,
        "motion_command_emitted": False,
        "control_authorized": False,
        "dynamic_compliant_insertion_pass_claimed": False,
    }


def evaluate_compliant_insertion_gate(
    readiness: CompliantInsertionReadiness,
    configuration: Mapping[str, Any] | None = None,
    controller_state: ControllerState | None = None,
    observation: InsertionObservation | None = None,
) -> dict[str, Any]:
    """Evaluate one pure policy step after all dynamic dependencies pass."""

    base = _base()
    if not isinstance(readiness, CompliantInsertionReadiness):
        return {**base, "rejection_code": "INVALID_READINESS_SNAPSHOT"}
    if any(type(getattr(readiness, item.name)) is not bool for item in fields(readiness)):
        return {**base, "rejection_code": "INVALID_READINESS_SNAPSHOT"}
    for field_name, code in GATE_ORDER:
        if not getattr(readiness, field_name):
            return {**base, "rejection_code": code}
    if (
        not isinstance(configuration, Mapping)
        or not isinstance(controller_state, ControllerState)
        or not isinstance(observation, InsertionObservation)
    ):
        return {**base, "rejection_code": "STATE_MACHINE_INPUT_MISSING"}

    command = step_compliant_insertion(
        configuration,
        controller_state,
        observation,
    )
    twist = [float(value) for value in command.twist_assembly]
    safe_abort = command.next_state.phase is InsertionState.SAFE_ABORT
    return {
        **base,
        "status": (
            "OFFLINE_SAFE_ABORT_CANDIDATE"
            if safe_abort
            else "OFFLINE_STATE_MACHINE_STEP_CANDIDATE"
        ),
        "rejection_code": (
            command.next_state.abort_reason
            if safe_abort
            else "DIAGNOSTIC_ONLY_NOT_MOTION_AUTHORITY"
        ),
        "twist_candidate_task": twist,
        "next_state": command.next_state.phase.value,
        "contact_class": command.contact_class.value,
        "controller_status": command.status,
        "stop_motion": bool(command.stop_motion or safe_abort),
        "request_reobserve": bool(command.request_reobserve),
        "state_machine_evaluated": True,
    }


def build_compliant_insertion_gate_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources: list[dict[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen D7 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen D7 source hash mismatch: {relative}")
        sources.append({"path": relative, "sha256": actual})

    master = yaml.safe_load(
        (root / "src/kcg_connector/config/d38999_master_model_contract_v1.yaml")
        .read_text(encoding="utf-8")
    )
    config = load_compliant_insertion_config(
        root / "src/kcg_connector/config/d38999_compliant_insertion_v2.yaml"
    )
    a1 = json.loads(
        (root / "artifacts/agent_control/tasks/EIGHT-HOUR-A1-INIT-DATUM/"
         "TASK_RESULT.json").read_text(encoding="utf-8")
    )
    dynamic_claim_fields = (
        ("EIGHT-HOUR-D1-RECEPTACLE-SAFE-STANDOFF", "dynamic_standoff_pass_claimed"),
        ("EIGHT-HOUR-D2-VISUAL-PREALIGN", "dynamic_prealign_pass_claimed"),
        ("EIGHT-HOUR-D3-LIGHT-FACE-CONTACT", "dynamic_light_contact_pass_claimed"),
        ("EIGHT-HOUR-D4-LATERAL-CENTERING", "dynamic_centering_pass_claimed"),
        ("EIGHT-HOUR-D5-ATTITUDE-LEVELING", "dynamic_attitude_leveling_pass_claimed"),
        ("EIGHT-HOUR-D6-SAFE-KEY-SEARCH", "dynamic_key_search_pass_claimed"),
    )
    claims: dict[str, bool] = {}
    for task_id, field_name in dynamic_claim_fields:
        task = json.loads(
            (root / f"artifacts/agent_control/tasks/{task_id}/TASK_RESULT.json")
            .read_text(encoding="utf-8")
        )
        value = task.get(field_name)
        if value is not False:
            raise ValueError(f"current D7 upstream dynamic claim changed: {task_id}")
        claims[field_name] = value
    limits = master["acceptance_limits"]
    motion = config["motion"]
    safety = config["safety"]
    boundaries = config["boundaries"]
    if (
        limits["force_component_limit_n_per_driven_body"]
        != FORMAL_FORCE_COMPONENT_LIMIT_N
        or limits["torque_component_limit_nm"]
        != FORMAL_MOMENT_COMPONENT_LIMIT_NM
        or a1["formal_a2_run_count"] != 0
        or a1["formal_acceptance_claimed"] is not False
        or config["control_rate_hz"] != 240
        or motion["axial_speed_m_s"] != 0.00020
        or motion["maximum_total_travel_m"] != 0.024
        or motion["maximum_retries"] != 2
        or safety["hard_axial_force_n"] != 5.0
        or safety["hard_lateral_force_n"] != 2.0
        or safety["hard_bending_moment_nm"] != 0.18
        or safety["hard_torsional_moment_nm"] != 0.05
        or boundaries["insertion_success_claimed"] is not False
    ):
        raise ValueError("authoritative D7 compliant insertion contract changed")

    current_readiness = CompliantInsertionReadiness(
        a2_nominal_insertion_dynamic_pass=False,
        d1_safe_standoff_dynamic_pass=False,
        d2_visual_prealign_dynamic_pass=False,
        d3_light_contact_dynamic_pass=False,
        d4_lateral_centering_dynamic_pass=False,
        d5_attitude_leveling_dynamic_pass=False,
        d6_safe_key_search_dynamic_pass=False,
    )
    current = evaluate_compliant_insertion_gate(current_readiness)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_COMPLIANT_INSERTION_GATE_READY",
        "classification": "UPSTREAM_GATED_EXISTING_COMPLIANT_STATE_MACHINE",
        "gate_order": [name for name, _ in GATE_ORDER],
        "current_dynamic_claims": claims,
        "current_readiness": current,
        "control_rate_hz": config["control_rate_hz"],
        "axial_speed_m_s": motion["axial_speed_m_s"],
        "maximum_total_travel_m": motion["maximum_total_travel_m"],
        "maximum_retries": motion["maximum_retries"],
        "formal_force_component_limit_n_per_driven_body": (
            FORMAL_FORCE_COMPONENT_LIMIT_N
        ),
        "formal_moment_component_limit_nm": FORMAL_MOMENT_COMPONENT_LIMIT_NM,
        "experimental_abort_envelope": {
            "axial_force_n": safety["hard_axial_force_n"],
            "lateral_force_n": safety["hard_lateral_force_n"],
            "bending_moment_nm": safety["hard_bending_moment_nm"],
            "torsional_moment_nm": safety["hard_torsional_moment_nm"],
        },
        "forbidden_inputs": list(boundaries["forbidden_inputs"]),
        "simulation_started": False,
        "dynamic_compliant_insertion_pass_claimed": False,
        "control_authorized": False,
        "hardware_authorized": False,
        "sources": sources,
    }


__all__ = [
    "CompliantInsertionReadiness",
    "FORMAL_FORCE_COMPONENT_LIMIT_N",
    "FORMAL_MOMENT_COMPONENT_LIMIT_NM",
    "FROZEN_SOURCES",
    "GATE_ORDER",
    "SCHEMA_VERSION",
    "build_compliant_insertion_gate_contract",
    "evaluate_compliant_insertion_gate",
]
