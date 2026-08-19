"""Evidence-gated segmented coupling-nut twist for the multilayer model.

This module plans no robot motion.  It binds the existing three-stroke q7
schedule to the current master-model lead and fails closed on the parked E3
regrasp evidence or an unsafe wrist-moment guard.  Object pose, contact
identity, contact normal, and assembly-event truth are intentionally absent
from the public inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .d38999_full_rotation import (
    EXPECTED_REWIND_COUNT,
    EXPECTED_STROKE_COUNT,
    STROKE_ANGLE_RAD,
    build_d38999_full_rotation_plan,
)
from .trajectory import Q7Action, load_q7_twist_config


TASK_ID = "EIGHT-HOUR-E4-SEGMENTED-TWIST"
E3_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-E3-REGRASP-NUT/TASK_RESULT.json"
)
B5_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-B5-WRIST-MOMENT-MONITOR/TASK_RESULT.json"
)
AUTHORIZED_MOMENT_COMPONENT_LIMIT_NM = 0.30
MASTER_LEAD_M_PER_REVOLUTION = 0.00762
LEGACY_PROXY_LEAD_M_PER_REVOLUTION = 0.004

FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/config/connector_task.yaml": (
        "4e8bee1abcda805fc0669cffbf34c095e509b3ecc96c0571e618e00d9940d36e"
    ),
    "src/kcg_connector/kcg_connector/d38999_full_rotation.py": (
        "e163a483ffbcbf9eeebeed4d792534d7048e6df77eb52eb89753300a8c2eff9c"
    ),
    "src/kcg_connector/kcg_connector/trajectory.py": (
        "ef0479941658e42f864ab8223cf0dd59794cf81716fe495325f70a4306838441"
    ),
    E3_RESULT_PATH: (
        "f513ad1aafd78ba55dc125ed07260d20df287f22b6af228acdf35c5b60e8c397"
    ),
    B5_RESULT_PATH: (
        "2bebe773c145d4afec89cdf1865ae97eb13db8bf9019d2006bb95ba635c38e0f"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


@dataclass(frozen=True)
class SegmentedTwistContract:
    source_rows: tuple[tuple[str, str], ...]
    q7_lower_limit_rad: float
    q7_upper_limit_rad: float
    tightening_direction: int
    stroke_angle_rad: float
    stroke_count: int
    rewind_count: int
    master_lead_m_per_revolution: float
    legacy_proxy_lead_m_per_revolution: float
    moment_component_limit_nm: float
    current_e3_outcome: str
    current_e3_dynamic_nut_regrasp_passed: bool
    current_e3_evidence_sha256: str


@dataclass(frozen=True)
class SegmentedTwistReadiness:
    e3_evidence_path: str
    e3_evidence_sha256: str
    e3_dynamic_nut_regrasp_passed: bool
    wrist_guard_safe_to_continue: bool
    wrist_fault_latched: bool


def load_segmented_twist_contract(
    repository_root: str | Path,
) -> SegmentedTwistContract:
    """Load the exact current sources and reject any content drift."""

    root = Path(repository_root).resolve()
    rows: list[tuple[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"frozen E4 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen E4 source hash mismatch: {relative}")
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
    task = _mapping(
        yaml.safe_load(
            (root / "src/kcg_connector/config/connector_task.yaml").read_text(
                encoding="utf-8"
            )
        ),
        "connector task",
    )
    e3 = _mapping(
        json.loads((root / E3_RESULT_PATH).read_text(encoding="utf-8")),
        "E3 task result",
    )
    b5 = _mapping(
        json.loads((root / B5_RESULT_PATH).read_text(encoding="utf-8")),
        "B5 task result",
    )

    thread = _mapping(master.get("thread"), "master.thread")
    control = _mapping(
        thread.get("control_representation"),
        "master.thread.control_representation",
    )
    lead_m = float(thread.get("lead_mm_per_revolution")) / 1000.0
    if not math.isclose(
        lead_m, MASTER_LEAD_M_PER_REVOLUTION, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("master thread lead changed")
    if (
        control.get("type")
        != "explicit_three_start_helical_rotation_translation_constraint"
        or control.get("runtime_engagement_switch_allowed") is not False
        or control.get("software_axial_pose_write_allowed") is not False
    ):
        raise ValueError("master thread control boundary changed")

    q7 = load_q7_twist_config(
        root / "src/kcg_connector/config/connector_task.yaml"
    )
    if (
        q7.tightening_direction != -1
        or not math.isclose(
            q7.maximum_segment_angle_rad,
            STROKE_ANGLE_RAD,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or EXPECTED_STROKE_COUNT != 3
        or EXPECTED_REWIND_COUNT != 2
    ):
        raise ValueError("existing segmented twist schedule changed")

    legacy_lead = float(
        _mapping(task.get("success"), "connector_task.success").get(
            "helical_lead_per_revolution"
        )
    )
    if not math.isclose(
        legacy_lead,
        LEGACY_PROXY_LEAD_M_PER_REVOLUTION,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("legacy proxy lead changed")
    if math.isclose(legacy_lead, lead_m, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("legacy proxy lead must not masquerade as master lead")

    master_limit = float(
        _mapping(
            master.get("acceptance_limits"), "master.acceptance_limits"
        ).get("torque_component_limit_nm")
    )
    b5_limit = float(b5.get("moment_limit_nm"))
    if (
        master_limit != AUTHORIZED_MOMENT_COMPONENT_LIMIT_NM
        or b5_limit != AUTHORIZED_MOMENT_COMPONENT_LIMIT_NM
        or b5.get("status") != "OFFLINE_PASS"
        or b5.get("dynamic_grasp_pass_claimed") is not False
    ):
        raise ValueError("0.30 N*m wrist-moment boundary changed")
    if e3.get("task_id") != "EIGHT-HOUR-E3-REGRASP-NUT":
        raise ValueError("wrong E3 evidence")
    if type(e3.get("dynamic_nut_regrasp_pass_claimed")) is not bool:
        raise ValueError("E3 dynamic result must be explicit bool")

    return SegmentedTwistContract(
        source_rows=tuple(rows),
        q7_lower_limit_rad=q7.safe_lower_rad,
        q7_upper_limit_rad=q7.safe_upper_rad,
        tightening_direction=q7.tightening_direction,
        stroke_angle_rad=STROKE_ANGLE_RAD,
        stroke_count=EXPECTED_STROKE_COUNT,
        rewind_count=EXPECTED_REWIND_COUNT,
        master_lead_m_per_revolution=lead_m,
        legacy_proxy_lead_m_per_revolution=legacy_lead,
        moment_component_limit_nm=master_limit,
        current_e3_outcome=str(e3.get("outcome")),
        current_e3_dynamic_nut_regrasp_passed=e3[
            "dynamic_nut_regrasp_pass_claimed"
        ],
        current_e3_evidence_sha256=FROZEN_SOURCES[E3_RESULT_PATH],
    )


def derive_segmented_twist_schedule(
    contract: SegmentedTwistContract,
    *,
    initial_q7_rad: float,
) -> dict[str, Any]:
    """Derive a non-executing 3x120-degree schedule from existing code."""

    plan = build_d38999_full_rotation_plan(
        initial_q7_rad=initial_q7_rad,
        q7_lower_limit_rad=contract.q7_lower_limit_rad,
        q7_upper_limit_rad=contract.q7_upper_limit_rad,
        lead_m_per_revolution=contract.master_lead_m_per_revolution,
    )
    stages: list[dict[str, Any]] = []
    for index, segment in enumerate(plan.segments):
        nut_progress = (
            -segment.connector_angle_delta
            if segment.action is Q7Action.TWIST
            else 0.0
        )
        stages.append(
            {
                "index": index,
                "action": segment.action.value,
                "q7_start_rad": segment.q7_start,
                "q7_end_rad": segment.q7_end,
                "q7_delta_rad": segment.q7_end - segment.q7_start,
                "nut_progress_rad": nut_progress,
            }
        )
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "schedule_only": True,
        "stages": stages,
        "stroke_count": plan.stroke_count,
        "rewind_count": plan.rewind_count,
        "target_nut_progress_rad": plan.target_nut_progress_rad,
        "master_expected_axial_travel_m": plan.expected_axial_travel_m,
        "legacy_proxy_lead_used": False,
        "e5_axial_follow_required": True,
        "robot_commands_emitted": 0,
        "control_authorized": False,
        "dynamic_segmented_twist_pass_claimed": False,
    }


def evaluate_segmented_twist_gate(
    contract: SegmentedTwistContract,
    readiness: SegmentedTwistReadiness,
) -> str | None:
    """Return the first failure code, or ``None`` for a schedule request."""

    if (
        readiness.e3_evidence_path != E3_RESULT_PATH
        or readiness.e3_evidence_sha256
        != contract.current_e3_evidence_sha256
    ):
        return "E3_EVIDENCE_ID_MISMATCH"
    if (
        contract.current_e3_dynamic_nut_regrasp_passed is not True
        or readiness.e3_dynamic_nut_regrasp_passed is not True
    ):
        return "E3_NUT_REGRASP_NOT_DYNAMIC"
    if readiness.wrist_fault_latched is True:
        return "WRIST_MOMENT_FAULT_LATCHED"
    if readiness.wrist_guard_safe_to_continue is not True:
        return "WRIST_MOMENT_GUARD_UNSAFE"
    return None


def build_segmented_twist_request(
    contract: SegmentedTwistContract,
    readiness: SegmentedTwistReadiness,
    *,
    initial_q7_rad: float,
) -> dict[str, Any]:
    """Build a plan request only; this API can never issue robot commands."""

    rejection = evaluate_segmented_twist_gate(contract, readiness)
    if rejection is not None:
        return {
            "schema_version": 1,
            "task_id": TASK_ID,
            "request_ready": False,
            "rejection_code": rejection,
            "schedule": None,
            "robot_commands_emitted": 0,
            "control_authorized": False,
            "dynamic_segmented_twist_pass_claimed": False,
        }
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "request_ready": True,
        "rejection_code": None,
        "schedule": derive_segmented_twist_schedule(
            contract, initial_q7_rad=initial_q7_rad
        ),
        "robot_commands_emitted": 0,
        "control_authorized": False,
        "dynamic_segmented_twist_pass_claimed": False,
    }


def current_readiness(contract: SegmentedTwistContract) -> SegmentedTwistReadiness:
    """Return the exact current, intentionally rejected upstream state."""

    return SegmentedTwistReadiness(
        e3_evidence_path=E3_RESULT_PATH,
        e3_evidence_sha256=contract.current_e3_evidence_sha256,
        e3_dynamic_nut_regrasp_passed=(
            contract.current_e3_dynamic_nut_regrasp_passed
        ),
        wrist_guard_safe_to_continue=False,
        wrist_fault_latched=False,
    )


__all__ = [
    "AUTHORIZED_MOMENT_COMPONENT_LIMIT_NM",
    "E3_RESULT_PATH",
    "SegmentedTwistContract",
    "SegmentedTwistReadiness",
    "build_segmented_twist_request",
    "current_readiness",
    "derive_segmented_twist_schedule",
    "evaluate_segmented_twist_gate",
    "load_segmented_twist_contract",
]
