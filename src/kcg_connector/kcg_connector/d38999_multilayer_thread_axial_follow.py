"""Non-executing D38999 thread rotation/translation relation for E5.

The task layer provides only a segmented nut-progress schedule.  This module
derives the master-contract axial relation for a physical constraint; it never
writes an object pose and has no contact or assembly-event truth inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .d38999_multilayer_segmented_twist import (
    derive_segmented_twist_schedule,
    load_segmented_twist_contract,
)
from .geometry import helical_travel


TASK_ID = "EIGHT-HOUR-E5-THREAD-AXIAL-FOLLOW"
E4_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-E4-SEGMENTED-TWIST/TASK_RESULT.json"
)
MASTER_LEAD_M_PER_REVOLUTION = 0.00762
EXPECTED_STROKE_PROGRESS_RAD = 2.0 * math.pi / 3.0
EXPECTED_STROKE_ADVANCE_M = 0.00254

FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/kcg_connector/geometry.py": (
        "b90698a641a0390a9b7914f0a94e7f1f241c6892c35f844d7f587f8a65b0ee00"
    ),
    "src/kcg_connector/kcg_connector/d38999_multilayer_segmented_twist.py": (
        "bf2068de577d51144ab9561a2c7a4aaea5e425c310289363c0da897dc20e6ee1"
    ),
    E4_RESULT_PATH: (
        "1086e785dbdbea23e26a537f02ad98ac3ce2815bace2075e1f56310e5a8ad052"
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


@dataclass(frozen=True)
class ThreadAxialFollowContract:
    source_rows: tuple[tuple[str, str], ...]
    lead_m_per_revolution: float
    stroke_progress_rad: float
    stroke_advance_m: float
    stroke_count: int
    rewind_count: int
    relation: str
    current_e4_outcome: str
    current_e4_dynamic_segmented_twist_passed: bool
    current_e4_evidence_sha256: str


@dataclass(frozen=True)
class ThreadAxialFollowReadiness:
    e4_evidence_path: str
    e4_evidence_sha256: str
    e4_dynamic_segmented_twist_passed: bool
    physical_constraint_runtime_ready: bool


def load_thread_axial_follow_contract(
    repository_root: str | Path,
) -> ThreadAxialFollowContract:
    """Bind the master lead, existing E4 schedule, and current E4 evidence."""

    root = Path(repository_root).resolve()
    rows: list[tuple[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"frozen E5 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen E5 source hash mismatch: {relative}")
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
    thread = _mapping(master.get("thread"), "master.thread")
    control = _mapping(
        thread.get("control_representation"),
        "master.thread.control_representation",
    )
    lead = _finite(thread.get("lead_mm_per_revolution"), "thread lead") / 1000.0
    relation = str(control.get("positive_insertion_relation"))
    if (
        not math.isclose(
            lead,
            MASTER_LEAD_M_PER_REVOLUTION,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or relation
        != "axial_advance_m=-nut_rotation_rad*0.00762/(2*pi)"
        or control.get("runtime_engagement_switch_allowed") is not False
        or control.get("software_axial_pose_write_allowed") is not False
    ):
        raise ValueError("master thread axial relation changed")

    e4_contract = load_segmented_twist_contract(root)
    if (
        e4_contract.stroke_count != 3
        or e4_contract.rewind_count != 2
        or not math.isclose(
            e4_contract.master_lead_m_per_revolution,
            lead,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("E4 schedule and E5 master lead differ")

    e4 = _mapping(
        json.loads((root / E4_RESULT_PATH).read_text(encoding="utf-8")),
        "E4 task result",
    )
    if (
        e4.get("task_id") != "EIGHT-HOUR-E4-SEGMENTED-TWIST"
        or e4.get("outcome") != "OFFLINE_PASS"
        or type(e4.get("dynamic_segmented_twist_pass_claimed")) is not bool
        or _finite(e4.get("master_lead_m_per_revolution"), "E4 lead")
        != lead
        or e4.get("legacy_proxy_lead_used") is not False
    ):
        raise ValueError("E4 evidence does not support E5")

    stroke_advance = helical_travel(EXPECTED_STROKE_PROGRESS_RAD, lead)
    if not math.isclose(
        stroke_advance,
        EXPECTED_STROKE_ADVANCE_M,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("one 120-degree stroke does not equal 2.54 mm")

    return ThreadAxialFollowContract(
        source_rows=tuple(rows),
        lead_m_per_revolution=lead,
        stroke_progress_rad=EXPECTED_STROKE_PROGRESS_RAD,
        stroke_advance_m=stroke_advance,
        stroke_count=e4_contract.stroke_count,
        rewind_count=e4_contract.rewind_count,
        relation=relation,
        current_e4_outcome=str(e4.get("outcome")),
        current_e4_dynamic_segmented_twist_passed=e4[
            "dynamic_segmented_twist_pass_claimed"
        ],
        current_e4_evidence_sha256=FROZEN_SOURCES[E4_RESULT_PATH],
    )


def derive_thread_axial_follow(
    contract: ThreadAxialFollowContract,
    segmented_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Map exact nut progress to constraint-space axial displacement."""

    schedule = _mapping(segmented_schedule, "segmented schedule")
    stages = schedule.get("stages")
    if (
        schedule.get("schedule_only") is not True
        or schedule.get("stroke_count") != contract.stroke_count
        or schedule.get("rewind_count") != contract.rewind_count
        or not isinstance(stages, list)
        or len(stages) != 10
        or schedule.get("legacy_proxy_lead_used") is not False
    ):
        raise ValueError("segmented schedule boundary changed")

    expected_actions = (
        "GRIP", "TWIST", "RELEASE", "REWIND", "REGRIP",
        "TWIST", "RELEASE", "REWIND", "REGRIP", "TWIST",
    )
    output: list[dict[str, Any]] = []
    cumulative_coordinate_delta = 0.0
    stroke_count = 0
    rewind_count = 0
    for index, (raw, expected_action) in enumerate(zip(stages, expected_actions)):
        stage = _mapping(raw, f"stages[{index}]")
        action = stage.get("action")
        progress = _finite(stage.get("nut_progress_rad"), f"stages[{index}].nut_progress_rad")
        if action != expected_action:
            raise ValueError("segmented action sequence changed")
        if action == "TWIST":
            if not math.isclose(
                progress,
                contract.stroke_progress_rad,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("twist progress must be exactly 120 degrees")
            insertion_advance = helical_travel(
                progress, contract.lead_m_per_revolution
            )
            coordinate_delta = -insertion_advance
            stroke_count += 1
        else:
            if progress != 0.0:
                raise ValueError("non-twist stage must not advance the nut")
            insertion_advance = 0.0
            coordinate_delta = 0.0
            if action == "REWIND":
                rewind_count += 1
        cumulative_coordinate_delta += coordinate_delta
        output.append(
            {
                "index": index,
                "action": action,
                "nut_progress_rad": progress,
                "insertion_advance_m": insertion_advance,
                "axial_coordinate_delta_m": coordinate_delta,
                "cumulative_axial_coordinate_delta_m": (
                    cumulative_coordinate_delta
                ),
            }
        )

    if stroke_count != 3 or rewind_count != 2:
        raise ValueError("E5 requires exactly three strokes and two rewinds")
    total_advance = -cumulative_coordinate_delta
    if not math.isclose(
        total_advance,
        contract.lead_m_per_revolution,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("one revolution must advance exactly one master lead")

    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "relation_only": True,
        "relation": contract.relation,
        "stages": output,
        "total_insertion_advance_m": total_advance,
        "total_axial_coordinate_delta_m": cumulative_coordinate_delta,
        "legacy_0p003_m_lead_used": False,
        "legacy_0p004_m_lead_used": False,
        "physical_model_internal_constraint_only": True,
        "software_pose_write_requested": False,
        "force_command_requested": False,
        "robot_commands_emitted": 0,
        "control_authorized": False,
        "dynamic_thread_follow_pass_claimed": False,
    }


def evaluate_thread_axial_follow_gate(
    contract: ThreadAxialFollowContract,
    readiness: ThreadAxialFollowReadiness,
) -> str | None:
    if (
        readiness.e4_evidence_path != E4_RESULT_PATH
        or readiness.e4_evidence_sha256
        != contract.current_e4_evidence_sha256
    ):
        return "E4_EVIDENCE_ID_MISMATCH"
    if (
        contract.current_e4_dynamic_segmented_twist_passed is not True
        or readiness.e4_dynamic_segmented_twist_passed is not True
    ):
        return "E4_SEGMENTED_TWIST_NOT_DYNAMIC"
    if readiness.physical_constraint_runtime_ready is not True:
        return "PHYSICAL_THREAD_CONSTRAINT_RUNTIME_NOT_READY"
    return None


def build_thread_axial_follow_request(
    contract: ThreadAxialFollowContract,
    readiness: ThreadAxialFollowReadiness,
    *,
    initial_q7_rad: float,
) -> dict[str, Any]:
    """Return a constraint request only after all evidence gates pass."""

    rejection = evaluate_thread_axial_follow_gate(contract, readiness)
    if rejection is not None:
        return {
            "schema_version": 1,
            "task_id": TASK_ID,
            "request_ready": False,
            "rejection_code": rejection,
            "relation": None,
            "software_pose_write_requested": False,
            "force_command_requested": False,
            "robot_commands_emitted": 0,
            "control_authorized": False,
            "dynamic_thread_follow_pass_claimed": False,
        }
    e4_contract = load_segmented_twist_contract(
        Path(__file__).resolve().parents[3]
    )
    schedule = derive_segmented_twist_schedule(
        e4_contract, initial_q7_rad=initial_q7_rad
    )
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "request_ready": True,
        "rejection_code": None,
        "relation": derive_thread_axial_follow(contract, schedule),
        "software_pose_write_requested": False,
        "force_command_requested": False,
        "robot_commands_emitted": 0,
        "control_authorized": False,
        "dynamic_thread_follow_pass_claimed": False,
    }


def current_readiness(contract: ThreadAxialFollowContract) -> ThreadAxialFollowReadiness:
    return ThreadAxialFollowReadiness(
        e4_evidence_path=E4_RESULT_PATH,
        e4_evidence_sha256=contract.current_e4_evidence_sha256,
        e4_dynamic_segmented_twist_passed=(
            contract.current_e4_dynamic_segmented_twist_passed
        ),
        physical_constraint_runtime_ready=False,
    )


__all__ = [
    "E4_RESULT_PATH",
    "ThreadAxialFollowContract",
    "ThreadAxialFollowReadiness",
    "build_thread_axial_follow_request",
    "current_readiness",
    "derive_thread_axial_follow",
    "evaluate_thread_axial_follow_gate",
    "load_thread_axial_follow_contract",
]
