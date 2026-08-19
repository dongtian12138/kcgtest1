"""Evidence-gated, non-executing post-insertion hold interface."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_postinsert_hold_v1"
FORMAL_MOMENT_COMPONENT_LIMIT_NM = 0.30
TIMING_COMPARISON_EPSILON_S = 1.0e-12
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-D7-COMPLIANT-INSERTION/"
    "TASK_RESULT.json": (
        "da156677861d69dfa2db8a292b4da6b9040cfe7e81a98232ab835970d6e38327"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-B5-WRIST-MOMENT-MONITOR/"
    "TASK_RESULT.json": (
        "2bebe773c145d4afec89cdf1865ae97eb13db8bf9019d2006bb95ba635c38e0f"
    ),
    "src/kcg_connector/kcg_connector/wrist_moment_safety_guard.py": (
        "779f7601a69f31c87ba44ad88584f540c2178c13b8c2bc09f5bde69385df0db8"
    ),
    "src/kcg_connector/config/d38999_compliant_insertion_v2.yaml": (
        "0f16e9b2fc5d615a4e8035dfa21c4ec9a18a341b4320ecbc3e66b874be489703"
    ),
    "src/kcg_connector/config/wrist_ft_v1_contract.yaml": (
        "5ce5609ec782ee9a1cf2598c26df05c2785e71aa7dc459568c1ba28a72efc110"
    ),
}


@dataclass(frozen=True)
class PostInsertHoldReadiness:
    """Explicit evidence gates; no object/contact/event truth is accepted."""

    formal_insertion_dynamic_pass: bool
    body_grasp_dynamic_pass: bool
    wrist_guard_safe: bool
    wrist_guard_fault_latched: bool
    insertion_evidence_id: str | None
    grasp_evidence_id: str | None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base(rejection_code: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "REJECTED_SAFE_HOLD",
        "rejection_code": rejection_code,
        "hold_request_candidate": None,
        "hold_complete_candidate": False,
        "stop_motion": True,
        "motion_command_emitted": False,
        "gripper_command_emitted": False,
        "control_authorized": False,
        "dynamic_postinsert_hold_pass_claimed": False,
        "hardware_authorized": False,
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _timing_authority(value: Any) -> tuple[float, dict[str, Any]] | None:
    if not isinstance(value, Mapping):
        return None
    source_path = value.get("source_path")
    source_sha256 = value.get("source_sha256")
    duration = _finite_number(value.get("hold_duration_s"))
    if (
        not isinstance(source_path, str)
        or not source_path.strip()
        or not isinstance(source_sha256, str)
        or SHA256_PATTERN.fullmatch(source_sha256) is None
        or duration is None
        or duration <= 0.0
    ):
        return None
    return duration, {
        "source_path": source_path,
        "source_sha256": source_sha256,
        "hold_duration_s": duration,
    }


def evaluate_postinsert_hold_gate(
    readiness: PostInsertHoldReadiness,
    *,
    timing_authority: Mapping[str, Any] | None,
    now_s: float,
    hold_started_s: float,
    wrench_timestamp_s: float,
    maximum_wrench_sample_age_s: float,
) -> dict[str, Any]:
    """Evaluate a hold candidate without emitting a robot or gripper command."""

    base = _base()
    if not isinstance(readiness, PostInsertHoldReadiness) or any(
        type(getattr(readiness, item.name)) is not bool
        for item in fields(readiness)[:4]
    ):
        return {**base, "rejection_code": "INVALID_READINESS_SNAPSHOT"}
    if not readiness.formal_insertion_dynamic_pass:
        return {**base, "rejection_code": "D7_INSERTION_NOT_DYNAMIC"}
    if not readiness.body_grasp_dynamic_pass:
        return {**base, "rejection_code": "B4_BODY_GRASP_NOT_DYNAMIC"}
    if (
        not isinstance(readiness.insertion_evidence_id, str)
        or not readiness.insertion_evidence_id.strip()
        or not isinstance(readiness.grasp_evidence_id, str)
        or not readiness.grasp_evidence_id.strip()
    ):
        return {**base, "rejection_code": "DYNAMIC_EVIDENCE_ID_MISSING"}
    timing = _timing_authority(timing_authority)
    if timing is None:
        return {**base, "rejection_code": "HOLD_DURATION_AUTHORITY_MISSING"}
    hold_duration_s, timing_row = timing
    now = _finite_number(now_s)
    started = _finite_number(hold_started_s)
    wrench_time = _finite_number(wrench_timestamp_s)
    maximum_age = _finite_number(maximum_wrench_sample_age_s)
    if (
        now is None
        or started is None
        or wrench_time is None
        or maximum_age is None
        or maximum_age <= 0.0
        or started > now
        or wrench_time > now
    ):
        return {**base, "rejection_code": "INVALID_HOLD_TIMING"}
    sample_age_s = now - wrench_time
    if sample_age_s > maximum_age + TIMING_COMPARISON_EPSILON_S:
        return {**base, "rejection_code": "WRIST_WRENCH_STALE"}
    if not readiness.wrist_guard_safe or readiness.wrist_guard_fault_latched:
        return {**base, "rejection_code": "WRIST_MOMENT_GUARD_REJECTED"}

    elapsed_s = now - started
    complete = elapsed_s >= hold_duration_s
    return {
        **base,
        "status": (
            "OFFLINE_HOLD_COMPLETE_CANDIDATE"
            if complete
            else "OFFLINE_HOLD_REQUEST_CANDIDATE"
        ),
        "rejection_code": "DIAGNOSTIC_ONLY_NOT_MOTION_AUTHORITY",
        "hold_request_candidate": {
            "cartesian_twist_task": [0.0] * 6,
            "preserve_current_gripper_setpoint": True,
            "elapsed_s": elapsed_s,
            "required_duration_s": hold_duration_s,
            "remaining_s": max(0.0, hold_duration_s - elapsed_s),
            "timing_authority": timing_row,
        },
        "hold_complete_candidate": complete,
        "sample_age_s": sample_age_s,
    }


def build_postinsert_hold_contract(repository_root: str | Path) -> dict[str, Any]:
    """Verify upstream evidence and report current fail-closed readiness."""

    root = Path(repository_root).resolve()
    sources: list[dict[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen E1 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen E1 source hash mismatch: {relative}")
        sources.append({"path": relative, "sha256": actual})
    d7 = json.loads(
        (root / "artifacts/agent_control/tasks/EIGHT-HOUR-D7-COMPLIANT-"
         "INSERTION/TASK_RESULT.json").read_text(encoding="utf-8")
    )
    b5 = json.loads(
        (root / "artifacts/agent_control/tasks/EIGHT-HOUR-B5-WRIST-MOMENT-"
         "MONITOR/TASK_RESULT.json").read_text(encoding="utf-8")
    )
    insertion = yaml.safe_load(
        (root / "src/kcg_connector/config/d38999_compliant_insertion_v2.yaml")
        .read_text(encoding="utf-8")
    )
    wrist_ft = yaml.safe_load(
        (root / "src/kcg_connector/config/wrist_ft_v1_contract.yaml")
        .read_text(encoding="utf-8")
    )
    maximum_age_s = insertion["safety"]["maximum_sample_age_s"]
    if (
        d7.get("outcome") != "OFFLINE_PASS"
        or d7.get("dynamic_compliant_insertion_pass_claimed") is not False
        or b5.get("status") != "OFFLINE_PASS"
        or b5.get("moment_limit_nm") != FORMAL_MOMENT_COMPONENT_LIMIT_NM
        or maximum_age_s != 0.020
        or "stable_hold" not in wrist_ft["success_requires_all"]
        or any(value is not None for value in wrist_ft["safety_limits"].values())
    ):
        raise ValueError("authoritative E1 upstream evidence changed")
    current = PostInsertHoldReadiness(
        formal_insertion_dynamic_pass=False,
        body_grasp_dynamic_pass=False,
        wrist_guard_safe=False,
        wrist_guard_fault_latched=True,
        insertion_evidence_id=None,
        grasp_evidence_id=None,
    )
    current_decision = evaluate_postinsert_hold_gate(
        current,
        timing_authority=None,
        now_s=0.0,
        hold_started_s=0.0,
        wrench_timestamp_s=0.0,
        maximum_wrench_sample_age_s=maximum_age_s,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_POSTINSERT_HOLD_GATE_READY",
        "current_decision": current_decision,
        "maximum_wrench_sample_age_s": maximum_age_s,
        "formal_moment_component_limit_nm": FORMAL_MOMENT_COMPONENT_LIMIT_NM,
        "authoritative_hold_duration_s": None,
        "hold_duration_authority_required": True,
        "simulation_started": False,
        "dynamic_postinsert_hold_pass_claimed": False,
        "control_authorized": False,
        "hardware_authorized": False,
        "sources": sources,
    }


__all__ = [
    "FORMAL_MOMENT_COMPONENT_LIMIT_NM",
    "FROZEN_SOURCES",
    "PostInsertHoldReadiness",
    "SCHEMA_VERSION",
    "TIMING_COMPARISON_EPSILON_S",
    "build_postinsert_hold_contract",
    "evaluate_postinsert_hold_gate",
]
