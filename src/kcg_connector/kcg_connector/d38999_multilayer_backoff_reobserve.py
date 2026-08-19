"""Static recovery plan from a D8 latch to precommitted C3 view requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "kcg_d38999_multilayer_backoff_reobserve_v1"
FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-C3-POSTGRASP-VIEW-PLAN/"
    "TASK_RESULT.json": (
        "c55c2d12559653d0457cb717fb466d7dffaf07940b55c7b67e3f2c7dc0d61a7c"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C3-POSTGRASP-VIEW-PLAN/"
    "VIEW_PLAN_MANIFEST.json": (
        "64eecf1a5e1ac04fd129453b042507e25ab2fbfe170fc035e7c00bcaba23921e"
    ),
    "src/kcg_connector/kcg_connector/d38999_multilayer_postgrasp_view_plan.py": (
        "2fae77f41d3d52eb751e8368252e8335c10289741c4ce56e93fddc1bd7399924"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D8-OVERFORCE-EXIT/"
    "TASK_RESULT.json": (
        "fa07a1ca8842b2799fb26ef91446ae64347d168c477bebffb76147e6524c81f3"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_backoff_reobserve_plan(
    repository_root: str | Path,
    *,
    exit_latched: bool,
    exit_failure_reason: str | None,
) -> dict[str, Any]:
    """Build a non-executing plan; no runtime observation can alter it."""

    root = Path(repository_root).resolve()
    sources: list[dict[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen D9 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen D9 source hash mismatch: {relative}")
        sources.append({"path": relative, "sha256": actual})
    c3 = json.loads(
        (root / "artifacts/agent_control/tasks/EIGHT-HOUR-C3-POSTGRASP-VIEW-PLAN/"
         "TASK_RESULT.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "artifacts/agent_control/tasks/EIGHT-HOUR-C3-POSTGRASP-VIEW-PLAN/"
         "VIEW_PLAN_MANIFEST.json").read_text(encoding="utf-8")
    )
    d8 = json.loads(
        (root / "artifacts/agent_control/tasks/EIGHT-HOUR-D8-OVERFORCE-EXIT/"
         "TASK_RESULT.json").read_text(encoding="utf-8")
    )
    if (
        c3["candidate_sequence"] != ["V0", "V1", "V2"]
        or c3["current_multilayer_dynamic_views_proven"] != 0
        or c3["formal_capture_authorized"] is not False
        or manifest["camera_binding"]["selected_camera_for_dynamic_capture"] is not None
        or any(row["dynamic_execution_authorized"] for row in manifest["candidate_views"])
        or d8["outcome"] != "OFFLINE_PASS"
        or d8["backoff_distance_m"] != 0.0004
        or d8["backoff_speed_m_s"] != 0.0003
    ):
        raise ValueError("authoritative D9 recovery sources changed")
    base = {
        "schema_version": SCHEMA_VERSION,
        "exit_latched": exit_latched if type(exit_latched) is bool else False,
        "exit_failure_reason": exit_failure_reason,
        "stages": [],
        "selected_view_for_execution": None,
        "robot_motion_started": False,
        "render_capture_performed": False,
        "motion_command_emitted": False,
        "capture_command_emitted": False,
        "control_authorized": False,
        "dynamic_reobserve_pass_claimed": False,
        "hardware_authorized": False,
        "sources": sources,
    }
    if type(exit_latched) is not bool or not isinstance(
        exit_failure_reason, (str, type(None))
    ):
        return {**base, "status": "REJECTED", "rejection_code": "INVALID_RECOVERY_INPUT"}
    if not exit_latched:
        return {**base, "status": "NO_RECOVERY_REQUEST", "rejection_code": "D8_NOT_LATCHED"}
    if exit_failure_reason is None or not exit_failure_reason.strip():
        return {**base, "status": "REJECTED", "rejection_code": "LATCH_REASON_MISSING"}

    views = []
    for row in manifest["candidate_views"]:
        views.append(
            {
                "view_id": row["view_id"],
                "sequence_index": row["sequence_index"],
                "tcp_delta_xyz_rpy": list(row["tcp_delta_xyz_rpy"]),
                "known_legacy_seed0_gate": row["known_legacy_seed0_gate"],
                "dynamic_execution_authorized": False,
                "request_only": True,
            }
        )
    stages = [
        {"ordinal": 1, "name": "STOP_ZERO_TWIST", "twist_task": [0.0] * 6},
        {
            "ordinal": 2,
            "name": "RETRACT_REQUEST",
            "distance_m": d8["backoff_distance_m"],
            "speed_m_s": d8["backoff_speed_m_s"],
            "execution_authorized": False,
        },
        {
            "ordinal": 3,
            "name": "HOLD_FOR_SETTLE_REQUEST",
            "duration_s": manifest["motion_limits"]["settle_duration_s"],
            "execution_authorized": False,
        },
        {
            "ordinal": 4,
            "name": "PRECOMMITTED_REOBSERVE_REQUEST",
            "candidate_views": views,
            "execution_authorized": False,
        },
    ]
    return {
        **base,
        "status": "PLANNED_NOT_AUTHORIZED",
        "rejection_code": "CURRENT_DYNAMIC_VIEW_EXECUTION_UNAVAILABLE",
        "stages": stages,
        "candidate_view_count": len(views),
        "current_multilayer_dynamic_views_proven": 0,
        "truth_firewall": {
            "object_pose_truth_changes_plan": False,
            "contact_name_or_normal_changes_plan": False,
            "runtime_image_changes_precommitted_motion": False,
            "postrun_object_pose_write_allowed": False,
        },
    }


__all__ = [
    "FROZEN_SOURCES",
    "SCHEMA_VERSION",
    "build_backoff_reobserve_plan",
]
