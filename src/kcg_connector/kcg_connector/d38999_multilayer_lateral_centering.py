"""Bounded truth-free XY centering law for D38999 D4."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_lateral_centering_v1"
GAIN_M_PER_N = 0.000025
MAXIMUM_STEP_M = 0.00002
MAXIMUM_RADIUS_M = 0.003
MINIMUM_LATERAL_FORCE_N = 0.10
FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_wrist_ft_guarded_insertion_v1.yaml": (
        "a6986329bec83ad4a0da077c88023f20f70c2ff33987bccc19806b835d4ab184"
    ),
    "src/kcg_connector/kcg_connector/d38999_wrist_ft_guarded_insertion.py": (
        "3e08b006365129a6c431b997b8065a4f7c602168f7a7aebe2aff5740d353ecbe"
    ),
    "src/kcg_connector/test/test_d38999_wrist_ft_guarded_insertion.py": (
        "ebe862bd1217497c967e13067e345fbc6de1dd739d2465f3f1be76938cd66634"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D3-LIGHT-FACE-CONTACT/"
    "TASK_RESULT.json": (
        "822af4fcc991c1bd7428ea74aeed91259bcac1d45f1652b0c2d4b03fbbdb99a2"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_sources(root: Path) -> list[dict[str, str]]:
    rows = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen D4 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen D4 source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return rows


def _vector2(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (str, bytes, bool)):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != 2 or not all(math.isfinite(item) for item in result):
        return None
    return result  # type: ignore[return-value]


def compute_bounded_xy_correction(
    lateral_force_task_n: Sequence[float],
    current_xy_offset_task_m: Sequence[float],
    *,
    upstream_light_contact_ready: bool,
) -> dict[str, Any]:
    """Compute one offline diagnostic correction; never authorize motion."""

    base = {
        "schema_version": SCHEMA_VERSION,
        "status": "REJECTED_SAFE_STOP",
        "rejection_code": None,
        "delta_tcp_task_m": [0.0, 0.0, 0.0],
        "next_xy_offset_task_m": None,
        "correction_candidate": False,
        "motion_command_emitted": False,
        "control_authorized": False,
        "dynamic_centering_pass_claimed": False,
    }
    force = _vector2(lateral_force_task_n)
    offset = _vector2(current_xy_offset_task_m)
    if force is None or offset is None or type(upstream_light_contact_ready) is not bool:
        return {**base, "rejection_code": "INVALID_CENTERING_INPUT"}
    if not upstream_light_contact_ready:
        return {**base, "rejection_code": "UPSTREAM_LIGHT_CONTACT_REJECTED"}
    force_norm = math.hypot(*force)
    if force_norm < MINIMUM_LATERAL_FORCE_N:
        return {
            **base,
            "status": "NO_CORRECTION_NEEDED_DIAGNOSTIC_ONLY",
            "rejection_code": "LATERAL_FORCE_BELOW_CANDIDATE_THRESHOLD",
            "next_xy_offset_task_m": list(offset),
        }
    correction = [GAIN_M_PER_N * force[0], GAIN_M_PER_N * force[1]]
    correction_norm = math.hypot(*correction)
    if correction_norm > MAXIMUM_STEP_M:
        scale = MAXIMUM_STEP_M / correction_norm
        correction = [value * scale for value in correction]
        correction_norm = MAXIMUM_STEP_M
    next_offset = [offset[index] + correction[index] for index in range(2)]
    if math.hypot(*next_offset) > MAXIMUM_RADIUS_M:
        return {**base, "rejection_code": "XY_SEARCH_RADIUS_EXCEEDED"}
    return {
        **base,
        "status": "OFFLINE_CORRECTION_CANDIDATE",
        "rejection_code": "CORRECTION_DIAGNOSTIC_ONLY",
        "delta_tcp_task_m": [correction[0], correction[1], 0.0],
        "next_xy_offset_task_m": next_offset,
        "correction_norm_m": correction_norm,
        "correction_candidate": True,
    }


def build_multilayer_lateral_centering_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    document = yaml.safe_load(
        (root / "src/kcg_connector/config/d38999_wrist_ft_guarded_insertion_v1.yaml").read_text()
    )
    d3 = json.loads(
        (root / "artifacts/agent_control/tasks/EIGHT-HOUR-D3-LIGHT-FACE-CONTACT/TASK_RESULT.json").read_text()
    )
    motion = document.get("motion", {}) if isinstance(document, dict) else {}
    interface = document.get("controller_interface", {}) if isinstance(document, dict) else {}
    if (
        motion.get("maximum_xy_correction_step_m") != MAXIMUM_STEP_M
        or motion.get("maximum_xy_search_radius_m") != MAXIMUM_RADIUS_M
        or motion.get("lateral_correction_gain_m_per_n") != GAIN_M_PER_N
        or motion.get("correction_direction") != "environment_force_on_tool"
        or document.get("contact_response", {}).get("minimum_lateral_correction_force_n")
        != MINIMUM_LATERAL_FORCE_N
        or interface.get("task_frame_id") != "connector_task_frame"
        or d3.get("dynamic_light_contact_pass_claimed") is not False
    ):
        raise ValueError("authoritative D4 centering contract changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_CENTERING_LAW_READY",
        "classification": "BOUNDED_WRENCH_ONLY_XY_CENTERING",
        "task_frame_id": "connector_task_frame",
        "input_components": ["Fx", "Fy"],
        "output_components": ["dx", "dy", "dz=0"],
        "correction_direction": "environment_force_on_tool",
        "lateral_correction_gain_m_per_n": GAIN_M_PER_N,
        "minimum_lateral_force_candidate_n": MINIMUM_LATERAL_FORCE_N,
        "maximum_xy_correction_step_m": MAXIMUM_STEP_M,
        "maximum_xy_search_radius_m": MAXIMUM_RADIUS_M,
        "preentry_requires_freeze_and_unload": True,
        "truth_pose_input_allowed": False,
        "contact_truth_input_allowed": False,
        "current_readiness": compute_bounded_xy_correction(
            (0.0, 0.0), (0.0, 0.0), upstream_light_contact_ready=False
        ),
        "simulation_started": False,
        "dynamic_centering_pass_claimed": False,
        "control_authorized": False,
        "hardware_authorized": False,
        "sources": sources,
    }


__all__ = [
    "FROZEN_SOURCES", "GAIN_M_PER_N", "MAXIMUM_RADIUS_M", "MAXIMUM_STEP_M",
    "MINIMUM_LATERAL_FORCE_N", "SCHEMA_VERSION",
    "build_multilayer_lateral_centering_contract", "compute_bounded_xy_correction",
]
