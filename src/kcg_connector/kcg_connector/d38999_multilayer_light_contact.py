"""Truth-free offline light-contact gate for D38999 D3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_light_contact_v1"
TASK_FRAME_ID = "connector_task_frame"
FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_wrist_ft_guarded_insertion_v1.yaml": (
        "a6986329bec83ad4a0da077c88023f20f70c2ff33987bccc19806b835d4ab184"
    ),
    "src/kcg_connector/config/d38999_tactile_engage_probe_v2.yaml": (
        "e55143c3101ad4e5008d73c5ddbaa3af9831b9076657026895b4975b84558a2f"
    ),
    "src/kcg_connector/kcg_connector/d38999_wrist_ft_guarded_insertion.py": (
        "3e08b006365129a6c431b997b8065a4f7c602168f7a7aebe2aff5740d353ecbe"
    ),
    "src/kcg_connector/test/test_d38999_wrist_ft_guarded_insertion.py": (
        "ebe862bd1217497c967e13067e345fbc6de1dd739d2465f3f1be76938cd66634"
    ),
    "src/kcg_connector/kcg_connector/wrist_moment_safety_guard.py": (
        "779f7601a69f31c87ba44ad88584f540c2178c13b8c2bc09f5bde69385df0db8"
    ),
    "src/kcg_connector/test/test_wrist_moment_safety_guard.py": (
        "9982a0781526683b77a9856ff42b2ca92b1ad7a4efd150b67afb4de3f56ce16d"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-B5-WRIST-MOMENT-MONITOR/"
    "TASK_RESULT.json": (
        "2bebe773c145d4afec89cdf1865ae97eb13db8bf9019d2006bb95ba635c38e0f"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D2-VISUAL-PREALIGN/"
    "TASK_RESULT.json": (
        "cfb2866fdf7d1a1e553404aff475cefedfb0dbc791808e78f7a38e8fbb27b7ee"
    ),
}


@dataclass(frozen=True)
class LightContactSample:
    timestamp_s: float
    sample_age_s: float
    frame_id: str
    compensated_wrench_task: Sequence[float]
    local_reference_ready: bool
    compressive_direction_calibrated: bool
    upstream_prealign_ready: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_sources(root: Path) -> list[dict[str, str]]:
    rows = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen D3 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen D3 source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return rows


def _yaml_mapping(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _json_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _base(code: str, *, contact_candidate: bool = False) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_DIAGNOSTIC_ONLY",
        "rejection_code": code,
        "light_contact_candidate": contact_candidate,
        "contact_confirmed": False,
        "motion_command_emitted": False,
        "actuator_command_issued": False,
        "control_authorized": False,
        "dynamic_light_contact_pass_claimed": False,
        "hardware_control_authorized": False,
    }


def evaluate_light_contact_sample(
    sample: LightContactSample,
) -> dict[str, Any]:
    """Classify one sample offline; never authorize or command motion."""

    if not isinstance(sample, LightContactSample):
        return _base("INVALID_SAMPLE")
    if (
        type(sample.local_reference_ready) is not bool
        or type(sample.compressive_direction_calibrated) is not bool
        or type(sample.upstream_prealign_ready) is not bool
        or not isinstance(sample.frame_id, str)
        or isinstance(sample.timestamp_s, bool)
        or isinstance(sample.sample_age_s, bool)
    ):
        return _base("INVALID_SAMPLE")
    try:
        timestamp = float(sample.timestamp_s)
        age = float(sample.sample_age_s)
        wrench = tuple(float(value) for value in sample.compensated_wrench_task)
    except (TypeError, ValueError):
        return _base("INVALID_SAMPLE")
    if (
        len(wrench) != 6
        or not math.isfinite(timestamp)
        or not math.isfinite(age)
        or age < 0.0
        or not all(math.isfinite(value) for value in wrench)
    ):
        return _base("NONFINITE_OR_INVALID_SAMPLE")
    if sample.frame_id != TASK_FRAME_ID:
        return _base("WRONG_WRENCH_FRAME")
    if not sample.upstream_prealign_ready:
        return _base("UPSTREAM_PREALIGN_REJECTED")
    if not sample.local_reference_ready:
        return _base("LOCAL_REFERENCE_NOT_READY")
    if not sample.compressive_direction_calibrated:
        return _base("CONTACT_DIRECTION_UNCALIBRATED")
    if age > 1.0 / 120.0:
        return _base("STALE_WRENCH_SAMPLE")
    fx, fy, fz, tx, ty, tz = wrench
    lateral_force = math.hypot(fx, fy)
    if abs(fz) > 5.0:
        return _base("AXIAL_FORCE_ABORT")
    if lateral_force > 2.0:
        return _base("LATERAL_FORCE_ABORT")
    if max(abs(tx), abs(ty), abs(tz)) > 0.30:
        return _base("WRIST_MOMENT_HARD_LIMIT")
    if math.hypot(tx, ty) > 0.18:
        return _base("BENDING_MOMENT_EXPERIMENT_ABORT")
    if abs(tz) > 0.05:
        return _base("TIGHTENING_MOMENT_EXPERIMENT_ABORT")
    if fz >= 0.25:
        return _base("LIGHT_CONTACT_CANDIDATE_DIAGNOSTIC_ONLY", contact_candidate=True)
    return _base("NO_LIGHT_CONTACT_CANDIDATE")


def build_multilayer_light_contact_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    guarded = _yaml_mapping(
        root / "src/kcg_connector/config/d38999_wrist_ft_guarded_insertion_v1.yaml"
    )
    tactile = _yaml_mapping(
        root / "src/kcg_connector/config/d38999_tactile_engage_probe_v2.yaml"
    )
    b5 = _json_mapping(
        root / "artifacts/agent_control/tasks/"
        "EIGHT-HOUR-B5-WRIST-MOMENT-MONITOR/TASK_RESULT.json"
    )
    d2 = _json_mapping(
        root / "artifacts/agent_control/tasks/"
        "EIGHT-HOUR-D2-VISUAL-PREALIGN/TASK_RESULT.json"
    )
    interface = guarded.get("controller_interface", {})
    motion = guarded.get("motion", {})
    abort = guarded.get("experimental_abort_envelope", {})
    contact = guarded.get("contact_response", {})
    sensor = tactile.get("sensor_policy", {})
    tactile_contact = tactile.get("contact_detection", {})
    if (
        interface.get("task_frame_id") != TASK_FRAME_ID
        or tuple(interface.get("wrench_order", ()))
        != ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
        or motion.get("control_rate_hz") != 240
        or motion.get("guarded_approach_speed_m_s") != 0.00035
        or motion.get("contact_retract_distance_m") != 0.00030
        or abort.get("maximum_axial_force_n") != 5.0
        or abort.get("maximum_lateral_force_n") != 2.0
        or abort.get("maximum_bending_torque_nm") != 0.18
        or abort.get("maximum_tightening_torque_nm") != 0.05
        or contact.get("minimum_axial_contact_force_n") != 0.25
        or sensor.get("controller_wrench_mode")
        != "subtract_stopped_preinsert_local_reference"
        or sensor.get("local_reference_samples") != 120
        or sensor.get("local_reference_is_safety_tare") is not False
        or sensor.get("requires_lip_contact_direction_calibration") is not True
        or tactile_contact.get("contact_on_compressive_axial_force_n") != 0.25
        or tactile_contact.get("contact_off_compressive_axial_force_n") != 0.10
        or b5.get("moment_limit_nm") != 0.30
        or d2.get("current_rejection_code") != "UPSTREAM_POSE_REJECTED"
    ):
        raise ValueError("authoritative D3 contact contract changed")
    forbidden = set(interface.get("forbidden_control_inputs", ()))
    required_forbidden = {
        "simulator_object_truth_pose", "simulator_truth_gap", "physx_contact_report",
        "physx_contact_manifold", "collider_path", "contact_normal",
        "contact_separation", "contact_material",
    }
    if forbidden != required_forbidden:
        raise ValueError("D3 forbidden-control-input boundary changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_CONTACT_GATE_READY",
        "classification": "WRENCH_ONLY_LIGHT_CONTACT_DIAGNOSTIC_GATE",
        "task_frame_id": TASK_FRAME_ID,
        "wrench_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "control_rate_hz": 240,
        "maximum_sample_age_s": 1.0 / 120.0,
        "local_reference_mode": sensor["controller_wrench_mode"],
        "local_reference_samples": 120,
        "local_reference_is_safety_tare": False,
        "compressive_direction_sign_candidate": 1,
        "compressive_direction_calibrated": False,
        "contact_on_candidate_n": 0.25,
        "contact_off_candidate_n": 0.10,
        "guarded_approach_speed_m_s": 0.00035,
        "contact_retract_distance_m": 0.00030,
        "experimental_abort_envelope": {
            "maximum_axial_force_n": 5.0,
            "maximum_lateral_force_n": 2.0,
            "maximum_bending_torque_nm": 0.18,
            "maximum_tightening_torque_nm": 0.05,
            "calibrated_hardware_safety_limit": False,
        },
        "formal_moment_component_limit_nm": 0.30,
        "forbidden_control_inputs": sorted(forbidden),
        "current_readiness": {
            "status": "REJECTED_SAFE_STOP",
            "rejection_code": "UPSTREAM_PREALIGN_REJECTED",
            "secondary_blocker": "CONTACT_DIRECTION_UNCALIBRATED",
            "motion_command_emitted": False,
            "control_authorized": False,
        },
        "simulation_started": False,
        "contact_simulation_started": False,
        "dynamic_light_contact_pass_claimed": False,
        "control_authorized": False,
        "hardware_authorized": False,
        "sources": sources,
    }


__all__ = [
    "FROZEN_SOURCES",
    "LightContactSample",
    "SCHEMA_VERSION",
    "TASK_FRAME_ID",
    "build_multilayer_light_contact_contract",
    "evaluate_light_contact_sample",
]
