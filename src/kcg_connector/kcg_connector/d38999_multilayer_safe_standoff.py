"""Evidence-bound D38999 pre-entry standoff contract.

The standoff geometry is read from the existing assembly and preinsert
contracts.  This module does not accept an object pose and has no actuator or
control-promotion path.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from kcg_connector.d38999_assembly_baseline import load_d38999_assembly_baseline
from kcg_connector.d38999_physical_insertion import load_d38999_physical_insertion


SCHEMA_VERSION = "kcg_d38999_multilayer_safe_standoff_v1"
FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_visual_xy_preinsert_probe_v1.yaml": (
        "7b983bfc211ef25e70ddb818c5773c7b748452e81ace9b8fd0cb464385c2f18c"
    ),
    "src/kcg_connector/config/d38999_visual_xy_control_adapter_v1.yaml": (
        "748c4f5e2c07c678e8bbb395b3a3ef3d656dea035fbebe09ee66886545148431"
    ),
    "src/kcg_connector/config/d38999_physical_insertion_v1.yaml": (
        "652b46768735e59146fb249c8c7944ead9a68dc60837c615bc561361f7ded10d"
    ),
    "src/kcg_connector/config/d38999_assembly_baseline_v1.yaml": (
        "4921246d03bbfead0c0c806402dadeb432ae913d66ba18c43cc179cb11d76f62"
    ),
    "src/kcg_connector/kcg_connector/d38999_visual_xy_preinsert_probe.py": (
        "7a24c743cb80c52d873c8748ef1e6b99bb2d07646183db29c74131b55253ff71"
    ),
    "src/kcg_connector/kcg_connector/d38999_assembly_baseline.py": (
        "be9dde7a916705edbca28bc8470545eccd89edba4bf61155bb5e2fe7eeee2b95"
    ),
    "src/kcg_connector/kcg_connector/d38999_physical_insertion.py": (
        "ca02106e5552b139b2cd9341bb293ccb644e990842c339c0368f7b306b181a10"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C10-POSE-CONFIDENCE-REJECTION/"
    "TASK_RESULT.json": (
        "da8a73cbe9970dc535357068b1667cc78b929a85fb04435760e6a347b3c0f64d"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_sources(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen D1 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen D1 source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return rows


def _json_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("C10 task result must be a mapping")
    return value


def _yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def evaluate_safe_standoff_readiness(
    pose_gate_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on the upstream pose gate without accepting any pose."""

    base = {
        "schema_version": SCHEMA_VERSION,
        "status": "REJECTED_SAFE_STOP",
        "static_standoff_contract_ready": True,
        "dynamic_standoff_ready": False,
        "rejection_code": None,
        "upstream_rejection_code": None,
        "target_pose_emitted": False,
        "path_planning_authorized": False,
        "actuator_command_issued": False,
        "control_authorized": False,
        "simulation_started": False,
        "hardware_control_authorized": False,
    }
    if not isinstance(pose_gate_result, Mapping):
        return {**base, "rejection_code": "UPSTREAM_POSE_GATE_INVALID"}
    if pose_gate_result.get("task_id") != (
        "EIGHT-HOUR-C10-POSE-CONFIDENCE-REJECTION"
    ) or pose_gate_result.get("outcome") != "OFFLINE_PASS":
        return {**base, "rejection_code": "UPSTREAM_POSE_GATE_INVALID"}
    if (
        pose_gate_result.get("control_authorized") is not False
        or pose_gate_result.get("hardware_authorized") is not False
    ):
        return {**base, "rejection_code": "UPSTREAM_AUTHORIZATION_INVALID"}
    upstream_code = pose_gate_result.get("current_rejection_code")
    if (
        not isinstance(upstream_code, str)
        or not upstream_code
        or pose_gate_result.get("dynamic_pose_pass_claimed") is not False
        or pose_gate_result.get("selected_for_control") is not None
    ):
        return {**base, "rejection_code": "UPSTREAM_POSE_GATE_INVALID"}
    return {
        **base,
        "rejection_code": "UPSTREAM_POSE_REJECTED",
        "upstream_rejection_code": upstream_code,
    }


def build_multilayer_safe_standoff_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    preinsert_document = _yaml_mapping(
        root / "src/kcg_connector/config/d38999_visual_xy_preinsert_probe_v1.yaml",
        "preinsert contract",
    )
    physical_document = _yaml_mapping(
        root / "src/kcg_connector/config/d38999_physical_insertion_v1.yaml",
        "physical insertion contract",
    )
    preinsert_inputs = preinsert_document.get("inputs", {})
    planning = preinsert_document.get("planning", {})
    axial = preinsert_document.get("axial_scope", {})
    physical_inputs = physical_document.get("inputs", {})
    if (
        preinsert_document.get("schema_version")
        != "kcg_d38999_visual_xy_preinsert_probe_v1"
        or preinsert_document.get("enabled_by_default") is not False
        or preinsert_document.get("status")
        != "prepared_cpu_plan_not_physx_executed"
        or preinsert_inputs.get("visual_xy_adapter")
        != {
            "path": "src/kcg_connector/config/d38999_visual_xy_control_adapter_v1.yaml",
            "sha256": FROZEN_SOURCES[
                "src/kcg_connector/config/d38999_visual_xy_control_adapter_v1.yaml"
            ],
        }
        or preinsert_inputs.get("nominal_insertion")
        != {
            "path": "src/kcg_connector/config/d38999_physical_insertion_v1.yaml",
            "sha256": FROZEN_SOURCES[
                "src/kcg_connector/config/d38999_physical_insertion_v1.yaml"
            ],
        }
        or planning.get("target_translation_source")
        != "visual_fixed_receptacle_xy"
        or planning.get("target_z_source") != "registered_nominal"
        or planning.get("target_orientation_source") != "registered_nominal_fk"
        or planning.get("use_body_truth_for_target") is not False
        or planning.get("use_fixed_truth_for_target") is not False
        or planning.get("use_truth_for_iterative_correction") is not False
        or axial.get("engage_target_planned") is not False
        or axial.get("insertion_target_planned") is not False
        or physical_inputs.get("assembly_baseline")
        != {
            "path": "src/kcg_connector/config/d38999_assembly_baseline_v1.yaml",
            "sha256": FROZEN_SOURCES[
                "src/kcg_connector/config/d38999_assembly_baseline_v1.yaml"
            ],
        }
    ):
        raise ValueError("D1-required source fields changed")
    insertion = load_d38999_physical_insertion(
        root / "src/kcg_connector/config/d38999_physical_insertion_v1.yaml"
    )
    assembly = load_d38999_assembly_baseline(
        root / "src/kcg_connector/config/d38999_assembly_baseline_v1.yaml"
    )
    preinsert_gap_m = float(axial["preinsert_gap_m"])
    entry_gap_m = float(axial["entry_gap_m"])
    registered_margin_m = float(axial["registered_margin_before_entry_m"])
    fixed = assembly.datums.fixed
    face_target = assembly.plug_position_for_gap_m(preinsert_gap_m)
    tcp_target = insertion.motion.preinsert_tcp_position_m
    tcp_from_face = tuple(
        float(tcp_target[index] - face_target[index]) for index in range(3)
    )
    if (
        fixed.position_world_m != (0.550, 0.185, 0.2615)
        or fixed.axis_world != (0.0, 0.0, 1.0)
        or not math.isclose(preinsert_gap_m, 0.012, abs_tol=1.0e-12)
        or not math.isclose(entry_gap_m, 0.010, abs_tol=1.0e-12)
        or not math.isclose(
            registered_margin_m, 0.002, abs_tol=1.0e-12
        )
        or any(
            not math.isclose(left, right, abs_tol=1.0e-12)
            for left, right in zip(face_target, (0.550, 0.185, 0.2735))
        )
        or any(
            not math.isclose(left, right, abs_tol=1.0e-12)
            for left, right in zip(tcp_target, (0.550, 0.185, 0.32198))
        )
        or any(
            not math.isclose(left, right, abs_tol=1.0e-12)
            for left, right in zip(tcp_from_face, (0.0, 0.0, 0.04848))
        )
    ):
        raise ValueError("authoritative D1 standoff relation changed")
    c10 = _json_mapping(
        root
        / "artifacts/agent_control/tasks/"
        "EIGHT-HOUR-C10-POSE-CONFIDENCE-REJECTION/TASK_RESULT.json"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_CONTRACT_READY",
        "classification": "EVIDENCE_BOUND_REGISTERED_STANDOFF",
        "coordinate_frame": "world",
        "fixed_datum": {
            "symbol": fixed.symbol,
            "prim_path": fixed.prim_path,
            "feature": fixed.feature,
            "position_world_m": list(fixed.position_world_m),
        },
        "plug_datum": {
            "symbol": assembly.datums.loose_plug.symbol,
            "prim_path": assembly.datums.loose_plug.prim_path,
            "feature": assembly.datums.loose_plug.feature,
        },
        "assembly_axis_world": list(fixed.axis_world),
        "approach_direction_world": [-value for value in fixed.axis_world],
        "gap_definition": "dot(P-F,Fz)",
        "positive_gap_direction": "fixed_positive_z",
        "preinsert_gap_m": preinsert_gap_m,
        "entry_gap_m": entry_gap_m,
        "registered_margin_before_entry_m": registered_margin_m,
        "margin_semantics": "REGISTERED_GEOMETRY_NOT_MEASURED_COLLISION_CLEARANCE",
        "plug_face_target_world_m": list(face_target),
        "nominal_tcp_target_world_m": list(tcp_target),
        "nominal_tcp_from_plug_face_m": list(tcp_from_face),
        "target_translation_source": planning["target_translation_source"],
        "target_z_source": planning["target_z_source"],
        "target_orientation_source": planning["target_orientation_source"],
        "historical_visual_adapter_loader_status": (
            "BLOCKED_UNRELATED_NOMINAL_PICK_HASH_DRIFT"
        ),
        "historical_visual_adapter_dynamic_claimed": False,
        "truth_pose_input_allowed": False,
        "contact_truth_input_allowed": False,
        "event_truth_input_allowed": False,
        "current_readiness": evaluate_safe_standoff_readiness(c10),
        "simulation_started": False,
        "robot_motion_started": False,
        "dynamic_standoff_pass_claimed": False,
        "control_authorized": False,
        "hardware_authorized": False,
        "sources": sources,
    }


__all__ = [
    "FROZEN_SOURCES",
    "SCHEMA_VERSION",
    "build_multilayer_safe_standoff_contract",
    "evaluate_safe_standoff_readiness",
]
