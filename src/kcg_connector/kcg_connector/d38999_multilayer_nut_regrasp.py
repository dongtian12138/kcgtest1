"""Evidence gate from body release to the existing nut-regrasp adapter."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from kcg_connector.d38999_visual_tactile_regrasp_adapter import (
    load_visual_tactile_regrasp_adapter_contract,
)


SCHEMA_VERSION = "kcg_d38999_multilayer_nut_regrasp_v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-E2-RELEASE-BODY/TASK_RESULT.json": "56e4c0076a4c087416936efb86b6911957b3aca81cb680a72a84c2237904b01d",
    "src/kcg_connector/kcg_connector/d38999_visual_tactile_regrasp_adapter.py": "f508558d0e1706318357a52666191aff14e1bae77f7e798da2306f27296361fd",
    "src/kcg_connector/config/d38999_visual_tactile_regrasp_adapter_v1.yaml": "25f122d7ae8a6059b0a8630fbf7a752179f979f43b39d98bb65e16dc8812787d",
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json": "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783",
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b",
}


@dataclass(frozen=True)
class NutRegraspReadiness:
    e2_body_release_dynamic_pass: bool
    visual_pose_accepted: bool
    tactile_ready_latched: bool
    wrist_guard_safe: bool
    wrist_guard_fault_latched: bool
    e2_evidence_id: str | None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base(code: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "REJECTED_SAFE_HOLD",
        "rejection_code": code,
        "regrasp_request_candidate": None,
        "robot_motion_command_emitted": False,
        "finger_command_emitted": False,
        "control_authorized": False,
        "dynamic_nut_regrasp_pass_claimed": False,
        "hardware_authorized": False,
    }


def evaluate_nut_regrasp_gate(
    readiness: NutRegraspReadiness,
    *,
    current_target_authority: Mapping[str, Any] | None,
    coupling_nut_path: str,
) -> dict[str, Any]:
    base = _base()
    if not isinstance(readiness, NutRegraspReadiness) or any(
        type(getattr(readiness, item.name)) is not bool
        for item in fields(readiness)[:5]
    ):
        return {**base, "rejection_code": "INVALID_READINESS_SNAPSHOT"}
    gates = (
        (readiness.e2_body_release_dynamic_pass, "E2_BODY_RELEASE_NOT_DYNAMIC"),
        (readiness.visual_pose_accepted, "VISUAL_POSE_NOT_ACCEPTED"),
        (readiness.tactile_ready_latched, "TACTILE_READY_NOT_LATCHED"),
        (readiness.wrist_guard_safe and not readiness.wrist_guard_fault_latched,
         "WRIST_MOMENT_GUARD_REJECTED"),
    )
    for passed, code in gates:
        if not passed:
            return {**base, "rejection_code": code}
    if not isinstance(readiness.e2_evidence_id, str) or not readiness.e2_evidence_id.strip():
        return {**base, "rejection_code": "E2_EVIDENCE_ID_MISSING"}
    if not isinstance(coupling_nut_path, str) or not coupling_nut_path.startswith("/World/"):
        return {**base, "rejection_code": "CURRENT_COUPLING_NUT_PATH_INVALID"}
    if not isinstance(current_target_authority, Mapping):
        return {**base, "rejection_code": "CURRENT_MULTILAYER_TARGET_AUTHORITY_MISSING"}
    required = {
        "source_path", "source_sha256", "plan_artifact_path", "plan_artifact_sha256",
        "coupling_nut_path", "visual_tactile_plan_ready",
    }
    if set(current_target_authority) != required:
        return {**base, "rejection_code": "CURRENT_MULTILAYER_TARGET_AUTHORITY_INVALID"}
    if (
        not all(isinstance(current_target_authority[key], str) and current_target_authority[key]
                for key in ("source_path", "plan_artifact_path"))
        or any(SHA256.fullmatch(str(current_target_authority[key])) is None
               for key in ("source_sha256", "plan_artifact_sha256"))
        or current_target_authority["coupling_nut_path"] != coupling_nut_path
        or current_target_authority["visual_tactile_plan_ready"] is not True
    ):
        return {**base, "rejection_code": "CURRENT_MULTILAYER_TARGET_AUTHORITY_INVALID"}
    return {
        **base,
        "status": "OFFLINE_NUT_REGRASP_REQUEST_CANDIDATE",
        "rejection_code": "DIAGNOSTIC_ONLY_NOT_MOTION_AUTHORITY",
        "regrasp_request_candidate": {
            "coupling_nut_path": coupling_nut_path,
            "plan_artifact_path": current_target_authority["plan_artifact_path"],
            "plan_artifact_sha256": current_target_authority["plan_artifact_sha256"],
            "adapter": "d38999_visual_tactile_regrasp_adapter_v1",
            "pose_values_embedded": False,
        },
    }


def build_nut_regrasp_contract(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"frozen E3 source hash mismatch: {relative}")
        sources.append({"path": relative, "sha256": expected})
    e2 = json.loads((root / next(p for p in FROZEN_SOURCES if "E2-" in p)).read_text())
    mapping = json.loads((root / "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json").read_text())
    legacy = load_visual_tactile_regrasp_adapter_contract(repository=root)
    nut_path = mapping["representations"]["D38999_ASSEMBLY_CONTROL_V1"]["logical_paths"]["coupling_nut"]
    if (
        e2.get("outcome") != "OFFLINE_PASS"
        or e2.get("dynamic_body_release_pass_claimed") is not False
        or legacy.enabled_by_default is not False
        or legacy.boundaries["production_control_authorized"] is not False
        or legacy.boundaries["truth_pose_feedback_used_for_target"] is not False
        or not nut_path.endswith("/LoosePlug/CouplingNut")
    ):
        raise ValueError("authoritative E3 regrasp boundary changed")
    current = evaluate_nut_regrasp_gate(
        NutRegraspReadiness(False, False, False, False, True, None),
        current_target_authority=None,
        coupling_nut_path=nut_path,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "coupling_nut_path": nut_path,
        "legacy_adapter_status": legacy.status,
        "legacy_target_order": list(legacy.target_seeds),
        "legacy_world_target_auto_migration_allowed": False,
        "current_multilayer_target_authority_available": False,
        "current_decision": current,
        "simulation_started": False,
        "dynamic_nut_regrasp_pass_claimed": False,
        "control_authorized": False,
        "hardware_authorized": False,
        "sources": sources,
    }


__all__ = ["FROZEN_SOURCES", "NutRegraspReadiness", "build_nut_regrasp_contract", "evaluate_nut_regrasp_gate"]
