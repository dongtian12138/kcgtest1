"""Post-grasp snapshot gate contract.

Reuses the primitive validators from ``postgrasp_snapshot_truth`` and defines
the full frozen-hold restore record: robot joints/velocities, finger state,
Plug root rigid state, optional CouplingNut joint state, Nut rigid state, and
provenance. The two non-articulated rigid bodies are restored only at an
explicit diagnostic initialization boundary, never by formal control.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from kcg_connector.postgrasp_snapshot_truth import (
    SETTLE_STEPS_DEFAULT,
    SnapshotContractError,
    SettleNotComplete,
    _array,
    _finite,
    _rigid_state,
)

GATE_SCHEMA_VERSION = "kcg_d38999_postgrasp_snapshot_gate_v1"
GATE_ROLE = "snapshot_restore_truth"
REPLAY_BUNDLE_SCHEMA_VERSION = "kcg_d38999_replay_bundle_manifest_v1"
REPLAY_BUNDLE_ROLE = "replay_bundle"
REPLAY_STAGE_FILENAME = "replay_stage.usda"
REPLAY_BUNDLE_MANIFEST_FILENAME = "replay_bundle_manifest.json"


def _rigid_posthoc(document, label):
    return _rigid_state(document, label)


def validate_snapshot_gate_document(document: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "role",
        "scope",
        "snapshot_id",
        "timestamp_utc",
        "episode",
        "seed",
        "global_step",
        "physics_step",
        "robot_state",
        "finger_state",
        "plug_root_state",
        "coupling_nut_joint_state",
        "nut_rigid_state_restore_only",
        "frozen_command",
        "provenance",
    }
    actual = set(document)
    if actual != required:
        raise SnapshotContractError(
            f"gate snapshot keys differ missing={sorted(required-actual)} extra={sorted(actual-required)}"
        )
    if document["schema_version"] != GATE_SCHEMA_VERSION:
        raise SnapshotContractError("unsupported gate snapshot schema")
    if document["role"] != GATE_ROLE:
        raise SnapshotContractError("gate snapshot role must be snapshot_restore_truth")
    if document["scope"] != "snapshot_restore_and_posthoc_evaluation_only":
        raise SnapshotContractError("bad gate snapshot scope")
    result = dict(document)
    robot = document["robot_state"]
    if set(robot) != {"q_rad", "qd_rad_s"}:
        raise SnapshotContractError("robot_state keys differ")
    result["robot_state"] = {
        "q_rad": _array(robot["q_rad"], "robot_state.q_rad", ndim=1).tolist(),
        "qd_rad_s": _array(robot["qd_rad_s"], "robot_state.qd_rad_s", ndim=1).tolist(),
    }
    finger = document["finger_state"]
    for key in (
        "hand_q_actual_rad",
        "hand_q_target_rad",
        "finger_root_torque_proxy_nm",
    ):
        if key not in finger:
            raise SnapshotContractError(f"finger_state missing {key}")
        _array(finger[key], f"finger_state.{key}", ndim=1)
    if np.asarray(finger["finger_root_torque_proxy_nm"]).shape != (3,):
        raise SnapshotContractError(
            "finger_state.finger_root_torque_proxy_nm must contain 3 values"
        )
    # The tare channel is recorded by every current capture.  Legacy
    # snapshots captured before this channel existed may omit it, so the
    # raw document contract tolerates absence; when present it must be a
    # strictly finite 3-channel array.  The replay bundle loader enforces
    # presence for any bundle consumed by a fresh replay process.
    tare = finger.get("finger_root_tare_efforts_nm")
    if tare is not None:
        _array(tare, "finger_state.finger_root_tare_efforts_nm", ndim=1)
        if np.asarray(tare, dtype=np.float64).shape != (3,):
            raise SnapshotContractError(
                "finger_state.finger_root_tare_efforts_nm must contain 3 values"
            )
    result["finger_state"] = {
        key: np.asarray(finger[key], dtype=np.float64).tolist()
        for key in finger
    }
    result["plug_root_state"] = _rigid_state(
        document["plug_root_state"], "plug_root_state"
    )
    joint = document["coupling_nut_joint_state"]
    if joint is not None:
        if set(joint) != {"q_rad", "qd_rad_s"}:
            raise SnapshotContractError("coupling_nut_joint_state keys differ")
        result["coupling_nut_joint_state"] = {
            "q_rad": _finite(joint["q_rad"], "joint.q_rad"),
            "qd_rad_s": _finite(joint["qd_rad_s"], "joint.qd_rad_s"),
        }
    else:
        result["coupling_nut_joint_state"] = None
    result["nut_rigid_state_restore_only"] = _rigid_state(
        document["nut_rigid_state_restore_only"],
        "nut_rigid_state_restore_only",
    )
    if not isinstance(document["frozen_command"], Mapping):
        raise SnapshotContractError("frozen_command must be mapping")
    frozen = document["frozen_command"]
    if "wrist_ft_snapshot_reference" not in frozen:
        raise SnapshotContractError("frozen_command missing wrist_ft_snapshot_reference")
    snapshot_wrist = _array(
        frozen["wrist_ft_snapshot_reference"],
        "frozen_command.wrist_ft_snapshot_reference",
        ndim=1,
    )
    if snapshot_wrist.shape != (6,):
        raise SnapshotContractError("wrist_ft_snapshot_reference must contain 6 values")
    if "wrist_ft_snapshot_global_step" not in frozen:
        raise SnapshotContractError("frozen_command missing wrist_ft_snapshot_global_step")
    snapshot_step = int(
        _finite(
            frozen["wrist_ft_snapshot_global_step"],
            "frozen_command.wrist_ft_snapshot_global_step",
        )
    )
    if snapshot_step != int(document["global_step"]):
        raise SnapshotContractError(
            "wrist snapshot step is not synchronized with document global_step"
        )
    if not isinstance(document["provenance"], Mapping):
        raise SnapshotContractError("provenance must be mapping")
    return result


def capture_snapshot_gate_document(
    *,
    snapshot_id,
    timestamp_utc,
    episode,
    seed,
    global_step,
    physics_step,
    robot_getters,
    finger_getters,
    plug_root_getters,
    nut_joint_getters,
    nut_posthoc_getters,
    frozen_command,
    provenance,
) -> dict[str, Any]:
    document = {
        "schema_version": GATE_SCHEMA_VERSION,
        "role": GATE_ROLE,
        "scope": "snapshot_restore_and_posthoc_evaluation_only",
        "snapshot_id": snapshot_id,
        "timestamp_utc": timestamp_utc,
        "episode": episode,
        "seed": int(seed),
        "global_step": int(global_step),
        "physics_step": int(physics_step),
        "robot_state": {
            "q_rad": np.asarray(robot_getters["q"](), dtype=np.float64).tolist(),
            "qd_rad_s": np.asarray(robot_getters["qd"](), dtype=np.float64).tolist(),
        },
        "finger_state": {
            "hand_q_actual_rad": np.asarray(
                finger_getters["hand_q_actual"](), dtype=np.float64
            ).tolist(),
            "hand_q_target_rad": np.asarray(
                finger_getters["hand_q_target"](), dtype=np.float64
            ).tolist(),
            "finger_root_torque_proxy_nm": np.asarray(
                finger_getters["finger_root_torque_proxy"](), dtype=np.float64
            ).tolist(),
            "finger_root_tare_efforts_nm": np.asarray(
                finger_getters["finger_root_tare_efforts"](), dtype=np.float64
            ).tolist(),
        },
        "plug_root_state": {
            "position_m": np.asarray(plug_root_getters["position"]()).tolist(),
            "orientation_wxyz": np.asarray(plug_root_getters["orientation"]()).tolist(),
            "linear_velocity_m_s": np.asarray(plug_root_getters["linear_velocity"]()).tolist(),
            "angular_velocity_rad_s": np.asarray(plug_root_getters["angular_velocity"]()).tolist(),
        },
        "coupling_nut_joint_state": (
            {
                "q_rad": float(nut_joint_getters["q"]()),
                "qd_rad_s": float(nut_joint_getters["qd"]()),
            }
            if nut_joint_getters is not None
            else None
        ),
        "nut_rigid_state_restore_only": {
            "position_m": np.asarray(nut_posthoc_getters["position"]()).tolist(),
            "orientation_wxyz": np.asarray(nut_posthoc_getters["orientation"]()).tolist(),
            "linear_velocity_m_s": np.asarray(nut_posthoc_getters["linear_velocity"]()).tolist(),
            "angular_velocity_rad_s": np.asarray(nut_posthoc_getters["angular_velocity"]()).tolist(),
        },
        "frozen_command": dict(frozen_command),
        "provenance": dict(provenance),
    }
    return validate_snapshot_gate_document(document)


def capture_allowed_after_settle(step_index, settle_steps=SETTLE_STEPS_DEFAULT):
    if int(step_index) < settle_steps:
        raise SettleNotComplete("snapshot settle window not complete")
    return True


@dataclass(frozen=True)
class RestoreConsistency:
    verified: bool
    reasons: tuple[str, ...]
    diagnostics: dict[str, Any]


def evaluate_restore_consistency(
    *,
    snapshot,
    tail_robot_qd,
    tail_robot_q,
    tail_finger_torque,
    tail_wrist_wrench,
    restored_plug_position,
    restored_plug_orientation,
    restored_nut_position,
    restored_nut_orientation,
) -> RestoreConsistency:
    snapshot = validate_snapshot_gate_document(snapshot)
    reasons = []
    diagnostics: dict[str, Any] = {}
    qd = np.asarray(tail_robot_qd, dtype=np.float64)
    q = np.asarray(tail_robot_q, dtype=np.float64)
    torque = np.asarray(tail_finger_torque, dtype=np.float64)
    wrench = np.asarray(tail_wrist_wrench, dtype=np.float64)
    for name, value in (
        ("q", q),
        ("qd", qd),
        ("torque", torque),
        ("wrench", wrench),
    ):
        if not np.all(np.isfinite(value)):
            reasons.append(f"NON_FINITE_{name.upper()}")
    if q.ndim != 2 or q.shape[1:] != np.asarray(
        snapshot["robot_state"]["q_rad"]
    ).shape:
        reasons.append("ROBOT_POSITION_TAIL_SHAPE_INVALID")
    elif float(
        np.max(
            np.abs(
                q[-1]
                - np.asarray(snapshot["robot_state"]["q_rad"])
            )
        )
    ) > 0.005:
        reasons.append("ROBOT_POSITION_INCONSISTENT")
    if qd.ndim == 2 and float(np.max(np.abs(np.mean(qd, axis=0)))) > 0.05:
        reasons.append("ROBOT_NOT_SETTLED")
    if torque.ndim == 2 and float(
        np.max(np.abs(np.mean(torque, axis=0) - np.asarray(snapshot["finger_state"]["finger_root_torque_proxy_nm"])))
    ) > 0.10:
        reasons.append("FINGER_LOAD_INCONSISTENT")
    snapshot_wrist = snapshot["frozen_command"]["wrist_ft_snapshot_reference"]
    if wrench.ndim == 2:
        delta = np.mean(wrench, axis=0) - np.asarray(snapshot_wrist)
        max_force_delta_n = float(np.max(np.abs(delta[:3])))
        max_moment_delta_nm = float(np.max(np.abs(delta[3:])))
        diagnostics["max_force_delta_n"] = max_force_delta_n
        diagnostics["max_moment_delta_nm"] = max_moment_delta_nm
        diagnostics["force_gate_n"] = 1.0
        diagnostics["moment_gate_nm"] = 0.10
        diagnostics["force_gate_threshold_label"] = "SIM_TUNING_ONLY_CANDIDATE"
        diagnostics["moment_gate_threshold_label"] = "SIM_TUNING_ONLY_CANDIDATE"
        if max_force_delta_n > 1.0:
            reasons.append("WRIST_FORCE_LOAD_INCONSISTENT")
        if max_moment_delta_nm > 0.10:
            reasons.append("WRIST_MOMENT_LOAD_INCONSISTENT")
    restored = np.asarray(restored_plug_position, dtype=np.float64)
    original = np.asarray(snapshot["plug_root_state"]["position_m"])
    if restored.shape != (3,) or float(np.linalg.norm(restored - original)) > 0.001:
        reasons.append("PLUG_POSITION_POSTHOC_INCONSISTENT")
    restored_q = np.asarray(restored_plug_orientation, dtype=np.float64)
    original_q = np.asarray(snapshot["plug_root_state"]["orientation_wxyz"])
    orientation_error_rad = math.inf
    if restored_q.shape == (4,) and np.linalg.norm(restored_q) > 0.0:
        restored_q = restored_q / np.linalg.norm(restored_q)
        original_q = original_q / np.linalg.norm(original_q)
        orientation_error_rad = 2.0 * math.acos(
            min(1.0, abs(float(np.dot(restored_q, original_q))))
        )
    if orientation_error_rad > 0.002:
        reasons.append("PLUG_ORIENTATION_POSTHOC_INCONSISTENT")
    restored_nut = np.asarray(restored_nut_position, dtype=np.float64)
    original_nut = np.asarray(
        snapshot["nut_rigid_state_restore_only"]["position_m"]
    )
    if (
        restored_nut.shape != (3,)
        or float(np.linalg.norm(restored_nut - original_nut)) > 0.001
    ):
        reasons.append("NUT_POSITION_POSTHOC_INCONSISTENT")
    restored_nut_q = np.asarray(restored_nut_orientation, dtype=np.float64)
    original_nut_q = np.asarray(
        snapshot["nut_rigid_state_restore_only"]["orientation_wxyz"]
    )
    nut_orientation_error_rad = math.inf
    if restored_nut_q.shape == (4,) and np.linalg.norm(restored_nut_q) > 0.0:
        restored_nut_q = restored_nut_q / np.linalg.norm(restored_nut_q)
        original_nut_q = original_nut_q / np.linalg.norm(original_nut_q)
        nut_orientation_error_rad = 2.0 * math.acos(
            min(1.0, abs(float(np.dot(restored_nut_q, original_nut_q))))
        )
    if nut_orientation_error_rad > 0.002:
        reasons.append("NUT_ORIENTATION_POSTHOC_INCONSISTENT")
    diagnostics.update(
        {
            "reasons": reasons,
            "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
            "diagnostic_only": True,
            "plug_orientation_error_rad": orientation_error_rad,
            "nut_orientation_error_rad": nut_orientation_error_rad,
        }
    )
    return RestoreConsistency(verified=not reasons, reasons=tuple(reasons), diagnostics=diagnostics)


def write_snapshot_gate_document(path: Path | str, document: Mapping[str, Any]):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            validate_snapshot_gate_document(document),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_snapshot_gate_document(path: Path | str) -> dict[str, Any]:
    return validate_snapshot_gate_document(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def sha256_file(path: Path | str, *, label: str = "artifact") -> str:
    """SHA-256 of an existing non-empty file.  Fail closed otherwise."""
    artifact = Path(path)
    if not artifact.is_file():
        raise SnapshotContractError(f"{label} missing: {artifact}")
    if artifact.stat().st_size <= 0:
        raise SnapshotContractError(f"{label} empty: {artifact}")
    return hashlib.sha256(artifact.read_bytes()).hexdigest()


def _sha256_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SnapshotContractError(f"{label} must be a 64-char hex SHA-256")
    return value


def validate_replay_bundle_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on replay bundle manifest contract violations.

    The manifest binds exactly one snapshot file and one stage file with
    their SHA-256 digests.  It declares that object truth restore is only
    legal at initialization and that neither the bundle nor the stage is
    ever a formal estimator input or control authorization.
    """
    if not isinstance(document, Mapping):
        raise SnapshotContractError("replay bundle manifest must be a mapping")
    required = {
        "schema_version",
        "role",
        "seed",
        "snapshot_file",
        "snapshot_sha256",
        "stage_file",
        "stage_sha256",
        "source_hashes",
        "restore_truth_scope",
        "formal_estimator_input",
        "control_authorized",
    }
    actual = set(document)
    if actual != required:
        raise SnapshotContractError(
            "replay bundle manifest keys differ "
            f"missing={sorted(required - actual)} extra={sorted(actual - required)}"
        )
    if document["schema_version"] != REPLAY_BUNDLE_SCHEMA_VERSION:
        raise SnapshotContractError("unsupported replay bundle manifest schema")
    if document["role"] != REPLAY_BUNDLE_ROLE:
        raise SnapshotContractError("replay bundle manifest role mismatch")
    _finite(document["seed"], "replay_bundle.seed")
    for key in ("snapshot_file", "stage_file"):
        if not isinstance(document[key], str) or not document[key]:
            raise SnapshotContractError(
                f"replay bundle {key} must be a non-empty filename"
            )
    _sha256_hex(document["snapshot_sha256"], "replay_bundle.snapshot_sha256")
    _sha256_hex(document["stage_sha256"], "replay_bundle.stage_sha256")
    if not isinstance(document["source_hashes"], Mapping):
        raise SnapshotContractError("replay bundle source_hashes must be a mapping")
    for key, value in document["source_hashes"].items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise SnapshotContractError(
                "replay bundle source_hashes must map str to str"
            )
    if document["restore_truth_scope"] != "INITIALIZATION_ONLY":
        raise SnapshotContractError(
            "replay bundle restore_truth_scope must be INITIALIZATION_ONLY"
        )
    for key in ("formal_estimator_input", "control_authorized"):
        if document[key] is not False:
            raise SnapshotContractError(
                f"replay bundle {key} must be strictly false"
            )
    result = dict(document)
    result["seed"] = int(_finite(document["seed"], "replay_bundle.seed"))
    result["source_hashes"] = dict(document["source_hashes"])
    return result


def build_replay_bundle_manifest(
    *,
    seed,
    snapshot_file,
    snapshot_sha256,
    stage_file,
    stage_sha256,
    source_hashes,
) -> dict[str, Any]:
    return validate_replay_bundle_manifest(
        {
            "schema_version": REPLAY_BUNDLE_SCHEMA_VERSION,
            "role": REPLAY_BUNDLE_ROLE,
            "seed": int(seed),
            "snapshot_file": snapshot_file,
            "snapshot_sha256": snapshot_sha256,
            "stage_file": stage_file,
            "stage_sha256": stage_sha256,
            "source_hashes": dict(source_hashes),
            "restore_truth_scope": "INITIALIZATION_ONLY",
            "formal_estimator_input": False,
            "control_authorized": False,
        }
    )


def write_replay_bundle_manifest(
    path: Path | str, document: Mapping[str, Any]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            validate_replay_bundle_manifest(document),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_replay_bundle_manifest(path: Path | str) -> dict[str, Any]:
    """Load a bundle manifest and re-verify BOTH declared artifact hashes.

    The snapshot and stage files are resolved relative to the manifest
    directory and re-hashed byte-for-byte.  Any missing, empty, or tampered
    artifact raises ``SnapshotContractError`` so a fresh replay process can
    trust the bundle only after this loader accepts it.
    """
    output = Path(path)
    try:
        document = validate_replay_bundle_manifest(
            json.loads(output.read_text(encoding="utf-8"))
        )
    except SnapshotContractError:
        raise
    except Exception as exception:
        raise SnapshotContractError(
            "replay bundle manifest cannot be decoded"
        ) from exception
    base = output.parent
    for filename, declared in (
        ("snapshot_file", "snapshot_sha256"),
        ("stage_file", "stage_sha256"),
    ):
        digest = sha256_file(
            base / document[filename], label=f"replay_bundle.{filename}"
        )
        if digest != document[declared]:
            raise SnapshotContractError(
                f"replay bundle {filename} SHA-256 mismatch"
            )
    # A fresh replay process consumes the bundle, not a legacy evidence
    # snapshot.  The raw snapshot document loader tolerates legacy v1/v2
    # snapshots that predate the tare channel so historical evidence can
    # still be inspected.  The bundle loader is the strict entry point and
    # must refuse those legacy snapshots.
    try:
        snapshot = validate_snapshot_gate_document(
            json.loads((base / document["snapshot_file"]).read_text(encoding="utf-8"))
        )
    except SnapshotContractError:
        raise
    except Exception as exception:
        raise SnapshotContractError(
            "replay bundle snapshot cannot be decoded"
        ) from exception
    finger_state = snapshot.get("finger_state") or {}
    if "finger_root_tare_efforts_nm" not in finger_state:
        raise SnapshotContractError(
            "replay bundle snapshot missing finger_root_tare_efforts_nm"
        )
    tare = _array(
        finger_state["finger_root_tare_efforts_nm"],
        "finger_state.finger_root_tare_efforts_nm",
        ndim=1,
    )
    if np.asarray(tare, dtype=np.float64).shape != (3,):
        raise SnapshotContractError(
            "replay bundle snapshot finger_root_tare_efforts_nm must contain 3 values"
        )
    return document


__all__ = [
    "GATE_SCHEMA_VERSION",
    "GATE_ROLE",
    "REPLAY_BUNDLE_ROLE",
    "REPLAY_BUNDLE_SCHEMA_VERSION",
    "REPLAY_BUNDLE_MANIFEST_FILENAME",
    "REPLAY_STAGE_FILENAME",
    "RestoreConsistency",
    "build_replay_bundle_manifest",
    "capture_allowed_after_settle",
    "capture_snapshot_gate_document",
    "evaluate_restore_consistency",
    "load_replay_bundle_manifest",
    "load_snapshot_gate_document",
    "sha256_file",
    "validate_replay_bundle_manifest",
    "validate_snapshot_gate_document",
    "write_replay_bundle_manifest",
    "write_snapshot_gate_document",
]
