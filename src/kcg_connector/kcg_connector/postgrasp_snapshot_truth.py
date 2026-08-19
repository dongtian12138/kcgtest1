"""Post-grasp snapshot truth capture/restore contract.

This module is deliberately importable without Isaac Sim.  All object APIs are
injected as callables so the truth firewall is testable on CPU.  Restore never
writes a linked-body world pose: only a Plug root rigid-body state plus the
CouplingNut revolute joint ``q/qd`` are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


POSTGRASP_SNAPSHOT_SCHEMA_VERSION = "kcg_d38999_postgrasp_snapshot_v1"
TRUTH_ROLE = "truth_restore"
SETTLE_STEPS_DEFAULT = 120  # 240 Hz * 0.5 s
TAIL_STEPS_DEFAULT = 60      # post-settle diagnostic tail


class TruthFirewallViolation(ValueError):
    """Raised when truth-role data is passed through a formal observation API."""


class SnapshotContractError(ValueError):
    """Raised for schema/state contract violations."""


class RestoreRejected(RuntimeError):
    """Raised when a snapshot cannot be safely restored."""


class SettleNotComplete(RuntimeError):
    """Raised when capture is requested before the post-restore settle window."""


@dataclass(frozen=True)
class RestoreProbe:
    status: str
    mode: str | None
    available_apis: tuple[str, ...]
    missing_apis: tuple[str, ...]
    forbidden_apis_present: tuple[str, ...]


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SnapshotContractError(f"{label} must be finite")
    return result


def _array(values: Any, label: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise SnapshotContractError(f"{label} must be non-empty and finite")
    if ndim is not None and array.ndim != ndim:
        raise SnapshotContractError(f"{label} must have ndim {ndim}")
    return array


def _quaternion(values: Any, label: str) -> tuple[float, float, float, float]:
    array = _array(values, label, ndim=1)
    if array.shape != (4,):
        raise SnapshotContractError(f"{label} must contain w,x,y,z")
    norm = float(np.linalg.norm(array))
    if abs(norm - 1.0) > 1.0e-6:
        raise SnapshotContractError(f"{label} must be unit norm")
    return tuple(float(value) for value in array)


def _rigid_state(document: Mapping[str, Any], label: str) -> dict[str, Any]:
    required = {
        "position_m",
        "orientation_wxyz",
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
    }
    actual = set(document)
    if actual != required:
        raise SnapshotContractError(f"{label} keys differ: {sorted(actual)}")
    position = _array(document["position_m"], f"{label}.position_m", ndim=1)
    orientation = _array(
        document["orientation_wxyz"], f"{label}.orientation_wxyz", ndim=1
    )
    linear = _array(
        document["linear_velocity_m_s"], f"{label}.linear_velocity_m_s", ndim=1
    )
    angular = _array(
        document["angular_velocity_rad_s"],
        f"{label}.angular_velocity_rad_s",
        ndim=1,
    )
    for name, array in (
        ("position_m", position),
        ("linear_velocity_m_s", linear),
        ("angular_velocity_rad_s", angular),
    ):
        if array.shape != (3,):
            raise SnapshotContractError(f"{label}.{name} must contain 3 values")
    if orientation.shape != (4,):
        raise SnapshotContractError(f"{label}.orientation_wxyz must contain 4 values")
    _quaternion(orientation, f"{label}.orientation_wxyz")
    return {
        "position_m": position.tolist(),
        "orientation_wxyz": orientation.tolist(),
        "linear_velocity_m_s": linear.tolist(),
        "angular_velocity_rad_s": angular.tolist(),
    }


def _joint_state(document: Mapping[str, Any], label: str) -> dict[str, Any]:
    actual = set(document)
    if actual != {"q_rad", "qd_rad_s"}:
        raise SnapshotContractError(f"{label} keys differ: {sorted(actual)}")
    q = _finite(document["q_rad"], f"{label}.q_rad")
    qd = _finite(document["qd_rad_s"], f"{label}.qd_rad_s")
    return {"q_rad": q, "qd_rad_s": qd}


def _robot_state(document: Mapping[str, Any]) -> dict[str, Any]:
    actual = set(document)
    if actual != {"q_rad", "qd_rad_s"}:
        raise SnapshotContractError(
            f"robot_state keys differ: {sorted(actual)}"
        )
    q = _array(document["q_rad"], "robot_state.q_rad", ndim=1)
    qd = _array(document["qd_rad_s"], "robot_state.qd_rad_s", ndim=1)
    if q.shape[0] != qd.shape[0]:
        raise SnapshotContractError("robot q and qd dimensions differ")
    return {"q_rad": q.tolist(), "qd_rad_s": qd.tolist()}


def validate_truth_snapshot(document: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on role/schema/state contract violations."""
    if not isinstance(document, Mapping):
        raise SnapshotContractError("snapshot must be a mapping")
    required = {
        "schema_version",
        "role",
        "scope",
        "snapshot_id",
        "timestamp_utc",
        "episode",
        "seed",
        "global_step",
        "plug_root_state",
        "coupling_nut_joint_state",
        "robot_state",
        "frozen_command",
        "source_hashes",
    }
    actual = set(document)
    if actual != required:
        raise SnapshotContractError(
            f"snapshot keys differ; missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required)}"
        )
    if document["schema_version"] != POSTGRASP_SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotContractError("unsupported snapshot schema")
    if document["role"] != TRUTH_ROLE:
        raise TruthFirewallViolation(
            f"role={document['role']!r} is not {TRUTH_ROLE!r}"
        )
    if document["scope"] != "snapshot_restore_and_posthoc_evaluation_only":
        raise SnapshotContractError("snapshot scope is not truth-only")
    for key in ("snapshot_id", "timestamp_utc", "episode"):
        if not isinstance(document[key], str) or not document[key]:
            raise SnapshotContractError(f"{key} must be non-empty text")
    _finite(document["seed"], "seed")
    _finite(document["global_step"], "global_step")
    result = dict(document)
    result["plug_root_state"] = _rigid_state(
        document["plug_root_state"], "plug_root_state"
    )
    joint = document["coupling_nut_joint_state"]
    if joint is not None:
        result["coupling_nut_joint_state"] = _joint_state(
            joint, "coupling_nut_joint_state"
        )
    else:
        result["coupling_nut_joint_state"] = None
    result["robot_state"] = _robot_state(document["robot_state"])
    frozen = document["frozen_command"]
    if not isinstance(frozen, Mapping):
        raise SnapshotContractError("frozen_command must be a mapping")
    result["frozen_command"] = dict(frozen)
    if not isinstance(document["source_hashes"], Mapping):
        raise SnapshotContractError("source_hashes must be a mapping")
    result["source_hashes"] = dict(document["source_hashes"])
    return result


def capture_truth_snapshot(
    *,
    snapshot_id: str,
    timestamp_utc: str,
    episode: str,
    seed: int,
    global_step: int,
    plug_root_getters: Mapping[str, Any],
    nut_joint_getters: Mapping[str, Any] | None,
    robot_getters: Mapping[str, Any],
    frozen_command: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Capture through injected object APIs.  No Isaac import required."""
    plug_position = plug_root_getters["position"]()
    plug_orientation = plug_root_getters["orientation"]()
    plug_linear = plug_root_getters["linear_velocity"]()
    plug_angular = plug_root_getters["angular_velocity"]()
    robot_q = robot_getters["q"]()
    robot_qd = robot_getters["qd"]()
    joint_state = None
    if nut_joint_getters is not None:
        joint_q = nut_joint_getters["q"]()
        joint_qd = nut_joint_getters["qd"]()
        joint_state = {"q_rad": float(joint_q), "qd_rad_s": float(joint_qd)}
    document: dict[str, Any] = {
        "schema_version": POSTGRASP_SNAPSHOT_SCHEMA_VERSION,
        "role": TRUTH_ROLE,
        "scope": "snapshot_restore_and_posthoc_evaluation_only",
        "snapshot_id": snapshot_id,
        "timestamp_utc": timestamp_utc,
        "episode": episode,
        "seed": int(seed),
        "global_step": int(global_step),
        "plug_root_state": {
            "position_m": np.asarray(plug_position, dtype=np.float64).tolist(),
            "orientation_wxyz": np.asarray(
                plug_orientation, dtype=np.float64
            ).tolist(),
            "linear_velocity_m_s": np.asarray(
                plug_linear, dtype=np.float64
            ).tolist(),
            "angular_velocity_rad_s": np.asarray(
                plug_angular, dtype=np.float64
            ).tolist(),
        },
        "coupling_nut_joint_state": joint_state,
        "robot_state": {
            "q_rad": np.asarray(robot_q, dtype=np.float64).tolist(),
            "qd_rad_s": np.asarray(robot_qd, dtype=np.float64).tolist(),
        },
        "frozen_command": dict(frozen_command),
        "source_hashes": dict(source_hashes),
    }
    return validate_truth_snapshot(document)


def probe_restore_api(
    bindings: Mapping[str, Any],
    *,
    known_forbidden_apis: tuple[str, ...] = (),
) -> RestoreProbe:
    """Read-only API probe.  Fails closed if no supported restore mode exists."""
    required_root = (
        "plug_root_set_pose",
        "plug_root_set_linear_velocity",
        "plug_root_set_angular_velocity",
    )
    required_joint = ("nut_joint_set_state",)
    root = tuple(key for key in required_root if callable(bindings.get(key)))
    joint = tuple(key for key in required_joint if callable(bindings.get(key)))
    missing_root = tuple(key for key in required_root if key not in root)
    missing_joint = tuple(key for key in required_joint if key not in joint)
    forbidden = tuple(
        key
        for key in known_forbidden_apis
        if callable(bindings.get(key))
    )
    native = bindings.get("native_articulation_state")
    native_usable = native is not None and callable(
        getattr(native, "restore", None)
    )
    mode: str | None = None
    if all(key in root for key in required_root) and joint:
        mode = "ROOT_BODY_JOINT_STATE"
    elif native_usable:
        mode = "NATIVE_ARTICULATION_STATE"
    status = "READY" if mode is not None else "API_UNSUPPORTED"
    return RestoreProbe(
        status=status,
        mode=mode,
        available_apis=tuple(sorted(set(root + joint))),
        missing_apis=tuple(sorted(set(missing_root + missing_joint))),
        forbidden_apis_present=forbidden,
    )


def restore_snapshot(
    snapshot: Mapping[str, Any],
    bindings: Mapping[str, Any],
    *,
    known_forbidden_apis: tuple[str, ...] = (
        "nut_set_world_pose",
        "nut_set_linear_velocity",
        "nut_set_angular_velocity",
    ),
) -> dict[str, Any]:
    """Restore validated truth snapshot.  Linked-body pose writes are forbidden."""
    document = validate_truth_snapshot(snapshot)
    probe = probe_restore_api(
        bindings, known_forbidden_apis=known_forbidden_apis
    )
    if probe.forbidden_apis_present:
        raise RestoreRejected(
            "forbidden linked-body pose API is present: "
            + ",".join(probe.forbidden_apis_present)
        )
    if probe.status == "API_UNSUPPORTED":
        raise RestoreRejected("snapshot restore API is unsupported")
    if probe.mode == "NATIVE_ARTICULATION_STATE":
        bindings["native_articulation_state"].restore(document)
        return {"restore_status": "RESTORED", "mode": probe.mode}
    root = document["plug_root_state"]
    bindings["plug_root_set_pose"](
        root["position_m"], root["orientation_wxyz"]
    )
    bindings["plug_root_set_linear_velocity"](
        root["linear_velocity_m_s"]
    )
    bindings["plug_root_set_angular_velocity"](
        root["angular_velocity_rad_s"]
    )
    joint = document["coupling_nut_joint_state"]
    if joint is None:
        raise RestoreRejected(
            "ROOT_BODY_JOINT_STATE mode requires coupling_nut_joint_state"
        )
    bindings["nut_joint_set_state"](
        joint["q_rad"], joint["qd_rad_s"]
    )
    return {"restore_status": "RESTORED", "mode": probe.mode}


def capture_allowed_after_step(
    step_index: int,
    *,
    settle_steps: int = SETTLE_STEPS_DEFAULT,
) -> bool:
    if not isinstance(step_index, (int, float)) or not math.isfinite(step_index):
        raise SettleNotComplete("step_index must be finite")
    if int(step_index) < settle_steps:
        raise SettleNotComplete(
            f"capture requested at step {step_index}, settle requires {settle_steps}"
        )
    return True


def settle_tail_diagnostic(
    tail_samples: np.ndarray,
    reference: np.ndarray,
    noise_variance: np.ndarray,
) -> dict[str, Any]:
    """Record-only z-score diagnostic.  It never authorizes capture."""
    tail = np.asarray(tail_samples, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    noise = np.asarray(noise_variance, dtype=np.float64)
    if tail.ndim != 2 or ref.shape != (tail.shape[1],):
        raise SnapshotContractError("tail_samples/reference shape mismatch")
    if noise.shape != ref.shape or np.any(noise <= 0.0):
        raise SnapshotContractError("noise_variance must be positive per channel")
    mean = np.mean(tail, axis=0)
    variance = np.var(tail, axis=0) + noise
    z = (mean - ref) / np.sqrt(variance)
    return {
        "tail_steps": int(tail.shape[0]),
        "mean": mean.tolist(),
        "reference": ref.tolist(),
        "z_score": z.tolist(),
        "diagnostic_only": True,
        "pass_gate": None,
    }


def sha256_json(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document, allow_nan=False, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_truth_snapshot(path: Path | str, document: Mapping[str, Any]) -> None:
    validated = validate_truth_snapshot(document)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validated, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def load_truth_snapshot(
    path: Path | str, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    output = Path(path)
    text = output.read_text(encoding="utf-8")
    if expected_sha256 is not None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != expected_sha256:
            raise SnapshotContractError("truth snapshot SHA-256 mismatch")
    document = json.loads(text)
    return validate_truth_snapshot(document)


__all__ = [
    "POSTGRASP_SNAPSHOT_SCHEMA_VERSION",
    "TRUTH_ROLE",
    "SETTLE_STEPS_DEFAULT",
    "TAIL_STEPS_DEFAULT",
    "RestoreProbe",
    "RestoreRejected",
    "SettleNotComplete",
    "SnapshotContractError",
    "TruthFirewallViolation",
    "capture_allowed_after_step",
    "capture_truth_snapshot",
    "load_truth_snapshot",
    "probe_restore_api",
    "restore_snapshot",
    "settle_tail_diagnostic",
    "sha256_json",
    "validate_truth_snapshot",
    "write_truth_snapshot",
]
