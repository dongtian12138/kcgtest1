"""Post-grasp snapshot save/restore/settle runtime hook.

The PlugBody and CouplingNut are separate rigid bodies connected by a plain
USD revolute joint, not an articulation. Snapshot restore therefore writes
both rigid-body states once at an explicit initialization boundary. No object
pose write is permitted in the following settle window.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kcg_connector.postgrasp_snapshot_gate import (
    REPLAY_BUNDLE_MANIFEST_FILENAME,
    REPLAY_STAGE_FILENAME,
    build_replay_bundle_manifest,
    capture_snapshot_gate_document,
    evaluate_restore_consistency,
    load_replay_bundle_manifest,
    write_replay_bundle_manifest,
    write_snapshot_gate_document,
)


def _method_or_none(obj, name):
    value = getattr(obj, name, None)
    return value if callable(value) else None


def _required_method(obj, name):
    method = _method_or_none(obj, name)
    if method is None:
        raise RuntimeError(f"required restore API unavailable: {name}")
    return method


def _read_live_wrist(get_latest_wrist_state, previous_step):
    state = get_latest_wrist_state()
    if not isinstance(state, Mapping):
        raise RuntimeError("live wrist getter did not return mapping")
    step = state.get("global_step")
    canonical = np.asarray(state.get("canonical"), dtype=np.float64).ravel()
    error = state.get("error")
    if error is not None or canonical.shape != (6,) or not np.all(
        np.isfinite(canonical)
    ):
        raise RuntimeError("live wrist state invalid")
    if not isinstance(step, (int, float)) or int(step) <= int(previous_step):
        raise RuntimeError("live wrist state stale")
    return int(step), canonical


def _restore_rigid_body(rigid_body, state):
    _required_method(rigid_body, "set_world_pose")(
        position=np.asarray(state["position_m"], dtype=np.float64),
        orientation=np.asarray(state["orientation_wxyz"], dtype=np.float64),
    )
    _required_method(rigid_body, "set_linear_velocity")(
        np.asarray(state["linear_velocity_m_s"], dtype=np.float64)
    )
    _required_method(rigid_body, "set_angular_velocity")(
        np.asarray(state["angular_velocity_rad_s"], dtype=np.float64)
    )


def _sha256_existing_artifact(path, label):
    artifact = Path(path)
    if not artifact.is_file():
        raise RuntimeError(f"{label} artifact missing: {artifact}")
    if artifact.stat().st_size <= 0:
        raise RuntimeError(f"{label} artifact empty: {artifact}")
    return hashlib.sha256(artifact.read_bytes()).hexdigest()


def _source_hash_mapping(source_hashes):
    """Keep only string-valued provenance entries for the bundle manifest.

    The runner-level provenance also carries scalar metadata such as ``seed``
    and null audit-mode placeholders.  The replay manifest only pins actual
    source/config digests, so those non-string entries are intentionally
    omitted from the manifest rather than aborting an otherwise valid bundle.
    """
    if source_hashes is None:
        return {}
    if not isinstance(source_hashes, Mapping):
        raise RuntimeError("source_hashes must be a mapping")
    return {
        key: value
        for key, value in source_hashes.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def run_postgrasp_snapshot_gate(
    *,
    arguments,
    global_step,
    physics_step,
    body,
    nut,
    robot,
    hand_indices,
    robot_set_q,
    robot_set_qd,
    current_arm_target,
    current_hand_target,
    observe_and_step,
    sample_post_tare_efforts,
    formal_wrist_payload_reference,
    get_latest_wrist_state,
    tare_efforts,
    stage_exporter=None,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output_dir = (
        Path(arguments.output_dir).expanduser().resolve()
        / "postgrasp_snapshot_gate"
    )
    result = {
        "requested": True,
        "snapshot_restore_verified": False,
        "control_authorized": False,
        "formal_estimator_input": False,
        "restore_scope": "INITIALIZATION_ONLY_NOT_FORMAL_CONTROL",
        "object_pose_writes_at_restore_boundary": 0,
        "object_pose_writes_during_settle": 0,
        "status": "STARTED",
    }
    try:
        pinned_source_hashes = _source_hash_mapping(source_hashes)
    except Exception as exception:
        result["status"] = "SNAPSHOT_GATE_ABORT_SAFE"
        result["error"] = f"{type(exception).__name__}: {exception}"
        return result
    if output_dir.exists():
        result["status"] = "FAIL_CLOSED_OUTPUT_EXISTS"
        return result
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        if not callable(robot_set_q) or not callable(robot_set_qd):
            raise RuntimeError("robot q/qd restore APIs unavailable")

        snapshot_wrist_step, snapshot_wrist = _read_live_wrist(
            get_latest_wrist_state, -1
        )
        if snapshot_wrist_step != int(global_step):
            raise RuntimeError(
                "live wrist sample is not synchronized with snapshot global_step"
            )
        timestamp = datetime.now(timezone.utc).isoformat()
        plug_position, plug_orientation = body.get_world_pose()
        nut_position, nut_orientation = nut.get_world_pose()
        robot_q = np.asarray(robot.get_joint_positions(), dtype=np.float64).ravel()
        robot_qd = np.asarray(robot.get_joint_velocities(), dtype=np.float64).ravel()
        hand_indices_array = np.asarray(hand_indices, dtype=np.int64)
        finger_q_actual = robot_q[hand_indices_array]
        tare_array = np.asarray(tare_efforts, dtype=np.float64).ravel()
        if tare_array.shape != (3,) or not np.all(np.isfinite(tare_array)):
            raise RuntimeError("finger root tare efforts must be 3 finite channels")
        measured_torque = (
            np.asarray(sample_post_tare_efforts(), dtype=np.float64).ravel()
            - tare_array
        )
        if measured_torque.shape != (3,):
            raise RuntimeError("finger root torque proxy must contain 3 signals")
        document = capture_snapshot_gate_document(
            snapshot_id=f"snap_gate_{arguments.seed:06d}",
            timestamp_utc=timestamp,
            episode=f"seed{arguments.seed:06d}",
            seed=int(arguments.seed),
            global_step=int(global_step),
            physics_step=int(physics_step),
            robot_getters={"q": lambda: robot_q, "qd": lambda: robot_qd},
            finger_getters={
                "hand_q_actual": lambda: finger_q_actual,
                "hand_q_target": lambda: np.asarray(current_hand_target),
                "finger_root_torque_proxy": lambda: measured_torque,
                "finger_root_tare_efforts": lambda: tare_array,
            },
            plug_root_getters={
                "position": lambda: np.asarray(plug_position),
                "orientation": lambda: np.asarray(plug_orientation),
                "linear_velocity": lambda: np.asarray(body.get_linear_velocity()),
                "angular_velocity": lambda: np.asarray(body.get_angular_velocity()),
            },
            nut_joint_getters=None,
            nut_posthoc_getters={
                "position": lambda: np.asarray(nut_position),
                "orientation": lambda: np.asarray(nut_orientation),
                "linear_velocity": lambda: np.asarray(nut.get_linear_velocity()),
                "angular_velocity": lambda: np.asarray(nut.get_angular_velocity()),
            },
            frozen_command={
                "arm_q_target_rad": np.asarray(current_arm_target).tolist(),
                "hand_q_target_rad": np.asarray(current_hand_target).tolist(),
                "wrist_ft_payload_reference": np.asarray(
                    formal_wrist_payload_reference
                ).ravel().tolist(),
                "wrist_ft_snapshot_reference": snapshot_wrist.tolist(),
                "wrist_ft_snapshot_global_step": snapshot_wrist_step,
            },
            provenance=dict(pinned_source_hashes),
        )
        write_snapshot_gate_document(output_dir / "snapshot_gate.json", document)

        # Replay stage export happens right after snapshot sampling, before
        # any restore or settle write, so the exported scene matches the
        # snapshot moment.  The stage is exported into the gate directory
        # ONLY for a future fresh replay process; no stage or object truth
        # value is handed to control or to any estimator here.
        if not callable(stage_exporter):
            raise RuntimeError("stage exporter unavailable: replay stage missing")
        stage_path = output_dir / REPLAY_STAGE_FILENAME
        if not stage_exporter(stage_path):
            raise RuntimeError("stage exporter reported failure")
        snapshot_sha256 = _sha256_existing_artifact(
            output_dir / "snapshot_gate.json", "snapshot"
        )
        stage_sha256 = _sha256_existing_artifact(stage_path, "stage")
        manifest = build_replay_bundle_manifest(
            seed=int(arguments.seed),
            snapshot_file="snapshot_gate.json",
            snapshot_sha256=snapshot_sha256,
            stage_file=REPLAY_STAGE_FILENAME,
            stage_sha256=stage_sha256,
            source_hashes=pinned_source_hashes,
        )
        write_replay_bundle_manifest(
            output_dir / REPLAY_BUNDLE_MANIFEST_FILENAME, manifest
        )
        # Self-verify: the loader re-hashes both artifacts, so any missing,
        # empty, or tampered bundle file fails this gate closed.
        load_replay_bundle_manifest(output_dir / REPLAY_BUNDLE_MANIFEST_FILENAME)
        result["replay_bundle"] = {
            "manifest": REPLAY_BUNDLE_MANIFEST_FILENAME,
            "snapshot_file": "snapshot_gate.json",
            "snapshot_sha256": snapshot_sha256,
            "stage_file": REPLAY_STAGE_FILENAME,
            "stage_sha256": stage_sha256,
            "restore_truth_scope": "INITIALIZATION_ONLY",
            "formal_estimator_input": False,
            "control_authorized": False,
            "stage_export_scope": "REPLAY_ONLY_NOT_CONTROL_OR_ESTIMATOR",
        }

        # Explicit diagnostic initialization boundary. These two writes are
        # counted and never occur in the following settle loop.
        robot_set_q(robot_q)
        robot_set_qd(robot_qd)
        _restore_rigid_body(body, document["plug_root_state"])
        _restore_rigid_body(nut, document["nut_rigid_state_restore_only"])
        result["object_pose_writes_at_restore_boundary"] = 2

        tails = {"q": [], "qd": [], "torque": [], "wrench": []}
        previous_wrist_step = snapshot_wrist_step
        for step in range(120):
            positions, velocities = observe_and_step(
                current_arm_target, current_hand_target, True
            )
            measured = (
                np.asarray(sample_post_tare_efforts(), dtype=np.float64).ravel()
                - tare_array
            )
            if measured.shape != (3,):
                raise RuntimeError("finger root torque proxy must contain 3 signals")
            previous_wrist_step, wrist_canonical = _read_live_wrist(
                get_latest_wrist_state, previous_wrist_step
            )
            if step >= 60:
                tails["q"].append(np.asarray(positions, dtype=np.float64))
                tails["qd"].append(np.asarray(velocities, dtype=np.float64))
                tails["torque"].append(measured)
                tails["wrench"].append(wrist_canonical)

        consistency = evaluate_restore_consistency(
            snapshot=document,
            tail_robot_q=np.asarray(tails["q"]),
            tail_robot_qd=np.asarray(tails["qd"]),
            tail_finger_torque=np.asarray(tails["torque"]),
            tail_wrist_wrench=np.asarray(tails["wrench"]),
            restored_plug_position=np.asarray(body.get_world_pose()[0]),
            restored_plug_orientation=np.asarray(body.get_world_pose()[1]),
            restored_nut_position=np.asarray(nut.get_world_pose()[0]),
            restored_nut_orientation=np.asarray(nut.get_world_pose()[1]),
        )
        result["consistency"] = {
            "verified": consistency.verified,
            "reasons": list(consistency.reasons),
            "diagnostics": consistency.diagnostics,
        }
        result["snapshot_restore_verified"] = consistency.verified
        result["status"] = (
            "SNAPSHOT_GATE_VERIFIED"
            if consistency.verified
            else "SNAPSHOT_GATE_VERIFICATION_FAILED"
        )
    except Exception as exception:
        result["status"] = "SNAPSHOT_GATE_ABORT_SAFE"
        result["error"] = f"{type(exception).__name__}: {exception}"
    return result


__all__ = ["run_postgrasp_snapshot_gate"]
