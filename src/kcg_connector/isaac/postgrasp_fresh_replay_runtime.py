"""CPU-testable fresh-process replay orchestrator.

The real Isaac shell opens a hash-pinned ``replay_stage.usda`` and supplies
robot/body/nut bindings.  This module only owns the strict sequence and
evidence bookkeeping:

1. load the replay bundle manifest and verify both declared artifact SHA-256;
2. reject legacy snapshots that do not carry the three-channel finger tare;
3. restore robot q/qd plus exactly two rigid-body object states;
4. settle for 120 physics steps with frozen commands and zero object writes;
5. evaluate the same restore-consistency contract as the in-process gate.

No Isaac import lives here, so the sequence is CPU-testable with mocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kcg_connector.postgrasp_snapshot_gate import (
    SnapshotContractError,
    evaluate_restore_consistency,
    load_replay_bundle_manifest,
    load_snapshot_gate_document,
)

FRESH_REPLAY_SCHEMA_VERSION = "kcg_d38999_fresh_replay_v1"
FRESH_REPLAY_RESULT_MARKER = "ISAAC D38999 FRESH REPLAY RESTORE V1 "
SETTLE_STEPS = 120
TAIL_STEPS = 60


def load_fresh_replay_bundle(
    manifest_path: Path | str,
    *,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Return the verified manifest, strict snapshot and stage path.

    ``load_replay_bundle_manifest`` already re-hashes both artifacts and now
    rejects legacy snapshots lacking ``finger_root_tare_efforts_nm``.  This
    wrapper pins the bundle seed and makes the three paths explicit for the
    GPU shell.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_replay_bundle_manifest(manifest_path)
    base = manifest_path.parent
    snapshot = load_snapshot_gate_document(
        base / manifest["snapshot_file"]
    )
    if int(snapshot["seed"]) != int(manifest["seed"]):
        raise SnapshotContractError(
            "replay bundle seed does not match snapshot seed"
        )
    if expected_seed is not None and int(manifest["seed"]) != int(
        expected_seed
    ):
        raise SnapshotContractError(
            "replay bundle seed does not match requested seed"
        )
    stage_path = (base / manifest["stage_file"]).resolve()
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "snapshot": snapshot,
        "snapshot_path": base / manifest["snapshot_file"],
        "stage_path": stage_path,
    }


def _finite_3(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise RuntimeError(f"{label} must be a finite 3-channel array")
    return array


def _wrist_state(state: Any, previous_step: int) -> tuple[int, np.ndarray]:
    if not isinstance(state, Mapping):
        raise RuntimeError("live wrist getter did not return a mapping")
    step = state.get("global_step")
    canonical = np.asarray(state.get("canonical"), dtype=np.float64).ravel()
    error = state.get("error")
    if error is not None or canonical.shape != (6,) or not np.all(
        np.isfinite(canonical)
    ):
        raise RuntimeError("live wrist state is invalid")
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, np.integer))
        or int(step) <= int(previous_step)
    ):
        raise RuntimeError("live wrist state is stale or non-integral")
    return int(step), canonical


def _verified_stage_path(stage_path: Path | str) -> Path:
    path = Path(stage_path)
    if not path.is_file():
        raise RuntimeError(f"replay stage missing: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"replay stage empty: {path}")
    return path


def _posthoc_value(value: Any, label: str) -> np.ndarray:
    """Accept either a value or a zero-arg getter for posthoc pose data."""
    resolved = value() if callable(value) else value
    array = np.asarray(resolved, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{label} is non-finite")
    return array


def _pose_write_count(counter: Any) -> int:
    value = getattr(counter, "value", None)
    if value is None:
        value = counter() if callable(counter) else counter
    result = int(value)
    if result < 0:
        raise RuntimeError("object pose write counter is negative")
    return result


def run_fresh_replay_restore_settle(
    *,
    snapshot: Mapping[str, Any],
    stage_path: Path | str,
    open_stage,
    reset_world,
    robot_set_q,
    robot_set_qd,
    restore_rigid_body,
    observe_and_step,
    sample_finger_torque_proxy,
    get_latest_wrist_state,
    restored_plug_position,
    restored_plug_orientation,
    restored_nut_position,
    restored_nut_orientation,
    object_write_counter=None,
    settle_steps: int = SETTLE_STEPS,
) -> dict[str, Any]:
    """Run the restore boundary and frozen-command settle window.

    ``restore_rigid_body(rigid_body, state)`` is called exactly twice.  The
    caller-provided counter must therefore increase by exactly two after the
    boundary and remain unchanged throughout the settle loop.
    """
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("fresh replay snapshot must be a mapping")
    for callable_label, value in (
        ("open_stage", open_stage),
        ("reset_world", reset_world),
        ("robot_set_q", robot_set_q),
        ("robot_set_qd", robot_set_qd),
        ("restore_rigid_body", restore_rigid_body),
        ("observe_and_step", observe_and_step),
        ("sample_finger_torque_proxy", sample_finger_torque_proxy),
        ("get_latest_wrist_state", get_latest_wrist_state),
    ):
        if not callable(value):
            raise RuntimeError(
                f"fresh replay binding unavailable: {callable_label}"
            )
    settle_steps = int(settle_steps)
    if settle_steps < 1:
        raise RuntimeError("settle_steps must be positive")
    result = {
        "schema_version": FRESH_REPLAY_SCHEMA_VERSION,
        "mode": "FRESH_PROCESS_SNAPSHOT_REPLAY",
        "restore_boundary_object_pose_writes": 0,
        "object_pose_writes_after_restore": 0,
        "settle_steps": settle_steps,
        "control_authorized": False,
        "formal_estimator_input": False,
        "restore_truth_scope": "INITIALIZATION_ONLY",
        "posthoc_truth_used_for_consistency_only": True,
        "status": "STARTED",
    }
    try:
        _verified_stage_path(stage_path)
        if open_stage(stage_path) is not True:
            raise RuntimeError("stage open call did not confirm success")
        reset_world()

        frozen_arm_target = np.asarray(
            snapshot["frozen_command"]["arm_q_target_rad"],
            dtype=np.float64,
        ).ravel()
        frozen_hand_target = np.asarray(
            snapshot["frozen_command"]["hand_q_target_rad"],
            dtype=np.float64,
        ).ravel()
        if frozen_arm_target.shape != (7,) or not np.all(
            np.isfinite(frozen_arm_target)
        ):
            raise RuntimeError("frozen arm target is invalid")
        if frozen_hand_target.shape != (4,) or not np.all(
            np.isfinite(frozen_hand_target)
        ):
            raise RuntimeError("frozen hand target is invalid")

        robot_q = np.asarray(
            snapshot["robot_state"]["q_rad"], dtype=np.float64
        ).ravel()
        robot_qd = np.asarray(
            snapshot["robot_state"]["qd_rad_s"], dtype=np.float64
        ).ravel()
        if robot_q.shape != (15,) or robot_qd.shape != (15,):
            raise RuntimeError("restored robot state must have 15 DOF")
        if not np.all(np.isfinite(robot_q)) or not np.all(
            np.isfinite(robot_qd)
        ):
            raise RuntimeError("restored robot state is non-finite")

        robot_set_q(robot_q)
        robot_set_qd(robot_qd)

        writes_at_boundary_start = _pose_write_count(object_write_counter)
        restore_rigid_body(
            "plug",
            snapshot["plug_root_state"],
        )
        restore_rigid_body(
            "nut",
            snapshot["nut_rigid_state_restore_only"],
        )
        writes_at_boundary_end = _pose_write_count(object_write_counter)
        boundary_writes = writes_at_boundary_end - writes_at_boundary_start
        if boundary_writes != 2:
            raise RuntimeError(
                "restore boundary must write exactly two object poses; "
                f"observed {boundary_writes}"
            )
        result["restore_boundary_object_pose_writes"] = boundary_writes

        tails = {"q": [], "qd": [], "torque": [], "wrench": []}
        previous_wrist_step = -1
        tail_start = settle_steps - TAIL_STEPS
        if tail_start < 0:
            tail_start = 0
        for step_index in range(settle_steps):
            positions, velocities = observe_and_step(
                frozen_arm_target, frozen_hand_target
            )
            torque = _finite_3(
                sample_finger_torque_proxy(),
                "finger root torque proxy",
            )
            wrist_step, wrist_canonical = _wrist_state(
                get_latest_wrist_state(), previous_wrist_step
            )
            previous_wrist_step = wrist_step
            if step_index >= tail_start:
                tails["q"].append(
                    np.asarray(positions, dtype=np.float64).ravel()
                )
                tails["qd"].append(
                    np.asarray(velocities, dtype=np.float64).ravel()
                )
                tails["torque"].append(torque)
                tails["wrench"].append(wrist_canonical)

        writes_after_settle = _pose_write_count(object_write_counter)
        extra_writes = writes_after_settle - writes_at_boundary_end
        result["object_pose_writes_after_restore"] = extra_writes
        if extra_writes != 0:
            raise RuntimeError(
                "object pose writes observed after restore boundary: "
                f"{extra_writes}"
            )

        consistency = evaluate_restore_consistency(
            snapshot=snapshot,
            tail_robot_q=np.asarray(tails["q"]),
            tail_robot_qd=np.asarray(tails["qd"]),
            tail_finger_torque=np.asarray(tails["torque"]),
            tail_wrist_wrench=np.asarray(tails["wrench"]),
            restored_plug_position=_posthoc_value(
                restored_plug_position, "restored_plug_position"
            ),
            restored_plug_orientation=_posthoc_value(
                restored_plug_orientation, "restored_plug_orientation"
            ),
            restored_nut_position=_posthoc_value(
                restored_nut_position, "restored_nut_position"
            ),
            restored_nut_orientation=_posthoc_value(
                restored_nut_orientation, "restored_nut_orientation"
            ),
        )
        result["consistency"] = {
            "verified": consistency.verified,
            "reasons": list(consistency.reasons),
            "diagnostics": consistency.diagnostics,
        }
        result["snapshot_restore_verified"] = consistency.verified
        result["status"] = (
            "FRESH_REPLAY_RESTORE_VERIFIED"
            if consistency.verified
            else "FRESH_REPLAY_RESTORE_INCONSISTENT"
        )
    except Exception as exception:
        result["status"] = "FRESH_REPLAY_ABORT_SAFE"
        result["snapshot_restore_verified"] = False
        result["error"] = f"{type(exception).__name__}: {exception}"
    return result


def snapshot_path_sha(path: Path | str) -> str:
    """Legacy compatibility helper; new callers use the bundle manifest."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "FRESH_REPLAY_RESULT_MARKER",
    "FRESH_REPLAY_SCHEMA_VERSION",
    "SETTLE_STEPS",
    "TAIL_STEPS",
    "load_fresh_replay_bundle",
    "run_fresh_replay_restore_settle",
    "snapshot_path_sha",
]
