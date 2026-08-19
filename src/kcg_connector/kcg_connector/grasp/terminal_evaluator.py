'''Log-only terminal evaluator snapshot for a failed formal lift.

After the formal lift gate first triggers, the controller is marked terminal:
no further lift command is allowed.  Before the sensor-only recovery starts,
the runner reads post-hoc truth exactly once and stores it through this pure
builder as a log-only snapshot.  The snapshot must never feed any robot/hand
command, recovery trajectory choice, or the PASS decision.

This module is intentionally pure (numpy + stdlib only): the simulator truth
read itself happens in the runner, which hands plain values to this builder.
'''

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _position(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be one finite 3-vector")
    return result


def _quaternion(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be one finite wxyz quaternion")
    if float(np.linalg.norm(result)) <= 0.0:
        raise ValueError(f"{label} must have non-zero norm")
    return result / float(np.linalg.norm(result))


def _transform_from_pose(
    position: Sequence[float], quaternion_wxyz: Sequence[float]
) -> np.ndarray:
    w, x, y, z = _quaternion(quaternion_wxyz, "quaternion")
    translation = _position(position, "position")
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
             2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
             1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def build_terminal_snapshot(
    *,
    reason: str,
    global_step: int,
    phase: str,
    plug_body_position_world: Sequence[float],
    plug_body_orientation_world: Sequence[float],
    nut_position_world: Sequence[float],
    nut_orientation_world: Sequence[float],
    hand_position_world: Sequence[float],
    hand_orientation_world: Sequence[float],
    settled_plug_position_world: Sequence[float],
    settled_plug_orientation_world: Sequence[float],
    settled_plug_z_m: float,
    lift_started_dz_m: float,
    contact_audit: Mapping[str, Any],
) -> dict[str, Any]:
    '''Build one log-only truth snapshot sampled with the controller terminal.

    All derivations (T_hand_plug, relative settled XYZ/yaw, lift-started flag)
    are computed here from the plain pose values the runner read once.  The
    returned dict carries ``posthoc_truth_evaluation_only=True`` and must only
    ever be written into the report/logs.
    '''
    if not isinstance(global_step, bool) and int(global_step) == global_step:
        global_step = int(global_step)
    else:
        raise ValueError("global_step must be integral")
    plug_body_position = _position(
        plug_body_position_world, "plug_body_position_world"
    )
    settled_position = _position(
        settled_plug_position_world, "settled_plug_position_world"
    )
    hand_position = _position(hand_position_world, "hand_position_world")
    plug_transform = _transform_from_pose(
        plug_body_position, plug_body_orientation_world
    )
    hand_transform = _transform_from_pose(
        hand_position, hand_orientation_world
    )
    t_hand_plug = np.linalg.inv(hand_transform) @ plug_transform
    relative_settled_xyz = plug_body_position - settled_position
    plug_yaw = _yaw_of(_quaternion(plug_body_orientation_world, "plug yaw"))
    settled_yaw = _yaw_of(
        _quaternion(settled_plug_orientation_world, "settled yaw")
    )
    relative_settled_yaw = math.atan2(
        math.sin(plug_yaw - settled_yaw),
        math.cos(plug_yaw - settled_yaw),
    )
    plug_z = float(plug_body_position[2])
    if not math.isfinite(settled_plug_z_m):
        raise ValueError("settled_plug_z_m must be finite")
    if (
        isinstance(lift_started_dz_m, bool)
        or not math.isfinite(float(lift_started_dz_m))
        or float(lift_started_dz_m) < 0.0
    ):
        raise ValueError("lift_started_dz_m must be finite and non-negative")
    plug_lift_started = bool(
        plug_z - float(settled_plug_z_m) >= float(lift_started_dz_m)
    )
    if not isinstance(contact_audit, Mapping):
        raise ValueError("contact_audit must be a mapping")
    return {
        "posthoc_truth_evaluation_only": True,
        "sampled_with_controller_terminal": True,
        "controller_terminal": True,
        "snapshot_kind": "terminal_evaluator_log_only",
        "reason": str(reason),
        "global_step": global_step,
        "phase": str(phase),
        "plug_body_pose_world": {
            "position_m": [float(value) for value in plug_body_position],
            "orientation_wxyz": [
                float(value)
                for value in _quaternion(
                    plug_body_orientation_world, "plug orientation"
                )
            ],
        },
        "nut_pose_world": {
            "position_m": [
                float(value)
                for value in _position(
                    nut_position_world, "nut_position_world"
                )
            ],
            "orientation_wxyz": [
                float(value)
                for value in _quaternion(
                    nut_orientation_world, "nut orientation"
                )
            ],
        },
        "hand_pose_world": {
            "position_m": [float(value) for value in hand_position],
            "orientation_wxyz": [
                float(value)
                for value in _quaternion(
                    hand_orientation_world, "hand orientation"
                )
            ],
        },
        "t_hand_plug": [
            [float(value) for value in row] for row in t_hand_plug
        ],
        "relative_settled_plug": {
            "xyz_m": [float(value) for value in relative_settled_xyz],
            "yaw_rad": relative_settled_yaw,
        },
        "plug_z_m": plug_z,
        "settled_plug_z_m": float(settled_plug_z_m),
        "plug_lift_started": plug_lift_started,
        "lift_started_dz_m": float(lift_started_dz_m),
        "episode_terminal_contact_audit": dict(contact_audit),
        "sinks": "report_log_only",
        "consumed_by": [],
    }


def _yaw_of(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
