"""CPU-testable diagnostics for post-grasp display motion.

The display phase is diagnostic-only.  This module provides:

- frozen formal wrist gates (8 N force-delta norm and 0.30 N*m
  three-component moment decomposition) which must always be evaluated;
- an additional, strictly more conservative EMA-residual candidate gate;
- a bounded in-memory trace ring buffer and atomic JSONL writer;
- a path-quality screen for Cartesian TCP waypoints.

No Isaac import lives here.
"""

from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.grasp_stability_monitor import (
    evaluate_wrist_moment_safety,
)

TRACE_CAPACITY = 240
FORMAL_MAX_WRIST_FORCE_N = 8.0
FORMAL_MAX_WRIST_MOMENT_NM = 0.30
EMA_FORCE_RESIDUAL_N = 4.0
EMA_MOMENT_RESIDUAL_NM = 0.50
EMA_ALPHA = 0.02
DIAGNOSTIC_SCHEMA_VERSION = "kcg_d38999_display_motion_diagnostics_v1"


def evaluate_display_wrist_evidence(
    *,
    current_wrench: Sequence[float],
    reference_wrench: Sequence[float],
    previous_raw_wrench: Sequence[float] | None,
    ema_wrench: Sequence[float] | None,
    alpha: float = EMA_ALPHA,
    formal_force_limit_n: float = FORMAL_MAX_WRIST_FORCE_N,
    formal_moment_limit_nm: float = FORMAL_MAX_WRIST_MOMENT_NM,
    ema_force_limit_n: float = EMA_FORCE_RESIDUAL_N,
    ema_moment_limit_nm: float = EMA_MOMENT_RESIDUAL_NM,
) -> dict[str, Any]:
    """Evaluate formal and diagnostic wrist evidence for one physics step.

    The returned ``formal_gate_triggered`` uses only the frozen formal limits.
    The EMA candidate gate is returned separately and is never allowed to
    replace the formal gate.
    """
    current = np.asarray(current_wrench, dtype=np.float64).ravel()
    reference = np.asarray(reference_wrench, dtype=np.float64).ravel()
    if current.shape != (6,) or reference.shape != (6,):
        raise ValueError("wrench inputs must be 6-vectors")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(reference)):
        raise ValueError("wrench inputs must be finite")

    force_delta = current[:3] - reference[:3]
    force_increment_n = float(np.linalg.norm(force_delta))
    force_formal_triggered = bool(force_increment_n > float(formal_force_limit_n))
    moment_evidence = evaluate_wrist_moment_safety(
        current[3:].tolist(),
        reference[3:].tolist(),
        float(formal_moment_limit_nm),
    )
    moment_formal_triggered = bool(moment_evidence["triggered"])
    formal_gate_triggered = bool(
        force_formal_triggered or moment_formal_triggered
    )

    evidence: dict[str, Any] = {
        "force_increment_n": force_increment_n,
        "force_formal_limit_n": float(formal_force_limit_n),
        "force_formal_triggered": force_formal_triggered,
        "moment_evidence": moment_evidence,
        "moment_formal_triggered": moment_formal_triggered,
        "formal_gate_triggered": formal_gate_triggered,
    }

    if previous_raw_wrench is not None:
        previous = np.asarray(previous_raw_wrench, dtype=np.float64).ravel()
        if previous.shape != (6,):
            raise ValueError("previous raw wrench must be a 6-vector")
        adjacent = current - previous
        evidence["adjacent_raw_force_delta_n"] = float(
            np.linalg.norm(adjacent[:3])
        )
        evidence["adjacent_raw_moment_delta_nm"] = float(
            np.linalg.norm(adjacent[3:])
        )

    if ema_wrench is None:
        ema = current.copy()
    else:
        ema = np.asarray(ema_wrench, dtype=np.float64).ravel()
        if ema.shape != (6,):
            raise ValueError("ema wrench must be a 6-vector")
    residual = current - ema
    ema_next = float(alpha) * current + (1.0 - float(alpha)) * ema
    ema_force_residual_n = float(np.linalg.norm(residual[:3]))
    ema_moment_residual_nm = float(np.linalg.norm(residual[3:]))
    ema_force_triggered = bool(
        ema_force_residual_n > float(ema_force_limit_n)
    )
    ema_moment_triggered = bool(
        ema_moment_residual_nm > float(ema_moment_limit_nm)
    )
    evidence.update(
        {
            "ema_wrench": ema_next.tolist(),
            "ema_force_residual_n": ema_force_residual_n,
            "ema_moment_residual_nm": ema_moment_residual_nm,
            "ema_force_limit_n": float(ema_force_limit_n),
            "ema_moment_limit_nm": float(ema_moment_limit_nm),
            "ema_force_triggered": ema_force_triggered,
            "ema_moment_triggered": ema_moment_triggered,
            "ema_candidate_triggered": bool(
                ema_force_triggered or ema_moment_triggered
            ),
        }
    )

    triggered = []
    if force_formal_triggered:
        triggered.append("formal_force")
    if moment_formal_triggered:
        triggered.append(
            "formal_moment:"
            + str(moment_evidence.get("trigger_component"))
        )
    if ema_force_triggered:
        triggered.append("ema_force_candidate")
    if ema_moment_triggered:
        triggered.append("ema_moment_candidate")
    evidence["triggered_gates"] = triggered
    evidence["any_gate_triggered"] = bool(triggered)
    return evidence


class DisplayMotionRingBuffer:
    """Bounded step trace with atomic JSONL output."""

    def __init__(self, capacity: int = TRACE_CAPACITY):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        self._records: deque[dict[str, Any]] = deque(maxlen=self.capacity)

    def append(self, record: Mapping[str, Any]) -> None:
        self._records.append(dict(record))

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def write_jsonl(self, path: Path | str) -> None:
        atomic_write_json_lines(path, self._records)


def atomic_write_json_lines(path: Path | str, records: Iterable[Mapping[str, Any]]):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, allow_nan=False, ensure_ascii=False)
                + "\n"
            )
    os.replace(temporary, output)


def atomic_write_json(path: Path | str, document: Mapping[str, Any]):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(document, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    cosine = max(-1.0, min(1.0, (float(np.trace(matrix)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    skew = np.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ),
        dtype=np.float64,
    )
    if angle < 1.0e-9:
        return 0.5 * skew
    return angle * skew / (2.0 * math.sin(angle))


def numeric_tcp_jacobian(joints: Sequence[float], forward_kinematics) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64).ravel()
    if joints.shape != (7,):
        raise ValueError("joints must be a 7-vector")
    epsilon = 1.0e-6
    jacobian = np.zeros((6, 6), dtype=np.float64)
    for index in range(6):
        plus = joints.copy()
        minus = joints.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_transform = np.asarray(
            forward_kinematics(tuple(float(value) for value in plus)),
            dtype=np.float64,
        )
        minus_transform = np.asarray(
            forward_kinematics(tuple(float(value) for value in minus)),
            dtype=np.float64,
        )
        jacobian[:3, index] = (
            plus_transform[:3, 3] - minus_transform[:3, 3]
        ) / (2.0 * epsilon)
        jacobian[3:, index] = _rotation_vector(
            plus_transform[:3, :3] @ minus_transform[:3, :3].T
        ) / (2.0 * epsilon)
    return jacobian


def tcp_clearance_screen(
    tcp_position: Sequence[float],
    *,
    table_top_z_m: float,
    fixture_center_m: Sequence[float],
    fixture_half_extent_m: Sequence[float],
) -> dict[str, Any]:
    position = np.asarray(tcp_position, dtype=np.float64).ravel()
    if position.shape != (3,):
        raise ValueError("tcp_position must be a 3-vector")
    center = np.asarray(fixture_center_m, dtype=np.float64).ravel()
    half = np.asarray(fixture_half_extent_m, dtype=np.float64).ravel()
    if center.shape != (3,) or half.shape != (3,):
        raise ValueError("fixture inputs must be 3-vectors")
    above_table_m = float(position[2] - float(table_top_z_m))
    separation = np.maximum(np.abs(position - center) - half, 0.0)
    fixture_clearance_m = float(np.linalg.norm(separation))
    return {
        "tcp_position_m": position.tolist(),
        "above_table_m": above_table_m,
        "fixture_clearance_m": fixture_clearance_m,
    }


def joint_target_limit_violations(
    target_q: Sequence[float],
    joint_limits: Sequence[tuple[float, float]],
    *,
    margin_rad: float = 0.010,
) -> list[dict[str, Any]]:
    """Return all 7-DOF joint-limit violations for one commanded target."""
    target = np.asarray(target_q, dtype=np.float64).ravel()
    if target.shape != (7,):
        raise ValueError("target_q must be a 7-vector")
    limits = list(joint_limits)
    if len(limits) != 7:
        raise ValueError("joint_limits must contain 7 (lower, upper) pairs")
    if float(margin_rad) < 0.0:
        raise ValueError("joint limit margin must be non-negative")
    violations = []
    for index, (lower, upper) in enumerate(limits):
        lower = float(lower)
        upper = float(upper)
        if upper <= lower:
            raise ValueError(f"joint {index} limits are invalid")
        value = float(target[index])
        if value > upper - float(margin_rad):
            violations.append(
                {
                    "joint_index": index,
                    "target_rad": value,
                    "upper_limit_rad": upper,
                    "margin_rad": float(margin_rad),
                    "effective_upper_rad": upper - float(margin_rad),
                }
            )
        if value < lower + float(margin_rad):
            violations.append(
                {
                    "joint_index": index,
                    "target_rad": value,
                    "lower_limit_rad": lower,
                    "margin_rad": float(margin_rad),
                    "effective_lower_rad": lower + float(margin_rad),
                }
            )
    return violations


def evaluate_display_sensor_gates(
    *,
    desired_arm_q: Sequence[float],
    actual_q: Sequence[float],
    velocities: Sequence[float],
    torque: Sequence[float],
    joint_limits: Sequence[tuple[float, float]] | None = None,
    joint_limit_margin_rad: float = 0.010,
    max_abs_torque_nm: float = 2.0,
    max_joint_speed_rad_s: float = 1.0,
    max_arm_tracking_error_rad: float = 0.030,
) -> dict[str, Any]:
    """Evaluate non-wrist display safety gates, including arm tracking."""
    desired = np.asarray(desired_arm_q, dtype=np.float64).ravel()
    actual = np.asarray(actual_q, dtype=np.float64).ravel()
    velocity = np.asarray(velocities, dtype=np.float64).ravel()
    finger_torque = np.asarray(torque, dtype=np.float64).ravel()
    if desired.shape != (7,) or actual.shape != (7,):
        raise ValueError("desired_arm_q and actual_q must be 7-vectors")
    if finger_torque.shape != (3,):
        raise ValueError("torque must be a 3-vector")
    if velocity.size == 0:
        raise ValueError("velocities must not be empty")
    if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(actual)):
        return {"ok": False, "reasons": ["nonfinite_joint_state"]}
    if not np.all(np.isfinite(velocity)) or not np.all(
        np.isfinite(finger_torque)
    ):
        return {"ok": False, "reasons": ["nonfinite_velocity_or_torque"]}
    reasons = []
    if float(np.max(np.abs(finger_torque))) > float(max_abs_torque_nm):
        reasons.append("finger_torque_limit")
    if float(np.max(np.abs(velocity))) > float(max_joint_speed_rad_s):
        reasons.append("joint_speed_limit")
    if joint_limits is not None:
        limit_violations = joint_target_limit_violations(
            desired,
            joint_limits,
            margin_rad=joint_limit_margin_rad,
        )
        if limit_violations:
            reasons.append("joint_target_limit_margin")
    tracking_error = float(np.max(np.abs(actual - desired)))
    if tracking_error > float(max_arm_tracking_error_rad):
        reasons.append("arm_tracking_limit")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "arm_tracking_error_rad": tracking_error,
        "max_arm_tracking_error_rad": float(max_arm_tracking_error_rad),
        "max_abs_torque_nm": float(max_abs_torque_nm),
        "max_joint_speed_rad_s": float(max_joint_speed_rad_s),
    }


def evaluate_waypoint_path_quality(
    waypoints: Sequence[Sequence[float]],
    *,
    forward_kinematics,
    physics_rate_hz: float,
    steps_per_waypoint: int,
    start_q: Sequence[float],
    table_top_z_m: float,
    fixture_center_m: Sequence[float],
    fixture_half_extent_m: Sequence[float],
    joint_limits: Sequence[tuple[float, float]] | None = None,
    joint_limit_margin_rad: float = 0.010,
    max_abs_dq_per_waypoint: float = 0.12,
    max_abs_qd_est_rad_s: float = 1.0,
    max_abs_qdd_est_rad_s2: float = 5.0,
    max_abs_jerk_est_rad_s3: float = 20.0,
    min_jacobian_singular_value: float = 0.02,
    max_jacobian_condition: float = 250.0,
    min_tcp_clearance_m: float = 0.08,
) -> dict[str, Any]:
    """Reject a bad waypoint path before any physics motion starts."""
    if int(steps_per_waypoint) < 1:
        raise ValueError("steps_per_waypoint must be positive")
    if float(physics_rate_hz) <= 0.0:
        raise ValueError("physics_rate_hz must be positive")
    dt = float(steps_per_waypoint) / float(physics_rate_hz)
    previous = np.asarray(start_q, dtype=np.float64).ravel()
    previous_qd = np.zeros(7, dtype=np.float64)
    previous_qdd = np.zeros(7, dtype=np.float64)
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "waypoint_count": 0,
        "peak_abs_dq_rad": 0.0,
        "peak_abs_qd_est_rad_s": 0.0,
        "peak_abs_qdd_est_rad_s2": 0.0,
        "peak_abs_jerk_est_rad_s3": 0.0,
        "minimum_jacobian_singular_value": math.inf,
        "maximum_jacobian_condition": 0.0,
        "minimum_tcp_above_table_m": math.inf,
        "minimum_fixture_clearance_m": math.inf,
        "joint_limit_margin_rad": joint_limit_margin_rad,
        "thresholds": {
            "max_abs_dq_per_waypoint": max_abs_dq_per_waypoint,
            "max_abs_qd_est_rad_s": max_abs_qd_est_rad_s,
            "max_abs_qdd_est_rad_s2": max_abs_qdd_est_rad_s2,
            "max_abs_jerk_est_rad_s3": max_abs_jerk_est_rad_s3,
            "min_jacobian_singular_value": min_jacobian_singular_value,
            "max_jacobian_condition": max_jacobian_condition,
            "min_tcp_clearance_m": min_tcp_clearance_m,
        },
        "reasons": reasons,
        "reject": False,
    }

    if joint_limits is not None:
        start_violations = joint_target_limit_violations(
            start_q, joint_limits, margin_rad=joint_limit_margin_rad
        )
        if start_violations:
            reasons.append(
                "start_q_joint_limit_margin:"
                + json.dumps(start_violations, sort_keys=True)
            )

    for waypoint_index, raw_waypoint in enumerate(waypoints, start=1):
        current = np.asarray(raw_waypoint, dtype=np.float64).ravel()
        if current.shape != (7,):
            raise ValueError("waypoint must be a 7-vector")
        if joint_limits is not None:
            limit_violations = joint_target_limit_violations(
                current,
                joint_limits,
                margin_rad=joint_limit_margin_rad,
            )
            if limit_violations:
                reasons.append(
                    f"waypoint_{waypoint_index}_joint_limit_margin:"
                    + json.dumps(limit_violations, sort_keys=True)
                )
        dq = current - previous
        dq_abs = float(np.max(np.abs(dq)))
        if dq_abs > float(max_abs_dq_per_waypoint):
            reasons.append(
                f"waypoint_{waypoint_index}_dq={dq_abs:.6f}"
            )
        qd = dq / dt
        qdd = (qd - previous_qd) / dt
        jerk = (qdd - previous_qdd) / dt
        transform = np.asarray(
            forward_kinematics(tuple(float(value) for value in current)),
            dtype=np.float64,
        )
        jacobian = numeric_tcp_jacobian(current, forward_kinematics)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        singular_min = float(np.min(singular_values))
        singular_max = float(np.max(singular_values))
        condition = (
            singular_max / singular_min if singular_min > 1.0e-12 else math.inf
        )
        clearance = tcp_clearance_screen(
            transform[:3, 3],
            table_top_z_m=table_top_z_m,
            fixture_center_m=fixture_center_m,
            fixture_half_extent_m=fixture_half_extent_m,
        )
        if singular_min < float(min_jacobian_singular_value):
            reasons.append(
                f"waypoint_{waypoint_index}_jacobian_smin="
                f"{singular_min:.6f}"
            )
        if condition > float(max_jacobian_condition):
            reasons.append(
                f"waypoint_{waypoint_index}_jacobian_cond={condition:.3f}"
            )
        if clearance["above_table_m"] < float(min_tcp_clearance_m):
            reasons.append(
                f"waypoint_{waypoint_index}_tcp_above_table_m="
                f"{clearance['above_table_m']:.6f}"
            )
        if clearance["fixture_clearance_m"] < 0.10:
            reasons.append(
                f"waypoint_{waypoint_index}_fixture_clearance_m="
                f"{clearance['fixture_clearance_m']:.6f}"
            )

        metrics["waypoint_count"] = waypoint_index
        metrics["peak_abs_dq_rad"] = max(
            metrics["peak_abs_dq_rad"], dq_abs
        )
        metrics["peak_abs_qd_est_rad_s"] = max(
            metrics["peak_abs_qd_est_rad_s"], float(np.max(np.abs(qd)))
        )
        metrics["peak_abs_qdd_est_rad_s2"] = max(
            metrics["peak_abs_qdd_est_rad_s2"], float(np.max(np.abs(qdd)))
        )
        metrics["peak_abs_jerk_est_rad_s3"] = max(
            metrics["peak_abs_jerk_est_rad_s3"], float(np.max(np.abs(jerk)))
        )
        metrics["minimum_jacobian_singular_value"] = min(
            metrics["minimum_jacobian_singular_value"], singular_min
        )
        metrics["maximum_jacobian_condition"] = max(
            metrics["maximum_jacobian_condition"], condition
        )
        metrics["minimum_tcp_above_table_m"] = min(
            metrics["minimum_tcp_above_table_m"],
            clearance["above_table_m"],
        )
        metrics["minimum_fixture_clearance_m"] = min(
            metrics["minimum_fixture_clearance_m"],
            clearance["fixture_clearance_m"],
        )
        previous = current
        previous_qd = qd
        previous_qdd = qdd

    if math.isinf(metrics["minimum_jacobian_singular_value"]):
        metrics["minimum_jacobian_singular_value"] = None
    metrics["reject"] = bool(reasons)
    return metrics


def build_failure_report(
    *,
    error: str,
    status: str,
    trace_records: Sequence[Mapping[str, Any]],
    path_quality_records: Sequence[Mapping[str, Any]],
    control_authorized: bool = False,
    formal_estimator_input: bool = False,
    threshold_label: str = "SIM_TUNING_ONLY_CANDIDATE",
) -> dict[str, Any]:
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "role": "display_motion_failure_report",
        "status": status,
        "error": error,
        "control_authorized": control_authorized,
        "formal_estimator_input": formal_estimator_input,
        "threshold_label": threshold_label,
        "trace_record_count": len(trace_records),
        "path_quality": list(path_quality_records),
        "trace_records": list(trace_records),
    }


__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "EMA_ALPHA",
    "EMA_FORCE_RESIDUAL_N",
    "EMA_MOMENT_RESIDUAL_NM",
    "FORMAL_MAX_WRIST_FORCE_N",
    "FORMAL_MAX_WRIST_MOMENT_NM",
    "TRACE_CAPACITY",
    "DisplayMotionRingBuffer",
    "atomic_write_json",
    "atomic_write_json_lines",
    "build_failure_report",
    "evaluate_display_sensor_gates",
    "evaluate_display_wrist_evidence",
    "evaluate_waypoint_path_quality",
    "joint_target_limit_violations",
    "numeric_tcp_jacobian",
    "tcp_clearance_screen",
]
