#!/usr/bin/env python3

"""Capture audit-only simulator truth and evaluate a saved V2 dynamic run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

if __package__:
    from .engine_health import pending_engine_fields
else:
    from engine_health import pending_engine_fields


TERMINAL_LINK_NAMES = ("f1Link3", "f2Link2", "f3Link3")


def _below(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _quaternion_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    value /= np.linalg.norm(value)
    w, x, y, z = value
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _quaternion_distance(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


def _relative_position(
    point: Sequence[float], origin: Sequence[float], orientation: Sequence[float]
) -> list[float]:
    rotation = _quaternion_rotation_matrix(orientation)
    return list(rotation.T @ (np.asarray(point) - np.asarray(origin)))


class IsolatedHandRecorder:
    """Record joint-side signals without loading or observing an object."""

    def __init__(self, *, robot, dof_names, active_names, hand_names,
                 physics_dt_s, drive_settings):
        self.robot = robot
        self.dof_names = tuple(dof_names)
        self.active_names = tuple(active_names)
        self.hand_names = tuple(hand_names)
        self.physics_dt_s = float(physics_dt_s)
        self.drive_settings = dict(drive_settings)
        self.samples = []
        self._previous_targets = None

    def _commanded_effort(self, name, target, position, velocity):
        prefix = "arm" if name.startswith("iiwa_") else "hand"
        kp = float(self.drive_settings[f"{prefix}_stiffness"])
        kd = float(self.drive_settings[f"{prefix}_damping"])
        cap = float(self.drive_settings[f"{prefix}_drive_maximum_effort_nm"])
        return float(np.clip(kp * (target - position) - kd * velocity, -cap, cap))

    def capture(self, *, step, phase, active_positions, active_velocities,
                active_efforts, active_targets, arm_control):
        positions = self.robot.get_dof_positions(indices=0).numpy()[0]
        velocities = self.robot.get_dof_velocities(indices=0).numpy()[0]
        efforts = self.robot.get_dof_projected_joint_forces(indices=0).numpy()[0]
        target_values = np.asarray(active_targets).tolist()
        targets = dict(zip(self.active_names, target_values))
        previous = targets if self._previous_targets is None else self._previous_targets
        target_deltas = [targets[name] - previous[name] for name in self.active_names]
        joints = {}
        for name in self.hand_names:
            index = self.dof_names.index(name)
            target = targets.get(name)
            joints[name] = {
                "target_position_rad": target,
                "target_delta_rad": None if target is None else target - previous[name],
                "actual_position_rad": float(positions[index]),
                "actual_velocity_rad_s": float(velocities[index]),
                "commanded_drive_effort_nm": None if target is None else (
                    self._commanded_effort(name, target, positions[index], velocities[index])
                ),
                "equivalent_joint_effort_nm": float(efforts[index]),
            }
        self.samples.append({
            "step": int(step), "simulation_time_s": (int(step) + 1) * self.physics_dt_s,
            "phase": str(phase), "active_targets_rad": target_values,
            "active_target_deltas_rad": target_deltas, "joints": joints,
        })
        self._previous_targets = targets


def evaluate_isolated_hand_trace(document: Mapping[str, object]) -> dict[str, object]:
    samples = document["samples"]
    if not samples:
        raise ValueError("isolated hand trace has no physics samples")
    rows = [(sample, sample["joints"]["f2j1"]) for sample in samples]
    peak_sample, peak = max(rows, key=lambda row: abs(row[1]["actual_velocity_rad_s"]))
    hand_peaks = {
        name: max(abs(sample["joints"][name]["actual_velocity_rad_s"])
                  for sample in samples)
        for name in samples[0]["joints"]
    }
    peak_hand_joint = max(hand_peaks, key=hand_peaks.get)
    max_target_delta = max(abs(row[1]["target_delta_rad"] or 0.0) for row in rows)
    follower_error = max(
        abs(sample["joints"]["f2j2"]["actual_position_rad"]
            - sample["joints"]["f2j1"]["actual_position_rad"])
        for sample in samples
    )
    outcome = document["controller_outcome"]
    target_match = bool(document["reference_target_comparison"]["matches"])
    speed_safe = hand_peaks[peak_hand_joint] <= float(
        document["maximum_joint_speed_limit_rad_s"]
    )
    failure = outcome["failure_reason"]
    if not target_match:
        failure = "TARGET_TRAJECTORY_MISMATCH"
    elif not speed_safe:
        failure = "MIMIC_OR_ACTIVE_HAND_SPEED_ABORT"
    return {
        "diagnostic_pass": bool(outcome["completed"] and target_match and speed_safe),
        "failure_reason": failure,
        "sample_count": len(samples),
        "physics_time_advanced_s": samples[-1]["simulation_time_s"],
        "maximum_hand_speed_joint": peak_hand_joint,
        "maximum_hand_joint_speed_rad_s": hand_peaks[peak_hand_joint],
        "hand_joint_peak_speeds_rad_s": hand_peaks,
        "f2j1_maximum_target_delta_rad": max_target_delta,
        "f2j1_maximum_speed_rad_s": abs(peak["actual_velocity_rad_s"]),
        "f2j1_peak_speed_step": peak_sample["step"],
        "f2j1_peak_speed_commanded_drive_effort_nm": peak[
            "commanded_drive_effort_nm"
        ],
        "f2j1_peak_speed_equivalent_joint_effort_nm": peak[
            "equivalent_joint_effort_nm"
        ],
        "f2j1_f2j2_maximum_position_error_rad": follower_error,
        "reference_target_prefix_matches": target_match,
        "reference_target_maximum_difference_rad": document[
            "reference_target_comparison"
        ]["maximum_absolute_difference_rad"],
        "maximum_joint_speed_limit_rad_s": document[
            "maximum_joint_speed_limit_rad_s"
        ],
        "hardware_authorized": False,
        "formal_dynamic_pass": False,
        "research_dynamic_pass": False,
    }


def audit_initial_joint_state(robot, dof_names) -> dict[str, object]:
    positions = robot.get_dof_positions(indices=0).numpy()[0]
    velocities = robot.get_dof_velocities(indices=0).numpy()[0]
    position_targets = robot.get_dof_position_targets(indices=0).numpy()[0]
    velocity_targets = robot.get_dof_velocity_targets(indices=0).numpy()[0]
    lower, upper = robot.get_dof_limits(indices=0)
    rows = zip(
        dof_names, positions, velocities, position_targets, velocity_targets,
        lower.numpy()[0], upper.numpy()[0],
    )
    return {
        name: {
            "position_rad": float(position), "velocity_rad_s": float(velocity),
            "position_target_rad": float(position_target),
            "velocity_target_rad_s": float(velocity_target),
            "lower_limit_rad": float(lower_limit), "upper_limit_rad": float(upper_limit),
        }
        for name, position, velocity, position_target, velocity_target,
        lower_limit, upper_limit in rows
    }


def audit_mimic_schema(stage, robot_root, mimic_joints) -> dict[str, object]:
    result = {}
    for follower, source in mimic_joints.items():
        prim = stage.GetPrimAtPath(f"{robot_root}/Physics/{follower}")
        relation = prim.GetRelationship("newton:mimicJoint") if prim.IsValid() else None
        targets = [] if relation is None else [str(path) for path in relation.GetTargets()]
        expected = f"{robot_root}/Physics/{source}"
        result[follower] = {
            "source": source, "relationship_targets": targets,
            "relationship_matches": targets == [expected],
            "applied_schemas": [] if not prim.IsValid() else list(prim.GetAppliedSchemas()),
        }
    return result


def compare_reference_targets(reference, observed) -> dict[str, object]:
    reference_rows = [row["active_targets_rad"] for row in reference["samples"]]
    observed_rows = [row["active_targets_rad"] for row in observed]
    count = min(len(reference_rows), len(observed_rows))
    if count == 0:
        raise ValueError("target comparison has no common physics samples")
    difference = np.abs(
        np.asarray(observed_rows[:count]) - np.asarray(reference_rows[:count])
    )
    maximum = float(np.max(difference))
    return {
        "compared_sample_count": count,
        "reference_sample_count": len(reference_rows),
        "observed_sample_count": len(observed_rows),
        "maximum_absolute_difference_rad": maximum,
        "matches": maximum <= 1.0e-12,
    }


class TruthAuditRecorder:
    """Read simulator truth for logging only; return nothing to online control."""

    def __init__(
        self,
        *,
        object_parts: Sequence[object],
        hand_base_prim,
        stage_modules: tuple[object, object, object],
        contact_interface,
        path_decoder,
        roots: Mapping[str, str],
        expected_total_mass_kg: float,
        part_bottom_offsets_m: Sequence[float],
        table_top_z_m: float,
        physics_dt_s: float,
    ) -> None:
        self.object_parts = tuple(object_parts)
        self.hand_base_prim = hand_base_prim
        self.Gf, self.Usd, self.UsdGeom = stage_modules
        self.contact_interface = contact_interface
        self.path_decoder = path_decoder
        self.roots = dict(roots)
        self.masses = np.asarray(
            [float(part.get_mass()) for part in self.object_parts], dtype=np.float64
        )
        self.local_coms = tuple(
            np.asarray(part.get_com()[0], dtype=np.float64).reshape(-1)
            for part in self.object_parts
        )
        self.bottom_offsets = tuple(map(float, part_bottom_offsets_m))
        if not self.object_parts or not (
            len(self.object_parts) == len(self.masses) == len(self.bottom_offsets)
        ):
            raise ValueError("object part audit inputs must be nonempty and aligned")
        if any(center.shape != (3,) for center in self.local_coms):
            raise ValueError("simulator local COM must contain exactly three values")
        if abs(float(np.sum(self.masses)) - float(expected_total_mass_kg)) > 1.0e-6:
            raise ValueError("simulator mass differs from the registered object model")
        self.table_top_z_m = float(table_top_z_m)
        self.physics_dt_s = float(physics_dt_s)
        self.samples: list[dict[str, object]] = []

    def _hand_pose(self) -> tuple[list[float], list[float]]:
        matrix = self.UsdGeom.Xformable(
            self.hand_base_prim
        ).ComputeLocalToWorldTransform(self.Usd.TimeCode.Default())
        transform = self.Gf.Transform(matrix)
        translation = transform.GetTranslation()
        quaternion = transform.GetRotation().GetQuat()
        imaginary = quaternion.GetImaginary()
        return (
            [float(translation[index]) for index in range(3)],
            [float(quaternion.GetReal())]
            + [float(imaginary[index]) for index in range(3)],
        )

    def _contact_counts(self) -> dict[str, object]:
        result: dict[str, object] = {
            "terminal_link_object": [0, 0, 0],
            "robot_object_unauthorized": 0,
            "robot_table": 0,
            "robot_fixture": 0,
            "object_table": 0,
            "examples": {},
        }
        headers, _, _ = self.contact_interface.get_full_contact_report()
        for header in headers:
            paths = tuple(
                str(self.path_decoder(value))
                for value in (
                    header.actor0,
                    header.actor1,
                    header.collider0,
                    header.collider1,
                )
            )
            records = int(header.num_contact_data)
            has_robot = any(_below(path, self.roots["robot"]) for path in paths)
            has_object = any(_below(path, self.roots["object"]) for path in paths)
            has_table = any(_below(path, self.roots["table"]) for path in paths)
            has_fixture = any(_below(path, self.roots["fixture"]) for path in paths)
            if has_object and has_table:
                result["object_table"] += records
            if has_robot and has_table:
                result["robot_table"] += records
                result["examples"].setdefault("robot_table", list(paths))
            if has_robot and has_fixture:
                result["robot_fixture"] += records
                result["examples"].setdefault("robot_fixture", list(paths))
            if not (has_robot and has_object):
                continue
            terminal_hits = [
                any(f"/{name}" in path for path in paths)
                for name in TERMINAL_LINK_NAMES
            ]
            if any(terminal_hits):
                for index, hit in enumerate(terminal_hits):
                    if hit:
                        result["terminal_link_object"][index] += records
            else:
                result["robot_object_unauthorized"] += records
                result["examples"].setdefault(
                    "robot_object_unauthorized", list(paths)
                )
        return result

    def capture(
        self,
        *,
        step: int,
        phase: str,
        active_positions: Sequence[float],
        active_velocities: Sequence[float],
        active_efforts: Sequence[float],
        active_targets: Sequence[float],
        arm_control: Mapping[str, object],
    ) -> None:
        poses = [part.get_world_pose() for part in self.object_parts]
        positions = np.asarray([pose[0] for pose in poses], dtype=np.float64)
        centers = np.asarray(
            [
                position
                + _quaternion_rotation_matrix(pose[1]) @ local_com
                for position, pose, local_com in zip(
                    positions, poses, self.local_coms
                )
            ],
            dtype=np.float64,
        )
        center = np.average(centers, axis=0, weights=self.masses)
        hand_position, hand_orientation = self._hand_pose()
        bottom = min(
            float(position[2]) + offset
            for position, offset in zip(positions, self.bottom_offsets)
        )
        self.samples.append(
            {
                "step": int(step),
                "simulation_time_s": (int(step) + 1) * self.physics_dt_s,
                "phase": str(phase),
                "active_positions_rad": list(map(float, active_positions)),
                "active_velocities_rad_s": list(map(float, active_velocities)),
                "active_efforts_nm": list(map(float, active_efforts)),
                "active_targets_rad": list(map(float, active_targets)),
                "arm_control": dict(arm_control),
                "object_part_positions_m": [list(map(float, row)) for row in positions],
                "object_part_orientations_wxyz": [
                    list(map(float, pose[1])) for pose in poses
                ],
                "reference_part_orientation_wxyz": list(map(float, poses[0][1])),
                "object_center_m": list(map(float, center)),
                "object_bottom_clearance_m": bottom - self.table_top_z_m,
                "hand_base_position_m": hand_position,
                "hand_base_orientation_wxyz": hand_orientation,
                "object_center_in_hand_base_m": _relative_position(
                    center, hand_position, hand_orientation
                ),
                "contacts": self._contact_counts(),
            }
        )


def _motion_metrics(samples, criteria, physics_dt_s: float) -> dict[str, object]:
    settled_rows = [row for row in samples if row["phase"] == "settle"]
    if not settled_rows:
        raise ValueError("dynamic trace contains no settle sample")
    settled = settled_rows[-1]
    grasped = [row for row in samples if row["phase"] in ("preload", "lift", "hold")]
    hold = [row for row in samples if row["phase"] == "hold"]
    baseline = grasped[0] if grasped else settled
    lift_rows = [row for row in samples if row["phase"] in ("lift", "hold")]
    maximum_lift = max(
        (row["object_center_m"][2] - settled["object_center_m"][2] for row in lift_rows),
        default=0.0,
    )
    reference_relative = np.asarray(
        baseline["object_center_in_hand_base_m"], dtype=np.float64
    )
    maximum_slip = max(
        (
            float(np.linalg.norm(
                np.asarray(row["object_center_in_hand_base_m"]) - reference_relative
            ))
            for row in grasped
        ),
        default=0.0,
    )
    reference_orientation = baseline.get(
        "reference_part_orientation_wxyz", baseline.get("body_orientation_wxyz")
    )
    maximum_orientation_change = max(
        (
            _quaternion_distance(
                reference_orientation,
                row.get(
                    "reference_part_orientation_wxyz",
                    row.get("body_orientation_wxyz"),
                ),
            )
            for row in grasped
        ),
        default=0.0,
    )
    return {
        "grasped": grasped,
        "maximum_lift_m": maximum_lift,
        "hold_duration_s": len(hold) * physics_dt_s,
        "table_released": bool(
            hold
            and all(row["contacts"]["object_table"] == 0 for row in hold)
            and min(row["object_bottom_clearance_m"] for row in hold)
            > float(criteria["table_release_clearance_m"])
        ),
        "maximum_slip_m": maximum_slip,
        "maximum_orientation_change_rad": maximum_orientation_change,
    }


def _contact_metrics(samples, grasped) -> dict[str, object]:
    terminal_records = [
        sum(row["contacts"]["terminal_link_object"][index] for row in grasped)
        for index in range(3)
    ]
    maximum_sustained = 0
    current_sustained = 0
    for row in grasped:
        simultaneous = all(
            value > 0 for value in row["contacts"]["terminal_link_object"]
        )
        current_sustained = current_sustained + 1 if simultaneous else 0
        maximum_sustained = max(maximum_sustained, current_sustained)
    examples: dict[str, object] = {}
    for row in samples:
        for key, value in row["contacts"].get("examples", {}).items():
            examples.setdefault(key, value)
    return {
        "terminal_records": terminal_records,
        "maximum_sustained": maximum_sustained,
        "any_terminal": any(
            sum(row["contacts"]["terminal_link_object"][index] for row in samples) > 0
            for index in range(3)
        ),
        "examples": examples,
    }


def _safety_metrics(samples, criteria) -> dict[str, object]:
    unauthorized = {
        key: sum(row["contacts"][key] for row in samples)
        for key in ("robot_object_unauthorized", "robot_table", "robot_fixture")
    }
    overall_penetration = max(
        0.0, -min(float(row["object_bottom_clearance_m"]) for row in samples)
    )
    post_settle = [row for row in samples if row["phase"] != "settle"]
    post_settle_penetration = max(
        0.0,
        -min(
            (float(row["object_bottom_clearance_m"]) for row in post_settle),
            default=0.0,
        ),
    )
    finite = all(
        np.all(np.isfinite(row[key]))
        for row in samples
        for key in (
            "active_positions_rad",
            "active_velocities_rad_s",
            "active_efforts_nm",
            "object_center_m",
        )
    )
    return {
        "unauthorized": unauthorized,
        "overall_penetration_m": overall_penetration,
        "post_settle_penetration_m": post_settle_penetration,
        "finite": finite,
        "collision_pass": all(value == 0 for value in unauthorized.values()),
        "penetration_pass": post_settle_penetration
        <= float(criteria["maximum_table_penetration_m"]),
    }


def _acceleration_metrics(samples, criteria, physics_dt_s: float) -> dict[str, object]:
    lift_rows = [row for row in samples if row["phase"] == "lift"]
    window = int(criteria["lift_acceleration_difference_window_samples"])
    hand_z = np.asarray([row["hand_base_position_m"][2] for row in lift_rows])
    denominator = (window * physics_dt_s) ** 2
    acceleration = (
        (hand_z[2 * window :] - 2.0 * hand_z[window:-window] + hand_z[: -2 * window])
        / denominator
        if len(hand_z) > 2 * window
        else np.asarray([], dtype=np.float64)
    )
    actual = float(np.max(np.abs(acceleration))) if len(acceleration) else None
    registered = float(criteria["registered_lift_peak_acceleration_m_s2"])
    passed = bool(
        actual is not None
        and actual
        <= registered + float(criteria["lift_acceleration_tolerance_m_s2"])
    )
    return {"actual": actual, "registered": registered, "passed": passed}


def evaluate_trace(document: Mapping[str, object]) -> dict[str, object]:
    """Apply frozen physical success criteria to an independently saved trace."""

    samples = list(document["samples"])
    criteria = document["criteria"]
    if not samples:
        raise ValueError("dynamic trace contains no samples")
    physics_dt_s = float(document["physics_dt_s"])
    motion = _motion_metrics(samples, criteria, physics_dt_s)
    contacts = _contact_metrics(samples, motion["grasped"])
    safety = _safety_metrics(samples, criteria)
    acceleration = _acceleration_metrics(samples, criteria, physics_dt_s)
    lift_pass = motion["maximum_lift_m"] + float(
        criteria["lift_tolerance_m"]
    ) >= float(criteria["lift_distance_m"])
    hold_pass = motion["hold_duration_s"] >= float(criteria["hold_duration_s"])
    contact_pass = contacts["maximum_sustained"] >= int(
        criteria["sustained_three_contact_samples"]
    )
    control_complete = bool(document["controller_outcome"]["completed"])
    shared_passes = (
        safety["finite"],
        control_complete,
        safety["collision_pass"],
        safety["penetration_pass"],
    )
    controller_preflight_pass = bool(
        document["mode"] == "preflight"
        and all(shared_passes)
        and not contacts["any_terminal"]
    )
    nominal_physical_pass = bool(
        document["mode"] == "grasp-lift"
        and all(shared_passes)
        and contact_pass
        and lift_pass
        and hold_pass
        and motion["table_released"]
        and acceleration["passed"]
    )
    pad_identity = bool(document.get("pad_surface_identity_verified", False))
    research_pass = bool(
        nominal_physical_pass
        and document["offline_task_gate_passed"]
        and pad_identity
    )
    return {
        "schema_version": "carts_grasp_v2_dynamic_evaluation_v2",
        "object_id": document["object_id"],
        "candidate_id": document["candidate_id"],
        "mode": document["mode"],
        "physics_time_advanced_s": len(samples) * physics_dt_s,
        "three_terminal_link_contacts_observed": contact_pass,
        "terminal_link_contact_records": contacts["terminal_records"],
        "pad_surface_identity_verified": pad_identity,
        "maximum_consecutive_simultaneous_contact_samples": (
            contacts["maximum_sustained"]
        ),
        "maximum_lift_m": motion["maximum_lift_m"],
        "lift_50mm_passed": lift_pass,
        "hold_duration_s": motion["hold_duration_s"],
        "hold_2s_passed": hold_pass,
        "table_contact_released_during_hold": motion["table_released"],
        "maximum_relative_slip_m": motion["maximum_slip_m"],
        "maximum_orientation_change_rad": motion[
            "maximum_orientation_change_rad"
        ],
        "actual_lift_peak_acceleration_m_s2": acceleration["actual"],
        "registered_lift_peak_acceleration_m_s2": acceleration["registered"],
        "lift_acceleration_consistent": acceleration["passed"],
        "maximum_table_penetration_m": safety["overall_penetration_m"],
        "maximum_post_settle_table_penetration_m": (
            safety["post_settle_penetration_m"]
        ),
        "unauthorized_contact_records": safety["unauthorized"],
        "first_unauthorized_contact_paths": contacts["examples"],
        "finite_throughout": safety["finite"],
        "controller_completed": control_complete,
        "controller_failure_reason": document["controller_outcome"]["failure_reason"],
        "preflight_pass": False,
        **pending_engine_fields(
            controller_preflight_pass,
            bool(document.get("identity_hash_check_pass", False)),
        ),
        "nominal_diagnostic_pass": nominal_physical_pass,
        "research_dynamic_pass": research_pass,
        "formal_dynamic_pass": False,
        "hardware_authorized": False,
        "evidence_binding": document.get("evidence_binding", {}),
        "evidence_limit": (
            "RESEARCH_ONLY_WHOLE_TERMINAL_LINK_CONVEX_CONTACT_NOT_FORMAL_PAD_PATCH"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_json")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    trace = json.loads(Path(arguments.trace_json).read_text(encoding="utf-8"))
    isolated = trace.get("mode") == "isolated-hand"
    result = evaluate_isolated_hand_trace(trace) if isolated else evaluate_trace(trace)
    Path(arguments.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = result["diagnostic_pass"] if isolated else result["research_dynamic_pass"]
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IsolatedHandRecorder", "TruthAuditRecorder", "audit_initial_joint_state",
    "audit_mimic_schema", "compare_reference_targets",
    "evaluate_isolated_hand_trace", "evaluate_trace",
]
