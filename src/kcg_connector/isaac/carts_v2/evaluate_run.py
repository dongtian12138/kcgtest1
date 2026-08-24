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


def _host_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=np.float64)


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
    return list(_quaternion_rotation_matrix(orientation).T
                @ (np.asarray(point) - np.asarray(origin)))


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
        engine_monitor,
    ) -> None:
        self.object_parts = tuple(object_parts)
        self.hand_base_prim = hand_base_prim
        self.Gf, self.Usd, self.UsdGeom = stage_modules
        self.contact_interface = contact_interface
        self.path_decoder = path_decoder
        self.roots = dict(roots)
        self.engine_monitor = engine_monitor
        self.masses = np.asarray([float(_host_array(part.get_mass()).reshape(-1)[0]) for part in self.object_parts])
        self.local_coms = tuple(
            _host_array(part.get_com()[0]).reshape(-1)
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
        self._event_headers: list[tuple[tuple[str, ...], int]] = []
        self._contact_report_subscription = contact_interface.subscribe_contact_report_events(
            self._on_contact_report
        )

    def _decode_headers(self, headers) -> list[tuple[tuple[str, ...], int]]:
        return [
            (
                tuple(str(self.path_decoder(value)) for value in (
                    header.actor0, header.actor1, header.collider0, header.collider1
                )),
                int(header.num_contact_data),
            )
            for header in headers
        ]

    def _on_contact_report(self, headers, _contact_data) -> None:
        self._event_headers.extend(self._decode_headers(headers))

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
            "terminal_link_object_examples": [None, None, None],
            "robot_object_unauthorized": 0,
            "robot_table": 0,
            "robot_fixture": 0,
            "robot_unclassified": 0,
            "object_table": 0,
            "event_header_count": 0,
            "event_contact_data_count": 0,
            "poll_header_count": 0,
            "poll_contact_data_count": 0,
            "contact_report_channels_agree": True,
            "event_headers": [],
            "poll_headers": [],
            "examples": {},
        }
        headers, _, _ = self.contact_interface.get_full_contact_report()
        polled = self._decode_headers(headers)
        events, self._event_headers = self._event_headers, []
        result.update({
            "event_header_count": len(events),
            "event_contact_data_count": sum(row[1] for row in events),
            "poll_header_count": len(polled),
            "poll_contact_data_count": sum(row[1] for row in polled),
            "event_headers": [{"paths": list(row[0]), "records": row[1]} for row in events],
            "poll_headers": [{"paths": list(row[0]), "records": row[1]} for row in polled],
        })
        event_map, poll_map = dict(events), dict(polled)
        result["contact_report_channels_agree"] = event_map == poll_map
        combined = poll_map | {
            paths: max(records, poll_map.get(paths, 0)) for paths, records in events
        }
        for paths, records in combined.items():
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
            if has_robot and not (has_object or has_table or has_fixture):
                result["robot_unclassified"] += records
                result["examples"].setdefault("robot_unclassified", list(paths))
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
                        if result["terminal_link_object_examples"][index] is None:
                            result["terminal_link_object_examples"][index] = list(paths)
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
        self.engine_monitor.sample()
        poses = [tuple(_host_array(value) for value in part.get_world_pose()) for part in self.object_parts]
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
    terminal_examples = [next((row["contacts"].get("terminal_link_object_examples", [None] * 3)[index]
                               for row in samples if row["contacts"].get("terminal_link_object_examples", [None] * 3)[index]), None) for index in range(3)]
    return {
        "terminal_records": terminal_records,
        "maximum_sustained": maximum_sustained,
        "any_terminal": any(
            sum(row["contacts"]["terminal_link_object"][index] for row in samples) > 0
            for index in range(3)
        ),
        "examples": examples,
        "terminal_examples": terminal_examples,
        "channels_agree": all(
            row["contacts"].get("contact_report_channels_agree") is True
            for row in samples
        ),
    }


def _safety_metrics(samples, criteria) -> dict[str, object]:
    unauthorized = {
        key: sum(row["contacts"][key] for row in samples)
        for key in ("robot_object_unauthorized", "robot_table", "robot_fixture",
                    "robot_unclassified")
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
    diagnostics = [row.get("arm_control", {}).get("f1_mimic_diagnostic", {}) for row in samples]
    diagnostic_values = [item.get(key) for item in diagnostics for key in ("position_error_rad", "velocity_error_rad_s")] + [item.get(name, {}).get(key) for item in diagnostics for name in ("f1j2", "f1j3") for key in ("position_rad", "velocity_rad_s", "equivalent_effort_nm")]
    limit_values = [item.get(name, {}).get("limit_margin_rad") for item in diagnostics for name in ("f1j2", "f1j3")]
    finite = finite and all({"f1j2", "f1j3", "position_error_rad", "velocity_error_rad_s"} <= item.keys() and all({"position_rad", "velocity_rad_s", "equivalent_effort_nm", "limit_margin_rad"} <= item[name].keys() for name in ("f1j2", "f1j3")) for item in diagnostics) and all(value is not None and math.isfinite(float(value)) for value in diagnostic_values) and all(value is None or math.isfinite(float(value)) for value in limit_values)
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
        document.get("contact_report_api_audit", {}).get("complete") is True,
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
    first_hold = [row for row in samples if row["phase"] == "finger_1_hold"]
    confirmation = next((index for index, row in enumerate(samples)
                         if row["phase"] == "finger_1_contact_confirmed"), None)
    evidence = [] if confirmation is None else samples[max(
        0, confirmation - int(criteria["sustained_three_contact_samples"])):confirmation]
    first_contact = evidence + [row for row in samples if row["phase"] in (
        "finger_1_contact_confirmed", "finger_1_contact_settle", "finger_1_hold")]
    proxy, terminal = (bool(document["controller_outcome"].get("contact_targets_rad")),
                       any(row["contacts"]["terminal_link_object"][0] > 0 for row in first_contact))
    witness_complete = document.get("contact_report_api_audit", {}).get("complete") is True
    contact_class = ("NO_CONTACT_PROXY" if not proxy else
                     "UNRESOLVED_CONTACT_REPORT_API_COVERAGE" if not witness_complete else
                     "UNRESOLVED_CONTACT_REPORT_DISAGREEMENT" if not contacts["channels_agree"] else
                     "FALSE_CONTACT_PROXY" if not terminal
                     else "UNRESOLVED_TERMINAL_LINK_CONTACT_PATCH" if not pad_identity else "ALLOWED_PAD_CONTACT")
    pregrasp_hand = document.get("motion_plan", {}).get("pregrasp_hand_positions_rad")
    only_first = bool(len(document["controller_outcome"].get("contact_targets_rad", ())) == 1 and pregrasp_hand is not None and all(np.allclose(row["active_targets_rad"][9:], pregrasp_hand[2:], atol=1.0e-12, rtol=0.0) for row in first_hold))
    first_controller_pass = bool(
        document["mode"] == "first-finger-diagnostic" and all(shared_passes)
        and len(first_hold) * physics_dt_s >= float(criteria["first_finger_diagnostic_duration_s"])
        and document["controller_outcome"]["maximum_finger_target_delta_rad"]
        <= float(criteria["maximum_finger_target_increment_rad"]) + 1.0e-12
        and only_first
        and contact_class == "ALLOWED_PAD_CONTACT")
    research_pass = bool(
        nominal_physical_pass
        and document["offline_task_gate_passed"]
        and pad_identity
    )
    truth_isolation = bool(
        document.get("online_object_or_contact_truth_used") is False
        and document.get("truth_audit_data_returned_to_controller") is False
        and document.get("object_pose_writes_after_start") == 0)
    return {
        "schema_version": "carts_grasp_v2_dynamic_evaluation_v2",
        "object_id": document["object_id"], "candidate_id": document["candidate_id"],
        "mode": document["mode"],
        "physics_time_advanced_s": len(samples) * physics_dt_s,
        "three_terminal_link_contacts_observed": contact_pass,
        "terminal_link_contact_records": contacts["terminal_records"],
        "pad_surface_identity_verified": pad_identity,
        "maximum_consecutive_simultaneous_contact_samples": contacts["maximum_sustained"],
        "maximum_lift_m": motion["maximum_lift_m"],
        "lift_50mm_passed": lift_pass,
        "hold_duration_s": motion["hold_duration_s"],
        "hold_2s_passed": hold_pass,
        "table_contact_released_during_hold": motion["table_released"],
        "maximum_relative_slip_m": motion["maximum_slip_m"],
        "maximum_orientation_change_rad": motion["maximum_orientation_change_rad"],
        "actual_lift_peak_acceleration_m_s2": acceleration["actual"],
        "registered_lift_peak_acceleration_m_s2": acceleration["registered"],
        "lift_acceleration_consistent": acceleration["passed"],
        "maximum_table_penetration_m": safety["overall_penetration_m"],
        "maximum_post_settle_table_penetration_m": safety["post_settle_penetration_m"],
        "unauthorized_contact_records": safety["unauthorized"],
        "first_unauthorized_contact_paths": contacts["examples"],
        "first_terminal_link_object_paths": contacts["terminal_examples"],
        "contact_report_channels_agree": contacts["channels_agree"],
        "contact_report_api_complete": witness_complete,
        "first_finger_contact_classification": contact_class,
        "first_finger_hold_duration_s": len(first_hold) * physics_dt_s,
        "first_finger_maximum_target_delta_rad": document["controller_outcome"].get("maximum_finger_target_delta_rad"), "only_first_finger_commanded": only_first,
        "controller_first_finger_diagnostic_pass": first_controller_pass,
        "first_finger_diagnostic_pass": False,
        "finite_throughout": safety["finite"],
        "controller_completed": control_complete,
        "controller_failure_reason": document["controller_outcome"]["failure_reason"],
        "truth_isolation_pass": truth_isolation,
        "accepted_preflight_bound": bool(document.get("accepted_preflight_bound")),
        "accepted_preflight_evaluation_sha256": document.get("accepted_preflight_evaluation_sha256"),
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
    result = evaluate_isolated_hand_trace(trace) if trace.get("mode") == "isolated-hand" else evaluate_trace(trace)
    Path(arguments.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = result["diagnostic_pass"] if trace.get("mode") == "isolated-hand" else result["research_dynamic_pass"]
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IsolatedHandRecorder", "TruthAuditRecorder", "audit_initial_joint_state",
    "audit_mimic_schema", "compare_reference_targets",
    "evaluate_isolated_hand_trace", "evaluate_trace",
]
