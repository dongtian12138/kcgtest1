#!/usr/bin/env python3

"""Capture audit-only simulator truth and evaluate a saved V2 dynamic run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
    from .engine_health import pending_engine_fields
else:
    from engine_health import pending_engine_fields


TERMINAL_LINK_NAMES = ("f1Link3", "f2Link2", "f3Link3")


def _below(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _rotation_matrix_quaternion(rotation: Sequence[Sequence[float]]) -> list[float]:
    xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


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
        robot_model,
        stage_modules: tuple[object, object, object],
        contact_interface,
        path_decoder,
        roots: Mapping[str, str],
        expected_total_mass_kg: float,
        part_bottom_offsets_m: Sequence[float],
        table_top_z_m: float,
        physics_dt_s: float,
        engine_monitor,
        physics_step_interface,
        tensor_contact_prim,
        tensor_contact_sensor_paths: Sequence[str],
        tensor_contact_max_count: int,
    ) -> None:
        self.object_parts = tuple(object_parts)
        self.hand_base_prim = hand_base_prim
        self.robot_model = robot_model
        self.Gf, self.Usd, self.UsdGeom = stage_modules
        self.contact_interface = contact_interface
        self.path_decoder = path_decoder
        self.roots = dict(roots)
        self.engine_monitor = engine_monitor
        self.tensor_contact_prim = tensor_contact_prim
        self.tensor_contact_sensor_paths = tuple(map(str, tensor_contact_sensor_paths))
        self.tensor_contact_max_count = int(tensor_contact_max_count)
        if (
            not self.tensor_contact_sensor_paths
            or self.tensor_contact_max_count <= 0
        ):
            raise ValueError("tensor contact audit inputs must be nonempty and bounded")
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
        self._physics_step_reports: list[list[dict[str, object]]] = []
        self._contact_report_subscription = contact_interface.subscribe_contact_report_events(
            self._on_contact_report
        )
        self._physics_step_subscription = (
            physics_step_interface.subscribe_physics_step_events(
                self._on_physics_step
            )
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

    def _decode_full_report(self, headers, contact_data) -> list[dict[str, object]]:
        decoded: list[dict[str, object]] = []
        contact_data_count = len(contact_data)
        for header in headers:
            offset = int(header.contact_data_offset)
            count = int(header.num_contact_data)
            if offset < 0 or offset + count > contact_data_count:
                raise RuntimeError("contact header data range is invalid")
            paths = tuple(
                str(self.path_decoder(value))
                for value in (
                    header.actor0,
                    header.actor1,
                    header.collider0,
                    header.collider1,
                )
            )
            contacts = []
            for index in range(offset, offset + count):
                record = contact_data[index]
                contacts.append({
                    "position_m": [float(record.position[axis]) for axis in range(3)],
                    "normal": [float(record.normal[axis]) for axis in range(3)],
                    "impulse_n_s": [float(record.impulse[axis]) for axis in range(3)],
                    "separation_m": float(record.separation),
                })
            decoded.append({
                "paths": paths,
                "records": count,
                "contact_data_offset": offset,
                "contacts": contacts,
            })
        return decoded

    def _on_contact_report(self, headers, _contact_data) -> None:
        self._event_headers.extend(self._decode_headers(headers))

    def _on_physics_step(self, _dt) -> None:
        headers, contact_data, _ = self.contact_interface.get_full_contact_report()
        self._physics_step_reports.append(
            self._decode_full_report(headers, contact_data)
        )

    def _tensor_contact_rows(self) -> list[dict[str, object]]:
        import warp as wp

        (forces, points, normals, separations, counts, start_indices,
         other_actor_ids) = self.tensor_contact_prim.get_raw_contact_data(dt=1.0)
        forces_array = np.asarray(forces.numpy(), dtype=np.float64).reshape(-1)
        points_array = np.asarray(points.numpy(), dtype=np.float64).reshape(-1, 3)
        normals_array = np.asarray(normals.numpy(), dtype=np.float64).reshape(-1, 3)
        separations_array = np.asarray(
            separations.numpy(), dtype=np.float64
        ).reshape(-1)
        counts_array = np.asarray(counts.numpy(), dtype=np.int64).reshape(-1)
        starts_array = np.asarray(
            start_indices.numpy(), dtype=np.int64
        ).reshape(-1)
        actor_ids_array = np.asarray(
            other_actor_ids.numpy(), dtype=np.uint64
        ).reshape(-1)
        if not (
            len(counts_array)
            == len(starts_array)
            == len(self.tensor_contact_sensor_paths)
        ):
            raise RuntimeError("tensor contact sensor layout does not match audited paths")
        total_count = int(np.sum(counts_array))
        if total_count >= self.tensor_contact_max_count:
            raise RuntimeError("tensor contact buffer capacity was reached")
        rows: list[dict[str, object]] = []
        for sensor_index, sensor_path in enumerate(
            self.tensor_contact_sensor_paths
        ):
            start = int(starts_array[sensor_index])
            count = int(counts_array[sensor_index])
            end = start + count
            if (
                start < 0
                or count < 0
                or end > len(actor_ids_array)
                or end > len(forces_array)
                or end > len(points_array)
                or end > len(normals_array)
                or end > len(separations_array)
            ):
                raise RuntimeError("tensor contact data range is invalid")
            if not count:
                continue
            ids = np.ascontiguousarray(actor_ids_array[start:end])
            ids_cpu = wp.array(ids, dtype=wp.uint64, device="cpu")
            other_paths = self.tensor_contact_prim.get_actor_paths_from_ids(
                ids_cpu
            )
            if len(other_paths) != count or any(not str(path) for path in other_paths):
                raise RuntimeError("tensor contact actor ID did not resolve to a USD path")
            grouped: dict[str, list[dict[str, object]]] = {}
            for offset, other_path_value in enumerate(other_paths):
                index = start + offset
                impulse = float(forces_array[index])
                point = points_array[index]
                normal = normals_array[index]
                separation = float(separations_array[index])
                if not (
                    math.isfinite(impulse)
                    and math.isfinite(separation)
                    and np.all(np.isfinite(point))
                    and np.all(np.isfinite(normal))
                ):
                    raise RuntimeError("tensor contact data is not finite")
                other_path = str(other_path_value)
                grouped.setdefault(other_path, []).append({
                    "other_actor_id": int(ids[offset]),
                    "position_m": point.tolist(),
                    "normal": normal.tolist(),
                    "normal_impulse_n_s": impulse,
                    "impulse_n_s": (normal * impulse).tolist(),
                    "separation_m": separation,
                })
            for other_path, contacts in grouped.items():
                rows.append({
                    "sensor_index": sensor_index,
                    "paths": (sensor_path, other_path),
                    "records": len(contacts),
                    "contacts": contacts,
                })
        return rows

    def _usd_hand_pose(self) -> tuple[list[float], list[float]]:
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

    def _hand_pose(
        self, active_positions: Sequence[float]
    ) -> tuple[list[float], list[float]]:
        transforms = self.robot_model.forward_kinematics(
            tuple(map(float, active_positions)), enforce_limits=False
        )
        if "handbase_link" not in transforms:
            raise RuntimeError("robot model FK did not return handbase_link")
        matrix = np.asarray(transforms["handbase_link"], dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("robot model FK hand-base transform is invalid")
        return (
            [float(value) for value in matrix[:3, 3]],
            _rotation_matrix_quaternion(matrix[:3, :3]),
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
            "physics_step_callback_count": 0,
            "contact_observation_backend": "tensor_raw_contact",
            "tensor_contact_sensor_count": len(self.tensor_contact_sensor_paths),
            "tensor_contact_header_count": 0,
            "tensor_contact_data_count": 0,
            "tensor_physical_contact_data_count": 0,
            "tensor_headers": [],
            "contact_report_channels_agree": True,
            "event_headers": [],
            "poll_headers": [],
            "examples": {},
        }
        if not self._physics_step_reports:
            raise RuntimeError(
                "native physics-step callback did not provide a contact report"
            )
        report_batches, self._physics_step_reports = (
            self._physics_step_reports,
            [],
        )
        report_rows = [row for batch in report_batches for row in batch]
        polled = [
            (row["paths"], int(row["records"])) for row in report_rows
        ]
        tensor_rows = self._tensor_contact_rows()
        tensor_physical_rows = [
            (
                row,
                sum(
                    math.isfinite(float(contact["normal_impulse_n_s"]))
                    and float(contact["normal_impulse_n_s"]) > 0.0
                    for contact in row["contacts"]
                ),
            )
            for row in tensor_rows
        ]
        events, self._event_headers = self._event_headers, []
        result.update({
            "event_header_count": len(events),
            "event_contact_data_count": sum(row[1] for row in events),
            "poll_header_count": len(polled),
            "poll_contact_data_count": sum(row[1] for row in polled),
            "physics_step_callback_count": len(report_batches),
            "tensor_contact_header_count": len(tensor_rows),
            "tensor_contact_data_count": sum(
                int(row["records"]) for row in tensor_rows
            ),
            "tensor_physical_contact_data_count": sum(
                records for _, records in tensor_physical_rows
            ),
            "tensor_headers": [
                {
                    "sensor_index": int(row["sensor_index"]),
                    "paths": list(row["paths"]),
                    "records": int(row["records"]),
                    "contacts": row["contacts"],
                }
                for row in tensor_rows
            ],
            "event_headers": [{"paths": list(row[0]), "records": row[1]} for row in events],
            "poll_headers": [
                {
                    "paths": list(row["paths"]),
                    "records": int(row["records"]),
                    "contact_data_offset": int(row["contact_data_offset"]),
                    "contacts": row["contacts"],
                }
                for row in report_rows
            ],
        })
        event_map, poll_map = dict(events), dict(polled)
        result["contact_report_channels_agree"] = event_map == poll_map
        combined = poll_map | {
            paths: max(records, poll_map.get(paths, 0)) for paths, records in events
        }
        for row, records in tensor_physical_rows:
            if records == 0:
                continue
            paths = tuple(sorted(map(str, row["paths"])))
            combined[paths] = max(
                records, combined.get(paths, 0)
            )
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
        hand_position, hand_orientation = self._hand_pose(active_positions)
        usd_hand_position, usd_hand_orientation = self._usd_hand_pose()
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
                "hand_base_pose_source": (
                    "MEASURED_ACTIVE_JOINTS_FROZEN_ROBOT_MODEL_FK"
                ),
                "hand_base_position_m": hand_position,
                "hand_base_orientation_wxyz": hand_orientation,
                "usd_hand_base_diagnostic_position_m": usd_hand_position,
                "usd_hand_base_diagnostic_orientation_wxyz": usd_hand_orientation,
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
    peak_index = int(np.argmax(np.abs(acceleration))) if len(acceleration) else None
    actual = (
        float(abs(acceleration[peak_index]))
        if peak_index is not None
        else None
    )
    signed_peak = (
        float(acceleration[peak_index])
        if peak_index is not None
        else None
    )
    peak_center_index = (
        peak_index + window if peak_index is not None else None
    )
    peak_row = (
        lift_rows[peak_center_index]
        if peak_center_index is not None
        else None
    )
    peak_lift_elapsed_s = (
        peak_center_index * physics_dt_s
        if peak_center_index is not None
        else None
    )
    peak_lift_fraction = (
        peak_center_index / (len(lift_rows) - 1)
        if peak_center_index is not None and len(lift_rows) > 1
        else None
    )
    registered = float(criteria["registered_lift_peak_acceleration_m_s2"])
    passed = bool(
        actual is not None
        and actual
        <= registered + float(criteria["lift_acceleration_tolerance_m_s2"])
    )
    return {
        "actual": actual,
        "signed_peak": signed_peak,
        "peak_simulation_time_s": (
            float(peak_row["simulation_time_s"])
            if peak_row is not None
            else None
        ),
        "peak_lift_elapsed_s": peak_lift_elapsed_s,
        "peak_lift_fraction": peak_lift_fraction,
        "registered": registered,
        "passed": passed,
    }


def _finger_clamp_effort_metrics(samples) -> dict[str, object]:
    """Summarize the three measured closing-joint efforts relative to tare."""

    tare_rows = [
        np.asarray(row["active_efforts_nm"], dtype=np.float64)[7:]
        for row in samples
        if row["phase"] == "tare"
    ]
    tare = np.mean(np.stack(tare_rows), axis=0) if tare_rows else np.zeros(4)
    result = {}
    for phase in ("preload", "lift", "hold"):
        rows = [
            np.asarray(row["active_efforts_nm"], dtype=np.float64)[7:] - tare
            for row in samples
            if row["phase"] == phase
        ]
        values = np.stack(rows) if rows else np.empty((0, 4))
        result[phase] = {
            name: {
                "median_tare_subtracted_nm": (
                    None if not len(values) else float(np.median(values[:, index]))
                ),
                "maximum_absolute_tare_subtracted_nm": (
                    None if not len(values) else float(np.max(np.abs(values[:, index])))
                ),
            }
            for name, index in (("finger_1", 1), ("finger_2", 2), ("finger_3", 3))
        }
    return result


def _terminal_contact_actor_paths(samples) -> tuple[tuple[str, ...], ...]:
    observed = []
    for index, name in enumerate(TERMINAL_LINK_NAMES):
        paths = set()
        for row in samples:
            contacts = row["contacts"]
            if contacts["terminal_link_object"][index] <= 0:
                continue
            examples = contacts.get("terminal_link_object_examples", [None] * 3)
            example = examples[index]
            if not example:
                continue
            matches = [
                str(path) for path in example
                if str(path).endswith("/" + name)
            ]
            if len(matches) != 1:
                return tuple()
            paths.add(matches[0])
        observed.append(tuple(sorted(paths)))
    return tuple(observed)


def _derive_pad_surface_identity(
    document: Mapping[str, object], samples, robot_asset_path: Path | None,
    inputs=None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "verified": False,
        "method": (
            "BOUND_TERMINAL_COLLIDER_PLUS_ALL_POSITIVE_CONTACT_POINTS_"
            "NEAREST_TO_USER_CONFIRMED_FULL_PAD"
        ),
        "reason": "ROBOT_ASSET_NOT_PROVIDED",
        "mappings": [],
    }
    if robot_asset_path is None:
        return result
    robot_asset = Path(robot_asset_path).resolve()
    if not robot_asset.is_file():
        result["reason"] = "ROBOT_ASSET_MISSING"
        result["robot_asset_path"] = str(robot_asset)
        return result

    expected_sha256 = document.get("evidence_binding", {}).get(
        "robot_asset_sha256"
    )
    actual_sha256 = _file_sha256(robot_asset)
    result.update({
        "robot_asset_path": str(robot_asset),
        "trace_bound_robot_asset_sha256": expected_sha256,
        "robot_asset_sha256": actual_sha256,
        "evaluator_source_sha256": _file_sha256(Path(__file__).resolve()),
        "asset_binding_matches": bool(
            isinstance(expected_sha256, str)
            and expected_sha256 == actual_sha256
        ),
    })
    if not result["asset_binding_matches"]:
        result["reason"] = "ROBOT_ASSET_SHA256_MISMATCH"
        return result

    actor_paths = _terminal_contact_actor_paths(samples)
    if len(actor_paths) != len(TERMINAL_LINK_NAMES) or any(
        len(paths) > 1 for paths in actor_paths
    ):
        result["reason"] = "TERMINAL_CONTACT_ACTOR_NOT_UNIQUE"
        result["observed_terminal_actor_paths"] = [
            list(paths) for paths in actor_paths
        ]
        return result

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(robot_asset))
    if stage is None:
        result["reason"] = "ROBOT_USD_OPEN_FAILED"
        return result
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        result["reason"] = "ROBOT_USD_DEFAULT_PRIM_MISSING"
        return result
    default_path = str(default_prim.GetPath())
    mappings = []
    for name, observed_paths in zip(TERMINAL_LINK_NAMES, actor_paths):
        if not observed_paths:
            continue
        actor_path = observed_paths[0]
        owners = [
            prim for prim in stage.Traverse()
            if prim.GetName() == name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(owners) != 1:
            result["reason"] = "TERMINAL_RIGID_BODY_NOT_UNIQUE"
            result["failed_terminal_link"] = name
            return result
        owner = owners[0]
        owner_path = str(owner.GetPath())
        owner_suffix = owner_path[len(default_path):]
        if not owner_suffix or not actor_path.endswith(owner_suffix):
            result["reason"] = "CONTACT_ACTOR_DOES_NOT_MATCH_BOUND_USD"
            result["failed_terminal_link"] = name
            return result
        colliders = []
        for prim in Usd.PrimRange(owner):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision_enabled = UsdPhysics.CollisionAPI(
                prim
            ).GetCollisionEnabledAttr().Get()
            if collision_enabled is True:
                colliders.append(prim)
        if len(colliders) != 1:
            result["reason"] = "ENABLED_COLLIDER_NOT_UNIQUE"
            result["failed_terminal_link"] = name
            result["enabled_collider_count"] = len(colliders)
            return result
        collider = colliders[0]
        material_role = collider.GetAttribute("kcg:materialRole").Get()
        source_mesh_uri = collider.GetAttribute("kcg:sourceMeshUri").Get()
        collider_path = str(collider.GetPath())
        collider_suffix = collider_path[len(owner_path):]
        mappings.append({
            "terminal_link": name,
            "contact_actor_path": actor_path,
            "asset_rigid_body_path": owner_path,
            "inferred_contact_collider_path": actor_path + collider_suffix,
            "asset_collider_path": collider_path,
            "collision_enabled": True,
            "material_role": material_role,
            "source_mesh_uri": source_mesh_uri,
        })

    result["mappings"] = mappings
    if inputs is None or inputs.task_grip_surfaces is None:
        result["reason"] = "USER_CONFIRMED_FULL_PAD_GEOMETRY_NOT_PROVIDED"
        return result
    roots = document.get("audit_roots")
    if not isinstance(roots, Mapping) or not isinstance(roots.get("object"), str):
        result["reason"] = "OBJECT_CONTACT_ROOT_NOT_RECORDED"
        return result

    from trimesh import Trimesh
    from trimesh.proximity import ProximityQuery
    from kcg_connector.grasp.carts_v2.task_grip_surface import (
        task_noncontact_triangles,
    )

    surfaces = {
        surface.link_name: surface
        for surface in inputs.task_grip_surfaces.values()
    }
    if set(surfaces) != set(TERMINAL_LINK_NAMES):
        result["reason"] = "USER_CONFIRMED_FULL_PAD_LINKS_INCOMPLETE"
        return result
    noncontact = task_noncontact_triangles(
        inputs.hand_collision_triangles_by_link, inputs.task_grip_surfaces
    )
    local_points: dict[str, list[np.ndarray]] = {
        name: [] for name in TERMINAL_LINK_NAMES
    }
    point_metadata: dict[str, list[dict[str, object]]] = {
        name: [] for name in TERMINAL_LINK_NAMES
    }
    object_root = roots["object"]
    for row in samples:
        world_points: dict[str, list[tuple[np.ndarray, Mapping[str, object]]]] = {
            name: [] for name in TERMINAL_LINK_NAMES
        }
        for header in row["contacts"].get("tensor_headers", ()):
            paths = tuple(map(str, header.get("paths", ())))
            if not any(_below(path, object_root) for path in paths):
                continue
            links = [
                name for name in TERMINAL_LINK_NAMES
                if any(f"/{name}" in path for path in paths)
            ]
            if len(links) != 1:
                continue
            for contact in header.get("contacts", ()):
                if float(contact.get("normal_impulse_n_s", 0.0)) > 0.0:
                    point = np.asarray(contact.get("position_m"), dtype=np.float64)
                    if point.shape != (3,) or not np.all(np.isfinite(point)):
                        result["reason"] = "NONFINITE_TERMINAL_CONTACT_POINT"
                        return result
                    world_points[links[0]].append((point, contact))
        if not any(world_points.values()):
            continue
        transforms = inputs.robot_model.forward_kinematics(
            row["active_positions_rad"], enforce_limits=False
        )
        for name, points in world_points.items():
            if not points:
                continue
            transform = np.asarray(transforms[name], dtype=np.float64)
            points_array = np.asarray([item[0] for item in points], dtype=np.float64)
            local_array = (
                points_array - transform[:3, 3]
            ) @ transform[:3, :3]
            local_points[name].extend(local_array)
            for local_point, (world_point, contact) in zip(local_array, points):
                world_normal = np.asarray(contact.get("normal"), dtype=np.float64)
                point_metadata[name].append({
                    "step": int(row["step"]),
                    "simulation_time_s": float(row["simulation_time_s"]),
                    "phase": str(row["phase"]),
                    "object_center_m": list(map(float, row["object_center_m"])),
                    "world_position_m": world_point.tolist(),
                    "world_normal": world_normal.tolist(),
                    "link_local_position_m": local_point.tolist(),
                    "link_local_normal": (world_normal @ transform[:3, :3]).tolist(),
                    "normal_impulse_n_s": float(contact["normal_impulse_n_s"]),
                    "separation_m": float(contact["separation_m"]),
                })

    def proximity(triangles: np.ndarray, points: np.ndarray):
        triangles = np.asarray(triangles, dtype=np.float64)
        vertices = triangles.reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
        mesh = Trimesh(vertices=vertices, faces=faces, process=False)
        return ProximityQuery(mesh).on_surface(points)

    per_link = []
    all_points_are_pad = True
    all_links_have_points = True
    for name in TERMINAL_LINK_NAMES:
        points = np.asarray(local_points[name], dtype=np.float64).reshape(-1, 3)
        if not len(points):
            all_links_have_points = False
            per_link.append({
                "terminal_link": name,
                "positive_contact_point_count": 0,
                "all_points_nearest_to_full_pad": None,
            })
            continue
        pad_surface = surfaces[name]
        pad_closest, pad_distance, pad_face = proximity(
            pad_surface.triangles_local_m, points
        )
        nonpad_closest, nonpad_distance, nonpad_face = proximity(
            noncontact[name], points
        )
        margin = nonpad_distance - pad_distance
        link_pass = bool(np.all(margin >= -1.0e-9))
        all_points_are_pad = all_points_are_pad and link_pass
        link_evidence = {
            "terminal_link": name,
            "positive_contact_point_count": len(points),
            "all_points_nearest_to_full_pad": link_pass,
            "maximum_pad_surface_residual_m": float(np.max(pad_distance)),
            "minimum_nonpad_minus_pad_distance_m": float(np.min(margin)),
            "nearest_full_pad_source_face_index_min": int(np.min(
                pad_surface.source_face_indices[pad_face]
            )),
            "nearest_full_pad_source_face_index_max": int(np.max(
                pad_surface.source_face_indices[pad_face]
            )),
        }
        nonpad_indices = np.flatnonzero(margin < -1.0e-9)
        if len(nonpad_indices):
            index = int(nonpad_indices[0])
            link_evidence["earliest_nonpad_contact"] = {
                **point_metadata[name][index],
                "pad_distance_m": float(pad_distance[index]),
                "nonpad_distance_m": float(nonpad_distance[index]),
                "nonpad_minus_pad_distance_m": float(margin[index]),
                "nearest_pad_position_link_local_m": pad_closest[index].tolist(),
                "nearest_nonpad_position_link_local_m": nonpad_closest[index].tolist(),
                "nearest_full_pad_source_face_index": int(
                    pad_surface.source_face_indices[pad_face[index]]
                ),
                "nearest_nonpad_triangle_index": int(nonpad_face[index]),
            }
        per_link.append(link_evidence)
    result["contact_point_projection"] = per_link
    if not all_points_are_pad:
        result["reason"] = "POSITIVE_CONTACT_POINT_NEAREST_TO_NONPAD_SURFACE"
        return result
    if not all_links_have_points:
        result["reason"] = "NOT_ALL_THREE_TERMINAL_LINKS_HAVE_POSITIVE_OBJECT_CONTACT"
        return result
    result.update({"verified": True, "reason": None})
    return result


def evaluate_trace(
    document: Mapping[str, object], *, robot_asset_path: Path | None = None,
    inputs=None,
) -> dict[str, object]:
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
    finger_efforts = _finger_clamp_effort_metrics(samples)
    lift_pass = motion["maximum_lift_m"] >= float(criteria["lift_distance_m"])
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
    pad_identity_evidence = _derive_pad_surface_identity(
        document, samples, robot_asset_path, inputs
    )
    pad_identity = bool(pad_identity_evidence["verified"])
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
        "pad_surface_identity_evidence": pad_identity_evidence,
        "maximum_consecutive_simultaneous_contact_samples": contacts["maximum_sustained"],
        "maximum_lift_m": motion["maximum_lift_m"],
        "lift_50mm_passed": lift_pass,
        "hold_duration_s": motion["hold_duration_s"],
        "hold_2s_passed": hold_pass,
        "table_contact_released_during_hold": motion["table_released"],
        "maximum_relative_slip_m": motion["maximum_slip_m"],
        "maximum_orientation_change_rad": motion["maximum_orientation_change_rad"],
        "actual_lift_peak_acceleration_m_s2": acceleration["actual"],
        "signed_lift_peak_acceleration_m_s2": acceleration["signed_peak"],
        "lift_peak_acceleration_simulation_time_s": acceleration[
            "peak_simulation_time_s"
        ],
        "lift_peak_acceleration_elapsed_s": acceleration[
            "peak_lift_elapsed_s"
        ],
        "lift_peak_acceleration_fraction": acceleration[
            "peak_lift_fraction"
        ],
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
        "maximum_joint_speed_rad_s": document["controller_outcome"].get(
            "maximum_joint_speed_rad_s"
        ),
        "maximum_joint_speed_joint": document["controller_outcome"].get(
            "maximum_joint_speed_joint"
        ),
        "maximum_absolute_hand_effort_nm": document["controller_outcome"].get(
            "maximum_absolute_hand_effort_nm"
        ),
        "finger_clamp_effort_nm": finger_efforts,
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
            "RESEARCH_ONLY_BOUND_USD_AND_FULL_PAD_CONTACT_POINT_PROJECTION_NOT_FORMAL_CERTIFICATION"
            if pad_identity else
            "RESEARCH_ONLY_TERMINAL_LINK_CONTACT_PAD_IDENTITY_UNRESOLVED"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot-asset")
    arguments = parser.parse_args()
    trace = json.loads(Path(arguments.trace_json).read_text(encoding="utf-8"))
    if trace.get("mode") != "isolated-hand" and not arguments.robot_asset:
        parser.error("dynamic trace evaluation requires --robot-asset")
    result = (
        evaluate_isolated_hand_trace(trace)
        if trace.get("mode") == "isolated-hand"
        else evaluate_trace(
            trace, robot_asset_path=Path(arguments.robot_asset).resolve()
        )
    )
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
