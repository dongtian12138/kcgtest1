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
PAD_NONPAD_PROJECTION_TOLERANCE_M = 1.0e-9
TE_OBJECT_ID = "te_deutsch_d38999_26fj35pn_step"
TE_INSERTION_AXIS_OBJECT = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
TE_MAIN_KEY_OBJECT = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
TE_MATING_FACE_CENTER_OBJECT_M = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
TE_PIN_PLANE_CENTER_OBJECT_M = np.asarray(
    (0.0, 0.0, -0.0145923), dtype=np.float64
)
POSTGRASP_DISTURBANCE_COM_SCHEMA = "te_postgrasp_disturbance_panel_v2"


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


def _pose_matrix(
    position: Sequence[float], orientation: Sequence[float]
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _quaternion_rotation_matrix(orientation)
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def _transform_point(matrix: np.ndarray, point: Sequence[float]) -> np.ndarray:
    return matrix[:3, :3] @ np.asarray(point, dtype=np.float64) + matrix[:3, 3]


def _inverse_transform_point(
    matrix: np.ndarray, point: Sequence[float]
) -> np.ndarray:
    return matrix[:3, :3].T @ (
        np.asarray(point, dtype=np.float64) - matrix[:3, 3]
    )


def _vector_angle(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))


def _minimal_rotation_between(
    source: Sequence[float], target: Sequence[float]
) -> np.ndarray:
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if cosine > 1.0 - 1.0e-14:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + 1.0e-14:
        basis = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(math.pi * axis).as_matrix()
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    skew = np.asarray(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        ),
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


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

    def _tensor_friction_rows(self) -> list[dict[str, object]]:
        forces, points, counts, start_indices = (
            self.tensor_contact_prim.get_friction_data(dt=1.0)
        )
        forces_array = np.asarray(
            forces.numpy(), dtype=np.float64
        ).reshape(-1, 3)
        points_array = np.asarray(
            points.numpy(), dtype=np.float64
        ).reshape(-1, 3)
        counts_array = np.asarray(
            counts.numpy(), dtype=np.int64
        )
        starts_array = np.asarray(
            start_indices.numpy(), dtype=np.int64
        )
        filter_paths = tuple(map(
            str, self.tensor_contact_prim._contact_filter_paths
        ))
        expected_shape = (
            len(self.tensor_contact_sensor_paths), len(filter_paths)
        )
        if counts_array.shape != expected_shape or starts_array.shape != expected_shape:
            raise RuntimeError("tensor friction pair layout does not match sensors and filters")
        total_count = int(np.sum(counts_array))
        if total_count >= self.tensor_contact_max_count:
            raise RuntimeError("tensor friction buffer capacity was reached")
        rows = []
        for sensor_index, sensor_path in enumerate(
            self.tensor_contact_sensor_paths
        ):
            for filter_index, filter_path in enumerate(filter_paths):
                start = int(starts_array[sensor_index, filter_index])
                count = int(counts_array[sensor_index, filter_index])
                end = start + count
                if (
                    start < 0
                    or count < 0
                    or end > len(forces_array)
                    or end > len(points_array)
                ):
                    raise RuntimeError("tensor friction data range is invalid")
                if not count:
                    continue
                records = []
                for index in range(start, end):
                    impulse = forces_array[index]
                    point = points_array[index]
                    if not (
                        np.all(np.isfinite(impulse))
                        and np.all(np.isfinite(point))
                    ):
                        raise RuntimeError("tensor friction data is not finite")
                    records.append({
                        "position_m": point.tolist(),
                        "tangential_impulse_n_s": impulse.tolist(),
                        "tangential_impulse_magnitude_n_s": float(
                            np.linalg.norm(impulse)
                        ),
                    })
                rows.append({
                    "sensor_index": sensor_index,
                    "filter_index": filter_index,
                    "paths": (sensor_path, filter_path),
                    "records": count,
                    "contacts": records,
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

    def _robot_link_poses(
        self, active_positions: Sequence[float], link_names: Sequence[str]
    ) -> dict[str, tuple[list[float], list[float]]]:
        transforms = self.robot_model.forward_kinematics(
            tuple(map(float, active_positions)), enforce_limits=False
        )
        result = {}
        for name in link_names:
            if name not in transforms:
                raise RuntimeError(f"robot model FK did not return {name}")
            matrix = np.asarray(transforms[name], dtype=np.float64)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise RuntimeError(f"robot model FK transform is invalid for {name}")
            result[name] = (
                [float(value) for value in matrix[:3, 3]],
                _rotation_matrix_quaternion(matrix[:3, :3]),
            )
        return result

    def _hand_pose(
        self, active_positions: Sequence[float]
    ) -> tuple[list[float], list[float]]:
        return self._robot_link_poses(
            active_positions, ("handbase_link",)
        )["handbase_link"]

    def _contact_counts(self) -> dict[str, object]:
        result: dict[str, object] = {
            "terminal_link_object": [0, 0, 0],
            "terminal_link_object_examples": [None, None, None],
            "robot_object_unauthorized": 0,
            "robot_table": 0,
            "robot_fixture": 0,
            "robot_unclassified": 0,
            "object_table": 0,
            "object_table_positive_normal_impulse_n_s": 0.0,
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
        friction_rows = self._tensor_friction_rows()
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
            "tensor_friction_header_count": len(friction_rows),
            "tensor_friction_data_count": sum(
                int(row["records"]) for row in friction_rows
            ),
            "friction_headers": [
                {
                    "sensor_index": int(row["sensor_index"]),
                    "filter_index": int(row["filter_index"]),
                    "paths": list(row["paths"]),
                    "records": int(row["records"]),
                    "contacts": row["contacts"],
                }
                for row in friction_rows
            ],
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
            paths = tuple(sorted(map(str, row["paths"])))
            if (
                any(_below(path, self.roots["object"]) for path in paths)
                and any(_below(path, self.roots["table"]) for path in paths)
            ):
                result["object_table_positive_normal_impulse_n_s"] += sum(
                    max(0.0, float(contact["normal_impulse_n_s"]))
                    for contact in row["contacts"]
                )
            if records == 0:
                continue
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
        link_poses = self._robot_link_poses(
            active_positions, ("handbase_link", *TERMINAL_LINK_NAMES)
        )
        hand_position, hand_orientation = link_poses["handbase_link"]
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
                "terminal_link_positions_m": {
                    name: link_poses[name][0] for name in TERMINAL_LINK_NAMES
                },
                "terminal_link_orientations_wxyz": {
                    name: link_poses[name][1] for name in TERMINAL_LINK_NAMES
                },
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


def _contact_surface_slip_metrics(
    document: Mapping[str, object], samples, inputs, physics_dt_s: float
) -> dict[str, object]:
    """Derive post-run rigid material-point tangential motion at each pad."""

    object_paths = tuple(map(str, document.get(
        "tensor_contact_view_audit", {}
    ).get("object_sensor_paths", ())))
    if not object_paths and len(samples[0].get("object_part_positions_m", ())) == 1:
        object_paths = (str(document.get("audit_roots", {}).get("object", "")),)
    if (
        not object_paths
        or len(object_paths) != len(samples[0].get("object_part_positions_m", ()))
    ):
        return {
            "status": "UNAVAILABLE",
            "reason": "OBJECT_PART_PATHS_DO_NOT_MATCH_RECORDED_POSES",
            "online_control_used": False,
        }
    if inputs is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "FROZEN_ROBOT_MODEL_NOT_AVAILABLE",
            "online_control_used": False,
        }

    fk_cache: dict[int, dict[str, np.ndarray]] = {}

    def terminal_matrices(index: int) -> dict[str, np.ndarray]:
        if index in fk_cache:
            return fk_cache[index]
        row = samples[index]
        positions = row.get("terminal_link_positions_m")
        orientations = row.get("terminal_link_orientations_wxyz")
        if isinstance(positions, Mapping) and isinstance(orientations, Mapping):
            result = {
                name: _pose_matrix(positions[name], orientations[name])
                for name in TERMINAL_LINK_NAMES
            }
        else:
            transforms = inputs.robot_model.forward_kinematics(
                tuple(map(float, row["active_positions_rad"])),
                enforce_limits=False,
            )
            result = {}
            for name in TERMINAL_LINK_NAMES:
                matrix = np.asarray(transforms[name], dtype=np.float64)
                if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                    raise ValueError(f"invalid frozen FK transform for {name}")
                result[name] = matrix
        fk_cache[index] = result
        return result

    def object_matrices(index: int) -> tuple[np.ndarray, ...]:
        row = samples[index]
        return tuple(
            _pose_matrix(position, orientation)
            for position, orientation in zip(
                row["object_part_positions_m"],
                row["object_part_orientations_wxyz"],
            )
        )

    def object_index(path: str) -> int | None:
        matches = [
            (len(root), index)
            for index, root in enumerate(object_paths)
            if _below(path, root)
        ]
        return None if not matches else max(matches)[1]

    def positive_contacts(index: int, terminal_link: str) -> list[dict[str, object]]:
        result = []
        suffix = "/" + terminal_link
        for header in samples[index]["contacts"].get("tensor_headers", ()):
            paths = tuple(map(str, header.get("paths", ())))
            if len(paths) != 2 or not paths[0].endswith(suffix):
                continue
            part_index = object_index(paths[1])
            if part_index is None:
                continue
            for contact in header.get("contacts", ()):
                impulse = float(contact.get("normal_impulse_n_s", 0.0))
                if not math.isfinite(impulse) or impulse <= 0.0:
                    continue
                result.append({
                    "part_index": part_index,
                    "part_path": object_paths[part_index],
                    "world_point_m": np.asarray(
                        contact["position_m"], dtype=np.float64
                    ),
                    "world_normal": np.asarray(
                        contact["normal"], dtype=np.float64
                    ),
                    "normal_impulse_n_s": impulse,
                    "separation_m": float(contact.get("separation_m", 0.0)),
                })
        return result

    def friction_contacts(index: int, terminal_link: str) -> list[dict[str, object]]:
        result = []
        suffix = "/" + terminal_link
        for header in samples[index]["contacts"].get("friction_headers", ()):
            paths = tuple(map(str, header.get("paths", ())))
            if (
                len(paths) != 2
                or not paths[0].endswith(suffix)
                or object_index(paths[1]) is None
            ):
                continue
            for contact in header.get("contacts", ()):
                impulse = np.asarray(
                    contact["tangential_impulse_n_s"], dtype=np.float64
                )
                point = np.asarray(contact["position_m"], dtype=np.float64)
                if (
                    impulse.shape != (3,)
                    or point.shape != (3,)
                    or not np.all(np.isfinite(impulse))
                    or not np.all(np.isfinite(point))
                ):
                    raise ValueError("friction contact data is invalid")
                result.append({
                    "part_index": int(object_index(paths[1])),
                    "part_path": paths[1],
                    "world_point_m": point,
                    "tangential_impulse_n_s": impulse,
                })
        return result

    material_projection_cache: dict[
        tuple[str, int, int], dict[str, object]
    ] = {}
    material_projection_complete = bool(
        inputs.task_grip_surfaces is not None
    )
    material_projection_reason = None
    if material_projection_complete:
        from trimesh import Trimesh
        from trimesh.proximity import ProximityQuery

        surfaces = {
            surface.link_name: surface
            for surface in inputs.task_grip_surfaces.values()
        }
        if set(surfaces) != set(TERMINAL_LINK_NAMES):
            material_projection_complete = False
            material_projection_reason = "FULL_PAD_SURFACE_SET_INCOMPLETE"
        else:
            def proximity_query(triangles: np.ndarray) -> ProximityQuery:
                triangles = np.asarray(triangles, dtype=np.float64)
                vertices = triangles.reshape(-1, 3)
                faces = np.arange(
                    len(vertices), dtype=np.int64
                ).reshape(-1, 3)
                return ProximityQuery(Trimesh(
                    vertices=vertices, faces=faces, process=False
                ))

            pad_queries = {
                name: proximity_query(surface.triangles_local_m)
                for name, surface in surfaces.items()
            }
            object_mesh = inputs.object_contract.model.mesh
            # The split Body and CouplingNut rigid frames are both authored in
            # the supplier object's local frame.  Contact-path identity selects
            # the moving rigid part; the unchanged supplier surface provides
            # the corresponding material point for either part.
            object_queries = {
                index: proximity_query(object_mesh.face_vertices_m)
                for index in range(len(object_paths))
            }
            for terminal_link in TERMINAL_LINK_NAMES:
                metadata = []
                finger_solver_points = []
                object_solver_points = []
                for index in range(len(samples)):
                    contacts = positive_contacts(index, terminal_link)
                    if not contacts:
                        continue
                    link_now = terminal_matrices(index)[terminal_link]
                    objects_now = object_matrices(index)
                    for offset, contact in enumerate(contacts):
                        part_index = int(contact["part_index"])
                        metadata.append((index, offset, part_index))
                        finger_solver_points.append(_inverse_transform_point(
                            link_now, contact["world_point_m"]
                        ))
                        object_solver_points.append(_inverse_transform_point(
                            objects_now[part_index], contact["world_point_m"]
                        ))
                if not metadata:
                    continue
                pad_closest, pad_distance, pad_face = pad_queries[
                    terminal_link
                ].on_surface(np.asarray(finger_solver_points))
                object_solver_points_array = np.asarray(object_solver_points)
                object_closest = np.empty_like(object_solver_points_array)
                object_distance = np.empty(len(metadata), dtype=np.float64)
                object_face = np.empty(len(metadata), dtype=np.int64)
                for part_index in sorted({row[2] for row in metadata}):
                    selected = np.asarray(
                        [
                            index
                            for index, row in enumerate(metadata)
                            if row[2] == part_index
                        ],
                        dtype=np.int64,
                    )
                    closest, distance, face = object_queries[
                        part_index
                    ].on_surface(object_solver_points_array[selected])
                    object_closest[selected] = closest
                    object_distance[selected] = distance
                    object_face[selected] = face
                source_pad_faces = surfaces[
                    terminal_link
                ].source_face_indices[pad_face]
                for row_index, key in enumerate(metadata):
                    material_projection_cache[
                        (terminal_link, key[0], key[1])
                    ] = {
                        "pad_surface_material_point_link_local_m": (
                            pad_closest[row_index]
                        ),
                        "pad_surface_projection_residual_m": float(
                            pad_distance[row_index]
                        ),
                        "pad_source_face_index": int(
                            source_pad_faces[row_index]
                        ),
                        "object_surface_material_point_object_local_m": (
                            object_closest[row_index]
                        ),
                        "object_surface_projection_residual_m": float(
                            object_distance[row_index]
                        ),
                        "object_source_face_index": int(
                            object_face[row_index]
                        ),
                    }
    else:
        material_projection_reason = "FULL_PAD_SURFACES_REQUIRED"

    friction_channel_present = all(
        "friction_headers" in row["contacts"] for row in samples
    )
    per_finger = []
    measurement_complete = True
    for terminal_link in TERMINAL_LINK_NAMES:
        cumulative_slip = 0.0
        maximum_step = 0.0
        maximum_speed = 0.0
        total_normal_impulse = 0.0
        total_tangential_impulse_magnitude = 0.0
        peak_tangential_impulse_magnitude = 0.0
        interval_count = 0
        contact_boundary_interval_count = 0
        contact_sample_count = 0
        finger_projection_complete = True
        establishment_events = []
        loss_events = []
        migration_events = []
        contact_trace = []
        active_previous = False
        previous_finger_centroid = None
        previous_object_centroids: dict[str, np.ndarray] = {}
        previous_active_index = None
        fingertip_patch_migration = 0.0
        object_patch_migration: dict[str, float] = {
            path: 0.0 for path in object_paths
        }
        contacted_parts = set()

        for index, row in enumerate(samples):
            contacts = positive_contacts(index, terminal_link)
            active = bool(contacts)
            if active and not active_previous:
                establishment_events.append({
                    "step": int(row["step"]),
                    "simulation_time_s": float(row["simulation_time_s"]),
                    "event_time_interval_s": [
                        max(
                            0.0,
                            float(row["simulation_time_s"]) - physics_dt_s,
                        ),
                        float(row["simulation_time_s"]),
                    ],
                    "phase": str(row["phase"]),
                })
            if not active and active_previous:
                previous = samples[index - 1]
                loss_events.append({
                    "last_active_step": int(previous["step"]),
                    "first_inactive_step": int(row["step"]),
                    "event_time_interval_s": [
                        float(previous["simulation_time_s"]),
                        float(row["simulation_time_s"]),
                    ],
                    "phase": str(row["phase"]),
                })
                previous_finger_centroid = None
                previous_object_centroids = {}
                previous_active_index = None
            active_previous = active
            if not active:
                continue

            contact_sample_count += 1
            link_now = terminal_matrices(index)[terminal_link]
            objects_now = object_matrices(index)
            friction = friction_contacts(index, terminal_link)
            friction_vectors = np.asarray([
                contact["tangential_impulse_n_s"] for contact in friction
            ], dtype=np.float64).reshape(-1, 3)
            friction_vector_sum = (
                np.sum(friction_vectors, axis=0)
                if len(friction_vectors) else np.zeros(3, dtype=np.float64)
            )
            friction_magnitude_sum = float(np.sum(
                np.linalg.norm(friction_vectors, axis=1)
            )) if len(friction_vectors) else 0.0
            total_tangential_impulse_magnitude += friction_magnitude_sum
            peak_tangential_impulse_magnitude = max(
                peak_tangential_impulse_magnitude,
                max(
                    (float(np.linalg.norm(value)) for value in friction_vectors),
                    default=0.0,
                ),
            )
            impulses = np.asarray([
                contact["normal_impulse_n_s"] for contact in contacts
            ], dtype=np.float64)
            weights = impulses / float(np.sum(impulses))
            total_normal_impulse += float(np.sum(impulses))
            finger_solver_points = np.asarray([
                _inverse_transform_point(link_now, contact["world_point_m"])
                for contact in contacts
            ])
            object_solver_points = np.asarray([
                _inverse_transform_point(
                    objects_now[int(contact["part_index"])],
                    contact["world_point_m"],
                )
                for contact in contacts
            ])
            projections = [
                material_projection_cache.get((terminal_link, index, offset))
                for offset in range(len(contacts))
            ]
            projection_available = all(value is not None for value in projections)
            finger_projection_complete = (
                finger_projection_complete and projection_available
            )
            finger_material_points = np.asarray([
                (
                    projection[
                        "pad_surface_material_point_link_local_m"
                    ]
                    if projection is not None else finger_solver_points[offset]
                )
                for offset, projection in enumerate(projections)
            ], dtype=np.float64)
            object_material_points = np.asarray([
                (
                    projection[
                        "object_surface_material_point_object_local_m"
                    ]
                    if projection is not None else object_solver_points[offset]
                )
                for offset, projection in enumerate(projections)
            ], dtype=np.float64)
            finger_centroid = np.average(
                finger_material_points, axis=0, weights=weights
            )
            object_centroids = {}
            for part_index, part_path in enumerate(object_paths):
                selected = [
                    offset for offset, contact in enumerate(contacts)
                    if contact["part_index"] == part_index
                ]
                if not selected:
                    continue
                contacted_parts.add(part_path)
                local_points = object_material_points[selected]
                part_weights = weights[selected]
                part_weights /= float(np.sum(part_weights))
                object_centroids[part_path] = np.average(
                    local_points, axis=0, weights=part_weights
                )

            migration_event = None
            if previous_active_index is not None and previous_active_index == index - 1:
                finger_migration_delta = float(np.linalg.norm(
                    finger_centroid - previous_finger_centroid
                ))
                fingertip_patch_migration += finger_migration_delta
                object_migration_deltas = {}
                for part_path, centroid in object_centroids.items():
                    if part_path in previous_object_centroids:
                        delta = float(np.linalg.norm(
                            centroid - previous_object_centroids[part_path]
                        ))
                        object_patch_migration[part_path] += delta
                        object_migration_deltas[part_path] = delta
                migration_event = {
                    "event": "MIGRATED",
                    "from_step": int(samples[index - 1]["step"]),
                    "to_step": int(row["step"]),
                    "event_time_interval_s": [
                        float(samples[index - 1]["simulation_time_s"]),
                        float(row["simulation_time_s"]),
                    ],
                    "fingertip_surface_centroid_delta_m": (
                        finger_migration_delta
                    ),
                    "object_surface_centroid_delta_m": (
                        object_migration_deltas
                    ),
                    "fingertip_surface_cumulative_m": (
                        fingertip_patch_migration
                    ),
                    "object_surface_cumulative_m": dict(
                        object_patch_migration
                    ),
                }
                migration_events.append(migration_event)
            previous_finger_centroid = finger_centroid
            previous_object_centroids = object_centroids
            previous_active_index = index

            step_mean = None
            step_maximum = None
            speed_mean = None
            speed_maximum = None
            tangential_vector = None
            interval_classification = None
            per_contact_tangential = [None] * len(contacts)
            if index + 1 < len(samples):
                dt = float(
                    samples[index + 1]["simulation_time_s"]
                    - row["simulation_time_s"]
                )
                if dt <= 0.0 or abs(dt - physics_dt_s) > 1.0e-9:
                    raise ValueError("contact slip samples are not one physics step apart")
                next_parts = {
                    contact["part_path"]
                    for contact in positive_contacts(index + 1, terminal_link)
                }
                continuous_indices = [
                    offset for offset, contact in enumerate(contacts)
                    if contact["part_path"] in next_parts
                ]
                if continuous_indices:
                    interval_classification = "CONTINUING_CONTACT"
                    link_next = terminal_matrices(index + 1)[terminal_link]
                    objects_next = object_matrices(index + 1)
                    tangent_deltas = []
                    tangent_weights = weights[continuous_indices]
                    tangent_weights /= float(np.sum(tangent_weights))
                    for offset in continuous_indices:
                        contact = contacts[offset]
                        finger_material = finger_material_points[offset]
                        object_material = object_material_points[offset]
                        normal = np.asarray(
                            contact["world_normal"], dtype=np.float64
                        )
                        normal_norm = float(np.linalg.norm(normal))
                        if not math.isfinite(normal_norm) or normal_norm <= 1.0e-12:
                            raise ValueError("contact normal is not finite and nonzero")
                        normal /= normal_norm
                        part_index = int(contact["part_index"])
                        separation_now = (
                            _transform_point(link_now, finger_material)
                            - _transform_point(
                                objects_now[part_index], object_material
                            )
                        )
                        separation_next = (
                            _transform_point(link_next, finger_material)
                            - _transform_point(
                                objects_next[part_index], object_material
                            )
                        )
                        relative_delta = separation_next - separation_now
                        tangent_delta = (
                            relative_delta
                            - normal * float(np.dot(normal, relative_delta))
                        )
                        tangent_deltas.append(tangent_delta)
                        per_contact_tangential[offset] = {
                            "tangential_step_m": float(
                                np.linalg.norm(tangent_delta)
                            ),
                            "tangential_speed_m_s": float(
                                np.linalg.norm(tangent_delta) / dt
                            ),
                            "tangential_displacement_vector_world_m": (
                                tangent_delta.tolist()
                            ),
                        }
                    tangent_deltas = np.asarray(
                        tangent_deltas, dtype=np.float64
                    )
                    tangent_norms = np.linalg.norm(tangent_deltas, axis=1)
                    step_mean = float(np.dot(tangent_weights, tangent_norms))
                    step_maximum = float(np.max(tangent_norms))
                    tangential_vector = np.average(
                        tangent_deltas, axis=0, weights=tangent_weights
                    )
                    speed_mean = step_mean / dt
                    speed_maximum = step_maximum / dt
                    cumulative_slip += step_mean
                    maximum_step = max(maximum_step, step_maximum)
                    maximum_speed = max(maximum_speed, speed_maximum)
                    interval_count += 1
                else:
                    interval_classification = (
                        "CONTACT_LOSS_BOUNDARY_NOT_INCLUDED_IN_MAIN_CUMULATIVE"
                    )
                    contact_boundary_interval_count += 1

            canonical_contact_points = []
            for offset, contact in enumerate(contacts):
                projection = projections[offset]
                canonical_contact_points.append({
                    "object_part_path": contact["part_path"],
                    "solver_contact_point_world_m": contact[
                        "world_point_m"
                    ].tolist(),
                    "solver_contact_normal_world": contact[
                        "world_normal"
                    ].tolist(),
                    "solver_contact_separation_m": contact["separation_m"],
                    "normal_impulse_n_s": contact["normal_impulse_n_s"],
                    "solver_contact_point_link_local_m": (
                        finger_solver_points[offset].tolist()
                    ),
                    "solver_contact_point_object_local_m": (
                        object_solver_points[offset].tolist()
                    ),
                    "pad_surface_material_point_link_local_m": (
                        None if projection is None else projection[
                            "pad_surface_material_point_link_local_m"
                        ].tolist()
                    ),
                    "pad_surface_projection_residual_m": (
                        None if projection is None else projection[
                            "pad_surface_projection_residual_m"
                        ]
                    ),
                    "pad_source_face_index": (
                        None if projection is None else projection[
                            "pad_source_face_index"
                        ]
                    ),
                    "object_surface_material_point_object_local_m": (
                        None if projection is None else projection[
                            "object_surface_material_point_object_local_m"
                        ].tolist()
                    ),
                    "object_surface_projection_residual_m": (
                        None if projection is None else projection[
                            "object_surface_projection_residual_m"
                        ]
                    ),
                    "object_source_face_index": (
                        None if projection is None else projection[
                            "object_source_face_index"
                        ]
                    ),
                    "continuing_contact_tangential_motion": (
                        per_contact_tangential[offset]
                    ),
                })
            friction_point_records = []
            for contact in friction:
                part_index = int(contact["part_index"])
                friction_point_records.append({
                    "object_part_path": object_paths[part_index],
                    "friction_point_world_m": contact[
                        "world_point_m"
                    ].tolist(),
                    "friction_point_link_local_m": _inverse_transform_point(
                        link_now, contact["world_point_m"]
                    ).tolist(),
                    "friction_point_object_local_m": _inverse_transform_point(
                        objects_now[part_index], contact["world_point_m"]
                    ).tolist(),
                    "tangential_impulse_world_n_s": contact[
                        "tangential_impulse_n_s"
                    ].tolist(),
                    "tangential_impulse_magnitude_n_s": float(np.linalg.norm(
                        contact["tangential_impulse_n_s"]
                    )),
                })

            contact_trace.append({
                "step": int(row["step"]),
                "simulation_time_s": float(row["simulation_time_s"]),
                "phase": str(row["phase"]),
                "contact_point_count": len(contacts),
                "normal_impulse_sum_n_s": float(np.sum(impulses)),
                "actual_tangential_friction_point_count": len(friction),
                "actual_tangential_impulse_vector_sum_n_s": (
                    friction_vector_sum.tolist()
                ),
                "actual_tangential_impulse_magnitude_sum_n_s": (
                    friction_magnitude_sum
                ),
                "canonical_contact_points": canonical_contact_points,
                "friction_contact_points": friction_point_records,
                "surface_material_projection_complete": projection_available,
                "fingertip_surface_material_point_centroid_local_m": (
                    finger_centroid.tolist() if projection_available else None
                ),
                "object_surface_material_point_centroids_local_m": (
                    {
                        path: value.tolist()
                        for path, value in object_centroids.items()
                    }
                    if projection_available else None
                ),
                "solver_contact_point_centroid_link_local_m": np.average(
                    finger_solver_points, axis=0, weights=weights
                ).tolist(),
                "mean_tangential_step_m": step_mean,
                "interval_classification": interval_classification,
                "maximum_tangential_step_m": step_maximum,
                "mean_tangential_speed_m_s": speed_mean,
                "maximum_tangential_speed_m_s": speed_maximum,
                "mean_tangential_displacement_vector_world_m": (
                    None if tangential_vector is None
                    else tangential_vector.tolist()
                ),
                "cumulative_tangential_slip_m": cumulative_slip,
                "contact_patch_migration_event": migration_event,
            })

        if active_previous:
            contact_trace[-1]["contact_still_active_at_trace_end"] = True
        finger_complete = bool(
            contact_sample_count > 0
            and interval_count > 0
            and material_projection_complete
            and finger_projection_complete
            and friction_channel_present
        )
        measurement_complete = measurement_complete and finger_complete
        per_finger.append({
            "terminal_link": terminal_link,
            "measurement_complete": finger_complete,
            "contact_sample_count": contact_sample_count,
            "evaluated_interval_count": interval_count,
            "excluded_contact_boundary_interval_count": (
                contact_boundary_interval_count
            ),
            "contacted_object_part_paths": sorted(contacted_parts),
            "contact_establishment_events": establishment_events,
            "contact_loss_events": loss_events,
            "contact_patch_migration_events": migration_events,
            "cumulative_tangential_slip_m": cumulative_slip,
            "maximum_tangential_step_m": maximum_step,
            "maximum_tangential_speed_m_s": maximum_speed,
            "normal_impulse_sum_n_s": total_normal_impulse,
            "actual_tangential_impulse_magnitude_sum_n_s": (
                total_tangential_impulse_magnitude
            ),
            "peak_actual_tangential_impulse_magnitude_n_s": (
                peak_tangential_impulse_magnitude
            ),
            "fingertip_patch_migration_cumulative_m": (
                fingertip_patch_migration
            ),
            "object_patch_migration_cumulative_m": object_patch_migration,
            "contact_material_point_trace": contact_trace,
        })

    return {
        "status": "COMPLETE" if measurement_complete else "INCOMPLETE",
        "measurement_complete": measurement_complete,
        "method": (
            "POST_RUN_RIGID_MATERIAL_POINT_RELATIVE_MOTION_PROJECTED_ON_"
            "RECORDED_CONTACT_TANGENT"
        ),
        "normal_impulse_weighting": True,
        "mirror_deduplication": "TERMINAL_LINK_SENSOR_SIDE_ONLY",
        "velocity_method": "CONSECUTIVE_1_OVER_240_S_BODY_POSE_DIFFERENCE",
        "surface_material_projection_complete": material_projection_complete,
        "surface_material_projection_reason": material_projection_reason,
        "object_surface_projection_scope": (
            "CONTACT_PATH_SELECTS_SPLIT_RIGID_PART;SUPPLIER_FULL_SURFACE_"
            "PROJECTED_IN_THAT_PART_LOCAL_FRAME"
        ),
        "actual_tangential_friction_impulse_available": (
            friction_channel_present
        ),
        "friction_buffer_saturation_policy": "RUNTIME_ERROR_BEFORE_EVALUATION",
        "equivalent_contact_data": [
            "contact_world_point",
            "contact_world_normal",
            "normal_impulse",
            "fingertip_material_point_local",
            "object_material_point_local",
            "relative_tangential_speed",
        ],
        "online_control_used": False,
        "per_finger": per_finger,
        "maximum_per_finger_cumulative_tangential_slip_m": max(
            (row["cumulative_tangential_slip_m"] for row in per_finger),
            default=0.0,
        ),
    }


def _hand_object_pose_scope(
    document: Mapping[str, object], samples, *, reference_index: int,
    reference_rule: str, required_contact_samples: int,
    object_part_index: int = 0,
) -> dict[str, object]:
    """Evaluate one full hand-to-TE pose scope without online truth use."""

    required = int(required_contact_samples)
    part_index = int(object_part_index)
    if part_index < 0 or any(
        len(row.get("object_part_positions_m", ())) <= part_index
        or len(row.get("object_part_orientations_wxyz", ())) <= part_index
        for row in samples
    ):
        raise ValueError("requested object part pose is not present in every sample")

    def relative_pose(row: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
        world_from_hand = _pose_matrix(
            row["hand_base_position_m"], row["hand_base_orientation_wxyz"]
        )
        world_from_object = _pose_matrix(
            row["object_part_positions_m"][part_index],
            row["object_part_orientations_wxyz"][part_index],
        )
        hand_from_object = np.linalg.inv(world_from_hand) @ world_from_object
        return hand_from_object[:3, 3], hand_from_object[:3, :3]

    rows = samples[reference_index:]
    reference_position, reference_rotation = relative_pose(rows[0])
    reference_axis_hand = reference_rotation @ TE_INSERTION_AXIS_OBJECT
    lateral_projector = np.eye(3) - np.outer(
        reference_axis_hand, reference_axis_hand
    )
    trace = []
    previous_position = None
    previous_rotation = None
    previous_clock = None
    previous_mating = None
    previous_pin_plane = None
    cumulative_translation = 0.0
    cumulative_rotation = 0.0
    cumulative_clock = 0.0
    cumulative_mating_lateral = 0.0
    cumulative_pin_plane_lateral = 0.0
    key_projection_degenerate = False

    for row in rows:
        position, rotation = relative_pose(row)
        delta_rotation = reference_rotation.T @ rotation
        rotation_vector = Rotation.from_matrix(delta_rotation).as_rotvec()
        full_angle = float(np.linalg.norm(rotation_vector))
        axis_hand = rotation @ TE_INSERTION_AXIS_OBJECT
        axis_tilt = _vector_angle(reference_axis_hand, axis_hand)
        current_axis_object = delta_rotation @ TE_INSERTION_AXIS_OBJECT
        swing_back = _minimal_rotation_between(
            current_axis_object, TE_INSERTION_AXIS_OBJECT
        )
        swing_removed_key = (
            swing_back @ delta_rotation @ TE_MAIN_KEY_OBJECT
        )
        key_projection = swing_removed_key - TE_INSERTION_AXIS_OBJECT * float(
            np.dot(TE_INSERTION_AXIS_OBJECT, swing_removed_key)
        )
        key_norm = float(np.linalg.norm(key_projection))
        if key_norm <= 1.0e-12:
            key_projection_degenerate = True
            clock_angle = float("nan")
        else:
            key_projection /= key_norm
            clock_angle = math.atan2(
                float(np.dot(
                    TE_INSERTION_AXIS_OBJECT,
                    np.cross(TE_MAIN_KEY_OBJECT, key_projection),
                )),
                float(np.dot(TE_MAIN_KEY_OBJECT, key_projection)),
            )
        mating_point = (
            position + rotation @ TE_MATING_FACE_CENTER_OBJECT_M
        )
        pin_plane_point = (
            position + rotation @ TE_PIN_PLANE_CENTER_OBJECT_M
        )
        reference_mating = (
            reference_position
            + reference_rotation @ TE_MATING_FACE_CENTER_OBJECT_M
        )
        reference_pin_plane = (
            reference_position
            + reference_rotation @ TE_PIN_PLANE_CENTER_OBJECT_M
        )
        mating_lateral_vector = lateral_projector @ (
            mating_point - reference_mating
        )
        pin_plane_lateral_vector = lateral_projector @ (
            pin_plane_point - reference_pin_plane
        )

        if previous_position is not None:
            cumulative_translation += float(np.linalg.norm(
                position - previous_position
            ))
            cumulative_rotation += float(np.linalg.norm(
                Rotation.from_matrix(previous_rotation.T @ rotation).as_rotvec()
            ))
            if math.isfinite(clock_angle) and math.isfinite(previous_clock):
                clock_delta = math.atan2(
                    math.sin(clock_angle - previous_clock),
                    math.cos(clock_angle - previous_clock),
                )
                cumulative_clock += abs(clock_delta)
            cumulative_mating_lateral += float(np.linalg.norm(
                lateral_projector @ (mating_point - previous_mating)
            ))
            cumulative_pin_plane_lateral += float(np.linalg.norm(
                lateral_projector @ (pin_plane_point - previous_pin_plane)
            ))
        previous_position = position
        previous_rotation = rotation
        previous_clock = clock_angle
        previous_mating = mating_point
        previous_pin_plane = pin_plane_point

        trace.append({
            "step": int(row["step"]),
            "simulation_time_s": float(row["simulation_time_s"]),
            "phase": str(row["phase"]),
            "hand_from_object_translation_m": position.tolist(),
            "hand_from_object_orientation_wxyz": (
                _rotation_matrix_quaternion(rotation)
            ),
            "translation_change_from_t0_m": (
                position - reference_position
            ).tolist(),
            "relative_rotation_vector_from_t0_rad": rotation_vector.tolist(),
            "full_relative_rotation_from_t0_rad": full_angle,
            "insertion_axis_tilt_from_t0_rad": axis_tilt,
            "main_key_clock_change_from_t0_rad": (
                None if not math.isfinite(clock_angle) else clock_angle
            ),
            "main_key_clock_method": (
                "MINIMAL_SWING_REMOVAL_THEN_SIGNED_TWIST_ABOUT_INSERTION_AXIS"
            ),
            "mating_face_lateral_vector_from_t0_m": (
                mating_lateral_vector.tolist()
            ),
            "mating_face_lateral_offset_from_t0_m": float(
                np.linalg.norm(mating_lateral_vector)
            ),
            "pin_plane_lateral_vector_from_t0_m": (
                pin_plane_lateral_vector.tolist()
            ),
            "pin_plane_lateral_offset_from_t0_m": float(
                np.linalg.norm(pin_plane_lateral_vector)
            ),
            "cumulative_relative_translation_m": cumulative_translation,
            "cumulative_relative_rotation_rad": cumulative_rotation,
            "cumulative_main_key_clock_motion_rad": cumulative_clock,
            "cumulative_mating_face_lateral_motion_m": (
                cumulative_mating_lateral
            ),
            "cumulative_pin_plane_lateral_motion_m": (
                cumulative_pin_plane_lateral
            ),
        })

    translations = np.asarray([
        row["translation_change_from_t0_m"] for row in trace
    ], dtype=np.float64)
    rotation_vectors = np.asarray([
        row["relative_rotation_vector_from_t0_rad"] for row in trace
    ], dtype=np.float64)
    translation_norms = np.linalg.norm(translations, axis=1)
    clock_values = [
        row["main_key_clock_change_from_t0_rad"] for row in trace
        if row["main_key_clock_change_from_t0_rad"] is not None
    ]
    return {
        "status": "INCOMPLETE" if key_projection_degenerate else "COMPLETE",
        "measurement_complete": not key_projection_degenerate,
        "position_axis_measurement_complete": True,
        "key_measurement_complete": not key_projection_degenerate,
        "reference_t0": {
            "rule": reference_rule,
            "required_consecutive_samples": required,
            "step": int(rows[0]["step"]),
            "simulation_time_s": float(rows[0]["simulation_time_s"]),
            "phase": str(rows[0]["phase"]),
        },
        "task_geometry": {
            "insertion_axis_object": TE_INSERTION_AXIS_OBJECT.tolist(),
            "unique_main_key_object": TE_MAIN_KEY_OBJECT.tolist(),
            "mating_face_center_object_m": TE_MATING_FACE_CENTER_OBJECT_M.tolist(),
            "pin_plane_center_object_m": TE_PIN_PLANE_CENTER_OBJECT_M.tolist(),
            "source": "TE_OFFICIAL_CUSTOMER_VIEW_CAD_AND_FROZEN_ASSEMBLY_CONTRACT",
        },
        "maximum_absolute_translation_components_m": np.max(
            np.abs(translations), axis=0
        ).tolist(),
        "final_translation_components_m": translations[-1].tolist(),
        "maximum_translation_change_m": float(np.max(translation_norms)),
        "final_translation_change_m": float(translation_norms[-1]),
        "maximum_absolute_rotation_vector_components_rad": np.max(
            np.abs(rotation_vectors), axis=0
        ).tolist(),
        "final_rotation_vector_components_rad": rotation_vectors[-1].tolist(),
        "maximum_full_relative_rotation_rad": max(
            row["full_relative_rotation_from_t0_rad"] for row in trace
        ),
        "final_full_relative_rotation_rad": trace[-1][
            "full_relative_rotation_from_t0_rad"
        ],
        "maximum_insertion_axis_tilt_rad": max(
            row["insertion_axis_tilt_from_t0_rad"] for row in trace
        ),
        "final_insertion_axis_tilt_rad": trace[-1][
            "insertion_axis_tilt_from_t0_rad"
        ],
        "maximum_absolute_main_key_clock_change_rad": (
            max(abs(value) for value in clock_values)
            if clock_values else None
        ),
        "final_main_key_clock_change_rad": trace[-1][
            "main_key_clock_change_from_t0_rad"
        ],
        "maximum_mating_face_lateral_offset_m": max(
            row["mating_face_lateral_offset_from_t0_m"] for row in trace
        ),
        "final_mating_face_lateral_offset_m": trace[-1][
            "mating_face_lateral_offset_from_t0_m"
        ],
        "maximum_pin_plane_lateral_offset_m": max(
            row["pin_plane_lateral_offset_from_t0_m"] for row in trace
        ),
        "final_pin_plane_lateral_offset_m": trace[-1][
            "pin_plane_lateral_offset_from_t0_m"
        ],
        "cumulative_relative_translation_m": trace[-1][
            "cumulative_relative_translation_m"
        ],
        "cumulative_relative_rotation_rad": trace[-1][
            "cumulative_relative_rotation_rad"
        ],
        "cumulative_main_key_clock_motion_rad": trace[-1][
            "cumulative_main_key_clock_motion_rad"
        ],
        "cumulative_mating_face_lateral_motion_m": trace[-1][
            "cumulative_mating_face_lateral_motion_m"
        ],
        "cumulative_pin_plane_lateral_motion_m": trace[-1][
            "cumulative_pin_plane_lateral_motion_m"
        ],
        "key_projection_degenerate": key_projection_degenerate,
        "online_control_used": False,
        "pose_trace": trace,
    }


def _split_part_relative_motion_scope(samples) -> dict[str, object]:
    """Measure nut-to-body motion only over the disturbance time window."""

    if not samples or any(
        len(row.get("object_part_positions_m", ())) < 2
        or len(row.get("object_part_orientations_wxyz", ())) < 2
        for row in samples
    ):
        return {
            "status": "NOT_APPLICABLE_OR_UNAVAILABLE",
            "measurement_complete": False,
            "online_control_used": False,
        }

    relative = []
    for row in samples:
        world_from_body = _pose_matrix(
            row["object_part_positions_m"][0],
            row["object_part_orientations_wxyz"][0],
        )
        world_from_nut = _pose_matrix(
            row["object_part_positions_m"][1],
            row["object_part_orientations_wxyz"][1],
        )
        relative.append(np.linalg.inv(world_from_body) @ world_from_nut)
    reference = relative[0]
    translations = np.asarray(
        [value[:3, 3] - reference[:3, 3] for value in relative],
        dtype=np.float64,
    )
    delta_rotations = [
        reference[:3, :3].T @ value[:3, :3] for value in relative
    ]
    axis_tilts = np.asarray([
        math.acos(float(np.clip(value[2, 2], -1.0, 1.0)))
        for value in delta_rotations
    ], dtype=np.float64)
    twists = np.unwrap(np.asarray([
        math.atan2(float(value[1, 0]), float(value[0, 0]))
        for value in delta_rotations
    ], dtype=np.float64))
    translation_norms = np.linalg.norm(translations, axis=1)
    return {
        "status": "COMPLETE",
        "measurement_complete": True,
        "reference_rule": "ZERO_WRENCH_SAMPLE_AT_DISTURBANCE_START",
        "maximum_relative_translation_change_m": float(
            np.max(translation_norms)
        ),
        "final_relative_translation_change_m": float(translation_norms[-1]),
        "maximum_noncoaxial_axis_tilt_rad": float(np.max(axis_tilts)),
        "final_noncoaxial_axis_tilt_rad": float(axis_tilts[-1]),
        "minimum_coaxial_rotation_rad": float(np.min(twists)),
        "maximum_coaxial_rotation_rad": float(np.max(twists)),
        "final_coaxial_rotation_rad": float(twists[-1]),
        "total_coaxial_rotation_range_rad": float(np.ptp(twists)),
        "online_control_used": False,
    }


def _hand_object_pose_metrics(
    document: Mapping[str, object], samples, criteria
) -> dict[str, object]:
    """Report both acquisition and loaded hand-to-TE pose scopes."""

    if document.get("object_id") != TE_OBJECT_ID:
        return {
            "status": "NOT_APPLICABLE",
            "reason": "TE_TASK_GEOMETRY_ONLY",
            "online_control_used": False,
        }
    acquisition_index = next((
        index for index, row in enumerate(samples)
        if any(value > 0 for value in row["contacts"]["terminal_link_object"])
    ), None)
    if acquisition_index is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "NO_TERMINAL_LINK_OBJECT_CONTACT",
            "measurement_complete": False,
            "online_control_used": False,
        }
    required = int(criteria["sustained_three_contact_samples"])
    streak = 0
    loaded_index = None
    for index, row in enumerate(samples):
        if all(value > 0 for value in row["contacts"]["terminal_link_object"]):
            streak += 1
            if streak == required:
                loaded_index = index - required + 1
                break
        else:
            streak = 0
    acquisition = _hand_object_pose_scope(
        document,
        samples,
        reference_index=acquisition_index,
        reference_rule="FIRST_TERMINAL_LINK_POSITIVE_CONTACT_POSTRUN",
        required_contact_samples=1,
    )
    loaded = (
        {
            "status": "UNAVAILABLE",
            "reason": "NO_CONTINUOUS_THREE_FINGER_CONTACT_WINDOW",
            "measurement_complete": False,
            "online_control_used": False,
        }
        if loaded_index is None
        else _hand_object_pose_scope(
            document,
            samples,
            reference_index=loaded_index,
            reference_rule=(
                "FIRST_FRAME_OF_EARLIEST_CONTINUOUS_THREE_FINGER_POSITIVE_"
                "CONTACT_WINDOW"
            ),
            required_contact_samples=required,
        )
    )
    complete = bool(
        acquisition.get("measurement_complete") is True
        and loaded.get("measurement_complete") is True
    )
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "measurement_complete": complete,
        "acquisition_scope": acquisition,
        "loaded_scope": loaded,
        "online_control_used": False,
    }


def _hand_grasp_part_pose_metrics(
    document: Mapping[str, object], samples, criteria
) -> dict[str, object]:
    """Measure coupling-nut motion relative to the hand after three-pad contact."""

    part_paths = tuple(map(str, document.get(
        "tensor_contact_view_audit", {}
    ).get("object_sensor_paths", ())))
    part_count = len(samples[0].get("object_part_positions_m", ()))
    if part_count != 2 or len(part_paths) != 2:
        return {
            "status": "NOT_APPLICABLE",
            "reason": "OBJECT_IS_NOT_THE_TWO_PART_SPLIT_PLUG",
            "online_control_used": False,
        }

    required = int(criteria["sustained_three_contact_samples"])
    streak = 0
    loaded_index = None
    for index, row in enumerate(samples):
        if all(value > 0 for value in row["contacts"]["terminal_link_object"]):
            streak += 1
            if streak == required:
                loaded_index = index - required + 1
                break
        else:
            streak = 0
    if loaded_index is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "NO_CONTINUOUS_THREE_FINGER_CONTACT_WINDOW",
            "measurement_complete": False,
            "online_control_used": False,
        }

    scope = _hand_object_pose_scope(
        document,
        samples,
        reference_index=loaded_index,
        reference_rule=(
            "FIRST_FRAME_OF_EARLIEST_CONTINUOUS_THREE_FINGER_POSITIVE_"
            "CONTACT_WINDOW"
        ),
        required_contact_samples=required,
        object_part_index=1,
    )
    return {
        "status": "COMPLETE",
        "measurement_complete": True,
        "meaning": (
            "COUPLING_NUT_RIGID_POSE_CHANGE_RELATIVE_TO_HAND_AFTER_"
            "SUSTAINED_THREE_FINGER_CONTACT"
        ),
        "grasped_part_path": part_paths[1],
        "reference_t0": scope["reference_t0"],
        "maximum_relative_translation_m": scope[
            "maximum_translation_change_m"
        ],
        "final_relative_translation_m": scope["final_translation_change_m"],
        "maximum_absolute_translation_components_m": scope[
            "maximum_absolute_translation_components_m"
        ],
        "final_translation_components_m": scope[
            "final_translation_components_m"
        ],
        "maximum_relative_rotation_rad": scope[
            "maximum_full_relative_rotation_rad"
        ],
        "final_relative_rotation_rad": scope["final_full_relative_rotation_rad"],
        "cumulative_relative_translation_m": scope[
            "cumulative_relative_translation_m"
        ],
        "cumulative_relative_rotation_rad": scope[
            "cumulative_relative_rotation_rad"
        ],
        "online_control_used": False,
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
    table_support_indices = [
        index
        for index, row in enumerate(lift_rows)
        if float(
            row["contacts"].get(
                "object_table_positive_normal_impulse_n_s", 0.0
            )
        )
        > 0.0
    ]
    last_table_support_index = (
        table_support_indices[-1] if table_support_indices else None
    )
    peak_context = None
    if peak_index is not None and peak_center_index is not None:
        stencil_indices = (peak_index, peak_center_index, peak_index + 2 * window)
        stencil_rows = [lift_rows[index] for index in stencil_indices]
        peak_context = {
            "difference_window_samples": window,
            "stencil_simulation_times_s": [
                float(row["simulation_time_s"]) for row in stencil_rows
            ],
            "stencil_hand_base_z_m": [
                float(row["hand_base_position_m"][2]) for row in stencil_rows
            ],
            "center_arm_positions_rad": list(map(
                float, peak_row["active_positions_rad"][:7]
            )),
            "center_arm_velocities_rad_s": list(map(
                float, peak_row["active_velocities_rad_s"][:7]
            )),
            "center_arm_targets_rad": list(map(
                float, peak_row["active_targets_rad"][:7]
            )),
            "center_arm_control": peak_row.get("arm_control", {}),
            "center_contact_counts": peak_row.get("contacts", {}),
            "center_object_bottom_clearance_m": float(
                peak_row["object_bottom_clearance_m"]
            ),
            "last_table_support_lift_elapsed_s": (
                last_table_support_index * physics_dt_s
                if last_table_support_index is not None
                else None
            ),
            "peak_after_last_table_support_s": (
                (peak_center_index - last_table_support_index) * physics_dt_s
                if last_table_support_index is not None
                else None
            ),
        }
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
        "peak_context": peak_context,
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


def _prelift_transition_effort_metrics(
    document: Mapping[str, object], samples
) -> dict[str, object]:
    """Record whether the LP-required finger efforts exist when lift starts."""

    tare_rows = [
        np.asarray(row["active_efforts_nm"], dtype=np.float64)[7:]
        for row in samples
        if row["phase"] == "tare"
    ]
    tare = np.mean(np.stack(tare_rows), axis=0) if tare_rows else np.zeros(4)
    transition_rows = [
        row for row in samples if row["phase"] == "prelift_effort_check"
    ]
    if not transition_rows:
        transition_rows = [row for row in samples if row["phase"] == "preload"]
    if not transition_rows:
        return {
            "observed": False,
            "reason": "NO_PRELIFT_OR_PRELOAD_SAMPLE",
        }

    row = transition_rows[-1]
    pregrasp = np.asarray(
        document["motion_plan"]["pregrasp_hand_positions_rad"],
        dtype=np.float64,
    )
    final = np.asarray(
        document["motion_plan"]["final_hand_positions_rad"],
        dtype=np.float64,
    )
    closing_direction = np.sign(final - pregrasp)[1:]
    tare_subtracted = (
        np.asarray(row["active_efforts_nm"], dtype=np.float64)[7:] - tare
    )[1:]
    resistive = closing_direction * tare_subtracted
    required = np.asarray(
        document["required_closing_joint_effort_nm"], dtype=np.float64
    )
    tolerance = float(document["effort_regulation_tolerance_nm"])
    margin = resistive - required
    terminal_counts = np.asarray(
        row["contacts"]["terminal_link_object"], dtype=np.int64
    )
    return {
        "observed": True,
        "simulation_time_s": float(row["simulation_time_s"]),
        "source_phase": str(row["phase"]),
        "closing_joint_names": ["f1j2", "f2j1", "f3j2"],
        "tare_subtracted_effort_nm": tare_subtracted.tolist(),
        "resistive_closing_effort_nm": resistive.tolist(),
        "required_closing_effort_nm": required.tolist(),
        "required_effort_margin_nm": margin.tolist(),
        "all_required_efforts_reached": bool(np.all(margin >= 0.0)),
        "effort_regulation_tolerance_nm": tolerance,
        "all_efforts_within_target_band": bool(
            np.all(np.abs(margin) <= tolerance)
        ),
        "terminal_link_object_contact_records": terminal_counts.tolist(),
        "all_three_terminal_links_in_contact": bool(np.all(terminal_counts > 0)),
        "object_bottom_clearance_m": float(row["object_bottom_clearance_m"]),
    }


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


def _closing_order_evidence(
    document: Mapping[str, object], samples
) -> dict[str, object]:
    """Recover the executed order without treating a historical default as physics."""

    declared = []
    for source, container in (
        ("motion_plan", document.get("motion_plan", {})),
        ("controller_outcome", document.get("controller_outcome", {})),
    ):
        if not isinstance(container, Mapping):
            continue
        raw = container.get("closing_order_finger_numbers")
        if raw is None:
            continue
        values = tuple(raw) if isinstance(raw, Sequence) else ()
        valid = (
            len(values) == 3
            and all(type(value) is int for value in values)
            and set(values) == {1, 2, 3}
        )
        declared.append({
            "source": source,
            "finger_numbers": list(values),
            "valid": valid,
        })

    observed = []
    for row in samples:
        phase = str(row.get("phase", ""))
        parts = phase.split("_")
        if len(parts) < 3 or parts[0] != "finger" or parts[1] not in {"1", "2", "3"}:
            continue
        finger = int(parts[1])
        if finger not in observed:
            observed.append(finger)

    valid_declared = [
        tuple(item["finger_numbers"]) for item in declared if item["valid"]
    ]
    invalid_declared = any(not item["valid"] for item in declared)
    declarations_agree = len(set(valid_declared)) <= 1
    declared_order = valid_declared[0] if valid_declared else None
    observed_order = tuple(observed)
    observed_matches_declared = bool(
        declared_order is None
        or observed_order == declared_order[:len(observed_order)]
    )

    resolved_order = None
    source = None
    reason = None
    if invalid_declared:
        reason = "INVALID_DECLARED_CLOSING_ORDER"
    elif not declarations_agree:
        reason = "DECLARED_CLOSING_ORDERS_DISAGREE"
    elif not observed_matches_declared:
        reason = "TRACE_PHASE_ORDER_DISAGREES_WITH_DECLARATION"
    elif declared_order is not None:
        resolved_order = declared_order
        source = "EXPLICIT_TRACE_METADATA"
    elif len(observed_order) == 3 and set(observed_order) == {1, 2, 3}:
        resolved_order = observed_order
        source = "OBSERVED_TRACE_PHASE_SEQUENCE"
    else:
        reason = "FULL_CLOSING_ORDER_NOT_OBSERVED_OR_DECLARED"

    return {
        "resolved": resolved_order is not None,
        "finger_numbers": (
            None if resolved_order is None else list(resolved_order)
        ),
        "source": source,
        "reason": reason,
        "declared": declared,
        "observed_phase_prefix_finger_numbers": list(observed_order),
        "observed_phase_order_matches_declaration": observed_matches_declared,
    }


def _pregrasp_hold_hand_base_target_error(
    document: Mapping[str, object], samples: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Compare the final saved pregrasp hold pose with its frozen IK target."""

    result: dict[str, object] = {
        "status": "MISSING",
        "measurement_complete": False,
        "evaluated_postrun_only": True,
        "online_control_used": False,
        "truth_returned_to_controller": False,
        "sample_selection_rule": "LAST_PREGRASP_HOLD_SAMPLE",
    }
    rows = [row for row in samples if row.get("phase") == "pregrasp_hold"]
    if not rows:
        result["reason"] = "PREGRASP_HOLD_SAMPLE_ABSENT"
        return result
    target = np.asarray(
        document.get("motion_plan", {}).get("world_from_hand_base_target"),
        dtype=np.float64,
    )
    row = rows[-1]
    actual_position = np.asarray(row.get("hand_base_position_m"), dtype=np.float64)
    actual_orientation = np.asarray(
        row.get("hand_base_orientation_wxyz"), dtype=np.float64
    )
    if (
        target.shape != (16,)
        or not np.all(np.isfinite(target))
        or actual_position.shape != (3,)
        or not np.all(np.isfinite(actual_position))
        or actual_orientation.shape != (4,)
        or not np.all(np.isfinite(actual_orientation))
        or not np.isfinite(np.linalg.norm(actual_orientation))
        or np.linalg.norm(actual_orientation) <= 0.0
    ):
        result["status"] = "INVALID"
        result["reason"] = "TARGET_OR_RECORDED_HAND_BASE_POSE_INVALID"
        return result
    target_matrix = target.reshape(4, 4)
    if not np.allclose(
        target_matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-12
    ):
        result["status"] = "INVALID"
        result["reason"] = "TARGET_HAND_BASE_TRANSFORM_NOT_HOMOGENEOUS"
        return result
    target_position = target_matrix[:3, 3]
    target_orientation = _rotation_matrix_quaternion(target_matrix[:3, :3])
    error_vector = actual_position - target_position
    result.update({
        "status": "COMPLETE",
        "measurement_complete": True,
        "reason": None,
        "step": int(row["step"]),
        "simulation_time_s": float(row["simulation_time_s"]),
        "phase": "pregrasp_hold",
        "actual_pose_source": row.get("hand_base_pose_source"),
        "target_world_from_hand_base_row_major": target.tolist(),
        "target_position_world_m": target_position.tolist(),
        "target_orientation_world_wxyz": target_orientation,
        "actual_position_world_m": actual_position.tolist(),
        "actual_orientation_world_wxyz": actual_orientation.tolist(),
        "position_error_vector_world_m": error_vector.tolist(),
        "position_error_m": float(np.linalg.norm(error_vector)),
        "orientation_error_rad": _quaternion_distance(
            target_orientation, actual_orientation
        ),
    })
    return result


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
        "pad_nonpad_projection_tolerance_m": (
            PAD_NONPAD_PROJECTION_TOLERANCE_M
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

    required_terminal_links = TERMINAL_LINK_NAMES
    if document.get("mode") == "first-finger-diagnostic":
        closing_order = tuple(
            document.get("motion_plan", {}).get(
                "closing_order_finger_numbers", ()
            )
        )
        if (
            len(closing_order) != 3
            or any(type(value) is not int for value in closing_order)
            or set(closing_order) != {1, 2, 3}
        ):
            result["reason"] = "CLOSING_ORDER_INVALID"
            return result
        required_terminal_links = (
            TERMINAL_LINK_NAMES[closing_order[0] - 1],
        )
    result.update({
        "required_terminal_links": list(required_terminal_links),
        "required_contact_scope": (
            "FIRST_ACTIVE_FINGER_ONLY"
            if document.get("mode") == "first-finger-diagnostic"
            else "ALL_THREE_TERMINAL_LINKS"
        ),
    })

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
            if len(paths) != 2 or not _below(paths[1], object_root):
                continue
            links = [
                name for name in TERMINAL_LINK_NAMES
                if paths[0].endswith("/" + name)
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
                    "object_reference_position_m": list(map(
                        float, row["object_part_positions_m"][0]
                    )),
                    "object_reference_orientation_wxyz": list(map(
                        float, row["reference_part_orientation_wxyz"]
                    )),
                    "active_positions_rad": list(map(
                        float, row["active_positions_rad"]
                    )),
                    "object_table_positive_normal_impulse_n_s": float(
                        row["contacts"][
                            "object_table_positive_normal_impulse_n_s"
                        ]
                    ),
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
    required_links_have_points = True
    for name in TERMINAL_LINK_NAMES:
        points = np.asarray(local_points[name], dtype=np.float64).reshape(-1, 3)
        if not len(points):
            if name in required_terminal_links:
                required_links_have_points = False
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
        clear_pad = margin > PAD_NONPAD_PROJECTION_TOLERANCE_M
        clear_nonpad = margin < -PAD_NONPAD_PROJECTION_TOLERANCE_M
        boundary_ambiguous = ~(clear_pad | clear_nonpad)
        link_pass = bool(np.all(~clear_nonpad))
        all_points_are_pad = all_points_are_pad and link_pass

        steps = np.asarray(
            [metadata["step"] for metadata in point_metadata[name]],
            dtype=np.int64,
        )

        def first_projection_event(
            trigger_mask: np.ndarray, trigger_class: str
        ) -> dict[str, object] | None:
            trigger_indices = np.flatnonzero(trigger_mask)
            if not len(trigger_indices):
                return None
            first_step = int(np.min(steps[trigger_indices]))
            indices = np.flatnonzero(steps == first_step)
            metadata = point_metadata[name][int(indices[0])]
            simulation_time_s = float(metadata["simulation_time_s"])
            physics_dt_s = float(document["physics_dt_s"])
            class_counts = {
                "clear_pad": int(np.count_nonzero(clear_pad[indices])),
                "clear_nonpad": int(np.count_nonzero(clear_nonpad[indices])),
                "boundary_ambiguous": int(np.count_nonzero(
                    boundary_ambiguous[indices]
                )),
            }
            if class_counts["clear_nonpad"] and class_counts["clear_pad"]:
                surface_class = "PAD_AND_NONPAD_IN_SAME_PHYSICS_STEP"
            elif class_counts["boundary_ambiguous"]:
                surface_class = "PAD_NONPAD_BOUNDARY_UNRESOLVED"
            elif class_counts["clear_nonpad"]:
                surface_class = "CLEAR_NONPAD_ONLY"
            else:
                surface_class = "CLEAR_PAD_ONLY"
            return {
                "trigger_class": trigger_class,
                "step": first_step,
                "sample_time_s": simulation_time_s,
                "event_time_interval_s": [
                    max(0.0, simulation_time_s - physics_dt_s),
                    simulation_time_s,
                ],
                "phase": str(metadata["phase"]),
                "surface_classification": surface_class,
                "surface_class_point_counts": class_counts,
                "positive_contact_point_count": int(len(indices)),
                "minimum_nonpad_minus_pad_distance_m": float(
                    np.min(margin[indices])
                ),
                "maximum_nonpad_minus_pad_distance_m": float(
                    np.max(margin[indices])
                ),
                "active_positions_rad": metadata["active_positions_rad"],
                "object_center_m": metadata["object_center_m"],
                "object_reference_position_m": metadata[
                    "object_reference_position_m"
                ],
                "object_reference_orientation_wxyz": metadata[
                    "object_reference_orientation_wxyz"
                ],
                "object_table_positive_normal_impulse_n_s": metadata[
                    "object_table_positive_normal_impulse_n_s"
                ],
            }

        link_evidence = {
            "terminal_link": name,
            "positive_contact_point_count": len(points),
            "all_points_nearest_to_full_pad": link_pass,
            "surface_projection_point_counts": {
                "clear_pad": int(np.count_nonzero(clear_pad)),
                "clear_nonpad": int(np.count_nonzero(clear_nonpad)),
                "boundary_ambiguous": int(np.count_nonzero(
                    boundary_ambiguous
                )),
            },
            "first_positive_object_contact_event": first_projection_event(
                np.ones(len(points), dtype=bool), "ANY_POSITIVE_CONTACT"
            ),
            "first_clear_pad_contact_event": first_projection_event(
                clear_pad, "CLEAR_PAD"
            ),
            "first_clear_nonpad_contact_event": first_projection_event(
                clear_nonpad, "CLEAR_NONPAD"
            ),
            "first_boundary_ambiguous_contact_event": first_projection_event(
                boundary_ambiguous, "PAD_NONPAD_BOUNDARY_UNRESOLVED"
            ),
            "maximum_pad_surface_residual_m": float(np.max(pad_distance)),
            "minimum_nonpad_minus_pad_distance_m": float(np.min(margin)),
            "nearest_full_pad_source_face_index_min": int(np.min(
                pad_surface.source_face_indices[pad_face]
            )),
            "nearest_full_pad_source_face_index_max": int(np.max(
                pad_surface.source_face_indices[pad_face]
            )),
        }

        def contact_snapshot(phase: str, *, last: bool) -> dict[str, object] | None:
            times = [
                float(metadata["simulation_time_s"])
                for metadata in point_metadata[name]
                if metadata["phase"] == phase
            ]
            if not times:
                return None
            target_time = max(times) if last else min(times)
            indices = np.asarray([
                index
                for index, metadata in enumerate(point_metadata[name])
                if metadata["phase"] == phase
                and float(metadata["simulation_time_s"]) == target_time
            ], dtype=np.int64)
            minimum_index = int(indices[np.argmin(margin[indices])])
            metadata = [point_metadata[name][int(index)] for index in indices]
            world_points = np.asarray(
                [item["world_position_m"] for item in metadata], dtype=np.float64
            )
            world_normals = np.asarray(
                [item["world_normal"] for item in metadata], dtype=np.float64
            )
            normal_impulses = np.asarray(
                [item["normal_impulse_n_s"] for item in metadata],
                dtype=np.float64,
            )
            object_position = np.asarray(
                metadata[0]["object_reference_position_m"], dtype=np.float64
            )
            object_rotation = _quaternion_rotation_matrix(
                metadata[0]["object_reference_orientation_wxyz"]
            )
            object_points = (world_points - object_position) @ object_rotation
            object_mesh = inputs.object_contract.model.mesh
            object_closest, object_distance, object_face = proximity(
                object_mesh.face_vertices_m, object_points
            )
            object_normals = object_mesh.face_normals[object_face]
            mean_object_normal = np.mean(object_normals, axis=0)
            mean_object_normal /= np.linalg.norm(mean_object_normal)
            physics_dt_s = float(document["physics_dt_s"])
            friction = float(
                document.get("robustness_perturbation", {}).get(
                    "contact_friction_coefficient",
                    inputs.object_contract.contact_material_uncertainty.
                    friction_coefficient_interval[0],
                )
            )
            upward_force_upper_bound = float(np.sum(
                normal_impulses
                / physics_dt_s
                * (
                    np.abs(world_normals[:, 2])
                    + friction * np.sqrt(np.maximum(
                        0.0, 1.0 - np.square(world_normals[:, 2])
                    ))
                )
            ))
            object_mass_kg = float(inputs.object_contract.model.mass_kg)
            object_weight_n = object_mass_kg * 9.80665
            table_force_n = float(
                metadata[0]["object_table_positive_normal_impulse_n_s"]
            ) / physics_dt_s
            return {
                "simulation_time_s": target_time,
                "positive_contact_point_count": int(len(indices)),
                "contact_centroid_link_local_m": np.mean(
                    points[indices], axis=0
                ).tolist(),
                "minimum_nonpad_minus_pad_distance_m": float(
                    margin[minimum_index]
                ),
                "minimum_margin_point_link_local_m": points[
                    minimum_index
                ].tolist(),
                "contact_points_world_m": world_points.tolist(),
                "contact_normals_world": world_normals.tolist(),
                "normal_impulses_n_s": normal_impulses.tolist(),
                "terminal_normal_force_sum_n": float(
                    np.sum(normal_impulses) / physics_dt_s
                ),
                "friction_coefficient": friction,
                "friction_cone_upward_force_upper_bound_n": (
                    upward_force_upper_bound
                ),
                "object_contact_centroid_m": np.mean(
                    object_closest, axis=0
                ).tolist(),
                "object_contact_normal": mean_object_normal.tolist(),
                "maximum_object_surface_residual_m": float(
                    np.max(object_distance)
                ),
                "object_source_face_indices": object_face.tolist(),
                "active_positions_rad": metadata[0]["active_positions_rad"],
                "object_weight_n": object_weight_n,
                "object_table_normal_force_n": table_force_n,
                "object_table_support_fraction_of_weight": (
                    table_force_n / object_weight_n
                ),
                "all_points_nearest_to_full_pad": bool(
                    np.all(~clear_nonpad[indices])
                ),
            }

        last_prelift = contact_snapshot("prelift_effort_check", last=True)
        first_lift = contact_snapshot("lift", last=False)
        link_evidence["last_prelift_contact_snapshot"] = last_prelift
        link_evidence["first_lift_contact_snapshot"] = first_lift
        if last_prelift is not None and first_lift is not None:
            prelift_centroid = np.asarray(
                last_prelift["contact_centroid_link_local_m"], dtype=np.float64
            )
            lift_centroid = np.asarray(
                first_lift["contact_centroid_link_local_m"], dtype=np.float64
            )
            link_evidence["prelift_to_first_lift_centroid_delta_link_local_m"] = (
                lift_centroid - prelift_centroid
            ).tolist()
            link_evidence["prelift_to_first_lift_minimum_margin_delta_m"] = float(
                first_lift["minimum_nonpad_minus_pad_distance_m"]
                - last_prelift["minimum_nonpad_minus_pad_distance_m"]
            )
        nonpad_indices = np.flatnonzero(clear_nonpad)
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
    if not required_links_have_points:
        result["reason"] = "REQUIRED_TERMINAL_LINK_LACKS_POSITIVE_OBJECT_CONTACT"
        return result
    result.update({"verified": True, "reason": None})
    return result


def _nominal_three_stage_closure_witness(
    document: Mapping[str, object],
    samples,
    pad_identity_evidence: Mapping[str, object],
    order_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate one saved PhysX trajectory, not a continuous-U reach set."""

    result: dict[str, object] = {
        "evaluated": document.get("mode") == "grasp-lift",
        "passed": None,
        "status": "NOT_APPLICABLE",
        "evidence_class": "NOMINAL_DISCRETE_PHYSX_TRAJECTORY_WITNESS",
        "physics_dt_s": float(document["physics_dt_s"]),
        "continuous_time_contact_order_proved": False,
        "continuous_pose_or_friction_set_covered": False,
        "finite_robust_UB_AE_registered": False,
        "prior_contacts_forced_to_persist": False,
    }
    if document.get("mode") != "grasp-lift":
        return result
    if not order_evidence.get("resolved"):
        result.update({
            "status": "UNKNOWN",
            "reason": order_evidence.get("reason"),
        })
        return result
    if not pad_identity_evidence.get("contact_point_projection"):
        result.update({
            "status": "UNKNOWN",
            "reason": "PAD_NONPAD_CONTACT_POINT_PROJECTION_UNAVAILABLE",
        })
        return result

    closing_order = tuple(order_evidence["finger_numbers"])
    projections = {
        row["terminal_link"]: row
        for row in pad_identity_evidence["contact_point_projection"]
    }

    def stage_relative_center(row: Mapping[str, object]) -> np.ndarray:
        value = row.get("object_center_in_hand_base_m")
        if value is not None:
            return np.asarray(value, dtype=np.float64)
        return np.asarray(_relative_position(
            row["object_center_m"],
            row["hand_base_position_m"],
            row["hand_base_orientation_wxyz"],
        ), dtype=np.float64)

    def stage_relative_orientation(row: Mapping[str, object]) -> list[float]:
        world_from_hand = _quaternion_rotation_matrix(
            row["hand_base_orientation_wxyz"]
        )
        world_from_object = _quaternion_rotation_matrix(
            row["reference_part_orientation_wxyz"]
        )
        return _rotation_matrix_quaternion(
            world_from_hand.T @ world_from_object
        )

    def stage_motion_summary(
        finger: int, prior_fingers: tuple[int, ...]
    ) -> dict[str, object] | None:
        rows = [
            row for row in samples
            if str(row.get("phase", "")).startswith(f"finger_{finger}_")
        ]
        if not rows:
            return None
        relative_centers = np.asarray(
            [stage_relative_center(row) for row in rows], dtype=np.float64
        )
        world_centers = np.asarray(
            [row["object_center_m"] for row in rows], dtype=np.float64
        )
        relative_orientations = [
            stage_relative_orientation(row) for row in rows
        ]
        relative_center_change = np.linalg.norm(
            relative_centers - relative_centers[0], axis=1
        )
        world_center_change = np.linalg.norm(
            world_centers - world_centers[0], axis=1
        )
        relative_orientation_change = np.asarray([
            _quaternion_distance(relative_orientations[0], value)
            for value in relative_orientations
        ], dtype=np.float64)
        table_impulses = np.asarray([
            row["contacts"]["object_table_positive_normal_impulse_n_s"]
            for row in rows
        ], dtype=np.float64)
        prior_contact_evidence = {}
        for prior in prior_fingers:
            counts = np.asarray([
                row["contacts"]["terminal_link_object"][prior - 1]
                for row in rows
            ], dtype=np.int64)
            prior_contact_evidence[f"F{prior}"] = {
                "minimum_positive_contact_record_count": int(np.min(counts)),
                "maximum_positive_contact_record_count": int(np.max(counts)),
                "fraction_of_stage_samples_with_positive_contact": float(
                    np.mean(counts > 0)
                ),
                "contact_loss_observed_during_stage": bool(np.any(counts == 0)),
            }
        return {
            "active_finger": f"F{finger}",
            "sample_count": len(rows),
            "start_phase": str(rows[0]["phase"]),
            "end_phase": str(rows[-1]["phase"]),
            "start_time_s": float(rows[0]["simulation_time_s"]),
            "end_time_s": float(rows[-1]["simulation_time_s"]),
            "start_object_reference_position_m": rows[0][
                "object_part_positions_m"
            ][0],
            "end_object_reference_position_m": rows[-1][
                "object_part_positions_m"
            ][0],
            "start_object_reference_orientation_wxyz": rows[0][
                "reference_part_orientation_wxyz"
            ],
            "end_object_reference_orientation_wxyz": rows[-1][
                "reference_part_orientation_wxyz"
            ],
            "maximum_hand_relative_object_center_change_m": float(
                np.max(relative_center_change)
            ),
            "final_hand_relative_object_center_change_m": float(
                relative_center_change[-1]
            ),
            "maximum_hand_relative_object_orientation_change_rad": float(
                np.max(relative_orientation_change)
            ),
            "final_hand_relative_object_orientation_change_rad": float(
                relative_orientation_change[-1]
            ),
            "maximum_world_object_center_change_m": float(
                np.max(world_center_change)
            ),
            "final_world_object_center_change_m": float(
                world_center_change[-1]
            ),
            "object_table_positive_impulse_sample_count": int(
                np.count_nonzero(table_impulses > 0.0)
            ),
            "object_table_positive_impulse_present_all_stage_samples": bool(
                np.all(table_impulses > 0.0)
            ),
            "minimum_object_bottom_clearance_m": float(min(
                row["object_bottom_clearance_m"] for row in rows
            )),
            "maximum_object_bottom_clearance_m": float(max(
                row["object_bottom_clearance_m"] for row in rows
            )),
            "prior_finger_contact_evidence_diagnostic_only": (
                prior_contact_evidence
            ),
        }

    finger_results = []
    statuses = []
    stage_summaries = []
    for stage_index, finger in enumerate(closing_order):
        terminal_link = TERMINAL_LINK_NAMES[finger - 1]
        projection = projections.get(terminal_link, {})
        first_contact = projection.get("first_positive_object_contact_event")
        first_pad = projection.get("first_clear_pad_contact_event")
        first_nonpad = projection.get("first_clear_nonpad_contact_event")

        status = "UNKNOWN"
        reason = "FIRST_CONTACT_SURFACE_UNRESOLVED"
        if first_contact is None:
            status = "FAIL"
            reason = "NO_POSITIVE_OBJECT_CONTACT_OBSERVED"
        elif first_pad is None:
            if first_nonpad is not None:
                status = "FAIL"
                reason = "CLEAR_NONPAD_CONTACT_OBSERVED_BEFORE_ANY_CLEAR_PAD"
            else:
                reason = "ONLY_PAD_NONPAD_BOUNDARY_CONTACT_OBSERVED"
        else:
            first_counts = first_contact["surface_class_point_counts"]
            if first_counts["boundary_ambiguous"]:
                reason = "FIRST_CONTACT_STEP_CONTAINS_BOUNDARY_AMBIGUITY"
            elif first_counts["clear_nonpad"]:
                if first_counts["clear_pad"]:
                    reason = "PAD_AND_NONPAD_FIRST_OBSERVED_IN_SAME_STEP"
                else:
                    status = "FAIL"
                    reason = "CLEAR_NONPAD_IS_FIRST_OBSERVED_CONTACT"
            elif first_nonpad is None:
                status = "PASS"
                reason = "CLEAR_PAD_FIRST_AND_NO_CLEAR_NONPAD_OBSERVED"
            else:
                pad_interval = first_pad["event_time_interval_s"]
                nonpad_interval = first_nonpad["event_time_interval_s"]
                if pad_interval[1] < nonpad_interval[0]:
                    status = "PASS"
                    reason = "CLEAR_PAD_EVENT_INTERVAL_STRICTLY_PRECEDES_NONPAD"
                elif nonpad_interval[1] < pad_interval[0]:
                    status = "FAIL"
                    reason = "CLEAR_NONPAD_EVENT_INTERVAL_PRECEDES_PAD"
                else:
                    reason = "PAD_AND_NONPAD_EVENT_INTERVALS_OVERLAP"

        statuses.append(status)
        finger_results.append({
            "stage_index": stage_index + 1,
            "active_finger": f"F{finger}",
            "terminal_link": terminal_link,
            "status": status,
            "reason": reason,
            "first_positive_object_contact_event": first_contact,
            "first_clear_pad_contact_event": first_pad,
            "first_clear_nonpad_contact_event": first_nonpad,
            "first_boundary_ambiguous_contact_event": projection.get(
                "first_boundary_ambiguous_contact_event"
            ),
        })
        stage_summaries.append(stage_motion_summary(
            finger, closing_order[:stage_index]
        ))

    if "FAIL" in statuses:
        overall_status = "FAIL"
    elif "UNKNOWN" in statuses or any(row is None for row in stage_summaries):
        overall_status = "UNKNOWN"
    else:
        overall_status = "PASS"
    result.update({
        "passed": overall_status == "PASS",
        "status": overall_status,
        "reason": (
            None if overall_status == "PASS"
            else "AT_LEAST_ONE_FINGER_ORDER_OR_STAGE_IS_NOT_CERTIFIED"
        ),
        "closing_order_finger_numbers": list(closing_order),
        "contact_order_event_interval_convention": (
            "EACH_FIRST_POSITIVE_CONTACT_IS_BOUNDED_BY_PREVIOUS_AND_CURRENT_"
            "1_OVER_240_S_SAMPLE; OVERLAPPING_INTERVALS_ARE_UNKNOWN"
        ),
        "finger_first_contact_results": finger_results,
        "stage_coupled_motion_observations": stage_summaries,
    })
    return result


def _postgrasp_disturbance_metrics(
    document: Mapping[str, object], samples, inputs, physics_dt_s: float,
    *, pad_surface_identity_verified: bool,
) -> dict[str, object]:
    contract = document.get("postgrasp_disturbance")
    execution = document.get("postgrasp_disturbance_execution", {})
    if not isinstance(contract, Mapping):
        return {
            "status": "NOT_RUN",
            "requested": False,
            "online_object_or_contact_truth_used_for_control": False,
        }
    phases = {
        "postgrasp_disturbance_baseline",
        "postgrasp_disturbance_ramp_up",
        "postgrasp_disturbance_plateau",
        "postgrasp_disturbance_ramp_down",
        "postgrasp_disturbance_recovery",
    }
    rows = [row for row in samples if row.get("phase") in phases]
    if not rows:
        return {
            "status": "NOT_EXECUTED",
            "requested": True,
            "execution": dict(execution),
            "online_object_or_contact_truth_used_for_control": False,
        }
    qualification = contract["postrun_qualification"]
    expected_steps = {
        "postgrasp_disturbance_baseline": 1,
        "postgrasp_disturbance_ramp_up": round(
            float(contract["timing"]["ramp_up_s"]) / physics_dt_s
        ),
        "postgrasp_disturbance_plateau": round(
            float(contract["timing"].get("plateau_s", 0.0)) / physics_dt_s
        ),
        "postgrasp_disturbance_ramp_down": round(
            float(contract["timing"]["ramp_down_s"]) / physics_dt_s
        ),
        "postgrasp_disturbance_recovery": round(
            float(contract["timing"]["recovery_s"]) / physics_dt_s
        ),
    }
    observed_steps = {
        phase: sum(row["phase"] == phase for row in rows)
        for phase in expected_steps
    }
    timing_complete = observed_steps == expected_steps
    expected_phase_sequence = (
        ["postgrasp_disturbance_baseline"]
        + ["postgrasp_disturbance_ramp_up"] * expected_steps[
            "postgrasp_disturbance_ramp_up"
        ]
        + ["postgrasp_disturbance_plateau"] * expected_steps[
            "postgrasp_disturbance_plateau"
        ]
        + ["postgrasp_disturbance_ramp_down"] * expected_steps[
            "postgrasp_disturbance_ramp_down"
        ]
        + ["postgrasp_disturbance_recovery"] * expected_steps[
            "postgrasp_disturbance_recovery"
        ]
    )
    condition = contract["condition"]
    force_task_peak = np.asarray(condition["force_task_n"], dtype=np.float64)
    moment_task_peak = np.asarray(condition["moment_task_nm"], dtype=np.float64)
    world_from_task = np.asarray(
        contract["coordinate_contract"][
            "frozen_world_from_task_rotation_row_major"
        ],
        dtype=np.float64,
    )
    coordinate = contract["coordinate_contract"]
    frame_source = coordinate.get("frozen_world_task_frame_source")
    frame_source_sample = coordinate.get(
        "frozen_world_task_frame_source_sample", {}
    )
    vector_frame_verified = True
    vector_frame_maximum_error = 0.0
    if (
        frame_source
        == "CURRENT_RUN_HELD_HAND_POSE_FROZEN_BEFORE_DISTURBANCE"
    ):
        task_from_hand = np.asarray(
            coordinate.get("task_from_hand_rotation_row_major"),
            dtype=np.float64,
        )
        held_hand_quaternion = np.asarray(
            frame_source_sample.get("held_hand_orientation_world_wxyz"),
            dtype=np.float64,
        )
        if (
            world_from_task.shape != (3, 3)
            or task_from_hand.shape != (3, 3)
            or held_hand_quaternion.shape != (4,)
            or not np.all(np.isfinite(world_from_task))
            or not np.all(np.isfinite(task_from_hand))
            or not np.all(np.isfinite(held_hand_quaternion))
        ):
            vector_frame_verified = False
            vector_frame_maximum_error = float("inf")
        else:
            expected_world_from_task = (
                _pose_matrix(
                    np.zeros(3, dtype=np.float64),
                    held_hand_quaternion,
                )[:3, :3]
                @ task_from_hand.T
            )
            vector_frame_maximum_error = float(
                np.max(np.abs(world_from_task - expected_world_from_task))
            )
            vector_frame_verified = bool(
                vector_frame_maximum_error <= 1.0e-12
            )
    force_world_peak = world_from_task @ force_task_peak
    moment_world_peak = world_from_task @ moment_task_peak

    def minimum_jerk(value: float) -> float:
        return 10.0 * value ** 3 - 15.0 * value ** 4 + 6.0 * value ** 5

    ramp_up_count = expected_steps["postgrasp_disturbance_ramp_up"]
    ramp_down_count = expected_steps["postgrasp_disturbance_ramp_down"]
    expected_scales = [0.0]
    expected_scales.extend(
        minimum_jerk(index / ramp_up_count)
        for index in range(1, ramp_up_count + 1)
    )
    expected_scales.extend(
        [1.0] * expected_steps["postgrasp_disturbance_plateau"]
    )
    expected_scales.extend(
        1.0 - minimum_jerk(index / ramp_down_count)
        for index in range(1, ramp_down_count + 1)
    )
    expected_scales.extend(
        [0.0] * expected_steps["postgrasp_disturbance_recovery"]
    )
    input_errors = []
    if not vector_frame_verified:
        input_errors.append("DISTURBANCE_VECTOR_FRAME_MISMATCH")
    maximum_scale_error = 0.0
    maximum_force_task_error = 0.0
    maximum_moment_task_error = 0.0
    maximum_force_world_error = 0.0
    maximum_moment_world_error = 0.0
    maximum_application_point_error = 0.0
    if len(rows) != len(expected_phase_sequence):
        input_errors.append("DISTURBANCE_SAMPLE_COUNT_MISMATCH")
    for index, row in enumerate(rows[:len(expected_phase_sequence)]):
        if row["phase"] != expected_phase_sequence[index]:
            input_errors.append(f"PHASE_ORDER_MISMATCH_AT_{index}")
            continue
        payload = row.get("postgrasp_disturbance_input")
        if not isinstance(payload, Mapping):
            input_errors.append(f"INPUT_LOG_MISSING_AT_{index}")
            continue
        scale = float(payload.get("scale", float("nan")))
        expected_scale = expected_scales[index]
        if math.isfinite(scale):
            maximum_scale_error = max(
                maximum_scale_error, abs(scale - expected_scale)
            )
        else:
            maximum_scale_error = float("inf")
        expected_force_task = expected_scale * force_task_peak
        expected_moment_task = expected_scale * moment_task_peak
        expected_force_world = expected_scale * force_world_peak
        expected_moment_world = expected_scale * moment_world_peak

        def vector_error(key: str, expected: np.ndarray) -> float:
            value = np.asarray(payload.get(key), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                return float("inf")
            return float(np.max(np.abs(value - expected)))

        force_task_error = vector_error("force_task_n", expected_force_task)
        moment_task_error = vector_error(
            "moment_task_nm", expected_moment_task
        )
        force_world_error = vector_error("force_world_n", expected_force_world)
        moment_world_error = vector_error(
            "moment_world_nm", expected_moment_world
        )
        maximum_force_task_error = max(
            maximum_force_task_error, force_task_error
        )
        maximum_moment_task_error = max(
            maximum_moment_task_error, moment_task_error
        )
        maximum_force_world_error = max(
            maximum_force_world_error, force_world_error
        )
        maximum_moment_world_error = max(
            maximum_moment_world_error, moment_world_error
        )
        body_com_application = bool(
            contract.get("schema_version") == "te_postgrasp_disturbance_panel_v2"
        )
        application_point_error = 0.0
        if body_com_application:
            local_application = np.asarray(
                payload.get("application_point_supplier_object_m"),
                dtype=np.float64,
            )
            observed_application = np.asarray(
                payload.get("application_point_world_m"), dtype=np.float64
            )
            application_body_origin = np.asarray(
                payload.get("application_body_origin_world_m"),
                dtype=np.float64,
            )
            application_body_orientation = np.asarray(
                payload.get("application_body_orientation_world_wxyz"),
                dtype=np.float64,
            )
            expected_local = np.asarray(
                contract["coordinate_contract"][
                    "application_point_supplier_object_m"
                ],
                dtype=np.float64,
            )
            if (
                local_application.shape != (3,)
                or observed_application.shape != (3,)
                or application_body_origin.shape != (3,)
                or application_body_orientation.shape != (4,)
                or not np.all(np.isfinite(local_application))
                or not np.all(np.isfinite(observed_application))
                or not np.all(np.isfinite(application_body_origin))
                or not np.all(np.isfinite(application_body_orientation))
                or not np.array_equal(local_application, expected_local)
            ):
                application_point_error = float("inf")
            else:
                world_from_body = _pose_matrix(
                    application_body_origin,
                    application_body_orientation,
                )
                expected_application = _transform_point(
                    world_from_body, expected_local
                )
                application_point_error = float(np.max(np.abs(
                    observed_application - expected_application
                )))
            maximum_application_point_error = max(
                maximum_application_point_error, application_point_error
            )
        application_point_matches = bool(
            payload.get("application_point")
            == (
                "CURRENT_TE_BODY_CENTER_OF_MASS"
                if body_com_application
                else "CURRENT_TE_RIGID_BODY_ORIGIN"
            )
            and payload.get("application_point_world_readback_used")
            is body_com_application
            and (
                not body_com_application
                or payload.get("application_point_world_readback_scope")
                == "TEST_LOAD_APPLICATION_ONLY_NOT_GRASP_CONTROL"
            )
            and application_point_error <= 5.0e-7
        )
        if (
            payload.get("condition_id") != condition["condition_id"]
            or not application_point_matches
            or payload.get("is_global") is not True
            or payload.get("submission_tensor_backend") != "torch"
            or not str(payload.get("submission_tensor_device", "")).startswith(
                "cuda:"
            )
            or int(payload.get("submission_count_this_physics_step", -1)) != 1
            or not math.isfinite(scale)
            or abs(scale - expected_scale) > 1.0e-14
            or force_task_error > 1.0e-12
            or moment_task_error > 1.0e-12
            or force_world_error > 5.0e-7
            or moment_world_error > 5.0e-8
        ):
            input_errors.append(f"INPUT_VALUE_MISMATCH_AT_{index}")
    step_sequence = [int(row["step"]) for row in rows]
    if any(
        right != left + 1 for left, right in zip(
            step_sequence, step_sequence[1:]
        )
    ):
        input_errors.append("DISTURBANCE_STEPS_NOT_CONSECUTIVE")
    if int(execution.get("submitted_wrench_call_count", -1)) != len(rows):
        input_errors.append("WRENCH_SUBMISSION_COUNT_MISMATCH")
    if execution.get("observed_steps") != observed_steps:
        input_errors.append("EXECUTION_PHASE_COUNT_MISMATCH")
    expected_peak_count = (
        expected_steps["postgrasp_disturbance_plateau"] + 1
    )
    if expected_scales.count(1.0) != expected_peak_count:
        input_errors.append("PEAK_SCALE_COUNT_MISMATCH")
    input_profile_verified = bool(
        timing_complete and not input_errors
    )
    disturbance_start_step = int(rows[0]["step"])
    predisturbance_rows = [
        row for row in samples if int(row["step"]) < disturbance_start_step
    ]
    criteria = document["criteria"]
    predisturbance_motion = _motion_metrics(
        predisturbance_rows, criteria, physics_dt_s
    )
    predisturbance_contacts = _contact_metrics(
        predisturbance_rows, predisturbance_motion["grasped"]
    )
    predisturbance_safety = _safety_metrics(
        predisturbance_rows, criteria
    )
    predisturbance_nominal_pass = bool(
        predisturbance_motion["maximum_lift_m"]
        >= float(criteria["lift_distance_m"])
        and predisturbance_motion["hold_duration_s"]
        >= float(criteria["hold_duration_s"])
        and predisturbance_motion["table_released"]
        and predisturbance_contacts["maximum_sustained"]
        >= int(criteria["sustained_three_contact_samples"])
        and pad_surface_identity_verified
        and predisturbance_safety["finite"]
        and predisturbance_safety["collision_pass"]
        and predisturbance_safety["penetration_pass"]
    )

    slip = _contact_surface_slip_metrics(
        document, rows, inputs, physics_dt_s
    )
    slip_summary = []
    nut = qualification["coupling_nut_grip_region_supplier_object"]
    nut_points = []
    for finger in slip.get("per_finger", ()):
        for contact_row in finger.get("contact_material_point_trace", ()):
            for point in contact_row.get("canonical_contact_points", ()):
                value = point.get(
                    "object_surface_material_point_object_local_m"
                )
                if value is not None:
                    nut_points.append(np.asarray(value, dtype=np.float64))
        slip_summary.append({
            key: finger.get(key) for key in (
                "terminal_link",
                "measurement_complete",
                "contact_sample_count",
                "evaluated_interval_count",
                "excluded_contact_boundary_interval_count",
                "cumulative_tangential_slip_m",
                "maximum_tangential_step_m",
                "maximum_tangential_speed_m_s",
                "actual_tangential_impulse_magnitude_sum_n_s",
                "contact_establishment_events",
                "contact_loss_events",
            )
        })
    nut_array = np.asarray(nut_points, dtype=np.float64).reshape(-1, 3)
    nut_radii = (
        np.linalg.norm(nut_array[:, :2], axis=1)
        if len(nut_array) else np.asarray([], dtype=np.float64)
    )
    nut_region_pass = bool(
        len(nut_array)
        and np.all(nut_radii >= float(nut["minimum_radius_m"]))
        and np.all(nut_array[:, 2] >= float(nut["minimum_z_m"]))
        and np.all(nut_array[:, 2] <= float(nut["maximum_z_m"]))
    )

    pose = _hand_object_pose_scope(
        document,
        rows,
        reference_index=0,
        reference_rule="ZERO_WRENCH_SAMPLE_BEFORE_FROZEN_DISTURBANCE",
        required_contact_samples=1,
    )
    pose_trace = pose.pop("pose_trace")
    disturbance_nut_body_motion = _split_part_relative_motion_scope(rows)
    compact_pose_response = [
        {
            key: row.get(key)
            for key in (
                "step",
                "simulation_time_s",
                "phase",
                "hand_from_object_translation_m",
                "hand_from_object_orientation_wxyz",
                "translation_change_from_t0_m",
                "full_relative_rotation_from_t0_rad",
                "insertion_axis_tilt_from_t0_rad",
                "main_key_clock_change_from_t0_rad",
                "mating_face_lateral_offset_from_t0_m",
                "pin_plane_lateral_offset_from_t0_m",
            )
        }
        for row in pose_trace
    ]
    grasp_part_pose = None
    if len(rows[0].get("object_part_positions_m", ())) > 1:
        grasp_part_pose = _hand_object_pose_scope(
            document,
            rows,
            reference_index=0,
            reference_rule=(
                "ZERO_WRENCH_SAMPLE_BEFORE_FROZEN_DISTURBANCE_GRASP_PART"
            ),
            required_contact_samples=1,
            object_part_index=1,
        )
        grasp_part_pose.pop("pose_trace")

    def relative_transform(row: Mapping[str, object]) -> np.ndarray:
        world_from_hand = _pose_matrix(
            row["hand_base_position_m"], row["hand_base_orientation_wxyz"]
        )
        world_from_object = _pose_matrix(
            row["object_part_positions_m"][0],
            row["object_part_orientations_wxyz"][0],
        )
        return np.linalg.inv(world_from_hand) @ world_from_object

    start = relative_transform(rows[0])
    final = relative_transform(rows[-1])
    axis_hand = start[:3, :3] @ TE_INSERTION_AXIS_OBJECT
    lateral_projector = np.eye(3) - np.outer(axis_hand, axis_hand)
    engagement_length = float(qualification["engagement_length_m"])
    face_object = TE_MATING_FACE_CENTER_OBJECT_M
    end_object = (
        TE_MATING_FACE_CENTER_OBJECT_M
        - engagement_length * TE_INSERTION_AXIS_OBJECT
    )

    def lateral_delta(point_object: np.ndarray) -> np.ndarray:
        initial_point = _transform_point(start, point_object)
        final_point = _transform_point(final, point_object)
        return lateral_projector @ (final_point - initial_point)

    face_delta_hand = lateral_delta(face_object)
    end_delta_hand = lateral_delta(end_object)
    shell_face_error = float(np.linalg.norm(face_delta_hand))
    shell_end_error = float(np.linalg.norm(end_delta_hand))
    shell_error = max(shell_face_error, shell_end_error)
    shell_limit = float(
        qualification["recovery_shell_two_point_lateral_limit_m"]
    )

    task_from_object = np.asarray(
        contract["coordinate_contract"][
            "task_from_supplier_object_rotation_row_major"
        ],
        dtype=np.float64,
    )
    face_delta_object = start[:3, :3].T @ face_delta_hand
    face_delta_task = task_from_object @ face_delta_object
    clock = pose["final_main_key_clock_change_rad"]
    key_radius = float(qualification["main_key_radius_m"])
    key_angles = (0.0, 80.0, 142.0, 196.0, 293.0)
    key_rows = []
    for index, angle_degrees in enumerate(key_angles):
        angle = math.radians(angle_degrees)
        tangent = np.asarray((-math.sin(angle), math.cos(angle)))
        clearance = float(
            qualification[
                "main_key_half_width_clearance_m"
                if index == 0 else "minor_key_half_width_clearance_m"
            ]
        )
        error = (
            None if clock is None else abs(
                float(np.dot(tangent, face_delta_task[:2]))
                + key_radius * math.sin(float(clock))
            )
        )
        key_rows.append({
            "key_index": index,
            "center_angle_task_deg": angle_degrees,
            "half_width_clearance_m": clearance,
            "residual_tangential_error_m": error,
            "remaining_margin_m": (
                None if error is None else clearance - error
            ),
            "passed": bool(error is not None and error < clearance),
        })
    conservative_key_error = (
        None if clock is None else
        float(np.linalg.norm(face_delta_task[:2]))
        + key_radius * abs(math.sin(float(clock)))
    )
    conservative_key_limit = float(
        qualification["recovery_conservative_key_envelope_limit_m"]
    )

    all_three_contact = bool(all(
        all(int(value) > 0 for value in row["contacts"]["terminal_link_object"])
        for row in rows
    ))
    table_recontact = any(
        int(row["contacts"]["object_table"]) > 0 for row in rows
    )
    unauthorized_keys = (
        "robot_object_unauthorized", "robot_table", "robot_fixture",
        "robot_unclassified",
    )
    unauthorized = {
        key: sum(int(row["contacts"].get(key, 0)) for row in rows)
        for key in unauthorized_keys
    }
    unauthorized_pass = all(value == 0 for value in unauthorized.values())
    recovery_complete = (
        observed_steps["postgrasp_disturbance_recovery"]
        == expected_steps["postgrasp_disturbance_recovery"]
    )
    safety_pass = bool(
        execution.get("completed") is True
        and execution.get("failure_reason") is None
        and document["controller_outcome"].get("failure_reason") is None
    )
    translation_limit = float(qualification["precision_limit_translation_m"])
    axis_tilt_limit = float(qualification["precision_limit_axis_tilt_rad"])
    position_axis_precision_pass = bool(
        pose.get("position_axis_measurement_complete") is True
        and pose.get("maximum_translation_change_m") is not None
        and float(pose["maximum_translation_change_m"])
        <= translation_limit
        and pose.get("maximum_insertion_axis_tilt_rad") is not None
        and float(pose["maximum_insertion_axis_tilt_rad"])
        <= axis_tilt_limit
    )
    position_axis_primary_scope = bool(
        contract.get("schema_version") == POSTGRASP_DISTURBANCE_COM_SCHEMA
    )
    key_precision_pass = bool(
        all(row["passed"] for row in key_rows)
        and conservative_key_error is not None
        and conservative_key_error < conservative_key_limit
    )
    core_pass = bool(
        timing_complete
        and input_profile_verified
        and predisturbance_nominal_pass
        and recovery_complete
        and slip.get("measurement_complete") is True
        and (
            pose.get("position_axis_measurement_complete") is True
            if position_axis_primary_scope
            else pose.get("measurement_complete") is True
        )
        and pad_surface_identity_verified
        and all_three_contact
        and nut_region_pass
        and not table_recontact
        and unauthorized_pass
        and safety_pass
        and (
            position_axis_precision_pass
            if position_axis_primary_scope
            else shell_error < shell_limit and key_precision_pass
        )
    )
    return {
        "status": "PASS" if core_pass else "FAIL",
        "requested": True,
        "condition": dict(contract["condition"]),
        "execution": dict(execution),
        "expected_steps": expected_steps,
        "observed_steps": observed_steps,
        "timing_complete": timing_complete,
        "input_profile": {
            "verified": input_profile_verified,
            "errors": input_errors,
            "expected_sample_count": len(expected_phase_sequence),
            "observed_sample_count": len(rows),
            "maximum_scale_error": maximum_scale_error,
            "maximum_force_task_error_n": maximum_force_task_error,
            "maximum_moment_task_error_nm": maximum_moment_task_error,
            "maximum_force_world_error_n": maximum_force_world_error,
            "maximum_moment_world_error_nm": maximum_moment_world_error,
            "maximum_application_point_world_error_m": (
                maximum_application_point_error
            ),
            "submitted_wrench_call_count": execution.get(
                "submitted_wrench_call_count"
            ),
            "peak_scale_occurrence_count": expected_scales.count(1.0),
            "expected_peak_scale_occurrence_count": expected_peak_count,
            "wrench_directions_frozen_in_world": execution.get(
                "wrench_directions_frozen_in_world"
            ),
            "online_object_pose_readback_used": execution.get(
                "online_object_pose_readback_used"
            ),
            "vector_frame_source": frame_source,
            "vector_frame_verified": vector_frame_verified,
            "vector_frame_maximum_rotation_entry_error": (
                vector_frame_maximum_error
            ),
        },
        "predisturbance_nominal_qualification": {
            "evaluated_postrun_only": True,
            "sample_count": len(predisturbance_rows),
            "maximum_lift_m": predisturbance_motion["maximum_lift_m"],
            "hold_duration_s": predisturbance_motion["hold_duration_s"],
            "table_released_during_hold": predisturbance_motion[
                "table_released"
            ],
            "maximum_consecutive_three_contact_samples": (
                predisturbance_contacts["maximum_sustained"]
            ),
            "finite": predisturbance_safety["finite"],
            "collision_pass": predisturbance_safety["collision_pass"],
            "penetration_pass": predisturbance_safety["penetration_pass"],
            "pad_surface_identity_verified_for_full_run": (
                pad_surface_identity_verified
            ),
            "passed": predisturbance_nominal_pass,
            "online_control_used": False,
        },
        "all_three_legal_pad_contacts_every_sample": bool(
            all_three_contact and pad_surface_identity_verified
        ),
        "drop_or_contact_loss_observed": not all_three_contact,
        "table_recontact_observed": table_recontact,
        "unauthorized_contact_records": unauthorized,
        "coupling_nut_grip_region": {
            "evaluated_contact_point_count": int(len(nut_array)),
            "minimum_radius_m": (
                None if not len(nut_radii) else float(np.min(nut_radii))
            ),
            "maximum_radius_m": (
                None if not len(nut_radii) else float(np.max(nut_radii))
            ),
            "minimum_z_m": (
                None if not len(nut_array) else float(np.min(nut_array[:, 2]))
            ),
            "maximum_z_m": (
                None if not len(nut_array) else float(np.max(nut_array[:, 2]))
            ),
            "passed": nut_region_pass,
            "contract": dict(nut),
        },
        "per_finger_contact_surface_slip": slip_summary,
        "pose_peak_and_residual": pose,
        "pose_response_trace": compact_pose_response,
        "grasp_part_pose_peak_and_residual": grasp_part_pose,
        "disturbance_nut_body_relative_motion": (
            disturbance_nut_body_motion
        ),
        "recovery_shell_two_point": {
            "mating_face_lateral_error_m": shell_face_error,
            "engagement_end_lateral_error_m": shell_end_error,
            "maximum_error_m": shell_error,
            "limit_m": shell_limit,
            "remaining_margin_m": shell_limit - shell_error,
            "passed": shell_error < shell_limit,
        },
        "recovery_key_clearance": {
            "face_lateral_delta_task_m": face_delta_task.tolist(),
            "main_key_clock_change_rad": clock,
            "per_key": key_rows,
            "conservative_envelope_error_m": conservative_key_error,
            "conservative_envelope_limit_m": conservative_key_limit,
            "conservative_remaining_margin_m": (
                None if conservative_key_error is None else
                conservative_key_limit - conservative_key_error
            ),
            "passed": key_precision_pass,
            "primary_condition": not position_axis_primary_scope,
        },
        "position_axis_precision": {
            "maximum_translation_change_m": pose.get(
                "maximum_translation_change_m"
            ),
            "translation_limit_m": translation_limit,
            "maximum_insertion_axis_tilt_rad": pose.get(
                "maximum_insertion_axis_tilt_rad"
            ),
            "axis_tilt_limit_rad": axis_tilt_limit,
            "passed": position_axis_precision_pass,
            "primary_condition": position_axis_primary_scope,
        },
        "safety_pass": safety_pass,
        "core_condition_pass": core_pass,
        "conclusion_scope": "ONLY_THIS_PREREGISTERED_FINITE_WRENCH_CONDITION",
        "online_object_or_contact_truth_used_for_control": False,
        "postrun_truth_evaluation_only": True,
    }


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
    prelift_transition = _prelift_transition_effort_metrics(document, samples)
    contact_surface_slip = _contact_surface_slip_metrics(
        document, samples, inputs, physics_dt_s
    )
    hand_object_pose = _hand_object_pose_metrics(document, samples, criteria)
    hand_grasp_part_pose = _hand_grasp_part_pose_metrics(
        document, samples, criteria
    )
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
    )
    pad_identity_evidence = _derive_pad_surface_identity(
        document, samples, robot_asset_path, inputs
    )
    pad_identity = bool(pad_identity_evidence["verified"])
    postgrasp_disturbance = _postgrasp_disturbance_metrics(
        document,
        samples,
        inputs,
        physics_dt_s,
        pad_surface_identity_verified=pad_identity,
    )
    pregrasp_hold_pose_error = _pregrasp_hold_hand_base_target_error(
        document, samples
    )
    order_evidence = _closing_order_evidence(document, samples)
    closing_order = tuple(
        order_evidence["finger_numbers"] or ()
    )
    closure_witness = _nominal_three_stage_closure_witness(
        document, samples, pad_identity_evidence, order_evidence
    )
    observed_order_prefix = tuple(
        order_evidence["observed_phase_prefix_finger_numbers"]
    )
    first_finger_number = (
        closing_order[0] if closing_order
        else observed_order_prefix[0] if observed_order_prefix
        else None
    )
    first_phase = (
        None if first_finger_number is None else f"finger_{first_finger_number}"
    )
    first_hold = [] if first_phase is None else [
        row for row in samples if row["phase"] == f"{first_phase}_hold"
    ]
    confirmation = None if first_phase is None else next((
        index for index, row in enumerate(samples)
        if row["phase"] == f"{first_phase}_contact_confirmed"
    ), None)
    evidence = [] if confirmation is None else samples[max(
        0, confirmation - int(criteria["sustained_three_contact_samples"])):confirmation]
    first_contact_phases = () if first_phase is None else (
        f"{first_phase}_contact_confirmed",
        f"{first_phase}_contact_settle",
        f"{first_phase}_hold",
    )
    first_contact = evidence + [
        row for row in samples if row["phase"] in first_contact_phases
    ]
    proxy, terminal = (bool(document["controller_outcome"].get("contact_targets_rad")),
                       first_finger_number is not None and any(
                           row["contacts"]["terminal_link_object"][
                               first_finger_number - 1
                           ] > 0
                           for row in first_contact
                       ))
    witness_complete = document.get("contact_report_api_audit", {}).get("complete") is True
    contact_class = ("NO_CONTACT_PROXY" if not proxy else
                     "UNRESOLVED_CONTACT_REPORT_API_COVERAGE" if not witness_complete else
                     "UNRESOLVED_CONTACT_REPORT_DISAGREEMENT" if not contacts["channels_agree"] else
                     "FALSE_CONTACT_PROXY" if not terminal
                     else "UNRESOLVED_TERMINAL_LINK_CONTACT_PATCH" if not pad_identity else "ALLOWED_PAD_CONTACT")
    pregrasp_hand = document.get("motion_plan", {}).get("pregrasp_hand_positions_rad")
    inactive_slots = tuple(
        slot for slot in (1, 2, 3)
        if first_finger_number is not None and slot != first_finger_number
    )
    only_first_terminal_contacted = bool(
        first_finger_number is not None
        and terminal
        and all(
            row["contacts"]["terminal_link_object"][slot - 1] == 0
            for row in samples
            for slot in inactive_slots
        )
    )
    only_first = bool(
        first_finger_number is not None
        and len(document["controller_outcome"].get("contact_targets_rad", ())) == 1
        and pregrasp_hand is not None
        and all(
            np.allclose(
                np.asarray(row["active_targets_rad"], dtype=np.float64)[
                    [7 + slot for slot in inactive_slots]
                ],
                np.asarray(pregrasp_hand, dtype=np.float64)[list(inactive_slots)],
                atol=1.0e-12,
                rtol=0.0,
            )
            for row in first_hold
        )
    )
    first_controller_pass = bool(
        document["mode"] == "first-finger-diagnostic" and all(shared_passes)
        and len(first_hold) * physics_dt_s >= float(criteria["first_finger_diagnostic_duration_s"])
        and document["controller_outcome"]["maximum_finger_target_delta_rad"]
        <= float(criteria["maximum_finger_target_increment_rad"]) + 1.0e-12
        and only_first
        and only_first_terminal_contacted
        and contact_class == "ALLOWED_PAD_CONTACT")
    truth_isolation = bool(
        document.get("online_object_or_contact_truth_used") is False
        and document.get("truth_audit_data_returned_to_controller") is False
        and document.get("object_pose_writes_after_start") == 0)
    nominal_three_stage_dynamic_witness_pass = bool(
        nominal_physical_pass
        and pad_identity
        and closure_witness["passed"] is True
        and truth_isolation
        and document.get("accepted_preflight_bound") is True
    )
    research_pass = bool(
        nominal_three_stage_dynamic_witness_pass
        and document.get("offline_task_gate_passed") is True
    )
    return {
        "schema_version": "carts_grasp_v2_dynamic_evaluation_v4",
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
        "legacy_maximum_hand_base_to_object_center_drift_m": motion[
            "maximum_slip_m"
        ],
        "legacy_maximum_object_world_orientation_change_rad": motion[
            "maximum_orientation_change_rad"
        ],
        "legacy_motion_metric_semantics": {
            "maximum_relative_slip_m": (
                "HAND_BASE_TO_OBJECT_CENTER_DRIFT_NOT_CONTACT_SURFACE_SLIP"
            ),
            "maximum_orientation_change_rad": (
                "OBJECT_WORLD_ORIENTATION_CHANGE_NOT_HAND_OBJECT_RELATIVE_ROTATION"
            ),
        },
        "fingertip_contact_surface_slip": contact_surface_slip,
        "hand_object_full_relative_pose": hand_object_pose,
        "hand_grasp_part_relative_pose": hand_grasp_part_pose,
        "pregrasp_hold_hand_base_target_error": pregrasp_hold_pose_error,
        "postgrasp_disturbance": postgrasp_disturbance,
        "te_stability_recording_complete": bool(
            contact_surface_slip.get("measurement_complete") is True
            and hand_object_pose.get("measurement_complete") is True
            and pad_identity
        ),
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
        "lift_peak_acceleration_context": acceleration["peak_context"],
        "registered_lift_peak_acceleration_m_s2": acceleration["registered"],
        "lift_acceleration_consistent": acceleration["passed"],
        "lift_acceleration_diagnostic_only": True,
        "lift_acceleration_hard_gate_used": False,
        "lift_acceleration_measurement": (
            "FINITE_DIFFERENCE_OF_HAND_BASE_WORLD_Z_DURING_LIFT"
        ),
        "lift_acceleration_safety_specification_source": None,
        "maximum_table_penetration_m": safety["overall_penetration_m"],
        "maximum_post_settle_table_penetration_m": safety["post_settle_penetration_m"],
        "unauthorized_contact_records": safety["unauthorized"],
        "first_unauthorized_contact_paths": contacts["examples"],
        "first_terminal_link_object_paths": contacts["terminal_examples"],
        "contact_report_channels_agree": contacts["channels_agree"],
        "contact_report_api_complete": witness_complete,
        "closing_order_finger_numbers": (
            None if not closing_order else list(closing_order)
        ),
        "closing_order_evidence": order_evidence,
        "nominal_three_stage_closure_witness": closure_witness,
        "first_active_finger_number": first_finger_number,
        "first_finger_contact_classification": contact_class,
        "first_finger_hold_duration_s": len(first_hold) * physics_dt_s,
        "first_finger_maximum_target_delta_rad": document["controller_outcome"].get("maximum_finger_target_delta_rad"), "only_first_finger_commanded": only_first,
        "only_first_terminal_link_contacted": only_first_terminal_contacted,
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
        "maximum_payload_compensation_nm": document["controller_outcome"].get(
            "maximum_payload_compensation_nm"
        ),
        "payload_compensation_model": document["controller_outcome"].get(
            "payload_compensation_model"
        ),
        "lift_arm_damping_switch_audit": document["controller_outcome"].get(
            "lift_arm_damping_switch_audit"
        ),
        "finger_clamp_effort_nm": finger_efforts,
        "prelift_transition_effort_nm": prelift_transition,
        "truth_isolation_pass": truth_isolation,
        "accepted_preflight_bound": bool(document.get("accepted_preflight_bound")),
        "accepted_preflight_evaluation_sha256": document.get("accepted_preflight_evaluation_sha256"),
        "preflight_pass": False,
        **pending_engine_fields(
            controller_preflight_pass,
            bool(document.get("identity_hash_check_pass", False)),
        ),
        "nominal_diagnostic_pass": nominal_physical_pass,
        "nominal_three_stage_dynamic_witness_pass": (
            nominal_three_stage_dynamic_witness_pass
        ),
        "legacy_offline_task_gate_passed": bool(
            document.get("offline_task_gate_passed")
        ),
        "legacy_offline_task_gate_used_for_nominal_witness": False,
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
    if trace.get("mode") == "isolated-hand":
        result = evaluate_isolated_hand_trace(trace)
    else:
        config_path = trace.get("runtime", {}).get("config_path")
        if not isinstance(config_path, str):
            parser.error("dynamic trace lacks its bound V2 config path")
        from kcg_connector.grasp.carts_v2.models import load_v2_inputs

        repository = Path(__file__).resolve().parents[4]
        inputs = load_v2_inputs(
            repository,
            config_path=Path(config_path).resolve(),
            object_id=str(trace["object_id"]),
        )
        result = evaluate_trace(
            trace,
            robot_asset_path=Path(arguments.robot_asset).resolve(),
            inputs=inputs,
        )
    Path(arguments.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if trace.get("mode") == "isolated-hand":
        passed = result["diagnostic_pass"]
    elif trace.get("mode") == "first-finger-diagnostic":
        passed = result["controller_first_finger_diagnostic_pass"]
    elif trace.get("mode") == "grasp-lift":
        passed = result["nominal_three_stage_dynamic_witness_pass"]
    else:
        passed = result["preflight_pass"]
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IsolatedHandRecorder", "TruthAuditRecorder", "audit_initial_joint_state",
    "audit_mimic_schema", "compare_reference_targets",
    "evaluate_isolated_hand_trace", "evaluate_trace",
]
