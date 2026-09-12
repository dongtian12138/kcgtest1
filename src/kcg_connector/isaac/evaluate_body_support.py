#!/usr/bin/env python3
"""Evaluate released-body support from saved truth, never in the online loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ijson
import numpy as np
from scipy.spatial.transform import Rotation
from trace_metadata import iter_truth_samples


def evaluate(directory: Path, phase_kind: str = "support", stage_name: str | None = None):
    stage_paths = {"support": "body_support/support_release_controller_result.json",
                   "nut_regrasp": "nut_regrasp/nut_regrasp_controller_result.json",
                   "nut_rotation": "nut_rotation/nut_rotation_controller_result.json"}
    stage_path = Path(stage_paths[phase_kind])
    if stage_name is not None:
        if Path(stage_name).name != stage_name or stage_name in (".", ".."):
            raise ValueError("stage name must identify one local stage directory")
        stage_path = Path(stage_name) / stage_path.name
    support_path = directory / "socket_transport" / stage_path
    support = json.loads(support_path.read_text())
    scene = json.loads((directory / "assembly_scene.json").read_text())
    part_paths = {name: scene["collision"][name + "_collision"].rsplit("/", 1)[0]
                  for name in ("body", "nut")}
    socket_position = np.asarray(scene["socket_initial_position_world_m"])
    start = int(support["first_step"]) - 1
    end = int(support["last_step"]) - 1
    # Same source-CAD front key edges and slot sections as the existing key-entry
    # evaluator. These are evaluation geometry, not controller tolerances.
    keys = np.asarray([10.0, 90.0, 157.0, 254.0, 308.0])
    widths = np.asarray([4.02158, 7.73810, 4.02158, 4.02158, 4.02158])
    slot_centers = (180.0 - keys) % 360.0
    slot_widths = np.asarray([4.817476, 9.643492, 4.817476, 4.817476, 4.817476])
    angles = np.deg2rad(np.column_stack((keys - widths / 2, keys + widths / 2)).ravel())
    edges = np.column_stack((0.0188214 * np.cos(angles), 0.0188214 * np.sin(angles),
                            np.full(len(angles), -0.000762)))
    rows, final_truth, preclose_truth = [], None, None
    for actual in iter_truth_samples(directory):
        step = int(actual["step"])
        if step < start:
            continue
        if step > end:
            break
        if (phase_kind == "nut_regrasp" and preclose_truth is None
                and actual["phase"] == "key_probe_nut_tare"):
            preclose_truth = final_truth
        if phase_kind == "nut_rotation" and preclose_truth is None:
            preclose_truth = actual
        final_truth = actual
        positions = np.asarray(actual["object_part_positions_m"])
        rotations = Rotation.from_quat(np.roll(np.asarray(
            actual["object_part_orientations_wxyz"]), -1, axis=1)).as_matrix()
        center = positions[0] - socket_position
        points = edges @ rotations[0].T + center
        polar = np.rad2deg(np.arctan2(points[:, 1], points[:, 0])).reshape(-1, 2)
        errors = (polar - slot_centers[:, None] + 180.0) % 360.0 - 180.0
        margins = np.min(slot_widths[:, None] / 2.0 - np.abs(errors), axis=1)
        radial = 0.0190373 - np.max(np.linalg.norm(points[:, :2], axis=1))
        tilt = np.rad2deg(np.arccos(np.clip(-rotations[0, 2, 2], -1.0, 1.0)))
        contact = actual["contacts"]
        impulses = {"body_socket": 0.0, "nut_socket": 0.0, "robot_socket": 0.0,
                    "finger_body": 0.0, "finger_nut": 0.0}
        finger_impulses = {"body": [0.0, 0.0, 0.0], "nut": [0.0, 0.0, 0.0]}
        penetration = {"body_socket": 0.0, "nut_socket": 0.0}
        for header in contact["tensor_headers"]:
            paths = header["paths"]
            is_socket = any("FixedReceptaclePose" in path for path in paths)
            is_robot = any(path.startswith("/World/HandArm/") for path in paths)
            part = next((name for name, path in part_paths.items() if path in paths), None)
            key = ((part + "_socket") if part and is_socket else
                   "robot_socket" if is_robot and is_socket else
                   "finger_" + part if part and is_robot else None)
            if key is None:
                continue
            for hit in header["contacts"]:
                impulse = max(0.0, float(hit["normal_impulse_n_s"]))
                impulses[key] += impulse
                if key.startswith("finger_"):
                    for finger, link in enumerate(("f1Link3", "f2Link2", "f3Link3")):
                        if any(path.endswith("/" + link) for path in paths):
                            finger_impulses[part][finger] += impulse
                if key in penetration:
                    penetration[key] = max(penetration[key], -float(hit["separation_m"]))
        rows.append({"step": step, "phase": actual["phase"],
            "time_s": actual["simulation_time_s"], "body_depth_m": -float(center[2]),
            "axis_tilt_deg": float(tilt), "lateral_offset_m": float(np.linalg.norm(center[:2])),
            "minimum_key_front_depth_m": float(np.min(-points[:, 2])),
            "minimum_slot_angular_margin_deg": float(np.min(margins)),
            "minimum_key_radial_clearance_m": float(radial),
            "five_keys_within_slot_front_bounds": bool(np.min(-points[:, 2]) > 0
                                                      and np.min(margins) > 0 and radial > 0),
            "positive_impulses_n_s": impulses, "penetration_m": penetration,
            "finger_part_positive_impulses_n_s": finger_impulses,
            "unauthorized_robot_object": int(contact.get("robot_object_unauthorized", 0)),
            "table_impulse_n_s": float(contact.get("object_table_positive_normal_impulse_n_s", 0)),
            "physics_step_callback_count": int(contact["physics_step_callback_count"])})
        if phase_kind == "nut_rotation":
            hand_rotation = Rotation.from_quat(np.roll(np.asarray(actual["hand_base_orientation_wxyz"]), -1)).as_matrix()
            hand_position = np.asarray(actual["hand_base_position_m"])
            relative_rotation = hand_rotation.T @ rotations[1]
            relative_translation = hand_rotation.T @ (positions[1] - hand_position)
            if len(rows) == 1:
                initial_rotations = rotations.copy()
                initial_hand_rotation = hand_rotation.copy()
                initial_relative_rotation = relative_rotation.copy()
                initial_relative_translation = relative_translation.copy()
            deltas = [rotations[part] @ initial_rotations[part].T for part in range(2)]
            deltas.append(hand_rotation @ initial_hand_rotation.T)
            rows[-1].update(
                body_clock_delta_deg=float(np.rad2deg(np.arctan2(deltas[0][1, 0], deltas[0][0, 0]))),
                nut_clock_delta_deg=float(np.rad2deg(np.arctan2(deltas[1][1, 0], deltas[1][0, 0]))),
                hand_clock_delta_deg=float(np.rad2deg(np.arctan2(deltas[2][1, 0], deltas[2][0, 0]))),
                nut_depth_m=float(socket_position[2] - positions[1, 2]),
                nut_in_hand_translation_change_m=float(np.linalg.norm(relative_translation - initial_relative_translation)),
                nut_in_hand_rotation_change_deg=float(np.rad2deg(Rotation.from_matrix(
                    relative_rotation @ initial_relative_rotation.T).magnitude())))
    if not rows or final_truth is None or rows[0]["step"] != start or rows[-1]["step"] != end:
        raise ValueError("the saved trace does not cover the complete support experiment")
    dt = float(np.median(np.diff([row["time_s"] for row in rows])))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("the executed stage has no valid physics time interval")
    two_second_samples = round(2.0 / dt)
    hold_phase = {"support": "key_probe_body_support_hold", "nut_regrasp": "key_probe_nut_grip_hold",
                  "nut_rotation": "key_probe_nut_rotation_hold"}[phase_kind]
    held = [row for row in rows if row["phase"] == hold_phase]
    released = [row for row in rows if row["positive_impulses_n_s"]["finger_body"]
                + row["positive_impulses_n_s"]["finger_nut"] <= 1e-9]
    depth = np.asarray([row["body_depth_m"] for row in rows])
    summary = {"simulation_only": True, "truth_role": "POST_MOTION_EVALUATION_ONLY",
        "phase_kind": phase_kind,
        "source_trace": str(trace_path),
        "controller_completed": support["completed"], "controller_release_trigger": support.get("release_trigger"),
        "sample_count": len(rows), "physics_dt_s": dt,
        "before_unload": rows[0], "final": rows[-1],
        "socket_contact_before_unload": bool(rows[0]["positive_impulses_n_s"]["body_socket"]
                                               + rows[0]["positive_impulses_n_s"]["nut_socket"] > 1e-9),
        "maximum_depth_change_from_release_start_m": float(np.max(np.abs(depth - depth[0]))),
        "maximum_axis_tilt_deg": max(row["axis_tilt_deg"] for row in rows),
        "minimum_slot_angular_margin_deg": min(row["minimum_slot_angular_margin_deg"] for row in rows),
        "minimum_key_radial_clearance_m": min(row["minimum_key_radial_clearance_m"] for row in rows),
        "five_key_front_bounds_maintained_throughout_release": all(row["five_keys_within_slot_front_bounds"] for row in rows),
        "all_fingers_detached_throughout_final_hold": bool(held and all(row in released for row in held)),
        "final_hold_depth_range_m": float(np.ptp([row["body_depth_m"] for row in held])) if held else None,
        "final_hold_tilt_range_deg": float(np.ptp([row["axis_tilt_deg"] for row in held])) if held else None,
        "body_socket_maximum_penetration_m": max(row["penetration_m"]["body_socket"] for row in rows),
        "nut_socket_maximum_penetration_m": max(row["penetration_m"]["nut_socket"] for row in rows),
        "robot_socket_positive_contact_samples": sum(row["positive_impulses_n_s"]["robot_socket"] > 1e-9 for row in rows),
        "unauthorized_robot_object_samples": sum(row["unauthorized_robot_object"] > 0 for row in rows),
        "object_table_contact_samples": sum(row["table_impulse_n_s"] > 1e-9 for row in rows),
        "single_physics_step_per_sample": all(row["physics_step_callback_count"] == 1 for row in rows),
        "evidence_boundary": "Front-edge containment and contact readback; not full thread engagement, locking, electrical continuity or hardware validation."}
    if phase_kind == "nut_regrasp":
        summary["before_regrasp"] = summary.pop("before_unload")
        summary["socket_contact_before_regrasp"] = summary.pop("socket_contact_before_unload")
        summary["maximum_depth_change_from_regrasp_start_m"] = summary.pop("maximum_depth_change_from_release_start_m")
        summary["five_key_front_bounds_maintained_throughout_regrasp"] = summary.pop("five_key_front_bounds_maintained_throughout_release")
        summary.pop("all_fingers_detached_throughout_final_hold")
        summary["controller_failure_reason"] = support.get("failure_reason")
        summary["controller_failure_stage"] = support.get("failure_stage")
        final_window = held[-two_second_samples:]
        summary["final_two_second_grip_sample_count"] = len(final_window)
        summary["three_fingers_on_nut_throughout_final_window"] = bool(final_window and all(
            all(value > 1e-9 for value in row["finger_part_positive_impulses_n_s"]["nut"])
            for row in final_window))
        summary["finger_body_contacts_during_regrasp_samples"] = sum(
            row["positive_impulses_n_s"]["finger_body"] > 1e-9 for row in rows[1:])
        summary["nut_only_grip_observed"] = bool(support["completed"] and len(final_window) == two_second_samples
            and summary["three_fingers_on_nut_throughout_final_window"]
            and not summary["finger_body_contacts_during_regrasp_samples"]
            and not summary["robot_socket_positive_contact_samples"]
            and not summary["unauthorized_robot_object_samples"])
    if phase_kind == "nut_rotation":
        summary["before_rotation"] = summary.pop("before_unload")
        summary["socket_contact_before_rotation"] = summary.pop("socket_contact_before_unload")
        summary["maximum_depth_change_from_rotation_start_m"] = summary.pop("maximum_depth_change_from_release_start_m")
        summary["five_key_front_bounds_maintained_throughout_rotation"] = summary.pop("five_key_front_bounds_maintained_throughout_release")
        summary.pop("all_fingers_detached_throughout_final_hold")
        for key in ("body_clock_delta_deg", "nut_clock_delta_deg", "hand_clock_delta_deg"):
            values = np.rad2deg(np.unwrap(np.deg2rad([row[key] for row in rows])))
            for row, value in zip(rows, values):
                row[key] = float(value)
            summary["final_" + key] = float(values[-1])
        summary.update(
            controller_failure_reason=support.get("failure_reason"),
            actual_body_axial_progress_m=rows[-1]["body_depth_m"]-rows[0]["body_depth_m"],
            actual_nut_axial_progress_m=rows[-1]["nut_depth_m"]-rows[0]["nut_depth_m"],
            maximum_nut_in_hand_translation_change_m=max(row["nut_in_hand_translation_change_m"] for row in rows),
            maximum_nut_in_hand_rotation_change_deg=max(row["nut_in_hand_rotation_change_deg"] for row in rows),
            three_fingers_on_nut_throughout_rotation=all(all(v > 1e-9 for v in
                row["finger_part_positive_impulses_n_s"]["nut"]) for row in rows),
            finger_body_contact_samples=sum(row["positive_impulses_n_s"]["finger_body"] > 1e-9 for row in rows),
            full_coupling_or_locking_verified=False)
        turning = [row for row in rows if row["phase"] == "key_probe_nut_rotation_turn"]
        if turning and np.ptp([row["nut_clock_delta_deg"] for row in turning]) > 1.0:
            theta = np.deg2rad([row["nut_clock_delta_deg"] for row in turning])
            z = np.asarray([row["nut_depth_m"] for row in turning])
            slope, intercept = np.polyfit(theta, z, 1)
            summary["observed_advance_per_clockwise_revolution_m"] = float(-2*np.pi*slope)
            summary["advance_versus_angle_line_rms_m"] = float(np.sqrt(np.mean((z-(slope*theta+intercept))**2)))
            summary["representative_mesh_lead_m_for_postrun_comparison_only"] = .00762
        summary["evidence_boundary"] = "One robot-driven thread pilot; measured motion and contacts do not establish complete coupling, detent retention, seal compression, electrical continuity or hardware success."
    camera_paths = {"support": "released_plug_observation/camera_and_estimate.json",
                    "nut_regrasp": "nut_regrasp/preclose_palm/camera_and_estimate.json",
                    "nut_rotation": "nut_rotation/postgrip_palm/camera_and_estimate.json"}
    observation_relative = Path(camera_paths[phase_kind])
    if stage_name is not None:
        observation_relative = Path(stage_name).joinpath(*observation_relative.parts[1:])
    observation_path = directory / "socket_transport" / observation_relative
    if observation_path.exists():
        observation = json.loads(observation_path.read_text())
        result = {"camera": "FIXED_PALM_MOUNT_FROM_ENCODERS",
                  "position_and_axis_measured": observation["position_and_axis_measured"],
                  "observation_status": observation["status"], "used_by_online_controller": False}
        if observation["position_and_axis_measured"]:
            observed = np.asarray(observation["world_from_plug_five_dof"])
            comparison_truth = preclose_truth if preclose_truth is not None else final_truth
            truth_rotation = Rotation.from_quat(np.roll(np.asarray(
                comparison_truth["object_part_orientations_wxyz"][0]), -1)).as_matrix()
            result.update(center_error_m=float(np.linalg.norm(observed[:3, 3] - np.asarray(
                comparison_truth["object_part_positions_m"][0]))),
                axis_error_deg=float(np.rad2deg(np.arccos(np.clip(
                    observed[:3, 2] @ truth_rotation[:, 2], -1.0, 1.0)))),
                comparison_trace_step=int(comparison_truth["step"]),
                comparison_epoch=("BEFORE_ROTATION_AFTER_PAUSED_POST_GRIP_IMAGE" if phase_kind == "nut_rotation" else
                                  "BEFORE_FIRST_NUT_TARE_AFTER_PAUSED_PRE_CLOSE_IMAGE"
                                  if preclose_truth is not None else "FINAL_PAUSED_SUPPORT_STATE"),
                axial_yaw_measured=False)
        summary["palm_observation_posthoc_accuracy"] = result
    prefix = stage_name or phase_kind
    result_path = directory / ("support_posthoc_v3.json" if phase_kind == "support" and stage_name is None else prefix + "_posthoc_v1.json")
    rows_path = directory / ("support_samples_v3.json" if phase_kind == "support" and stage_name is None else prefix + "_samples_v1.json")
    if result_path.exists() or rows_path.exists():
        raise FileExistsError("refusing to overwrite an existing support evaluation")
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    rows_path.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--phase-kind", choices=("support", "nut_regrasp", "nut_rotation"), default="support")
    parser.add_argument("--stage-name", help="Alternate repeated-stage directory, such as nut_regrasp_after_index")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.directory.resolve(), args.phase_kind, args.stage_name), ensure_ascii=False, indent=2))
