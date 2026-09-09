#!/usr/bin/env python3
"""Evaluate physical retention during a completed nut release/open-hand index."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def evaluate(directory):
    controller = json.loads((directory / "socket_transport/nut_reindex/nut_reindex_controller_result.json").read_text())
    first, last = int(controller["first_step"])-1, int(controller["last_step"])-1
    socket = np.asarray(json.loads((directory / "assembly_scene.json").read_text())["socket_initial_position_world_m"])
    rows, reference = [], None
    with (directory / "truth_samples.jsonl").open() as stream:
        for line in stream:
            sample = json.loads(line)
            if sample["step"] < first:
                continue
            if sample["step"] > last:
                break
            part_positions = np.asarray(sample["object_part_positions_m"])
            rotations = Rotation.from_quat(np.roll(np.asarray(sample["object_part_orientations_wxyz"]), -1, axis=1)).as_matrix()
            hand_rotation = Rotation.from_quat(np.roll(sample["hand_base_orientation_wxyz"], -1)).as_matrix()
            if reference is None:
                reference = [rotations[0], rotations[1], hand_rotation]
            angles = []
            for R, initial in zip([rotations[0], rotations[1], hand_rotation], reference):
                delta = R @ initial.T
                angles.append(float(np.rad2deg(np.arctan2(delta[1, 0], delta[0, 0]))))
            present, magnitudes = set(), {"body_socket": 0., "nut_socket": 0., "finger_nut": 0., "finger_body": 0.}
            seen = set()
            for header in sample["contacts"]["tensor_headers"]:
                paths = header["paths"]
                pair = tuple(sorted(paths))
                if pair in seen:
                    continue
                seen.add(pair)
                impulse = sum(float(np.linalg.norm(c["impulse_n_s"])) for c in header["contacts"])
                if impulse <= 1e-9:
                    continue
                body = any(p.endswith("/Body") for p in paths)
                nut = any(p.endswith("/CouplingNut") for p in paths)
                terminal = any(p.endswith(("/f1Link3", "/f2Link2", "/f3Link3")) for p in paths)
                socket_contact = any("FixedReceptaclePose" in p for p in paths)
                robot = any(p.startswith("/World/HandArm/") for p in paths)
                for condition, name in ((body and socket_contact, "body_socket"),
                                        (nut and socket_contact, "nut_socket"),
                                        (terminal and nut, "finger_nut"), (terminal and body, "finger_body")):
                    if condition:
                        magnitudes[name] += impulse
                        present.add(name)
                if robot and socket_contact:
                    present.add("robot_socket")
                if robot and any(p.endswith(("/Table", "/FixedFixture")) for p in paths):
                    present.add("robot_table_or_fixture")
                if robot and (body or nut) and not terminal:
                    present.add("robot_object_nonterminal")
            rows.append({"step": sample["step"], "phase": sample["phase"], "time_s": sample["simulation_time_s"],
                         "body_depth_m": float(socket[2]-part_positions[0, 2]),
                         "body_lateral_offset_m": float(np.linalg.norm(part_positions[0, :2]-socket[:2])),
                         "body_axis_tilt_deg": float(np.rad2deg(np.arccos(np.clip(-rotations[0, 2, 2], -1., 1.)))),
                         "body_nut_hand_clock_delta_deg": angles,
                         "contact_presence": sorted(present), "normal_impulse_magnitudes_n_s": magnitudes})
    if not rows or [r["step"] for r in rows] != list(range(first, last+1)):
        raise ValueError("incomplete reindex interval")
    angles = np.rad2deg(np.unwrap(np.deg2rad([r["body_nut_hand_clock_delta_deg"] for r in rows]), axis=0))
    for row, angle in zip(rows, angles):
        row["body_nut_hand_clock_delta_deg"] = angle.tolist()
    free = [r for r in rows if r["phase"].startswith("nut_index_free_")]
    result = {"scope": "POSTRUN_RETENTION_AND_CONTACTS_DURING_NUT_RELEASE_AND_OPEN_HAND_REINDEX",
              "controller_completed": controller["completed"], "controller_failure_reason": controller.get("failure_reason"),
              "window_steps": [first, last], "sample_count": len(rows), "free_hand_sample_count": len(free),
              "before": rows[0], "final": rows[-1],
              "maximum_depth_change_from_start_m": max(abs(r["body_depth_m"]-rows[0]["body_depth_m"]) for r in rows),
              "maximum_body_tilt_deg": max(r["body_axis_tilt_deg"] for r in rows),
              "free_hand_object_contact_samples": sum(bool(set(r["contact_presence"]) & {"finger_nut", "finger_body", "robot_object_nonterminal"}) for r in free),
              "free_hand_robot_environment_contact_samples": sum(bool(set(r["contact_presence"]) & {"robot_socket", "robot_table_or_fixture"}) for r in free),
              "free_hand_body_depth_range_m": float(np.ptp([r["body_depth_m"] for r in free])) if free else None,
              "free_hand_body_nut_hand_clock_change_deg": (np.asarray(free[-1]["body_nut_hand_clock_delta_deg"])-np.asarray(free[0]["body_nut_hand_clock_delta_deg"])).tolist() if free else None,
              "online_control_used": False,
              "full_key_source_extent_and_slot_contacts_require_separate_evaluation": True,
              "complete_coupling_or_electrical_continuity_claimed": False}
    output = directory / "nut_reindex_posthoc_v1.json"
    output.open("x").write(json.dumps(result, indent=2)+"\n")
    (directory / "nut_reindex_samples_v1.json").open("x").write(json.dumps(rows, separators=(",", ":"))+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(evaluate(parser.parse_args().directory.resolve()), indent=2))
