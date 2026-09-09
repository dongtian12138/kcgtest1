#!/usr/bin/env python3
"""Postrun native-versus-FK link pose and unfiltered wrist signal summary."""
import argparse
import gzip
import json
from pathlib import Path

import ijson
import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
destination = args.run / "native_pose_and_raw_wrist_posthoc_v1.json"
if destination.exists():
    raise FileExistsError(destination)

def peak(record, key, value, step):
    if value is not None and (key not in record or float(value) > record[key]["value"]):
        record[key] = {"value": float(value), "step": int(step)}

wrist = {}
with gzip.open(args.run / "wrist_ft_samples.json.gz", "rb") as stream:
    for row in ijson.items(stream, "item", use_float=True):
        phase = wrist.setdefault(row["phase"], {"samples": 0, "gate_enabled_samples": 0,
                                               "force_gate_exceedances": 0, "torque_gate_exceedances": 0})
        phase["samples"] += 1
        step = row["step"]
        raw = np.asarray(row["hand2arm_raw_wrench"])
        peak(phase, "raw_sensor_force_norm_n", np.linalg.norm(raw[:3]), step)
        peak(phase, "raw_sensor_torque_norm_nm", np.linalg.norm(raw[3:]), step)
        peak(phase, "unfiltered_compensated_force_norm_n", row["resultant_force_n"], step)
        peak(phase, "unfiltered_compensated_torque_norm_nm", row["resultant_torque_nm"], step)
        peak(phase, "force_used_for_protection_norm_n", row["resultant_force_used_for_protection_n"], step)
        peak(phase, "maximum_absolute_hand_active_effort_nm", np.max(np.abs(row["active_efforts_nm"][-4:])), step)
        if row["ft_gate_enabled"]:
            phase["gate_enabled_samples"] += 1
            force, torque = row["resultant_force_used_for_protection_n"], row["resultant_torque_nm"]
            phase["force_gate_exceedances"] += int(force is not None and force > row["effective_force_limit_n"])
            phase["torque_gate_exceedances"] += int(row.get("torque_stop_enabled", True)
                and torque is not None and torque > row["effective_torque_limit_nm"])

poses = {}
with (args.run / "truth_samples.jsonl").open() as stream:
    for line in stream:
        row = json.loads(line)
        native = row.get("native_robot_link_pose_audit", {})
        if native.get("source") != "EXISTING_NATIVE_PHYSICS_TENSOR_RIGID_LINK_TRANSFORMS":
            raise ValueError("native pose provenance missing")
        phase = poses.setdefault(row["phase"], {"samples": 0, "links": {}})
        phase["samples"] += 1
        for name, fk_position in row["terminal_link_positions_m"].items():
            actual = native["poses"][name]
            link = phase["links"].setdefault(name, {})
            error = np.linalg.norm(np.asarray(actual["position_world_m"])-fk_position)
            qn = np.asarray(actual["orientation_world_wxyz"])
            qf = np.asarray(row["terminal_link_orientations_wxyz"][name])
            angle = (Rotation.from_quat(qn[[1,2,3,0]]).inv()*Rotation.from_quat(qf[[1,2,3,0]])).magnitude()
            peak(link, "native_vs_encoder_fk_position_error_m", error, row["step"])
            peak(link, "native_vs_encoder_fk_orientation_error_deg", np.degrees(angle), row["step"])

result = {"scope": "POSTRUN_ONLY_NO_CONTROL_INPUT", "run": str(args.run.resolve()),
          "wrist_raw_definition": "Native hand2arm wrench including hand gravity; not the contact residual",
          "compensated_definition": "Logged gravity/dynamic-compensated residual; torque is unfiltered",
          "pose_comparison": "Native tensor terminal link transforms against separately logged encoder/nominal-mimic FK",
          "wrist_by_phase": wrist, "native_vs_fk_by_phase": poses}
destination.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps({"output": str(destination), "wrist_phases": len(wrist), "pose_phases": len(poses)}))
