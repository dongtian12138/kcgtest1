#!/usr/bin/env python3
"""Evaluate saved native Body--Nut joint readbacks, without controlling a run."""

import argparse
import json
from pathlib import Path

import ijson
from trace_metadata import read_metadata_field
import numpy as np
from scipy.spatial.transform import Rotation


def evaluate(directory):
    meta = read_metadata_field(directory, "object_internal_joint_reader")
    if meta["body_names"] != ["Body", "CouplingNut"] or meta["joint_indices"] != {"CouplingNutRevolute": 0}:
        raise ValueError("the incoming joint wrench row cannot be identified")
    rows = []
    source = directory / "truth_samples.jsonl"
    if not source.exists():
        source = directory / "trace.json"
    with source.open("rb") as stream:
        samples = ((json.loads(line) for line in stream) if source.suffix == ".jsonl"
                   else ijson.items(stream, "samples.item", use_float=True))
        for sample in samples:
            joint = sample["object_internal_joint_audit"]
            if joint is None:
                raise ValueError("native joint readbacks are absent")
            r = Rotation.from_quat(np.roll(np.asarray(sample["object_part_orientations_wxyz"]), -1, axis=1)).as_matrix()
            relative = r[0].T @ r[1]
            q_from_parts = float(np.arctan2(relative[1, 0], relative[0, 0]))
            q = float(np.asarray(joint["positions_rad"]).reshape(1)[0])
            dq = float(np.asarray(joint["velocities_rad_s"]).reshape(1)[0])
            wrench = np.asarray(joint["incoming_link_wrenches_joint_frame_n_nm"])
            if wrench.shape != (2, 6):
                raise ValueError("unexpected native incoming-link wrench shape")
            rows.append((int(sample["step"]), float(sample["simulation_time_s"]),
                         sample["phase"], q, dq, q_from_parts, *wrench[1]))
    values = np.asarray([[r[0], r[1], *r[3:]] for r in rows], dtype=np.float64)
    phases = np.asarray([r[2] for r in rows])
    discrepancy = (values[:, 2]-values[:, 4]+np.pi)%(2*np.pi)-np.pi
    result = {"scope": "POSTRUN_NATIVE_PASSIVE_JOINT_READBACK",
              "run": str(directory), "sample_count": len(rows), "metadata": meta,
              "torque_axis": "COAXIAL_JOINT_Z; INCOMING_FORCE_ON_CHILD_LINK",
              "maximum_joint_pose_crosscheck_error_rad": float(np.max(np.abs(discrepancy))),
              "online_control_used": False,
              "passivity_or_manufacturer_resistance_certified": False,
              "limits": "Incoming wrench is solver time-step average; velocity is an end-step state. Their product is not an exact substep energy audit.",
              "phases": {}}
    for phase in dict.fromkeys(phases):
        a = values[phases == phase]
        result["phases"][phase] = {
            "sample_count": len(a), "joint_angle_start_end_deg": np.rad2deg(a[[0, -1], 2]).tolist(),
            "joint_velocity_range_rad_s": [float(a[:, 3].min()), float(a[:, 3].max())],
            "incoming_joint_torsion_mean_nm": float(a[:, 10].mean()),
            "incoming_joint_torsion_range_nm": [float(a[:, 10].min()), float(a[:, 10].max())],
            "mean_incoming_joint_wrench_n_nm": a[:, 5:11].mean(0).tolist(),
        }
    stem = directory / "passive_joint_response_posthoc_v1"
    if stem.with_suffix(".json").exists() or stem.with_suffix(".npz").exists():
        raise FileExistsError("refusing to overwrite passive-joint evidence")
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(stem.with_suffix(".npz"), phases=phases, values=values,
                        columns=["step", "time_s", "q_rad", "dq_rad_s", "q_from_parts_rad",
                                 "Fx_N", "Fy_N", "Fz_N", "Tx_Nm", "Ty_Nm", "Tz_Nm"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = evaluate(parser.parse_args().directory.resolve())
    print(json.dumps({k: v for k, v in result.items() if k != "phases"}, indent=2))
