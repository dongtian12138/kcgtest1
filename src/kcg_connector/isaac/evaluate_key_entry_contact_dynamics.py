#!/usr/bin/env python3
"""Inspect a completed Body-held key-entry episode's contact impulse history."""

import argparse
import gzip
import json
from pathlib import Path

import ijson
from trace_metadata import read_metadata_field
import numpy as np
from scipy.spatial.transform import Rotation


def evaluate(directory):
    saved = json.loads((directory / "physical_key_entry_samples.json").read_text())
    first, last = int(saved[0]["step"])-1, int(saved[-1]["step"])-1
    dt = float(read_metadata_field(directory, "physics_dt_s"))
    contact_view = read_metadata_field(directory, "tensor_contact_view_audit")
    if not any("FixedReceptaclePose" in p for p in contact_view["contact_filter_paths"]):
        raise ValueError("complete Socket friction recording is required")
    steps, times, positions, rotations, omegas, forces, penetrations = ([] for _ in range(7))
    with (directory / "truth_samples.jsonl").open() as stream:
        for line in stream:
            s = json.loads(line)
            if s["step"] < first:
                continue
            if s["step"] > last:
                break
            w = np.zeros((2, 6))
            pen = np.zeros(2)
            origin = np.asarray(s["object_part_positions_m"])[0]
            for field, vector in (("tensor_headers", "impulse_n_s"),
                                  ("friction_headers", "tangential_impulse_n_s")):
                used = set()
                for h in s["contacts"][field]:
                    paths = h["paths"]
                    if not any("FixedReceptaclePose" in p for p in paths):
                        continue
                    part_actor = next(((i, j) for i, name in enumerate(("Body", "CouplingNut"))
                                       for j, p in enumerate(paths) if p.endswith("/"+name)), None)
                    if part_actor is None:
                        continue
                    pair = tuple(sorted(paths))
                    if pair in used:
                        continue
                    used.add(pair)
                    part, actor = part_actor
                    for c in h["contacts"]:
                        impulse = np.asarray(c[vector])*(1 if actor == 0 else -1)
                        w[part, :3] += impulse / dt
                        w[part, 3:] += np.cross(np.asarray(c["position_m"])-origin, impulse) / dt
                        if field == "tensor_headers":
                            pen[part] = max(pen[part], -float(c["separation_m"]))
            steps.append(s["step"])
            times.append(s["simulation_time_s"])
            positions.append(s["object_part_positions_m"])
            rotations.append(Rotation.from_quat(np.roll(np.asarray(
                s["object_part_orientations_wxyz"]), -1, axis=1)).as_matrix())
            omegas.append(s.get("object_part_angular_velocities_world_rad_s", np.full((2, 3), np.nan)))
            forces.append(w)
            penetrations.append(pen)
    if steps != list(range(first, last+1)):
        raise ValueError("the complete controller interval is not present in the truth trace")
    times, positions, rotations, omegas, forces, penetrations = map(
        np.asarray, (times, positions, rotations, omegas, forces, penetrations))
    if not np.allclose(np.diff(times), dt, atol=1e-9, rtol=0):
        raise ValueError("trace time intervals differ from the recorded physics dt")
    fd_omega = np.full_like(omegas, np.nan)
    for part in range(2):
        dr = rotations[1:, part] @ np.swapaxes(rotations[:-1, part], 1, 2)
        fd_omega[1:, part] = Rotation.from_matrix(dr).as_rotvec()/dt
    wrist = {}
    with gzip.open(directory / "wrist_ft_samples.json.gz", "rb") as stream:
        for s in ijson.items(stream, "item", use_float=True):
            if first <= s["step"] <= last:
                wrist[int(s["step"])] = (s["resultant_force_n"], s["effective_force_limit_n"])
    wrist = np.asarray([wrist[k] for k in steps])
    elapsed = times-times[0]
    last_second = times > times[-1]-1.0+dt/2
    stats = {"scope": "COMPLETED_BODY_HELD_KEY_ENTRY_CONTACT_DYNAMICS_ONLY",
             "source_run": str(directory), "window_steps": [first, last],
             "physics_dt_s": dt, "sample_count": len(steps),
             "online_control_used": False, "socket_friction_recorded": True,
             "final_wrist_resultant_n": float(wrist[-1, 0]),
             "final_wrist_limit_n": float(wrist[-1, 1]),
             "native_velocity_is_not_assumed_to_equal_pose_increment_over_dt": True,
             "no_joint_brake_torque_or_passivity_inferred_from_contact_mean": True,
             "parts": {}}
    for part, name in enumerate(("Body", "CouplingNut")):
        stats["parts"][name] = {
            "maximum_recorded_penetration_m": float(penetrations[:, part].max()),
            "last_second_mean_socket_force_n": forces[last_second, part, :3].mean(0).tolist(),
            "last_second_socket_axial_force_range_n": [float(forces[last_second, part, 2].min()),
                                                       float(forces[last_second, part, 2].max())],
            "last_second_pose_difference_omega_z_rms_rad_s": float(np.sqrt(np.nanmean(fd_omega[last_second, part, 2]**2))),
            "native_angular_velocity_available": bool(np.isfinite(omegas[:, part]).all()),
        }
        if np.isfinite(omegas[:, part]).all():
            stats["parts"][name]["last_second_native_omega_z_rms_rad_s"] = float(np.sqrt(np.mean(omegas[last_second, part, 2]**2)))
    stem = directory / "key_entry_contact_dynamics_v1"
    if stem.with_suffix(".json").exists() or stem.with_suffix(".npz").exists():
        raise FileExistsError("refusing to overwrite completed contact dynamics evidence")
    stem.with_suffix(".json").write_text(json.dumps(stats, indent=2)+"\n")
    np.savez_compressed(stem.with_suffix(".npz"), steps=steps, elapsed_s=elapsed,
                        positions_world_m=positions, socket_wrenches_world_about_body_n_nm=forces,
                        native_omega_world_rad_s=omegas, pose_difference_omega_world_rad_s=fd_omega,
                        recorded_penetration_m=penetrations, wrist_resultant_and_limit_n=wrist)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(evaluate(parser.parse_args().directory.resolve()), indent=2))
