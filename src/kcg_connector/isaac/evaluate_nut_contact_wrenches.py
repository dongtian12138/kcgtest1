#!/usr/bin/env python3
"""Sum recorded normal and friction impulses after a completed nut turn.

This evaluates native contact data. It neither commands a joint nor infers a
manufacturer torque rating from the configured passive-joint drive cap.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from trace_metadata import read_metadata_field


def evaluate(directory, stage_name="nut_rotation"):
    if Path(stage_name).name != stage_name or stage_name in (".", ".."):
        raise ValueError("stage name must identify one local stage directory")
    controller = json.loads((directory / "socket_transport" / stage_name / "nut_rotation_controller_result.json").read_text())
    view = read_metadata_field(directory, "tensor_contact_view_audit")
    socket_friction_configured = bool(view.get("valid_after_reset") and any(
        "FixedReceptaclePose" in path for path in view["contact_filter_paths"]))
    first, last = int(controller["first_step"]), int(controller["last_step"])-1
    if last < first:
        raise ValueError("no executed rotation-stage sample")
    names = ("finger_on_nut", "socket_on_nut", "socket_on_body")
    steps, times, values = [], [], []
    duplicate_reports = 0
    friction_pairs = set()
    with (directory / "truth_samples.jsonl").open() as stream:
        for line in stream:
            s = json.loads(line)
            step = int(s["step"])
            if step < first:
                continue
            if step > last:
                break
            origin = np.asarray(s["object_part_positions_m"][1])
            wrenches = np.zeros((len(names), 6))
            for field, impulse_field in (("tensor_headers", "impulse_n_s"),
                                         ("friction_headers", "tangential_impulse_n_s")):
                used = {}
                for h in s["contacts"][field]:
                    paths = h["paths"]
                    nut = next((i for i, p in enumerate(paths) if p.endswith("/CouplingNut")), None)
                    body = next((i for i, p in enumerate(paths) if p.endswith("/Body")), None)
                    finger = any(p.endswith(("/f1Link3", "/f2Link2", "/f3Link3")) for p in paths)
                    socket = any("FixedReceptaclePose" in p for p in paths)
                    kind, actor = ((0, nut) if nut is not None and finger else
                                   (1, nut) if nut is not None and socket else
                                   (2, body) if body is not None and socket else (None, None))
                    if kind is None:
                        continue
                    pair = tuple(sorted(paths))
                    # Tensor impulse is the force on the first (sensor) actor.
                    sign = 1. if actor == 0 else -1.
                    contribution = np.zeros(6)
                    for c in h["contacts"]:
                        impulse = sign*np.asarray(c[impulse_field])
                        contribution[:3] += impulse
                        contribution[3:] += np.cross(np.asarray(c["position_m"])-origin, impulse)
                    if pair in used:
                        if not np.allclose(used[pair], contribution, rtol=1e-7, atol=1e-12):
                            raise ValueError("opposite-side contact reports disagree after sign transport")
                        duplicate_reports += 1
                        continue
                    used[pair] = contribution
                    wrenches[kind] += contribution
                    if field == "friction_headers":
                        friction_pairs.add(names[kind])
            steps.append(step)
            times.append(s["simulation_time_s"])
            values.append(wrenches)
    if steps != list(range(first, last+1)) or len(times) < 2:
        raise ValueError("incomplete executed-stage trace")
    times = np.asarray(times)
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, rtol=0, atol=1e-9):
        raise ValueError("nonuniform physics time")
    values = np.asarray(values)/dt
    windows = {"whole_stage": np.ones(len(times), bool),
               "last_two_seconds": times >= times[-1]-2.}
    result = {"scope": "POSTRUN_CONTACT_IMPULSE_WRENCH_SUM",
        "run": str(directory), "window_steps": [first, last], "dt_s": dt,
        "coordinates": "WORLD_AXES_ABOUT_EACH_SAMPLE_NUT_ORIGIN",
        "normal_and_recorded_friction_included": True, "online_control_used": False,
        "interfaces_with_recorded_friction": sorted(friction_pairs),
        "native_contact_filter_paths": view["contact_filter_paths"],
        "socket_friction_filter_validated_after_reset": socket_friction_configured,
        "friction_coverage_limitation": (
            "Both source parts and the fixed socket are included in the validated friction view; listed interfaces sum normal and friction records."
            if socket_friction_configured else
            "The existing friction view filters the plug parts; socket friction is absent. Only the finger-on-nut wrench includes both channels. Socket entries are normal contributions, not full wrenches."),
        "matching_opposite_side_normal_headers_counted_once": duplicate_reports,
        "not_a_direct_passive_drive_torque_measurement": True,
        "windows": {}}
    for label, mask in windows.items():
        result["windows"][label] = {
            name: {"mean_wrench_n_nm": values[mask, i].mean(axis=0).tolist(),
                   "torsion_range_nm": [float(values[mask, i, 5].min()), float(values[mask, i, 5].max())]}
            for i, name in enumerate(names)}
        torque_key = ("mean_nut_external_contact_torsion_nm" if socket_friction_configured else
                      "mean_recorded_nut_contact_torsion_missing_socket_friction_nm")
        result["windows"][label][torque_key] = float(
            (values[mask, 0, 5]+values[mask, 1, 5]).mean())
    prefix = "nut_contact_wrench" if stage_name == "nut_rotation" else stage_name + "_contact_wrench"
    summary = directory / (prefix+"_posthoc_v1.json")
    samples = directory / (prefix+"_samples_v1.npz")
    if summary.exists() or samples.exists():
        raise FileExistsError("refusing to overwrite completed wrench evidence")
    summary.write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(samples, steps=np.asarray(steps), time_s=times,
                        interface_names=names, wrenches_world_about_nut_n_nm=values)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--stage-name", default="nut_rotation")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.directory.resolve(), args.stage_name), indent=2))
