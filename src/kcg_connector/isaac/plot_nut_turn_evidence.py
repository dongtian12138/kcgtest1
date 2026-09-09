#!/usr/bin/env python3
"""Plot a completed development turn from saved motion and controller records."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from trace_metadata import iter_control_samples


def plot(directory, destination):
    rows = json.loads((directory / "nut_rotation_samples_v1.json").read_text())
    summary = json.loads((directory / "nut_rotation_posthoc_v1.json").read_text())
    controller = json.loads((directory / "socket_transport/nut_rotation/nut_rotation_controller_result.json").read_text())
    extent_path = directory / "full_key_axial_extent_posthoc_v1.json"
    extent = json.loads(extent_path.read_text()) if extent_path.exists() else {}
    time = np.asarray([r["time_s"] for r in rows]); time -= time[0]
    angle = -np.asarray([r["nut_clock_delta_deg"] for r in rows])
    hand = -np.asarray([r["hand_clock_delta_deg"] for r in rows])
    body = -np.asarray([r["body_clock_delta_deg"] for r in rows])
    advance = np.asarray([r["nut_depth_m"] for r in rows]); advance = (advance-advance[0])*1e3
    sample_time = {r["step"]: t for r, t in zip(rows, time)}
    valid = [r for r in iter_control_samples(controller) if r["step"]-1 in sample_time]
    control_time = np.asarray([sample_time[r["step"]-1] for r in valid])
    wrench = np.asarray([r["contact_wrench_at_virtual_plug_origin"] for r in valid])
    raw = np.asarray([r["raw_contact_wrench_at_virtual_plug_origin"] for r in valid])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), layout="constrained")
    a = axes[0, 0]
    a.plot(time, angle, label="Nut", color="#007a87", lw=2.2)
    a.plot(time, hand, label="Hand", color="#ee9b00", ls="--", lw=1.3)
    a.plot(time, body, label="Body", color="#6b4c9a", lw=1.5)
    a.set(xlabel="Time from pilot start (s)", ylabel="Clockwise rotation (deg)", title="A  Actual rotation")
    a.legend(frameon=False)
    if extent.get("first_all_keys_behind_mouth"):
        first_full = extent["first_all_keys_behind_mouth"]
        a.axvline(first_full["time_s"]-rows[0]["time_s"], color="#445555", ls=":", lw=1.)
        a.text(.03, .52, "Dotted line: all key rear vertices\nfirst crossed the socket mouth",
               transform=a.transAxes, fontsize=8)
    a = axes[0, 1]
    a.plot(angle, advance, color="#007a87", lw=2., label="Measured nut advance")
    a.plot(angle, angle*7.62/360., color="#777777", ls="--", label="7.62 mm/rev reference")
    a.set(xlabel="Actual nut rotation (deg)", ylabel="Axial advancement (mm)", title="B  Advancement from contact and force control")
    a.legend(frameon=False)
    a = axes[1, 0]
    a.plot(control_time, np.linalg.norm(raw[:, :2], axis=1), color="#b7bec4", lw=.45, label="Raw lateral estimate")
    a.plot(control_time, np.linalg.norm(wrench[:, :2], axis=1), color="#b34e2e", lw=1.6, label="Filtered lateral estimate")
    a.plot(control_time, wrench[:, 2], color="#007a87", lw=1.3, label="Filtered axial estimate")
    a.axhline(controller["settings"]["stops"]["lateral_force_n"], color="#b34e2e", ls=":", label="Additional lateral stop")
    a.set(xlabel="Time from pilot start (s)", ylabel="Interface force estimate (N)", title="C  Recorded force estimate at the virtual origin")
    a.legend(frameon=False, fontsize=8)
    a = axes[1, 1]
    dt = float(np.median(np.diff(time)))
    for name, color, label in (("body_socket", "#6b4c9a", "Body–socket"), ("nut_socket", "#007a87", "Nut–socket")):
        normal = np.asarray([r["positive_impulses_n_s"][name] for r in rows])/dt
        a.plot(time, normal, color=color, lw=.75, label=label)
    a.set(xlabel="Time from pilot start (s)", ylabel="Sum of positive normal impulses / dt (N)", title="D  Recorded normal contact loading")
    a.legend(frameon=False)
    for a in axes.flat:
        a.grid(alpha=.18)
    headline = ("Completed bounded robot-driven nut turn" if controller["completed"] else
                "Partial robot-driven nut turn — stopped by sustained lateral loading")
    subtitle = ("All key axial extents entered; loaded key-side contact remains; full coupling unverified"
                if extent.get("first_all_keys_behind_mouth") else
                "Three fingers retained nut-only contact; full insertion and coupling remain unverified")
    fig.suptitle(headline+"\n"+subtitle, fontsize=13)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".json"):
        if destination.with_suffix(suffix).exists():
            raise FileExistsError(destination.with_suffix(suffix))
    fig.savefig(destination.with_suffix(".png"), dpi=220)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)
    destination.with_suffix(".json").write_text(json.dumps({
        "source_run": str(directory), "source_evaluation": str(directory / "nut_rotation_posthoc_v1.json"),
        "scope": "ONE_COMPLETED_DEVELOPMENT_EPISODE_NOT_FULL_ASSEMBLY_OR_CAMERA_IMAGE",
        "actual_nut_clock_delta_deg": summary["final_nut_clock_delta_deg"],
        "actual_nut_axial_progress_m": summary["actual_nut_axial_progress_m"],
        "controller_sample_alignment": "A controller record at step k observes the completed physics sample k-1.",
        "lead_line": "Postrun representative geometric reference only; not used by online axial commands.",
        "normal_contact_loading": "Sum of recorded positive normal impulses divided by dt, not net axial force or complete contact wrench."
    }, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.directory.resolve(), args.output.resolve())
