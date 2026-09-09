#!/usr/bin/env python3
"""Plot one completed key-entry/support/nut-grip episode from saved evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Wedge
from scipy.spatial.transform import Rotation


def plot(directory: Path, output: Path):
    entry = json.loads((directory / "physical_key_entry_result.json").read_text())
    entry_rows = json.loads((directory / "physical_key_entry_samples.json").read_text())
    support = json.loads((directory / "support_posthoc_v3.json").read_text())
    support_rows = json.loads((directory / "support_samples_v3.json").read_text())
    grip = json.loads((directory / "nut_regrasp_posthoc_v1.json").read_text())
    grip_rows = json.loads((directory / "nut_regrasp_samples_v1.json").read_text())
    scene = json.loads((directory / "assembly_scene.json").read_text())
    wanted_step = int(entry["final"]["step"]) - 1
    with (directory / "truth_samples.jsonl").open() as stream:
        for line in stream:
            actual = json.loads(line)
            if int(actual["step"]) == wanted_step:
                break
        else:
            raise ValueError("the entry endpoint is not in the recorded truth")
    rotation = Rotation.from_quat(np.roll(actual["object_part_orientations_wxyz"][0], -1)).as_matrix()
    center = np.asarray(actual["object_part_positions_m"][0]) - scene["socket_initial_position_world_m"]
    keys = np.asarray([10.0, 90.0, 157.0, 254.0, 308.0])
    widths = np.asarray([4.02158, 7.73810, 4.02158, 4.02158, 4.02158])
    slot_centers = (180.0 - keys) % 360.0
    slot_widths = np.asarray([4.817476, 9.643492, 4.817476, 4.817476, 4.817476])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    ax = axes[0, 0]
    for index, (angle, width, sc, sw) in enumerate(zip(keys, widths, slot_centers, slot_widths)):
        ax.add_patch(Wedge((0, 0), 19.0373, sc-sw/2, sc+sw/2, width=1.20,
            facecolor="#f8d8a8", edgecolor="#b76914", linewidth=1.2,
            label="Socket slot front bounds" if index == 0 else None))
        theta = np.deg2rad([angle-width/2, angle+width/2, angle+width/2, angle-width/2])
        radius = np.asarray([17.83715, 17.83715, 18.8214, 18.8214]) * .001
        local = np.column_stack((radius*np.cos(theta), radius*np.sin(theta), np.full(4, -.000762)))
        points = (local @ rotation.T + center) * 1000
        ax.add_patch(Polygon(points[:, :2], facecolor="#2b718c", edgecolor="#174456", linewidth=.8,
            label="Plug key at recorded pose" if index == 0 else None))
        label = "Main" if index == 1 else f"Key {index+1}"
        ax.text(22*np.cos(np.deg2rad(sc)), 22*np.sin(np.deg2rad(sc)), label, ha="center", va="center", fontsize=9)
    circle = plt.Circle((0, 0), 17.9705, fill=False, ls=":", lw=.8, ec="#8b9599")
    ax.add_patch(circle)
    ax.set(xlim=(-27, 27), ylim=(-25, 26), aspect="equal", xlabel="Socket x (mm)", ylabel="Socket y (mm)")
    ax.set_title("a  Five keys inside the slot front bounds", loc="left", fontweight="bold")
    ax.legend(loc="center", frameon=False, fontsize=9)
    ax.text(0, -8, "Post-run CAD projection\nEntry endpoint; hidden surfaces shown", ha="center", va="center", fontsize=9, color="#47545a")
    t0 = entry_rows[0]["step"] / 240
    entry_x = np.asarray([r["step"] / 240 - t0 for r in entry_rows])
    others = support_rows + grip_rows
    other_x = np.asarray([r["time_s"] - t0 for r in others])
    x = np.r_[entry_x, other_x]
    depth = np.r_[[r["true_depth_m"] for r in entry_rows], [r["body_depth_m"] for r in others]] * 1000
    keydepth = np.r_[[r["minimum_key_front_depth_m"] for r in entry_rows], [r["minimum_key_front_depth_m"] for r in others]] * 1000
    margin = np.r_[[r["minimum_slot_angular_margin_deg"] for r in entry_rows], [r["minimum_slot_angular_margin_deg"] for r in others]]
    release_t = support["before_unload"]["time_s"] - t0
    regrasp_t = grip["before_regrasp"]["time_s"] - t0
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axvspan(release_t, regrasp_t, color="#e5f0e6", zorder=0)
        ax.axvspan(regrasp_t, x[-1], color="#ecedf8", zorder=0)
        ax.set_xlabel("Time since low-force entry started (s)")
        ax.grid(axis="y", alpha=.18)
        ax.set_xlim(x[0], x[-1])
    ax = axes[0, 1]
    ax.plot(x, depth, color="#2b718c", lw=1.6, label="Body front")
    ax.plot(x, keydepth, color="#ae6423", lw=1.2, label="Shallowest key front")
    ax.axhline(0, color="#646d70", lw=.8, ls="--")
    ax.set_ylabel("Depth below socket mouth (mm)")
    ax.set_title("b  Entry retained through support and regrasp", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.text((release_t+regrasp_t)/2, -1.4, "Unload / open", fontsize=9,
            rotation=90, ha="center", va="center")
    ax.text((regrasp_t+x[-1])/2, -1.4, "Move / nut grip", fontsize=9, ha="center")
    ax = axes[1, 0]
    ax.plot(x, margin, color="#33485c", lw=1.2)
    ax.axhline(0, color="#b44d43", lw=1, ls="--")
    ax.set_ylabel("Minimum slot angular margin (deg)")
    ax.set_title("c  Loaded regrasp has a small negative margin", loc="left", fontweight="bold")
    ax.annotate(f"Minimum {min(margin):.4f}°\nNot strictly zero interference",
        xy=(x[int(np.argmin(margin))], min(margin)), xytext=(.37, .88), textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "color": "#b44d43"}, fontsize=9, color="#913d36", va="top")
    ax = axes[1, 1]
    gx = np.asarray([r["time_s"]-t0 for r in grip_rows])
    for finger, color in enumerate(("#236983", "#bc7625", "#6e5aa4")):
        force = np.asarray([r["finger_part_positive_impulses_n_s"]["nut"][finger] for r in grip_rows]) * 240
        ax.plot(gx, force, color=color, lw=1, label=f"Finger {finger+1}")
    ax.set_ylabel("Recorded nut normal impulse / dt (N)")
    ax.set_title("d  Three fingers transfer contact to the nut", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    fig.suptitle("Verified key entry and nut regrasp — one completed simulation episode", fontsize=14, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        p = output.with_suffix(suffix)
        if p.exists():
            raise FileExistsError(p)
        fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "source_episode": str(directory), "source_truth_step_for_front_projection": wanted_step,
        "uses_recorded_simulator_truth_after_episode_only": True,
        "projection_scope": "SOURCE_CAD_FRONT_KEY_AND_SLOT_BOUNDS_NOT_A_CAMERA_IMAGE",
        "images_or_sensor_data_synthesized": False,
        "full_coupling_or_locking_demonstrated": False,
        "caption": "One developmental episode with the original smooth-bore nut. Support and three-finger nut contact are observed, while a small negative key-edge margin remains during loaded regrasp. No success rate, zero-interference mating or locking claim follows from this figure."
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.directory.resolve(), args.output.resolve())
