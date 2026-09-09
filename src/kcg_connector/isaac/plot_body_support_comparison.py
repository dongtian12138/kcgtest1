#!/usr/bin/env python3
"""Plot two developmental support trials without implying a success-rate study."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


repository = Path(__file__).resolve().parents[3]
root = repository / "artifacts/kcg_connector/isaac/te_full_assembly_20260905"
output = root / "figures"
output.mkdir(exist_ok=True)
trials = [
    (repository / "artifacts/kcg_connector/isaac/te_body_assembly_20260905/support_source_nut_02",
     "Release after shallow entry", "#BD4C42"),
    (root / "contact_seat_filtered_02", "Release after force-confirmed support", "#187A8B"),
]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "axes.labelsize": 10})
fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), sharex=True)
records = []
for path, label, color in trials:
    rows = json.loads((path / "support_samples_v3.json").read_text())
    summary = json.loads((path / "support_posthoc_v3.json").read_text())
    t = np.asarray([r["time_s"] for r in rows]); t -= t[0]
    depth = np.asarray([r["body_depth_m"] for r in rows])
    tilt = [r["axis_tilt_deg"] for r in rows]
    margin = [r["minimum_slot_angular_margin_deg"] for r in rows]
    finger_load = [(r["positive_impulses_n_s"]["finger_body"]
                    + r["positive_impulses_n_s"]["finger_nut"]) * 240 for r in rows]
    for ax, values in zip(axes.flat, ((depth-depth[0])*1000, tilt, margin, finger_load)):
        ax.plot(t, values, color=color, lw=1.65, label=label)
    records.append({"run": str(path), "label": label,
                    "maximum_depth_change_m": summary["maximum_depth_change_from_release_start_m"],
                    "maximum_tilt_deg": summary["maximum_axis_tilt_deg"],
                    "minimum_key_slot_margin_deg": summary["minimum_slot_angular_margin_deg"]})
axes[0, 0].set_ylabel("Axial change (mm)")
axes[0, 1].set_ylabel("Axis tilt (deg)")
axes[1, 0].set_ylabel("Minimum key-slot margin (deg)")
axes[1, 1].set_ylabel("Summed finger normal load (N)")
axes[0, 0].text(.05, .80, "Maximum change: 3.392 mm", color=trials[0][2], transform=axes[0, 0].transAxes, fontsize=9)
axes[0, 0].text(.05, .69, "Maximum change: 0.029 mm", color=trials[1][2], transform=axes[0, 0].transAxes, fontsize=9)
axes[1, 0].axhline(0, color="0.35", ls="--", lw=.8)
for index, ax in enumerate(axes.flat):
    ax.grid(True, alpha=.18)
    ax.text(.02, .94, chr(97+index), transform=ax.transAxes, fontweight="bold", va="top")
    ax.axvline(2.0, color="0.65", ls=":", lw=.8)
for ax in axes[1]:
    ax.set_xlabel("Time from grip-unload start (s)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.01))
fig.tight_layout(rect=(0, 0.035, 1, .94))
fig.text(.5, .009, "Two developmental trials; simulator contact data evaluated after motion. No population success-rate claim.",
         ha="center", fontsize=8, color="0.35")
for extension in ("png", "svg"):
    target = output / ("support_release_development_comparison_v2." + extension)
    if target.exists():
        raise FileExistsError(target)
    fig.savefig(target, dpi=300, bbox_inches="tight")
(output / "support_release_development_comparison_v2_sources.json").write_text(
    json.dumps({"trials": records, "physics_dt_s": 1/240,
                "contact_force_calculation": "sum of positive contact normal impulses divided by physics_dt; not a net force or wrist sensor measurement",
                "confounding_limit": "Same grasp force and source geometry; visual alignment outcomes differ between these developmental runs."}, indent=2) + "\n")
print(output / "support_release_development_comparison_v2.png")
