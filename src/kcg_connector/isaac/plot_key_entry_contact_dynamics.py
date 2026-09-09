#!/usr/bin/env python3
"""Plot the recorded final contact transient of one completed key-entry episode."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
r, output = args.directory.resolve(), args.output.resolve()
if any(output.with_suffix(ext).exists() for ext in (".png", ".svg", ".json")):
    raise FileExistsError("refusing to overwrite a saved scientific figure")
a = np.load(r / "key_entry_contact_dynamics_v1.npz")
c = json.loads((r / "physical_key_entry_samples.json").read_text())
result = json.loads((r / "physical_key_entry_result.json").read_text())
meta = json.loads((r / "key_entry_contact_dynamics_v1.json").read_text())
t = a["elapsed_s"]
mask = t > t[-1] - .30
w = a["socket_wrenches_world_about_body_n_nm"]
raw = np.asarray([x["raw_contact_force_and_moment"] for x in c])
filtered = np.asarray([x["contact_force_and_moment"] for x in c])
fig, ax = plt.subplots(4, 1, figsize=(9.3, 9.2), sharex=True)
fig.subplots_adjust(left=.12, right=.985, top=.905, bottom=.075, hspace=.13)
ax[0].plot(t[mask], raw[mask, 2], label="Raw wrist-based interface estimate", lw=1.2)
ax[0].plot(t[mask], filtered[mask, 2], label="50 ms filtered estimate", lw=1.7)
ax[0].axhline(.2, color="k", ls=":", label="0.20 N reference")
ax[0].set_ylabel("Axial estimate (N)")
ax[0].legend(loc="upper left", fontsize=8)
ax[1].plot(t[mask], w[mask, 1, 2], label="Socket on nut: axial")
ax[1].plot(t[mask], np.linalg.norm(w[mask, 1, :2], axis=1), label="Socket on nut: lateral")
ax[1].plot(t[mask], np.linalg.norm(w[mask, 0, :3], axis=1), label="Socket on body: resultant", ls="--")
ax[1].set_ylabel("Contact force (N)")
ax[1].legend(loc="upper left", fontsize=8)
ax[2].plot(t[mask], np.asarray([x["true_depth_m"] for x in c])[mask]*1000, color="C2")
ax[2].set_ylabel("Body insertion (mm)")
for field, label in (("true_body_yaw_deg", "Body key yaw"), ("true_nut_yaw_deg", "Nut yaw")):
    ax[3].plot(t[mask], np.asarray([x[field] for x in c])[mask], label=label)
ax[3].set_ylabel("Yaw error (deg)")
ax[3].set_xlabel("Time since key probe began (s)")
ax[3].legend(loc="best", fontsize=8)
for axis in ax:
    axis.grid(alpha=.22)
    axis.axvline(t[-1], color="C3", ls=":", lw=1)
caption = ("Stopped before nut regrasp or commanded turning" if not result["physical_key_entry_observed"]
           else "End of the Body-held contact stage; full key insertion is not implied")
fig.suptitle("Thread-entry transient during Body-held insertion\n"+caption, fontsize=13, y=.986)
hz = round(1/meta["physics_dt_s"])
fig.text(.5, .014, f"One completed {hz} Hz episode; normal and friction contact impulses. Post-run evaluation only.",
         ha="center", fontsize=8)
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output.with_suffix(".png"), dpi=170)
fig.savefig(output.with_suffix(".svg"))
plt.close(fig)
output.with_suffix(".json").write_text(json.dumps({
    "source_run": str(r), "source_data": str(r / "key_entry_contact_dynamics_v1.npz"),
    "source_script": str(Path(__file__).resolve()), "window_duration_s": .3,
    "same_episode": True, "full_key_insertion_or_nut_rotation_claimed": False,
    "not_camera_image": True}, indent=2)+"\n")
print(output.with_suffix(".png"))
