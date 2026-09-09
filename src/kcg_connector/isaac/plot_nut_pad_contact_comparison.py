#!/usr/bin/env python3
"""Plot measured old/new contact locations against the original pad faces."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
import trimesh


def plot(previous, current, output):
    repository = Path(__file__).resolve().parents[3]
    pad_root = repository / "artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TERMINAL_PAD_EXACT_SOURCE_V2"
    old, new = np.load(previous), np.load(current)
    if any(output.with_suffix(suffix).exists() for suffix in (".png", ".svg", ".json")):
        raise FileExistsError("refusing to overwrite an existing figure")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    for index, (ax, link) in enumerate(zip(axes, ("f1Link3", "f2Link2", "f3Link3")), 1):
        mesh = trimesh.load(repository / f"src/iiwa_description/meshes/hand/{link}.STL", process=False)
        pad = np.load(pad_root / f"{link}_PAD_BODY_raw_source_local_m.npz")
        ax.add_collection(PolyCollection(np.asarray(mesh.triangles)[:, :, :2] * 1000,
                                        facecolors="#d8dde2", edgecolors="none"))
        ax.add_collection(PolyCollection(pad["points_local_m"][pad["faces"]][:, :, :2] * 1000,
                                        facecolors="#4e91a7", edgecolors="none"))
        for samples, color, marker, size in ((old, "#bc423c", "o", 23), (new, "#147447", "*", 85)):
            points = samples[link + "_contact_points_link_m"] * 1000
            ax.scatter(points[:, 0], points[:, 1], color=color, marker=marker, s=size, zorder=3)
        ax.autoscale_view()
        ax.set(aspect="equal", title=f"Finger {index}", xlabel="Finger-link x (mm)")
        ax.set_ylabel("Finger-link y (mm)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Measured nut contacts: centred reuse reaches all three finger pads", fontsize=14)
    fig.legend(handles=[Line2D([], [], color="#bc423c", marker="o", ls="", label="Previous nail-end contacts"),
                        Line2D([], [], color="#147447", marker="*", ls="", markersize=10, label="New measured pad contacts"),
                        Line2D([], [], color="#4e91a7", lw=5, label="Original pad surfaces")],
               loc="outside lower center", ncol=3, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "scope": "ORTHOGRAPHIC_SOURCE_MESH_AND_MEASURED_CONTACT_PROJECTION_NOT_A_CAMERA_IMAGE",
        "previous_contact_samples": str(previous), "new_contact_samples": str(current),
        "pad_source_manifest": str(pad_root / "TERMINAL_PAD_SOURCE_MANIFEST.json"),
        "samples": "last two seconds of each completed nut grasp",
        "limitation": "Nearest-source-face assignment of native convex-decomposition contacts; not exact PhysX source-face IDs.",
        "full_insertion_or_rotation_depicted": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.previous, args.current, args.output)
