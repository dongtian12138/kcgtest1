#!/usr/bin/env python3
"""Scientific comparison of captured images, source sections and measured contacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    run = args.directory.resolve()
    prefix = args.output_prefix.resolve()
    for ext in (".png", ".svg", ".json"):
        if prefix.with_suffix(ext).exists():
            raise FileExistsError(prefix.with_suffix(ext))
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "font.size": 10, "svg.fonttype": "none"})
    frame = run / "socket_transport/nut_reindex/before_nut_release"
    rows = json.loads((run / "nut_rotation_samples_v1.json").read_text())
    pads = np.load(run / "nut_rotation_pad_surface_samples_v1.npz")
    audit = json.loads((run / "nut_rotation_pad_surface_posthoc_v1.json").read_text())
    model = repo / "artifacts/carts_grasp/CARTS_GRASP_V1/object_models/te_j35"
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), gridspec_kw={"height_ratios": [1., 1.12]})
    ax = axes[0, 0]
    rgb = plt.imread(frame / "wrist_rgbd/rgb.png")
    ax.imshow(rgb)
    ax.set_xlim(760, 1230); ax.set_ylim(680, 140)
    ax.set_title("a 旋拧后腕相机局部（实际帧裁切）", loc="left")
    ax.axis("off")
    ax = axes[0, 1]
    ax.imshow(plt.imread(frame / "rgbd/rgb.png"))
    ax.set_title("b 同时刻掌心相机（中央为本体后端）", loc="left")
    ax.axis("off")
    ax = axes[1, 0]
    colors = {"Body": "#417ba4", "Nut": "#d18a2f"}
    extrema = {}
    for part, name in (("Body", "plug_body_visual_mesh.npz"),
                       ("Nut", "coupling_nut_visual_mesh.npz")):
        with np.load(model / name) as data:
            v = data["vertices_m"] if "vertices_m" in data else data["vertices"]
            mesh = trimesh.Trimesh(v, data["faces"], process=False)
        radius = np.linalg.norm(v[:, :2], axis=1)
        edges = mesh.edges_unique
        edge_points = v[edges[np.max(radius[edges], axis=1) >= .013]]
        midpoint = edge_points.mean(axis=1)
        segments = np.concatenate((np.stack((edge_points[:, 0], midpoint), axis=1),
                                   np.stack((midpoint, edge_points[:, 1]), axis=1)))
        projected = np.stack((np.linalg.norm(segments[:, :, :2], axis=2),
                              segments[:, :, 2]), axis=2) * 1000
        ax.add_collection(LineCollection(projected, colors=colors[part],
                          linewidths=.35, alpha=.23, rasterized=True,
                          label="本体源网格" if part == "Body" else "螺母源网格"))
        extrema[part] = float(np.linalg.norm(v[:, :2], axis=1).max())
    markers = ["o", "s", "^"]
    finger_records = []
    for i, link in enumerate(("f1Link3", "f2Link2", "f3Link3")):
        p = pads[link + "_nut_points_m"]
        radial = np.linalg.norm(p[:, :2], axis=1)
        stride = max(1, len(p)//500)
        ax.scatter(radial[::stride]*1000, p[::stride, 2]*1000, s=18, marker=markers[i],
                   color=["#a13164", "#7354a4", "#153d53"][i], label=f"指{i+1}实测触点", zorder=4)
        finger_records.append({"link": link, "contact_count": len(p),
                               "radius_range_m": [float(radial.min()), float(radial.max())],
                               "axial_range_m": [float(p[:, 2].min()), float(p[:, 2].max())]})
    ax.axvline(extrema["Body"]*1000, color=colors["Body"], linestyle=":", linewidth=1)
    ax.set(xlim=(13, 25), ylim=(-33, 1), xlabel="半径 r (mm)", ylabel="零件轴向坐标 z (mm)")
    ax.set_title("c 原 CAD 径向投影与全旋拧阶段触点", loc="left")
    ax.legend(fontsize=8, loc="lower left", framealpha=.9)
    ax.grid(alpha=.16)
    ax = axes[1, 1]
    t = np.array([r["time_s"] for r in rows]); t -= t[0]
    for key, color, label, style in (("nut_clock_delta_deg", "#d18a2f", "螺母", "-"),
                                     ("hand_clock_delta_deg", "#454545", "手", "--")):
        ax.plot(t, [r[key] for r in rows], color=color, linestyle=style, label=label)
    other = ax.twinx()
    other.plot(t, [r["body_clock_delta_deg"] for r in rows], color="#278565", label="本体（右轴）", linewidth=1)
    other.set_ylabel("本体转角 (°)，独立右轴", color="#278565")
    other.tick_params(axis="y", colors="#278565")
    other.set_ylim(-.7, .7)
    ax.set(xlabel="从旋拧阶段起点计时 (s)", ylabel="螺母 / 手转角 (°)")
    ax.set_title("d 实际转角：螺母旋转，本体小幅游隙", loc="left")
    ax.grid(alpha=.16)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = other.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="lower left", fontsize=8)
    fig.suptitle("指腹抓螺母的接触证据｜同一开发回合，尚未完成装配", fontsize=15, y=.985)
    fig.text(.055, .028,
             "图 a/b 为仿真相机实录；图 c 为事后源几何及接触记录投影，三指全段仅接触 Nut。\n"
             "全键轴向进入发生在首段旋拧途中；有加载键侧相交，不把局部进展称为严格无干涉或完整耦合。", fontsize=9)
    fig.subplots_adjust(left=.06, right=.92, top=.92, bottom=.13, hspace=.29, wspace=.34)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=180)
    fig.savefig(prefix.with_suffix(".svg"))
    metadata = {"run": str(run), "capture_stage": "before_nut_release_after_first_turn",
                "source_images": [str(frame / "wrist_rgbd/rgb.png"), str(frame / "rgbd/rgb.png")],
                "native_images_cropped_for_layout": True, "generated_empirical_images": False,
                "section_scope": "Source mesh edges projected to radius with edge midpoint samples, including all nut knurl azimuths; contact azimuths collapsed in Nut coordinates",
                "contact_source": str(run / "nut_rotation_pad_surface_samples_v1.npz"),
                "pad_audit": str(run / "nut_rotation_pad_surface_posthoc_v1.json"),
                "fingers": finger_records, "source_maximum_radii_m": extrema,
                "pad_source_projection_is_not_native_exact_triangle_identity": True,
                "physical_sample_count": audit["physical_sample_count"],
                "full_assembly_claimed": False}
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
