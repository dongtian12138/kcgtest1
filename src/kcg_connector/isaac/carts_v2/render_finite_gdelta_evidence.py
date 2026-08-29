#!/usr/bin/env python3
"""Render compact, data-bound evidence for the frozen finite grasp study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from PIL import Image

import finite_cad_search as finite
from kcg_connector.grasp.carts_v2.models import V2Inputs, load_v2_inputs
from kcg_connector.grasp.carts_v2.task_grip_surface import (
    generate_axial_pad_intersection_grasp,
)


PAD_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
DISPLAY_TRIANGLE_LIMIT = 6000


def _arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(repository / "src/kcg_connector/config/carts_grasp_v2.yaml"),
    )
    parser.add_argument("--current-search", required=True)
    parser.add_argument("--te-search", required=True)
    parser.add_argument("--current-baseline-eval", required=True)
    parser.add_argument("--current-auto-run", required=True)
    parser.add_argument("--current-auto-eval")
    parser.add_argument("--current-robust-search", required=True)
    parser.add_argument("--current-robust-eval", required=True)
    parser.add_argument("--te-baseline-eval", required=True)
    parser.add_argument("--te-auto-run", required=True)
    parser.add_argument("--te-auto-eval")
    parser.add_argument("--te-robust-search", required=True)
    parser.add_argument("--te-robust-eval", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _single_baseline(search: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = [
        row
        for row in search["candidates"]
        if row.get("is_nominal_baseline_grid_point") is True
    ]
    if (
        len(rows) != 1
        or not isinstance(rows[0].get("path"), Mapping)
        or not isinstance(rows[0].get("quality"), Mapping)
    ):
        raise ValueError(
            "search must contain exactly one geometrically reconstructed baseline grid point"
        )
    return rows[0]


def _patch_arrays(patch: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(patch, Mapping):
        points = patch["points_object_m"]
        normals = patch["normals_object"]
        faces = patch.get("object_source_face_indices", ())
    else:
        points = patch.points_object_m
        normals = patch.normals_object
        faces = patch.object_face_indices
    point_array = np.asarray(points, dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if (
        point_array.ndim != 2
        or point_array.shape[1] != 3
        or normal_array.shape != point_array.shape
        or (len(face_array) not in (0, len(point_array)))
        or not np.all(np.isfinite(point_array))
        or not np.all(np.isfinite(normal_array))
    ):
        raise ValueError("contact patch arrays are malformed")
    return point_array, normal_array, face_array


def _reconstruct_baseline_patches(
    inputs: V2Inputs,
    baseline: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    base_generation = generate_axial_pad_intersection_grasp(
        inputs,
        palm_joint_position_rad=float(baseline["palm_rad"]),
        grasp_axis_position_m=float(baseline["cad_reference_section_z_m"]),
        hand_yaw_rad=0.0,
        apply_table_clearance=False,
    )
    generation = finite._place_base_generation(
        inputs, base_generation, float(baseline["yaw_rad"])
    )
    scene = finite.ExactCollisionScene(inputs)
    path, reason = finite._evaluate_control_path(
        inputs,
        scene,
        generation["control_plan"],
        inputs.config.section("finite_cad_search"),
    )
    if path is None:
        raise RuntimeError(f"baseline reconstruction disagrees with search: {reason}")
    path_record = baseline["path"]
    expected_counts = list(
        path_record.get(
            "collision_witness_counts",
            path_record.get("contact_patch_counts", ()),
        )
    )
    observed_counts = [len(patch.points_object_m) for patch in path.patches]
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"baseline patch counts changed: {observed_counts} != {expected_counts}"
        )
    return path.patches


def _selected_patches(search: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    patches = search["selected_candidate"]["full_pad_contact_patches"]
    if len(patches) != 3:
        raise ValueError("selected grasp must contain three full-pad patches")
    return patches


def _quality_values(search: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(
        [
            float(row["quality"]["task_load_factor"])
            for row in search["candidates"]
            if row.get("status") == "EXECUTABLE_OFFLINE"
        ],
        dtype=np.float64,
    )
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("search contains no finite executable quality values")
    return values


def _quality_statistics(search: Mapping[str, Any]) -> dict[str, Any]:
    values = _quality_values(search)
    return {
        "finite_set_size": int(search["grid"]["evaluated_candidate_count"]),
        "executable_count": int(len(values)),
        "minimum": float(np.min(values)),
        "median": float(np.quantile(values, 0.5, method="linear")),
        "p90": float(np.quantile(values, 0.9, method="linear")),
        "maximum": float(np.max(values)),
        "status_counts": dict(search["status_counts"]),
    }


def _unauthorized_total(evaluation: Mapping[str, Any]) -> int:
    return int(sum(int(value) for value in evaluation["unauthorized_contact_records"].values()))


def _comparison_row(
    object_label: str,
    method: str,
    search_record: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    acceleration_tolerance_m_s2: float,
) -> dict[str, Any]:
    quality = search_record["quality"]
    path = search_record["path"]
    return {
        "object": object_label,
        "method": method,
        "predicted_task_load_factor": float(quality["task_load_factor"]),
        "predicted_forbidden_clearance_mm": 1000.0
        * float(path["minimum_forbidden_clearance_m"]),
        "predicted_closure_reserve_rad": float(path["minimum_closure_reserve_rad"]),
        "predicted_joint_torque_margin_nm": float(
            quality["minimum_joint_torque_margin_at_unit_task_nm"]
        ),
        "predicted_pad_force_margin_n": float(
            quality["minimum_pad_normal_force_margin_at_unit_task_n"]
        ),
        "isaac_three_full_pads": bool(
            evaluation["three_terminal_link_contacts_observed"]
            and evaluation["pad_surface_identity_verified"]
        ),
        "isaac_off_table": bool(evaluation["table_contact_released_during_hold"]),
        "isaac_lift_mm": 1000.0 * float(evaluation["maximum_lift_m"]),
        "isaac_hold_s": float(evaluation["hold_duration_s"]),
        "isaac_slip_mm": 1000.0 * float(evaluation["maximum_relative_slip_m"]),
        "isaac_orientation_change_deg": math.degrees(
            float(evaluation["maximum_orientation_change_rad"])
        ),
        "isaac_maximum_hand_effort_nm": float(
            evaluation["maximum_absolute_hand_effort_nm"]
        ),
        "isaac_peak_acceleration_m_s2": float(
            evaluation["actual_lift_peak_acceleration_m_s2"]
        ),
        "isaac_acceleration_limit_m_s2": float(
            evaluation["registered_lift_peak_acceleration_m_s2"]
            + acceleration_tolerance_m_s2
        ),
        "isaac_unauthorized_contacts": _unauthorized_total(evaluation),
        "isaac_maximum_table_penetration_mm": 1000.0
        * float(evaluation["maximum_table_penetration_m"]),
        "isaac_post_settle_table_penetration_mm": 1000.0
        * float(evaluation["maximum_post_settle_table_penetration_m"]),
        "isaac_controller_completed": bool(evaluation["controller_completed"]),
        "isaac_nominal_dynamic_safety_pass": bool(
            evaluation.get("controller_nominal_physical_pass", False)
        ),
        "isaac_nominal_research_dynamic_pass": bool(
            evaluation.get("nominal_research_dynamic_pass", False)
        ),
        "isaac_formal_dynamic_pass": bool(
            evaluation.get(
                "formal_dynamic_pass",
                evaluation.get("controller_nominal_physical_pass", False),
            )
        ),
    }


def _display_triangles(inputs: V2Inputs) -> np.ndarray:
    triangles = np.asarray(
        inputs.object_contract.model.mesh.face_vertices_m, dtype=np.float64
    )
    if len(triangles) <= DISPLAY_TRIANGLE_LIMIT:
        return triangles
    indices = np.unique(
        np.linspace(0, len(triangles) - 1, DISPLAY_TRIANGLE_LIMIT, dtype=np.int64)
    )
    return triangles[indices]


def _draw_contact_geometry(
    axis: Any,
    inputs: V2Inputs,
    patches: Sequence[Any],
    friction: float,
    edge_count: int,
    title: str,
) -> None:
    triangles_mm = 1000.0 * _display_triangles(inputs)
    mesh_artist = Poly3DCollection(
        triangles_mm,
        facecolor="#bec5ce",
        edgecolor="#69717c",
        linewidth=0.04,
        alpha=0.18,
    )
    axis.add_collection3d(mesh_artist)
    all_vertices = np.asarray(inputs.object_contract.model.mesh.vertices_m) * 1000.0
    lower = np.min(all_vertices, axis=0)
    upper = np.max(all_vertices, axis=0)
    span = np.maximum(upper - lower, 1.0)
    padding = 0.06 * float(np.max(span))
    axis.set_xlim(lower[0] - padding, upper[0] + padding)
    axis.set_ylim(lower[1] - padding, upper[1] + padding)
    axis.set_zlim(lower[2] - padding, upper[2] + padding)
    axis.set_box_aspect(span)

    for pad_index, (patch, color) in enumerate(zip(patches, PAD_COLORS), start=1):
        points, normals, _ = _patch_arrays(patch)
        points_mm = 1000.0 * points
        axis.scatter(
            points_mm[:, 0],
            points_mm[:, 1],
            points_mm[:, 2],
            s=13,
            color=color,
            depthshade=False,
            label=f"pad {pad_index}",
        )
        normal_segments: list[np.ndarray] = []
        friction_segments: list[np.ndarray] = []
        for point_mm, normal in zip(points_mm, normals):
            normal = normal / np.linalg.norm(normal)
            normal_segments.append(np.stack((point_mm, point_mm + 2.8 * normal)))
            for force in finite._friction_basis(normal, friction, edge_count):
                force = force / np.linalg.norm(force)
                friction_segments.append(
                    np.stack((point_mm, point_mm + 2.2 * force))
                )
        axis.add_collection3d(
            Line3DCollection(normal_segments, colors="#111111", linewidths=0.45, alpha=0.42)
        )
        axis.add_collection3d(
            Line3DCollection(friction_segments, colors=color, linewidths=0.35, alpha=0.12)
        )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("object x [mm]")
    axis.set_ylabel("object y [mm]")
    axis.set_zlabel("object z [mm]")
    axis.view_init(elev=24, azim=-52)


def _render_contacts(
    cases: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(20, 12), constrained_layout=True)
    for row_index, case in enumerate(cases):
        friction = float(
            case["inputs"].object_contract.contact_material_uncertainty
            .friction_coefficient_interval[0]
        )
        edge_count = int(
            case["inputs"].config.section("finite_cad_search")[
                "friction_cone_edge_count"
            ]
        )
        baseline_axis = figure.add_subplot(2, 3, row_index * 3 + 1, projection="3d")
        _draw_contact_geometry(
            baseline_axis,
            case["inputs"],
            case["baseline_patches"],
            friction,
            edge_count,
            f"{case['label']} baseline (offline reconstruction)",
        )
        auto_axis = figure.add_subplot(2, 3, row_index * 3 + 2, projection="3d")
        _draw_contact_geometry(
            auto_axis,
            case["inputs"],
            case["auto_patches"],
            friction,
            edge_count,
            f"{case['label']} V4 nominal G_delta best",
        )
        robust_axis = figure.add_subplot(2, 3, row_index * 3 + 3, projection="3d")
        _draw_contact_geometry(
            robust_axis,
            case["inputs"],
            case["robust_patches"],
            friction,
            edge_count,
            f"{case['label']} V5 robust selector",
        )
    figure.suptitle(
        "CAD/FCL contact witnesses: black = outward normals; colored = 8-edge unilateral friction rays",
        fontsize=13,
    )
    figure.savefig(output / "cad_contacts_normals_friction.png", dpi=220)
    plt.close(figure)


def _quality_heatmap(search: Mapping[str, Any]) -> np.ndarray:
    palms = len(search["grid"]["palm_rad"])
    yaws = len(search["grid"]["yaw_rad"])
    values = np.full((palms, yaws), np.nan, dtype=np.float64)
    for row in search["candidates"]:
        quality = row.get("quality")
        if not isinstance(quality, Mapping):
            continue
        palm = int(row["grid_indices"]["palm"])
        yaw = int(row["grid_indices"]["yaw"])
        value = float(quality["task_load_factor"])
        values[palm, yaw] = (
            value if np.isnan(values[palm, yaw]) else max(values[palm, yaw], value)
        )
    return values


def _render_quality(cases: Sequence[Mapping[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row_index, case in enumerate(cases):
        search = case["search"]
        baseline = case["baseline"]
        selected = search["selected_candidate"]
        robust_selected = case["robust_search"]["selected_candidate"]
        values = _quality_values(search)
        histogram = axes[row_index, 0]
        histogram.hist(values, bins=45, color="#4c78a8", alpha=0.82)
        histogram.axvline(1.0, color="#111111", linestyle=":", label="task threshold")
        histogram.axvline(
            float(baseline["quality"]["task_load_factor"]),
            color="#777777",
            linestyle="--",
            label="baseline",
        )
        histogram.axvline(
            float(selected["quality"]["task_load_factor"]),
            color="#d62728",
            linestyle="-",
            label="V4 nominal best",
        )
        histogram.axvline(
            float(robust_selected["quality"]["task_load_factor"]),
            color="#2f6fbb",
            linestyle="-.",
            label="V5 robust selector",
        )
        histogram.set_title(f"{case['label']}: executable candidate quality")
        histogram.set_xlabel("task load factor lambda")
        histogram.set_ylabel("candidate count")
        histogram.legend(fontsize=8)

        heatmap = axes[row_index, 1]
        image = heatmap.imshow(
            _quality_heatmap(search),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )
        baseline_indices = baseline["grid_indices"]
        selected_indices = selected["grid_indices"]
        heatmap.scatter(
            baseline_indices["yaw"], baseline_indices["palm"],
            marker="o", s=55, facecolors="none", edgecolors="white", linewidths=1.5,
            label="baseline",
        )
        heatmap.scatter(
            selected_indices["yaw"], selected_indices["palm"],
            marker="*", s=100, color="#d62728", label="V4 nominal best",
        )
        robust_indices = robust_selected["grid_indices"]
        heatmap.scatter(
            robust_indices["yaw"], robust_indices["palm"],
            marker="D", s=50, color="#2f6fbb", label="V5 robust selector",
        )
        heatmap.set_title(f"{case['label']}: max lambda over axial grid")
        heatmap.set_xlabel("yaw index (15 deg steps)")
        heatmap.set_ylabel("palm-joint index")
        heatmap.legend(fontsize=8, loc="lower right")
        figure.colorbar(image, ax=heatmap, label="max task load factor")
    status_lines = [
        f"{case['label']}: "
        + ", ".join(
            f"{name}={count}" for name, count in case["search"]["status_counts"].items()
        )
        for case in cases
    ]
    figure.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
    figure.text(
        0.5,
        0.012,
        "\n".join(status_lines),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.savefig(
        output / "gdelta_search_and_quality_distribution.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(figure)


def _render_comparison(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    metrics = (
        ("predicted_task_load_factor", "predicted task load factor"),
        ("predicted_forbidden_clearance_mm", "forbidden clearance [mm]"),
        ("predicted_joint_torque_margin_nm", "joint torque margin [Nm]"),
        ("predicted_pad_force_margin_n", "pad normal-force margin [N]"),
        ("isaac_lift_mm", "Isaac lift [mm]"),
        ("isaac_slip_mm", "Isaac relative slip [mm]"),
        ("isaac_orientation_change_deg", "Isaac orientation change [deg]"),
        ("isaac_maximum_hand_effort_nm", "Isaac max hand effort [Nm]"),
        ("isaac_peak_acceleration_m_s2", "Isaac peak acceleration [m/s2]"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    labels = [f"{row['object']}\n{row['method']}" for row in rows]
    colors = ["#9a9a9a" if row["method"] == "baseline" else "#2f6fbb" for row in rows]
    for axis, (field, title) in zip(axes.flat, metrics):
        values = [float(row[field]) for row in rows]
        bars = axis.bar(np.arange(len(rows)), values, color=colors)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.22)
        axis.bar_label(bars, fmt="%.3g", padding=2, fontsize=7)
    figure.suptitle(
        "Historical baseline versus V4 nominal G_delta best (independent metrics)",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.003,
        "Historical baseline used an earlier controller, so this panel is descriptive rather than a causal comparison. "
        "All runs formed three full-pad contacts, left the table, and held for 2.000 s.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output / "baseline_vs_finite_best.png", dpi=220)
    plt.close(figure)


def _render_fair_comparison(
    rows: Sequence[Mapping[str, Any]], output: Path
) -> None:
    metrics = (
        ("predicted_task_load_factor", "predicted task load factor"),
        ("predicted_forbidden_clearance_mm", "forbidden clearance [mm]"),
        ("predicted_joint_torque_margin_nm", "joint torque margin [Nm]"),
        ("predicted_pad_force_margin_n", "pad normal-force margin [N]"),
        ("isaac_lift_mm", "Isaac lift [mm]"),
        ("isaac_slip_mm", "Isaac relative slip [mm]"),
        ("isaac_orientation_change_deg", "Isaac orientation change [deg]"),
        ("isaac_maximum_hand_effort_nm", "Isaac max hand effort [Nm]"),
        ("isaac_peak_acceleration_m_s2", "Isaac peak acceleration [m/s2]"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    labels = [f"{row['object']}\n{row['method']}" for row in rows]
    colors = ["#777777" if row["method"].startswith("V4") else "#2f6fbb" for row in rows]
    for axis, (field, title) in zip(axes.flat, metrics):
        values = [float(row[field]) for row in rows]
        bars = axis.bar(np.arange(len(rows)), values, color=colors)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.22)
        axis.bar_label(bars, fmt="%.3g", padding=2, fontsize=7)
        if field == "isaac_peak_acceleration_m_s2":
            axis.axhline(
                float(rows[0]["isaac_acceleration_limit_m_s2"]),
                color="#d62728",
                linestyle=":",
                label="frozen limit",
            )
            axis.legend(fontsize=8)
    figure.suptitle(
        "Fair dynamic comparison: V4 nominal optimum versus V5 robust selector",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.003,
        "Same controller, object properties, full-pad semantics, and safety criteria. "
        "V4 maximizes nominal task load in G_delta; V5 is a robust selector, not the nominal optimum.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output / "v4_nominal_vs_v5_robust_fair.png", dpi=220)
    plt.close(figure)


def _hold_image(auto_run: Path) -> tuple[Image.Image, Mapping[str, Any]]:
    evidence = _read_json(auto_run / "visual_evidence.json")
    records = [record for record in evidence["records"] if record.get("phase") == "hold"]
    if len(records) != 1:
        raise ValueError(f"{auto_run} must contain exactly one hold image")
    image_path = Path(records[0]["file"])
    return Image.open(image_path).convert("RGB"), records[0]


def _render_isaac_holds(cases: Sequence[Mapping[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 5.4), constrained_layout=True)
    for axis, case in zip(axes, cases):
        image, record = _hold_image(case["auto_run"])
        axis.imshow(image)
        axis.axis("off")
        evaluation = case["visual_evaluation"]
        axis.set_title(
            f"{case['label']} | lift={1000.0 * float(evaluation['maximum_lift_m']):.3f} mm, "
            f"hold={float(evaluation['hold_duration_s']):.3f} s\n"
            f"visual frame: table contacts={record['object_table_contact_count']}, "
            f"pad contacts={record['terminal_link_object_contact_counts']}",
            fontsize=10,
        )
    figure.suptitle(
        "True Isaac Sim final-hold frames for the V4 automatically selected grasps"
    )
    figure.savefig(output / "isaac_two_model_final_hold.png", dpi=220)
    plt.close(figure)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _contact_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for method, patches, source in (
            (
                "baseline",
                case["baseline_patches"],
                "DETERMINISTIC_OFFLINE_RECONSTRUCTION_OF_FROZEN_BASELINE_GRID_POINT",
            ),
            ("v4_nominal_best", case["auto_patches"], "STORED_V4_FINITE_SEARCH_RESULT"),
            ("v5_robust_selector", case["robust_patches"], "STORED_V5_ROBUST_SEARCH_RESULT"),
        ):
            for pad_index, patch in enumerate(patches, start=1):
                points, normals, faces = _patch_arrays(patch)
                for witness_index, (point, normal) in enumerate(zip(points, normals)):
                    rows.append(
                        {
                            "object": case["label"],
                            "method": method,
                            "pad": pad_index,
                            "witness": witness_index,
                            "point_object_x_m": float(point[0]),
                            "point_object_y_m": float(point[1]),
                            "point_object_z_m": float(point[2]),
                            "object_outward_normal_x": float(normal[0]),
                            "object_outward_normal_y": float(normal[1]),
                            "object_outward_normal_z": float(normal[2]),
                            "object_source_face_index": (
                                int(faces[witness_index]) if len(faces) else ""
                            ),
                            "source": source,
                            "isaac_measured_contact_geometry": False,
                        }
                    )
    return rows


def _case(
    repository: Path,
    config: Path,
    label: str,
    search_path: Path,
    baseline_evaluation_path: Path,
    auto_evaluation_path: Path,
    auto_run: Path,
    robust_search_path: Path,
    robust_evaluation_path: Path,
) -> dict[str, Any]:
    search = _read_json(search_path)
    robust_search = _read_json(robust_search_path)
    inputs = load_v2_inputs(
        repository, config_path=config, object_id=str(search["object_id"])
    )
    baseline = _single_baseline(search)
    auto_evaluation = _read_json(auto_evaluation_path)
    if auto_evaluation["object_id"] != search["object_id"]:
        raise ValueError("automatic dynamic evaluation object does not match search")
    robust_evaluation = _read_json(robust_evaluation_path)
    if (
        robust_search["object_id"] != search["object_id"]
        or robust_evaluation["object_id"] != search["object_id"]
    ):
        raise ValueError("robust evidence object does not match nominal search")
    return {
        "label": label,
        "search_path": search_path,
        "search": search,
        "inputs": inputs,
        "baseline": baseline,
        "baseline_patches": _reconstruct_baseline_patches(inputs, baseline),
        "baseline_evaluation": _read_json(baseline_evaluation_path),
        "auto_patches": _selected_patches(search),
        "auto_evaluation": auto_evaluation,
        "auto_run": auto_run,
        "visual_evaluation": _read_json(auto_run / "evaluation.json"),
        "robust_search_path": robust_search_path,
        "robust_search": robust_search,
        "robust_patches": _selected_patches(robust_search),
        "robust_evaluation": robust_evaluation,
        "acceleration_tolerance_m_s2": float(
            inputs.config.section("dynamic")[
                "lift_acceleration_tolerance_m_s2"
            ]
        ),
    }


def main() -> int:
    arguments = _arguments()
    repository = Path(__file__).resolve().parents[4]
    config = Path(arguments.config).expanduser().resolve()
    output = Path(arguments.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = (
        _case(
            repository,
            config,
            "Current D38999/26KJ61SN",
            Path(arguments.current_search).resolve(),
            Path(arguments.current_baseline_eval).resolve(),
            Path(
                arguments.current_auto_eval
                or Path(arguments.current_auto_run) / "evaluation.json"
            ).resolve(),
            Path(arguments.current_auto_run).resolve(),
            Path(arguments.current_robust_search).resolve(),
            Path(arguments.current_robust_eval).resolve(),
        ),
        _case(
            repository,
            config,
            "TE D38999/26FJ35PN",
            Path(arguments.te_search).resolve(),
            Path(arguments.te_baseline_eval).resolve(),
            Path(
                arguments.te_auto_eval
                or Path(arguments.te_auto_run) / "evaluation.json"
            ).resolve(),
            Path(arguments.te_auto_run).resolve(),
            Path(arguments.te_robust_search).resolve(),
            Path(arguments.te_robust_eval).resolve(),
        ),
    )
    comparison_rows: list[dict[str, Any]] = []
    fair_rows: list[dict[str, Any]] = []
    for case in cases:
        comparison_rows.extend(
            (
                _comparison_row(
                    case["label"],
                    "baseline",
                    case["baseline"],
                    case["baseline_evaluation"],
                    case["acceleration_tolerance_m_s2"],
                ),
                _comparison_row(
                    case["label"],
                    "finite_set_best",
                    case["search"]["selected_candidate"],
                    case["auto_evaluation"],
                    case["acceleration_tolerance_m_s2"],
                ),
            )
        )
        fair_rows.extend(
            (
                _comparison_row(
                    case["label"],
                    "V4 nominal G_delta best",
                    case["search"]["selected_candidate"],
                    case["auto_evaluation"],
                    case["acceleration_tolerance_m_s2"],
                ),
                _comparison_row(
                    case["label"],
                    "V5 robust selector",
                    case["robust_search"]["selected_candidate"],
                    case["robust_evaluation"],
                    case["acceleration_tolerance_m_s2"],
                ),
            )
        )

    _render_contacts(cases, output)
    _render_quality(cases, output)
    _render_comparison(comparison_rows, output)
    _render_fair_comparison(fair_rows, output)
    _render_isaac_holds(cases, output)
    _write_csv(output / "comparison.csv", comparison_rows)
    _write_csv(output / "v4_vs_v5_fair_comparison.csv", fair_rows)
    contact_rows = _contact_rows(cases)
    _write_csv(output / "contact_witnesses.csv", contact_rows)
    summary = {
        "schema_version": "finite_gdelta_two_model_evidence_v1",
        "claim": "V4_BEST_NOMINAL_TASK_LOAD_IN_FROZEN_AXIS_ALIGNED_FINITE_G_DELTA",
        "not_claimed": [
            "CONTINUOUS_SPACE_GLOBAL_OPTIMUM",
            "HARDWARE_VALIDATION",
            "FORMAL_ROBUSTNESS",
            "ISAAC_MEASURED_CONTACT_POSITIONS_OR_NORMALS",
        ],
        "contact_geometry_boundary": (
            "finite-set-best witnesses are stored CAD/FCL predictions; baseline witnesses "
            "are deterministic offline reconstructions of the one frozen baseline grid point"
        ),
        "dynamic_comparison_boundary": (
            "historical baseline rows use an earlier controller and are descriptive; "
            "V4-versus-V5 rows use the same frozen controller and safety criteria"
        ),
        "quality_statistics": {
            case["label"]: _quality_statistics(case["search"]) for case in cases
        },
        "comparison": comparison_rows,
        "v4_vs_v5_fair_comparison": fair_rows,
        "figures": [
            "cad_contacts_normals_friction.png",
            "gdelta_search_and_quality_distribution.png",
            "baseline_vs_finite_best.png",
            "v4_nominal_vs_v5_robust_fair.png",
            "isaac_two_model_final_hold.png",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
