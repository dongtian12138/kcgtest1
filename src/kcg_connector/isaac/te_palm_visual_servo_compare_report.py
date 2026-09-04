#!/usr/bin/env python3
"""Plot palm-servo estimator and control-policy comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "SAM-6D re-estimation": "#2878B5",
    "FoundationPose tracking": "#E07B39",
    "Five-DoF geometry": "#3A923A",
    "Frozen Five-DoF plan": "#D1495B",
    "Geometry + simultaneous": "#7A5195",
    "Geometry + camera orbit": "#2A9D8F",
}


def _matrix4(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(
        math.acos(np.clip(float(first @ second), -1.0, 1.0))
    )


def _raw_axial_yaw_deg(frame: dict[str, object]) -> float:
    raw = _matrix4(frame["estimated_world_from_object_raw_row_major"])
    yaw_free = _matrix4(frame["estimated_world_from_object_yaw_free_row_major"])
    axis = yaw_free[:3, 2]
    reference_x = yaw_free[:3, 0]
    raw_x = raw[:3, 0] - float(raw[:3, 0] @ axis) * axis
    norm = float(np.linalg.norm(raw_x))
    if norm < 1.0e-8:
        return math.nan
    raw_x /= norm
    return math.degrees(
        math.atan2(float(axis @ np.cross(reference_x, raw_x)), float(reference_x @ raw_x))
    )


def _load_run(path: Path, label: str) -> dict[str, object]:
    runtime = json.loads((path / "runtime_result.json").read_text(encoding="utf-8"))
    trace_path = path / "visual_servo_trace.json"
    if not trace_path.exists():
        trace_path = path / "frozen_target_plan_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    frames = trace["frames"]
    truth_target = _matrix4(trace["posthoc_truth_target_world_from_hand_row_major"])
    iterations = np.asarray([int(frame["iteration"]) for frame in frames])
    position_error = np.asarray(
        [float(frame["posthoc_pose_error"]["position_mm"]) for frame in frames]
    )
    axis_error = np.asarray(
        [float(frame["posthoc_pose_error"]["axis_deg"]) for frame in frames]
    )
    physical_position_remaining = []
    physical_axis_remaining = []
    for frame in frames:
        actual_hand = _matrix4(frame["actual_world_from_hand_row_major"])
        physical_position_remaining.append(
            1000.0 * float(np.linalg.norm(actual_hand[:3, 3] - truth_target[:3, 3]))
        )
        physical_axis_remaining.append(
            _angle_deg(actual_hand[:3, 2], truth_target[:3, 2])
        )
    raw_yaw = (
        np.full(len(frames), np.nan, dtype=np.float64)
        if label.startswith("Five-DoF")
        or label.startswith("Frozen Five-DoF")
        or label.startswith("Geometry +")
        else np.asarray([_raw_axial_yaw_deg(frame) for frame in frames])
    )
    finite = np.isfinite(raw_yaw)
    if np.any(finite):
        raw_yaw[finite] = np.degrees(np.unwrap(np.radians(raw_yaw[finite])))
    inference = np.asarray(
        [float(frame["perception_timing_s"]["total"]) for frame in frames]
    )
    plug_clearance = np.asarray(
        [
            math.nan
            if frame.get("collision_check") is None
            else 1000.0
            * float(frame["collision_check"]["minimum_clearance_m"]["plug"])
            for frame in frames
        ]
    )
    return {
        "path": path,
        "label": label,
        "runtime": runtime,
        "trace": trace,
        "frames": frames,
        "iterations": iterations,
        "position_error_mm": position_error,
        "axis_error_deg": axis_error,
        "physical_position_remaining_mm": np.asarray(physical_position_remaining),
        "physical_axis_remaining_deg": np.asarray(physical_axis_remaining),
        "raw_axial_yaw_deg": raw_yaw,
        "inference_s": inference,
        "plug_clearance_mm": plug_clearance,
    }


def _style_axis(axis: plt.Axes, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("Control observation index")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def _save_metrics_figure(runs: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    panels = (
        ("position_error_mm", "(a) Estimated position error", "Error [mm]"),
        ("axis_error_deg", "(b) Estimated approach-axis error", "Error [deg]"),
        ("raw_axial_yaw_deg", "(c) Raw axial yaw (not controlled)", "Unwrapped yaw [deg]"),
        (
            "physical_position_remaining_mm",
            "(d) Physical distance remaining to pregrasp",
            "Distance [mm]",
        ),
        (
            "physical_axis_remaining_deg",
            "(e) Physical axis error remaining",
            "Error [deg]",
        ),
        ("inference_s", "(f) Perception time per update", "Time [s]"),
    )
    for axis, (key, title, ylabel) in zip(axes.ravel(), panels):
        for run in runs:
            axis.plot(
                run["iterations"],
                run[key],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=str(run["label"]),
                color=COLORS[str(run["label"])],
            )
        _style_axis(axis, title, ylabel)
        if key == "inference_s":
            axis.set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Palm-camera visual servo: estimator accuracy and closed-loop outcome",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output / "figure_estimator_and_servo_comparison.png", dpi=300)
    figure.savefig(output / "figure_estimator_and_servo_comparison.pdf")
    plt.close(figure)


def _project(camera_point: np.ndarray, camera_matrix: np.ndarray) -> tuple[int, int] | None:
    if camera_point[2] <= 1.0e-6:
        return None
    pixel = camera_matrix @ camera_point
    return int(round(pixel[0] / pixel[2])), int(round(pixel[1] / pixel[2]))


def _annotated_rgb(
    frame: dict[str, object],
    camera_matrix: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    frame_dir = Path(str(frame["frame_dir"]))
    image = cv2.imread(str(frame_dir / "rgb.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame_dir / "rgb.png")
    pose = _matrix4(frame["camera_from_object_raw_row_major"])
    center = _project(pose[:3, 3], camera_matrix)
    endpoint = _project(pose[:3, 3] + 0.04 * pose[:3, 2], camera_matrix)
    if center is not None:
        cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 22, 3)
    if center is not None and endpoint is not None:
        cv2.arrowedLine(image, center, endpoint, (0, 220, 0), 4, tipLength=0.18)
    error = frame["posthoc_pose_error"]
    lines = (
        label,
        f"iteration {frame['iteration']}",
        f"position error {float(error['position_mm']):.2f} mm",
        f"axis error {float(error['axis_deg']):.2f} deg",
    )
    y = 34
    for line in lines:
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _save_observation_figure(runs: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(
        len(runs),
        3,
        figsize=(15.5, 4.0 * len(runs)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, run in enumerate(runs):
        frames = run["frames"]
        indices = sorted({0, len(frames) // 2, len(frames) - 1})
        while len(indices) < 3:
            indices.append(indices[-1])
        camera_matrix = np.asarray(
            run["runtime"]["palm_camera_intrinsics_3x3"], dtype=np.float64
        )
        for column, frame_index in enumerate(indices[:3]):
            axes[row, column].imshow(
                _annotated_rgb(
                    frames[frame_index], camera_matrix, label=str(run["label"])
                )
            )
            axes[row, column].set_axis_off()
            axes[row, column].set_title(("Initial", "Middle", "Terminal")[column])
    figure.suptitle(
        "Observed palm-camera sequence (red: estimated center, green: estimated axis)",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output / "figure_observation_sequence_comparison.png", dpi=300)
    figure.savefig(output / "figure_observation_sequence_comparison.pdf")
    plt.close(figure)


def _save_control_policy_figure(
    runs: list[dict[str, object]], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    panels = (
        (
            "physical_position_remaining_mm",
            "(a) Physical distance remaining to pregrasp",
            "Distance [mm]",
        ),
        (
            "physical_axis_remaining_deg",
            "(b) Physical axis error remaining",
            "Error [deg]",
        ),
        (
            "plug_clearance_mm",
            "(c) Minimum predicted hand-to-plug clearance",
            "Clearance [mm]",
        ),
        (
            "axis_error_deg",
            "(d) Five-DoF estimator axis error",
            "Error [deg]",
        ),
    )
    for axis, (key, title, ylabel) in zip(axes.ravel(), panels):
        for run in runs:
            axis.plot(
                run["iterations"],
                run[key],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=str(run["label"]),
                color=COLORS[str(run["label"])],
            )
        _style_axis(axis, title, ylabel)
        if key == "plug_clearance_mm":
            axis.axhline(0.0, color="black", linewidth=1.0)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Same Five-DoF estimator, different visual-servo paths",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output / "figure_geometry_control_policy_comparison.png", dpi=300)
    figure.savefig(output / "figure_geometry_control_policy_comparison.pdf")
    plt.close(figure)


def _last_rgb(run_path: Path) -> np.ndarray:
    candidates = sorted((run_path / "visual_servo_frames").glob("frame_*/rgb.png"))
    if not candidates:
        candidates = sorted((run_path / "palm_handoff").glob("frame_*/rgb.png"))
    if not candidates:
        raise FileNotFoundError(f"no palm RGB frame under {run_path}")
    image = cv2.imread(str(candidates[-1]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(candidates[-1])
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _save_control_failure_figure(
    simultaneous: dict[str, object],
    axis_then_path: Path,
    orbit: dict[str, object],
    output: Path,
) -> None:
    axis_runtime = json.loads(
        (axis_then_path / "runtime_result.json").read_text(encoding="utf-8")
    )
    rows = (
        (
            _last_rgb(Path(simultaneous["path"])),
            "(a) Simultaneous rotation + approach",
            "Predicted finger sweep collision; 84.6 mm remained",
        ),
        (
            _last_rgb(axis_then_path),
            "(b) Rotate hand at fixed hand position",
            f"Connector left the image; {axis_runtime.get('error', 'perception rejected')}",
        ),
        (
            _last_rgb(Path(orbit["path"])),
            "(c) Camera orbit + approach",
            "View retained; f2Link2 collision predicted with 49.4 mm remaining",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.1), constrained_layout=True)
    for axis, (image, title, detail) in zip(axes, rows):
        axis.imshow(image)
        axis.set_axis_off()
        axis.set_title(title, fontweight="bold")
        axis.text(
            0.5,
            -0.05,
            detail,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            wrap=True,
        )
    figure.suptitle(
        "Observed failure modes of three no-contact control paths",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output / "figure_geometry_control_failure_modes.png", dpi=300)
    figure.savefig(output / "figure_geometry_control_failure_modes.pdf")
    plt.close(figure)


def _save_geometry_fusion_replay(
    run: dict[str, object], output: Path
) -> dict[str, object]:
    trace = run["trace"]
    truth = _matrix4(trace["posthoc_truth_world_from_object_row_major"])
    truth_axis = truth[:3, 2] / np.linalg.norm(truth[:3, 2])
    truth_position = truth[:3, 3]
    raw_positions: list[np.ndarray] = []
    raw_axes: list[np.ndarray] = []
    raw_position_error = []
    raw_axis_error = []
    cumulative_position_error = []
    cumulative_axis_error = []
    sign_reference = None
    for frame in run["frames"]:
        geometry = frame.get("geometry_estimation")
        if not isinstance(geometry, dict):
            raise ValueError("geometry replay requires stored unfused poses")
        raw = _matrix4(geometry["unfused_world_from_object_row_major"])
        axis = raw[:3, 2] / np.linalg.norm(raw[:3, 2])
        if sign_reference is None:
            sign_reference = axis.copy()
        if float(axis @ sign_reference) < 0.0:
            axis = -axis
        raw_positions.append(raw[:3, 3].copy())
        raw_axes.append(axis)
        raw_position_error.append(
            1000.0 * float(np.linalg.norm(raw[:3, 3] - truth_position))
        )
        raw_axis_error.append(_angle_deg(axis, truth_axis))
        fused_position = np.median(np.stack(raw_positions), axis=0)
        fused_axis = np.median(np.stack(raw_axes), axis=0)
        fused_axis /= np.linalg.norm(fused_axis)
        cumulative_position_error.append(
            1000.0 * float(np.linalg.norm(fused_position - truth_position))
        )
        cumulative_axis_error.append(_angle_deg(fused_axis, truth_axis))

    iterations = np.asarray(run["iterations"])
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    series = (
        (
            np.asarray(raw_position_error),
            np.asarray(run["position_error_mm"]),
            np.asarray(cumulative_position_error),
            "(a) Position error",
            "Error [mm]",
        ),
        (
            np.asarray(raw_axis_error),
            np.asarray(run["axis_error_deg"]),
            np.asarray(cumulative_axis_error),
            "(b) Directed-axis error",
            "Error [deg]",
        ),
    )
    for axis, (raw_values, recorded_values, cumulative_values, title, ylabel) in zip(
        axes, series
    ):
        axis.plot(
            iterations,
            raw_values,
            color="#B5B5B5",
            marker=".",
            linewidth=1.2,
            label="Single frame",
        )
        axis.plot(
            iterations,
            recorded_values,
            color="#E07B39",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="Executed 3-frame mean",
        )
        axis.plot(
            iterations,
            cumulative_values,
            color="#2A9D8F",
            marker="o",
            markersize=3,
            linewidth=1.8,
            label="Offline cumulative median replay",
        )
        _style_axis(axis, title, ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Five-DoF geometry: effect of static-object temporal fusion",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output / "figure_geometry_fusion_replay.png", dpi=300)
    figure.savefig(output / "figure_geometry_fusion_replay.pdf")
    plt.close(figure)
    return {
        "evidence_type": "OFFLINE_REPLAY_OF_RECORDED_RAW_ESTIMATES",
        "dynamically_executed": False,
        "truth_used_for_fusion": False,
        "position_error_mm_median": float(np.median(cumulative_position_error)),
        "position_error_mm_maximum": float(np.max(cumulative_position_error)),
        "axis_error_deg_median": float(np.median(cumulative_axis_error)),
        "axis_error_deg_maximum": float(np.max(cumulative_axis_error)),
    }


def _save_final_outcome_figure(
    runs: list[dict[str, object]], output: Path
) -> None:
    labels = [str(run["label"]) for run in runs]
    position = np.asarray(
        [
            float(
                run["trace"].get(
                    "posthoc_final_hand_target_execution_error", {}
                ).get(
                    "position_mm",
                    run["trace"]["posthoc_final_position_error_mm"],
                )
            )
            for run in runs
        ]
    )
    axis = np.asarray(
        [
            float(
                run["trace"].get(
                    "posthoc_final_hand_target_execution_error", {}
                ).get(
                    "axis_deg",
                    run["trace"]["posthoc_final_axis_error_deg"],
                )
            )
            for run in runs
        ]
    )
    colors = [COLORS[label] for label in labels]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for plot, values, title, ylabel, threshold in (
        (axes[0], position, "(a) Final hand position error", "Error [mm]", 1.0),
        (axes[1], axis, "(b) Final hand directed-axis error", "Error [deg]", 1.0),
    ):
        bars = plot.bar(np.arange(len(labels)), values, color=colors)
        plot.set_yscale("log")
        plot.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="1 mm / 1 deg reference",
        )
        plot.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
        plot.set_ylabel(ylabel)
        plot.set_title(title, loc="left", fontweight="bold")
        plot.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            plot.text(
                bar.get_x() + bar.get_width() / 2.0,
                value * 1.12,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Continuous visual servo baselines versus one frozen-target plan",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output / "figure_frozen_plan_vs_visual_servo_outcome.png", dpi=300)
    figure.savefig(output / "figure_frozen_plan_vs_visual_servo_outcome.pdf")
    plt.close(figure)


def _summary(run: dict[str, object]) -> dict[str, object]:
    trace = run["trace"]
    hand_execution_error = trace.get(
        "posthoc_final_hand_target_execution_error", {}
    )
    inference = np.asarray(run["inference_s"], dtype=np.float64)
    tail_inference = inference[1:] if len(inference) > 1 else inference
    yaw = np.asarray(run["raw_axial_yaw_deg"], dtype=np.float64)
    yaw_span = (
        None
        if not np.any(np.isfinite(yaw))
        else float(np.nanmax(yaw) - np.nanmin(yaw))
    )
    result = {
        "termination": trace["termination"],
        "controller": trace["controller"],
        "online_target_reached": trace["online_target_reached"],
        "motion_count": trace["motion_count"],
        "capture_count": trace["capture_count"],
        "elapsed_wall_s": trace["elapsed_wall_s"],
        "position_error_mm_median": float(np.nanmedian(run["position_error_mm"])),
        "position_error_mm_maximum": float(np.nanmax(run["position_error_mm"])),
        "axis_error_deg_median": float(np.nanmedian(run["axis_error_deg"])),
        "axis_error_deg_maximum": float(np.nanmax(run["axis_error_deg"])),
        "post_initialization_update_time_s_median": float(np.nanmedian(tail_inference)),
        "raw_axial_yaw_span_deg": yaw_span,
        "posthoc_final_hand_position_error_mm": hand_execution_error.get(
            "position_mm", trace["posthoc_final_position_error_mm"]
        ),
        "posthoc_final_hand_axis_error_deg": hand_execution_error.get(
            "axis_deg", trace["posthoc_final_axis_error_deg"]
        ),
    }
    geometry_times = [
        float(frame["perception_timing_s"].get("geometry", math.nan))
        for frame in run["frames"]
    ]
    if np.any(np.isfinite(geometry_times)):
        result["geometry_only_time_s_median"] = float(
            np.nanmedian(geometry_times)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-run", type=Path, required=True)
    parser.add_argument("--foundation-run", type=Path, required=True)
    parser.add_argument("--geometry-run", type=Path)
    parser.add_argument("--frozen-run", type=Path)
    parser.add_argument("--geometry-simultaneous-run", type=Path)
    parser.add_argument("--axis-then-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = [
        _load_run(args.sam_run.resolve(), "SAM-6D re-estimation"),
        _load_run(args.foundation_run.resolve(), "FoundationPose tracking"),
    ]
    geometry = None
    if args.geometry_run is not None:
        geometry = _load_run(args.geometry_run.resolve(), "Five-DoF geometry")
        runs.append(geometry)
    frozen = None
    if args.frozen_run is not None:
        frozen = _load_run(args.frozen_run.resolve(), "Frozen Five-DoF plan")
        runs.append(frozen)
    _save_metrics_figure(runs, output)
    _save_observation_figure(runs, output)
    if frozen is not None:
        _save_final_outcome_figure(runs, output)
    summary = {str(run["label"]): _summary(run) for run in runs}
    if geometry is not None:
        summary["Five-DoF cumulative-median replay"] = (
            _save_geometry_fusion_replay(geometry, output)
        )
    if args.geometry_simultaneous_run is not None and geometry is not None:
        geometry_simultaneous = _load_run(
            args.geometry_simultaneous_run.resolve(),
            "Geometry + simultaneous",
        )
        geometry_orbit = _load_run(
            args.geometry_run.resolve(), "Geometry + camera orbit"
        )
        _save_control_policy_figure(
            [geometry_simultaneous, geometry_orbit], output
        )
        summary["Geometry + simultaneous"] = _summary(geometry_simultaneous)
        summary["Geometry + camera orbit"] = _summary(geometry_orbit)
        if args.axis_then_run is not None:
            _save_control_failure_figure(
                geometry_simultaneous,
                args.axis_then_run.resolve(),
                geometry_orbit,
                output,
            )
            axis_runtime = json.loads(
                (args.axis_then_run.resolve() / "runtime_result.json").read_text(
                    encoding="utf-8"
                )
            )
            summary["Geometry + axis then position"] = {
                "termination": "PERCEPTION_REJECTED",
                "error_type": axis_runtime.get("error_type"),
                "error": axis_runtime.get("error"),
                "online_target_reached": False,
            }
    summary["comparison_scope"] = (
        "same nominal scene; separate deterministic resets; simulation-only; "
        "posthoc truth used only for scoring; the three continuous visual-servo "
        "baselines did not reach the historical pregrasp, while the frozen-target "
        "plan reached it without visual updates during motion"
        if frozen is not None
        else (
            "same nominal scene and bounded no-contact steps; separate deterministic "
            "resets; simulation-only; posthoc truth used only for scoring"
        )
    )
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
