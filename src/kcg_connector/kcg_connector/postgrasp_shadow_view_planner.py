"""Post-grasp shadow view planning and POSTHOC_TRUTH_ONLY visibility A/B.

The formal view score consumes only RGB, depth, camera calibration and FK.
The A/B evaluator renders proxy CAD at posthoc-truth poses; its output is
explicitly marked POSTHOC_TRUTH_ONLY and never enters estimator or control.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from kcg_connector.d38999_cad_registration import (
    CameraModel,
    PLUG_MATING,
    RECEPTACLE_MATING,
    fixed_camera_model,
    project,
    proxy_cad_points,
    render_points,
    transform_points,
)
from kcg_connector.d38999_inhand_multiview import (
    camera_from_plug_pose,
    compose_pose,
    matrix_pose,
)

AB_SCHEMA_VERSION = "kcg_d38999_postgrasp_visibility_ab_v1"
DIAG_MOUNT_SCHEMA_VERSION = "kcg_d38999_diag_mount_search_v1"


DIAGNOSTIC_MOUNT_CANDIDATES = (
    ("C1", (0.060, 0.0, -0.055)),
    ("C2", (-0.060, 0.0, -0.055)),
    ("C3", (0.040, 0.040, -0.050)),
    ("C4", (0.040, -0.040, -0.050)),
    ("C5", (-0.040, 0.040, -0.050)),
    ("C6", (-0.040, -0.040, -0.050)),
    ("C7", (0.030, 0.0, -0.030)),
    ("C8", (0.0, 0.040, -0.040)),
)
DIAG_MOUNT_TARGET_P = (0.0, 0.0, 0.002)
DIAG_MOUNT_OPTICAL_ANGLE_LIMITS_DEG = (25.0, 50.0)
AB_TRUTH_LABEL = "POSTHOC_TRUTH_ONLY"


@dataclass(frozen=True)
class PredefinedViewPlan:
    view_id: str
    tcp_delta_xyz_rpy: tuple[float, float, float, float, float, float]
    group: str = "postgrasp_inhand_views"
    optical_axis_min_deg: float = 25.0
    optical_axis_max_deg: float = 70.0


DEFAULT_POSTGRASP_PLANS = (
    PredefinedViewPlan("V0", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    PredefinedViewPlan(
        "V1", (0.012, -0.006, -0.030, math.radians(4.0), math.radians(-10.0), 0.0)
    ),
    PredefinedViewPlan(
        "V2", (-0.012, 0.006, -0.030, math.radians(-4.0), math.radians(10.0), 0.0)
    ),
)
DEFAULT_PREINSERT_PLANS = (
    PredefinedViewPlan(
        "FINAL_PREINSERT_VIEW", (0.0, 0.0, -0.012, 0.0, 0.0, 0.0),
        group="final_preinsert_views",
    ),
    PredefinedViewPlan(
        "MULTIVIEW_VIEW_1", (0.012, -0.006, -0.030, math.radians(4.0), math.radians(-10.0), 0.0),
        group="final_preinsert_views",
    ),
    PredefinedViewPlan(
        "MULTIVIEW_VIEW_2", (-0.012, 0.006, -0.030, math.radians(-4.0), math.radians(10.0), 0.0),
        group="final_preinsert_views",
    ),
)


def plug_relative_camera_pose(
    T_WH: np.ndarray, T_HC: np.ndarray, T_HP: np.ndarray
) -> np.ndarray:
    """Return ``T_CP = inv(T_HC) @ T_HP`` for a rigid hand-mounted camera.

    If the camera, hand and Plug are rigidly co-moving, this transform does not
    depend on ``T_WH`` at all.  Moving the arm therefore provides zero new
    Plug-relative viewpoint and must not be counted as T_HP multiview.
    """
    return np.linalg.inv(np.asarray(T_HC, dtype=np.float64)) @ np.asarray(
        T_HP, dtype=np.float64
    )


def fixed_world_camera_plug_pose(
    T_WC: np.ndarray, T_WH: np.ndarray, T_HP: np.ndarray
) -> np.ndarray:
    """Return ``T_CP = inv(T_WC) @ T_WH @ T_HP`` for a fixed world camera."""
    return (
        np.linalg.inv(np.asarray(T_WC, dtype=np.float64))
        @ np.asarray(T_WH, dtype=np.float64)
        @ np.asarray(T_HP, dtype=np.float64)
    )


def score_formal_view(
    *,
    view_id: str,
    timestamp_utc: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    camera: CameraModel,
) -> dict[str, Any]:
    """Semantic-free formal view quality score.

    All numeric thresholds are SIM_TUNING_ONLY_CANDIDATE diagnostics, not
    pass gates.  This function never reads semantic labels or object truth.
    """
    image = np.asarray(rgb, dtype=np.uint8)
    depth_image = np.asarray(depth, dtype=np.float32)
    if image.ndim != 3 or depth_image.shape != image.shape[:2]:
        raise ValueError("rgb/depth shape mismatch")
    finite = np.isfinite(depth_image) & (depth_image > 0.0)
    depth_valid_fraction = float(np.mean(finite)) if finite.size else 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 30, 90) > 0
    edge_fraction = float(np.mean(edge)) if edge.size else 0.0
    height, width = depth_image.shape
    central = np.zeros_like(finite)
    h0, h1 = int(height * 0.2), int(height * 0.8)
    w0, w1 = int(width * 0.2), int(width * 0.8)
    central[h0:h1, w0:w1] = True
    central_depth_fraction = (
        float(np.mean(finite & central)) if np.any(central) else 0.0
    )
    score = (
        2.0 * min(1.0, depth_valid_fraction / 0.10)
        + 1.0 * min(1.0, central_depth_fraction / 0.10)
        + 1.0 * min(1.0, edge_fraction / 0.02)
    )
    return {
        "schema_version": AB_SCHEMA_VERSION,
        "view_id": view_id,
        "timestamp_utc": timestamp_utc,
        "score": float(score),
        "depth_valid_fraction": depth_valid_fraction,
        "central_depth_fraction": central_depth_fraction,
        "rgb_edge_fraction": edge_fraction,
        "semantic_input_used": False,
        "object_truth_used": False,
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
        "pass_gate": None,
    }


def _pixel_diameter(camera: CameraModel, xyz_world: np.ndarray) -> float:
    uv, depth = project(camera, xyz_world)
    valid = (
        (depth > 0.03)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < camera.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < camera.height)
    )
    if int(np.sum(valid)) < 10:
        return 0.0
    points = uv[valid]
    return float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1])))


def _information_condition(
    receptacle_cad,
    plug_pose_receptacle,
    view_pose_receptacle,
    mount_eye,
    mount_target,
) -> float:
    rim = receptacle_cad.xyz[receptacle_cad.label == RECEPTACLE_MATING]
    keep = np.linspace(0, len(rim) - 1, min(240, len(rim)), dtype=np.int64)
    rim = rim[keep]
    steps = np.asarray(
        (0.0002, 0.0002, 0.0002, math.radians(0.15), math.radians(0.15), math.radians(0.15))
    )

    def pixels(candidate):
        return project(
            camera_from_plug_pose(candidate, mount_eye, mount_target, resolution=(640, 480)),
            rim,
        )[0].ravel()

    base = np.asarray(view_pose_receptacle, dtype=np.float64)
    jac = np.column_stack(
        tuple(
            (pixels(base + np.eye(6)[i] * steps[i]) - pixels(base - np.eye(6)[i] * steps[i]))
            / (2.0 * steps[i])
            for i in range(6)
        )
    )
    scale = np.asarray((0.0005, 0.0005, 0.0010, math.radians(2), math.radians(2), math.radians(2)))
    hessian = (jac * scale).T @ (jac * scale)
    values = np.linalg.eigvalsh(hessian)
    return float(values[-1] / max(values[0], 1.0e-12))


def evaluate_visibility_ab(
    *,
    postgrasp_plug_pose_receptacle,
    preinsert_plug_pose_receptacle=(0.0, 0.0, -0.012, 0.0, 0.0, 0.0),
    postgrasp_view_poses_receptacle: Sequence[Sequence[float]] | None = None,
    preinsert_view_poses_receptacle: Sequence[Sequence[float]] | None = None,
    mount_eye_plug=(0.120, 0.0, 0.060),
    mount_target_plug=(0.0, 0.0, 0.006),
    resolution=(1280, 720),
) -> dict[str, Any]:
    """POSTHOC_TRUTH_ONLY same-frame visibility A/B diagnostic."""
    plug_cad, receptacle_cad = proxy_cad_points()
    nominal = np.asarray(preinsert_plug_pose_receptacle, dtype=np.float64)
    if postgrasp_view_poses_receptacle is None:
        postgrasp_view_poses_receptacle = [
            np.asarray(plan.tcp_delta_xyz_rpy, dtype=np.float64)
            for plan in DEFAULT_POSTGRASP_PLANS
        ]
    if preinsert_view_poses_receptacle is None:
        preinsert_view_poses_receptacle = [
            np.asarray(plan.tcp_delta_xyz_rpy, dtype=np.float64)
            for plan in DEFAULT_PREINSERT_PLANS
        ]
    actual = np.asarray(postgrasp_plug_pose_receptacle, dtype=np.float64)
    families = {
        "A_POSTGRASP_H0": (actual, postgrasp_view_poses_receptacle),
        "B_PREINSERT": (nominal, preinsert_view_poses_receptacle),
    }
    report = {
        "schema_version": AB_SCHEMA_VERSION,
        "truth_scope": AB_TRUTH_LABEL,
        "truth_used_for_control": False,
        "truth_used_for_estimator": False,
        "families": {},
        "decision": "UNDECIDED",
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }
    family_summary = {}
    for family_name, (family_pose, deltas) in families.items():
        views = []
        hessians = []
        for view_index, delta in enumerate(deltas):
            view_pose = compose_pose(family_pose, delta)
            camera = camera_from_plug_pose(
                view_pose, mount_eye_plug, mount_target_plug, resolution=resolution
            )
            plug_world = transform_points(plug_cad, view_pose)
            receptacle_world = transform_points(
                receptacle_cad, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            )
            observation = render_points(camera, (plug_world, receptacle_world))
            plug_pixels = int(np.sum(observation["label"] == PLUG_MATING))
            receptacle_pixels = int(
                np.sum(observation["label"] == RECEPTACLE_MATING)
            )
            plug_diameter = _pixel_diameter(
                camera, plug_world.xyz[plug_world.label == PLUG_MATING]
            )
            receptacle_diameter = _pixel_diameter(
                camera, receptacle_world.xyz[receptacle_world.label == RECEPTACLE_MATING]
            )
            condition = _information_condition(
                receptacle_cad, family_pose, view_pose, mount_eye_plug, mount_target_plug
            )
            hessians.append(condition)
            views.append(
                {
                    "view_index": view_index,
                    "plug_mating_visible_pixels": plug_pixels,
                    "receptacle_mating_visible_pixels": receptacle_pixels,
                    "plug_projected_diameter_px": plug_diameter,
                    "receptacle_projected_diameter_px": receptacle_diameter,
                    "same_frame_visible": bool(plug_pixels >= 1 and receptacle_pixels >= 1),
                    "single_view_condition_number": condition,
                }
            )
        joint_condition = float(max(hessians)) if hessians else math.inf
        summary = {
            "view_count": len(views),
            "views": views,
            "all_views_same_frame_visible": all(
                item["same_frame_visible"] for item in views
            ),
            "min_plug_pixels": min(item["plug_mating_visible_pixels"] for item in views),
            "min_receptacle_pixels": min(
                item["receptacle_mating_visible_pixels"] for item in views
            ),
            "min_plug_diameter_px": min(
                item["plug_projected_diameter_px"] for item in views
            ),
            "min_receptacle_diameter_px": min(
                item["receptacle_projected_diameter_px"] for item in views
            ),
            "max_single_view_condition_number": joint_condition,
            "joint_visibility_feasible_posthoc": bool(
                all(item["same_frame_visible"] for item in views)
            ),
        }
        family_summary[family_name] = summary
        report["families"][family_name] = summary
    a_ok = family_summary["A_POSTGRASP_H0"]["joint_visibility_feasible_posthoc"]
    b_ok = family_summary["B_PREINSERT"]["joint_visibility_feasible_posthoc"]
    if a_ok and b_ok:
        report["decision"] = "A_AND_B_FEASIBLE_POSTHOC"
    elif a_ok:
        report["decision"] = "A_FEASIBLE_B_ONLY_IF_TRANSPORTED"
    elif b_ok:
        report["decision"] = "B_FEASIBLE_SPLIT_ARCHITECTURE_REQUIRED"
    else:
        report["decision"] = "NEITHER_FAMILY_FEASIBLE_POSTHOC"
    return report


def _camera_from_t_wc(t_wc, resolution=(1280, 720)):
    forward = np.asarray(t_wc[:3, :3]) @ np.asarray((0.0, 0.0, 1.0))
    target = np.asarray(t_wc[:3, 3]) + forward
    return fixed_camera_model(
        eye=tuple(float(v) for v in t_wc[:3, 3]),
        target=tuple(float(v) for v in target),
        resolution=resolution,
    )


def _family_report_from_frames(hand_poses, t_hc, t_wr, t_hp, plug_cad, receptacle_cad, resolution):
    views = []
    conditions = []
    for hand_pose in hand_poses:
        t_wh = np.asarray(hand_pose, dtype=np.float64)
        t_wc = t_wh @ np.asarray(t_hc, dtype=np.float64)
        camera = _camera_from_t_wc(t_wc, resolution=resolution)
        t_wp = t_wh @ np.asarray(t_hp, dtype=np.float64)
        t_rp = np.linalg.inv(np.asarray(t_wr, dtype=np.float64)) @ t_wp
        plug_world = transform_points(
            plug_cad, matrix_pose(np.asarray(t_wp, dtype=np.float64))
        )
        receptacle_world = transform_points(
            receptacle_cad,
            matrix_pose(np.asarray(t_wr, dtype=np.float64)),
        )
        observation = render_points(camera, (plug_world, receptacle_world))
        plug_pixels = int(np.sum(observation["label"] == PLUG_MATING))
        receptacle_pixels = int(np.sum(observation["label"] == RECEPTACLE_MATING))
        plug_diameter = _pixel_diameter(
            camera, plug_world.xyz[plug_world.label == PLUG_MATING]
        )
        receptacle_diameter = _pixel_diameter(
            camera, receptacle_world.xyz[receptacle_world.label == RECEPTACLE_MATING]
        )
        condition = _condition_from_world_camera(
            camera, receptacle_world, matrix_pose(t_rp)
        )
        conditions.append(condition)
        views.append(
            {
                "plug_mating_visible_pixels": plug_pixels,
                "receptacle_mating_visible_pixels": receptacle_pixels,
                "plug_projected_diameter_px": plug_diameter,
                "receptacle_projected_diameter_px": receptacle_diameter,
                "same_frame_visible": bool(plug_pixels >= 1 and receptacle_pixels >= 1),
                "condition_number": condition,
            }
        )
    return {
        "views": views,
        "all_views_same_frame_visible": all(v["same_frame_visible"] for v in views),
        "max_condition_number": float(max(conditions)) if conditions else math.inf,
    }


def _condition_from_world_camera(camera, receptacle_world, t_rp_pose):
    rim = receptacle_world.xyz[receptacle_world.label == RECEPTACLE_MATING]
    keep = np.linspace(0, len(rim) - 1, min(240, len(rim)), dtype=np.int64)
    rim = rim[keep]
    steps = np.asarray((0.0002, 0.0002, 0.0002, math.radians(0.15), math.radians(0.15), math.radians(0.15)))
    base = np.asarray(t_rp_pose, dtype=np.float64)

    def pixels(candidate):
        # Posthoc-only condition proxy around the current frame.
        return project(camera, rim)[0].ravel()
    jac = np.column_stack(
        tuple(
            (pixels(base + np.eye(6)[i] * steps[i]) - pixels(base - np.eye(6)[i] * steps[i]))
            / (2.0 * steps[i])
            for i in range(6)
        )
    )
    scale = np.asarray((0.0005, 0.0005, 0.0010, math.radians(2), math.radians(2), math.radians(2)))
    values = np.linalg.eigvalsh((jac * scale).T @ (jac * scale))
    return float(values[-1] / max(values[0], 1.0e-12))


def evaluate_visibility_ab_from_frames(
    *,
    postgrasp_hand_poses,
    preinsert_hand_poses,
    t_hc,
    t_wr,
    t_hp,
    resolution=(1280, 720),
):
    plug_cad, receptacle_cad = proxy_cad_points()
    a = _family_report_from_frames(
        postgrasp_hand_poses, t_hc, t_wr, t_hp, plug_cad, receptacle_cad, resolution
    )
    b = _family_report_from_frames(
        preinsert_hand_poses, t_hc, t_wr, t_hp, plug_cad, receptacle_cad, resolution
    )
    a_ok = a["all_views_same_frame_visible"]
    b_ok = b["all_views_same_frame_visible"]
    decision = (
        "A_AND_B_FEASIBLE_POSTHOC"
        if a_ok and b_ok
        else "A_FEASIBLE_B_ONLY_IF_TRANSPORTED"
        if a_ok
        else "B_FEASIBLE_SPLIT_ARCHITECTURE_REQUIRED"
        if b_ok
        else "NEITHER_FAMILY_FEASIBLE_POSTHOC"
    )
    return {
        "schema_version": AB_SCHEMA_VERSION,
        "truth_scope": AB_TRUTH_LABEL,
        "truth_used_for_control": False,
        "truth_used_for_estimator": False,
        "decision": decision,
        "families": {"A_POSTGRASP_H0": a, "B_PREINSERT": b},
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }


def diagnostic_optical_axis_angle_deg(eye_plug, target_plug=DIAG_MOUNT_TARGET_P):
    eye = np.asarray(eye_plug, dtype=np.float64)
    target = np.asarray(target_plug, dtype=np.float64)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    cos_angle = float(np.clip(forward[2], -1.0, 1.0))
    return float(math.degrees(math.acos(cos_angle)))


def diagnostic_hp_envelope_samples(nominal_hp):
    """Truth-free deterministic 6D envelope samples: center, axis extremes, combos."""
    nominal = np.asarray(nominal_hp, dtype=np.float64)
    translation = 0.002
    rotation = math.radians(6.0)
    samples = [nominal.copy()]
    for index in range(6):
        step = translation if index < 3 else rotation
        plus = nominal.copy()
        plus[index] += step
        samples.append(plus)
        minus = nominal.copy()
        minus[index] -= step
        samples.append(minus)
    sign_pairs = ((1, 1, 1, -1, -1, 1), (1, -1, 1, 1, -1, -1), (-1, 1, -1, 1, 1, -1), (-1, -1, -1, -1, 1, 1))
    for signs in sign_pairs:
        sample = nominal.copy()
        for index, sign in enumerate(signs):
            step = translation if index < 3 else rotation
            sample[index] += sign * step
        samples.append(sample)
    return samples


def diagnostic_mount_hard_gates(eye_plug, target_plug=DIAG_MOUNT_TARGET_P):
    angle = diagnostic_optical_axis_angle_deg(eye_plug, target_plug)
    low, high = DIAG_MOUNT_OPTICAL_ANGLE_LIMITS_DEG
    eye = np.asarray(eye_plug, dtype=np.float64)
    reasons = []
    if not (low <= angle <= high):
        reasons.append("OPTICAL_AXIS_ANGLE_OUT_OF_RANGE")
    if eye[2] >= -0.010:
        reasons.append("CAMERA_HOUSING_TOO_CLOSE_TO_MATING_FACE")
    if float(np.linalg.norm(eye)) < 0.020:
        reasons.append("CAMERA_TOO_CLOSE_TO_MATING_FACE")
    return {"passed": not reasons, "reasons": reasons, "angle_deg": angle}


def diagnostic_mount_score(metrics):
    shell = float(metrics.get("projected_shell_depth_support", 0.0))
    socket = float(metrics.get("projected_socket_depth_support", 0.0))
    central = float(metrics.get("central_depth_fraction", 0.0))
    edge = float(metrics.get("edge_support_fraction", 0.0))
    occlusion = float(metrics.get("foreground_occlusion_fraction", 0.0))
    condition = max(1.0, float(metrics.get("condition_number_5d", 1.0)))
    return float(
        0.25 * shell + 0.30 * socket + 0.15 * central + 0.15 * edge
        - 0.15 * occlusion + min(0.0, math.log10(1.0e6 / condition))
    )


def write_ab_report(path: Path | str, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "AB_SCHEMA_VERSION",
    "AB_TRUTH_LABEL",
    "DEFAULT_POSTGRASP_PLANS",
    "DEFAULT_PREINSERT_PLANS",
    "PredefinedViewPlan",
    "evaluate_visibility_ab",
    "score_formal_view",
    "write_ab_report",
]


def _camera_pose_from_model(camera: CameraModel) -> np.ndarray:
    rows = np.asarray(camera.world_to_camera, dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rows.T
    pose[:3, 3] = np.asarray(camera.position_world, dtype=np.float64)
    return pose


def run_fixed_camera_visibility_ab(
    *,
    report_path: Path | str,
    controller_steps_path: Path | str,
    pick_config_path: Path | str,
    rgbd_config_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """POSTHOC_TRUTH_ONLY A/B: fixed world camera vs co-moving wrist V0.

    This is the minimal offline evidence required before any wrist-arm motion
    is attempted for T_HP.  The fixed camera observes the Plug from a world
    frame; the wrist camera is rigidly attached to the hand and is therefore a
    single Plug-relative viewpoint regardless of arm motion.
    """
    from kcg_connector.d38999_tabletop_pick import (
        iiwa14_grasp_tcp_transform,
        load_d38999_tabletop_pick_config,
    )

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    last = None
    with Path(controller_steps_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    if last is None or report.get("passed") is not True:
        return {
            "schema_version": AB_SCHEMA_VERSION,
            "truth_scope": AB_TRUTH_LABEL,
            "decision": "INSUFFICIENT_POSTHOC_FRAME_DATA",
        }
    pick = load_d38999_tabletop_pick_config(Path(pick_config_path))
    rgbd_doc = yaml.safe_load(Path(rgbd_config_path).read_text(encoding="utf-8"))
    arm_q = np.asarray(last["arm_q_actual_rad"], dtype=np.float64)
    tcp_world = np.asarray(
        iiwa14_grasp_tcp_transform(tuple(float(v) for v in arm_q))
    )
    tcp_from_hand = np.eye(4, dtype=np.float64)
    tcp_from_hand[2, 3] = -float(pick.geometry_candidate.handbase_to_tcp_m)
    t_wh_actual = tcp_world @ tcp_from_hand
    t_hp_actual = np.asarray(report["posthoc_t_hand_plug_actual"])
    t_wp_actual = t_wh_actual @ t_hp_actual
    plug_cad, receptacle_cad = proxy_cad_points()

    fixed_camera = fixed_camera_model(
        eye=tuple(rgbd_doc["camera"]["eye_m"]),
        target=tuple(rgbd_doc["camera"]["target_m"]),
        resolution=tuple(rgbd_doc["camera"]["resolution"]),
    )
    mount_eye = (0.120, 0.0, 0.060)
    mount_target = (0.0, 0.0, 0.006)
    wrist_camera_in_plug = fixed_camera_model(
        eye=mount_eye, target=mount_target, resolution=(1280, 720)
    )
    nominal_hp = np.asarray(report["posthoc_t_hand_plug_nominal"])
    t_hc_wrist = nominal_hp @ _camera_pose_from_model(wrist_camera_in_plug)

    t_wc_wrist = t_wh_actual @ t_hc_wrist
    wrist_eye = tuple(float(v) for v in t_wc_wrist[:3, 3])
    wrist_forward = t_wc_wrist[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
    wrist_target = tuple(
        float(v) for v in (np.asarray(t_wc_wrist[:3, 3]) + wrist_forward)
    )
    wrist_camera = fixed_camera_model(
        eye=wrist_eye, target=wrist_target, resolution=(1280, 720)
    )
    families = {
        "A_FIXED_WORLD_CAMERA": fixed_camera,
        "B_WRIST_COMOVING_V0": wrist_camera,
    }
    result = {"schema_version": AB_SCHEMA_VERSION, "truth_scope": AB_TRUTH_LABEL}
    summary = {}
    for name, camera in families.items():
        plug_world = transform_points(plug_cad, matrix_pose(t_wp_actual))
        receptacle_world = transform_points(
            receptacle_cad, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        observation = render_points(camera, (plug_world, receptacle_world))
        summary[name] = {
            "plug_mating_visible_pixels": int(
                np.sum(observation["label"] == PLUG_MATING)
            ),
            "receptacle_mating_visible_pixels": int(
                np.sum(observation["label"] == RECEPTACLE_MATING)
            ),
            "plug_receptacle_same_frame": bool(
                np.any(observation["label"] == PLUG_MATING)
                and np.any(observation["label"] == RECEPTACLE_MATING)
            ),
        }
    fixed_ok = summary["A_FIXED_WORLD_CAMERA"]["plug_mating_visible_pixels"] >= 100
    result["families"] = summary
    result["T_HP_observation_division"] = {
        "fixed_world_camera": "independent_T_HP_view_source",
        "wrist_comoving_views": "single_plug_relative_viewpoint_do_not_count_as_multiview",
    }
    result["decision"] = (
        "FIXED_CAMERA_PLUG_VISIBLE_OFFLINE"
        if fixed_ok
        else "FIXED_CAMERA_PLUG_NOT_VISIBLE_OFFLINE"
    )
    result["threshold_label"] = "SIM_TUNING_ONLY_CANDIDATE"
    write_ab_report(Path(output_path), result)
    return result


def fixed_camera_visibility_ab_cli() -> int:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "phase1_codex_shadow_smoke_mountfix1/seed000"
            / "nominal_physics_report.json"
        ),
    )
    parser.add_argument(
        "--controller-steps",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "phase1_codex_shadow_smoke_mountfix1/seed000"
            / "controller_steps.jsonl"
        ),
    )
    parser.add_argument(
        "--pick-config",
        default=str(
            repository / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
        ),
    )
    parser.add_argument(
        "--rgbd-config",
        default=str(
            repository / "src/kcg_connector/config/d38999_rgbd_bootstrap_v1.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "deepseek/offline_fixed_camera_visibility_ab.json"
        ),
    )
    args = parser.parse_args()
    result = run_fixed_camera_visibility_ab(
        report_path=args.report,
        controller_steps_path=args.controller_steps,
        pick_config_path=args.pick_config,
        rgbd_config_path=args.rgbd_config,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(fixed_camera_visibility_ab_cli())


def plan_two_fixed_camera_inspection_poses(
    current_arm_q,
    *,
    solve_arm,
    handbase_to_tcp_m,
    nominal_hand_to_plug,
    fixed_camera_eye,
    fixed_camera_target,
    max_joint_inf_rad=0.20,
):
    """Generate two distinct, bounded inspection arm targets.

    Truth-free: uses current robot q, nominal T_HP, fixed camera calibration,
    and two deterministic TCP rotation candidates.
    """
    from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform
    from kcg_connector.d38999_inhand_multiview import pose_matrix

    q0 = np.asarray(current_arm_q, dtype=np.float64)
    tcp0 = np.asarray(iiwa14_grasp_tcp_transform(tuple(q0)))
    tcp_from_hand = np.eye(4)
    tcp_from_hand[2, 3] = -float(handbase_to_tcp_m)
    hand0 = tcp0 @ tcp_from_hand
    plug0 = hand0 @ np.asarray(nominal_hand_to_plug)
    camera = np.asarray(fixed_camera_eye, dtype=np.float64)
    target = np.asarray(fixed_camera_target, dtype=np.float64)
    look = target - camera
    look = look / np.linalg.norm(look)
    current_forward = plug0[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
    dot = float(np.dot(current_forward, look))
    sign = 1.0 if dot >= 0.0 else -1.0
    poses = []
    for multiplier in (1.0, -1.0):
        desired_tcp = tcp0 @ pose_matrix(
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    sign * multiplier * math.radians(2.0),
                    -sign * multiplier * math.radians(2.0),
                    0.0,
                ]
            )
        )
        target_q = np.asarray(
            solve_arm(
                tuple(q0),
                tuple(desired_tcp[:3, 3]),
                target_rotation=desired_tcp[:3, :3],
                maximum_iterations=120,
                damping=1.0e-3,
            ),
            dtype=np.float64,
        )
        if np.max(np.abs(target_q - q0)) > max_joint_inf_rad:
            raise ValueError("inspection pose exceeds joint budget")
        poses.append(
            {
                "arm_q_rad": target_q.tolist(),
                "tcp_target": desired_tcp.tolist(),
                "max_abs_dq_rad": float(np.max(np.abs(target_q - q0))),
            }
        )
    if np.max(np.abs(np.asarray(poses[0]["arm_q_rad"]) - np.asarray(poses[1]["arm_q_rad"]))) < 1.0e-6:
        raise ValueError("inspection poses are not distinct")
    return poses


FC_PGDF_DISPLAY_CANDIDATES = (
    {"view_id": "F0", "tcp_delta_xyz_m": (-0.12, -0.10, 0.08)},
    {"view_id": "F1", "tcp_delta_xyz_m": (0.04, -0.14, 0.00)},
)


def plan_two_fixed_camera_display_feasibility_poses(
    current_arm_q,
    *,
    solve_arm,
    handbase_to_tcp_m,
    nominal_hand_to_plug,
    fixed_camera_eye,
    max_joint_inf_rad=1.25,
    candidates=FC_PGDF_DISPLAY_CANDIDATES,
):
    """Generate two deterministic display poses for FC-PGDF-01a.

    The candidates are larger bounded TCP translations than the old +-2 deg
    inspection family.  They keep the current TCP orientation and use the
    nominal hand-to-plug transform only as a static model prior to predict
    the Plug center and camera-to-Plug direction for reporting; no live
    object pose, contact report or collider identity is read.
    """
    from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform

    q0 = np.asarray(current_arm_q, dtype=np.float64)
    if q0.shape != (7,) or not np.all(np.isfinite(q0)):
        raise ValueError("current_arm_q must be a finite 7-vector")
    tcp0 = np.asarray(iiwa14_grasp_tcp_transform(tuple(q0)))
    tcp_from_hand = np.eye(4, dtype=np.float64)
    tcp_from_hand[2, 3] = -float(handbase_to_tcp_m)
    tcp_to_plug_nominal = tcp_from_hand @ np.asarray(
        nominal_hand_to_plug, dtype=np.float64
    )
    camera = np.asarray(fixed_camera_eye, dtype=np.float64)
    if camera.shape != (3,) or not np.all(np.isfinite(camera)):
        raise ValueError("fixed_camera_eye must be a finite 3-vector")

    poses = []
    for candidate in candidates:
        view_id = candidate["view_id"]
        delta = np.asarray(candidate["tcp_delta_xyz_m"], dtype=np.float64)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError(f"{view_id} TCP delta is invalid")
        target_tcp_position = tcp0[:3, 3] + delta
        target_q = np.asarray(
            solve_arm(
                tuple(q0),
                tuple(target_tcp_position),
                target_rotation=tcp0[:3, :3],
                maximum_iterations=400,
                damping=1.0e-4,
            ),
            dtype=np.float64,
        )
        if target_q.shape != (7,) or not np.all(np.isfinite(target_q)):
            raise ValueError(f"{view_id} IK returned invalid joints")
        max_abs_dq = float(np.max(np.abs(target_q - q0)))
        if max_abs_dq > float(max_joint_inf_rad):
            raise ValueError(
                f"{view_id} exceeds joint budget: "
                f"max_abs_dq={max_abs_dq:.6f}"
            )
        target_tcp = np.asarray(
            iiwa14_grasp_tcp_transform(tuple(target_q))
        )
        nominal_plug_center = (
            target_tcp @ tcp_to_plug_nominal
        )[:3, 3]
        camera_to_plug = camera - nominal_plug_center
        camera_distance_m = float(np.linalg.norm(camera_to_plug))
        if camera_distance_m <= 1.0e-9:
            raise ValueError(f"{view_id} is degenerate with camera center")
        poses.append(
            {
                "view_id": view_id,
                "tcp_delta_xyz_m": delta.tolist(),
                "arm_q_rad": target_q.tolist(),
                "tcp_target": target_tcp.tolist(),
                "max_abs_dq_rad": max_abs_dq,
                "nominal_plug_center_m": nominal_plug_center.tolist(),
                "camera_to_plug_direction": (
                    camera_to_plug / camera_distance_m
                ).tolist(),
                "camera_distance_m": camera_distance_m,
            }
        )

    first = np.asarray(poses[0]["camera_to_plug_direction"])
    second = np.asarray(poses[1]["camera_to_plug_direction"])
    direction_difference_deg = float(
        math.degrees(
            math.acos(
                max(-1.0, min(1.0, float(np.dot(first, second))))
            )
        )
    )
    if np.max(
        np.abs(
            np.asarray(poses[0]["arm_q_rad"])
            - np.asarray(poses[1]["arm_q_rad"])
        )
    ) < 1.0e-6:
        raise ValueError("display poses are not distinct")
    return poses, direction_difference_deg


ROBOT_SIDE_CAMERA_DISPLAY_CANDIDATES = (
    {"view_id": "F0", "plug_center_xyz_m": (0.42, -0.21, 0.37)},
    {"view_id": "F1", "plug_center_xyz_m": (0.48, -0.16, 0.36)},
)


def plan_two_robot_side_camera_display_poses(
    current_arm_q,
    *,
    solve_arm,
    handbase_to_tcp_m,
    nominal_hand_to_plug,
    fixed_camera_eye,
    max_joint_inf_rad=1.25,
    candidates=ROBOT_SIDE_CAMERA_DISPLAY_CANDIDATES,
):
    """Generate two display poses under the robot-side near-field camera.

    The plug face is upward-facing in the nominal grasp, so both candidates
    keep the current TCP orientation and translate the nominal Plug center to
    the two deterministic positions under the overhead camera.  As with the
    original planner, only static model priors and robot FK are used.
    """
    from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform

    q0 = np.asarray(current_arm_q, dtype=np.float64)
    if q0.shape != (7,) or not np.all(np.isfinite(q0)):
        raise ValueError("current_arm_q must be a finite 7-vector")
    tcp0 = np.asarray(iiwa14_grasp_tcp_transform(tuple(q0)))
    tcp_from_hand = np.eye(4, dtype=np.float64)
    tcp_from_hand[2, 3] = -float(handbase_to_tcp_m)
    tcp_to_plug_nominal = tcp_from_hand @ np.asarray(
        nominal_hand_to_plug, dtype=np.float64
    )
    camera = np.asarray(fixed_camera_eye, dtype=np.float64)
    if camera.shape != (3,) or not np.all(np.isfinite(camera)):
        raise ValueError("fixed_camera_eye must be a finite 3-vector")

    poses = []
    for candidate in candidates:
        view_id = candidate["view_id"]
        plug_center = np.asarray(
            candidate["plug_center_xyz_m"], dtype=np.float64
        )
        if plug_center.shape != (3,) or not np.all(np.isfinite(plug_center)):
            raise ValueError(f"{view_id} plug center candidate is invalid")
        # ``tcp_to_plug_nominal[:3,3]`` is expressed in the TCP frame.
        # Rotate it into the world frame before subtracting it from the
        # desired Plug-center world position.
        target_tcp_position = plug_center - (
            tcp_to_plug_nominal[:3, :3]
            @ tcp_to_plug_nominal[:3, 3]
        )
        target_q = np.asarray(
            solve_arm(
                tuple(q0),
                tuple(target_tcp_position),
                target_rotation=tcp0[:3, :3],
                maximum_iterations=400,
                damping=1.0e-4,
            ),
            dtype=np.float64,
        )
        if target_q.shape != (7,) or not np.all(np.isfinite(target_q)):
            raise ValueError(f"{view_id} IK returned invalid joints")
        max_abs_dq = float(np.max(np.abs(target_q - q0)))
        if max_abs_dq > float(max_joint_inf_rad):
            raise ValueError(
                f"{view_id} exceeds joint budget: "
                f"max_abs_dq={max_abs_dq:.6f}"
            )
        target_tcp = np.asarray(
            iiwa14_grasp_tcp_transform(tuple(target_q))
        )
        nominal_plug_center = (
            target_tcp @ tcp_to_plug_nominal
        )[:3, 3]
        camera_to_plug = camera - nominal_plug_center
        camera_distance_m = float(np.linalg.norm(camera_to_plug))
        if camera_distance_m <= 1.0e-9:
            raise ValueError(f"{view_id} is degenerate with camera center")
        poses.append(
            {
                "view_id": view_id,
                "plug_center_candidate_m": plug_center.tolist(),
                "arm_q_rad": target_q.tolist(),
                "tcp_target": target_tcp.tolist(),
                "max_abs_dq_rad": max_abs_dq,
                "nominal_plug_center_m": nominal_plug_center.tolist(),
                "camera_to_plug_direction": (
                    camera_to_plug / camera_distance_m
                ).tolist(),
                "camera_distance_m": camera_distance_m,
            }
        )

    first = np.asarray(poses[0]["camera_to_plug_direction"])
    second = np.asarray(poses[1]["camera_to_plug_direction"])
    direction_difference_deg = float(
        math.degrees(
            math.acos(
                max(-1.0, min(1.0, float(np.dot(first, second))))
            )
        )
    )
    if np.max(
        np.abs(
            np.asarray(poses[0]["arm_q_rad"])
            - np.asarray(poses[1]["arm_q_rad"])
        )
    ) < 1.0e-6:
        raise ValueError("robot-side display poses are not distinct")
    return poses, direction_difference_deg


def plan_cartesian_tcp_waypoints(
    current_arm_q,
    target_tcp,
    *,
    solve_arm,
    maximum_step_m=0.005,
):
    """Plan bounded Cartesian TCP waypoints using the local fixed-q7 IK.

    Joint-space interpolation between two valid IK families can dip the TCP
    toward the table.  This helper interpolates the TCP position in the world
    frame and re-solves each waypoint from the previous solution, so the TCP
    height stays near the straight Cartesian path.
    """
    from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform

    q0 = np.asarray(current_arm_q, dtype=np.float64)
    target = np.asarray(target_tcp, dtype=np.float64)
    if q0.shape != (7,) or target.shape != (4, 4):
        raise ValueError("invalid waypoint planning inputs")
    tcp0 = np.asarray(iiwa14_grasp_tcp_transform(tuple(q0)))
    start_position = tcp0[:3, 3]
    target_position = target[:3, 3]
    distance = float(np.linalg.norm(target_position - start_position))
    steps = max(1, int(math.ceil(distance / float(maximum_step_m))))
    waypoints = []
    joints = q0.copy()
    for index in range(1, steps + 1):
        alpha = float(index) / float(steps)
        position = start_position + alpha * (
            target_position - start_position
        )
        joints = np.asarray(
            solve_arm(
                tuple(joints),
                tuple(position),
                target_rotation=tcp0[:3, :3],
                maximum_iterations=200,
                damping=1.0e-4,
            ),
            dtype=np.float64,
        )
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"waypoint {index} returned invalid joints")
        waypoints.append(joints.copy())
    return waypoints
