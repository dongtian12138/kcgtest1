"""CPU static searcher for T_HP display poses using the existing fixed RGB-D.

This is the bounded B-route.  It does not add a third camera and does not run
Isaac.  It searches a finite candidate set, solves IK, checks all seven joint
limits with margin, screens the Cartesian path quality and scene clearance,
and projects the Plug mating-face CAD to require about 80 px short axis.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from kcg_connector.d38999_cad_registration import (
    PLUG_MATING,
    CameraModel,
    fixed_camera_model,
    project,
    proxy_cad_points,
)
from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform
from kcg_connector.display_motion_diagnostics import (
    evaluate_waypoint_path_quality,
    joint_target_limit_violations,
)
from kcg_connector.postgrasp_shadow_view_planner import (
    plan_cartesian_tcp_waypoints,
)
from kcg_connector.wrist_receptacle_view_design import plug_world_pose

SCHEMA_VERSION = "kcg_d38999_fixed_camera_t_hp_display_search_v1"
JOINT_LIMIT_MARGIN_RAD = 0.010
MINIMUM_SHORT_AXIS_PX = 80.0
MINIMUM_PAIR_DIRECTION_DEG = 15.0


def _project_plug_mating(camera: CameraModel, plug_world: np.ndarray, plug_cad):
    points = plug_cad.xyz[plug_cad.label == PLUG_MATING]
    world_points = (plug_world[:3, :3] @ points.T).T + plug_world[:3, 3]
    uv, depth = project(camera, world_points)
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < camera.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < camera.height)
        & (depth > 0)
        & np.isfinite(depth)
    )
    u = uv[inside]
    if len(u) < 2:
        return {
            "pixels": 0,
            "short_axis_px": 0.0,
            "bbox_width_px": 0.0,
            "bbox_height_px": 0.0,
        }
    return {
        "pixels": int(len(u)),
        "short_axis_px": float(
            min(
                np.max(u[:, 0]) - np.min(u[:, 0]),
                np.max(u[:, 1]) - np.min(u[:, 1]),
            )
        ),
        "bbox_width_px": float(np.max(u[:, 0]) - np.min(u[:, 0])),
        "bbox_height_px": float(np.max(u[:, 1]) - np.min(u[:, 1])),
    }


def _candidate_tcp_positions(current_tcp_position, fixed_camera_eye, max_candidates):
    current = np.asarray(current_tcp_position, dtype=np.float64)
    eye = np.asarray(fixed_camera_eye, dtype=np.float64)
    offsets = []
    for dx in np.arange(-0.12, 0.1201, 0.03):
        for dy in np.arange(-0.18, 0.0201, 0.03):
            for dz in np.arange(-0.04, 0.1601, 0.03):
                offsets.append((float(dx), float(dy), float(dz)))
    positions = [current + np.asarray(offset) for offset in offsets]
    positions.sort(
        key=lambda position: float(np.linalg.norm(position - eye))
    )
    return positions[: int(max_candidates)]


def classify_b_search_status(candidates, pair):
    """CANDIDATE_FOUND requires a valid direction-different pair."""
    if pair is not None and len(pair.get("candidates", [])) >= 2:
        return "CANDIDATE_FOUND", "pair_direction_gate_passed"
    if candidates:
        return (
            "T_HP_OBSERVABILITY_REJECTED",
            "single_or_unpaired_candidate_fail_closed",
        )
    return "T_HP_OBSERVABILITY_REJECTED", "no_80px_safe_candidate"


def search_fixed_camera_t_hp_display_candidates(
    *,
    current_arm_q: Sequence[float],
    solve_arm,
    tcp_from_handbase: np.ndarray,
    nominal_hand_to_plug: np.ndarray,
    joint_limits: Sequence[tuple[float, float]],
    fixed_camera_eye: Sequence[float] = (0.550, -0.850, 0.720),
    fixed_camera_target: Sequence[float] = (0.535, -0.0125, 0.231),
    table_top_z_m: float = 0.20,
    fixture_center_m: Sequence[float] = (0.55, 0.185, 0.22),
    fixture_half_extent_m: Sequence[float] = (0.07, 0.07, 0.02),
    max_candidates: int = 120,
    max_wall_seconds: float = 20.0,
    minimum_short_axis_px: float = MINIMUM_SHORT_AXIS_PX,
    minimum_pair_direction_deg: float = MINIMUM_PAIR_DIRECTION_DEG,
) -> dict[str, Any]:
    """Search a bounded candidate set and report whether a safe 80 px pose exists."""
    q0 = np.asarray(current_arm_q, dtype=np.float64).ravel()
    if q0.shape != (7,):
        raise ValueError("current_arm_q must be a 7-vector")
    if int(max_candidates) < 1:
        raise ValueError("max_candidates must be positive")
    start_violations = joint_target_limit_violations(
        q0, joint_limits, margin_rad=JOINT_LIMIT_MARGIN_RAD
    )
    if start_violations:
        return {
            "schema_version": SCHEMA_VERSION,
            "role": "fixed_camera_t_hp_display_search_cpu",
            "status": "T_HP_OBSERVABILITY_REJECTED",
            "reason": "start_q_joint_limit_margin",
            "start_q_violations": start_violations,
            "visual_authorization": False,
            "control_authorized": False,
        }

    camera = fixed_camera_model(
        eye=tuple(fixed_camera_eye),
        target=tuple(fixed_camera_target),
        resolution=(640, 480),
    )
    plug_cad, _ = proxy_cad_points()
    current_tcp = np.asarray(
        iiwa14_grasp_tcp_transform(tuple(float(value) for value in q0))
    )
    candidates = []
    rejected_reasons = []
    search_started = time.monotonic()
    candidates_checked = 0
    for index, tcp_position in enumerate(
        _candidate_tcp_positions(
            current_tcp[:3, 3], fixed_camera_eye, max_candidates
        ),
        start=1,
    ):
        if time.monotonic() - search_started > float(max_wall_seconds):
            rejected_reasons.append("time_budget_exhausted")
            break
        candidates_checked += 1
        try:
            q = np.asarray(
                solve_arm(
                    tuple(float(value) for value in q0),
                    tuple(tcp_position),
                    target_rotation=current_tcp[:3, :3],
                    maximum_iterations=400,
                    damping=1.0e-4,
                ),
                dtype=np.float64,
            )
        except Exception:
            rejected_reasons.append(f"candidate_{index}_ik_failed")
            continue
        limit_violations = joint_target_limit_violations(
            q, joint_limits, margin_rad=JOINT_LIMIT_MARGIN_RAD
        )
        if limit_violations:
            rejected_reasons.append(f"candidate_{index}_joint_limit_margin")
            continue
        target_tcp = np.asarray(
            iiwa14_grasp_tcp_transform(tuple(float(value) for value in q))
        )
        try:
            waypoints = plan_cartesian_tcp_waypoints(
                q0,
                target_tcp,
                solve_arm=solve_arm,
                maximum_step_m=0.005,
            )
        except Exception:
            rejected_reasons.append(f"candidate_{index}_waypoint_failed")
            continue
        distance = float(
            np.linalg.norm(target_tcp[:3, 3] - current_tcp[:3, 3])
        )
        steps_per_waypoint = max(
            1, round(max(4.0, distance / 0.010) * 240.0 / len(waypoints))
        )
        quality = evaluate_waypoint_path_quality(
            waypoints,
            forward_kinematics=iiwa14_grasp_tcp_transform,
            physics_rate_hz=240.0,
            steps_per_waypoint=steps_per_waypoint,
            start_q=q0,
            table_top_z_m=table_top_z_m,
            fixture_center_m=fixture_center_m,
            fixture_half_extent_m=fixture_half_extent_m,
            joint_limits=joint_limits,
            joint_limit_margin_rad=JOINT_LIMIT_MARGIN_RAD,
        )
        if quality["reject"]:
            rejected_reasons.append(f"candidate_{index}_path_quality_reject")
            continue
        plug_world = plug_world_pose(
            q,
            tcp_from_handbase=tcp_from_handbase,
            nominal_hand_to_plug=nominal_hand_to_plug,
        )
        projection = _project_plug_mating(camera, plug_world, plug_cad)
        if projection["short_axis_px"] < float(minimum_short_axis_px):
            rejected_reasons.append(
                f"candidate_{index}_short_axis_"
                f"{projection['short_axis_px']:.2f}px"
            )
            continue
        plug_center = plug_world[:3, 3]
        to_camera = np.asarray(camera.position_world) - plug_center
        distance_to_camera = float(np.linalg.norm(to_camera))
        candidates.append(
            {
                "candidate_index": index,
                "arm_q_rad": q.tolist(),
                "tcp_position_m": tcp_position.tolist(),
                "plug_center_m": plug_center.tolist(),
                "camera_distance_m": distance_to_camera,
                "camera_to_plug_direction": (
                    to_camera / distance_to_camera
                ).tolist(),
                "projection": projection,
                "path_quality": quality,
            }
        )

    pair = None
    for first in candidates:
        for second in candidates:
            if first is second:
                continue
            first_dir = np.asarray(first["camera_to_plug_direction"])
            second_dir = np.asarray(second["camera_to_plug_direction"])
            angle = float(
                math.degrees(
                    math.acos(
                        max(-1.0, min(1.0, float(np.dot(first_dir, second_dir))))
                    )
                )
            )
            if angle >= minimum_pair_direction_deg:
                pair = {
                    "view_ids": ["F0", "F1"],
                    "direction_difference_deg": angle,
                    "candidates": [first, second],
                }
                break
        if pair is not None:
            break

    status, status_reason = classify_b_search_status(candidates, pair)
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "fixed_camera_t_hp_display_search_cpu",
        "status": status,
        "status_reason": status_reason,
        "candidate_budget": int(max_candidates),
        "candidates_checked": candidates_checked,
        "max_wall_seconds": float(max_wall_seconds),
        "minimum_short_axis_px": float(minimum_short_axis_px),
        "minimum_pair_direction_deg": float(minimum_pair_direction_deg),
        "candidates": candidates,
        "pair": pair,
        "rejected_reasons": rejected_reasons[-20:],
        "visual_authorization": False,
        "control_authorized": False,
        "third_camera_added": False,
    }


__all__ = [
    "SCHEMA_VERSION",
    "classify_b_search_status",
    "search_fixed_camera_t_hp_display_candidates",
]
