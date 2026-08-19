"""CPU-only wrist-mounted RGB-D design for observing fixed Receptacle.

The wrist camera is rigidly attached to the hand, so arm motion does not
create new Plug-relative views for T_hand_plug.  It does create new
Receptacle-relative views because Receptacle is world-fixed.  This module
designs and screens those T_receptacle_plug views without Isaac and without
reading live object truth or contact reports.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from kcg_connector.d38999_cad_registration import (
    RECEPTACLE_MATING,
    CameraModel,
    fixed_camera_model,
    project,
    proxy_cad_points,
)
from kcg_connector.d38999_inhand_multiview import pose_matrix
from kcg_connector.d38999_tabletop_pick import (
    iiwa14_grasp_tcp_transform,
)
from kcg_connector.display_motion_diagnostics import (
    joint_target_limit_violations,
)

SCHEMA_VERSION = "kcg_d38999_wrist_receptacle_view_design_v1"
WRIST_CAMERA_RESOLUTION = (1280, 720)
WRIST_FOCAL_LENGTH_MM = 24.0
WRIST_HORIZONTAL_APERTURE_MM = 20.955
JOINT_LIMIT_MARGIN_RAD = 0.010


@dataclass(frozen=True)
class WristViewPlan:
    view_id: str
    tcp_delta_pose6: tuple[float, ...]


DEFAULT_WRIST_RECEPTACLE_PLANS = (
    WristViewPlan("W_R0", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    WristViewPlan(
        "W_R1",
        (0.020, -0.020, -0.020, 0.017453292519943295, -0.04363323129985824, 0.0),
    ),
    WristViewPlan(
        "W_R2",
        (-0.012, 0.006, -0.025, -0.08, 0.10, 0.0),
    ),
)


def _camera_pose_from_look_at(eye: Sequence[float], target: Sequence[float]) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if eye.shape != (3,) or target.shape != (3,):
        raise ValueError("eye and target must be 3-vectors")
    forward = target - eye
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-12:
        raise ValueError("camera eye and target must differ")
    forward = forward / norm
    reference = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = np.cross(forward, reference)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-9:
        reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        right = np.cross(forward, reference)
        right = right / np.linalg.norm(right)
    else:
        right = right / right_norm
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = -up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def _camera_model_from_pose(pose: np.ndarray, resolution=(1280, 720)) -> CameraModel:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("camera pose must be 4x4")
    width, height = resolution
    fx = WRIST_FOCAL_LENGTH_MM / WRIST_HORIZONTAL_APERTURE_MM * width
    # CameraModel.world_to_camera rows are camera-frame basis vectors in
    # world coordinates: right, -up, forward.  The 4x4 pose columns are
    # exactly [right, -up, forward], so the rows are the transpose of the
    # rotation block.
    rotation = pose[:3, :3]
    return CameraModel(
        width=width,
        height=height,
        fx=fx,
        fy=fx,
        cx=0.5 * (width - 1),
        cy=0.5 * (height - 1),
        position_world=tuple(float(value) for value in pose[:3, 3]),
        world_to_camera=tuple(
            tuple(float(value) for value in row)
            for row in rotation.T
        ),
    )


def _rotation_from_wxyz(value: Sequence[float]) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64).ravel()
    quat = quat / np.linalg.norm(quat)
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def receptacle_world_pose(
    fixed_position: Sequence[float], fixed_axis: Sequence[float]
) -> np.ndarray:
    position = np.asarray(fixed_position, dtype=np.float64).ravel()
    axis = np.asarray(fixed_axis, dtype=np.float64).ravel()
    if position.shape != (3,) or axis.shape != (3,):
        raise ValueError("receptacle datum must be 3-vectors")
    axis = axis / np.linalg.norm(axis)
    reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    x = reference - np.dot(reference, axis) * axis
    if np.linalg.norm(x) <= 1.0e-9:
        reference = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        x = reference - np.dot(reference, axis) * axis
    x = x / np.linalg.norm(x)
    y = np.cross(axis, x)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = axis
    pose[:3, 3] = position
    return pose


def plug_world_pose(
    arm_q: Sequence[float],
    *,
    tcp_from_handbase: np.ndarray,
    nominal_hand_to_plug: np.ndarray,
) -> np.ndarray:
    tcp = np.asarray(iiwa14_grasp_tcp_transform(tuple(float(v) for v in arm_q)))
    hand = tcp @ np.asarray(tcp_from_handbase, dtype=np.float64)
    return hand @ np.asarray(nominal_hand_to_plug, dtype=np.float64)


def frozen_wrist_camera_hand_transform(
    nominal_hand_to_plug: np.ndarray,
    *,
    eye_plug_m=(0.120, 0.0, 0.060),
    target_plug_m=(0.0, 0.0, 0.006),
    resolution=(1280, 720),
) -> np.ndarray:
    """Return the FROZEN hand-to-camera extrinsic.

    This is the same formula as the validated shadow-capture runtime.  The
    configured eye/target are authored in the Plug mating frame; the camera
    is calibrated once from the nominal hand-to-plug transform.  No live
    object pose or ideal look-at pose is used.
    """
    camera_model = fixed_camera_model(
        eye=tuple(float(value) for value in eye_plug_m),
        target=tuple(float(value) for value in target_plug_m),
        resolution=tuple(int(value) for value in resolution),
    )
    rows = np.asarray(camera_model.world_to_camera, dtype=np.float64)
    camera_in_plug = np.eye(4, dtype=np.float64)
    camera_in_plug[:3, :3] = rows.T
    camera_in_plug[:3, 3] = np.asarray(
        camera_model.position_world, dtype=np.float64
    )
    return np.asarray(nominal_hand_to_plug, dtype=np.float64) @ camera_in_plug


def wrist_camera_world_pose(
    arm_q: Sequence[float],
    *,
    tcp_from_handbase: np.ndarray,
    tcp_to_camera: np.ndarray,
) -> np.ndarray:
    tcp = np.asarray(iiwa14_grasp_tcp_transform(tuple(float(v) for v in arm_q)))
    hand = tcp @ np.asarray(tcp_from_handbase, dtype=np.float64)
    return hand @ np.asarray(tcp_to_camera, dtype=np.float64)


def _visible_projection(camera: CameraModel, cad, world_pose: np.ndarray, label: int):
    points = cad.xyz[cad.label == label]
    world_points = (
        world_pose[:3, :3] @ points.T
    ).T + world_pose[:3, 3]
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


def _finite_difference_jacobian(
    cameras: Sequence[CameraModel],
    receptacle_cad,
    t_wp_nominal: np.ndarray,
    t_rp_nominal: np.ndarray,
    *,
    label=RECEPTACLE_MATING,
    sample_every: int = 8,
):
    points = receptacle_cad.xyz[receptacle_cad.label == label][::sample_every]
    if len(points) < 6:
        return None
    epsilons = (
        0.0001,
        0.0001,
        0.0001,
        0.002,
        0.002,
        0.002,
    )
    rows = []
    for camera in cameras:
        base_pose = t_wp_nominal @ np.linalg.inv(t_rp_nominal)
        base_points = (base_pose[:3, :3] @ points.T).T + base_pose[:3, 3]
        base_uv, _ = project(camera, base_points)
        base_flat = np.asarray(base_uv, dtype=np.float64).ravel()
        columns = []
        for index in range(6):
            delta = np.zeros(6, dtype=np.float64)
            delta[index] = epsilons[index]
            perturbed = pose_matrix(delta) @ t_rp_nominal
            world_pose = t_wp_nominal @ np.linalg.inv(perturbed)
            world_points = (world_pose[:3, :3] @ points.T).T + world_pose[:3, 3]
            uv, _ = project(camera, world_points)
            columns.append(np.asarray(uv, dtype=np.float64).ravel() - base_flat)
        rows.append(np.column_stack(columns) / np.asarray(epsilons))
    return np.vstack(rows)


def normalize_observability_jacobian(
    jacobian: np.ndarray,
    *,
    position_scale_m: float = 1.0e-3,
    angle_scale_rad: float = 1.0e-3,
    pixel_scale_px: float = 1.0,
) -> np.ndarray:
    """Column-normalize pixels/m and pixels/rad into comparable units.

    Residuals are pixels; the state order is [x,y,z,rx,ry,rz] with x/y/z in
    metres and rx/ry/rz in radians.  Each column is multiplied by its
    parameter scale and divided by the pixel scale.  The result is invariant
    when the same physical problem is re-expressed in mm (position columns
    shrink by 0.001 and position_scale becomes 1 mm, so their product is
    unchanged).
    """
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 6:
        raise ValueError("jacobian must have six columns")
    if min(position_scale_m, angle_scale_rad, pixel_scale_px) <= 0.0:
        raise ValueError("scales must be positive")
    normalized = matrix.copy()
    normalized[:, :3] *= float(position_scale_m) / float(pixel_scale_px)
    normalized[:, 3:] *= float(angle_scale_rad) / float(pixel_scale_px)
    return normalized


def _condition_from_jacobian(jacobian: np.ndarray | None):
    if jacobian is None or not np.all(np.isfinite(jacobian)):
        return None
    normalized = normalize_observability_jacobian(jacobian)
    full_values = np.linalg.svd(normalized, compute_uv=False)
    smax_full = float(np.max(full_values))
    smin_full = float(np.min(full_values))

    jacobian_5d = normalized[:, :5]
    values_5d = np.linalg.svd(jacobian_5d, compute_uv=False)
    smax_5d = float(np.max(values_5d))
    smin_5d = float(np.min(values_5d))
    observable_5d = bool(smin_5d > 1.0e-9 * smax_5d)
    return {
        "parameterization": {
            "order": ["x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad"],
            "perturbation": "left_multiply_SE3_local_xyz_rpy",
        },
        "scales": {
            "position_scale_m": 1.0e-3,
            "angle_scale_rad": 1.0e-3,
            "pixel_scale_px": 1.0,
        },
        "rank_6d": int(np.sum(full_values > 1.0e-9 * smax_full)),
        "singular_values_6d_normalized": [
            float(value) for value in full_values
        ],
        "condition_6d": (
            math.inf
            if smin_full <= 1.0e-12
            else float(smax_full / smin_full)
        ),
        "singular_values_5d_normalized": [
            float(value) for value in values_5d
        ],
        "condition_5d": (
            math.inf
            if smin_5d <= 1.0e-12
            else float(smax_5d / smin_5d)
        ),
        "observable_5d": observable_5d,
        "rz_sensitivity": float(
            np.linalg.norm(normalized[:, 5])
            / max(np.linalg.norm(normalized[:, 0]), 1.0e-12)
        ),
        "rz_excluded_from_5d": True,
    }


def design_wrist_receptacle_views(
    *,
    base_arm_q: Sequence[float],
    solve_arm,
    tcp_from_handbase: np.ndarray,
    nominal_hand_to_plug: np.ndarray,
    tcp_to_camera: np.ndarray,
    receptacle_world: np.ndarray,
    joint_limits: Sequence[tuple[float, float]],
    plans: Sequence[WristViewPlan] = DEFAULT_WRIST_RECEPTACLE_PLANS,
    extrinsic_source: dict[str, Any] | None = None,
    minimum_pairwise_angle_deg: float = 15.0,
    minimum_short_axis_px: float = 80.0,
    minimum_pixels: int = 100,
    maximum_condition_5d: float = 1.0e6,
) -> dict[str, Any]:
    """Design and screen wrist-camera views of the fixed Receptacle."""
    base_q = np.asarray(base_arm_q, dtype=np.float64).ravel()
    if base_q.shape != (7,):
        raise ValueError("base_arm_q must be a 7-vector")
    base_violations = joint_target_limit_violations(
        base_q, joint_limits, margin_rad=JOINT_LIMIT_MARGIN_RAD
    )
    if base_violations:
        raise ValueError("base arm q violates joint limit margin")

    _, receptacle_cad = proxy_cad_points()
    t_wp_nominal = plug_world_pose(
        base_q,
        tcp_from_handbase=tcp_from_handbase,
        nominal_hand_to_plug=nominal_hand_to_plug,
    )
    t_rp_nominal = np.linalg.inv(receptacle_world) @ t_wp_nominal

    views = []
    for plan in plans:
        delta = np.asarray(plan.tcp_delta_pose6, dtype=np.float64).ravel()
        if delta.shape != (6,):
            raise ValueError(f"{plan.view_id} delta must have 6 values")
        base_tcp = np.asarray(
            iiwa14_grasp_tcp_transform(tuple(float(v) for v in base_q))
        )
        desired_tcp = base_tcp @ pose_matrix(delta)
        arm_q = np.asarray(
            solve_arm(
                tuple(float(v) for v in base_q),
                tuple(desired_tcp[:3, 3]),
                target_rotation=desired_tcp[:3, :3],
                maximum_iterations=400,
                damping=1.0e-4,
            ),
            dtype=np.float64,
        )
        limit_violations = joint_target_limit_violations(
            arm_q, joint_limits, margin_rad=JOINT_LIMIT_MARGIN_RAD
        )
        camera_world = wrist_camera_world_pose(
            arm_q,
            tcp_from_handbase=tcp_from_handbase,
            tcp_to_camera=tcp_to_camera,
        )
        camera = _camera_model_from_pose(
            camera_world, resolution=WRIST_CAMERA_RESOLUTION
        )
        projection = _visible_projection(
            camera, receptacle_cad, receptacle_world, RECEPTACLE_MATING
        )
        camera_to_receptacle = (
            np.asarray(receptacle_world[:3, 3]) - camera_world[:3, 3]
        )
        distance = float(np.linalg.norm(camera_to_receptacle))
        views.append(
            {
                "view_id": plan.view_id,
                "arm_q_rad": arm_q.tolist(),
                "joint_limit_violations": limit_violations,
                "camera_world_4x4": camera_world.tolist(),
                "camera_to_receptacle_direction": (
                    camera_to_receptacle / distance
                ).tolist(),
                "camera_distance_m": distance,
                "receptacle_mating_projection": projection,
            }
        )

    valid_views = [
        view
        for view in views
        if not view["joint_limit_violations"]
        and view["receptacle_mating_projection"]["pixels"] >= minimum_pixels
        and view["receptacle_mating_projection"]["short_axis_px"]
        >= minimum_short_axis_px
    ]

    selected = []
    best_pair = None
    for first_index, first_view in enumerate(valid_views):
        for second_view in valid_views[first_index + 1:]:
            first_dir = np.asarray(
                first_view["camera_to_receptacle_direction"]
            )
            second_dir = np.asarray(
                second_view["camera_to_receptacle_direction"]
            )
            angle_deg = float(
                math.degrees(
                    math.acos(
                        max(
                            -1.0,
                            min(1.0, float(np.dot(first_dir, second_dir))),
                        )
                    )
                )
            )
            if angle_deg < minimum_pairwise_angle_deg:
                continue
            min_short_axis = min(
                first_view["receptacle_mating_projection"][
                    "short_axis_px"
                ],
                second_view["receptacle_mating_projection"][
                    "short_axis_px"
                ],
            )
            if best_pair is None or min_short_axis > best_pair[0]:
                best_pair = (min_short_axis, first_view, second_view)
    if best_pair is not None:
        selected = [best_pair[1], best_pair[2]]
        # Optional third view only if it adds another >= minimum angle
        for view in valid_views:
            if view in selected:
                continue
            ok = all(
                float(
                    math.degrees(
                        math.acos(
                            max(
                                -1.0,
                                min(
                                    1.0,
                                    float(
                                        np.dot(
                                            np.asarray(
                                                view[
                                                    "camera_to_receptacle_direction"
                                                ]
                                            ),
                                            np.asarray(
                                                other[
                                                    "camera_to_receptacle_direction"
                                                ]
                                            ),
                                        )
                                    ),
                                )
                            )
                        )
                    )
                )
                >= minimum_pairwise_angle_deg
                for other in selected
            )
            if ok:
                selected.append(view)
            if len(selected) >= 3:
                break

    condition = None
    if selected:
        selected_cameras = [
            _camera_model_from_pose(
                np.asarray(view["camera_world_4x4"]),
                resolution=WRIST_CAMERA_RESOLUTION,
            )
            for view in selected
        ]
        jacobian = _finite_difference_jacobian(
            selected_cameras,
            receptacle_cad,
            t_wp_nominal,
            t_rp_nominal,
        )
        condition = _condition_from_jacobian(jacobian)

    c2_branches = [
        {
            "linked_hypothesis_id": "C2_LINKED_BRANCH_0",
            "t_hp_branch_id": "T_HP_YAW_0",
            "t_rp_branch_id": "T_RP_YAW_0",
            "covariance_5x5_status": "UNVALIDATED",
            "coverage_calibrated": False,
            "visual_authorization": False,
        },
        {
            "linked_hypothesis_id": "C2_LINKED_BRANCH_PI",
            "t_hp_branch_id": "T_HP_YAW_PI",
            "t_rp_branch_id": "T_RP_YAW_PI",
            "covariance_5x5_status": "UNVALIDATED",
            "coverage_calibrated": False,
            "visual_authorization": False,
        },
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "role": "wrist_receptacle_view_design_cpu",
        "formal_estimator_input": False,
        "control_authorized": False,
        "extrinsic_source": extrinsic_source,
        "base_arm_q_rad": base_q.tolist(),
        "t_wp_nominal_4x4": t_wp_nominal.tolist(),
        "t_rp_nominal_4x4": t_rp_nominal.tolist(),
        "views": views,
        "selected_view_ids": [view["view_id"] for view in selected],
        "pairwise_angle_minimum_deg": minimum_pairwise_angle_deg,
        "condition": condition,
        "observability_screen": {
            "selected_count": len(selected),
            "minimum_views_required": 2,
            "passed": bool(
                len(selected) >= 2
                and condition is not None
                and condition["observable_5d"]
                and condition["condition_5d"] <= maximum_condition_5d
            ),
            "maximum_condition_5d": maximum_condition_5d,
        },
        "c2": {
            "retain_hypotheses": 2,
            "averaged": False,
            "cross_product_hypotheses": 0,
            "branches": c2_branches,
        },
        "covariance": {
            "status": "UNVALIDATED",
            "coverage_calibrated": False,
            "visual_authorization": False,
            "control_authorized": False,
        },
    }


__all__ = [
    "DEFAULT_WRIST_RECEPTACLE_PLANS",
    "SCHEMA_VERSION",
    "WristViewPlan",
    "_camera_model_from_pose",
    "_camera_pose_from_look_at",
    "frozen_wrist_camera_hand_transform",
    "normalize_observability_jacobian",
    "design_wrist_receptacle_views",
    "plug_world_pose",
    "receptacle_world_pose",
    "wrist_camera_world_pose",
]
