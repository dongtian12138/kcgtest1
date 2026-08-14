"""Direct in-hand multiview registration of ``T_receptacle_plug``.

The camera-to-plug transform is calibrated and fixed.  Active view motions are
known TCP/FK increments from one reference pose.  A single reference relative
pose is optimized jointly across all observations; no per-view poses are
estimated or averaged.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from kcg_connector.d38999_cad_registration import (
    CadPoints, CameraModel, PLUG_MATING, RECEPTACLE_MATING,
    RECEPTACLE_SHELL_FLANGE, fixed_camera_model, project,
)


def pose_matrix(pose6) -> np.ndarray:
    pose = np.asarray(pose6, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_pose(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate((matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")))


def compose_pose(first, second) -> np.ndarray:
    return matrix_pose(pose_matrix(first) @ pose_matrix(second))


def relative_delta(reference_pose, view_pose) -> np.ndarray:
    return matrix_pose(np.linalg.inv(pose_matrix(reference_pose)) @ pose_matrix(view_pose))


def camera_from_plug_pose(
    plug_pose_receptacle,
    mount_eye_plug,
    mount_target_plug,
    *,
    resolution=(1280, 720),
) -> CameraModel:
    transform = pose_matrix(plug_pose_receptacle)
    eye = transform[:3, :3] @ np.asarray(mount_eye_plug) + transform[:3, 3]
    target = transform[:3, :3] @ np.asarray(mount_target_plug) + transform[:3, 3]
    return fixed_camera_model(eye=eye, target=target, resolution=resolution)


def register_inhand_relative_pose_multiview(
    receptacle_cad: CadPoints,
    observations: Sequence[Mapping[str, np.ndarray]],
    view_deltas_from_reference: Sequence[Sequence[float]],
    initial_reference_pose,
    *,
    mount_eye_plug,
    mount_target_plug,
    yaw_hypothesis_rad: float,
    semantic_mask: bool = True,
    yaw_half_window_rad: float = math.radians(3.0),
) -> dict[str, Any]:
    if len(observations) != len(view_deltas_from_reference) or len(observations) < 2:
        raise ValueError("joint in-hand registration requires matching >=2 observations")
    initial = np.asarray(initial_reference_pose, dtype=np.float64).copy()
    initial[5] = yaw_hypothesis_rad
    mating_selector = receptacle_cad.label == RECEPTACLE_MATING
    xyz = receptacle_cad.xyz[mating_selector]
    normal = receptacle_cad.normal[mating_selector]
    edge = receptacle_cad.edge[mating_selector]
    sample = np.linspace(0, len(xyz) - 1, min(1100, len(xyz)), dtype=np.int64)
    xyz, normal, edge = xyz[sample], normal[sample], edge[sample]
    prepared = []
    for observation in observations:
        label = observation["label"]
        mask = (label == RECEPTACLE_MATING) | ((label == RECEPTACLE_SHELL_FLANGE) if semantic_mask else False)
        if int(np.sum(mask)) < 30:
            return {"success": False, "reject_reason": "RECEPTACLE_MATING_FEATURE_NOT_VISIBLE", "condition_number": math.inf}
        mask_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        observed_edge = cv2.Canny(observation["rgb"], 30, 90)
        observed_edge &= cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) * 255
        edge_distance = cv2.distanceTransform((observed_edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
        prepared.append((observation, mask_distance, edge_distance))

    def residual(parameters):
        chunks = []
        for delta, (observation, mask_distance, edge_distance) in zip(view_deltas_from_reference, prepared):
            view_pose = compose_pose(parameters, delta)
            camera = camera_from_plug_pose(view_pose, mount_eye_plug, mount_target_plug, resolution=(observation["rgb"].shape[1], observation["rgb"].shape[0]))
            uv, predicted_depth = project(camera, xyz)
            u = np.rint(uv[:, 0]).astype(np.int32)
            v = np.rint(uv[:, 1]).astype(np.int32)
            valid = (predicted_depth > 0.03) & (u >= 1) & (u < camera.width - 1) & (v >= 1) & (v < camera.height - 1)
            uc = np.clip(u, 0, camera.width - 1)
            vc = np.clip(v, 0, camera.height - 1)
            map_x = uv[:, 0].astype(np.float32).reshape(-1, 1)
            map_y = uv[:, 1].astype(np.float32).reshape(-1, 1)
            silhouette = cv2.remap(mask_distance.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=60.0).ravel()
            silhouette = np.where(valid, silhouette / 1.5, 30.0)
            edge_indices = np.flatnonzero(edge)
            edge_pixels = cv2.remap(edge_distance.astype(np.float32), map_x[edge_indices], map_y[edge_indices], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=60.0).ravel()
            edge_residual = np.where(valid[edge_indices], edge_pixels / 1.25, 30.0)
            observed_depth = observation["depth"][vc, uc]
            depth_valid = valid & np.isfinite(observed_depth)
            visible = depth_valid & (predicted_depth <= observed_depth + 0.0015)
            depth_residual = np.zeros(len(xyz))
            depth_residual[visible] = (predicted_depth[visible] - observed_depth[visible]) / 0.00075
            occlusion = np.zeros(len(xyz))
            behind = depth_valid & ~visible
            occlusion[behind] = np.minimum(4.0, np.maximum(0.0, predicted_depth[behind] - observed_depth[behind]) / 0.0015)
            observed_normal = observation["normal"][vc, uc]
            normal_valid = visible & (np.linalg.norm(observed_normal, axis=1) > 0.5)
            normal_residual = np.zeros(len(xyz))
            normal_residual[normal_valid] = (1.0 - np.sum(normal[normal_valid] * observed_normal[normal_valid], axis=1)) / 0.08
            chunks.extend((silhouette, edge_residual, depth_residual, normal_residual, occlusion))
        prior_scale = np.asarray((0.004, 0.004, 0.006, math.radians(5), math.radians(5), math.radians(3)))
        chunks.append(0.12 * (parameters - initial) / prior_scale)
        return np.concatenate(chunks)

    lower = initial + np.asarray((-0.006, -0.006, -0.008, -math.radians(7), -math.radians(7), -yaw_half_window_rad))
    upper = initial + np.asarray((0.006, 0.006, 0.008, math.radians(7), math.radians(7), yaw_half_window_rad))
    parameter_scale = np.asarray((0.001, 0.001, 0.001, 0.05, 0.05, 0.05))
    def normalized_residual(normalized):
        return residual(initial + parameter_scale * normalized)
    result = least_squares(normalized_residual, np.zeros(6), bounds=((lower - initial) / parameter_scale, (upper - initial) / parameter_scale), max_nfev=75, loss="huber", f_scale=1.0)
    pose_result = initial + parameter_scale * result.x
    hessian = result.jac.T @ result.jac
    eigenvalues = np.linalg.eigvalsh(hessian)
    condition = math.inf if eigenvalues[0] <= 1.0e-12 else float(eigenvalues[-1] / eigenvalues[0])
    covariance_normalized = np.linalg.pinv(hessian) * (float(np.sum(result.fun ** 2)) / max(1, len(result.fun) - 6))
    covariance = np.diag(parameter_scale) @ covariance_normalized @ np.diag(parameter_scale)
    # Yaw is a discrete C2 hypothesis and may be locally unobservable for the
    # circular mating rim.  Report the five-dimensional conditional posterior
    # separately so that yaw singularity cannot inflate X/Y/Rx/Ry gates.
    residual_variance = float(np.sum(result.fun ** 2)) / max(1, len(result.fun) - 6)
    hessian_5d = hessian[:5, :5]
    eigenvalues_5d = np.linalg.eigvalsh(hessian_5d)
    condition_5d = math.inf if eigenvalues_5d[0] <= 1.0e-12 else float(eigenvalues_5d[-1] / eigenvalues_5d[0])
    covariance_5d_normalized = np.linalg.pinv(hessian_5d) * residual_variance
    covariance_5d = np.diag(parameter_scale[:5]) @ covariance_5d_normalized @ np.diag(parameter_scale[:5])
    return {
        "success": bool(result.success and np.all(np.isfinite(pose_result))),
        "reference_relative_pose_xyz_rpy": pose_result.tolist(),
        "covariance_6x6": covariance.tolist(),
        "hessian_6x6": hessian.tolist(),
        "hessian_coordinates": "normalized_1mm_0p05rad",
        "parameter_scale_xyz_rpy": parameter_scale.tolist(),
        "condition_number": condition,
        "conditional_covariance_xyz_rx_ry_5x5": covariance_5d.tolist(),
        "conditional_condition_number_5d": condition_5d,
        "cost": float(result.cost),
        "function_evaluations": int(result.nfev),
        "view_count": len(observations),
        "joint_single_reference_pose_optimization": True,
        "per_view_pose_then_average": False,
        "camera_extrinsic_optimized": False,
        "object_truth_used": False,
        "c2_yaw_hypothesis_rad": yaw_hypothesis_rad,
        "losses": ["mating_rim_rgb_subpixel_edge", "silhouette_distance_transform", "visible_depth_point_to_plane", "normal", "in_hand_kinematic_prior", "z_buffer_occlusion"],
    }


__all__ = ["camera_from_plug_pose", "compose_pose", "matrix_pose", "pose_matrix", "register_inhand_relative_pose_multiview", "relative_delta"]
