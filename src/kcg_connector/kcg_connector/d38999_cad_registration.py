"""Known-proxy CAD render registration for D38999 mating features.

This CPU implementation is deliberately explicit about part roles.  Plug
body/nose and receptacle entry/rim contribute to the assembly residual;
coupling nut, rear shell, and flange are rendered as occluders only.  Camera
extrinsics are immutable.  The optimized state is one C2 branch of
``T_receptacle_plug``: relative XYZ, Rx/Ry, and a locally bounded Rz.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


PLUG_MATING = 1
PLUG_NUT_BODY = 2
RECEPTACLE_MATING = 3
RECEPTACLE_SHELL_FLANGE = 4


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    position_world: tuple[float, float, float]
    world_to_camera: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class CadPoints:
    xyz: np.ndarray
    normal: np.ndarray
    label: np.ndarray
    edge: np.ndarray


def fixed_camera_model(
    eye=(0.550, -0.850, 0.720),
    target=(0.535, -0.0125, 0.231),
    resolution=(640, 480),
    focal_length_mm=24.0,
    horizontal_aperture_mm=20.955,
) -> CameraModel:
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    width, height = resolution
    fx = focal_length_mm / horizontal_aperture_mm * width
    return CameraModel(
        width, height, fx, fx, 0.5 * (width - 1), 0.5 * (height - 1),
        tuple(eye), tuple(tuple(float(value) for value in row) for row in np.vstack((right, -up, forward))),
    )


def _ring(radius, z, label, count=240, edge=True, normal_z=-1.0):
    angle = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    xyz = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.full(count, z)))
    normal = np.tile((0.0, 0.0, normal_z), (count, 1))
    return xyz, normal, np.full(count, label), np.full(count, edge)


def _cylinder(radius, z0, z1, label, azimuth=240, layers=20):
    angle = np.linspace(0.0, 2.0 * math.pi, azimuth, endpoint=False)
    z = np.linspace(z0, z1, layers)
    aa, zz = np.meshgrid(angle, z)
    xyz = np.column_stack((radius * np.cos(aa.ravel()), radius * np.sin(aa.ravel()), zz.ravel()))
    normal = np.column_stack((np.cos(aa.ravel()), np.sin(aa.ravel()), np.zeros(aa.size)))
    edge = np.isclose(zz.ravel(), z0) | np.isclose(zz.ravel(), z1)
    return xyz, normal, np.full(len(xyz), label), edge


PLUG_REAR_BODY = 5
SHELL25J_CAD_PROFILE_ID = "d38999_shell25j_c2_visible_cad_v1"


@dataclass(frozen=True)
class Shell25JCadProfile:
    plug_mating: CadPoints
    plug_occluders: CadPoints
    receptacle: CadPoints
    metadata: dict[str, Any]


def shell25j_contact_positions(count=61):
    """Deterministic 1+6+12+18+24 layout copied from the bound generator."""
    if count != 61:
        raise ValueError("Shell25J visual contact layout requires 61 contacts")
    positions = [(0.0, 0.0)]
    for ring_index, ring_count in enumerate((6, 12, 18, 24), start=1):
        radius_fraction = 0.20 * ring_index
        angular_offset = 0.5 * math.pi / ring_count
        for index in range(ring_count):
            angle = 2.0 * math.pi * index / ring_count + angular_offset
            positions.append(
                (
                    radius_fraction * math.cos(angle),
                    radius_fraction * math.sin(angle),
                )
            )
    return tuple(positions)


def _box_points(center, size, rotation_z):
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    rotation = Rotation.from_euler("z", rotation_z).as_matrix()
    half = 0.5 * size
    x = np.linspace(-half[0], half[0], 4)
    y = np.linspace(-half[1], half[1], 4)
    z = np.linspace(-half[2], half[2], 4)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    local = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    world = local @ rotation.T + center
    normals = []
    edges = []
    for point, lx, ly, lz in zip(local, xx.ravel(), yy.ravel(), zz.ravel()):
        absx, absy, absz = abs(lx), abs(ly), abs(lz)
        dominant = max((absx, 0), (absy, 1), (absz, 2))[1]
        normal = np.zeros(3)
        if dominant == 0:
            normal[0] = np.sign(lx) if lx != 0 else 0.0
        elif dominant == 1:
            normal[1] = np.sign(ly) if ly != 0 else 0.0
        else:
            normal[2] = np.sign(lz) if lz != 0 else 0.0
        normals.append(normal @ rotation.T)
        edges.append(bool(absx == half[0] or absy == half[1] or absz == half[2]))
    return world, np.asarray(normals), np.asarray(edges)


def _disk_points(radius, z, count_r=3, count_theta=32):
    xyz = [(0.0, 0.0, z)]
    normals = [(0.0, 0.0, 1.0)]
    edges = [False]
    for ring in range(1, count_r + 1):
        r = radius * ring / count_r
        for index in range(count_theta):
            angle = 2.0 * math.pi * index / count_theta
            xyz.append((r * math.cos(angle), r * math.sin(angle), z))
            normals.append((0.0, 0.0, 1.0))
            edges.append(ring == count_r)
    return np.asarray(xyz), np.asarray(normals), np.asarray(edges)


def _load_shell25j_profile_geometry() -> tuple[Any, dict[str, Any]]:
    from kcg_connector.d38999_proxy import load_d38999_shell25j_proxy

    package_root = Path(__file__).resolve().parents[1]
    config_path = package_root / "config/d38999_shell25j_proxy_v1.yaml"
    config = load_d38999_shell25j_proxy(config_path)
    metadata = {
        "profile_id": SHELL25J_CAD_PROFILE_ID,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "asset_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[3]
                / "artifacts/kcg_connector/isaac"
                / "d38999_shell25j_61_pair_proxy_v1.usda"
            ).read_bytes()
        ).hexdigest(),
        "symmetry_order": 2,
        "mating_shell_segment_count": 20,
        "socket_count": 61,
    }
    return config, metadata


def _cylinder_positive_points(radius, z0, z1, label, azimuth=96, layers=4):
    angle = np.linspace(0.0, 2.0 * math.pi, azimuth, endpoint=False)
    z = np.linspace(z0, z1, layers)
    aa, zz = np.meshgrid(angle, z)
    xyz = np.column_stack((radius * np.cos(aa.ravel()), radius * np.sin(aa.ravel()), zz.ravel()))
    normal = np.column_stack((np.cos(aa.ravel()), np.sin(aa.ravel()), np.zeros(aa.size)))
    edge = np.isclose(zz.ravel(), z0) | np.isclose(zz.ravel(), z1)
    return xyz, normal, np.full(len(xyz), label), edge


def shell25j_plug_cad_profile(
    feature_set: str = "shell_plus_socket",
) -> Shell25JCadProfile:
    """Shell25J plug profile from the hash-bound proxy YAML loader.

    All geometry values are read through ``load_d38999_shell25j_proxy`` and
    therefore change with the bound config rather than silent local constants.
    Positive-Z mating geometry and positive-Z nut/rear-body occluders follow
    ``create_d38999_proxy_asset.py``.
    """
    config, metadata = _load_shell25j_profile_geometry()
    plug = config.plug_geometry_m
    if feature_set not in {"shell_only", "socket_only", "shell_plus_socket"}:
        raise ValueError(f"unknown feature_set: {feature_set}")
    parts = []
    if feature_set in {"shell_only", "shell_plus_socket"}:
        segment_radius = 0.5 * (
            plug.mating_shell_inner_radius + plug.mating_shell_outer_radius
        )
        radial_size = plug.mating_shell_outer_radius - plug.mating_shell_inner_radius
        tangential_size = 0.91 * 2.0 * math.pi * segment_radius / 20.0
        for index in range(20):
            angle = 2.0 * math.pi * index / 20.0
            xyz, normal, edge = _box_points(
                (
                    segment_radius * math.cos(angle),
                    segment_radius * math.sin(angle),
                    0.5 * plug.mating_shell_length,
                ),
                (radial_size, tangential_size, plug.mating_shell_length),
                angle,
            )
            parts.append((xyz, normal, np.full(len(xyz), PLUG_MATING), edge))
    if feature_set in {"socket_only", "shell_plus_socket"}:
        face_xyz, face_normal, face_edge = _disk_points(
            plug.contact_face_radius, 0.001, 2, 32
        )
        parts.append((face_xyz, face_normal, np.full(len(face_xyz), PLUG_MATING), face_edge))
        contact_scale = 0.90 * plug.contact_face_radius
        for x_fraction, y_fraction in shell25j_contact_positions(61):
            cx = contact_scale * x_fraction
            cy = contact_scale * y_fraction
            z = 0.00115
            theta = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
            xyz = np.column_stack((cx + plug.contact_visual_radius * np.cos(theta), cy + plug.contact_visual_radius * np.sin(theta), np.full(len(theta), z)))
            normal = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(len(theta))))
            parts.append((xyz, normal, np.full(len(xyz), PLUG_MATING), np.ones(len(xyz), dtype=bool)))

    def combine(part_list):
        return CadPoints(
            np.concatenate([item[0] for item in part_list], axis=0),
            np.concatenate([item[1] for item in part_list], axis=0),
            np.concatenate([item[2] for item in part_list]),
            np.concatenate([item[3] for item in part_list]),
        )

    occluder_parts = []
    nut_radius = 0.5 * (
        plug.coupling_nut_inner_radius + plug.coupling_nut_outer_radius
    )
    nut_radial = plug.coupling_nut_outer_radius - plug.coupling_nut_inner_radius
    nut_tangential = 0.91 * 2.0 * math.pi * nut_radius / plug.grip_segment_count
    for index in range(plug.grip_segment_count):
        angle = 2.0 * math.pi * index / plug.grip_segment_count
        xyz, normal, edge = _box_points(
            (
                nut_radius * math.cos(angle),
                nut_radius * math.sin(angle),
                0.5 * plug.overall_length,
            ),
            (nut_radial, nut_tangential, plug.coupling_nut_length),
            angle,
        )
        occluder_parts.append((xyz, normal, np.full(len(xyz), PLUG_NUT_BODY), edge))
    xyz, normal, label, edge = _cylinder_positive_points(
        plug.rear_body_radius,
        plug.overall_length - plug.rear_body_length,
        plug.overall_length,
        PLUG_REAR_BODY,
    )
    occluder_parts.append((xyz, normal, label, edge))
    _, receptacle_cad = proxy_cad_points()
    return Shell25JCadProfile(
        plug_mating=combine(parts),
        plug_occluders=combine(occluder_parts),
        receptacle=receptacle_cad,
        metadata=metadata,
    )


def shell25j_cad_profile_metadata() -> dict[str, Any]:
    return shell25j_plug_cad_profile().metadata


def _legacy_axisymmetric_plug_profile(count=2000) -> CadPoints:
    xyz, normal, label, edge = _ring(
        0.01775, 0.005, PLUG_MATING, count=count, normal_z=-1.0
    )
    return CadPoints(xyz, normal, label, edge)


def proxy_cad_points() -> tuple[CadPoints, CadPoints]:
    plug_parts = [
        _ring(0.0182, 0.0, PLUG_MATING),
        _ring(0.0165, 0.0, PLUG_MATING),
        _cylinder(0.0188, -0.012, -0.002, PLUG_MATING),
        _cylinder(0.02215, -0.026, -0.012, PLUG_NUT_BODY),
        _cylinder(0.0240, -0.029, -0.012, PLUG_NUT_BODY),
    ]
    receptacle_parts = [
        _ring(0.0203, 0.0, RECEPTACLE_MATING, normal_z=-1.0),
        _ring(0.0215, 0.0, RECEPTACLE_MATING, normal_z=-1.0),
        _cylinder(0.0215, 0.0, 0.010, RECEPTACLE_MATING),
        _cylinder(0.0215, 0.010, 0.0315, RECEPTACLE_SHELL_FLANGE),
    ]
    # Square flange edge/occluder at z=10 mm.
    t = np.linspace(-0.023, 0.023, 180)
    flange = np.vstack((
        np.column_stack((t, np.full_like(t, -0.023), np.full_like(t, 0.010))),
        np.column_stack((t, np.full_like(t, 0.023), np.full_like(t, 0.010))),
        np.column_stack((np.full_like(t, -0.023), t, np.full_like(t, 0.010))),
        np.column_stack((np.full_like(t, 0.023), t, np.full_like(t, 0.010))),
    ))
    receptacle_parts.append((flange, np.tile((0.0, 0.0, -1.0), (len(flange), 1)), np.full(len(flange), RECEPTACLE_SHELL_FLANGE), np.ones(len(flange), dtype=bool)))

    def combine(parts):
        values = [np.concatenate([item[index] for item in parts], axis=0) for index in range(4)]
        return CadPoints(values[0], values[1], values[2].astype(np.int16), values[3].astype(bool))
    return combine(plug_parts), combine(receptacle_parts)


def transform_points(cad: CadPoints, pose6) -> CadPoints:
    pose = np.asarray(pose6, dtype=np.float64)
    # Pose angular coordinates are explicit assembly-frame roll/pitch/yaw.
    # A rotation-vector parameterization cannot represent the C2 branch as
    # ``small roll, small pitch, yaw + pi`` without coupling all three axes.
    rotation = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    return CadPoints(cad.xyz @ rotation.T + pose[:3], cad.normal @ rotation.T, cad.label, cad.edge)


def project(camera: CameraModel, xyz_world):
    xyz = np.asarray(xyz_world, dtype=np.float64)
    camera_xyz = (xyz - np.asarray(camera.position_world)) @ np.asarray(camera.world_to_camera).T
    depth = camera_xyz[:, 2]
    uv = np.column_stack((camera.fx * camera_xyz[:, 0] / depth + camera.cx, camera.fy * camera_xyz[:, 1] / depth + camera.cy))
    return uv, depth


def render_points(camera: CameraModel, parts: tuple[CadPoints, ...], *, depth_noise_std_m=0.0, seed=0):
    xyz = np.concatenate([item.xyz for item in parts])
    normal = np.concatenate([item.normal for item in parts])
    label = np.concatenate([item.label for item in parts])
    edge = np.concatenate([item.edge for item in parts])
    uv, depth = project(camera, xyz)
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    valid = (depth > 0.1) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    indices = np.flatnonzero(valid)
    order = indices[np.argsort(depth[indices])[::-1]]
    depth_image = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
    label_image = np.zeros((camera.height, camera.width), dtype=np.int16)
    normal_image = np.zeros((camera.height, camera.width, 3), dtype=np.float32)
    edge_image = np.zeros((camera.height, camera.width), dtype=np.uint8)
    for index in order:
        depth_image[v[index], u[index]] = depth[index]
        label_image[v[index], u[index]] = label[index]
        normal_image[v[index], u[index]] = normal[index]
        if edge[index]:
            edge_image[v[index], u[index]] = 255
    # Close sampling holes without moving subpixel edge locations materially.
    kernels = np.ones((3, 3), np.uint8)
    for value in (PLUG_MATING, PLUG_NUT_BODY, RECEPTACLE_MATING, RECEPTACLE_SHELL_FLANGE):
        mask = (label_image == value).astype(np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernels)
        label_image[(closed > 0) & (label_image == 0)] = value
    if depth_noise_std_m > 0.0:
        finite = np.isfinite(depth_image)
        depth_image[finite] += np.random.default_rng(seed).normal(0.0, depth_noise_std_m, int(np.sum(finite))).astype(np.float32)
    rgb = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    colors = {PLUG_MATING: (225, 140, 35), PLUG_NUT_BODY: (120, 125, 135), RECEPTACLE_MATING: (40, 145, 225), RECEPTACLE_SHELL_FLANGE: (55, 75, 105)}
    for value, color in colors.items():
        rgb[label_image == value] = color
    return {"rgb": rgb, "depth": depth_image, "label": label_image, "normal": normal_image, "edge": edge_image}


def _mask_for_mode(observation, mode):
    label = observation["label"]
    if mode == "ideal_part":
        return label == PLUG_MATING
    if mode in {"semantic_ideal_depth", "semantic_noisy_depth"}:
        return (label == PLUG_MATING) | (label == PLUG_NUT_BODY)
    raise ValueError(mode)


def register_relative_pose(
    camera: CameraModel,
    plug_cad: CadPoints,
    receptacle_pose_world,
    observation: Mapping[str, np.ndarray],
    initial_relative_pose,
    *,
    mode: str,
    yaw_hypothesis_rad: float,
    yaw_half_window_rad: float = math.radians(3.0),
) -> dict[str, Any]:
    """Optimize one C2 branch without changing camera extrinsics."""
    mask = _mask_for_mode(observation, mode)
    if int(np.sum(mask)) < 20:
        return {"success": False, "reject_reason": "MATING_FEATURE_NOT_VISIBLE", "condition_number": math.inf}
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    observed_edge = cv2.Canny(observation["rgb"], 30, 90)
    observed_edge &= cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) * 255
    edge_distance = cv2.distanceTransform((observed_edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    depth_image = observation["depth"]
    normal_image = observation["normal"]
    rows, columns = np.nonzero(mask & np.isfinite(depth_image))
    if len(rows) > 2500:
        keep = np.linspace(0, len(rows) - 1, 2500, dtype=np.int64)
        rows, columns = rows[keep], columns[keep]
    observed_depth_values = depth_image[rows, columns].astype(np.float64)
    camera_points = np.column_stack((
        (columns - camera.cx) * observed_depth_values / camera.fx,
        (rows - camera.cy) * observed_depth_values / camera.fy,
        observed_depth_values,
    ))
    observed_world = camera_points @ np.asarray(camera.world_to_camera) + np.asarray(camera.position_world)
    observed_tree = cKDTree(observed_world)
    receptacle = np.asarray(receptacle_pose_world, dtype=np.float64)
    receptacle_rotation = Rotation.from_euler("xyz", receptacle[3:]).as_matrix()
    initial = np.asarray(initial_relative_pose, dtype=np.float64).copy()
    initial[5] = yaw_hypothesis_rad
    mating = CadPoints(
        plug_cad.xyz[plug_cad.label == PLUG_MATING],
        plug_cad.normal[plug_cad.label == PLUG_MATING],
        plug_cad.label[plug_cad.label == PLUG_MATING],
        plug_cad.edge[plug_cad.label == PLUG_MATING],
    )
    sample = np.linspace(0, len(mating.xyz) - 1, min(900, len(mating.xyz)), dtype=np.int64)
    xyz_local = mating.xyz[sample]
    normal_local = mating.normal[sample]
    edge_local = mating.edge[sample]

    def relative_to_world(parameters):
        rel_rotation = Rotation.from_euler("xyz", parameters[3:]).as_matrix()
        xyz_receptacle = xyz_local @ rel_rotation.T + parameters[:3]
        normal_receptacle = normal_local @ rel_rotation.T
        return xyz_receptacle @ receptacle_rotation.T + receptacle[:3], normal_receptacle @ receptacle_rotation.T

    def residual(parameters):
        xyz_world, normal_world = relative_to_world(parameters)
        uv, predicted_depth = project(camera, xyz_world)
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        valid = (predicted_depth > 0.1) & (u >= 1) & (u < camera.width - 1) & (v >= 1) & (v < camera.height - 1)
        uc = np.clip(u, 0, camera.width - 1)
        vc = np.clip(v, 0, camera.height - 1)
        map_x = uv[:, 0].astype(np.float32).reshape(-1, 1)
        map_y = uv[:, 1].astype(np.float32).reshape(-1, 1)
        sampled_distance = cv2.remap(
            distance.astype(np.float32), map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=50.0,
        ).ravel()
        silhouette_residual = np.where(valid, sampled_distance / 2.0, 25.0)
        edge_indices = np.flatnonzero(edge_local)
        edge_residual = np.full(len(edge_indices), 25.0)
        edge_valid = valid[edge_indices]
        sampled_edge_distance = cv2.remap(
            edge_distance.astype(np.float32), map_x[edge_indices],
            map_y[edge_indices], cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=50.0,
        ).ravel()
        edge_residual[edge_valid] = sampled_edge_distance[edge_valid] / 1.5
        observed_depth = depth_image[vc, uc]
        depth_valid = valid & np.isfinite(observed_depth)
        depth_residual = np.full(len(u), 8.0)
        depth_residual[depth_valid] = (predicted_depth[depth_valid] - observed_depth[depth_valid]) / 0.0010
        observed_normal = normal_image[vc, uc]
        normal_valid = valid & (np.linalg.norm(observed_normal, axis=1) > 0.5)
        normal_residual = np.full(len(u), 2.0)
        normal_residual[normal_valid] = (1.0 - np.sum(normal_world[normal_valid] * observed_normal[normal_valid], axis=1)) / 0.10
        nearest_distance, nearest_index = observed_tree.query(xyz_world, k=1)
        nearest_vector = xyz_world - observed_world[nearest_index]
        # Vector ICP is the visible depth point-to-plane/point component.  It
        # is evaluated only on mating CAD samples; semantic clutter in B/C can
        # therefore degrade conditioning but cannot become the assembly model.
        icp_residual = nearest_vector.ravel() / 0.0010
        # Registered in-hand/FK prior is deliberately finite, not a pose truth clamp.
        prior_scale = np.asarray((0.004, 0.004, 0.008, math.radians(4), math.radians(4), math.radians(3)))
        prior_residual = (parameters - initial) / prior_scale
        return np.concatenate((silhouette_residual, edge_residual, depth_residual, normal_residual, icp_residual, 0.20 * prior_residual))

    # The semantic-depth/PCA initializer observes the visible surface, not the
    # hidden mating center.  At the oblique fixed camera its deterministic
    # center bias can exceed 10 mm, so the offline observability diagnostic
    # must not be clipped by a 6 mm implementation window.  Yaw remains local
    # to each explicit C2 branch.
    lower = initial + np.asarray((-0.020, -0.020, -0.015, -math.radians(8), -math.radians(8), -yaw_half_window_rad))
    upper = initial + np.asarray((0.020, 0.020, 0.015, math.radians(8), math.radians(8), yaw_half_window_rad))
    parameter_scale = np.asarray((0.001, 0.001, 0.001, 0.05, 0.05, 0.05))
    def normalized_residual(normalized):
        return residual(initial + parameter_scale * normalized)
    result = least_squares(normalized_residual, np.zeros(6), bounds=((lower - initial) / parameter_scale, (upper - initial) / parameter_scale), max_nfev=60, loss="huber", f_scale=1.0)
    pose_result = initial + parameter_scale * result.x
    hessian = result.jac.T @ result.jac
    eigenvalues = np.linalg.eigvalsh(hessian)
    condition = math.inf if eigenvalues[0] <= 1.0e-12 else float(eigenvalues[-1] / eigenvalues[0])
    covariance_normalized = np.linalg.pinv(hessian) * (float(np.sum(result.fun ** 2)) / max(1, len(result.fun) - 6))
    covariance = np.diag(parameter_scale) @ covariance_normalized @ np.diag(parameter_scale)
    return {
        "success": bool(result.success and np.all(np.isfinite(pose_result))),
        "relative_pose_xyz_rpy": pose_result.tolist(),
        # Compatibility alias for artifacts produced before the C2
        # parameterization correction.  Values are now RPY, as stated above.
        "relative_pose_xyz_rotvec": pose_result.tolist(),
        "covariance_6x6": covariance.tolist(),
        "hessian_6x6": hessian.tolist(),
        "hessian_coordinates": "normalized_1mm_0p05rad",
        "parameter_scale_xyz_rpy": parameter_scale.tolist(),
        "condition_number": condition,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "function_evaluations": int(result.nfev),
        "losses": ["mating_rim_rgb_subpixel_edge", "silhouette_distance_transform", "visible_depth_point_to_plane", "normal", "in_hand_kinematic_prior", "z_buffer_occlusion"],
        "camera_extrinsic_optimized": False,
        "whole_mask_centroid_used": False,
        "c2_yaw_hypothesis_rad": yaw_hypothesis_rad,
    }


def register_relative_pose_multiview(
    cameras: tuple[CameraModel, ...] | list[CameraModel],
    plug_cad: CadPoints,
    receptacle_pose_world,
    observations: tuple[Mapping[str, np.ndarray], ...] | list[Mapping[str, np.ndarray]],
    initial_relative_pose,
    *,
    mode: str,
    yaw_hypothesis_rad: float,
    yaw_half_window_rad: float = math.radians(3.0),
) -> dict[str, Any]:
    """Jointly optimize one relative pose against all calibrated views.

    This is one least-squares problem and one Hessian.  It intentionally does
    not estimate a pose per view, and camera extrinsics are immutable.
    """
    if len(cameras) != len(observations) or len(cameras) < 2:
        raise ValueError("multiview registration requires matching >=2 views")
    receptacle = np.asarray(receptacle_pose_world, dtype=np.float64)
    receptacle_rotation = Rotation.from_euler("xyz", receptacle[3:]).as_matrix()
    initial = np.asarray(initial_relative_pose, dtype=np.float64).copy()
    initial[5] = yaw_hypothesis_rad
    mating_mask = plug_cad.label == PLUG_MATING
    mating = CadPoints(
        plug_cad.xyz[mating_mask], plug_cad.normal[mating_mask],
        plug_cad.label[mating_mask], plug_cad.edge[mating_mask],
    )
    sample = np.linspace(0, len(mating.xyz) - 1, min(900, len(mating.xyz)), dtype=np.int64)
    xyz_local = mating.xyz[sample]
    normal_local = mating.normal[sample]
    edge_local = mating.edge[sample]
    view_data = []
    for camera, observation in zip(cameras, observations):
        mask = _mask_for_mode(observation, mode)
        if int(np.sum(mask)) < 20:
            return {"success": False, "reject_reason": "MATING_FEATURE_NOT_VISIBLE", "condition_number": math.inf}
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        observed_edge = cv2.Canny(observation["rgb"], 30, 90)
        observed_edge &= cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) * 255
        edge_distance = cv2.distanceTransform((observed_edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
        rows, columns = np.nonzero(mask & np.isfinite(observation["depth"]))
        if len(rows) > 2500:
            keep = np.linspace(0, len(rows) - 1, 2500, dtype=np.int64)
            rows, columns = rows[keep], columns[keep]
        depth_values = observation["depth"][rows, columns].astype(np.float64)
        camera_points = np.column_stack((
            (columns - camera.cx) * depth_values / camera.fx,
            (rows - camera.cy) * depth_values / camera.fy,
            depth_values,
        ))
        observed_world = camera_points @ np.asarray(camera.world_to_camera) + np.asarray(camera.position_world)
        view_data.append((camera, observation, distance, edge_distance, observed_world, cKDTree(observed_world)))

    def relative_to_world(parameters):
        rel_rotation = Rotation.from_euler("xyz", parameters[3:]).as_matrix()
        xyz_receptacle = xyz_local @ rel_rotation.T + parameters[:3]
        normal_receptacle = normal_local @ rel_rotation.T
        return xyz_receptacle @ receptacle_rotation.T + receptacle[:3], normal_receptacle @ receptacle_rotation.T

    def residual(parameters):
        xyz_world, normal_world = relative_to_world(parameters)
        chunks = []
        for camera, observation, distance, edge_distance, observed_world, observed_tree in view_data:
            uv, predicted_depth = project(camera, xyz_world)
            u = np.rint(uv[:, 0]).astype(np.int32)
            v = np.rint(uv[:, 1]).astype(np.int32)
            valid = (predicted_depth > 0.1) & (u >= 1) & (u < camera.width - 1) & (v >= 1) & (v < camera.height - 1)
            uc = np.clip(u, 0, camera.width - 1)
            vc = np.clip(v, 0, camera.height - 1)
            map_x = uv[:, 0].astype(np.float32).reshape(-1, 1)
            map_y = uv[:, 1].astype(np.float32).reshape(-1, 1)
            sampled_distance = cv2.remap(distance.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=50.0).ravel()
            silhouette = np.where(valid, sampled_distance / 2.0, 25.0)
            edge_indices = np.flatnonzero(edge_local)
            edge_sample = cv2.remap(edge_distance.astype(np.float32), map_x[edge_indices], map_y[edge_indices], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=50.0).ravel()
            edge_residual = np.where(valid[edge_indices], edge_sample / 1.5, 25.0)
            observed_depth = observation["depth"][vc, uc]
            depth_valid = valid & np.isfinite(observed_depth)
            depth_residual = np.full(len(u), 8.0)
            depth_residual[depth_valid] = (predicted_depth[depth_valid] - observed_depth[depth_valid]) / 0.0010
            # Samples deeper than the measured z-buffer are occluded; their
            # RGB/depth residuals are down-weighted instead of pulling the pose
            # toward the occluding nut, fingers, or receptacle shell.
            visible = depth_valid & (predicted_depth <= observed_depth + 0.0015)
            occlusion_residual = np.zeros(len(u))
            occlusion_residual[depth_valid & ~visible] = np.minimum(4.0, (predicted_depth[depth_valid & ~visible] - observed_depth[depth_valid & ~visible]) / 0.0015)
            observed_normal = observation["normal"][vc, uc]
            normal_valid = visible & (np.linalg.norm(observed_normal, axis=1) > 0.5)
            normal_residual = np.zeros(len(u))
            normal_residual[normal_valid] = (1.0 - np.sum(normal_world[normal_valid] * observed_normal[normal_valid], axis=1)) / 0.10
            _, nearest_index = observed_tree.query(xyz_world, k=1)
            icp = (xyz_world - observed_world[nearest_index]).ravel() / 0.0010
            chunks.extend((silhouette, edge_residual, depth_residual, normal_residual, occlusion_residual, icp))
        prior_scale = np.asarray((0.004, 0.004, 0.008, math.radians(4), math.radians(4), math.radians(3)))
        chunks.append(0.20 * (parameters - initial) / prior_scale)
        return np.concatenate(chunks)

    lower = initial + np.asarray((-0.006, -0.006, -0.010, -math.radians(6), -math.radians(6), -yaw_half_window_rad))
    upper = initial + np.asarray((0.006, 0.006, 0.010, math.radians(6), math.radians(6), yaw_half_window_rad))
    parameter_scale = np.asarray((0.001, 0.001, 0.001, 0.05, 0.05, 0.05))
    def normalized_residual(normalized):
        return residual(initial + parameter_scale * normalized)
    result = least_squares(normalized_residual, np.zeros(6), bounds=((lower - initial) / parameter_scale, (upper - initial) / parameter_scale), max_nfev=70, loss="huber", f_scale=1.0)
    pose_result = initial + parameter_scale * result.x
    hessian = result.jac.T @ result.jac
    eigenvalues = np.linalg.eigvalsh(hessian)
    condition = math.inf if eigenvalues[0] <= 1.0e-12 else float(eigenvalues[-1] / eigenvalues[0])
    covariance_normalized = np.linalg.pinv(hessian) * (float(np.sum(result.fun ** 2)) / max(1, len(result.fun) - 6))
    covariance = np.diag(parameter_scale) @ covariance_normalized @ np.diag(parameter_scale)
    return {
        "success": bool(result.success and np.all(np.isfinite(pose_result))),
        "relative_pose_xyz_rpy": pose_result.tolist(),
        "relative_pose_xyz_rotvec": pose_result.tolist(),
        "covariance_6x6": covariance.tolist(),
        "hessian_6x6": hessian.tolist(),
        "hessian_coordinates": "normalized_1mm_0p05rad",
        "parameter_scale_xyz_rpy": parameter_scale.tolist(),
        "condition_number": condition,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "function_evaluations": int(result.nfev),
        "view_count": len(cameras),
        "joint_single_pose_optimization": True,
        "per_view_pose_then_average": False,
        "camera_extrinsic_optimized": False,
        "whole_mask_centroid_used": False,
        "c2_yaw_hypothesis_rad": yaw_hypothesis_rad,
        "losses": ["mating_rim_rgb_subpixel_edge", "silhouette_distance_transform", "visible_depth_point_to_plane", "normal", "in_hand_kinematic_prior", "z_buffer_occlusion"],
    }


def group_whitened_observable_covariance(
    jacobian,
    residual,
    group_sizes,
    *,
    lambda_cut_ratio: float = 1.0e-9,
    mad_floor_ratio: float = 1.0e-6,
) -> dict[str, Any]:
    """Whiten residual groups, then invert the no-damping ``J^T J`` on its
    observable subspace.

    This helper is intentionally small and reused by the post-grasp shadow
    estimator.  It never adds LM damping, never uses a robust-loss Hessian, and
    never applies an online scalar inflation to the returned covariance.
    ``lambda_cut_ratio`` and ``mad_floor_ratio`` are SIM_TUNING_ONLY_CANDIDATE
    numerical thresholds, not authorization gates.
    """
    jac = np.asarray(jacobian, dtype=np.float64)
    res = np.asarray(residual, dtype=np.float64)
    sizes = tuple(int(value) for value in group_sizes)
    if res.ndim != 1 or jac.ndim != 2 or jac.shape[0] != res.shape[0]:
        raise ValueError("jacobian and residual shape mismatch")
    if sum(sizes) != res.shape[0] or any(value <= 0 for value in sizes):
        raise ValueError("group_sizes must partition the residual")
    group_scales = []
    start = 0
    for size in sizes:
        stop = start + size
        chunk = res[start:stop]
        median = float(np.median(chunk))
        mad = float(np.median(np.abs(chunk - median)))
        scale = 1.4826 * mad
        scale = max(scale, mad_floor_ratio * max(1.0, float(np.max(np.abs(chunk)) + 1.0)))
        group_scales.append(scale)
        start = stop
    group_scales = np.asarray(group_scales, dtype=np.float64)
    scales = group_scales
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("residual group scales are not finite and positive")
    group_scale = np.repeat(scales, sizes)
    rw = res / group_scale
    jw = jac / group_scale[:, None]
    hessian = jw.T @ jw
    if not np.all(np.isfinite(hessian)):
        raise ValueError("whitened Hessian is non-finite")
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    lambda_cut = lambda_cut_ratio * max(float(np.max(eigenvalues)), 1.0)
    observable = eigenvalues > lambda_cut
    rank = int(np.sum(observable))
    covariance = np.zeros_like(hessian)
    if rank > 0:
        vals = eigenvalues[observable]
        vecs = eigenvectors[:, observable]
        covariance = (vecs / vals[None, :]) @ vecs.T
    dof = max(1, res.shape[0] - rank)
    chi2_dof = float(np.sum(rw ** 2)) / float(dof)
    return {
        "residual_group_scales": group_scales.tolist(),
        "whitened_hessian": hessian.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "lambda_cut": float(lambda_cut),
        "lambda_cut_ratio": float(lambda_cut_ratio),
        "observable_mask": observable.tolist(),
        "observable_dofs": rank,
        "covariance_observable_subspace": covariance.tolist(),
        "chi2_dof": chi2_dof,
        "lm_damping_added": False,
        "robust_loss_used": False,
        "calibration_status": "UNVALIDATED",
    }


def overlay_registration(camera, plug_cad, receptacle_pose_world, observation, relative_pose):
    receptacle = np.asarray(receptacle_pose_world)
    rel = np.asarray(relative_pose)
    rr = Rotation.from_euler("xyz", receptacle[3:]).as_matrix()
    plug = transform_points(plug_cad, rel)
    plug_world = CadPoints(plug.xyz @ rr.T + receptacle[:3], plug.normal @ rr.T, plug.label, plug.edge)
    predicted = render_points(camera, (plug_world,))
    image = observation["rgb"].copy()
    predicted_edge = cv2.Canny(predicted["rgb"], 30, 90) > 0
    observed_edge = cv2.Canny(observation["rgb"], 30, 90) > 0
    image[observed_edge] = (30, 255, 80)
    image[predicted_edge] = (255, 60, 220)
    return image


__all__ = ["CameraModel", "CadPoints", "fixed_camera_model", "overlay_registration", "project", "proxy_cad_points", "register_relative_pose", "register_relative_pose_multiview", "render_points", "transform_points"]
