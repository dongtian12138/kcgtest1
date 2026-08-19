"""RGB-D-only post-grasp shadow estimator.

Formal inputs are strictly RGB, depth, camera calibration, robot FK and
timestamps.  Isaac semantic masks / labels / object truth are rejected by the
archive loader and are never referenced here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from kcg_connector.d38999_cad_registration import (
    CameraModel,
    CadPoints,
    PLUG_MATING,
    PLUG_NUT_BODY,
    PLUG_REAR_BODY,
    RECEPTACLE_MATING,
    group_whitened_observable_covariance,
    project,
    proxy_cad_points,
    shell25j_plug_cad_profile,
)
from kcg_connector.d38999_inhand_multiview import (
    c2_action_pose6,
    c2_action_state12,
    pose_matrix,
)
from kcg_connector.d38999_key_branch_selector import (
    blocked_key_branch_selection,
)

SHADOW_RESULT_SCHEMA_VERSION = "kcg_d38999_postgrasp_shadow_v1"
ALL_THRESHOLDS_CANDIDATE = {
    "edge_scale_px": 1.25,
    "depth_scale_m": 0.00075,
    "depth_visibility_margin_m": 0.0015,
    "normal_scale": 0.08,
    "occlusion_clip": 4.0,
    "missing_depth_support": 2.0,
    "prior_weight": 0.12,
    "parameter_scale": [0.001, 0.001, 0.001, 0.05, 0.05, 0.05],
    "lambda_cut_ratio": 1.0e-9,
    "rz_sensitivity_unobservable": 1.0e-6,
    "branch_5dof_consensus": [0.0005, 0.0005, 0.0005, math.radians(1.0), math.radians(1.0)],
    "hp_rz_search_half_width_rad": math.radians(7.0),
    "rp_rz_search_half_width_rad": math.radians(7.0),
    "max_nfev": 75,
    "physical_jacobian_translation_step_m": 1.0e-4,
    "physical_jacobian_rotation_step_rad": 1.0e-4,
    "multistart_translation_envelope_m": 0.002,
    "multistart_rotation_envelope_rad": 0.10471975511965977,
    "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
}


def _current_proxy_key_branch_block() -> dict[str, Any]:
    """Make the current unkeyed asset an explicit pre-observation blocker."""
    return blocked_key_branch_selection(
        "KEYED_GEOMETRY_UNAVAILABLE",
        "current Shell25J C2 proxy has no traceable unique key geometry",
        current_model_id="d38999_shell25j_proxy_v1",
    )


class FormalArchiveError(ValueError):
    """Raised when a formal observation archive is invalid or polluted."""


@dataclass(frozen=True)
class FormalView:
    view_id: str
    timestamp_utc: str
    rgb: np.ndarray
    depth: np.ndarray
    camera: CameraModel
    T_WH: np.ndarray
    T_WC: np.ndarray
    T_HC: np.ndarray | None = None
    group: str = "postgrasp_inhand_views"
    extrinsic_source: str = "T_HC_calibrated"


class _ResidualProblem:
    """One C2 branch residual built from CAD and RGB/depth observations."""

    def __init__(
        self,
        views: Sequence[FormalView],
        plug_cad: CadPoints,
        receptacle_cad: CadPoints,
        initial_state,
        *,
        frozen_mask: tuple[bool, ...] | None = None,
        include_prior: bool = True,
        endpoints: tuple[str, ...] = ("plug", "receptacle"),
        plug_labels: Sequence[int] = (PLUG_MATING,),
        plug_occluder_cad: CadPoints | None = None,
        hand_occluder_cad: CadPoints | None = None,
        occlusion_policy: str = "baseline",
        minimum_visible_support_fraction: float = 0.05,
        edge_policy: str = "global",
        edge_depth_band_m: float = 0.005,
        minimum_edge_support_fraction: float = 0.02,
        missing_surface_margin_m: float = 0.015,
        missing_surface_support: float = 2.0,
    ):
        if len(views) < 1:
            raise FormalArchiveError("at least one formal view is required")
        self.views = tuple(views)
        self.initial = np.asarray(initial_state, dtype=np.float64)
        if self.initial.shape != (12,):
            raise ValueError("initial_state must have shape (12,)")
        self.include_prior = bool(include_prior)
        self.frozen_mask = (
            tuple(bool(value) for value in frozen_mask)
            if frozen_mask is not None
            else tuple(False for _ in range(12))
        )
        if len(self.frozen_mask) != 12:
            raise ValueError("frozen_mask must have length 12")
        if not set(endpoints).issubset({"plug", "receptacle"}) or not endpoints:
            raise ValueError("endpoints must be a non-empty subset of plug/receptacle")
        self.endpoints = tuple(endpoints)
        if occlusion_policy not in {
            "baseline",
            "ignore_foreground_occluded",
            "ignore_foreground_and_cad_occluder",
        }:
            raise ValueError(f"unknown occlusion_policy: {occlusion_policy}")
        if edge_policy not in {"global", "depth_gated"}:
            raise ValueError(f"unknown edge_policy: {edge_policy}")
        self.occlusion_policy = occlusion_policy
        self.edge_policy = edge_policy
        self.edge_depth_band_m = float(edge_depth_band_m)
        self.minimum_edge_support_fraction = float(
            minimum_edge_support_fraction
        )
        self.missing_surface_margin_m = float(missing_surface_margin_m)
        self.missing_surface_support = float(missing_surface_support)
        self.minimum_visible_support_fraction = float(
            minimum_visible_support_fraction
        )
        self.plug_samples = _sample_labels(
            plug_cad, tuple(plug_labels), 999999
        )
        occluder_source = plug_occluder_cad if plug_occluder_cad is not None else plug_cad
        occluder_labels = (
            (PLUG_NUT_BODY, PLUG_REAR_BODY)
            if plug_occluder_cad is not None
            else (PLUG_NUT_BODY,)
        )
        self.plug_occluder_samples = (
            _sample_labels(occluder_source, occluder_labels, 999999)
            if plug_occluder_cad is not None
            else None
        )
        self.hand_occluder_cad = hand_occluder_cad
        self._hand_depth_images = self._build_hand_depth_images()
        self.receptacle_samples = _sample_mating(
            receptacle_cad, RECEPTACLE_MATING, 999999
        )
        self.view_prep = tuple(_prepare_view(view) for view in views)
        self._group_sizes = self._compute_group_sizes()
        self.parameter_scale = np.asarray(
            ALL_THRESHOLDS_CANDIDATE["parameter_scale"] * 2, dtype=np.float64
        )
        self.prior_scale = np.asarray(
            (0.004, 0.004, 0.008, math.radians(5), math.radians(5), math.radians(6))
            * 2,
            dtype=np.float64,
        )
        self._last_residual: np.ndarray | None = None
        self.last_plug_support: list[dict[str, Any]] = []

    def _build_hand_depth_images(self) -> tuple[np.ndarray | None, ...]:
        """One z-buffer per view of the hand occluder vertices (state-free)."""
        if self.hand_occluder_cad is None:
            return tuple(None for _ in self.views)
        images = []
        hand_xyz = self.hand_occluder_cad.xyz
        for view in self.views:
            world = (
                hand_xyz @ view.T_WH[:3, :3].T
                + view.T_WH[:3, 3].reshape(1, 3)
            )
            uv, depth = project(view.camera, world)
            u = np.rint(uv[:, 0]).astype(np.int64)
            v = np.rint(uv[:, 1]).astype(np.int64)
            valid = (
                (depth > 0.03)
                & (u >= 0)
                & (u < view.camera.width)
                & (v >= 0)
                & (v < view.camera.height)
            )
            image = np.full(
                (view.camera.height, view.camera.width),
                np.inf,
                dtype=np.float64,
            )
            indices = np.flatnonzero(valid)
            order = indices[np.argsort(-depth[indices])]
            image[v[order], u[order]] = depth[order]
            images.append(image)
        return tuple(images)

    def _compute_group_sizes(self) -> tuple[int, ...]:
        sizes: list[int] = []
        samples = tuple(
            sample
            for name, sample in (("plug", self.plug_samples), ("receptacle", self.receptacle_samples))
            if name in self.endpoints
        )
        for _ in self.views:
            for sample in samples:
                # edge, depth, normal, depth-support, occlusion are all
                # full-sample groups with fixed boundaries for whitening.
                sizes.extend((len(sample.xyz),) * 5)
        if self.include_prior:
            sizes.append(12)
        return tuple(sizes)

    def endpoint_camera(self, T_WE: np.ndarray, camera: CameraModel, cad: CadPoints):
        """Return endpoint points/normals in world and camera coordinates."""
        world = cad.xyz @ T_WE[:3, :3].T + T_WE[:3, 3]
        world_normal = cad.normal @ T_WE[:3, :3].T
        cam = (world - np.asarray(camera.position_world)) @ np.asarray(
            camera.world_to_camera
        ).T
        cam_normal = world_normal @ np.asarray(camera.world_to_camera).T
        return world, world_normal, cam, cam_normal

    def _endpoint_residual(
        self,
        state,
        view_index: int,
        endpoint: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        view = self.views[view_index]
        prep = self.view_prep[view_index]
        hp = state[:6]
        rp = state[6:]
        T_HP = pose_matrix(hp)
        T_RP = pose_matrix(rp)
        T_WP = view.T_WH @ T_HP
        if endpoint == "plug":
            T_WE = T_WP
            cad = self.plug_samples
        else:
            T_WE = T_WP @ np.linalg.inv(T_RP)
            cad = self.receptacle_samples
        world_xyz, world_normal, cam_xyz, cam_normal = self.endpoint_camera(
            T_WE, view.camera, cad
        )
        uv, predicted_depth = project(view.camera, world_xyz)
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        valid = (
            (predicted_depth > 0.03)
            & (u >= 1)
            & (u < view.camera.width - 1)
            & (v >= 1)
            & (v < view.camera.height - 1)
        )
        uc = np.clip(u, 0, view.camera.width - 1)
        vc = np.clip(v, 0, view.camera.height - 1)
        observed_depth = view.depth[vc, uc]
        depth_valid = valid & np.isfinite(observed_depth) & (observed_depth > 0.0)

        depth_res = np.full(len(cad.xyz), 2.0, dtype=np.float64)
        normal_res = np.zeros(len(cad.xyz), dtype=np.float64)
        support_res = np.full(len(cad.xyz), 2.0, dtype=np.float64)
        edge_res = np.zeros(len(cad.xyz), dtype=np.float64)
        occlusion_res = np.zeros(len(cad.xyz), dtype=np.float64)

        # CAD occluder coverage is computed for the plug endpoint whenever an
        # occluder CAD is available; it defines which points can never be
        # visible from this camera (behind the Plug's own coupling nut / rear
        # body).  The residual policy is unchanged; the support diagnostics
        # additionally report the occluder-adjusted denominator.
        cad_occluded = np.zeros(len(cad.xyz), dtype=bool)
        hand_image = self._hand_depth_images[view_index]
        if endpoint == "plug" and hand_image is not None:
            hand_front = (
                (hand_image[vc, uc] < predicted_depth - 0.0005)
                & valid
            )
            cad_occluded = cad_occluded | hand_front
        if endpoint == "plug" and self.plug_occluder_samples is not None:
            # Sightline occlusion: the Plug's own occluders (coupling nut /
            # rear body) move with the Plug, so their z-buffer is rebuilt at
            # the current state.  A face sample is occluder-covered when the
            # occluder surface in front of it is nearer than the sample.
            occluder_world, _, _, _ = self.endpoint_camera(
                T_WE, view.camera, self.plug_occluder_samples
            )
            occluder_uv, occluder_depth = project(
                view.camera, occluder_world
            )
            occluder_u = np.rint(occluder_uv[:, 0]).astype(np.int64)
            occluder_v = np.rint(occluder_uv[:, 1]).astype(np.int64)
            occluder_valid = (
                (occluder_depth > 0.03)
                & (occluder_u >= 0)
                & (occluder_u < view.camera.width)
                & (occluder_v >= 0)
                & (occluder_v < view.camera.height)
            )
            occluder_buffer = np.full(
                (view.camera.height, view.camera.width),
                np.inf,
                dtype=np.float64,
            )
            occluder_indices = np.flatnonzero(occluder_valid)
            occluder_order = occluder_indices[
                np.argsort(-occluder_depth[occluder_indices])
            ]
            occluder_buffer[
                occluder_v[occluder_order],
                occluder_u[occluder_order],
            ] = occluder_depth[occluder_order]
            occluder_front = (
                (occluder_buffer[vc, uc] < predicted_depth - 0.0005)
                & valid
            )
            cad_occluded = cad_occluded | occluder_front
        visible = depth_valid & (
            predicted_depth <= observed_depth + 1.5e-3
        )
        missing = depth_valid & (
            observed_depth > predicted_depth + self.missing_surface_margin_m
        )
        visible = visible & ~missing
        depth_res[visible] = (
            (predicted_depth[visible] - observed_depth[visible]) / 0.00075
        )
        support_res[depth_valid] = 0.0
        support_res[missing] = self.missing_surface_support
        depth_res[missing] = 0.0
        occlusion_res[missing] = 0.0
        behind = depth_valid & ~visible & ~missing
        ignored = behind | cad_occluded
        occlusion_res[behind] = np.minimum(
            4.0,
            np.maximum(
                0.0,
                (predicted_depth[behind] - observed_depth[behind]) / 0.0015,
            ),
        )
        if self.occlusion_policy != "baseline":
            depth_res[~visible] = 0.0
            normal_res[ignored] = 0.0
            support_res[~visible & ~missing] = 0.0
            occlusion_res[ignored] = 0.0
        observed_normal = prep["normal_image"][vc, uc]
        normal_valid = visible & (
            np.linalg.norm(observed_normal, axis=1) > 0.5
        )
        normal_res[normal_valid] = (
            1.0
            - np.sum(
                cam_normal[normal_valid] * observed_normal[normal_valid], axis=1
            )
        ) / 0.08
        edge_indices = np.flatnonzero(cad.edge)
        edge_sample = cv2.remap(
            prep["edge_distance"].astype(np.float32),
            uv[edge_indices, 0].astype(np.float32).reshape(-1, 1),
            uv[edge_indices, 1].astype(np.float32).reshape(-1, 1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=60.0,
        ).ravel()
        edge_res[edge_indices] = np.where(
            valid[edge_indices], edge_sample / 1.25, 30.0
        )
        if self.edge_policy == "depth_gated":
            edge_valid_indices = edge_indices[valid[edge_indices]]
            edge_depth = observed_depth[edge_valid_indices]
            edge_predicted = predicted_depth[edge_valid_indices]
            edge_active = (
                np.isfinite(edge_depth)
                & (edge_depth > 0.0)
                & (
                    np.abs(edge_predicted - edge_depth)
                    <= self.edge_depth_band_m
                )
            )
            self._last_edge_active_count = int(np.sum(edge_active))
            self._last_edge_valid_count = max(1, len(edge_valid_indices))
            edge_res[edge_valid_indices[~edge_active]] = 0.0
        else:
            self._last_edge_active_count = int(np.sum(valid[edge_indices]))
            self._last_edge_valid_count = max(1, int(np.sum(valid[edge_indices])))
        if self.occlusion_policy != "baseline":
            edge_res[edge_indices[ignored[edge_indices]]] = 0.0
            edge_res[edge_indices[missing[edge_indices]]] = 0.0
            edge_res[edge_indices[~valid[edge_indices]]] = 0.0
        if endpoint == "plug":
            in_frame_count = int(np.sum(valid))
            in_frame_fraction = (
                in_frame_count / max(1, len(cad.xyz))
            )
            visible_count = int(np.sum(visible))
            finite_count = int(np.sum(depth_valid))
            edge_count = int(np.sum(valid[edge_indices]))
            edge_support_fraction = (
                float(
                    np.mean(
                        edge_sample[np.flatnonzero(valid[edge_indices])] < 5.0
                    )
                )
                if edge_count > 0
                else 0.0
            )
            depth_gated_edge_support_fraction = float(
                getattr(self, "_last_edge_active_count", 0)
                / max(1, getattr(self, "_last_edge_valid_count", 1))
            )
            self.last_plug_support.append(
                {
                    "view_id": view.view_id,
                    "in_frame_fraction": in_frame_fraction,
                    "finite_depth_fraction": finite_count / max(1, in_frame_count),
                    "visible_depth_support_fraction": visible_count
                    / max(1, in_frame_count),
                    "visible_depth_support_fraction_occluder_adjusted": (
                        visible_count
                        / max(1, in_frame_count - int(np.sum(cad_occluded)))
                    ),
                    "informative_support_fraction": (
                        visible_count
                        / max(
                            1,
                            in_frame_count
                            - int(np.sum(behind))
                            - int(np.sum(missing)),
                        )
                    ),
                    "foreground_occluded_fraction": int(np.sum(behind))
                    / max(1, in_frame_count),
                    "missing_surface_fraction": int(np.sum(missing))
                    / max(1, in_frame_count),
                    "cad_occluder_fraction": int(np.sum(cad_occluded))
                    / max(1, in_frame_count),
                    "edge_support_fraction": edge_support_fraction,
                    "depth_gated_edge_support_fraction": depth_gated_edge_support_fraction,
                    "minimum_edge_support_fraction_candidate": self.minimum_edge_support_fraction,
                    "minimum_visible_support_fraction_candidate": self.minimum_visible_support_fraction,
                }
            )
        return edge_res, depth_res, normal_res, support_res, occlusion_res

    def residual(self, state):
        self.last_plug_support = []
        chunks: list[np.ndarray] = []
        for view_index in range(len(self.views)):
            for endpoint in self.endpoints:
                chunks.extend(
                    self._endpoint_residual(state, view_index, endpoint)
                )
        if self.include_prior:
            prior = (np.asarray(state) - self.initial) / self.prior_scale
            prior = np.where(self.frozen_mask, 0.0, prior)
            chunks.append(0.12 * prior)
        result = np.concatenate(chunks)
        self._last_residual = result
        return result

    @property
    def group_sizes(self):
        return self._group_sizes

    def normalized_residual(self, normalized):
        state = self.initial + self.parameter_scale * normalized
        state = np.where(self.frozen_mask, self.initial, state)
        return self.residual(state)

    def _physical_normalized_jacobian(self):
        translation_step = ALL_THRESHOLDS_CANDIDATE[
            "physical_jacobian_translation_step_m"
        ]
        rotation_step = ALL_THRESHOLDS_CANDIDATE[
            "physical_jacobian_rotation_step_rad"
        ]
        physical_steps = np.asarray(
            [translation_step] * 3 + [rotation_step] * 3 + [translation_step] * 3 + [rotation_step] * 3,
            dtype=np.float64,
        )
        normalized_steps = physical_steps / self.parameter_scale

        def jacobian(normalized):
            residual0 = self.normalized_residual(normalized)
            columns = []
            for index in range(12):
                if self.frozen_mask[index]:
                    columns.append(np.zeros_like(residual0))
                    continue
                step = normalized_steps[index]
                plus = normalized.copy()
                plus[index] += step
                minus = normalized.copy()
                minus[index] -= step
                columns.append(
                    (
                        self.normalized_residual(plus)
                        - self.normalized_residual(minus)
                    )
                    / (2.0 * step)
                )
            return np.column_stack(columns)

        return jacobian, normalized_steps

    def solve(self, max_nfev: int = 75, jacobian_mode: str = "default"):
        zero = np.zeros(12, dtype=np.float64)
        lower = self.initial + np.asarray(
            (
                -0.006, -0.006, -0.008, -math.radians(7), -math.radians(7),
                -ALL_THRESHOLDS_CANDIDATE["hp_rz_search_half_width_rad"],
                -0.012, -0.012, -0.020, -math.radians(8),
                -math.radians(8),
                -ALL_THRESHOLDS_CANDIDATE["rp_rz_search_half_width_rad"],
            ),
            dtype=np.float64,
        )
        upper = self.initial + np.asarray(
            (
                0.006, 0.006, 0.008, math.radians(7), math.radians(7),
                ALL_THRESHOLDS_CANDIDATE["hp_rz_search_half_width_rad"],
                0.012, 0.012, 0.020, math.radians(8),
                math.radians(8),
                ALL_THRESHOLDS_CANDIDATE["rp_rz_search_half_width_rad"],
            ),
            dtype=np.float64,
        )
        normalized_lower = np.where(
            self.frozen_mask,
            -1.0e-12,
            (lower - self.initial) / self.parameter_scale,
        )
        normalized_upper = np.where(
            self.frozen_mask,
            1.0e-12,
            (upper - self.initial) / self.parameter_scale,
        )
        jacobian = None
        normalized_steps = None
        if jacobian_mode == "physical_central":
            jacobian, normalized_steps = self._physical_normalized_jacobian()
        elif jacobian_mode != "default":
            raise ValueError(f"unknown jacobian_mode: {jacobian_mode}")
        solver_kwargs = {
            "bounds": (normalized_lower, normalized_upper),
            "max_nfev": max_nfev,
            "loss": "linear",
        }
        if jacobian is not None:
            solver_kwargs["jac"] = jacobian
        result = least_squares(
            self.normalized_residual,
            zero,
            **solver_kwargs,
        )
        state = self.initial + self.parameter_scale * result.x
        state = np.where(self.frozen_mask, self.initial, state)
        residual = self.residual(state)
        jac = result.jac
        return {
            "state": state,
            "residual": residual,
            "jacobian": jac,
            "jacobian_mode": jacobian_mode,
            "physical_jacobian_normalized_steps": (
                None if normalized_steps is None else normalized_steps.tolist()
            ),
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "solver_nfev": int(result.nfev),
            "solver_njev": int(result.njev),
            "solver_optimality": float(result.optimality),
            "solver_active_mask": np.asarray(
                result.active_mask, dtype=int
            ).tolist() if result.active_mask is not None else [],
            "success": bool(result.success and np.all(np.isfinite(state))),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
        }


def _sample_labels(
    cad: CadPoints, labels: Sequence[int], count: int
) -> CadPoints:
    mask = np.isin(cad.label, tuple(labels))
    xyz = cad.xyz[mask]
    normal = cad.normal[mask]
    edge = cad.edge[mask]
    keep = np.linspace(0, len(xyz) - 1, min(count, len(xyz)), dtype=np.int64)
    return CadPoints(xyz[keep], normal[keep], cad.label[mask][keep], edge[keep])


def _sample_mating(cad: CadPoints, label: int, count: int) -> CadPoints:
    return _sample_labels(cad, (label,), count)


def _prepare_view(view: FormalView) -> dict[str, np.ndarray]:
    if view.rgb.ndim != 3 or view.rgb.shape[:2] != view.depth.shape:
        raise FormalArchiveError("rgb/depth shape mismatch")
    if view.rgb.shape[0] != view.camera.height or view.rgb.shape[1] != view.camera.width:
        raise FormalArchiveError("rgb/depth resolution does not match camera")
    gray = cv2.cvtColor(np.asarray(view.rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    observed_edge = cv2.Canny(gray, 30, 90)
    edge_distance = cv2.distanceTransform(
        (observed_edge == 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    normal_image = _depth_normals(view.depth, view.camera)
    return {
        "gray": gray,
        "edge_distance": edge_distance,
        "normal_image": normal_image,
    }


def _depth_normals(depth: np.ndarray, camera: CameraModel) -> np.ndarray:
    image = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(image) & (image > 0.0)
    z = np.where(valid, image, 1.0)
    dzdx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / max(camera.fx, 1.0e-9)
    dzdy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / max(camera.fy, 1.0e-9)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(z)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    norm = np.maximum(norm, 1.0e-9)
    out = np.dstack((nx / norm, ny / norm, nz / norm)).astype(np.float32)
    out[~valid] = 0.0
    return out


def _pose_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    return np.abs(da - db)


def _optimize_branch(
    views: Sequence[FormalView],
    plug_cad: CadPoints,
    receptacle_cad: CadPoints,
    initial_state,
    *,
    frozen_mask: tuple[bool, ...] | None = None,
    max_nfev: int = 75,
    endpoints: tuple[str, ...] = ("plug", "receptacle"),
    plug_labels: Sequence[int] = (PLUG_MATING,),
    plug_occluder_cad: CadPoints | None = None,
    hand_occluder_cad: CadPoints | None = None,
    occlusion_policy: str = "baseline",
    minimum_visible_support_fraction: float = 0.05,
    edge_policy: str = "global",
    edge_depth_band_m: float = 0.005,
    minimum_edge_support_fraction: float = 0.02,
    jacobian_mode: str = "default",
) -> dict[str, Any]:
    problem = _ResidualProblem(
        views,
        plug_cad,
        receptacle_cad,
        initial_state,
        frozen_mask=frozen_mask,
        endpoints=endpoints,
        plug_labels=plug_labels,
        plug_occluder_cad=plug_occluder_cad,
        hand_occluder_cad=hand_occluder_cad,
        occlusion_policy=occlusion_policy,
        minimum_visible_support_fraction=minimum_visible_support_fraction,
        edge_policy=edge_policy,
        edge_depth_band_m=edge_depth_band_m,
        minimum_edge_support_fraction=minimum_edge_support_fraction,
    )
    solved = problem.solve(max_nfev=max_nfev, jacobian_mode=jacobian_mode)
    covariance = group_whitened_observable_covariance(
        solved["jacobian"],
        solved["residual"],
        problem.group_sizes,
        lambda_cut_ratio=ALL_THRESHOLDS_CANDIDATE["lambda_cut_ratio"],
    )
    solved["covariance"] = covariance
    solved["group_sizes"] = list(problem.group_sizes)
    solved["plug_support_diagnostics"] = list(problem.last_plug_support)
    solved["plug_support_gate_failed"] = bool(
        any(
            item.get(
                "informative_support_fraction",
                item.get(
                    "visible_depth_support_fraction_occluder_adjusted",
                    item["visible_depth_support_fraction"],
                ),
            )
            < minimum_visible_support_fraction
            or item["depth_gated_edge_support_fraction"]
            < minimum_edge_support_fraction
            for item in problem.last_plug_support
        )
    ) if problem.last_plug_support else True
    solved["frozen_mask"] = list(problem.frozen_mask)
    return solved


def _rz_sensitivity(
    problem: _ResidualProblem, state: np.ndarray, rz_index: int = 11
) -> float:
    base = float(np.sum(problem.residual(state) ** 2))
    eps = 0.05
    plus = state.copy()
    plus[rz_index] += eps
    minus = state.copy()
    minus[rz_index] -= eps
    return max(
        abs(float(np.sum(problem.residual(plus) ** 2)) - base),
        abs(float(np.sum(problem.residual(minus) ** 2)) - base),
    ) / (eps * eps)


def estimate_joint_c2(
    views: Sequence[FormalView],
    initial_state,
    *,
    plug_cad: CadPoints | None = None,
    receptacle_cad: CadPoints | None = None,
) -> dict[str, Any]:
    """Estimate two full C2 branches.  Auth flags are always false."""
    if len(views) < 2:
        return {
            "success": False,
            "status": "REJECTED",
            "reject_reason": "INSUFFICIENT_VIEWS",
            "shadow_authorized": False,
            "control_authorized": False,
            "covariance_calibrated": False,
        }
    if plug_cad is None or receptacle_cad is None:
        plug_cad, receptacle_cad = proxy_cad_points()
    initial = np.asarray(initial_state, dtype=np.float64)
    if initial.shape != (12,):
        raise ValueError("initial_state must have shape (12,)")
    branches = []
    for yaw in (0.0, math.pi):
        branch_initial = c2_action_state12(initial) if yaw else initial.copy()
        solved = _optimize_branch(
            views, plug_cad, receptacle_cad, branch_initial
        )
        solved["yaw_hypothesis_rad"] = yaw
        branches.append(solved)
    rz_sens = min(
        _rz_sensitivity(
            _ResidualProblem(
                views, plug_cad, receptacle_cad, br["state"], include_prior=False
            ),
            br["state"],
        )
        for br in branches
    )
    rz_unobservable = bool(rz_sens <= ALL_THRESHOLDS_CANDIDATE["rz_sensitivity_unobservable"])
    consensus = _pose_error(branches[0]["state"][6:11], branches[1]["state"][6:11])
    consensus_ok = all(
        float(value) <= threshold
        for value, threshold in zip(
            consensus, ALL_THRESHOLDS_CANDIDATE["branch_5dof_consensus"]
        )
    )
    all_success = all(item["success"] for item in branches)
    covariance_ok = all(
        np.all(np.isfinite(item["covariance"]["covariance_observable_subspace"]))
        for item in branches
    )
    if all_success and rz_unobservable and consensus_ok and covariance_ok:
        status = "VALID_5DOF_C2_UNRESOLVED"
        success = True
    elif all_success and not rz_unobservable and consensus_ok:
        status = "VALID_6DOF_C2_RESOLVED_BY_OBSERVATION"
        success = True
    else:
        status = "REJECTED"
        success = False
    result = {
        "schema_version": SHADOW_RESULT_SCHEMA_VERSION,
        "success": success,
        "status": status,
        "reject_reason": None if success else "C2_BRANCH_FAILED",
        "c2": {
            "retained_hypotheses": 2,
            "averaged": False,
            "selected_for_control": None,
            "resolution": "C2_UNRESOLVED" if rz_unobservable else "C2_RESOLVED_BY_OBSERVATION",
            "observable_dofs": 5 if rz_unobservable else 6,
            "rz_sensitivity_cost_per_rad2": rz_sens,
            "branch_5dof_consensus": consensus.tolist(),
            "branch_5dof_consensus_ok": consensus_ok,
            "hypotheses": [
                {
                    "id": "YAW_0" if item["yaw_hypothesis_rad"] == 0.0 else "YAW_PI",
                    "yaw_hypothesis_rad": item["yaw_hypothesis_rad"],
                    "T_hand_plug_xyz_rpy": item["state"][:6].tolist(),
                    "T_receptacle_plug_xyz_rpy": item["state"][6:].tolist(),
                    "cost": item["cost"],
                    "success": item["success"],
                }
                for item in branches
            ],
        },
        "covariance": {
            "calibration_status": "UNVALIDATED",
            "coverage_calibrated": False,
            "whitened_observable_covariance_per_branch": [
                item["covariance"] for item in branches
            ],
            "conditional_covariance_xyz_rx_ry_5x5_per_branch": [
                np.asarray(item["covariance"]["covariance_observable_subspace"])[
                    np.ix_([6, 7, 8, 9, 10], [6, 7, 8, 9, 10])
                ].tolist()
                for item in branches
            ],
        },
        "shadow_authorized": False,
        "control_authorized": False,
        "covariance_calibrated": False,
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }
    if status == "VALID_5DOF_C2_UNRESOLVED":
        result["T_receptacle_plug_5dof"] = branches[0]["state"][6:11].tolist()
    return result


def _solve_hp_branch(
    views,
    plug_cad,
    receptacle_cad,
    initial_state,
    hp_initial,
    *,
    frozen,
    plug_labels,
    plug_occluder_cad,
    hand_occluder_cad,
    occlusion_policy,
    minimum_visible_support_fraction,
    edge_policy,
    edge_depth_band_m,
    minimum_edge_support_fraction,
    jacobian_mode,
):
    branch_state = np.concatenate(
        (np.asarray(hp_initial, dtype=np.float64), np.asarray(initial_state[6:], dtype=np.float64))
    )
    return _optimize_branch(
        views,
        plug_cad,
        receptacle_cad,
        branch_state,
        frozen_mask=frozen,
        endpoints=("plug",),
        plug_labels=plug_labels,
        plug_occluder_cad=plug_occluder_cad,
        hand_occluder_cad=hand_occluder_cad,
        occlusion_policy=occlusion_policy,
        minimum_visible_support_fraction=minimum_visible_support_fraction,
        edge_policy=edge_policy,
        edge_depth_band_m=edge_depth_band_m,
        minimum_edge_support_fraction=minimum_edge_support_fraction,
        jacobian_mode=jacobian_mode,
    )


def _deterministic_hp_starts(nominal_hp, count: int = 9):
    """Truth-free symmetric starts from the postgrasp design envelope."""
    nominal = np.asarray(nominal_hp, dtype=np.float64)
    translation = ALL_THRESHOLDS_CANDIDATE["multistart_translation_envelope_m"]
    rotation = ALL_THRESHOLDS_CANDIDATE["multistart_rotation_envelope_rad"]
    if count < 1:
        raise ValueError("multistart count must be positive")
    starts = [nominal.copy()]
    signs = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
    while len(starts) < count:
        index = (len(starts) - 1) % 6
        sign_x, sign_y = signs[((len(starts) - 1) // 6) % len(signs)]
        start = nominal.copy()
        if index < 3:
            start[index] += sign_x * translation
        else:
            start[index] += sign_x * rotation
        start[(index + 3) % 6] += sign_y * (
            translation if (index + 3) % 6 < 3 else rotation
        )
        starts.append(start)
    return starts[:count]


def estimate_postgrasp_T_HP(
    views: Sequence[FormalView],
    initial_state,
    *,
    plug_cad: CadPoints | None = None,
    receptacle_cad: CadPoints | None = None,
    plug_occluder_cad: CadPoints | None = None,
    hand_occluder_cad: CadPoints | None = None,
    plug_feature_set: str = "mating_only",
    cad_profile: str = "legacy_axisymmetric",
    cad_profile_feature_set: str = "shell_plus_socket",
    occlusion_policy: str = "baseline",
    minimum_visible_support_fraction: float = 0.05,
    edge_policy: str = "global",
    edge_depth_band_m: float = 0.005,
    minimum_edge_support_fraction: float = 0.02,
    optimizer_variant: str = "baseline",
    multistart_count: int = 9,
) -> dict[str, Any]:
    """Split-architecture branch A with two complete matrix-right C2 branches.

    The circular proxy cannot observe Plug-frame Rz, so the two branches are
    retained as equivalent hypotheses; a single ``T_hand_plug`` is never
    emitted as if Rz were resolved.
    """
    if plug_cad is None or receptacle_cad is None:
        if cad_profile == "legacy_axisymmetric":
            plug_cad, receptacle_cad = proxy_cad_points()
            # keep an explicitly supplied plug_occluder_cad (e.g. the
            # current-asset nut/rear occluders paired with the legacy
            # mating features) when one was passed in
            cad_profile_metadata = {
                "profile_id": "legacy_axisymmetric_proxy_cad_v0",
                "symmetry_order": 20,
            }
        elif cad_profile == "shell25j_c2_visible":
            shell_profile = shell25j_plug_cad_profile(
                feature_set=cad_profile_feature_set
            )
            plug_cad = shell_profile.plug_mating
            receptacle_cad = shell_profile.receptacle
            if plug_occluder_cad is None:
                plug_occluder_cad = shell_profile.plug_occluders
            cad_profile_metadata = shell_profile.metadata
        else:
            raise ValueError(f"unknown cad_profile: {cad_profile}")
    else:
        cad_profile_metadata = {
            "profile_id": cad_profile,
            "symmetry_order": 2,
        }
    initial = np.asarray(initial_state, dtype=np.float64)
    frozen = tuple(False for _ in range(6)) + tuple(True for _ in range(6))
    if plug_feature_set == "mating_only":
        plug_labels = (PLUG_MATING,)
    elif plug_feature_set == "mating_plus_nut_body":
        plug_labels = (PLUG_MATING, PLUG_NUT_BODY)
    else:
        raise ValueError(f"unknown plug_feature_set: {plug_feature_set}")
    wrist_views = [
        view for view in views if view.group == "postgrasp_inhand_views"
    ]
    secondary_inhand_views = [
        view
        for view in views
        if view.group == "postgrasp_second_inhand_camera_views"
    ]
    allowed_independent_groups = {
        "fixed_world_camera_views",
        "postgrasp_second_inhand_camera_views",
    }
    for view in views:
        if (
            view.group != "postgrasp_inhand_views"
            and view.group not in allowed_independent_groups
        ):
            raise FormalArchiveError(
                f"unknown T_HP independent view group: {view.group!r}"
            )
    independent_views = [
        view
        for view in views
        if view.group == "fixed_world_camera_views"
    ]
    ignored_comoving = []
    if wrist_views:
        # Rigid hand-mounted camera: different arm poses do not change T_CP.
        # Keep one representative wrist view for T_HP and never count the
        # others as independent Plug viewpoints.
        wrist_views = sorted(wrist_views, key=lambda view: view.view_id)
        independent_views.append(wrist_views[0])
        ignored_comoving = wrist_views[1:]
    if secondary_inhand_views:
        # A second hand-mounted camera with a DIFFERENT calibrated T_HC is a
        # genuinely different Plug viewpoint (its T_CP is a different
        # constant), so one representative is allowed as an independent view.
        # Re-using the same camera at another arm pose must NOT be placed in
        # this group.
        secondary_inhand_views = sorted(
            secondary_inhand_views, key=lambda view: view.view_id
        )
        independent_views.append(secondary_inhand_views[0])
        ignored_comoving.extend(secondary_inhand_views[1:])
    views = tuple(independent_views)
    if not views:
        return {
            "schema_version": SHADOW_RESULT_SCHEMA_VERSION,
            "success": False,
            "optimizer_converged": False,
            "pose_valid": False,
            "pose_valid_reasons": ["NO_INDEPENDENT_T_HP_VIEW"],
            "status": "REJECTED_T_HP_POSE_INVALID",
            "reject_reason": "NO_INDEPENDENT_T_HP_VIEW",
            "shadow_authorized": False,
            "control_authorized": False,
            "covariance_calibration_status": "UNVALIDATED",
            "real_keying_modeled": False,
            "keying_model_id": None,
            "key_branch_selection": _current_proxy_key_branch_block(),
        }
    jacobian_mode = (
        "physical_central"
        if optimizer_variant in {"physical_jacobian", "multistart_physical_jacobian"}
        else "default"
    )
    branch0_solved = None
    multistart_starts = []
    if optimizer_variant == "multistart_physical_jacobian":
        starts = _deterministic_hp_starts(initial[:6], multistart_count)
        for hp_start in starts:
            multistart_starts.append(hp_start.tolist())
            solved = _solve_hp_branch(
                views, plug_cad, receptacle_cad, initial, hp_start,
                frozen=frozen, plug_labels=plug_labels,
                plug_occluder_cad=plug_occluder_cad,
        hand_occluder_cad=hand_occluder_cad,
                occlusion_policy=occlusion_policy,
                minimum_visible_support_fraction=minimum_visible_support_fraction,
                edge_policy=edge_policy,
                edge_depth_band_m=edge_depth_band_m,
                minimum_edge_support_fraction=minimum_edge_support_fraction,
                jacobian_mode=jacobian_mode,
            )
            if branch0_solved is None or solved["cost"] < branch0_solved["cost"]:
                branch0_solved = solved
    else:
        branch0_solved = _solve_hp_branch(
            views, plug_cad, receptacle_cad, initial, initial[:6],
            frozen=frozen, plug_labels=plug_labels,
            plug_occluder_cad=plug_occluder_cad,
        hand_occluder_cad=hand_occluder_cad,
            occlusion_policy=occlusion_policy,
            minimum_visible_support_fraction=minimum_visible_support_fraction,
            edge_policy=edge_policy,
            edge_depth_band_m=edge_depth_band_m,
            minimum_edge_support_fraction=minimum_edge_support_fraction,
            jacobian_mode=jacobian_mode,
        )
    branch1_solved = _solve_hp_branch(
        views, plug_cad, receptacle_cad, initial,
        c2_action_pose6(np.asarray(branch0_solved["state"][:6])),
        frozen=frozen, plug_labels=plug_labels,
        plug_occluder_cad=plug_occluder_cad,
        hand_occluder_cad=hand_occluder_cad,
        occlusion_policy=occlusion_policy,
        minimum_visible_support_fraction=minimum_visible_support_fraction,
        edge_policy=edge_policy,
        edge_depth_band_m=edge_depth_band_m,
        minimum_edge_support_fraction=minimum_edge_support_fraction,
        jacobian_mode=jacobian_mode,
    )
    hypotheses = []
    for branch_id, solved in (("YAW_0", branch0_solved), ("YAW_PI", branch1_solved)):
        covariance = np.asarray(
            solved["covariance"]["covariance_observable_subspace"]
        )
        hp_cov = covariance[
            np.ix_([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
        ]
        eigvals = np.clip(np.linalg.eigvalsh(hp_cov), 0.0, None)
        observable_eigvals = eigvals[eigvals > 1.0e-12]
        condition = (
            None
            if observable_eigvals.size == 0
            else float(observable_eigvals[-1] / observable_eigvals[0])
        )
        condition_status = (
            "NO_OBSERVABLE_SUBSPACE"
            if condition is None
            else (
                "FULL_6DOF"
                if observable_eigvals.size == 6
                else "OBSERVABLE_SUBSPACE_ONLY"
            )
        )
        residual = np.asarray(solved["residual"], dtype=np.float64)
        hypotheses.append(
            {
                "id": branch_id,
                "T_hand_plug_xyz_rpy": solved["state"][:6].tolist(),
                "cost": solved["cost"],
                "probability": None,
                "probability_status": "UNVALIDATED",
                "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
                "condition_number": condition,
                "condition_number_status": condition_status,
                "observable_dofs": None,
                "covariance_6x6": hp_cov.tolist(),
                "covariance_calibration_status": "UNVALIDATED",
                "plug_support_gate_failed": solved.get(
                    "plug_support_gate_failed", True
                ),
                "plug_support_diagnostics": solved.get(
                    "plug_support_diagnostics", []
                ),
                "jacobian_mode": solved.get("jacobian_mode"),
                "solver_status": solved.get("solver_status"),
                "solver_message": solved.get("solver_message"),
                "solver_nfev": solved.get("solver_nfev"),
                "solver_njev": solved.get("solver_njev"),
                "solver_optimality": solved.get("solver_optimality"),
                "solver_active_mask": solved.get("solver_active_mask", []),
                "success": solved["success"],
            }
        )
    success = all(item["success"] for item in hypotheses)
    local_rz_problem = _ResidualProblem(
        views,
        plug_cad,
        receptacle_cad,
        np.concatenate((np.asarray(hypotheses[0]["T_hand_plug_xyz_rpy"]), initial[6:])),
        include_prior=False,
        endpoints=("plug",),
    )
    local_rz_sensitivity = _rz_sensitivity(
        local_rz_problem,
        np.concatenate((np.asarray(hypotheses[0]["T_hand_plug_xyz_rpy"]), initial[6:])),
        rz_index=5,
    )
    local_rz_observable_candidate = bool(
        local_rz_sensitivity
        > ALL_THRESHOLDS_CANDIDATE["rz_sensitivity_unobservable"]
    )
    # The proxy CAD has no asymmetric key observation.  Local Rz sensitivity
    # is noise/occlusion, not discrete C2 evidence: both branches are always
    # retained and C2 is never resolved by this estimator.
    rz_status = "C2_UNRESOLVED"
    observable_dofs = 5
    pose_valid_reasons = [
        "COVARIANCE_CALIBRATION_UNVALIDATED",
    ]
    residual_max = max(float(item["residual_rms"]) for item in hypotheses)
    condition_max = max(
        float(item["condition_number"] or 0.0) for item in hypotheses
    )
    residual_pose_valid_candidate = 5.0
    condition_pose_valid_candidate = 1.0e5
    support_gate_failed = any(
        item.get("plug_support_gate_failed") for item in hypotheses
    )
    if support_gate_failed:
        pose_valid_reasons.append("PLUG_VISIBLE_SUPPORT_BELOW_CANDIDATE")
    if residual_max > residual_pose_valid_candidate:
        pose_valid_reasons.append("RESIDUAL_RMS_EXCEEDS_CANDIDATE")
    if condition_max > condition_pose_valid_candidate:
        pose_valid_reasons.append("CONDITION_NUMBER_EXCEEDS_CANDIDATE")
    optimizer_converged = success
    pose_valid = False
    for item in hypotheses:
        item["observable_dofs"] = observable_dofs
    if optimizer_converged and pose_valid:
        status = "VALID_T_HP_5DOF_C2_UNRESOLVED"
        success = True
        reject_reason = None
    elif optimizer_converged:
        status = "REJECTED_T_HP_POSE_INVALID"
        success = False
        reject_reason = ";".join(pose_valid_reasons)
    else:
        status = "REJECTED"
        success = False
        reject_reason = "T_HP_C2_OPTIMIZATION_FAILED"
    matrix0 = pose_matrix(np.asarray(hypotheses[0]["T_hand_plug_xyz_rpy"]))
    axis = matrix0[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
    axis_norm = np.linalg.norm(axis)
    return {
        "schema_version": SHADOW_RESULT_SCHEMA_VERSION,
        "success": success,
        "status": status,
        "c2": {
            "retained_hypotheses": 2,
            "averaged": False,
            "selected_for_control": None,
            "selected_for_shadow": None,
            "resolution": "C2_UNRESOLVED",
            "rz_status": rz_status,
            "discrete_c2_unresolved": True,
            "local_rz_observable_candidate": local_rz_observable_candidate,
            "local_rz_sensitivity_cost_per_rad2": local_rz_sensitivity,
            "rz_sensitivity_cost_per_rad2": None,
            "local_rz_sensitivity_not_discrete_evidence": True,
            "observable_dofs": observable_dofs,
            "hypotheses": hypotheses,
        },
        "c2_invariant_5dof": [
            float(matrix0[0, 3]),
            float(matrix0[1, 3]),
            float(matrix0[2, 3]),
            float(axis[0] / axis_norm),
            float(axis[1] / axis_norm),
        ],
        "optimizer_converged": optimizer_converged,
        "pose_valid": pose_valid,
        "pose_valid_reasons": pose_valid_reasons,
        "pose_validation_thresholds": {
            "residual_rms_candidate": residual_pose_valid_candidate,
            "condition_number_candidate": condition_pose_valid_candidate,
            "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
        },
        "covariance_calibration_status": "UNVALIDATED",
        "shadow_authorized": False,
        "control_authorized": False,
        "real_keying_modeled": False,
        "keying_model_id": None,
        "key_branch_selection": _current_proxy_key_branch_block(),
        "plug_feature_set": plug_feature_set,
        "cad_profile": cad_profile,
        "cad_profile_feature_set": cad_profile_feature_set,
        "cad_profile_metadata": cad_profile_metadata,
        "occlusion_policy": occlusion_policy,
        "optimizer_variant": optimizer_variant,
        "multistart_count": multistart_count if optimizer_variant == "multistart_physical_jacobian" else 0,
        "multistart_starts": multistart_starts,
        "edge_policy": edge_policy,
        "edge_depth_band_m_candidate": edge_depth_band_m,
        "plug_support_diagnostics": [
            item.get("plug_support_diagnostics") for item in hypotheses
        ],
        "plug_support_gate_failed": support_gate_failed,
        "T_HP_independent_view_count": len(views),
        "T_HP_multiview": len(views) > 1,
        "T_HP_ignored_comoving_view_ids": [
            view.view_id for view in ignored_comoving
        ],
        "T_receptacle_plug_status": "UNAVAILABLE_NO_RECEPTACLE_VIEWS",
        "reject_reason": reject_reason,
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }


def estimate_preinsert_T_RP(
    views: Sequence[FormalView],
    T_hand_plug,
    initial_state,
    *,
    plug_cad: CadPoints | None = None,
    receptacle_cad: CadPoints | None = None,
) -> dict[str, Any]:
    """Split-architecture branch B: estimate T_RP from preinsert views."""
    if plug_cad is None or receptacle_cad is None:
        plug_cad, receptacle_cad = proxy_cad_points()
    hp = np.asarray(T_hand_plug, dtype=np.float64)
    if hp.shape != (6,):
        raise ValueError("T_hand_plug must have shape (6,)")
    initial = np.asarray(initial_state, dtype=np.float64)
    state0 = np.concatenate((hp, initial[6:]))
    frozen = tuple(True for _ in range(6)) + tuple(False for _ in range(6))
    solved = _optimize_branch(
        views,
        plug_cad,
        receptacle_cad,
        state0,
        frozen_mask=frozen,
        endpoints=("receptacle",),
    )
    covariance = np.asarray(
        solved["covariance"]["covariance_observable_subspace"]
    )
    rp_cov = covariance[np.ix_([6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11])].tolist()
    return {
        "schema_version": SHADOW_RESULT_SCHEMA_VERSION,
        "success": solved["success"],
        "status": "VALID_T_RP_ONLY" if solved["success"] else "REJECTED",
        "T_hand_plug_xyz_rpy": hp.tolist(),
        "T_receptacle_plug_xyz_rpy": solved["state"][6:].tolist(),
        "covariance_6x6": rp_cov,
        "covariance_calibration_status": "UNVALIDATED",
        "shadow_authorized": False,
        "control_authorized": False,
        "reject_reason": None if solved["success"] else "T_RP_OPTIMIZATION_FAILED",
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }


def estimate_grouped_views(
    *,
    postgrasp_inhand_views: Sequence[FormalView],
    final_preinsert_views: Sequence[FormalView] = (),
    initial_state,
    T_hand_plug_fixed=None,
) -> dict[str, Any]:
    """Grouped-view API required by the split-architecture A/B decision."""
    postgrasp = tuple(postgrasp_inhand_views)
    preinsert = tuple(final_preinsert_views)
    if not preinsert:
        return estimate_postgrasp_T_HP(postgrasp, initial_state)
    if not postgrasp:
        if T_hand_plug_fixed is None:
            raise FormalArchiveError(
                "T_hand_plug_fixed is required when only preinsert views exist"
            )
        return estimate_preinsert_T_RP(
            preinsert, T_hand_plug_fixed, initial_state
        )
    return estimate_joint_c2(postgrasp + preinsert, initial_state)


def load_formal_archive(path: Path | str) -> list[FormalView]:
    """Load a strict formal view archive.  Semantic/truth fields are rejected."""
    root = Path(path)
    manifest_path = root / "formal_manifest.json"
    if not manifest_path.is_file():
        raise FormalArchiveError("formal_manifest.json is missing")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SHADOW_RESULT_SCHEMA_VERSION:
        raise FormalArchiveError("unsupported formal archive schema")
    if document.get("role") != "formal_observation":
        raise FormalArchiveError("archive role is not formal_observation")
    views = []
    for record in document.get("views", []):
        view_dir = root / record["view_id"]
        rgb = _read_image(view_dir / "rgb.png")
        depth = np.load(view_dir / "depth.npy")
        camera_record = json.loads((view_dir / "camera.json").read_text())
        camera = CameraModel(
            width=int(camera_record["width"]),
            height=int(camera_record["height"]),
            fx=float(camera_record["fx"]),
            fy=float(camera_record["fy"]),
            cx=float(camera_record["cx"]),
            cy=float(camera_record["cy"]),
            position_world=tuple(camera_record["position_world"]),
            world_to_camera=tuple(tuple(row) for row in camera_record["world_to_camera"]),
        )
        fk = json.loads((view_dir / "fk.json").read_text())
        t_hc = None if fk.get("T_HC") is None else np.asarray(
            fk["T_HC"], dtype=np.float64
        )
        views.append(
            FormalView(
                view_id=str(record["view_id"]),
                timestamp_utc=str(record["timestamp_utc"]),
                rgb=rgb,
                depth=depth.astype(np.float32),
                camera=camera,
                T_WH=np.asarray(fk["T_WH"], dtype=np.float64),
                T_HC=t_hc,
                T_WC=np.asarray(fk["T_WC"], dtype=np.float64),
                group=str(record.get("group", "postgrasp_inhand_views")),
                extrinsic_source=str(
                    fk.get(
                        "extrinsic_source",
                        "fixed_camera_config_T_WC"
                        if t_hc is None
                        else "T_HC_calibrated",
                    )
                ),
            )
        )
    return views


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FormalArchiveError(f"missing rgb image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_formal_archive(root: Path | str, views: Sequence[FormalView]) -> dict[str, Any]:
    """Write an RGB/depth/FK-only archive.  Semantic sidecars are rejected."""
    output = Path(root)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for view in views:
        if view.view_id in ("posthoc_semantic", "truth_restore"):
            raise FormalArchiveError("reserved view id is not a formal view")
        view_dir = output / view.view_id
        view_dir.mkdir(parents=True, exist_ok=False)
        rgb = np.asarray(view.rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise FormalArchiveError("formal rgb must be HxWx3 uint8")
        cv2.imwrite(
            str(view_dir / "rgb.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        depth = np.asarray(view.depth, dtype=np.float32)
        if depth.shape != rgb.shape[:2]:
            raise FormalArchiveError("depth/rgb shape mismatch")
        np.save(view_dir / "depth.npy", depth)
        camera = view.camera
        (view_dir / "camera.json").write_text(
            json.dumps(
                {
                    "width": int(camera.width),
                    "height": int(camera.height),
                    "fx": float(camera.fx),
                    "fy": float(camera.fy),
                    "cx": float(camera.cx),
                    "cy": float(camera.cy),
                    "position_world": [float(v) for v in camera.position_world],
                    "world_to_camera": [
                        [float(v) for v in row] for row in camera.world_to_camera
                    ],
                },
                allow_nan=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        T_WH = np.asarray(view.T_WH, dtype=np.float64)
        T_WC = np.asarray(view.T_WC, dtype=np.float64)
        for name, matrix in (("T_WH", T_WH), ("T_WC", T_WC)):
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise FormalArchiveError(f"{name} must be finite 4x4")
        T_HC = None if view.T_HC is None else np.asarray(view.T_HC, dtype=np.float64)
        if T_HC is not None and (
            T_HC.shape != (4, 4) or not np.all(np.isfinite(T_HC))
        ):
            raise FormalArchiveError("T_HC must be finite 4x4 or null")
        (view_dir / "fk.json").write_text(
            json.dumps(
                {
                    "T_WH": T_WH.tolist(),
                    "T_HC": None if T_HC is None else T_HC.tolist(),
                    "T_WC": T_WC.tolist(),
                    "extrinsic_source": view.extrinsic_source,
                },
                allow_nan=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record = {
            "view_id": view.view_id,
            "timestamp_utc": view.timestamp_utc,
            "group": view.group,
            "extrinsic_source": view.extrinsic_source,
        }
        records.append(record)
    manifest = {
        "schema_version": SHADOW_RESULT_SCHEMA_VERSION,
        "role": "formal_observation",
        "formal_inputs": [
            "rgb",
            "depth",
            "camera_intrinsics",
            "camera_prim_contract",
            "T_HC_calibrated",
            "actual_arm_q_rad",
            "T_WH_from_fk",
            "timestamp_utc",
        ],
        "forbidden_inputs": [
            "semantic",
            "id_to_labels",
            "registered_truth_xy_m",
            "object_truth",
            "contact_report",
        ],
        "views": records,
    }
    (output / "formal_manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "SHADOW_RESULT_SCHEMA_VERSION",
    "ALL_THRESHOLDS_CANDIDATE",
    "FormalArchiveError",
    "FormalView",
    "estimate_grouped_views",
    "estimate_joint_c2",
    "estimate_postgrasp_T_HP",
    "estimate_preinsert_T_RP",
    "load_formal_archive",
    "write_formal_archive",
]
