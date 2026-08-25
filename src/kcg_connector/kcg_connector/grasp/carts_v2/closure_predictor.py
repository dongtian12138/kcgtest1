"""Predict registered sequential finger closure with first PAD contact stopping."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from kcg_connector.grasp.carts_v2.models import (
    ClosurePrediction,
    PredictedContact,
    V2Inputs,
    farthest_point_indices,
    joint_positions_for_phases,
)
from kcg_connector.grasp.carts_v2.surface_contact import (
    ExactPadSurfaceQuery,
    NearestSurface,
    nearest_motion_compatible_index,
)
from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
)


def _closest_points_on_triangles(
    triangles: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Vectorized closest points for one point paired with one triangle."""

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap = points - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    result = np.empty_like(points)
    assigned = (d1 <= 0.0) & (d2 <= 0.0)
    result[assigned] = a[assigned]

    bp = points - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    mask = (~assigned) & (d3 >= 0.0) & (d4 <= d3)
    result[mask] = b[mask]
    assigned |= mask
    vc = d1 * d4 - d3 * d2
    mask = (~assigned) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    fraction = np.divide(d1, d1 - d3, out=np.zeros_like(d1), where=(d1 != d3))
    result[mask] = a[mask] + fraction[mask, None] * ab[mask]
    assigned |= mask

    cp = points - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    mask = (~assigned) & (d6 >= 0.0) & (d5 <= d6)
    result[mask] = c[mask]
    assigned |= mask
    vb = d5 * d2 - d1 * d6
    mask = (~assigned) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    fraction = np.divide(d2, d2 - d6, out=np.zeros_like(d2), where=(d2 != d6))
    result[mask] = a[mask] + fraction[mask, None] * ac[mask]
    assigned |= mask

    va = d3 * d6 - d5 * d4
    edge = (d4 - d3) + (d5 - d6)
    mask = (~assigned) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    fraction = np.divide(
        d4 - d3, edge, out=np.zeros_like(edge), where=(edge != 0.0)
    )
    result[mask] = b[mask] + fraction[mask, None] * (c - b)[mask]
    assigned |= mask

    denominator = va + vb + vc
    inverse = np.divide(
        1.0, denominator, out=np.zeros_like(denominator), where=(denominator != 0.0)
    )
    v, w = vb * inverse, vc * inverse
    result[~assigned] = (
        a[~assigned]
        + v[~assigned, None] * ab[~assigned]
        + w[~assigned, None] * ac[~assigned]
    )
    return result


class _MeshProximityIndex:
    """Legacy centroid-neighborhood surface query retained for the baseline."""

    def __init__(self, inputs: V2Inputs) -> None:
        triangles = np.asarray(inputs.object_contract.model.mesh.face_vertices_m)
        self._triangles = triangles
        self._centroids = np.mean(triangles, axis=1)
        self._tree = cKDTree(self._centroids)
        self._neighbor_count = int(
            inputs.config.section("closure_prediction")[
                "nearest_face_candidate_count"
            ]
        )

    def query(self, points_m: np.ndarray) -> NearestSurface:
        count = min(self._neighbor_count, len(self._triangles))
        _distance, face_ids = self._tree.query(points_m, k=count)
        if count == 1:
            face_ids = np.asarray(face_ids)[:, None]
        face_ids = np.asarray(face_ids, dtype=np.int64)
        repeated_points = np.repeat(points_m, count, axis=0)
        local_triangles = self._triangles[face_ids.reshape(-1)]
        local_closest = _closest_points_on_triangles(
            local_triangles, repeated_points
        )
        local_distance = np.linalg.norm(local_closest - repeated_points, axis=1)
        local_distance = local_distance.reshape(len(points_m), count)
        local_closest = local_closest.reshape(len(points_m), count, 3)
        selected = np.argmin(local_distance, axis=1)
        rows = np.arange(len(points_m))
        return NearestSurface(
            point_m=local_closest[rows, selected],
            distance_m=local_distance[rows, selected],
            face_index=face_ids[rows, selected],
        )


def _farthest_surface_points(points: np.ndarray, count: int) -> np.ndarray:
    if count >= len(points):
        return np.array(points, copy=True)
    return np.asarray(points[farthest_point_indices(points, count)], dtype=np.float64)


def closure_phase_samples(
    inputs: V2Inputs,
    phases: tuple[float, float, float],
    phase_index: int,
    maximum_phase: float,
    reference_joint_positions_rad: tuple[float, ...],
) -> np.ndarray:
    """Sample one finger no coarser than its real per-cycle joint command."""

    settings = inputs.config.section("closure_prediction")
    minimum_intervals = int(settings["phase_sample_count"]) - 1
    if settings.get("phase_sampling_rule", "FIXED_COUNT") == "FIXED_COUNT":
        return np.linspace(phases[phase_index], maximum_phase, minimum_intervals + 1)[1:]
    if settings["phase_sampling_rule"] != "DYNAMIC_CONTROL_STEP_BOUNDED":
        raise ValueError("unknown closure phase sampling rule")
    start_phases = tuple(float(value) for value in phases)
    stop_phases = list(start_phases)
    stop_phases[phase_index] = float(maximum_phase)
    start_joints = joint_positions_for_phases(
        inputs, start_phases,
        reference_joint_positions_rad=reference_joint_positions_rad,
    )
    stop_joints = joint_positions_for_phases(
        inputs, tuple(stop_phases),
        reference_joint_positions_rad=reference_joint_positions_rad,
    )
    maximum_increment = (
        float(inputs.config.section("dynamic")["finger_maximum_speed_rad_s"])
        * float(inputs.config.section("dynamic")["physics_dt_s"])
    )
    if (
        minimum_intervals < 1
        or not np.isfinite(maximum_increment)
        or maximum_increment <= 0.0
    ):
        raise ValueError("closure sampling bounds must be positive")
    required = int(np.ceil(np.max(np.abs(stop_joints - start_joints)) / maximum_increment))
    interval_count = max(minimum_intervals, required, 1)
    return np.linspace(phases[phase_index], maximum_phase, interval_count + 1)[1:]


class SequentialClosurePredictor:
    """Reusable object index and real PAD surface samples for many candidates."""

    def __init__(self, inputs: V2Inputs) -> None:
        self.inputs = inputs
        backend = str(
            inputs.config.section("closure_prediction").get(
                "surface_query_backend", "LOCAL_CENTROID_BASELINE"
            )
        )
        if backend not in {"LOCAL_CENTROID_BASELINE", "FCL_EXACT_PAD_MESH"}:
            raise ValueError(f"unknown closure surface query backend {backend!r}")
        self._surface_query_backend = backend
        self._proximity = (
            _MeshProximityIndex(inputs)
            if backend == "LOCAL_CENTROID_BASELINE"
            else None
        )
        self._exact_pad_mesh = (
            ExactPadSurfaceQuery(inputs)
            if backend == "FCL_EXACT_PAD_MESH"
            else None
        )
        count = int(
            inputs.config.section("closure_prediction")["pad_surface_sample_count"]
        )
        self._pad_points: dict[str, np.ndarray] = {}
        for pad in inputs.hand_contract.pads:
            centroids = np.mean(pad.points_local_m[pad.faces], axis=1)
            surface = np.vstack((pad.points_local_m, centroids))
            self._pad_points[pad.name] = _farthest_surface_points(surface, count)
        self._pad_to_phase = {
            pad.name: index for index, pad in enumerate(inputs.hand_contract.pads)
        }

    def _world_pad_points(
        self, pad_name: str, phases: tuple[float, float, float], base: np.ndarray,
        reference_joint_positions_rad: tuple[float, ...],
    ) -> np.ndarray:
        joints = joint_positions_for_phases(
            self.inputs, phases,
            reference_joint_positions_rad=reference_joint_positions_rad,
        )
        transform = self.inputs.hand_model.pad_transforms(
            joints, base_transform=base
        )[pad_name]
        local = self._pad_points[pad_name]
        return local @ transform[:3, :3].T + transform[:3, 3]

    def _pad_transform(
        self, pad_name: str, phases: tuple[float, float, float], base: np.ndarray,
        reference_joint_positions_rad: tuple[float, ...],
    ) -> np.ndarray:
        joints = joint_positions_for_phases(
            self.inputs, phases,
            reference_joint_positions_rad=reference_joint_positions_rad,
        )
        return self.inputs.hand_model.pad_transforms(
            joints, base_transform=base
        )[pad_name]

    def _contact_at_phase(
        self,
        pad_name: str,
        phases: tuple[float, float, float],
        base: np.ndarray,
        reference_joint_positions_rad: tuple[float, ...],
    ) -> tuple[int, NearestSurface, np.ndarray, np.ndarray]:
        if self._exact_pad_mesh is not None:
            return self._exact_contact_at_phase(
                pad_name, phases, base, reference_joint_positions_rad
            )
        points = self._world_pad_points(
            pad_name, phases, base, reference_joint_positions_rad
        )
        assert self._proximity is not None
        nearest = self._proximity.query(points)
        phase_index = self._pad_to_phase[pad_name]
        delta = min(
            float(
                self.inputs.config.section("closure_prediction")[
                    "motion_derivative_phase_step"
                ]
            ),
            1.0 - phases[phase_index],
        )
        moved_phases = list(phases)
        moved_phases[phase_index] += delta
        moved = self._world_pad_points(
            pad_name, tuple(moved_phases), base, reference_joint_positions_rad
        )
        motion = (moved - points) / delta
        origin = self.inputs.object_contract.model.assembly_axis_origin_m
        axis = self.inputs.object_contract.model.assembly_axis
        relative = nearest.point_m - origin
        radial = relative - np.outer(relative @ axis, axis)
        radial_norm = np.linalg.norm(radial, axis=1)
        valid_radial = radial_norm > np.finfo(np.float64).eps
        normals = np.zeros_like(radial)
        normals[valid_radial] = radial[valid_radial] / radial_norm[valid_radial, None]
        inward = -np.einsum("ij,ij->i", motion, normals)
        minimum_motion = float(
            self.inputs.config.section("closure_prediction")[
                "minimum_inward_motion_m_per_phase"
            ]
        )
        eligible = valid_radial & (inward >= minimum_motion)
        eligible_distance = np.where(eligible, nearest.distance_m, np.inf)
        selected = int(np.argmin(eligible_distance)) if np.any(eligible) else -1
        return selected, nearest, normals, inward

    def _exact_contact_at_phase(
        self,
        pad_name: str,
        phases: tuple[float, float, float],
        base: np.ndarray,
        reference_joint_positions_rad: tuple[float, ...],
    ) -> tuple[int, NearestSurface, np.ndarray, np.ndarray]:
        assert self._exact_pad_mesh is not None
        current = self._pad_transform(
            pad_name, phases, base, reference_joint_positions_rad
        )
        nearest, pad_point, normal = self._exact_pad_mesh.query_pad(pad_name, current)
        phase_index = self._pad_to_phase[pad_name]
        delta = min(
            float(self.inputs.config.section("closure_prediction")[
                "motion_derivative_phase_step"
            ]),
            1.0 - phases[phase_index],
        )
        if nearest.intersecting or delta <= 0.0:
            return -1, nearest, normal[None, :], np.asarray((-np.inf,))
        moved_phases = list(phases)
        moved_phases[phase_index] += delta
        moved = self._pad_transform(
            pad_name, tuple(moved_phases), base, reference_joint_positions_rad
        )
        pad_local = (pad_point - current[:3, 3]) @ current[:3, :3]
        moved_point = pad_local @ moved[:3, :3].T + moved[:3, 3]
        inward = -float(np.dot((moved_point - pad_point) / delta, normal))
        minimum = float(self.inputs.config.section("closure_prediction")[
            "minimum_inward_motion_m_per_phase"
        ])
        if inward >= minimum or nearest.distance_m[0] > float(
            self.inputs.config.section("closure_prediction")["contact_distance_m"]
        ):
            selected = 0 if inward >= minimum else -1
            return selected, nearest, normal[None, :], np.asarray((inward,))
        local = self._pad_points[pad_name]
        current_points = local @ current[:3, :3].T + current[:3, 3]
        moved_points = local @ moved[:3, :3].T + moved[:3, 3]
        dense, dense_normals = self._exact_pad_mesh.query_points(current_points)
        dense_inward = -np.einsum(
            "ij,ij->i", (moved_points - current_points) / delta, dense_normals
        )
        selected = nearest_motion_compatible_index(
            dense.distance_m, dense_inward, minimum
        )
        return selected, dense, dense_normals, dense_inward

    def _initial_clearance(
        self,
        seed_base: np.ndarray,
        phases: tuple[float, ...],
        reference: tuple[float, ...],
    ) -> float:
        clearances = []
        for pad in self.inputs.hand_contract.pads:
            if self._exact_pad_mesh is not None:
                transform = self._pad_transform(
                    pad.name, phases, seed_base, reference
                )
                nearest, _pad_point, _normal = self._exact_pad_mesh.query_pad(
                    pad.name, transform
                )
            else:
                points = self._world_pad_points(
                    pad.name, phases, seed_base, reference
                )
                assert self._proximity is not None
                nearest = self._proximity.query(points)
            clearances.append(float(np.min(nearest.distance_m)))
        return min(clearances)

    def predict(self, seed) -> ClosurePrediction:
        settings = self.inputs.config.section("closure_prediction")
        generation = self.inputs.config.section("candidate_generation")
        base = seed.object_from_hand_matrix()
        reference = seed.pregrasp_joint_positions_rad
        full_palm = generation.get("backend") == "GRASPGENX_FULL_PALM"
        if full_palm and seed.palm_configuration_rad is None:
            return self._failure(
                seed,
                seed.pregrasp_closure_phases,
                0.0,
                "PALM_CONFIGURATION_LOST_IN_PIPELINE",
            )
        if seed.palm_configuration_rad is not None:
            palm_index = self.inputs.hand_model.independent_joint_names.index("f1j1")
            if not np.isclose(
                reference[palm_index], seed.palm_configuration_rad, atol=1.0e-12
            ):
                return self._failure(
                    seed,
                    seed.pregrasp_closure_phases,
                    0.0,
                    "PALM_CONFIGURATION_LOST_IN_PIPELINE",
                )
        phases = list(seed.pregrasp_closure_phases)
        initial_clearance = self._initial_clearance(base, tuple(phases), reference)
        if initial_clearance <= float(settings["initial_clearance_m"]):
            return self._failure(seed, phases, initial_clearance, "INITIAL_PAD_TOO_CLOSE")

        contacts: list[PredictedContact] = []
        maximum = (
            float(generation["maximum_closure_phase"])
            if seed.maximum_closure_phase is None
            else float(seed.maximum_closure_phase)
        )
        if not 0.0 < maximum <= float(generation["maximum_closure_phase"]):
            raise ValueError("candidate closure phase exceeds the configured ceiling")
        contact_distance = float(settings["contact_distance_m"])
        for pad_name in settings["closing_order"]:
            phase_index = self._pad_to_phase[str(pad_name)]
            start = float(phases[phase_index])
            previous = start
            contact: PredictedContact | None = None
            motion_incompatible_contact_seen = False
            for phase in closure_phase_samples(
                self.inputs, tuple(phases), phase_index, maximum, reference
            ):
                phases[phase_index] = float(phase)
                selected, nearest, normals, inward = self._contact_at_phase(
                    str(pad_name), tuple(phases), base, reference
                )
                if nearest.intersecting:
                    return self._failure(
                        seed,
                        phases,
                        initial_clearance,
                        f"PAD_OBJECT_INTERSECTION_BEFORE_VALID_CONTACT_{pad_name}",
                        contacts,
                    )
                if selected < 0:
                    if float(np.min(nearest.distance_m)) <= contact_distance:
                        motion_incompatible_contact_seen = True
                    previous = float(phase)
                    continue
                distance = float(nearest.distance_m[selected])
                if distance <= contact_distance:
                    face_index = int(nearest.face_index[selected])
                    if not self.inputs.face_roles.face_is_allowed[face_index]:
                        return self._failure(
                            seed,
                            phases,
                            initial_clearance,
                            f"FORBIDDEN_FIRST_CONTACT_FACE_{face_index}",
                            contacts,
                        )
                    contact = PredictedContact(
                        pad_name=str(pad_name),
                        object_position_m=tuple(
                            float(value) for value in nearest.point_m[selected]
                        ),
                        path_local_free_side_normal_object=tuple(
                            float(value) for value in normals[selected]
                        ),
                        object_face_index=face_index,
                        phase_lower=previous,
                        phase_upper=float(phase),
                        clearance_m=distance,
                        inward_motion_m_per_phase=float(inward[selected]),
                    )
                    break
                previous = float(phase)
            if contact is None:
                reason = (
                    f"NO_MOTION_COMPATIBLE_PAD_POINT_{pad_name}"
                    if motion_incompatible_contact_seen
                    else f"NO_EFFECTIVE_CONTACT_{pad_name}"
                )
                return self._failure(
                    seed,
                    phases,
                    initial_clearance,
                    reason,
                    contacts,
                )
            contacts.append(contact)

        joints = joint_positions_for_phases(
            self.inputs, tuple(phases), reference_joint_positions_rad=reference
        )
        planned = tuple(
            PlannedPadContact(
                pad_name=contact.pad_name,
                position_object_m=contact.object_position_m,
                path_local_free_side_normal_object=(
                    contact.path_local_free_side_normal_object
                ),
                surface_coordinates=(contact.phase_lower, contact.phase_upper),
            )
            for contact in contacts
        )
        grasp = GraspCandidate.from_matrix(
            object_from_hand=base,
            independent_joint_positions_rad=joints,
            planned_pad_contacts=planned,
            internal_normal_forces_n=(0.0, 0.0, 0.0),
        )
        return ClosurePrediction(
            seed=seed,
            status="CLOSURE_SURVIVE",
            contacts=tuple(contacts),
            final_joint_positions_rad=tuple(float(value) for value in joints),
            final_closure_phases=tuple(float(value) for value in phases),
            minimum_initial_pad_clearance_m=initial_clearance,
            grasp_candidate=grasp,
        )

    def _failure(
        self,
        seed,
        phases,
        clearance: float,
        reason: str,
        contacts=(),
    ) -> ClosurePrediction:
        joints = joint_positions_for_phases(
            self.inputs, tuple(phases),
            reference_joint_positions_rad=seed.pregrasp_joint_positions_rad,
        )
        return ClosurePrediction(
            seed=seed,
            status="CLOSURE_REJECT",
            contacts=tuple(contacts),
            final_joint_positions_rad=tuple(float(value) for value in joints),
            final_closure_phases=tuple(float(value) for value in phases),
            minimum_initial_pad_clearance_m=float(clearance),
            reason=reason,
        )


__all__ = ["SequentialClosurePredictor", "closure_phase_samples"]
