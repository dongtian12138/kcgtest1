"""Predict registered sequential finger closure with first PAD contact stopping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from kcg_connector.grasp.carts_v2.models import (
    ClosurePrediction,
    PredictedContact,
    V2Inputs,
    farthest_point_indices,
    joint_positions_for_phases,
)
from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
)


@dataclass(frozen=True)
class _NearestSurface:
    point_m: np.ndarray
    distance_m: np.ndarray
    face_index: np.ndarray


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
    """Small approximate BVH: centroid tree followed by exact local triangles."""

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

    def query(self, points_m: np.ndarray) -> _NearestSurface:
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
        return _NearestSurface(
            point_m=local_closest[rows, selected],
            distance_m=local_distance[rows, selected],
            face_index=face_ids[rows, selected],
        )


def _farthest_surface_points(points: np.ndarray, count: int) -> np.ndarray:
    if count >= len(points):
        return np.array(points, copy=True)
    return np.asarray(points[farthest_point_indices(points, count)], dtype=np.float64)


class SequentialClosurePredictor:
    """Reusable object index and real PAD surface samples for many candidates."""

    def __init__(self, inputs: V2Inputs) -> None:
        self.inputs = inputs
        self._proximity = _MeshProximityIndex(inputs)
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

    def _contact_at_phase(
        self,
        pad_name: str,
        phases: tuple[float, float, float],
        base: np.ndarray,
        reference_joint_positions_rad: tuple[float, ...],
    ) -> tuple[int, _NearestSurface, np.ndarray, np.ndarray]:
        points = self._world_pad_points(
            pad_name, phases, base, reference_joint_positions_rad
        )
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
        selected = int(np.argmin(eligible_distance))
        return selected, nearest, normals, inward

    def _initial_clearance(
        self,
        seed_base: np.ndarray,
        phases: tuple[float, ...],
        reference: tuple[float, ...],
    ) -> float:
        clearances = []
        for pad in self.inputs.hand_contract.pads:
            points = self._world_pad_points(pad.name, phases, seed_base, reference)
            clearances.append(float(np.min(self._proximity.query(points).distance_m)))
        return min(clearances)

    def predict(self, seed) -> ClosurePrediction:
        settings = self.inputs.config.section("closure_prediction")
        generation = self.inputs.config.section("candidate_generation")
        base = seed.object_from_hand_matrix()
        reference = seed.pregrasp_joint_positions_rad
        phases = list(seed.pregrasp_closure_phases)
        initial_clearance = self._initial_clearance(base, tuple(phases), reference)
        if initial_clearance <= float(settings["initial_clearance_m"]):
            return self._failure(seed, phases, initial_clearance, "INITIAL_PAD_TOO_CLOSE")

        contacts: list[PredictedContact] = []
        maximum = float(generation["maximum_closure_phase"])
        contact_distance = float(settings["contact_distance_m"])
        sample_count = int(settings["phase_sample_count"])
        for pad_name in settings["closing_order"]:
            phase_index = self._pad_to_phase[str(pad_name)]
            start = float(phases[phase_index])
            previous = start
            contact: PredictedContact | None = None
            for phase in np.linspace(start, maximum, sample_count)[1:]:
                phases[phase_index] = float(phase)
                selected, nearest, normals, inward = self._contact_at_phase(
                    str(pad_name), tuple(phases), base, reference
                )
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
                return self._failure(
                    seed,
                    phases,
                    initial_clearance,
                    f"NO_EFFECTIVE_CONTACT_{pad_name}",
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


__all__ = ["SequentialClosurePredictor"]
