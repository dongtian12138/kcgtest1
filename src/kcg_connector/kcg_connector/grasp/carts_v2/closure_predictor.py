"""Predict registered sequential finger closure with first PAD contact stopping."""

from __future__ import annotations

import numpy as np

from kcg_connector.grasp.carts_v2.models import (
    ClosurePrediction,
    PredictedContact,
    V2Inputs,
    farthest_point_indices,
    joint_positions_for_phases,
)
from kcg_connector.grasp.carts_v2.surface_contact import (
    ExactPadSurfaceQuery,
    LegacyMeshProximityIndex,
    NearestSurface,
    nearest_motion_compatible_index,
)
from kcg_connector.grasp.carts_v2.task_grip_surface import (
    allowed_object_grasp_center_m,
)
from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
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
        if backend not in {
            "LOCAL_CENTROID_BASELINE",
            "FCL_EXACT_PAD_MESH",
            "FCL_EXACT_TASK_GRIP_SURFACE",
        }:
            raise ValueError(f"unknown closure surface query backend {backend!r}")
        if (backend == "FCL_EXACT_TASK_GRIP_SURFACE") != (
            inputs.task_grip_surfaces is not None
        ):
            raise ValueError("TASK_GRIP_SURFACE backend and hand variant must be bound together")
        self._surface_query_backend = backend
        self._proximity = (
            LegacyMeshProximityIndex(inputs)
            if backend == "LOCAL_CENTROID_BASELINE"
            else None
        )
        self._exact_pad_mesh = (
            ExactPadSurfaceQuery(inputs)
            if backend in {"FCL_EXACT_PAD_MESH", "FCL_EXACT_TASK_GRIP_SURFACE"}
            else None
        )
        self._task_surface_mode = inputs.task_grip_surfaces is not None
        self._object_grasp_center_m = (
            allowed_object_grasp_center_m(inputs)
            if self._task_surface_mode else None
        )
        count = int(
            inputs.config.section("closure_prediction")["pad_surface_sample_count"]
        )
        self._pad_points: dict[str, np.ndarray] = {}
        geometry = (
            {
                pad.name: (pad.points_local_m, pad.faces)
                for pad in inputs.hand_contract.pads
            }
            if inputs.task_grip_surfaces is None
            else {
                name: (surface.points_local_m, surface.faces)
                for name, surface in inputs.task_grip_surfaces.items()
            }
        )
        for name, (points, faces) in geometry.items():
            centroids = np.mean(points[faces], axis=1)
            samples = np.vstack((points, centroids))
            self._pad_points[name] = _farthest_surface_points(samples, count)
        self._pad_to_phase = {
            pad.name: index for index, pad in enumerate(inputs.hand_contract.pads)
        }

    def _world_pad_points(
        self, pad_name: str, phases: tuple[float, float, float], base: np.ndarray,
        reference_joint_positions_rad: tuple[float, ...],
    ) -> np.ndarray:
        transform = self._pad_transform(
            pad_name, phases, base, reference_joint_positions_rad
        )
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
        if self.inputs.task_grip_surfaces is not None:
            link_name = self.inputs.task_grip_surfaces[pad_name].link_name
            return self.inputs.hand_model.forward_kinematics(
                joints, base_transform=base
            )[link_name]
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
        if self._task_surface_mode:
            nearest = None
        else:
            nearest, pad_point, normal = self._exact_pad_mesh.query_pad(pad_name, current)
        phase_index = self._pad_to_phase[pad_name]
        delta = min(
            float(self.inputs.config.section("closure_prediction")[
                "motion_derivative_phase_step"
            ]),
            1.0 - phases[phase_index],
        )
        if delta <= 0.0:
            if nearest is None:
                nearest, _points, normals = self._exact_pad_mesh.query_task_surface_witnesses(
                    pad_name, current, 16)
            else:
                normals = np.asarray(normal, dtype=np.float64)[None, :]
            return -1, nearest, normals, np.full(len(normals), -np.inf)
        moved_phases = list(phases)
        moved_phases[phase_index] += delta
        moved = self._pad_transform(
            pad_name, tuple(moved_phases), base, reference_joint_positions_rad
        )
        minimum = float(self.inputs.config.section("closure_prediction")[
            "minimum_inward_motion_m_per_phase"
        ])
        if self._task_surface_mode:
            assert self._object_grasp_center_m is not None
            return self._exact_pad_mesh.select_task_surface_contact(
                pad_name, current, moved, self._object_grasp_center_m, minimum,
                delta, float(self.inputs.config.section("closure_prediction")[
                    "contact_distance_m"]),
            )
        pad_points = np.asarray(pad_point, dtype=np.float64)[None, :]
        normals = np.asarray(normal, dtype=np.float64)[None, :]
        pad_local = (pad_points - current[:3, 3]) @ current[:3, :3]
        moved_points = pad_local @ moved[:3, :3].T + moved[:3, 3]
        inward = -np.einsum("ij,ij->i", (moved_points - pad_points) / delta, normals)
        compatible = inward >= minimum
        if compatible[0] or nearest.distance_m[0] > float(
                self.inputs.config.section("closure_prediction")["contact_distance_m"]):
            return (0 if compatible[0] else -1), nearest, normals, inward
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
                if nearest.forbidden_first_contact:
                    face = nearest.forbidden_face_index
                    return self._failure(
                        seed, phases, initial_clearance,
                        f"FORBIDDEN_OBJECT_FIRST_CONTACT_FACE_{face}", contacts,
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
                        hand_surface_face_index=(
                            None
                            if nearest.surface_face_index is None
                            else int(nearest.surface_face_index[selected])
                        ),
                        hand_surface_normal_object=(
                            None
                            if nearest.surface_normal_m is None
                            else tuple(
                                float(value)
                                for value in nearest.surface_normal_m[selected]
                            )
                        ),
                        hand_surface_legacy_blue_pad=(
                            None if nearest.surface_legacy_blue_pad is None
                            else bool(nearest.surface_legacy_blue_pad[selected])
                        ),
                    )
                    break
                previous = float(phase)
            if contact is None:
                reason = (
                    (
                        f"NO_MOTION_COMPATIBLE_TASK_GRIP_SURFACE_{pad_name}"
                        if self._task_surface_mode
                        else f"NO_MOTION_COMPATIBLE_PAD_POINT_{pad_name}"
                    )
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
