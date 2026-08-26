"""Bounded joint pose refinement for one-against-two opposition anchors."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import time
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from kcg_connector.grasp.carts_v2.full_palm_search import bind_pregrasp
from kcg_connector.grasp.carts_v2.height_projected_search import (
    sampled_height_path_states,
    sampled_table_path_requirement,
)
from kcg_connector.grasp.carts_v2.height_projection import (
    minimum_z_over_finite_table_top,
)
from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed,
    V2Inputs,
    joint_positions_for_phases,
)
from kcg_connector.grasp.carts_v2.surface_contact import (
    ExactContactSurfaceQuery,
    nearest_motion_compatible_index,
)
from kcg_connector.grasp.carts_v2.task_grip_surface import (
    allowed_object_grasp_center_m,
    motion_compatible_with_object_witness,
)


REFINEMENT_VARIABLES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "dq_p")


def finite_refinement_vector(
    value: Sequence[float], length: int, label: str
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be one finite {length}-vector")
    return result


def three_registered_contacts(inputs: V2Inputs, prediction) -> bool:
    expected = {pad.name for pad in inputs.hand_contract.pads}
    return (
        prediction.status == "CLOSURE_SURVIVE"
        and len(prediction.contacts) == 3
        and {contact.pad_name for contact in prediction.contacts} == expected
    )


def initial_refinement_population(bounds, witness, safe, seed) -> np.ndarray:
    lower = np.asarray([row[0] for row in bounds], dtype=np.float64)
    upper = np.asarray([row[1] for row in bounds], dtype=np.float64)
    span = upper - lower
    rows = [np.zeros(7), witness, safe]
    for center in (witness, safe):
        for index in (0, 1, 3, 4, 5, 6):
            for sign in (-1.0, 1.0):
                value = np.array(center, copy=True)
                value[index] += sign * 0.08 * span[index]
                rows.append(np.clip(value, lower, upper))
    sampler = qmc.Sobol(d=7, scramble=True, seed=int(seed))
    sobol = qmc.scale(sampler.random_base2(5), lower, upper)
    rows.extend(sobol)
    unique = []
    for row in rows:
        clipped = np.clip(np.asarray(row, dtype=np.float64), lower, upper)
        if not any(np.array_equal(clipped, old) for old in unique):
            unique.append(clipped)
        if len(unique) == 28:
            break
    if len(unique) != 28:
        raise RuntimeError("deterministic refinement population is incomplete")
    return np.asarray(unique, dtype=np.float64)


def ranked_refinement_population(records, fallback: np.ndarray) -> np.ndarray:
    ranked = sorted(records, key=lambda row: (
        np.inf if row["maximum_contact_gap_m"] is None
        else float(row["maximum_contact_gap_m"]),
        np.inf if row["contact_gap_range_m"] is None
        else float(row["contact_gap_range_m"]),
        row["candidate_id"],
    ))
    rows = [np.asarray([row["variables"][name] for name in REFINEMENT_VARIABLES])
            for row in ranked]
    rows.extend(np.asarray(fallback, dtype=np.float64))
    unique = []
    for row in rows:
        if not any(np.array_equal(row, old) for old in unique):
            unique.append(row)
        if len(unique) == 28:
            break
    if len(unique) != 28:
        raise RuntimeError("stage-B deterministic population is incomplete")
    return np.asarray(unique, dtype=np.float64)


def _sparse_table_proxy(inputs, seed, final_phases, required_clearance):
    """Use stratified real-mesh states only to order expensive exact checks."""

    states = tuple(sampled_height_path_states(inputs, seed, final_phases))
    groups: dict[str, list[int]] = {}
    for index, (stage, _base, _joints) in enumerate(states):
        if stage.startswith("PALM_FAR_"):
            group = "PALM"
        elif stage.startswith("PRESHAPE_FAR_"):
            group = "PRESHAPE"
        elif stage.startswith(("APPROACH_", "PREGRASP")):
            group = "APPROACH"
        elif stage.startswith(("FINGER_1_", "CONTACT_STOP_1")):
            group = "FINGER_1"
        elif stage.startswith(("FINGER_2_", "CONTACT_STOP_2")):
            group = "FINGER_2"
        elif stage.startswith(("FINGER_3_", "CONTACT_STOP_3")):
            group = "FINGER_3"
        else:
            group = "LIFT_START"
        groups.setdefault(group, []).append(index)
    selected = set()
    for indices in groups.values():
        count = min(7, len(indices))
        selected.update(indices[index] for index in np.linspace(
            0, len(indices) - 1, count, dtype=np.int64
        ))
    minimum = None
    minimum_stage = ""
    for index in sorted(selected):
        stage, base, joints = states[index]
        transforms = inputs.hand_model.forward_kinematics(joints, base_transform=base)
        for link, triangles in inputs.hand_collision_triangles_by_link.items():
            transform = transforms[link]
            world = triangles @ transform[:3, :3].T + transform[:3, 3]
            value, _face = minimum_z_over_finite_table_top(
                world, inputs.table_xy_bounds_m
            )
            if value is not None and (minimum is None or value < minimum):
                minimum, minimum_stage = float(value), str(stage)
    if minimum is None:
        return None, None, minimum_stage, len(selected), len(states)
    clearance = minimum - float(inputs.table_top_z_m)
    return (clearance, max(0.0, float(required_clearance) - clearance),
            minimum_stage, len(selected), len(states))


class OppositionPoseEvaluator:
    """Cache exact contact-gap and complete sampled-table evaluations."""

    def __init__(self, inputs, anchor, stop_phases, bounds, budget, deadline):
        self.inputs = inputs
        if inputs.task_grip_surfaces is None:
            raise ValueError("opposition refinement requires TASK_GRIP_SURFACE")
        self.anchor = anchor
        self.stop_phases = tuple(float(value) for value in stop_phases)
        self.bounds = tuple((float(a), float(b)) for a, b in bounds)
        self.budget = int(budget)
        self.deadline = float(deadline)
        self.query = ExactContactSurfaceQuery(inputs)
        self.center = allowed_object_grasp_center_m(inputs)
        self.cache: dict[bytes, dict[str, object]] = {}
        self.stop_reason = ""
        self.task_frame = np.asarray(
            inputs.object_contract.task_frame_rotation_object, dtype=np.float64
        )
        self.anchor_pose = anchor.object_from_hand_matrix()
        approach = np.asarray(anchor.approach_direction_object, dtype=np.float64)
        self.local_approach = self.anchor_pose[:3, :3].T @ approach
        self.palm_index = inputs.hand_model.independent_joint_names.index("f1j1")
        self.contact_distance = float(
            inputs.config.section("closure_prediction")["contact_distance_m"]
        )
        self.motion_step = float(
            inputs.config.section("closure_prediction")[
                "motion_derivative_phase_step"
            ]
        )
        self.minimum_motion = float(
            inputs.config.section("closure_prediction")[
                "minimum_inward_motion_m_per_phase"
            ]
        )
        self.required_table = float(
            inputs.config.section("height_projection")["table_operation_clearance_m"]
        )

    def _candidate(self, vector: np.ndarray) -> CandidateSeed:
        pose = np.array(self.anchor_pose, copy=True)
        pose[:3, :3] = self.anchor_pose[:3, :3] @ Rotation.from_euler(
            "xyz", vector[3:6]
        ).as_matrix()
        pose[:3, 3] += self.task_frame @ vector[:3]
        palm = float(self.anchor.palm_configuration_rad + vector[6])
        reference = np.asarray(
            self.anchor.pregrasp_joint_positions_rad, dtype=np.float64
        ).copy()
        reference[self.palm_index] = palm
        suffix = hashlib.sha256(np.asarray(vector, dtype="<f8").tobytes()).hexdigest()[:12]
        approach = pose[:3, :3] @ self.local_approach
        seed = replace(
            self.anchor,
            candidate_id=f"{self.anchor.candidate_id}_r{suffix}",
            object_from_hand=tuple(float(value) for value in pose.ravel()),
            pregrasp_joint_positions_rad=tuple(float(value) for value in reference),
            palm_configuration_rad=palm,
            approach_direction_object=tuple(float(value) for value in approach),
        )
        return bind_pregrasp(self.inputs, seed, seed.pregrasp_closure_phases)

    def _surface_gap(self, seed: CandidateSeed) -> tuple[list[float], str]:
        gaps = []
        for phase_index, (pad_name, surface) in enumerate(
            sorted(self.inputs.task_grip_surfaces.items())
        ):
            phases = list(self.stop_phases)
            delta = min(self.motion_step, 1.0 - phases[phase_index])
            if delta <= 0.0:
                return [], f"NO_CLOSING_DERIVATIVE:{pad_name}"
            current_positions = self._joint_positions(seed, tuple(phases))
            phases[phase_index] += delta
            moved_positions = self._joint_positions(seed, tuple(phases))
            current = self.inputs.hand_model.forward_kinematics(
                current_positions, base_transform=seed.object_from_hand_matrix()
            )[surface.link_name]
            moved = self.inputs.hand_model.forward_kinematics(
                moved_positions, base_transform=seed.object_from_hand_matrix()
            )[surface.link_name]
            nearest, pad_points, object_normals = self.query.query_task_surface_witnesses(
                pad_name, current, 16
            )
            if nearest.intersecting:
                return [], f"TASK_SURFACE_OBJECT_INTERSECTION:{pad_name}"
            local = (pad_points - current[:3, 3]) @ current[:3, :3]
            moved_points = local @ moved[:3, :3].T + moved[:3, 3]
            motion = (moved_points - pad_points) / delta
            assert nearest.surface_normal_m is not None
            compatible = motion_compatible_with_object_witness(
                pad_points,
                nearest.point_m,
                nearest.surface_normal_m,
                object_normals,
                motion,
                self.center,
                self.minimum_motion,
            )
            inward = -np.einsum("ij,ij->i", motion, object_normals)
            selected = nearest_motion_compatible_index(
                nearest.distance_m,
                np.where(compatible, inward, -np.inf),
                self.minimum_motion,
            )
            if selected < 0:
                return [], f"NO_MOTION_COMPATIBLE_TASK_GRIP_SURFACE:{pad_name}"
            distance = float(nearest.distance_m[selected])
            forbidden = nearest.forbidden_distance_m
            tolerance = 64.0 * np.finfo(np.float64).eps
            if forbidden is not None and forbidden <= distance + tolerance:
                return [], (
                    f"FORBIDDEN_OBJECT_FIRST_CONTACT:{pad_name}:"
                    f"{nearest.forbidden_face_index}"
                )
            gaps.append(distance)
        return gaps, ""

    def _joint_positions(self, seed: CandidateSeed, phases):
        return joint_positions_for_phases(
            self.inputs,
            phases,
            reference_joint_positions_rad=seed.pregrasp_joint_positions_rad,
        )

    def evaluate(
        self, raw_vector: Sequence[float], *, with_table: bool = False
    ) -> dict[str, object]:
        vector = finite_refinement_vector(raw_vector, 7, "refinement vector")
        key = np.asarray(vector, dtype="<f8").tobytes()
        if key in self.cache:
            row = self.cache[key]
            if not with_table or row["table_evaluated"] or row["reason"]:
                return row
            if time.monotonic() >= self.deadline:
                self.stop_reason = "MAXIMUM_WALL_TIME"
                raise RuntimeError(self.stop_reason)
            self._bind_table_result(row)
            return row
        if len(self.cache) >= self.budget or time.monotonic() >= self.deadline:
            self.stop_reason = (
                "MAXIMUM_EVALUATION_BUDGET" if len(self.cache) >= self.budget
                else "MAXIMUM_WALL_TIME"
            )
            raise RuntimeError(self.stop_reason)
        seed = self._candidate(vector)
        gaps, reason = self._surface_gap(seed)
        maximum_gap = None if reason else float(max(gaps))
        gap_range = None if reason else float(max(gaps) - min(gaps))
        contact_witness = bool(maximum_gap is not None
                               and maximum_gap <= self.contact_distance)
        row = {
            "evaluation_index": len(self.cache),
            "candidate_id": seed.candidate_id,
            "variables": {name: float(value) for name, value in zip(
                REFINEMENT_VARIABLES, vector)},
            "status": (
                "A_FIXED_PHASE_GAP_WITNESS" if contact_witness
                else "A_GEOMETRIC_GUIDANCE" if not reason else "A_HARD_REJECT"
            ),
            "reason": reason,
            "contact_gap_by_finger_m": gaps,
            "maximum_contact_gap_m": maximum_gap,
            "contact_gap_range_m": gap_range,
            "table_evaluated": False,
            "proxy_minimum_table_clearance_m": None,
            "proxy_table_deficit_to_required_clearance_m": None,
            "proxy_minimum_table_stage": "",
            "proxy_checked_state_count": 0,
            "full_path_state_count": 0,
            "candidate": seed,
        }
        self.cache[key] = row
        if with_table and not reason:
            self._bind_table_result(row)
        return row

    def _bind_table_result(self, row: dict[str, object]) -> None:
        seed = row["candidate"]
        clearance, deficit, stage, checked, full_count = _sparse_table_proxy(
            self.inputs, seed, self.stop_phases, self.required_table
        )
        row["table_evaluated"] = True
        row["proxy_minimum_table_stage"] = stage or ""
        row["proxy_checked_state_count"] = checked
        row["full_path_state_count"] = full_count
        row["proxy_minimum_table_clearance_m"] = clearance
        row["proxy_table_deficit_to_required_clearance_m"] = deficit

    def exact_table_replay(self, seed: CandidateSeed) -> dict[str, object]:
        requirement, stage, checked = sampled_table_path_requirement(
            self.inputs, seed, self.stop_phases, self.required_table
        )
        world_z = float(
            (self.inputs.frozen_world_from_object
             @ seed.object_from_hand_matrix())[2, 3]
        )
        minimum = requirement.minimum_handbase_z_m
        clearance = None if minimum is None else (
            self.required_table + world_z - float(minimum)
        )
        return {
            "minimum_table_clearance_m": clearance,
            "table_deficit_to_required_clearance_m": (
                None if minimum is None else max(0.0, float(minimum) - world_z)
            ),
            "minimum_table_stage": stage or "",
            "checked_state_count": checked,
        }


def refinement_public_row(row: dict[str, object]) -> dict[str, object]:
    result = dict(row)
    seed = result.pop("candidate")
    result["object_from_hand_row_major"] = list(seed.object_from_hand)
    result["pregrasp_joint_positions_rad"] = list(seed.pregrasp_joint_positions_rad)
    result["pregrasp_closure_phases"] = list(seed.pregrasp_closure_phases)
    result["palm_configuration_rad"] = seed.palm_configuration_rad
    return result


__all__ = [
    "OppositionPoseEvaluator", "REFINEMENT_VARIABLES",
    "finite_refinement_vector", "initial_refinement_population",
    "ranked_refinement_population", "refinement_public_row",
    "three_registered_contacts",
]
