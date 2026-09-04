#!/usr/bin/env python3
"""Complete evaluation of a frozen finite nail-free three-finger grasp set."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import fcl
import numpy as np
from scipy.optimize import linprog
import yaml

import controller as control
from kcg_connector.grasp.carts_v2.models import V2Inputs, load_v2_inputs
from kcg_connector.grasp.carts_v2.task_grip_surface import (
    generate_axial_pad_intersection_grasp,
    task_noncontact_triangles,
)
from kcg_connector.grasp.robust.bounded_hand_base_ik import (
    bounded_ik_settings,
    pregrasp_seeds,
)
from kcg_connector.grasp.robust.collision_roster import (
    load_authoritative_collision_link_roster,
)
from kcg_connector.grasp.robust.hand_model import rpy_rotation
from kcg_connector.grasp.robust.object_model import load_stl_mesh
from kcg_connector.grasp.robust.surface_atlas import (
    load_step_triangle_role_partition,
)


PAD_ORDER = ("finger_1_pad", "finger_2_pad", "finger_3_pad")
CLOSING_JOINTS = ("f1j2", "f2j1", "f3j2")
EXPECTED_VARIANT = "DIRECT_USER_NAILFREE_STL"
EXPECTED_METHOD = "COMPLETE_AXIS_ALIGNED_FULL_PAD_GRID_V4"
ARM_SELF_CONTACT_ENVELOPE_M = 2.0 * 5.0e-5
ROBUST_SELECTION_METRICS = (
    "worst_minimum_forbidden_clearance_m",
    "worst_task_load_factor",
    "worst_minimum_commanded_hand_joint_limit_margin_rad",
    "worst_minimum_closure_reserve_rad",
)


@dataclass(frozen=True)
class GridPoint:
    candidate_id: str
    palm_index: int
    yaw_index: int
    axial_index: int
    palm_rad: float
    yaw_rad: float
    axial_m: float
    is_nominal_baseline_point: bool


@dataclass(frozen=True)
class PadPatch:
    pad_name: str
    link_name: str
    points_object_m: np.ndarray
    normals_object: np.ndarray
    object_face_indices: np.ndarray
    penetration_depths_m: np.ndarray


@dataclass(frozen=True)
class PadResultant:
    pad_name: str
    point_object_m: np.ndarray
    normal_object: np.ndarray
    collision_witness_count: int


@dataclass(frozen=True)
class PathEvaluation:
    contact_joint_positions_rad: np.ndarray
    patches: tuple[PadPatch, PadPatch, PadPatch]
    minimum_closure_reserve_rad: float
    minimum_commanded_hand_joint_limit_margin_rad: float
    minimum_forbidden_clearance_m: float


@dataclass(frozen=True)
class TaskQuality:
    task_load_factor: float
    hold_load_factor: float
    lift_load_factor: float
    limiting_stage: str
    maximum_joint_torque_at_unit_task_nm: float
    minimum_joint_torque_margin_at_unit_task_nm: float
    maximum_pad_normal_force_at_unit_task_n: float
    minimum_pad_normal_force_margin_at_unit_task_n: float
    required_closing_joint_effort_nm: tuple[float, float, float]
    wrench_span_rank: int
    force_closure_friction_pyramid: bool


@dataclass(frozen=True)
class OfflineRobustScenario:
    name: str
    translation_world_m: np.ndarray
    yaw_world_rad: float
    friction_coefficient: float | None
    mass_scale: float
    center_of_mass_delta_object_m: np.ndarray
    finger_3_joint_target_offset_rad: float
    offline_components: tuple[str, ...]
    dynamic_only_components: tuple[str, ...]


def _arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object-id", default="current_d38999_26kj61sn_public_spec"
    )
    parser.add_argument(
        "--config",
        default=str(repository / "src/kcg_connector/config/carts_grasp_v2.yaml"),
    )
    parser.add_argument(
        "--selection-mode",
        choices=("nominal", "robust", "friction_margin"),
        default="nominal",
        help=(
            "Keep the frozen nominal selection by default, or replay the same "
            "G_delta over configured offline-expressible robustness scenarios, "
            "or select the executable member with the lowest finite-grid task "
            "friction requirement."
        ),
    )
    parser.add_argument(
        "--replay-candidate-id",
        help=(
            "Evaluate and materialize exactly one declared G_delta member; "
            "this mode makes no finite-set optimality claim."
        ),
    )
    parser.add_argument("--output-directory", required=True)
    return parser.parse_args()


def _rotation_about_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


_OFFLINE_ROBUSTNESS_COMPONENTS = frozenset(
    (
        "object_translation_delta_m",
        "object_yaw_delta_rad",
        "contact_friction_coefficient",
        "object_mass_scale",
        "center_of_mass_delta_object_m",
        "finger_joint_target_offset_rad",
    )
)
_DYNAMIC_ONLY_ROBUSTNESS_COMPONENTS = frozenset(("finger_preload_scale",))


def _load_offline_robust_scenarios(
    inputs: V2Inputs,
) -> tuple[OfflineRobustScenario, ...]:
    robustness = inputs.config.section("robustness_evaluation")
    scenarios = robustness.get("scenarios")
    if (
        robustness.get("frozen_before_first_dynamic_run") is not True
        or not isinstance(scenarios, Mapping)
        or next(iter(scenarios), None) != "nominal"
        or scenarios.get("nominal") != {}
    ):
        raise ValueError("frozen robustness scenarios require nominal first")
    known = (
        _OFFLINE_ROBUSTNESS_COMPONENTS
        | _DYNAMIC_ONLY_ROBUSTNESS_COMPONENTS
    )
    result = []
    for raw_name, raw_value in scenarios.items():
        name, value = str(raw_name), dict(raw_value)
        unknown = set(value) - known
        translation = np.asarray(
            value.get("object_translation_delta_m", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        com_delta = np.asarray(
            value.get("center_of_mass_delta_object_m", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        joint = value.get("finger_joint_target_offset_rad", {})
        preload = value.get("finger_preload_scale", {})
        yaw = float(value.get("object_yaw_delta_rad", 0.0))
        friction_value = value.get("contact_friction_coefficient")
        friction = None if friction_value is None else float(friction_value)
        mass_scale = float(value.get("object_mass_scale", 1.0))
        if (
            unknown
            or not name
            or (name != "nominal" and not value)
            or translation.shape != (3,)
            or com_delta.shape != (3,)
            or not np.all(np.isfinite(translation))
            or not np.all(np.isfinite(com_delta))
            or np.any(translation[1:] != 0.0)
            or np.any(com_delta[1:] != 0.0)
            or not isinstance(joint, Mapping)
            or set(joint) - {"finger_3"}
            or not isinstance(preload, Mapping)
            or set(preload) - {"finger_3"}
            or not math.isfinite(yaw)
            or not math.isfinite(mass_scale)
            or mass_scale <= 0.0
            or friction is not None
            and (not math.isfinite(friction) or friction < 0.0)
        ):
            raise ValueError(f"robustness scenario {name!r} is malformed")
        result.append(
            OfflineRobustScenario(
                name=name,
                translation_world_m=translation,
                yaw_world_rad=yaw,
                friction_coefficient=friction,
                mass_scale=mass_scale,
                center_of_mass_delta_object_m=com_delta,
                finger_3_joint_target_offset_rad=float(
                    joint.get("finger_3", 0.0)
                ),
                offline_components=tuple(key for key in value if key in _OFFLINE_ROBUSTNESS_COMPONENTS),
                dynamic_only_components=tuple(key for key in value if key in _DYNAMIC_ONLY_ROBUSTNESS_COMPONENTS),
            )
        )
    return tuple(result)
def _anchored_grid(
    lower: float,
    upper: float,
    step: float,
    anchor: float,
    *,
    include_endpoints: bool,
) -> np.ndarray:
    if not all(math.isfinite(value) for value in (lower, upper, step, anchor)):
        raise ValueError("finite grid contains a non-finite value")
    if step <= 0.0 or upper < lower or not lower <= anchor <= upper:
        raise ValueError("finite anchored grid bounds are malformed")
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, abs(lower), abs(upper), abs(anchor)
    )
    first = int(math.ceil((lower - anchor) / step - tolerance))
    last = int(math.floor((upper - anchor) / step + tolerance))
    values = [anchor + index * step for index in range(first, last + 1)]
    values = [
        lower if abs(value - lower) <= tolerance else
        upper if abs(value - upper) <= tolerance else
        anchor if abs(value - anchor) <= tolerance else
        float(value)
        for value in values
        if lower - tolerance <= value <= upper + tolerance
    ]
    if include_endpoints:
        values.extend((lower, upper))
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(float(value))
    if not result or not any(value == anchor for value in result):
        raise ValueError("anchored grid lost its exact anchor")
    return np.asarray(result, dtype=np.float64)


def _declared_grid(
    inputs: V2Inputs, settings: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[GridPoint]]:
    lower, upper = inputs.hand_model.joint_limit_vectors()
    palm_slot = inputs.hand_model.independent_joint_names.index("f1j1")
    palms = _anchored_grid(
        float(lower[palm_slot]),
        float(upper[palm_slot]),
        float(settings["palm_step_rad"]),
        float(settings["palm_anchor_rad"]),
        include_endpoints=bool(settings["include_palm_joint_limit_endpoints"]),
    )
    yaw_step = float(settings["object_yaw_step_rad"])
    yaw_anchor = float(settings["object_yaw_anchor_rad"])
    yaw_count = int(round(2.0 * math.pi / yaw_step))
    if yaw_count < 1 or not math.isclose(
        yaw_count * yaw_step, 2.0 * math.pi, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("object yaw step must exactly divide 2*pi")
    yaws = np.mod(
        yaw_anchor + yaw_step * np.arange(yaw_count, dtype=np.float64),
        2.0 * math.pi,
    )
    yaws[0] = yaw_anchor

    mesh = inputs.object_contract.model.mesh
    semantic_allowed = np.fromiter(
        (
            semantic in inputs.object_contract.model.allowed_contact_semantics
            for semantic in mesh.face_semantics
        ),
        dtype=np.bool_,
        count=len(mesh.faces),
    )
    allowed_vertices = np.asarray(
        mesh.face_vertices_m[semantic_allowed], dtype=np.float64
    )
    origin = np.asarray(
        inputs.object_contract.model.assembly_axis_origin_m, dtype=np.float64
    )
    basis = np.asarray(
        inputs.object_contract.task_frame_rotation_object, dtype=np.float64
    )
    vertices_task = (allowed_vertices - origin) @ basis
    axial_lower = float(np.min(vertices_task[..., 2]))
    axial_upper = float(np.max(vertices_task[..., 2]))
    com_task = (
        np.asarray(inputs.object_contract.model.com_m, dtype=np.float64) - origin
    ) @ basis
    if settings["object_axial_anchor"] != "OBJECT_CENTER_OF_MASS":
        raise ValueError("finite axial anchor identity changed")
    axial_anchor = float(com_task[2])
    axial = _anchored_grid(
        axial_lower,
        axial_upper,
        float(settings["object_axial_step_m"]),
        axial_anchor,
        include_endpoints=bool(settings["include_allowed_axial_band_endpoints"]),
    )

    points: list[GridPoint] = []
    for palm_index, palm in enumerate(palms):
        for axial_index, axial_m in enumerate(axial):
            for yaw_index, yaw in enumerate(yaws):
                candidate_id = (
                    f"p{palm_index:02d}_y{yaw_index:02d}_z{axial_index:02d}"
                )
                points.append(
                    GridPoint(
                        candidate_id=candidate_id,
                        palm_index=palm_index,
                        yaw_index=yaw_index,
                        axial_index=axial_index,
                        palm_rad=float(palm),
                        yaw_rad=float(yaw),
                        axial_m=float(axial_m),
                        is_nominal_baseline_point=(
                            float(palm) == float(settings["palm_anchor_rad"])
                            and float(yaw) == yaw_anchor
                            and float(axial_m) == axial_anchor
                        ),
                    )
                )
    return palms, yaws, axial, points


def _place_base_generation(
    inputs: V2Inputs,
    base_generation: Mapping[str, object],
    yaw_rad: float,
) -> dict[str, object]:
    """Rotate one unshifted (palm,z) generation and apply its exact table shift."""

    generation = copy.deepcopy(base_generation)
    plan = generation["control_plan"]
    evidence = generation["evidence"]
    pose = np.asarray(plan["object_from_hand_row_major"], dtype=np.float64).reshape(
        4, 4
    )
    points = np.asarray(
        evidence["predicted_contact_points_object_m"], dtype=np.float64
    )
    if yaw_rad != 0.0:
        base_rotation = np.array(pose[:3, :3], copy=True)
        hand_points = (points - pose[:3, 3]) @ base_rotation
        pose[:3, :3] = base_rotation @ _rotation_about_z(yaw_rad)
        points = hand_points @ pose[:3, :3].T + pose[:3, 3]

    pregrasp = np.asarray(plan["pregrasp_joint_positions_rad"], dtype=np.float64)
    hand_from_links = inputs.hand_model.forward_kinematics(pregrasp)
    world_from_hand = inputs.frozen_world_from_object @ pose
    minimum_world_z = math.inf
    limiting_link = ""
    for link_name, triangles in inputs.hand_collision_triangles_by_link.items():
        hand_from_link = hand_from_links[link_name]
        world_from_link = world_from_hand @ hand_from_link
        points_world = (
            np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
            @ world_from_link[:3, :3].T
            + world_from_link[:3, 3]
        )
        link_minimum = float(np.min(points_world[:, 2]))
        if link_minimum < minimum_world_z:
            minimum_world_z = link_minimum
            limiting_link = link_name
    initial_clearance = minimum_world_z - float(inputs.table_top_z_m)
    required_clearance = float(
        inputs.config.section("nominal_grasp_generation")[
            "minimum_pregrasp_hand_table_clearance_m"
        ]
    )
    shift_world_z = max(0.0, required_clearance - initial_clearance)
    if shift_world_z > 0.0:
        shift_object = inputs.frozen_world_from_object[:3, :3].T @ np.asarray(
            (0.0, 0.0, shift_world_z), dtype=np.float64
        )
        pose[:3, 3] += shift_object
        points += shift_object

    model = inputs.object_contract.model
    origin = np.asarray(model.assembly_axis_origin_m, dtype=np.float64)
    basis = np.asarray(
        inputs.object_contract.task_frame_rotation_object, dtype=np.float64
    )
    vertices_task = (np.asarray(model.mesh.vertices_m) - origin) @ basis
    points_task = (points - origin) @ basis
    if (
        np.any(points_task[:, 2] < float(np.min(vertices_task[:, 2])))
        or np.any(points_task[:, 2] > float(np.max(vertices_task[:, 2])))
    ):
        raise ValueError("table-cleared full-pad contact lies outside CAD axial bounds")

    plan["object_from_hand_row_major"] = pose.ravel().tolist()
    evidence["predicted_contact_points_object_m"] = points.tolist()
    evidence["hand_yaw_rad"] = float(yaw_rad)
    evidence["pregrasp_limiting_hand_link"] = limiting_link
    evidence["pregrasp_initial_hand_table_clearance_m"] = initial_clearance
    evidence["minimum_pregrasp_hand_table_clearance_m"] = required_clearance
    evidence["applied_hand_table_clearance_shift_world_z_m"] = shift_world_z
    return generation


def _fcl_model(triangles: np.ndarray):
    vertices = np.ascontiguousarray(
        np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    )
    faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
    model = fcl.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(vertices, faces)
    model.endModel()
    return model


class ArmSelfCollisionScene:
    """Non-adjacent iiwa collision meshes used by the frozen Isaac asset."""

    def __init__(self, inputs: V2Inputs) -> None:
        roster = load_authoritative_collision_link_roster(
            inputs.config.section("inputs")["collision_roster"],
            repository_root=inputs.repository_root,
        )
        self.objects = {}
        for link in roster.links:
            if (
                not link.link_name.startswith("iiwa_link_")
                or link.link_name == "iiwa_link_ee"
            ):
                continue
            mesh, provenance = load_stl_mesh(
                link.absolute_path, unit=link.unit, orient_outward=False
            )
            if provenance.source_sha256 != link.sha256:
                raise ValueError(f"arm collision mesh hash changed for {link.link_name}")
            triangles = np.asarray(mesh.face_vertices_m) * np.asarray(link.scale)
            triangles = (
                triangles @ rpy_rotation(link.origin_rpy_rad).T
                + np.asarray(link.origin_xyz_m)
            )
            self.objects[link.link_name] = fcl.CollisionObject(
                _fcl_model(triangles)
            )
        expected = {f"iiwa_link_{index}" for index in range(8)}
        if set(self.objects) != expected:
            raise ValueError("complete iiwa collision mesh set is unavailable")
        adjacent = {
            tuple(sorted((joint.parent_link, joint.child_link)))
            for joint in inputs.robot_model.joints.values()
            if joint.parent_link in self.objects and joint.child_link in self.objects
        }
        links = tuple(sorted(self.objects))
        self.pairs = tuple(
            (first, second)
            for first_index, first in enumerate(links)
            for second in links[first_index + 1 :]
            if tuple(sorted((first, second))) not in adjacent
        )
        self.inputs = inputs
        self.collision_request = fcl.CollisionRequest(
            num_max_contacts=1, enable_contact=False
        )
        self.distance_request = fcl.DistanceRequest(
            enable_nearest_points=False
        )

    def state_clearance(
        self, arm_positions: np.ndarray, hand_positions: np.ndarray
    ) -> tuple[float, tuple[str, str] | None]:
        complete = np.concatenate((arm_positions, hand_positions))
        transforms = self.inputs.robot_model.forward_kinematics(complete)
        for name, collision_object in self.objects.items():
            transform = transforms[name]
            collision_object.setTransform(
                fcl.Transform(transform[:3, :3], transform[:3, 3])
            )
        minimum = math.inf
        for first, second in self.pairs:
            if fcl.collide(
                self.objects[first],
                self.objects[second],
                self.collision_request,
                fcl.CollisionResult(),
            ):
                return 0.0, (first, second)
            distance = float(
                fcl.distance(
                    self.objects[first],
                    self.objects[second],
                    self.distance_request,
                    fcl.DistanceResult(),
                )
            )
            if not math.isfinite(distance) or distance < 0.0:
                raise RuntimeError("iiwa self-collision distance is unavailable")
            minimum = min(minimum, distance)
        return minimum, None


def _piecewise_joint_samples(
    waypoints: Sequence[Sequence[float]], samples_per_segment: int
) -> list[np.ndarray]:
    rows = [np.asarray(row, dtype=np.float64) for row in waypoints]
    if not rows:
        return []
    result = [rows[0]]
    for start, end in zip(rows, rows[1:]):
        result.extend(
            (1.0 - fraction) * start + fraction * end
            for fraction in np.linspace(0.0, 1.0, samples_per_segment)[1:]
        )
    return result


def _arm_route_states(
    inputs: V2Inputs, motion_plan: Mapping[str, object]
) -> list[np.ndarray]:
    dynamic = inputs.config.section("dynamic")
    physics_dt = float(dynamic["physics_dt_s"])
    home_steps = round(float(dynamic["approach_above_duration_s"]) / physics_dt)
    approach = np.asarray(
        motion_plan["approach_arm_waypoints_rad"], dtype=np.float64
    )
    home_route = [
        control.minimum_jerk_blend(index / home_steps) * approach[0]
        for index in range(home_steps + 1)
    ]
    samples_per_segment = int(
        inputs.config.section("finite_cad_search")["approach_path_sample_count"]
    )
    return (
        home_route
        + _piecewise_joint_samples(approach, samples_per_segment)[1:]
        + _piecewise_joint_samples(
            motion_plan["lift_arm_waypoints_rad"], samples_per_segment
        )[1:]
    )


def _select_arm_route(
    repository: Path,
    inputs: V2Inputs,
    control_plan: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Choose a collision-free redundant IK branch without weighted scoring."""

    solver_settings = inputs.config.section("ik")["solver"]
    settings = bounded_ik_settings(solver_settings)
    lower, upper = inputs.robot_model.joint_limit_vectors()
    lower = np.asarray(lower[:7], dtype=np.float64)
    upper = np.asarray(upper[:7], dtype=np.float64)
    seeds = pregrasp_seeds(
        home_arm=np.zeros(7, dtype=np.float64),
        lower=lower,
        upper=upper,
        settings=settings,
    )
    scene = ArmSelfCollisionScene(inputs)
    branches: list[dict[str, object]] = []
    unique_solutions: list[np.ndarray] = []
    best_key = None
    best_motion = None
    best_record = None
    for seed_index, seed in enumerate(seeds):
        seeded_plan = copy.deepcopy(control_plan)
        seeded_plan["approach_high_seed_arm_positions_rad"] = seed.tolist()
        try:
            motion = control.build_joint_motion_plan(
                repository,
                inputs,
                seeded_plan,
                inputs.frozen_world_from_object,
                include_lift=True,
            )
        except (RuntimeError, ValueError) as error:
            branches.append(
                {
                    "seed_index": seed_index,
                    "status": "IK_INFEASIBLE",
                    "reason": f"{type(error).__name__}:{error}",
                }
            )
            continue
        approach_seed = np.asarray(
            motion["approach_arm_waypoints_rad"][0], dtype=np.float64
        )
        duplicate = next(
            (
                index
                for index, previous in enumerate(unique_solutions)
                if np.linalg.norm(approach_seed - previous) < 1.0e-5
            ),
            None,
        )
        if duplicate is not None:
            branches.append(
                {
                    "seed_index": seed_index,
                    "status": "DUPLICATE_IK_BRANCH",
                    "duplicate_unique_solution_index": duplicate,
                }
            )
            continue
        unique_solutions.append(approach_seed)
        route = np.asarray(_arm_route_states(inputs, motion), dtype=np.float64)
        hand = np.asarray(motion["pregrasp_hand_positions_rad"], dtype=np.float64)
        minimum_clearance = math.inf
        first_collision = None
        for route_index, arm in enumerate(route):
            clearance, pair = scene.state_clearance(arm, hand)
            minimum_clearance = min(minimum_clearance, clearance)
            if pair is not None:
                first_collision = {
                    "route_sample_index": route_index,
                    "link_pair": list(pair),
                }
                break
        absolute_margins = np.minimum(route - lower, upper - route)
        minimum_absolute_margin = float(np.min(absolute_margins))
        minimum_normalized_margin = float(
            np.min(absolute_margins / (upper - lower))
        )
        home_l2 = float(np.linalg.norm(approach_seed))
        eligible = (
            first_collision is None
            and minimum_clearance >= ARM_SELF_CONTACT_ENVELOPE_M
            and minimum_normalized_margin > 0.0
        )
        record = {
            "seed_index": seed_index,
            "status": "ELIGIBLE" if eligible else "ARM_ROUTE_INFEASIBLE",
            "approach_high_arm_positions_rad": approach_seed.tolist(),
            "maximum_ik_position_error_m": motion[
                "maximum_ik_position_error_m"
            ],
            "maximum_ik_orientation_error_rad": motion[
                "maximum_ik_orientation_error_rad"
            ],
            "minimum_self_collision_clearance_m": minimum_clearance,
            "minimum_absolute_joint_limit_margin_rad": minimum_absolute_margin,
            "minimum_normalized_joint_limit_margin": minimum_normalized_margin,
            "home_path_l2_rad": home_l2,
            "first_self_collision": first_collision,
        }
        branches.append(record)
        if eligible:
            key = (minimum_normalized_margin, -home_l2, -seed_index)
            if best_key is None or key > best_key:
                best_key = key
                best_motion = motion
                best_record = record
    if best_motion is None or best_record is None:
        raise RuntimeError("selected grasp has no collision-safe redundant arm route")
    return best_motion, {
        "method": "FINITE_SEED_COLLISION_SAFE_REDUNDANT_IK_V1",
        "selection_rule": (
            "IK_TOLERANCE_THEN_NO_NONADJACENT_IIWA_COLLISION_THEN_"
            "CONTACT_ENVELOPE_THEN_POSITIVE_JOINT_MARGIN_THEN_"
            "MAXIMUM_MINIMUM_NORMALIZED_JOINT_MARGIN_THEN_SHORTEST_HOME_L2"
        ),
        "self_contact_envelope_m": ARM_SELF_CONTACT_ENVELOPE_M,
        "self_contact_envelope_source": (
            "SUM_OF_TWO_FROZEN_50_MICROMETRE_ROBOT_CONTACT_OFFSETS"
        ),
        "declared_seed_count": len(seeds),
        "unique_feasible_ik_branch_count": len(unique_solutions),
        "selected": best_record,
        "branches": branches,
    }


class ExactCollisionScene:
    """FCL checks using the complete pad meshes and source CAD triangles."""

    def __init__(self, inputs: V2Inputs) -> None:
        self.inputs = inputs
        assert inputs.task_grip_surfaces is not None
        registered = inputs.hand_collision_triangles_by_link
        nonpad = task_noncontact_triangles(registered, inputs.task_grip_surfaces)
        self.full_object = {
            name: fcl.CollisionObject(_fcl_model(triangles))
            for name, triangles in registered.items()
        }
        self.full_world = {
            name: fcl.CollisionObject(_fcl_model(triangles))
            for name, triangles in registered.items()
        }
        self.nonpad = {
            name: fcl.CollisionObject(_fcl_model(triangles))
            for name, triangles in nonpad.items()
        }
        self.pad = {
            pad_name: fcl.CollisionObject(_fcl_model(surface.triangles_local_m))
            for pad_name, surface in inputs.task_grip_surfaces.items()
        }
        mesh = inputs.object_contract.model.mesh
        object_triangles = np.asarray(mesh.face_vertices_m, dtype=np.float64)
        self.object_face_is_contact_semantically_allowed = np.asarray(
            inputs.face_roles.face_is_allowed, dtype=np.bool_
        )
        if self.object_face_is_contact_semantically_allowed.shape != (
            len(mesh.faces),
        ):
            raise ValueError("functional object-face role coverage changed")
        winding_signs = np.asarray(
            inputs.object_contract.orientation_certificate
            .positive_volume_winding_sign_by_source_face,
            dtype=np.float64,
        )
        self.object_outward_face_normals = np.asarray(
            mesh.face_normals, dtype=np.float64
        ) * winding_signs[:, None]
        self.object_outward_face_normals.setflags(write=False)
        self.object = fcl.CollisionObject(_fcl_model(object_triangles))
        forbidden = object_triangles[
            ~self.object_face_is_contact_semantically_allowed
        ]
        self.forbidden_object = (
            None
            if not len(forbidden)
            else fcl.CollisionObject(_fcl_model(forbidden))
        )
        atlas_bindings = inputs.config.section("inputs").get(
            "functional_step_contact_atlas_by_object", {}
        )
        if not isinstance(atlas_bindings, Mapping):
            raise ValueError("functional STEP atlas bindings must be a mapping")
        atlas_path = atlas_bindings.get(inputs.object_contract.object_id)
        self.step_role_objects: dict[str, object] | None = None
        self.step_role_parent_faces: dict[str, np.ndarray] | None = None
        if atlas_path is not None:
            partition = load_step_triangle_role_partition(
                str(atlas_path), repository_root=inputs.repository_root
            )
            self.step_role_objects = {
                "proven_searchable": fcl.CollisionObject(
                    _fcl_model(partition.proven_searchable_triangles_m)
                ),
                "unresolved": fcl.CollisionObject(
                    _fcl_model(partition.unresolved_triangles_m)
                ),
                "hard_forbidden": fcl.CollisionObject(
                    _fcl_model(partition.hard_forbidden_triangles_m)
                ),
            }
            self.step_role_parent_faces = {
                "proven_searchable": (
                    partition.proven_searchable_parent_face_index
                ),
                "unresolved": partition.unresolved_parent_face_index,
                "hard_forbidden": (
                    partition.hard_forbidden_parent_face_index
                ),
            }
        links = tuple(sorted(self.full_object))
        adjacent = {
            tuple(sorted((joint.parent_link, joint.child_link)))
            for joint in inputs.hand_model.joints.values()
            if joint.parent_link in self.full_object
            and joint.child_link in self.full_object
        }
        self.self_pairs = tuple(
            (first, second)
            for first_index, first in enumerate(links)
            for second in links[first_index + 1 :]
            if tuple(sorted((first, second))) not in adjacent
        )
        bounds = inputs.table_xy_bounds_m
        size_x = float(bounds[0, 1] - bounds[0, 0])
        size_y = float(bounds[1, 1] - bounds[1, 0])
        thickness = 1.0
        center = np.asarray(
            (
                float(np.mean(bounds[0])),
                float(np.mean(bounds[1])),
                inputs.table_top_z_m - 0.5 * thickness,
            ),
            dtype=np.float64,
        )
        self.table = fcl.CollisionObject(
            fcl.Box(size_x, size_y, thickness), fcl.Transform(center)
        )
        self.collision_request = fcl.CollisionRequest(
            num_max_contacts=1, enable_contact=False
        )
        self.distance_request = fcl.DistanceRequest(enable_nearest_points=False)

    def set_state(self, object_from_hand: np.ndarray, joints: np.ndarray) -> None:
        transforms = self.inputs.hand_model.forward_kinematics(
            joints, base_transform=object_from_hand
        )
        world_from_object = self.inputs.frozen_world_from_object
        assert self.inputs.task_grip_surfaces is not None
        pad_by_link = {
            surface.link_name: pad_name
            for pad_name, surface in self.inputs.task_grip_surfaces.items()
        }
        for link_name in self.full_object:
            object_from_link = transforms[link_name]
            object_transform = fcl.Transform(
                object_from_link[:3, :3], object_from_link[:3, 3]
            )
            self.full_object[link_name].setTransform(object_transform)
            self.nonpad[link_name].setTransform(object_transform)
            if link_name in pad_by_link:
                self.pad[pad_by_link[link_name]].setTransform(object_transform)
            world_from_link = world_from_object @ object_from_link
            self.full_world[link_name].setTransform(
                fcl.Transform(world_from_link[:3, :3], world_from_link[:3, 3])
            )

    def _collides(self, first, second) -> bool:
        return bool(
            fcl.collide(
                first, second, self.collision_request, fcl.CollisionResult()
            )
        )

    def hard_collision_reason(self, stage: str) -> str:
        for first, second in self.self_pairs:
            if self._collides(self.full_object[first], self.full_object[second]):
                return f"SELF_COLLISION:{stage}:{first}:{second}"
        for link_name in sorted(self.nonpad):
            if self._collides(self.nonpad[link_name], self.object):
                return f"NONPAD_OBJECT_COLLISION:{stage}:{link_name}"
            if self._collides(self.full_world[link_name], self.table):
                return f"HAND_TABLE_COLLISION:{stage}:{link_name}"
        if self.step_role_objects is not None:
            hard = self.step_role_objects["hard_forbidden"]
            for pad_name in PAD_ORDER:
                if self._collides(self.pad[pad_name], hard):
                    return (
                        "PAD_STEP_HARD_OR_SHARED_BOUNDARY_CONTACT_"
                        f"REQUIRES_EXACT_REVIEW:{stage}:{pad_name}"
                    )
        return ""

    def pad_patch(self, pad_name: str) -> PadPatch | None:
        assert self.inputs.task_grip_surfaces is not None
        surface = self.inputs.task_grip_surfaces[pad_name]
        maximum = int(len(surface.triangles_local_m))
        request = fcl.CollisionRequest(
            num_max_contacts=maximum, enable_contact=True
        )
        result = fcl.CollisionResult()
        reported = int(fcl.collide(self.pad[pad_name], self.object, request, result))
        contacts = result.contacts
        if reported == 0:
            return None
        if reported != len(contacts) or len(contacts) >= maximum:
            raise RuntimeError(f"contact patch capacity exhausted for {pad_name}")
        faces = np.asarray([int(row.b2) for row in contacts], dtype=np.int64)
        face_count = len(self.inputs.object_contract.model.mesh.faces)
        if np.any(faces < 0) or np.any(faces >= face_count):
            raise RuntimeError(f"FCL object face identity unavailable for {pad_name}")
        points = np.asarray([row.pos for row in contacts], dtype=np.float64)
        depths = np.asarray(
            [float(row.penetration_depth) for row in contacts], dtype=np.float64
        )
        normals = self.object_outward_face_normals[faces]
        return PadPatch(
            pad_name=pad_name,
            link_name=surface.link_name,
            points_object_m=points,
            normals_object=normals,
            object_face_indices=faces,
            penetration_depths_m=depths,
        )

    def pad_has_object_contact(self, pad_name: str) -> bool:
        return self._collides(self.pad[pad_name], self.object)

    def pad_contact_reason(self, patch: PadPatch | None, stage: str) -> str:
        if patch is None:
            return ""
        if self.step_role_objects is not None:
            pad = self.pad[patch.pad_name]
            searchable = self._collides(
                pad, self.step_role_objects["proven_searchable"]
            )
            unresolved = self._collides(
                pad, self.step_role_objects["unresolved"]
            )
            hard = self._collides(
                pad, self.step_role_objects["hard_forbidden"]
            )
            if hard:
                return (
                    "PAD_STEP_HARD_OR_SHARED_BOUNDARY_CONTACT_"
                    f"REQUIRES_EXACT_REVIEW:{stage}:{patch.pad_name}"
                )
            if searchable:
                return ""
            if unresolved:
                return (
                    f"PAD_STEP_UNRESOLVED_CONTACT:{stage}:{patch.pad_name}"
                )
            return f"PAD_STEP_ROLE_MISMATCH:{stage}:{patch.pad_name}"
        allowed = self.object_face_is_contact_semantically_allowed[
            patch.object_face_indices
        ]
        if not np.all(allowed):
            face = int(patch.object_face_indices[np.flatnonzero(~allowed)[0]])
            return f"PAD_FORBIDDEN_OBJECT_CONTACT:{stage}:{patch.pad_name}:{face}"
        return ""

    def _distance(self, first, second) -> float:
        result = fcl.DistanceResult()
        value = float(fcl.distance(first, second, self.distance_request, result))
        return max(0.0, value)

    def minimum_forbidden_clearance_m(self) -> float:
        distances: list[float] = []
        for first, second in self.self_pairs:
            distances.append(
                self._distance(self.full_object[first], self.full_object[second])
            )
        for link_name in sorted(self.nonpad):
            distances.append(self._distance(self.nonpad[link_name], self.object))
            distances.append(self._distance(self.full_world[link_name], self.table))
        if self.forbidden_object is not None:
            if self.step_role_objects is None:
                for pad_name in PAD_ORDER:
                    distances.append(
                        self._distance(self.pad[pad_name], self.forbidden_object)
                    )
        if self.step_role_objects is not None:
            hard = self.step_role_objects["hard_forbidden"]
            for pad_name in PAD_ORDER:
                distances.append(self._distance(self.pad[pad_name], hard))
        return min(distances) if distances else math.inf


def _evaluate_control_path(
    inputs: V2Inputs,
    scene: ExactCollisionScene,
    control_plan: Mapping[str, object],
    settings: Mapping[str, object],
) -> tuple[PathEvaluation | None, str]:
    pose = np.asarray(
        control_plan["object_from_hand_row_major"], dtype=np.float64
    ).reshape(4, 4)
    pregrasp = np.asarray(
        control_plan["pregrasp_joint_positions_rad"], dtype=np.float64
    )
    endpoint = np.asarray(
        control_plan["final_joint_positions_rad"], dtype=np.float64
    )
    direction = np.asarray(
        control_plan["approach_direction_object"], dtype=np.float64
    )
    direction /= np.linalg.norm(direction)
    dynamic = inputs.config.section("dynamic")
    clearance = float(dynamic["approach_clearance_height_m"])
    approach_count = int(settings["approach_path_sample_count"])
    for index, fraction in enumerate(np.linspace(1.0, 0.0, approach_count)):
        transform = np.array(pose, copy=True)
        transform[:3, 3] -= direction * clearance * float(fraction)
        scene.set_state(transform, pregrasp)
        reason = scene.hard_collision_reason(f"APPROACH_{index:03d}")
        if reason:
            return None, reason
        for pad_name in PAD_ORDER:
            if scene.pad_has_object_contact(pad_name):
                patch = scene.pad_patch(pad_name)
                reason = scene.pad_contact_reason(
                    patch, f"APPROACH_{index:03d}"
                )
                if reason:
                    return None, reason
                return None, f"PAD_OBJECT_CONTACT_DURING_APPROACH:{pad_name}"

    current = np.array(pregrasp, copy=True)
    maximum_increment = float(dynamic["finger_maximum_speed_rad_s"]) * float(
        dynamic["physics_dt_s"]
    )
    slots = [
        inputs.hand_model.independent_joint_names.index(name)
        for name in CLOSING_JOINTS
    ]
    for finger_index, (pad_name, slot) in enumerate(zip(PAD_ORDER, slots)):
        start = float(current[slot])
        stop = float(endpoint[slot])
        count = max(1, int(math.ceil(abs(stop - start) / maximum_increment)))
        contacted = False
        for step_index, value in enumerate(
            np.linspace(start, stop, count + 1)[1:], start=1
        ):
            current[slot] = float(value)
            stage = f"FINGER_{finger_index + 1}_{step_index:04d}"
            scene.set_state(pose, current)
            reason = scene.hard_collision_reason(stage)
            if reason:
                return None, reason
            if scene.pad_has_object_contact(pad_name):
                active_patch = scene.pad_patch(pad_name)
                reason = scene.pad_contact_reason(active_patch, stage)
                if reason:
                    return None, reason
                contacted = True
                break
        if not contacted:
            return None, f"NO_ALLOWED_PAD_CONTACT:{pad_name}"

    scene.set_state(pose, current)
    reason = scene.hard_collision_reason("THREE_PAD_CONTACT")
    if reason:
        return None, reason
    patches: list[PadPatch] = []
    for pad_name in PAD_ORDER:
        patch = scene.pad_patch(pad_name)
        reason = scene.pad_contact_reason(patch, "THREE_PAD_CONTACT")
        if reason:
            return None, reason
        if patch is None:
            return None, f"LOST_ALLOWED_PAD_CONTACT:{pad_name}"
        patches.append(patch)
    closing_direction = np.sign(endpoint - pregrasp)
    if any(float(closing_direction[slot]) == 0.0 for slot in slots):
        return None, "ZERO_CLOSING_DIRECTION"
    closure_reserve = min(
        float(
            closing_direction[slot] * (endpoint[slot] - current[slot])
        )
        for slot in slots
    )
    if closure_reserve < 0.0:
        return None, "NEGATIVE_CLOSURE_RESERVE"
    maximum_preload_increment = float(dynamic["preload_increment_rad"])
    if closure_reserve + 1.0e-12 < maximum_preload_increment:
        return None, "INSUFFICIENT_MAXIMUM_PRELOAD_CLOSURE_RESERVE"
    maximum_preload_state = np.array(current, copy=True)
    for slot in slots:
        maximum_preload_state[slot] += (
            closing_direction[slot] * maximum_preload_increment
        )
    lower, upper = inputs.hand_model.joint_limit_vectors()
    commanded_states = np.asarray(
        (pregrasp, current, maximum_preload_state), dtype=np.float64
    )
    joint_limit_margin = float(
        np.min(
            np.minimum(
                commanded_states - lower[None, :],
                upper[None, :] - commanded_states,
            )
        )
    )
    if joint_limit_margin <= 0.0:
        return None, "NONPOSITIVE_COMMANDED_HAND_JOINT_LIMIT_MARGIN"
    return (
        PathEvaluation(
            contact_joint_positions_rad=np.array(current, copy=True),
            patches=(patches[0], patches[1], patches[2]),
            minimum_closure_reserve_rad=closure_reserve,
            minimum_commanded_hand_joint_limit_margin_rad=(
                joint_limit_margin
            ),
            minimum_forbidden_clearance_m=(
                scene.minimum_forbidden_clearance_m()
            ),
        ),
        "",
    )


def _friction_basis(
    normal_outward: np.ndarray, friction: float, count: int
) -> np.ndarray:
    normal = np.asarray(normal_outward, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(normal @ reference)) > 0.8:
        reference = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    tangent_1 = np.cross(normal, reference)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    angles = 2.0 * math.pi * np.arange(count, dtype=np.float64) / count
    return np.asarray(
        [
            -normal
            + friction
            * (math.cos(angle) * tangent_1 + math.sin(angle) * tangent_2)
            for angle in angles
        ],
        dtype=np.float64,
    )


def _single_pad_resultant(patch: PadPatch) -> PadResultant | None:
    """Collapse one full-pad collision patch to one resultant point contact."""

    point = np.mean(patch.points_object_m, axis=0)
    mean_normal = np.mean(patch.normals_object, axis=0)
    magnitude = float(np.linalg.norm(mean_normal))
    if (
        not np.all(np.isfinite(point))
        or not np.all(np.isfinite(mean_normal))
        or not math.isfinite(magnitude)
        or magnitude == 0.0
    ):
        return None
    return PadResultant(
        pad_name=patch.pad_name,
        point_object_m=point,
        normal_object=mean_normal / magnitude,
        collision_witness_count=len(patch.points_object_m),
    )


def _scene_gravity_world_m_s2(inputs: V2Inputs) -> np.ndarray:
    scene = inputs.config.section("dynamic")["object_scenes"][
        inputs.object_contract.object_id
    ]
    path_keys = [
        key for key in ("scene_config", "environment_scene_config") if key in scene
    ]
    if len(path_keys) != 1:
        raise ValueError("object scene must bind exactly one tabletop contract")
    document = yaml.safe_load(
        (inputs.repository_root / str(scene[path_keys[0]])).read_text(
            encoding="utf-8"
        )
    )
    gravity_z = float(document["physics"]["gravity_m_s2"])
    if not math.isfinite(gravity_z):
        raise ValueError("Isaac scene gravity is not finite")
    return np.asarray((0.0, 0.0, gravity_z), dtype=np.float64)


def _force_closure_diagnostic(wrench: np.ndarray) -> bool:
    if np.linalg.matrix_rank(wrench, tol=1.0e-10) < 6:
        return False
    count = wrench.shape[1]
    objective = np.zeros(count + 1, dtype=np.float64)
    objective[-1] = -1.0
    equality = np.zeros((7, count + 1), dtype=np.float64)
    equality[:6, :count] = wrench
    equality[6, :count] = 1.0
    target = np.zeros(7, dtype=np.float64)
    target[6] = 1.0
    upper = np.zeros((count, count + 1), dtype=np.float64)
    upper[np.arange(count), np.arange(count)] = -1.0
    upper[:, -1] = 1.0
    solution = linprog(
        objective,
        A_ub=upper,
        b_ub=np.zeros(count),
        A_eq=equality,
        b_eq=target,
        bounds=[(0.0, None)] * (count + 1),
        method="highs",
    )
    return bool(solution.success and solution.x[-1] > 1.0e-10)


def _task_quality(
    inputs: V2Inputs,
    settings: Mapping[str, object],
    object_from_hand: np.ndarray,
    contact_joints: np.ndarray,
    patches: Sequence[PadPatch],
    *,
    friction_coefficient: float | None = None,
    mass_kg: float | None = None,
    center_of_mass_object_m: np.ndarray | None = None,
) -> TaskQuality | None:
    edge_count = int(settings["friction_cone_edge_count"])
    friction = float(
        inputs.object_contract.contact_material_uncertainty
        .friction_coefficient_interval[0]
        if friction_coefficient is None
        else friction_coefficient
    )
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("task-quality friction must be finite and nonnegative")
    force_cap = float(settings["nominal_normal_force_cap_n"])
    resultants = [_single_pad_resultant(patch) for patch in patches]
    if any(resultant is None for resultant in resultants):
        return None
    points = np.asarray(
        [resultant.point_object_m for resultant in resultants], dtype=np.float64
    )
    normals = np.asarray(
        [resultant.normal_object for resultant in resultants], dtype=np.float64
    )
    pad_indices = np.arange(len(PAD_ORDER), dtype=np.int64)
    bases = [
        _friction_basis(normal, friction, edge_count) for normal in normals
    ]
    coefficient_count = len(points) * edge_count
    joint_count = len(inputs.hand_model.independent_joint_names)
    closing_slots = np.asarray(
        [
            inputs.hand_model.independent_joint_names.index(name)
            for name in CLOSING_JOINTS
        ],
        dtype=np.int64,
    )
    wrench = np.zeros((6, coefficient_count), dtype=np.float64)
    torque = np.zeros((joint_count, coefficient_count), dtype=np.float64)
    assert inputs.task_grip_surfaces is not None
    hand_from_links = inputs.hand_model.forward_kinematics(contact_joints)
    com = np.asarray(
        inputs.object_contract.model.com_m
        if center_of_mass_object_m is None
        else center_of_mass_object_m,
        dtype=np.float64,
    )
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        raise ValueError("task-quality center of mass must be one finite 3-vector")
    for contact_index, (point, basis_vectors, pad_index) in enumerate(
        zip(points, bases, pad_indices)
    ):
        surface = inputs.task_grip_surfaces[PAD_ORDER[int(pad_index)]]
        contact_hand = object_from_hand[:3, :3].T @ (
            point - object_from_hand[:3, 3]
        )
        hand_from_link = hand_from_links[surface.link_name]
        contact_link = hand_from_link[:3, :3].T @ (
            contact_hand - hand_from_link[:3, 3]
        )
        jacobian_hand = inputs.hand_model.geometric_jacobian(
            surface.link_name,
            contact_joints,
            point_local_m=contact_link,
        )[:3]
        jacobian_object = object_from_hand[:3, :3] @ jacobian_hand
        for edge_index, force in enumerate(basis_vectors):
            column = contact_index * edge_count + edge_index
            wrench[:3, column] = force
            wrench[3:, column] = np.cross(point - com, force)
            torque[:, column] = jacobian_object.T @ (-force)

    dynamic = inputs.config.section("dynamic")
    drive_limit = float(dynamic["hand_drive_maximum_effort_nm"])
    closing_limit = min(
        drive_limit, float(dynamic["measured_effort_abort_nm"])
    )
    torque_limits = np.asarray(
        [
            closing_limit if name in CLOSING_JOINTS else drive_limit
            for name in inputs.hand_model.independent_joint_names
        ],
        dtype=np.float64,
    )
    lift_peak = (
        10.0
        / math.sqrt(3.0)
        * float(dynamic["lift_command_distance_m"])
        / float(dynamic["lift_duration_s"]) ** 2
    )
    object_from_world_rotation = inputs.frozen_world_from_object[:3, :3].T
    gravity_world = _scene_gravity_world_m_s2(inputs)
    mass = float(
        inputs.object_contract.model.mass_kg if mass_kg is None else mass_kg
    )
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("task-quality mass must be finite and positive")
    required_forces = (
        mass * (object_from_world_rotation @ (-gravity_world)),
        mass
        * (
            object_from_world_rotation
            @ (np.asarray((0.0, 0.0, lift_peak)) - gravity_world)
        ),
    )

    stage_rows: list[dict[str, object]] = []
    for stage, required_force in zip(("hold", "lift"), required_forces):
        objective = np.zeros(coefficient_count + 1, dtype=np.float64)
        objective[-1] = -1.0
        equality = np.zeros((6, coefficient_count + 1), dtype=np.float64)
        equality[:, :coefficient_count] = wrench
        equality[:, -1] = -np.concatenate((required_force, np.zeros(3)))
        upper_rows: list[np.ndarray] = []
        upper_values: list[float] = []
        for pad_index in range(len(PAD_ORDER)):
            row = np.zeros(coefficient_count + 1, dtype=np.float64)
            for contact_index in np.flatnonzero(pad_indices == pad_index):
                start = int(contact_index) * edge_count
                row[start : start + edge_count] = 1.0
            upper_rows.append(row)
            upper_values.append(force_cap)
        for joint_index, limit in enumerate(torque_limits):
            for sign in (-1.0, 1.0):
                row = np.zeros(coefficient_count + 1, dtype=np.float64)
                row[:coefficient_count] = sign * torque[joint_index]
                upper_rows.append(row)
                upper_values.append(float(limit))
        solution = linprog(
            objective,
            A_ub=np.asarray(upper_rows),
            b_ub=np.asarray(upper_values),
            A_eq=equality,
            b_eq=np.zeros(6, dtype=np.float64),
            bounds=[(0.0, None)] * coefficient_count + [(0.0, None)],
            method="highs",
        )
        if not solution.success or not np.isfinite(solution.x[-1]):
            return None
        factor = float(solution.x[-1])
        if factor <= 0.0:
            return None
        unit_coefficients = solution.x[:coefficient_count] / factor
        unit_torque = np.abs(torque @ unit_coefficients)
        unit_pad_force = np.asarray(
            [
                sum(
                    float(
                        np.sum(
                            unit_coefficients[
                                int(contact_index) * edge_count:
                                (int(contact_index) + 1) * edge_count
                            ]
                        )
                    )
                    for contact_index in np.flatnonzero(
                        pad_indices == pad_index
                    )
                )
                for pad_index in range(len(PAD_ORDER))
            ],
            dtype=np.float64,
        )
        stage_rows.append(
            {
                "stage": stage,
                "factor": factor,
                "maximum_joint_torque": float(np.max(unit_torque)),
                "minimum_joint_margin": float(
                    np.min(torque_limits - unit_torque)
                ),
                "maximum_pad_force": float(np.max(unit_pad_force)),
                "minimum_pad_margin": float(
                    np.min(force_cap - unit_pad_force)
                ),
                "closing_joint_effort": unit_torque[closing_slots],
            }
        )
    limiting = min(stage_rows, key=lambda row: float(row["factor"]))
    required_closing_effort = np.maximum.reduce(
        [np.asarray(row["closing_joint_effort"]) for row in stage_rows]
    )
    return TaskQuality(
        task_load_factor=float(limiting["factor"]),
        hold_load_factor=float(stage_rows[0]["factor"]),
        lift_load_factor=float(stage_rows[1]["factor"]),
        limiting_stage=str(limiting["stage"]),
        maximum_joint_torque_at_unit_task_nm=float(
            limiting["maximum_joint_torque"]
        ),
        minimum_joint_torque_margin_at_unit_task_nm=float(
            limiting["minimum_joint_margin"]
        ),
        maximum_pad_normal_force_at_unit_task_n=float(
            limiting["maximum_pad_force"]
        ),
        minimum_pad_normal_force_margin_at_unit_task_n=float(
            limiting["minimum_pad_margin"]
        ),
        required_closing_joint_effort_nm=tuple(
            float(value) for value in required_closing_effort
        ),
        wrench_span_rank=int(np.linalg.matrix_rank(wrench, tol=1.0e-10)),
        force_closure_friction_pyramid=_force_closure_diagnostic(wrench),
    )


def _quality_record(quality: TaskQuality) -> dict[str, object]:
    return {
        "task_load_factor": quality.task_load_factor,
        "hold_load_factor": quality.hold_load_factor,
        "lift_load_factor": quality.lift_load_factor,
        "limiting_stage": quality.limiting_stage,
        "maximum_joint_torque_at_unit_task_nm": (
            quality.maximum_joint_torque_at_unit_task_nm
        ),
        "minimum_joint_torque_margin_at_unit_task_nm": (
            quality.minimum_joint_torque_margin_at_unit_task_nm
        ),
        "maximum_pad_normal_force_at_unit_task_n": (
            quality.maximum_pad_normal_force_at_unit_task_n
        ),
        "minimum_pad_normal_force_margin_at_unit_task_n": (
            quality.minimum_pad_normal_force_margin_at_unit_task_n
        ),
        "required_closing_joint_effort_nm": dict(
            zip(CLOSING_JOINTS, quality.required_closing_joint_effort_nm)
        ),
        "wrench_span_rank": quality.wrench_span_rank,
        "force_closure_friction_pyramid": (
            quality.force_closure_friction_pyramid
        ),
    }


def _minimum_task_friction_record(
    inputs: V2Inputs,
    settings: Mapping[str, object],
    object_from_hand: np.ndarray,
    contact_joints: np.ndarray,
    patches: Sequence[PadPatch],
    upper_quality: TaskQuality,
) -> dict[str, object]:
    """Locate the first feasible coefficient on one frozen finite friction grid."""

    resolution = float(settings["minimum_task_friction_resolution"])
    upper = float(
        inputs.object_contract.contact_material_uncertainty
        .friction_coefficient_interval[0]
    )
    if (
        not math.isfinite(resolution)
        or resolution <= 0.0
        or not math.isfinite(upper)
        or upper <= 0.0
    ):
        raise ValueError("minimum task friction grid must be positive and finite")
    upper_index = int(round(upper / resolution))
    if upper_index < 1 or not math.isclose(
        upper_index * resolution, upper, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "contact-contract lower friction must lie exactly on the frozen grid"
        )
    if upper_quality.task_load_factor < 1.0:
        raise ValueError("friction-margin input is not feasible at its upper bound")

    zero_quality = _task_quality(
        inputs,
        settings,
        object_from_hand,
        contact_joints,
        patches,
        friction_coefficient=0.0,
    )
    if zero_quality is not None and zero_quality.task_load_factor >= 1.0:
        first_feasible_index = 0
    else:
        lower_index = 0
        first_feasible_index = upper_index
        while first_feasible_index - lower_index > 1:
            middle_index = (lower_index + first_feasible_index) // 2
            middle_quality = _task_quality(
                inputs,
                settings,
                object_from_hand,
                contact_joints,
                patches,
                friction_coefficient=middle_index * resolution,
            )
            if (
                middle_quality is not None
                and middle_quality.task_load_factor >= 1.0
            ):
                first_feasible_index = middle_index
            else:
                lower_index = middle_index

    minimum_required = first_feasible_index * resolution
    return {
        "definition": (
            "FIRST_FROZEN_FRICTION_GRID_COEFFICIENT_WITH_"
            "MINIMUM_HOLD_AND_UPWARD_LIFT_LOAD_FACTOR_AT_LEAST_ONE"
        ),
        "minimum_required_friction_coefficient": minimum_required,
        "analytic_margin_to_contact_contract_lower_bound": upper - minimum_required,
        "friction_grid_resolution": resolution,
        "friction_grid_lower": 0.0,
        "friction_grid_upper": upper,
        "friction_grid_count": upper_index + 1,
    }


def _grid_record(point: GridPoint) -> dict[str, object]:
    return {
        "candidate_id": point.candidate_id,
        "grid_indices": {
            "palm": point.palm_index,
            "yaw": point.yaw_index,
            "axial": point.axial_index,
        },
        "palm_rad": point.palm_rad,
        "yaw_rad": point.yaw_rad,
        "cad_reference_section_z_m": point.axial_m,
        "is_nominal_baseline_grid_point": point.is_nominal_baseline_point,
    }


def _robust_row_selection_key(row: Mapping[str, object]) -> tuple[float, ...]:
    robustness = row["robustness"]
    indices = row["grid_indices"]
    if not isinstance(robustness, Mapping) or not isinstance(indices, Mapping):
        raise ValueError("robust candidate row has invalid metric or grid data")
    return tuple(float(robustness[name]) for name in ROBUST_SELECTION_METRICS) + (
        -float(indices["palm"]),
        -float(indices["yaw"]),
        -float(indices["axial"]),
    )


def _dynamic_palm_component_panel(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Expose one frozen-static representative per separated palm branch."""

    eligible = [
        row
        for row in rows
        if isinstance(row.get("robustness"), Mapping)
        and row["robustness"].get("status") == "ROBUST_EXECUTABLE_OFFLINE"
    ]
    palm_indices = sorted(
        {int(row["grid_indices"]["palm"]) for row in eligible}
    )
    components: list[list[int]] = []
    for palm_index in palm_indices:
        if not components or palm_index != components[-1][-1] + 1:
            components.append([palm_index])
        else:
            components[-1].append(palm_index)

    records: list[dict[str, object]] = []
    for component_index, component in enumerate(components):
        members = [
            row
            for row in eligible
            if int(row["grid_indices"]["palm"]) in component
        ]
        representative = max(members, key=_robust_row_selection_key)
        robustness = representative["robustness"]
        path = representative["path"]
        records.append(
            {
                "component_index": component_index,
                "palm_grid_indices": component,
                "palm_rad": sorted({float(row["palm_rad"]) for row in members}),
                "candidate_count": len(members),
                "representative": {
                    "candidate_id": representative["candidate_id"],
                    "grid_indices": representative["grid_indices"],
                    "palm_rad": representative["palm_rad"],
                    "yaw_rad": representative["yaw_rad"],
                    "cad_reference_section_z_m": representative[
                        "cad_reference_section_z_m"
                    ],
                    "contact_patch_centroids_object_m": path[
                        "contact_patch_centroids_object_m"
                    ],
                    "robustness_metrics": {
                        name: robustness[name]
                        for name in ROBUST_SELECTION_METRICS
                    },
                },
            }
        )
    return {
        "method": "CONNECTED_COMPONENTS_OF_FEASIBLE_PALM_GRID_PROJECTION_V1",
        "adjacency_rule": "ABSOLUTE_PALM_GRID_INDEX_DIFFERENCE_EQUALS_ONE",
        "representative_selection_rule": (
            "ORIGINAL_FROZEN_V5_ROBUST_LEXICOGRAPHIC_KEY_WITHIN_COMPONENT"
        ),
        "dynamic_result_used_in_panel_construction": False,
        "physical_scope": (
            "DIVERSIFIES_SEPARATED_PALM_SPREAD_BRANCHES_ONLY; DOES_NOT_CLAIM_"
            "COMPLETE_CONTACT_MODE_CLUSTERING_OR_DYNAMIC_OPTIMALITY"
        ),
        "component_count": len(records),
        "components": records,
    }


def _path_summary(path: PathEvaluation) -> dict[str, object]:
    return {
        "contact_joint_positions_rad": (
            path.contact_joint_positions_rad.tolist()
        ),
        "collision_witness_counts": [
            len(patch.points_object_m) for patch in path.patches
        ],
        "contact_patch_centroids_object_m": [
            np.mean(patch.points_object_m, axis=0).tolist()
            for patch in path.patches
        ],
        "minimum_closure_reserve_rad": path.minimum_closure_reserve_rad,
        "minimum_commanded_hand_joint_limit_margin_rad": (
            path.minimum_commanded_hand_joint_limit_margin_rad
        ),
        "minimum_forbidden_clearance_m": path.minimum_forbidden_clearance_m,
    }


def _selected_detail(
    point: GridPoint,
    generation: Mapping[str, object],
    path: PathEvaluation,
    quality: TaskQuality,
) -> dict[str, object]:
    result = {
        **_grid_record(point),
        "control_plan": generation["control_plan"],
        "generation_evidence": generation["evidence"],
        "path": _path_summary(path),
        "quality": _quality_record(quality),
        "full_pad_contact_patches": [],
        "single_resultant_contacts": [],
    }
    for patch in path.patches:
        resultant = _single_pad_resultant(patch)
        if resultant is None:
            raise RuntimeError("selected full-pad contact has no resultant normal")
        result["full_pad_contact_patches"].append(
            {
                "pad_name": patch.pad_name,
                "link_name": patch.link_name,
                "points_object_m": patch.points_object_m.tolist(),
                "normals_object": patch.normals_object.tolist(),
                "object_source_face_indices": (
                    patch.object_face_indices.tolist()
                ),
                "penetration_depths_m": (
                    patch.penetration_depths_m.tolist()
                ),
            }
        )
        result["single_resultant_contacts"].append(
            {
                "pad_name": resultant.pad_name,
                "point_object_m": resultant.point_object_m.tolist(),
                "normal_object": resultant.normal_object.tolist(),
                "source_collision_witness_count": (
                    resultant.collision_witness_count
                ),
                "independent_torsional_moment_allowed": False,
            }
        )
    return result


def _motion_target_key(
    inputs: V2Inputs, control_plan: Mapping[str, object]
) -> tuple[float, ...]:
    pose = np.asarray(
        control_plan["object_from_hand_row_major"], dtype=np.float64
    ).reshape(4, 4)
    target = inputs.frozen_world_from_object @ pose
    direction = inputs.frozen_world_from_object[:3, :3] @ np.asarray(
        control_plan["approach_direction_object"], dtype=np.float64
    )
    return tuple(float(value) for value in np.round(
        np.concatenate((target.ravel(), direction)), decimals=12
    ))


def _retarget_cached_motion_plan(
    template: Mapping[str, object], control_plan: Mapping[str, object]
) -> dict[str, object]:
    result = copy.deepcopy(template)
    result["pregrasp_hand_positions_rad"] = tuple(
        float(value)
        for value in control_plan["pregrasp_joint_positions_rad"]
    )
    result["final_hand_positions_rad"] = tuple(
        float(value) for value in control_plan["final_joint_positions_rad"]
    )
    return result


def _solve_or_reuse_motion_plan(
    repository: Path,
    inputs: V2Inputs,
    control_plan: Mapping[str, object],
    yaw_index: int,
    cache: dict[tuple[float, ...], Mapping[str, object] | Exception],
    seed_by_yaw: dict[int, Sequence[float]],
    fallback_seed: Sequence[float],
) -> dict[str, object]:
    key = _motion_target_key(inputs, control_plan)
    cached = cache.get(key)
    if isinstance(cached, Exception):
        raise cached
    if cached is not None:
        return _retarget_cached_motion_plan(cached, control_plan)

    seed = seed_by_yaw.get(yaw_index, fallback_seed)
    seeded_plan = copy.deepcopy(control_plan)
    seeded_plan["approach_high_seed_arm_positions_rad"] = [
        float(value) for value in seed
    ]
    try:
        motion = control.build_joint_motion_plan(
            repository,
            inputs,
            seeded_plan,
            inputs.frozen_world_from_object,
            include_lift=True,
        )
    except (RuntimeError, ValueError):
        try:
            motion = control.build_joint_motion_plan(
                repository,
                inputs,
                control_plan,
                inputs.frozen_world_from_object,
                include_lift=True,
            )
        except (RuntimeError, ValueError) as error:
            cache[key] = error
            raise
    cache[key] = motion
    seed_by_yaw[yaw_index] = tuple(motion["approach_arm_waypoints_rad"][0])
    return motion


def _robust_scenario_plan(
    inputs: V2Inputs,
    control_plan: Mapping[str, object],
    scenario: OfflineRobustScenario,
) -> tuple[V2Inputs, dict[str, object]]:
    """Keep the nominal world hand target while perturbing the object pose."""

    nominal_world_object = np.asarray(inputs.frozen_world_from_object)
    actual_world_object = np.array(nominal_world_object, copy=True)
    actual_world_object[:3, :3] = (
        _rotation_about_z(scenario.yaw_world_rad)
        @ nominal_world_object[:3, :3]
    )
    actual_world_object[:3, 3] += scenario.translation_world_m
    scenario_inputs = replace(
        inputs, frozen_world_from_object=actual_world_object
    )
    plan = copy.deepcopy(control_plan)
    nominal_object_hand = np.asarray(
        plan["object_from_hand_row_major"], dtype=np.float64
    ).reshape(4, 4)
    plan["object_from_hand_row_major"] = (
        np.linalg.inv(actual_world_object)
        @ nominal_world_object
        @ nominal_object_hand
    ).ravel().tolist()
    direction_world = (
        nominal_world_object[:3, :3]
        @ np.asarray(plan["approach_direction_object"], dtype=np.float64)
    )
    plan["approach_direction_object"] = (
        actual_world_object[:3, :3].T @ direction_world
    ).tolist()
    if scenario.finger_3_joint_target_offset_rad:
        slot = inputs.hand_model.independent_joint_names.index("f3j2")
        for key in ("pregrasp_joint_positions_rad", "final_joint_positions_rad"):
            joints = np.asarray(plan[key], dtype=np.float64).copy()
            joints[slot] += scenario.finger_3_joint_target_offset_rad
            plan[key] = joints.tolist()
    return scenario_inputs, plan


def _robust_candidate_summary(
    inputs: V2Inputs,
    scene: ExactCollisionScene,
    settings: Mapping[str, object],
    control_plan: Mapping[str, object],
    nominal_path: PathEvaluation,
    nominal_quality: TaskQuality,
    scenarios: Sequence[OfflineRobustScenario],
) -> dict[str, object]:
    expected = ["nominal"] + [
        row.name for row in scenarios[1:] if row.offline_components
    ]
    evaluated = ["nominal"]
    metrics = {
        "task": [nominal_quality.task_load_factor, "nominal"],
        "clearance": [nominal_path.minimum_forbidden_clearance_m, "nominal"],
        "joint": [
            nominal_path.minimum_commanded_hand_joint_limit_margin_rad,
            "nominal",
        ],
        "closure": [nominal_path.minimum_closure_reserve_rad, "nominal"],
    }
    required_effort = np.asarray(
        nominal_quality.required_closing_joint_effort_nm
    )
    nominal_pose = np.asarray(
        control_plan["object_from_hand_row_major"], dtype=np.float64
    ).reshape(4, 4)
    geometry = {
        "object_translation_delta_m",
        "object_yaw_delta_rad",
        "finger_joint_target_offset_rad",
    }

    def failure(
        status: str, reason: str, scenario: str, factor: float | None = None
    ) -> dict[str, object]:
        result = {
            "status": status,
            "reason": reason,
            "failure_scenario": scenario,
            "offline_evaluated_scenarios": list(evaluated),
        }
        if factor is not None:
            result["failure_task_load_factor"] = factor
        return result

    for scenario in scenarios[1:]:
        if not scenario.offline_components:
            continue
        if geometry.intersection(scenario.offline_components):
            scenario_inputs, plan = _robust_scenario_plan(
                inputs, control_plan, scenario
            )
            scene.inputs = scenario_inputs
            try:
                path, reason = _evaluate_control_path(
                    scenario_inputs, scene, plan, settings
                )
            finally:
                scene.inputs = inputs
            evaluated.append(scenario.name)
            if path is None:
                return failure(
                    "ROBUST_PATH_INFEASIBLE", reason, scenario.name
                )
            pose = np.asarray(
                plan["object_from_hand_row_major"], dtype=np.float64
            ).reshape(4, 4)
        else:
            scenario_inputs, path, pose = inputs, nominal_path, nominal_pose
            evaluated.append(scenario.name)
        quality = _task_quality(
            scenario_inputs,
            settings,
            pose,
            path.contact_joint_positions_rad,
            path.patches,
            friction_coefficient=scenario.friction_coefficient,
            mass_kg=inputs.object_contract.model.mass_kg * scenario.mass_scale,
            center_of_mass_object_m=(
                inputs.object_contract.model.com_m
                + scenario.center_of_mass_delta_object_m
            ),
        )
        if quality is None:
            return failure(
                "ROBUST_TASK_INFEASIBLE",
                "TASK_LOAD_LINEAR_PROGRAM_INFEASIBLE",
                scenario.name,
            )
        if quality.task_load_factor < 1.0:
            return failure(
                "ROBUST_TASK_INFEASIBLE",
                "TASK_LOAD_FACTOR_BELOW_ONE",
                scenario.name,
                quality.task_load_factor,
            )
        values = {
            "task": quality.task_load_factor,
            "clearance": path.minimum_forbidden_clearance_m,
            "joint": path.minimum_commanded_hand_joint_limit_margin_rad,
            "closure": path.minimum_closure_reserve_rad,
        }
        for key, value in values.items():
            if value < metrics[key][0]:
                metrics[key] = [value, scenario.name]
        required_effort = np.maximum(
            required_effort, quality.required_closing_joint_effort_nm
        )

    if evaluated != expected:
        raise RuntimeError("robust candidate evaluation is incomplete")
    return {
        "status": "ROBUST_EXECUTABLE_OFFLINE",
        "offline_evaluated_scenarios": evaluated,
        "worst_task_load_factor": metrics["task"][0],
        "worst_task_load_factor_scenario": metrics["task"][1],
        "worst_minimum_forbidden_clearance_m": metrics["clearance"][0],
        "worst_minimum_forbidden_clearance_scenario": metrics["clearance"][1],
        "worst_minimum_commanded_hand_joint_limit_margin_rad": metrics["joint"][0],
        "worst_minimum_commanded_hand_joint_limit_margin_scenario": metrics["joint"][1],
        "worst_minimum_closure_reserve_rad": metrics["closure"][0],
        "worst_minimum_closure_reserve_scenario": metrics["closure"][1],
        "required_closing_joint_effort_nm": dict(
            zip(CLOSING_JOINTS, (float(value) for value in required_effort))
        ),
    }
def main() -> int:
    started = time.monotonic()
    arguments = _arguments()
    repository = Path(__file__).resolve().parents[4]
    config_path = Path(arguments.config).expanduser().resolve()
    output = Path(arguments.output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    inputs = load_v2_inputs(
        repository, config_path=config_path, object_id=arguments.object_id
    )
    if inputs.hand_variant != EXPECTED_VARIANT or inputs.task_grip_surfaces is None:
        raise ValueError("finite search requires the direct nail-free hand binding")
    settings = inputs.config.section("finite_cad_search")
    robust_selection_enabled = arguments.selection_mode == "robust"
    friction_margin_selection_enabled = (
        arguments.selection_mode == "friction_margin"
    )
    robust_scenarios = (
        _load_offline_robust_scenarios(inputs)
        if robust_selection_enabled
        else ()
    )
    if settings.get("method") != EXPECTED_METHOD:
        raise ValueError("finite search method identity changed")
    if (
        settings.get("search_space_frozen_before_evaluation") is not True
        or settings.get("require_complete_candidate_evaluation") is not True
        or settings.get("heuristic_candidate_deletion_allowed") is not False
        or settings.get("task_quality")
        != "MINIMUM_HOLD_AND_UPWARD_LIFT_LOAD_FACTOR"
        or settings.get("closure_path_step_rule")
        != "ISAAC_FINGER_SPEED_TIMES_PHYSICS_DT"
        or settings.get("contact_patch_witness_limit_rule")
        != "FULL_PAD_TRIANGLE_COUNT"
        or settings.get("full_pad_contact_force_model")
        != "ONE_EQUAL_WITNESS_RESULTANT_PER_FULL_PAD"
        or settings.get("contact_resultant_point_rule")
        != "ARITHMETIC_MEAN_OF_FCL_COLLISION_WITNESS_POSITIONS"
        or settings.get("contact_resultant_normal_rule")
        != "NORMALIZED_ARITHMETIC_MEAN_OF_OBJECT_OUTWARD_FACE_NORMALS"
        or settings.get("independent_contact_torsional_moment_allowed") is not False
        or settings.get("maximum_preload_increment_rule")
        != "MEASURED_EFFORT_ABORT_DIVIDED_BY_HAND_STIFFNESS"
        or settings.get("commanded_hand_joint_limit_margin_rule")
        != "STRICTLY_POSITIVE_AT_PREGRASP_CONTACT_AND_MAXIMUM_PRELOAD"
        or list(settings.get("hand_lateral_offset_task_m", ())) != [0.0, 0.0]
        or list(settings.get("hand_tilt_task_rad", ())) != [0.0, 0.0]
    ):
        raise ValueError("finite search definition is not frozen as declared")
    dynamic = inputs.config.section("dynamic")
    physical_preload_limit = float(dynamic["measured_effort_abort_nm"]) / float(
        dynamic["hand_stiffness"]
    )
    if not math.isclose(
        float(dynamic["preload_increment_rad"]),
        physical_preload_limit,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "maximum preload increment is not the declared bounded effort limit"
        )

    palms, yaws, axial, points = _declared_grid(inputs, settings)
    source_declared_count = len(points)
    replay_candidate_id = arguments.replay_candidate_id
    candidate_replay_only = replay_candidate_id is not None
    if candidate_replay_only:
        replay_points = [
            point for point in points if point.candidate_id == replay_candidate_id
        ]
        if len(replay_points) != 1:
            raise ValueError("replay candidate is not one declared G_delta member")
        points = replay_points
    declared_count = len(points)
    print(
        json.dumps(
            {
                "stage": "declared_grid",
                "object_id": arguments.object_id,
                "declared_candidate_count": declared_count,
            }
        ),
        flush=True,
    )
    scene = ExactCollisionScene(inputs)
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    selected = None
    selected_key = None
    selected_robustness = None
    selected_friction_margin = None
    selected_motion_plan = None
    selected_arm_route = None
    base_cache: dict[tuple[int, int], Mapping[str, object] | Exception] = {}
    motion_cache: dict[
        tuple[float, ...], Mapping[str, object] | Exception
    ] = {}
    baseline_generation = generate_axial_pad_intersection_grasp(inputs)
    baseline_motion = control.build_joint_motion_plan(
        repository,
        inputs,
        baseline_generation["control_plan"],
        inputs.frozen_world_from_object,
        include_lift=True,
    )
    baseline_seed = tuple(baseline_motion["approach_arm_waypoints_rad"][0])
    motion_cache[
        _motion_target_key(inputs, baseline_generation["control_plan"])
    ] = baseline_motion
    seed_by_yaw: dict[int, Sequence[float]] = {0: baseline_seed}
    palm_anchor = float(settings["palm_anchor_rad"])
    evaluation_points = (
        points
        if candidate_replay_only
        else sorted(
            points,
            key=lambda point: (
                abs(point.palm_rad - palm_anchor),
                point.palm_index,
                point.axial_index,
                point.yaw_index,
            ),
        )
    )

    for evaluated_index, point in enumerate(evaluation_points, start=1):
        base_key = (point.palm_index, point.axial_index)
        base = base_cache.get(base_key)
        if base is None:
            try:
                base = generate_axial_pad_intersection_grasp(
                    inputs,
                    palm_joint_position_rad=point.palm_rad,
                    grasp_axis_position_m=point.axial_m,
                    hand_yaw_rad=0.0,
                    apply_table_clearance=False,
                )
            except (RuntimeError, ValueError) as error:
                base = error
            base_cache[base_key] = base
        row = _grid_record(point)
        if isinstance(base, Exception):
            row.update(
                {
                    "status": "GENERATION_INFEASIBLE",
                    "reason": f"{type(base).__name__}:{base}",
                }
            )
        else:
            try:
                generation = _place_base_generation(inputs, base, point.yaw_rad)
            except (RuntimeError, ValueError) as error:
                row.update(
                    {
                        "status": "GENERATION_INFEASIBLE",
                        "reason": f"{type(error).__name__}:{error}",
                    }
                )
            else:
                path, reason = _evaluate_control_path(
                    inputs, scene, generation["control_plan"], settings
                )
                if path is None:
                    row.update(
                        {"status": "PATH_INFEASIBLE", "reason": reason}
                    )
                else:
                    pose = np.asarray(
                        generation["control_plan"][
                            "object_from_hand_row_major"
                        ],
                        dtype=np.float64,
                    ).reshape(4, 4)
                    quality = _task_quality(
                        inputs,
                        settings,
                        pose,
                        path.contact_joint_positions_rad,
                        path.patches,
                    )
                    if quality is None:
                        row.update(
                            {
                                "status": "TASK_INFEASIBLE",
                                "reason": "TASK_LOAD_LINEAR_PROGRAM_INFEASIBLE",
                                "path": _path_summary(path),
                            }
                        )
                    elif quality.task_load_factor < 1.0:
                        row.update(
                            {
                                "status": "TASK_INFEASIBLE",
                                "reason": "TASK_LOAD_FACTOR_BELOW_ONE",
                                "path": _path_summary(path),
                                "quality": _quality_record(quality),
                            }
                        )
                    else:
                        try:
                            motion_plan = _solve_or_reuse_motion_plan(
                                repository,
                                inputs,
                                generation["control_plan"],
                                point.yaw_index,
                                motion_cache,
                                seed_by_yaw,
                                baseline_seed,
                            )
                        except (RuntimeError, ValueError) as error:
                            row.update(
                                {
                                    "status": "ARM_IK_OR_LIFT_PATH_INFEASIBLE",
                                    "reason": f"{type(error).__name__}:{error}",
                                    "path": _path_summary(path),
                                    "quality": _quality_record(quality),
                                }
                            )
                        else:
                            row.update(
                                {
                                    "status": "EXECUTABLE_OFFLINE",
                                    "reason": "",
                                    "path": _path_summary(path),
                                    "quality": _quality_record(quality),
                                }
                            )
                            friction_margin = None
                            if friction_margin_selection_enabled:
                                friction_margin = _minimum_task_friction_record(
                                    inputs,
                                    settings,
                                    pose,
                                    path.contact_joint_positions_rad,
                                    path.patches,
                                    quality,
                                )
                                row["friction_margin"] = friction_margin
                                key = (
                                    -float(
                                        friction_margin[
                                            "minimum_required_friction_coefficient"
                                        ]
                                    ),
                                    quality.task_load_factor,
                                    path.minimum_forbidden_clearance_m,
                                    path.minimum_commanded_hand_joint_limit_margin_rad,
                                    path.minimum_closure_reserve_rad,
                                    -point.palm_index,
                                    -point.yaw_index,
                                    -point.axial_index,
                                )
                            elif robust_selection_enabled:
                                robustness = _robust_candidate_summary(
                                    inputs,
                                    scene,
                                    settings,
                                    generation["control_plan"],
                                    path,
                                    quality,
                                    robust_scenarios,
                                )
                                row["robustness"] = robustness
                                key = None
                                if robustness["status"] == "ROBUST_EXECUTABLE_OFFLINE":
                                    key = tuple(
                                        robustness[name]
                                        for name in ROBUST_SELECTION_METRICS
                                    ) + (
                                        -point.palm_index,
                                        -point.yaw_index,
                                        -point.axial_index,
                                    )
                            else:
                                key = (
                                    quality.task_load_factor,
                                    path.minimum_forbidden_clearance_m,
                                    path.minimum_commanded_hand_joint_limit_margin_rad,
                                    path.minimum_closure_reserve_rad,
                                    -point.palm_index,
                                    -point.yaw_index,
                                    -point.axial_index,
                                )
                            if key is not None and (
                                selected_key is None or key > selected_key
                            ):
                                selected_key = key
                                selected = (point, generation, path, quality)
                                selected_motion_plan = motion_plan
                                if robust_selection_enabled:
                                    selected_robustness = robustness
                                if friction_margin_selection_enabled:
                                    selected_friction_margin = friction_margin
        if robust_selection_enabled and "robustness" not in row:
            row["robustness"] = {
                "status": "ROBUST_INFEASIBLE_NOMINAL",
                "reason": f"{row['status']}:{row['reason']}",
                "failure_scenario": "nominal",
                "offline_evaluated_scenarios": ["nominal"],
            }
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        rows.append(row)
        if evaluated_index % 100 == 0 or evaluated_index == declared_count:
            print(
                json.dumps(
                    {
                        "stage": "candidate_evaluation",
                        "evaluated": evaluated_index,
                        "declared": declared_count,
                        "current_status": row["status"],
                    }
                ),
                flush=True,
            )

    if len(rows) != declared_count:
        raise RuntimeError("declared finite set was not completely evaluated")
    selected_record = None
    if selected is not None:
        selected_motion_plan, selected_arm_route = _select_arm_route(
            repository, inputs, selected[1]["control_plan"]
        )
        selected[1]["control_plan"]["approach_high_seed_arm_positions_rad"] = (
            list(selected_motion_plan["approach_arm_waypoints_rad"][0])
        )
        selected_record = _selected_detail(*selected)
        if robust_selection_enabled:
            if selected_robustness is None:
                raise RuntimeError("robust selected candidate lacks robustness data")
            selected_record["robustness"] = selected_robustness
        if friction_margin_selection_enabled:
            if selected_friction_margin is None:
                raise RuntimeError(
                    "friction-margin selected candidate lacks its score"
                )
            selected_record["friction_margin"] = selected_friction_margin
    robust_selection_identity = (
        "BEST_MINIMUM_FORBIDDEN_CLEARANCE_THEN_TASK_LOAD_IN_FROZEN_"
        "AXIS_ALIGNED_FINITE_G_DELTA_OVER_OFFLINE_EXPRESSIBLE_"
        "ROBUSTNESS_COMPONENTS"
    )
    friction_margin_selection_identity = (
        "MINIMUM_TASK_FEASIBLE_FRICTION_ON_FROZEN_FINITE_FRICTION_GRID_"
        "IN_FROZEN_AXIS_ALIGNED_FINITE_G_DELTA"
    )
    search_claim = (
        "NO_OPTIMALITY_CLAIM_SINGLE_DECLARED_G_DELTA_MEMBER_REPLAY"
        if candidate_replay_only
        else
        (
            robust_selection_identity
            if selected_record is not None
            else (
                "FROZEN_AXIS_ALIGNED_FINITE_G_DELTA_HAS_NO_MEMBER_"
                "FEASIBLE_IN_ALL_OFFLINE_EXPRESSIBLE_ROBUSTNESS_COMPONENTS"
            )
        )
        if robust_selection_enabled
        else (
            (
                friction_margin_selection_identity
                if selected_record is not None
                else "FROZEN_AXIS_ALIGNED_FINITE_G_DELTA_HAS_NO_EXECUTABLE_MEMBER"
            )
            if friction_margin_selection_enabled
            else (
                "BEST_IN_FROZEN_AXIS_ALIGNED_FINITE_G_DELTA"
                if selected_record is not None
                else "FROZEN_AXIS_ALIGNED_FINITE_G_DELTA_HAS_NO_EXECUTABLE_MEMBER"
            )
        )
    )
    result = {
        "schema_version": (
            "finite_cad_nailfree_search_v4_single_member_replay_v1"
            if candidate_replay_only
            else "finite_cad_nailfree_search_v4_robust_selection_v1"
            if robust_selection_enabled
            else (
                "finite_cad_nailfree_search_v4_friction_margin_selection_v1"
                if friction_margin_selection_enabled
                else "finite_cad_nailfree_search_v4"
            )
        ),
        "object_id": arguments.object_id,
        "hand_variant": inputs.hand_variant,
        "hardware_authorized": False,
        "formal_dynamic_pass": False,
        "contact_mechanics": {
            "full_pad_geometry_used_for_collision_and_semantics": True,
            "force_model": settings["full_pad_contact_force_model"],
            "resultant_point_rule": settings["contact_resultant_point_rule"],
            "resultant_normal_rule": settings["contact_resultant_normal_rule"],
            "independent_contact_torsional_moment_allowed": False,
            "model_boundary": (
                "EACH_COMPLETE_PAD REMAINS A FINITE CONTACT REGION; "
                "ITS ONE DETECTED COLLISION PATCH CONTRIBUTES ONE RESULTANT "
                "FRICTIONAL POINT CONTACT TO THE TASK-WRENCH MODEL"
            ),
        },
        "execution_margin_contract": {
            "maximum_preload_increment_rad": float(
                dynamic["preload_increment_rad"]
            ),
            "maximum_preload_increment_rule": settings[
                "maximum_preload_increment_rule"
            ],
            "commanded_hand_joint_limit_margin_rule": settings[
                "commanded_hand_joint_limit_margin_rule"
            ],
            "pre_lift_effort_rule": (
                "EACH_CLOSING_JOINT_REACHES_COMPONENTWISE_MAXIMUM_"
                "OF_HOLD_AND_LIFT_UNIT_TASK_LP_EFFORT"
            ),
        },
        "search_claim": search_claim,
        "claim_does_not_cover_continuous_or_tilted_or_laterally_offset_grasps": True,
        "grid": {
            "palm_rad": palms.tolist(),
            "yaw_rad": yaws.tolist(),
            "cad_reference_section_z_m": axial.tolist(),
            "lateral_offset_task_m": [0.0, 0.0],
            "tilt_task_rad": [0.0, 0.0],
            "declared_candidate_count": declared_count,
            "evaluated_candidate_count": len(rows),
            "source_g_delta_declared_candidate_count": source_declared_count,
            "replay_candidate_id": replay_candidate_id,
        },
        "status_counts": counts,
        "unique_hand_base_motion_targets_evaluated": len(motion_cache),
        "candidates": rows,
        "selected_candidate": selected_record,
        "selected_motion_plan": selected_motion_plan,
        "selected_arm_route": selected_arm_route,
        "elapsed_s": time.monotonic() - started,
    }
    if robust_selection_enabled:
        dynamic_panel = _dynamic_palm_component_panel(rows)
        dynamic_panel["dynamic_score_contract"] = {
            "formula": (
                "R_TASK=I_SAFETY*I_LEGAL_FULL_PAD*I_TABLE_RELEASE*"
                "MIN(1,MAXIMUM_LIFT_M/LIFT_TARGET_M,"
                "SUSPENDED_HOLD_S/HOLD_TARGET_S)"
            ),
            "lift_target_m": float(dynamic["lift_distance_m"]),
            "hold_target_s": float(dynamic["hold_duration_s"]),
            "suspended_hold_rule": (
                "HOLD_DURATION_COUNTS_ONLY_WHILE_TABLE_CONTACT_IS_RELEASED"
            ),
            "safety_indicator_rule": (
                "ALL_EXISTING_FINITE_SIGNAL_JOINT_SPEED_ARM_TRACKING_"
                "HAND_EFFORT_UNAUTHORIZED_CONTACT_AND_PENETRATION_GATES"
            ),
            "finite_perturbation_aggregation": "MINIMUM_R_TASK_OVER_DECLARED_U_DELTA",
            "execution_retention_s_ret": (
                "SECONDARY_POST_ROLLOUT_DIAGNOSTIC_ONLY_NOT_A_STATIC_PREDICTOR"
            ),
            "weighted_diagnostic_terms": False,
        }
        result["robustness_selection"] = {
            "enabled": True,
            "selection_rule": robust_selection_identity,
            "selection_key_order": [
                "MAXIMIZE_WORST_MINIMUM_FORBIDDEN_CLEARANCE_M",
                "MAXIMIZE_WORST_TASK_LOAD_FACTOR",
                "MAXIMIZE_WORST_MINIMUM_COMMANDED_HAND_JOINT_LIMIT_MARGIN_RAD",
                "MAXIMIZE_WORST_MINIMUM_CLOSURE_RESERVE_RAD",
                "FIXED_ASCENDING_PALM_YAW_AXIAL_GRID_INDEX",
            ],
            "offline_evaluated_scenarios": ["nominal"]
            + [
                scenario.name
                for scenario in robust_scenarios[1:]
                if scenario.offline_components
            ],
            "dynamic_only_not_in_offline_selection": [
                {
                    "scenario": scenario.name,
                    "components": list(scenario.dynamic_only_components),
                }
                for scenario in robust_scenarios
                if scenario.dynamic_only_components
            ],
            "finger_preload_scale_scope": (
                "dynamic_only_not_in_offline_selection"
            ),
            "collision_scope": (
                "EXISTING_SAMPLED_HAND_APPROACH_AND_SEQUENTIAL_CLOSURE_"
                "COLLISION_MODEL; PERTURBED_FULL_ARM_ENVIRONMENT_CLEARANCE_"
                "REQUIRES_DYNAMIC_PREFLIGHT"
            ),
            "dynamic_candidate_panel": dynamic_panel,
        }
    if friction_margin_selection_enabled:
        result["friction_margin_selection"] = {
            "enabled": True,
            "selection_rule": friction_margin_selection_identity,
            "selection_key_order": [
                "MINIMIZE_FIRST_TASK_FEASIBLE_FRICTION_GRID_COEFFICIENT",
                "MAXIMIZE_TASK_LOAD_FACTOR_AT_CONTACT_CONTRACT_LOWER_FRICTION",
                "MAXIMIZE_MINIMUM_FORBIDDEN_CLEARANCE_M",
                "MAXIMIZE_MINIMUM_COMMANDED_HAND_JOINT_LIMIT_MARGIN_RAD",
                "MAXIMIZE_MINIMUM_CLOSURE_RESERVE_RAD",
                "FIXED_ASCENDING_PALM_YAW_AXIAL_GRID_INDEX",
            ],
            "physical_scope": (
                "STATIC_TASK_WRENCH_AND_JOINT_LIMIT_DIAGNOSTIC_ONLY; "
                "DOES_NOT_PROVE_DYNAMIC_CONTACT_ESTABLISHMENT_OR_RETENTION"
            ),
        }
    if candidate_replay_only:
        result["single_member_replay"] = {
            "enabled": True,
            "candidate_id": replay_candidate_id,
            "source_g_delta_declared_candidate_count": source_declared_count,
            "optimality_claim": False,
        }
    (output / "search_result.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if selected is not None:
        point = selected[0]
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        derived = copy.deepcopy(document)
        generation_settings = derived["nominal_grasp_generation"]
        generation_settings["palm_joint_position_rad"] = point.palm_rad
        generation_settings["grasp_axis_position_m"] = point.axial_m
        generation_settings["hand_yaw_rad"] = point.yaw_rad
        generation_settings["approach_high_seed_arm_positions_rad"] = list(
            selected_motion_plan["approach_arm_waypoints_rad"][0]
        )
        if candidate_replay_only and robust_selection_enabled:
            if selected_robustness is None:
                raise RuntimeError(
                    "robust replay candidate lacks robustness effort data"
                )
            generation_settings["selected_by"] = (
                "REPLAYED_FROZEN_G_DELTA_MEMBER_WITH_FROZEN_V5_"
                "ROBUST_EFFORT_NO_OPTIMALITY_CLAIM"
            )
            generation_settings["required_closing_joint_effort_nm"] = dict(
                selected_robustness["required_closing_joint_effort_nm"]
            )
            generation_settings["required_closing_joint_effort_source"] = (
                "COMPONENTWISE_MAXIMUM_OF_HOLD_AND_LIFT_UNIT_TASK_LP_"
                "OVER_ALL_OFFLINE_EXPRESSIBLE_ROBUSTNESS_SCENARIOS"
            )
        elif candidate_replay_only:
            generation_settings["selected_by"] = (
                "REPLAYED_FROZEN_G_DELTA_MEMBER_NO_OPTIMALITY_CLAIM"
            )
            generation_settings["required_closing_joint_effort_nm"] = dict(
                zip(CLOSING_JOINTS, selected[3].required_closing_joint_effort_nm)
            )
            generation_settings["required_closing_joint_effort_source"] = (
                "COMPONENTWISE_MAXIMUM_OF_HOLD_AND_LIFT_UNIT_TASK_LP_AT_"
                "FROZEN_FRICTION_LOWER_BOUND"
            )
        elif robust_selection_enabled:
            assert selected_robustness is not None
            generation_settings["selected_by"] = robust_selection_identity
            generation_settings["required_closing_joint_effort_nm"] = dict(
                selected_robustness["required_closing_joint_effort_nm"]
            )
            generation_settings["required_closing_joint_effort_source"] = (
                "COMPONENTWISE_MAXIMUM_OF_HOLD_AND_LIFT_UNIT_TASK_LP_"
                "OVER_ALL_OFFLINE_EXPRESSIBLE_ROBUSTNESS_SCENARIOS"
            )
        elif friction_margin_selection_enabled:
            generation_settings["selected_by"] = (
                friction_margin_selection_identity
            )
            generation_settings["required_closing_joint_effort_nm"] = dict(
                zip(CLOSING_JOINTS, selected[3].required_closing_joint_effort_nm)
            )
            generation_settings["required_closing_joint_effort_source"] = (
                "COMPONENTWISE_MAXIMUM_OF_HOLD_AND_LIFT_UNIT_TASK_LP_AT_"
                "FROZEN_FRICTION_LOWER_BOUND"
            )
        else:
            generation_settings["selected_by"] = (
                "BEST_IN_FROZEN_AXIS_ALIGNED_FINITE_G_DELTA"
            )
            generation_settings["required_closing_joint_effort_nm"] = dict(
                zip(CLOSING_JOINTS, selected[3].required_closing_joint_effort_nm)
            )
            generation_settings["required_closing_joint_effort_source"] = (
                "COMPONENTWISE_MAXIMUM_OF_HOLD_AND_LIFT_UNIT_TASK_LP_AT_"
                "FROZEN_FRICTION_LOWER_BOUND"
            )
        (output / "selected_config.yaml").write_text(
            yaml.safe_dump(derived, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "object_id": arguments.object_id,
                "search_claim": result["search_claim"],
                "selected_candidate": (
                    None
                    if selected_record is None
                    else {
                        "candidate_id": selected_record["candidate_id"],
                        "quality": selected_record["quality"],
                    }
                ),
                "elapsed_s": result["elapsed_s"],
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
