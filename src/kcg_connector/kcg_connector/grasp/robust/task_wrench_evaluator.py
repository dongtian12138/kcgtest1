"""Task-wrench evaluation for object-independent CARTS-Grasp candidates.

Coordinate convention:

* object_from_hand maps hand-base coordinates into the object frame;
* planned positions and outward normals are expressed in the object frame;
* contact forces are forces exerted by the hand on the object;
* a compressive force is opposite the certified path-local free-side normal;
  equivalence to a solid-outward normal requires the separate complete
  external-first-contact collision certificate;
* external gravity points along gravity_direction_object;
* positive lift_acceleration_m_s2 denotes acceleration opposite gravity.

All wrenches are taken about the object's centre of mass.  The nominal
gravity/lift wrench therefore has zero moment, while a contact force contributes
(p_contact - p_COM) cross f.  No object name, stored grasp pose, or historical
candidate lookup participates in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    ObjectSurfaceModel,
    WrenchEvaluation,
)
from kcg_connector.grasp.robust.hand_model import (
    HandModelError,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.object_model import ObjectGraspModel
from kcg_connector.grasp.robust.robust_wrench import (
    LinearProgramSolverOptions,
    TaskWrenchMarginResult,
    build_polyhedral_contact_wrench_model,
    maximum_task_wrench_polytope_margin,
)


_FLOAT_EPS = np.finfo(np.float64).eps
FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE = (
    "CERTIFIED_SET_CONTAINS_ONLY_DECLARED_FRICTION_INTERVAL_"
    "ALL_OTHER_UNCERTAINTIES_REQUIRE_CALIBRATION"
)
COMPLETE_CONTINUOUS_TRAJECTORY_CLEARANCE_SCOPE = (
    "COMPLETE_HAND_OBJECT_ENVIRONMENT_APPROACH_CLOSURE_LIFT_"
    "CONTINUOUS_COLLISION_CERTIFIED_LOWER_BOUND"
)


class TaskWrenchEvaluationError(RuntimeError):
    """Fail-closed result when a margin cannot be numerically certified."""


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def _deep_freeze(value: Any) -> Any:
    """Return an immutable diagnostic snapshot with no writable arrays."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, np.ndarray):
        return _deep_freeze(value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _finite_vector(
    value: Sequence[float], *, shape: tuple[int, ...], label: str
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must have finite shape {shape}")
    return result


def _unit_vector(value: Sequence[float], *, label: str) -> np.ndarray:
    result = _finite_vector(value, shape=(3,), label=label)
    norm = float(np.linalg.norm(result))
    if norm <= _FLOAT_EPS:
        raise ValueError(f"{label} is numerically zero")
    return result / norm


def _proper_rotation(
    value: Sequence[Sequence[float]], *, label: str
) -> np.ndarray:
    rotation = _finite_vector(value, shape=(3, 3), label=label)
    # Roundoff check, not a physical gate.  The multiplier bounds the floating
    # operations in two 3-by-3 products and a determinant.
    roundoff = 64.0 * _FLOAT_EPS
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=np.inf)
    )
    determinant = float(np.linalg.det(rotation))
    if (
        orthogonality_error > roundoff
        or determinant <= 0.0
        or abs(determinant - 1.0) > roundoff
    ):
        raise ValueError(f"{label} must be a proper orthonormal rotation")
    return rotation


def _rigid_transform(value: Sequence[float], *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be finite")
    roundoff = 64.0 * _FLOAT_EPS
    if float(np.linalg.norm(matrix[3] - (0.0, 0.0, 0.0, 1.0))) > roundoff:
        raise ValueError(f"{label} must have a homogeneous final row")
    _proper_rotation(matrix[:3, :3], label=f"{label} rotation")
    return matrix


@dataclass(frozen=True)
class ContactActuationModel:
    """Linear contact-to-independent-joint map in the object frame."""

    torque_from_object_contact_forces: np.ndarray
    independent_joint_effort_limits: np.ndarray
    contact_points_link_m: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        torque_map = np.asarray(
            self.torque_from_object_contact_forces, dtype=np.float64
        )
        effort = np.asarray(self.independent_joint_effort_limits, dtype=np.float64)
        if (
            torque_map.ndim != 2
            or torque_map.shape[0] != effort.size
            or torque_map.shape[1] != 3 * len(self.contact_points_link_m)
            or not np.all(np.isfinite(torque_map))
            or effort.shape != (torque_map.shape[0],)
            or not np.all(np.isfinite(effort))
            or np.any(effort <= 0.0)
        ):
            raise ValueError("contact actuation model has inconsistent finite dimensions")
        object.__setattr__(
            self, "torque_from_object_contact_forces", _readonly(torque_map)
        )
        object.__setattr__(
            self, "independent_joint_effort_limits", _readonly(effort)
        )


@dataclass(frozen=True)
class TaskWrenchDefinition:
    """The preregistered nominal wrench and 12-vertex disturbance body."""

    nominal_external_wrench: np.ndarray
    disturbance_vertices: np.ndarray
    force_scale_n: float
    moment_scale_nm: float
    wrench_origin_object_m: np.ndarray

    def __post_init__(self) -> None:
        nominal = _finite_vector(
            self.nominal_external_wrench,
            shape=(6,),
            label="nominal_external_wrench",
        )
        vertices = _finite_vector(
            self.disturbance_vertices,
            shape=(12, 6),
            label="disturbance_vertices",
        )
        origin = _finite_vector(
            self.wrench_origin_object_m,
            shape=(3,),
            label="wrench_origin_object_m",
        )
        if (
            not math.isfinite(self.force_scale_n)
            or self.force_scale_n <= 0.0
            or not math.isfinite(self.moment_scale_nm)
            or self.moment_scale_nm <= 0.0
        ):
            raise ValueError("task wrench scales must be finite and positive")
        object.__setattr__(self, "nominal_external_wrench", _readonly(nominal))
        object.__setattr__(self, "disturbance_vertices", _readonly(vertices))
        object.__setattr__(self, "wrench_origin_object_m", _readonly(origin))


@dataclass(frozen=True)
class TaskWrenchOnlyEvaluation:
    """Collision-independent task-wrench result for one common QMC design.

    This type deliberately has no trajectory-clearance field.  It allows every
    unique static V9 candidate to receive exactly one wrench evaluation even
    while the independent complete-trajectory collision certificate remains
    unresolved.  Consequently it is diagnostic evidence, not by itself a
    formally selectable grasp.
    """

    task_margins: tuple[float, ...]
    hard_bound_minimum_task_margin: float
    peak_normal_force_n: float
    joint_torque_utilization: float
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        margins = tuple(float(value) for value in self.task_margins)
        scalars = (
            float(self.hard_bound_minimum_task_margin),
            float(self.peak_normal_force_n),
            float(self.joint_torque_utilization),
        )
        if not margins or not all(
            math.isfinite(value) for value in margins + scalars
        ):
            raise ValueError(
                "task-wrench-only evaluation must contain finite margins"
            )
        if scalars[0] < 0.0:
            raise ValueError("hard-bound task margin cannot be negative")
        if scalars[1] < 0.0 or scalars[2] < 0.0:
            raise ValueError(
                "task-wrench-only force and torque burdens cannot be negative"
            )
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("task-wrench-only diagnostics must be a mapping")
        object.__setattr__(self, "task_margins", margins)
        object.__setattr__(
            self, "hard_bound_minimum_task_margin", scalars[0]
        )
        object.__setattr__(self, "peak_normal_force_n", scalars[1])
        object.__setattr__(self, "joint_torque_utilization", scalars[2])
        object.__setattr__(self, "diagnostics", _deep_freeze(self.diagnostics))


class TaskWrenchEvaluator:
    """Common-Sobol friction evaluation with finite PAD/actuator constraints.

    The certified set in this evaluator contains only the explicitly declared
    friction interval.  Pose, surface, mass-property, joint-tracking and
    actuator residuals require a separate calibrated contract and are not
    silently represented by zero-width intervals.
    """

    uncertainty_dimension = 1

    def __init__(
        self,
        *,
        object_model: ObjectGraspModel,
        characteristic_radius_m: float,
        friction_coefficient_interval: Sequence[float],
        uncertainty_claim_scope: str,
        gravity_direction_object: Sequence[float],
        task_frame_rotation_object: Sequence[Sequence[float]],
        gravity_acceleration_m_s2: float,
        lift_acceleration_m_s2: float,
        maximum_inner_approximation_relative_error: float,
        cone_edge_multiplier: int,
        solver_options: LinearProgramSolverOptions,
    ) -> None:
        if not isinstance(object_model, ObjectGraspModel):
            raise TypeError("object_model must be an ObjectGraspModel")
        radius = float(characteristic_radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("characteristic_radius_m must be finite and positive")
        friction = _finite_vector(
            friction_coefficient_interval,
            shape=(2,),
            label="friction_coefficient_interval",
        )
        if friction[0] < 0.0 or friction[1] < friction[0]:
            raise ValueError("friction interval must satisfy 0 <= lower <= upper")
        if (
            uncertainty_claim_scope
            != FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ):
            raise ValueError(
                "uncertainty_claim_scope must explicitly limit certification "
                "to the declared friction interval"
            )
        gravity_acceleration = float(gravity_acceleration_m_s2)
        lift_acceleration = float(lift_acceleration_m_s2)
        if not math.isfinite(gravity_acceleration) or gravity_acceleration <= 0.0:
            raise ValueError("gravity_acceleration_m_s2 must be finite and positive")
        if not math.isfinite(lift_acceleration):
            raise ValueError("lift_acceleration_m_s2 must be finite")
        cone_error = float(maximum_inner_approximation_relative_error)
        if not math.isfinite(cone_error) or not 0.0 < cone_error < 1.0:
            raise ValueError(
                "maximum_inner_approximation_relative_error must lie in (0, 1)"
            )
        if (
            isinstance(cone_edge_multiplier, bool)
            or int(cone_edge_multiplier) != cone_edge_multiplier
            or cone_edge_multiplier < 1
        ):
            raise ValueError("cone_edge_multiplier must be a positive integer")
        if not isinstance(solver_options, LinearProgramSolverOptions):
            raise TypeError("solver_options must be LinearProgramSolverOptions")

        self.object_model = object_model
        self.characteristic_radius_m = radius
        self.friction_coefficient_interval = tuple(float(item) for item in friction)
        self.uncertainty_claim_scope = uncertainty_claim_scope
        self.gravity_direction_object = _readonly(
            _unit_vector(
                gravity_direction_object, label="gravity_direction_object"
            )
        )
        self.task_frame_rotation_object = _readonly(
            _proper_rotation(
                task_frame_rotation_object,
                label="task_frame_rotation_object",
            )
        )
        self.gravity_acceleration_m_s2 = gravity_acceleration
        self.lift_acceleration_m_s2 = lift_acceleration
        self.maximum_inner_approximation_relative_error = cone_error
        self.cone_edge_multiplier = int(cone_edge_multiplier)
        self.solver_options = solver_options
        self.task_wrench_definition = self._build_task_wrench_definition()

    def _build_task_wrench_definition(self) -> TaskWrenchDefinition:
        mass = self.object_model.mass_kg
        force_scale = mass * self.gravity_acceleration_m_s2
        moment_scale = force_scale * self.characteristic_radius_m
        task_axes = self.task_frame_rotation_object
        vertices = np.zeros((12, 6), dtype=np.float64)
        for axis_index in range(3):
            force_vertex = force_scale * task_axes[:, axis_index]
            moment_vertex = moment_scale * task_axes[:, axis_index]
            vertices[2 * axis_index, :3] = force_vertex
            vertices[2 * axis_index + 1, :3] = -force_vertex
            vertices[6 + 2 * axis_index, 3:] = moment_vertex
            vertices[6 + 2 * axis_index + 1, 3:] = -moment_vertex

        # d points with gravity.  Acceleration -a_lift*d requires contact force
        # -m(g+a_lift)d, so the external load balanced by the hand is positive.
        nominal = np.zeros(6, dtype=np.float64)
        nominal[:3] = (
            mass
            * (self.gravity_acceleration_m_s2 + self.lift_acceleration_m_s2)
            * self.gravity_direction_object
        )
        return TaskWrenchDefinition(
            nominal_external_wrench=nominal,
            disturbance_vertices=vertices,
            force_scale_n=force_scale,
            moment_scale_nm=moment_scale,
            wrench_origin_object_m=self.object_model.center_of_mass_m,
        )

    def friction_coefficients_from_unit(
        self, scenario_parameters_unit: np.ndarray
    ) -> np.ndarray:
        """Map the common one-dimensional Sobol design into declared friction."""

        scenarios = np.asarray(scenario_parameters_unit, dtype=np.float64)
        if (
            scenarios.ndim != 2
            or scenarios.shape[1] != self.uncertainty_dimension
            or scenarios.shape[0] < 1
            or not np.all(np.isfinite(scenarios))
            or np.any(scenarios < 0.0)
            or np.any(scenarios > 1.0)
        ):
            raise ValueError(
                "scenario_parameters_unit must have finite shape (S, 1) in [0, 1]"
            )
        lower, upper = self.friction_coefficient_interval
        return _readonly(lower + scenarios[:, 0] * (upper - lower))

    def _contact_tangent(
        self, inward_normal_object: np.ndarray
    ) -> np.ndarray:
        """Choose a task-frame tangent equivariantly, without an angle gate."""

        for axis_index in range(3):
            axis = self.task_frame_rotation_object[:, axis_index]
            tangent = axis - float(axis @ inward_normal_object) * inward_normal_object
            norm = float(np.linalg.norm(tangent))
            if norm > _FLOAT_EPS:
                return tangent / norm
        raise TaskWrenchEvaluationError(
            "proper task frame produced no finite contact tangent"
        )

    def independent_joint_torque_map(
        self,
        candidate: GraspCandidate,
        hand_model: ThreeFingerHandModel,
    ) -> ContactActuationModel:
        """Return tau = T f_object using independent/mimic-aware Jacobians."""

        transform_object_from_hand = _rigid_transform(
            candidate.object_from_hand,
            label="candidate.object_from_hand",
        )
        rotation_object_from_hand = transform_object_from_hand[:3, :3]
        rotation_hand_from_object = rotation_object_from_hand.T
        translation_object_from_hand = transform_object_from_hand[:3, 3]
        joint_positions = candidate.independent_joint_positions_rad
        try:
            link_transforms = hand_model.forward_kinematics(joint_positions)
        except HandModelError as error:
            raise TaskWrenchEvaluationError(
                f"hand forward kinematics failed: {error}"
            ) from error

        joint_names = tuple(hand_model.independent_joint_names)
        effort_limits: list[float] = []
        for joint_name in joint_names:
            joint_limit = hand_model.independent_joint_limits[joint_name]
            effort = joint_limit.effort
            if effort is None or not math.isfinite(float(effort)) or effort <= 0.0:
                raise TaskWrenchEvaluationError(
                    f"independent joint {joint_name} has no finite positive effort limit"
                )
            effort_limits.append(float(effort))

        torque_map = np.zeros(
            (len(joint_names), 3 * len(candidate.planned_pad_contacts)),
            dtype=np.float64,
        )
        contact_points_link: list[tuple[float, float, float]] = []
        for contact_index, contact in enumerate(candidate.planned_pad_contacts):
            pad = hand_model.pads.get(contact.pad_name)
            if pad is None:
                raise TaskWrenchEvaluationError(
                    f"planned contact names unknown PAD {contact.pad_name!r}"
                )
            point_object = np.asarray(contact.position_object_m, dtype=np.float64)
            point_hand = rotation_hand_from_object @ (
                point_object - translation_object_from_hand
            )
            transform_hand_from_link = np.asarray(
                link_transforms[pad.link_name], dtype=np.float64
            )
            rotation_hand_from_link = transform_hand_from_link[:3, :3]
            point_link = rotation_hand_from_link.T @ (
                point_hand - transform_hand_from_link[:3, 3]
            )
            try:
                jacobian_hand = hand_model.geometric_jacobian(
                    pad.link_name,
                    joint_positions,
                    point_local_m=point_link,
                )
            except HandModelError as error:
                raise TaskWrenchEvaluationError(
                    f"PAD Jacobian failed for {contact.pad_name}: {error}"
                ) from error
            expected_shape = (6, len(joint_names))
            if (
                jacobian_hand.shape != expected_shape
                or not np.all(np.isfinite(jacobian_hand))
            ):
                raise TaskWrenchEvaluationError(
                    f"PAD Jacobian has shape {jacobian_hand.shape}, expected {expected_shape}"
                )
            torque_map[
                :, 3 * contact_index : 3 * contact_index + 3
            ] = jacobian_hand[:3].T @ rotation_hand_from_object
            contact_points_link.append(tuple(float(item) for item in point_link))

        return ContactActuationModel(
            torque_from_object_contact_forces=torque_map,
            independent_joint_effort_limits=np.asarray(effort_limits),
            contact_points_link_m=tuple(contact_points_link),
        )

    def _contact_inputs(
        self,
        candidate: GraspCandidate,
        hand_model: ThreeFingerHandModel,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        contact_names = tuple(
            contact.pad_name for contact in candidate.planned_pad_contacts
        )
        if len(set(contact_names)) != len(contact_names):
            raise TaskWrenchEvaluationError("planned PAD contacts are not unique")
        if set(contact_names) != set(hand_model.pads):
            raise TaskWrenchEvaluationError(
                "candidate must contain exactly one contact for every physical PAD"
            )

        points = np.asarray(
            [contact.position_object_m for contact in candidate.planned_pad_contacts],
            dtype=np.float64,
        )
        free_side = np.asarray(
            [
                contact.path_local_free_side_normal_object
                for contact in candidate.planned_pad_contacts
            ],
            dtype=np.float64,
        )
        compressive_axis = -free_side
        tangents = np.asarray(
            [self._contact_tangent(normal) for normal in compressive_axis],
            dtype=np.float64,
        )
        capacities: list[float] = []
        for name in contact_names:
            capacity = hand_model.pads[name].normal_force_capacity_n
            if (
                capacity is None
                or not math.isfinite(float(capacity))
                or capacity <= 0.0
            ):
                raise TaskWrenchEvaluationError(
                    f"PAD {name} has no finite positive normal-force capacity"
                )
            capacities.append(float(capacity))
        preload = np.asarray(candidate.internal_normal_forces_n, dtype=np.float64)
        capacity_array = np.asarray(capacities, dtype=np.float64)
        if preload.shape != capacity_array.shape or np.any(preload < 0.0):
            raise TaskWrenchEvaluationError(
                "candidate preload must contain one non-negative value per PAD"
            )
        if np.any(preload > capacity_array):
            raise TaskWrenchEvaluationError(
                "candidate preload exceeds a physical PAD normal-force capacity"
            )
        return points, compressive_axis, tangents, capacity_array, preload

    def _constraint_matrix(
        self,
        *,
        actuation: ContactActuationModel,
        inward_normals: np.ndarray,
        preload_normal_forces_n: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        torque_map = actuation.torque_from_object_contact_forces
        effort = actuation.independent_joint_effort_limits
        contact_count = inward_normals.shape[0]
        preload_rows = np.zeros((contact_count, 3 * contact_count), dtype=np.float64)
        for index, normal in enumerate(inward_normals):
            preload_rows[index, 3 * index : 3 * index + 3] = -normal
        matrix = np.vstack((torque_map, -torque_map, preload_rows))
        bounds = np.concatenate((effort, effort, -preload_normal_forces_n))
        return matrix, bounds

    def _solve_margin(
        self,
        *,
        candidate: GraspCandidate,
        hand_model: ThreeFingerHandModel,
        friction_coefficient: float,
        actuation: ContactActuationModel,
        contact_inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> TaskWrenchMarginResult:
        del candidate, hand_model
        points, compressive_axes, tangents, capacities, preload = contact_inputs
        constraint_matrix, constraint_bounds = self._constraint_matrix(
            actuation=actuation,
            inward_normals=compressive_axes,
            preload_normal_forces_n=preload,
        )
        model = build_polyhedral_contact_wrench_model(
            contact_points_m=points,
            inward_normals=compressive_axes,
            tangent_directions=tangents,
            friction_coefficients=float(friction_coefficient),
            normal_force_caps_n=capacities,
            wrench_origin_m=self.object_model.center_of_mass_m,
            maximum_inner_approximation_relative_error=(
                self.maximum_inner_approximation_relative_error
            ),
            cone_edge_multiplier=self.cone_edge_multiplier,
            contact_force_inequality_matrix=constraint_matrix,
            contact_force_inequality_upper_bounds=constraint_bounds,
        )
        torque_from_rays = (
            actuation.torque_from_object_contact_forces
            @ model.contact_force_matrix
        )
        torque_utilization_from_rays = torque_from_rays / (
            actuation.independent_joint_effort_limits[:, None]
        )
        signed_torque_utilization_from_rays = np.vstack(
            (
                torque_utilization_from_rays,
                -torque_utilization_from_rays,
            )
        )
        result = maximum_task_wrench_polytope_margin(
            model,
            nominal_external_wrench=(
                self.task_wrench_definition.nominal_external_wrench
            ),
            disturbance_vertices=self.task_wrench_definition.disturbance_vertices,
            solver_options=self.solver_options,
            lexicographic_ray_load_groups=(
                model.normal_force_matrix,
                signed_torque_utilization_from_rays,
            ),
        )
        if not result.solver_success or result.maximum_margin is None:
            raise TaskWrenchEvaluationError(
                "task-wrench LP failed closed: "
                f"status={result.solver_status}, message={result.solver_message}"
            )
        if (
            result.maximum_scaled_equilibrium_residual is None
            or result.maximum_scaled_inequality_violation is None
            or result.maximum_scaled_equilibrium_residual
            > self.solver_options.primal_feasibility_tolerance
            or result.maximum_scaled_inequality_violation
            > self.solver_options.primal_feasibility_tolerance
        ):
            raise TaskWrenchEvaluationError(
                "task-wrench LP certificate exceeds the explicit primal tolerance"
            )
        if not math.isfinite(result.maximum_margin) or result.maximum_margin < 0.0:
            raise TaskWrenchEvaluationError(
                "task-wrench LP returned an invalid non-finite/negative margin"
            )
        if (
            result.lexicographic_optimal_loads is None
            or len(result.lexicographic_optimal_loads) != 2
            or not all(
                math.isfinite(float(value)) and float(value) >= 0.0
                for value in result.lexicographic_optimal_loads
            )
        ):
            raise TaskWrenchEvaluationError(
                "task-wrench LP omitted its two certified lexicographic loads"
            )
        expected_stage_names = (
            "MAXIMIZE_SHARED_TASK_MARGIN",
            "MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_0",
            "MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_1",
        )
        stage_results = result.lexicographic_stage_results
        if (
            tuple(stage.stage_name for stage in stage_results)
            != expected_stage_names
            or any(
                not stage.solver_success
                or stage.optimal_value is None
                or stage.maximum_scaled_equilibrium_residual is None
                or stage.maximum_scaled_inequality_violation is None
                or stage.maximum_scaled_equilibrium_residual
                > self.solver_options.primal_feasibility_tolerance
                or stage.maximum_scaled_inequality_violation
                > self.solver_options.primal_feasibility_tolerance
                for stage in stage_results
            )
        ):
            raise TaskWrenchEvaluationError(
                "task-wrench LP omitted a certified lexicographic stage"
            )
        expected_stage_values = (
            float(result.maximum_margin),
            float(result.lexicographic_optimal_loads[0]),
            float(result.lexicographic_optimal_loads[1]),
        )
        if tuple(stage.optimal_value for stage in stage_results) != (
            expected_stage_values
        ):
            raise TaskWrenchEvaluationError(
                "task-wrench LP stage objectives disagree with its certificate"
            )
        if (
            result.ray_coefficients_by_vertex is None
            or result.contact_forces_by_vertex is None
            or result.normal_forces_by_vertex is None
        ):
            raise TaskWrenchEvaluationError(
                "successful lexicographic LP omitted its final-stage allocation"
            )
        ray_coefficients = np.asarray(result.ray_coefficients_by_vertex)
        contact_forces = np.asarray(result.contact_forces_by_vertex)
        normal_forces = np.asarray(result.normal_forces_by_vertex)
        vertex_count = self.task_wrench_definition.disturbance_vertices.shape[0]
        if (
            ray_coefficients.shape != (vertex_count, model.ray_count)
            or contact_forces.shape
            != (vertex_count, model.contact_count, 3)
            or normal_forces.shape != (vertex_count, model.contact_count)
            or not np.all(np.isfinite(ray_coefficients))
            or not np.all(np.isfinite(contact_forces))
            or not np.all(np.isfinite(normal_forces))
        ):
            raise TaskWrenchEvaluationError(
                "task-wrench LP returned malformed final-stage allocations"
            )
        return result

    @staticmethod
    def _trajectory_clearance(
        *,
        surface_model: ObjectSurfaceModel,
        candidate: GraspCandidate,
        hand_model: ThreeFingerHandModel,
    ) -> float:
        scope = getattr(surface_model, "trajectory_clearance_scope", None)
        if scope != COMPLETE_CONTINUOUS_TRAJECTORY_CLEARANCE_SCOPE:
            raise TaskWrenchEvaluationError(
                "surface model trajectory clearance is not certified for the "
                "complete hand-object-environment approach/closure/lift path"
            )
        method = getattr(surface_model, "trajectory_clearance_m", None)
        if not callable(method):
            raise TaskWrenchEvaluationError(
                "surface model must explicitly provide trajectory_clearance_m"
            )
        clearance = float(method(candidate, hand_model))
        if not math.isfinite(clearance):
            raise TaskWrenchEvaluationError(
                "surface trajectory clearance must be finite"
            )
        return clearance

    def evaluate_task_wrench(
        self,
        candidate: GraspCandidate,
        scenario_parameters_unit: np.ndarray,
        *,
        hand_model: ThreeFingerHandModel,
    ) -> TaskWrenchOnlyEvaluation:
        """Evaluate one candidate once, without inventing collision clearance."""

        friction_values = self.friction_coefficients_from_unit(
            scenario_parameters_unit
        )
        contact_inputs = self._contact_inputs(candidate, hand_model)
        actuation = self.independent_joint_torque_map(candidate, hand_model)
        scenario_results = tuple(
            self._solve_margin(
                candidate=candidate,
                hand_model=hand_model,
                friction_coefficient=float(friction),
                actuation=actuation,
                contact_inputs=contact_inputs,
            )
            for friction in friction_values
        )
        lower_bound_result = self._solve_margin(
            candidate=candidate,
            hand_model=hand_model,
            friction_coefficient=self.friction_coefficient_interval[0],
            actuation=actuation,
            contact_inputs=contact_inputs,
        )
        all_results = scenario_results + (lower_bound_result,)
        margins = tuple(float(result.maximum_margin) for result in scenario_results)

        lexicographic_loads = tuple(
            result.lexicographic_optimal_loads for result in all_results
        )
        if any(loads is None for loads in lexicographic_loads):
            raise TaskWrenchEvaluationError(
                "successful LP omitted its lexicographic load certificate"
            )
        peak_normal_force = max(
            float(loads[0]) for loads in lexicographic_loads if loads is not None
        )
        maximum_torque_utilization = max(
            float(loads[1]) for loads in lexicographic_loads if loads is not None
        )

        def stage_diagnostics(
            result: TaskWrenchMarginResult,
        ) -> tuple[Mapping[str, Any], ...]:
            return tuple(
                MappingProxyType(
                    {
                        "stage_name": stage.stage_name,
                        "solver_success": stage.solver_success,
                        "solver_status": stage.solver_status,
                        "solver_message": stage.solver_message,
                        "optimal_value": stage.optimal_value,
                        "maximum_scaled_equilibrium_residual": (
                            stage.maximum_scaled_equilibrium_residual
                        ),
                        "maximum_scaled_inequality_violation": (
                            stage.maximum_scaled_inequality_violation
                        ),
                    }
                )
                for stage in result.lexicographic_stage_results
            )

        diagnostics: Mapping[str, Any] = MappingProxyType(
            {
                "uncertainty_parameter_names": ("friction_coefficient",),
                "certified_uncertainty_scope": self.uncertainty_claim_scope,
                "uncalibrated_uncertainties_excluded_from_certified_set": (
                    "object_pose",
                    "surface_position_and_normal",
                    "mass_center_of_mass_and_inertia",
                    "joint_tracking",
                    "actuator_capability",
                ),
                "friction_coefficients": tuple(
                    float(value) for value in friction_values
                ),
                "hard_bound_friction_coefficient": float(
                    self.friction_coefficient_interval[0]
                ),
                "disturbance_body": "CENTRALLY_SYMMETRIC_6D_CROSS_POLYTOPE",
                "disturbance_vertex_count": 12,
                "force_scale_n": self.task_wrench_definition.force_scale_n,
                "moment_scale_nm": self.task_wrench_definition.moment_scale_nm,
                "nominal_external_wrench": tuple(
                    float(value)
                    for value in self.task_wrench_definition.nominal_external_wrench
                ),
                "wrench_origin_object_m": tuple(
                    float(value)
                    for value in self.task_wrench_definition.wrench_origin_object_m
                ),
                "nominal_moment_about_com_nm": (0.0, 0.0, 0.0),
                "gravity_direction_sign_convention": (
                    "EXTERNAL_GRAVITY_ALONG_DECLARED_DIRECTION;"
                    "POSITIVE_LIFT_ACCELERATION_OPPOSES_GRAVITY"
                ),
                "normal_force_capacity_n": tuple(
                    float(value) for value in contact_inputs[3]
                ),
                "independent_joint_effort_limits": tuple(
                    float(value)
                    for value in actuation.independent_joint_effort_limits
                ),
                "independent_joint_effort_limit_role": (
                    "URDF_DECLARED_UNCALIBRATED_OPTIMIZATION_CONSTRAINT"
                ),
                "independent_joint_torque_map_shape": tuple(
                    int(value)
                    for value in actuation.torque_from_object_contact_forces.shape
                ),
                "lexicographic_load_group_roles": (
                    "PEAK_PAD_NORMAL_FORCE_N",
                    "PEAK_ABSOLUTE_INDEPENDENT_JOINT_TORQUE_UTILIZATION",
                ),
                "scenario_lexicographic_optimal_loads": tuple(
                    tuple(float(value) for value in result.lexicographic_optimal_loads)
                    for result in scenario_results
                    if result.lexicographic_optimal_loads is not None
                ),
                "hard_bound_lexicographic_optimal_loads": tuple(
                    float(value)
                    for value in lower_bound_result.lexicographic_optimal_loads
                ),
                "scenario_lexicographic_stage_reports": tuple(
                    stage_diagnostics(result) for result in scenario_results
                ),
                "hard_bound_lexicographic_stage_reports": stage_diagnostics(
                    lower_bound_result
                ),
                "solver": self.solver_options.solver,
                "requested_constraint_scaling": (
                    self.solver_options.requested_constraint_scaling
                ),
                "actual_constraint_scaling": scenario_results[
                    0
                ].constraint_scaling_implementation,
                "residual_coordinate_system": (
                    "EXPLICIT_EQUILIBRATED_SOLVER_COORDINATES"
                ),
                "primal_feasibility_tolerance": (
                    self.solver_options.primal_feasibility_tolerance
                ),
                "dual_feasibility_tolerance": (
                    self.solver_options.dual_feasibility_tolerance
                ),
                "ipm_optimality_tolerance": (
                    self.solver_options.ipm_optimality_tolerance
                ),
            }
        )
        return TaskWrenchOnlyEvaluation(
            task_margins=margins,
            hard_bound_minimum_task_margin=float(
                lower_bound_result.maximum_margin
            ),
            peak_normal_force_n=peak_normal_force,
            joint_torque_utilization=maximum_torque_utilization,
            diagnostics=diagnostics,
        )

    def evaluate(
        self,
        candidate: GraspCandidate,
        scenario_parameters_unit: np.ndarray,
        *,
        surface_model: ObjectSurfaceModel,
        hand_model: ThreeFingerHandModel,
    ) -> WrenchEvaluation:
        """Compatibility API that combines wrench and certified clearance."""

        wrench = self.evaluate_task_wrench(
            candidate,
            scenario_parameters_unit,
            hand_model=hand_model,
        )
        clearance = self._trajectory_clearance(
            surface_model=surface_model,
            candidate=candidate,
            hand_model=hand_model,
        )
        diagnostics = dict(wrench.diagnostics)
        diagnostics["trajectory_clearance_scope"] = (
            COMPLETE_CONTINUOUS_TRAJECTORY_CLEARANCE_SCOPE
        )
        return WrenchEvaluation(
            task_margins=wrench.task_margins,
            hard_bound_minimum_task_margin=(
                wrench.hard_bound_minimum_task_margin
            ),
            peak_normal_force_n=wrench.peak_normal_force_n,
            joint_torque_utilization=wrench.joint_torque_utilization,
            trajectory_clearance_m=clearance,
            feasible=True,
            diagnostics=_deep_freeze(diagnostics),
        )


__all__ = [
    "COMPLETE_CONTINUOUS_TRAJECTORY_CLEARANCE_SCOPE",
    "ContactActuationModel",
    "FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE",
    "TaskWrenchOnlyEvaluation",
    "TaskWrenchDefinition",
    "TaskWrenchEvaluationError",
    "TaskWrenchEvaluator",
]
