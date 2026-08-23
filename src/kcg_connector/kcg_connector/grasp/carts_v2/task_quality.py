"""Evaluate task-load robustness and actuator burden for V2 candidates."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from kcg_connector.grasp.carts_v2.models import (
    ClosurePrediction,
    TaskQualityResult,
    V2Inputs,
)
from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
    deterministic_sobol,
)
from kcg_connector.grasp.robust.hand_model import ThreeFingerHandModel
from kcg_connector.grasp.robust.pareto_ranker import qmc_lower_tail_mean
from kcg_connector.grasp.robust.robust_wrench import (
    LinearProgramSolverOptions,
    friction_cone_inner_relative_error,
)
from kcg_connector.grasp.robust.task_wrench_evaluator import (
    FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE,
    TaskWrenchEvaluationError,
    TaskWrenchEvaluator,
)


_SCENARIO_DIMENSION = 26


def minimum_jerk_peak_acceleration(distance_m: float, duration_s: float) -> float:
    """Peak absolute acceleration of 10u^3 - 15u^4 + 6u^5."""

    if distance_m <= 0.0 or duration_s <= 0.0:
        raise ValueError("minimum-jerk distance and duration must be positive")
    return (10.0 * math.sqrt(3.0) / 3.0) * distance_m / duration_s**2


def common_uncertainty_design(inputs: V2Inputs) -> np.ndarray:
    settings = inputs.config.section("task_quality")
    return deterministic_sobol(
        dimension=_SCENARIO_DIMENSION,
        count=int(settings["scenario_count"]),
        seed=int(settings["random_seed"]),
    )


def _bounded_vector(values: np.ndarray, radius: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return radius * vector / max(1.0, norm)


def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle <= np.finfo(np.float64).eps:
        return np.eye(3)
    axis = vector / angle
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _perturb_normal(
    normal: np.ndarray, unit_values: np.ndarray, maximum_angle: float
) -> np.ndarray:
    axis_seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
    tangent_1 = np.cross(normal, axis_seed)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    tangent_offset = _bounded_vector(unit_values, math.tan(maximum_angle))
    result = normal + tangent_offset[0] * tangent_1 + tangent_offset[1] * tangent_2
    return result / np.linalg.norm(result)


def _perturbed_candidate(
    prediction: ClosurePrediction, signed: np.ndarray, settings
) -> GraspCandidate:
    candidate = prediction.grasp_candidate
    assert candidate is not None
    common_shift = _bounded_vector(
        signed[8:11], float(settings["common_contact_position_error_m"])
    )
    individual = signed[11:20].reshape(3, 3)
    normal_values = signed[20:26].reshape(3, 2)
    contacts = []
    for index, contact in enumerate(candidate.planned_pad_contacts):
        point = np.asarray(contact.position_object_m) + common_shift
        point += _bounded_vector(
            individual[index], float(settings["contact_position_error_m"])
        )
        normal = _perturb_normal(
            np.asarray(contact.path_local_free_side_normal_object),
            normal_values[index],
            float(settings["contact_normal_error_rad"]),
        )
        contacts.append(
            PlannedPadContact(
                pad_name=contact.pad_name,
                position_object_m=tuple(float(value) for value in point),
                path_local_free_side_normal_object=tuple(
                    float(value) for value in normal
                ),
                surface_coordinates=contact.surface_coordinates,
            )
        )
    return GraspCandidate(
        object_from_hand=candidate.object_from_hand,
        independent_joint_positions_rad=candidate.independent_joint_positions_rad,
        planned_pad_contacts=tuple(contacts),
        internal_normal_forces_n=candidate.internal_normal_forces_n,
        stiffness_diagonal=candidate.stiffness_diagonal,
        damping_diagonal=candidate.damping_diagonal,
    )


def _operation_hand(inputs: V2Inputs, normal_force_cap_n: float) -> ThreeFingerHandModel:
    source = inputs.hand_model
    pads = {
        name: replace(pad, normal_force_capacity_n=normal_force_cap_n)
        for name, pad in source.pads.items()
    }
    return ThreeFingerHandModel(
        base_link=source.base_link,
        joints=source.joints,
        joint_order=source.joint_order,
        finger_joint_names={
            name: finger.joint_names for name, finger in source.fingers.items()
        },
        pads=pads,
    )


def _evaluator(
    inputs: V2Inputs,
    *,
    model,
    friction: float,
    gravity_direction: np.ndarray,
) -> TaskWrenchEvaluator:
    settings = inputs.config.section("task_quality")
    edge_count = int(settings["friction_cone_edges"])
    cone_error = float(
        np.nextafter(friction_cone_inner_relative_error(edge_count), np.inf)
    )
    dynamic = inputs.config.section("dynamic")
    lift_acceleration = minimum_jerk_peak_acceleration(
        float(dynamic["lift_distance_m"]), float(dynamic["lift_duration_s"])
    )
    return TaskWrenchEvaluator(
        object_model=model,
        characteristic_radius_m=inputs.object_contract.characteristic_radius_m,
        friction_coefficient_interval=(friction, friction),
        uncertainty_claim_scope=FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE,
        gravity_direction_object=gravity_direction,
        task_frame_rotation_object=inputs.object_contract.task_frame_rotation_object,
        gravity_acceleration_m_s2=float(settings["gravity_acceleration_m_s2"]),
        lift_acceleration_m_s2=lift_acceleration,
        maximum_inner_approximation_relative_error=cone_error,
        cone_edge_multiplier=1,
        solver_options=LinearProgramSolverOptions.from_mapping(
            settings["linear_program"]
        ),
    )


def _evaluate_scenario(
    inputs: V2Inputs,
    prediction: ClosurePrediction,
    unit: np.ndarray,
    hand: ThreeFingerHandModel,
    settings,
    friction_interval: tuple[float, float],
) -> tuple[float | None, float | None, float | None, str]:
    signed = 2.0 * unit - 1.0
    mass = inputs.object_contract.model.mass_kg * (
        1.0 + float(settings["mass_relative_error"]) * signed[1]
    )
    com = inputs.object_contract.model.center_of_mass_m + _bounded_vector(
        signed[2:5], float(settings["center_of_mass_error_m"])
    )
    model = replace(inputs.object_contract.model, mass_kg=mass, center_of_mass_m=com)
    rotation_vector = _bounded_vector(
        signed[5:8], float(settings["gravity_direction_error_rad"])
    )
    gravity = _rotation_from_vector(rotation_vector).T @ (
        inputs.object_contract.nominal_validation_gravity_direction_object
    )
    friction = friction_interval[0] + unit[0] * (
        friction_interval[1] - friction_interval[0]
    )
    try:
        result = _evaluator(
            inputs,
            model=model,
            friction=float(friction),
            gravity_direction=gravity,
        ).evaluate_task_wrench(
            _perturbed_candidate(prediction, signed, settings),
            np.asarray(((0.5,),)),
            hand_model=hand,
        )
    except TaskWrenchEvaluationError as error:
        return (None, None, None, error.reason_code + ":" + str(error))
    return (
        float(result.hard_bound_minimum_task_margin),
        result.prescribed_task_peak_normal_force_n,
        result.prescribed_task_joint_torque_utilization,
        "",
    )


def _rejected_quality(
    candidate_id: str,
    margins: tuple[float | None, ...],
    reason: str,
    nominal_failure_count: int = 0,
) -> TaskQualityResult:
    return TaskQualityResult(
        candidate_id=candidate_id,
        status="TASK_REJECT",
        scenario_margins=margins,
        worst_task_margin=None,
        lower_tail_mean_margin=None,
        required_peak_normal_force_n=None,
        maximum_joint_load_utilization=None,
        maximum_generalized_joint_torque_nm=None,
        wrist_load_utilization=None,
        sensitivity=None,
        failure_reason=reason,
        nominal_balance_infeasible_count=nominal_failure_count,
    )


def _nominal_infeasible_quality(
    candidate_id: str,
    margins: tuple[float | None, ...],
    lower_tail_fraction: float,
    nominal_failure_count: int,
) -> TaskQualityResult:
    """Rank proven nominal-load failures at a conservative zero lower bound."""

    ordering_margins = tuple(
        0.0 if value is None else float(value) for value in margins
    )
    return TaskQualityResult(
        candidate_id=candidate_id,
        status="TASK_REJECT",
        scenario_margins=margins,
        worst_task_margin=0.0,
        lower_tail_mean_margin=qmc_lower_tail_mean(
            ordering_margins, lower_tail_fraction
        ),
        required_peak_normal_force_n=None,
        maximum_joint_load_utilization=None,
        maximum_generalized_joint_torque_nm=None,
        wrist_load_utilization=None,
        sensitivity=max(ordering_margins) - min(ordering_margins),
        failure_reason=(
            "NOMINAL_LOAD_INFEASIBLE_ZERO_IS_RANKING_LOWER_BOUND_NOT_FEASIBILITY"
        ),
        nominal_balance_infeasible_count=nominal_failure_count,
    )


def evaluate_task_quality(
    inputs: V2Inputs,
    prediction: ClosurePrediction,
    scenario_design: np.ndarray,
) -> TaskQualityResult:
    """Evaluate all preregistered error rows; any failed row rejects the grasp."""

    if prediction.grasp_candidate is None:
        return _rejected_quality(
            prediction.seed.candidate_id,
            (),
            "NO_THREE_CONTACT_GRASP_CANDIDATE",
        )
    design = np.asarray(scenario_design, dtype=np.float64)
    if design.ndim != 2 or design.shape[1] != _SCENARIO_DIMENSION:
        raise ValueError("V2 uncertainty design has the wrong dimension")
    settings = inputs.config.section("task_quality")
    force_cap = float(settings["normal_force_operation_cap_n"])
    hand = _operation_hand(inputs, force_cap)
    friction_interval = (
        inputs.object_contract.contact_material_uncertainty.friction_coefficient_interval
    )
    margins: list[float | None] = []
    forces: list[float] = []
    joint_loads: list[float] = []
    prescribed_infeasible = False
    nominal_balance_failure_count = 0
    for row_index, unit in enumerate(design):
        margin, force, joint_load, failure = _evaluate_scenario(
            inputs, prediction, unit, hand, settings, friction_interval
        )
        if failure:
            if failure.startswith("NOMINAL_LOAD_INFEASIBLE:"):
                margins.append(None)
                prescribed_infeasible = True
                nominal_balance_failure_count += 1
                continue
            return _rejected_quality(
                prediction.seed.candidate_id,
                tuple(margins),
                f"SCENARIO_{row_index}:{failure}",
                nominal_balance_failure_count,
            )
        assert margin is not None
        margins.append(margin)
        if force is None or joint_load is None:
            prescribed_infeasible = True
        else:
            forces.append(float(force))
            joint_loads.append(float(joint_load))
    if nominal_balance_failure_count:
        return _nominal_infeasible_quality(
            prediction.seed.candidate_id,
            tuple(margins),
            float(settings["lower_tail_fraction"]),
            nominal_balance_failure_count,
        )
    finite_margins = [float(value) for value in margins if value is not None]
    worst = min(finite_margins)
    survives = worst >= 1.0 and not prescribed_infeasible
    efforts = np.asarray(
        [
            hand.independent_joint_limits[name].effort
            for name in hand.independent_joint_names
        ],
        dtype=np.float64,
    )
    common_effort = float(efforts[0]) if np.all(efforts == efforts[0]) else None
    return TaskQualityResult(
        candidate_id=prediction.seed.candidate_id,
        status="TASK_SURVIVE" if survives else "TASK_REJECT",
        scenario_margins=tuple(finite_margins),
        worst_task_margin=worst,
        lower_tail_mean_margin=qmc_lower_tail_mean(
            finite_margins, float(settings["lower_tail_fraction"])
        ),
        required_peak_normal_force_n=max(forces) if survives else None,
        maximum_joint_load_utilization=None,
        maximum_generalized_joint_torque_nm=(
            max(joint_loads) * common_effort
            if survives and common_effort is not None
            else None
        ),
        wrist_load_utilization=None,
        sensitivity=max(finite_margins) - min(finite_margins),
        failure_reason=(
            ""
            if survives
            else (
                "PRESCRIBED_TASK_SCALE_INFEASIBLE_UNDER_OPERATION_CAP"
            )
        ),
        nominal_balance_infeasible_count=0,
    )


__all__ = [
    "common_uncertainty_design",
    "evaluate_task_quality",
    "minimum_jerk_peak_acceleration",
]
