"""Tests for the coordinate-explicit CARTS task-wrench evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
)
from kcg_connector.grasp.robust.hand_model import ThreeFingerHandModel
from kcg_connector.grasp.robust.object_model import (
    AssetProvenance,
    ObjectGraspModel,
    TriangleMesh,
)
from kcg_connector.grasp.robust.robust_wrench import (
    LinearProgramSolverOptions,
)
import kcg_connector.grasp.robust.task_wrench_evaluator as evaluator_module
from kcg_connector.grasp.robust.task_wrench_evaluator import (
    COMPLETE_CONTINUOUS_TRAJECTORY_CLEARANCE_SCOPE,
    FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE,
    TaskWrenchEvaluationError,
    TaskWrenchEvaluator,
    TaskWrenchOnlyEvaluation,
)


REPOSITORY = Path(__file__).resolve().parents[4]
HAND_URDF = REPOSITORY / "src/iiwa_description/urdf/hand.xacro"

LP_OPTIONS = LinearProgramSolverOptions.from_mapping(
    {
        "solver": "SCIPY_HIGHS",
        "constraint_scaling": "ROW_AND_COLUMN_INF_NORM",
        "maximum_iterations": 10000,
        "primal_feasibility_tolerance": 1.0e-9,
        "dual_feasibility_tolerance": 1.0e-9,
        "ipm_optimality_tolerance": 1.0e-10,
        "physical_acceptance_gate": False,
    }
)


def _object_model(
    *,
    mass_kg: float = 1.0,
    center_of_mass_m=(0.0, 0.0, 0.0),
) -> ObjectGraspModel:
    vertices = np.asarray(
        (
            (+0.20, +0.00, -0.10),
            (-0.10, +0.18, -0.10),
            (-0.10, -0.18, -0.10),
            (+0.00, +0.00, +0.20),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        dtype=np.int64,
    )
    mesh = TriangleMesh(
        vertices_m=vertices,
        faces=faces,
        face_semantics=("external",) * 4,
    )
    provenance = AssetProvenance(
        source_path="synthetic_fixture.stl",
        source_sha256="0" * 64,
        source_class="SYNTHETIC_ANALYTIC_TEST_FIXTURE",
        source_format="ASCII_STL",
        source_unit="m",
        meters_per_source_unit=1.0,
    )
    return ObjectGraspModel(
        mesh=mesh,
        provenance=provenance,
        assembly_axis=np.asarray((0.0, 0.0, 1.0)),
        mass_kg=mass_kg,
        center_of_mass_m=np.asarray(center_of_mass_m, dtype=np.float64),
        inertia_kg_m2=np.diag((0.01, 0.01, 0.02)),
        allowed_contact_semantics=frozenset(("external",)),
    )


@dataclass(frozen=True)
class _Pad:
    link_name: str
    normal_force_capacity_n: float


@dataclass(frozen=True)
class _Limit:
    effort: float


class _AnalyticHand:
    """One zero-torque generalized coordinate and three physical PADs."""

    independent_joint_names = ("q",)

    def __init__(self, normal_force_capacity_n: float) -> None:
        self.pads = MappingProxyType(
            {
                f"pad_{index}": _Pad(
                    link_name=f"link_{index}",
                    normal_force_capacity_n=float(normal_force_capacity_n),
                )
                for index in range(3)
            }
        )
        self.independent_joint_limits = MappingProxyType(
            {"q": _Limit(effort=1.0e6)}
        )

    def forward_kinematics(self, positions):
        assert tuple(positions) == (0.0,)
        return MappingProxyType(
            {
                "base": np.eye(4),
                **{pad.link_name: np.eye(4) for pad in self.pads.values()},
            }
        )

    def geometric_jacobian(
        self, link_name, positions, *, point_local_m, base_transform=None
    ):
        del point_local_m, base_transform
        assert link_name in {pad.link_name for pad in self.pads.values()}
        assert tuple(positions) == (0.0,)
        return np.zeros((6, 1), dtype=np.float64)


class _TorqueLimitedHand(_AnalyticHand):
    def __init__(
        self, normal_force_capacity_n: float, independent_effort_limit: float
    ) -> None:
        super().__init__(normal_force_capacity_n)
        self.independent_joint_limits = MappingProxyType(
            {"q": _Limit(effort=float(independent_effort_limit))}
        )

    def geometric_jacobian(
        self, link_name, positions, *, point_local_m, base_transform=None
    ):
        jacobian = super().geometric_jacobian(
            link_name,
            positions,
            point_local_m=point_local_m,
            base_transform=base_transform,
        )
        jacobian[2, 0] = 1.0
        return jacobian


class _Surface:
    parameter_dimension = 1
    trajectory_clearance_scope = (
        COMPLETE_CONTINUOUS_TRAJECTORY_CLEARANCE_SCOPE
    )

    @staticmethod
    def trajectory_clearance_m(candidate, hand_model) -> float:
        del candidate, hand_model
        return 0.01


def _symmetric_candidate(
    *,
    object_from_hand: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> GraspCandidate:
    frame_rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    frame_translation = (
        np.zeros(3) if translation is None else np.asarray(translation)
    )
    contacts = []
    radius_m = 0.10
    for index, angle in enumerate((0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)):
        radial = np.asarray((math.cos(angle), math.sin(angle), 0.0))
        contacts.append(
            PlannedPadContact(
                pad_name=f"pad_{index}",
                position_object_m=tuple(
                    frame_rotation @ (radius_m * radial) + frame_translation
                ),
                path_local_free_side_normal_object=tuple(
                    frame_rotation @ radial
                ),
            )
        )
    transform = np.eye(4) if object_from_hand is None else object_from_hand
    return GraspCandidate.from_matrix(
        object_from_hand=transform,
        independent_joint_positions_rad=(0.0,),
        planned_pad_contacts=contacts,
        internal_normal_forces_n=(0.0, 0.0, 0.0),
    )


def _evaluator(
    object_model: ObjectGraspModel,
    *,
    friction_interval=(0.5, 0.5),
    characteristic_radius_m: float = 0.005,
    gravity_direction=(0.0, 0.0, -1.0),
    task_frame_rotation=np.eye(3),
    gravity_acceleration_m_s2: float = 1.0,
    lift_acceleration_m_s2: float = 0.0,
) -> TaskWrenchEvaluator:
    return TaskWrenchEvaluator(
        object_model=object_model,
        characteristic_radius_m=characteristic_radius_m,
        friction_coefficient_interval=friction_interval,
        uncertainty_claim_scope=(
            FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ),
        gravity_direction_object=gravity_direction,
        task_frame_rotation_object=task_frame_rotation,
        gravity_acceleration_m_s2=gravity_acceleration_m_s2,
        lift_acceleration_m_s2=lift_acceleration_m_s2,
        maximum_inner_approximation_relative_error=0.30,
        cone_edge_multiplier=1,
        solver_options=LP_OPTIONS,
    )


def _evaluate(
    evaluator: TaskWrenchEvaluator,
    hand,
    candidate: GraspCandidate,
    scenarios=((0.5,),),
):
    return evaluator.evaluate(
        candidate,
        np.asarray(scenarios, dtype=np.float64),
        surface_model=_Surface(),
        hand_model=hand,
    )


def test_twelve_vertex_definition_and_gravity_lift_sign_about_com() -> None:
    model = _object_model(
        mass_kg=2.0,
        center_of_mass_m=(0.08, -0.03, 0.11),
    )
    evaluator = _evaluator(
        model,
        characteristic_radius_m=0.25,
        gravity_direction=(0.0, 0.0, -1.0),
        gravity_acceleration_m_s2=3.0,
        lift_acceleration_m_s2=0.5,
    )
    definition = evaluator.task_wrench_definition
    assert definition.force_scale_n == pytest.approx(6.0)
    assert definition.moment_scale_nm == pytest.approx(1.5)
    np.testing.assert_allclose(
        definition.nominal_external_wrench,
        (0.0, 0.0, -7.0, 0.0, 0.0, 0.0),
        atol=0.0,
    )
    np.testing.assert_allclose(
        definition.wrench_origin_object_m,
        model.center_of_mass_m,
        atol=0.0,
    )
    assert definition.disturbance_vertices.shape == (12, 6)
    for pair_start in range(0, 12, 2):
        np.testing.assert_allclose(
            definition.disturbance_vertices[pair_start]
            + definition.disturbance_vertices[pair_start + 1],
            np.zeros(6),
            atol=0.0,
        )
    np.testing.assert_allclose(
        np.linalg.norm(definition.disturbance_vertices[:6, :3], axis=1),
        np.full(6, 6.0),
    )
    np.testing.assert_allclose(
        np.linalg.norm(definition.disturbance_vertices[6:, 3:], axis=1),
        np.full(6, 1.5),
    )


def test_three_finger_symmetric_margin_matches_vertical_load_analysis() -> None:
    model = _object_model(mass_kg=1.0)
    hand = _AnalyticHand(normal_force_capacity_n=10.0)
    evaluator = _evaluator(
        model,
        friction_interval=(0.1, 0.1),
        characteristic_radius_m=0.005,
        gravity_acceleration_m_s2=1.0,
    )
    result = _evaluate(evaluator, hand, _symmetric_candidate())
    # The limiting vertex adds one downward weight to nominal gravity.  Three
    # PADs supply 3*mu*F vertical force, so rho=3*mu*F/(mg)-1.
    expected = 3.0 * 0.1 * 10.0 / 1.0 - 1.0
    assert result.task_margins == pytest.approx((expected,), rel=1.0e-9)
    assert result.hard_bound_minimum_task_margin == pytest.approx(
        expected, rel=1.0e-9
    )
    # Zero planned preload is a lower bound, not a command that suppresses the
    # non-zero contact forces needed by the task-wrench certificate.
    assert result.peak_normal_force_n > 0.0
    assert result.diagnostics["lexicographic_load_group_roles"] == (
        "PEAK_PAD_NORMAL_FORCE_N",
        "PEAK_ABSOLUTE_INDEPENDENT_JOINT_TORQUE_UTILIZATION",
    )
    hard_bound_stages = result.diagnostics[
        "hard_bound_lexicographic_stage_reports"
    ]
    assert tuple(stage["stage_name"] for stage in hard_bound_stages) == (
        "MAXIMIZE_SHARED_TASK_MARGIN",
        "MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_0",
        "MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_1",
    )
    assert all(stage["solver_success"] for stage in hard_bound_stages)
    assert all(
        stage["maximum_scaled_equilibrium_residual"]
        <= LP_OPTIONS.primal_feasibility_tolerance
        and stage["maximum_scaled_inequality_violation"]
        <= LP_OPTIONS.primal_feasibility_tolerance
        for stage in hard_bound_stages
    )


def test_preload_above_physical_pad_capacity_fails_closed() -> None:
    candidate = _symmetric_candidate()
    over_capacity = GraspCandidate.from_matrix(
        object_from_hand=candidate.object_from_hand_matrix(),
        independent_joint_positions_rad=candidate.independent_joint_positions_rad,
        planned_pad_contacts=candidate.planned_pad_contacts,
        internal_normal_forces_n=(10.0 + 1.0, 0.0, 0.0),
    )
    with pytest.raises(TaskWrenchEvaluationError, match="exceeds"):
        _evaluate(
            _evaluator(_object_model()),
            _AnalyticHand(10.0),
            over_capacity,
        )


def test_margin_is_monotone_in_friction_and_physical_pad_capacity() -> None:
    model = _object_model()
    candidate = _symmetric_candidate()
    low_mu = _evaluate(
        _evaluator(model, friction_interval=(0.30, 0.30)),
        _AnalyticHand(10.0),
        candidate,
    ).task_margins[0]
    high_mu = _evaluate(
        _evaluator(model, friction_interval=(0.60, 0.60)),
        _AnalyticHand(10.0),
        candidate,
    ).task_margins[0]
    low_capacity = _evaluate(
        _evaluator(model, friction_interval=(0.50, 0.50)),
        _AnalyticHand(6.0),
        candidate,
    ).task_margins[0]
    high_capacity = _evaluate(
        _evaluator(model, friction_interval=(0.50, 0.50)),
        _AnalyticHand(12.0),
        candidate,
    ).task_margins[0]
    assert high_mu >= low_mu
    assert high_capacity >= low_capacity


def test_weight_and_disturbance_scale_with_mass_without_retuning() -> None:
    light = _evaluator(_object_model(mass_kg=1.0))
    heavy = _evaluator(_object_model(mass_kg=2.0))
    assert heavy.task_wrench_definition.force_scale_n == pytest.approx(
        2.0 * light.task_wrench_definition.force_scale_n
    )
    assert heavy.task_wrench_definition.moment_scale_nm == pytest.approx(
        2.0 * light.task_wrench_definition.moment_scale_nm
    )
    np.testing.assert_allclose(
        heavy.task_wrench_definition.nominal_external_wrench,
        2.0 * light.task_wrench_definition.nominal_external_wrench,
    )
    hand = _AnalyticHand(10.0)
    candidate = _symmetric_candidate()
    light_margin = _evaluate(light, hand, candidate).task_margins[0]
    heavy_margin = _evaluate(heavy, hand, candidate).task_margins[0]
    assert heavy_margin < light_margin


def test_independent_joint_effort_constraint_is_bilateral() -> None:
    model = _object_model()
    candidate = _symmetric_candidate()
    hand = _TorqueLimitedHand(
        normal_force_capacity_n=10.0,
        independent_effort_limit=2.0,
    )
    downward_gravity = _evaluate(
        _evaluator(model, gravity_direction=(0.0, 0.0, -1.0)),
        hand,
        candidate,
    )
    upward_gravity = _evaluate(
        _evaluator(model, gravity_direction=(0.0, 0.0, 1.0)),
        hand,
        candidate,
    )
    # tau=J^T f has equal +effort and -effort inequalities.  Reversing gravity
    # therefore preserves the margin instead of disabling one torque sign.
    assert downward_gravity.task_margins == pytest.approx(
        upward_gravity.task_margins,
        rel=1.0e-9,
        abs=1.0e-9,
    )
    assert downward_gravity.joint_torque_utilization == pytest.approx(1.0)
    assert upward_gravity.joint_torque_utilization == pytest.approx(1.0)


def _rotation() -> np.ndarray:
    cz, sz = math.cos(0.61), math.sin(0.61)
    cy, sy = math.cos(-0.37), math.sin(-0.37)
    return np.asarray(
        (
            (cz * cy, -sz, cz * sy),
            (sz * cy, cz, sz * sy),
            (-sy, 0.0, cy),
        )
    )


def test_margin_is_equivariant_when_object_task_and_candidate_are_transformed() -> None:
    model = _object_model()
    hand = _AnalyticHand(10.0)
    candidate = _symmetric_candidate()
    base = _evaluator(model)
    reference = _evaluate(base, hand, candidate).task_margins

    rotation = _rotation()
    translation = np.asarray((0.31, -0.27, 0.14))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transformed_model = model.transformed(transform)
    transformed_candidate = _symmetric_candidate(
        object_from_hand=transform,
        rotation=rotation,
        translation=translation,
    )
    transformed_evaluator = _evaluator(
        transformed_model,
        gravity_direction=rotation @ np.asarray((0.0, 0.0, -1.0)),
        task_frame_rotation=rotation,
    )
    transformed = _evaluate(
        transformed_evaluator,
        hand,
        transformed_candidate,
    ).task_margins
    assert transformed == pytest.approx(reference, rel=1.0e-9, abs=1.0e-9)


def test_task_frame_is_mandatory_proper_rotation_and_never_guessed() -> None:
    model = _object_model()
    required = dict(
        object_model=model,
        characteristic_radius_m=0.01,
        friction_coefficient_interval=(0.4, 0.6),
        uncertainty_claim_scope=(
            FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ),
        gravity_direction_object=(0.0, 0.0, -1.0),
        gravity_acceleration_m_s2=1.0,
        lift_acceleration_m_s2=0.0,
        maximum_inner_approximation_relative_error=0.30,
        cone_edge_multiplier=1,
        solver_options=LP_OPTIONS,
    )
    with pytest.raises(TypeError, match="task_frame_rotation_object"):
        TaskWrenchEvaluator(**required)
    with pytest.raises(ValueError, match="proper orthonormal"):
        TaskWrenchEvaluator(
            **required,
            task_frame_rotation_object=np.diag((1.0, 1.0, 2.0)),
        )
    with pytest.raises(ValueError, match="proper orthonormal"):
        TaskWrenchEvaluator(
            **required,
            task_frame_rotation_object=np.diag((1.0, 1.0, -1.0)),
        )


def test_uncertainty_claim_scope_is_explicit_and_friction_only() -> None:
    model = _object_model()
    required = dict(
        object_model=model,
        characteristic_radius_m=0.01,
        friction_coefficient_interval=(0.4, 0.6),
        gravity_direction_object=(0.0, 0.0, -1.0),
        task_frame_rotation_object=np.eye(3),
        gravity_acceleration_m_s2=1.0,
        lift_acceleration_m_s2=0.0,
        maximum_inner_approximation_relative_error=0.30,
        cone_edge_multiplier=1,
        solver_options=LP_OPTIONS,
    )
    with pytest.raises(TypeError, match="uncertainty_claim_scope"):
        TaskWrenchEvaluator(**required)
    with pytest.raises(ValueError, match="explicitly limit certification"):
        TaskWrenchEvaluator(
            **required,
            uncertainty_claim_scope="ALL_UNCERTAINTIES_CERTIFIED",
        )

    evaluator = TaskWrenchEvaluator(
        **required,
        uncertainty_claim_scope=(
            FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ),
    )
    result = _evaluate(
        evaluator,
        _AnalyticHand(normal_force_capacity_n=10.0),
        _symmetric_candidate(),
    )
    assert result.diagnostics["certified_uncertainty_scope"] == (
        FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
    )
    assert "object_pose" in result.diagnostics[
        "uncalibrated_uncertainties_excluded_from_certified_set"
    ]


def _real_pad_contract() -> dict:
    return {
        "pads": {
            "finger_1_pad": {
                "finger_name": "f1",
                "link_name": "f1Link3",
                "geometry_source": "URDF_COLLISION",
                "normal_force_capacity_n": 15.0,
            },
            "finger_2_pad": {
                "finger_name": "f2",
                "link_name": "f2Link2",
                "geometry_source": "URDF_COLLISION",
                "normal_force_capacity_n": 15.0,
            },
            "finger_3_pad": {
                "finger_name": "f3",
                "link_name": "f3Link3",
                "geometry_source": "URDF_COLLISION",
                "normal_force_capacity_n": 15.0,
            },
        }
    }


def test_real_hand_jacobian_builds_independent_mimic_torque_map() -> None:
    hand = ThreeFingerHandModel.from_urdf(
        HAND_URDF,
        pad_geometry_contract=_real_pad_contract(),
    )
    joints = (0.30, 0.45, 0.50, 0.55)
    rotation = _rotation()
    object_from_hand = np.eye(4)
    object_from_hand[:3, :3] = rotation
    object_from_hand[:3, 3] = (0.03, -0.02, 0.04)
    pad_transforms = hand.pad_transforms(
        joints,
        base_transform=object_from_hand,
    )
    contacts = tuple(
        PlannedPadContact(
            pad_name=name,
            position_object_m=tuple(pad_transforms[name][:3, 3]),
            path_local_free_side_normal_object=tuple(
                rotation @ np.asarray((1.0, 0.0, 0.0))
            ),
        )
        for name in hand.pads
    )
    candidate = GraspCandidate.from_matrix(
        object_from_hand=object_from_hand,
        independent_joint_positions_rad=joints,
        planned_pad_contacts=contacts,
        internal_normal_forces_n=(1.0, 1.0, 1.0),
    )
    actuation = _evaluator(_object_model()).independent_joint_torque_map(
        candidate, hand
    )
    assert actuation.torque_from_object_contact_forces.shape == (4, 9)
    assert actuation.independent_joint_effort_limits.shape == (4,)
    np.testing.assert_allclose(
        actuation.independent_joint_effort_limits,
        np.full(4, 100.0),
        atol=0.0,
    )
    rotation_hand_from_object = rotation.T
    for index, contact in enumerate(contacts):
        pad = hand.pads[contact.pad_name]
        jacobian = hand.geometric_jacobian(
            pad.link_name,
            joints,
            point_local_m=actuation.contact_points_link_m[index],
        )
        np.testing.assert_allclose(
            actuation.torque_from_object_contact_forces[
                :, 3 * index : 3 * index + 3
            ],
            jacobian[:3].T @ rotation_hand_from_object,
            rtol=0.0,
            atol=64.0 * np.finfo(np.float64).eps,
        )
    # f3j1 mimics f1j1; its contact therefore contributes to the first
    # independent generalized torque column rather than creating a fifth DOF.
    assert np.linalg.norm(
        actuation.torque_from_object_contact_forces[0, 6:9]
    ) > np.finfo(np.float64).eps


def test_common_sobol_scenarios_map_and_reproduce_exactly() -> None:
    evaluator = _evaluator(
        _object_model(),
        friction_interval=(0.20, 0.80),
    )
    scenarios = np.asarray(((0.125,), (0.625,)))
    mapped = evaluator.friction_coefficients_from_unit(scenarios)
    np.testing.assert_allclose(mapped, (0.275, 0.575), atol=0.0)
    hand = _AnalyticHand(10.0)
    candidate = _symmetric_candidate()
    first = _evaluate(evaluator, hand, candidate, scenarios)
    second = _evaluate(evaluator, hand, candidate, scenarios)
    assert first.task_margins == second.task_margins
    assert first.hard_bound_minimum_task_margin == (
        second.hard_bound_minimum_task_margin
    )
    assert first.diagnostics["friction_coefficients"] == pytest.approx(
        (0.275, 0.575)
    )
    assert first.diagnostics["disturbance_vertex_count"] == 12
    assert first.diagnostics["requested_constraint_scaling"] == (
        "ROW_AND_COLUMN_INF_NORM"
    )
    assert first.diagnostics["actual_constraint_scaling"] == (
        "EXPLICIT_AUGMENTED_ROW_AND_COLUMN_INF_NORM_V1_"
        "PLUS_HIGHS_INTERNAL_AUTOMATIC"
    )
    assert first.diagnostics["residual_coordinate_system"] == (
        "EXPLICIT_EQUILIBRATED_SOLVER_COORDINATES"
    )
    assert first.diagnostics["independent_joint_effort_limit_role"] == (
        "URDF_DECLARED_UNCALIBRATED_OPTIMIZATION_CONSTRAINT"
    )


def test_task_wrench_only_evaluation_is_clearance_independent_and_immutable(
) -> None:
    evaluator = _evaluator(
        _object_model(),
        friction_interval=(0.20, 0.80),
    )
    scenarios = np.asarray(((0.125,), (0.625,)), dtype=np.float64)
    candidate = _symmetric_candidate()
    hand = _AnalyticHand(10.0)

    wrench = evaluator.evaluate_task_wrench(
        candidate,
        scenarios,
        hand_model=hand,
    )
    combined = evaluator.evaluate(
        candidate,
        scenarios,
        surface_model=_Surface(),
        hand_model=hand,
    )

    assert isinstance(wrench, TaskWrenchOnlyEvaluation)
    assert not hasattr(wrench, "trajectory_clearance_m")
    assert "trajectory_clearance_scope" not in wrench.diagnostics
    assert wrench.task_margins == combined.task_margins
    assert wrench.hard_bound_minimum_task_margin == (
        combined.hard_bound_minimum_task_margin
    )
    assert wrench.peak_normal_force_n == combined.peak_normal_force_n
    assert wrench.joint_torque_utilization == (
        combined.joint_torque_utilization
    )
    with pytest.raises(TypeError):
        wrench.diagnostics["fabricated_clearance_m"] = 1.0


def test_missing_trajectory_clearance_hook_has_no_silent_default() -> None:
    class SurfaceWithoutClearance:
        parameter_dimension = 1

    evaluator = _evaluator(_object_model())
    with pytest.raises(TaskWrenchEvaluationError, match="not certified"):
        evaluator.evaluate(
            _symmetric_candidate(),
            np.asarray(((0.5,),)),
            surface_model=SurfaceWithoutClearance(),
            hand_model=_AnalyticHand(10.0),
        )

    class IncompleteWitnessOnlySurface:
        parameter_dimension = 1
        trajectory_clearance_scope = "FINITE_PAD_WITNESS_PATH_ONLY"

        @staticmethod
        def trajectory_clearance_m(candidate, hand_model) -> float:
            del candidate, hand_model
            return 0.0

    with pytest.raises(TaskWrenchEvaluationError, match="not certified"):
        evaluator.evaluate(
            _symmetric_candidate(),
            np.asarray(((0.5,),)),
            surface_model=IncompleteWitnessOnlySurface(),
            hand_model=_AnalyticHand(10.0),
        )


def test_solver_failure_raises_without_fabricating_a_margin(monkeypatch) -> None:
    evaluator = _evaluator(_object_model())

    def failed_solver(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            solver_success=False,
            maximum_margin=None,
            solver_status=4,
            solver_message="synthetic numerical failure",
        )

    monkeypatch.setattr(
        evaluator_module,
        "maximum_task_wrench_polytope_margin",
        failed_solver,
    )
    with pytest.raises(TaskWrenchEvaluationError, match="failed closed"):
        _evaluate(evaluator, _AnalyticHand(10.0), _symmetric_candidate())


def test_ranking_loads_are_read_only_from_lexicographic_certificate(
    monkeypatch,
) -> None:
    evaluator = _evaluator(_object_model())

    def successful_stage(name: str, value: float):
        return SimpleNamespace(
            stage_name=name,
            solver_success=True,
            solver_status=0,
            solver_message="synthetic certified stage",
            optimal_value=value,
            maximum_scaled_equilibrium_residual=0.0,
            maximum_scaled_inequality_violation=0.0,
        )

    stage_results = (
        successful_stage("MAXIMIZE_SHARED_TASK_MARGIN", 1.0),
        successful_stage("MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_0", 2.0),
        successful_stage("MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_1", 0.25),
    )

    def lexicographic_solver(*args, **kwargs):
        model = args[0]
        vertex_count = np.asarray(kwargs["disturbance_vertices"]).shape[0]
        # These deliberately non-minimal arrays must not leak into ranking.
        return SimpleNamespace(
            solver_success=True,
            maximum_margin=1.0,
            solver_status=0,
            solver_message="synthetic certified lexicographic solve",
            maximum_scaled_equilibrium_residual=0.0,
            maximum_scaled_inequality_violation=0.0,
            lexicographic_optimal_loads=(2.0, 0.25),
            lexicographic_stage_results=stage_results,
            ray_coefficients_by_vertex=np.full(
                (vertex_count, model.ray_count), 1.0e6
            ),
            contact_forces_by_vertex=np.full(
                (vertex_count, model.contact_count, 3), 1.0e6
            ),
            normal_forces_by_vertex=np.full(
                (vertex_count, model.contact_count), 1.0e6
            ),
            constraint_scaling_implementation=(
                "EXPLICIT_AUGMENTED_ROW_AND_COLUMN_INF_NORM_V1_"
                "PLUS_HIGHS_INTERNAL_AUTOMATIC"
            ),
        )

    monkeypatch.setattr(
        evaluator_module,
        "maximum_task_wrench_polytope_margin",
        lexicographic_solver,
    )
    result = _evaluate(
        evaluator,
        _AnalyticHand(10.0),
        _symmetric_candidate(),
    )
    assert result.peak_normal_force_n == pytest.approx(2.0)
    assert result.joint_torque_utilization == pytest.approx(0.25)
    assert result.peak_normal_force_n != 1.0e6


def test_evaluator_source_contains_no_connector_or_legacy_tokens() -> None:
    source = Path(evaluator_module.__file__).read_text(encoding="utf-8").lower()
    forbidden = (
        ("d" + "38999").lower(),
        ("j" + "35").lower(),
        ("h" + "25").lower(),
        ("cad" + "_").lower(),
    )
    assert all(token not in source for token in forbidden)
