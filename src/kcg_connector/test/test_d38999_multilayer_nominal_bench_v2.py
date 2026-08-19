from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = (
    WORKSPACE_ROOT / "src/kcg_connector/isaac/d38999_multilayer_nominal_bench.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("d38999_a2_nominal_v2_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a2_v2_inputs_and_minimum_jerk_are_frozen() -> None:
    runner = _load_runner()
    frozen = runner._load_frozen_inputs(
        str(WORKSPACE_ROOT / runner.CONTRACT_RELATIVE_PATH),
        str(WORKSPACE_ROOT / runner.MODEL_RELATIVE_PATH),
    )
    assert frozen["input_sha256"] == runner.EXPECTED_SHA256
    assert runner.BENCH_ID == "DYN-A2-NOMINAL-INSERTION-V2"
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "TASK-R12-MULTILAYER-005" not in source
    assert "configure_continuous_plug_guide_runtime_collision" not in source
    assert "formal_p1_pass_claimed" in source

    duration = 1.875 * (
        runner.END_SEPARATION_M - runner.START_SEPARATION_M
    ) / 0.0005
    middle_position, middle_derivative = runner._minimum_jerk(0.5)
    assert math.isclose(duration, 35.8125, abs_tol=1.0e-12)
    assert math.isclose(middle_position, 0.5, abs_tol=1.0e-15)
    assert math.isclose(middle_derivative, 1.875, abs_tol=1.0e-15)


def test_a2_v2_isolates_external_finger_grasp_proxy_before_reset() -> None:
    runner = _load_runner()
    source = RUNNER_PATH.read_text(encoding="utf-8")
    runtime_source = source[source.index("def _runtime(") :]
    helper_source = inspect.getsource(runner._author_grasp_proxy_collision_filter)

    assert runner.HYPOTHESIS_ID == "A2-V2-H13-METAL-STOP-DESCENDANT-PATH-EVALUATOR"
    assert runner.RUN_INDEX_CHOICES == (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    assert runner.EXPECTED_FIXED_COLLIDER_COUNT == 199
    assert runner.EXPECTED_BODY_COLLIDER_COUNT == 72
    assert runner.EXPECTED_NUT_COLLIDER_COUNT == 7
    assert runner.GRIP_PATH == runner.NUT_PATH + "/CouplingNutGraspCollision"
    assert len(runner.BODY_SHOULDER_PATHS) == 2
    assert len(runner.NUT_SHOULDER_PATHS) == 6
    assert all("/NutBearingShoulders/" in path for path in runner.BODY_SHOULDER_PATHS)
    assert all("/NutBearingShoulders/" in path for path in runner.NUT_SHOULDER_PATHS)
    assert runtime_source.index("grasp_proxy_collision_filter = _author_grasp_proxy_collision_filter") < runtime_source.index("world.reset()")
    assert '"grasp_proxy_collision_filter": grasp_proxy_collision_filter' in runtime_source
    assert "CollisionGroup.Define" in helper_source
    assert 'register("FixedReceptacle", fixed_paths)' in helper_source
    assert '"BodyNonShoulder", body_non_shoulder_paths' in helper_source
    assert '"BodyShoulderPositive"' in helper_source
    assert '"BodyShoulderNegative"' in helper_source
    assert 'register("AuthorizedCouplingNutGrip", grip_paths)' in helper_source
    assert '"NutShoulderPositive"' in helper_source
    assert '"NutShoulderNegative"' in helper_source
    assert "body_negative_group_path" in helper_source
    assert "body_positive_group_path" in helper_source
    assert '"default_unlisted_connector_pairs_filtered": True' in helper_source
    assert "sdf.Path(target)" in helper_source
    assert '"AuthorizedCouplingNutGrip": sorted(' in helper_source
    assert '"NutShoulderPositive": sorted(' in helper_source
    assert '"NutShoulderNegative": sorted(' in helper_source
    assert "get_full_contact_report" not in helper_source
    assert '"controller_input": False' in helper_source
    assert '"connector_receptacle_collisions_removed": False' in helper_source
    assert '"body_vs_fixed_receptacle_collisions_preserved": True' in helper_source
    assert '"external_finger_contact_preserved": True' in helper_source
    assert '"physical_shoulder_collisions_preserved": True' in helper_source
    assert runner.PHYSICAL_EFFECT_IMPLEMENTATIONS["nut_body_physical_shoulder"] == "physx_continuous_real_collision"


def test_a2_v2_shared_spring_and_thread_discretizations_restore() -> None:
    runner = _load_runner()
    dt = 1.0 / 240.0
    force, spring = runner._backward_euler_shared_spring_force(
        [np.asarray((20.0e-6, 0.0)) for _ in range(12)],
        [np.zeros(2) for _ in range(12)],
        active_fraction=1.0,
        per_channel_stiffness_n_m=12000.0,
        per_channel_damping_n_s_m=1.0,
        integration_dt_s=dt,
        effective_mass_kg=0.31,
    )
    assert force[0] < 0.0
    assert spring["implicit_denominator"] > 1.0
    assert spring["continuous_parameter_values_unchanged"] is True

    thread_force, thread = runner._backward_euler_thread_force(
        10.0e-6,
        0.0,
        stiffness_n_m=10000.0,
        damping_n_s_m=20.0,
        integration_dt_s=dt,
        body_mass_kg=0.23,
        nut_yaw_inertia_kg_m2=0.0000405124,
        lead_m_per_revolution=0.00762,
    )
    assert 0.0 < thread_force < 0.1
    assert thread["implicit_denominator"] > 1.0


def test_a2_v2_planar_same_label_wrench_is_finite_and_restoring() -> None:
    runner = _load_runner()
    angles = np.linspace(0.0, 2.0 * np.pi, 61, endpoint=False)
    arms = [0.01 * np.asarray((np.cos(angle), np.sin(angle))) for angle in angles]
    force, torque, audit = runner._backward_euler_planar_channel_wrench(
        [np.asarray((2.0e-6, 0.0)) for _ in arms],
        [np.zeros(2) for _ in arms],
        arms,
        active_fraction=1.0,
        per_channel_stiffness_n_m=24000.0,
        per_channel_damping_n_s_m=0.6,
        integration_dt_s=1.0 / 240.0,
        effective_mass_kg=0.31,
        effective_yaw_inertia_kg_m2=0.0000566766,
    )
    assert np.all(np.isfinite(force)) and math.isfinite(torque)
    assert force[0] < 0.0
    assert audit["implicit_matrix_condition_number"] < 2.0
    assert audit["continuous_parameter_values_unchanged"] is True


def test_a2_v2_axis_driver_is_component_limited_and_anti_windup_bounded() -> None:
    runner = _load_runner()
    output, integral, requested, saturated = runner._axis_position_driver(
        target_position=-0.01505,
        target_velocity=0.0,
        actual_position=-0.01300,
        actual_velocity=0.0,
        integral_n=-7.4,
        dt=1.0 / 240.0,
        position_gain_n_m=600.0,
        velocity_gain_n_s_m=8.0,
        integral_gain_n_m_s=1000.0,
        integral_limit_n=7.5,
        force_limit_n=8.0,
    )
    assert requested < -8.0
    assert output == -8.0
    assert saturated is True
    assert -7.5 <= integral <= 7.5


def test_a2_v2_target_hold_integral_tracking_uses_existing_headroom_only() -> None:
    runner = _load_runner()
    dt = 1.0 / 240.0
    feedforward = -7.865375
    integral = 0.12537894972447283
    previous = integral
    outputs = []
    for _ in range(120):
        output, integral, requested, saturated = runner._axis_position_driver(
            target_position=-0.01505,
            target_velocity=0.0,
            actual_position=-0.015037061646580696,
            actual_velocity=0.0,
            integral_n=integral,
            dt=dt,
            position_gain_n_m=600.0,
            velocity_gain_n_s_m=40.0,
            integral_gain_n_m_s=1000.0,
            integral_limit_n=7.5,
            force_limit_n=8.0,
            feedforward_n=feedforward,
            completion_force_direction=-1.0,
            integral_tracking_rate_n_s=(
                runner.AXIAL_TARGET_HOLD_INTEGRAL_TRACKING_RATE_N_S
            ),
        )
        assert abs(integral - previous) <= dt + 1.0e-15
        assert -8.0 <= output <= 8.0
        previous = integral
        outputs.append(output)
    assert math.isclose(outputs[-1], -8.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert requested >= -8.0 - 1.0e-12
    assert saturated is False
    assert integral < 0.0

    original_integral = 0.1
    _, reversed_integral, _, _ = runner._axis_position_driver(
        target_position=-0.01505,
        target_velocity=0.0,
        actual_position=-0.01506,
        actual_velocity=0.0,
        integral_n=original_integral,
        dt=dt,
        position_gain_n_m=600.0,
        velocity_gain_n_s_m=40.0,
        integral_gain_n_m_s=1000.0,
        integral_limit_n=7.5,
        force_limit_n=8.0,
        feedforward_n=feedforward,
        completion_force_direction=-1.0,
        integral_tracking_rate_n_s=(
            runner.AXIAL_TARGET_HOLD_INTEGRAL_TRACKING_RATE_N_S
        ),
    )
    assert reversed_integral > original_integral

    helper_signature = inspect.signature(runner._axis_position_driver)
    assert "completion_force_direction" in helper_signature.parameters
    assert "integral_tracking_rate_n_s" in helper_signature.parameters
    helper_source = inspect.getsource(runner._axis_position_driver)
    assert "_contact_rows" not in helper_source
    assert "event_first" not in helper_source
    assert "contact_normal" not in helper_source


def test_a2_v2_terminal_approach_tracking_is_predeclared_and_truth_free() -> None:
    runner = _load_runner()
    helper = runner._predeclared_terminal_approach_integral_tracking
    window = runner.AXIAL_TERMINAL_APPROACH_TRACKING_WINDOW_M
    speed = runner.AXIAL_TERMINAL_APPROACH_TRACKING_MAX_TARGET_SPEED_M_S

    outside = helper(runner.END_SEPARATION_M - window - 1.0e-9, 0.0)
    too_fast = helper(runner.END_SEPARATION_M - 0.5 * window, speed + 1.0e-9)
    approach = helper(runner.END_SEPARATION_M - 0.5 * window, 0.5 * speed)
    hold = helper(runner.END_SEPARATION_M, 0.0)

    assert outside["enabled"] is False
    assert too_fast["enabled"] is False
    assert approach["enabled"] is True
    assert approach["phase"] == "terminal_approach"
    assert hold["enabled"] is True
    assert hold["phase"] == "target_hold"
    assert approach["terminal_window_m"] == 1.0e-6
    assert approach["maximum_target_speed_m_s"] == 1.0e-6
    assert approach["target_schedule_only"] is True
    assert approach["contact_or_event_truth_input"] is False

    helper_signature = inspect.signature(helper)
    assert set(helper_signature.parameters) == {
        "target_separation_m",
        "target_separation_rate_m_s",
    }
    helper_source = inspect.getsource(helper)
    assert "actual_position" not in helper_source
    assert "_contact_rows" not in helper_source
    assert "event_first" not in helper_source
    assert "contact_normal" not in helper_source
    assert "contact_object" not in helper_source


def test_a2_v2_metal_stop_evaluator_accepts_generated_descendants_only() -> None:
    runner = _load_runner()
    fixed_segment = runner.FIXED_STOP_PATH + "/StopSegment_020"
    plug_child = runner.PLUG_STOP_PATH + "/GeneratedFace"

    assert runner._is_metal_stop_pair(
        (runner.FIXED_STOP_PATH, runner.PLUG_STOP_PATH)
    )
    assert runner._is_metal_stop_pair((fixed_segment, runner.PLUG_STOP_PATH))
    assert runner._is_metal_stop_pair((plug_child, fixed_segment))
    assert not runner._is_metal_stop_pair(
        (runner.FIXED_STOP_PATH + "Fake/StopSegment_020", runner.PLUG_STOP_PATH)
    )
    assert not runner._is_metal_stop_pair(
        (runner.FIXED_STOP_PATH, runner.FIXED_STOP_PATH + "/StopSegment_001")
    )
    assert not runner._is_metal_stop_pair(
        (runner.BODY_PATH + "/Other", fixed_segment)
    )
    assert not runner._is_metal_stop_pair((runner.FIXED_STOP_PATH,))

    helper_source = inspect.getsource(runner._is_metal_stop_pair)
    contact_source = inspect.getsource(runner._contact_rows)
    assert "_path_is_at_or_below" in helper_source
    assert "_is_metal_stop_pair(collider_paths)" in contact_source
    assert "set(collider_paths)" not in contact_source


def test_a2_v2_nut_yaw_integral_is_bounded_truth_free_and_anti_windup_safe() -> None:
    runner = _load_runner()
    dt = 1.0 / 240.0
    integral = 0.0
    maximum_output = 0.0
    for _ in range(6016):
        output, integral, requested, saturated, error, velocity_error = (
            runner._bounded_angular_position_driver(
                target_position_rad=-6.283185307179585,
                target_velocity_rad_s=0.0,
                actual_position_rad=-6.20974201423038,
                actual_velocity_rad_s=0.0,
                integral_nm=integral,
                dt=dt,
                position_gain_nm_rad=0.8,
                velocity_gain_nm_s_rad=0.01,
                integral_gain_nm_rad_s=runner.NUT_YAW_INTEGRAL_GAIN_NM_RAD_S,
                integral_limit_nm=runner.NUT_YAW_INTEGRAL_LIMIT_NM,
                torque_limit_nm=0.30,
            )
        )
        maximum_output = max(maximum_output, abs(output))
    assert error < 0.0
    assert velocity_error == 0.0
    assert math.isclose(integral, -0.10, rel_tol=0.0, abs_tol=1.0e-12)
    assert requested > -0.30
    assert saturated is False
    assert maximum_output < 0.30

    output, held_integral, requested, saturated, _, _ = (
        runner._bounded_angular_position_driver(
            target_position_rad=-1.0,
            target_velocity_rad_s=0.0,
            actual_position_rad=0.0,
            actual_velocity_rad_s=0.0,
            integral_nm=-0.10,
            dt=dt,
            position_gain_nm_rad=0.8,
            velocity_gain_nm_s_rad=0.01,
            integral_gain_nm_rad_s=runner.NUT_YAW_INTEGRAL_GAIN_NM_RAD_S,
            integral_limit_nm=runner.NUT_YAW_INTEGRAL_LIMIT_NM,
            torque_limit_nm=0.30,
        )
    )
    assert output == -0.30
    assert requested < -0.30
    assert saturated is True
    assert held_integral == -0.10

    helper_signature = inspect.signature(runner._bounded_angular_position_driver)
    assert set(helper_signature.parameters) == {
        "target_position_rad",
        "target_velocity_rad_s",
        "actual_position_rad",
        "actual_velocity_rad_s",
        "integral_nm",
        "dt",
        "position_gain_nm_rad",
        "velocity_gain_nm_s_rad",
        "integral_gain_nm_rad_s",
        "integral_limit_nm",
        "torque_limit_nm",
    }
    helper_source = inspect.getsource(runner._bounded_angular_position_driver)
    assert "_contact_rows" not in helper_source
    assert "event_first" not in helper_source
    assert "contact_normal" not in helper_source
    assert "contact_object" not in helper_source


def test_a2_v2_thread_thrust_phase_lead_is_single_smooth_truth_free_schedule() -> None:
    runner = _load_runner()
    frozen = runner._load_frozen_inputs(
        str(WORKSPACE_ROOT / runner.CONTRACT_RELATIVE_PATH),
        str(WORKSPACE_ROOT / runner.MODEL_RELATIVE_PATH),
    )
    entry = float(
        frozen["contract"]["thread"]["nominal_entry_separation_m"]["value"]
    )
    midpoint = 0.5 * (entry + runner.END_SEPARATION_M)
    before = runner._predeclared_thread_thrust_phase_lead(
        frozen, entry - 1.0e-6, 0.0005
    )
    at_entry = runner._predeclared_thread_thrust_phase_lead(
        frozen, entry, 0.0005
    )
    at_midpoint = runner._predeclared_thread_thrust_phase_lead(
        frozen, midpoint, 0.0005
    )
    at_bottom = runner._predeclared_thread_thrust_phase_lead(
        frozen, runner.END_SEPARATION_M, 0.0
    )
    assert before["yaw_offset_rad"] == 0.0
    assert before["omega_offset_rad_s"] == 0.0
    assert at_entry["yaw_offset_rad"] == 0.0
    assert at_entry["omega_offset_rad_s"] == 0.0
    assert math.isclose(at_midpoint["smooth_scale"], 0.5, abs_tol=1.0e-15)
    assert math.isclose(
        at_midpoint["yaw_offset_rad"],
        -0.5 * runner.THREAD_THRUST_TERMINAL_YAW_LEAD_RAD,
        abs_tol=1.0e-15,
    )
    assert at_midpoint["omega_offset_rad_s"] < 0.0
    assert math.isclose(
        at_bottom["yaw_offset_rad"],
        -runner.THREAD_THRUST_TERMINAL_YAW_LEAD_RAD,
        abs_tol=1.0e-15,
    )
    assert at_bottom["omega_offset_rad_s"] == 0.0
    assert at_bottom["contact_or_event_truth_input"] is False

    thread_force, audit = runner._backward_euler_thread_force(
        -0.00012363031008791908,
        0.0,
        stiffness_n_m=10000.0,
        damping_n_s_m=20.0,
        integration_dt_s=1.0 / 240.0,
        body_mass_kg=0.23,
        nut_yaw_inertia_kg_m2=0.0000405124,
        lead_m_per_revolution=0.00762,
    )
    assert math.isclose(
        thread_force,
        -runner.THREAD_THRUST_TARGET_N,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert audit["continuous_parameter_values_unchanged"] is True

    helper_signature = inspect.signature(
        runner._predeclared_thread_thrust_phase_lead
    )
    assert set(helper_signature.parameters) == {
        "frozen",
        "target_separation_m",
        "target_separation_rate_m_s",
    }
    helper_source = inspect.getsource(runner._predeclared_thread_thrust_phase_lead)
    assert "actual_position" not in helper_source
    assert "_contact_rows" not in helper_source
    assert "event_first" not in helper_source
    assert "contact_normal" not in helper_source


def test_a2_v2_nominal_axial_load_feedforward_is_bounded_and_truth_free() -> None:
    runner = _load_runner()
    frozen = runner._load_frozen_inputs(
        str(WORKSPACE_ROOT / runner.CONTRACT_RELATIVE_PATH),
        str(WORKSPACE_ROOT / runner.MODEL_RELATIVE_PATH),
    )
    start = runner._nominal_axial_load_feedforward(
        frozen, runner.START_SEPARATION_M
    )
    bottom = runner._nominal_axial_load_feedforward(
        frozen, runner.END_SEPARATION_M
    )
    assert start["total_n"] == 0.0
    assert math.isclose(
        bottom["spring_finger_axial_n"], 2.88, rel_tol=0.0, abs_tol=1.0e-12
    )
    assert math.isclose(
        bottom["pin_isolation_axial_n"], 4.49875, rel_tol=0.0, abs_tol=1.0e-12
    )
    assert math.isclose(
        bottom["peripheral_seal_axial_n"], 8.352, rel_tol=0.0, abs_tol=1.0e-12
    )
    assert math.isclose(
        bottom["total_n"], 15.73075, rel_tol=0.0, abs_tol=1.0e-12
    )
    assert math.isclose(
        bottom["per_driven_body_component_n"],
        7.865375,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert bottom["per_driven_body_component_n"] < 8.0

    helper_signature = inspect.signature(runner._nominal_axial_load_feedforward)
    assert set(helper_signature.parameters) == {"frozen", "target_separation_m"}
    helper_source = inspect.getsource(runner._nominal_axial_load_feedforward)
    assert "actual_position" not in helper_source
    assert "_contact_rows" not in helper_source
    assert "event_first" not in helper_source
    assert "metal_contact_force" not in helper_source
    assert "internal[" not in helper_source
    assert "metal_stop" not in helper_source

    dt = 1.0 / 240.0
    duration = 35.8125
    steps = 8595
    previous = 0.0
    maximum_slew = 0.0
    for index in range(steps):
        position, _ = runner._minimum_jerk((index + 1) * dt / duration)
        target = runner.START_SEPARATION_M + (
            runner.END_SEPARATION_M - runner.START_SEPARATION_M
        ) * position
        current = runner._nominal_axial_load_feedforward(
            frozen, target
        )["per_driven_body_component_n"]
        maximum_slew = max(maximum_slew, abs(current - previous) / dt)
        previous = current
    assert math.isclose(
        maximum_slew, 3.574892151944953, rel_tol=0.0, abs_tol=1.0e-12
    )


def test_a2_v2_shoulder_endplay_maps_one_coordinate_to_two_targets() -> None:
    runner = _load_runner()
    frozen = runner._load_frozen_inputs(
        str(WORKSPACE_ROOT / runner.CONTRACT_RELATIVE_PATH),
        str(WORKSPACE_ROOT / runner.MODEL_RELATIVE_PATH),
    )
    target_separation = runner.END_SEPARATION_M
    target_rate = 0.0004
    targets = runner._shoulder_aware_axial_targets(
        frozen, target_separation, target_rate
    )
    assert math.isclose(
        targets["insertion_shoulder_endplay_m"],
        -50.0e-6,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        targets["body_target_z_m"], -target_separation, abs_tol=1.0e-15
    )
    assert math.isclose(
        targets["nut_target_z_m"],
        -target_separation - 50.0e-6,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert targets["body_target_velocity_z_m_s"] == -target_rate
    assert targets["nut_target_velocity_z_m_s"] == -target_rate
    assert targets["contact_or_event_truth_input"] is False

    helper_signature = inspect.signature(runner._shoulder_aware_axial_targets)
    assert set(helper_signature.parameters) == {
        "frozen",
        "target_separation_m",
        "target_separation_rate_m_s",
    }
    helper_source = inspect.getsource(runner._shoulder_aware_axial_targets)
    assert "physical_shoulder_endplay_m" in helper_source
    assert "physical_shoulder_contact_endplay_m" in helper_source
    assert "_contact_rows" not in helper_source
    assert "event_first" not in helper_source
    assert "actual_position" not in helper_source

    zero_rpy = np.zeros(3, dtype=np.float64)
    body_position = np.asarray((0.0, 0.0, targets["body_target_z_m"]))
    nut_position = np.asarray((0.0, 0.0, targets["nut_target_z_m"]))
    body_velocity = np.asarray((0.0, 0.0, -target_rate, 0.0, 0.0, 0.0))
    nut_velocity = body_velocity.copy()
    state = {
        "body_force_integral_n": 0.0,
        "nut_force_integral_n": 0.0,
        "nut_yaw_integral_nm": 0.0,
    }
    command = runner._driver_commands(
        frozen,
        dt=1.0 / 240.0,
        target_separation_m=target_separation,
        target_separation_rate_m_s=target_rate,
        body_position=body_position,
        body_rpy=zero_rpy,
        body_velocity=body_velocity,
        nut_position=nut_position,
        nut_rpy=zero_rpy,
        nut_unwrapped_yaw=0.0,
        nut_velocity=nut_velocity,
        state=state,
    )
    assert math.isclose(
        command["body_axial_force_requested_n"],
        command["nut_axial_force_requested_n"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert command["body_axial_force_saturated"] is False
    assert command["nut_axial_force_saturated"] is False


def test_a2_v2_joint_interval_lateral_actions_are_partitioned_and_stable() -> None:
    runner = _load_runner()
    frozen = runner._load_frozen_inputs(
        str(WORKSPACE_ROOT / runner.CONTRACT_RELATIVE_PATH),
        str(WORKSPACE_ROOT / runner.MODEL_RELATIVE_PATH),
    )
    arms = [np.asarray(row["center_m"], dtype=np.float64) for row in frozen["pairs"]]
    dt = 1.0 / 240.0
    mass = 0.31
    inertia = 0.0000566766

    def one_step(state: np.ndarray) -> np.ndarray:
        q = state[:3]
        velocity = state[3:]
        pair_errors = []
        pair_velocities = []
        for arm in arms:
            jacobian = np.asarray(
                ((1.0, 0.0, -arm[1]), (0.0, 1.0, arm[0])),
                dtype=np.float64,
            )
            pair_errors.append(jacobian @ q)
            pair_velocities.append(jacobian @ velocity)
        result = runner._joint_interval_lateral_actions(
            [q[:2].copy() for _ in range(12)],
            [velocity[:2].copy() for _ in range(12)],
            pair_errors,
            pair_velocities,
            arms,
            spring_active_fraction=1.0,
            spring_per_channel_stiffness_n_m=12000.0,
            spring_per_channel_damping_n_s_m=1.0,
            pair_active_fraction=1.0,
            pair_per_channel_stiffness_n_m=24000.0,
            pair_per_channel_damping_n_s_m=0.6,
            body_driver_error_m=q[:2],
            body_driver_velocity_m_s=velocity[:2],
            nut_driver_error_m=q[:2],
            nut_driver_velocity_m_s=velocity[:2],
            translation_driver_stiffness_n_m=600.0,
            translation_driver_damping_n_s_m=8.0,
            body_yaw_error_rad=float(q[2]),
            body_yaw_velocity_rad_s=float(velocity[2]),
            body_yaw_driver_stiffness_nm_rad=0.8,
            body_yaw_driver_damping_nm_s_rad=0.01,
            integration_dt_s=dt,
            effective_translation_mass_kg=mass,
            body_yaw_inertia_kg_m2=inertia,
        )
        acceleration = result["predicted_generalized_acceleration"]
        velocity_end = velocity + dt * acceleration
        q_end = q + dt * velocity_end
        assert result["partition_residual_norm"] < 1.0e-9
        assert result["continuous_parameter_values_unchanged"] is True
        return np.concatenate((q_end, velocity_end))

    perturbation = 1.0e-6
    basis = perturbation * np.eye(6, dtype=np.float64)
    state_matrix = np.column_stack(
        [one_step(column) / perturbation for column in basis]
    )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(state_matrix))))
    assert math.isclose(
        spectral_radius, 0.13660925999069462, rel_tol=0.0, abs_tol=1.0e-12
    )

    source = RUNNER_PATH.read_text(encoding="utf-8")
    internal_source = source[source.index("def _internal_effects(") : source.index("def _axis_position_driver(")]
    runtime_source = source[source.index("def _runtime(") :]
    assert "_joint_interval_lateral_actions(" in internal_source
    assert "_backward_euler_shared_spring_force(" not in internal_source
    assert "_backward_euler_planar_channel_wrench(" not in internal_source
    assert 'coupled_driver = internal["coupled_driver_overrides"]' in runtime_source
