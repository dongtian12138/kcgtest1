"""Static contracts for the independent Isaac visual-XY pick entrypoint."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac/d38999_visual_xy_pick_smoke.py"
)
E2E_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac/d38999_tabletop_pick_smoke.py"
)


def _source():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _module():
    return runpy.run_path(str(SCRIPT_PATH), run_name="visual_pick_test")


def _healthy_capture_side_effects():
    return {
        "world_reset_or_clear_calls": 0,
        "object_pose_writes_after_start": 0,
        "resource_cleanup": {
            "annotator_detach_count": 3,
            "camera_destroyed": True,
            "errors": [],
            "render_product_destroyed": True,
            "resources_released": True,
            "scene_cleared": False,
            "stage_prims_removed": 0,
            "world_reset": False,
        },
        "timeline_state": {
            "playing_after_cleanup": False,
            "playing_after_restore": True,
            "playing_before_capture": True,
            "restore_attempted": True,
            "restored": True,
        },
    }


def test_entrypoint_import_is_lazy_and_one_independent_app():
    before = set(sys.modules)
    module = _module()
    imported = set(sys.modules) - before
    assert not any(
        name == "isaacsim" or name.startswith(("isaacsim.", "omni.", "pxr"))
        for name in imported
    )
    assert module["RESULT_MARKER"] == (
        "ISAAC D38999 VISUAL XY PICK PROBE V1"
    )
    assert module["PREINSERT_RESULT_MARKER"] == (
        "ISAAC D38999 VISUAL XY PREINSERT PROBE V1"
    )
    assert module["TACTILE_LIP_RESULT_MARKER"] == (
        "ISAAC D38999 TACTILE LIP CALIBRATION V1"
    )
    assert module["TACTILE_LIP_MANIFOLD_RESULT_MARKER"] == (
        "ISAAC D38999 TACTILE LIP MANIFOLD CAPTURE V1"
    )
    assert _source().count("SimulationApp(") == 1


def test_fk_arm_adapter_flattens_numpy_vector_and_rejects_wrong_shape():
    adapt = _module()["seven_float_arm_tuple"]
    adapt_position = _module()["three_float_position_tuple"]
    vector = np.arange(7, dtype=np.float64)
    assert adapt(vector) == tuple(float(value) for value in range(7))
    with pytest.raises(ValueError, match="flat numeric"):
        adapt(vector.reshape(1, 7))
    with pytest.raises(ValueError, match="exactly 7"):
        adapt(vector[:6])
    position = np.asarray((0.55, 0.185, 0.32), dtype=np.float64)
    assert adapt_position(position) == pytest.approx((0.55, 0.185, 0.32))
    with pytest.raises(ValueError, match="flat numeric"):
        adapt_position(position.reshape(1, 3))
    with pytest.raises(ValueError, match="exactly 3"):
        adapt_position(position[:2])

    source = _source()
    assert 'seven_float_arm_tuple(seed, "tactile_ik_seed")' in source
    assert "three_float_position_tuple(" in source
    assert 'tcp_target, "tactile_ik_tcp_target"' in source


def test_command_continuous_retract_path_is_exact_monotonic_and_bounded():
    build = _module()["build_command_continuous_retract_path"]
    from kcg_connector.d38999_physical_insertion import (
        solve_fixed_q7_tcp_pose,
    )
    from kcg_connector.d38999_tabletop_pick import (
        iiwa14_grasp_tcp_transform,
    )

    # This is the exact last applied command reconstructed at third-GPU lip
    # contact.  Keeping it as the first action is the regression under test;
    # the measured q from that artifact must never replace it.
    start = (
        0.4422378774963382,
        0.49963438725048576,
        -0.23922471555847377,
        -0.7814709867929961,
        0.11910571496499482,
        1.8705293266986878,
        0.650482794,
    )

    def solve_target(seed, target_position, target_rotation):
        return solve_fixed_q7_tcp_pose(
            seed,
            target_position,
            target_rotation=np.asarray(target_rotation),
            maximum_iterations=50,
            damping=1.0e-6,
        )

    result = build(
        start,
        (0.0, 0.0, 1.0),
        0.0003,
        240.0,
        0.0005,
        iiwa14_grasp_tcp_transform,
        solve_target,
        maximum_fk_position_error_m=1.0e-7,
        maximum_fk_orientation_error_rad=1.0e-7,
    )
    commands = np.asarray(result["commands"])
    axial = np.asarray(result["command_fk_axial_progress_m"])
    assert result["first_command_exact"] is True
    assert tuple(commands[0]) == start
    assert np.all(commands[:, 6] == start[6])
    assert np.all(np.diff(axial) >= -1.0e-12)
    assert axial[0] == pytest.approx(0.0, abs=1.0e-15)
    assert axial[-1] == pytest.approx(0.0003, abs=1.0e-7)
    assert result["peak_command_fk_axial_speed_m_s"] <= 0.0005
    assert max(result["command_fk_lateral_error_m"]) <= 1.0e-7
    assert max(result["command_fk_orientation_error_rad"]) <= 1.0e-7

    with pytest.raises(ValueError, match="exactly 7"):
        build(
            start[:-1],
            (0.0, 0.0, 1.0),
            0.0003,
            240.0,
            0.0005,
            iiwa14_grasp_tcp_transform,
            solve_target,
            maximum_fk_position_error_m=1.0e-7,
            maximum_fk_orientation_error_rad=1.0e-7,
        )


def test_preflight_minimum_jerk_has_command_speed_headroom():
    module = _module()
    steps_for_speed = module["minimum_jerk_steps_for_peak_speed"]
    rate_hz = 240.0
    command_speed_m_s = 0.00035
    expected = {
        0.00045: 579,
        0.00030: 386,
    }
    for distance_m, expected_steps in expected.items():
        steps = steps_for_speed(
            distance_m, rate_hz, command_speed_m_s
        )
        assert steps == expected_steps
        continuous_peak = 1.875 * distance_m * rate_hz / steps
        assert continuous_peak <= command_speed_m_s
        assert (
            1.875 * distance_m * rate_hz / (steps - 1)
            > command_speed_m_s
        )
    with pytest.raises(ValueError, match="positive"):
        steps_for_speed(0.00045, rate_hz, 0.0)
    with pytest.raises(ValueError, match="positive"):
        steps_for_speed(float("nan"), rate_hz, command_speed_m_s)


def test_mid_slope_interrupt_plan_is_unique_discrete_and_has_knife_edge():
    module = _module()
    build = module["build_discrete_mid_slope_interrupt_plan"]
    steps_for_speed = module["minimum_jerk_steps_for_peak_speed"]
    steps_with_headroom = module[
        "minimum_jerk_steps_with_strict_headroom"
    ]
    rate_hz = 240.0
    distance_m = 0.0025
    ceiling_m_s = 0.00005

    def linear_fk(command):
        transform = np.eye(4, dtype=np.float64)
        transform[2, 3] = -float(command[0])
        return transform

    required_steps = steps_for_speed(distance_m, rate_hz, ceiling_m_s)
    assert required_steps == 22500
    planner = steps_with_headroom(distance_m, rate_hz, ceiling_m_s)
    assert planner == {
        "semantics": (
            "continuous_minimum_plus_one_discrete_headroom_tick"
        ),
        "base_required_command_count": 22500,
        "added_headroom_command_count": 1,
        "final_command_count": 22501,
        "nominal_peak_speed_ceiling_m_s": ceiling_m_s,
    }
    plan = build(
        (0.0,) * 7,
        (distance_m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        planner["final_command_count"],
        (0.0, 0.0, 1.0),
        rate_hz,
        linear_fk,
        strict_headroom_plan=planner,
    )
    assert plan["unique_argmax_count"] == 1
    assert plan["unique_argmax_motion_index"] == 11250
    assert plan["interior_argmax_gate"] is True
    assert plan["maximum_downward_speed_m_s"] <= ceiling_m_s
    assert [
        item["motion_index"]
        for item in plan["planned_argmax_neighborhood"]
    ] == [11249, 11250, 11251]
    assert build(
        (0.0,) * 7,
        (distance_m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        planner["final_command_count"],
        (0.0, 0.0, 1.0),
        rate_hz,
        linear_fk,
        strict_headroom_plan=planner,
    )["full_step_down_sha256"] == plan["full_step_down_sha256"]
    with pytest.raises(ValueError, match="lacks strict headroom"):
        build(
            (0.0,) * 7,
            (distance_m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            required_steps,
            (0.0, 0.0, 1.0),
            rate_hz,
            linear_fk,
            strict_headroom_plan=planner,
        )
    assert plan["maximum_downward_speed_m_s"] < ceiling_m_s

    def constant_fk(_command):
        return np.eye(4, dtype=np.float64)

    with pytest.raises(ValueError, match="argmax is not unique"):
        build(
            (0.0,) * 7,
            (distance_m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            planner["final_command_count"],
            (0.0, 0.0, 1.0),
            rate_hz,
            constant_fk,
            strict_headroom_plan=planner,
        )


def _matched_reversal_samples():
    samples = []
    command = [0.1 * index for index in range(7)]
    for index in range(6):
        measured_tcp = [0.01 + 1.0e-6 * index, 0.02, 0.03]
        samples.append(
            {
                "global_step": 100 + index,
                "motion_index": 50 + index,
                "command_arm_rad": command,
                "measured_arm_rad": [
                    value + 1.0e-5 * (index + 1) for value in command
                ],
                "measured_arm_velocity_rad_s": [
                    -0.01 + 0.001 * index,
                    0.002,
                    -0.003,
                    0.004,
                    -0.005,
                    0.006,
                    -0.007,
                ],
                "command_fk_tcp_world_m": [0.011, 0.021, 0.031],
                "measured_arm_fk_tcp_world_m": [0.0105, 0.0205, 0.0305],
                "measured_tcp_prim_world_m": measured_tcp,
                "pre_measured_tcp_prim_world_m": [
                    measured_tcp[0] - 1.0e-6,
                    measured_tcp[1],
                    measured_tcp[2],
                ],
            }
        )
    return samples


def test_reversal_state_uses_rotation_transpose_and_signed_ranges():
    module = _module()
    features = module["tactile_reversal_state_features"]
    compare = module["compare_tactile_reversal_state_equivalence"]
    samples = _matched_reversal_samples()
    # Task axes are columns in world: task X=world Y, task Y=-world X.
    rotation = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    reference = features(samples, rotation, 240.0)
    assert reference["sample_count"] == 6
    assert reference["samples"][0][
        "measured_tcp_velocity_task_m_s"
    ] == pytest.approx((0.0, -0.00024, 0.0), abs=1.0e-15)
    assert compare(samples, reference, rotation, 240.0)["passed"] is True

    opposite_velocity = copy.deepcopy(samples)
    opposite_velocity[0]["measured_arm_velocity_rad_s"][0] = 0.01
    rejected = compare(opposite_velocity, reference, rotation, 240.0)
    assert rejected["passed"] is False
    assert rejected["comparisons"][
        "signed_range:measured_arm_velocity_rad_s"
    ]["gate"] is False

    faster_downward = copy.deepcopy(samples)
    faster_downward[-1]["measured_tcp_prim_world_m"][2] -= 1.0e-6
    rejected = compare(faster_downward, reference, rotation, 240.0)
    assert rejected["passed"] is False
    assert rejected["comparisons"][
        "task_z_velocity_one_sided_lower_bound"
    ]["gate"] is False
    with pytest.raises(ValueError, match="exactly 6"):
        features(samples[:-1], rotation, 240.0)


def test_failed_stage_a_artifact_locks_fixed_bound_and_exact_six_freeze():
    module = _module()
    report_path = (
        SCRIPT_PATH.parents[3]
        / "artifacts/kcg_connector/"
        "d38999_tactile_lip_manifold_capture_v1/"
        "gpu_20260812T204048Z/report.json"
    )
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        "b726dfab339018409d51de8dc5f820410602a56dcbb6c19c75a27cd6b4243567"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    section = report["tactile_lip_manifold_capture"]
    assert report["passed"] is False
    assert section["failure_global_step"] == 15913
    assert section["abort_retract"] is None
    ring = section["failure_transition_ring"]
    physical_frames = [
        item
        for item in ring
        if item["phase"].endswith("plus_x_guarded_approach")
        and item["post_step"]["loose_fixed_contact_records"] == 3
    ]
    assert [
        item["post_step"]["global_step"] for item in physical_frames
    ] == list(range(15906, 15912))
    frozen_command = physical_frames[0]["planned_command_arm_rad"]
    assert all(
        item["planned_command_arm_rad"] == frozen_command
        for item in physical_frames
    )
    terminal = ring[-2:]
    start_z = terminal[0]["pre_step"]["measured_tcp_prim_world_m"][2]
    progress = [
        item["post_step"]["measured_tcp_prim_world_m"][2] - start_z
        for item in terminal
    ]
    fixed = module["TACTILE_FIXED_NEGATIVE_RETRACT_BOUND_M"]
    assert fixed == 1.1928619392254092e-7
    assert progress[0] >= -fixed
    assert progress[1] < -fixed
    assert module["TACTILE_LIP_MANIFOLD_APPROACH_SPEED_M_S"] == 0.00005


def test_release_endpoint_hold_indices_are_bounded_and_never_overrun():
    build = _module()["bounded_endpoint_hold_path_indices"]
    indices = build(4, 6)
    assert indices == (0, 1, 2, 3, 3, 3, 3, 3, 3, 3)
    assert len(indices) == 10
    # Whether measured +0.3 mm is never reached, or reached at the last path
    # tick without a release window, the scheduler can execute only six extra
    # guarded holds and always indexes the final command.
    never_reached_executed = tuple(indices)
    last_tick_reached_executed = tuple(indices)
    assert max(never_reached_executed) == 3
    assert max(last_tick_reached_executed) == 3
    assert never_reached_executed[4:] == (3,) * 6
    assert last_tick_reached_executed[4:] == (3,) * 6
    with pytest.raises(ValueError, match="positive integer"):
        build(0, 6)
    with pytest.raises(ValueError, match="nonnegative integer"):
        build(4, -1)


def test_applied_action_gate_is_strict_float32_equivalent():
    compare = _module()["compare_applied_arm_command_float32"]
    expected = np.asarray(
        (0.4422, 0.4996, -0.2392, -0.7814, 0.1191, 1.8705, 0.6504),
        dtype=np.float64,
    )
    arm_indices = np.arange(7, dtype=np.int32)
    controlled_indices = np.arange(10, dtype=np.int32)
    full_readback = np.zeros(12, dtype=np.float32)
    full_readback[arm_indices] = expected.astype(np.float32)
    result = compare(
        full_readback,
        controlled_indices,
        arm_indices,
        expected,
        12,
    )
    assert result["float32_equivalent_gate"] is True
    assert result["maximum_abs_arm_error_float32_rad"] == 0.0
    assert result["storage"] == "full_articulation"

    subset = full_readback[controlled_indices]
    subset_result = compare(
        subset,
        controlled_indices,
        arm_indices,
        expected,
        12,
    )
    assert subset_result["float32_equivalent_gate"] is True
    assert subset_result["storage"] == "controlled_subset"

    mutated = full_readback.copy()
    mutated[0] = np.nextafter(mutated[0], np.float32(np.inf))
    rejected = compare(
        mutated,
        controlled_indices,
        arm_indices,
        expected,
        12,
    )
    assert rejected["float32_equivalent_gate"] is False
    assert rejected["maximum_abs_arm_error_float32_rad"] > 0.0
    with pytest.raises(ValueError, match="shape changed"):
        compare(
            full_readback[:-1],
            controlled_indices,
            arm_indices,
            expected,
            12,
        )
    nonfinite = full_readback.copy()
    nonfinite[0] = np.nan
    with pytest.raises(ValueError, match="finite flat"):
        compare(
            nonfinite,
            controlled_indices,
            arm_indices,
            expected,
            12,
        )


def test_retract_preflight_report_is_bound_to_exact_runtime_and_variant():
    validate = _module()["validate_tactile_retract_preflight_report"]
    phase_names = ("descent", "reversal", "recovery")
    peak_names = (
        "absolute_axial_force_n",
        "lateral_force_n",
        "bending_torque_nm",
        "absolute_tightening_torque_nm",
        "absolute_finger_base_torque_nm",
    )
    contact_names = (
        "loose_fixed",
        "intended_lip",
        "unexpected_loose_fixed",
        "loose_fixture",
        "loose_table",
    )
    ceilings = {
        "absolute_axial_force_n": 5.0,
        "lateral_force_n": 2.0,
        "bending_torque_nm": 0.18,
        "absolute_tightening_torque_nm": 0.05,
        "absolute_finger_base_torque_nm": 2.0,
    }
    peak_snapshots = (
        dict(zip(peak_names, (0.2, 0.1, 0.01, 0.002, 0.3))),
        dict(zip(peak_names, (0.3, 0.2, 0.02, 0.003, 0.4))),
        dict(zip(peak_names, (0.4, 0.25, 0.03, 0.004, 0.5))),
    )
    sample_rate_hz = 240.0

    def phase_evidence(base_z, base_gap, progress, peaks):
        progress = tuple(progress)
        tcp = [[0.55, 0.185, base_z + value] for value in progress]
        gaps = [base_gap + value for value in progress]
        peak_speed = max(
            abs(progress[index] - progress[index - 1]) * sample_rate_hz
            for index in range(1, len(progress))
        )
        count = len(progress)
        return {
            "measured_start_tcp_prim_world_m": [0.55, 0.185, base_z],
            "estimated_start_gap_m": base_gap,
            "command_start_fk_tcp_world_m": [0.55, 0.185, base_z],
            "estimated_gap_samples_m": gaps,
            "measured_tcp_prim_world_samples_m": tcp,
            "task_z_progress_samples_m": list(progress),
            "command_fk_task_z_progress_samples_m": list(progress),
            "command_fk_tcp_world_samples_m": copy.deepcopy(tcp),
            "peak_abs_task_z_speed_m_s": peak_speed,
            "peak_abs_command_fk_task_z_speed_m_s": peak_speed,
            "checked_sample_count": count,
            "finite_sample_count": count,
            "all_samples_finite": True,
            "minimum_body_contact_finger_count": 3,
            "applied_action_precheck_count": count,
            "applied_action_postcheck_count": count,
            "applied_action_all_float32_equivalent": True,
            "applied_action_maximum_abs_arm_error_float32_rad": 0.0,
            "contact_record_totals": {name: 0 for name in contact_names},
            "peak_experimental_observations": copy.deepcopy(peaks),
        }

    phase_evidence_values = (
        phase_evidence(
            0.322,
            0.012,
            (-1.0e-6, -2.0e-6, -3.0e-6),
            peak_snapshots[0],
        ),
        phase_evidence(
            0.321997,
            0.011997,
            (-1.0e-6, 0.0, 1.0e-6),
            peak_snapshots[1],
        ),
        phase_evidence(
            0.321998,
            0.011998,
            (1.0e-6, 2.0e-6, 3.0e-6),
            peak_snapshots[2],
        ),
    )
    authored = {
        "loose_plug_xy_m": [0.53, -0.2],
        "fixed_receptacle_xy_m": [0.55, 0.185],
        "loose_yaw_rad": 0.0,
        "fixed_yaw_rad": 0.0,
    }
    report = {
        "passed": True,
        "preinsert_probe_requested": True,
        "tactile_retract_preflight_requested": True,
        "tactile_lip_calibration_requested": False,
        "tactile_lip_manifold_capture_requested": False,
        "truth_xy_used_for_target": False,
        "object_pose_writes_after_physics": 0,
        "config_sha256": "visual-sha",
        "preinsert_config_sha256": "preinsert-sha",
        "tactile_engage_config_sha256": "tactile-sha",
        "runtime_source_import_sha256": "runtime-sha",
        "runtime_source_start_sha256": "runtime-sha",
        "runtime_source_finalize_sha256": "runtime-sha",
        "runtime_source_unchanged": True,
        "trial_id": "loose_plus_10mm_xy_fixed_nominal",
        "authored_before_physics": copy.deepcopy(authored),
        "tactile_retract_preflight": {
            "passed": True,
            "status": (
                "PASSED_NO_CONTACT_COMMAND_REVERSAL_AT_PREINSERT"
            ),
            "lip_contact_executed": False,
            "touch_trials_executed": 0,
            "engage_executed": False,
            "insertion_executed": False,
            "twist_executed": False,
            "home_return_executed": False,
            "assembly_success_claimed": False,
            "all_trajectory_samples_outside_entry_gate": True,
            "measured_descent_bound_gate": True,
            "reversal_negative_progress_bound_gate": True,
            "all_phase_measured_speed_gates": True,
            "all_phase_command_speed_gates": True,
            "phase_sample_count_gates": True,
            "all_samples_finite_gate": True,
            "three_finger_body_contact_gate": True,
            "applied_action_all_float32_equivalent": True,
            "applied_action_evidence_gate": True,
            "all_contact_record_totals_zero_gate": True,
            "experimental_abort_envelope_gate": True,
            "observed_negative_reversal_progress_bound_m": 1.0e-6,
            "measured_descent_tcp_prim_m": 3.0e-6,
            "commanded_descent_m": 0.00045,
            "maximum_actual_descent_m": 0.0005,
            "maximum_commanded_tcp_speed_m_s": 0.00035,
            "maximum_measured_tcp_speed_m_s": 0.0005,
            "minimum_measured_gap_m": 0.011996,
            "entry_gap_floor_m": 0.010,
            "task_frame_axes_world": {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
                "determinant": 1.0,
            },
            "measured_phase_peak_speeds_m_s": {
                name: phase_evidence_values[index][
                    "peak_abs_task_z_speed_m_s"
                ]
                for index, name in enumerate(phase_names)
            },
            "command_fk_phase_peak_speeds_m_s": {
                name: phase_evidence_values[index][
                    "peak_abs_command_fk_task_z_speed_m_s"
                ]
                for index, name in enumerate(phase_names)
            },
            "trajectory_sample_counts": {name: 3 for name in phase_names},
            "total_checked_sample_count": 9,
            "total_finite_sample_count": 9,
            "minimum_body_contact_finger_count": 3,
            "applied_action_precheck_count": 9,
            "applied_action_postcheck_count": 9,
            "applied_action_maximum_abs_arm_error_float32_rad": 0.0,
            "contact_record_totals": {name: 0 for name in contact_names},
            "peak_experimental_observations": copy.deepcopy(
                peak_snapshots[-1]
            ),
            "experimental_abort_ceilings": copy.deepcopy(ceilings),
            "descent_evidence": phase_evidence_values[0],
            "reversal_evidence": phase_evidence_values[1],
            "recovery_evidence": phase_evidence_values[2],
        },
    }
    expected = {
        "expected_visual_config_sha256": "visual-sha",
        "expected_preinsert_config_sha256": "preinsert-sha",
        "expected_tactile_config_sha256": "tactile-sha",
        "expected_runtime_source_import_sha256": "runtime-sha",
        "expected_runtime_source_start_sha256": "runtime-sha",
        "expected_trial_id": "loose_plus_10mm_xy_fixed_nominal",
        "expected_authored_before_physics": authored,
        "maximum_negative_progress_m": 0.0005,
        "minimum_allowed_gap_m": 0.010,
        "sample_rate_hz": sample_rate_hz,
        "expected_commanded_descent_m": 0.00045,
        "expected_maximum_commanded_speed_m_s": 0.00035,
        "maximum_measured_speed_m_s": 0.0005,
        "expected_experimental_abort_ceilings": ceilings,
        "expected_body_contact_finger_count": 3,
    }
    assert validate(report, **expected) == pytest.approx(1.0e-6)

    # Equality at each immutable ceiling is accepted.  The synchronized raw
    # TCP/progress mutation ensures this tests the derived speed boundary, not
    # merely a trusted summary scalar.
    boundary = copy.deepcopy(report)
    descent = boundary["tactile_retract_preflight"]["descent_evidence"]
    hard_speed_step_m = 0.0005 / sample_rate_hz
    descent_progress = [
        -hard_speed_step_m * multiplier for multiplier in (1.0, 2.0, 3.0)
    ]
    descent["task_z_progress_samples_m"] = descent_progress
    descent["measured_tcp_prim_world_samples_m"] = [
        [0.55, 0.185, 0.322 + value] for value in descent_progress
    ]
    descent["estimated_gap_samples_m"] = [
        0.012 + value for value in descent_progress
    ]
    descent["peak_abs_task_z_speed_m_s"] = 0.0005
    section = boundary["tactile_retract_preflight"]
    section["measured_descent_tcp_prim_m"] = -min(descent_progress)
    section["minimum_measured_gap_m"] = min(
        sample
        for name in phase_names
        for sample in section[f"{name}_evidence"][
            "estimated_gap_samples_m"
        ]
    )
    section["measured_phase_peak_speeds_m_s"]["descent"] = 0.0005
    recovery = section["recovery_evidence"]
    command_speed_step_m = 0.00035 / sample_rate_hz
    recovery_command_progress = [
        command_speed_step_m * multiplier
        for multiplier in (1.0, 2.0, 3.0)
    ]
    recovery["command_fk_task_z_progress_samples_m"] = (
        recovery_command_progress
    )
    recovery["command_fk_tcp_world_samples_m"] = [
        [0.55, 0.185, 0.321998 + value]
        for value in recovery_command_progress
    ]
    recovery["peak_abs_command_fk_task_z_speed_m_s"] = 0.00035
    section["command_fk_phase_peak_speeds_m_s"]["recovery"] = 0.00035
    recovery["peak_experimental_observations"][
        "absolute_axial_force_n"
    ] = 5.0
    section["peak_experimental_observations"][
        "absolute_axial_force_n"
    ] = 5.0
    assert validate(boundary, **expected) == pytest.approx(1.0e-6)

    actual_over = copy.deepcopy(boundary)
    over_section = actual_over["tactile_retract_preflight"]
    over_descent = over_section["descent_evidence"]
    over_actual_speed = 0.000500001
    over_step = over_actual_speed / sample_rate_hz
    over_progress = [-over_step * value for value in (1.0, 2.0, 3.0)]
    over_descent["task_z_progress_samples_m"] = over_progress
    over_descent["measured_tcp_prim_world_samples_m"] = [
        [0.55, 0.185, 0.322 + value] for value in over_progress
    ]
    over_descent["estimated_gap_samples_m"] = [
        0.012 + value for value in over_progress
    ]
    over_descent["peak_abs_task_z_speed_m_s"] = over_actual_speed
    over_section["measured_phase_peak_speeds_m_s"][
        "descent"
    ] = over_actual_speed
    over_section["measured_descent_tcp_prim_m"] = -min(over_progress)
    over_section["minimum_measured_gap_m"] = min(
        sample
        for name in phase_names
        for sample in over_section[f"{name}_evidence"][
            "estimated_gap_samples_m"
        ]
    )
    with pytest.raises(ValueError, match="speed evidence"):
        validate(actual_over, **expected)

    command_over = copy.deepcopy(boundary)
    over_section = command_over["tactile_retract_preflight"]
    over_recovery = over_section["recovery_evidence"]
    over_command_speed = 0.000350001
    over_step = over_command_speed / sample_rate_hz
    over_progress = [over_step * value for value in (1.0, 2.0, 3.0)]
    over_recovery["command_fk_task_z_progress_samples_m"] = over_progress
    over_recovery["command_fk_tcp_world_samples_m"] = [
        [0.55, 0.185, 0.321998 + value] for value in over_progress
    ]
    over_recovery[
        "peak_abs_command_fk_task_z_speed_m_s"
    ] = over_command_speed
    over_section["command_fk_phase_peak_speeds_m_s"][
        "recovery"
    ] = over_command_speed
    with pytest.raises(ValueError, match="speed evidence"):
        validate(command_over, **expected)

    mutations = (
        (("config_sha256",), "other-visual"),
        (("preinsert_config_sha256",), "other-preinsert"),
        (("tactile_engage_config_sha256",), "other-tactile"),
        (("runtime_source_import_sha256",), "other-runtime"),
        (("runtime_source_start_sha256",), "other-runtime"),
        (("runtime_source_finalize_sha256",), "other-runtime"),
        (("runtime_source_unchanged",), False),
        (("trial_id",), "loose_plus_20mm_x_fixed_nominal"),
        (("authored_before_physics", "loose_plug_xy_m"), [0.54, -0.2]),
        (("object_pose_writes_after_physics",), 1),
        (("tactile_retract_preflight_requested",), False),
        (("tactile_retract_preflight", "passed"), False),
        (("tactile_retract_preflight", "lip_contact_executed"), True),
        (("tactile_retract_preflight", "touch_trials_executed"), True),
        (
            (
                "tactile_retract_preflight",
                "all_trajectory_samples_outside_entry_gate",
            ),
            False,
        ),
        (
            (
                "tactile_retract_preflight",
                "measured_descent_bound_gate",
            ),
            False,
        ),
        (
            (
                "tactile_retract_preflight",
                "reversal_negative_progress_bound_gate",
            ),
            False,
        ),
        (
            (
                "tactile_retract_preflight",
                "observed_negative_reversal_progress_bound_m",
            ),
            0.00051,
        ),
        (
            (
                "tactile_retract_preflight",
                "all_phase_measured_speed_gates",
            ),
            False,
        ),
        (
            (
                "tactile_retract_preflight",
                "commanded_descent_m",
            ),
            0.0005,
        ),
        (
            (
                "tactile_retract_preflight",
                "maximum_commanded_tcp_speed_m_s",
            ),
            0.00036,
        ),
        (
            (
                "tactile_retract_preflight",
                "total_finite_sample_count",
            ),
            8,
        ),
        (
            (
                "tactile_retract_preflight",
                "minimum_body_contact_finger_count",
            ),
            2,
        ),
        (
            (
                "tactile_retract_preflight",
                "applied_action_precheck_count",
            ),
            8,
        ),
        (
            (
                "tactile_retract_preflight",
                "applied_action_maximum_abs_arm_error_float32_rad",
            ),
            1.0e-7,
        ),
        (
            (
                "tactile_retract_preflight",
                "contact_record_totals",
                "loose_table",
            ),
            1,
        ),
        (
            (
                "tactile_retract_preflight",
                "peak_experimental_observations",
                "absolute_axial_force_n",
            ),
            5.0001,
        ),
        (
            (
                "tactile_retract_preflight",
                "descent_evidence",
                "estimated_gap_samples_m",
            ),
            [0.00999, 0.011998, 0.011997],
        ),
        (
            (
                "tactile_retract_preflight",
                "reversal_evidence",
                "measured_tcp_prim_world_samples_m",
            ),
            [
                [0.55, 0.185, float("nan")],
                [0.55, 0.185, 0.321997],
                [0.55, 0.185, 0.321998],
            ],
        ),
        (
            (
                "tactile_retract_preflight",
                "descent_evidence",
                "task_z_progress_samples_m",
            ),
            [-1.0e-6, -1.0e-3, -3.0e-6],
        ),
        (
            (
                "tactile_retract_preflight",
                "recovery_evidence",
                "command_fk_task_z_progress_samples_m",
            ),
            [1.0e-6, 1.0e-3, 3.0e-6],
        ),
        (
            (
                "tactile_retract_preflight",
                "descent_evidence",
                "applied_action_postcheck_count",
            ),
            2,
        ),
        (
            (
                "tactile_retract_preflight",
                "descent_evidence",
                "contact_record_totals",
                "unexpected_loose_fixed",
            ),
            1,
        ),
        (
            (
                "tactile_retract_preflight",
                "reversal_evidence",
                "peak_experimental_observations",
                "absolute_axial_force_n",
            ),
            0.1,
        ),
        (
            (
                "tactile_retract_preflight",
                "task_frame_axes_world",
                "z",
            ),
            [0.0, 0.0, 2.0],
        ),
    )
    for path, replacement in mutations:
        mutated = copy.deepcopy(report)
        destination = mutated
        for name in path[:-1]:
            destination = destination[name]
        destination[path[-1]] = replacement
        with pytest.raises(ValueError):
            validate(mutated, **expected)


def test_first_gpu_preflight_failure_preserves_hard_motion_ceilings():
    repository = Path(__file__).resolve().parents[3]
    artifact = (
        repository
        / "artifacts/kcg_connector/d38999_tactile_retract_preflight_v1/"
        "gpu_20260812T184500Z/report.json"
    )
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "ce5297579adf6f6e41265d0b536d20e2184304a622b49aa47ccbd34287881a1a"
    )
    report = json.loads(artifact.read_text(encoding="utf-8"))
    section = report["tactile_retract_preflight"]
    assert report["passed"] is False
    assert section["passed"] is False
    assert section["status"] == "REJECTED_FAIL_CLOSED"
    # The artifact proves why command headroom was introduced; neither of the
    # immutable actual-motion safety ceilings is relaxed by the fix.
    assert section["measured_descent_tcp_prim_m"] > 0.0005
    measured_peaks = {
        name: section[f"{name}_evidence"][
            "peak_abs_task_z_speed_m_s"
        ]
        for name in ("descent", "reversal", "recovery")
    }
    assert all(value > 0.0005 for value in measured_peaks.values())
    assert section["minimum_measured_gap_m"] > 0.010
    assert section["observed_negative_reversal_progress_bound_m"] == 0.0
    module = _module()
    assert module["TACTILE_RETRACT_PREFLIGHT_COMMAND_DESCENT_M"] == 0.00045
    assert module["TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S"] == 0.00035
    assert module["TACTILE_RETRACT_PREFLIGHT_DESCENT_M"] == 0.0005


def test_script_requires_explicit_run_and_is_not_wired_into_e2e():
    source = _source()
    assert '"--run"' in source
    assert "requires explicit --run" in source
    assert "enabled_by_default" not in source
    assert "d38999_visual_xy_pick_smoke" not in E2E_PATH.read_text(
        encoding="utf-8"
    )


def test_preinsert_continuation_is_separate_opt_in_and_same_world_only():
    source = _source()
    assert '"--preinsert-probe"' in source
    assert '"--preinsert-config"' in source
    assert source.count("World(") == 1
    assert source.count("world.reset()") == 1
    assert "capture_world_reference = world" in source
    assert "world is capture_world_reference" in source
    assert "preinsert_plan.capture_id == plan.capture_id" in source

    build = source.index("build_visual_xy_preinsert_plan(")
    truth = source.index("evaluate_visual_xy_truth_only(")
    pick_pass = source.index("prior_visual_pick_passed = bool(passed)")
    first_motion = source.index('"visual_xy_transport_to_fixed_safe"')
    post_hoc = source.index("actual_alignment = measure_alignment(")
    assert build < truth < pick_pass < first_motion < post_hoc


def test_preinsert_continuation_has_strict_runtime_failure_gates():
    source = _source()
    for phase in (
        "visual_xy_transport_to_fixed_safe",
        "visual_xy_align_above_entry",
        "visual_xy_preinsert",
    ):
        assert phase in source
    for token in (
        "zero_preentry_contact_gate",
        "body_contact_retention_gate",
        "torque_hard_stop_gate",
        "finite_preinsert_gate",
        "tracking_and_speed_gate",
        "tcp_target_gate",
        "outside_entry_gate",
        "object_pose_write_gate",
        "same_world_capture_gate",
        "preinsert_minimum_body_contact_fingers == 3",
        "maximum_absolute_torque_delta_nm",
        '"truth_pose_feedback_used_for_target": False',
        '"engage_executed": False',
        '"assembly_success_claimed": False',
    ):
        assert token in source
    assert "set_world_pose" not in source
    assert "set_local_pose" not in source


def test_tactile_lip_calibration_is_explicit_lazy_and_stops_at_preinsert():
    source = _source()
    assert '"--tactile-lip-calibration"' in source
    assert "retired: the hand has no fingertip tactile sensor" in source
    assert '"contact truth is forbidden for control; use the wrist-FT-only "' in source
    assert '"--tactile-engage-config"' in source
    assert "tactile runtime requires explicit --preinsert-probe" in source
    assert '"--tactile-retract-preflight"' in source
    assert '"--tactile-retract-preflight-report"' in source
    assert (
        "tactile lip calibration requires a passed retract preflight"
        in source
    )
    assert "if arguments.tactile_lip_calibration:" in source
    assert "from kcg_connector.virtual_wrist_ft_runtime import (" in source
    assert "load_tactile_engage_contract" in source
    assert source.count("World(") == 1
    assert source.count("world.reset()") == 1
    for token in (
        "TACTILE_LIP_OFFSET_M = 0.0006",
        '("plus_x", (1.0, 0.0), "My", 4)',
        '("minus_x", (-1.0, 0.0), "My", 4)',
        '("plus_y", (0.0, 1.0), "Mx", 3)',
        '("minus_y", (0.0, -1.0), "Mx", 3)',
        'phase = "unsupported_final_hold"',
        '!= "INSERT"',
        "contact_candidate(",
        "contact_release_candidate(",
        "compressive_axial_force_sign_candidate",
        '"x_to_normalized_lever_x", "plus_x", "minus_x"',
        '"y_to_normalized_lever_y", "plus_y", "minus_y"',
        '"inferred_rx_equals_minus_My_over_compression"',
        '"inferred_ry_equals_Mx_over_compression"',
        '"minimum_diagnostic_increase_m"',
        '"diagnostic_unregistered_correlated_samples_"',
        '"absolute_single_touch_sign_claimed": False',
        '"truth_pose_used": False',
        '"local_moment_response_jacobian_only_not_center_"',
        '"pairwise_moment_response_gates"',
        "unload_retract_distance_m",
        '"engage_executed": False',
        '"insertion_executed": False',
        '"twist_executed": False',
        '"home_return_executed": False',
        '"hardware_safety_calibration_claimed": False',
        '"truth_pose_used_for_touch_control": False',
    ):
        assert token in source

    monitor_import = source.index("VirtualWristFtMonitor,")
    runtime_guard = source.rfind(
        "if tactile_runtime_requested:", 0, monitor_import
    )
    assert runtime_guard >= 0
    tree = ast.parse(source)
    initial_report = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "report"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    )
    initial_keys = {
        key.value
        for key in initial_report.keys
        if isinstance(key, ast.Constant)
    }
    assert "runtime_source_import_sha256" not in initial_keys
    assert '"runtime_source_import_sha256": (' in source


def test_tactile_pair_classifier_accepts_only_segmented_mating_lip():
    classify = _module()["classify_tactile_lip_contact_pair"]
    body = "/World/Pair/LoosePlug/BodyAssembly"
    nut = "/World/Pair/LoosePlug/CouplingNut"
    fixed = "/World/Pair/FixedReceptacle"
    intended = (
        body + "/MatingShell/Segment_08",
        fixed + "/EntryShell/Segment_13",
    )
    assert classify(intended, body, nut, fixed) == "intended_segmented_lip"
    assert classify(tuple(reversed(intended)), body, nut, fixed) == (
        "intended_segmented_lip"
    )
    assert classify(
        (nut + "/ToothSegment_08", fixed + "/EntryShell/Segment_13"),
        body,
        nut,
        fixed,
    ) == "unexpected_loose_fixed"
    assert classify(
        (body + "/RearBody", fixed + "/EntryShell/Segment_13"),
        body,
        nut,
        fixed,
    ) == "unexpected_loose_fixed"
    assert classify(
        (body + "/MatingShell/Segment_08", fixed + "/RearBody"),
        body,
        nut,
        fixed,
    ) == "unexpected_loose_fixed"
    assert classify((body, body + "/RearBody"), body, nut, fixed) is None


def test_tactile_manifold_evidence_preserves_order_and_recomputes_axes():
    module = _module()
    build = module["build_tactile_manifold_pair_evidence"]
    body = "/World/Pair/LoosePlug/BodyAssembly"
    fixed = "/World/Pair/FixedReceptacle"
    loose_segment = body + "/MatingShell/Segment_03"
    fixed_segment = fixed + "/EntryShell/Segment_04"
    point = {
        "normal": [0.0, 0.0, 1.0],
        "impulse": [0.0, 0.0, 0.0],
        "position": [0.55, 0.185, 0.2615],
        "separation": 2.0e-5,
        "material0": "/World/Looks/Loose",
        "material1": "/World/Looks/Fixed",
        "face_index0": 7,
        "face_index1": 9,
    }
    result = build(
        actor_paths=(body, fixed),
        collider_paths=(loose_segment, fixed_segment),
        event_type="PERSIST",
        contact_points=(point,),
        body_root=body,
        fixed_root=fixed,
        task_rotation_world=np.eye(3),
        task_origin_world=(0.55, 0.185, 0.26),
        physics_dt_s=1.0 / 240.0,
    )
    assert result["actor0"] == body
    assert result["actor1"] == fixed
    assert result["collider0"] == loose_segment
    assert result["collider1"] == fixed_segment
    assert result["loose_side"] == 0
    assert result["fixed_side"] == 1
    assert result["normal_convention"] == (
        "physx_pxcontactpoint_normal_points_shape1_to_shape0"
    )
    copied = result["contact_points"][0]
    assert copied["normal_world_reported"] == [0.0, 0.0, 1.0]
    assert copied["loose_resolution_normal_task"] == [0.0, 0.0, 1.0]
    assert copied["impulse_over_dt_task_diagnostic_n"] == [0.0] * 3
    assert copied["position_task_from_preinsert_origin_m"] == pytest.approx(
        [0.0, 0.0, 0.0015]
    )
    assert copied["separation_m"] == pytest.approx(2.0e-5)

    swapped = copy.deepcopy(point)
    swapped["normal"] = [0.0, 0.0, -1.0]
    swapped["material0"], swapped["material1"] = (
        swapped["material1"],
        swapped["material0"],
    )
    swapped["face_index0"], swapped["face_index1"] = (
        swapped["face_index1"],
        swapped["face_index0"],
    )
    reversed_result = build(
        actor_paths=(fixed, body),
        collider_paths=(fixed_segment, loose_segment),
        event_type="PERSIST",
        contact_points=(swapped,),
        body_root=body,
        fixed_root=fixed,
        task_rotation_world=np.eye(3),
        task_origin_world=(0.55, 0.185, 0.26),
        physics_dt_s=1.0 / 240.0,
    )
    assert reversed_result["loose_side"] == 1
    assert reversed_result["contact_points"][0][
        "loose_resolution_normal_task"
    ] == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="not unit"):
        build(
            actor_paths=(body, fixed),
            collider_paths=(loose_segment, fixed_segment),
            event_type="PERSIST",
            contact_points=({**point, "normal": [0.0, 0.0, 0.9]},),
            body_root=body,
            fixed_root=fixed,
            task_rotation_world=np.eye(3),
            task_origin_world=(0.55, 0.185, 0.26),
            physics_dt_s=1.0 / 240.0,
        )


def test_tactile_manifold_mode_is_single_direction_frozen_and_terminal():
    source = _source()
    for token in (
        '"--tactile-lip-manifold-capture"',
        "TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S = 0.00035",
        '"PLUS_X_LIP_MANIFOLD_CAPTURE_ONLY"',
        '"manifold_capture_only": True',
        '"ft_sign_calibrated": False',
        '"stage_b_authorized": False',
        "capture_manifold=True",
        "contact_hold_limit = (",
        "tactile_probe.sensor.contact_debounce_samples",
        '"frozen_command_gate"',
        '"exact_contact_manifold_gate"',
        '"PASSED_PLUS_X_LIP_MANIFOLD_CAPTURE_"',
        "perform_tactile_abort_retract(",
        "manifold_capture_success=True",
        "raise TactileManifoldComplete",
        "result_marker = TACTILE_LIP_MANIFOLD_RESULT_MARKER",
    ):
        assert token in source
    branch_start = source.index(
        "if arguments.tactile_lip_manifold_capture:",
        source.index("raise TactilePreflightComplete"),
    )
    branch_end = source.index(
        "if arguments.tactile_lip_calibration:", branch_start
    )
    branch = source[branch_start:branch_end]
    assert "TACTILE_LIP_MANIFOLD_DIRECTION" in branch
    assert "TACTILE_LIP_DIRECTIONS" not in branch
    assert "contact_candidate(" not in branch
    assert "moment_guided" not in branch
    assert '"terminal_retract": terminal_retract' in branch
    assert '"engage_executed": False' in branch

    # Mutation checks ensure the static contract detects accidental preload,
    # a larger command speed, or authorization leakage.
    mutations = (
        (
            "TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S = 0.00035",
            "TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S = 0.0005",
            lambda candidate: (
                "TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S = 0.00035"
                in candidate
            ),
        ),
        (
            '"stage_b_authorized": False',
            '"stage_b_authorized": True',
            lambda candidate: (
                '"stage_b_authorized": False' in candidate
            ),
        ),
        (
            "capture_manifold=True",
            "capture_manifold=False",
            lambda candidate: "capture_manifold=True" in candidate,
        ),
    )
    for original, replacement, contract in mutations:
        mutated = source.replace(original, replacement)
        assert contract(mutated) is False


def test_tactile_manifold_runtime_validator_is_live_and_raw_fail_closed():
    source = _source()
    branch_start = source.index(
        "if arguments.tactile_lip_manifold_capture:",
        source.index("raise TactilePreflightComplete"),
    )
    branch_end = source.index(
        "if arguments.tactile_lip_calibration:", branch_start
    )
    branch = source[branch_start:branch_end]
    for token in (
        "validate_tactile_manifold_capture_evidence(",
        'tactile_report["raw_evidence_validation"]',
        'manifold_validation.get("validated") is True',
        "record_raw_guarded_samples=True",
        "release_after_retract=manifold_mode",
        '"guarded_tick_samples"',
        '"applied_pre_readback_float32"',
        '"applied_post_readback_float32"',
        '"release_compression_ceiling_n"',
        '"release_bending_ceiling_nm"',
        '"contact_query_owner_type_name"',
        '"VALIDATING_RAW_MANIFOLD_EVIDENCE_"',
    ):
        assert token in source
    assert branch.count("record_raw_guarded_samples=True") == 2
    # The legacy four-way mode keeps its old release-window behavior; only the
    # new terminal manifold unload opts into post-distance release debounce.
    assert "release_after_retract=True" not in branch
    assert source.count("release_after_retract=manifold_mode") == 1
    validator_call = branch.index(
        "validate_tactile_manifold_capture_evidence("
    )
    candidate_fail_closed = branch.index(
        '"passed": False',
        branch.index('"VALIDATING_RAW_MANIFOLD_EVIDENCE_"'),
    )
    final_pass_commit = branch.index(
        'tactile_report["passed"] = manifold_passed'
    )
    assert candidate_fail_closed < validator_call < final_pass_commit

    mutations = (
        (
            "validate_tactile_manifold_capture_evidence(",
            "disabled_manifold_evidence_validator(",
        ),
        (
            'manifold_validation.get("validated") is True',
            "True",
        ),
        (
            "record_raw_guarded_samples=True",
            "record_raw_guarded_samples=False",
        ),
        (
            "release_after_retract=manifold_mode",
            "release_after_retract=False",
        ),
    )
    for original, replacement in mutations:
        # Mutate the retired branch itself.  The same helper arguments are now
        # also used by the replacement wrist-FT preflight earlier in the file,
        # so a whole-file first-occurrence replacement is no longer evidence
        # that this legacy branch is fail closed.
        mutation_start = (
            source.index(original)
            if original == "release_after_retract=manifold_mode"
            else branch_start
        )
        mutated = (
            source[:mutation_start]
            + source[mutation_start:].replace(original, replacement, 1)
        )
        mutated_branch = mutated[
            mutated.index(
                "if arguments.tactile_lip_manifold_capture:",
                mutated.index("raise TactilePreflightComplete"),
            ):
        ]
        if original == "validate_tactile_manifold_capture_evidence(":
            assert "validate_tactile_manifold_capture_evidence(" not in (
                mutated_branch
            )
        elif original == 'manifold_validation.get("validated") is True':
            assert original not in mutated_branch
        elif original == "record_raw_guarded_samples=True":
            assert mutated_branch.count(original) < branch.count(original)
        else:
            assert mutated.count(original) < source.count(original)


def test_tactile_manifold_raw_builder_rejects_actor_collider_mismatch():
    build = _module()["build_tactile_manifold_pair_evidence"]
    body = "/World/Pair/LoosePlug/BodyAssembly"
    fixed = "/World/Pair/FixedReceptacle"
    point = {
        "normal": [1.0, 0.0, 0.0],
        "impulse": [0.0, 0.0, 0.0],
        "position": [0.55, 0.185, 0.26],
        "separation": 0.0,
        "material0": "/World/Looks/Loose",
        "material1": "/World/Looks/Fixed",
        "face_index0": 1,
        "face_index1": 2,
    }
    with pytest.raises(ValueError, match="actor ordering"):
        build(
            actor_paths=(fixed, body),
            collider_paths=(
                body + "/MatingShell/Segment_03",
                fixed + "/EntryShell/Segment_04",
            ),
            event_type="PERSIST",
            contact_points=(point,),
            body_root=body,
            fixed_root=fixed,
            task_rotation_world=np.eye(3),
            task_origin_world=(0.55, 0.185, 0.26),
            physics_dt_s=1.0 / 240.0,
        )


def test_manifold_validator_accepts_raw_fixture_and_rejects_mutations():
    module = _module()
    validate = module["validate_tactile_manifold_capture_evidence"]
    build_pair = module["build_tactile_manifold_pair_evidence"]
    peak_names = module["TACTILE_PREFLIGHT_PEAK_FIELDS"]
    contact_names = module["TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS"]
    body = "/World/Pair/LoosePlug/BodyAssembly"
    fixed = "/World/Pair/FixedReceptacle"
    loose_segment = body + "/MatingShell/Segment_03"
    fixed_segment = fixed + "/EntryShell/Segment_04"
    rate = 240.0
    origin = [0.55, 0.185, 0.32]
    arm = [0.0] * 7
    arm32 = [float(value) for value in np.asarray(arm, dtype=np.float32)]
    ceilings = dict(zip(peak_names, (5.0, 2.0, 0.18, 0.05, 2.0)))
    raw_peaks = dict(zip(peak_names, (0.0, 0.0, 0.0, 0.0, 0.1)))
    raw_point = {
        "normal": [1.0, 0.0, 0.0],
        "impulse": [0.0, 0.0, 0.0],
        "position": [0.5506, 0.185, 0.309],
        "separation": 1.0e-5,
        "material0": "/World/Looks/Loose",
        "material1": "/World/Looks/Fixed",
        "face_index0": 3,
        "face_index1": 4,
    }
    manifold = build_pair(
        actor_paths=(body, fixed),
        collider_paths=(loose_segment, fixed_segment),
        event_type="PERSIST",
        contact_points=(raw_point,),
        body_root=body,
        fixed_root=fixed,
        task_rotation_world=np.eye(3),
        task_origin_world=origin,
        physics_dt_s=1.0 / rate,
    )
    pair = {
        "paths": [body, fixed, loose_segment, fixed_segment],
        "contact_records": 1,
        "contact_manifold": manifold,
    }

    def phase(start, deltas, global_start, *, intended=False, peaks=None):
        measured = [
            [start[axis] + delta[axis] for axis in range(3)]
            for delta in deltas
        ]
        progress = [position[2] - start[2] for position in measured]
        gaps = [0.012 + value for value in progress]
        ticks = []
        for index, position in enumerate(measured):
            pairs = [copy.deepcopy(pair)] if intended else []
            ticks.append(
                {
                    "global_step": global_start + index,
                    "command_arm_rad": list(arm),
                    "command_fk_tcp_world_m": list(position),
                    "measured_tcp_prim_world_m": list(position),
                    "estimated_gap_m": gaps[index],
                    "finite": True,
                    "body_contact_fingers": ["f1", "f2", "f3"],
                    "finger_base_torque_delta_nm": [0.1, 0.1, 0.1],
                    "raw_wrench": [0.0] * 6,
                    "canonical_wrench_sensor": [0.0] * 6,
                    "compensated_wrench_sensor": [0.0] * 6,
                    "compensated_wrench_task": [0.0] * 6,
                    "applied_pre_float32_gate": True,
                    "applied_post_float32_gate": True,
                    "applied_pre_max_error_float32_rad": 0.0,
                    "applied_post_max_error_float32_rad": 0.0,
                    "applied_pre_readback_float32": list(arm32),
                    "applied_pre_expected_float32": list(arm32),
                    "applied_post_readback_float32": list(arm32),
                    "applied_post_expected_float32": list(arm32),
                    "loose_fixed_contact_records": int(intended),
                    "intended_lip_contact_pairs": pairs,
                    "unexpected_loose_fixed_contact_pairs": [],
                    "loose_fixture_contact_pairs": [],
                    "loose_table_contact_pairs": [],
                }
            )
        positions = [start, *measured]
        full_speed = max(
            np.linalg.norm(
                np.asarray(positions[index + 1])
                - np.asarray(positions[index])
            )
            * rate
            for index in range(len(measured))
        )
        axial_speed = max(
            abs(progress[index] - (0.0 if index == 0 else progress[index - 1]))
            * rate
            for index in range(len(progress))
        )
        contacts = {name: 0 for name in contact_names}
        if intended:
            contacts["loose_fixed"] = len(measured)
            contacts["intended_lip"] = len(measured)
        return {
            "measured_start_tcp_prim_world_m": list(start),
            "command_start_fk_tcp_world_m": list(start),
            "command_start_arm_rad": list(arm),
            "estimated_start_gap_m": 0.012,
            "measured_tcp_prim_world_samples_m": measured,
            "command_fk_tcp_world_samples_m": copy.deepcopy(measured),
            "task_z_progress_samples_m": progress,
            "command_fk_task_z_progress_samples_m": list(progress),
            "estimated_gap_samples_m": gaps,
            "guarded_tick_samples": ticks,
            "checked_sample_count": len(measured),
            "finite_sample_count": len(measured),
            "all_samples_finite": True,
            "minimum_body_contact_finger_count": 3,
            "applied_action_precheck_count": len(measured),
            "applied_action_postcheck_count": len(measured),
            "applied_action_all_float32_equivalent": True,
            "applied_action_maximum_abs_arm_error_float32_rad": 0.0,
            "contact_record_totals": contacts,
            "peak_experimental_observations": copy.deepcopy(
                raw_peaks if peaks is None else peaks
            ),
            "minimum_task_z_progress_m": min(progress),
            "maximum_task_z_progress_m": max(progress),
            "final_task_z_progress_m": progress[-1],
            "peak_abs_task_z_speed_m_s": axial_speed,
            "peak_abs_command_fk_task_z_speed_m_s": axial_speed,
            "peak_abs_tcp_speed_m_s": full_speed,
            "peak_abs_command_fk_tcp_speed_m_s": full_speed,
            "required_retract_distance_reached": False,
            "post_retract_release_ticks": 0,
            "release_samples": [],
        }

    lateral = phase(
        origin,
        ([1.0e-6, 0.0, 0.0], [2.0e-6, 0.0, 0.0]),
        100,
    )
    approach_start = lateral["measured_tcp_prim_world_samples_m"][-1]
    approach = phase(
        approach_start,
        ([0.0, 0.0, 0.0],) * 6,
        102,
        intended=True,
        peaks=raw_peaks,
    )
    retract_start = approach["measured_tcp_prim_world_samples_m"][-1]
    retract_progress = [
        0.0003 * float(index) / 206.0 for index in range(1, 207)
    ] + [0.0003] * 5
    retract = phase(
        retract_start,
        tuple([0.0, 0.0, value] for value in retract_progress),
        108,
    )
    release_samples = [
        {
            "task_z_progress_m": 0.0003,
            "physical_contact": False,
            "signed_compression_n": 0.0,
            "bending_torque_nm": 0.0,
            "release_candidate": True,
            "intended_lip_contact_pairs": [],
        }
        for _ in range(6)
    ]
    retract.update(
        {
            "required_retract_distance_reached": True,
            "post_retract_release_ticks": 6,
            "release_samples": release_samples,
        }
    )
    frames = []
    for index, tick in enumerate(approach["guarded_tick_samples"]):
        frames.append(
            {
                "frame_index": index,
                "global_step": tick["global_step"],
                "command_arm_rad": list(arm),
                "command_fk_tcp_world_m": list(
                    tick["command_fk_tcp_world_m"]
                ),
                "measured_arm_rad": list(arm),
                "applied_action_arm_readback_float32": list(arm32),
                "expected_command_float32": list(arm32),
                "applied_action_arm_error_float32_rad": [0.0] * 7,
                "applied_action_float32_equivalent_gate": True,
                "measured_tcp_prim_world_m": list(
                    tick["measured_tcp_prim_world_m"]
                ),
                "raw_wrench": [0.0] * 6,
                "canonical_wrench_sensor": [0.0] * 6,
                "compensated_wrench_sensor": [0.0] * 6,
                "compensated_wrench_task": [0.0] * 6,
                "estimated_gap_m": tick["estimated_gap_m"],
                "intended_lip_contact_pairs": [copy.deepcopy(pair)],
                "unexpected_loose_fixed_contact_pairs": [],
                "loose_fixture_contact_pairs": [],
                "loose_table_contact_pairs": [],
            }
        )
    phases = (lateral, approach, retract)
    names = (
        "plus_x_offset",
        "guarded_approach_and_frozen_hold",
        "terminal_retract",
    )
    contact_totals = {
        name: sum(
            phase_value["contact_record_totals"][name]
            for phase_value in phases
        )
        for name in contact_names
    }
    report = {
        "task_frame_axes_world": {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
            "determinant": 1.0,
        },
        "task_frame_origin_world_m": list(origin),
        "manifold_capture_only": True,
        "ft_sign_calibrated": False,
        "stage_b_authorized": False,
        "direction": "plus_x",
        "known_offset_task_xy_m": [0.0006, 0.0],
        "maximum_commanded_tcp_speed_m_s": 0.00035,
        "maximum_measured_tcp_speed_m_s": 0.0005,
        "entry_gap_floor_m": 0.01,
        "release_compression_ceiling_n": 0.1,
        "release_bending_ceiling_nm": 0.004,
        "expected_manifold_frame_count": 6,
        "contact_runtime_identity": {
            "contact_query_callable_module": "omni.physx.bindings",
            "contact_query_callable_name": "get_full_contact_report",
            "contact_query_owner_type_module": "omni.physx.bindings",
            "contact_query_owner_type_name": "IPhysxSimulation",
            "normal_convention": (
                "physx_pxcontactpoint_normal_points_shape1_to_shape0"
            ),
            "simulation_app_type_module": "isaacsim",
            "simulation_app_type_name": "SimulationApp",
        },
        "manifold_frames": frames,
        "lateral_motion_evidence": lateral,
        "approach_and_hold_motion_evidence": approach,
        "actual_phase_peak_speeds_m_s": {
            name: phase_value["peak_abs_tcp_speed_m_s"]
            for name, phase_value in zip(names, phases)
        },
        "command_phase_peak_speeds_m_s": {
            name: phase_value["peak_abs_command_fk_tcp_speed_m_s"]
            for name, phase_value in zip(names, phases)
        },
        "minimum_measured_gap_m": min(
            min(phase_value["estimated_gap_samples_m"])
            for phase_value in phases
        ),
        "total_checked_sample_count": sum(
            phase_value["checked_sample_count"] for phase_value in phases
        ),
        "total_finite_sample_count": sum(
            phase_value["finite_sample_count"] for phase_value in phases
        ),
        "minimum_body_contact_finger_count": 3,
        "applied_action_precheck_count": sum(
            phase_value["checked_sample_count"] for phase_value in phases
        ),
        "applied_action_postcheck_count": sum(
            phase_value["checked_sample_count"] for phase_value in phases
        ),
        "applied_action_maximum_abs_arm_error_float32_rad": 0.0,
        "contact_record_totals": contact_totals,
        "peak_experimental_observations": copy.deepcopy(raw_peaks),
        "experimental_abort_ceilings": ceilings,
        "terminal_retract": {
            "attempted": True,
            "commanded_retract_m": 0.0003,
            "measured_tcp_prim_retract_m": retract[
                "final_task_z_progress_m"
            ],
            "minimum_task_z_progress_m": retract[
                "minimum_task_z_progress_m"
            ],
            "release_samples": release_samples,
            "release_debounced": True,
            "terminal_state": "TERMINAL_MANIFOLD_CAPTURE",
            "resume_attempted": False,
            "motion_evidence": retract,
            "start_and_target_diagnostics": {
                "first_command_exact": True,
                "measured_state_used_for_command": False,
                "maximum_commanded_tcp_speed_m_s": 0.00035,
            },
        },
        "truth_pose_used_for_touch_control": False,
        "engage_executed": False,
        "insertion_executed": False,
        "twist_executed": False,
        "home_return_executed": False,
        "production_control_authorized": False,
        "hardware_safety_calibration_claimed": False,
        "assembly_success_claimed": False,
    }
    for gate in (
        "frame_count_gate",
        "consecutive_step_gate",
        "frozen_command_gate",
        "exact_contact_manifold_gate",
        "actual_speed_gate",
        "command_speed_gate",
        "outside_entry_gate",
        "cumulative_finite_gate",
        "three_finger_body_contact_gate",
        "applied_action_gate",
        "contact_scope_gate",
        "experimental_abort_envelope_gate",
        "retract_gate",
    ):
        report[gate] = True
    kwargs = {
        "body_root": body,
        "fixed_root": fixed,
        "task_rotation_world": np.eye(3),
        "task_origin_world": origin,
        "physics_dt_s": 1.0 / rate,
        "sample_rate_hz": rate,
        "expected_frame_count": 6,
        "expected_offset_task_xy_m": (0.0006, 0.0),
        "maximum_command_speed_m_s": 0.00035,
        "maximum_measured_speed_m_s": 0.0005,
        "minimum_gap_m": 0.01,
        "expected_preinsert_gap_m": 0.012,
        "required_retract_distance_m": 0.0003,
        "maximum_negative_retract_progress_m": 1.0e-6,
        "expected_release_compression_ceiling_n": 0.1,
        "expected_release_bending_ceiling_nm": 0.004,
        "compressive_axial_force_sign_candidate": 1.0,
        "expected_experimental_abort_ceilings": ceilings,
    }
    assert validate(report, **kwargs)["validated"] is True

    mutations = (
        ("manifold_frames.0.contact.normal", 0.5),
        ("manifold_frames.1.command", 1.0e-3),
        ("manifold_frames.global_step", 100),
        ("approach.tick.readback", 1.0e-3),
        ("approach.first_pre_detached", 1.0e-3),
        ("lateral.speed", 1.0e-3),
        ("approach.contact_total", 7),
        ("approach.per_tick_contact_shift", 1),
        ("approach.peak_inflation", 0.2),
        ("approach.gap_detached", 1.0),
        ("release.compression", 0.2),
        ("release.raw_bending", 0.1),
        ("release.raw_contact", 1),
        ("release.extra_raw_crossing", 1),
        ("release.ceiling", 0.2),
        ("release.distance", 0.0002),
    )
    for name, value in mutations:
        candidate = copy.deepcopy(report)
        if name == "manifold_frames.0.contact.normal":
            candidate["manifold_frames"][0][
                "intended_lip_contact_pairs"
            ][0]["contact_manifold"]["contact_points"][0][
                "normal_world_reported"
            ][0] = value
        elif name == "manifold_frames.1.command":
            candidate["manifold_frames"][1]["command_arm_rad"][0] = value
        elif name == "manifold_frames.global_step":
            for frame in candidate["manifold_frames"]:
                frame["global_step"] += value
        elif name == "approach.tick.readback":
            candidate["approach_and_hold_motion_evidence"][
                "guarded_tick_samples"
            ][0]["applied_post_readback_float32"][0] = value
        elif name == "approach.first_pre_detached":
            evidence = candidate["approach_and_hold_motion_evidence"]
            evidence["guarded_tick_samples"][0][
                "applied_pre_readback_float32"
            ][0] = value
            evidence["guarded_tick_samples"][0][
                "applied_pre_expected_float32"
            ][0] = value
        elif name == "lateral.speed":
            candidate["lateral_motion_evidence"][
                "measured_tcp_prim_world_samples_m"
            ][0][0] = origin[0] + value
        elif name == "approach.contact_total":
            candidate["approach_and_hold_motion_evidence"][
                "contact_record_totals"
            ]["intended_lip"] = value
        elif name == "approach.per_tick_contact_shift":
            ticks = candidate["approach_and_hold_motion_evidence"][
                "guarded_tick_samples"
            ]
            ticks[0]["loose_fixed_contact_records"] -= value
            ticks[1]["loose_fixed_contact_records"] += value
        elif name == "approach.peak_inflation":
            candidate["approach_and_hold_motion_evidence"][
                "peak_experimental_observations"
            ]["absolute_finger_base_torque_nm"] = value
        elif name == "approach.gap_detached":
            evidence = candidate["approach_and_hold_motion_evidence"]
            evidence["estimated_start_gap_m"] += value
            evidence["estimated_gap_samples_m"] = [
                gap + value
                for gap in evidence["estimated_gap_samples_m"]
            ]
            for tick in evidence["guarded_tick_samples"]:
                tick["estimated_gap_m"] += value
            for frame in candidate["manifold_frames"]:
                frame["estimated_gap_m"] += value
        elif name == "release.compression":
            candidate["terminal_retract"]["release_samples"][-1][
                "signed_compression_n"
            ] = value
        elif name == "release.raw_bending":
            evidence = candidate["terminal_retract"]["motion_evidence"]
            for tick in evidence["guarded_tick_samples"][-6:]:
                tick["compensated_wrench_task"][3] = value
            evidence["peak_experimental_observations"][
                "bending_torque_nm"
            ] = value
            candidate["peak_experimental_observations"][
                "bending_torque_nm"
            ] = value
        elif name == "release.raw_contact":
            evidence = candidate["terminal_retract"]["motion_evidence"]
            for tick in evidence["guarded_tick_samples"][-6:]:
                tick["loose_fixed_contact_records"] = 1
                tick["intended_lip_contact_pairs"] = [
                    copy.deepcopy(pair)
                ]
            for key in ("loose_fixed", "intended_lip"):
                evidence["contact_record_totals"][key] = 6
                candidate["contact_record_totals"][key] += 6
        elif name == "release.extra_raw_crossing":
            evidence = candidate["terminal_retract"]["motion_evidence"]
            insert_at = len(evidence["task_z_progress_samples_m"]) - 6
            for field in (
                "measured_tcp_prim_world_samples_m",
                "command_fk_tcp_world_samples_m",
                "task_z_progress_samples_m",
                "command_fk_task_z_progress_samples_m",
                "estimated_gap_samples_m",
            ):
                evidence[field].insert(
                    insert_at, copy.deepcopy(evidence[field][insert_at])
                )
            tick = copy.deepcopy(
                evidence["guarded_tick_samples"][insert_at]
            )
            evidence["guarded_tick_samples"].insert(insert_at, tick)
            for index in range(
                insert_at + 1, len(evidence["guarded_tick_samples"])
            ):
                evidence["guarded_tick_samples"][index]["global_step"] += 1
            for field in (
                "checked_sample_count",
                "finite_sample_count",
                "applied_action_precheck_count",
                "applied_action_postcheck_count",
            ):
                evidence[field] += 1
            for field in (
                "total_checked_sample_count",
                "total_finite_sample_count",
                "applied_action_precheck_count",
                "applied_action_postcheck_count",
            ):
                candidate[field] += 1
        elif name == "release.ceiling":
            candidate["release_compression_ceiling_n"] = value
        else:
            candidate["terminal_retract"]["release_samples"][-1][
                "task_z_progress_m"
            ] = value
        with pytest.raises(ValueError):
            validate(candidate, **kwargs)


def test_tactile_environment_classifier_covers_non_robot_pairs():
    classify = _module()["classify_loose_environment_contact"]
    loose = "/World/Pair/LoosePlug"
    fixed = "/World/Pair/FixedReceptacle"
    fixture = "/World/Pair/Fixture"
    table = "/World/Table"
    assert classify(
        (loose + "/BodyAssembly", fixed + "/EntryShell/Segment_01"),
        loose,
        fixed,
        fixture,
        table,
    ) == "loose_fixed"
    assert classify(
        (loose + "/CouplingNut/Segment_01", fixture + "/Top"),
        loose,
        fixed,
        fixture,
        table,
    ) == "loose_fixture"
    assert classify(
        (table, loose + "/BodyAssembly/RearBody"),
        loose,
        fixed,
        fixture,
        table,
    ) == "loose_table"
    assert classify(
        (fixture, table), loose, fixed, fixture, table
    ) is None


def test_tactile_contact_requires_physical_pair_and_wrench_debounce():
    source = _source()
    for token in (
        "wrench_contact_candidate = contact_candidate(",
        'tactile_pair_class == "intended_segmented_lip"',
        '!= "intended_segmented_lip"',
        '"intended_lip_contact_pairs"',
        '"unexpected_loose_fixed_contact_pairs"',
        '"loose_fixture_contact_pairs"',
        '"loose_table_contact_pairs"',
        "classify_loose_environment_contact(",
        '"loose_fixture",',
        '"loose_table",',
        '"first_forbidden_contact"',
        (
            "signed_compression\n"
            "                                >= contact_on_compression_n"
        ),
        '"contact_window_evidence"',
        '"rejected_physical_contact_samples"',
        '"rejection_reasons"',
        '"signed_compression_below_"',
        '"minimum_required_compression_n"',
        '"raw_wrench": ft_sample["raw_wrench"]',
        '"canonical_wrench_sensor": ft_sample[',
        '"compensated_wrench_sensor": ft_sample[',
        '"compensated_wrench_task": ft_sample[',
        '"contact_pair_evidence"',
        "build_proxy_collision_filter_plan(",
        "apply_proxy_collision_filter(",
        'proxy_collision_filter["enabled"]',
        'proxy_collision_filter["pair_count"]',
        '"filtered_proxy_collision_pair_count"',
    ):
        assert token in source
    # Minimum-jerk derivative peaks at 1.875.  These durations therefore make
    # commanded peak speed equal, not exceed, each contract velocity ceiling.
    assert "1.875\n                        * TACTILE_LIP_OFFSET_M" in source
    assert "1.875\n                        * approach_travel_m" in source
    assert "build_command_continuous_retract_path(" in source
    assert '"peak_command_fk_axial_speed_m_s": path[' in source


def test_tactile_failure_evidence_is_persisted_before_terminal_abort():
    source = _source()
    assert "rejected_physical_contact_samples[-12:]" in source
    assert "no_contact_return" not in source

    no_debounce_start = source.index(
        "if len(contact_window) != ("
    )
    no_debounce_raise = source.index(
        "lip contact did not debounce", no_debounce_start
    )
    no_debounce_section = source[
        no_debounce_start:no_debounce_raise
    ]
    assert "trial.update(" in no_debounce_section
    assert '"last_guarded_approach_sample"' in no_debounce_section
    assert '"peak_experimental_observations"' in no_debounce_section
    assert '"rejected_physical_contact_samples"' in no_debounce_section

    normal_retract_start = source.index(
        "retract_diagnostics.update("
    )
    release_failure = source.index(
        "if not release_gate:", normal_retract_start
    )
    retract_section = source[normal_retract_start:release_failure]
    assert "trial.update(" in retract_section
    assert '"retract_start_and_target_diagnostics"' in retract_section
    assert '"retract_motion_evidence"' in retract_section
    assert '"release_debounced": release_gate' in retract_section


def test_tactile_reversal_freezes_contact_and_zero_steps_hard_failures():
    source = _source()
    for token in (
        "if physical_contact and not drive_frozen:",
        "drive_frozen = True",
        "first intended physical lip record",
        "physical_contact_hold_steps",
        "entry_confirmation_samples",
        "bounded_endpoint_hold_path_indices(",
        "loop_cursor < len(motion_path_indices)",
        "evaluate_release",
        "retract_distance_reached",
        "post_retract_release_ticks",
        "and retract_distance_reached",
        "TactileSafetyStop(",
        "zero_step_abort=True",
        '"world_steps_after_original_failure": 0',
        '"attempted": False',
        "controller.get_applied_action()",
        "compare_applied_arm_command_float32(",
        '"applied_action_position_readback"',
        '"applied_action_float32_equivalent_gate"',
        '"applied_action_maximum_abs_arm_error_float32_rad"',
        '"command_fk_jump_world_m"',
        '"pre_step": pre_state',
        '"post_step": post_state',
        "del tactile_transition_ring[:-12]",
        "contact retract exceeded the GPU preflight",
    ):
        assert token in source

    apply_index = source.index("robot.apply_action(")
    step_index = source.index("world.step(", apply_index)
    wrench_index = source.index(
        "robot.get_measured_joint_forces(", step_index
    )
    assert apply_index < step_index < wrench_index
    pre_readback_gate = source.index(
        'if not pre_state[\n'
        '                            "applied_action_float32_equivalent_gate"'
    )
    tactile_observe = source.index(
        "observe_and_step(", pre_readback_gate
    )
    assert pre_readback_gate < tactile_observe

    contact_loss = source.index("if len(body_contact_fingers) != 3:")
    contact_loss_end = source.index(
        "return positions, velocities", contact_loss
    )
    contact_loss_section = source[contact_loss:contact_loss_end]
    assert "raise TactileSafetyStop(" in contact_loss_section
    assert "contact_loss_reason, zero_step_abort=True" in contact_loss_section


def test_tactile_source_mutations_break_zero_step_and_readback_contracts():
    source = _source()

    def assert_contract(candidate):
        contact_loss = candidate.index(
            "if len(body_contact_fingers) != 3:"
        )
        contact_loss_end = candidate.index(
            "return positions, velocities", contact_loss
        )
        contact_section = candidate[contact_loss:contact_loss_end]
        assert "raise TactileSafetyStop(" in contact_section
        assert (
            "contact_loss_reason, zero_step_abort=True" in contact_section
        )
        pre_gate = candidate.index(
            'if not pre_state[\n'
            '                            "applied_action_float32_'
            'equivalent_gate"'
        )
        observe = candidate.index("observe_and_step(", pre_gate)
        assert pre_gate < observe
        assert (
            "applied_gate = compare_applied_arm_command_float32("
            in candidate
        )

    assert_contract(source)
    mutations = (
        (
            "contact_loss_reason, zero_step_abort=True",
            "contact_loss_reason, zero_step_abort=False",
        ),
        (
            "applied_gate = compare_applied_arm_command_float32(",
            "disabled_applied_command_gate(",
        ),
        (
            'if not pre_state[\n'
            '                            "applied_action_float32_'
            'equivalent_gate"',
            'if False and not pre_state[\n'
            '                            "applied_action_float32_'
            'equivalent_gate"',
        ),
    )
    for original, replacement in mutations:
        mutated = source.replace(original, replacement, 1)
        with pytest.raises((AssertionError, ValueError)):
            assert_contract(mutated)


def test_retract_preflight_is_independent_no_contact_and_required_by_lip():
    source = _source()
    for token in (
        "TACTILE_RETRACT_PREFLIGHT_COMMAND_DESCENT_M = 0.00045",
        "TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S = 0.00035",
        "TACTILE_RETRACT_PREFLIGHT_DESCENT_M = 0.0005",
        "minimum_jerk_steps_for_peak_speed(",
        '"--tactile-retract-preflight"',
        '"--tactile-retract-preflight-report"',
        "PASSED_NO_CONTACT_COMMAND_REVERSAL_",
        '"lip_contact_executed": False',
        '"touch_trials_executed": 0',
        '"observed_negative_reversal_"',
        '"progress_bound_m"',
        "TactilePreflightComplete",
        "if arguments.tactile_lip_calibration:",
        "validate_tactile_retract_preflight_report(",
        "expected_visual_config_sha256=_sha256(config_path)",
        "expected_preinsert_config_sha256=_sha256(",
        "expected_runtime_source_import_sha256=(",
        "expected_runtime_source_start_sha256=(",
        "expected_trial_id=probe.trial_id",
        '"runtime_source_finalize_sha256": (',
        '"runtime_source_unchanged": runtime_source_unchanged',
        '"measured_tcp_prim_world_samples_m"',
        '"estimated_gap_samples_m"',
        '"all_trajectory_samples_outside_entry_gate"',
        '"measured_descent_bound_gate"',
        '"reversal_negative_progress_bound_gate"',
        '"all_phase_measured_speed_gates"',
        '"all_phase_command_speed_gates"',
        '"peak_experimental_observations"',
        '"experimental_abort_ceilings"',
        '"minimum_body_contact_finger_count"',
        '"applied_action_precheck_count"',
        '"applied_action_postcheck_count"',
        '"contact_record_totals"',
        '"all_samples_finite_gate"',
        "and all_trajectory_samples_outside_entry_gate",
        "and measured_descent_bound_gate",
        "and reversal_negative_progress_bound_gate",
        "and all_phase_measured_speed_gates",
        "and all_phase_command_speed_gates",
        "and experimental_abort_envelope_gate",
    ):
        assert token in source
    preflight = source.index("if arguments.tactile_retract_preflight:")
    lip_loop = source.index("TACTILE_LIP_DIRECTIONS", preflight)
    assert preflight < lip_loop


def test_tactile_failures_use_distinct_result_marker_and_fail_closed_report():
    source = _source()
    assert "result_marker = TACTILE_LIP_RESULT_MARKER" in source
    assert '"status": "FAILED_RUNTIME_FAIL_CLOSED"' in source
    assert '"failure_phase": locals().get("phase")' in source
    assert '"failure_global_step": locals().get("global_step")' in source
    assert "tactile_failure_phase = locals().get(\"phase\")" in source
    assert "tactile_failure_step = locals().get(\"global_step\")" in source
    assert "tactile_abort_report = tactile_abort_retract()" in source
    assert '"abort_retract": tactile_abort_report' in source
    assert '"terminal_state": "TERMINAL_ABORT"' in source
    assert '"resume_attempted": False' in source
    assert "command_continuous_retract_plan(" in source
    assert "measured_arm_and_retract_target(" not in source
    assert '"measured_state_used_for_command": False' in source
    assert '"first_command_exact": path["first_command_exact"]' in source
    assert '"measured_q7_tracking_error_rad"' in source
    assert '"command_vs_measured_max_abs_rad"' in source
    assert '"measured_fk_to_tcp_prim_error_m"' in source
    assert '"requested_command_target_tcp_world_m"' in source
    assert '"target_command_fk_tcp_world_m"' in source
    assert '"measured_tcp_prim_retract_m"' in source
    assert '"measured_robot_fk_retract_m_deprecated": True' in source
    assert '"minimum_task_z_progress_m"' in source
    assert '"release_samples"' in source
    assert '"task_frame_axes_world"' in source
    assert '"determinant": float(np.linalg.det(task_rotation))' in source
    assert "simulation_app.close(exit_code=0 if passed else 1)" in source


def test_tactile_pair_statistics_and_filter_scope_are_fail_closed():
    source = _source()
    assert (
        "contact_debounce_samples: 6"
        in (
            Path(__file__).resolve().parents[1]
            / "config/d38999_tactile_engage_probe_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert "np.std(inferred_lever_samples_m, ddof=1)" in source
    assert "signed_compression_samples" in source
    assert '"minimum_debounced_compression_n"' in source
    assert '"inferred_lever_response_samples_m"' in source
    assert 'trial["passed"] = bool(' in source
    assert 'trial["touch_passed"] and pair_passed' in source
    assert "len(trials) == len(TACTILE_LIP_DIRECTIONS)" in source
    assert 'all(trial["passed"] for trial in trials)' in source
    # The 500-pair plan is narrow: 24x20 nut/entry pairs plus only the 20
    # same-index mating/entry pairs. Cross-angle mating lip pairs remain live.
    assert (
        'proxy_collision_filter["pair_count"]\n'
        '                != tactile_probe.proxy_boundaries['
    ) in source
    assert '"filtered_proxy_collision_pair_count"' in source


def test_loose_fixed_pair_classifier_is_order_independent():
    classify = _module()["contact_pair_crosses_prim_roots"]
    loose = "/World/Pair/LoosePlug"
    fixed = "/World/Pair/FixedReceptacle"
    paths = (
        fixed,
        loose + "/BodyAssembly",
        fixed + "/Entry/Segment_00",
        loose + "/BodyAssembly/Mating/Segment_00",
    )
    assert classify(paths, loose, fixed) is True
    assert classify(tuple(reversed(paths)), loose, fixed) is True
    assert classify((loose, loose + "/BodyAssembly"), loose, fixed) is False


def test_non_nominal_scene_is_authored_once_before_physics():
    source = _source()
    main = source.split("def main():", 1)[1]
    assert main.index("_scene_for_probe(") < main.index(
        "author_d38999_tabletop_scene("
    )
    assert main.index("author_d38999_tabletop_scene(") < main.index(
        "world.reset()"
    )
    after_reset = main.split("world.reset()", 1)[1]
    assert "_scene_for_probe(" not in after_reset
    assert "author_d38999_tabletop_scene(" not in after_reset
    assert ".set_world_pose(" not in source
    assert ".set_local_pose(" not in source
    assert '"object_pose_writes_after_physics": 0' in source


def test_same_world_rgbd_drives_adapter_plan_before_truth_evaluation():
    source = _source()
    capture_index = source.index("capture_d38999_rgbd_runtime(")
    side_effect_index = source.index(
        "runtime_side_effects = validate_rgbd_capture_side_effects("
    )
    sample_index = source.index("pose_provider_sample_from_rgbd_metrics(")
    plan_index = source.index("build_visual_xy_pick_plan(")
    truth_index = source.index("evaluate_visual_xy_truth_only(")
    motion_index = source.index('phase = "home_hand_open"')
    assert (
        capture_index
        < side_effect_index
        < sample_index
        < plan_index
        < truth_index
        < motion_index
    )
    assert "capture.passed" not in source
    truth_call = source[truth_index:motion_index]
    assert "capture.loose_position_world_m" in truth_call
    assert "capture.fixed_position_world_m" in truth_call
    plan_call = source[plan_index:truth_index]
    assert "capture.loose_position_world_m" not in plan_call
    assert "capture.fixed_position_world_m" not in plan_call
    assert '"truth_xy_used_for_target": False' in source


def test_capture_side_effect_gate_accepts_only_healthy_runtime_lifecycle():
    validate = _module()["validate_rgbd_capture_side_effects"]
    metrics = _healthy_capture_side_effects()
    # Poison the aggregate and endpoint truth/error gates.  The lifecycle gate
    # must neither require, interpret, nor reconstruct any of them.
    metrics.update(
        {
            "passed": False,
            "loose_plug": {
                "passed": False,
                "registered_truth_xy_m": object(),
                "xy_error_m": object(),
            },
            "fixed_receptacle": {
                "passed": False,
                "registered_truth_xy_m": object(),
                "xy_error_m": object(),
            },
        }
    )
    result = validate(metrics)
    assert result == {
        "world_reset_or_clear_calls": 0,
        "endpoint_pose_writes_after_physics": 0,
        "resource_cleanup_verified": True,
        "timeline_state_restored": True,
        "playing_before_capture": True,
        "playing_after_restore": True,
        "truth_or_error_gate_consulted": False,
    }


def test_capture_side_effect_gate_fails_closed_on_each_unsafe_effect():
    validate = _module()["validate_rgbd_capture_side_effects"]
    mutations = (
        ("world_reset_or_clear_calls", 1),
        ("object_pose_writes_after_start", 1),
        ("resource_cleanup.resources_released", False),
        ("resource_cleanup.world_reset", True),
        ("resource_cleanup.stage_prims_removed", 1),
        ("timeline_state.restored", False),
        ("timeline_state.playing_after_restore", False),
    )
    for dotted, value in mutations:
        metrics = copy.deepcopy(_healthy_capture_side_effects())
        target = metrics
        names = dotted.split(".")
        for name in names[:-1]:
            target = target[name]
        target[names[-1]] = value
        try:
            validate(metrics)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe side effect accepted: {dotted}")


def test_capture_side_effect_gate_rejects_missing_extra_and_loose_types():
    validate = _module()["validate_rgbd_capture_side_effects"]
    cases = []
    missing_root = _healthy_capture_side_effects()
    del missing_root["object_pose_writes_after_start"]
    cases.append(missing_root)
    missing = _healthy_capture_side_effects()
    del missing["timeline_state"]["restored"]
    cases.append(missing)
    extra = _healthy_capture_side_effects()
    extra["resource_cleanup"]["unknown"] = False
    cases.append(extra)
    loose_counter = _healthy_capture_side_effects()
    loose_counter["world_reset_or_clear_calls"] = False
    cases.append(loose_counter)
    loose_timeline = _healthy_capture_side_effects()
    loose_timeline["timeline_state"]["restored"] = 1
    cases.append(loose_timeline)
    for metrics in cases:
        try:
            validate(metrics)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed lifecycle metrics were accepted")


def test_output_directory_must_be_new_and_is_created_atomically(tmp_path):
    create = _module()["create_exclusive_output_directory"]
    target = tmp_path / "new" / "run-001"
    assert create(target) == target
    assert target.is_dir()
    (target / "existing-report.json").write_text("old", encoding="utf-8")
    try:
        create(target)
    except FileExistsError:
        pass
    else:
        raise AssertionError(
            "an existing nonempty output directory was reused"
        )


def test_registered_orientation_and_partial_pose_scope_are_explicit():
    source = _source()
    for token in (
        '"orientation_source": "registered_nominal"',
        '"uses_truth_position": False',
        '"uses_truth_orientation": False',
        '"full_6d": False',
        '"control_authorized": False',
        '"production_control_authorized": False',
        '"collision_planned": False',
    ):
        assert token in source


def test_home_pick_lift_hold_and_nominal_safety_gates_are_present():
    source = _source()
    phases = (
        "home_hand_open",
        "visual_xy_high_approach_to_pregrasp",
        "visual_xy_open_hand_descent",
        "visual_xy_physical_hand_closure",
        "visual_xy_closed_hand_seating",
        "visual_xy_grip_lift",
        "visual_xy_unsupported_hold",
    )
    assert all(phase in source for phase in phases)
    for token in (
        "maximum_absolute_torque_delta_nm",
        "minimum_loaded_channels",
        "maximum_body_tcp_slip_m",
        "maximum_body_nut_separation_change_m",
        "minimum_body_lift_m",
        "minimum_final_bottom_clearance_m",
        "maximum_final_hold_displacement_m",
        "maximum_observed_joint_speed_rad_s",
        "maximum_joint_limit_violation_rad",
        "maximum_grasp_tcp_position_error_m",
        "maximum_grasp_tcp_axis_error_rad",
        "zero_forbidden_contacts",
        "_all_fingers_have_body_contact",
        "non-finite",
    ):
        assert token in source
    assert "2 Nm finger torque hard stop exceeded" in source
    assert "sample_torque=True" in source


def test_script_has_no_top_level_isaac_or_gpu_imports():
    tree = ast.parse(_source())
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {"isaacsim", "omni", "pxr", "numpy", "PIL", "torch", "rclpy"}
    )


def test_simulation_app_close_receives_fail_closed_exit_code():
    tree = ast.parse(_source())
    close_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "simulation_app"
        and node.func.attr == "close"
    ]
    assert len(close_calls) == 1
    keywords = {item.arg: item.value for item in close_calls[0].keywords}
    exit_code = keywords["exit_code"]
    assert isinstance(exit_code, ast.IfExp)
    assert isinstance(exit_code.test, ast.Name)
    assert exit_code.test.id == "passed"
    assert ast.literal_eval(exit_code.body) == 0
    assert ast.literal_eval(exit_code.orelse) == 1
