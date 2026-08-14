from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kcg_connector.d38999_wrist_ft_guarded_insertion import (
    GuardedInsertionPhase,
    initial_guarded_insertion_state,
    load_guarded_insertion_contract,
    parse_guarded_insertion_observation,
    step_guarded_insertion,
    verify_guarded_insertion_inputs,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = REPOSITORY / "src/kcg_connector/config/" \
    "d38999_wrist_ft_guarded_insertion_v1.yaml"
ISAAC_RUNNER = REPOSITORY / (
    "src/kcg_connector/isaac/d38999_visual_xy_pick_smoke.py"
)


def _observation(**updates):
    value = {
        "timestamp_s": 1.0,
        "sample_age_s": 0.001,
        "compensated_wrench_task": [0.0] * 6,
        "measured_tcp_position_task_m": [0.0, 0.0, 0.012],
        "commanded_tcp_position_task_m": [0.0, 0.0, 0.012],
        "arm_tracking_error_rad": 0.001,
        "maximum_joint_speed_rad_s": 0.1,
        "gripper_position_drift_from_preinsert_rad": 0.001,
        "robot_state_finite": True,
        "vision_preinsert_id": "capture-1:fixed-1",
    }
    value.update(updates)
    return parse_guarded_insertion_observation(value)


def test_contract_is_disabled_and_truth_free():
    contract = load_guarded_insertion_contract(CONFIG)
    assert contract.enabled is False
    assert contract.status == (
        "runtime_integrated_cpu_static_ready_gpu_not_validated"
    )
    verify_guarded_insertion_inputs(contract, REPOSITORY)


@pytest.mark.parametrize(
    "forbidden",
    [
        "physx_contact_report",
        "physx_contact_manifold",
        "collider_path",
        "contact_normal",
        "simulator_object_truth_pose",
        "simulator_truth_gap",
    ],
)
def test_observation_rejects_simulator_contact_and_truth(forbidden):
    value = {
        "timestamp_s": 1.0,
        "sample_age_s": 0.001,
        "compensated_wrench_task": [0.0] * 6,
        "measured_tcp_position_task_m": [0.0, 0.0, 0.012],
        "commanded_tcp_position_task_m": [0.0, 0.0, 0.012],
        "arm_tracking_error_rad": 0.001,
        "maximum_joint_speed_rad_s": 0.1,
        "gripper_position_drift_from_preinsert_rad": 0.001,
        "robot_state_finite": True,
        "vision_preinsert_id": "capture-1:fixed-1",
        forbidden: object(),
    }
    with pytest.raises(ValueError, match="extra"):
        parse_guarded_insertion_observation(value)


def test_preinsert_starts_and_free_approach_moves_only_down_task_z():
    contract = load_guarded_insertion_contract(CONFIG)
    observation = _observation()
    state = initial_guarded_insertion_state(observation)
    command = step_guarded_insertion(contract, state, observation)
    assert command.status == "START_GUARDED_APPROACH"
    command = step_guarded_insertion(
        contract, command.next_state, observation
    )
    assert command.next_state.phase is GuardedInsertionPhase.GUARDED_APPROACH
    assert command.delta_tcp_task_m[:2] == (0.0, 0.0)
    assert command.delta_tcp_task_m[2] < 0.0


def test_wrist_force_generates_bounded_xy_correction_without_contact_truth():
    contract = load_guarded_insertion_contract(CONFIG)
    preinsert = _observation()
    observation = _observation(
        compensated_wrench_task=[-0.8, 0.4, 0.1, 0.0, 0.0, 0.0],
        measured_tcp_position_task_m=[0.0, 0.0, 0.009],
    )
    state = replace(
        initial_guarded_insertion_state(preinsert),
        phase=GuardedInsertionPhase.INSERT,
    )
    command = step_guarded_insertion(contract, state, observation)
    assert command.status == "WRIST_FT_XY_CORRECTION"
    assert command.delta_tcp_task_m[0] < 0.0
    assert command.delta_tcp_task_m[1] > 0.0
    assert command.delta_tcp_task_m[2] == 0.0
    assert sum(v * v for v in command.delta_tcp_task_m[:2]) ** 0.5 <= (
        contract.maximum_xy_correction_step_m
    )


def test_axial_contact_before_entry_freezes_then_retracts_and_corrects():
    contract = load_guarded_insertion_contract(CONFIG)
    observation = _observation(
        compensated_wrench_task=[0.4, -0.2, 0.3, 0.0, 0.0, 0.0]
    )
    state = replace(
        initial_guarded_insertion_state(observation),
        phase=GuardedInsertionPhase.GUARDED_APPROACH,
    )
    command = step_guarded_insertion(contract, state, observation)
    assert command.stop_motion is False
    assert command.status == "WRIST_FT_EARLY_CONTACT_FREEZE"
    assert command.delta_tcp_task_m == (0.0, 0.0, 0.0)
    assert command.next_state.phase is GuardedInsertionPhase.CONTACT_RETRACT
    assert command.next_state.contact_retry_count == 1

    unloaded = _observation(
        measured_tcp_position_task_m=[0.0, 0.0, 0.012]
    )
    command = step_guarded_insertion(contract, command.next_state, unloaded)
    assert command.status == "WRIST_FT_CONTACT_RETRACT"
    assert command.delta_tcp_task_m[2] > 0.0

    target_z = command.next_state.contact_retract_target_z_task_m
    retracted = _observation(
        measured_tcp_position_task_m=[0.0, 0.0, target_z]
    )
    command = step_guarded_insertion(
        contract, command.next_state, retracted
    )
    assert command.status == "WRIST_FT_POST_RETRACT_XY_CORRECTION"
    assert command.delta_tcp_task_m[0] > 0.0
    assert command.delta_tcp_task_m[1] < 0.0
    assert command.delta_tcp_task_m[2] == 0.0


def test_early_axial_contact_without_lateral_direction_fails_closed():
    contract = load_guarded_insertion_contract(CONFIG)
    observation = _observation(
        compensated_wrench_task=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
    )
    state = replace(
        initial_guarded_insertion_state(observation),
        phase=GuardedInsertionPhase.GUARDED_APPROACH,
    )
    command = step_guarded_insertion(contract, state, observation)
    assert command.stop_motion is True
    assert command.next_state.abort_reason == (
        "early_axial_contact_no_lateral_direction"
    )


def test_preentry_lateral_signal_retracts_before_any_xy_motion():
    contract = load_guarded_insertion_contract(CONFIG)
    observation = _observation(
        compensated_wrench_task=[0.12, -0.11, 0.05, 0.02, 0.0, 0.0]
    )
    state = replace(
        initial_guarded_insertion_state(observation),
        phase=GuardedInsertionPhase.GUARDED_APPROACH,
    )
    command = step_guarded_insertion(contract, state, observation)
    assert command.status == "WRIST_FT_EARLY_CONTACT_FREEZE"
    assert command.delta_tcp_task_m == (0.0, 0.0, 0.0)
    assert command.next_state.phase is GuardedInsertionPhase.CONTACT_RETRACT
    assert command.next_state.pending_xy_correction_task_m != (0.0, 0.0)


@pytest.mark.parametrize(
    ("wrench", "reason"),
    [
        ([0.0, 0.0, 5.1, 0.0, 0.0, 0.0], "axial_force"),
        ([2.1, 0.0, 0.0, 0.0, 0.0, 0.0], "lateral_force"),
        ([0.0, 0.0, 0.0, 0.181, 0.0, 0.0], "bending_torque"),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.051], "tightening_torque"),
    ],
)
def test_experimental_abort_envelope_stops_without_another_command(wrench, reason):
    contract = load_guarded_insertion_contract(CONFIG)
    observation = _observation(compensated_wrench_task=wrench)
    state = replace(
        initial_guarded_insertion_state(observation),
        phase=GuardedInsertionPhase.INSERT,
    )
    command = step_guarded_insertion(contract, state, observation)
    assert command.stop_motion is True
    assert command.delta_tcp_task_m == (0.0, 0.0, 0.0)
    assert command.next_state.phase is GuardedInsertionPhase.ABORT
    assert command.next_state.abort_reason == reason


def test_measured_travel_selects_insert_then_hold_and_complete():
    contract = load_guarded_insertion_contract(CONFIG)
    preinsert = _observation()
    state = replace(
        initial_guarded_insertion_state(preinsert),
        phase=GuardedInsertionPhase.GUARDED_APPROACH,
    )
    inserted = _observation(measured_tcp_position_task_m=[0.0, 0.0, 0.003])
    command = step_guarded_insertion(contract, state, inserted)
    assert command.status == "ENTER_HOLD"
    state = replace(command.next_state, hold_count=contract.hold_steps - 1)
    command = step_guarded_insertion(contract, state, inserted)
    assert command.status == "COMPLETE"
    assert command.next_state.phase is GuardedInsertionPhase.COMPLETE
    assert command.stop_motion is True


def test_stale_wrench_and_nonfinite_robot_state_fail_closed():
    contract = load_guarded_insertion_contract(CONFIG)
    for observation, reason in (
        (_observation(sample_age_s=0.021), "wrench_sample_stale"),
        (_observation(robot_state_finite=False), "robot_state_nonfinite"),
    ):
        state = initial_guarded_insertion_state(observation)
        command = step_guarded_insertion(contract, state, observation)
        assert command.next_state.phase is GuardedInsertionPhase.ABORT
        assert command.next_state.abort_reason == reason


def test_gripper_drift_is_relative_to_preinsert_and_fails_closed():
    contract = load_guarded_insertion_contract(CONFIG)
    observation = _observation(
        gripper_position_drift_from_preinsert_rad=0.021
    )
    state = initial_guarded_insertion_state(observation)
    command = step_guarded_insertion(contract, state, observation)
    assert command.next_state.phase is GuardedInsertionPhase.ABORT
    assert command.next_state.abort_reason == (
        "gripper_position_drift_from_preinsert"
    )


def test_absolute_gripper_command_error_is_not_an_allowed_observation():
    value = {
        "timestamp_s": 1.0,
        "sample_age_s": 0.001,
        "compensated_wrench_task": [0.0] * 6,
        "measured_tcp_position_task_m": [0.0, 0.0, 0.012],
        "commanded_tcp_position_task_m": [0.0, 0.0, 0.012],
        "arm_tracking_error_rad": 0.001,
        "maximum_joint_speed_rad_s": 0.1,
        "gripper_position_error_rad": 0.001,
        "robot_state_finite": True,
        "vision_preinsert_id": "capture-1:fixed-1",
    }
    with pytest.raises(ValueError, match="keys are invalid"):
        parse_guarded_insertion_observation(value)


def test_maximum_control_steps_fails_closed():
    contract = load_guarded_insertion_contract(CONFIG)
    observation = _observation()
    state = replace(
        initial_guarded_insertion_state(observation),
        step_count=contract.maximum_control_steps,
    )
    command = step_guarded_insertion(contract, state, observation)
    assert command.stop_motion is True
    assert command.next_state.abort_reason == "maximum_control_steps"


def test_isaac_runtime_branch_excludes_physx_contact_truth():
    source = ISAAC_RUNNER.read_text(encoding="utf-8")
    assert '"--wrist-ft-guarded-insertion"' in source
    assert "retired: the hand has no fingertip tactile sensor" in source
    start = source.index(
        "if wrist_ft_guarded_contract is not None:",
        source.index("passed = bool(passed and preinsert_passed)"),
    )
    end = source.index("if tactile_probe is not None:", start)
    branch = source[start:end]
    for required in (
        "def wrist_ft_only_step(",
        "step_guarded_insertion(",
        "parse_guarded_insertion_observation(",
        '"physx_contact_queries_during_control": 0',
        '"fingertip_tactile_sensor_used": False',
        '"physx_contact_truth_used_for_control": False',
    ):
        assert required in branch
    for forbidden in (
        "get_full_contact_report",
        "contact_snapshot(",
        "loose_fixed_contact_count(",
        "classify_tactile_lip_contact_pair(",
        "body.get_world_pose(",
        "fixed_prim",
    ):
        assert forbidden not in branch
