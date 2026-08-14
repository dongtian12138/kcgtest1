import math
from dataclasses import fields

import numpy as np
import pytest

from kcg_connector.compliant_insertion import (
    ControllerState, InsertionObservation, InsertionState,
    _probe_command,
    effective_lateral_posthoc, full_seated_posthoc,
    load_compliant_insertion_config, response_model_twist,
    step_compliant_insertion,
)


def test_full_seated_posthoc_requires_depth_and_guided_pose():
    assert full_seated_posthoc(0.00900, 0.00900, True)
    assert full_seated_posthoc(0.00891, 0.00900, True)
    assert not full_seated_posthoc(0.00889, 0.00900, True)
    assert not full_seated_posthoc(0.00910, 0.00900, False)
    assert not full_seated_posthoc(math.inf, 0.00900, True)


def test_effective_lateral_combines_position_and_axis_error():
    value = effective_lateral_posthoc(0.00010, math.radians(2.0), 0.012)
    assert value == pytest.approx(
        0.00010 + 0.012 * math.tan(math.radians(2.0))
    )
    assert math.isinf(effective_lateral_posthoc(-0.1, 0.0, 0.012))


def test_control_observation_truth_firewall():
    names = {item.name for item in fields(InsertionObservation)}
    assert names == {
        "timestamp_s",
        "sample_age_s",
        "wrench_assembly",
        "tcp_position_assembly_m",
        "tcp_rotation_vector_assembly_rad",
        "vision_control_authorized",
        "synchronized_capture",
        "ft_valid",
        "ft_tared",
        "payload_compensated",
    }
    assert not names & {
        "object_truth",
        "contact_normal",
        "contact_point",
        "collider_identity",
        "penetration_depth",
        "posthoc_seated",
    }


def test_four_dimensional_probe_can_select_negative_rx():
    config = load_compliant_insertion_config()
    state = ControllerState(
        phase=InsertionState.ACTIVE_PROBE,
        probe_baseline_score=0.20,
    )
    position = np.zeros(3)
    rotation = np.zeros(3)
    angular_command_seen = False
    for _ in range(6000):
        # A deliberately coupled-free synthetic contact: +Rx increases the
        # measured cost and -Rx decreases it.  No pose truth enters the probe.
        wrench = np.zeros(6)
        wrench[2] = 0.20 + 100.0 * rotation[0]
        state, command, _ = _probe_command(
            config, state, position, rotation, wrench
        )
        position += command[:3] / config["control_rate_hz"]
        rotation += command[3:] / config["control_rate_hz"]
        angular_command_seen |= bool(np.linalg.norm(command[3:5]) > 0.0)
        if state.phase is not InsertionState.ACTIVE_PROBE:
            break
    assert angular_command_seen
    assert state.phase is InsertionState.CONTACT_UNLOAD
    assert state.probe_selected_xy == (0.0, 0.0)
    assert state.probe_selected_tilt == (-1.0, 0.0)


def test_unambiguous_first_probe_pair_selects_without_cross_axis_motion():
    config = load_compliant_insertion_config()
    state = ControllerState(
        phase=InsertionState.ACTIVE_PROBE,
        probe_baseline_score=0.20,
    )
    position = np.zeros(3)
    rotation = np.zeros(3)
    visited_angular_command = False
    for _ in range(2000):
        wrench = np.zeros(6)
        # +X increases cost and -X strongly decreases cost.
        wrench[2] = 0.20 + 8000.0 * position[0]
        state, command, status = _probe_command(
            config, state, position, rotation, wrench
        )
        position += command[:3] / config["control_rate_hz"]
        rotation += command[3:] / config["control_rate_hz"]
        visited_angular_command |= bool(np.linalg.norm(command[3:5]) > 0.0)
        if state.phase is not InsertionState.ACTIVE_PROBE:
            break
    assert status == "PROBE_EARLY_PAIR_SELECTED"
    assert state.phase is InsertionState.CONTACT_UNLOAD
    assert state.probe_selected_xy == (-1.0, 0.0)
    assert not visited_angular_command


def test_probe_returns_before_formal_soft_gate():
    config = load_compliant_insertion_config()
    state = ControllerState(
        phase=InsertionState.ACTIVE_PROBE,
        probe_origin_xy_m=(0.0, 0.0),
        probe_origin_tilt_rad=(0.0, 0.0),
        probe_baseline_score=0.10,
    )
    wrench = np.zeros(6)
    wrench[0] = 1.20  # below 1.4 N soft gate, above 80% early return
    next_state, command, status = _probe_command(
        config,
        state,
        np.asarray((0.00001, 0.0, 0.0)),
        np.zeros(3),
        wrench,
    )
    assert status == "ACTIVE_PROBE_EARLY_RETURN"
    assert next_state.probe_leg == 1
    assert command[0] < 0.0


def test_recontact_reuses_measured_tilt_direction_without_reversing():
    config = load_compliant_insertion_config()
    state = ControllerState(
        phase=InsertionState.CONTACT_CLASSIFY,
        step_count=100,
        contact_realign_count=1,
        retained_probe_tilt=(-1.0, 0.0),
        cumulative_unloaded_tilt_rad=math.radians(0.40),
        latched_contact_class="SINGLE_EDGE_CONTACT",
        latched_contact_score=0.20,
        latched_contact_wrench=(0.0, 0.1, -0.1, 0.01, 0.0, 0.0),
    )
    command = step_compliant_insertion(
        config,
        state,
        observation(100, (0.0, 0.0, 0.0), (0.0,) * 6),
    )
    assert command.status == "REUSE_MEASURED_TILT_DIRECTION"
    assert command.next_state.phase is InsertionState.CONTACT_UNLOAD
    assert command.next_state.probe_selected_tilt == (-1.0, 0.0)


def test_recontact_reuses_measured_xy_without_switching_to_tilt():
    config = load_compliant_insertion_config()
    state = ControllerState(
        phase=InsertionState.CONTACT_CLASSIFY,
        step_count=100,
        contact_realign_count=1,
        retained_probe_xy=(-1.0, 0.0),
        latched_contact_class="SINGLE_EDGE_CONTACT",
        latched_contact_score=0.20,
        latched_contact_wrench=(0.1, 0.0, -0.1, 0.0, 0.01, 0.0),
    )
    command = step_compliant_insertion(
        config,
        state,
        observation(100, (0.00006, 0.0, 0.0), (0.0,) * 6),
    )
    assert command.status == "REUSE_MEASURED_XY_DIRECTION"
    assert command.next_state.phase is InsertionState.CONTACT_UNLOAD
    assert command.next_state.probe_selected_xy == (-1.0, 0.0)
    assert command.next_state.probe_selected_tilt == (0.0, 0.0)


def test_recontact_fails_closed_after_total_tilt_budget_is_used():
    config = load_compliant_insertion_config()
    state = ControllerState(
        phase=InsertionState.CONTACT_CLASSIFY,
        step_count=100,
        contact_realign_count=1,
        retained_probe_tilt=(-1.0, 0.0),
        cumulative_unloaded_tilt_rad=float(
            config["motion"]["maximum_search_angle_rad"]
        ),
        latched_contact_class="SINGLE_EDGE_CONTACT",
        latched_contact_score=0.20,
        latched_contact_wrench=(0.0, 0.1, -0.1, 0.01, 0.0, 0.0),
    )
    command = step_compliant_insertion(
        config,
        state,
        observation(100, (0.0, 0.0, 0.0), (0.0,) * 6),
    )
    assert command.status == "PERSISTENT_TILT_BUDGET_EXHAUSTED_BACKOFF"
    assert command.next_state.phase is InsertionState.BACKOFF
    assert command.twist_assembly[2] < 0.0


def test_recontact_budget_counts_current_fk_probe_residual():
    config = load_compliant_insertion_config()
    maximum = float(config["motion"]["maximum_search_angle_rad"])
    state = ControllerState(
        phase=InsertionState.CONTACT_CLASSIFY,
        step_count=100,
        contact_realign_count=1,
        retained_probe_tilt=(-1.0, 0.0),
        cumulative_unloaded_tilt_rad=math.radians(0.39),
        latched_contact_class="SINGLE_EDGE_CONTACT",
        latched_contact_score=0.20,
        latched_contact_wrench=(0.0, 0.1, -0.1, 0.01, 0.0, 0.0),
    )
    sample = observation(100, (0.0, 0.0, 0.0), (0.0,) * 6)
    sample = InsertionObservation(
        **{
            **sample.__dict__,
            "tcp_rotation_vector_assembly_rad": (maximum, 0.0, 0.0),
        }
    )
    command = step_compliant_insertion(config, state, sample)
    assert command.status == "PERSISTENT_TILT_BUDGET_EXHAUSTED_BACKOFF"
    assert command.next_state.phase is InsertionState.BACKOFF


def test_measured_response_model_corrects_positive_and_negative_x():
    config = load_compliant_insertion_config()
    matrix = np.asarray(
        config["local_response_model"]["response_matrix_wrench_per_motion"]
    )
    positive_x_wrench = np.zeros(6)
    positive_x_wrench[[0, 1, 3, 4]] = matrix[:, 0] * 0.00002
    assert response_model_twist(config, positive_x_wrench)[0] < 0.0
    assert response_model_twist(config, -positive_x_wrench)[0] > 0.0


def test_measured_response_model_corrects_positive_rx():
    config = load_compliant_insertion_config()
    matrix = np.asarray(
        config["local_response_model"]["response_matrix_wrench_per_motion"]
    )
    positive_rx_wrench = np.zeros(6)
    positive_rx_wrench[[0, 1, 3, 4]] = matrix[:, 2] * 0.0003490658504
    assert response_model_twist(config, positive_rx_wrench)[3] < 0.0


def observation(step, position, wrench, *, authorized=True):
    return InsertionObservation(
        timestamp_s=step / 240.0,
        sample_age_s=0.0,
        wrench_assembly=tuple(wrench),
        tcp_position_assembly_m=tuple(position),
        tcp_rotation_vector_assembly_rad=(0.0, 0.0, 0.0),
        vision_control_authorized=authorized,
        synchronized_capture=True,
        ft_valid=True,
        ft_tared=True,
        payload_compensated=True,
    )


def test_vision_gate_fails_closed_before_motion():
    config = load_compliant_insertion_config()
    command = step_compliant_insertion(
        config, ControllerState(),
        observation(0, (0.0, 0.0, 0.0), (0.0,) * 6, authorized=False),
    )
    assert command.next_state.phase is InsertionState.REOBSERVE
    assert command.stop_motion
    assert command.twist_assembly == (0.0,) * 6


def test_axial_edge_contact_runs_bounded_symmetric_probe_and_backoff():
    config = load_compliant_insertion_config()
    state = ControllerState()
    position = [0.0, 0.0, 0.0]
    visited = []
    for step in range(5000):
        wrench = (0.0,) * 6 if step < 20 else (0.20, -0.10, 0.40, 0.010, -0.004, 0.0)
        command = step_compliant_insertion(
            config, state, observation(step, position, wrench)
        )
        if not visited or command.next_state.phase is not visited[-1]:
            visited.append(command.next_state.phase)
        for index in range(3):
            position[index] += command.twist_assembly[index] / 240.0
        state = command.next_state
        if state.phase in {InsertionState.BACKOFF, InsertionState.SAFE_ABORT}:
            break
    assert InsertionState.FIRST_CONTACT in visited
    assert InsertionState.CONTACT_HOLD in visited
    assert InsertionState.CONTACT_CLASSIFY in visited
    assert InsertionState.ACTIVE_PROBE in visited
    assert state.phase is InsertionState.BACKOFF
    assert state.probe_leg == 16
    assert state.probe_total_steps <= config["active_probe"]["maximum_total_steps"]
    assert math.hypot(*state.xy_search_offset_m) <= config["motion"]["maximum_search_radius_m"]


def test_hard_gate_aborts_without_probe_or_retry():
    config = load_compliant_insertion_config()
    command = step_compliant_insertion(
        config, ControllerState(),
        observation(0, (0.0, 0.0, 0.0), (0.0, 0.0, 50.0, 0.0, 0.0, 0.0)),
    )
    assert command.next_state.phase is InsertionState.SAFE_ABORT
    assert command.next_state.abort_reason == "HARD_SAFETY_GATE"
    assert command.twist_assembly == (0.0,) * 6


def test_negative_fz_reaction_limits_positive_z_insertion():
    config = load_compliant_insertion_config()
    free_state = ControllerState(phase=InsertionState.INSERT_ADVANCE)
    resisted_state = ControllerState(phase=InsertionState.INSERT_ADVANCE)
    for step in range(12):
        free = step_compliant_insertion(
            config,
            free_state,
            observation(step, (0.0, 0.0, 0.0), (0.0,) * 6),
        )
        resisted = step_compliant_insertion(
            config,
            resisted_state,
            observation(
                step,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, -0.34, 0.0, 0.0, 0.0),
            ),
        )
        free_state = free.next_state
        resisted_state = resisted.next_state
    assert free.twist_assembly[2] > 0.0
    assert resisted.twist_assembly[2] >= 0.0
    assert resisted.twist_assembly[2] < free.twist_assembly[2]


def test_positive_fz_does_not_look_like_insertion_resistance():
    config = load_compliant_insertion_config()
    command = step_compliant_insertion(
        config,
        ControllerState(phase=InsertionState.INSERT_ADVANCE),
        observation(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.34, 0.0, 0.0, 0.0)),
    )
    assert command.twist_assembly[2] > 0.0
