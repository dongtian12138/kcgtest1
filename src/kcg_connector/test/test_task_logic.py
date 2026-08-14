import math

from kcg_connector.task_logic import (
    ConnectorMetrics,
    ConnectorPhase,
    ConnectorTaskConfig,
    evaluate_connector_task,
)


CONFIG = ConnectorTaskConfig()


def metrics(**overrides):
    values = {
        "grasp_distance": 0.002,
        "loaded_torque_channels": 3,
        "maximum_absolute_finger_torque": 0.3,
        "lateral_error": 0.0005,
        "angular_error": math.radians(0.5),
        "key_error": math.radians(1.0),
        "insertion_depth": CONFIG.engage_depth,
        "coupling_angle": CONFIG.target_coupling_angle,
        "axial_lock_travel": CONFIG.helical_lead,
        "hold_seconds": CONFIG.hold_duration,
        "engaged": True,
    }
    values.update(overrides)
    return ConnectorMetrics(**values)


def test_complete_connector_state_passes():
    result = evaluate_connector_task(metrics(), CONFIG)
    assert result.phase == ConnectorPhase.PASSED
    assert result.success
    assert result.failure_reason == ""


def test_alignment_precedes_insertion():
    result = evaluate_connector_task(
        metrics(lateral_error=0.003, insertion_depth=0.0), CONFIG
    )
    assert result.phase == ConnectorPhase.PREALIGN
    assert not result.success


def test_inserting_while_misaligned_fails_explicitly():
    result = evaluate_connector_task(
        metrics(lateral_error=0.003, insertion_depth=0.002), CONFIG
    )
    assert result.phase == ConnectorPhase.FAILED
    assert result.failure_reason == "misaligned"


def test_incomplete_rotation_stays_in_screw_phase():
    angle = math.pi
    result = evaluate_connector_task(
        metrics(
            coupling_angle=angle,
            axial_lock_travel=CONFIG.helical_lead / 2.0,
            hold_seconds=0.0,
        ),
        CONFIG,
    )
    assert result.phase == ConnectorPhase.SCREW


def test_inconsistent_rotation_and_travel_is_cross_thread():
    result = evaluate_connector_task(
        metrics(coupling_angle=math.pi, axial_lock_travel=0.0), CONFIG
    )
    assert result.phase == ConnectorPhase.FAILED
    assert result.failure_reason == "cross_thread"


def test_hold_must_be_continuous_before_success():
    result = evaluate_connector_task(metrics(hold_seconds=0.5), CONFIG)
    assert result.phase == ConnectorPhase.HOLD
    assert not result.success


def test_nonfinite_state_fails():
    result = evaluate_connector_task(metrics(lateral_error=math.nan), CONFIG)
    assert result.failure_reason == "invalid_physics"


def test_finger_overload_fails():
    result = evaluate_connector_task(
        metrics(maximum_absolute_finger_torque=1.1), CONFIG
    )
    assert result.failure_reason == "finger_overload"
