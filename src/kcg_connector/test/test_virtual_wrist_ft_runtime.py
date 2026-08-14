"""Pure tests for the opt-in D38999 virtual wrist-wrench monitor."""

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from kcg_connector.virtual_wrist_ft_runtime import (
    DEFAULT_CONFIG_PATH,
    VirtualWristFtMonitor,
    classify_e2e_wrist_ft_phase,
    column_rotation_from_gf_matrix3d,
    load_virtual_wrist_ft_monitor_config,
    inverse_wrench_transform,
    reaction_row_index,
    transform_wrench_between_frames,
    transform_wrench_to_task,
    verify_virtual_wrist_ft_monitor_inputs,
)


REPOSITORY = Path(__file__).resolve().parents[3]


def _document():
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path, document):
    path = tmp_path / "wrist_ft_monitor.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def _monitor():
    config = load_virtual_wrist_ft_monitor_config()
    return VirtualWristFtMonitor(
        config,
        reaction_row=8,
        task_origin_world=(0.0, 0.0, 0.0),
        task_z_axis_world=(0.0, 0.0, 1.0),
    )


def _observe(monitor, raw, step, phase, sensor_position=(0.0, 0.0, 0.0)):
    return monitor.observe(
        raw,
        global_step=step,
        runtime_phase=phase,
        sensor_position_world=sensor_position,
        sensor_rotation_world=np.eye(3),
    )


def test_shipped_contract_is_hash_bound_monitor_only_and_not_v1():
    config = load_virtual_wrist_ft_monitor_config()
    resolved = verify_virtual_wrist_ft_monitor_inputs(config, REPOSITORY)
    assert config.metadata_joint_index_offset == 1
    assert config.canonical_from_raw == tuple(
        tuple(float(value) for value in row) for row in -np.eye(6)
    )
    assert config.monitor_only is True
    assert all(limit is None for limit in config.safety_limits)
    assert config.threshold_repeat_count == 0
    for name, path in resolved.items():
        expected = getattr(
            config,
            f"{name.replace('wrist_ft_', 'wrist_')}_sha256",
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_reaction_row_is_metadata_joint_index_plus_one():
    config = load_virtual_wrist_ft_monitor_config()
    assert reaction_row_index({"hand2arm": 7}, config) == 8
    with pytest.raises(ValueError, match="absent"):
        reaction_row_index({"iiwa_joint_7": 6}, config)


@pytest.mark.parametrize(
    ("runtime_phase", "policy_phase"),
    (
        ("initial_settle", "HOME_FREE_SPACE_EMPTY_HAND"),
        ("unsupported_final_hold", "POST_GRASP_FREE_SPACE"),
        ("mixed_grip_physical_insert_01", "INSERT"),
        ("contact_response_plus_X_hold", "INSERT"),
        ("engaged_keying_proxy_activation", "ENGAGE"),
        ("end_to_end_rotation_2_motion", "SCREW"),
        ("end_to_end_rotation_2_hold", "HOLD"),
        ("end_to_end_reverse_mid_to_home", "OTHER"),
    ),
)
def test_detailed_e2e_phases_map_to_strict_tare_policy(
    runtime_phase, policy_phase
):
    assert classify_e2e_wrist_ft_phase(runtime_phase) == policy_phase


def test_wrench_transform_shifts_moment_to_engagement_datum():
    # A +Y force applied 1 m along +X from the datum adds +Z moment.
    transformed = transform_wrench_to_task(
        (0.0, 2.0, 0.0, 0.0, 0.0, 0.0),
        sensor_position_world=(1.0, 0.0, 0.0),
        sensor_rotation_world=np.eye(3),
        task_origin_world=(0.0, 0.0, 0.0),
        task_rotation_world=np.eye(3),
    )
    assert transformed == pytest.approx((0.0, 2.0, 0.0, 0.0, 0.0, 2.0))


def test_t_a_s_coincident_origins_preserves_pure_force():
    transformed = transform_wrench_between_frames(
        (1.0, -2.0, 3.0, 0.0, 0.0, 0.0),
        np.eye(3),
        (0.0, 0.0, 0.0),
    )
    assert transformed == pytest.approx((1.0, -2.0, 3.0, 0.0, 0.0, 0.0))


def test_t_a_s_rotates_force_ninety_degrees_about_z():
    rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    transformed = transform_wrench_between_frames(
        (2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        rotation,
        (0.0, 0.0, 0.0),
    )
    assert transformed == pytest.approx((0.0, 2.0, 0.0, 0.0, 0.0, 0.0))


def test_t_a_s_rotates_pure_moment_without_spurious_force():
    rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    transformed = transform_wrench_between_frames(
        (0.0, 0.0, 0.0, 4.0, 0.0, 0.0),
        rotation,
        (0.3, -0.2, 0.1),
    )
    assert transformed == pytest.approx((0.0, 0.0, 0.0, 0.0, 4.0, 0.0))


def test_t_a_s_known_offset_force_adds_p_cross_f_moment():
    transformed = transform_wrench_between_frames(
        (0.0, 3.0, 0.0, 0.0, 0.0, 0.0),
        np.eye(3),
        (0.5, 0.0, 0.0),
    )
    assert transformed == pytest.approx((0.0, 3.0, 0.0, 0.0, 0.0, 1.5))


def test_t_a_s_wrench_round_trip():
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    source = np.asarray((1.2, -0.8, 2.5, 0.4, -0.7, 1.1))
    transformed = transform_wrench_between_frames(
        source, rotation, (0.12, -0.03, 0.07)
    )
    recovered = inverse_wrench_transform(
        transformed, rotation, (0.12, -0.03, 0.07)
    )
    assert recovered == pytest.approx(source)


def test_t_a_s_reversed_lever_arm_is_detected_by_offset_case():
    expected = np.asarray((0.0, 2.0, 0.0, 0.0, 0.0, 2.0))
    wrong = transform_wrench_between_frames(
        (0.0, 2.0, 0.0, 0.0, 0.0, 0.0),
        np.eye(3),
        (-1.0, 0.0, 0.0),
    )
    assert not np.allclose(wrong, expected)
    assert wrong[5] == pytest.approx(-2.0)


def test_gf_row_rotation_is_transposed_for_column_wrench_math():
    # Matrix3d for +90 deg about Z: local +X becomes world +Y under Gf's
    # ``vector * matrix`` convention.
    gf_rotation_rows = np.asarray(
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    sensor_rotation = column_rotation_from_gf_matrix3d(gf_rotation_rows)
    assert sensor_rotation @ np.asarray((1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 1.0, 0.0)
    )


def test_gf_rotation_adapter_rejects_non_rotation_matrix():
    with pytest.raises(ValueError, match="right-handed orthonormal"):
        column_rotation_from_gf_matrix3d(np.diag((1.0, 1.0, 2.0)))


def test_baselines_are_phase_limited_and_protected_peaks_are_timestamped():
    monitor = _monitor()
    for step in range(1, 121):
        _observe(monitor, (-1, -2, -3, -4, -5, -6), step, "initial_settle")
    assert monitor.capture_home_tare() == pytest.approx((1, 2, 3, 4, 5, 6))

    for step in range(121, 241):
        _observe(
            monitor,
            (-2, -4, -6, -8, -10, -12),
            step,
            "unsupported_final_hold",
        )
    assert monitor.capture_payload_baseline() == pytest.approx(
        (2, 4, 6, 8, 10, 12)
    )

    sample = _observe(
        monitor,
        (-5, -8, -13, -8, -10, -14),
        241,
        "mixed_grip_physical_insert_01",
    )
    assert sample["compensated_wrench_task"] == pytest.approx(
        (3, 4, 7, 0, 0, 2)
    )
    assert sample["timestamp_s"] == pytest.approx(241 / 240)
    assert sample["source_frame"] == "handbase_link"
    assert sample["target_frame"] == "connector_task_frame"
    report = monitor.report()
    peaks = report["protected_phase_peaks"]["INSERT"]
    assert peaks["lateral_force_n"]["absolute_peak"] == pytest.approx(5.0)
    assert peaks["axial_force_n"]["signed_value_at_peak"] == pytest.approx(7.0)
    assert peaks["tightening_torque_nm"]["absolute_peak"] == pytest.approx(2.0)
    assert report["dynamic_inertia_compensation_complete"] is False
    assert report["calibrated_safety_limits"] is None
    assert report["modifies_e2e_pass_gate"] is False


def test_payload_capture_requires_home_and_contact_requires_payload():
    monitor = _monitor()
    for step in range(1, 121):
        _observe(monitor, np.zeros(6), step, "unsupported_final_hold")
    with pytest.raises(RuntimeError, match="home empty-hand"):
        monitor.capture_payload_baseline()

    monitor = _monitor()
    with pytest.raises(RuntimeError, match="before payload"):
        _observe(
            monitor,
            np.zeros(6),
            1,
            "end_to_end_rotation_1_motion",
        )


def test_nonfinite_or_nonmonotonic_samples_fail_closed():
    monitor = _monitor()
    _observe(monitor, np.zeros(6), 1, "initial_settle")
    with pytest.raises(ValueError, match="strictly increasing"):
        _observe(monitor, np.zeros(6), 1, "initial_settle")
    with pytest.raises(ValueError, match="finite vector"):
        _observe(monitor, [0, 0, np.nan, 0, 0, 0], 2, "initial_settle")


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda doc: doc.update(extra=True), "keys are invalid"),
        (
            lambda doc: doc["source"].update(
                canonical_from_raw=np.eye(6).tolist()
            ),
            "exactly -I6",
        ),
        (
            lambda doc: doc["compatibility"].update(
                modifies_active_interface=True
            ),
            "must remain false",
        ),
        (
            lambda doc: doc["safety_limits"].update(
                maximum_axial_force_n=100.0
            ),
            "require completed",
        ),
        (
            lambda doc: doc["threshold_calibration"].update(
                repeat_count=1
            ),
            "pending with zero runs",
        ),
        (
            lambda doc: doc["phase_policy"].update(
                payload_capture_allowed=["INSERT"]
            ),
            "tare capture phases changed",
        ),
    ),
)
def test_loader_rejects_schema_drift_threshold_invention_or_overclaim(
    tmp_path, mutator, message
):
    document = deepcopy(_document())
    mutator(document)
    with pytest.raises(ValueError, match=message):
        load_virtual_wrist_ft_monitor_config(_write(tmp_path, document))
