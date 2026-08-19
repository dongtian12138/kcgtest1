from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = (
    WORKSPACE_ROOT
    / "src/kcg_connector/isaac/d38999_multilayer_event_onset_probe_v2.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("d38999_event_onset_v2_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_event_probe_is_one_event_per_fresh_process_and_frozen() -> None:
    runner = _load_runner()
    assert runner._run_hypothesis_id("spring_finger_engagement", 0) == runner.HYPOTHESIS_ID
    assert (
        runner._run_hypothesis_id("first_pin_socket_spring_touch", 1)
        == runner.EVENT04_FIX_HYPOTHESIS_ID
    )
    assert (
        runner._run_hypothesis_id("first_pin_socket_spring_touch", 2)
        == runner.EVENT04_PASSIVITY_HYPOTHESIS_ID
    )
    assert tuple(runner.EVENT_SPECS) == runner.EVENT_ORDER
    assert [runner.EVENT_SPECS[event]["ordinal"] for event in runner.EVENT_ORDER] == list(
        range(1, 8)
    )
    assert runner.PROBE_HALF_WINDOW_M == 0.00025
    assert runner.PROFILE_DURATION_S == 2.0
    peak_speed = 1.875 * 2.0 * runner.PROBE_HALF_WINDOW_M / runner.PROFILE_DURATION_S
    assert peak_speed == 0.00046875
    assert peak_speed <= 0.0005
    for label, relative in (
        ("master_contract", runner.MASTER_RELATIVE),
        ("physical_contract", runner.PHYSICAL_RELATIVE),
        ("acceptance_contract", runner.ACCEPTANCE_RELATIVE),
        ("authorized_overrides", runner.OVERRIDES_RELATIVE),
        ("model", runner.MODEL_RELATIVE),
        ("mapping", runner.MAPPING_RELATIVE),
        ("fine_offset_result", runner.FINE_OFFSET_RESULT_RELATIVE),
        ("initialization_aggregate", runner.INITIALIZATION_AGGREGATE_RELATIVE),
    ):
        assert runner._sha256(WORKSPACE_ROOT / relative) == runner.EXPECTED_SHA256[label]
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "configure_continuous_plug_guide_runtime_collision" not in source
    assert '"controller_stop_condition": "fixed_step_count_only"' in source
    assert '"controller_consumed_contact_names": False' in source
    assert '"controller_consumed_contact_normals": False' in source
    assert '"controller_consumed_event_truth": False' in source
    assert '"object_pose_write_after_physics_start_count": 0' in source
    assert '"--validation-attempt"' in source


def test_minimum_jerk_and_internal_actions_have_expected_onsets() -> None:
    runner = _load_runner()
    assert runner._minimum_jerk(0.0) == (0.0, 0.0)
    end_position, end_derivative = runner._minimum_jerk(1.0)
    assert math.isclose(end_position, 1.0, abs_tol=1.0e-15)
    assert math.isclose(end_derivative, 0.0, abs_tol=1.0e-15)
    middle_position, middle_derivative = runner._minimum_jerk(0.5)
    assert math.isclose(middle_position, 0.5, abs_tol=1.0e-15)
    assert math.isclose(middle_derivative, 1.875, abs_tol=1.0e-15)

    master = yaml.safe_load((WORKSPACE_ROOT / runner.MASTER_RELATIVE).read_text())
    scale = float(master["contact_layout"]["coordinate_scale_m_per_in"])
    pairs = [
        {
            "label": str(row["label"]),
            "center_m": np.asarray(row["center_in"], dtype=np.float64) * scale,
            "same_label_only": True,
        }
        for row in master["contact_layout"]["pairs"]
    ]
    nominal = {
        row["name"]: float(row["nominal_separation_m"])
        for row in master["assembly_events"]["ordered"]
    }
    zero3 = np.zeros(3, dtype=np.float64)
    zero6 = np.zeros(6, dtype=np.float64)
    moving_mass = sum(
        float(master["mass_properties"]["bodies"][name]["mass_kg"])
        for name in ("loose_plug_body_assembly", "coupling_nut")
    )
    for event in runner.EVENT_ORDER[1:6]:
        before_body = np.asarray((0.0, 0.0, -(nominal[event] - 1.0e-06)))
        after_x = 2.0e-06 if event == "first_pin_socket_spring_touch" else 20.0e-06 if event == "spring_finger_engagement" else 0.0
        after_body = np.asarray((after_x, 0.0, -(nominal[event] + 1.0e-06)))
        before = runner._internal_action(
            event,
            master,
            pairs,
            fixed_position=zero3,
            fixed_velocity=zero6,
            body_position=before_body,
            body_velocity=zero6,
            body_yaw=0.0,
            body_omega_z=0.0,
            nut_yaw=0.0,
            nut_omega_z=0.0,
            integration_dt_s=1.0 / 240.0,
            effective_mass_kg=moving_mass,
        )
        after = runner._internal_action(
            event,
            master,
            pairs,
            fixed_position=zero3,
            fixed_velocity=zero6,
            body_position=after_body,
            body_velocity=zero6,
            body_yaw=0.0,
            body_omega_z=0.0,
            nut_yaw=0.0,
            nut_omega_z=0.0,
            integration_dt_s=1.0 / 240.0,
            effective_mass_kg=moving_mass,
        )
        assert before["signal_n"] == 0.0
        assert after["signal_n"] > 0.0
        assert np.max(np.abs(after["body_force_n"])) < 8.0
        assert np.max(np.abs(after["nut_torque_nm"])) < 0.30


def test_same_label_shared_spring_is_stable_at_frozen_240hz() -> None:
    runner = _load_runner()
    master = yaml.safe_load((WORKSPACE_ROOT / runner.MASTER_RELATIVE).read_text())
    contact = master["elastic_contact_models"]["socket_contact_per_label"]
    count = int(contact["count"])
    stiffness = float(contact["aggregate_stiffness_n_m"])
    damping = float(contact["aggregate_damping_n_s_m"])
    mass = sum(
        float(master["mass_properties"]["bodies"][name]["mass_kg"])
        for name in ("loose_plug_body_assembly", "coupling_nut")
    )
    dt = 1.0 / 240.0
    x = 2.0e-06
    velocity = 0.0
    maximum_displacement = abs(x)
    for _ in range(720):
        force, audit = runner._backward_euler_shared_spring_force(
            [np.asarray((x, 0.0)) for _ in range(count)],
            [np.asarray((velocity, 0.0)) for _ in range(count)],
            active_fraction=1.0,
            per_channel_stiffness_n_m=stiffness,
            per_channel_damping_n_s_m=damping,
            integration_dt_s=dt,
            effective_mass_kg=mass,
        )
        velocity += dt * float(force[0]) / mass
        x += dt * velocity
        maximum_displacement = max(maximum_displacement, abs(x))
    assert audit["continuous_parameter_values_unchanged"] is True
    assert audit["effective_total_stiffness_n_m"] == count * stiffness
    assert maximum_displacement <= 2.0e-06
    assert abs(x) < 1.0e-15
    assert abs(velocity) < 1.0e-12


def test_same_label_passivity_audit_rescores_immutable_validation_trace() -> None:
    runner = _load_runner()
    master = yaml.safe_load((WORKSPACE_ROOT / runner.MASTER_RELATIVE).read_text())
    trace = (
        WORKSPACE_ROOT
        / "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
        "DYN-A1-EVENT-ONSET-CALIBRATION-V2/EVENT_PROBES/"
        "EVENT_04_first_pin_socket_spring_touch_VALIDATION_01/trace.jsonl"
    )
    samples = [json.loads(line) for line in trace.read_text().splitlines()]
    audit = runner._same_label_passivity_audit(samples, master, 1.0e-9)
    assert audit["passed"] is True
    assert audit["active_sample_count"] == 15
    assert audit["legacy_interval_force_dot_displacement_positive_sample_count"] == 2
    assert audit["spring_restoring_pass"] is True
    assert audit["damping_dissipative_pass"] is True
    assert audit["force_law_pass"] is True
    assert audit["bounded_response_pass"] is True
    assert audit["final_convergence_pass"] is True
    assert audit["maximum_raw_force_law_residual_n"] < 1.0e-16


def test_axis_driver_preserves_internal_action_and_total_limit() -> None:
    runner = _load_runner()
    driver, integral, total, _saturated = runner._axis_driver(
        target_position=-0.015,
        target_velocity=-0.0005,
        actual_position=-0.0148,
        actual_velocity=0.0,
        integral_n=-7.4,
        internal_force_n=4.8,
        dt=1.0 / 240.0,
        force_limit_n=8.0,
    )
    assert -8.0 <= total <= 8.0
    assert math.isclose(total, driver + 4.8, abs_tol=1.0e-12)
    assert -7.5 <= integral <= 7.5


def test_thread_direction_audit_includes_axial_and_rotational_channels() -> None:
    runner = _load_runner()
    lead = 0.00762

    def row(
        step: int,
        phase_error: float,
        phase_rate: float,
        body_velocity_z: float,
    ) -> dict:
        force = 10000.0 * phase_error + 20.0 * phase_rate
        torque = -force * lead / (2.0 * math.pi)
        return {
            "step": step,
            "event_force_signal_n": abs(force),
            "body_internal_force_n": [0.0, 0.0, force],
            "body_velocity_m_s": [0.0, 0.0, body_velocity_z],
            "nut_total_applied_torque_nm": [0.0, 0.0, torque],
            "event_internal_directional_dot": force
            * max(0.0, -body_velocity_z),
            "event_internal_detail": {
                "thread_phase_error_m": phase_error,
                "thread_phase_rate_m_s": phase_rate,
                "thread_lead_m_per_revolution": lead,
            },
        }

    samples = [
        row(1, 6.0e-7, 4.7e-4, -1.9e-4),
        row(2, -5.0e-6, 0.0, -1.0e-4),
    ]
    assert samples[1]["event_internal_directional_dot"] < 0.0
    audit = runner._thread_constraint_direction_audit(samples, 1.0e-9)
    assert audit["passed"] is True
    assert audit["initial_resistance_pass"] is True
    assert audit["force_law_pass"] is True
    assert audit["torque_mapping_pass"] is True
    assert audit["legacy_axial_only_negative_sample_count"] == 1

    invalid = [dict(samples[0]), dict(samples[1])]
    invalid[1]["nut_total_applied_torque_nm"] = [0.0, 0.0, 0.1]
    rejected = runner._thread_constraint_direction_audit(invalid, 1.0e-9)
    assert rejected["passed"] is False
    assert rejected["torque_mapping_pass"] is False


def test_physical_force_noise_uses_only_pre_manifold_samples() -> None:
    runner = _load_runner()
    samples = [
        {
            "separation_m": 0.006480,
            "event_force_signal_n": 0.0,
            "event_manifold_active": False,
        },
        {
            "separation_m": 0.006483,
            "event_force_signal_n": 0.0,
            "event_manifold_active": True,
        },
        {
            "separation_m": 0.006498,
            "event_force_signal_n": 0.0,
            "event_manifold_active": True,
        },
        {
            "separation_m": 0.006500,
            "event_force_signal_n": 0.02,
            "event_manifold_active": True,
        },
        {
            "separation_m": 0.006499,
            "event_force_signal_n": 0.25,
            "event_manifold_active": True,
        },
    ]
    onset = runner._signal_onset(
        samples,
        "event_force_signal_n",
        0.0065,
        inactive_key="event_manifold_active",
    )
    assert onset["observed"] is True
    assert onset["pre_event_noise_max_n"] == 0.0
    assert onset["threshold_n"] == runner.NUMERICAL_FORCE_FLOOR_N
    assert onset["noise_sample_criterion"] == "event_manifold_active=false"
    assert 0.006498 <= onset["estimated_separation_m"] <= 0.006500


def test_event1_collision_partition_allows_only_same_index_keys_and_guides() -> None:
    runner = _load_runner()
    rows = []
    for index in range(5):
        for side in ("left", "right"):
            rows.append(
                {
                    "path": f"/fixed/keyway_{index}_{side}",
                    "owner": "fixed_receptacle",
                    "role": "continuous_keyway_wall",
                    "trace_label": f"keyway_{index}",
                }
            )
        rows.append(
            {
                "path": f"/body/key_{index}",
                "owner": "body_assembly",
                "role": "continuous_polarizing_key",
                "trace_label": f"key_{index}",
            }
        )
    rows.extend(
        [
            {
                "path": "/fixed/guide",
                "owner": "fixed_receptacle",
                "role": "continuous_shell_and_guidance",
                "trace_label": "fixed_guide",
            },
            {
                "path": "/body/guide",
                "owner": "body_assembly",
                "role": "continuous_shell_and_guidance",
                "trace_label": "body_guide",
            },
            {
                "path": "/fixed/other",
                "owner": "fixed_receptacle",
                "role": "other",
                "trace_label": None,
            },
            {
                "path": "/body/other",
                "owner": "body_assembly",
                "role": "other",
                "trace_label": None,
            },
            {
                "path": "/nut/collider",
                "owner": "coupling_nut",
                "role": "nut",
                "trace_label": None,
            },
        ]
    )
    members, allowed = runner._probe_collision_partition(
        rows, "five_key_polarization"
    )
    assert set(allowed) == {
        ("FixedGuide", "BodyGuide"),
        *((f"FixedKey{index}", f"BodyKey{index}") for index in range(5)),
    }
    assert len(allowed) == 6
    for index in range(5):
        assert len(members[f"FixedKey{index}"]) == 2
        assert len(members[f"BodyKey{index}"]) == 1
        assert (f"FixedKey{index}", f"BodyKey{(index + 1) % 5}") not in allowed
    assert runner._key_labels_correspond(
        ["continuous_keyway_wall", "continuous_polarizing_key"],
        ["keyway_3", "key_3"],
    ) is True
    assert runner._key_labels_correspond(
        ["continuous_keyway_wall", "continuous_polarizing_key"],
        ["keyway_3", "key_4"],
    ) is False
