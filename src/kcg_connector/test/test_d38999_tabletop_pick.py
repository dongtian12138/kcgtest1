"""Pure tests for the independent D38999 tabletop pick candidate."""

from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from kcg_connector.d38999_tabletop_pick import (
    D38999_PICK_SCHEMA_VERSION,
    EXPECTED_D38999_CLOSURE_CLEARANCE_ARM_RAD,
    EXPECTED_D38999_GRASP_ARM_RAD,
    EXPECTED_TORQUE_JOINT_NAMES,
    iiwa14_grasp_tcp_transform,
    interpolate_arm,
    load_d38999_tabletop_pick_config,
    minimum_jerk_blend,
    verify_d38999_pick_dependencies,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config/d38999_tabletop_pick_v1.yaml"
SYNTHETIC_GRASP_ARM = (
    -0.164155590,
    0.426740717,
    -0.376494151,
    -0.980754913,
    0.155526632,
    1.758561897,
    -0.096543358,
)


def _config():
    return load_d38999_tabletop_pick_config(CONFIG_PATH)


def _invalid_document(tmp_path, mutator):
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutator(document)
    path = tmp_path / "invalid_d38999_pick.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_contract_is_independent_d38999_not_synthetic_pick():
    config = _config()
    assert config.schema_version == D38999_PICK_SCHEMA_VERSION
    assert config.scene.tabletop_config == "d38999_tabletop_scene_v1.yaml"
    assert config.scene.proxy_config == "d38999_shell25j_proxy_v1.yaml"
    assert "connector_pair.usda" not in json.dumps(config.as_dict())
    assert config.motion.grasp_arm_rad == pytest.approx(
        EXPECTED_D38999_GRASP_ARM_RAD
    )
    assert config.motion.grasp_arm_rad != pytest.approx(SYNTHETIC_GRASP_ARM)


def test_geometry_screen_matches_48mm_short_proxy_without_claiming_dynamics():
    config = _config()
    candidate = config.geometry_candidate
    assert 2.0 * candidate.rear_body_radius_m == pytest.approx(0.0443)
    assert 2.0 * candidate.coupling_nut_outer_radius_m == pytest.approx(0.048)
    assert candidate.rear_body_world_z_interval_m == pytest.approx(
        (0.217, 0.231)
    )
    assert candidate.coupling_nut_world_z_interval_m == pytest.approx(
        (0.207, 0.224)
    )
    assert candidate.grip_local_z_interval_m == pytest.approx(
        (0.41748, 0.44148)
    )
    assert candidate.dynamics_validated is False


def test_terminal_and_implemented_closure_envelope_clear_the_table():
    config = _config()
    candidate = config.geometry_candidate
    handbase_z = (
        config.motion.grasp_tcp_position_m[2] + candidate.handbase_to_tcp_m
    )
    clearance = handbase_z - candidate.maximum_closed_finger_local_z_m - 0.200
    assert clearance == pytest.approx(0.03088)
    assert clearance >= (
        candidate.minimum_predicted_terminal_finger_table_clearance_m
    )
    swept = (
        handbase_z - candidate.maximum_closure_swept_finger_local_z_m - 0.200
    )
    assert swept == pytest.approx(0.011807)
    assert swept == pytest.approx(
        candidate.predicted_closure_sweep_table_clearance_m
    )
    assert candidate.closure_sweep_collision_free is True


def test_proposed_clearance_ik_has_5mm_nominal_sweep_margin_only():
    candidate = _config().geometry_candidate
    transform = iiwa14_grasp_tcp_transform(
        candidate.proposed_clearance_arm_rad
    )
    position = tuple(transform[index][3] for index in range(3))
    axis = tuple(transform[index][2] for index in range(3))
    assert position == pytest.approx(
        candidate.proposed_clearance_tcp_position_m, abs=1.0e-7
    )
    assert axis == pytest.approx((0.0, 0.0, -1.0), abs=1.0e-6)
    handbase_z = position[2] + candidate.handbase_to_tcp_m
    margin = (
        handbase_z - candidate.maximum_closure_swept_finger_local_z_m - 0.200
    )
    assert margin == pytest.approx(0.011807)
    assert margin == pytest.approx(
        candidate.proposed_clearance_nominal_table_margin_m
    )
    config = _config()
    assert candidate.proposed_motion_implemented is True
    assert config.motion.closure_clearance_arm_rad == pytest.approx(
        EXPECTED_D38999_CLOSURE_CLEARANCE_ARM_RAD
    )
    assert config.motion.closure_clearance_arm_rad == pytest.approx(
        candidate.proposed_clearance_arm_rad
    )
    assert config.motion.closure_clearance_tcp_position_m == pytest.approx(
        candidate.proposed_clearance_tcp_position_m
    )


def test_pure_iiwa_fk_reproduces_new_d38999_ik_candidate():
    config = _config()
    transform = iiwa14_grasp_tcp_transform(config.motion.grasp_arm_rad)
    position = tuple(transform[index][3] for index in range(3))
    down_axis = tuple(transform[index][2] for index in range(3))
    assert position == pytest.approx(
        config.motion.grasp_tcp_position_m, abs=1.0e-7
    )
    assert down_axis == pytest.approx((0.0, 0.0, -1.0), abs=1.0e-6)
    assert config.motion.grasp_tcp_position_m == pytest.approx(
        (0.520, -0.210, 0.24848)
    )


def test_motion_peaks_and_phase_steps_are_bounded():
    config = _config()
    motion = config.motion
    starts = (config.robot.home_arm_rad,) + tuple(
        item.target_arm_rad for item in motion.approach_segments[:-1]
    )
    for start, segment in zip(starts, motion.approach_segments):
        peak = max(
            1.875 * abs(end - begin) / segment.duration_s
            for begin, end in zip(start, segment.target_arm_rad)
        )
        assert peak <= 0.3
    arm_segments = (
        (
            motion.approach_segments[-1].target_arm_rad,
            motion.closure_clearance_arm_rad,
            motion.descent_duration_s,
        ),
        (
            motion.closure_clearance_arm_rad,
            motion.grasp_arm_rad,
            motion.closed_seating_duration_s,
        ),
        (
            motion.grasp_arm_rad,
            motion.approach_segments[-1].target_arm_rad,
            motion.lift_duration_s,
        ),
    )
    arm_peaks = [
        max(
            1.875 * abs(end - begin) / duration
            for begin, end in zip(start, target)
        )
        for start, target, duration in arm_segments
    ]
    closure_peak = max(
        1.875 * abs(end - begin) / motion.closure_duration_s
        for begin, end in zip(
            config.robot.open_hand_rad, motion.grasp_hand_rad
        )
    )
    assert all(peak <= 0.3 for peak in arm_peaks)
    assert closure_peak <= 0.5
    durations = [item.duration_s for item in motion.approach_segments]
    durations.extend(
        (
            motion.hand_open_duration_s,
            motion.pregrasp_hold_duration_s,
            motion.descent_duration_s,
            motion.open_tare_duration_s,
            motion.closure_duration_s,
            motion.closed_seating_duration_s,
            motion.preload_duration_s,
            motion.lift_duration_s,
            motion.final_hold_duration_s,
            motion.effort_sample_duration_s,
        )
    )
    assert all(value * 240 == round(value * 240) for value in durations)


def test_sensor_and_contact_gates_match_available_hardware():
    config = _config()
    assert config.sensing.torque_joint_names == EXPECTED_TORQUE_JOINT_NAMES
    assert config.sensing.maximum_absolute_torque_delta_nm == pytest.approx(
        2.0
    )
    assert config.sensing.operational_torque_target_nm == pytest.approx(1.8)
    assert config.sensing.minimum_loaded_channels == 2
    assert config.sensing.fingertip_tactile_available is False
    acceptance = config.acceptance
    assert acceptance.maximum_final_observable_joint_speed_rad_s == (
        pytest.approx(0.030)
    )
    assert acceptance.maximum_final_post_solver_joint_speed_rad_s == (
        pytest.approx(0.050)
    )
    assert acceptance.require_only_finger_loose_plug_contacts
    assert acceptance.require_zero_preclosure_robot_loose_plug_contacts
    assert acceptance.require_zero_robot_table_contacts
    assert acceptance.require_zero_robot_fixture_contacts
    assert acceptance.require_zero_robot_fixed_endpoint_contacts


def test_dependency_verifier_hash_pins_and_cross_checks_contracts():
    dependencies = verify_d38999_pick_dependencies(
        _config(), CONFIG_PATH, PACKAGE_ROOT.parents[1]
    )
    assert dependencies["d38999_asset"].name == (
        "d38999_shell25j_61_pair_proxy_v1.usda"
    )
    assert dependencies["robot_asset"].name == "handarm.usda"
    assert dependencies["proxy"].identity.proxy_id == (
        dependencies["tabletop"].asset.proxy_id
    )


def test_fail_closed_boundaries_are_explicit():
    boundaries = _config().boundaries
    assert boundaries.attachment_allowed is False
    assert boundaries.object_drive_allowed is False
    assert boundaries.object_pose_writes_after_start_allowed is False
    assert boundaries.collision_planned is False
    assert boundaries.self_collision_verified is False


@pytest.mark.parametrize(
    "mutator,message",
    (
        (lambda doc: doc.update(unexpected=True), "keys are invalid"),
        (
            lambda doc: doc.update(schema_version="wrong"),
            "unsupported",
        ),
        (
            lambda doc: doc["scene"].update(
                tabletop_config="connector_tabletop_scene_v1.yaml"
            ),
            "D38999 tabletop",
        ),
        (
            lambda doc: doc["geometry_candidate"].update(
                coupling_nut_outer_radius_m=0.045
            ),
            "not canonical",
        ),
        (
            lambda doc: doc["geometry_candidate"].update(
                dynamics_validated=True
            ),
            "must not claim",
        ),
        (
            lambda doc: doc["geometry_candidate"].update(
                proposed_motion_implemented=False
            ),
            "must remain implemented",
        ),
        (
            lambda doc: doc["motion"].update(
                grasp_arm_rad=list(SYNTHETIC_GRASP_ARM)
            ),
            "D38999 grasp IK",
        ),
        (
            lambda doc: doc["motion"].update(
                grasp_tcp_position_m=[0.520, -0.210, 0.291]
            ),
            "D38999 grasp TCP",
        ),
        (
            lambda doc: doc["motion"].update(
                closure_clearance_tcp_position_m=[0.520, -0.210, 0.243]
            ),
            "closure-clearance TCP",
        ),
        (
            lambda doc: doc["motion"].update(
                grasp_hand_rad=[1.0, 0.75, 0.50, 0.75]
            ),
            "D38999 hand target",
        ),
        (
            lambda doc: doc["motion"].update(closure_duration_s=3.0),
            "durations are not canonical",
        ),
        (
            lambda doc: doc["sensing"].update(
                torque_joint_names=["f1j1", "f2j1", "f3j2"]
            ),
            "real base torque axes",
        ),
        (
            lambda doc: doc["sensing"].update(
                maximum_absolute_torque_delta_nm=2.1
            ),
            "thresholds are not canonical",
        ),
        (
            lambda doc: doc["sensing"].update(
                operational_torque_target_nm=2.0
            ),
            "thresholds are not canonical",
        ),
        (
            lambda doc: doc["acceptance"].update(
                maximum_body_tcp_slip_m=0.006
            ),
            "safety bound",
        ),
        (
            lambda doc: doc["acceptance"].update(
                require_only_finger_loose_plug_contacts=False
            ),
            "must be enabled",
        ),
        (
            lambda doc: doc["boundaries"].update(attachment_allowed=True),
            "fail-closed",
        ),
    ),
)
def test_rejects_unsafe_or_synthetic_semantic_drift(
    tmp_path, mutator, message
):
    with pytest.raises(ValueError, match=message):
        load_d38999_tabletop_pick_config(_invalid_document(tmp_path, mutator))


def test_interpolation_is_minimum_jerk_and_endpoint_exact():
    start = (0.0,) * 7
    end = EXPECTED_D38999_GRASP_ARM_RAD
    assert minimum_jerk_blend(0.0) == 0.0
    assert minimum_jerk_blend(1.0) == 1.0
    assert interpolate_arm(start, end, 0.0) == pytest.approx(start)
    assert interpolate_arm(start, end, 1.0) == pytest.approx(end)
    assert minimum_jerk_blend(0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        minimum_jerk_blend(math.nextafter(1.0, math.inf))


def test_config_is_immutable_and_json_safe():
    config = _config()
    json.dumps(config.as_dict(), allow_nan=False, sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        config.motion.descent_duration_s = 0.0


def test_import_is_pure_without_isaac_omni_pxr_or_scipy():
    script = r"""
import json
import sys
from kcg_connector.d38999_tabletop_pick import (
    load_d38999_tabletop_pick_config,
)
load_d38999_tabletop_pick_config()
for name in ("isaacsim", "omni", "pxr", "scipy"):
    assert name not in sys.modules, name
print(json.dumps({"pure_d38999_pick_import": True}))
"""
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(PACKAGE_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == {"pure_d38999_pick_import": True}
