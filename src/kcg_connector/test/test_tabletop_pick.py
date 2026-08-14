"""Pure contract tests for deterministic tabletop pick v1."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from kcg_connector.tabletop_pick import (
    EXPECTED_GRASP_ARM_RAD,
    EXPECTED_TORQUE_JOINT_NAMES,
    PICK_SCHEMA_VERSION,
    load_tabletop_pick_config,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config/connector_tabletop_pick_v1.yaml"


def _config():
    return load_tabletop_pick_config(CONFIG_PATH)


def _invalid_document(tmp_path, mutator):
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutator(document)
    path = tmp_path / "invalid_pick.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_pick_v1_exact_motion_and_sensor_contract():
    config = _config()
    assert config.schema_version == PICK_SCHEMA_VERSION
    assert config.pregrasp.config == (
        "connector_home_to_pregrasp_v1.yaml"
    )
    assert config.motion.grasp_arm_rad == pytest.approx(
        EXPECTED_GRASP_ARM_RAD
    )
    assert config.motion.grasp_tcp_position_m == pytest.approx(
        (0.520, -0.210, 0.291)
    )
    assert config.motion.grasp_hand_rad == pytest.approx(
        (1.0, 0.75, 0.50, 0.75)
    )
    assert config.sensing.torque_joint_names == (
        EXPECTED_TORQUE_JOINT_NAMES
    )
    assert config.sensing.fingertip_tactile_available is False


def test_phase_durations_are_exact_whole_240hz_steps():
    motion = _config().motion
    durations = (
        motion.descent_duration_s,
        motion.open_tare_duration_s,
        motion.closure_duration_s,
        motion.preload_duration_s,
        motion.lift_duration_s,
        motion.final_hold_duration_s,
        motion.effort_sample_duration_s,
    )
    assert durations == pytest.approx(
        (2.5, 0.5, 3.0, 0.5, 2.5, 4.0, 0.5)
    )
    assert all(
        duration * 240 == round(duration * 240)
        for duration in durations
    )


def test_force_and_physical_contact_gates_are_fail_closed():
    config = _config()
    assert config.sensing.loaded_torque_threshold_nm == pytest.approx(0.020)
    assert config.sensing.minimum_loaded_channels == 2
    assert config.sensing.maximum_absolute_torque_delta_nm == (
        pytest.approx(1.0)
    )
    acceptance = config.acceptance
    assert acceptance.maximum_body_tcp_slip_m == pytest.approx(0.005)
    assert acceptance.minimum_body_lift_m == pytest.approx(0.040)
    assert acceptance.minimum_final_bottom_clearance_m == pytest.approx(0.030)
    assert acceptance.maximum_final_observable_joint_speed_rad_s == (
        pytest.approx(0.030)
    )
    assert acceptance.maximum_final_post_solver_joint_speed_rad_s == (
        pytest.approx(0.050)
    )
    assert acceptance.require_zero_preclosure_robot_connector_contacts
    assert acceptance.require_zero_robot_table_contacts
    assert acceptance.require_zero_robot_fixture_contacts
    assert acceptance.require_zero_robot_fixed_endpoint_contacts
    assert acceptance.require_zero_final_plug_table_contacts
    assert acceptance.require_physical_grip_contact


@pytest.mark.parametrize(
    "mutator,message",
    (
        (
            lambda document: document.update(unexpected=True),
            "keys are invalid",
        ),
        (
            lambda document: document.update(schema_version="wrong"),
            "unsupported",
        ),
        (
            lambda document: document["pregrasp"].update(config="wrong"),
            "not canonical",
        ),
        (
            lambda document: document["motion"].update(
                grasp_arm_rad=[0] * 7
            ),
            "grasp arm seed",
        ),
        (
            lambda document: document["motion"].update(
                grasp_hand_rad=[0, 0, 0, 0]
            ),
            "grasp hand target",
        ),
        (
            lambda document: document["motion"].update(
                closure_duration_s=2.9
            ),
            "durations are not canonical",
        ),
        (
            lambda document: document["motion"].update(
                grip_static_friction=0.5
            ),
            "grip physics values",
        ),
        (
            lambda document: document["sensing"].update(
                torque_joint_names=["f1j1", "f2j1", "f3j2"]
            ),
            "real base axes",
        ),
        (
            lambda document: document["sensing"].update(
                fingertip_tactile_available=True
            ),
            "tactile must remain unavailable",
        ),
        (
            lambda document: document["sensing"].update(
                minimum_loaded_channels=1
            ),
            "exactly two",
        ),
        (
            lambda document: document["acceptance"].update(
                maximum_body_tcp_slip_m=0.010
            ),
            "safety bound",
        ),
        (
            lambda document: document["acceptance"].update(
                maximum_final_observable_joint_speed_rad_s=0.031
            ),
            "safety bound",
        ),
        (
            lambda document: document["acceptance"].update(
                maximum_final_post_solver_joint_speed_rad_s=0.051
            ),
            "safety bound",
        ),
        (
            lambda document: document["acceptance"].update(
                minimum_body_lift_m=0.020
            ),
            "at least 40 mm",
        ),
        (
            lambda document: document["acceptance"].update(
                require_zero_final_plug_table_contacts=False
            ),
            "must be enabled",
        ),
    ),
)
def test_pick_v1_rejects_unsafe_or_malformed_values(
    tmp_path, mutator, message
):
    with pytest.raises(ValueError, match=message):
        load_tabletop_pick_config(_invalid_document(tmp_path, mutator))


def test_pick_config_is_immutable_and_json_safe():
    config = _config()
    json.dumps(config.as_dict(), allow_nan=False, sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        config.motion.closure_duration_s = 0.0


def test_import_is_pure_without_isaac_omni_or_pxr():
    script = r'''
import json
import sys
from kcg_connector.tabletop_pick import load_tabletop_pick_config
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({"pure_pick_import": True}))
'''
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(PACKAGE_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == {"pure_pick_import": True}
