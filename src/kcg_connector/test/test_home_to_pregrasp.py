"""Pure tests for deterministic Home-to-pregrasp motion v1."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from kcg_connector.home_to_pregrasp import (
    EXPECTED_ACTIVE_HAND_JOINT_NAMES,
    EXPECTED_ARM_JOINT_NAMES,
    PREGRASP_SCHEMA_VERSION,
    interpolate_segment,
    load_home_to_pregrasp_config,
    minimum_jerk_blend,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config/connector_home_to_pregrasp_v1.yaml"


def _config():
    return load_home_to_pregrasp_config(CONFIG_PATH)


def _invalid_document(tmp_path, mutator):
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutator(document)
    path = tmp_path / "invalid_pregrasp.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_exact_home_open_hand_and_three_stage_contract():
    config = _config()
    assert config.schema_version == PREGRASP_SCHEMA_VERSION
    assert config.robot.arm_joint_names == EXPECTED_ARM_JOINT_NAMES
    assert (
        config.robot.active_hand_joint_names
        == EXPECTED_ACTIVE_HAND_JOINT_NAMES
    )
    assert config.robot.home_arm_rad == pytest.approx((0.0,) * 7)
    assert config.robot.open_hand_rad == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert config.motion.hand_open_duration_s == pytest.approx(16.0)
    assert config.robot.arm_stiffness == pytest.approx(24000.0)
    assert config.robot.arm_damping == pytest.approx(400.0)
    assert (
        config.acceptance.maximum_observed_arm_tracking_error_rad
        == pytest.approx(0.020)
    )
    assert (
        config.acceptance.maximum_observed_hand_tracking_error_rad
        == pytest.approx(0.020)
    )
    assert [segment.name for segment in config.motion.segments] == [
        "home_to_safe_mid",
        "safe_mid_to_high_approach",
        "high_approach_to_pregrasp",
    ]
    assert [segment.duration_s for segment in config.motion.segments] == (
        pytest.approx([6.2, 3.7, 2.5])
    )


def test_pregrasp_target_and_final_seed_are_exact():
    config = _config()
    assert config.motion.target_tcp_position_m == pytest.approx(
        (0.520, -0.210, 0.360)
    )
    assert config.motion.target_tcp_down_axis_world == pytest.approx(
        (0.0, 0.0, -1.0)
    )
    assert config.motion.segments[-1].target_arm_rad == pytest.approx(
        (
            -0.226630425,
            0.483930143,
            -0.343343557,
            -0.710155158,
            0.170575717,
            1.966619177,
            -0.082977338,
        )
    )


def test_minimum_jerk_endpoints_monotonicity_and_zero_endpoint_slope():
    samples = [minimum_jerk_blend(index / 100.0) for index in range(101)]
    assert samples[0] == 0.0
    assert samples[-1] == 1.0
    assert all(first <= second for first, second in zip(samples, samples[1:]))
    epsilon = 1.0e-5
    assert minimum_jerk_blend(epsilon) / epsilon < 1.0e-7
    assert (
        1.0 - minimum_jerk_blend(1.0 - epsilon)
    ) / epsilon < 1.0e-7


def test_segment_interpolation_preserves_exact_endpoints():
    start = (0.0,) * 7
    target = _config().motion.segments[0].target_arm_rad
    assert interpolate_segment(start, target, 0.0) == pytest.approx(start)
    assert interpolate_segment(start, target, 1.0) == pytest.approx(target)
    midpoint = interpolate_segment(start, target, 0.5)
    assert midpoint == pytest.approx(tuple(value * 0.5 for value in target))


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
            lambda document: document["robot"].update(
                home_arm_rad=[0, 0, 0, 0, 0, 0, 0.1]
            ),
            "Home arm",
        ),
        (
            lambda document: document["robot"].update(
                open_hand_rad=[0, 0, 0, 0]
            ),
            "open hand",
        ),
        (
            lambda document: document["motion"].update(
                interpolation="linear"
            ),
            "minimum_jerk",
        ),
        (
            lambda document: document["motion"].update(
                hand_open_duration_s=0.5
            ),
            "at least two seconds",
        ),
        (
            lambda document: document["motion"]["segments"][0].update(
                duration_s=1.9
            ),
            "at least 2.5 seconds",
        ),
        (
            lambda document: document["motion"]["segments"][0].update(
                duration_s=6.2001
            ),
            "whole 240 Hz steps",
        ),
        (
            lambda document: document["motion"]["segments"][1].update(
                target_arm_rad=[0, 0, 0, 0, 0, 0, 0]
            ),
            "waypoints are not canonical",
        ),
        (
            lambda document: document["motion"].update(
                target_tcp_position_m=[0.52, -0.21, 0.30]
            ),
            "pregrasp",
        ),
        (
            lambda document: document["motion"].update(
                target_tcp_down_axis_world=[0, 0, -2]
            ),
            "unit length",
        ),
        (
            lambda document: document["acceptance"].update(
                maximum_tcp_position_error_m=0.01
            ),
            "safety bound",
        ),
        (
            lambda document: document["acceptance"].update(
                require_zero_robot_connector_contacts=False
            ),
            "must be enabled",
        ),
        (
            lambda document: document["scene"].update(
                articulation_prim_path="/World/HandArm/Wrong"
            ),
            "articulation path is not canonical",
        ),
        (
            lambda document: document["robot"].update(
                arm_stiffness=10000
            ),
            "PD gains are not canonical",
        ),
    ),
)
def test_contract_rejects_unsafe_or_malformed_values(
    tmp_path, mutator, message
):
    with pytest.raises(ValueError, match=message):
        load_home_to_pregrasp_config(
            _invalid_document(tmp_path, mutator)
        )


def test_config_is_immutable_and_json_safe():
    config = _config()
    json.dumps(config.as_dict(), allow_nan=False, sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        config.motion.hold_duration_s = 0.0


def test_import_does_not_load_isaac_omni_or_pxr():
    script = r'''
import json
import sys
from kcg_connector.home_to_pregrasp import load_home_to_pregrasp_config
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({"pure_import": True}))
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
    assert json.loads(result.stdout) == {"pure_import": True}
