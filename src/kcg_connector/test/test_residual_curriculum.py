"""Pure tests for the versioned 20/60/120 residual curriculum."""

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import yaml

from kcg_connector.residual_curriculum import (
    CURRICULUM_SCHEMA_VERSION,
    CURRICULUM_STAGE_NAMES,
    DEFAULT_Q7_COMMAND_RESERVE_RAD,
    load_connector_residual_curriculum,
    resolve_stage,
    resolved_stage_document,
)
from kcg_connector.residual_rl import load_connector_residual_config


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_PATH = (
    PACKAGE_ROOT / "config/connector_residual_curriculum_v1.yaml"
)
TASK_PATH = PACKAGE_ROOT / "config/connector_task.yaml"
INITIAL_Q7_RAD = -0.046556


def _curriculum():
    return load_connector_residual_curriculum(CURRICULUM_PATH)


def _base():
    return load_connector_residual_config(TASK_PATH)


def _invalid_document(tmp_path, mutator):
    document = yaml.safe_load(CURRICULUM_PATH.read_text(encoding="utf-8"))
    mutator(document)
    path = tmp_path / "invalid_curriculum.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_versioned_three_stage_schedule_is_exact_and_monotonic():
    curriculum = _curriculum()
    assert curriculum.schema_version == CURRICULUM_SCHEMA_VERSION
    assert curriculum.interface_version == (
        "kcg_connector_twist_residual_v0"
    )
    assert curriculum.default_stage_name == "stage20"
    assert tuple(stage.name for stage in curriculum.stages) == (
        CURRICULUM_STAGE_NAMES
    )
    assert [
        math.degrees(stage.target_angle_rad)
        for stage in curriculum.stages
    ] == pytest.approx([20.0, 60.0, 120.0])
    assert [
        stage.success_hold_duration_s for stage in curriculum.stages
    ] == pytest.approx([0.5, 1.0, 2.0])
    assert [
        stage.maximum_episode_steps for stage in curriculum.stages
    ] == [40, 100, 180]
    assert [
        stage.minimum_required_episode_steps
        for stage in curriculum.stages
    ] == [40, 95, 180]
    assert [
        stage.minimum_axial_progress_fraction
        for stage in curriculum.stages
    ] == pytest.approx([0.75, 0.90, 0.90])
    assert curriculum.maximum_helical_error_m == pytest.approx(0.0001)
    assert math.degrees(
        curriculum.success_angle_tolerance_rad
    ) == pytest.approx(0.5)
    assert curriculum.q7_command_reserve_rad == pytest.approx(
        DEFAULT_Q7_COMMAND_RESERVE_RAD
    )


@pytest.mark.parametrize("stage_name", CURRICULUM_STAGE_NAMES)
def test_current_q7_start_resolves_every_stage_without_changing_v0_shape(
    stage_name,
):
    base = _base()
    resolved = resolve_stage(
        base,
        stage_name,
        INITIAL_Q7_RAD,
        curriculum=_curriculum(),
    )
    assert resolved.residual_config.interface_version == (
        base.interface_version
    )
    assert resolved.residual_config.clamp_joint_names == (
        base.clamp_joint_names
    )
    assert resolved.residual_config.clamp_position_residual_limits_rad == (
        base.clamp_position_residual_limits_rad
    )
    assert resolved.residual_config.success_hold_duration_s == (
        resolved.stage.success_hold_duration_s
    )
    assert resolved.residual_config.helical_error_tolerance_m == (
        pytest.approx(0.0001)
    )
    assert resolved.residual_config.minimum_axial_progress_fraction == (
        pytest.approx(resolved.stage.minimum_axial_progress_fraction)
    )
    assert math.degrees(
        resolved.residual_config.success_angle_tolerance_rad
    ) == pytest.approx(0.5)
    assert resolved.stage.maximum_episode_steps >= (
        resolved.stage.minimum_required_episode_steps
    )
    assert base.target_angle_rad == pytest.approx(math.radians(20.0))
    assert base.helical_error_tolerance_m == pytest.approx(0.0005)


def test_stage120_endpoint_and_headroom_are_explicit_and_json_safe():
    resolved = resolve_stage(
        _base(),
        "stage120",
        INITIAL_Q7_RAD,
        curriculum=_curriculum(),
    )
    expected_endpoint = INITIAL_Q7_RAD - math.radians(120.0)
    expected_lower_headroom = expected_endpoint - _base().q7_safe_lower_rad
    assert resolved.planned_final_q7_rad == pytest.approx(expected_endpoint)
    assert resolved.lower_headroom_rad == pytest.approx(
        expected_lower_headroom
    )
    assert math.degrees(resolved.lower_headroom_rad) == pytest.approx(
        20.5719864717
    )
    assert resolved.lower_headroom_rad >= math.radians(10.0)
    document = resolved_stage_document(resolved)
    json.dumps(document, allow_nan=False)
    assert document["stage_name"] == "stage120"
    assert document["target_segment_degrees"] == pytest.approx(120.0)
    assert document["maximum_episode_steps"] == 180
    assert document["minimum_axial_progress_fraction"] == pytest.approx(0.9)
    assert document["maximum_helical_error_m"] == pytest.approx(0.0001)
    assert document["success_angle_tolerance_degrees"] == pytest.approx(0.5)
    assert document["q7_command_reserve_degrees"] == pytest.approx(10.0)
    assert isinstance(
        document["resolved_residual_config"]["clamp_joint_names"], list
    )


def test_none_stage_selects_stage20_default():
    resolved = resolve_stage(
        _base(), None, INITIAL_Q7_RAD, curriculum=_curriculum()
    )
    assert resolved.stage.name == "stage20"


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        resolve_stage(
            _base(), "stage360", INITIAL_Q7_RAD, curriculum=_curriculum()
        )


def test_reversed_tightening_direction_is_rejected():
    reversed_config = replace(_base(), tightening_direction=1)
    with pytest.raises(ValueError, match="reversed"):
        resolve_stage(
            reversed_config,
            "stage120",
            INITIAL_Q7_RAD,
            curriculum=_curriculum(),
        )


def test_insufficient_directional_capacity_is_rejected():
    narrowed = replace(_base(), q7_safe_lower_rad=-2.0)
    with pytest.raises(ValueError, match="endpoint"):
        resolve_stage(
            narrowed,
            "stage120",
            INITIAL_Q7_RAD,
            curriculum=_curriculum(),
        )


def test_initial_q7_must_also_have_the_reserved_headroom():
    with pytest.raises(ValueError, match="initial q7"):
        resolve_stage(
            _base(),
            "stage20",
            -2.40,
            curriculum=_curriculum(),
        )


def test_reserve_below_ten_degrees_is_rejected():
    with pytest.raises(ValueError, match="at least 10"):
        resolve_stage(
            _base(),
            "stage20",
            INITIAL_Q7_RAD,
            reserve=math.radians(9.99),
            curriculum=_curriculum(),
        )


@pytest.mark.parametrize(
    "mutator,message",
    (
        (
            lambda document: document.update({"unexpected": True}),
            "keys are invalid",
        ),
        (
            lambda document: document.update({"schema_version": "wrong"}),
            "unsupported",
        ),
        (
            lambda document: document["contract"].update(
                {"policy_rate_hz": 20.0}
            ),
            "10 Hz",
        ),
        (
            lambda document: document["stages"]["stage60"].update(
                {"target_segment_degrees": 10.0}
            ),
            "increase strictly",
        ),
        (
            lambda document: document["stages"]["stage120"].update(
                {"target_segment_degrees": 121.0}
            ),
            "single-stroke range",
        ),
        (
            lambda document: document["stages"]["stage120"].update(
                {"maximum_episode_steps": 179}
            ),
            "10 Hz margin",
        ),
        (
            lambda document: document["stages"]["stage60"].update(
                {"minimum_axial_progress_fraction": 0.0}
            ),
            r"in \(0, 1\]",
        ),
        (
            lambda document: document["acceptance"].update(
                {"maximum_helical_error_m": 0.0}
            ),
            "helical error must be positive",
        ),
        (
            lambda document: document["acceptance"].update(
                {"success_angle_tolerance_degrees": float("nan")}
            ),
            "must be finite",
        ),
        (
            lambda document: document["acceptance"].update(
                {"q7_command_reserve_degrees": 9.0}
            ),
            "at least 10",
        ),
    ),
)
def test_invalid_curriculum_fails_closed(
    tmp_path, mutator, message
):
    path = _invalid_document(tmp_path, mutator)
    with pytest.raises(ValueError, match=message):
        load_connector_residual_curriculum(path)


def test_base_rate_and_minimum_speed_must_match_curriculum():
    with pytest.raises(ValueError, match="policy rate"):
        resolve_stage(
            replace(_base(), policy_rate_hz=20.0),
            "stage20",
            INITIAL_Q7_RAD,
            curriculum=_curriculum(),
        )
    with pytest.raises(ValueError, match="minimum tightening speed"):
        resolve_stage(
            replace(
                _base(),
                q7_speed_residual_rad_s=math.radians(3.0),
            ),
            "stage20",
            INITIAL_Q7_RAD,
            curriculum=_curriculum(),
        )
