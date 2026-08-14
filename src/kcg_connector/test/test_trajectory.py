import math
from pathlib import Path

import pytest
import yaml

from kcg_connector.trajectory import (
    Q7Action,
    bounded_setpoints,
    helical_setpoints,
    load_q7_twist_config,
    plan_q7_segmented_twist,
)


CONFIG_PATH = Path(__file__).parents[1] / "config" / "connector_task.yaml"


def test_bounded_setpoints_include_exact_endpoints():
    points = bounded_setpoints(0.0, 0.010, 0.003)
    assert points[0] == 0.0
    assert points[-1] == 0.010
    assert max(abs(b - a) for a, b in zip(points, points[1:])) <= 0.003


def test_bounded_setpoints_support_reverse_motion():
    points = bounded_setpoints(1.0, -1.0, 0.6)
    assert points[0] == 1.0
    assert points[-1] == -1.0
    assert all(b < a for a, b in zip(points, points[1:]))


def test_helical_setpoints_obey_the_relation_and_end_exactly():
    lead = 0.004
    target = 2.0 * math.pi
    points = helical_setpoints(target, lead, math.radians(7.0))
    assert points[0].coupling_angle == 0.0
    assert points[-1].coupling_angle == target
    assert points[-1].axial_lock_travel == pytest.approx(lead)
    for point in points:
        expected = lead * point.coupling_angle / (2.0 * math.pi)
        assert point.axial_lock_travel == pytest.approx(expected)


@pytest.mark.parametrize(
    "arguments",
    [
        (0.0, 0.004, 0.1),
        (math.nan, 0.004, 0.1),
        (math.pi, 0.0, 0.1),
        (math.pi, 0.004, 0.0),
    ],
)
def test_invalid_helical_schedule_is_rejected(arguments):
    with pytest.raises(ValueError):
        helical_setpoints(*arguments)


def test_q7_full_revolution_is_three_conservative_neutral_strokes():
    target = math.radians(360.0)
    lower = -2.5
    upper = 2.5
    initial = math.radians(-3.09)
    maximum_segment = math.radians(120.0)

    segments = plan_q7_segmented_twist(
        target,
        lower,
        upper,
        initial_q7=initial,
        maximum_segment_angle=maximum_segment,
    )

    assert [segment.action for segment in segments] == [
        Q7Action.GRIP,
        Q7Action.TWIST,
        Q7Action.RELEASE,
        Q7Action.REWIND,
        Q7Action.REGRIP,
        Q7Action.TWIST,
        Q7Action.RELEASE,
        Q7Action.REWIND,
        Q7Action.REGRIP,
        Q7Action.TWIST,
    ]
    for segment in segments:
        assert lower <= segment.q7_start <= upper
        assert lower <= segment.q7_end <= upper
    for previous, following in zip(segments, segments[1:]):
        assert following.q7_start == previous.q7_end

    twists = [
        segment for segment in segments if segment.action == Q7Action.TWIST
    ]
    assert len(twists) == 3
    assert all(twist.q7_start == initial for twist in twists)
    assert all(
        twist.connector_angle_delta == pytest.approx(maximum_segment)
        for twist in twists
    )
    assert twists[-1].cumulative_connector_angle == target
    cumulative_twist = math.fsum(
        segment.connector_angle_delta for segment in twists
    )
    assert cumulative_twist == pytest.approx(target)


def test_yaml_q7_config_plans_360_as_three_120_degree_strokes():
    config = load_q7_twist_config(CONFIG_PATH)
    target = math.radians(360.0)
    initial = math.radians(-3.09)

    segments = config.plan(target, initial_q7=initial)
    twists = [
        segment for segment in segments if segment.action == Q7Action.TWIST
    ]

    assert config.safe_lower_rad == -2.5
    assert config.safe_upper_rad == 2.5
    assert config.tightening_direction == -1
    assert config.maximum_segment_angle_rad == pytest.approx(
        math.radians(120.0)
    )
    assert len(twists) == 3
    assert all(twist.q7_end < twist.q7_start for twist in twists)
    twist_degrees = [
        math.degrees(segment.connector_angle_delta) for segment in twists
    ]
    assert twist_degrees == pytest.approx([120.0, 120.0, 120.0])
    assert twists[-1].cumulative_connector_angle == target


def test_yaml_q7_config_rejects_insufficient_directional_capacity():
    config = load_q7_twist_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="directional capacity"):
        config.plan(
            math.radians(360.0),
            initial_q7=config.safe_lower_rad + math.radians(100.0),
        )


@pytest.mark.parametrize(
    "override",
    [
        {"safe_lower_rad": 2.5},
        {"safe_upper_rad": -2.5},
        {"maximum_segment_degrees": 0.0},
        {"probe_degrees": 0.0},
        {"probe_speed_degrees_per_second": 0.0},
        {"maximum_speed": 0.0},
        {"regrasp_clearance_m": 0.0},
        {"tightening_direction": 0},
        {"tightening_direction": 2},
        {"tightening_direction": 1.5},
        {"maximum_segment_degrees": 300.0},
        {"probe_degrees": 121.0},
        {"probe_speed_degrees_per_second": 21.0},
    ],
)
def test_invalid_yaml_q7_config_is_rejected(tmp_path, override):
    values = {
        "safe_lower_rad": -2.5,
        "safe_upper_rad": 2.5,
        "tightening_direction": -1,
        "maximum_segment_degrees": 120.0,
        "probe_degrees": 20.0,
        "probe_speed_degrees_per_second": 10.0,
        "maximum_speed": 20.0,
        "regrasp_clearance_m": 0.00075,
    }
    values.update(override)
    config_path = tmp_path / "connector_task.yaml"
    config_path.write_text(
        yaml.safe_dump({"q7_twist": values}), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_q7_twist_config(config_path)


def test_q7_rewind_does_not_change_connector_angle():
    initial = math.radians(-3.09)
    segments = plan_q7_segmented_twist(
        math.radians(360.0),
        -2.5,
        2.5,
        initial_q7=initial,
        maximum_segment_angle=math.radians(120.0),
    )
    rewinds = [
        segment for segment in segments if segment.action == Q7Action.REWIND
    ]
    assert len(rewinds) == 2
    assert all(rewind.q7_end == initial for rewind in rewinds)
    assert all(rewind.connector_angle_delta == 0.0 for rewind in rewinds)


def test_q7_short_twist_needs_no_release_cycle():
    target = math.radians(30.0)
    segments = plan_q7_segmented_twist(
        target,
        math.radians(-175.0),
        math.radians(175.0),
        safety_margin=math.radians(5.0),
        initial_q7=math.radians(10.0),
    )
    assert [segment.action for segment in segments] == [
        Q7Action.GRIP,
        Q7Action.TWIST,
    ]
    assert segments[-1].q7_end == pytest.approx(math.radians(40.0))
    assert segments[-1].cumulative_connector_angle == target


def test_q7_negative_twist_also_rewinds_to_neutral():
    target = math.radians(-360.0)
    lower = math.radians(-175.0)
    upper = math.radians(175.0)
    margin = math.radians(5.0)
    segments = plan_q7_segmented_twist(
        target,
        lower,
        upper,
        safety_margin=margin,
        initial_q7=0.0,
        maximum_segment_angle=math.radians(120.0),
    )
    twists = [
        segment for segment in segments if segment.action == Q7Action.TWIST
    ]
    rewinds = [
        segment for segment in segments if segment.action == Q7Action.REWIND
    ]
    assert all(rewind.q7_end == 0.0 for rewind in rewinds)
    assert all(segment.connector_angle_delta < 0.0 for segment in twists)
    assert twists[-1].cumulative_connector_angle == target
    cumulative_twist = math.fsum(
        segment.connector_angle_delta for segment in twists
    )
    assert cumulative_twist == pytest.approx(target)


def test_zero_q7_target_has_no_actions():
    assert plan_q7_segmented_twist(0.0, -1.0, 1.0) == ()


def test_uncapped_q7_schedule_reuses_all_neutral_headroom():
    target = math.radians(1000.0)
    lower = math.radians(-175.0)
    upper = math.radians(175.0)
    margin = math.radians(5.0)
    initial = math.radians(10.0)
    segments = plan_q7_segmented_twist(
        target,
        lower,
        upper,
        safety_margin=margin,
        initial_q7=initial,
    )
    twists = [
        segment for segment in segments if segment.action == Q7Action.TWIST
    ]
    rewinds = [
        segment for segment in segments if segment.action == Q7Action.REWIND
    ]
    assert len(twists) > 1
    assert all(twist.q7_start == initial for twist in twists)
    assert all(rewind.q7_end == initial for rewind in rewinds)
    assert twists[-1].cumulative_connector_angle == target


@pytest.mark.parametrize(
    "arguments",
    [
        (math.nan, -1.0, 1.0, 0.0, 0.0),
        (1.0, -math.inf, 1.0, 0.0, 0.0),
        (1.0, -1.0, math.inf, 0.0, 0.0),
        (1.0, -1.0, 1.0, math.nan, 0.0),
        (1.0, -1.0, 1.0, 0.0, math.inf),
        (1.0, 1.0, 1.0, 0.0, 1.0),
        (1.0, 2.0, 1.0, 0.0, 1.5),
        (1.0, -1.0, 1.0, -0.1, 0.0),
        (1.0, -1.0, 1.0, 1.0, 0.0),
        (1.0, -1.0, 1.0, 0.1, 0.95),
        (1.0, -1.0, 1.0, 0.0, 1.0),
    ],
)
def test_invalid_q7_schedule_is_rejected(arguments):
    with pytest.raises(ValueError):
        plan_q7_segmented_twist(*arguments)


@pytest.mark.parametrize("maximum_segment", [0.0, -0.1, math.nan, math.inf])
def test_invalid_q7_segment_cap_is_rejected(maximum_segment):
    with pytest.raises(ValueError):
        plan_q7_segmented_twist(
            1.0,
            -2.5,
            2.5,
            initial_q7=0.0,
            maximum_segment_angle=maximum_segment,
        )
