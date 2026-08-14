"""Pure regression tests for the D38999 three-stroke adapter."""

import math

import pytest

from kcg_connector.d38999_full_rotation import (
    build_d38999_full_rotation_plan,
    evaluate_d38999_full_rotation,
    validate_final_seating_contact_pairs,
)
from kcg_connector.trajectory import Q7Action


def _plan():
    return build_d38999_full_rotation_plan(
        initial_q7_rad=0.650482794,
        q7_lower_limit_rad=-2.5,
        q7_upper_limit_rad=2.5,
        lead_m_per_revolution=0.003,
    )


def _stroke(passed=True, nut=math.tau / 3.0, axial=-0.001):
    return {
        "passed": passed,
        "actual_nut_delta_rad": nut,
        "actual_axial_travel_m": axial,
    }


def _rewind(passed=True):
    return {
        "passed": passed,
        "q7_rewind_tracking_error_rad": 0.0,
        "maximum_released_nut_drift_rad": 0.0,
    }


def test_plan_reuses_generic_segmented_planner_for_three_strokes():
    plan = _plan()
    assert plan.stroke_count == 3
    assert plan.rewind_count == 2
    assert math.isclose(plan.target_nut_progress_rad, math.tau)
    assert math.isclose(plan.expected_axial_travel_m, -0.003)
    assert [segment.action for segment in plan.segments] == [
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


@pytest.mark.parametrize(
    "kwargs",
    (
        {"initial_q7_rad": float("nan")},
        {"lead_m_per_revolution": 0.0},
        {"q7_lower_limit_rad": 0.0, "q7_upper_limit_rad": 1.0},
    ),
)
def test_plan_fails_closed_for_invalid_or_insufficient_inputs(kwargs):
    values = {
        "initial_q7_rad": 0.650482794,
        "q7_lower_limit_rad": -2.5,
        "q7_upper_limit_rad": 2.5,
        "lead_m_per_revolution": 0.003,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        build_d38999_full_rotation_plan(**values)


def test_three_strokes_and_two_rewinds_aggregate_to_full_rotation():
    evidence = evaluate_d38999_full_rotation(
        [_stroke(), _stroke(), _stroke()],
        [_rewind(), _rewind()],
    )
    assert evidence.passed is True
    assert math.isclose(evidence.cumulative_nut_progress_rad, math.tau)
    assert math.isclose(evidence.cumulative_axial_travel_m, -0.003)
    assert evidence.missing_or_failed == ()


def test_intermediate_failure_cannot_be_hidden_by_final_totals():
    evidence = evaluate_d38999_full_rotation(
        [_stroke(), _stroke(passed=False), _stroke()],
        [_rewind(), _rewind()],
    )
    assert evidence.passed is False
    assert evidence.missing_or_failed == ("stroke_2_failed",)


def test_wrong_direction_and_cumulative_progress_fail():
    evidence = evaluate_d38999_full_rotation(
        [_stroke(), _stroke(nut=-math.tau / 3.0, axial=0.001), _stroke()],
        [_rewind(), _rewind()],
    )
    assert evidence.passed is False
    assert "stroke_2_direction" in evidence.missing_or_failed
    assert "cumulative_nut_progress" in evidence.missing_or_failed
    assert "cumulative_axial_travel" in evidence.missing_or_failed


@pytest.mark.parametrize(
    "strokes,rewinds",
    (([_stroke()] * 2, [_rewind()] * 2), ([_stroke()] * 3, [_rewind()])),
)
def test_aggregate_requires_exact_report_counts(strokes, rewinds):
    with pytest.raises(ValueError):
        evaluate_d38999_full_rotation(strokes, rewinds)


@pytest.mark.parametrize(
    "bad_value",
    (None, True, "1.0", float("nan"), float("inf")),
)
def test_aggregate_rejects_nonfinite_or_ambiguous_evidence(bad_value):
    report = _stroke()
    report["actual_nut_delta_rad"] = bad_value
    with pytest.raises(ValueError):
        evaluate_d38999_full_rotation(
            [report, _stroke(), _stroke()], [_rewind(), _rewind()]
        )


def _seating_records():
    return {
        " <-> ".join(
            sorted(("/Fixed/RearBody", f"/Loose/Mating/Segment_{i:02d}"))
        ): 12
        for i in range(20)
    }


def test_final_seating_requires_all_and_only_twenty_ring_pairs():
    records = _seating_records()
    assert validate_final_seating_contact_pairs(
        records,
        fixed_rear_path="/Fixed/RearBody",
        body_mating_root="/Loose/Mating",
    )
    records.pop(next(iter(records)))
    assert not validate_final_seating_contact_pairs(
        records,
        fixed_rear_path="/Fixed/RearBody",
        body_mating_root="/Loose/Mating",
    )


def test_final_seating_rejects_extra_pair_and_invalid_counts():
    records = _seating_records()
    records["/Fixed/RearBody <-> /Loose/Nut"] = 1
    assert not validate_final_seating_contact_pairs(
        records,
        fixed_rear_path="/Fixed/RearBody",
        body_mating_root="/Loose/Mating",
    )
    records = _seating_records()
    records[next(iter(records))] = True
    assert not validate_final_seating_contact_pairs(
        records,
        fixed_rear_path="/Fixed/RearBody",
        body_mating_root="/Loose/Mating",
    )
