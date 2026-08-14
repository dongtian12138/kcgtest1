"""Pure scheduling and evidence checks for the D38999 360 degree proxy.

The module deliberately contains no Isaac, ROS, or USD imports.  The physical
runner owns contacts and dynamics; this layer only reuses the generic q7
segmented planner and rejects incomplete or internally inconsistent reports.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence

from .trajectory import Q7Action, Q7ActionSegment, plan_q7_segmented_twist


STROKE_ANGLE_RAD = math.tau / 3.0
TARGET_NUT_PROGRESS_RAD = math.tau
EXPECTED_STROKE_COUNT = 3
EXPECTED_REWIND_COUNT = 2


@dataclass(frozen=True)
class D38999FullRotationPlan:
    """A q7 plan plus task-space expectations for the proxy connector."""

    segments: tuple[Q7ActionSegment, ...]
    stroke_count: int
    rewind_count: int
    target_nut_progress_rad: float
    expected_axial_travel_m: float


@dataclass(frozen=True)
class D38999FullRotationEvidence:
    """Strict aggregate result computed from three physical stroke reports."""

    passed: bool
    stroke_count: int
    rewind_count: int
    cumulative_nut_progress_rad: float
    cumulative_axial_travel_m: float
    missing_or_failed: tuple[str, ...]


def build_d38999_full_rotation_plan(
    *,
    initial_q7_rad: float,
    q7_lower_limit_rad: float,
    q7_upper_limit_rad: float,
    lead_m_per_revolution: float,
) -> D38999FullRotationPlan:
    """Build exactly three tightening strokes and two physical rewinds.

    D38999 tightening uses negative q7 motion in the validated proxy while
    measured nut progress is positive.  The generic planner therefore receives
    ``-2*pi`` and this adapter keeps the sign conversion explicit.
    """

    values = (
        initial_q7_rad,
        q7_lower_limit_rad,
        q7_upper_limit_rad,
        lead_m_per_revolution,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("full-rotation plan inputs must be finite numbers")
    if float(lead_m_per_revolution) <= 0.0:
        raise ValueError("lead_m_per_revolution must be positive")

    segments = plan_q7_segmented_twist(
        target_angle=-TARGET_NUT_PROGRESS_RAD,
        q7_lower_limit=float(q7_lower_limit_rad),
        q7_upper_limit=float(q7_upper_limit_rad),
        initial_q7=float(initial_q7_rad),
        maximum_segment_angle=STROKE_ANGLE_RAD,
    )
    stroke_count = sum(
        segment.action is Q7Action.TWIST for segment in segments
    )
    rewind_count = sum(
        segment.action is Q7Action.REWIND for segment in segments
    )
    if (
        stroke_count != EXPECTED_STROKE_COUNT
        or rewind_count != EXPECTED_REWIND_COUNT
    ):
        raise ValueError(
            "q7 limits cannot realize the required three-stroke schedule"
        )
    for segment in segments:
        if segment.action is Q7Action.TWIST and not math.isclose(
            segment.connector_angle_delta,
            -STROKE_ANGLE_RAD,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "every D38999 stroke must be exactly -120 degrees"
            )

    return D38999FullRotationPlan(
        segments=segments,
        stroke_count=stroke_count,
        rewind_count=rewind_count,
        target_nut_progress_rad=TARGET_NUT_PROGRESS_RAD,
        expected_axial_travel_m=-float(lead_m_per_revolution),
    )


def _strict_real(report: Mapping[str, Any], name: str) -> float:
    value = report.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_final_seating_contact_pairs(
    records: Mapping[str, Any],
    *,
    fixed_rear_path: str,
    body_mating_root: str,
    segment_count: int = 20,
) -> bool:
    """Accept only the complete ring-to-rear-stop contact set at gap zero."""

    if not isinstance(records, Mapping):
        raise ValueError("final seating contact records must be a mapping")
    if (
        not isinstance(fixed_rear_path, str)
        or not fixed_rear_path.startswith("/")
        or not isinstance(body_mating_root, str)
        or not body_mating_root.startswith("/")
        or isinstance(segment_count, bool)
        or not isinstance(segment_count, int)
        or segment_count <= 0
    ):
        raise ValueError("final seating contact geometry is invalid")
    expected = {
        " <-> ".join(
            sorted(
                (
                    fixed_rear_path,
                    body_mating_root + f"/Segment_{index:02d}",
                )
            )
        )
        for index in range(segment_count)
    }
    if set(records) != expected:
        return False
    return all(
        not isinstance(value, bool)
        and isinstance(value, int)
        and value > 0
        for value in records.values()
    )


def evaluate_d38999_full_rotation(
    stroke_reports: Sequence[Mapping[str, Any]],
    rewind_reports: Sequence[Mapping[str, Any]],
    *,
    expected_stroke_progress_rad: float = STROKE_ANGLE_RAD,
    expected_stroke_axial_travel_m: float = -0.001,
    maximum_cumulative_nut_error_rad: float = math.radians(6.0),
    maximum_cumulative_axial_error_m: float = 0.00015,
) -> D38999FullRotationEvidence:
    """Aggregate physical reports without hiding a failed intermediate stage.

    Runtime reports remain the source of truth.  A malformed report raises a
    validation error; a well-formed but failed report returns ``passed=False``.
    This distinction prevents missing/NaN evidence from being treated as a
    normal physical failure.
    """

    if len(stroke_reports) != EXPECTED_STROKE_COUNT:
        raise ValueError("exactly three stroke reports are required")
    if len(rewind_reports) != EXPECTED_REWIND_COUNT:
        raise ValueError("exactly two rewind reports are required")
    tolerances = (
        expected_stroke_progress_rad,
        expected_stroke_axial_travel_m,
        maximum_cumulative_nut_error_rad,
        maximum_cumulative_axial_error_m,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in tolerances
    ):
        raise ValueError("full-rotation evidence limits must be finite")
    if (
        float(expected_stroke_progress_rad) <= 0.0
        or float(expected_stroke_axial_travel_m) >= 0.0
        or float(maximum_cumulative_nut_error_rad) <= 0.0
        or float(maximum_cumulative_axial_error_m) <= 0.0
    ):
        raise ValueError("full-rotation evidence limits have invalid signs")

    missing_or_failed: list[str] = []
    cumulative_nut = 0.0
    cumulative_axial = 0.0
    for index, report in enumerate(stroke_reports, start=1):
        if not isinstance(report, Mapping):
            raise ValueError("stroke report must be a mapping")
        if type(report.get("passed")) is not bool:
            raise ValueError("stroke report passed must be bool")
        nut_delta = _strict_real(report, "actual_nut_delta_rad")
        axial_delta = _strict_real(report, "actual_axial_travel_m")
        cumulative_nut += nut_delta
        cumulative_axial += axial_delta
        if report["passed"] is not True:
            missing_or_failed.append(f"stroke_{index}_failed")
        # The runner owns tight tolerances.  This layer still rejects a report
        # that claims pass with the wrong gross physical direction.
        if nut_delta <= 0.0 or axial_delta >= 0.0:
            missing_or_failed.append(f"stroke_{index}_direction")
        if abs(nut_delta - float(expected_stroke_progress_rad)) > math.radians(
            10.0
        ):
            missing_or_failed.append(f"stroke_{index}_progress")
        if abs(axial_delta - float(expected_stroke_axial_travel_m)) > 0.0002:
            missing_or_failed.append(f"stroke_{index}_axial")

    for index, report in enumerate(rewind_reports, start=1):
        if not isinstance(report, Mapping):
            raise ValueError("rewind report must be a mapping")
        if type(report.get("passed")) is not bool:
            raise ValueError("rewind report passed must be bool")
        _strict_real(report, "q7_rewind_tracking_error_rad")
        _strict_real(report, "maximum_released_nut_drift_rad")
        if report["passed"] is not True:
            missing_or_failed.append(f"rewind_{index}_failed")

    if abs(cumulative_nut - TARGET_NUT_PROGRESS_RAD) > float(
        maximum_cumulative_nut_error_rad
    ):
        missing_or_failed.append("cumulative_nut_progress")
    if abs(cumulative_axial + 0.003) > float(
        maximum_cumulative_axial_error_m
    ):
        missing_or_failed.append("cumulative_axial_travel")

    return D38999FullRotationEvidence(
        passed=not missing_or_failed,
        stroke_count=len(stroke_reports),
        rewind_count=len(rewind_reports),
        cumulative_nut_progress_rad=cumulative_nut,
        cumulative_axial_travel_m=cumulative_axial,
        missing_or_failed=tuple(missing_or_failed),
    )
