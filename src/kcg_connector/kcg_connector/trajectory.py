"""Deterministic insertion and helical setpoints used before residual RL."""

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path

import yaml

from .geometry import helical_travel


@dataclass(frozen=True)
class HelicalSetpoint:
    """One exact coupling-angle/axial-travel pair."""

    coupling_angle: float
    axial_lock_travel: float


class Q7Action(str, Enum):
    """Discrete actions in a release-and-regrip joint-7 screw schedule."""

    GRIP = "GRIP"
    TWIST = "TWIST"
    RELEASE = "RELEASE"
    REWIND = "REWIND"
    REGRIP = "REGRIP"


@dataclass(frozen=True)
class Q7ActionSegment:
    """One atomic action; all angular values are radians.

    Only ``TWIST`` changes ``cumulative_connector_angle``.  During ``REWIND``
    the fingers are released and the engaged thread is assumed to hold the
    connector, so its angle increment is exactly zero.
    """

    action: Q7Action
    q7_start: float
    q7_end: float
    connector_angle_delta: float
    cumulative_connector_angle: float


@dataclass(frozen=True)
class Q7TwistConfig:
    """Validated q7-only control parameters, converted to SI units."""

    safe_lower_rad: float
    safe_upper_rad: float
    tightening_direction: int
    maximum_segment_angle_rad: float
    probe_angle_rad: float
    probe_speed_rad_per_second: float
    maximum_speed_rad_per_second: float
    regrasp_clearance_m: float

    def plan(self, target_angle, initial_q7):
        """Plan signed connector progress and map it onto the q7 axis.

        Positive ``target_angle`` always means tightening progress.  The
        configured direction converts that task-space convention to the
        imported robot's signed q7 motion.
        """
        target_angle = _finite_float("target_angle", target_angle)
        initial_q7 = _finite_float("initial_q7", initial_q7)
        q7_target_delta = target_angle * self.tightening_direction
        if not self.safe_lower_rad <= initial_q7 <= self.safe_upper_rad:
            raise ValueError("initial_q7 must lie inside the q7 safe window")
        if q7_target_delta != 0.0:
            directional_capacity = (
                self.safe_upper_rad - initial_q7
                if q7_target_delta > 0.0
                else initial_q7 - self.safe_lower_rad
            )
            if self.maximum_segment_angle_rad > directional_capacity:
                raise ValueError(
                    "configured maximum q7 segment exceeds the directional "
                    "capacity from initial_q7"
                )
        q7_segments = plan_q7_segmented_twist(
            target_angle=q7_target_delta,
            q7_lower_limit=self.safe_lower_rad,
            q7_upper_limit=self.safe_upper_rad,
            initial_q7=initial_q7,
            maximum_segment_angle=self.maximum_segment_angle_rad,
        )
        return tuple(
            Q7ActionSegment(
                action=segment.action,
                q7_start=segment.q7_start,
                q7_end=segment.q7_end,
                connector_angle_delta=(
                    segment.connector_angle_delta
                    * self.tightening_direction
                ),
                cumulative_connector_angle=(
                    segment.cumulative_connector_angle
                    * self.tightening_direction
                ),
            )
            for segment in q7_segments
        )


MAX_Q7_TWIST_SEGMENTS = 100_000


def bounded_setpoints(start, target, maximum_step):
    """Return endpoints with uniform increments no larger than maximum_step."""
    values = (float(start), float(target), float(maximum_step))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("setpoint arguments must be finite")
    if maximum_step <= 0.0:
        raise ValueError("maximum_step must be positive")
    distance = target - start
    step_count = max(1, math.ceil(abs(distance) / maximum_step))
    points = [
        start + distance * index / step_count
        for index in range(step_count + 1)
    ]
    points[0] = start
    points[-1] = target
    return tuple(points)


def helical_setpoints(
    target_angle, lead_per_revolution, maximum_angle_step
):
    """Create an exact, bounded screw schedule from zero to target_angle."""
    if target_angle <= 0.0 or not math.isfinite(target_angle):
        raise ValueError("target_angle must be finite and positive")
    if (
        lead_per_revolution <= 0.0
        or not math.isfinite(lead_per_revolution)
    ):
        raise ValueError("lead_per_revolution must be finite and positive")
    angles = bounded_setpoints(0.0, target_angle, maximum_angle_step)
    return tuple(
        HelicalSetpoint(
            coupling_angle=angle,
            axial_lock_travel=helical_travel(angle, lead_per_revolution),
        )
        for angle in angles
    )


def _finite_float(name, value):
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def load_q7_twist_config(config_path):
    """Load and validate conservative q7 control parameters from task YAML."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        raw = document["q7_twist"]
        config = Q7TwistConfig(
            safe_lower_rad=float(raw["safe_lower_rad"]),
            safe_upper_rad=float(raw["safe_upper_rad"]),
            tightening_direction=int(raw["tightening_direction"]),
            maximum_segment_angle_rad=math.radians(
                float(raw["maximum_segment_degrees"])
            ),
            probe_angle_rad=math.radians(float(raw["probe_degrees"])),
            probe_speed_rad_per_second=math.radians(
                float(raw["probe_speed_degrees_per_second"])
            ),
            maximum_speed_rad_per_second=math.radians(
                float(raw["maximum_speed"])
            ),
            regrasp_clearance_m=float(raw["regrasp_clearance_m"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid q7 twist config: {path}") from error

    limits = (config.safe_lower_rad, config.safe_upper_rad)
    if not all(math.isfinite(value) for value in limits):
        raise ValueError("q7 safe limits must be finite")
    if config.safe_lower_rad >= config.safe_upper_rad:
        raise ValueError(
            "q7 safe lower limit must be smaller than upper limit"
        )
    if config.tightening_direction not in (-1, 1) or float(
        raw["tightening_direction"]
    ) != config.tightening_direction:
        raise ValueError("q7 tightening direction must be exactly -1 or 1")

    positive_values = (
        config.maximum_segment_angle_rad,
        config.probe_angle_rad,
        config.probe_speed_rad_per_second,
        config.maximum_speed_rad_per_second,
        config.regrasp_clearance_m,
    )
    if not all(
        math.isfinite(value) and value > 0.0 for value in positive_values
    ):
        raise ValueError(
            "q7 twist magnitudes, speeds, and clearance must be positive"
        )

    safe_span = config.safe_upper_rad - config.safe_lower_rad
    if config.maximum_segment_angle_rad > safe_span:
        raise ValueError("maximum q7 segment exceeds the complete safe window")
    if config.probe_angle_rad > config.maximum_segment_angle_rad:
        raise ValueError("q7 probe angle must not exceed the maximum segment")
    if config.probe_speed_rad_per_second > config.maximum_speed_rad_per_second:
        raise ValueError("q7 probe speed must not exceed maximum speed")
    return config


def _estimated_twist_count(target_magnitude, stroke_capacity):
    quotient = target_magnitude / stroke_capacity
    if not math.isfinite(quotient) or quotient > MAX_Q7_TWIST_SEGMENTS:
        raise ValueError(
            "target_angle requires too many explicit q7 twist segments"
        )
    return math.ceil(quotient)


def plan_q7_segmented_twist(
    target_angle,
    q7_lower_limit,
    q7_upper_limit,
    safety_margin=0.0,
    initial_q7=0.0,
    maximum_segment_angle=None,
):
    """Plan an exact connector rotation using only ``iiwa_joint_7``.

    Args:
        target_angle: Signed cumulative connector rotation from the current
            connector state.  Positive motion increases q7; negative motion
            decreases it.
        q7_lower_limit: Mechanical lower q7 limit in radians.
        q7_upper_limit: Mechanical upper q7 limit in radians.
        safety_margin: Clearance retained inside both mechanical limits.
        initial_q7: q7 position at the start of the schedule and the neutral
            position to which every released rewind returns.
        maximum_segment_angle: Optional positive cap for the connector angle
            produced by one gripped q7 stroke.  With ``None``, each stroke may
            use all directional travel from neutral to the safe q7 limit.

    Returns:
        A tuple of :class:`Q7ActionSegment`.  The schedule starts gripped and
        ends gripped.  Whenever one q7 stroke is insufficient it emits
        ``RELEASE -> REWIND-to-neutral -> REGRIP`` before the next ``TWIST``.

    The helper assumes the threaded connector does not unwind while released.
    It is purely geometric and does not command a robot or simulate contact.
    """
    target_angle = _finite_float("target_angle", target_angle)
    q7_lower_limit = _finite_float("q7_lower_limit", q7_lower_limit)
    q7_upper_limit = _finite_float("q7_upper_limit", q7_upper_limit)
    safety_margin = _finite_float("safety_margin", safety_margin)
    initial_q7 = _finite_float("initial_q7", initial_q7)
    if maximum_segment_angle is not None:
        maximum_segment_angle = _finite_float(
            "maximum_segment_angle", maximum_segment_angle
        )

    if q7_lower_limit >= q7_upper_limit:
        raise ValueError("q7_lower_limit must be smaller than q7_upper_limit")
    if safety_margin < 0.0:
        raise ValueError("safety_margin must be nonnegative")

    safe_lower = q7_lower_limit + safety_margin
    safe_upper = q7_upper_limit - safety_margin
    if not math.isfinite(safe_lower) or not math.isfinite(safe_upper):
        raise ValueError("effective q7 limits must be finite")
    if safe_lower >= safe_upper:
        raise ValueError("safety_margin leaves no usable q7 range")
    if not safe_lower <= initial_q7 <= safe_upper:
        raise ValueError(
            "initial_q7 must lie inside the safety-adjusted limits"
        )
    if maximum_segment_angle is not None and maximum_segment_angle <= 0.0:
        raise ValueError("maximum_segment_angle must be positive")
    if target_angle == 0.0:
        return ()

    direction = 1.0 if target_angle > 0.0 else -1.0
    directional_capacity = (
        safe_upper - initial_q7
        if direction > 0.0
        else initial_q7 - safe_lower
    )
    if directional_capacity <= 0.0:
        raise ValueError(
            "initial_q7 leaves no safe q7 travel in the target direction"
        )
    stroke_capacity = directional_capacity
    if maximum_segment_angle is not None:
        stroke_capacity = min(stroke_capacity, maximum_segment_angle)
    twist_count = _estimated_twist_count(abs(target_angle), stroke_capacity)
    if twist_count > MAX_Q7_TWIST_SEGMENTS:
        raise ValueError(
            "target_angle requires too many explicit q7 twist segments"
        )

    segments = [
        Q7ActionSegment(Q7Action.GRIP, initial_q7, initial_q7, 0.0, 0.0)
    ]
    current_q7 = initial_q7
    cumulative_angle = 0.0

    while cumulative_angle != target_angle:
        remaining = target_angle - cumulative_angle
        if abs(remaining) <= stroke_capacity:
            connector_delta = remaining
            if abs(remaining) == directional_capacity:
                next_q7 = safe_upper if direction > 0.0 else safe_lower
                connector_delta = next_q7 - current_q7
            else:
                next_q7 = current_q7 + connector_delta
            next_cumulative = target_angle
        else:
            if stroke_capacity == directional_capacity:
                next_q7 = safe_upper if direction > 0.0 else safe_lower
                connector_delta = next_q7 - current_q7
            else:
                connector_delta = direction * stroke_capacity
                next_q7 = current_q7 + connector_delta
            next_cumulative = cumulative_angle + connector_delta

        if not safe_lower <= next_q7 <= safe_upper:
            raise RuntimeError("internal q7 schedule exceeded its safe limits")
        segments.append(
            Q7ActionSegment(
                Q7Action.TWIST,
                current_q7,
                next_q7,
                connector_delta,
                next_cumulative,
            )
        )
        current_q7 = next_q7
        cumulative_angle = next_cumulative

        if cumulative_angle != target_angle:
            segments.extend(
                (
                    Q7ActionSegment(
                        Q7Action.RELEASE,
                        current_q7,
                        current_q7,
                        0.0,
                        cumulative_angle,
                    ),
                    Q7ActionSegment(
                        Q7Action.REWIND,
                        current_q7,
                        initial_q7,
                        0.0,
                        cumulative_angle,
                    ),
                    Q7ActionSegment(
                        Q7Action.REGRIP,
                        initial_q7,
                        initial_q7,
                        0.0,
                        cumulative_angle,
                    ),
                )
            )
            current_q7 = initial_q7

    return tuple(segments)
