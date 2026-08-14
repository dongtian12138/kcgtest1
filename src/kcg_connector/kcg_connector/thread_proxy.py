"""Unit-safe helpers for the simplified PhysX thread coupling."""

import math


def rack_and_pinion_ratio(lead_meters, meters_per_stage_unit, direction=1):
    """Return the PhysX rack ratio in degrees per stage distance unit."""
    lead = float(lead_meters)
    meters_per_unit = float(meters_per_stage_unit)
    if not math.isfinite(lead) or lead <= 0.0:
        raise ValueError("lead_meters must be finite and positive")
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise ValueError("meters_per_stage_unit must be finite and positive")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    lead_in_stage_units = lead / meters_per_unit
    return direction * 360.0 / lead_in_stage_units
