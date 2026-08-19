'''Field-addressed deterministic randomization for the tabletop physical grasp.

Each field owns a frozen integer stream id and an independent ``Random``
instance.  There is no global or sequential RNG consumption, so adding a new
field never perturbs existing values and two methods that realize the same
seed receive byte-identical parameters.  No cryptographic fingerprint is
computed or stored.
'''

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import random
from typing import Any, Mapping, Sequence

RANDOMIZER_SCHEMA = "kcg_d38999_randomization_v1"

FIELD_NAMES = (
    "plug_x_offset_m",
    "plug_y_offset_m",
    "plug_yaw_deg",
    "arm_center_error_x_m",
    "arm_center_error_y_m",
    "finger_start_delay_steps",
    "table_static_friction",
    "table_dynamic_friction",
    "fingertip_static_friction",
    "fingertip_dynamic_friction",
    "plug_mass_scale",
    "center_of_mass_offset_m",
    "lift_speed_scale",
)

_FIELD_STREAM_IDS = {
    name: index + 1 for index, name in enumerate(FIELD_NAMES)
}
_FIELD_STREAM_IDS["finger_start_delay_anchor"] = len(FIELD_NAMES) + 1


@dataclass(frozen=True)
class IntervalContract:
    '''One inclusive finite interval with explicit unit in the field name.'''

    low: float
    high: float

    def __post_init__(self) -> None:
        for value in (self.low, self.high):
            if isinstance(value, bool) or not math.isfinite(
                float(value)
            ):
                raise ValueError(
                    "interval bounds must be finite non-bool numbers"
                )
        if self.low > self.high:
            raise ValueError("interval low must not exceed high")


@dataclass(frozen=True)
class RandomizationContract:
    '''Strict ranges for every randomized field of the physical grasp.'''

    schema_version: str = RANDOMIZER_SCHEMA
    plug_x_offset_m: IntervalContract = IntervalContract(-0.0005, 0.0005)
    plug_y_offset_m: IntervalContract = IntervalContract(-0.0005, 0.0005)
    plug_yaw_deg: IntervalContract = IntervalContract(-5.0, 5.0)
    arm_center_error_x_m: IntervalContract = IntervalContract(
        -0.0003, 0.0003
    )
    arm_center_error_y_m: IntervalContract = IntervalContract(
        -0.0003, 0.0003
    )
    finger_start_delay_steps: tuple[int, ...] = (0, 5, 12, 24, 36)
    table_static_friction: IntervalContract = IntervalContract(0.80, 1.00)
    table_dynamic_friction: IntervalContract = IntervalContract(0.65, 0.85)
    fingertip_static_friction: IntervalContract = IntervalContract(
        1.20, 1.60
    )
    fingertip_dynamic_friction: IntervalContract = IntervalContract(
        1.20, 1.60
    )
    plug_mass_scale: IntervalContract = IntervalContract(0.90, 1.10)
    center_of_mass_offset_m: IntervalContract = IntervalContract(
        -0.001, 0.001
    )
    lift_speed_scale: IntervalContract = IntervalContract(0.90, 1.10)

    def __post_init__(self) -> None:
        if self.schema_version != RANDOMIZER_SCHEMA:
            raise ValueError("unsupported randomization schema_version")
        if not isinstance(self.finger_start_delay_steps, tuple) or not (
            self.finger_start_delay_steps
        ):
            raise ValueError("finger delay set must be a non-empty tuple")
        for value in self.finger_start_delay_steps:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "finger delay set must contain non-negative integers"
                )
        if 0 not in self.finger_start_delay_steps:
            raise ValueError("finger delay set must contain 0")
        if len(set(self.finger_start_delay_steps)) != len(
            self.finger_start_delay_steps
        ):
            raise ValueError("finger delay set members must be unique")
        for name, interval in (
            ("table_static_friction", self.table_static_friction),
            ("table_dynamic_friction", self.table_dynamic_friction),
            ("fingertip_static_friction", self.fingertip_static_friction),
            ("fingertip_dynamic_friction", self.fingertip_dynamic_friction),
        ):
            if interval.low <= 0.0:
                raise ValueError(f"{name} must stay positive")
        # Conditional sampling contract: dynamic is drawn first, then static
        # from [max(dynamic, static_low), static_high].  That interval is
        # guaranteed non-empty only when every possible dynamic value is at
        # most static_high.  The loader rejects unsolvable ranges.
        if self.table_dynamic_friction.high > self.table_static_friction.high:
            raise ValueError(
                "table static friction range cannot guarantee static >= "
                "dynamic for every sampled dynamic value"
            )
        if (
            self.fingertip_dynamic_friction.high
            > self.fingertip_static_friction.high
        ):
            raise ValueError(
                "fingertip static friction range cannot guarantee static "
                ">= dynamic for every sampled dynamic value"
            )


@dataclass(frozen=True)
class RealizedRandomization:
    '''One fully realized, JSON-safe parameter set for a seed.'''

    schema_version: str
    seed: int
    plug_x_offset_m: float
    plug_y_offset_m: float
    plug_yaw_deg: float
    arm_center_error_x_m: float
    arm_center_error_y_m: float
    finger_start_delay_steps: tuple[int, int, int]
    finger_start_delay_raw_draws: tuple[int, int, int]
    table_static_friction: float
    table_dynamic_friction: float
    fingertip_static_friction: float
    fingertip_dynamic_friction: float
    plug_mass_scale: float
    center_of_mass_offset_m: tuple[float, float, float]
    lift_speed_scale: float
    finger_start_delay_anchor_index: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != RANDOMIZER_SCHEMA:
            raise ValueError("unsupported realized schema_version")
        if (
            isinstance(self.finger_start_delay_anchor_index, bool)
            or not isinstance(self.finger_start_delay_anchor_index, int)
            or not 0 <= self.finger_start_delay_anchor_index <= 2
        ):
            raise ValueError("finger delay anchor index must be in [0, 2]")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        scalar_fields = (
            "plug_x_offset_m",
            "plug_y_offset_m",
            "plug_yaw_deg",
            "arm_center_error_x_m",
            "arm_center_error_y_m",
            "table_static_friction",
            "table_dynamic_friction",
            "fingertip_static_friction",
            "fingertip_dynamic_friction",
            "plug_mass_scale",
            "lift_speed_scale",
        )
        for name in scalar_fields:
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"realized {name} must be finite")
        if len(self.finger_start_delay_steps) != 3:
            raise ValueError("realized finger delays need three values")
        for value in self.finger_start_delay_steps:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "realized finger delays must be non-negative integers"
                )
        if min(self.finger_start_delay_steps) != 0:
            raise ValueError(
                "anchor contract requires the anchor finger delay to be zero"
            )
        if len(self.finger_start_delay_raw_draws) != 3:
            raise ValueError("finger delay raw draws need three values")
        if self.table_static_friction < self.table_dynamic_friction:
            raise ValueError("realized table static friction is below dynamic")
        if self.fingertip_static_friction < self.fingertip_dynamic_friction:
            raise ValueError(
                "realized fingertip static friction is below dynamic"
            )
        if len(self.center_of_mass_offset_m) != 3 or not all(
            math.isfinite(float(value))
            for value in self.center_of_mass_offset_m
        ):
            raise ValueError("center_of_mass_offset_m needs three finite values")

    def canonical_payload(self) -> dict[str, Any]:
        '''Return the method-free canonical payload with explicit units.'''
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "plug_x_offset_m": self.plug_x_offset_m,
            "plug_y_offset_m": self.plug_y_offset_m,
            "plug_yaw_deg": self.plug_yaw_deg,
            "arm_center_error_x_m": self.arm_center_error_x_m,
            "arm_center_error_y_m": self.arm_center_error_y_m,
            "finger_start_delay_steps": list(self.finger_start_delay_steps),
            "finger_start_delay_raw_draws": list(
                self.finger_start_delay_raw_draws
            ),
            "finger_start_delay_anchor_index": (
                self.finger_start_delay_anchor_index
            ),
            "table_static_friction": self.table_static_friction,
            "table_dynamic_friction": self.table_dynamic_friction,
            "fingertip_static_friction": self.fingertip_static_friction,
            "fingertip_dynamic_friction": self.fingertip_dynamic_friction,
            "plug_mass_scale": self.plug_mass_scale,
            "center_of_mass_offset_m": list(self.center_of_mass_offset_m),
            "lift_speed_scale": self.lift_speed_scale,
        }

    def payload_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

def _field_fraction(seed: int, field_name: str, component: int = 0) -> float:
    '''Deterministic [0,1) fraction addressed by schema/seed/field/component.'''
    try:
        stream_id = _FIELD_STREAM_IDS[field_name]
    except KeyError as error:
        raise ValueError(f"unregistered randomization field: {field_name}") from error
    if isinstance(component, bool) or not isinstance(component, int) or component < 0:
        raise ValueError("randomization component must be a non-negative integer")
    stream_seed = (
        int(seed) * 1_000_003
        + stream_id * 10_007
        + component * 101
        + 0x4B4347
    )
    return random.Random(stream_seed).random()


def _sample_interval(interval: IntervalContract, fraction: float) -> float:
    return interval.low + fraction * (interval.high - interval.low)


def _sample_discrete(values: Sequence[int], fraction: float) -> int:
    index = min(len(values) - 1, int(fraction * len(values)))
    return int(values[index])


def realize_randomization(
    contract: RandomizationContract, seed: int
) -> RealizedRandomization:
    '''Realize every field for one seed with field-addressed sampling.

    Finger delays use the anchor contract: a field-addressed stream selects
    one anchor finger whose realized delay is exactly 0, and the other two
    fingers draw directly from the configured discrete set.  Every realized
    delay is therefore an original member of the configured set (the frozen
    physical timing slots), never a difference of two draws.
    '''
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    fraction = lambda name, component=0: _field_fraction(  # noqa: E731
        seed, name, component
    )

    anchor_index = int(
        fraction("finger_start_delay_anchor") * 3.0
    )
    anchor_index = min(2, anchor_index)
    raw_delays = []
    delays = []
    for index in range(3):
        if index == anchor_index:
            draw = 0
        else:
            draw = _sample_discrete(
                contract.finger_start_delay_steps,
                fraction("finger_start_delay_steps", index),
            )
        raw_delays.append(draw)
        delays.append(draw)

    table_dynamic = _sample_interval(
        contract.table_dynamic_friction, fraction("table_dynamic_friction")
    )
    table_static = _sample_interval(
        IntervalContract(
            max(table_dynamic, contract.table_static_friction.low),
            contract.table_static_friction.high,
        ),
        fraction("table_static_friction"),
    )
    fingertip_dynamic = _sample_interval(
        contract.fingertip_dynamic_friction,
        fraction("fingertip_dynamic_friction"),
    )
    fingertip_static = _sample_interval(
        IntervalContract(
            max(fingertip_dynamic, contract.fingertip_static_friction.low),
            contract.fingertip_static_friction.high,
        ),
        fraction("fingertip_static_friction"),
    )

    return RealizedRandomization(
        schema_version=contract.schema_version,
        seed=seed,
        plug_x_offset_m=_sample_interval(
            contract.plug_x_offset_m, fraction("plug_x_offset_m")
        ),
        plug_y_offset_m=_sample_interval(
            contract.plug_y_offset_m, fraction("plug_y_offset_m")
        ),
        plug_yaw_deg=_sample_interval(
            contract.plug_yaw_deg, fraction("plug_yaw_deg")
        ),
        arm_center_error_x_m=_sample_interval(
            contract.arm_center_error_x_m, fraction("arm_center_error_x_m")
        ),
        arm_center_error_y_m=_sample_interval(
            contract.arm_center_error_y_m, fraction("arm_center_error_y_m")
        ),
        finger_start_delay_steps=tuple(delays),
        finger_start_delay_raw_draws=tuple(raw_delays),
        finger_start_delay_anchor_index=anchor_index,
        table_static_friction=table_static,
        table_dynamic_friction=table_dynamic,
        fingertip_static_friction=fingertip_static,
        fingertip_dynamic_friction=fingertip_dynamic,
        plug_mass_scale=_sample_interval(
            contract.plug_mass_scale, fraction("plug_mass_scale")
        ),
        center_of_mass_offset_m=tuple(
            _sample_interval(
                contract.center_of_mass_offset_m,
                fraction("center_of_mass_offset_m", axis),
            )
            for axis in range(3)
        ),
        lift_speed_scale=_sample_interval(
            contract.lift_speed_scale, fraction("lift_speed_scale")
        ),
    )


def validate_realized(
    contract: RandomizationContract, realized: RealizedRandomization
) -> None:
    '''Assert the realized values stay inside the configured contract.'''
    checks = (
        ("plug_x_offset_m", contract.plug_x_offset_m),
        ("plug_y_offset_m", contract.plug_y_offset_m),
        ("plug_yaw_deg", contract.plug_yaw_deg),
        ("arm_center_error_x_m", contract.arm_center_error_x_m),
        ("arm_center_error_y_m", contract.arm_center_error_y_m),
        ("table_dynamic_friction", contract.table_dynamic_friction),
        ("fingertip_dynamic_friction", contract.fingertip_dynamic_friction),
        ("plug_mass_scale", contract.plug_mass_scale),
        ("lift_speed_scale", contract.lift_speed_scale),
    )
    for name, interval in checks:
        value = float(getattr(realized, name))
        if not interval.low <= value <= interval.high:
            raise ValueError(f"realized {name} escapes its configured range")
    for name, interval in (
        ("table_static_friction", contract.table_static_friction),
        ("fingertip_static_friction", contract.fingertip_static_friction),
    ):
        value = float(getattr(realized, name))
        dynamic = float(
            getattr(realized, name.replace("static", "dynamic"))
        )
        if not max(dynamic, interval.low) <= value <= interval.high:
            raise ValueError(
                f"realized {name} violates the conditional static range"
            )
    for axis, value in enumerate(realized.center_of_mass_offset_m):
        if not (
            contract.center_of_mass_offset_m.low
            <= float(value)
            <= contract.center_of_mass_offset_m.high
        ):
            raise ValueError("realized COM offset escapes its configured range")
    delay_set = set(contract.finger_start_delay_steps)
    for value in realized.finger_start_delay_steps:
        if value < 0:
            raise ValueError("realized finger delay is negative")
        if value not in delay_set:
            raise ValueError(
                "realized finger delay is not a member of the configured "
                "discrete set"
            )
    for value in realized.finger_start_delay_raw_draws:
        if value not in delay_set:
            raise ValueError(
                "realized finger delay raw draw is not a member of the "
                "configured discrete set"
            )
    anchor = realized.finger_start_delay_anchor_index
    if not 0 <= anchor <= 2:
        raise ValueError("finger delay anchor index must be in [0, 2]")
    if realized.finger_start_delay_steps[anchor] != 0:
        raise ValueError(
            "the anchor finger delay must be exactly zero"
        )


# Active-field policy per mode.  The method (synchronous vs sequential) does
# not change field activity today; the mode does.  zero-lift-hold marks
# lift_speed_scale inactive because no lift runs; single-finger marks both
# finger delays and lift speed inactive because only one finger moves and no
# lift runs.
ACTIVE_FIELDS_BY_MODE: Mapping[str, frozenset[str]] = {
    "staged": frozenset(FIELD_NAMES),
    "zero-lift-hold": frozenset(FIELD_NAMES) - {"lift_speed_scale"},
    "single-finger": frozenset(FIELD_NAMES)
    - {"finger_start_delay_steps", "lift_speed_scale"},
}

VALID_MODES = tuple(ACTIVE_FIELDS_BY_MODE)


def active_fields(mode: str) -> frozenset[str]:
    if mode not in ACTIVE_FIELDS_BY_MODE:
        raise ValueError(f"unknown randomization mode {mode!r}")
    return ACTIVE_FIELDS_BY_MODE[mode]


def distinct_active_fields(
    first: RealizedRandomization,
    second: RealizedRandomization,
    mode: str,
) -> tuple[str, ...]:
    '''Return the active fields that differ between two realizations.

    Raises ValueError when the two seeds realize an identical active scene,
    because that would silently turn n=2 into n=1 evidence.
    '''
    fields = active_fields(mode)
    differences = tuple(
        sorted(
            name
            for name in fields
            if getattr(first, name) != getattr(second, name)
        )
    )
    if not differences:
        raise ValueError(
            f"seeds {first.seed} and {second.seed} realize the same active "
            f"scene for mode {mode!r}"
        )
    return differences
