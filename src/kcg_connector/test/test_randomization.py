'''Pure tests for field-addressed deterministic randomization.'''

from __future__ import annotations

import json
import math

import pytest

from kcg_connector.grasp.randomization import (
    IntervalContract,
    RandomizationContract,
    RealizedRandomization,
    active_fields,
    distinct_active_fields,
    realize_randomization,
    validate_realized,
)


def _contract(**overrides):
    values = {
        "plug_x_offset_m": IntervalContract(-0.0005, 0.0005),
        "plug_y_offset_m": IntervalContract(-0.0005, 0.0005),
        "plug_yaw_deg": IntervalContract(-5.0, 5.0),
        "arm_center_error_x_m": IntervalContract(-0.0003, 0.0003),
        "arm_center_error_y_m": IntervalContract(-0.0003, 0.0003),
        "finger_start_delay_steps": (0, 5, 12, 24, 36),
        "table_static_friction": IntervalContract(0.80, 1.00),
        "table_dynamic_friction": IntervalContract(0.65, 0.85),
        "fingertip_static_friction": IntervalContract(1.20, 1.60),
        "fingertip_dynamic_friction": IntervalContract(1.20, 1.60),
        "plug_mass_scale": IntervalContract(0.90, 1.10),
        "center_of_mass_offset_m": IntervalContract(-0.001, 0.001),
        "lift_speed_scale": IntervalContract(0.90, 1.10),
    }
    values.update(overrides)
    return RandomizationContract(**values)


def test_same_seed_realizes_byte_identical_payload_without_fingerprint():
    first = realize_randomization(_contract(), 7)
    second = realize_randomization(_contract(), 7)
    assert first == second
    assert first.payload_json() == second.payload_json()


def test_payload_is_method_free_and_json_safe():
    realized = realize_randomization(_contract(), 3)
    payload = json.loads(realized.payload_json())
    assert "method" not in payload
    assert payload["schema_version"] == "kcg_d38999_randomization_v1"
    assert payload["seed"] == 3
    assert all(
        math.isfinite(float(value))
        for value in payload.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def test_changing_one_field_range_does_not_perturb_other_fields():
    base = realize_randomization(_contract(), 11)
    changed_contract = _contract(
        plug_yaw_deg=IntervalContract(-20.0, 20.0)
    )
    changed = realize_randomization(changed_contract, 11)
    for name in (
        "plug_x_offset_m",
        "plug_y_offset_m",
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
    ):
        assert getattr(base, name) == getattr(changed, name), name
    assert base.plug_yaw_deg != changed.plug_yaw_deg


def test_finger_delays_obey_anchor_contract():
    contract = _contract()
    delay_set = set(contract.finger_start_delay_steps)
    for seed in range(40):
        realized = realize_randomization(contract, seed)
        delays = realized.finger_start_delay_steps
        anchor = realized.finger_start_delay_anchor_index
        assert 0 <= anchor <= 2
        # The anchor finger is exactly zero and every realized delay is an
        # original member of the configured physical timing slots.
        assert delays[anchor] == 0
        assert min(delays) == 0
        for value in delays:
            assert value in delay_set, (
                f"seed {seed} realized delay {value} is not a configured "
                "physical timing slot"
            )
        for draw in realized.finger_start_delay_raw_draws:
            assert draw in delay_set


def test_seed_6_delay_regression_remains_in_the_declared_slot_set():
    contract = _contract()
    realized = realize_randomization(contract, 6)
    delay_set = set(contract.finger_start_delay_steps)
    # The rejected relative-delay implementation produced (19, 0, 19), which
    # is not a configured slot; the anchor contract must never do that.
    assert all(
        value in delay_set
        for value in realized.finger_start_delay_steps
    )
    assert realized.finger_start_delay_steps[
        realized.finger_start_delay_anchor_index
    ] == 0


def test_friction_conditional_sampling_keeps_static_above_dynamic():
    contract = _contract()
    for seed in range(20):
        realized = realize_randomization(contract, seed)
        assert (
            realized.table_static_friction
            >= realized.table_dynamic_friction
        )
        assert (
            realized.fingertip_static_friction
            >= realized.fingertip_dynamic_friction
        )
        assert (
            realized.table_static_friction
            <= contract.table_static_friction.high
        )
        assert (
            realized.table_static_friction
            >= max(
                realized.table_dynamic_friction,
                contract.table_static_friction.low,
            )
        )


def test_unsolvable_friction_ranges_are_rejected():
    with pytest.raises(ValueError, match="static"):
        _contract(
            table_static_friction=IntervalContract(0.50, 0.60),
            table_dynamic_friction=IntervalContract(0.65, 0.85),
        )
    with pytest.raises(ValueError, match="static"):
        _contract(
            fingertip_static_friction=IntervalContract(1.00, 1.10),
            fingertip_dynamic_friction=IntervalContract(1.20, 1.60),
        )


def test_contract_rejects_bad_intervals_and_delay_sets():
    with pytest.raises(ValueError, match="low must not exceed high"):
        IntervalContract(1.0, 0.0)
    with pytest.raises(ValueError, match="finite non-bool"):
        IntervalContract(float("nan"), 1.0)
    with pytest.raises(ValueError, match="finite non-bool"):
        IntervalContract(True, 1.0)
    with pytest.raises(ValueError, match="non-empty"):
        _contract(finger_start_delay_steps=())
    with pytest.raises(ValueError, match="non-negative"):
        _contract(finger_start_delay_steps=(0, -1))
    with pytest.raises(ValueError, match="contain 0"):
        _contract(finger_start_delay_steps=(5, 12, 24))
    with pytest.raises(ValueError, match="unique"):
        _contract(finger_start_delay_steps=(0, 5, 5))


def test_realize_validates_seed():
    with pytest.raises(ValueError, match="seed"):
        realize_randomization(_contract(), -1)
    with pytest.raises(ValueError, match="seed"):
        realize_randomization(_contract(), 1.5)


def test_validate_realized_accepts_and_rejects():
    realized = realize_randomization(_contract(), 5)
    validate_realized(_contract(), realized)
    bad = RealizedRandomization(
        schema_version=realized.schema_version,
        seed=realized.seed,
        plug_x_offset_m=0.5,
        plug_y_offset_m=realized.plug_y_offset_m,
        plug_yaw_deg=realized.plug_yaw_deg,
        arm_center_error_x_m=realized.arm_center_error_x_m,
        arm_center_error_y_m=realized.arm_center_error_y_m,
        finger_start_delay_steps=realized.finger_start_delay_steps,
        finger_start_delay_raw_draws=realized.finger_start_delay_raw_draws,
        table_static_friction=realized.table_static_friction,
        table_dynamic_friction=realized.table_dynamic_friction,
        fingertip_static_friction=realized.fingertip_static_friction,
        fingertip_dynamic_friction=realized.fingertip_dynamic_friction,
        plug_mass_scale=realized.plug_mass_scale,
        center_of_mass_offset_m=realized.center_of_mass_offset_m,
        lift_speed_scale=realized.lift_speed_scale,
    )
    with pytest.raises(ValueError, match="escapes"):
        validate_realized(_contract(), bad)


def test_validate_realized_rejects_non_member_delays():
    contract = _contract()
    realized = realize_randomization(contract, 5)
    forged = RealizedRandomization(
        schema_version=realized.schema_version,
        seed=realized.seed,
        plug_x_offset_m=realized.plug_x_offset_m,
        plug_y_offset_m=realized.plug_y_offset_m,
        plug_yaw_deg=realized.plug_yaw_deg,
        arm_center_error_x_m=realized.arm_center_error_x_m,
        arm_center_error_y_m=realized.arm_center_error_y_m,
        finger_start_delay_steps=(19, 0, 19),
        finger_start_delay_raw_draws=(24, 5, 24),
        finger_start_delay_anchor_index=1,
        table_static_friction=realized.table_static_friction,
        table_dynamic_friction=realized.table_dynamic_friction,
        fingertip_static_friction=realized.fingertip_static_friction,
        fingertip_dynamic_friction=realized.fingertip_dynamic_friction,
        plug_mass_scale=realized.plug_mass_scale,
        center_of_mass_offset_m=realized.center_of_mass_offset_m,
        lift_speed_scale=realized.lift_speed_scale,
    )
    with pytest.raises(ValueError, match="configured"):
        validate_realized(contract, forged)


def test_realized_rejects_static_below_dynamic():
    realized = realize_randomization(_contract(), 5)
    with pytest.raises(ValueError, match="below dynamic"):
        RealizedRandomization(
            schema_version=realized.schema_version,
            seed=realized.seed,
            plug_x_offset_m=realized.plug_x_offset_m,
            plug_y_offset_m=realized.plug_y_offset_m,
            plug_yaw_deg=realized.plug_yaw_deg,
            arm_center_error_x_m=realized.arm_center_error_x_m,
            arm_center_error_y_m=realized.arm_center_error_y_m,
            finger_start_delay_steps=(0, 0, 0),
            finger_start_delay_raw_draws=(0, 0, 0),
            table_static_friction=0.5,
            table_dynamic_friction=0.7,
            fingertip_static_friction=realized.fingertip_static_friction,
            fingertip_dynamic_friction=realized.fingertip_dynamic_friction,
            plug_mass_scale=realized.plug_mass_scale,
            center_of_mass_offset_m=realized.center_of_mass_offset_m,
            lift_speed_scale=realized.lift_speed_scale,
        )


def test_active_fields_per_mode():
    staged = active_fields("staged")
    assert "lift_speed_scale" in staged
    assert "finger_start_delay_steps" in staged
    zero_lift = active_fields("zero-lift-hold")
    assert "lift_speed_scale" not in zero_lift
    assert "finger_start_delay_steps" in zero_lift
    single = active_fields("single-finger")
    assert "lift_speed_scale" not in single
    assert "finger_start_delay_steps" not in single
    with pytest.raises(ValueError, match="mode"):
        active_fields("nonsense")


def test_distinct_active_fields_ignores_inactive_fields():
    first = realize_randomization(_contract(), 1)
    second = RealizedRandomization(
        schema_version=first.schema_version,
        seed=2,
        plug_x_offset_m=first.plug_x_offset_m,
        plug_y_offset_m=first.plug_y_offset_m,
        plug_yaw_deg=first.plug_yaw_deg,
        arm_center_error_x_m=first.arm_center_error_x_m,
        arm_center_error_y_m=first.arm_center_error_y_m,
        finger_start_delay_steps=first.finger_start_delay_steps,
        finger_start_delay_raw_draws=first.finger_start_delay_raw_draws,
        finger_start_delay_anchor_index=(
            first.finger_start_delay_anchor_index
        ),
        table_static_friction=first.table_static_friction,
        table_dynamic_friction=first.table_dynamic_friction,
        fingertip_static_friction=first.fingertip_static_friction,
        fingertip_dynamic_friction=first.fingertip_dynamic_friction,
        plug_mass_scale=first.plug_mass_scale,
        center_of_mass_offset_m=first.center_of_mass_offset_m,
        lift_speed_scale=first.lift_speed_scale + 0.01,
    )
    # Only the inactive lift speed differs: zero-lift sees an identical
    # active scene and must refuse to treat it as a second seed.
    with pytest.raises(ValueError, match="same active scene"):
        distinct_active_fields(first, second, "zero-lift-hold")
    differences = distinct_active_fields(first, second, "staged")
    assert differences == ("lift_speed_scale",)
    assert distinct_active_fields(
        realize_randomization(_contract(), 1),
        realize_randomization(_contract(), 2),
        "staged",
    )
