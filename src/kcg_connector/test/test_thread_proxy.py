import pytest

from kcg_connector.thread_proxy import rack_and_pinion_ratio


def test_ratio_is_degrees_per_meter_for_meter_stage():
    assert rack_and_pinion_ratio(0.004, 1.0) == pytest.approx(90000.0)


def test_ratio_respects_non_meter_stage_units():
    assert rack_and_pinion_ratio(0.004, 0.01) == pytest.approx(900.0)


def test_ratio_direction_is_explicit():
    assert rack_and_pinion_ratio(0.004, 1.0, direction=-1) == pytest.approx(
        -90000.0
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (0.0, 1.0, 1),
        (0.004, 0.0, 1),
        (0.004, 1.0, 0),
    ],
)
def test_invalid_ratio_inputs_are_rejected(arguments):
    with pytest.raises(ValueError):
        rack_and_pinion_ratio(*arguments)
