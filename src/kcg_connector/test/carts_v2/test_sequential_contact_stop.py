"""Regressions for bounded sequential contact transitions."""

from pathlib import Path
import sys


ISAAC_V2 = Path(__file__).resolve().parents[2] / "isaac/carts_v2"
sys.path.insert(0, str(ISAAC_V2))

from controller import SequentialEffortContactController  # noqa: E402


def _controller(start=0.399834) -> SequentialEffortContactController:
    controller = SequentialEffortContactController(
        [0.0, 0.251334, 0.0, 0.0], [0.0, 0.65, 0.5, 0.5],
        effort_rise_nm=0.02, position_error_rad=0.004,
        consecutive_samples=1, endpoint_timeout_samples=10, hand_stiffness=12.0,
    )
    controller.target[1] = start
    return controller


def test_run17_confirmation_is_bumpless_and_bounded() -> None:
    controller = _controller()
    previous = controller.target.copy()
    target = controller.step(
        measured_position=[0.0, 0.372807622, 0.0, 0.0],
        measured_effort_delta=[0.0, -0.094812326, 0.0, 0.0],
        maximum_increment_rad=0.0015,
    )
    assert target[1] == previous[1]
    assert abs(target[1] - previous[1]) <= 0.0015
    assert controller.state == "CONTACT_CONFIRMED"
    assert controller.contact_targets_rad == (0.372807622,)
    assert controller.active_finger == 0


def test_settle_uses_bounded_effort_correction_before_next_finger() -> None:
    controller = _controller()
    measured = [0.0, 0.372807622, 0.0, 0.0]
    controller.step(measured, [0.0, -0.094812326, 0.0, 0.0], 0.0015)
    confirmed = controller.target.copy()
    controller.step(measured, [0.0] * 4, 0.0015)
    low = controller.step(measured, [0.0, 0.002, 0.0, 0.0], 0.0015)
    assert controller.state == "CONTACT_SETTLE"
    assert 0.0 <= low[1] - confirmed[1] <= 0.0015 + 1.0e-12
    settled = controller.step(measured, [0.0, 0.10, 0.0, 0.0], 0.0015)
    assert abs(settled[1] - confirmed[1]) <= 1.0e-12
    assert controller.state == "CONTACT_SETTLE"
    held = controller.step(measured, [0.0, 0.02, 0.0, 0.0], 0.0015)
    assert controller.state == "HOLD"
    held = controller.step(measured, [0.0, 0.02, 0.0, 0.0], 0.0015)
    assert controller.active_finger == 1
    assert held[1] == settled[1]
    next_target = controller.step(measured, [0.0] * 4, 0.0015)
    assert next_target[2] == 0.0015
    assert controller.contact_targets_rad == (0.372807622,)


def test_hold_can_be_kept_on_first_finger_without_advancing() -> None:
    controller = SequentialEffortContactController(
        [0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.5],
        effort_rise_nm=0.02, position_error_rad=0.004,
        consecutive_samples=1, endpoint_timeout_samples=10, hand_stiffness=12.0,
    )
    rows = []
    for _ in range(3):
        rows.append(controller.step(
            [0.0, -0.01, 0.0, 0.0], [0.0, 0.02, 0.0, 0.0], 0.0015,
            advance_after_hold=False,
        ))
    assert controller.state == "HOLD"
    before = controller.target.copy()
    after = controller.step(
        [0.0, -0.01, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 0.0015,
        advance_after_hold=False,
    )
    assert controller.active_finger == 0
    assert 0.0 <= after[1] - before[1] <= 0.0015
    assert all(abs(right[1] - left[1]) <= 0.0015 for left, right in zip(rows, rows[1:]))
