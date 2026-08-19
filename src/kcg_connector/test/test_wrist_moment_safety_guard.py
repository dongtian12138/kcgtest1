from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from kcg_connector.wrist_moment_safety_guard import (
    EXPECTED_FRAME_ID,
    EXPECTED_SEMANTICS,
    WristMomentSafetyGuard,
    load_frozen_wrist_moment_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REFERENCE = (0.0, 0.55, 0.0)


def _guard() -> WristMomentSafetyGuard:
    return WristMomentSafetyGuard.from_frozen_contracts(
        REFERENCE, REPOSITORY_ROOT
    )


def _observe(
    guard: WristMomentSafetyGuard,
    moment,
    timestamp: float = 1.0,
    frame_id: str = EXPECTED_FRAME_ID,
):
    return guard.observe(
        moment,
        timestamp_s=timestamp,
        frame_id=frame_id,
    )


def test_frozen_sources_cross_check_exact_limit_and_hashes():
    contract = load_frozen_wrist_moment_contract(REPOSITORY_ROOT)
    assert contract["limit_nm"] == 0.30
    assert contract["semantics"] == EXPECTED_SEMANTICS
    assert contract["expected_frame_id"] == EXPECTED_FRAME_ID
    assert set(contract["cross_checked_values"].values()) == {0.30}
    assert len(contract["sources"]) == 4
    assert all(len(source["sha256"]) == 64 for source in contract["sources"])


def test_exact_perpendicular_limit_passes_and_over_limit_latches():
    exact = _guard()
    decision = _observe(exact, (0.30, 0.55, 0.0))
    assert decision["safe_to_continue"] is True
    assert decision["last_evaluation"]["perpendicular_nm"] == pytest.approx(
        0.30
    )

    over = _guard()
    decision = _observe(over, (0.30 + 1e-9, 0.55, 0.0))
    assert decision["safe_to_continue"] is False
    assert decision["failure_reason"] == "wrist_moment_limit"
    assert decision["trigger_component"] == "perpendicular"


@pytest.mark.parametrize(
    "moment,component",
    [
        ((0.0, 0.85 + 1e-9, 0.0), "magnitude_increase"),
        ((0.31, 0.55, 0.0), "perpendicular"),
        ((0.0, -0.31, 0.0), "reversal"),
    ],
)
def test_each_frozen_decomposition_component_can_latch(moment, component):
    decision = _observe(_guard(), moment)
    assert decision["safe_to_continue"] is False
    assert decision["trigger_component"] == component


@pytest.mark.parametrize(
    "bad_moment",
    [
        (0.0, float("nan"), 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 0.0),
        "bad",
        True,
    ],
)
def test_bad_moment_fails_closed_without_exception(bad_moment):
    decision = _observe(_guard(), bad_moment)
    assert decision["safe_to_continue"] is False
    assert decision["failure_reason"] == "nonfinite_or_invalid_wrist_moment"
    assert decision["sample_consumed"] is False


def test_timestamp_must_be_strictly_monotonic():
    guard = _guard()
    assert _observe(guard, REFERENCE, timestamp=2.0)["safe_to_continue"]
    decision = _observe(guard, REFERENCE, timestamp=2.0)
    assert decision["safe_to_continue"] is False
    assert decision["failure_reason"] == "nonmonotonic_wrench_timestamp"


@pytest.mark.parametrize("bad_timestamp", [float("nan"), math.inf, True])
def test_invalid_timestamp_fails_closed(bad_timestamp):
    decision = _observe(_guard(), REFERENCE, timestamp=bad_timestamp)
    assert decision["safe_to_continue"] is False
    assert decision["failure_reason"] == "invalid_timestamp"


def test_wrong_frame_fails_closed():
    decision = _observe(_guard(), REFERENCE, frame_id="world")
    assert decision["safe_to_continue"] is False
    assert decision["failure_reason"] == "unexpected_wrench_frame"


def test_fault_is_latched_until_explicit_reset():
    guard = _guard()
    first = _observe(guard, (0.31, 0.55, 0.0), timestamp=1.0)
    assert first["safe_to_continue"] is False
    held = _observe(guard, REFERENCE, timestamp=2.0)
    assert held["safe_to_continue"] is False
    assert held["sample_consumed"] is False
    assert held["failure_reason"] == "wrist_moment_limit"
    with pytest.raises(ValueError):
        guard.reset_latch(explicit_authorization=False, reason="test")
    with pytest.raises(ValueError):
        guard.reset_latch(explicit_authorization=True, reason="")
    guard.reset_latch(explicit_authorization=True, reason="offline unit test")
    assert _observe(guard, REFERENCE, timestamp=1.0)["safe_to_continue"]
    assert guard.report()["reset_count"] == 1


def test_report_is_json_safe_and_never_claims_dynamic_pass():
    guard = _guard()
    _observe(guard, REFERENCE)
    report = guard.report()
    json.dumps(report, allow_nan=False)
    assert report["dynamic_grasp_pass_claimed"] is False
    assert report["formal_physics_pass_claimed"] is False
    assert all(value is False for value in report["truth_firewall"].values())


def test_guard_public_inputs_exclude_privileged_simulation_truth():
    import inspect

    names = set(inspect.signature(WristMomentSafetyGuard.observe).parameters)
    assert names == {"self", "current_moment_nm", "timestamp_s", "frame_id"}
    forbidden = {
        "object_pose",
        "contact_name",
        "contact_normal",
        "event_truth",
        "collider_path",
    }
    assert names.isdisjoint(forbidden)
