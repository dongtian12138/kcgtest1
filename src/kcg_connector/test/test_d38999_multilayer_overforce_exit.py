from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_overforce_exit import (
    EXPECTED_FRAME_ID,
    FROZEN_SOURCES,
    OverforceExitLatch,
    load_overforce_exit_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _latch():
    return OverforceExitLatch.from_frozen_contracts(REPOSITORY_ROOT)


def _observe(latch, wrench=(0.0,) * 6, **overrides):
    values = {
        "timestamp_s": 1.0,
        "sample_age_s": 0.0,
        "frame_id": EXPECTED_FRAME_ID,
        "ft_valid": True,
        "ft_tared": True,
        "payload_compensated": True,
    }
    values.update(overrides)
    return latch.observe(wrench, **values)


def _copy_sources(tmp_path):
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
    return root


def test_contract_cross_checks_all_limits_and_retract():
    contract = load_overforce_exit_contract(REPOSITORY_ROOT)
    assert contract["formal_force_component_limit_n"] == 8.0
    assert contract["formal_moment_component_limit_nm"] == 0.30
    assert contract["experimental_abort_envelope"] == {
        "axial_force_n": 5.0,
        "lateral_force_n": 2.0,
        "bending_moment_nm": 0.18,
        "torsional_moment_nm": 0.05,
    }
    assert contract["maximum_sample_age_s"] == 0.020
    assert contract["backoff_distance_m"] == 0.00040
    assert contract["backoff_speed_m_s"] == 0.00030


def test_exact_experimental_boundaries_pass():
    decision = _observe(_latch(), (2.0, 0.0, 5.0, 0.18, 0.0, 0.05))
    assert decision["safe_to_continue"] is True
    assert decision["fault_latched"] is False


@pytest.mark.parametrize(
    ("wrench", "reason"),
    (
        ((0.0, 0.0, 5.000001, 0.0, 0.0, 0.0), "EXPERIMENTAL_AXIAL_FORCE_ABORT"),
        ((2.000001, 0.0, 0.0, 0.0, 0.0, 0.0), "EXPERIMENTAL_LATERAL_FORCE_ABORT"),
        ((0.0, 0.0, 0.0, 0.180001, 0.0, 0.0), "EXPERIMENTAL_BENDING_MOMENT_ABORT"),
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.050001), "EXPERIMENTAL_TORSIONAL_MOMENT_ABORT"),
        ((8.000001, 0.0, 0.0, 0.0, 0.0, 0.0), "FORMAL_FORCE_COMPONENT_LIMIT"),
        ((0.0, 0.0, 0.0, 0.300001, 0.0, 0.0), "FORMAL_MOMENT_COMPONENT_LIMIT"),
    ),
)
def test_each_overforce_class_latches_retract(wrench, reason):
    decision = _observe(_latch(), wrench)
    assert decision["safe_to_continue"] is False
    assert decision["failure_reason"] == reason
    assert decision["action"] == "SAFE_STOP_RETRACT_REOBSERVE"
    assert decision["retract_requested"] is True
    assert decision["requested_retract_distance_m"] == pytest.approx(0.0004)
    assert decision["insertion_twist_candidate_task"] == [0.0] * 6
    assert decision["motion_command_emitted"] is False


@pytest.mark.parametrize(
    ("wrench", "overrides", "reason"),
    (
        ((0.0, float("nan"), 0.0, 0.0, 0.0, 0.0), {}, "NONFINITE_OR_INVALID_WRENCH"),
        ((0.0,) * 5, {}, "NONFINITE_OR_INVALID_WRENCH"),
        ((0.0,) * 6, {"frame_id": "world"}, "UNEXPECTED_WRENCH_FRAME"),
        ((0.0,) * 6, {"sample_age_s": 0.020001}, "STALE_OR_INVALID_WRENCH"),
        ((0.0,) * 6, {"sample_age_s": -1.0}, "STALE_OR_INVALID_WRENCH"),
        ((0.0,) * 6, {"ft_valid": False}, "FT_TARE_OR_PAYLOAD_INVALID"),
        ((0.0,) * 6, {"ft_tared": False}, "FT_TARE_OR_PAYLOAD_INVALID"),
        ((0.0,) * 6, {"payload_compensated": False}, "FT_TARE_OR_PAYLOAD_INVALID"),
    ),
)
def test_invalid_or_unqualified_samples_latch_stop_without_retract(
    wrench, overrides, reason
):
    decision = _observe(_latch(), wrench, **overrides)
    assert decision["safe_to_continue"] is False
    assert decision["failure_reason"] == reason
    assert decision["action"] == "SAFE_STOP_REOBSERVE"
    assert decision["retract_requested"] is False
    assert decision["insertion_twist_candidate_task"] == [0.0] * 6


def test_timestamp_must_be_strictly_monotonic():
    latch = _latch()
    assert _observe(latch, timestamp_s=2.0)["safe_to_continue"] is True
    decision = _observe(latch, timestamp_s=2.0)
    assert decision["failure_reason"] == "NONMONOTONIC_TIMESTAMP"
    assert decision["safe_to_continue"] is False


@pytest.mark.parametrize("timestamp", (float("nan"), math.inf, True))
def test_invalid_timestamp_latches(timestamp):
    decision = _observe(_latch(), timestamp_s=timestamp)
    assert decision["failure_reason"] == "INVALID_TIMESTAMP"
    assert decision["safe_to_continue"] is False


def test_fault_remains_latched_until_explicit_reset():
    latch = _latch()
    first = _observe(latch, (0.0, 0.0, 5.1, 0.0, 0.0, 0.0), timestamp_s=1.0)
    assert first["fault_latched"] is True
    held = _observe(latch, (0.0,) * 6, timestamp_s=2.0)
    assert held["fault_latched"] is True
    assert held["sample_consumed"] is False
    assert held["failure_reason"] == "EXPERIMENTAL_AXIAL_FORCE_ABORT"
    with pytest.raises(ValueError):
        latch.reset_latch(explicit_authorization=False, reason="test")
    with pytest.raises(ValueError):
        latch.reset_latch(explicit_authorization=True, reason="")
    latch.reset_latch(explicit_authorization=True, reason="offline unit test")
    assert _observe(latch, timestamp_s=1.0)["safe_to_continue"] is True
    assert latch.report()["reset_count"] == 1


def test_public_api_excludes_privileged_truth():
    names = set(inspect.signature(OverforceExitLatch.observe).parameters)
    assert names == {
        "self", "compensated_wrench_task", "timestamp_s", "sample_age_s",
        "frame_id", "ft_valid", "ft_tared", "payload_compensated",
    }
    assert names.isdisjoint(
        {"object_pose", "contact_name", "contact_normal", "event_truth", "collider_path"}
    )


def test_frozen_source_drift_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_overforce_exit_contract(root)


def test_report_is_strict_json_and_never_claims_dynamic_pass():
    latch = _latch()
    _observe(latch)
    report = latch.report()
    json.dumps(report, sort_keys=True, allow_nan=False)
    assert report["dynamic_overforce_exit_pass_claimed"] is False
    assert report["control_authorized"] is False
    assert report["hardware_authorized"] is False
    assert all(value is False for value in report["truth_firewall"].values())
