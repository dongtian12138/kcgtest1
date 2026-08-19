from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_backoff_reobserve import (
    FROZEN_SOURCES,
    build_backoff_reobserve_plan,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _plan(**overrides):
    values = {"exit_latched": True, "exit_failure_reason": "EXPERIMENTAL_AXIAL_FORCE_ABORT"}
    values.update(overrides)
    return build_backoff_reobserve_plan(REPOSITORY_ROOT, **values)


def _copy_sources(tmp_path):
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
    return root


def test_recovery_sequence_is_stop_retract_hold_reobserve():
    plan = _plan()
    assert [row["name"] for row in plan["stages"]] == [
        "STOP_ZERO_TWIST", "RETRACT_REQUEST", "HOLD_FOR_SETTLE_REQUEST",
        "PRECOMMITTED_REOBSERVE_REQUEST",
    ]
    assert [row["ordinal"] for row in plan["stages"]] == [1, 2, 3, 4]


def test_backoff_and_hold_values_are_frozen():
    plan = _plan()
    assert plan["stages"][1]["distance_m"] == pytest.approx(0.0004)
    assert plan["stages"][1]["speed_m_s"] == pytest.approx(0.0003)
    assert plan["stages"][2]["duration_s"] == pytest.approx(0.5)


def test_exact_precommitted_view_ids_and_deltas_are_retained():
    views = _plan()["stages"][3]["candidate_views"]
    assert [row["view_id"] for row in views] == ["V0", "V1", "V2"]
    assert views[0]["tcp_delta_xyz_rpy"] == [0.0] * 6
    assert views[1]["tcp_delta_xyz_rpy"] == [
        0.012, -0.006, -0.03, 0.06981317007977318, -0.17453292519943295, 0.0
    ]
    assert views[2]["tcp_delta_xyz_rpy"] == [
        -0.012, 0.006, -0.03, -0.06981317007977318, 0.17453292519943295, 0.0
    ]


def test_legacy_view_failures_are_preserved_and_none_is_selected():
    plan = _plan()
    views = plan["stages"][3]["candidate_views"]
    assert [row["known_legacy_seed0_gate"] for row in views] == [
        "CAPTURE_PATH_ONLY", "PLANNED_MAX_JOINT_INF_RAD_EXCEEDED", "IK_FAILURE"
    ]
    assert plan["selected_view_for_execution"] is None
    assert all(row["dynamic_execution_authorized"] is False for row in views)


def test_unlatched_exit_does_not_create_recovery_steps():
    plan = _plan(exit_latched=False, exit_failure_reason=None)
    assert plan["status"] == "NO_RECOVERY_REQUEST"
    assert plan["stages"] == []


@pytest.mark.parametrize("reason", (None, "", "   "))
def test_latched_exit_requires_reason(reason):
    plan = _plan(exit_failure_reason=reason)
    assert plan["rejection_code"] == "LATCH_REASON_MISSING"
    assert plan["stages"] == []


def test_invalid_input_type_fails_closed():
    plan = _plan(exit_latched=1)
    assert plan["rejection_code"] == "INVALID_RECOVERY_INPUT"
    assert plan["stages"] == []


def test_plan_never_emits_motion_capture_or_dynamic_claim():
    plan = _plan()
    assert plan["status"] == "PLANNED_NOT_AUTHORIZED"
    assert plan["robot_motion_started"] is False
    assert plan["render_capture_performed"] is False
    assert plan["motion_command_emitted"] is False
    assert plan["capture_command_emitted"] is False
    assert plan["control_authorized"] is False
    assert plan["dynamic_reobserve_pass_claimed"] is False


def test_truth_firewall_is_all_false():
    assert all(value is False for value in _plan()["truth_firewall"].values())
    names = set(inspect.signature(build_backoff_reobserve_plan).parameters)
    assert names.isdisjoint(
        {"object_pose", "contact_name", "contact_normal", "event_truth", "collider_path"}
    )


def test_frozen_source_drift_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_backoff_reobserve_plan(
            root, exit_latched=True, exit_failure_reason="test"
        )


def test_plan_is_strict_json():
    json.dumps(_plan(), sort_keys=True, allow_nan=False)
