from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_safe_key_search import (
    BRANCH_IDS,
    FROZEN_SOURCES,
    MAXIMUM_RZ_STEP_RAD,
    MAXIMUM_SEARCH_ANGLE_RAD,
    VISION_YAW_SOURCE,
    build_safe_key_search_contract,
    evaluate_safe_key_search_step,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _step(**overrides):
    values = {
        "visual_yaw_error_rad": 0.1,
        "current_search_angle_rad": 0.0,
        "search_attempt_count": 0,
        "registered_preentry_command_fk_gap_m": 0.010,
        "registered_preentry_measured_fk_gap_m": 0.010,
        "wrist_moment_task_nm": (0.0, 0.0, 0.0),
        "upstream_attitude_ready": True,
        "vision_pose_control_authorized": True,
        "selected_c2_branch_id": BRANCH_IDS[0],
        "visual_yaw_source": VISION_YAW_SOURCE,
    }
    values.update(overrides)
    return evaluate_safe_key_search_step(**values)


def _copy_sources(tmp_path):
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
    return root


def test_contract_cross_checks_key_geometry_preentry_and_limits():
    contract = build_safe_key_search_contract(REPOSITORY_ROOT)
    assert contract["key_count"] == 5
    assert contract["key_angles_deg"] == [0.0, 80.0, 142.0, 196.0, 293.0]
    assert contract["minimum_preentry_gap_m"] == pytest.approx(0.010)
    assert contract["maximum_rz_step_rad"] == pytest.approx(0.010 / 240.0)
    assert contract["maximum_search_angle_rad"] == pytest.approx(0.008)
    assert contract["maximum_search_attempts"] == 2
    assert contract["formal_moment_component_limit_nm"] == pytest.approx(0.30)


def test_current_unresolved_upstream_emits_exact_zero():
    current = build_safe_key_search_contract(REPOSITORY_ROOT)["current_readiness"]
    assert current["rejection_code"] == "UPSTREAM_ATTITUDE_REJECTED"
    assert current["delta_twist_task"] == [0.0] * 6
    assert current["motion_command_emitted"] is False
    assert current["control_authorized"] is False


@pytest.mark.parametrize("yaw_error", (0.1, -0.1))
def test_candidate_step_is_directional_and_rate_bounded(yaw_error):
    result = _step(visual_yaw_error_rad=yaw_error)
    assert result["key_search_candidate"] is True
    assert result["delta_twist_task"][:5] == [0.0] * 5
    assert result["delta_twist_task"][5] == pytest.approx(
        math.copysign(MAXIMUM_RZ_STEP_RAD, yaw_error)
    )
    assert result["motion_command_emitted"] is False


def test_last_step_is_clipped_to_total_angle_budget():
    current = MAXIMUM_SEARCH_ANGLE_RAD - 1.0e-5
    result = _step(current_search_angle_rad=current)
    assert result["delta_twist_task"][5] == pytest.approx(1.0e-5)
    assert result["next_search_angle_rad"] == pytest.approx(
        MAXIMUM_SEARCH_ANGLE_RAD
    )


def test_outward_motion_at_angle_budget_requires_retract():
    result = _step(current_search_angle_rad=MAXIMUM_SEARCH_ANGLE_RAD)
    assert result["rejection_code"] == "SEARCH_ANGLE_BUDGET_EXHAUSTED"
    assert result["retract_required"] is True
    assert result["delta_twist_task"] == [0.0] * 6


def test_inward_motion_at_angle_boundary_remains_bounded():
    result = _step(
        current_search_angle_rad=MAXIMUM_SEARCH_ANGLE_RAD,
        visual_yaw_error_rad=-0.1,
    )
    assert result["key_search_candidate"] is True
    assert result["delta_twist_task"][5] == pytest.approx(-MAXIMUM_RZ_STEP_RAD)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("upstream_attitude_ready", False, "UPSTREAM_ATTITUDE_REJECTED"),
        ("vision_pose_control_authorized", False, "VISION_KEY_YAW_UNAUTHORIZED"),
        ("selected_c2_branch_id", None, "C2_BRANCH_UNRESOLVED"),
        ("selected_c2_branch_id", "YAW_0", "C2_BRANCH_UNRESOLVED"),
        ("visual_yaw_source", "SIMULATOR_OBJECT_TRUTH", "VISUAL_YAW_PROVENANCE_REJECTED"),
    ),
)
def test_upstream_and_visual_gates_fail_closed(field, value, code):
    result = _step(**{field: value})
    assert result["rejection_code"] == code
    assert result["delta_twist_task"] == [0.0] * 6


@pytest.mark.parametrize(
    "field",
    (
        "registered_preentry_command_fk_gap_m",
        "registered_preentry_measured_fk_gap_m",
    ),
)
def test_both_registered_fk_gaps_must_remain_at_preentry(field):
    result = _step(**{field: 0.009999})
    assert result["rejection_code"] == "PIN_DEPTH_WINDOW_CLOSED"
    assert result["retract_required"] is True
    assert result["delta_twist_task"] == [0.0] * 6


def test_search_attempt_budget_is_two():
    result = _step(search_attempt_count=2)
    assert result["rejection_code"] == "SEARCH_ATTEMPT_BUDGET_EXHAUSTED"
    assert result["retract_required"] is True


def test_key_mismatch_torsion_stops_rotation_and_requires_retract():
    result = _step(wrist_moment_task_nm=(0.0, 0.0, 0.025))
    assert result["rejection_code"] == "KEY_MISMATCH_TORSION_DETECTED"
    assert result["retract_required"] is True
    assert result["delta_twist_task"] == [0.0] * 6


@pytest.mark.parametrize(
    ("moment", "code"),
    (
        ((0.300000001, 0.0, 0.0), "FORMAL_MOMENT_COMPONENT_LIMIT"),
        ((0.181, 0.0, 0.0), "EXPERIMENTAL_BENDING_ABORT"),
        ((0.0, 0.0, 0.051), "EXPERIMENTAL_TORSION_ABORT"),
    ),
)
def test_wrist_moment_limits_fail_closed(moment, code):
    result = _step(wrist_moment_task_nm=moment)
    assert result["rejection_code"] == code
    assert result["delta_twist_task"] == [0.0] * 6


def test_zero_visual_error_is_not_claimed_as_keying_success():
    result = _step(visual_yaw_error_rad=0.0)
    assert result["status"] == "VISUAL_YAW_ERROR_ZERO_DIAGNOSTIC_ONLY"
    assert result["key_search_candidate"] is False
    assert result["dynamic_key_search_pass_claimed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("visual_yaw_error_rad", float("nan")),
        ("current_search_angle_rad", float("inf")),
        ("search_attempt_count", True),
        ("wrist_moment_task_nm", (0.0, 0.0)),
        ("wrist_moment_task_nm", (0.0, float("nan"), 0.0)),
    ),
)
def test_invalid_inputs_fail_closed(field, value):
    result = _step(**{field: value})
    assert result["rejection_code"] == "INVALID_KEY_SEARCH_INPUT"
    assert result["delta_twist_task"] == [0.0] * 6


def test_api_excludes_object_contact_and_event_truth():
    parameters = set(inspect.signature(evaluate_safe_key_search_step).parameters)
    assert parameters.isdisjoint(
        {
            "object_pose",
            "contact_name",
            "contact_normal",
            "collider_path",
            "event_truth",
            "first_pin_event",
        }
    )


def test_frozen_source_drift_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_safe_key_search_contract(root)


def test_contract_and_candidate_are_strict_json_without_dynamic_claim():
    contract = build_safe_key_search_contract(REPOSITORY_ROOT)
    candidate = _step()
    json.dumps(contract, sort_keys=True, allow_nan=False)
    json.dumps(candidate, sort_keys=True, allow_nan=False)
    assert contract["simulation_started"] is False
    assert contract["dynamic_key_search_pass_claimed"] is False
    assert contract["control_authorized"] is False
    assert candidate["motion_command_emitted"] is False
    assert candidate["control_authorized"] is False
