from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_lateral_centering import (
    FROZEN_SOURCES,
    build_multilayer_lateral_centering_contract,
    compute_bounded_xy_correction,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _compute(force=(0.8, -0.4), offset=(0.0, 0.0), ready=True):
    return compute_bounded_xy_correction(
        force, offset, upstream_light_contact_ready=ready
    )


def _copy_sources(tmp_path):
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
    return root


def test_contract_preserves_direction_gain_and_bounds():
    contract = build_multilayer_lateral_centering_contract(REPOSITORY_ROOT)
    assert contract["status"] == "OFFLINE_CENTERING_LAW_READY"
    assert contract["correction_direction"] == "environment_force_on_tool"
    assert contract["lateral_correction_gain_m_per_n"] == pytest.approx(0.000025)
    assert contract["maximum_xy_correction_step_m"] == pytest.approx(0.00002)
    assert contract["maximum_xy_search_radius_m"] == pytest.approx(0.003)
    assert contract["preentry_requires_freeze_and_unload"] is True


def test_current_upstream_rejection_outputs_exact_zero():
    result = _compute(ready=False)
    assert result["rejection_code"] == "UPSTREAM_LIGHT_CONTACT_REJECTED"
    assert result["delta_tcp_task_m"] == [0.0, 0.0, 0.0]
    assert result["motion_command_emitted"] is False
    assert result["control_authorized"] is False


def test_force_direction_is_preserved_and_z_is_always_zero():
    result = _compute(force=(-0.8, 0.4))
    dx, dy, dz = result["delta_tcp_task_m"]
    assert dx < 0.0 and dy > 0.0 and dz == 0.0
    assert result["correction_candidate"] is True


def test_large_force_is_norm_clamped_to_twenty_micrometres():
    result = _compute(force=(100.0, 100.0))
    assert math.hypot(*result["delta_tcp_task_m"][:2]) == pytest.approx(0.00002)
    assert result["delta_tcp_task_m"][2] == 0.0


def test_exact_step_boundary_is_not_rejected():
    result = _compute(force=(0.8, 0.0))
    assert result["correction_norm_m"] == pytest.approx(0.00002)
    assert result["correction_candidate"] is True


def test_total_search_radius_is_fail_closed_without_partial_step():
    result = _compute(force=(0.8, 0.0), offset=(0.00299, 0.0))
    assert result["rejection_code"] == "XY_SEARCH_RADIUS_EXCEEDED"
    assert result["delta_tcp_task_m"] == [0.0, 0.0, 0.0]
    assert result["next_xy_offset_task_m"] is None


def test_below_candidate_force_outputs_no_correction():
    result = _compute(force=(0.099, 0.0))
    assert result["rejection_code"] == "LATERAL_FORCE_BELOW_CANDIDATE_THRESHOLD"
    assert result["delta_tcp_task_m"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("force", "offset"),
    (
        ((float("nan"), 0.0), (0.0, 0.0)),
        ((float("inf"), 0.0), (0.0, 0.0)),
        ((1.0,), (0.0, 0.0)),
        ((1.0, 0.0), (0.0,)),
        ("bad", (0.0, 0.0)),
    ),
)
def test_invalid_inputs_fail_closed(force, offset):
    result = _compute(force=force, offset=offset)
    assert result["rejection_code"] == "INVALID_CENTERING_INPUT"
    assert result["delta_tcp_task_m"] == [0.0, 0.0, 0.0]


def test_frozen_source_drift_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_lateral_centering_contract(root)


def test_api_has_no_z_force_pose_or_contact_truth_inputs():
    parameters = set(inspect.signature(compute_bounded_xy_correction).parameters)
    assert parameters == {
        "lateral_force_task_n", "current_xy_offset_task_m", "upstream_light_contact_ready"
    }
    assert parameters.isdisjoint(
        {"fz", "object_pose", "contact_name", "contact_normal", "event_truth", "collider_path"}
    )


def test_contract_and_candidate_are_strict_json_without_dynamic_claim():
    contract = build_multilayer_lateral_centering_contract(REPOSITORY_ROOT)
    result = _compute()
    json.dumps(contract, allow_nan=False, sort_keys=True)
    json.dumps(result, allow_nan=False, sort_keys=True)
    assert contract["simulation_started"] is False
    assert contract["dynamic_centering_pass_claimed"] is False
    assert result["motion_command_emitted"] is False
    assert result["control_authorized"] is False
