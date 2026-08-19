from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_light_contact import (
    FROZEN_SOURCES,
    LightContactSample,
    build_multilayer_light_contact_contract,
    evaluate_light_contact_sample,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _sample(**updates):
    values = dict(
        timestamp_s=1.0,
        sample_age_s=0.001,
        frame_id="connector_task_frame",
        compensated_wrench_task=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        local_reference_ready=True,
        compressive_direction_calibrated=True,
        upstream_prealign_ready=True,
    )
    values.update(updates)
    return LightContactSample(**values)


def _copy_sources(tmp_path):
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
    return root


def test_contract_preserves_wrench_frame_reference_and_candidate_thresholds():
    contract = build_multilayer_light_contact_contract(REPOSITORY_ROOT)
    assert contract["status"] == "OFFLINE_CONTACT_GATE_READY"
    assert contract["task_frame_id"] == "connector_task_frame"
    assert contract["control_rate_hz"] == 240
    assert contract["local_reference_samples"] == 120
    assert contract["local_reference_is_safety_tare"] is False
    assert contract["contact_on_candidate_n"] == pytest.approx(0.25)
    assert contract["contact_off_candidate_n"] == pytest.approx(0.10)
    assert contract["compressive_direction_calibrated"] is False


def test_contract_keeps_experiment_and_formal_limits_distinct():
    contract = build_multilayer_light_contact_contract(REPOSITORY_ROOT)
    assert contract["experimental_abort_envelope"] == {
        "maximum_axial_force_n": 5.0,
        "maximum_lateral_force_n": 2.0,
        "maximum_bending_torque_nm": 0.18,
        "maximum_tightening_torque_nm": 0.05,
        "calibrated_hardware_safety_limit": False,
    }
    assert contract["formal_moment_component_limit_nm"] == pytest.approx(0.30)


def test_current_readiness_is_rejected_for_upstream_and_direction():
    contract = build_multilayer_light_contact_contract(REPOSITORY_ROOT)
    assert contract["current_readiness"]["rejection_code"] == (
        "UPSTREAM_PREALIGN_REJECTED"
    )
    assert contract["current_readiness"]["secondary_blocker"] == (
        "CONTACT_DIRECTION_UNCALIBRATED"
    )
    assert contract["control_authorized"] is False


@pytest.mark.parametrize(
    ("changes", "code"),
    (
        ({"upstream_prealign_ready": False}, "UPSTREAM_PREALIGN_REJECTED"),
        ({"local_reference_ready": False}, "LOCAL_REFERENCE_NOT_READY"),
        ({"compressive_direction_calibrated": False}, "CONTACT_DIRECTION_UNCALIBRATED"),
        ({"frame_id": "world"}, "WRONG_WRENCH_FRAME"),
        ({"sample_age_s": 0.009}, "STALE_WRENCH_SAMPLE"),
        ({"compensated_wrench_task": (0, 0, 5.01, 0, 0, 0)}, "AXIAL_FORCE_ABORT"),
        ({"compensated_wrench_task": (2.01, 0, 0, 0, 0, 0)}, "LATERAL_FORCE_ABORT"),
        ({"compensated_wrench_task": (0, 0, 0, 0.300000001, 0, 0)}, "WRIST_MOMENT_HARD_LIMIT"),
        ({"compensated_wrench_task": (0, 0, 0, 0.181, 0, 0)}, "BENDING_MOMENT_EXPERIMENT_ABORT"),
        ({"compensated_wrench_task": (0, 0, 0, 0, 0, 0.051)}, "TIGHTENING_MOMENT_EXPERIMENT_ABORT"),
    ),
)
def test_failure_and_abort_paths_never_command_motion(changes, code):
    result = evaluate_light_contact_sample(replace(_sample(), **changes))
    assert result["rejection_code"] == code
    assert result["motion_command_emitted"] is False
    assert result["control_authorized"] is False


def test_contact_threshold_is_diagnostic_only_and_not_confirmation():
    result = evaluate_light_contact_sample(
        _sample(compensated_wrench_task=(0, 0, 0.25, 0, 0, 0))
    )
    assert result["rejection_code"] == "LIGHT_CONTACT_CANDIDATE_DIAGNOSTIC_ONLY"
    assert result["light_contact_candidate"] is True
    assert result["contact_confirmed"] is False
    assert result["dynamic_light_contact_pass_claimed"] is False


def test_subthreshold_sample_reports_no_candidate_and_no_command():
    result = evaluate_light_contact_sample(
        _sample(compensated_wrench_task=(0, 0, 0.249999, 0, 0, 0))
    )
    assert result["rejection_code"] == "NO_LIGHT_CONTACT_CANDIDATE"
    assert result["light_contact_candidate"] is False
    assert result["motion_command_emitted"] is False


@pytest.mark.parametrize(
    "changes",
    (
        {"timestamp_s": float("nan")},
        {"sample_age_s": float("inf")},
        {"compensated_wrench_task": (0, 0, float("nan"), 0, 0, 0)},
        {"compensated_wrench_task": (0, 0, 0)},
    ),
)
def test_nonfinite_and_bad_shapes_fail_closed(changes):
    result = evaluate_light_contact_sample(replace(_sample(), **changes))
    assert result["rejection_code"] == "NONFINITE_OR_INVALID_SAMPLE"
    assert result["control_authorized"] is False


def test_frozen_source_drift_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_light_contact_contract(root)


def test_public_api_excludes_contact_and_pose_truth():
    parameters = set(inspect.signature(evaluate_light_contact_sample).parameters)
    assert parameters == {"sample"}
    assert parameters.isdisjoint(
        {"contact_name", "contact_normal", "collider_path", "event_truth", "object_pose", "ground_truth_pose"}
    )


def test_contract_and_result_are_finite_json_without_dynamic_claim():
    contract = build_multilayer_light_contact_contract(REPOSITORY_ROOT)
    result = evaluate_light_contact_sample(_sample())
    json.dumps(contract, sort_keys=True, allow_nan=False)
    json.dumps(result, sort_keys=True, allow_nan=False)
    assert contract["simulation_started"] is False
    assert contract["contact_simulation_started"] is False
    assert contract["dynamic_light_contact_pass_claimed"] is False
