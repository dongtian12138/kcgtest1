"""Pure tests for the disabled residual learnability challenge v1."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from kcg_connector.residual_learning_challenge import (
    BASE_CLAMP_NOMINAL_POSITIONS_RAD,
    CHALLENGE_SCHEMA_VERSION,
    CLAMP_RESIDUAL_ACTION_LIMITS_RAD,
    DEFAULT_CHALLENGE_PATH,
    RESIDUAL_ACTION_SIZE,
    RESIDUAL_INTERFACE_VERSION,
    RESIDUAL_OBSERVATION_SIZE,
    evaluate_challenge_acceptance,
    load_residual_learning_challenge,
    oracle_compensation,
    prove_learnability,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_PATH = (
    PACKAGE_ROOT
    / "config/connector_residual_learning_challenge_v1.yaml"
)


def _load():
    return load_residual_learning_challenge(CHALLENGE_PATH)


def _invalid_document(tmp_path, mutator):
    document = yaml.safe_load(CHALLENGE_PATH.read_text(encoding="utf-8"))
    mutator(document)
    path = tmp_path / "invalid_learning_challenge.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_default_contract_is_disabled_and_preserves_residual_v0_shape():
    challenge = _load()
    assert DEFAULT_CHALLENGE_PATH == CHALLENGE_PATH
    assert challenge.schema_version == CHALLENGE_SCHEMA_VERSION
    assert challenge.enabled is False
    assert challenge.interface_version == RESIDUAL_INTERFACE_VERSION
    assert challenge.action_size == RESIDUAL_ACTION_SIZE == 4
    assert challenge.observation_size == RESIDUAL_OBSERVATION_SIZE == 24
    assert challenge.operational_stage_reference == "stage20"


def test_challenge_task_and_control_path_are_exact():
    challenge = _load()
    task = challenge.task
    control = challenge.control_path
    assert task.target_angle_degrees == pytest.approx(20.0)
    assert task.success_hold_duration_s == pytest.approx(0.5)
    assert task.hold_entry_detection_policy_steps == 1
    assert task.policy_rate_hz == pytest.approx(10.0)
    assert task.maximum_episode_steps == 28
    assert task.nominal_q7_speed_degrees_per_second == pytest.approx(10.0)
    assert task.q7_speed_residual_degrees_per_second == pytest.approx(2.0)
    assert task.maximum_q7_speed_degrees_per_second == pytest.approx(20.0)
    assert control.q7_velocity_scale_lower == pytest.approx(0.85)
    assert control.q7_velocity_scale_upper == pytest.approx(1.15)
    assert control.clamp_offset_lower_rad == pytest.approx((-0.015,) * 3)
    assert control.clamp_offset_upper_rad == pytest.approx((0.015,) * 3)
    assert control.clamp_observation_reference == "base_nominal"
    assert control.clamp_base_nominal_positions_rad == pytest.approx(
        BASE_CLAMP_NOMINAL_POSITIONS_RAD
    )
    assert control.clamp_residual_action_limits_rad == pytest.approx(
        CLAMP_RESIDUAL_ACTION_LIMITS_RAD
    )


def test_deadline_is_penalized_but_is_not_a_safety_failure():
    challenge = _load()
    assert challenge.deadline.penalty == pytest.approx(-10.0)
    assert challenge.deadline.termination_reason == "time_limit"
    assert challenge.deadline.counts_as_safety_failure is False
    assert challenge.acceptance.zero_policy_allowed_failure_reasons == (
        "time_limit",
    )
    assert (
        challenge.acceptance.maximum_zero_policy_raw_safety_failures == 0
    )
    assert challenge.acceptance.maximum_oracle_raw_safety_failures == 0


def test_mass_friction_and_helical_lead_are_all_fixed():
    fixed = _load().fixed_physics
    assert fixed.mass_randomized is False
    assert fixed.friction_randomized is False
    assert fixed.helical_lead_randomized is False


def test_seed_ranges_are_exact_disjoint_and_have_required_counts():
    challenge = _load()
    assert (
        challenge.tuning_seeds.start,
        challenge.tuning_seeds.end_inclusive,
        challenge.tuning_seeds.count,
    ) == (20000, 20063, 64)
    assert (
        challenge.validation_seeds.start,
        challenge.validation_seeds.end_inclusive,
        challenge.validation_seeds.count,
    ) == (30000, 30127, 128)
    assert (
        challenge.final_paired_seeds.start,
        challenge.final_paired_seeds.end_inclusive,
        challenge.final_paired_seeds.count,
    ) == (10000, 10099, 100)
    assert challenge.tuning_seeds.contains(20000)
    assert challenge.tuning_seeds.contains(20063)
    assert not challenge.tuning_seeds.contains(20064)
    assert not challenge.tuning_seeds.contains(True)


def test_acceptance_gates_are_exact_and_require_all_pair_matches():
    acceptance = _load().acceptance
    assert acceptance.zero_policy_success_rate_minimum == pytest.approx(0.65)
    assert acceptance.zero_policy_success_rate_maximum == pytest.approx(0.85)
    assert acceptance.oracle_minimum_success_rate == pytest.approx(0.98)
    assert acceptance.require_all_paired_randomization_matches is True


@pytest.mark.parametrize(
    "scale,offsets",
    (
        (0.85, (-0.015, 0.015, 0.0)),
        (1.00, (0.0, 0.0, 0.0)),
        (1.15, (0.015, -0.015, 0.015)),
    ),
)
def test_oracle_exactly_cancels_every_control_path_boundary(scale, offsets):
    challenge = _load()
    result = oracle_compensation(challenge, scale, offsets)
    assert len(result.normalized_action) == 4
    assert max(abs(value) for value in result.normalized_action) <= 1.0
    assert result.q7_effective_speed_degrees_per_second == pytest.approx(
        challenge.task.nominal_q7_speed_degrees_per_second
    )
    assert result.compensated_clamp_positions_rad == pytest.approx(
        BASE_CLAMP_NOMINAL_POSITIONS_RAD
    )
    json.dumps(result.as_dict(), allow_nan=False)


def test_oracle_q7_and_clamp_actions_have_stated_closed_form_bounds():
    challenge = _load()
    slow = oracle_compensation(challenge, 0.85, (0.015,) * 3)
    fast = oracle_compensation(challenge, 1.15, (-0.015,) * 3)
    assert slow.normalized_action[0] == pytest.approx(0.8823529411764706)
    assert fast.normalized_action[0] == pytest.approx(-0.6521739130434785)
    assert slow.normalized_action[1:] == pytest.approx((-0.75,) * 3)
    assert fast.normalized_action[1:] == pytest.approx((0.75,) * 3)


@pytest.mark.parametrize(
    "scale,offsets,message",
    (
        (0.849, (0.0, 0.0, 0.0), "scale"),
        (1.151, (0.0, 0.0, 0.0), "scale"),
        (math.nan, (0.0, 0.0, 0.0), "finite"),
        (1.0, (0.016, 0.0, 0.0), "offset 0"),
        (1.0, (0.0, 0.0), "3 finite"),
    ),
)
def test_oracle_rejects_values_outside_the_frozen_domain(
    scale, offsets, message
):
    with pytest.raises(ValueError, match=message):
        oracle_compensation(_load(), scale, offsets)


def test_mathematical_proof_covers_speed_actions_and_28_step_budget():
    proof = prove_learnability(_load())
    assert proof.minimum_effective_q7_speed_degrees_per_second == (
        pytest.approx(6.8)
    )
    assert proof.maximum_effective_q7_speed_degrees_per_second == (
        pytest.approx(13.8)
    )
    assert proof.maximum_allowed_q7_speed_degrees_per_second == (
        pytest.approx(20.0)
    )
    assert proof.effective_speed_bound_proved
    assert -1.0 <= proof.q7_oracle_action_minimum
    assert proof.q7_oracle_action_maximum <= 1.0
    assert proof.q7_oracle_action_bound_proved
    assert proof.clamp_oracle_maximum_absolute_action == pytest.approx(0.75)
    assert proof.clamp_oracle_action_bound_proved
    assert proof.oracle_motion_policy_steps == 20
    assert proof.hold_entry_detection_policy_steps == 1
    assert proof.success_hold_policy_steps == 5
    assert proof.oracle_required_policy_steps == 26
    assert proof.maximum_episode_steps == 28
    assert proof.oracle_budget_margin_policy_steps == 2
    assert proof.oracle_budget_proved
    assert proof.slowest_zero_policy_required_steps == 30
    assert proof.slowest_zero_policy_hits_deadline
    assert proof.passed
    json.dumps(proof.as_dict(), allow_nan=False)


@pytest.mark.parametrize("zero_rate", (0.65, 0.75, 0.85))
def test_executable_acceptance_gate_passes_inclusive_boundaries(zero_rate):
    result = evaluate_challenge_acceptance(
        _load(),
        zero_policy_success_rate=zero_rate,
        oracle_success_rate=0.98,
        zero_policy_raw_safety_failures=0,
        oracle_raw_safety_failures=0,
        zero_policy_failure_reasons=("time_limit",),
        paired_randomization_matches=(True,) * 100,
    )
    assert result.zero_policy_success_rate_in_range
    assert result.oracle_success_rate_passed
    assert result.zero_policy_raw_safety_passed
    assert result.oracle_raw_safety_passed
    assert result.zero_policy_failure_reasons_passed
    assert result.paired_randomization_matches_passed
    assert result.passed
    json.dumps(result.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "overrides,failed_field",
    (
        (
            {"zero_policy_success_rate": 0.64},
            "zero_policy_success_rate_in_range",
        ),
        (
            {"zero_policy_success_rate": 0.86},
            "zero_policy_success_rate_in_range",
        ),
        ({"oracle_success_rate": 0.97}, "oracle_success_rate_passed"),
        (
            {"zero_policy_raw_safety_failures": 1},
            "zero_policy_raw_safety_passed",
        ),
        (
            {"oracle_raw_safety_failures": 1},
            "oracle_raw_safety_passed",
        ),
        (
            {"zero_policy_failure_reasons": ("cross_thread",)},
            "zero_policy_failure_reasons_passed",
        ),
        (
            {"paired_randomization_matches": (True,) * 99 + (False,)},
            "paired_randomization_matches_passed",
        ),
    ),
)
def test_executable_acceptance_gate_fails_each_required_condition(
    overrides, failed_field
):
    values = {
        "zero_policy_success_rate": 0.75,
        "oracle_success_rate": 0.99,
        "zero_policy_raw_safety_failures": 0,
        "oracle_raw_safety_failures": 0,
        "zero_policy_failure_reasons": ("time_limit",),
        "paired_randomization_matches": (True,) * 100,
    }
    values.update(overrides)
    result = evaluate_challenge_acceptance(_load(), **values)
    assert getattr(result, failed_field) is False
    assert result.passed is False


@pytest.mark.parametrize(
    "overrides,message",
    (
        ({"zero_policy_success_rate": math.nan}, "finite"),
        ({"oracle_success_rate": 1.01}, r"\[0, 1\]"),
        ({"zero_policy_raw_safety_failures": True}, "integer"),
        ({"paired_randomization_matches": (True,) * 99}, "all final"),
        (
            {"paired_randomization_matches": (True,) * 99 + (1,)},
            "only booleans",
        ),
    ),
)
def test_executable_acceptance_rejects_malformed_evidence(
    overrides, message
):
    values = {
        "zero_policy_success_rate": 0.75,
        "oracle_success_rate": 0.99,
        "zero_policy_raw_safety_failures": 0,
        "oracle_raw_safety_failures": 0,
        "zero_policy_failure_reasons": ("time_limit",),
        "paired_randomization_matches": (True,) * 100,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        evaluate_challenge_acceptance(_load(), **values)


def test_challenge_document_is_finite_json_safe_and_immutable():
    challenge = _load()
    document = challenge.as_dict()
    encoded = json.dumps(document, allow_nan=False, sort_keys=True)
    assert "NaN" not in encoded
    assert document["enabled"] is False
    assert document["action_size"] == 4
    assert isinstance(
        document["control_path"]["clamp_offset_lower_rad"], list
    )
    with pytest.raises(Exception):
        challenge.enabled = True


@pytest.mark.parametrize(
    "mutator,message",
    (
        (
            lambda document: document.update({"unexpected": True}),
            "keys are invalid",
        ),
        (
            lambda document: document.update({"enabled": True}),
            "remain disabled",
        ),
        (
            lambda document: document.update({"enabled": 0}),
            "boolean",
        ),
        (
            lambda document: document["contract"].update(
                {"action_size": 5}
            ),
            "remain 4",
        ),
        (
            lambda document: document["contract"].update(
                {"observation_size": 25}
            ),
            "remain 24",
        ),
        (
            lambda document: document["task"].update(
                {"target_angle_degrees": float("nan")}
            ),
            "finite",
        ),
        (
            lambda document: document["task"].update(
                {"maximum_episode_steps": 29}
            ),
            "exactly 28",
        ),
        (
            lambda document: document[
                "control_path_randomization"
            ]["q7_velocity_scale"].update({"lower": 0.84}),
            "exactly 0.85",
        ),
        (
            lambda document: document[
                "control_path_randomization"
            ]["clamp_position_offset"].update(
                {"upper_rad": [0.016, 0.015, 0.015]}
            ),
            r"\+/-0.015",
        ),
        (
            lambda document: document["deadline"].update(
                {"counts_as_safety_failure": True}
            ),
            "must not count",
        ),
        (
            lambda document: document["fixed_physics"].update(
                {"mass_randomized": True}
            ),
            "must remain fixed",
        ),
        (
            lambda document: document["seed_ranges"]["tuning"].update(
                {"end_inclusive": 20064}
            ),
            "must be exactly",
        ),
        (
            lambda document: document["acceptance"].update(
                {"oracle_minimum_success_rate": 0.97}
            ),
            "exactly 0.98",
        ),
        (
            lambda document: document["acceptance"].update(
                {"maximum_oracle_raw_safety_failures": 1}
            ),
            "both be zero",
        ),
        (
            lambda document: document["acceptance"].update(
                {"zero_policy_allowed_failure_reasons": ["cross_thread"]}
            ),
            "only be time_limit",
        ),
        (
            lambda document: document["acceptance"].update(
                {"require_all_paired_randomization_matches": False}
            ),
            "all paired",
        ),
    ),
)
def test_schema_and_safety_drift_fail_closed(
    tmp_path, mutator, message
):
    path = _invalid_document(tmp_path, mutator)
    with pytest.raises(ValueError, match=message):
        load_residual_learning_challenge(path)


def test_import_does_not_load_runtime_training_or_simulator_modules():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE_ROOT)
    script = """
import importlib
import sys

module = importlib.import_module(
    "kcg_connector.residual_learning_challenge"
)
module.load_residual_learning_challenge()
for name in ("torch", "gymnasium", "stable_baselines3", "isaacsim", "omni"):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
