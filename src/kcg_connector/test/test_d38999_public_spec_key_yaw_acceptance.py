import math

import numpy as np
import pytest

from kcg_connector.d38999_key_yaw_acceptance import (
    DEFAULT_SIMULATION_MINIMUM_SAMPLES,
    SIMULATION_THRESHOLD_LABEL,
    evaluate_public_spec_sim_key_yaw_acceptance,
)
from kcg_connector.d38999_keyed_public_spec_v2 import PLUG_MODEL_ID


def _evaluate(error_deg: float, **overrides):
    count = overrides.pop("minimum_samples", DEFAULT_SIMULATION_MINIMUM_SAMPLES)
    truth = np.linspace(-math.pi, math.pi, count, endpoint=False)
    estimate = truth + math.radians(error_deg)
    arguments = {
        "keyed_model_id": PLUG_MODEL_ID,
        "minimum_samples": count,
        "dataset_tag": "heldout_public_spec_v2_seed_partition",
        "withheld_truth": True,
    }
    arguments.update(overrides)
    return evaluate_public_spec_sim_key_yaw_acceptance(
        estimate, truth, **arguments
    )


def test_default_public_spec_stress_profile_passes_shadow_only():
    result = _evaluate(0.02)
    assert result["status"] == "PASSED_PUBLIC_SPEC_SIMULATION_SHADOW_ONLY"
    assert result["profile_name"] == "adversarial_gdt_stress"
    assert result["clearance_derivation_kind"] == (
        "project_adversarial_gdt_stress_assumption"
    )
    assert result["drawing_specified_mechanical_yaw_clearance"] is False
    assert result["threshold_label"] == SIMULATION_THRESHOLD_LABEL
    assert result["required_yaw_error_p95_deg"] == pytest.approx(
        0.030275467425980793
    )
    assert result["passed"] is True
    assert result["shadow_authorized"] is True
    assert result["selected_for_control_allowed"] is False
    assert result["simulation_insertion_control_authorized"] is False
    assert result["robot_control_authorized"] is False
    assert result["hardware_control_authorized"] is False
    assert result["real_measured_clearance_deg"] is None


def test_public_spec_stress_profile_rejects_above_half_window():
    result = _evaluate(0.031)
    assert result["status"] == "REJECTED_PUBLIC_SPEC_SIM_YAW_P95_THRESHOLD"
    assert result["passed"] is False
    assert result["shadow_authorized"] is False
    assert result["reason"] == (
        "P95_MUST_BE_STRICTLY_BELOW_HALF_DERIVED_SIM_CLEARANCE"
    )


def test_tight_size_profile_is_reported_separately():
    result = _evaluate(0.2, profile_name="tight_size_centered")
    assert result["status"] == "PASSED_PUBLIC_SPEC_SIMULATION_SHADOW_ONLY"
    assert result["derived_peak_to_peak_clearance_deg"] == pytest.approx(
        0.6661267828482448
    )
    assert result["required_yaw_error_p95_deg"] == pytest.approx(
        0.3330633914241224
    )


def test_wrong_model_id_and_too_few_samples_are_fail_closed():
    with pytest.raises(ValueError, match="keyed_model_id"):
        _evaluate(0.01, keyed_model_id="fake-keyed-v2")
    with pytest.raises(ValueError, match="at least"):
        _evaluate(0.01, minimum_samples=999)


def test_nonwithheld_or_nonfinite_data_cannot_pass():
    result = _evaluate(0.01, withheld_truth=False)
    assert result["status"] == "REJECTED_NOT_WITHHELD_TRUTH"
    assert result["shadow_authorized"] is False

    count = DEFAULT_SIMULATION_MINIMUM_SAMPLES
    truth = np.zeros(count)
    estimate = np.zeros(count)
    estimate[10] = np.nan
    result = evaluate_public_spec_sim_key_yaw_acceptance(
        estimate,
        truth,
        keyed_model_id=PLUG_MODEL_ID,
        dataset_tag="heldout_nonfinite_probe",
        withheld_truth=True,
    )
    assert result["status"] == "REJECTED_NONFINITE_YAW"
    assert result["shadow_authorized"] is False
