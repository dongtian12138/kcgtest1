"""Pure-CPU tests for the real-clearance keyed-yaw acceptance gate."""

import math

import numpy as np
import pytest

from kcg_connector.d38999_key_yaw_acceptance import (
    DEFAULT_MINIMUM_SAMPLES,
    THRESHOLD_LABEL,
    evaluate_key_yaw_acceptance,
)


def _constant_error(error_deg, *, count=DEFAULT_MINIMUM_SAMPLES):
    truth = np.zeros(count, dtype=np.float64)
    estimated = np.full(count, math.radians(error_deg), dtype=np.float64)
    return estimated, truth


def _evaluate(error_deg, *, clearance_deg=10.0, withheld_truth=True, count=30):
    estimated, truth = _constant_error(error_deg, count=count)
    return evaluate_key_yaw_acceptance(
        estimated,
        truth,
        clearance_deg,
        dataset_tag="real-keyed-v2-withheld-001",
        withheld_truth=withheld_truth,
    )


def test_wraps_error_across_minus_pi_plus_pi_boundary():
    estimated = np.full(30, math.radians(-179.0))
    truth = np.full(30, math.radians(179.0))

    result = evaluate_key_yaw_acceptance(
        estimated,
        truth,
        6.0,
        dataset_tag="wrap-boundary-withheld",
        withheld_truth=True,
    )

    assert result["observed_yaw_error_p95_deg"] == pytest.approx(2.0)
    assert result["required_yaw_error_p95_deg"] == pytest.approx(3.0)
    assert result["passed"] is True


def test_unknown_real_clearance_is_blocked_even_with_perfect_estimates():
    estimated, truth = _constant_error(0.0)

    result = evaluate_key_yaw_acceptance(
        estimated,
        truth,
        None,
        dataset_tag="clearance-not-measured",
        withheld_truth=True,
    )

    assert result["status"] == "BLOCKED_REAL_CLEARANCE_UNKNOWN"
    assert result["required_yaw_error_p95_deg"] is None
    assert result["passed"] is False
    assert result["shadow_authorized"] is False
    assert result["control_authorized"] is False
    assert result["threshold_label"] == THRESHOLD_LABEL


@pytest.mark.parametrize("clearance", [0.0, -1.0, math.nan, math.inf, True, "5"])
def test_invalid_or_nonpositive_clearance_raises(clearance):
    estimated, truth = _constant_error(0.0)
    with pytest.raises(ValueError, match="clearance"):
        evaluate_key_yaw_acceptance(
            estimated,
            truth,
            clearance,
            dataset_tag="invalid-clearance",
            withheld_truth=True,
        )


def test_insufficient_sample_count_is_structured_rejection():
    result = _evaluate(0.0, count=DEFAULT_MINIMUM_SAMPLES - 1)

    assert result["status"] == "REJECTED_INSUFFICIENT_SAMPLES"
    assert result["sample_count"] == DEFAULT_MINIMUM_SAMPLES - 1
    assert result["passed"] is False
    assert result["control_authorized"] is False


def test_non_withheld_truth_is_structured_rejection():
    result = _evaluate(0.0, withheld_truth=False)

    assert result["status"] == "REJECTED_NOT_WITHHELD_TRUTH"
    assert result["withheld_truth"] is False
    assert result["passed"] is False
    assert result["control_authorized"] is False


def test_p95_equal_to_half_clearance_fails_strict_inequality():
    result = _evaluate(5.0, clearance_deg=10.0)

    assert result["observed_yaw_error_p95_deg"] == pytest.approx(5.0)
    assert result["required_yaw_error_p95_deg"] == pytest.approx(5.0)
    assert result["status"] == "REJECTED_YAW_P95_THRESHOLD"
    assert result["passed"] is False
    assert result["shadow_authorized"] is False


def test_p95_strictly_below_half_clearance_passes_evaluation_only():
    result = _evaluate(4.0, clearance_deg=10.0)

    assert result["status"] == "PASSED_EVALUATION_ONLY"
    assert result["passed"] is True
    assert result["shadow_authorized"] is True
    assert result["authorization_scope"] == "EVALUATION_ONLY_NO_CONTROL"
    assert result["control_authorized"] is False


def test_nonfinite_yaw_is_structured_rejection():
    estimated, truth = _constant_error(0.0)
    estimated[7] = math.nan

    result = evaluate_key_yaw_acceptance(
        estimated,
        truth,
        10.0,
        dataset_tag="nonfinite-yaw",
        withheld_truth=True,
    )

    assert result["status"] == "REJECTED_NONFINITE_YAW"
    assert result["observed_yaw_error_p95_deg"] is None
    assert result["passed"] is False
    assert result["control_authorized"] is False


@pytest.mark.parametrize(
    ("estimated", "truth"),
    [
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.zeros(30), np.zeros(31)),
    ],
)
def test_invalid_yaw_shapes_raise(estimated, truth):
    with pytest.raises(ValueError, match="one-dimensional|same shape"):
        evaluate_key_yaw_acceptance(
            estimated,
            truth,
            10.0,
            dataset_tag="invalid-shape",
            withheld_truth=True,
        )


def test_control_authorization_is_false_for_every_result_class():
    estimated, truth = _constant_error(0.0)
    results = [
        _evaluate(0.0),
        _evaluate(9.0),
        _evaluate(0.0, count=29),
        _evaluate(0.0, withheld_truth=False),
        evaluate_key_yaw_acceptance(
            estimated,
            truth,
            None,
            dataset_tag="unknown-clearance",
            withheld_truth=True,
        ),
    ]

    assert all(result["control_authorized"] is False for result in results)
