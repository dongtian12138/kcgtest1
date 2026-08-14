import numpy as np
import pytest

from kcg_connector.wrist_ft_calibration import (
    WRENCH_AXIS_NAMES,
    analyze_axis_calibration,
)


def calibration_cases(raw_from_applied=None):
    response = (
        -np.eye(6, dtype=np.float64)
        if raw_from_applied is None
        else np.asarray(raw_from_applied, dtype=np.float64)
    )
    magnitudes = np.asarray([4.0, 4.0, 4.0, 0.4, 0.4, 0.4])
    cases = {}
    for index, axis_name in enumerate(WRENCH_AXIS_NAMES):
        full = response[:, index] * magnitudes[index]
        half = response[:, index] * magnitudes[index] * 0.5
        cases[axis_name] = {
            "plus_full": full,
            "minus_full": -full,
            "plus_half": half,
            "minus_half": -half,
        }
    return cases


def analyze(cases):
    return analyze_axis_calibration(
        cases,
        force_magnitude_n=4.0,
        torque_magnitude_nm=0.4,
    )


def test_negative_identity_yields_canonical_environment_on_tool_mapping():
    result = analyze(calibration_cases())
    assert result["passed"] is True
    assert result["mapping_is_signed_permutation"] is True
    assert np.array_equal(
        result["canonical_from_raw_sign_permutation"], -np.eye(6)
    )
    assert all(
        record["raw_to_canonical_sign"] == -1
        for record in result["axis_records"]
    )


def test_permutation_and_sign_are_inferred_within_force_and_torque_groups():
    response = np.zeros((6, 6), dtype=np.float64)
    response[[1, 2, 0], [0, 1, 2]] = [1.0, -1.0, 1.0]
    response[[5, 3, 4], [3, 4, 5]] = [-1.0, 1.0, -1.0]
    result = analyze(calibration_cases(response))
    mapping = np.asarray(result["canonical_from_raw_sign_permutation"])
    assert result["passed"] is True
    assert np.allclose(mapping @ response, np.eye(6))


def test_cross_axis_response_fails_closed():
    response = -np.eye(6, dtype=np.float64)
    response[1, 0] = 0.06
    result = analyze(calibration_cases(response))
    assert result["passed"] is False
    assert result["axis_records"][0]["same_kind_cross_axis_ratio"] == 0.06


def test_nonodd_or_nonlinear_response_fails_closed():
    cases = calibration_cases()
    cases["Fz"]["plus_full"] = (
        np.asarray(cases["Fz"]["plus_full"]) + [0.0, 0.0, 0.5, 0, 0, 0]
    )
    result = analyze(cases)
    assert result["passed"] is False
    fz = result["axis_records"][2]
    assert fz["full_odd_symmetry_error_ratio"] > 0.05


def test_rejects_missing_nonfinite_or_bad_magnitude_inputs():
    cases = calibration_cases()
    del cases["Tz"]
    with pytest.raises(ValueError, match="exactly the six"):
        analyze(cases)

    cases = calibration_cases()
    cases["Tx"]["plus_full"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analyze(cases)

    with pytest.raises(ValueError, match="positive"):
        analyze_axis_calibration(
            calibration_cases(),
            force_magnitude_n=0.0,
            torque_magnitude_nm=0.4,
        )
