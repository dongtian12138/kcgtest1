import inspect
import math

import numpy as np
import pytest

from kcg_connector.d38999_key_region_detector import (
    EXPECTED_MASTER_TO_MINOR_WIDTH_RATIO,
    MASTER_KEY_WIDTH_MM,
    MINOR_KEY_WIDTH_MM,
    SUPPORTED_KEYED_MODEL_ID,
    detect_key_region_from_palm_rgbd,
)


KEY_ANGLES_DEG = (0.0, 80.0, 142.0, 196.0, 293.0)


def _synthetic_observation(
    *,
    key_angles_deg=KEY_ANGLES_DEG,
    key_widths_deg=(16.0, 8.0, 8.0, 8.0, 8.0),
    center_uv=(90.0, 90.0),
    shape=(181, 181),
    body_radius_px=48.0,
    protrusion_height_px=10.0,
):
    rows, columns = np.indices(shape, dtype=np.float64)
    du = columns - center_uv[0]
    dv = rows - center_uv[1]
    radii = np.hypot(du, dv)
    angles = np.mod(np.arctan2(dv, du), 2.0 * math.pi)
    face = radii <= body_radius_px
    for angle_deg, width_deg in zip(key_angles_deg, key_widths_deg):
        angular_difference = np.abs(
            np.angle(
                np.exp(1j * (angles - math.radians(float(angle_deg))))
            )
        )
        face |= (
            angular_difference <= math.radians(float(width_deg)) / 2.0
        ) & (radii <= body_radius_px + protrusion_height_px)
    depth = np.full(shape, np.nan, dtype=np.float64)
    depth[face] = 0.42
    occlusion = np.zeros(shape, dtype=np.bool_)
    return face, depth, occlusion


def _detect(face, depth, occlusion, **overrides):
    arguments = {
        "connector_face_mask": face,
        "depth_m": depth,
        "face_center_uv": (90.0, 90.0),
        "keyed_model_id": SUPPORTED_KEYED_MODEL_ID,
        "occlusion_mask": occlusion,
    }
    arguments.update(overrides)
    return detect_key_region_from_palm_rgbd(**arguments)


def _assert_shadow_only(result):
    assert result["shadow_only"] is True
    assert result["control_authorized"] is False
    assert result["confidence_calibrated"] is False
    assert "selected_for_control" not in result
    assert "object_pose_truth" not in result
    assert "semantic_segmentation_truth" not in result


def test_clear_five_key_face_detects_unique_wide_master_key():
    face, depth, occlusion = _synthetic_observation()

    result = _detect(face, depth, occlusion)

    assert result["status"] == "KEY_DIRECTION_DETECTED_SHADOW_ONLY"
    assert result["passed"] is True
    assert result["reason"] is None
    assert np.allclose(result["key_direction_uv"], (1.0, 0.0), atol=0.02)
    probability = result["key_probability"]
    assert probability.shape == face.shape
    assert np.all((0.0 <= probability) & (probability <= 1.0))
    assert np.count_nonzero(probability) >= 12
    assert np.count_nonzero(probability[:, :90]) == 0

    quality = result["quality_diagnostics"]
    assert quality["candidate_count"] == 5
    assert quality["master_candidate_index"] in range(5)
    assert quality["detection_confidence"] >= quality[
        "minimum_detection_confidence"
    ]
    assert quality["master_to_second_width_ratio"] > 1.35
    assert quality["n_pattern_consistency_score"] > 0.9
    assert math.isclose(
        quality["reference_master_to_minor_width_ratio"],
        EXPECTED_MASTER_TO_MINOR_WIDTH_RATIO,
    )
    assert quality["reference_master_key_width_mm"] == MASTER_KEY_WIDTH_MM
    assert quality["reference_minor_key_width_mm"] == MINOR_KEY_WIDTH_MM
    _assert_shadow_only(result)


def test_master_direction_rotates_with_the_observed_contour():
    angles = tuple((angle + 142.0) % 360.0 for angle in KEY_ANGLES_DEG)
    face, depth, occlusion = _synthetic_observation(key_angles_deg=angles)

    result = _detect(face, depth, occlusion)

    expected = (
        math.cos(math.radians(142.0)),
        math.sin(math.radians(142.0)),
    )
    assert result["passed"] is True
    assert np.allclose(result["key_direction_uv"], expected, atol=0.03)
    _assert_shadow_only(result)


def test_mirrored_image_axes_accept_only_the_same_n_pattern():
    mirrored_angles = tuple((-angle) % 360.0 for angle in KEY_ANGLES_DEG)
    face, depth, occlusion = _synthetic_observation(
        key_angles_deg=mirrored_angles
    )

    result = _detect(face, depth, occlusion)

    assert result["passed"] is True
    assert result["quality_diagnostics"]["image_pattern_chirality"] == (
        "MIRRORED_IMAGE_UV"
    )
    assert np.allclose(result["key_direction_uv"], (1.0, 0.0), atol=0.02)
    _assert_shadow_only(result)


def test_wrong_five_bump_pattern_cannot_impersonate_n_polarization():
    face, depth, occlusion = _synthetic_observation(
        key_angles_deg=(0.0, 72.0, 144.0, 216.0, 288.0)
    )

    result = _detect(face, depth, occlusion)

    assert result["status"] == "LOW_CONFIDENCE"
    assert result["reason"] == "N_POLARIZATION_KEY_PATTERN_INCONSISTENT"
    quality = result["quality_diagnostics"]
    assert quality["maximum_minor_pattern_angle_error_deg"] > quality[
        "maximum_allowed_minor_pattern_angle_error_deg"
    ]
    _assert_shadow_only(result)


def test_only_allowed_observation_fields_are_in_the_public_signature():
    signature = inspect.signature(detect_key_region_from_palm_rgbd)
    assert tuple(signature.parameters) == (
        "connector_face_mask",
        "depth_m",
        "face_center_uv",
        "keyed_model_id",
        "occlusion_mask",
    )
    assert signature.parameters["occlusion_mask"].kind is inspect.Parameter.KEYWORD_ONLY

    face, depth, occlusion = _synthetic_observation()
    with pytest.raises(TypeError):
        detect_key_region_from_palm_rgbd(
            face,
            depth,
            (90.0, 90.0),
            SUPPORTED_KEYED_MODEL_ID,
            occlusion_mask=occlusion,
            object_pose_truth=np.eye(4),
        )


def test_missing_occlusion_mask_fails_closed():
    face, depth, _ = _synthetic_observation()

    result = _detect(face, depth, None)

    assert result["status"] == "OCCLUSION_UNKNOWN"
    assert result["reason"] == "OCCLUSION_MASK_MISSING"
    assert result["rejection_code"] == "KEY_REGION_OCCLUSION_UNKNOWN"
    assert np.count_nonzero(result["key_probability"]) == 0
    _assert_shadow_only(result)


def test_occlusion_over_a_key_fails_closed():
    face, depth, occlusion = _synthetic_observation()
    occlusion[85:96, 136:151] = True

    result = _detect(face, depth, occlusion)

    assert result["status"] == "OCCLUDED"
    assert result["reason"] == "OCCLUSION_INTERSECTS_CONNECTOR_ROI"
    assert result["rejection_code"] == "KEY_REGION_OCCLUDED"
    assert result["quality_diagnostics"]["occluded_connector_roi_pixels"] > 0
    _assert_shadow_only(result)


def test_occlusion_far_outside_connector_roi_does_not_fake_a_key_occlusion():
    face, depth, occlusion = _synthetic_observation()
    occlusion[2:8, 2:8] = True

    result = _detect(face, depth, occlusion)

    assert result["passed"] is True
    _assert_shadow_only(result)


def test_face_cut_by_image_border_is_out_of_frame():
    face, depth, occlusion = _synthetic_observation(center_uv=(52.0, 90.0))

    result = _detect(
        face,
        depth,
        occlusion,
        face_center_uv=(52.0, 90.0),
    )

    assert result["status"] == "OUT_OF_FRAME"
    assert result["reason"] == "CONNECTOR_FACE_TOUCHES_IMAGE_BORDER"
    assert result["rejection_code"] == "CONNECTOR_FACE_OUT_OF_FRAME"
    _assert_shadow_only(result)


def test_face_center_outside_image_is_out_of_frame():
    face, depth, occlusion = _synthetic_observation()

    result = _detect(
        face,
        depth,
        occlusion,
        face_center_uv=(999.0, 90.0),
    )

    assert result["status"] == "OUT_OF_FRAME"
    assert result["reason"] == "FACE_CENTER_OUTSIDE_IMAGE"
    _assert_shadow_only(result)


def test_one_missing_face_depth_pixel_fails_closed():
    face, depth, occlusion = _synthetic_observation()
    depth[90, 140] = np.nan
    assert face[90, 140]

    result = _detect(face, depth, occlusion)

    assert result["status"] == "DEPTH_MISSING"
    assert result["reason"] == "CONNECTOR_FACE_DEPTH_INCOMPLETE"
    assert result["rejection_code"] == "KEY_REGION_DEPTH_MISSING"
    assert result["quality_diagnostics"]["valid_face_depth_fraction"] < 1.0
    _assert_shadow_only(result)


@pytest.mark.parametrize(
    "key_angles,key_widths,expected_count",
    (
        (KEY_ANGLES_DEG[:4], (16.0, 8.0, 8.0, 8.0), 4),
        (
            (0.0, 80.0, 142.0, 196.0, 245.0, 293.0),
            (16.0, 8.0, 8.0, 8.0, 8.0, 8.0),
            6,
        ),
    ),
)
def test_candidate_count_must_be_exactly_five(
    key_angles, key_widths, expected_count
):
    face, depth, occlusion = _synthetic_observation(
        key_angles_deg=key_angles,
        key_widths_deg=key_widths,
    )

    result = _detect(face, depth, occlusion)

    assert result["status"] == "LOW_CONFIDENCE"
    assert result["reason"] == "KEY_CANDIDATE_COUNT_NOT_FIVE"
    assert result["quality_diagnostics"]["candidate_count"] == expected_count
    _assert_shadow_only(result)


@pytest.mark.parametrize(
    "widths",
    (
        (8.0, 8.0, 8.0, 8.0, 8.0),
        (16.0, 16.0, 8.0, 8.0, 8.0),
    ),
)
def test_master_key_width_must_be_unique(widths):
    face, depth, occlusion = _synthetic_observation(key_widths_deg=widths)

    result = _detect(face, depth, occlusion)

    assert result["status"] == "AMBIGUOUS"
    assert result["reason"] == "MASTER_KEY_WIDTH_NOT_UNIQUE"
    assert result["rejection_code"] == "KEY_DIRECTION_AMBIGUOUS"
    assert result["quality_diagnostics"]["candidate_count"] == 5
    _assert_shadow_only(result)


def test_barely_wider_master_key_is_rejected_as_low_confidence():
    face, depth, occlusion = _synthetic_observation(
        key_widths_deg=(13.0, 8.0, 8.0, 8.0, 8.0)
    )

    result = _detect(face, depth, occlusion)

    assert result["status"] == "LOW_CONFIDENCE"
    assert result["reason"] == "MASTER_KEY_CONFIDENCE_BELOW_LIMIT"
    quality = result["quality_diagnostics"]
    assert quality["master_to_second_width_ratio"] >= 1.35
    assert quality["detection_confidence"] < quality[
        "minimum_detection_confidence"
    ]
    _assert_shadow_only(result)


@pytest.mark.parametrize(
    "model_id",
    (
        None,
        "",
        "d38999_shell25j_proxy_v1",
        "C2_LINKED_BRANCH_0",
        "d38999_20kj61pn_keyed_proxy_v2",
        "d38999_26kj61sn_keyed_proxy_v2_typo",
    ),
)
def test_missing_old_or_nonexact_model_identity_fails_closed(model_id):
    face, depth, occlusion = _synthetic_observation()

    result = _detect(
        face,
        depth,
        occlusion,
        keyed_model_id=model_id,
    )

    assert result["status"] == "MODEL_NOT_KEYED_V2"
    assert result["rejection_code"] == "KEYED_MODEL_ID_UNAVAILABLE"
    assert result["passed"] is False
    _assert_shadow_only(result)


@pytest.mark.parametrize(
    "overrides",
    (
        {"connector_face_mask": np.zeros((20, 20), dtype=np.uint8)},
        {"connector_face_mask": np.zeros((20, 20, 1), dtype=np.bool_)},
        {"depth_m": np.zeros((20, 20, 1), dtype=np.float64)},
        {"depth_m": np.zeros((180, 181), dtype=np.float64)},
        {"face_center_uv": (90.0,)},
        {"face_center_uv": (math.nan, 90.0)},
        {"occlusion_mask": np.zeros((181, 181), dtype=np.uint8)},
        {"occlusion_mask": np.zeros((180, 181), dtype=np.bool_)},
    ),
)
def test_invalid_programming_shapes_or_dtypes_raise_value_error(overrides):
    face, depth, occlusion = _synthetic_observation()
    arguments = {
        "connector_face_mask": face,
        "depth_m": depth,
        "face_center_uv": (90.0, 90.0),
        "keyed_model_id": SUPPORTED_KEYED_MODEL_ID,
        "occlusion_mask": occlusion,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError):
        detect_key_region_from_palm_rgbd(**arguments)
