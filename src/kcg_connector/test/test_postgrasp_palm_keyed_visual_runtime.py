"""Pure-CPU contracts for the atomic Palm keyed observation/control gate."""

from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.d38999_key_yaw_acceptance import (
    evaluate_public_spec_sim_key_yaw_acceptance,
)
from kcg_connector.d38999_keyed_public_spec_v2 import PAIR_MODEL_ID, PLUG_MODEL_ID


RUNTIME_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "postgrasp_palm_keyed_visual_runtime.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "postgrasp_palm_keyed_visual_runtime", RUNTIME_PATH
)
runtime = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(runtime)


def _rx(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
    )


def _ry(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
    )


def _rz(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def _project(point, intrinsics):
    return np.asarray(
        (
            intrinsics["fx"] * point[0] / point[2] + intrinsics["cx"],
            intrinsics["fy"] * point[1] / point[2] + intrinsics["cy"],
        )
    )


def _palm_observation(yaw_deg=35.0):
    height, width = 720, 1280
    intrinsics = {
        "fx": 420.0,
        "fy": 420.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
        "width": width,
        "height": height,
    }
    tilt = _ry(math.radians(-12.0)) @ _rx(math.radians(18.0))
    translation = np.asarray((0.0, 0.0, 0.5))
    normal = tilt[:, 2]
    plane_offset = -float(np.dot(normal, translation))

    rows, columns = np.indices((height, width))
    rays = np.stack(
        (
            (columns - intrinsics["cx"]) / intrinsics["fx"],
            (rows - intrinsics["cy"]) / intrinsics["fy"],
            np.ones((height, width)),
        ),
        axis=-1,
    )
    scale = -plane_offset / np.einsum("hwk,k->hw", rays, normal)
    center_uv = _project(translation, intrinsics)
    actual_rotation = tilt @ _rz(math.radians(yaw_deg))
    key_point = translation + actual_rotation @ np.asarray((0.050, 0.0, 0.0))
    projected_key = _project(key_point, intrinsics)
    master_angle = math.atan2(
        projected_key[1] - center_uv[1], projected_key[0] - center_uv[0]
    )
    du = columns - center_uv[0]
    dv = rows - center_uv[1]
    radius = np.hypot(du, dv)
    pixel_angle = np.mod(np.arctan2(dv, du), 2.0 * math.pi)
    face = radius <= 48.0
    for offset_deg, width_deg in zip(
        (0.0, 80.0, 142.0, 196.0, 293.0),
        (16.0, 8.0, 8.0, 8.0, 8.0),
    ):
        angular_difference = np.abs(
            np.angle(
                np.exp(
                    1j
                    * (pixel_angle - master_angle - math.radians(offset_deg))
                )
            )
        )
        face |= (angular_difference <= math.radians(width_deg) / 2.0) & (
            radius <= 58.0
        )
    depth = np.where(face, scale, np.nan)
    rgb = np.full((height, width, 3), 28, dtype=np.uint8)
    rgb[face] = (156, 162, 168)

    candidates = []
    for hypothesis_id, candidate_yaw in (("YAW_0", 0.0), ("YAW_PI", math.pi)):
        transform = np.eye(4)
        transform[:3, :3] = tilt @ _rz(candidate_yaw)
        transform[:3, 3] = translation
        candidates.append(
            {
                "hypothesis_id": hypothesis_id,
                "T_camera_plug": transform,
                "axial_yaw_rad": candidate_yaw,
            }
        )
    return {
        "rgb": rgb,
        "depth_m": depth,
        "connector_face_mask": face,
        "face_center_uv": center_uv,
        "occlusion_mask": np.zeros_like(face),
        "camera_intrinsics": intrinsics,
        "c2_candidates": candidates,
    }


def _passing_yaw_gate():
    truth = np.linspace(-math.pi, math.pi, 1000, endpoint=False)
    estimates = truth + math.radians(0.01)
    report = evaluate_public_spec_sim_key_yaw_acceptance(
        estimates,
        truth,
        keyed_model_id=PLUG_MODEL_ID,
        dataset_tag="cpu_contract_hypothetical_same_camera_pass",
        withheld_truth=True,
    )
    assert report["passed"] is True
    return report


def _t_hc():
    transform = np.eye(4)
    transform[:3, 3] = (0.0, 0.0, 0.315)
    return transform


def _t_wr_fixed():
    transform = np.eye(4)
    transform[:3, :3] = np.diag((1.0, -1.0, -1.0))
    transform[:3, 3] = (0.55, 0.185, 0.24)
    return transform


def _accepted_kwargs(yaw_deg=35.0):
    return {
        "scene_schema_version": runtime.EXPECTED_SCENE_SCHEMA_VERSION,
        "scene_profile_id": PAIR_MODEL_ID,
        "fixed_orientation_token": runtime.EXPECTED_FIXED_ORIENTATION_TOKEN,
        "keyed_model_id": PLUG_MODEL_ID,
        "camera_contract": {
            "camera_name": "Palm",
            "parent_frame": "handbase_link",
            "resolution_px": [1280, 720],
            "channels_exactly": ["rgb", "distance_to_image_plane"],
            "near_clip_m": 0.02,
            "T_HC_source": "FROZEN_SCENE_CONFIG",
            "camera_contract_id": runtime.EXPECTED_CAMERA_CONTRACT_ID,
        },
        **_palm_observation(yaw_deg),
        "image_mask_quality": {
            "schema_version": runtime.MASK_QUALITY_SCHEMA_VERSION,
            "source": "PALM_RGB_IMAGE_ONLY",
            "status": "ACCEPTED",
            "passed": True,
            "confidence_calibrated": True,
            "observed_confidence": 0.96,
            "minimum_confidence": 0.90,
            "camera_contract_id": runtime.EXPECTED_CAMERA_CONTRACT_ID,
        },
        "pose_quality": {
            "schema_version": runtime.POSE_QUALITY_SCHEMA_VERSION,
            "source": "PALM_RGBD_C2_ESTIMATOR",
            "status": "ACCEPTED",
            "pose_valid": True,
            "quality_gate_passed": True,
            "covariance_calibration_status": "CALIBRATED",
            "calibration_id": "hypothetical_cpu_contract_calibration",
            "scene_profile_id": PAIR_MODEL_ID,
            "camera_contract_id": runtime.EXPECTED_CAMERA_CONTRACT_ID,
        },
        "yaw_acceptance": _passing_yaw_gate(),
        "accuracy_gate": {
            "schema_version": runtime.ACCURACY_GATE_SCHEMA_VERSION,
            "status": "ACCEPTED",
            "accuracy_gate_passed": True,
            "camera_contract_id": runtime.EXPECTED_CAMERA_CONTRACT_ID,
            "scene_profile_id": PAIR_MODEL_ID,
            "keyed_model_id": PLUG_MODEL_ID,
            "calibration_id": "hypothetical_cpu_contract_calibration",
            "yaw_dataset_tag": "cpu_contract_hypothetical_same_camera_pass",
        },
        "actual_arm_q_before_capture": np.zeros(7),
        "actual_arm_q_after_capture": np.full(7, 0.0001),
        "T_HC_frozen_configured": _t_hc(),
        "T_WH_from_actual_q": np.eye(4),
        "T_WR_fixed_configured": _t_wr_fixed(),
        "T_RP_target_configured": np.eye(4),
    }


@pytest.fixture(scope="module")
def accepted_result():
    return runtime.evaluate_postgrasp_palm_keyed_visual_control(
        **_accepted_kwargs()
    )


def test_exact_keyed_scene_and_all_calibrated_gates_emit_one_pure_plan(
    accepted_result,
):
    result = accepted_result
    assert result["status"] == "VISUAL_OBSERVATION_AND_CONTROL_PLAN_ACCEPTED"
    assert result["observation_passed"] is True
    assert result["plan_authorized"] is True
    assert result["simulation_prealign_target_authorized"] is True
    assert result["continue_to_path_planner_authorized"] is True
    assert result["control_authorized"] is False
    assert result["simulation_prealign_control_authorized"] is False
    assert result["simulation_insertion_control_authorized"] is False
    assert result["hardware_control_authorized"] is False
    assert result["safe_stop_required"] is False
    assert result["selected_hypothesis_id"] == "YAW_0"
    assert math.degrees(result["estimated_axial_yaw_rad"]) == pytest.approx(
        35.0, abs=0.15
    )
    assert result["actuator_command_issued"] is False
    assert result["truth_correction_applied"] is False


def test_plan_transform_chain_uses_visual_hp_and_configured_fixed_target(
    accepted_result,
):
    result = accepted_result
    t_hp = np.asarray(result["T_HP_selected"])
    t_wp_target = np.asarray(result["T_WP_target"])
    t_wh_target = np.asarray(result["T_WH_target"])
    assert np.all(np.isfinite(t_hp))
    assert np.allclose(t_wp_target, _t_wr_fixed())
    assert np.allclose(t_wh_target @ t_hp, t_wp_target)
    assert np.allclose(
        result["target_plan"]["T_WH_target"], result["T_WH_target"]
    )
    assert result["target_plan"]["requires_collision_checked_path_planner"] is True
    assert result["target_plan"]["insertion_motion_included"] is False


def test_strict_c2_branch_projections_are_derived_from_keyed_plus_x(
    accepted_result,
):
    records = accepted_result["keyed_branch_projections"]
    assert [item["hypothesis_id"] for item in records] == ["YAW_0", "YAW_PI"]
    assert all(item["source_axis"] == "KEYED_PLUG_FRAME_PLUS_X" for item in records)
    assert np.dot(records[0]["direction_uv"], records[1]["direction_uv"]) < -0.99


def test_current_uncalibrated_pose_quality_observes_but_safely_stops():
    kwargs = _accepted_kwargs()
    kwargs["pose_quality"].update(
        status="REJECTED",
        pose_valid=False,
        quality_gate_passed=False,
        covariance_calibration_status="UNVALIDATED",
        calibration_id="",
    )
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["observation_passed"] is True
    assert result["control_authorized"] is False
    assert result["safe_stop_required"] is True
    assert "POSE_QUALITY_REJECTED" in result["gate_diagnostics"]["failed_control_gates"]
    assert result["T_HP_selected"] is None
    assert result["target_plan"] is None


def test_missing_same_camera_public_spec_p95_evidence_safely_stops():
    kwargs = _accepted_kwargs()
    kwargs["yaw_acceptance"] = None
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["observation_passed"] is True
    assert result["control_authorized"] is False
    assert result["rejection_code"] == "PUBLIC_SPEC_YAW_P95_GATE_MISSING"
    assert result["T_WH_target"] is None


def test_missing_same_camera_accuracy_gate_safely_stops_even_with_p95_pass():
    kwargs = _accepted_kwargs()
    kwargs["accuracy_gate"] = None
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["observation_passed"] is True
    assert result["control_authorized"] is False
    assert result["rejection_code"] == "SAME_CAMERA_ACCURACY_GATE_MISSING"
    assert result["target_plan"] is None


def test_accuracy_gate_must_reference_the_exact_p95_dataset():
    kwargs = _accepted_kwargs()
    kwargs["accuracy_gate"]["yaw_dataset_tag"] = "different_camera_dataset"
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["observation_passed"] is True
    assert result["control_authorized"] is False
    assert result["rejection_code"] == "SAME_CAMERA_ACCURACY_GATE_P95_DATASET_MISMATCH"


def test_failed_or_wrong_profile_p95_evidence_safely_stops():
    kwargs = _accepted_kwargs()
    kwargs["yaw_acceptance"] = dict(kwargs["yaw_acceptance"])
    kwargs["yaw_acceptance"]["profile_name"] = "nominal"
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["observation_passed"] is True
    assert result["control_authorized"] is False
    assert result["rejection_code"] == "PUBLIC_SPEC_YAW_P95_PROFILE_MISMATCH"


def test_uncalibrated_image_only_mask_cannot_authorize_control():
    kwargs = _accepted_kwargs()
    kwargs["image_mask_quality"]["confidence_calibrated"] = False
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["observation_passed"] is True
    assert result["control_authorized"] is False
    assert result["rejection_code"] == "IMAGE_MASK_CONFIDENCE_UNCALIBRATED"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("occluded", "KEY_REGION_OCCLUDED"),
        ("missing_depth", "KEY_REGION_DEPTH_MISSING"),
        ("out_of_frame", "CONNECTOR_FACE_OUT_OF_FRAME"),
    ),
)
def test_bad_palm_observation_rejects_before_any_control_plan(
    mutation, expected_code
):
    kwargs = _accepted_kwargs()
    if mutation == "occluded":
        kwargs["occlusion_mask"] = kwargs["connector_face_mask"].copy()
    elif mutation == "missing_depth":
        rows, columns = np.nonzero(kwargs["connector_face_mask"])
        missing = math.ceil(rows.size * 0.03)
        kwargs["depth_m"][rows[:missing], columns[:missing]] = np.nan
    elif mutation == "out_of_frame":
        kwargs["face_center_uv"] = (-1.0, 20.0)
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["control_authorized"] is False
    assert result["rejection_code"] == expected_code
    assert result["target_plan"] is None


def test_robot_motion_during_capture_safely_stops_before_image_selection():
    kwargs = _accepted_kwargs()
    kwargs["actual_arm_q_after_capture"][3] = (
        runtime.MAXIMUM_CAPTURE_Q_DRIFT_RAD + 1.0e-4
    )
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["rejection_code"] == "CAPTURE_Q_DRIFT_ABOVE_LIMIT"
    assert result["control_authorized"] is False
    assert result["keyed_yaw_observation"] is None


def test_unknown_occlusion_has_a_precise_safe_stop_code():
    kwargs = _accepted_kwargs()
    kwargs["occlusion_mask"] = None
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["rejection_code"] == "KEY_REGION_OCCLUSION_UNKNOWN"
    assert result["observation_passed"] is False
    assert result["plan_authorized"] is False
    assert result["control_authorized"] is False


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_code"),
    (
        ("scene_schema_version", "legacy_scene", "KEYED_SCENE_SCHEMA_MISMATCH"),
        ("scene_profile_id", "legacy_profile", "KEYED_SCENE_PROFILE_MISMATCH"),
        ("keyed_model_id", "d38999_shell25j_proxy_v1", "KEYED_MODEL_ID_MISMATCH"),
    ),
)
def test_wrong_scene_or_model_identity_is_rejected(field, bad_value, expected_code):
    kwargs = _accepted_kwargs()
    kwargs[field] = bad_value
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["rejection_code"] == expected_code
    assert result["control_authorized"] is False


def test_camera_contract_is_exact_1280x720_rgbd_with_0p02_near_clip():
    kwargs = _accepted_kwargs()
    kwargs["camera_contract"]["near_clip_m"] = 0.10
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["rejection_code"] == "CAMERA_NEAR_CLIP_NOT_0P02_M"
    assert result["control_authorized"] is False


def test_fixed_endpoint_rotation_must_match_scene_configuration():
    kwargs = _accepted_kwargs()
    kwargs["T_WR_fixed_configured"] = np.eye(4)
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["rejection_code"] == "FIXED_CONFIG_ROTATION_MISMATCH"
    assert result["control_authorized"] is False


def test_runtime_api_has_no_episode_truth_or_old_control_pose_channels():
    parameters = set(
        inspect.signature(
            runtime.evaluate_postgrasp_palm_keyed_visual_control
        ).parameters
    )
    assert {
        "rgb",
        "depth_m",
        "connector_face_mask",
        "actual_arm_q_before_capture",
        "actual_arm_q_after_capture",
        "T_HC_frozen_configured",
        "T_WH_from_actual_q",
        "T_WR_fixed_configured",
        "T_RP_target_configured",
    } <= parameters
    assert not parameters & {
        "truth",
        "object_pose",
        "body_pose",
        "nut_pose",
        "contact",
        "collider",
        "semantic",
        "selected_t_hp_control_pose",
    }


def test_c2_records_accept_only_strict_yaw_zero_and_pi_candidates():
    kwargs = _accepted_kwargs()
    kwargs["c2_candidates"][1]["hypothesis_id"] = "OTHER"
    result = runtime.evaluate_postgrasp_palm_keyed_visual_control(**kwargs)
    assert result["rejection_code"] == "C2_OR_IMAGE_CONTRACT_INVALID"
    assert result["control_authorized"] is False
