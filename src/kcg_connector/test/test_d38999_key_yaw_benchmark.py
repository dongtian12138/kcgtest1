import inspect
import json
import math
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

from kcg_connector.d38999_key_yaw_benchmark import (
    CLAIMED_REVEAL_SCHEMA_VERSION,
    FORMAL_WITHHELD_BLOCKED_STATUS,
    LOCAL_CONSISTENCY_CHECK,
    PROJECT_STRESS_INTERPRETATION,
    evaluate_local_key_yaw_benchmark_metrics,
    refine_keyed_axial_yaw_from_rgbd,
    write_local_key_yaw_prediction_artifact,
)
from kcg_connector.d38999_key_region_detector import detect_key_region_from_palm_rgbd
from kcg_connector.d38999_keyed_public_spec_v2 import PLUG_MODEL_ID


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


def _tilted_face_observation(yaw_deg=35.0):
    height, width = 300, 400
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
    face_center_uv = _project(translation, intrinsics)
    actual_rotation = tilt @ _rz(math.radians(yaw_deg))
    key_point = translation + actual_rotation @ np.asarray((0.050, 0.0, 0.0))
    projected_key = _project(key_point, intrinsics)
    master_angle = math.atan2(
        projected_key[1] - face_center_uv[1],
        projected_key[0] - face_center_uv[0],
    )
    du = columns - face_center_uv[0]
    dv = rows - face_center_uv[1]
    pixel_radius = np.hypot(du, dv)
    pixel_angle = np.mod(np.arctan2(dv, du), 2.0 * math.pi)
    face_mask = pixel_radius <= 48.0
    for offset_deg, width_deg in zip(
        (0.0, 80.0, 142.0, 196.0, 293.0),
        (16.0, 8.0, 8.0, 8.0, 8.0),
    ):
        angular_difference = np.abs(
            np.angle(
                np.exp(
                    1j
                    * (
                        pixel_angle
                        - master_angle
                        - math.radians(offset_deg)
                    )
                )
            )
        )
        face_mask |= (angular_difference <= math.radians(width_deg) / 2.0) & (
            pixel_radius <= 58.0
        )
    depth_m = np.where(face_mask, scale, np.nan)

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
        "depth_m": depth_m,
        "connector_face_mask": face_mask,
        "face_center_uv": face_center_uv,
        "camera_intrinsics": intrinsics,
        "c2_candidates": candidates,
        "keyed_model_id": PLUG_MODEL_ID,
        "occlusion_mask": np.zeros_like(face_mask),
    }


@pytest.mark.parametrize(
    ("yaw_deg", "expected_branch"),
    ((35.0, "YAW_0"), (140.0, "YAW_PI"), (-40.0, "YAW_0"), (-135.0, "YAW_PI")),
)
def test_tilted_depth_plane_selects_and_refines_continuous_yaw(
    yaw_deg, expected_branch
):
    result = refine_keyed_axial_yaw_from_rgbd(**_tilted_face_observation(yaw_deg))

    assert result["passed"] is True
    assert result["selected_hypothesis_id"] == expected_branch
    assert math.degrees(result["estimated_axial_yaw_rad"]) == pytest.approx(
        yaw_deg, abs=0.15
    )
    assert result["quality_diagnostics"]["plane_rmse_m"] < 1.0e-9
    assert result["shadow_only"] is True
    assert result["shadow_authorized"] is False
    assert result["control_authorized"] is False
    assert result["selected_for_control_allowed"] is False


def test_exact_five_key_detector_evidence_is_used_for_centroid():
    observation = _tilted_face_observation(28.0)

    result = refine_keyed_axial_yaw_from_rgbd(**observation)

    assert result["passed"] is True
    assert math.degrees(result["estimated_axial_yaw_rad"]) == pytest.approx(
        28.0, abs=0.15
    )
    assert (
        result["quality_diagnostics"]["key_observation_source"]
        == "EXACT_FIVE_KEY_N_PATTERN_DETECTOR_PROBABILITY_CENTROID"
    )
    assert result["quality_diagnostics"]["key_detector_candidate_count"] == 5
    assert result["quality_diagnostics"]["local_key_depth_complete"] is True


@pytest.mark.parametrize(
    ("mutation", "status", "code"),
    (
        ("missing_occlusion", "OCCLUSION_UNKNOWN", "KEY_REGION_OCCLUSION_UNKNOWN"),
        ("occluded", "OCCLUDED", "KEY_REGION_OCCLUDED"),
        ("missing_depth", "DEPTH_MISSING", "KEY_REGION_DEPTH_MISSING"),
        ("out_of_frame", "OUT_OF_FRAME", "CONNECTOR_FACE_OUT_OF_FRAME"),
        ("old_model", "MODEL_NOT_KEYED_V2", "KEYED_MODEL_ID_UNAVAILABLE"),
        ("ambiguous_branch", "AMBIGUOUS", "KEY_BRANCH_AMBIGUOUS"),
    ),
)
def test_inference_fail_closed_conditions(mutation, status, code):
    observation = _tilted_face_observation(35.0)
    if mutation == "missing_occlusion":
        observation["occlusion_mask"] = None
    elif mutation == "occluded":
        observation["occlusion_mask"] = observation["connector_face_mask"].copy()
    elif mutation == "missing_depth":
        face_rows, face_columns = np.nonzero(observation["connector_face_mask"])
        missing_count = math.ceil(face_rows.size * 0.03)
        observation["depth_m"][
            face_rows[:missing_count], face_columns[:missing_count]
        ] = np.nan
    elif mutation == "out_of_frame":
        observation["face_center_uv"] = np.asarray((-1.0, 20.0))
    elif mutation == "old_model":
        observation["keyed_model_id"] = "d38999_shell25j_proxy_v1"
    elif mutation == "ambiguous_branch":
        observation = _tilted_face_observation(90.0)

    result = refine_keyed_axial_yaw_from_rgbd(**observation)

    assert result["passed"] is False
    assert result["status"] == status
    assert isinstance(result["status"], str)
    assert result["rejection_code"] == code
    assert result["estimated_axial_yaw_rad"] is None
    assert result["shadow_authorized"] is False
    assert result["control_authorized"] is False


def test_inference_contract_has_no_truth_or_simulator_truth_channels():
    parameters = set(inspect.signature(refine_keyed_axial_yaw_from_rgbd).parameters)
    assert {"depth_m", "connector_face_mask", "face_center_uv"} <= parameters
    assert {"camera_intrinsics", "c2_candidates", "keyed_model_id"} <= parameters
    assert {"occlusion_mask"} <= parameters
    assert not parameters & {"key_centroid_uv", "key_direction_uv"}
    assert not parameters & {
        "truth",
        "truth_yaw",
        "semantic",
        "object_pose",
        "contact",
        "collider",
    }


def test_pure_circle_and_injected_direction_cannot_bypass_five_key_detector():
    observation = _tilted_face_observation(20.0)
    rows, columns = np.indices(observation["depth_m"].shape)
    center = observation["face_center_uv"]
    circle = np.hypot(columns - center[0], rows - center[1]) <= 48.0
    observation["connector_face_mask"] = circle
    observation["depth_m"] = np.where(circle, observation["depth_m"], np.nan)

    result = refine_keyed_axial_yaw_from_rgbd(**observation)

    assert result["passed"] is False
    assert result["status"] == "LOW_CONFIDENCE"
    assert result["rejection_code"] == "KEY_REGION_LOW_CONFIDENCE"
    with pytest.raises(TypeError):
        refine_keyed_axial_yaw_from_rgbd(
            **observation,
            key_direction_uv=(1.0, 0.0),
        )


def test_one_missing_master_key_depth_pixel_is_rejected():
    observation = _tilted_face_observation(35.0)
    detector = detect_key_region_from_palm_rgbd(
        observation["connector_face_mask"],
        observation["depth_m"],
        observation["face_center_uv"],
        observation["keyed_model_id"],
        occlusion_mask=observation["occlusion_mask"],
    )
    key_row, key_column = np.argwhere(detector["key_probability"] > 0.0)[0]
    observation["depth_m"][key_row, key_column] = np.nan

    result = refine_keyed_axial_yaw_from_rgbd(**observation)

    assert result["passed"] is False
    assert result["status"] == "DEPTH_MISSING"
    assert result["rejection_code"] == "KEY_REGION_DEPTH_MISSING"


def test_c2_candidates_require_one_of_each_exact_hypothesis_id():
    observation = _tilted_face_observation(35.0)
    observation["c2_candidates"][1]["hypothesis_id"] = "YAW_0"

    with pytest.raises(ValueError, match="one YAW_0 and one YAW_PI"):
        refine_keyed_axial_yaw_from_rgbd(**observation)


def test_c2_transform_x_axes_must_be_near_exactly_antipodal():
    observation = _tilted_face_observation(35.0)
    first_rotation = observation["c2_candidates"][0]["T_camera_plug"][:3, :3]
    observation["c2_candidates"][1]["T_camera_plug"][:3, :3] = (
        first_rotation @ _rz(math.radians(175.0))
    )

    with pytest.raises(ValueError, match="must be antipodal"):
        refine_keyed_axial_yaw_from_rgbd(**observation)


def test_c2_transform_rotation_must_match_axial_yaw_difference():
    observation = _tilted_face_observation(35.0)
    first_rotation = observation["c2_candidates"][0]["T_camera_plug"][:3, :3]
    observation["c2_candidates"][1]["T_camera_plug"][:3, :3] = (
        first_rotation @ _rz(math.radians(179.0))
    )

    with pytest.raises(ValueError, match="disagrees with axial_yaw"):
        refine_keyed_axial_yaw_from_rgbd(**observation)


def test_c2_translation_must_match_observed_depth_face_center():
    observation = _tilted_face_observation(35.0)
    for candidate in observation["c2_candidates"]:
        candidate["T_camera_plug"][0, 3] += 0.005

    result = refine_keyed_axial_yaw_from_rgbd(**observation)

    assert result["passed"] is False
    assert result["reason"] == "C2_TRANSLATION_DISAGREES_WITH_OBSERVED_FACE_CENTER"
    assert result["control_authorized"] is False


DECLARED_YAW_VALUES = tuple(
    -math.pi + 2.0 * math.pi * index / 64 for index in range(64)
)
DECLARED_STRATA = {
    "yaw": [f"yaw-{index:02d}" for index in range(64)],
    "light": [f"light-{index}" for index in range(4)],
    "pose": [f"pose-{index}" for index in range(4)],
}


def _benchmark_records(count=2048, error_deg=0.01, include_rejections=True):
    predictions = []
    truth = []
    for index in range(count):
        yaw_index = index % 64
        block = index // 64
        yaw = DECLARED_YAW_VALUES[yaw_index]
        branch = "YAW_0" if abs(yaw) <= math.pi / 2.0 else "YAW_PI"
        sample_id = f"sample-{index:04d}"
        predictions.append(
            {
                "sample_id": sample_id,
                "passed": True,
                "estimated_axial_yaw_rad": yaw + math.radians(error_deg),
                "selected_hypothesis_id": branch,
                "shadow_only": True,
                "control_authorized": False,
            }
        )
        truth.append(
            {
                "sample_id": sample_id,
                "expected_outcome": "VISIBLE_VALID",
                "axial_yaw_truth_rad": yaw,
                "expected_hypothesis_id": branch,
                "strata": {
                    "yaw": f"yaw-{yaw_index:02d}",
                    "light": f"light-{block % 4}",
                    "pose": f"pose-{(block // 4) % 4}",
                },
            }
        )
    if include_rejections:
        for rejection_class in (
            "OCCLUDED",
            "OUT_OF_FRAME",
            "DEPTH_MISSING",
            "LOW_CONFIDENCE",
        ):
            for index in range(256):
                sample_id = f"reject-{rejection_class.lower()}-{index:03d}"
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "passed": False,
                        "estimated_axial_yaw_rad": None,
                        "selected_hypothesis_id": None,
                        "shadow_only": True,
                        "control_authorized": False,
                    }
                )
                truth.append(
                    {
                        "sample_id": sample_id,
                        "expected_outcome": "MUST_REJECT",
                        "axial_yaw_truth_rad": None,
                        "expected_hypothesis_id": None,
                        "rejection_class": rejection_class,
                        "strata": {},
                    }
                )
    return predictions, truth


def _local_context(predictions, truth, tmp_path):
    artifact = tmp_path / "predictions.jsonl"
    manifest_path = tmp_path / "predictions.manifest.json"
    manifest = write_local_key_yaw_prediction_artifact(
        predictions,
        prediction_artifact_path=artifact,
        prediction_manifest_path=manifest_path,
        dataset_tag="cpu-held-out-yaw-light-pose-v1",
        run_id="cpu-yaw-run-v1",
        declared_yaw_values_rad=DECLARED_YAW_VALUES,
        declared_strata=DECLARED_STRATA,
    )
    completed = datetime.fromisoformat(
        manifest["prediction_completed_at_utc"].replace("Z", "+00:00")
    )
    reveal = {
        "schema_version": CLAIMED_REVEAL_SCHEMA_VERSION,
        "status": "CALLER_CLAIMS_TRUTH_AVAILABLE_FOR_EVALUATION",
        "run_id": manifest["run_id"],
        "dataset_tag": manifest["dataset_tag"],
        "truth_reveal_id": "cpu-yaw-run-v1:truth-reveal",
        "truth_record_count": len(truth),
        "truth_revealed_at_utc": (
            completed + timedelta(seconds=1)
        ).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "prediction_manifest_path": str(manifest_path.resolve()),
    }
    return manifest_path, reveal


def _evaluate(predictions, truth, tmp_path, **overrides):
    manifest_path, reveal = _local_context(predictions, truth, tmp_path)
    arguments = {
        "keyed_model_id": PLUG_MODEL_ID,
        "dataset_tag": "cpu-held-out-yaw-light-pose-v1",
        "claimed_reveal_metadata": reveal,
    }
    arguments.update(overrides)
    return evaluate_local_key_yaw_benchmark_metrics(
        manifest_path,
        truth,
        **arguments,
    )


def test_64_yaw_2048_sample_withheld_gate_passes_shadow_only(tmp_path):
    predictions, truth = _benchmark_records()

    report = _evaluate(predictions, truth, tmp_path)

    assert report["passed"] is False
    assert report["metric_gates_passed"] is True
    assert report["formal_withheld_evidence_status"] == (
        FORMAL_WITHHELD_BLOCKED_STATUS
    )
    assert report["formal_withheld_evidence_verified"] is False
    assert report["p95_gate"]["sample_count"] == 2048
    assert report["p95_gate"]["profile_name"] == "adversarial_gdt_stress"
    assert report["p95_gate"]["withheld_truth"] is False
    assert report["p95_gate"]["formal_withheld_evidence_verified"] is False
    assert report["p95_gate"]["formal_withheld_evidence_status"] == (
        FORMAL_WITHHELD_BLOCKED_STATUS
    )
    assert report["p95_gate"]["required_yaw_error_p95_deg"] == pytest.approx(
        0.030275467425980793
    )
    assert report["threshold_interpretation"] == PROJECT_STRESS_INTERPRETATION
    assert report["drawing_specified_mechanical_yaw_clearance"] is False
    assert report["c2_misselection_count"] == 0
    assert report["actual_distinct_yaw_value_count"] == 64
    assert report["coverage_self_consistency_passed"] is True
    assert report["withheld_protocol"] == LOCAL_CONSISTENCY_CHECK
    assert report["two_stage_withheld_protocol_verified"] is False
    assert report["caller_claimed_reveal_trusted"] is False
    assert report["must_reject_sample_count"] == 1024
    assert report["must_reject_count_gate_passed"] is True
    assert report["rejection_class_quotas_passed"] is True
    assert {item["sample_count"] for item in report["rejection_class_reports"]} == {
        256
    }
    assert report["bootstrap_p95_upper_bound_gate_passed"] is True
    assert report["bootstrap_p95_one_sided_95_upper_bound_deg"] == pytest.approx(
        0.01
    )
    assert report["all_strata_passed"] is True
    assert len(report["stratum_reports"]) == 72
    assert report["shadow_only"] is True
    assert report["shadow_authorized"] is False
    assert report["control_authorized"] is False
    assert report["p95_gate"]["shadow_authorized"] is False
    assert report["p95_gate"]["control_authorized"] is False
    assert report["simulation_insertion_control_authorized"] is False
    assert report["robot_control_authorized"] is False
    assert report["hardware_control_authorized"] is False


def test_rejected_visible_samples_receive_180_degree_penalty(tmp_path):
    predictions, truth = _benchmark_records()
    for prediction in predictions[:105]:
        prediction.update(
            passed=False,
            estimated_axial_yaw_rad=None,
            selected_hypothesis_id=None,
        )

    report = _evaluate(predictions, truth, tmp_path)

    assert report["passed"] is False
    assert report["visible_valid_yield"] == pytest.approx(1943 / 2048)
    assert report["visible_yield_gate_passed"] is False
    assert report["observed_penalized_yaw_error_p95_deg"] == pytest.approx(180.0)
    assert report["p95_gate"]["passed"] is False
    assert report["control_authorized"] is False


def test_one_c2_misselection_fails_even_when_yaw_p95_passes(tmp_path):
    predictions, truth = _benchmark_records()
    predictions[0]["selected_hypothesis_id"] = "YAW_0"
    assert truth[0]["expected_hypothesis_id"] == "YAW_PI"

    report = _evaluate(predictions, truth, tmp_path)

    assert report["p95_gate"]["passed"] is True
    assert report["c2_misselection_count"] == 1
    assert report["c2_misselection_gate_passed"] is False
    assert report["passed"] is False


def test_required_rejection_false_accept_fails_closed(tmp_path):
    predictions, truth = _benchmark_records()
    predictions[2048].update(
        passed=True,
        estimated_axial_yaw_rad=0.0,
        selected_hypothesis_id="YAW_0",
    )

    report = _evaluate(predictions, truth, tmp_path)

    assert report["must_reject_sample_count"] == 1024
    assert report["must_reject_false_accept_count"] == 1
    assert report["must_reject_gate_passed"] is False
    assert report["passed"] is False


def test_stratum_p95_can_fail_when_global_p95_still_passes(tmp_path):
    predictions, truth = _benchmark_records()
    yaw_zero_indices = list(range(0, 2048, 64))
    for index in yaw_zero_indices[:10]:
        predictions[index]["estimated_axial_yaw_rad"] = (
            truth[index]["axial_yaw_truth_rad"] + math.radians(0.04)
        )

    report = _evaluate(predictions, truth, tmp_path)

    assert report["p95_gate"]["passed"] is True
    failed = [item for item in report["stratum_reports"] if not item["passed"]]
    assert [(item["axis"], item["label"]) for item in failed] == [
        ("yaw", "yaw-00")
    ]
    assert failed[0]["observed_yaw_error_p95_deg"] == pytest.approx(0.04)
    assert report["all_strata_passed"] is False
    assert report["passed"] is False


def test_zero_must_reject_samples_cannot_pass(tmp_path):
    predictions, truth = _benchmark_records(include_rejections=False)

    report = _evaluate(predictions, truth, tmp_path)

    assert report["p95_gate"]["passed"] is True
    assert report["must_reject_sample_count"] == 0
    assert report["must_reject_count_gate_passed"] is False
    assert report["rejection_class_quotas_passed"] is False
    assert report["must_reject_gate_passed"] is False
    assert report["passed"] is False


def test_each_required_rejection_class_needs_256_samples(tmp_path):
    predictions, truth = _benchmark_records()
    first_occluded = next(
        item for item in truth if item.get("rejection_class") == "OCCLUDED"
    )
    first_occluded["rejection_class"] = "OUT_OF_FRAME"

    report = _evaluate(predictions, truth, tmp_path)

    assert report["must_reject_count_gate_passed"] is True
    counts = {
        item["rejection_class"]: item["sample_count"]
        for item in report["rejection_class_reports"]
    }
    assert counts["OCCLUDED"] == 255
    assert counts["OUT_OF_FRAME"] == 257
    assert report["rejection_class_quotas_passed"] is False
    assert report["passed"] is False


def test_bootstrap_upper_bound_must_pass_in_addition_to_observed_p95(tmp_path):
    predictions, truth = _benchmark_records()
    for index in np.linspace(0, 2047, 92, dtype=int):
        predictions[index]["estimated_axial_yaw_rad"] = (
            truth[index]["axial_yaw_truth_rad"] + math.radians(0.04)
        )

    report = _evaluate(predictions, truth, tmp_path)

    assert report["p95_gate"]["passed"] is True
    assert report["p95_gate"]["observed_yaw_error_p95_deg"] == pytest.approx(0.01)
    assert report["bootstrap_p95_one_sided_95_upper_bound_deg"] > 0.03
    assert report["bootstrap_p95_upper_bound_gate_passed"] is False
    assert report["passed"] is False


@pytest.mark.parametrize(
    ("extra_field", "value"),
    (
        ("truth_yaw_rad", 0.0),
        ("robot_control_authorized", False),
        ("shadow_authorized", False),
    ),
)
def test_prediction_schema_rejects_truth_and_extra_authorization_fields(
    tmp_path, extra_field, value
):
    predictions, _ = _benchmark_records()
    predictions[0][extra_field] = value

    with pytest.raises(ValueError, match="exactly the local prediction fields"):
        write_local_key_yaw_prediction_artifact(
            predictions,
            prediction_artifact_path=tmp_path / "predictions.jsonl",
            prediction_manifest_path=tmp_path / "manifest.json",
            dataset_tag="cpu-held-out-yaw-light-pose-v1",
            run_id="schema-rejection-run",
            declared_yaw_values_rad=DECLARED_YAW_VALUES,
            declared_strata=DECLARED_STRATA,
        )


def test_local_prediction_paths_are_exclusive_and_manifest_has_no_hash(tmp_path):
    predictions, truth = _benchmark_records()
    manifest_path, _ = _local_context(predictions, truth, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert not any("hash" in key.lower() for key in manifest)
    with pytest.raises(FileExistsError, match="non-overwritable"):
        write_local_key_yaw_prediction_artifact(
            predictions,
            prediction_artifact_path=tmp_path / "predictions.jsonl",
            prediction_manifest_path=manifest_path,
            dataset_tag="cpu-held-out-yaw-light-pose-v1",
            run_id="cpu-yaw-run-v1",
            declared_yaw_values_rad=DECLARED_YAW_VALUES,
            declared_strata=DECLARED_STRATA,
        )


def test_accidental_prediction_artifact_change_is_detected(tmp_path):
    predictions, truth = _benchmark_records()
    manifest_path, reveal = _local_context(predictions, truth, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest["prediction_artifact_path"]
    with open(artifact, "a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="changed after local recording"):
        evaluate_local_key_yaw_benchmark_metrics(
            manifest_path,
            truth,
            keyed_model_id=PLUG_MODEL_ID,
            dataset_tag="cpu-held-out-yaw-light-pose-v1",
            claimed_reveal_metadata=reveal,
        )


@pytest.mark.parametrize("mutation", ("missing", "wrong_run", "too_early"))
def test_claimed_reveal_must_match_local_prediction_run(tmp_path, mutation):
    predictions, truth = _benchmark_records()
    manifest_path, reveal = _local_context(predictions, truth, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        reveal = None
    elif mutation == "wrong_run":
        reveal["run_id"] = "different-run"
    else:
        reveal["truth_revealed_at_utc"] = manifest[
            "prediction_completed_at_utc"
        ]

    with pytest.raises(ValueError):
        evaluate_local_key_yaw_benchmark_metrics(
            manifest_path,
            truth,
            keyed_model_id=PLUG_MODEL_ID,
            dataset_tag="cpu-held-out-yaw-light-pose-v1",
            claimed_reveal_metadata=reveal,
        )


@pytest.mark.parametrize(
    "weakened",
    (
        {"profile_name": "nominal_centered"},
        {"minimum_samples": 999},
        {"minimum_visible_yield": 0.98},
        {"required_stratum_axes": ("yaw",)},
        {"minimum_stratum_samples": 29},
        {"minimum_must_reject_samples": 1023},
        {"minimum_must_reject_per_class": 255},
    ),
)
def test_benchmark_contract_cannot_be_weakened(tmp_path, weakened):
    predictions, truth = _benchmark_records()
    manifest_path, reveal = _local_context(predictions, truth, tmp_path)

    with pytest.raises(ValueError, match="cannot weaken|must remain exactly"):
        evaluate_local_key_yaw_benchmark_metrics(
            manifest_path,
            truth,
            keyed_model_id=PLUG_MODEL_ID,
            dataset_tag="cpu-held-out-yaw-light-pose-v1",
            claimed_reveal_metadata=reveal,
            **weakened,
        )


@pytest.mark.parametrize("coverage_failure", ("63_yaws", "one_light_label"))
def test_actual_truth_must_match_predeclared_yaw_light_pose_coverage(
    tmp_path, coverage_failure
):
    predictions, truth = _benchmark_records()
    for record in truth[:2048]:
        if coverage_failure == "63_yaws" and record["strata"]["yaw"] == "yaw-63":
            record["axial_yaw_truth_rad"] = DECLARED_YAW_VALUES[0]
        elif coverage_failure == "one_light_label":
            record["strata"]["light"] = "light-0"
    manifest_path, reveal = _local_context(predictions, truth, tmp_path)

    with pytest.raises(ValueError, match="64 declared yaw|declared coverage"):
        evaluate_local_key_yaw_benchmark_metrics(
            manifest_path,
            truth,
            keyed_model_id=PLUG_MODEL_ID,
            dataset_tag="cpu-held-out-yaw-light-pose-v1",
            claimed_reveal_metadata=reveal,
        )


def test_api_has_no_self_reported_withheld_boolean():
    parameters = inspect.signature(evaluate_local_key_yaw_benchmark_metrics).parameters
    assert "withheld_truth" not in parameters
    assert "predictions" not in parameters
    assert "prediction_manifest_path" in parameters
    assert "claimed_reveal_metadata" in parameters


def _change_one_insignificant_digit(artifact_path):
    original = artifact_path.read_text(encoding="utf-8")
    marker = '"estimated_axial_yaw_rad": '
    search_from = 0
    while True:
        number_start = original.index(marker, search_from) + len(marker)
        number_end = original.index(",", number_start)
        if any(character.isdigit() for character in original[number_start:number_end]):
            break
        search_from = number_end
    digit_index = next(
        index
        for index in range(number_end - 1, number_start, -1)
        if original[index].isdigit()
    )
    replacement = "1" if original[digit_index] != "1" else "2"
    changed = original[:digit_index] + replacement + original[digit_index + 1 :]
    assert len(changed.encode("utf-8")) == len(original.encode("utf-8"))
    artifact_path.write_text(changed, encoding="utf-8")


def _assert_formal_withheld_remains_blocked(report):
    assert report["metric_gates_passed"] is True
    assert report["passed"] is False
    assert report["two_stage_withheld_protocol_verified"] is False
    assert report["formal_withheld_evidence_verified"] is False
    assert report["formal_withheld_evidence_status"] == (
        "BLOCKED_REQUIRES_OS_ISOLATED_EVALUATOR"
    )


def test_same_account_can_rewrite_same_size_and_restore_mtime_but_not_pass(tmp_path):
    predictions, truth = _benchmark_records()
    manifest_path, claimed_reveal = _local_context(predictions, truth, tmp_path)
    artifact = tmp_path / "predictions.jsonl"
    original_stat = artifact.stat()
    _change_one_insignificant_digit(artifact)
    os.utime(
        artifact,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    report = evaluate_local_key_yaw_benchmark_metrics(
        manifest_path,
        truth,
        keyed_model_id=PLUG_MODEL_ID,
        dataset_tag="cpu-held-out-yaw-light-pose-v1",
        claimed_reveal_metadata=claimed_reveal,
    )

    _assert_formal_withheld_remains_blocked(report)


def test_same_account_can_update_artifact_and_manifest_but_not_pass(tmp_path):
    predictions, truth = _benchmark_records()
    manifest_path, claimed_reveal = _local_context(predictions, truth, tmp_path)
    artifact = tmp_path / "predictions.jsonl"
    _change_one_insignificant_digit(artifact)
    artifact_stat = artifact.stat()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_size_bytes"] = artifact_stat.st_size
    manifest["artifact_mtime_ns"] = artifact_stat.st_mtime_ns
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_local_key_yaw_benchmark_metrics(
        manifest_path,
        truth,
        keyed_model_id=PLUG_MODEL_ID,
        dataset_tag="cpu-held-out-yaw-light-pose-v1",
        claimed_reveal_metadata=claimed_reveal,
    )

    _assert_formal_withheld_remains_blocked(report)


def test_same_account_can_fabricate_claimed_reveal_but_not_pass(tmp_path):
    predictions, truth = _benchmark_records()
    manifest_path, claimed_reveal = _local_context(predictions, truth, tmp_path)
    claimed_reveal["truth_reveal_id"] = "caller-fabricated-reveal"
    claimed_reveal["truth_revealed_at_utc"] = "2099-01-01T00:00:00.000000Z"

    report = evaluate_local_key_yaw_benchmark_metrics(
        manifest_path,
        truth,
        keyed_model_id=PLUG_MODEL_ID,
        dataset_tag="cpu-held-out-yaw-light-pose-v1",
        claimed_reveal_metadata=claimed_reveal,
    )

    assert report["caller_claimed_reveal_trusted"] is False
    _assert_formal_withheld_remains_blocked(report)


def test_same_account_can_recreate_bundle_paths_but_not_pass(tmp_path):
    predictions, truth = _benchmark_records()
    manifest_path, claimed_reveal = _local_context(predictions, truth, tmp_path)
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_artifact = tmp_path / "predictions.jsonl"
    copied_artifact = tmp_path / "copied_predictions.jsonl"
    copied_manifest_path = tmp_path / "copied_predictions.manifest.json"
    copied_artifact.write_bytes(original_artifact.read_bytes())
    copied_stat = copied_artifact.stat()
    copied_manifest = dict(original_manifest)
    copied_manifest.update(
        prediction_artifact_path=str(copied_artifact.resolve()),
        prediction_manifest_path=str(copied_manifest_path.resolve()),
        artifact_size_bytes=copied_stat.st_size,
        artifact_mtime_ns=copied_stat.st_mtime_ns,
    )
    copied_manifest_path.write_text(json.dumps(copied_manifest), encoding="utf-8")
    claimed_reveal["prediction_manifest_path"] = str(
        copied_manifest_path.resolve()
    )

    report = evaluate_local_key_yaw_benchmark_metrics(
        copied_manifest_path,
        truth,
        keyed_model_id=PLUG_MODEL_ID,
        dataset_tag="cpu-held-out-yaw-light-pose-v1",
        claimed_reveal_metadata=claimed_reveal,
    )

    _assert_formal_withheld_remains_blocked(report)
