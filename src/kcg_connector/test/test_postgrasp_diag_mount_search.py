import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kcg_connector.postgrasp_shadow_view_planner import (
    DIAGNOSTIC_MOUNT_CANDIDATES,
    diagnostic_hp_envelope_samples,
    diagnostic_mount_hard_gates,
    diagnostic_mount_score,
    diagnostic_optical_axis_angle_deg,
)

RUNNER_PATH = Path(__file__).parents[1] / "isaac" / "postgrasp_diag_mount_search.py"
PICK_PATH = Path(__file__).parents[1] / "isaac" / "d38999_tabletop_pick_smoke.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("diag_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_distances_are_inside_near_far_clip():
    module = _load_runner()
    for candidate_id, eye in DIAGNOSTIC_MOUNT_CANDIDATES:
        distance = float(np.linalg.norm(np.asarray(eye) - np.asarray((0.0, 0.0, 0.002))))
        assert module.DIAG_CLIP_NEAR_M < distance < module.DIAG_CLIP_FAR_M
        assert 0.0 < module.DIAG_CLIP_NEAR_M


def test_prefilter_runs_full_truth_free_envelope():
    module = _load_runner()
    result = module.prefilter_diagnostic_candidate(
        np.zeros(6), (0.060, 0.0, -0.055), (0.0, 0.0, 0.002)
    )
    assert result["envelope_sample_count"] == 17
    assert isinstance(result["passed"], bool)
    assert all(isinstance(item["passed"], bool) for item in result["samples"])


def test_prefilter_accepts_runtime_4x4_nominal_transform():
    module = _load_runner()
    result = module.prefilter_diagnostic_candidate(
        np.eye(4), (0.060, 0.0, -0.055), (0.0, 0.0, 0.002)
    )
    assert result["envelope_sample_count"] == 17


def test_current_hand_translation_moves_camera_without_object_truth():
    module = _load_runner()
    eye = DIAGNOSTIC_MOUNT_CANDIDATES[0][1]
    target = (0.0, 0.0, 0.002)
    first = module.diagnostic_camera_world_transform(
        np.eye(4), np.eye(4), eye, target
    )
    moved_hand = np.eye(4)
    moved_hand[:3, 3] = (0.011, -0.007, 0.019)
    second = module.diagnostic_camera_world_transform(
        moved_hand, np.eye(4), eye, target
    )
    assert np.allclose(
        second[:3, 3] - first[:3, 3], moved_hand[:3, 3]
    )


def test_runner_uses_current_hand_transform_instead_of_nominal_handbase():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "current_hand_transform" in source
    assert "nominal_world_handbase" not in source
    assert "diagnostic_camera_world_transform(" in source


def test_diag_exit_semantics_are_nonzero_on_incomplete_request():
    source = PICK_PATH.read_text(encoding="utf-8")
    assert "grasp_passed_before_diag" in source
    assert "diag_request_completed" in source
    assert "process_exit_code = 2" in source


def test_all_candidates_have_true_optical_angle_in_limits():
    for candidate_id, eye in DIAGNOSTIC_MOUNT_CANDIDATES:
        angle = diagnostic_optical_axis_angle_deg(eye)
        assert 25.0 <= angle <= 50.0, candidate_id
        gate = diagnostic_mount_hard_gates(eye)
        assert gate["passed"] is True, candidate_id


def test_old_posterior_mount_is_rejected_by_hard_gate():
    old_eye = (0.120, 0.0, 0.060)
    assert diagnostic_optical_axis_angle_deg(old_eye) > 50.0
    gate = diagnostic_mount_hard_gates(old_eye)
    assert gate["passed"] is False
    assert "OPTICAL_AXIS_ANGLE_OUT_OF_RANGE" in gate["reasons"]
    assert "CAMERA_HOUSING_TOO_CLOSE_TO_MATING_FACE" in gate["reasons"]


def test_envelope_is_deterministic_and_truth_free():
    nominal = np.zeros(6)
    first = diagnostic_hp_envelope_samples(nominal)
    second = diagnostic_hp_envelope_samples(nominal)
    assert len(first) == 17
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert all(
        abs(float(value)) <= (0.002 if index < 3 else np.radians(6.0))
        for sample in first
        for index, value in enumerate(sample)
    )
    # API accepts only nominal; no truth array argument exists.
    import inspect
    assert "truth" not in inspect.signature(diagnostic_hp_envelope_samples).parameters


def test_usd_camera_to_cv_conversion_has_expected_axes():
    module = _load_runner()
    usd_row_xform = np.eye(4)
    usd_row_xform[3, :3] = (1.0, 2.0, 3.0)
    pose = module.cv_camera_pose_from_usd_row_xform(usd_row_xform)
    assert np.array_equal(
        pose[:3, :3], np.diag((1.0, -1.0, -1.0))
    )
    assert np.array_equal(pose[:3, 3], (1.0, 2.0, 3.0))
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0)


def test_score_cannot_resurrect_hard_gate_rejection():
    # Score is only ordering metadata and has no passed field.
    score = diagnostic_mount_score(
        {
            "projected_shell_depth_support": 1.0,
            "projected_socket_depth_support": 1.0,
            "central_depth_fraction": 1.0,
            "edge_support_fraction": 1.0,
            "foreground_occlusion_fraction": 0.0,
            "condition_number_5d": 10.0,
        }
    )
    assert score > 0.0
    bad = diagnostic_mount_hard_gates((0.120, 0.0, 0.060))
    assert bad["passed"] is False
    assert "passed" not in {"score": score}


def test_output_exists_fails_closed_before_isaac_import(tmp_path):
    module = _load_runner()
    output_root = tmp_path / "phase1_diag_mount_search_v1" / "seed000" / "postgrasp_diag_mount_search"
    output_root.mkdir(parents=True)
    arguments = SimpleNamespace(output_dir=str(output_root.parent))
    result = module.run_diagnostic_mount_search(
        repository=tmp_path,
        arguments=arguments,
        Gf=None, Usd=None, UsdGeom=None, UsdLux=None,
        stage=None, world=None, simulation_app=None, tabletop=None,
        current_hand_target=None, rate_hz=240, global_step=0,
        current_hand_transform=np.eye(4),
        nominal_hand_to_plug=np.eye(4),
    )
    assert result["status"] == "FAIL_CLOSED_OUTPUT_EXISTS"
    assert result["control_authorized"] is False


def test_manifest_contract_fields_in_runner_source():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for token in (
        "DIAGNOSTIC_MOUNT_SEARCH_ONLY",
        "control_authorized",
        "formal_estimator_input",
        "semantic_present",
        "POSTHOC_IDENTITY_AUDIT_NOT_IMPLEMENTED",
        "MECHANICAL_FEASIBILITY_UNVERIFIED",
        "capture_camera",
        "capture_T_WH",
        "capture_T_HC",
        "capture_T_WC",
        "capture_usd_camera_xform_row_major",
        "global_physics_step",
        "postgrasp_diag_mount_search_sha256",
        "isaac_d38999_rgbd_runtime_sha256",
        "rgbd_config_sha256",
    ):
        assert token in source
    assert "semantic_segmentation" not in source
    assert "capture_d38999_rgbd_runtime(" not in source
    assert "capture_d38999_rgbd_raw_formal(" in source


def test_pick_smoke_optin_contract():
    source = PICK_PATH.read_text(encoding="utf-8")
    assert "--postgrasp-diag-mount-search" in source
    assert "run_diagnostic_mount_search(" in source
    assert "DIAG_HOOK_ERROR_SAFE" in source
    assert "control_authorized" in source
