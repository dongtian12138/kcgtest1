import json
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.d38999_inhand_multiview import matrix_pose
from kcg_connector.posthoc_shadow_evaluation import evaluate_posthoc_shadow


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _truth_report():
    # Self-contained synthetic truth: nominal-like frame with known error.
    return {
        "passed": True,
        "posthoc_t_hand_plug_actual": np.eye(4).tolist(),
        "posthoc_t_hand_plug_nominal": np.eye(4).tolist(),
    }


def _shadow_result():
    pose = np.eye(4)
    pose[:3, 3] = [0.001, 0.0, 0.0]
    pose[:3, :3] = np.array(
        [[0.9986, 0.0, 0.0523], [0.0, 1.0, 0.0], [-0.0523, 0.0, 0.9986]]
    )
    rpy = matrix_pose(pose).tolist()
    cov = np.zeros((6, 6))
    cov[:3, :3] = np.eye(3) * 1.0e-9
    cov[3:, 3:] = np.eye(3) * (1.0e-6) ** 2
    return {
        "pose_valid": False,
        "optimizer_converged": True,
        "c2": {
            "resolution": "C2_UNRESOLVED",
            "hypotheses": [
                {
                    "id": "YAW_0",
                    "T_hand_plug_xyz_rpy": rpy,
                    "cost": 10.0,
                    "residual_rms": 3.0,
                    "condition_number": 100.0,
                    "covariance_6x6": cov.tolist(),
                },
                {
                    "id": "YAW_PI",
                    "T_hand_plug_xyz_rpy": rpy,
                    "cost": 10.0,
                    "residual_rms": 3.0,
                    "condition_number": 100.0,
                    "covariance_6x6": cov.tolist(),
                },
            ]
        },
    }


def test_posthoc_evaluator_reports_overconfidence_without_feedback(tmp_path):
    report_path = tmp_path / "report.json"
    shadow_path = tmp_path / "shadow.json"
    _write(report_path, _truth_report())
    _write(shadow_path, _shadow_result())
    result = evaluate_posthoc_shadow(
        report_path=report_path, shadow_result_path=shadow_path
    )
    assert result["truth_scope"] == "POSTHOC_TRUTH_ONLY"
    assert result["formal_pose_valid_claim"] is False
    assert result["c2_cost_tie_posthoc"] is True
    assert result["overconfidence_single_episode"] is True
    assert result["control_authorized"] is False
    assert result["truth_feedback_to_formal_estimator"] is False
    assert abs(result["hypotheses"][0]["posthoc_error"]["translation_error_m"] - 0.001) < 1.0e-9
    assert result["hypotheses"][0]["posthoc_error"]["rotation_geodesic_error_deg"] > 2.9


def test_real_dual_camera_archive_replay_contract():
    repository = Path(__file__).resolve().parents[3]
    base = (
        repository
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "phase1_codex_shadow_smoke_dual_camera_v0/seed000"
    )
    if not (base / "nominal_physics_report.json").is_file():
        pytest.skip("real dual-camera artifact not present")
    from kcg_connector.posthoc_shadow_evaluation import replay_shadow_estimate

    replay = replay_shadow_estimate(
        report_path=base / "nominal_physics_report.json",
        formal_archive_path=base / "postgrasp_shadow/formal_views",
    )
    assert replay["c2"]["rz_status"] == "C2_UNRESOLVED"
    assert replay["optimizer_converged"] is True
    assert replay["pose_valid"] is False
    assert replay["success"] is False
    assert replay["status"] == "REJECTED_T_HP_POSE_INVALID"
    assert replay["reject_reason"] is not None
    assert replay["control_authorized"] is False
    assert replay["c2"]["resolution"] == "C2_UNRESOLVED"
    assert replay["T_HP_independent_view_count"] == 2
    assert replay["T_HP_multiview"] is True


def test_real_view_ab_groups_are_all_pose_invalid():
    repository = Path(__file__).resolve().parents[3]
    base = (
        repository
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "phase1_codex_shadow_smoke_dual_camera_v0/seed000"
    )
    if not (base / "nominal_physics_report.json").is_file():
        pytest.skip("real dual-camera artifact not present")
    from kcg_connector.posthoc_shadow_evaluation import run_T_HP_view_ab

    ab = run_T_HP_view_ab(
        report_path=base / "nominal_physics_report.json",
        formal_archive_path=base / "postgrasp_shadow/formal_views",
        output_path=Path(__file__).resolve().parents[3]
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "deepseek/offline_view_ab_test",
    )
    assert ab["decision"] == "NO_GROUP_POSE_VALID"
    for name in ("A_wrist_v0_only", "B_fixed_world_only", "C_wrist_plus_fixed"):
        assert ab["groups"][name]["pose_valid"] is False
        assert ab["groups"][name]["success"] is False


def test_real_occlusion_ab_candidates_are_not_accepted():
    repository = Path(__file__).resolve().parents[3]
    base = (
        repository
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "phase1_codex_shadow_smoke_dual_camera_v0/seed000"
    )
    if not (base / "nominal_physics_report.json").is_file():
        pytest.skip("real dual-camera artifact not present")
    from kcg_connector.posthoc_shadow_evaluation import run_occlusion_policy_ab

    ab = run_occlusion_policy_ab(
        report_path=base / "nominal_physics_report.json",
        formal_archive_path=base / "postgrasp_shadow/formal_views",
        output_path=Path(__file__).resolve().parents[3]
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "deepseek/offline_occlusion_ab_test",
    )
    assert ab["decision"] == "NO_CANDIDATE_ACCEPTED"
    assert ab["groups"]["A0_baseline"]["pose_valid"] is False
    assert ab["groups"]["A1_ignore_foreground_occluded"]["pose_valid"] is False
    assert ab["groups"]["A2_ignore_foreground_and_cad_occluder"]["pose_valid"] is False
