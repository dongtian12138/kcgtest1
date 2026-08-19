from pathlib import Path

import pytest

from kcg_connector.fc_pgdf_posthoc import (
    MINIMUM_PLUG_FACE_SHORT_AXIS_PX,
    evaluate_fc_pgdf_01a_posthoc,
)


def test_fc_pgdf_01a_posthoc_reports_pixel_infeasible():
    repository = Path(__file__).resolve().parents[3]
    output_root = (
        repository
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "phase3_fc_pgdf_01a_smoke_v1/seed000"
    )
    snapshot_path = (
        repository
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "phase1_snapshot_gate_v3/seed000"
        / "postgrasp_snapshot_gate/snapshot_gate.json"
    )
    if not output_root.is_dir() or not snapshot_path.is_file():
        pytest.skip("FC-PGDF-01a smoke artifacts not present")

    report = evaluate_fc_pgdf_01a_posthoc(
        snapshot_path=snapshot_path,
        output_root=output_root,
    )
    assert report["status"] == "CURRENT_FIXED_CAMERA_PIXEL_INFEASIBLE"
    assert (
        report["maximum_plug_face_short_axis_px"]
        < MINIMUM_PLUG_FACE_SHORT_AXIS_PX
    )
    assert report["direction_gate"] is True
    assert report["slip_gate"] is True
    assert report["object_pose_writes_after_restore"] == 0
    assert report["formal_estimator_input"] is False
    assert report["control_authorized"] is False
    assert (output_root / "fc_pgdf_posthoc_report.json").is_file()
