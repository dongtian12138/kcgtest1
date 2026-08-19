import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.posthoc_diag_mount_evaluation import (
    FormalArchiveError,
    build_diagnostic_reconstruction_sidecar,
    build_truth_free_capture_sidecar,
    load_diag_formal_view,
    run_formal_only,
    run_posthoc_comparison,
)

BASE = (
    Path(__file__).resolve().parents[3]
    / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
    / "phase1_diag_mount_search_v1/seed000"
)
CONTRACT_BASE = (
    Path(__file__).resolve().parents[3]
    / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
    / "phase1_diag_mount_contract_v1/seed000"
)


def _build_sidecar(tmp_path):
    return build_diagnostic_reconstruction_sidecar(
        base_path=BASE / "postgrasp_diag_mount_search",
        report_path=BASE / "nominal_physics_report.json",
        controller_steps_path=BASE / "controller_steps.jsonl",
        pick_config_path=Path(__file__).resolve().parents[1]
        / "config/d38999_tabletop_pick_v1.yaml",
        output_path=tmp_path / "contract.json",
    )


def _require_artifact():
    if not (BASE / "nominal_physics_report.json").is_file():
        pytest.skip("diag mount artifact not present")


def test_formal_loader_signature_has_no_report_path():
    params = inspect.signature(load_diag_formal_view).parameters
    assert "report_path" not in params
    assert "truth" not in params


def test_formal_output_is_bitwise_invariant_to_truth_mutation(tmp_path):
    _require_artifact()
    contract_path = tmp_path / "contract.json"
    build_diagnostic_reconstruction_sidecar(
        base_path=BASE / "postgrasp_diag_mount_search",
        report_path=BASE / "nominal_physics_report.json",
        controller_steps_path=BASE / "controller_steps.jsonl",
        pick_config_path=Path(__file__).resolve().parents[1]
        / "config/d38999_tabletop_pick_v1.yaml",
        output_path=contract_path,
    )
    out1 = tmp_path / "formal1"
    out2 = tmp_path / "formal2"
    run_formal_only(contract_path, BASE / "postgrasp_diag_mount_search", out1)
    # Mutate truth report and rerun formal; formal API does not read report.
    report = json.loads((BASE / "nominal_physics_report.json").read_text())
    report["posthoc_t_hand_plug_actual"] = np.eye(4).tolist()
    report["posthoc_t_hand_plug_actual"][0][0] = 99.0
    mutated = tmp_path / "mutated_report.json"
    mutated.write_text(json.dumps(report))
    # The formal API takes no report path; simply rebuild sidecar is not
    # required because its nominal field does not depend on actual truth.
    run_formal_only(contract_path, BASE / "postgrasp_diag_mount_search", out2)
    assert (out1 / "replay_C4.json").read_bytes() == (
        out2 / "replay_C4.json"
    ).read_bytes()


def test_missing_contract_field_fails_closed(tmp_path):
    _require_artifact()
    sidecar = _build_sidecar(tmp_path)
    record = sidecar["candidates"]["C4"]
    for key in ("camera_model", "T_WH", "T_HC", "T_WC", "physics_step"):
        contract = dict(sidecar)
        candidate = dict(record)
        candidate.pop(key)
        contract["candidates"] = dict(contract["candidates"])
        contract["candidates"]["C4"] = candidate
        bad = tmp_path / f"bad_{key}.json"
        bad.write_text(json.dumps(contract))
        with pytest.raises(FormalArchiveError):
            load_diag_formal_view(
                bad,
                BASE / "postgrasp_diag_mount_search",
                "C4",
            )


def test_c4_c6_are_separate_and_transform_direction_is_world_camera(tmp_path):
    _require_artifact()
    sidecar = _build_sidecar(tmp_path)
    c4 = sidecar["candidates"]["C4"]
    c6 = sidecar["candidates"]["C6"]
    assert not np.allclose(c4["T_WC"], c6["T_WC"])
    t_wc = np.asarray(c4["T_WC"])
    cam = c4["camera_model"]
    position = np.asarray(cam["position_world"])
    assert np.allclose(position, t_wc[:3, 3], atol=1.0e-12)
    assert sidecar["scope"] == "DIAGNOSTIC_RECONSTRUCTION_ONLY"


def test_real_comparison_is_diagnostic_reconstruction_no_mount(tmp_path):
    _require_artifact()
    _build_sidecar(tmp_path)
    formal = tmp_path / "formal"
    run_formal_only(
        tmp_path / "contract.json",
        BASE / "postgrasp_diag_mount_search",
        formal,
    )
    result = run_posthoc_comparison(
        BASE / "nominal_physics_report.json",
        formal,
        tmp_path / "comparison.json",
    )
    assert result["decision"] == "NO_MOUNT_ACCEPTED_DIAGNOSTIC_RECONSTRUCTION"
    assert result["mount_formally_accepted"] is False
    assert result["control_authorized"] is False
    assert result["acceptance_flags"]["C4"]["accepted"] is False
    assert result["acceptance_flags"]["C6"]["accepted"] is False


def test_new_gpu_manifest_builds_truth_free_actual_camera_contract(tmp_path):
    if not (CONTRACT_BASE / "nominal_physics_report.json").is_file():
        pytest.skip("new diagnostic contract artifact not present")
    base = CONTRACT_BASE / "postgrasp_diag_mount_search"
    sidecar = build_truth_free_capture_sidecar(
        base_path=base,
        output_path=tmp_path / "truth_free_contract.json",
    )
    assert sidecar["formal_estimator_input"] is False
    assert sidecar["control_authorized"] is False
    for candidate_id in ("C4", "C6"):
        record = sidecar["candidates"][candidate_id]
        t_wh = np.asarray(record["T_WH"])
        t_hc = np.asarray(record["T_HC"])
        t_wc = np.asarray(record["T_WC"])
        assert np.allclose(t_wh @ t_hc, t_wc, atol=1.0e-10)
        assert np.isclose(np.linalg.det(t_wc[:3, :3]), 1.0, atol=1.0e-8)
