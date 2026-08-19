from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kcg_connector.stable_closeout_integrity import (
    G7_SELF_PATHS,
    MUTABLE_CONTROL_PATHS,
    SCHEMA_VERSION,
    build_stable_manifest,
    check_stable_manifest,
    write_new_json,
)


ROOT = Path(__file__).resolve().parents[3]


def _baseline():
    return build_stable_manifest(
        repository_root=ROOT,
        work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
        master_state_path="artifacts/agent_control/MASTER_STATE.json",
        blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        readiness_report_path="artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json",
        preflight_path="artifacts/agent_control/tasks/EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT/FINAL_REVIEW_PREFLIGHT.json",
        generated_at_utc="2026-08-18T01:20:00Z",
    )


def test_current_stable_manifest_is_direct_and_large():
    baseline = _baseline()
    assert baseline["schema_version"] == SCHEMA_VERSION
    assert baseline["result"] == "BASELINE_CREATED"
    assert baseline["stable_source_count"] > 220
    assert len(baseline["stable_manifest_digest"]) == 64
    assert len(baseline["stable_sources"]) == baseline["stable_source_count"]


def test_mutable_control_paths_are_exactly_excluded():
    baseline = _baseline()
    paths = {row["path"] for row in baseline["stable_sources"]}
    assert baseline["mutable_control_path_count"] == 9
    assert set(baseline["mutable_control_paths_excluded"]) == MUTABLE_CONTROL_PATHS
    assert not paths & MUTABLE_CONTROL_PATHS


def test_g7_self_files_are_monitored():
    paths = {row["path"] for row in _baseline()["stable_sources"]}
    assert set(G7_SELF_PATHS) <= paths


def test_three_frozen_high_detail_files_match():
    baseline = _baseline()
    assert baseline["frozen_high_detail_asset_count"] == 3
    rows = {row["role"]: row for row in baseline["frozen_high_detail_assets"]}
    assert rows["high_detail_baseline"]["sha256"] == "5eb9ad82940e58a1592b6a66fd824c480ba24268cb1c20bcc84de653bb12c995"
    assert rows["rejected_local_variant"]["sha256"] == "d41477ee18052662904212444b907607874a8c6c27399d3d344e44ee4fd18d67"
    assert rows["rejected_variant_build_result"]["sha256"] == "4551c1a9900b421e85c53e7dabd9821cb8349ada80fb5ab95a1522951b79febb"


def test_immediate_check_has_zero_missing_and_drift():
    check = check_stable_manifest(
        repository_root=ROOT,
        baseline=_baseline(),
        checked_at_utc="2026-08-18T01:21:00Z",
    )
    assert check["result"] == "PASS"
    assert check["missing_source_count"] == 0
    assert check["drift_source_count"] == 0
    assert check["current_manifest_digest"] == check["baseline_manifest_digest"]


def test_tampered_baseline_is_detected_without_editing_source():
    baseline = copy.deepcopy(_baseline())
    baseline["stable_sources"][0]["sha256"] = "0" * 64
    check = check_stable_manifest(
        repository_root=ROOT,
        baseline=baseline,
        checked_at_utc="2026-08-18T01:21:00Z",
    )
    assert check["result"] == "FAIL"
    assert check["drift_source_count"] == 1


def test_formal_outputs_and_runtime_side_effects_are_zero():
    baseline = _baseline()
    assert baseline["formal_final_output_count"] == 0
    assert baseline["g1_recursive_reference_classifier_used"] is False
    assert baseline["simulation_started"] is False
    assert baseline["robot_commands_emitted"] == 0


def test_rows_are_unique_existing_repository_paths():
    rows = _baseline()["stable_sources"]
    paths = [row["path"] for row in rows]
    assert len(paths) == len(set(paths))
    assert all(not Path(path).is_absolute() for path in paths)
    assert all((ROOT / path).is_file() for path in paths)
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_outputs_are_immutable(tmp_path):
    output = tmp_path / "baseline.json"
    write_new_json(_baseline(), output)
    assert json.loads(output.read_text())["result"] == "BASELINE_CREATED"
    with pytest.raises(FileExistsError, match="immutable"):
        write_new_json(_baseline(), output)


def test_no_acceptance_or_hardware_claim():
    baseline = _baseline()
    assert baseline["assembly_success_claimed"] is False
    assert baseline["formal_r12_generated"] is False
    assert baseline["control_authorized"] is False
    assert baseline["hardware_authorized"] is False
