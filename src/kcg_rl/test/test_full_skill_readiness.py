"""Pure tests for the fail-closed full-skill training gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from kcg_rl.full_skill_readiness import (
    EVIDENCE_SCHEMA_VERSION,
    FULL_SKILL_INTERFACE_VERSION,
    POLICY_ACTIVE_STAGES,
    REQUIRED_GATE_CATEGORIES,
    REQUIRED_GATE_IDS,
    WORKFLOW_STAGES,
    check_training_readiness,
    load_readiness_manifest,
    require_training_ready,
)


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "config"
    / "d38999_full_skill_rl_readiness_v1.yaml"
)
REPOSITORY = CONFIG_PATH.parents[3]


def _document():
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _passing_metric(rule):
    if "equal" in rule:
        return rule["equal"]
    if "minimum" in rule:
        return rule["minimum"]
    return rule["maximum"]


def _limited(document, evidence_id):
    return next(
        item
        for item in document["limited_evidence"]
        if item["evidence_id"] == evidence_id
    )


def _refresh_binding(document, tmp_path, evidence_id, artifact_key):
    artifact = _limited(document, evidence_id)["artifacts"][artifact_key]
    payload = (tmp_path / artifact["path"]).read_bytes()
    artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    artifact["size_bytes"] = len(payload)


def _rewrite_manifest(config_path, document):
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def _make_passing_repository(tmp_path):
    """Create synthetic, internally hashed evidence for every gate."""
    document = _document()
    document["training"]["enabled"] = True
    config_path = tmp_path / "readiness.yaml"
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    # Limited evidence remains independently hash/schema checked even in the
    # synthetic all-formal-gates-pass fixture.  Copy exact immutable inputs;
    # never replace them with filename-only test doubles.
    copied = set()
    for limited in document["limited_evidence"]:
        for artifact in limited["artifacts"].values():
            relative = Path(artifact["path"])
            if relative in copied:
                continue
            source = REPOSITORY / relative
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.add(relative)
    for gate in document["gates"]:
        artifacts = {}
        for key in gate["required_artifacts"]:
            relative = Path("artifacts") / gate["gate_id"] / f"{key}.txt"
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = f"evidence for {gate['gate_id']} {key}\n".encode()
            target.write_bytes(payload)
            artifacts[key] = {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        evidence = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "contract_version": FULL_SKILL_INTERFACE_VERSION,
            "gate_id": gate["gate_id"],
            "passed": True,
            "run_id": f"test-{gate['gate_id']}",
            "generated_utc": "2026-08-12T12:34:56Z",
            "command": "synthetic pure-test producer",
            "checks": {name: True for name in gate["required_checks"]},
            "metrics": {
                name: _passing_metric(rule)
                for name, rule in gate["required_metrics"].items()
            },
            "artifacts": artifacts,
        }
        evidence_path = tmp_path / gate["evidence_path"]
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            yaml.safe_dump(evidence, sort_keys=False), encoding="utf-8"
        )
    return config_path, document


def test_checked_in_manifest_is_disabled_complete_and_does_not_mutate_v0():
    manifest = load_readiness_manifest(CONFIG_PATH)
    document = _document()
    training = document["training"]
    workflow = document["workflow"]
    assert manifest.training_enabled is False
    assert training["base_residual_interface_version"] == (
        "kcg_connector_twist_residual_v0"
    )
    assert training["active_v0_modified"] is False
    assert training["action_size"] == 4
    assert training["observation_size"] == 30
    assert tuple(workflow["stage_order"]) == WORKFLOW_STAGES
    assert tuple(workflow["policy_active_stages"]) == POLICY_ACTIVE_STAGES
    assert workflow["simulation_ground_truth_control_authority"] is False
    assert {gate.gate_id for gate in manifest.gates} == REQUIRED_GATE_IDS
    assert {gate.category for gate in manifest.gates} == (
        REQUIRED_GATE_CATEGORIES
    )
    assert len(manifest.limited_evidence) == 4


def test_checked_in_tooth_v2_is_active_valid_limited_and_nonpromotional():
    manifest = load_readiness_manifest(CONFIG_PATH)
    report = check_training_readiness(manifest, REPOSITORY)
    assert report.ready is False
    assert report.training_enabled is False
    assert len(report.limited_evidence_results) == 4
    active = [
        item
        for item in report.limited_evidence_results
        if item.disposition == "active"
    ]
    assert len(active) == 4
    assert all(item.valid for item in active)
    tooth = next(
        item
        for item in report.limited_evidence_results
        if item.evidence_id == "nut_tooth_six_view_identity_limited_v2"
    )
    assert tooth.disposition == "active"
    assert tooth.valid is True
    assert not tooth.reasons
    assert "combined_sequence_identity_union_is_24_of_24_but_not_rgb_only" in (
        tooth.scope
    )
    assert "render_jitter_remains_unresolved" in tooth.limitations
    assert (
        "zero_transitions_have_all_24_identities_after_posthoc_recovery"
        in tooth.limitations
    )
    assert all(not item.passed for item in report.gate_results)
    encoded = report.as_dict()
    assert all(
        item["closes_full_skill_gate"] is False
        for item in encoded["limited_evidence_results"]
    )
    encoded_tooth = next(
        item
        for item in encoded["limited_evidence_results"]
        if item["evidence_id"]
        == "nut_tooth_six_view_identity_limited_v2"
    )
    assert encoded_tooth["disposition"] == "active"
    assert encoded_tooth["counts_toward_readiness"] is True
    assert encoded_tooth["closes_full_skill_gate"] is False
    scopes = {
        item.evidence_id: set(item.limitations)
        for item in report.limited_evidence_results
    }
    assert "no_keyed_yaw_or_full_6d_pose" in scopes[
        "multisite_rgbd_xy_five_of_five"
    ]
    assert "continuous_collision_verification_is_false" in scopes[
        "smooth_e2e_three_headless_runs"
    ]


def test_checked_in_manifest_fails_closed_with_actionable_missing_evidence(
    tmp_path,
):
    manifest = load_readiness_manifest(CONFIG_PATH)
    report = check_training_readiness(manifest, tmp_path)
    assert report.ready is False
    assert report.training_enabled is False
    assert "training.enabled is false" in report.global_reasons[0]
    assert len(report.gate_results) == len(REQUIRED_GATE_IDS)
    assert all(not result.passed for result in report.gate_results)
    assert all(
        result.reasons[0].startswith("missing evidence file:")
        for result in report.gate_results
    )


def test_all_gates_enable_training_only_with_verified_artifacts(tmp_path):
    config_path, _ = _make_passing_repository(tmp_path)
    manifest = load_readiness_manifest(config_path)
    report = check_training_readiness(manifest, tmp_path)
    assert report.ready is True
    assert not report.global_reasons
    assert all(result.passed for result in report.gate_results)
    assert require_training_ready(config_path, tmp_path).ready is True


def test_false_required_check_blocks_training(tmp_path):
    config_path, document = _make_passing_repository(tmp_path)
    gate = document["gates"][0]
    evidence_path = tmp_path / gate["evidence_path"]
    evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
    check_name = gate["required_checks"][0]
    evidence["checks"][check_name] = False
    evidence_path.write_text(
        yaml.safe_dump(evidence, sort_keys=False), encoding="utf-8"
    )
    report = check_training_readiness(
        load_readiness_manifest(config_path), tmp_path
    )
    result = next(
        item for item in report.gate_results if item.gate_id == gate["gate_id"]
    )
    assert not report.ready
    assert f"required check is not true: {check_name}" in result.reasons


def test_out_of_bound_metric_blocks_training(tmp_path):
    config_path, document = _make_passing_repository(tmp_path)
    gate = next(
        item
        for item in document["gates"]
        if item["gate_id"] == "coupling_nut_tooth_jitter"
    )
    evidence_path = tmp_path / gate["evidence_path"]
    evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
    evidence["metrics"]["unresolved_jitter_events"] = 1
    evidence_path.write_text(
        yaml.safe_dump(evidence, sort_keys=False), encoding="utf-8"
    )
    report = check_training_readiness(
        load_readiness_manifest(config_path), tmp_path
    )
    result = next(
        item for item in report.gate_results if item.gate_id == gate["gate_id"]
    )
    assert not report.ready
    assert any(
        "unresolved_jitter_events=1 != 0" in reason
        for reason in result.reasons
    )


def test_artifact_tampering_is_detected_by_size_and_hash(tmp_path):
    config_path, document = _make_passing_repository(tmp_path)
    gate = document["gates"][0]
    evidence_path = tmp_path / gate["evidence_path"]
    evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
    artifact = evidence["artifacts"][gate["required_artifacts"][0]]
    (tmp_path / artifact["path"]).write_bytes(b"tampered")
    report = check_training_readiness(
        load_readiness_manifest(config_path), tmp_path
    )
    result = next(
        item for item in report.gate_results if item.gate_id == gate["gate_id"]
    )
    assert not report.ready
    assert any("size mismatch" in reason for reason in result.reasons)
    assert any("SHA256 mismatch" in reason for reason in result.reasons)


def test_limited_evidence_tampering_is_detected_before_schema_use(tmp_path):
    config_path, document = _make_passing_repository(tmp_path)
    limited = _limited(document, "multisite_rgbd_xy_five_of_five")
    artifact = limited["artifacts"]["report"]
    (tmp_path / artifact["path"]).write_bytes(b"tampered")
    report = check_training_readiness(
        load_readiness_manifest(config_path), tmp_path
    )
    result = next(
        item
        for item in report.limited_evidence_results
        if item.evidence_id == limited["evidence_id"]
    )
    assert report.ready is False
    assert result.valid is False
    assert any("size mismatch" in reason for reason in result.reasons)
    assert any("SHA256 mismatch" in reason for reason in result.reasons)


def test_multisite_schema_boundary_rejects_full_6d_overclaim(tmp_path):
    config_path, document = _make_passing_repository(tmp_path)
    evidence_id = "multisite_rgbd_xy_five_of_five"
    artifact = _limited(document, evidence_id)["artifacts"]["report"]
    path = tmp_path / artifact["path"]
    report_document = json.loads(path.read_text(encoding="utf-8"))
    report_document["pose_scope"]["full_6d"] = True
    path.write_text(json.dumps(report_document), encoding="utf-8")
    _refresh_binding(document, tmp_path, evidence_id, "report")
    _rewrite_manifest(config_path, document)

    report = check_training_readiness(
        load_readiness_manifest(config_path), tmp_path
    )
    result = next(
        item
        for item in report.limited_evidence_results
        if item.evidence_id == evidence_id
    )
    assert report.ready is False
    assert result.valid is False
    assert "pose_scope.full_6d must remain false" in result.reasons[0]


def test_ft_schema_boundary_rejects_training_ready_claim(tmp_path):
    config_path, document = _make_passing_repository(tmp_path)
    evidence_ids = (
        "wrist_ft_monitor_three_run_repeatability",
        "smooth_e2e_three_headless_runs",
    )
    artifact = _limited(document, evidence_ids[0])["artifacts"][
        "repeatability_report"
    ]
    path = tmp_path / artifact["path"]
    report_document = json.loads(path.read_text(encoding="utf-8"))
    report_document["claims"]["training_ready_claimed"] = True
    path.write_text(json.dumps(report_document), encoding="utf-8")
    for evidence_id in evidence_ids:
        _refresh_binding(
            document, tmp_path, evidence_id, "repeatability_report"
        )
    _rewrite_manifest(config_path, document)

    report = check_training_readiness(
        load_readiness_manifest(config_path), tmp_path
    )
    invalid = {
        item.evidence_id: item.reasons
        for item in report.limited_evidence_results
        if item.disposition == "active" and not item.valid
    }
    assert set(invalid) == set(evidence_ids)
    assert all(
        "claim boundary mismatch" in reasons[0]
        for reasons in invalid.values()
    )


def test_tooth_v2_schema_rejects_render_jitter_resolution_overclaim(tmp_path):
    config_path, document = _make_passing_repository(tmp_path)
    evidence_id = "nut_tooth_six_view_identity_limited_v2"
    limited = _limited(document, evidence_id)
    artifact = limited["artifacts"]["segment23_report"]
    path = tmp_path / artifact["path"]
    report_document = json.loads(path.read_text(encoding="utf-8"))
    report_document["visual_diagnostics_only"][
        "render_jitter_absence_claim_authorized"
    ] = True
    path.write_text(json.dumps(report_document), encoding="utf-8")
    _refresh_binding(document, tmp_path, evidence_id, "segment23_report")

    # Keep the producer manifest internally bound so the mutation reaches the
    # semantic no-overclaim boundary instead of failing first at provenance.
    manifest_artifact = limited["artifacts"]["segment23_manifest"]
    manifest_path = tmp_path / manifest_artifact["path"]
    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = path.read_bytes()
    manifest_document["outputs"]["report"] = {
        "path": artifact["path"],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    manifest_path.write_text(json.dumps(manifest_document), encoding="utf-8")
    _refresh_binding(document, tmp_path, evidence_id, "segment23_manifest")
    _rewrite_manifest(config_path, document)

    report = check_training_readiness(
        load_readiness_manifest(config_path), tmp_path
    )
    result = next(
        item
        for item in report.limited_evidence_results
        if item.evidence_id == evidence_id
    )
    assert result.valid is False
    assert "must not become a render no-jitter claim" in result.reasons[0]


def test_tooth_v2_execution_source_drift_invalidates_active_record(tmp_path):
    config_path, document = _make_passing_repository(tmp_path)
    evidence_id = "nut_tooth_six_view_identity_limited_v2"
    artifact = _limited(document, evidence_id)["artifacts"][
        "prepared_runner_source"
    ]
    path = tmp_path / artifact["path"]
    path.write_bytes(path.read_bytes() + b"\n# synthetic source drift\n")

    report = check_training_readiness(
        load_readiness_manifest(config_path), tmp_path
    )
    result = next(
        item
        for item in report.limited_evidence_results
        if item.evidence_id == evidence_id
    )
    assert result.valid is False
    assert any("size mismatch" in reason for reason in result.reasons)
    assert any("SHA256 mismatch" in reason for reason in result.reasons)


def test_manifest_rejects_path_escape_and_missing_gate(tmp_path):
    document = _document()
    document["gates"][0]["evidence_path"] = "../outside.yaml"
    path = tmp_path / "escape.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="repository-relative safe path"):
        load_readiness_manifest(path)

    document = _document()
    document["gates"].pop()
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        load_readiness_manifest(path)


def test_manifest_requires_complete_hash_bound_limited_evidence(tmp_path):
    document = _document()
    document["limited_evidence"].pop()
    path = tmp_path / "missing_limited.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="limited evidence ids"):
        load_readiness_manifest(path)

    document = _document()
    report = document["limited_evidence"][0]["artifacts"]["report"]
    report.pop("sha256")
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="keys do not match schema"):
        load_readiness_manifest(path)

    document = _document()
    document["limited_evidence"][0]["disposition"] = "retired"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="active or superseded"):
        load_readiness_manifest(path)


def test_require_ready_lists_blockers_in_exception(tmp_path):
    with pytest.raises(RuntimeError, match="perception_tabletop_pose"):
        require_training_ready(CONFIG_PATH, tmp_path)


def test_cli_returns_one_for_checked_in_disabled_contract(tmp_path):
    package_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "kcg_rl.full_skill_readiness",
            "--config",
            str(CONFIG_PATH),
            "--repo-root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(package_root)},
    )
    assert completed.returncode == 1
    assert "FULL SKILL RL READINESS: BLOCKED" in completed.stdout
    assert "INVALID LIMITED" in completed.stdout
    assert "perception_tabletop_pose" in completed.stdout


def test_import_does_not_load_ros_isaac_or_torch():
    package_root = Path(__file__).resolve().parents[1]
    script = """
import importlib
import sys

module = importlib.import_module("kcg_rl.full_skill_readiness")
assert module.ACTION_SIZE == 4
assert module.OBSERVATION_SIZE == 30
for name in ("rclpy", "omni", "isaacsim", "torch"):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(package_root)},
    )
    assert completed.returncode == 0, completed.stderr
