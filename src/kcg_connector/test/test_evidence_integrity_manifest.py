from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from kcg_connector.evidence_integrity_manifest import (
    SCHEMA_VERSION,
    build_integrity_manifest,
    parse_structured_file,
    render_integrity_markdown,
    validate_declared_hash_pairs,
    write_integrity_outputs,
)


ROOT = Path(__file__).resolve().parents[3]
QUEUE = Path("artifacts/agent_control/WORK_QUEUE.yaml")
MODULE = ROOT / "src/kcg_connector/kcg_connector/evidence_integrity_manifest.py"
OUTPUTS = (
    Path("artifacts/agent_control/tasks/EIGHT-HOUR-G1-EVIDENCE-INTEGRITY-MANIFEST/INTEGRITY_MANIFEST.json"),
    Path("artifacts/agent_control/tasks/EIGHT-HOUR-G1-EVIDENCE-INTEGRITY-MANIFEST/INTEGRITY_MANIFEST_CN.md"),
)


@pytest.fixture(scope="module")
def manifest():
    return build_integrity_manifest(
        repository_root=ROOT,
        work_queue_path=QUEUE,
        generated_at_utc="2026-08-18T00:28:58Z",
        output_paths=OUTPUTS,
    )


def test_current_explicit_scope_is_complete_and_parseable(manifest):
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["scope_kind"] == "EXPLICIT_REFERENCE_GRAPH_NO_REPOSITORY_WALK"
    assert manifest["entry_count"] >= 70
    assert manifest["total_size_bytes"] > 70_000_000
    assert manifest["queue_task_count"] == 41
    assert manifest["queue_evidence_count"] == 41
    assert manifest["declared_hash_pair_count"] >= 25
    assert manifest["declared_hash_pairs_validated"] is True
    assert manifest["structured_parse_failure_count"] == 0


def test_every_manifest_entry_exists_and_matches_hash(manifest):
    paths = []
    for row in manifest["entries"]:
        path = ROOT / row["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert row["size_bytes"] == path.stat().st_size
        assert row["roles"] == sorted(set(row["roles"]))
        paths.append(row["path"])
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))


def test_frozen_high_detail_assets_match_human_decision(manifest):
    assert manifest["high_detail_frozen_asset_count"] == 2
    assert manifest["high_detail_baseline_sha256"] == (
        "5eb9ad82940e58a1592b6a66fd824c480ba24268cb1c20bcc84de653bb12c995"
    )
    assert manifest["rejected_local_variant_sha256"] == (
        "d41477ee18052662904212444b907607874a8c6c27399d3d344e44ee4fd18d67"
    )


def test_parked_unhashed_code_is_not_promoted_to_verified(manifest):
    assert manifest["parked_unverified_code_paths"] == [
        "src/kcg_connector/kcg_connector/d38999_multilayer_nut_regrasp.py",
        "src/kcg_connector/test/test_d38999_multilayer_nut_regrasp.py",
    ]


def test_outputs_are_excluded_and_no_dynamic_claim_is_created(manifest):
    paths = {row["path"] for row in manifest["entries"]}
    assert paths.isdisjoint({str(path) for path in OUTPUTS})
    assert "artifacts/agent_control/EIGHT_HOUR_FINAL_REPORT_CN.md" not in paths
    assert manifest["excluded_output_paths"] == sorted(str(path) for path in OUTPUTS)
    assert manifest["simulation_started"] is False
    assert manifest["assembly_success_claimed"] is False
    assert manifest["formal_r12_generated"] is False
    assert manifest["control_authorized"] is False
    assert manifest["hardware_authorized"] is False


def test_entry_content_is_deterministic_across_timestamps(manifest):
    second = build_integrity_manifest(
        repository_root=ROOT,
        work_queue_path=QUEUE,
        generated_at_utc="2026-08-18T00:29:00Z",
        output_paths=OUTPUTS,
    )
    assert second["entries"] == manifest["entries"]
    assert second["declared_hash_pair_count"] == manifest["declared_hash_pair_count"]
    assert second["generated_at_utc"] != manifest["generated_at_utc"]


def test_declared_hash_drift_and_path_escape_are_rejected(tmp_path):
    root = tmp_path / "repository"
    source = root / "artifacts/evidence.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")
    good = hashlib.sha256(source.read_bytes()).hexdigest()
    verified, unverified = validate_declared_hash_pairs(
        root, {"path": "artifacts/evidence.json", "sha256": good}
    )
    assert verified == [{"path": "artifacts/evidence.json", "sha256": good}]
    assert unverified == []
    with pytest.raises(ValueError, match="declared hash drift"):
        validate_declared_hash_pairs(
            root, {"path": "artifacts/evidence.json", "sha256": "0" * 64}
        )
    with pytest.raises(ValueError, match="absolute path is forbidden"):
        validate_declared_hash_pairs(
            root, {"path": "/etc/passwd", "sha256": "0" * 64}
        )


def test_corrupt_and_nonfinite_structured_inputs_are_rejected(tmp_path):
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        parse_structured_file(corrupt)
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        parse_structured_file(nonfinite)
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("a,a\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        parse_structured_file(duplicate)


def test_implementation_contains_no_broad_repository_walk():
    source = MODULE.read_text(encoding="utf-8")
    assert ".rglob(" not in source
    assert "os.walk" not in source
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports.isdisjoint({"isaacsim", "omni", "pxr", "rclpy", "torch"})


def test_output_pair_is_immutable_json_safe_and_markdown_bound(manifest, tmp_path):
    json_path = tmp_path / "manifest.json"
    markdown_path = tmp_path / "manifest.md"
    write_integrity_outputs(manifest, json_path, markdown_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert markdown == render_integrity_markdown(manifest)
    assert "不进行全仓库遍历" in markdown
    with pytest.raises(FileExistsError, match="immutable"):
        write_integrity_outputs(manifest, json_path, markdown_path)
