from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from kcg_connector.assembly_evidence_report import (
    EXPECTED_TASK_KEYS,
    SCHEMA_VERSION,
    VIDEO_MANIFEST_SCHEMA,
    build_assembly_evidence_report,
    render_report_markdown,
    write_report_pair,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORK_QUEUE = REPOSITORY_ROOT / "artifacts/agent_control/WORK_QUEUE.yaml"
GENERATED_AT = "2026-08-18T00:04:19Z"


def _queue() -> dict:
    return yaml.safe_load(WORK_QUEUE.read_text(encoding="utf-8"))


def _copy_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "repository"
    queue = _queue()
    for group in queue["groups"].values():
        for task in group["tasks"].values():
            evidence = task.get("evidence")
            if not isinstance(evidence, str):
                continue
            source = REPOSITORY_ROOT / evidence
            destination = root / evidence
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            document = json.loads(source.read_text(encoding="utf-8"))
            for row in document.get("code_files", []):
                code_source = REPOSITORY_ROOT / row["path"]
                code_destination = root / row["path"]
                code_destination.parent.mkdir(parents=True, exist_ok=True)
                code_destination.write_bytes(code_source.read_bytes())
    queue_path = root / "artifacts/agent_control/WORK_QUEUE.yaml"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return root, queue_path, queue


def _write_queue(path: Path, queue: dict) -> None:
    path.write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _build(root: Path = REPOSITORY_ROOT, queue: Path = WORK_QUEUE, **kwargs):
    return build_assembly_evidence_report(
        repository_root=root,
        work_queue_path=queue.relative_to(root),
        generated_at_utc=GENERATED_AT,
        **kwargs,
    )


def test_current_queue_is_fully_inventoried_without_dynamic_claim():
    report = _build()
    assert tuple(row["task_key"] for row in report["tasks"]) == EXPECTED_TASK_KEYS
    assert report["task_count"] == 41
    assert report["status_counts"]["OFFLINE_PASS"] == 30
    assert report["status_counts"]["STATIC_PASS"] == 4
    assert report["status_counts"]["PARKED"] == 3
    assert report["status_counts"]["NOT_STARTED"] == 3
    assert report["status_counts"]["IMPLEMENTING"] == 1
    assert report["task_evidence_present_count"] == 35
    assert report["missing_expected_task_evidence"] == ["A1", "A2"]
    assert report["dynamic_task_count"] == 0
    assert report["current_frontier_state"] == "HOME"
    assert report["assembly_success_claimed"] is False


def test_current_report_has_no_fake_video_or_performance():
    report = _build()
    assert report["video"]["available"] is False
    assert report["video"]["count"] == 0
    assert report["performance"]["available"] is False
    assert report["performance"]["target_process_vram_peak_mib"] is None
    assert report["performance"]["physics_steps_per_wall_second"] is None
    assert report["performance"]["render_fps"] is None
    assert report["simulation_started_by_reporter"] is False
    assert report["robot_commands_emitted_by_reporter"] == 0


def test_all_present_evidence_is_hashed_and_status_matched():
    report = _build()
    present = [row for row in report["tasks"] if row["evidence"]["path"]]
    assert len(present) == 35
    assert all(len(row["evidence"]["sha256"]) == 64 for row in present)
    assert all(
        row["queue_status"] == row["evidence"]["document_status"]
        for row in present
    )
    e3 = next(row for row in present if row["task_key"] == "E3")
    assert e3["queue_status"] == "PARKED"
    assert e3["evidence"]["declared_code_files_checked"] == 0
    assert e3["evidence"]["declared_code_files_unverified"] == 2


def test_unknown_status_and_inventory_change_are_rejected(tmp_path):
    root, queue_path, queue = _copy_fixture(tmp_path)
    queue["groups"]["B"]["tasks"]["B2"]["status"] = "SUCCESS"
    _write_queue(queue_path, queue)
    with pytest.raises(ValueError, match="unsupported status"):
        _build(root, queue_path)
    queue = _queue()
    del queue["groups"]["B"]["tasks"]["B2"]
    _write_queue(queue_path, queue)
    with pytest.raises(ValueError, match="inventory/order"):
        _build(root, queue_path)


def test_evidence_path_escape_and_status_mismatch_are_rejected(tmp_path):
    root, queue_path, queue = _copy_fixture(tmp_path)
    queue["groups"]["B"]["tasks"]["B5"]["evidence"] = "../../etc/passwd"
    _write_queue(queue_path, queue)
    with pytest.raises(ValueError, match="escapes repository root"):
        _build(root, queue_path)
    queue = _queue()
    queue["groups"]["B"]["tasks"]["B5"]["status"] = "STATIC_PASS"
    _write_queue(queue_path, queue)
    with pytest.raises(ValueError, match="queue/evidence status mismatch"):
        _build(root, queue_path)


def test_true_dynamic_claim_in_offline_evidence_is_rejected(tmp_path):
    root, queue_path, queue = _copy_fixture(tmp_path)
    path = root / queue["groups"]["C"]["tasks"]["C1"]["evidence"]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["dynamic_camera_pass_claimed"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="non-dynamic evidence contains true claims"):
        _build(root, queue_path)


def test_dynamic_pass_requires_strict_independent_envelope(tmp_path):
    root, queue_path, queue = _copy_fixture(tmp_path)
    evidence = Path("artifacts/agent_control/runtime_evidence/B2.json")
    path = root / evidence
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "DYNAMIC_PASS", "dynamic_pass_claimed": True}),
        encoding="utf-8",
    )
    queue["groups"]["B"]["tasks"]["B2"].update(
        {"status": "DYNAMIC_PASS", "evidence": str(evidence)}
    )
    _write_queue(queue_path, queue)
    with pytest.raises(ValueError, match="strict evidence_level"):
        _build(root, queue_path)


def test_declared_code_hash_drift_is_rejected(tmp_path):
    root, queue_path, queue = _copy_fixture(tmp_path)
    f2 = root / queue["groups"]["F"]["tasks"]["F2"]["evidence"]
    document = json.loads(f2.read_text(encoding="utf-8"))
    code = root / document["code_files"][0]["path"]
    code.write_text(code.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declared code file hash drift"):
        _build(root, queue_path)


def test_prior_report_detects_task_evidence_hash_drift(tmp_path):
    root, queue_path, queue = _copy_fixture(tmp_path)
    first = _build(root, queue_path)
    prior = root / "artifacts/agent_control/tasks/EIGHT-HOUR-F3-EVIDENCE-REPORT/prior.json"
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text(json.dumps(first), encoding="utf-8")
    evidence = root / queue["groups"]["B"]["tasks"]["B1"]["evidence"]
    document = json.loads(evidence.read_text(encoding="utf-8"))
    document["audit_note"] = "changed"
    evidence.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="task evidence hash drift"):
        _build(root, queue_path, prior_report_path=prior.relative_to(root))


def test_video_inventory_accepts_only_attributable_hashed_runtime_files(tmp_path):
    root, queue_path, _ = _copy_fixture(tmp_path)
    video = root / "artifacts/agent_control/runtime_evidence/videos/run.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"runtime-video-container-placeholder")
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    manifest = root / "artifacts/agent_control/runtime_evidence/video_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": VIDEO_MANIFEST_SCHEMA,
                "videos": [
                    {
                        "path": str(video.relative_to(root)),
                        "sha256": digest,
                        "source": "isaac_runtime_capture",
                        "synthetic": False,
                        "simulation_started": True,
                        "source_process_pid": 4242,
                        "role": "overview",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = _build(root, queue_path, video_manifest_path=manifest.relative_to(root))
    assert report["video"]["available"] is True
    assert report["video"]["count"] == 1
    assert report["video"]["playback_content_verified"] is False
    bad = json.loads(manifest.read_text(encoding="utf-8"))
    bad["videos"][0]["synthetic"] = True
    manifest.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="not attributable runtime evidence"):
        _build(root, queue_path, video_manifest_path=manifest.relative_to(root))


def _performance_document(offline: bool) -> dict:
    return {
        "status": "OFFLINE_PASS" if offline else "MEASURED",
        "measurement_kind": "OFFLINE_TEST_FIXTURE" if offline else "REAL",
        "offline_fixture": offline,
        "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
        "target_process_vram_peak_mib": 4096.0,
        "physics_steps_per_wall_second": {"values": [700.0, 710.0, 705.0]},
        "render": {"overall_fps": 45.0},
        "dynamic_pass_claimed": False,
        "hardware_authorized": False,
    }


def test_performance_inventory_rejects_fixture_and_accepts_real_summary(tmp_path):
    root, queue_path, _ = _copy_fixture(tmp_path)
    relative = Path(
        "artifacts/agent_control/tasks/EIGHT-HOUR-A4-PERFORMANCE-MEASURED/summary.json"
    )
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_performance_document(True)), encoding="utf-8")
    with pytest.raises(ValueError, match="not real process-bound evidence"):
        _build(root, queue_path, performance_summary_path=relative)
    path.write_text(json.dumps(_performance_document(False)), encoding="utf-8")
    report = _build(root, queue_path, performance_summary_path=relative)
    assert report["performance"]["available"] is True
    assert report["performance"]["target_process_vram_peak_mib"] == 4096.0
    assert report["performance"]["physics_steps_per_wall_second"] == [700.0, 710.0, 705.0]
    assert report["performance"]["render_fps"] == 45.0


def test_report_pair_is_json_safe_markdown_first_table_and_immutable(tmp_path):
    report = _build()
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_report_pair(report, json_path, markdown_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert markdown.startswith("# 完整装配证据、视频与统计报告\n\n| 任务 |")
    assert "本报告器不启动仿真" in markdown
    with pytest.raises(FileExistsError, match="immutable"):
        write_report_pair(report, json_path, markdown_path)


def test_markdown_preserves_missing_measurements_instead_of_fabricating_values():
    markdown = render_report_markdown(_build())
    assert "可追溯视频数：0" in markdown
    assert "实测性能：未提供" in markdown
    assert "显存峰值 0" not in markdown
