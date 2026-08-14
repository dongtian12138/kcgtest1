"""Pure contracts for hash-bound prepared-twist tooth frame capture."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_PATH = PACKAGE_ROOT / "isaac/d38999_tooth_sync_capture.py"
REGRASP_PATH = PACKAGE_ROOT / "isaac/d38999_nut_regrasp_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "d38999_tooth_sync_capture_test", CAPTURE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner():
    isaac_directory = str(REGRASP_PATH.parent)
    sys.path.insert(0, isaac_directory)
    try:
        spec = importlib.util.spec_from_file_location(
            "d38999_nut_regrasp_sync_capture_test", REGRASP_PATH
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(isaac_directory)


def _dual_rows(module, samples):
    rows = []
    for sample_index, (step, phase, phase_step) in enumerate(samples):
        for view_id in module.VIEW_IDS:
            rows.append(
                {
                    "frame_index": len(rows),
                    "sample_index": sample_index,
                    "view_id": view_id,
                    "global_step": step,
                    "phase": phase,
                    "phase_step": phase_step,
                    "simulation_time_s": step / 240.0,
                    "timestamp_s": 10.0 + len(rows),
                    "rgb_filename": (
                        f"frames/{view_id}/frame_{sample_index:06d}.png"
                    ),
                }
            )
    return rows


def test_240_hz_physics_has_explicit_30_hz_step_decimation():
    sampling = _module().validate_sampling_rates(240, 30)
    assert sampling == {
        "capture_rate_hz": 30,
        "physics_rate_hz": 240,
        "physics_steps_per_frame": 8,
        "sampling_kind": "fixed_integer_physics_step_decimation",
    }
    with pytest.raises(ValueError):
        _module().validate_sampling_rates(240, 29)


def test_output_directory_rejects_stale_mixed_run_files(tmp_path):
    module = _module()
    empty = tmp_path / "new"
    assert module.prepare_empty_output_directory(empty) == empty.resolve()
    (empty / "stale.txt").write_text("old run", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        module.prepare_empty_output_directory(empty)


def test_camera_rig_covers_front_and_rear_with_fixed_priority():
    module = _module()
    rig = module.build_camera_rig_contract(
        module.DEFAULT_RESOLUTION,
        module.DEFAULT_CAMERA_EYE_M,
        module.DEFAULT_CAMERA_TARGET_M,
    )
    assert rig["view_priority"] == [
        "rear_left",
        "rear_right",
        "front_left",
        "front_right",
    ]
    rear_left, rear_right, front_left, front_right = rig["views"]
    assert rear_left["eye_m"] == [0.30, -0.46, 0.39]
    assert rear_right["eye_m"] == pytest.approx([0.80, -0.46, 0.39])
    assert front_left["eye_m"] == pytest.approx([0.30, 0.83, 0.39])
    assert front_right["eye_m"] == pytest.approx([0.80, 0.83, 0.39])
    assert all(
        view["target_m"] == rear_left["target_m"]
        for view in rig["views"]
    )
    assert rig["same_completed_physics_step"] is True


def test_capture_builder_exactly_matches_analysis_camera_contract():
    module = _module()
    from kcg_connector import d38999_tooth_sync_analysis as analysis

    rig = module.build_camera_rig_contract(
        module.DEFAULT_RESOLUTION,
        module.DEFAULT_CAMERA_EYE_M,
        module.DEFAULT_CAMERA_TARGET_M,
    )
    # This is deliberately exact: the evidence loader hashes and verifies the
    # authored camera contract instead of accepting a nearby camera silently.
    assert rig == analysis.EXPECTED_CAMERA_RIG


def test_capture_schedule_is_phase_local_and_includes_each_phase_start():
    should = _module().should_capture_phase_step
    assert should(
        "nut_only_final_hold", 1, physics_steps_per_frame=8
    )
    assert not should(
        "nut_only_final_hold", 7, physics_steps_per_frame=8
    )
    assert should(
        "nut_only_final_hold", 8, physics_steps_per_frame=8
    )
    assert should(
        "q7_twist_probe_motion", 1, physics_steps_per_frame=8
    )
    assert should(
        "q7_twist_probe_hold", 1, physics_steps_per_frame=8
    )
    assert not should("mixed_preload", 8, physics_steps_per_frame=8)


def test_sync_validation_requires_all_three_ordered_evidence_phases():
    module = _module()
    sampling = module.validate_sampling_rates(240, 30)
    rows = _dual_rows(
        module,
        (
            (3481, "nut_only_final_hold", 1),
            (4451, "q7_twist_probe_motion", 1),
            (5411, "q7_twist_probe_hold", 1),
        ),
    )
    report = module.validate_sync_rows(rows, sampling)
    assert report["frame_count"] == 3 * len(module.VIEW_IDS)
    assert report["sample_count"] == 3
    assert report["phases"] == list(module.CAPTURE_PHASES)
    with pytest.raises(ValueError, match="missing phases"):
        module.validate_sync_rows(rows[: len(module.VIEW_IDS)], sampling)


def test_complete_phase_schedule_detects_a_dropped_sample():
    module = _module()
    sampling = module.validate_sampling_rates(240, 30)
    samples = []
    for phase_index, phase in enumerate(module.CAPTURE_PHASES):
        for phase_step in module.expected_sampled_phase_steps(16, 8):
            samples.append(
                (phase_index * 100 + phase_step, phase, phase_step)
            )
    rows = _dual_rows(module, samples)
    totals = {phase: 16 for phase in module.CAPTURE_PHASES}
    module.validate_sync_rows(rows, sampling, phase_step_totals=totals)
    start = len(module.VIEW_IDS)
    del rows[start : start + len(module.VIEW_IDS)]  # noqa: E203
    for index, row in enumerate(rows):
        row["frame_index"] = index
        row["sample_index"] = index // len(module.VIEW_IDS)
    with pytest.raises(ValueError, match="schedule is incomplete"):
        module.validate_sync_rows(rows, sampling, phase_step_totals=totals)


def test_manifest_binds_sync_and_every_rgb_frame(tmp_path):
    module = _module()
    sampling = module.validate_sampling_rates(240, 30)
    rows = _dual_rows(
        module,
        (
            (3481, "nut_only_final_hold", 1),
            (4451, "q7_twist_probe_motion", 1),
            (5411, "q7_twist_probe_hold", 1),
        ),
    )
    for row in rows:
        row["phase_step"] = 1
        path = tmp_path / row["rgb_filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frame {row['frame_index']}".encode())
    sync_path = tmp_path / "video_frame_sync.csv"
    with sync_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=module.SYNC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.csv"
    report_path.write_text("{}\n", encoding="utf-8")
    summary_path.write_text("global_step,phase\n", encoding="utf-8")
    physics_evidence = {
        "report_path": str(report_path),
        "report_sha256": module.sha256_file(report_path),
        "summary_path": str(summary_path),
        "summary_sha256": module.sha256_file(summary_path),
    }
    manifest = module.build_hash_manifest(
        output_directory=tmp_path,
        sync_rows=rows,
        sampling=sampling,
        camera_rig=module.build_camera_rig_contract(
            module.DEFAULT_RESOLUTION,
            module.DEFAULT_CAMERA_EYE_M,
            module.DEFAULT_CAMERA_TARGET_M,
        ),
        render_settings={"renderer": "RTX"},
        cleanup={"resources_released": True},
        physics_evidence=physics_evidence,
        capture_source={
            "path": str(CAPTURE_PATH),
            "sha256_at_import": module.sha256_file(CAPTURE_PATH),
            "sha256_at_start": module.sha256_file(CAPTURE_PATH),
            "sha256_at_finalize": module.sha256_file(CAPTURE_PATH),
            "unchanged_during_capture": True,
        },
    )
    assert manifest["sync_csv_sha256"] == module.sha256_file(sync_path)
    assert manifest["capture_source"]["sha256_at_start"] == module.sha256_file(
        CAPTURE_PATH
    )
    assert set(manifest["frame_files_sha256"]) == {
        row["rgb_filename"] for row in rows
    }
    assert manifest["physics_evidence"] == physics_evidence


def test_runner_wiring_is_opt_in_fixed_camera_and_after_step():
    source = REGRASP_PATH.read_text(encoding="utf-8")
    assert "--nut-tooth-sync-capture-output" in source
    assert "D38999ToothSyncCapture(" in source
    assert "tooth_sync_capture.maybe_capture(" in source
    assert "tooth_sync_capture.finalize(" in source
    assert "--gui requires" not in source
    # Existing physical safety/drive boundaries remain explicit.
    assert '"object_pose_writes_after_start": 0' in source
    assert 'world.step(render=arguments.gui)' in source
    assert source.index("world.step(render=arguments.gui)") < source.index(
        "tooth_sync_capture.maybe_capture("
    )


def test_sync_capture_cli_is_strict_and_forces_colour_ids(tmp_path):
    runner = _runner()
    common = [
        "--gui",
        "--twist-probe",
        "--nut-tooth-jitter-output",
        str(tmp_path / "physics"),
        "--nut-tooth-sync-capture-output",
        str(tmp_path / "frames"),
    ]
    arguments = runner._parse_arguments(PACKAGE_ROOT.parents[1], common)
    assert arguments.nut_tooth_sync_capture_output == str(
        tmp_path / "frames"
    )
    assert arguments.nut_tooth_jitter_colorize is True
    for missing in ("--gui", "--twist-probe", "--nut-tooth-jitter-output"):
        candidate = list(common)
        index = candidate.index(missing)
        del candidate[index]
        if missing == "--nut-tooth-jitter-output":
            del candidate[index]
        with pytest.raises(SystemExit):
            runner._parse_arguments(PACKAGE_ROOT.parents[1], candidate)


def test_manifest_schema_is_json_serializable():
    module = _module()
    document = {
        "schema_version": module.SCHEMA_VERSION,
        "capture_phases": list(module.CAPTURE_PHASES),
        "sync_fields": list(module.SYNC_FIELDS),
        "view_ids": list(module.VIEW_IDS),
    }
    assert json.loads(json.dumps(document)) == document
