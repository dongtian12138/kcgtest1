"""CPU-only contracts for the opt-in two-view axial tooth supplement."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
AXIAL_PATH = PACKAGE_ROOT / "isaac/d38999_tooth_axial_capture.py"
WRAPPER_PATH = PACKAGE_ROOT / "isaac/d38999_nut_regrasp_axial_smoke.py"
BASE_PATH = PACKAGE_ROOT / "isaac/d38999_tooth_sync_capture.py"
RUNNER_PATH = PACKAGE_ROOT / "isaac/d38999_nut_regrasp_smoke.py"
GHOST_PATH = PACKAGE_ROOT / "isaac/d38999_tooth_ghost_runtime.py"


def _load(path, name):
    isaac = str(path.parent)
    sys.path.insert(0, isaac)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(isaac)


def _axial_module():
    return _load(AXIAL_PATH, "d38999_tooth_axial_capture_test")


def _wrapper_module():
    return _load(WRAPPER_PATH, "d38999_nut_regrasp_axial_smoke_test")


def _rows(module, samples):
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
                    "timestamp_s": 100.0 + len(rows),
                    "rgb_filename": (
                        f"frames/{view_id}/frame_{sample_index:06d}.png"
                    ),
                }
            )
    return rows


def test_targeted_axial_geometry_exposes_both_missing_teeth():
    module = _axial_module()
    rig = module.build_axial_camera_rig_contract()
    assert rig["view_priority"] == ["axial_segment13", "axial_segment23"]
    assert [view["target_segment"] for view in rig["views"]] == [
        "Segment_13",
        "Segment_23",
    ]
    for view, expected_azimuth in zip(rig["views"], (225.0, 315.0)):
        exposure = view["analytic_target_exposure"]
        assert exposure["camera_azimuth_degrees"] == pytest.approx(
            expected_azimuth
        )
        assert exposure["camera_elevation_degrees"] == pytest.approx(
            50.6909191
        )
        assert exposure["axial_top_face_cosine"] > 0.77
        assert exposure["radial_outer_face_cosine"] > 0.54


def test_axial_top_exposure_is_over_four_times_original_oblique():
    module = _axial_module()
    axial = module.build_axial_camera_rig_contract()["views"][0]
    target = module.DEFAULT_CAMERA_TARGET_M
    original_eye = (0.30, -0.46, 0.39)
    distance = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(original_eye, target))
    )
    original_top_cosine = (original_eye[2] - target[2]) / distance
    axial_top_cosine = axial["analytic_target_exposure"][
        "axial_top_face_cosine"
    ]
    assert axial_top_cosine / original_top_cosine > 4.0


def test_two_views_share_exact_base_phase_schedule():
    module = _axial_module()
    sampling = module._base_capture.validate_sampling_rates(240, 30)
    samples = []
    for phase_index, phase in enumerate(module.CAPTURE_PHASES):
        for phase_step in module._base_capture.expected_sampled_phase_steps(
            16, 8
        ):
            samples.append(
                (1000 * phase_index + phase_step, phase, phase_step)
            )
    rows = _rows(module, samples)
    result = module.validate_sync_rows(
        rows,
        sampling,
        phase_step_totals={phase: 16 for phase in module.CAPTURE_PHASES},
    )
    assert result["sample_count"] == 3 * 3
    assert result["frame_count"] == 2 * result["sample_count"]
    del rows[2:4]
    for index, row in enumerate(rows):
        row["frame_index"] = index
        row["sample_index"] = index // 2
    with pytest.raises(ValueError, match="schedule is incomplete"):
        module.validate_sync_rows(
            rows,
            sampling,
            phase_step_totals={phase: 16 for phase in module.CAPTURE_PHASES},
        )


def test_wrapper_requires_baseline_ghost_capture_and_strips_only_new_flag(
    tmp_path,
):
    wrapper = _wrapper_module()
    base = [
        "--gui",
        "--twist-probe",
        "--nut-tooth-jitter-output",
        str(tmp_path / "physics"),
        "--nut-tooth-sync-capture-output",
        str(tmp_path / "base"),
        "--nut-tooth-ghost-fingers-output",
        str(tmp_path / "ghost"),
    ]
    output, remaining = wrapper.split_axial_arguments(
        [
            *base,
            wrapper.AXIAL_OUTPUT_FLAG,
            str(tmp_path / "axial"),
        ]
    )
    assert output == str(tmp_path / "axial")
    assert remaining == base
    with pytest.raises(ValueError, match="requires the baseline"):
        wrapper.split_axial_arguments(
            [
                *base[:-2],
                wrapper.AXIAL_OUTPUT_FLAG,
                str(tmp_path / "bad"),
            ]
        )
    with pytest.raises(ValueError, match="baseline prepared twist"):
        wrapper.split_axial_arguments(
            [
                *base,
                "--rewind-probe",
                wrapper.AXIAL_OUTPUT_FLAG,
                str(tmp_path / "bad"),
            ]
        )


def test_extension_has_no_physics_step_or_pose_write_and_preserves_sources():
    module = _axial_module()
    source = AXIAL_PATH.read_text(encoding="utf-8")
    assert "world.step(" not in source
    assert "set_world_pose(" not in source
    assert "SetWorldPose(" not in source
    assert '"object_pose_writes": 0' in source
    assert '"physics_steps": 0' in source
    assert module.sha256_file(RUNNER_PATH) == (
        "f8a77d1b56982b7225a5b3e2858bcd6b4fe139c5487915e6e79fbab98311e13b"
    )
    assert module.sha256_file(BASE_PATH) == (
        "bd7d9d15b5b745d9585d6c9cd0ec104fd532d173eebbf9c32251a2c58aa1a577"
    )
    assert module.sha256_file(GHOST_PATH) == (
        "89fd99343692204a875d5fc66db72a87d3bb3ab0188530413f77166a049dc57f"
    )


def test_wrapper_injects_adapter_before_importing_unchanged_runner():
    source = WRAPPER_PATH.read_text(encoding="utf-8")
    swap = 'sys.modules["d38999_tooth_sync_capture"] = axial'
    runner_import = "import d38999_nut_regrasp_smoke as runner"
    assert swap in source
    assert runner_import in source
    assert source.index(swap) < source.index(runner_import)


def test_wrapper_calls_no_argument_runner_with_temporary_sys_argv(
    tmp_path, monkeypatch
):
    wrapper = _wrapper_module()
    original_argv = sys.argv
    observed = {}

    def configure_axial_extension(**kwargs):
        observed["configuration"] = kwargs

    def finalize_axial_ghost_bundle(**kwargs):
        observed["bundle"] = kwargs

    fake_axial = SimpleNamespace(
        configure_axial_extension=configure_axial_extension,
        finalize_axial_ghost_bundle=finalize_axial_ghost_bundle,
    )

    def runner_main():
        observed["argv"] = list(sys.argv)
        observed["capture_module"] = sys.modules[
            "d38999_tooth_sync_capture"
        ]
        return 0

    fake_runner = SimpleNamespace(main=runner_main)
    monkeypatch.setitem(sys.modules, "d38999_tooth_axial_capture", fake_axial)
    monkeypatch.setitem(sys.modules, "d38999_nut_regrasp_smoke", fake_runner)
    base = [
        "--gui",
        "--twist-probe",
        "--nut-tooth-jitter-output",
        str(tmp_path / "physics"),
        "--nut-tooth-sync-capture-output",
        str(tmp_path / "base"),
        "--nut-tooth-ghost-fingers-output",
        str(tmp_path / "ghost"),
    ]
    axial_output = str(tmp_path / "axial")
    assert wrapper.main(
        [
            *base,
            wrapper.AXIAL_OUTPUT_FLAG,
            axial_output,
        ]
    ) == 0
    assert observed["argv"][1:] == base
    assert observed["capture_module"] is fake_axial
    assert observed["bundle"] == {
        "axial_output": axial_output,
        "ghost_output": str(tmp_path / "ghost"),
    }
    assert sys.argv is original_argv


def test_post_run_bundle_binds_same_base_runner_and_ghost_lifecycle(tmp_path):
    module = _axial_module()
    axial_root = tmp_path / "axial"
    ghost_root = tmp_path / "ghost"
    base_root = tmp_path / "base"
    axial_root.mkdir()
    ghost_root.mkdir()
    base_root.mkdir()
    base_manifest = base_root / "video_capture_manifest.json"
    base_manifest.write_text('{"passed":true}\n', encoding="utf-8")
    runner_binding = module.file_binding(RUNNER_PATH)
    axial_manifest = axial_root / module.AXIAL_MANIFEST_NAME
    axial_manifest.write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "passed": True,
                "base_four_view_binding": module.file_binding(base_manifest),
                "provenance": {"prepared_runner": runner_binding},
            }
        ),
        encoding="utf-8",
    )
    sidecar = ghost_root / "visibility_sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "active": False,
                "passed": True,
                "cleanup": {
                    "effective_visibility_restored_to_pre_author_state": True,
                    "session_visibility_opinions_removed": True,
                },
                "mutation_audit": {
                    "collision_api_writes": 0,
                    "material_writes": 0,
                    "object_pose_writes": 0,
                    "physics_api_writes": 0,
                    "xform_writes": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    ghost_manifest = ghost_root / "manifest.json"
    ghost_manifest.write_text(
        json.dumps(
            {
                "schema_version": "kcg_d38999_tooth_ghost_manifest_v1",
                "inputs": {
                    "capture_manifest": module.file_binding(base_manifest)
                },
                "outputs": {
                    "visibility_sidecar": module.file_binding(sidecar)
                },
                "sources": {"prepared_tooth_runner": runner_binding},
            }
        ),
        encoding="utf-8",
    )
    bundle = module.finalize_axial_ghost_bundle(
        axial_output=axial_root, ghost_output=ghost_root
    )
    assert bundle["passed"] is True
    assert bundle["same_base_capture"] is True
    assert bundle["visibility_only_zero_physics_or_pose_writes"] is True
    sidecar.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="size/SHA"):
        module.finalize_axial_ghost_bundle(
            axial_output=axial_root, ghost_output=ghost_root
        )
