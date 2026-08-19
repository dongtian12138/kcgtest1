from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.d38999_cad_registration import CameraModel
from kcg_connector.d38999_multilayer_rgbd_archive import (
    FROZEN_SOURCES,
    build_multilayer_rgbd_archive_contract,
    write_offline_multilayer_rgbd_archive,
)
from kcg_connector.postgrasp_shadow_estimator import (
    FormalArchiveError,
    FormalView,
    load_formal_archive,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _view(view_id="V0", timestamp="2026-08-17T20:00:00Z"):
    width, height = 8, 6
    camera = CameraModel(
        width=width,
        height=height,
        fx=10.0,
        fy=10.0,
        cx=3.5,
        cy=2.5,
        position_world=(0.0, 0.0, 0.0),
        world_to_camera=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    depth = np.full((height, width), 0.5, dtype=np.float32)
    depth[0, 0] = np.inf
    return FormalView(
        view_id=view_id,
        timestamp_utc=timestamp,
        rgb=np.full((height, width, 3), 127, dtype=np.uint8),
        depth=depth,
        camera=camera,
        T_WH=np.eye(4),
        T_WC=np.eye(4),
        T_HC=np.eye(4),
    )


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_current_archive_contract_is_traceable_and_offline_only():
    contract = build_multilayer_rgbd_archive_contract(REPOSITORY_ROOT)
    assert contract["status"] == "OFFLINE_PASS"
    assert len(contract["sources"]) == 7
    assert contract["current_readiness"] == {
        "offline_fixture_archive_allowed": True,
        "dynamic_capture_archives_available": 0,
        "dynamic_archive_write_authorized": False,
        "formal_dynamic_rgbd_pass_claimed": False,
    }


def test_camera_and_multilayer_targets_are_bound_without_dynamic_promotion():
    contract = build_multilayer_rgbd_archive_contract(REPOSITORY_ROOT)
    assert contract["observation_target"]["representation"] == (
        "D38999_VISUAL_COMPLETE_V1"
    )
    assert contract["camera_interfaces"]["Palm"]["dynamic_capture_passed"] is False
    assert contract["camera_interfaces"]["Wrist"]["dynamic_capture_passed"] is False


def test_archive_writes_exact_files_digests_and_roundtrips(tmp_path):
    output = tmp_path / "archive"
    provenance = write_offline_multilayer_rgbd_archive(
        REPOSITORY_ROOT,
        output,
        [_view("V0"), _view("V1", "2026-08-17T20:00:01+00:00")],
        fixture_id="unit_fixture_01",
    )
    expected = {
        "formal_manifest.json",
        "archive_provenance.json",
        "V0/rgb.png",
        "V0/depth.npy",
        "V0/camera.json",
        "V0/fk.json",
        "V1/rgb.png",
        "V1/depth.npy",
        "V1/camera.json",
        "V1/fk.json",
    }
    actual = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    for record in provenance["files"]:
        assert hashlib.sha256((output / record["path"]).read_bytes()).hexdigest() == (
            record["sha256"]
        )
    loaded = load_formal_archive(output)
    assert [view.view_id for view in loaded] == ["V0", "V1"]
    assert np.isinf(loaded[0].depth[0, 0])


def test_existing_output_is_rejected_without_modification(tmp_path):
    output = tmp_path / "archive"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        write_offline_multilayer_rgbd_archive(
            REPOSITORY_ROOT, output, [_view()], fixture_id="existing"
        )
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("view_id", ["../escape", "/absolute", "truth_restore"])
def test_unsafe_or_reserved_view_id_is_rejected_before_write(tmp_path, view_id):
    output = tmp_path / "archive"
    with pytest.raises(FormalArchiveError, match="unsafe or reserved"):
        write_offline_multilayer_rgbd_archive(
            REPOSITORY_ROOT,
            output,
            [_view(view_id)],
            fixture_id="bad_view",
        )
    assert not output.exists()


def test_duplicate_view_ids_are_rejected_before_write(tmp_path):
    output = tmp_path / "archive"
    with pytest.raises(FormalArchiveError, match="unique"):
        write_offline_multilayer_rgbd_archive(
            REPOSITORY_ROOT,
            output,
            [_view("V0"), _view("V0")],
            fixture_id="duplicate",
        )
    assert not output.exists()


def test_timestamp_must_be_timezone_aware(tmp_path):
    with pytest.raises(FormalArchiveError, match="timezone-aware"):
        write_offline_multilayer_rgbd_archive(
            REPOSITORY_ROOT,
            tmp_path / "archive",
            [_view(timestamp="2026-08-17T20:00:00")],
            fixture_id="bad_time",
        )


def test_invalid_depth_or_transform_is_rejected_before_write(tmp_path):
    bad_depth = _view()
    bad_depth.depth[0, 1] = np.nan
    with pytest.raises(FormalArchiveError, match="invalid offline RGB or depth"):
        write_offline_multilayer_rgbd_archive(
            REPOSITORY_ROOT,
            tmp_path / "depth",
            [bad_depth],
            fixture_id="bad_depth",
        )
    bad_transform = replace(_view(), T_WH=np.full((4, 4), np.nan))
    with pytest.raises(FormalArchiveError, match="T_WH"):
        write_offline_multilayer_rgbd_archive(
            REPOSITORY_ROOT,
            tmp_path / "transform",
            [bad_transform],
            fixture_id="bad_transform",
        )


def test_formal_archive_and_provenance_exclude_truth_fields(tmp_path):
    output = tmp_path / "archive"
    provenance = write_offline_multilayer_rgbd_archive(
        REPOSITORY_ROOT, output, [_view()], fixture_id="firewall"
    )
    manifest = json.loads((output / "formal_manifest.json").read_text())
    assert {"semantic", "object_truth", "contact_report"} <= set(
        manifest["forbidden_inputs"]
    )
    assert provenance["semantic_input_used"] is False
    assert provenance["object_pose_truth_used"] is False
    assert provenance["contact_truth_used"] is False
    assert provenance["dynamic_capture_claimed"] is False


def test_frozen_source_tamper_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    source = root / "src/kcg_connector/kcg_connector/postgrasp_shadow_estimator.py"
    source.write_text(source.read_text() + "\n# tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_rgbd_archive_contract(root)


def test_public_write_api_has_no_truth_or_dynamic_claim_inputs():
    names = set(inspect.signature(write_offline_multilayer_rgbd_archive).parameters)
    assert names == {"repository_root", "output_dir", "views", "fixture_id"}
    assert names.isdisjoint(
        {
            "semantic",
            "object_pose",
            "contact_report",
            "contact_name",
            "contact_normal",
            "dynamic_pass",
            "formal_pass",
        }
    )
