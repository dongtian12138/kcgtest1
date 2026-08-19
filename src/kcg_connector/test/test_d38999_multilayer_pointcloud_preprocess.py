from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.d38999_multilayer_pointcloud_preprocess import (
    FROZEN_SOURCES,
    SELECTED_BACKEND,
    build_multilayer_pointcloud_preprocess_contract,
    preprocess_rgbd_pointcloud,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _intrinsics(width=2, height=2):
    return {
        "width": width,
        "height": height,
        "fx": 1.0,
        "fy": 1.0,
        "cx": 0.0,
        "cy": 0.0,
    }


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_current_contract_uses_explicit_numpy_equivalent_path():
    contract = build_multilayer_pointcloud_preprocess_contract(REPOSITORY_ROOT)
    assert contract["status"] == "OFFLINE_PASS"
    assert contract["selected_backend"] == SELECTED_BACKEND
    assert contract["dependency_resolution"] == "EQUIVALENT_NUMPY_PATH_SELECTED"
    assert contract["current_readiness"]["dynamic_archives_available"] == 0
    assert contract["dynamic_pointcloud_pass_claimed"] is False
    assert len(contract["sources"]) == 4


def test_pinhole_backprojection_is_in_camera_optical_frame():
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    result = preprocess_rgbd_pointcloud(
        depth,
        rgb,
        _intrinsics(),
        frame_id="PalmCamera_optical",
        voxel_size_m=0.1,
    )
    assert np.allclose(
        result.points_camera_m,
        [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    )
    assert result.summary["point_frame"] == "camera_optical"
    assert result.summary["frame_id"] == "PalmCamera_optical"


def test_invalid_and_out_of_clip_depth_is_filtered_without_mask():
    depth = np.asarray([[0.5, np.inf, np.nan], [0.0, -1.0, 11.0]], dtype=np.float32)
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    result = preprocess_rgbd_pointcloud(
        depth,
        rgb,
        _intrinsics(width=3),
        frame_id="camera",
        voxel_size_m=0.01,
    )
    assert result.summary["valid_depth_point_count"] == 1
    assert result.summary["filtered_depth_pixel_count"] == 5
    assert result.summary["semantic_mask_used"] is False
    assert np.isfinite(result.points_camera_m).all()


def test_voxel_centroid_and_color_mean_are_deterministic():
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.asarray(
        [[[0, 10, 20], [20, 30, 40]], [[40, 50, 60], [60, 70, 80]]],
        dtype=np.uint8,
    )
    first = preprocess_rgbd_pointcloud(
        depth, rgb, _intrinsics(), frame_id="camera", voxel_size_m=10.0
    )
    second = preprocess_rgbd_pointcloud(
        depth.copy(), rgb.copy(), _intrinsics(), frame_id="camera", voxel_size_m=10.0
    )
    assert np.allclose(first.points_camera_m, [[0.5, 0.5, 1.0]])
    assert first.colors_rgb_u8.tolist() == [[30, 40, 50]]
    assert first.summary["points_sha256"] == second.summary["points_sha256"]
    assert first.summary["colors_sha256"] == second.summary["colors_sha256"]


def test_depth_clip_boundaries_are_inclusive():
    depth = np.asarray([[0.02, 10.0]], dtype=np.float32)
    result = preprocess_rgbd_pointcloud(
        depth,
        np.zeros((1, 2, 3), dtype=np.uint8),
        _intrinsics(width=2, height=1),
        frame_id="camera",
        voxel_size_m=0.001,
    )
    assert result.summary["valid_depth_point_count"] == 2


@pytest.mark.parametrize(
    "depth,rgb",
    [
        (np.ones((2, 2)), np.zeros((2, 3, 3), dtype=np.uint8)),
        (np.ones((2, 2)), np.zeros((2, 2, 3), dtype=np.float32)),
    ],
)
def test_rgbd_shape_and_dtype_are_strict(depth, rgb):
    with pytest.raises(ValueError):
        preprocess_rgbd_pointcloud(
            depth, rgb, _intrinsics(), frame_id="camera", voxel_size_m=0.01
        )


def test_intrinsics_keys_and_dimensions_are_strict():
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    bad = _intrinsics()
    bad["truth_pose"] = [0.0] * 7
    with pytest.raises(ValueError, match="keys differ"):
        preprocess_rgbd_pointcloud(
            depth, rgb, bad, frame_id="camera", voxel_size_m=0.01
        )
    bad = _intrinsics(width=3)
    with pytest.raises(ValueError, match="dimensions"):
        preprocess_rgbd_pointcloud(
            depth, rgb, bad, frame_id="camera", voxel_size_m=0.01
        )


def test_no_valid_depth_fails_closed():
    with pytest.raises(ValueError, match="no valid depth"):
        preprocess_rgbd_pointcloud(
            np.full((2, 2), np.inf),
            np.zeros((2, 2, 3), dtype=np.uint8),
            _intrinsics(),
            frame_id="camera",
            voxel_size_m=0.01,
        )


@pytest.mark.parametrize("value", [0.0, -0.1, np.inf, np.nan])
def test_voxel_size_must_be_explicit_finite_positive(value):
    with pytest.raises(ValueError, match="voxel_size_m"):
        preprocess_rgbd_pointcloud(
            np.ones((2, 2), dtype=np.float32),
            np.zeros((2, 2, 3), dtype=np.uint8),
            _intrinsics(),
            frame_id="camera",
            voxel_size_m=value,
        )


def test_summary_is_finite_json_and_never_claims_dynamic_pass():
    result = preprocess_rgbd_pointcloud(
        np.ones((2, 2), dtype=np.float32),
        np.zeros((2, 2, 3), dtype=np.uint8),
        _intrinsics(),
        frame_id="camera",
        voxel_size_m=0.1,
    )
    json.dumps(result.summary, allow_nan=False)
    assert result.summary["object_pose_truth_used"] is False
    assert result.summary["contact_truth_used"] is False
    assert result.summary["dynamic_pointcloud_pass_claimed"] is False


def test_frozen_source_tamper_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    source = root / "src/kcg_connector/config/d38999_master_model_contract_v1.yaml"
    source.write_text(source.read_text() + "\n# tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_pointcloud_preprocess_contract(root)


def test_public_api_has_no_mask_pose_contact_or_backend_switch():
    names = set(inspect.signature(preprocess_rgbd_pointcloud).parameters)
    assert names == {
        "depth_m",
        "rgb",
        "camera_intrinsics",
        "frame_id",
        "voxel_size_m",
    }
    assert names.isdisjoint(
        {
            "mask",
            "semantic_mask",
            "object_pose",
            "contact_report",
            "contact_name",
            "contact_normal",
            "backend",
        }
    )
