from pathlib import Path

import numpy as np
import pytest

import te_plug_five_dof_geometry as geometry


def scene(monkeypatch, *, occlusion="none", radius=.015):
    monkeypatch.setattr(geometry, "_visible_face_geometry", lambda _: (.015, .03))
    y, x = np.indices((512, 512))
    K = np.array([[1000., 0., 256.], [0., 1000., 256.], [0., 0., 1.]])
    xx = (x + .5 - 256.) * .1 / 1000.
    yy = (y + .5 - 256.) * .1 / 1000.
    face = xx*xx + yy*yy < radius*radius
    depth = np.full(face.shape, .2)
    depth[face] = .1
    if occlusion == "partial":
        depth[(x > 330) & (y < 215)] = .08
        depth[(x < 180) & (y < 220)] = .08
    elif occlusion == "narrow":
        depth[x < 330] = .08
    mask = face & (depth == .1)
    return depth, mask, K


def estimate(depth, mask, K):
    return geometry.estimate_plug_rear_circle_from_float_depth(
        depth_m=depth, mask=mask, intrinsics=K, mesh_path=Path("synthetic-circle.obj"),
        pixel_center_offset_px=.5, plane_iterations=128)


def test_complete_circle_retains_original_fit(monkeypatch):
    depth, mask, K = scene(monkeypatch)
    original_K = K.copy()
    result = estimate(depth, mask, K)
    np.testing.assert_allclose(result["camera_from_object"][:3, 3], [0, 0, .13], atol=2e-5)
    assert not result["metrics"]["foreground_occlusion_boundary_filtered"]
    np.testing.assert_array_equal(K, original_K)


def test_measured_foreground_edges_do_not_shift_circle_center(monkeypatch):
    depth, mask, K = scene(monkeypatch, occlusion="partial")
    result = estimate(depth, mask, K)
    metrics = result["metrics"]
    assert metrics["foreground_occlusion_boundary_filtered"]
    assert metrics["original_seed_contour_rms_m"] > metrics["circle_quality_limit_m"]
    assert metrics["circle_residual_rms_m"] <= metrics["circle_quality_limit_m"]
    assert metrics["visible_rim_support"]["occupied_ten_degree_bins"] >= 18
    np.testing.assert_allclose(result["camera_from_object"][:3, 3], [0, 0, .13], atol=4e-5)
    assert metrics["axial_yaw_estimated"] is False


def test_too_little_visible_rim_is_rejected(monkeypatch):
    depth, mask, K = scene(monkeypatch, occlusion="narrow")
    with pytest.raises(RuntimeError, match="sufficiently constrain"):
        estimate(depth, mask, K)


def test_wrong_radius_without_foreground_is_not_excused(monkeypatch):
    depth, mask, K = scene(monkeypatch, radius=.013)
    with pytest.raises(RuntimeError, match="foreground occlusion"):
        estimate(depth, mask, K)
