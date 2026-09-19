import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("warp", reason="Run these vision geometry tests with the Isaac Python runtime")

from te_nut_phase_surface import SourceTriangleSurface
from te_nut_phase_vision import fit_phase


def test_triangle_interior_and_edge_distance_ignore_unreferenced_points():
    vertices = np.array([[0, 0, 0], [.02, 0, 0], [0, .02, 0],
                         [.003, .004, .0001]])
    surface = SourceTriangleSurface(vertices, [[0, 1, 2]])
    distances, faces = surface.query([[.003, .004, .0001], [.01, -.003, .004]])
    np.testing.assert_allclose(distances, [.0001, .005], atol=2e-9, rtol=0)
    assert faces.tolist() == [0, 0]
    with pytest.raises(ValueError, match="finite"):
        surface.query([[np.nan, 0, 0]])


def test_phase_does_not_follow_the_visible_point_sampling():
    vertices = np.array([[.022, -.004, -.022], [.022, .004, -.022],
                         [.022, .004, -.017], [.022, -.004, -.017]])
    surface = SourceTriangleSurface(vertices, [[0, 1, 2], [0, 2, 3]])
    rotation = Rotation.from_euler('z', 12.3, degrees=True).as_matrix()
    for seed, limits in [(4, (-.0035, .001)), (8, (-.001, .0035))]:
        rng = np.random.default_rng(seed)
        points = np.column_stack((np.full(1800, .022),
                                  rng.uniform(*limits, 1800),
                                  rng.uniform(-.0215, -.0175, 1800))) @ rotation.T
        fit = fit_phase(surface, points, prior_deg=13.)
        assert abs(fit['best']['yaw_deg'] - 12.3) < 1e-8
        assert fit['best']['clipped_rms_m'] < 3e-9
        assert fit['distance_metric'] == surface.method


def test_invalid_source_triangles_are_rejected():
    with pytest.raises(ValueError, match="source triangles"):
        SourceTriangleSurface([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]])
