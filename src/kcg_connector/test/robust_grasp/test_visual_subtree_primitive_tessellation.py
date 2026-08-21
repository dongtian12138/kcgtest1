from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[4]
SOURCE = (
    REPOSITORY
    / "src/kcg_connector/isaac/robust_grasp/export_visual_subtree_mesh.py"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "carts_visual_subtree_export", SOURCE
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _assert_outward(points: np.ndarray, faces: np.ndarray) -> None:
    triangles = points[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    centroids = np.mean(triangles, axis=1)
    non_axial = np.linalg.norm(centroids, axis=1) > np.finfo(np.float64).eps
    assert np.all(np.einsum("ij,ij->i", normals, centroids)[non_axial] > 0.0)


def test_circle_resolution_is_derived_and_converges_from_inside() -> None:
    module = _module()
    error = 0.001
    base = module._regular_polygon_edges(error, 1)
    assert 1.0 - np.cos(np.pi / base) <= error
    assert 1.0 - np.cos(np.pi / (base - 1)) > error
    assert module._regular_polygon_edges(error, 2) == 2 * base
    assert module._regular_polygon_edges(error, 4) == 4 * base


def test_cylinder_and_sphere_are_closed_outward_and_deterministic() -> None:
    module = _module()
    edges = module._regular_polygon_edges(0.001, 1)
    cylinder_first = module._cylinder_mesh(
        radius=0.2, height=0.7, axis="Z", edge_count=edges
    )
    cylinder_second = module._cylinder_mesh(
        radius=0.2, height=0.7, axis="Z", edge_count=edges
    )
    np.testing.assert_array_equal(cylinder_first[0], cylinder_second[0])
    np.testing.assert_array_equal(cylinder_first[1], cylinder_second[1])
    _assert_outward(*cylinder_first)

    sphere_first = module._sphere_mesh(radius=0.3, edge_count=edges)
    sphere_second = module._sphere_mesh(radius=0.3, edge_count=edges)
    np.testing.assert_array_equal(sphere_first[0], sphere_second[0])
    np.testing.assert_array_equal(sphere_first[1], sphere_second[1])
    _assert_outward(*sphere_first)


def test_cylinder_axis_maps_are_proper_and_preserve_radius() -> None:
    module = _module()
    edges = module._regular_polygon_edges(0.01, 1)
    for axis, axial_index in (("X", 0), ("Y", 1), ("Z", 2)):
        points, _faces = module._cylinder_mesh(
            radius=0.2, height=0.7, axis=axis, edge_count=edges
        )
        assert np.max(points[:, axial_index]) == 0.35
        assert np.min(points[:, axial_index]) == -0.35
