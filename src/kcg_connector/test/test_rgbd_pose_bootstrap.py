from copy import deepcopy
import ast
from pathlib import Path

import numpy as np
import pytest
import yaml

from kcg_connector.rgbd_pose_bootstrap import (
    DEFAULT_RGBD_BOOTSTRAP_CONFIG_PATH,
    RGBD_BOOTSTRAP_SCHEMA_VERSION,
    intersect_camera_ray_with_horizontal_plane,
    load_rgbd_bootstrap,
    robust_semantic_mask_center_uv,
    robust_world_xy_centroid,
    semantic_ids_for_label,
    summarize_mask_depth,
)


MODULE_PATH = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "rgbd_pose_bootstrap.py"
)


def _document():
    return yaml.safe_load(
        DEFAULT_RGBD_BOOTSTRAP_CONFIG_PATH.read_text(encoding="utf-8")
    )


def test_module_has_no_isaac_ros_or_torch_imports():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint({"isaacsim", "omni", "pxr", "rclpy", "torch"})


def test_shipped_contract_loads_with_explicit_non_detector_scope():
    contract = load_rgbd_bootstrap()
    assert contract.schema_version == RGBD_BOOTSTRAP_SCHEMA_VERSION
    assert contract.camera.resolution == (640, 480)
    assert contract.labels.loose_plug != contract.labels.fixed_receptacle
    assert (
        contract.position_estimator.kind
        == "ray_plane_registered_model_height"
    )
    assert contract.output.report_filename == "report.json"


def test_registered_heights_are_derived_from_versioned_model_geometry():
    contract = load_rgbd_bootstrap()
    package_root = DEFAULT_RGBD_BOOTSTRAP_CONFIG_PATH.parents[1]
    tabletop = yaml.safe_load(
        (package_root / contract.tabletop_config.split("/", 2)[-1])
        .read_text(encoding="utf-8")
    )
    proxy = yaml.safe_load(
        (package_root / "config/d38999_shell25j_proxy_v1.yaml")
        .read_text(encoding="utf-8")
    )
    table = tabletop["table"]
    loose = tabletop["loose_endpoint"]
    fixed = tabletop["fixed_endpoint"]
    geometry = proxy["proxy_geometry_m"]
    table_top_z = table["center_m"][2] + table["size_m"][2] / 2.0
    expected_loose_z = (
        table_top_z
        - loose["body_bottom_offset_m"]
        + geometry["plug"]["overall_length"] / 2.0
    )
    expected_fixed_z = fixed["receptacle_origin_m"][2] + (
        geometry["receptacle"]["front_shell_length"]
        - geometry["receptacle"]["rear_body_length"]
    ) / 2.0
    assert (
        contract.position_estimator.loose_plug_registered_model_height_m
        == pytest.approx(expected_loose_z)
    )
    configured_fixed_z = (
        contract.position_estimator.
        fixed_receptacle_registered_model_height_m
    )
    assert configured_fixed_z == pytest.approx(expected_fixed_z)


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "camera",
        "labels",
        "position_estimator",
        "acceptance",
        "output",
    ),
)
def test_top_level_schema_is_exact(tmp_path, field):
    document = _document()
    del document[field]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_rgbd_bootstrap(path)


def test_rejects_boolean_and_nonfinite_numeric_values(tmp_path):
    for value in (True, float("nan"), float("inf")):
        document = deepcopy(_document())
        document["acceptance"]["maximum_xy_centroid_error_m"] = value
        path = tmp_path / f"bad_{str(value)}.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(ValueError, match="finite number|finite"):
            load_rgbd_bootstrap(path)


def test_semantic_id_lookup_handles_json_string_and_integer_keys():
    mapping = {
        "3": {"class": "loose"},
        8: {"class": "loose"},
        "9": {"class": "fixed"},
    }
    assert semantic_ids_for_label(mapping, "class", "loose") == (3, 8)
    assert semantic_ids_for_label(mapping, "class", "fixed") == (9,)


def test_semantic_id_lookup_fails_when_label_is_absent():
    with pytest.raises(ValueError, match="not present"):
        semantic_ids_for_label(
            {"0": {"class": "background"}}, "class", "loose"
        )


def test_mask_depth_statistics_and_mask_are_exact():
    semantic = np.asarray([[0, 3], [3, 9]], dtype=np.uint32)
    depth = np.asarray([[1.0, 0.5], [0.7, 0.9]], dtype=np.float32)
    statistics, mask = summarize_mask_depth(semantic, depth, (3,))
    assert statistics.pixel_count == 2
    assert statistics.visible_fraction == 0.5
    assert statistics.valid_depth_count == 2
    assert statistics.minimum_depth_m == pytest.approx(0.5)
    assert statistics.median_depth_m == pytest.approx(0.6)
    assert mask.tolist() == [[False, True], [True, False]]


@pytest.mark.parametrize(
    ("semantic", "depth"),
    (
        (np.zeros((2, 2)), np.zeros((3, 2))),
        (np.zeros((2, 2, 1)), np.zeros((2, 2, 1))),
    ),
)
def test_mask_depth_rejects_shape_mismatch(semantic, depth):
    with pytest.raises(ValueError, match="equal 2D"):
        summarize_mask_depth(semantic, depth, (0,))


def test_mask_depth_rejects_empty_or_invalid_depth_mask():
    semantic = np.asarray([[0, 1]], dtype=np.uint32)
    with pytest.raises(ValueError, match="empty"):
        summarize_mask_depth(semantic, np.ones((1, 2)), (2,))
    with pytest.raises(ValueError, match="valid positive"):
        summarize_mask_depth(
            semantic, np.asarray([[1.0, float("inf")]]), (1,)
        )


def test_world_xy_centroid_is_robust_median_and_fail_closed():
    points = np.asarray(
        [[0.50, -0.20, 0.22], [0.52, -0.21, 0.25], [9.0, 8.0, 0.3]]
    )
    assert robust_world_xy_centroid(points) == (0.52, -0.2)
    with pytest.raises(ValueError, match="finite"):
        robust_world_xy_centroid([[0.0, float("nan"), 0.0]])


def test_semantic_mask_center_is_coordinatewise_median_and_strict():
    mask = np.zeros((7, 8), dtype=bool)
    mask[2:5, 3:6] = True
    mask[0, 7] = True
    assert robust_semantic_mask_center_uv(mask) == (4.0, 3.0)
    with pytest.raises(ValueError, match="2D boolean"):
        robust_semantic_mask_center_uv(mask.astype(np.uint8))
    with pytest.raises(ValueError, match="empty"):
        robust_semantic_mask_center_uv(np.zeros((2, 2), dtype=bool))


def test_registered_height_ray_intersection_is_finite_and_forward():
    intersection = intersect_camera_ray_with_horizontal_plane(
        [0.0, 0.0, 1.0], [1.0, 2.0, 0.0], 0.5
    )
    assert intersection == pytest.approx((0.5, 1.0, 0.5))
    with pytest.raises(ValueError, match="finite three-vector"):
        intersect_camera_ray_with_horizontal_plane(
            [0.0, 0.0, float("nan")], [1.0, 2.0, 0.0], 0.5
        )
    with pytest.raises(ValueError, match="parallel"):
        intersect_camera_ray_with_horizontal_plane(
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], 0.5
        )
    with pytest.raises(ValueError, match="in front"):
        intersect_camera_ray_with_horizontal_plane(
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -1.0
        )
