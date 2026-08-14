import ast
from pathlib import Path
import runpy
import sys

import pytest
import yaml


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "d38999_rgbd_bootstrap_smoke.py"
)
RUNTIME_PATH = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "isaac_d38999_rgbd_runtime.py"
)
CONFIG_PATH = (
    Path(__file__).parents[1]
    / "config"
    / "d38999_rgbd_bootstrap_v1.yaml"
)


def _source():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _runtime_source():
    return RUNTIME_PATH.read_text(encoding="utf-8")


def _module():
    return runpy.run_path(str(SCRIPT_PATH), run_name="rgbd_smoke_test")


def test_script_import_is_lazy_and_does_not_load_isaac():
    before = set(sys.modules)
    _module()
    imported = set(sys.modules) - before
    assert not any(
        name == "isaacsim" or name.startswith(("isaacsim.", "omni.", "pxr"))
        for name in imported
    )


def test_script_uses_direct_replicator_shared_render_product():
    source = _runtime_source()
    assert "Camera(" in source
    assert "rep.functional.create.camera(" in source
    assert "rep.create.render_product(" in source
    assert source.count("rep.AnnotatorRegistry.get_annotator(") == 3
    assert 'rep.AnnotatorRegistry.get_annotator("rgb")' in source
    assert '"distance_to_image_plane"' in source
    assert '"semantic_segmentation"' in source
    assert '"semanticFilter": f"{rgbd.labels.taxonomy}:*"' in source
    assert "annotator.attach([render_product_path])" in source
    assert '"camera_semantic_wrapper_used": False' in source
    assert '"shared_render_product_for_rgb_depth_semantics": True' in source
    assert "add_semantic_segmentation_to_frame" not in source
    assert "semanticTypes" not in source
    assert "capture_d38999_rgbd_runtime(" in _source()


def test_standalone_can_capture_twice_without_resetting_the_world():
    source = _source()
    assert '"--capture-count"' in source
    assert "for capture_index in range(arguments.capture_count):" in source
    assert 'else output_dir / f"repeat_{capture_index + 1:02d}"' in source
    capture_loop = source.split(
        "for capture_index in range(arguments.capture_count):", 1
    )[1].split("if capture is None", 1)[0]
    assert "capture_d38999_rgbd_runtime(" in capture_loop
    assert "world.reset" not in capture_loop
    assert "world.step" not in capture_loop
    assert "World.clear_instance" not in capture_loop


def test_script_labels_each_endpoint_root_once_without_diagnostics():
    source = _runtime_source()
    assert "Usd.PrimRange(root_prim)" in source
    assert "if prim.IsA(UsdGeom.Gprim)" in source
    assert "get_labels(" in source
    assert "simulation_app.update()" in source
    # Exactly one label is authored per endpoint root.  Descendant renderable
    # geometry inherits it; duplicating the same taxonomy on children can
    # produce a comma-joined Replicator label and defeat strict ID lookup.
    assert "def label_endpoint(root_prim, label):" in source
    assert "authored_labels != [label]" in source
    assert source.count("add_labels(") == 1
    assert source.count("label_endpoint(") == 3
    assert "RgbdSemanticControl" not in source
    assert "rgbd_semantic_control" not in source
    assert "SemanticsAPI" not in source
    assert "rep_functional" not in source
    assert "get_world_points_from_image_coords" in source
    assert "robust_semantic_mask_center_uv" in source
    assert "intersect_camera_ray_with_horizontal_plane" in source
    assert '"uses_registered_truth_xy": False' in source
    assert '"learned_detector_present": False' in source
    assert '"foundation_pose_present": False' in source
    assert '"full_keyed_6d_vision_pose_claimed": False' in source
    assert '"full_rgbd_pose_accepted": False' in _source()
    assert "make_sim_ground_truth_observation" in _source()


def test_script_saves_rgb_depth_semantic_and_json_evidence():
    source = _source() + _runtime_source()
    for field in (
        "rgb_filename",
        "depth_preview_filename",
        "depth_numpy_filename",
        "semantic_preview_filename",
        "report_filename",
    ):
        assert field in source
    assert "allow_nan=False" in source
    assert '"camera_frame_diagnostics"' in source
    assert '"/tmp/kcg_d38999_rgbd_failed_rgb.png"' in source


def test_script_never_writes_endpoint_pose_after_physics_start():
    source = _source() + _runtime_source()
    assert '"object_pose_writes_after_start": 0' in source
    assert ".set_world_pose(" not in source
    assert ".set_local_pose(" not in source


def test_endpoint_projection_gate_records_real_frame_margin():
    project = _module()["_endpoint_projection_records"]
    records = project(
        {
            "loose_plug": (303.0, 353.0),
            "fixed_receptacle": (332.0, 158.0),
        },
        (640, 480),
    )
    assert records["loose_plug"]["in_frame"] is True
    assert records["fixed_receptacle"]["in_frame"] is True
    assert records["loose_plug"]["margin_px"] == 16

    outside = project({"loose_plug": (5.0, 353.0)}, (640, 480))
    assert outside["loose_plug"]["in_frame"] is False
    with pytest.raises(ValueError, match="finite"):
        project({"loose_plug": (float("nan"), 1.0)}, (640, 480))


def test_camera_contract_centers_both_endpoints_and_uses_short_warmup():
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert document["camera"]["eye_m"] == [0.55, -0.85, 0.72]
    assert document["camera"]["target_m"] == [0.535, -0.0125, 0.231]
    assert document["camera"]["resolution"] == [640, 480]
    assert document["camera"]["warmup_frames"] == 5
    estimator = document["position_estimator"]
    assert estimator["kind"] == "ray_plane_registered_model_height"
    assert (
        estimator["mask_center_statistic"]
        == "coordinatewise_median_semantic_mask_pixels"
    )


def test_script_keeps_visible_depth_as_diagnostic_and_gates_ray_estimate():
    source = _runtime_source()
    assert "visible_surface_depth_median_world_xy_m" in source
    assert "visible_surface_depth_median_xy_error_m" in source
    assert "ray_plane_registered_model_height_world_xyz_m" in source
    assert "registered_model_height_source" in source
    assert "ray_parallel_gate_passed" in source
    assert "mask_center_records" in source
    assert "and mask_frame" in source
    assert "intersect_camera_ray_with_horizontal_plane" in source
    assert (
        '"ray_plane_registered_model_height_world_xy_only"' in source
    )


def test_endpoint_semantic_ids_must_be_real_and_rendered():
    validate = _module()["_validate_real_endpoint_semantic_ids"]
    result = validate(
        {"loose_plug": (2,), "fixed_receptacle": (3,)},
        (0, 1, 2, 3),
    )
    assert result == {"loose_plug": (2,), "fixed_receptacle": (3,)}

    with pytest.raises(RuntimeError, match="BACKGROUND/UNLABELLED"):
        validate({"loose_plug": (1,)}, (0, 1))
    with pytest.raises(RuntimeError, match="no rendered pixels"):
        validate({"loose_plug": (2,)}, (0, 1))


def test_only_main_imports_heavy_runtime_dependencies():
    tree = ast.parse(_source())
    top_level_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_roots.update(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_roots.add(node.module.split(".")[0])
    assert top_level_roots.isdisjoint(
        {"isaacsim", "omni", "pxr", "numpy", "PIL", "scipy"}
    )
