import ast
import json
import math
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from kcg_connector.d38999_key_shadow_pipeline import (
    run_palm_key_shadow_pipeline,
)


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "d38999_keyed_v2_palm_front_probe.py"
)
ASSET_PATH = (
    Path(__file__).parents[3]
    / "artifacts"
    / "kcg_connector"
    / "isaac"
    / "d38999_shell25j_25_61_n_keyed_public_spec_v2.usda"
)
KEYED_MODEL_ID = "d38999_26kj61sn_keyed_proxy_v2"
KEY_ANGLES_DEG = (0.0, 80.0, 142.0, 196.0, 293.0)


def _source():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _module():
    return runpy.run_path(str(SCRIPT_PATH), run_name="keyed_v2_probe_cpu_test")


def _isolated_depth_observation():
    shape = (241, 241)
    center = (120.0, 120.0)
    rows, columns = np.indices(shape, dtype=np.float64)
    radii = np.hypot(columns - center[0], rows - center[1])
    angles = np.mod(
        np.arctan2(rows - center[1], columns - center[0]),
        2.0 * math.pi,
    )
    depth = np.full(shape, np.inf, dtype=np.float64)
    front_face = radii <= 64.0
    widths_deg = (16.0, 8.0, 8.0, 8.0, 8.0)
    for angle_deg, width_deg in zip(KEY_ANGLES_DEG, widths_deg):
        difference = np.abs(
            np.angle(
                np.exp(1j * (angles - math.radians(angle_deg)))
            )
        )
        front_face |= (
            difference <= 0.5 * math.radians(width_deg)
        ) & (radii <= 76.0)
    depth[front_face] = 0.120
    # A larger rear shell is visible in depth but lies outside the front band.
    depth[(radii > 76.0) & (radii <= 88.0)] = 0.132
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    rgb[front_face] = (180, 120, 70)
    return rgb, depth, front_face, center


def _pipeline_result(module):
    _, depth, _, _ = _isolated_depth_observation()
    inputs = module["derive_isolated_probe_inputs"](depth)
    result = run_palm_key_shadow_pipeline(
        inputs["connector_face_mask"],
        depth,
        inputs["face_center_uv"],
        ((1.0, 0.0), (-1.0, 0.0)),
        KEYED_MODEL_ID,
        occlusion_mask=inputs["occlusion_mask"],
    )
    return depth, inputs, result


def test_script_import_is_lazy_and_does_not_load_isaac_modules():
    before = set(sys.modules)
    module = _module()
    imported = set(sys.modules) - before

    assert module["PROBE_SCOPE"] == "ISOLATED_KEYED_V2_PALM_FRONT_PROBE_ONLY"
    assert not any(
        name == "isaacsim" or name.startswith(("isaacsim.", "omni.", "pxr"))
        for name in imported
    )


def test_probe_targets_only_the_new_public_spec_asset():
    module = _module()

    assert ASSET_PATH.is_file()
    assert module["ASSET_RELATIVE_PATH"] == Path(
        "artifacts/kcg_connector/isaac/"
        "d38999_shell25j_25_61_n_keyed_public_spec_v2.usda"
    )
    assert module["KEYED_MODEL_ID"] == KEYED_MODEL_ID
    assert module["FIXED_RECEPTACLE_PRIM_PATH"].endswith("/FixedReceptacle")
    assert module["LOOSE_PLUG_PRIM_PATH"].endswith("/LoosePlug")


def test_depth_only_derivation_keeps_front_keys_and_excludes_deeper_rear_shell():
    module = _module()
    _, depth, expected_front, center = _isolated_depth_observation()

    inputs = module["derive_isolated_probe_inputs"](depth)

    face = inputs["connector_face_mask"]
    assert face.dtype == np.bool_
    assert np.array_equal(face, expected_front)
    assert face[120, 194]
    assert not face[120, 205]
    assert np.all(inputs["occlusion_mask"] == 0)
    assert np.allclose(inputs["face_center_uv"], center, atol=0.6)
    diagnostics = inputs["diagnostics"]
    assert diagnostics["scope"] == module["PROBE_SCOPE"]
    assert "DISTANCE_TO_IMAGE_PLANE" in diagnostics["face_mask_source"]
    assert diagnostics["occlusion_estimator_general_scene_valid"] is False
    assert diagnostics["integrated_runtime_input_claimed"] is False


def test_depth_derived_inputs_run_only_the_shadow_pipeline():
    module = _module()
    _, inputs, result = _pipeline_result(module)

    assert result["status"] == "SHADOW_C2_BRANCH_SELECTED"
    assert result["passed"] is True
    assert result["selected_for_shadow"] == "C2_LINKED_BRANCH_0"
    assert result["control_authorized"] is False
    assert result["selected_for_control_allowed"] is False
    assert result["key_region_detection"]["control_authorized"] is False
    assert result["key_branch_selection"]["control_authorized"] is False
    assert result["key_region_detection"]["quality_diagnostics"][
        "candidate_count"
    ] == 5
    assert np.all(inputs["occlusion_mask"] == 0)


@pytest.mark.parametrize(
    "depth,band,match",
    (
        (np.zeros((10, 10, 1)), 0.0015, "shape"),
        (np.full((20, 20), np.inf), 0.0015, "insufficient"),
        (np.ones((20, 20)), 0.0, "positive"),
        (np.ones((20, 20)), math.nan, "finite"),
    ),
)
def test_invalid_or_empty_depth_derivation_is_rejected(depth, band, match):
    derive = _module()["derive_isolated_probe_inputs"]
    with pytest.raises(ValueError, match=match):
        derive(depth, front_surface_band_m=band)


def test_existing_output_directory_is_never_overwritten(tmp_path):
    resolve = _module()["resolve_new_output_directory"]
    new_path = tmp_path / "new_probe"
    assert resolve(new_path) == new_path.resolve()

    new_path.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        resolve(new_path)


def test_static_capture_requests_exactly_rgb_and_planar_depth():
    tree = ast.parse(_source())
    annotator_names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get_annotator" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant):
            annotator_names.append(node.args[0].value)

    assert annotator_names == ["rgb", "distance_to_image_plane"]
    capture = _source().split("def _capture_isolated_rgbd(", 1)[1].split(
        "def _save_artifacts(", 1
    )[0]
    for forbidden in (
        "get_world_pose",
        "get_local_pose",
        "GetLocalTransformation",
        "ComputeLocalToWorldTransform",
        "get_contact",
        "get_collider",
        "get_semantics",
    ):
        assert forbidden not in capture


def test_static_stage_isolates_fixed_end_and_uses_fixed_front_camera():
    source = _source()

    assert "context.open_stage(str(asset_path))" in source
    assert "UsdGeom.Imageable(fixed_prim).MakeInvisible()" in source
    assert "fixed_prim.SetActive(False)" in source
    assert "matrix.SetTranslateOnly" in source
    assert '"FIXED_WORLD_FRONT_VIEW_ALONG_MINUS_Z"' in source
    assert "run_palm_key_shadow_pipeline(" in source
    assert '"control_authorized": False' in source
    assert '"selected_for_control_allowed": False' in source


def test_probe_saves_complete_non_overwriting_shadow_evidence(tmp_path):
    module = _module()
    rgb, _, _, _ = _isolated_depth_observation()
    depth, inputs, result = _pipeline_result(module)
    output_dir = tmp_path / "probe_evidence"
    capture = {
        "annotators": ["rgb", "distance_to_image_plane"],
        "fixed_receptacle_isolated": True,
        "semantic_annotator_used": False,
        "object_pose_queries": 0,
        "contact_queries": 0,
        "collider_queries": 0,
    }

    report = module["_save_artifacts"](
        output_dir=output_dir,
        rgb=rgb,
        depth=depth,
        inputs=inputs,
        result=result,
        capture_diagnostics=capture,
    )

    expected = {
        "rgb.png",
        "depth_m.npy",
        "depth_preview.png",
        "connector_face_mask.npy",
        "connector_face_mask.png",
        "occlusion_mask.npy",
        "occlusion_mask.png",
        "key_probability.npy",
        "key_probability.png",
        "shadow_result.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    stored = json.loads((output_dir / "shadow_result.json").read_text())
    assert stored["probe_scope"] == module["PROBE_SCOPE"]
    assert stored["shadow_only"] is True
    assert stored["control_authorized"] is False
    assert stored["selected_for_control_allowed"] is False
    assert stored["shadow_result"]["control_authorized"] is False
    assert stored["claims"]["integrated_runtime_validated"] is False
    assert stored["claims"]["insertion_control_validated"] is False
    assert report["passed"] is True

    with pytest.raises(FileExistsError):
        module["_save_artifacts"](
            output_dir=output_dir,
            rgb=rgb,
            depth=depth,
            inputs=inputs,
            result=result,
            capture_diagnostics=capture,
        )


def test_shadow_boundary_guard_rejects_any_control_promotion():
    module = _module()
    _, _, result = _pipeline_result(module)
    module["_require_shadow_only"](result)

    promoted = dict(result)
    promoted["control_authorized"] = True
    with pytest.raises(RuntimeError, match="authorize control"):
        module["_require_shadow_only"](promoted)
