import ast
from collections import Counter
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "isaac/d38999_keyed_v2_yaw_dataset.py"
CONFIG_PATH = PACKAGE_ROOT / "config/d38999_keyed_v2_yaw_dataset_v1.yaml"


def _load_module():
    name = "d38999_keyed_v2_yaw_dataset_cpu_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dataset = _load_module()


def _observation(width=8, height=6):
    # Deliberately use a noncanonical key order: field order is not data truth.
    return {
        "intrinsics": {
            "cy_px": (height - 1) / 2,
            "cx_px": (width - 1) / 2,
            "fy_px": 20.0,
            "fx_px": 20.0,
            "height_px": height,
            "width_px": width,
        },
        "connector_face_mask": np.ones((height, width), dtype=np.bool_),
        "occlusion_mask": np.zeros((height, width), dtype=np.bool_),
        "depth_m": np.full((height, width), 0.1, dtype=np.float32),
        "rgb": np.zeros((height, width, 3), dtype=np.uint8),
    }


def _synthetic_rgbd(center_uv=(50.0, 50.0)):
    rows, columns = np.indices((101, 101), dtype=np.float64)
    face = np.hypot(rows - center_uv[1], columns - center_uv[0]) <= 30.0
    rgb = np.zeros((101, 101, 3), dtype=np.uint8)
    rgb[face] = (180, 120, 70)
    depth = np.full((101, 101), np.inf, dtype=np.float64)
    depth[face] = 0.12
    return rgb, depth, face


def _final_observation(rgb, depth):
    face, occlusion = dataset.derive_image_masks(rgb, depth)
    return {
        "rgb": rgb,
        "depth_m": depth,
        "connector_face_mask": face,
        "occlusion_mask": occlusion,
        "intrinsics": {
            "width_px": rgb.shape[1],
            "height_px": rgb.shape[0],
            "fx_px": 100.0,
            "fy_px": 100.0,
            "cx_px": 50.0,
            "cy_px": 50.0,
        },
    }


def _injection_plan(reason, injection):
    return dataset.SamplePlan(
        sample_id="synthetic",
        split="dev",
        authored_yaw_deg=0.0,
        yaw_stratum=32,
        authored_pose={},
        authored_light={},
        expected_reject=reason is not None,
        expected_reject_reason=reason,
        reject_injection=injection,
    )


def test_import_is_cpu_only_and_isaac_imports_are_lazy():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_imports = {
        node.names[0].name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.names
    }
    assert not {"isaacsim", "omni", "pxr"} & top_imports


def test_contract_freezes_counts_ranges_and_control_denial():
    contract = dataset.load_contract(CONFIG_PATH)
    schedule = contract["frozen_schedule"]
    assert schedule["dev"]["total"] == 1024
    assert schedule["heldout"]["total"] == 3072
    assert schedule["heldout"]["visible_valid"] == {
        "count": 2048,
        "yaw_strata": 64,
        "nuisance_per_stratum": 32,
    }
    assert schedule["heldout"]["must_reject"]["categories"] == {
        reason: 256 for reason in dataset.REJECTION_REASONS
    }
    nuisance = contract["nuisance_ranges"]
    assert nuisance["camera_axial_distance_m"] == [0.09, 0.14]
    assert nuisance["lateral_x_m"] == [-0.006, 0.006]
    assert nuisance["lateral_y_m"] == [-0.006, 0.006]
    assert nuisance["tilt_x_deg"] == [-8.0, 8.0]
    assert nuisance["tilt_y_deg"] == [-8.0, 8.0]
    postcondition = contract["pixel_postcondition"]
    assert tuple(postcondition["classes_exactly"]) == dataset.POSTCONDITION_CLASSES
    assert postcondition["empty_face_policy"] == "ERROR_NO_WRITE"
    assert postcondition["occlusion_unknown_policy"] == "ERROR_NO_WRITE"
    assert postcondition["missing_depth_support_semantics"] == (
        "RGB_FOREGROUND_INVALID_DEPTH_NO_VALID_DEPTH_IN_3X3"
    )
    assert postcondition["missing_depth_3x3_boundary_policy"] == (
        "OUT_OF_IMAGE_TREATED_AS_NO_VALID_DEPTH"
    )
    assert postcondition["missing_depth_component_connectivity"] == 8
    assert postcondition["missing_depth_minimum_component_pixels"] == 16
    assert postcondition["depth_missing_scope"] == (
        "SUBSTANTIAL_SIMULATED_DROPOUT_DATASET_POSTCONDITION_ONLY"
    )
    assert postcondition[
        "refiner_key_centroid_3x3_strict_missing_depth_gate_replaced"
    ] is False
    assert contract["directory_contract"][
        "write_requires_pixel_postcondition_match"
    ] is True
    diagnostics = contract["failure_diagnostics"]
    assert diagnostics["allowed_mode"] == "PREFLIGHT_FAILURE_ONLY"
    assert diagnostics["existing_sample_policy"] == "REFUSE_OVERWRITE"
    assert set(diagnostics["files_exactly"]) == dataset.DIAGNOSTIC_FILES
    assert set(diagnostics["json_fields_exactly"]) == (
        dataset.DIAGNOSTIC_JSON_FIELDS
    )
    assert set(diagnostics["forbidden_truth_fields"]) == (
        dataset.DIAGNOSTIC_FORBIDDEN_TRUTH_FIELDS
    )
    authorization = contract["authorization"]
    assert all(
        authorization[field] is False
        for field in (
            "selected_for_control_allowed",
            "simulation_insertion_control_authorized",
            "robot_control_authorized",
            "hardware_control_authorized",
        )
    )


@pytest.fixture(scope="module")
def schedules():
    return dataset.build_dataset_schedule(dataset.load_contract(CONFIG_PATH))


@pytest.mark.parametrize(
    "split,total,visible,per_reject,visible_per_stratum,reject_per_stratum",
    (
        ("dev", 1024, 512, 128, 8, 2),
        ("heldout", 3072, 2048, 256, 32, 4),
    ),
)
def test_schedule_has_exact_stratified_counts(
    schedules,
    split,
    total,
    visible,
    per_reject,
    visible_per_stratum,
    reject_per_stratum,
):
    plans = schedules[split]
    assert len(plans) == total
    assert len({plan.sample_id for plan in plans}) == total
    assert sum(not plan.expected_reject for plan in plans) == visible
    assert Counter(
        plan.expected_reject_reason for plan in plans if plan.expected_reject
    ) == {reason: per_reject for reason in dataset.REJECTION_REASONS}
    assert Counter(
        plan.yaw_stratum for plan in plans if not plan.expected_reject
    ) == {stratum: visible_per_stratum for stratum in range(64)}
    for reason in dataset.REJECTION_REASONS:
        assert Counter(
            plan.yaw_stratum
            for plan in plans
            if plan.expected_reject_reason == reason
        ) == {stratum: reject_per_stratum for stratum in range(64)}


def test_schedule_values_stay_inside_frozen_domain(schedules):
    for plans in schedules.values():
        for plan in plans:
            x, y, axial = plan.authored_pose["translation_m"]
            tilt_x, tilt_y, yaw = plan.authored_pose["rotation_xyz_deg"]
            assert -180.0 <= plan.authored_yaw_deg < 180.0
            assert yaw == plan.authored_yaw_deg
            assert 0.09 <= axial < 0.14
            assert -0.006 <= x < 0.006
            assert -0.006 <= y < 0.006
            assert -8.0 <= tilt_x < 8.0
            assert -8.0 <= tilt_y < 8.0
            assert 500.0 <= plan.authored_light["key_light_intensity"] < 1400.0
            assert 3500.0 <= plan.authored_light["color_temperature_k"] < 7500.0
            assert -1.0 <= plan.authored_light["exposure_ev"] < 1.0


def test_rng_and_first_records_freeze_seed_and_order(schedules):
    rng = dataset.SplitMix64(0)
    assert [rng.next_u64() for _ in range(4)] == [
        16294208416658607535,
        7960286522194355700,
        487617019471545679,
        17909611376780542444,
    ]
    dev = schedules["dev"][:3]
    assert [(plan.yaw_stratum, plan.expected_reject_reason) for plan in dev] == [
        (43, "KEY_REGION_OCCLUDED"),
        (38, None),
        (23, None),
    ]
    assert [plan.authored_yaw_deg for plan in dev] == pytest.approx(
        [62.22576792981122, 36.3156828193259, -47.94612617635981]
    )
    contract = dataset.load_contract(CONFIG_PATH)
    heldout_direct = dataset.build_split_schedule(contract, "heldout")
    dataset.build_split_schedule(contract, "dev")
    heldout_after_dev = dataset.build_split_schedule(contract, "heldout")
    assert [plan.truth_record() for plan in heldout_direct] == [
        plan.truth_record() for plan in schedules["heldout"]
    ]
    assert [plan.truth_record() for plan in heldout_after_dev] == [
        plan.truth_record() for plan in heldout_direct
    ]


def test_split_cli_has_distinct_safe_default_output_roots(tmp_path):
    dev = dataset._arguments(["--run", "--split", "dev"])
    heldout = dataset._arguments(["--run", "--split", "heldout"])
    all_splits = dataset._arguments(["--run"])
    assert dev.split == "dev"
    assert heldout.split == "heldout"
    assert all_splits.split == "all"
    assert dev.output_dir is None
    dev_root = dataset.default_output_root(tmp_path, dev.split)
    heldout_root = dataset.default_output_root(tmp_path, heldout.split)
    assert dev_root != heldout_root
    assert dev_root.name.endswith("_dev")
    assert heldout_root.name.endswith("_heldout")

    preflight = dataset._arguments(["--run", "--preflight"])
    preflight_root = dataset.default_output_root(
        tmp_path, preflight.split, preflight=True
    )
    assert preflight.preflight is True
    assert preflight_root not in {dev_root, heldout_root}
    assert preflight_root.name.endswith("_preflight")
    with pytest.raises(SystemExit):
        dataset._arguments(["--run", "--preflight", "--split", "dev"])


def test_preflight_is_a_deterministic_five_class_subset_of_frozen_dev(schedules):
    contract = dataset.load_contract(CONFIG_PATH)
    first = dataset.build_preflight_schedule(contract)["dev"]
    second = dataset.build_preflight_schedule(contract)["dev"]
    assert len(first) == 5
    assert tuple(dataset.expected_postcondition(plan) for plan in first) == (
        dataset.POSTCONDITION_CLASSES
    )
    assert [plan.truth_record() for plan in first] == [
        plan.truth_record() for plan in second
    ]
    frozen_dev = {plan.sample_id: plan.truth_record() for plan in schedules["dev"]}
    assert all(frozen_dev[plan.sample_id] == plan.truth_record() for plan in first)


@pytest.mark.parametrize("truth_field", sorted(dataset.FORBIDDEN_PREDICTION_FIELDS))
def test_prediction_api_rejects_every_declared_truth_field(truth_field):
    observation = _observation()
    observation[truth_field] = "forbidden"
    with pytest.raises(ValueError, match="truth fields reached prediction"):
        dataset.validate_prediction_observation(observation)


def test_predictor_receives_only_images_and_cannot_authorize_control():
    seen = {}

    def predictor(observation):
        seen.update(observation)
        return {"yaw_deg": 1.25, "control_authorized": False}

    result = dataset.call_prediction_api(predictor, _observation())
    assert set(seen) == dataset.ALLOWED_OBSERVATION_FIELDS
    assert all(result[field] is False for field in dataset.CONTROL_FIELDS)
    assert result["authorization_scope"].endswith("NO_CONTROL")

    with pytest.raises(RuntimeError, match="attempted to promote"):
        dataset.call_prediction_api(
            lambda _observation: {"robot_control_authorized": True},
            _observation(),
        )


def test_final_pixels_classify_visible_and_all_four_reject_categories():
    rgb, depth, expected_face = _synthetic_rgbd()
    visible = _final_observation(rgb, depth)
    visible_face = visible["connector_face_mask"]
    visible_occlusion = visible["occlusion_mask"]
    assert np.array_equal(visible_face, expected_face)
    assert visible_occlusion is not None
    assert not np.any(visible_occlusion)
    assert dataset.classify_observation_postcondition(visible) == "VISIBLE_VALID"

    occlusion_plan = _injection_plan(
        "KEY_REGION_OCCLUDED",
        {"occluded_face_fraction": 0.25, "angle_deg": 15.0},
    )
    occluded_rgb, occluded_depth = dataset._apply_reject_injection(
        occlusion_plan, rgb, depth
    )
    occluded = _final_observation(occluded_rgb, occluded_depth)
    occluded_face = occluded["connector_face_mask"]
    occlusion = occluded["occlusion_mask"]
    assert occlusion is not None and np.any(occlusion)
    assert not np.any(occluded_face & occlusion)
    assert np.all(~occluded_face[occlusion])
    assert dataset.classify_observation_postcondition(occluded) == (
        "KEY_REGION_OCCLUDED"
    )

    missing_plan = _injection_plan(
        "KEY_REGION_DEPTH_MISSING",
        {"key_region_depth_dropout_fraction": 0.90, "angle_deg": -40.0},
    )
    missing_rgb, missing_depth = dataset._apply_reject_injection(
        missing_plan, rgb, depth
    )
    missing = _final_observation(missing_rgb, missing_depth)
    missing_face = missing["connector_face_mask"]
    missing_occlusion = missing["occlusion_mask"]
    dropped_face = expected_face & ~np.isfinite(missing_depth)
    valid_depth_in_3x3 = dataset._any_valid_depth_in_3x3(
        np.isfinite(missing_depth) & (missing_depth > 0.0)
    )
    substantial_dropout = dropped_face & ~valid_depth_in_3x3
    assert np.any(dropped_face)
    assert np.any(substantial_dropout)
    assert np.all(missing_face[substantial_dropout])
    assert missing_occlusion is not None and not np.any(missing_occlusion)
    assert dataset.classify_observation_postcondition(missing) == (
        "KEY_REGION_DEPTH_MISSING"
    )

    low_plan = _injection_plan(
        "KEY_REGION_LOW_CONFIDENCE",
        {"contrast_scale": 0.02, "gaussian_blur_radius_px": 5.0},
    )
    low_rgb, low_depth = dataset._apply_reject_injection(low_plan, rgb, depth)
    low = _final_observation(low_rgb, low_depth)
    assert dataset.classify_observation_postcondition(low) == (
        "KEY_REGION_LOW_CONFIDENCE"
    )

    out_rgb, out_depth, _ = _synthetic_rgbd(center_uv=(5.0, 50.0))
    out_of_frame = _final_observation(out_rgb, out_depth)
    assert np.any(out_of_frame["connector_face_mask"][:, :2])
    assert dataset.classify_observation_postcondition(out_of_frame) == (
        "CONNECTOR_FACE_OUT_OF_FRAME"
    )


def test_valid_depth_3x3_boundary_treats_outside_as_not_valid():
    empty = np.zeros((3, 4), dtype=np.bool_)
    assert not np.any(dataset._any_valid_depth_in_3x3(empty))

    valid = empty.copy()
    valid[0, 0] = True
    neighborhood = dataset._any_valid_depth_in_3x3(valid)
    expected = np.zeros_like(valid)
    expected[:2, :2] = True
    assert np.array_equal(neighborhood, expected)


def test_missing_depth_component_filter_uses_8_connectivity_and_minimum_16():
    candidate = np.zeros((24, 24), dtype=np.bool_)
    candidate[1, 20] = True
    candidate[3:6, 3:6] = True
    candidate[12:16, 12:16] = True
    filtered = dataset._filter_small_8_connected_components(candidate)
    expected = np.zeros_like(candidate)
    expected[12:16, 12:16] = True
    assert np.array_equal(filtered, expected)

    diagonal = np.eye(16, dtype=np.bool_)
    assert np.array_equal(
        dataset._filter_small_8_connected_components(diagonal), diagonal
    )


def test_single_pixel_and_small_missing_fragments_are_not_face_support():
    rgb, depth, _ = _synthetic_rgbd()
    rgb[4, 4] = (180, 120, 70)
    rgb[8:11, 8:11] = (180, 120, 70)
    observation = _final_observation(rgb, depth)
    face = observation["connector_face_mask"]
    assert not face[4, 4]
    assert not np.any(face[8:11, 8:11])
    assert dataset.classify_observation_postcondition(observation) == (
        "VISIBLE_VALID"
    )


def test_v4_like_thin_missing_depth_contour_is_not_dataset_dropout():
    rows, columns = np.indices((257, 257), dtype=np.float64)
    radius = np.hypot(rows - 128.0, columns - 128.0)
    rgb_face = radius <= 114.0
    valid_face = radius <= 113.0
    thin_missing_contour = rgb_face & ~valid_face
    rgb = np.zeros((257, 257, 3), dtype=np.uint8)
    rgb[rgb_face] = (180, 120, 70)
    depth = np.full((257, 257), np.inf, dtype=np.float64)
    depth[valid_face] = 0.12

    valid_depth_in_3x3 = dataset._any_valid_depth_in_3x3(valid_face)
    assert np.count_nonzero(thin_missing_contour) > 400
    assert np.all(valid_depth_in_3x3[thin_missing_contour])
    observation = _final_observation(rgb, depth)
    assert not np.any(
        observation["connector_face_mask"] & thin_missing_contour
    )
    assert dataset.classify_observation_postcondition(observation) == (
        "VISIBLE_VALID"
    )


def test_single_internal_missing_pixel_is_ignored_by_dataset_postcondition():
    rgb, depth, _ = _synthetic_rgbd()
    depth[50, 50] = np.nan
    valid = np.isfinite(depth) & (depth > 0.0)
    assert dataset._any_valid_depth_in_3x3(valid)[50, 50]

    observation = _final_observation(rgb, depth)
    assert not observation["connector_face_mask"][50, 50]
    assert dataset.classify_observation_postcondition(observation) == (
        "VISIBLE_VALID"
    )


def test_substantial_outer_annulus_dropout_remains_depth_missing():
    rgb, depth, expected_face = _synthetic_rgbd()
    plan = _injection_plan(
        "KEY_REGION_DEPTH_MISSING",
        {"key_region_depth_dropout_fraction": 0.90, "angle_deg": -40.0},
    )
    missing_rgb, missing_depth = dataset._apply_reject_injection(
        plan, rgb, depth
    )
    observation = _final_observation(missing_rgb, missing_depth)
    valid = np.isfinite(missing_depth) & (missing_depth > 0.0)
    substantial_dropout = (
        expected_face
        & ~valid
        & ~dataset._any_valid_depth_in_3x3(valid)
    )
    assert np.any(substantial_dropout)
    assert np.all(observation["connector_face_mask"][substantial_dropout])
    assert dataset.classify_observation_postcondition(observation) == (
        "KEY_REGION_DEPTH_MISSING"
    )


def test_postcondition_rejects_empty_unknown_and_overlapping_masks():
    rgb, depth, _ = _synthetic_rgbd()
    observation = _final_observation(rgb, depth)

    empty = dict(observation)
    empty["connector_face_mask"] = np.zeros_like(
        observation["connector_face_mask"]
    )
    with pytest.raises(RuntimeError, match="face_mask is empty"):
        dataset.classify_observation_postcondition(empty)

    unknown = dict(observation)
    unknown["occlusion_mask"] = None
    with pytest.raises(RuntimeError, match="occlusion is unknown"):
        dataset.classify_observation_postcondition(unknown)

    overlap = dict(observation)
    overlap["occlusion_mask"] = observation["connector_face_mask"].copy()
    with pytest.raises(RuntimeError, match="must be disjoint"):
        dataset.classify_observation_postcondition(overlap)


def test_injection_refuses_an_empty_initial_face_mask():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.full((32, 32), np.inf, dtype=np.float64)
    plan = _injection_plan(
        "KEY_REGION_DEPTH_MISSING",
        {"key_region_depth_dropout_fraction": 0.9, "angle_deg": 0.0},
    )
    with pytest.raises(RuntimeError, match="initial connector face mask is empty"):
        dataset._apply_reject_injection(plan, rgb, depth)


def test_occlusion_unknown_fails_closed_without_calling_predictor():
    observation = _observation()
    observation["depth_m"][:] = np.inf
    observation["connector_face_mask"][:] = False
    observation["occlusion_mask"] = None
    called = False

    def predictor(_observation):
        nonlocal called
        called = True
        return {}

    result = dataset.call_prediction_api(predictor, observation)
    assert called is False
    assert result["status"] == "OCCLUSION_UNKNOWN"
    assert result["rejection_code"] == "KEY_REGION_OCCLUSION_UNKNOWN"
    assert all(result[field] is False for field in dataset.CONTROL_FIELDS)


def test_inference_and_truth_are_separate_and_never_overwritten(tmp_path, schedules):
    contract = dataset.load_contract(CONFIG_PATH)
    output = tmp_path / "new_dataset"
    layout = dataset.prepare_output_layout(output, contract, ("heldout",))
    assert "inference_dev" not in layout
    assert "truth_dev" not in layout
    plan = next(plan for plan in schedules["heldout"] if not plan.expected_reject)
    rgb, depth, _ = _synthetic_rgbd()
    observation = _final_observation(rgb, depth)
    inference_dir, truth_path = dataset.write_separated_sample(
        layout, plan, observation
    )
    assert inference_dir.parent == output / "inference_inputs/heldout"
    assert truth_path.parent == output / "truth/heldout"
    assert {item.name for item in inference_dir.iterdir()} == dataset.INFERENCE_SAMPLE_FILES
    assert not any(path.suffix == ".json" and path.name != "intrinsics.json" for path in inference_dir.iterdir())
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    assert truth["authored_yaw_deg"] == plan.authored_yaw_deg
    assert truth["expected_reject"] is False
    with pytest.raises(FileExistsError):
        dataset.write_separated_sample(layout, plan, observation)
    with pytest.raises(FileExistsError):
        dataset.prepare_output_layout(output, contract)


def test_postcondition_failures_create_no_sample_or_truth_files(tmp_path):
    contract = dataset.load_contract(CONFIG_PATH)
    rgb, depth, _ = _synthetic_rgbd()
    visible = _final_observation(rgb, depth)
    cases = []

    mismatch_plan = _injection_plan(
        "KEY_REGION_OCCLUDED",
        {"occluded_face_fraction": 0.25, "angle_deg": 0.0},
    )
    cases.append(("mismatch", mismatch_plan, visible, "postcondition mismatch"))

    empty = dict(visible)
    empty["connector_face_mask"] = np.zeros_like(visible["connector_face_mask"])
    cases.append(("empty", _injection_plan(None, {}), empty, "face_mask is empty"))

    unknown = dict(visible)
    unknown["occlusion_mask"] = None
    cases.append(("unknown", _injection_plan(None, {}), unknown, "occlusion is unknown"))

    for label, plan, observation, message in cases:
        layout = dataset.prepare_output_layout(
            tmp_path / label, contract, ("dev",)
        )
        with pytest.raises(RuntimeError, match=message):
            dataset.write_separated_sample(layout, plan, observation)
        assert not (layout["inference_dev"] / plan.sample_id).exists()
        assert not (layout["truth_dev"] / f"{plan.sample_id}.json").exists()


def test_preflight_mismatch_writes_diagnostic_but_no_dataset_sample(tmp_path):
    contract = dataset.load_contract(CONFIG_PATH)
    layout = dataset.prepare_output_layout(
        tmp_path / "preflight_mismatch", contract, ("dev",)
    )
    rgb, depth, _ = _synthetic_rgbd()
    observation = _final_observation(rgb, depth)
    plan = _injection_plan(
        "KEY_REGION_OCCLUDED",
        {"occluded_face_fraction": 0.25, "angle_deg": 0.0},
    )

    with pytest.raises(RuntimeError, match="postcondition mismatch"):
        dataset._raise_pixel_postcondition_mismatch(
            layout,
            plan,
            observation,
            "VISIBLE_VALID",
            preflight=True,
        )

    diagnostic_dir = layout["diagnostics_root"] / plan.sample_id
    assert diagnostic_dir.parent == layout["root"] / "diagnostics"
    assert not diagnostic_dir.is_relative_to(layout["inference_root"])
    assert not diagnostic_dir.is_relative_to(layout["truth_root"])
    assert {item.name for item in diagnostic_dir.iterdir()} == (
        dataset.DIAGNOSTIC_FILES
    )
    from PIL import Image

    saved_rgb = np.asarray(Image.open(diagnostic_dir / "rgb.png"))
    saved_depth = np.load(diagnostic_dir / "depth_m.npy", allow_pickle=False)
    saved_face = np.load(
        diagnostic_dir / "connector_face_mask.npy", allow_pickle=False
    )
    saved_occlusion = np.load(
        diagnostic_dir / "occlusion_mask.npy", allow_pickle=False
    )
    assert np.array_equal(saved_rgb, observation["rgb"])
    assert np.array_equal(saved_depth, observation["depth_m"].astype(np.float32))
    assert np.array_equal(saved_face, observation["connector_face_mask"])
    assert np.array_equal(saved_occlusion, observation["occlusion_mask"])

    diagnostic = json.loads(
        (diagnostic_dir / "diagnostic.json").read_text(encoding="utf-8")
    )
    assert set(diagnostic) == dataset.DIAGNOSTIC_JSON_FIELDS
    assert diagnostic["expected"] == "KEY_REGION_OCCLUDED"
    assert diagnostic["observed"] == "VISIBLE_VALID"
    assert diagnostic["face_pixels"] == int(
        np.count_nonzero(observation["connector_face_mask"])
    )
    assert diagnostic["face_missing_depth_pixels"] == 0
    assert diagnostic["valid_depth_pixels"] == diagnostic["face_pixels"]
    assert diagnostic["occlusion_pixels"] == 0
    assert diagnostic["rgb_range"] == [0, 180]
    assert diagnostic["rgb_std"] == pytest.approx(
        float(np.std(observation["rgb"].astype(np.float64)))
    )
    assert diagnostic["dataset_sample"] is False
    assert diagnostic["control"] is False
    assert diagnostic["truth_fields_included"] is False
    assert not (set(diagnostic) & dataset.DIAGNOSTIC_FORBIDDEN_TRUTH_FIELDS)
    assert not any(layout["inference_dev"].iterdir())
    assert not any(layout["truth_dev"].iterdir())
    assert dataset._completed_sample_count(layout, ("dev",)) == 0

    with pytest.raises(FileExistsError):
        dataset._write_preflight_mismatch_diagnostic(
            layout,
            plan,
            observation,
            "VISIBLE_VALID",
            preflight=True,
        )


def test_failure_diagnostics_are_preflight_only_and_encode_unknown(tmp_path):
    contract = dataset.load_contract(CONFIG_PATH)
    rgb, depth, _ = _synthetic_rgbd()
    observation = _final_observation(rgb, depth)
    plan = _injection_plan(
        "KEY_REGION_OCCLUDED",
        {"occluded_face_fraction": 0.25, "angle_deg": 0.0},
    )
    dataset_layout = dataset.prepare_output_layout(
        tmp_path / "dataset_mode", contract, ("dev",)
    )
    with pytest.raises(RuntimeError, match="postcondition mismatch"):
        dataset._raise_pixel_postcondition_mismatch(
            dataset_layout,
            plan,
            observation,
            "VISIBLE_VALID",
            preflight=False,
        )
    assert not dataset_layout["diagnostics_root"].exists()

    unknown_layout = dataset.prepare_output_layout(
        tmp_path / "unknown_occlusion", contract, ("dev",)
    )
    unknown = dict(observation)
    unknown["occlusion_mask"] = None
    diagnostic_dir = dataset._write_preflight_mismatch_diagnostic(
        unknown_layout,
        plan,
        unknown,
        "VISIBLE_VALID",
        preflight=True,
    )
    encoded = np.load(
        diagnostic_dir / "occlusion_mask.npy", allow_pickle=False
    )
    assert encoded.shape == ()
    assert encoded.item() == "OCCLUSION_UNKNOWN"
    diagnostic = json.loads(
        (diagnostic_dir / "diagnostic.json").read_text(encoding="utf-8")
    )
    assert diagnostic["occlusion_pixels"] is None
    assert diagnostic["dataset_sample"] is False
    assert not (set(diagnostic) & dataset.DIAGNOSTIC_FORBIDDEN_TRUTH_FIELDS)
    assert dataset._completed_sample_count(unknown_layout, ("dev",)) == 0


def test_contract_tampering_fails_closed(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["authorization"]["robot_control_authorized"] = True
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must remain false"):
        dataset.load_contract(path)


def test_isaac_capture_requests_no_truth_annotators_or_queries():
    assert tuple(inspect.signature(dataset.derive_image_masks).parameters) == (
        "rgb",
        "depth_m",
    )
    assert tuple(
        inspect.signature(dataset.classify_observation_postcondition).parameters
    ) == ("observation",)
    source = inspect.getsource(dataset._generate_with_isaac)
    tree = ast.parse(source)

    def is_literal_annotator_call(node):
        checks = (
            isinstance(node, ast.Call),
            isinstance(getattr(node, "func", None), ast.Attribute),
            getattr(getattr(node, "func", None), "attr", None) == "get_annotator",
            bool(getattr(node, "args", ())),
        )
        return all(checks) and isinstance(node.args[0], ast.Constant)

    requested = [
        node.args[0].value
        for node in ast.walk(tree)
        if is_literal_annotator_call(node)
    ]
    assert requested == ["rgb", "distance_to_image_plane"]
    for forbidden_call in (
        "get_world_pose",
        "get_contact_report",
        "get_contact_sensor",
        "get_collision",
        "get_semantics",
    ):
        assert forbidden_call not in source


def test_main_prechecks_output_before_simulation_and_gates_preflight_report():
    source = inspect.getsource(dataset.main)
    assert source.index("require_new_output_root(output_root)") < source.index(
        "from isaacsim import SimulationApp"
    )
    assert 'report["preflight_passed"] = True' in source
    assert '"active_gpu": 0' in source
    assert '"physics_gpu": 0' in source
    success_gate = inspect.getsource(dataset._validate_generation_success)
    assert "expected_total != 5" in success_gate
    assert "counts.get(name) != 1" in success_gate
    assert "files != INFERENCE_SAMPLE_FILES" in success_gate


def test_runtime_reports_are_written_before_explicit_app_close():
    contract = dataset.load_contract(CONFIG_PATH)
    reports = contract["runtime_reports"]
    assert reports["report_write_before_app_close"] is True
    assert reports["traceback_on_failure"] is True
    assert reports["app_close_exit_code_success"] == 0
    assert reports["app_close_exit_code_failure"] == 1

    source = inspect.getsource(dataset.main)
    success_write = source.index('with report_path.open(')
    failure_write = source.index("_write_generation_failure(")
    app_close = source.index("app.close(exit_code=0 if completed else 1)")
    assert success_write < app_close
    assert failure_write < app_close
    assert "traceback.print_exc()" in source
    assert source.rstrip().endswith(
        "app.close(exit_code=0 if completed else 1)"
    )


def test_runtime_capture_warms_annotators_and_validates_every_frame():
    contract = dataset.load_contract(CONFIG_PATH)
    camera = contract["camera"]
    runtime = contract["runtime_capture"]
    assert camera["prim_path"] == "/World/KeyedV2YawDatasetCamera"
    assert camera["render_product_name"] == "D38999KeyedV2YawDatasetProduct"
    assert runtime["app_updates_after_annotator_attach"] >= 1
    assert runtime["replicator_warmup_frames"] >= 1
    assert runtime["per_sample_render_frames"] >= 1

    source = inspect.getsource(dataset._generate_with_isaac)
    camera_define = source.index(
        'UsdGeom.Camera.Define(stage, camera_config["prim_path"])'
    )
    camera_prim = source.index("camera_prim = camera.GetPrim()", camera_define)
    product = source.index("rep.create.render_product(", camera_prim)
    attach = source.index("annotator.attach([render_product_path])")
    update = source.index("simulation_app.update()", attach)
    warmup = source.index('runtime["replicator_warmup_frames"]', update)
    sample_loop = source.index("for split, plans in schedules.items()", warmup)
    assert camera_define < camera_prim < product < attach < update < warmup < sample_loop
    assert 'name=camera_config["render_product_name"]' in source
    assert "render_product_path is None" in source
    assert "rgba_data is None" in source
    assert "depth_data is None" in source
    assert "rgba.shape[:2] != expected_shape" in source
    assert "depth.shape != expected_shape" in source
    assert "not np.any(rgb)" in source
    assert "not np.any(valid_depth)" in source


def _install_fake_isaac(monkeypatch):
    module = types.ModuleType("isaacsim")

    class FakeSimulationApp:
        instances = []

        def __init__(self, _settings):
            self.closed = False
            self.close_exit_code = None
            self.__class__.instances.append(self)

        def close(self, *, exit_code):
            self.closed = True
            self.close_exit_code = exit_code

    module.SimulationApp = FakeSimulationApp
    monkeypatch.setitem(sys.modules, "isaacsim", module)
    return FakeSimulationApp


def _fake_success_report():
    return {
        "schema_version": dataset.SCHEMA_VERSION,
        "generated_samples": 5,
        "pixel_postcondition_counts": {
            name: 1 for name in dataset.POSTCONDITION_CLASSES
        },
        "annotators": ["rgb", "distance_to_image_plane"],
        "truth_and_inference_directories_separate": True,
        **{field: False for field in dataset.CONTROL_FIELDS},
    }


def test_runtime_exception_writes_failure_and_returns_one(monkeypatch, tmp_path):
    fake_app = _install_fake_isaac(monkeypatch)

    def fail_generation(*_args, **_kwargs):
        raise RuntimeError("DispatchSync produced no annotator frame")

    monkeypatch.setattr(dataset, "_generate_with_isaac", fail_generation)
    output = tmp_path / "runtime_failure"
    result = dataset.main(
        ["--run", "--preflight", "--output-dir", str(output)]
    )
    assert result == 1
    assert fake_app.instances[-1].closed is True
    assert fake_app.instances[-1].close_exit_code == 1
    assert not (output / "generation_report.json").exists()
    failure_path = output / "generation_failure.json"
    assert failure_path.is_file()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert set(failure) == set(dataset.FAILURE_REPORT_FIELDS)
    assert failure["exception_type"] == "RuntimeError"
    assert "DispatchSync" in failure["exception_message"]
    assert failure["generated_samples"] == 0
    assert all(failure[field] is False for field in dataset.CONTROL_FIELDS)
    assert not ({"authored_yaw_deg", "authored_pose", "expected_reject"} & set(failure))


def test_empty_output_directories_cannot_exit_zero(monkeypatch, tmp_path):
    fake_app = _install_fake_isaac(monkeypatch)
    monkeypatch.setattr(
        dataset,
        "_generate_with_isaac",
        lambda *_args, **_kwargs: _fake_success_report(),
    )
    output = tmp_path / "empty_runtime"
    result = dataset.main(
        ["--run", "--preflight", "--output-dir", str(output)]
    )
    assert result == 1
    assert fake_app.instances[-1].closed is True
    assert fake_app.instances[-1].close_exit_code == 1
    assert not (output / "generation_report.json").exists()
    failure = json.loads(
        (output / "generation_failure.json").read_text(encoding="utf-8")
    )
    assert failure["generated_samples"] == 0
    assert "incomplete" in failure["exception_message"]


def test_success_requires_five_disk_samples_and_report(monkeypatch, tmp_path):
    fake_app = _install_fake_isaac(monkeypatch)

    def fake_generation(
        _app, _asset, _contract, schedules, layout, *, preflight
    ):
        assert preflight is True
        for split, plans in schedules.items():
            for plan in plans:
                sample = layout[f"inference_{split}"] / plan.sample_id
                sample.mkdir()
                for filename in dataset.INFERENCE_SAMPLE_FILES:
                    (sample / filename).write_bytes(b"observation")
                (layout[f"truth_{split}"] / f"{plan.sample_id}.json").write_text(
                    "{}\n", encoding="utf-8"
                )
        return _fake_success_report()

    monkeypatch.setattr(dataset, "_generate_with_isaac", fake_generation)
    output = tmp_path / "successful_runtime"
    result = dataset.main(
        ["--run", "--preflight", "--output-dir", str(output)]
    )
    assert result == 0
    assert fake_app.instances[-1].closed is True
    assert fake_app.instances[-1].close_exit_code == 0
    assert not (output / "generation_failure.json").exists()
    report_path = output / "generation_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["generated_samples"] == 5
    assert report["preflight_passed"] is True
    assert report["pixel_postcondition_counts"] == {
        name: 1 for name in dataset.POSTCONDITION_CLASSES
    }
