import ast
import copy
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY = PACKAGE_ROOT.parents[1]
SCRIPT = PACKAGE_ROOT / "isaac/d38999_keyed_v2_hand_camera_probe.py"
CONFIG = PACKAGE_ROOT / "config/d38999_keyed_v2_hand_camera_probe_v1.yaml"


def _source():
    return SCRIPT.read_text(encoding="utf-8")


def _module():
    return runpy.run_path(str(SCRIPT), run_name="keyed_v2_hand_camera_probe_cpu_test")


def _contract():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_import_is_lazy_and_does_not_load_isaac_or_pxr():
    before = set(sys.modules)
    module = _module()
    imported = set(sys.modules) - before

    assert module["PROBE_SCOPE"] == "SIMULATION_ONLY_FIXED_HAND_CAMERA_PROBE"
    assert not any(
        name == "isaacsim" or name.startswith(("isaacsim.", "omni.", "pxr"))
        for name in imported
    )


def test_contract_loads_exact_handarm_and_r2_keyed_identity():
    module = _module()
    contract = module["load_probe_contract"](CONFIG)

    assert contract["assets"]["robot"].endswith("handarm/handarm.usda")
    assert contract["assets"]["keyed_pair"].endswith(
        "keyed_v2_contact_offset_r2/"
        "d38999_shell25j_25_61_n_keyed_public_spec_v2.usda"
    )
    assert contract["assets"]["keyed_asset_root_prim"] == (
        "/World/D38999Shell25JKeyedPublicSpecV2"
    )
    assert contract["assets"]["keyed_model_id"] == (
        "d38999_26kj61sn_keyed_proxy_v2"
    )
    assert contract["directory_contract"]["existing_output_policy"] == (
        "REFUSE_OVERWRITE"
    )


def test_v5_camera_mounts_and_formal_optics_are_explicit_and_exact():
    module = _module()
    contract = module["load_probe_contract"](CONFIG)
    rig = contract["camera_rig"]

    assert rig["mount_contract"] == "SIM_VISUAL_MOUNT_CANDIDATE_FIXED_T_HC_V5"
    assert rig["resolution_px"] == [1280, 720]
    assert rig["channels_exactly"] == ["rgb", "distance_to_image_plane"]
    assert rig["focal_length_mm"] == 24.0
    assert rig["horizontal_aperture_mm"] == 20.955
    assert rig["vertical_aperture_mm"] == pytest.approx(20.955 * 720 / 1280)
    for key in ("palm", "wrist"):
        camera = rig[key]
        derived = module["camera_cv_pose_from_eye_target"](
            camera["v5_eye_handbase_m"], camera["v5_target_handbase_m"]
        )
        assert np.allclose(derived, camera["T_HC_cv"], atol=1e-14)
        assert np.linalg.det(derived[:3, :3]) == pytest.approx(1.0)
        assert camera["prim_suffix"] in ("/PalmCamera", "/WristCamera")


def test_mount_residual_recovers_cv_transform_from_usd_camera_pose():
    module = _module()
    t_hc = module["camera_cv_pose_from_eye_target"](
        (-0.150, 0.0, 0.060), (-0.090, 0.0, 0.480)
    )
    angle = 0.37
    t_wh = np.eye(4)
    t_wh[:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t_wh[:3, 3] = (0.4, -0.2, 0.6)
    t_wu = t_wh @ module["camera_cv_pose_to_usd"](t_hc)

    result = module["mount_residual"](t_hc, t_wh, t_wu)

    assert result["translation_m"] < 1e-12
    assert result["rotation_rad"] < 1e-7
    assert np.allclose(result["observed_T_HC_cv"], t_hc, atol=1e-12)


def test_two_fresh_episodes_freeze_static_and_safe_three_segment_probes():
    module = _module()
    contract = module["load_probe_contract"](CONFIG)
    episodes = contract["episodes"]

    assert [item["id"] for item in episodes] == [
        "stationary_pregrasp_repeat",
        "home_safe_three_segment_to_pregrasp",
    ]
    assert all(item["fresh_stage"] is True for item in episodes)
    assert episodes[0]["initial_arm_pose"] == "pregrasp"
    assert episodes[0]["capture_endpoints"] == [
        "pregrasp_repeat_00", "pregrasp_repeat_01"
    ]
    assert episodes[0]["rgbd_quality_required_endpoints"] == [
        "pregrasp_repeat_00", "pregrasp_repeat_01"
    ]
    assert episodes[1]["initial_arm_pose"] == "home"
    assert episodes[1]["rgbd_quality_required_endpoints"] == [
        "high_approach_to_pregrasp"
    ]
    assert len(contract["motion"]["approach_segments"]) == 3
    schedule = module["build_precommitted_motion_schedule"](contract)
    assert [item["name"] for item in schedule] == [
        "home_to_safe_mid",
        "safe_mid_to_high_approach",
        "high_approach_to_pregrasp",
    ]
    assert [len(item["commands"]) for item in schedule] == [1488, 888, 600]
    assert np.allclose(schedule[0]["commands"][0], np.zeros(7), atol=1e-8)
    assert np.allclose(schedule[-1]["commands"][-1], schedule[-1]["target"])


def test_minimum_jerk_has_zero_end_slopes_and_rejects_bad_input():
    minimum_jerk = _module()["minimum_jerk_arm"]
    start = np.zeros(7)
    target = np.ones(7)

    assert np.array_equal(minimum_jerk(start, target, 0.0), start)
    assert np.array_equal(minimum_jerk(start, target, 1.0), target)
    assert np.allclose(minimum_jerk(start, target, 0.5), 0.5)
    with pytest.raises(ValueError, match="exactly 7"):
        minimum_jerk(np.zeros(6), target, 0.5)
    with pytest.raises(ValueError, match="lie in"):
        minimum_jerk(start, target, 1.1)


def test_gravity_config_is_converted_to_isaac_scalar_z_value():
    gravity_z_scalar = _module()["gravity_z_scalar"]
    assert gravity_z_scalar([0.0, 0.0, 0.0]) == 0.0
    assert gravity_z_scalar([0.0, 0.0, -9.81]) == -9.81
    with pytest.raises(ValueError, match="Z axis"):
        gravity_z_scalar([0.1, 0.0, -9.81])
    with pytest.raises(ValueError, match="XYZ vector"):
        gravity_z_scalar([0.0, -9.81])


def test_rgbd_quality_allows_empty_home_but_requires_pregrasp_observation():
    evaluate = _module()["evaluate_rgbd_quality"]
    thresholds = {
        "minimum_rgb_dynamic_range": 8,
        "minimum_rgb_standard_deviation": 1.0,
        "minimum_valid_depth_pixels": 100,
    }

    def endpoint(name, good):
        camera = {
            "rgb_dynamic_range": 20 if good else 0,
            "rgb_standard_deviation": 4.0 if good else 0.0,
            "valid_depth_pixels": 1000 if good else 0,
        }
        return {"name": name, "cameras": {"palm": camera, "wrist": camera}}

    result = evaluate(
        [endpoint("home", False), endpoint("high_approach_to_pregrasp", True)],
        ["high_approach_to_pregrasp"],
        thresholds,
    )
    assert result["passed"] is True
    assert result["quality_by_captured_endpoint"] == {
        "home": False,
        "high_approach_to_pregrasp": True,
    }
    with pytest.raises(ValueError, match="not captured"):
        evaluate([endpoint("home", False)], ["pregrasp"], thresholds)


def test_failure_report_sanitizes_nonfinite_values_without_hiding_them():
    json_safe = _module()["json_safe"]
    assert json_safe({"bad": np.nan, "good": 1.0}) == {
        "bad": None,
        "good": 1.0,
    }


def test_integrated_palm_shadow_input_keeps_occlusion_unknown():
    from kcg_connector.d38999_key_shadow_pipeline import (
        run_palm_key_shadow_pipeline,
    )

    derive = _module()["derive_unknown_occlusion_shadow_inputs"]
    depth = np.full((720, 1280), np.inf, dtype=np.float32)
    depth[250:470, 500:780] = 0.12

    inputs = derive(depth)

    assert inputs["connector_face_mask"].dtype == np.bool_
    assert inputs["connector_face_mask"].shape == depth.shape
    assert inputs["occlusion_mask"] is None
    assert inputs["diagnostics"]["occlusion_status"] == "UNKNOWN_FAIL_CLOSED"
    assert inputs["diagnostics"]["semantic_or_object_truth_used"] is False
    result = run_palm_key_shadow_pipeline(
        inputs["connector_face_mask"],
        depth,
        inputs["face_center_uv"],
        ((1.0, 0.0), (-1.0, 0.0)),
        "d38999_26kj61sn_keyed_proxy_v2",
        occlusion_mask=inputs["occlusion_mask"],
    )
    assert result["rejection_code"] == "KEY_REGION_OCCLUSION_UNKNOWN"
    assert result["control_authorized"] is False
    assert result["selected_for_control_allowed"] is False


def test_control_and_truth_firewall_are_fail_closed():
    module = _module()
    contract = module["load_probe_contract"](CONFIG)
    authorization = contract["authorization"]
    assert all(
        authorization[key] is False
        for key in (
            "control_authorized",
            "visual_control_authorized",
            "insertion_control_authorized",
            "grasp_authorized",
            "selected_for_control_allowed",
        )
    )
    assert contract["capture"]["wrist_shadow_allowed"] is False
    assert contract["capture"]["palm_shadow_expected_rejection_code"] == (
        "KEY_REGION_OCCLUSION_UNKNOWN"
    )
    forbidden = set(contract["truth_firewall"]["forbidden_observations"])
    assert {
        "semantic_segmentation_truth",
        "object_pose_truth",
        "contact_report",
        "collider_identity",
    } <= forbidden
    assert contract["truth_firewall"]["image_or_truth_feedback_changes_motion"] is False


def test_contract_rejects_camera_or_authorization_drift(tmp_path):
    module = _module()
    original = _contract()
    for name, mutate, match in (
        (
            "camera",
            lambda document: document["camera_rig"].update(resolution_px=[640, 360]),
            "1280x720",
        ),
        (
            "authorization",
            lambda document: document["authorization"].update(
                insertion_control_authorized=True
            ),
            "authorization",
        ),
        (
            "trajectory",
            lambda document: document["motion"]["approach_segments"][0].update(
                duration_s=1.0
            ),
            "duration",
        ),
    ):
        changed = copy.deepcopy(original)
        mutate(changed)
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            module["load_probe_contract"](path)


def test_output_directory_refuses_overwrite(tmp_path):
    resolve = _module()["safe_new_output_dir"]
    output = tmp_path / "new_probe"
    assert resolve(output) == output.resolve()
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        resolve(output)


def test_static_runtime_authors_camera_once_and_only_requests_rgbd():
    source = _source()
    tree = ast.parse(source)
    annotators = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_annotator"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            annotators.append(node.args[0].value)

    assert annotators == ["rgb", "distance_to_image_plane"]
    assert "mount_write_count" in source
    assert 'rig["mount_write_count"][name] = 1' in source
    assert "ComputeLocalToWorldTransform" in source
    assert "import run_palm_key_shadow_pipeline" in source
    assert "shadow_pipeline(" in source
    assert 'name == "palm"' in source
    assert 'name == "wrist"' not in source
    assert "SingleRigidPrim" not in source
    assert "get_world_pose(" not in source
    assert "get_contact" not in source
    assert "get_semantics" not in source
    assert "CreateRigidBodyEnabledAttr().Set(False)" in source
    assert 'joint.SetActive(False)' in source
    assert 'positions[arm_indices] = initial_arm' in source
    assert 'positions[hand_indices]' not in source
    assert 'hand_open_settle_s' in source
    assert "json_safe(report)" in source


def test_static_probe_has_two_new_stages_and_no_grasp_or_insertion_command():
    source = _source()
    run_episode = source.split("def _run_episode(", 1)[1].split(
        "def main(", 1
    )[0]

    assert "World.clear_instance()" in run_episode
    assert "get_context().new_stage()" in run_episode
    assert "for index, episode in enumerate(contract[\"episodes\"])" in source
    assert "build_precommitted_motion_schedule(contract)" in source
    assert "grasp_hand" not in source
    assert "insertion_target" not in source
    assert '"grasp_or_insertion_entered": False' in source
    assert '"control_authorized": False' in source
