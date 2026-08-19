from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_wrist_camera_interface import (
    FROZEN_SOURCES,
    build_multilayer_wrist_camera_interface,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _interface():
    return build_multilayer_wrist_camera_interface(REPOSITORY_ROOT)


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_current_multilayer_wrist_interface_is_traceable():
    interface = _interface()
    assert interface["status"] == "OFFLINE_PASS"
    assert len(interface["sources"]) == 5
    assert all(len(row["sha256"]) == 64 for row in interface["sources"])
    assert interface["observation_target"] == {
        "representation": "D38999_VISUAL_COMPLETE_V1",
        "root": "/World/D38999MultilayerV1/D38999_VISUAL_COMPLETE_V1",
        "pair_root": (
            "/World/D38999MultilayerV1/"
            "D38999_VISUAL_COMPLETE_V1/D38999Pair"
        ),
        "visible_geometry_preserved": True,
    }


def test_wrist_mount_and_pose_remain_exact_v5_values():
    interface = _interface()
    assert interface["interface_role"] == "WRIST_LAYOUT_EVIDENCE_ONLY"
    assert interface["parent_frame"] == "handbase_link"
    assert interface["camera_prim"].endswith("/handbase_link/WristCamera")
    assert interface["v5_eye_handbase_m"] == [-0.150, 0.0, 0.060]
    assert interface["v5_target_handbase_m"] == [-0.090, 0.0, 0.480]
    assert interface["T_HC_cv"] == [
        [0.9899494936611665, 0.0, 0.1414213562373095, -0.150],
        [0.0, 1.0, 0.0, 0.0],
        [-0.1414213562373095, 0.0, 0.9899494936611665, 0.060],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_wrist_optics_and_channels_remain_exact():
    interface = _interface()
    assert interface["resolution_px"] == [1280, 720]
    assert interface["channels_exactly"] == [
        "rgb",
        "distance_to_image_plane",
    ]
    assert interface["optics"] == {
        "focal_length_mm": 24.0,
        "horizontal_aperture_mm": 20.955,
        "vertical_aperture_mm": 11.7871875,
        "clipping_range_m": [0.02, 10.0],
    }


def test_layout_capture_does_not_authorize_shadow_or_control():
    policy = _interface()["layout_evidence_policy"]
    assert policy == {
        "rgbd_capture_for_layout_evidence_allowed": True,
        "shadow_inference_allowed": False,
        "control_input_allowed": False,
        "selected_for_control_allowed": False,
        "human_review_required": True,
        "source_wrist_shadow_allowed": False,
    }


def test_camera_is_runtime_nonphysical_and_authored_once():
    authoring = _interface()["runtime_authoring"]
    assert authoring == {
        "camera_embedded_in_robot_asset": False,
        "create_before_physics_start": True,
        "mount_write_count_required": 1,
        "physics_rigid_body_api_allowed": False,
        "physics_collision_api_allowed": False,
        "robot_asset_write_allowed": False,
    }


def test_truth_and_authorization_stay_fail_closed():
    interface = _interface()
    forbidden = set(interface["truth_firewall"]["forbidden_observations"])
    assert {
        "semantic_segmentation_truth",
        "object_pose_truth",
        "contact_report",
        "collider_identity",
    } <= forbidden
    assert all(
        value is False
        for key, value in interface["authorization"].items()
        if key != "simulation_only"
    )
    assert interface["authorization"]["simulation_only"] is True
    assert interface["palm_camera_exposed"] is False
    assert interface["legacy_connector_identity_carried_forward"] is False


def test_interface_never_claims_render_shadow_or_dynamic_vision():
    interface = _interface()
    assert interface["simulation_started"] is False
    assert interface["render_capture_performed"] is False
    assert interface["wrist_shadow_pass_claimed"] is False
    assert interface["dynamic_camera_pass_claimed"] is False
    assert interface["visual_pose_pass_claimed"] is False
    json.dumps(interface, allow_nan=False)


def test_frozen_source_tamper_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    config = (
        root
        / "src/kcg_connector/config/"
        "d38999_keyed_v2_hand_camera_probe_v1.yaml"
    )
    config.write_text(
        config.read_text().replace(
            "role: WRIST_LAYOUT_EVIDENCE_ONLY",
            "role: WRIST_CONTROL_INPUT",
        )
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_wrist_camera_interface(root)


def test_missing_current_robot_asset_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    robot = (
        root
        / "artifacts/kcg_connector/isaac/robot/"
        "handarm_keyed_v3_physical_r7/handarm.usda"
    )
    robot.unlink()
    with pytest.raises(ValueError, match="source missing"):
        build_multilayer_wrist_camera_interface(root)


def test_public_builder_accepts_no_pose_contact_or_image_inputs():
    names = set(inspect.signature(build_multilayer_wrist_camera_interface).parameters)
    assert names == {"repository_root"}
    assert names.isdisjoint(
        {
            "object_pose",
            "contact_name",
            "contact_normal",
            "event_truth",
            "rgb",
            "depth",
        }
    )
