from __future__ import annotations

import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_palm_camera_interface import (
    FROZEN_SOURCES,
    build_multilayer_palm_camera_interface,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _interface():
    return build_multilayer_palm_camera_interface(REPOSITORY_ROOT)


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_current_multilayer_palm_interface_is_traceable():
    interface = _interface()
    assert interface["status"] == "OFFLINE_PASS"
    assert len(interface["sources"]) == 5
    assert all(len(row["sha256"]) == 64 for row in interface["sources"])
    assert interface["observation_target"]["representation"] == (
        "D38999_VISUAL_COMPLETE_V1"
    )
    assert interface["observation_target"]["visible_geometry_preserved"] is True


def test_palm_mount_and_optics_remain_exact_v5_values():
    interface = _interface()
    assert interface["parent_frame"] == "handbase_link"
    assert interface["camera_prim"].endswith("/handbase_link/PalmCamera")
    assert interface["T_HC_cv"] == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.315],
        [0.0, 0.0, 0.0, 1.0],
    ]
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
    assert all(value is False for key, value in interface["authorization"].items() if key != "simulation_only")
    assert interface["authorization"]["simulation_only"] is True
    assert interface["wrist_camera_exposed"] is False
    assert interface["legacy_connector_identity_carried_forward"] is False


def test_interface_never_claims_render_or_dynamic_vision():
    interface = _interface()
    assert interface["simulation_started"] is False
    assert interface["render_capture_performed"] is False
    assert interface["dynamic_camera_pass_claimed"] is False
    assert interface["visual_pose_pass_claimed"] is False
    json.dumps(interface, allow_nan=False)


def test_frozen_source_tamper_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    config = root / "src/kcg_connector/config/d38999_keyed_v2_hand_camera_probe_v1.yaml"
    config.write_text(config.read_text().replace("focal_length_mm: 24.0", "focal_length_mm: 25.0"))
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_palm_camera_interface(root)


def test_missing_current_robot_asset_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    robot = root / "artifacts/kcg_connector/isaac/robot/handarm_keyed_v3_physical_r7/handarm.usda"
    robot.unlink()
    with pytest.raises(ValueError, match="source missing"):
        build_multilayer_palm_camera_interface(root)


def test_public_builder_accepts_no_pose_or_contact_inputs():
    import inspect

    names = set(inspect.signature(build_multilayer_palm_camera_interface).parameters)
    assert names == {"repository_root"}
    assert names.isdisjoint(
        {"object_pose", "contact_name", "contact_normal", "event_truth"}
    )
