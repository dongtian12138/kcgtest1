"""Static Palm RGB-D interface for the frozen D38999 multilayer model.

The module carries only the already frozen V5 Palm mount and optics onto the
current keyed-v3 robot and visual representation paths.  It performs no USD
write, rendering, object-pose readback, or controller authorization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_palm_camera_interface_v1"
PALM_MOUNT_CONTRACT = "SIM_VISUAL_MOUNT_CANDIDATE_FIXED_T_HC_V5"
ROBOT_STAGE_ROOT = "/World/HandArm"
HANDBASE_SUFFIX = (
    "/Geometry/world/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3/"
    "iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7/iiwa_link_ee/"
    "handbase_link"
)
PALM_PRIM_SUFFIX = "/PalmCamera"

FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_keyed_v2_hand_camera_probe_v1.yaml": (
        "bea07afbc0e2fb3e45b8677a184791292ca83267d6cda66c739b71fa628357ec"
    ),
    "src/kcg_connector/isaac/d38999_keyed_v2_hand_camera_probe.py": (
        "feea0487bb8b5797750ede0e83d4862abf658590647cf7ffa0d61ca8de492cdb"
    ),
    "artifacts/kcg_connector/isaac/robot/handarm_keyed_v3_physical_r7/handarm.usda": (
        "29ec40f5e52f29ad14d0dae5b841ce5702dd9c901ada964661b16dbcb74148d8"
    ),
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json": (
        "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783"
    ),
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
    "D38999_VISUAL_COMPLETE_V1.usda": (
        "69fe6dc3ca9caace8bb26cd0cfad68c0eb84111f09697da6068cd91802d65c0a"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _verified_sources(root: Path) -> tuple[dict[str, str], ...]:
    rows = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen Palm source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen Palm source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return tuple(rows)


def build_multilayer_palm_camera_interface(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    config = _mapping(
        root / "src/kcg_connector/config/"
        "d38999_keyed_v2_hand_camera_probe_v1.yaml",
        "frozen hand camera contract",
    )
    model_mapping = _mapping(
        root / "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
        "MODEL_MAPPING.json",
        "multilayer mapping",
    )
    robot_text = (
        root
        / "artifacts/kcg_connector/isaac/robot/"
        "handarm_keyed_v3_physical_r7/handarm.usda"
    ).read_text(encoding="utf-8")
    visual_text = (
        root
        / "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
        "D38999_VISUAL_COMPLETE_V1.usda"
    ).read_text(encoding="utf-8")

    rig = config.get("camera_rig")
    if not isinstance(rig, Mapping):
        raise ValueError("camera_rig missing")
    palm = rig.get("palm")
    if not isinstance(palm, Mapping):
        raise ValueError("Palm camera definition missing")
    expected_parent = ROBOT_STAGE_ROOT + HANDBASE_SUFFIX
    if (
        rig.get("mount_contract") != PALM_MOUNT_CONTRACT
        or rig.get("parent_frame") != "handbase_link"
        or config["assets"].get("handbase_prim") != expected_parent
        or palm.get("prim_suffix") != PALM_PRIM_SUFFIX
        or palm.get("role") != "PALM_RGBD_SHADOW_CAPABLE"
    ):
        raise ValueError("frozen Palm mount identity changed")
    if palm.get("T_HC_cv") != [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.315],
        [0.0, 0.0, 0.0, 1.0],
    ]:
        raise ValueError("frozen Palm T_HC changed")
    if (
        rig.get("resolution_px") != [1280, 720]
        or rig.get("channels_exactly")
        != ["rgb", "distance_to_image_plane"]
        or float(rig.get("focal_length_mm")) != 24.0
        or float(rig.get("horizontal_aperture_mm")) != 20.955
        or float(rig.get("vertical_aperture_mm")) != 11.7871875
        or rig.get("clipping_range_m") != [0.02, 10.0]
    ):
        raise ValueError("frozen Palm optics or output channels changed")

    if 'over "handbase_link" (' not in robot_text:
        raise ValueError("keyed-v3 robot handbase_link prim missing")
    if any(name in robot_text for name in ("PalmCamera", "WristCamera")):
        raise ValueError("keyed-v3 robot must not embed runtime cameras")

    try:
        visual = model_mapping["representations"]["D38999_VISUAL_COMPLETE_V1"]
    except (KeyError, TypeError):
        raise ValueError("visual representation mapping missing") from None
    if (
        visual.get("root")
        != "/World/D38999MultilayerV1/D38999_VISUAL_COMPLETE_V1"
        or visual.get("pair_root")
        != "/World/D38999MultilayerV1/D38999_VISUAL_COMPLETE_V1/D38999Pair"
        or visual.get("visible_geometry_preserved") is not True
        or visual.get("direct_reference_sha256")
        != "5eb9ad82940e58a1592b6a66fd824c480ba24268cb1c20bcc84de653bb12c995"
    ):
        raise ValueError("visual representation mapping changed")
    for token in (
        'custom string kcg:representationId = "D38999_VISUAL_COMPLETE_V1"',
        "custom bool kcg:visibleGeometryPreserved = 1",
        "custom int kcg:visiblePinCount = 61",
        "custom int kcg:visibleSocketCount = 61",
        'def Xform "D38999Pair"',
    ):
        if token not in visual_text:
            raise ValueError(f"visual representation token missing: {token}")

    truth = config.get("truth_firewall")
    authorization = config.get("authorization")
    if not isinstance(truth, Mapping) or not isinstance(authorization, Mapping):
        raise ValueError("Palm truth or authorization boundary missing")
    if truth.get("inference_channels_exactly") != [
        "rgb",
        "distance_to_image_plane",
    ]:
        raise ValueError("Palm inference channels changed")
    forbidden = set(truth.get("forbidden_observations", []))
    required_forbidden = {
        "semantic_segmentation_truth",
        "object_pose_truth",
        "keyed_object_transform_readback",
        "contact_report",
        "contact_point_truth",
        "collider_identity",
        "penetration_depth_truth",
        "physx_manifold_truth",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("Palm truth firewall weakened")
    if any(
        authorization.get(key) is not False
        for key in (
            "control_authorized",
            "visual_control_authorized",
            "insertion_control_authorized",
            "grasp_authorized",
            "selected_for_control_allowed",
            "real_mount_calibrated",
            "real_hardware_validated",
        )
    ):
        raise ValueError("Palm authorization must remain false")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PASS",
        "interface_role": "PALM_RGBD_SHADOW_CAPABLE",
        "mount_contract": PALM_MOUNT_CONTRACT,
        "robot_asset": (
            "artifacts/kcg_connector/isaac/robot/"
            "handarm_keyed_v3_physical_r7/handarm.usda"
        ),
        "parent_frame": "handbase_link",
        "parent_prim": expected_parent,
        "camera_prim": expected_parent + PALM_PRIM_SUFFIX,
        "T_HC_cv": palm["T_HC_cv"],
        "camera_cv_axes": rig["camera_cv_axes"],
        "usd_camera_forward_axis": rig["usd_camera_forward_axis"],
        "resolution_px": rig["resolution_px"],
        "channels_exactly": rig["channels_exactly"],
        "optics": {
            "focal_length_mm": rig["focal_length_mm"],
            "horizontal_aperture_mm": rig["horizontal_aperture_mm"],
            "vertical_aperture_mm": rig["vertical_aperture_mm"],
            "clipping_range_m": rig["clipping_range_m"],
        },
        "observation_target": {
            "representation": "D38999_VISUAL_COMPLETE_V1",
            "root": visual["root"],
            "pair_root": visual["pair_root"],
            "visible_geometry_preserved": True,
        },
        "runtime_authoring": {
            "camera_embedded_in_robot_asset": False,
            "create_before_physics_start": True,
            "mount_write_count_required": 1,
            "physics_rigid_body_api_allowed": False,
            "physics_collision_api_allowed": False,
            "robot_asset_write_allowed": False,
        },
        "truth_firewall": {
            "allowed_observations": list(truth["allowed_observations"]),
            "forbidden_observations": sorted(required_forbidden),
            "image_or_truth_feedback_changes_motion": False,
        },
        "authorization": {
            "control_authorized": False,
            "visual_control_authorized": False,
            "selected_for_control_allowed": False,
            "simulation_only": True,
            "real_mount_calibrated": False,
            "real_hardware_validated": False,
        },
        "wrist_camera_exposed": False,
        "legacy_connector_identity_carried_forward": False,
        "sources": list(sources),
        "simulation_started": False,
        "render_capture_performed": False,
        "dynamic_camera_pass_claimed": False,
        "visual_pose_pass_claimed": False,
    }
