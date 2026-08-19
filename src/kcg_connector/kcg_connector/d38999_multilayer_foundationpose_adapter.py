"""Static, fail-closed FoundationPose adapter for the multilayer model.

This module does not import or execute FoundationPose.  It binds the existing
disabled bootstrap to the current RGB-D/model contracts, validates a future
image-derived input envelope, and expands one external pose estimate into the
two unselected C2 hypotheses.  It never authorizes inference or control.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from kcg_connector.d38999_foundationpose_bootstrap import (
    evaluate_foundationpose_readiness,
    load_foundationpose_bootstrap_contract,
)
from kcg_connector.d38999_key_branch_selector import BRANCH_IDS


SCHEMA_VERSION = "kcg_d38999_multilayer_foundationpose_adapter_v1"
MASK_PROVENANCE = "IMAGE_DERIVED_INSTANCE_SEGMENTATION"
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9_./:-]{1,256}$")

FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-C4-RGBD-SAVE/"
    "ARCHIVE_CONTRACT_MANIFEST.json": (
        "9c076b68373beb6b62595f3032533add2bcc5a80523bd20cb059e808d6b371b1"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C7-ICP-REFINEMENT/"
    "ICP_REFINEMENT_CONTRACT_MANIFEST.json": (
        "e0136c0c3386705533c3bd7880e3ead8409ce769bee23c5e0f2251191375bb44"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C7-ICP-REFINEMENT/"
    "TASK_RESULT.json": (
        "bbc70c70e503c242c23a6ac58d02d23c311757ad94cddb48c2f6879483139824"
    ),
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json": (
        "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783"
    ),
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/config/d38999_foundationpose_bootstrap_v1.yaml": (
        "4c38b8dc50e8b7f534c9d0057f9a6e5cd67826a50487d550e5d283792f7ff7c0"
    ),
    "src/kcg_connector/kcg_connector/d38999_foundationpose_bootstrap.py": (
        "47879b1891416c220c8833923359c95acdf7585ae7089f97e43e57f5ee6cf63b"
    ),
    "artifacts/kcg_connector/foundationpose_1.0.1_onnx_local_v1/"
    "model_manifest.json": (
        "6806688e4c1718f06267bc63f582a64b9dc9c2f72052f66d24670f3841ff5dd5"
    ),
}


class FoundationPoseAdapterError(ValueError):
    """A fail-closed adapter rejection with a stable code."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class FoundationPoseInputEnvelope:
    rgb: np.ndarray
    depth_m: np.ndarray
    instance_mask: np.ndarray
    camera_intrinsics: dict[str, float]
    summary: dict[str, Any]


@dataclass(frozen=True)
class FoundationPoseC2AdapterResult:
    candidates: tuple[dict[str, Any], dict[str, Any]]
    summary: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _verified_sources(root: Path) -> tuple[dict[str, str], ...]:
    rows = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen FoundationPose adapter source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen FoundationPose adapter source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return tuple(rows)


def build_multilayer_foundationpose_adapter_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    c4 = _json_mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C4-RGBD-SAVE/"
        "ARCHIVE_CONTRACT_MANIFEST.json",
        "C4 RGB-D contract",
    )
    c7 = _json_mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C7-ICP-REFINEMENT/"
        "ICP_REFINEMENT_CONTRACT_MANIFEST.json",
        "C7 ICP contract",
    )
    c7_result = _json_mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C7-ICP-REFINEMENT/"
        "TASK_RESULT.json",
        "C7 task result",
    )
    mapping = _json_mapping(
        root
        / "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
        "MODEL_MAPPING.json",
        "multilayer model mapping",
    )
    if (
        c4.get("status") != "OFFLINE_PASS"
        or c4.get("raw_capture", {}).get("channels_exactly")
        != ["rgb", "distance_to_image_plane"]
        or c4.get("current_readiness", {}).get("dynamic_capture_archives_available") != 0
        or c4.get("dynamic_rgbd_pass_claimed") is not False
    ):
        raise ValueError("C4 RGB-D evidence boundary changed")
    if (
        c7.get("status") != "OFFLINE_PASS"
        or c7.get("output_candidate_count") != 2
        or c7.get("branch_selection_allowed") is not False
        or c7.get("dynamic_icp_pass_claimed") is not False
        or c7_result.get("outcome") != "OFFLINE_PASS"
        or c7_result.get("dynamic_icp_pass_claimed") is not False
    ):
        raise ValueError("C7 ICP evidence boundary changed")
    try:
        visual = mapping["representations"]["D38999_VISUAL_COMPLETE_V1"]
    except (KeyError, TypeError):
        raise ValueError("multilayer visual mapping missing") from None
    if visual.get("visible_geometry_preserved") is not True:
        raise ValueError("visual-complete geometry preservation changed")

    bootstrap_path = (
        root / "src/kcg_connector/config/d38999_foundationpose_bootstrap_v1.yaml"
    )
    bootstrap = load_foundationpose_bootstrap_contract(bootstrap_path)
    readiness = evaluate_foundationpose_readiness(bootstrap, root)
    models_verified = sum(
        item.get("verified") is True for item in readiness["models"].values()
    )
    meshes_verified = sum(
        item.get("verified") is True for item in readiness["meshes"].values()
    )
    blockers = list(readiness["blockers"])
    blockers.extend(
        (
            "current_multilayer_visual_obj_export_not_verified",
            "image_derived_instance_mask_bridge_not_available",
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "STATIC_PASS_RUNTIME_BLOCKED_EXTERNAL",
        "classification": "FOUNDATIONPOSE_ADAPTER_STATIC_READY_ONLY",
        "input_rgbd_contract": c4["schema_version"],
        "input_channels": ["rgb", "distance_to_image_plane"],
        "required_additional_input": "image_derived_instance_mask",
        "allowed_mask_provenance": MASK_PROVENANCE,
        "simulator_semantic_mask_allowed": False,
        "visual_representation": "D38999_VISUAL_COMPLETE_V1",
        "visual_root": visual["root"],
        "visual_pair_root": visual["pair_root"],
        "visible_geometry_preserved": True,
        "current_multilayer_obj_mesh_verified": False,
        "legacy_bootstrap": {
            "schema_version": readiness["schema_version"],
            "status": readiness["status"],
            "config_enabled": readiness["config_enabled"],
            "onnx_models_verified": models_verified,
            "legacy_proxy_meshes_verified": meshes_verified,
            "artifact_bundle_verified": readiness["gates"]["artifact_bundle_verified"],
            "runtime_environment_ready": readiness["gates"]["runtime_environment_ready"],
            "foundationpose_inference_ready": readiness["gates"]["foundationpose_inference_ready"],
            "host_os": readiness["host_runtime"]["os"],
            "container_runtime_available": readiness["host_runtime"]["container_runtime_available"],
            "isaac_ros_foundationpose_package_available": readiness["host_runtime"]["isaac_ros_foundationpose_package_available"],
            "engine_plans_exist": all(item["exists"] for item in readiness["engines"].values()),
        },
        "output_adapter": {
            "candidate_count": 2,
            "branch_ids": list(BRANCH_IDS),
            "c2_expansion": "ESTIMATE_AND_RIGHT_MULTIPLY_MODEL_RZ_PI",
            "selected_for_control": None,
            "confidence_calibrated": False,
        },
        "blockers": blockers,
        "inference_command": None,
        "inference_execution_authorized": False,
        "parameter_search_allowed": False,
        "truth_firewall": {
            "simulator_semantic_mask_allowed": False,
            "ground_truth_object_pose_allowed": False,
            "contact_report_allowed": False,
            "collider_identity_allowed": False,
            "contact_normal_allowed": False,
            "event_truth_allowed": False,
            "postrun_object_pose_write_allowed": False,
        },
        "current_readiness": {
            "adapter_static_interface_ready": True,
            "dynamic_rgbd_archives_available": 0,
            "foundationpose_inference_runs": 0,
            "selected_for_control": None,
            "dynamic_foundationpose_pass_claimed": False,
        },
        "sources": list(sources),
        "simulation_started": False,
        "dynamic_foundationpose_pass_claimed": False,
        "hardware_authorized": False,
    }


def _intrinsics(value: Mapping[str, Any], shape: tuple[int, int]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
    }:
        raise FoundationPoseAdapterError("INVALID_INTRINSICS", "camera keys differ")
    result = {name: float(value[name]) for name in value}
    height, width = shape
    if (
        result["width"] != width
        or result["height"] != height
        or result["fx"] <= 0.0
        or result["fy"] <= 0.0
        or not all(math.isfinite(number) for number in result.values())
    ):
        raise FoundationPoseAdapterError("INVALID_INTRINSICS", "camera values differ")
    return result


def validate_foundationpose_input_envelope(
    rgb: Any,
    depth_m: Any,
    camera_intrinsics: Mapping[str, Any],
    instance_mask: Any,
    *,
    frame_id: str,
    mask_provenance: str,
) -> FoundationPoseInputEnvelope:
    """Validate future inputs without invoking or authorizing inference."""

    color = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(instance_mask)
    if depth.ndim != 2 or color.shape != (*depth.shape, 3) or mask.shape != depth.shape:
        raise FoundationPoseAdapterError("SHAPE_MISMATCH", "RGB, depth and mask differ")
    if color.dtype != np.uint8 or mask.dtype != np.bool_:
        raise FoundationPoseAdapterError("INVALID_DTYPE", "RGB must be uint8 and mask bool")
    if not isinstance(frame_id, str) or FRAME_ID_PATTERN.fullmatch(frame_id) is None:
        raise FoundationPoseAdapterError("INVALID_FRAME_ID", "frame_id is empty or unsafe")
    if mask_provenance != MASK_PROVENANCE:
        raise FoundationPoseAdapterError(
            "MASK_PROVENANCE_REJECTED",
            "only image-derived instance segmentation is allowed",
        )
    camera = _intrinsics(camera_intrinsics, depth.shape)
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    if int(np.count_nonzero(valid)) == 0:
        raise FoundationPoseAdapterError("NO_VALID_MASKED_DEPTH", "mask has no valid depth")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_INPUT_ENVELOPE_VALID",
        "frame_id": frame_id,
        "mask_provenance": MASK_PROVENANCE,
        "masked_pixel_count": int(np.count_nonzero(mask)),
        "valid_masked_depth_count": int(np.count_nonzero(valid)),
        "simulator_semantic_truth_used": False,
        "ground_truth_object_pose_used": False,
        "inference_performed": False,
        "control_authorized": False,
    }
    return FoundationPoseInputEnvelope(
        rgb=np.ascontiguousarray(color),
        depth_m=np.ascontiguousarray(depth),
        instance_mask=np.ascontiguousarray(mask),
        camera_intrinsics=camera,
        summary=summary,
    )


def _transform(value: Any) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise FoundationPoseAdapterError("INVALID_ESTIMATE", "estimate must be finite 4x4")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12):
        raise FoundationPoseAdapterError("INVALID_ESTIMATE", "homogeneous row differs")
    rotation = transform[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9)
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-9
    ):
        raise FoundationPoseAdapterError("INVALID_ESTIMATE", "rotation is not SO(3)")
    return transform.copy()


def adapt_foundationpose_estimate_to_c2(
    foundationpose_estimate_T_camera_model: Any,
    *,
    frame_id: str,
    inference_provenance: str,
) -> FoundationPoseC2AdapterResult:
    """Expand one untrusted estimate into two unselected C2 hypotheses."""

    if not isinstance(frame_id, str) or FRAME_ID_PATTERN.fullmatch(frame_id) is None:
        raise FoundationPoseAdapterError("INVALID_FRAME_ID", "frame_id is empty or unsafe")
    if inference_provenance not in {"OFFLINE_FIXTURE_ONLY", "EXTERNAL_UNVERIFIED"}:
        raise FoundationPoseAdapterError(
            "INFERENCE_PROVENANCE_REJECTED", "inference provenance is not allowed"
        )
    base = _transform(foundationpose_estimate_T_camera_model)
    rz_pi = np.diag((-1.0, -1.0, 1.0))
    second = base.copy()
    second[:3, :3] = base[:3, :3] @ rz_pi
    candidates = tuple(
        {
            "branch_id": branch_id,
            "T_camera_model": transform.tolist(),
            "selected_for_control": False,
        }
        for branch_id, transform in zip(BRANCH_IDS, (base, second))
    )
    return FoundationPoseC2AdapterResult(
        candidates=(candidates[0], candidates[1]),
        summary={
            "schema_version": SCHEMA_VERSION,
            "status": "C2_ADAPTER_OUTPUT_UNCALIBRATED",
            "frame_id": frame_id,
            "inference_provenance": inference_provenance,
            "candidate_count": 2,
            "candidate_branch_ids": list(BRANCH_IDS),
            "selected_for_control": None,
            "confidence_calibrated": False,
            "ground_truth_object_pose_used": False,
            "contact_truth_used": False,
            "dynamic_foundationpose_pass_claimed": False,
            "control_authorized": False,
        },
    )


__all__ = [
    "FROZEN_SOURCES",
    "FoundationPoseAdapterError",
    "FoundationPoseC2AdapterResult",
    "FoundationPoseInputEnvelope",
    "MASK_PROVENANCE",
    "SCHEMA_VERSION",
    "adapt_foundationpose_estimate_to_c2",
    "build_multilayer_foundationpose_adapter_contract",
    "validate_foundationpose_input_envelope",
]
