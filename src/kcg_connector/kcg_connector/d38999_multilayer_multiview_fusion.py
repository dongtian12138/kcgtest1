"""Truth-free equal-weight fusion of independent C2 pose observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from kcg_connector.d38999_key_branch_selector import BRANCH_IDS
from kcg_connector.postgrasp_shadow_estimator import ALL_THRESHOLDS_CANDIDATE


SCHEMA_VERSION = "kcg_d38999_multilayer_multiview_fusion_v1"
MAXIMUM_TRANSLATION_SPREAD_M = float(
    ALL_THRESHOLDS_CANDIDATE["multistart_translation_envelope_m"]
)
MAXIMUM_ROTATION_SPREAD_RAD = float(
    ALL_THRESHOLDS_CANDIDATE["multistart_rotation_envelope_rad"]
)
THRESHOLD_LABEL = str(ALL_THRESHOLDS_CANDIDATE["threshold_label"])
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:/-]{1,256}$")

FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-C3-POSTGRASP-VIEW-PLAN/"
    "VIEW_PLAN_MANIFEST.json": (
        "64eecf1a5e1ac04fd129453b042507e25ab2fbfe170fc035e7c00bcaba23921e"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C3-POSTGRASP-VIEW-PLAN/"
    "TASK_RESULT.json": (
        "c55c2d12559653d0457cb717fb466d7dffaf07940b55c7b67e3f2c7dc0d61a7c"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C7-ICP-REFINEMENT/"
    "TASK_RESULT.json": (
        "bbc70c70e503c242c23a6ac58d02d23c311757ad94cddb48c2f6879483139824"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C8-FOUNDATIONPOSE-ADAPTER/"
    "ADAPTER_CONTRACT_MANIFEST.json": (
        "0750bee2a7811308a8d4f28b12d50caf51c91dc677337631880aaa185b6072c7"
    ),
    "src/kcg_connector/kcg_connector/postgrasp_shadow_estimator.py": (
        "b164a83fa5039d39664cbfa6ac4bca1ca7976158f019f3fe174de71afaaa6a8e"
    ),
    "src/kcg_connector/kcg_connector/d38999_inhand_multiview.py": (
        "2848647aeb668850aa11b70d400b6da99d2e2e7d2efa4d8010820147057563ea"
    ),
    "src/kcg_connector/kcg_connector/d38999_key_branch_selector.py": (
        "aeac14722045418a942223d08863b27a99539c1dc192a0440e21789f6b93d09c"
    ),
}


class MultiviewFusionError(ValueError):
    """A fail-closed fusion rejection with a stable code."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class MultiviewC2Observation:
    view_id: str
    independence_id: str
    capture_batch_id: str
    timestamp_utc: str
    frame_id: str
    candidates: tuple[Mapping[str, Any], Mapping[str, Any]]
    evidence_level: str = "OFFLINE_FIXTURE"


@dataclass(frozen=True)
class MultiviewFusionResult:
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
            raise ValueError(f"frozen fusion source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen fusion source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return tuple(rows)


def build_multilayer_multiview_fusion_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    c3 = _json_mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C3-POSTGRASP-VIEW-PLAN/"
        "TASK_RESULT.json",
        "C3 view-plan result",
    )
    c7 = _json_mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C7-ICP-REFINEMENT/"
        "TASK_RESULT.json",
        "C7 ICP result",
    )
    c8 = _json_mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C8-FOUNDATIONPOSE-ADAPTER/"
        "ADAPTER_CONTRACT_MANIFEST.json",
        "C8 adapter contract",
    )
    if (
        c3.get("outcome") != "STATIC_PASS"
        or c3.get("candidate_view_count") != 3
        or c3.get("current_multilayer_dynamic_views_proven") != 0
        or c3.get("dynamic_visual_pass_claimed") is not False
    ):
        raise ValueError("C3 view evidence boundary changed")
    if (
        c7.get("outcome") != "OFFLINE_PASS"
        or c7.get("candidate_count") != 2
        or c7.get("dynamic_icp_pass_claimed") is not False
    ):
        raise ValueError("C7 candidate evidence boundary changed")
    if (
        c8.get("status") != "STATIC_PASS_RUNTIME_BLOCKED_EXTERNAL"
        or c8.get("output_adapter", {}).get("candidate_count") != 2
        or c8.get("output_adapter", {}).get("selected_for_control") is not None
        or c8.get("dynamic_foundationpose_pass_claimed") is not False
    ):
        raise ValueError("C8 adapter evidence boundary changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PASS",
        "classification": "TRUTH_FREE_C2_MULTIVIEW_FUSION_INTERFACE",
        "minimum_independent_views": 2,
        "candidate_branch_ids": list(BRANCH_IDS),
        "fusion_policy": "EQUAL_WEIGHT_PER_BRANCH_MARKLEY_ROTATION_AND_TRANSLATION_MEAN",
        "cross_branch_fusion_allowed": False,
        "automatic_outlier_removal_allowed": False,
        "maximum_translation_spread_m": MAXIMUM_TRANSLATION_SPREAD_M,
        "maximum_rotation_spread_rad": MAXIMUM_ROTATION_SPREAD_RAD,
        "threshold_label": THRESHOLD_LABEL,
        "threshold_semantics": "EXISTING_SIM_TUNING_CANDIDATE_NOT_FORMAL_ACCEPTANCE",
        "parameter_search_allowed": False,
        "confidence_calibrated": False,
        "independence_policy": {
            "view_id_unique": True,
            "independence_id_unique": True,
            "capture_batch_id_equal": True,
            "frame_id_equal": True,
            "timestamps_unique_utc": True,
            "comoving_repeated_camera_counts_once": True,
        },
        "truth_firewall": {
            "ground_truth_object_pose_allowed": False,
            "semantic_truth_allowed": False,
            "contact_report_allowed": False,
            "collider_identity_allowed": False,
            "contact_normal_allowed": False,
            "event_truth_allowed": False,
            "postrun_object_pose_write_allowed": False,
        },
        "current_readiness": {
            "offline_fixture_fusion_allowed": True,
            "dynamic_independent_views_proven": 0,
            "selected_for_control": None,
            "dynamic_multiview_fusion_pass_claimed": False,
        },
        "sources": list(sources),
        "simulation_started": False,
        "dynamic_multiview_fusion_pass_claimed": False,
        "hardware_authorized": False,
    }


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise MultiviewFusionError("INVALID_IDENTIFIER", f"{label} is empty or unsafe")
    return value


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise MultiviewFusionError("INVALID_TIMESTAMP", "timestamp must end in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise MultiviewFusionError("INVALID_TIMESTAMP", "timestamp is not ISO-8601") from exc
    return parsed


def _transform(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise MultiviewFusionError("INVALID_TRANSFORM", f"{label} must be finite 4x4")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12):
        raise MultiviewFusionError("INVALID_TRANSFORM", f"{label} homogeneous row differs")
    rotation = transform[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9)
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-9
    ):
        raise MultiviewFusionError("INVALID_TRANSFORM", f"{label} rotation is not SO(3)")
    return transform.copy()


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(float(cosine)))


def _average_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    quaternions = Rotation.from_matrix(np.asarray(rotations)).as_quat()
    accumulator = sum(np.outer(value, value) for value in quaternions)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return Rotation.from_quat(quaternion).as_matrix()


def _validate_observations(
    observations: Sequence[MultiviewC2Observation],
) -> tuple[list[MultiviewC2Observation], list[tuple[np.ndarray, np.ndarray]]]:
    if isinstance(observations, (str, bytes)) or len(observations) < 2:
        raise MultiviewFusionError("INSUFFICIENT_INDEPENDENT_VIEWS", "need at least two views")
    values = list(observations)
    if not all(isinstance(value, MultiviewC2Observation) for value in values):
        raise MultiviewFusionError("INVALID_OBSERVATION", "wrong observation type")
    view_ids = [_identifier(value.view_id, "view_id") for value in values]
    independence_ids = [
        _identifier(value.independence_id, "independence_id") for value in values
    ]
    batches = [_identifier(value.capture_batch_id, "capture_batch_id") for value in values]
    frames = [_identifier(value.frame_id, "frame_id") for value in values]
    timestamps = [_timestamp(value.timestamp_utc) for value in values]
    if len(set(view_ids)) != len(view_ids):
        raise MultiviewFusionError("DUPLICATE_VIEW", "view_id is repeated")
    if len(set(independence_ids)) != len(independence_ids):
        raise MultiviewFusionError(
            "NONINDEPENDENT_VIEW_REUSE", "independence_id is repeated"
        )
    if len(set(batches)) != 1:
        raise MultiviewFusionError("CAPTURE_BATCH_MISMATCH", "capture batches differ")
    if len(set(frames)) != 1:
        raise MultiviewFusionError("FRAME_MISMATCH", "pose frames differ")
    if len(set(timestamps)) != len(timestamps):
        raise MultiviewFusionError("DUPLICATE_TIMESTAMP", "timestamps are repeated")
    transforms = []
    forbidden = (
        "ground_truth",
        "object_pose",
        "semantic_truth",
        "contact_name",
        "contact_normal",
        "collider",
        "event_truth",
    )
    for view in values:
        if view.evidence_level not in {"OFFLINE_FIXTURE", "DYNAMIC_SENSOR_UNVERIFIED"}:
            raise MultiviewFusionError("INVALID_EVIDENCE_LEVEL", "view evidence differs")
        if [item.get("branch_id") for item in view.candidates] != list(BRANCH_IDS):
            raise MultiviewFusionError("INVALID_C2_BRANCHES", "branch identities differ")
        branch_transforms = []
        for index, candidate in enumerate(view.candidates):
            if any(any(token in str(key).lower() for token in forbidden) for key in candidate):
                raise MultiviewFusionError("TRUTH_FIELD_REJECTED", "candidate contains truth")
            if candidate.get("selected_for_control") is not False:
                raise MultiviewFusionError("CONTROL_SELECTION_REJECTED", "branch was selected")
            branch_transforms.append(
                _transform(candidate.get("T_camera_model"), f"{view.view_id}[{index}]")
            )
        rz_pi = np.diag((-1.0, -1.0, 1.0))
        if (
            not np.allclose(branch_transforms[1][:3, :3], branch_transforms[0][:3, :3] @ rz_pi, atol=1.0e-8)
            or not np.allclose(branch_transforms[1][:3, 3], branch_transforms[0][:3, 3], atol=1.0e-8)
        ):
            raise MultiviewFusionError("INVALID_C2_RELATION", "view branches are not pi-linked")
        transforms.append((branch_transforms[0], branch_transforms[1]))
    order = sorted(range(len(values)), key=lambda index: timestamps[index])
    return [values[index] for index in order], [transforms[index] for index in order]


def fuse_c2_multiview(
    observations: Sequence[MultiviewC2Observation],
) -> MultiviewFusionResult:
    """Fuse like-named branches only; reject rather than remove outliers."""

    views, transforms = _validate_observations(observations)
    fused_candidates = []
    branch_diagnostics = {}
    for branch_index, branch_id in enumerate(BRANCH_IDS):
        branch = [value[branch_index] for value in transforms]
        translations = np.asarray([value[:3, 3] for value in branch])
        rotations = [value[:3, :3] for value in branch]
        translation_spreads = [
            float(np.linalg.norm(translations[first] - translations[second]))
            for first in range(len(branch))
            for second in range(first + 1, len(branch))
        ]
        rotation_spreads = [
            _rotation_distance(rotations[first], rotations[second])
            for first in range(len(branch))
            for second in range(first + 1, len(branch))
        ]
        maximum_translation = max(translation_spreads)
        maximum_rotation = max(rotation_spreads)
        if maximum_translation > MAXIMUM_TRANSLATION_SPREAD_M:
            raise MultiviewFusionError(
                "TRANSLATION_OUTLIER_SET_REJECTED", "view set exceeds translation envelope"
            )
        if maximum_rotation > MAXIMUM_ROTATION_SPREAD_RAD:
            raise MultiviewFusionError(
                "ROTATION_OUTLIER_SET_REJECTED", "view set exceeds rotation envelope"
            )
        fused = np.eye(4)
        fused[:3, :3] = _average_rotation(rotations)
        fused[:3, 3] = translations.mean(axis=0)
        residual_translation = [
            float(np.linalg.norm(value - fused[:3, 3])) for value in translations
        ]
        residual_rotation = [
            _rotation_distance(value, fused[:3, :3]) for value in rotations
        ]
        fused_candidates.append(
            {
                "branch_id": branch_id,
                "T_camera_model": fused.tolist(),
                "selected_for_control": False,
            }
        )
        branch_diagnostics[branch_id] = {
            "maximum_pairwise_translation_spread_m": maximum_translation,
            "maximum_pairwise_rotation_spread_rad": maximum_rotation,
            "translation_residual_rms_m": float(np.sqrt(np.mean(np.square(residual_translation)))),
            "rotation_residual_rms_rad": float(np.sqrt(np.mean(np.square(residual_rotation)))),
        }
    rz_pi = np.diag((-1.0, -1.0, 1.0))
    first = np.asarray(fused_candidates[0]["T_camera_model"])
    second = np.asarray(fused_candidates[1]["T_camera_model"])
    if (
        not np.allclose(second[:3, :3], first[:3, :3] @ rz_pi, atol=1.0e-8)
        or not np.allclose(second[:3, 3], first[:3, 3], atol=1.0e-8)
    ):
        raise MultiviewFusionError("FUSED_C2_RELATION_LOST", "fusion broke pi relation")
    evidence_levels = sorted({view.evidence_level for view in views})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PASS",
        "classification": "C2_MULTIVIEW_FUSION_UNCALIBRATED",
        "view_count": len(views),
        "view_ids": [view.view_id for view in views],
        "independence_ids": [view.independence_id for view in views],
        "capture_batch_id": views[0].capture_batch_id,
        "frame_id": views[0].frame_id,
        "timestamps_utc": [view.timestamp_utc for view in views],
        "input_evidence_levels": evidence_levels,
        "candidate_count": 2,
        "candidate_branch_ids": list(BRANCH_IDS),
        "branch_diagnostics": branch_diagnostics,
        "threshold_label": THRESHOLD_LABEL,
        "automatic_outlier_removal_used": False,
        "cross_branch_fusion_used": False,
        "selected_for_control": None,
        "confidence_calibrated": False,
        "ground_truth_object_pose_used": False,
        "semantic_truth_used": False,
        "contact_truth_used": False,
        "event_truth_used": False,
        "postrun_object_pose_write_count": 0,
        "dynamic_multiview_fusion_pass_claimed": False,
        "control_authorized": False,
    }
    return MultiviewFusionResult(
        candidates=(fused_candidates[0], fused_candidates[1]), summary=summary
    )


__all__ = [
    "FROZEN_SOURCES",
    "MAXIMUM_ROTATION_SPREAD_RAD",
    "MAXIMUM_TRANSLATION_SPREAD_M",
    "MultiviewC2Observation",
    "MultiviewFusionError",
    "MultiviewFusionResult",
    "SCHEMA_VERSION",
    "build_multilayer_multiview_fusion_contract",
    "fuse_c2_multiview",
]
