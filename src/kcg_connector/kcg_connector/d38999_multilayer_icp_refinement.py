"""Bounded, truth-free point-to-point ICP for both D38999 C2 branches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from kcg_connector.d38999_key_branch_selector import BRANCH_IDS
from kcg_connector.d38999_multilayer_coarse_registration import (
    CoarseRegistrationResult,
    SCHEMA_VERSION as COARSE_SCHEMA_VERSION,
)


SCHEMA_VERSION = "kcg_d38999_multilayer_icp_refinement_v1"
MAXIMUM_ITERATIONS = 20
MINIMUM_CORRESPONDENCES = 8
MAXIMUM_CORRESPONDENCE_DISTANCE_M = 0.020
MAXIMUM_STEP_ROTATION_RAD = math.radians(8.0)
MAXIMUM_TOTAL_ROTATION_RAD = math.radians(8.0)
MAXIMUM_TOTAL_TRANSLATION_M = 0.020

FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-C6-COARSE-REGISTRATION/"
    "COARSE_REGISTRATION_CONTRACT_MANIFEST.json": (
        "66162716fcfaeff6396b4324e2631b35bdec1e1dcf6d0312026b76534a79b1b8"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C6-COARSE-REGISTRATION/"
    "TASK_RESULT.json": (
        "d0b9b3a805dd7664209b6c6da8e06677d391d756c2470ef0bcbb2a5dec87d1f5"
    ),
    "src/kcg_connector/kcg_connector/"
    "d38999_multilayer_coarse_registration.py": (
        "2977802a6a97513d86c329f75fc3892edeb8967f717f1398d54e8021a23032be"
    ),
    "src/kcg_connector/kcg_connector/d38999_cad_registration.py": (
        "8097c5ee387eb224a49372e129d2ceb2208b9b7ea6487b559d4745ef0c3eaa11"
    ),
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
}


class IcpRefinementError(ValueError):
    """A fail-closed ICP rejection with a stable code."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class IcpRefinementResult:
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
            raise ValueError(f"frozen ICP source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen ICP source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return tuple(rows)


def build_multilayer_icp_refinement_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    c6 = _json_mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C6-COARSE-REGISTRATION/"
        "COARSE_REGISTRATION_CONTRACT_MANIFEST.json",
        "C6 coarse-registration contract",
    )
    c6_result = _json_mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C6-COARSE-REGISTRATION/"
        "TASK_RESULT.json",
        "C6 task result",
    )
    if (
        c6.get("schema_version") != COARSE_SCHEMA_VERSION
        or c6.get("status") != "OFFLINE_PASS"
        or c6.get("candidate_count") != 2
        or c6.get("candidate_branch_ids") != list(BRANCH_IDS)
        or c6.get("parameter_search_allowed") is not False
        or c6.get("current_readiness", {}).get("selected_for_control") is not None
        or c6.get("dynamic_registration_pass_claimed") is not False
        or c6_result.get("outcome") != "OFFLINE_PASS"
        or c6_result.get("dynamic_registration_pass_claimed") is not False
    ):
        raise ValueError("C6 evidence boundary changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PASS",
        "classification": "BOUNDED_TRUTH_FREE_C2_ICP_INTERFACE",
        "input_contract": COARSE_SCHEMA_VERSION,
        "input_candidate_branch_ids": list(BRANCH_IDS),
        "output_candidate_count": 2,
        "algorithm": "POINT_TO_POINT_NEAREST_NEIGHBOR_KABSCH",
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "minimum_correspondences": MINIMUM_CORRESPONDENCES,
        "maximum_correspondence_distance_m": MAXIMUM_CORRESPONDENCE_DISTANCE_M,
        "maximum_step_rotation_rad": MAXIMUM_STEP_ROTATION_RAD,
        "maximum_total_rotation_rad": MAXIMUM_TOTAL_ROTATION_RAD,
        "maximum_total_translation_m": MAXIMUM_TOTAL_TRANSLATION_M,
        "limit_semantics": "IMPLEMENTATION_SAFETY_BOUND_NOT_FORMAL_ACCEPTANCE_THRESHOLD",
        "parameter_search_allowed": False,
        "branch_selection_allowed": False,
        "confidence_calibrated": False,
        "truth_firewall": {
            "semantic_mask_allowed": False,
            "object_pose_allowed": False,
            "contact_report_allowed": False,
            "collider_identity_allowed": False,
            "contact_normal_allowed": False,
            "event_truth_allowed": False,
            "postrun_object_pose_write_allowed": False,
        },
        "current_readiness": {
            "offline_fixture_refinement_allowed": True,
            "dynamic_pointclouds_available": 0,
            "selected_for_control": None,
            "dynamic_icp_pass_claimed": False,
        },
        "sources": list(sources),
        "simulation_started": False,
        "dynamic_icp_pass_claimed": False,
        "hardware_authorized": False,
    }


def _points(value: Any, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise IcpRefinementError("INVALID_POINT_SHAPE", f"{label} must be Nx3")
    if len(points) < MINIMUM_CORRESPONDENCES:
        raise IcpRefinementError(
            "INSUFFICIENT_POINTS", f"{label} has too few points"
        )
    if not np.all(np.isfinite(points)):
        raise IcpRefinementError("NONFINITE_POINTS", f"{label} is non-finite")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    points = np.ascontiguousarray(points[order], dtype="<f8")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise IcpRefinementError("RANK_DEGENERATE", f"{label} has rank below two")
    return points


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(float(cosine)))


def _transform(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise IcpRefinementError("INVALID_TRANSFORM", f"{label} must be finite 4x4")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12):
        raise IcpRefinementError("INVALID_TRANSFORM", f"{label} homogeneous row differs")
    rotation = transform[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9)
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-9
    ):
        raise IcpRefinementError("INVALID_TRANSFORM", f"{label} rotation is not SO(3)")
    return transform.copy()


def _validate_coarse_result(
    coarse_result: CoarseRegistrationResult,
) -> tuple[str, tuple[np.ndarray, np.ndarray]]:
    if not isinstance(coarse_result, CoarseRegistrationResult):
        raise IcpRefinementError(
            "INVALID_COARSE_RESULT", "coarse_result must be the C6 result type"
        )
    summary = coarse_result.summary
    if (
        summary.get("schema_version") != COARSE_SCHEMA_VERSION
        or summary.get("candidate_count") != 2
        or summary.get("candidate_branch_ids") != list(BRANCH_IDS)
        or summary.get("selected_for_control") is not None
        or summary.get("dynamic_registration_pass_claimed") is not False
        or summary.get("object_pose_truth_used") is not False
        or summary.get("contact_truth_used") is not False
        or summary.get("event_truth_used") is not False
        or summary.get("semantic_mask_used") is not False
    ):
        raise IcpRefinementError("INVALID_COARSE_RESULT", "C6 summary boundary changed")
    if [item.get("branch_id") for item in coarse_result.candidates] != list(BRANCH_IDS):
        raise IcpRefinementError("INVALID_C2_BRANCHES", "branch identities differ")
    transforms = []
    forbidden = (
        "ground_truth",
        "object_pose",
        "semantic_mask",
        "contact_name",
        "contact_normal",
        "collider",
        "event_truth",
    )
    for index, candidate in enumerate(coarse_result.candidates):
        if not isinstance(candidate, Mapping):
            raise IcpRefinementError("INVALID_COARSE_RESULT", "candidate is not a mapping")
        if any(any(token in str(key).lower() for token in forbidden) for key in candidate):
            raise IcpRefinementError("TRUTH_FIELD_REJECTED", "candidate contains truth field")
        if candidate.get("selected_for_control") is not False:
            raise IcpRefinementError("CONTROL_SELECTION_REJECTED", "C6 branch was selected")
        transforms.append(_transform(candidate.get("T_camera_model"), f"candidate {index}"))
    rz_pi = np.diag((-1.0, -1.0, 1.0))
    if (
        not np.allclose(transforms[1][:3, :3], transforms[0][:3, :3] @ rz_pi, atol=1.0e-8)
        or not np.allclose(transforms[1][:3, 3], transforms[0][:3, 3], atol=1.0e-8)
    ):
        raise IcpRefinementError("INVALID_C2_RELATION", "coarse branches are not pi-linked")
    frame_id = summary.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        raise IcpRefinementError("INVALID_FRAME_ID", "C6 frame_id is missing")
    return frame_id, (transforms[0], transforms[1])


def _matches(
    source: np.ndarray,
    observed_tree: cKDTree,
    observed: np.ndarray,
    transform: np.ndarray,
    maximum_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    transformed = source @ transform[:3, :3].T + transform[:3, 3]
    distances, indices = observed_tree.query(transformed, k=1)
    keep = distances <= maximum_distance
    if int(np.count_nonzero(keep)) < MINIMUM_CORRESPONDENCES:
        raise IcpRefinementError(
            "INSUFFICIENT_CORRESPONDENCES",
            f"only {int(np.count_nonzero(keep))} points are within the bound",
        )
    source_match = transformed[keep]
    target_match = observed[indices[keep]]
    rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
    return source_match, target_match, keep, rmse


def _refine_branch(
    source: np.ndarray,
    observed: np.ndarray,
    observed_tree: cKDTree,
    initial: np.ndarray,
    maximum_distance: float,
) -> dict[str, Any]:
    transform = initial.copy()
    _, _, initial_keep, initial_rmse = _matches(
        source, observed_tree, observed, transform, maximum_distance
    )
    numerical_tolerance = math.sqrt(np.finfo(np.float64).eps)
    converged = False
    iterations = 0
    for iteration in range(1, MAXIMUM_ITERATIONS + 1):
        current, target, _, _ = _matches(
            source, observed_tree, observed, transform, maximum_distance
        )
        current_center = current.mean(axis=0)
        target_center = target.mean(axis=0)
        current_zero = current - current_center
        target_zero = target - target_center
        cross = current_zero.T @ target_zero
        left, singular, right_t = np.linalg.svd(cross)
        singular_tolerance = (
            np.finfo(np.float64).eps
            * max(current.shape)
            * max(1.0, float(singular[0]))
            * 16.0
        )
        if singular[1] <= singular_tolerance:
            raise IcpRefinementError(
                "CORRESPONDENCE_GEOMETRY_DEGENERATE", "matched points have rank below two"
            )
        correction_rotation = right_t.T @ left.T
        if np.linalg.det(correction_rotation) < 0.0:
            right_t[-1, :] *= -1.0
            correction_rotation = right_t.T @ left.T
        correction_translation = target_center - correction_rotation @ current_center
        step_rotation = _rotation_angle(correction_rotation)
        step_centroid_motion = float(
            np.linalg.norm(
                correction_rotation @ current_center
                + correction_translation
                - current_center
            )
        )
        if (
            step_rotation > MAXIMUM_STEP_ROTATION_RAD
            or step_centroid_motion > MAXIMUM_CORRESPONDENCE_DISTANCE_M
        ):
            raise IcpRefinementError("CORRECTION_STEP_OUT_OF_BOUNDS", "ICP step is too large")
        correction = np.eye(4)
        correction[:3, :3] = correction_rotation
        correction[:3, 3] = correction_translation
        transform = correction @ transform
        total_rotation = _rotation_angle(transform[:3, :3] @ initial[:3, :3].T)
        total_translation = float(np.linalg.norm(transform[:3, 3] - initial[:3, 3]))
        if (
            total_rotation > MAXIMUM_TOTAL_ROTATION_RAD
            or total_translation > MAXIMUM_TOTAL_TRANSLATION_M
        ):
            raise IcpRefinementError("TOTAL_CORRECTION_OUT_OF_BOUNDS", "ICP left local basin")
        iterations = iteration
        if step_rotation <= numerical_tolerance and step_centroid_motion <= numerical_tolerance:
            converged = True
            break
    if not converged:
        raise IcpRefinementError("ITERATION_LIMIT_REACHED", "ICP did not numerically converge")
    _, _, final_keep, final_rmse = _matches(
        source, observed_tree, observed, transform, maximum_distance
    )
    if final_rmse > initial_rmse + numerical_tolerance:
        raise IcpRefinementError("RESIDUAL_INCREASED", "nearest-neighbour RMSE increased")
    return {
        "T_camera_model": transform.tolist(),
        "initial_rmse_m": initial_rmse,
        "final_rmse_m": final_rmse,
        "rmse_improvement_m": initial_rmse - final_rmse,
        "initial_correspondence_count": int(np.count_nonzero(initial_keep)),
        "final_correspondence_count": int(np.count_nonzero(final_keep)),
        "iterations": iterations,
        "converged": True,
        "total_correction_rotation_rad": _rotation_angle(
            transform[:3, :3] @ initial[:3, :3].T
        ),
        "total_correction_translation_m": float(
            np.linalg.norm(transform[:3, 3] - initial[:3, 3])
        ),
        "selected_for_control": False,
    }


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def refine_c2_candidates_icp(
    cad_points_model_m: Any,
    observed_points_camera_m: Any,
    coarse_result: CoarseRegistrationResult,
    *,
    maximum_correspondence_distance_m: float,
) -> IcpRefinementResult:
    """Refine both C2 candidates without selecting or authorizing either."""

    source = _points(cad_points_model_m, "cad_points_model_m")
    observed = _points(observed_points_camera_m, "observed_points_camera_m")
    frame_id, initial_transforms = _validate_coarse_result(coarse_result)
    maximum_distance = float(maximum_correspondence_distance_m)
    if (
        not math.isfinite(maximum_distance)
        or maximum_distance <= 0.0
        or maximum_distance > MAXIMUM_CORRESPONDENCE_DISTANCE_M
    ):
        raise IcpRefinementError(
            "INVALID_CORRESPONDENCE_BOUND",
            f"maximum_correspondence_distance_m must be in (0, {MAXIMUM_CORRESPONDENCE_DISTANCE_M}]",
        )
    tree = cKDTree(observed)
    refined = []
    for branch_id, initial in zip(BRANCH_IDS, initial_transforms):
        candidate = _refine_branch(source, observed, tree, initial, maximum_distance)
        candidate["branch_id"] = branch_id
        refined.append(candidate)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PASS",
        "classification": "C2_ICP_CANDIDATES_UNCALIBRATED",
        "evidence_level": "OFFLINE_FIXTURE_OR_CALLER_SUPPLIED_POINTCLOUD",
        "frame_id": frame_id,
        "point_frame": "camera_optical",
        "model_frame": "plug_datum_B",
        "algorithm": "POINT_TO_POINT_NEAREST_NEIGHBOR_KABSCH",
        "candidate_count": 2,
        "candidate_branch_ids": list(BRANCH_IDS),
        "selected_for_control": None,
        "confidence_calibrated": False,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "maximum_correspondence_distance_m": maximum_distance,
        "parameter_search_used": False,
        "cad_points_sha256": _array_sha256(source),
        "observed_points_sha256": _array_sha256(observed),
        "semantic_mask_used": False,
        "object_pose_truth_used": False,
        "contact_truth_used": False,
        "event_truth_used": False,
        "postrun_object_pose_write_count": 0,
        "dynamic_icp_pass_claimed": False,
        "control_authorized": False,
    }
    return IcpRefinementResult(candidates=(refined[0], refined[1]), summary=summary)


__all__ = [
    "FROZEN_SOURCES",
    "IcpRefinementError",
    "IcpRefinementResult",
    "MAXIMUM_CORRESPONDENCE_DISTANCE_M",
    "MAXIMUM_ITERATIONS",
    "SCHEMA_VERSION",
    "build_multilayer_icp_refinement_contract",
    "refine_c2_candidates_icp",
]
