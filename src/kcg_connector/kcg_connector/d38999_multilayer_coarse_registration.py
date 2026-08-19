"""Truth-free, C2-aware coarse registration for the multilayer D38999 model.

The model and observed point sets are unpaired.  A covariance eigensystem is
used only to recover the insertion axis.  The frozen CAD is laterally
isotropic at second order, so yaw is recovered analytically from its non-zero
6th and 20th angular harmonics.  Because ``7*6 - 2*20 == 2``, the combined
phase has exactly the two C2 solutions separated by pi.  No search, semantic
mask, object pose or contact truth is accepted.
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
import yaml

from kcg_connector.d38999_cad_registration import shell25j_cad_profile_metadata
from kcg_connector.d38999_key_branch_selector import BRANCH_IDS


SCHEMA_VERSION = "kcg_d38999_multilayer_coarse_registration_v1"
HARMONIC_ORDERS = (6, 20)
BEZOUT_COEFFICIENTS = (7, -2)
MINIMUM_POINT_COUNT = 8
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9_./:-]{1,256}$")

FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-C5-OPEN3D-PREPROCESS/"
    "PREPROCESS_CONTRACT_MANIFEST.json": (
        "68ecd1766587d813523a30be655c208fbbefd17dd469c8e9d06b4c49aec9e0c8"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C5-OPEN3D-PREPROCESS/"
    "TASK_RESULT.json": (
        "6b0f6fbe4f9bd4bbecb836970369145a0a1c832f82ff741a2c2175037e892922"
    ),
    "src/kcg_connector/kcg_connector/"
    "d38999_multilayer_pointcloud_preprocess.py": (
        "a21553d8556a13e509a1341b72ba9b02b6f78063480bbbd994baca182a3de897"
    ),
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/kcg_connector/d38999_cad_registration.py": (
        "8097c5ee387eb224a49372e129d2ceb2208b9b7ea6487b559d4745ef0c3eaa11"
    ),
    "src/kcg_connector/kcg_connector/postgrasp_shadow_estimator.py": (
        "b164a83fa5039d39664cbfa6ac4bca1ca7976158f019f3fe174de71afaaa6a8e"
    ),
    "src/kcg_connector/kcg_connector/d38999_key_branch_selector.py": (
        "aeac14722045418a942223d08863b27a99539c1dc192a0440e21789f6b93d09c"
    ),
}


class CoarseRegistrationError(ValueError):
    """A fail-closed coarse-registration rejection with a stable code."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class CoarseRegistrationResult:
    candidates: tuple[dict[str, Any], dict[str, Any]]
    summary: dict[str, Any]


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
            raise ValueError(f"frozen coarse-registration source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen coarse-registration source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return tuple(rows)


def build_multilayer_coarse_registration_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    c5 = _mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C5-OPEN3D-PREPROCESS/"
        "PREPROCESS_CONTRACT_MANIFEST.json",
        "C5 preprocess contract",
    )
    c5_result = _mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C5-OPEN3D-PREPROCESS/"
        "TASK_RESULT.json",
        "C5 task result",
    )
    master = _mapping(
        root / "src/kcg_connector/config/d38999_master_model_contract_v1.yaml",
        "master model contract",
    )
    try:
        axis = master["coordinate_frames"]["assembly_axis"]
        plug_frame = master["coordinate_frames"]["plug_datum_B"]
    except (KeyError, TypeError):
        raise ValueError("coarse-registration coordinate authority missing") from None
    if (
        c5.get("status") != "OFFLINE_PASS"
        or c5.get("output_frame_semantics") != "INPUT_CAMERA_OPTICAL_FRAME"
        or c5.get("current_readiness", {}).get("dynamic_archives_available") != 0
        or c5.get("dynamic_pointcloud_pass_claimed") is not False
        or c5_result.get("outcome") != "OFFLINE_PASS"
        or c5_result.get("dynamic_pointcloud_pass_claimed") is not False
    ):
        raise ValueError("C5 evidence boundary changed")
    firewall = c5.get("truth_firewall", {})
    if any(value is not False for value in firewall.values()):
        raise ValueError("C5 truth firewall changed")
    if (
        axis.get("receptacle_local_vector") != [0.0, 0.0, 1.0]
        or plug_frame.get("local_plus_x")
        != "main_key_centerline_zero_degrees"
    ):
        raise ValueError("master insertion-axis or key-zero convention changed")
    metadata = shell25j_cad_profile_metadata()
    if metadata.get("symmetry_order") != 2 or tuple(BRANCH_IDS) != (
        "C2_LINKED_BRANCH_0",
        "C2_LINKED_BRANCH_PI",
    ):
        raise ValueError("C2 CAD or branch authority changed")
    if (
        math.gcd(*HARMONIC_ORDERS) != 2
        or sum(
            coefficient * order
            for coefficient, order in zip(
                BEZOUT_COEFFICIENTS, HARMONIC_ORDERS
            )
        )
        != 2
    ):
        raise ValueError("C2 harmonic identity changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PASS",
        "classification": "TRUTH_FREE_C2_COARSE_REGISTRATION_INTERFACE",
        "input_contract": c5["schema_version"],
        "input_frame_semantics": "ONE_DECLARED_CAMERA_OPTICAL_FRAME",
        "model_frame": "plug_datum_B",
        "model_insertion_axis": [0.0, 0.0, 1.0],
        "axis_direction_convention": "OBSERVED_AXIS_POSITIVE_CAMERA_OPTICAL_Z",
        "cad_profile_id": metadata["profile_id"],
        "cad_symmetry_order": metadata["symmetry_order"],
        "harmonic_orders": list(HARMONIC_ORDERS),
        "bezout_coefficients": list(BEZOUT_COEFFICIENTS),
        "bezout_identity": "7*6-2*20=2",
        "candidate_branch_ids": list(BRANCH_IDS),
        "candidate_count": 2,
        "parameter_search_allowed": False,
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
            "offline_fixture_registration_allowed": True,
            "dynamic_pointclouds_available": 0,
            "selected_for_control": None,
            "dynamic_registration_pass_claimed": False,
        },
        "sources": list(sources),
        "simulation_started": False,
        "dynamic_registration_pass_claimed": False,
        "hardware_authorized": False,
    }


def _points(value: Any, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise CoarseRegistrationError("INVALID_POINT_SHAPE", f"{label} must be Nx3")
    if len(points) < MINIMUM_POINT_COUNT:
        raise CoarseRegistrationError(
            "INSUFFICIENT_POINTS",
            f"{label} has {len(points)} points; need {MINIMUM_POINT_COUNT}",
        )
    if not np.all(np.isfinite(points)):
        raise CoarseRegistrationError("NONFINITE_POINTS", f"{label} is non-finite")
    # Point clouds are sets.  Canonical row ordering makes their centroid,
    # covariance, angular moments and evidence hash independent of upstream
    # pixel/voxel traversal order before any floating-point reduction occurs.
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return np.ascontiguousarray(points[order], dtype="<f8")


def _covariance_axis(points: np.ndarray, *, model: bool) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / float(len(points))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = (
        np.finfo(np.float64).eps
        * max(points.shape)
        * max(1.0, float(eigenvalues[-1]))
        * 16.0
    )
    if eigenvalues[0] <= tolerance:
        raise CoarseRegistrationError(
            "RANK_DEGENERATE", "point covariance is not full rank"
        )
    if model:
        model_axis = np.asarray((0.0, 0.0, 1.0))
        axis_index = int(np.argmax(np.abs(eigenvectors.T @ model_axis)))
        axis = eigenvectors[:, axis_index]
        if float(axis @ model_axis) < 0.0:
            axis = -axis
        lateral = [index for index in range(3) if index != axis_index]
        if min(abs(eigenvalues[index] - eigenvalues[axis_index]) for index in lateral) <= tolerance:
            raise CoarseRegistrationError(
                "INSERTION_AXIS_DEGENERATE", "model insertion-axis eigenvalue is not unique"
            )
        if 1.0 - float(axis @ model_axis) > 1024.0 * np.finfo(np.float64).eps:
            raise CoarseRegistrationError(
                "MODEL_AXIS_MISALIGNED", "model points are not aligned to plug_datum_B +Z"
            )
        return eigenvalues, axis
    gaps = np.asarray(
        [
            min(abs(eigenvalues[index] - eigenvalues[other]) for other in range(3) if other != index)
            for index in range(3)
        ]
    )
    axis_index = int(np.argmax(gaps))
    axis = eigenvectors[:, axis_index]
    if axis[2] < 0.0:
        axis = -axis
    if abs(float(axis[2])) <= 64.0 * np.finfo(np.float64).eps:
        raise CoarseRegistrationError(
            "AXIS_DIRECTION_UNRESOLVED",
            "insertion axis is perpendicular to camera optical +Z convention",
        )
    return eigenvalues, axis


def _transverse_basis(axis: np.ndarray) -> np.ndarray:
    canonical = np.eye(3)[int(np.argmin(np.abs(axis)))]
    x_axis = canonical - float(canonical @ axis) * axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    return np.column_stack((x_axis, y_axis, axis))


def _angular_moment(
    points: np.ndarray, center: np.ndarray, basis: np.ndarray, order: int
) -> complex:
    local = (points - center) @ basis
    transverse = local[:, 0] + 1j * local[:, 1]
    radius = np.abs(transverse)
    usable = radius > np.finfo(np.float64).eps
    if int(np.count_nonzero(usable)) < MINIMUM_POINT_COUNT:
        raise CoarseRegistrationError(
            "TRANSVERSE_SUPPORT_DEGENERATE", "too few off-axis points"
        )
    return complex(np.mean((transverse[usable] / radius[usable]) ** order))


def _rz(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def coarse_register_c2(
    cad_points_model_m: Any,
    observed_points_camera_m: Any,
    *,
    frame_id: str,
) -> CoarseRegistrationResult:
    """Return two unselected C2 coarse candidates in the declared camera frame."""

    if not isinstance(frame_id, str) or FRAME_ID_PATTERN.fullmatch(frame_id) is None:
        raise CoarseRegistrationError("INVALID_FRAME_ID", "frame_id is empty or unsafe")
    source = _points(cad_points_model_m, "cad_points_model_m")
    observed = _points(observed_points_camera_m, "observed_points_camera_m")
    source_center = source.mean(axis=0)
    observed_center = observed.mean(axis=0)
    source_eigenvalues, source_axis = _covariance_axis(source, model=True)
    observed_eigenvalues, observed_axis = _covariance_axis(observed, model=False)
    source_basis = np.eye(3, dtype=np.float64)
    observed_basis = _transverse_basis(observed_axis)

    source_moments = {
        order: _angular_moment(source, source_center, source_basis, order)
        for order in HARMONIC_ORDERS
    }
    observed_moments = {
        order: _angular_moment(observed, observed_center, observed_basis, order)
        for order in HARMONIC_ORDERS
    }
    numerical_floor = 256.0 * np.finfo(np.float64).eps
    for order in HARMONIC_ORDERS:
        if (
            abs(source_moments[order]) <= numerical_floor
            or abs(observed_moments[order]) <= numerical_floor
        ):
            raise CoarseRegistrationError(
                "C2_HARMONIC_DEGENERATE",
                f"angular harmonic {order} has no numerical phase",
            )
    phase_ratios = {
        order: (observed_moments[order] / source_moments[order])
        / abs(observed_moments[order] / source_moments[order])
        for order in HARMONIC_ORDERS
    }
    c2_phase = (
        phase_ratios[HARMONIC_ORDERS[0]] ** BEZOUT_COEFFICIENTS[0]
        * np.conj(phase_ratios[HARMONIC_ORDERS[1]])
        ** abs(BEZOUT_COEFFICIENTS[1])
    )
    base_yaw = 0.5 * math.atan2(float(c2_phase.imag), float(c2_phase.real))

    candidates = []
    for branch_id, yaw_offset in zip(BRANCH_IDS, (0.0, math.pi)):
        yaw = base_yaw + yaw_offset
        rotation = observed_basis @ _rz(yaw) @ source_basis.T
        translation = observed_center - rotation @ source_center
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        harmonic_phase_residuals = {}
        for order in HARMONIC_ORDERS:
            predicted = complex(math.cos(order * yaw), math.sin(order * yaw))
            residual = phase_ratios[order] * np.conj(predicted)
            harmonic_phase_residuals[str(order)] = math.atan2(
                float(residual.imag), float(residual.real)
            )
        candidates.append(
            {
                "branch_id": branch_id,
                "yaw_about_model_insertion_axis_rad": yaw,
                "T_camera_model": transform.tolist(),
                "rotation_model_to_camera": rotation.tolist(),
                "translation_camera_m": translation.tolist(),
                "model_insertion_axis_camera": (rotation @ source_axis).tolist(),
                "centroid_alignment_error_m": float(
                    np.linalg.norm(rotation @ source_center + translation - observed_center)
                ),
                "harmonic_phase_residual_rad": harmonic_phase_residuals,
                "selected_for_control": False,
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PASS",
        "classification": "C2_COARSE_CANDIDATES_UNCALIBRATED",
        "evidence_level": "OFFLINE_FIXTURE_OR_CALLER_SUPPLIED_POINTCLOUD",
        "frame_id": frame_id,
        "point_frame": "camera_optical",
        "model_frame": "plug_datum_B",
        "axis_direction_convention": "OBSERVED_AXIS_POSITIVE_CAMERA_OPTICAL_Z",
        "candidate_count": 2,
        "candidate_branch_ids": list(BRANCH_IDS),
        "selected_for_control": None,
        "confidence_calibrated": False,
        "harmonic_orders": list(HARMONIC_ORDERS),
        "bezout_coefficients": list(BEZOUT_COEFFICIENTS),
        "parameter_search_used": False,
        "source_point_count": int(len(source)),
        "observed_point_count": int(len(observed)),
        "source_covariance_eigenvalues_m2": source_eigenvalues.tolist(),
        "observed_covariance_eigenvalues_m2": observed_eigenvalues.tolist(),
        "source_harmonic_magnitudes": {
            str(order): abs(source_moments[order]) for order in HARMONIC_ORDERS
        },
        "observed_harmonic_magnitudes": {
            str(order): abs(observed_moments[order]) for order in HARMONIC_ORDERS
        },
        "cad_points_sha256": _array_sha256(source),
        "observed_points_sha256": _array_sha256(observed),
        "semantic_mask_used": False,
        "object_pose_truth_used": False,
        "contact_truth_used": False,
        "event_truth_used": False,
        "postrun_object_pose_write_count": 0,
        "dynamic_registration_pass_claimed": False,
        "control_authorized": False,
    }
    return CoarseRegistrationResult(
        candidates=(candidates[0], candidates[1]), summary=summary
    )


__all__ = [
    "BEZOUT_COEFFICIENTS",
    "CoarseRegistrationError",
    "CoarseRegistrationResult",
    "FROZEN_SOURCES",
    "HARMONIC_ORDERS",
    "SCHEMA_VERSION",
    "build_multilayer_coarse_registration_contract",
    "coarse_register_c2",
]
