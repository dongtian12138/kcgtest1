"""Shared V2 data and thin adapters to the registered object and hand models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import yaml

from kcg_connector.grasp.robust.grasp_optimizer import GraspCandidate
from kcg_connector.grasp.robust.hand_contract import (
    CARTSHandContract,
    load_carts_hand_contract,
)
from kcg_connector.grasp.robust.hand_model import ThreeFingerHandModel
from kcg_connector.grasp.robust.object_contract import (
    LoadedObjectContract,
    load_object_contract,
)


_SCHEMA = "carts_grasp_v2"
_ROLE_METHOD = "TASK_AXIS_OUTER_ENVELOPE_V1"


def rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    cosine = (float(np.trace(left.T @ right)) - 1.0) * 0.5
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def farthest_point_indices(features: np.ndarray, count: int | None = None) -> np.ndarray:
    """Return a deterministic farthest-point prefix over finite feature rows."""

    values = np.asarray(features, dtype=np.float64)
    requested = len(values) if count is None else int(count)
    if values.ndim != 2 or len(values) == 0 or not 1 <= requested <= len(values):
        raise ValueError("farthest-point input/count is invalid")
    centered = values - np.mean(values, axis=0)
    first = int(np.argmax(np.einsum("ij,ij->i", centered, centered)))
    order = np.empty(requested, dtype=np.int64)
    order[0] = first
    selected = np.zeros(len(values), dtype=np.bool_)
    selected[first] = True
    minimum_distance = np.sum((values - values[first]) ** 2, axis=1)
    for index in range(1, requested):
        minimum_distance[selected] = -1.0
        chosen = int(np.argmax(minimum_distance))
        order[index] = chosen
        selected[chosen] = True
        distance = np.sum((values - values[chosen]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return order


def _readonly(value: Any, dtype: Any, *, ndim: int | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected rank {ndim}, got {array.ndim}")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("array contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CARTSV2Config:
    path: Path
    values: Mapping[str, Any]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"missing V2 config section {name!r}")
        return value


@dataclass(frozen=True)
class FaceRoleMap:
    object_id: str
    face_is_allowed: np.ndarray
    reason_code: np.ndarray
    method: str
    allowed_area_m2: float
    total_area_m2: float

    def __post_init__(self) -> None:
        allowed = _readonly(self.face_is_allowed, np.bool_, ndim=1)
        reasons = _readonly(self.reason_code, np.uint8, ndim=1)
        if len(allowed) != len(reasons) or len(allowed) == 0:
            raise ValueError("face role arrays must have the same non-zero length")
        if np.any(allowed != (reasons == 0)):
            raise ValueError("reason code zero must uniquely mean allowed")
        if not np.any(allowed):
            raise ValueError(f"{self.object_id} has no V2 allowed grip face")
        object.__setattr__(self, "face_is_allowed", allowed)
        object.__setattr__(self, "reason_code", reasons)

    @property
    def allowed_face_indices(self) -> np.ndarray:
        result = np.flatnonzero(self.face_is_allowed)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class V2Inputs:
    config: CARTSV2Config
    object_contract: LoadedObjectContract
    hand_contract: CARTSHandContract
    hand_model: ThreeFingerHandModel
    closing_directions: np.ndarray
    face_roles: FaceRoleMap


@dataclass(frozen=True)
class CandidateSeed:
    candidate_id: str
    object_id: str
    anchor_face_index: int
    anchor_position_object_m: tuple[float, float, float]
    object_from_hand: tuple[float, ...]
    pregrasp_joint_positions_rad: tuple[float, ...]
    pregrasp_closure_phases: tuple[float, float, float]
    source_sample_index: int

    def object_from_hand_matrix(self) -> np.ndarray:
        return np.asarray(self.object_from_hand, dtype=np.float64).reshape(4, 4)


@dataclass(frozen=True)
class PredictedContact:
    pad_name: str
    object_position_m: tuple[float, float, float]
    path_local_free_side_normal_object: tuple[float, float, float]
    object_face_index: int
    phase_lower: float
    phase_upper: float
    clearance_m: float
    inward_motion_m_per_phase: float


@dataclass(frozen=True)
class ClosurePrediction:
    seed: CandidateSeed
    status: str
    contacts: tuple[PredictedContact, ...]
    final_joint_positions_rad: tuple[float, ...]
    final_closure_phases: tuple[float, float, float]
    minimum_initial_pad_clearance_m: float
    reason: str = ""
    grasp_candidate: GraspCandidate | None = None


@dataclass(frozen=True)
class FastFilterResult:
    candidate_id: str
    status: str
    reasons: tuple[str, ...]
    unresolved_checks: tuple[str, ...]


@dataclass(frozen=True)
class TaskQualityResult:
    candidate_id: str
    status: str
    scenario_margins: tuple[float | None, ...]
    worst_task_margin: float | None
    lower_tail_mean_margin: float | None
    required_peak_normal_force_n: float | None
    maximum_joint_load_utilization: float | None
    maximum_generalized_joint_torque_nm: float | None
    wrist_load_utilization: float | None
    sensitivity: float | None
    failure_reason: str = ""
    evidence_scope: str = "RESEARCH_OFFLINE_QMC_NOT_FORMAL"
    nominal_balance_infeasible_count: int = 0


@dataclass(frozen=True)
class SelectedCandidate:
    rank: int
    prediction: ClosurePrediction
    fast_filter: FastFilterResult
    task_quality: TaskQualityResult
    path_minimum_clearance_m: float | None
    offline_task_gate_passed: bool


def joint_positions_for_phases(
    inputs: V2Inputs, phases: tuple[float, float, float]
) -> np.ndarray:
    """Map three normalized finger phases through the registered actuation rows."""

    if len(phases) != 3 or any(not 0.0 <= float(value) <= 1.0 for value in phases):
        raise ValueError("three closure phases must lie in [0, 1]")
    hand = inputs.hand_model
    names = tuple(hand.independent_joint_names)
    lower = np.asarray([hand.joints[name].limit.lower for name in names])
    upper = np.asarray([hand.joints[name].limit.upper for name in names])
    positions = lower.copy()
    preshape = inputs.config.section("candidate_generation")[
        "preshape_joint_positions_rad"
    ]
    for name, value in preshape.items():
        positions[names.index(str(name))] = float(value)
    for phase, row in zip(phases, inputs.closing_directions):
        for joint_index in np.flatnonzero(row):
            span = upper[joint_index] - lower[joint_index]
            if row[joint_index] > 0.0:
                positions[joint_index] = lower[joint_index] + float(phase) * span
            else:
                positions[joint_index] = upper[joint_index] - float(phase) * span
    hand.resolve_joint_positions(positions, enforce_limits=True)
    return positions


def load_v2_config(path: Path | str) -> CARTSV2Config:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != _SCHEMA:
        raise ValueError("CARTS-Grasp V2 config schema mismatch")
    if value.get("hardware_authorized") is not False:
        raise ValueError("hardware_authorized must remain false")
    generation = value.get("candidate_generation", {})
    count = int(generation.get("candidate_count", 0))
    if count < 32 or count > 64:
        raise ValueError("candidate_count must stay in [32, 64]")
    dynamic = value.get("dynamic", {})
    if (
        float(dynamic.get("lift_distance_m", 0.0)) != 0.05
        or float(dynamic.get("hold_duration_s", 0.0)) < 2.0
        or dynamic.get("online_object_truth_allowed") is not False
        or dynamic.get("online_contact_truth_allowed") is not False
        or dynamic.get("object_pose_write_after_start_allowed") is not False
    ):
        raise ValueError("V2 dynamic safety and 50 mm / 2 s boundaries changed")
    return CARTSV2Config(config_path, MappingProxyType(dict(value)))


def build_face_role_map(
    loaded: LoadedObjectContract, config: CARTSV2Config
) -> FaceRoleMap:
    """Partition every source face once using shared task-axis geometry rules."""

    settings = config.section("surface_roles")
    if settings.get("method") != _ROLE_METHOD:
        raise ValueError("unsupported V2 face-role method")
    model = loaded.model
    mesh = model.mesh
    basis = loaded.task_frame_rotation_object
    centers = (mesh.face_centroids_m - model.assembly_axis_origin_m) @ basis
    normals = mesh.face_normals @ basis
    radial_distance = np.linalg.norm(centers[:, :2], axis=1)
    epsilon = float(settings["radial_numerical_epsilon_m"])
    radial_unit = np.zeros_like(normals)
    non_axis = radial_distance > epsilon
    radial_unit[non_axis, :2] = (
        centers[non_axis, :2] / radial_distance[non_axis, None]
    )
    radial_alignment = np.abs(np.einsum("ij,ij->i", normals, radial_unit))
    axial_alignment = np.abs(normals[:, 2])

    azimuth_count = int(settings["angular_bin_count"])
    axial_count = int(settings["axial_bin_count"])
    theta = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2.0 * np.pi)
    z = centers[:, 2]
    z_span = max(float(np.ptp(z)), np.finfo(np.float64).eps)
    axial_bin = np.minimum(
        ((z - float(np.min(z))) / z_span * axial_count).astype(np.int64),
        axial_count - 1,
    )
    azimuth_bin = np.minimum(
        (theta / (2.0 * np.pi) * azimuth_count).astype(np.int64),
        azimuth_count - 1,
    )
    bin_index = axial_bin * azimuth_count + azimuth_bin
    outer_radius = np.full(axial_count * azimuth_count, -np.inf)
    np.maximum.at(outer_radius, bin_index, radial_distance)
    on_outer_envelope = radial_distance >= (
        outer_radius[bin_index] - float(settings["outer_envelope_depth_m"])
    )

    semantic_allowed = np.fromiter(
        (semantic in model.allowed_contact_semantics for semantic in mesh.face_semantics),
        dtype=np.bool_,
        count=len(mesh.faces),
    )
    lateral = (
        radial_alignment >= float(settings["minimum_radial_normal_component"])
    ) & (axial_alignment <= float(settings["maximum_axial_normal_component"]))
    allowed = semantic_allowed & non_axis & lateral & on_outer_envelope
    reason = np.zeros(len(allowed), dtype=np.uint8)
    reason[~semantic_allowed] = 1
    reason[semantic_allowed & ~non_axis] = 2
    reason[semantic_allowed & non_axis & ~lateral] = 3
    reason[semantic_allowed & non_axis & lateral & ~on_outer_envelope] = 4
    area = mesh.face_areas_m2
    return FaceRoleMap(
        object_id=loaded.object_id,
        face_is_allowed=allowed,
        reason_code=reason,
        method=_ROLE_METHOD,
        allowed_area_m2=float(np.sum(area[allowed])),
        total_area_m2=float(np.sum(area)),
    )


def load_v2_inputs(
    repository_root: Path | str,
    *,
    config_path: Path | str,
    object_id: str,
) -> V2Inputs:
    root = Path(repository_root).resolve()
    config = load_v2_config(config_path)
    inputs = config.section("inputs")
    if object_id not in {inputs["development_object"], inputs["transfer_object"]}:
        raise ValueError(f"object {object_id!r} is outside the frozen V2 pair")
    object_contract = load_object_contract(
        inputs["object_contract"], object_id=object_id, repository_root=root
    )
    hand_contract = load_carts_hand_contract(
        inputs["hand_contract"], repository_root=root
    )
    if hand_contract.hardware_authorized or not hand_contract.truth_firewall_all_false:
        raise ValueError("hand safety/truth firewall contract changed")
    hand_model = hand_contract.build_hand_model()
    closing = hand_contract.closing_actuation_directions_unit(hand_model)
    closing = _readonly(closing, np.float64, ndim=2)
    face_roles = build_face_role_map(object_contract, config)
    return V2Inputs(
        config=config,
        object_contract=object_contract,
        hand_contract=hand_contract,
        hand_model=hand_model,
        closing_directions=closing,
        face_roles=face_roles,
    )
