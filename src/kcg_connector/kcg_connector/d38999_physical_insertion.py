"""Pure contract and geometry helpers for the D38999 insertion probe.

Isaac owns all physical state.  This module deliberately has no Isaac, ROS or
USD imports so its coordinate conventions, dependency hashes and small IK
correction can be tested with the ordinary project Python interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from kcg_connector.d38999_tabletop_pick import (
    iiwa14_grasp_tcp_transform,
)


SCHEMA_VERSION = "kcg_d38999_physical_insertion_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_physical_insertion_v1.yaml"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected, label: str) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ValueError(
            f"{label} keys differ: missing={sorted(wanted - actual)}, "
            f"extra={sorted(actual - wanted)}"
        )


def _real(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be finite and positive")
    return result


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    # Runtime geometry is naturally represented by NumPy arrays, whereas YAML
    # supplies lists.  Accept both finite indexable vectors, but not text or
    # mappings whose iteration semantics would silently use keys.
    if (
        isinstance(value, (str, bytes, Mapping))
        or not hasattr(value, "__len__")
        or len(value) != length
    ):
        raise ValueError(f"{label} must contain exactly {length} values")
    return tuple(
        _real(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


@dataclass(frozen=True)
class InputFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class InsertionInputs:
    tabletop_pick: InputFile
    assembly_baseline: InputFile
    ik_family: InputFile


@dataclass(frozen=True)
class InsertionMotion:
    interpolation: str
    transport_safe_tcp_position_m: tuple[float, ...]
    transport_safe_arm_rad: tuple[float, ...]
    transport_duration_s: float
    axis_high_gap_m: float
    axis_high_tcp_position_m: tuple[float, ...]
    axis_high_arm_rad: tuple[float, ...]
    axis_high_duration_s: float
    axis_high_hold_s: float
    preinsert_tcp_position_m: tuple[float, ...]
    preinsert_arm_rad: tuple[float, ...]
    preinsert_duration_s: float
    preinsert_hold_s: float
    engage_tcp_position_m: tuple[float, ...]
    engage_arm_rad: tuple[float, ...]
    insertion_duration_s: float
    engage_hold_s: float
    fixed_q7_rad: float
    runtime_body_in_tcp_compensation: bool


@dataclass(frozen=True)
class InsertionProxyCollisionFilter:
    mode: str
    expected_nut_segment_count: int
    expected_body_mating_segment_count: int
    expected_fixed_entry_segment_count: int
    expected_filtered_pair_count: int


@dataclass(frozen=True)
class InsertionAcceptance:
    maximum_preinsert_gap_error_m: float
    maximum_axis_high_gap_error_m: float
    maximum_engage_gap_error_m: float
    maximum_lateral_error_m: float
    maximum_axis_error_rad: float
    maximum_combined_entry_error_m: float
    entry_evaluation_length_m: float
    maximum_insertion_travel_error_m: float
    maximum_body_tcp_slip_m: float
    maximum_arm_tracking_error_rad: float
    maximum_joint_speed_rad_s: float
    maximum_joint_limit_violation_rad: float
    maximum_fixed_translation_drift_m: float
    maximum_fixed_rotation_drift_rad: float


@dataclass(frozen=True)
class InsertionBoundaries:
    pose_source: str
    vision_included: bool
    attachment_allowed: bool
    object_drive_allowed: bool
    object_pose_writes_after_start_allowed: bool
    collision_planned: bool
    continuous_collision_verified: bool
    real_keying_modeled: bool
    thread_teeth_modeled: bool
    assembly_success_claimed: bool


@dataclass(frozen=True)
class D38999PhysicalInsertion:
    schema_version: str
    enabled: bool
    status: str
    inputs: InsertionInputs
    motion: InsertionMotion
    proxy_collision_filter: InsertionProxyCollisionFilter
    acceptance: InsertionAcceptance
    boundaries: InsertionBoundaries


@dataclass(frozen=True)
class AlignmentMeasurement:
    gap_m: float
    lateral_error_m: float
    axis_error_rad: float
    combined_entry_error_m: float


def axial_gap_waypoints(
    start_gap_m: Real,
    end_gap_m: Real,
    maximum_step_m: Real,
) -> tuple[float, ...]:
    """Return monotonic endpoint-inclusive axial servo waypoints."""

    start = _real(start_gap_m, "start_gap_m")
    end = _real(end_gap_m, "end_gap_m")
    step = _real(maximum_step_m, "maximum_step_m", positive=True)
    distance = abs(end - start)
    if distance <= 0.0:
        raise ValueError("axial gap path must have non-zero travel")
    # Decimal millimetre values are not exact binary floats; the epsilon keeps
    # an exact conceptual 5.00/0.25 mm plan at 20 steps instead of 21.
    count = int(math.ceil(distance / step - 1.0e-12))
    return tuple(
        start + (end - start) * float(index) / float(count)
        for index in range(1, count + 1)
    )


def _load_input(value: Any, label: str) -> InputFile:
    document = _mapping(value, label)
    _exact_keys(document, ("path", "sha256"), label)
    path = _text(document["path"], f"{label}.path")
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError(f"{label}.path must be repository-relative")
    digest = _text(document["sha256"], f"{label}.sha256")
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
    return InputFile(path, digest)


def _matrix(transform) -> np.ndarray:
    return np.asarray(transform, dtype=np.float64)


def _forward_kinematics(joints: np.ndarray) -> np.ndarray:
    """Adapt NumPy work arrays to the checked-in pure FK sequence API."""

    return _matrix(
        iiwa14_grasp_tcp_transform(
            tuple(float(value) for value in joints)
        )
    )


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the shortest SO(3) logarithm, stable near zero rotation."""

    cosine = max(-1.0, min(1.0, (float(np.trace(matrix)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    skew = np.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ),
        dtype=np.float64,
    )
    if angle < 1.0e-9:
        return 0.5 * skew
    return angle * skew / (2.0 * math.sin(angle))


def solve_fixed_q7_tcp_pose(
    seed_arm_rad: Sequence[Real],
    target_position_m: Sequence[Real],
    *,
    target_rotation: Any = None,
    maximum_iterations: int = 20,
    damping: float = 1.0e-6,
) -> tuple[float, ...]:
    """Numerically correct a nearby TCP target while keeping q7 fixed.

    The target orientation defaults to the seed orientation.  This is
    intentionally a local correction around an already checked IK family, not
    a general motion planner or global IK solver.
    """

    joints = np.asarray(_vector(seed_arm_rad, 7, "seed_arm_rad"))
    target = np.asarray(_vector(target_position_m, 3, "target_position_m"))
    seed_rotation = _forward_kinematics(joints)[:3, :3]
    if target_rotation is None:
        target_rotation_array = seed_rotation
    else:
        target_rotation_array = np.asarray(target_rotation, dtype=np.float64)
        if target_rotation_array.shape != (3, 3) or not np.all(
            np.isfinite(target_rotation_array)
        ):
            raise ValueError("target_rotation must be a finite 3x3 matrix")
    epsilon = 1.0e-6
    for _ in range(maximum_iterations):
        current = _forward_kinematics(joints)
        error = np.concatenate(
            (
                target - current[:3, 3],
                _rotation_vector(
                    target_rotation_array @ current[:3, :3].T
                ),
            )
        )
        if float(np.linalg.norm(error[:3])) <= 1.0e-9 and float(
            np.linalg.norm(error[3:])
        ) <= 1.0e-9:
            break
        jacobian = np.zeros((6, 6), dtype=np.float64)
        for index in range(6):
            plus = joints.copy()
            minus = joints.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_transform = _forward_kinematics(plus)
            minus_transform = _forward_kinematics(minus)
            jacobian[:3, index] = (
                plus_transform[:3, 3] - minus_transform[:3, 3]
            ) / (2.0 * epsilon)
            jacobian[3:, index] = _rotation_vector(
                plus_transform[:3, :3] @ minus_transform[:3, :3].T
            ) / (2.0 * epsilon)
        normal = jacobian.T @ jacobian + damping * np.eye(6)
        joints[:6] += np.linalg.solve(normal, jacobian.T @ error)
    result = _forward_kinematics(joints)
    position_error = float(np.linalg.norm(target - result[:3, 3]))
    orientation_error = float(
        np.linalg.norm(
            _rotation_vector(target_rotation_array @ result[:3, :3].T)
        )
    )
    if position_error > 1.0e-7 or orientation_error > 1.0e-7:
        raise ValueError(
            "fixed-q7 local IK did not converge: "
            f"position_error={position_error}, "
            f"orientation_error={orientation_error}"
        )
    return tuple(float(value) for value in joints)


def compensated_tcp_position(
    desired_body_position_world_m: Sequence[Real],
    body_point_in_tcp_m: Sequence[Real],
    target_arm_rad: Sequence[Real],
) -> tuple[float, ...]:
    """Place the measured body-origin point at a desired world position."""

    desired = np.asarray(
        _vector(desired_body_position_world_m, 3, "desired_body_position")
    )
    body_local = np.asarray(
        _vector(body_point_in_tcp_m, 3, "body_point_in_tcp")
    )
    rotation = _matrix(iiwa14_grasp_tcp_transform(target_arm_rad))[:3, :3]
    return tuple(float(value) for value in desired - rotation @ body_local)


def pose_transform(
    position_m: Sequence[Real], quaternion_wxyz: Sequence[Real]
) -> np.ndarray:
    """Return a column-vector homogeneous transform from a wxyz pose."""

    position = np.asarray(_vector(position_m, 3, "position_m"))
    quaternion = np.asarray(
        _vector(quaternion_wxyz, 4, "quaternion_wxyz")
    )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("quaternion_wxyz must have non-zero norm")
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ),
            (
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ),
            (
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def compensated_tcp_transform(
    desired_body_world_transform: Any,
    measured_tcp_world_transform: Any,
    measured_body_world_transform: Any,
) -> np.ndarray:
    """Preserve the complete measured TCP-to-body grasp transform."""

    desired = np.asarray(desired_body_world_transform, dtype=np.float64)
    measured_tcp = np.asarray(measured_tcp_world_transform, dtype=np.float64)
    measured_body = np.asarray(
        measured_body_world_transform, dtype=np.float64
    )
    for label, value in (
        ("desired_body_world_transform", desired),
        ("measured_tcp_world_transform", measured_tcp),
        ("measured_body_world_transform", measured_body),
    ):
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError(f"{label} must be a finite 4x4 matrix")
    tcp_to_body = np.linalg.inv(measured_tcp) @ measured_body
    return desired @ np.linalg.inv(tcp_to_body)


def measure_alignment(
    body_position_world_m: Sequence[Real],
    body_axis_world: Sequence[Real],
    fixed_position_world_m: Sequence[Real],
    fixed_axis_world: Sequence[Real],
    entry_evaluation_length_m: Real,
) -> AlignmentMeasurement:
    body = np.asarray(_vector(body_position_world_m, 3, "body_position"))
    fixed = np.asarray(_vector(fixed_position_world_m, 3, "fixed_position"))
    body_axis = np.asarray(_vector(body_axis_world, 3, "body_axis"))
    fixed_axis = np.asarray(_vector(fixed_axis_world, 3, "fixed_axis"))
    body_axis /= np.linalg.norm(body_axis)
    fixed_axis /= np.linalg.norm(fixed_axis)
    delta = body - fixed
    gap = float(delta @ fixed_axis)
    lateral = float(np.linalg.norm(delta - gap * fixed_axis))
    angle = math.acos(
        max(-1.0, min(1.0, float(body_axis @ fixed_axis)))
    )
    length = _real(
        entry_evaluation_length_m, "entry_evaluation_length_m", positive=True
    )
    return AlignmentMeasurement(
        gap_m=gap,
        lateral_error_m=lateral,
        axis_error_rad=angle,
        combined_entry_error_m=lateral + length * math.sin(angle),
    )


def load_d38999_physical_insertion(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> D38999PhysicalInsertion:
    path = Path(config_path).expanduser().resolve()
    document = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")), "root"
    )
    _exact_keys(
        document,
        (
            "schema_version",
            "enabled",
            "status",
            "inputs",
            "motion",
            "proxy_collision_filter",
            "acceptance",
            "boundaries",
        ),
        "root",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected physical insertion schema_version")
    inputs_document = _mapping(document["inputs"], "inputs")
    _exact_keys(
        inputs_document,
        ("tabletop_pick", "assembly_baseline", "ik_family"),
        "inputs",
    )
    inputs = InsertionInputs(
        tabletop_pick=_load_input(
            inputs_document["tabletop_pick"], "inputs.tabletop_pick"
        ),
        assembly_baseline=_load_input(
            inputs_document["assembly_baseline"],
            "inputs.assembly_baseline",
        ),
        ik_family=_load_input(
            inputs_document["ik_family"], "inputs.ik_family"
        ),
    )
    motion_document = _mapping(document["motion"], "motion")
    _exact_keys(
        motion_document, InsertionMotion.__dataclass_fields__, "motion"
    )
    motion = InsertionMotion(
        interpolation=_text(
            motion_document["interpolation"], "motion.interpolation"
        ),
        transport_safe_tcp_position_m=_vector(
            motion_document["transport_safe_tcp_position_m"],
            3,
            "motion.transport_safe_tcp_position_m",
        ),
        transport_safe_arm_rad=_vector(
            motion_document["transport_safe_arm_rad"],
            7,
            "motion.transport_safe_arm_rad",
        ),
        transport_duration_s=_real(
            motion_document["transport_duration_s"],
            "motion.transport_duration_s",
            positive=True,
        ),
        axis_high_gap_m=_real(
            motion_document["axis_high_gap_m"],
            "motion.axis_high_gap_m",
            positive=True,
        ),
        axis_high_tcp_position_m=_vector(
            motion_document["axis_high_tcp_position_m"],
            3,
            "motion.axis_high_tcp_position_m",
        ),
        axis_high_arm_rad=_vector(
            motion_document["axis_high_arm_rad"],
            7,
            "motion.axis_high_arm_rad",
        ),
        axis_high_duration_s=_real(
            motion_document["axis_high_duration_s"],
            "motion.axis_high_duration_s",
            positive=True,
        ),
        axis_high_hold_s=_real(
            motion_document["axis_high_hold_s"],
            "motion.axis_high_hold_s",
            positive=True,
        ),
        preinsert_tcp_position_m=_vector(
            motion_document["preinsert_tcp_position_m"],
            3,
            "motion.preinsert_tcp_position_m",
        ),
        preinsert_arm_rad=_vector(
            motion_document["preinsert_arm_rad"],
            7,
            "motion.preinsert_arm_rad",
        ),
        preinsert_duration_s=_real(
            motion_document["preinsert_duration_s"],
            "motion.preinsert_duration_s",
            positive=True,
        ),
        preinsert_hold_s=_real(
            motion_document["preinsert_hold_s"],
            "motion.preinsert_hold_s",
            positive=True,
        ),
        engage_tcp_position_m=_vector(
            motion_document["engage_tcp_position_m"],
            3,
            "motion.engage_tcp_position_m",
        ),
        engage_arm_rad=_vector(
            motion_document["engage_arm_rad"],
            7,
            "motion.engage_arm_rad",
        ),
        insertion_duration_s=_real(
            motion_document["insertion_duration_s"],
            "motion.insertion_duration_s",
            positive=True,
        ),
        engage_hold_s=_real(
            motion_document["engage_hold_s"],
            "motion.engage_hold_s",
            positive=True,
        ),
        fixed_q7_rad=_real(
            motion_document["fixed_q7_rad"], "motion.fixed_q7_rad"
        ),
        runtime_body_in_tcp_compensation=_bool(
            motion_document["runtime_body_in_tcp_compensation"],
            "motion.runtime_body_in_tcp_compensation",
        ),
    )
    if (
        motion.interpolation != "minimum_jerk"
        or not motion.runtime_body_in_tcp_compensation
    ):
        raise ValueError(
            "physical insertion requires minimum-jerk runtime compensation"
        )

    filter_document = _mapping(
        document["proxy_collision_filter"], "proxy_collision_filter"
    )
    _exact_keys(
        filter_document,
        InsertionProxyCollisionFilter.__dataclass_fields__,
        "proxy_collision_filter",
    )
    integer_fields = (
        "expected_nut_segment_count",
        "expected_body_mating_segment_count",
        "expected_fixed_entry_segment_count",
        "expected_filtered_pair_count",
    )
    for key in integer_fields:
        value = filter_document[key]
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"proxy_collision_filter.{key} must be a positive integer"
            )
    proxy_collision_filter = InsertionProxyCollisionFilter(
        mode=_text(
            filter_document["mode"], "proxy_collision_filter.mode"
        ),
        **{key: filter_document[key] for key in integer_fields},
    )
    if (
        proxy_collision_filter.mode != "proxy_false_contacts_only"
        or proxy_collision_filter.expected_nut_segment_count != 24
        or proxy_collision_filter.expected_body_mating_segment_count != 20
        or proxy_collision_filter.expected_fixed_entry_segment_count != 20
        or proxy_collision_filter.expected_filtered_pair_count != 500
    ):
        raise ValueError("unexpected D38999 proxy collision filter contract")
    for name, arm, expected_position in (
        (
            "transport_safe",
            motion.transport_safe_arm_rad,
            motion.transport_safe_tcp_position_m,
        ),
        (
            "preinsert",
            motion.preinsert_arm_rad,
            motion.preinsert_tcp_position_m,
        ),
        (
            "axis_high",
            motion.axis_high_arm_rad,
            motion.axis_high_tcp_position_m,
        ),
        (
            "engage",
            motion.engage_arm_rad,
            motion.engage_tcp_position_m,
        ),
    ):
        transform = _matrix(iiwa14_grasp_tcp_transform(arm))
        if (
            float(np.linalg.norm(transform[:3, 3] - expected_position))
            > 5.0e-6
        ):
            raise ValueError(f"{name} arm target fails pure FK position")
        if (
            float(np.linalg.norm(transform[:3, 2] - (0.0, 0.0, -1.0)))
            > 5.0e-6
        ):
            raise ValueError(f"{name} arm target fails pure FK axis")
        if abs(arm[6] - motion.fixed_q7_rad) > 1.0e-9:
            raise ValueError(f"{name} arm target must preserve fixed q7")
    acceptance_document = _mapping(document["acceptance"], "acceptance")
    _exact_keys(
        acceptance_document,
        InsertionAcceptance.__dataclass_fields__,
        "acceptance",
    )
    acceptance = InsertionAcceptance(
        **{
            key: _real(
                acceptance_document[key],
                f"acceptance.{key}",
                positive=True,
            )
            for key in InsertionAcceptance.__dataclass_fields__
        }
    )
    boundaries_document = _mapping(document["boundaries"], "boundaries")
    _exact_keys(
        boundaries_document,
        InsertionBoundaries.__dataclass_fields__,
        "boundaries",
    )
    boundaries = InsertionBoundaries(
        pose_source=_text(
            boundaries_document["pose_source"], "boundaries.pose_source"
        ),
        **{
            key: _bool(boundaries_document[key], f"boundaries.{key}")
            for key in InsertionBoundaries.__dataclass_fields__
            if key != "pose_source"
        },
    )
    if boundaries.pose_source != "isaac_ground_truth_runtime":
        raise ValueError("v1 insertion pose source must be Isaac ground truth")
    if any(
        (
            boundaries.vision_included,
            boundaries.attachment_allowed,
            boundaries.object_drive_allowed,
            boundaries.object_pose_writes_after_start_allowed,
            boundaries.collision_planned,
            boundaries.continuous_collision_verified,
            boundaries.real_keying_modeled,
            boundaries.thread_teeth_modeled,
            boundaries.assembly_success_claimed,
        )
    ):
        raise ValueError("v1 insertion safety boundaries must remain false")
    return D38999PhysicalInsertion(
        schema_version=SCHEMA_VERSION,
        enabled=_bool(document["enabled"], "enabled"),
        status=_text(document["status"], "status"),
        inputs=inputs,
        motion=motion,
        proxy_collision_filter=proxy_collision_filter,
        acceptance=acceptance,
        boundaries=boundaries,
    )


def verify_insertion_inputs(
    contract: D38999PhysicalInsertion, repository: Path | str
) -> dict[str, Path]:
    root = Path(repository).expanduser().resolve()
    result = {}
    for name in InsertionInputs.__dataclass_fields__:
        item = getattr(contract.inputs, name)
        path = (root / item.path).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"missing insertion input: {item.path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item.sha256:
            raise ValueError(f"insertion input hash mismatch: {item.path}")
        result[name] = path
    return result


__all__ = [
    "AlignmentMeasurement",
    "DEFAULT_CONFIG_PATH",
    "D38999PhysicalInsertion",
    "axial_gap_waypoints",
    "compensated_tcp_transform",
    "compensated_tcp_position",
    "load_d38999_physical_insertion",
    "measure_alignment",
    "pose_transform",
    "solve_fixed_q7_tcp_pose",
    "verify_insertion_inputs",
]
