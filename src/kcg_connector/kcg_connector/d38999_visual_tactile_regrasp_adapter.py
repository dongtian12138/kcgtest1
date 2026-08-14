"""Pure CPU adapter from accepted visual+tactile XY to assembly targets.

The existing D38999 nut-only regrasp and retreat targets are authored at the
absolute nominal fixed-end XY.  This module provides a disabled seam for a
future runner: only after the tactile state machine has latched
``READY_FOR_EXISTING_PROXY_TWIST`` does it add the accepted tactile search
offset to the already bounded visual fixed-end translation.  The tactile
offset is expressed in connector-task XY, so its hash-bound task rotation is
validated and applied before any world-XY addition.  The module then shifts
the registered target family and re-solves each target with the existing
local fixed-q7 IK.

There are deliberately no Isaac, USD, ROS, camera, or truth-pose inputs here.
The generated plan is a CPU contract artifact, not GPU/PhysX evidence and not
authorization to execute regrasp, twist, retreat, Home, or hardware motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import yaml

from kcg_connector.d38999_assembly_baseline import (
    D38999AssemblyBaseline,
    load_d38999_assembly_baseline,
)
from kcg_connector.d38999_physical_insertion import (
    D38999PhysicalInsertion,
    load_d38999_physical_insertion,
    solve_fixed_q7_tcp_pose,
)
from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform
from kcg_connector.d38999_tactile_engage_probe import (
    EngageState,
    TactileEngageContract,
    load_tactile_engage_contract,
)
from kcg_connector.d38999_visual_xy_control_adapter import (
    VisualXyAdaptationResult,
    VisualXyControlAdapterContract,
    load_visual_xy_control_adapter_contract,
)
from kcg_connector.virtual_wrist_ft_runtime import (
    VirtualWristFtMonitorConfig,
    load_virtual_wrist_ft_monitor_config,
    task_rotation_world_from_axis,
)


SCHEMA_VERSION = "kcg_d38999_visual_tactile_regrasp_adapter_v1"
READY_INPUT_SCHEMA_VERSION = "kcg_d38999_tactile_ready_center_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_visual_tactile_regrasp_adapter_v1.yaml"
)
TARGET_ORDER = (
    "axis_high",
    "preinsert",
    "engage_hold",
    "nut_only_regrasp",
    "safe_retreat",
)
POST_READY_EXECUTION_ORDER = (
    "engage_hold",
    "nut_only_regrasp",
    "safe_retreat",
)
READY_SOURCE_SCOPES = (
    "runtime_tactile_engage_evidence",
    "cpu_contract_fixture_not_runtime_evidence",
)
TASK_ROTATION_GENERIC_TOLERANCE = 1.0e-9
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class InputArtifact:
    """One repository-relative, content-addressed dependency."""

    path: str
    sha256: str


@dataclass(frozen=True)
class LocalFixedQ7Policy:
    """Acceptance limits for the reused local IK solver."""

    maximum_iterations: int
    damping: float
    maximum_fk_position_error_m: float
    maximum_fk_orientation_error_rad: float
    maximum_abs_joint_delta_from_nominal_rad: float


@dataclass(frozen=True)
class TactileReadyCenterInput:
    """Strict handoff emitted only by a latched tactile READY state.

    ``source_scope`` keeps synthetic CPU fixtures distinguishable from future
    runtime evidence.  Neither scope upgrades this adapter to an executable or
    GPU-validated controller.
    """

    schema_version: str
    state: EngageState
    capture_id: str
    task_frame_id: str
    task_rotation_world: tuple[tuple[float, float, float], ...]
    search_offset_task_xy_m: tuple[float, float]
    offset_source: str
    truth_used_for_tactile_search: bool
    posthoc_truth_used_for_target: bool
    source_scope: str

    def to_mapping(self) -> dict[str, Any]:
        """Return the complete exact-schema READY handoff."""

        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "capture_id": self.capture_id,
            "task_frame_id": self.task_frame_id,
            "task_rotation_world": [
                list(row) for row in self.task_rotation_world
            ],
            "search_offset_task_xy_m": list(
                self.search_offset_task_xy_m
            ),
            "offset_source": self.offset_source,
            "truth_used_for_tactile_search": (
                self.truth_used_for_tactile_search
            ),
            "posthoc_truth_used_for_target": (
                self.posthoc_truth_used_for_target
            ),
            "source_scope": self.source_scope,
        }


@dataclass(frozen=True)
class TargetSeed:
    """Registered nominal TCP and its already validated fixed-q7 seed."""

    tcp_position_world_m: tuple[float, float, float]
    arm_rad: tuple[float, ...]


@dataclass(frozen=True)
class VisualTactileRegraspAdapterContract:
    """Resolved dependencies and immutable adapter limits."""

    schema_version: str
    enabled_by_default: bool
    status: str
    inputs: dict[str, InputArtifact]
    input_paths: dict[str, Path]
    visual: VisualXyControlAdapterContract
    tactile: TactileEngageContract
    insertion: D38999PhysicalInsertion
    assembly: D38999AssemblyBaseline
    wrist_ft: VirtualWristFtMonitorConfig
    required_visual_status: str
    required_ready_state: EngageState
    required_task_frame_id: str
    expected_task_rotation_world: tuple[
        tuple[float, float, float], ...
    ]
    maximum_orthogonality_error: float
    maximum_determinant_error: float
    maximum_expected_rotation_error: float
    maximum_task_xy_world_z_component: float
    nominal_fixed_xy_m: tuple[float, float]
    target_seeds: dict[str, TargetSeed]
    ik: LocalFixedQ7Policy
    config_path: Path
    repository: Path
    cpu_sample_artifact: str
    boundaries: dict[str, bool]


@dataclass(frozen=True)
class VisualTactileRegraspPlan:
    """A target-only CPU plan whose accepted center has explicit provenance."""

    capture_id: str
    ready_source_scope: str
    task_frame_id: str
    task_rotation_world: tuple[tuple[float, float, float], ...]
    visual_fixed_delta_xy_m: tuple[float, float]
    tactile_search_offset_task_xy_m: tuple[float, float]
    tactile_search_offset_world_m: tuple[float, float, float]
    tactile_search_offset_world_xy_m: tuple[float, float]
    accepted_center_delta_xy_m: tuple[float, float]
    accepted_center_world_xy_m: tuple[float, float]
    tcp_targets_world_m: dict[str, tuple[float, float, float]]
    arm_targets_rad: dict[str, tuple[float, ...]]
    fk_position_errors_m: dict[str, float]
    fk_orientation_errors_rad: dict[str, float]
    target_joint_deltas_from_nominal_rad: dict[str, float]

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-safe report without upgrading validation scope."""

        return {
            "schema_version": SCHEMA_VERSION,
            "status": "CPU_TARGET_PLAN_BUILT_NOT_GPU_VALIDATED",
            "capture_id": self.capture_id,
            "ready_input_state": (
                EngageState.READY_FOR_EXISTING_PROXY_TWIST.value
            ),
            "ready_source_scope": self.ready_source_scope,
            "task_frame_id": self.task_frame_id,
            "task_rotation_world": [
                list(row) for row in self.task_rotation_world
            ],
            "task_rotation_storage": (
                "columns_are_task_axes_expressed_in_world"
            ),
            "visual_fixed_delta_xy_m": list(
                self.visual_fixed_delta_xy_m
            ),
            "tactile_search_offset_task_xy_m": list(
                self.tactile_search_offset_task_xy_m
            ),
            "tactile_search_offset_world_m": list(
                self.tactile_search_offset_world_m
            ),
            "tactile_search_offset_world_xy_m": list(
                self.tactile_search_offset_world_xy_m
            ),
            "accepted_center_formula": (
                "visual_fixed_world_delta_xy_plus_task_rotation_world_"
                "times_tactile_task_xy"
            ),
            "accepted_center_delta_xy_m": list(
                self.accepted_center_delta_xy_m
            ),
            "accepted_center_world_xy_m": list(
                self.accepted_center_world_xy_m
            ),
            "target_order": list(TARGET_ORDER),
            "post_ready_execution_order": list(
                POST_READY_EXECUTION_ORDER
            ),
            "tcp_targets_world_m": {
                name: list(values)
                for name, values in self.tcp_targets_world_m.items()
            },
            "arm_targets_rad": {
                name: list(values)
                for name, values in self.arm_targets_rad.items()
            },
            "fk_position_errors_m": dict(self.fk_position_errors_m),
            "fk_orientation_errors_rad": dict(
                self.fk_orientation_errors_rad
            ),
            "target_joint_deltas_from_nominal_rad": dict(
                self.target_joint_deltas_from_nominal_rad
            ),
            "world_xy_shift_applied_to_nut_only_regrasp": True,
            "world_xy_shift_applied_to_safe_retreat": True,
            "world_xy_shift_applied_to_assembly_targets": True,
            "fixed_q7_preserved_per_target": True,
            "truth_position_used_for_target": False,
            "truth_pose_feedback_used_for_target": False,
            "sim_truth_audit_used_for_target": False,
            "task_xy_assumed_equal_to_world_xy": False,
            "orientation_estimated_from_rgbd": False,
            "gpu_or_physx_validated": False,
            "collision_planned": False,
            "production_control_authorized": False,
            "assembly_success_claimed": False,
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty unpadded text")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        suffix = " and positive" if positive else ""
        raise ValueError(f"{label} must be finite{suffix}")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _vector(
    value: Any, length: int, label: str
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{label} must contain {length} finite numbers")
    try:
        items = tuple(value)
    except TypeError as error:
        raise ValueError(
            f"{label} must contain {length} finite numbers"
        ) from error
    if len(items) != length:
        raise ValueError(f"{label} must contain {length} finite numbers")
    return tuple(
        _finite(item, f"{label}[{index}]")
        for index, item in enumerate(items)
    )


def _rotation_matrix(
    value: Any, label: str
) -> tuple[tuple[float, float, float], ...]:
    """Parse one finite, orthogonal, right-handed task-to-world rotation."""

    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    try:
        rows = tuple(value)
    except TypeError as error:
        raise ValueError(f"{label} must be a finite 3x3 matrix") from error
    if len(rows) != 3:
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    parsed = tuple(
        _vector(row, 3, f"{label}[{index}]")
        for index, row in enumerate(rows)
    )
    matrix = np.asarray(parsed, dtype=np.float64)
    orthogonality_error = float(
        np.max(np.abs(matrix.T @ matrix - np.eye(3)))
    )
    determinant_error = abs(float(np.linalg.det(matrix)) - 1.0)
    if (
        orthogonality_error > TASK_ROTATION_GENERIC_TOLERANCE
        or determinant_error > TASK_ROTATION_GENERIC_TOLERANCE
    ):
        raise ValueError(
            f"{label} must be orthogonal and right-handed with det +1"
        )
    return parsed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_inputs(
    value: Any, repository: Path
) -> tuple[dict[str, InputArtifact], dict[str, Path]]:
    document = _mapping(value, "inputs")
    names = {
        "assembly_baseline",
        "visual_xy_adapter",
        "tactile_engage_probe",
        "physical_insertion",
        "nut_regrasp_physx",
        "nut_regrasp_cpu_search",
        "nut_regrasp_cpu_search_report",
        "virtual_wrist_ft_monitor",
    }
    _exact(document, names, "inputs")
    artifacts: dict[str, InputArtifact] = {}
    paths: dict[str, Path] = {}
    for name in sorted(names):
        label = f"inputs.{name}"
        item = _mapping(document[name], label)
        _exact(item, {"path", "sha256"}, label)
        relative_text = _text(item["path"], f"{label}.path")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label}.path must be repository-relative")
        digest = _text(item["sha256"], f"{label}.sha256")
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
        resolved = (repository / relative).resolve()
        if repository not in resolved.parents or not resolved.is_file():
            raise ValueError(f"{label}.path is missing or outside repository")
        if _sha256(resolved) != digest:
            raise ValueError(f"{label} SHA-256 mismatch")
        artifacts[name] = InputArtifact(relative.as_posix(), digest)
        paths[name] = resolved
    return artifacts, paths


def _rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = max(
        -1.0,
        min(1.0, (float(np.trace(relative)) - 1.0) / 2.0),
    )
    return math.acos(cosine)


def _load_regrasp_seeds(
    input_paths: Mapping[str, Path],
    insertion: D38999PhysicalInsertion,
    repository: Path,
) -> tuple[dict[str, TargetSeed], float]:
    """Bind the target seeds to existing regrasp/search evidence."""

    regrasp = _mapping(
        yaml.safe_load(
            input_paths["nut_regrasp_physx"].read_text(encoding="utf-8")
        ),
        "nut_regrasp_physx",
    )
    _exact(
        regrasp,
        {
            "schema_version",
            "enabled",
            "status",
            "inputs",
            "prepared_engage",
            "uncompensated_seed",
            "tracking_compensation",
            "nut_only_candidate",
            "control",
            "sensing",
            "acceptance",
            "boundaries",
        },
        "nut_regrasp_physx",
    )
    if (
        regrasp["schema_version"] != "kcg_d38999_nut_regrasp_physx_v1"
        or regrasp["enabled"] is not True
        or regrasp["status"]
        != "independent_prepared_engage_physics_ab"
    ):
        raise ValueError("nut regrasp contract identity changed")

    regrasp_inputs = _mapping(regrasp["inputs"], "regrasp.inputs")
    for adapter_name, regrasp_name in (
        ("nut_regrasp_cpu_search", "cpu_search_config"),
        ("nut_regrasp_cpu_search_report", "cpu_search_report"),
    ):
        item = _mapping(
            regrasp_inputs.get(regrasp_name),
            f"regrasp.inputs.{regrasp_name}",
        )
        expected = input_paths[adapter_name]
        if (
            (repository / str(item.get("path", ""))).resolve()
            != expected
            or item.get("sha256") != _sha256(expected)
        ):
            raise ValueError(
                f"regrasp {regrasp_name} differs from adapter input"
            )

    search = _mapping(
        yaml.safe_load(
            input_paths["nut_regrasp_cpu_search"].read_text(
                encoding="utf-8"
            )
        ),
        "nut_regrasp_cpu_search",
    )
    if (
        search.get("schema_version") != "kcg_d38999_nut_regrasp_search_v1"
        or search.get("enabled") is not False
        or search.get("status")
        != "cpu_geometry_search_only_not_runtime_authorized"
    ):
        raise ValueError("CPU regrasp search scope changed")
    robot = _mapping(search.get("robot"), "search.robot")
    fixed_q7 = _finite(robot.get("fixed_q7_rad"), "search fixed_q7")

    report = _mapping(
        yaml.safe_load(
            input_paths["nut_regrasp_cpu_search_report"].read_text(
                encoding="utf-8"
            )
        ),
        "nut_regrasp_cpu_search_report",
    )
    if (
        report.get("schema_version")
        != "kcg_d38999_nut_regrasp_search_report_v1"
        or report.get("status")
        != "FAIL_CLOSED_CONTINUOUS_COLLISION_UNVERIFIED"
        or report.get("nut_only_command_candidate_found") is not True
        or report.get("candidate_may_proceed_to_physx_static_ab") is not True
        or report.get("continuous_collision_verified") is not False
        or report.get("config_sha256")
        != _sha256(input_paths["nut_regrasp_cpu_search"])
    ):
        raise ValueError("CPU regrasp report does not support this seam")

    candidate = _mapping(
        regrasp["nut_only_candidate"], "regrasp.nut_only_candidate"
    )
    _exact(
        candidate,
        {
            "tcp_position_world_m",
            "desired_physical_tcp_position_world_m",
            "tcp_down_axis_world",
            "arm_rad",
            "hand_rad",
            "cpu_minimum_acceptance_margin_m",
            "cpu_minimum_body_clearance_m",
            "cpu_worst_nut_signed_distance_m",
        },
        "regrasp.nut_only_candidate",
    )
    candidate_tcp = _vector(
        candidate["tcp_position_world_m"], 3, "candidate TCP"
    )
    candidate_arm = _vector(candidate["arm_rad"], 7, "candidate arm")
    if (
        _vector(candidate["tcp_down_axis_world"], 3, "candidate axis")
        != (0.0, 0.0, -1.0)
        or not math.isclose(candidate_arm[6], fixed_q7, abs_tol=1.0e-12)
    ):
        raise ValueError("nut-only candidate leaves the fixed-q7 family")

    motion = insertion.motion
    seeds = {
        "axis_high": TargetSeed(
            motion.axis_high_tcp_position_m, motion.axis_high_arm_rad
        ),
        "preinsert": TargetSeed(
            motion.preinsert_tcp_position_m, motion.preinsert_arm_rad
        ),
        "engage_hold": TargetSeed(
            motion.engage_tcp_position_m, motion.engage_arm_rad
        ),
        "nut_only_regrasp": TargetSeed(candidate_tcp, candidate_arm),
        "safe_retreat": TargetSeed(
            motion.transport_safe_tcp_position_m,
            motion.transport_safe_arm_rad,
        ),
    }
    if tuple(seeds) != TARGET_ORDER:
        raise ValueError("internal target order changed")
    for name, seed in seeds.items():
        if not math.isclose(seed.arm_rad[6], fixed_q7, abs_tol=1.0e-12):
            raise ValueError(f"{name} seed does not preserve search fixed q7")
        transform = np.asarray(
            iiwa14_grasp_tcp_transform(seed.arm_rad), dtype=np.float64
        )
        if float(
            np.linalg.norm(
                transform[:3, 3]
                - np.asarray(seed.tcp_position_world_m, dtype=np.float64)
            )
        ) > 5.0e-6:
            raise ValueError(f"{name} seed TCP does not match pure FK")
    return seeds, fixed_q7


def load_visual_tactile_regrasp_adapter_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repository: str | Path | None = None,
) -> VisualTactileRegraspAdapterContract:
    """Load all hash edges and reject any scope or target-set upgrade."""

    config_path = Path(path).expanduser().resolve()
    root = (
        Path(repository).expanduser().resolve()
        if repository is not None
        else config_path.parents[3]
    )
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "root"
    )
    _exact(
        document,
        {
            "schema_version",
            "enabled_by_default",
            "status",
            "inputs",
            "ready_input_boundary",
            "task_frame_contract",
            "target_adapter",
            "local_fixed_q7_ik",
            "output",
            "boundaries",
        },
        "root",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "visual+tactile regrasp adapter schema is unsupported"
        )
    if document["enabled_by_default"] is not False:
        raise ValueError("visual+tactile adapter must remain disabled")
    status = _text(document["status"], "status")
    if status != (
        "prepared_cpu_target_adapter_ready_input_required_not_gpu_validated"
    ):
        raise ValueError("visual+tactile adapter status overclaims validation")

    inputs, input_paths = _resolve_inputs(document["inputs"], root)
    visual = load_visual_xy_control_adapter_contract(
        input_paths["visual_xy_adapter"], repository=root
    )
    tactile = load_tactile_engage_contract(
        input_paths["tactile_engage_probe"], repository=root
    )
    insertion = load_d38999_physical_insertion(
        input_paths["physical_insertion"]
    )
    assembly = load_d38999_assembly_baseline(
        input_paths["assembly_baseline"]
    )
    wrist_ft = load_virtual_wrist_ft_monitor_config(
        input_paths["virtual_wrist_ft_monitor"]
    )
    tactile_insertion = tactile.inputs["physical_insertion_contract"]
    if (
        tactile_insertion.path != inputs["physical_insertion"].path
        or tactile_insertion.sha256 != inputs["physical_insertion"].sha256
    ):
        raise ValueError("tactile and adapter insertion inputs differ")
    tactile_wrist = tactile.inputs["virtual_wrist_ft_monitor"]
    insertion_assembly = insertion.inputs.assembly_baseline
    if (
        tactile_wrist.path != inputs["virtual_wrist_ft_monitor"].path
        or tactile_wrist.sha256
        != inputs["virtual_wrist_ft_monitor"].sha256
        or insertion_assembly.path != inputs["assembly_baseline"].path
        or insertion_assembly.sha256 != inputs["assembly_baseline"].sha256
    ):
        raise ValueError("task-frame dependency edges differ")

    ready = _mapping(document["ready_input_boundary"], "ready boundary")
    _exact(
        ready,
        {
            "schema_version",
            "required_state",
            "required_visual_status",
            "require_same_capture_id",
            "required_task_frame_id",
            "tactile_offset_source",
            "tactile_offset_input_coordinates",
            "accepted_center_formula",
            "require_truth_not_used_for_visual_translation",
            "require_truth_not_used_for_tactile_search",
            "require_posthoc_truth_not_used_for_target",
        },
        "ready boundary",
    )
    expected_ready = {
        "schema_version": READY_INPUT_SCHEMA_VERSION,
        "required_state": EngageState.READY_FOR_EXISTING_PROXY_TWIST.value,
        "required_visual_status": (
            "ELIGIBLE_FOR_INDEPENDENT_VISUAL_XY_PROBE"
        ),
        "require_same_capture_id": True,
        "required_task_frame_id": "connector_task_frame",
        "tactile_offset_source": (
            "accepted_force_moment_search_offset_xy"
        ),
        "tactile_offset_input_coordinates": "task_frame_xy",
        "accepted_center_formula": (
            "visual_fixed_world_delta_xy_plus_task_rotation_world_times_"
            "tactile_task_xy"
        ),
        "require_truth_not_used_for_visual_translation": True,
        "require_truth_not_used_for_tactile_search": True,
        "require_posthoc_truth_not_used_for_target": True,
    }
    if dict(ready) != expected_ready:
        raise ValueError("READY input boundary changed")

    task_raw = _mapping(document["task_frame_contract"], "task frame")
    _exact(
        task_raw,
        {
            "rotation_storage",
            "rotation_source",
            "maximum_orthogonality_error",
            "maximum_determinant_error",
            "maximum_expected_rotation_error",
            "require_task_xy_plane_parallel_world_xy",
            "maximum_task_xy_world_z_component",
            "tactile_to_world_formula",
            "world_target_uses_xy_only",
        },
        "task frame",
    )
    maximum_orthogonality_error = _finite(
        task_raw["maximum_orthogonality_error"],
        "maximum_orthogonality_error",
        positive=True,
    )
    maximum_determinant_error = _finite(
        task_raw["maximum_determinant_error"],
        "maximum_determinant_error",
        positive=True,
    )
    maximum_expected_rotation_error = _finite(
        task_raw["maximum_expected_rotation_error"],
        "maximum_expected_rotation_error",
        positive=True,
    )
    maximum_task_xy_world_z_component = _finite(
        task_raw["maximum_task_xy_world_z_component"],
        "maximum_task_xy_world_z_component",
        positive=True,
    )
    if (
        ready["required_task_frame_id"] != wrist_ft.task_frame_id
        or task_raw["rotation_storage"]
        != "columns_are_task_axes_expressed_in_world"
        or task_raw["rotation_source"]
        != "assembly_fixed_axis_plus_wrist_ft_x_reference"
        or maximum_orthogonality_error != 1.0e-12
        or maximum_determinant_error != 1.0e-12
        or maximum_expected_rotation_error != 1.0e-12
        or maximum_task_xy_world_z_component != 1.0e-12
        or task_raw["require_task_xy_plane_parallel_world_xy"] is not True
        or task_raw["tactile_to_world_formula"]
        != "task_rotation_world_times_dx_dy_zero"
        or task_raw["world_target_uses_xy_only"] is not True
    ):
        raise ValueError("task-frame conversion contract changed")
    expected_task_rotation_array = task_rotation_world_from_axis(
        assembly.datums.fixed.axis_world,
        wrist_ft.task_x_reference_world,
    )
    expected_task_rotation = tuple(
        tuple(float(value) for value in row)
        for row in expected_task_rotation_array
    )
    if float(
        np.max(np.abs(expected_task_rotation_array[2, :2]))
    ) > maximum_task_xy_world_z_component:
        raise ValueError("configured task XY plane is not world-XY parallel")

    adapter = _mapping(document["target_adapter"], "target adapter")
    _exact(
        adapter,
        {
            "target_order",
            "post_ready_execution_order",
            "nominal_fixed_xy_m",
            "preserve_nominal_z",
            "preserve_nominal_orientation",
            "shift_world_xy_only",
            "fixed_q7_local_ik",
            "nut_only_regrasp_source",
            "safe_retreat_source",
        },
        "target adapter",
    )
    nominal_fixed = _vector(
        adapter["nominal_fixed_xy_m"], 2, "nominal fixed XY"
    )
    if (
        tuple(adapter["target_order"]) != TARGET_ORDER
        or tuple(adapter["post_ready_execution_order"])
        != POST_READY_EXECUTION_ORDER
        or nominal_fixed != visual.nominal_fixed_xy_m
        or nominal_fixed != insertion.motion.engage_tcp_position_m[:2]
        or adapter["preserve_nominal_z"] is not True
        or adapter["preserve_nominal_orientation"] is not True
        or adapter["shift_world_xy_only"] is not True
        or adapter["fixed_q7_local_ik"] is not True
        or adapter["nut_only_regrasp_source"]
        != "nut_regrasp_physx.nut_only_candidate"
        or adapter["safe_retreat_source"]
        != "physical_insertion.motion.transport_safe"
    ):
        raise ValueError("target adapter definition changed")

    ik_raw = _mapping(document["local_fixed_q7_ik"], "local IK")
    _exact(
        ik_raw,
        {
            "maximum_iterations",
            "damping",
            "maximum_fk_position_error_m",
            "maximum_fk_orientation_error_rad",
            "maximum_abs_joint_delta_from_nominal_rad",
            "q7_preserved_per_target",
        },
        "local IK",
    )
    ik = LocalFixedQ7Policy(
        maximum_iterations=_positive_integer(
            ik_raw["maximum_iterations"], "maximum_iterations"
        ),
        damping=_finite(ik_raw["damping"], "damping", positive=True),
        maximum_fk_position_error_m=_finite(
            ik_raw["maximum_fk_position_error_m"],
            "maximum_fk_position_error_m",
            positive=True,
        ),
        maximum_fk_orientation_error_rad=_finite(
            ik_raw["maximum_fk_orientation_error_rad"],
            "maximum_fk_orientation_error_rad",
            positive=True,
        ),
        maximum_abs_joint_delta_from_nominal_rad=_finite(
            ik_raw["maximum_abs_joint_delta_from_nominal_rad"],
            "maximum_abs_joint_delta_from_nominal_rad",
            positive=True,
        ),
    )
    if ik_raw["q7_preserved_per_target"] is not True:
        raise ValueError("local IK must preserve q7")

    output = _mapping(document["output"], "output")
    _exact(output, {"cpu_sample_artifact"}, "output")
    artifact = _text(output["cpu_sample_artifact"], "cpu sample artifact")
    artifact_path = Path(artifact)
    if artifact_path.is_absolute() or ".." in artifact_path.parts:
        raise ValueError(
            "CPU sample artifact path must be repository-relative"
        )

    boundaries = dict(_mapping(document["boundaries"], "boundaries"))
    expected_boundaries = {
        "explicit_opt_in_required": True,
        "ready_state_required": True,
        "truth_position_used_for_target": False,
        "truth_pose_feedback_used_for_target": False,
        "sim_truth_audit_used_for_target": False,
        "task_xy_assumed_equal_to_world_xy": False,
        "orientation_estimated_from_rgbd": False,
        "existing_runner_modified": False,
        "active_config_modified": False,
        "robot_or_connector_asset_modified": False,
        "frozen_baseline_modified": False,
        "collision_planned": False,
        "continuous_collision_verified": False,
        "tactile_engage_executed": False,
        "regrasp_executed": False,
        "twist_executed": False,
        "retreat_executed": False,
        "home_return_executed": False,
        "gpu_or_physx_validated": False,
        "production_control_authorized": False,
        "assembly_success_claimed": False,
    }
    if boundaries != expected_boundaries:
        raise ValueError("adapter boundaries changed")

    seeds, _ = _load_regrasp_seeds(input_paths, insertion, root)
    return VisualTactileRegraspAdapterContract(
        schema_version=SCHEMA_VERSION,
        enabled_by_default=False,
        status=status,
        inputs=inputs,
        input_paths=input_paths,
        visual=visual,
        tactile=tactile,
        insertion=insertion,
        assembly=assembly,
        wrist_ft=wrist_ft,
        required_visual_status=ready["required_visual_status"],
        required_ready_state=EngageState(ready["required_state"]),
        required_task_frame_id=ready["required_task_frame_id"],
        expected_task_rotation_world=expected_task_rotation,
        maximum_orthogonality_error=maximum_orthogonality_error,
        maximum_determinant_error=maximum_determinant_error,
        maximum_expected_rotation_error=maximum_expected_rotation_error,
        maximum_task_xy_world_z_component=(
            maximum_task_xy_world_z_component
        ),
        nominal_fixed_xy_m=nominal_fixed,
        target_seeds=seeds,
        ik=ik,
        config_path=config_path,
        repository=root,
        cpu_sample_artifact=artifact,
        boundaries=boundaries,
    )


def parse_tactile_ready_center(value: Any) -> TactileReadyCenterInput:
    """Parse the exact future runtime handoff without accepting extras."""

    document = _mapping(value, "tactile READY center")
    _exact(
        document,
        {
            "schema_version",
            "state",
            "capture_id",
            "task_frame_id",
            "task_rotation_world",
            "search_offset_task_xy_m",
            "offset_source",
            "truth_used_for_tactile_search",
            "posthoc_truth_used_for_target",
            "source_scope",
        },
        "tactile READY center",
    )
    schema = _text(document["schema_version"], "schema_version")
    if schema != READY_INPUT_SCHEMA_VERSION:
        raise ValueError("tactile READY input schema is unsupported")
    try:
        state = EngageState(_text(document["state"], "state"))
    except ValueError as error:
        raise ValueError("tactile READY input state is unsupported") from error
    source_scope = _text(document["source_scope"], "source_scope")
    if source_scope not in READY_SOURCE_SCOPES:
        raise ValueError("tactile READY source scope is unsupported")
    return TactileReadyCenterInput(
        schema_version=schema,
        state=state,
        capture_id=_text(document["capture_id"], "capture_id"),
        task_frame_id=_text(
            document["task_frame_id"], "task_frame_id"
        ),
        task_rotation_world=_rotation_matrix(
            document["task_rotation_world"], "task_rotation_world"
        ),
        search_offset_task_xy_m=_vector(
            document["search_offset_task_xy_m"],
            2,
            "search_offset_task_xy_m",
        ),
        offset_source=_text(document["offset_source"], "offset_source"),
        truth_used_for_tactile_search=_boolean(
            document["truth_used_for_tactile_search"],
            "truth_used_for_tactile_search",
        ),
        posthoc_truth_used_for_target=_boolean(
            document["posthoc_truth_used_for_target"],
            "posthoc_truth_used_for_target",
        ),
        source_scope=source_scope,
    )


def build_visual_tactile_regrasp_plan(
    contract: VisualTactileRegraspAdapterContract,
    visual_result: VisualXyAdaptationResult,
    tactile_ready: TactileReadyCenterInput,
    *,
    explicit_adapter_opt_in: bool,
) -> VisualTactileRegraspPlan:
    """Combine accepted XY and generate fixed-q7 target parameters.

    Runtime failures raise before returning any target map.  Post-hoc simulator
    truth may still audit a completed tactile engage elsewhere, but it has no
    value-bearing argument in this API and cannot alter these targets.
    """

    if not isinstance(contract, VisualTactileRegraspAdapterContract):
        raise ValueError("contract has the wrong type")
    if not isinstance(visual_result, VisualXyAdaptationResult):
        raise ValueError("visual_result has the wrong type")
    if not isinstance(tactile_ready, TactileReadyCenterInput):
        raise ValueError("tactile_ready has the wrong type")
    if explicit_adapter_opt_in is not True:
        raise ValueError("explicit visual+tactile adapter opt-in is required")
    if (
        visual_result.status != contract.required_visual_status
        or not visual_result.eligible_for_independent_probe
        or visual_result.rejection_reasons
        or visual_result.fixed_translation_xy_m is None
        or visual_result.uses_truth_orientation
        or visual_result.translation_source
        != "vision_semantic_mask_ray_plane_registered_height_xy"
        or visual_result.orientation_source != "registered_nominal"
    ):
        raise ValueError("visual fixed XY input is not eligible")
    # Bind the accepted delta back to the immutable adapter outputs.  A caller
    # may not replace only ``fixed_translation_xy_m`` while leaving nominal or
    # unrelated targets in the result object.
    for target_name in (
        "transport_safe_tcp",
        "axis_high_tcp",
        "preinsert_tcp",
        "engage_tcp",
    ):
        nominal = contract.visual.nominal_targets[target_name]
        expected = (
            nominal[0] + visual_result.fixed_translation_xy_m[0],
            nominal[1] + visual_result.fixed_translation_xy_m[1],
            nominal[2],
        )
        observed = visual_result.world_targets.get(target_name)
        if observed is None or not np.allclose(
            observed, expected, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("visual adapter target provenance changed")
    if tactile_ready.state is not contract.required_ready_state:
        raise ValueError(
            "tactile input has not latched the required READY state"
        )
    if tactile_ready.capture_id != visual_result.capture_id:
        raise ValueError("visual and tactile capture IDs differ")
    if tactile_ready.task_frame_id != contract.required_task_frame_id:
        raise ValueError("tactile task frame ID differs from contract")
    if tactile_ready.offset_source != (
        "accepted_force_moment_search_offset_xy"
    ):
        raise ValueError("tactile search offset source is unsupported")
    if (
        tactile_ready.truth_used_for_tactile_search
        or tactile_ready.posthoc_truth_used_for_target
    ):
        raise ValueError("truth-contaminated tactile input is forbidden")

    visual_delta = _vector(
        visual_result.fixed_translation_xy_m, 2, "visual fixed delta"
    )
    tactile_task_delta = _vector(
        tactile_ready.search_offset_task_xy_m,
        2,
        "tactile task-XY search offset",
    )
    if math.hypot(*tactile_task_delta) > (
        contract.tactile.motion.maximum_search_radius_m + 1.0e-12
    ):
        raise ValueError("tactile search offset exceeds bounded radius")
    task_rotation_tuple = _rotation_matrix(
        tactile_ready.task_rotation_world, "tactile task rotation"
    )
    task_rotation = np.asarray(task_rotation_tuple, dtype=np.float64)
    orthogonality_error = float(
        np.max(np.abs(task_rotation.T @ task_rotation - np.eye(3)))
    )
    determinant_error = abs(float(np.linalg.det(task_rotation)) - 1.0)
    expected_rotation_error = float(
        np.max(
            np.abs(
                task_rotation
                - np.asarray(
                    contract.expected_task_rotation_world,
                    dtype=np.float64,
                )
            )
        )
    )
    if (
        orthogonality_error > contract.maximum_orthogonality_error
        or determinant_error > contract.maximum_determinant_error
        or expected_rotation_error
        > contract.maximum_expected_rotation_error
        or float(np.max(np.abs(task_rotation[2, :2])))
        > contract.maximum_task_xy_world_z_component
    ):
        raise ValueError("tactile task rotation differs from hash-bound frame")
    tactile_world_delta_array = task_rotation @ np.asarray(
        (tactile_task_delta[0], tactile_task_delta[1], 0.0),
        dtype=np.float64,
    )
    tactile_world_delta = tuple(
        float(value) for value in tactile_world_delta_array
    )
    tactile_world_delta_xy = tactile_world_delta[:2]
    if any(
        abs(value) > limit + 1.0e-12
        for value, limit in zip(
            visual_delta,
            contract.visual.fixed_maximum_abs_translation_xy_m,
        )
    ):
        raise ValueError("visual fixed delta exceeds bounded domain")

    accepted_delta = (
        visual_delta[0] + tactile_world_delta_xy[0],
        visual_delta[1] + tactile_world_delta_xy[1],
    )
    accepted_world = (
        contract.nominal_fixed_xy_m[0] + accepted_delta[0],
        contract.nominal_fixed_xy_m[1] + accepted_delta[1],
    )
    tcp_targets: dict[str, tuple[float, float, float]] = {}
    arm_targets: dict[str, tuple[float, ...]] = {}
    position_errors: dict[str, float] = {}
    orientation_errors: dict[str, float] = {}
    joint_deltas: dict[str, float] = {}

    for name in TARGET_ORDER:
        seed = contract.target_seeds[name]
        target = (
            seed.tcp_position_world_m[0] + accepted_delta[0],
            seed.tcp_position_world_m[1] + accepted_delta[1],
            seed.tcp_position_world_m[2],
        )
        nominal_transform = np.asarray(
            iiwa14_grasp_tcp_transform(seed.arm_rad), dtype=np.float64
        )
        solved = solve_fixed_q7_tcp_pose(
            seed.arm_rad,
            target,
            target_rotation=nominal_transform[:3, :3],
            maximum_iterations=contract.ik.maximum_iterations,
            damping=contract.ik.damping,
        )
        solved_transform = np.asarray(
            iiwa14_grasp_tcp_transform(solved), dtype=np.float64
        )
        position_error = float(
            np.linalg.norm(
                solved_transform[:3, 3]
                - np.asarray(target, dtype=np.float64)
            )
        )
        orientation_error = _rotation_error(
            solved_transform[:3, :3], nominal_transform[:3, :3]
        )
        joint_delta = max(
            abs(value - base)
            for value, base in zip(solved, seed.arm_rad)
        )
        if not math.isclose(solved[6], seed.arm_rad[6], abs_tol=1.0e-12):
            raise ValueError(f"{name} local IK changed q7")
        if (
            position_error > contract.ik.maximum_fk_position_error_m
            or orientation_error
            > contract.ik.maximum_fk_orientation_error_rad
            or joint_delta
            > contract.ik.maximum_abs_joint_delta_from_nominal_rad
        ):
            raise ValueError(f"{name} local IK leaves bounded acceptance")
        tcp_targets[name] = target
        arm_targets[name] = solved
        position_errors[name] = position_error
        orientation_errors[name] = orientation_error
        joint_deltas[name] = joint_delta

    return VisualTactileRegraspPlan(
        capture_id=visual_result.capture_id,
        ready_source_scope=tactile_ready.source_scope,
        task_frame_id=tactile_ready.task_frame_id,
        task_rotation_world=task_rotation_tuple,
        visual_fixed_delta_xy_m=visual_delta,
        tactile_search_offset_task_xy_m=tactile_task_delta,
        tactile_search_offset_world_m=tactile_world_delta,
        tactile_search_offset_world_xy_m=tactile_world_delta_xy,
        accepted_center_delta_xy_m=accepted_delta,
        accepted_center_world_xy_m=accepted_world,
        tcp_targets_world_m=tcp_targets,
        arm_targets_rad=arm_targets,
        fk_position_errors_m=position_errors,
        fk_orientation_errors_rad=orientation_errors,
        target_joint_deltas_from_nominal_rad=joint_deltas,
    )


def build_cpu_sample_artifact(
    contract: VisualTactileRegraspAdapterContract,
    visual_result: VisualXyAdaptationResult,
    tactile_ready: TactileReadyCenterInput,
) -> dict[str, Any]:
    """Build the complete reproducible non-runtime CPU sample artifact."""

    plan = build_visual_tactile_regrasp_plan(
        contract,
        visual_result,
        tactile_ready,
        explicit_adapter_opt_in=True,
    )
    try:
        config_relative = contract.config_path.relative_to(
            contract.repository
        ).as_posix()
        module_relative = Path(__file__).resolve().relative_to(
            contract.repository
        ).as_posix()
    except ValueError as error:
        raise ValueError(
            "sample sources must remain inside repository"
        ) from error
    return {
        "artifact_schema_version": (
            "kcg_d38999_visual_tactile_regrasp_cpu_sample_v1"
        ),
        "artifact_scope": (
            "synthetic_cpu_contract_fixture_not_runtime_evidence"
        ),
        "config": {
            "path": config_relative,
            "sha256": _sha256(contract.config_path),
        },
        "module": {
            "path": module_relative,
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "input_dependencies": {
            name: {
                "path": artifact.path,
                "sha256": artifact.sha256,
            }
            for name, artifact in sorted(contract.inputs.items())
        },
        "visual_input": visual_result.to_mapping(),
        "tactile_ready_input": tactile_ready.to_mapping(),
        "plan": plan.to_mapping(),
    }


def validate_cpu_sample_artifact(
    contract: VisualTactileRegraspAdapterContract,
    visual_result: VisualXyAdaptationResult,
    tactile_ready: TactileReadyCenterInput,
    value: Any,
) -> dict[str, Any]:
    """Reject any schema, dependency, input, target, IK, or hash mutation."""

    document = _mapping(value, "CPU sample artifact")
    _exact(
        document,
        {
            "artifact_schema_version",
            "artifact_scope",
            "config",
            "module",
            "input_dependencies",
            "visual_input",
            "tactile_ready_input",
            "plan",
        },
        "CPU sample artifact",
    )
    expected = build_cpu_sample_artifact(
        contract, visual_result, tactile_ready
    )
    if dict(document) != expected:
        raise ValueError(
            "CPU sample artifact differs from exact regenerated content"
        )
    return expected


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "POST_READY_EXECUTION_ORDER",
    "READY_INPUT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TARGET_ORDER",
    "InputArtifact",
    "LocalFixedQ7Policy",
    "TactileReadyCenterInput",
    "TargetSeed",
    "VisualTactileRegraspAdapterContract",
    "VisualTactileRegraspPlan",
    "build_cpu_sample_artifact",
    "build_visual_tactile_regrasp_plan",
    "load_visual_tactile_regrasp_adapter_contract",
    "parse_tactile_ready_center",
    "validate_cpu_sample_artifact",
]
