"""Pure CPU plan for visual-XY pick continuation to 12 mm preinsert.

The module is deliberately independent from the accepted end-to-end runner.
It consumes an immutable :class:`VisualXyPickPlan`, reuses the visual fixed-end
translation already emitted by the adapter, and solves three nearby fixed-q7
targets.  No simulator pose is accepted by this API, so truth feedback cannot
silently enter the target-generation path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    verify_insertion_inputs,
)
from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform
from kcg_connector.d38999_visual_xy_control_adapter import (
    VisualXyControlAdapterContract,
    load_visual_xy_control_adapter_contract,
)
from kcg_connector.d38999_visual_xy_pick_probe import (
    VisualXyPickPlan,
    VisualXyPickProbeContract,
)


SCHEMA_VERSION = "kcg_d38999_visual_xy_preinsert_probe_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_visual_xy_preinsert_probe_v1.yaml"
)
TARGET_ORDER = ("transport_safe", "axis_high", "preinsert")


@dataclass(frozen=True)
class LocalFixedQ7Policy:
    maximum_iterations: int
    damping: float
    maximum_fk_position_error_m: float
    maximum_fk_orientation_error_rad: float
    maximum_abs_joint_delta_from_nominal_rad: float
    maximum_planned_joint_speed_rad_s: float


@dataclass(frozen=True)
class VisualXyPreinsertProbeContract:
    schema_version: str
    enabled_by_default: bool
    status: str
    input_paths: dict[str, Path]
    adapter: VisualXyControlAdapterContract
    insertion: D38999PhysicalInsertion
    insertion_input_paths: dict[str, Path]
    assembly: D38999AssemblyBaseline
    ik: LocalFixedQ7Policy
    preinsert_gap_m: float
    entry_gap_m: float
    registered_margin_before_entry_m: float
    runtime_failure_gates: dict[str, bool]
    cpu_plan_filename: str
    boundaries: dict[str, bool]


@dataclass(frozen=True)
class VisualXyPreinsertPlan:
    """Immutable continuation targets for a future same-World Isaac probe."""

    trial_id: str
    capture_id: str
    fixed_translation_xy_m: tuple[float, float]
    arm_targets_rad: dict[str, tuple[float, ...]]
    tcp_targets_world_m: dict[str, tuple[float, float, float]]
    fk_position_errors_m: dict[str, float]
    fk_orientation_errors_rad: dict[str, float]
    target_joint_deltas_from_nominal_rad: dict[str, float]
    transition_peak_joint_speeds_rad_s: dict[str, float]
    maximum_abs_joint_delta_from_nominal_rad: float
    planned_peak_joint_speed_rad_s: float
    registered_margin_before_entry_m: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "CPU_PLAN_READY_FOR_VISUAL_XY_PREINSERT_PROBE",
            "trial_id": self.trial_id,
            "capture_id": self.capture_id,
            "fixed_translation_xy_m": list(self.fixed_translation_xy_m),
            "target_order": list(TARGET_ORDER),
            "stop_stage": "PREINSERT",
            "arm_targets_rad": {
                name: list(values)
                for name, values in self.arm_targets_rad.items()
            },
            "tcp_targets_world_m": {
                name: list(values)
                for name, values in self.tcp_targets_world_m.items()
            },
            "fk_position_errors_m": dict(self.fk_position_errors_m),
            "fk_orientation_errors_rad": dict(
                self.fk_orientation_errors_rad
            ),
            "target_joint_deltas_from_nominal_rad": dict(
                self.target_joint_deltas_from_nominal_rad
            ),
            "transition_peak_joint_speeds_rad_s": dict(
                self.transition_peak_joint_speeds_rad_s
            ),
            "maximum_abs_joint_delta_from_nominal_rad": (
                self.maximum_abs_joint_delta_from_nominal_rad
            ),
            "planned_peak_joint_speed_rad_s": (
                self.planned_peak_joint_speed_rad_s
            ),
            "registered_margin_before_entry_m": (
                self.registered_margin_before_entry_m
            ),
            "translation_source": "visual_fixed_receptacle_xy",
            "z_source": "registered_nominal",
            "orientation_source": "registered_nominal_fk",
            "truth_xy_used_for_target": False,
            "truth_pose_feedback_used_for_target": False,
            "full_6d": False,
            "engage_executed": False,
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


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _resolve_input(
    value: Any, label: str, repository: Path
) -> Path:
    document = _mapping(value, label)
    _exact(document, {"path", "sha256"}, label)
    relative = Path(_text(document["path"], f"{label}.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}.path must be repository-relative")
    path = (repository / relative).resolve()
    if repository not in path.parents or not path.is_file():
        raise ValueError(f"{label}.path is missing")
    expected = _text(document["sha256"], f"{label}.sha256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError(f"{label} hash mismatch")
    return path


def _rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))
    return math.acos(cosine)


def _peak_minimum_jerk_speed(
    start: Sequence[Real], target: Sequence[Real], duration_s: float
) -> float:
    # The derivative peak of 10t^3 - 15t^4 + 6t^5 is 1.875 at t=0.5.
    return 1.875 * max(
        abs(float(right) - float(left))
        for left, right in zip(start, target)
    ) / duration_s


def load_visual_xy_preinsert_probe_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repository: str | Path | None = None,
) -> VisualXyPreinsertProbeContract:
    """Load and cross-check the disabled continuation contract."""

    config_path = Path(path).expanduser().resolve()
    root = (
        Path(repository).expanduser().resolve()
        if repository is not None
        else config_path.parents[3]
    )
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "document"
    )
    _exact(
        document,
        {
            "schema_version",
            "enabled_by_default",
            "status",
            "inputs",
            "planning",
            "local_fixed_q7_ik",
            "axial_scope",
            "runtime_failure_gates",
            "output",
            "boundaries",
        },
        "document",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("visual XY preinsert schema is unsupported")
    if document["enabled_by_default"] is not False:
        raise ValueError("visual XY preinsert must remain disabled by default")
    if document["status"] != "prepared_cpu_plan_not_physx_executed":
        raise ValueError("visual XY preinsert status is unsupported")

    inputs_raw = _mapping(document["inputs"], "inputs")
    _exact(inputs_raw, {"visual_xy_adapter", "nominal_insertion"}, "inputs")
    input_paths = {
        name: _resolve_input(inputs_raw[name], f"inputs.{name}", root)
        for name in ("visual_xy_adapter", "nominal_insertion")
    }
    adapter = load_visual_xy_control_adapter_contract(
        input_paths["visual_xy_adapter"], repository=root
    )
    insertion = load_d38999_physical_insertion(
        input_paths["nominal_insertion"]
    )
    insertion_inputs = verify_insertion_inputs(insertion, root)
    assembly = load_d38999_assembly_baseline(
        insertion_inputs["assembly_baseline"]
    )
    if adapter.input_paths["nominal_insertion"] != input_paths[
        "nominal_insertion"
    ]:
        raise ValueError("visual adapter and insertion inputs differ")

    planning = _mapping(document["planning"], "planning")
    _exact(
        planning,
        {
            "start_stage",
            "target_order",
            "stop_stage",
            "target_translation_source",
            "target_z_source",
            "target_orientation_source",
            "use_body_truth_for_target",
            "use_fixed_truth_for_target",
            "use_truth_for_iterative_correction",
            "interpolation",
        },
        "planning",
    )
    if (
        planning["start_stage"] != "VISUAL_XY_UNSUPPORTED_HOLD_PASSED"
        or tuple(planning["target_order"]) != TARGET_ORDER
        or planning["stop_stage"] != "PREINSERT"
        or planning["target_translation_source"]
        != "visual_fixed_receptacle_xy"
        or planning["target_z_source"] != "registered_nominal"
        or planning["target_orientation_source"] != "registered_nominal_fk"
        or planning["use_body_truth_for_target"] is not False
        or planning["use_fixed_truth_for_target"] is not False
        or planning["use_truth_for_iterative_correction"] is not False
        or planning["interpolation"] != "minimum_jerk"
    ):
        raise ValueError("visual XY preinsert planning scope changed")

    ik_raw = _mapping(document["local_fixed_q7_ik"], "local_fixed_q7_ik")
    _exact(
        ik_raw,
        {
            "maximum_iterations",
            "damping",
            "maximum_fk_position_error_m",
            "maximum_fk_orientation_error_rad",
            "maximum_abs_joint_delta_from_nominal_rad",
            "q7_preserved_per_target",
            "maximum_planned_joint_speed_rad_s",
        },
        "local_fixed_q7_ik",
    )
    ik = LocalFixedQ7Policy(
        maximum_iterations=_positive_integer(
            ik_raw["maximum_iterations"], "maximum_iterations"
        ),
        damping=_positive(ik_raw["damping"], "damping"),
        maximum_fk_position_error_m=_positive(
            ik_raw["maximum_fk_position_error_m"],
            "maximum_fk_position_error_m",
        ),
        maximum_fk_orientation_error_rad=_positive(
            ik_raw["maximum_fk_orientation_error_rad"],
            "maximum_fk_orientation_error_rad",
        ),
        maximum_abs_joint_delta_from_nominal_rad=_positive(
            ik_raw["maximum_abs_joint_delta_from_nominal_rad"],
            "maximum_abs_joint_delta_from_nominal_rad",
        ),
        maximum_planned_joint_speed_rad_s=_positive(
            ik_raw["maximum_planned_joint_speed_rad_s"],
            "maximum_planned_joint_speed_rad_s",
        ),
    )
    if (
        ik.maximum_iterations != 50
        or ik.damping != 1.0e-6
        or ik.maximum_fk_position_error_m != 1.0e-7
        or ik.maximum_fk_orientation_error_rad != 1.0e-7
        or ik.maximum_abs_joint_delta_from_nominal_rad != 0.150
        or ik.maximum_planned_joint_speed_rad_s
        != insertion.acceptance.maximum_joint_speed_rad_s
        or ik_raw["q7_preserved_per_target"] is not True
    ):
        raise ValueError("visual XY preinsert IK policy changed")

    axial = _mapping(document["axial_scope"], "axial_scope")
    _exact(
        axial,
        {
            "preinsert_gap_m",
            "entry_gap_m",
            "registered_margin_before_entry_m",
            "engage_target_planned",
            "insertion_target_planned",
        },
        "axial_scope",
    )
    preinsert_gap = _positive(axial["preinsert_gap_m"], "preinsert_gap_m")
    entry_gap = _positive(axial["entry_gap_m"], "entry_gap_m")
    margin = _positive(
        axial["registered_margin_before_entry_m"],
        "registered_margin_before_entry_m",
    )
    if (
        not math.isclose(
            preinsert_gap, assembly.axial_plan.preinsert_gap_m, abs_tol=1e-12
        )
        or not math.isclose(
            entry_gap, assembly.axial_plan.entry_gap_m, abs_tol=1e-12
        )
        or not math.isclose(margin, preinsert_gap - entry_gap, abs_tol=1e-12)
        or axial["engage_target_planned"] is not False
        or axial["insertion_target_planned"] is not False
    ):
        raise ValueError("visual XY preinsert axial scope changed")

    runtime_gates = dict(
        _mapping(document["runtime_failure_gates"], "runtime_failure_gates")
    )
    expected_runtime_gates = {
        "require_prior_visual_pick_pass": True,
        "require_same_world_and_capture_id": True,
        "require_no_object_pose_writes_after_start": True,
        "require_no_robot_table_fixture_or_fixed_contact": True,
        "require_no_loose_fixed_contact_before_entry": True,
        "require_all_fingers_retain_body_contact": True,
        "require_finger_torque_below_hard_stop": True,
        "actual_body_fixed_alignment_truth_evaluation_only": True,
        "actual_alignment_must_not_change_targets": True,
    }
    if runtime_gates != expected_runtime_gates:
        raise ValueError("visual XY preinsert runtime gates changed")

    output = _mapping(document["output"], "output")
    _exact(output, {"cpu_plan_filename"}, "output")
    cpu_plan_filename = _text(
        output["cpu_plan_filename"], "output.cpu_plan_filename"
    )
    output_path = Path(cpu_plan_filename)
    if (
        output_path.is_absolute()
        or len(output_path.parts) != 1
        or output_path.suffix != ".json"
    ):
        raise ValueError("output.cpu_plan_filename must be one JSON filename")

    boundaries = dict(_mapping(document["boundaries"], "boundaries"))
    expected_boundaries = {
        "explicit_opt_in_required": True,
        "existing_e2e_modified": False,
        "existing_visual_pick_default_modified": False,
        "frozen_baseline_modified": False,
        "truth_xy_used_for_target": False,
        "truth_pose_feedback_used_for_target": False,
        "orientation_estimated_from_rgbd": False,
        "full_6d_claimed": False,
        "engage_executed": False,
        "insertion_executed": False,
        "twist_executed": False,
        "home_return_executed": False,
        "collision_planned": False,
        "gpu_or_physx_validated": False,
        "production_control_authorized": False,
        "assembly_success_claimed": False,
    }
    if boundaries != expected_boundaries:
        raise ValueError("visual XY preinsert boundaries changed")

    return VisualXyPreinsertProbeContract(
        schema_version=SCHEMA_VERSION,
        enabled_by_default=False,
        status=document["status"],
        input_paths=input_paths,
        adapter=adapter,
        insertion=insertion,
        insertion_input_paths=insertion_inputs,
        assembly=assembly,
        ik=ik,
        preinsert_gap_m=preinsert_gap,
        entry_gap_m=entry_gap,
        registered_margin_before_entry_m=margin,
        runtime_failure_gates=runtime_gates,
        cpu_plan_filename=cpu_plan_filename,
        boundaries=boundaries,
    )


def build_visual_xy_preinsert_plan(
    contract: VisualXyPreinsertProbeContract,
    pick_contract: VisualXyPickProbeContract,
    pick_plan: VisualXyPickPlan,
    *,
    explicit_probe_opt_in: bool,
) -> VisualXyPreinsertPlan:
    """Solve the three open-loop continuation targets without pose truth."""

    if not isinstance(contract, VisualXyPreinsertProbeContract):
        raise ValueError("contract has the wrong type")
    if not isinstance(pick_contract, VisualXyPickProbeContract):
        raise ValueError("pick_contract has the wrong type")
    if not isinstance(pick_plan, VisualXyPickPlan):
        raise ValueError("pick_plan has the wrong type")
    if explicit_probe_opt_in is not True:
        raise ValueError("explicit visual XY preinsert opt-in is required")
    if (
        pick_contract.input_paths["visual_xy_adapter"]
        != contract.input_paths["visual_xy_adapter"]
        or pick_contract.input_paths["nominal_pick"]
        != contract.insertion_input_paths["tabletop_pick"]
    ):
        raise ValueError("pick and preinsert contracts do not share inputs")
    result = pick_plan.adapter_result
    if (
        pick_plan.trial_id != pick_contract.trial_id
        or result.capture_id != pick_plan.capture_id
        or not result.eligible_for_independent_probe
        or result.rejection_reasons
        or result.orientation_source != "registered_nominal"
        or result.uses_truth_orientation
        or result.fixed_translation_xy_m is None
    ):
        raise ValueError("visual pick plan is not eligible for continuation")

    # Ensure the immutable pick plan still contains exactly the adapter targets
    # it was built from before solving any downstream IK.
    for name, target in pick_plan.tcp_targets_world_m.items():
        if name not in result.world_targets or not np.allclose(
            target, result.world_targets[name], atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("visual pick plan target provenance changed")

    motion = contract.insertion.motion
    nominal_arms = {
        "transport_safe": motion.transport_safe_arm_rad,
        "axis_high": motion.axis_high_arm_rad,
        "preinsert": motion.preinsert_arm_rad,
    }
    nominal_tcp = {
        "transport_safe": motion.transport_safe_tcp_position_m,
        "axis_high": motion.axis_high_tcp_position_m,
        "preinsert": motion.preinsert_tcp_position_m,
    }
    durations = {
        "transport_safe": motion.transport_duration_s,
        "axis_high": motion.axis_high_duration_s,
        "preinsert": motion.preinsert_duration_s,
    }

    solved: dict[str, tuple[float, ...]] = {}
    tcp_targets: dict[str, tuple[float, float, float]] = {}
    position_errors: dict[str, float] = {}
    orientation_errors: dict[str, float] = {}
    joint_deltas: dict[str, float] = {}
    transition_speeds: dict[str, float] = {}
    previous_arm = pick_plan.arm_targets_rad["pregrasp"]
    fixed_delta = result.fixed_translation_xy_m

    for name in TARGET_ORDER:
        target_name = f"{name}_tcp"
        if target_name not in result.world_targets:
            raise ValueError(f"visual adapter omitted {target_name}")
        target = tuple(float(value) for value in result.world_targets[target_name])
        nominal = nominal_tcp[name]
        expected = (
            nominal[0] + fixed_delta[0],
            nominal[1] + fixed_delta[1],
            nominal[2],
        )
        if not np.allclose(target, expected, atol=1.0e-12, rtol=0.0):
            raise ValueError(f"{name} target is not the visual fixed translation")

        seed = nominal_arms[name]
        nominal_transform = np.asarray(
            iiwa14_grasp_tcp_transform(seed), dtype=np.float64
        )
        arm = tuple(
            float(value)
            for value in solve_fixed_q7_tcp_pose(
                seed,
                target,
                target_rotation=nominal_transform[:3, :3],
                maximum_iterations=contract.ik.maximum_iterations,
                damping=contract.ik.damping,
            )
        )
        transform = np.asarray(
            iiwa14_grasp_tcp_transform(arm), dtype=np.float64
        )
        position_error = float(
            np.linalg.norm(transform[:3, 3] - np.asarray(target))
        )
        orientation_error = _rotation_error(
            transform[:3, :3], nominal_transform[:3, :3]
        )
        joint_delta = max(abs(value - base) for value, base in zip(arm, seed))
        if not math.isclose(arm[6], seed[6], abs_tol=1.0e-12):
            raise ValueError(f"{name} local IK changed q7")
        if (
            position_error > contract.ik.maximum_fk_position_error_m
            or orientation_error
            > contract.ik.maximum_fk_orientation_error_rad
            or joint_delta
            > contract.ik.maximum_abs_joint_delta_from_nominal_rad
        ):
            raise ValueError(f"{name} local IK leaves bounded acceptance")
        transition_speed = _peak_minimum_jerk_speed(
            previous_arm, arm, durations[name]
        )
        if transition_speed > contract.ik.maximum_planned_joint_speed_rad_s:
            raise ValueError(f"{name} transition exceeds joint-speed gate")

        solved[name] = arm
        tcp_targets[name] = target
        position_errors[name] = position_error
        orientation_errors[name] = orientation_error
        joint_deltas[name] = joint_delta
        transition_speeds[name] = transition_speed
        previous_arm = arm

    return VisualXyPreinsertPlan(
        trial_id=pick_plan.trial_id,
        capture_id=pick_plan.capture_id,
        fixed_translation_xy_m=fixed_delta,
        arm_targets_rad=solved,
        tcp_targets_world_m=tcp_targets,
        fk_position_errors_m=position_errors,
        fk_orientation_errors_rad=orientation_errors,
        target_joint_deltas_from_nominal_rad=joint_deltas,
        transition_peak_joint_speeds_rad_s=transition_speeds,
        maximum_abs_joint_delta_from_nominal_rad=max(joint_deltas.values()),
        planned_peak_joint_speed_rad_s=max(transition_speeds.values()),
        registered_margin_before_entry_m=(
            contract.registered_margin_before_entry_m
        ),
    )


__all__ = (
    "DEFAULT_CONFIG_PATH",
    "SCHEMA_VERSION",
    "TARGET_ORDER",
    "LocalFixedQ7Policy",
    "VisualXyPreinsertPlan",
    "VisualXyPreinsertProbeContract",
    "build_visual_xy_preinsert_plan",
    "load_visual_xy_preinsert_probe_contract",
)
