#!/usr/bin/env python3
"""Run one bounded robot-driven TE split-nut +20 degree development probe.

The connector asset has no joint or drive.  This runner may command only the
existing robot arm and active hand joints.  Body/nut simulator poses are copied
to a post-run-only trace and cannot affect the online state machine.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

if __package__:
    from .carts_v2 import controller as control
else:
    from carts_v2 import controller as control

from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.robust.bounded_hand_base_ik import (
    solve_bounded_hand_base_ik,
)
from kcg_connector.virtual_wrist_ft_runtime import transform_wrench_to_task


SCHEMA_VERSION = "kcg_te_split_lock_increment_v1"
CONTACT_EFFORT_CAP_SCHEMA_VERSION = (
    "kcg_te_split_lock_increment_contact_effort_cap_v1"
)
EXPLICIT_TORQUE_CONTACT_SCHEMA_VERSION = (
    "kcg_te_split_lock_increment_explicit_torque_contact_v1"
)
VELOCITY_CONTACT_SCHEMA_VERSION = (
    "kcg_te_split_lock_increment_velocity_contact_v1"
)
ROBOT_ROOT = "/World/HandArm"
ARTICULATION_PATH = ROBOT_ROOT + "/Geometry/world"
FINGERTIP_MATERIAL_PATH = ROBOT_ROOT + "/PhysicsMaterials/fingertip_pad"
EXPECTED_DOF_NAMES = control.ARM_JOINT_NAMES + (
    "f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3",
)
OBJECT_ID = "te_deutsch_d38999_26fj35pn_step"
CLOSING_DOF_NAMES = ("f1j2", "f2j1", "f3j2")
CONTACT_FINGER_LABELS = ("finger_1", "finger_2", "finger_3")
FK_SENSOR_LIMIT_CLAMP_TOLERANCE_RAD = 1.0e-6
FORBIDDEN_RUNTIME_METHODS = frozenset(
    {
        "apply_force",
        "apply_forces",
        "apply_torque",
        "apply_torques",
        "apply_forces_and_torques_at_pos",
        "set_angular_velocity",
        "set_angular_velocities",
        "set_linear_velocity",
        "set_linear_velocities",
        "set_local_pose",
        "set_world_pose",
        "set_world_poses",
    }
)


def _arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository
        / "src/kcg_connector/config/te_split_lock_increment_v1.yaml",
    )
    parser.add_argument("--mode", choices=("static-check", "run"), required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--gui", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "run" and arguments.output_directory is None:
        parser.error("run mode requires --output-directory")
    if arguments.mode == "static-check" and arguments.output_directory is not None:
        parser.error("static-check does not create a run output directory")
    return arguments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _matrix(value: Any, rows: int, columns: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (rows, columns) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite {rows}x{columns}")
    return result


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite {size}-vector")
    return result


def _resolve_bound_input(
    repository: Path, row: Mapping[str, Any], label: str
) -> Path:
    relative = Path(str(row.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}.path must remain repository-relative")
    path = (repository / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(row.get("sha256", ""))
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label} sha256 changed: {observed}")
    return path


def _rotation_about_world_axis(
    axis_rotation: np.ndarray, pivot_world: np.ndarray
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = axis_rotation
    result[:3, 3] = pivot_world - axis_rotation @ pivot_world
    return result


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _load_explicit_torque_contact_variant(
    repository: Path,
    config_path: Path,
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if raw.get("hardware_authorized") is not False:
        raise PermissionError("explicit torque contact must remain simulation-only")
    expected_root_keys = {
        "schema_version",
        "hardware_authorized",
        "mode",
        "base_contract",
        "decision_evidence",
        "falsifiable_question",
        "single_variable",
        "frozen_unchanged",
        "boundaries",
    }
    if set(raw) != expected_root_keys:
        raise ValueError("explicit torque contact has unexpected root fields")
    base_path = _resolve_bound_input(
        repository,
        _mapping(raw["base_contract"], "base_contract"),
        "base_contract",
    )
    base = _mapping(
        yaml.safe_load(base_path.read_text(encoding="utf-8")), "base root"
    )
    if base.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("explicit torque contact base schema changed")

    evidence = _mapping(raw["decision_evidence"], "decision_evidence")
    trace_path = _resolve_bound_input(
        repository,
        _mapping(
            evidence["contact_effort_cap_trace"],
            "contact_effort_cap_trace",
        ),
        "contact_effort_cap_trace",
    )
    summary_path = _resolve_bound_input(
        repository,
        _mapping(
            evidence["contact_effort_cap_summary"],
            "contact_effort_cap_summary",
        ),
        "contact_effort_cap_summary",
    )
    cap_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    cap_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tare = np.mean(
        np.asarray(
            [
                row["active_efforts_nm"][7:]
                for row in cap_trace.get("samples", [])
                if row.get("phase") == "tare"
            ],
            dtype=np.float64,
        ),
        axis=0,
    )
    finger_one_rows = [
        row for row in cap_trace.get("samples", [])
        if str(row.get("phase", "")).startswith("finger_1")
    ]
    if tare.shape != (4,) or not finger_one_rows:
        raise ValueError("contact cap evidence lacks finger-one effort data")
    peak_effort = max(
        float(row["active_efforts_nm"][8]) - float(tare[1])
        for row in finger_one_rows
    )
    if not (
        cap_summary.get("earliest_failure")
        == evidence.get("observed_first_failure")
        == "FINGER_1_NO_CONTACT_SIGNAL"
        and evidence.get("observed_physical_contact_postrun") is True
        and math.isclose(
            peak_effort,
            float(evidence["observed_peak_tare_subtracted_f1_effort_nm"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(evidence["observed_f1_contact_position_rad"])
            - float(evidence["observed_f1_pregrasp_position_rad"]),
            float(evidence["observed_f1_free_motion_to_contact_rad"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and evidence.get("parked_routes")
        == [
            "SEQUENTIAL_POSITION_CONTACT_WITH_SPEED_CHANGE",
            "SEQUENTIAL_POSITION_CONTACT_WITH_DRIVE_CAP",
        ]
    ):
        raise ValueError("contact cap decision evidence changed")

    variable = _mapping(raw["single_variable"], "single_variable")
    frozen = _mapping(raw["frozen_unchanged"], "frozen_unchanged")
    if not (
        variable.get("name") == "CONTACT_CONTROL_MECHANISM"
        and variable.get("value") == "SEQUENTIAL_EXPLICIT_JOINT_TORQUE_RAMP_V1"
        and tuple(variable.get("closing_dof_names", ()))
        == ("f1j2", "f2j1", "f3j2")
        and variable.get("closing_order")
        == ["finger_1", "finger_2", "finger_3"]
        and tuple(map(float, variable.get("positive_closing_direction", ())))
        == (1.0, 1.0, 1.0)
        and math.isclose(
            float(variable.get("maximum_absolute_command_nm")),
            0.023,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(float(variable.get("linear_ramp_duration_s")), 0.5)
        and math.isclose(
            float(variable.get("confirmed_finger_hold_command_nm")), 0.020
        )
        and math.isclose(float(variable.get("per_finger_timeout_s")), 3.0)
        and math.isclose(
            float(variable.get("minimum_movement_before_confirmation_rad")),
            0.020,
        )
        and str(variable.get("minimum_movement_source", "")).startswith(
            "CONTACT_EFFORT_CAP_RUN02_OBSERVED_0P0388975_RAD_FREE_MOTION"
        )
        and math.isclose(
            float(variable.get("near_zero_velocity_threshold_rad_s")), 0.010
        )
        and math.isclose(
            float(variable.get("maximum_closing_speed_rad_s")), 0.18
        )
        and variable.get("maximum_closing_speed_action")
        == "ABORT_CLEAR_EFFORT_AND_RESTORE_DRIVES"
        and math.isclose(
            float(variable.get("measured_effort_threshold_nm")), 0.020
        )
        and int(variable.get("consecutive_confirmation_samples")) == 6
        and variable.get("active_and_confirmed_position_drive_kp_kd")
        == [0.0, 0.0]
        and variable.get("palm_and_not_yet_active_position_drive_unchanged") is True
        and variable.get("explicit_effort_written_every_physics_step") is True
        and variable.get("transition_position_bias_rule")
        == "CURRENT_ANGLE_PLUS_0P02_DIVIDED_BY_HAND_STIFFNESS"
        and variable.get("clear_explicit_effort_before_drive_restore") is True
        and variable.get("exact_gain_and_cap_readback_required") is True
        and variable.get("effort_command_buffer_zero_readback_required") is True
        and variable.get("per_step_evidence")
        == "COMMAND_AND_POSTSTEP_PROJECTED_JOINT_EFFORT"
        and variable.get("restore_in_finally_before_preload_or_release_recovery") is True
    ):
        raise ValueError("explicit torque contact mechanism contract changed")
    if not (
        math.isclose(float(frozen.get("finger_maximum_speed_rad_s")), 0.18)
        and math.isclose(float(frozen.get("contact_effort_rise_nm")), 0.02)
        and int(frozen.get("contact_consecutive_samples")) == 6
        and math.isclose(float(frozen.get("hand_stiffness")), 12.0)
        and math.isclose(float(frozen.get("hand_damping")), 2.0)
        and math.isclose(
            float(frozen.get("restored_hand_drive_maximum_effort_nm")), 1.0
        )
        and all(
            frozen.get(name) is False
            for name in (
                "pose_changed",
                "friction_changed",
                "wrist_ft_limits_changed",
                "joint_safety_limits_changed",
                "preload_changed",
            )
        )
    ):
        raise ValueError("explicit torque frozen controls changed")
    boundaries = _mapping(raw["boundaries"], "variant boundaries")
    if not (
        boundaries.get("allowed_new_robot_command_channel")
        == "active_hand_joint_effort_targets"
        and boundaries.get("online_object_or_contact_truth_used") is False
        and boundaries.get(
            "connector_pose_force_torque_or_drive_command_allowed"
        ) is False
        and boundaries.get("hardware_authorized") is False
        and boundaries.get("simulation_app_may_start_only_in_run_mode") is True
    ):
        raise PermissionError("explicit torque safety boundary changed")

    effective = copy.deepcopy(dict(base))
    effective["control"]["finger_maximum_speed_rad_s"] = 0.18
    allowed = list(effective["control"]["allowed_command_channels"])
    allowed.append("active_hand_joint_effort_targets")
    effective["control"]["allowed_command_channels"] = allowed
    effective["explicit_torque_contact"] = copy.deepcopy(dict(variable))
    variant = {
        "schema_version": EXPLICIT_TORQUE_CONTACT_SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "base_config_path": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "contact_effort_cap_trace": str(trace_path),
        "contact_effort_cap_trace_sha256": _sha256(trace_path),
        "contact_effort_cap_summary": str(summary_path),
        "contact_effort_cap_summary_sha256": _sha256(summary_path),
        "observed_peak_tare_subtracted_f1_effort_nm": peak_effort,
        "parked_routes": list(evidence["parked_routes"]),
    }
    return effective, variant


def _load_velocity_contact_variant(
    repository: Path,
    config_path: Path,
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if raw.get("hardware_authorized") is not False:
        raise PermissionError("velocity contact must remain simulation-only")
    expected_root_keys = {
        "schema_version",
        "hardware_authorized",
        "mode",
        "base_contract",
        "decision_evidence",
        "falsifiable_question",
        "single_variable",
        "frozen_unchanged",
        "boundaries",
    }
    if set(raw) != expected_root_keys:
        raise ValueError("velocity contact has unexpected root fields")
    base_path = _resolve_bound_input(
        repository,
        _mapping(raw["base_contract"], "base_contract"),
        "base_contract",
    )
    base = _mapping(
        yaml.safe_load(base_path.read_text(encoding="utf-8")), "base root"
    )
    if base.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("velocity contact base schema changed")

    evidence = _mapping(raw["decision_evidence"], "decision_evidence")
    trace_path = _resolve_bound_input(
        repository,
        _mapping(evidence["explicit_torque_trace"], "explicit_torque_trace"),
        "explicit_torque_trace",
    )
    summary_path = _resolve_bound_input(
        repository,
        _mapping(
            evidence["explicit_torque_summary"], "explicit_torque_summary"
        ),
        "explicit_torque_summary",
    )
    prior_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prior_audit = _mapping(
        prior_trace.get("explicit_torque_contact_audit"),
        "prior explicit torque audit",
    )
    prior_finger = _mapping(
        prior_audit["finger_results"][0], "prior finger-one result"
    )
    if not (
        prior_summary.get("earliest_failure")
        == evidence.get("observed_first_failure")
        == "FINGER_1_EXPLICIT_TORQUE_CONTACT_TIMEOUT"
        and math.isclose(
            float(prior_finger["start_position_before_mode_tare_rad"]),
            float(evidence["observed_mode_tare_start_position_rad"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            float(prior_finger["ramp_start_position_rad"]),
            float(evidence["observed_mode_tare_end_position_rad"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            float(prior_audit["restore"]["fallback_current_position_target"][
                "command_rad"
            ][1]),
            float(evidence["observed_ramp_final_position_rad"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            float(prior_finger["maximum_ramp_movement_rad"]),
            float(evidence["observed_ramp_maximum_movement_rad"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            float(prior_finger["peak_tare_subtracted_projected_effort_nm"]),
            float(evidence["observed_peak_tare_subtracted_effort_nm"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            float(evidence["observed_final_pad_to_nut_distance_m"]),
            0.0018783964128638595,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and evidence.get("parked_routes")
        == [
            "SEQUENTIAL_POSITION_CONTACT_WITH_SPEED_CHANGE",
            "SEQUENTIAL_POSITION_CONTACT_WITH_DRIVE_CAP",
            "SEQUENTIAL_EXPLICIT_JOINT_TORQUE_RAMP_V1",
        ]
    ):
        raise ValueError("explicit torque decision evidence changed")

    variable = _mapping(raw["single_variable"], "single_variable")
    frozen = _mapping(raw["frozen_unchanged"], "frozen_unchanged")
    if not (
        variable.get("name") == "CONTACT_CONTROL_MECHANISM"
        and variable.get("value") == "SEQUENTIAL_VELOCITY_DRIVE_CONTACT_V1"
        and tuple(variable.get("closing_dof_names", ())) == CLOSING_DOF_NAMES
        and variable.get("closing_order") == list(CONTACT_FINGER_LABELS)
        and tuple(map(float, variable.get("positive_closing_direction", ())))
        == (1.0, 1.0, 1.0)
        and math.isclose(float(variable.get("velocity_target_rad_s")), 0.020)
        and str(variable.get("velocity_target_source", "")).startswith(
            "WITH_KP_ZERO_AND_FROZEN_KD_2P0"
        )
        and math.isclose(float(variable.get("per_finger_timeout_s")), 3.0)
        and math.isclose(
            float(variable.get("minimum_movement_before_confirmation_rad")),
            0.020,
        )
        and variable.get("minimum_movement_source")
        == "CONTACT_EFFORT_CAP_RUN02_OBSERVED_0P0388975_RAD_FREE_MOTION"
        and math.isclose(
            float(variable.get("near_zero_velocity_threshold_rad_s")), 0.010
        )
        and math.isclose(
            float(variable.get("maximum_actual_closing_speed_rad_s")), 0.18
        )
        and variable.get("maximum_actual_closing_speed_action")
        == "ABORT_ZERO_VELOCITY_TARGETS_AND_RESTORE_DRIVES"
        and math.isclose(float(variable.get("maximum_drive_effort_nm")), 0.023)
        and math.isclose(
            float(variable.get("measured_effort_threshold_nm")), 0.020
        )
        and int(variable.get("consecutive_confirmation_samples")) == 6
        and variable.get("active_and_confirmed_drive_kp_kd") == [0.0, 2.0]
        and variable.get("palm_and_not_yet_active_position_drive_unchanged") is True
        and math.isclose(
            float(variable.get("active_and_confirmed_velocity_target_rad_s")),
            0.020,
        )
        and math.isclose(
            float(variable.get("future_and_palm_velocity_target_rad_s")), 0.0
        )
        and variable.get(
            "full_active_hand_velocity_vector_written_every_physics_step"
        ) is True
        and variable.get("explicit_effort_buffer_required_zero") is True
        and variable.get("set_dof_efforts_allowed_for_velocity_progress") is False
        and variable.get("transition_position_bias_rule")
        == "CURRENT_ANGLE_PLUS_0P02_DIVIDED_BY_HAND_STIFFNESS"
        and variable.get(
            "zero_velocity_targets_before_position_target_and_drive_restore"
        ) is True
        and variable.get("exact_velocity_gain_cap_readback_required") is True
        and variable.get("restore_in_finally_before_preload_or_release_recovery") is True
    ):
        raise ValueError("velocity contact mechanism contract changed")
    if not (
        math.isclose(float(frozen.get("finger_maximum_speed_rad_s")), 0.18)
        and math.isclose(float(frozen.get("contact_effort_rise_nm")), 0.02)
        and int(frozen.get("contact_consecutive_samples")) == 6
        and math.isclose(float(frozen.get("hand_stiffness")), 12.0)
        and math.isclose(float(frozen.get("hand_damping")), 2.0)
        and math.isclose(
            float(frozen.get("restored_hand_drive_maximum_effort_nm")), 1.0
        )
        and all(
            frozen.get(name) is False
            for name in (
                "pose_changed",
                "friction_changed",
                "wrist_ft_limits_changed",
                "joint_safety_limits_changed",
                "preload_changed",
            )
        )
    ):
        raise ValueError("velocity contact frozen controls changed")
    boundaries = _mapping(raw["boundaries"], "variant boundaries")
    if not (
        boundaries.get("allowed_new_robot_command_channel")
        == "active_hand_joint_velocity_targets"
        and boundaries.get("online_object_or_contact_truth_used") is False
        and boundaries.get(
            "connector_pose_force_torque_or_drive_command_allowed"
        ) is False
        and boundaries.get("hardware_authorized") is False
        and boundaries.get("simulation_app_may_start_only_in_run_mode") is True
    ):
        raise PermissionError("velocity contact safety boundary changed")

    effective = copy.deepcopy(dict(base))
    effective["control"]["finger_maximum_speed_rad_s"] = 0.18
    allowed = list(effective["control"]["allowed_command_channels"])
    allowed.append("active_hand_joint_velocity_targets")
    effective["control"]["allowed_command_channels"] = allowed
    effective["velocity_drive_contact"] = copy.deepcopy(dict(variable))
    variant = {
        "schema_version": VELOCITY_CONTACT_SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "base_config_path": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "explicit_torque_trace": str(trace_path),
        "explicit_torque_trace_sha256": _sha256(trace_path),
        "explicit_torque_summary": str(summary_path),
        "explicit_torque_summary_sha256": _sha256(summary_path),
        "observed_final_pad_to_nut_distance_m": float(
            evidence["observed_final_pad_to_nut_distance_m"]
        ),
        "parked_routes": list(evidence["parked_routes"]),
    }
    return effective, variant


def _load_effective_document(
    repository: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "root"
    )
    schema = raw.get("schema_version")
    if schema == SCHEMA_VERSION:
        return dict(raw), None
    if schema == EXPLICIT_TORQUE_CONTACT_SCHEMA_VERSION:
        return _load_explicit_torque_contact_variant(
            repository, config_path, raw
        )
    if schema == VELOCITY_CONTACT_SCHEMA_VERSION:
        return _load_velocity_contact_variant(repository, config_path, raw)
    if schema != CONTACT_EFFORT_CAP_SCHEMA_VERSION:
        raise ValueError("unexpected split-lock increment schema")
    if raw.get("hardware_authorized") is not False:
        raise PermissionError("contact effort cap variant must remain simulation-only")
    expected_root_keys = {
        "schema_version",
        "hardware_authorized",
        "mode",
        "base_contract",
        "decision_evidence",
        "falsifiable_question",
        "single_variable",
        "frozen_unchanged",
        "boundaries",
    }
    if set(raw) != expected_root_keys:
        raise ValueError("contact effort cap variant has unexpected root fields")
    base_path = _resolve_bound_input(
        repository,
        _mapping(raw["base_contract"], "base_contract"),
        "base_contract",
    )
    base = _mapping(
        yaml.safe_load(base_path.read_text(encoding="utf-8")), "base root"
    )
    if base.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("contact effort cap variant base schema changed")

    evidence = _mapping(raw["decision_evidence"], "decision_evidence")
    trace_path = _resolve_bound_input(
        repository,
        _mapping(evidence["slow_contact_trace"], "slow_contact_trace"),
        "slow_contact_trace",
    )
    summary_path = _resolve_bound_input(
        repository,
        _mapping(evidence["slow_contact_summary"], "slow_contact_summary"),
        "slow_contact_summary",
    )
    slow_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    slow_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ft_stop_rows = [
        row for row in slow_trace.get("samples", [])
        if row.get("ft_gate_enabled") is True
        and row.get("ft_limit_reason") is not None
    ]
    if len(ft_stop_rows) != 1:
        raise ValueError("slow-contact evidence must contain one first FT stop")
    stop = ft_stop_rows[0]
    bending = float(np.linalg.norm(
        np.asarray(stop["hand2arm_task_residual_wrench"], dtype=np.float64)[3:5]
    ))
    if not (
        slow_summary.get("earliest_failure")
        == evidence["observed_first_failure"]
        == "WRIST_FT_BENDING_MOMENT_NM_ABORT"
        and slow_trace.get("static_check", {}).get(
            "effective_finger_maximum_speed_rad_s"
        )
        == float(evidence["observed_finger_maximum_speed_rad_s"])
        == 0.09
        and math.isclose(
            bending,
            float(evidence["observed_peak_bending_moment_nm"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and evidence.get("speed_route_status") == "PARKED"
    ):
        raise ValueError("slow-contact decision evidence changed")

    variable = _mapping(raw["single_variable"], "single_variable")
    frozen = _mapping(raw["frozen_unchanged"], "frozen_unchanged")
    if not (
        variable.get("name")
        == "CONTACT_PHASE_ACTIVE_HAND_DOF_MAXIMUM_EFFORT_NM"
        and math.isclose(float(variable.get("value_nm")), 0.023, abs_tol=0.0)
        and tuple(variable.get("active_hand_dof_names", ()))
        == control.ACTIVE_HAND_JOINT_NAMES
        and variable.get("phase_scope")
        == ["finger_effort_tare", "sequential_contact_closure"]
        and variable.get("apply_immediately_before_tare_and_close") is True
        and variable.get("exact_readback_required_before_closing") is True
        and variable.get("restore_in_finally_immediately_after_tare_and_close") is True
        and variable.get("restore_before_preload_or_release_recovery") is True
        and math.isclose(
            float(variable.get("expected_original_and_restored_value_nm")),
            1.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("contact effort cap single-variable contract changed")
    if not (
        math.isclose(float(frozen.get("finger_maximum_speed_rad_s")), 0.18, abs_tol=0.0)
        and math.isclose(float(frozen.get("contact_effort_rise_nm")), 0.02, abs_tol=0.0)
        and int(frozen.get("contact_consecutive_samples")) == 6
        and all(
            frozen.get(name) is False
            for name in (
                "pose_changed",
                "friction_changed",
                "wrist_ft_limits_changed",
                "joint_safety_limits_changed",
                "preload_changed",
            )
        )
    ):
        raise ValueError("contact effort cap frozen controls changed")
    variant_boundaries = _mapping(raw["boundaries"], "variant boundaries")
    if not (
        variant_boundaries.get("online_object_or_contact_truth_used") is False
        and variant_boundaries.get(
            "connector_pose_force_torque_or_drive_command_allowed"
        ) is False
        and variant_boundaries.get("hardware_authorized") is False
        and variant_boundaries.get("simulation_app_may_start_only_in_run_mode") is True
    ):
        raise PermissionError("contact effort cap variant safety boundary changed")

    effective = copy.deepcopy(dict(base))
    effective["control"]["finger_maximum_speed_rad_s"] = 0.18
    effective["contact_phase_active_hand_effort_cap"] = copy.deepcopy(
        dict(variable)
    )
    variant = {
        "schema_version": CONTACT_EFFORT_CAP_SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "base_config_path": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "slow_contact_trace": str(trace_path),
        "slow_contact_trace_sha256": _sha256(trace_path),
        "slow_contact_summary": str(summary_path),
        "slow_contact_summary_sha256": _sha256(summary_path),
        "observed_slow_contact_bending_moment_nm": bending,
        "speed_route_status": "PARKED",
    }
    return effective, variant


def _load_contract(repository: Path, config_path: Path) -> dict[str, Any]:
    document, variant = _load_effective_document(repository, config_path)
    if document.get("hardware_authorized") is not False:
        raise PermissionError("hardware_authorized must remain false")
    boundaries = _mapping(document.get("boundaries"), "boundaries")
    forbidden_true = (
        "world_fixed_connector_or_fixture_allowed",
        "connector_joint_or_drive_allowed",
        "direct_pose_or_scripted_progress_allowed",
    )
    if any(boundaries.get(name) is not False for name in forbidden_true):
        raise PermissionError("a forbidden physical shortcut was enabled")
    control_doc = _mapping(document.get("control"), "control")
    if (
        control_doc.get("object_command_channels") != []
        or control_doc.get("online_object_or_contact_truth_used") is not False
        or control_doc.get("body_or_nut_pose_write_after_start_allowed") is not False
        or control_doc.get("body_or_nut_direct_force_or_torque_allowed") is not False
        or control_doc.get("body_or_nut_drive_allowed") is not False
    ):
        raise PermissionError("connector truth firewall or command boundary changed")
    if variant is None and "finger_maximum_speed_rad_s" in control_doc:
        speed = float(control_doc["finger_maximum_speed_rad_s"])
        if not math.isclose(speed, 0.09, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("the registered single-variable closing speed must be 0.09 rad/s")

    inputs_doc = _mapping(document.get("inputs"), "inputs")
    paths = {
        name: _resolve_bound_input(repository, _mapping(row, name), name)
        for name, row in inputs_doc.items()
    }
    audit = json.loads(paths["split_static_audit"].read_text(encoding="utf-8"))
    gates = _mapping(audit.get("gates"), "split static audit gates")
    if not (
        audit.get("passed") is True
        and audit.get("simulation_app_started") is False
        and gates.get("exactly_expected_rigid_bodies") is True
        and gates.get("all_rigid_bodies_dynamic") is True
        and gates.get("fixture_and_table_contact_geometry_is_visible") is True
        and gates.get("no_joint_or_constraint_prim") is True
        and gates.get("no_constraint_body_relationship") is True
        and gates.get("no_drive_schema") is True
    ):
        raise ValueError("run07 split asset static audit is not accepted")
    if audit["asset"].get("sha256") != inputs_doc["split_asset"]["sha256"]:
        raise ValueError("split asset and static audit hashes disagree")

    trace = json.loads(paths["development_grasp_trace"].read_text(encoding="utf-8"))
    if not (
        trace.get("object_id") == OBJECT_ID
        and trace.get("hardware_authorized") is False
        and trace.get("online_object_or_contact_truth_used") is False
        and trace.get("truth_audit_data_returned_to_controller") is False
        and trace.get("controller_outcome", {}).get("completed") is True
        and trace.get("evidence_binding", {}).get("robot_asset_sha256")
        == inputs_doc["robot_asset"]["sha256"]
        and trace.get("evidence_binding", {}).get("controller_source_sha256")
        == inputs_doc["controller_source"]["sha256"]
    ):
        raise ValueError("development transport grasp trace boundary changed")
    grasp = _mapping(trace.get("registered_grasp"), "registered grasp")
    motion_plan = _mapping(trace.get("motion_plan"), "motion plan")
    if not (
        grasp.get("hardware_authorized") is False
        and grasp.get("control_plan", {}).get("closing_order")
        == ["finger_1", "finger_2", "finger_3"]
        and motion_plan.get("online_object_or_contact_truth_used") is False
    ):
        raise ValueError("development grasp control plan changed")
    grasp_config = _mapping(
        yaml.safe_load(paths["development_grasp_config"].read_text(encoding="utf-8")),
        "development grasp config",
    )
    if grasp_config.get("hardware_authorized") is not False:
        raise PermissionError("development grasp hardware gate changed")
    if (
        variant is not None
        and variant["schema_version"] == CONTACT_EFFORT_CAP_SCHEMA_VERSION
    ):
        cap = _mapping(
            document["contact_phase_active_hand_effort_cap"],
            "contact_phase_active_hand_effort_cap",
        )
        dynamic = _mapping(grasp_config["dynamic"], "development dynamic")
        if not (
            math.isclose(
                float(control_doc["finger_maximum_speed_rad_s"]),
                float(dynamic["finger_maximum_speed_rad_s"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                float(dynamic["finger_maximum_speed_rad_s"]), 0.18, abs_tol=0.0
            )
            and math.isclose(
                float(dynamic["contact_effort_rise_nm"]), 0.02, abs_tol=0.0
            )
            and int(dynamic["contact_consecutive_samples"]) == 6
            and float(cap["value_nm"])
            > float(dynamic["contact_effort_rise_nm"])
            and float(cap["value_nm"])
            < float(cap["expected_original_and_restored_value_nm"])
        ):
            raise ValueError("contact effort cap is not the sole effective variable")
    elif (
        variant is not None
        and variant["schema_version"]
        == EXPLICIT_TORQUE_CONTACT_SCHEMA_VERSION
    ):
        torque = _mapping(
            document["explicit_torque_contact"], "explicit_torque_contact"
        )
        dynamic = _mapping(grasp_config["dynamic"], "development dynamic")
        expected_channels = [
            "robot_arm_joint_position_targets",
            "active_hand_joint_position_targets",
            "active_hand_joint_effort_targets",
        ]
        if not (
            control_doc["allowed_command_channels"] == expected_channels
            and math.isclose(
                float(control_doc["finger_maximum_speed_rad_s"]),
                float(dynamic["finger_maximum_speed_rad_s"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                float(dynamic["finger_maximum_speed_rad_s"]), 0.18
            )
            and math.isclose(float(dynamic["contact_effort_rise_nm"]), 0.02)
            and int(dynamic["contact_consecutive_samples"]) == 6
            and math.isclose(float(dynamic["hand_stiffness"]), 12.0)
            and math.isclose(float(dynamic["hand_damping"]), 2.0)
            and math.isclose(
                float(torque["maximum_absolute_command_nm"]), 0.023
            )
            and float(torque["measured_effort_threshold_nm"])
            == float(dynamic["contact_effort_rise_nm"])
            and int(torque["consecutive_confirmation_samples"])
            == int(dynamic["contact_consecutive_samples"])
            and float(torque["maximum_closing_speed_rad_s"])
            == float(dynamic["finger_maximum_speed_rad_s"])
        ):
            raise ValueError(
                "explicit torque contact changed a frozen control or channel"
            )
    elif (
        variant is not None
        and variant["schema_version"] == VELOCITY_CONTACT_SCHEMA_VERSION
    ):
        velocity = _mapping(
            document["velocity_drive_contact"], "velocity_drive_contact"
        )
        dynamic = _mapping(grasp_config["dynamic"], "development dynamic")
        expected_channels = [
            "robot_arm_joint_position_targets",
            "active_hand_joint_position_targets",
            "active_hand_joint_velocity_targets",
        ]
        if not (
            control_doc["allowed_command_channels"] == expected_channels
            and math.isclose(
                float(control_doc["finger_maximum_speed_rad_s"]),
                float(dynamic["finger_maximum_speed_rad_s"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(float(dynamic["finger_maximum_speed_rad_s"]), 0.18)
            and math.isclose(float(dynamic["contact_effort_rise_nm"]), 0.02)
            and int(dynamic["contact_consecutive_samples"]) == 6
            and math.isclose(float(dynamic["hand_stiffness"]), 12.0)
            and math.isclose(float(dynamic["hand_damping"]), 2.0)
            and math.isclose(float(velocity["velocity_target_rad_s"]), 0.020)
            and math.isclose(float(velocity["maximum_drive_effort_nm"]), 0.023)
            and float(velocity["measured_effort_threshold_nm"])
            == float(dynamic["contact_effort_rise_nm"])
            and int(velocity["consecutive_confirmation_samples"])
            == int(dynamic["contact_consecutive_samples"])
            and float(velocity["maximum_actual_closing_speed_rad_s"])
            == float(dynamic["finger_maximum_speed_rad_s"])
        ):
            raise ValueError(
                "velocity contact changed a frozen control or channel"
            )

    scene = _mapping(document.get("scene"), "scene")
    world_from_supplier = _matrix(
        scene["historical_world_from_supplier_row_major"], 4, 4,
        "historical_world_from_supplier",
    )
    model_from_supplier = _matrix(
        scene["split_model_from_supplier_row_major"], 4, 4,
        "split_model_from_supplier",
    )
    world_from_model = _matrix(
        scene["frozen_world_from_split_model_row_major"], 4, 4,
        "world_from_split_model",
    )
    derived_world_from_model = world_from_supplier @ np.linalg.inv(
        model_from_supplier
    )
    if not np.allclose(
        derived_world_from_model, world_from_model, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("split model placement does not preserve the frozen grasp frame")
    task_rotation = _matrix(
        scene["task_rotation_world_row_major"], 3, 3, "task rotation"
    )
    if not (
        np.allclose(task_rotation.T @ task_rotation, np.eye(3), atol=1.0e-12)
        and math.isclose(float(np.linalg.det(task_rotation)), 1.0, abs_tol=1.0e-12)
    ):
        raise ValueError("task rotation is not right-handed orthonormal")

    limits = _mapping(document.get("wrist_ft_safety"), "wrist_ft_safety")
    proxy_limits = audit["simulation_research_limits"]
    if not math.isclose(
        float(limits["maximum_tightening_moment_nm"]),
        float(proxy_limits["locking_drive_torque_cap_nm"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("runner torque cap differs from split asset contract")
    acceptance = _mapping(document.get("acceptance"), "acceptance")
    rotation_deg = float(control_doc["rotation_deg"])
    lead = float(audit["asset"].get("thread_lead_m_per_revolution", 0.0))
    if lead == 0.0:
        lead = float(audit["clearances"]["thread_expected_progress_from_frozen_probe_m"]) * 360.0 / rotation_deg
    expected_progress = lead * rotation_deg / 360.0
    if not math.isclose(
        expected_progress,
        float(acceptance["expected_axial_progress_m"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("+20 degree lead relation changed")
    return {
        "document": document,
        "paths": paths,
        "audit": audit,
        "trace": trace,
        "grasp_config": grasp_config,
        "world_from_model": world_from_model,
        "task_origin_world": _vector(
            scene["task_origin_world_m"], 3, "task origin"
        ),
        "task_rotation_world": task_rotation,
        "variant": variant,
    }


def _prepare_robot_plan(repository: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    paths = contract["paths"]
    trace = contract["trace"]
    document = contract["document"]
    inputs = load_v2_inputs(
        repository,
        config_path=paths["development_grasp_config"],
        object_id=OBJECT_ID,
    )
    solver = contract["grasp_config"]["ik"]["solver"]
    motion_plan = trace["motion_plan"]
    start_target = _matrix(
        np.asarray(motion_plan["world_from_hand_base_target"]).reshape(4, 4),
        4,
        4,
        "development hand target",
    )
    start_arm = _vector(
        motion_plan["pregrasp_arm_positions_rad"], 7, "pregrasp arm"
    )
    hand = _vector(motion_plan["final_hand_positions_rad"], 4, "final hand")
    control_doc = document["control"]
    task_origin = contract["task_origin_world"]
    task_rotation = contract["task_rotation_world"]
    task_z = task_rotation[:, 2]
    if not np.allclose(task_z, (0.0, 0.0, 1.0), atol=1.0e-12):
        raise ValueError("this narrow runner supports the frozen world +Z task axis only")
    count = int(control_doc["rotation_waypoint_count"])
    if count < 2:
        raise ValueError("rotation path needs at least two waypoints")
    degrees = np.linspace(0.0, float(control_doc["rotation_deg"]), count)
    arm_rows = [start_arm]
    target_rows = [start_target]
    errors = []
    seed = start_arm
    for index, degrees_value in enumerate(degrees[1:], 1):
        angle = math.radians(float(degrees_value))
        target = _rotation_about_world_axis(
            _rotation_z(angle), task_origin
        ) @ start_target
        arm, position_error, orientation_error, seed_index = (
            solve_bounded_hand_base_ik(
                solver,
                model=inputs.robot_model,
                hand_positions=hand,
                target_world_from_hand_base=target,
                seed_arm_positions=(seed,),
                label=f"TE_SPLIT_LOCK_{index:02d}",
            )
        )
        seed = np.asarray(arm, dtype=np.float64)
        arm_rows.append(seed)
        target_rows.append(target)
        errors.append(
            {
                "waypoint_index": index,
                "rotation_deg": float(degrees_value),
                "position_error_m": float(position_error),
                "orientation_error_rad": float(orientation_error),
                "seed_index": int(seed_index),
            }
        )
    retreat_target = np.array(target_rows[-1], copy=True)
    retreat_target[:3, 3] += (
        float(control_doc["retreat_distance_task_plus_z_m"]) * task_z
    )
    retreat_arm, position_error, orientation_error, seed_index = (
        solve_bounded_hand_base_ik(
            solver,
            model=inputs.robot_model,
            hand_positions=np.zeros(4, dtype=np.float64),
            target_world_from_hand_base=retreat_target,
            seed_arm_positions=(arm_rows[-1],),
            label="TE_SPLIT_LOCK_RETREAT",
        )
    )
    return {
        "inputs": inputs,
        "motion_plan": motion_plan,
        "rotation_arm_waypoints_rad": np.asarray(arm_rows, dtype=np.float64),
        "rotation_hand_targets_world": np.asarray(target_rows, dtype=np.float64),
        "rotation_ik": errors,
        "retreat_arm_target_rad": np.asarray(retreat_arm, dtype=np.float64),
        "retreat_hand_target_world": retreat_target,
        "retreat_ik": {
            "position_error_m": float(position_error),
            "orientation_error_rad": float(orientation_error),
            "seed_index": int(seed_index),
        },
    }


def _source_firewall_audit() -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    forbidden_calls: list[dict[str, Any]] = []
    truth_reads_outside_recorder: list[dict[str, Any]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.classes: list[str] = []
            self.functions: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.classes.append(node.name)
            self.generic_visit(node)
            self.classes.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
                location = {
                    "line": int(node.lineno),
                    "class": self.classes[-1] if self.classes else None,
                    "function": self.functions[-1] if self.functions else None,
                    "method": name,
                }
                if name in FORBIDDEN_RUNTIME_METHODS:
                    forbidden_calls.append(location)
                if name == "get_world_pose" and not (
                    self.classes and self.classes[-1] == "_TruthRecorder"
                    and self.functions and self.functions[-1] == "capture"
                ):
                    truth_reads_outside_recorder.append(location)
            self.generic_visit(node)

    Visitor().visit(tree)

    def call_names(nodes: Sequence[ast.AST]) -> set[str]:
        names: set[str] = set()
        for root in nodes:
            for node in ast.walk(root):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    names.add(node.func.attr)
        return names

    contact_cap_try_finally = any(
        "_tare_and_close" in call_names(node.body)
        and "_set_active_hand_effort_caps" in call_names(node.finalbody)
        for node in ast.walk(tree)
        if isinstance(node, ast.Try) and node.finalbody
    )
    cap_setter_names = call_names(
        [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_set_active_hand_effort_caps"
        ]
    )
    contact_cap_command_and_readback_present = {
        "set_dof_max_efforts",
        "_read_active_hand_effort_caps",
    }.issubset(cap_setter_names)
    explicit_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_explicit_torque_contact"
    ]
    explicit_call_names = call_names(explicit_functions)
    explicit_finally_restore = any(
        {
            "_write_explicit_dof_efforts",
            "_set_indexed_dof_gains",
            "_set_indexed_dof_max_efforts",
        }.issubset(call_names(node.finalbody))
        for function in explicit_functions
        for node in ast.walk(function)
        if isinstance(node, ast.Try) and node.finalbody
    )
    controller_source = source_path.with_name("carts_v2") / "controller.py"
    controller_text = controller_source.read_text(encoding="utf-8")
    explicit_torque_api_present = {
        "set_dof_efforts",
        "set_dof_gains",
        "set_dof_max_efforts",
    }.issubset(call_names([tree])) and "get_dof_projected_joint_forces" in controller_text
    explicit_per_step_hook_present = bool(
        "pre_step_hook=effort_hook" in source_text
        and "EXPLICIT_TORQUE_SPEED_LIMIT_ABORT" in source_text
        and "position >= final_hand[hand_offset]" in source_text
        and "movement >= movement_required" in source_text
        and "abs(velocity) <= velocity_threshold" in source_text
        and "measured_effort >= effort_threshold" in source_text
    )
    explicit_online_truth_absent = not any(
        name in explicit_call_names
        for name in ("get_world_pose", "get_contact", "get_full_contact_report")
    )
    velocity_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_velocity_drive_contact"
    ]
    velocity_call_names = call_names(velocity_functions)
    velocity_source = "\n".join(
        ast.get_source_segment(source_text, node) or ""
        for node in velocity_functions
    )
    velocity_finally_nodes = [
        node
        for function in velocity_functions
        for node in ast.walk(function)
        if isinstance(node, ast.Try) and node.finalbody
    ]
    velocity_finally_restore = any(
        {
            "_set_active_hand_velocity_targets",
            "_set_active_hand_position_targets",
            "_set_indexed_dof_gains",
            "_set_indexed_dof_max_efforts",
        }.issubset(call_names(node.finalbody))
        and {"advance", "step"}.isdisjoint(call_names(node.finalbody))
        for node in velocity_finally_nodes
    )
    velocity_finally_best_effort = bool(
        velocity_finally_restore
        and velocity_source.count("except Exception as error:") >= 8
        and "restore_errors.append" in velocity_source
        and "no_physics_step_between_restore_actions" in velocity_source
    )
    velocity_api_present = {
        "set_dof_velocity_targets",
        "get_dof_velocity_targets",
        "set_dof_gains",
        "set_dof_max_efforts",
    }.issubset(call_names([tree]))
    velocity_per_step_and_signal_gates_present = bool(
        "pre_step_hook=velocity_hook" in velocity_source
        and "VELOCITY_SPEED_LIMIT_ABORT" in velocity_source
        and "for driven_index in range(finger_index + 1)" in velocity_source
        and "_VELOCITY_ENDPOINT_NO_CONTACT" in velocity_source
        and "movement >= movement_required" in velocity_source
        and "abs(velocity) <= near_zero_velocity" in velocity_source
        and "measured_effort >= effort_threshold" in velocity_source
        and "VELOCITY_CONTACT_EXPLICIT_EFFORT_BUFFER_NONZERO_ABORT"
        in velocity_source
        and "VELOCITY_TARGET_READBACK_MISMATCH_ABORT" in velocity_source
        and "zero velocity target could not be confirmed before physics"
        in velocity_source
        and "if not target_audit[" in velocity_source
    )
    velocity_uses_pre_switch_tare_without_mode_step = bool(
        "pre_switch_tare = float(preload_tare[hand_offset])" in velocity_source
        and "post_switch_zero_velocity_tare_steps" in velocity_source
        and "velocity_contact_{finger_label}_mode_tare" not in velocity_source
    )
    velocity_no_explicit_effort_progress = not any(
        name in velocity_call_names
        for name in ("set_dof_efforts", "_write_explicit_dof_efforts")
    )
    velocity_online_truth_absent = not any(
        name in velocity_call_names
        for name in ("get_world_pose", "get_contact", "get_full_contact_report")
    )
    velocity_branch = source_text.find("if velocity_spec is not None:")
    velocity_call = source_text.find(
        "_run_velocity_drive_contact(", velocity_branch
    )
    torque_branch = source_text.find(
        "elif torque_spec is not None:", velocity_branch
    )
    velocity_main_priority = bool(
        velocity_branch >= 0
        and velocity_branch < velocity_call < torque_branch
    )
    return {
        "runner_source": str(source_path),
        "runner_source_sha256": _sha256(source_path),
        "forbidden_connector_mutation_calls": forbidden_calls,
        "truth_pose_reads_outside_post_step_recorder": truth_reads_outside_recorder,
        "contact_effort_cap_tare_and_close_has_finally_restore": (
            contact_cap_try_finally
        ),
        "contact_effort_cap_command_and_readback_present": (
            contact_cap_command_and_readback_present
        ),
        "explicit_torque_contact_api_present": explicit_torque_api_present,
        "explicit_torque_contact_per_step_hook_and_signal_gates_present": (
            explicit_per_step_hook_present
        ),
        "explicit_torque_contact_finally_clears_effort_and_restores_drive": (
            explicit_finally_restore
        ),
        "explicit_torque_contact_online_truth_absent": (
            explicit_online_truth_absent
        ),
        "velocity_contact_api_present": velocity_api_present,
        "velocity_contact_main_branch_precedes_parked_routes": (
            velocity_main_priority
        ),
        "velocity_contact_per_step_speed_endpoint_and_signal_gates_present": (
            velocity_per_step_and_signal_gates_present
        ),
        "velocity_contact_uses_pre_switch_tare_without_post_switch_mode_step": (
            velocity_uses_pre_switch_tare_without_mode_step
        ),
        "velocity_contact_explicit_effort_progress_absent": (
            velocity_no_explicit_effort_progress
        ),
        "velocity_contact_finally_attempts_zero_position_gain_cap_without_step": (
            velocity_finally_restore
        ),
        "velocity_contact_finally_best_effort_readbacks_present": (
            velocity_finally_best_effort
        ),
        "velocity_contact_online_truth_absent": velocity_online_truth_absent,
        "pass": bool(
            not forbidden_calls
            and not truth_reads_outside_recorder
            and contact_cap_try_finally
            and contact_cap_command_and_readback_present
            and explicit_torque_api_present
            and explicit_per_step_hook_present
            and explicit_finally_restore
            and explicit_online_truth_absent
            and velocity_api_present
            and velocity_main_priority
            and velocity_per_step_and_signal_gates_present
            and velocity_uses_pre_switch_tare_without_mode_step
            and velocity_no_explicit_effort_progress
            and velocity_finally_restore
            and velocity_finally_best_effort
            and velocity_online_truth_absent
        ),
    }


def _static_check(repository: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = _load_contract(repository, config_path)
    plan = _prepare_robot_plan(repository, contract)
    source_audit = _source_firewall_audit()
    audit = contract["audit"]
    arm_rows = plan["rotation_arm_waypoints_rad"]
    maximum_step = float(np.max(np.abs(np.diff(arm_rows, axis=0))))
    contact_cap = contract["document"].get(
        "contact_phase_active_hand_effort_cap"
    )
    explicit_torque = contract["document"].get("explicit_torque_contact")
    velocity_contact = contract["document"].get("velocity_drive_contact")
    result = {
        "schema_version": "kcg_te_split_lock_increment_static_check_v1",
        "status": "STATIC_PASS" if source_audit["pass"] else "STATIC_FAIL",
        "passed": bool(source_audit["pass"]),
        "simulation_app_started": False,
        "physical_progress_verified": False,
        "input_hashes_verified": True,
        "split_asset_static_audit_status": audit["status"],
        "connector_rigid_body_paths": audit["asset"]["rigid_body_paths"],
        "connector_joint_like_paths": audit["asset"]["joint_like_paths"],
        "connector_drive_schema_paths": audit["asset"]["drive_schema_paths"],
        "connector_constraint_relationship_paths": audit["asset"][
            "constraint_relationship_paths"
        ],
        "fixture_collision_count": len(audit["asset"]["fixture_collision_paths"]),
        "table_collision_count": len(audit["asset"]["table_collision_paths"]),
        "rotation_waypoint_count": int(len(arm_rows)),
        "effective_finger_maximum_speed_rad_s": float(
            contract["document"]["control"].get(
                "finger_maximum_speed_rad_s",
                contract["grasp_config"]["dynamic"]["finger_maximum_speed_rad_s"],
            )
        ),
        "contact_phase_active_hand_effort_cap": (
            None
            if contact_cap is None
            else {
                "value_nm": float(contact_cap["value_nm"]),
                "active_hand_dof_names": list(
                    contact_cap["active_hand_dof_names"]
                ),
                "expected_original_and_restored_value_nm": float(
                    contact_cap["expected_original_and_restored_value_nm"]
                ),
                "contact_effort_rise_nm": float(
                    contract["grasp_config"]["dynamic"][
                        "contact_effort_rise_nm"
                    ]
                ),
                "contact_consecutive_samples": int(
                    contract["grasp_config"]["dynamic"][
                        "contact_consecutive_samples"
                    ]
                ),
                "restore_before_preload_or_release_recovery": bool(
                    contact_cap["restore_before_preload_or_release_recovery"]
                ),
            }
        ),
        "explicit_torque_contact": (
            None
            if explicit_torque is None
            else {
                "mechanism": explicit_torque["value"],
                "closing_dof_names": list(
                    explicit_torque["closing_dof_names"]
                ),
                "closing_order": list(explicit_torque["closing_order"]),
                "maximum_absolute_command_nm": float(
                    explicit_torque["maximum_absolute_command_nm"]
                ),
                "minimum_movement_before_confirmation_rad": float(
                    explicit_torque[
                        "minimum_movement_before_confirmation_rad"
                    ]
                ),
                "minimum_movement_source": explicit_torque[
                    "minimum_movement_source"
                ],
                "near_zero_velocity_threshold_rad_s": float(
                    explicit_torque["near_zero_velocity_threshold_rad_s"]
                ),
                "maximum_closing_speed_rad_s": float(
                    explicit_torque["maximum_closing_speed_rad_s"]
                ),
                "maximum_closing_speed_action": explicit_torque[
                    "maximum_closing_speed_action"
                ],
                "measured_effort_threshold_nm": float(
                    explicit_torque["measured_effort_threshold_nm"]
                ),
                "consecutive_confirmation_samples": int(
                    explicit_torque["consecutive_confirmation_samples"]
                ),
                "allowed_command_channels": list(
                    contract["document"]["control"][
                        "allowed_command_channels"
                    ]
                ),
                "restore_in_finally_before_preload_or_release_recovery": bool(
                    explicit_torque[
                        "restore_in_finally_before_preload_or_release_recovery"
                    ]
                ),
            }
        ),
        "velocity_drive_contact": (
            None
            if velocity_contact is None
            else {
                "mechanism": velocity_contact["value"],
                "closing_dof_names": list(
                    velocity_contact["closing_dof_names"]
                ),
                "closing_order": list(velocity_contact["closing_order"]),
                "velocity_target_rad_s": float(
                    velocity_contact["velocity_target_rad_s"]
                ),
                "velocity_target_source": velocity_contact[
                    "velocity_target_source"
                ],
                "active_and_confirmed_drive_kp_kd": list(
                    velocity_contact["active_and_confirmed_drive_kp_kd"]
                ),
                "maximum_drive_effort_nm": float(
                    velocity_contact["maximum_drive_effort_nm"]
                ),
                "minimum_movement_before_confirmation_rad": float(
                    velocity_contact[
                        "minimum_movement_before_confirmation_rad"
                    ]
                ),
                "near_zero_velocity_threshold_rad_s": float(
                    velocity_contact["near_zero_velocity_threshold_rad_s"]
                ),
                "maximum_actual_closing_speed_rad_s": float(
                    velocity_contact["maximum_actual_closing_speed_rad_s"]
                ),
                "maximum_actual_closing_speed_action": velocity_contact[
                    "maximum_actual_closing_speed_action"
                ],
                "measured_effort_threshold_nm": float(
                    velocity_contact["measured_effort_threshold_nm"]
                ),
                "consecutive_confirmation_samples": int(
                    velocity_contact["consecutive_confirmation_samples"]
                ),
                "post_switch_zero_velocity_tare_steps": 0,
                "allowed_command_channels": list(
                    contract["document"]["control"][
                        "allowed_command_channels"
                    ]
                ),
                "set_dof_efforts_allowed_for_velocity_progress": bool(
                    velocity_contact[
                        "set_dof_efforts_allowed_for_velocity_progress"
                    ]
                ),
                "restore_in_finally_before_preload_or_release_recovery": bool(
                    velocity_contact[
                        "restore_in_finally_before_preload_or_release_recovery"
                    ]
                ),
            }
        ),
        "config_variant": contract["variant"],
        "maximum_rotation_waypoint_joint_step_rad": maximum_step,
        "maximum_rotation_ik_position_error_m": max(
            row["position_error_m"] for row in plan["rotation_ik"]
        ),
        "maximum_rotation_ik_orientation_error_rad": max(
            row["orientation_error_rad"] for row in plan["rotation_ik"]
        ),
        "retreat_ik": plan["retreat_ik"],
        "source_firewall_audit": source_audit,
        "boundaries": {
            "object_command_channels": [],
            "online_truth_used": False,
            "old_ft_monitor_calibration_claimed": False,
            "connector_joint_or_drive_present": False,
            "hardware_authorized": False,
        },
    }
    return result, contract, plan


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _read_active_hand_effort_caps(
    robot: Any, active_hand_indices: np.ndarray
) -> np.ndarray:
    values = _host_array(
        robot.get_dof_max_efforts(
            indices=0, dof_indices=active_hand_indices
        )
    ).reshape(-1)
    if (
        values.shape != (len(control.ACTIVE_HAND_JOINT_NAMES),)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise RuntimeError("active hand effort cap readback is invalid")
    return values


def _set_active_hand_effort_caps(
    robot: Any, active_hand_indices: np.ndarray, value_nm: float
) -> dict[str, Any]:
    value = float(value_nm)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("active hand effort cap must be finite and positive")
    command = np.full(
        len(control.ACTIVE_HAND_JOINT_NAMES), value, dtype=np.float32
    )
    robot.set_dof_max_efforts(
        command,
        indices=0,
        dof_indices=active_hand_indices,
    )
    observed = _read_active_hand_effort_caps(robot, active_hand_indices)
    expected_readback = command.astype(np.float64)
    return {
        "command_nm": command.tolist(),
        "readback_nm": observed.tolist(),
        "readback_matches_float32_command_exactly": bool(
            np.array_equal(observed, expected_readback)
        ),
    }


def _read_indexed_dof_gains(
    robot: Any, dof_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    stiffnesses, dampings = robot.get_dof_gains(
        indices=0, dof_indices=dof_indices
    )
    stiffness = _host_array(stiffnesses).reshape(-1)
    damping = _host_array(dampings).reshape(-1)
    if not (
        stiffness.shape == damping.shape == (len(dof_indices),)
        and np.all(np.isfinite(stiffness))
        and np.all(np.isfinite(damping))
        and np.all(stiffness >= 0.0)
        and np.all(damping >= 0.0)
    ):
        raise RuntimeError("indexed DOF gain readback is invalid")
    return stiffness, damping


def _set_indexed_dof_gains(
    robot: Any,
    dof_indices: np.ndarray,
    stiffness_nm_rad: Sequence[float],
    damping_nm_s_rad: Sequence[float],
) -> dict[str, Any]:
    stiffness = np.asarray(stiffness_nm_rad, dtype=np.float32).reshape(-1)
    damping = np.asarray(damping_nm_s_rad, dtype=np.float32).reshape(-1)
    if stiffness.shape != (len(dof_indices),) or damping.shape != stiffness.shape:
        raise ValueError("indexed DOF gain command shape is invalid")
    robot.set_dof_gains(
        stiffness,
        damping,
        indices=0,
        dof_indices=dof_indices,
        update_default_gains=False,
    )
    observed_stiffness, observed_damping = _read_indexed_dof_gains(
        robot, dof_indices
    )
    return {
        "stiffness_command_nm_rad": stiffness.tolist(),
        "damping_command_nm_s_rad": damping.tolist(),
        "stiffness_readback_nm_rad": observed_stiffness.tolist(),
        "damping_readback_nm_s_rad": observed_damping.tolist(),
        "readback_matches_float32_command_exactly": bool(
            np.array_equal(observed_stiffness, stiffness.astype(np.float64))
            and np.array_equal(observed_damping, damping.astype(np.float64))
        ),
    }


def _read_indexed_dof_max_efforts(
    robot: Any, dof_indices: np.ndarray
) -> np.ndarray:
    values = _host_array(
        robot.get_dof_max_efforts(indices=0, dof_indices=dof_indices)
    ).reshape(-1)
    if (
        values.shape != (len(dof_indices),)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise RuntimeError("indexed DOF maximum-effort readback is invalid")
    return values


def _set_indexed_dof_max_efforts(
    robot: Any, dof_indices: np.ndarray, values_nm: Sequence[float]
) -> dict[str, Any]:
    command = np.asarray(values_nm, dtype=np.float32).reshape(-1)
    if command.shape != (len(dof_indices),) or np.any(command <= 0.0):
        raise ValueError("indexed DOF maximum-effort command is invalid")
    robot.set_dof_max_efforts(
        command, indices=0, dof_indices=dof_indices
    )
    observed = _read_indexed_dof_max_efforts(robot, dof_indices)
    return {
        "command_nm": command.tolist(),
        "readback_nm": observed.tolist(),
        "readback_matches_float32_command_exactly": bool(
            np.array_equal(observed, command.astype(np.float64))
        ),
    }


def _write_explicit_dof_efforts(
    robot: Any, dof_indices: np.ndarray, commands_nm: Sequence[float]
) -> dict[str, Any]:
    command = np.asarray(commands_nm, dtype=np.float32).reshape(-1)
    if command.shape != (len(dof_indices),) or not np.all(np.isfinite(command)):
        raise ValueError("explicit DOF effort command is invalid")
    robot.set_dof_efforts(command, indices=0, dof_indices=dof_indices)
    buffer_readback = _host_array(
        robot.get_dof_efforts(indices=0, dof_indices=dof_indices)
    ).reshape(-1)
    if buffer_readback.shape != command.shape or not np.all(
        np.isfinite(buffer_readback)
    ):
        raise RuntimeError("explicit DOF actuation buffer readback is invalid")
    return {
        "command_nm": command.tolist(),
        "actuation_buffer_readback_nm": buffer_readback.tolist(),
        "buffer_readback_is_not_physical_effort_measurement": True,
        "buffer_matches_float32_command_exactly": bool(
            np.array_equal(buffer_readback, command.astype(np.float64))
        ),
    }


def _read_indexed_dof_effort_buffer(
    robot: Any, dof_indices: np.ndarray
) -> np.ndarray:
    values = _host_array(
        robot.get_dof_efforts(indices=0, dof_indices=dof_indices)
    ).reshape(-1)
    if values.shape != (len(dof_indices),) or not np.all(np.isfinite(values)):
        raise RuntimeError("explicit DOF effort buffer readback is invalid")
    return values


def _set_active_hand_position_targets(
    robot: Any,
    active_hand_indices: np.ndarray,
    targets_rad: Sequence[float],
) -> dict[str, Any]:
    command = np.asarray(targets_rad, dtype=np.float32).reshape(-1)
    if command.shape != (len(active_hand_indices),) or not np.all(
        np.isfinite(command)
    ):
        raise ValueError("active hand position target command is invalid")
    robot.set_dof_position_targets(
        command,
        indices=0,
        dof_indices=active_hand_indices,
    )
    observed = _host_array(
        robot.get_dof_position_targets(
            indices=0, dof_indices=active_hand_indices
        )
    ).reshape(-1)
    return {
        "command_rad": command.tolist(),
        "readback_rad": observed.tolist(),
        "readback_matches_float32_command_exactly": bool(
            np.array_equal(observed, command.astype(np.float64))
        ),
    }


def _set_active_hand_velocity_targets(
    robot: Any,
    active_hand_indices: np.ndarray,
    targets_rad_s: Sequence[float],
) -> dict[str, Any]:
    command = np.asarray(targets_rad_s, dtype=np.float32).reshape(-1)
    if command.shape != (len(active_hand_indices),) or not np.all(
        np.isfinite(command)
    ):
        raise ValueError("active hand velocity target command is invalid")
    robot.set_dof_velocity_targets(
        command,
        indices=0,
        dof_indices=active_hand_indices,
    )
    observed = _host_array(
        robot.get_dof_velocity_targets(
            indices=0, dof_indices=active_hand_indices
        )
    ).reshape(-1)
    return {
        "command_rad_s": command.tolist(),
        "readback_rad_s": observed.tolist(),
        "readback_matches_float32_command_exactly": bool(
            np.array_equal(observed, command.astype(np.float64))
        ),
    }


class _ExplicitTorqueContactResult:
    def __init__(
        self,
        *,
        target: np.ndarray,
        contact_targets: Sequence[float | None],
        failure_reason: str | None,
    ) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        self.contact_targets_rad = tuple(contact_targets)
        self.failure_reason = failure_reason
        self.complete = failure_reason is None and all(
            value is not None for value in contact_targets
        )
        self.finger_order = (1, 2, 3)
        self.maximum_target_delta_rad = 0.0


class _VelocityDriveContactResult:
    def __init__(
        self,
        *,
        target: np.ndarray,
        contact_targets: Sequence[float | None],
        failure_reason: str | None,
    ) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        self.contact_targets_rad = tuple(contact_targets)
        self.failure_reason = failure_reason
        self.complete = failure_reason is None and all(
            value is not None for value in contact_targets
        )
        self.finger_order = (1, 2, 3)
        self.maximum_target_delta_rad = 0.0


def _quaternion_rotation_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    value = _vector(quaternion_wxyz, 4, "quaternion")
    value /= np.linalg.norm(value)
    w, x, y, z = value
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _pose_matrix(position: Sequence[float], orientation_wxyz: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _quaternion_rotation_matrix(orientation_wxyz)
    result[:3, 3] = _vector(position, 3, "position")
    return result


class _ViewportEvidence:
    """Save fixed-view Isaac frames without returning image data to control."""

    def __init__(
        self, *, world, output: Path, task_origin_world: np.ndarray
    ) -> None:
        import omni.kit.renderer_capture
        from omni.kit.viewport.utility import get_active_viewport

        self.world = world
        self.output = output / "visuals"
        self.output.mkdir(parents=True, exist_ok=False)
        self.target = np.asarray(task_origin_world, dtype=np.float64) + np.asarray(
            (0.0, 0.0, 0.025), dtype=np.float64
        )
        self.eye = self.target + np.asarray(
            (0.30, -0.32, 0.22), dtype=np.float64
        )
        self.viewport = get_active_viewport()
        if self.viewport is None:
            raise RuntimeError("visual evidence viewport is unavailable")
        self.viewport.resolution = (1600, 900)
        self.renderer_capture = (
            omni.kit.renderer_capture.acquire_renderer_capture_interface()
        )
        self.records: list[dict[str, Any]] = []
        self.errors: list[dict[str, str]] = []

    def capture(self, name: str, *, required: bool) -> None:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            from omni.kit.viewport.utility import capture_viewport_to_file

            set_camera_view(
                eye=self.eye, target=self.target, viewport_api=self.viewport
            )
            for _ in range(8):
                self.world.render()
            path = self.output / f"{name}.png"
            capture_viewport_to_file(self.viewport, file_path=str(path))
            for _ in range(16):
                self.world.render()
            self.renderer_capture.wait_async_capture()
            for _ in range(8):
                if path.is_file() and path.stat().st_size > 0:
                    break
                self.world.render()
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"viewport capture did not produce {path}")
            self.records.append(
                {
                    "file": str(path),
                    "phase_label": name,
                    "camera_eye_world_m": self.eye.tolist(),
                    "camera_target_world_m": self.target.tolist(),
                    "image_feedback_used_by_control": False,
                }
            )
        except Exception as error:
            self.errors.append(
                {
                    "phase_label": name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if required:
                raise


class _TruthRecorder:
    """Log simulator part poses only after the online safety decision is made."""

    def __init__(
        self,
        *,
        parts: Mapping[str, Any],
        ft_articulation: Any,
        reaction_row: int,
        robot_model: Any,
        task_origin_world: np.ndarray,
        task_rotation_world: np.ndarray,
        physics_dt_s: float,
        limits: Mapping[str, Any],
    ) -> None:
        self.parts = dict(parts)
        self.ft_articulation = ft_articulation
        self.reaction_row = int(reaction_row)
        self.robot_model = robot_model
        self.task_origin_world = np.asarray(task_origin_world, dtype=np.float64)
        self.task_rotation_world = np.asarray(task_rotation_world, dtype=np.float64)
        self.physics_dt_s = float(physics_dt_s)
        self.limits = dict(limits)
        self.stepper = None
        self.samples: list[dict[str, Any]] = []
        self.tare_samples: list[np.ndarray] = []
        self.tare_task_wrench: np.ndarray | None = None
        self.first_ft_stop: dict[str, Any] | None = None
        self.gate_enabled = False
        self.recovery_observe_only = False
        self.last_active_target: np.ndarray | None = None
        self.control_finished = False
        self.maximum_fk_zero_clamp_rad = 0.0
        self.explicit_contact_step_command: dict[str, Any] | None = None
        self.explicit_contact_runtime_audit: dict[str, Any] | None = None
        self.velocity_contact_step_command: dict[str, Any] | None = None
        self.velocity_contact_runtime_audit: dict[str, Any] | None = None

    def finalize_tare(self, minimum_samples: int) -> None:
        if len(self.tare_samples) < int(minimum_samples):
            raise RuntimeError("insufficient run-specific wrist FT tare samples")
        values = np.stack(self.tare_samples)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("run-specific wrist FT tare contains nonfinite values")
        self.tare_task_wrench = np.mean(values, axis=0)
        self.gate_enabled = True

    def begin_release_recovery(self) -> None:
        self.gate_enabled = False
        self.recovery_observe_only = True

    def _robot_tcp_from_joint_state(self, active_positions: np.ndarray) -> np.ndarray:
        fk_positions = np.asarray(active_positions, dtype=np.float64).copy()
        numerical_zero = np.abs(fk_positions) < FK_SENSOR_LIMIT_CLAMP_TOLERANCE_RAD
        if np.any(numerical_zero):
            self.maximum_fk_zero_clamp_rad = max(
                self.maximum_fk_zero_clamp_rad,
                float(np.max(np.abs(fk_positions[numerical_zero]))),
            )
            fk_positions[numerical_zero] = 0.0
        links = self.robot_model.forward_kinematics(tuple(map(float, fk_positions)))
        matrix = np.asarray(links["handbase_link"], dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("robot-model handbase FK is invalid")
        return matrix

    def _read_task_wrench(self, tcp_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The articulation runs on CUDA.  Isaac's indexed API requires a
        # backend-native torch index, while the unindexed table has already
        # been validated above.  Read that table and select on the host.
        values = _host_array(self.ft_articulation.get_measured_joint_forces())
        if (
            values.ndim != 2
            or values.shape[1] != 6
            or self.reaction_row >= len(values)
            or not np.all(np.isfinite(values))
        ):
            raise RuntimeError(f"unexpected hand2arm reaction shape/value: {values.shape}")
        raw_sensor = values[self.reaction_row]
        canonical_sensor = -raw_sensor
        task = transform_wrench_to_task(
            canonical_sensor,
            tcp_world[:3, 3],
            tcp_world[:3, :3],
            self.task_origin_world,
            self.task_rotation_world,
        )
        return raw_sensor, task

    def _limit_reason(self, residual: np.ndarray) -> tuple[str | None, dict[str, float]]:
        observed = {
            "lateral_force_n": float(np.linalg.norm(residual[:2])),
            "axial_force_n": float(abs(residual[2])),
            "bending_moment_nm": float(np.linalg.norm(residual[3:5])),
            "tightening_moment_nm": float(abs(residual[5])),
        }
        limits = {
            "lateral_force_n": float(self.limits["maximum_lateral_force_n"]),
            "axial_force_n": float(self.limits["maximum_axial_force_n"]),
            "bending_moment_nm": float(self.limits["maximum_bending_moment_nm"]),
            "tightening_moment_nm": float(self.limits["maximum_tightening_moment_nm"]),
        }
        exceeded = [name for name in observed if observed[name] > limits[name]]
        return (None if not exceeded else "WRIST_FT_" + exceeded[0].upper() + "_ABORT"), observed

    def capture(
        self,
        *,
        step: int,
        phase: str,
        active_positions: Sequence[float],
        active_velocities: Sequence[float],
        active_efforts: Sequence[float],
        active_targets: Sequence[float],
        arm_control: Mapping[str, Any],
    ) -> None:
        positions = np.asarray(active_positions, dtype=np.float64)
        velocities = np.asarray(active_velocities, dtype=np.float64)
        efforts = np.asarray(active_efforts, dtype=np.float64)
        targets = np.asarray(active_targets, dtype=np.float64)
        tcp_world = self._robot_tcp_from_joint_state(positions)
        raw_wrench, task_wrench = self._read_task_wrench(tcp_world)
        if phase == "ft_free_space_tare":
            self.tare_samples.append(task_wrench.copy())
        residual = (
            None
            if self.tare_task_wrench is None
            else task_wrench - self.tare_task_wrench
        )
        limit_reason = None
        observed_limits = None
        if residual is not None:
            limit_reason, observed_limits = self._limit_reason(residual)
        if self.gate_enabled and limit_reason is not None:
            if self.first_ft_stop is None:
                self.first_ft_stop = {
                    "step": int(step),
                    "phase": str(phase),
                    "reason": limit_reason,
                    "task_residual_wrench": residual.tolist(),
                    "observed": observed_limits,
                }
            if self.stepper is None:
                raise RuntimeError("FT recorder is not attached to the joint stepper")
            self.stepper.abort_reason = limit_reason

        # Truth firewall: the control decision above is complete before these
        # simulator part poses are read.  The values are appended and never
        # returned to the stepper or state machine.
        part_truth: dict[str, Any] = {}
        for name, part in self.parts.items():
            part_position, part_orientation = (
                _host_array(value) for value in part.get_world_pose()
            )
            part_truth[name] = {
                "position_m": part_position.reshape(3).tolist(),
                "orientation_wxyz": part_orientation.reshape(4).tolist(),
            }
        self.last_active_target = targets.copy()
        self.samples.append(
            {
                "step": int(step),
                "simulation_time_s": (int(step) + 1) * self.physics_dt_s,
                "phase": str(phase),
                "active_positions_rad": positions.tolist(),
                "active_velocities_rad_s": velocities.tolist(),
                "active_efforts_nm": efforts.tolist(),
                "active_targets_rad": targets.tolist(),
                "tcp_position_world_m": tcp_world[:3, 3].tolist(),
                "tcp_rotation_world_row_major": tcp_world[:3, :3].tolist(),
                "hand2arm_raw_wrench": raw_wrench.tolist(),
                "hand2arm_task_wrench": task_wrench.tolist(),
                "hand2arm_task_tare": (
                    None
                    if self.tare_task_wrench is None
                    else self.tare_task_wrench.tolist()
                ),
                "hand2arm_task_residual_wrench": (
                    None if residual is None else residual.tolist()
                ),
                "ft_limit_reason": limit_reason,
                "ft_gate_enabled": bool(self.gate_enabled),
                "ft_recovery_observe_only": bool(self.recovery_observe_only),
                "arm_control": {
                    "drive_target_rad": arm_control["drive_target_rad"],
                    "gravity_compensation_nm": arm_control[
                        "gravity_compensation_nm"
                    ],
                    "saturated": arm_control["saturated"],
                },
                "explicit_torque_contact_control": (
                    None
                    if self.explicit_contact_step_command is None
                    else {
                        **copy.deepcopy(self.explicit_contact_step_command),
                        "poststep_projected_active_hand_efforts_nm": (
                            efforts[7:].tolist()
                        ),
                        "poststep_active_hand_positions_rad": (
                            positions[7:].tolist()
                        ),
                        "poststep_active_hand_velocities_rad_s": (
                            velocities[7:].tolist()
                        ),
                    }
                ),
                "velocity_drive_contact_control": (
                    None
                    if self.velocity_contact_step_command is None
                    else {
                        **copy.deepcopy(self.velocity_contact_step_command),
                        "poststep_projected_active_hand_efforts_nm": (
                            efforts[7:].tolist()
                        ),
                        "poststep_active_hand_positions_rad": (
                            positions[7:].tolist()
                        ),
                        "poststep_active_hand_velocities_rad_s": (
                            velocities[7:].tolist()
                        ),
                    }
                ),
                "post_step_truth_not_returned_to_control": part_truth,
            }
        )


def _run_explicit_torque_contact(
    *,
    stepper: Any,
    robot: Any,
    active_indices: np.ndarray,
    motion_plan: Mapping[str, Any],
    settings: Mapping[str, Any],
    pregrasp: Mapping[str, Any],
    contract: Mapping[str, Any],
    recorder: _TruthRecorder,
) -> tuple[
    _ExplicitTorqueContactResult,
    np.ndarray,
    dict[str, Any],
    str | None,
]:
    dt = float(settings["physics_dt_s"])
    torque = _mapping(contract, "explicit_torque_contact")
    closing_names = tuple(torque["closing_dof_names"])
    if closing_names != CLOSING_DOF_NAMES:
        raise RuntimeError("explicit torque closing DOF order changed")
    active_hand_indices = np.asarray(active_indices[7:], dtype=np.int32)
    closing_indices = np.asarray(
        [robot.dof_names.index(name) for name in closing_names],
        dtype=np.int32,
    )
    active_hand_names = tuple(control.ACTIVE_HAND_JOINT_NAMES)
    hand_offsets = tuple(active_hand_names.index(name) for name in closing_names)
    hand_offset_indices = np.asarray(hand_offsets, dtype=np.int32)
    if tuple(active_hand_indices[hand_offset_indices].tolist()) != tuple(
        closing_indices.tolist()
    ):
        raise RuntimeError("explicit torque active-hand index mapping changed")

    pregrasp_hand = np.asarray(pregrasp["hand"], dtype=np.float64)
    final_hand = np.asarray(
        motion_plan["final_hand_positions_rad"], dtype=np.float64
    )
    directions = np.asarray(
        torque["positive_closing_direction"], dtype=np.float64
    )
    maximum_command = float(torque["maximum_absolute_command_nm"])
    hold_command = float(torque["confirmed_finger_hold_command_nm"])
    ramp_steps = round(float(torque["linear_ramp_duration_s"]) / dt)
    timeout_steps = round(float(torque["per_finger_timeout_s"]) / dt)
    tare_steps = round(float(settings["effort_tare_duration_s"]) / dt)
    movement_required = float(
        torque["minimum_movement_before_confirmation_rad"]
    )
    velocity_threshold = float(torque["near_zero_velocity_threshold_rad_s"])
    speed_limit = float(torque["maximum_closing_speed_rad_s"])
    effort_threshold = float(torque["measured_effort_threshold_nm"])
    consecutive_required = int(torque["consecutive_confirmation_samples"])
    transition_bias = effort_threshold / float(settings["hand_stiffness"])
    if not (
        ramp_steps > 0
        and timeout_steps > 0
        and tare_steps > 0
        and 0.0 < hold_command <= maximum_command <= 0.023
        and movement_required == 0.020
        and velocity_threshold == 0.010
        and speed_limit == float(settings["finger_maximum_speed_rad_s"])
        and effort_threshold == float(settings["contact_effort_rise_nm"])
        and consecutive_required == int(settings["contact_consecutive_samples"])
    ):
        raise RuntimeError("explicit torque contact numeric contract changed")

    original_stiffness, original_damping = _read_indexed_dof_gains(
        robot, active_hand_indices
    )
    original_caps = _read_indexed_dof_max_efforts(
        robot, active_hand_indices
    )
    expected_stiffness = np.full(
        len(active_hand_indices), float(settings["hand_stiffness"])
    )
    expected_damping = np.full(
        len(active_hand_indices), float(settings["hand_damping"])
    )
    expected_caps = np.full(
        len(active_hand_indices), float(settings["hand_drive_maximum_effort_nm"])
    )
    original_matches = bool(
        np.array_equal(original_stiffness, expected_stiffness)
        and np.array_equal(original_damping, expected_damping)
        and np.array_equal(original_caps, expected_caps)
    )
    audit: dict[str, Any] = {
        "schema_version": "kcg_te_explicit_torque_contact_runtime_audit_v1",
        "enabled": True,
        "api": {
            "effort_command": "Articulation.set_dof_efforts",
            "effort_command_buffer_readback": "Articulation.get_dof_efforts",
            "physical_effort_measurement": (
                "Articulation.get_dof_projected_joint_forces via JointSignalStepper"
            ),
            "command_buffer_is_not_physical_measurement": True,
        },
        "active_hand_dof_names": list(active_hand_names),
        "active_hand_dof_indices": active_hand_indices.tolist(),
        "closing_dof_names": list(closing_names),
        "closing_dof_indices": closing_indices.tolist(),
        "original": {
            "stiffness_nm_rad": original_stiffness.tolist(),
            "damping_nm_s_rad": original_damping.tolist(),
            "maximum_effort_nm": original_caps.tolist(),
            "matches_frozen_position_drive": original_matches,
        },
        "maximum_absolute_command_nm": maximum_command,
        "maximum_closing_speed_rad_s": speed_limit,
        "minimum_movement_before_confirmation_rad": movement_required,
        "minimum_movement_source": torque["minimum_movement_source"],
        "near_zero_velocity_threshold_rad_s": velocity_threshold,
        "measured_effort_threshold_nm": effort_threshold,
        "consecutive_confirmation_samples": consecutive_required,
        "activation_readbacks": [],
        "finger_results": [],
        "transition": None,
        "restore": None,
        "restored_before_preload_or_release_recovery": False,
    }
    recorder.explicit_contact_runtime_audit = audit
    failure: str | None = None
    contact_targets: list[float | None] = [None, None, None]
    preload_tare = np.zeros(4, dtype=np.float64)
    transition_target = pregrasp_hand.copy()
    transition_prepared = False

    def hand_target_for_torque_mode(active_finger_count: int) -> np.ndarray:
        target = pregrasp_hand.copy()
        latest_hand = (
            pregrasp_hand
            if stepper.latest is None
            else np.asarray(stepper.latest[0][7:], dtype=np.float64)
        )
        for index in range(active_finger_count):
            target[hand_offsets[index]] = latest_hand[hand_offsets[index]]
        return target

    def effort_hook(
        commands: np.ndarray,
        *,
        phase_label: str,
        active_finger: str | None,
        confirmed_fingers: Sequence[str],
        command_fraction: float,
    ):
        def apply() -> None:
            if np.max(np.abs(commands)) > maximum_command + 1.0e-12:
                raise RuntimeError("explicit torque command exceeded frozen cap")
            buffer_audit = _write_explicit_dof_efforts(
                robot, closing_indices, commands
            )
            recorder.explicit_contact_step_command = {
                "phase_label": phase_label,
                "active_finger": active_finger,
                "confirmed_fingers": list(confirmed_fingers),
                "closing_dof_names": list(closing_names),
                "command_fraction": float(command_fraction),
                **buffer_audit,
            }

        return apply

    try:
        if not original_matches:
            failure = "EXPLICIT_TORQUE_ORIGINAL_DRIVE_READBACK_MISMATCH"

        zero_commands = np.zeros(3, dtype=np.float64)
        preload_tare_rows = []
        for _ in stepper.active_steps(tare_steps if failure is None else 0):
            latest = stepper.advance(
                "explicit_torque_preload_tare",
                pregrasp["arm"],
                pregrasp_hand,
                pre_step_hook=effort_hook(
                    zero_commands,
                    phase_label="preload_tare_zero_explicit",
                    active_finger=None,
                    confirmed_fingers=(),
                    command_fraction=0.0,
                ),
            )
            if latest is not None and stepper.abort_reason is None:
                maximum_closing_speed = max(
                    abs(float(latest[1][7 + offset]))
                    for offset in hand_offsets
                )
                if maximum_closing_speed > speed_limit:
                    failure = "EXPLICIT_TORQUE_PRELOAD_TARE_SPEED_LIMIT_ABORT"
                    stepper.abort_reason = failure
                    break
                preload_tare_rows.append(latest[2][7:].copy())
        if failure is None and stepper.abort_reason is not None:
            failure = stepper.abort_reason
        if failure is None:
            if len(preload_tare_rows) != tare_steps:
                failure = "EXPLICIT_TORQUE_PRELOAD_TARE_INCOMPLETE"
            else:
                preload_tare = np.mean(
                    np.stack(preload_tare_rows), axis=0
                )

        confirmed_labels: list[str] = []
        for finger_index, (finger_label, hand_offset) in enumerate(
            zip(CONTACT_FINGER_LABELS, hand_offsets)
        ):
            if failure is not None:
                break
            active_index = closing_indices[finger_index : finger_index + 1]
            gain_audit = _set_indexed_dof_gains(
                robot, active_index, (0.0,), (0.0,)
            )
            cap_audit = _set_indexed_dof_max_efforts(
                robot, active_index, (maximum_command,)
            )
            active_stiffness, active_damping = _read_indexed_dof_gains(
                robot, active_hand_indices
            )
            active_caps = _read_indexed_dof_max_efforts(
                robot, active_hand_indices
            )
            expected_stage_stiffness = original_stiffness.copy()
            expected_stage_damping = original_damping.copy()
            expected_stage_caps = original_caps.copy()
            for activated in range(finger_index + 1):
                offset = hand_offsets[activated]
                expected_stage_stiffness[offset] = 0.0
                expected_stage_damping[offset] = 0.0
                expected_stage_caps[offset] = np.float32(maximum_command)
            stage_matches = bool(
                gain_audit["readback_matches_float32_command_exactly"]
                and cap_audit["readback_matches_float32_command_exactly"]
                and np.array_equal(active_stiffness, expected_stage_stiffness)
                and np.array_equal(active_damping, expected_stage_damping)
                and np.array_equal(active_caps, expected_stage_caps)
            )
            audit["activation_readbacks"].append(
                {
                    "finger": finger_label,
                    "gain": gain_audit,
                    "maximum_effort": cap_audit,
                    "all_active_hand_stiffness_nm_rad": active_stiffness.tolist(),
                    "all_active_hand_damping_nm_s_rad": active_damping.tolist(),
                    "all_active_hand_maximum_effort_nm": active_caps.tolist(),
                    "palm_and_future_position_drives_unchanged": stage_matches,
                }
            )
            if not stage_matches:
                failure = f"{finger_label.upper()}_MODE_SWITCH_READBACK_MISMATCH"
                break

            start_position = float(stepper.latest[0][7 + hand_offset])
            mode_tare_rows = []
            mode_tare_peak_speed = 0.0
            for _ in stepper.active_steps(tare_steps):
                commands = np.zeros(3, dtype=np.float64)
                commands[:finger_index] = hold_command * directions[:finger_index]
                latest = stepper.advance(
                    f"explicit_torque_{finger_label}_mode_tare",
                    pregrasp["arm"],
                    hand_target_for_torque_mode(finger_index + 1),
                    pre_step_hook=effort_hook(
                        commands,
                        phase_label="zero_effort_mode_tare",
                        active_finger=finger_label,
                        confirmed_fingers=confirmed_labels,
                        command_fraction=0.0,
                    ),
                )
                if latest is None:
                    break
                speed = abs(float(latest[1][7 + hand_offset]))
                mode_tare_peak_speed = max(mode_tare_peak_speed, speed)
                maximum_closing_speed = max(
                    abs(float(latest[1][7 + offset]))
                    for offset in hand_offsets
                )
                if maximum_closing_speed > speed_limit:
                    failure = f"{finger_label.upper()}_EXPLICIT_TORQUE_SPEED_LIMIT_ABORT"
                    stepper.abort_reason = failure
                    break
                if stepper.abort_reason is not None:
                    failure = stepper.abort_reason
                    break
                mode_tare_rows.append(float(latest[2][7 + hand_offset]))
            if failure is not None:
                break
            if len(mode_tare_rows) != tare_steps:
                failure = f"{finger_label.upper()}_MODE_TARE_INCOMPLETE"
                break
            mode_tare = float(np.mean(mode_tare_rows))
            ramp_start_position = float(stepper.latest[0][7 + hand_offset])
            if (
                directions[finger_index]
                * (ramp_start_position - start_position)
                >= movement_required
            ):
                failure = f"{finger_label.upper()}_ZERO_EFFORT_TARE_MOVED_TOO_FAR"
                break

            streak = 0
            confirmed = False
            maximum_movement = 0.0
            peak_speed = 0.0
            peak_measured_effort = -math.inf
            for step_index in stepper.active_steps(timeout_steps):
                fraction = min((step_index + 1) / ramp_steps, 1.0)
                commands = np.zeros(3, dtype=np.float64)
                commands[:finger_index] = hold_command * directions[:finger_index]
                commands[finger_index] = (
                    maximum_command * fraction * directions[finger_index]
                )
                latest = stepper.advance(
                    f"explicit_torque_{finger_label}_ramp",
                    pregrasp["arm"],
                    hand_target_for_torque_mode(finger_index + 1),
                    pre_step_hook=effort_hook(
                        commands,
                        phase_label="finite_linear_torque_ramp",
                        active_finger=finger_label,
                        confirmed_fingers=confirmed_labels,
                        command_fraction=fraction,
                    ),
                )
                if latest is None:
                    break
                position = float(latest[0][7 + hand_offset])
                velocity = float(latest[1][7 + hand_offset])
                measured_effort = directions[finger_index] * (
                    float(latest[2][7 + hand_offset]) - mode_tare
                )
                movement = directions[finger_index] * (
                    position - ramp_start_position
                )
                maximum_movement = max(maximum_movement, movement)
                peak_speed = max(peak_speed, abs(velocity))
                peak_measured_effort = max(
                    peak_measured_effort, measured_effort
                )
                maximum_closing_speed = max(
                    abs(float(latest[1][7 + offset]))
                    for offset in hand_offsets
                )
                if maximum_closing_speed > speed_limit:
                    failure = f"{finger_label.upper()}_EXPLICIT_TORQUE_SPEED_LIMIT_ABORT"
                    stepper.abort_reason = failure
                    break
                if stepper.abort_reason is not None:
                    failure = stepper.abort_reason
                    break
                if position >= final_hand[hand_offset]:
                    failure = f"{finger_label.upper()}_TORQUE_ENDPOINT_NO_CONTACT"
                    break
                qualifies = bool(
                    movement >= movement_required
                    and abs(velocity) <= velocity_threshold
                    and measured_effort >= effort_threshold
                )
                streak = streak + 1 if qualifies else 0
                if streak >= consecutive_required:
                    contact_targets[finger_index] = position
                    confirmed_labels.append(finger_label)
                    confirmed = True
                    break
            audit["finger_results"].append(
                {
                    "finger": finger_label,
                    "start_position_before_mode_tare_rad": start_position,
                    "ramp_start_position_rad": ramp_start_position,
                    "mode_tare_projected_effort_nm": mode_tare,
                    "mode_tare_peak_speed_rad_s": mode_tare_peak_speed,
                    "maximum_ramp_movement_rad": maximum_movement,
                    "peak_absolute_speed_rad_s": peak_speed,
                    "peak_tare_subtracted_projected_effort_nm": (
                        None
                        if not math.isfinite(peak_measured_effort)
                        else peak_measured_effort
                    ),
                    "confirmation_streak": streak,
                    "confirmed": confirmed,
                    "contact_position_rad": contact_targets[finger_index],
                    "online_contact_truth_used": False,
                }
            )
            if failure is not None:
                break
            if not confirmed:
                failure = f"{finger_label.upper()}_EXPLICIT_TORQUE_CONTACT_TIMEOUT"
                break

        if failure is None and len(confirmed_labels) == 3:
            current_hand = np.asarray(
                stepper.latest[0][7:], dtype=np.float64
            )
            transition_target = current_hand.copy()
            for finger_index, hand_offset in enumerate(hand_offsets):
                transition_target[hand_offset] = min(
                    final_hand[hand_offset],
                    current_hand[hand_offset]
                    + directions[finger_index] * transition_bias,
                )
            transition_position_audit = _set_active_hand_position_targets(
                robot, active_hand_indices, transition_target
            )
            transition_prepared = bool(
                transition_position_audit[
                    "readback_matches_float32_command_exactly"
                ]
            )
            audit["transition"] = {
                "rule": torque["transition_position_bias_rule"],
                "bias_rad": transition_bias,
                "current_hand_positions_rad": current_hand.tolist(),
                "position_target": transition_position_audit,
                "prepared_before_effort_clear_and_drive_restore": (
                    transition_prepared
                ),
            }
            if not transition_prepared:
                failure = "EXPLICIT_TORQUE_TRANSITION_TARGET_READBACK_MISMATCH"
    finally:
        if not transition_prepared:
            current_hand = (
                pregrasp_hand.copy()
                if stepper.latest is None
                else np.asarray(stepper.latest[0][7:], dtype=np.float64)
            )
            transition_target = current_hand
            fallback_target_audit = _set_active_hand_position_targets(
                robot, active_hand_indices, transition_target
            )
        else:
            fallback_target_audit = None
        clear_audit = _write_explicit_dof_efforts(
            robot, closing_indices, np.zeros(3, dtype=np.float64)
        )
        restore_gain_audit = _set_indexed_dof_gains(
            robot,
            closing_indices,
            original_stiffness[list(hand_offsets)],
            original_damping[list(hand_offsets)],
        )
        restore_cap_audit = _set_indexed_dof_max_efforts(
            robot,
            closing_indices,
            original_caps[list(hand_offsets)],
        )
        restored_stiffness, restored_damping = _read_indexed_dof_gains(
            robot, active_hand_indices
        )
        restored_caps = _read_indexed_dof_max_efforts(
            robot, active_hand_indices
        )
        restore_matches = bool(
            clear_audit["buffer_matches_float32_command_exactly"]
            and restore_gain_audit[
                "readback_matches_float32_command_exactly"
            ]
            and restore_cap_audit[
                "readback_matches_float32_command_exactly"
            ]
            and np.array_equal(restored_stiffness, original_stiffness)
            and np.array_equal(restored_damping, original_damping)
            and np.array_equal(restored_caps, original_caps)
        )
        audit["restore"] = {
            "fallback_current_position_target": fallback_target_audit,
            "explicit_effort_clear": clear_audit,
            "gain": restore_gain_audit,
            "maximum_effort": restore_cap_audit,
            "all_active_hand_stiffness_nm_rad": restored_stiffness.tolist(),
            "all_active_hand_damping_nm_s_rad": restored_damping.tolist(),
            "all_active_hand_maximum_effort_nm": restored_caps.tolist(),
            "matches_original_exactly": restore_matches,
            "completed_at_step": int(stepper.step_index),
        }
        audit["restored_before_preload_or_release_recovery"] = True
        recorder.explicit_contact_step_command = None
        if not restore_matches:
            failure = failure or "EXPLICIT_TORQUE_DRIVE_RESTORE_READBACK_MISMATCH"

    audit["failure_reason"] = failure
    audit["complete"] = failure is None and all(
        value is not None for value in contact_targets
    )
    result = _ExplicitTorqueContactResult(
        target=transition_target,
        contact_targets=contact_targets,
        failure_reason=failure,
    )
    return result, preload_tare, audit, failure


def _run_velocity_drive_contact(
    *,
    stepper: Any,
    robot: Any,
    active_indices: np.ndarray,
    motion_plan: Mapping[str, Any],
    settings: Mapping[str, Any],
    pregrasp: Mapping[str, Any],
    contract: Mapping[str, Any],
    recorder: _TruthRecorder,
) -> tuple[
    _VelocityDriveContactResult,
    np.ndarray,
    dict[str, Any],
    str | None,
]:
    dt = float(settings["physics_dt_s"])
    velocity_contract = _mapping(contract, "velocity_drive_contact")
    closing_names = tuple(velocity_contract["closing_dof_names"])
    if closing_names != CLOSING_DOF_NAMES:
        raise RuntimeError("velocity contact closing DOF order changed")
    active_hand_indices = np.asarray(active_indices[7:], dtype=np.int32)
    closing_indices = np.asarray(
        [robot.dof_names.index(name) for name in closing_names], dtype=np.int32
    )
    active_hand_names = tuple(control.ACTIVE_HAND_JOINT_NAMES)
    hand_offsets = tuple(active_hand_names.index(name) for name in closing_names)
    if tuple(active_hand_indices[list(hand_offsets)].tolist()) != tuple(
        closing_indices.tolist()
    ):
        raise RuntimeError("velocity contact active-hand index mapping changed")

    pregrasp_hand = np.asarray(pregrasp["hand"], dtype=np.float64)
    final_hand = np.asarray(
        motion_plan["final_hand_positions_rad"], dtype=np.float64
    )
    directions = np.asarray(
        velocity_contract["positive_closing_direction"], dtype=np.float64
    )
    velocity_target = float(velocity_contract["velocity_target_rad_s"])
    timeout_steps = round(
        float(velocity_contract["per_finger_timeout_s"]) / dt
    )
    tare_steps = round(float(settings["effort_tare_duration_s"]) / dt)
    movement_required = float(
        velocity_contract["minimum_movement_before_confirmation_rad"]
    )
    near_zero_velocity = float(
        velocity_contract["near_zero_velocity_threshold_rad_s"]
    )
    speed_limit = float(
        velocity_contract["maximum_actual_closing_speed_rad_s"]
    )
    drive_cap = float(velocity_contract["maximum_drive_effort_nm"])
    effort_threshold = float(
        velocity_contract["measured_effort_threshold_nm"]
    )
    consecutive_required = int(
        velocity_contract["consecutive_confirmation_samples"]
    )
    transition_bias = effort_threshold / float(settings["hand_stiffness"])
    if not (
        timeout_steps > 0
        and tare_steps > 0
        and velocity_target == 0.020
        and movement_required == 0.020
        and near_zero_velocity == 0.010
        and speed_limit == float(settings["finger_maximum_speed_rad_s"])
        and drive_cap == 0.023
        and effort_threshold == float(settings["contact_effort_rise_nm"])
        and consecutive_required == int(settings["contact_consecutive_samples"])
    ):
        raise RuntimeError("velocity contact numeric contract changed")

    original_stiffness, original_damping = _read_indexed_dof_gains(
        robot, active_hand_indices
    )
    original_caps = _read_indexed_dof_max_efforts(
        robot, active_hand_indices
    )
    original_velocity_targets = _host_array(
        robot.get_dof_velocity_targets(
            indices=0, dof_indices=active_hand_indices
        )
    ).reshape(-1)
    explicit_effort_buffer = _read_indexed_dof_effort_buffer(
        robot, closing_indices
    )
    expected_stiffness = np.full(
        len(active_hand_indices), float(settings["hand_stiffness"])
    )
    expected_damping = np.full(
        len(active_hand_indices), float(settings["hand_damping"])
    )
    expected_caps = np.full(
        len(active_hand_indices), float(settings["hand_drive_maximum_effort_nm"])
    )
    original_matches = bool(
        np.array_equal(original_stiffness, expected_stiffness)
        and np.array_equal(original_damping, expected_damping)
        and np.array_equal(original_caps, expected_caps)
        and np.array_equal(
            original_velocity_targets,
            np.zeros(len(active_hand_indices), dtype=np.float64),
        )
        and np.array_equal(
            explicit_effort_buffer,
            np.zeros(len(closing_indices), dtype=np.float64),
        )
    )
    audit: dict[str, Any] = {
        "schema_version": "kcg_te_velocity_drive_contact_runtime_audit_v1",
        "enabled": True,
        "api": {
            "velocity_command": "Articulation.set_dof_velocity_targets",
            "velocity_readback": "Articulation.get_dof_velocity_targets",
            "physical_effort_measurement": (
                "Articulation.get_dof_projected_joint_forces via JointSignalStepper"
            ),
            "set_dof_efforts_called_by_velocity_mechanism": False,
        },
        "active_hand_dof_names": list(active_hand_names),
        "active_hand_dof_indices": active_hand_indices.tolist(),
        "closing_dof_names": list(closing_names),
        "closing_dof_indices": closing_indices.tolist(),
        "original": {
            "stiffness_nm_rad": original_stiffness.tolist(),
            "damping_nm_s_rad": original_damping.tolist(),
            "maximum_effort_nm": original_caps.tolist(),
            "velocity_targets_rad_s": original_velocity_targets.tolist(),
            "explicit_effort_buffer_nm": explicit_effort_buffer.tolist(),
            "matches_frozen_position_drive_and_zero_buffers": original_matches,
        },
        "velocity_target_rad_s": velocity_target,
        "velocity_target_source": velocity_contract["velocity_target_source"],
        "maximum_actual_closing_speed_rad_s": speed_limit,
        "minimum_movement_before_confirmation_rad": movement_required,
        "minimum_movement_source": velocity_contract["minimum_movement_source"],
        "near_zero_velocity_threshold_rad_s": near_zero_velocity,
        "maximum_drive_effort_nm": drive_cap,
        "measured_effort_threshold_nm": effort_threshold,
        "consecutive_confirmation_samples": consecutive_required,
        "activation_readbacks": [],
        "finger_results": [],
        "transition": None,
        "restore": None,
        "restored_before_preload_or_release_recovery": False,
    }
    recorder.velocity_contact_runtime_audit = audit
    failure: str | None = None
    contact_targets: list[float | None] = [None, None, None]
    preload_tare = np.zeros(4, dtype=np.float64)
    transition_target = pregrasp_hand.copy()
    transition_prepared = False

    def hand_target_for_velocity_mode(active_finger_count: int) -> np.ndarray:
        target = pregrasp_hand.copy()
        latest_hand = (
            pregrasp_hand
            if stepper.latest is None
            else np.asarray(stepper.latest[0][7:], dtype=np.float64)
        )
        for index in range(active_finger_count):
            target[hand_offsets[index]] = latest_hand[hand_offsets[index]]
        return target

    def velocity_hook(
        targets: np.ndarray,
        *,
        phase_label: str,
        active_finger: str | None,
        confirmed_fingers: Sequence[str],
    ):
        def apply() -> None:
            nonzero_effort = _read_indexed_dof_effort_buffer(
                robot, closing_indices
            )
            effective_targets = np.asarray(targets, dtype=np.float64)
            if not np.array_equal(
                nonzero_effort,
                np.zeros(len(closing_indices), dtype=np.float64),
            ):
                effective_targets = np.zeros(
                    len(active_hand_indices), dtype=np.float64
                )
                stepper.abort_reason = (
                    "VELOCITY_CONTACT_EXPLICIT_EFFORT_BUFFER_NONZERO_ABORT"
                )
            target_audit = _set_active_hand_velocity_targets(
                robot, active_hand_indices, effective_targets
            )
            zero_after_readback_mismatch = None
            if not target_audit[
                "readback_matches_float32_command_exactly"
            ]:
                zero_after_readback_mismatch = (
                    _set_active_hand_velocity_targets(
                        robot, active_hand_indices, zero_velocity_targets
                    )
                )
                stepper.abort_reason = (
                    "VELOCITY_TARGET_READBACK_MISMATCH_ABORT"
                )
                if not zero_after_readback_mismatch[
                    "readback_matches_float32_command_exactly"
                ]:
                    raise RuntimeError(
                        "zero velocity target could not be confirmed before physics"
                    )
            recorder.velocity_contact_step_command = {
                "phase_label": phase_label,
                "active_finger": active_finger,
                "confirmed_fingers": list(confirmed_fingers),
                "active_hand_dof_names": list(active_hand_names),
                "explicit_effort_buffer_nm": nonzero_effort.tolist(),
                "explicit_effort_buffer_required_zero": True,
                "velocity_target": target_audit,
                "zero_after_readback_mismatch": (
                    zero_after_readback_mismatch
                ),
            }

        return apply

    zero_velocity_targets = np.zeros(
        len(active_hand_indices), dtype=np.float64
    )
    try:
        if not original_matches:
            failure = "VELOCITY_CONTACT_ORIGINAL_DRIVE_OR_BUFFER_MISMATCH"

        preload_tare_rows = []
        for _ in stepper.active_steps(tare_steps if failure is None else 0):
            latest = stepper.advance(
                "velocity_contact_preload_tare",
                pregrasp["arm"],
                pregrasp_hand,
                pre_step_hook=velocity_hook(
                    zero_velocity_targets,
                    phase_label="preload_tare_zero_velocity",
                    active_finger=None,
                    confirmed_fingers=(),
                ),
            )
            if latest is None:
                break
            maximum_speed = max(
                abs(float(latest[1][7 + offset])) for offset in hand_offsets
            )
            if maximum_speed > speed_limit:
                failure = "VELOCITY_CONTACT_PRELOAD_TARE_SPEED_LIMIT_ABORT"
                stepper.abort_reason = failure
                break
            if stepper.abort_reason is not None:
                failure = stepper.abort_reason
                break
            preload_tare_rows.append(latest[2][7:].copy())
        if failure is None:
            if len(preload_tare_rows) != tare_steps:
                failure = "VELOCITY_CONTACT_PRELOAD_TARE_INCOMPLETE"
            else:
                preload_tare = np.mean(np.stack(preload_tare_rows), axis=0)

        confirmed_labels: list[str] = []
        for finger_index, (finger_label, hand_offset) in enumerate(
            zip(CONTACT_FINGER_LABELS, hand_offsets)
        ):
            if failure is not None:
                break
            active_index = closing_indices[finger_index : finger_index + 1]
            gain_audit = _set_indexed_dof_gains(
                robot,
                active_index,
                (0.0,),
                (float(settings["hand_damping"]),),
            )
            cap_audit = _set_indexed_dof_max_efforts(
                robot, active_index, (drive_cap,)
            )
            active_stiffness, active_damping = _read_indexed_dof_gains(
                robot, active_hand_indices
            )
            active_caps = _read_indexed_dof_max_efforts(
                robot, active_hand_indices
            )
            expected_stage_stiffness = original_stiffness.copy()
            expected_stage_damping = original_damping.copy()
            expected_stage_caps = original_caps.copy()
            for activated in range(finger_index + 1):
                offset = hand_offsets[activated]
                expected_stage_stiffness[offset] = 0.0
                expected_stage_damping[offset] = float(settings["hand_damping"])
                expected_stage_caps[offset] = np.float32(drive_cap)
            stage_matches = bool(
                gain_audit["readback_matches_float32_command_exactly"]
                and cap_audit["readback_matches_float32_command_exactly"]
                and np.array_equal(active_stiffness, expected_stage_stiffness)
                and np.array_equal(active_damping, expected_stage_damping)
                and np.array_equal(active_caps, expected_stage_caps)
            )
            audit["activation_readbacks"].append(
                {
                    "finger": finger_label,
                    "gain": gain_audit,
                    "maximum_effort": cap_audit,
                    "all_active_hand_stiffness_nm_rad": active_stiffness.tolist(),
                    "all_active_hand_damping_nm_s_rad": active_damping.tolist(),
                    "all_active_hand_maximum_effort_nm": active_caps.tolist(),
                    "palm_and_future_position_drives_unchanged": stage_matches,
                }
            )
            if not stage_matches:
                failure = f"{finger_label.upper()}_VELOCITY_MODE_READBACK_MISMATCH"
                break

            # The projected-effort baseline is sampled while the frozen
            # position drive is still active.  Once kp is set to zero there
            # is no static position restoring term; spending physics steps at
            # a zero velocity target would therefore change the starting
            # state before the tested command.  The next physics step below is
            # the first bounded +0.020 rad/s velocity-drive step.
            start_position = float(stepper.latest[0][7 + hand_offset])
            velocity_start_position = start_position
            pre_switch_tare = float(preload_tare[hand_offset])

            streak = 0
            confirmed = False
            maximum_movement = 0.0
            peak_speed = 0.0
            peak_measured_effort = -math.inf
            for _ in stepper.active_steps(timeout_steps):
                targets = np.zeros(
                    len(active_hand_indices), dtype=np.float64
                )
                for driven_index in range(finger_index + 1):
                    targets[hand_offsets[driven_index]] = (
                        directions[driven_index] * velocity_target
                    )
                latest = stepper.advance(
                    f"velocity_contact_{finger_label}_drive",
                    pregrasp["arm"],
                    hand_target_for_velocity_mode(finger_index + 1),
                    pre_step_hook=velocity_hook(
                        targets,
                        phase_label="bounded_velocity_drive",
                        active_finger=finger_label,
                        confirmed_fingers=confirmed_labels,
                    ),
                )
                if latest is None:
                    break
                position = float(latest[0][7 + hand_offset])
                velocity = float(latest[1][7 + hand_offset])
                measured_effort = directions[finger_index] * (
                    float(latest[2][7 + hand_offset]) - pre_switch_tare
                )
                movement = directions[finger_index] * (
                    position - velocity_start_position
                )
                maximum_movement = max(maximum_movement, movement)
                peak_speed = max(peak_speed, abs(velocity))
                peak_measured_effort = max(peak_measured_effort, measured_effort)
                maximum_speed = max(
                    abs(float(latest[1][7 + offset]))
                    for offset in hand_offsets
                )
                if maximum_speed > speed_limit:
                    failure = f"{finger_label.upper()}_VELOCITY_SPEED_LIMIT_ABORT"
                    stepper.abort_reason = failure
                    break
                if stepper.abort_reason is not None:
                    failure = stepper.abort_reason
                    break
                endpoint_finger = next(
                    (
                        driven_index
                        for driven_index in range(finger_index + 1)
                        if directions[driven_index]
                        * (
                            float(
                                latest[0][7 + hand_offsets[driven_index]]
                            )
                            - final_hand[hand_offsets[driven_index]]
                        )
                        >= 0.0
                    ),
                    None,
                )
                if endpoint_finger is not None:
                    failure = (
                        f"{CONTACT_FINGER_LABELS[endpoint_finger].upper()}"
                        "_VELOCITY_ENDPOINT_NO_CONTACT"
                    )
                    break
                qualifies = bool(
                    movement >= movement_required
                    and abs(velocity) <= near_zero_velocity
                    and measured_effort >= effort_threshold
                )
                streak = streak + 1 if qualifies else 0
                if streak >= consecutive_required:
                    contact_targets[finger_index] = position
                    confirmed_labels.append(finger_label)
                    confirmed = True
                    break
            audit["finger_results"].append(
                {
                    "finger": finger_label,
                    "start_position_before_velocity_drive_rad": start_position,
                    "velocity_start_position_rad": velocity_start_position,
                    "pre_switch_position_drive_tare_projected_effort_nm": (
                        pre_switch_tare
                    ),
                    "post_switch_zero_velocity_tare_steps": 0,
                    "maximum_drive_movement_rad": maximum_movement,
                    "peak_absolute_speed_rad_s": peak_speed,
                    "peak_tare_subtracted_projected_effort_nm": (
                        None
                        if not math.isfinite(peak_measured_effort)
                        else peak_measured_effort
                    ),
                    "confirmation_streak": streak,
                    "confirmed": confirmed,
                    "contact_position_rad": contact_targets[finger_index],
                    "online_contact_truth_used": False,
                }
            )
            if failure is not None:
                break
            if not confirmed:
                failure = f"{finger_label.upper()}_VELOCITY_CONTACT_TIMEOUT"
                break

        transition_prepared = failure is None and len(confirmed_labels) == 3
    finally:
        # No physics step is allowed in this hand-off.  Each recovery action
        # is attempted even if an earlier command or readback fails.
        restore_errors: list[dict[str, str]] = []
        velocity_zero_audit = None
        transition_position_audit = None
        restore_gain_audit = None
        restore_cap_audit = None
        restored_stiffness = None
        restored_damping = None
        restored_caps = None
        restored_velocity = None
        restored_effort_buffer = None
        current_hand = (
            pregrasp_hand.copy()
            if stepper.latest is None
            else np.asarray(stepper.latest[0][7:], dtype=np.float64)
        )
        transition_target = current_hand.copy()
        if transition_prepared:
            for finger_index, hand_offset in enumerate(hand_offsets):
                transition_target[hand_offset] = min(
                    final_hand[hand_offset],
                    current_hand[hand_offset]
                    + directions[finger_index] * transition_bias,
                )

        try:
            velocity_zero_audit = _set_active_hand_velocity_targets(
                robot, active_hand_indices, zero_velocity_targets
            )
        except Exception as error:
            restore_errors.append(
                {"stage": "velocity_zero", "error": repr(error)}
            )
        try:
            transition_position_audit = _set_active_hand_position_targets(
                robot, active_hand_indices, transition_target
            )
        except Exception as error:
            restore_errors.append(
                {"stage": "position_target", "error": repr(error)}
            )
        try:
            restore_gain_audit = _set_indexed_dof_gains(
                robot,
                closing_indices,
                original_stiffness[list(hand_offsets)],
                original_damping[list(hand_offsets)],
            )
        except Exception as error:
            restore_errors.append(
                {"stage": "gain_restore", "error": repr(error)}
            )
        try:
            restore_cap_audit = _set_indexed_dof_max_efforts(
                robot,
                closing_indices,
                original_caps[list(hand_offsets)],
            )
        except Exception as error:
            restore_errors.append(
                {"stage": "cap_restore", "error": repr(error)}
            )
        try:
            restored_stiffness, restored_damping = _read_indexed_dof_gains(
                robot, active_hand_indices
            )
        except Exception as error:
            restore_errors.append(
                {"stage": "gain_final_readback", "error": repr(error)}
            )
        try:
            restored_caps = _read_indexed_dof_max_efforts(
                robot, active_hand_indices
            )
        except Exception as error:
            restore_errors.append(
                {"stage": "cap_final_readback", "error": repr(error)}
            )
        try:
            restored_velocity = _host_array(
                robot.get_dof_velocity_targets(
                    indices=0, dof_indices=active_hand_indices
                )
            ).reshape(-1)
        except Exception as error:
            restore_errors.append(
                {"stage": "velocity_final_readback", "error": repr(error)}
            )
        try:
            restored_effort_buffer = _read_indexed_dof_effort_buffer(
                robot, closing_indices
            )
        except Exception as error:
            restore_errors.append(
                {"stage": "effort_buffer_final_readback", "error": repr(error)}
            )

        transition_matches = bool(
            velocity_zero_audit is not None
            and transition_position_audit is not None
            and velocity_zero_audit[
                "readback_matches_float32_command_exactly"
            ]
            and transition_position_audit[
                "readback_matches_float32_command_exactly"
            ]
        )
        restore_matches = bool(
            transition_matches
            and restore_gain_audit is not None
            and restore_cap_audit is not None
            and restored_stiffness is not None
            and restored_damping is not None
            and restored_caps is not None
            and restored_velocity is not None
            and restored_effort_buffer is not None
            and velocity_zero_audit[
                "readback_matches_float32_command_exactly"
            ]
            and restore_gain_audit[
                "readback_matches_float32_command_exactly"
            ]
            and restore_cap_audit[
                "readback_matches_float32_command_exactly"
            ]
            and np.array_equal(restored_stiffness, original_stiffness)
            and np.array_equal(restored_damping, original_damping)
            and np.array_equal(restored_caps, original_caps)
            and np.array_equal(restored_velocity, zero_velocity_targets)
            and np.array_equal(
                restored_effort_buffer,
                np.zeros(len(closing_indices), dtype=np.float64),
            )
        )
        audit["transition"] = {
            "rule": (
                velocity_contract["transition_position_bias_rule"]
                if transition_prepared
                else "CURRENT_ANGLE_WITH_NO_CLOSING_BIAS_AFTER_FAILURE"
            ),
            "bias_rad": transition_bias if transition_prepared else 0.0,
            "velocity_zero": velocity_zero_audit,
            "current_hand_positions_rad": current_hand.tolist(),
            "position_target": transition_position_audit,
            "prepared_before_drive_restore": transition_matches,
            "no_physics_step_between_commands": True,
        }
        audit["restore"] = {
            "velocity_zero": velocity_zero_audit,
            "position_target": transition_position_audit,
            "gain": restore_gain_audit,
            "maximum_effort": restore_cap_audit,
            "all_active_hand_stiffness_nm_rad": (
                None if restored_stiffness is None else restored_stiffness.tolist()
            ),
            "all_active_hand_damping_nm_s_rad": (
                None if restored_damping is None else restored_damping.tolist()
            ),
            "all_active_hand_maximum_effort_nm": (
                None if restored_caps is None else restored_caps.tolist()
            ),
            "all_active_hand_velocity_targets_rad_s": (
                None if restored_velocity is None else restored_velocity.tolist()
            ),
            "explicit_effort_buffer_nm": (
                None
                if restored_effort_buffer is None
                else restored_effort_buffer.tolist()
            ),
            "attempt_errors": restore_errors,
            "matches_original_and_zero_buffers_exactly": restore_matches,
            "completed_at_step": int(stepper.step_index),
            "no_physics_step_between_restore_actions": True,
        }
        audit["restored_before_preload_or_release_recovery"] = restore_matches
        recorder.velocity_contact_step_command = None
        if not restore_matches:
            failure = failure or "VELOCITY_CONTACT_DRIVE_RESTORE_READBACK_MISMATCH"

    audit["failure_reason"] = failure
    audit["complete"] = failure is None and all(
        value is not None for value in contact_targets
    )
    result = _VelocityDriveContactResult(
        target=transition_target,
        contact_targets=contact_targets,
        failure_reason=failure,
    )
    return result, preload_tare, audit, failure


def _relative_pose(row: Mapping[str, Any], parent: str, child: str) -> np.ndarray:
    truth = row["post_step_truth_not_returned_to_control"]
    parent_pose = _pose_matrix(
        truth[parent]["position_m"], truth[parent]["orientation_wxyz"]
    )
    child_pose = _pose_matrix(
        truth[child]["position_m"], truth[child]["orientation_wxyz"]
    )
    return np.linalg.inv(parent_pose) @ child_pose


def _signed_z_rotation_deg(reference: np.ndarray, current: np.ndarray) -> float:
    delta = reference[:3, :3].T @ current[:3, :3]
    return math.degrees(math.atan2(float(delta[1, 0]), float(delta[0, 0])))


def _last_phase(samples: Sequence[Mapping[str, Any]], phase: str) -> Mapping[str, Any] | None:
    return next((row for row in reversed(samples) if row["phase"] == phase), None)


def _evaluate(
    recorder: _TruthRecorder,
    document: Mapping[str, Any],
    *,
    primary_failure: str | None,
    release_failure: str | None,
    rotation_completed: bool,
) -> dict[str, Any]:
    if not recorder.control_finished:
        raise RuntimeError("truth evaluation is forbidden before control is finished")
    samples = recorder.samples
    reference = _last_phase(samples, "preload") or _last_phase(
        samples, "prelift_effort_check"
    )
    rotation_end = _last_phase(samples, "lock_rotation_hold") or _last_phase(
        samples, "lock_rotation"
    )
    release_end = _last_phase(samples, "release_hold")
    acceptance = document["acceptance"]
    if reference is None or rotation_end is None or release_end is None:
        return {
            "status": "DYNAMIC_FAIL",
            "passed": False,
            "earliest_failure": primary_failure or release_failure or "REQUIRED_PHASE_MISSING",
            "rotation_completed": rotation_completed,
            "truth_evaluation_started_after_control_finished": True,
        }
    body_nut_reference = _relative_pose(reference, "plug_body", "coupling_nut")
    body_nut_rotation_end = _relative_pose(rotation_end, "plug_body", "coupling_nut")
    body_nut_release_end = _relative_pose(release_end, "plug_body", "coupling_nut")
    receptacle_body_reference = _relative_pose(reference, "receptacle", "plug_body")
    receptacle_body_rotation_end = _relative_pose(
        rotation_end, "receptacle", "plug_body"
    )
    relative_rotation_deg = _signed_z_rotation_deg(
        body_nut_reference, body_nut_rotation_end
    )
    release_rotation_backdrive_deg = abs(
        _signed_z_rotation_deg(body_nut_rotation_end, body_nut_release_end)
    )
    axial_progress_m = -float(
        body_nut_rotation_end[2, 3] - body_nut_reference[2, 3]
    )
    release_progress_backdrive_m = abs(
        float(body_nut_release_end[2, 3] - body_nut_rotation_end[2, 3])
    )
    lead_error_m = abs(
        axial_progress_m - float(acceptance["expected_axial_progress_m"])
    )
    body_wrong_rotation_deg = abs(
        _signed_z_rotation_deg(
            receptacle_body_reference, receptacle_body_rotation_end
        )
    )
    open_error = float(
        np.max(
            np.abs(
                np.asarray(release_end["active_positions_rad"][7:], dtype=np.float64)
                - np.asarray(document["control"]["full_hand_open_target_rad"])
            )
        )
    )
    release_start = _last_phase(samples, "complete_hand_release")
    retreat_m = (
        None
        if release_start is None
        else float(
            np.dot(
                np.asarray(release_end["tcp_position_world_m"])
                - np.asarray(release_start["tcp_position_world_m"]),
                np.asarray(document["scene"]["task_rotation_world_row_major"])[:, 2],
            )
        )
    )
    residuals = [
        np.asarray(row["hand2arm_task_residual_wrench"], dtype=np.float64)
        for row in samples
        if row["hand2arm_task_residual_wrench"] is not None
    ]
    residual_array = np.stack(residuals) if residuals else np.empty((0, 6))
    peaks = {
        "maximum_lateral_force_n": (
            None
            if not len(residual_array)
            else float(np.max(np.linalg.norm(residual_array[:, :2], axis=1)))
        ),
        "maximum_axial_force_n": (
            None if not len(residual_array) else float(np.max(np.abs(residual_array[:, 2])))
        ),
        "maximum_bending_moment_nm": (
            None
            if not len(residual_array)
            else float(np.max(np.linalg.norm(residual_array[:, 3:5], axis=1)))
        ),
        "maximum_tightening_moment_nm": (
            None if not len(residual_array) else float(np.max(np.abs(residual_array[:, 5])))
        ),
    }
    gates = {
        "rotation_motion_completed": bool(rotation_completed),
        "relative_nut_rotation_in_frozen_range": float(
            acceptance["minimum_relative_nut_rotation_deg"]
        )
        <= relative_rotation_deg
        <= float(acceptance["maximum_relative_nut_rotation_deg"]),
        "lead_relation_error_within_frozen_bound": lead_error_m
        <= float(acceptance["maximum_lead_relation_absolute_error_m"]),
        "plug_body_did_not_wrongly_follow_rotation": body_wrong_rotation_deg
        <= float(acceptance["maximum_plug_body_relative_receptacle_rotation_deg"]),
        "release_rotation_backdrive_within_bound": release_rotation_backdrive_deg
        <= float(acceptance["maximum_release_rotation_backdrive_deg"]),
        "release_progress_backdrive_within_bound": release_progress_backdrive_m
        <= float(acceptance["maximum_release_progress_backdrive_m"]),
        "three_finger_hand_fully_open": open_error
        <= float(acceptance["maximum_open_hand_joint_error_rad"]),
        "robot_retreat_reached": retreat_m is not None
        and retreat_m >= float(acceptance["minimum_actual_retreat_m"]),
        "no_ft_safety_stop": recorder.first_ft_stop is None,
        "no_joint_or_other_control_failure": primary_failure is None
        and release_failure is None,
    }
    passed = all(gates.values())
    return {
        "status": "DYNAMIC_PASS" if passed else "DYNAMIC_FAIL",
        "passed": passed,
        "scope": "SINGLE_SIMULATION_ONLY_SMALL_ANGLE_CONTACT_RELATION_PROBE",
        "earliest_failure": (
            None
            if passed
            else primary_failure
            or release_failure
            or next(name for name, value in gates.items() if not value)
        ),
        "measured": {
            "relative_nut_rotation_deg": relative_rotation_deg,
            "axial_locking_progress_m": axial_progress_m,
            "lead_relation_absolute_error_m": lead_error_m,
            "plug_body_relative_receptacle_rotation_deg": body_wrong_rotation_deg,
            "release_rotation_backdrive_deg": release_rotation_backdrive_deg,
            "release_progress_backdrive_m": release_progress_backdrive_m,
            "open_hand_joint_error_rad": open_error,
            "robot_retreat_task_plus_z_m": retreat_m,
            "wrist_ft_task_residual_peaks": peaks,
        },
        "gates": gates,
        "first_ft_safety_stop": recorder.first_ft_stop,
        "truth_evaluation_started_after_control_finished": True,
        "online_object_or_contact_truth_used": False,
        "body_or_nut_pose_writes_after_start": 0,
        "direct_body_or_nut_force_or_torque_commands": 0,
        "connector_joint_or_drive_commands": 0,
        "full_locking_claimed": False,
    }


def _execute_run(
    repository: Path,
    config_path: Path,
    static_result: Mapping[str, Any],
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    output: Path,
    *,
    gui: bool,
) -> int:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    recorder = None
    visuals = None
    contact_effort_cap_audit = None
    explicit_torque_contact_audit = None
    velocity_drive_contact_audit = None
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade

        document = contract["document"]
        control_doc = document["control"]
        dynamic = dict(contract["grasp_config"]["dynamic"])
        if "finger_maximum_speed_rad_s" in control_doc:
            dynamic["finger_maximum_speed_rad_s"] = float(
                control_doc["finger_maximum_speed_rad_s"]
            )
        dynamic["required_closing_joint_effort_nm"] = list(
            contract["trace"]["required_closing_joint_effort_nm"]
        )
        dynamic["finger_preload_scales"] = list(
            contract["trace"]["effective_finger_preload_scales"]
        )
        if not math.isclose(
            float(dynamic["physics_dt_s"]),
            float(control_doc["physics_dt_s"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("grasp and lock runner physics_dt differ")

        World.clear_instance()
        SimulationManager.set_physics_sim_device("cuda:0")
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(dynamic["physics_dt_s"]),
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cuda:0",
            sim_params={"use_gpu_pipeline": True},
        )
        stage = get_current_stage()
        add_reference_to_stage(str(contract["paths"]["robot_asset"]), ROBOT_ROOT)
        reference_root = str(document["scene"]["reference_root"])
        add_reference_to_stage(str(contract["paths"]["split_asset"]), reference_root)
        reference_prim = stage.GetPrimAtPath(reference_root)
        reference_xform = UsdGeom.Xformable(reference_prim)
        if reference_xform.GetOrderedXformOps():
            raise RuntimeError("split reference root unexpectedly has a transform stack")
        reference_xform.AddTransformOp().Set(
            Gf.Matrix4d(*contract["world_from_model"].T.ravel().tolist())
        )
        dome = UsdLux.DomeLight.Define(
            stage, "/World/TESplitLockDevelopmentEvidenceDome"
        )
        dome.CreateIntensityAttr(850.0)
        dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
        key_light = UsdLux.DistantLight.Define(
            stage, "/World/TESplitLockDevelopmentEvidenceKey"
        )
        key_light.CreateIntensityAttr(2200.0)
        key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.82))
        UsdGeom.Xformable(key_light).AddRotateXYZOp().Set(
            Gf.Vec3f(-45.0, 30.0, 35.0)
        )

        component_paths = document["scene"]["component_paths"]
        for name in ("receptacle", "plug_body", "coupling_nut", "support_fixture"):
            prim = stage.GetPrimAtPath(str(component_paths[name]))
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"split component rigid body missing: {name}")
            if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True:
                raise RuntimeError(f"split component became kinematic: {name}")
        connector_root = str(document["scene"]["model_path_under_reference"])
        connector_joints = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(connector_root + "/")
            and "Joint" in prim.GetTypeName()
        ]
        if connector_joints:
            raise RuntimeError(f"connector joint appeared at runtime: {connector_joints}")

        fingertip_material_prim = stage.GetPrimAtPath(FINGERTIP_MATERIAL_PATH)
        if not fingertip_material_prim.IsValid():
            raise RuntimeError("nail-free fingertip physics material is missing")
        fingertip_material = UsdPhysics.MaterialAPI(fingertip_material_prim)
        fingertip_material.GetStaticFrictionAttr().Set(0.45)
        fingertip_material.GetDynamicFrictionAttr().Set(0.45)
        fingertip_physx = PhysxSchema.PhysxMaterialAPI.Apply(
            fingertip_material_prim
        )
        fingertip_physx.GetFrictionCombineModeAttr().Set(PhysxSchema.Tokens.min)
        nut_material_path = connector_root + "/Materials/coupling_nut_outer_grip"
        nut_material_prim = stage.GetPrimAtPath(nut_material_path)
        if not nut_material_prim.IsValid():
            raise RuntimeError("coupling-nut outer-grip material is missing")
        PhysxSchema.PhysxMaterialAPI.Apply(
            nut_material_prim
        ).GetFrictionCombineModeAttr().Set(PhysxSchema.Tokens.min)

        parts = {
            name: world.scene.add(
                SingleRigidPrim(
                    prim_path=str(component_paths[name]),
                    name=f"te_split_lock_{name}",
                )
            )
            for name in ("receptacle", "plug_body", "coupling_nut", "support_fixture")
        }
        ft_articulation = world.scene.add(
            SingleArticulation(
                prim_path=ARTICULATION_PATH,
                name="te_split_lock_hand2arm_reaction_reader",
            )
        )
        world.get_physics_context().set_gravity(float(document["scene"]["gravity_m_s2"]))
        world.reset()
        visuals = _ViewportEvidence(
            world=world,
            output=output,
            task_origin_world=contract["task_origin_world"],
        )
        # This required capture happens before any robot motion.  A viewport
        # infrastructure failure therefore cannot strand the hand in contact.
        visuals.capture("01_initial_scene", required=True)
        metadata = ft_articulation._articulation_view._metadata
        joint_indices = dict(metadata.joint_indices)
        if document["wrist_ft_safety"]["source_joint"] not in joint_indices:
            raise RuntimeError("hand2arm is absent from articulation reaction metadata")
        reaction_row = int(joint_indices["hand2arm"]) + 1
        all_reactions = _host_array(
            ft_articulation.get_measured_joint_forces()
        )
        if reaction_row >= len(all_reactions):
            raise RuntimeError("hand2arm reaction row is outside measured joint forces")

        robot_data = control.create_native_gravity_compensated_robot(
            ARTICULATION_PATH, EXPECTED_DOF_NAMES, dynamic
        )
        robot, active_indices, arm_indices, lower, upper, drive_audit = robot_data
        recorder = _TruthRecorder(
            parts=parts,
            ft_articulation=ft_articulation,
            reaction_row=reaction_row,
            robot_model=plan["inputs"].robot_model,
            task_origin_world=contract["task_origin_world"],
            task_rotation_world=contract["task_rotation_world"],
            physics_dt_s=float(dynamic["physics_dt_s"]),
            limits=document["wrist_ft_safety"],
        )
        stepper = control.JointSignalStepper(
            robot=robot,
            world=world,
            auditor=recorder,
            active_indices=active_indices,
            arm_indices=arm_indices,
            arm_lower_limits=lower,
            arm_upper_limits=upper,
            settings=dynamic,
            render=gui,
        )
        recorder.stepper = stepper
        motion_plan = dict(plan["motion_plan"])
        pregrasp = control.run_pregrasp_sequence(stepper, motion_plan, dynamic)

        dt = float(dynamic["physics_dt_s"])
        tare_steps = round(float(control_doc["free_space_ft_tare_duration_s"]) / dt)
        for _ in stepper.active_steps(tare_steps):
            stepper.advance(
                "ft_free_space_tare", pregrasp["arm"], pregrasp["hand"]
            )
        primary_failure = stepper.abort_reason
        if primary_failure is None:
            recorder.finalize_tare(
                int(control_doc["free_space_ft_tare_minimum_samples"])
            )

        contact = None
        finger_tare = np.zeros(4, dtype=np.float64)
        final_hand = np.asarray(
            motion_plan["final_hand_positions_rad"], dtype=np.float64
        )
        if primary_failure is None:
            cap_spec = document.get("contact_phase_active_hand_effort_cap")
            torque_spec = document.get("explicit_torque_contact")
            velocity_spec = document.get("velocity_drive_contact")
            if velocity_spec is not None:
                (
                    contact,
                    finger_tare,
                    velocity_drive_contact_audit,
                    primary_failure,
                ) = _run_velocity_drive_contact(
                    stepper=stepper,
                    robot=robot,
                    active_indices=active_indices,
                    motion_plan=motion_plan,
                    settings=dynamic,
                    pregrasp=pregrasp,
                    contract=velocity_spec,
                    recorder=recorder,
                )
            elif torque_spec is not None:
                (
                    contact,
                    finger_tare,
                    explicit_torque_contact_audit,
                    primary_failure,
                ) = _run_explicit_torque_contact(
                    stepper=stepper,
                    robot=robot,
                    active_indices=active_indices,
                    motion_plan=motion_plan,
                    settings=dynamic,
                    pregrasp=pregrasp,
                    contract=torque_spec,
                    recorder=recorder,
                )
            elif cap_spec is None:
                contact, finger_tare, final_hand, _ = control._tare_and_close(
                    stepper, motion_plan, dynamic, pregrasp, False
                )
                primary_failure = contact.failure_reason or stepper.abort_reason
            else:
                active_hand_indices = np.asarray(
                    active_indices[7:], dtype=np.int32
                )
                expected_original = float(
                    cap_spec["expected_original_and_restored_value_nm"]
                )
                original_readback = _read_active_hand_effort_caps(
                    robot, active_hand_indices
                )
                contact_effort_cap_audit = {
                    "enabled": True,
                    "scope": list(cap_spec["phase_scope"]),
                    "active_hand_dof_names": list(
                        control.ACTIVE_HAND_JOINT_NAMES
                    ),
                    "active_hand_dof_indices": active_hand_indices.tolist(),
                    "contact_effort_rise_nm": float(
                        dynamic["contact_effort_rise_nm"]
                    ),
                    "contact_consecutive_samples": int(
                        dynamic["contact_consecutive_samples"]
                    ),
                    "finger_maximum_speed_rad_s": float(
                        dynamic["finger_maximum_speed_rad_s"]
                    ),
                    "original_readback_nm": original_readback.tolist(),
                    "expected_original_nm": expected_original,
                    "original_readback_matches_expected": bool(
                        np.array_equal(
                            original_readback,
                            np.full(
                                len(control.ACTIVE_HAND_JOINT_NAMES),
                                expected_original,
                                dtype=np.float64,
                            ),
                        )
                    ),
                    "apply_before_step": int(stepper.step_index),
                    "apply": None,
                    "restore": None,
                    "restored_before_preload_or_release_recovery": False,
                }
                try:
                    if not contact_effort_cap_audit[
                        "original_readback_matches_expected"
                    ]:
                        primary_failure = (
                            "CONTACT_EFFORT_CAP_ORIGINAL_READBACK_MISMATCH"
                        )
                    else:
                        applied = _set_active_hand_effort_caps(
                            robot,
                            active_hand_indices,
                            float(cap_spec["value_nm"]),
                        )
                        contact_effort_cap_audit["apply"] = applied
                        if not applied[
                            "readback_matches_float32_command_exactly"
                        ]:
                            primary_failure = (
                                "CONTACT_EFFORT_CAP_APPLY_READBACK_MISMATCH"
                            )
                        else:
                            contact, finger_tare, final_hand, _ = (
                                control._tare_and_close(
                                    stepper,
                                    motion_plan,
                                    dynamic,
                                    pregrasp,
                                    False,
                                )
                            )
                            primary_failure = (
                                contact.failure_reason or stepper.abort_reason
                            )
                finally:
                    restored = _set_active_hand_effort_caps(
                        robot,
                        active_hand_indices,
                        expected_original,
                    )
                    contact_effort_cap_audit["restore"] = restored
                    contact_effort_cap_audit["restore_after_step"] = int(
                        stepper.step_index
                    )
                    contact_effort_cap_audit[
                        "restored_before_preload_or_release_recovery"
                    ] = True
                    if not restored[
                        "readback_matches_float32_command_exactly"
                    ]:
                        primary_failure = (
                            "CONTACT_EFFORT_CAP_RESTORE_READBACK_MISMATCH"
                        )
        if primary_failure is None and contact is not None and contact.complete:
            preload_settings = dict(dynamic)
            preload_settings["lift_duration_s"] = 0.0
            preload_settings["hold_duration_s"] = 0.0
            preload_motion = dict(motion_plan)
            preload_motion["lift_arm_waypoints_rad"] = [
                list(map(float, pregrasp["arm"]))
            ]
            primary_failure = control._run_preload_lift_hold(
                stepper,
                preload_motion,
                preload_settings,
                pregrasp,
                contact,
                finger_tare,
                final_hand,
            )
        elif primary_failure is None:
            primary_failure = "THREE_FINGER_CONTACT_NOT_COMPLETE"
        if recorder.first_ft_stop is not None and primary_failure is not None:
            visuals.capture("02_ft_safety_stop", required=False)
        if primary_failure is None:
            visuals.capture("02_three_finger_grip", required=False)

        held_hand = (
            np.asarray(pregrasp["hand"], dtype=np.float64)
            if recorder.last_active_target is None
            else recorder.last_active_target[7:].copy()
        )
        rotation_steps = round(float(control_doc["rotation_duration_s"]) / dt)
        executed_rotation_steps = 0
        if primary_failure is None:
            for index in stepper.active_steps(rotation_steps):
                blend = control.minimum_jerk_blend((index + 1) / rotation_steps)
                arm_target = control.piecewise_waypoint(
                    plan["rotation_arm_waypoints_rad"], blend
                )
                stepper.advance("lock_rotation", arm_target, held_hand)
                executed_rotation_steps += 1
            primary_failure = stepper.abort_reason
        rotation_completed = bool(
            primary_failure is None and executed_rotation_steps == rotation_steps
        )
        if rotation_completed:
            for _ in stepper.active_steps(
                round(float(control_doc["post_rotation_hold_duration_s"]) / dt)
            ):
                stepper.advance(
                    "lock_rotation_hold",
                    plan["rotation_arm_waypoints_rad"][-1],
                    held_hand,
                )
            primary_failure = stepper.abort_reason
            visuals.capture("03_rotation_end", required=False)

        recorder.begin_release_recovery()
        release_failure = None
        stepper.abort_reason = None
        if stepper.latest is None:
            current = np.zeros(11, dtype=np.float64)
        else:
            current = stepper.latest[0].copy()
        release_arm = current[:7].copy()
        release_hand_start = current[7:].copy()
        open_hand = np.asarray(
            control_doc["full_hand_open_target_rad"], dtype=np.float64
        )
        release_steps = round(float(control_doc["hand_release_duration_s"]) / dt)
        for index in stepper.active_steps(release_steps):
            blend = control.minimum_jerk_blend((index + 1) / release_steps)
            hand_target = (1.0 - blend) * release_hand_start + blend * open_hand
            stepper.advance("complete_hand_release", release_arm, hand_target)
        release_failure = stepper.abort_reason

        retreat_arm_target = plan["retreat_arm_target_rad"]
        if release_failure is None:
            retreat_steps = round(float(control_doc["retreat_duration_s"]) / dt)
            for index in stepper.active_steps(retreat_steps):
                blend = control.minimum_jerk_blend((index + 1) / retreat_steps)
                arm_target = (1.0 - blend) * release_arm + blend * retreat_arm_target
                stepper.advance("robot_retreat", arm_target, open_hand)
            release_failure = stepper.abort_reason
        if release_failure is None:
            for _ in stepper.active_steps(
                round(float(control_doc["release_hold_duration_s"]) / dt)
            ):
                stepper.advance("release_hold", retreat_arm_target, open_hand)
            release_failure = stepper.abort_reason
        visuals.capture("04_release_end", required=False)

        recorder.control_finished = True
        summary = _evaluate(
            recorder,
            document,
            primary_failure=primary_failure,
            release_failure=release_failure,
            rotation_completed=rotation_completed,
        )
        trace = {
            "schema_version": "kcg_te_split_lock_increment_trace_v1",
            "hardware_authorized": False,
            "simulation_only": True,
            "static_check": dict(static_result),
            "evidence_binding": {
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "split_asset": str(contract["paths"]["split_asset"]),
                "split_asset_sha256": _sha256(contract["paths"]["split_asset"]),
                "robot_asset": str(contract["paths"]["robot_asset"]),
                "robot_asset_sha256": _sha256(contract["paths"]["robot_asset"]),
                "development_grasp_trace": str(
                    contract["paths"]["development_grasp_trace"]
                ),
                "development_grasp_trace_sha256": _sha256(
                    contract["paths"]["development_grasp_trace"]
                ),
                "controller_source_sha256": _sha256(
                    contract["paths"]["controller_source"]
                ),
                "runner_source_sha256": _sha256(Path(__file__).resolve()),
            },
            "scene_placement_before_physics_start": contract[
                "world_from_model"
            ].ravel().tolist(),
            "robot_plan": {
                "rotation_arm_waypoints_rad": plan[
                    "rotation_arm_waypoints_rad"
                ].tolist(),
                "rotation_hand_targets_world_row_major": plan[
                    "rotation_hand_targets_world"
                ].reshape(-1, 16).tolist(),
                "rotation_ik": plan["rotation_ik"],
                "retreat_arm_target_rad": plan["retreat_arm_target_rad"].tolist(),
                "retreat_ik": plan["retreat_ik"],
            },
            "native_robot_drive_audit": drive_audit,
            "contact_phase_active_hand_effort_cap_audit": (
                contact_effort_cap_audit
            ),
            "explicit_torque_contact_audit": (
                explicit_torque_contact_audit
                if explicit_torque_contact_audit is not None
                else recorder.explicit_contact_runtime_audit
            ),
            "velocity_drive_contact_audit": (
                velocity_drive_contact_audit
                if velocity_drive_contact_audit is not None
                else recorder.velocity_contact_runtime_audit
            ),
            "hand2arm_reaction_row": reaction_row,
            "run_specific_task_wrench_tare": (
                None
                if recorder.tare_task_wrench is None
                else recorder.tare_task_wrench.tolist()
            ),
            "old_ft_monitor_calibration_used": False,
            "ft_reading_changed_control": recorder.first_ft_stop is not None,
            "object_pose_writes_after_start": 0,
            "direct_object_force_or_torque_commands": 0,
            "connector_joint_or_drive_commands": 0,
            "online_object_or_contact_truth_used": False,
            "truth_audit_data_returned_to_controller": False,
            "control_finished_before_truth_evaluation": True,
            "maximum_sensor_to_fk_zero_clamp_rad": (
                recorder.maximum_fk_zero_clamp_rad
            ),
            "sensor_to_fk_zero_clamp_threshold_rad": (
                FK_SENSOR_LIMIT_CLAMP_TOLERANCE_RAD
            ),
            "visual_evidence": {
                "records": visuals.records,
                "errors": visuals.errors,
                "image_feedback_used_by_control": False,
            },
            "samples": recorder.samples,
        }
        (output / "trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
        return 0 if summary["passed"] else 2
    except Exception:
        failure = {
            "status": "DYNAMIC_FAIL",
            "passed": False,
            "exception": traceback.format_exc(),
            "simulation_only": True,
            "hardware_authorized": False,
            "object_pose_writes_after_start": 0,
            "direct_object_force_or_torque_commands": 0,
            "connector_joint_or_drive_commands": 0,
            "contact_phase_active_hand_effort_cap_audit": (
                contact_effort_cap_audit
            ),
            "explicit_torque_contact_audit": (
                explicit_torque_contact_audit
                if explicit_torque_contact_audit is not None
                else (
                    None
                    if recorder is None
                    else recorder.explicit_contact_runtime_audit
                )
            ),
            "velocity_drive_contact_audit": (
                velocity_drive_contact_audit
                if velocity_drive_contact_audit is not None
                else (
                    None
                    if recorder is None
                    else recorder.velocity_contact_runtime_audit
                )
            ),
        }
        if recorder is not None:
            failure.update(
                {
                    "partial_sample_count": len(recorder.samples),
                    "last_recorded_phase": (
                        None
                        if not recorder.samples
                        else recorder.samples[-1]["phase"]
                    ),
                    "first_ft_safety_stop": recorder.first_ft_stop,
                    "control_finished_before_truth_evaluation": False,
                }
            )
            partial_trace = {
                "schema_version": "kcg_te_split_lock_increment_partial_trace_v1",
                "hardware_authorized": False,
                "simulation_only": True,
                "control_finished": False,
                "online_object_or_contact_truth_used": False,
                "truth_audit_data_returned_to_controller": False,
                "maximum_sensor_to_fk_zero_clamp_rad": (
                    recorder.maximum_fk_zero_clamp_rad
                ),
                "sensor_to_fk_zero_clamp_threshold_rad": (
                    FK_SENSOR_LIMIT_CLAMP_TOLERANCE_RAD
                ),
                "first_ft_safety_stop": recorder.first_ft_stop,
                "contact_phase_active_hand_effort_cap_audit": (
                    contact_effort_cap_audit
                ),
                "explicit_torque_contact_audit": (
                    explicit_torque_contact_audit
                    if explicit_torque_contact_audit is not None
                    else recorder.explicit_contact_runtime_audit
                ),
                "velocity_drive_contact_audit": (
                    velocity_drive_contact_audit
                    if velocity_drive_contact_audit is not None
                    else recorder.velocity_contact_runtime_audit
                ),
                "visual_evidence": (
                    None
                    if visuals is None
                    else {
                        "records": visuals.records,
                        "errors": visuals.errors,
                        "image_feedback_used_by_control": False,
                    }
                ),
                "samples": recorder.samples,
            }
            (output / "partial_trace.json").write_text(
                json.dumps(
                    partial_trace,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        (output / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, ensure_ascii=False, allow_nan=False, indent=2))
        return 2
    finally:
        simulation_app.close()


def main() -> int:
    arguments = _arguments()
    repository = Path(__file__).resolve().parents[3]
    config_path = arguments.config.expanduser().resolve()
    static_result, contract, plan = _static_check(repository, config_path)
    if arguments.mode == "static-check":
        print(json.dumps(static_result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0 if static_result["passed"] else 2
    if not static_result["passed"]:
        raise RuntimeError("static check must pass before run mode")
    output = arguments.output_directory.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return _execute_run(
        repository,
        config_path,
        static_result,
        contract,
        plan,
        output,
        gui=arguments.gui,
    )


if __name__ == "__main__":
    raise SystemExit(main())
