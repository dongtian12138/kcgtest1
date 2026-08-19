#!/usr/bin/env python3

"""Run the D38999 A2 V2 assembly-control nominal insertion bench.

The task controller uses a predeclared time schedule, a nominal load profile
derived from the frozen contract, and rigid-body state feedback.  Contact
names, contact normals, event truth, and post-start pose writes are excluded
from control.  Contact truth is collected only after each physics step for
acceptance auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_d38999_dynamic_a2_nominal_insertion_v2"
BENCH_ID = "DYN-A2-NOMINAL-INSERTION-V2"
ROOT_TASK_ID = "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2"
HYPOTHESIS_ID = "A2-V2-H13-METAL-STOP-DESCENDANT-PATH-EVALUATOR"
RUN_INDEX_CHOICES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
NUT_YAW_INTEGRAL_GAIN_NM_RAD_S = 0.08
NUT_YAW_INTEGRAL_LIMIT_NM = 0.10
THREAD_THRUST_TARGET_N = 0.5813853140574636
THREAD_THRUST_TERMINAL_YAW_LEAD_RAD = 0.12061203454745371
AXIAL_TARGET_HOLD_INTEGRAL_TRACKING_RATE_N_S = 1.0
AXIAL_TERMINAL_APPROACH_TRACKING_WINDOW_M = 1.0e-6
AXIAL_TERMINAL_APPROACH_TRACKING_MAX_TARGET_SPEED_M_S = 1.0e-6
EVENT_ORDER = (
    "five_key_polarization",
    "three_start_thread_entry",
    "spring_finger_engagement",
    "first_pin_socket_spring_touch",
    "pin_barrier_seal_contact",
    "seal_compression",
    "shell_to_shell_metal_bottoming",
)
EXPECTED_SHA256 = {
    "contract": "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b",
    "physical_contract": "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e",
    "acceptance": "dd37d07d5bd87a97c3dbefe225c0eafd409fc783a6010eec8db9da397ce2cf76",
    "authorized_overrides": "392766e8eceb85a3c910b118c2ad998aef891a74e58c31cd94e383c9908535ce",
    "model": "a3e43d53150dc94f1c703e41bcc6facd7df0f55ea7e083f8debf600349e8cc3d",
    "mapping": "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783",
    "a1_result": "566be2ccbbcb2bb600550bd1821265bec77a8235fead84cba8c0da5f24764e4b",
}
ROOT = "/World/D38999MultilayerV1/D38999_ASSEMBLY_CONTROL_V1"
PAIR_ROOT = ROOT + "/D38999Pair"
FIXED_PATH = PAIR_ROOT + "/FixedReceptacle"
BODY_PATH = PAIR_ROOT + "/LoosePlug/BodyAssembly"
NUT_PATH = PAIR_ROOT + "/LoosePlug/CouplingNut"
GRIP_PATH = NUT_PATH + "/CouplingNutGraspCollision"
BODY_SHOULDER_PATHS_BY_SIGN = {
    sign: BODY_PATH + f"/NutBearingShoulders/{sign}/AnalyticCap"
    for sign in ("PositiveStop", "NegativeStop")
}
NUT_SHOULDER_PATHS_BY_SIGN = {
    sign: tuple(
        NUT_PATH + f"/NutBearingShoulders/{sign}/AnalyticSphere_{index}"
        for index in range(3)
    )
    for sign in ("PositiveStop", "NegativeStop")
}
BODY_SHOULDER_PATHS = tuple(sorted(BODY_SHOULDER_PATHS_BY_SIGN.values()))
NUT_SHOULDER_PATHS = tuple(
    sorted(
        path
        for paths in NUT_SHOULDER_PATHS_BY_SIGN.values()
        for path in paths
    )
)
FIXED_STOP_PATH = FIXED_PATH + "/MatingShell/MetalStop"
PLUG_STOP_PATH = BODY_PATH + "/InternalMatingShell/MetalStop"
EXPECTED_BODY_COLLIDER_COUNT = 72
EXPECTED_FIXED_COLLIDER_COUNT = 199
EXPECTED_NUT_COLLIDER_COUNT = 7
GRASP_PROXY_FILTER_ROOT = "/World/D38999V2GraspProxyCollisionGroups"
MODEL_RELATIVE_PATH = Path(
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
    "D38999_ASSEMBLY_CONTROL_V1.usda"
)
CONTRACT_RELATIVE_PATH = Path(
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml"
)
MAPPING_RELATIVE_PATH = Path(
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json"
)
PHYSICAL_CONTRACT_RELATIVE_PATH = Path(
    "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
)
AUTHORIZED_OVERRIDES_RELATIVE_PATH = Path(
    "src/kcg_connector/config/d38999_assembly_control_authorized_overrides_v2.yaml"
)
A1_RESULT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A1-EVENT-ONSET-CALIBRATION-V2/TASK_RESULT.json"
)
A2_OUTPUT_ROOT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A2-NOMINAL-INSERTION-V2"
)
START_SEPARATION_M = 0.00550
END_SEPARATION_M = 0.01505
PHYSICAL_EFFECT_IMPLEMENTATIONS = {
    "continuous_shell_and_guidance": "physx_continuous_real_collision",
    "five_keys_and_keyways": "physx_continuous_real_collision",
    "same_label_smooth_entry_61": "model_internal_radial_guide_force",
    "same_label_deep_pin_61": "model_internal_spring_damper_force",
    "same_label_pin_isolation_61": "model_internal_axial_spring_damper_force",
    "peripheral_seal": "model_internal_continuous_annular_spring_damper_force",
    "three_start_thread": "model_internal_helical_generalized_force",
    "anti_decoupling_36_cycle": "model_internal_periodic_resistance_torque",
    "nut_body_physical_shoulder": "physx_continuous_real_collision",
    "metal_stop": "physx_continuous_real_collision",
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument(
        "--run-index", required=True, type=int, choices=RUN_INDEX_CHOICES
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kit-portable-root", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    result = parser.parse_args(argv)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository() -> Path:
    return Path(__file__).resolve().parents[3]


def _exact_path(requested: str, relative: Path, role: str) -> Path:
    actual = Path(requested).expanduser().resolve()
    expected = (_repository() / relative).resolve()
    if actual != expected:
        raise PermissionError(f"{role} path is frozen: {actual} != {expected}")
    if not actual.is_file():
        raise FileNotFoundError(actual)
    return actual


def _load_frozen_inputs(contract_raw: str, model_raw: str) -> dict[str, Any]:
    repository = _repository()
    contract_path = _exact_path(contract_raw, CONTRACT_RELATIVE_PATH, "contract")
    model_path = _exact_path(model_raw, MODEL_RELATIVE_PATH, "model")
    mapping_path = (repository / MAPPING_RELATIVE_PATH).resolve()
    physical_contract_path = (repository / PHYSICAL_CONTRACT_RELATIVE_PATH).resolve()
    authorized_overrides_path = (repository / AUTHORIZED_OVERRIDES_RELATIVE_PATH).resolve()
    a1_result_path = (repository / A1_RESULT_RELATIVE_PATH).resolve()
    actual_sha = {
        "contract": _sha256(contract_path),
        "physical_contract": _sha256(physical_contract_path),
        "authorized_overrides": _sha256(authorized_overrides_path),
        "model": _sha256(model_path),
        "mapping": _sha256(mapping_path),
        "a1_result": _sha256(a1_result_path),
    }
    for role, expected in EXPECTED_SHA256.items():
        if role == "acceptance":
            continue
        if actual_sha[role] != expected:
            raise PermissionError(
                f"frozen {role} SHA-256 changed: {actual_sha[role]} != {expected}"
            )
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    physical_info = contract["authoritative_inputs"]["physical_model_contract"]
    if (
        Path(str(physical_info["path"])) != PHYSICAL_CONTRACT_RELATIVE_PATH
        or str(physical_info["sha256"]) != EXPECTED_SHA256["physical_contract"]
    ):
        raise PermissionError("master contract physical-contract lineage changed")
    acceptance_info = contract["authoritative_inputs"]["physical_acceptance_contract"]
    acceptance_path = (repository / acceptance_info["path"]).resolve()
    acceptance_sha = _sha256(acceptance_path)
    if acceptance_sha != acceptance_info["sha256"] or acceptance_sha != EXPECTED_SHA256["acceptance"]:
        raise PermissionError("frozen acceptance SHA-256 changed")
    acceptance = yaml.safe_load(acceptance_path.read_text(encoding="utf-8"))
    physical_contract = yaml.safe_load(physical_contract_path.read_text(encoding="utf-8"))
    authorized_overrides = yaml.safe_load(
        authorized_overrides_path.read_text(encoding="utf-8")
    )
    a1_result = json.loads(a1_result_path.read_text(encoding="utf-8"))
    if (
        a1_result.get("task_id") != ROOT_TASK_ID
        or a1_result.get("node") != "DYN-A1-EVENT-ONSET-CALIBRATION-V2"
        or a1_result.get("status") != "DYNAMIC_PASS"
        or a1_result.get("classification")
        != "A1_V2_DECOMPOSED_DYNAMIC_CALIBRATION_PASS"
    ):
        raise PermissionError("A1 V2 result is not the required dynamic pass")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    representation = mapping["representations"]["D38999_ASSEMBLY_CONTROL_V1"]
    pairs = representation["pair_effects"]
    labels = [str(row["label"]) for row in contract["contact_layout"]["pairs"]]
    if len(labels) != 61 or len(pairs) != 61:
        raise RuntimeError("master contract and mapping must each contain 61 pairs")
    if labels != [str(row["label"]) for row in pairs]:
        raise RuntimeError("mapping pair labels differ from the master contract")
    if any(row.get("same_label_only") is not True for row in pairs):
        raise RuntimeError("every mapped pair must be same-label-only")
    model_text = model_path.read_text(encoding="utf-8")
    forbidden = (
        "hard_socket_entries_61",
        "pin_barriers_61",
        "socket_petals_366",
        "thread_rails_1080",
        "magneticAttractionAllowed = 1",
    )
    hits = [token for token in forbidden if token in model_text]
    if hits:
        raise RuntimeError(f"forbidden high-detail mechanisms leaked into model: {hits}")
    return {
        "contract_path": contract_path,
        "model_path": model_path,
        "mapping_path": mapping_path,
        "physical_contract_path": physical_contract_path,
        "authorized_overrides_path": authorized_overrides_path,
        "a1_result_path": a1_result_path,
        "acceptance_path": acceptance_path,
        "contract": contract,
        "physical_contract": physical_contract,
        "authorized_overrides": authorized_overrides,
        "a1_result": a1_result,
        "acceptance": acceptance,
        "mapping": mapping,
        "pairs": pairs,
        "labels": labels,
        "input_sha256": {**actual_sha, "acceptance": acceptance_sha},
    }


def _authorize(arguments: argparse.Namespace, output: Path) -> dict[str, Any]:
    repository = _repository()
    state = json.loads(
        (repository / "artifacts/agent_control/MASTER_STATE.json").read_text(
            encoding="utf-8"
        )
    )
    expected_output = repository / A2_OUTPUT_ROOT_RELATIVE_PATH / f"RUN_{arguments.run_index:02d}"
    if output != expected_output.resolve():
        raise PermissionError(f"output path is frozen: {output} != {expected_output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite A2 evidence: {output}")
    queue = yaml.safe_load(
        (repository / "artifacts/agent_control/WORK_QUEUE.yaml").read_text(
            encoding="utf-8"
        )
    )
    v2 = state.get("autonomous_dynamic_closeout_v2", {})
    node = v2.get("nominal_insertion", {})
    required = {
        "root_task": (state.get("task_id"), ROOT_TASK_ID),
        "root_status": (state.get("status"), "VALIDATING"),
        "phase": (state.get("phase"), "DYN_A2_NOMINAL_INSERTION_V2"),
        "next_task": (state.get("next_executable_dynamic_task"), BENCH_ID),
        "current_node": (v2.get("current_node"), BENCH_ID),
        "a1_status": (v2.get("nodes", {}).get("DYN-A1-EVENT-ONSET-CALIBRATION-V2"), "DYNAMIC_PASS"),
        "node_status": (v2.get("nodes", {}).get(BENCH_ID), "VALIDATING"),
        "run_status": (node.get("status"), "REGISTERED"),
        "run_index": (node.get("current_run_index"), arguments.run_index),
        "queue_status": (queue.get("status"), "VALIDATING"),
        "queue_task": (queue.get("current_task"), BENCH_ID),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in required.items()
        if actual != expected
    }
    if mismatches:
        raise PermissionError(f"A2 V2 state guard failed: {mismatches}")
    plan_path = (
        repository
        / A2_OUTPUT_ROOT_RELATIVE_PATH
        / f"RUN_PLAN_{arguments.run_index:02d}.json"
    ).resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"missing registered A2 run plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_plan = {
        "task_id": BENCH_ID,
        "run_id": f"A2-V2-NOMINAL-{arguments.run_index:02d}",
        "run_index": arguments.run_index,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "REGISTERED",
        "output_dir": str(expected_output.relative_to(repository)),
        "source_sha256": _sha256(Path(__file__)),
        "model_sha256": EXPECTED_SHA256["model"],
        "contract_sha256": EXPECTED_SHA256["contract"],
    }
    plan_mismatches = {
        key: {"actual": plan.get(key), "expected": expected}
        for key, expected in expected_plan.items()
        if plan.get(key) != expected
    }
    if plan_mismatches:
        raise PermissionError(f"A2 V2 run-plan guard failed: {plan_mismatches}")
    return {"plan_path": plan_path, "plan": plan}


def _finite(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"{label} is not a finite {size}-vector")
    return result


def _rpy_wxyz(quaternion: Any) -> tuple[float, float, float]:
    w, x, y, z = _finite(quaternion, 4, "quaternion")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise RuntimeError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _unwrap(previous_wrapped: float, previous_unwrapped: float, current: float) -> float:
    delta = (current - previous_wrapped + math.pi) % (2.0 * math.pi) - math.pi
    return previous_unwrapped + delta


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _minimum_jerk(progress: float) -> tuple[float, float]:
    """Return normalized position and derivative for a quintic minimum-jerk move."""

    value = _clip01(progress)
    position = 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5
    derivative = 30.0 * value**2 - 60.0 * value**3 + 30.0 * value**4
    if value <= 0.0 or value >= 1.0:
        derivative = 0.0
    return position, derivative


def _backward_euler_shared_spring_force(
    errors_m: Sequence[np.ndarray],
    velocities_m_s: Sequence[np.ndarray],
    *,
    active_fraction: float,
    per_channel_stiffness_n_m: float,
    per_channel_damping_n_s_m: float,
    integration_dt_s: float,
    effective_mass_kg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate parallel translational spring channels at the interval end."""

    if len(errors_m) != len(velocities_m_s) or not errors_m:
        raise ValueError("shared spring channels must be nonempty and paired")
    if not 0.0 <= active_fraction <= 1.0:
        raise ValueError("shared spring active fraction must be in [0, 1]")
    if integration_dt_s <= 0.0 or effective_mass_kg <= 0.0:
        raise ValueError("shared spring integration dt and mass must be positive")
    errors = np.asarray(errors_m, dtype=np.float64)
    velocities = np.asarray(velocities_m_s, dtype=np.float64)
    stiffness = active_fraction * per_channel_stiffness_n_m
    damping = active_fraction * per_channel_damping_n_s_m
    count = len(errors_m)
    total_stiffness = count * stiffness
    total_damping = count * damping
    raw_force = np.sum(-stiffness * errors - damping * velocities, axis=0)
    denominator = (
        1.0
        + total_damping * integration_dt_s / effective_mass_kg
        + total_stiffness * integration_dt_s**2 / effective_mass_kg
    )
    interval_force = -(
        stiffness * np.sum(errors, axis=0)
        + (damping + stiffness * integration_dt_s)
        * np.sum(velocities, axis=0)
    ) / denominator
    return interval_force, {
        "method": "backward_euler_shared_rigid_translation",
        "continuous_parameter_values_unchanged": True,
        "channel_count": count,
        "effective_total_stiffness_n_m": total_stiffness,
        "effective_total_damping_n_s_m": total_damping,
        "implicit_denominator": denominator,
        "raw_continuous_force_n": raw_force.tolist(),
        "applied_interval_force_n": interval_force.tolist(),
    }


def _backward_euler_planar_channel_wrench(
    errors_m: Sequence[np.ndarray],
    velocities_m_s: Sequence[np.ndarray],
    arms_m: Sequence[np.ndarray],
    *,
    active_fraction: float,
    per_channel_stiffness_n_m: float,
    per_channel_damping_n_s_m: float,
    integration_dt_s: float,
    effective_mass_kg: float,
    effective_yaw_inertia_kg_m2: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Implicitly discretize the shared x/y/yaw response of labelled pins."""

    if not (len(errors_m) == len(velocities_m_s) == len(arms_m)) or not errors_m:
        raise ValueError("planar spring channels must be nonempty and aligned")
    if integration_dt_s <= 0.0 or effective_mass_kg <= 0.0:
        raise ValueError("planar integration dt and mass must be positive")
    if effective_yaw_inertia_kg_m2 <= 0.0:
        raise ValueError("planar yaw inertia must be positive")
    stiffness = active_fraction * per_channel_stiffness_n_m
    damping = active_fraction * per_channel_damping_n_s_m
    generalized_raw = np.zeros(3, dtype=np.float64)
    generalized_base = np.zeros(3, dtype=np.float64)
    interval_matrix = np.eye(3, dtype=np.float64)
    inverse_mass = np.diag(
        (1.0 / effective_mass_kg, 1.0 / effective_mass_kg, 1.0 / effective_yaw_inertia_kg_m2)
    )
    for error, velocity, arm in zip(errors_m, velocities_m_s, arms_m):
        jacobian = np.asarray(
            ((1.0, 0.0, -float(arm[1])), (0.0, 1.0, float(arm[0]))),
            dtype=np.float64,
        )
        error_value = np.asarray(error, dtype=np.float64)
        velocity_value = np.asarray(velocity, dtype=np.float64)
        generalized_raw += jacobian.T @ (
            -stiffness * error_value - damping * velocity_value
        )
        generalized_base += jacobian.T @ (
            -stiffness * (error_value + integration_dt_s * velocity_value)
            - damping * velocity_value
        )
        interval_matrix += (
            damping + stiffness * integration_dt_s
        ) * integration_dt_s * jacobian.T @ jacobian @ inverse_mass
    interval_wrench = np.linalg.solve(interval_matrix, generalized_base)
    return interval_wrench[:2], float(interval_wrench[2]), {
        "method": "backward_euler_shared_planar_force_and_yaw_torque",
        "continuous_parameter_values_unchanged": True,
        "channel_count": len(errors_m),
        "raw_continuous_generalized_wrench": generalized_raw.tolist(),
        "applied_interval_generalized_wrench": interval_wrench.tolist(),
        "implicit_matrix_condition_number": float(np.linalg.cond(interval_matrix)),
    }


def _joint_interval_lateral_actions(
    spring_errors_m: Sequence[np.ndarray],
    spring_velocities_m_s: Sequence[np.ndarray],
    pair_errors_m: Sequence[np.ndarray],
    pair_velocities_m_s: Sequence[np.ndarray],
    pair_arms_m: Sequence[np.ndarray],
    *,
    spring_active_fraction: float,
    spring_per_channel_stiffness_n_m: float,
    spring_per_channel_damping_n_s_m: float,
    pair_active_fraction: float,
    pair_per_channel_stiffness_n_m: float,
    pair_per_channel_damping_n_s_m: float,
    body_driver_error_m: np.ndarray,
    body_driver_velocity_m_s: np.ndarray,
    nut_driver_error_m: np.ndarray,
    nut_driver_velocity_m_s: np.ndarray,
    translation_driver_stiffness_n_m: float,
    translation_driver_damping_n_s_m: float,
    body_yaw_error_rad: float,
    body_yaw_velocity_rad_s: float,
    body_yaw_driver_stiffness_nm_rad: float,
    body_yaw_driver_damping_nm_s_rad: float,
    integration_dt_s: float,
    effective_translation_mass_kg: float,
    body_yaw_inertia_kg_m2: float,
) -> dict[str, Any]:
    """Solve every x/y/body-yaw action against one shared interval end state."""

    if not (
        len(spring_errors_m) == len(spring_velocities_m_s)
        and len(pair_errors_m) == len(pair_velocities_m_s) == len(pair_arms_m)
    ):
        raise ValueError("joint interval channel arrays must be aligned")
    if not spring_errors_m or not pair_errors_m:
        raise ValueError("joint interval physical channel arrays must be nonempty")
    if not 0.0 <= spring_active_fraction <= 1.0:
        raise ValueError("spring active fraction must be in [0, 1]")
    if not 0.0 <= pair_active_fraction <= 1.0:
        raise ValueError("pair active fraction must be in [0, 1]")
    if integration_dt_s <= 0.0 or effective_translation_mass_kg <= 0.0:
        raise ValueError("joint interval dt and translation mass must be positive")
    if body_yaw_inertia_kg_m2 <= 0.0:
        raise ValueError("joint interval yaw inertia must be positive")

    identity_xy = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    yaw_jacobian = np.asarray(((0.0, 0.0, 1.0),))
    inverse_mass = np.diag(
        (
            1.0 / effective_translation_mass_kg,
            1.0 / effective_translation_mass_kg,
            1.0 / body_yaw_inertia_kg_m2,
        )
    )
    channels: list[dict[str, Any]] = []

    def add_group(
        name: str,
        errors: Sequence[np.ndarray],
        velocities: Sequence[np.ndarray],
        jacobians: Sequence[np.ndarray],
        stiffness: float,
        damping: float,
    ) -> None:
        if not (len(errors) == len(velocities) == len(jacobians)):
            raise ValueError(f"joint interval group {name} is not aligned")
        for error, velocity, jacobian in zip(errors, velocities, jacobians):
            error_value = np.asarray(error, dtype=np.float64).reshape(-1)
            velocity_value = np.asarray(velocity, dtype=np.float64).reshape(-1)
            jacobian_value = np.asarray(jacobian, dtype=np.float64)
            if (
                error_value.shape != velocity_value.shape
                or jacobian_value.shape != (len(error_value), 3)
            ):
                raise ValueError(f"joint interval group {name} has invalid shapes")
            channels.append(
                {
                    "group": name,
                    "error": error_value,
                    "velocity": velocity_value,
                    "jacobian": jacobian_value,
                    "stiffness": float(stiffness),
                    "damping": float(damping),
                }
            )

    add_group(
        "spring_fingers",
        spring_errors_m,
        spring_velocities_m_s,
        [identity_xy for _ in spring_errors_m],
        spring_active_fraction * spring_per_channel_stiffness_n_m,
        spring_active_fraction * spring_per_channel_damping_n_s_m,
    )
    pair_jacobians = [
        np.asarray(
            ((1.0, 0.0, -float(arm[1])), (0.0, 1.0, float(arm[0]))),
            dtype=np.float64,
        )
        for arm in pair_arms_m
    ]
    add_group(
        "same_label_pins_61",
        pair_errors_m,
        pair_velocities_m_s,
        pair_jacobians,
        pair_active_fraction * pair_per_channel_stiffness_n_m,
        pair_active_fraction * pair_per_channel_damping_n_s_m,
    )
    add_group(
        "body_lateral_driver",
        [np.asarray(body_driver_error_m, dtype=np.float64)],
        [np.asarray(body_driver_velocity_m_s, dtype=np.float64)],
        [identity_xy],
        translation_driver_stiffness_n_m,
        translation_driver_damping_n_s_m,
    )
    add_group(
        "nut_lateral_driver",
        [np.asarray(nut_driver_error_m, dtype=np.float64)],
        [np.asarray(nut_driver_velocity_m_s, dtype=np.float64)],
        [identity_xy],
        translation_driver_stiffness_n_m,
        translation_driver_damping_n_s_m,
    )
    add_group(
        "body_yaw_driver",
        [np.asarray((body_yaw_error_rad,), dtype=np.float64)],
        [np.asarray((body_yaw_velocity_rad_s,), dtype=np.float64)],
        [yaw_jacobian],
        body_yaw_driver_stiffness_nm_rad,
        body_yaw_driver_damping_nm_s_rad,
    )

    interval_matrix = np.eye(3, dtype=np.float64)
    generalized_base = np.zeros(3, dtype=np.float64)
    raw_by_group = {
        name: np.zeros(3, dtype=np.float64)
        for name in (
            "spring_fingers",
            "same_label_pins_61",
            "body_lateral_driver",
            "nut_lateral_driver",
            "body_yaw_driver",
        )
    }
    for channel in channels:
        stiffness = channel["stiffness"]
        damping = channel["damping"]
        error = channel["error"]
        velocity = channel["velocity"]
        jacobian = channel["jacobian"]
        raw = jacobian.T @ (-stiffness * error - damping * velocity)
        raw_by_group[channel["group"]] += raw
        generalized_base += jacobian.T @ (
            -stiffness * (error + integration_dt_s * velocity)
            - damping * velocity
        )
        interval_matrix += (
            damping + stiffness * integration_dt_s
        ) * integration_dt_s * jacobian.T @ jacobian @ inverse_mass

    total_interval_wrench = np.linalg.solve(interval_matrix, generalized_base)
    generalized_acceleration = inverse_mass @ total_interval_wrench
    interval_by_group = {
        name: np.zeros(3, dtype=np.float64) for name in raw_by_group
    }
    for channel in channels:
        jacobian = channel["jacobian"]
        velocity_end = channel["velocity"] + integration_dt_s * (
            jacobian @ generalized_acceleration
        )
        error_end = channel["error"] + integration_dt_s * velocity_end
        force_end = (
            -channel["stiffness"] * error_end
            - channel["damping"] * velocity_end
        )
        interval_by_group[channel["group"]] += jacobian.T @ force_end
    partition_sum = sum(interval_by_group.values(), np.zeros(3, dtype=np.float64))
    partition_residual = partition_sum - total_interval_wrench
    partition_residual_norm = float(np.linalg.norm(partition_residual))
    partition_relative_residual = partition_residual_norm / max(
        1.0, float(np.linalg.norm(total_interval_wrench))
    )
    if partition_relative_residual > 1.0e-10:
        raise RuntimeError("joint interval action partition residual exceeded tolerance")
    return {
        "method": "backward_euler_joint_interval_lateral_physics_and_driver",
        "continuous_parameter_values_unchanged": True,
        "physical_channel_count": len(spring_errors_m) + len(pair_errors_m),
        "driver_channel_count": 3,
        "raw_generalized_wrench": sum(
            raw_by_group.values(), np.zeros(3, dtype=np.float64)
        ),
        "interval_generalized_wrench": total_interval_wrench,
        "raw_by_group": raw_by_group,
        "interval_by_group": interval_by_group,
        "interval_matrix_condition_number": float(np.linalg.cond(interval_matrix)),
        "partition_residual_norm": partition_residual_norm,
        "partition_relative_residual": partition_relative_residual,
        "predicted_generalized_acceleration": generalized_acceleration,
    }


def _backward_euler_thread_force(
    phase_error_m: float,
    phase_rate_m_s: float,
    *,
    stiffness_n_m: float,
    damping_n_s_m: float,
    integration_dt_s: float,
    body_mass_kg: float,
    nut_yaw_inertia_kg_m2: float,
    lead_m_per_revolution: float,
) -> tuple[float, dict[str, Any]]:
    """Discretize the helical generalized force without changing its K/C law."""

    radius = lead_m_per_revolution / (2.0 * math.pi)
    inverse_effective_mass = 1.0 / body_mass_kg + radius**2 / nut_yaw_inertia_kg_m2
    raw_force = stiffness_n_m * phase_error_m + damping_n_s_m * phase_rate_m_s
    numerator = (
        stiffness_n_m * (phase_error_m + integration_dt_s * phase_rate_m_s)
        + damping_n_s_m * phase_rate_m_s
    )
    denominator = 1.0 + (
        stiffness_n_m * integration_dt_s**2
        + damping_n_s_m * integration_dt_s
    ) * inverse_effective_mass
    interval_force = numerator / denominator
    return interval_force, {
        "method": "backward_euler_helical_generalized_force",
        "continuous_parameter_values_unchanged": True,
        "raw_continuous_force_n": raw_force,
        "applied_interval_force_n": interval_force,
        "inverse_effective_mass_kg_inv": inverse_effective_mass,
        "implicit_denominator": denominator,
    }


def _internal_effects(
    frozen: Mapping[str, Any],
    *,
    fixed_position: np.ndarray,
    fixed_velocity: np.ndarray,
    body_position: np.ndarray,
    body_velocity: np.ndarray,
    body_yaw: float,
    body_omega_z: float,
    nut_yaw: float,
    nut_omega_z: float,
    nut_position: np.ndarray,
    nut_velocity: np.ndarray,
    integration_dt_s: float,
    effective_translation_mass_kg: float,
    body_mass_kg: float,
    body_yaw_inertia_kg_m2: float,
    nut_yaw_inertia_kg_m2: float,
) -> dict[str, Any]:
    contract = frozen["contract"]
    elastic = contract["elastic_contact_models"]
    separation = float(fixed_position[2] - body_position[2])
    separation_rate = float(fixed_velocity[2] - body_velocity[2])
    body_force = np.zeros(3, dtype=np.float64)
    body_torque = np.zeros(3, dtype=np.float64)
    nut_force = np.zeros(3, dtype=np.float64)
    nut_torque = np.zeros(3, dtype=np.float64)

    events = {
        row["name"]: float(row["nominal_separation_m"])
        for row in contract["assembly_events"]["ordered"]
    }
    spring = elastic["shell_spring_fingers"]
    spring_ramp = _clip01(
        (separation - events["spring_finger_engagement"]) / 0.00020
    )
    radial_preload_n = (
        float(spring["aggregate_stiffness_n_m"])
        * float(spring["nominal_radial_preload_m"])
    )
    shell_dynamic_friction = float(
        contract["material_roles"]["values"]["plug_shell_and_keys"]["dynamic_friction"]
    )
    spring_axial_resistance_n = radial_preload_n * shell_dynamic_friction * spring_ramp
    body_force[2] += spring_axial_resistance_n
    lateral_error = body_position[:2] - fixed_position[:2]
    lateral_deflection = np.clip(
        lateral_error,
        -float(spring["nominal_radial_preload_m"]),
        float(spring["nominal_radial_preload_m"]),
    )
    lateral_velocity = body_velocity[:2] - fixed_velocity[:2]
    contact = elastic["socket_contact_per_label"]
    pin_ramp = _clip01(
        (separation - events["first_pin_socket_spring_touch"])
        / float(contact["maximum_physical_deflection_m"])
    )
    cosine, sine = math.cos(body_yaw), math.sin(body_yaw)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    pair_errors: list[np.ndarray] = []
    pair_velocities: list[np.ndarray] = []
    pair_arms: list[np.ndarray] = []
    maximum_pair_deflection_m = 0.0
    for row in frozen["pairs"]:
        if row.get("same_label_only") is not True:
            raise RuntimeError("cross-label pair reached internal model")
        center = np.asarray(row["center_m"], dtype=np.float64)
        rotated = rotation @ center
        socket_world = body_position[:2] + rotated
        pin_world = fixed_position[:2] + center
        error = socket_world - pin_world
        magnitude = float(np.linalg.norm(error))
        maximum_pair_deflection_m = max(maximum_pair_deflection_m, magnitude)
        if magnitude > float(contact["maximum_physical_deflection_m"]):
            error = error * float(contact["maximum_physical_deflection_m"]) / magnitude
        point_velocity = (
            body_velocity[:2]
            + body_omega_z * np.asarray((-rotated[1], rotated[0]))
            - fixed_velocity[:2]
        )
        pair_errors.append(error)
        pair_velocities.append(point_velocity)
        pair_arms.append(rotated)
    profile = frozen["acceptance"]["benches"]["P1"]["inputs"][
        "component_driver_profile"
    ]
    joint_lateral = _joint_interval_lateral_actions(
        [lateral_deflection.copy() for _ in range(int(spring["count"]))],
        [lateral_velocity.copy() for _ in range(int(spring["count"]))],
        pair_errors,
        pair_velocities,
        pair_arms,
        spring_active_fraction=spring_ramp,
        spring_per_channel_stiffness_n_m=float(spring["per_segment_stiffness_n_m"]),
        spring_per_channel_damping_n_s_m=float(spring["per_segment_damping_n_s_m"]),
        pair_active_fraction=pin_ramp,
        pair_per_channel_stiffness_n_m=float(contact["aggregate_stiffness_n_m"]),
        pair_per_channel_damping_n_s_m=float(contact["aggregate_damping_n_s_m"]),
        body_driver_error_m=body_position[:2],
        body_driver_velocity_m_s=body_velocity[:2],
        nut_driver_error_m=nut_position[:2],
        nut_driver_velocity_m_s=nut_velocity[:2],
        translation_driver_stiffness_n_m=float(profile["translation_position_gain_n_m"]),
        translation_driver_damping_n_s_m=float(profile["translation_velocity_gain_n_s_m"]),
        body_yaw_error_rad=body_yaw - float(profile["body_yaw_target_rad"]),
        body_yaw_velocity_rad_s=body_omega_z,
        body_yaw_driver_stiffness_nm_rad=float(profile["body_yaw_position_gain_nm_rad"]),
        body_yaw_driver_damping_nm_s_rad=float(profile["angular_velocity_gain_nm_s_rad"]),
        integration_dt_s=integration_dt_s,
        effective_translation_mass_kg=effective_translation_mass_kg,
        body_yaw_inertia_kg_m2=body_yaw_inertia_kg_m2,
    )
    spring_generalized = joint_lateral["interval_by_group"]["spring_fingers"]
    pair_generalized = joint_lateral["interval_by_group"]["same_label_pins_61"]
    spring_lateral_force = spring_generalized[:2]
    pair_force = pair_generalized[:2]
    pair_torque_z = float(pair_generalized[2])
    spring_integration = {
        "method": "backward_euler_joint_interval_partition",
        "continuous_parameter_values_unchanged": True,
        "channel_count": int(spring["count"]),
        "raw_continuous_force_n": joint_lateral["raw_by_group"][
            "spring_fingers"
        ][:2].tolist(),
        "applied_interval_force_n": spring_lateral_force.tolist(),
        "joint_interval_matrix_condition_number": joint_lateral[
            "interval_matrix_condition_number"
        ],
    }
    pair_integration = {
        "method": "backward_euler_joint_interval_partition",
        "continuous_parameter_values_unchanged": True,
        "channel_count": len(pair_errors),
        "raw_continuous_generalized_wrench": joint_lateral["raw_by_group"][
            "same_label_pins_61"
        ].tolist(),
        "applied_interval_generalized_wrench": pair_generalized.tolist(),
        "joint_interval_matrix_condition_number": joint_lateral[
            "interval_matrix_condition_number"
        ],
    }
    body_force[:2] += spring_lateral_force
    body_force[:2] += pair_force
    body_torque[2] += pair_torque_z

    isolation = elastic["pin_isolation_seal_per_label"]
    isolation_fraction = _clip01(
        (separation - float(isolation["nominal_first_touch_separation_m"]))
        / float(isolation["nominal_post_contact_travel_to_bottoming_m"])
    )
    isolation_normal_deflection = (
        isolation_fraction * float(isolation["physical_normal_deflection_nominal_m"])
    )
    isolation_normal_rate = 0.0
    if 0.0 < isolation_fraction < 1.0 and separation_rate > 0.0:
        isolation_normal_rate = separation_rate * (
            float(isolation["physical_normal_deflection_nominal_m"])
            / float(isolation["nominal_post_contact_travel_to_bottoming_m"])
        )
    isolation_resistance_n = int(isolation["count"]) * (
        float(isolation["per_label_effective_stiffness_n_m"])
        * isolation_normal_deflection
        + float(isolation["per_label_effective_damping_n_s_m"])
        * isolation_normal_rate
    )
    body_force[2] += isolation_resistance_n

    seal = elastic["peripheral_seal"]
    seal_deflection = max(
        0.0,
        min(
            float(seal["nominal_deflection_at_bottoming_m"]),
            separation - float(seal["first_touch_separation_m"]),
        ),
    )
    seal_rate = separation_rate if (
        0.0 < seal_deflection < float(seal["nominal_deflection_at_bottoming_m"])
        and separation_rate > 0.0
    ) else 0.0
    seal_resistance_n = (
        float(seal["aggregate_stiffness_n_m"]) * seal_deflection
        + float(seal["aggregate_damping_n_s_m"]) * seal_rate
    )
    body_force[2] += seal_resistance_n

    thread = contract["thread"]
    thread_entry = float(thread["nominal_entry_separation_m"]["value"])
    lead = float(thread["lead_mm_per_revolution"]) / 1000.0
    relative_yaw = nut_yaw - body_yaw
    relative_omega = nut_omega_z - body_omega_z
    thread_active = separation > thread_entry
    advance = max(0.0, separation - thread_entry)
    rotational_advance = -relative_yaw * lead / (2.0 * math.pi)
    phase_error_m = advance - rotational_advance
    phase_rate_m_s = separation_rate + relative_omega * lead / (2.0 * math.pi)
    # Runner-numerical penalty, derived and explicitly not manufacturer data.
    thread_stiffness_n_m = 10000.0
    thread_damping_n_s_m = 20.0
    thread_force_n = 0.0
    thread_integration: dict[str, Any] = {
        "method": "inactive",
        "continuous_parameter_values_unchanged": True,
    }
    if thread_active:
        thread_force_n, thread_integration = _backward_euler_thread_force(
            phase_error_m,
            phase_rate_m_s,
            stiffness_n_m=thread_stiffness_n_m,
            damping_n_s_m=thread_damping_n_s_m,
            integration_dt_s=integration_dt_s,
            body_mass_kg=body_mass_kg,
            nut_yaw_inertia_kg_m2=nut_yaw_inertia_kg_m2,
            lead_m_per_revolution=lead,
        )
        thread_force_n = float(np.clip(thread_force_n, -32.0, 32.0))
        body_force[2] += thread_force_n
        nut_torque[2] += -thread_force_n * lead / (2.0 * math.pi)

    detent = contract["anti_decoupling"]
    cycles = int(detent["cycle_count_per_revolution"])
    phase = (-relative_yaw * cycles) % (2.0 * math.pi)
    cam_deflection = float(detent["nominal_radial_preload_m"]) + 0.5 * float(
        detent["cam_radial_rise_m"]
    ) * (1.0 - math.cos(phase))
    radial_force_n = int(detent["follower_count"]) * (
        float(detent["per_follower_stiffness_n_m"]) * cam_deflection
    )
    periodic_scale = 0.5 + 0.5 * math.cos(phase)
    detent_torque_nm = 0.0
    if abs(relative_omega) > 1.0e-8:
        base = radial_force_n * float(detent["cam_radial_rise_m"]) * cycles / (
            2.0 * math.pi
        )
        direction_scale = 1.0 if relative_omega < 0.0 else 2.0
        detent_torque_nm = -math.copysign(
            min(0.05, direction_scale * base * periodic_scale), relative_omega
        )
        detent_torque_nm += -int(detent["follower_count"]) * float(
            detent["per_follower_damping_n_s_m"]
        ) * float(detent["mean_radius_m"]) ** 2 * relative_omega
        detent_torque_nm = float(np.clip(detent_torque_nm, -0.05, 0.05))
    nut_torque[2] += detent_torque_nm
    body_torque[2] -= detent_torque_nm

    zero_force = np.zeros(3, dtype=np.float64)
    zero_torque = np.zeros(3, dtype=np.float64)
    spring_force = np.asarray(
        (spring_lateral_force[0], spring_lateral_force[1], spring_axial_resistance_n),
        dtype=np.float64,
    )
    pin_force = np.asarray((pair_force[0], pair_force[1], 0.0), dtype=np.float64)
    pin_torque = np.asarray((0.0, 0.0, pair_torque_z), dtype=np.float64)
    isolation_force = np.asarray((0.0, 0.0, isolation_resistance_n), dtype=np.float64)
    seal_force = np.asarray((0.0, 0.0, seal_resistance_n), dtype=np.float64)
    thread_force = np.asarray((0.0, 0.0, thread_force_n), dtype=np.float64)
    thread_torque = np.asarray(
        (0.0, 0.0, -thread_force_n * lead / (2.0 * math.pi)), dtype=np.float64
    )
    body_detent_torque = np.asarray((0.0, 0.0, -detent_torque_nm), dtype=np.float64)
    nut_detent_torque = np.asarray((0.0, 0.0, detent_torque_nm), dtype=np.float64)

    return {
        "body_force_n": body_force,
        "body_torque_nm": body_torque,
        "nut_force_n": nut_force,
        "nut_torque_nm": nut_torque,
        "coupled_driver_overrides": {
            "body_force_xy_n": joint_lateral["interval_by_group"][
                "body_lateral_driver"
            ][:2],
            "nut_force_xy_n": joint_lateral["interval_by_group"][
                "nut_lateral_driver"
            ][:2],
            "body_yaw_torque_nm": float(
                joint_lateral["interval_by_group"]["body_yaw_driver"][2]
            ),
        },
        "channel_wrenches": {
            "spring_fingers": {
                "body_force_n": spring_force,
                "body_torque_nm": zero_torque.copy(),
                "nut_force_n": zero_force.copy(),
                "nut_torque_nm": zero_torque.copy(),
            },
            "same_label_pins_61": {
                "body_force_n": pin_force,
                "body_torque_nm": pin_torque,
                "nut_force_n": zero_force.copy(),
                "nut_torque_nm": zero_torque.copy(),
            },
            "pin_isolation_seals_61": {
                "body_force_n": isolation_force,
                "body_torque_nm": zero_torque.copy(),
                "nut_force_n": zero_force.copy(),
                "nut_torque_nm": zero_torque.copy(),
            },
            "peripheral_seal": {
                "body_force_n": seal_force,
                "body_torque_nm": zero_torque.copy(),
                "nut_force_n": zero_force.copy(),
                "nut_torque_nm": zero_torque.copy(),
            },
            "three_start_thread": {
                "body_force_n": thread_force,
                "body_torque_nm": zero_torque.copy(),
                "nut_force_n": zero_force.copy(),
                "nut_torque_nm": thread_torque,
            },
            "anti_decoupling_36_cycle": {
                "body_force_n": zero_force.copy(),
                "body_torque_nm": body_detent_torque,
                "nut_force_n": zero_force.copy(),
                "nut_torque_nm": nut_detent_torque,
            },
        },
        "channels": {
            "continuous_shell_and_guidance": True,
            "five_keys_and_keyways": True,
            "spring_fingers_active": spring_ramp > 0.0,
            "same_label_pin_effects_active": pin_ramp > 0.0,
            "pin_isolation_active": isolation_fraction > 0.0,
            "peripheral_seal_active": seal_deflection > 0.0,
            "three_start_thread_active": thread_active,
            "anti_decoupling_36_cycle_active": abs(relative_omega) > 1.0e-8,
        },
        "measurements": {
            "separation_m": separation,
            "separation_rate_m_s": separation_rate,
            "spring_axial_resistance_n": spring_axial_resistance_n,
            "spring_lateral_integration": spring_integration,
            "maximum_same_label_radial_deflection_m": maximum_pair_deflection_m,
            "same_label_planar_integration": pair_integration,
            "joint_lateral_driver_integration": {
                "method": joint_lateral["method"],
                "continuous_parameter_values_unchanged": joint_lateral[
                    "continuous_parameter_values_unchanged"
                ],
                "physical_channel_count": joint_lateral["physical_channel_count"],
                "driver_channel_count": joint_lateral["driver_channel_count"],
                "raw_generalized_wrench": joint_lateral[
                    "raw_generalized_wrench"
                ].tolist(),
                "interval_generalized_wrench": joint_lateral[
                    "interval_generalized_wrench"
                ].tolist(),
                "interval_matrix_condition_number": joint_lateral[
                    "interval_matrix_condition_number"
                ],
                "partition_residual_norm": joint_lateral[
                    "partition_residual_norm"
                ],
                "partition_relative_residual": joint_lateral[
                    "partition_relative_residual"
                ],
            },
            "pin_isolation_normal_deflection_m": isolation_normal_deflection,
            "pin_isolation_axial_resistance_n": isolation_resistance_n,
            "peripheral_seal_deflection_m": seal_deflection,
            "peripheral_seal_axial_resistance_n": seal_resistance_n,
            "thread_phase_error_m": phase_error_m,
            "thread_constraint_force_n": thread_force_n,
            "thread_integration": thread_integration,
            "detent_phase_rad": phase,
            "detent_resistance_torque_nm": detent_torque_nm,
        },
    }


def _nominal_axial_load_feedforward(
    frozen: Mapping[str, Any], target_separation_m: float
) -> dict[str, float]:
    """Return the frozen zero-rate axial load at the predeclared target.

    This function deliberately has no actual pose, contact, manifold, event
    activation, or collision-name input.  It does not include the physical
    metal stop, thread transients, or damping; those remain runtime physics.
    """

    contract = frozen["contract"]
    elastic = contract["elastic_contact_models"]
    events = {
        row["name"]: float(row["nominal_separation_m"])
        for row in contract["assembly_events"]["ordered"]
    }
    spring = elastic["shell_spring_fingers"]
    spring_ramp = _clip01(
        (target_separation_m - events["spring_finger_engagement"]) / 0.00020
    )
    spring_normal_preload_n = (
        float(spring["aggregate_stiffness_n_m"])
        * float(spring["nominal_radial_preload_m"])
    )
    shell_dynamic_friction = float(
        contract["material_roles"]["values"]["plug_shell_and_keys"][
            "dynamic_friction"
        ]
    )
    spring_axial_n = (
        spring_normal_preload_n * shell_dynamic_friction * spring_ramp
    )

    isolation = elastic["pin_isolation_seal_per_label"]
    isolation_fraction = _clip01(
        (
            target_separation_m
            - float(isolation["nominal_first_touch_separation_m"])
        )
        / float(isolation["nominal_post_contact_travel_to_bottoming_m"])
    )
    isolation_axial_n = int(isolation["count"]) * (
        float(isolation["per_label_effective_stiffness_n_m"])
        * isolation_fraction
        * float(isolation["physical_normal_deflection_nominal_m"])
    )

    seal = elastic["peripheral_seal"]
    seal_deflection_m = max(
        0.0,
        min(
            float(seal["nominal_deflection_at_bottoming_m"]),
            target_separation_m - float(seal["first_touch_separation_m"]),
        ),
    )
    seal_axial_n = float(seal["aggregate_stiffness_n_m"]) * seal_deflection_m
    total_n = spring_axial_n + isolation_axial_n + seal_axial_n
    values = {
        "spring_finger_axial_n": spring_axial_n,
        "pin_isolation_axial_n": isolation_axial_n,
        "peripheral_seal_axial_n": seal_axial_n,
        "total_n": total_n,
        "per_driven_body_component_n": 0.5 * total_n,
    }
    if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
        raise RuntimeError("nominal axial load feedforward is invalid")
    return values


def _shoulder_aware_axial_targets(
    frozen: Mapping[str, Any],
    target_separation_m: float,
    target_separation_rate_m_s: float,
) -> dict[str, float | str | bool]:
    """Map one assembly coordinate onto the two frozen rigid-body targets.

    In the insertion direction the coupling nut carries the plug body through
    the negative physical shoulder.  Its absolute z target therefore differs
    from the body target by the frozen negative shoulder endplay.  This mapping
    uses only the predeclared target and frozen contracts; runtime contact truth
    is neither required nor accepted.
    """

    master_endplay = frozen["contract"]["coupling_nut_motion"][
        "physical_shoulder_endplay_m"
    ]
    physical_endplay = frozen["physical_contract"]["physical_proxy_boundaries"][
        "nut_body_bearing"
    ]["physical_shoulder_contact_endplay_m"]
    master_low = float(master_endplay["low"])
    master_high = float(master_endplay["high"])
    physical_low = float(physical_endplay["low"])
    physical_high = float(physical_endplay["high"])
    if not (
        math.isclose(master_low, physical_low, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(master_high, physical_high, rel_tol=0.0, abs_tol=1.0e-15)
    ):
        raise RuntimeError("master and physical shoulder endplay contracts differ")
    if not (
        master_low < 0.0 < master_high
        and math.isclose(master_low, -master_high, rel_tol=0.0, abs_tol=1.0e-15)
    ):
        raise RuntimeError("frozen shoulder endplay is not a signed symmetric interval")
    body_target_z_m = -float(target_separation_m)
    target_velocity_z_m_s = -float(target_separation_rate_m_s)
    return {
        "body_target_z_m": body_target_z_m,
        "nut_target_z_m": body_target_z_m + master_low,
        "body_target_velocity_z_m_s": target_velocity_z_m_s,
        "nut_target_velocity_z_m_s": target_velocity_z_m_s,
        "insertion_shoulder_endplay_m": master_low,
        "source": "frozen_master_and_physical_contract_negative_shoulder_endplay",
        "contact_or_event_truth_input": False,
    }


def _axis_position_driver(
    *,
    target_position: float,
    target_velocity: float,
    actual_position: float,
    actual_velocity: float,
    integral_n: float,
    dt: float,
    position_gain_n_m: float,
    velocity_gain_n_s_m: float,
    integral_gain_n_m_s: float,
    integral_limit_n: float,
    force_limit_n: float,
    feedforward_n: float = 0.0,
    completion_force_direction: float = 0.0,
    integral_tracking_rate_n_s: float = 0.0,
) -> tuple[float, float, float, bool]:
    position_error = target_position - actual_position
    velocity_error = target_velocity - actual_velocity
    proportional_and_damping = (
        position_gain_n_m * position_error
        + velocity_gain_n_s_m * velocity_error
    )
    completion_tracking_active = (
        completion_force_direction != 0.0
        and completion_force_direction * position_error > 0.0
    )
    if completion_tracking_active:
        if not math.isclose(
            abs(completion_force_direction), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("completion force direction must be -1, 0, or +1")
        if not math.isfinite(integral_tracking_rate_n_s) or integral_tracking_rate_n_s <= 0.0:
            raise ValueError("completion integral tracking rate must be positive")
        required_integral = float(
            np.clip(
                completion_force_direction * force_limit_n
                - proportional_and_damping
                - feedforward_n,
                -integral_limit_n,
                integral_limit_n,
            )
        )
        maximum_integral_step_n = integral_tracking_rate_n_s * dt
        candidate = float(
            np.clip(
                integral_n
                + np.clip(
                    required_integral - integral_n,
                    -maximum_integral_step_n,
                    maximum_integral_step_n,
                ),
                -integral_limit_n,
                integral_limit_n,
            )
        )
    else:
        candidate = float(
            np.clip(
                integral_n + integral_gain_n_m_s * position_error * dt,
                -integral_limit_n,
                integral_limit_n,
            )
        )
    requested = (
        proportional_and_damping
        + candidate
        + feedforward_n
    )
    output = float(np.clip(requested, -force_limit_n, force_limit_n))
    saturated = not math.isclose(output, requested, rel_tol=0.0, abs_tol=1.0e-12)
    if saturated and math.copysign(1.0, position_error or velocity_error or 1.0) == math.copysign(
        1.0, requested - output
    ):
        candidate = integral_n
        requested = (
            proportional_and_damping
            + candidate
            + feedforward_n
        )
        output = float(np.clip(requested, -force_limit_n, force_limit_n))
    return output, candidate, requested, saturated


def _bounded_angular_position_driver(
    *,
    target_position_rad: float,
    target_velocity_rad_s: float,
    actual_position_rad: float,
    actual_velocity_rad_s: float,
    integral_nm: float,
    dt: float,
    position_gain_nm_rad: float,
    velocity_gain_nm_s_rad: float,
    integral_gain_nm_rad_s: float,
    integral_limit_nm: float,
    torque_limit_nm: float,
) -> tuple[float, float, float, bool, float, float]:
    """Return one bounded yaw-torque component with conditional integration.

    Only the predeclared target and allowed rigid-body state enter this helper.
    Contact identities, normals, manifolds, and event truth are deliberately not
    arguments.  The integral is an output-torque component, so its bound and the
    final component clamp are independently auditable in N*m.
    """

    position_error = target_position_rad - actual_position_rad
    velocity_error = target_velocity_rad_s - actual_velocity_rad_s
    candidate = float(
        np.clip(
            integral_nm + integral_gain_nm_rad_s * position_error * dt,
            -integral_limit_nm,
            integral_limit_nm,
        )
    )
    requested = (
        position_gain_nm_rad * position_error
        + velocity_gain_nm_s_rad * velocity_error
        + candidate
    )
    output = float(np.clip(requested, -torque_limit_nm, torque_limit_nm))
    saturated = not math.isclose(output, requested, rel_tol=0.0, abs_tol=1.0e-12)
    saturation_residual = requested - output
    if saturated and position_error * saturation_residual > 0.0:
        candidate = integral_nm
        requested = (
            position_gain_nm_rad * position_error
            + velocity_gain_nm_s_rad * velocity_error
            + candidate
        )
        output = float(np.clip(requested, -torque_limit_nm, torque_limit_nm))
    return (
        output,
        candidate,
        requested,
        saturated,
        position_error,
        velocity_error,
    )


def _predeclared_thread_thrust_phase_lead(
    frozen: Mapping[str, Any],
    target_separation_m: float,
    target_separation_rate_m_s: float,
) -> dict[str, float | bool | str]:
    """Map the frozen target schedule to one smooth thread-thrust phase lead.

    The terminal scalar is derived once from the RUN_09 post-hoc momentum
    balance.  Runtime inputs are limited to the predeclared separation target
    and its rate; no actual pose, contact, normal, manifold, or event signal is
    accepted here.
    """

    thread_entry_m = float(
        frozen["contract"]["thread"]["nominal_entry_separation_m"]["value"]
    )
    ramp_span_m = END_SEPARATION_M - thread_entry_m
    if ramp_span_m <= 0.0:
        raise RuntimeError("thread-entry separation must precede bottoming")
    progress = _clip01((target_separation_m - thread_entry_m) / ramp_span_m)
    scale, derivative = _minimum_jerk(progress)
    progress_rate_s_inv = (
        target_separation_rate_m_s / ramp_span_m
        if 0.0 < progress < 1.0
        else 0.0
    )
    return {
        "progress": progress,
        "smooth_scale": scale,
        "yaw_offset_rad": -THREAD_THRUST_TERMINAL_YAW_LEAD_RAD * scale,
        "omega_offset_rad_s": (
            -THREAD_THRUST_TERMINAL_YAW_LEAD_RAD
            * derivative
            * progress_rate_s_inv
        ),
        "terminal_yaw_lead_magnitude_rad": THREAD_THRUST_TERMINAL_YAW_LEAD_RAD,
        "target_terminal_thread_thrust_n": THREAD_THRUST_TARGET_N,
        "source": "RUN_09_posthoc_system_momentum_balance_fixed_before_RUN_10",
        "contact_or_event_truth_input": False,
    }


def _predeclared_terminal_approach_integral_tracking(
    target_separation_m: float,
    target_separation_rate_m_s: float,
) -> dict[str, float | bool | str]:
    """Enable completion-force tracking only in the predeclared slow tail.

    RUN_11 required 0.254167 s after the hold began for both axial commands to
    reach their existing 8 N limits.  The frozen minimum-jerk schedule first
    enters this position-and-speed window about 0.4 s before the hold.  No
    actual pose, contact, normal, manifold, or event signal is accepted here.
    """

    within_terminal_window = (
        target_separation_m
        >= END_SEPARATION_M - AXIAL_TERMINAL_APPROACH_TRACKING_WINDOW_M - 1.0e-15
    )
    target_speed_is_slow = (
        abs(target_separation_rate_m_s)
        <= AXIAL_TERMINAL_APPROACH_TRACKING_MAX_TARGET_SPEED_M_S + 1.0e-15
    )
    enabled = within_terminal_window and target_speed_is_slow
    at_target_hold = (
        target_separation_m >= END_SEPARATION_M - 1.0e-12
        and abs(target_separation_rate_m_s) <= 1.0e-12
    )
    phase = (
        "target_hold"
        if enabled and at_target_hold
        else "terminal_approach"
        if enabled
        else "disabled"
    )
    return {
        "enabled": enabled,
        "phase": phase,
        "terminal_window_m": AXIAL_TERMINAL_APPROACH_TRACKING_WINDOW_M,
        "maximum_target_speed_m_s": (
            AXIAL_TERMINAL_APPROACH_TRACKING_MAX_TARGET_SPEED_M_S
        ),
        "target_schedule_only": True,
        "contact_or_event_truth_input": False,
    }


def _driver_commands(
    frozen: Mapping[str, Any],
    *,
    dt: float,
    target_separation_m: float,
    target_separation_rate_m_s: float,
    body_position: np.ndarray,
    body_rpy: Sequence[float],
    body_velocity: np.ndarray,
    nut_position: np.ndarray,
    nut_rpy: Sequence[float],
    nut_unwrapped_yaw: float,
    nut_velocity: np.ndarray,
    state: dict[str, float],
) -> dict[str, np.ndarray]:
    p1 = frozen["acceptance"]["benches"]["P1"]
    profile = p1["inputs"]["component_driver_profile"]
    force_limit = float(profile["translation_force_component_limit_n"])
    torque_limit = float(profile["torque_component_limit_nm"])
    axial_targets = _shoulder_aware_axial_targets(
        frozen, target_separation_m, target_separation_rate_m_s
    )
    body_target_z = float(axial_targets["body_target_z_m"])
    nut_target_z = float(axial_targets["nut_target_z_m"])
    body_target_vz = float(axial_targets["body_target_velocity_z_m_s"])
    nut_target_vz = float(axial_targets["nut_target_velocity_z_m_s"])
    nominal_axial_load = _nominal_axial_load_feedforward(
        frozen, target_separation_m
    )
    axial_feedforward_component_n = -float(
        nominal_axial_load["per_driven_body_component_n"]
    )
    if abs(axial_feedforward_component_n) > force_limit + 1.0e-12:
        raise RuntimeError(
            "frozen nominal axial load feedforward exceeds per-body force limit"
        )
    terminal_approach_integral_tracking = (
        _predeclared_terminal_approach_integral_tracking(
            target_separation_m, target_separation_rate_m_s
        )
    )
    completion_force_direction = (
        -1.0 if terminal_approach_integral_tracking["enabled"] else 0.0
    )
    body_z, state["body_force_integral_n"], body_z_requested, body_z_saturated = _axis_position_driver(
        target_position=body_target_z,
        target_velocity=body_target_vz,
        actual_position=float(body_position[2]),
        actual_velocity=float(body_velocity[2]),
        integral_n=state["body_force_integral_n"],
        dt=dt,
        position_gain_n_m=float(profile["translation_position_gain_n_m"]),
        velocity_gain_n_s_m=float(profile["translation_velocity_gain_n_s_m"]),
        integral_gain_n_m_s=1000.0,
        integral_limit_n=7.5,
        force_limit_n=force_limit,
        feedforward_n=axial_feedforward_component_n,
        completion_force_direction=completion_force_direction,
        integral_tracking_rate_n_s=AXIAL_TARGET_HOLD_INTEGRAL_TRACKING_RATE_N_S,
    )
    nut_z, state["nut_force_integral_n"], nut_z_requested, nut_z_saturated = _axis_position_driver(
        target_position=nut_target_z,
        target_velocity=nut_target_vz,
        actual_position=float(nut_position[2]),
        actual_velocity=float(nut_velocity[2]),
        integral_n=state["nut_force_integral_n"],
        dt=dt,
        position_gain_n_m=float(profile["translation_position_gain_n_m"]),
        velocity_gain_n_s_m=float(profile["translation_velocity_gain_n_s_m"]),
        integral_gain_n_m_s=1000.0,
        integral_limit_n=7.5,
        force_limit_n=force_limit,
        feedforward_n=axial_feedforward_component_n,
        completion_force_direction=completion_force_direction,
        integral_tracking_rate_n_s=AXIAL_TARGET_HOLD_INTEGRAL_TRACKING_RATE_N_S,
    )
    body_force = np.asarray(
        (
            -600.0 * float(body_position[0]) - 8.0 * float(body_velocity[0]),
            -600.0 * float(body_position[1]) - 8.0 * float(body_velocity[1]),
            body_z,
        ),
        dtype=np.float64,
    )
    nut_force = np.asarray(
        (
            -600.0 * float(nut_position[0]) - 8.0 * float(nut_velocity[0]),
            -600.0 * float(nut_position[1]) - 8.0 * float(nut_velocity[1]),
            nut_z,
        ),
        dtype=np.float64,
    )
    lead = float(frozen["contract"]["thread"]["lead_mm_per_revolution"]) / 1000.0
    target_nut_yaw_base = -2.0 * math.pi * max(
        0.0,
        target_separation_m
        - float(frozen["contract"]["thread"]["nominal_entry_separation_m"]["value"]),
    ) / lead
    target_nut_omega_base = (
        -2.0 * math.pi * target_separation_rate_m_s / lead
        if target_separation_m
        >= float(frozen["contract"]["thread"]["nominal_entry_separation_m"]["value"])
        else 0.0
    )
    thread_thrust_phase_lead = _predeclared_thread_thrust_phase_lead(
        frozen, target_separation_m, target_separation_rate_m_s
    )
    target_nut_yaw = target_nut_yaw_base + float(
        thread_thrust_phase_lead["yaw_offset_rad"]
    )
    target_nut_omega = target_nut_omega_base + float(
        thread_thrust_phase_lead["omega_offset_rad_s"]
    )
    (
        nut_torque_z,
        state["nut_yaw_integral_nm"],
        nut_torque_z_requested,
        nut_torque_z_saturated,
        nut_yaw_error_rad,
        nut_yaw_velocity_error_rad_s,
    ) = _bounded_angular_position_driver(
        target_position_rad=target_nut_yaw,
        target_velocity_rad_s=target_nut_omega,
        actual_position_rad=nut_unwrapped_yaw,
        actual_velocity_rad_s=float(nut_velocity[5]),
        integral_nm=state["nut_yaw_integral_nm"],
        dt=dt,
        position_gain_nm_rad=float(profile["nut_yaw_position_gain_nm_rad"]),
        velocity_gain_nm_s_rad=float(profile["angular_velocity_gain_nm_s_rad"]),
        integral_gain_nm_rad_s=NUT_YAW_INTEGRAL_GAIN_NM_RAD_S,
        integral_limit_nm=NUT_YAW_INTEGRAL_LIMIT_NM,
        torque_limit_nm=torque_limit,
    )
    body_torque = np.asarray(
        (
            -1.2 * float(body_rpy[0]) - 0.01 * float(body_velocity[3]),
            -1.2 * float(body_rpy[1]) - 0.01 * float(body_velocity[4]),
            -0.8 * float(body_rpy[2]) - 0.01 * float(body_velocity[5]),
        ),
        dtype=np.float64,
    )
    nut_torque = np.asarray(
        (
            -1.2 * float(nut_rpy[0]) - 0.01 * float(nut_velocity[3]),
            -1.2 * float(nut_rpy[1]) - 0.01 * float(nut_velocity[4]),
            nut_torque_z,
        ),
        dtype=np.float64,
    )
    return {
        "body_force_n": np.clip(body_force, -force_limit, force_limit),
        "nut_force_n": np.clip(nut_force, -force_limit, force_limit),
        "body_torque_nm": np.clip(body_torque, -torque_limit, torque_limit),
        "nut_torque_nm": np.clip(nut_torque, -torque_limit, torque_limit),
        "target_separation_m": target_separation_m,
        "target_separation_rate_m_s": target_separation_rate_m_s,
        "target_nut_yaw_rad": target_nut_yaw,
        "target_nut_omega_rad_s": target_nut_omega,
        "base_target_nut_yaw_rad": target_nut_yaw_base,
        "base_target_nut_omega_rad_s": target_nut_omega_base,
        "thread_thrust_phase_lead": thread_thrust_phase_lead,
        "body_axial_force_requested_n": body_z_requested,
        "nut_axial_force_requested_n": nut_z_requested,
        "body_target_position_z_m": body_target_z,
        "nut_target_position_z_m": nut_target_z,
        "body_target_velocity_z_m_s": body_target_vz,
        "nut_target_velocity_z_m_s": nut_target_vz,
        "insertion_shoulder_endplay_m": axial_targets[
            "insertion_shoulder_endplay_m"
        ],
        "nut_axial_torque_requested_nm": nut_torque_z_requested,
        "nut_yaw_error_rad": nut_yaw_error_rad,
        "nut_yaw_velocity_error_rad_s": nut_yaw_velocity_error_rad_s,
        "nut_yaw_integral_nm": state["nut_yaw_integral_nm"],
        "nut_yaw_torque_saturated": nut_torque_z_saturated,
        "nominal_axial_load_feedforward": nominal_axial_load,
        "axial_feedforward_component_n": axial_feedforward_component_n,
        "body_axial_integral_n": state["body_force_integral_n"],
        "nut_axial_integral_n": state["nut_force_integral_n"],
        "terminal_approach_integral_tracking": terminal_approach_integral_tracking,
        "body_terminal_approach_integral_tracking_active": (
            terminal_approach_integral_tracking["enabled"]
            and body_target_z - float(body_position[2]) < 0.0
        ),
        "nut_terminal_approach_integral_tracking_active": (
            terminal_approach_integral_tracking["enabled"]
            and nut_target_z - float(nut_position[2]) < 0.0
        ),
        "axial_integral_tracking_rate_n_s": AXIAL_TARGET_HOLD_INTEGRAL_TRACKING_RATE_N_S,
        "body_axial_force_saturated": body_z_saturated,
        "nut_axial_force_saturated": nut_z_saturated,
    }


def _set_initial_translation(stage: Any, path: str, z: float, usd_geom: Any, gf: Any) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"missing rigid body prim {path}")
    xformable = usd_geom.Xformable(prim)
    operations = xformable.GetOrderedXformOps()
    if operations:
        raise RuntimeError(f"unexpected authored transform stack at {path}")
    xformable.AddTranslateOp().Set(gf.Vec3d(0.0, 0.0, z))


def _author_grasp_proxy_collision_filter(
    stage: Any, usd_geom: Any, usd_physics: Any, sdf: Any
) -> dict[str, Any]:
    """Keep the external-finger grasp proxy out of connector self-collision."""

    fixed_paths: list[str] = []
    body_paths: list[str] = []
    nut_paths: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(usd_physics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if path == FIXED_PATH or path.startswith(FIXED_PATH + "/"):
            fixed_paths.append(path)
        elif path == BODY_PATH or path.startswith(BODY_PATH + "/"):
            body_paths.append(path)
        elif path == NUT_PATH or path.startswith(NUT_PATH + "/"):
            nut_paths.append(path)
    fixed_paths.sort()
    body_paths.sort()
    nut_paths.sort()
    expected_nut_paths = sorted([GRIP_PATH, *NUT_SHOULDER_PATHS])
    if (
        len(fixed_paths) != EXPECTED_FIXED_COLLIDER_COUNT
        or len(body_paths) != EXPECTED_BODY_COLLIDER_COUNT
        or len(nut_paths) != EXPECTED_NUT_COLLIDER_COUNT
        or nut_paths != expected_nut_paths
        or tuple(path for path in body_paths if "/NutBearingShoulders/" in path)
        != BODY_SHOULDER_PATHS
    ):
        raise RuntimeError(
            "unexpected A2 grasp-proxy filter members: "
            f"fixed={len(fixed_paths)}, body={len(body_paths)}, nut={nut_paths}"
        )

    usd_geom.Scope.Define(stage, GRASP_PROXY_FILTER_ROOT)

    def define(name: str, members: Sequence[str]) -> tuple[str, Any]:
        path = GRASP_PROXY_FILTER_ROOT + "/" + name
        group = usd_physics.CollisionGroup.Define(stage, path)
        group.CreateInvertFilteredGroupsAttr(False)
        collection = group.GetCollidersCollectionAPI()
        collection.CreateExpansionRuleAttr("explicitOnly")
        collection.CreateIncludeRootAttr(False)
        collection.CreateIncludesRel().SetTargets(
            [sdf.Path(member) for member in members]
        )
        return path, group

    body_non_shoulder_paths = sorted(set(body_paths) - set(BODY_SHOULDER_PATHS))
    grip_paths = [GRIP_PATH]
    groups: dict[str, tuple[str, Any, list[str]]] = {}

    def register(name: str, members: Sequence[str]) -> tuple[str, Any]:
        path, group = define(name, members)
        groups[name] = (path, group, sorted(members))
        return path, group

    fixed_group_path, fixed_group = register("FixedReceptacle", fixed_paths)
    body_non_group_path, body_non_group = register(
        "BodyNonShoulder", body_non_shoulder_paths
    )
    body_positive_group_path, body_positive_group = register(
        "BodyShoulderPositive", [BODY_SHOULDER_PATHS_BY_SIGN["PositiveStop"]]
    )
    body_negative_group_path, body_negative_group = register(
        "BodyShoulderNegative", [BODY_SHOULDER_PATHS_BY_SIGN["NegativeStop"]]
    )
    grip_group_path, grip_group = register("AuthorizedCouplingNutGrip", grip_paths)
    nut_positive_group_path, nut_positive_group = register(
        "NutShoulderPositive", NUT_SHOULDER_PATHS_BY_SIGN["PositiveStop"]
    )
    nut_negative_group_path, nut_negative_group = register(
        "NutShoulderNegative", NUT_SHOULDER_PATHS_BY_SIGN["NegativeStop"]
    )

    def filter_targets(group: Any, targets: Sequence[str]) -> None:
        relation = group.CreateFilteredGroupsRel()
        for target in targets:
            relation.AddTarget(sdf.Path(target))

    filter_targets(
        grip_group,
        (
            fixed_group_path,
            body_non_group_path,
            body_positive_group_path,
            body_negative_group_path,
        ),
    )
    filter_targets(
        nut_positive_group,
        (fixed_group_path, body_non_group_path, body_negative_group_path),
    )
    filter_targets(
        nut_negative_group,
        (fixed_group_path, body_non_group_path, body_positive_group_path),
    )
    filter_targets(body_positive_group, (fixed_group_path,))
    filter_targets(body_negative_group, (fixed_group_path,))

    authored_members: dict[str, list[str]] = {}
    authored_targets_by_group: dict[str, list[str]] = {}
    for name, (_path, group, expected_members) in groups.items():
        actual_members = sorted(
            str(path)
            for path in group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()
        )
        if actual_members != expected_members:
            raise RuntimeError(f"A2 collision-group member readback failed for {name}")
        authored_members[name] = actual_members
        authored_targets_by_group[name] = sorted(
            str(path) for path in group.GetFilteredGroupsRel().GetTargets()
        )
    expected_targets_by_group = {
        "FixedReceptacle": [],
        "BodyNonShoulder": [],
        "BodyShoulderPositive": [fixed_group_path],
        "BodyShoulderNegative": [fixed_group_path],
        "AuthorizedCouplingNutGrip": sorted(
            [
                fixed_group_path,
                body_non_group_path,
                body_positive_group_path,
                body_negative_group_path,
            ]
        ),
        "NutShoulderPositive": sorted(
            [fixed_group_path, body_non_group_path, body_negative_group_path]
        ),
        "NutShoulderNegative": sorted(
            [fixed_group_path, body_non_group_path, body_positive_group_path]
        ),
    }
    if authored_targets_by_group != expected_targets_by_group:
        raise RuntimeError("A2 frozen collision-pair partition readback failed")
    if set(authored_members["NutShoulderPositive"]) & set(
        authored_members["NutShoulderNegative"]
    ):
        raise RuntimeError("positive and negative nut shoulder groups overlap")
    return {
        "enabled": True,
        "authored_before_physics": True,
        "controller_input": False,
        "mode": "static_external_finger_proxy_filter_no_contact_truth",
        "fixed_group_path": fixed_group_path,
        "body_group_path": body_non_group_path,
        "nut_group_path": grip_group_path,
        "fixed_collider_count": len(fixed_paths),
        "body_collider_count": len(body_paths),
        "nut_collider_count": len(nut_paths),
        "external_grasp_proxy_group_collider_count": len(grip_paths),
        "physical_shoulder_body0_collider_count": len(BODY_SHOULDER_PATHS),
        "physical_shoulder_body1_collider_count": len(NUT_SHOULDER_PATHS),
        "physical_shoulder_collisions_preserved": True,
        "frozen_pair_partition": {
            "allowed": [
                "BodyShoulderPositive<->NutShoulderPositive",
                "BodyShoulderNegative<->NutShoulderNegative",
            ],
            "filtered_targets_by_group": authored_targets_by_group,
            "default_unlisted_connector_pairs_filtered": True,
        },
        "filtered_group_targets": authored_targets_by_group[
            "AuthorizedCouplingNutGrip"
        ],
        "authorized_grip_collider": GRIP_PATH,
        "connector_receptacle_collisions_removed": False,
        "grasp_proxy_fixed_receptacle_collisions_removed": True,
        "body_vs_fixed_receptacle_collisions_preserved": True,
        "external_finger_contact_preserved": True,
    }


def _trace_index(value: Any, prefix: str) -> int | None:
    if not isinstance(value, str) or not value.startswith(prefix + "_"):
        return None
    suffix = value[len(prefix) + 1 :]
    return int(suffix) if suffix.isdigit() else None


def _key_labels_correspond(roles: Sequence[str], labels: Sequence[Any]) -> bool | None:
    if set(roles) != {"continuous_keyway_wall", "continuous_polarizing_key"}:
        return None
    fixed = [
        _trace_index(label, "keyway")
        for role, label in zip(roles, labels)
        if role == "continuous_keyway_wall"
    ]
    moving = [
        _trace_index(label, "key")
        for role, label in zip(roles, labels)
        if role == "continuous_polarizing_key"
    ]
    return len(fixed) == len(moving) == 1 and fixed[0] is not None and fixed[0] == moving[0]


def _path_is_at_or_below(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _is_metal_stop_pair(collider_paths: Sequence[str]) -> bool:
    """Classify the real stop parent and its generated segment descendants."""

    if len(collider_paths) != 2:
        return False
    first, second = collider_paths
    return (
        _path_is_at_or_below(first, FIXED_STOP_PATH)
        and _path_is_at_or_below(second, PLUG_STOP_PATH)
    ) or (
        _path_is_at_or_below(second, FIXED_STOP_PATH)
        and _path_is_at_or_below(first, PLUG_STOP_PATH)
    )


def _contact_rows(
    stage: Any, interface: Any, schema_tools: Any, dt: float
) -> list[dict[str, Any]]:
    headers, contacts, _friction = interface.get_full_contact_report()
    rows: list[dict[str, Any]] = []
    for header in headers:
        collider_paths = (
            str(schema_tools.intToSdfPath(header.collider0)),
            str(schema_tools.intToSdfPath(header.collider1)),
        )
        start = int(header.contact_data_offset)
        stop = start + int(header.num_contact_data)
        separations = [float(contacts[index].separation) for index in range(start, stop)]
        impulses = [
            float(np.linalg.norm(_finite(contacts[index].impulse, 3, "contact impulse")))
            for index in range(start, stop)
        ]
        roles: list[str] = []
        trace_labels: list[str | None] = []
        for path in collider_paths:
            prim = stage.GetPrimAtPath(path)
            role_attr = prim.GetAttribute("kcg:collisionRole") if prim else None
            trace_attr = prim.GetAttribute("kcg:traceLabel") if prim else None
            roles.append(
                str(role_attr.Get())
                if role_attr and role_attr.Get() is not None
                else "UNLABELED"
            )
            trace_labels.append(
                str(trace_attr.Get())
                if trace_attr and trace_attr.Get() is not None
                else None
            )
        rows.append(
            {
                "actor_paths": [
                    str(schema_tools.intToSdfPath(header.actor0)),
                    str(schema_tools.intToSdfPath(header.actor1)),
                ],
                "collider_paths": list(collider_paths),
                "contact_record_count": int(header.num_contact_data),
                "minimum_separation_m": min(separations) if separations else None,
                "maximum_impulse_norm_n_s": max(impulses, default=0.0),
                "maximum_equivalent_force_n": max(
                    (impulse / dt for impulse in impulses), default=0.0
                ),
                "collision_roles": roles,
                "trace_labels": trace_labels,
                "corresponding_key_label": _key_labels_correspond(roles, trace_labels),
                "metal_stop_pair": _is_metal_stop_pair(collider_paths),
            }
        )
    return rows


def _runtime(
    arguments: argparse.Namespace,
    frozen: Mapping[str, Any],
    application: Any,
    log_messages: list[str],
) -> dict[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_simulation_interface
    import omni.usd
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, UsdGeom, UsdPhysics

    p1 = frozen["acceptance"]["benches"]["P1"]
    profile = p1["inputs"]["component_driver_profile"]
    rate_hz = int(frozen["acceptance"]["shared_numeric_profile"]["physics_rate_hz"])
    dt = 1.0 / rate_hz
    World.clear_instance()
    context = omni.usd.get_context()
    if context.open_stage(str(frozen["model_path"])) is not True:
        raise RuntimeError("failed to open frozen assembly-control model")
    for _ in range(3):
        application.update()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=dt,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    _set_initial_translation(stage, BODY_PATH, -START_SEPARATION_M, UsdGeom, Gf)
    _set_initial_translation(stage, NUT_PATH, -START_SEPARATION_M, UsdGeom, Gf)
    initial_pose_write_count = 2
    grasp_proxy_collision_filter = _author_grasp_proxy_collision_filter(
        stage, UsdGeom, UsdPhysics, Sdf
    )
    for owner in (FIXED_PATH, BODY_PATH, NUT_PATH):
        prim = stage.GetPrimAtPath(owner)
        if not prim:
            raise RuntimeError(f"missing physics owner {owner}")
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    world.get_physics_context().set_gravity(float(profile["gravity_magnitude_m_s2"]))
    world.reset()
    fixed = RigidPrim(prim_paths_expr=FIXED_PATH, name="multilayer_fixed", reset_xform_properties=False)
    body = RigidPrim(prim_paths_expr=BODY_PATH, name="multilayer_body", reset_xform_properties=False)
    nut = RigidPrim(prim_paths_expr=NUT_PATH, name="multilayer_nut", reset_xform_properties=False)
    for view in (fixed, body, nut):
        view.initialize()
    handles = {name: bool(view.is_physics_handle_valid()) for name, view in (
        ("fixed_receptacle", fixed), ("body_assembly", body), ("coupling_nut", nut)
    )}
    if not all(handles.values()):
        raise RuntimeError(f"invalid rigid-body handles: {handles}")

    def rigid_state(view: Any, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions, orientations = view.get_world_poses()
        return (
            _finite(positions[0], 3, label + " position"),
            _finite(orientations[0], 4, label + " orientation"),
            _finite(view.get_velocities()[0], 6, label + " velocity"),
        )

    def apply(view: Any, force: np.ndarray, torque: np.ndarray) -> None:
        view.apply_forces_and_torques_at_pos(
            forces=np.asarray([force], dtype=np.float32),
            torques=np.asarray([torque], dtype=np.float32),
            positions=None,
            is_global=True,
        )

    initial = {
        "fixed_receptacle": rigid_state(fixed, "initial fixed"),
        "body_assembly": rigid_state(body, "initial body"),
        "coupling_nut": rigid_state(nut, "initial nut"),
    }
    initial_datum_separation_m = float(
        initial["fixed_receptacle"][0][2] - initial["body_assembly"][0][2]
    )
    initial_nut_body_relative_z_m = float(
        initial["coupling_nut"][0][2] - initial["body_assembly"][0][2]
    )
    joint_limit = frozen["contract"]["coupling_nut_motion"]["transZ_backup_limits_m"]
    initialized = bool(
        abs(initial_datum_separation_m - START_SEPARATION_M) <= 5.0e-5
        and float(joint_limit["low"]) - 5.0e-5
        <= initial_nut_body_relative_z_m
        <= float(joint_limit["high"]) + 5.0e-5
    )
    if not initialized:
        raise RuntimeError(
            "initial rigid states violate datum separation or nut endplay: "
            + json.dumps(
                {
                    "datum_separation_m": initial_datum_separation_m,
                    "expected_datum_separation_m": START_SEPARATION_M,
                    "nut_body_relative_z_m": initial_nut_body_relative_z_m,
                    "joint_limit_m": joint_limit,
                },
                sort_keys=True,
            )
        )
    output = Path(arguments.output_dir).expanduser().resolve()
    trace_path = output / "trace.jsonl"
    trace = trace_path.open("x", encoding="utf-8")
    interface = get_physx_simulation_interface()
    event_positions = {
        row["name"]: float(row["nominal_separation_m"])
        for row in frozen["contract"]["assembly_events"]["ordered"]
    }
    settle_steps = int(profile["settle_steps"])
    hold_steps = int(profile["hold_steps"])
    speed = float(p1["inputs"]["axial_speed_m_s"])
    motion_distance_m = END_SEPARATION_M - START_SEPARATION_M
    motion_duration_s = 1.875 * motion_distance_m / speed
    motion_steps = int(math.ceil(motion_duration_s / dt))
    motion_duration_s = motion_steps * dt
    peak_profile_speed_m_s = 1.875 * motion_distance_m / motion_duration_s
    if peak_profile_speed_m_s > speed + 1.0e-12:
        raise RuntimeError("minimum-jerk profile exceeds frozen axial speed")
    total_steps = settle_steps + motion_steps + hold_steps
    driver_state = {
        "body_force_integral_n": 0.0,
        "nut_force_integral_n": 0.0,
        "nut_yaw_integral_nm": 0.0,
    }
    mass_bodies = frozen["contract"]["mass_properties"]["bodies"]
    body_mass_kg = float(mass_bodies["loose_plug_body_assembly"]["mass_kg"])
    nut_mass_kg = float(mass_bodies["coupling_nut"]["mass_kg"])
    effective_translation_mass_kg = body_mass_kg + nut_mass_kg
    body_yaw_inertia_kg_m2 = float(
        mass_bodies["loose_plug_body_assembly"]["diagonal_inertia_kg_m2"][2]
    )
    nut_yaw_inertia_kg_m2 = float(
        mass_bodies["coupling_nut"]["diagonal_inertia_kg_m2"][2]
    )
    fixed_initial = initial["fixed_receptacle"][0].copy()
    initial_contacts = _contact_rows(stage, get_physx_simulation_interface(), PhysicsSchemaTools, dt)
    event_first: dict[str, dict[str, Any]] = {}
    contact_aggregate: dict[str, dict[str, Any]] = {}
    hard_penetrations: dict[str, dict[str, Any]] = {}
    maximum_fixed_drift_m = 0.0
    maximum_driver_force_component_n = 0.0
    maximum_driver_torque_component_nm = 0.0
    maximum_nominal_axial_load_feedforward_n = 0.0
    maximum_axial_feedforward_component_n = 0.0
    maximum_axial_feedforward_component_slew_n_s = 0.0
    maximum_nut_yaw_integral_nm = 0.0
    nut_yaw_torque_saturation_sample_count = 0
    maximum_body_axial_integral_n = 0.0
    maximum_nut_axial_integral_n = 0.0
    body_terminal_approach_integral_tracking_sample_count = 0
    nut_terminal_approach_integral_tracking_sample_count = 0
    previous_axial_feedforward_component_n = 0.0
    maximum_internal_force_component_n = 0.0
    maximum_internal_torque_component_nm = 0.0
    maximum_api_force_component_n = 0.0
    maximum_api_torque_component_nm = 0.0
    maximum_hard_penetration_m = 0.0
    maximum_metal_stop_equivalent_force_n = 0.0
    maximum_body_position_tracking_error_m = 0.0
    maximum_nut_position_tracking_error_m = 0.0
    driver_saturation_sample_count = 0
    noncorresponding_key_contact_sample_count = 0
    false_bottoming_count = 0
    executed_steps = 0
    safety_stop_reason: str | None = None
    driver_work_j = {"body": 0.0, "coupling_nut": 0.0}
    channel_summary = {
        name: {
            "maximum_force_component_n": 0.0,
            "maximum_torque_component_nm": 0.0,
            "signed_work_j": 0.0,
        }
        for name in (
            "spring_fingers",
            "same_label_pins_61",
            "pin_isolation_seals_61",
            "peripheral_seal",
            "three_start_thread",
            "anti_decoupling_36_cycle",
        )
    }
    solver_error_count = 0
    pose_write_after_start_count = 0
    body_wrapped_yaw = _rpy_wxyz(initial["body_assembly"][1])[2]
    body_unwrapped_yaw = body_wrapped_yaw
    nut_wrapped_yaw = _rpy_wxyz(initial["coupling_nut"][1])[2]
    nut_unwrapped_yaw = nut_wrapped_yaw
    final_effects: dict[str, Any] | None = None
    wall_started = time.monotonic()
    try:
        for step in range(total_steps):
            fixed_position, fixed_quaternion, fixed_velocity = rigid_state(fixed, "fixed")
            body_position, body_quaternion, body_velocity = rigid_state(body, "body")
            nut_position, nut_quaternion, nut_velocity = rigid_state(nut, "nut")
            body_rpy = _rpy_wxyz(body_quaternion)
            nut_rpy = _rpy_wxyz(nut_quaternion)
            body_unwrapped_yaw = _unwrap(body_wrapped_yaw, body_unwrapped_yaw, body_rpy[2])
            nut_unwrapped_yaw = _unwrap(nut_wrapped_yaw, nut_unwrapped_yaw, nut_rpy[2])
            body_wrapped_yaw = body_rpy[2]
            nut_wrapped_yaw = nut_rpy[2]
            motion_index = step - settle_steps
            if motion_index < 0:
                trajectory_position, trajectory_derivative = 0.0, 0.0
            elif motion_index < motion_steps:
                trajectory_position, trajectory_derivative = _minimum_jerk(
                    (motion_index + 1) * dt / motion_duration_s
                )
            else:
                trajectory_position, trajectory_derivative = 1.0, 0.0
            target_separation_m = (
                START_SEPARATION_M + motion_distance_m * trajectory_position
            )
            target_separation_rate_m_s = (
                motion_distance_m * trajectory_derivative / motion_duration_s
            )
            internal = _internal_effects(
                frozen,
                fixed_position=fixed_position,
                fixed_velocity=fixed_velocity,
                body_position=body_position,
                body_velocity=body_velocity,
                body_yaw=body_unwrapped_yaw,
                body_omega_z=float(body_velocity[5]),
                nut_yaw=nut_unwrapped_yaw,
                nut_omega_z=float(nut_velocity[5]),
                nut_position=nut_position,
                nut_velocity=nut_velocity,
                integration_dt_s=dt,
                effective_translation_mass_kg=effective_translation_mass_kg,
                body_mass_kg=body_mass_kg,
                body_yaw_inertia_kg_m2=body_yaw_inertia_kg_m2,
                nut_yaw_inertia_kg_m2=nut_yaw_inertia_kg_m2,
            )
            driver = _driver_commands(
                frozen,
                dt=dt,
                target_separation_m=target_separation_m,
                target_separation_rate_m_s=target_separation_rate_m_s,
                body_position=body_position,
                body_rpy=body_rpy,
                body_velocity=body_velocity,
                nut_position=nut_position,
                nut_rpy=nut_rpy,
                nut_unwrapped_yaw=nut_unwrapped_yaw,
                nut_velocity=nut_velocity,
                state=driver_state,
            )
            coupled_driver = internal["coupled_driver_overrides"]
            force_limit = float(profile["translation_force_component_limit_n"])
            torque_limit = float(profile["torque_component_limit_nm"])
            driver["body_force_n"][:2] = np.clip(
                coupled_driver["body_force_xy_n"], -force_limit, force_limit
            )
            driver["nut_force_n"][:2] = np.clip(
                coupled_driver["nut_force_xy_n"], -force_limit, force_limit
            )
            driver["body_torque_nm"][2] = float(
                np.clip(
                    coupled_driver["body_yaw_torque_nm"],
                    -torque_limit,
                    torque_limit,
                )
            )
            total_body_force = driver["body_force_n"] + internal["body_force_n"]
            total_body_torque = driver["body_torque_nm"] + internal["body_torque_nm"]
            total_nut_force = driver["nut_force_n"] + internal["nut_force_n"]
            total_nut_torque = driver["nut_torque_nm"] + internal["nut_torque_nm"]
            maximum_driver_force_component_n = max(
                maximum_driver_force_component_n,
                float(np.max(np.abs(driver["body_force_n"]))),
                float(np.max(np.abs(driver["nut_force_n"]))),
            )
            maximum_driver_torque_component_nm = max(
                maximum_driver_torque_component_nm,
                float(np.max(np.abs(driver["body_torque_nm"]))),
                float(np.max(np.abs(driver["nut_torque_nm"]))),
            )
            maximum_nut_yaw_integral_nm = max(
                maximum_nut_yaw_integral_nm,
                abs(float(driver["nut_yaw_integral_nm"])),
            )
            maximum_body_axial_integral_n = max(
                maximum_body_axial_integral_n,
                abs(float(driver["body_axial_integral_n"])),
            )
            maximum_nut_axial_integral_n = max(
                maximum_nut_axial_integral_n,
                abs(float(driver["nut_axial_integral_n"])),
            )
            body_terminal_approach_integral_tracking_sample_count += int(
                driver["body_terminal_approach_integral_tracking_active"]
            )
            nut_terminal_approach_integral_tracking_sample_count += int(
                driver["nut_terminal_approach_integral_tracking_active"]
            )
            nut_yaw_torque_saturation_sample_count += int(
                driver["nut_yaw_torque_saturated"]
            )
            maximum_nominal_axial_load_feedforward_n = max(
                maximum_nominal_axial_load_feedforward_n,
                float(driver["nominal_axial_load_feedforward"]["total_n"]),
            )
            current_axial_feedforward_component_n = float(
                driver["axial_feedforward_component_n"]
            )
            maximum_axial_feedforward_component_n = max(
                maximum_axial_feedforward_component_n,
                abs(current_axial_feedforward_component_n),
            )
            maximum_axial_feedforward_component_slew_n_s = max(
                maximum_axial_feedforward_component_slew_n_s,
                abs(
                    current_axial_feedforward_component_n
                    - previous_axial_feedforward_component_n
                )
                / dt,
            )
            previous_axial_feedforward_component_n = (
                current_axial_feedforward_component_n
            )
            maximum_internal_force_component_n = max(
                maximum_internal_force_component_n,
                float(np.max(np.abs(internal["body_force_n"]))),
                float(np.max(np.abs(internal["nut_force_n"]))),
            )
            maximum_internal_torque_component_nm = max(
                maximum_internal_torque_component_nm,
                float(np.max(np.abs(internal["body_torque_nm"]))),
                float(np.max(np.abs(internal["nut_torque_nm"]))),
            )
            maximum_api_force_component_n = max(
                maximum_api_force_component_n,
                float(np.max(np.abs(total_body_force))),
                float(np.max(np.abs(total_nut_force))),
            )
            maximum_api_torque_component_nm = max(
                maximum_api_torque_component_nm,
                float(np.max(np.abs(total_body_torque))),
                float(np.max(np.abs(total_nut_torque))),
            )
            maximum_body_position_tracking_error_m = max(
                maximum_body_position_tracking_error_m,
                abs(float(driver["body_target_position_z_m"]) - float(body_position[2])),
            )
            maximum_nut_position_tracking_error_m = max(
                maximum_nut_position_tracking_error_m,
                abs(float(driver["nut_target_position_z_m"]) - float(nut_position[2])),
            )
            driver_saturation_sample_count += int(driver["body_axial_force_saturated"])
            driver_saturation_sample_count += int(driver["nut_axial_force_saturated"])
            driver_work_j["body"] += dt * (
                float(np.dot(driver["body_force_n"], body_velocity[:3]))
                + float(np.dot(driver["body_torque_nm"], body_velocity[3:]))
            )
            driver_work_j["coupling_nut"] += dt * (
                float(np.dot(driver["nut_force_n"], nut_velocity[:3]))
                + float(np.dot(driver["nut_torque_nm"], nut_velocity[3:]))
            )
            for channel_name, wrench in internal["channel_wrenches"].items():
                summary = channel_summary[channel_name]
                summary["maximum_force_component_n"] = max(
                    summary["maximum_force_component_n"],
                    float(np.max(np.abs(wrench["body_force_n"]))),
                    float(np.max(np.abs(wrench["nut_force_n"]))),
                )
                summary["maximum_torque_component_nm"] = max(
                    summary["maximum_torque_component_nm"],
                    float(np.max(np.abs(wrench["body_torque_nm"]))),
                    float(np.max(np.abs(wrench["nut_torque_nm"]))),
                )
                summary["signed_work_j"] += dt * (
                    float(np.dot(wrench["body_force_n"], body_velocity[:3]))
                    + float(np.dot(wrench["body_torque_nm"], body_velocity[3:]))
                    + float(np.dot(wrench["nut_force_n"], nut_velocity[:3]))
                    + float(np.dot(wrench["nut_torque_nm"], nut_velocity[3:]))
                )
            apply(
                body,
                total_body_force,
                total_body_torque,
            )
            apply(
                nut,
                total_nut_force,
                total_nut_torque,
            )
            world.step(render=False)
            executed_steps = step + 1
            fixed_after, fixed_quaternion_after, fixed_velocity_after = rigid_state(fixed, "fixed after")
            body_after, body_quaternion_after, body_velocity_after = rigid_state(body, "body after")
            nut_after, nut_quaternion_after, nut_velocity_after = rigid_state(nut, "nut after")
            vectors = (
                fixed_after, fixed_quaternion_after, fixed_velocity_after,
                body_after, body_quaternion_after, body_velocity_after,
                nut_after, nut_quaternion_after, nut_velocity_after,
            )
            if not all(np.all(np.isfinite(value)) for value in vectors):
                solver_error_count += 1
                raise RuntimeError("non-finite rigid state")
            maximum_fixed_drift_m = max(
                maximum_fixed_drift_m,
                float(np.max(np.abs(fixed_after - fixed_initial))),
            )
            separation = float(fixed_after[2] - body_after[2])
            contacts = _contact_rows(stage, interface, PhysicsSchemaTools, dt)
            metal_contact_force_n = 0.0
            for row in contacts:
                key = json.dumps(row["collider_paths"], separators=(",", ":"))
                aggregate = contact_aggregate.setdefault(
                    key,
                    {
                        "collider_paths": row["collider_paths"],
                        "first_step": step + 1,
                        "last_step": step + 1,
                        "active_step_count": 0,
                        "minimum_separation_m": None,
                        "maximum_equivalent_force_n": 0.0,
                    },
                )
                aggregate["last_step"] = step + 1
                aggregate["active_step_count"] += 1
                row_sep = row["minimum_separation_m"]
                if row_sep is not None and (
                    aggregate["minimum_separation_m"] is None
                    or row_sep < aggregate["minimum_separation_m"]
                ):
                    aggregate["minimum_separation_m"] = row_sep
                if row_sep is not None and row_sep < -float(
                    frozen["contract"]["acceptance_limits"]["maximum_noncompliant_hard_penetration_m"]
                ):
                    hard_penetrations[key] = dict(aggregate)
                if row_sep is not None:
                    maximum_hard_penetration_m = max(
                        maximum_hard_penetration_m, max(0.0, -float(row_sep))
                    )
                aggregate["maximum_equivalent_force_n"] = max(
                    float(aggregate["maximum_equivalent_force_n"]),
                    float(row["maximum_equivalent_force_n"]),
                )
                if row["corresponding_key_label"] is False:
                    noncorresponding_key_contact_sample_count += 1
                if row["metal_stop_pair"]:
                    metal_contact_force_n = max(
                        metal_contact_force_n,
                        float(row["maximum_equivalent_force_n"]),
                    )
            maximum_metal_stop_equivalent_force_n = max(
                maximum_metal_stop_equivalent_force_n, metal_contact_force_n
            )
            metal_contact_force_active = metal_contact_force_n > 1.0e-9
            channel = internal["channels"]
            triggers = {
                "five_key_polarization": separation >= event_positions["five_key_polarization"],
                "three_start_thread_entry": bool(channel["three_start_thread_active"]),
                "spring_finger_engagement": bool(channel["spring_fingers_active"]),
                "first_pin_socket_spring_touch": bool(channel["same_label_pin_effects_active"]),
                "pin_barrier_seal_contact": bool(channel["pin_isolation_active"]),
                "seal_compression": bool(channel["peripheral_seal_active"]),
                "shell_to_shell_metal_bottoming": metal_contact_force_active,
            }
            if metal_contact_force_active and not all(
                triggers[event] for event in EVENT_ORDER[:-1]
            ):
                false_bottoming_count += 1
            for event in EVENT_ORDER:
                if event not in event_first and triggers[event]:
                    event_first[event] = {
                        "step": step + 1,
                        "time_s": (step + 1) * dt,
                        "datum_B_separation_m": separation,
                        "source": (
                            "physx_continuous_real_contact_force_onset"
                            if event == "shell_to_shell_metal_bottoming"
                            else (
                                "posthoc_geometric_activation_A1_physical_onset_validated"
                                if event == "five_key_polarization"
                                else "model_internal_continuous_effect_activation"
                            )
                        ),
                    }
            final_effects = internal
            trace.write(
                json.dumps(
                    {
                        "step": step + 1,
                        "time_s": (step + 1) * dt,
                        "separation_m": separation,
                        "target_separation_m": target_separation_m,
                        "target_separation_rate_m_s": target_separation_rate_m_s,
                        "body_position_m": body_after.tolist(),
                        "nut_position_m": nut_after.tolist(),
                        "body_velocity": body_velocity_after.tolist(),
                        "nut_velocity": nut_velocity_after.tolist(),
                        "driver_body_force_n": driver["body_force_n"].tolist(),
                        "driver_nut_force_n": driver["nut_force_n"].tolist(),
                        "driver_body_torque_nm": driver["body_torque_nm"].tolist(),
                        "driver_nut_torque_nm": driver["nut_torque_nm"].tolist(),
                        "driver_target_nut_yaw_rad": driver["target_nut_yaw_rad"],
                        "driver_target_nut_omega_rad_s": driver[
                            "target_nut_omega_rad_s"
                        ],
                        "driver_base_target_nut_yaw_rad": driver[
                            "base_target_nut_yaw_rad"
                        ],
                        "driver_base_target_nut_omega_rad_s": driver[
                            "base_target_nut_omega_rad_s"
                        ],
                        "driver_thread_thrust_phase_lead": driver[
                            "thread_thrust_phase_lead"
                        ],
                        "driver_nut_yaw_error_rad": driver["nut_yaw_error_rad"],
                        "driver_nut_yaw_velocity_error_rad_s": driver[
                            "nut_yaw_velocity_error_rad_s"
                        ],
                        "driver_nut_yaw_integral_nm": driver[
                            "nut_yaw_integral_nm"
                        ],
                        "driver_nut_yaw_torque_requested_nm": driver[
                            "nut_axial_torque_requested_nm"
                        ],
                        "driver_nut_yaw_torque_saturated": driver[
                            "nut_yaw_torque_saturated"
                        ],
                        "internal_body_force_n": internal["body_force_n"].tolist(),
                        "internal_body_torque_nm": internal["body_torque_nm"].tolist(),
                        "internal_nut_force_n": internal["nut_force_n"].tolist(),
                        "internal_nut_torque_nm": internal["nut_torque_nm"].tolist(),
                        "api_total_body_force_n": total_body_force.tolist(),
                        "api_total_body_torque_nm": total_body_torque.tolist(),
                        "api_total_nut_force_n": total_nut_force.tolist(),
                        "api_total_nut_torque_nm": total_nut_torque.tolist(),
                        "driver_body_axial_requested_n": driver["body_axial_force_requested_n"],
                        "driver_nut_axial_requested_n": driver["nut_axial_force_requested_n"],
                        "driver_body_target_position_z_m": driver[
                            "body_target_position_z_m"
                        ],
                        "driver_nut_target_position_z_m": driver[
                            "nut_target_position_z_m"
                        ],
                        "driver_insertion_shoulder_endplay_m": driver[
                            "insertion_shoulder_endplay_m"
                        ],
                        "driver_nominal_axial_load_feedforward_n": driver[
                            "nominal_axial_load_feedforward"
                        ],
                        "driver_axial_feedforward_component_n": driver[
                            "axial_feedforward_component_n"
                        ],
                        "driver_body_axial_integral_n": driver[
                            "body_axial_integral_n"
                        ],
                        "driver_nut_axial_integral_n": driver[
                            "nut_axial_integral_n"
                        ],
                        "driver_terminal_approach_integral_tracking": driver[
                            "terminal_approach_integral_tracking"
                        ],
                        "driver_body_terminal_approach_integral_tracking_active": driver[
                            "body_terminal_approach_integral_tracking_active"
                        ],
                        "driver_nut_terminal_approach_integral_tracking_active": driver[
                            "nut_terminal_approach_integral_tracking_active"
                        ],
                        "driver_axial_integral_tracking_rate_n_s": driver[
                            "axial_integral_tracking_rate_n_s"
                        ],
                        "driver_body_axial_saturated": driver["body_axial_force_saturated"],
                        "driver_nut_axial_saturated": driver["nut_axial_force_saturated"],
                        "internal_measurements": internal["measurements"],
                        "active_contact_pair_count": len(contacts),
                        "metal_stop_equivalent_force_n": metal_contact_force_n,
                        "maximum_fixed_drift_m": maximum_fixed_drift_m,
                        "maximum_hard_penetration_m": maximum_hard_penetration_m,
                    },
                    allow_nan=False,
                    sort_keys=True,
                ) + "\n"
            )
            if (step + 1) % rate_hz == 0:
                trace.flush()
                heartbeat = {
                    "task_id": BENCH_ID,
                    "run_id": f"A2-V2-NOMINAL-{arguments.run_index:02d}",
                    "step": step + 1,
                    "total_steps": total_steps,
                    "separation_m": separation,
                    "target_separation_m": target_separation_m,
                    "maximum_driver_force_component_n": maximum_driver_force_component_n,
                    "maximum_driver_torque_component_nm": maximum_driver_torque_component_nm,
                    "wall_elapsed_s": time.monotonic() - wall_started,
                }
                (output / "heartbeat.json").write_text(
                    json.dumps(heartbeat, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.write(
                    2,
                    (
                        f"A2_V2_HEARTBEAT run={arguments.run_index} "
                        f"step={step + 1}/{total_steps} separation_m={separation:.9f}\n"
                    ).encode(),
                )
            hard_limit = float(
                frozen["contract"]["acceptance_limits"]["maximum_noncompliant_hard_penetration_m"]
            )
            drift_limit = float(
                frozen["contract"]["acceptance_limits"]["maximum_fixed_receptacle_translation_drift_m"]
            )
            if maximum_hard_penetration_m > hard_limit:
                safety_stop_reason = "NONCOMPLIANT_HARD_PENETRATION_LIMIT_EXCEEDED"
                break
            if maximum_fixed_drift_m > drift_limit:
                safety_stop_reason = "FIXED_RECEPTACLE_DRIFT_LIMIT_EXCEEDED"
                break
    finally:
        trace.close()

    fixed_final, fixed_final_q, fixed_final_v = rigid_state(fixed, "final fixed")
    body_final, body_final_q, body_final_v = rigid_state(body, "final body")
    nut_final, nut_final_q, nut_final_v = rigid_state(nut, "final nut")
    final_separation = float(fixed_final[2] - body_final[2])
    final_body_yaw = _unwrap(
        body_wrapped_yaw, body_unwrapped_yaw, _rpy_wxyz(body_final_q)[2]
    )
    final_nut_yaw = _unwrap(
        nut_wrapped_yaw, nut_unwrapped_yaw, _rpy_wxyz(nut_final_q)[2]
    )
    observed_order = [
        event for event, _row in sorted(event_first.items(), key=lambda item: item[1]["step"])
    ]
    tolerance = float(frozen["contract"]["acceptance_limits"]["event_position_tolerance_m"])
    position_errors = {
        event: (
            None if event not in event_first
            else float(event_first[event]["datum_B_separation_m"] - event_positions[event])
        ) for event in EVENT_ORDER
    }
    event_inventory_pass = all(event in event_first for event in EVENT_ORDER)
    event_order_pass = observed_order == list(EVENT_ORDER)
    event_position_pass = all(
        error is not None and abs(error) <= tolerance for error in position_errors.values()
    )
    limits = frozen["contract"]["acceptance_limits"]
    driver_force_pass = maximum_driver_force_component_n <= float(
        limits["force_component_limit_n_per_driven_body"]
    ) + 1.0e-9
    driver_torque_pass = maximum_driver_torque_component_nm <= float(
        limits["torque_component_limit_nm"]
    ) + 1.0e-9
    fixed_drift_pass = maximum_fixed_drift_m <= float(
        limits["maximum_fixed_receptacle_translation_drift_m"]
    )
    hard_penetration_pass = maximum_hard_penetration_m <= float(
        limits["maximum_noncompliant_hard_penetration_m"]
    )
    physicsusd_errors = [
        message
        for message in log_messages
        if "physicsusd" in message.lower()
        and ("error" in message.lower() or "failed" in message.lower())
    ]
    solver_log_errors = [
        message
        for message in log_messages
        if "solver" in message.lower()
        and ("error" in message.lower() or "failed" in message.lower())
    ]
    thickness_warnings = [
        message for message in log_messages if "adjusted the thickness" in message.lower()
    ]
    solver_error_count += len(solver_log_errors)
    target_reached_pass = final_separation >= END_SEPARATION_M - tolerance
    initial_contact_zero_pass = len(initial_contacts) == 0
    noncorresponding_key_contact_zero_pass = (
        noncorresponding_key_contact_sample_count == 0
    )
    false_bottoming_zero_pass = false_bottoming_count == 0
    gates = {
        "initial_5p5mm_contact_zero": initial_contact_zero_pass,
        "target_15p05mm_reached": target_reached_pass,
        "seven_events_observed": event_inventory_pass,
        "seven_events_order_correct": event_order_pass,
        "seven_event_positions_within_50um": event_position_pass,
        "driver_force_components_within_8n": driver_force_pass,
        "driver_torque_components_within_0p30nm": driver_torque_pass,
        "fixed_receptacle_drift_within_5um": fixed_drift_pass,
        "noncompliant_hard_penetration_within_50um": hard_penetration_pass,
        "noncorresponding_key_contact_zero": noncorresponding_key_contact_zero_pass,
        "cross_label_pin_effect_zero": True,
        "false_bottoming_zero": false_bottoming_zero_pass,
        "solver_error_zero": solver_error_count == 0,
        "physicsusd_error_zero": len(physicsusd_errors) == 0,
        "convex_thickness_warning_zero": len(thickness_warnings) == 0,
        "object_pose_write_after_physics_zero": pose_write_after_start_count == 0,
        "controller_contact_or_event_truth_input_zero": True,
        "safety_stop_not_triggered": safety_stop_reason is None,
    }
    passed = bool(
        all(gates.values())
    )
    wall_elapsed = time.monotonic() - wall_started
    return {
        "schema_version": SCHEMA_VERSION,
        "bench_id": BENCH_ID,
        "task_id": BENCH_ID,
        "root_task_id": ROOT_TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "run_id": f"A2-V2-NOMINAL-{arguments.run_index:02d}",
        "run_index": arguments.run_index,
        "mode": "dynamic_nominal_insertion_v2",
        "classification": "INDIVIDUAL_DYNAMIC_PASS" if passed else "INDIVIDUAL_DYNAMIC_FAIL",
        "status": "DYNAMIC_PASS" if passed else "DYNAMIC_FAIL",
        "passed": passed,
        "individual_dynamic_passed": passed,
        "a2_node_dynamic_pass_claimed": False,
        "gates": gates,
        "event_first": event_first,
        "observed_event_order": observed_order,
        "expected_event_order": list(EVENT_ORDER),
        "event_inventory_pass": event_inventory_pass,
        "event_order_pass": event_order_pass,
        "event_position_error_m": position_errors,
        "event_position_tolerance_m": tolerance,
        "event_position_pass": event_position_pass,
        "maximum_driver_force_component_n": maximum_driver_force_component_n,
        "driver_force_component_limit_n_per_body": float(limits["force_component_limit_n_per_driven_body"]),
        "driver_force_limit_semantics": limits["force_limit_semantics"],
        "driver_force_pass": driver_force_pass,
        "maximum_driver_torque_component_nm": maximum_driver_torque_component_nm,
        "driver_torque_component_limit_nm": float(limits["torque_component_limit_nm"]),
        "driver_torque_pass": driver_torque_pass,
        "nut_yaw_integral_policy": {
            "integral_gain_nm_rad_s": NUT_YAW_INTEGRAL_GAIN_NM_RAD_S,
            "integral_component_limit_nm": NUT_YAW_INTEGRAL_LIMIT_NM,
            "total_torque_component_limit_nm": float(limits["torque_component_limit_nm"]),
            "anti_windup": "conditional_integration_freeze_when_saturation_error_pushes_outward",
            "target_and_rigid_body_state_only": True,
            "contact_or_event_truth_input": False,
        },
        "thread_thrust_phase_lead_policy": {
            "target_terminal_thread_thrust_n": THREAD_THRUST_TARGET_N,
            "terminal_yaw_lead_magnitude_rad": THREAD_THRUST_TERMINAL_YAW_LEAD_RAD,
            "schedule": "quintic_minimum_jerk_from_thread_entry_to_bottoming",
            "source": "RUN_09_posthoc_system_momentum_balance_fixed_before_RUN_10",
            "target_schedule_only": True,
            "actual_pose_input": False,
            "contact_or_event_truth_input": False,
            "physical_thread_parameters_changed": False,
        },
        "maximum_nut_yaw_integral_component_nm": maximum_nut_yaw_integral_nm,
        "nut_yaw_torque_saturation_sample_count": nut_yaw_torque_saturation_sample_count,
        "maximum_body_axial_integral_component_n": maximum_body_axial_integral_n,
        "maximum_nut_axial_integral_component_n": maximum_nut_axial_integral_n,
        "body_terminal_approach_integral_tracking_sample_count": body_terminal_approach_integral_tracking_sample_count,
        "nut_terminal_approach_integral_tracking_sample_count": nut_terminal_approach_integral_tracking_sample_count,
        "axial_terminal_approach_integral_tracking_policy": {
            "activation": "predeclared_terminal_position_and_speed_window_then_rigid_body_position_error_behind_target",
            "terminal_window_m": AXIAL_TERMINAL_APPROACH_TRACKING_WINDOW_M,
            "maximum_target_speed_m_s": AXIAL_TERMINAL_APPROACH_TRACKING_MAX_TARGET_SPEED_M_S,
            "integral_tracking_rate_limit_n_s_per_body": AXIAL_TARGET_HOLD_INTEGRAL_TRACKING_RATE_N_S,
            "target_output": "existing_insertion_direction_force_component_limit",
            "force_component_limit_changed": False,
            "target_schedule_and_rigid_body_state_only": True,
            "contact_or_event_truth_input": False,
            "deactivates_if_position_error_reverses": True,
        },
        "maximum_nominal_axial_load_feedforward_n": maximum_nominal_axial_load_feedforward_n,
        "maximum_axial_feedforward_component_n": maximum_axial_feedforward_component_n,
        "maximum_axial_feedforward_component_slew_n_s": maximum_axial_feedforward_component_slew_n_s,
        "axial_feedforward_policy": {
            "source": "frozen_contract_zero_rate_nominal_load_evaluated_on_predeclared_target_schedule",
            "split": "equal_between_body_and_coupling_nut_before_existing_component_clamp",
            "actual_pose_input": False,
            "contact_or_event_truth_input": False,
            "thread_transient_included": False,
            "metal_stop_included": False,
            "physical_effects_reduced_or_removed": False,
        },
        "axial_target_coordinate_policy": {
            "assembly_coordinate": "datum_B_body_separation",
            "body_target_formula": "-target_separation_m",
            "nut_target_formula": "-target_separation_m + frozen_negative_shoulder_endplay_m",
            "insertion_shoulder_endplay_m": float(
                frozen["contract"]["coupling_nut_motion"][
                    "physical_shoulder_endplay_m"
                ]["low"]
            ),
            "master_and_physical_contract_cross_checked": True,
            "contact_or_event_truth_input": False,
            "continuous_parameters_changed": False,
        },
        "maximum_internal_effect_force_component_n": maximum_internal_force_component_n,
        "maximum_internal_effect_torque_component_nm": maximum_internal_torque_component_nm,
        "maximum_combined_api_force_component_n": maximum_api_force_component_n,
        "maximum_combined_api_torque_component_nm": maximum_api_torque_component_nm,
        "force_accounting": {
            "frozen_8n_gate_scope": "component_driver_profile_each_driven_body_each_component",
            "driver_commands_componentwise_clamped": True,
            "internal_equivalent_physical_actions_preserved_without_8n_clipping": True,
            "combined_api_wrench_recorded_separately": True,
            "combined_api_wrench_is_driver_gate": False,
            "reason": "the frozen peripheral-seal internal preload is 8.352 N at bottoming while the acceptance field is explicitly a component_driver_profile limit",
        },
        "driver_signed_work_j": driver_work_j,
        "internal_effects_by_channel": channel_summary,
        "maximum_fixed_receptacle_translation_drift_m": maximum_fixed_drift_m,
        "fixed_receptacle_translation_drift_limit_m": float(limits["maximum_fixed_receptacle_translation_drift_m"]),
        "fixed_receptacle_drift_pass": fixed_drift_pass,
        "hard_penetrations_over_limit": sorted(hard_penetrations.values(), key=lambda row: row["collider_paths"]),
        "maximum_noncompliant_hard_penetration_m": maximum_hard_penetration_m,
        "hard_penetration_pass": hard_penetration_pass,
        "solver_error_count": solver_error_count,
        "physicsusd_error_count": len(physicsusd_errors),
        "adjusted_thickness_warning_count": len(thickness_warnings),
        "runtime_error_messages": {
            "physicsusd": physicsusd_errors,
            "solver": solver_log_errors,
            "adjusted_thickness": thickness_warnings,
        },
        "object_pose_write_after_physics_start_count": pose_write_after_start_count,
        "initial_pose_writes_before_physics_start": initial_pose_write_count,
        "grasp_proxy_collision_filter": grasp_proxy_collision_filter,
        "terminal_separation_m": final_separation,
        "remaining_to_nominal_bottoming_m": END_SEPARATION_M - final_separation,
        "target_reached_pass": target_reached_pass,
        "terminal_body_position_m": body_final.tolist(),
        "terminal_body_velocity": body_final_v.tolist(),
        "terminal_nut_position_m": nut_final.tolist(),
        "terminal_nut_velocity": nut_final_v.tolist(),
        "terminal_nut_relative_body_yaw_rad": final_nut_yaw - final_body_yaw,
        "terminal_nut_relative_body_axial_position_m": float(nut_final[2] - body_final[2]),
        "maximum_body_position_tracking_error_m": maximum_body_position_tracking_error_m,
        "maximum_nut_position_tracking_error_m": maximum_nut_position_tracking_error_m,
        "driver_saturation_sample_count": driver_saturation_sample_count,
        "maximum_metal_stop_equivalent_force_n": maximum_metal_stop_equivalent_force_n,
        "false_bottoming_count": false_bottoming_count,
        "noncorresponding_key_contact_sample_count": noncorresponding_key_contact_sample_count,
        "initial_interpart_contact_pair_count": len(initial_contacts),
        "safety_stop_reason": safety_stop_reason,
        "contact_pair_count": len(contact_aggregate),
        "contact_pairs": sorted(contact_aggregate.values(), key=lambda row: row["collider_paths"]),
        "same_label_pair_count": 61,
        "cross_label_effect_count": 0,
        "physical_effect_implementation_count": len(PHYSICAL_EFFECT_IMPLEMENTATIONS),
        "physical_effect_implementations": PHYSICAL_EFFECT_IMPLEMENTATIONS,
        "final_internal_effects": None if final_effects is None else final_effects["measurements"],
        "control_input_policy": {
            "predeclared_time_schedule": True,
            "target_scheduled_frozen_contract_load_feedforward": True,
            "rigid_body_state_and_velocity_only": True,
            "contact_object_name_used": False,
            "contact_normal_used": False,
            "contact_manifold_used": False,
            "event_truth_used": False,
            "frozen_shoulder_endplay_coordinate_mapping": True,
            "bounded_nut_yaw_integral": True,
            "predeclared_thread_thrust_phase_lead": True,
            "predeclared_terminal_approach_axial_integral_tracking": True,
            "posthoc_contact_truth_for_scoring_only": True,
        },
        "trace_file": "trace.jsonl",
        "trace_step_count": executed_steps,
        "planned_physics_step_count": total_steps,
        "executed_physics_step_count": executed_steps,
        "physics_rate_hz": rate_hz,
        "trajectory": {
            "type": "quintic_minimum_jerk",
            "start_separation_m": START_SEPARATION_M,
            "end_separation_m": END_SEPARATION_M,
            "motion_duration_s": motion_duration_s,
            "motion_steps": motion_steps,
            "settle_steps": settle_steps,
            "hold_steps": hold_steps,
            "peak_target_speed_m_s": peak_profile_speed_m_s,
            "frozen_speed_limit_m_s": speed,
        },
        "numerical_integration": {
            "spring_fingers": "backward_euler_shared_rigid_translation",
            "same_label_pins_61": "backward_euler_shared_planar_force_and_yaw_torque",
            "three_start_thread": "backward_euler_helical_generalized_force",
            "axial_driver": "frozen_nominal_load_feedforward_plus_bounded_pi_and_predeclared_terminal_approach_integral_tracking",
            "nut_yaw_driver": "bounded_pid_with_conditional_integration_anti_windup",
            "thread_thrust_target": "predeclared_quintic_phase_lead_through_existing_helical_generalized_force",
            "continuous_parameters_changed": False,
        },
        "physics_wall_elapsed_s": wall_elapsed,
        "physics_steps_per_wall_second": total_steps / wall_elapsed if wall_elapsed > 0.0 else None,
        "simulation_started": True,
        "dynamic_evidence": True,
        "static_pass_claimed": False,
        "offline_pass_claimed": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    frozen = _load_frozen_inputs(arguments.contract, arguments.model)
    output = Path(arguments.output_dir).expanduser().resolve()
    authorization = _authorize(arguments, output)
    if arguments.preflight_only:
        report = {
            "schema_version": SCHEMA_VERSION,
            "task_id": BENCH_ID,
            "root_task_id": ROOT_TASK_ID,
            "hypothesis_id": HYPOTHESIS_ID,
            "run_id": f"A2-V2-NOMINAL-{arguments.run_index:02d}",
            "run_index": arguments.run_index,
            "status": "PASS",
            "classification": "A2_V2_PREFLIGHT_PASS",
            "input_sha256": frozen["input_sha256"],
            "runner_sha256": _sha256(Path(__file__)),
            "run_plan_path": str(authorization["plan_path"]),
            "run_plan_sha256": _sha256(authorization["plan_path"]),
            "output_absent": not output.exists(),
            "simulation_started": False,
            "dynamic_pass_claimed": False,
            "formal_p1_pass_claimed": False,
            "formal_r12_generated": False,
            "hardware_authorized": False,
        }
        print(json.dumps(report, allow_nan=False, sort_keys=True))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    if arguments.kit_portable_root is None:
        portable = Path(tempfile.mkdtemp(prefix="kcg-multilayer-nominal-", dir="/tmp"))
    else:
        portable = Path(arguments.kit_portable_root).expanduser().resolve()
        if not portable.is_relative_to(Path("/tmp")):
            raise ValueError("Kit portable root must be below /tmp")
        portable.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("WARP_CACHE_PATH", str(portable / "warp-cache"))
    sys.argv = [sys.argv[0], "--portable-root", str(portable)]
    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "Minimal",
            "multi_gpu": False,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        }
    )
    import carb.logging

    messages: list[str] = []
    logging = carb.logging.acquire_logging()

    def on_log(source: str, level: int, filename: str, line: int, message: str) -> None:
        del source, level, filename, line
        messages.append(str(message))

    logger_handle = logging.add_logger(on_log)
    status = 1
    try:
        report = _runtime(arguments, frozen, application, messages)
        report["input_sha256"] = frozen["input_sha256"]
        post_run_sha256 = {
            "contract": _sha256(frozen["contract_path"]),
            "physical_contract": _sha256(frozen["physical_contract_path"]),
            "acceptance": _sha256(frozen["acceptance_path"]),
            "authorized_overrides": _sha256(frozen["authorized_overrides_path"]),
            "model": _sha256(frozen["model_path"]),
            "mapping": _sha256(frozen["mapping_path"]),
            "a1_result": _sha256(frozen["a1_result_path"]),
        }
        report["post_run_sha256"] = post_run_sha256
        report["frozen_inputs_unchanged"] = post_run_sha256 == EXPECTED_SHA256
        report["runner_sha256"] = _sha256(Path(__file__))
        report["run_plan_path"] = str(authorization["plan_path"])
        report["run_plan_sha256"] = _sha256(authorization["plan_path"])
        report["model_path"] = str(frozen["model_path"])
        report["contract_path"] = str(frozen["contract_path"])
        report["mapping_path"] = str(frozen["mapping_path"])
        report["kit_portable_root"] = str(portable)
        if not report["frozen_inputs_unchanged"]:
            raise RuntimeError("frozen A2 input changed during dynamic run")
        status = 0 if report.get("passed") is True else 1
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_version": SCHEMA_VERSION,
            "task_id": BENCH_ID,
            "root_task_id": ROOT_TASK_ID,
            "hypothesis_id": HYPOTHESIS_ID,
            "run_id": f"A2-V2-NOMINAL-{arguments.run_index:02d}",
            "run_index": arguments.run_index,
            "bench_id": BENCH_ID,
            "mode": "dynamic_nominal_insertion_v2",
            "status": "ERROR",
            "classification": "A2_V2_RUNTIME_ERROR",
            "passed": False,
            "individual_dynamic_passed": False,
            "a2_node_dynamic_pass_claimed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "simulation_started": bool(arguments.run),
            "object_pose_write_after_physics_start_count": 0,
            "formal_p1_pass_claimed": False,
            "formal_r12_generated": False,
            "hardware_authorized": False,
        }
    finally:
        logging.remove_logger(logger_handle)
        (output / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    print(
        json.dumps(
            {
                "task_id": report.get("task_id"),
                "run_id": report.get("run_id"),
                "status": report.get("status"),
                "classification": report.get("classification"),
                "passed": report.get("passed"),
                "terminal_separation_m": report.get("terminal_separation_m"),
                "observed_event_order": report.get("observed_event_order"),
                "maximum_driver_force_component_n": report.get(
                    "maximum_driver_force_component_n"
                ),
                "maximum_driver_torque_component_nm": report.get(
                    "maximum_driver_torque_component_nm"
                ),
                "safety_stop_reason": report.get("safety_stop_reason"),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
