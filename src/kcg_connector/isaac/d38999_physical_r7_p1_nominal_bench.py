#!/usr/bin/env python3

"""Run the physical-r11 P1 nominal component bench without pose writes.

The loose plug is driven only by bounded external force/torque servos whose
targets are fixed before physics starts.  Contact/collider truth is sampled
after each step for scoring and never changes the command.  Evidence uses
semantic paths and raw numeric traces; no file fingerprint is computed.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "kcg_d38999_physical_r11_p1_raw_v1"
GENERATOR_ID = "kcg_d38999_physical_r11_p1_force_probe_v1"
PASS_BANNER = "ISAAC PHYSICAL R11 P1 NOMINAL PASSED"
FAIL_BANNER = "ISAAC PHYSICAL R11 P1 NOMINAL FAILED"
EVENT_ORDER = (
    "five_key_polarization",
    "three_start_thread_entry",
    "spring_finger_engagement",
    "first_pin_socket_spring_touch",
    "pin_barrier_seal_contact",
    "seal_compression",
    "shell_to_shell_metal_bottoming",
)
DIAGNOSTIC_DRIVER_KINDS = (
    "bounded_axial_integral",
    "bounded_axial_force_feedforward",
)
TASK_R12_006C_ID = "TASK-R12-006C"
TASK_R12_006C_BUILD_RESULT_REL = Path(
    "artifacts/agent_control/tasks/TASK-R12-006B/candidate/"
    "CANDIDATE_BUILD_RESULT.json"
)
TASK_R12_006C_SCENE_REL = TASK_R12_006C_BUILD_RESULT_REL.parent / "scene.yaml"
TASK_R12_006C_OUTPUT_REL = Path(
    "artifacts/agent_control/tasks/TASK-R12-006C/DIAGNOSTIC_H2_LOCAL"
)
TASK_R12_006C_BUILD_RESULT_SHA256 = (
    "4551c1a9900b421e85c53e7dabd9821cb8349ada80fb5ab95a1522951b79febb"
)
TASK_R12_006C_CANDIDATE_ASSET_SHA256 = (
    "d41477ee18052662904212444b907607874a8c6c27399d3d344e44ee4fd18d67"
)
TASK_R12_006C_KIT_PORTABLE_ROOT = Path("/tmp/task-r12-006c-h2-local")
INITIALIZATION_POSITION_TOLERANCE_M = 5.0e-5
INITIALIZATION_QUATERNION_TOLERANCE = 1.0e-3
INITIALIZATION_MAX_ABS_VELOCITY = 10.0
PREMOTION_MOVING_BODY_POSITION_TOLERANCE_M = 5.0e-5
PREMOTION_MOVING_BODY_QUATERNION_TOLERANCE = 1.0e-3
PREMOTION_MOVING_BODY_MAX_ABS_VELOCITY = 2.0e-2
PREMOTION_FIXED_POSITION_TOLERANCE_M = 5.0e-6
PREMOTION_FIXED_QUATERNION_TOLERANCE = 1.0e-5
PREMOTION_FIXED_MAX_ABS_VELOCITY = 1.0e-4
CONTACT_EVENT_FAMILIES = {
    frozenset(("thread_rails_3", "thread_followers_3")): (
        "three_start_thread_entry"
    ),
    frozenset(("spring_fingers_12", "receptacle_bore_targets_12")): (
        "spring_finger_engagement"
    ),
    frozenset(("pins_61", "socket_petals_366")): (
        "first_pin_socket_spring_touch"
    ),
    frozenset(("pin_barriers_61", "hard_socket_entries_61")): (
        "pin_barrier_seal_contact"
    ),
    frozenset(("seal_segments_24", "seal_targets_24")): (
        "seal_compression"
    ),
    frozenset(("fixed_metal_stop_48", "plug_metal_stop_48")): (
        "shell_to_shell_metal_bottoming"
    ),
}


def _task_r12_006c_local_h2_authorized(
    arguments: argparse.Namespace,
    repository: Path,
) -> bool:
    current_task_path = repository / "artifacts/agent_control/CURRENT_TASK.md"
    master_state_path = repository / "artifacts/agent_control/MASTER_STATE.json"
    try:
        current_task_first_line = current_task_path.read_text(
            encoding="utf-8"
        ).splitlines()[0]
        master_state = json.loads(master_state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, IndexError, json.JSONDecodeError, OSError):
        return False
    authorization = master_state.get("diagnostic_authorization")
    if not isinstance(authorization, Mapping):
        return False
    exact_paths = (
        Path(arguments.authorized_local_candidate_result).expanduser().resolve()
        == (repository / TASK_R12_006C_BUILD_RESULT_REL).resolve()
        and Path(arguments.scene_config).expanduser().resolve()
        == (repository / TASK_R12_006C_SCENE_REL).resolve()
        and Path(arguments.output_dir).expanduser().resolve()
        == (repository / TASK_R12_006C_OUTPUT_REL).resolve()
        and Path(arguments.kit_portable_root).expanduser().resolve()
        == TASK_R12_006C_KIT_PORTABLE_ROOT
    )
    exact_driver = (
        arguments.diagnostic_driver_kind
        == "bounded_axial_force_feedforward"
        and math.isclose(
            arguments.diagnostic_feedforward_component_n,
            6.5,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            arguments.diagnostic_feedforward_start_separation_m,
            0.01075,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            arguments.diagnostic_feedforward_ramp_m,
            0.002,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    exact_control_state = (
        current_task_first_line == f"# 当前任务：{TASK_R12_006C_ID}"
        and master_state.get("task_id") == TASK_R12_006C_ID
        and master_state.get("status") == "RUNNING"
        and authorization.get("authorized") is True
        and authorization.get("diagnostic_only") is True
        and authorization.get("formal_p1_pass_claimed") is False
        and authorization.get("driver_kind")
        == "bounded_axial_force_feedforward"
        and authorization.get("run_limit") == 1
        and authorization.get("runs_completed") == 0
    )
    return bool(
        arguments.run
        and arguments.candidate_index == 2
        and arguments.authorized_local_candidate_result_sha256
        == TASK_R12_006C_BUILD_RESULT_SHA256
        and Path(arguments.model_contract).name
        == "d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
        and Path(arguments.acceptance_config).name
        == "d38999_keyed_v3_physical_acceptance_r12_v1.yaml"
        and exact_paths
        and exact_driver
        and exact_control_state
    )


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scene-config",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_keyed_v2_tabletop_scene_v1.yaml"
        ),
    )
    parser.add_argument(
        "--model-contract",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_keyed_v2_physical_model_contract_v1.yaml"
        ),
    )
    parser.add_argument(
        "--acceptance-config",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_keyed_v2_physical_acceptance_v1.yaml"
        ),
    )
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--authorized-local-candidate-result", default=None)
    parser.add_argument("--authorized-local-candidate-result-sha256", default=None)
    parser.add_argument("--kit-portable-root", required=True)
    parser.add_argument("--start-separation-m", type=float, default=0.00550)
    parser.add_argument("--end-separation-m", type=float, default=0.01505)
    parser.add_argument("--axial-speed-m-s", type=float, default=0.00050)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--hold-steps", type=int, default=240)
    parser.add_argument(
        "--diagnostic-driver-kind",
        choices=DIAGNOSTIC_DRIVER_KINDS,
        default=None,
        help=(
            "TASK-R12-005 or the exact TASK-R12-006C replay only; "
            "never produces a formal P1 pass"
        ),
    )
    parser.add_argument(
        "--diagnostic-integral-gain-n-m-s",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--diagnostic-integral-component-limit-n",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--diagnostic-feedforward-component-n",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--diagnostic-feedforward-start-separation-m",
        type=float,
        default=0.01075,
    )
    parser.add_argument(
        "--diagnostic-feedforward-ramp-m",
        type=float,
        default=0.001,
    )
    result = parser.parse_args(argv)
    local_fields = (
        result.authorized_local_candidate_result,
        result.authorized_local_candidate_result_sha256,
    )
    if any(local_fields) and not all(local_fields):
        parser.error("local candidate result path and SHA-256 must be supplied together")
    task_r12_006c_local_h2_authorized = bool(
        all(local_fields)
        and result.diagnostic_driver_kind is not None
        and _task_r12_006c_local_h2_authorized(result, repository)
    )
    result.task_r12_006c_local_h2_authorized = (
        task_r12_006c_local_h2_authorized
    )
    if all(local_fields) and (
        result.candidate_index != 2
        or Path(result.model_contract).name
        != "d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
        or Path(result.acceptance_config).name
        != "d38999_keyed_v3_physical_acceptance_r12_v1.yaml"
        or (
            result.diagnostic_driver_kind is not None
            and not task_r12_006c_local_h2_authorized
        )
    ):
        parser.error(
            "the pinned local candidate diagnostic is authorized only for the "
            "exact TASK-R12-006C H2 replay"
        )
    if not result.run:
        parser.error("P1 physics execution requires --run")
    if not (
        math.isfinite(result.start_separation_m)
        and math.isfinite(result.end_separation_m)
        and 0.0 < result.start_separation_m < result.end_separation_m
    ):
        parser.error("separation interval must be finite, positive, and ordered")
    if not math.isclose(result.start_separation_m, 0.00550, abs_tol=1.0e-12):
        parser.error("P1 start separation is frozen at 5.50 mm")
    if not math.isclose(result.end_separation_m, 0.01505, abs_tol=1.0e-12):
        parser.error("P1 end separation is frozen at 15.05 mm")
    if not math.isclose(result.axial_speed_m_s, 0.00050, abs_tol=1.0e-12):
        parser.error("P1 axial speed is frozen at 0.5 mm/s")
    if result.settle_steps != 120 or result.hold_steps != 240:
        parser.error("P1 settle/hold step counts changed")
    if not (
        math.isfinite(result.diagnostic_integral_gain_n_m_s)
        and 0.0 < result.diagnostic_integral_gain_n_m_s <= 1000.0
    ):
        parser.error("diagnostic integral gain must be in (0, 1000] N/(m*s)")
    if not (
        math.isfinite(result.diagnostic_integral_component_limit_n)
        and 0.0 < result.diagnostic_integral_component_limit_n <= 4.0
    ):
        parser.error("diagnostic integral component limit must be in (0, 4] N")
    if not (
        math.isfinite(result.diagnostic_feedforward_component_n)
        and 0.0 < result.diagnostic_feedforward_component_n <= 8.0
    ):
        parser.error("diagnostic feedforward component must be in (0, 8] N")
    if not (
        result.start_separation_m
        <= result.diagnostic_feedforward_start_separation_m
        <= result.end_separation_m
    ):
        parser.error("diagnostic feedforward start separation is outside P1 travel")
    if not (
        math.isfinite(result.diagnostic_feedforward_ramp_m)
        and 0.0 < result.diagnostic_feedforward_ramp_m <= 0.004
    ):
        parser.error("diagnostic feedforward ramp must be in (0, 0.004] m")
    return result


def _emit(value: Any) -> None:
    os.write(1, (str(value) + "\n").encode("utf-8"))


def _finite_vector(value: Any, expected_size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (expected_size,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"{label} must be a finite {expected_size}-vector")
    return result


def _quat_to_rpy_wxyz(value: Any) -> tuple[float, float, float]:
    w, x, y, z = _finite_vector(value, 4, "orientation quaternion")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise RuntimeError("orientation quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _quaternion_error_wxyz(value: Any, expected: Sequence[float]) -> float:
    actual = _finite_vector(value, 4, "orientation quaternion")
    target = _finite_vector(expected, 4, "expected orientation quaternion")
    actual_norm = float(np.linalg.norm(actual))
    target_norm = float(np.linalg.norm(target))
    if actual_norm <= 1.0e-12 or target_norm <= 1.0e-12:
        raise RuntimeError("orientation quaternion has zero norm")
    actual = actual / actual_norm
    target = target / target_norm
    return float(
        min(
            np.max(np.abs(actual - target)),
            np.max(np.abs(actual + target)),
        )
    )


def _unwrap(previous_wrapped: float, previous_unwrapped: float, current: float) -> float:
    delta = (current - previous_wrapped + math.pi) % (2.0 * math.pi) - math.pi
    return previous_unwrapped + delta


def _clamp_vector(value: np.ndarray, component_limit: float) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float64), -component_limit, component_limit)


def _set_existing_transform(
    stage: Any,
    path: str,
    translation: Sequence[float],
    rotation_xyz_deg: Sequence[float],
    usd_geom: Any,
    gf: Any,
) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"missing transform prim: {path}")
    operations = usd_geom.Xformable(prim).GetOrderedXformOps()
    if len(operations) != 2:
        raise RuntimeError(f"unexpected transform stack at {path}")
    if operations[0].GetOpType() != usd_geom.XformOp.TypeTranslate:
        raise RuntimeError(f"first transform operation is not translate at {path}")
    if operations[1].GetOpType() != usd_geom.XformOp.TypeRotateXYZ:
        raise RuntimeError(f"second transform operation is not rotateXYZ at {path}")
    operations[0].Set(gf.Vec3d(*translation))
    operations[1].Set(gf.Vec3f(*rotation_xyz_deg))


def _family_for_collider(stage: Any, path: str) -> str | None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return None
    attribute = prim.GetAttribute("kcg:primitiveFamily")
    value = attribute.Get() if attribute else None
    return None if value is None else str(value)


def _contact_rows(stage: Any, interface: Any, schema_tools: Any) -> list[dict[str, Any]]:
    headers, contacts, friction = interface.get_full_contact_report()
    rows: list[dict[str, Any]] = []
    for header in headers:
        actor_paths = (
            str(schema_tools.intToSdfPath(header.actor0)),
            str(schema_tools.intToSdfPath(header.actor1)),
        )
        collider_paths = (
            str(schema_tools.intToSdfPath(header.collider0)),
            str(schema_tools.intToSdfPath(header.collider1)),
        )
        families = tuple(_family_for_collider(stage, path) for path in collider_paths)
        event = CONTACT_EVENT_FAMILIES.get(frozenset(families))
        start = int(header.contact_data_offset)
        stop = start + int(header.num_contact_data)
        separations = [float(contacts[index].separation) for index in range(start, stop)]
        impulses = [
            math.sqrt(
                sum(float(value) ** 2 for value in contacts[index].impulse)
            )
            for index in range(start, stop)
        ]
        contact_points = [
            {
                "position_m": [float(value) for value in contacts[index].position],
                "normal": [float(value) for value in contacts[index].normal],
                "separation_m": float(contacts[index].separation),
                "impulse_ns": [float(value) for value in contacts[index].impulse],
            }
            for index in range(start, stop)
        ]
        friction_start = int(getattr(header, "friction_anchors_offset", 0))
        friction_stop = friction_start + int(
            getattr(header, "num_friction_anchors_data", 0)
        )
        friction_impulses = [
            [float(value) for value in friction[index].impulse]
            for index in range(friction_start, friction_stop)
        ]
        rows.append(
            {
                "event": event,
                "actor_paths": list(actor_paths),
                "families": list(families),
                "collider_paths": list(collider_paths),
                "contact_record_count": int(header.num_contact_data),
                "minimum_separation_m": min(separations) if separations else None,
                "maximum_impulse_norm": max(impulses) if impulses else 0.0,
                "contact_points": contact_points,
                "friction_anchor_impulses_ns": friction_impulses,
            }
        )
    return rows


def _jam_from_window(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if len(rows) != 10:
        return None
    if not all(abs(float(row["nut_torque_nm"][2])) >= 0.297 for row in rows):
        return None
    yaw_delta = abs(
        float(rows[-1]["nut_unwrapped_yaw_rad"])
        - float(rows[0]["nut_unwrapped_yaw_rad"])
    )
    axial_delta = abs(
        float(rows[-1]["observed_separation_m"])
        - float(rows[0]["observed_separation_m"])
    )
    if yaw_delta >= math.radians(0.5) or axial_delta >= 0.00002:
        return None
    return {
        "start_step": int(rows[0]["step"]),
        "start_time_s": float(rows[0]["time_s"]),
        "confirmation_end_step": int(rows[-1]["step"]),
        "confirmation_end_time_s": float(rows[-1]["time_s"]),
        "yaw_increment_deg_over_10_steps": math.degrees(yaw_delta),
        "axial_increment_m_over_10_steps": axial_delta,
        "nut_body_relative_yaw_rad": float(
            rows[0]["nut_unwrapped_yaw_rad"]
            - rows[0]["body_unwrapped_yaw_rad"]
        ),
        "nut_body_relative_z_m": float(
            rows[0]["nut_position_m"][2] - rows[0]["body_position_m"][2]
        ),
    }


def _write_first_jam_evidence(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    jam: Mapping[str, Any] | None,
    *,
    dt: float,
    axis_xy: Sequence[float],
) -> Mapping[str, Any]:
    window_path = output / "first_jam_window.jsonl"
    contacts_path = output / "first_jam_contacts.csv"
    analysis_path = output / "first_jam_analysis.json"
    fields = (
        "time_s",
        "collider0_path",
        "collider1_path",
        "family0",
        "family1",
        "contact_position_m",
        "contact_normal",
        "separation_m",
        "normal_impulse_ns",
        "friction_impulses_ns",
        "tau_z_impulse_nm_s",
    )
    family_torque_impulse: dict[str, float] = {}
    pair_scores: dict[str, dict[str, Any]] = {}
    if jam is None:
        window_path.write_text(
            json.dumps({"status": "NO_SUSTAINED_TORQUE_JAM"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        low = float(jam["start_time_s"]) - 0.5
        high = float(jam["start_time_s"]) + 0.5
        with window_path.open("x", encoding="utf-8") as stream:
            for record in records:
                if low <= float(record["trace"]["time_s"]) <= high:
                    stream.write(
                        json.dumps(record["trace"], allow_nan=False, sort_keys=True)
                        + "\n"
                    )
    with contacts_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        if jam is not None:
            start_step = int(jam["start_step"])
            end_step = int(jam["confirmation_end_step"])
            for record in records:
                trace = record["trace"]
                if not start_step <= int(trace["step"]) <= end_step:
                    continue
                for header in record["contacts"]:
                    families = [value or "UNAVAILABLE" for value in header["families"]]
                    family_key = " <-> ".join(sorted(families))
                    pair_key = " <-> ".join(header["collider_paths"])
                    for contact in header["contact_points"]:
                        position = contact["position_m"]
                        impulse = contact["impulse_ns"]
                        tau_impulse = (
                            (float(position[0]) - float(axis_xy[0])) * float(impulse[1])
                            - (float(position[1]) - float(axis_xy[1])) * float(impulse[0])
                        )
                        family_torque_impulse[family_key] = (
                            family_torque_impulse.get(family_key, 0.0) + tau_impulse
                        )
                        score = pair_scores.setdefault(
                            pair_key,
                            {
                                "collider_paths": list(header["collider_paths"]),
                                "families": families,
                                "abs_tau_z_impulse_nm_s": 0.0,
                                "normal_weighted_sum": [0.0, 0.0, 0.0],
                                "normal_weight": 0.0,
                            },
                        )
                        score["abs_tau_z_impulse_nm_s"] += abs(tau_impulse)
                        weight = math.sqrt(sum(float(value) ** 2 for value in impulse))
                        score["normal_weight"] += weight
                        for axis in range(3):
                            score["normal_weighted_sum"][axis] += (
                                weight * float(contact["normal"][axis])
                            )
                        writer.writerow(
                            {
                                "time_s": trace["time_s"],
                                "collider0_path": header["collider_paths"][0],
                                "collider1_path": header["collider_paths"][1],
                                "family0": families[0],
                                "family1": families[1],
                                "contact_position_m": json.dumps(position),
                                "contact_normal": json.dumps(contact["normal"]),
                                "separation_m": contact["separation_m"],
                                "normal_impulse_ns": json.dumps(impulse),
                                "friction_impulses_ns": json.dumps(
                                    header["friction_anchor_impulses_ns"]
                                ),
                                "tau_z_impulse_nm_s": tau_impulse,
                            }
                        )
    duration = 10.0 * dt
    family_torque = {
        key: value / duration for key, value in family_torque_impulse.items()
    }
    dominant = None
    if pair_scores:
        dominant = max(
            pair_scores.values(), key=lambda row: row["abs_tau_z_impulse_nm_s"]
        )
        weight = float(dominant.pop("normal_weight"))
        vector = dominant.pop("normal_weighted_sum")
        dominant["impulse_weighted_mean_normal"] = (
            [value / weight for value in vector] if weight > 0.0 else "UNAVAILABLE"
        )
    analysis = {
        "status": "JAM_DETECTED" if jam is not None else "NO_SUSTAINED_TORQUE_JAM",
        "jam": jam,
        "window_duration_s": duration if jam is not None else None,
        "equivalent_tau_z_by_family_pair_nm": family_torque,
        "dominant_collider_pair": dominant,
        "contact_truth_role": "posthoc_diagnostic_only_never_command_input",
    }
    analysis_path.write_text(
        json.dumps(analysis, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return analysis


def _run(arguments: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from omni.physx import get_physx_simulation_interface
    from omni.physx.scripts import physicsUtils
    import omni.usd
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, UsdGeom, UsdPhysics, UsdShade

    from kcg_connector.d38999_tabletop_scene import (
        author_d38999_tabletop_scene,
        load_d38999_tabletop_scene,
        verify_d38999_tabletop_asset,
    )
    from kcg_connector.d38999_keyed_v2_physical_model_contract import WORKSPACE_ROOT
    from validate_physical_r7_composed_scene import _run_validation
    from validate_physical_r11_cooked_geometry import _run as run_cooked_geometry_gate

    acceptance_path = Path(arguments.acceptance_config).expanduser().resolve()
    if acceptance_path.name == "d38999_keyed_v3_physical_acceptance_r12_v1.yaml":
        from kcg_connector.d38999_keyed_v3_physical_r12_acceptance import (
            NOMINAL_R7_EVENT_ORDER,
            load_r12_physical_acceptance_matrix,
        )

        matrix = load_r12_physical_acceptance_matrix(acceptance_path)
    else:
        from kcg_connector.d38999_keyed_v2_physical_acceptance import (
            NOMINAL_R7_EVENT_ORDER,
            load_physical_acceptance_matrix,
        )

        if arguments.candidate_index is not None:
            raise ValueError("candidate index is available only for r12")
        matrix = load_physical_acceptance_matrix(acceptance_path)
    p1_contract = matrix.document["benches"]["P1"]
    driver = dict(p1_contract["inputs"]["component_driver_profile"])
    settle_steps = int(driver["settle_steps"])
    hold_steps = int(driver["hold_steps"])
    if (
        arguments.settle_steps != settle_steps
        or arguments.hold_steps != hold_steps
    ):
        raise RuntimeError("P1 command-line settle/hold differs from frozen driver")
    if tuple(NOMINAL_R7_EVENT_ORDER) != EVENT_ORDER:
        raise RuntimeError("P1 event order differs between runner and threshold contract")

    local_authorization = None
    authorized_model = None
    authorized_local_asset_path = None
    if arguments.authorized_local_candidate_result is not None:
        from kcg_connector.d38999_keyed_v3_physical_r12_contract import (
            candidate_model,
            load_r12_physical_model_contract,
        )
        from kcg_connector.d38999_r12_local_candidate import (
            authorize_task_r12_006b_local_candidate,
        )

        frozen_model = candidate_model(
            load_r12_physical_model_contract(arguments.model_contract), 2
        )
        local_authorization = authorize_task_r12_006b_local_candidate(
            model=frozen_model,
            result_path=arguments.authorized_local_candidate_result,
            expected_result_sha256=(
                arguments.authorized_local_candidate_result_sha256
            ),
            scene_config=arguments.scene_config,
            repository_root=WORKSPACE_ROOT,
        )
        authorized_model = local_authorization.model
        authorized_local_asset_path = (
            local_authorization.candidate_asset_relative_path
        )
        if (
            arguments.task_r12_006c_local_h2_authorized
            and local_authorization.candidate_asset_sha256
            != TASK_R12_006C_CANDIDATE_ASSET_SHA256
        ):
            raise RuntimeError(
                "TASK-R12-006C local candidate asset SHA-256 changed"
            )

    a2_report = _run_validation(
        argparse.Namespace(
            scene_config=str(Path(arguments.scene_config).resolve()),
            model_contract=str(Path(arguments.model_contract).resolve()),
            candidate_index=arguments.candidate_index,
        ),
        authorized_model=authorized_model,
        authorized_local_asset_path=authorized_local_asset_path,
    )
    if a2_report.get("status") != "PASSED":
        raise RuntimeError("A2 composed-stage release did not pass in this process")
    cooked_geometry_report = run_cooked_geometry_gate(
        arguments.model_contract,
        arguments.candidate_index,
        authorized_model=authorized_model,
    )
    if cooked_geometry_report.get("status") != "PASSED":
        raise RuntimeError("A2 PhysX-cooked geometry release did not pass")

    config = load_d38999_tabletop_scene(
        arguments.scene_config,
        authorized_local_asset_path=authorized_local_asset_path,
    )
    asset_path = verify_d38999_tabletop_asset(
        config,
        WORKSPACE_ROOT,
        authorized_local_asset_path=authorized_local_asset_path,
        authorized_model=authorized_model,
    )
    dt = 1.0 / float(config.physics.rate_hz)

    World.clear_instance()
    omni.usd.get_context().new_stage()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=dt,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    authored = author_d38999_tabletop_scene(
        stage,
        config,
        asset_path,
        add_reference_to_stage=add_reference_to_stage,
        Gf=Gf,
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
        physics_utils=physicsUtils,
        authorized_local_asset_path=authorized_local_asset_path,
    )
    if authored["object_pose_writes_after_start"] != 0:
        raise RuntimeError("scene reports a post-start object pose write")

    fixed_origin = np.asarray(config.fixed_endpoint.receptacle_origin_m, dtype=np.float64)
    initial_plug_origin = fixed_origin + np.asarray(
        (0.0, 0.0, -float(arguments.start_separation_m)), dtype=np.float64
    )
    _set_existing_transform(
        stage,
        config.asset.loose_plug_prim_path,
        initial_plug_origin,
        (0.0, 0.0, 0.0),
        UsdGeom,
        Gf,
    )

    for owner_path in (
        config.asset.fixed_receptacle_prim_path,
        config.asset.body_prim_path,
        config.asset.nut_prim_path,
    ):
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(owner_path)
        )
        report_api.CreateThresholdAttr().Set(0.0)

    fixture = RigidPrim(
        prim_paths_expr=config.fixed_endpoint.fixture_prim_path,
        name="physical_r7_p1_fixture",
        reset_xform_properties=False,
    )
    fixed = RigidPrim(
        prim_paths_expr=config.asset.fixed_receptacle_prim_path,
        name="physical_r7_p1_fixed_receptacle",
        reset_xform_properties=False,
    )
    body = RigidPrim(
        prim_paths_expr=config.asset.body_prim_path,
        name="physical_r7_p1_body",
        reset_xform_properties=False,
    )
    nut = RigidPrim(
        prim_paths_expr=config.asset.nut_prim_path,
        name="physical_r7_p1_nut",
        reset_xform_properties=False,
    )
    if driver["gravity_set_before_world_reset"] is not True:
        raise RuntimeError("P1 gravity must be set before world reset")
    world.get_physics_context().set_gravity(
        float(driver["gravity_magnitude_m_s2"])
    )
    world.reset()
    for view in (fixture, fixed, body, nut):
        view.initialize()
    if not all(
        view.is_physics_handle_valid() for view in (fixture, fixed, body, nut)
    ):
        raise RuntimeError("P1 rigid-body tensor views are invalid")

    interface = get_physx_simulation_interface()
    event_first: dict[str, dict[str, Any]] = {}
    contact_pair_aggregate: dict[str, dict[str, Any]] = {}
    thread_start_paths: set[str] = set()
    solver_error_count = 0
    false_bottoming_count = 0
    external_work_j = 0.0
    pose_write_after_start_count = 0
    maximum_force_component_n = 0.0
    maximum_torque_component_nm = 0.0
    consecutive_torque_saturation_steps = 0
    maximum_consecutive_torque_saturation_steps = 0
    trace_jam_window: deque[Mapping[str, Any]] = deque(maxlen=10)
    diagnostic_history: deque[Mapping[str, Any]] = deque(maxlen=240)
    captured_jam_records: list[Mapping[str, Any]] = []
    first_jam: Mapping[str, Any] | None = None
    jam_capture_end_step: int | None = None
    maximum_fixed_translation_drift_m = 0.0
    maximum_fixture_translation_drift_m = 0.0

    body_wrapped_yaw = 0.0
    body_unwrapped_yaw = 0.0
    nut_wrapped_yaw = 0.0
    nut_unwrapped_yaw = 0.0
    body_integral_component_n = 0.0
    nut_integral_component_n = 0.0

    def state(view: Any, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions, orientations = view.get_world_poses()
        position = _finite_vector(positions[0], 3, f"{label} position")
        quaternion = _finite_vector(orientations[0], 4, f"{label} orientation")
        velocity = _finite_vector(view.get_velocities()[0], 6, f"{label} velocity")
        return position, quaternion, velocity

    def apply(view: Any, force: np.ndarray, torque: np.ndarray) -> None:
        view.apply_forces_and_torques_at_pos(
            forces=np.asarray([force], dtype=np.float32),
            torques=np.asarray([torque], dtype=np.float32),
            positions=None,
            is_global=True,
        )

    def accumulate_contact_rows(
        step: int, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        for row in rows:
            key = json.dumps(
                row["actor_paths"] + row["collider_paths"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            aggregate = contact_pair_aggregate.setdefault(
                key,
                {
                    "actor_paths": list(row["actor_paths"]),
                    "collider_paths": list(row["collider_paths"]),
                    "families": list(row["families"]),
                    "event": row["event"],
                    "first_step": step,
                    "last_step": step,
                    "active_step_count": 0,
                    "contact_record_count": 0,
                    "minimum_separation_m": None,
                    "maximum_impulse_norm": 0.0,
                },
            )
            aggregate["last_step"] = step
            aggregate["active_step_count"] += 1
            aggregate["contact_record_count"] += int(
                row["contact_record_count"]
            )
            separation = row["minimum_separation_m"]
            if separation is not None and (
                aggregate["minimum_separation_m"] is None
                or separation < aggregate["minimum_separation_m"]
            ):
                aggregate["minimum_separation_m"] = float(separation)
            aggregate["maximum_impulse_norm"] = max(
                float(aggregate["maximum_impulse_norm"]),
                float(row["maximum_impulse_norm"]),
            )

    def write_contact_audit() -> None:
        audit_path = output / "contact_audit.json"
        if audit_path.exists():
            return
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "kcg_d38999_physical_r12_contact_audit_v1"
                        if acceptance_path.name
                        == "d38999_keyed_v3_physical_acceptance_r12_v1.yaml"
                        else "kcg_d38999_physical_r11_contact_audit_v1"
                    ),
                    "role": "posthoc_diagnostic_only_never_command_input",
                    "pair_count": len(contact_pair_aggregate),
                    "pairs": sorted(
                        contact_pair_aggregate.values(),
                        key=lambda row: (
                            int(row["first_step"]),
                            row["collider_paths"],
                        ),
                    ),
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    initial_expectations = {
        "fixture": (
            fixture,
            np.asarray(config.fixed_endpoint.fixture_center_m, dtype=np.float64),
            (1.0, 0.0, 0.0, 0.0),
        ),
        "fixed_receptacle": (
            fixed,
            fixed_origin,
            (0.0, 1.0, 0.0, 0.0),
        ),
        "body": (
            body,
            initial_plug_origin,
            (1.0, 0.0, 0.0, 0.0),
        ),
        "nut": (
            nut,
            initial_plug_origin,
            (1.0, 0.0, 0.0, 0.0),
        ),
    }
    initial_state: dict[str, Any] = {}
    initial_state_pass = True
    for label, (view, expected_position, expected_quaternion) in (
        initial_expectations.items()
    ):
        position, quaternion, velocity = state(view, f"initial-{label}")
        position_error = float(np.max(np.abs(position - expected_position)))
        quaternion_error = _quaternion_error_wxyz(
            quaternion, expected_quaternion
        )
        velocity_error = float(np.max(np.abs(velocity)))
        row_pass = bool(
            position_error <= INITIALIZATION_POSITION_TOLERANCE_M
            and quaternion_error <= INITIALIZATION_QUATERNION_TOLERANCE
            and velocity_error <= INITIALIZATION_MAX_ABS_VELOCITY
        )
        initial_state[label] = {
            "position_m": position.tolist(),
            "expected_position_m": expected_position.tolist(),
            "orientation_wxyz": quaternion.tolist(),
            "expected_orientation_wxyz": list(expected_quaternion),
            "velocity": velocity.tolist(),
            "max_position_error_m": position_error,
            "max_quaternion_component_error": quaternion_error,
            "max_velocity_error": velocity_error,
            "passed": row_pass,
        }
        initial_state_pass = initial_state_pass and row_pass
    if not initial_state_pass:
        accumulate_contact_rows(
            0, _contact_rows(stage, interface, PhysicsSchemaTools)
        )
        write_contact_audit()
        raise RuntimeError(
            "P1 initial rigid state differs before the first commanded step: "
            + json.dumps(initial_state, allow_nan=False, sort_keys=True)
        )

    initial_state_thresholds = {
        "max_position_error_m": INITIALIZATION_POSITION_TOLERANCE_M,
        "max_quaternion_component_error": (
            INITIALIZATION_QUATERNION_TOLERANCE
        ),
        "max_abs_velocity": INITIALIZATION_MAX_ABS_VELOCITY,
    }

    trace_path = output / "trace.jsonl"
    trace_stream = trace_path.open("x", encoding="utf-8")

    total_motion_steps = int(
        math.ceil(
            (arguments.end_separation_m - arguments.start_separation_m)
            / (arguments.axial_speed_m_s * dt)
        )
    )
    total_steps = settle_steps + total_motion_steps + hold_steps
    entry_separation = float(
        p1_contract["pass"]["nominal_event_datum_B_separation_mm"]
        ["three_start_thread_entry"]
    ) / 1000.0
    lead_m = 0.00762
    angular_speed_rad_s = (
        -2.0 * math.pi * arguments.axial_speed_m_s / lead_m
    )
    previous_observed_separation = float(
        fixed_origin[2] - initial_state["body"]["position_m"][2]
    )
    premotion_state: dict[str, Any] | None = None
    premotion_state_pass = False

    try:
        for step_index in range(total_steps):
            body_position, body_quaternion, body_velocity = state(body, "body")
            nut_position, nut_quaternion, nut_velocity = state(nut, "nut")
            body_rpy = _quat_to_rpy_wxyz(body_quaternion)
            nut_rpy = _quat_to_rpy_wxyz(nut_quaternion)
            if step_index == 0:
                body_wrapped_yaw = body_rpy[2]
                body_unwrapped_yaw = body_rpy[2]
                nut_wrapped_yaw = nut_rpy[2]
                nut_unwrapped_yaw = nut_rpy[2]
            else:
                body_unwrapped_yaw = _unwrap(
                    body_wrapped_yaw, body_unwrapped_yaw, body_rpy[2]
                )
                nut_unwrapped_yaw = _unwrap(
                    nut_wrapped_yaw, nut_unwrapped_yaw, nut_rpy[2]
                )
                body_wrapped_yaw = body_rpy[2]
                nut_wrapped_yaw = nut_rpy[2]

            motion_index = max(0, step_index - settle_steps)
            commanded_motion_index = min(motion_index, total_motion_steps)
            target_separation = min(
                arguments.end_separation_m,
                arguments.start_separation_m
                + commanded_motion_index * dt * arguments.axial_speed_m_s,
            )
            moving = settle_steps <= step_index < (
                settle_steps + total_motion_steps
            )
            target_vz = -arguments.axial_speed_m_s if moving else 0.0
            target_position = fixed_origin + np.asarray(
                (0.0, 0.0, -target_separation), dtype=np.float64
            )
            target_relative_yaw = -2.0 * math.pi * max(
                0.0, target_separation - entry_separation
            ) / lead_m
            target_nut_yaw = target_relative_yaw
            target_nut_omega = angular_speed_rad_s if (
                moving and target_separation >= entry_separation
            ) else 0.0

            diagnostic_body_axial_n = 0.0
            diagnostic_nut_axial_n = 0.0
            if (
                arguments.diagnostic_driver_kind == "bounded_axial_integral"
                and step_index >= settle_steps
            ):
                integral_gain = float(
                    arguments.diagnostic_integral_gain_n_m_s
                )
                integral_limit = float(
                    arguments.diagnostic_integral_component_limit_n
                )
                body_integral_component_n = max(
                    -integral_limit,
                    min(
                        0.0,
                        body_integral_component_n
                        + integral_gain
                        * float(target_position[2] - body_position[2])
                        * dt,
                    ),
                )
                nut_integral_component_n = max(
                    -integral_limit,
                    min(
                        0.0,
                        nut_integral_component_n
                        + integral_gain
                        * float(target_position[2] - nut_position[2])
                        * dt,
                    ),
                )
                diagnostic_body_axial_n = body_integral_component_n
                diagnostic_nut_axial_n = nut_integral_component_n
            elif arguments.diagnostic_driver_kind == (
                "bounded_axial_force_feedforward"
            ):
                ramp_fraction = max(
                    0.0,
                    min(
                        1.0,
                        (
                            target_separation
                            - float(
                                arguments.diagnostic_feedforward_start_separation_m
                            )
                        )
                        / float(arguments.diagnostic_feedforward_ramp_m),
                    ),
                )
                feedforward = -float(
                    arguments.diagnostic_feedforward_component_n
                ) * ramp_fraction
                diagnostic_body_axial_n = feedforward
                diagnostic_nut_axial_n = feedforward

            body_diagnostic_force = np.asarray(
                (0.0, 0.0, diagnostic_body_axial_n), dtype=np.float64
            )
            nut_diagnostic_force = np.asarray(
                (0.0, 0.0, diagnostic_nut_axial_n), dtype=np.float64
            )

            body_force = _clamp_vector(
                float(driver["translation_position_gain_n_m"])
                * (target_position - body_position)
                + float(driver["translation_velocity_gain_n_s_m"])
                * (
                    np.asarray((0.0, 0.0, target_vz), dtype=np.float64)
                    - body_velocity[:3]
                )
                + body_diagnostic_force,
                float(driver["translation_force_component_limit_n"]),
            )
            nut_force = _clamp_vector(
                float(driver["translation_position_gain_n_m"])
                * (target_position - nut_position)
                + float(driver["translation_velocity_gain_n_s_m"])
                * (
                    np.asarray((0.0, 0.0, target_vz), dtype=np.float64)
                    - nut_velocity[:3]
                )
                + nut_diagnostic_force,
                float(driver["translation_force_component_limit_n"]),
            )
            body_torque = _clamp_vector(
                np.asarray(
                    (
                        -float(driver["roll_pitch_position_gain_nm_rad"])
                        * body_rpy[0]
                        - float(driver["angular_velocity_gain_nm_s_rad"])
                        * body_velocity[3],
                        -float(driver["roll_pitch_position_gain_nm_rad"])
                        * body_rpy[1]
                        - float(driver["angular_velocity_gain_nm_s_rad"])
                        * body_velocity[4],
                        float(driver["body_yaw_position_gain_nm_rad"])
                        * (float(driver["body_yaw_target_rad"]) - body_unwrapped_yaw)
                        + float(driver["angular_velocity_gain_nm_s_rad"])
                        * (0.0 - body_velocity[5]),
                    )
                ),
                float(driver["torque_component_limit_nm"]),
            )
            nut_torque = _clamp_vector(
                np.asarray(
                    (
                        -float(driver["roll_pitch_position_gain_nm_rad"])
                        * nut_rpy[0]
                        - float(driver["angular_velocity_gain_nm_s_rad"])
                        * nut_velocity[3],
                        -float(driver["roll_pitch_position_gain_nm_rad"])
                        * nut_rpy[1]
                        - float(driver["angular_velocity_gain_nm_s_rad"])
                        * nut_velocity[4],
                        float(driver["nut_yaw_position_gain_nm_rad"])
                        * (target_nut_yaw - nut_unwrapped_yaw)
                        + float(driver["angular_velocity_gain_nm_s_rad"])
                        * (target_nut_omega - nut_velocity[5]),
                    )
                ),
                float(driver["torque_component_limit_nm"]),
            )
            maximum_force_component_n = max(
                maximum_force_component_n,
                float(np.max(np.abs(body_force))),
                float(np.max(np.abs(nut_force))),
            )
            maximum_torque_component_nm = max(
                maximum_torque_component_nm,
                float(np.max(np.abs(body_torque))),
                float(np.max(np.abs(nut_torque))),
            )
            if abs(float(nut_torque[2])) >= 0.297:
                consecutive_torque_saturation_steps += 1
            else:
                consecutive_torque_saturation_steps = 0
            maximum_consecutive_torque_saturation_steps = max(
                maximum_consecutive_torque_saturation_steps,
                consecutive_torque_saturation_steps,
            )
            apply(body, body_force, body_torque)
            apply(nut, nut_force, nut_torque)
            external_work_j += dt * float(
                body_force @ body_velocity[:3]
                + body_torque @ body_velocity[3:]
                + nut_force @ nut_velocity[:3]
                + nut_torque @ nut_velocity[3:]
            )
            world.step(render=False)

            body_position_after, body_quaternion_after, body_velocity_after = state(
                body, "body-after-step"
            )
            nut_position_after, nut_quaternion_after, nut_velocity_after = state(
                nut, "nut-after-step"
            )
            fixed_position_after, fixed_quaternion_after, fixed_velocity_after = state(
                fixed, "fixed-after-step"
            )
            fixture_position_after, fixture_quaternion_after, fixture_velocity_after = state(
                fixture, "fixture-after-step"
            )
            fixed_drift = float(
                np.max(np.abs(fixed_position_after - fixed_origin))
            )
            fixture_drift = float(
                np.max(
                    np.abs(
                        fixture_position_after
                        - np.asarray(
                            config.fixed_endpoint.fixture_center_m,
                            dtype=np.float64,
                        )
                    )
                )
            )
            maximum_fixed_translation_drift_m = max(
                maximum_fixed_translation_drift_m, fixed_drift
            )
            maximum_fixture_translation_drift_m = max(
                maximum_fixture_translation_drift_m, fixture_drift
            )
            finite = bool(
                np.all(np.isfinite(body_position_after))
                and np.all(np.isfinite(body_quaternion_after))
                and np.all(np.isfinite(body_velocity_after))
                and np.all(np.isfinite(nut_position_after))
                and np.all(np.isfinite(nut_quaternion_after))
                and np.all(np.isfinite(nut_velocity_after))
            )
            if not finite:
                solver_error_count += 1
                raise RuntimeError("non-finite rigid-body state during P1")
            all_contact_rows = _contact_rows(
                stage, interface, PhysicsSchemaTools
            )
            accumulate_contact_rows(step_index + 1, all_contact_rows)
            if step_index == settle_steps - 1:
                premotion_state = {}
                premotion_state_pass = True
                post_settle_values = {
                    "fixture": state(fixture, "premotion-fixture"),
                    "fixed_receptacle": state(fixed, "premotion-fixed"),
                    "body": (
                        body_position_after,
                        body_quaternion_after,
                        body_velocity_after,
                    ),
                    "nut": (
                        nut_position_after,
                        nut_quaternion_after,
                        nut_velocity_after,
                    ),
                }
                for label, values in post_settle_values.items():
                    position, quaternion, velocity = values
                    _view, expected_position, expected_quaternion = (
                        initial_expectations[label]
                    )
                    position_error = float(
                        np.max(np.abs(position - expected_position))
                    )
                    quaternion_error = _quaternion_error_wxyz(
                        quaternion, expected_quaternion
                    )
                    velocity_error = float(np.max(np.abs(velocity)))
                    if label in ("fixture", "fixed_receptacle"):
                        position_limit = PREMOTION_FIXED_POSITION_TOLERANCE_M
                        quaternion_limit = PREMOTION_FIXED_QUATERNION_TOLERANCE
                        velocity_limit = PREMOTION_FIXED_MAX_ABS_VELOCITY
                    else:
                        position_limit = (
                            PREMOTION_MOVING_BODY_POSITION_TOLERANCE_M
                        )
                        quaternion_limit = (
                            PREMOTION_MOVING_BODY_QUATERNION_TOLERANCE
                        )
                        velocity_limit = (
                            PREMOTION_MOVING_BODY_MAX_ABS_VELOCITY
                        )
                    row_pass = bool(
                        position_error <= position_limit
                        and quaternion_error <= quaternion_limit
                        and velocity_error <= velocity_limit
                    )
                    premotion_state[label] = {
                        "position_m": position.tolist(),
                        "expected_position_m": expected_position.tolist(),
                        "orientation_wxyz": quaternion.tolist(),
                        "expected_orientation_wxyz": list(
                            expected_quaternion
                        ),
                        "velocity": velocity.tolist(),
                        "max_position_error_m": position_error,
                        "max_quaternion_component_error": quaternion_error,
                        "max_velocity_error": velocity_error,
                        "position_limit_m": position_limit,
                        "quaternion_limit": quaternion_limit,
                        "velocity_limit": velocity_limit,
                        "passed": row_pass,
                    }
                    premotion_state_pass = premotion_state_pass and row_pass
                if not premotion_state_pass:
                    write_contact_audit()
                    raise RuntimeError(
                        "P1 did not settle to a valid premotion state: "
                        + json.dumps(
                            premotion_state,
                            allow_nan=False,
                            sort_keys=True,
                        )
                    )
            observed_separation = float(fixed_origin[2] - body_position_after[2])
            if (
                "five_key_polarization" not in event_first
                and moving
                and previous_observed_separation < 0.00650
                <= observed_separation
            ):
                event_first["five_key_polarization"] = {
                    "step": step_index + 1,
                    "time_s": (step_index + 1) * dt,
                    "datum_B_separation_m": observed_separation,
                    "source": "collision_geometry_bounds_crossing",
                }
            previous_observed_separation = observed_separation

            active_contacts = [
                row for row in all_contact_rows if row["event"] is not None
            ]
            for row in active_contacts:
                event = row["event"]
                if event == "three_start_thread_entry":
                    for path in row["collider_paths"]:
                        if "/CouplingThread/Rail_" in path:
                            thread_start_paths.add(path.split("/Seg_", 1)[0])
                if event not in event_first:
                    event_first[event] = {
                        "step": step_index + 1,
                        "time_s": (step_index + 1) * dt,
                        "datum_B_separation_m": observed_separation,
                        "source": "physx_contact_report",
                        "first_contact": row,
                    }
                    if event == "shell_to_shell_metal_bottoming" and not all(
                        predecessor in event_first for predecessor in EVENT_ORDER[:-1]
                    ):
                        false_bottoming_count += 1

            trace_row = {
                        "step": step_index + 1,
                        "time_s": (step_index + 1) * dt,
                        "target_separation_m": target_separation,
                        "observed_separation_m": observed_separation,
                        "body_position_m": body_position_after.tolist(),
                        "nut_position_m": nut_position_after.tolist(),
                        "fixed_receptacle_position_m": fixed_position_after.tolist(),
                        "fixture_position_m": fixture_position_after.tolist(),
                        "body_orientation_wxyz": body_quaternion_after.tolist(),
                        "nut_orientation_wxyz": nut_quaternion_after.tolist(),
                        "fixed_receptacle_orientation_wxyz": fixed_quaternion_after.tolist(),
                        "fixture_orientation_wxyz": fixture_quaternion_after.tolist(),
                        "body_velocity": body_velocity_after.tolist(),
                        "nut_velocity": nut_velocity_after.tolist(),
                        "target_nut_joint_rotZ_rad": target_relative_yaw,
                        "target_nut_world_yaw_rad": target_nut_yaw,
                        "body_unwrapped_yaw_rad": body_unwrapped_yaw,
                        "nut_unwrapped_yaw_rad": nut_unwrapped_yaw,
                        "body_force_n": body_force.tolist(),
                        "nut_force_n": nut_force.tolist(),
                        "diagnostic_body_axial_force_n": diagnostic_body_axial_n,
                        "diagnostic_nut_axial_force_n": diagnostic_nut_axial_n,
                        "body_torque_nm": body_torque.tolist(),
                        "nut_torque_nm": nut_torque.tolist(),
                        "fixed_receptacle_max_component_drift_m": fixed_drift,
                        "fixture_max_component_drift_m": fixture_drift,
                        "active_scored_contact_count": len(active_contacts),
                        "active_scored_contact_events": sorted(
                            {row["event"] for row in active_contacts}
                        ),
            }
            trace_stream.write(
                json.dumps(trace_row, allow_nan=False, sort_keys=True) + "\n"
            )
            trace_jam_window.append(trace_row)
            diagnostic_record = {
                "trace": trace_row,
                "contacts": all_contact_rows,
            }
            diagnostic_history.append(diagnostic_record)
            if first_jam is None:
                detected = _jam_from_window(tuple(trace_jam_window))
                if detected is not None:
                    first_jam = detected
                    captured_jam_records = list(diagnostic_history)
                    jam_capture_end_step = int(detected["start_step"]) + int(
                        round(0.5 / dt)
                    )
            elif (
                jam_capture_end_step is not None
                and step_index + 1 <= jam_capture_end_step
            ):
                captured_jam_records.append(diagnostic_record)
    finally:
        trace_stream.close()

    write_contact_audit()
    jam_analysis = _write_first_jam_evidence(
        output,
        captured_jam_records,
        first_jam,
        dt=dt,
        axis_xy=fixed_origin[:2],
    )

    expected_positions = {
        key: float(value) / 1000.0
        for key, value in p1_contract["pass"][
            "nominal_event_datum_B_separation_mm"
        ].items()
    }
    tolerance = float(p1_contract["pass"]["nominal_event_position_tolerance_m"])
    observed_order = tuple(
        event for event, _row in sorted(event_first.items(), key=lambda item: item[1]["step"])
    )
    position_errors = {
        event: (
            None
            if event not in event_first
            else float(event_first[event]["datum_B_separation_m"] - expected_positions[event])
        )
        for event in EVENT_ORDER
    }
    event_inventory_complete = all(event in event_first for event in EVENT_ORDER)
    event_order_pass = observed_order == EVENT_ORDER
    position_pass = all(
        error is not None and abs(error) <= tolerance
        for error in position_errors.values()
    )
    all_three_starts = len(thread_start_paths) == 3
    fixed_anchor_chain_exists = bool(
        a2_report.get("fixture_load_path")
        == "FixedReceptacle->FixedFixture->world"
    )
    fixed_drift_limit_m = float(
        config.physics.maximum_fixed_translation_drift_m
    )
    fixed_anchor_pass = bool(
        fixed_anchor_chain_exists
        and maximum_fixed_translation_drift_m <= fixed_drift_limit_m
        and maximum_fixture_translation_drift_m <= fixed_drift_limit_m
    )
    sustained_torque_saturation_pass = bool(
        maximum_consecutive_torque_saturation_steps < 10
    )
    acceptance_predicate_met = bool(
        event_inventory_complete
        and event_order_pass
        and position_pass
        and all_three_starts
        and false_bottoming_count == 0
        and solver_error_count == 0
        and pose_write_after_start_count == 0
        and initial_state_pass
        and premotion_state_pass
        and fixed_anchor_pass
        and sustained_torque_saturation_pass
    )
    diagnostic_only = arguments.diagnostic_driver_kind is not None
    passed = bool(acceptance_predicate_met and not diagnostic_only)
    is_r12 = acceptance_path.name == (
        "d38999_keyed_v3_physical_acceptance_r12_v1.yaml"
    )
    return {
        "schema_version": (
            "kcg_d38999_physical_r12_p1_raw_v1" if is_r12 else SCHEMA_VERSION
        ),
        "generator_id": (
            "kcg_d38999_physical_r12_p1_force_bench_v1"
            if is_r12
            else GENERATOR_ID
        ),
        "contract_revision": a2_report["contract_revision"],
        "bench_id": "P1",
        "bench_name": p1_contract["name"],
        "bench_mode": p1_contract["mode"],
        "status": (
            "DIAGNOSTIC_COMPLETED"
            if diagnostic_only
            else ("PASSED" if passed else "FAILED")
        ),
        "a2_same_process_gate": a2_report,
        "cooked_geometry_same_process_gate": cooked_geometry_report,
        "asset_path": str(asset_path),
        "local_candidate_authorization": (
            local_authorization.evidence()
            if local_authorization is not None
            else None
        ),
        "task_r12_006c_local_h2_authorization": {
            "authorized": arguments.task_r12_006c_local_h2_authorized,
            "task_id": (
                TASK_R12_006C_ID
                if arguments.task_r12_006c_local_h2_authorized
                else None
            ),
            "candidate_index": (
                arguments.candidate_index
                if arguments.task_r12_006c_local_h2_authorized
                else None
            ),
            "candidate_asset_sha256_expected": (
                TASK_R12_006C_CANDIDATE_ASSET_SHA256
                if arguments.task_r12_006c_local_h2_authorized
                else None
            ),
            "candidate_build_result_sha256_expected": (
                TASK_R12_006C_BUILD_RESULT_SHA256
                if arguments.task_r12_006c_local_h2_authorized
                else None
            ),
        },
        "physics_rate_hz": config.physics.rate_hz,
        "inputs": dict(p1_contract["inputs"]),
        "component_driver_profile": driver,
        "component_driver_profile_matches_frozen_contract": not diagnostic_only,
        "base_component_driver_profile_matches_frozen_contract": True,
        "diagnostic_only": diagnostic_only,
        "formal_p1_pass_claimed": False if diagnostic_only else passed,
        "acceptance_predicate_met_diagnostic_only": (
            acceptance_predicate_met if diagnostic_only else None
        ),
        "diagnostic_driver": {
            "kind": arguments.diagnostic_driver_kind,
            "control_inputs": (
                "existing_target_minus_component_axial_state_only"
                if arguments.diagnostic_driver_kind == "bounded_axial_integral"
                else "predeclared_target_separation_schedule_only"
            ) if diagnostic_only else None,
            "contact_name_used_for_control": False,
            "contact_normal_used_for_control": False,
            "receptacle_pose_used_for_diagnostic_increment": False,
            "integral_gain_n_m_s": (
                arguments.diagnostic_integral_gain_n_m_s
                if arguments.diagnostic_driver_kind == "bounded_axial_integral"
                else None
            ),
            "integral_component_limit_n": (
                arguments.diagnostic_integral_component_limit_n
                if arguments.diagnostic_driver_kind == "bounded_axial_integral"
                else None
            ),
            "feedforward_component_n": (
                arguments.diagnostic_feedforward_component_n
                if arguments.diagnostic_driver_kind
                == "bounded_axial_force_feedforward"
                else None
            ),
            "feedforward_start_separation_m": (
                arguments.diagnostic_feedforward_start_separation_m
                if arguments.diagnostic_driver_kind
                == "bounded_axial_force_feedforward"
                else None
            ),
            "feedforward_ramp_m": (
                arguments.diagnostic_feedforward_ramp_m
                if arguments.diagnostic_driver_kind
                == "bounded_axial_force_feedforward"
                else None
            ),
        },
        "initial_state": initial_state,
        "initial_state_pass": initial_state_pass,
        "initial_state_thresholds": initial_state_thresholds,
        "premotion_state": premotion_state,
        "premotion_state_pass": premotion_state_pass,
        "event_first": event_first,
        "observed_event_order": list(observed_order),
        "position_error_m": position_errors,
        "event_position_tolerance_m": tolerance,
        "event_inventory_complete": event_inventory_complete,
        "event_order_pass": event_order_pass,
        "event_position_pass": position_pass,
        "thread_start_paths": sorted(thread_start_paths),
        "all_three_thread_starts_enter": all_three_starts,
        "maximum_force_component_n": maximum_force_component_n,
        "force_component_limit_n": float(
            driver["translation_force_component_limit_n"]
        ),
        "maximum_torque_component_nm": maximum_torque_component_nm,
        "maximum_consecutive_torque_saturation_steps": (
            maximum_consecutive_torque_saturation_steps
        ),
        "sustained_torque_saturation_pass": sustained_torque_saturation_pass,
        "first_sustained_torque_jam": first_jam,
        "first_jam_analysis_file": "first_jam_analysis.json",
        "first_jam_analysis": jam_analysis,
        "fixed_anchor_chain_exists": fixed_anchor_chain_exists,
        "fixed_anchor_chain": a2_report.get("fixture_load_path"),
        "maximum_fixed_receptacle_translation_drift_m": (
            maximum_fixed_translation_drift_m
        ),
        "maximum_fixture_translation_drift_m": (
            maximum_fixture_translation_drift_m
        ),
        "fixed_translation_drift_limit_m": fixed_drift_limit_m,
        "fixed_anchor_pass": fixed_anchor_pass,
        "false_bottoming_count": false_bottoming_count,
        "solver_error_count": solver_error_count,
        "object_pose_write_after_physics_start_count": pose_write_after_start_count,
        "external_work_j": external_work_j,
        "trace_file": "trace.jsonl",
        "contact_audit_file": "contact_audit.json",
        "first_jam_window_file": "first_jam_window.jsonl",
        "first_jam_contacts_file": "first_jam_contacts.csv",
        "trace_step_count": total_steps,
        "contact_truth_role": "posthoc_scoring_only_never_command_input",
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    portable = Path(arguments.kit_portable_root).expanduser().resolve()
    if not portable.is_relative_to(Path("/tmp")):
        raise ValueError("Kit portable root must be below /tmp")
    portable.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("WARP_CACHE_PATH", str(portable / "warp-cache"))
    sys.argv.extend(["--portable-root", str(portable)])
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P1 evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

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
    status = 1
    report: Mapping[str, Any]
    try:
        report = _run(arguments, output)
        status = 0 if (report["passed"] or report.get("diagnostic_only")) else 1
    except BaseException as error:
        is_r12 = Path(arguments.acceptance_config).name == (
            "d38999_keyed_v3_physical_acceptance_r12_v1.yaml"
        )
        report = {
            "schema_version": (
                "kcg_d38999_physical_r12_p1_raw_v1"
                if is_r12
                else SCHEMA_VERSION
            ),
            "generator_id": (
                "kcg_d38999_physical_r12_p1_force_bench_v1"
                if is_r12
                else GENERATOR_ID
            ),
            "contract_revision": (
                "keyed_v3_physical_r12"
                if is_r12
                else "keyed_v3_physical_r11"
            ),
            "bench_id": "P1",
            "status": "FAILED",
            "diagnostic_only": arguments.diagnostic_driver_kind is not None,
            "formal_p1_pass_claimed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "passed": False,
        }
        traceback.print_exc()
    finally:
        (output / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    _emit(json.dumps(report, allow_nan=False, sort_keys=True))
    if Path(arguments.acceptance_config).name == (
        "d38999_keyed_v3_physical_acceptance_r12_v1.yaml"
    ):
        if arguments.diagnostic_driver_kind is not None and status == 0:
            _emit("ISAAC PHYSICAL R12 P1 DIAGNOSTIC COMPLETED")
        else:
            _emit(
                "ISAAC PHYSICAL R12 P1 NOMINAL PASSED"
                if status == 0
                else "ISAAC PHYSICAL R12 P1 NOMINAL FAILED"
            )
    else:
        _emit(PASS_BANNER if status == 0 else FAIL_BANNER)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
