#!/usr/bin/env python3

"""One bounded Isaac check for D38999 event onsets or one A1 reset repeat."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


TASK_ID = "DYN-A1-EVENT-ONSET-CALIBRATION"
ROOT = "/World/D38999MultilayerV1/D38999_ASSEMBLY_CONTROL_V1"
PAIR_ROOT = ROOT + "/D38999Pair"
FIXED_PATH = PAIR_ROOT + "/FixedReceptacle"
BODY_PATH = PAIR_ROOT + "/LoosePlug/BodyAssembly"
NUT_PATH = PAIR_ROOT + "/LoosePlug/CouplingNut"
MODEL_RELATIVE = Path(
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
    "D38999_ASSEMBLY_CONTROL_V1.usda"
)
CONTRACT_RELATIVE = Path("src/kcg_connector/config/d38999_master_model_contract_v1.yaml")
ACCEPTANCE_RELATIVE = Path(
    "src/kcg_connector/config/d38999_keyed_v3_physical_acceptance_r12_v1.yaml"
)
BUILD_RELATIVE = Path(
    "artifacts/agent_control/tasks/DYN-A1-EVENT-ONSET-CALIBRATION/BUILD_RESULT.json"
)
ONSET_OUTPUT_RELATIVE = Path(
    "artifacts/agent_control/tasks/DYN-A1-EVENT-ONSET-CALIBRATION/ONSET_PROBE"
)
EXPECTED_SHA256 = {
    "contract": "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b",
    "acceptance": "dd37d07d5bd87a97c3dbefe225c0eafd409fc783a6010eec8db9da397ce2cf76",
    "model": "f0b07519f107f8cc35bca3230d1fd7352ea529c4ab77f7e969771cee75687874",
}
START_M = 0.0055
TOLERANCE_M = 0.00005
EVENT_ORDER = (
    "five_key_polarization",
    "three_start_thread_entry",
    "spring_finger_engagement",
    "first_pin_socket_spring_touch",
    "pin_barrier_seal_contact",
    "seal_compression",
    "shell_to_shell_metal_bottoming",
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--onset", action="store_true")
    mode.add_argument("--initial-validation", action="store_true")
    parser.add_argument("--run-index", type=int, choices=(1, 2, 3))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kit-portable-root", default=None)
    return parser.parse_args(argv)


def _repository() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _emit(value: Mapping[str, Any]) -> None:
    os.write(1, (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode())


def _load_and_authorize(arguments: argparse.Namespace) -> dict[str, Any]:
    repository = _repository()
    paths = {
        "contract": repository / CONTRACT_RELATIVE,
        "acceptance": repository / ACCEPTANCE_RELATIVE,
        "model": repository / MODEL_RELATIVE,
    }
    actual_sha = {name: _sha256(path) for name, path in paths.items()}
    if actual_sha != EXPECTED_SHA256:
        raise PermissionError(f"frozen input fingerprint mismatch: {actual_sha}")
    state = json.loads(
        (repository / "artifacts/agent_control/MASTER_STATE.json").read_text(
            encoding="utf-8"
        )
    )
    node = state.get("dynamic_red_gate", {}).get("event_onset_calibration", {})
    if not (
        state.get("task_id") == TASK_ID
        and state.get("status") == "IMPLEMENTING"
        and node.get("hypothesis_id") == "H-ONSET-01"
        and node.get("targeted_fix_count") == 1
        and node.get("assembly_control_sha256_after") == EXPECTED_SHA256["model"]
    ):
        raise PermissionError("master state does not authorize H-ONSET-01 validation")
    if arguments.onset:
        expected_output = repository / ONSET_OUTPUT_RELATIVE
        if arguments.run_index is not None:
            raise PermissionError("onset mode has no run index")
        if not (
            node.get("onset_probe_runs_started") == 1
            and node.get("onset_probe_runs_completed") == 0
        ):
            raise PermissionError("onset probe counter is not exactly 1 started / 0 completed")
    else:
        if arguments.run_index is None:
            raise PermissionError("initial validation requires --run-index")
        expected_output = repository / (
            "artifacts/agent_control/tasks/DYN-A1-EVENT-ONSET-CALIBRATION/"
            f"VALIDATION_{arguments.run_index:02d}"
        )
        if node.get("onset_probe_outcome") != "PASS":
            raise PermissionError("initial repeats require a passed onset probe")
        if not (
            node.get("validation_processes_started") == arguments.run_index
            and node.get("validation_processes_completed") == arguments.run_index - 1
        ):
            raise PermissionError("independent validation counter is inconsistent")
    output = Path(arguments.output_dir).expanduser().resolve()
    if output != expected_output.resolve():
        raise PermissionError(f"output path differs: {output} != {expected_output}")
    if output.exists():
        raise FileExistsError(output)
    contract = yaml.safe_load(paths["contract"].read_text(encoding="utf-8"))
    acceptance = yaml.safe_load(paths["acceptance"].read_text(encoding="utf-8"))
    build = json.loads((repository / BUILD_RELATIVE).read_text(encoding="utf-8"))
    if build.get("assembly_control", {}).get("sha256_after") != actual_sha["model"]:
        raise PermissionError("build result does not bind the current assembly asset")
    scale = float(contract["contact_layout"]["coordinate_scale_m_per_in"])
    pairs = [
        {
            "label": str(row["label"]),
            "center_m": [float(value) * scale for value in row["center_in"]],
            "same_label_only": True,
        }
        for row in contract["contact_layout"]["pairs"]
    ]
    return {
        "repository": repository,
        "output": output,
        "paths": paths,
        "input_sha256": actual_sha,
        "contract": contract,
        "acceptance": acceptance,
        "pairs": pairs,
    }


def _set_initial_pose(
    stage: Any,
    path: str,
    *,
    x_m: float,
    z_m: float,
    yaw_deg: float,
    UsdGeom: Any,
    Gf: Any,
) -> None:
    prim = stage.GetPrimAtPath(path)
    xformable = UsdGeom.Xformable(prim)
    if not prim or xformable.GetOrderedXformOps():
        raise RuntimeError(f"unexpected initial transform stack: {path}")
    xformable.AddTranslateOp().Set(Gf.Vec3d(x_m, 0.0, z_m))
    if yaw_deg:
        xformable.AddRotateZOp().Set(float(yaw_deg))


def _attribute(stage: Any, path: str, name: str) -> Any:
    prim = stage.GetPrimAtPath(path)
    attribute = prim.GetAttribute(name) if prim else None
    return attribute.Get() if attribute and attribute.HasAuthoredValueOpinion() else None


def _contact_rows(stage: Any, interface: Any, schema_tools: Any) -> list[dict[str, Any]]:
    headers, contacts, _friction = interface.get_full_contact_report()
    rows: list[dict[str, Any]] = []
    for header in headers:
        paths = [
            str(schema_tools.intToSdfPath(header.collider0)),
            str(schema_tools.intToSdfPath(header.collider1)),
        ]
        actors = [
            str(schema_tools.intToSdfPath(header.actor0)),
            str(schema_tools.intToSdfPath(header.actor1)),
        ]
        start = int(header.contact_data_offset)
        stop = start + int(header.num_contact_data)
        separations = [float(contacts[index].separation) for index in range(start, stop)]
        rows.append(
            {
                "actor_paths": actors,
                "collider_paths": paths,
                "collision_roles": [
                    _attribute(stage, path, "kcg:collisionRole") for path in paths
                ],
                "trace_labels": [
                    _attribute(stage, path, "kcg:traceLabel") for path in paths
                ],
                "minimum_separation_m": min(separations) if separations else None,
                "contact_record_count": int(header.num_contact_data),
                "interpart": FIXED_PATH in actors
                and any(actor in {BODY_PATH, NUT_PATH} for actor in actors),
            }
        )
    return rows


def _family(row: Mapping[str, Any]) -> str | None:
    roles = [str(value) for value in row["collision_roles"] if value is not None]
    if sorted(roles) == ["continuous_keyway_wall", "continuous_polarizing_key"]:
        return "five_keys_and_keyways"
    if roles == ["continuous_shell_and_guidance", "continuous_shell_and_guidance"] or roles == [
        "continuous_shell_and_guidance",
        "continuous_shell_and_guidance",
    ]:
        return "continuous_shell_and_guidance"
    if sorted(roles) == [
        "continuous_real_metal_stop_fixed",
        "continuous_real_metal_stop_plug",
    ]:
        return "metal_stop"
    return None


def _key_label_mismatch(row: Mapping[str, Any]) -> bool:
    labels = [str(value) for value in row["trace_labels"] if value is not None]
    key = next((value for value in labels if value.startswith("key_")), None)
    keyway = next((value for value in labels if value.startswith("keyway_")), None)
    return bool(key and keyway and key.split("_", 1)[1] != keyway.split("_", 1)[1])


def _static_geometry(stage: Any, UsdGeom: Any) -> dict[str, Any]:
    key_rows = []
    keyway_rows = []
    for index in range(5):
        key_path = BODY_PATH + f"/PolarizingKeys/Key_{index}"
        key_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(key_path)).GetLocalTransformation()
        key_translation = key_matrix.ExtractTranslation()
        key_rows.append(
            {
                "index": index,
                "translation_m": [float(value) for value in key_translation],
                "radius_m": math.hypot(float(key_translation[0]), float(key_translation[1])),
                "angle_deg": math.degrees(
                    math.atan2(float(key_translation[1]), float(key_translation[0]))
                )
                % 360.0,
            }
        )
        walls = []
        for side in ("LeftWall", "RightWall"):
            path = FIXED_PATH + f"/MatingShell/KeywayShell/Keyway_{index}_{side}"
            matrix = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()
            translation = matrix.ExtractTranslation()
            walls.append([float(value) for value in translation])
        keyway_rows.append({"index": index, "wall_translations_m": walls})
    guide_root = stage.GetPrimAtPath(BODY_PATH + "/MatingShell/ContinuousPlugGuideCollision")
    fixed_stop_root = stage.GetPrimAtPath(FIXED_PATH + "/MatingShell/MetalStop")
    return {
        "key_rows": key_rows,
        "keyway_rows": keyway_rows,
        "plug_guide_segment_count": sum(1 for _ in guide_root.GetChildren()),
        "fixed_stop_segment_count": sum(1 for _ in fixed_stop_root.GetChildren()),
        "key_radius_min_m": min(row["radius_m"] for row in key_rows),
        "key_radius_max_m": max(row["radius_m"] for row in key_rows),
        "key_angles_deg": [row["angle_deg"] for row in key_rows],
    }


def _run_scenario(
    *,
    application: Any,
    frozen: Mapping[str, Any],
    name: str,
    x_offset_m: float,
    yaw_deg: float,
    end_m: float | None,
    include_internal: bool,
    trace: Any,
) -> dict[str, Any]:
    from d38999_multilayer_nominal_bench import (
        _finite,
        _internal_effects,
        _rpy_wxyz,
        _velocity_pi,
    )
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_simulation_interface
    import omni.usd
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, UsdGeom

    World.clear_instance()
    context = omni.usd.get_context()
    if context.open_stage(str(frozen["paths"]["model"])) is not True:
        raise RuntimeError("assembly-control stage did not open")
    for _ in range(3):
        application.update()
    rate_hz = int(frozen["acceptance"]["shared_numeric_profile"]["physics_rate_hz"])
    dt = 1.0 / rate_hz
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=dt,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    _set_initial_pose(
        stage, BODY_PATH, x_m=x_offset_m, z_m=-START_M, yaw_deg=yaw_deg,
        UsdGeom=UsdGeom, Gf=Gf,
    )
    _set_initial_pose(
        stage, NUT_PATH, x_m=x_offset_m, z_m=-START_M, yaw_deg=yaw_deg,
        UsdGeom=UsdGeom, Gf=Gf,
    )
    for owner in (FIXED_PATH, BODY_PATH, NUT_PATH):
        PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(owner)
        ).CreateThresholdAttr().Set(0.0)
    world.get_physics_context().set_gravity(0.0)
    world.reset()
    fixed = RigidPrim(prim_paths_expr=FIXED_PATH, name=name + "_fixed", reset_xform_properties=False)
    body = RigidPrim(prim_paths_expr=BODY_PATH, name=name + "_body", reset_xform_properties=False)
    nut = RigidPrim(prim_paths_expr=NUT_PATH, name=name + "_nut", reset_xform_properties=False)
    for view in (fixed, body, nut):
        view.initialize()

    def state(view: Any, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    interface = get_physx_simulation_interface()
    fixed_initial, _fixed_q, _fixed_v = state(fixed, name + " initial fixed")
    body_initial, _body_q, _body_v = state(body, name + " initial body")
    nut_initial, _nut_q, _nut_v = state(nut, name + " initial nut")
    datum_after_reset = float(fixed_initial[2] - body_initial[2])
    world.step(render=False)
    fixed_first, _q, _v = state(fixed, name + " first fixed")
    body_first, _q, _v = state(body, name + " first body")
    first_rows = [
        row for row in _contact_rows(stage, interface, PhysicsSchemaTools) if row["interpart"]
    ]
    datum_after_first_step = float(fixed_first[2] - body_first[2])
    onsets: dict[str, dict[str, Any]] = {}
    cross_key_rows: list[dict[str, Any]] = []
    premature_stop_rows: list[dict[str, Any]] = []
    maximum_penetration_m = 0.0
    maximum_fixed_drift_m = float(np.max(np.abs(fixed_first - fixed_initial)))
    force_integrals = {"body": 0.0, "nut": 0.0}
    speed = float(frozen["acceptance"]["benches"]["P1"]["inputs"]["axial_speed_m_s"])
    force_limit = float(
        frozen["acceptance"]["benches"]["P1"]["inputs"]["component_driver_profile"][
            "translation_force_component_limit_n"
        ]
    )
    steps = 1
    final_separation = datum_after_first_step
    if end_m is not None:
        maximum_steps = int(math.ceil(2.0 * (end_m - START_M) / speed / dt)) + 240
        wall_last_heartbeat = time.monotonic()
        stop_hold = 0
        for index in range(maximum_steps):
            fixed_p, fixed_q, fixed_v = state(fixed, name + " fixed")
            body_p, body_q, body_v = state(body, name + " body")
            nut_p, nut_q, nut_v = state(nut, name + " nut")
            body_z, force_integrals["body"] = _velocity_pi(
                target=-speed,
                actual=float(body_v[2]),
                integral=force_integrals["body"],
                dt=dt,
                kp=20.0,
                ki=1000.0,
                limit=force_limit,
            )
            nut_z, force_integrals["nut"] = _velocity_pi(
                target=-speed,
                actual=float(nut_v[2]),
                integral=force_integrals["nut"],
                dt=dt,
                kp=20.0,
                ki=1000.0,
                limit=force_limit,
            )
            body_force = np.asarray((0.0, 0.0, body_z), dtype=np.float64)
            body_torque = np.zeros(3, dtype=np.float64)
            nut_force = np.asarray((0.0, 0.0, nut_z), dtype=np.float64)
            nut_torque = np.zeros(3, dtype=np.float64)
            internal = None
            if include_internal:
                internal = _internal_effects(
                    frozen,
                    fixed_position=fixed_p,
                    fixed_velocity=fixed_v,
                    body_position=body_p,
                    body_velocity=body_v,
                    body_yaw=_rpy_wxyz(body_q)[2],
                    body_omega_z=float(body_v[5]),
                    nut_yaw=_rpy_wxyz(nut_q)[2],
                    nut_omega_z=float(nut_v[5]),
                )
                body_force += internal["body_force_n"]
                body_torque += internal["body_torque_nm"]
                nut_force += internal["nut_force_n"]
                nut_torque += internal["nut_torque_nm"]
            apply(body, body_force, body_torque)
            apply(nut, nut_force, nut_torque)
            world.step(render=False)
            steps += 1
            fixed_after, _q, _v = state(fixed, name + " fixed after")
            body_after, _q, _v = state(body, name + " body after")
            final_separation = float(fixed_after[2] - body_after[2])
            maximum_fixed_drift_m = max(
                maximum_fixed_drift_m,
                float(np.max(np.abs(fixed_after - fixed_initial))),
            )
            rows = [
                row
                for row in _contact_rows(stage, interface, PhysicsSchemaTools)
                if row["interpart"]
            ]
            for row in rows:
                separation = row["minimum_separation_m"]
                if separation is not None:
                    maximum_penetration_m = max(maximum_penetration_m, max(0.0, -separation))
                family = _family(row)
                if family and family not in onsets:
                    onsets[family] = {
                        "datum_B_separation_m": final_separation,
                        "step": steps,
                        "collider_paths": row["collider_paths"],
                    }
                if family == "five_keys_and_keyways" and _key_label_mismatch(row):
                    cross_key_rows.append(row)
                if family == "metal_stop" and final_separation < 0.015:
                    premature_stop_rows.append(row)
            if internal is not None:
                measurements = internal["measurements"]
                candidates = {
                    "three_start_thread_entry": abs(measurements["thread_constraint_force_n"]),
                    "spring_finger_engagement": abs(measurements["spring_axial_resistance_n"]),
                    "first_pin_socket_spring_touch": (
                        measurements["maximum_same_label_radial_deflection_m"]
                        if internal["channels"]["same_label_pin_effects_active"]
                        else 0.0
                    ),
                    "pin_barrier_seal_contact": abs(
                        measurements["pin_isolation_axial_resistance_n"]
                    ),
                    "seal_compression": abs(measurements["peripheral_seal_axial_resistance_n"]),
                }
                for event, magnitude in candidates.items():
                    if event not in onsets and float(magnitude) > 1.0e-12:
                        onsets[event] = {
                            "datum_B_separation_m": float(measurements["separation_m"]),
                            "step": steps,
                            "physical_magnitude": float(magnitude),
                        }
            if steps % rate_hz == 0 or any(row["step"] == steps for row in onsets.values()):
                trace.write(
                    json.dumps(
                        {
                            "scenario": name,
                            "step": steps,
                            "datum_B_separation_m": final_separation,
                            "active_interpart_contact_count": len(rows),
                            "observed_onsets": sorted(onsets),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                trace.flush()
            if time.monotonic() - wall_last_heartbeat >= 60.0:
                heartbeat = frozen["output"] / "heartbeat.json"
                heartbeat.write_text(
                    json.dumps(
                        {
                            "task_id": TASK_ID,
                            "scenario": name,
                            "step": steps,
                            "datum_B_separation_m": final_separation,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                wall_last_heartbeat = time.monotonic()
            if "metal_stop" in onsets:
                stop_hold += 1
                if stop_hold >= 20:
                    break
            elif final_separation >= end_m:
                break
    final_fixed, _q, _v = state(fixed, name + " final fixed")
    final_body, _q, _v = state(body, name + " final body")
    final_nut, _q, _v = state(nut, name + " final nut")
    result = {
        "name": name,
        "initial_x_offset_m": x_offset_m,
        "initial_yaw_deg": yaw_deg,
        "datum_after_reset_m": datum_after_reset,
        "datum_after_first_step_m": datum_after_first_step,
        "initial_datum_error_m": max(
            abs(datum_after_reset - START_M), abs(datum_after_first_step - START_M)
        ),
        "initial_interpart_contact_pair_count": len(first_rows),
        "initial_interpart_contacts": first_rows,
        "onsets": onsets,
        "cross_key_contact_count": len(cross_key_rows),
        "cross_key_contacts": cross_key_rows[:20],
        "premature_metal_stop_contact_count": len(premature_stop_rows),
        "maximum_interpart_penetration_m": maximum_penetration_m,
        "maximum_fixed_receptacle_translation_drift_m": maximum_fixed_drift_m,
        "final_datum_B_separation_m": float(final_fixed[2] - final_body[2]),
        "final_nut_body_relative_z_m": float(final_nut[2] - final_body[2]),
        "physics_step_count": steps,
        "object_pose_write_after_physics_start_count": 0,
    }
    world.stop()
    World.clear_instance()
    return result


def _onset_result(
    application: Any,
    frozen: Mapping[str, Any],
    trace: Any,
    log_messages: list[str],
) -> dict[str, Any]:
    # Fixed, derived perturbations: 0.30 deg exceeds the narrow-key 0.25 deg
    # side clearance; 0.130 mm exceeds the shell radial clearance by 17.5 um.
    key = _run_scenario(
        application=application, frozen=frozen, name="key_guidance",
        x_offset_m=0.0, yaw_deg=0.30, end_m=0.00658,
        include_internal=False, trace=trace,
    )
    guide = _run_scenario(
        application=application, frozen=frozen, name="shell_guidance",
        x_offset_m=0.000130, yaw_deg=0.0, end_m=0.00658,
        include_internal=False, trace=trace,
    )
    nominal = _run_scenario(
        application=application, frozen=frozen, name="nominal_effects_and_stop",
        x_offset_m=0.000050, yaw_deg=0.0, end_m=0.01508,
        include_internal=True, trace=trace,
    )
    event_nominal = {
        str(row["name"]): float(row["nominal_separation_m"])
        for row in frozen["contract"]["assembly_events"]["ordered"]
    }
    event_actual: dict[str, float | None] = {
        "five_key_polarization": key["onsets"].get("five_keys_and_keyways", {}).get(
            "datum_B_separation_m"
        ),
        "three_start_thread_entry": nominal["onsets"].get(
            "three_start_thread_entry", {}
        ).get("datum_B_separation_m"),
        "spring_finger_engagement": nominal["onsets"].get(
            "spring_finger_engagement", {}
        ).get("datum_B_separation_m"),
        "first_pin_socket_spring_touch": nominal["onsets"].get(
            "first_pin_socket_spring_touch", {}
        ).get("datum_B_separation_m"),
        "pin_barrier_seal_contact": nominal["onsets"].get(
            "pin_barrier_seal_contact", {}
        ).get("datum_B_separation_m"),
        "seal_compression": nominal["onsets"].get("seal_compression", {}).get(
            "datum_B_separation_m"
        ),
        "shell_to_shell_metal_bottoming": nominal["onsets"].get("metal_stop", {}).get(
            "datum_B_separation_m"
        ),
    }
    event_errors = {
        event: (None if event_actual[event] is None else event_actual[event] - event_nominal[event])
        for event in EVENT_ORDER
    }
    guide_onset = guide["onsets"].get("continuous_shell_and_guidance", {}).get(
        "datum_B_separation_m"
    )
    observed_order = [
        event
        for event, value in sorted(
            event_actual.items(), key=lambda item: math.inf if item[1] is None else item[1]
        )
        if value is not None
    ]
    physicsusd_errors = [message for message in log_messages if "PhysicsUSD:" in message]
    solver_errors = [
        message
        for message in log_messages
        if "solver" in message.lower() and "error" in message.lower()
    ]
    checks = {
        "all_eight_effect_onsets_observed": all(value is not None for value in event_actual.values())
        and guide_onset is not None,
        "seven_event_position_errors_within_50um": all(
            value is not None and abs(value) <= TOLERANCE_M for value in event_errors.values()
        ),
        "continuous_guide_onset_within_50um": guide_onset is not None
        and abs(guide_onset - event_nominal["five_key_polarization"]) <= TOLERANCE_M,
        "seven_event_order_correct": observed_order == list(EVENT_ORDER),
        "all_initial_interpart_contact_counts_zero": all(
            row["initial_interpart_contact_pair_count"] == 0 for row in (key, guide, nominal)
        ),
        "all_reset_datum_errors_within_50um": all(
            row["initial_datum_error_m"] <= TOLERANCE_M for row in (key, guide, nominal)
        ),
        "cross_key_contact_count_zero": key["cross_key_contact_count"] == 0,
        "premature_metal_stop_contact_count_zero": sum(
            row["premature_metal_stop_contact_count"] for row in (key, guide, nominal)
        ) == 0,
        "physicsusd_error_count_zero": len(physicsusd_errors) == 0,
        "solver_error_count_zero": len(solver_errors) == 0,
        "post_start_pose_write_count_zero": True,
    }
    return {
        "schema_version": "kcg_d38999_event_onset_probe_v1",
        "task_id": TASK_ID,
        "mode": "onset",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": (
            "EVENT_ONSETS_CALIBRATED" if all(checks.values()) else "EVENT_ONSET_VALIDATION_FAILED"
        ),
        "checks": checks,
        "event_nominal_m": event_nominal,
        "event_actual_m": event_actual,
        "event_error_m": event_errors,
        "continuous_guide_actual_onset_m": guide_onset,
        "observed_event_order": observed_order,
        "scenarios": {"key_guidance": key, "shell_guidance": guide, "nominal": nominal},
        "physicsusd_errors": physicsusd_errors,
        "solver_errors": solver_errors,
        "diagnostic_only": True,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }


def _initial_result(
    application: Any,
    frozen: Mapping[str, Any],
    trace: Any,
    log_messages: list[str],
    run_index: int,
) -> dict[str, Any]:
    scenario = _run_scenario(
        application=application, frozen=frozen, name=f"validation_{run_index:02d}",
        x_offset_m=0.0, yaw_deg=0.0, end_m=None,
        include_internal=False, trace=trace,
    )
    physicsusd_errors = [message for message in log_messages if "PhysicsUSD:" in message]
    solver_errors = [
        message
        for message in log_messages
        if "solver" in message.lower() and "error" in message.lower()
    ]
    joint = frozen["contract"]["coupling_nut_motion"]["transZ_backup_limits_m"]
    relative_z = scenario["final_nut_body_relative_z_m"]
    checks = {
        "initial_interpart_contact_pair_count_zero": scenario[
            "initial_interpart_contact_pair_count"
        ] == 0,
        "reset_datum_error_within_50um": scenario["initial_datum_error_m"] <= TOLERANCE_M,
        "nut_body_relation_legal": float(joint["low"]) <= relative_z <= float(joint["high"]),
        "physicsusd_error_count_zero": len(physicsusd_errors) == 0,
        "solver_error_count_zero": len(solver_errors) == 0,
        "post_start_pose_write_count_zero": scenario[
            "object_pose_write_after_physics_start_count"
        ] == 0,
    }
    return {
        "schema_version": "kcg_d38999_event_onset_a1_repeat_v1",
        "task_id": TASK_ID,
        "mode": "initial_validation",
        "run_index": run_index,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "individual_dynamic_pass": all(checks.values()),
        "checks": checks,
        "scenario": scenario,
        "physicsusd_errors": physicsusd_errors,
        "solver_errors": solver_errors,
        "object_pose_write_after_physics_start_count": 0,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    frozen = _load_and_authorize(arguments)
    frozen["output"].mkdir(parents=True, exist_ok=False)
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
    import omni.usd
    from pxr import UsdGeom

    messages: list[str] = []
    logging = carb.logging.acquire_logging()

    def on_log(source: str, level: int, filename: str, line: int, message: str) -> None:
        del source, level, filename, line
        messages.append(str(message))

    handle = logging.add_logger(on_log)
    trace = (frozen["output"] / "trace.jsonl").open("x", encoding="utf-8")
    status = 1
    result: dict[str, Any]
    try:
        context = omni.usd.get_context()
        if context.open_stage(str(frozen["paths"]["model"])) is not True:
            raise RuntimeError("stage did not open for static geometry readback")
        for _ in range(3):
            application.update()
        static_geometry = _static_geometry(context.get_stage(), UsdGeom)
        result = (
            _onset_result(application, frozen, trace, messages)
            if arguments.onset
            else _initial_result(
                application, frozen, trace, messages, int(arguments.run_index)
            )
        )
        result["static_geometry"] = static_geometry
        result["input_sha256"] = frozen["input_sha256"]
        result["object_pose_write_after_physics_start_count"] = 0
        result["simulation_started"] = True
        status = 0 if result["status"] == "PASS" else 2
    except BaseException as error:
        traceback.print_exc()
        result = {
            "schema_version": "kcg_d38999_event_onset_probe_v1",
            "task_id": TASK_ID,
            "mode": "onset" if arguments.onset else "initial_validation",
            "status": "ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "input_sha256": frozen["input_sha256"],
            "object_pose_write_after_physics_start_count": 0,
            "formal_p1_pass_claimed": False,
            "formal_r12_generated": False,
            "hardware_authorized": False,
        }
    finally:
        trace.close()
        logging.remove_logger(handle)
        application.close()
    (frozen["output"] / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _emit(
        {
            "task_id": TASK_ID,
            "mode": result["mode"],
            "status": result["status"],
            "report": str(frozen["output"] / "report.json"),
        }
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
