#!/usr/bin/env python3

"""Run one connector-only J599/25-35 assembly case in a fresh Isaac process.

The fixture is authored before physics starts.  During physics, only the
predeclared joint-drive command changes; connector world poses are never
written.  The nominal case validates the nut/thread proxy, while the wrong-key
case isolates that proxy and validates the five-key geometry with bounded
linear force.  Every invocation executes exactly one case so their Isaac
process lineage remains independent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = MODEL_ROOT / "generated" / "j599_25_35_pair_assembly.usdc"
DEFAULT_CONTRACT = MODEL_ROOT / "config" / "model_contract.json"
BUILD_PLAN = MODEL_ROOT / "plans" / "BUILD_AND_VALIDATION_PLAN.json"

PAIR_ROOT = "/World/J599_25_35_N_Pair"
FIXED_PATH = PAIR_ROOT + "/FixedReceptacle_J599_20FJ35SN"
BODY_PATH = PAIR_ROOT + "/LoosePlug_J599_26FJ35PN/Body"
NUT_PATH = PAIR_ROOT + "/LoosePlug_J599_26FJ35PN/CouplingNut"
HINGE_PATH = PAIR_ROOT + "/LoosePlug_J599_26FJ35PN/CouplingNutRevolute"
PRISMATIC_PATH = PAIR_ROOT + "/ValidationFixture/InsertionPrismatic"
THREAD_PATH = PAIR_ROOT + "/ValidationFixture/ThreadCoupling"
SCENE_PATH = "/World/J599_25_35_ValidationPhysicsScene"

PHYSICS_HZ = 240
DRIVE_SPEED_DEG_S = 120.0
DRIVE_MAX_TORQUE_NM = 0.0095
DRIVE_DAMPING_NM_PER_DEG_S = 0.01
LINEAR_DRIVE_STIFFNESS_N_PER_M = 2000.0
LINEAR_DRIVE_DAMPING_N_S_PER_M = 20.0
TRACE_STRIDE = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _inside_model_root(path: Path) -> bool:
    return path == MODEL_ROOT or MODEL_ROOT in path.parents


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=("nominal", "wrong_key_3deg"),
    )
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kit-portable-root", type=Path)
    return parser.parse_args(argv)


def _relative_z_angle(Gf: Any, Usd: Any, UsdGeom: Any, body: Any, nut: Any) -> float:
    body_matrix = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    nut_matrix = UsdGeom.Xformable(nut).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    body_quaternion = Gf.Transform(body_matrix).GetRotation().GetQuat()
    nut_quaternion = Gf.Transform(nut_matrix).GetRotation().GetQuat()
    relative = body_quaternion.GetInverse() * nut_quaternion
    imaginary = relative.GetImaginary()
    angle = 2.0 * math.atan2(float(imaginary[2]), float(relative.GetReal()))
    return math.atan2(math.sin(angle), math.cos(angle))


def _unwrap(previous: float, wrapped: float) -> float:
    previous_wrapped = math.atan2(math.sin(previous), math.cos(previous))
    delta = math.atan2(
        math.sin(wrapped - previous_wrapped),
        math.cos(wrapped - previous_wrapped),
    )
    return previous + delta


def _world_translation(Usd: Any, UsdGeom: Any, prim: Any) -> tuple[float, float, float]:
    value = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).ExtractTranslation()
    result = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError(f"non-finite world translation at {prim.GetPath()}")
    return result


def _collision_role(stage: Any, path: str) -> str:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return "UNLABELED"
    value = prim.GetCustomDataByKey("j599:collisionRole")
    return str(value) if value is not None else "UNLABELED"


def _contact_rows(
    stage: Any,
    interface: Any,
    schema_tools: Any,
    dt: float,
) -> list[dict[str, Any]]:
    headers, contacts, _friction = interface.get_full_contact_report()
    rows: list[dict[str, Any]] = []
    for header in headers:
        actors = (
            str(schema_tools.intToSdfPath(header.actor0)),
            str(schema_tools.intToSdfPath(header.actor1)),
        )
        colliders = (
            str(schema_tools.intToSdfPath(header.collider0)),
            str(schema_tools.intToSdfPath(header.collider1)),
        )
        searchable = actors + colliders
        fixed_involved = any(path.startswith(FIXED_PATH) for path in searchable)
        loose_involved = any(path.startswith(BODY_PATH) for path in searchable)
        if not (fixed_involved and loose_involved):
            continue
        start = int(header.contact_data_offset)
        stop = start + int(header.num_contact_data)
        separations: list[float] = []
        impulse_norms: list[float] = []
        for index in range(start, stop):
            record = contacts[index]
            separation = float(record.separation)
            impulse = tuple(float(item) for item in record.impulse)
            if not math.isfinite(separation) or not all(
                math.isfinite(item) for item in impulse
            ):
                raise RuntimeError("non-finite contact record")
            separations.append(separation)
            impulse_norms.append(math.sqrt(sum(item * item for item in impulse)))
        roles = tuple(_collision_role(stage, path) for path in colliders)
        role_set = set(roles)
        rows.append(
            {
                "actor_paths": list(actors),
                "collider_paths": list(colliders),
                "collision_roles": list(roles),
                "contact_record_count": int(header.num_contact_data),
                "minimum_separation_m": min(separations) if separations else None,
                "maximum_impulse_norm_n_s": max(impulse_norms, default=0.0),
                "maximum_equivalent_force_n": max(
                    (value / dt for value in impulse_norms), default=0.0
                ),
                "wrong_key_pair": (
                    "plug_polarizing_key" in role_set
                    and bool(
                        role_set
                        & {
                            "fixed_keyway_blocking_shell",
                            "fixed_keyway_sidewall",
                        }
                    )
                ),
                "metal_stop_pair": {
                    "fixed_metal_stop",
                    "plug_metal_stop",
                }.issubset(role_set),
            }
        )
    return rows


def _author_case_fixture(
    stage: Any,
    contract: dict[str, Any],
    case: str,
    Gf: Any,
    PhysxSchema: Any,
    Sdf: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> dict[str, Any]:
    geometry = contract["simulation_geometry"]
    initial_z = float(geometry["initial_plug_face_z_m"])
    yaw_deg = (
        0.0
        if case == "nominal"
        else float(
            contract["acceptance"]["static"]["wrong_yaw_deg_for_negative_case"]
        )
    )
    body = stage.GetPrimAtPath(BODY_PATH)
    nut = stage.GetPrimAtPath(NUT_PATH)
    hinge = UsdPhysics.RevoluteJoint.Get(stage, HINGE_PATH)
    if not body or not nut or not hinge.GetPrim():
        raise RuntimeError("assembly asset is missing body, nut, or nut revolute joint")

    body_xform = UsdGeom.Xformable(body)
    nut_xform = UsdGeom.Xformable(nut)
    body_xform.AddRotateZOp().Set(yaw_deg)
    nut_xform.AddRotateZOp().Set(yaw_deg)

    scene = UsdPhysics.Scene.Define(stage, SCENE_PATH)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(0.0)

    fixture = UsdGeom.Scope.Define(stage, PAIR_ROOT + "/ValidationFixture")
    fixture.GetPrim().SetCustomDataByKey("j599:case", case)
    fixture.GetPrim().SetCustomDataByKey("j599:prePhysicsYawDeg", yaw_deg)
    fixture.GetPrim().SetCustomDataByKey("j599:robotOrHandIncluded", False)
    fixture.GetPrim().SetCustomDataByKey("j599:postStartPoseWritesAllowed", False)

    half_yaw = 0.5 * math.radians(yaw_deg)
    yaw_quaternion = Gf.Quatf(
        math.cos(half_yaw), Gf.Vec3f(0.0, 0.0, math.sin(half_yaw))
    )
    prismatic = UsdPhysics.PrismaticJoint.Define(stage, PRISMATIC_PATH)
    prismatic.CreateAxisAttr("Z")
    prismatic.CreateBody1Rel().SetTargets([body.GetPath()])
    prismatic.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, initial_z))
    prismatic.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    prismatic.CreateLocalRot0Attr(yaw_quaternion)
    prismatic.CreateLocalRot1Attr(Gf.Quatf(1.0))
    prismatic.CreateLowerLimitAttr(-initial_z - 0.00035)
    prismatic.CreateUpperLimitAttr(0.00035)
    prismatic.CreateCollisionEnabledAttr(False)

    lead_m = (
        float(contract["public_interface_geometry"]["thread"]["lead_mm_per_revolution"])
        * 1.0e-3
    )
    ratio = 360.0 / lead_m
    if case == "nominal":
        rack = PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(stage, THREAD_PATH)
        rack.CreateBody0Rel().SetTargets([nut.GetPath()])
        rack.CreateBody1Rel().SetTargets([body.GetPath()])
        rack.CreateHingeRel().SetTargets([Sdf.Path(HINGE_PATH)])
        rack.CreatePrismaticRel().SetTargets([Sdf.Path(PRISMATIC_PATH)])
        rack.CreateRatioAttr(ratio)

        drive = UsdPhysics.DriveAPI.Apply(
            hinge.GetPrim(), UsdPhysics.Tokens.angular
        )
        drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(DRIVE_DAMPING_NM_PER_DEG_S)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(DRIVE_MAX_TORQUE_NM)
        drive_mode = "angular_velocity_thread_proxy"
        linear_force_limit_n = None
    else:
        linear_force_limit_n = DRIVE_MAX_TORQUE_NM * 2.0 * math.pi / lead_m
        drive = UsdPhysics.DriveAPI.Apply(
            prismatic.GetPrim(), UsdPhysics.Tokens.linear
        )
        drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        drive.CreateStiffnessAttr(LINEAR_DRIVE_STIFFNESS_N_PER_M)
        drive.CreateDampingAttr(LINEAR_DRIVE_DAMPING_N_S_PER_M)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(linear_force_limit_n)
        drive_mode = "bounded_linear_key_negative_fixture"

    PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr().Set(0.0)
    return {
        "yaw_deg": yaw_deg,
        "initial_z_m": initial_z,
        "lead_m_per_revolution": lead_m,
        "ratio_degrees_per_meter": ratio,
        "target_angle_deg": 360.0 * initial_z / lead_m,
        "drive_mode": drive_mode,
        "thread_proxy_active": case == "nominal",
        "linear_target_position_m": -initial_z,
        "linear_force_limit_n": linear_force_limit_n,
        "drive": drive,
    }


def _run_case(
    arguments: argparse.Namespace,
    contract: dict[str, Any],
    application: Any,
    output: Path,
    log_records: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics

    asset = arguments.asset.expanduser().resolve()
    context = omni.usd.get_context()
    if context.open_stage(str(asset)) is not True:
        raise RuntimeError(f"Isaac Sim failed to open assembly asset: {asset}")
    for _ in range(3):
        application.update()
    stage = get_current_stage()
    pair = stage.GetPrimAtPath(PAIR_ROOT)
    if not pair:
        raise RuntimeError("assembly root prim is absent after Isaac import")
    expected_root_metadata = {
        "j599:partNumbers": "J599/26FJ35PN,J599/20FJ35SN",
        "j599:contactCount": 128,
        "j599:polarization": "N",
        "j599:hardwareAuthorized": False,
        "j599:hardwareExactFidelity": False,
    }
    root_metadata_readback = {
        key: pair.GetCustomDataByKey(key) for key in expected_root_metadata
    }
    metadata_readback = {
        "root": root_metadata_readback,
        "plug_part_number": stage.GetPrimAtPath(BODY_PATH).GetCustomDataByKey(
            "j599:partNumber"
        ),
        "receptacle_part_number": stage.GetPrimAtPath(
            FIXED_PATH
        ).GetCustomDataByKey("j599:partNumber"),
    }
    identity_metadata_exact = (
        root_metadata_readback == expected_root_metadata
        and metadata_readback["plug_part_number"] == "J599/26FJ35PN"
        and metadata_readback["receptacle_part_number"] == "J599/20FJ35SN"
    )
    if not identity_metadata_exact:
        raise RuntimeError(
            f"assembly identity metadata mismatch: {metadata_readback!r}"
        )

    fixture = _author_case_fixture(
        stage,
        contract,
        arguments.case,
        Gf,
        PhysxSchema,
        Sdf,
        UsdGeom,
        UsdPhysics,
    )
    fixture_stage = output / "fixture_stage_prephysics.usda"
    stage.GetRootLayer().Export(str(fixture_stage))

    dt = 1.0 / PHYSICS_HZ
    World.clear_instance()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=dt,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    world.get_physics_context().set_gravity(0.0)
    world.reset()
    body = stage.GetPrimAtPath(BODY_PATH)
    nut = stage.GetPrimAtPath(NUT_PATH)
    initial_body_position = _world_translation(Usd, UsdGeom, body)
    initial_nut_position = _world_translation(Usd, UsdGeom, nut)
    initial_wrapped = _relative_z_angle(Gf, Usd, UsdGeom, body, nut)
    unwrapped_angle = initial_wrapped
    interface = get_physx_simulation_interface()

    # One zero-drive frame establishes reset/contact evidence.  The subsequent
    # drive and hold durations are fixed and do not branch on measured state.
    world.step(render=False)
    reset_body_position = _world_translation(Usd, UsdGeom, body)
    reset_contacts = _contact_rows(stage, interface, PhysicsSchemaTools, dt)
    if fixture["thread_proxy_active"]:
        fixture["drive"].GetTargetVelocityAttr().Set(DRIVE_SPEED_DEG_S)
    else:
        fixture["drive"].GetTargetPositionAttr().Set(
            fixture["linear_target_position_m"]
        )
    post_start_drive_command_write_count = 1
    post_start_object_pose_write_count = 0

    drive_steps = int(
        round(fixture["target_angle_deg"] / DRIVE_SPEED_DEG_S * PHYSICS_HZ)
    )
    hold_steps = int(
        contract["acceptance"]["isaac_dynamic"]["nominal_hold_steps"]
    )
    trace_path = output / "trace.jsonl"
    contacts_path = output / "contacts.jsonl"
    contact_aggregates: dict[str, dict[str, Any]] = {}
    wrong_key_contact_sample_count = 0
    metal_stop_contact_sample_count = 0
    maximum_hard_penetration_m = 0.0
    maximum_equivalent_contact_force_n = 0.0
    wall_last_heartbeat = time.monotonic()

    def observe(step: int, phase: str, trace_stream: Any, contact_stream: Any) -> float:
        nonlocal unwrapped_angle
        nonlocal wrong_key_contact_sample_count
        nonlocal metal_stop_contact_sample_count
        nonlocal maximum_hard_penetration_m
        nonlocal maximum_equivalent_contact_force_n
        wrapped = _relative_z_angle(Gf, Usd, UsdGeom, body, nut)
        unwrapped_angle = _unwrap(unwrapped_angle, wrapped)
        body_position = _world_translation(Usd, UsdGeom, body)
        nut_position = _world_translation(Usd, UsdGeom, nut)
        rows = _contact_rows(stage, interface, PhysicsSchemaTools, dt)
        for row in rows:
            key = json.dumps(
                [row["collision_roles"], row["collider_paths"]],
                separators=(",", ":"),
                sort_keys=True,
            )
            aggregate = contact_aggregates.setdefault(
                key,
                {
                    "collision_roles": row["collision_roles"],
                    "collider_paths": row["collider_paths"],
                    "active_sample_count": 0,
                    "first_step": step,
                    "last_step": step,
                    "minimum_separation_m": None,
                    "maximum_equivalent_force_n": 0.0,
                    "wrong_key_pair": bool(row["wrong_key_pair"]),
                    "metal_stop_pair": bool(row["metal_stop_pair"]),
                },
            )
            aggregate["active_sample_count"] += 1
            aggregate["last_step"] = step
            separation = row["minimum_separation_m"]
            if separation is not None:
                if (
                    aggregate["minimum_separation_m"] is None
                    or separation < aggregate["minimum_separation_m"]
                ):
                    aggregate["minimum_separation_m"] = separation
                maximum_hard_penetration_m = max(
                    maximum_hard_penetration_m, max(0.0, -separation)
                )
            force = float(row["maximum_equivalent_force_n"])
            aggregate["maximum_equivalent_force_n"] = max(
                float(aggregate["maximum_equivalent_force_n"]), force
            )
            maximum_equivalent_contact_force_n = max(
                maximum_equivalent_contact_force_n, force
            )
            wrong_key_contact_sample_count += int(row["wrong_key_pair"])
            metal_stop_contact_sample_count += int(row["metal_stop_pair"])
            contact_stream.write(
                json.dumps(
                    {"step": step, "phase": phase, **row},
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
        if step == 1 or step % TRACE_STRIDE == 0:
            trace_stream.write(
                json.dumps(
                    {
                        "step": step,
                        "time_s": step * dt,
                        "phase": phase,
                        "body_position_m": list(body_position),
                        "nut_position_m": list(nut_position),
                        "relative_nut_angle_deg": math.degrees(unwrapped_angle),
                        "active_contact_pair_count": len(rows),
                        "post_start_object_pose_write_count": 0,
                    },
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
            trace_stream.flush()
            contact_stream.flush()
        return body_position[2]

    with trace_path.open("x", encoding="utf-8") as trace_stream, contacts_path.open(
        "x", encoding="utf-8"
    ) as contact_stream:
        for step in range(1, drive_steps + 1):
            world.step(render=False)
            observe(step, "drive", trace_stream, contact_stream)
            if time.monotonic() - wall_last_heartbeat >= 60.0:
                heartbeat = {
                    "case": arguments.case,
                    "step": step,
                    "total_steps": drive_steps + hold_steps,
                    "object_pose_writes_after_start": 0,
                }
                (output / "heartbeat.json").write_text(
                    json.dumps(heartbeat, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"J599_ISAAC_HEARTBEAT case={arguments.case} step={step}",
                    flush=True,
                )
                wall_last_heartbeat = time.monotonic()

        if fixture["thread_proxy_active"]:
            fixture["drive"].GetTargetVelocityAttr().Set(0.0)
            post_start_drive_command_write_count += 1
        hold_start_z = _world_translation(Usd, UsdGeom, body)[2]
        hold_positions: list[float] = []
        for offset in range(1, hold_steps + 1):
            step = drive_steps + offset
            world.step(render=False)
            hold_positions.append(observe(step, "hold", trace_stream, contact_stream))

    final_body_position = _world_translation(Usd, UsdGeom, body)
    final_nut_position = _world_translation(Usd, UsdGeom, nut)
    final_wrapped = _relative_z_angle(Gf, Usd, UsdGeom, body, nut)
    unwrapped_angle = _unwrap(unwrapped_angle, final_wrapped)
    axial_travel_m = initial_body_position[2] - final_body_position[2]
    helical_expected_travel_m = (
        fixture["lead_m_per_revolution"]
        * abs(unwrapped_angle - initial_wrapped)
        / (2.0 * math.pi)
    )
    thread_relation_error_m = abs(axial_travel_m - helical_expected_travel_m)
    hold_drift_m = max(
        (abs(value - hold_start_z) for value in hold_positions), default=0.0
    )
    final_face_error_m = abs(
        final_body_position[2]
        - float(contract["simulation_geometry"]["final_plug_face_z_m"])
    )

    relevant_logs = [
        record
        for record in log_records
        if any(
            token in record["message"].lower()
            for token in (
                "solver",
                "physicsusd",
                "physx",
                "joint",
                "collision",
                "error",
                "failed",
                "invalid",
            )
        )
    ]
    solver_errors = [
        record
        for record in log_records
        if "solver" in record["message"].lower()
        and any(
            token in record["message"].lower()
            for token in ("error", "failed", "invalid", "nan")
        )
    ]
    physicsusd_errors = [
        record
        for record in log_records
        if "physicsusd" in record["message"].lower()
        and any(
            token in record["message"].lower()
            for token in ("error", "failed", "invalid")
        )
    ]
    (output / "runtime_log_relevant.json").write_text(
        json.dumps(relevant_logs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    limits = contract["acceptance"]["isaac_dynamic"]
    common_gates = {
        "identity_metadata_readback_exact": identity_metadata_exact,
        "robot_or_hand_prim_count_zero": not any(
            any(
                token in str(prim.GetPath()).lower()
                for token in ("/robot", "/hand", "/finger", "iiwa")
            )
            for prim in stage.Traverse()
        ),
        "initial_pose_preserved_after_reset": abs(
            reset_body_position[2] - fixture["initial_z_m"]
        )
        <= 5.0e-5,
        "post_start_object_pose_write_count_zero": (
            post_start_object_pose_write_count
            == int(limits["post_start_object_pose_write_count"])
        ),
        "solver_error_count_zero": len(solver_errors) == 0,
        "physicsusd_error_count_zero": len(physicsusd_errors) == 0,
        "finite_terminal_state": all(
            math.isfinite(value)
            for value in (
                *final_body_position,
                *final_nut_position,
                unwrapped_angle,
                axial_travel_m,
                thread_relation_error_m,
            )
        ),
    }
    if arguments.case == "nominal":
        case_gates = {
            "final_face_plane_reached": final_face_error_m
            <= float(limits["nominal_case_must_reach_final_face_z_tolerance_m"]),
            "nominal_hold_drift_within_tolerance": hold_drift_m
            <= float(limits["nominal_hold_drift_max_m"]),
            "metal_stop_contact_observed": metal_stop_contact_sample_count > 0,
            "wrong_key_contact_not_observed": wrong_key_contact_sample_count == 0,
            "thread_relation_within_tolerance": thread_relation_error_m
            <= float(limits["thread_relation_error_max_m"]),
        }
    else:
        case_gates = {
            "wrong_key_stopped_above_face_threshold": final_body_position[2]
            >= float(limits["wrong_key_case_must_stop_above_face_z_m"]),
            "wrong_key_collision_observed": wrong_key_contact_sample_count > 0,
            "false_full_assembly_not_reached": final_face_error_m
            > float(limits["nominal_case_must_reach_final_face_z_tolerance_m"]),
            "metal_stop_not_reached": metal_stop_contact_sample_count == 0,
            "thread_proxy_isolated_for_key_negative_case": not fixture[
                "thread_proxy_active"
            ],
            "bounded_linear_force_fixture_used": fixture["linear_force_limit_n"]
            is not None
            and fixture["linear_force_limit_n"] <= 8.0,
        }
    gates = {**common_gates, **case_gates}
    passed = all(gates.values())
    aggregate_rows = sorted(
        contact_aggregates.values(),
        key=lambda item: (item["collision_roles"], item["collider_paths"]),
    )
    return {
        "schema_version": "j599_25_35_isaac_dynamic_case_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "J599-25-35-CONNECTOR-ONLY-MODEL-AND-ISAAC-ASSEMBLY",
        "hypothesis_id": (
            "J35-DYN-H1-NOMINAL-N-KEY-ASSEMBLY"
            if arguments.case == "nominal"
            else "J35-DYN-H5-INDEPENDENT-WRONG-KEY-FORCE-BLOCK"
        ),
        "case": arguments.case,
        "status": "DYNAMIC_PASS" if passed else "DYNAMIC_FAIL",
        "passed": passed,
        "simulation_started": True,
        "fresh_process_required": True,
        "fixture_authored_before_physics": True,
        "fixed_command_schedule": {
            "drive_mode": fixture["drive_mode"],
            "physics_hz": PHYSICS_HZ,
            "drive_duration_s": drive_steps / PHYSICS_HZ,
            "drive_speed_deg_s": (
                DRIVE_SPEED_DEG_S if fixture["thread_proxy_active"] else None
            ),
            "drive_target_rotation_deg_from_lead": (
                fixture["target_angle_deg"]
                if fixture["thread_proxy_active"]
                else None
            ),
            "drive_step_count_from_lead": drive_steps,
            "drive_damping_nm_per_deg_s": (
                DRIVE_DAMPING_NM_PER_DEG_S
                if fixture["thread_proxy_active"]
                else None
            ),
            "drive_max_torque_nm": (
                DRIVE_MAX_TORQUE_NM if fixture["thread_proxy_active"] else None
            ),
            "maximum_thread_equivalent_axial_force_n": (
                DRIVE_MAX_TORQUE_NM
                * 2.0
                * math.pi
                / fixture["lead_m_per_revolution"]
            ),
            "linear_target_position_m": (
                fixture["linear_target_position_m"]
                if not fixture["thread_proxy_active"]
                else None
            ),
            "linear_stiffness_n_per_m": (
                LINEAR_DRIVE_STIFFNESS_N_PER_M
                if not fixture["thread_proxy_active"]
                else None
            ),
            "linear_damping_n_s_per_m": (
                LINEAR_DRIVE_DAMPING_N_S_PER_M
                if not fixture["thread_proxy_active"]
                else None
            ),
            "linear_force_limit_n": fixture["linear_force_limit_n"],
            "thread_proxy_active": fixture["thread_proxy_active"],
            "hold_steps": hold_steps,
            "post_start_drive_command_write_count": post_start_drive_command_write_count,
            "feedback_or_contact_truth_used_to_change_schedule": False,
        },
        "inputs": {
            "asset": str(asset),
            "asset_sha256": _sha256(asset),
            "contract": str(arguments.contract.expanduser().resolve()),
            "contract_sha256": _sha256(arguments.contract.expanduser().resolve()),
            "build_plan": str(BUILD_PLAN),
            "build_plan_sha256": _sha256(BUILD_PLAN),
            "fixture_stage": str(fixture_stage),
            "fixture_stage_sha256": _sha256(fixture_stage),
        },
        "identity_metadata_readback": metadata_readback,
        "prephysics_initial_yaw_deg": fixture["yaw_deg"],
        "initial_body_position_m": list(initial_body_position),
        "initial_nut_position_m": list(initial_nut_position),
        "reset_body_position_after_zero_drive_frame_m": list(reset_body_position),
        "reset_contact_pair_count": len(reset_contacts),
        "final_body_position_m": list(final_body_position),
        "final_nut_position_m": list(final_nut_position),
        "final_face_error_m": final_face_error_m,
        "axial_travel_m": axial_travel_m,
        "relative_nut_angle_deg": math.degrees(unwrapped_angle - initial_wrapped),
        "helical_expected_travel_m": helical_expected_travel_m,
        "thread_relation_error_m": thread_relation_error_m,
        "thread_relation_applicable": fixture["thread_proxy_active"],
        "hold_start_body_z_m": hold_start_z,
        "hold_drift_m": hold_drift_m,
        "wrong_key_contact_sample_count": wrong_key_contact_sample_count,
        "metal_stop_contact_sample_count": metal_stop_contact_sample_count,
        "maximum_hard_penetration_m": maximum_hard_penetration_m,
        "maximum_equivalent_contact_force_n": maximum_equivalent_contact_force_n,
        "contact_aggregates": aggregate_rows,
        "solver_error_count": len(solver_errors),
        "physicsusd_error_count": len(physicsusd_errors),
        "solver_errors": solver_errors,
        "physicsusd_errors": physicsusd_errors,
        "gates": gates,
        "object_pose_write_after_physics_start_count": (
            post_start_object_pose_write_count
        ),
        "claims": {
            "result_claim": limits["result_claim"],
            "connector_only_simulation_dynamic_pass": passed,
            "old_project_formal_gate_claimed": False,
            "robot_or_hand_assembly_claimed": False,
            "real_hardware_assembly_success_claimed": False,
            "hardware_authorized": False,
            "manufacturer_exact_fidelity": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(argv)
    arguments.asset = arguments.asset.expanduser().resolve()
    arguments.contract = arguments.contract.expanduser().resolve()
    output = arguments.output_dir.expanduser().resolve()
    for path in (arguments.asset, arguments.contract, BUILD_PLAN):
        if not path.is_file():
            raise FileNotFoundError(path)
        if not _inside_model_root(path):
            raise ValueError(f"input must remain inside isolated model root: {path}")
    if not _inside_model_root(output):
        raise ValueError("output directory must remain inside isolated model root")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    contract = _load_json(arguments.contract)
    if contract["scope"] != {
        **contract["scope"],
        "robot_or_hand_included": False,
        "old_model_modification_allowed": False,
        "hardware_authorized": False,
        "hardware_exact_fidelity": False,
    }:
        raise RuntimeError("contract isolation/truth flags are not fail-closed")
    plan = _load_json(BUILD_PLAN)
    hypothesis = next(
        item
        for item in plan["dynamic_validation_hypotheses"]
        if item["case"] == arguments.case
    )
    authorization = {
        "schema_version": "j599_25_35_dynamic_run_authorization_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": arguments.case,
        "hypothesis": hypothesis,
        "source_plan": str(BUILD_PLAN),
        "source_plan_sha256": _sha256(BUILD_PLAN),
        "asset_sha256_before_run": _sha256(arguments.asset),
        "contract_sha256_before_run": _sha256(arguments.contract),
        "source_asset_write_allowed": False,
        "robot_or_hand_included": False,
    }
    (output / "run_authorization.json").write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if arguments.kit_portable_root is None:
        portable = Path(
            tempfile.mkdtemp(prefix=f"j599-{arguments.case}-", dir="/tmp")
        )
    else:
        portable = arguments.kit_portable_root.expanduser().resolve()
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
            "active_gpu": 0,
            "physics_gpu": 0,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        }
    )
    import carb.logging

    log_records: list[dict[str, Any]] = []
    logging = carb.logging.acquire_logging()

    def on_log(source: str, level: int, filename: str, line: int, message: str) -> None:
        log_records.append(
            {
                "source": str(source),
                "level": int(level),
                "filename": str(filename),
                "line": int(line),
                "message": str(message),
            }
        )

    handle = logging.add_logger(on_log)
    exit_code = 1
    report: dict[str, Any]
    asset_hash_before = _sha256(arguments.asset)
    try:
        report = _run_case(
            arguments,
            contract,
            application,
            output,
            log_records,
        )
        report["source_asset_sha256_after_run"] = _sha256(arguments.asset)
        report["source_asset_unchanged"] = (
            report["source_asset_sha256_after_run"] == asset_hash_before
        )
        report["runner_sha256"] = _sha256(Path(__file__))
        report["kit_portable_root"] = str(portable)
        if not report["source_asset_unchanged"]:
            raise RuntimeError("source assembly asset changed during validation")
        exit_code = 0 if report["passed"] else 3
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_version": "j599_25_35_isaac_dynamic_case_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": "J599-25-35-CONNECTOR-ONLY-MODEL-AND-ISAAC-ASSEMBLY",
            "hypothesis_id": hypothesis["id"],
            "case": arguments.case,
            "status": "ERROR",
            "passed": False,
            "simulation_started": True,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "source_asset_sha256_before_run": asset_hash_before,
            "source_asset_sha256_after_run": _sha256(arguments.asset),
            "object_pose_write_after_physics_start_count": 0,
            "claims": {
                "connector_only_simulation_dynamic_pass": False,
                "old_project_formal_gate_claimed": False,
                "robot_or_hand_assembly_claimed": False,
                "real_hardware_assembly_success_claimed": False,
                "hardware_authorized": False,
                "manufacturer_exact_fidelity": False,
            },
        }
    finally:
        logging.remove_logger(handle)
        (output / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    print(
        json.dumps(
            {
                "case": report.get("case"),
                "status": report.get("status"),
                "passed": report.get("passed"),
                "final_body_z_m": (report.get("final_body_position_m") or [None] * 3)[2],
                "thread_relation_error_m": report.get("thread_relation_error_m"),
                "wrong_key_contact_sample_count": report.get(
                    "wrong_key_contact_sample_count"
                ),
                "metal_stop_contact_sample_count": report.get(
                    "metal_stop_contact_sample_count"
                ),
                "error": report.get("error"),
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
