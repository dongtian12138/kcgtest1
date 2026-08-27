#!/usr/bin/env python3
"""Isolate Isaac/PhysX contact reporting with two primitive box rigid bodies."""
from __future__ import annotations
import argparse, json, math, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/carts_v2/contactopt_1488_fast6h/primitive_contact_report_isolation/result.json"
DT, SIDE, INITIAL_GAP, SPEED = 1.0 / 120.0, 0.04, 0.10, 0.20
MOVING, FIXED = "/World/MovingBox", "/World/FixedBox"
def _host(value) -> np.ndarray:
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw", "sensor"), default="raw")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    return parser.parse_args()


def _run(mode: str, backend: str):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0})
    import carb
    import omni.physics.core
    import torch
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.version import get_version
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics

    use_gpu = backend == "gpu"
    device = "cuda:0" if use_gpu else "cpu"
    World.clear_instance()
    SimulationManager.set_physics_sim_device(device)
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=DT,
        backend="torch",
        device=device,
        sim_params={
            "use_gpu_pipeline": use_gpu,
            "gpu_found_lost_aggregate_pairs_capacity": 8192,
            "gpu_total_aggregate_pairs_capacity": 16384,
        },
    )
    stage, context = get_current_stage(), world.get_physics_context()
    context.set_gravity(0.0)

    def make_box(path: str, x: float, kinematic: bool) -> None:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(SIDE)
        cube.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0))
        collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision.CreateCollisionEnabledAttr().Set(True)
        body = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        body.CreateRigidBodyEnabledAttr().Set(True)
        body.CreateKinematicEnabledAttr().Set(kinematic)
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr().Set(1.0)
        PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim()).CreateThresholdAttr().Set(0.0)

    make_box(MOVING, -(SIDE + INITIAL_GAP), False)
    make_box(FIXED, 0.0, True)
    sensor = None
    sensor_path = MOVING + "/ContactSensor"
    if mode == "sensor":
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor

        sensor = ContactSensor(Contact.create(sensor_path, min_threshold=0.0,
            max_threshold=10_000_000.0, radius=-1.0, translations=[[0.0, 0.0, 0.0]]))

    moving = world.scene.add(SingleRigidPrim(MOVING, "primitive_moving_box"))
    fixed = world.scene.add(SingleRigidPrim(FIXED, "primitive_fixed_box"))
    decoder = PhysicsSchemaTools.intToSdfPath

    def path_record(value) -> dict:
        encoded = int(value)
        try:
            return {"encoded": encoded, "path": str(decoder(encoded)), "decode_error": None}
        except Exception as exc:
            return {"encoded": encoded, "path": None, "decode_error": f"{type(exc).__name__}: {exc}"}

    def vector(value) -> list[float]:
        if isinstance(value, dict):
            return [float(value[key]) for key in ("x", "y", "z")]
        return [float(item) for item in value]

    def packet(headers, data, channel: str, token: int) -> dict:
        rows = []
        for header in headers:
            offset, count = int(header.contact_data_offset), int(header.num_contact_data)
            contacts = []
            for item in data[offset:offset + count]:
                contacts.append({"position": vector(item.position), "normal": vector(item.normal),
                    "impulse": vector(item.impulse), "separation": float(item.separation),
                    "face_index0": int(item.face_index0) if hasattr(item, "face_index0") else None,
                    "face_index1": int(item.face_index1) if hasattr(item, "face_index1") else None})
            rows.append({"type": str(header.type), "actor0": path_record(header.actor0),
                "actor1": path_record(header.actor1), "collider0": path_record(header.collider0),
                "collider1": path_record(header.collider1), "stage_id": int(header.stage_id),
                "contact_data_offset": offset, "num_contact_data": count, "contact_data": contacts})
        return {"channel": channel, "command_step_token": token, "header_count": len(headers),
            "contact_data_vector_count": len(data), "headers": rows}

    token = -1
    full_events, basic_events, step_polls, step_callbacks = [], [], [], []
    project_polls = []
    simulation = get_physx_simulation_interface()
    current_simulation = omni.physics.core.get_physics_simulation_interface()

    def on_full(headers, data, _friction) -> None:
        full_events.append(packet(headers, data, "full_callback", token))

    def on_basic(headers, data, _friction) -> None:
        basic_events.append(packet(headers, data, "isaac6_core_contact_callback", token))

    def on_step(step_dt: float, _context) -> None:
        headers, data, _ = simulation.get_full_contact_report()
        step_callbacks.append({"command_step_token": token, "dt_s": float(step_dt)})
        step_polls.append(packet(headers, data, "physics_step_callback_poll", token))
        if mode == "sensor":
            from run_opposition60_local_contact import _contacts

            project_polls.append({"command_step_token": token,
                "result": _contacts(simulation, decoder, {"hand": MOVING, "object": FIXED})})

    subscriptions = [simulation.subscribe_full_contact_report_events(on_full),
        current_simulation.subscribe_physics_contact_report_events(on_basic),
        current_simulation.subscribe_physics_on_step_events(False, 0, on_step)]
    api_before = {path: {"applied": stage.GetPrimAtPath(path).HasAPI(PhysxSchema.PhysxContactReportAPI),
        "threshold": float(PhysxSchema.PhysxContactReportAPI(stage.GetPrimAtPath(path)).GetThresholdAttr().Get())}
        for path in (MOVING, FIXED)}
    lifecycle = {"shapes_and_bodies_created_before_reset": True,
        "contact_report_api_before_reset": api_before, "subscriptions_before_reset": True,
        "held_subscription_count": len(subscriptions), "first_initialization_operation": "world.reset"}
    world.reset()
    full_events.clear(); basic_events.clear(); step_polls.clear(); step_callbacks.clear(); project_polls.clear()
    moving._rigid_prim_view.set_velocities(
        torch.tensor([[SPEED, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device))
    start_x = float(_host(moving.get_world_pose()[0]).reshape(-1)[0])
    view = SimulationManager.get_physics_simulation_view()

    def offsets(path: str) -> dict:
        body_view = view.create_rigid_body_view([path])
        contact = _host(body_view.get_contact_offsets()).reshape(-1)
        rest = _host(body_view.get_rest_offsets()).reshape(-1)
        return {"shape_count": int(body_view.max_shapes), "contact_offset_m": contact.tolist(),
            "rest_offset_m": rest.tolist()}

    runtime_offsets = {path: offsets(path) for path in (MOVING, FIXED)}
    combined = max(runtime_offsets[MOVING]["contact_offset_m"]) + max(runtime_offsets[FIXED]["contact_offset_m"])
    samples, response_seen, response_step = [], False, None
    for index in range(120):
        token = index
        before_full, before_basic = len(full_events), len(basic_events)
        before_poll, before_project = len(step_polls), len(project_polls)
        world.step(render=False)
        headers, data, _ = simulation.get_full_contact_report()
        outside_poll = packet(headers, data, "after_world_step_poll", token)
        position = _host(moving.get_world_pose()[0]).reshape(-1)
        velocity = _host(moving.get_linear_velocity()).reshape(-1)
        gap = float(-SIDE / 2.0 - (position[0] + SIDE / 2.0))
        expected_x = start_x + SPEED * DT * (index + 1)
        deviation = float(expected_x - position[0])
        solver_response = abs(float(velocity[0]) - SPEED) > 1e-3 or deviation > 2e-5
        if solver_response and not response_seen:
            response_seen, response_step = True, index
        phase = "C_SOLVER_CONTACT" if response_seen or gap <= 0.0 else (
            "B_POSITIVE_GAP_WITHIN_OFFSETS" if gap < combined else "A_OUTSIDE_OFFSETS")
        sensor_row = None
        if sensor is not None:
            raw = sensor.get_raw_data()
            reading = sensor.get_sensor_reading()
            sensor_row = {"raw": [{"body0": path_record(item["body0"]),
                "body1": path_record(item["body1"]), "position": vector(item["position"]),
                "normal": vector(item["normal"]), "impulse": vector(item["impulse"]),
                "time_s": float(item["time"]), "dt_s": float(item["dt"])} for item in raw],
                "filtered": {"is_valid": bool(reading.is_valid), "in_contact": bool(reading.in_contact),
                    "value_n": float(reading.value), "time_s": float(reading.time)}}
        samples.append({"step": index, "phase": phase, "geometry_gap_m": gap,
            "combined_contact_offset_m": combined, "moving_position_m": position.tolist(),
            "moving_linear_velocity_m_s": velocity.tolist(), "no_collision_predicted_x_m": expected_x,
            "pose_deviation_from_no_collision_m": deviation, "solver_response": solver_response,
            "full_callback": full_events[before_full:], "basic_callback": basic_events[before_basic:],
            "physics_step_callback_poll": step_polls[before_poll:], "after_world_step_poll": outside_poll,
            "project_aggregation": project_polls[before_project:], "contact_sensor": sensor_row})
        if response_seen and index >= int(response_step) + 8:
            break

    context_snapshot = {"requested_backend": backend, "actual_data_backend": str(world.backend),
        "world_device": str(world.device), "physics_context_device": str(context.device),
        "gpu_sim": bool(context.use_gpu_sim), "gpu_pipeline": bool(context.use_gpu_pipeline),
        "gpu_dynamics_enabled": bool(context.is_gpu_dynamics_enabled()),
        "broadphase_type": str(context.get_broadphase_type()), "physics_dt_s": DT,
        "disable_contact_processing_setting": carb.settings.get_settings().get(SETTING_DISABLE_CONTACT_PROCESSING),
        "legacy_physx_attached_stage_id": int(simulation.get_attached_stage()),
        "isaac6_core_attached_stage_id": int(current_simulation.get_attached_stage()),
        "isaac_version": list(get_version()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    collision = {path: {"collision_enabled": bool(UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get()),
        "filtered_pairs": [str(value) for value in stage.GetPrimAtPath(path).GetRelationship("physics:filteredPairs").GetTargets()],
        "rigid_body_parent": path, "rigid_body_type": "kinematic" if path == FIXED else "dynamic"}
        for path in (MOVING, FIXED)}
    collision["collision_groups"] = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.CollisionGroup)]
    api_after = {path: {"applied": stage.GetPrimAtPath(path).HasAPI(PhysxSchema.PhysxContactReportAPI),
        "threshold": float(PhysxSchema.PhysxContactReportAPI(stage.GetPrimAtPath(path)).GetThresholdAttr().Get())}
        for path in (MOVING, FIXED)}
    lifecycle["contact_report_api_after_reset"] = api_after
    lifecycle["physics_step_callback_count"] = len(step_callbacks)
    lifecycle["recorded_step_count"] = len(samples)
    result = {"mode": mode, "backend": backend, "runtime": context_snapshot,
        "lifecycle": lifecycle, "collision_and_body_audit": collision,
        "runtime_offsets": runtime_offsets, "motion_command": {"method": "controlled_initial_velocity",
            "set_after_reset_before_first_recorded_step": True, "post_initialization_pose_writes": 0,
            "speed_m_s": SPEED, "initial_geometry_gap_m": INITIAL_GAP}, "samples": samples}
    return result, app


def _pair_header(header: dict) -> bool:
    paths = [header[key]["path"] for key in ("actor0", "collider0", "actor1", "collider1")]
    return MOVING in paths and FIXED in paths


def _raw_gate(run: dict) -> dict:
    a = [row for row in run["samples"] if row["phase"] == "A_OUTSIDE_OFFSETS"]
    b = [row for row in run["samples"] if row["phase"] == "B_POSITIVE_GAP_WITHIN_OFFSETS"]
    c = [row for row in run["samples"] if row["phase"] == "C_SOLVER_CONTACT"]
    headers = [header for row in c for channel in ("full_callback", "basic_callback")
        for event in row[channel] for header in event["headers"] if _pair_header(header)]
    data = [item for header in headers for item in header["contact_data"]]
    a_data = [header for row in a for channel in ("full_callback", "basic_callback")
        for event in row[channel] for header in event["headers"] if _pair_header(header) and header["num_contact_data"] > 0]
    finite = any(math.isfinite(item["separation"]) and all(math.isfinite(v) for v in item["impulse"]) for item in data)
    gate = {"stage_a_step_count": len(a), "stage_b_step_count": len(b), "stage_c_step_count": len(c),
        "solver_response": any(row["solver_response"] for row in c), "correct_pair_callback_header": bool(headers),
        "callback_num_contact_data_positive": bool(data), "finite_separation_and_impulse": finite,
        "actor_and_collider_paths_decoded": bool(headers) and all(header[key]["path"] is not None
            for header in headers for key in ("actor0", "actor1", "collider0", "collider1")),
        "stage_a_has_no_contact_data": not a_data, "stage_a_geometry_valid": len(a) >= 5,
        "stage_b_geometry_valid": bool(b) and all(0.0 < row["geometry_gap_m"] < row["combined_contact_offset_m"] for row in b)}
    gate["pass"] = all(gate[key] for key in ("solver_response", "correct_pair_callback_header",
        "callback_num_contact_data_positive", "finite_separation_and_impulse", "actor_and_collider_paths_decoded",
        "stage_a_has_no_contact_data", "stage_a_geometry_valid", "stage_b_geometry_valid"))
    gate["status"] = "PRIMITIVE_RAW_CONTACT_REPORT_PASS" if gate["pass"] else (
        "PRIMITIVE_SOLVER_RESPONSE_WITHOUT_CONTACT_REPORT" if gate["solver_response"] and not headers and
        not any(row["after_world_step_poll"]["header_count"] for row in c) else "UNRESOLVED")
    gate["poll_timing"] = "EVENT_PRESENT_POLL_EMPTY_TIMING_SEMANTICS" if headers and not any(
        row["after_world_step_poll"]["header_count"] for row in c) else "NO_EVENT_POLL_TIMING_DIFFERENCE"
    return gate


def main() -> None:
    args = _arguments()
    prior = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    if args.mode == "sensor" and not prior.get("gpu_raw", {}).get("raw_gate", {}).get("pass"):
        raise RuntimeError("sensor layer is forbidden until the GPU primitive raw gate passes")
    run, app = _run(args.mode, args.backend)
    gate = _raw_gate(run)
    report = prior or {
        "schema_version": "physx_primitive_contact_report_isolation_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "LOCAL_ISAAC_DIAGNOSTIC_ONLY_NOT_GRASP_OR_HARDWARE_EVIDENCE",
        "hardware_authorized": False, "formal_dynamic_pass": False, "research_dynamic_pass": False,
        "q09_a13": {"candidate_disposition": "REJECTED", "candidate_failure_class": "NON_TASK_GEOMETRY_FIRST",
            "task_first_margin_m": -0.0005071808379559516}, "contact_pipeline_status": "CONTACT_TELEMETRY_UNVERIFIED",
        "contact_root_cause": "UNRESOLVED", "root_cause_proven": False,
        "offsets_modified": False, "candidate_selection_allowed": False, "q09_replayed": False}
    key = f"{args.backend}_{args.mode}"
    if key in report:
        report[key + "_before_interface_fix"] = report[key]
    report[key] = {"run": run, "raw_gate": gate}
    if args.backend == "gpu" and args.mode == "raw":
        report["primitive_raw_status"] = gate["status"]
        if gate["pass"]:
            report.update(contact_root_cause="UNRESOLVED", root_cause_proven=False)
        elif gate["status"] == "PRIMITIVE_SOLVER_RESPONSE_WITHOUT_CONTACT_REPORT":
            report.update(contact_root_cause=gate["status"], root_cause_proven=True)
    elif args.backend == "cpu" and args.mode == "raw":
        gpu_gate = report.get("gpu_raw", {}).get("raw_gate", {})
        if gate["pass"] and gpu_gate.get("status") == "PRIMITIVE_SOLVER_RESPONSE_WITHOUT_CONTACT_REPORT":
            classification = "GPU_CONTACT_REPORT_CONFIGURATION_OR_BACKEND_SPECIFIC_FAILURE"
            report.update(cpu_gpu_classification=classification,
                contact_root_cause=classification, root_cause_proven=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "run_key": key, "raw_gate": gate}, indent=2), flush=True)
    app.close()


if __name__ == "__main__":
    main()
