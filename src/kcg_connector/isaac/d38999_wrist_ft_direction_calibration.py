#!/usr/bin/env python3

"""Calibrate hand2arm reaction-wrench sign and lever arm on the V2 payload."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback

import numpy as np
import yaml

from kcg_connector.d38999_insert_proxy_v2 import load_insert_proxy_v2
from kcg_connector.d38999_tabletop_pick import (
    iiwa14_grasp_tcp_transform,
    load_d38999_tabletop_pick_config,
    verify_d38999_pick_dependencies,
)
from kcg_connector.virtual_wrist_ft_runtime import (
    VirtualWristFtMonitor,
    column_rotation_from_gf_matrix3d,
    load_virtual_wrist_ft_monitor_config,
    reaction_row_index,
    transform_wrench_to_task,
    verify_virtual_wrist_ft_monitor_inputs,
)

from d38999_iiwa_hand_v2_scene import (
    ARM_NAMES,
    ARTICULATION_ROOT,
    HAND_BASE,
    PLUG_BODY,
    author_scene,
    topology_report,
)


ACTIVE_HAND_NAMES = ("f1j1", "f1j2", "f2j1", "f3j2")


def _arguments(repository):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_capture_sweep_v1.yaml"
        ),
    )
    parser.add_argument(
        "--asset",
        default=str(
            repository
            / "artifacts/kcg_connector/isaac/d38999_insert_proxy_v2.usda"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_insert_proxy_v2/wrist_ft_direction_calibration_v1"
        ),
    )
    result = parser.parse_args()
    if not result.run:
        parser.error("direction calibration requires --run")
    return result


def _rotation_from_prim(prim, Gf, Usd, UsdGeom):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform(matrix)
    rotation = column_rotation_from_gf_matrix3d(
        np.asarray(Gf.Matrix3d(transform.GetRotation()), dtype=np.float64)
    )
    return (
        np.asarray(transform.GetTranslation(), dtype=np.float64),
        rotation,
    )


def main():
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository)
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    document = yaml.safe_load(Path(arguments.config).read_text())
    proxy = load_insert_proxy_v2(repository / document["proxy_config"])
    pick_path = repository / document["robot_pick_config"]
    pick = load_d38999_tabletop_pick_config(pick_path)
    dependencies = verify_d38999_pick_dependencies(
        pick, pick_path, repository
    )
    wrist_config = load_virtual_wrist_ft_monitor_config(
        repository / document["wrist_ft_monitor_config"]
    )
    wrist_inputs = verify_virtual_wrist_ft_monitor_inputs(
        wrist_config, repository
    )
    if wrist_inputs["robot_asset"] != dependencies["robot_asset"]:
        raise RuntimeError("calibration and pick robot assets differ")
    base_arm = np.asarray(
        document["nominal_scene"]["initial_arm_rad"], dtype=np.float64
    )
    latch_offset = float(
        document["nominal_scene"]["latch_tcp_to_plug_body_m"]
    )

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": not arguments.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    passed = False
    report = {
        "schema_version": "kcg_d38999_wrist_ft_direction_calibration_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_semantics": "parent_on_child",
        "raw_frame": "hand2arm_child_joint_frame",
        "raw_order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        "quasistatic_relation": (
            "w_environment_on_tool = -(w_parent_on_child_raw - "
            "no_contact_baseline)"
        ),
        "controller_inputs_used": [],
        "object_truth_used_for_control": False,
        "contact_normal_used": False,
        "collider_identity_used": False,
        "cases": [],
        "passed": False,
    }
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim, SingleArticulation
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
        )
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        rate = int(document["experiment"]["physics_rate_hz"])
        dt = 1.0 / rate
        global_step = 0

        def build_scene(arm, include_payload):
            World.clear_instance()
            omni.usd.get_context().new_stage()
            app.update()
            world = World(
                stage_units_in_meters=1.0,
                physics_dt=dt,
                rendering_dt=1.0 / 60.0,
                backend="numpy",
                device="cpu",
            )
            stage = get_current_stage()
            tcp = np.asarray(
                iiwa14_grasp_tcp_transform(tuple(float(v) for v in arm))
            )
            initial_tcp = np.asarray(iiwa14_grasp_tcp_transform((0.0,) * 7))
            author = author_scene(
                stage=stage,
                robot_asset=dependencies["robot_asset"],
                v2_asset=Path(arguments.asset),
                arm_rad=arm,
                tcp_transform=tcp,
                initial_tcp_transform=initial_tcp,
                proxy=proxy,
                add_reference_to_stage=add_reference_to_stage,
                Gf=Gf,
                Sdf=Sdf,
                UsdGeom=UsdGeom,
                UsdPhysics=UsdPhysics,
                include_payload=include_payload,
                latch_tcp_to_body_m=latch_offset,
                # Eight centimetres keeps calibration loads contact-free while
                # leaving the Plug at exactly the nominal latch transform.
                preinsert_gap_m=0.080 if include_payload else None,
            )
            robot = world.scene.add(
                SingleArticulation(
                    prim_path=ARTICULATION_ROOT,
                    name="d38999_v2_calibration_handarm",
                )
            )
            world.reset()
            if not robot.handles_initialized:
                raise RuntimeError("iiwa-hand articulation did not initialize")
            world.get_physics_context().set_gravity(-9.81)
            dof_map = {name: index for index, name in enumerate(robot.dof_names)}
            required = set(ARM_NAMES + ACTIVE_HAND_NAMES)
            if not required.issubset(dof_map):
                raise RuntimeError(
                    f"missing controlled DOFs: {sorted(required-set(dof_map))}"
                )
            arm_indices = np.asarray(
                [dof_map[name] for name in ARM_NAMES], dtype=np.int32
            )
            hand_indices = np.asarray(
                [dof_map[name] for name in ACTIVE_HAND_NAMES], dtype=np.int32
            )
            controlled = np.concatenate((arm_indices, hand_indices))
            target = np.concatenate(
                (np.asarray(arm), np.asarray((1.0, 0.0, 0.0, 0.0)))
            ).astype(np.float32)
            controller = robot.get_articulation_controller()
            kps = np.zeros(robot.num_dof, dtype=np.float32)
            kds = np.zeros(robot.num_dof, dtype=np.float32)
            kps[arm_indices] = pick.robot.arm_stiffness
            kds[arm_indices] = pick.robot.arm_damping
            kps[hand_indices] = pick.robot.hand_stiffness
            kds[hand_indices] = pick.robot.hand_damping
            controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
            metadata = dict(robot._articulation_view._metadata.joint_indices)
            row = reaction_row_index(metadata, wrist_config)
            raw = np.asarray(
                robot.get_measured_joint_forces(
                    joint_indices=np.asarray([row], dtype=np.int32)
                )
            )
            if raw.shape != (1, 6):
                raise RuntimeError("hand2arm selected wrench is not 1x6")
            sensor_prim = stage.GetPrimAtPath(HAND_BASE)
            if not sensor_prim.IsValid():
                raise RuntimeError("handbase sensor prim is missing")
            body_view = None
            topology = None
            if include_payload:
                body_view = RigidPrim(
                    prim_paths_expr=PLUG_BODY,
                    name="d38999_v2_calibration_plug_view",
                    reset_xform_properties=False,
                )
                body_view.initialize()
                if not body_view.is_physics_handle_valid():
                    raise RuntimeError("PlugBody physics view is invalid")
                topology = topology_report(stage, UsdPhysics)
                if topology["world_to_plug_fixed_joints"]:
                    raise RuntimeError("world-to-Plug fixed joint is forbidden")
            return {
                "world": world,
                "stage": stage,
                "robot": robot,
                "row": row,
                "sensor_prim": sensor_prim,
                "body_view": body_view,
                "controlled": controlled,
                "target": target,
                "authoring": author,
                "topology": topology,
                "arm_indices": arm_indices,
            }

        def raw_sample(scene):
            value = np.asarray(
                scene["robot"].get_measured_joint_forces(
                    joint_indices=np.asarray([scene["row"]], dtype=np.int32)
                ),
                dtype=np.float64,
            )
            if value.shape != (1, 6) or not np.all(np.isfinite(value)):
                raise RuntimeError("invalid hand2arm reaction sample")
            return value[0]

        orientation_reports = []
        for orientation_name, q7_delta in (
            ("NOMINAL_ORIENTATION", 0.0),
            ("Q7_PLUS_90_DEG", math.pi / 2.0),
        ):
            arm = base_arm.copy()
            arm[6] += q7_delta
            tcp = np.asarray(
                iiwa14_grasp_tcp_transform(tuple(float(v) for v in arm))
            )
            task_rotation = tcp[:3, :3]
            task_origin = tcp[:3, 3] + task_rotation @ np.asarray(
                (0.0, 0.0, latch_offset)
            )

            empty = build_scene(arm, False)
            monitor = VirtualWristFtMonitor(
                wrist_config,
                reaction_row=empty["row"],
                task_origin_world=task_origin,
                task_z_axis_world=task_rotation[:, 2],
            )

            def move_to_calibration_pose(scene):
                start = np.asarray(
                    scene["robot"].get_joint_positions(), dtype=np.float64
                )[scene["controlled"]]
                steps = 960
                for index in range(1, steps + 1):
                    fraction = index / steps
                    blend = fraction * fraction * fraction * (
                        10.0 + fraction * (-15.0 + 6.0 * fraction)
                    )
                    command = start + blend * (
                        scene["target"].astype(np.float64) - start
                    )
                    scene["robot"].apply_action(
                        ArticulationAction(
                            joint_positions=command.astype(np.float32),
                            joint_indices=scene["controlled"],
                        )
                    )
                    scene["world"].step(render=arguments.gui)
                for _ in range(120):
                    scene["robot"].apply_action(
                        ArticulationAction(
                            joint_positions=scene["target"],
                            joint_indices=scene["controlled"],
                        )
                    )
                    scene["world"].step(render=arguments.gui)
                measured = np.asarray(
                    scene["robot"].get_joint_positions(), dtype=np.float64
                )[scene["arm_indices"]]
                scene["initial_arm_error_rad"] = float(
                    np.max(np.abs(measured - arm))
                )

            move_to_calibration_pose(empty)

            def step_scene(scene, phase, applied=None, samples=None):
                nonlocal global_step
                scene["robot"].apply_action(
                    ArticulationAction(
                        joint_positions=scene["target"],
                        joint_indices=scene["controlled"],
                    )
                )
                if applied is not None:
                    force_a = np.asarray(applied["force_a"])
                    moment_a = np.asarray(applied["moment_a"])
                    offset_a = np.asarray(applied["offset_a"])
                    body_position = np.asarray(
                        scene["body_view"].get_world_poses()[0][0],
                        dtype=np.float64,
                    )
                    scene["body_view"].apply_forces_and_torques_at_pos(
                        forces=np.asarray(
                            [task_rotation @ force_a], dtype=np.float32
                        ),
                        torques=np.asarray(
                            [task_rotation @ moment_a], dtype=np.float32
                        ),
                        positions=np.asarray(
                            [body_position + task_rotation @ offset_a],
                            dtype=np.float32,
                        ),
                        is_global=True,
                    )
                scene["world"].step(render=arguments.gui)
                global_step += 1
                raw = raw_sample(scene)
                sensor_position, sensor_rotation = _rotation_from_prim(
                    scene["sensor_prim"], Gf, Usd, UsdGeom
                )
                observed = monitor.observe(
                    raw,
                    global_step=global_step,
                    runtime_phase=phase,
                    sensor_position_world=sensor_position,
                    sensor_rotation_world=sensor_rotation,
                )
                if samples is not None:
                    samples.append((raw.copy(), observed))

            empty_samples = []
            for _ in range(wrist_config.home_tare_window_steps):
                step_scene(empty, "initial_settle", samples=empty_samples)
            home_baseline = monitor.capture_home_tare()

            loaded = build_scene(arm, True)
            if loaded["row"] != empty["row"]:
                raise RuntimeError("hand2arm reaction row changed with payload")
            move_to_calibration_pose(loaded)
            loaded_samples = []
            for _ in range(wrist_config.payload_baseline_window_steps):
                step_scene(
                    loaded,
                    "unsupported_final_hold",
                    samples=loaded_samples,
                )
            payload_baseline = monitor.capture_payload_baseline()
            baseline_raw = np.mean(
                np.asarray([item[0] for item in loaded_samples[-60:]]), axis=0
            )
            empty_raw = np.mean(
                np.asarray([item[0] for item in empty_samples[-60:]]), axis=0
            )
            payload_gravity_delta = baseline_raw - empty_raw

            cases = []
            case_specs = []
            force_magnitude = 2.0
            torque_magnitude = 0.20
            for axis, label in enumerate(("X", "Y", "Z")):
                for sign, prefix in ((1.0, "+"), (-1.0, "-")):
                    force = np.zeros(3)
                    force[axis] = sign * force_magnitude
                    case_specs.append(
                        (f"{prefix}{label}", force, np.zeros(3), np.zeros(3))
                    )
            for axis, label in enumerate(("Mx", "My", "Mz")):
                moment = np.zeros(3)
                moment[axis] = torque_magnitude
                case_specs.append(
                    (label, np.zeros(3), moment, np.zeros(3))
                )
            case_specs.append(
                (
                    "OFFSET_PLUS_Y_AT_PLUS_X",
                    np.asarray((0.0, force_magnitude, 0.0)),
                    np.zeros(3),
                    np.asarray((0.030, 0.0, 0.0)),
                )
            )
            for name, force_a, moment_a, offset_a in case_specs:
                for _ in range(30):
                    step_scene(loaded, "calibration_settle")
                load_samples = []
                applied = {
                    "force_a": force_a,
                    "moment_a": moment_a,
                    "offset_a": offset_a,
                }
                for _ in range(90):
                    step_scene(
                        loaded,
                        "mixed_grip_physical_insert_01",
                        applied=applied,
                        samples=load_samples,
                    )
                raw_mean = np.mean(
                    np.asarray([item[0] for item in load_samples[-45:]]),
                    axis=0,
                )
                compensated = raw_mean - baseline_raw
                environment_sensor = -compensated
                sensor_position, sensor_rotation = _rotation_from_prim(
                    loaded["sensor_prim"], Gf, Usd, UsdGeom
                )
                assembly = transform_wrench_to_task(
                    environment_sensor,
                    sensor_position,
                    sensor_rotation,
                    task_origin,
                    task_rotation,
                )
                expected = np.concatenate(
                    (force_a, moment_a + np.cross(offset_a, force_a))
                )
                error = assembly - expected
                expected_norm = float(np.linalg.norm(expected))
                magnitude_error = float(np.linalg.norm(error))
                active_indices = np.flatnonzero(np.abs(expected) > 1.0e-12)
                sign_ok = bool(
                    active_indices.size
                    and all(
                        assembly[index] * expected[index] > 0.0
                        for index in active_indices
                    )
                )
                cross_indices = [
                    index for index in range(6) if index not in active_indices
                ]
                cross_error = float(
                    np.max(np.abs(assembly[cross_indices]))
                    if cross_indices
                    else 0.0
                )
                moment_arm_error = float(
                    np.linalg.norm(error[3:])
                    if np.linalg.norm(offset_a) > 0.0
                    else 0.0
                )
                case = {
                    "orientation": orientation_name,
                    "name": name,
                    "expected_wrench_assembly": expected.tolist(),
                    "raw_parent_on_child": raw_mean.tolist(),
                    "no_contact_baseline_raw": baseline_raw.tolist(),
                    "compensated_parent_on_child": compensated.tolist(),
                    "environment_on_tool_sensor": environment_sensor.tolist(),
                    "assembly_frame_wrench": assembly.tolist(),
                    "magnitude_error": magnitude_error,
                    "relative_error": magnitude_error / max(expected_norm, 1e-12),
                    "sign_ok": sign_ok,
                    "cross_axis_error": cross_error,
                    "moment_arm_error": moment_arm_error,
                    "application_offset_assembly_m": offset_a.tolist(),
                }
                cases.append(case)
                report["cases"].append(case)
            orientation_reports.append(
                {
                    "orientation": orientation_name,
                    "arm_rad": arm.tolist(),
                    "empty_initial_arm_error_rad": empty[
                        "initial_arm_error_rad"
                    ],
                    "loaded_initial_arm_error_rad": loaded[
                        "initial_arm_error_rad"
                    ],
                    "home_empty_baseline_canonical": home_baseline,
                    "payload_baseline_canonical": payload_baseline,
                    "payload_gravity_raw_delta": payload_gravity_delta.tolist(),
                    "payload_gravity_force_magnitude_n": float(
                        np.linalg.norm(payload_gravity_delta[:3])
                    ),
                    "topology": loaded["topology"],
                    "authoring": loaded["authoring"],
                    "maximum_relative_error": max(
                        item["relative_error"] for item in cases
                    ),
                    "maximum_cross_axis_error": max(
                        item["cross_axis_error"] for item in cases
                    ),
                    "maximum_moment_arm_error": max(
                        item["moment_arm_error"] for item in cases
                    ),
                    "all_signs_ok": all(item["sign_ok"] for item in cases),
                }
            )

        report["orientations"] = orientation_reports
        report["acceptance"] = {
            "maximum_relative_error": 0.08,
            "maximum_force_cross_axis_n": 0.08,
            "maximum_moment_arm_error_nm": 0.010,
            # This is a pose-hold health check, not a wrench or hardware
            # safety limit.  With the repository's frozen iiwa gains, both
            # the empty and 120 g loaded scenes reproducibly settle at about
            # 1.92e-3 rad maximum joint error.  Keep a measured 3e-3 rad
            # SIM_DEBUG_GATE so that a broken controller is still rejected
            # without misclassifying normal gravity deflection as a wrench
            # direction-calibration failure.
            "maximum_initial_arm_error_rad": 0.003,
            "arm_tracking_gate_basis": (
                "measured steady pose-hold error with frozen iiwa gains; "
                "not a hardware safety threshold"
            ),
            "all_signs_correct": True,
            "label": "SIM_DEBUG_GATE",
        }
        report["passed"] = bool(
            all(item["all_signs_ok"] for item in orientation_reports)
            and max(
                item["maximum_relative_error"]
                for item in orientation_reports
            )
            <= report["acceptance"]["maximum_relative_error"]
            and max(
                item["maximum_cross_axis_error"]
                for item in orientation_reports
            )
            <= report["acceptance"]["maximum_force_cross_axis_n"]
            and max(
                item["maximum_moment_arm_error"]
                for item in orientation_reports
            )
            <= report["acceptance"]["maximum_moment_arm_error_nm"]
            and max(
                max(item["empty_initial_arm_error_rad"], item["loaded_initial_arm_error_rad"])
                for item in orientation_reports
            )
            <= report["acceptance"]["maximum_initial_arm_error_rad"]
        )
        passed = report["passed"]
    except BaseException as exception:
        report["error"] = f"{type(exception).__name__}: {exception}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (output / "calibration.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        summary = f"""# D38999 V2 hand2arm wrench direction calibration

- passed: `{report.get('passed', False)}`
- raw semantic: `parent_on_child` in the hand2arm child joint frame
- environment relation: `-(raw - no_contact_baseline)`
- direct Plug/Nut motion actuator used: `false`
- known calibration loads applied to PlugBody: `true (calibration only)`
- thresholds: `SIM_DEBUG_GATE`, not hardware safety values
"""
        (output / "REPORT.md").write_text(summary)
        print(
            json.dumps(
                {
                    "passed": report.get("passed", False),
                    "case_count": len(report.get("cases", [])),
                    "error": report.get("error"),
                },
                sort_keys=True,
            )
        )
        print(
            "ISAAC D38999 V2 WRIST FT DIRECTION CALIBRATION "
            + ("PASSED" if passed else "FAILED")
        )
        app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
