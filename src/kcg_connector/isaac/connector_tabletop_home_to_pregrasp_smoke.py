#!/usr/bin/env python3

"""Physically validate deterministic KUKA Home-to-pregrasp motion v1.

This is an external-contact and tracking acceptance smoke, not a collision-
free planner.  The imported robot currently has self collision disabled, so
the emitted report explicitly keeps ``self_collision_verified`` false.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback


EXPECTED_DOF_NAMES = tuple(f"iiwa_joint_{index}" for index in range(1, 8)) + (
    "f1j1",
    "f1j2",
    "f1j3",
    "f2j1",
    "f2j2",
    "f3j1",
    "f3j2",
    "f3j3",
)
MIMIC_HAND_JOINT_NAMES = ("f1j3", "f2j2", "f3j1", "f3j3")
CAMERA_EYE_M = (1.95, 1.55, 1.45)
CAMERA_TARGET_M = (0.40, -0.05, 0.68)


def _path_is_at_or_below(path, root):
    value = str(path)
    prefix = str(root)
    return value == prefix or value.startswith(prefix + "/")


def _classify_external_contact(
    paths, robot_root, table_path, fixture_path, connector_root
):
    """Classify a contact header against exact scene subtrees."""

    values = tuple(str(path) for path in paths)
    if not any(_path_is_at_or_below(path, robot_root) for path in values):
        return None
    if any(_path_is_at_or_below(path, table_path) for path in values):
        return "table"
    if any(_path_is_at_or_below(path, fixture_path) for path in values):
        return "fixture"
    if any(_path_is_at_or_below(path, connector_root) for path in values):
        return "connector"
    return None


def _gf_quaternion_error_radians(first, second):
    relative = first.GetInverse() * second
    real = max(-1.0, min(1.0, abs(float(relative.GetReal()))))
    return 2.0 * math.acos(real)


def _gf_quaternion_finite(value):
    imaginary = value.GetImaginary()
    return all(
        math.isfinite(item)
        for item in (
            float(value.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    )


def _world_pose(Gf, Usd, UsdGeom, prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform(matrix)
    return transform.GetTranslation(), transform.GetRotation().GetQuat()


def _quaternion_world_z_axis(value):
    imaginary = value.GetImaginary()
    w = float(value.GetReal())
    x = float(imaginary[0])
    y = float(imaginary[1])
    z = float(imaginary[2])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0 or not math.isfinite(norm):
        return (float("nan"),) * 3
    w, x, y, z = (item / norm for item in (w, x, y, z))
    return (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


def _axis_error_radians(first, second):
    cosine = sum(left * right for left, right in zip(first, second))
    return math.acos(max(-1.0, min(1.0, cosine)))


def main():
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-asset",
        default=str(
            repository
            / "artifacts/kcg_connector/isaac/robot/handarm/handarm.usda"
        ),
    )
    parser.add_argument(
        "--connector-asset",
        default=str(
            repository / "artifacts/kcg_connector/isaac/connector_pair.usda"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(
            repository
            / "src/kcg_connector/config/connector_home_to_pregrasp_v1.yaml"
        ),
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    arguments = parser.parse_args()
    if arguments.keep_open and not arguments.gui:
        parser.error("--keep-open requires --gui")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not arguments.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    passed = False
    metrics = {
        "acceptance_scope": "external contacts, tracking and TCP only",
        "gui": arguments.gui,
        "keep_open": arguments.keep_open,
        "object_pose_writes_after_start": 0,
        "passed": False,
        "scene": "kcg_connector_home_to_pregrasp_v1",
        "self_collision_enabled_in_asset": False,
        "self_collision_verified": False,
        "trajectory_kind": (
            "joint_interpolation_screening_not_collision_planned"
        ),
    }
    try:
        import numpy as np

        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
        )
        from isaacsim.core.utils.types import ArticulationAction
        from omni.physx import get_physx_simulation_interface
        from omni.physx.scripts import physicsUtils
        from pxr import (
            Gf,
            PhysxSchema,
            PhysicsSchemaTools,
            Sdf,
            Usd,
            UsdGeom,
            UsdPhysics,
            UsdShade,
        )

        from kcg_connector.home_to_pregrasp import (
            interpolate_segment,
            load_home_to_pregrasp_config,
            minimum_jerk_blend,
        )
        from kcg_connector.isaac_tabletop_scene import (
            author_isaac_tabletop_scene,
            load_connector_tabletop_scene,
        )

        config_path = Path(arguments.config).expanduser().resolve()
        robot_asset = Path(arguments.robot_asset).expanduser().resolve()
        connector_asset = Path(
            arguments.connector_asset
        ).expanduser().resolve()
        for path in (config_path, robot_asset, connector_asset):
            if not path.is_file():
                raise FileNotFoundError(path)
        config = load_home_to_pregrasp_config(config_path)
        tabletop_path = config_path.parent / config.scene.tabletop_config
        tabletop = load_connector_tabletop_scene(tabletop_path)
        rate_hz = tabletop.physics.rate_hz

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / rate_hz,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        stage = get_current_stage()
        author_isaac_tabletop_scene(
            stage,
            tabletop,
            connector_asset,
            add_reference_to_stage=add_reference_to_stage,
            Gf=Gf,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
            physics_utils=physicsUtils,
        )
        add_reference_to_stage(
            str(robot_asset), config.scene.robot_root_prim_path
        )
        tcp_prim = stage.GetPrimAtPath(config.scene.grasp_tcp_prim_path)
        fixed_prim = stage.GetPrimAtPath(
            tabletop.fixed_endpoint.receptacle_prim_path
        )
        for path, prim in (
            (config.scene.grasp_tcp_prim_path, tcp_prim),
            (tabletop.fixed_endpoint.receptacle_prim_path, fixed_prim),
        ):
            if not prim.IsValid():
                raise RuntimeError(f"required scene prim is missing: {path}")

        contact_report_body_count = 0
        for prim in stage.Traverse():
            if not _path_is_at_or_below(
                prim.GetPath(), config.scene.robot_root_prim_path
            ):
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report.CreateThresholdAttr().Set(0.0)
            contact_report_body_count += 1
        if contact_report_body_count < 17:
            raise RuntimeError("robot contact reporting is incomplete")

        if arguments.gui:
            from isaacsim.core.rendering_manager import ViewportManager
            from pxr import UsdLux

            lighting_root = "/World/HomeToPregraspGuiLighting"
            UsdGeom.Xform.Define(stage, lighting_root)
            dome = UsdLux.DomeLight.Define(stage, lighting_root + "/Fill")
            dome.CreateIntensityAttr(tabletop.render.dome_light_intensity)
            dome.CreateColorAttr(
                Gf.Vec3f(*tabletop.render.dome_light_color_rgb)
            )
            key = UsdLux.DistantLight.Define(stage, lighting_root + "/Key")
            key.CreateIntensityAttr(tabletop.render.key_light_intensity)
            key.CreateAngleAttr(2.0)
            key.CreateColorAttr(
                Gf.Vec3f(*tabletop.render.key_light_color_rgb)
            )
            UsdGeom.Xformable(key).AddRotateXYZOp().Set(
                Gf.Vec3f(*tabletop.render.key_light_rotation_degrees_xyz)
            )
            ViewportManager.set_camera_view(
                camera="/OmniverseKit_Persp",
                eye=np.asarray(CAMERA_EYE_M, dtype=np.float64),
                target=np.asarray(CAMERA_TARGET_M, dtype=np.float64),
            )
            simulation_app.update()

        robot = world.scene.add(
            SingleArticulation(
                prim_path=config.scene.articulation_prim_path,
                name="tabletop_home_to_pregrasp_handarm",
            )
        )
        body = world.scene.add(
            SingleRigidPrim(
                prim_path=tabletop.loose_endpoint.body_prim_path,
                name="pregrasp_loose_body",
            )
        )
        nut = world.scene.add(
            SingleRigidPrim(
                prim_path=tabletop.loose_endpoint.nut_prim_path,
                name="pregrasp_loose_nut",
            )
        )
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError(
                "robot articulation handles were not initialized"
            )
        dof_names = tuple(robot.dof_names)
        if set(dof_names) != set(EXPECTED_DOF_NAMES) or len(dof_names) != 15:
            raise RuntimeError("unexpected articulation DOF layout")
        name_to_index = {
            name: index for index, name in enumerate(dof_names)
        }
        arm_indices = np.asarray(
            [name_to_index[name] for name in config.robot.arm_joint_names],
            dtype=np.int32,
        )
        hand_indices = np.asarray(
            [
                name_to_index[name]
                for name in config.robot.active_hand_joint_names
            ],
            dtype=np.int32,
        )
        controlled_indices = np.concatenate((arm_indices, hand_indices))

        zero_positions = np.zeros(robot.num_dof, dtype=np.float32)
        robot.set_joint_positions(zero_positions)
        robot.set_joint_velocities(zero_positions)
        controller = robot.get_articulation_controller()
        kps = np.zeros(robot.num_dof, dtype=np.float32)
        kds = np.zeros(robot.num_dof, dtype=np.float32)
        kps[arm_indices] = config.robot.arm_stiffness
        kds[arm_indices] = config.robot.arm_damping
        kps[hand_indices] = config.robot.hand_stiffness
        kds[hand_indices] = config.robot.hand_damping
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        world.get_physics_context().set_gravity(
            tabletop.physics.gravity_m_s2
        )

        home_arm = np.asarray(config.robot.home_arm_rad, dtype=np.float64)
        closed_home_hand = np.zeros(4, dtype=np.float64)
        open_hand = np.asarray(config.robot.open_hand_rad, dtype=np.float64)
        current_arm_target = home_arm.copy()
        current_hand_target = closed_home_hand.copy()
        dof_properties = robot.dof_properties
        maximum_joint_limit_violation = 0.0
        maximum_joint_speed = 0.0
        finite_throughout = True
        contact_counts = {"table": 0, "fixture": 0, "connector": 0}
        contact_header_counts = {
            "table": 0,
            "fixture": 0,
            "connector": 0,
        }
        maximum_tracking_error = 0.0
        maximum_arm_tracking_error = 0.0
        maximum_hand_tracking_error = 0.0

        initial_body_position, _ = body.get_world_pose()
        initial_nut_position, _ = nut.get_world_pose()
        initial_loose_center = 0.6 * np.asarray(
            initial_body_position, dtype=np.float64
        ) + 0.4 * np.asarray(initial_nut_position, dtype=np.float64)
        fixed_initial_position, fixed_initial_orientation = _world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )

        def observe_and_step(arm_target, hand_target):
            nonlocal finite_throughout
            nonlocal maximum_joint_limit_violation
            nonlocal maximum_joint_speed
            nonlocal maximum_tracking_error
            nonlocal maximum_arm_tracking_error
            nonlocal maximum_hand_tracking_error
            target = np.concatenate((arm_target, hand_target)).astype(
                np.float32
            )
            robot.apply_action(
                ArticulationAction(
                    joint_positions=target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            body_position, body_orientation = body.get_world_pose()
            nut_position, nut_orientation = nut.get_world_pose()
            sampled = np.concatenate(
                (
                    positions,
                    velocities,
                    np.asarray(body_position, dtype=np.float64),
                    np.asarray(body_orientation, dtype=np.float64),
                    np.asarray(nut_position, dtype=np.float64),
                    np.asarray(nut_orientation, dtype=np.float64),
                )
            )
            finite_throughout = bool(
                finite_throughout and np.all(np.isfinite(sampled))
            )
            maximum_joint_speed = max(
                maximum_joint_speed,
                float(np.max(np.abs(velocities))),
            )
            maximum_tracking_error = max(
                maximum_tracking_error,
                float(
                    np.max(
                        np.abs(positions[controlled_indices] - target)
                    )
                ),
            )
            maximum_arm_tracking_error = max(
                maximum_arm_tracking_error,
                float(
                    np.max(
                        np.abs(positions[arm_indices] - arm_target)
                    )
                ),
            )
            maximum_hand_tracking_error = max(
                maximum_hand_tracking_error,
                float(
                    np.max(
                        np.abs(positions[hand_indices] - hand_target)
                    )
                ),
            )
            for dof_index in range(robot.num_dof):
                if bool(dof_properties[dof_index]["hasLimits"]):
                    lower = float(dof_properties[dof_index]["lower"])
                    upper = float(dof_properties[dof_index]["upper"])
                    maximum_joint_limit_violation = max(
                        maximum_joint_limit_violation,
                        lower - float(positions[dof_index]),
                        float(positions[dof_index]) - upper,
                    )
            headers, _, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                category = _classify_external_contact(
                    paths,
                    config.scene.robot_root_prim_path,
                    tabletop.table.prim_path,
                    tabletop.fixed_endpoint.fixture_prim_path,
                    tabletop.world.connector_reference_prim_path,
                )
                if category is not None:
                    contact_header_counts[category] += 1
                    contact_counts[category] += int(header.num_contact_data)
            return positions, velocities

        # Establish gravity and let the loose endpoint settle while the robot
        # holds exact Home.  Object poses are never written after simulation.
        settle_steps = tabletop.physics.settle_steps
        for _ in range(settle_steps):
            observe_and_step(current_arm_target, current_hand_target)

        # Open the hand at high Home before moving the arm.  This avoids a
        # finger sweep near the connector pickup site.
        hand_open_steps = round(config.motion.hand_open_duration_s * rate_hz)
        for index in range(hand_open_steps):
            fraction = float(index + 1) / float(hand_open_steps)
            blend = minimum_jerk_blend(fraction)
            current_hand_target = (
                closed_home_hand + blend * (open_hand - closed_home_hand)
            )
            observe_and_step(current_arm_target, current_hand_target)
        current_hand_target = open_hand.copy()

        segment_metrics = []
        for segment in config.motion.segments:
            start_arm = current_arm_target.copy()
            final_arm = np.asarray(segment.target_arm_rad, dtype=np.float64)
            segment_steps = round(segment.duration_s * rate_hz)
            segment_max_tracking_error_before = maximum_tracking_error
            for index in range(segment_steps):
                fraction = float(index + 1) / float(segment_steps)
                current_arm_target = np.asarray(
                    interpolate_segment(
                        tuple(float(value) for value in start_arm),
                        tuple(float(value) for value in final_arm),
                        fraction,
                    ),
                    dtype=np.float64,
                )
                observe_and_step(current_arm_target, current_hand_target)
            current_arm_target = final_arm.copy()
            segment_metrics.append(
                {
                    "duration_s": segment.duration_s,
                    "name": segment.name,
                    "steps": segment_steps,
                    "tracking_error_delta_rad": max(
                        0.0,
                        maximum_tracking_error
                        - segment_max_tracking_error_before,
                    ),
                }
            )

        hold_steps = round(config.motion.hold_duration_s * rate_hz)
        tail_start_body_position, _ = body.get_world_pose()
        tail_start_nut_position, _ = nut.get_world_pose()
        tail_start_center = 0.6 * np.asarray(
            tail_start_body_position, dtype=np.float64
        ) + 0.4 * np.asarray(tail_start_nut_position, dtype=np.float64)
        maximum_tail_displacement = 0.0
        for _ in range(hold_steps):
            observe_and_step(current_arm_target, current_hand_target)
            body_position, _ = body.get_world_pose()
            nut_position, _ = nut.get_world_pose()
            center = 0.6 * np.asarray(
                body_position, dtype=np.float64
            ) + 0.4 * np.asarray(nut_position, dtype=np.float64)
            maximum_tail_displacement = max(
                maximum_tail_displacement,
                float(np.linalg.norm(center - tail_start_center)),
            )

        final_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        final_velocities = np.asarray(
            robot.get_joint_velocities(), dtype=np.float64
        )
        final_body_position, _ = body.get_world_pose()
        final_nut_position, _ = nut.get_world_pose()
        final_loose_center = 0.6 * np.asarray(
            final_body_position, dtype=np.float64
        ) + 0.4 * np.asarray(final_nut_position, dtype=np.float64)
        tcp_position, tcp_orientation = _world_pose(
            Gf, Usd, UsdGeom, tcp_prim
        )
        fixed_final_position, fixed_final_orientation = _world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )
        tcp_position_error = float(
            np.linalg.norm(
                np.asarray(tcp_position, dtype=np.float64)
                - np.asarray(
                    config.motion.target_tcp_position_m, dtype=np.float64
                )
            )
        )
        tcp_axis = _quaternion_world_z_axis(tcp_orientation)
        tcp_axis_error = _axis_error_radians(
            tcp_axis, config.motion.target_tcp_down_axis_world
        )
        final_arm_tracking_error = float(
            np.max(np.abs(final_positions[arm_indices] - current_arm_target))
        )
        final_joint_speed = float(np.max(np.abs(final_velocities)))
        loose_xy_drift = float(
            np.linalg.norm(
                final_loose_center[:2] - initial_loose_center[:2]
            )
        )
        fixed_translation_drift = float(
            np.linalg.norm(
                np.asarray(fixed_final_position, dtype=np.float64)
                - np.asarray(fixed_initial_position, dtype=np.float64)
            )
        )
        fixed_rotation_drift = _gf_quaternion_error_radians(
            fixed_initial_orientation, fixed_final_orientation
        )
        finite_final = bool(
            np.all(np.isfinite(final_positions))
            and np.all(np.isfinite(final_velocities))
            and np.all(np.isfinite(final_loose_center))
            and _gf_quaternion_finite(tcp_orientation)
            and _gf_quaternion_finite(fixed_final_orientation)
            and math.isfinite(tcp_position_error)
            and math.isfinite(tcp_axis_error)
        )
        acceptance = config.acceptance
        external_contacts_zero = bool(
            contact_counts["table"] == 0
            and contact_counts["fixture"] == 0
            and contact_counts["connector"] == 0
        )
        passed = bool(
            finite_throughout
            and finite_final
            and maximum_joint_limit_violation
            <= acceptance.maximum_joint_limit_violation_rad
            and maximum_joint_speed
            <= acceptance.maximum_observed_joint_speed_rad_s
            and maximum_arm_tracking_error
            <= acceptance.maximum_observed_arm_tracking_error_rad
            and maximum_hand_tracking_error
            <= acceptance.maximum_observed_hand_tracking_error_rad
            and final_joint_speed
            <= acceptance.maximum_final_joint_speed_rad_s
            and final_arm_tracking_error
            <= acceptance.maximum_final_arm_tracking_error_rad
            and tcp_position_error
            <= acceptance.maximum_tcp_position_error_m
            and tcp_axis_error <= acceptance.maximum_tcp_axis_error_rad
            and loose_xy_drift
            <= acceptance.maximum_loose_endpoint_xy_drift_m
            and maximum_tail_displacement
            <= acceptance.maximum_loose_endpoint_tail_displacement_m
            and fixed_translation_drift
            <= acceptance.maximum_fixed_translation_drift_m
            and fixed_rotation_drift
            <= acceptance.maximum_fixed_rotation_drift_rad
            and external_contacts_zero
        )
        metrics.update(
            {
                "contact_counts": contact_counts,
                "contact_header_counts": contact_header_counts,
                "external_contacts_zero": external_contacts_zero,
                "final_arm_tracking_error_rad": (
                    final_arm_tracking_error
                ),
                "final_joint_speed_rad_s": final_joint_speed,
                "finite_final": finite_final,
                "finite_throughout": finite_throughout,
                "fixed_rotation_drift_rad": fixed_rotation_drift,
                "fixed_translation_drift_m": fixed_translation_drift,
                "hand_opened_at_home_before_arm_motion": True,
                "hold_duration_s": config.motion.hold_duration_s,
                "joint_limit_violation_rad": max(
                    0.0, maximum_joint_limit_violation
                ),
                "loose_endpoint_tail_displacement_m": (
                    maximum_tail_displacement
                ),
                "loose_endpoint_xy_drift_m": loose_xy_drift,
                "maximum_joint_speed_rad_s": maximum_joint_speed,
                "maximum_arm_tracking_error_rad": (
                    maximum_arm_tracking_error
                ),
                "maximum_hand_tracking_error_rad": (
                    maximum_hand_tracking_error
                ),
                "maximum_tracking_error_rad": maximum_tracking_error,
                "motion_interpolation": config.motion.interpolation,
                "physics_rate_hz": rate_hz,
                "segment_metrics": segment_metrics,
                "tcp_axis_error_rad": tcp_axis_error,
                "tcp_position_error_m": tcp_position_error,
                "tcp_world_position_m": [
                    float(value) for value in tcp_position
                ],
                "tcp_world_z_axis": list(tcp_axis),
                "passed": passed,
            }
        )
        json.dumps(metrics, allow_nan=False, sort_keys=True)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print(
            "ISAAC CONNECTOR HOME TO PREGRASP V1 "
            + ("PASSED" if passed else "FAILED"),
            flush=True,
        )
    except BaseException as exception:
        metrics.update(
            {
                "error": f"{type(exception).__name__}: {exception}",
                "passed": False,
            }
        )
        traceback.print_exc()
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print("ISAAC CONNECTOR HOME TO PREGRASP V1 FAILED", flush=True)
    finally:
        if arguments.keep_open and arguments.gui:
            print(
                "ISAAC CONNECTOR HOME TO PREGRASP GUI REMAINS OPEN; "
                "close the window to exit",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
