#!/usr/bin/env python3

"""Verify that the imported three-finger hand physically holds the plug."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback

import yaml


def _load_curriculum(config_path):
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    robot = document["robot_curriculum"]
    geometry = document["geometry"]
    success = document["success"]
    result = {
        "pre_approach_arm": robot["pre_approach_joint_positions"],
        "approach_arm": robot["approach_joint_positions"],
        "grasp_arm": robot["connector_grasp_joint_positions"],
        "open_hand": robot["open_hand_positions"],
        "grasp_hand": robot["grasp_hand_positions"],
        "grasp_center": robot["grasp_center_world"],
        "plug_height": geometry["initial_plug_center_height"],
        "arm_seed_fraction": robot["connector_grasp_approach_fraction"],
        "maximum_finger_torque": success[
            "maximum_absolute_finger_torque"
        ],
    }
    expected_lengths = {
        "pre_approach_arm": 7,
        "approach_arm": 7,
        "grasp_arm": 7,
        "open_hand": 4,
        "grasp_hand": 4,
        "grasp_center": 3,
    }
    for name, expected_length in expected_lengths.items():
        values = [float(value) for value in result[name]]
        if len(values) != expected_length or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError(f"invalid robot curriculum field: {name}")
        result[name] = values
    result["plug_height"] = float(result["plug_height"])
    if (
        not math.isfinite(result["plug_height"])
        or result["plug_height"] <= 0.0
    ):
        raise ValueError(
            "initial plug center height must be finite and positive"
        )
    result["arm_seed_fraction"] = float(result["arm_seed_fraction"])
    if not math.isfinite(result["arm_seed_fraction"]) or not (
        0.0 <= result["arm_seed_fraction"] <= 1.0
    ):
        raise ValueError("connector grasp approach fraction must be in [0, 1]")
    result["maximum_finger_torque"] = float(
        result["maximum_finger_torque"]
    )
    if not math.isfinite(result["maximum_finger_torque"]) or not (
        result["maximum_finger_torque"] > 0.0
    ):
        raise ValueError("maximum finger torque must be finite and positive")
    return result


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
            repository / "src/kcg_connector/config/connector_task.yaml"
        ),
    )
    parser.add_argument("--closure-steps", type=int, default=480)
    parser.add_argument("--hold-steps", type=int, default=480)
    parser.add_argument("--grasp-height-offset", type=float, default=0.0)
    parser.add_argument(
        "--arm-seed-fraction",
        type=float,
        default=None,
        help="override connector_grasp_approach_fraction from the task config",
    )
    parser.add_argument("--arm-kp", type=float, default=8000.0)
    parser.add_argument("--arm-kd", type=float, default=220.0)
    parser.add_argument("--hand-kp", type=float, default=5.0)
    parser.add_argument("--hand-kd", type=float, default=0.7)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="open the Isaac Sim window and render physics steps",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="keep updating the GUI after printing the final result",
    )
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
        "closure_steps": arguments.closure_steps,
        "grasp_height_offset": arguments.grasp_height_offset,
        "arm_kp": arguments.arm_kp,
        "arm_kd": arguments.arm_kd,
        "hand_kp": arguments.hand_kp,
        "hand_kd": arguments.hand_kd,
        "gui": arguments.gui,
        "keep_open": arguments.keep_open,
        "hold_steps": arguments.hold_steps,
        "physical_attachment": "none",
        "tactile_sensors": "none",
        "torque_channels": ["f1j2", "f2j1", "f3j2"],
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

        from kcg_connector.robot_model import (
            ACTIVE_HAND_JOINT_NAMES,
            ARM_JOINT_NAMES,
            named_joint_target,
        )

        robot_asset = Path(arguments.robot_asset).expanduser().resolve()
        connector_asset = (
            Path(arguments.connector_asset).expanduser().resolve()
        )
        config_path = Path(arguments.config).expanduser().resolve()
        for path in (robot_asset, connector_asset, config_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if arguments.closure_steps < 120 or arguments.hold_steps < 120:
            raise ValueError(
                "closure and hold must each run for at least 120 steps"
            )
        if not math.isfinite(arguments.grasp_height_offset):
            raise ValueError("grasp height offset must be finite")
        if not math.isfinite(arguments.arm_kp) or arguments.arm_kp <= 0.0:
            raise ValueError("arm kp must be finite and positive")
        if not math.isfinite(arguments.arm_kd) or arguments.arm_kd <= 0.0:
            raise ValueError("arm kd must be finite and positive")
        if not math.isfinite(arguments.hand_kp) or arguments.hand_kp <= 0.0:
            raise ValueError("hand kp must be finite and positive")
        if not math.isfinite(arguments.hand_kd) or arguments.hand_kd <= 0.0:
            raise ValueError("hand kd must be finite and positive")
        curriculum = _load_curriculum(config_path)
        arm_seed_fraction = (
            curriculum["arm_seed_fraction"]
            if arguments.arm_seed_fraction is None
            else arguments.arm_seed_fraction
        )
        if not math.isfinite(arm_seed_fraction) or not (
            0.0 <= arm_seed_fraction <= 1.0
        ):
            raise ValueError("arm seed fraction must be in [0, 1]")
        metrics["arm_seed_fraction"] = arm_seed_fraction
        metrics["arm_seed_fraction_source"] = (
            "config" if arguments.arm_seed_fraction is None else "command_line"
        )
        pre_approach_arm = np.asarray(
            curriculum["pre_approach_arm"], dtype=np.float64
        )
        approach_arm = np.asarray(curriculum["approach_arm"], dtype=np.float64)
        if arguments.arm_seed_fraction is None:
            arm_target = np.asarray(
                curriculum["grasp_arm"], dtype=np.float64
            )
            metrics["arm_seed_source"] = "cartesian_corrected_config"
        else:
            arm_target = (
                pre_approach_arm
                + arm_seed_fraction * (approach_arm - pre_approach_arm)
            )
            metrics["arm_seed_source"] = "fraction_override"

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 240.0,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        robot_root = "/World/HandArm"
        articulation_path = f"{robot_root}/Geometry/world"
        connector_root = "/World/ConnectorTask"
        add_reference_to_stage(str(robot_asset), robot_root)
        connector_prim = add_reference_to_stage(
            str(connector_asset), connector_root
        )
        stage = get_current_stage()
        nested_scene_path = f"{connector_root}/PhysicsScene"
        if stage.GetPrimAtPath(nested_scene_path).IsValid():
            stage.RemovePrim(nested_scene_path)

        grasp_center = np.asarray(curriculum["grasp_center"], dtype=np.float64)
        connector_center = grasp_center.copy()
        connector_center[2] += arguments.grasp_height_offset
        connector_origin = connector_center.copy()
        connector_origin[2] -= curriculum["plug_height"]
        UsdGeom.Xformable(connector_prim).AddTranslateOp().Set(
            Gf.Vec3d(*connector_origin)
        )

        if arguments.gui:
            from isaacsim.core.rendering_manager import ViewportManager
            from pxr import UsdLux

            # SimulationApp's default Perspective camera can point away from
            # this compact scene after World creates a fresh stage.  Give the
            # interactive smoke test a deterministic, well-lit close view of
            # the hand/connector workspace without changing the physics.
            lighting_root = "/World/GuiLighting"
            UsdGeom.Xform.Define(stage, lighting_root)
            dome_light = UsdLux.DomeLight.Define(
                stage, f"{lighting_root}/Fill"
            )
            dome_light.CreateIntensityAttr(650.0)
            dome_light.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
            key_light = UsdLux.DistantLight.Define(
                stage, f"{lighting_root}/Key"
            )
            key_light.CreateIntensityAttr(2200.0)
            key_light.CreateAngleAttr(2.0)
            key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.82))
            UsdGeom.Xformable(key_light).AddRotateXYZOp().Set(
                Gf.Vec3f(-45.0, 30.0, 35.0)
            )

            camera_target = connector_center + np.asarray(
                [0.0, 0.0, 0.04], dtype=np.float64
            )
            camera_eye = camera_target + np.asarray(
                [0.85, 0.90, 0.55], dtype=np.float64
            )
            ViewportManager.set_camera_view(
                camera="/OmniverseKit_Persp",
                eye=camera_eye,
                target=camera_target,
            )
            metrics["gui_camera_eye"] = camera_eye.tolist()
            metrics["gui_camera_target"] = camera_target.tolist()
            metrics["gui_lighting"] = {
                "dome_intensity": 650.0,
                "key_intensity": 2200.0,
            }
            simulation_app.update()

        # Match the validated Gazebo pickup condition: the plug rests on a
        # small physical pedestal while the fingers build preload.  The
        # pedestal is deleted before gravity/hold evaluation, so it cannot
        # contribute to the reported grasp.
        support_path = "/World/ConnectorGraspSupport"
        support = UsdGeom.Cylinder.Define(stage, support_path)
        support.CreateAxisAttr(UsdGeom.Tokens.z)
        support.CreateRadiusAttr(0.025)
        support.CreateHeightAttr(0.020)
        support.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.12, 0.14)])
        UsdGeom.Xformable(support).AddTranslateOp().Set(
            Gf.Vec3d(
                connector_center[0],
                connector_center[1],
                0.230 + arguments.grasp_height_offset,
            )
        )
        UsdPhysics.CollisionAPI.Apply(support.GetPrim())

        grip_material_path = "/World/GripPhysicsMaterial"
        grip_material = UsdShade.Material.Define(stage, grip_material_path)
        grip_physics = UsdPhysics.MaterialAPI.Apply(grip_material.GetPrim())
        grip_physics.CreateStaticFrictionAttr(1.4)
        grip_physics.CreateDynamicFrictionAttr(1.4)
        grip_physics.CreateRestitutionAttr(0.0)
        bound_hand_collision_anchors = []
        bound_nut_collisions = []
        nut_path_prefix = f"{connector_root}/Plug/CouplingNut/"
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            # URDF Importer stores each finger collider inside an instanceable
            # ``*_convex`` reference.  The collision API itself lives on an
            # instance proxy and is therefore not returned by Stage.Traverse().
            # Bind the material to the editable reference root; USD physics
            # material binding is inherited by its collision child.
            is_hand_collision_anchor = (
                prim_path.startswith(robot_root + "/")
                and prim.GetName().endswith("_convex")
                and any(
                    link_name in prim_path
                    for link_name in ("/f1Link", "/f2Link", "/f3Link")
                )
            )
            is_nut_collision = (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                and prim_path.startswith(nut_path_prefix)
            )
            if is_hand_collision_anchor or is_nut_collision:
                physicsUtils.add_physics_material_to_prim(
                    stage, prim, Sdf.Path(grip_material_path)
                )
                if is_hand_collision_anchor:
                    bound_hand_collision_anchors.append(prim_path)
                else:
                    bound_nut_collisions.append(prim_path)
        if len(bound_hand_collision_anchors) != 8:
            raise RuntimeError(
                "expected 8 finger collision anchors, found "
                f"{len(bound_hand_collision_anchors)}"
            )
        if not bound_nut_collisions:
            raise RuntimeError(
                "no coupling-nut collision prims received grip material"
            )
        metrics["grip_material_hand_anchor_count"] = len(
            bound_hand_collision_anchors
        )
        metrics["grip_material_nut_collision_count"] = len(
            bound_nut_collisions
        )

        proxy_material_bindings = {}
        robot_prim = stage.GetPrimAtPath(robot_root)
        for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
            prim_path = str(prim.GetPath())
            if (
                prim.IsInstanceProxy()
                and prim.HasAPI(UsdPhysics.CollisionAPI)
                and any(
                    link_name in prim_path
                    for link_name in ("/f1Link", "/f2Link", "/f3Link")
                )
            ):
                bound_material, binding_relation = UsdShade.MaterialBindingAPI(
                    prim
                ).ComputeBoundMaterial("physics")
                proxy_material_bindings[prim_path] = {
                    "binding_relationship": (
                        str(binding_relation.GetPath())
                        if binding_relation
                        else None
                    ),
                    "material": (
                        str(bound_material.GetPath())
                        if bound_material
                        else None
                    ),
                }
        metrics["finger_instance_proxy_material_bindings"] = (
            proxy_material_bindings
        )
        proxy_material_binding_ok = bool(
            len(proxy_material_bindings) == 8
            and all(
                binding["material"] == grip_material_path
                for binding in proxy_material_bindings.values()
            )
        )
        metrics["finger_instance_proxy_material_binding_ok"] = (
            proxy_material_binding_ok
        )
        metrics["grip_material_dynamic_friction"] = 1.4
        metrics["grip_material_static_friction"] = 1.4

        for rigid_body_path in (
            f"{connector_root}/Plug/BodyAssembly",
            f"{connector_root}/Plug/CouplingNut",
        ):
            contact_report = PhysxSchema.PhysxContactReportAPI.Apply(
                stage.GetPrimAtPath(rigid_body_path)
            )
            contact_report.CreateThresholdAttr().Set(0.0)

        def capture_connector_contacts():
            headers, contacts, friction_anchors = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            result = []
            for header in headers:
                actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
                actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
                collider0 = str(
                    PhysicsSchemaTools.intToSdfPath(header.collider0)
                )
                collider1 = str(
                    PhysicsSchemaTools.intToSdfPath(header.collider1)
                )
                all_paths = (actor0, actor1, collider0, collider1)
                if not any(
                    path.startswith(connector_root) for path in all_paths
                ):
                    continue
                contact_records = []
                for index in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data,
                ):
                    contact = contacts[index]
                    contact_records.append(
                        {
                            "impulse": [
                                float(value) for value in contact.impulse
                            ],
                            "material0": str(
                                PhysicsSchemaTools.intToSdfPath(
                                    contact.material0
                                )
                            ),
                            "material1": str(
                                PhysicsSchemaTools.intToSdfPath(
                                    contact.material1
                                )
                            ),
                            "normal": [
                                float(value) for value in contact.normal
                            ],
                            "position": [
                                float(value) for value in contact.position
                            ],
                            "separation": float(contact.separation),
                        }
                    )
                anchor_impulses = []
                for index in range(
                    header.friction_anchors_offset,
                    header.friction_anchors_offset
                    + header.num_friction_anchors_data,
                ):
                    anchor = friction_anchors[index]
                    anchor_impulses.append(
                        [float(value) for value in anchor.impulse]
                    )
                result.append(
                    {
                        "actor0": actor0,
                        "actor1": actor1,
                        "collider0": collider0,
                        "collider1": collider1,
                        "contacts": contact_records,
                        "friction_anchor_impulses": anchor_impulses,
                    }
                )
            return result

        robot = world.scene.add(
            SingleArticulation(
                prim_path=articulation_path,
                name="connector_grasp_handarm",
            )
        )
        body = world.scene.add(
            SingleRigidPrim(
                prim_path=f"{connector_root}/Plug/BodyAssembly",
                name="connector_grasp_body",
            )
        )
        nut = world.scene.add(
            SingleRigidPrim(
                prim_path=f"{connector_root}/Plug/CouplingNut",
                name="connector_grasp_nut",
            )
        )

        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError(
                "robot articulation handles were not initialized"
            )
        grasp_tcp_path = (
            articulation_path
            + "/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3"
            + "/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7"
            + "/iiwa_link_ee/handbase_link/grasp_tcp"
        )
        grasp_tcp_prim = stage.GetPrimAtPath(grasp_tcp_path)
        if not grasp_tcp_prim.IsValid():
            raise RuntimeError(f"missing grasp TCP prim: {grasp_tcp_path}")

        def grasp_tcp_world_transform():
            return UsdGeom.XformCache().GetLocalToWorldTransform(
                grasp_tcp_prim
            )

        def body_position_in_grasp_frame(body_position):
            point = grasp_tcp_world_transform().GetInverse().Transform(
                Gf.Vec3d(*(float(value) for value in body_position))
            )
            return np.asarray(point, dtype=np.float64)

        dof_names = tuple(robot.dof_names)
        name_to_index = {name: index for index, name in enumerate(dof_names)}
        controlled_names = ARM_JOINT_NAMES + ACTIVE_HAND_JOINT_NAMES
        controlled_indices = np.asarray(
            [name_to_index[name] for name in controlled_names], dtype=np.int32
        )
        arm_indices = np.asarray(
            [name_to_index[name] for name in ARM_JOINT_NAMES], dtype=np.int32
        )
        hand_indices = np.asarray(
            [name_to_index[name] for name in ACTIVE_HAND_JOINT_NAMES],
            dtype=np.int32,
        )
        sensor_names = ("f1j2", "f2j1", "f3j2")
        sensor_indices = np.asarray(
            [name_to_index[name] for name in sensor_names], dtype=np.int32
        )
        dof_properties = robot.dof_properties
        max_abs_joint_velocity = 0.0
        max_joint_limit_violation = 0.0

        def observe_joint_state():
            nonlocal max_abs_joint_velocity, max_joint_limit_violation
            observed_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            observed_velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            max_abs_joint_velocity = max(
                max_abs_joint_velocity,
                float(np.max(np.abs(observed_velocities))),
            )
            for observed_index in range(robot.num_dof):
                if bool(dof_properties[observed_index]["hasLimits"]):
                    lower = float(dof_properties[observed_index]["lower"])
                    upper = float(dof_properties[observed_index]["upper"])
                    max_joint_limit_violation = max(
                        max_joint_limit_violation,
                        lower - float(observed_positions[observed_index]),
                        float(observed_positions[observed_index]) - upper,
                    )
            return observed_positions, observed_velocities

        initial_positions = named_joint_target(
            dof_names,
            arm_target,
            curriculum["open_hand"],
        ).astype(np.float32)
        robot.set_joint_positions(initial_positions)
        robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))

        controller = robot.get_articulation_controller()
        kps = np.zeros(robot.num_dof, dtype=np.float32)
        kds = np.zeros(robot.num_dof, dtype=np.float32)
        kps[arm_indices] = arguments.arm_kp
        kds[arm_indices] = arguments.arm_kd
        kps[hand_indices] = arguments.hand_kp
        kds[hand_indices] = arguments.hand_kd
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)

        open_target = initial_positions[controlled_indices].copy()
        grasp_target = named_joint_target(
            dof_names,
            arm_target,
            curriculum["grasp_hand"],
        )[controlled_indices].astype(np.float32)
        world.get_physics_context().set_gravity(0.0)
        robot.apply_action(
            ArticulationAction(
                joint_positions=open_target,
                joint_indices=controlled_indices,
            )
        )
        for _ in range(30):
            world.step(render=arguments.gui)
        initial_body_position, _ = body.get_world_pose()
        initial_nut_position, _ = nut.get_world_pose()
        tare_efforts = np.asarray(
            robot.get_measured_joint_efforts(joint_indices=sensor_indices),
            dtype=np.float64,
        )

        finite_throughout = True
        for step_index in range(arguments.closure_steps):
            blend = float(step_index + 1) / float(arguments.closure_steps)
            target = open_target + blend * (grasp_target - open_target)
            robot.apply_action(
                ArticulationAction(
                    joint_positions=target.astype(np.float32),
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            positions, velocities = observe_joint_state()
            body_position, _ = body.get_world_pose()
            if not (
                np.all(np.isfinite(positions))
                and np.all(np.isfinite(velocities))
                and np.all(np.isfinite(body_position))
            ):
                finite_throughout = False
                break

        supported_body_position, _ = body.get_world_pose()
        supported_nut_position, _ = nut.get_world_pose()
        closure_displacement = float(
            np.linalg.norm(supported_body_position - initial_body_position)
        )
        contact_efforts = np.asarray(
            robot.get_measured_joint_efforts(joint_indices=sensor_indices),
            dtype=np.float64,
        )
        torque_deltas = contact_efforts - tare_efforts
        metrics["supported_contact_snapshot"] = capture_connector_contacts()

        stage.RemovePrim(support_path)
        stage.RemovePrim(f"{connector_root}/Receptacle")
        simulation_app.update()
        for _ in range(30):
            robot.apply_action(
                ArticulationAction(
                    joint_positions=grasp_target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
        pre_gravity_body_position, _ = body.get_world_pose()
        pre_gravity_nut_position, _ = nut.get_world_pose()
        pre_gravity_joint_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        pre_gravity_tcp_transform = grasp_tcp_world_transform()
        pre_gravity_tcp_position = np.asarray(
            pre_gravity_tcp_transform.ExtractTranslation(), dtype=np.float64
        )
        pre_gravity_body_in_grasp = body_position_in_grasp_frame(
            pre_gravity_body_position
        )
        unsupported_settle_displacement = float(
            np.linalg.norm(pre_gravity_body_position - supported_body_position)
        )
        metrics["pre_gravity_contact_snapshot"] = capture_connector_contacts()

        world.get_physics_context().set_gravity(-9.81)
        last_window_start = None
        last_window_relative_start = None
        effort_samples = []
        max_abs_measured_arm_effort = 0.0
        window_steps = min(120, arguments.hold_steps // 2)
        for step_index in range(arguments.hold_steps):
            robot.apply_action(
                ArticulationAction(
                    joint_positions=grasp_target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            measured_arm_efforts = np.asarray(
                robot.get_measured_joint_efforts(joint_indices=arm_indices),
                dtype=np.float64,
            )
            max_abs_measured_arm_effort = max(
                max_abs_measured_arm_effort,
                float(np.max(np.abs(measured_arm_efforts))),
            )
            if step_index == arguments.hold_steps - window_steps - 1:
                last_window_start, _ = body.get_world_pose()
                last_window_relative_start = body_position_in_grasp_frame(
                    last_window_start
                )
            if step_index >= arguments.hold_steps - window_steps:
                effort_samples.append(
                    np.asarray(
                        robot.get_measured_joint_efforts(
                            joint_indices=sensor_indices
                        ),
                        dtype=np.float64,
                    )
                )
            positions, velocities = observe_joint_state()
            body_position, _ = body.get_world_pose()
            if not (
                np.all(np.isfinite(positions))
                and np.all(np.isfinite(velocities))
                and np.all(np.isfinite(body_position))
            ):
                finite_throughout = False
                break

        final_body_position, _ = body.get_world_pose()
        final_nut_position, _ = nut.get_world_pose()
        final_tcp_transform = grasp_tcp_world_transform()
        final_tcp_position = np.asarray(
            final_tcp_transform.ExtractTranslation(), dtype=np.float64
        )
        final_body_in_grasp = body_position_in_grasp_frame(final_body_position)
        metrics["final_contact_snapshot"] = capture_connector_contacts()
        if last_window_start is None:
            last_window_start = final_body_position.copy()
        if last_window_relative_start is None:
            last_window_relative_start = final_body_in_grasp.copy()
        world_hold_displacement = float(
            np.linalg.norm(final_body_position - pre_gravity_body_position)
        )
        hold_displacement = float(
            np.linalg.norm(final_body_in_grasp - pre_gravity_body_in_grasp)
        )
        world_settling_speed = float(
            np.linalg.norm(final_body_position - last_window_start)
            / (window_steps / 240.0)
        )
        settling_speed = float(
            np.linalg.norm(final_body_in_grasp - last_window_relative_start)
            / (window_steps / 240.0)
        )
        hand_world_sag = float(
            np.linalg.norm(final_tcp_position - pre_gravity_tcp_position)
        )
        joint_separation_change = abs(
            float(
                np.linalg.norm(final_nut_position - final_body_position)
                - np.linalg.norm(
                    pre_gravity_nut_position - pre_gravity_body_position
                )
            )
        )
        mean_hold_efforts = (
            np.mean(np.stack(effort_samples), axis=0)
            if effort_samples
            else contact_efforts
        )
        loaded_channels = int(np.count_nonzero(np.abs(torque_deltas) >= 0.02))
        maximum_absolute_torque_delta = float(
            np.max(np.abs(torque_deltas))
        )
        final_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        final_velocities = np.asarray(
            robot.get_joint_velocities(), dtype=np.float64
        )
        finite = bool(
            finite_throughout
            and np.all(np.isfinite(final_positions))
            and np.all(np.isfinite(final_velocities))
            and np.all(np.isfinite(final_body_position))
            and np.all(np.isfinite(final_nut_position))
        )
        stable = bool(
            finite
            and max_abs_joint_velocity <= 20.0
            and max_joint_limit_violation <= 0.02
        )
        relative_grasp_ok = bool(
            hold_displacement <= 0.005 and settling_speed <= 0.002
        )
        world_pose_control_ok = bool(
            world_hold_displacement <= 0.005
            and hand_world_sag <= 0.005
            and world_settling_speed <= 0.002
        )
        final_contact_snapshot = metrics["final_contact_snapshot"]
        grip_material_contact_count = 0
        for contact_pair in final_contact_snapshot:
            pair_paths = (
                contact_pair["actor0"],
                contact_pair["actor1"],
                contact_pair["collider0"],
                contact_pair["collider1"],
            )
            is_hand_connector_pair = any(
                path.startswith(robot_root) for path in pair_paths
            ) and any(path.startswith(connector_root) for path in pair_paths)
            if is_hand_connector_pair:
                grip_material_contact_count += sum(
                    contact["material0"] == grip_material_path
                    and contact["material1"] == grip_material_path
                    for contact in contact_pair["contacts"]
                )
        passed = bool(
            stable
            and proxy_material_binding_ok
            and grip_material_contact_count > 0
            and loaded_channels >= 2
            and maximum_absolute_torque_delta
            <= curriculum["maximum_finger_torque"]
            and closure_displacement <= 0.005
            and unsupported_settle_displacement <= 0.003
            and relative_grasp_ok
            and world_pose_control_ok
            and joint_separation_change <= 0.001
        )
        metrics.update(
            {
                "body_position_before_gravity": (
                    pre_gravity_body_position.tolist()
                ),
                "body_position_initial": initial_body_position.tolist(),
                "body_position_final": final_body_position.tolist(),
                "body_position_in_grasp_before_gravity": (
                    pre_gravity_body_in_grasp.tolist()
                ),
                "body_position_in_grasp_final": final_body_in_grasp.tolist(),
                "closure_displacement": closure_displacement,
                "contact_torque_deltas": {
                    name: round(float(value), 6)
                    for name, value in zip(sensor_names, torque_deltas)
                },
                "finite": finite,
                "hold_displacement": hold_displacement,
                "hand_world_sag": hand_world_sag,
                "arm_joint_delta_during_hold": {
                    name: round(
                        float(
                            final_positions[name_to_index[name]]
                            - pre_gravity_joint_positions[name_to_index[name]]
                        ),
                        6,
                    )
                    for name in ARM_JOINT_NAMES
                },
                "arm_joint_target_error_final": {
                    name: round(
                        float(
                            arm_target[ARM_JOINT_NAMES.index(name)]
                            - final_positions[name_to_index[name]]
                        ),
                        6,
                    )
                    for name in ARM_JOINT_NAMES
                },
                "joint_separation_change": joint_separation_change,
                "loaded_torque_channels": loaded_channels,
                "maximum_absolute_finger_torque_delta": (
                    maximum_absolute_torque_delta
                ),
                "maximum_allowed_finger_torque": curriculum[
                    "maximum_finger_torque"
                ],
                "grip_material_contact_count": grip_material_contact_count,
                "mean_hold_efforts": {
                    name: round(float(value), 6)
                    for name, value in zip(sensor_names, mean_hold_efforts)
                },
                "nut_position_final": final_nut_position.tolist(),
                "nut_position_initial": initial_nut_position.tolist(),
                "max_abs_joint_velocity": max_abs_joint_velocity,
                "max_abs_measured_arm_effort": max_abs_measured_arm_effort,
                "max_joint_limit_violation": max(
                    0.0, max_joint_limit_violation
                ),
                "receptacle_removed_before_hold": True,
                "relative_grasp_ok": relative_grasp_ok,
                "support_removed_before_hold": True,
                "unsupported_settle_displacement": (
                    unsupported_settle_displacement
                ),
                "settling_speed": settling_speed,
                "stable": stable,
                "tcp_position_before_gravity": (
                    pre_gravity_tcp_position.tolist()
                ),
                "tcp_position_final": final_tcp_position.tolist(),
                "world_hold_displacement": world_hold_displacement,
                "world_pose_control_ok": world_pose_control_ok,
                "world_settling_speed": world_settling_speed,
                "passed": passed,
            }
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print(
            "ISAAC CONNECTOR PHYSICAL GRASP "
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
        print("ISAAC CONNECTOR PHYSICAL GRASP FAILED", flush=True)
    finally:
        if arguments.keep_open and arguments.gui:
            print(
                "ISAAC CONNECTOR PHYSICAL GRASP GUI REMAINS OPEN; "
                "close the window to exit",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
