#!/usr/bin/env python3

"""Physically screen deterministic tabletop pick of the synthetic plug.

The robot moves from Home through the accepted pregrasp seed, descends with
an open hand, closes by real PhysX contact, lifts back to pregrasp, and holds.
No attachment, object pose write, kinematic object drive, or fingertip sensor
is used.  This remains joint interpolation screening, not collision planning;
self collision is disabled in the imported robot and is reported unverified.
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
CAMERA_EYE_M = (1.55, 1.25, 1.05)
CAMERA_TARGET_M = (0.43, -0.08, 0.47)


def _path_is_at_or_below(path, root):
    value = str(path)
    prefix = str(root)
    return value == prefix or value.startswith(prefix + "/")


def _pair_contains_subtree(paths, root):
    return any(_path_is_at_or_below(path, root) for path in paths)


def _classify_robot_external_contact(
    paths,
    robot_root,
    table_path,
    fixture_path,
    receptacle_path,
    plug_path,
):
    """Classify only contacts containing the exact robot subtree."""

    values = tuple(str(path) for path in paths)
    if not _pair_contains_subtree(values, robot_root):
        return None
    for category, path in (
        ("table", table_path),
        ("fixture", fixture_path),
        ("fixed_endpoint", receptacle_path),
        ("loose_plug", plug_path),
    ):
        if _pair_contains_subtree(values, path):
            return category
    return None


def _is_plug_table_contact(paths, plug_path, table_path):
    values = tuple(str(path) for path in paths)
    return bool(
        _pair_contains_subtree(values, plug_path)
        and _pair_contains_subtree(values, table_path)
    )


def _is_finger_plug_contact(paths, robot_root, plug_path):
    """Require both loose plug and one modeled finger-link subtree."""

    values = tuple(str(path) for path in paths)
    finger_path_present = any(
        _path_is_at_or_below(path, robot_root)
        and any(
            link_name in path
            for link_name in ("/f1Link", "/f2Link", "/f3Link")
        )
        for path in values
    )
    return bool(
        finger_path_present and _pair_contains_subtree(values, plug_path)
    )


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


def _array_quaternion_error_radians(first, second):
    """Return shortest rotation angle for scalar-first quaternions."""

    first_values = tuple(float(value) for value in first)
    second_values = tuple(float(value) for value in second)
    first_norm = math.sqrt(sum(value * value for value in first_values))
    second_norm = math.sqrt(sum(value * value for value in second_values))
    if first_norm <= 0.0 or second_norm <= 0.0:
        return float("nan")
    dot = sum(
        left * right
        for left, right in zip(first_values, second_values)
    ) / (first_norm * second_norm)
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(dot))))


def _world_pose(Gf, Usd, UsdGeom, prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform(matrix)
    return transform.GetTranslation(), transform.GetRotation().GetQuat()


def _quaternion_world_z_axis(value):
    imaginary = value.GetImaginary()
    values = (
        float(value.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 0.0 or not math.isfinite(norm):
        return (float("nan"),) * 3
    w, x, y, z = (item / norm for item in values)
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
            / "src/kcg_connector/config/connector_tabletop_pick_v1.yaml"
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
        "attachment": "none",
        "fingertip_tactile_sensors": "none",
        "gui": arguments.gui,
        "keep_open": arguments.keep_open,
        "object_pose_writes_after_start": 0,
        "object_drive": "none",
        "passed": False,
        "scene": "kcg_connector_tabletop_pick_v1",
        "self_collision_enabled_in_asset": False,
        "self_collision_verified": False,
        "torque_channels": ["f1j2", "f2j1", "f3j2"],
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
        from kcg_connector.tabletop_pick import load_tabletop_pick_config

        config_path = Path(arguments.config).expanduser().resolve()
        robot_asset = Path(arguments.robot_asset).expanduser().resolve()
        connector_asset = Path(
            arguments.connector_asset
        ).expanduser().resolve()
        for path in (config_path, robot_asset, connector_asset):
            if not path.is_file():
                raise FileNotFoundError(path)
        pick = load_tabletop_pick_config(config_path)
        pregrasp_path = config_path.parent / pick.pregrasp.config
        pregrasp = load_home_to_pregrasp_config(pregrasp_path)
        tabletop_path = pregrasp_path.parent / pregrasp.scene.tabletop_config
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
            str(robot_asset), pregrasp.scene.robot_root_prim_path
        )
        tcp_prim = stage.GetPrimAtPath(pregrasp.scene.grasp_tcp_prim_path)
        fixed_prim = stage.GetPrimAtPath(
            tabletop.fixed_endpoint.receptacle_prim_path
        )
        for path, prim in (
            (pregrasp.scene.grasp_tcp_prim_path, tcp_prim),
            (tabletop.fixed_endpoint.receptacle_prim_path, fixed_prim),
        ):
            if not prim.IsValid():
                raise RuntimeError(f"required scene prim is missing: {path}")

        grip_material_path = "/World/TabletopPickGripMaterial"
        grip_material = UsdShade.Material.Define(stage, grip_material_path)
        grip_api = UsdPhysics.MaterialAPI.Apply(grip_material.GetPrim())
        grip_api.CreateStaticFrictionAttr(pick.motion.grip_static_friction)
        grip_api.CreateDynamicFrictionAttr(
            pick.motion.grip_dynamic_friction
        )
        grip_api.CreateRestitutionAttr(pick.motion.grip_restitution)
        finger_collision_anchors = []
        nut_collision_prims = []
        nut_prefix = tabletop.loose_endpoint.nut_prim_path + "/"
        robot_root = pregrasp.scene.robot_root_prim_path
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            finger_anchor = bool(
                prim_path.startswith(robot_root + "/")
                and prim.GetName().endswith("_convex")
                and any(
                    link_name in prim_path
                    for link_name in ("/f1Link", "/f2Link", "/f3Link")
                )
            )
            nut_collision = bool(
                prim.HasAPI(UsdPhysics.CollisionAPI)
                and prim_path.startswith(nut_prefix)
            )
            if finger_anchor or nut_collision:
                physicsUtils.add_physics_material_to_prim(
                    stage, prim, Sdf.Path(grip_material_path)
                )
                if finger_anchor:
                    finger_collision_anchors.append(prim_path)
                else:
                    nut_collision_prims.append(prim_path)
        if len(finger_collision_anchors) != 8:
            raise RuntimeError(
                "expected 8 finger collision anchors, found "
                f"{len(finger_collision_anchors)}"
            )
        if not nut_collision_prims:
            raise RuntimeError("no nut colliders received grip material")

        proxy_material_bindings = {}
        robot_prim = stage.GetPrimAtPath(robot_root)
        for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
            prim_path = str(prim.GetPath())
            if not (
                prim.IsInstanceProxy()
                and prim.HasAPI(UsdPhysics.CollisionAPI)
                and any(
                    link_name in prim_path
                    for link_name in ("/f1Link", "/f2Link", "/f3Link")
                )
            ):
                continue
            bound_material, _ = UsdShade.MaterialBindingAPI(
                prim
            ).ComputeBoundMaterial("physics")
            proxy_material_bindings[prim_path] = (
                str(bound_material.GetPath()) if bound_material else None
            )
        proxy_material_binding_ok = bool(
            len(proxy_material_bindings) == 8
            and all(
                material == grip_material_path
                for material in proxy_material_bindings.values()
            )
        )

        contact_report_body_count = 0
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            is_robot_body = bool(
                prim_path.startswith(robot_root + "/")
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            is_loose_body = prim_path in (
                tabletop.loose_endpoint.body_prim_path,
                tabletop.loose_endpoint.nut_prim_path,
            )
            if is_robot_body or is_loose_body:
                report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                report.CreateThresholdAttr().Set(0.0)
                if is_robot_body:
                    contact_report_body_count += 1
        if contact_report_body_count < 17:
            raise RuntimeError("robot contact reporting is incomplete")

        if arguments.gui:
            from isaacsim.core.rendering_manager import ViewportManager
            from pxr import UsdLux

            lighting_root = "/World/TabletopPickGuiLighting"
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
                prim_path=pregrasp.scene.articulation_prim_path,
                name="connector_tabletop_pick_handarm",
            )
        )
        body = world.scene.add(
            SingleRigidPrim(
                prim_path=tabletop.loose_endpoint.body_prim_path,
                name="tabletop_pick_plug_body",
            )
        )
        nut = world.scene.add(
            SingleRigidPrim(
                prim_path=tabletop.loose_endpoint.nut_prim_path,
                name="tabletop_pick_coupling_nut",
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
            [name_to_index[name] for name in pregrasp.robot.arm_joint_names],
            dtype=np.int32,
        )
        hand_indices = np.asarray(
            [
                name_to_index[name]
                for name in pregrasp.robot.active_hand_joint_names
            ],
            dtype=np.int32,
        )
        sensor_indices = np.asarray(
            [name_to_index[name] for name in pick.sensing.torque_joint_names],
            dtype=np.int32,
        )
        controlled_indices = np.concatenate((arm_indices, hand_indices))

        zero_positions = np.zeros(robot.num_dof, dtype=np.float32)
        robot.set_joint_positions(zero_positions)
        robot.set_joint_velocities(zero_positions)
        controller = robot.get_articulation_controller()
        kps = np.zeros(robot.num_dof, dtype=np.float32)
        kds = np.zeros(robot.num_dof, dtype=np.float32)
        kps[arm_indices] = pregrasp.robot.arm_stiffness
        kds[arm_indices] = pregrasp.robot.arm_damping
        kps[hand_indices] = pregrasp.robot.hand_stiffness
        kds[hand_indices] = pregrasp.robot.hand_damping
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        world.get_physics_context().set_gravity(
            tabletop.physics.gravity_m_s2
        )

        home_arm = np.asarray(pregrasp.robot.home_arm_rad, dtype=np.float64)
        open_hand = np.asarray(pregrasp.robot.open_hand_rad, dtype=np.float64)
        grasp_hand = np.asarray(pick.motion.grasp_hand_rad, dtype=np.float64)
        pregrasp_arm = np.asarray(
            pregrasp.motion.segments[-1].target_arm_rad,
            dtype=np.float64,
        )
        grasp_arm = np.asarray(pick.motion.grasp_arm_rad, dtype=np.float64)
        current_arm_target = home_arm.copy()
        current_hand_target = np.zeros(4, dtype=np.float64)
        dof_properties = robot.dof_properties

        finite_throughout = True
        maximum_joint_limit_violation = 0.0
        maximum_joint_speed = 0.0
        maximum_arm_tracking_error = 0.0
        external_contact_records = {
            "table": 0,
            "fixture": 0,
            "fixed_endpoint": 0,
            "loose_plug_preclosure": 0,
            "loose_plug_allowed": 0,
            "loose_plug_unexpected_robot_link": 0,
        }
        external_contact_headers = {
            key: 0 for key in external_contact_records
        }
        grip_material_contact_records = 0
        phase_steps = {}
        phase = "initial_settle"

        fixed_initial_position, fixed_initial_orientation = _world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )

        def body_in_tcp_frame(body_position):
            tcp_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
                tcp_prim
            )
            point = tcp_matrix.GetInverse().Transform(
                Gf.Vec3d(*(float(value) for value in body_position))
            )
            return np.asarray(point, dtype=np.float64)

        def contact_snapshot():
            headers, contacts, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            result = {
                "grip_material_records": 0,
                "finger_loose_plug_records": 0,
                "plug_table_records": 0,
                "robot_loose_plug_records": 0,
                "unexpected_robot_link_records": 0,
            }
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                if _is_plug_table_contact(
                    paths,
                    tabletop.loose_endpoint.plug_prim_path,
                    tabletop.table.prim_path,
                ):
                    result["plug_table_records"] += int(
                        header.num_contact_data
                    )
                category = _classify_robot_external_contact(
                    paths,
                    robot_root,
                    tabletop.table.prim_path,
                    tabletop.fixed_endpoint.fixture_prim_path,
                    tabletop.fixed_endpoint.receptacle_prim_path,
                    tabletop.loose_endpoint.plug_prim_path,
                )
                if category != "loose_plug":
                    continue
                result["robot_loose_plug_records"] += int(
                    header.num_contact_data
                )
                is_finger_contact = _is_finger_plug_contact(
                    paths, robot_root, tabletop.loose_endpoint.plug_prim_path
                )
                if not is_finger_contact:
                    result["unexpected_robot_link_records"] += int(
                        header.num_contact_data
                    )
                    continue
                result["finger_loose_plug_records"] += int(
                    header.num_contact_data
                )
                for index in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data,
                ):
                    contact = contacts[index]
                    materials = (
                        str(
                            PhysicsSchemaTools.intToSdfPath(
                                contact.material0
                            )
                        ),
                        str(
                            PhysicsSchemaTools.intToSdfPath(
                                contact.material1
                            )
                        ),
                    )
                    if materials == (
                        grip_material_path,
                        grip_material_path,
                    ):
                        result["grip_material_records"] += 1
            return result

        def observe_and_step(arm_target, hand_target, allow_loose_contact):
            nonlocal finite_throughout
            nonlocal maximum_joint_limit_violation
            nonlocal maximum_joint_speed
            nonlocal maximum_arm_tracking_error
            nonlocal grip_material_contact_records
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
            phase_steps[phase] = phase_steps.get(phase, 0) + 1
            positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            body_position, body_orientation = body.get_world_pose()
            nut_position, nut_orientation = nut.get_world_pose()
            body_linear = np.asarray(
                body.get_linear_velocity(), dtype=np.float64
            )
            body_angular = np.asarray(
                body.get_angular_velocity(), dtype=np.float64
            )
            sampled = np.concatenate(
                (
                    positions,
                    velocities,
                    np.asarray(body_position, dtype=np.float64),
                    np.asarray(body_orientation, dtype=np.float64),
                    np.asarray(nut_position, dtype=np.float64),
                    np.asarray(nut_orientation, dtype=np.float64),
                    body_linear,
                    body_angular,
                )
            )
            finite_throughout = bool(
                finite_throughout and np.all(np.isfinite(sampled))
            )
            maximum_joint_speed = max(
                maximum_joint_speed,
                float(np.max(np.abs(velocities))),
            )
            maximum_arm_tracking_error = max(
                maximum_arm_tracking_error,
                float(np.max(np.abs(positions[arm_indices] - arm_target))),
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

            headers, contacts, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                category = _classify_robot_external_contact(
                    paths,
                    robot_root,
                    tabletop.table.prim_path,
                    tabletop.fixed_endpoint.fixture_prim_path,
                    tabletop.fixed_endpoint.receptacle_prim_path,
                    tabletop.loose_endpoint.plug_prim_path,
                )
                if category is None:
                    continue
                key = category
                if category == "loose_plug":
                    if not allow_loose_contact:
                        key += "_preclosure"
                    elif _is_finger_plug_contact(
                        paths,
                        robot_root,
                        tabletop.loose_endpoint.plug_prim_path,
                    ):
                        key += "_allowed"
                    else:
                        key += "_unexpected_robot_link"
                external_contact_headers[key] += 1
                external_contact_records[key] += int(
                    header.num_contact_data
                )
                if key != "loose_plug_allowed":
                    continue
                for index in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data,
                ):
                    contact = contacts[index]
                    materials = (
                        str(
                            PhysicsSchemaTools.intToSdfPath(
                                contact.material0
                            )
                        ),
                        str(
                            PhysicsSchemaTools.intToSdfPath(
                                contact.material1
                            )
                        ),
                    )
                    if materials == (
                        grip_material_path,
                        grip_material_path,
                    ):
                        grip_material_contact_records += 1
            return positions, velocities

        phase = "initial_settle"
        for _ in range(tabletop.physics.settle_steps):
            observe_and_step(
                current_arm_target, current_hand_target, False
            )
        settled_body_position, _ = body.get_world_pose()
        settled_nut_position, _ = nut.get_world_pose()
        settled_bottom = min(
            float(settled_body_position[2])
            - tabletop.loose_endpoint.body_bottom_offset_m,
            float(settled_nut_position[2])
            - tabletop.loose_endpoint.nut_bottom_offset_m,
        )
        settled_on_table = bool(
            -tabletop.physics.maximum_table_penetration_m
            <= settled_bottom - tabletop.table.top_z_m
            <= tabletop.physics.maximum_surface_gap_m
        )

        phase = "home_hand_open"
        closed_home_hand = current_hand_target.copy()
        hand_open_steps = round(
            pregrasp.motion.hand_open_duration_s * rate_hz
        )
        for index in range(hand_open_steps):
            blend = minimum_jerk_blend(
                float(index + 1) / float(hand_open_steps)
            )
            current_hand_target = (
                closed_home_hand + blend * (open_hand - closed_home_hand)
            )
            observe_and_step(
                current_arm_target, current_hand_target, False
            )
        current_hand_target = open_hand.copy()

        for segment in pregrasp.motion.segments:
            phase = segment.name
            start_arm = current_arm_target.copy()
            final_arm = np.asarray(segment.target_arm_rad, dtype=np.float64)
            segment_steps = round(segment.duration_s * rate_hz)
            for index in range(segment_steps):
                current_arm_target = np.asarray(
                    interpolate_segment(
                        tuple(float(value) for value in start_arm),
                        tuple(float(value) for value in final_arm),
                        float(index + 1) / float(segment_steps),
                    ),
                    dtype=np.float64,
                )
                observe_and_step(
                    current_arm_target, current_hand_target, False
                )
            current_arm_target = final_arm.copy()

        phase = "pregrasp_hold"
        for _ in range(round(pregrasp.motion.hold_duration_s * rate_hz)):
            observe_and_step(
                current_arm_target, current_hand_target, False
            )

        phase = "open_hand_descent"
        descent_start = current_arm_target.copy()
        descent_steps = round(pick.motion.descent_duration_s * rate_hz)
        for index in range(descent_steps):
            current_arm_target = np.asarray(
                interpolate_segment(
                    tuple(float(value) for value in descent_start),
                    tuple(float(value) for value in grasp_arm),
                    float(index + 1) / float(descent_steps),
                ),
                dtype=np.float64,
            )
            observe_and_step(
                current_arm_target, current_hand_target, False
            )
        current_arm_target = grasp_arm.copy()

        # Change only hand drive gains at the open grasp pose, then tare the
        # three real one-dimensional base-torque channels before contact.
        kps[hand_indices] = pick.motion.grip_hand_stiffness
        kds[hand_indices] = pick.motion.grip_hand_damping
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        phase = "open_grasp_tare"
        tare_effort_samples = []
        tare_steps = round(pick.motion.open_tare_duration_s * rate_hz)
        for _ in range(tare_steps):
            observe_and_step(
                current_arm_target, current_hand_target, False
            )
            tare_effort_samples.append(
                np.asarray(
                    robot.get_measured_joint_efforts(
                        joint_indices=sensor_indices
                    ),
                    dtype=np.float64,
                )
            )
        tare_efforts = np.mean(np.stack(tare_effort_samples), axis=0)
        maximum_post_tare_absolute_delta_by_channel = np.zeros(
            len(sensor_indices), dtype=np.float64
        )

        def sample_post_tare_efforts():
            nonlocal finite_throughout
            measured = np.asarray(
                robot.get_measured_joint_efforts(
                    joint_indices=sensor_indices
                ),
                dtype=np.float64,
            )
            delta = measured - tare_efforts
            finite_throughout = bool(
                finite_throughout
                and np.all(np.isfinite(measured))
                and np.all(np.isfinite(delta))
            )
            np.maximum(
                maximum_post_tare_absolute_delta_by_channel,
                np.abs(delta),
                out=maximum_post_tare_absolute_delta_by_channel,
            )
            return measured

        grasp_tcp_position, grasp_tcp_orientation = _world_pose(
            Gf, Usd, UsdGeom, tcp_prim
        )
        grasp_tcp_position_error = float(
            np.linalg.norm(
                np.asarray(grasp_tcp_position, dtype=np.float64)
                - np.asarray(
                    pick.motion.grasp_tcp_position_m, dtype=np.float64
                )
            )
        )
        grasp_tcp_axis = _quaternion_world_z_axis(grasp_tcp_orientation)
        grasp_tcp_axis_error = _axis_error_radians(
            grasp_tcp_axis, pick.motion.grasp_tcp_down_axis_world
        )
        positions_at_grasp = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        grasp_endpoint_arm_error = float(
            np.max(np.abs(positions_at_grasp[arm_indices] - grasp_arm))
        )

        phase = "physical_hand_closure"
        closure_start = current_hand_target.copy()
        closure_steps = round(pick.motion.closure_duration_s * rate_hz)
        for index in range(closure_steps):
            blend = minimum_jerk_blend(
                float(index + 1) / float(closure_steps)
            )
            current_hand_target = (
                closure_start + blend * (grasp_hand - closure_start)
            )
            observe_and_step(
                current_arm_target, current_hand_target, True
            )
            sample_post_tare_efforts()
        current_hand_target = grasp_hand.copy()

        phase = "physical_grip_preload"
        preload_effort_samples = []
        preload_steps = round(pick.motion.preload_duration_s * rate_hz)
        for _ in range(preload_steps):
            observe_and_step(
                current_arm_target, current_hand_target, True
            )
            preload_effort_samples.append(sample_post_tare_efforts())
        contact_efforts = np.mean(
            np.stack(preload_effort_samples), axis=0
        )
        torque_deltas = contact_efforts - tare_efforts
        loaded_channels = int(
            np.count_nonzero(
                np.abs(torque_deltas)
                >= pick.sensing.loaded_torque_threshold_nm
            )
        )
        maximum_absolute_torque_delta = float(
            np.max(np.abs(torque_deltas))
        )
        postclosure_body_position, _ = body.get_world_pose()
        postclosure_nut_position, _ = nut.get_world_pose()
        postclosure_body_in_tcp = body_in_tcp_frame(
            postclosure_body_position
        )
        postclosure_body_nut_separation = float(
            np.linalg.norm(
                postclosure_nut_position - postclosure_body_position
            )
        )
        postclosure_contacts = contact_snapshot()

        phase = "physical_grip_lift"
        lift_start = current_arm_target.copy()
        lift_steps = round(pick.motion.lift_duration_s * rate_hz)
        for index in range(lift_steps):
            current_arm_target = np.asarray(
                interpolate_segment(
                    tuple(float(value) for value in lift_start),
                    tuple(float(value) for value in pregrasp_arm),
                    float(index + 1) / float(lift_steps),
                ),
                dtype=np.float64,
            )
            observe_and_step(
                current_arm_target, current_hand_target, True
            )
            sample_post_tare_efforts()
        current_arm_target = pregrasp_arm.copy()

        phase = "unsupported_final_hold"
        hold_start_body_position, _ = body.get_world_pose()
        maximum_final_hold_displacement = 0.0
        final_effort_samples = []
        final_hold_steps = round(
            pick.motion.final_hold_duration_s * rate_hz
        )
        tail_window_steps = min(120, final_hold_steps)
        tail_solver_velocity_samples = []
        tail_pose_difference_velocity_samples = []
        tail_body_pose_difference_linear_speeds = []
        tail_body_pose_difference_angular_speeds = []
        previous_tail_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        previous_tail_body_position, previous_tail_body_orientation = (
            body.get_world_pose()
        )
        previous_tail_body_position = np.asarray(
            previous_tail_body_position, dtype=np.float64
        )
        previous_tail_body_orientation = np.asarray(
            previous_tail_body_orientation, dtype=np.float64
        )
        effort_sample_steps = round(
            pick.motion.effort_sample_duration_s * rate_hz
        )
        for index in range(final_hold_steps):
            observe_and_step(
                current_arm_target, current_hand_target, True
            )
            measured_final_effort = sample_post_tare_efforts()
            current_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            current_solver_velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            current_body_position, current_body_orientation = (
                body.get_world_pose()
            )
            current_body_position = np.asarray(
                current_body_position, dtype=np.float64
            )
            current_body_orientation = np.asarray(
                current_body_orientation, dtype=np.float64
            )
            if index >= final_hold_steps - tail_window_steps:
                tail_solver_velocity_samples.append(
                    current_solver_velocities.copy()
                )
                tail_pose_difference_velocity_samples.append(
                    (current_positions - previous_tail_positions) * rate_hz
                )
                tail_body_pose_difference_linear_speeds.append(
                    float(
                        np.linalg.norm(
                            current_body_position
                            - previous_tail_body_position
                        )
                        * rate_hz
                    )
                )
                tail_body_pose_difference_angular_speeds.append(
                    _array_quaternion_error_radians(
                        previous_tail_body_orientation,
                        current_body_orientation,
                    )
                    * rate_hz
                )
            previous_tail_positions = current_positions
            previous_tail_body_position = current_body_position
            previous_tail_body_orientation = current_body_orientation
            maximum_final_hold_displacement = max(
                maximum_final_hold_displacement,
                float(
                    np.linalg.norm(
                        current_body_position - hold_start_body_position
                    )
                ),
            )
            if index >= final_hold_steps - effort_sample_steps:
                final_effort_samples.append(measured_final_effort.copy())

        final_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        final_velocities = np.asarray(
            robot.get_joint_velocities(), dtype=np.float64
        )
        final_joint_speed_by_name = {
            name: float(abs(final_velocities[index]))
            for index, name in enumerate(dof_names)
        }
        tail_solver_velocities = np.abs(
            np.stack(tail_solver_velocity_samples)
        )
        tail_pose_difference_velocities = np.abs(
            np.stack(tail_pose_difference_velocity_samples)
        )
        maximum_final_observable_joint_speed = float(
            np.max(tail_pose_difference_velocities)
        )
        maximum_final_post_solver_joint_speed = float(
            np.max(tail_solver_velocities)
        )
        tail_solver_speed_peak_by_name = {
            name: float(np.max(tail_solver_velocities[:, index]))
            for index, name in enumerate(dof_names)
        }
        tail_solver_speed_median_by_name = {
            name: float(np.median(tail_solver_velocities[:, index]))
            for index, name in enumerate(dof_names)
        }
        tail_pose_difference_speed_peak_by_name = {
            name: float(
                np.max(tail_pose_difference_velocities[:, index])
            )
            for index, name in enumerate(dof_names)
        }
        tail_pose_difference_speed_median_by_name = {
            name: float(
                np.median(tail_pose_difference_velocities[:, index])
            )
            for index, name in enumerate(dof_names)
        }
        final_body_position, final_body_orientation = body.get_world_pose()
        final_nut_position, final_nut_orientation = nut.get_world_pose()
        final_body_linear_speed = float(
            np.linalg.norm(body.get_linear_velocity())
        )
        final_body_angular_speed = float(
            np.linalg.norm(body.get_angular_velocity())
        )
        fixed_final_position, fixed_final_orientation = _world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )
        final_contacts = contact_snapshot()
        final_body_in_tcp = body_in_tcp_frame(final_body_position)
        body_tcp_slip = float(
            np.linalg.norm(final_body_in_tcp - postclosure_body_in_tcp)
        )
        body_nut_separation_change = abs(
            float(
                np.linalg.norm(final_nut_position - final_body_position)
                - postclosure_body_nut_separation
            )
        )
        body_lift = float(
            final_body_position[2] - settled_body_position[2]
        )
        final_bottom = min(
            float(final_body_position[2])
            - tabletop.loose_endpoint.body_bottom_offset_m,
            float(final_nut_position[2])
            - tabletop.loose_endpoint.nut_bottom_offset_m,
        )
        final_bottom_clearance = final_bottom - tabletop.table.top_z_m
        final_arm_tracking_error = float(
            np.max(
                np.abs(final_positions[arm_indices] - current_arm_target)
            )
        )
        final_joint_speed = float(np.max(np.abs(final_velocities)))
        fixed_translation_drift = float(
            np.linalg.norm(
                np.asarray(fixed_final_position, dtype=np.float64)
                - np.asarray(fixed_initial_position, dtype=np.float64)
            )
        )
        fixed_rotation_drift = _gf_quaternion_error_radians(
            fixed_initial_orientation, fixed_final_orientation
        )
        final_efforts = np.mean(np.stack(final_effort_samples), axis=0)
        final_torque_deltas = final_efforts - tare_efforts
        final_loaded_channels = int(
            np.count_nonzero(
                np.abs(final_torque_deltas)
                >= pick.sensing.loaded_torque_threshold_nm
            )
        )
        final_maximum_absolute_torque_delta = float(
            np.max(np.abs(final_torque_deltas))
        )
        maximum_post_tare_absolute_torque_delta = float(
            np.max(maximum_post_tare_absolute_delta_by_channel)
        )
        finite_final = bool(
            np.all(np.isfinite(final_positions))
            and np.all(np.isfinite(final_velocities))
            and np.all(np.isfinite(final_body_position))
            and np.all(np.isfinite(final_body_orientation))
            and np.all(np.isfinite(final_nut_position))
            and np.all(np.isfinite(final_nut_orientation))
            and np.all(np.isfinite(tare_efforts))
            and np.all(np.isfinite(contact_efforts))
            and np.all(np.isfinite(final_efforts))
            and _gf_quaternion_finite(fixed_final_orientation)
        )
        tail_diagnostics_finite = bool(
            tail_window_steps == 120
            and len(tail_solver_velocity_samples) == tail_window_steps
            and len(tail_pose_difference_velocity_samples)
            == tail_window_steps
            and len(tail_body_pose_difference_linear_speeds)
            == tail_window_steps
            and len(tail_body_pose_difference_angular_speeds)
            == tail_window_steps
            and np.all(np.isfinite(tail_solver_velocities))
            and np.all(np.isfinite(tail_pose_difference_velocities))
            and np.all(
                np.isfinite(
                    tail_body_pose_difference_linear_speeds
                )
            )
            and np.all(
                np.isfinite(
                    tail_body_pose_difference_angular_speeds
                )
            )
        )

        acceptance = pick.acceptance
        zero_forbidden_contacts = bool(
            external_contact_records["table"] == 0
            and external_contact_records["fixture"] == 0
            and external_contact_records["fixed_endpoint"] == 0
            and external_contact_records["loose_plug_preclosure"] == 0
            and external_contact_records[
                "loose_plug_unexpected_robot_link"
            ]
            == 0
        )
        force_gate = bool(
            loaded_channels >= pick.sensing.minimum_loaded_channels
            and maximum_absolute_torque_delta
            <= pick.sensing.maximum_absolute_torque_delta_nm
            and final_loaded_channels
            >= pick.sensing.minimum_loaded_channels
            and final_maximum_absolute_torque_delta
            <= pick.sensing.maximum_absolute_torque_delta_nm
            and maximum_post_tare_absolute_torque_delta
            <= pick.sensing.maximum_absolute_torque_delta_nm
        )
        physical_contact_gate = bool(
            proxy_material_binding_ok
            and grip_material_contact_records > 0
            and postclosure_contacts["grip_material_records"] > 0
            and final_contacts["grip_material_records"] > 0
            and final_contacts["finger_loose_plug_records"] > 0
            and final_contacts["unexpected_robot_link_records"] == 0
        )
        final_unsupported = bool(
            final_contacts["plug_table_records"] == 0
            and body_lift >= acceptance.minimum_body_lift_m
            and final_bottom_clearance
            >= acceptance.minimum_final_bottom_clearance_m
        )
        passed = bool(
            finite_throughout
            and finite_final
            and tail_diagnostics_finite
            and settled_on_table
            and maximum_joint_limit_violation
            <= acceptance.maximum_joint_limit_violation_rad
            and maximum_joint_speed
            <= acceptance.maximum_observed_joint_speed_rad_s
            and maximum_arm_tracking_error
            <= acceptance.maximum_arm_tracking_error_rad
            and grasp_endpoint_arm_error
            <= acceptance.maximum_endpoint_arm_tracking_error_rad
            and final_arm_tracking_error
            <= acceptance.maximum_endpoint_arm_tracking_error_rad
            and maximum_final_observable_joint_speed
            <= acceptance.maximum_final_observable_joint_speed_rad_s
            and maximum_final_post_solver_joint_speed
            <= acceptance.maximum_final_post_solver_joint_speed_rad_s
            and grasp_tcp_position_error
            <= acceptance.maximum_grasp_tcp_position_error_m
            and grasp_tcp_axis_error
            <= acceptance.maximum_grasp_tcp_axis_error_rad
            and force_gate
            and physical_contact_gate
            and zero_forbidden_contacts
            and final_unsupported
            and body_tcp_slip <= acceptance.maximum_body_tcp_slip_m
            and body_nut_separation_change
            <= acceptance.maximum_body_nut_separation_change_m
            and maximum_final_hold_displacement
            <= acceptance.maximum_final_hold_displacement_m
            and final_body_linear_speed
            <= acceptance.maximum_final_body_linear_speed_m_s
            and final_body_angular_speed
            <= acceptance.maximum_final_body_angular_speed_rad_s
            and fixed_translation_drift
            <= acceptance.maximum_fixed_translation_drift_m
            and fixed_rotation_drift
            <= acceptance.maximum_fixed_rotation_drift_rad
        )
        metrics.update(
            {
                "body_lift_m": body_lift,
                "body_nut_separation_change_m": (
                    body_nut_separation_change
                ),
                "body_tcp_slip_m": body_tcp_slip,
                "contact_efforts_nm": {
                    name: float(value)
                    for name, value in zip(
                        pick.sensing.torque_joint_names, contact_efforts
                    )
                },
                "contact_torque_deltas_nm": {
                    name: float(value)
                    for name, value in zip(
                        pick.sensing.torque_joint_names, torque_deltas
                    )
                },
                "external_contact_headers": external_contact_headers,
                "external_contact_records": external_contact_records,
                "final_arm_tracking_error_rad": (
                    final_arm_tracking_error
                ),
                "final_body_angular_speed_rad_s": (
                    final_body_angular_speed
                ),
                "final_body_linear_speed_m_s": final_body_linear_speed,
                "final_bottom_clearance_m": final_bottom_clearance,
                "final_contacts": final_contacts,
                "final_hold_displacement_m": (
                    maximum_final_hold_displacement
                ),
                "final_joint_speed_rad_s": final_joint_speed,
                "final_joint_speed_rad_s_semantics": (
                    "diagnostic_alias_post_solver_single_frame_"
                    "not_observable_motion"
                ),
                "final_joint_speed_by_name_rad_s": (
                    final_joint_speed_by_name
                ),
                "final_tail_window_steps": tail_window_steps,
                "final_tail_diagnostics_finite": (
                    tail_diagnostics_finite
                ),
                "final_observable_joint_speed_peak_rad_s": (
                    maximum_final_observable_joint_speed
                ),
                "final_observable_joint_speed_threshold_rad_s": (
                    acceptance.maximum_final_observable_joint_speed_rad_s
                ),
                "final_post_solver_joint_speed_peak_rad_s": (
                    maximum_final_post_solver_joint_speed
                ),
                "final_post_solver_joint_speed_threshold_rad_s": (
                    acceptance.maximum_final_post_solver_joint_speed_rad_s
                ),
                "final_speed_gate_semantics": {
                    "observable": (
                        "peak absolute integrated-position difference "
                        "over final 120 physics steps"
                    ),
                    "post_solver_health": (
                        "peak absolute PhysX post-solver velocity over "
                        "final 120 physics steps; empirical v1 health gate"
                    ),
                },
                "final_tail_solver_speed_peak_by_name_rad_s": (
                    tail_solver_speed_peak_by_name
                ),
                "final_tail_solver_speed_median_by_name_rad_s": (
                    tail_solver_speed_median_by_name
                ),
                "final_tail_pose_difference_speed_peak_by_name_rad_s": (
                    tail_pose_difference_speed_peak_by_name
                ),
                "final_tail_pose_difference_speed_median_by_name_rad_s": (
                    tail_pose_difference_speed_median_by_name
                ),
                "final_tail_body_pose_difference_linear_speed_peak_m_s": (
                    float(
                        np.max(
                            tail_body_pose_difference_linear_speeds
                        )
                    )
                ),
                "final_tail_body_pose_difference_linear_speed_median_m_s": (
                    float(
                        np.median(
                            tail_body_pose_difference_linear_speeds
                        )
                    )
                ),
                "final_tail_body_pose_difference_angular_speed_peak_rad_s": (
                    float(
                        np.max(
                            tail_body_pose_difference_angular_speeds
                        )
                    )
                ),
                "final_tail_body_pose_difference_angular_speed_median_rad_s": (
                    float(
                        np.median(
                            tail_body_pose_difference_angular_speeds
                        )
                    )
                ),
                "final_loaded_torque_channels": final_loaded_channels,
                "final_maximum_absolute_torque_delta_nm": (
                    final_maximum_absolute_torque_delta
                ),
                "final_torque_deltas_nm": {
                    name: float(value)
                    for name, value in zip(
                        pick.sensing.torque_joint_names,
                        final_torque_deltas,
                    )
                },
                "final_unsupported": final_unsupported,
                "finite_final": finite_final,
                "finite_throughout": finite_throughout,
                "fixed_rotation_drift_rad": fixed_rotation_drift,
                "fixed_translation_drift_m": fixed_translation_drift,
                "force_gate": force_gate,
                "grasp_endpoint_arm_error_rad": (
                    grasp_endpoint_arm_error
                ),
                "grasp_material_contact_records_total": (
                    grip_material_contact_records
                ),
                "grasp_tcp_axis_error_rad": grasp_tcp_axis_error,
                "grasp_tcp_position_error_m": grasp_tcp_position_error,
                "joint_limit_violation_rad": max(
                    0.0, maximum_joint_limit_violation
                ),
                "loaded_torque_channels": loaded_channels,
                "maximum_absolute_torque_delta_nm": (
                    maximum_absolute_torque_delta
                ),
                "maximum_post_tare_absolute_delta_by_channel_nm": {
                    name: float(value)
                    for name, value in zip(
                        pick.sensing.torque_joint_names,
                        maximum_post_tare_absolute_delta_by_channel,
                    )
                },
                "maximum_post_tare_absolute_torque_delta_nm": (
                    maximum_post_tare_absolute_torque_delta
                ),
                "maximum_arm_tracking_error_rad": (
                    maximum_arm_tracking_error
                ),
                "maximum_joint_speed_rad_s": maximum_joint_speed,
                "phase_steps": phase_steps,
                "physical_contact_gate": physical_contact_gate,
                "postclosure_contacts": postclosure_contacts,
                "proxy_material_binding_ok": proxy_material_binding_ok,
                "settled_on_table": settled_on_table,
                "tare_efforts_nm": {
                    name: float(value)
                    for name, value in zip(
                        pick.sensing.torque_joint_names, tare_efforts
                    )
                },
                "zero_forbidden_contacts": zero_forbidden_contacts,
                "passed": passed,
            }
        )
        json.dumps(metrics, allow_nan=False, sort_keys=True)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print(
            "ISAAC CONNECTOR TABLETOP PICK V1 "
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
        print("ISAAC CONNECTOR TABLETOP PICK V1 FAILED", flush=True)
    finally:
        if arguments.keep_open and arguments.gui:
            print(
                "ISAAC CONNECTOR TABLETOP PICK V1 GUI REMAINS OPEN; "
                "close the window to exit",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
