#!/usr/bin/env python3

"""Prove a short connector twist is caused only by KUKA joint 7."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

import yaml


def _load_task(config_path):
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    robot = document["robot_curriculum"]
    geometry = document["geometry"]
    success = document["success"]
    q7 = document["q7_twist"]
    result = {
        "pre_arm": robot["pre_approach_joint_positions"],
        "approach_arm": robot["approach_joint_positions"],
        "grasp_arm": robot["connector_grasp_joint_positions"],
        "insert_arm": robot["connector_insert_joint_positions"],
        "grasp_fraction": robot["connector_grasp_approach_fraction"],
        "open_hand": robot["open_hand_positions"],
        "grasp_hand": robot["grasp_hand_positions"],
        "grasp_center": robot["grasp_center_world"],
        "plug_height": geometry["initial_plug_center_height"],
        "lead": success["helical_lead_per_revolution"],
        "engage_depth": success["engage_depth"],
        "helical_error_tolerance": success[
            "helical_error_tolerance"
        ],
        "minimum_loaded_channels": success[
            "minimum_loaded_torque_channels"
        ],
        "maximum_finger_torque": success[
            "maximum_absolute_finger_torque"
        ],
        "q7_safe_lower": q7["safe_lower_rad"],
        "q7_safe_upper": q7["safe_upper_rad"],
        "q7_tightening_direction": q7["tightening_direction"],
        "probe_degrees": q7["probe_degrees"],
        "probe_speed_degrees": q7[
            "probe_speed_degrees_per_second"
        ],
        "maximum_segment_degrees": q7["maximum_segment_degrees"],
        "maximum_speed_degrees": q7["maximum_speed"],
        "regrasp_clearance": q7["regrasp_clearance_m"],
        "target_coupling_degrees": success[
            "target_coupling_angle_degrees"
        ],
        "coupling_tolerance_degrees": success[
            "coupling_angle_tolerance_degrees"
        ],
        "hold_duration": success["hold_duration"],
    }
    for name, length in (
        ("pre_arm", 7),
        ("approach_arm", 7),
        ("grasp_arm", 7),
        ("insert_arm", 7),
        ("open_hand", 4),
        ("grasp_hand", 4),
        ("grasp_center", 3),
    ):
        values = [float(value) for value in result[name]]
        if len(values) != length or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError(f"invalid task vector: {name}")
        result[name] = values
    scalar_names = (
        "grasp_fraction",
        "plug_height",
        "lead",
        "engage_depth",
        "helical_error_tolerance",
        "maximum_finger_torque",
        "q7_safe_lower",
        "q7_safe_upper",
        "probe_degrees",
        "probe_speed_degrees",
        "maximum_segment_degrees",
        "maximum_speed_degrees",
        "regrasp_clearance",
        "target_coupling_degrees",
        "coupling_tolerance_degrees",
        "hold_duration",
    )
    for name in scalar_names:
        result[name] = float(result[name])
        if not math.isfinite(result[name]):
            raise ValueError(f"invalid task scalar: {name}")
    result["minimum_loaded_channels"] = int(
        result["minimum_loaded_channels"]
    )
    result["q7_tightening_direction"] = int(
        result["q7_tightening_direction"]
    )
    if not 0.0 <= result["grasp_fraction"] <= 1.0:
        raise ValueError("grasp fraction must be in [0, 1]")
    positive_names = (
        "plug_height",
        "lead",
        "engage_depth",
        "maximum_finger_torque",
        "probe_degrees",
        "probe_speed_degrees",
        "maximum_segment_degrees",
        "maximum_speed_degrees",
        "regrasp_clearance",
        "target_coupling_degrees",
        "coupling_tolerance_degrees",
        "hold_duration",
    )
    if not all(result[name] > 0.0 for name in positive_names):
        raise ValueError("task magnitudes must be positive")
    if result["q7_safe_lower"] >= result["q7_safe_upper"]:
        raise ValueError("invalid q7 safe window")
    if not 1 <= result["minimum_loaded_channels"] <= 3:
        raise ValueError("invalid loaded torque channel count")
    if result["q7_tightening_direction"] not in (-1, 1):
        raise ValueError("q7 tightening direction must be -1 or 1")
    return result


def _wrapped_relative_z_angle(Gf, Usd, UsdGeom, body_prim, nut_prim):
    body_matrix = UsdGeom.Xformable(
        body_prim
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    nut_matrix = UsdGeom.Xformable(
        nut_prim
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    body_quaternion = Gf.Transform(body_matrix).GetRotation().GetQuat()
    nut_quaternion = Gf.Transform(nut_matrix).GetRotation().GetQuat()
    relative = body_quaternion.GetInverse() * nut_quaternion
    imaginary = relative.GetImaginary()
    angle = 2.0 * math.atan2(
        float(imaginary[2]), float(relative.GetReal())
    )
    return math.atan2(math.sin(angle), math.cos(angle))


def _unwrap(previous, wrapped):
    previous_wrapped = math.atan2(math.sin(previous), math.cos(previous))
    delta = math.atan2(
        math.sin(wrapped - previous_wrapped),
        math.cos(wrapped - previous_wrapped),
    )
    return previous + delta


def _matrix_pose(Gf, Usd, UsdGeom, prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform(matrix)
    return (
        transform.GetTranslation(),
        transform.GetRotation().GetQuat(),
    )


def _quaternion_error_radians(first, second):
    relative = first.GetInverse() * second
    real = max(-1.0, min(1.0, abs(float(relative.GetReal()))))
    return 2.0 * math.acos(real)


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
            repository
            / "artifacts/kcg_connector/isaac/connector_pair.usda"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(
            repository / "src/kcg_connector/config/connector_task.yaml"
        ),
    )
    parser.add_argument(
        "--residual-randomization-config",
        default=None,
        help=(
            "optional seeded residual-v0 randomization YAML; formal train/"
            "evaluate modes use the repository v1 YAML by default"
        ),
    )
    parser.add_argument(
        "--residual-stage",
        choices=("stage20", "stage60", "stage120"),
        default="stage20",
        help=(
            "versioned single-stroke residual curriculum stage; defaults "
            "to stage20"
        ),
    )
    parser.add_argument(
        "--fixed-residual-domain",
        action="store_true",
        help=(
            "explicitly disable the default v1 randomization in formal "
            "train/evaluate modes"
        ),
    )
    parser.add_argument(
        "--reset-seed",
        type=int,
        default=None,
        help=(
            "repeatable backend seed for residual zero/action-effect gates"
        ),
    )
    parser.add_argument("--probe-degrees", type=float, default=None)
    parser.add_argument("--probe-speed-degrees", type=float, default=None)
    parser.add_argument("--direction", type=int, choices=(-1, 1))
    parser.add_argument(
        "--mode",
        choices=(
            "twist",
            "q7-static-axial",
            "open-hand",
            "segmented",
            "residual-zero",
            "residual-action-effect",
            "residual-sac-smoke",
            "residual-train",
            "residual-evaluate",
            "residual-paired-evaluate",
        ),
        default="twist",
    )
    parser.add_argument(
        "--axial-counterfactual-distance",
        type=float,
        default=0.002,
        help="extra nominal insertion command used only by q7-static-axial",
    )
    parser.add_argument("--closure-steps", type=int, default=480)
    parser.add_argument("--insertion-steps", type=int, default=720)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="episode count used only by residual-zero mode",
    )
    parser.add_argument(
        "--maximum-episode-steps",
        type=int,
        default=None,
        help=(
            "deprecated validation-only override; when supplied it must "
            "equal the selected curriculum stage limit"
        ),
    )
    parser.add_argument(
        "--training-timesteps",
        type=int,
        default=32,
        help="SAC timesteps used only by residual-sac-smoke mode",
    )
    parser.add_argument(
        "--action-effect-steps",
        type=int,
        default=10,
        help=(
            "10 Hz policy steps per case used only by "
            "residual-action-effect mode"
        ),
    )
    parser.add_argument(
        "--training-output",
        default=str(
            repository
            / "artifacts/kcg_connector/residual_sac_smoke"
        ),
    )
    parser.add_argument(
        "--formal-run-config",
        default=str(
            repository
            / "src/kcg_rl/config/connector_residual_sac.yaml"
        ),
        help="formal SAC algorithm/run configuration",
    )
    parser.add_argument(
        "--formal-timesteps",
        type=int,
        default=None,
        help=(
            "required explicit timestep count used only by residual-train"
        ),
    )
    parser.add_argument(
        "--allow-long-training",
        action="store_true",
        help="explicitly permit formal training beyond the config dry-run cap",
    )
    parser.add_argument(
        "--formal-output-root",
        default=str(
            repository
            / "artifacts/kcg_connector/residual_sac_v0"
        ),
        help="parent of unique formal train/evaluate run directories",
    )
    parser.add_argument(
        "--formal-run-dir",
        default=None,
        help=(
            "complete formal training run directory required by "
            "residual-evaluate/residual-paired-evaluate"
        ),
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=None,
        help="override formal deterministic evaluation episode count",
    )
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
    if arguments.reset_seed is not None and arguments.reset_seed < 0:
        parser.error("--reset-seed must be nonnegative")
    if (
        arguments.fixed_residual_domain
        and arguments.residual_randomization_config is not None
    ):
        parser.error(
            "--fixed-residual-domain conflicts with "
            "--residual-randomization-config"
        )
    if (
        arguments.mode in (
            "residual-train",
            "residual-evaluate",
            "residual-paired-evaluate",
        )
        and arguments.residual_randomization_config is None
        and not arguments.fixed_residual_domain
    ):
        arguments.residual_randomization_config = str(
            repository
            / "src/kcg_connector/config/"
            "connector_residual_randomization_v1.yaml"
        )
    if arguments.keep_open and not arguments.gui:
        parser.error("--keep-open requires --gui")
    if (
        arguments.mode == "residual-sac-smoke"
        and not 2 <= arguments.training_timesteps <= 64
    ):
        parser.error(
            "residual-sac-smoke requires 2..64 timesteps; use "
            "residual-train for formal runs"
        )
    if (
        arguments.mode == "residual-train"
        and arguments.formal_timesteps is None
    ):
        parser.error(
            "residual-train requires an explicit --formal-timesteps value"
        )
    if (
        arguments.mode in (
            "residual-evaluate",
            "residual-paired-evaluate",
        )
        and arguments.formal_run_dir is None
    ):
        parser.error(
            "formal evaluation requires --formal-run-dir"
        )
    banner = {
        "twist": "ISAAC CONNECTOR Q7 PHYSICAL TWIST",
        "q7-static-axial": (
            "ISAAC CONNECTOR Q7-STATIC AXIAL COUNTERFACTUAL"
        ),
        "open-hand": "ISAAC CONNECTOR OPEN-HAND COUNTERFACTUAL",
        "segmented": "ISAAC CONNECTOR Q7 SEGMENTED 360 TWIST",
        "residual-zero": "ISAAC CONNECTOR ZERO-RESIDUAL 2-EPISODE",
        "residual-action-effect": (
            "ISAAC CONNECTOR RESIDUAL ACTION EFFECT"
        ),
        "residual-sac-smoke": "ISAAC CONNECTOR CUDA SAC TRAIN SMOKE",
        "residual-train": "ISAAC CONNECTOR FORMAL RESIDUAL SAC TRAIN",
        "residual-evaluate": (
            "ISAAC CONNECTOR FORMAL RESIDUAL SAC EVALUATION"
        ),
        "residual-paired-evaluate": (
            "ISAAC CONNECTOR FORMAL PAIRED ZERO VS MODEL BENCHMARK"
        ),
    }[arguments.mode]

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
        "actuated_rotation_joint": "iiwa_joint_7",
        "finger_role": "clamp_only",
        "nut_direct_drive": False,
        "object_pose_writes_after_start": 0,
        "gripper_attachment": "none",
        "gui": arguments.gui,
        "keep_open": arguments.keep_open,
        "mode": arguments.mode,
        "thread_constraint": "world_prismatic+rack_proxy",
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
        from kcg_connector.isaac_residual_backend import (
            ConnectorResidualIsaacBackend,
            PreparedConnectorScene,
            RuntimeThreadSpec,
            create_runtime_thread,
            summarize_reset_diagnostics,
        )
        from kcg_connector.thread_proxy import rack_and_pinion_ratio
        from kcg_connector.trajectory import (
            Q7Action,
            load_q7_twist_config,
        )

        robot_asset = Path(arguments.robot_asset).expanduser().resolve()
        connector_asset = Path(
            arguments.connector_asset
        ).expanduser().resolve()
        config_path = Path(arguments.config).expanduser().resolve()
        randomization_config_path = None
        if arguments.residual_randomization_config is not None:
            randomization_config_path = Path(
                arguments.residual_randomization_config
            ).expanduser().resolve()
        curriculum_config_path = (
            repository
            / "src/kcg_connector/config/"
            "connector_residual_curriculum_v1.yaml"
        ).resolve()
        required_paths = [robot_asset, connector_asset, config_path]
        if randomization_config_path is not None:
            required_paths.append(randomization_config_path)
        if arguments.mode.startswith("residual-"):
            required_paths.append(curriculum_config_path)
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        if arguments.closure_steps < 120:
            raise ValueError("closure must run for at least 120 steps")
        if arguments.insertion_steps < 120:
            raise ValueError("insertion must run for at least 120 steps")
        if arguments.settle_steps < 30:
            raise ValueError("settle must run for at least 30 steps")
        if not math.isfinite(
            arguments.axial_counterfactual_distance
        ) or not (
            0.0 < arguments.axial_counterfactual_distance <= 0.005
        ):
            raise ValueError(
                "axial counterfactual distance must be in (0, 0.005]"
            )

        task = _load_task(config_path)
        probe_degrees = (
            task["probe_degrees"]
            if arguments.probe_degrees is None
            else float(arguments.probe_degrees)
        )
        probe_speed_degrees = (
            task["probe_speed_degrees"]
            if arguments.probe_speed_degrees is None
            else float(arguments.probe_speed_degrees)
        )
        if not math.isfinite(probe_degrees) or not (
            0.0 < probe_degrees <= task["probe_degrees"]
        ):
            raise ValueError(
                "probe degrees must be positive and no larger than config"
            )
        if not math.isfinite(probe_speed_degrees) or not (
            0.0 < probe_speed_degrees <= task["probe_speed_degrees"]
        ):
            raise ValueError(
                "probe speed must be positive and no larger than config"
            )
        tightening_direction = (
            task["q7_tightening_direction"]
            if arguments.direction is None
            else arguments.direction
        )
        probe_angle = math.radians(probe_degrees) * tightening_direction
        probe_speed = math.radians(probe_speed_degrees)
        metrics.update(
            {
                "q7_tightening_direction": tightening_direction,
                "probe_degrees": probe_degrees,
                "probe_speed_degrees_per_second": probe_speed_degrees,
            }
        )

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
        body_path = f"{connector_root}/Plug/BodyAssembly"
        nut_path = f"{connector_root}/Plug/CouplingNut"
        hinge_path = f"{connector_root}/Plug/CouplingNutJoint"
        add_reference_to_stage(str(robot_asset), robot_root)
        connector_prim = add_reference_to_stage(
            str(connector_asset), connector_root
        )
        stage = get_current_stage()
        nested_scene_path = f"{connector_root}/PhysicsScene"
        if stage.GetPrimAtPath(nested_scene_path).IsValid():
            stage.RemovePrim(nested_scene_path)

        grasp_center = np.asarray(
            task["grasp_center"], dtype=np.float64
        )
        connector_origin = grasp_center.copy()
        connector_origin[2] -= task["plug_height"]
        UsdGeom.Xformable(connector_prim).AddTranslateOp().Set(
            Gf.Vec3d(*connector_origin)
        )

        if arguments.gui:
            from isaacsim.core.rendering_manager import ViewportManager
            from pxr import UsdLux

            # Keep the interactive proof focused on the hand, coupling nut,
            # and receptacle.  These display-only prims and the viewport
            # camera do not participate in physics.
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

            camera_target = grasp_center + np.asarray(
                [0.0, 0.0, -0.015], dtype=np.float64
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

        support_path = "/World/ConnectorTwistSupport"
        support = UsdGeom.Cylinder.Define(stage, support_path)
        support.CreateAxisAttr(UsdGeom.Tokens.z)
        support.CreateRadiusAttr(0.025)
        support.CreateHeightAttr(0.020)
        UsdGeom.Xformable(support).AddTranslateOp().Set(
            Gf.Vec3d(grasp_center[0], grasp_center[1], 0.230)
        )
        UsdPhysics.CollisionAPI.Apply(support.GetPrim())

        grip_material_path = "/World/GripPhysicsMaterial"
        grip_material = UsdShade.Material.Define(
            stage, grip_material_path
        )
        grip_api = UsdPhysics.MaterialAPI.Apply(grip_material.GetPrim())
        grip_api.CreateStaticFrictionAttr(1.4)
        grip_api.CreateDynamicFrictionAttr(1.4)
        grip_api.CreateRestitutionAttr(0.0)
        hand_anchor_count = 0
        nut_collision_count = 0
        nut_prefix = nut_path + "/"
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            is_hand_anchor = (
                prim_path.startswith(robot_root + "/")
                and prim.GetName().endswith("_convex")
                and any(
                    token in prim_path
                    for token in ("/f1Link", "/f2Link", "/f3Link")
                )
            )
            is_nut_collision = (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                and prim_path.startswith(nut_prefix)
            )
            if is_hand_anchor or is_nut_collision:
                physicsUtils.add_physics_material_to_prim(
                    stage, prim, Sdf.Path(grip_material_path)
                )
                if is_hand_anchor:
                    hand_anchor_count += 1
                else:
                    nut_collision_count += 1
        if hand_anchor_count != 8 or nut_collision_count != 16:
            raise RuntimeError(
                "grip material was not bound to expected colliders"
            )
        metrics["grip_material_hand_anchor_count"] = hand_anchor_count
        metrics["grip_material_nut_collision_count"] = (
            nut_collision_count
        )

        PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(nut_path)
        ).CreateThresholdAttr().Set(0.0)

        def count_hand_nut_contact_records():
            headers, _, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            count = 0
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                if any(
                    path.startswith(robot_root) for path in paths
                ) and any(path.startswith(nut_path) for path in paths):
                    count += int(header.num_contact_data)
            return count

        robot = world.scene.add(
            SingleArticulation(
                prim_path=articulation_path,
                name="connector_q7_handarm",
            )
        )
        body = world.scene.add(
            SingleRigidPrim(
                prim_path=body_path,
                name="connector_q7_body",
            )
        )
        nut = world.scene.add(
            SingleRigidPrim(
                prim_path=nut_path,
                name="connector_q7_nut",
            )
        )
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError("robot handles were not initialized")
        grasp_tcp_path = (
            articulation_path
            + "/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3"
            + "/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7"
            + "/iiwa_link_ee/handbase_link/grasp_tcp"
        )
        grasp_tcp_prim = stage.GetPrimAtPath(grasp_tcp_path)
        if not grasp_tcp_prim.IsValid():
            raise RuntimeError(f"missing grasp TCP prim: {grasp_tcp_path}")

        def grasp_tcp_position():
            position, _ = _matrix_pose(
                Gf, Usd, UsdGeom, grasp_tcp_prim
            )
            return np.asarray(position, dtype=np.float64)

        dof_names = tuple(robot.dof_names)
        name_to_index = {
            name: index for index, name in enumerate(dof_names)
        }
        controlled_names = ARM_JOINT_NAMES + ACTIVE_HAND_JOINT_NAMES
        controlled_indices = np.asarray(
            [name_to_index[name] for name in controlled_names],
            dtype=np.int32,
        )
        arm_indices = np.asarray(
            [name_to_index[name] for name in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        hand_indices = np.asarray(
            [name_to_index[name] for name in ACTIVE_HAND_JOINT_NAMES],
            dtype=np.int32,
        )
        q7_index = name_to_index["iiwa_joint_7"]
        sensor_names = ("f1j2", "f2j1", "f3j2")
        sensor_indices = np.asarray(
            [name_to_index[name] for name in sensor_names],
            dtype=np.int32,
        )
        dof_properties = robot.dof_properties

        grasp_arm = np.asarray(task["grasp_arm"], dtype=np.float64)
        insert_arm = np.asarray(task["insert_arm"], dtype=np.float64)
        initial_positions = named_joint_target(
            dof_names, grasp_arm, task["open_hand"]
        ).astype(np.float32)
        robot.set_joint_positions(initial_positions)
        robot.set_joint_velocities(
            np.zeros(robot.num_dof, dtype=np.float32)
        )

        controller = robot.get_articulation_controller()
        kps = np.zeros(robot.num_dof, dtype=np.float32)
        kds = np.zeros(robot.num_dof, dtype=np.float32)
        kps[arm_indices] = 8000.0
        kds[arm_indices] = 220.0
        kps[hand_indices] = 5.0
        kds[hand_indices] = 0.7
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)

        open_target = initial_positions[controlled_indices].copy()
        grasp_target = named_joint_target(
            dof_names, grasp_arm, task["grasp_hand"]
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
        tare_efforts = np.asarray(
            robot.get_measured_joint_efforts(
                joint_indices=sensor_indices
            ),
            dtype=np.float64,
        )

        max_abs_velocity = 0.0
        max_limit_violation = 0.0
        max_finger_torque_delta = 0.0
        finite_throughout = True

        def observe():
            nonlocal max_abs_velocity
            nonlocal max_limit_violation
            nonlocal max_finger_torque_delta
            nonlocal finite_throughout
            positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            efforts = np.asarray(
                robot.get_measured_joint_efforts(
                    joint_indices=sensor_indices
                ),
                dtype=np.float64,
            )
            body_position, _ = body.get_world_pose()
            nut_position, _ = nut.get_world_pose()
            finite_now = bool(
                np.all(np.isfinite(positions))
                and np.all(np.isfinite(velocities))
                and np.all(np.isfinite(efforts))
                and np.all(np.isfinite(body_position))
                and np.all(np.isfinite(nut_position))
            )
            finite_throughout = finite_throughout and finite_now
            max_abs_velocity = max(
                max_abs_velocity,
                float(np.max(np.abs(velocities))),
            )
            max_finger_torque_delta = max(
                max_finger_torque_delta,
                float(np.max(np.abs(efforts - tare_efforts))),
            )
            for index in range(robot.num_dof):
                if bool(dof_properties[index]["hasLimits"]):
                    lower = float(dof_properties[index]["lower"])
                    upper = float(dof_properties[index]["upper"])
                    max_limit_violation = max(
                        max_limit_violation,
                        lower - float(positions[index]),
                        float(positions[index]) - upper,
                    )
            return positions, velocities, efforts

        for step_index in range(arguments.closure_steps):
            blend = float(step_index + 1) / float(
                arguments.closure_steps
            )
            target = open_target + blend * (
                grasp_target - open_target
            )
            robot.apply_action(
                ArticulationAction(
                    joint_positions=target.astype(np.float32),
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            observe()

        grasp_efforts = np.asarray(
            robot.get_measured_joint_efforts(
                joint_indices=sensor_indices
            ),
            dtype=np.float64,
        )
        grasp_torque_deltas = grasp_efforts - tare_efforts
        loaded_before_insertion = int(
            np.count_nonzero(np.abs(grasp_torque_deltas) >= 0.02)
        )
        supported_body_position, _ = body.get_world_pose()
        _, supported_body_orientation = _matrix_pose(
            Gf,
            Usd,
            UsdGeom,
            stage.GetPrimAtPath(body_path),
        )

        # The pedestal is only a preload aid.  From here onward the plug is
        # moved solely through physical finger contact with the robot.
        stage.RemovePrim(support_path)
        simulation_app.update()
        world.get_physics_context().set_gravity(-9.81)
        for _ in range(arguments.settle_steps):
            robot.apply_action(
                ArticulationAction(
                    joint_positions=grasp_target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            observe()

        insertion_start_position, _ = body.get_world_pose()
        _, insertion_start_orientation = _matrix_pose(
            Gf,
            Usd,
            UsdGeom,
            stage.GetPrimAtPath(body_path),
        )
        supported_body_axis = supported_body_orientation.Transform(
            Gf.Vec3d(0.0, 0.0, 1.0)
        )
        insertion_start_axis = insertion_start_orientation.Transform(
            Gf.Vec3d(0.0, 0.0, 1.0)
        )
        supported_axis_error = math.acos(
            max(-1.0, min(1.0, float(supported_body_axis[2])))
        )
        insertion_start_axis_error = math.acos(
            max(-1.0, min(1.0, float(insertion_start_axis[2])))
        )
        insertion_target = named_joint_target(
            dof_names, insert_arm, task["grasp_hand"]
        )[controlled_indices].astype(np.float32)
        for step_index in range(arguments.insertion_steps):
            blend = float(step_index + 1) / float(
                arguments.insertion_steps
            )
            target = grasp_target + blend * (
                insertion_target - grasp_target
            )
            robot.apply_action(
                ArticulationAction(
                    joint_positions=target.astype(np.float32),
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            observe()

        for _ in range(arguments.settle_steps):
            robot.apply_action(
                ArticulationAction(
                    joint_positions=insertion_target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            observe()

        inserted_body_position, _ = body.get_world_pose()
        insertion_displacement = float(
            insertion_start_position[2] - inserted_body_position[2]
        )
        unsupported_settle_displacement = float(
            np.linalg.norm(
                insertion_start_position - supported_body_position
            )
        )
        pre_constraint_body_position, pre_constraint_body_orientation = (
            _matrix_pose(
                Gf,
                Usd,
                UsdGeom,
                stage.GetPrimAtPath(body_path),
            )
        )
        pre_constraint_nut_angle = _wrapped_relative_z_angle(
            Gf,
            Usd,
            UsdGeom,
            stage.GetPrimAtPath(body_path),
            stage.GetPrimAtPath(nut_path),
        )
        pre_constraint_body_array = np.asarray(
            pre_constraint_body_position, dtype=np.float64
        )
        (
            pre_constraint_body_default_position,
            pre_constraint_body_default_orientation,
        ) = body.get_world_pose()
        pre_constraint_robot_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float32
        )
        insertion_lateral_offset = float(
            np.linalg.norm(
                pre_constraint_body_array[:2] - grasp_center[:2]
            )
        )
        body_axis = pre_constraint_body_orientation.Transform(
            Gf.Vec3d(0.0, 0.0, 1.0)
        )
        insertion_axis_error = math.acos(
            max(-1.0, min(1.0, float(body_axis[2])))
        )
        total_approach_displacement = float(
            supported_body_position[2] - inserted_body_position[2]
        )
        (
            pre_constraint_nut_position,
            pre_constraint_nut_orientation,
        ) = nut.get_world_pose()

        hinge_prim = stage.GetPrimAtPath(hinge_path)
        hinge_drive_properties = sorted(
            str(prop.GetName())
            for prop in hinge_prim.GetProperties()
            if str(prop.GetName()).startswith("drive:")
        )
        hinge_drive_schemas = sorted(
            str(schema)
            for schema in hinge_prim.GetAppliedSchemas()
            if str(schema).startswith("PhysicsDriveAPI")
        )
        metrics["nut_hinge_drive_properties"] = hinge_drive_properties
        metrics["nut_hinge_drive_schemas"] = hinge_drive_schemas
        if hinge_drive_properties or hinge_drive_schemas:
            raise RuntimeError("coupling-nut hinge unexpectedly has a drive")

        # Freeze the current insertion pose into a world-to-body prismatic
        # joint, then couple that joint to the existing passive nut hinge.
        # The world joint locks lateral/tilt motion but leaves axial travel to
        # be generated bidirectionally by the helical rack relation.
        world.pause()
        runtime_root = f"{connector_root}/RuntimeThread"
        maximum_travel = 1.5 * task["lead"]
        thread_proxy_direction = 1
        ratio = rack_and_pinion_ratio(
            task["lead"], 1.0, thread_proxy_direction
        )
        runtime_thread_spec = RuntimeThreadSpec(
            stage=stage,
            body_path=body_path,
            nut_path=nut_path,
            hinge_path=hinge_path,
            runtime_root=runtime_root,
            maximum_travel_m=maximum_travel,
            ratio_degrees_per_meter=ratio,
        )
        prismatic, rack = create_runtime_thread(
            runtime_thread_spec,
            pre_constraint_body_position,
            pre_constraint_body_orientation,
        )
        metrics["thread_ratio_degrees_per_meter"] = ratio
        metrics["thread_proxy_direction"] = thread_proxy_direction

        world.play()
        simulation_app.update()
        residual_modes = {
            "residual-zero",
            "residual-action-effect",
            "residual-sac-smoke",
            "residual-train",
            "residual-evaluate",
            "residual-paired-evaluate",
        }
        constraint_activation_steps = (
            arguments.settle_steps
            if arguments.mode in residual_modes
            else 10
        )
        for _ in range(constraint_activation_steps):
            robot.apply_action(
                ArticulationAction(
                    joint_positions=insertion_target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            observe()

        constrained_body_position, constrained_body_orientation = (
            _matrix_pose(
                Gf,
                Usd,
                UsdGeom,
                stage.GetPrimAtPath(body_path),
            )
        )
        constrained_nut_angle = _wrapped_relative_z_angle(
            Gf,
            Usd,
            UsdGeom,
            stage.GetPrimAtPath(body_path),
            stage.GetPrimAtPath(nut_path),
        )
        constraint_position_jump = float(
            np.linalg.norm(
                np.asarray(constrained_body_position, dtype=np.float64)
                - pre_constraint_body_array
            )
        )
        constraint_orientation_jump = _quaternion_error_radians(
            pre_constraint_body_orientation,
            constrained_body_orientation,
        )
        constraint_nut_angle_jump = abs(
            math.atan2(
                math.sin(
                    constrained_nut_angle - pre_constraint_nut_angle
                ),
                math.cos(
                    constrained_nut_angle - pre_constraint_nut_angle
                ),
            )
        )

        if arguments.mode == "segmented":
            controlled_name_to_offset = {
                name: index for index, name in enumerate(controlled_names)
            }
            q7_command_offset = controlled_name_to_offset[
                "iiwa_joint_7"
            ]
            q7_config = load_q7_twist_config(config_path)
            target_connector_angle = math.radians(
                task["target_coupling_degrees"]
            )
            neutral_q7 = float(
                insertion_target[q7_command_offset]
            )
            schedule = q7_config.plan(
                target_connector_angle, initial_q7=neutral_q7
            )
            twist_segments = [
                segment
                for segment in schedule
                if segment.action == Q7Action.TWIST
            ]
            if len(twist_segments) != 3:
                raise RuntimeError(
                    "360 degree schedule must contain three twist strokes"
                )

            current_target = insertion_target.copy()
            unwrapped_nut_angle = constrained_nut_angle
            segmented_initial_body_position, _ = body.get_world_pose()
            segmented_initial_nut_position, _ = nut.get_world_pose()
            accumulated_q7_twist = 0.0
            maximum_open_hand_contact_records = 0
            maximum_brake_pose_drift = 0.0
            maximum_brake_angle_drift = 0.0
            maximum_unlock_position_jump = 0.0
            maximum_unlock_angle_jump = 0.0
            thread_proxy_rebases = 0
            segment_reports = []
            regrip_reports = []
            clearance_reports = []
            rewind_reports = []
            active_brake_paths = None
            brake_reference_position = None
            brake_reference_orientation = None
            brake_reference_angle = None

            def update_nut_angle():
                nonlocal unwrapped_nut_angle
                wrapped = _wrapped_relative_z_angle(
                    Gf,
                    Usd,
                    UsdGeom,
                    stage.GetPrimAtPath(body_path),
                    stage.GetPrimAtPath(nut_path),
                )
                unwrapped_nut_angle = _unwrap(
                    unwrapped_nut_angle, wrapped
                )

            def ramp_target(start, end, steps, require_open=False):
                nonlocal maximum_open_hand_contact_records
                for ramp_index in range(steps):
                    blend = float(ramp_index + 1) / float(steps)
                    target = start + blend * (end - start)
                    robot.apply_action(
                        ArticulationAction(
                            joint_positions=target.astype(np.float32),
                            joint_indices=controlled_indices,
                        )
                    )
                    world.step(render=arguments.gui)
                    observe()
                    update_nut_angle()
                    if require_open:
                        maximum_open_hand_contact_records = max(
                            maximum_open_hand_contact_records,
                            count_hand_nut_contact_records(),
                        )

            def hold_target(target, steps, require_open=False):
                ramp_target(target, target, steps, require_open)

            def create_thread_brake(index):
                nonlocal active_brake_paths
                nonlocal brake_reference_position
                nonlocal brake_reference_orientation
                nonlocal brake_reference_angle
                world.pause()
                UsdPhysics.Joint(hinge_prim).CreateJointEnabledAttr(
                    False
                )
                rack.CreateJointEnabledAttr(False)
                prismatic.CreateJointEnabledAttr(False)
                body_position_now, body_orientation_now = _matrix_pose(
                    Gf,
                    Usd,
                    UsdGeom,
                    stage.GetPrimAtPath(body_path),
                )
                nut_position_now, nut_orientation_now = _matrix_pose(
                    Gf,
                    Usd,
                    UsdGeom,
                    stage.GetPrimAtPath(nut_path),
                )
                brake_reference_position = np.asarray(
                    body_position_now, dtype=np.float64
                )
                brake_reference_orientation = body_orientation_now
                brake_reference_angle = unwrapped_nut_angle

                world_body_path = (
                    f"{runtime_root}/Brake{index}_WorldBody"
                )
                world_body_brake = UsdPhysics.FixedJoint.Define(
                    stage, world_body_path
                )
                world_body_brake.CreateBody1Rel().SetTargets(
                    [Sdf.Path(body_path)]
                )
                world_body_brake.CreateLocalPos0Attr(
                    Gf.Vec3f(
                        float(body_position_now[0]),
                        float(body_position_now[1]),
                        float(body_position_now[2]),
                    )
                )
                body_imaginary = body_orientation_now.GetImaginary()
                world_body_brake.CreateLocalRot0Attr(
                    Gf.Quatf(
                        float(body_orientation_now.GetReal()),
                        Gf.Vec3f(
                            float(body_imaginary[0]),
                            float(body_imaginary[1]),
                            float(body_imaginary[2]),
                        ),
                    )
                )
                world_body_brake.CreateLocalPos1Attr(Gf.Vec3f(0.0))
                world_body_brake.CreateLocalRot1Attr(Gf.Quatf(1.0))
                world_body_brake.CreateCollisionEnabledAttr(False)

                body_nut_path = (
                    f"{runtime_root}/Brake{index}_BodyNut"
                )
                body_nut_brake = UsdPhysics.FixedJoint.Define(
                    stage, body_nut_path
                )
                body_nut_brake.CreateBody0Rel().SetTargets(
                    [Sdf.Path(body_path)]
                )
                body_nut_brake.CreateBody1Rel().SetTargets(
                    [Sdf.Path(nut_path)]
                )
                world_delta = Gf.Vec3d(
                    float(body_position_now[0] - nut_position_now[0]),
                    float(body_position_now[1] - nut_position_now[1]),
                    float(body_position_now[2] - nut_position_now[2]),
                )
                local_position1 = (
                    nut_orientation_now.GetInverse().Transform(world_delta)
                )
                local_rotation1 = (
                    nut_orientation_now.GetInverse()
                    * body_orientation_now
                )
                local_imaginary = local_rotation1.GetImaginary()
                body_nut_brake.CreateLocalPos0Attr(Gf.Vec3f(0.0))
                body_nut_brake.CreateLocalRot0Attr(Gf.Quatf(1.0))
                body_nut_brake.CreateLocalPos1Attr(
                    Gf.Vec3f(
                        float(local_position1[0]),
                        float(local_position1[1]),
                        float(local_position1[2]),
                    )
                )
                body_nut_brake.CreateLocalRot1Attr(
                    Gf.Quatf(
                        float(local_rotation1.GetReal()),
                        Gf.Vec3f(
                            float(local_imaginary[0]),
                            float(local_imaginary[1]),
                            float(local_imaginary[2]),
                        ),
                    )
                )
                body_nut_brake.CreateCollisionEnabledAttr(False)
                active_brake_paths = (world_body_path, body_nut_path)
                world.play()
                simulation_app.update()
                hold_target(current_target, 10)

            def update_brake_drift():
                nonlocal maximum_brake_pose_drift
                nonlocal maximum_brake_angle_drift
                body_position_now, body_orientation_now = _matrix_pose(
                    Gf,
                    Usd,
                    UsdGeom,
                    stage.GetPrimAtPath(body_path),
                )
                position_drift = float(
                    np.linalg.norm(
                        np.asarray(body_position_now, dtype=np.float64)
                        - brake_reference_position
                    )
                )
                orientation_drift = _quaternion_error_radians(
                    brake_reference_orientation, body_orientation_now
                )
                angle_drift = abs(
                    unwrapped_nut_angle - brake_reference_angle
                )
                maximum_brake_pose_drift = max(
                    maximum_brake_pose_drift,
                    position_drift,
                    orientation_drift * 0.001,
                )
                maximum_brake_angle_drift = max(
                    maximum_brake_angle_drift, angle_drift
                )

            def remove_thread_brake():
                nonlocal active_brake_paths
                nonlocal maximum_unlock_position_jump
                nonlocal maximum_unlock_angle_jump
                nonlocal thread_proxy_rebases
                before_position, before_orientation = _matrix_pose(
                    Gf,
                    Usd,
                    UsdGeom,
                    stage.GetPrimAtPath(body_path),
                )
                before_angle = unwrapped_nut_angle
                world.pause()
                # Rebase both passive joint coordinates at the held physical
                # pose before re-enabling the rack.  Without this, PhysX
                # restores the first segment's zero and erases accumulated
                # axial travel at every regrasp.
                nut_position_now, nut_orientation_now = _matrix_pose(
                    Gf,
                    Usd,
                    UsdGeom,
                    stage.GetPrimAtPath(nut_path),
                )
                hinge_joint = UsdPhysics.RevoluteJoint(hinge_prim)
                world_delta = Gf.Vec3d(
                    float(before_position[0] - nut_position_now[0]),
                    float(before_position[1] - nut_position_now[1]),
                    float(before_position[2] - nut_position_now[2]),
                )
                hinge_local_position1 = (
                    nut_orientation_now.GetInverse().Transform(world_delta)
                )
                hinge_local_rotation1 = (
                    nut_orientation_now.GetInverse()
                    * before_orientation
                )
                hinge_local_imaginary = (
                    hinge_local_rotation1.GetImaginary()
                )
                hinge_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0))
                hinge_joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0))
                hinge_joint.GetLocalPos1Attr().Set(
                    Gf.Vec3f(
                        float(hinge_local_position1[0]),
                        float(hinge_local_position1[1]),
                        float(hinge_local_position1[2]),
                    )
                )
                hinge_joint.GetLocalRot1Attr().Set(
                    Gf.Quatf(
                        float(hinge_local_rotation1.GetReal()),
                        Gf.Vec3f(
                            float(hinge_local_imaginary[0]),
                            float(hinge_local_imaginary[1]),
                            float(hinge_local_imaginary[2]),
                        ),
                    )
                )
                prismatic.GetLocalPos0Attr().Set(
                    Gf.Vec3f(
                        float(before_position[0]),
                        float(before_position[1]),
                        float(before_position[2]),
                    )
                )
                body_imaginary = before_orientation.GetImaginary()
                prismatic.GetLocalRot0Attr().Set(
                    Gf.Quatf(
                        float(before_orientation.GetReal()),
                        Gf.Vec3f(
                            float(body_imaginary[0]),
                            float(body_imaginary[1]),
                            float(body_imaginary[2]),
                        ),
                    )
                )
                prismatic.GetLocalPos1Attr().Set(Gf.Vec3f(0.0))
                prismatic.GetLocalRot1Attr().Set(Gf.Quatf(1.0))
                for brake_path in active_brake_paths:
                    stage.RemovePrim(brake_path)
                UsdPhysics.Joint(hinge_prim).CreateJointEnabledAttr(True)
                rack.CreateJointEnabledAttr(True)
                prismatic.CreateJointEnabledAttr(True)
                thread_proxy_rebases += 1
                active_brake_paths = None
                world.play()
                simulation_app.update()
                hold_target(current_target, 10)
                after_position, after_orientation = _matrix_pose(
                    Gf,
                    Usd,
                    UsdGeom,
                    stage.GetPrimAtPath(body_path),
                )
                maximum_unlock_position_jump = max(
                    maximum_unlock_position_jump,
                    float(
                        np.linalg.norm(
                            np.asarray(after_position, dtype=np.float64)
                            - np.asarray(
                                before_position, dtype=np.float64
                            )
                        )
                    ),
                )
                maximum_unlock_angle_jump = max(
                    maximum_unlock_angle_jump,
                    _quaternion_error_radians(
                        before_orientation, after_orientation
                    ),
                    abs(unwrapped_nut_angle - before_angle),
                )

            brake_index = 0
            for action_segment in schedule:
                if action_segment.action == Q7Action.GRIP:
                    continue
                if action_segment.action == Q7Action.TWIST:
                    segment_start_positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    segment_start_q7 = float(
                        segment_start_positions[q7_index]
                    )
                    segment_start_angle = unwrapped_nut_angle
                    segment_start_body_position, _ = (
                        body.get_world_pose()
                    )
                    segment_target = current_target.copy()
                    segment_target[q7_command_offset] = (
                        action_segment.q7_end
                    )
                    segment_steps = int(
                        math.ceil(
                            abs(
                                action_segment.q7_end
                                - action_segment.q7_start
                            )
                            / q7_config.maximum_speed_rad_per_second
                            * 240.0
                        )
                    )
                    ramp_target(
                        current_target, segment_target, segment_steps
                    )
                    current_target = segment_target
                    hold_target(current_target, 30)
                    segment_end_positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    segment_end_body_position, _ = body.get_world_pose()
                    measured_q7_delta = float(
                        segment_end_positions[q7_index]
                        - segment_start_q7
                    )
                    measured_nut_delta = float(
                        unwrapped_nut_angle - segment_start_angle
                    )
                    measured_axial_delta = float(
                        segment_end_body_position[2]
                        - segment_start_body_position[2]
                    )
                    expected_axial_delta = float(
                        -task["lead"]
                        * measured_nut_delta
                        / (2.0 * math.pi)
                    )
                    segment_helical_error = (
                        measured_axial_delta - expected_axial_delta
                    )
                    segment_ok = bool(
                        abs(
                            measured_q7_delta
                            - (
                                action_segment.q7_end
                                - action_segment.q7_start
                            )
                        )
                        <= math.radians(2.0)
                        and abs(
                            measured_nut_delta
                            + measured_q7_delta
                        )
                        <= math.radians(3.0)
                        and abs(segment_helical_error) <= 0.0001
                    )
                    accumulated_q7_twist += measured_q7_delta
                    segment_reports.append(
                        {
                            "axial_delta_m": measured_axial_delta,
                            "helical_error_m": segment_helical_error,
                            "nut_delta_degrees": math.degrees(
                                measured_nut_delta
                            ),
                            "passed": segment_ok,
                            "q7_delta_degrees": math.degrees(
                                measured_q7_delta
                            ),
                            "steps": segment_steps,
                        }
                    )
                elif action_segment.action == Q7Action.RELEASE:
                    brake_index += 1
                    create_thread_brake(brake_index)
                    open_target = named_joint_target(
                        dof_names, insert_arm, task["open_hand"]
                    )[controlled_indices].astype(np.float32)
                    open_target[q7_command_offset] = (
                        current_target[q7_command_offset]
                    )
                    ramp_target(
                        current_target,
                        open_target,
                        arguments.closure_steps,
                    )
                    current_target = open_target
                    hold_target(current_target, arguments.settle_steps)
                    maximum_open_hand_contact_records = max(
                        maximum_open_hand_contact_records,
                        count_hand_nut_contact_records(),
                    )
                    clearance_start_tcp = grasp_tcp_position()
                    lift_target = current_target.copy()
                    lift_scale = (
                        task["regrasp_clearance"]
                        / total_approach_displacement
                    )
                    lift_target[:6] += lift_scale * (
                        grasp_target[:6] - insertion_target[:6]
                    )
                    ramp_target(
                        current_target,
                        lift_target,
                        120,
                        require_open=True,
                    )
                    current_target = lift_target
                    hold_target(current_target, 30, require_open=True)
                    clearance_end_tcp = grasp_tcp_position()
                    measured_clearance = float(
                        clearance_end_tcp[2] - clearance_start_tcp[2]
                    )
                    clearance_reports.append(
                        {
                            "commanded_m": task["regrasp_clearance"],
                            "measured_m": measured_clearance,
                            "passed": bool(
                                0.0005 <= measured_clearance <= 0.0015
                            ),
                        }
                    )
                    update_brake_drift()
                elif action_segment.action == Q7Action.REWIND:
                    rewind_start_positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    rewind_target = current_target.copy()
                    rewind_target[q7_command_offset] = (
                        action_segment.q7_end
                    )
                    rewind_steps = int(
                        math.ceil(
                            abs(
                                action_segment.q7_end
                                - action_segment.q7_start
                            )
                            / q7_config.maximum_speed_rad_per_second
                            * 240.0
                        )
                    )
                    ramp_target(
                        current_target,
                        rewind_target,
                        rewind_steps,
                        require_open=True,
                    )
                    current_target = rewind_target
                    hold_target(current_target, 30, require_open=True)
                    rewind_end_positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    rewind_error = float(
                        rewind_end_positions[q7_index]
                        - action_segment.q7_end
                    )
                    rewind_reports.append(
                        {
                            "actual_delta_degrees": math.degrees(
                                float(
                                    rewind_end_positions[q7_index]
                                    - rewind_start_positions[q7_index]
                                )
                            ),
                            "end_error_degrees": math.degrees(
                                rewind_error
                            ),
                            "passed": bool(
                                abs(rewind_error) <= math.radians(2.0)
                            ),
                        }
                    )
                    update_brake_drift()
                elif action_segment.action == Q7Action.REGRIP:
                    reapproach_target = current_target.copy()
                    reapproach_target[:6] = insertion_target[:6]
                    ramp_target(
                        current_target,
                        reapproach_target,
                        120,
                        require_open=True,
                    )
                    current_target = reapproach_target
                    close_target = insertion_target.copy()
                    ramp_target(
                        current_target,
                        close_target,
                        arguments.closure_steps,
                    )
                    current_target = close_target
                    hold_target(current_target, arguments.settle_steps)
                    regrip_efforts = np.asarray(
                        robot.get_measured_joint_efforts(
                            joint_indices=sensor_indices
                        ),
                        dtype=np.float64,
                    )
                    regrip_loaded = int(
                        np.count_nonzero(
                            np.abs(regrip_efforts - tare_efforts) >= 0.02
                        )
                    )
                    regrip_contacts = count_hand_nut_contact_records()
                    update_brake_drift()
                    remove_thread_brake()
                    regrip_reports.append(
                        {
                            "contact_records": regrip_contacts,
                            "loaded_channels": regrip_loaded,
                            "passed": bool(
                                regrip_loaded
                                >= task["minimum_loaded_channels"]
                                and regrip_contacts > 0
                            ),
                        }
                    )

            hold_start_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            hold_start_body_position, _ = body.get_world_pose()
            hold_start_angle = unwrapped_nut_angle
            hold_steps = int(round(task["hold_duration"] * 240.0))
            hold_target(current_target, hold_steps)
            hold_end_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            hold_end_body_position, _ = body.get_world_pose()
            hold_nut_drift = float(
                unwrapped_nut_angle - hold_start_angle
            )
            hold_axial_drift = float(
                hold_end_body_position[2]
                - hold_start_body_position[2]
            )
            hold_q7_drift = float(
                hold_end_positions[q7_index]
                - hold_start_positions[q7_index]
            )
            hold_contact_records = count_hand_nut_contact_records()
            hold_ok = bool(
                abs(hold_nut_drift) <= math.radians(0.5)
                and abs(hold_axial_drift) <= 0.0001
                and abs(hold_q7_drift) <= math.radians(0.5)
                and hold_contact_records > 0
            )

            final_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            final_velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            final_efforts = np.asarray(
                robot.get_measured_joint_efforts(
                    joint_indices=sensor_indices
                ),
                dtype=np.float64,
            )
            final_body_position, _ = body.get_world_pose()
            final_nut_position, _ = nut.get_world_pose()
            final_loaded_channels = int(
                np.count_nonzero(
                    np.abs(final_efforts - tare_efforts) >= 0.02
                )
            )
            total_nut_angle = float(
                unwrapped_nut_angle - constrained_nut_angle
            )
            total_axial_travel = float(
                final_body_position[2]
                - float(constrained_body_position[2])
            )
            expected_total_axial = float(
                -task["lead"]
                * total_nut_angle
                / (2.0 * math.pi)
            )
            total_helical_error = (
                total_axial_travel - expected_total_axial
            )
            final_joint_separation_change = abs(
                float(
                    np.linalg.norm(
                        final_nut_position - final_body_position
                    )
                    - np.linalg.norm(
                        segmented_initial_nut_position
                        - segmented_initial_body_position
                    )
                )
            )
            finite = bool(
                finite_throughout
                and np.all(np.isfinite(final_positions))
                and np.all(np.isfinite(final_velocities))
                and np.all(np.isfinite(final_efforts))
                and np.all(np.isfinite(final_body_position))
                and np.all(np.isfinite(final_nut_position))
            )
            stable = bool(
                finite
                and max_abs_velocity <= 20.0
                and max_limit_violation <= 0.02
            )
            constraint_activation_ok = bool(
                constraint_position_jump <= 0.001
                and constraint_orientation_jump <= math.radians(0.5)
                and constraint_nut_angle_jump <= math.radians(0.5)
            )
            insertion_ok = bool(
                total_approach_displacement >= 0.025
                and insertion_lateral_offset <= 0.0015
                and insertion_axis_error <= math.radians(2.0)
            )
            target_angle_ok = bool(
                abs(
                    math.degrees(total_nut_angle)
                    - task["target_coupling_degrees"]
                )
                <= task["coupling_tolerance_degrees"]
            )
            passed = bool(
                stable
                and constraint_activation_ok
                and insertion_ok
                and all(
                    report["passed"] for report in segment_reports
                )
                and all(report["passed"] for report in regrip_reports)
                and all(
                    report["passed"] for report in clearance_reports
                )
                and all(report["passed"] for report in rewind_reports)
                and maximum_open_hand_contact_records == 0
                and maximum_brake_pose_drift <= 0.0001
                and maximum_brake_angle_drift <= math.radians(0.5)
                and maximum_unlock_position_jump <= 0.0001
                and maximum_unlock_angle_jump <= math.radians(0.5)
                and target_angle_ok
                and hold_ok
                and abs(total_helical_error)
                <= task["helical_error_tolerance"]
                and final_loaded_channels
                >= task["minimum_loaded_channels"]
                and max_finger_torque_delta
                <= task["maximum_finger_torque"]
                and final_joint_separation_change <= 0.001
                and not hinge_drive_properties
                and not hinge_drive_schemas
            )
            metrics.update(
                {
                    "accumulated_q7_twist_degrees": math.degrees(
                        accumulated_q7_twist
                    ),
                    "constraint_activation_ok": (
                        constraint_activation_ok
                    ),
                    "clearance_reports": clearance_reports,
                    "final_loaded_channels": final_loaded_channels,
                    "finite": finite,
                    "insertion_axis_error_degrees": math.degrees(
                        insertion_axis_error
                    ),
                    "insertion_lateral_offset_m": (
                        insertion_lateral_offset
                    ),
                    "insertion_ok": insertion_ok,
                    "hold_axial_drift_m": hold_axial_drift,
                    "hold_contact_records": hold_contact_records,
                    "hold_duration_s": task["hold_duration"],
                    "hold_nut_drift_degrees": math.degrees(
                        hold_nut_drift
                    ),
                    "hold_ok": hold_ok,
                    "hold_q7_drift_degrees": math.degrees(
                        hold_q7_drift
                    ),
                    "maximum_absolute_finger_torque_delta_nm": (
                        max_finger_torque_delta
                    ),
                    "maximum_brake_angle_drift_degrees": math.degrees(
                        maximum_brake_angle_drift
                    ),
                    "maximum_brake_pose_drift_m": (
                        maximum_brake_pose_drift
                    ),
                    "maximum_joint_limit_violation_rad": (
                        max_limit_violation
                    ),
                    "maximum_joint_speed_rad_s": max_abs_velocity,
                    "maximum_open_hand_contact_records": (
                        maximum_open_hand_contact_records
                    ),
                    "maximum_unlock_angle_jump_degrees": math.degrees(
                        maximum_unlock_angle_jump
                    ),
                    "maximum_unlock_position_jump_m": (
                        maximum_unlock_position_jump
                    ),
                    "passed": passed,
                    "planned_actions": [
                        segment.action.value for segment in schedule
                    ],
                    "regrasp_clearance_m": task["regrasp_clearance"],
                    "regrasp_reports": regrip_reports,
                    "rewind_reports": rewind_reports,
                    "segment_reports": segment_reports,
                    "stable": stable,
                    "target_angle_ok": target_angle_ok,
                    "thread_proxy_rebases": thread_proxy_rebases,
                    "total_axial_travel_m": total_axial_travel,
                    "total_helical_error_m": total_helical_error,
                    "total_nut_angle_degrees": math.degrees(
                        total_nut_angle
                    ),
                }
            )
            print(json.dumps(metrics, sort_keys=True), flush=True)
            print(
                banner + " " + ("PASSED" if passed else "FAILED"),
                flush=True,
            )
            return

        if arguments.mode in (
            "residual-zero",
            "residual-action-effect",
            "residual-sac-smoke",
            "residual-train",
            "residual-evaluate",
            "residual-paired-evaluate",
        ):
            from kcg_connector.residual_rl import (
                load_connector_residual_config,
                loaded_torque_channels,
            )
            from kcg_connector.residual_curriculum import (
                load_connector_residual_curriculum,
                resolve_stage,
                resolved_stage_document,
            )
            from kcg_connector.residual_randomization import (
                load_connector_residual_randomization_config,
                reproducible_stream_reset_seed,
            )

            if (
                arguments.mode == "residual-zero"
                and arguments.episodes != 2
            ):
                raise ValueError(
                    "residual-zero gate requires exactly two episodes"
                )
            if (
                arguments.mode == "residual-sac-smoke"
                and arguments.training_timesteps < 2
            ):
                raise ValueError(
                    "SAC smoke requires at least two training timesteps"
                )
            if arguments.action_effect_steps < 3:
                raise ValueError(
                    "action-effect gate requires at least three policy steps"
                )
            base_residual_config = load_connector_residual_config(
                config_path
            )
            curriculum = load_connector_residual_curriculum(
                curriculum_config_path
            )
            resolved_stage = resolve_stage(
                base_residual_config,
                arguments.residual_stage,
                float(pre_constraint_robot_positions[q7_index]),
                curriculum=curriculum,
            )
            residual_config = resolved_stage.residual_config
            maximum_episode_steps = (
                resolved_stage.stage.maximum_episode_steps
            )
            if (
                arguments.maximum_episode_steps is not None
                and arguments.maximum_episode_steps
                != maximum_episode_steps
            ):
                raise ValueError(
                    "--maximum-episode-steps must equal the selected "
                    f"{arguments.residual_stage} curriculum limit "
                    f"({maximum_episode_steps})"
                )
            resolved_curriculum_stage = resolved_stage_document(
                resolved_stage
            )
            metrics.update(
                {
                    "maximum_episode_steps": maximum_episode_steps,
                    "minimum_axial_progress_fraction": (
                        residual_config.minimum_axial_progress_fraction
                    ),
                    "residual_curriculum_stage": arguments.residual_stage,
                    "resolved_curriculum_stage": (
                        resolved_curriculum_stage
                    ),
                }
            )
            randomization_config = (
                None
                if randomization_config_path is None
                else load_connector_residual_randomization_config(
                    randomization_config_path
                )
            )
            if (
                randomization_config is not None
                and arguments.mode
                in ("residual-zero", "residual-action-effect")
                and arguments.reset_seed is None
            ):
                raise ValueError(
                    "randomized residual zero/action-effect gates require "
                    "--reset-seed"
                )
            controlled_name_to_offset = {
                name: index
                for index, name in enumerate(controlled_names)
            }
            q7_command_offset = controlled_name_to_offset[
                "iiwa_joint_7"
            ]
            clamp_command_offsets = np.asarray(
                [
                    controlled_name_to_offset[name]
                    for name in residual_config.clamp_joint_names
                ],
                dtype=np.int32,
            )

            checkpoint_positions = (
                pre_constraint_robot_positions.copy()
            )
            checkpoint_body_position = np.asarray(
                pre_constraint_body_default_position,
                dtype=np.float32,
            )
            checkpoint_body_orientation = np.asarray(
                pre_constraint_body_default_orientation,
                dtype=np.float32,
            )
            checkpoint_nut_position = np.asarray(
                pre_constraint_nut_position, dtype=np.float32
            )
            checkpoint_nut_orientation = np.asarray(
                pre_constraint_nut_orientation, dtype=np.float32
            )
            zero_linear_velocity = np.zeros(3, dtype=np.float32)
            zero_angular_velocity = np.zeros(3, dtype=np.float32)
            robot.set_joints_default_state(
                positions=checkpoint_positions,
                velocities=np.zeros(robot.num_dof, dtype=np.float32),
            )
            body.set_default_state(
                position=np.asarray(
                    checkpoint_body_position, dtype=np.float32
                ),
                orientation=np.asarray(
                    checkpoint_body_orientation, dtype=np.float32
                ),
                linear_velocity=zero_linear_velocity,
                angular_velocity=zero_angular_velocity,
            )
            nut.set_default_state(
                position=np.asarray(
                    checkpoint_nut_position, dtype=np.float32
                ),
                orientation=np.asarray(
                    checkpoint_nut_orientation, dtype=np.float32
                ),
                linear_velocity=zero_linear_velocity,
                angular_velocity=zero_angular_velocity,
            )
            runtime_thread_prim_count = sum(
                1
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(runtime_root + "/")
            )

            prepared_scene = PreparedConnectorScene(
                simulation_app=simulation_app,
                world=world,
                stage=stage,
                robot=robot,
                body=body,
                nut=nut,
                grasp_tcp_prim=grasp_tcp_prim,
                thread_spec=runtime_thread_spec,
                controlled_indices=controlled_indices,
                sensor_indices=sensor_indices,
                q7_index=q7_index,
                q7_command_offset=q7_command_offset,
                clamp_command_offsets=clamp_command_offsets,
                insertion_target=insertion_target,
                kps=kps,
                kds=kds,
                tare_efforts=tare_efforts,
                dof_properties=dof_properties,
                checkpoint_positions=checkpoint_positions,
                checkpoint_body_position=checkpoint_body_position,
                checkpoint_body_orientation=(
                    checkpoint_body_orientation
                ),
                checkpoint_nut_position=checkpoint_nut_position,
                checkpoint_nut_orientation=checkpoint_nut_orientation,
                residual_config=residual_config,
                resolved_curriculum_stage=resolved_curriculum_stage,
                settle_steps=arguments.settle_steps,
                maximum_episode_steps=maximum_episode_steps,
                render=arguments.gui,
                randomization_config=randomization_config,
            )
            adapter = ConnectorResidualIsaacBackend(prepared_scene)
            if arguments.mode in (
                "residual-train",
                "residual-evaluate",
                "residual-paired-evaluate",
            ):
                from kcg_rl.connector_residual_env import (
                    ConnectorResidualEnv,
                )
                from kcg_rl.connector_residual_sac import (
                    load_connector_residual_sac_config,
                    run_formal_evaluation,
                    run_formal_paired_evaluation,
                    run_formal_training,
                )

                formal_run_config_path = Path(
                    arguments.formal_run_config
                ).expanduser().resolve()
                formal_config = load_connector_residual_sac_config(
                    formal_run_config_path
                )
                if (
                    formal_config.interface_version
                    != residual_config.interface_version
                ):
                    raise RuntimeError(
                        "formal SAC and physical task interface versions "
                        "differ"
                    )
                formal_runner_path = (
                    repository
                    / "src/kcg_rl/kcg_rl/connector_residual_sac.py"
                )
                gym_environment_path = (
                    repository
                    / "src/kcg_rl/kcg_rl/connector_residual_env.py"
                )
                provenance_paths = {
                    "backend": (
                        repository
                        / "src/kcg_connector/kcg_connector/"
                        "isaac_residual_backend.py"
                    ),
                    "config": config_path,
                    "connector_asset": connector_asset,
                    "curriculum_config": curriculum_config_path,
                    "curriculum_contract": (
                        repository
                        / "src/kcg_connector/kcg_connector/"
                        "residual_curriculum.py"
                    ),
                    "gym_environment": gym_environment_path,
                    "isaac_rl_requirements": (
                        repository
                        / "src/kcg_rl/requirements-isaac-rl.txt"
                    ),
                    "isaac_python_wrapper": (
                        repository
                        / "src/kcg_connector/isaac/"
                        "run_isaac_python.sh"
                    ),
                    "isaacsim_requirements": (
                        repository
                        / "src/kcg_connector/"
                        "requirements-isaacsim.txt"
                    ),
                    "residual_contract": (
                        repository
                        / "src/kcg_connector/kcg_connector/residual_rl.py"
                    ),
                    "robot_asset": robot_asset,
                    "runner": formal_runner_path,
                    "script": Path(__file__).resolve(),
                    "training_config": formal_run_config_path,
                }
                if randomization_config_path is not None:
                    provenance_paths["randomization_config"] = (
                        randomization_config_path
                    )
                    provenance_paths["randomization_contract"] = (
                        repository
                        / "src/kcg_connector/kcg_connector/"
                        "residual_randomization.py"
                    )
                gym_environment = ConnectorResidualEnv(adapter)
                if arguments.mode == "residual-train":
                    formal_metrics = run_formal_training(
                        gym_environment,
                        adapter,
                        formal_config,
                        requested_timesteps=(
                            arguments.formal_timesteps
                        ),
                        allow_long_training=(
                            arguments.allow_long_training
                        ),
                        output_root=arguments.formal_output_root,
                        provenance_paths=provenance_paths,
                    )
                    passed = bool(formal_metrics["passed"])
                else:
                    formal_run_directory = Path(
                        arguments.formal_run_dir
                    ).expanduser().resolve()
                    if not formal_run_directory.is_dir():
                        raise FileNotFoundError(
                            formal_run_directory
                        )
                    formal_model_path = (
                        formal_run_directory / "final_model.zip"
                    )
                    if arguments.mode == "residual-paired-evaluate":
                        formal_metrics = run_formal_paired_evaluation(
                            gym_environment,
                            adapter,
                            formal_config,
                            model_path=formal_model_path,
                            episodes=arguments.evaluation_episodes,
                            output_root=arguments.formal_output_root,
                            provenance_paths=provenance_paths,
                        )
                        passed = bool(
                            formal_metrics["benchmark_integrity_passed"]
                        )
                    else:
                        formal_metrics = run_formal_evaluation(
                            gym_environment,
                            adapter,
                            formal_config,
                            model_path=formal_model_path,
                            episodes=arguments.evaluation_episodes,
                            output_root=arguments.formal_output_root,
                            provenance_paths=provenance_paths,
                        )
                        passed = bool(
                            formal_metrics["acceptance_passed"]
                        )
                metrics.update(formal_metrics)
                metrics.update(
                    {
                        "episode_object_pose_writes": 0,
                        "passed": passed,
                        "reset_object_pose_writes": 0,
                        "residual_action_size": 4,
                        "residual_observation_size": 24,
                    }
                )
                print(json.dumps(metrics, sort_keys=True), flush=True)
                print(
                    banner
                    + " "
                    + ("PASSED" if passed else "FAILED"),
                    flush=True,
                )
                if arguments.mode == "residual-paired-evaluate":
                    print(
                        "POLICY IMPROVEMENT CLAIM "
                        + (
                            "TRUE"
                            if formal_metrics[
                                "policy_improvement_claim"
                            ]
                            else "FALSE"
                        ),
                        flush=True,
                    )
                return

            if arguments.mode == "residual-action-effect":
                action_cases = (
                    ("baseline", (0.0, 0.0, 0.0, 0.0)),
                    ("q7_slow", (-1.0, 0.0, 0.0, 0.0)),
                    ("q7_fast", (1.0, 0.0, 0.0, 0.0)),
                    ("f1j2_tighter", (0.0, 1.0, 0.0, 0.0)),
                    ("f2j1_tighter", (0.0, 0.0, 1.0, 0.0)),
                    ("f3j2_tighter", (0.0, 0.0, 0.0, 1.0)),
                )
                case_reports = {}
                case_histories = {}
                initial_signatures = []
                initial_clamp_positions = []
                initial_finger_torques = []
                for case_name, action_values in action_cases:
                    action = np.asarray(
                        action_values, dtype=np.float32
                    )
                    _, reset_info = adapter.reset(
                        seed=arguments.reset_seed
                    )
                    initial_signatures.append(
                        adapter.initial_signature
                    )
                    initial_clamp_positions.append(
                        np.asarray(
                            adapter.previous_state.clamp_positions_rad,
                            dtype=np.float64,
                        )
                    )
                    initial_finger_torques.append(
                        np.asarray(
                            adapter.previous_state.finger_torques_nm,
                            dtype=np.float64,
                        )
                    )
                    states = []
                    nut_progress_samples = [0.0]
                    terminated = False
                    truncated = False
                    final_info = {}
                    for _ in range(arguments.action_effect_steps):
                        (
                            _,
                            _,
                            terminated,
                            truncated,
                            final_info,
                        ) = adapter.step(action)
                        states.append(adapter.previous_state)
                        nut_progress_samples.append(
                            adapter.previous_state.nut_angle_rad
                        )
                        if terminated or truncated:
                            break
                    tail_states = states[-min(5, len(states)):]
                    torque_history = np.asarray(
                        [
                            state.finger_torques_nm
                            for state in states
                        ],
                        dtype=np.float64,
                    )
                    clamp_history = np.asarray(
                        [
                            state.clamp_positions_rad
                            for state in states
                        ],
                        dtype=np.float64,
                    )
                    tail_torque_median = np.median(
                        np.asarray(
                            [
                                state.finger_torques_nm
                                for state in tail_states
                            ],
                            dtype=np.float64,
                        ),
                        axis=0,
                    )
                    tail_clamp_median = np.median(
                        np.asarray(
                            [
                                state.clamp_positions_rad
                                for state in tail_states
                            ],
                            dtype=np.float64,
                        ),
                        axis=0,
                    )
                    tail_nut_velocity = float(
                        np.median(
                            [
                                state.nut_angular_velocity_rad_s
                                for state in tail_states
                            ]
                        )
                    )
                    nut_step_increments = np.diff(
                        np.asarray(
                            nut_progress_samples,
                            dtype=np.float64,
                        )
                    )
                    minimum_loaded_channels = min(
                        loaded_torque_channels(
                            state, residual_config
                        )
                        for state in states
                    )
                    episode_safety = adapter.episode_safety
                    case_ok = bool(
                        len(states) == arguments.action_effect_steps
                        and not terminated
                        and not truncated
                        and episode_safety.finite_throughout
                        and np.all(np.isfinite(torque_history))
                        and np.all(np.isfinite(clamp_history))
                        and minimum_loaded_channels
                        >= residual_config.
                        minimum_loaded_torque_channels
                        and episode_safety.max_finger_torque_delta
                        <= residual_config.
                        maximum_absolute_finger_torque_nm
                        and episode_safety.max_limit_violation <= 0.02
                    )
                    case_histories[case_name] = {
                        "clamp_median": tail_clamp_median,
                        "nut_increments": nut_step_increments,
                        "nut_velocity": tail_nut_velocity,
                        "torque_median": tail_torque_median,
                    }
                    case_reports[case_name] = {
                        "action": list(action_values),
                        "axial_travel_m": (
                            states[-1].axial_travel_m
                        ),
                        "final_clamp_positions_rad": [
                            float(value)
                            for value in states[-1].clamp_positions_rad
                        ],
                        "final_finger_torques_nm": [
                            float(value)
                            for value in states[-1].finger_torques_nm
                        ],
                        "episode_randomization": reset_info.get(
                            "episode_randomization"
                        ),
                        "loaded_channels_minimum": (
                            minimum_loaded_channels
                        ),
                        "maximum_finger_torque_nm": (
                            episode_safety.max_finger_torque_delta
                        ),
                        "nut_angle_degrees": math.degrees(
                            states[-1].nut_angle_rad
                        ),
                        "nut_minimum_step_degrees": math.degrees(
                            float(np.min(nut_step_increments))
                        ),
                        "nut_tail_median_speed_degrees_s": (
                            math.degrees(tail_nut_velocity)
                        ),
                        "passed": case_ok,
                        "policy_steps": len(states),
                        "reset_checkpoint": reset_info[
                            "reset_checkpoint"
                        ],
                        "tail_clamp_median_rad": [
                            float(value)
                            for value in tail_clamp_median
                        ],
                        "tail_torque_median_nm": [
                            float(value)
                            for value in tail_torque_median
                        ],
                        "termination_reason": final_info.get(
                            "termination_reason", ""
                        ),
                    }

                slow = case_histories["q7_slow"]
                fast = case_histories["q7_fast"]
                slow_final_degrees = case_reports[
                    "q7_slow"
                ]["nut_angle_degrees"]
                fast_final_degrees = case_reports[
                    "q7_fast"
                ]["nut_angle_degrees"]
                speed_difference_degrees_s = math.degrees(
                    fast["nut_velocity"] - slow["nut_velocity"]
                )
                progress_difference_degrees = (
                    fast_final_degrees - slow_final_degrees
                )
                q7_effect_ok = bool(
                    slow_final_degrees > 0.0
                    and fast_final_degrees > 0.0
                    and math.degrees(
                        float(np.min(slow["nut_increments"]))
                    )
                    >= -0.05
                    and math.degrees(
                        float(np.min(fast["nut_increments"]))
                    )
                    >= -0.05
                    and speed_difference_degrees_s >= 2.0
                    and progress_difference_degrees >= 2.0
                )
                baseline_torques = case_histories[
                    "baseline"
                ]["torque_median"]
                baseline_clamps = case_histories[
                    "baseline"
                ]["clamp_median"]
                torque_response_rows = []
                clamp_position_response = []
                clamp_effect_checks = []
                for channel_index, case_name in enumerate(
                    (
                        "f1j2_tighter",
                        "f2j1_tighter",
                        "f3j2_tighter",
                    )
                ):
                    torque_response = (
                        case_histories[case_name]["torque_median"]
                        - baseline_torques
                    )
                    position_response = (
                        case_histories[case_name]["clamp_median"]
                        - baseline_clamps
                    )
                    torque_response_rows.append(
                        [float(value) for value in torque_response]
                    )
                    clamp_position_response.append(
                        [float(value) for value in position_response]
                    )
                    clamp_effect_checks.append(
                        abs(float(torque_response[channel_index]))
                        >= 0.02
                    )
                clamp_effect_ok = all(clamp_effect_checks)

                reference_signature = initial_signatures[0]
                reset_position_repeatability = max(
                    max(
                        float(
                            np.linalg.norm(
                                signature["body_position"]
                                - reference_signature[
                                    "body_position"
                                ]
                            )
                        ),
                        float(
                            np.linalg.norm(
                                signature["nut_position"]
                                - reference_signature[
                                    "nut_position"
                                ]
                            )
                        ),
                    )
                    for signature in initial_signatures
                )
                reset_q7_repeatability = max(
                    abs(
                        signature["q7"]
                        - reference_signature["q7"]
                    )
                    for signature in initial_signatures
                )
                initial_clamp_array = np.asarray(
                    initial_clamp_positions, dtype=np.float64
                )
                initial_torque_array = np.asarray(
                    initial_finger_torques, dtype=np.float64
                )
                initial_clamp_spread = float(
                    np.max(np.ptp(initial_clamp_array, axis=0))
                )
                initial_torque_spread = float(
                    np.max(np.ptp(initial_torque_array, axis=0))
                )
                reset_repeatability_ok = bool(
                    reset_position_repeatability <= 0.0001
                    and reset_q7_repeatability <= math.radians(0.1)
                    and initial_clamp_spread <= 0.001
                    and initial_torque_spread <= 0.01
                )
                reset_snap_maxima, reset_snap_ok = (
                    summarize_reset_diagnostics(
                        adapter.reset_diagnostics
                    )
                )
                final_runtime_thread_prim_count = sum(
                    1
                    for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith(
                        runtime_root + "/"
                    )
                )
                passed = bool(
                    all(
                        report["passed"]
                        for report in case_reports.values()
                    )
                    and q7_effect_ok
                    and clamp_effect_ok
                    and reset_repeatability_ok
                    and adapter.reset_count == len(action_cases)
                    and adapter.thread_proxy_rebuild_count
                    == len(action_cases)
                    and reset_snap_ok
                    and final_runtime_thread_prim_count
                    == runtime_thread_prim_count
                )
                metrics.update(
                    {
                        "case_reports": case_reports,
                        "clamp_effect_checks": clamp_effect_checks,
                        "clamp_effect_ok": clamp_effect_ok,
                        "clamp_position_response_matrix_rad": (
                            clamp_position_response
                        ),
                        "episode_object_pose_writes": 0,
                        "hard_reset_count": adapter.reset_count,
                        "initial_clamp_position_spread_rad": (
                            initial_clamp_spread
                        ),
                        "initial_torque_spread_nm": (
                            initial_torque_spread
                        ),
                        "interface_version": (
                            residual_config.interface_version
                        ),
                        "control_observation_randomization_applied": (
                            adapter.randomization_enabled
                        ),
                        "episode_randomization_history": (
                            adapter.episode_randomization_history
                        ),
                        "passed": passed,
                        "physicsusd_disjoint_warning": (
                            "expected_absent_after_proxy_removal"
                        ),
                        "reset_object_pose_writes": 0,
                        "safety_signal_source": "raw_physics",
                        "q7_effect_ok": q7_effect_ok,
                        "q7_progress_difference_degrees": (
                            progress_difference_degrees
                        ),
                        "q7_speed_difference_degrees_s": (
                            speed_difference_degrees_s
                        ),
                        "reset_checkpoint_snap_maxima": (
                            reset_snap_maxima
                        ),
                        "reset_checkpoint_snap_ok": reset_snap_ok,
                        "reset_position_repeatability_m": (
                            reset_position_repeatability
                        ),
                        "reset_q7_repeatability_degrees": math.degrees(
                            reset_q7_repeatability
                        ),
                        "reset_repeatability_ok": (
                            reset_repeatability_ok
                        ),
                        "residual_action_size": 4,
                        "residual_observation_size": 24,
                        "scene_build_count": 1,
                        "simulation_app_count": 1,
                        "thread_proxy_rebuild_count": (
                            adapter.thread_proxy_rebuild_count
                        ),
                        "thread_proxy_reset_strategy": (
                            "remove_hard_reset_contact_recovery_recreate"
                        ),
                        "torque_response_matrix_nm": (
                            torque_response_rows
                        ),
                    }
                )
                print(
                    json.dumps(metrics, sort_keys=True), flush=True
                )
                print(
                    banner
                    + " "
                    + ("PASSED" if passed else "FAILED"),
                    flush=True,
                )
                return

            if arguments.mode == "residual-sac-smoke":
                import gymnasium
                from kcg_rl.connector_residual_env import (
                    ConnectorResidualEnv,
                )
                from stable_baselines3 import SAC
                from stable_baselines3.common.env_checker import (
                    check_env,
                )
                from stable_baselines3.common.monitor import Monitor
                import stable_baselines3
                import torch

                if torch.__version__ != "2.11.0+cu128":
                    raise RuntimeError(
                        "CUDA SAC smoke refuses changed torch: "
                        + torch.__version__
                    )
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA SAC smoke forbids CPU fallback"
                    )

                gym_environment = ConnectorResidualEnv(adapter)
                check_env(
                    gym_environment,
                    warn=True,
                    skip_render_check=True,
                )
                monitored_environment = Monitor(gym_environment)
                model = SAC(
                    "MlpPolicy",
                    monitored_environment,
                    device="cuda",
                    seed=42,
                    verbose=0,
                    learning_starts=0,
                    buffer_size=128,
                    batch_size=2,
                    train_freq=1,
                    gradient_steps=1,
                    policy_kwargs={"net_arch": [64, 64]},
                )
                actor_before = torch.cat(
                    [
                        parameter.detach().flatten().cpu()
                        for parameter in model.actor.parameters()
                    ]
                )
                model.learn(
                    total_timesteps=arguments.training_timesteps
                )
                actor_after = torch.cat(
                    [
                        parameter.detach().flatten().cpu()
                        for parameter in model.actor.parameters()
                    ]
                )
                actor_parameter_delta = float(
                    torch.max(torch.abs(actor_after - actor_before))
                )
                output_directory = Path(
                    arguments.training_output
                ).expanduser().resolve()
                output_directory.mkdir(
                    parents=True, exist_ok=True
                )
                model_base_path = (
                    output_directory
                    / "connector_residual_sac_smoke"
                )
                model.save(str(model_base_path))
                reloaded = SAC.load(
                    str(model_base_path),
                    env=monitored_environment,
                    device="cuda",
                )
                actor_device = str(
                    next(model.actor.parameters()).device
                )
                reloaded_actor_device = str(
                    next(reloaded.actor.parameters()).device
                )
                replay_size = int(model.replay_buffer.size())
                model_path = model_base_path.with_suffix(".zip")
                reset_snap_maxima, reset_snap_ok = (
                    summarize_reset_diagnostics(
                        adapter.reset_diagnostics
                    )
                )
                final_runtime_thread_prim_count = sum(
                    1
                    for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith(
                        runtime_root + "/"
                    )
                )
                source_path = Path(__file__).resolve()
                backend_path = (
                    repository
                    / "src/kcg_connector/kcg_connector/"
                    "isaac_residual_backend.py"
                )
                residual_contract_path = (
                    repository
                    / "src/kcg_connector/kcg_connector/residual_rl.py"
                )
                gym_environment_path = (
                    repository
                    / "src/kcg_rl/kcg_rl/connector_residual_env.py"
                )
                randomization_contract_path = (
                    repository
                    / "src/kcg_connector/kcg_connector/"
                    "residual_randomization.py"
                )

                def file_sha256(path):
                    return hashlib.sha256(path.read_bytes()).hexdigest()

                source_sha256 = file_sha256(source_path)
                backend_sha256 = file_sha256(backend_path)
                residual_contract_sha256 = file_sha256(
                    residual_contract_path
                )
                gym_environment_sha256 = file_sha256(
                    gym_environment_path
                )
                config_sha256 = file_sha256(config_path)
                robot_asset_sha256 = file_sha256(robot_asset)
                connector_asset_sha256 = file_sha256(connector_asset)
                randomization_config_sha256 = (
                    None
                    if randomization_config_path is None
                    else file_sha256(randomization_config_path)
                )
                randomization_contract_sha256 = file_sha256(
                    randomization_contract_path
                )
                passed = bool(
                    int(model.num_timesteps)
                    >= arguments.training_timesteps
                    and replay_size > 0
                    and actor_device.startswith("cuda")
                    and reloaded_actor_device.startswith("cuda")
                    and actor_parameter_delta > 0.0
                    and model_path.is_file()
                    and reset_snap_ok
                    and adapter.thread_proxy_rebuild_count
                    == adapter.reset_count
                    and final_runtime_thread_prim_count
                    == runtime_thread_prim_count
                )
                training_metrics = {
                    "actor_device": actor_device,
                    "actor_parameter_max_delta": (
                        actor_parameter_delta
                    ),
                    "environment_hard_resets": (
                        adapter.reset_count
                    ),
                    "control_observation_randomization_applied": (
                        adapter.randomization_enabled
                    ),
                    "episode_randomization_history": (
                        adapter.episode_randomization_history
                    ),
                    "episode_object_pose_writes": 0,
                    "gymnasium": gymnasium.__version__,
                    "passed": passed,
                    "physicsusd_disjoint_warning": (
                        "expected_absent_after_proxy_removal"
                    ),
                    "physics_randomization_applied": False,
                    "model_path": str(model_path),
                    "model_timesteps": int(model.num_timesteps),
                    "random_seed": 42,
                    "reloaded_actor_device": (
                        reloaded_actor_device
                    ),
                    "replay_size": replay_size,
                    "reset_checkpoint_snap_maxima": (
                        reset_snap_maxima
                    ),
                    "reset_checkpoint_snap_ok": reset_snap_ok,
                    "reset_object_pose_writes": 0,
                    "run_timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "source_backend_path": str(backend_path),
                    "source_backend_sha256": backend_sha256,
                    "source_config_path": str(config_path),
                    "source_config_sha256": config_sha256,
                    "source_connector_asset_path": str(
                        connector_asset
                    ),
                    "source_connector_asset_sha256": (
                        connector_asset_sha256
                    ),
                    "source_gym_environment_path": str(
                        gym_environment_path
                    ),
                    "source_gym_environment_sha256": (
                        gym_environment_sha256
                    ),
                    "source_residual_contract_path": str(
                        residual_contract_path
                    ),
                    "source_residual_contract_sha256": (
                        residual_contract_sha256
                    ),
                    "source_randomization_config_path": (
                        None
                        if randomization_config_path is None
                        else str(randomization_config_path)
                    ),
                    "source_randomization_config_sha256": (
                        randomization_config_sha256
                    ),
                    "source_randomization_contract_path": str(
                        randomization_contract_path
                    ),
                    "source_randomization_contract_sha256": (
                        randomization_contract_sha256
                    ),
                    "source_robot_asset_path": str(robot_asset),
                    "source_robot_asset_sha256": robot_asset_sha256,
                    "source_script_path": str(source_path),
                    "source_script_sha256": source_sha256,
                    "stable_baselines3": (
                        stable_baselines3.__version__
                    ),
                    "safety_signal_source": "raw_physics",
                    "thread_proxy_rebuild_count": (
                        adapter.thread_proxy_rebuild_count
                    ),
                    "thread_proxy_reset_strategy": (
                        "remove_hard_reset_contact_recovery_recreate"
                    ),
                    "torch": torch.__version__,
                    "torch_cuda_build": torch.version.cuda,
                }
                metadata_path = (
                    output_directory
                    / "connector_residual_sac_smoke_metadata.json"
                )
                metadata_path.write_text(
                    json.dumps(
                        training_metrics,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                metrics.update(training_metrics)
                metrics.update(
                    {
                        "environment_checker": "passed",
                        "interface_version": (
                            residual_config.interface_version
                        ),
                        "passed": passed,
                        "residual_action_size": 4,
                        "residual_observation_size": 24,
                        "scene_build_count": 1,
                        "simulation_app_count": 1,
                    }
                )
                monitored_environment.close()
                print(
                    json.dumps(metrics, sort_keys=True),
                    flush=True,
                )
                print(
                    banner
                    + " "
                    + ("PASSED" if passed else "FAILED"),
                    flush=True,
                )
                return

            zero_action = np.zeros(4, dtype=np.float32)
            episode_reports = []
            initial_signatures = []
            for episode_index in range(arguments.episodes):
                observation, reset_info = adapter.reset(
                    seed=reproducible_stream_reset_seed(
                        arguments.reset_seed, episode_index
                    )
                )
                initial_signatures.append(
                    adapter.initial_signature
                )
                terminated = False
                truncated = False
                episode_return = 0.0
                final_info = {}
                while not (terminated or truncated):
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        final_info,
                    ) = adapter.step(zero_action)
                    episode_return += reward
                final_state = adapter.previous_state
                expected_axial = (
                    residual_config.helical_lead_m
                    * final_state.nut_angle_rad
                    / (2.0 * math.pi)
                )
                helical_error = (
                    final_state.axial_travel_m - expected_axial
                )
                minimum_axial_travel = (
                    residual_config.minimum_axial_progress_fraction
                    * expected_axial
                )
                axial_progress_fraction = (
                    0.0
                    if expected_axial <= 0.0
                    else final_state.axial_travel_m / expected_axial
                )
                axial_progress_gate_passed = bool(
                    expected_axial > 0.0
                    and final_state.axial_travel_m
                    >= minimum_axial_travel
                )
                final_q7_delta = (
                    final_state.q7_position_rad - adapter.start_q7
                )
                episode_safety = adapter.episode_safety
                raw_safety_report = adapter.raw_safety_report
                raw_safety_projection_ok = bool(
                    final_info.get("raw_safety_passed")
                    is raw_safety_report["passed"]
                    and final_info.get(
                        "raw_safety_failure_reasons"
                    )
                    == raw_safety_report["failure_reasons"]
                    and final_info.get("raw_safety_peaks")
                    == raw_safety_report["metrics"]
                    and final_info.get("safety_signal_source")
                    == "raw_physics"
                )
                episode_ok = bool(
                    terminated
                    and not truncated
                    and final_info.get("termination_reason")
                    == "success"
                    and observation.shape == (24,)
                    and np.all(np.isfinite(observation))
                    and abs(
                        final_state.nut_angle_rad
                        - residual_config.target_angle_rad
                    )
                    <= residual_config.success_angle_tolerance_rad
                    and abs(
                        final_q7_delta
                        + final_state.nut_angle_rad
                    )
                    <= math.radians(3.0)
                    and abs(helical_error)
                    <= residual_config.helical_tolerance_m(
                        final_state.nut_angle_rad
                    )
                    and axial_progress_gate_passed
                    and loaded_torque_channels(
                        final_state, residual_config
                    )
                    >= residual_config.minimum_loaded_torque_channels
                    and episode_safety.max_finger_torque_delta
                    <= residual_config.
                    maximum_absolute_finger_torque_nm
                    and episode_safety.finite_throughout
                    and episode_safety.max_limit_violation <= 0.02
                    and raw_safety_report["passed"] is True
                    and raw_safety_projection_ok
                )
                episode_reports.append(
                    {
                        "axial_travel_m": (
                            final_state.axial_travel_m
                        ),
                        "axial_progress_fraction": (
                            axial_progress_fraction
                        ),
                        "axial_progress_gate_passed": (
                            axial_progress_gate_passed
                        ),
                        "episode": episode_index + 1,
                        "episode_randomization": reset_info.get(
                            "episode_randomization"
                        ),
                        "helical_error_m": helical_error,
                        "loaded_channels": loaded_torque_channels(
                            final_state, residual_config
                        ),
                        "maximum_finger_torque_nm": (
                            episode_safety.max_finger_torque_delta
                        ),
                        "minimum_axial_progress_fraction": (
                            residual_config.minimum_axial_progress_fraction
                        ),
                        "minimum_axial_travel_m": (
                            minimum_axial_travel
                        ),
                        "nut_angle_degrees": math.degrees(
                            final_state.nut_angle_rad
                        ),
                        "passed": episode_ok,
                        "policy_steps": adapter.policy_steps,
                        "q7_delta_degrees": math.degrees(
                            final_q7_delta
                        ),
                        "raw_safety_failure_reasons": (
                            raw_safety_report["failure_reasons"]
                        ),
                        "raw_safety_passed": raw_safety_report[
                            "passed"
                        ],
                        "raw_safety_projection_ok": (
                            raw_safety_projection_ok
                        ),
                        "raw_safety_report": raw_safety_report,
                        "reset": reset_info["reset"],
                        "reset_checkpoint": reset_info[
                            "reset_checkpoint"
                        ],
                        "return": episode_return,
                        "stable_hold_seconds": (
                            final_state.stable_hold_seconds
                        ),
                        "termination_reason": final_info.get(
                            "termination_reason", "time_limit"
                        ),
                    }
                )

            first_signature, second_signature = initial_signatures
            reset_position_error = max(
                float(
                    np.linalg.norm(
                        first_signature["body_position"]
                        - second_signature["body_position"]
                    )
                ),
                float(
                    np.linalg.norm(
                        first_signature["nut_position"]
                        - second_signature["nut_position"]
                    )
                ),
            )
            reset_q7_error = abs(
                first_signature["q7"] - second_signature["q7"]
            )
            final_runtime_thread_prim_count = sum(
                1
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(runtime_root + "/")
            )
            reset_snap_maxima, reset_snap_ok = (
                summarize_reset_diagnostics(adapter.reset_diagnostics)
            )
            passed = bool(
                all(report["passed"] for report in episode_reports)
                and reset_position_error <= 0.0001
                and reset_q7_error <= math.radians(0.1)
                and adapter.reset_count == arguments.episodes
                and adapter.thread_proxy_rebuild_count
                == arguments.episodes
                and reset_snap_ok
                and final_runtime_thread_prim_count
                == runtime_thread_prim_count
                and not hinge_drive_properties
                and not hinge_drive_schemas
            )
            metrics.update(
                {
                    "episode_object_pose_writes": 0,
                    "episode_reports": episode_reports,
                    "episodes": arguments.episodes,
                    "hard_reset_count": arguments.episodes,
                    "interface_version": (
                        residual_config.interface_version
                    ),
                    "control_observation_randomization_applied": (
                        adapter.randomization_enabled
                    ),
                    "episode_randomization_history": (
                        adapter.episode_randomization_history
                    ),
                    "passed": passed,
                    "policy_rate_hz": (
                        residual_config.policy_rate_hz
                    ),
                    "reset_position_repeatability_m": (
                        reset_position_error
                    ),
                    "reset_checkpoint_snap_maxima": (
                        reset_snap_maxima
                    ),
                    "reset_checkpoint_snap_ok": reset_snap_ok,
                    "reset_object_pose_writes": 0,
                    "safety_signal_source": "raw_physics",
                    "reset_q7_repeatability_degrees": math.degrees(
                        reset_q7_error
                    ),
                    "reset_via": (
                        "remove_hard_reset_contact_recovery_recreate"
                    ),
                    "residual_action_size": 4,
                    "residual_observation_size": 24,
                    "runtime_thread_prim_count": (
                        final_runtime_thread_prim_count
                    ),
                    "scene_build_count": 1,
                    "simulation_app_count": 1,
                    "thread_proxy_rebuild_count": (
                        adapter.thread_proxy_rebuild_count
                    ),
                }
            )
            print(json.dumps(metrics, sort_keys=True), flush=True)
            print(
                banner + " " + ("PASSED" if passed else "FAILED"),
                flush=True,
            )
            return

        controlled_name_to_offset = {
            name: index for index, name in enumerate(controlled_names)
        }
        q7_command_offset = controlled_name_to_offset["iiwa_joint_7"]
        motion_start_target = insertion_target.copy()
        released_hand_error = None
        thread_brake_applied = False
        maximum_open_hand_contact_records = 0
        if arguments.mode == "open-hand":
            # The ideal rack is intentionally lossless and otherwise
            # back-drives into a max-angular-velocity instability when the
            # fingers release.  Replace the rack/hinge temporarily with an
            # explicit current-pose fixed brake.  This models static thread
            # self-lock during regrasp; it is not a motor or pose write.
            world.pause()
            UsdPhysics.Joint(hinge_prim).CreateJointEnabledAttr(False)
            rack.CreateJointEnabledAttr(False)
            brake_body_position, brake_body_orientation = _matrix_pose(
                Gf,
                Usd,
                UsdGeom,
                stage.GetPrimAtPath(body_path),
            )
            brake_nut_position, brake_nut_orientation = _matrix_pose(
                Gf,
                Usd,
                UsdGeom,
                stage.GetPrimAtPath(nut_path),
            )
            world_delta = Gf.Vec3d(
                float(
                    brake_body_position[0] - brake_nut_position[0]
                ),
                float(
                    brake_body_position[1] - brake_nut_position[1]
                ),
                float(
                    brake_body_position[2] - brake_nut_position[2]
                ),
            )
            local_position1 = brake_nut_orientation.GetInverse().Transform(
                world_delta
            )
            local_rotation1 = (
                brake_nut_orientation.GetInverse()
                * brake_body_orientation
            )
            local_rotation1_imaginary = local_rotation1.GetImaginary()
            brake_path = f"{runtime_root}/ReleasedThreadBrake"
            brake = UsdPhysics.FixedJoint.Define(stage, brake_path)
            brake.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
            brake.CreateBody1Rel().SetTargets([Sdf.Path(nut_path)])
            brake.CreateLocalPos0Attr(Gf.Vec3f(0.0))
            brake.CreateLocalRot0Attr(Gf.Quatf(1.0))
            brake.CreateLocalPos1Attr(
                Gf.Vec3f(
                    float(local_position1[0]),
                    float(local_position1[1]),
                    float(local_position1[2]),
                )
            )
            brake.CreateLocalRot1Attr(
                Gf.Quatf(
                    float(local_rotation1.GetReal()),
                    Gf.Vec3f(
                        float(local_rotation1_imaginary[0]),
                        float(local_rotation1_imaginary[1]),
                        float(local_rotation1_imaginary[2]),
                    ),
                )
            )
            brake.CreateCollisionEnabledAttr(False)
            thread_brake_applied = True
            metrics["thread_brake"] = "current_pose_fixed_joint"
            world.play()
            simulation_app.update()
            for _ in range(10):
                robot.apply_action(
                    ArticulationAction(
                        joint_positions=insertion_target,
                        joint_indices=controlled_indices,
                    )
                )
                world.step(render=arguments.gui)
                observe()
            release_target = named_joint_target(
                dof_names, insert_arm, task["open_hand"]
            )[controlled_indices].astype(np.float32)
            for step_index in range(arguments.closure_steps):
                blend = float(step_index + 1) / float(
                    arguments.closure_steps
                )
                target = insertion_target + blend * (
                    release_target - insertion_target
                )
                robot.apply_action(
                    ArticulationAction(
                        joint_positions=target.astype(np.float32),
                        joint_indices=controlled_indices,
                    )
                )
                world.step(render=arguments.gui)
                observe()
            motion_start_target = release_target
            for _ in range(arguments.settle_steps):
                robot.apply_action(
                    ArticulationAction(
                        joint_positions=motion_start_target,
                        joint_indices=controlled_indices,
                    )
                )
                world.step(render=arguments.gui)
                observe()
            released_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            released_hand_error = float(
                np.max(
                    np.abs(
                        released_positions[hand_indices]
                        - motion_start_target[len(ARM_JOINT_NAMES):]
                    )
                )
            )
            maximum_open_hand_contact_records = (
                count_hand_nut_contact_records()
            )

        motion_end_target = motion_start_target.copy()
        q7_command_start = float(
            motion_start_target[q7_command_offset]
        )
        if arguments.mode in ("twist", "open-hand"):
            motion_end_target[q7_command_offset] = (
                q7_command_start + probe_angle
            )
        else:
            axial_scale = (
                arguments.axial_counterfactual_distance
                / total_approach_displacement
            )
            motion_end_target[:6] += axial_scale * (
                insertion_target[:6] - grasp_target[:6]
            )
        q7_command_target = float(
            motion_end_target[q7_command_offset]
        )
        if not (
            task["q7_safe_lower"]
            <= q7_command_target
            <= task["q7_safe_upper"]
        ):
            raise ValueError("q7 probe target is outside the safe window")

        pre_twist_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        pre_twist_q7 = float(pre_twist_positions[q7_index])
        pre_twist_body_position, _ = body.get_world_pose()
        pre_twist_nut_position, _ = nut.get_world_pose()
        motion_start_nut_angle = _wrapped_relative_z_angle(
            Gf,
            Usd,
            UsdGeom,
            stage.GetPrimAtPath(body_path),
            stage.GetPrimAtPath(nut_path),
        )
        unwrapped_nut_angle = motion_start_nut_angle
        twist_steps = int(
            math.ceil(abs(probe_angle) / probe_speed * 240.0)
        )
        for step_index in range(twist_steps):
            blend = float(step_index + 1) / float(twist_steps)
            target = motion_start_target + blend * (
                motion_end_target - motion_start_target
            )
            robot.apply_action(
                ArticulationAction(
                    joint_positions=target.astype(np.float32),
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            observe()
            if arguments.mode == "open-hand":
                maximum_open_hand_contact_records = max(
                    maximum_open_hand_contact_records,
                    count_hand_nut_contact_records(),
                )
            wrapped_nut_angle = _wrapped_relative_z_angle(
                Gf,
                Usd,
                UsdGeom,
                stage.GetPrimAtPath(body_path),
                stage.GetPrimAtPath(nut_path),
            )
            unwrapped_nut_angle = _unwrap(
                unwrapped_nut_angle, wrapped_nut_angle
            )

        for _ in range(arguments.settle_steps):
            robot.apply_action(
                ArticulationAction(
                    joint_positions=motion_end_target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            observe()
            if arguments.mode == "open-hand":
                maximum_open_hand_contact_records = max(
                    maximum_open_hand_contact_records,
                    count_hand_nut_contact_records(),
                )
            wrapped_nut_angle = _wrapped_relative_z_angle(
                Gf,
                Usd,
                UsdGeom,
                stage.GetPrimAtPath(body_path),
                stage.GetPrimAtPath(nut_path),
            )
            unwrapped_nut_angle = _unwrap(
                unwrapped_nut_angle, wrapped_nut_angle
            )

        final_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        final_velocities = np.asarray(
            robot.get_joint_velocities(), dtype=np.float64
        )
        final_efforts = np.asarray(
            robot.get_measured_joint_efforts(
                joint_indices=sensor_indices
            ),
            dtype=np.float64,
        )
        final_body_position, _ = body.get_world_pose()
        final_nut_position, _ = nut.get_world_pose()
        final_torque_deltas = final_efforts - tare_efforts
        loaded_after_twist = int(
            np.count_nonzero(np.abs(final_torque_deltas) >= 0.02)
        )

        q7_delta = float(final_positions[q7_index] - pre_twist_q7)
        nut_delta = float(
            unwrapped_nut_angle - motion_start_nut_angle
        )
        axial_travel = float(
            final_body_position[2] - pre_twist_body_position[2]
        )
        expected_axial_travel = float(
            -thread_proxy_direction
            * task["lead"]
            * nut_delta
            / (2.0 * math.pi)
        )
        helical_error = axial_travel - expected_axial_travel
        commanded_q7_delta = q7_command_target - q7_command_start
        commanded_non_q7_delta = float(
            np.max(
                np.abs(
                    motion_end_target[:6] - motion_start_target[:6]
                )
            )
        )
        q7_tracking_error = q7_delta - commanded_q7_delta
        # Imported q7 positive and the connector's local nut angle have
        # opposite signs.  Their magnitudes must nevertheless track.
        q7_to_nut_slip = q7_delta + nut_delta
        non_q7_motion = float(
            np.max(
                np.abs(
                    final_positions[arm_indices[:-1]]
                    - pre_twist_positions[arm_indices[:-1]]
                )
            )
        )
        joint_separation_change = abs(
            float(
                np.linalg.norm(final_nut_position - final_body_position)
                - np.linalg.norm(
                    pre_twist_nut_position - pre_twist_body_position
                )
            )
        )
        finite = bool(
            finite_throughout
            and np.all(np.isfinite(final_positions))
            and np.all(np.isfinite(final_velocities))
            and np.all(np.isfinite(final_efforts))
            and np.all(np.isfinite(final_body_position))
            and np.all(np.isfinite(final_nut_position))
        )
        stable = bool(
            finite
            and max_abs_velocity <= 20.0
            and max_limit_violation <= 0.02
        )
        constraint_activation_ok = bool(
            constraint_position_jump <= 0.001
            and constraint_orientation_jump <= math.radians(0.5)
            and constraint_nut_angle_jump <= math.radians(0.5)
        )
        insertion_ok = bool(
            total_approach_displacement >= 0.025
            and insertion_lateral_offset <= 0.0015
            and insertion_axis_error <= math.radians(2.0)
        )
        q7_tracking_ok = bool(
            abs(q7_tracking_error) <= math.radians(2.0)
        )
        q7_transmission_ok = bool(
            nut_delta * probe_angle < 0.0
            and abs(q7_to_nut_slip) <= math.radians(3.0)
        )
        helical_relation_ok = bool(
            axial_travel < 0.0
            and axial_travel * expected_axial_travel > 0.0
            and abs(helical_error)
            <= min(
                task["helical_error_tolerance"],
                max(0.00005, 0.25 * abs(expected_axial_travel)),
            )
            and abs(axial_travel)
            >= 0.75 * abs(expected_axial_travel)
        )
        grip_ok = bool(
            loaded_before_insertion >= task["minimum_loaded_channels"]
            and loaded_after_twist >= task["minimum_loaded_channels"]
            and max_finger_torque_delta
            <= task["maximum_finger_torque"]
            and joint_separation_change <= 0.001
        )
        q7_only_ok = bool(
            non_q7_motion <= 0.02
            and not hinge_drive_properties
            and not hinge_drive_schemas
        )
        nominal_twist_passed = bool(
            stable
            and constraint_activation_ok
            and insertion_ok
            and q7_tracking_ok
            and q7_transmission_ok
            and helical_relation_ok
            and grip_ok
            and q7_only_ok
        )
        counterfactual_motion_absent = bool(
            abs(nut_delta) <= math.radians(2.0)
            and abs(axial_travel) <= 0.0001
        )
        static_axial_counterfactual_ok = bool(
            stable
            and constraint_activation_ok
            and insertion_ok
            and grip_ok
            and abs(q7_delta) <= math.radians(0.5)
            and commanded_non_q7_delta > 0.0
            and counterfactual_motion_absent
            and not hinge_drive_properties
            and not hinge_drive_schemas
        )
        open_hand_counterfactual_ok = bool(
            stable
            and constraint_activation_ok
            and insertion_ok
            and released_hand_error is not None
            and released_hand_error <= 0.1
            and maximum_open_hand_contact_records == 0
            and q7_tracking_ok
            and counterfactual_motion_absent
            and not hinge_drive_properties
            and not hinge_drive_schemas
        )
        if arguments.mode == "twist":
            passed = nominal_twist_passed
        elif arguments.mode == "q7-static-axial":
            passed = static_axial_counterfactual_ok
        else:
            passed = open_hand_counterfactual_ok
        metrics.update(
            {
                "axial_travel_m": axial_travel,
                "commanded_non_q7_delta_rad": (
                    commanded_non_q7_delta
                ),
                "commanded_q7_delta_degrees": math.degrees(
                    commanded_q7_delta
                ),
                "constraint_activation_ok": constraint_activation_ok,
                "constraint_nut_angle_jump_degrees": math.degrees(
                    constraint_nut_angle_jump
                ),
                "constraint_orientation_jump_degrees": math.degrees(
                    constraint_orientation_jump
                ),
                "constraint_position_jump_m": constraint_position_jump,
                "counterfactual_motion_absent": (
                    counterfactual_motion_absent
                ),
                "expected_axial_travel_m": expected_axial_travel,
                "finite": finite,
                "grip_ok": grip_ok,
                "helical_error_m": helical_error,
                "helical_relation_ok": helical_relation_ok,
                "insertion_displacement_m": insertion_displacement,
                "insertion_axis_error_degrees": math.degrees(
                    insertion_axis_error
                ),
                "insertion_start_axis_error_degrees": math.degrees(
                    insertion_start_axis_error
                ),
                "insertion_lateral_offset_m": insertion_lateral_offset,
                "insertion_ok": insertion_ok,
                "joint_separation_change_m": joint_separation_change,
                "loaded_channels_after_twist": loaded_after_twist,
                "loaded_channels_before_insertion": (
                    loaded_before_insertion
                ),
                "maximum_absolute_finger_torque_delta_nm": (
                    max_finger_torque_delta
                ),
                "maximum_joint_limit_violation_rad": (
                    max_limit_violation
                ),
                "maximum_joint_speed_rad_s": max_abs_velocity,
                "maximum_open_hand_contact_records": (
                    maximum_open_hand_contact_records
                ),
                "nut_rotation_degrees": math.degrees(nut_delta),
                "nominal_twist_passed": nominal_twist_passed,
                "open_hand_counterfactual_ok": (
                    open_hand_counterfactual_ok
                ),
                "passed": passed,
                "q7_command_start_rad": q7_command_start,
                "q7_command_target_rad": q7_command_target,
                "q7_only_ok": q7_only_ok,
                "q7_rotation_degrees": math.degrees(q7_delta),
                "q7_to_nut_slip_degrees": math.degrees(
                    q7_to_nut_slip
                ),
                "q7_tracking_error_degrees": math.degrees(
                    q7_tracking_error
                ),
                "q7_tracking_ok": q7_tracking_ok,
                "q7_transmission_ok": q7_transmission_ok,
                "released_hand_error_rad": released_hand_error,
                "stable": stable,
                "static_axial_counterfactual_ok": (
                    static_axial_counterfactual_ok
                ),
                "supported_axis_error_degrees": math.degrees(
                    supported_axis_error
                ),
                "thread_brake_applied": thread_brake_applied,
                "twist_steps": twist_steps,
                "total_approach_displacement_m": (
                    total_approach_displacement
                ),
                "unloaded_support_settle_m": (
                    unsupported_settle_displacement
                ),
            }
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print(
            banner + " " + ("PASSED" if passed else "FAILED"),
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
        print(banner + " FAILED", flush=True)
    finally:
        if arguments.keep_open and arguments.gui:
            print(
                banner + " GUI REMAINS OPEN; "
                "close the window to exit",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
