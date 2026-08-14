#!/usr/bin/env python3

"""Validate D38999 tabletop settle beside a KUKA held at Home.

This is a Home-only physics and visualization node.  It does not move toward,
perceive, plan for, or grasp either connector endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback


ARM_JOINT_NAMES = tuple(f"iiwa_joint_{index}" for index in range(1, 8))
ACTIVE_HAND_JOINT_NAMES = ("f1j1", "f1j2", "f2j1", "f3j2")
MIMIC_HAND_JOINT_NAMES = ("f1j3", "f2j2", "f3j1", "f3j3")
EXPECTED_DOF_NAMES = ARM_JOINT_NAMES + (
    "f1j1",
    "f1j2",
    "f1j3",
    "f2j1",
    "f2j2",
    "f3j1",
    "f3j2",
    "f3j3",
)
HOME_ACTIVE_JOINT_POSITIONS_RAD = (0.0,) * 11
ROBOT_ROOT_PATH = "/World/HandArm"
ARTICULATION_PATH = ROBOT_ROOT_PATH + "/Geometry/world"
BASE_LINK_PATH = ARTICULATION_PATH + "/iiwa_link_0"
ROBOT_ASSET_SHA256 = {
    "handarm.usda": (
        "031f8241c9dd1e2af96d7b1dde7d2adda7744891832a05e2580fd3398da4216b"
    ),
    "payloads/Physics/mujoco.usda": (
        "cf66f529b2a2a24710e5dc9231ee05b4434ca92c41173430ece9fcba63a1588f"
    ),
    "payloads/Physics/physics.usda": (
        "f86b2379f41e979f64e84cfff6251b2923e34484c6442deb3cb12aa81e21390e"
    ),
    "payloads/Physics/physx.usda": (
        "b80f1622746e2880782dbdd526834c400a0d2bc5a6e4a989d77a601ebc9160cf"
    ),
    "payloads/base.usda": (
        "7788566265d6720bfe72455dd7bc84d2c317b8e2587f067f59d2f22336874708"
    ),
    "payloads/geometries.usd": (
        "ea2ea46f30d8335080f7176139ce05b6523f7c19b7de33951c56c517cd35bff7"
    ),
    "payloads/instances.usda": (
        "f8aa794e99c85b7de1ee551f03835c22911eafac7a7c48cb2cbcb6043ebf637e"
    ),
    "payloads/materials.usda": (
        "74ee9bf9a67ea38c89807b1f5afc1989798ed8e5746f7456c13b812e3b6540e6"
    ),
    "payloads/robot.usda": (
        "2e619c6d7a9232e801b1271855cd0ff329d06912103c80bf7a39e7ce8b6f99af"
    ),
}
ROBOT_TABLE_MINIMUM_CLEARANCE_M = 0.005
MAXIMUM_BASE_TRANSLATION_DRIFT_M = 0.0001
MAXIMUM_BASE_ROTATION_DRIFT_RAD = 0.0001
MAXIMUM_JOINT_LIMIT_VIOLATION_RAD = 0.02
MAXIMUM_FINAL_HOME_ERROR_RAD = 0.05
MAXIMUM_OBSERVED_HOME_ERROR_RAD = 0.05
MAXIMUM_OBSERVED_JOINT_SPEED_RAD_S = 5.0
MAXIMUM_FINAL_JOINT_SPEED_RAD_S = 0.5
ARM_STIFFNESS = 400.0
ARM_DAMPING = 40.0
HAND_STIFFNESS = 25.0
HAND_DAMPING = 2.0
CAMERA_EYE_M = (2.10, 1.65, 1.55)
CAMERA_TARGET_M = (0.31, 0.0, 0.70)


def _verify_robot_asset_bundle(robot_asset):
    """Hash-pin the validated robot root and all of its local payloads."""
    root_asset = Path(robot_asset).expanduser().resolve()
    asset_directory = root_asset.parent
    if root_asset.name != "handarm.usda":
        raise ValueError("robot asset must be the validated handarm.usda")
    verified = {}
    for relative_path, expected_sha256 in ROBOT_ASSET_SHA256.items():
        path = (asset_directory / relative_path).resolve()
        try:
            path.relative_to(asset_directory)
        except ValueError as error:
            raise ValueError(
                "robot payload escapes asset directory"
            ) from error
        if not path.is_file():
            raise ValueError(
                f"robot asset payload is missing: {relative_path}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f"robot asset SHA-256 mismatch: {relative_path}"
            )
        verified[relative_path] = path
    return verified


def _home_control_spec(dof_names):
    """Return canonical active indices and an all-zero 7+4 Home target."""
    names = tuple(str(name) for name in dof_names)
    if len(names) != len(set(names)):
        raise ValueError("articulation DOF names must be unique")
    expected = set(EXPECTED_DOF_NAMES)
    if set(names) != expected or len(names) != len(EXPECTED_DOF_NAMES):
        raise ValueError(
            "unexpected articulation DOF layout: "
            f"missing={sorted(expected - set(names))}, "
            f"unexpected={sorted(set(names) - expected)}"
        )
    active_names = ARM_JOINT_NAMES + ACTIVE_HAND_JOINT_NAMES
    indices = tuple(names.index(name) for name in active_names)
    return active_names, indices, HOME_ACTIVE_JOINT_POSITIONS_RAD


def _path_is_at_or_below(path, root):
    value = str(path)
    prefix = str(root)
    return value == prefix or value.startswith(prefix + "/")


def _classify_robot_external_contact(
    paths, robot_root, table_path, fixture_path, d38999_root
):
    """Classify only contacts containing the exact robot subtree."""
    values = tuple(str(path) for path in paths)
    if not any(_path_is_at_or_below(path, robot_root) for path in values):
        return None
    for category, root in (
        ("table", table_path),
        ("fixture", fixture_path),
        ("d38999", d38999_root),
    ):
        if any(_path_is_at_or_below(path, root) for path in values):
            return category
    return None


def _gf_world_pose(Gf, Usd, UsdGeom, prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform(matrix)
    return transform.GetTranslation(), transform.GetRotation().GetQuat()


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


def _quaternion_z_axis(value):
    import numpy as np

    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        return np.full(3, np.nan, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        return np.full(3, np.nan, dtype=np.float64)
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dtype=np.float64,
    )


def _tilt_from_world_z(value):
    axis = _quaternion_z_axis(value)
    if not all(math.isfinite(float(item)) for item in axis):
        return float("inf")
    return math.acos(max(-1.0, min(1.0, float(axis[2]))))


def _world_aligned_bounds(Usd, UsdGeom, prim):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=False,
    )
    aligned_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return aligned_range.GetMin(), aligned_range.GetMax()


def _arguments(repository):
    parser = argparse.ArgumentParser(
        description="Hold KUKA Home beside the physical D38999 tabletop"
    )
    parser.add_argument(
        "--robot-asset",
        default=str(
            repository
            / "artifacts/kcg_connector/isaac/robot/handarm/handarm.usda"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_tabletop_scene_v1.yaml"
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="render KUKA Home, collision table, fixture and D38999 pair",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="keep the accepted final GUI frame open until closed",
    )
    arguments = parser.parse_args()
    if arguments.keep_open and not arguments.gui:
        parser.error("--keep-open requires --gui")
    return arguments


def main():
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository)

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
        "gui": arguments.gui,
        "keep_open": arguments.keep_open,
        "home_definition": "iiwa 7 + active hand 4, all zero radians",
        "object_pose_writes_after_start": 0,
        "passed": False,
        "perception_available": False,
        "planning_available": False,
        "robot_joint_state_initialization_writes": 2,
        "scene": "kcg_d38999_tabletop_robot_smoke_v1",
        "task_scope": "Home hold only",
    }
    try:
        import numpy as np

        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.rendering_manager import ViewportManager
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

        from kcg_connector.d38999_tabletop_scene import (
            author_d38999_tabletop_scene,
            load_d38999_tabletop_scene,
            verify_d38999_tabletop_asset,
        )

        config_path = Path(arguments.config).expanduser().resolve()
        config = load_d38999_tabletop_scene(config_path)
        d38999_asset = verify_d38999_tabletop_asset(config, repository)
        robot_asset = Path(arguments.robot_asset).expanduser().resolve()
        verified_robot_files = _verify_robot_asset_bundle(robot_asset)
        metrics["verified_robot_asset_file_count"] = len(
            verified_robot_files
        )
        metrics["robot_asset_root_sha256"] = ROBOT_ASSET_SHA256[
            "handarm.usda"
        ]

        table_front_x = (
            config.table.center_m[0] - 0.5 * config.table.size_m[0]
        )
        if not math.isclose(table_front_x, 0.150, abs_tol=1.0e-9):
            raise ValueError("D38999 table front edge must remain x=0.150 m")

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / config.physics.rate_hz,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        stage = get_current_stage()
        metrics["d38999_authoring"] = author_d38999_tabletop_scene(
            stage,
            config,
            d38999_asset,
            add_reference_to_stage=add_reference_to_stage,
            Gf=Gf,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
            physics_utils=physicsUtils,
        )
        add_reference_to_stage(str(robot_asset), ROBOT_ROOT_PATH)

        articulation_prim = stage.GetPrimAtPath(ARTICULATION_PATH)
        base_prim = stage.GetPrimAtPath(BASE_LINK_PATH)
        fixed_prim = stage.GetPrimAtPath(
            config.asset.fixed_receptacle_prim_path
        )
        for path, prim in (
            (ARTICULATION_PATH, articulation_prim),
            (BASE_LINK_PATH, base_prim),
            (config.asset.fixed_receptacle_prim_path, fixed_prim),
        ):
            if not prim.IsValid():
                raise RuntimeError(f"required scene prim is missing: {path}")

        contact_report_body_count = 0
        for prim in stage.Traverse():
            if not _path_is_at_or_below(prim.GetPath(), ROBOT_ROOT_PATH):
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report.CreateThresholdAttr().Set(0.0)
            contact_report_body_count += 1
        if contact_report_body_count < 17:
            raise RuntimeError(
                "robot rigid-body contact reporting is incomplete: "
                f"{contact_report_body_count}"
            )

        if arguments.gui:
            from pxr import UsdLux

            lighting_root = "/World/D38999TabletopRobotGuiLighting"
            UsdGeom.Xform.Define(stage, lighting_root)
            dome = UsdLux.DomeLight.Define(stage, lighting_root + "/Fill")
            dome.CreateIntensityAttr(config.render.dome_light_intensity)
            dome.CreateColorAttr(
                Gf.Vec3f(*config.render.dome_light_color_rgb)
            )
            key = UsdLux.DistantLight.Define(stage, lighting_root + "/Key")
            key.CreateIntensityAttr(config.render.key_light_intensity)
            key.CreateAngleAttr(2.0)
            key.CreateColorAttr(Gf.Vec3f(*config.render.key_light_color_rgb))
            UsdGeom.Xformable(key).AddRotateXYZOp().Set(
                Gf.Vec3f(*config.render.key_light_rotation_degrees_xyz)
            )
            ViewportManager.set_camera_view(
                camera="/OmniverseKit_Persp",
                eye=np.asarray(CAMERA_EYE_M, dtype=np.float64),
                target=np.asarray(CAMERA_TARGET_M, dtype=np.float64),
            )
            metrics["gui_camera_eye_m"] = list(CAMERA_EYE_M)
            metrics["gui_camera_target_m"] = list(CAMERA_TARGET_M)
            simulation_app.update()

        robot = world.scene.add(
            SingleArticulation(
                prim_path=ARTICULATION_PATH,
                name="d38999_tabletop_home_handarm",
            )
        )
        body = world.scene.add(
            SingleRigidPrim(
                prim_path=config.asset.body_prim_path,
                name="d38999_robot_scene_loose_plug_body",
            )
        )
        nut = world.scene.add(
            SingleRigidPrim(
                prim_path=config.asset.nut_prim_path,
                name="d38999_robot_scene_coupling_nut",
            )
        )

        # All scene-object poses are authored before this reset.  The two
        # explicit writes below initialize only robot joint state to Home.
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError(
                "robot articulation handles were not initialized"
            )

        dof_names = tuple(robot.dof_names)
        active_names, active_indices_tuple, home_tuple = _home_control_spec(
            dof_names
        )
        active_indices = np.asarray(active_indices_tuple, dtype=np.int32)
        home_targets = np.asarray(home_tuple, dtype=np.float32)
        all_zero_home = np.zeros(robot.num_dof, dtype=np.float32)
        robot.set_joint_positions(all_zero_home)
        robot.set_joint_velocities(all_zero_home)

        controller = robot.get_articulation_controller()
        kps = np.zeros(robot.num_dof, dtype=np.float32)
        kds = np.zeros(robot.num_dof, dtype=np.float32)
        name_to_index = {
            name: index for index, name in enumerate(dof_names)
        }
        for name in ARM_JOINT_NAMES:
            kps[name_to_index[name]] = ARM_STIFFNESS
            kds[name_to_index[name]] = ARM_DAMPING
        for name in ACTIVE_HAND_JOINT_NAMES:
            kps[name_to_index[name]] = HAND_STIFFNESS
            kds[name_to_index[name]] = HAND_DAMPING
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        hold_action = ArticulationAction(
            joint_positions=home_targets,
            joint_indices=active_indices,
        )
        robot.apply_action(hold_action)
        world.get_physics_context().set_gravity(config.physics.gravity_m_s2)

        initial_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        initial_velocities = np.asarray(
            robot.get_joint_velocities(), dtype=np.float64
        )
        initial_body_position, initial_body_orientation = body.get_world_pose()
        initial_nut_position, initial_nut_orientation = nut.get_world_pose()
        initial_center = (
            2.0 * np.asarray(initial_body_position, dtype=np.float64)
            + np.asarray(initial_nut_position, dtype=np.float64)
        ) / 3.0
        fixed_initial_position, fixed_initial_orientation = _gf_world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )
        base_initial_position, base_initial_orientation = _gf_world_pose(
            Gf, Usd, UsdGeom, base_prim
        )
        initial_bound_min, initial_bound_max = _world_aligned_bounds(
            Usd, UsdGeom, articulation_prim
        )

        finite_throughout = bool(
            np.all(np.isfinite(initial_positions))
            and np.all(np.isfinite(initial_velocities))
        )
        maximum_joint_limit_violation = 0.0
        maximum_joint_speed = 0.0
        maximum_home_error = 0.0
        maximum_base_translation_drift = 0.0
        maximum_base_rotation_drift = 0.0
        contact_counts = {"table": 0, "fixture": 0, "d38999": 0}
        contact_header_counts = {
            "table": 0,
            "fixture": 0,
            "d38999": 0,
        }
        maximum_transient_penetration = 0.0
        maximum_d38999_xy_drift = 0.0
        maximum_d38999_axis_tilt = 0.0
        maximum_fixed_translation_drift = 0.0
        maximum_fixed_rotation_drift = 0.0
        maximum_tail_displacement = 0.0
        maximum_tail_linear_speed = 0.0
        maximum_tail_angular_speed = 0.0
        tail_start_center = None
        dof_properties = robot.dof_properties

        for step_index in range(config.physics.settle_steps):
            robot.apply_action(hold_action)
            world.step(render=arguments.gui)

            positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            body_position, body_orientation = body.get_world_pose()
            nut_position, nut_orientation = nut.get_world_pose()
            body_position = np.asarray(body_position, dtype=np.float64)
            nut_position = np.asarray(nut_position, dtype=np.float64)
            body_orientation = np.asarray(
                body_orientation, dtype=np.float64
            )
            nut_orientation = np.asarray(nut_orientation, dtype=np.float64)
            body_linear = np.asarray(
                body.get_linear_velocity(), dtype=np.float64
            )
            nut_linear = np.asarray(
                nut.get_linear_velocity(), dtype=np.float64
            )
            body_angular = np.asarray(
                body.get_angular_velocity(), dtype=np.float64
            )
            nut_angular = np.asarray(
                nut.get_angular_velocity(), dtype=np.float64
            )
            current_center = (2.0 * body_position + nut_position) / 3.0
            base_position, base_orientation = _gf_world_pose(
                Gf, Usd, UsdGeom, base_prim
            )
            fixed_position, fixed_orientation = _gf_world_pose(
                Gf, Usd, UsdGeom, fixed_prim
            )
            base_position_array = np.asarray(
                base_position, dtype=np.float64
            )
            fixed_position_array = np.asarray(
                fixed_position, dtype=np.float64
            )

            sample = np.concatenate(
                (
                    positions,
                    velocities,
                    body_position,
                    body_orientation,
                    nut_position,
                    nut_orientation,
                    body_linear,
                    nut_linear,
                    body_angular,
                    nut_angular,
                    base_position_array,
                    fixed_position_array,
                )
            )
            finite_throughout = bool(
                finite_throughout
                and np.all(np.isfinite(sample))
                and _gf_quaternion_finite(base_orientation)
                and _gf_quaternion_finite(fixed_orientation)
            )

            maximum_joint_speed = max(
                maximum_joint_speed,
                float(np.max(np.abs(velocities))),
            )
            maximum_home_error = max(
                maximum_home_error,
                float(np.max(np.abs(positions[active_indices]))),
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

            maximum_base_translation_drift = max(
                maximum_base_translation_drift,
                float(
                    np.linalg.norm(
                        base_position_array
                        - np.asarray(
                            base_initial_position, dtype=np.float64
                        )
                    )
                ),
            )
            maximum_base_rotation_drift = max(
                maximum_base_rotation_drift,
                _gf_quaternion_error_radians(
                    base_initial_orientation, base_orientation
                ),
            )
            maximum_fixed_translation_drift = max(
                maximum_fixed_translation_drift,
                float(
                    np.linalg.norm(
                        fixed_position_array
                        - np.asarray(
                            fixed_initial_position, dtype=np.float64
                        )
                    )
                ),
            )
            maximum_fixed_rotation_drift = max(
                maximum_fixed_rotation_drift,
                _gf_quaternion_error_radians(
                    fixed_initial_orientation, fixed_orientation
                ),
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
                category = _classify_robot_external_contact(
                    paths,
                    ROBOT_ROOT_PATH,
                    config.table.prim_path,
                    config.fixed_endpoint.fixture_prim_path,
                    config.asset.model_root_prim_path,
                )
                if category is not None:
                    contact_header_counts[category] += 1
                    contact_counts[category] += int(header.num_contact_data)

            current_bottom = min(
                float(body_position[2])
                + config.loose_endpoint.body_bottom_offset_m,
                float(nut_position[2])
                + config.loose_endpoint.nut_bottom_offset_m,
            )
            maximum_transient_penetration = max(
                maximum_transient_penetration,
                config.table.top_z_m - current_bottom,
            )
            maximum_d38999_xy_drift = max(
                maximum_d38999_xy_drift,
                float(
                    np.linalg.norm(
                        current_center[:2] - initial_center[:2]
                    )
                ),
            )
            maximum_d38999_axis_tilt = max(
                maximum_d38999_axis_tilt,
                _tilt_from_world_z(body_orientation),
                _tilt_from_world_z(nut_orientation),
            )

            if step_index == (
                config.physics.settle_steps
                - config.physics.tail_steps
                - 1
            ):
                tail_start_center = current_center.copy()
            if tail_start_center is not None:
                maximum_tail_displacement = max(
                    maximum_tail_displacement,
                    float(np.linalg.norm(current_center - tail_start_center)),
                )
                maximum_tail_linear_speed = max(
                    maximum_tail_linear_speed,
                    float(np.linalg.norm(body_linear)),
                    float(np.linalg.norm(nut_linear)),
                )
                maximum_tail_angular_speed = max(
                    maximum_tail_angular_speed,
                    float(np.linalg.norm(body_angular)),
                    float(np.linalg.norm(nut_angular)),
                )

        if tail_start_center is None:
            raise RuntimeError("D38999 tail window was not sampled")

        final_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        final_velocities = np.asarray(
            robot.get_joint_velocities(), dtype=np.float64
        )
        final_body_position, final_body_orientation = body.get_world_pose()
        final_nut_position, final_nut_orientation = nut.get_world_pose()
        final_body_position = np.asarray(
            final_body_position, dtype=np.float64
        )
        final_nut_position = np.asarray(final_nut_position, dtype=np.float64)
        final_center = (
            2.0 * final_body_position + final_nut_position
        ) / 3.0
        final_bound_min, final_bound_max = _world_aligned_bounds(
            Usd, UsdGeom, articulation_prim
        )

        final_home_error = float(
            np.max(np.abs(final_positions[active_indices]))
        )
        final_joint_speed = float(np.max(np.abs(final_velocities)))
        joint_limits_safe = bool(
            maximum_joint_limit_violation
            <= MAXIMUM_JOINT_LIMIT_VIOLATION_RAD
        )
        joints_stable_at_home = bool(
            joint_limits_safe
            and maximum_home_error <= MAXIMUM_OBSERVED_HOME_ERROR_RAD
            and maximum_joint_speed <= MAXIMUM_OBSERVED_JOINT_SPEED_RAD_S
            and final_home_error <= MAXIMUM_FINAL_HOME_ERROR_RAD
            and final_joint_speed <= MAXIMUM_FINAL_JOINT_SPEED_RAD_S
        )
        robot_base_immobile = bool(
            maximum_base_translation_drift
            <= MAXIMUM_BASE_TRANSLATION_DRIFT_M
            and maximum_base_rotation_drift
            <= MAXIMUM_BASE_ROTATION_DRIFT_RAD
        )
        minimum_robot_table_clearance = min(
            table_front_x - float(initial_bound_max[0]),
            table_front_x - float(final_bound_max[0]),
        )
        contact_free = bool(
            all(count == 0 for count in contact_counts.values())
        )
        robot_clear_of_table = bool(
            contact_counts["table"] == 0
            and minimum_robot_table_clearance
            >= ROBOT_TABLE_MINIMUM_CLEARANCE_M
        )

        vertical_drop = float(initial_center[2] - final_center[2])
        final_bottom = min(
            float(final_body_position[2])
            + config.loose_endpoint.body_bottom_offset_m,
            float(final_nut_position[2])
            + config.loose_endpoint.nut_bottom_offset_m,
        )
        final_surface_error = final_bottom - config.table.top_z_m
        dropped_under_gravity = bool(
            config.physics.minimum_vertical_drop_m
            <= vertical_drop
            <= config.physics.maximum_vertical_drop_m
        )
        penetration_safe = bool(
            maximum_transient_penetration
            <= config.physics.maximum_transient_table_penetration_m
        )
        on_table_surface = bool(
            -config.physics.maximum_transient_table_penetration_m
            <= final_surface_error
            <= config.physics.maximum_final_surface_gap_m
        )
        d38999_drift_safe = bool(
            maximum_d38999_xy_drift
            <= config.physics.maximum_xy_drift_m
        )
        d38999_tilt_safe = bool(
            maximum_d38999_axis_tilt
            <= config.physics.maximum_upright_axis_tilt_rad
        )
        d38999_tail_stable = bool(
            maximum_tail_displacement
            <= config.physics.maximum_tail_displacement_m
            and maximum_tail_linear_speed
            <= config.physics.maximum_tail_linear_speed_m_s
            and maximum_tail_angular_speed
            <= config.physics.maximum_tail_angular_speed_rad_s
        )
        fixed_endpoint_immobile = bool(
            maximum_fixed_translation_drift
            <= config.physics.maximum_fixed_translation_drift_m
            and maximum_fixed_rotation_drift
            <= config.physics.maximum_fixed_rotation_drift_rad
        )
        bounds_finite = all(
            math.isfinite(float(value))
            for value in (
                *initial_bound_min,
                *initial_bound_max,
                *final_bound_min,
                *final_bound_max,
            )
        )
        finite_final = bool(
            np.all(np.isfinite(final_positions))
            and np.all(np.isfinite(final_velocities))
            and np.all(np.isfinite(final_center))
            and bounds_finite
        )

        passed = bool(
            finite_throughout
            and finite_final
            and joints_stable_at_home
            and robot_base_immobile
            and robot_clear_of_table
            and contact_free
            and dropped_under_gravity
            and penetration_safe
            and on_table_surface
            and d38999_drift_safe
            and d38999_tilt_safe
            and d38999_tail_stable
            and fixed_endpoint_immobile
            and metrics["object_pose_writes_after_start"] == 0
        )
        metrics.update(
            {
                "active_home_joint_names": list(active_names),
                "contact_counts": contact_counts,
                "contact_free": contact_free,
                "contact_header_counts": contact_header_counts,
                "contact_report_robot_body_count": (
                    contact_report_body_count
                ),
                "d38999_drift_safe": d38999_drift_safe,
                "d38999_tail_stable": d38999_tail_stable,
                "d38999_tilt_safe": d38999_tilt_safe,
                "dof_count": robot.num_dof,
                "dof_names": list(dof_names),
                "dropped_under_gravity": dropped_under_gravity,
                "final_home_error_rad": final_home_error,
                "final_joint_speed_rad_s": final_joint_speed,
                "final_surface_error_m": final_surface_error,
                "finite_final": finite_final,
                "finite_throughout": finite_throughout,
                "fixed_endpoint_immobile": fixed_endpoint_immobile,
                "joint_limit_violation_rad": max(
                    0.0, maximum_joint_limit_violation
                ),
                "joint_limits_safe": joint_limits_safe,
                "joints_stable_at_home": joints_stable_at_home,
                "maximum_base_rotation_drift_rad": (
                    maximum_base_rotation_drift
                ),
                "maximum_base_translation_drift_m": (
                    maximum_base_translation_drift
                ),
                "maximum_d38999_axis_tilt_rad": (
                    maximum_d38999_axis_tilt
                ),
                "maximum_d38999_xy_drift_m": maximum_d38999_xy_drift,
                "maximum_fixed_rotation_drift_rad": (
                    maximum_fixed_rotation_drift
                ),
                "maximum_fixed_translation_drift_m": (
                    maximum_fixed_translation_drift
                ),
                "maximum_home_error_rad": maximum_home_error,
                "maximum_joint_speed_rad_s": maximum_joint_speed,
                "maximum_tail_angular_speed_rad_s": (
                    maximum_tail_angular_speed
                ),
                "maximum_tail_displacement_m": maximum_tail_displacement,
                "maximum_tail_linear_speed_m_s": (
                    maximum_tail_linear_speed
                ),
                "maximum_transient_table_penetration_m": max(
                    0.0, maximum_transient_penetration
                ),
                "minimum_robot_table_clearance_m": (
                    minimum_robot_table_clearance
                ),
                "on_table_surface": on_table_surface,
                "passed": passed,
                "pd_gains": {
                    "arm_damping": ARM_DAMPING,
                    "arm_stiffness": ARM_STIFFNESS,
                    "hand_damping": HAND_DAMPING,
                    "hand_stiffness": HAND_STIFFNESS,
                },
                "penetration_safe": penetration_safe,
                "robot_base_immobile": robot_base_immobile,
                "robot_clear_of_table": robot_clear_of_table,
                "robot_pd_hold_duration_s": (
                    config.physics.settle_duration_s
                ),
                "settle_steps": config.physics.settle_steps,
                "vertical_drop_m": vertical_drop,
            }
        )
        json.dumps(metrics, allow_nan=False, sort_keys=True)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print(
            "ISAAC D38999 TABLETOP ROBOT HOME V1 "
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
        print("ISAAC D38999 TABLETOP ROBOT HOME V1 FAILED", flush=True)
    finally:
        if arguments.keep_open and arguments.gui:
            print(
                "ISAAC D38999 TABLETOP ROBOT HOME V1 GUI REMAINS OPEN; "
                "close the window to exit",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
