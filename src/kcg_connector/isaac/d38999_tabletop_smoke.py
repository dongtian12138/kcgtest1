#!/usr/bin/env python3

"""Run the independent 240 Hz D38999 tabletop physical settle gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback


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


def _arguments(repository):
    parser = argparse.ArgumentParser(
        description="Validate D38999 loose-plug gravity settle on tabletop"
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
        help="render the D38999 pair, collision table, and fixture",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="keep the final GUI frame open until the window is closed",
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
        "object_pose_writes_after_start": 0,
        "passed": False,
        "scene": "kcg_d38999_tabletop_scene_v1",
    }
    try:
        import numpy as np

        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
        )
        from omni.physx.scripts import physicsUtils
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        from kcg_connector.d38999_tabletop_scene import (
            author_d38999_tabletop_scene,
            load_d38999_tabletop_scene,
            verify_d38999_tabletop_asset,
        )
        from kcg_connector.connector_pose import (
            load_connector_pose_contract,
            pair_connector_pose_observations,
        )
        from kcg_connector.sim_pose_provider import (
            make_sim_ground_truth_observation,
        )

        config_path = Path(arguments.config).expanduser().resolve()
        config = load_d38999_tabletop_scene(config_path)
        asset_path = verify_d38999_tabletop_asset(config, repository)

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / config.physics.rate_hz,
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
        )
        metrics.update(authored)

        if arguments.gui:
            from isaacsim.core.rendering_manager import ViewportManager
            from pxr import UsdLux

            lighting_root = config.world.root_prim_path + "/GuiLighting"
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
                eye=np.asarray(config.render.camera_eye_m),
                target=np.asarray(config.render.camera_target_m),
            )
            simulation_app.update()

        body = world.scene.add(
            SingleRigidPrim(
                prim_path=config.asset.body_prim_path,
                name="d38999_loose_plug_body",
            )
        )
        nut = world.scene.add(
            SingleRigidPrim(
                prim_path=config.asset.nut_prim_path,
                name="d38999_loose_coupling_nut",
            )
        )

        # Every pose was authored above.  Physics starts here, after which the
        # smoke only steps and reads state; it never calls a pose setter.
        world.reset()
        world.get_physics_context().set_gravity(config.physics.gravity_m_s2)

        initial_body_position, initial_body_orientation = body.get_world_pose()
        initial_nut_position, initial_nut_orientation = nut.get_world_pose()
        fixed_prim = stage.GetPrimAtPath(
            config.asset.fixed_receptacle_prim_path
        )
        fixed_initial_position, fixed_initial_orientation = _gf_world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )
        initial_center = (
            2.0 * np.asarray(initial_body_position, dtype=np.float64)
            + np.asarray(initial_nut_position, dtype=np.float64)
        ) / 3.0

        finite_throughout = True
        maximum_transient_penetration = 0.0
        maximum_xy_drift = 0.0
        maximum_axis_tilt = 0.0
        maximum_fixed_translation_drift = 0.0
        maximum_fixed_rotation_drift = 0.0
        maximum_observed_linear_speed = 0.0
        maximum_observed_angular_speed = 0.0
        maximum_tail_linear_speed = 0.0
        maximum_tail_angular_speed = 0.0
        maximum_tail_displacement = 0.0
        tail_start_center = None

        for step_index in range(config.physics.settle_steps):
            world.step(render=arguments.gui)
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
            fixed_position, fixed_orientation = _gf_world_pose(
                Gf, Usd, UsdGeom, fixed_prim
            )
            fixed_position = np.asarray(fixed_position, dtype=np.float64)

            values = np.concatenate(
                (
                    body_position,
                    body_orientation,
                    nut_position,
                    nut_orientation,
                    body_linear,
                    nut_linear,
                    body_angular,
                    nut_angular,
                    fixed_position,
                )
            )
            finite_throughout = bool(
                finite_throughout
                and np.all(np.isfinite(values))
                and _gf_quaternion_finite(fixed_orientation)
            )

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
            maximum_xy_drift = max(
                maximum_xy_drift,
                float(
                    np.linalg.norm(
                        current_center[:2] - initial_center[:2]
                    )
                ),
            )
            current_axis_tilt = max(
                _tilt_from_world_z(body_orientation),
                _tilt_from_world_z(nut_orientation),
            )
            maximum_axis_tilt = max(
                maximum_axis_tilt, current_axis_tilt
            )
            fixed_translation_drift = float(
                np.linalg.norm(
                    fixed_position
                    - np.asarray(
                        fixed_initial_position, dtype=np.float64
                    )
                )
            )
            fixed_rotation_drift = _gf_quaternion_error_radians(
                fixed_initial_orientation, fixed_orientation
            )
            maximum_fixed_translation_drift = max(
                maximum_fixed_translation_drift,
                fixed_translation_drift,
            )
            maximum_fixed_rotation_drift = max(
                maximum_fixed_rotation_drift, fixed_rotation_drift
            )
            linear_speed = max(
                float(np.linalg.norm(body_linear)),
                float(np.linalg.norm(nut_linear)),
            )
            angular_speed = max(
                float(np.linalg.norm(body_angular)),
                float(np.linalg.norm(nut_angular)),
            )
            maximum_observed_linear_speed = max(
                maximum_observed_linear_speed, linear_speed
            )
            maximum_observed_angular_speed = max(
                maximum_observed_angular_speed, angular_speed
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
                    maximum_tail_linear_speed, linear_speed
                )
                maximum_tail_angular_speed = max(
                    maximum_tail_angular_speed, angular_speed
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
        if tail_start_center is None:
            raise RuntimeError("D38999 tail window was not sampled")
        final_bottom = min(
            float(final_body_position[2])
            + config.loose_endpoint.body_bottom_offset_m,
            float(final_nut_position[2])
            + config.loose_endpoint.nut_bottom_offset_m,
        )
        final_surface_error = final_bottom - config.table.top_z_m
        vertical_drop = float(initial_center[2] - final_center[2])
        final_xy_drift = float(
            np.linalg.norm(final_center[:2] - initial_center[:2])
        )
        final_tail_displacement = float(
            np.linalg.norm(final_center - tail_start_center)
        )

        fixed_final_position, fixed_final_orientation = _gf_world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )
        fixed_imaginary = fixed_final_orientation.GetImaginary()
        fixed_wxyz = (
            float(fixed_final_orientation.GetReal()),
            float(fixed_imaginary[0]),
            float(fixed_imaginary[1]),
            float(fixed_imaginary[2]),
        )
        pose_contract = load_connector_pose_contract()
        pose_timestamp_s = float(config.physics.settle_duration_s)
        loose_observation = make_sim_ground_truth_observation(
            pose_contract,
            model_id="d38999_26kj61sn_proxy_v1",
            role="loose_plug",
            timestamp_s=pose_timestamp_s,
            now_s=pose_timestamp_s,
            frame_id="world",
            position_xyz_m=final_body_position,
            quaternion_wxyz=final_body_orientation,
        )
        fixed_observation = make_sim_ground_truth_observation(
            pose_contract,
            model_id="d38999_20kj61pn_proxy_v1",
            role="fixed_receptacle",
            timestamp_s=pose_timestamp_s,
            now_s=pose_timestamp_s,
            frame_id="world",
            position_xyz_m=fixed_final_position,
            quaternion_wxyz=fixed_wxyz,
        )
        pose_pair = pair_connector_pose_observations(
            loose_observation,
            fixed_observation,
            pose_contract,
            now_s=pose_timestamp_s,
        )
        pose_contract_pair_valid = bool(
            pose_pair.loose_plug.model_id
            == "d38999_26kj61sn_proxy_v1"
            and pose_pair.fixed_receptacle.model_id
            == "d38999_20kj61pn_proxy_v1"
            and not pose_contract.object_target_transforms
        )

        dropped = bool(
            config.physics.minimum_vertical_drop_m
            <= vertical_drop
            <= config.physics.maximum_vertical_drop_m
        )
        penetration_safe = bool(
            maximum_transient_penetration
            <= config.physics.maximum_transient_table_penetration_m
        )
        on_surface = bool(
            -config.physics.maximum_transient_table_penetration_m
            <= final_surface_error
            <= config.physics.maximum_final_surface_gap_m
        )
        drift_safe = bool(
            maximum_xy_drift <= config.physics.maximum_xy_drift_m
        )
        tilt_safe = bool(
            maximum_axis_tilt
            <= config.physics.maximum_upright_axis_tilt_rad
        )
        tail_stable = bool(
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
        passed = bool(
            finite_throughout
            and dropped
            and penetration_safe
            and on_surface
            and drift_safe
            and tilt_safe
            and tail_stable
            and fixed_endpoint_immobile
            and pose_contract_pair_valid
            and metrics["object_pose_writes_after_start"] == 0
        )
        metrics.update(
            {
                "dropped_under_gravity": dropped,
                "final_surface_error_m": final_surface_error,
                "final_tail_displacement_m": final_tail_displacement,
                "final_xy_drift_m": final_xy_drift,
                "finite_throughout": finite_throughout,
                "fixed_endpoint_immobile": fixed_endpoint_immobile,
                "maximum_fixed_rotation_drift_rad": (
                    maximum_fixed_rotation_drift
                ),
                "maximum_fixed_translation_drift_m": (
                    maximum_fixed_translation_drift
                ),
                "maximum_observed_angular_speed_rad_s": (
                    maximum_observed_angular_speed
                ),
                "maximum_observed_linear_speed_m_s": (
                    maximum_observed_linear_speed
                ),
                "maximum_tail_angular_speed_rad_s": (
                    maximum_tail_angular_speed
                ),
                "maximum_tail_displacement_m": maximum_tail_displacement,
                "maximum_tail_linear_speed_m_s": maximum_tail_linear_speed,
                "maximum_transient_table_penetration_m": max(
                    0.0, maximum_transient_penetration
                ),
                "maximum_upright_axis_tilt_rad": maximum_axis_tilt,
                "maximum_xy_drift_m": maximum_xy_drift,
                "on_table_surface": on_surface,
                "passed": passed,
                "penetration_safe": penetration_safe,
                "pose_contract": {
                    "fixed_model_id": pose_pair.fixed_receptacle.model_id,
                    "frame_id": pose_pair.loose_plug.frame_id,
                    "loose_model_id": pose_pair.loose_plug.model_id,
                    "object_target_transforms_available": bool(
                        pose_contract.object_target_transforms
                    ),
                    "pair_valid": pose_contract_pair_valid,
                    "source": pose_pair.loose_plug.source.value,
                    "vision_detector_present": False,
                },
                "settle_duration_s": config.physics.settle_duration_s,
                "settle_steps": config.physics.settle_steps,
                "tail_stable": tail_stable,
                "tilt_safe": tilt_safe,
                "vertical_drop_m": vertical_drop,
                "xy_drift_safe": drift_safe,
            }
        )
        json.dumps(metrics, allow_nan=False, sort_keys=True)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print(
            "ISAAC D38999 TABLETOP V1 "
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
        print("ISAAC D38999 TABLETOP V1 FAILED", flush=True)
    finally:
        if arguments.keep_open and arguments.gui:
            print(
                "ISAAC D38999 TABLETOP V1 GUI REMAINS OPEN; "
                "close the window to exit",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
