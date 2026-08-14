#!/usr/bin/env python3

"""Build and validate the standalone physical connector tabletop v1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback


def _quaternion_error_radians(first, second):
    relative = first.GetInverse() * second
    real = max(-1.0, min(1.0, abs(float(relative.GetReal()))))
    return 2.0 * math.acos(real)


def _world_pose(Gf, Usd, UsdGeom, prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform(matrix)
    return transform.GetTranslation(), transform.GetRotation().GetQuat()


def _array_quaternion_error_radians(first, second):
    import numpy as np

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first_array))
    second_norm = float(np.linalg.norm(second_array))
    if first_norm <= 0.0 or second_norm <= 0.0:
        return float("inf")
    cosine = abs(
        float(np.dot(first_array, second_array))
        / (first_norm * second_norm)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, cosine)))


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


def _axis_error_radians(first, second):
    import numpy as np

    first_axis = _quaternion_z_axis(first)
    second_axis = _quaternion_z_axis(second)
    if not (
        np.all(np.isfinite(first_axis))
        and np.all(np.isfinite(second_axis))
    ):
        return float("inf")
    cosine = float(np.dot(first_axis, second_axis))
    return math.acos(max(-1.0, min(1.0, cosine)))


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


def main():
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
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
            / "src/kcg_connector/config/connector_tabletop_scene_v1.yaml"
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="open the Isaac Sim window and render the two-second settle",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="keep the final GUI frame open until the window is closed",
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
        "gui": arguments.gui,
        "keep_open": arguments.keep_open,
        "object_pose_writes_after_start": 0,
        "passed": False,
        "scene": "kcg_connector_tabletop_scene_v1",
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

        from kcg_connector.isaac_tabletop_scene import (
            author_isaac_tabletop_scene,
            load_connector_tabletop_scene,
        )

        config_path = Path(arguments.config).expanduser().resolve()
        connector_asset = Path(
            arguments.connector_asset
        ).expanduser().resolve()
        for path in (config_path, connector_asset):
            if not path.is_file():
                raise FileNotFoundError(path)
        config = load_connector_tabletop_scene(config_path)

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / config.physics.rate_hz,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        stage = get_current_stage()
        authored = author_isaac_tabletop_scene(
            stage,
            config,
            connector_asset,
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
                prim_path=config.loose_endpoint.body_prim_path,
                name="tabletop_loose_plug_body",
            )
        )
        nut = world.scene.add(
            SingleRigidPrim(
                prim_path=config.loose_endpoint.nut_prim_path,
                name="tabletop_loose_coupling_nut",
            )
        )
        world.reset()
        world.get_physics_context().set_gravity(
            config.physics.gravity_m_s2
        )

        initial_body_position, initial_body_orientation = body.get_world_pose()
        initial_nut_position, initial_nut_orientation = nut.get_world_pose()
        receptacle_prim = stage.GetPrimAtPath(
            config.fixed_endpoint.receptacle_prim_path
        )
        fixed_initial_position, fixed_initial_orientation = _world_pose(
            Gf, Usd, UsdGeom, receptacle_prim
        )
        initial_center = 0.6 * np.asarray(
            initial_body_position, dtype=np.float64
        ) + 0.4 * np.asarray(initial_nut_position, dtype=np.float64)
        tail_start_center = None
        maximum_tail_displacement = 0.0
        maximum_tail_linear_speed = 0.0
        maximum_tail_angular_speed = 0.0

        finite_throughout = True
        maximum_linear_speed = 0.0
        maximum_angular_speed = 0.0
        for step_index in range(config.physics.settle_steps):
            world.step(render=arguments.gui)
            body_position, body_orientation = body.get_world_pose()
            nut_position, nut_orientation = nut.get_world_pose()
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
            values = np.concatenate(
                (
                    np.asarray(body_position, dtype=np.float64),
                    np.asarray(body_orientation, dtype=np.float64),
                    np.asarray(nut_position, dtype=np.float64),
                    np.asarray(nut_orientation, dtype=np.float64),
                    body_linear,
                    nut_linear,
                    body_angular,
                    nut_angular,
                )
            )
            finite_throughout = bool(
                finite_throughout and np.all(np.isfinite(values))
            )
            maximum_linear_speed = max(
                maximum_linear_speed,
                float(np.linalg.norm(body_linear)),
                float(np.linalg.norm(nut_linear)),
            )
            maximum_angular_speed = max(
                maximum_angular_speed,
                float(np.linalg.norm(body_angular)),
                float(np.linalg.norm(nut_angular)),
            )
            if step_index == (
                config.physics.settle_steps
                - config.physics.tail_steps
                - 1
            ):
                tail_start_center = 0.6 * np.asarray(
                    body_position, dtype=np.float64
                ) + 0.4 * np.asarray(nut_position, dtype=np.float64)
            if tail_start_center is not None:
                current_center = 0.6 * np.asarray(
                    body_position, dtype=np.float64
                ) + 0.4 * np.asarray(nut_position, dtype=np.float64)
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

        final_body_position, final_body_orientation = body.get_world_pose()
        final_nut_position, final_nut_orientation = nut.get_world_pose()
        final_body_linear = np.asarray(
            body.get_linear_velocity(), dtype=np.float64
        )
        final_nut_linear = np.asarray(
            nut.get_linear_velocity(), dtype=np.float64
        )
        final_body_angular = np.asarray(
            body.get_angular_velocity(), dtype=np.float64
        )
        final_nut_angular = np.asarray(
            nut.get_angular_velocity(), dtype=np.float64
        )
        fixed_final_position, fixed_final_orientation = _world_pose(
            Gf, Usd, UsdGeom, receptacle_prim
        )

        final_center = 0.6 * np.asarray(
            final_body_position, dtype=np.float64
        ) + 0.4 * np.asarray(final_nut_position, dtype=np.float64)
        if tail_start_center is None:
            raise RuntimeError("tabletop tail window was not sampled")
        vertical_drop = float(initial_center[2] - final_center[2])
        xy_drift = float(
            np.linalg.norm(final_center[:2] - initial_center[:2])
        )
        tail_displacement = float(
            np.linalg.norm(final_center - tail_start_center)
        )
        final_bottom = float(
            min(
                float(final_body_position[2])
                - config.loose_endpoint.body_bottom_offset_m,
                float(final_nut_position[2])
                - config.loose_endpoint.nut_bottom_offset_m,
            )
        )
        surface_error = final_bottom - config.table.top_z_m
        final_linear_speed = max(
            float(np.linalg.norm(final_body_linear)),
            float(np.linalg.norm(final_nut_linear)),
        )
        final_angular_speed = max(
            float(np.linalg.norm(final_body_angular)),
            float(np.linalg.norm(final_nut_angular)),
        )
        fixed_translation_drift = float(
            np.linalg.norm(
                np.asarray(fixed_final_position, dtype=np.float64)
                - np.asarray(fixed_initial_position, dtype=np.float64)
            )
        )
        fixed_rotation_drift = _quaternion_error_radians(
            fixed_initial_orientation, fixed_final_orientation
        )
        body_nut_orientation_change = max(
            _array_quaternion_error_radians(
                initial_body_orientation,
                final_body_orientation,
            ),
            _array_quaternion_error_radians(
                initial_nut_orientation,
                final_nut_orientation,
            ),
        )
        maximum_upright_axis_tilt = max(
            _axis_error_radians(
                initial_body_orientation, final_body_orientation
            ),
            _axis_error_radians(
                initial_nut_orientation, final_nut_orientation
            ),
        )
        finite_throughout = bool(
            finite_throughout
            and np.all(np.isfinite(final_body_orientation))
            and np.all(np.isfinite(final_nut_orientation))
            and _gf_quaternion_finite(fixed_initial_orientation)
            and _gf_quaternion_finite(fixed_final_orientation)
            and math.isfinite(body_nut_orientation_change)
            and math.isfinite(maximum_upright_axis_tilt)
        )
        on_table = bool(
            -config.physics.maximum_table_penetration_m
            <= surface_error
            <= config.physics.maximum_surface_gap_m
        )
        dropped = bool(
            config.physics.minimum_vertical_drop_m
            <= vertical_drop
            <= config.physics.maximum_vertical_drop_m
        )
        settled = bool(
            maximum_tail_displacement
            <= config.physics.maximum_tail_displacement_m
            and maximum_tail_linear_speed
            <= config.physics.maximum_final_linear_speed_m_s
            and maximum_tail_angular_speed
            <= config.physics.maximum_final_angular_speed_rad_s
        )
        fixed_endpoint_immobile = bool(
            fixed_translation_drift
            <= config.physics.maximum_fixed_translation_drift_m
            and fixed_rotation_drift
            <= config.physics.maximum_fixed_rotation_drift_rad
        )
        stayed_in_pickup_region = bool(
            xy_drift <= config.physics.maximum_pickup_xy_drift_m
        )
        remained_upright = bool(
            maximum_upright_axis_tilt
            <= config.physics.maximum_upright_axis_tilt_rad
        )
        passed = bool(
            finite_throughout
            and dropped
            and on_table
            and settled
            and fixed_endpoint_immobile
            and stayed_in_pickup_region
            and remained_upright
        )
        metrics.update(
            {
                "body_nut_max_orientation_change_rad": (
                    body_nut_orientation_change
                ),
                "dropped_under_gravity": dropped,
                "final_angular_speed_rad_s": final_angular_speed,
                "final_linear_speed_m_s": final_linear_speed,
                "finite_throughout": finite_throughout,
                "fixed_endpoint_immobile": fixed_endpoint_immobile,
                "fixed_rotation_drift_rad": fixed_rotation_drift,
                "fixed_translation_drift_m": fixed_translation_drift,
                "maximum_observed_angular_speed_rad_s": (
                    maximum_angular_speed
                ),
                "maximum_observed_linear_speed_m_s": maximum_linear_speed,
                "maximum_tail_angular_speed_rad_s": (
                    maximum_tail_angular_speed
                ),
                "maximum_tail_displacement_m": maximum_tail_displacement,
                "maximum_tail_linear_speed_m_s": maximum_tail_linear_speed,
                "maximum_upright_axis_tilt_rad": (
                    maximum_upright_axis_tilt
                ),
                "on_table": on_table,
                "passed": passed,
                "settle_duration_s": config.physics.settle_duration_s,
                "settle_steps": config.physics.settle_steps,
                "settled": settled,
                "remained_upright": remained_upright,
                "surface_error_m": surface_error,
                "tail_displacement_m": tail_displacement,
                "vertical_drop_m": vertical_drop,
                "xy_drift_m": xy_drift,
            }
        )
        json.dumps(metrics, allow_nan=False, sort_keys=True)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print(
            "ISAAC CONNECTOR TABLETOP V1 "
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
        print("ISAAC CONNECTOR TABLETOP V1 FAILED", flush=True)
    finally:
        if arguments.keep_open and arguments.gui:
            print(
                "ISAAC CONNECTOR TABLETOP V1 GUI REMAINS OPEN; "
                "close the window to exit",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
