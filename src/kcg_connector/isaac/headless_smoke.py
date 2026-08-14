#!/usr/bin/env python3

"""Launch Isaac Sim headless and verify a rigid body settles on a plane."""

import argparse
import json
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=240)
    arguments = parser.parse_args()
    if arguments.steps <= 0:
        raise ValueError("--steps must be positive")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )

    passed = False
    try:
        import numpy as np

        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid

        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0)
        world.scene.add_default_ground_plane()
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/HeadlessSmokeCube",
                name="headless_smoke_cube",
                position=np.array([0.0, 0.0, 0.5]),
                scale=np.array([0.1, 0.1, 0.1]),
                color=np.array([0.8, 0.2, 0.1]),
                mass=0.1,
            )
        )
        world.reset()
        settling_window = min(60, max(1, arguments.steps // 4))
        window_start_position = None
        for step_index in range(arguments.steps):
            world.step(render=False)
            if step_index == arguments.steps - settling_window - 1:
                window_start_position, _ = cube.get_world_pose()

        position, _ = cube.get_world_pose()
        linear_velocity = cube.get_linear_velocity()
        if window_start_position is None:
            window_start_position = position.copy()
        measured_linear_speed = float(
            np.linalg.norm(position - window_start_position)
            / (settling_window * world.get_physics_dt())
        )
        finite = all(
            math.isfinite(float(value))
            for value in [*position, *linear_velocity, measured_linear_speed]
        )
        settled = (
            finite
            and 0.045 <= float(position[2]) <= 0.055
            and measured_linear_speed <= 0.002
        )
        metrics = {
            "finite": finite,
            "position": [round(float(value), 6) for value in position],
            "reported_linear_speed": round(
                float(np.linalg.norm(linear_velocity)), 6
            ),
            "measured_linear_speed": round(measured_linear_speed, 6),
            "settled": settled,
            "settling_window_steps": settling_window,
            "steps": arguments.steps,
        }
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if settled:
            print("ISAAC SIM HEADLESS PHYSICS PASSED", flush=True)
            passed = True
        else:
            print("ISAAC SIM HEADLESS PHYSICS FAILED", flush=True)
    finally:
        # Isaac Sim's fast-shutdown path terminates the process, so propagate the
        # test result through close() instead of raising before cleanup.
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
