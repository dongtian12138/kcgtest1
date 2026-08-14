#!/usr/bin/env python3

"""Exercise the imported hand-arm articulation with Isaac Sim/PhysX."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path


ARM_JOINTS = tuple(f"iiwa_joint_{index}" for index in range(1, 8))
ACTIVE_HAND_JOINTS = ("f1j1", "f1j2", "f2j1", "f3j2")
MIMIC_JOINTS = {
    "f1j3": "f1j2",
    "f2j2": "f2j1",
    "f3j1": "f1j1",
    "f3j3": "f3j2",
}
EXPECTED_DOF_NAMES = ARM_JOINTS + (
    "f1j1",
    "f1j2",
    "f1j3",
    "f2j1",
    "f2j2",
    "f3j1",
    "f3j2",
    "f3j3",
)


def _named_values(names, values):
    return {name: round(float(value), 6) for name, value in zip(names, values)}


def main():
    default_asset = (
        Path(__file__).resolve().parents[3]
        / "artifacts/kcg_connector/isaac/robot/handarm/handarm.usda"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=str(default_asset))
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--warmup-steps", type=int, default=30)
    arguments = parser.parse_args()

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
    metrics = {
        "asset": str(Path(arguments.asset).expanduser().resolve()),
        "physics_backend": "PhysX",
        "steps": arguments.steps,
        "warmup_steps": arguments.warmup_steps,
    }
    try:
        import numpy as np

        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from isaacsim.core.utils.types import ArticulationAction

        asset_path = Path(arguments.asset).expanduser().resolve()
        if not asset_path.is_file():
            raise FileNotFoundError(asset_path)
        if arguments.steps < 120:
            raise ValueError("--steps must be at least 120")
        if arguments.warmup_steps < 1:
            raise ValueError("--warmup-steps must be positive")

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 120.0,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        robot_prim_path = "/World/HandArm"
        articulation_prim_path = f"{robot_prim_path}/Geometry/world"
        add_reference_to_stage(str(asset_path), robot_prim_path)

        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(articulation_prim_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"missing articulation prim: {articulation_prim_path}")

        authored_mimic = {}
        for follower, source in MIMIC_JOINTS.items():
            joint_prim = stage.GetPrimAtPath(f"{robot_prim_path}/Physics/{follower}")
            schemas = list(joint_prim.GetAppliedSchemas())
            relation = joint_prim.GetRelationship("newton:mimicJoint")
            authored_targets = [str(path) for path in relation.GetTargets()]
            authored_mimic[follower] = {
                "expected_source": source,
                "newton_mimic_api": "NewtonMimicAPI" in schemas,
                "physx_mimic_api": any(
                    schema.startswith("PhysxMimicJointAPI:") for schema in schemas
                ),
                "relationship_targets": authored_targets,
            }
        metrics["authored_mimic"] = authored_mimic

        robot = world.scene.add(
            SingleArticulation(
                prim_path=articulation_prim_path,
                name="handarm_physics_smoke",
            )
        )
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError("articulation handles were not initialized")

        dof_names = tuple(robot.dof_names)
        name_to_index = {name: index for index, name in enumerate(dof_names)}
        missing_dofs = sorted(set(EXPECTED_DOF_NAMES) - set(dof_names))
        unexpected_dofs = sorted(set(dof_names) - set(EXPECTED_DOF_NAMES))
        dof_layout_valid = (
            robot.num_dof == len(EXPECTED_DOF_NAMES)
            and len(name_to_index) == len(dof_names)
            and not missing_dofs
            and not unexpected_dofs
        )
        metrics.update(
            {
                "articulation_initialized": robot.handles_initialized,
                "dof_count": robot.num_dof,
                "dof_names": list(dof_names),
                "dof_layout_valid": dof_layout_valid,
                "missing_dofs": missing_dofs,
                "unexpected_dofs": unexpected_dofs,
            }
        )
        if not dof_layout_valid:
            raise RuntimeError("unexpected articulation DOF layout")

        initial_positions = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        initial_velocities = np.asarray(robot.get_joint_velocities(), dtype=np.float64)
        if not np.all(np.isfinite(initial_positions)) or not np.all(
            np.isfinite(initial_velocities)
        ):
            raise RuntimeError("initial articulation state is not finite")

        controller = robot.get_articulation_controller()
        imported_kps, imported_kds = controller.get_gains()
        imported_kps = np.asarray(imported_kps, dtype=np.float64)
        imported_kds = np.asarray(imported_kds, dtype=np.float64)
        metrics["imported_drive_gains"] = {
            "damping": _named_values(dof_names, imported_kds),
            "stiffness": _named_values(dof_names, imported_kps),
        }

        # The converter preserves effort and damping but this asset has no
        # position-drive stiffness.  Add conservative, in-memory smoke-test
        # gains only to the joints that are truly commanded.  Mimic followers
        # stay completely passive so their motion cannot be faked by targets.
        smoke_kps = np.zeros(robot.num_dof, dtype=np.float32)
        smoke_kds = np.zeros(robot.num_dof, dtype=np.float32)
        for name in ARM_JOINTS:
            smoke_kps[name_to_index[name]] = 400.0
            smoke_kds[name_to_index[name]] = 40.0
        for name in ACTIVE_HAND_JOINTS:
            smoke_kps[name_to_index[name]] = 25.0
            smoke_kds[name_to_index[name]] = 2.0
        controller.set_gains(kps=smoke_kps, kds=smoke_kds, save_to_usd=False)

        controlled_names = ARM_JOINTS + ACTIVE_HAND_JOINTS
        controlled_indices = np.asarray(
            [name_to_index[name] for name in controlled_names], dtype=np.int32
        )
        hold_positions = initial_positions[controlled_indices].astype(np.float32)
        hold_action = ArticulationAction(
            joint_positions=hold_positions,
            joint_indices=controlled_indices,
        )
        robot.apply_action(hold_action)
        for _ in range(arguments.warmup_steps):
            world.step(render=False)

        baseline_positions = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        baseline_velocities = np.asarray(robot.get_joint_velocities(), dtype=np.float64)
        if not np.all(np.isfinite(baseline_positions)) or not np.all(
            np.isfinite(baseline_velocities)
        ):
            raise RuntimeError("warmup articulation state is not finite")

        commanded_delta_by_name = {
            "iiwa_joint_1": 0.12,
            "iiwa_joint_2": 0.10,
            "iiwa_joint_3": -0.09,
            "iiwa_joint_4": 0.11,
            "iiwa_joint_5": -0.08,
            "iiwa_joint_6": 0.09,
            "iiwa_joint_7": 0.07,
            "f1j1": 0.28,
            "f1j2": 0.32,
            "f2j1": 0.30,
            "f3j2": 0.31,
        }
        target_positions = baseline_positions[controlled_indices].copy()
        dof_properties = robot.dof_properties
        for target_index, name in enumerate(controlled_names):
            dof_index = name_to_index[name]
            candidate = baseline_positions[dof_index] + commanded_delta_by_name[name]
            if bool(dof_properties[dof_index]["hasLimits"]):
                lower = float(dof_properties[dof_index]["lower"])
                upper = float(dof_properties[dof_index]["upper"])
                margin = min(0.03, max(0.0, (upper - lower) * 0.05))
                candidate = float(np.clip(candidate, lower + margin, upper - margin))
            target_positions[target_index] = candidate

        ramp_steps = max(1, arguments.steps // 2)
        max_abs_position = float(np.max(np.abs(baseline_positions)))
        max_abs_velocity = float(np.max(np.abs(baseline_velocities)))
        finite_throughout = True
        limit_violation = 0.0
        for step_index in range(arguments.steps):
            blend = min(1.0, float(step_index + 1) / float(ramp_steps))
            step_targets = (
                baseline_positions[controlled_indices]
                + blend
                * (target_positions - baseline_positions[controlled_indices])
            ).astype(np.float32)
            robot.apply_action(
                ArticulationAction(
                    joint_positions=step_targets,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=False)
            positions = np.asarray(robot.get_joint_positions(), dtype=np.float64)
            velocities = np.asarray(robot.get_joint_velocities(), dtype=np.float64)
            if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
                finite_throughout = False
                break
            max_abs_position = max(max_abs_position, float(np.max(np.abs(positions))))
            max_abs_velocity = max(max_abs_velocity, float(np.max(np.abs(velocities))))
            for dof_index in range(robot.num_dof):
                if bool(dof_properties[dof_index]["hasLimits"]):
                    lower = float(dof_properties[dof_index]["lower"])
                    upper = float(dof_properties[dof_index]["upper"])
                    limit_violation = max(
                        limit_violation,
                        lower - float(positions[dof_index]),
                        float(positions[dof_index]) - upper,
                    )

        final_positions = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        final_velocities = np.asarray(robot.get_joint_velocities(), dtype=np.float64)
        finite_final = bool(
            np.all(np.isfinite(final_positions))
            and np.all(np.isfinite(final_velocities))
        )

        response = {}
        for target_index, name in enumerate(controlled_names):
            dof_index = name_to_index[name]
            commanded_delta = float(
                target_positions[target_index] - baseline_positions[dof_index]
            )
            measured_delta = float(final_positions[dof_index] - baseline_positions[dof_index])
            directional_progress = (
                measured_delta * np.sign(commanded_delta)
                if abs(commanded_delta) > 1.0e-9
                else 0.0
            )
            response[name] = {
                "commanded_delta": round(commanded_delta, 6),
                "final_error": round(
                    float(target_positions[target_index] - final_positions[dof_index]), 6
                ),
                "measured_delta": round(measured_delta, 6),
                "progress_fraction": round(
                    float(directional_progress / max(abs(commanded_delta), 1.0e-9)), 6
                ),
            }

        arm_response_ok = all(
            response[name]["progress_fraction"] >= 0.35 for name in ARM_JOINTS
        )
        active_hand_response_ok = all(
            response[name]["progress_fraction"] >= 0.35
            for name in ACTIVE_HAND_JOINTS
        )

        mimic_response = {}
        for follower, source in MIMIC_JOINTS.items():
            follower_index = name_to_index[follower]
            source_index = name_to_index[source]
            follower_delta = float(
                final_positions[follower_index] - baseline_positions[follower_index]
            )
            source_delta = float(
                final_positions[source_index] - baseline_positions[source_index]
            )
            ratio = follower_delta / source_delta if abs(source_delta) > 1.0e-6 else None
            pair_error = abs(float(final_positions[follower_index] - final_positions[source_index]))
            followed = bool(
                ratio is not None
                and 0.7 <= ratio <= 1.3
                and pair_error <= 0.08
            )
            mimic_response[follower] = {
                "expected_source": source,
                "follower_delta": round(follower_delta, 6),
                "position_error": round(pair_error, 6),
                "response_ratio": None if ratio is None else round(float(ratio), 6),
                "source_delta": round(source_delta, 6),
                "followed": followed,
            }
        mimic_executed = all(item["followed"] for item in mimic_response.values())

        stable = bool(
            finite_throughout
            and finite_final
            and limit_violation <= 0.02
            and max_abs_velocity <= 20.0
        )
        passed = bool(
            dof_layout_valid
            and arm_response_ok
            and active_hand_response_ok
            and mimic_executed
            and stable
        )
        metrics.update(
            {
                "active_hand_command_names": list(ACTIVE_HAND_JOINTS),
                "arm_response_ok": arm_response_ok,
                "active_hand_response_ok": active_hand_response_ok,
                "baseline_positions": _named_values(dof_names, baseline_positions),
                "final_positions": _named_values(dof_names, final_positions),
                "final_velocities": _named_values(dof_names, final_velocities),
                "finite_throughout": finite_throughout,
                "limit_violation_rad": round(max(0.0, limit_violation), 6),
                "max_abs_position_rad": round(max_abs_position, 6),
                "max_abs_velocity_rad_s": round(max_abs_velocity, 6),
                "mimic_executed_by_physx": mimic_executed,
                "mimic_response": mimic_response,
                "response": response,
                "stable": stable,
                "target_positions": _named_values(controlled_names, target_positions),
                "test_drive_gains": {
                    "damping": _named_values(dof_names, smoke_kds),
                    "stiffness": _named_values(dof_names, smoke_kps),
                },
                "passed": passed,
            }
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print(
            "ISAAC SIM ROBOT ARTICULATION PHYSICS "
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
        print("ISAAC SIM ROBOT ARTICULATION PHYSICS FAILED", flush=True)
    finally:
        # Fast shutdown terminates standalone Isaac Python.  Propagate the
        # smoke-test result through close() so failures always return non-zero.
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
