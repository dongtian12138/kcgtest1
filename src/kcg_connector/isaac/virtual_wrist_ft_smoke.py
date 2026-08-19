#!/usr/bin/env python3

"""Read a virtual six-axis wrist wrench at the existing hand2arm joint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback


def _asset_file_metadata(asset_path):
    root = asset_path.parent
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
    )
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]


def _rounded(values):
    return [round(float(value), 7) for value in values]


def main():
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-asset",
        default=str(
            repository
            / "artifacts/kcg_connector/isaac/robot/"
            "handarm_keyed_v3_physical_r7/handarm.usda"
        ),
    )
    parser.add_argument("--zero-gravity-steps", type=int, default=60)
    parser.add_argument("--gravity-steps", type=int, default=120)
    parser.add_argument(
        "--minimum-gravity-force-response",
        type=float,
        default=0.1,
    )
    parser.add_argument("--calibration-settle-steps", type=int, default=60)
    parser.add_argument("--calibration-load-steps", type=int, default=120)
    parser.add_argument("--calibration-force-n", type=float, default=4.0)
    parser.add_argument(
        "--calibration-torque-nm", type=float, default=0.4
    )
    parser.add_argument("--calibration-half-scale", type=float, default=0.5)
    arguments = parser.parse_args()
    if arguments.zero_gravity_steps < 2:
        parser.error("--zero-gravity-steps must be at least 2")
    if arguments.gravity_steps < 2:
        parser.error("--gravity-steps must be at least 2")
    if arguments.calibration_settle_steps < 2:
        parser.error("--calibration-settle-steps must be at least 2")
    if arguments.calibration_load_steps < 2:
        parser.error("--calibration-load-steps must be at least 2")
    if not (
        math.isfinite(arguments.minimum_gravity_force_response)
        and arguments.minimum_gravity_force_response > 0.0
    ):
        parser.error("--minimum-gravity-force-response must be positive")
    if not (
        math.isfinite(arguments.calibration_force_n)
        and arguments.calibration_force_n > 0.0
    ):
        parser.error("--calibration-force-n must be positive")
    if not (
        math.isfinite(arguments.calibration_torque_nm)
        and arguments.calibration_torque_nm > 0.0
    ):
        parser.error("--calibration-torque-nm must be positive")
    if not (
        math.isfinite(arguments.calibration_half_scale)
        and 0.0 < arguments.calibration_half_scale < 1.0
    ):
        parser.error("--calibration-half-scale must be between zero and one")

    robot_asset = Path(arguments.robot_asset).expanduser().resolve()
    if not robot_asset.is_file():
        raise FileNotFoundError(robot_asset)
    asset_files_before = _asset_file_metadata(robot_asset)
    asset_file_count = len(asset_files_before)

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
        "asset": str(robot_asset),
        "asset_file_count": asset_file_count,
        "asset_identity_basis": "exact_path_USD_semantics_and_file_metadata_no_fingerprint",
        "canonical_sign_applied": False,
        "measurement_joint": "hand2arm",
        "raw_frame": "handbase_link",
        "raw_semantics": "isaac_incoming_joint_reaction_on_child",
        "wrench_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "zero_gravity_steps": arguments.zero_gravity_steps,
        "gravity_steps": arguments.gravity_steps,
        "axis_calibration_load_application": (
            "physical_force_or_torque_on_handbase_link_local_frame"
        ),
        "axis_calibration_force_application_point": (
            "handbase_link_origin"
        ),
        "calibration_settle_steps": arguments.calibration_settle_steps,
        "calibration_load_steps": arguments.calibration_load_steps,
        "calibration_force_n": arguments.calibration_force_n,
        "calibration_torque_nm": arguments.calibration_torque_nm,
        "calibration_half_scale": arguments.calibration_half_scale,
    }
    try:
        import numpy as np

        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim, SingleArticulation
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
        )
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import UsdGeom, UsdPhysics

        from kcg_connector.robot_model import (
            ACTIVE_HAND_JOINT_NAMES,
            ARM_JOINT_NAMES,
        )
        from kcg_connector.wrist_ft_calibration import (
            WRENCH_AXIS_NAMES,
            analyze_axis_calibration,
        )

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 240.0,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        robot_root = "/World/VirtualWristFtHandArm"
        articulation_path = f"{robot_root}/Geometry/world"
        add_reference_to_stage(str(robot_asset), robot_root)
        stage = get_current_stage()

        fixed_joints = [
            prim
            for prim in stage.Traverse()
            if prim.GetName() == "hand2arm"
            and prim.IsA(UsdPhysics.FixedJoint)
        ]
        if len(fixed_joints) != 1:
            raise RuntimeError(
                "expected exactly one PhysicsFixedJoint named hand2arm, "
                f"found {len(fixed_joints)}"
            )
        hand2arm = UsdPhysics.FixedJoint(fixed_joints[0])
        body0_targets = [
            str(path) for path in hand2arm.GetBody0Rel().GetTargets()
        ]
        body1_targets = [
            str(path) for path in hand2arm.GetBody1Rel().GetTargets()
        ]
        metrics["body0_targets"] = body0_targets
        metrics["body1_targets"] = body1_targets
        boundary_links_ok = (
            len(body0_targets) == 1
            and body0_targets[0].endswith("/iiwa_link_ee")
            and len(body1_targets) == 1
            and body1_targets[0].endswith("/handbase_link")
        )
        metrics["boundary_links_ok"] = boundary_links_ok

        robot = world.scene.add(
            SingleArticulation(
                prim_path=articulation_path,
                name="virtual_wrist_ft_handarm",
            )
        )
        chain_prefix = (
            articulation_path
            + "/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3"
            + "/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7"
            + "/iiwa_link_ee/handbase_link"
        )
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError(
                "robot articulation handles were not initialized"
            )
        handbase_load_view = RigidPrim(
            prim_paths_expr=chain_prefix,
            name="virtual_wrist_ft_handbase_load_view",
            reset_xform_properties=False,
        )
        handbase_load_view.initialize()
        if not handbase_load_view.is_physics_handle_valid():
            raise RuntimeError(
                "handbase physical-load view was not initialized"
            )

        metadata = robot._articulation_view._metadata
        joint_indices = dict(metadata.joint_indices)
        if "hand2arm" not in joint_indices:
            raise RuntimeError("hand2arm is absent from articulation metadata")
        metadata_joint_index = int(joint_indices["hand2arm"])
        reaction_row_index = metadata_joint_index + 1
        metrics["metadata_joint_index"] = metadata_joint_index
        metrics["reaction_row_index"] = reaction_row_index

        all_wrenches = np.asarray(
            robot.get_measured_joint_forces(), dtype=np.float64
        )
        selected_wrench = np.asarray(
            robot.get_measured_joint_forces(
                joint_indices=np.asarray(
                    [reaction_row_index], dtype=np.int32
                )
            ),
            dtype=np.float64,
        )
        metrics["all_wrench_shape"] = list(all_wrenches.shape)
        metrics["selected_wrench_shape"] = list(selected_wrench.shape)
        index_mapping_ok = bool(
            selected_wrench.shape == (1, 6)
            and reaction_row_index < all_wrenches.shape[0]
            and np.allclose(
                selected_wrench[0],
                all_wrenches[reaction_row_index],
                rtol=0.0,
                atol=0.0,
            )
        )
        metrics["joint_index_plus_one_ok"] = index_mapping_ok

        handbase_prim = stage.GetPrimAtPath(chain_prefix)
        grasp_tcp_prim = stage.GetPrimAtPath(f"{chain_prefix}/grasp_tcp")
        if not handbase_prim.IsValid() or not grasp_tcp_prim.IsValid():
            raise RuntimeError("handbase_link or grasp_tcp prim is missing")

        def tcp_offset_in_handbase():
            cache = UsdGeom.XformCache()
            hand_transform = cache.GetLocalToWorldTransform(handbase_prim)
            tcp_transform = cache.GetLocalToWorldTransform(grasp_tcp_prim)
            point = hand_transform.GetInverse().Transform(
                tcp_transform.ExtractTranslation()
            )
            return np.asarray(point, dtype=np.float64)

        tcp_offset_before = tcp_offset_in_handbase()
        metrics["tcp_offset_before_m"] = _rounded(tcp_offset_before)

        dof_names = tuple(robot.dof_names)
        dof_map = {name: index for index, name in enumerate(dof_names)}
        controlled_names = ARM_JOINT_NAMES + ACTIVE_HAND_JOINT_NAMES
        missing_dofs = sorted(set(controlled_names) - set(dof_map))
        if missing_dofs:
            raise RuntimeError(f"missing controlled DOFs: {missing_dofs}")
        controlled_indices = np.asarray(
            [dof_map[name] for name in controlled_names], dtype=np.int32
        )
        arm_indices = np.asarray(
            [dof_map[name] for name in ARM_JOINT_NAMES], dtype=np.int32
        )
        hand_indices = np.asarray(
            [dof_map[name] for name in ACTIVE_HAND_JOINT_NAMES],
            dtype=np.int32,
        )
        hold_target = np.asarray(
            robot.get_joint_positions(), dtype=np.float32
        )[controlled_indices]
        controller = robot.get_articulation_controller()
        kps = np.zeros(robot.num_dof, dtype=np.float32)
        kds = np.zeros(robot.num_dof, dtype=np.float32)
        kps[arm_indices] = 8000.0
        kds[arm_indices] = 220.0
        kps[hand_indices] = 5.0
        kds[hand_indices] = 0.7
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)

        def read_reaction_wrench():
            values = np.asarray(
                robot.get_measured_joint_forces(
                    joint_indices=np.asarray(
                        [reaction_row_index], dtype=np.int32
                    )
                ),
                dtype=np.float64,
            )
            if values.shape != (1, 6):
                raise RuntimeError(
                    f"unexpected hand2arm wrench shape: {values.shape}"
                )
            return values[0]

        def collect(steps, force=None, torque=None):
            tail_length = min(30, max(1, steps // 2))
            samples = []
            for step_index in range(steps):
                robot.apply_action(
                    ArticulationAction(
                        joint_positions=hold_target,
                        joint_indices=controlled_indices,
                    )
                )
                if force is not None or torque is not None:
                    forces = (
                        None
                        if force is None
                        else np.asarray([force], dtype=np.float32)
                    )
                    torques = (
                        None
                        if torque is None
                        else np.asarray([torque], dtype=np.float32)
                    )
                    positions = (
                        None
                        if force is None
                        else np.zeros((1, 3), dtype=np.float32)
                    )
                    handbase_load_view.apply_forces_and_torques_at_pos(
                        forces=forces,
                        torques=torques,
                        positions=positions,
                        is_global=False,
                    )
                world.step(render=False)
                if step_index >= steps - tail_length:
                    samples.append(read_reaction_wrench())
            return np.mean(np.asarray(samples, dtype=np.float64), axis=0)

        world.get_physics_context().set_gravity(0.0)
        zero_gravity_wrench = collect(arguments.zero_gravity_steps)
        world.get_physics_context().set_gravity(-9.81)
        gravity_wrench = collect(arguments.gravity_steps)
        gravity_delta = gravity_wrench - zero_gravity_wrench
        gravity_force_response = float(np.linalg.norm(gravity_delta[:3]))
        finite = bool(
            np.all(np.isfinite(zero_gravity_wrench))
            and np.all(np.isfinite(gravity_wrench))
            and np.all(np.isfinite(gravity_delta))
        )
        includes_distal_gravity = bool(
            finite
            and gravity_force_response
            >= arguments.minimum_gravity_force_response
        )
        metrics["zero_gravity_wrench"] = _rounded(zero_gravity_wrench)
        metrics["gravity_wrench"] = _rounded(gravity_wrench)
        metrics["gravity_delta_wrench"] = _rounded(gravity_delta)
        metrics["gravity_force_response_n"] = round(
            gravity_force_response, 7
        )
        metrics["includes_distal_gravity_load"] = includes_distal_gravity
        metrics["finite"] = finite

        world.get_physics_context().set_gravity(0.0)
        calibration_cases = {}
        load_case_metrics = {}
        unit_axes = np.eye(6, dtype=np.float64)
        load_magnitudes = np.asarray(
            [arguments.calibration_force_n] * 3
            + [arguments.calibration_torque_nm] * 3,
            dtype=np.float64,
        )
        case_specs = (
            ("plus_full", 1.0),
            ("minus_full", -1.0),
            ("plus_half", arguments.calibration_half_scale),
            ("minus_half", -arguments.calibration_half_scale),
        )
        for axis_index, axis_name in enumerate(WRENCH_AXIS_NAMES):
            calibration_cases[axis_name] = {}
            for case_name, signed_scale in case_specs:
                applied = (
                    unit_axes[axis_index]
                    * load_magnitudes[axis_index]
                    * signed_scale
                )
                force = applied[:3] if axis_index < 3 else None
                torque = applied[3:] if axis_index >= 3 else None
                baseline_before = collect(
                    arguments.calibration_settle_steps
                )
                loaded = collect(
                    arguments.calibration_load_steps,
                    force=force,
                    torque=torque,
                )
                baseline_after = collect(
                    arguments.calibration_settle_steps
                )
                baseline = 0.5 * (baseline_before + baseline_after)
                raw_delta = loaded - baseline
                calibration_cases[axis_name][case_name] = raw_delta
                load_case_metrics[f"{axis_name}_{case_name}"] = {
                    "applied_canonical_wrench": _rounded(applied),
                    "raw_baseline_before": _rounded(baseline_before),
                    "raw_loaded": _rounded(loaded),
                    "raw_baseline_after": _rounded(baseline_after),
                    "raw_delta": _rounded(raw_delta),
                }
        calibration = analyze_axis_calibration(
            calibration_cases,
            force_magnitude_n=arguments.calibration_force_n,
            torque_magnitude_nm=arguments.calibration_torque_nm,
            half_scale=arguments.calibration_half_scale,
        )
        metrics["axis_calibration"] = calibration
        metrics["axis_calibration_load_cases"] = load_case_metrics

        tcp_offset_after = tcp_offset_in_handbase()
        expected_tcp_offset = np.asarray([0.0, 0.0, 0.4])
        tcp_offset_error = float(
            np.linalg.norm(tcp_offset_after - expected_tcp_offset)
        )
        tcp_offset_change = float(
            np.linalg.norm(tcp_offset_after - tcp_offset_before)
        )
        tcp_unchanged = bool(
            tcp_offset_error <= 1.0e-6 and tcp_offset_change <= 1.0e-9
        )
        metrics["tcp_offset_after_m"] = _rounded(tcp_offset_after)
        metrics["tcp_offset_error_m"] = tcp_offset_error
        metrics["tcp_offset_change_m"] = tcp_offset_change
        metrics["tcp_unchanged"] = tcp_unchanged

        asset_files_after = _asset_file_metadata(robot_asset)
        after_file_count = len(asset_files_after)
        asset_unchanged = bool(
            asset_files_after == asset_files_before
            and after_file_count == asset_file_count
        )
        metrics["asset_file_metadata_unchanged"] = asset_unchanged
        metrics["asset_unchanged"] = asset_unchanged

        passed = bool(
            boundary_links_ok
            and index_mapping_ok
            and finite
            and includes_distal_gravity
            and calibration["passed"]
            and tcp_unchanged
            and asset_unchanged
        )
    except Exception as error:
        metrics["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        metrics["passed"] = passed
        print(json.dumps(metrics, sort_keys=True), flush=True)
        print(
            "ISAAC VIRTUAL WRIST FT PASSED"
            if passed
            else "ISAAC VIRTUAL WRIST FT FAILED",
            flush=True,
        )
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
