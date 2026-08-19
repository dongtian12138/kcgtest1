#!/usr/bin/env python3
"""Fresh-process post-grasp snapshot replay smoke runner.

This runner opens the hash-pinned replay stage written by the validated
snapshot gate, creates the same Isaac wrappers as the frozen tabletop runner,
restores robot q/qd plus exactly two rigid-body object states, and then holds
the snapshot commands for 120 physics steps.  It does not plan camera poses,
estimate poses, authorize control, or insert.

All Isaac imports are lazy and happen only after SimulationApp is created.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from kcg_connector.display_motion_diagnostics import (
    DisplayMotionRingBuffer,
    atomic_write_json,
    atomic_write_json_lines,
    build_failure_report,
)


def _arguments(repository):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-manifest",
        required=True,
        help="path to replay_bundle_manifest.json from a verified snapshot gate",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pick-config",
        default=str(
            repository / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
        ),
    )
    parser.add_argument(
        "--wrist-ft-config",
        default=str(
            repository / "src/kcg_connector/config/d38999_wrist_ft_monitor_v1.yaml"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--fixed-camera-two-pose",
        action="store_true",
        help=(
            "after the restore settle gate, move to two deterministic fixed"
            " camera display poses and capture raw RGB-D plus robot/sensor "
            " state only; no pose estimation or control authorization"
        ),
    )
    parser.add_argument(
        "--palm-h0-capture",
        action="store_true",
        help="capture one raw palm-camera view at H0 facing the Plug mating face",
    )
    parser.add_argument(
        "--wrist-h0-capture",
        action="store_true",
        help=(
            "after restore settle, capture one raw wrist-camera view at H0 "
            "using the frozen T_HC; no arm motion, no pose estimation"
        ),
    )
    parser.add_argument(
        "--wrist-receptacle-view-validation",
        action="store_true",
        help=(
            "after restore settle, move through the static transport/preinsert "
            "targets and capture W_R0/W_R1 wrist-camera views of the fixed "
            "Receptacle using the frozen T_HC; diagnostic only"
        ),
    )
    parser.add_argument(
        "--robot-side-camera-two-pose",
        action="store_true",
        help=(
            "after the restore settle gate, move to two deterministic poses"
            " under the robot-side near-field fixed camera and capture raw"
            " RGB-D plus robot/sensor state only"
        ),
    )
    parser.add_argument(
        "--visual-chain",
        action="store_true",
        help=(
            "run the formal visual chain: palm T_HP capture+estimate at H0, "
            "then wrist preinsert T_RP views+estimate; no insertion control"
        ),
    )
    parser.add_argument(
        "--rgbd-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_rgbd_bootstrap_v1.yaml"
        ),
    )
    return parser.parse_args()


def main():
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository)
    output_root = Path(arguments.output_dir).expanduser().resolve()
    if output_root.exists():
        raise SystemExit("fresh replay output directory already exists")
    if arguments.visual_chain:
        arguments.palm_h0_capture = True
        arguments.wrist_h0_capture = True
        arguments.wrist_receptacle_view_validation = True
    if (
        arguments.fixed_camera_two_pose
        and arguments.robot_side_camera_two_pose
    ):
        raise SystemExit(
            "choose exactly one of --fixed-camera-two-pose or "
            "--robot-side-camera-two-pose"
        )

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
    process_exit_code = 1
    metrics = {
        "mode": "FRESH_PROCESS_SNAPSHOT_REPLAY",
        "control_authorized": False,
        "formal_estimator_input": False,
        "restore_truth_scope": "INITIALIZATION_ONLY",
        "passed": False,
        "seed": arguments.seed,
    }
    display_trace_buffer = None
    display_path_quality_records = []
    posthoc_sidecar_records = []
    display_posthoc_audit_writer = None
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        from omni.usd import get_context

        from kcg_connector.d38999_tabletop_pick import (
            load_d38999_tabletop_pick_config,
            verify_d38999_pick_dependencies,
        )
        from kcg_connector.virtual_wrist_ft_runtime import (
            load_virtual_wrist_ft_monitor_config,
            reaction_row_index,
            verify_virtual_wrist_ft_monitor_inputs,
        )
        from postgrasp_fresh_replay_runtime import (
            FRESH_REPLAY_RESULT_MARKER,
            load_fresh_replay_bundle,
            run_fresh_replay_restore_settle,
        )

        pick_path = Path(arguments.pick_config).expanduser().resolve()
        pick = load_d38999_tabletop_pick_config(pick_path)
        dependencies = verify_d38999_pick_dependencies(
            pick, pick_path, repository
        )
        tabletop = dependencies["tabletop"]
        robot_asset = dependencies["robot_asset"]
        rate_hz = tabletop.physics.rate_hz

        bundle = load_fresh_replay_bundle(
            arguments.bundle_manifest, expected_seed=arguments.seed
        )
        snapshot = bundle["snapshot"]
        stage_path = bundle["stage_path"]
        metrics["bundle"] = {
            "manifest_path": str(bundle["manifest_path"]),
            "snapshot_sha256": bundle["manifest"]["snapshot_sha256"],
            "stage_sha256": bundle["manifest"]["stage_sha256"],
            "restore_truth_scope": bundle["manifest"][
                "restore_truth_scope"
            ],
            "formal_estimator_input": bundle["manifest"][
                "formal_estimator_input"
            ],
            "control_authorized": bundle["manifest"]["control_authorized"],
        }

        wrist_ft_path = Path(arguments.wrist_ft_config).expanduser().resolve()
        wrist_ft_config = load_virtual_wrist_ft_monitor_config(wrist_ft_path)
        wrist_ft_inputs = verify_virtual_wrist_ft_monitor_inputs(
            wrist_ft_config, repository
        )
        if wrist_ft_inputs["robot_asset"] != robot_asset:
            raise RuntimeError(
                "wrist FT monitor does not bind the active robot asset"
            )

        # The bundle loader already re-hashed both files.  Open that exact
        # stage in this new process; no scene authoring or randomization is
        # copied here.
        if not get_context().open_stage(str(stage_path)):
            raise RuntimeError("replay stage open failed")
        stage = get_current_stage()
        required_prims = (
            pick.scene.articulation_prim_path,
            tabletop.asset.body_prim_path,
            tabletop.asset.nut_prim_path,
        )
        for prim_path in required_prims:
            prim = stage.GetPrimAtPath(prim_path)
            if prim is None or not prim.IsValid():
                raise RuntimeError(f"required replay prim missing: {prim_path}")

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / rate_hz,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        robot = world.scene.add(
            SingleArticulation(
                prim_path=pick.scene.articulation_prim_path,
                name="fresh_replay_handarm",
            )
        )
        body_primitive = world.scene.add(
            SingleRigidPrim(
                prim_path=tabletop.asset.body_prim_path,
                name="fresh_replay_body",
            )
        )
        nut_primitive = world.scene.add(
            SingleRigidPrim(
                prim_path=tabletop.asset.nut_prim_path,
                name="fresh_replay_nut",
            )
        )
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError("robot articulation handles were not initialized")

        dof_names = tuple(robot.dof_names)
        expected_dof_names = tuple(
            f"iiwa_joint_{index}" for index in range(1, 8)
        ) + ("f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3")
        if set(dof_names) != set(expected_dof_names) or len(dof_names) != 15:
            raise RuntimeError("unexpected articulation DOF layout")
        name_to_index = {name: index for index, name in enumerate(dof_names)}
        arm_indices = np.asarray(
            [name_to_index[name] for name in pick.robot.arm_joint_names],
            dtype=np.int32,
        )
        hand_indices = np.asarray(
            [name_to_index[name] for name in pick.robot.active_hand_joint_names],
            dtype=np.int32,
        )
        sensor_indices = np.asarray(
            [name_to_index[name] for name in pick.sensing.torque_joint_names],
            dtype=np.int32,
        )
        arm_joint_limits = []
        for arm_index in range(7):
            dof_property = robot.dof_properties[arm_index]
            if not bool(dof_property["hasLimits"]):
                raise RuntimeError(
                    f"arm joint {arm_index} has no configured limits"
                )
            arm_joint_limits.append(
                (
                    float(dof_property["lower"]),
                    float(dof_property["upper"]),
                )
            )
        expected_sensor_names = tuple(
            pick.robot.active_hand_joint_names[index]
            for index in (1, 2, 3)
        )
        if expected_sensor_names != tuple(pick.sensing.torque_joint_names):
            raise RuntimeError(
                "finger target-to-root-torque mapping changed"
            )
        controlled_indices = np.concatenate((arm_indices, hand_indices))

        controller = robot.get_articulation_controller()
        kps = np.zeros(robot.num_dof, dtype=np.float32)
        kds = np.zeros(robot.num_dof, dtype=np.float32)
        kps[arm_indices] = pick.robot.arm_stiffness
        kds[arm_indices] = pick.robot.arm_damping
        kps[hand_indices] = pick.motion.grip_hand_stiffness
        kds[hand_indices] = pick.motion.grip_hand_damping
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        world.get_physics_context().set_gravity(
            tabletop.physics.gravity_m_s2
        )

        metadata_joint_indices = dict(
            robot._articulation_view._metadata.joint_indices
        )
        wrist_reaction_row = reaction_row_index(
            metadata_joint_indices, wrist_ft_config
        )
        raw_wrench = np.asarray(
            robot.get_measured_joint_forces(
                joint_indices=np.asarray(
                    [wrist_reaction_row], dtype=np.int32
                )
            ),
            dtype=np.float64,
        )
        all_wrenches = np.asarray(
            robot.get_measured_joint_forces(), dtype=np.float64
        )
        if (
            raw_wrench.shape != (1, 6)
            or wrist_reaction_row >= all_wrenches.shape[0]
            or not np.array_equal(
                raw_wrench[0], all_wrenches[wrist_reaction_row]
            )
        ):
            raise RuntimeError(
                "hand2arm reaction row failed joint-index-plus-one check"
            )
        canonical_from_raw = np.asarray(
            wrist_ft_config.canonical_from_raw, dtype=np.float64
        )

        tare_efforts = np.asarray(
            snapshot["finger_state"]["finger_root_tare_efforts_nm"],
            dtype=np.float64,
        ).ravel()
        if tare_efforts.shape != (3,) or not np.all(np.isfinite(tare_efforts)):
            raise RuntimeError("snapshot finger tare is invalid")

        class PoseWriteCounter:
            def __init__(self):
                self.value = 0

            def increment(self):
                self.value += 1

        pose_write_counter = PoseWriteCounter()

        class CountedRigidBody:
            def __init__(self, primitive, counter):
                self._primitive = primitive
                self._counter = counter

            def get_world_pose(self):
                return self._primitive.get_world_pose()

            def get_linear_velocity(self):
                return self._primitive.get_linear_velocity()

            def get_angular_velocity(self):
                return self._primitive.get_angular_velocity()

            def set_world_pose(self, *, position, orientation):
                self._counter.increment()
                return self._primitive.set_world_pose(
                    position=position, orientation=orientation
                )

            def set_linear_velocity(self, value):
                return self._primitive.set_linear_velocity(value)

            def set_angular_velocity(self, value):
                return self._primitive.set_angular_velocity(value)

        body = CountedRigidBody(body_primitive, pose_write_counter)
        nut = CountedRigidBody(nut_primitive, pose_write_counter)

        def restore_rigid_body(name, rigid_state):
            target = body if name == "plug" else nut
            target.set_world_pose(
                position=np.asarray(
                    rigid_state["position_m"], dtype=np.float64
                ),
                orientation=np.asarray(
                    rigid_state["orientation_wxyz"], dtype=np.float64
                ),
            )
            target.set_linear_velocity(
                np.asarray(
                    rigid_state["linear_velocity_m_s"], dtype=np.float64
                )
            )
            target.set_angular_velocity(
                np.asarray(
                    rigid_state["angular_velocity_rad_s"], dtype=np.float64
                )
            )

        runtime_state = {"global_step": 0}

        def observe_and_step(arm_target, hand_target):
            target = np.concatenate(
                (np.asarray(arm_target), np.asarray(hand_target))
            ).astype(np.float32)
            robot.apply_action(
                ArticulationAction(
                    joint_positions=target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            runtime_state["global_step"] += 1
            return (
                np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                ).ravel(),
                np.asarray(
                    robot.get_joint_velocities(), dtype=np.float64
                ).ravel(),
            )

        def sample_finger_torque_proxy():
            measured = np.asarray(
                robot.get_measured_joint_efforts(
                    joint_indices=sensor_indices
                ),
                dtype=np.float64,
            ).ravel()
            delta = measured - tare_efforts
            if not np.all(np.isfinite(delta)):
                raise RuntimeError("finger torque proxy is non-finite")
            return delta

        def get_latest_wrist_state():
            raw = np.asarray(
                robot.get_measured_joint_forces(
                    joint_indices=np.asarray(
                        [wrist_reaction_row], dtype=np.int32
                    )
                ),
                dtype=np.float64,
            ).ravel()
            if raw.shape != (6,) or not np.all(np.isfinite(raw)):
                return {
                    "global_step": runtime_state["global_step"],
                    "canonical": np.full(6, np.nan),
                    "error": "wrist reaction row invalid",
                }
            return {
                "global_step": runtime_state["global_step"],
                "raw_wrench": raw.tolist(),
                "canonical": canonical_from_raw @ raw,
                "error": None,
            }

        result = run_fresh_replay_restore_settle(
            snapshot=snapshot,
            stage_path=stage_path,
            open_stage=lambda path: True,
            reset_world=lambda: None,
            robot_set_q=robot.set_joint_positions,
            robot_set_qd=robot.set_joint_velocities,
            restore_rigid_body=restore_rigid_body,
            observe_and_step=observe_and_step,
            sample_finger_torque_proxy=sample_finger_torque_proxy,
            get_latest_wrist_state=get_latest_wrist_state,
            restored_plug_position=lambda: np.asarray(
                body.get_world_pose()[0], dtype=np.float64
            ),
            restored_plug_orientation=lambda: np.asarray(
                body.get_world_pose()[1], dtype=np.float64
            ),
            restored_nut_position=lambda: np.asarray(
                nut.get_world_pose()[0], dtype=np.float64
            ),
            restored_nut_orientation=lambda: np.asarray(
                nut.get_world_pose()[1], dtype=np.float64
            ),
            object_write_counter=pose_write_counter,
        )
        metrics["fresh_replay"] = result
        if arguments.wrist_h0_capture or arguments.palm_h0_capture:
            if result.get("status") != "FRESH_REPLAY_RESTORE_VERIFIED":
                raise RuntimeError("wrist H0 capture requires verified restore")
            from dataclasses import replace
            from isaacsim.sensors.camera import Camera
            import omni.replicator.core as rep
            from PIL import Image
            from pxr import Gf, Usd, UsdGeom, UsdLux

            from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform
            from kcg_connector.d38999_cad_registration import fixed_camera_model
            from kcg_connector.isaac_d38999_rgbd_runtime import capture_d38999_rgbd_raw_formal
            from kcg_connector.rgbd_pose_bootstrap import load_rgbd_bootstrap
            from postgrasp_shadow_capture_runtime import (
                _calibrated_hand_camera_from_nominal_plug,
                _camera_target_from_t_wc,
            )

            output_root.mkdir(parents=True, exist_ok=False)
            rgbd_base = load_rgbd_bootstrap(Path(arguments.rgbd_config).expanduser().resolve())
            tcp_from_handbase = np.eye(4, dtype=np.float64)
            tcp_from_handbase[2, 3] = -float(pick.geometry_candidate.handbase_to_tcp_m)
            nominal_tcp = np.asarray(iiwa14_grasp_tcp_transform(tuple(pick.motion.grasp_arm_rad)))
            nominal_plug = np.eye(4, dtype=np.float64)
            nominal_plug[:3, 3] = np.asarray(pick.geometry_candidate.loose_settled_origin_m)
            nominal_hand_to_plug = np.linalg.inv(nominal_tcp @ tcp_from_handbase) @ nominal_plug
            capture_views = []
            if arguments.palm_h0_capture:
                import os
                eyes_text = os.environ.get("PALM_EYES")
                if eyes_text:
                    for index, group in enumerate(eyes_text.split(";")):
                        palm_eye = tuple(
                            float(value) for value in group.split(",")
                        )
                        if len(palm_eye) != 3 or not all(
                            np.isfinite(palm_eye)
                        ):
                            raise RuntimeError(
                                f"invalid PALM_EYES group {index}: {group!r}"
                            )
                        capture_views.append(
                            (f"PALM_H0_K{index}", "palm", palm_eye)
                        )
                else:
                    palm_eye = tuple(
                        float(value)
                        for value in os.environ.get(
                            "PALM_EYE", "-0.030,-0.070,0.100"
                        ).split(",")
                    )
                    capture_views.append(("PALM_H0", "palm", palm_eye))
            if arguments.wrist_h0_capture:
                capture_views.append(("WRIST_H0", "wrist", None))
            actual_arm_q = np.asarray(robot.get_joint_positions(), dtype=np.float64).ravel()[:7]
            tcp_fk = np.asarray(iiwa14_grasp_tcp_transform(tuple(actual_arm_q)))
            t_wh = tcp_fk @ tcp_from_handbase
            camera_path = "/World/PostgraspShadowWristRgbdCamera"
            camera_prim = stage.GetPrimAtPath(camera_path)
            if camera_prim is None or not camera_prim.IsValid():
                camera_prim = UsdGeom.Camera.Define(stage, camera_path)
            else:
                camera_prim = UsdGeom.Camera(camera_prim)
            identity = Gf.Matrix4d(1.0); identity.SetTranslateOnly(Gf.Vec3d(0,0,0))
            UsdGeom.Xformable(camera_prim).ClearXformOpOrder(); UsdGeom.Xformable(camera_prim).AddTransformOp().Set(identity)
            camera_prim.CreateFocalLengthAttr(24.0); camera_prim.CreateHorizontalApertureAttr(20.955)
            camera_prim.CreateVerticalApertureAttr(20.955*720/1280); camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1,10.0))
            view_records = []
            for view_id, kind, palm_eye in capture_views:
                if kind == "palm":
                    palm_model = fixed_camera_model(
                        eye=palm_eye,
                        target=(0.001, 0.0, 0.0),
                        resolution=(1280, 720),
                    )
                    palm_rows = np.asarray(
                        palm_model.world_to_camera, dtype=np.float64
                    )
                    camera_in_plug = np.eye(4, dtype=np.float64)
                    camera_in_plug[:3, :3] = palm_rows.T
                    camera_in_plug[:3, 3] = np.asarray(
                        palm_model.position_world, dtype=np.float64
                    )
                    t_hc = nominal_hand_to_plug @ camera_in_plug
                else:
                    t_hc = _calibrated_hand_camera_from_nominal_plug(
                        nominal_hand_to_plug, (0.120, 0.0, 0.060), (0.0, 0.0, 0.006), (1280, 720)
                    )
                t_wc = t_wh @ t_hc
                eye_world = tuple(float(v) for v in t_wc[:3, 3])
                target_world = tuple(float(v) for v in _camera_target_from_t_wc(t_wc))
                direction = np.asarray(target_world) - np.asarray(eye_world)
                direction = direction / np.linalg.norm(direction)
                camera_rotation = Gf.Rotation(Gf.Vec3d(0,0,-1), Gf.Vec3d(*direction))
                camera_matrix = Gf.Matrix4d(1.0); camera_matrix.SetRotate(camera_rotation); camera_matrix.SetTranslateOnly(Gf.Vec3d(*eye_world))
                UsdGeom.Xformable(camera_prim).ClearXformOpOrder(); UsdGeom.Xformable(camera_prim).AddTransformOp().Set(camera_matrix)
                wrist_rgbd = replace(rgbd_base, camera=replace(rgbd_base.camera, prim_path=camera_path, frame_id="postgrasp_wrist_rgbd_camera_optical", eye_m=eye_world, target_m=target_world, resolution=(1280,720)))
                view_dir = output_root / "formal_views" / view_id
                capture = capture_d38999_rgbd_raw_formal(
                    bindings={"Camera":Camera,"Gf":Gf,"Image":Image,"Usd":Usd,"UsdGeom":UsdGeom,"UsdLux":UsdLux,"rep":rep},
                    simulation_app=simulation_app, world=world, stage=stage, tabletop=tabletop,
                    rgbd=wrist_rgbd, output_dir=view_dir, camera_clipping_range_m=(0.1,10.0),
                )
                if capture.passed is not True:
                    raise RuntimeError(f"raw capture failed for view {view_id}")
                (view_dir/"fk.json").write_text(json.dumps({"arm_q_actual_rad":actual_arm_q.tolist(),"tcp_pose_4x4":tcp_fk.tolist(),"T_WH_4x4":t_wh.tolist(),"T_WC_4x4":t_wc.tolist()},indent=2)+"\n")
                camera_record = {"prim_path":camera_path,"frame_id":"postgrasp_wrist_rgbd_camera_optical","eye_m":list(eye_world),"target_m":list(target_world),"intrinsics":capture.metrics["camera"]["intrinsics"]}
                if kind == "palm":
                    camera_record["palm_eye_plug_m"] = list(palm_eye)
                    camera_record["palm_target_plug_m"] = [0.001, 0.0, 0.0]
                    camera_record["T_HC_4x4"] = t_hc.tolist()
                (view_dir/"camera.json").write_text(json.dumps(camera_record,indent=2)+"\n")
                view_records.append({"view_id":view_id,"output_directory":str(view_dir)})
                metrics[f"capture_{view_id}"] = {"status":"GPU_PASS","view_id":view_id,"control_authorized":False,"formal_estimator_input":True}
            (output_root/"formal_manifest.json").write_text(json.dumps({"schema_version":"kcg_d38999_wrist_h0_v1","role":"formal_raw_observation","formal_estimator_input":True,"estimator_run":False,"control_authorized":False,"object_truth_present":False,"contact_report_present":False,"views":view_records},indent=2)+"\n")
            plug_pose = body.get_world_pose()
            nut_pose = nut.get_world_pose()
            (output_root/"posthoc_truth_sidecar.json").write_text(json.dumps({
                "schema_version": "kcg_d38999_capture_time_posthoc_truth_v1",
                "role": "posthoc_truth_sidecar",
                "truth_scope": "capture_time_posthoc_evaluation_only",
                "formal_estimator_input": False,
                "control_authorized": False,
                "object_truth_present": True,
                "contact_report_present": False,
                "plug_position_m": [float(v) for v in plug_pose[0]],
                "plug_orientation_wxyz": [float(v) for v in plug_pose[1]],
                "nut_position_m": [float(v) for v in nut_pose[0]],
                "nut_orientation_wxyz": [float(v) for v in nut_pose[1]],
                "arm_q_actual_rad": actual_arm_q.tolist(),
            }, indent=2) + "\n")
            if arguments.wrist_h0_capture and not arguments.palm_h0_capture:
                metrics["wrist_h0_capture"] = {"status":"GPU_PASS","view_id":"WRIST_H0","control_authorized":False,"formal_estimator_input":True}


        if arguments.wrist_receptacle_view_validation:
            if result.get("status") != "FRESH_REPLAY_RESTORE_VERIFIED":
                raise RuntimeError(
                    "wrist-receptacle validation requires a verified restore"
                )
            from dataclasses import replace
            from isaacsim.sensors.camera import Camera
            import omni.replicator.core as rep
            from PIL import Image
            from pxr import Gf, Usd, UsdGeom, UsdLux

            from kcg_connector.d38999_physical_insertion import (
                load_d38999_physical_insertion,
                solve_fixed_q7_tcp_pose,
            )
            from kcg_connector.d38999_tabletop_pick import (
                iiwa14_grasp_tcp_transform,
            )
            from kcg_connector.display_motion_diagnostics import (  # noqa: F811
                evaluate_display_sensor_gates,
                evaluate_display_wrist_evidence,
                evaluate_waypoint_path_quality,
            )
            from kcg_connector.isaac_d38999_rgbd_runtime import (
                capture_d38999_rgbd_raw_formal,
            )
            from kcg_connector.rgbd_pose_bootstrap import load_rgbd_bootstrap
            from kcg_connector.wrist_receptacle_view_design import (
                design_wrist_receptacle_views,
                receptacle_world_pose,
            )
            from postgrasp_shadow_capture_runtime import (
                _camera_target_from_t_wc,
                _calibrated_hand_camera_from_nominal_plug,
            )

            output_root.mkdir(parents=True, exist_ok=True)
            display_trace_buffer = DisplayMotionRingBuffer(capacity=240)
            display_path_quality_records = []
            posthoc_sidecar_records = []

            insertion = load_d38999_physical_insertion(
                Path(arguments.insertion_config).expanduser().resolve()
            ) if hasattr(arguments, "insertion_config") else None
            if insertion is None:
                insertion = load_d38999_physical_insertion(
                    repository
                    / "src/kcg_connector/config/d38999_physical_insertion_v1.yaml"
                )
            cpu_plan_path = (
                repository
                / "artifacts/kcg_connector/d38999_visual_xy_preinsert_probe_v1"
                / "first_gpu_20260812T154220Z/preinsert_cpu_plan.json"
            )
            cpu_plan = json.loads(cpu_plan_path.read_text(encoding="utf-8"))
            rgbd_base = load_rgbd_bootstrap(
                Path(arguments.rgbd_config).expanduser().resolve()
            )

            tcp_from_handbase = np.eye(4, dtype=np.float64)
            tcp_from_handbase[2, 3] = -float(
                pick.geometry_candidate.handbase_to_tcp_m
            )
            nominal_tcp = np.asarray(
                iiwa14_grasp_tcp_transform(
                    tuple(float(value) for value in pick.motion.grasp_arm_rad)
                ),
                dtype=np.float64,
            )
            nominal_plug = np.eye(4, dtype=np.float64)
            nominal_plug[:3, 3] = np.asarray(
                pick.geometry_candidate.loose_settled_origin_m,
                dtype=np.float64,
            )
            nominal_hand_to_plug = (
                np.linalg.inv(nominal_tcp @ tcp_from_handbase)
                @ nominal_plug
            )
            t_hc_frozen = _calibrated_hand_camera_from_nominal_plug(
                nominal_hand_to_plug,
                (0.120, 0.0, 0.060),
                (0.0, 0.0, 0.006),
                (1280, 720),
            )
            frozen_hand_target = np.asarray(
                snapshot["frozen_command"]["hand_q_target_rad"],
                dtype=np.float64,
            )
            frozen_arm_target = np.asarray(
                snapshot["frozen_command"]["arm_q_target_rad"],
                dtype=np.float64,
            )
            wrist_reference = np.asarray(
                snapshot["frozen_command"]["wrist_ft_snapshot_reference"],
                dtype=np.float64,
            )
            previous_raw_wrist = wrist_reference.copy()
            ema_wrist = wrist_reference.copy()
            commanded_arm = frozen_arm_target.copy()

            camera_path = "/World/PostgraspShadowWristRgbdCamera"
            camera_prim = stage.GetPrimAtPath(camera_path)
            if camera_prim is None or not camera_prim.IsValid():
                camera_prim = UsdGeom.Camera.Define(stage, camera_path)
            else:
                camera_prim = UsdGeom.Camera(camera_prim)
            identity_matrix = Gf.Matrix4d(1.0)
            identity_matrix.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, 0.0))
            UsdGeom.Xformable(camera_prim).ClearXformOpOrder()
            UsdGeom.Xformable(camera_prim).AddTransformOp().Set(identity_matrix)
            camera_prim.CreateFocalLengthAttr(24.0)
            camera_prim.CreateHorizontalApertureAttr(20.955)
            camera_prim.CreateVerticalApertureAttr(
                20.955 * 720.0 / 1280.0
            )
            camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10.0))

            camera_bindings = {
                "Camera": Camera,
                "Gf": Gf,
                "Image": Image,
                "Usd": Usd,
                "UsdGeom": UsdGeom,
                "UsdLux": UsdLux,
                "rep": rep,
            }

            def record_trace(phase, desired_arm, positions, velocities, torque, wrist, sensor_gates, evidence):
                display_trace_buffer.append(
                    {
                        "role": "display_motion_trace_record",
                        "global_step": runtime_state["global_step"],
                        "phase": phase,
                        "desired_q": np.asarray(desired_arm).tolist(),
                        "actual_q": np.asarray(positions).tolist(),
                        "qd": np.asarray(velocities).tolist(),
                        "sensor_gates": sensor_gates,
                        "canonical_wrench": np.asarray(wrist).tolist(),
                        "triggered_gates": evidence["triggered_gates"],
                        "moment_evidence": evidence["moment_evidence"],
                    }
                )

            def move_guarded(target_arm, phase):
                nonlocal commanded_arm
                nonlocal previous_raw_wrist
                nonlocal ema_wrist
                target = np.asarray(target_arm, dtype=np.float64)
                start = commanded_arm.copy()
                duration_s = max(10.0, float(np.max(np.abs(target - start))) / 0.010)
                steps = max(1, round(duration_s * rate_hz))
                for index in range(steps):
                    blend = float(index + 1) / float(steps)
                    desired_arm = np.asarray(
                        start + blend * (target - start),
                        dtype=np.float64,
                    )
                    positions, velocities = observe_and_step(
                        desired_arm, frozen_hand_target
                    )
                    torque = sample_finger_torque_proxy()
                    wrist_state = get_latest_wrist_state()
                    wrist = np.asarray(wrist_state["canonical"], dtype=np.float64)
                    sensor_gates = evaluate_display_sensor_gates(
                        desired_arm_q=desired_arm,
                        actual_q=np.asarray(positions[:7]),
                        velocities=velocities,
                        torque=torque,
                        joint_limits=arm_joint_limits,
                        joint_limit_margin_rad=0.010,
                        max_abs_torque_nm=float(
                            pick.sensing.maximum_absolute_torque_delta_nm
                        ),
                        max_joint_speed_rad_s=1.0,
                        max_arm_tracking_error_rad=0.030,
                    )
                    evidence = evaluate_display_wrist_evidence(
                        current_wrench=wrist,
                        reference_wrench=wrist_reference,
                        previous_raw_wrench=previous_raw_wrist,
                        ema_wrench=ema_wrist,
                    )
                    previous_raw_wrist = wrist.copy()
                    ema_wrist = np.asarray(evidence["ema_wrench"], dtype=np.float64)
                    record_trace(phase, desired_arm, positions, velocities, torque, wrist, sensor_gates, evidence)
                    if (
                        not sensor_gates["ok"]
                        or evidence["formal_gate_triggered"]
                        or evidence["ema_candidate_triggered"]
                    ):
                        raise RuntimeError(
                            f"{phase} safety fail-closed at step {index}: "
                            f"sensor={sensor_gates['reasons']}, "
                            f"gates={evidence['triggered_gates']}"
                        )
                commanded_arm = target.copy()
                for hold_index in range(120):
                    positions, velocities = observe_and_step(
                        target, frozen_hand_target
                    )
                    torque = sample_finger_torque_proxy()
                    wrist_state = get_latest_wrist_state()
                    wrist = np.asarray(wrist_state["canonical"], dtype=np.float64)
                    sensor_gates = evaluate_display_sensor_gates(
                        desired_arm_q=target,
                        actual_q=np.asarray(positions[:7]),
                        velocities=velocities,
                        torque=torque,
                        joint_limits=arm_joint_limits,
                        joint_limit_margin_rad=0.010,
                        max_abs_torque_nm=float(
                            pick.sensing.maximum_absolute_torque_delta_nm
                        ),
                        max_joint_speed_rad_s=1.0,
                        max_arm_tracking_error_rad=0.030,
                    )
                    evidence = evaluate_display_wrist_evidence(
                        current_wrench=wrist,
                        reference_wrench=wrist_reference,
                        previous_raw_wrench=previous_raw_wrist,
                        ema_wrench=ema_wrist,
                    )
                    previous_raw_wrist = wrist.copy()
                    ema_wrist = np.asarray(evidence["ema_wrench"], dtype=np.float64)
                    record_trace(f"{phase}_hold", target, positions, velocities, torque, wrist, sensor_gates, evidence)
                    if (
                        not sensor_gates["ok"]
                        or evidence["formal_gate_triggered"]
                        or evidence["ema_candidate_triggered"]
                    ):
                        raise RuntimeError(
                            f"{phase} hold fail-closed: "
                            f"sensor={sensor_gates['reasons']}, "
                            f"gates={evidence['triggered_gates']}"
                        )

            for stage_name in cpu_plan["target_order"]:
                move_guarded(cpu_plan["arm_targets_rad"][stage_name], stage_name)

            preinsert_arm = np.asarray(
                cpu_plan["arm_targets_rad"]["preinsert"], dtype=np.float64
            )
            receptacle = receptacle_world_pose(
                (0.550, 0.185, 0.2615), (0.0, 0.0, 1.0)
            )
            design = design_wrist_receptacle_views(
                base_arm_q=preinsert_arm,
                solve_arm=solve_fixed_q7_tcp_pose,
                tcp_from_handbase=tcp_from_handbase,
                nominal_hand_to_plug=nominal_hand_to_plug,
                tcp_to_camera=t_hc_frozen,
                receptacle_world=receptacle,
                joint_limits=arm_joint_limits,
            )
            validation_views = []
            for view in design["views"]:
                if view["view_id"] not in design["selected_view_ids"]:
                    continue
                move_guarded(view["arm_q_rad"], f"wrist_{view['view_id']}")
                actual_arm_q = np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                ).ravel()[:7]
                tcp_fk = np.asarray(
                    iiwa14_grasp_tcp_transform(tuple(actual_arm_q))
                )
                t_wh = tcp_fk @ tcp_from_handbase
                t_wc = t_wh @ t_hc_frozen
                eye_world = tuple(float(v) for v in t_wc[:3, 3])
                target_world = tuple(float(v) for v in _camera_target_from_t_wc(t_wc))
                direction = np.asarray(target_world) - np.asarray(eye_world)
                direction = direction / np.linalg.norm(direction)
                camera_rotation = Gf.Rotation(
                    Gf.Vec3d(0.0, 0.0, -1.0), Gf.Vec3d(*direction)
                )
                camera_matrix = Gf.Matrix4d(1.0)
                camera_matrix.SetRotate(camera_rotation)
                camera_matrix.SetTranslateOnly(Gf.Vec3d(*eye_world))
                UsdGeom.Xformable(camera_prim).ClearXformOpOrder()
                UsdGeom.Xformable(camera_prim).AddTransformOp().Set(camera_matrix)
                wrist_rgbd = replace(
                    rgbd_base,
                    camera=replace(
                        rgbd_base.camera,
                        prim_path=camera_path,
                        frame_id="postgrasp_wrist_rgbd_camera_optical",
                        eye_m=eye_world,
                        target_m=target_world,
                        resolution=(1280, 720),
                    ),
                )
                view_dir = output_root / "formal_views" / view["view_id"]
                capture = capture_d38999_rgbd_raw_formal(
                    bindings=camera_bindings,
                    simulation_app=simulation_app,
                    world=world,
                    stage=stage,
                    tabletop=tabletop,
                    rgbd=wrist_rgbd,
                    output_dir=view_dir,
                    camera_clipping_range_m=(0.1, 10.0),
                )
                if capture.passed is not True:
                    raise RuntimeError(f"{view['view_id']} wrist capture failed")
                (view_dir / "fk.json").write_text(
                    json.dumps(
                        {
                            "arm_q_actual_rad": actual_arm_q.tolist(),
                            "tcp_pose_4x4": tcp_fk.tolist(),
                            "T_WH_4x4": t_wh.tolist(),
                            "T_WC_4x4": t_wc.tolist(),
                        },
                        allow_nan=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (view_dir / "camera.json").write_text(
                    json.dumps(
                        {
                            "prim_path": camera_path,
                            "frame_id": "postgrasp_wrist_rgbd_camera_optical",
                            "eye_m": list(eye_world),
                            "target_m": list(target_world),
                            "intrinsics": capture.metrics[
                                "camera"
                            ]["intrinsics"],
                        },
                        allow_nan=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                view_plug_pose = body.get_world_pose()
                (view_dir / "posthoc_plug_pose.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "kcg_d38999_capture_time_posthoc_truth_v1",
                            "role": "posthoc_truth_sidecar",
                            "truth_scope": "capture_time_posthoc_evaluation_only",
                            "formal_estimator_input": False,
                            "control_authorized": False,
                            "plug_position_m": [
                                float(v) for v in view_plug_pose[0]
                            ],
                            "plug_orientation_wxyz": [
                                float(v) for v in view_plug_pose[1]
                            ],
                        },
                        allow_nan=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                validation_views.append(
                    {
                        "view_id": view["view_id"],
                        "output_directory": str(view_dir),
                        "capture_metrics": capture.metrics,
                    }
                )
            (output_root / "formal_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "kcg_d38999_wrist_receptacle_validation_v1",
                        "role": "formal_raw_observation",
                        "formal_estimator_input": True,
                        "estimator_run": False,
                        "control_authorized": False,
                        "object_truth_present": False,
                        "contact_report_present": False,
                        "views": validation_views,
                        "design": design,
                    },
                    allow_nan=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            atomic_write_json_lines(
                output_root / "motion_trace.jsonl",
                display_trace_buffer.records(),
            )
            metrics["wrist_receptacle_validation"] = {
                "status": "GPU_PASS",
                "selected_views": design["selected_view_ids"],
                "condition": design["condition"],
                "control_authorized": False,
                "formal_estimator_input": True,
            }
        if arguments.fixed_camera_two_pose or arguments.robot_side_camera_two_pose:
            if result.get("status") != "FRESH_REPLAY_RESTORE_VERIFIED":
                raise RuntimeError(
                    "fixed-camera two-pose smoke requires a verified restore"
                )
            from isaacsim.sensors.camera import Camera
            from omni.physx import get_physx_simulation_interface
            import omni.replicator.core as rep
            from PIL import Image
            from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdLux
            from scipy.spatial.transform import Rotation
            from d38999_tabletop_pick_smoke import (
                _classify_robot_external_contact,
                _is_finger_plug_contact,
                _is_plug_table_contact,
            )

            from kcg_connector.d38999_physical_insertion import (
                solve_fixed_q7_tcp_pose,
            )
            from kcg_connector.d38999_tabletop_pick import (
                iiwa14_grasp_tcp_transform,
                interpolate_arm,
            )
            from kcg_connector.d38999_tabletop_pick import (
                iiwa14_grasp_tcp_transform,
            )
            from kcg_connector.display_motion_diagnostics import (  # noqa: F811
                evaluate_display_sensor_gates,
                evaluate_display_wrist_evidence,
                evaluate_waypoint_path_quality,
            )
            from kcg_connector.isaac_d38999_rgbd_runtime import (
                capture_d38999_rgbd_raw_formal,
            )
            from kcg_connector.postgrasp_shadow_view_planner import (
                plan_cartesian_tcp_waypoints,
                plan_two_fixed_camera_display_feasibility_poses,
                plan_two_robot_side_camera_display_poses,
            )
            from kcg_connector.rgbd_pose_bootstrap import load_rgbd_bootstrap

            rgbd_path = Path(arguments.rgbd_config).expanduser().resolve()
            rgbd = load_rgbd_bootstrap(rgbd_path)
            bound_tabletop_path = (
                repository / rgbd.tabletop_config
            ).resolve()
            active_tabletop_path = (
                pick_path.parent / pick.scene.tabletop_config
            ).resolve()
            if bound_tabletop_path != active_tabletop_path:
                raise RuntimeError(
                    "RGB-D config does not bind the active tabletop config"
                )

            nominal_world_tcp = np.asarray(
                iiwa14_grasp_tcp_transform(
                    tuple(float(value) for value in pick.motion.grasp_arm_rad)
                ),
                dtype=np.float64,
            )
            tcp_from_handbase = np.eye(4, dtype=np.float64)
            tcp_from_handbase[2, 3] = -float(
                pick.geometry_candidate.handbase_to_tcp_m
            )
            nominal_world_handbase = (
                nominal_world_tcp @ tcp_from_handbase
            )
            nominal_world_plug = np.eye(4, dtype=np.float64)
            nominal_world_plug[:3, 3] = np.asarray(
                pick.geometry_candidate.loose_settled_origin_m,
                dtype=np.float64,
            )
            nominal_hand_to_plug = (
                np.linalg.inv(nominal_world_handbase)
                @ nominal_world_plug
            )

            q0_arm = np.asarray(
                snapshot["robot_state"]["q_rad"][:7], dtype=np.float64
            )
            if arguments.robot_side_camera_two_pose:
                display_poses, predicted_direction_deg = (
                    plan_two_robot_side_camera_display_poses(
                        q0_arm,
                        solve_arm=solve_fixed_q7_tcp_pose,
                        handbase_to_tcp_m=(
                            pick.geometry_candidate.handbase_to_tcp_m
                        ),
                        nominal_hand_to_plug=nominal_hand_to_plug,
                        fixed_camera_eye=tuple(rgbd.camera.eye_m),
                    )
                )
            else:
                display_poses, predicted_direction_deg = (
                    plan_two_fixed_camera_display_feasibility_poses(
                        q0_arm,
                        solve_arm=solve_fixed_q7_tcp_pose,
                        handbase_to_tcp_m=(
                            pick.geometry_candidate.handbase_to_tcp_m
                        ),
                        nominal_hand_to_plug=nominal_hand_to_plug,
                        fixed_camera_eye=tuple(rgbd.camera.eye_m),
                    )
                )

            output_root.mkdir(parents=True, exist_ok=True)
            display_trace_buffer = DisplayMotionRingBuffer(capacity=240)
            display_path_quality_records = []
            posthoc_sidecar_records = []

            def _matrix_from_wxyz(orientation_wxyz):
                value = np.asarray(orientation_wxyz, dtype=np.float64)
                value = value / np.linalg.norm(value)
                return Rotation.from_quat(
                    [value[1], value[2], value[3], value[0]]
                ).as_matrix()

            def posthoc_contact_audit():
                headers, _, _ = (
                    get_physx_simulation_interface().get_full_contact_report()
                )
                counts = {
                    "plug_table": 0,
                    "robot_table": 0,
                    "robot_fixture": 0,
                    "robot_fixed_endpoint": 0,
                    "robot_loose_plug": 0,
                    "finger_loose_plug": 0,
                }
                for header in headers:
                    paths = (
                        str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                        str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                        str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                        str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                    )
                    count = int(header.num_contact_data)
                    if _is_plug_table_contact(
                        paths,
                        tabletop.asset.loose_plug_prim_path,
                        tabletop.table.prim_path,
                    ):
                        counts["plug_table"] += count
                    category = _classify_robot_external_contact(
                        paths,
                        pick.scene.robot_root_prim_path,
                        tabletop.table.prim_path,
                        tabletop.fixed_endpoint.fixture_prim_path,
                        tabletop.asset.fixed_receptacle_prim_path,
                        tabletop.asset.loose_plug_prim_path,
                    )
                    if category is not None:
                        counts[f"robot_{category}"] += count
                    if _is_finger_plug_contact(
                        paths,
                        pick.scene.robot_root_prim_path,
                        tabletop.asset.loose_plug_prim_path,
                    ):
                        counts["finger_loose_plug"] += count
                return counts

            def append_posthoc_sidecar_record(
                *,
                phase,
                arm_q_actual,
                hand_q_actual,
                velocities,
                torque,
                wrist,
            ):
                tcp_transform = np.asarray(
                    iiwa14_grasp_tcp_transform(tuple(arm_q_actual))
                )
                hand_transform = tcp_transform @ tcp_from_handbase
                plug_position, plug_orientation = body.get_world_pose()
                nut_position, nut_orientation = nut.get_world_pose()
                plug_matrix = np.eye(4, dtype=np.float64)
                plug_matrix[:3, 3] = np.asarray(plug_position)
                plug_matrix[:3, :3] = _matrix_from_wxyz(
                    plug_orientation
                )
                nut_matrix = np.eye(4, dtype=np.float64)
                nut_matrix[:3, 3] = np.asarray(nut_position)
                nut_matrix[:3, :3] = _matrix_from_wxyz(nut_orientation)
                t_hand_plug = np.linalg.inv(hand_transform) @ plug_matrix
                t_hand_nut = np.linalg.inv(hand_transform) @ nut_matrix
                posthoc_sidecar_records.append(
                    {
                        "role": "posthoc_truth_sidecar_record",
                        "scope": "posthoc_evaluation_only",
                        "formal_estimator_input": False,
                        "control_authorized": False,
                        "global_step": runtime_state["global_step"],
                        "phase": phase,
                        "arm_q_actual_rad": [
                            float(value) for value in arm_q_actual
                        ],
                        "hand_q_actual_rad": [
                            float(value) for value in hand_q_actual
                        ],
                        "joint_qd_rad_s": [
                            float(value) for value in velocities
                        ],
                        "finger_torque_proxy_nm": [
                            float(value) for value in torque
                        ],
                        "wrist_canonical": [
                            float(value) for value in wrist
                        ],
                        "plug_position_m": [
                            float(value) for value in plug_position
                        ],
                        "plug_orientation_wxyz": [
                            float(value) for value in plug_orientation
                        ],
                        "plug_linear_velocity_m_s": [
                            float(value) for value in body.get_linear_velocity()
                        ],
                        "plug_angular_velocity_rad_s": [
                            float(value) for value in body.get_angular_velocity()
                        ],
                        "nut_position_m": [
                            float(value) for value in nut_position
                        ],
                        "nut_orientation_wxyz": [
                            float(value) for value in nut_orientation
                        ],
                        "nut_linear_velocity_m_s": [
                            float(value) for value in nut.get_linear_velocity()
                        ],
                        "nut_angular_velocity_rad_s": [
                            float(value) for value in nut.get_angular_velocity()
                        ],
                        "t_hand_plug_4x4": t_hand_plug.tolist(),
                        "t_hand_nut_4x4": t_hand_nut.tolist(),
                        "contact_audit": posthoc_contact_audit(),
                    }
                )

            frozen_hand_target = np.asarray(
                snapshot["frozen_command"]["hand_q_target_rad"],
                dtype=np.float64,
            )
            frozen_arm_target = np.asarray(
                snapshot["frozen_command"]["arm_q_target_rad"],
                dtype=np.float64,
            )
            commanded_arm_target = frozen_arm_target.copy()
            wrist_reference = np.asarray(
                snapshot["frozen_command"]["wrist_ft_snapshot_reference"],
                dtype=np.float64,
            )
            previous_display_wrist = wrist_reference.copy()
            previous_raw_wrist = wrist_reference.copy()
            peak_display_reference_force_n = 0.0
            peak_display_reference_moment_score_nm = 0.0
            peak_display_force_rate_n = 0.0
            peak_display_moment_rate_nm = 0.0
            hold_steps = 120
            inspection_views = []
            motion_records = []

            display_posthoc_audit_writer = append_posthoc_sidecar_record
            camera_bindings = {
                "Camera": Camera,
                "Gf": Gf,
                "Usd": Usd,
                "UsdGeom": UsdGeom,
                "UsdLux": UsdLux,
                "Image": Image,
                "rep": rep,
            }

            def display_sensor_gates(desired_arm, positions, velocities, torque):
                return evaluate_display_sensor_gates(
                    desired_arm_q=desired_arm,
                    actual_q=np.asarray(
                        positions[:7], dtype=np.float64
                    ).ravel(),
                    velocities=velocities,
                    torque=torque,
                    joint_limits=arm_joint_limits,
                    joint_limit_margin_rad=0.010,
                    max_abs_torque_nm=float(
                        pick.sensing.maximum_absolute_torque_delta_nm
                    ),
                    max_joint_speed_rad_s=1.0,
                    max_arm_tracking_error_rad=0.030,
                )

            for pose in display_poses:
                view_id = pose["view_id"]
                view_dir = output_root / "formal_views" / view_id
                if view_dir.exists():
                    raise RuntimeError(f"{view_id} output already exists")
                target_arm = np.asarray(
                    pose["arm_q_rad"], dtype=np.float64
                )
                q_before_move = np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                ).ravel()[:7]
                current_tcp = np.asarray(
                    iiwa14_grasp_tcp_transform(tuple(q_before_move))
                )
                planned_tcp = np.asarray(pose["tcp_target"])
                tcp_distance = float(
                    np.linalg.norm(
                        planned_tcp[:3, 3] - current_tcp[:3, 3]
                    )
                )
                # Cartesian TCP waypoints prevent the joint-space path from
                # dipping toward the table.  The commanded speed is kept in
                # the same order as the validated slow lift (<= 0.02 m/s).
                planned_waypoints = plan_cartesian_tcp_waypoints(
                    q_before_move,
                    planned_tcp,
                    solve_arm=solve_fixed_q7_tcp_pose,
                    maximum_step_m=0.005,
                )
                move_duration_s = max(4.0, tcp_distance / 0.010)
                steps_per_waypoint = max(
                    1,
                    round(
                        move_duration_s
                        * rate_hz
                        / float(len(planned_waypoints))
                    ),
                )
                table_center = np.asarray(
                    tabletop.table.center_m, dtype=np.float64
                )
                table_size = np.asarray(
                    tabletop.table.size_m, dtype=np.float64
                )
                path_quality = evaluate_waypoint_path_quality(
                    planned_waypoints,
                    forward_kinematics=iiwa14_grasp_tcp_transform,
                    physics_rate_hz=rate_hz,
                    steps_per_waypoint=steps_per_waypoint,
                    start_q=q_before_move,
                    joint_limits=arm_joint_limits,
                    joint_limit_margin_rad=0.010,
                    table_top_z_m=float(
                        table_center[2] + 0.5 * table_size[2]
                    ),
                    fixture_center_m=(
                        tabletop.fixed_endpoint.fixture_center_m
                    ),
                    fixture_half_extent_m=tuple(
                        0.5 * value
                        for value in tabletop.fixed_endpoint.fixture_size_m
                    ),
                )
                display_path_quality_records.append(
                    {
                        "view_id": view_id,
                        "tcp_distance_m": tcp_distance,
                        "steps_per_waypoint": steps_per_waypoint,
                        **path_quality,
                    }
                )
                if path_quality["reject"]:
                    raise RuntimeError(
                        f"{view_id} path quality rejected before motion: "
                        + "; ".join(path_quality["reasons"])
                    )
                motion_step_index = 0
                segment_start_arm = commanded_arm_target.copy()
                for waypoint_index, waypoint_arm in enumerate(
                    planned_waypoints, start=1
                ):
                    for sub_step in range(steps_per_waypoint):
                        desired_arm = np.asarray(
                            interpolate_arm(
                                tuple(segment_start_arm),
                                tuple(waypoint_arm),
                                float(sub_step + 1)
                                / float(steps_per_waypoint),
                            ),
                            dtype=np.float64,
                        )
                        positions, velocities = observe_and_step(
                            desired_arm, frozen_hand_target
                        )
                        torque = sample_finger_torque_proxy()
                        wrist_state = get_latest_wrist_state()
                        if wrist_state.get("error") is not None:
                            raise RuntimeError(
                                f"{view_id} wrist error during display motion"
                            )
                        wrist = np.asarray(
                            wrist_state["canonical"], dtype=np.float64
                        )
                        actual_arm_q = np.asarray(
                            positions[:7], dtype=np.float64
                        )
                        actual_hand_q = np.asarray(
                            positions[7:], dtype=np.float64
                        )
                        desired_tcp = np.asarray(
                            iiwa14_grasp_tcp_transform(tuple(desired_arm))
                        )
                        actual_tcp = np.asarray(
                            iiwa14_grasp_tcp_transform(tuple(actual_arm_q))
                        )
                        evidence = evaluate_display_wrist_evidence(
                            current_wrench=wrist,
                            reference_wrench=wrist_reference,
                            previous_raw_wrench=previous_raw_wrist,
                            ema_wrench=previous_display_wrist,
                        )
                        previous_raw_wrist = wrist.copy()
                        previous_display_wrist = np.asarray(
                            evidence["ema_wrench"], dtype=np.float64
                        )
                        peak_display_reference_force_n = max(
                            peak_display_reference_force_n,
                            evidence["force_increment_n"],
                        )
                        peak_display_reference_moment_score_nm = max(
                            peak_display_reference_moment_score_nm,
                            evidence["moment_evidence"]["gate_score_nm"],
                        )
                        peak_display_force_rate_n = max(
                            peak_display_force_rate_n,
                            evidence.get(
                                "adjacent_raw_force_delta_n", 0.0
                            ),
                        )
                        peak_display_moment_rate_nm = max(
                            peak_display_moment_rate_nm,
                            evidence.get(
                                "adjacent_raw_moment_delta_nm", 0.0
                            ),
                        )
                        sensor_gates = display_sensor_gates(
                            desired_arm, positions, velocities, torque
                        )
                        gate_failed = bool(
                            not sensor_gates["ok"]
                            or evidence["formal_gate_triggered"]
                            or evidence["ema_candidate_triggered"]
                        )
                        trace_record = {
                            "role": "display_motion_trace_record",
                            "global_step": runtime_state["global_step"],
                            "view_id": view_id,
                            "phase": "display_move",
                            "waypoint": waypoint_index,
                            "substep": sub_step,
                            "desired_q": desired_arm.tolist(),
                            "actual_q": positions.tolist(),
                            "qd": velocities.tolist(),
                            "arm_tracking_error_rad": float(
                                np.max(np.abs(actual_arm_q - desired_arm))
                            ),
                            "desired_tcp": desired_tcp.tolist(),
                            "actual_tcp": actual_tcp.tolist(),
                            "raw_wrench": wrist_state.get(
                                "raw_wrench", wrist.tolist()
                            ),
                            "canonical_wrench": wrist.tolist(),
                            "snapshot_reference_force_delta_n": (
                                evidence["force_increment_n"]
                            ),
                            "moment_evidence": evidence["moment_evidence"],
                            "adjacent_raw_force_delta_n": evidence.get(
                                "adjacent_raw_force_delta_n"
                            ),
                            "adjacent_raw_moment_delta_nm": evidence.get(
                                "adjacent_raw_moment_delta_nm"
                            ),
                            "ema_force_residual_n": evidence[
                                "ema_force_residual_n"
                            ],
                            "ema_moment_residual_nm": evidence[
                                "ema_moment_residual_nm"
                            ],
                            "triggered_gates": evidence[
                                "triggered_gates"
                            ],
                            "gate_failed": gate_failed,
                            "finger_torque_proxy_nm": torque.tolist(),
                            "finger_q_actual": actual_hand_q.tolist(),
                            "sensor_gates": sensor_gates,
                        }
                        display_trace_buffer.append(trace_record)
                        if gate_failed:
                            raise RuntimeError(
                                f"{view_id} display motion safety fail-closed "
                                f"at waypoint {waypoint_index}, step "
                                f"{motion_step_index}: gates="
                                f"{evidence['triggered_gates']}, "
                                f"sensor_gates={sensor_gates['reasons']}, "
                                f"tracking_error_rad="
                                f"{sensor_gates['arm_tracking_error_rad']:.6f}, "
                                f"force_increment_n="
                                f"{evidence['force_increment_n']:.6f}, "
                                f"moment_gate_score_nm="
                                f"{evidence['moment_evidence']['gate_score_nm']:.6f}, "
                                f"ema_force_residual_n="
                                f"{evidence['ema_force_residual_n']:.6f}, "
                                f"ema_moment_residual_nm="
                                f"{evidence['ema_moment_residual_nm']:.6f}"
                            )
                        motion_step_index += 1
                    segment_start_arm = waypoint_arm.copy()
                    motion_records.append(
                        {
                            "view_id": view_id,
                            "phase": "display_move",
                            "waypoint": waypoint_index,
                            "waypoint_count": len(planned_waypoints),
                            "global_step": runtime_state["global_step"],
                            "arm_q_actual_rad": np.asarray(
                                positions[:7], dtype=np.float64
                            ).tolist(),
                            "joint_qd_rad_s": np.asarray(
                                velocities, dtype=np.float64
                            ).tolist(),
                            "finger_torque_proxy_nm": torque.tolist(),
                            "wrist_canonical": wrist.tolist(),
                            "wrist_reference_force_delta_n": (
                                evidence["force_increment_n"]
                            ),
                            "wrist_reference_moment_score_nm": (
                                evidence["moment_evidence"]["gate_score_nm"]
                            ),
                            "wrist_ema_force_residual_n": evidence[
                                "ema_force_residual_n"
                            ],
                            "wrist_ema_moment_residual_nm": evidence[
                                "ema_moment_residual_nm"
                            ],
                        }
                    )

                for hold_index in range(hold_steps):
                    positions, velocities = observe_and_step(
                        target_arm, frozen_hand_target
                    )
                    torque = sample_finger_torque_proxy()
                    wrist_state = get_latest_wrist_state()
                    wrist = np.asarray(
                        wrist_state["canonical"], dtype=np.float64
                    )
                    actual_arm_q = np.asarray(
                        positions[:7], dtype=np.float64
                    )
                    actual_hand_q = np.asarray(
                        positions[7:], dtype=np.float64
                    )
                    actual_tcp = np.asarray(
                        iiwa14_grasp_tcp_transform(tuple(actual_arm_q))
                    )
                    evidence = evaluate_display_wrist_evidence(
                        current_wrench=wrist,
                        reference_wrench=wrist_reference,
                        previous_raw_wrench=previous_raw_wrist,
                        ema_wrench=previous_display_wrist,
                    )
                    previous_raw_wrist = wrist.copy()
                    previous_display_wrist = np.asarray(
                        evidence["ema_wrench"], dtype=np.float64
                    )
                    peak_display_reference_force_n = max(
                        peak_display_reference_force_n,
                        evidence["force_increment_n"],
                    )
                    peak_display_reference_moment_score_nm = max(
                        peak_display_reference_moment_score_nm,
                        evidence["moment_evidence"]["gate_score_nm"],
                    )
                    peak_display_force_rate_n = max(
                        peak_display_force_rate_n,
                        evidence.get(
                            "adjacent_raw_force_delta_n", 0.0
                        ),
                    )
                    peak_display_moment_rate_nm = max(
                        peak_display_moment_rate_nm,
                        evidence.get(
                            "adjacent_raw_moment_delta_nm", 0.0
                        ),
                    )
                    sensor_gates = display_sensor_gates(
                        target_arm, positions, velocities, torque
                    )
                    gate_failed = bool(
                        not sensor_gates["ok"]
                        or evidence["formal_gate_triggered"]
                        or evidence["ema_candidate_triggered"]
                    )
                    trace_record = {
                        "role": "display_motion_trace_record",
                        "global_step": runtime_state["global_step"],
                        "view_id": view_id,
                        "phase": "display_hold",
                        "hold_step": hold_index,
                        "desired_q": target_arm.tolist(),
                        "actual_q": positions.tolist(),
                        "qd": velocities.tolist(),
                        "arm_tracking_error_rad": float(
                            np.max(np.abs(actual_arm_q - target_arm))
                        ),
                        "actual_tcp": actual_tcp.tolist(),
                        "raw_wrench": wrist_state.get(
                            "raw_wrench", wrist.tolist()
                        ),
                        "canonical_wrench": wrist.tolist(),
                        "snapshot_reference_force_delta_n": (
                            evidence["force_increment_n"]
                        ),
                        "moment_evidence": evidence["moment_evidence"],
                        "adjacent_raw_force_delta_n": evidence.get(
                            "adjacent_raw_force_delta_n"
                        ),
                        "adjacent_raw_moment_delta_nm": evidence.get(
                            "adjacent_raw_moment_delta_nm"
                        ),
                        "ema_force_residual_n": evidence[
                            "ema_force_residual_n"
                        ],
                        "ema_moment_residual_nm": evidence[
                            "ema_moment_residual_nm"
                        ],
                        "triggered_gates": evidence[
                            "triggered_gates"
                        ],
                        "gate_failed": gate_failed,
                        "finger_torque_proxy_nm": torque.tolist(),
                        "finger_q_actual": actual_hand_q.tolist(),
                        "sensor_gates": sensor_gates,
                    }
                    display_trace_buffer.append(trace_record)
                    if gate_failed:
                        raise RuntimeError(
                            f"{view_id} display hold safety fail-closed "
                            f"at hold step {hold_index}: gates="
                            f"{evidence['triggered_gates']}, "
                            f"sensor_gates={sensor_gates['reasons']}, "
                            f"tracking_error_rad="
                            f"{sensor_gates['arm_tracking_error_rad']:.6f}, "
                            f"force_increment_n="
                            f"{evidence['force_increment_n']:.6f}, "
                            f"moment_gate_score_nm="
                            f"{evidence['moment_evidence']['gate_score_nm']:.6f}, "
                            f"ema_force_residual_n="
                            f"{evidence['ema_force_residual_n']:.6f}, "
                            f"ema_moment_residual_nm="
                            f"{evidence['ema_moment_residual_nm']:.6f}"
                        )

                commanded_arm_target = target_arm.copy()
                capture = capture_d38999_rgbd_raw_formal(
                    bindings=camera_bindings,
                    simulation_app=simulation_app,
                    world=world,
                    stage=stage,
                    tabletop=tabletop,
                    rgbd=rgbd,
                    output_dir=view_dir,
                    camera_clipping_range_m=(0.1, 10.0),
                )
                if capture.passed is not True:
                    raise RuntimeError(f"{view_id} raw RGB-D capture failed")
                actual_arm_q = np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                ).ravel()
                tcp_transform = np.asarray(
                    iiwa14_grasp_tcp_transform(
                        tuple(actual_arm_q[:7])
                    ),
                    dtype=np.float64,
                )
                camera_prim = stage.GetPrimAtPath(rgbd.camera.prim_path)
                if camera_prim is None or not camera_prim.IsValid():
                    raise RuntimeError("fixed camera prim is missing")
                camera_matrix = np.asarray(
                    UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    ),
                    dtype=np.float64,
                )
                (view_dir / "fk.json").write_text(
                    json.dumps(
                        {
                            "arm_q_actual_rad": actual_arm_q[:7].tolist(),
                            "tcp_pose_4x4": tcp_transform.tolist(),
                            "hand_q_actual_rad": actual_arm_q[7:].tolist(),
                            "fk_source": "iiwa14_grasp_tcp_transform",
                        },
                        allow_nan=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (view_dir / "camera.json").write_text(
                    json.dumps(
                        {
                            "prim_path": rgbd.camera.prim_path,
                            "frame_id": rgbd.camera.frame_id,
                            "eye_m": rgbd.camera.eye_m,
                            "target_m": rgbd.camera.target_m,
                            "world_4x4": camera_matrix.tolist(),
                            "intrinsics": capture.metrics[
                                "camera"
                            ]["intrinsics"],
                        },
                        allow_nan=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (view_dir / "sensor_state.json").write_text(
                    json.dumps(
                        {
                            "global_step": runtime_state["global_step"],
                            "timestamp_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "finger_torque_proxy_nm": sample_finger_torque_proxy().tolist(),
                            "wrist_canonical": np.asarray(
                                get_latest_wrist_state()["canonical"],
                                dtype=np.float64,
                            ).tolist(),
                        },
                        allow_nan=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                inspection_views.append(
                    {
                        "view_id": view_id,
                        "output_directory": str(view_dir),
                        "rgb_filename": rgbd.output.rgb_filename,
                        "depth_filename": rgbd.output.depth_numpy_filename,
                        "camera_file": "camera.json",
                        "fk_file": "fk.json",
                        "sensor_state_file": "sensor_state.json",
                        "capture_metrics": capture.metrics,
                        "planned_arm_q_rad": pose["arm_q_rad"],
                        "planned_tcp_delta_xyz_m": pose.get(
                            "tcp_delta_xyz_m"
                        ),
                        "planned_plug_center_candidate_m": pose.get(
                            "plug_center_candidate_m"
                        ),
                        "max_abs_dq_rad": pose["max_abs_dq_rad"],
                        "predicted_nominal_plug_center_m": pose[
                            "nominal_plug_center_m"
                        ],
                        "predicted_camera_to_plug_direction": pose[
                            "camera_to_plug_direction"
                        ],
                    }
                )

            if display_posthoc_audit_writer is not None:
                final_actual_q = np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                ).ravel()
                final_wrist_state = get_latest_wrist_state()
                display_posthoc_audit_writer(
                    phase="final_after_control_termination",
                    arm_q_actual=final_actual_q[:7],
                    hand_q_actual=final_actual_q[7:],
                    velocities=np.asarray(
                        robot.get_joint_velocities(), dtype=np.float64
                    ).ravel(),
                    torque=sample_finger_torque_proxy(),
                    wrist=np.asarray(
                        final_wrist_state["canonical"], dtype=np.float64
                    ),
                )
            (output_root / "formal_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "kcg_d38999_fc_pgdf_01a_v1",
                        "role": "formal_raw_observation",
                        "formal_estimator_input": True,
                        "estimator_run": False,
                        "control_authorized": False,
                        "object_truth_present": False,
                        "contact_report_present": False,
                        "views": inspection_views,
                    },
                    allow_nan=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (output_root / "posthoc_truth_sidecar.json").write_text(
                json.dumps(
                    {
                        "role": "posthoc_truth_sidecar",
                        "scope": "posthoc_evaluation_only",
                        "formal_estimator_input": False,
                        "control_authorized": False,
                        "object_pose_writes_total": pose_write_counter.value,
                        "object_pose_writes_after_restore": (
                            pose_write_counter.value - 2
                        ),
                        "final_plug_position_m": np.asarray(
                            body.get_world_pose()[0], dtype=np.float64
                        ).tolist(),
                        "final_plug_orientation_wxyz": np.asarray(
                            body.get_world_pose()[1], dtype=np.float64
                        ).tolist(),
                        "final_nut_position_m": np.asarray(
                            nut.get_world_pose()[0], dtype=np.float64
                        ).tolist(),
                        "final_nut_orientation_wxyz": np.asarray(
                            nut.get_world_pose()[1], dtype=np.float64
                        ).tolist(),
                    },
                    allow_nan=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            inspection_metrics = {
                "status": "FC_PGDF_RAW_CAPTURE_COMPLETE",
                "views": inspection_views,
                "motion_records": motion_records,
                "predicted_direction_difference_deg": (
                    predicted_direction_deg
                ),
                "control_authorized": False,
                "formal_estimator_input": True,
                "object_pose_writes_after_restore": 0,
                "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
                "display_safety_gate_semantics": (
                    "display_ema_deviation_candidate_v1"
                ),
                "peak_wrist_reference_force_delta_n": (
                    peak_display_reference_force_n
                ),
                "peak_wrist_reference_moment_score_nm": (
                    peak_display_reference_moment_score_nm
                ),
                "peak_adjacent_raw_force_delta_n": (
                    peak_display_force_rate_n
                ),
                "peak_adjacent_raw_moment_delta_nm": (
                    peak_display_moment_rate_nm
                ),
            }
            metrics["fc_pgdf_01a_raw_capture"] = inspection_metrics
            atomic_write_json_lines(
                output_root / "motion_trace.jsonl",
                display_trace_buffer.records(),
            )
            atomic_write_json_lines(
                output_root / "posthoc_truth_sidecar.jsonl",
                posthoc_sidecar_records,
            )
        if arguments.visual_chain:
            # Formal visual estimation chain, in-process and CPU-only:
            #   S1 palm view -> T_HP (C2 retained, never averaged)
            #   S2 wrist preinsert views -> T_RP
            #   S3 assembly transform synthesis (diagnostic only)
            # No object truth, no contact report, no control authorization.
            import cv2
            import math as _math

            from kcg_connector.d38999_cad_registration import (
                CameraModel,
                fixed_camera_model,
            )
            from kcg_connector.d38999_inhand_multiview import (
                matrix_pose,
                pose_matrix,
            )
            from kcg_connector.postgrasp_shadow_estimator import (
                FormalView,
                estimate_postgrasp_T_HP,
                estimate_preinsert_T_RP,
            )

            def load_chain_view(view_dir, view_id, group):
                rgb = cv2.cvtColor(
                    cv2.imread(str(view_dir / "rgb.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                )
                depth = np.load(view_dir / "depth_m.npy").astype(np.float32)
                camera_record = json.loads(
                    (view_dir / "camera.json").read_text(encoding="utf-8")
                )
                model = fixed_camera_model(
                    eye=tuple(camera_record["eye_m"]),
                    target=tuple(camera_record["target_m"]),
                    resolution=(1280, 720),
                )
                intrinsics = np.asarray(camera_record["intrinsics"])
                camera = CameraModel(
                    1280,
                    720,
                    float(intrinsics[0, 0]),
                    float(intrinsics[1, 1]),
                    float(intrinsics[0, 2]),
                    float(intrinsics[1, 2]),
                    tuple(model.position_world),
                    tuple(model.world_to_camera),
                )
                fk = json.loads(
                    (view_dir / "fk.json").read_text(encoding="utf-8")
                )
                return FormalView(
                    view_id=view_id,
                    timestamp_utc="2026-08-15T00:00:00Z",
                    rgb=rgb,
                    depth=depth,
                    camera=camera,
                    T_WH=np.asarray(fk["T_WH_4x4"]),
                    T_WC=np.asarray(fk["T_WC_4x4"]),
                    group=group,
                    extrinsic_source="T_HC_calibrated",
                )

            initial_state = np.zeros(12, dtype=np.float64)
            initial_state[:6] = np.array(
                [0.0, 0.0, 0.4485, _math.pi, 0.0, 0.0]
            )
            palm_view_id = (
                "PALM_H0_K0"
                if (output_root / "formal_views" / "PALM_H0_K0").is_dir()
                else "PALM_H0"
            )
            palm_view = load_chain_view(
                output_root / "formal_views" / palm_view_id,
                palm_view_id,
                "postgrasp_inhand_views",
            )
            wrist_h0_view = None
            wrist_h0_dir = output_root / "formal_views" / "WRIST_H0"
            if (wrist_h0_dir / "rgb.png").is_file():
                wrist_h0_view = load_chain_view(
                    wrist_h0_dir,
                    "WRIST_H0",
                    "postgrasp_second_inhand_camera_views",
                )
            t_hp_views = [palm_view]
            if wrist_h0_view is not None:
                t_hp_views.append(wrist_h0_view)
            from kcg_connector.d38999_cad_registration import (
                proxy_cad_points,
                shell25j_plug_cad_profile,
            )
            from kcg_connector.hand_occluder_cad import (
                build_hand_occluder_cad,
            )

            legacy_plug, legacy_receptacle = proxy_cad_points()
            shell_profile = shell25j_plug_cad_profile(
                feature_set="shell_plus_socket"
            )
            hand_occluder = build_hand_occluder_cad(
                snapshot["robot_state"]["q_rad"][7:15],
                repository / "artifacts/kcg_connector/urdf/handarm.urdf",
                repository / "src/iiwa_description/meshes/hand",
            )
            t_hp_result = estimate_postgrasp_T_HP(
                t_hp_views,
                initial_state,
                plug_cad=legacy_plug,
                receptacle_cad=legacy_receptacle,
                plug_occluder_cad=shell_profile.plug_occluders,
                hand_occluder_cad=hand_occluder,
                occlusion_policy="baseline",
                edge_policy="depth_gated",
                optimizer_variant="multistart_physical_jacobian",
                multistart_count=17,
            )
            hp_estimate = np.asarray(
                t_hp_result["c2"]["hypotheses"][0][
                    "T_hand_plug_xyz_rpy"
                ],
                dtype=np.float64,
            )
            hp_error_vs_nominal = matrix_pose(
                np.linalg.inv(pose_matrix(initial_state[:6]))
                @ pose_matrix(hp_estimate)
            )
            hp_matrix = pose_matrix(hp_estimate)
            hp_axis = hp_matrix[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
            hp_axis = hp_axis / np.linalg.norm(hp_axis)
            hp_tilt_deg = float(
                _math.degrees(
                    _math.acos(
                        max(-1.0, min(1.0, abs(float(hp_axis[2]))))
                    )
                )
            )
            wrist_views = []
            for record in design["views"]:
                if record["view_id"] not in design["selected_view_ids"]:
                    continue
                view_dir = output_root / "formal_views" / record["view_id"]
                if not (view_dir / "rgb.png").is_file():
                    continue
                wrist_views.append(
                    load_chain_view(
                        view_dir, record["view_id"], "final_preinsert_views"
                    )
                )
            t_rp_result = None
            t_rp_initial = np.zeros(6, dtype=np.float64)
            if wrist_views:
                t_wp_nominal = wrist_views[0].T_WH @ pose_matrix(
                    initial_state[:6]
                )
                t_rp_initial = matrix_pose(
                    np.linalg.inv(t_wp_nominal) @ np.asarray(receptacle)
                )
                chain_initial = np.concatenate((hp_estimate, t_rp_initial))
                t_rp_result = estimate_preinsert_T_RP(
                    wrist_views, hp_estimate, chain_initial
                )
            chain_report = {
                "schema_version": "kcg_d38999_visual_chain_v1",
                "role": "formal_visual_chain_report",
                "control_authorized": False,
                "object_truth_used": False,
                "contact_report_used": False,
                "stage_t_hp": {
                    "view_id": palm_view_id,
                    "t_hp_view_ids": [v.view_id for v in t_hp_views],
                    "optimizer_converged": t_hp_result["optimizer_converged"],
                    "pose_valid": t_hp_result["pose_valid"],
                    "pose_valid_reasons": t_hp_result["pose_valid_reasons"],
                    "residual_rms": [
                        h["residual_rms"]
                        for h in t_hp_result["c2"]["hypotheses"]
                    ],
                    "support_gate_failed": t_hp_result[
                        "plug_support_gate_failed"
                    ],
                    "c2_resolution": t_hp_result["c2"]["resolution"],
                    "c2_hypotheses": [
                        {
                            "id": h["id"],
                            "T_hand_plug_xyz_rpy": h[
                                "T_hand_plug_xyz_rpy"
                            ],
                        }
                        for h in t_hp_result["c2"]["hypotheses"]
                    ],
                    "axis_tilt_from_hand_z_deg": hp_tilt_deg,
                    "t_hp_error_vs_nominal_xyz_rpy": hp_error_vs_nominal.tolist(),
                    "nominal_t_hp_xyz_rpy": initial_state[:6].tolist(),
                },
                "stage_t_rp": None
                if t_rp_result is None
                else {
                    "status": t_rp_result["status"],
                    "T_receptacle_plug_xyz_rpy": t_rp_result[
                        "T_receptacle_plug_xyz_rpy"
                    ],
                    "reject_reason": t_rp_result["reject_reason"],
                    "view_ids": [v.view_id for v in wrist_views],
                },
                "stage_insertion": {
                    "status": "NOT_RUN_IN_VISUAL_CHAIN_V1"
                },
                "stage_screw": {
                    "status": "NOT_RUN_IN_VISUAL_CHAIN_V1"
                },
                "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
            }
            chain_end_plug_pose = body.get_world_pose()
            (output_root / "posthoc_truth_sidecar_end.json").write_text(
                json.dumps(
                    {
                        "schema_version": "kcg_d38999_capture_time_posthoc_truth_v1",
                        "role": "posthoc_truth_sidecar",
                        "truth_scope": "capture_time_posthoc_evaluation_only",
                        "formal_estimator_input": False,
                        "control_authorized": False,
                        "object_truth_present": True,
                        "contact_report_present": False,
                        "plug_position_m": [
                            float(v) for v in chain_end_plug_pose[0]
                        ],
                        "plug_orientation_wxyz": [
                            float(v) for v in chain_end_plug_pose[1]
                        ],
                        "recorded_after": "wrist_receptacle_views",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (output_root / "visual_chain_report.json").write_text(
                json.dumps(
                    chain_report, allow_nan=False, ensure_ascii=False, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
            metrics["visual_chain"] = {
                "status": "REPORTED",
                "control_authorized": False,
                "formal_estimator_input": False,
            }
        metrics["control_reads_object_truth"] = False
        metrics["control_reads_contact_report"] = False
        metrics["posthoc_audit_reads_object_truth"] = bool(
            display_posthoc_audit_writer is not None
        )
        metrics["posthoc_audit_reads_contact_report"] = bool(
            display_posthoc_audit_writer is not None
        )
        metrics["posthoc_audit_after_control_termination"] = bool(
            display_posthoc_audit_writer is not None
        )
        metrics["posthoc_truth_used_for_consistency_only"] = True
        passed = result.get("status") == "FRESH_REPLAY_RESTORE_VERIFIED"
        metrics["passed"] = passed
        process_exit_code = 0 if passed else 2
        metrics["process_exit_code"] = process_exit_code
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "fresh_replay_report.json").write_text(
            json.dumps(metrics, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(
            FRESH_REPLAY_RESULT_MARKER + ("PASSED" if passed else "FAILED"),
            flush=True,
        )
    except BaseException as exception:
        process_exit_code = 1
        metrics["passed"] = False
        metrics["error"] = f"{type(exception).__name__}: {exception}"
        metrics["process_exit_code"] = process_exit_code
        import traceback

        traceback.print_exc()
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            trace_records = (
                display_trace_buffer.records()
                if display_trace_buffer is not None
                else []
            )
            try:
                if (
                    display_posthoc_audit_writer is not None
                    and "robot" in locals()
                    and "body" in locals()
                    and "nut" in locals()
                ):
                    final_actual_q = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    ).ravel()
                    final_wrist_state = get_latest_wrist_state()
                    display_posthoc_audit_writer(
                        phase="final_after_control_termination",
                        arm_q_actual=final_actual_q[:7],
                        hand_q_actual=final_actual_q[7:],
                        velocities=np.asarray(
                            robot.get_joint_velocities(),
                            dtype=np.float64,
                        ).ravel(),
                        torque=sample_finger_torque_proxy(),
                        wrist=np.asarray(
                            final_wrist_state["canonical"],
                            dtype=np.float64,
                        ),
                    )
            except Exception as audit_exception:
                metrics["posthoc_audit_write_error"] = (
                    f"{type(audit_exception).__name__}: {audit_exception}"
                )
            atomic_write_json_lines(
                output_root / "motion_trace.jsonl", trace_records
            )
            atomic_write_json_lines(
                output_root / "posthoc_truth_sidecar.jsonl",
                posthoc_sidecar_records,
            )
            failure_report = build_failure_report(
                error=f"{type(exception).__name__}: {exception}",
                status="DISPLAY_SAFETY_FAIL_CLOSED",
                trace_records=trace_records,
                path_quality_records=display_path_quality_records,
            )
            atomic_write_json(
                output_root / "failure_report.json", failure_report
            )
            metrics["failure_evidence_written"] = True
        except Exception as write_exception:
            metrics["failure_evidence_written"] = False
            metrics["failure_evidence_write_error"] = (
                f"{type(write_exception).__name__}: {write_exception}"
            )
        print(json.dumps(metrics, allow_nan=False, ensure_ascii=False), flush=True)
        print(FRESH_REPLAY_RESULT_MARKER + "FAILED", flush=True)
    finally:
        simulation_app.close(exit_code=process_exit_code)
    return process_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
